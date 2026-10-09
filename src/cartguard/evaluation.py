"""Metrics and validation-only operating-point selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def threshold_from_validation(target, probabilities):
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(target, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(target, predictions, task, threshold=0.5):
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite")
    if task == "rating":
        return {
            "mae": float(mean_absolute_error(target, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
            "r2": float(r2_score(target, predictions)),
        }
    labels = predictions >= threshold
    return {
        "average_precision": float(average_precision_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, predictions)),
        "f1": float(f1_score(target, labels, zero_division=0)),
        "precision": float(precision_score(target, labels, zero_division=0)),
        "recall": float(recall_score(target, labels, zero_division=0)),
        "accuracy": float(accuracy_score(target, labels)),
        "log_loss": float(log_loss(target, predictions, labels=[0, 1])),
        "brier": float(brier_score_loss(target, predictions)),
    }


def grouped_bootstrap(target, predicted, groups, task, repeats=500):
    rng = np.random.default_rng(91)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(repeats):
        sample = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        if task == "risk" and len(np.unique(target[sample])) < 2:
            continue
        key = "mae" if task == "rating" else "average_precision"
        values.append(metrics(target[sample], predicted[sample], task)[key])
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]
