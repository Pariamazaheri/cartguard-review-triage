"""Product-held-out missingness and imbalance benchmark."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .data import NUMERIC, download, load, mask_numeric, split
from .evaluation import metrics, threshold_from_validation
from .models import make_model
from .visualization import render


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(root):
    root = Path(root)
    report, models = root / "reports", root / "models"
    report.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    frame, audit = load(download(root))
    parts = split(frame)
    write_json(report / "data_audit.json", audit)
    manifest = pd.concat(
        [v[["review_id", "Clothing ID"]].assign(split=k) for k, v in parts.items()]
    )
    manifest.to_csv(report / "split_manifest.csv", index=False)
    counts = parts["train"].risk.value_counts().to_dict()
    weights = {int(k): len(parts["train"]) / (2 * count) for k, count in counts.items()}
    write_json(report / "class_weights.json", {str(k): v for k, v in weights.items()})
    masked = {k: mask_numeric(v, seed=42 + i) for i, (k, v) in enumerate(parts.items())}
    missing = [
        {
            "split": k,
            "column": column,
            "masked": int(v[column].isna().sum()),
            "rows": len(v),
        }
        for k, v in masked.items()
        for column in NUMERIC
    ]
    pd.DataFrame(missing).to_csv(report / "missingness_stress.csv", index=False)
    configs = [
        {
            "id": "imbalance_" + s,
            "scenario": "natural",
            "strategy": s,
            "imputation": "median",
        }
        for s in [
            "none",
            "under_half",
            "under_equal",
            "over_equal",
            "smotenc",
            "weighted",
            "full_text",
        ]
    ]
    configs += [
        {
            "id": "imputation_" + method,
            "scenario": "mcar20",
            "strategy": "none",
            "imputation": method,
        }
        for method in ["mean", "median", "knn", "iterative"]
    ]
    validation, cv_rows, candidates, resampling = [], [], {}, []
    with threadpool_limits(limits=2):
        for config in configs:
            data = masked if config["scenario"] == "mcar20" else parts
            train = data["train"]
            for fold, (tr, va) in enumerate(
                GroupKFold(3).split(train, groups=train["Clothing ID"]), 1
            ):
                fold_counts = train.iloc[tr].risk.value_counts().to_dict()
                fold_weights = {
                    int(k): len(tr) / (2 * n) for k, n in fold_counts.items()
                }
                model = make_model(
                    config["strategy"], config["imputation"], fold_weights
                )
                model.fit(train.iloc[tr], train.iloc[tr].risk)
                p = model.predict_proba(train.iloc[va])[:, 1]
                score = metrics(train.iloc[va].risk.to_numpy(), p, "risk")
                cv_rows.append({"id": config["id"], "fold": fold, **score})
            model = make_model(config["strategy"], config["imputation"], weights)
            start = time.perf_counter()
            model.fit(train, train.risk)
            val_p = model.predict_proba(data["validation"])[:, 1]
            threshold = threshold_from_validation(
                data["validation"].risk.to_numpy(), val_p
            )
            validation.append(
                {
                    **config,
                    **metrics(
                        data["validation"].risk.to_numpy(), val_p, "risk", threshold
                    ),
                    "threshold": threshold,
                    "fit_seconds": time.perf_counter() - start,
                }
            )
            encoder = model.named_steps.get("encode")
            sampler = model.named_steps.get("sample", "passthrough")
            if sampler == "passthrough":
                y_resampled = train.risk.to_numpy()
            else:
                x_resampled, y_resampled = sampler.fit_resample(
                    encoder.transform(train), train.risk
                )
                cat = x_resampled[:, -3:]
                assert np.equal(cat, np.floor(cat)).all()
            resampling.extend(
                {
                    "id": config["id"],
                    "class": label,
                    "before": int((train.risk == label).sum()),
                    "after": int((y_resampled == label).sum()),
                }
                for label in [0, 1]
            )
            candidates[config["id"]] = (model, data, threshold, config)
            print(config["id"], validation[-1]["average_precision"], flush=True)
    comparison = pd.DataFrame(validation)
    comparison.to_csv(report / "validation_experiments.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(report / "training_group_cv.csv", index=False)
    pd.DataFrame(resampling).to_csv(report / "resampling_counts.csv", index=False)
    selection = {
        scenario: group.sort_values("average_precision", ascending=False).iloc[0].id
        for scenario, group in comparison.groupby("scenario")
    }
    write_json(report / "selection.json", selection)
    outputs, predictions = [], []
    # All configurations are predeclared; winner labels remain validation-only.
    for identity, (model, data, threshold, config) in candidates.items():
        p = model.predict_proba(data["test"])[:, 1]
        outputs.append(
            {
                **config,
                **metrics(data["test"].risk.to_numpy(), p, "risk", threshold),
                "threshold": threshold,
            }
        )
        predictions.append(
            data["test"][["review_id", "Clothing ID"]].assign(
                id=identity, target=data["test"].risk, probability=p
            )
        )
    winner, _, threshold, _ = candidates[selection["natural"]]
    joblib.dump(winner, models / "review_triage.joblib")
    restored = joblib.load(models / "review_triage.joblib")
    np.testing.assert_allclose(
        winner.predict_proba(parts["test"]), restored.predict_proba(parts["test"])
    )
    write_json(
        models / "inference.json",
        {"threshold": threshold, "model": selection["natural"]},
    )
    pd.DataFrame(outputs).to_csv(report / "test_metrics.csv", index=False)
    pd.concat(predictions).to_csv(report / "test_predictions.csv", index=False)
    write_json(
        report / "run_metadata.json",
        {
            "seed": 42,
            "split_rows": {k: len(v) for k, v in parts.items()},
            "positive_label": "non-recommendation",
            "stress_scenario": (
                "20% MCAR per numeric column; simulated, not source defects"
            ),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["scikit-learn", "imbalanced-learn", "pandas", "numpy"]
            },
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "src/cartguard").glob("*.py")
            },
        },
    )
    render(root)
    return pd.DataFrame(outputs)
