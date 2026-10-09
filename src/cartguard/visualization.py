"""Pipeline quality and operating tradeoff figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def render(root):
    report = Path(root) / "reports"
    folder = report / "figures"
    sns.set_theme(style="whitegrid")

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"), dpi=150, bbox_inches="tight")
        plt.close()

    scores = pd.read_csv(report / "test_metrics.csv")
    for scenario in ["natural", "mcar20"]:
        s = scores[scores.scenario == scenario]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(s.id, s.average_precision, color="#287f8e")
        ax.set(
            title=f"Held-out product quality: {scenario}",
            xlabel="Average precision",
            xlim=(0, 1),
        )
        save(scenario + "_comparison")
    counts = pd.read_csv(report / "resampling_counts.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        counts[counts.id.str.startswith("imbalance")],
        x="id",
        y="after",
        hue="class",
        ax=ax,
    )
    ax.set(
        title="Training-only class counts after resampling",
        xlabel="",
        ylabel="Training vectors",
    )
    ax.tick_params(axis="x", rotation=30)
    save("resampling_counts")
    prediction = pd.read_csv(report / "test_predictions.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for identity, g in prediction[prediction.id.str.startswith("imbalance")].groupby(
        "id"
    ):
        precision, recall, _ = precision_recall_curve(g.target, g.probability)
        ax.plot(recall, precision, label=identity.replace("imbalance_", ""))
    ax.set(
        title="Review-triage precision and recall", xlabel="Recall", ylabel="Precision"
    )
    ax.legend()
    save("precision_recall")
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = {
        "none": (8, 22),
        "weighted": (8, 0),
        "smotenc": (8, -20),
        "under_equal": (8, -14),
        "under_half": (-8, 16),
        "over_equal": (8, 4),
        "full_text": (-14, -20),
    }
    for row in scores[scores.scenario == "natural"].itertuples():
        ax.scatter(row.recall, row.precision, s=90)
        ax.annotate(
            row.strategy,
            (row.recall, row.precision),
            xytext=offsets[row.strategy],
            ha="right" if row.strategy == "full_text" else "left",
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7},
            textcoords="offset points",
        )
    ax.set(
        title="Validation thresholds applied to held-out products",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.margins(x=0.1, y=0.2)
    save("operating_tradeoff")
