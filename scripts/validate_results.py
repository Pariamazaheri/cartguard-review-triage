"""Independent published-result verification without data downloads."""

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from cartguard.evaluation import metrics


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "reports"
    manifest = pd.read_csv(report / "split_manifest.csv")
    assert (
        manifest.review_id.is_unique
        and manifest.groupby("Clothing ID").split.nunique().max() == 1
    )
    test_ids = set(manifest[manifest.split == "test"].review_id)
    predictions = pd.read_csv(report / "test_predictions.csv")
    table = pd.read_csv(report / "test_metrics.csv")
    for row in table.itertuples():
        p = predictions[predictions.id == row.id]
        assert set(p.review_id) == test_ids and len(p) == len(test_ids)
        actual = metrics(
            p.target.to_numpy(), p.probability.to_numpy(), "risk", row.threshold
        )
        for name, value in actual.items():
            np.testing.assert_allclose(value, getattr(row, name), atol=1e-7, rtol=1e-5)
    val = pd.read_csv(report / "validation_experiments.csv")
    selected = json.loads((report / "selection.json").read_text())
    for scenario, winner in selected.items():
        assert (
            winner
            == val[val.scenario == scenario]
            .sort_values("average_precision", ascending=False)
            .iloc[0]
            .id
        )
    cv = pd.read_csv(report / "training_group_cv.csv")
    assert (cv.groupby("id").size() == 3).all()
    metadata = json.loads((report / "run_metadata.json").read_text())
    for filename, digest in metadata["source_hashes"].items():
        assert (
            hashlib.sha256((root / "src/cartguard" / filename).read_bytes()).hexdigest()
            == digest
        )
    book = nbformat.read(root / "notebooks/review_resilience.ipynb", as_version=4)
    cells = [c for c in book.cells if c.cell_type == "code"]
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    print(f"Verified {len(table)} pipelines and {len(cv)} group-CV fits")


if __name__ == "__main__":
    main()
