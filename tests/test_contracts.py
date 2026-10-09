"""Leakage boundaries, resampling contracts and complete serialization."""

import joblib
import numpy as np
import pandas as pd
import pytest

from cartguard.data import (
    CATEGORIES,
    NUMERIC,
    build_features,
    mask_numeric,
    split,
    validate,
)
from cartguard.models import make_model


def fixture():
    frame = pd.DataFrame(
        {
            "Age": np.arange(60) % 40 + 20,
            "Review Text": [
                f"product excellent comfortable item number {i}"
                if i % 5
                else f"poor broken disappointing garment {i}"
                for i in range(60)
            ],
            "Clothing ID": np.arange(60) // 3,
            "Recommended IND": (np.arange(60) % 5 != 0).astype(int),
            **{c: ["A" if i % 2 else "B" for i in range(60)] for c in CATEGORIES},
        }
    )
    return build_features(frame)


def test_product_isolation_and_mcar_not_mutating_source():
    source = fixture()
    parts = split(source)
    sets = [set(f["Clothing ID"]) for f in parts.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    masked = mask_numeric(source, rate=0.5)
    assert masked[NUMERIC].isna().any().all()
    assert not source[NUMERIC].isna().any().any()


@pytest.mark.parametrize(
    "strategy",
    [
        "none",
        "under_half",
        "under_equal",
        "over_equal",
        "smotenc",
        "weighted",
        "full_text",
    ],
)
def test_train_predict_and_complete_roundtrip(strategy, tmp_path):
    x = fixture()
    y = 1 - x["Recommended IND"]
    model = make_model(strategy, class_weights={0: 0.625, 1: 2.5}, components=2)
    model.fit(x, y)
    before = model.predict_proba(x.iloc[:5])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(before, joblib.load(path).predict_proba(x.iloc[:5]))
    if strategy not in ["none", "weighted", "full_text"]:
        sampler = model.named_steps["sample"]
        _, labels = sampler.fit_resample(model.named_steps["encode"].transform(x), y)
        assert len(labels) != len(y)
    # Predictions retain one row per input: samplers must not run at inference.
    assert before.shape == (5, 2)


@pytest.mark.parametrize("method", ["mean", "median", "knn", "iterative"])
def test_imputer_fitted_on_training_and_unknown_categories(method):
    x = mask_numeric(fixture(), rate=0.2)
    y = 1 - x["Recommended IND"]
    model = make_model(imputation=method, components=2).fit(x, y)
    validation = x.iloc[:4].copy()
    validation["Age"] = 200
    validation["Class Name"] = "unseen"
    prediction = model.predict_proba(validation)
    assert np.isfinite(prediction).all()
    original = (
        model.named_steps["encode"].numeric_.named_steps["standardscaler"].mean_.copy()
    )
    model.predict_proba(validation)
    np.testing.assert_array_equal(
        original,
        model.named_steps["encode"].numeric_.named_steps["standardscaler"].mean_,
    )


def test_target_and_product_columns_never_enter_features():
    x = fixture()
    y = 1 - x["Recommended IND"]
    model = make_model(components=2).fit(x, y)
    p = model.predict_proba(x)
    changed = x.assign(
        Rating=1,
        risk=1,
        **{"Recommended IND": 0, "Clothing ID": 999, "Positive Feedback Count": 100000},
    )
    np.testing.assert_array_equal(p, model.predict_proba(changed))


def test_invalid_labels_fail_before_cleaning():
    data = fixture()
    data.loc[0, "Recommended IND"] = 3
    with pytest.raises(ValueError):
        validate(data)
