"""Composable validation and product-disjoint retail review ingestion."""

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CATEGORIES = ["Division Name", "Department Name", "Class Name"]
NUMERIC = ["Age", "word_count", "character_count"]


def download(root):
    folder = Path(root) / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "reviews.csv"
    if not path.exists():
        archive = folder / "reviews.zip"
        urllib.request.urlretrieve(
            "https://www.kaggle.com/api/v1/datasets/download/nicapotato/womens-ecommerce-clothing-reviews",
            archive,
        )
        with zipfile.ZipFile(archive) as z:
            path.write_bytes(
                z.read(next(n for n in z.namelist() if n.endswith(".csv")))
            )
    return path


def validate(frame):
    required = {"Clothing ID", "Age", "Review Text", "Recommended IND", *CATEGORIES}
    if missing := required - set(frame):
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if not frame["Recommended IND"].isin([0, 1]).all():
        raise ValueError("Recommendation labels must be binary")
    if frame["Clothing ID"].isna().any():
        raise ValueError("Product identifiers cannot be missing")
    if not (frame.Age.isna() | frame.Age.between(18, 100)).all():
        raise ValueError("Invalid customer ages")
    return frame.copy()


def build_features(frame):
    result = frame.copy()
    result["text"] = (
        result["Review Text"]
        .fillna("")
        .map(lambda x: " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", str(x).lower())))
    )
    result["word_count"] = result.text.str.split().str.len().astype(float)
    result["character_count"] = result.text.str.len().astype(float)
    for column in CATEGORIES:
        result[column] = (
            result[column].astype(object).where(result[column].notna(), np.nan)
        )
    return result


def remove_empty(frame):
    return frame.loc[frame.text.ne("")].copy()


def remove_duplicates(frame):
    return frame.drop_duplicates("text").copy()


def load(path):
    raw = pd.read_csv(path)
    raw["review_id"] = np.arange(len(raw))
    frame = (
        raw.pipe(validate)
        .pipe(build_features)
        .pipe(remove_empty)
        .pipe(remove_duplicates)
    )
    frame["risk"] = 1 - frame["Recommended IND"]
    audit = {
        "source_rows": len(raw),
        "usable_rows": len(frame),
        "empty_reviews": int(raw["Review Text"].fillna("").eq("").sum()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "natural_missing_numeric": raw["Age"].isna().sum().item(),
        "natural_missing_categories": {c: int(raw[c].isna().sum()) for c in CATEGORIES},
    }
    return frame.reset_index(drop=True), audit


def split(frame):
    train, rest = next(
        GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42).split(
            frame, groups=frame["Clothing ID"]
        )
    )
    subset = frame.iloc[rest]
    val, test = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43).split(
            subset, groups=subset["Clothing ID"]
        )
    )
    return {
        "train": frame.iloc[train].copy(),
        "validation": frame.iloc[rest[val]].copy(),
        "test": frame.iloc[rest[test]].copy(),
    }


def mask_numeric(frame, seed=42, rate=0.2):
    """Controlled MCAR stress scenario; not a claim about source missingness."""
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for column in NUMERIC:
        mask = rng.random(len(result)) < rate
        result.loc[mask, column] = np.nan
    return result
