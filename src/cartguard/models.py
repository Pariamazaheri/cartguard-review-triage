"""Complete trainable pipelines; resampling is confined to fit calls."""

import numpy as np
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORIES, NUMERIC


class ReviewEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, imputation="median", components=32):
        self.imputation = imputation
        self.components = components

    def fit(self, x, y=None):
        imputer = {
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=5),
            "iterative": IterativeImputer(random_state=42, max_iter=15),
        }[self.imputation]
        self.numeric_ = make_pipeline(imputer, StandardScaler()).fit(x[NUMERIC])
        self.categories_ = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ).fit(x[CATEGORIES])
        self.text_ = make_pipeline(
            TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1, 2)),
            TruncatedSVD(n_components=self.components, random_state=42),
            StandardScaler(),
        ).fit(x.text)
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        return np.column_stack(
            [
                self.numeric_.transform(x[NUMERIC]),
                self.text_.transform(x.text),
                self.categories_.transform(x[CATEGORIES]),
            ]
        )


def make_model(strategy="none", imputation="median", class_weights=None, components=32):
    if strategy == "full_text":
        features = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(SimpleImputer(strategy=imputation), StandardScaler()),
                    NUMERIC,
                ),
                (
                    "category",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    CATEGORIES,
                ),
                (
                    "text",
                    TfidfVectorizer(max_features=20000, min_df=2, ngram_range=(1, 2)),
                    "text",
                ),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("classifier", LogisticRegression(C=2, max_iter=1000, random_state=42)),
            ]
        )
    continuous = len(NUMERIC) + components
    categories = list(range(continuous, continuous + len(CATEGORIES)))
    sampler = {
        "none": "passthrough",
        "weighted": "passthrough",
        "under_half": RandomUnderSampler(sampling_strategy=0.5, random_state=42),
        "under_equal": RandomUnderSampler(sampling_strategy=1.0, random_state=42),
        "over_equal": RandomOverSampler(sampling_strategy=1.0, random_state=42),
        "smotenc": SMOTENC(
            categorical_features=categories,
            sampling_strategy=1.0,
            random_state=42,
            k_neighbors=5,
        ),
    }[strategy]
    output = ColumnTransformer(
        [
            ("continuous", "passthrough", list(range(continuous))),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categories,
            ),
        ]
    )
    return Pipeline(
        [
            ("encode", ReviewEncoder(imputation, components)),
            ("sample", sampler),
            ("output", output),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight=class_weights if strategy == "weighted" else None,
                ),
            ),
        ]
    )
