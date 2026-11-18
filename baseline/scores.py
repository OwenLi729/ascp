"""Fit stage classifiers using training data alone, then freeze their scores."""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class StageModel:
    columns: np.ndarray
    classifier: object


def fit_stage_models(X, y, stages, categorical=()):
    """Fit standardized logistic regression on each stage's available features."""
    models = []
    for columns in stages:
        numeric = [i for i, column in enumerate(columns) if column not in categorical]
        categories = [i for i, column in enumerate(columns) if column in categorical]
        numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
        )
        if categories:
            preprocessing = ColumnTransformer([
                ("numeric", numeric_pipeline, numeric),
                ("categorical", make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ), categories),
            ])
            steps = [preprocessing]
        else:
            steps = [transformer for _, transformer in numeric_pipeline.steps]
        classifier = make_pipeline(
            *steps,
            LogisticRegression(C=1.0, solver="lbfgs", tol=1e-8, max_iter=5000),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(X[:, columns], y) # try setting this smaller to make it a bit worse for a study
        models.append(StageModel(np.asarray(columns), classifier))
    return models


def compute_scores(models, X):
    """Return 1 - class probability with shape (observations, stages, classes)."""
    return np.stack([
        1 - model.classifier.predict_proba(X[:, model.columns]) for model in models
    ], axis=1)
