"""Fit stage classifiers using training data alone, then freeze their scores."""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class StageModel:
    columns: np.ndarray
    classifier: object


def fit_stage_models(X, y, stages):
    """Fit standardized logistic regression on each stage's available features."""
    models = []
    for columns in stages:
        classifier = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", tol=1e-8, max_iter=5000),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(X[:, columns], y)
        models.append(StageModel(np.asarray(columns), classifier))
    return models


def compute_scores(models, X):
    """Return 1 - class probability with shape (observations, stages, classes)."""
    return np.stack([
        1 - model.classifier.predict_proba(X[:, model.columns]) for model in models
    ], axis=1)
