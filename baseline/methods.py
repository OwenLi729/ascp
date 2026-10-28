"""Naive and equal Bonferroni split-conformal baselines, independent of models."""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


METHODS = ("Naive", "Bonferroni")


def _score_array(scores):
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 3 or min(scores.shape) < 1 or scores.shape[2] < 2
            or not np.isfinite(scores).all()):
        raise ValueError("scores must be finite with shape (n, stages, classes >= 2)")
    return scores


def _marks(shape, tie_breakers, rng):
    if tie_breakers is None:
        tie_breakers = (rng if rng is not None else np.random.default_rng()).random(shape)
    marks = np.asarray(tie_breakers, dtype=float)
    if marks.shape != shape or not np.isfinite(marks).all() or ((marks < 0) | (marks >= 1)).any():
        raise ValueError("tie_breakers must have shape (n, stages) and values in [0, 1)")
    return marks


@dataclass
class CalibrationResult:
    method: str
    thresholds: np.ndarray
    threshold_tie_breakers: np.ndarray
    n_classes: int


def calibrate(method, calibration_scores, labels, alpha=0.1, rng=None, tie_breakers=None):
    """Use alpha per stage for Naive and alpha / stages for Bonferroni.

    Both use the full calibration sample. Independent uniform marks resolve
    equal scores; share explicit marks to compare methods on the same ordering.
    """
    if method not in METHODS:
        raise ValueError("method must be Naive or Bonferroni")
    alpha = Fraction(str(alpha))
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    scores = _score_array(calibration_scores)
    n, stages, classes = scores.shape
    labels = np.asarray(labels)
    if (labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer)
            or ((labels < 0) | (labels >= classes)).any()):
        raise ValueError("labels must be integer class indices in the score array")
    values = scores[np.arange(n)[:, None], np.arange(stages), labels[:, None]]
    marks = _marks((n, stages), tie_breakers, rng)
    level = alpha if method == "Naive" else alpha / stages
    target = (n + 1) * (1 - level)
    rank = -(-target.numerator // target.denominator)
    thresholds, cut_marks = np.full(stages, np.inf), np.ones(stages)
    for t in range(stages):
        if len(np.unique(np.column_stack([values[:, t], marks[:, t]]), axis=0)) != n:
            raise ValueError("score/auxiliary pairs must be distinct; redraw marks")
        if rank <= n:
            index = np.lexsort((marks[:, t], values[:, t]))[rank - 1]
            thresholds[t], cut_marks[t] = values[index, t], marks[index, t]
    return CalibrationResult(method, thresholds, cut_marks, classes)


def raw_prediction_sets(result, candidate_scores, tie_breakers=None, rng=None):
    """Apply randomized stage cutoffs; marks are shared across candidate labels."""
    scores = _score_array(candidate_scores)
    if scores.shape[1:] != (len(result.thresholds), result.n_classes):
        raise ValueError("candidate stages/classes must match calibration")
    marks = _marks(scores.shape[:2], tie_breakers, rng)[:, :, None]
    cutoffs = result.thresholds[None, :, None]
    cut_marks = result.threshold_tie_breakers[None, :, None]
    return (scores < cutoffs) | ((scores == cutoffs) & (marks <= cut_marks))


def predict_sequence(result, candidate_scores, tie_breakers=None, rng=None):
    """Return nested prediction sets by intersecting stagewise acceptances."""
    return np.logical_and.accumulate(
        raw_prediction_sets(result, candidate_scores, tie_breakers, rng), axis=1,
    )
