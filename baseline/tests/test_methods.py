from fractions import Fraction

import numpy as np
import pytest

from baseline.methods import METHODS, calibrate, predict_sequence, raw_prediction_sets
from baseline.metrics import evaluate


@pytest.mark.parametrize("stages,classes", [(1, 2), (2, 6), (4, 3)])
@pytest.mark.parametrize("method", METHODS)
def test_randomized_cutoffs_match_direct_conformal_pvalues(stages, classes, method):
    rng = np.random.default_rng(4)
    # Discrete scores exercise exact ties; these are unit-test inputs, not a simulation runner.
    scores = rng.integers(0, 4, size=(100, stages, classes)) / 4
    labels = rng.integers(classes, size=100)
    test = rng.integers(0, 4, size=(20, stages, classes)) / 4
    cal_marks, test_marks = rng.random((100, stages)), rng.random((20, stages))
    fitted = calibrate(method, scores, labels, alpha=0.1, tie_breakers=cal_marks)
    values = scores[np.arange(100)[:, None], np.arange(stages), labels[:, None]]
    reference = np.zeros_like(test, dtype=bool)
    level = Fraction(1, 10) if method == "Naive" else Fraction(1, 10 * stages)
    for t in range(stages):
        for k in range(classes):
            greater = ((values[:, t, None] > test[None, :, t, k]) |
                       ((values[:, t, None] == test[None, :, t, k]) &
                        (cal_marks[:, t, None] >= test_marks[None, :, t])))
            reference[:, t, k] = (1 + greater.sum(axis=0)) / 101 > float(level)
    np.testing.assert_array_equal(raw_prediction_sets(fitted, test, test_marks), reference)
    nested = predict_sequence(fitted, test, test_marks)
    np.testing.assert_array_equal(nested, np.logical_and.accumulate(reference, axis=1))


def test_bonferroni_contains_matched_naive_with_all_tied_scores():
    scores = np.ones((100, 2, 6))
    labels = np.zeros(100, dtype=int)
    rng = np.random.default_rng(2)
    marks, test_marks = rng.random((100, 2)), rng.random((20, 2))
    test = np.ones((20, 2, 6))
    masks = [predict_sequence(calibrate(method, scores, labels, tie_breakers=marks), test, test_marks)
             for method in METHODS]
    assert not (masks[0] & ~masks[1]).any()
    # Reordering rows along with their auxiliary marks cannot change cutoffs.
    order = rng.permutation(100)
    first = calibrate("Bonferroni", scores, labels, tie_breakers=marks)
    second = calibrate("Bonferroni", scores[order], labels[order], tie_breakers=marks[order])
    np.testing.assert_array_equal(first.threshold_tie_breakers, second.threshold_tie_breakers)


def test_seed_replay_and_small_calibration_infinite_cutoff():
    scores = np.ones((3, 2, 6))
    labels = np.zeros(3, dtype=int)
    a = calibrate("Naive", scores, labels, alpha=0.5, rng=np.random.default_rng(2))
    b = calibrate("Naive", scores, labels, alpha=0.5, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a.threshold_tie_breakers, b.threshold_tie_breakers)
    for method in METHODS:
        fitted = calibrate(method, scores, labels, alpha=0.05)
        assert np.isinf(fitted.thresholds).all()
        assert predict_sequence(fitted, scores).all()


def test_rejects_invalid_labels_and_mismatched_prediction_shape():
    scores = np.ones((10, 2, 6))
    with pytest.raises(ValueError, match="labels"):
        calibrate("Naive", scores, np.full(10, 6))
    fitted = calibrate("Naive", scores, np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="match calibration"):
        predict_sequence(fitted, np.ones((3, 3, 6)))
    with pytest.raises(ValueError, match="distinct"):
        calibrate("Naive", scores, np.zeros(10, dtype=int), tie_breakers=np.zeros((10, 2)))


def test_metrics_count_empty_sets_as_unresolved_and_measure_any_stage_error():
    masks = np.array([
        [[1, 0], [1, 0]],
        [[1, 1], [0, 1]],
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    ], dtype=bool)
    result = evaluate(masks, np.array([0, 1, 0, 1]))
    assert result["amr"] == 0.25
    assert result["mean_stopping_stage"] == 2.25
    assert result["unresolved_rate"] == 0.5
    np.testing.assert_allclose(result["mean_set_size"], [1.25, 1.0])
    np.testing.assert_allclose(result["empty_set_rate"], [0.25, 0.25])
    masks[2, 1, 0] = True
    with pytest.raises(ValueError, match="nested"):
        evaluate(masks, np.array([0, 1, 0, 1]))
