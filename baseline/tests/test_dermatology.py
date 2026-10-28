from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

import baseline.data as data_module
from baseline.data import STAGES, load_dermatology
from baseline.experiment import run_experiment
from baseline.scores import compute_scores, fit_stage_models


@pytest.fixture
def data_file(tmp_path):
    # Small deterministic parser fixture in UCI's exact 35-column layout.
    rows = []
    for label in range(1, 7):
        for i in range(12):
            features = [str((label + j + i) % 4) for j in range(33)]
            features.append("?" if i == 0 else str(20 + i))
            rows.append(",".join(features + [str(label)]))
    path = tmp_path / "dermatology.data"
    path.write_text("\n".join(rows))
    return path


def test_loader_maps_classes_and_preserves_missing_age(data_file):
    X, y = load_dermatology(data_file)
    assert X.shape == (72, 34)
    np.testing.assert_array_equal(np.unique(y), np.arange(6))
    assert np.isnan(X[:, 33]).sum() == 6
    assert len(STAGES[0]) == 12 and STAGES[0][-1] == 33
    assert len(STAGES[1]) == 34
    np.testing.assert_array_equal(np.sort(STAGES[1]), np.arange(34))
    assert not set(STAGES[0]) & set(range(11, 33))


def test_download_is_cached_without_network_calls_afterward(data_file, tmp_path, monkeypatch):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("dermatology.data", data_file.read_bytes())
    calls = []
    def open_archive(*args, **kwargs):
        calls.append(args)
        return BytesIO(buffer.getvalue())
    monkeypatch.setattr(data_module, "urlopen", open_archive)
    target = tmp_path / "cache" / "dermatology.data"
    first = load_dermatology(target)
    second = load_dermatology(target)
    assert len(calls) == 1
    np.testing.assert_array_equal(first[0], second[0])


def test_imputation_and_scaling_fit_only_on_training_and_no_future_leakage(data_file):
    X, y = load_dermatology(data_file)
    models = fit_stage_models(X, y, STAGES)
    medians = models[0].classifier.named_steps["simpleimputer"].statistics_.copy()
    scores = compute_scores(models, X)
    assert scores.shape == (72, 2, 6)
    np.testing.assert_allclose((1 - scores).sum(axis=2), 1)
    changed = X.copy()
    changed[:, 11:33] = 10000
    np.testing.assert_array_equal(compute_scores(models, changed)[:, 0], scores[:, 0])
    changed[:, 33] = 10000
    assert np.isfinite(compute_scores(models, changed)).all()
    np.testing.assert_array_equal(models[0].classifier.named_steps["simpleimputer"].statistics_, medians)


def test_real_data_runner_is_reproducible_and_writes_only_two_baselines(data_file, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = run_experiment(data_file, a, repetitions=3)
    assert first == run_experiment(data_file, b, repetitions=3)
    assert {row["method"] for row in first} == {"Naive", "Bonferroni"}
    for name in ("repetitions.csv", "summary.csv", "run.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    metadata = json.loads((a / "run.json").read_text())
    train, held_out = set(metadata["training_indices"]), set(metadata["held_out_indices"])
    assert not train & held_out and len(train | held_out) == 72
    assert metadata["calibration_size"] + metadata["test_size"] == len(held_out)
    with pytest.raises(ValueError, match="empty"):
        run_experiment(data_file, a, repetitions=3)


def test_loader_rejects_bad_file_layout_and_missing_non_age(data_file, tmp_path):
    path = tmp_path / "bad.data"
    path.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="34 features"):
        load_dermatology(path)
    fields = data_file.read_text().splitlines()[0].split(",")
    fields[0] = "?"
    path.write_text(",".join(fields))
    with pytest.raises(ValueError, match="only age"):
        load_dermatology(path)
