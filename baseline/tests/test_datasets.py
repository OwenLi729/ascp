import csv
import json

import numpy as np
import pytest
from openpyxl import Workbook

from baseline.datasets import APPENDICITIS_GROUPS, Z_GROUPS, load_dataset
from baseline.experiment import run_experiment
from baseline.scores import compute_scores, fit_stage_models


@pytest.fixture
def appendicitis_file(tmp_path):
    fields = [name for group in APPENDICITIS_GROUPS for name in group]
    fields += ["Diagnosis", "Management", "Severity", "Diagnosis_Presumptive", "Length_of_Stay",
               "Alvarado_Score", "Paedriatic_Appendicitis_Score", "US_Number"]
    path = tmp_path / "appendicitis.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(25):
            row = {name: str(i % 4) for name in fields}
            row.update(Sex="female" if i % 2 else "male",
                       Diagnosis=("appendicitis" if i % 2 else "no appendicitis") if i < 24 else "NaN",
                       Management="outcome", Severity="outcome", Diagnosis_Presumptive="outcome",
                       Length_of_Stay="outcome", US_Number="ID", CRP="NaN" if i == 0 else "2")
            writer.writerow(row)
    return path


@pytest.fixture
def heart_file(tmp_path):
    path = tmp_path / "heart.data"
    rows = []
    for i in range(40):
        row = [str(1 + (i + j) % 3) for j in range(14)]
        row[-1] = str(i % 5)
        if i == 0:
            row[11] = "?"
        rows.append(",".join(row))
    path.write_text("\n".join(rows))
    return path


@pytest.fixture
def z_file(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    columns = [name for group in Z_GROUPS for name in group]
    sheet.append(columns + ["Cath"])
    for i in range(20):
        sheet.append(["Male" if name == "Sex" else i % 3 for name in columns] +
                     ["Cad" if i % 2 else "Normal"])
    sheet.append([0] * len(columns) + [None])
    path = tmp_path / "z.xlsx"
    workbook.save(path)
    workbook.close()
    return path


def test_appendicitis_excludes_outcomes_and_drops_only_unlabeled_rows(appendicitis_file):
    d = load_dataset("appendicitis", appendicitis_file)
    assert d.X.shape == (24, 46) and d.excluded_rows == 1
    assert d.classes == ["appendicitis", "no appendicitis"]
    assert [len(stage) for stage in d.stages] == [17, 29, 46]
    assert not set(d.feature_names) & {
        "Diagnosis", "Management", "Severity", "Diagnosis_Presumptive", "Length_of_Stay",
        "Alvarado_Score", "Paedriatic_Appendicitis_Score", "US_Number",
    }
    clinical = {d.feature_names[i] for i in d.stages[0]}
    assert not clinical & {"WBC_Count", "CRP", "Appendix_Diameter"}
    assert "Appendix_Diameter" not in {d.feature_names[i] for i in d.stages[1]}
    assert np.isnan(d.X[0, d.feature_names.index("CRP")])


def test_heart_maps_severity_to_binary_and_isolates_advanced_tests(heart_file):
    d = load_dataset("heart", heart_file)
    assert len(d.y) == 40 and len(d.classes) == 2
    assert d.classes[d.y[0]] == "no disease"
    assert all(d.classes[d.y[i]] == "disease" for i in (1, 2, 3, 4))
    assert [len(stage) for stage in d.stages] == [4, 7, 12, 13]
    assert "num" not in d.feature_names
    assert "ca" not in {d.feature_names[i] for i in d.stages[2]}
    assert d.feature_names[d.stages[3][-1]] == "ca"
    assert d.feature_names.index("cp") in d.categorical


def test_z_target_is_excluded_and_all_workbook_features_are_grouped(z_file):
    d = load_dataset("z_alizadeh", z_file)
    assert d.X.shape == (20, 55)
    assert d.excluded_rows == 1
    assert d.classes == ["cad", "normal"]
    assert "Cath" not in d.feature_names
    assert [len(stage) for stage in d.stages] == [17, 31, 38, 55]
    np.testing.assert_array_equal(d.stages[-1], np.arange(55))


def test_categorical_encoder_is_training_only_and_future_features_do_not_leak():
    X = np.array([[float(i % 5), "a" if i % 2 else "b", float(i % 3)]
                  for i in range(40)], dtype=object)
    y = np.arange(40) % 2
    models = fit_stage_models(X, y, (np.array([0]), np.array([0, 1, 2])), categorical=(1,))
    before = compute_scores(models, X)
    changed = X.copy()
    changed[:, 1] = "unseen-category"
    changed[:, 2] = 100000.
    after = compute_scores(models, changed)
    np.testing.assert_array_equal(before[:, 0], after[:, 0])
    assert np.isfinite(after).all()
    np.testing.assert_allclose((1 - after).sum(axis=2), 1)
    encoder = (models[1].classifier.named_steps["columntransformer"]
               .named_transformers_["categorical"].named_steps["onehotencoder"])
    assert set(encoder.categories_[0]) == {"a", "b"}


def test_new_runner_uses_binary_labels_four_stages_and_records_profile(heart_file, tmp_path):
    output = tmp_path / "run"
    summary = run_experiment(heart_file, output, repetitions=2, dataset="heart")
    assert {row["method"] for row in summary} == {"Naive", "Bonferroni"}
    assert any(row["metric"] == "mean_set_size_stage4" for row in summary)
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["dataset_key"] == "heart"
    assert metadata["classifier"]["C"] == 1.0
    assert metadata["alpha"] == 0.10
    assert len(metadata["class_names"]) == 2
    assert len(metadata["stages"]) == len(metadata["stage_names"]) == 4
    assert not set(metadata["training_indices"]) & set(metadata["held_out_indices"])
    for figure in ("stage_metrics", "overall_metrics"):
        for extension in ("png", "pdf"):
            assert (output / f"{figure}.{extension}").stat().st_size > 0
