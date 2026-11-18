"""Dataset-specific labels and acquisition groups for real-data baselines."""
import csv
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from .data import DEFAULT_DATA, STAGES, load_dermatology


DATASETS = ("dermatology", "appendicitis", "heart", "z_alizadeh")


@dataclass
class Dataset:
    key: str
    title: str
    source: str
    path: Path
    X: np.ndarray
    y: np.ndarray
    classes: list
    feature_names: list
    stages: tuple
    stage_names: tuple
    categorical: tuple = ()
    excluded_columns: list = field(default_factory=list)
    excluded_rows: int = 0
    notes: list = field(default_factory=list)


def _fetch(path, url, member=None):
    path = Path(path)
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if member:
            with ZipFile(BytesIO(payload)) as archive:
                payload = archive.read(member)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return path


def _missing(value):
    return value is None or str(value).strip().lower() in ("", "?", "nan", "na", "none")


def _assemble(key, title, source, path, rows, target, groups, stage_names,
              categorical=(), notes=()):
    if not rows:
        raise ValueError("dataset has no observations")
    features = [name for group in groups for name in group]
    if len(set(features)) != len(features) or target in features:
        raise ValueError("acquisition groups must be disjoint and exclude the target")
    missing_columns = set(features + [target]) - set(rows[0])
    if missing_columns:
        raise ValueError("missing dataset columns: " + ", ".join(sorted(missing_columns)))
    labeled = [row for row in rows if not _missing(row[target])]
    if not labeled:
        raise ValueError("dataset has no labeled observations")
    classes, y = np.unique([str(row[target]).strip() for row in labeled], return_inverse=True)
    if len(classes) < 2:
        raise ValueError("classification needs at least two classes")
    X = np.empty((len(labeled), len(features)), dtype=object)
    categories = []
    for j, name in enumerate(features):
        values = [np.nan if _missing(row[name]) else str(row[name]).strip() for row in labeled]
        try:
            numeric = np.asarray(values, dtype=float)
        except ValueError:
            numeric = None
        if name in categorical or numeric is None:
            X[:, j] = values
            categories.append(j)
        else:
            if np.isinf(numeric).any():
                raise ValueError(f"infinite numeric values in {name}")
            X[:, j] = numeric
    stages = tuple(np.arange(end) for end in np.cumsum([len(group) for group in groups]))
    return Dataset(key, title, source, Path(path), X, y, classes.tolist(), features,
                   stages, tuple(stage_names), tuple(categories),
                   sorted(set(rows[0]) - set(features) - {target}),
                   len(rows) - len(labeled), list(notes))


APPENDICITIS_GROUPS = (
    ("Age", "BMI", "Sex", "Height", "Weight", "Migratory_Pain", "Lower_Right_Abd_Pain",
     "Contralateral_Rebound_Tenderness", "Coughing_Pain", "Nausea", "Loss_of_Appetite",
     "Body_Temperature", "Dysuria", "Stool", "Peritonitis", "Psoas_Sign",
     "Ipsilateral_Rebound_Tenderness"),
    ("WBC_Count", "Neutrophil_Percentage", "Segmented_Neutrophils", "Neutrophilia",
     "RBC_Count", "Hemoglobin", "RDW", "Thrombocyte_Count", "Ketones_in_Urine",
     "RBC_in_Urine", "WBC_in_Urine", "CRP"),
    ("Appendix_on_US", "Appendix_Diameter", "Free_Fluids", "Appendix_Wall_Layers",
     "Target_Sign", "Appendicolith", "Perfusion", "Perforation", "Surrounding_Tissue_Reaction",
     "Appendicular_Abscess", "Pathological_Lymph_Nodes", "Bowel_Wall_Thickening",
     "Conglomerate_of_Bowel_Loops", "Ileus", "Coprostasis", "Meteorism", "Enteritis"),
)
HEART_COLUMNS = ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                 "exang", "oldpeak", "slope", "ca", "thal", "num")
HEART_GROUPS = (("age", "sex", "cp", "trestbps"), ("chol", "fbs", "restecg"),
                ("thalach", "exang", "oldpeak", "slope", "thal"), ("ca",))
Z_GROUPS = (
    ("Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker",
     "FH", "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP"),
    ("BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur",
     "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical",
     "Nonanginal", "Exertional CP", "LowTH Ang"),
    ("Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"),
    ("FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph",
     "Neut", "PLT", "EF-TTE", "Region RWMA", "VHD"),
)


def load_dataset(name="dermatology", path=None):
    """Load a cached source file, downloading only public datasets if needed."""
    if name == "dermatology":
        path = Path(path or DEFAULT_DATA)
        X, y = load_dermatology(path)
        return Dataset(name, "UCI Dermatology", "https://archive.ics.uci.edu/dataset/33/dermatology",
                       path, X, y, [str(i) for i in range(1, 7)],
                       [f"UCI_column_{i}" for i in range(1, 35)], STAGES,
                       ("Clinical", "Clinical +\nhistopathology"))
    if name == "appendicitis":
        path = _fetch(path or "data/appendicitis.csv",
                      "https://archive.ics.uci.edu/static/public/938/data.csv")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return _assemble(name, "Regensburg Pediatric Appendicitis",
                         "https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis",
                         path, rows, "Diagnosis", APPENDICITIS_GROUPS,
                         ("Clinical", "+ Labs", "+ Ultrasound"), notes=(
                             "Tabular findings only; ultrasound images are not used.",
                             "Exclude other outcomes, presumptive diagnosis, length of stay, image IDs, composite scores, and free-text fields.",
                             "For non-operated patients, the source diagnosis partly depends on clinical scores and appendix diameter; it is not an independent gold standard.",))
    if name == "heart":
        path = _fetch(path or "data/heart.data",
                      "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
                      "processed.cleveland.data")
        with path.open(newline="") as handle:
            rows = [dict(zip(HEART_COLUMNS, row)) for row in csv.reader(handle) if row]
        for row in rows:
            diagnosis = float(row["num"])
            if diagnosis not in range(5):
                raise ValueError("Cleveland num must be 0 through 4")
            row["num"] = "disease" if diagnosis > 0 else "no disease"
        return _assemble(name, "UCI Heart Disease (Cleveland)",
                         "https://archive.ics.uci.edu/dataset/45/heart+disease", path, rows,
                         "num", HEART_GROUPS, ("Clinical", "+ Labs/ECG", "+ Stress tests", "+ Angiography"),
                         categorical=("sex", "cp", "fbs", "restecg", "exang", "slope", "thal"),
                         notes=("Cleveland cohort only; binary num > 0 versus num = 0.",
                                "ca is the final fluoroscopy/angiography feature; stages are feature-group proxies, not recorded timestamps."))
    if name == "z_alizadeh":
        path = _fetch(path or "data/z_alizadeh.xlsx",
                      "https://archive.ics.uci.edu/static/public/412/z%2Balizadeh%2Bsani.zip",
                      "Z-Alizadeh sani dataset.xlsx")
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook.active.iter_rows(values_only=True)
            header = [str(value).strip() for value in next(sheet)]
            rows = [dict(zip(header, row)) for row in sheet if any(v is not None for v in row)]
        finally:
            workbook.close()
        for row in rows:
            if _missing(row["Cath"]):
                continue
            row["Cath"] = str(row["Cath"]).strip().lower()
            if row["Cath"] not in ("cad", "normal"):
                raise ValueError("unexpected Z-Alizadeh Cath label")
        return _assemble(name, "Z-Alizadeh Sani", "https://archive.ics.uci.edu/dataset/412/z",
                         path, rows, "Cath", Z_GROUPS,
                         ("Demographics", "+ Examination", "+ ECG", "+ Labs/echo"),
                         notes=("Use the source's four feature groups; Cath is the target, never a predictor.",))
    raise ValueError("unknown dataset: " + name)
