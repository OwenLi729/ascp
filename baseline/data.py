"""UCI Dermatology data and its clinical / histopathology acquisition stages."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


DATA_URL = "https://archive.ics.uci.edu/static/public/33/dermatology.zip"
DEFAULT_DATA = Path("data/dermatology.data")
CLINICAL = np.array(list(range(11)) + [33])  # UCI columns 1-11 and 34 (age).
HISTOPATHOLOGY = np.arange(11, 33)  # UCI columns 12-33.
STAGES = (CLINICAL, np.concatenate([CLINICAL, HISTOPATHOLOGY]))


def load_dermatology(path=DEFAULT_DATA):
    """Cache the original UCI file; preserve '?' ages for training-only imputation."""
    path = Path(path)
    if not path.exists():
        with urlopen(DATA_URL, timeout=30) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                payload = archive.read("dermatology.data")
        # Parse before caching, so a malformed download does not become the cache.
        values = _parse(BytesIO(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    else:
        values = _parse(path)
    return values[:, :34], values[:, 34].astype(int) - 1


def _parse(source):
    values = np.genfromtxt(source, delimiter=",", missing_values="?", filling_values=np.nan, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 35 or not len(values):
        raise ValueError("expected UCI Dermatology's 34 features and one class column")
    if not np.isfinite(values[:, :33]).all() or np.isinf(values[:, 33]).any():
        raise ValueError("only age may be missing; all other features must be finite")
    if not np.isin(values[:, 34], np.arange(1, 7)).all():
        raise ValueError("Dermatology class labels must be integers 1 through 6")
    return values
