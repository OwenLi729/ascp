# ascp

Naive and Bonferroni conformal baselines on
[UCI Dermatology](https://archive.ics.uci.edu/dataset/33/dermatology).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest baseline/tests -q
.venv/bin/python -m baseline.experiment
```

Data downloads to `data/`. Results and PNG/PDF graphs go to `results/dermatology/`.
Use `--output PATH` for another run, or `--help` to see all options.

- Stage 1: 12 clinical features.
- Stage 2: add 22 histopathology features.
- Defaults: alpha = 0.05, 110 training patients, and 500 splits with 183
  calibration and 73 test patients. Both methods use the same logistic models and splits.

Regenerate graphs from saved results:

```sh
.venv/bin/python -m baseline.plotting --results results/dermatology
```

Data: Ilter & Guvenir (1998), [UCI Dermatology](https://doi.org/10.24432/C5FK5P), CC BY 4.0.
