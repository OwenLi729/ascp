# ascp

Naive and Bonferroni conformal baselines on staged real-data features.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest baseline/tests -q
.venv/bin/python -m baseline.experiment
.venv/bin/python -m baseline.experiment --dataset appendicitis heart z_alizadeh
```

Data downloads to `data/`. Current results and PNG/PDF graphs use alpha = 0.10
and go to `results/<dataset>/`. Earlier alpha = 0.05 runs are in
`results/alpha0.05/<dataset>/`.
Use `--output PATH` for another run, or `--help` to see all options.

| Dataset | Target | Cumulative stages |
|---|---|---|
| [Dermatology](https://archive.ics.uci.edu/dataset/33/dermatology) (default) | Six diagnoses | Clinical → histopathology |
| [Appendicitis](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) | Appendicitis / no appendicitis | Clinical → labs → ultrasound findings |
| [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | Disease / no disease, Cleveland cohort | Clinical → labs/ECG → stress tests → angiography |
| [Z-Alizadeh Sani](https://archive.ics.uci.edu/dataset/412/z) | CAD / normal | Demographics → examination → ECG → labs/echo |

## Choices for all datasets

- Default alpha is 0.10 and seed is 0. Keep 30% for fixed, stratified training;
  randomly split the rest into roughly 50% calibration and 20% test, 500 times.
  Calibration/test splits do not use labels for stratification.
- Fit one logistic regression per cumulative stage, with default regularization
  (`C=1`) and no tuning. Scores are `1 - predicted class probability`.
- Fit preprocessing on training data only: median imputation and scaling for
  numeric features; a missing category and one-hot encoding for categorical ones.
  Drop rows without labels; keep rows with missing features.
- Naive uses alpha at each stage; Bonferroni uses alpha divided by the number of
  stages. Intersect sets across stages. Both methods share models, splits, and
  random numbers used to break score ties for paired comparisons.
- Stages represent feature groups, not measured acquisition times.
- A case resolves at its first singleton set. Empty sets do not resolve it.
  Cases that never resolve count as one stage beyond the last in mean stopping stage.
- Graphs show means with ±1.96 MCSE error bars. These measure split variability
  with training fixed, not uncertainty across new datasets. Dotted lines mark alpha.

## Dataset-specific choices

- **Dermatology:** keep all 366 rows, including missing ages. Stage 1 uses the
  11 clinical attributes plus age; stage 2 adds 22 histopathology attributes.
  Treat ordinal scores as numeric.
- **Appendicitis:** predict `Diagnosis` using tabular features, without images.
  Drop two unlabeled rows. Exclude other outcomes, hospital stay, identifiers,
  composite scores, and free text. The source diagnosis partly depends on
  ultrasound findings, so the label is not independent of these predictors.
- **Heart Disease:** use only the 303 Cleveland patients. Map diagnosis codes
  1–4 to disease and 0 to no disease. Encode nominal codes as categories;
  put the vessel-count feature (`ca`) in the final angiography stage.
- **Z-Alizadeh Sani:** predict `Cath` using all 55 other workbook columns
  (the UCI description says 54). Group them into the source's four feature
  categories, ordered as shown above.
- **ADNI:** skipped as requested; approved data access is required.

`run.json` records exact features, exclusions, class mappings, split indices,
and the data hash.

For all four datasets, the alpha = 0.10 runs use the original `C=1` models,
features, splits, and tie randomness; only alpha changes from the 0.05 runs.

Regenerate graphs from saved results:

```sh
.venv/bin/python -m baseline.plotting --results results/dermatology
```

Data: Ilter & Guvenir (1998), [Dermatology](https://doi.org/10.24432/C5FK5P);
Marcinkevics et al. (2023), [Appendicitis](https://doi.org/10.5281/zenodo.7669442);
Janosi et al. (1988), [Heart Disease](https://doi.org/10.24432/C52P4X);
Alizadehsani et al. (2013), [Z-Alizadeh Sani](https://doi.org/10.24432/C5Q31T).
