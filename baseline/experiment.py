"""Run Naive and Bonferroni on UCI Dermatology; no external-method benchmark."""
import argparse
import csv
import hashlib
from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .data import DEFAULT_DATA, STAGES, load_dermatology
from .methods import METHODS, calibrate, predict_sequence
from .metrics import evaluate
from .plotting import plot_metrics
from .scores import compute_scores, fit_stage_models


def _flatten(metrics):
    flat = {}
    for name, value in metrics.items():
        if isinstance(value, list):
            flat.update({f"{name}_stage{t}": v for t, v in enumerate(value, 1)})
        else:
            flat[name] = value
    return flat


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(data=DEFAULT_DATA, output=Path("results/dermatology"),
                   repetitions=500, alpha=0.05, seed=0):
    """Fixed stratified training set; random calibration/test splits of the rest.

    Calibration/test membership is not label-stratified, preserving exchangeable
    sampling from the held-out pool. Methods share data, models, and tie marks.
    """
    if repetitions < 1 or not 0 < alpha < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError("use positive repetitions, alpha in (0, 1), and a nonnegative integer seed")
    output, data = Path(output), Path(data)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; choose a new output path")
    X, y = load_dermatology(data)
    train_size, cal_size = round(0.3 * len(y)), round(0.5 * len(y))
    train, pool = train_test_split(np.arange(len(y)), train_size=train_size,
                                   stratify=y, random_state=seed)
    if not np.array_equal(np.unique(y[train]), np.arange(6)):
        raise ValueError("training sample must contain all six disease classes")
    with threadpool_limits(limits=1):
        models = fit_stage_models(X[train], y[train], STAGES)
        pool_scores = compute_scores(models, X[pool])
    records = []
    for repetition in range(repetitions):
        streams = np.random.SeedSequence([seed, repetition]).spawn(3)
        order = np.random.default_rng(streams[0]).permutation(len(pool))
        calibration, test = order[:cal_size], order[cal_size:]
        cal_marks = np.random.default_rng(streams[1]).random((len(calibration), len(STAGES)))
        test_marks = np.random.default_rng(streams[2]).random((len(test), len(STAGES)))
        predictions = {}
        for method in METHODS:
            fitted = calibrate(method, pool_scores[calibration], y[pool[calibration]],
                               alpha=alpha, tie_breakers=cal_marks)
            predictions[method] = predict_sequence(fitted, pool_scores[test], test_marks)
            metrics = evaluate(predictions[method], y[pool[test]])
            records.append({"repetition": repetition, "method": method, **_flatten(metrics)})
        if (predictions["Naive"] & ~predictions["Bonferroni"]).any():
            raise AssertionError("Bonferroni sets must contain matched Naive sets")
    summary = []
    for method in METHODS:
        subset = [row for row in records if row["method"] == method]
        for name in subset[0]:
            if name in ("repetition", "method"):
                continue
            values = np.array([row[name] for row in subset])
            summary.append({"method": method, "metric": name, "mean": float(values.mean()),
                            "mcse": float(values.std(ddof=1) / np.sqrt(repetitions)) if repetitions > 1 else None})
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "repetitions.csv", records)
    _write_csv(output / "summary.csv", summary)
    metadata = {
        "dataset": "https://archive.ics.uci.edu/dataset/33/dermatology",
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "alpha": alpha, "seed": seed, "repetitions": repetitions,
        "training_indices": train.tolist(), "held_out_indices": pool.tolist(),
        "calibration_size": cal_size, "test_size": len(pool) - cal_size,
        "stages": [stage.tolist() for stage in STAGES],
        "preprocessing": "training-median imputation and training-only standardization",
        "classifier": {"type": "logistic_regression", "C": 1.0, "tol": 1e-8, "max_iter": 5000},
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_metrics(summary, output, alpha)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results/dermatology"))
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        summary = run_experiment(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    for row in summary:
        if row["metric"] == "amr":
            print(f"{row['method']}: AMR={row['mean']:.4f}")


if __name__ == "__main__":
    main()
