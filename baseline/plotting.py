"""Plot evaluation summaries, including previously completed runs."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from .methods import METHODS


def plot_metrics(summary, output, alpha):
    """Save stagewise and overall metrics as PNG/PDF with 1.96-MCSE bars."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    lookup = {(row["method"], row["metric"]): row for row in summary}
    colors = {"Naive": "#c44e52", "Bonferroni": "#4c72b0"}
    uncertainty = ("Error bars: mean ± 1.96 MCSE across calibration/test splits"
                   if any(row["mcse"] is not None for row in summary)
                   else "Single split; Monte Carlo error bars unavailable")

    def values(method, names):
        rows = [lookup[method, name] for name in names]
        return ([row["mean"] for row in rows],
                [0 if row["mcse"] is None else 1.96 * row["mcse"] for row in rows])

    def save(fig, name):
        fig.suptitle("UCI Dermatology — Naive vs Bonferroni", fontsize=15)
        fig.supxlabel(uncertainty, fontsize=9)
        for ax in fig.axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=180,
                        bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    stage_panels = [("cumulative_miscoverage", "Cumulative miscoverage", True),
                    ("mean_set_size", "Mean prediction-set size", False),
                    ("empty_set_rate", "Empty prediction sets", True)]
    for ax, (metric, title, proportion) in zip(axes, stage_panels):
        for index, method in enumerate(METHODS):
            means, errors = values(method, [f"{metric}_stage1", f"{metric}_stage2"])
            offset = (index - (len(METHODS) - 1) / 2) * 0.32
            positions = np.array([1, 2]) + offset
            ax.bar(positions, means, yerr=errors, width=0.3, capsize=4,
                   color=colors[method], alpha=0.9, label=method)
            for position, mean, error in zip(positions, means, errors):
                label = f"{mean:.2%}" if proportion else f"{mean:.3f}"
                ax.annotate(label, (position, mean + error), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=title, xticks=[1, 2], xlim=(0.5, 2.5))
        ax.margins(y=0.25)
        ax.set_xticklabels(["Clinical", "Clinical +\nhistopathology"])
        if proportion:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if metric == "cumulative_miscoverage":
            ax.axhline(alpha, color="#666666", linestyle="--", linewidth=1,
                       label=f"Miscoverage target: {alpha:.0%}")
            ax.legend(fontsize=9, loc="upper left")
    save(fig, "stage_metrics")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    overall_panels = [("amr", "Any-stage miscoverage", True),
                      ("mean_stopping_stage", "Mean first-singleton stage", False),
                      ("unresolved_rate", "Never reaches a singleton", True)]
    for ax, (metric, title, proportion) in zip(axes, overall_panels):
        means, errors = zip(*(values(method, [metric]) for method in METHODS))
        means, errors = np.ravel(means), np.ravel(errors)
        ax.bar(METHODS, means, yerr=errors, capsize=4,
               color=[colors[method] for method in METHODS], width=0.6, alpha=0.9)
        ax.set_title(title)
        for index, (mean, error) in enumerate(zip(means, errors)):
            label = f"{mean:.2%}" if proportion else f"{mean:.3f}"
            ax.annotate(label, (index, mean + error), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=10)
        if proportion:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if metric == "amr":
            ax.axhline(alpha, color="#666666", linestyle="--", linewidth=1,
                       label=f"Target {alpha:.0%}")
            ax.legend(fontsize=9)
        if metric == "mean_stopping_stage":
            ax.set(ylim=(0, 3), ylabel="Unresolved assigned stage 3")
        else:
            ax.margins(y=0.2)
    save(fig, "overall_metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/dermatology"))
    args = parser.parse_args()
    with (args.results / "summary.csv").open(newline="") as handle:
        summary = [{**row, "mean": float(row["mean"]),
                    "mcse": float(row["mcse"]) if row["mcse"] else None}
                   for row in csv.DictReader(handle)]
    alpha = json.loads((args.results / "run.json").read_text())["alpha"]
    plot_metrics(summary, args.results, alpha)


if __name__ == "__main__":
    main()
