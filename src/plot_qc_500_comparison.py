from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORTS = {
    "PPO\n(200k)": Path(
        "reports/experiments/evaluations/ppo/20261106T024555.791245Z"
    ),
    "A2C\n(200k)": Path(
        "reports/experiments/evaluations/a2c/20261106T030612.463317Z"
    ),
    "DQN\n(200k)": Path(
        "reports/experiments/evaluations/dqn/20261106T030601.945348Z"
    ),
    "Random": Path(
        "reports/experiments/baselines/random/20261106T024601.962072Z"
    ),
    "Even spacing": Path(
        "reports/experiments/baselines/even_spacing/20261106T024608.619008Z"
    ),
}
COLORS = ["#4c78a8", "#59a14f", "#e15759", "#7f7f7f", "#f0ad4e"]


def load_rmse(report_dir: Path) -> np.ndarray:
    with (report_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
        return np.asarray(
            [float(row["rmse"]) for row in csv.DictReader(file)],
            dtype=np.float64,
        )


def main() -> None:
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(REPORTS)
    values = [load_rmse(path) for path in REPORTS.values()]
    means = np.asarray([value.mean() for value in values])
    stds = np.asarray([value.std(ddof=0) for value in values])
    best_index = int(np.argmin(means))

    figure, axis = plt.subplots(figsize=(11, 6.5))
    bars = axis.bar(
        np.arange(len(names)),
        means,
        yerr=stds,
        capsize=6,
        color=COLORS,
        edgecolor="#333333",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )
    bars[best_index].set_edgecolor("#1b5e20")
    bars[best_index].set_linewidth(2.5)
    for index, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        label = f"{mean:.2f}"
        if index == best_index:
            label += "\n(best)"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.35,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axis.set_title(
        "500-Field QC Evaluation: Mean RMSE",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    axis.set_ylabel("RMSE (lower is better)", fontsize=12)
    axis.set_xticks(np.arange(len(names)), names, fontsize=11)
    axis.set_ylim(0, max(means + stds) * 1.25)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.text(
        0.99,
        0.97,
        "500 one-to-one QC fields\n7 holes per episode\nError bars: +/- 1 standard deviation",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )
    figure.tight_layout()
    figure.savefig(output_dir / "qc_500_mean_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    boxplot = axis.boxplot(
        values,
        tick_labels=names,
        patch_artist=True,
        boxprops={"linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 2.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    for patch, color in zip(boxplot["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axis.set_title(
        "500-Field QC Evaluation: RMSE Distribution",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    axis.set_ylabel("Per-field RMSE (lower is better)", fontsize=12)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.text(
        0.99,
        0.97,
        "Each point represents one evaluation field\nMedian shown by black line",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "qc_500_rmse_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    summary = {
        name.replace("\n", " "): {
            "mean_rmse": float(mean),
            "std_rmse": float(std),
            "fields": int(len(value)),
        }
        for name, value, mean, std in zip(names, values, means, stds)
    }
    (output_dir / "qc_500_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved diagrams to {output_dir}")


if __name__ == "__main__":
    main()
