from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from configuration import project_path


def save_comparison_plot(
    cfg: Any,
    report_dirs: list[tuple[str, Path]],
) -> Path:
    """Create publication-ready metric plots from reports generated in one run."""
    if not report_dirs:
        raise ValueError("Cannot create a comparison plot without reports.")

    rows = []
    for label, report_dir in report_dirs:
        summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        rows.append(
            {
                "model": label,
                "rmse_mean": float(metrics["rmse"]["mean"]),
                "rmse_std": float(metrics["rmse"]["std"]),
            }
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = project_path(str(cfg.paths.results_dir)) / "comparisons" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    positions = np.arange(len(rows))
    colors = [
        "#0072B2" if label in {"PPO", "A2C", "DQN"} else "#E69F00"
        for label in labels
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    bars = axis.bar(
        positions,
        [row["rmse_mean"] for row in rows],
        yerr=[row["rmse_std"] for row in rows],
        width=0.55,
        capsize=4,
        color=colors,
        edgecolor="#1f1f1f",
        linewidth=0.6,
    )
    axis.set_title("Prediction error", fontweight="bold")
    axis.set_ylabel("RMSE (lower is better)")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(top=max(row["rmse_mean"] + row["rmse_std"] for row in rows) * 1.18)
    for bar, row in zip(bars, rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["rmse_std"] + axis.get_ylim()[1] * 0.015,
            f"{row['rmse_mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    figure.suptitle("Drilling policy comparison", fontsize=14, fontweight="bold")
    figure.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    figure.savefig(output_dir / "model_comparison.pdf", bbox_inches="tight")
    plt.close(figure)
    return output_dir
