from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from omegaconf import DictConfig

from configuration import as_dict, project_path


def save_experiment_report(
    cfg: DictConfig,
    category: str,
    experiment: str,
    episode_results: list[dict[str, Any]],
) -> Path:
    """Save episode metrics and a reproducible summary in a unique report folder."""
    if not episode_results:
        raise ValueError("Cannot save an experiment report without episode results.")

    metric_names = list(episode_results[0])
    if any(list(result) != metric_names for result in episode_results):
        raise ValueError("Every episode result must contain the same metrics.")

    generated_at = datetime.now(UTC)
    report_dir = (
        project_path(str(cfg.paths.results_dir))
        / category
        / experiment
        / generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    )
    report_dir.mkdir(parents=True, exist_ok=False)

    summary = {}
    for metric in metric_names:
        values = [result[metric] for result in episode_results]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            summary[metric] = {
                "mean": fmean(float(value) for value in values),
                "std": pstdev(float(value) for value in values),
            }
    report = {
        "category": category,
        "experiment": experiment,
        "generated_at_utc": generated_at.isoformat(),
        "seed": int(cfg.seed),
        "episode_count": len(episode_results),
        "metrics": summary,
        "config": as_dict(cfg),
    }

    with (report_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_names)
        writer.writeheader()
        writer.writerows(episode_results)

    with (report_dir / "summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, sort_keys=True)
        json_file.write("\n")

    return report_dir
