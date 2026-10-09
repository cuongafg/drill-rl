import csv
import json

from omegaconf import OmegaConf

from experiment_reports import save_experiment_report


def test_saves_episode_metrics_and_summary_under_report_root(tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 19,
            "paths": {"results_dir": str(tmp_path)},
            "evaluation": {"algorithm": "ppo", "episodes": 2},
        }
    )
    episode_results = [
        {"num_holes": 4, "rmse": 1.25},
        {"num_holes": 6, "rmse": 0.75},
    ]

    report_dir = save_experiment_report(
        cfg, "evaluations", "ppo", episode_results
    )

    assert report_dir.parent == tmp_path / "evaluations" / "ppo"
    with (report_dir / "episodes.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows == [
        {"num_holes": "4", "rmse": "1.25"},
        {"num_holes": "6", "rmse": "0.75"},
    ]

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 19
    assert summary["episode_count"] == 2
    assert summary["metrics"]["num_holes"] == {"mean": 5.0, "std": 1.0}
    assert summary["metrics"]["rmse"] == {"mean": 1.0, "std": 0.25}
    assert summary["config"]["evaluation"]["algorithm"] == "ppo"


def test_each_run_gets_a_separate_report_directory(tmp_path):
    cfg = OmegaConf.create({"seed": 1, "paths": {"results_dir": str(tmp_path)}})
    episode_results = [{"num_holes": 2, "rmse": 0.5}]

    first = save_experiment_report(cfg, "baselines", "random", episode_results)
    second = save_experiment_report(cfg, "baselines", "random", episode_results)

    assert first != second
    assert (first / "summary.json").is_file()
    assert (second / "summary.json").is_file()
