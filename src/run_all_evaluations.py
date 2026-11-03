import hydra
from omegaconf import DictConfig, OmegaConf

from baseline_runner import run_baseline
from comparison_plot import save_comparison_plot
from configuration import CONFIG_DIR
from evaluation import evaluate


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    report_dirs = []
    for algorithm in cfg.orchestration.evaluation_algorithms:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.evaluation = OmegaConf.load(
            CONFIG_DIR / "evaluation" / f"{algorithm}.yaml"
        )
        print(f"Evaluating {str(algorithm).upper()}")
        report_dirs.append((str(algorithm).upper(), evaluate(run_cfg)))

    for scenario in cfg.orchestration.baseline_scenarios:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.baseline = OmegaConf.load(
            CONFIG_DIR / "baseline" / f"{scenario}.yaml"
        )
        print(f"Running baseline: {scenario}")
        report_dirs.append(
            (str(scenario).replace("_", " ").title(), run_baseline(run_cfg))
        )

    if bool(cfg.orchestration.get("generate_diagram", False)):
        comparison_dir = save_comparison_plot(cfg, report_dirs)
        print(f"Comparison diagram saved to {comparison_dir}")


if __name__ == "__main__":
    main()
