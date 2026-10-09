import hydra
from omegaconf import DictConfig, OmegaConf

from baseline_runner import run_baseline
from configuration import CONFIG_DIR
from evaluation import evaluate


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    for algorithm in cfg.orchestration.evaluation_algorithms:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.evaluation = OmegaConf.load(
            CONFIG_DIR / "evaluation" / f"{algorithm}.yaml"
        )
        print(f"Evaluating {str(algorithm).upper()}")
        evaluate(run_cfg)

    for scenario in cfg.orchestration.baseline_scenarios:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.baseline = OmegaConf.load(
            CONFIG_DIR / "baseline" / f"{scenario}.yaml"
        )
        print(f"Running baseline: {scenario}")
        run_baseline(run_cfg)


if __name__ == "__main__":
    main()
