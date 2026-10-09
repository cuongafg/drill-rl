import hydra
from omegaconf import DictConfig

from baseline_runner import run_baseline


@hydra.main(version_base=None, config_path="../conf", config_name="run_baseline_even_spacing")
def main(cfg: DictConfig) -> None:
    run_baseline(cfg)


if __name__ == "__main__":
    main()
