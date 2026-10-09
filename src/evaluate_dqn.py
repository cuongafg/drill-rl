import hydra
from omegaconf import DictConfig

from evaluation import evaluate


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate_dqn")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
