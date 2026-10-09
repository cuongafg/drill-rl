import hydra
from omegaconf import DictConfig

from training import train


@hydra.main(version_base=None, config_path="../conf", config_name="train_a2c")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
