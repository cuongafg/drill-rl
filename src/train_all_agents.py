import hydra
from omegaconf import DictConfig, OmegaConf

from configuration import CONFIG_DIR
from training import train


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    for algorithm in cfg.orchestration.training_algorithms:
        agent_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        agent_cfg.training = OmegaConf.load(
            CONFIG_DIR / "training" / f"{algorithm}.yaml"
        )
        print(f"Training {str(algorithm).upper()}")
        train(agent_cfg)


if __name__ == "__main__":
    main()
