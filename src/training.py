from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from stable_baselines3 import A2C, DQN, PPO

from configuration import project_path
from environment import DrillEnv


AGENT_TYPES = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def train(cfg: DictConfig) -> None:
    """Build, train, and save the configured Stable-Baselines3 agent."""
    algorithm = str(cfg.training.algorithm).lower()
    if algorithm not in AGENT_TYPES:
        raise ValueError(f"Unsupported training algorithm: {algorithm}")

    env = DrillEnv(cfg.environment)
    algorithm_type = AGENT_TYPES[algorithm]
    model = algorithm_type(
        str(cfg.training.policy),
        env,
        verbose=int(cfg.training.verbose),
        **_resolve_parameters(cfg.training.parameters),
    )
    model.learn(total_timesteps=int(cfg.training.total_timesteps))

    model_path = project_path(f"{cfg.paths.model_dir}/{cfg.training.model_name}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Saved {algorithm.upper()} model to {model_path}")


def _resolve_parameters(parameters: DictConfig) -> dict[str, Any]:
    return {key: value for key, value in parameters.items()}
