from __future__ import annotations

import json
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C

from environment import DrillEnv
from masked_policies import (
    MaskedActorCriticPolicy,
    MaskedDQN,
    MaskedDQNPolicy,
    MaskedPPOActorCriticPolicy,
    SharedCnnFeatureExtractor,
)
from model_artifacts import (
    create_model_run_id,
    model_checkpoint_filename,
    model_run_directory,
)


AGENT_TYPES = {"ppo": MaskablePPO, "a2c": A2C, "dqn": MaskedDQN}


def train(cfg: DictConfig) -> None:
    """Build, train, and save the configured Stable-Baselines3 agent."""
    algorithm = str(cfg.training.algorithm).lower()
    if algorithm not in AGENT_TYPES:
        raise ValueError(f"Unsupported training algorithm: {algorithm}")

    env = DrillEnv(cfg.environment)
    algorithm_type = AGENT_TYPES[algorithm]
    use_cnn = str(cfg.model.get("encoder", "mlp")).lower() == "cnn"
    if algorithm == "ppo":
        policy = MaskedPPOActorCriticPolicy if use_cnn else str(cfg.training.policy)
    elif algorithm == "a2c":
        policy = MaskedActorCriticPolicy
    else:
        policy = MaskedDQNPolicy
    policy_kwargs = _resolve_policy_kwargs(cfg.model, cfg.environment)
    model = algorithm_type(
        policy,
        env,
        verbose=int(cfg.training.verbose),
        policy_kwargs=policy_kwargs,
        **_resolve_parameters(cfg.training.parameters),
    )
    model.learn(total_timesteps=int(cfg.training.total_timesteps))

    run_id = create_model_run_id()
    run_dir = model_run_directory(str(cfg.paths.model_dir), algorithm, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_filename = model_checkpoint_filename(str(cfg.training.model_name))
    model_path = run_dir / checkpoint_filename
    model.save(str(model_path))

    config_snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(config_snapshot, "training.run_id", run_id, force_add=True)
    (run_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config_snapshot, resolve=True), encoding="utf-8"
    )
    metadata = {
        "run_id": run_id,
        "algorithm": algorithm,
        "label": str(cfg.training.model_name),
        "total_timesteps": int(cfg.training.total_timesteps),
        "model_file": checkpoint_filename,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {algorithm.upper()} model run {run_id} to {run_dir}")


def _resolve_parameters(parameters: DictConfig) -> dict[str, Any]:
    return {key: value for key, value in parameters.items()}


def _resolve_policy_kwargs(
    model_config: DictConfig,
    environment_config: DictConfig,
) -> dict[str, Any]:
    """Resolve the shared policy architecture for PPO, A2C, and DQN."""
    policy_kwargs = dict(model_config.policy_kwargs)
    if str(model_config.get("encoder", "mlp")).lower() == "cnn":
        observation = environment_config["observation"]
        policy_kwargs["features_extractor_class"] = SharedCnnFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "field_shape": tuple(int(value) for value in observation["field_shape"]),
            "output_dim": int(policy_kwargs.pop("feature_extractor_output_dim", 128)),
        }
        return policy_kwargs
    activation_name = str(policy_kwargs.get("activation_fn", "tanh")).lower()
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    try:
        policy_kwargs["activation_fn"] = activations[activation_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported MLP activation '{activation_name}'. "
            f"Choose one of {sorted(activations)}."
        ) from error
    policy_kwargs["net_arch"] = [int(layer) for layer in policy_kwargs["net_arch"]]
    return policy_kwargs
