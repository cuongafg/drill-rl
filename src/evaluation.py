from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import A2C, DQN, PPO

from environment_eval import DrillEnv
from experiment_reports import save_experiment_report
from model_artifacts import model_checkpoint_filename, model_run_directory


AGENT_TYPES = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def evaluate(cfg: DictConfig) -> None:
    """Evaluate the configured policy and report mean boreholes and RMSE."""
    algorithm = str(cfg.evaluation.algorithm).lower()
    if algorithm not in AGENT_TYPES:
        raise ValueError(f"Unsupported evaluation algorithm: {algorithm}")
    model_path = (
        model_run_directory(
            str(cfg.paths.model_dir),
            algorithm,
            str(cfg.evaluation.model_run_id),
        )
        / model_checkpoint_filename(str(cfg.evaluation.model_name))
    )
    model = AGENT_TYPES[algorithm].load(str(model_path))
    env = DrillEnv(cfg.environment)

    episode_results: list[dict[str, int | float]] = []
    for _ in range(int(cfg.evaluation.episodes)):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=bool(cfg.evaluation.deterministic),
            )
            observation, _, terminated, truncated, info = env.step(action)
        episode_results.append(
            {"num_holes": int(info["num_holes"]), "rmse": float(info["rmse"])}
        )

    report_dir = save_experiment_report(
        cfg, "evaluations", algorithm, episode_results
    )
    print(f"Mean holes: {np.mean([r['num_holes'] for r in episode_results]):.3f}")
    print(f"Mean RMSE: {np.mean([r['rmse'] for r in episode_results]):.3f}")
    print(f"Evaluation report saved to {report_dir}")
