from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from environment import DrillEnv
from experiment_reports import save_experiment_report


def run_baseline(cfg: DictConfig) -> None:
    """Run a configured random or fixed-spacing baseline."""
    settings = cfg.baseline
    env = DrillEnv(cfg.environment)
    episode_results: list[dict[str, int | float]] = []
    rng = np.random.default_rng(int(cfg.seed))

    for _ in range(int(settings.episodes)):
        observation, _ = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
        terminated = False
        sequence_index = 0
        steps = 0
        while not terminated:
            if settings.strategy == "random":
                if len(env.drilled_columns) >= env.max_holes:
                    break
                action = int(rng.integers(0, int(settings.random_action_max) + 1))
            elif settings.strategy == "fixed":
                if len(env.drilled_columns) >= env.max_holes:
                    break
                action = int(settings.fallback_action)
            else:
                sequence = list(settings.action_sequence)
                action = int(
                    sequence[sequence_index]
                    if sequence_index < len(sequence)
                    else settings.fallback_action
                )
                sequence_index += 1

            observation, _, terminated, truncated, _ = env.step(action)
            terminated = bool(terminated or truncated)
            steps += 1

        episode_results.append(
            {
                "num_holes": len(env.drilled_columns),
                "rmse": float(env._compute_rmse()),
                "steps": steps,
                "budget": env.max_holes,
            }
        )

    experiment = str(settings.name)
    report_dir = save_experiment_report(
        cfg, "baselines", experiment, episode_results
    )
    print(f"Mean holes: {np.mean([r['num_holes'] for r in episode_results]):.3f}")
    print(f"Mean RMSE: {np.mean([r['rmse'] for r in episode_results]):.3f}")
    print(f"Baseline report saved to {report_dir}")
