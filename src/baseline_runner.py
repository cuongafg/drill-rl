from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from environment_eval import DrillEnv
from experiment_reports import save_experiment_report


def run_baseline(cfg: DictConfig) -> Path:
    """Run a configured random or fixed-spacing baseline."""
    settings = cfg.baseline
    mode = str(settings.get("mode", "random")).lower()
    if mode not in {"random", "one_to_one"}:
        raise ValueError(
            f"Unsupported baseline evaluation mode: {mode}. "
            "Choose 'random' or 'one_to_one'."
        )
    env = DrillEnv(cfg.environment)
    episode_results: list[dict[str, object]] = []
    rng = np.random.default_rng(int(cfg.seed))

    for episode_index in range(int(settings.episodes)):
        reset_options = (
            {"field_index": episode_index} if mode == "one_to_one" else None
        )
        observation, _ = env.reset(
            seed=int(rng.integers(0, 2**32 - 1)),
            options=reset_options,
        )
        terminated = False
        sequence_index = 0
        steps = 0
        while not terminated:
            if settings.strategy == "random":
                if len(env.drilled_columns) >= env.max_holes:
                    break
                coordinate = env.available_coordinates[
                    int(rng.integers(0, len(env.available_coordinates)))
                ]
                action = env.coordinate_to_action(coordinate)
            elif settings.strategy == "fixed":
                if len(env.drilled_columns) >= env.max_holes:
                    break
                distance = [2, 4, 8, 12, 16][int(settings.fallback_action)]
                x = min(
                    env.field.shape[0] - 1,
                    env.initial_position[0] + distance * (len(env.drilled_columns)),
                )
                action = env.coordinate_to_action((x, env.field.shape[1] // 2))
            else:
                sequence = list(settings.action_sequence)
                action_index = int(
                    sequence[sequence_index]
                    if sequence_index < len(sequence)
                    else settings.fallback_action
                )
                sequence_index += 1
                distance = [2, 4, 8, 12, 16][action_index]
                x = min(
                    env.field.shape[0] - 1,
                    env.initial_position[0] + distance * sequence_index,
                )
                candidate = (x, env.field.shape[1] // 2)
                action = env.coordinate_to_action(candidate)

            observation, _, terminated, truncated, _ = env.step(action)
            terminated = bool(terminated or truncated)
            steps += 1

        episode_results.append(
            {
                "num_holes": len(env.drilled_columns),
                "rmse": float(env._compute_rmse()),
                "steps": steps,
                "budget": env.max_holes,
                "drilled_coordinates": json.dumps(
                    [[int(x), int(y)] for x, y in env.drilled_columns]
                ),
            }
        )

    experiment = str(settings.name)
    report_dir = save_experiment_report(
        cfg, "baselines", experiment, episode_results
    )
    print(f"Mean holes: {np.mean([r['num_holes'] for r in episode_results]):.3f}")
    print(f"Mean RMSE: {np.mean([r['rmse'] for r in episode_results]):.3f}")
    print(f"Baseline report saved to {report_dir}")
    return report_dir
