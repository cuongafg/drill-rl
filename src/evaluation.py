from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C

from baseline_runner import run_baseline
from environment_eval import DrillEnv
from experiment_reports import save_experiment_report
from model_artifacts import model_checkpoint_filename, model_run_directory
from masked_policies import MaskedDQN


AGENT_TYPES = {"ppo": MaskablePPO, "a2c": A2C, "dqn": MaskedDQN}
COMPARISON_CONFIGS = {
    "random": {
        "name": "random",
        "strategy": "random",
        "target_holes": 7,
        "random_action_max": 4,
        "action_sequence": [],
        "fallback_action": 3,
    },
    "even_spacing": {
        "name": "even_spacing",
        "strategy": "sequence",
        "target_holes": None,
        "random_action_max": 4,
        "action_sequence": [3, 3, 3, 3, 3, 3, 3],
        "fallback_action": 3,
    },
    "random_7holes": {
        "name": "random_7holes",
        "strategy": "random",
        "target_holes": 7,
        "random_action_max": 4,
        "action_sequence": [],
        "fallback_action": 3,
    },
    "even_spacing_7holes": {
        "name": "even_spacing_7holes",
        "strategy": "fixed",
        "target_holes": 7,
        "random_action_max": 4,
        "action_sequence": [],
        "fallback_action": 4,
    },
}


def evaluate(cfg: DictConfig) -> Path:
    """Evaluate the configured policy and report mean boreholes and RMSE."""
    algorithm = str(cfg.evaluation.algorithm).lower()
    if algorithm not in AGENT_TYPES:
        raise ValueError(f"Unsupported evaluation algorithm: {algorithm}")
    evaluation_mode = str(cfg.evaluation.get("mode", "random")).lower()
    if evaluation_mode not in {"random", "one_to_one"}:
        raise ValueError(
            f"Unsupported evaluation mode: {evaluation_mode}. "
            "Choose 'random' or 'one_to_one'."
        )
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

    episode_results: list[dict[str, object]] = []
    episodes = int(cfg.evaluation.episodes)
    for episode_index in range(episodes):
        reset_options = (
            {"field_index": episode_index}
            if evaluation_mode == "one_to_one"
            else None
        )
        observation, _ = env.reset(options=reset_options)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            predict_kwargs = {
                "deterministic": bool(cfg.evaluation.deterministic)
            }
            if algorithm == "ppo":
                predict_kwargs["action_masks"] = env.action_masks()
            action, _ = model.predict(observation, **predict_kwargs)
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
        episode_results.append(
            {
                "num_holes": int(info["num_holes"]),
                "rmse": float(info["rmse"]),
                "steps": steps,
                "budget": int(info["budget"]),
                "drilled_coordinates": json.dumps(
                    [
                        [int(x), int(y)]
                        for x, y in env.drilled_columns
                    ]
                ),
            }
        )

    report_dir = save_experiment_report(
        cfg, "evaluations", algorithm, episode_results
    )
    print(f"Mean holes: {np.mean([r['num_holes'] for r in episode_results]):.3f}")
    print(f"Mean RMSE: {np.mean([r['rmse'] for r in episode_results]):.3f}")
    print(f"Evaluation report saved to {report_dir}")

    comparisons = cfg.evaluation.get("comparisons", [])
    if isinstance(comparisons, str):
        comparisons = [comparisons]
    for comparison in comparisons:
        comparison_name = str(comparison).lower().replace("-", "_").replace(" ", "_")
        try:
            baseline_config = COMPARISON_CONFIGS[comparison_name]
        except KeyError as error:
            available = ", ".join(sorted(COMPARISON_CONFIGS))
            raise ValueError(
                f"Unknown comparison '{comparison}'. Choose from: {available}."
            ) from error

        comparison_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        comparison_cfg.baseline = OmegaConf.create(
            {
                **baseline_config,
                "episodes": episodes,
                "mode": evaluation_mode,
            }
        )
        run_baseline(comparison_cfg)
    return report_dir
