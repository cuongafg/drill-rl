import numpy as np
from environment import DrillEnv


def test_environment_reset_returns_valid_observation(tmp_path):
    pool_path = tmp_path / "pool.npy"
    np.save(pool_path, np.ones((1, 6, 4, 3), dtype=np.int8))
    env = DrillEnv({
        "reward": {"rmse_weight": 1.0, "borehole_weight": 0.1, "initial_rmse": 5.0},
        "pool": str(pool_path),
        "initial_position": 0,
        "observation": {
            "mode": "full_observed_map",
            "representation": "mlp",
            "field_shape": [6, 4, 3],
            "include_position": True,
            "unknown_value": 0.0,
            "value_low": 1.0,
            "value_high": 5.0,
        },
    })

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)
