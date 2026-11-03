import numpy as np
import pytest
from environment import DrillEnv
from observation_builder import LocalPatchObservationBuilder


def _environment_config(pool_path):
   return {
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
   }


def test_environment_step_returns_valid_transition(tmp_path):
   pool_path = tmp_path / "pool.npy"
   np.save(pool_path, np.ones((2, 6, 4, 3), dtype=np.int8))
   env = DrillEnv(_environment_config(pool_path))
   observation, _ = env.reset(seed=42)

   next_observation, reward, terminated, truncated, info = env.step(
       env.action_space.sample()
   )

   assert env.observation_space.contains(observation)
   assert env.observation_space.contains(next_observation)
   assert np.isfinite(reward)
   assert isinstance(terminated, bool)
   assert isinstance(truncated, bool)
   assert info["num_holes"] >= 1
   assert info["budget"] == 7
   assert np.isfinite(info["rmse"])


def test_coordinate_actions_support_backward_moves_and_reject_duplicates(tmp_path):
   pool_path = tmp_path / "pool.npy"
   np.save(pool_path, np.ones((1, 6, 4, 3), dtype=np.int8))
   env = DrillEnv(_environment_config(pool_path))
   env.reset(seed=42)

   first_action = env.coordinate_to_action((4, 1))
   _, _, _, truncated, info = env.step(first_action)
   assert not truncated
   assert env.position == (4, 1)
   assert env.drilled_columns[-1] == (4, 1)

   second_action = env.coordinate_to_action((1, 2))
   _, _, _, truncated, info = env.step(second_action)
   assert not truncated
   assert not info["invalid_action"]
   assert env.position == (1, 2)

   _, _, _, truncated, info = env.step(env.action_space.n - 1)
   assert not truncated
   assert not info["invalid_action"]
   assert len(env.drilled_columns) == 4
   assert (4, 1) not in env.available_coordinates


def test_action_mask_disables_initial_and_drilled_coordinates(tmp_path):
   pool_path = tmp_path / "pool.npy"
   np.save(pool_path, np.ones((1, 6, 4, 3), dtype=np.int8))
   env = DrillEnv(_environment_config(pool_path))
   env.reset(seed=42)

   initial_action = env.coordinate_to_action(env.initial_position)
   available_action = env.coordinate_to_action((4, 1))
   assert not env.action_masks()[initial_action]
   assert env.action_masks()[available_action]

   env.step(available_action)
   assert not env.action_masks()[available_action]


def test_mlp_observation_contains_map_and_position(tmp_path):
   pool_path = tmp_path / "pool.npy"
   np.save(pool_path, np.full((1, 6, 4, 3), 5, dtype=np.int8))
   env = DrillEnv(_environment_config(pool_path))
   observation, _ = env.reset(seed=42)

   assert observation.shape == (2 * 6 * 4 * 3 + 2 + 6 * 4,)
   assert env.observation_space.contains(observation)
   assert np.count_nonzero(observation[: 2 * 6 * 4 * 3 - 1]) == 6
   assert np.allclose(observation[2 * 6 * 4 * 3 : 2 * 6 * 4 * 3 + 2], [0.0, 2.0 / 3.0])
   assert np.count_nonzero(observation[-6 * 4 :]) == 23


def test_reset_can_select_a_specific_pool_field(tmp_path):
   pool_path = tmp_path / "pool.npy"
   pool = np.stack(
       [
           np.zeros((6, 4, 3), dtype=np.int8),
           np.ones((6, 4, 3), dtype=np.int8),
       ]
   )
   np.save(pool_path, pool)
   env = DrillEnv(_environment_config(pool_path))

   observation, _ = env.reset(options={"field_index": 1})

   assert np.array_equal(env.field, pool[1])


def test_reset_rejects_invalid_pool_field_index(tmp_path):
   pool_path = tmp_path / "pool.npy"
   np.save(pool_path, np.zeros((2, 6, 4, 3), dtype=np.int8))
   env = DrillEnv(_environment_config(pool_path))

   with pytest.raises(ValueError, match="field_index"):
       env.reset(options={"field_index": 2})


def test_local_patch_builder_uses_only_drilled_data():
   field = np.zeros((10, 5, 5), dtype=np.float32)
   field[2, :, :] = 3.0
   field[5, :, :] = 7.0
   field[8, :, :] = 9.0

   builder = LocalPatchObservationBuilder(context_radius=2)
   observation = builder.build(field, position=5, drilled_columns=[(3, 2), (5, 3), (7, 4)])

   assert observation["patch"].shape == (5, 4, 5)
   assert np.all((observation["mask"] == 0.0) | (observation["mask"] == 1.0))
   assert np.any(observation["mask"][0, :, :] == 1.0)
   assert np.any(observation["mask"][2, :, :] == 1.0)
   assert observation["patch"][0, 1, :].shape == (5,)
   assert np.allclose(observation["patch"][2, 2, :], field[5, 3, :])
   assert np.allclose(observation["patch"][0, 1, :], field[3, 2, :])
