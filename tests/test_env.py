from environment import DrillEnv
from observation_builder import LocalPatchObservationBuilder
import numpy as np


def test_environment_step_returns_valid_transition():
   env = DrillEnv()
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
   assert np.isfinite(info["rmse"])


def test_local_patch_builder_uses_only_drilled_data():
   field = np.zeros((10, 5, 5), dtype=np.float32)
   field[2, :, :] = 3.0
   field[5, :, :] = 7.0
   field[8, :, :] = 9.0

   builder = LocalPatchObservationBuilder(context_radius=2)
   observation = builder.build(field, position=5, drilled_columns=[3, 5, 7])

   assert observation["patch"].shape == (5, 5, 5)
   assert np.all((observation["mask"] == 0.0) | (observation["mask"] == 1.0))
   assert np.any(observation["mask"][0, :, :] == 1.0)
   assert np.any(observation["mask"][2, :, :] == 1.0)
   assert observation["patch"][2, :, :].shape == (5, 5)
   assert np.allclose(observation["patch"][2, :, :], field[5, :, :])
   assert np.allclose(observation["patch"][0, :, :], field[3, :, :])
