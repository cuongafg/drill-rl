from environment import DrillEnv
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
