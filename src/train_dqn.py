from stable_baselines3 import DQN

from environment import DrillEnv

env = DrillEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=200000
)

model.save(
    "DQN_baseline"
)
