from stable_baselines3 import A2C

from environment import DrillEnv

env = DrillEnv()

model = A2C(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=200000
)

model.save(
    "a2c_baseline"
)
