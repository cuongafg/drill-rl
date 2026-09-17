from stable_baselines3 import PPO

from environment import DrillEnv

env = DrillEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=200000
)

model.save(
    "ppo_drill_baseline_f1_2"
)
