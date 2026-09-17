import numpy as np

from stable_baselines3 import PPO

from environment import DrillEnv

env = DrillEnv()

model = PPO.load(
    "ppo_drill_baseline_f1_2"
)

all_holes = []
all_rmse = []

for episode in range(20):

    obs, info = env.reset()

    terminated = False

    while not terminated:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

    all_holes.append(
        info["num_holes"]
    )

    all_rmse.append(
        info["rmse"]
    )

print(
    "Mean holes:",
    np.mean(all_holes)
)

print(
    "Mean RMSE:",
    np.mean(all_rmse)
)
