import random
import numpy as np

from environment import DrillEnv

N_EPISODES = 100
TARGET_HOLES = 7

env = DrillEnv()

all_holes = []
all_rmse = []

for episode in range(N_EPISODES):

    obs, info = env.reset()

    terminated = False

    while not terminated:

        # dừng khi đủ 7 holes
        if len(env.drilled_columns) >= TARGET_HOLES:
            break

        action = random.randint(0, 4)

        obs, reward, terminated, truncated, info = env.step(
            action
        )

    rmse = env._compute_rmse()

    all_holes.append(
        len(env.drilled_columns)
    )

    all_rmse.append(
        rmse
    )

print()
print("=== Random Baseline (7 holes) ===")
print("Mean holes:", np.mean(all_holes))
print("Mean RMSE :", np.mean(all_rmse))
