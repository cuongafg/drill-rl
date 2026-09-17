import random
import numpy as np

from environment import DrillEnv

N_EPISODES = 100

env = DrillEnv()

all_holes = []
all_rmse = []

for episode in range(N_EPISODES):

    obs, info = env.reset()

    terminated = False

    while not terminated:

        action = random.randint(0, 4)

        obs, reward, terminated, truncated, info = env.step(
            action
        )

    all_holes.append(
        info["num_holes"]
    )

    all_rmse.append(
        info["rmse"]
    )

print()
print("=== Random Baseline ===")
print(
    "Mean holes:",
    np.mean(all_holes)
)

print(
    "Mean RMSE:",
    np.mean(all_rmse)
)
