import numpy as np

from environment import DrillEnv

N_EPISODES = 100

env = DrillEnv()

all_holes = []
all_rmse = []

# khoảng cách gần tương ứng:
# 0->14->28->42...
ACTION_SEQUENCE = [3, 3, 3, 3, 3, 3, 3]

for episode in range(N_EPISODES):

    obs, info = env.reset()

    terminated = False

    i = 0

    while not terminated:

        if i < len(ACTION_SEQUENCE):
            action = ACTION_SEQUENCE[i]
        else:
            action = 3

        i += 1

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
print("=== Even Spacing Baseline ===")

print(
    "Mean holes:",
    np.mean(all_holes)
)

print(
    "Mean RMSE:",
    np.mean(all_rmse)
)
