
import numpy as np

from stable_baselines3 import A2C

from environment import DrillEnv

env = DrillEnv()

model = A2C.load(
    "a2c_baseline"
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
