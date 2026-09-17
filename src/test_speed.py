import time
from environment import DrillEnv

env = DrillEnv()

t0 = time.time()

obs, info = env.reset()

print("obs =", obs)

print(
    "reset time:",
    time.time() - t0
)
