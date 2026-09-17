# src/test_env.py
from environment import DrillEnv

env = DrillEnv()
obs, info = env.reset(seed=42)
print("initial obs:", obs)

terminated = False
step_num = 0
while not terminated:
    action = env.action_space.sample()  # random action for now
    obs, reward, terminated, truncated, info = env.step(action)
    step_num += 1
    print(f"step {step_num}: action={action} -> obs={obs}, reward={reward:.3f}, "
          f"rmse={info['rmse']:.3f}, num_holes={info['num_holes']}, terminated={terminated}")

print("\nEpisode finished.")
print("Total holes drilled:", info["num_holes"])
print("Final RMSE:", info["rmse"])
