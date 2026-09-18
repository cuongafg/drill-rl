import subprocess

scripts = [
    "train_ppo.py",
    "train_a2c.py",
    "train_dqn.py",
]

for script in scripts:
    print("\n" + "=" * 60)
    print(f"Running {script}")
    print("=" * 60)

    subprocess.run(
        ["python", script],
        check=True
    )
