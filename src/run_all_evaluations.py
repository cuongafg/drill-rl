import subprocess

scripts = [
    "evaluate_ppo.py",
    "evaluate_a2c.py",
    "evaluate_dqn.py",
    "random_baseline.py",
    "even_spacing_baseline.py",
    "random_baseline_7holes.py",
    "even_spacing_7holes.py",
]

for script in scripts:
    print("\n" + "=" * 60)
    print(f"Running {script}")
    print("=" * 60)

    subprocess.run(
        ["python", script],
        check=True
    )
