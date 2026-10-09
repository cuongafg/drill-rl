# Drill-RL

## Configuration

Runtime settings are managed with Hydra YAML files in `conf/`. The root
configuration composes separate groups for the site grid, geology, Gym
environment, training, evaluation, field pools, and visualization.

Generate and visualize a continuous qc ground-truth field:

```bash
uv run python src/generate_environment.py
```

The PNG preview is saved to
`environment_visualisation/realistic_qc_ground_truth.png`; the generated
NumPy field remains at `data/realistic_qc_ground_truth.npz`.
Other PNG previews are also stored in `environment_visualisation/`.

Generate a GemPy field pool (2D remains the legacy default):

```bash
uv run python src/generate_field_pool.py
uv run python src/generate_field_pool.py field_pool.mode=3d
```

Hydra overrides can change settings without editing YAML, for example:

```bash
uv run python src/generate_environment.py seed=7 grid.nz=41 geology.lens_count=32
uv run python src/train_ppo.py training.total_timesteps=50000 training.parameters.learning_rate=0.0001
```

Configuration groups:

- `conf/grid/`: site dimensions and physical coordinate limits
- `conf/geology/`: continuous qc ranges, spatial correlations, lenses, and rock
- `conf/environment/`: actions, observations, reward weights, and field-pool paths
- `conf/training/`: PPO, A2C, and DQN settings and algorithm parameters
- `conf/evaluation/` and `conf/baseline/`: evaluation and baseline settings
- `conf/gempy/`: GemPy geometry and generated input-data settings
- `conf/visualization/`: plot scales, sampling, camera, and image dimensions

## Tests

Python tests are kept in `tests/`. Install the project and its development
dependencies, including pytest, into the project virtual environment with:

```bash
uv sync --dev
```

Run the test suite with:

```bash
uv run pytest
```

To add pytest manually to an existing uv project, run `uv add --dev pytest`.

## Model checkpoints

Training saves each Stable-Baselines3 checkpoint in a unique directory such as
`local_models/ppo/20261009T155400.123456Z-a1b2c3d4/`. The directory contains
the configured checkpoint name (for example, `ppo_drill_baseline_200k.zip`),
the resolved `config.yaml`, and a `metadata.json` with its run ID and algorithm.
Use `evaluation.model_run_id` and `evaluation.model_name` in the selected
`conf/evaluation/*.yaml` file to choose a checkpoint.

New training runs and evaluation use `local_models/`, which is ignored by Git.
Selected checkpoints in `shared_models/` are tracked for sharing. To evaluate a
shared checkpoint, copy its run folder into the matching location under
`local_models/` (for example,
`cp -R shared_models/ppo/ppo-drill-baseline-200k local_models/ppo/`), then set
the matching `evaluation.model_run_id` and `evaluation.model_name` values.
Share additional selected checkpoints by adding them under `shared_models/` and
committing them. For larger collections, use Git LFS or a shared artifact store
instead of committing every run.

## Documentation and reports

- `doc/` contains research-question analysis and experiment methodology.
- `reports/` contains recorded results and baseline comparisons. Running a
  policy evaluation or baseline also creates a unique run folder under
  `reports/experiments/{evaluations,baselines}/<name>/`, containing
  `episodes.csv` and `summary.json` (including the resolved configuration).
- `src/evaluation.py` writes policy evaluation results, while
  `src/baseline_runner.py` writes baseline results. Training checkpoints remain
  in the configured model directory.

Hydra is a declared project dependency. The local `.venv` is intentionally
ignored by Git.