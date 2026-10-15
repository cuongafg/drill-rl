# Drill-RL

Drill-RL is a Gymnasium/Stable-Baselines3 project for comparing borehole
placement policies on generated geological fields. Hydra YAML files configure
the site, geology, environment, training, evaluation, and output locations.

This README explains how the main pieces connect, where to change settings,
how to run the workflows, and what each source/configuration file does.

## Project at a glance

```text
conf/ --------------------------- Hydra settings used by the workflows
  |
  +-- generate_environment.py -> PropertyField3D -> qc ground truth (.npz + PNG)
  |
  +-- generate_field_pool.py -> soil_field -> GemPy (sim3D) -> training/eval pools (.npy)
  |
  +-- train_*.py -> training.py -> environment.DrillEnv -> local_models/<algorithm>/<run-id>/
  |
  +-- evaluate_*.py -> evaluation.py -> environment_eval.DrillEnv
  |                                      -> reports/experiments/evaluations/
  |
  +-- *_baseline.py -> baseline_runner.py -> environment.DrillEnv
                                         -> reports/experiments/baselines/
```

There are two geology-generation paths:

1. **Continuous qc visualization/ground truth:** `generate_environment.py`
   uses `PropertyField3D` to generate a continuous qc volume and preview image.
2. **RL environment fields:** `generate_field_pool.py` uses GemPy to create
   arrays saved as NumPy pools. The training and evaluation environments load
   these pools. The continuous qc `.npz` is not the pool loaded by the RL
   environments.

## Quick start

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
# Install runtime and development dependencies, including pytest.
uv sync --dev

# Generate the continuous qc ground-truth volume and its preview.
uv run python src/generate_environment.py

# Generate the default 2D GemPy training pool.
uv run python src/generate_field_pool.py

# Run the tests.
uv run pytest
```

The RL environments need generated field-pool files before training, evaluation,
or baseline runs. The default training pool is `data/field_pool_train.npy`; the
default evaluation pool is `data/field_pool_eval.npy`. To generate the
evaluation pool separately, for example:

```bash
uv run python src/generate_field_pool.py \
  field_pool.output=data/field_pool_eval.npy \
  field_pool.seed_start=100
```

Pool generation defaults to the legacy 2D mode. To generate a 3D training pool:

```bash
uv run python src/generate_field_pool.py field_pool.mode=3d
```

The corresponding 3D pool is written to `data/field_pool_train_3d.npy` by
default. Select the pool used by the environments with the settings in
`conf/environment/default.yaml`.

## Typical workflows

### Generate and inspect continuous qc geology

```bash
uv run python src/generate_environment.py
```

This writes the field to `data/realistic_qc_ground_truth.npz` and the image to
`environment_visualisation/realistic_qc_ground_truth.png`. The generator,
field implementation, and output paths are configured in:

### Make a drilled-hole preview with custom coordinates

```bash
uv run python src/generate_drilled_holes_preview.py
```

This uses the Hydra config in `conf/environment/drilled_holes/default.yaml` and
saves the result to `environment_visualisation/drilled_holes_preview.png` by
default. To override the hole positions from the command line:

```bash
uv run python src/generate_drilled_holes_preview.py \
  environment/drilled_holes.positions='[[10.0, 20.0], [32.0, 55.0], [70.0, 82.0], [90.0, 40.0]]' \
  environment/drilled_holes.z=0.0 \
  environment/drilled_holes.output=environment_visualisation/my_holes.png
```

### Make a 3D drilled-hole volume preview

```bash
uv run python src/generate_drilled_holes_3d_preview.py
```

This creates a 3D field preview with vertical drill markers through the full
volume and saves it to `environment_visualisation/drilled_holes_3d_preview.png`
by default. You can override the hole coordinates and output path with Hydra:

```bash
uv run python src/generate_drilled_holes_3d_preview.py \
  visualization.drilled_holes.positions='[[10.0, 20.0], [32.0, 55.0], [70.0, 82.0], [90.0, 40.0]]' \
  visualization.drilled_holes.output='environment_visualisation/my_3d_holes.png'
```

The same script can also be used to generate a deeper slice or a different set
of illustrative borehole locations without editing the Python code.

- `src/generate_environment.py` — command-line entry point.
- `src/property_field.py` — geological field generation, sampling,
  serialization, and visualization.
- `src/grid3d.py` — physical grid coordinates and index conversions.
- `conf/geology/realistic_qc.yaml`, `conf/grid/site.yaml`,
  `conf/visualization/default.yaml`, and `conf/paths/default.yaml` — parameters
  and output locations.

Example Hydra overrides:

```bash
uv run python src/generate_environment.py seed=7 grid.nz=41 geology.lens_count=32
```

### Train a reinforcement-learning agent

```bash
uv run python src/train_ppo.py
uv run python src/train_a2c.py
uv run python src/train_dqn.py
```

Each small `train_*.py` script selects its Hydra config and calls the shared
implementation in `src/training.py`. That module constructs a Stable-Baselines3
agent, trains it on `environment.DrillEnv`, then saves:

```text
local_models/<algorithm>/<run-id>/
  <configured-model-name>.zip
  config.yaml
  metadata.json
```

Run IDs are unique UTC timestamp identifiers. The checkpoint keeps the
configured filename, while the saved config and metadata record the run ID and
training settings. New training runs go to `local_models/`, which is ignored by
Git.

Training settings are in `conf/training/ppo.yaml`, `a2c.yaml`, and `dqn.yaml`.
For example, to shorten a PPO run or change its learning rate:

```bash
uv run python src/train_ppo.py \
  training.total_timesteps=50000 \
  training.parameters.learning_rate=0.0001
```

`src/train_all_agents.py` trains the algorithms listed in
`conf/orchestration/default.yaml`.

### Evaluate a saved model

```bash
uv run python src/evaluate_ppo.py
uv run python src/evaluate_a2c.py
uv run python src/evaluate_dqn.py
```

The wrapper selects an evaluation config and calls `src/evaluation.py`. The
evaluator loads the model from `local_models/<algorithm>/<model_run_id>/`,
runs the configured episodes against the evaluation field pool, and writes
results under:

```text
reports/experiments/evaluations/<algorithm>/<UTC-run-time>/
  episodes.csv
  summary.json
```

Choose the model in the corresponding `conf/evaluation/{ppo,a2c,dqn}.yaml`
file by setting both `model_run_id` and `model_name`. The configured
evaluation run IDs currently refer to the selected checkpoints in
`shared_models/`. To evaluate one of those checkpoints locally, copy its run
folder into the same relative location under `local_models/`. For example:

```bash
mkdir -p local_models/ppo
cp -R shared_models/ppo/ppo-drill-baseline-200k local_models/ppo/
```

You can also select another run without editing YAML:

```bash
uv run python src/evaluate_ppo.py \
  evaluation.model_run_id=<run-id> \
  evaluation.model_name=<checkpoint-filename-without-.zip>
```

`src/run_all_evaluations.py` runs the algorithms and baseline scenarios listed
in `conf/orchestration/default.yaml`.

### Run a baseline

```bash
uv run python src/random_baseline.py
uv run python src/even_spacing_baseline.py
uv run python src/random_baseline_7holes.py
uv run python src/even_spacing_7holes.py
```

Each wrapper selects a baseline config and calls `src/baseline_runner.py`.
Baseline results use the training field pool and are written under
`reports/experiments/baselines/<scenario>/<UTC-run-time>/`, with one row per
episode in `episodes.csv` and aggregate metrics/configuration in
`summary.json`.

## How a drilling episode works

`src/environment.py` implements the Gymnasium environment used for training
and baselines. `src/environment_eval.py` provides the evaluation environment,
which uses the evaluation pool. Both use the same action, observation, and
reward settings in `conf/environment/default.yaml`.

- The environment samples a geological field from its configured pool.
- An action selects a configured movement distance along the drilling line.
- The observation contains position and summary statistics for the current
  field slice/column.
- The environment interpolates between drilled positions and calculates RMSE
  against the complete field. It combines RMSE and borehole count into reward.
- Episode `info` includes RMSE and the number of drilled holes.

The complete field is available internally to calculate reward/metrics. The
agent receives the configured observation, not the entire field array.

## Configuration guide

`conf/config.yaml` is the root Hydra config. It composes configuration groups,
sets the default random seed, and keeps Hydra from changing the working
directory. Entry-point configs such as `train_ppo.yaml` compose the root config
and select a particular training, evaluation, or baseline group. Values can be
overridden on the command line using `group.key=value`.

| Configuration path | Responsibility |
| --- | --- |
| `conf/grid/site.yaml` | Grid dimensions and world-coordinate bounds; the surface is z=0 and depth is negative. |
| `conf/geology/realistic_qc.yaml` | Continuous qc value range, spatial correlations, lenses, surface, and bedrock characteristics. |
| `conf/environment/default.yaml` | Discrete drilling distances, observations, reward weights, starting position, and training/evaluation pool paths. |
| `conf/field_pool/default.yaml` | GemPy pool size, mode, seed range, retry count, and output files. |
| `conf/gempy/default.yaml` | GemPy extent, formations, input point counts, and legacy 2D resolution. |
| `conf/paths/default.yaml` | Central locations for generated data, model checkpoints, reports, and visualizations. |
| `conf/visualization/default.yaml` | Plot size, colormaps, sampling, camera angle, and rendering options. |
| `conf/training/{ppo,a2c,dqn}.yaml` | Algorithm, policy, timesteps, checkpoint filename, and algorithm parameters. |
| `conf/evaluation/{ppo,a2c,dqn}.yaml` | Selected algorithm/checkpoint, episode count, and deterministic prediction setting. |
| `conf/evaluation/default.yaml` | Legacy/default evaluation names; individual evaluation entry points select algorithm-specific files. |
| `conf/baseline/{random,even_spacing,random_7holes,even_spacing_7holes}.yaml` | Episode count, strategy, target holes, action sequence, and fallback action for each baseline. |
| `conf/baseline/default.yaml` | Baseline fallback settings. |
| `conf/orchestration/default.yaml` | Algorithms/scenarios included in combined train/evaluation commands. |
| `conf/train_{ppo,a2c,dqn}.yaml` | Root config plus selection of one training group. |
| `conf/evaluate_{ppo,a2c,dqn}.yaml` | Root config plus selection of one evaluation group. |
| `conf/run_baseline_{random,even_spacing,random_7holes,even_spacing_7holes}.yaml` | Root config plus selection of one baseline scenario. |

## Source code map

### Core models and shared workflow code

| File | Responsibility |
| --- | --- |
| `src/configuration.py` | Resolves project/config paths and loads YAML sections for both Hydra and non-Hydra callers. |
| `src/grid3d.py` | Defines the physical grid, world/index coordinate conversion, and volume allocation. |
| `src/property_field.py` | Generates continuous 3D qc geology, samples boreholes, saves/loads fields, and creates plots. |
| `src/environment.py` | Training and baseline Gymnasium drilling environment; reads the training pool. |
| `src/environment_eval.py` | Evaluation Gymnasium environment; reads the evaluation pool. |
| `src/training.py` | Shared PPO/A2C/DQN construction, training, unique run IDs, and checkpoint/config/metadata saving. |
| `src/evaluation.py` | Shared PPO/A2C/DQN model loading, episode evaluation, and metric reporting. |
| `src/baseline_runner.py` | Executes configured random/fixed/sequence baselines and records episode metrics. |
| `src/model_artifacts.py` | Generates run IDs, resolves algorithm/run directories, and validates checkpoint names. |
| `src/experiment_reports.py` | Writes per-episode CSV and aggregate JSON experiment reports. |
| `src/soil_field.py` | Generates seeded GemPy 2D or 3D fields and retries failed GemPy solves. |
| `src/sim3D.py` | Builds/computes GemPy models, extracts 2D/3D fields, and can save a 3D preview. |
| `src/sim2D.py` | Compatibility re-export of the simulation functions in `sim3D.py`. |
| `src/generate_geology_csvs.py` | Writes randomized GemPy orientation and surface-point CSV inputs. |

### Command-line entry points and adapters

| File(s) | Responsibility |
| --- | --- |
| `src/generate_environment.py` | Generates and saves continuous qc ground truth and its preview. |
| `src/generate_field_pool.py` | Generates and saves a stack of GemPy fields as a NumPy pool. |
| `src/generate_orientations.py` | Generates the orientation CSV used by GemPy. |
| `src/generate_surface_points.py` | Generates the surface-point CSV used by GemPy. |
| `src/train_ppo.py`, `src/train_a2c.py`, `src/train_dqn.py` | Hydra entry points for training one algorithm; delegate to `training.py`. |
| `src/train_all_agents.py` | Trains each algorithm configured for orchestration. |
| `src/evaluate_ppo.py`, `src/evaluate_a2c.py`, `src/evaluate_dqn.py` | Hydra entry points for evaluating one algorithm; delegate to `evaluation.py`. |
| `src/run_all_evaluations.py` | Runs all configured policy evaluations and baselines. |
| `src/random_baseline.py`, `src/even_spacing_baseline.py`, `src/random_baseline_7holes.py`, `src/even_spacing_7holes.py` | Hydra entry points for individual baseline scenarios; delegate to `baseline_runner.py`. |
| `src/sim2D_backup.py` | Older standalone simulation implementation kept as a backup; it is not used by the active configured pipeline. |
| `src/drill_rl/__init__.py` | Current placeholder for the installed `drill-rl` console command; it only prints a greeting. Use the `src/*.py` commands above for project workflows. |

## Data, models, and reports

| Folder | Contents and Git behavior |
| --- | --- |
| `data/` | Generated GemPy pools (`.npy`), continuous qc ground truth (`.npz`), and GemPy CSV inputs. Generated files are ignored by Git. |
| `environment_visualisation/` | Generated PNG previews. PNGs are ignored by Git. |
| `local_models/` | Local training outputs and local evaluation copies. Ignored by Git. |
| `shared_models/` | Selected model checkpoint ZIP files retained in Git for sharing. New files must be added and committed. |
| `reports/` | Hand-maintained Markdown reports and generated evaluation/baseline run output. The entire folder is ignored by Git. |
| `doc/` | Analysis and methodology Markdown documents. |
| `tests/` | Automated Python tests. |

Current Markdown documents include:

- `doc/rq1_analysis.md` and `doc/rql_analysis.md` — research-question analysis.
- `reports/rq1_synthetic_results.md`, `reports/results_summary.md`,
  `reports/baseline_comparison.md`, and `reports/baseline_results.md` —
  manually maintained experiment and baseline results.
- `reports/rq1_analysis.md.save` — editor backup of an analysis document, not
  used by the application.

Because all of `reports/` is Git-ignored, files there remain local unless they
are deliberately force-added. `doc/` is the location for analysis documents
that should be versioned with the source.

An evaluation or baseline run creates a timestamped folder with:

- `episodes.csv` — metrics from each episode.
- `summary.json` — means, standard deviations, seed, and resolved configuration.

The scripts print mean holes/RMSE to the terminal as well. Training does not
write experiment reports; it writes its model checkpoint and run metadata.

## Tests

Tests live in `tests/`; pytest is a development dependency.

```bash
uv sync --dev
uv run pytest
```

For verbose per-test results or captured `print()` output:

```bash
uv run pytest -v
uv run pytest -v -s
```

The pytest configuration in `pyproject.toml` points test discovery to `tests/`
and adds `src/` to the import path.

| Test file | Coverage |
| --- | --- |
| `tests/test_check_env.py` | Gymnasium environment API contract. |
| `tests/test_configuration.py` | Hydra group composition, CLI overrides, and configured environment spaces. |
| `tests/test_env.py` | A representative environment step and returned metrics. |
| `tests/test_experiment_reports.py` | Report CSV/JSON content and unique report directories. |
| `tests/test_grid3d.py` | Grid extents, coordinate conversions, and volume dimensions. |
| `tests/test_model_artifacts.py` | Model run IDs, safe paths, and checkpoint filename validation. |
| `tests/test_property_field.py` | qc field bounds/geology, sampling, repeatability, persistence, and visualization. |
| `tests/test_seed.py` | Different seeds produce different continuous qc fields. |
| `tests/test_speed.py` | Environment reset returns a valid observation (despite the legacy filename, it is not a benchmark). |

## Project and dependency files

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Python version, runtime dependencies, pytest settings, dev dependencies, and build metadata. |
| `uv.lock` | Exact resolved dependency versions for reproducible `uv sync` installs. |
| `.gitignore` | Excludes virtual environments, caches, generated data/images, local models, and reports. |
