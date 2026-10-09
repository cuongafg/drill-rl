# Drill-RL Project

Author: Cuong Dinh

Branch:

release/christian-handoff

---

# Project Goal

Investigate whether Reinforcement Learning can learn better borehole placement strategies than simple drilling heuristics.

Pipeline:

GemPy
→ Geological Field Generation
→ Field Pools
→ Gym Environment
→ PPO / A2C / DQN
→ Borehole Placement
→ RMSE Evaluation

---

# Quick Start

If you only want to verify that everything works:

```bash
git clone https://github.com/cuongafg/drill-rl.git

cd drill-rl

git checkout release/christian-handoff

pip install \
gempy \
gempy_viewer \
gymnasium \
stable-baselines3 \
numpy \
scipy \
matplotlib \
pandas

cd src

python run_all_evaluations.py
```

This uses the pre-trained models already included in the repository.

No retraining is required.

---

# Full Reproduction

If you want to reproduce the study from scratch:

```bash
cd src

python train_all_agents.py
```

This will train:

- PPO
- A2C
- DQN

Generated models:

```text
ppo_drill_baseline_200k.zip
a2c_baseline.zip
DQN_baseline.zip
```

After training:

```bash
python run_all_evaluations.py
```

---

# Project Structure

## Environment

```text
environment.py
```

Training environment.

Uses:

```text
field_pool_train.npy
```

---

```text
environment_eval.py
```

Evaluation environment.

Uses:

```text
field_pool_eval.npy
```

---

## Training

```text
train_ppo.py
train_a2c.py
train_dqn.py

train_all_agents.py
```

---

## Evaluation

```text
evaluate_ppo.py
evaluate_a2c.py
evaluate_dqn.py

run_all_evaluations.py
```

---

## Baselines

```text
random_baseline.py
even_spacing_baseline.py

random_baseline_7holes.py
even_spacing_7holes.py
```

---

# Experimental Configuration

Training Pool:

```text
20 synthetic fields
```

Evaluation Pool:

```text
100 unseen synthetic fields
```

Reward:

```python
reward = -(F1 * RMSE + F2 * num_holes)
```

Parameters:

```text
F1 = 1.0
F2 = 0.1
```

---

# Interpretation of Results

Two kinds of comparisons are reported:

## Original comparison

Random and Even Spacing are allowed to drill more boreholes.

This usually gives lower RMSE but requires higher drilling cost.

---

## Fair comparison

All methods are constrained to:

```text
7 boreholes
```

This comparison is more meaningful because drilling cost is fixed.

In our experiments:

- All DRL agents outperform Random drilling.
- All DRL agents outperform Evenly Spaced drilling.
- PPO, A2C and DQN achieve similar performance.

---

# Notes

Evaluation fields are sampled from the evaluation pool.

Because of this:

- RMSE values may vary slightly between runs.
- Small RMSE differences between PPO, A2C and DQN should not be overinterpreted.

The main conclusion is:

DRL agents learn more efficient borehole placement strategies than simple baselines when drilling budget is fixed.

---

# Current Status

Completed:

- GemPy field generation
- Training pool
- Evaluation pool
- PPO baseline
- A2C baseline
- DQN baseline
- Random baseline
- Even-spacing baseline
- Synthetic-data experiments (RQ1)

Remaining:

- CNN environment
- CNN agents
- MLP vs CNN comparison
- Premstaller CPT dataset
- Real-data experiments
