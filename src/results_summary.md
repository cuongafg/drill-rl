# Results Summary

## Project

Reinforcement Learning for Borehole Placement on Synthetic Geological Fields

### Experimental Configuration

Training Pool:

- field_pool_train.npy
- 20 GemPy-generated geological fields

Evaluation Pool:

- field_pool_eval.npy
- 100 unseen GemPy-generated geological fields

Reward Function:

reward = -(F1 * RMSE + F2 * num_holes)

Parameters:

- F1 = 1.0
- F2 = 0.1

Metrics:

- Mean Holes = average number of drilled boreholes
- Mean RMSE = reconstruction error

Lower RMSE is better.

Lower number of holes is better.

---

# DRL Agents

| Method | Mean Holes | Mean RMSE |
|----------|----------:|----------:|
| PPO | 7.00 | 0.908 |
| A2C | 7.00 | 0.923 |
| DQN | 7.00 | 0.896 |

## Observation

- All DRL agents converged to approximately 7 boreholes.
- DQN achieved the lowest RMSE among the evaluated DRL agents.
- PPO and A2C achieved similar drilling budgets but higher RMSE values.

---

# Original Baseline Comparison

| Method | Mean Holes | Mean RMSE |
|----------|----------:|----------:|
| Random | 12.64 | 0.779 |
| Even Spacing | 9.00 | 0.775 |
| PPO | 7.00 | 0.908 |
| A2C | 7.00 | 0.923 |
| DQN | 7.00 | 0.896 |

## Observation

Random and Even Spacing achieve lower RMSE values because they use substantially more boreholes than the DRL agents.

Therefore, this comparison is not completely fair in terms of drilling budget.

---

# Fair Comparison (Same Borehole Budget)

To compare placement strategies fairly, Random and Even Spacing were constrained to the same drilling budget used by the DRL agents.

Borehole Budget:

- 7 holes

| Method | Mean Holes | Mean RMSE |
|----------|----------:|----------:|
| DQN | 7.00 | 0.896 |
| PPO | 7.00 | 0.908 |
| A2C | 7.00 | 0.923 |
| Even Spacing (7 holes) | 7.00 | 0.991 |
| Random (7 holes) | 7.00 | 1.622 |

## Observation

Under the same drilling budget:

- All DRL methods outperform Random drilling.
- All DRL methods outperform Evenly Spaced drilling.
- DQN achieves the lowest RMSE among all methods when constrained to 7 boreholes.

---

# Key Takeaway

When allowed to use more boreholes, Random and Even Spacing can achieve lower reconstruction errors.

However, when all methods are constrained to the same drilling budget of seven boreholes, DRL agents achieve substantially better reconstruction performance.

This indicates that the DRL agents learn more effective borehole placement strategies rather than simply benefiting from additional drilling information.

---

# Official Models

Main Models:

- ppo_drill_baseline_200k.zip
- a2c_baseline.zip
- DQN_baseline.zip

Additional Reward-Tuning Experiments:

- ppo_drill_baseline_f1_2.zip
- ppo_drill_baseline_f1_5.zip
