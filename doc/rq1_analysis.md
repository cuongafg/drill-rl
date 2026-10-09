# RQ1 Analysis

## Research Question

How do DRL agents (PPO, A2C, DQN) compare to simple baselines (Random and Evenly Spaced drilling) when placing boreholes, measured by prediction error (RMSE) and number of holes required?

---

# Experimental Setup

Synthetic geological fields were generated using GemPy.

The environment was implemented using Gymnasium.

Training:

- 20 synthetic fields
- field_pool_train.npy

Evaluation:

- 100 unseen synthetic fields
- field_pool_eval.npy

Reward Function:

reward = -(F1 * RMSE + F2 * num_holes)

where:

- F1 = 1.0
- F2 = 0.1

The reward balances reconstruction accuracy and drilling cost.

---

# Results

## Original Comparison

| Method | Mean Holes | Mean RMSE |
|----------|----------:|----------:|
| Random | 12.64 | 0.779 |
| Even Spacing | 9.00 | 0.775 |
| PPO | 7.00 | 0.908 |
| A2C | 7.00 | 0.923 |
| DQN | 7.00 | 0.896 |

At first glance, the simple baselines appear to outperform the DRL agents because they obtain lower RMSE values.

However, the baselines achieve this performance by drilling substantially more boreholes.

---

# Fair Comparison

To remove the effect of unequal drilling cost, Random and Even Spacing were evaluated again using the same borehole budget as the DRL agents.

Hole Budget:

- 7 holes

Results:

| Method | Mean Holes | Mean RMSE |
|----------|----------:|----------:|
| DQN | 7.00 | 0.896 |
| PPO | 7.00 | 0.908 |
| A2C | 7.00 | 0.923 |
| Even Spacing (7 holes) | 7.00 | 0.991 |
| Random (7 holes) | 7.00 | 1.622 |

---

# Discussion

The original comparison suggests that Random and Even Spacing produce lower RMSE values.

However, those methods achieve their performance by using significantly more boreholes.

Once all methods are constrained to the same drilling budget, the results change substantially.

All DRL agents outperform the simple baselines when using seven boreholes.

This indicates that the DRL agents learn meaningful placement strategies rather than simply relying on additional drilling information.

Among the evaluated DRL methods:

1. DQN achieved the best RMSE.
2. PPO achieved the second-best RMSE.
3. A2C achieved the third-best RMSE.

The difference between the three DRL methods is relatively small compared to their advantage over Random drilling.

---

# Preliminary Answer to RQ1

For synthetic geological data, DRL agents successfully learned efficient borehole placement strategies.

While Random and Evenly Spaced drilling can achieve lower RMSE values by drilling more holes, all evaluated DRL methods outperform these baselines when constrained to the same borehole budget.

Among the tested DRL algorithms, DQN achieved the best reconstruction accuracy, followed by PPO and A2C.

Therefore, the results suggest that DRL-based borehole placement is more efficient than simple heuristic drilling strategies under equal drilling-cost constraints.

---

# Remaining Work

RQ1 is only partially answered.

Completed:

- Synthetic data experiments
- PPO
- A2C
- DQN
- Random baseline
- Even Spacing baseline

Remaining:

- Premstaller CPT dataset
- Real-data evaluation

Final conclusions for RQ1 will be drawn after experiments on real CPT data are completed.
