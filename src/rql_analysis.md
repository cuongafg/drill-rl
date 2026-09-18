Method              Holes      RMSE
-------------------------------------
Random              12.66      0.702
Even Spacing         9.00      0.711
PPO                  7.00      0.834
A2C                  7.00      0.896
DQN                  7.00      0.933


Among DRL agents,
PPO performs best.

PPO requires the fewest boreholes.

However,
Random and Even Spacing achieve lower RMSE because they place substantially more boreholes.

This indicates that PPO learns a stronger trade-off between cost and prediction accuracy.
