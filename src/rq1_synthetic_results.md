RQ1 Results on Synthetic Data

Method              Mean Holes      Mean RMSE
------------------------------------------------
Random               12.66           0.702
Even Spacing          9.00           0.711
PPO 200k              7.00           0.834

Observations:

- PPO used the fewest boreholes.
- Even spacing and Random achieved lower RMSE.
- PPO learned a trade-off between drilling cost and reconstruction accuracy.
