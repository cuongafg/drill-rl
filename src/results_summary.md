Method              Holes      RMSE
------------------------------------
Random              12.66      0.702
Even Spacing         9.00      0.711
PPO (F1=1)           7.00      0.834
PPO (F1=2)           7.00      0.947
PPO (F1=5)           7.00      0.856
We experimented with different reward weightings to investigate the trade-off between model accuracy (RMSE) and drilling cost (number of boreholes). Contrary to the initial expectation, increasing the RMSE weight from F1=1.0 to F1=2.0 and F1=5.0 did not improve performance. The best evaluation result was achieved with F1=1.0 and F2=0.1, suggesting that the original reward balance already provided a suitable trade-off for this environment.


Method              Holes      RMSE
-------------------------------------
Random              12.66      0.702
Even Spacing         9.00      0.711
PPO                  7.00      0.834
A2C                  7.00      0.896
DQN                  7.00      0.933


We force  7 holes
Method              Holes      RMSE
-------------------------------------
PPO                  7.00      0.834
Even Spacing         7.00      0.873
PPO                  7.00      1.506


On synthetic GemPy-generated soil fields, PPO achieved the best performance among the evaluated DRL methods (PPO, A2C, DQN), obtaining an average of 7 boreholes with RMSE 0.834. While Random and Even Spacing achieved lower RMSE when allowed to use more boreholes, a fair comparison at the same borehole budget (7 boreholes) showed that PPO outperformed both Random (RMSE 1.506) and Even Spacing (RMSE 0.873). This indicates that PPO learns more effective borehole placement strategies than simple baselines under equal cost constraints.
