# src/environment.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import interp1d

from soil_field import generate_soil_field

# --- Actions: distance (in columns) to the next hole, rightward only ---
# Matches the 2024 paper's Case 1 (MLP model) action set exactly.
ACTIONS = [2, 4, 8, 12, 16]

# --- Reward weights ---
# NOT specified anywhere in the 2024 paper — this is a filled-in gap.
# Log this choice in the shared decision log.
F1 = 1.0   # weight on RMSE
F2 = 0.1   # weight on number of holes

FIELD_ROWS = 10     # depth
FIELD_COLS = 100    # distance


class DrillEnv(gym.Env):
    """
    State:  [position, mean_IC, std_IC] of the most recently drilled column.
    Action: index into ACTIONS — distance to the next hole.
    Reward: -(F2 * num_holes + F1 * rmse), given after each drill.
    Episode ends when position moves past the last column.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            high=np.array([FIELD_COLS, 6.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.field = None
        self.position = 0
        self.drilled_columns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.field = generate_soil_field(seed=seed)  # shape (10, 100)

        # ASSUMPTION: the episode starts with a hole already drilled at
        # position 0 (the paper isn't explicit about the very first test).
        # Logged as a decision-log entry.
        self.position = 0
        self.drilled_columns = [0]

        return self._get_obs(), {}

    def step(self, action_idx):
        distance = ACTIONS[action_idx]
        self.position += distance

        terminated = self.position >= FIELD_COLS

        if not terminated:
            self.drilled_columns.append(self.position)

        rmse = self._compute_rmse()
        num_holes = len(self.drilled_columns)

        # Negative because Stable-Baselines3 maximizes reward, and we
        # want to minimize both RMSE and the number of holes.
        reward = -(F2 * num_holes + F1 * rmse)

        obs = self._get_obs()
        info = {"rmse": rmse, "num_holes": num_holes}

        return obs, reward, terminated, False, info

    def _get_obs(self):
        # Use the last successfully drilled column, even if the agent's
        # position has since moved past the field boundary.
        last_col = self.drilled_columns[-1]
        column = self.field[:, last_col]
        return np.array(
            [self.position, column.mean(), column.std()], dtype=np.float32
        )

    def _compute_rmse(self):
        if len(self.drilled_columns) < 2:
            # Can't interpolate from a single point.
            # ASSUMPTION: return a large fixed penalty instead of NaN,
            # so training doesn't break on the first step. Logged.
            return 5.0

        known_x = np.array(self.drilled_columns)
        interpolated = np.zeros_like(self.field, dtype=np.float64)

        for row in range(FIELD_ROWS):
            known_y = self.field[row, known_x]
            f = interp1d(
                known_x, known_y, kind="linear", fill_value="extrapolate"
            )
            interpolated[row, :] = f(np.arange(FIELD_COLS))

        return float(np.sqrt(np.mean((interpolated - self.field) ** 2)))
