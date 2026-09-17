# src/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.interpolate import interp1d
from soil_field import generate_soil_field

# --------------------------------------------------
# Action space from the paper
# --------------------------------------------------
ACTIONS = [2, 4, 8, 12, 16]

# --------------------------------------------------
# Reward weights — not specified in the 2024 paper,
# filled in by us. Logged in the decision log.
# --------------------------------------------------
F1 = 1.0  # RMSE weight
F2 = 0.1  # number of holes weight


class DrillEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0], dtype=np.float32),
            high=np.array([100, 6, 5], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.field = generate_soil_field(seed=seed)
        self.position = 0
        self.drilled_columns = [0]
        return self._get_obs(), {}

    def step(self, action_idx):
        distance = ACTIONS[action_idx]
        new_position = self.position + distance

        if new_position >= self.field.shape[1]:
            # Ran off the edge of the field: episode ends, but no
            # drill happens here — do NOT count this as a hole.
            self.position = self.field.shape[1] - 1
            terminated = True
        else:
            self.position = new_position
            terminated = False
            if self.position not in self.drilled_columns:
                self.drilled_columns.append(self.position)

        rmse = self._compute_rmse()
        num_holes = len(self.drilled_columns)
        reward = -(F1 * rmse + F2 * num_holes)

        info = {"rmse": rmse, "num_holes": num_holes}

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        col = self.field[:, self.position]
        mean_ic = col.mean()
        std_ic = col.std()
        return np.array([self.position, mean_ic, std_ic], dtype=np.float32)

    def _compute_rmse(self):
        if len(self.drilled_columns) < 2:
            return 5.0

        known_x = np.array(self.drilled_columns)
        field_rows = self.field.shape[0]
        field_cols = self.field.shape[1]
        interpolated = np.zeros_like(self.field, dtype=np.float64)

        for row in range(field_rows):
            known_y = self.field[row, known_x]
            f = interp1d(
                known_x,
                known_y,
                kind="linear",
                bounds_error=False,
                fill_value=(known_y[0], known_y[-1]),
            )
            interpolated[row, :] = f(np.arange(field_cols))

        rmse = np.sqrt(np.mean((interpolated - self.field) ** 2))
        return float(rmse)