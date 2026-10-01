# src/environment_eval.py
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.interpolate import interp1d


FIELD_POOL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "field_pool_eval.npy"
)

_FIELD_POOL = None


def _load_pool():
    global _FIELD_POOL

    if _FIELD_POOL is None:
        _FIELD_POOL = np.load(FIELD_POOL_PATH)

    return _FIELD_POOL

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
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([100, 6, 5], dtype=np.float32),
            dtype=np.float32,
        )
        self.field = None
        self.position = 0
        self.drilled_columns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pool = _load_pool()

        idx = self.np_random.integers(0, len(pool))
        self.field = np.asarray(pool[idx])
        self.position = 0
        self.drilled_columns = [0]
        return self._get_obs(), {}

    def step(self, action_idx):
        distance = ACTIONS[action_idx]
        max_position = self.field.shape[0] - 1 if self.field.ndim == 3 else self.field.shape[1] - 1
        new_position = self.position + distance

        if new_position >= max_position:
            self.position = max_position
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
        if self.field.ndim == 3:
            slice_2d = self.field[self.position, :, :]
            mean_ic = float(slice_2d.mean())
            std_ic = float(slice_2d.std())
            return np.array([self.position, mean_ic, std_ic], dtype=np.float32)

        col = self.field[:, self.position]
        mean_ic = float(col.mean())
        std_ic = float(col.std())
        return np.array([self.position, mean_ic, std_ic], dtype=np.float32)

    def _compute_rmse(self):
        if len(self.drilled_columns) < 2:
            return 5.0

        known_x = np.asarray(self.drilled_columns, dtype=np.int64)

        if self.field.ndim == 3:
            x_size = self.field.shape[0]
            interpolated = np.zeros_like(self.field, dtype=np.float64)
            for y in range(self.field.shape[1]):
                for z in range(self.field.shape[2]):
                    known_y = self.field[known_x, y, z]
                    if known_y.size == 0:
                        continue
                    f = interp1d(
                        known_x,
                        known_y,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(known_y[0], known_y[-1]),
                    )
                    interpolated[:, y, z] = f(np.arange(x_size))
            rmse = np.sqrt(np.mean((interpolated - self.field) ** 2))
            return float(rmse)

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
