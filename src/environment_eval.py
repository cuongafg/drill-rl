# src/environment_eval.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.interpolate import interp1d

from configuration import as_dict, load_config_section, project_path

from environment import _load_pool


class DrillEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.config = load_config_section("environment") if config is None else as_dict(config)
        self.actions = [int(action) for action in self.config["actions"]]
        self.rmse_weight = float(self.config["reward"]["rmse_weight"])
        self.borehole_weight = float(self.config["reward"]["borehole_weight"])
        self.initial_rmse = float(self.config["reward"]["initial_rmse"])
        self.initial_position = int(self.config["initial_position"])
        self.pool_paths = [project_path(self.config["eval_pool"])]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=np.asarray(self.config["observation"]["low"], dtype=np.float32),
            high=np.asarray(self.config["observation"]["high"], dtype=np.float32),
            dtype=np.float32,
        )
        self.field = None
        self.position = 0
        self.drilled_columns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pool = _load_pool(self.pool_paths)

        idx = self.np_random.integers(0, len(pool))
        self.field = np.asarray(pool[idx])
        self.position = self.initial_position
        self.drilled_columns = [self.initial_position]
        return self._get_obs(), {}

    def step(self, action_idx):
        distance = self.actions[int(action_idx)]
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
        reward = -(self.rmse_weight * rmse + self.borehole_weight * num_holes)

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
            return self.initial_rmse

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
