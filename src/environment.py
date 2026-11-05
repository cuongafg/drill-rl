# src/environment.py
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from configuration import as_dict, load_config_section, project_path
from observation_builder import build_observation_builder


_FIELD_POOL_CACHE: dict[Path, np.ndarray] = {}


def _load_pool(candidate_paths: list[Path]) -> np.ndarray:
    for candidate in candidate_paths:
        if candidate in _FIELD_POOL_CACHE:
            return _FIELD_POOL_CACHE[candidate]
        if candidate.is_file():
            _FIELD_POOL_CACHE[candidate] = np.load(candidate)
            return _FIELD_POOL_CACHE[candidate]
    raise FileNotFoundError(
        "No configured field pool exists. Generate it with src/generate_field_pool.py."
    )


class DrillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        self.config = load_config_section("environment") if config is None else as_dict(config)
        self.rmse_weight = float(self.config["reward"]["rmse_weight"])
        self.borehole_weight = float(self.config["reward"]["borehole_weight"])
        self.initial_rmse = float(self.config["reward"]["initial_rmse"])
        self.use_initial_rmse_placeholder = bool(
            self.config["reward"].get("use_initial_rmse_placeholder", True)
        )
        self.max_holes = int(self.config.get("max_holes", 7))
        if self.max_holes < 1:
            raise ValueError("max_holes must be at least 1.")
        self.pool_paths = [project_path(self.config["pool"])]
        observation_cfg = self.config["observation"]
        self.field_shape = tuple(int(value) for value in observation_cfg["field_shape"])
        if len(self.field_shape) != 3:
            raise ValueError("The 3D environment requires observation.field_shape=[nx, ny, nz].")
        self.action_space = spaces.Discrete(self.field_shape[0] * self.field_shape[1])
        initial_position = self.config.get("initial_position", [0, self.field_shape[1] // 2])
        if isinstance(initial_position, (list, tuple)):
            if len(initial_position) != 2:
                raise ValueError("initial_position must contain [x, y].")
            self.initial_position = (int(initial_position[0]), int(initial_position[1]))
        else:
            self.initial_position = (int(initial_position), self.field_shape[1] // 2)
        if not (
            0 <= self.initial_position[0] < self.field_shape[0]
            and 0 <= self.initial_position[1] < self.field_shape[1]
        ):
            raise ValueError(f"initial_position {self.initial_position} is outside the field.")
        base_observation_space = self._build_observation_builder(observation_cfg).space(
            self.field_shape
        )
        if not isinstance(base_observation_space, spaces.Box) or len(base_observation_space.shape) != 1:
            raise ValueError("Coordinate masking requires a flat Box observation.")
        action_count = self.action_space.n
        self.observation_space = spaces.Box(
            low=np.concatenate((base_observation_space.low, np.zeros(action_count, dtype=np.float32))),
            high=np.concatenate((base_observation_space.high, np.ones(action_count, dtype=np.float32))),
            dtype=np.float32,
        )
        self.field = None
        self.position = self.initial_position
        self.drilled_columns = []
        self.available_coordinates: list[tuple[int, int]] = []
        self.observation_builder = self._build_observation_builder(observation_cfg)

    @staticmethod
    def _build_observation_builder(observation_cfg):
        return build_observation_builder(
            str(observation_cfg["mode"]),
            representation=str(observation_cfg.get("representation", "map")),
            include_position=bool(observation_cfg.get("include_position", True)),
            unknown_value=float(observation_cfg.get("unknown_value", 0.0)),
            value_low=float(observation_cfg.get("value_low", 1.0)),
            value_high=float(observation_cfg.get("value_high", 5.0)),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pool = _load_pool(self.pool_paths)

        options = {} if options is None else options
        if "field_index" in options:
            idx = int(options["field_index"])
            if not 0 <= idx < len(pool):
                raise ValueError(
                    f"field_index must be between 0 and {len(pool) - 1}, got {idx}."
                )
        else:
            idx = self.np_random.integers(0, len(pool))
        self.field = np.asarray(pool[idx])
        if self.field.shape != self.field_shape:
            raise ValueError(
                f"Configured field shape {self.field_shape} does not match loaded "
                f"field shape {self.field.shape}."
            )
        self.position = self.initial_position
        self.drilled_columns = [self.initial_position]
        self.available_coordinates = [
            (x, y)
            for x in range(self.field_shape[0])
            for y in range(self.field_shape[1])
            if (x, y) != self.initial_position
        ]
        return self._get_obs(), {}

    def step(self, action):
        if not self.available_coordinates:
            raise RuntimeError("No drilling coordinates remain available.")
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(f"Action index must be in [0, {self.action_space.n - 1}].")
        coordinate = self.action_to_coordinate(action_index)

        if coordinate not in self.available_coordinates:
            rmse = self._compute_rmse()
            info = {
                "rmse": rmse,
                "num_holes": len(self.drilled_columns),
                "budget": self.max_holes,
                "invalid_action": True,
                "drilled_coordinate": coordinate,
                "available_coordinates": len(self.available_coordinates),
            }
            return self._get_obs(), -(rmse + 1.0), False, True, info

        self.position = coordinate
        self.drilled_columns.append(coordinate)
        self.available_coordinates.remove(coordinate)
        terminated = False
        truncated = len(self.drilled_columns) >= self.max_holes

        rmse = self._compute_rmse()
        num_holes = len(self.drilled_columns)
        reward = -(self.rmse_weight * rmse + self.borehole_weight * num_holes)

        info = {
            "rmse": rmse,
            "num_holes": num_holes,
            "budget": self.max_holes,
            "invalid_action": False,
            "drilled_coordinate": coordinate,
            "available_coordinates": len(self.available_coordinates),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def action_to_coordinate(self, action: int) -> tuple[int, int]:
        """Map a stable discrete action to its field coordinate."""
        action_index = int(action)
        width = self.field_shape[1]
        return action_index // width, action_index % width

    def coordinate_to_action(self, coordinate: tuple[int, int]) -> int:
        """Map a field coordinate to its stable discrete action."""
        x, y = (int(coordinate[0]), int(coordinate[1]))
        if not (0 <= x < self.field_shape[0] and 0 <= y < self.field_shape[1]):
            raise ValueError(f"Coordinate {(x, y)} is outside the field.")
        return x * self.field_shape[1] + y

    def action_masks(self) -> np.ndarray:
        """Return the MaskablePPO mask for currently available coordinates."""
        mask = np.ones(self.action_space.n, dtype=bool)
        for coordinate in self.drilled_columns:
            mask[self.coordinate_to_action(coordinate)] = False
        return mask

    def build_spatial_observation(self):
        """Return the current observation according to the configured builder."""
        if self.field is None:
            raise ValueError("Environment field is not initialized.")
        return self.observation_builder.build(self.field, self.position, self.drilled_columns)

    def _get_obs(self):
        observation = self.build_spatial_observation()
        if not isinstance(observation, np.ndarray):
            raise ValueError("Coordinate masking requires a flat array observation.")
        return np.concatenate((observation, self.action_masks().astype(np.float32)))

    def _compute_rmse(self):
        if self.use_initial_rmse_placeholder and len(self.drilled_columns) < 2:
            return self.initial_rmse

        points = np.asarray(self.drilled_columns, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(
            np.arange(self.field.shape[0]),
            np.arange(self.field.shape[1]),
            indexing="ij",
        )
        distances = (
            (grid_x[..., None] - points[:, 0]) ** 2
            + (grid_y[..., None] - points[:, 1]) ** 2
        )
        nearest = np.argmin(distances, axis=-1)
        interpolated = self.field[points[:, 0].astype(int), points[:, 1].astype(int), :][nearest]
        rmse = np.sqrt(np.mean((interpolated - self.field) ** 2))
        return float(rmse)
