from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter
from scipy.special import expit

from configuration import load_config_file
from grid3d import Grid3D


def world_to_index(grid: Grid3D, x: float, y: float, z: float) -> tuple[int, int, int]:
    """Map physical coordinates in metres to the nearest voxel index."""
    return grid.world_to_index(x, y, z)


def index_to_world(grid: Grid3D, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
    """Map voxel indices to physical coordinates in metres."""
    return grid.index_to_world(ix, iy, iz)


class PropertyField3D:
    """Reproducible, continuous 3D cone-resistance ground-truth model.

    ``field[ix, iy, iz]`` stores qc in MPa. Geological units are represented
    only through smooth property contrasts and masks; the returned volume is
    continuous and has no categorical lithology labels.
    """

    def __init__(
        self,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        property_name: str | None = None,
        unit: str | None = None,
        property_range: tuple[float, float] | None = None,
        seed: int | None = None,
        lens_count: int | None = None,
        include_fault: bool | None = None,
        geology_config: dict[str, Any] | None = None,
    ) -> None:
        self.geology_config = (
            load_config_file("geology/realistic_qc.yaml")
            if geology_config is None
            else geology_config
        )
        geology = self.geology_config
        grid_settings = load_config_file("grid/site.yaml")
        requested_grid = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        }
        for name, value in requested_grid.items():
            if value is None:
                requested_grid[name] = grid_settings[name]
        nx, ny, nz = (int(requested_grid[name]) for name in ("nx", "ny", "nz"))
        x_min, x_max = requested_grid["x_min"], requested_grid["x_max"]
        y_min, y_max = requested_grid["y_min"], requested_grid["y_max"]
        z_min, z_max = requested_grid["z_min"], requested_grid["z_max"]
        if lens_count is None:
            lens_count = int(geology["lens_count"])
        if include_fault is None:
            include_fault = bool(geology["include_fault"])
        if property_name is None:
            property_name = str(geology["property_name"])
        if unit is None:
            unit = str(geology["unit"])
        if property_range is None:
            property_range = tuple(float(value) for value in geology["property_range"])

        self.grid = Grid3D(
            nx=nx,
            ny=ny,
            nz=nz,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        if property_range[0] >= property_range[1]:
            raise ValueError("property_range must have an increasing lower and upper bound.")
        if lens_count < 0:
            raise ValueError("lens_count must be non-negative.")

        self.property_name = property_name
        self.unit = unit
        self.property_range = (float(property_range[0]), float(property_range[1]))
        self.seed = seed
        self.lens_count = lens_count
        self.include_fault = include_fault
        self.field: np.ndarray | None = None
        self.bedrock_depth: np.ndarray | None = None

    def world_to_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Convert a world coordinate in metres to its nearest voxel index."""
        return world_to_index(self.grid, x, y, z)

    def index_to_world(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
        """Convert a voxel index to its physical coordinate in metres."""
        return index_to_world(self.grid, ix, iy, iz)

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        spread = float(values.std())
        if spread < 1e-7:
            return np.zeros_like(values)
        return (values - float(values.mean())) / spread

    def _noise_3d(
        self,
        rng: np.random.Generator,
        correlation_m: tuple[float, float, float],
    ) -> np.ndarray:
        """Create anisotropic correlated noise using physical correlation lengths."""
        spacing = []
        for coords in (self.grid.x_coords, self.grid.y_coords, self.grid.z_coords):
            spacing.append(abs(float(coords[1] - coords[0])) if coords.size > 1 else 1.0)
        sigma = tuple(
            max(0.5, correlation / (2.0 * spacing_axis))
            for correlation, spacing_axis in zip(correlation_m, spacing)
        )
        white_noise = rng.normal(size=(self.grid.nx, self.grid.ny, self.grid.nz)).astype(np.float32)
        return self._standardize(gaussian_filter(white_noise, sigma=sigma, mode="reflect"))

    def _noise_2d(
        self,
        rng: np.random.Generator,
        correlation_m: tuple[float, float],
    ) -> np.ndarray:
        spacing_x = abs(float(self.grid.x_coords[1] - self.grid.x_coords[0])) if self.grid.nx > 1 else 1.0
        spacing_y = abs(float(self.grid.y_coords[1] - self.grid.y_coords[0])) if self.grid.ny > 1 else 1.0
        sigma = (
            max(0.5, correlation_m[0] / (2.0 * spacing_x)),
            max(0.5, correlation_m[1] / (2.0 * spacing_y)),
        )
        white_noise = rng.normal(size=(self.grid.nx, self.grid.ny)).astype(np.float32)
        return self._standardize(gaussian_filter(white_noise, sigma=sigma, mode="reflect"))

    def _make_bedrock_surface(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Generate irregular bedrock depth (positive metres below ground)."""
        bedrock = self.geology_config["bedrock"]
        correlation = self.geology_config["correlation"]
        broad = self._noise_2d(rng, tuple(correlation["bedrock_broad"]))
        local = self._noise_2d(rng, tuple(correlation["bedrock_local"]))
        depth = (
            float(bedrock["mean_depth"])
            + float(bedrock["broad_relief"]) * broad
            + float(bedrock["local_relief"]) * local
        )

        # A broad, asymmetric paleo-valley provides site-scale geological relief.
        center_x = rng.uniform(self.grid.x_min, self.grid.x_max)
        center_y = rng.uniform(self.grid.y_min, self.grid.y_max)
        angle = rng.uniform(0.0, np.pi)
        dx = x_grid - center_x
        dy = y_grid - center_y
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        valley_length = bedrock["valley_length"]
        valley_width = bedrock["valley_width"]
        valley = np.exp(
            -0.5
            * (
                (along / rng.uniform(*valley_length)) ** 2
                + (across / rng.uniform(*valley_width)) ** 2
            )
        )
        depth -= rng.uniform(*bedrock["valley_depth"]) * valley
        mean_depth = float(bedrock["mean_depth"])
        relief_scale = float(bedrock["broad_relief"]) * 2.0
        depth = mean_depth + relief_scale * np.tanh((depth - mean_depth) / relief_scale)
        return depth.astype(np.float32)

    @staticmethod
    def _lens_mask(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        depth_grid: np.ndarray,
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        angle: float,
        boundary_noise: np.ndarray,
        boundary_irregularity: float,
        transition_steepness: float,
    ) -> np.ndarray:
        """Smooth asymmetric ellipsoid mask for a geological lens."""
        cx, cy, cd = center
        rx, ry, rd = radii
        dx = x_grid - cx
        dy = y_grid - cy
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        distance = np.sqrt(
            (along / rx) ** 2
            + (across / ry) ** 2
            + ((depth_grid - cd) / rd) ** 2
        )
        irregular_distance = distance - boundary_irregularity * boundary_noise
        return expit((1.0 - irregular_distance) * transition_steepness).astype(np.float32)

    def generate(
        self,
        seed: int | None = None,
        property_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Generate layered soils, irregular lenses, weathered rock, and bedrock.

        Generation is deterministic for a given seed. Geological bodies are
        smooth 3D ellipsoidal random fields with anisotropic continuity, not
        periodic patterns or independent random voxel assignments.
        """
        seed_used = self.seed if seed is None else seed
        rng = np.random.default_rng(seed_used)
        if property_range is not None:
            if property_range[0] >= property_range[1]:
                raise ValueError("property_range must have an increasing lower and upper bound.")
            self.property_range = (float(property_range[0]), float(property_range[1]))

        x = self.grid.x_coords.astype(np.float32)
        y = self.grid.y_coords.astype(np.float32)
        depth = -self.grid.z_coords.astype(np.float32)
        x_grid, y_grid, depth_grid = np.meshgrid(x, y, depth, indexing="ij")
        depth_fraction = depth_grid / max(float(depth.max()), 1e-6)

        geology = self.geology_config
        correlation = geology["correlation"]
        background = geology["background"]
        broad_noise = self._noise_3d(rng, tuple(correlation["broad"]))
        medium_noise = self._noise_3d(rng, tuple(correlation["medium"]))
        fine_noise = self._noise_3d(rng, tuple(correlation["fine"]))
        volumetric_variation = (
            float(background["broad_noise_weight"]) * broad_noise
            + float(background["medium_noise_weight"]) * medium_noise
            + float(background["fine_noise_weight"]) * fine_noise
        )

        # The background trend is depth-dependent; correlated 3D terms ensure
        # the field is not a 2D map extruded through depth.
        qc = (
            float(background["qc_at_surface"])
            + float(background["depth_increment"]) * depth_fraction
            + volumetric_variation
            + float(background["depth_noise_weight"]) * depth_fraction * medium_noise
        ).astype(np.float32)

        bedrock = self._make_bedrock_surface(
            rng,
            x_grid[:, :, 0],
            y_grid[:, :, 0],
        )
        if self.include_fault:
            fault_x = rng.uniform(self.grid.x_min, self.grid.x_max)
            fault_y = rng.uniform(self.grid.y_min, self.grid.y_max)
            fault_angle = rng.uniform(0.0, np.pi)
            across_fault = (
                -(x_grid[:, :, 0] - fault_x) * np.sin(fault_angle)
                + (y_grid[:, :, 0] - fault_y) * np.cos(fault_angle)
            )
            bedrock_settings = geology["bedrock"]
            throw = rng.uniform(*bedrock_settings["fault_throw"])
            bedrock = np.clip(
                bedrock + np.where(across_fault > 0.0, throw, -throw),
                float(bedrock_settings["minimum_depth"]),
                float(bedrock_settings["maximum_depth"]),
            ).astype(np.float32)
        self.bedrock_depth = bedrock

        lens_settings = geology["lenses"]
        boundary_noise = self._noise_3d(rng, tuple(correlation["boundaries"]))
        for lens_index in range(self.lens_count):
            kind = rng.choice(lens_settings["kinds"], p=lens_settings["probabilities"])
            center_x = rng.uniform(self.grid.x_min, self.grid.x_max)
            center_y = rng.uniform(self.grid.y_min, self.grid.y_max)
            center_depth = rng.uniform(*lens_settings["depth"])
            radii = (
                rng.uniform(*lens_settings["radius_x"]),
                rng.uniform(*lens_settings["radius_y"]),
                rng.uniform(*lens_settings["radius_z"]),
            )
            boundary_irregularity = float(lens_settings["boundary_irregularity"])
            transition_steepness = float(lens_settings["transition_steepness"])
            mask = self._lens_mask(
                x_grid,
                y_grid,
                depth_grid,
                (center_x, center_y, center_depth),
                radii,
                rng.uniform(0.0, np.pi),
                boundary_noise,
                boundary_irregularity,
                transition_steepness,
            )
            if kind == "clay":
                target_qc = rng.uniform(*lens_settings["clay_qc"]) + float(
                    lens_settings["boundary_clay_noise"]
                ) * medium_noise
            elif kind == "dense_sand":
                target_qc = rng.uniform(*lens_settings["dense_sand_qc"]) + float(
                    lens_settings["boundary_dense_sand_noise"]
                ) * medium_noise
            else:
                target_qc = rng.uniform(*lens_settings["sand_qc"]) + float(
                    lens_settings["boundary_sand_noise"]
                ) * medium_noise
            qc = qc * (1.0 - mask) + target_qc.astype(np.float32) * mask

        # Localized dense sand bodies extend laterally while retaining vertical
        # thickness and irregular boundaries.
        dense_settings = geology["dense_zones"]
        dense_count = max(
            int(dense_settings["minimum_count"]),
            self.lens_count // int(dense_settings["count_per_six_lenses"]),
        )
        for _ in range(dense_count):
            center_x = rng.uniform(self.grid.x_min, self.grid.x_max)
            center_y = rng.uniform(self.grid.y_min, self.grid.y_max)
            center_depth = rng.uniform(*dense_settings["depth"])
            mask = self._lens_mask(
                x_grid,
                y_grid,
                depth_grid,
                (center_x, center_y, center_depth),
                (
                    rng.uniform(*dense_settings["radius_x"]),
                    rng.uniform(*dense_settings["radius_y"]),
                    rng.uniform(*dense_settings["radius_z"]),
                ),
                rng.uniform(0.0, np.pi),
                boundary_noise,
                boundary_irregularity,
                transition_steepness,
            )
            target_qc = (
                float(dense_settings["qc_mean"])
                + float(dense_settings["broad_noise_weight"]) * broad_noise
                + float(dense_settings["medium_noise_weight"]) * medium_noise
            )
            qc = qc * (1.0 - mask) + target_qc * mask

        # Progressively weathered material transitions into high-resistance rock.
        rock_settings = geology["rock"]
        bedrock_3d = bedrock[:, :, None]
        weathered_fraction = expit(
            (
                depth_grid
                - (bedrock_3d - float(rock_settings["weathering_thickness"]))
            )
            / float(rock_settings["weathering_transition"])
        ).astype(np.float32)
        bedrock_fraction = expit(
            (depth_grid - bedrock_3d) / float(rock_settings["bedrock_transition"])
        ).astype(np.float32)
        weathered_qc = np.clip(
            float(rock_settings["weathered_qc_mean"])
            + float(rock_settings["weathered_broad_noise_weight"]) * broad_noise
            + float(rock_settings["weathered_medium_noise_weight"]) * medium_noise,
            *rock_settings["weathered_qc_range"],
        )
        competent_rock_qc = np.clip(
            float(rock_settings["competent_qc_mean"])
            + float(rock_settings["competent_broad_noise_weight"]) * broad_noise
            + float(rock_settings["competent_fine_noise_weight"]) * fine_noise,
            *rock_settings["competent_qc_range"],
        )
        qc = qc * (1.0 - weathered_fraction) + weathered_qc * weathered_fraction
        qc = qc * (1.0 - bedrock_fraction) + competent_rock_qc * bedrock_fraction

        # A surface deposit remains predominantly soft-to-medium qc while still
        # carrying gradual spatial variation.
        surface_settings = geology["surface"]
        surface_fraction = expit(
            (float(surface_settings["thickness"]) - depth_grid)
            / float(surface_settings["transition"])
        ).astype(np.float32)
        surface_qc = np.clip(
            float(surface_settings["qc_mean"])
            + float(surface_settings["broad_noise_weight"]) * broad_noise
            + float(surface_settings["medium_noise_weight"]) * medium_noise,
            *surface_settings["qc_range"],
        )
        qc = qc * (1.0 - surface_fraction) + surface_qc * surface_fraction

        self.field = np.clip(qc, *self.property_range).astype(np.float32)
        self.seed = seed_used
        return self.field

    def save(self, path: str | Path) -> str:
        """Save generated values, grid coordinates, and model metadata to NPZ."""
        if self.field is None or self.bedrock_depth is None:
            raise ValueError("Generate the field before saving it.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            field=self.field,
            bedrock_depth=self.bedrock_depth,
            x_coords=self.grid.x_coords,
            y_coords=self.grid.y_coords,
            z_coords=self.grid.z_coords,
            property_name=np.asarray(self.property_name),
            unit=np.asarray(self.unit),
            property_range=np.asarray(self.property_range, dtype=np.float64),
            seed=np.asarray(-1 if self.seed is None else self.seed, dtype=np.int64),
            lens_count=np.asarray(self.lens_count, dtype=np.int64),
            include_fault=np.asarray(self.include_fault, dtype=np.bool_),
            geology_config=np.asarray(
                OmegaConf.to_yaml(OmegaConf.create(self.geology_config))
            ),
        )
        return str(destination)

    @classmethod
    def load(cls, path: str | Path) -> "PropertyField3D":
        """Load a saved model, including metadata from earlier NPZ versions."""
        with np.load(path) as loaded:
            x_coords = np.asarray(loaded["x_coords"])
            y_coords = np.asarray(loaded["y_coords"])
            z_coords = np.asarray(loaded["z_coords"])
            field = np.asarray(loaded["field"])
            property_name = str(np.asarray(loaded["property_name"]).item())
            unit = str(np.asarray(loaded["unit"]).item())
            property_range = tuple(
                np.asarray(loaded["property_range"], dtype=np.float64).tolist()
            )
            bedrock = (
                np.asarray(loaded["bedrock_depth"], dtype=np.float32)
                if "bedrock_depth" in loaded.files
                else None
            )
            seed = int(np.asarray(loaded["seed"]).item()) if "seed" in loaded.files else -1
            lens_count = int(np.asarray(loaded["lens_count"]).item()) if "lens_count" in loaded.files else 24
            include_fault = (
                bool(np.asarray(loaded["include_fault"]).item())
                if "include_fault" in loaded.files
                else False
            )
            geology_config = (
                OmegaConf.to_container(
                    OmegaConf.create(str(np.asarray(loaded["geology_config"]).item())),
                    resolve=True,
                )
                if "geology_config" in loaded.files
                else None
            )
        obj = cls(
            nx=x_coords.size,
            ny=y_coords.size,
            nz=z_coords.size,
            x_min=float(x_coords[0]),
            x_max=float(x_coords[-1]),
            y_min=float(y_coords[0]),
            y_max=float(y_coords[-1]),
            z_min=float(z_coords[0]),
            z_max=float(z_coords[-1]),
            property_name=property_name,
            unit=unit,
            property_range=property_range,
            seed=None if seed < 0 else seed,
            lens_count=lens_count,
            include_fault=include_fault,
            geology_config=geology_config,
        )
        if field.shape != (obj.grid.nx, obj.grid.ny, obj.grid.nz):
            raise ValueError("Saved field dimensions do not match the stored coordinate vectors.")
        obj.field = field
        obj.bedrock_depth = bedrock
        return obj

    def sample_borehole(self, x: float, y: float) -> np.ndarray:
        """Return ``(z_m, qc_mpa)`` samples vertically from surface to model base."""
        if self.field is None:
            raise ValueError("The field has not been generated yet. Call generate() first.")
        if not (self.grid.x_min <= x <= self.grid.x_max):
            raise ValueError(f"x={x} lies outside the model extent.")
        if not (self.grid.y_min <= y <= self.grid.y_max):
            raise ValueError(f"y={y} lies outside the model extent.")
        ix, iy, _ = self.world_to_index(x, y, 0.0)
        depth_order = np.argsort(-self.grid.z_coords)
        return np.column_stack(
            (self.grid.z_coords[depth_order], self.field[ix, iy, depth_order])
        )

    def __repr__(self) -> str:
        return (
            f"PropertyField3D(nx={self.grid.nx}, ny={self.grid.ny}, nz={self.grid.nz}, "
            f"property={self.property_name} [{self.unit}], range={self.property_range}, "
            f"lenses={self.lens_count}, seed={self.seed})"
        )
