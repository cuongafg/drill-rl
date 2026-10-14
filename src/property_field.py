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
        visualization_config: dict[str, Any] | None = None,
    ) -> None:
        self.geology_config = (
            load_config_file("geology/realistic_qc.yaml")
            if geology_config is None
            else geology_config
        )
        self.visualization_config = (
            load_config_file("visualization/default.yaml")
            if visualization_config is None
            else visualization_config
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
            visualization_config=np.asarray(
                OmegaConf.to_yaml(OmegaConf.create(self.visualization_config))
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
            visualization_config = (
                OmegaConf.to_container(
                    OmegaConf.create(str(np.asarray(loaded["visualization_config"]).item())),
                    resolve=True,
                )
                if "visualization_config" in loaded.files
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
            visualization_config=visualization_config,
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

    def plot_horizontal_slice(
        self,
        z: float,
        ax: Any | None = None,
        output_path: str | Path | None = None,
    ) -> Any:
        """Plot a qc map at the nearest requested elevation in metres."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        if self.field is None:
            raise ValueError("The field has not been generated yet. Call generate() first.")
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.visualization_config["slice_figsize"]))
        iz = int(np.argmin(np.abs(self.grid.z_coords - z)))
        image = ax.imshow(
            self.field[:, :, iz].T,
            origin="lower",
            aspect="equal",
            cmap=self.visualization_config["cmap"],
            norm=PowerNorm(
                gamma=float(self.visualization_config["qc_gamma"]),
                vmin=float(self.field.min()),
                vmax=float(self.field.max()),
            ),
            extent=[self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max],
        )
        ax.set(title=f"qc at z={self.grid.z_coords[iz]:.2f} m", xlabel="X (m)", ylabel="Y (m)")
        if own_figure:
            ax.figure.colorbar(image, ax=ax, label=f"{self.property_name} ({self.unit})")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(
                destination,
                dpi=int(self.visualization_config["preview_dpi"]),
                bbox_inches="tight",
            )
        return ax

    def plot_vertical_section(
        self,
        plane: str = "xz",
        coordinate: float | None = None,
        ax: Any | None = None,
        output_path: str | Path | None = None,
    ) -> Any:
        """Plot an X-Z or Y-Z section at the nearest horizontal coordinate."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        if self.field is None:
            raise ValueError("The field has not been generated yet. Call generate() first.")
        plane = plane.lower()
        if plane not in {"xz", "yz"}:
            raise ValueError("plane must be 'xz' or 'yz'.")
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.visualization_config["section_figsize"]))
        if plane == "xz":
            coordinate = float(self.grid.y_coords[self.grid.ny // 2] if coordinate is None else coordinate)
            iy = int(np.argmin(np.abs(self.grid.y_coords - coordinate)))
            section = self.field[:, iy, :].T
            horizontal_coords = self.grid.x_coords
            horizontal_label = "X (m)"
            actual_coordinate = self.grid.y_coords[iy]
        else:
            coordinate = float(self.grid.x_coords[self.grid.nx // 2] if coordinate is None else coordinate)
            ix = int(np.argmin(np.abs(self.grid.x_coords - coordinate)))
            section = self.field[ix, :, :].T
            horizontal_coords = self.grid.y_coords
            horizontal_label = "Y (m)"
            actual_coordinate = self.grid.x_coords[ix]

        z_bottom = float(np.min(self.grid.z_coords))
        z_surface = float(np.max(self.grid.z_coords))
        image = ax.imshow(
            section,
            origin="upper",
            aspect="auto",
            cmap=self.visualization_config["cmap"],
            norm=PowerNorm(
                gamma=float(self.visualization_config["qc_gamma"]),
                vmin=float(self.field.min()),
                vmax=float(self.field.max()),
            ),
            extent=[
                float(horizontal_coords[0]),
                float(horizontal_coords[-1]),
                z_bottom,
                z_surface,
            ],
        )
        ax.set(xlabel=horizontal_label, ylabel="Z (m)")
        ax.set_title(f"{plane.upper()} section at {actual_coordinate:.1f} m")
        if own_figure:
            ax.figure.colorbar(image, ax=ax, label=f"{self.property_name} ({self.unit})")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(
                destination,
                dpi=int(self.visualization_config["preview_dpi"]),
                bbox_inches="tight",
            )
        return ax

    def plot_bedrock_surface(
        self,
        ax: Any | None = None,
        output_path: str | Path | None = None,
    ) -> Any:
        """Plot the undulating bedrock elevation (negative z, in metres)."""
        import matplotlib.pyplot as plt

        if self.bedrock_depth is None:
            raise ValueError("Generate the field before plotting its bedrock surface.")
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.visualization_config["slice_figsize"]))
        elevation = -self.bedrock_depth.T
        image = ax.imshow(
            elevation,
            origin="lower",
            aspect="equal",
            cmap=self.visualization_config["bedrock_cmap"],
            extent=[self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max],
        )
        ax.set(title="Bedrock surface elevation", xlabel="X (m)", ylabel="Y (m)")
        if own_figure:
            ax.figure.colorbar(image, ax=ax, label="Bedrock Z (m)")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(
                destination,
                dpi=int(self.visualization_config["preview_dpi"]),
                bbox_inches="tight",
            )
        return ax

    def plot_drilled_holes(
        self,
        drilled_positions: Sequence[tuple[float, float]] | None = None,
        z: float | None = None,
        ax: Any | None = None,
        output_path: str | Path | None = None,
        as_indices: bool = False,
    ) -> Any:
        """Plot a QC slice with drilled x-y locations highlighted as boreholes.

        Positions may be given either in physical metres or as grid indices. The
        default behaviour shows a few representative holes, which is useful for
        generating a static preview in ``environment_visualisation/``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        if self.field is None:
            raise ValueError("The field has not been generated yet. Call generate() first.")

        if drilled_positions is None:
            drilled_positions = [
                (18.0, 22.0),
                (32.0, 68.0),
                (58.0, 42.0),
                (72.0, 76.0),
                (84.0, 30.0),
            ]

        z_to_plot = float(self.grid.z_coords[0] if z is None else z)
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.visualization_config["slice_figsize"]))

        iz = int(np.argmin(np.abs(self.grid.z_coords - z_to_plot)))
        image = ax.imshow(
            self.field[:, :, iz].T,
            origin="lower",
            aspect="equal",
            cmap=self.visualization_config["cmap"],
            norm=PowerNorm(
                gamma=float(self.visualization_config["qc_gamma"]),
                vmin=float(self.field.min()),
                vmax=float(self.field.max()),
            ),
            extent=[self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max],
        )

        x_points: list[float] = []
        y_points: list[float] = []
        for position in drilled_positions:
            if len(position) != 2:
                raise ValueError("Each drilled position must contain exactly (x, y).")
            x_world, y_world = float(position[0]), float(position[1])
            if as_indices:
                ix = int(round(x_world))
                iy = int(round(y_world))
                if not (0 <= ix < self.grid.nx and 0 <= iy < self.grid.ny):
                    raise ValueError(f"Drilled index ({ix}, {iy}) is outside the model extent.")
                x_world = float(self.grid.x_coords[ix])
                y_world = float(self.grid.y_coords[iy])
            else:
                if not (self.grid.x_min <= x_world <= self.grid.x_max and self.grid.y_min <= y_world <= self.grid.y_max):
                    raise ValueError(
                        f"Drilled world position ({x_world}, {y_world}) is outside the model extent."
                    )
            x_points.append(x_world)
            y_points.append(y_world)

        ax.scatter(
            x_points,
            y_points,
            s=180,
            c="white",
            edgecolors="black",
            linewidths=1.8,
            marker="o",
            alpha=0.95,
            zorder=3,
        )
        ax.scatter(
            x_points,
            y_points,
            s=55,
            c="black",
            marker="x",
            linewidths=1.8,
            zorder=4,
        )
        ax.set(
            title=f"Observed qc map at z={self.grid.z_coords[iz]:.2f} m with drilled holes",
            xlabel="X (m)",
            ylabel="Y (m)",
        )
        if own_figure:
            ax.figure.colorbar(image, ax=ax, label=f"{self.property_name} ({self.unit})")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(
                destination,
                dpi=int(self.visualization_config["preview_dpi"]),
                bbox_inches="tight",
            )
        return ax

    def visualize_3d(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Create the required depth slices, sections, 3D volume, and bedrock map."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import PowerNorm

        if self.field is None or self.bedrock_depth is None:
            raise ValueError("Generate the field before visualizing it.")

        figure_size = (
            tuple(self.visualization_config["preview_figsize"])
            if figsize is None
            else figsize
        )
        fig = plt.figure(figsize=figure_size, constrained_layout=True)
        layout = fig.add_gridspec(2, 4)
        axes = [
            [fig.add_subplot(layout[0, column]) for column in range(4)],
            [
                fig.add_subplot(layout[1, column], projection="3d" if column >= 2 else None)
                for column in range(4)
            ],
        ]
        fig.suptitle(f"3D {self.property_name.upper()} geological ground truth")

        for ax, level in zip(axes[0][:3], (0.0, -5.0, -10.0)):
            self.plot_horizontal_slice(level, ax=ax)
        self.plot_bedrock_surface(ax=axes[0][3])
        self.plot_vertical_section("xz", ax=axes[1][0])
        self.plot_vertical_section("yz", ax=axes[1][1])

        volume_ax = axes[1][2]
        x_idx = np.unique(
            np.linspace(
                0,
                self.grid.nx - 1,
                min(self.grid.nx, int(self.visualization_config["volume_max_x_samples"])),
                dtype=int,
            )
        )
        y_idx = np.unique(
            np.linspace(
                0,
                self.grid.ny - 1,
                min(self.grid.ny, int(self.visualization_config["volume_max_y_samples"])),
                dtype=int,
            )
        )
        z_idx = np.unique(
            np.linspace(
                0,
                self.grid.nz - 1,
                min(self.grid.nz, int(self.visualization_config["volume_max_z_samples"])),
                dtype=int,
            )
        )
        x = self.grid.x_coords[x_idx]
        y = self.grid.y_coords[y_idx]
        z_indices = z_idx[::-1]
        z = self.grid.z_coords[z_indices]
        cmap = plt.get_cmap(self.visualization_config["cmap"])
        norm = PowerNorm(
            gamma=float(self.visualization_config["qc_gamma"]),
            vmin=float(self.field.min()),
            vmax=float(self.field.max()),
        )

        # Orthogonal cut planes reveal interior geology instead of hiding it
        # behind the opaque outer faces of a filled voxel block.
        x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
        top_values = self.field[np.ix_(x_idx, y_idx, [0])][:, :, 0]
        volume_ax.plot_surface(
            x_mesh,
            y_mesh,
            np.full_like(x_mesh, float(self.grid.z_coords[0])),
            facecolors=cmap(norm(top_values)),
            shade=False,
            linewidth=0,
        )

        middle_y = y_idx[len(y_idx) // 2]
        section_x_mesh, section_z_mesh = np.meshgrid(x, z, indexing="ij")
        xz_values = self.field[np.ix_(x_idx, [middle_y], z_indices)][:, 0, :]
        volume_ax.plot_surface(
            section_x_mesh,
            np.full_like(section_x_mesh, self.grid.y_coords[middle_y]),
            section_z_mesh,
            facecolors=cmap(norm(xz_values)),
            shade=False,
            linewidth=0,
        )

        middle_x = x_idx[len(x_idx) // 2]
        section_y_mesh, section_z_mesh = np.meshgrid(y, z, indexing="ij")
        yz_values = self.field[np.ix_([middle_x], y_idx, z_indices)][0, :, :]
        volume_ax.plot_surface(
            np.full_like(section_y_mesh, self.grid.x_coords[middle_x]),
            section_y_mesh,
            section_z_mesh,
            facecolors=cmap(norm(yz_values)),
            shade=False,
            linewidth=0,
        )
        volume_ax.set(
            title="3D qc volume (orthogonal sections)",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            xlim=(self.grid.x_min, self.grid.x_max),
            ylim=(self.grid.y_min, self.grid.y_max),
            zlim=(float(np.min(self.grid.z_coords)), float(np.max(self.grid.z_coords))),
        )
        volume_ax.set_box_aspect(self.visualization_config["volume_aspect"])
        volume_ax.view_init(
            elev=int(self.visualization_config["elevation"]),
            azim=int(self.visualization_config["azimuth"]),
        )

        bedrock_ax = axes[1][3]
        x_mesh, y_mesh = np.meshgrid(self.grid.x_coords, self.grid.y_coords, indexing="ij")
        bedrock_ax.plot_surface(
            x_mesh,
            y_mesh,
            -self.bedrock_depth,
            cmap=self.visualization_config["bedrock_cmap"],
            linewidth=0,
            antialiased=True,
        )
        bedrock_ax.set(
            title="3D bedrock surface",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            xlim=(self.grid.x_min, self.grid.x_max),
            ylim=(self.grid.y_min, self.grid.y_max),
            zlim=(float(np.min(self.grid.z_coords)), 0.0),
        )
        bedrock_ax.set_box_aspect(self.visualization_config["bedrock_aspect"])
        bedrock_ax.view_init(
            elev=int(self.visualization_config["elevation"]),
            azim=int(self.visualization_config["azimuth"]),
        )

        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=[*axes[0], axes[1][0], axes[1][1], volume_ax],
            shrink=float(self.visualization_config["colorbar_shrink"]),
            pad=float(self.visualization_config["colorbar_pad"]),
        )
        colorbar.set_label(f"{self.property_name} ({self.unit})")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                destination,
                dpi=int(self.visualization_config["preview_dpi"]),
                bbox_inches="tight",
            )
        return fig

    def visualize(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Backward-compatible alias for :meth:`visualize_3d`."""
        return self.visualize_3d(output_path=output_path, figsize=figsize)

    def __repr__(self) -> str:
        return (
            f"PropertyField3D(nx={self.grid.nx}, ny={self.grid.ny}, nz={self.grid.nz}, "
            f"property={self.property_name} [{self.unit}], range={self.property_range}, "
            f"lenses={self.lens_count}, seed={self.seed})"
        )
