from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit

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
        nx: int = 101,
        ny: int = 101,
        nz: int = 21,
        x_min: float = 0.0,
        x_max: float = 100.0,
        y_min: float = 0.0,
        y_max: float = 100.0,
        z_min: float = 0.0,
        z_max: float = -10.0,
        property_name: str = "qc",
        unit: str = "MPa",
        property_range: tuple[float, float] = (0.0, 300.0),
        seed: int | None = None,
        lens_count: int = 24,
        include_fault: bool = False,
    ) -> None:
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
        broad = self._noise_2d(rng, (55.0, 48.0))
        local = self._noise_2d(rng, (24.0, 20.0))
        depth = 8.0 + 0.9 * broad + 0.25 * local

        # A broad, asymmetric paleo-valley provides site-scale geological relief.
        center_x = rng.uniform(30.0, 70.0)
        center_y = rng.uniform(30.0, 70.0)
        angle = rng.uniform(0.0, np.pi)
        dx = x_grid - center_x
        dy = y_grid - center_y
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        valley = np.exp(-0.5 * ((along / rng.uniform(30.0, 48.0)) ** 2
                                + (across / rng.uniform(10.0, 22.0)) ** 2))
        depth -= rng.uniform(0.7, 1.5) * valley
        depth = 7.8 + 1.8 * np.tanh((depth - 7.8) / 1.8)
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
        irregular_distance = distance - 0.10 * boundary_noise
        return expit((1.0 - irregular_distance) * 7.0).astype(np.float32)

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

        broad_noise = self._noise_3d(rng, (42.0, 36.0, 3.8))
        medium_noise = self._noise_3d(rng, (17.0, 14.0, 1.8))
        fine_noise = self._noise_3d(rng, (6.0, 7.0, 0.8))
        volumetric_variation = (
            2.8 * broad_noise + 1.7 * medium_noise + 0.65 * fine_noise
        )

        # The background trend is depth-dependent; correlated 3D terms ensure
        # the field is not a 2D map extruded through depth.
        qc = (
            6.0
            + 17.0 * depth_fraction
            + volumetric_variation
            + 2.0 * depth_fraction * medium_noise
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
            throw = rng.uniform(0.4, 1.0)
            bedrock = np.clip(
                bedrock + np.where(across_fault > 0.0, throw, -throw),
                5.8,
                9.7,
            ).astype(np.float32)
        self.bedrock_depth = bedrock

        boundary_noise = self._noise_3d(rng, (13.0, 11.0, 1.2))
        for lens_index in range(self.lens_count):
            kind = rng.choice(("sand", "clay", "dense_sand"), p=(0.55, 0.28, 0.17))
            center_x = rng.uniform(self.grid.x_min, self.grid.x_max)
            center_y = rng.uniform(self.grid.y_min, self.grid.y_max)
            center_depth = rng.uniform(1.0, 7.3)
            radii = (
                rng.uniform(8.0, 22.0),
                rng.uniform(6.0, 18.0),
                rng.uniform(0.55, 1.8),
            )
            mask = self._lens_mask(
                x_grid,
                y_grid,
                depth_grid,
                (center_x, center_y, center_depth),
                radii,
                rng.uniform(0.0, np.pi),
                boundary_noise,
            )
            if kind == "clay":
                target_qc = rng.uniform(2.8, 6.0) + 0.6 * medium_noise
            elif kind == "dense_sand":
                target_qc = rng.uniform(55.0, 82.0) + 3.0 * medium_noise
            else:
                target_qc = rng.uniform(20.0, 43.0) + 2.0 * medium_noise
            qc = qc * (1.0 - mask) + target_qc.astype(np.float32) * mask

        # Localized dense sand bodies extend laterally while retaining vertical
        # thickness and irregular boundaries.
        for _ in range(max(1, self.lens_count // 6)):
            center_x = rng.uniform(self.grid.x_min, self.grid.x_max)
            center_y = rng.uniform(self.grid.y_min, self.grid.y_max)
            center_depth = rng.uniform(2.5, 7.0)
            mask = self._lens_mask(
                x_grid,
                y_grid,
                depth_grid,
                (center_x, center_y, center_depth),
                (rng.uniform(15.0, 30.0), rng.uniform(9.0, 20.0), rng.uniform(0.7, 1.5)),
                rng.uniform(0.0, np.pi),
                boundary_noise,
            )
            target_qc = 68.0 + 4.0 * broad_noise + 2.0 * medium_noise
            qc = qc * (1.0 - mask) + target_qc * mask

        # Progressively weathered material transitions into high-resistance rock.
        bedrock_3d = bedrock[:, :, None]
        weathered_fraction = expit(
            (depth_grid - (bedrock_3d - 2.0)) / 0.42
        ).astype(np.float32)
        bedrock_fraction = expit(
            (depth_grid - bedrock_3d) / 0.20
        ).astype(np.float32)
        weathered_qc = np.clip(112.0 + 13.0 * broad_noise + 6.0 * medium_noise, 75.0, 165.0)
        competent_rock_qc = np.clip(252.0 + 18.0 * broad_noise + 8.0 * fine_noise, 205.0, 300.0)
        qc = qc * (1.0 - weathered_fraction) + weathered_qc * weathered_fraction
        qc = qc * (1.0 - bedrock_fraction) + competent_rock_qc * bedrock_fraction

        # A surface deposit remains predominantly soft-to-medium qc while still
        # carrying gradual spatial variation.
        surface_fraction = expit((1.25 - depth_grid) / 0.28).astype(np.float32)
        surface_qc = np.clip(7.0 + 2.2 * broad_noise + 1.0 * medium_noise, 1.0, 14.0)
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
            _, ax = plt.subplots(figsize=(7, 6))
        iz = int(np.argmin(np.abs(self.grid.z_coords - z)))
        image = ax.imshow(
            self.field[:, :, iz].T,
            origin="lower",
            aspect="equal",
            cmap="viridis",
            norm=PowerNorm(
                gamma=0.45,
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
            ax.figure.savefig(destination, dpi=180, bbox_inches="tight")
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
            _, ax = plt.subplots(figsize=(9, 5))
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
            cmap="viridis",
            norm=PowerNorm(
                gamma=0.45,
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
            ax.figure.savefig(destination, dpi=180, bbox_inches="tight")
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
            _, ax = plt.subplots(figsize=(7, 6))
        elevation = -self.bedrock_depth.T
        image = ax.imshow(
            elevation,
            origin="lower",
            aspect="equal",
            cmap="terrain",
            extent=[self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max],
        )
        ax.set(title="Bedrock surface elevation", xlabel="X (m)", ylabel="Y (m)")
        if own_figure:
            ax.figure.colorbar(image, ax=ax, label="Bedrock Z (m)")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(destination, dpi=180, bbox_inches="tight")
        return ax

    def visualize_3d(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (20, 10),
    ) -> Any:
        """Create the required depth slices, sections, 3D volume, and bedrock map."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import PowerNorm

        if self.field is None or self.bedrock_depth is None:
            raise ValueError("Generate the field before visualizing it.")

        fig = plt.figure(figsize=figsize, constrained_layout=True)
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
        x_idx = np.unique(np.linspace(0, self.grid.nx - 1, min(self.grid.nx, 31), dtype=int))
        y_idx = np.unique(np.linspace(0, self.grid.ny - 1, min(self.grid.ny, 31), dtype=int))
        z_idx = np.unique(np.linspace(0, self.grid.nz - 1, min(self.grid.nz, 17), dtype=int))
        x = self.grid.x_coords[x_idx]
        y = self.grid.y_coords[y_idx]
        z_indices = z_idx[::-1]
        z = self.grid.z_coords[z_indices]
        cmap = plt.get_cmap("viridis")
        norm = PowerNorm(gamma=0.45, vmin=float(self.field.min()), vmax=float(self.field.max()))

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
        volume_ax.set_box_aspect((1.0, 1.0, 0.35))
        volume_ax.view_init(elev=24, azim=-55)

        bedrock_ax = axes[1][3]
        x_mesh, y_mesh = np.meshgrid(self.grid.x_coords, self.grid.y_coords, indexing="ij")
        bedrock_ax.plot_surface(
            x_mesh,
            y_mesh,
            -self.bedrock_depth,
            cmap="terrain",
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
        bedrock_ax.set_box_aspect((1.0, 1.0, 0.25))
        bedrock_ax.view_init(elev=24, azim=-55)

        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=[*axes[0], axes[1][0], axes[1][1], volume_ax],
            shrink=0.78,
            pad=0.02,
        )
        colorbar.set_label(f"{self.property_name} ({self.unit})")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=180, bbox_inches="tight")
        return fig

    def visualize(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (20, 10),
    ) -> Any:
        """Backward-compatible alias for :meth:`visualize_3d`."""
        return self.visualize_3d(output_path=output_path, figsize=figsize)

    def __repr__(self) -> str:
        return (
            f"PropertyField3D(nx={self.grid.nx}, ny={self.grid.ny}, nz={self.grid.nz}, "
            f"property={self.property_name} [{self.unit}], range={self.property_range}, "
            f"lenses={self.lens_count}, seed={self.seed})"
        )
