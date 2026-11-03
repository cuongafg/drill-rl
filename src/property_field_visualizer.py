from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from configuration import load_config_file
from property_field import PropertyField3D


class PropertyFieldVisualizer:
    """Render slices and 3D views of a generated ``PropertyField3D``."""

    def __init__(
        self,
        field: PropertyField3D,
        visualization_config: dict[str, Any] | None = None,
    ) -> None:
        self.field_model = field
        self.config = (
            load_config_file("visualization/default.yaml")
            if visualization_config is None
            else visualization_config
        )

    def _require_field(self) -> np.ndarray:
        if self.field_model.field is None:
            raise ValueError("The field has not been generated yet. Call generate() first.")
        return self.field_model.field

    def _save(self, figure: Any, output_path: str | Path | None) -> None:
        if output_path is None:
            return
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            destination,
            dpi=int(self.config["preview_dpi"]),
            bbox_inches="tight",
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

        field = self._require_field()
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.config["slice_figsize"]))
        iz = int(np.argmin(np.abs(self.field_model.grid.z_coords - z)))
        image = ax.imshow(
            field[:, :, iz].T,
            origin="lower",
            aspect="equal",
            cmap=self.config["cmap"],
            norm=PowerNorm(
                gamma=float(self.config["qc_gamma"]),
                vmin=float(field.min()),
                vmax=float(field.max()),
            ),
            extent=[
                self.field_model.grid.x_min,
                self.field_model.grid.x_max,
                self.field_model.grid.y_min,
                self.field_model.grid.y_max,
            ],
        )
        ax.set(
            title=f"qc at z={self.field_model.grid.z_coords[iz]:.2f} m",
            xlabel="X (m)",
            ylabel="Y (m)",
        )
        if own_figure:
            ax.figure.colorbar(
                image,
                ax=ax,
                label=f"{self.field_model.property_name} ({self.field_model.unit})",
            )
        self._save(ax.figure, output_path)
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

        field = self._require_field()
        grid = self.field_model.grid
        plane = plane.lower()
        if plane not in {"xz", "yz"}:
            raise ValueError("plane must be 'xz' or 'yz'.")
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.config["section_figsize"]))
        if plane == "xz":
            coordinate = float(grid.y_coords[grid.ny // 2] if coordinate is None else coordinate)
            iy = int(np.argmin(np.abs(grid.y_coords - coordinate)))
            section = field[:, iy, :].T
            horizontal_coords = grid.x_coords
            horizontal_label = "X (m)"
            actual_coordinate = grid.y_coords[iy]
        else:
            coordinate = float(grid.x_coords[grid.nx // 2] if coordinate is None else coordinate)
            ix = int(np.argmin(np.abs(grid.x_coords - coordinate)))
            section = field[ix, :, :].T
            horizontal_coords = grid.y_coords
            horizontal_label = "Y (m)"
            actual_coordinate = grid.x_coords[ix]

        image = ax.imshow(
            section,
            origin="upper",
            aspect="auto",
            cmap=self.config["cmap"],
            norm=PowerNorm(
                gamma=float(self.config["qc_gamma"]),
                vmin=float(field.min()),
                vmax=float(field.max()),
            ),
            extent=[
                float(horizontal_coords[0]),
                float(horizontal_coords[-1]),
                float(np.min(grid.z_coords)),
                float(np.max(grid.z_coords)),
            ],
        )
        ax.set(xlabel=horizontal_label, ylabel="Z (m)")
        ax.set_title(f"{plane.upper()} section at {actual_coordinate:.1f} m")
        if own_figure:
            ax.figure.colorbar(
                image,
                ax=ax,
                label=f"{self.field_model.property_name} ({self.field_model.unit})",
            )
        self._save(ax.figure, output_path)
        return ax

    def plot_bedrock_surface(
        self,
        ax: Any | None = None,
        output_path: str | Path | None = None,
    ) -> Any:
        """Plot the undulating bedrock elevation (negative z, in metres)."""
        import matplotlib.pyplot as plt

        if self.field_model.bedrock_depth is None:
            raise ValueError("Generate the field before plotting its bedrock surface.")
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.config["slice_figsize"]))
        image = ax.imshow(
            -self.field_model.bedrock_depth.T,
            origin="lower",
            aspect="equal",
            cmap=self.config["bedrock_cmap"],
            extent=[
                self.field_model.grid.x_min,
                self.field_model.grid.x_max,
                self.field_model.grid.y_min,
                self.field_model.grid.y_max,
            ],
        )
        ax.set(title="Bedrock surface elevation", xlabel="X (m)", ylabel="Y (m)")
        if own_figure:
            ax.figure.colorbar(image, ax=ax, label="Bedrock Z (m)")
        self._save(ax.figure, output_path)
        return ax

    def plot_drilled_holes(
        self,
        drilled_positions: Sequence[tuple[float, float]] | None = None,
        z: float | None = None,
        ax: Any | None = None,
        output_path: str | Path | None = None,
        as_indices: bool = False,
    ) -> Any:
        """Plot a qc slice with drilled x-y locations highlighted."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        field = self._require_field()
        grid = self.field_model.grid
        if drilled_positions is None:
            drilled_positions = [
                (18.0, 22.0),
                (32.0, 68.0),
                (58.0, 42.0),
                (72.0, 76.0),
                (84.0, 30.0),
            ]
        z_to_plot = float(grid.z_coords[0] if z is None else z)
        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=tuple(self.config["slice_figsize"]))
        iz = int(np.argmin(np.abs(grid.z_coords - z_to_plot)))
        image = ax.imshow(
            field[:, :, iz].T,
            origin="lower",
            aspect="equal",
            cmap=self.config["cmap"],
            norm=PowerNorm(
                gamma=float(self.config["qc_gamma"]),
                vmin=float(field.min()),
                vmax=float(field.max()),
            ),
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        )

        x_points: list[float] = []
        y_points: list[float] = []
        for position in drilled_positions:
            if len(position) != 2:
                raise ValueError("Each drilled position must contain exactly (x, y).")
            x_world, y_world = float(position[0]), float(position[1])
            if as_indices:
                ix, iy = int(round(x_world)), int(round(y_world))
                if not (0 <= ix < grid.nx and 0 <= iy < grid.ny):
                    raise ValueError(f"Drilled index ({ix}, {iy}) is outside the model extent.")
                x_world, y_world = float(grid.x_coords[ix]), float(grid.y_coords[iy])
            elif not (grid.x_min <= x_world <= grid.x_max and grid.y_min <= y_world <= grid.y_max):
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
            title=f"Observed qc map at z={grid.z_coords[iz]:.2f} m with drilled holes",
            xlabel="X (m)",
            ylabel="Y (m)",
        )
        if own_figure:
            ax.figure.colorbar(
                image,
                ax=ax,
                label=f"{self.field_model.property_name} ({self.field_model.unit})",
            )
        self._save(ax.figure, output_path)
        return ax

    def visualize_3d(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Create depth slices, sections, a 3D volume, and a bedrock map."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import PowerNorm

        field = self._require_field()
        if self.field_model.bedrock_depth is None:
            raise ValueError("Generate the field before visualizing it.")
        grid = self.field_model.grid
        figure_size = tuple(self.config["preview_figsize"]) if figsize is None else figsize
        fig = plt.figure(figsize=figure_size, constrained_layout=True)
        layout = fig.add_gridspec(2, 4)
        axes = [
            [fig.add_subplot(layout[0, column]) for column in range(4)],
            [
                fig.add_subplot(layout[1, column], projection="3d" if column >= 2 else None)
                for column in range(4)
            ],
        ]
        fig.suptitle(f"3D {self.field_model.property_name.upper()} geological ground truth")

        for ax, level in zip(axes[0][:3], (0.0, -5.0, -10.0)):
            self.plot_horizontal_slice(level, ax=ax)
        self.plot_bedrock_surface(ax=axes[0][3])
        self.plot_vertical_section("xz", ax=axes[1][0])
        self.plot_vertical_section("yz", ax=axes[1][1])

        volume_ax = axes[1][2]
        x_idx = np.unique(
            np.linspace(
                0,
                grid.nx - 1,
                min(grid.nx, int(self.config["volume_max_x_samples"])),
                dtype=int,
            )
        )
        y_idx = np.unique(
            np.linspace(
                0,
                grid.ny - 1,
                min(grid.ny, int(self.config["volume_max_y_samples"])),
                dtype=int,
            )
        )
        z_idx = np.unique(
            np.linspace(
                0,
                grid.nz - 1,
                min(grid.nz, int(self.config["volume_max_z_samples"])),
                dtype=int,
            )
        )
        x = grid.x_coords[x_idx]
        y = grid.y_coords[y_idx]
        z_indices = z_idx[::-1]
        z = grid.z_coords[z_indices]
        cmap = plt.get_cmap(self.config["cmap"])
        norm = PowerNorm(
            gamma=float(self.config["qc_gamma"]),
            vmin=float(field.min()),
            vmax=float(field.max()),
        )

        x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
        top_values = field[np.ix_(x_idx, y_idx, [0])][:, :, 0]
        volume_ax.plot_surface(
            x_mesh,
            y_mesh,
            np.full_like(x_mesh, float(grid.z_coords[0])),
            facecolors=cmap(norm(top_values)),
            shade=False,
            linewidth=0,
        )

        middle_y = y_idx[len(y_idx) // 2]
        section_x_mesh, section_z_mesh = np.meshgrid(x, z, indexing="ij")
        xz_values = field[np.ix_(x_idx, [middle_y], z_indices)][:, 0, :]
        volume_ax.plot_surface(
            section_x_mesh,
            np.full_like(section_x_mesh, grid.y_coords[middle_y]),
            section_z_mesh,
            facecolors=cmap(norm(xz_values)),
            shade=False,
            linewidth=0,
        )

        middle_x = x_idx[len(x_idx) // 2]
        section_y_mesh, section_z_mesh = np.meshgrid(y, z, indexing="ij")
        yz_values = field[np.ix_([middle_x], y_idx, z_indices)][0, :, :]
        volume_ax.plot_surface(
            np.full_like(section_y_mesh, grid.x_coords[middle_x]),
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
            xlim=(grid.x_min, grid.x_max),
            ylim=(grid.y_min, grid.y_max),
            zlim=(float(np.min(grid.z_coords)), float(np.max(grid.z_coords))),
        )
        volume_ax.set_box_aspect(self.config["volume_aspect"])
        volume_ax.view_init(
            elev=int(self.config["elevation"]),
            azim=int(self.config["azimuth"]),
        )

        bedrock_ax = axes[1][3]
        x_mesh, y_mesh = np.meshgrid(grid.x_coords, grid.y_coords, indexing="ij")
        bedrock_ax.plot_surface(
            x_mesh,
            y_mesh,
            -self.field_model.bedrock_depth,
            cmap=self.config["bedrock_cmap"],
            linewidth=0,
            antialiased=True,
        )
        bedrock_ax.set(
            title="3D bedrock surface",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            xlim=(grid.x_min, grid.x_max),
            ylim=(grid.y_min, grid.y_max),
            zlim=(float(np.min(grid.z_coords)), 0.0),
        )
        bedrock_ax.set_box_aspect(self.config["bedrock_aspect"])
        bedrock_ax.view_init(
            elev=int(self.config["elevation"]),
            azim=int(self.config["azimuth"]),
        )

        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=[*axes[0], axes[1][0], axes[1][1], volume_ax],
            shrink=float(self.config["colorbar_shrink"]),
            pad=float(self.config["colorbar_pad"]),
        )
        colorbar.set_label(f"{self.field_model.property_name} ({self.field_model.unit})")
        self._save(fig, output_path)
        return fig

    def visualize(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Alias for :meth:`visualize_3d`."""
        return self.visualize_3d(output_path=output_path, figsize=figsize)
