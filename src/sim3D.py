# src/sim3D.py
import numpy as np
import gempy as gp
from pathlib import Path

from configuration import load_config_file, project_path
from grid3d import Grid3D


def _build_geomodel(
    path_to_orientations,
    path_to_surface_points,
    resolution,
    gempy_config=None,
):
    settings = load_config_file("gempy/default.yaml") if gempy_config is None else gempy_config
    geo_model = gp.create_geomodel(
        extent=settings["extent"],
        resolution=resolution,
        refinement=int(settings["refinement"]),
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=str(path_to_orientations),
            path_to_surface_points=str(path_to_surface_points),
            hash_surface_points=None,
            hash_orientations=None,
        ),
    )
    gp.compute_model(geo_model)
    return geo_model


def run_sim2d(path_to_orientations, path_to_surface_points, gempy_config=None):
    settings = load_config_file("gempy/default.yaml") if gempy_config is None else gempy_config
    geo_model = _build_geomodel(
        path_to_orientations,
        path_to_surface_points,
        settings["legacy_2d_resolution"],
        settings,
    )
    resolution = geo_model.grid.regular_grid.resolution
    lith_block_3d = geo_model.solutions.raw_arrays.lith_block.reshape(resolution)

    y_index = int(settings["legacy_2d_y_index"])
    htmp_2d = lith_block_3d[:, y_index, :]   # (100, 10)
    htmp_2d_rot = np.rot90(htmp_2d)          # (10, 100)

    return htmp_2d_rot


def run_sim3d(
    path_to_orientations,
    path_to_surface_points,
    nx=None,
    ny=None,
    nz=None,
    grid_config=None,
    gempy_config=None,
):
    grid_settings = load_config_file("grid/site.yaml") if grid_config is None else grid_config
    gempy_settings = load_config_file("gempy/default.yaml") if gempy_config is None else gempy_config
    grid_settings = dict(grid_settings)
    if nx is not None:
        grid_settings["nx"] = nx
    if ny is not None:
        grid_settings["ny"] = ny
    if nz is not None:
        grid_settings["nz"] = nz
    grid = Grid3D(**grid_settings)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    resolution = [nx, ny, nz]
    geo_model = _build_geomodel(
        path_to_orientations,
        path_to_surface_points,
        resolution,
        gempy_settings,
    )
    resolution = geo_model.grid.regular_grid.resolution
    lith_block_3d = geo_model.solutions.raw_arrays.lith_block.reshape(resolution)
    lith_block_3d = np.asarray(lith_block_3d)

    if lith_block_3d.shape != (nx, ny, nz):
        lith_block_3d = np.reshape(lith_block_3d, (nx, ny, nz))

    return lith_block_3d, grid


def save_3d_preview(output_path=None, nx=None, ny=None, nz=None, grid_config=None, gempy_config=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from configuration import load_config_file

    grid_settings = load_config_file("grid/site.yaml") if grid_config is None else grid_config
    plot_settings = load_config_file("visualization/default.yaml")
    field, grid = run_sim3d(
        project_path("data/orientations.csv"),
        project_path("data/surface_points.csv"),
        nx=nx,
        ny=ny,
        nz=nz,
        grid_config=grid_settings,
        gempy_config=gempy_config,
    )

    x_grid, y_grid, z_grid = np.meshgrid(grid.x_coords, grid.y_coords, grid.z_coords, indexing="ij")
    voxels = field > 0

    xs = x_grid[voxels]
    ys = y_grid[voxels]
    zs = z_grid[voxels]
    colors = field[voxels]

    fig = plt.figure(figsize=tuple(plot_settings["gempy_preview_figsize"]))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xs,
        ys,
        zs,
        c=colors,
        cmap=plot_settings["cmap"],
        s=int(plot_settings["gempy_point_size"]),
        alpha=float(plot_settings["gempy_point_alpha"]),
    )
    ax.set_title("3D GemPy geological environment")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_zlabel("Z coordinate (m, surface at z=0)")
    ax.set_xlim(grid.x_min, grid.x_max)
    ax.set_ylim(grid.y_min, grid.y_max)
    ax.set_zlim(float(np.min(grid.z_coords)), float(np.max(grid.z_coords)))
    ax.set_box_aspect(plot_settings["gempy_box_aspect"])
    ax.view_init(
        elev=int(plot_settings["gempy_elevation"]),
        azim=int(plot_settings["gempy_azimuth"]),
    )

    if output_path is None:
        output_path = project_path("environment_visualisation/3d_environment_preview.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=int(plot_settings["preview_dpi"]))
    plt.close(fig)

    print(f"Saved 3D preview to {output_path}")
    return output_path


if __name__ == "__main__":
    save_3d_preview()
