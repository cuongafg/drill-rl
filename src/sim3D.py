# src/sim3D.py
import os
import numpy as np
import gempy as gp

from grid3d import Grid3D


def _build_geomodel(path_to_orientations, path_to_surface_points, resolution):
    geo_model = gp.create_geomodel(
        extent=[0, 100, 0, 100, 0, 10],
        resolution=resolution,
        refinement=1,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_orientations,
            path_to_surface_points=path_to_surface_points,
            hash_surface_points=None,
            hash_orientations=None,
        ),
    )
    gp.compute_model(geo_model)
    return geo_model


def run_sim2d(path_to_orientations, path_to_surface_points):
    geo_model = _build_geomodel(path_to_orientations, path_to_surface_points, [100, 1, 10])
    resolution = geo_model.grid.regular_grid.resolution
    lith_block_3d = geo_model.solutions.raw_arrays.lith_block.reshape(resolution)

    y_index = 0
    htmp_2d = lith_block_3d[:, y_index, :]   # (100, 10)
    htmp_2d_rot = np.rot90(htmp_2d)          # (10, 100)

    return htmp_2d_rot


def run_sim3d(path_to_orientations, path_to_surface_points, nx=101, ny=101, nz=11):
    grid = Grid3D(nx=nx, ny=ny, nz=nz, x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0, z_min=0.0, z_max=-10.0)
    geo_model = _build_geomodel(path_to_orientations, path_to_surface_points, [nx, ny, nz])
    resolution = geo_model.grid.regular_grid.resolution
    lith_block_3d = geo_model.solutions.raw_arrays.lith_block.reshape(resolution)
    lith_block_3d = np.asarray(lith_block_3d)

    if lith_block_3d.shape != (nx, ny, nz):
        lith_block_3d = np.reshape(lith_block_3d, (nx, ny, nz))

    return lith_block_3d, grid


def save_3d_preview(output_path=None, nx=101, ny=101, nz=11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    field, grid = run_sim3d(
        os.path.join(BASE_DIR, "..", "data", "orientations.csv"),
        os.path.join(BASE_DIR, "..", "data", "surface_points.csv"),
        nx=nx,
        ny=ny,
        nz=nz,
    )

    x_grid, y_grid, z_grid = np.meshgrid(grid.x_coords, grid.y_coords, grid.z_coords, indexing="ij")
    voxels = field > 0

    xs = x_grid[voxels]
    ys = y_grid[voxels]
    zs = z_grid[voxels]
    colors = field[voxels]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=colors, cmap="viridis", s=8, alpha=0.8)
    ax.set_title("3D GemPy geological environment")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_zlabel("Z coordinate (m, surface at z=0)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(-10, 0)
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=28, azim=-45)

    if output_path is None:
        output_path = os.path.join(BASE_DIR, "..", "data", "3d_environment_preview.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved 3D preview to {output_path}")
    return output_path


if __name__ == "__main__":
    save_3d_preview()
