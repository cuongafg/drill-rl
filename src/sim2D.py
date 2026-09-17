# src/sim2D.py
import os
import numpy as np
import gempy as gp


def run_sim2d(path_to_orientations, path_to_surface_points):
    geo_model = gp.create_geomodel(
        extent=[0, 100, 0, 100, 0, 10],
        resolution=[100, 1, 10],
        refinement=1,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_orientations,
            path_to_surface_points=path_to_surface_points,
            hash_surface_points=None,
            hash_orientations=None,
        ),
    )
    sol = gp.compute_model(geo_model)

    resolution = geo_model.grid.regular_grid.resolution
    lith_block_3d = sol.raw_arrays.lith_block.reshape(resolution)

    y_index = 0
    htmp_2d = lith_block_3d[:, y_index, :]   # (100, 10)
    htmp_2d_rot = np.rot90(htmp_2d)          # (10, 100)

    return htmp_2d_rot


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    field = run_sim2d(
        os.path.join(BASE_DIR, "..", "data", "orientations.csv"),
        os.path.join(BASE_DIR, "..", "data", "surface_points.csv"),
    )
    plt.imshow(field, cmap="viridis", aspect="auto")
    plt.colorbar(label="Lithology / IC value")
    plt.title("2D soil field slice (static CSVs)")
    plt.show()
