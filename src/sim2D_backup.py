import numpy as np
import gempy as gp
import gempy_viewer as gpv
import matplotlib.pyplot as plt
from numpy import ma  # For masking
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_orientations = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "orientations.csv"
)

data_surface_points = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "surface_points.csv"
)

# resolution=[100, 1, 10] gives x=100 columns, y=1 slice, z=10 rows,
# matching the paper's 10-row x 100-column field.
# refinement=1 is the default and is ignored once resolution is set,
# but passing resolution alone is what forces a plain dense grid
# instead of GemPy's octree solver (which is what produced the
# earlier "Chunking done" messages and the flat 409600-length array).
geo_model: gp.data.GeoModel = gp.create_geomodel(
    extent=[0, 100, 0, 100, 0, 10],
    resolution=[100, 1, 10],
    refinement=1,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_orientations,
        path_to_surface_points=data_surface_points,
        hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
        hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
    )
)

sol = gp.compute_model(geo_model)

# ---- reshape the flat lith_block back into a 3D grid ----
resolution = geo_model.grid.regular_grid.resolution
print("resolution:", resolution)              # expect [100   1  10]

lith_block_3d = sol.raw_arrays.lith_block.reshape(resolution)
print("3D shape:", lith_block_3d.shape)        # expect (100, 1, 10)

# ---- slice out the single y-transect and orient it for plotting ----
y_index = 0   # only one y-slice exists now, so this must be 0, not 1
htmp_2d = lith_block_3d[:, y_index, :]          # shape (100, 10): columns x depth
htmp_2d_rot = np.rot90(htmp_2d)                 # shape (10, 100): depth x columns, depth on top

print("2D field shape:", htmp_2d_rot.shape)     # expect (10, 100)

# ---- visualize to confirm it looks like a layered soil profile ----
plt.figure(figsize=(10, 3))
plt.imshow(htmp_2d_rot, cmap="viridis", aspect="auto")
plt.colorbar(label="Lithology / IC value")
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.title("2D soil field slice")
plt.tight_layout()
plt.show()