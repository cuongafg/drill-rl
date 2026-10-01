# src/soil_field.py
import os
import numpy as np

from generate_geology_csvs import write_orientations, write_surface_points
from sim2D import run_sim2d, run_sim3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIENTATIONS_PATH = os.path.join(BASE_DIR, "..", "data", "orientations.csv")
SURFACE_POINTS_PATH = os.path.join(BASE_DIR, "..", "data", "surface_points.csv")


def generate_soil_field(seed=None, max_retries=3, mode="2d"):
    """
    Generates a fresh, seeded random geology via GemPy.

    mode can be either "2d" (legacy behaviour: returns a single 10x100 slice)
    or "3d" (returns the full volumetric GemPy field).

    max_retries: with only 3 random orientation points across 4 possible
    formations, some seeds may leave a formation with no orientation
    data, which can make GemPy's solve degenerate or fail. Retries with
    a shifted seed rather than crashing the whole run on one bad draw.
    """
    if mode not in {"2d", "3d"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use '2d' or '3d'.")

    for attempt in range(max_retries):
        rng = np.random.default_rng(None if seed is None else seed + attempt * 1000)

        write_orientations(ORIENTATIONS_PATH, rng)
        write_surface_points(SURFACE_POINTS_PATH, rng)

        try:
            if mode == "2d":
                return run_sim2d(ORIENTATIONS_PATH, SURFACE_POINTS_PATH)
            return run_sim3d(ORIENTATIONS_PATH, SURFACE_POINTS_PATH)
        except Exception as e:
            print(f"[soil_field] seed={seed} mode={mode} attempt={attempt} failed: {e}")

    raise RuntimeError(
        f"generate_soil_field failed after {max_retries} attempts "
        f"(seed={seed}, mode={mode})"
    )
