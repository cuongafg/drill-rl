# src/soil_field.py
import numpy as np

from configuration import load_config_file, project_path
from generate_geology_csvs import write_orientations, write_surface_points
from sim2D import run_sim2d, run_sim3d


def generate_soil_field(
    seed=None,
    max_retries=3,
    mode="2d",
    grid_config=None,
    gempy_config=None,
    paths_config=None,
):
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

    grid_settings = (
        load_config_file("grid/site.yaml")
        if grid_config is None
        else grid_config
    )
    gempy_settings = (
        load_config_file("gempy/default.yaml")
        if gempy_config is None
        else gempy_config
    )
    paths_settings = (
        load_config_file("paths/default.yaml")
        if paths_config is None
        else paths_config
    )
    orientations_path = project_path(paths_settings["orientations"])
    surface_points_path = project_path(paths_settings["surface_points"])
    orientations_path.parent.mkdir(parents=True, exist_ok=True)
    surface_points_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        rng = np.random.default_rng(None if seed is None else seed + attempt * 1000)

        csv_settings = {
            "formations": gempy_settings["formations"],
            "extent": gempy_settings["extent"],
            "margin": gempy_settings["point_margin"],
        }
        write_orientations(
            orientations_path,
            rng,
            n_points=int(gempy_settings["orientation_points"]),
            **csv_settings,
            azimuth_range=gempy_settings["azimuth_range"],
            dip_range=gempy_settings["dip_range"],
            polarity_values=gempy_settings["polarity_values"],
        )
        write_surface_points(
            surface_points_path,
            rng,
            n_points=int(gempy_settings["surface_points"]),
            **csv_settings,
        )

        try:
            if mode == "2d":
                return run_sim2d(
                    orientations_path,
                    surface_points_path,
                    gempy_config=gempy_settings,
                )
            field, _ = run_sim3d(
                orientations_path,
                surface_points_path,
                grid_config=grid_settings,
                gempy_config=gempy_settings,
            )
            return field
        except Exception as e:
            print(f"[soil_field] seed={seed} mode={mode} attempt={attempt} failed: {e}")

    raise RuntimeError(
        f"generate_soil_field failed after {max_retries} attempts "
        f"(seed={seed}, mode={mode})"
    )
