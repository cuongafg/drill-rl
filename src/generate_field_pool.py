from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig
from tempfile import TemporaryDirectory

from configuration import as_dict, project_path
from property_field import PropertyField3D
from soil_field import generate_soil_field


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    field_pool = cfg.field_pool
    generator = str(field_pool.get("generator", "gempy")).lower()
    if generator not in {"qc", "gempy"}:
        raise ValueError("field_pool.generator must be either 'qc' or 'gempy'.")
    if generator == "qc" and str(field_pool.mode) != "3d":
        raise ValueError("The QC field-pool generator only supports field_pool.mode=3d.")
    paths_config = as_dict(cfg.paths)
    count = int(field_pool.count)
    if count < 1:
        raise ValueError("field_pool.count must be at least 1.")

    if generator == "qc":
        grid = as_dict(cfg.grid)
        field_shape = (int(grid["nx"]), int(grid["ny"]), int(grid["nz"]))
        generate_field = lambda seed: PropertyField3D(
            **grid,
            seed=seed,
            geology_config=as_dict(cfg.geology),
        ).generate()
    else:
        parallel = bool(field_pool.get("parallel", False))

        def generate_field(seed):
            if parallel:
                with TemporaryDirectory(prefix=f"field_{seed}_") as temp_dir:
                    isolated_paths = {
                        **paths_config,
                        "orientations": f"{temp_dir}/orientations.csv",
                        "surface_points": f"{temp_dir}/surface_points.csv",
                    }
                    return generate_soil_field(
                        seed=seed,
                        mode=str(field_pool.mode),
                        max_retries=int(field_pool.max_retries),
                        grid_config=as_dict(cfg.grid),
                        gempy_config=as_dict(cfg.gempy),
                        paths_config=isolated_paths,
                    )
            return generate_soil_field(
                seed=seed,
                mode=str(field_pool.mode),
                max_retries=int(field_pool.max_retries),
                grid_config=as_dict(cfg.grid),
                gempy_config=as_dict(cfg.gempy),
                paths_config=paths_config,
            )

        field_shape = None

    configured_output = field_pool.output
    if configured_output is None:
        if generator == "qc":
            configured_output = field_pool.output_qc
        else:
            configured_output = (
                field_pool.output_3d
                if str(field_pool.mode) == "3d"
                else field_pool.output_2d
            )
    output_path = project_path(str(configured_output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    pool = None
    try:
        for index in range(count):
            seed = int(field_pool.seed_start) + index
            print(
                f"Generating field {index + 1}/{count} "
                f"(seed={seed}, generator={generator})"
            )
            field = np.asarray(generate_field(seed), dtype=np.float32)
            if field_shape is None:
                field_shape = field.shape
            if field.shape != field_shape:
                raise ValueError(
                    f"Generated field shape {field.shape} does not match {field_shape}."
                )
            if pool is None:
                pool = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(count, *field_shape),
                )
            pool[index] = field
        pool.flush()
        del pool
        temporary_path.replace(output_path)
    except Exception:
        if pool is not None:
            del pool
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved {(count, *field_shape)} to {output_path}")


if __name__ == "__main__":
    main()
