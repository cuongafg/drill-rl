from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from configuration import as_dict, project_path
from soil_field import generate_soil_field


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    field_pool = cfg.field_pool
    fields = []

    for index in range(int(field_pool.count)):
        seed = int(field_pool.seed_start) + index
        print(
            f"Generating field {index + 1}/{field_pool.count} "
            f"(seed={seed}, mode={field_pool.mode})"
        )
        fields.append(
            generate_soil_field(
                seed=seed,
                mode=str(field_pool.mode),
                max_retries=int(field_pool.max_retries),
                grid_config=as_dict(cfg.grid),
                gempy_config=as_dict(cfg.gempy),
                paths_config=as_dict(cfg.paths),
            )
        )

    pool = np.stack(fields)
    configured_output = field_pool.output
    if configured_output is None:
        configured_output = (
            field_pool.output_3d
            if str(field_pool.mode) == "3d"
            else field_pool.output_2d
        )
    output_path = project_path(str(configured_output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pool)
    print(f"Saved {pool.shape} to {output_path}")


if __name__ == "__main__":
    main()
