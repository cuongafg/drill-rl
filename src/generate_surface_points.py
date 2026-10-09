import hydra
import numpy as np
from omegaconf import DictConfig

from configuration import as_dict, project_path
from generate_geology_csvs import write_surface_points


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    settings = as_dict(cfg.gempy)
    write_surface_points(
        project_path(str(cfg.paths.surface_points)),
        np.random.default_rng(int(cfg.seed)),
        n_points=int(settings["surface_points"]),
        formations=settings["formations"],
        extent=settings["extent"],
        margin=float(settings["point_margin"]),
    )


if __name__ == "__main__":
    main()
