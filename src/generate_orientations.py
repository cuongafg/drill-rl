import hydra
import numpy as np
from omegaconf import DictConfig

from configuration import as_dict, project_path
from generate_geology_csvs import write_orientations


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    settings = as_dict(cfg.gempy)
    write_orientations(
        project_path(str(cfg.paths.orientations)),
        np.random.default_rng(int(cfg.seed)),
        n_points=int(settings["orientation_points"]),
        formations=settings["formations"],
        extent=settings["extent"],
        margin=float(settings["point_margin"]),
        azimuth_range=settings["azimuth_range"],
        dip_range=settings["dip_range"],
        polarity_values=settings["polarity_values"],
    )


if __name__ == "__main__":
    main()
