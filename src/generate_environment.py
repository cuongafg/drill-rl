from __future__ import annotations

import hydra
from omegaconf import DictConfig

from configuration import as_dict, project_path
from property_field import PropertyField3D
from property_field_visualizer import PropertyFieldVisualizer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    field = PropertyField3D(
        **as_dict(cfg.grid),
        seed=int(cfg.seed),
        geology_config=as_dict(cfg.geology),
    )
    field.generate()
    field_path = project_path(str(cfg.paths.ground_truth))
    preview_path = project_path(str(cfg.paths.preview))
    field.save(field_path)
    PropertyFieldVisualizer(field, as_dict(cfg.visualization)).visualize_3d(preview_path)
    print(f"Saved ground truth to {field_path}")
    print(f"Saved visualization to {preview_path}")


if __name__ == "__main__":
    main()
