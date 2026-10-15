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

    drilled_cfg = getattr(cfg, "drilled_holes", None)
    if drilled_cfg is None and hasattr(cfg, "environment"):
        drilled_cfg = getattr(cfg.environment, "drilled_holes", None)

    if drilled_cfg is None:
        raise ValueError("No drilled_holes config was found under cfg.drilled_holes or cfg.environment.drilled_holes.")

    positions = [
        (float(x), float(y))
        for x, y in drilled_cfg.positions
    ]
    output_path = project_path(str(drilled_cfg.output))

    PropertyFieldVisualizer(field, as_dict(cfg.visualization)).plot_drilled_holes(
        drilled_positions=positions,
        z=float(drilled_cfg.z),
        output_path=output_path,
    )
    print(f"Saved drilled-hole preview to {output_path}")
    print(f"Hole positions: {positions}")


if __name__ == "__main__":
    main()
