from __future__ import annotations

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from omegaconf import DictConfig

from configuration import as_dict, project_path
from observation_builder import FullObservedMapObservationBuilder
from property_field import PropertyField3D


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    field = PropertyField3D(
        **as_dict(cfg.grid),
        seed=int(cfg.seed),
        geology_config=as_dict(cfg.geology),
    )
    field.generate()

    drilled_cfg = getattr(cfg.environment, "drilled_holes", None)
    if drilled_cfg is None and hasattr(cfg, "drilled_holes"):
        drilled_cfg = cfg.drilled_holes
    if drilled_cfg is None and hasattr(cfg.visualization, "drilled_holes"):
        drilled_cfg = cfg.visualization.drilled_holes

    if drilled_cfg is None:
        raise ValueError(
            "No drilled_holes config was found under cfg.environment.drilled_holes, "
            "cfg.drilled_holes, or cfg.visualization.drilled_holes."
        )

    positions = [(float(x), float(y)) for x, y in drilled_cfg.positions]
    default_output = "environment_visualisation/drilled_holes_3d_preview.png"
    configured_output = str(getattr(drilled_cfg, "output", default_output))
    output_path = project_path(default_output if configured_output == "environment_visualisation/drilled_holes_preview.png" else configured_output)

    observation_cfg = getattr(cfg, "environment", None)
    if observation_cfg is not None:
        observation_cfg = getattr(observation_cfg, "observation", None)
    unknown_value = float(getattr(observation_cfg, "unknown_value", 0.0)) if observation_cfg is not None else 0.0
    builder = FullObservedMapObservationBuilder(unknown_value=unknown_value)
    observed = builder.build(
        field.field,
        position=0,
        drilled_columns=positions,
    )
    observed_values = observed[0]
    known_mask = observed[1] > 0.5

    visualization_config = as_dict(cfg.visualization)
    fig = plt.figure(figsize=tuple(visualization_config["preview_figsize"]))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap(visualization_config["cmap"])
    norm = Normalize(vmin=float(field.field.min()), vmax=float(field.field.max()))

    unknown_x, unknown_y, unknown_z = np.where(~known_mask)
    if unknown_x.size:
        ax.scatter(
            field.grid.x_coords[unknown_x],
            field.grid.y_coords[unknown_y],
            field.grid.z_coords[unknown_z],
            c="#d9d9d9",
            s=10,
            alpha=0.18,
            edgecolors="none",
        )

    known_x, known_y, known_z = np.where(known_mask)
    if known_x.size:
        ax.scatter(
            field.grid.x_coords[known_x],
            field.grid.y_coords[known_y],
            field.grid.z_coords[known_z],
            c=observed_values[known_mask],
            cmap=cmap,
            norm=norm,
            s=30,
            alpha=0.9,
            edgecolors="none",
        )

    ax.set(
        title=f"Known qc volume from FullObservedMapObservationBuilder ({len(positions)} drilled points)",
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    ax.set_xlim(field.grid.x_min, field.grid.x_max)
    ax.set_ylim(field.grid.y_min, field.grid.y_max)
    ax.set_zlim(float(np.min(field.grid.z_coords)), float(np.max(field.grid.z_coords)))
    ax.view_init(elev=int(visualization_config["elevation"]), azim=int(visualization_config["azimuth"]))
    ax.set_box_aspect(tuple(visualization_config["volume_aspect"]))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.82, label=f"{field.property_name} ({field.unit})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(visualization_config["preview_dpi"]), bbox_inches="tight")
    print(f"Saved 3D observed-state preview to {output_path}")
    print(f"Hole positions: {positions}")


if __name__ == "__main__":
    main()
