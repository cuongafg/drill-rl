from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class ObservationBuilder:
    """Base class for turning drill state into agent observations."""

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int]) -> np.ndarray | dict[str, Any]:
        raise NotImplementedError


@dataclass
class LegacyObservationBuilder(ObservationBuilder):
    """Build the compact scalar observation used by the current RL baseline.

    This matches the project's historically working observation shape and keeps
    the legacy training behavior valid while the codebase grows more modular.
    """

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int]) -> np.ndarray:
        if field.ndim == 3:
            slice_2d = field[position, :, :]
            mean_ic = float(slice_2d.mean())
            std_ic = float(slice_2d.std())
            return np.array([position, mean_ic, std_ic], dtype=np.float32)

        col = field[:, position]
        mean_ic = float(col.mean())
        std_ic = float(col.std())
        return np.array([position, mean_ic, std_ic], dtype=np.float32)


@dataclass
class LocalPatchObservationBuilder(ObservationBuilder):
    """Build a partially observed local patch for CNN-based policies.

    The patch only contains values from already drilled locations. Unknown cells
    are masked with the configured `unknown_value` so the policy never receives the
    hidden full geology field as a direct input.
    """

    context_radius: int = 2
    unknown_value: float = 0.0

    def __post_init__(self) -> None:
        if self.context_radius < 0:
            raise ValueError("context_radius must be non-negative.")

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int]) -> dict[str, Any]:
        if field.ndim != 3:
            raise ValueError("LocalPatchObservationBuilder expects a 3D field array.")

        x_start = max(0, position - self.context_radius)
        x_stop = min(field.shape[0], position + self.context_radius + 1)
        patch = np.full((x_stop - x_start, *field.shape[1:]), self.unknown_value, dtype=np.float32)
        mask = np.zeros_like(patch, dtype=np.float32)

        for drilled_x in drilled_columns:
            if not (x_start <= drilled_x < x_stop):
                continue
            relative_x = drilled_x - x_start
            patch[relative_x, :, :] = field[drilled_x, :, :]
            mask[relative_x, :, :] = 1.0

        return {
            "patch": patch,
            "mask": mask,
            "position": int(position),
            "context_radius": int(self.context_radius),
            "shape": patch.shape,
        }

    def as_tensor(self, field: np.ndarray, position: int, drilled_columns: Sequence[int]) -> np.ndarray:
        data = self.build(field, position, drilled_columns)
        patch = data["patch"]
        mask = data["mask"]
        return np.stack([patch, mask], axis=0).astype(np.float32)


BUILDER_TYPES = {
    "legacy": LegacyObservationBuilder,
    "local_patch": LocalPatchObservationBuilder,
}


def build_observation_builder(mode: str, **kwargs: Any) -> ObservationBuilder:
    """Resolve an observation builder from Hydra config or code."""
    builder_type = BUILDER_TYPES.get(str(mode).lower())
    if builder_type is None:
        raise ValueError(f"Unsupported observation builder mode: '{mode}'")

    if builder_type is LegacyObservationBuilder:
        return builder_type()
    if builder_type is LocalPatchObservationBuilder:
        return builder_type(
            context_radius=int(kwargs.get("context_radius", 2)),
            unknown_value=float(kwargs.get("unknown_value", 0.0)),
        )

    return builder_type(**kwargs)
