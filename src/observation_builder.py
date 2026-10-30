from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from gymnasium import spaces
import numpy as np


class ObservationBuilder:
    """Base class for turning drill state into agent observations."""

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int]) -> np.ndarray | dict[str, Any]:
        raise NotImplementedError

    def space(self, field_shape: tuple[int, ...]) -> spaces.Space:
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

    The patch only contains values from already drilled x,y coordinates. Unknown
    cells remain masked so the policy never receives the hidden full geology field
    as a direct input. The current implementation treats each drilled location as a
    specific point in the x-y plane, and reveals the full vertical column beneath it.
    """

    context_radius: int = 2
    unknown_value: float = 0.0

    def __post_init__(self) -> None:
        if self.context_radius < 0:
            raise ValueError("context_radius must be non-negative.")

    def _resolve_drilled_points(
        self,
        field: np.ndarray,
        position: int,
        drilled_columns: Sequence[int | tuple[int, int]],
    ) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for drilled in drilled_columns:
            if isinstance(drilled, tuple):
                x_idx, y_idx = int(drilled[0]), int(drilled[1])
            else:
                x_idx = int(drilled)
                y_idx = field.shape[1] // 2
            points.append((x_idx, y_idx))
        if not points:
            points.append((int(position), field.shape[1] // 2))
        return points

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int | tuple[int, int]]) -> dict[str, Any]:
        if field.ndim != 3:
            raise ValueError("LocalPatchObservationBuilder expects a 3D field array.")

        drilled_points = self._resolve_drilled_points(field, position, drilled_columns)
        x_center = int(position)
        y_center = int(np.median([y for _, y in drilled_points])) if drilled_points else field.shape[1] // 2
        x_start = max(0, x_center - self.context_radius)
        x_stop = min(field.shape[0], x_center + self.context_radius + 1)
        y_start = max(0, y_center - self.context_radius)
        y_stop = min(field.shape[1], y_center + self.context_radius + 1)

        patch = np.full((x_stop - x_start, y_stop - y_start, field.shape[2]), self.unknown_value, dtype=np.float32)
        mask = np.zeros_like(patch, dtype=np.float32)

        for drilled_x, drilled_y in drilled_points:
            if not (x_start <= drilled_x < x_stop and y_start <= drilled_y < y_stop):
                continue
            local_x = drilled_x - x_start
            local_y = drilled_y - y_start
            patch[local_x, local_y, :] = field[drilled_x, drilled_y, :]
            mask[local_x, local_y, :] = 1.0

        return {
            "patch": patch,
            "mask": mask,
            "position": int(position),
            "context_radius": int(self.context_radius),
            "shape": patch.shape,
        }

    def as_tensor(self, field: np.ndarray, position: int, drilled_columns: Sequence[int | tuple[int, int]]) -> np.ndarray:
        data = self.build(field, position, drilled_columns)
        patch = data["patch"]
        mask = data["mask"]
        return np.stack([patch, mask], axis=0).astype(np.float32)


@dataclass
class CnnPatchObservationBuilder(LocalPatchObservationBuilder):
    """CNN-ready observation builder.

    This is the same partial local patch concept as LocalPatchObservationBuilder,
    but it is explicitly intended to be consumed by a convolutional policy. The
    tensor shape is fixed to [2, x_window, y_window, z], where the first channel is
    the observed qc values and the second is the known/unknown mask.
    """

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int | tuple[int, int]]) -> np.ndarray:
        return self.as_tensor(field, position, drilled_columns)


@dataclass
class FullObservedMapObservationBuilder(ObservationBuilder):
    """Build the cumulative observed map of drilled x,y measurements.

    Each drilled location contributes only the ground truth values below that
    exact (x, y) coordinate. The map keeps the hidden full field private while
    revealing all previously observed vertical columns at the chosen x,y points.
    """

    unknown_value: float = 0.0
    value_low: float = 1.0
    value_high: float = 5.0

    def __post_init__(self) -> None:
        if self.value_high <= self.value_low:
            raise ValueError("value_high must be greater than value_low.")

    def space(self, field_shape: tuple[int, ...]) -> spaces.Box:
        if len(field_shape) != 3:
            raise ValueError("FullObservedMapObservationBuilder expects a 3D field shape.")
        return spaces.Box(
            low=np.full((2, *field_shape), self.unknown_value, dtype=np.float32),
            high=np.ones((2, *field_shape), dtype=np.float32),
            dtype=np.float32,
        )

    def build(self, field: np.ndarray, position: int, drilled_columns: Sequence[int | tuple[int, int]]) -> np.ndarray:
        if field.ndim != 3:
            raise ValueError("FullObservedMapObservationBuilder expects a 3D field array.")

        observed = np.full(field.shape, self.unknown_value, dtype=np.float32)
        mask = np.zeros(field.shape, dtype=np.float32)

        points: list[tuple[int, int]] = []
        for drilled in drilled_columns:
            if isinstance(drilled, tuple):
                x_idx, y_idx = int(drilled[0]), int(drilled[1])
            else:
                x_idx = int(drilled)
                y_idx = field.shape[1] // 2
            points.append((x_idx, y_idx))

        for x_idx, y_idx in points:
            values = (
                np.asarray(field[x_idx, y_idx, :], dtype=np.float32) - self.value_low
            ) / (self.value_high - self.value_low)
            observed[x_idx, y_idx, :] = np.clip(values, 0.0, 1.0)
            mask[x_idx, y_idx, :] = 1.0

        return np.stack([observed, mask], axis=0).astype(np.float32)


@dataclass
class MlpObservationBuilder(ObservationBuilder):
    """Flatten a canonical 3D map and append normalized position for an MLP."""

    base_builder: FullObservedMapObservationBuilder
    include_position: bool = True

    def space(self, field_shape: tuple[int, ...]) -> spaces.Box:
        if len(field_shape) != 3:
            raise ValueError("MlpObservationBuilder expects a 3D field shape.")
        size = 2 * int(np.prod(field_shape))
        if self.include_position:
            size += 1
        return spaces.Box(
            low=np.zeros(size, dtype=np.float32),
            high=np.ones(size, dtype=np.float32),
            dtype=np.float32,
        )

    def build(
        self,
        field: np.ndarray,
        position: int,
        drilled_columns: Sequence[int | tuple[int, int]],
    ) -> np.ndarray:
        volume = self.base_builder.build(field, position, drilled_columns)
        values = volume.reshape(-1)
        if not self.include_position:
            return values.astype(np.float32)
        denominator = max(field.shape[0] - 1, 1)
        normalized_position = np.asarray([position / denominator], dtype=np.float32)
        return np.concatenate((values, normalized_position)).astype(np.float32)


BUILDER_TYPES = {
    "legacy": LegacyObservationBuilder,
    "local_patch": LocalPatchObservationBuilder,
    "cnn_patch": CnnPatchObservationBuilder,
    "full_observed_map": FullObservedMapObservationBuilder,
}


def build_observation_builder(mode: str, **kwargs: Any) -> ObservationBuilder:
    """Resolve an observation builder from Hydra config or code."""
    builder_type = BUILDER_TYPES.get(str(mode).lower())
    if builder_type is None:
        raise ValueError(f"Unsupported observation builder mode: '{mode}'")

    if builder_type is LegacyObservationBuilder:
        return builder_type()
    if builder_type in {LocalPatchObservationBuilder, CnnPatchObservationBuilder}:
        return builder_type(
            context_radius=int(kwargs.get("context_radius", 2)),
            unknown_value=float(kwargs.get("unknown_value", 0.0)),
        )
    if builder_type is FullObservedMapObservationBuilder:
        base_builder = builder_type(
            unknown_value=float(kwargs.get("unknown_value", 0.0)),
            value_low=float(kwargs.get("value_low", 1.0)),
            value_high=float(kwargs.get("value_high", 5.0)),
        )
        if str(kwargs.get("representation", "map")).lower() == "mlp":
            return MlpObservationBuilder(
                base_builder=base_builder,
                include_position=bool(kwargs.get("include_position", True)),
            )
        return base_builder

    return builder_type(**kwargs)
