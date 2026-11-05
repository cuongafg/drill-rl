from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configuration import project_path


def visualize_field_pool(
    pool_path: str | Path,
    field_number: int,
    output_path: str | Path | None = None,
) -> Path:
    """Save three orthogonal QC views for one 1-based pool field."""
    resolved_pool = project_path(pool_path)
    pool = np.load(resolved_pool, mmap_mode="r")
    if pool.ndim != 4:
        raise ValueError(
            f"Expected a pool with shape (fields, x, y, depth), got {pool.shape}."
        )
    if not 1 <= field_number <= pool.shape[0]:
        raise ValueError(
            f"field_number must be between 1 and {pool.shape[0]}, got {field_number}."
        )

    field = np.asarray(pool[field_number - 1], dtype=np.float32)
    if output_path is None:
        output_path = Path("environment_visualisation/field_pool_preview.png")
    resolved_output = project_path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    x_mid, y_mid, z_mid = (size // 2 for size in field.shape)
    views = (
        (field[:, :, z_mid].T, "Top view (x-y), middle depth", "X", "Y"),
        (field[:, y_mid, :].T, "Vertical slice (x-depth), middle y", "X", "Depth"),
        (field[x_mid, :, :].T, "Vertical slice (y-depth), middle x", "Y", "Depth"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (values, title, xlabel, ylabel) in zip(axes, views):
        image = axis.imshow(
            values,
            origin="upper",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=300.0,
        )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(image, ax=axis, shrink=0.82, label="qc (MPa)")
    fig.suptitle(
        f"{resolved_pool.name} | field {field_number}/{pool.shape[0]} | "
        f"shape={field.shape} | range={field.min():.1f}-{field.max():.1f} MPa"
    )
    fig.savefig(resolved_output, dpi=180)
    plt.close(fig)
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field_number", type=int, help="1-based field number")
    parser.add_argument(
        "--pool",
        default="data/field_pool_train_qc_20.npy",
        help="Pool .npy path relative to the repository root",
    )
    parser.add_argument("--output", help="Output PNG path")
    args = parser.parse_args()
    output = visualize_field_pool(args.pool, args.field_number, args.output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
