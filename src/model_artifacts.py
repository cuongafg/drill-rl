from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from configuration import project_path


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def create_model_run_id() -> str:
    """Create a sortable, collision-resistant identifier for a training run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def model_run_directory(model_dir: str, algorithm: str, run_id: str) -> Path:
    """Resolve a model run directory while rejecting path-like identifiers."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid model run ID: {run_id!r}")
    if not RUN_ID_PATTERN.fullmatch(algorithm):
        raise ValueError(f"Invalid model algorithm: {algorithm!r}")
    return project_path(model_dir) / algorithm.lower() / run_id


def model_checkpoint_filename(model_name: str) -> str:
    """Return a safe Stable-Baselines3 archive filename for a configured name."""
    if not RUN_ID_PATTERN.fullmatch(model_name) or model_name.endswith(".zip"):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return f"{model_name}.zip"
