from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "conf"


def as_dict(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Resolve a Hydra/OmegaConf node to ordinary Python containers."""
    resolved = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    if not isinstance(resolved, dict):
        raise TypeError("Expected a mapping-style configuration.")
    return resolved


def project_path(path: str | Path) -> Path:
    """Resolve configured relative paths from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config_section(section: str) -> dict[str, Any]:
    """Load a default YAML section for programmatic APIs used outside Hydra CLIs."""
    return load_config_file(f"{section}/default.yaml")


def load_config_file(relative_path: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / relative_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return as_dict(OmegaConf.load(config_path))
