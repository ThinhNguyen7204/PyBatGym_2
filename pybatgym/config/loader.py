"""Configuration loader with YAML support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from pybatgym.config.base_config import PyBatGymConfig

_DEFAULT_CONFIG_PATHS = [
    Path("configs/default.yaml"),
    Path("config.yaml"),
]


def load_config(path: Optional[str | Path] = None) -> PyBatGymConfig:
    """Load configuration from YAML file or return defaults.

    Args:
        path: Path to YAML config file. If None, searches default locations.

    Returns:
        Validated PyBatGymConfig instance.
    """
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return _parse_yaml(config_path)

    for default_path in _DEFAULT_CONFIG_PATHS:
        if default_path.exists():
            return _parse_yaml(default_path)

    return PyBatGymConfig()


def _parse_yaml(path: Path) -> PyBatGymConfig:
    """Parse YAML file into PyBatGymConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return PyBatGymConfig()

    return PyBatGymConfig.model_validate(raw)


def load_preset(name: str) -> PyBatGymConfig:
    """Load a named preset config from configs/ directory.

    Example:
        config = load_preset("small_batsim")  # → configs/small_batsim.yaml
    """
    preset_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    preset_path = preset_dir / f"{name}.yaml"
    if not preset_path.exists():
        available = [p.stem for p in preset_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Preset '{name}' not found at {preset_path}. "
            f"Available: {available}"
        )
    return _parse_yaml(preset_path)
