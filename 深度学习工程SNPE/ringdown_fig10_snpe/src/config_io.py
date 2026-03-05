from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def project_root_from_config(config_path: Path) -> Path:
    # Expected: <project_root>/configs/fig10.yaml
    return config_path.resolve().parents[1]

