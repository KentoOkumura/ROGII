from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp245_selector_context_parity_on_exp238"
PACKAGE_DIR = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    value = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value
