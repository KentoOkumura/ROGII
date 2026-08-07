from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation"
PACKAGE_DIR = Path.cwd()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        *[
            parent / "experiments" / EXPERIMENT_NAME / "config.yaml"
            for parent in PACKAGE_DIR.parents
        ],
    ]
    for path in candidates:
        if not path.exists():
            continue
        value: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"Could not resolve config.yaml for {EXPERIMENT_NAME}")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value
