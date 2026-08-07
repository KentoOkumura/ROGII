from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp514_exp413_likpf_seed_bank_reuse_on_exp512"
PACKAGE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_config() -> dict[str, Any]:
    return read_yaml(PACKAGE_DIR / "config.yaml")


class ExperimentPaths:
    """Design-stage paths only; executable orchestration is not implemented."""

    root = ROOT
    experiment_dir = PACKAGE_DIR
    config_path = PACKAGE_DIR / "config.yaml"
    artifacts_dir = PACKAGE_DIR / "artifacts"
    features_dir = PACKAGE_DIR / "features"
    metrics_path = PACKAGE_DIR / "metrics.json"

    @property
    def config(self) -> dict[str, Any]:
        return load_config()
