from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp434_physics_candidate_public_lb_audit"
PACKAGE_DIR = Path(__file__).resolve().parent


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_config() -> dict[str, Any]:
    return read_yaml(PACKAGE_DIR / "config.yaml")


def require_kaggle_run_approval() -> None:
    execution = load_config().get("execution", {})
    required = (
        "canonical_notebook_adoption_approved",
        "kaggle_package_approved",
        "kaggle_push_approved",
        "kaggle_run_approved",
        "run_inference",
        "create_submission",
    )
    missing = [key for key in required if not bool(execution.get(key))]
    if missing:
        raise RuntimeError(
            "exp434 implementation exists, but Kaggle run is not approved: "
            + ", ".join(missing)
        )
