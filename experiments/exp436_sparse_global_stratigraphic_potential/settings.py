"""Execution-state helpers for exp436.

The scientific implementation lives in the compact self-contained Jupytext
source. Kaggle execution remains separately authorization-gated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp436_sparse_global_stratigraphic_potential"
IMPLEMENTATION_STATUS = "stage0_implemented_unrun"


def experiment_dir() -> Path:
    return Path.cwd() / "experiments" / EXPERIMENT_NAME


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = experiment_dir() / "config.yaml" if path is None else path
    with selected.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def require_stage0_execution_authorization(
    config: dict[str, Any] | None = None,
) -> None:
    value = load_config() if config is None else config
    required = (
        bool(value["runtime"]["run_approved"]),
        bool(value["authorization"]["canonical_train_notebook_adopted"]),
        bool(value["authorization"]["kaggle_package_authorized"]),
        bool(value["authorization"]["kaggle_push_authorized"]),
        bool(value["authorization"]["kaggle_execution_authorized"]),
        bool(value["authorization"]["stage0_run_authorized"]),
    )
    if not all(required):
        raise RuntimeError(
            "exp436 Stage 0 is implemented but Kaggle execution remains locked"
        )
