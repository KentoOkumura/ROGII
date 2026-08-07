from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp301_gauge_invariant_multiformation_edge_potential"
PACKAGE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_defaults(project: dict[str, Any]) -> dict[str, Any]:
    data = project.get("data", {})
    defaults = project.get("defaults", {})
    submission = project.get("submission", {})
    target_columns = submission.get("target_columns") or []
    return {
        "validation": {
            "strategy": defaults.get("primary_validation"),
            "metric": defaults.get("metric"),
            "seed": defaults.get("seed"),
            "n_folds": defaults.get("n_folds"),
            "group_column": data.get("group_column"),
            "score_rows": data.get("score_rows"),
        },
        "data": {
            "raw_dir": data.get("raw_dir"),
            "train_dir": data.get("train_dir"),
            "test_dir": data.get("test_dir"),
            "target_column": data.get("target_column"),
            "id_column": submission.get("id_column"),
            "sample_submission": submission.get("sample_file"),
            "submission_target_column": target_columns[0] if target_columns else None,
        },
    }


def load_config() -> dict[str, Any]:
    project = read_yaml(ROOT / "project.yml")
    experiment = read_yaml(PACKAGE_DIR / "config.yaml")
    return deep_merge(project_defaults(project), experiment)


def require_implementation_authorized(config: dict[str, Any] | None = None) -> None:
    active = config or load_config()
    execution = active.get("execution", {})
    if not execution.get("implementation_authorized", False):
        raise RuntimeError(
            "exp301 solver implementation requires an explicit user request."
        )


def require_kaggle_execution_authorized(config: dict[str, Any] | None = None) -> None:
    active = config or load_config()
    execution = active.get("execution", {})
    if not execution.get("kaggle_execution_authorized", False):
        raise RuntimeError(
            "exp301 is implemented but Kaggle execution remains separately gated."
        )


CONFIG = load_config()
