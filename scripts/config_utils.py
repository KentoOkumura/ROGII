from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = ROOT / "project.yml"
TODO_VALUES = {"", "TODO", "TBD", "FIXME", None}


def load_project_config(path: Path = PROJECT_CONFIG) -> dict[str, Any]:
    with path.open() as fp:
        config = yaml.safe_load(fp) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def is_todo(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() in TODO_VALUES
    try:
        return value in TODO_VALUES
    except TypeError:
        return False


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def first_submission_target(config: dict[str, Any]) -> Any:
    target_columns = get_nested(config, "submission.target_columns")
    if isinstance(target_columns, list) and target_columns:
        return target_columns[0]
    return None


def project_experiment_defaults(project_config: dict[str, Any]) -> dict[str, Any]:
    data_dir = get_nested(project_config, "paths.data_dir") or "data"
    raw_dir = get_nested(project_config, "data.raw_dir") or f"{data_dir}/raw"
    processed_dir = get_nested(project_config, "data.processed_dir") or f"{data_dir}/processed"

    defaults: dict[str, Any] = {
        "validation": {
            "strategy": get_nested(project_config, "defaults.primary_validation"),
            "n_folds": get_nested(project_config, "defaults.n_folds"),
            "seed": get_nested(project_config, "defaults.seed"),
            "metric": get_nested(project_config, "defaults.metric"),
            "group_column": get_nested(project_config, "data.group_column"),
            "score_rows": get_nested(project_config, "data.score_rows"),
        },
        "data": {
            "raw_dir": raw_dir,
            "train_dir": get_nested(project_config, "data.train_dir") or f"{raw_dir}/train",
            "test_dir": get_nested(project_config, "data.test_dir") or f"{raw_dir}/test",
            "processed_dir": processed_dir,
            "sample_submission": get_nested(project_config, "submission.sample_file"),
            "target_column": get_nested(project_config, "data.target_column"),
            "id_column": get_nested(project_config, "submission.id_column"),
            "submission_target_column": first_submission_target(project_config),
        },
        "project": {
            "competition": project_config.get("competition", {}),
            "submission": project_config.get("submission", {}),
        },
    }

    runtime = project_config.get("runtime")
    if isinstance(runtime, dict):
        defaults["runtime"] = {"kaggle": runtime.get("kaggle", {})}

    return defaults
