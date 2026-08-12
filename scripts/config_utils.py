from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = ROOT / "project.yml"
TODO_VALUES = {"", "TODO", "TBD", "FIXME", None}
NOTEBOOK_KIND_RE = re.compile(r"[a-z0-9][a-z0-9_]*")
KAGGLE_RUNTIME_METADATA_KEYS = (
    "enable_gpu",
    "enable_internet",
    "enable_tpu",
    "machine_shape",
)


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


def validate_notebook_kind(value: str, *, allow_both: bool = False) -> str:
    if allow_both and value == "both":
        return value
    if not NOTEBOOK_KIND_RE.fullmatch(value):
        raise ValueError(
            "notebook kind must contain only lowercase letters, digits, and underscores, "
            "and must start with a letter or digit"
        )
    return value


def effective_kaggle_runtime(
    project_config: dict[str, Any],
    experiment_config: dict[str, Any] | None = None,
    notebook_kind: str | None = None,
) -> dict[str, Any]:
    """Resolve Kaggle metadata settings from project, experiment, then notebook config."""
    resolved: dict[str, Any] = {}
    sources: list[Any] = [get_nested(project_config, "runtime.kaggle")]
    if experiment_config is not None:
        experiment_runtime = get_nested(experiment_config, "runtime.kaggle")
        sources.append(experiment_runtime)
        if notebook_kind and isinstance(experiment_runtime, dict):
            sources.append(experiment_runtime.get(notebook_kind))

    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in KAGGLE_RUNTIME_METADATA_KEYS:
            if key in source:
                resolved[key] = source[key]
        if "machine_shape" not in source and "machineShape" in source:
            resolved["machine_shape"] = source["machineShape"]
    return resolved


def kaggle_runtime_errors(
    settings: dict[str, Any],
    *,
    require_core_fields: bool = True,
) -> list[str]:
    errors: list[str] = []
    for key in ("enable_gpu", "enable_internet"):
        value = settings.get(key)
        if value is None and not require_core_fields:
            continue
        if not isinstance(value, bool):
            errors.append(f"runtime.kaggle.{key} must be true or false")

    enable_tpu = settings.get("enable_tpu", False)
    if not isinstance(enable_tpu, bool):
        errors.append("runtime.kaggle.enable_tpu must be true or false")
    elif enable_tpu:
        errors.append("runtime.kaggle.enable_tpu=true is unsupported by this repository")

    machine_shape = settings.get("machine_shape")
    if machine_shape is not None and (
        not isinstance(machine_shape, str) or not machine_shape.strip()
    ):
        errors.append("runtime.kaggle.machine_shape must be a non-empty string")
    return errors


def project_path(config: dict[str, Any], dotted_key: str) -> Path:
    value = get_nested(config, dotted_key)
    if is_todo(value):
        raise ValueError(f"project path is not configured: {dotted_key}")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


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
    seed = get_nested(project_config, "defaults.seed")

    defaults: dict[str, Any] = {
        "validation": {
            "strategy": get_nested(project_config, "defaults.primary_validation"),
            "n_folds": get_nested(project_config, "defaults.n_folds"),
            "seed": seed,
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
        "reproducibility": {"seed": seed},
    }

    runtime = project_config.get("runtime")
    if isinstance(runtime, dict):
        defaults["runtime"] = {"kaggle": runtime.get("kaggle", {})}

    return defaults
