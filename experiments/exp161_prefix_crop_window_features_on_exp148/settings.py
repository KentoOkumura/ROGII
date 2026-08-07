from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp161_prefix_crop_window_features_on_exp148"
PACKAGE_DIR = Path(__file__).resolve().parent
TODO_VALUES = {"", "TODO", "TBD", "FIXME", None}
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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


def first_submission_target(project_config: dict[str, Any]) -> Any:
    target_columns = get_nested(project_config, "submission.target_columns")
    if isinstance(target_columns, list) and target_columns:
        return target_columns[0]
    return None


def load_project_config() -> dict[str, Any]:
    return read_yaml(ROOT / "project.yml")


def is_todo_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() in TODO_VALUES
    try:
        return value in TODO_VALUES
    except TypeError:
        return False


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def allow_local_notebook_execution() -> bool:
    return os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1"


def kaggle_competition_input_dir(project_config: dict[str, Any]) -> Path | None:
    slug = get_nested(project_config, "competition.slug")
    input_root = KAGGLE_INPUT_ROOT
    if not input_root.exists():
        return None
    if not is_todo_value(slug):
        candidate = input_root / str(slug)
        if candidate.exists():
            return candidate
    for candidate in sorted(input_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
        if (candidate / "sample_submission.csv").exists():
            return candidate
    for candidate in sorted(input_root.rglob("sample_submission.csv")):
        parent = candidate.parent
        if (parent / "train").is_dir() or (parent / "test").is_dir():
            return parent
    return None


def project_experiment_defaults(project_config: dict[str, Any]) -> dict[str, Any]:
    if not project_config:
        return {}

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


def load_config() -> dict[str, Any]:
    project_config = load_project_config()
    config_path = Path(__file__).with_name("config.yaml")
    experiment_config = read_yaml(config_path)
    return deep_merge(project_experiment_defaults(project_config), experiment_config)


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_name: str = EXPERIMENT_NAME

    @property
    def config(self) -> dict[str, Any]:
        return load_config()

    @property
    def root(self) -> Path:
        return ROOT

    @property
    def output_root(self) -> Path:
        if is_kaggle_runtime():
            return KAGGLE_WORKING_ROOT
        return self.root

    @property
    def experiment_dir(self) -> Path:
        if is_kaggle_runtime():
            return KAGGLE_WORKING_ROOT
        experiments_dir = self.resolve_project_path("paths.experiments_dir", "experiments")
        candidate = experiments_dir / self.experiment_name
        if experiments_dir.exists() or candidate.exists():
            return candidate
        return PACKAGE_DIR

    @property
    def data_dir(self) -> Path:
        return self.resolve_project_path("paths.data_dir", "data")

    @property
    def raw_data_dir(self) -> Path:
        local_path = self.resolve_config_path("data.raw_dir", self.data_dir / "raw")
        return self.kaggle_path_or_local(local_path)

    @property
    def train_data_dir(self) -> Path:
        local_path = self.resolve_config_path("data.train_dir", self.raw_data_dir / "train")
        return self.kaggle_path_or_local(local_path, "train")

    @property
    def test_data_dir(self) -> Path:
        local_path = self.resolve_config_path("data.test_dir", self.raw_data_dir / "test")
        return self.kaggle_path_or_local(local_path, "test")

    @property
    def sample_submission_path(self) -> Path:
        local_path = self.resolve_config_path(
            "data.sample_submission",
            self.raw_data_dir / "sample_submission.csv",
        )
        return self.kaggle_path_or_local(local_path, "sample_submission.csv")

    @property
    def processed_data_dir(self) -> Path:
        if is_kaggle_runtime():
            return self.output_root / "data" / "processed"
        return self.resolve_config_path("data.processed_dir", self.data_dir / "processed")

    @property
    def artifacts_dir(self) -> Path:
        return self.experiment_dir / "artifacts"

    @property
    def features_dir(self) -> Path:
        return self.experiment_dir / "features"

    @property
    def metrics_path(self) -> Path:
        return self.experiment_dir / "metrics.json"

    @property
    def submission_path(self) -> Path:
        output_file = (
            get_nested(load_project_config(), "submission.output_file") or "submission.csv"
        )
        path = Path(str(output_file))
        if path.is_absolute():
            return path
        return self.output_root / path

    def resolve_path(self, value: Any, default: str | Path) -> Path:
        path_value = default if value in TODO_VALUES else value
        path = path_value if isinstance(path_value, Path) else Path(str(path_value))
        if path.is_absolute():
            return path
        return self.root / path

    def resolve_config_path(self, dotted_key: str, default: str | Path) -> Path:
        return self.resolve_path(get_nested(self.config, dotted_key), default)

    def resolve_project_path(self, dotted_key: str, default: str | Path) -> Path:
        return self.resolve_path(get_nested(load_project_config(), dotted_key), default)

    def kaggle_path_or_local(self, local_path: Path, relative: str | None = None) -> Path:
        if is_kaggle_runtime():
            kaggle_input = kaggle_competition_input_dir(load_project_config())
            if kaggle_input is None:
                raise FileNotFoundError(
                    "Kaggle runtime detected, but no competition input directory was found "
                    f"under {KAGGLE_INPUT_ROOT}."
                )
            if relative is None:
                return kaggle_input
            candidate = kaggle_input / relative
            if not candidate.exists():
                raise FileNotFoundError(f"Kaggle input path not found: {candidate}")
            return candidate
        if local_path.exists():
            return local_path
        kaggle_input = kaggle_competition_input_dir(load_project_config())
        if kaggle_input is None:
            return local_path
        if relative is None:
            return kaggle_input
        candidate = kaggle_input / relative
        return candidate if candidate.exists() else local_path

    def require_kaggle_runtime(self) -> None:
        if is_kaggle_runtime() or allow_local_notebook_execution():
            return
        raise RuntimeError(
            "Notebook execution is configured for Kaggle. "
            "Use task prepare-kaggle-notebooks and push the generated kernel; "
            "set EXPERIMENT_ALLOW_LOCAL=1 only for an explicit local smoke run."
        )

    def ensure_output_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
