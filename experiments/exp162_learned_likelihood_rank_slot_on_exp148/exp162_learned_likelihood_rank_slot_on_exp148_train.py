# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp162_learned_likelihood_rank_slot_on_exp148 compact self-contained train
#
# Compact self-contained train notebook source. It keeps the exp162 CPU train path visible in the notebook and avoids local experiment helper imports.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Train feature assembly helpers
# 4. Model training and artifact helpers
# 5. Setup and configuration
# 6. Input and feature contract
# 7. Train learned rank-slot variant
# 8. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%


EXPERIMENT_NAME = "exp162_learned_likelihood_rank_slot_on_exp148"
PACKAGE_DIR = Path.cwd()
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
    return start


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
    config_path = PACKAGE_DIR / "config.yaml"
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


# %% [markdown]
# ## 3. Train feature assembly helpers

# %%

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
EXP145_TRAIN_ARTIFACTS = (
    Path("experiments")
    / "exp145_learned_likelihood_rawtest_feature_generator_parity"
    / "kaggle"
    / "output"
    / "train_v2"
    / "artifacts"
)
EXP145_INFERENCE_ARTIFACTS = (
    Path("experiments")
    / "exp145_learned_likelihood_rawtest_feature_generator_parity"
    / "kaggle"
    / "output"
    / "inference_v3"
    / "artifacts"
)
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
FULL_REPLAY_CACHE_SUMMARY = "exp063_full_replay_feature_cache_summary.json"
EXP145_TRAIN_ML_FEATURES = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_full_train_ml_features.csv.gz"
)
EXP145_RAWTEST_ML_FEATURES = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_rawtest_ml_features.csv.gz"
)
EXP145_FEATURE_SCHEMA = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_feature_schema.csv"
)
EXP145_SUMMARY = "exp145_learned_likelihood_rawtest_feature_generator_parity_summary.json"
EXPERIMENT_NAME = "exp162_learned_likelihood_rank_slot_on_exp148"
OUTPUT_PREFIX = EXPERIMENT_NAME
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196


@dataclass(frozen=True)
class LearnedRankCandidateSpec:
    name: str
    source_column: str
    transform: str
    family: str
    probability_column: str
    error_column: str
    candidate_tvt_column: str | None = None
    enabled: bool = True


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(np.asarray(values, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def exp063_lgb_config_family(*, fast: bool = False) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbose": -1,
        "max_bin": 255,
    }
    n_estimators = 600 if fast else 5000
    return [
        {
            **base,
            "num_leaves": 255,
            "min_child_samples": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.05,
            "learning_rate": 0.03,
            "n_estimators": n_estimators,
            "seed": 123,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 0,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 29,
        },
    ]


def apply_mode_overrides(
    configs: list[dict[str, Any]],
    mode_config: dict[str, Any],
) -> list[dict[str, Any]]:
    use_gpu = bool(mode_config.get("use_gpu", False))
    common = dict(mode_config.get("common_overrides") or {})
    updated: list[dict[str, Any]] = []
    for params in configs:
        merged = dict(params)
        if use_gpu:
            merged["device_type"] = "gpu"
        else:
            merged.pop("device_type", None)
            merged.pop("gpu_use_dp", None)
        merged.update(common)
        if use_gpu and "gpu_use_dp" not in merged:
            merged["gpu_use_dp"] = False
        updated.append(merged)
    return updated


def load_exp072_full_replay_cache_frame(
    cache_path: str | Path | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
    frame = pd.read_csv(source, nrows=max_rows, dtype={"id": str, "well": str})
    required = {"id", "well", "target", "last_known_tvt", "z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    feature_columns = [col for col in frame.columns if col not in META_COLUMNS]
    if len(feature_columns) != EXPECTED_FULL_REPLAY_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FULL_REPLAY_FEATURE_COUNT} full replay features, "
            f"got {len(feature_columns)} from {source}"
        )
    for col in ["target", *feature_columns]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[["target", *feature_columns]].to_numpy(np.float32)).all():
        raise ValueError("exp072 full replay cache contains non-finite numeric values")

    schema_path: Path | None = None
    summary_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    try:
        summary_path = find_artifact(FULL_REPLAY_CACHE_SUMMARY)
    except FileNotFoundError:
        summary_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_experiment": "exp072_exp063_full_replay_feature_cache",
        "source_kind": "exp063_full_public_replay_train_feature_cache",
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "summary": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
    }
    return frame, feature_columns, metadata


def load_known_prefix_anchors(train_dir: str | Path, wells: list[str] | pd.Series) -> pd.DataFrame:
    train_dir = Path(train_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train well file not found for anchor recovery: {path}")
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "anchor_tvt_true": float(anchor["TVT"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_anchor_columns(
    frame: pd.DataFrame,
    train_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_known_prefix_anchors(train_dir, frame["well"])
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Anchor merge produced missing prefix anchor values")
    t0_delta = merged["last_known_tvt"].to_numpy(np.float32) - merged["anchor_t0"].to_numpy(
        np.float32
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw prefix T0 does not match feature cache last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


def load_inference_prefix_anchors(
    test_dir: str | Path,
    wells: list[str] | pd.Series,
) -> pd.DataFrame:
    test_dir = Path(test_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = test_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw test well file not found for anchor recovery: {path}")
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for test well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_inference_anchor_columns(
    frame: pd.DataFrame,
    test_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_inference_prefix_anchors(test_dir, frame["well"])
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Inference anchor merge produced missing prefix anchor values")
    t0_delta = merged["last_known_tvt"].to_numpy(np.float32) - merged["anchor_t0"].to_numpy(
        np.float32
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw test prefix T0 does not match feature last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


def find_model_manifest(explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        path = Path(explicit_path)
        candidates.append(path if path.name == "manifest.json" else path / "manifest.json")
    candidates.extend(
        [
            Path.cwd() / "artifacts" / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path.cwd() / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path("experiments")
            / "exp092_u_projection_correction_disagreement_fullrun"
            / "kaggle"
            / "output"
            / "train"
            / "artifacts"
            / f"{OUTPUT_PREFIX}_lgb_models"
            / "manifest.json",
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{OUTPUT_PREFIX}_lgb_models/manifest.json"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"model manifest not found. Checked:\n{checked}")


def _tail_rank(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(-1).to_numpy(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _tail_rank(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _source_tvt(frame: pd.DataFrame, name: str, spec: dict[str, Any]) -> np.ndarray:
    if spec.get("enabled", True) is False:
        raise ValueError(f"Projection source is disabled: {name}")
    value_column = spec.get("value_column")
    delta_column = spec.get("delta_column")
    if value_column:
        if value_column not in frame.columns:
            raise ValueError(f"Projection source {name} missing value_column={value_column}")
        return frame[value_column].to_numpy(np.float32)
    if delta_column:
        if delta_column not in frame.columns:
            raise ValueError(f"Projection source {name} missing delta_column={delta_column}")
        return frame["last_known_tvt"].to_numpy(np.float32) + frame[delta_column].to_numpy(
            np.float32
        )
    raise ValueError(f"Projection source {name} needs value_column or delta_column")


def _weighted_polyfit_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    robust_iters: int,
    clip_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        fill = float(np.nanmedian(y[finite])) if finite.any() else 0.0
        pred = np.full(len(y), fill, dtype=np.float32)
        zeros = np.zeros(len(y), dtype=np.float32)
        return pred, zeros, zeros, 0

    x_fit = x[finite]
    y_fit = y[finite]
    x_center = float(np.median(x_fit))
    x_scale = float(np.nanpercentile(x_fit, 95) - np.nanpercentile(x_fit, 5))
    if not np.isfinite(x_scale) or x_scale <= 1e-6:
        x_scale = max(float(np.max(x_fit) - np.min(x_fit)), 1.0)
    x_norm = (x - x_center) / x_scale
    x_fit_norm = x_norm[finite]
    fit_degree = int(min(max(degree, 0), max(finite.sum() - 1, 0)))
    if np.unique(np.round(x_fit_norm, 8)).size <= fit_degree:
        fit_degree = max(int(np.unique(np.round(x_fit_norm, 8)).size) - 1, 0)

    weights = np.ones(len(y_fit), dtype=np.float64)
    coef = np.array([float(np.mean(y_fit))])
    for _ in range(max(int(robust_iters), 1)):
        coef = np.polyfit(x_fit_norm, y_fit, deg=fit_degree, w=weights)
        residual = y_fit - np.polyval(coef, x_fit_norm)
        mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
        if not np.isfinite(mad) or mad <= 1e-6:
            break
        weights = np.minimum(1.0, (float(clip_sigma) * mad) / (np.abs(residual) + 1e-6))

    poly = np.poly1d(coef)
    pred = poly(x_norm)
    deriv1 = np.polyder(poly, 1)(x_norm) / x_scale
    if fit_degree >= 2:
        deriv2 = np.polyder(poly, 2)(x_norm) / (x_scale * x_scale)
    else:
        deriv2 = np.zeros_like(x_norm)
    return (
        pred.astype(np.float32),
        deriv1.astype(np.float32),
        deriv2.astype(np.float32),
        fit_degree,
    )


def build_u_projection_features(
    frame: pd.DataFrame,
    *,
    source_specs: dict[str, dict[str, Any]],
    degree: int = 3,
    robust_iters: int = 3,
    clip_sigma: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    enabled_specs = {
        str(name): dict(spec)
        for name, spec in source_specs.items()
        if dict(spec).get("enabled", True)
    }
    if len(enabled_specs) < 2:
        raise ValueError("At least two enabled projection sources are required")

    result = pd.DataFrame({"id": frame["id"].to_numpy(), "well": frame["well"].to_numpy()})
    group_columns: dict[str, list[str]] = {
        "projection_correction": [],
        "projection_shape": [],
        "u_disagreement": [],
    }
    summary_rows: list[dict[str, Any]] = []
    z = frame["z"].to_numpy(np.float32)
    u0 = frame["anchor_t0"].to_numpy(np.float32) + frame["anchor_z0"].to_numpy(np.float32)
    md_since = frame.get("md_since")
    if md_since is None:
        x_all = np.maximum(
            frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False).astype(float).to_numpy(),
            0.0,
        )
    else:
        x_all = pd.to_numeric(md_since, errors="coerce").to_numpy(np.float32)

    source_u_columns: list[str] = []
    source_corr_columns: list[str] = []
    for source_name, spec in enabled_specs.items():
        prefix = f"uproj_{source_name}"
        tvt_source = _source_tvt(frame, source_name, spec)
        source_u = (tvt_source + z - u0).astype(np.float32)
        result[f"{prefix}_u"] = source_u
        source_u_columns.append(f"{prefix}_u")

        poly = np.zeros(len(frame), dtype=np.float32)
        slope = np.zeros(len(frame), dtype=np.float32)
        curvature = np.zeros(len(frame), dtype=np.float32)
        fit_degree = np.zeros(len(frame), dtype=np.int16)
        for _, idx in frame.groupby("well", sort=False).indices.items():
            idx_array = np.asarray(idx, dtype=np.int64)
            pred, deriv1, deriv2, used_degree = _weighted_polyfit_predict(
                x_all[idx_array],
                source_u[idx_array],
                degree=int(spec.get("degree", degree)),
                robust_iters=int(spec.get("robust_iters", robust_iters)),
                clip_sigma=float(spec.get("clip_sigma", clip_sigma)),
            )
            poly[idx_array] = pred
            slope[idx_array] = deriv1
            curvature[idx_array] = deriv2
            fit_degree[idx_array] = int(used_degree)

        resid = (source_u - poly).astype(np.float32)
        corr = (poly - source_u).astype(np.float32)
        abs_resid = np.abs(resid).astype(np.float32)
        mad_by_well = (
            pd.DataFrame({"well": frame["well"], "abs_resid": abs_resid})
            .groupby("well")["abs_resid"]
            .transform("median")
            .to_numpy(np.float32)
        )
        result[f"{prefix}_poly"] = poly
        result[f"{prefix}_resid"] = resid
        result[f"{prefix}_corr"] = corr
        result[f"{prefix}_abs_resid"] = abs_resid
        result[f"{prefix}_resid_mad"] = mad_by_well
        result[f"{prefix}_slope"] = slope
        result[f"{prefix}_curvature"] = curvature
        result[f"{prefix}_fit_degree"] = fit_degree.astype(np.float32)

        group_columns["projection_correction"].extend(
            [
                f"{prefix}_corr",
                f"{prefix}_resid",
                f"{prefix}_abs_resid",
                f"{prefix}_resid_mad",
            ]
        )
        group_columns["projection_shape"].extend(
            [
                f"{prefix}_poly",
                f"{prefix}_slope",
                f"{prefix}_curvature",
                f"{prefix}_fit_degree",
            ]
        )
        source_corr_columns.append(f"{prefix}_corr")
        summary_rows.append(
            {
                "source": source_name,
                "rows": int(len(source_u)),
                "u_mean": float(np.mean(source_u)),
                "u_std": float(np.std(source_u)),
                "abs_resid_mean": float(np.mean(abs_resid)),
                "abs_resid_p95": float(np.quantile(abs_resid, 0.95)),
                "resid_mad_mean": float(np.mean(mad_by_well)),
            }
        )

    source_names = list(enabled_specs)
    for left_i, left in enumerate(source_names):
        for right in source_names[left_i + 1 :]:
            left_col = f"uproj_{left}_u"
            right_col = f"uproj_{right}_u"
            diff_col = f"uproj_diff_{left}_minus_{right}"
            abs_col = f"uproj_absdiff_{left}_{right}"
            result[diff_col] = result[left_col].to_numpy(np.float32) - result[right_col].to_numpy(
                np.float32
            )
            result[abs_col] = np.abs(result[diff_col].to_numpy(np.float32))
            group_columns["u_disagreement"].extend([diff_col, abs_col])

    source_u_matrix = result[source_u_columns].to_numpy(np.float32)
    result["uproj_source_u_std"] = np.std(source_u_matrix, axis=1).astype(np.float32)
    result["uproj_source_u_range"] = (
        np.max(source_u_matrix, axis=1) - np.min(source_u_matrix, axis=1)
    ).astype(np.float32)
    group_columns["u_disagreement"].extend(["uproj_source_u_std", "uproj_source_u_range"])
    if len(source_corr_columns) >= 2:
        corr_matrix = result[source_corr_columns].to_numpy(np.float32)
        result["uproj_corr_std"] = np.std(corr_matrix, axis=1).astype(np.float32)
        result["uproj_corr_range"] = (
            np.max(corr_matrix, axis=1) - np.min(corr_matrix, axis=1)
        ).astype(np.float32)
        group_columns["u_disagreement"].extend(["uproj_corr_std", "uproj_corr_range"])

    numeric_cols = [col for col in result.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("U-projection feature frame contains non-finite values")
    return result, group_columns, pd.DataFrame(summary_rows)


def load_learned_likelihood_ml_features(
    feature_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    *,
    feature_filename: str = EXP145_TRAIN_ML_FEATURES,
    local_artifacts: Path = EXP145_TRAIN_ARTIFACTS,
    source_experiment: str = "exp145_learned_likelihood_rawtest_feature_generator_parity",
    source_kind: str = "target_free_full_train_learned_likelihood_ml_features",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(feature_filename, feature_path, local_artifacts=local_artifacts)
    frame = pd.read_csv(source, dtype={"id": str, "well": str})
    required = {"id", "well", "fold", "md_since"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing learned likelihood feature columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if frame.duplicated(["id", "well"]).any():
        duplicated = int(frame.duplicated(["id", "well"]).sum())
        raise ValueError(
            f"learned likelihood ML feature cache has duplicated id/well rows: {duplicated}"
        )
    numeric_cols = [col for col in frame.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood ML feature cache contains non-finite numeric values")

    resolved_schema: Path | None
    resolved_summary: Path | None
    try:
        resolved_schema = find_artifact(
            EXP145_FEATURE_SCHEMA,
            schema_path,
            local_artifacts=local_artifacts,
        )
    except FileNotFoundError:
        resolved_schema = None
    try:
        resolved_summary = find_artifact(
            EXP145_SUMMARY,
            summary_path,
            local_artifacts=local_artifacts,
        )
    except FileNotFoundError:
        resolved_summary = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "source_experiment": source_experiment,
        "source_kind": source_kind,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": int(len(frame.columns)),
        "numeric_columns": numeric_cols,
        "schema": str(resolved_schema) if resolved_schema else None,
        "schema_sha256": sha256_file(resolved_schema) if resolved_schema else None,
        "summary": str(resolved_summary) if resolved_summary else None,
        "summary_sha256": sha256_file(resolved_summary) if resolved_summary else None,
    }
    return frame, metadata


def generate_current_test_learned_likelihood_ml_features(
    *,
    test_frame: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from learned_likelihood_rawtest_feature_generator_parity import (
        DEFAULT_EXP111_MANIFEST,
        DEFAULT_EXP111_SCHEMA,
        candidate_specs_from_config,
        ensure_multiobs_columns,
        exp111_model_feature_columns,
        find_artifact as find_generator_artifact,
        generate_ml_features_from_frame,
        load_exp111_models,
        load_feature_schema,
        sha256_path,
        source_required_columns,
        write_ml_features,
    )
    from settings import get_nested, load_config

    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    exp111_artifacts = Path(str(get_nested(config, "data.exp111_artifact_dir_local") or ""))
    schema_path = find_generator_artifact(
        DEFAULT_EXP111_SCHEMA,
        get_nested(config, "data.exp111_feature_schema"),
        local_dirs=[exp111_artifacts],
    )
    manifest_path = find_generator_artifact(
        DEFAULT_EXP111_MANIFEST,
        get_nested(config, "data.exp111_model_manifest"),
        local_dirs=[exp111_artifacts],
    )
    row_feature_columns = load_feature_schema(schema_path)
    model_feature_columns = exp111_model_feature_columns(row_feature_columns)
    classifier, error_model, model_meta = load_exp111_models(manifest_path=manifest_path)

    source_frame, multiobs_meta = ensure_multiobs_columns(test_frame, candidates, config=config)
    required_columns = source_required_columns(config, candidates)
    missing = [column for column in required_columns if column not in source_frame.columns]
    if missing:
        raise ValueError(f"current test frame missing learned likelihood source columns: {missing}")
    features, long_likelihood = generate_ml_features_from_frame(
        source_frame[required_columns],
        candidates=candidates,
        row_feature_columns=row_feature_columns,
        model_feature_columns=model_feature_columns,
        classifier=classifier,
        error_model=error_model,
        config=config,
    )
    feature_path = (
        output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_ml_features.csv.gz"
    )
    long_path = output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_long.csv.gz"
    write_ml_features(feature_path, features)
    long_likelihood.to_csv(long_path, index=False, compression="gzip")
    return features, {
        "source": str(feature_path),
        "source_sha256": sha256_path(feature_path),
        "source_decompressed_sha256": sha256_path(feature_path, decompressed=True),
        "source_experiment": OUTPUT_PREFIX,
        "source_kind": "target_free_current_test_generated_learned_likelihood_ml_features",
        "rows": int(len(features)),
        "wells": int(features["well"].nunique()),
        "columns": int(len(features.columns)),
        "exp111_schema": str(schema_path),
        "exp111_manifest": str(manifest_path),
        "exp111_model_meta": _jsonable(model_meta),
        "multiobs_generation": _jsonable(multiobs_meta),
        "long_likelihood": {
            "path": str(long_path),
            "rows": int(len(long_likelihood)),
            "sha256": sha256_path(long_path),
            "decompressed_sha256": sha256_path(long_path, decompressed=True),
        },
    }


def learned_feature_keys_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_keys = left[["id", "well"]].astype(str).sort_values(["id", "well"]).reset_index(drop=True)
    right_keys = (
        right[["id", "well"]].astype(str).sort_values(["id", "well"]).reset_index(drop=True)
    )
    return left_keys.equals(right_keys)


def build_learned_likelihood_features(
    learned_source: pd.DataFrame,
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "ll_")
    key_cols = ["id", "well"]
    group_columns: dict[str, list[str]] = {"learned_likelihood_confidence": []}

    direct_columns = [str(col) for col in config.get("direct_columns") or []]
    weighted_tvt_columns = [str(col) for col in config.get("weighted_tvt_columns") or []]
    candidate_tvt_columns = [str(col) for col in config.get("candidate_tvt_columns") or []]
    requested = direct_columns + weighted_tvt_columns + candidate_tvt_columns
    missing = [col for col in requested if col not in learned_source.columns]
    if missing:
        raise ValueError(
            f"learned likelihood ML feature cache missing configured columns: {missing}"
        )

    base_lookup = base_frame[key_cols + ["last_known_tvt", "likpf_mean_d"]].copy()
    base_lookup["likpf_mean_tvt"] = (
        base_lookup["last_known_tvt"].to_numpy(np.float32)
        + base_lookup["likpf_mean_d"].to_numpy(np.float32)
    ).astype(np.float32)
    joined = learned_source[key_cols + requested].merge(
        base_lookup,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError(
            "No shared rows between learned likelihood ML features and base feature frame"
        )
    features = joined[key_cols].copy()

    for col in direct_columns:
        out = f"{prefix}{col}"
        features[out] = joined[col].to_numpy(np.float32)
        group_columns["learned_likelihood_confidence"].append(out)

    for col in weighted_tvt_columns + candidate_tvt_columns:
        raw = joined[col].to_numpy(np.float32)
        minus_last = (raw - joined["last_known_tvt"].to_numpy(np.float32)).astype(np.float32)
        minus_likpf = (raw - joined["likpf_mean_tvt"].to_numpy(np.float32)).astype(np.float32)
        out_last = f"{prefix}{col}_minus_last_known_tvt"
        out_likpf = f"{prefix}{col}_minus_likpf_mean_tvt"
        features[out_last] = minus_last
        features[out_likpf] = minus_likpf
        group_columns["learned_likelihood_confidence"].extend([out_last, out_likpf])

    feature_cols = [col for col in features.columns if col not in key_cols]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(features[feature_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood feature frame contains non-finite values")

    summary = pd.DataFrame(
        [
            {
                "feature_group": "learned_likelihood_confidence",
                "configured_direct_columns": len(direct_columns),
                "configured_weighted_tvt_columns": len(weighted_tvt_columns),
                "configured_candidate_tvt_columns": len(candidate_tvt_columns),
                "generated_features": len(feature_cols),
                "rows": int(len(features)),
                "wells": int(features["well"].nunique()),
            }
        ]
    )
    return features, group_columns, summary


def learned_rank_candidate_specs_from_config(
    config: dict[str, Any],
) -> list[LearnedRankCandidateSpec]:
    specs: list[LearnedRankCandidateSpec] = []
    for item in config.get("candidates") or []:
        name = str(item["name"])
        specs.append(
            LearnedRankCandidateSpec(
                name=name,
                source_column=str(item.get("source_column") or name),
                transform=str(item.get("transform", "absolute")),
                family=str(item.get("family", name)),
                probability_column=str(item.get("probability_column") or f"learned_prob_{name}"),
                error_column=str(item.get("error_column") or f"learned_pred_abs_error_{name}"),
                candidate_tvt_column=(
                    str(item["candidate_tvt_column"])
                    if item.get("candidate_tvt_column") is not None
                    else None
                ),
                enabled=bool(item.get("enabled", True)),
            )
        )
    if not specs:
        raise ValueError("learned_likelihood_rank_slot.candidates must configure candidates")
    return specs


def _learned_rank_candidate_tvt(
    base_frame: pd.DataFrame,
    learned_source: pd.DataFrame,
    spec: LearnedRankCandidateSpec,
) -> np.ndarray:
    if spec.candidate_tvt_column and spec.candidate_tvt_column in learned_source.columns:
        return learned_source[spec.candidate_tvt_column].to_numpy(np.float32)
    if spec.source_column not in base_frame.columns:
        raise ValueError(f"Candidate {spec.name} missing source_column={spec.source_column}")
    values = base_frame[spec.source_column].to_numpy(np.float32)
    if spec.transform == "absolute":
        return values.astype(np.float32)
    if spec.transform == "base_plus_delta":
        return (base_frame["last_known_tvt"].to_numpy(np.float32) + values).astype(np.float32)
    raise ValueError(f"Unsupported candidate transform for {spec.name}: {spec.transform}")


def _heuristic_rank_scores(
    base_frame: pd.DataFrame,
    candidate_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n = len(base_frame)
    last_known = base_frame["last_known_tvt"].to_numpy(np.float32)
    scores: dict[str, np.ndarray] = {}
    for name, values in candidate_values.items():
        if name == "pf_ancc":
            pf_std = np.abs(
                pd.to_numeric(
                    base_frame.get("pf_ancc_std", pd.Series(50.0, index=base_frame.index)),
                    errors="coerce",
                )
                .fillna(50.0)
                .to_numpy(np.float32)
            )
            likpf = candidate_values.get("likpf_mean")
            if likpf is None:
                likpf = np.full(n, np.nan, dtype=np.float32)
            pf_likpf = np.nan_to_num(np.abs(values - likpf), nan=50.0)
            score = 1.0 / (1.0 + pf_std / 50.0 + pf_likpf / 100.0)
        elif name == "beam_mean":
            pf = candidate_values.get("pf_ancc")
            if pf is None:
                pf = np.full(n, np.nan, dtype=np.float32)
            disagreement = np.nan_to_num(np.abs(pf - values), nan=75.0)
            score = 1.0 / (1.0 + disagreement / 75.0)
        elif name == "likpf_mean":
            score = 1.0 / (1.0 + np.abs(values - last_known) / 150.0)
        elif name in {"sc_ens", "hyb"}:
            score = 0.5 / (1.0 + np.abs(values - last_known) / 200.0)
        else:
            score = np.full(n, 0.35, dtype=np.float32)
        scores[name] = np.asarray(score, dtype=np.float32)
    return scores


def _entropy_from_score_matrix(score_matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(score_matrix.astype(np.float64), 1e-9, None)
    prob = clipped / clipped.sum(axis=1, keepdims=True)
    return (-(prob * np.log(prob)).sum(axis=1) / np.log(score_matrix.shape[1])).astype(np.float32)


def build_learned_likelihood_rank_slot_features(
    learned_source: pd.DataFrame,
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "llrs_")
    specs = [spec for spec in learned_rank_candidate_specs_from_config(config) if spec.enabled]
    top_k = int(config.get("top_k", 3))
    if top_k < 1:
        raise ValueError("learned_likelihood_rank_slot.top_k must be >= 1")
    if len(specs) < top_k:
        raise ValueError(
            f"learned_likelihood_rank_slot top_k={top_k} exceeds candidate count={len(specs)}"
        )

    key_cols = ["id", "well"]
    required_learned = sorted(
        {
            col
            for spec in specs
            for col in [spec.probability_column, spec.error_column, spec.candidate_tvt_column]
            if col
        }
    )
    missing = [col for col in required_learned if col not in learned_source.columns]
    if missing:
        raise ValueError(f"learned likelihood source missing rank-slot columns: {missing}")

    base_cols = [
        "id",
        "well",
        "last_known_tvt",
        "z",
        "anchor_t0",
        "anchor_z0",
        "md_since",
        "pf_ancc_std",
        *sorted({spec.source_column for spec in specs}),
    ]
    base_cols = [col for col in base_cols if col in base_frame.columns]
    joined = base_frame[base_cols].merge(
        learned_source[key_cols + required_learned],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("No shared rows for learned likelihood rank-slot features")
    if len(joined) != len(base_frame):
        raise ValueError(
            "learned likelihood rank-slot source does not cover every base row: "
            f"{len(joined)} of {len(base_frame)}"
        )

    candidate_names = [spec.name for spec in specs]
    source_codes = {name: float(index) for index, name in enumerate(candidate_names)}
    candidate_values = {
        spec.name: _learned_rank_candidate_tvt(joined, joined, spec) for spec in specs
    }
    prob_matrix = np.column_stack(
        [
            np.clip(
                pd.to_numeric(joined[spec.probability_column], errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32),
                0.0,
                None,
            )
            for spec in specs
        ]
    ).astype(np.float32)
    error_matrix = np.column_stack(
        [
            np.clip(
                pd.to_numeric(joined[spec.error_column], errors="coerce")
                .fillna(float(config.get("missing_error_fill", 250.0)))
                .to_numpy(np.float32),
                0.0,
                None,
            )
            for spec in specs
        ]
    ).astype(np.float32)
    error_scale = float(config.get("error_scale", 50.0))
    learned_score_matrix = (prob_matrix / (1.0 + error_matrix / max(error_scale, 1e-6))).astype(
        np.float32
    )
    order = np.argsort(-learned_score_matrix, axis=1)
    heuristic_scores = _heuristic_rank_scores(joined, candidate_values)
    heuristic_matrix = np.column_stack([heuristic_scores[name] for name in candidate_names]).astype(
        np.float32
    )
    heuristic_order = np.argsort(-heuristic_matrix, axis=1)

    result = joined[key_cols].copy()
    group_columns: dict[str, list[str]] = {
        "learned_likelihood_rank_slot_identity": [],
        "learned_likelihood_rank_slot_delta": [],
        "learned_likelihood_rank_slot_u_projection": [],
        "learned_likelihood_rank_slot_u_disagreement": [],
        "learned_likelihood_rank_slot_exp098_compare": [],
    }
    row_index = np.arange(len(joined))
    last_known = joined["last_known_tvt"].to_numpy(np.float32)
    z = joined["z"].to_numpy(np.float32)
    u0 = joined["anchor_t0"].to_numpy(np.float32) + joined["anchor_z0"].to_numpy(np.float32)
    md_since = joined.get("md_since")
    if md_since is None:
        x_all = np.maximum(_tail_rank(joined["id"]).astype(np.float32), 0.0)
    else:
        x_all = pd.to_numeric(md_since, errors="coerce").fillna(0.0).to_numpy(np.float32)
    value_matrix = np.column_stack([candidate_values[name] for name in candidate_names]).astype(
        np.float32
    )
    slot_u_columns: list[str] = []

    for slot in range(top_k):
        slot_prefix = f"{prefix}rank{slot + 1}"
        col_index = order[:, slot]
        slot_tvt = value_matrix[row_index, col_index].astype(np.float32)
        slot_prob = prob_matrix[row_index, col_index].astype(np.float32)
        slot_error = error_matrix[row_index, col_index].astype(np.float32)
        slot_score = learned_score_matrix[row_index, col_index].astype(np.float32)
        slot_code = np.asarray(
            [source_codes[candidate_names[int(idx)]] for idx in col_index],
            dtype=np.float32,
        )
        slot_u = (slot_tvt + z - u0).astype(np.float32)
        slot_u_columns.append(f"{slot_prefix}_u")

        result[f"{slot_prefix}_source_code"] = slot_code
        result[f"{slot_prefix}_prob"] = slot_prob
        result[f"{slot_prefix}_pred_error"] = slot_error
        result[f"{slot_prefix}_score"] = slot_score
        result[f"{slot_prefix}_candidate_minus_last_anchor"] = (slot_tvt - last_known).astype(
            np.float32
        )
        result[f"{slot_prefix}_u"] = slot_u
        group_columns["learned_likelihood_rank_slot_identity"].extend(
            [
                f"{slot_prefix}_source_code",
                f"{slot_prefix}_prob",
                f"{slot_prefix}_pred_error",
                f"{slot_prefix}_score",
            ]
        )
        group_columns["learned_likelihood_rank_slot_delta"].append(
            f"{slot_prefix}_candidate_minus_last_anchor"
        )
        for name in candidate_names:
            flag_col = f"{slot_prefix}_is_{name}"
            result[flag_col] = (slot_code == source_codes[name]).astype(np.float32)
            group_columns["learned_likelihood_rank_slot_identity"].append(flag_col)

        poly = np.zeros(len(joined), dtype=np.float32)
        slope = np.zeros(len(joined), dtype=np.float32)
        curvature = np.zeros(len(joined), dtype=np.float32)
        fit_degree = np.zeros(len(joined), dtype=np.int16)
        for _, idx in joined.groupby("well", sort=False).indices.items():
            idx_array = np.asarray(idx, dtype=np.int64)
            pred, deriv1, deriv2, used_degree = _weighted_polyfit_predict(
                x_all[idx_array],
                slot_u[idx_array],
                degree=int(config.get("degree", 3)),
                robust_iters=int(config.get("robust_iters", 3)),
                clip_sigma=float(config.get("clip_sigma", 4.0)),
            )
            poly[idx_array] = pred
            slope[idx_array] = deriv1
            curvature[idx_array] = deriv2
            fit_degree[idx_array] = int(used_degree)
        resid = (slot_u - poly).astype(np.float32)
        abs_resid = np.abs(resid).astype(np.float32)
        mad_by_well = (
            pd.DataFrame({"well": joined["well"], "abs_resid": abs_resid})
            .groupby("well")["abs_resid"]
            .transform("median")
            .to_numpy(np.float32)
        )
        result[f"{slot_prefix}_u_corr"] = (poly - slot_u).astype(np.float32)
        result[f"{slot_prefix}_u_resid"] = resid
        result[f"{slot_prefix}_u_abs_resid"] = abs_resid
        result[f"{slot_prefix}_u_resid_mad"] = mad_by_well
        result[f"{slot_prefix}_u_slope"] = slope
        result[f"{slot_prefix}_u_curvature"] = curvature
        result[f"{slot_prefix}_u_fit_degree"] = fit_degree.astype(np.float32)
        group_columns["learned_likelihood_rank_slot_u_projection"].extend(
            [
                f"{slot_prefix}_u_corr",
                f"{slot_prefix}_u_resid",
                f"{slot_prefix}_u_abs_resid",
                f"{slot_prefix}_u_resid_mad",
                f"{slot_prefix}_u_slope",
                f"{slot_prefix}_u_curvature",
                f"{slot_prefix}_u_fit_degree",
            ]
        )

    for left in range(top_k):
        for right in range(left + 1, top_k):
            left_prefix = f"{prefix}rank{left + 1}"
            right_prefix = f"{prefix}rank{right + 1}"
            delta_col = f"{left_prefix}_minus_{right_prefix}_candidate_delta"
            abs_col = f"{left_prefix}_{right_prefix}_candidate_absdiff"
            score_gap_col = f"{left_prefix}_minus_{right_prefix}_score_gap"
            prob_gap_col = f"{left_prefix}_minus_{right_prefix}_prob_gap"
            error_gap_col = f"{left_prefix}_minus_{right_prefix}_pred_error_gap"
            u_diff_col = f"{left_prefix}_minus_{right_prefix}_u_diff"
            u_abs_col = f"{left_prefix}_{right_prefix}_u_absdiff"
            result[delta_col] = (
                result[f"{left_prefix}_candidate_minus_last_anchor"].to_numpy(np.float32)
                - result[f"{right_prefix}_candidate_minus_last_anchor"].to_numpy(np.float32)
            ).astype(np.float32)
            result[abs_col] = np.abs(result[delta_col].to_numpy(np.float32))
            result[score_gap_col] = (
                result[f"{left_prefix}_score"].to_numpy(np.float32)
                - result[f"{right_prefix}_score"].to_numpy(np.float32)
            ).astype(np.float32)
            result[prob_gap_col] = (
                result[f"{left_prefix}_prob"].to_numpy(np.float32)
                - result[f"{right_prefix}_prob"].to_numpy(np.float32)
            ).astype(np.float32)
            result[error_gap_col] = (
                result[f"{left_prefix}_pred_error"].to_numpy(np.float32)
                - result[f"{right_prefix}_pred_error"].to_numpy(np.float32)
            ).astype(np.float32)
            result[u_diff_col] = (
                result[f"{left_prefix}_u"].to_numpy(np.float32)
                - result[f"{right_prefix}_u"].to_numpy(np.float32)
            ).astype(np.float32)
            result[u_abs_col] = np.abs(result[u_diff_col].to_numpy(np.float32))
            group_columns["learned_likelihood_rank_slot_delta"].extend([delta_col, abs_col])
            group_columns["learned_likelihood_rank_slot_identity"].extend(
                [score_gap_col, prob_gap_col, error_gap_col]
            )
            group_columns["learned_likelihood_rank_slot_u_disagreement"].extend(
                [u_diff_col, u_abs_col]
            )

    result[f"{prefix}score_entropy"] = _entropy_from_score_matrix(learned_score_matrix)
    result[f"{prefix}prob_entropy"] = _entropy_from_score_matrix(prob_matrix)
    result[f"{prefix}top1_margin"] = (
        result[f"{prefix}rank1_score"].to_numpy(np.float32)
        - result[f"{prefix}rank2_score"].to_numpy(np.float32)
    ).astype(np.float32)
    group_columns["learned_likelihood_rank_slot_identity"].extend(
        [f"{prefix}score_entropy", f"{prefix}prob_entropy", f"{prefix}top1_margin"]
    )
    slot_u_matrix = result[slot_u_columns].to_numpy(np.float32)
    result[f"{prefix}slot_u_std"] = np.std(slot_u_matrix, axis=1).astype(np.float32)
    result[f"{prefix}slot_u_range"] = (
        np.max(slot_u_matrix, axis=1) - np.min(slot_u_matrix, axis=1)
    ).astype(np.float32)
    group_columns["learned_likelihood_rank_slot_u_disagreement"].extend(
        [f"{prefix}slot_u_std", f"{prefix}slot_u_range"]
    )

    ll_rank1_code = result[f"{prefix}rank1_source_code"].to_numpy(np.float32)
    heuristic_rank1_index = heuristic_order[:, 0]
    heuristic_rank1_code = np.asarray(
        [source_codes[candidate_names[int(idx)]] for idx in heuristic_rank1_index],
        dtype=np.float32,
    )
    heuristic_rank1_tvt = value_matrix[row_index, heuristic_rank1_index].astype(np.float32)
    ll_rank1_tvt = (
        result[f"{prefix}rank1_candidate_minus_last_anchor"].to_numpy(np.float32) + last_known
    )
    result[f"{prefix}exp098_rank1_source_code"] = heuristic_rank1_code
    result[f"{prefix}exp098_rank1_source_eq_ll_rank1_source"] = (
        heuristic_rank1_code == ll_rank1_code
    ).astype(np.float32)
    result[f"{prefix}exp098_rank1_tvt_minus_ll_rank1_tvt"] = (
        heuristic_rank1_tvt - ll_rank1_tvt
    ).astype(np.float32)
    result[f"{prefix}entropy_x_exp098_rank_disagreement"] = (
        result[f"{prefix}prob_entropy"].to_numpy(np.float32)
        * (1.0 - result[f"{prefix}exp098_rank1_source_eq_ll_rank1_source"].to_numpy(np.float32))
    ).astype(np.float32)
    group_columns["learned_likelihood_rank_slot_exp098_compare"].extend(
        [
            f"{prefix}exp098_rank1_source_code",
            f"{prefix}exp098_rank1_source_eq_ll_rank1_source",
            f"{prefix}exp098_rank1_tvt_minus_ll_rank1_tvt",
            f"{prefix}entropy_x_exp098_rank_disagreement",
        ]
    )

    numeric_cols = [col for col in result.columns if col not in key_cols]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood rank-slot feature frame contains non-finite values")

    summary_rows: list[dict[str, Any]] = []
    for slot in range(top_k):
        slot_prefix = f"{prefix}rank{slot + 1}"
        for name in candidate_names:
            mask = result[f"{slot_prefix}_is_{name}"].to_numpy(np.float32).astype(bool)
            summary_rows.append(
                {
                    "slot": f"rank{slot + 1}",
                    "candidate": name,
                    "selected_rows": int(mask.sum()),
                    "selected_rate": float(mask.mean()),
                    "score_mean_when_selected": (
                        float(result.loc[mask, f"{slot_prefix}_score"].mean())
                        if mask.any()
                        else 0.0
                    ),
                    "prob_mean_when_selected": (
                        float(result.loc[mask, f"{slot_prefix}_prob"].mean()) if mask.any() else 0.0
                    ),
                    "pred_error_mean_when_selected": (
                        float(result.loc[mask, f"{slot_prefix}_pred_error"].mean())
                        if mask.any()
                        else 0.0
                    ),
                }
            )
    return result, group_columns, pd.DataFrame(summary_rows)


def feature_columns_for_variant(
    base_feature_columns: list[str],
    feature_group_columns: dict[str, list[str]],
    variant: dict[str, Any],
) -> list[str]:
    columns = list(base_feature_columns)
    groups = list(variant.get("feature_groups") or [])
    extra: list[str] = []
    for group in groups:
        if group not in feature_group_columns:
            raise ValueError(f"Unknown feature group for variant {variant}: {group}")
        extra.extend(feature_group_columns[group])
    for col in variant.get("extra_columns") or []:
        extra.append(str(col))
    seen = set(columns)
    for col in extra:
        if col not in seen:
            columns.append(col)
            seen.add(col)
    return columns


# %% [markdown]
# ## 4. Model training and artifact helpers


# %%
def _by_well_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    return (
        frame.groupby(["variant", "mode", "model", "well"], as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_mean=("error_tvt", "mean"),
            error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
        )
        .sort_values(["variant", "mode", "model", "rmse_tvt"], ascending=[True, True, True, False])
    )


def _bucket_metrics(predictions: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[["id", "variant", "mode", "model", "target_tvt", "pred_tvt"]].copy()
    context = source_frame[["id"]].copy()
    distance_source = source_frame.get("md_since", pd.Series(np.nan, index=source_frame.index))
    context["distance_bucket"] = _distance_bucket(distance_source)
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    frame = frame.merge(context, on="id", how="left", validate="many_to_one")
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    rows: list[pd.DataFrame] = []
    for bucket_col in ["distance_bucket", "tail_rank_bucket"]:
        grouped = (
            frame.groupby(["variant", "mode", "model", bucket_col], observed=True)
            .agg(
                rows=("id", "size"),
                rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
                error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            )
            .reset_index()
            .rename(columns={bucket_col: "bucket"})
        )
        grouped.insert(3, "bucket_family", bucket_col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _fit_one_variant_mode(
    *,
    variant: dict[str, Any],
    mode_name: str,
    mode_config: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    n_splits: int,
    fast: bool,
    early_stopping_rounds: int,
    max_train_rows: int | None,
    save_models: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(42)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_model: list[np.ndarray] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models" / variant_name / mode_name
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "variant": variant_name,
                "mode": mode_name,
                "rows": int(len(frame)),
                "features": int(len(feature_columns)),
                "configs": int(len(configs)),
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_index, params in enumerate(configs):
        oof = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(x_matrix, y, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            model = LGBMRegressor(**params)
            model.fit(
                x_matrix[train_idx],
                y[train_idx],
                eval_set=[(x_matrix[valid_idx], y[valid_idx])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(early_stopping_rounds), verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(x_matrix[valid_idx], num_iteration=best_iter).astype(np.float32)
            oof[valid_idx] = pred
            pred_tvt = base[valid_idx] + pred
            model_file = None
            model_sha = None
            if save_models:
                model_file = f"{mode_name}__lgb{model_index}__fold{fold}.txt"
                model_path = model_dir / model_file
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_rows.append(
                {
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(valid_idx)),
                    "train_rows": int(len(train_idx)),
                    "features": int(len(feature_columns)),
                    "feature_groups": ",".join(variant.get("feature_groups") or []),
                    "best_iteration": best_iter,
                    "rmse_tvt": rmse(target_tvt[valid_idx], pred_tvt),
                    "rmse_target": rmse(y[valid_idx], pred),
                    "prediction_sha256": prediction_sha256(
                        frame.iloc[valid_idx]["id"],
                        pred_tvt,
                        label=f"{variant_name}/{mode_name}/lgb{model_index}/fold{fold}/tvt",
                    ),
                    "model_file": model_file,
                    "model_sha256": model_sha,
                }
            )
            for feature, importance in zip(
                feature_columns,
                model.feature_importances_,
                strict=False,
            ):
                importance_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
            if save_models:
                model_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "model_index": int(model_index),
                        "fold": int(fold),
                        "best_iteration": best_iter,
                        "file": f"{variant_name}/{mode_name}/{model_file}",
                        "sha256": model_sha,
                    }
                )
            print(
                json.dumps(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                        "best_iteration": best_iter,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        oof_by_model.append(oof)
        pred_tvt = base + oof
        metric_rows.append(
            {
                "variant": variant_name,
                "mode": mode_name,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "feature_groups": ",".join(variant.get("feature_groups") or []),
                "best_iteration": None,
                "rmse_tvt": rmse(target_tvt, pred_tvt),
                "rmse_target": rmse(y, oof),
                "prediction_sha256": prediction_sha256(
                    frame["id"],
                    pred_tvt,
                    label=f"{variant_name}/{mode_name}/lgb{model_index}/pooled/tvt",
                ),
                "model_file": None,
                "model_sha256": None,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": frame["id"].to_numpy(),
                    "well": frame["well"].to_numpy(),
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "last_known_tvt": base,
                    "target": y,
                    "target_tvt": target_tvt,
                    "pred_target": oof,
                    "pred_tvt": pred_tvt,
                }
            )
        )

    ensemble = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    ensemble_tvt = base + ensemble
    ensemble_sha = prediction_sha256(
        frame["id"],
        ensemble_tvt,
        label=f"{variant_name}/{mode_name}/lgb_mean/pooled/tvt",
    )
    metric_rows.append(
        {
            "variant": variant_name,
            "mode": mode_name,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(frame)),
            "train_rows": None,
            "features": int(len(feature_columns)),
            "feature_groups": ",".join(variant.get("feature_groups") or []),
            "best_iteration": None,
            "rmse_tvt": rmse(target_tvt, ensemble_tvt),
            "rmse_target": rmse(y, ensemble),
            "prediction_sha256": ensemble_sha,
            "model_file": None,
            "model_sha256": None,
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well"].to_numpy(),
                "variant": variant_name,
                "mode": mode_name,
                "model": "lgb_mean",
                "last_known_tvt": base,
                "target": y,
                "target_tvt": target_tvt,
                "pred_target": ensemble,
                "pred_tvt": ensemble_tvt,
            }
        )
    )
    mode_summary = {
        "variant": variant_name,
        "mode": mode_name,
        "description": mode_config.get("description"),
        "feature_count": int(len(feature_columns)),
        "feature_groups": list(variant.get("feature_groups") or []),
        "use_gpu": bool(mode_config.get("use_gpu", False)),
        "common_overrides": mode_config.get("common_overrides") or {},
        "lgb_configs": configs,
        "lgb_mean_prediction_sha256": ensemble_sha,
        "model_count": int(len(model_rows)),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(importance_rows),
        model_rows,
        mode_summary,
    )


def _plot_mean_importance(mean_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    variants = mean_importance["variant"].drop_duplicates().tolist()
    if not variants:
        return
    fig, axes = plt.subplots(
        len(variants),
        1,
        figsize=(12, max(4, 0.28 * int(top_n) * len(variants))),
        squeeze=False,
    )
    for ax, variant in zip(axes.ravel(), variants, strict=False):
        subset = mean_importance[mean_importance["variant"].eq(variant)].nlargest(
            top_n,
            "mean_importance",
        )
        subset = subset.sort_values("mean_importance", ascending=True)
        ax.barh(subset["feature"], subset["mean_importance"], color="#2f6f8f")
        ax.set_title(str(variant))
        ax.set_xlabel("mean feature_importances_")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_learned_likelihood_rank_slot_on_exp148(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    learned_rank_slot_config: dict[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
    modes: dict[str, dict[str, Any]] | None = None,
    active_modes: list[str] | tuple[str, ...] | None = None,
    n_splits: int = 5,
    fast: bool = False,
    early_stopping_rounds: int = 250,
    max_rows: int | None = None,
    max_train_rows: int | None = None,
    save_models: bool = True,
    save_predictions: bool = True,
    top_n_importance: int = 40,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, base_feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    learned_features_source, learned_source_meta = load_learned_likelihood_ml_features(
        learned_feature_path,
        schema_path=learned_schema_path,
        summary_path=learned_summary_path,
    )

    projection_config = projection_config or {}
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError(
            "LGB OOF U-projection features require nested fold generation. "
            "This first ablation keeps them disabled to avoid leakage."
        )
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    full_frame = pd.concat(
        [
            frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config or {},
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
    full_frame = full_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if full_frame.empty:
        raise ValueError(
            "No shared rows between exp072/exp092 feature surface and learned likelihood features"
        )
    coverage_meta = {
        "base_rows_before_feature_join": int(before_rows),
        "base_wells_before_feature_join": int(before_wells),
        "learned_feature_rows": int(learned_source_meta["rows"]),
        "learned_feature_wells": int(learned_source_meta["wells"]),
        "joined_rows": int(len(full_frame)),
        "joined_wells": int(full_frame["well"].nunique()),
        "dropped_base_rows": int(before_rows - len(full_frame)),
        "dropped_base_wells": int(before_wells - full_frame["well"].nunique()),
        "full_train_coverage_pass": bool(
            before_rows == len(full_frame) and before_wells == full_frame["well"].nunique()
        ),
    }
    learned_rank_slot_config = learned_rank_slot_config or {}
    rank_slot_features, rank_slot_group_columns, rank_slot_summary = (
        build_learned_likelihood_rank_slot_features(
            learned_features_source,
            full_frame,
            learned_rank_slot_config,
        )
    )
    rank_slot_feature_columns = [
        col for col in rank_slot_features.columns if col not in {"id", "well"}
    ]
    full_frame = full_frame.merge(
        rank_slot_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(full_frame) != coverage_meta["joined_rows"]:
        raise ValueError(
            "Learned likelihood rank-slot features changed row coverage: "
            f"{len(full_frame)} of {coverage_meta['joined_rows']}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **rank_slot_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
        index=False,
    )
    rank_slot_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_rank_slot_feature_summary.csv",
        index=False,
    )

    selected_variants = list(variants or [])
    if not selected_variants:
        raise ValueError("No feature ablation variants configured")
    variant_names = [str(variant.get("name")) for variant in selected_variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"Duplicate variant names: {variant_names}")
    mode_map = modes or {}
    selected_modes = list(active_modes or mode_map)
    if not selected_modes:
        raise ValueError("No active LightGBM modes configured")

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    feature_schema_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        if not variant.get("enabled", True):
            continue
        variant_name = str(variant["name"])
        feature_columns = feature_columns_for_variant(
            base_feature_columns,
            feature_group_columns,
            variant,
        )
        for index, feature in enumerate(feature_columns):
            feature_schema_rows.append(
                {
                    "variant": variant_name,
                    "feature_index": int(index),
                    "feature": feature,
                    "is_projection_feature": bool(feature in projection_feature_columns),
                    "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                    "is_learned_likelihood_rank_slot_feature": bool(
                        feature in rank_slot_feature_columns
                    ),
                }
            )
        for mode_name in selected_modes:
            if mode_name not in mode_map:
                raise ValueError(
                    f"active mode is not defined under model.training.modes: {mode_name}"
                )
            metrics, predictions, importance, models, mode_summary = _fit_one_variant_mode(
                variant=variant,
                mode_name=mode_name,
                mode_config=mode_map[mode_name],
                frame=full_frame,
                feature_columns=feature_columns,
                output_dir=output_dir,
                n_splits=n_splits,
                fast=fast,
                early_stopping_rounds=early_stopping_rounds,
                max_train_rows=max_train_rows,
                save_models=save_models,
            )
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
            importance_frames.append(importance)
            model_rows.extend(models)
            mode_summaries.append(mode_summary)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    mean_importance = (
        importance.groupby(["variant", "mode", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            fold_model_records=("importance", "size"),
        )
        .sort_values(["variant", "mode", "mean_importance"], ascending=[True, True, False])
    )
    by_well = _by_well_metrics(predictions)
    bucket_metrics = _bucket_metrics(predictions, full_frame)

    metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    importance.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv", index=False)
    mean_importance.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
        index=False,
    )
    _plot_mean_importance(
        mean_importance,
        output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
        int(top_n_importance),
    )
    if save_predictions:
        predictions.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(feature_schema_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        index=False,
    )

    model_root = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "rank_slot_source_parent": "exp098_selector_rank_slot_features_on_exp073",
        "mode": "learned_likelihood_rank_slot_on_exp148_full_train_rows",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config or {},
        "learned_rank_slot_config": learned_rank_slot_config,
        "projection_feature_groups": projection_group_columns,
        "learned_feature_groups": learned_group_columns,
        "learned_rank_slot_feature_groups": rank_slot_group_columns,
        "n_splits": int(n_splits),
        "variants": selected_variants,
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    lgb_mean = pooled[pooled["model"].eq("lgb_mean")].sort_values("rmse_tvt")
    best = lgb_mean.iloc[0].to_dict() if not lgb_mean.empty else None
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "learned_likelihood_rank_slot_on_exp148_full_train_rows",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "rank_slot_source_parent": "exp098_selector_rank_slot_features_on_exp073",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_variants": variant_names,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "learned_feature_summary": f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
            "learned_rank_slot_feature_summary": (
                f"{OUTPUT_PREFIX}_learned_rank_slot_feature_summary.csv"
            ),
            "feature_importance": f"{OUTPUT_PREFIX}_feature_importance.csv",
            "feature_importance_mean": f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
            "feature_importance_plot": f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
            "predictions": f"{OUTPUT_PREFIX}_predictions.csv.gz" if save_predictions else None,
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "model_manifest": f"{OUTPUT_PREFIX}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_saved_model_inference(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    data_dir: str | Path,
    test_dir: str | Path,
    model_manifest_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    learned_rank_slot_config: dict[str, Any] | None = None,
    variant_name: str = "learned_likelihood_rank_slot_addonly",
    mode_name: str = "cpu_deterministic_threads8",
    model_name: str = "lgb1",
    submission_target_column: str = "tvt",
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
) -> dict[str, Any]:
    import lightgbm as lgb

    try:
        from public_notebook_replay_audit import (
            build_replay_test_frame,
            configure_public_runtime,
        )
    except ModuleNotFoundError:
        from src.public_notebook_replay_audit import (  # type: ignore[import-not-found]
            build_replay_test_frame,
            configure_public_runtime,
        )

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    data_dir = Path(data_dir)
    test_dir = Path(test_dir)
    manifest_path = find_model_manifest(model_manifest_path)
    model_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    projection_config = projection_config or dict(manifest.get("projection_config") or {})
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError("LGB OOF U-projection features are disabled for exp092 inference")

    print(f"loading saved LightGBM boosters from {model_root}", flush=True)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu=use_gpu,
    )
    base_test_frame, test_meta = build_replay_test_frame()
    base_test_frame["id"] = base_test_frame["id"].astype(str)
    base_test_frame["well"] = base_test_frame["well"].astype(str)
    base_feature_columns = [str(col) for col in manifest["feature_source"]["feature_columns"]]
    missing_base = sorted(set(base_feature_columns) - set(base_test_frame.columns))
    if missing_base:
        raise ValueError(f"raw-test replay frame is missing base features: {missing_base[:40]}")
    anchored_frame, anchor_meta = add_inference_anchor_columns(base_test_frame, test_dir)
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        anchored_frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    configured_groups = manifest.get("projection_feature_groups") or {}
    if configured_groups and {
        key: list(value) for key, value in projection_group_columns.items()
    } != {key: list(value) for key, value in configured_groups.items()}:
        raise ValueError("Projection feature groups differ from train manifest")

    variant_configs = {
        str(item["name"]): dict(item)
        for item in manifest.get("variants", [])
        if item.get("enabled", True)
    }
    if variant_name not in variant_configs:
        raise ValueError(f"variant={variant_name} not found in train manifest")
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    test_frame = pd.concat(
        [
            anchored_frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    try:
        rawtest_learned_features, rawtest_learned_meta = load_learned_likelihood_ml_features(
            learned_feature_path,
            schema_path=learned_schema_path,
            summary_path=learned_summary_path,
            feature_filename=EXP145_RAWTEST_ML_FEATURES,
            local_artifacts=EXP145_INFERENCE_ARTIFACTS,
            source_kind="target_free_rawtest_learned_likelihood_ml_features",
        )
    except FileNotFoundError:
        rawtest_learned_features, rawtest_learned_meta = (
            generate_current_test_learned_likelihood_ml_features(
                test_frame=anchored_frame,
                output_dir=output_dir,
            )
        )
    else:
        if not learned_feature_keys_match(rawtest_learned_features, anchored_frame):
            rawtest_learned_features, rawtest_learned_meta = (
                generate_current_test_learned_likelihood_ml_features(
                    test_frame=anchored_frame,
                    output_dir=output_dir,
                )
            )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        rawtest_learned_features,
        test_frame,
        learned_feature_config or dict(manifest.get("learned_feature_config") or {}),
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_join_rows = len(test_frame)
    test_frame = test_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_join_rows:
        raise ValueError(
            "Raw-test learned likelihood features do not cover every replay test row: "
            f"{len(test_frame)} of {before_join_rows}"
        )
    learned_rank_slot_config = learned_rank_slot_config or dict(
        manifest.get("learned_rank_slot_config") or {}
    )
    rank_slot_features, rank_slot_group_columns, rank_slot_summary = (
        build_learned_likelihood_rank_slot_features(
            rawtest_learned_features,
            test_frame,
            learned_rank_slot_config,
        )
    )
    rank_slot_feature_columns = [
        col for col in rank_slot_features.columns if col not in {"id", "well"}
    ]
    test_frame = test_frame.merge(
        rank_slot_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_join_rows:
        raise ValueError(
            "Raw-test learned likelihood rank-slot features do not cover every replay test row: "
            f"{len(test_frame)} of {before_join_rows}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **rank_slot_group_columns,
    }
    configured_learned_groups = manifest.get("learned_feature_groups") or {}
    if configured_learned_groups and {
        key: list(value) for key, value in learned_group_columns.items()
    } != {key: list(value) for key, value in configured_learned_groups.items()}:
        raise ValueError("Learned likelihood feature groups differ from train manifest")
    configured_rank_slot_groups = manifest.get("learned_rank_slot_feature_groups") or {}
    if configured_rank_slot_groups and {
        key: list(value) for key, value in rank_slot_group_columns.items()
    } != {key: list(value) for key, value in configured_rank_slot_groups.items()}:
        raise ValueError("Learned likelihood rank-slot feature groups differ from train manifest")
    feature_columns = feature_columns_for_variant(
        base_feature_columns,
        feature_group_columns,
        variant_configs[variant_name],
    )

    missing_model = sorted(set(feature_columns) - set(test_frame.columns))
    if missing_model:
        raise ValueError(f"test frame is missing model features: {missing_model[:40]}")
    for col in feature_columns:
        test_frame[col] = pd.to_numeric(test_frame[col], errors="raise").astype(np.float32)
    if not np.isfinite(test_frame[feature_columns].to_numpy(np.float32)).all():
        raise ValueError("test feature matrix contains non-finite values")

    model_rows = [
        item
        for item in manifest.get("models", [])
        if str(item.get("variant")) == variant_name
        and str(item.get("mode")) == mode_name
        and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
    ]
    if not model_rows:
        raise ValueError(
            f"No saved models for variant={variant_name} mode={mode_name} model={model_name}"
        )

    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows: list[dict[str, Any]] = []
    for item in model_rows:
        model_file = model_root / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred = booster.predict(x_matrix).astype(np.float32)
        pred_delta += pred / float(len(model_rows))
        loaded_rows.append(
            {
                "variant": item.get("variant"),
                "mode": item.get("mode"),
                "model": item.get("model"),
                "fold": item.get("fold"),
                "file": str(item.get("file")),
                "sha256": item.get("sha256"),
                "rows": int(len(pred)),
            }
        )

    base = test_frame["last_known_tvt"].to_numpy(np.float32)
    pred_tvt = (base + pred_delta).astype(np.float32)
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].to_numpy(),
            "well": test_frame["well"].to_numpy(),
            "variant": variant_name,
            "mode": mode_name,
            "model": model_name,
            "last_known_tvt": base,
            "pred_delta": pred_delta,
            "pred_tvt": pred_tvt,
        }
    )

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    pred_map = dict(zip(predictions["id"].astype(str), predictions["pred_tvt"], strict=False))
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(predictions["pred_tvt"].mean())
    missing_mask = mapped.isna()

    predictions_path = output_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    projection_summary_path = (
        output_dir / f"{OUTPUT_PREFIX}_inference_projection_feature_summary.csv"
    )
    feature_schema_path = output_dir / f"{OUTPUT_PREFIX}_inference_feature_schema.csv"
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    projection_summary.to_csv(projection_summary_path, index=False)
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
        index=False,
    )
    rank_slot_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_learned_rank_slot_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                "is_learned_likelihood_rank_slot_feature": bool(
                    feature in rank_slot_feature_columns
                ),
            }
            for index, feature in enumerate(feature_columns)
        ]
    ).to_csv(feature_schema_path, index=False)

    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    submission_sha = sha256_file(submission_path)
    prediction_sha = prediction_sha256(
        predictions["id"],
        pred_delta,
        label=f"{variant_name}/{mode_name}/{model_name}/test",
    )
    metrics = {
        "variant": variant_name,
        "mode": mode_name,
        "model": model_name,
        "model_count": int(len(model_rows)),
        "feature_count": int(len(feature_columns)),
        "test_rows": int(len(test_frame)),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "prediction_sha256": prediction_sha,
        "submission_sha256": submission_sha,
    }
    pd.DataFrame([metrics]).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_metrics.csv",
        index=False,
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "anchor_source": anchor_meta,
        "learned_feature_groups": learned_group_columns,
        "learned_rank_slot_feature_groups": rank_slot_group_columns,
        "selected": {
            "variant": variant_name,
            "mode": mode_name,
            "model": model_name,
            "model_count": int(len(model_rows)),
        },
        "metrics": metrics,
        "loaded_models": loaded_rows,
        "artifacts": {
            "predictions": predictions_path.name,
            "projection_feature_summary": projection_summary_path.name,
            "learned_feature_summary": f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
            "learned_rank_slot_feature_summary": (
                f"{OUTPUT_PREFIX}_inference_learned_rank_slot_feature_summary.csv"
            ),
            "feature_schema": feature_schema_path.name,
            "metrics": f"{OUTPUT_PREFIX}_inference_metrics.csv",
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
            "submission": str(submission_path),
        },
        "known_followup_risk": "OOF worst-well degradation risk remains unresolved.",
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


# %% [markdown]
# ## 5. Setup and configuration


# %%
def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

active_variants = [
    v
    for v in cfg_get(config, "model.feature_ablation.active_variants", [])
    if v.get("enabled", True)
]
active_modes = cfg_get(config, "model.training.active_modes", [])
lgb_config_count = 3
n_folds = int(cfg_get(config, "validation.n_folds", 5))
booster_count = len(active_variants) * len(active_modes) * lgb_config_count * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Learned likelihood parent:", cfg_get(config, "lineage.learned_likelihood_parent"))
print("Rank-slot source parent:", cfg_get(config, "lineage.rank_slot_source_parent"))
print("Kaggle GPU enabled:", cfg_get(config, "runtime.kaggle.enable_gpu"))
print("Active modes:", active_modes)
print("Active variants:", [v["name"] for v in active_variants])
print("Planned LightGBM configs:", lgb_config_count, "folds:", n_folds, "boosters:", booster_count)


# %% [markdown]
# ## 6. Input and feature contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
print("exp072 full replay train cache:", cache_path)
print("exp145 full-train learned likelihood feature cache:", learned_path)

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
learned_preview, learned_meta = load_learned_likelihood_ml_features(
    cfg_get(config, "data.learned_likelihood_train_features_local"),
    schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
)
print(
    "learned feature rows:",
    learned_meta["rows"],
    "wells:",
    learned_meta["wells"],
    "columns:",
    learned_meta["columns"],
)
display(
    base_preview[
        [
            c
            for c in [
                "id",
                "well",
                "target",
                "last_known_tvt",
                "z",
                "md_since",
                "pf_ancc",
                "likpf_mean_d",
            ]
            if c in base_preview.columns
        ]
    ]
)
display(
    learned_preview.head()[
        [
            "id",
            "well",
            "learned_prob_pf_ancc",
            "learned_pred_abs_error_pf_ancc",
            "learned_prob_likpf_mean",
            "learned_pred_abs_error_likpf_mean",
            "learned_prob_entropy",
        ]
    ]
)


# %% [markdown]
# ## 7. Train learned rank-slot variant

# %%
summary = run_learned_likelihood_rank_slot_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    learned_rank_slot_config=cfg_get(config, "model.learned_likelihood_rank_slot", {}),
    variants=cfg_get(config, "model.feature_ablation.active_variants", []),
    modes=cfg_get(config, "model.training.modes", {}),
    active_modes=cfg_get(config, "model.training.active_modes", []),
    n_splits=int(cfg_get(config, "validation.n_folds", 5)),
    fast=bool(cfg_get(config, "audit.fast", False)),
    early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
    max_rows=cfg_get(config, "model.training.max_rows"),
    max_train_rows=cfg_get(config, "model.training.max_train_rows"),
    save_models=bool(cfg_get(config, "model.training.save_models", True)),
    save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
    top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 80)),
)
print(
    json.dumps(
        {
            "status": summary["status"],
            "active_modes": summary["active_modes"],
            "best_lgb_mean_by_rmse_tvt": summary["best_lgb_mean_by_rmse_tvt"],
            "feature_join_coverage": summary["feature_join_coverage"],
        },
        indent=2,
    )
)


# %% [markdown]
# ## 8. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
by_well = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv")
bucket_metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
projection_summary = pd.read_csv(
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv"
)
learned_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv")
rank_slot_summary = pd.read_csv(
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_learned_rank_slot_feature_summary.csv"
)
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

pooled = metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt")
display(pooled)
display(learned_summary)
display(rank_slot_summary)
display(projection_summary)
display(bucket_metrics.head(50))
display(by_well.head(30))
display(importance_mean.head(80))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
print(
    "Feature importance plot:",
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
)
