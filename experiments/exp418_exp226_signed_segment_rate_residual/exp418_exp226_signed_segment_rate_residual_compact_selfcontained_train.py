# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp418 exp226 signed segment-rate residual — Stage 0 + Stage 1
#
# This implementation keeps the saved exp333 strict-nested exp226 predictions,
# fold identity, target-free 136-feature surface, and fixed LightGBM config.
# The only scientific change is the target/application pair: each well receives
# 16 zero-intercept residual rates (ft/row), and predicted rates are integrated
# continuously from an exact zero correction at the first suffix row.
#
# The saved exp333 nested artifact is mandatory. This notebook never fits or
# regenerates exp226. Stage 0 and Stage 1 execution remain separately
# fail-closed in `config.yaml`; inference and submission are outside scope.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment boundary
# 2. Runtime, configuration, path, SHA, and serialization helpers
# 3. Frozen implementation and execution contract
# 4. Saved exp333 nested-input and exp226 target-free checks
# 5. K16 cumulative-rate basis and target helpers
# 6. Truth-free freeze and late truth attachment
# 7. Stage 0 continuous-rate oracle readout
# 8. Target-free exp333-compatible 136-feature surface
# 9. Strict-nested segment-rate samples and LightGBM training
# 10. Continuous row integration, metrics, tails, and promotion gate
# 11. Generated artifacts and guarded orchestration
# 12. Setup and contract preview

# %% [markdown]
# ## 1. Imports and immutable experiment boundary

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import importlib.util
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp418_exp226_signed_segment_rate_residual"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KEY_COLUMNS = ("well_id", "row_idx")
TARGET_FREE_COLUMNS = ("well_id", "row_idx", "suffix_offset", "tvt_pred", "fold")
TRUTH_COLUMNS = ("well_id", "row_idx", "tvt_true")
NESTED_COLUMNS = (
    "outer_fold",
    "role",
    "inner_fold",
    "well_id",
    "row_idx",
    "suffix_offset",
    "segment_id",
    "tvt_pred",
)
K_SEGMENTS = 16
ALLOWED_FEATURE_GROUPS = (
    "projection_correction",
    "u_disagreement",
    "gr_wavelet_rotation_confidence",
)
STRUCTURAL_FEATURE_COLUMNS = (
    "segment_id",
    "segment_position",
    "segment_row_count",
    "segment_md_span",
    "exp226_pred_mean",
    "exp226_pred_start",
    "exp226_pred_end_minus_start",
)
FORBIDDEN_PRE_FREEZE_TOKENS = (
    "tvt_true",
    "target",
    "error",
    "residual",
    "oracle",
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP418_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)

# %% [markdown]
# ## 2. Runtime, configuration, path, SHA, and serialization helpers

# %%
def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd()


def load_config() -> dict[str, Any]:
    candidates = (
        (Path.cwd() / "config.yaml", experiment_dir() / "config.yaml")
        if Path.cwd() == experiment_dir()
        else (experiment_dir() / "config.yaml", Path.cwd() / "config.yaml")
    )
    for path in candidates:
        if path.is_file():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            return value
    raise FileNotFoundError("exp418 config.yaml was not found")


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)]
    digest = hashlib.sha256(canonical_json_bytes({"columns": list(columns)}))
    digest.update(
        pd.util.hash_pandas_object(selected, index=False).to_numpy(np.uint64).tobytes()
    )
    return digest.hexdigest()


def canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = frame.loc[:, list(columns)].to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def write_csv_gzip(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )
    return artifact_evidence(path)


def artifact_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }
    if path.suffix == ".gz":
        evidence["decompressed_sha256"] = sha256_gzip_decompressed(path)
    return evidence


def resolve_existing(filename: str, patterns: Iterable[str]) -> Path:
    roots = (project_root(), Path.cwd())
    seen: set[Path] = set()
    resolved: list[Path] = []
    for raw in patterns:
        pattern = str(raw)
        wildcard = any(token in pattern for token in ("*", "?", "["))
        candidates: list[Path] = []
        if wildcard and Path(pattern).is_absolute():
            candidates.extend(Path(item) for item in glob.glob(pattern, recursive=True))
        elif wildcard:
            for root in roots:
                candidates.extend(root.glob(pattern))
        else:
            path = Path(pattern)
            candidates.append(path if path.is_absolute() else project_root() / path)
        for path in candidates:
            if path.is_file() and path not in seen:
                seen.add(path)
                resolved.append(path)
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename)):
            if path.is_file() and path not in seen:
                seen.add(path)
                resolved.append(path)
    if not resolved:
        raise FileNotFoundError(f"could not resolve required input {filename}")
    return resolved[0]


def output_artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "artifacts"
    return experiment_dir() / "artifacts"


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(truth, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64), dtype=np.float64)))

# %% [markdown]
# ## 3. Frozen implementation and execution contract

# %%
def validate_implementation_contract(
    config: Mapping[str, Any], *, execution_stage: str | None = None
) -> dict[str, Any]:
    exact = {
        "experiment.route": "ensemble",
        "implementation.enabled": True,
        "implementation.scope": "stage_0_and_stage_1_train",
        "implementation.stage_0_enabled": True,
        "implementation.stage_1_enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.n_outer_folds": 5,
        "validation.n_inner_folds": 4,
        "segmentation.k_segments": 16,
        "rate_target.name": "zero_intercept_k16_cumulative_residual_rate",
        "rate_target.residual_sign": (
            "true_tvt_minus_strict_nested_exp226_prediction"
        ),
        "rate_target.unit": "ft_per_row",
        "rate_target.basis.first_row_all_zero": True,
        "rate_target.basis.interval_assignment": "destination_row_segment",
        "rate_target.solver.name": "numpy_linalg_lstsq",
        "rate_target.solver.dtype": "float64",
        "rate_target.solver.rcond": None,
        "rate_target.solver.intercept": False,
        "rate_target.solver.ridge": 0.0,
        "rate_target.segment_sample_weight": "segment_row_count",
        "correction.name": "continuous_k16_rate_integration",
        "correction.first_unknown_row_correction_ft": 0.0,
        "correction.clipping": "none",
        "correction.shrinkage": "none",
        "correction.taper": "none",
        "correction.interpolation": "none",
        "correction.boundary_level_step": "none",
        "correction.absolute_reanchor": "none",
        "features.expected_feature_count": 136,
        "features.row_to_segment_aggregation": "finite_float64_mean",
        "model.model_configs": 1,
        "model.trained_folds": 5,
        "model.total_boosters": 5,
        "model.sample_weight": "segment_row_count",
        "model.params.random_state": 0,
        "model.params.deterministic": True,
        "model.params.force_col_wise": True,
        "model.params.n_jobs": 8,
        "model.params.num_threads": 8,
        "execution_contract.implementation_approved": True,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.boosters": 5,
        "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.exp226_fits": 0,
        (
            "execution_contract.stage_1_if_stage_0_pass_and_separately_approved."
            "parent_control_retraining"
        ): False,
        "execution_contract.inference_approved": False,
        "execution_contract.submission_approved": False,
    }
    changed = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if changed:
        raise ValueError(f"exp418 frozen implementation contract changed: {changed}")
    if tuple(get_nested(config, "features.row_groups", ())) != ALLOWED_FEATURE_GROUPS:
        raise ValueError("exp418 feature group allowlist changed")
    if tuple(get_nested(config, "features.structural_columns", ())) != (
        STRUCTURAL_FEATURE_COLUMNS
    ):
        raise ValueError("exp418 structural feature contract changed")
    if tuple(get_nested(config, "model.active_variants", ())) != (
        "signed_k16_rate",
    ):
        raise ValueError("exp418 must contain one signed-rate variant")
    if execution_stage is not None:
        selected = get_nested(config, "execution_contract.selected_stage")
        approval_key = {
            "stage_0": "execution_contract.stage_0_run_approved",
            "stage_1": "execution_contract.stage_1_run_approved",
        }.get(execution_stage)
        if approval_key is None:
            raise ValueError(f"unknown execution stage: {execution_stage}")
        if not (
            selected == execution_stage
            and bool(get_nested(config, "execution_contract.kaggle_push_approved"))
            and bool(get_nested(config, approval_key))
        ):
            raise RuntimeError(f"{execution_stage} execution is not authorized")
    return {
        "fixed_values": exact,
        "selected_stage": get_nested(config, "execution_contract.selected_stage"),
        "stage_0_run_approved": bool(
            get_nested(config, "execution_contract.stage_0_run_approved")
        ),
        "stage_1_run_approved": bool(
            get_nested(config, "execution_contract.stage_1_run_approved")
        ),
    }

# %% [markdown]
# ## 4. Saved exp333 nested-input and exp226 target-free checks

# %%
def reject_forbidden_pre_freeze_columns(columns: Sequence[str]) -> None:
    lowered = tuple(str(column).lower() for column in columns)
    hits = sorted(
        column
        for column in lowered
        if any(token in column for token in FORBIDDEN_PRE_FREEZE_TOKENS)
    )
    if hits:
        raise ValueError(f"forbidden columns requested before freeze: {hits}")


def _manifest_record(manifest: pd.DataFrame, filename: str) -> dict[str, Any]:
    rows = manifest.loc[manifest["filename"].astype(str).eq(filename)]
    if len(rows) != 1:
        raise ValueError(f"exp333 SHA manifest must contain exactly one {filename}")
    return rows.iloc[0].to_dict()


def validate_file_against_manifest(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    file_sha = sha256_file(path)
    if file_sha != str(record["file_sha256"]):
        raise ValueError(f"saved exp333 file SHA mismatch: {path.name}")
    evidence: dict[str, Any] = {
        "filename": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha,
    }
    expected_decompressed = record.get("decompressed_sha256")
    if path.suffix == ".gz" and pd.notna(expected_decompressed):
        decompressed = sha256_gzip_decompressed(path)
        if decompressed != str(expected_decompressed):
            raise ValueError(f"saved exp333 decompressed SHA mismatch: {path.name}")
        evidence["decompressed_sha256"] = decompressed
    return evidence


@dataclass(frozen=True)
class FrozenExp333Inputs:
    nested: pd.DataFrame
    fold_manifest: pd.DataFrame
    feature_schema: pd.DataFrame
    evidence: dict[str, Any]


def validate_nested_predictions(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_expected_counts: bool,
) -> pd.DataFrame:
    missing = sorted(set(NESTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"saved exp333 nested predictions are missing {missing}")
    nested = frame.loc[:, list(NESTED_COLUMNS)].copy()
    reject_forbidden_pre_freeze_columns(nested.columns)
    nested["well_id"] = nested["well_id"].astype(str)
    nested["role"] = nested["role"].astype(str)
    for column in (
        "outer_fold",
        "inner_fold",
        "row_idx",
        "suffix_offset",
        "segment_id",
    ):
        values = pd.to_numeric(nested[column], errors="raise").to_numpy(np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain finite integers")
        nested[column] = values.astype(np.int64)
    nested["tvt_pred"] = pd.to_numeric(nested["tvt_pred"], errors="raise").astype(
        np.float64
    )
    if not np.isfinite(nested["tvt_pred"].to_numpy()).all():
        raise ValueError("saved exp333 nested predictions contain non-finite values")
    if nested.duplicated(["outer_fold", "role", "well_id", "row_idx"]).any():
        raise ValueError("saved exp333 nested context keys are not unique")
    if set(nested["role"]) != {"inner_oof_train", "outer_valid"}:
        raise ValueError("saved exp333 nested roles changed")
    if set(nested["outer_fold"]) != {0, 1, 2, 3, 4}:
        raise ValueError("saved exp333 outer folds changed")
    if int(nested["segment_id"].min()) != 0 or int(nested["segment_id"].max()) != 15:
        raise ValueError("saved exp333 K16 segment range changed")
    for (_outer_fold, _well_id), part in nested.groupby(
        ["outer_fold", "well_id"], sort=False
    ):
        ordered = part.sort_values("suffix_offset", kind="mergesort")
        expected_offset = np.arange(len(ordered), dtype=np.int64)
        if not np.array_equal(ordered["suffix_offset"].to_numpy(), expected_offset):
            raise ValueError("saved exp333 suffix offsets are not contiguous")
        expected_segment = exact_k16_segment_ids(len(ordered))
        if not np.array_equal(ordered["segment_id"].to_numpy(), expected_segment):
            raise ValueError("saved exp333 K16 assignment differs from exp226")
    if enforce_expected_counts:
        expected_rows = int(get_nested(config, "validation.expected_rows"))
        expected_wells = int(get_nested(config, "validation.expected_wells"))
        if len(nested) != expected_rows * 5:
            raise ValueError("saved exp333 nested row coverage changed")
        if nested["well_id"].nunique() != expected_wells:
            raise ValueError("saved exp333 nested well coverage changed")
    return nested.sort_values(
        ["outer_fold", "role", "well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


def load_frozen_exp333_inputs(
    config: Mapping[str, Any], *, enforce_expected_counts: bool = True
) -> FrozenExp333Inputs:
    spec = get_nested(config, "data.exp333_stage1")
    sha_path = resolve_existing(
        str(spec["sha_manifest_filename"]), spec["sha_manifest_patterns"]
    )
    if sha256_file(sha_path) != str(spec["expected_sha_manifest_file_sha256"]):
        raise ValueError("saved exp333 SHA manifest is not the frozen version")
    manifest = pd.read_csv(sha_path)
    if not {"filename", "file_sha256"}.issubset(manifest.columns):
        raise ValueError("saved exp333 SHA manifest schema is incomplete")

    nested_path = resolve_existing(
        str(spec["nested_prediction_filename"]), spec["nested_prediction_patterns"]
    )
    fold_path = resolve_existing(
        str(spec["fold_manifest_filename"]), spec["fold_manifest_patterns"]
    )
    schema_path = resolve_existing(
        str(spec["feature_schema_filename"]), spec["feature_schema_patterns"]
    )
    nested_evidence = validate_file_against_manifest(
        nested_path, _manifest_record(manifest, nested_path.name)
    )
    fold_evidence = validate_file_against_manifest(
        fold_path, _manifest_record(manifest, fold_path.name)
    )
    schema_evidence = validate_file_against_manifest(
        schema_path, _manifest_record(manifest, schema_path.name)
    )
    if schema_evidence["file_sha256"] != str(
        spec["expected_feature_schema_file_sha256"]
    ):
        raise ValueError("saved exp333 feature-schema file SHA mismatch")

    header = tuple(str(column) for column in pd.read_csv(nested_path, nrows=0).columns)
    reject_forbidden_pre_freeze_columns(header)
    nested = validate_nested_predictions(
        pd.read_csv(nested_path, usecols=list(NESTED_COLUMNS)),
        config,
        enforce_expected_counts=enforce_expected_counts,
    )
    fold_manifest = pd.read_csv(fold_path, dtype={"well_id": str})
    expected_fold_columns = ("outer_fold", "well_id", "inner_fold", "inner_digest")
    if tuple(fold_manifest.columns) != expected_fold_columns:
        raise ValueError("saved exp333 fold manifest schema changed")
    feature_schema = pd.read_csv(schema_path)
    expected_schema_columns = (
        "feature_name",
        "source_group",
        "row_to_segment_aggregation",
        "all_nonfinite_policy",
    )
    if tuple(feature_schema.columns) != expected_schema_columns:
        raise ValueError("saved exp333 feature schema columns changed")
    if len(feature_schema) != int(spec["expected_feature_count"]):
        raise ValueError("saved exp333 feature count changed")
    schema_content_sha = hashed_frame_sha256(feature_schema, expected_schema_columns)
    if schema_content_sha != str(spec["expected_feature_schema_content_sha256"]):
        raise ValueError("saved exp333 feature-schema content SHA mismatch")
    evidence = {
        "sha_manifest": {
            "path": str(sha_path),
            "file_sha256": sha256_file(sha_path),
        },
        "nested": nested_evidence,
        "fold_manifest": fold_evidence,
        "feature_schema": {
            **schema_evidence,
            "content_sha256": schema_content_sha,
        },
        "truth_or_error_columns_loaded": 0,
        "exp226_fits": 0,
    }
    return FrozenExp333Inputs(nested, fold_manifest, feature_schema, evidence)


def resolve_exp226_oof(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.exp226_oof")
    return resolve_existing(str(spec["filename"]), spec["patterns"])


def load_exp226_target_free(
    path: Path,
    config: Mapping[str, Any],
    *,
    enforce_expected_counts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof")
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("saved exp226 decompressed SHA mismatch")
    columns = tuple(str(value) for value in spec["target_free_columns"])
    if columns != TARGET_FREE_COLUMNS:
        raise ValueError("saved exp226 target-free allowlist changed")
    reject_forbidden_pre_freeze_columns(columns)
    frame = pd.read_csv(path, usecols=list(columns))
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_pred"] = pd.to_numeric(frame["tvt_pred"], errors="raise").astype(
        np.float64
    )
    frame = frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("saved exp226 row keys are not unique")
    expected_offset = frame.groupby("well_id", sort=False).cumcount().to_numpy(np.int64)
    if not np.array_equal(frame["suffix_offset"].to_numpy(), expected_offset):
        raise ValueError("saved exp226 suffix offsets are not contiguous")
    if enforce_expected_counts:
        if len(frame) != int(get_nested(config, "validation.expected_rows")):
            raise ValueError("saved exp226 row coverage changed")
        if frame["well_id"].nunique() != int(
            get_nested(config, "validation.expected_wells")
        ):
            raise ValueError("saved exp226 well coverage changed")
    return frame, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "loaded_columns": list(columns),
        "truth_or_error_columns_loaded": 0,
    }


def outer_valid_parent_parity(
    nested: pd.DataFrame, saved_exp226: pd.DataFrame
) -> float:
    outer_valid = nested.loc[nested["role"].eq("outer_valid")]
    parity = outer_valid.merge(
        saved_exp226[["well_id", "row_idx", "fold", "tvt_pred"]],
        left_on=["well_id", "row_idx", "outer_fold"],
        right_on=["well_id", "row_idx", "fold"],
        how="left",
        validate="one_to_one",
        suffixes=("_nested", "_saved"),
    )
    if len(parity) != len(saved_exp226) or parity["tvt_pred_saved"].isna().any():
        raise ValueError("saved exp333 outer-valid parity coverage failed")
    return float(
        np.max(
            np.abs(
                parity["tvt_pred_nested"].to_numpy(np.float64)
                - parity["tvt_pred_saved"].to_numpy(np.float64)
            )
        )
    )

# %% [markdown]
# ## 5. K16 cumulative-rate basis and target helpers

# %%
def exact_k16_segment_ids(length: int, k_segments: int = K_SEGMENTS) -> np.ndarray:
    if length <= 0:
        raise ValueError("unknown suffix length must be positive")
    if k_segments != K_SEGMENTS:
        raise ValueError("exp418 is fixed to K16")
    edges = np.linspace(0.0, float(length), k_segments + 1)
    one_based_row = np.arange(1, length + 1, dtype=np.float64)
    return np.clip(
        np.searchsorted(edges[1:], one_based_row, side="left"),
        0,
        k_segments - 1,
    ).astype(np.int16)


def cumulative_rate_basis(
    segment_id: np.ndarray, k_segments: int = K_SEGMENTS
) -> np.ndarray:
    segment = np.asarray(segment_id, dtype=np.int64)
    if segment.ndim != 1 or len(segment) == 0:
        raise ValueError("segment_id must be a non-empty vector")
    if k_segments != K_SEGMENTS:
        raise ValueError("exp418 cumulative basis is fixed to K16")
    if int(segment.min()) < 0 or int(segment.max()) >= k_segments:
        raise ValueError("segment ids are outside K16")
    basis = np.zeros((len(segment), k_segments), dtype=np.float64)
    if len(segment) > 1:
        increments = np.zeros((len(segment) - 1, k_segments), dtype=np.float64)
        increments[np.arange(len(segment) - 1), segment[1:]] = 1.0
        basis[1:] = np.cumsum(increments, axis=0, dtype=np.float64)
    return basis


def integrate_segment_rates_sequential(
    segment_id: np.ndarray, rates: np.ndarray
) -> np.ndarray:
    segment = np.asarray(segment_id, dtype=np.int64)
    rate = np.asarray(rates, dtype=np.float64)
    if rate.shape != (K_SEGMENTS,):
        raise ValueError("exp418 requires exactly 16 predicted rates")
    correction = np.zeros(len(segment), dtype=np.float64)
    for row in range(1, len(segment)):
        correction[row] = correction[row - 1] + rate[segment[row]]
    return correction


@dataclass(frozen=True)
class RateSolution:
    rates: np.ndarray
    correction: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition_number: float
    integration_max_abs_diff: float


def solve_zero_intercept_rates(
    residual: np.ndarray, segment_id: np.ndarray
) -> RateSolution:
    error = np.asarray(residual, dtype=np.float64)
    basis = cumulative_rate_basis(segment_id)
    if error.shape != (len(basis),) or not np.isfinite(error).all():
        raise ValueError("residual must be a finite vector aligned to the basis")
    rates, _sse, rank, singular = np.linalg.lstsq(basis, error, rcond=None)
    correction = basis @ rates
    sequential = integrate_segment_rates_sequential(segment_id, rates)
    max_abs = float(np.max(np.abs(correction - sequential)))
    positive = singular[singular > 0.0]
    condition = (
        float(positive.max() / positive.min()) if len(positive) else float("inf")
    )
    return RateSolution(
        rates=np.asarray(rates, dtype=np.float64),
        correction=np.asarray(correction, dtype=np.float64),
        rank=int(rank),
        singular_values=np.asarray(singular, dtype=np.float64),
        condition_number=condition,
        integration_max_abs_diff=max_abs,
    )


def integrate_predicted_rates(
    segment_id: np.ndarray, rates: np.ndarray
) -> tuple[np.ndarray, float]:
    basis = cumulative_rate_basis(segment_id)
    matrix = basis @ np.asarray(rates, dtype=np.float64)
    sequential = integrate_segment_rates_sequential(segment_id, rates)
    return matrix, float(np.max(np.abs(matrix - sequential)))


def rate_basis_content_sha256(rows: pd.DataFrame) -> str:
    digest = hashlib.sha256(b"exp418|zero_intercept_k16_rate_basis|float64|v1")
    ordered = rows.sort_values(["well_id", "suffix_offset"], kind="mergesort")
    for well_id, part in ordered.groupby("well_id", sort=True):
        segment = part["segment_id"].to_numpy(np.int16)
        basis = cumulative_rate_basis(segment)
        digest.update(str(well_id).encode())
        digest.update(np.asarray([len(part)], dtype="<i8").tobytes())
        digest.update(segment.astype("<i2", copy=False).tobytes())
        digest.update(basis.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()

# %% [markdown]
# ## 6. Truth-free freeze and late truth attachment

# %%
def build_truth_free_freeze(
    frozen: FrozenExp333Inputs,
    saved_exp226: pd.DataFrame,
    exp226_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    outer_valid = frozen.nested.loc[frozen.nested["role"].eq("outer_valid")].copy()
    parity = outer_valid_parent_parity(frozen.nested, saved_exp226)
    maximum_parity = 1e-8
    if parity > maximum_parity:
        raise ValueError(f"saved exp333 outer-valid parity failed: {parity}")
    basis_sha = rate_basis_content_sha256(outer_valid)
    freeze = {
        "stage": "truth_free_saved_exp333_nested_and_rate_basis",
        "exp333_sha_manifest_file_sha256": frozen.evidence["sha_manifest"][
            "file_sha256"
        ],
        "nested_file_sha256": frozen.evidence["nested"]["file_sha256"],
        "nested_decompressed_sha256": frozen.evidence["nested"].get(
            "decompressed_sha256"
        ),
        "nested_logical_sha256": hashed_frame_sha256(
            frozen.nested,
            (
                "outer_fold",
                "role",
                "inner_fold",
                "well_id",
                "row_idx",
                "tvt_pred",
            ),
        ),
        "fold_manifest_logical_sha256": hashed_frame_sha256(
            frozen.fold_manifest,
            ("outer_fold", "well_id", "inner_fold", "inner_digest"),
        ),
        "segment_assignment_logical_sha256": hashed_frame_sha256(
            frozen.nested,
            (
                "outer_fold",
                "role",
                "inner_fold",
                "well_id",
                "row_idx",
                "segment_id",
            ),
        ),
        "feature_schema_content_sha256": frozen.evidence["feature_schema"][
            "content_sha256"
        ],
        "expected_exp333_feature_freeze_sha256": get_nested(
            config, "data.exp333_stage1.expected_feature_freeze_sha256"
        ),
        "saved_exp226_decompressed_sha256": exp226_evidence[
            "decompressed_sha256"
        ],
        "outer_valid_parent_parity_max_abs_ft": parity,
        "rate_basis_sha256": basis_sha,
        "rows": len(outer_valid),
        "wells": int(outer_valid["well_id"].nunique()),
        "segments": int(
            outer_valid[["well_id", "segment_id"]].drop_duplicates().shape[0]
        ),
        "truth_or_error_columns_loaded_before_freeze": 0,
        "exp226_fits": 0,
        "parent_control_retraining": False,
    }
    if freeze["rows"] != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("outer-valid row count changed before freeze")
    if freeze["wells"] != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("outer-valid well count changed before freeze")
    if freeze["segments"] != int(get_nested(config, "validation.expected_segments")):
        raise ValueError("outer-valid segment count changed before freeze")
    freeze["truth_free_contract_sha256"] = mapping_sha256(freeze)
    return freeze


def load_exp226_truth(path: Path, *, truth_free_contract_sha256: str) -> pd.DataFrame:
    if not truth_free_contract_sha256:
        raise ValueError("late truth requires a frozen truth-free contract")
    truth = pd.read_csv(path, usecols=list(TRUTH_COLUMNS))
    truth["well_id"] = truth["well_id"].astype(str)
    truth["row_idx"] = pd.to_numeric(truth["row_idx"], errors="raise").astype(np.int64)
    truth["tvt_true"] = pd.to_numeric(truth["tvt_true"], errors="raise").astype(
        np.float64
    )
    if truth.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("late truth row keys are not unique")
    if not np.isfinite(truth["tvt_true"].to_numpy()).all():
        raise ValueError("late truth contains non-finite values")
    return truth.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def attach_truth_after_freeze(
    target_free: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_free_contract_sha256: str,
) -> pd.DataFrame:
    if not truth_free_contract_sha256:
        raise ValueError("late truth requires a frozen truth-free contract")
    joined = target_free.merge(
        truth,
        on=list(KEY_COLUMNS),
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(joined) != len(target_free) or joined["tvt_true"].isna().any():
        raise ValueError("late truth join did not preserve row coverage")
    return joined

# %% [markdown]
# ## 7. Stage 0 continuous-rate oracle readout

# %%
@dataclass(frozen=True)
class OracleReadout:
    row_readout: pd.DataFrame
    rate_targets: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict[str, Any]


def build_rate_target_rows(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = joined.sort_values(["well_id", "suffix_offset"], kind="mergesort").copy()
    correction = np.empty(len(ordered), dtype=np.float64)
    records: list[dict[str, Any]] = []
    ranks: list[int] = []
    conditions: list[float] = []
    integration_diffs: list[float] = []
    first_corrections: list[float] = []
    for well_id, positions in ordered.groupby("well_id", sort=True).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        part = ordered.iloc[index]
        residual = (
            part["tvt_true"].to_numpy(np.float64)
            - part["tvt_pred"].to_numpy(np.float64)
        )
        solution = solve_zero_intercept_rates(
            residual, part["segment_id"].to_numpy(np.int16)
        )
        correction[index] = solution.correction
        ranks.append(solution.rank)
        conditions.append(solution.condition_number)
        integration_diffs.append(solution.integration_max_abs_diff)
        first_corrections.append(float(solution.correction[0]))
        counts = (
            part.groupby("segment_id", sort=True, observed=True)
            .size()
            .reindex(range(K_SEGMENTS), fill_value=0)
            .to_numpy(np.int64)
        )
        fold = int(part["outer_fold"].iloc[0])
        for segment_id, rate in enumerate(solution.rates):
            records.append(
                {
                    "outer_fold": fold,
                    "well_id": str(well_id),
                    "segment_id": segment_id,
                    "segment_row_count": int(counts[segment_id]),
                    "segment_rate_target": float(rate),
                    "basis_rank": solution.rank,
                    "basis_condition_number": solution.condition_number,
                }
            )
    ordered["rate_correction"] = correction
    ordered["tvt_pred_rate_oracle"] = (
        ordered["tvt_pred"].to_numpy(np.float64) + correction
    )
    rate_targets = pd.DataFrame(records).sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    diagnostics = {
        "required_rank": K_SEGMENTS,
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "condition_number_min": min(conditions),
        "condition_number_median": float(np.median(conditions)),
        "condition_number_max": max(conditions),
        "finite_target_fraction": float(
            np.isfinite(rate_targets["segment_rate_target"]).mean()
        ),
        "finite_correction_fraction": float(np.isfinite(correction).mean()),
        "first_row_correction_abs_max_ft": float(
            np.max(np.abs(first_corrections))
        ),
        "matrix_vs_sequential_integration_abs_max_ft": max(integration_diffs),
    }
    return ordered, rate_targets, diagnostics


def build_stage0_oracle_readout(
    outer_valid: pd.DataFrame,
    truth: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    truth_free_contract_sha256: str,
    enforce_expected_counts: bool = True,
) -> OracleReadout:
    joined = attach_truth_after_freeze(
        outer_valid,
        truth,
        truth_free_contract_sha256=truth_free_contract_sha256,
    )
    rows, targets, diagnostics = build_rate_target_rows(joined)
    fold_records: list[dict[str, Any]] = []
    minimum_fold_gain = float(get_nested(config, "stage_0_gate.minimum_fold_gain_vs_exp226_ft"))
    for fold in range(int(get_nested(config, "validation.n_outer_folds"))):
        part = rows.loc[rows["outer_fold"].eq(fold)]
        base_score = rmse(part["tvt_true"].to_numpy(), part["tvt_pred"].to_numpy())
        oracle_score = rmse(
            part["tvt_true"].to_numpy(), part["tvt_pred_rate_oracle"].to_numpy()
        )
        fold_records.append(
            {
                "outer_fold": fold,
                "rows": len(part),
                "wells": int(part["well_id"].nunique()),
                "exp226_rmse": base_score,
                "rate_oracle_rmse": oracle_score,
                "gain_vs_exp226_ft": base_score - oracle_score,
                "gate_pass": bool(base_score - oracle_score >= minimum_fold_gain),
            }
        )
    fold_metrics = pd.DataFrame(fold_records)
    base_rmse = rmse(rows["tvt_true"].to_numpy(), rows["tvt_pred"].to_numpy())
    oracle_rmse = rmse(
        rows["tvt_true"].to_numpy(), rows["tvt_pred_rate_oracle"].to_numpy()
    )
    expected_base = float(get_nested(config, "data.exp226_oof.expected_rmse"))
    base_parity = abs(base_rmse - expected_base)
    gates = get_nested(config, "stage_0_gate")
    technical_checks = {
        "row_count": (
            not enforce_expected_counts
            or len(rows) == int(gates["required_rows"])
        ),
        "well_count": (
            not enforce_expected_counts
            or rows["well_id"].nunique() == int(gates["required_wells"])
        ),
        "segment_count": (
            not enforce_expected_counts
            or len(targets) == int(gates["required_segments"])
        ),
        "basis_rank": diagnostics["rank_min"]
        == int(gates["required_basis_rank_per_well"]),
        "finite_target": diagnostics["finite_target_fraction"]
        == float(gates["required_finite_target_fraction"]),
        "finite_correction": diagnostics["finite_correction_fraction"]
        == float(gates["required_finite_correction_fraction"]),
        "first_row_anchor": diagnostics["first_row_correction_abs_max_ft"]
        <= float(gates["maximum_first_row_correction_abs_ft"]),
        "integration_parity": diagnostics[
            "matrix_vs_sequential_integration_abs_max_ft"
        ]
        <= float(gates["maximum_matrix_vs_sequential_integration_abs_ft"]),
        "base_parity": not enforce_expected_counts or base_parity <= 1e-8,
    }
    scientific_checks = {
        "pooled_gain": base_rmse - oracle_rmse
        >= float(gates["minimum_oracle_gain_vs_exp226_ft"]),
        "all_fold_gains": int(fold_metrics["gate_pass"].sum())
        == int(gates["required_improved_folds"]),
    }
    technical_pass = all(bool(value) for value in technical_checks.values())
    scientific_pass = technical_pass and all(
        bool(value) for value in scientific_checks.values()
    )
    first_rows = rows.groupby("well_id", sort=False).head(1)
    first_row_parity = float(
        np.max(
            np.abs(
                first_rows["tvt_pred_rate_oracle"].to_numpy(np.float64)
                - first_rows["tvt_pred"].to_numpy(np.float64)
            )
        )
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "status": "completed" if technical_pass else "technical_fail",
        "decision": "PASS_STAGE0" if scientific_pass else "FAIL_CLOSE_BRANCH",
        "technical_pass": technical_pass,
        "scientific_pass": scientific_pass,
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "rows": len(rows),
        "wells": int(rows["well_id"].nunique()),
        "segments": len(targets),
        "exp226_rmse": base_rmse,
        "expected_exp226_rmse": expected_base,
        "exp226_rmse_parity_abs_ft": base_parity,
        "rate_oracle_rmse": oracle_rmse,
        "gain_vs_exp226_ft": base_rmse - oracle_rmse,
        "first_row_prediction_parity_abs_max_ft": first_row_parity,
        "diagnostics": diagnostics,
        "rate_target_sha256": hashed_frame_sha256(
            targets,
            (
                "outer_fold",
                "well_id",
                "segment_id",
                "segment_row_count",
                "segment_rate_target",
            ),
        ),
        "oracle_prediction_persisted": False,
        "exp226_fits": 0,
        "models": 0,
        "boosters": 0,
    }
    return OracleReadout(rows, targets, fold_metrics, summary)

# %% [markdown]
# ## 8. Target-free exp333-compatible 136-feature surface

# %%
def resolve_train_dir() -> Path:
    local = project_root() / "data" / "raw" / "train"
    if local.is_dir() and next(local.glob("*__horizontal_well.csv"), None) is not None:
        return local
    preferred = (
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
    )
    for path in preferred:
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
            return path
    matches = sorted(
        path
        for path in KAGGLE_INPUT_ROOT.rglob("train")
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"raw competition train directory was not unique: {matches}")
    return matches[0]


def load_feature_source() -> Any:
    try:
        from inputs.exp228_source import direct_residual_correction_on_exp226

        return direct_residual_correction_on_exp226
    except ModuleNotFoundError:
        local = (
            project_root()
            / "experiments"
            / "exp228_direct_residual_correction_on_exp226"
            / "direct_residual_correction_on_exp226.py"
        )
        if not local.is_file():
            raise RuntimeError(
                "exp228 target-free feature source was not bootstrapped"
            ) from None
        spec = importlib.util.spec_from_file_location("exp418_exp228_source", local)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                "could not load the frozen exp228 feature source"
            ) from None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_target_free_exp072_frame(
    path: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    header = [str(column) for column in pd.read_csv(path, nrows=0).columns]
    if not {"id", "well", "target"}.issubset(header):
        raise ValueError("exp072 cache schema is incomplete")
    usecols = [column for column in header if column != "target"]
    frame = pd.read_csv(path, usecols=usecols, dtype={"id": str, "well": str})
    if "target" in frame.columns:
        raise ValueError("exp072 target was loaded before feature freeze")
    feature_columns = [column for column in frame.columns if column not in {"id", "well"}]
    spec = get_nested(config, "data.exp072_feature_cache")
    if len(feature_columns) != int(spec["expected_feature_count"]):
        raise ValueError("exp072 target-free feature count changed")
    schema_path = resolve_existing(str(spec["schema_filename"]), spec["schema_patterns"])
    schema = pd.read_csv(schema_path).sort_values("feature_index", kind="mergesort")
    if schema["feature"].astype(str).tolist() != feature_columns:
        raise ValueError("exp072 feature columns differ from the frozen schema")
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame.duplicated(["id", "well"]).any():
        raise ValueError("exp072 target-free cache keys are not unique")
    return frame.reset_index(drop=True), feature_columns, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "decompressed_sha256": sha256_gzip_decompressed(path),
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "features": len(feature_columns),
        "schema_path": str(schema_path),
        "schema_file_sha256": sha256_file(schema_path),
        "target_columns_loaded": 0,
    }


def add_target_free_anchor_columns(
    frame: pd.DataFrame, train_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for well_id in sorted(frame["well"].astype(str).unique()):
        path = train_dir / f"{well_id}__horizontal_well.csv"
        raw = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        known = raw.loc[pd.to_numeric(raw["TVT_input"], errors="coerce").notna()]
        if known.empty:
            raise ValueError(f"no known TVT_input prefix for {well_id}")
        anchor = known.iloc[-1]
        records.append(
            {
                "well": well_id,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "known_prefix_rows": len(known),
            }
        )
    anchors = pd.DataFrame(records).set_index("well")
    result = frame.copy()
    for column in ("anchor_md", "anchor_z0", "anchor_t0"):
        result[column] = result["well"].map(anchors[column]).astype(np.float32)
    result["known_prefix_rows"] = result["well"].map(
        anchors["known_prefix_rows"]
    ).astype(np.int32)
    if result[["anchor_md", "anchor_z0", "anchor_t0"]].isna().any().any():
        raise ValueError("target-free anchor merge produced missing values")
    delta = (
        result["last_known_tvt"].to_numpy(np.float32)
        - result["anchor_t0"].to_numpy(np.float32)
    )
    max_abs = float(np.max(np.abs(delta)))
    if max_abs > 0.05:
        raise ValueError(f"target-free anchor parity failed: {max_abs}")
    return result, {
        "anchor_wells": len(anchors),
        "anchor_t0_vs_last_known_abs_max": max_abs,
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
        "raw_columns_loaded": ["MD", "Z", "TVT_input"],
        "target_columns_loaded": 0,
    }


def _row_index_from_id(ids: pd.Series) -> np.ndarray:
    values = pd.to_numeric(
        ids.astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
    ).to_numpy(np.float64)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("exp072 id suffix is not an integer row index")
    return values.astype(np.int64)


@dataclass(frozen=True)
class FeatureSurface:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    schema: pd.DataFrame
    metadata: dict[str, Any]
    projection_summary: pd.DataFrame
    grwr_summary: pd.DataFrame


def build_feature_surface(
    config: Mapping[str, Any],
    train_dir: Path,
    feature_source: Any,
    frozen_schema: pd.DataFrame,
) -> FeatureSurface:
    cache_spec = get_nested(config, "data.exp072_feature_cache")
    cache_path = resolve_existing(str(cache_spec["filename"]), cache_spec["patterns"])
    base, _base_columns, cache_meta = load_target_free_exp072_frame(
        cache_path, config
    )
    base, anchor_meta = add_target_free_anchor_columns(base, train_dir)
    u_config = dict(get_nested(config, "features.u_projection"))
    projection, projection_groups, projection_summary = (
        feature_source.build_u_projection_features(
            base,
            source_specs=dict(u_config["sources"]),
            degree=int(u_config["degree"]),
            robust_iters=int(u_config["robust_iters"]),
            clip_sigma=float(u_config["clip_sigma"]),
        )
    )
    if not base[["id", "well"]].equals(projection[["id", "well"]]):
        raise ValueError("projection feature row identity changed")
    grwr, grwr_groups, grwr_summary, grwr_meta = (
        feature_source.build_gr_wavelet_rotation_confidence_features(
            base,
            train_dir,
            dict(get_nested(config, "features.gr_wavelet_rotation_confidence")),
        )
    )
    if not base[["id", "well"]].equals(grwr[["id", "well"]]):
        raise ValueError("GRWR feature row identity changed")
    group_columns = {
        "projection_correction": list(projection_groups["projection_correction"]),
        "u_disagreement": list(projection_groups["u_disagreement"]),
        "gr_wavelet_rotation_confidence": list(
            grwr_groups["gr_wavelet_rotation_confidence"]
        ),
    }
    selected: list[str] = []
    column_group: dict[str, str] = {}
    for group in ALLOWED_FEATURE_GROUPS:
        for column in group_columns[group]:
            if column not in selected:
                selected.append(column)
                column_group[column] = group
    forbidden = sorted(
        column
        for column in selected
        if column.lower() in {"tvt_true", "target", "error", "residual", "well_id"}
        or column.startswith("ll_")
        or "selector" in column.lower()
        or "oracle" in column.lower()
    )
    if forbidden:
        raise ValueError(f"forbidden exp418 model features: {forbidden}")
    surface = pd.DataFrame(
        {
            "well_id": base["well"].astype(str),
            "row_idx": _row_index_from_id(base["id"]),
            "md_since": pd.to_numeric(base["md_since"], errors="coerce").astype(
                np.float64
            ),
        }
    )
    projection_selected = [
        column
        for column in selected
        if column_group[column] in {"projection_correction", "u_disagreement"}
    ]
    for column in projection_selected:
        surface[column] = projection[column].to_numpy(np.float32, copy=False)
    for column in group_columns["gr_wavelet_rotation_confidence"]:
        surface[column] = grwr[column].to_numpy(np.float32, copy=False)
    if surface.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("feature surface row keys are not unique")
    schema = pd.DataFrame(
        [
            {
                "feature_name": column,
                "source_group": column_group[column],
                "row_to_segment_aggregation": "finite_float64_mean",
                "all_nonfinite_policy": "preserve_nan",
            }
            for column in selected
        ]
        + [
            {
                "feature_name": column,
                "source_group": "structural",
                "row_to_segment_aggregation": "fixed_definition",
                "all_nonfinite_policy": "not_applicable",
            }
            for column in STRUCTURAL_FEATURE_COLUMNS
        ]
    )
    pd.testing.assert_frame_equal(
        schema.reset_index(drop=True),
        frozen_schema.reset_index(drop=True),
        check_dtype=False,
    )
    metadata = {
        "cache": cache_meta,
        "anchor": anchor_meta,
        "grwr": grwr_meta,
        "allowed_groups": list(ALLOWED_FEATURE_GROUPS),
        "row_feature_count": len(selected),
        "model_feature_count": len(selected) + len(STRUCTURAL_FEATURE_COLUMNS),
        "row_feature_schema_sha256": hashed_frame_sha256(
            schema, tuple(schema.columns)
        ),
        "row_feature_content_sha256": hashed_frame_sha256(
            surface, ("well_id", "row_idx", "md_since", *selected)
        ),
        "target_or_error_columns_loaded_before_freeze": 0,
    }
    expected_row_sha = str(
        get_nested(config, "data.exp333_stage1.expected_row_feature_content_sha256")
    )
    if metadata["row_feature_content_sha256"] != expected_row_sha:
        raise ValueError("reconstructed exp333 row-feature content SHA mismatch")
    return FeatureSurface(
        frame=surface.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
            drop=True
        ),
        feature_columns=tuple(selected),
        schema=schema,
        metadata=metadata,
        projection_summary=projection_summary,
        grwr_summary=grwr_summary,
    )


def validate_exp333_feature_freeze(
    frozen: FrozenExp333Inputs,
    surface: FeatureSurface,
    exp226_evidence: Mapping[str, Any],
    parity_max_abs_ft: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    feature_freeze = {
        "saved_exp226_decompressed_sha256": exp226_evidence["decompressed_sha256"],
        "fold_manifest_sha256": hashed_frame_sha256(
            frozen.fold_manifest,
            ("outer_fold", "well_id", "inner_fold", "inner_digest"),
        ),
        "segment_assignment_sha256": hashed_frame_sha256(
            frozen.nested,
            (
                "outer_fold",
                "role",
                "inner_fold",
                "well_id",
                "row_idx",
                "segment_id",
            ),
        ),
        "nested_exp226_prediction_sha256": hashed_frame_sha256(
            frozen.nested,
            (
                "outer_fold",
                "role",
                "inner_fold",
                "well_id",
                "row_idx",
                "tvt_pred",
            ),
        ),
        **surface.metadata,
        "outer_valid_parent_parity_max_abs_ft": parity_max_abs_ft,
        "residual_target_attached": False,
    }
    feature_freeze["feature_freeze_sha256"] = mapping_sha256(feature_freeze)
    expected = str(
        get_nested(config, "data.exp333_stage1.expected_feature_freeze_sha256")
    )
    if feature_freeze["feature_freeze_sha256"] != expected:
        raise ValueError("reconstructed exp333 feature-freeze SHA mismatch")
    return feature_freeze

# %% [markdown]
# ## 9. Strict-nested segment-rate samples and LightGBM training

# %%
def finite_group_mean(
    values: np.ndarray, group_index: np.ndarray, n_groups: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    sums = np.bincount(group_index[finite], weights=array[finite], minlength=n_groups)
    counts = np.bincount(group_index[finite], minlength=n_groups)
    result = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def aggregate_rate_segments(
    nested_rows: pd.DataFrame,
    row_surface: pd.DataFrame,
    truth: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    truth_free_contract_sha256: str,
) -> pd.DataFrame:
    context = nested_rows.merge(
        row_surface,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    context = attach_truth_after_freeze(
        context,
        truth,
        truth_free_contract_sha256=truth_free_contract_sha256,
    )
    if context[["md_since", "tvt_true"]].isna().any().any():
        raise ValueError("rate-segment context is missing required rows")
    context = context.sort_values(
        ["well_id", "suffix_offset"], kind="mergesort"
    ).reset_index(drop=True)
    keys = ["well_id", "segment_id"]
    group_index = context.groupby(keys, sort=True, observed=True).ngroup().to_numpy(
        np.int64
    )
    group_meta = (
        context.groupby(keys, sort=True, observed=True)
        .agg(
            outer_fold=("outer_fold", "first"),
            role=("role", "first"),
            inner_fold=("inner_fold", "first"),
            segment_row_count=("row_idx", "size"),
            segment_md_min=("md_since", "min"),
            segment_md_max=("md_since", "max"),
            exp226_pred_start=("tvt_pred", "first"),
            exp226_pred_end=("tvt_pred", "last"),
        )
        .reset_index()
    )
    n_groups = len(group_meta)
    segment = group_meta.drop(
        columns=["segment_md_min", "segment_md_max", "exp226_pred_end"]
    )
    segment["segment_position"] = (
        segment["segment_id"].to_numpy(np.float64) + 0.5
    ) / K_SEGMENTS
    segment["segment_md_span"] = (
        group_meta["segment_md_max"].to_numpy(np.float64)
        - group_meta["segment_md_min"].to_numpy(np.float64)
    )
    segment["exp226_pred_mean"] = finite_group_mean(
        context["tvt_pred"].to_numpy(), group_index, n_groups
    )
    segment["exp226_pred_end_minus_start"] = (
        group_meta["exp226_pred_end"].to_numpy(np.float64)
        - group_meta["exp226_pred_start"].to_numpy(np.float64)
    )
    for column in feature_columns:
        segment[column] = finite_group_mean(
            context[column].to_numpy(), group_index, n_groups
        )
    target_by_key: dict[tuple[str, int], float] = {}
    rank_by_well: dict[str, int] = {}
    for well_id, part in context.groupby("well_id", sort=True):
        residual = (
            part["tvt_true"].to_numpy(np.float64)
            - part["tvt_pred"].to_numpy(np.float64)
        )
        solution = solve_zero_intercept_rates(
            residual, part["segment_id"].to_numpy(np.int16)
        )
        rank_by_well[str(well_id)] = solution.rank
        for segment_id, rate in enumerate(solution.rates):
            target_by_key[(str(well_id), segment_id)] = float(rate)
    segment["segment_rate_target"] = [
        target_by_key[(str(well_id), int(segment_id))]
        for well_id, segment_id in zip(
            segment["well_id"], segment["segment_id"], strict=True
        )
    ]
    segment["basis_rank"] = segment["well_id"].map(rank_by_well).astype(np.int16)
    ordered = [
        "outer_fold",
        "role",
        "inner_fold",
        "well_id",
        "segment_id",
        *STRUCTURAL_FEATURE_COLUMNS[1:],
        *feature_columns,
        "segment_rate_target",
        "basis_rank",
    ]
    return segment.loc[:, ordered].sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)


def fit_rate_fold(
    train_segments: pd.DataFrame,
    valid_segments: pd.DataFrame,
    model_features: Sequence[str],
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    model_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(**dict(get_nested(config, "model.params")))
    train_weight = train_segments["segment_row_count"].to_numpy(np.float64)
    valid_weight = valid_segments["segment_row_count"].to_numpy(np.float64)
    model.fit(
        train_segments.loc[:, list(model_features)],
        train_segments["segment_rate_target"].to_numpy(np.float64),
        sample_weight=train_weight,
        eval_set=[
            (
                valid_segments.loc[:, list(model_features)],
                valid_segments["segment_rate_target"].to_numpy(np.float64),
            )
        ],
        eval_sample_weight=[valid_weight],
        eval_metric="rmse",
        callbacks=[
            early_stopping(
                int(get_nested(config, "model.early_stopping_rounds")), verbose=True
            ),
            log_evaluation(100),
        ],
    )
    prediction = model.predict(
        valid_segments.loc[:, list(model_features)],
        num_iteration=model.best_iteration_,
    ).astype(np.float64)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"outer_fold_{outer_fold}.txt"
    model.booster_.save_model(str(model_path))
    importance = pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "feature": list(model_features),
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    )
    record = {
        "outer_fold": outer_fold,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "best_iteration": int(model.best_iteration_),
        "train_segments": len(train_segments),
        "valid_segments": len(valid_segments),
        "feature_count": len(model_features),
        "params": dict(get_nested(config, "model.params")),
    }
    return prediction, importance, record

# %% [markdown]
# ## 10. Continuous row integration, metrics, tails, and promotion gate

# %%
def integrate_valid_rows(
    valid_rows: pd.DataFrame,
    valid_segments: pd.DataFrame,
    predicted_rates: np.ndarray,
    row_surface: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_free_contract_sha256: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rate_lookup = valid_segments[["well_id", "segment_id"]].copy()
    rate_lookup["segment_rate_target"] = valid_segments[
        "segment_rate_target"
    ].to_numpy(np.float64)
    rate_lookup["segment_rate_pred"] = np.asarray(predicted_rates, dtype=np.float64)
    rows = valid_rows.merge(
        row_surface[["well_id", "row_idx", "md_since"]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    rows = attach_truth_after_freeze(
        rows,
        truth,
        truth_free_contract_sha256=truth_free_contract_sha256,
    )
    correction = np.empty(len(rows), dtype=np.float64)
    target_correction = np.empty(len(rows), dtype=np.float64)
    integration_diffs: list[float] = []
    first_corrections: list[float] = []
    for well_id, positions in rows.groupby("well_id", sort=True).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        part = rows.iloc[index].sort_values("suffix_offset", kind="mergesort")
        lookup = rate_lookup.loc[rate_lookup["well_id"].astype(str).eq(str(well_id))]
        lookup = lookup.sort_values("segment_id", kind="mergesort")
        if lookup["segment_id"].tolist() != list(range(K_SEGMENTS)):
            raise ValueError(f"predicted rate coverage changed for {well_id}")
        predicted = lookup["segment_rate_pred"].to_numpy(np.float64)
        target = lookup["segment_rate_target"].to_numpy(np.float64)
        predicted_correction, diff = integrate_predicted_rates(
            part["segment_id"].to_numpy(np.int16), predicted
        )
        target_integrated, _ = integrate_predicted_rates(
            part["segment_id"].to_numpy(np.int16), target
        )
        ordered_index = part.index.to_numpy(np.int64)
        correction[ordered_index] = predicted_correction
        target_correction[ordered_index] = target_integrated
        integration_diffs.append(diff)
        first_corrections.append(float(predicted_correction[0]))
    rows["rate_correction_pred"] = correction
    rows["rate_correction_target"] = target_correction
    rows["tvt_pred_stage1"] = rows["tvt_pred"].to_numpy(np.float64) + correction
    rows["segment_local_offset"] = rows.groupby(
        ["well_id", "segment_id"], sort=False, observed=True
    ).cumcount()
    rows["segment_row_count"] = rows.groupby(
        ["well_id", "segment_id"], sort=False, observed=True
    )["row_idx"].transform("size")
    edge_distance = np.minimum(
        rows["segment_local_offset"].to_numpy(np.int64),
        rows["segment_row_count"].to_numpy(np.int64)
        - 1
        - rows["segment_local_offset"].to_numpy(np.int64),
    )
    rows["boundary_band_pm8"] = edge_distance < 8
    diagnostics = {
        "first_row_correction_abs_max_ft": float(
            np.max(np.abs(first_corrections))
        ),
        "matrix_vs_sequential_integration_abs_max_ft": max(integration_diffs),
    }
    return (
        rows.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True),
        diagnostics,
    )


def balanced_sign_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.sign(np.asarray(target, dtype=np.float64)).astype(np.int8)
    predicted = np.sign(np.asarray(prediction, dtype=np.float64)).astype(np.int8)
    recalls = [
        float(np.mean(predicted[actual == label] == label))
        for label in (-1, 0, 1)
        if np.any(actual == label)
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def metric_comparison(
    frame: pd.DataFrame, mask: np.ndarray | pd.Series, *, label: str
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return {
            "scope": label,
            "rows": 0,
            "wells": 0,
            "exp226_rmse": float("nan"),
            "stage1_rmse": float("nan"),
            "delta_stage1_minus_exp226": float("nan"),
        }
    base_score = rmse(
        frame.loc[selected, "tvt_true"].to_numpy(),
        frame.loc[selected, "tvt_pred"].to_numpy(),
    )
    stage1_score = rmse(
        frame.loc[selected, "tvt_true"].to_numpy(),
        frame.loc[selected, "tvt_pred_stage1"].to_numpy(),
    )
    return {
        "scope": label,
        "rows": int(selected.sum()),
        "wells": int(frame.loc[selected, "well_id"].nunique()),
        "exp226_rmse": base_score,
        "stage1_rmse": stage1_score,
        "delta_stage1_minus_exp226": stage1_score - base_score,
    }


def resolve_hidden_assignment(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like")
    path = resolve_existing(str(spec["filename"]), spec["patterns"])
    file_sha = sha256_file(path)
    if file_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    if required - set(frame.columns):
        raise ValueError("hidden-like assignment columns are incomplete")
    return frame, {"path": str(path), "file_sha256": file_sha, "rows": len(frame)}


def evaluate_stage1_outputs(
    row_oof: pd.DataFrame,
    segment_oof: pd.DataFrame,
    hidden_assignment: pd.DataFrame,
    integration_diagnostics: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pooled = metric_comparison(
        row_oof, np.ones(len(row_oof), dtype=bool), label="pooled"
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(int(get_nested(config, "validation.n_outer_folds"))):
        row = metric_comparison(
            row_oof, row_oof["outer_fold"].eq(fold), label=f"fold_{fold}"
        )
        valid = segment_oof.loc[segment_oof["outer_fold"].eq(fold)]
        weight = valid["segment_row_count"].to_numpy(np.float64)
        target = valid["segment_rate_target"].to_numpy(np.float64)
        prediction = valid["segment_rate_pred"].to_numpy(np.float64)
        zero_rmse = float(np.sqrt(np.average(np.square(target), weights=weight)))
        model_rmse = float(
            np.sqrt(np.average(np.square(target - prediction), weights=weight))
        )
        row.update(
            {
                "outer_fold": fold,
                "rate_target_zero_prior_weighted_rmse": zero_rmse,
                "rate_target_model_weighted_rmse": model_rmse,
                "rate_target_improved": bool(model_rmse < zero_rmse),
                "rate_sign_balanced_accuracy": balanced_sign_accuracy(
                    target, prediction
                ),
            }
        )
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = row_oof["md_since"].to_numpy(np.float64)
    bucket_masks = {
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    bucket_metrics = pd.DataFrame(
        metric_comparison(row_oof, mask, label=name)
        for name, mask in bucket_masks.items()
    )
    hidden_lookup = hidden_assignment.set_index("well_id")
    hidden_metrics = pd.DataFrame(
        metric_comparison(
            row_oof,
            row_oof["well_id"].map(hidden_lookup[column]).eq("valid"),
            label=column,
        )
        for column in (
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
        )
    )
    boundary_metrics = pd.DataFrame(
        [
            metric_comparison(
                row_oof,
                row_oof["boundary_band_pm8"],
                label="segment_boundary_pm8_rows",
            )
        ]
    )
    by_well_records: list[dict[str, Any]] = []
    for well_id, part in row_oof.groupby("well_id", sort=True):
        base_score = rmse(part["tvt_true"].to_numpy(), part["tvt_pred"].to_numpy())
        stage1_score = rmse(
            part["tvt_true"].to_numpy(), part["tvt_pred_stage1"].to_numpy()
        )
        by_well_records.append(
            {
                "well_id": str(well_id),
                "rows": len(part),
                "exp226_rmse": base_score,
                "stage1_rmse": stage1_score,
                "delta_stage1_minus_exp226": stage1_score - base_score,
            }
        )
    by_well = pd.DataFrame(by_well_records)
    base_p95 = float(by_well["exp226_rmse"].quantile(0.95))
    stage1_p95 = float(by_well["stage1_rmse"].quantile(0.95))
    worst_delta = float(by_well["delta_stage1_minus_exp226"].max())
    gates = get_nested(config, "stage_1_gate")
    bucket_lookup = bucket_metrics.set_index("scope")
    hidden_lookup_metrics = hidden_metrics.set_index("scope")
    sign_improved_folds = int(
        (
            fold_metrics["rate_sign_balanced_accuracy"]
            > float(gates["minimum_rate_sign_balanced_accuracy"])
        ).sum()
    )
    checks = {
        "pooled_rmse": pooled["stage1_rmse"]
        <= float(gates["maximum_pooled_rmse"]),
        "gain_vs_exp228": float(get_nested(config, "data.exp228_reference.expected_rmse"))
        - pooled["stage1_rmse"]
        >= float(gates["minimum_gain_vs_exp228_ft"]),
        "gain_vs_exp333": float(
            get_nested(config, "data.exp333_reference.expected_rmse")
        )
        - pooled["stage1_rmse"]
        >= float(gates["minimum_gain_vs_exp333_ft"]),
        "improved_outer_folds": int(
            (fold_metrics["delta_stage1_minus_exp226"] < 0.0).sum()
        )
        >= int(gates["minimum_improved_folds_vs_exp226"]),
        "near_0_250_nonworse": float(
            bucket_lookup.loc["near_0_250", "delta_stage1_minus_exp226"]
        )
        <= float(gates["maximum_near_0_250_delta_vs_exp226_ft"]),
        "1000_plus_nonworse": float(
            bucket_lookup.loc["1000_plus", "delta_stage1_minus_exp226"]
        )
        <= float(gates["maximum_1000_plus_delta_vs_exp226_ft"]),
        "hidden_spatial_nonworse": float(
            hidden_lookup_metrics.loc[
                "verification_like_spatial_role", "delta_stage1_minus_exp226"
            ]
        )
        <= float(gates["maximum_hidden_spatial_delta_vs_exp226_ft"]),
        "hidden_typewell_nonworse": float(
            hidden_lookup_metrics.loc[
                "verification_like_typewell_purged_role",
                "delta_stage1_minus_exp226",
            ]
        )
        <= float(gates["maximum_hidden_typewell_purged_delta_vs_exp226_ft"]),
        "boundary_nonworse": float(
            boundary_metrics.loc[0, "delta_stage1_minus_exp226"]
        )
        <= float(gates["maximum_boundary_pm8_delta_vs_exp226_ft"]),
        "by_well_p95_nonworse": stage1_p95 - base_p95
        <= float(gates["maximum_by_well_p95_delta_vs_exp226_ft"]),
        "worst_well_delta": worst_delta
        <= float(gates["maximum_worst_well_delta_vs_exp226_ft"]),
        "rate_target_all_folds": int(fold_metrics["rate_target_improved"].sum())
        == int(gates["required_rate_target_rmse_improved_folds_vs_zero"]),
        "rate_sign_balanced_accuracy": sign_improved_folds
        >= int(gates["required_rate_sign_balanced_accuracy_folds"]),
        "first_row_anchor": float(
            integration_diagnostics["first_row_correction_abs_max_ft"]
        )
        <= float(gates["maximum_first_row_correction_abs_ft"]),
        "integration_parity": float(
            integration_diagnostics[
                "matrix_vs_sequential_integration_abs_max_ft"
            ]
        )
        <= float(gates["maximum_matrix_vs_sequential_integration_abs_ft"]),
    }
    scientific_pass = all(bool(value) for value in checks.values())
    return {
        "pooled": pooled,
        "fold_metrics": fold_metrics,
        "bucket_metrics": bucket_metrics,
        "hidden_metrics": hidden_metrics,
        "boundary_metrics": boundary_metrics,
        "by_well": by_well,
        "by_well_exp226_p95": base_p95,
        "by_well_stage1_p95": stage1_p95,
        "by_well_p95_delta": stage1_p95 - base_p95,
        "worst_well_delta": worst_delta,
        "rate_sign_balanced_accuracy_passed_folds": sign_improved_folds,
        "integration_diagnostics": dict(integration_diagnostics),
        "gate_checks": checks,
        "scientific_pass": scientific_pass,
        "decision": "PASS_STAGE1" if scientific_pass else "FAIL_CLOSE_BRANCH",
    }


def save_feature_importance_plot(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    mean_gain = (
        importance.groupby("feature", sort=False)["gain"]
        .mean()
        .sort_values(ascending=False)
        .head(30)
        .sort_values()
    )
    figure, axis = plt.subplots(figsize=(10, 9))
    mean_gain.plot.barh(ax=axis)
    axis.set_title("exp418 mean LightGBM gain importance (top 30)")
    axis.set_xlabel("mean gain across five outer folds")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.show()
    plt.close(figure)

# %% [markdown]
# ## 11. Generated artifacts and guarded orchestration

# %%
def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_implementation_contract(config, execution_stage="stage_0")
    artifacts = output_artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_exp333_inputs(config)
    exp226_path = resolve_exp226_oof(config)
    saved_exp226, exp226_evidence = load_exp226_target_free(exp226_path, config)
    freeze = build_truth_free_freeze(frozen, saved_exp226, exp226_evidence, config)
    outer_valid = frozen.nested.loc[frozen.nested["role"].eq("outer_valid")].copy()
    truth = load_exp226_truth(
        exp226_path,
        truth_free_contract_sha256=freeze["truth_free_contract_sha256"],
    )
    readout = build_stage0_oracle_readout(
        outer_valid,
        truth,
        config,
        truth_free_contract_sha256=freeze["truth_free_contract_sha256"],
    )
    paths = {
        "contract": artifacts / f"{OUTPUT_PREFIX}_stage0_truth_free_freeze.json",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_stage0_input_manifest.json",
        "rate_targets": artifacts / f"{OUTPUT_PREFIX}_stage0_rate_targets.csv",
        "fold_metrics": artifacts / f"{OUTPUT_PREFIX}_stage0_fold_metrics.csv",
        "summary": artifacts / f"{OUTPUT_PREFIX}_stage0_summary.json",
    }
    write_json(paths["contract"], freeze)
    write_json(
        paths["input_manifest"],
        {"exp333": frozen.evidence, "exp226": exp226_evidence},
    )
    write_csv(paths["rate_targets"], readout.rate_targets)
    write_csv(paths["fold_metrics"], readout.fold_metrics)
    write_json(paths["summary"], readout.summary)
    sha_manifest = pd.DataFrame(artifact_evidence(path) for path in paths.values())
    write_csv(artifacts / f"{OUTPUT_PREFIX}_stage0_sha_manifest.csv", sha_manifest)
    metrics_path = (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "metrics.stage0.runtime.json"
    )
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "stage0_completed",
            "route": "ensemble",
            "stage0": readout.summary,
            "cv": readout.summary["rate_oracle_rmse"],
            "public_lb": None,
            "private_lb": None,
        },
    )
    print(json.dumps(to_jsonable(readout.summary), indent=2, sort_keys=True))
    return readout.summary


def load_stage0_pass_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = get_nested(config, "data.exp418_stage0")
    expected_sha = spec.get("expected_summary_file_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise RuntimeError(
            "Stage 1 requires the completed Stage 0 summary SHA to be frozen in config"
        )
    path = resolve_existing(str(spec["filename"]), spec["patterns"])
    if sha256_file(path) != expected_sha:
        raise ValueError("frozen exp418 Stage 0 summary file SHA mismatch")
    summary = json.loads(path.read_text())
    if summary.get("decision") != str(spec["expected_decision"]):
        raise RuntimeError("Stage 0 did not pass; Stage 1 remains fail-closed")
    if not bool(summary.get("technical_pass")) or not bool(
        summary.get("scientific_pass")
    ):
        raise RuntimeError("Stage 0 pass evidence is incomplete")
    return {
        "path": str(path),
        "file_sha256": expected_sha,
        "decision": summary["decision"],
        "rate_target_sha256": summary.get("rate_target_sha256"),
    }


def run_stage1(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_implementation_contract(config, execution_stage="stage_1")
    stage0_evidence = load_stage0_pass_evidence(config)
    artifacts = output_artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_exp333_inputs(config)
    exp226_path = resolve_exp226_oof(config)
    saved_exp226, exp226_evidence = load_exp226_target_free(exp226_path, config)
    freeze = build_truth_free_freeze(frozen, saved_exp226, exp226_evidence, config)
    train_dir = resolve_train_dir()
    surface = build_feature_surface(
        config, train_dir, load_feature_source(), frozen.feature_schema
    )
    parity = outer_valid_parent_parity(frozen.nested, saved_exp226)
    feature_freeze = validate_exp333_feature_freeze(
        frozen, surface, exp226_evidence, parity, config
    )
    freeze["reconstructed_exp333_feature_freeze_sha256"] = feature_freeze[
        "feature_freeze_sha256"
    ]
    freeze["row_feature_content_sha256"] = surface.metadata[
        "row_feature_content_sha256"
    ]
    freeze["truth_free_contract_sha256"] = mapping_sha256(
        {key: value for key, value in freeze.items() if key != "truth_free_contract_sha256"}
    )
    truth = load_exp226_truth(
        exp226_path,
        truth_free_contract_sha256=freeze["truth_free_contract_sha256"],
    )
    model_features = (*STRUCTURAL_FEATURE_COLUMNS, *surface.feature_columns)
    if len(model_features) != int(get_nested(config, "features.expected_feature_count")):
        raise ValueError("exp418 model feature count changed")

    row_oof_frames: list[pd.DataFrame] = []
    segment_oof_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    integration_records: list[dict[str, float]] = []
    model_dir = artifacts / f"{OUTPUT_PREFIX}_stage1_lgb_models"
    for outer_fold in range(int(get_nested(config, "validation.n_outer_folds"))):
        train_rows = frozen.nested.loc[
            frozen.nested["outer_fold"].eq(outer_fold)
            & frozen.nested["role"].eq("inner_oof_train")
        ].copy()
        valid_rows = frozen.nested.loc[
            frozen.nested["outer_fold"].eq(outer_fold)
            & frozen.nested["role"].eq("outer_valid")
        ].copy()
        train_segments = aggregate_rate_segments(
            train_rows,
            surface.frame,
            truth,
            surface.feature_columns,
            truth_free_contract_sha256=freeze["truth_free_contract_sha256"],
        )
        valid_segments = aggregate_rate_segments(
            valid_rows,
            surface.frame,
            truth,
            surface.feature_columns,
            truth_free_contract_sha256=freeze["truth_free_contract_sha256"],
        )
        prediction, importance, model_record = fit_rate_fold(
            train_segments,
            valid_segments,
            model_features,
            config,
            outer_fold=outer_fold,
            model_dir=model_dir,
        )
        valid_output = valid_segments[
            [
                "outer_fold",
                "well_id",
                "segment_id",
                "segment_row_count",
                "segment_rate_target",
                "basis_rank",
            ]
        ].copy()
        valid_output["segment_rate_pred"] = prediction
        integrated, diagnostics = integrate_valid_rows(
            valid_rows,
            valid_segments,
            prediction,
            surface.frame,
            truth,
            truth_free_contract_sha256=freeze["truth_free_contract_sha256"],
        )
        row_oof_frames.append(integrated)
        segment_oof_frames.append(valid_output)
        importance_frames.append(importance)
        model_records.append(model_record)
        integration_records.append(diagnostics)

    row_oof = pd.concat(row_oof_frames, ignore_index=True).sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    segment_oof = pd.concat(segment_oof_frames, ignore_index=True).sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(row_oof) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp418 OOF row coverage changed")
    if len(segment_oof) != int(get_nested(config, "validation.expected_segments")):
        raise ValueError("exp418 segment OOF coverage changed")
    combined_diagnostics = {
        "first_row_correction_abs_max_ft": max(
            item["first_row_correction_abs_max_ft"] for item in integration_records
        ),
        "matrix_vs_sequential_integration_abs_max_ft": max(
            item["matrix_vs_sequential_integration_abs_max_ft"]
            for item in integration_records
        ),
    }
    # Hidden-like roles are intentionally loaded only after every OOF prediction is frozen.
    hidden, hidden_evidence = resolve_hidden_assignment(config)
    evaluation = evaluate_stage1_outputs(
        row_oof, segment_oof, hidden, combined_diagnostics, config
    )
    importance = pd.concat(importance_frames, ignore_index=True)
    model_manifest = {
        "experiment": EXPERIMENT_NAME,
        "variant": "signed_k16_rate",
        "config": "exp333_lgb1_single_fixed",
        "boosters": len(model_records),
        "models": model_records,
        "feature_columns": list(model_features),
        "feature_schema_sha256": surface.metadata["row_feature_schema_sha256"],
        "feature_freeze_sha256": feature_freeze["feature_freeze_sha256"],
        "exp226_fits": 0,
        "parent_control_retraining": False,
    }
    paths = {
        "contract": artifacts / f"{OUTPUT_PREFIX}_stage1_contract.json",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_input_manifest.json",
        "feature_schema": artifacts / f"{OUTPUT_PREFIX}_stage1_feature_schema.csv",
        "projection_summary": artifacts
        / f"{OUTPUT_PREFIX}_stage1_projection_feature_summary.csv",
        "grwr_summary": artifacts / f"{OUTPUT_PREFIX}_stage1_grwr_feature_summary.csv",
        "segment": artifacts / f"{OUTPUT_PREFIX}_stage1_segment_rate_predictions.csv.gz",
        "oof": artifacts / f"{OUTPUT_PREFIX}_stage1_oof_predictions.csv.gz",
        "fold_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_fold_metrics.csv",
        "bucket_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_bucket_metrics.csv",
        "hidden_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_hidden_like_metrics.csv",
        "boundary_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_boundary_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv",
        "importance": artifacts / f"{OUTPUT_PREFIX}_stage1_feature_importance.csv",
        "importance_plot": artifacts
        / f"{OUTPUT_PREFIX}_stage1_feature_importance_top30.png",
        "model_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_model_manifest.json",
        "summary": artifacts / f"{OUTPUT_PREFIX}_stage1_summary.json",
    }
    contract = {
        "stage": "stage_1_strict_nested_signed_k16_rate",
        "route": "ensemble",
        "variants": 1,
        "model_configs": 1,
        "outer_folds": 5,
        "boosters": 5,
        "exp226_fits": 0,
        "parent_control_retraining": False,
        "gpu": False,
        "inference": False,
        "submission": False,
        "truth_free_freeze": freeze,
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    write_json(paths["contract"], contract)
    write_json(
        paths["input_manifest"],
        {
            "exp333": frozen.evidence,
            "exp226": exp226_evidence,
            "exp418_stage0": stage0_evidence,
            "hidden_like": hidden_evidence,
            "feature_surface": surface.metadata,
        },
    )
    write_csv(paths["feature_schema"], surface.schema)
    write_csv(paths["projection_summary"], surface.projection_summary)
    write_csv(paths["grwr_summary"], surface.grwr_summary)
    write_csv_gzip(paths["segment"], segment_oof)
    oof_output = row_oof[
        [
            "well_id",
            "row_idx",
            "suffix_offset",
            "outer_fold",
            "segment_id",
            "md_since",
            "tvt_true",
            "tvt_pred",
            "rate_correction_pred",
            "tvt_pred_stage1",
            "boundary_band_pm8",
        ]
    ].copy()
    write_csv_gzip(paths["oof"], oof_output)
    write_csv(paths["fold_metrics"], evaluation["fold_metrics"])
    write_csv(paths["bucket_metrics"], evaluation["bucket_metrics"])
    write_csv(paths["hidden_metrics"], evaluation["hidden_metrics"])
    write_csv(paths["boundary_metrics"], evaluation["boundary_metrics"])
    write_csv(paths["by_well"], evaluation["by_well"])
    write_csv(paths["importance"], importance)
    save_feature_importance_plot(importance, paths["importance_plot"])
    write_json(paths["model_manifest"], model_manifest)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_1",
        "status": "completed",
        "decision": evaluation["decision"],
        "scientific_pass": evaluation["scientific_pass"],
        "pooled": evaluation["pooled"],
        "gate_checks": evaluation["gate_checks"],
        "integration_diagnostics": evaluation["integration_diagnostics"],
        "by_well_exp226_p95": evaluation["by_well_exp226_p95"],
        "by_well_stage1_p95": evaluation["by_well_stage1_p95"],
        "by_well_p95_delta": evaluation["by_well_p95_delta"],
        "worst_well_delta": evaluation["worst_well_delta"],
        "feature_freeze": feature_freeze,
        "rate_target_sha256": hashed_frame_sha256(
            segment_oof,
            (
                "outer_fold",
                "well_id",
                "segment_id",
                "segment_row_count",
                "segment_rate_target",
            ),
        ),
        "segment_prediction_sha256": hashed_frame_sha256(
            segment_oof,
            ("outer_fold", "well_id", "segment_id", "segment_rate_pred"),
        ),
        "oof_prediction_sha256": hashed_frame_sha256(
            oof_output,
            ("well_id", "row_idx", "outer_fold", "tvt_pred_stage1"),
        ),
        "model_manifest_sha256": sha256_file(paths["model_manifest"]),
        "boosters": len(model_records),
        "exp226_fits": 0,
        "parent_control_retraining": False,
        "inference_approved": False,
        "submission_approved": False,
    }
    write_json(paths["summary"], summary)
    sha_manifest = pd.DataFrame(
        artifact_evidence(path)
        for path in [*paths.values(), *sorted(model_dir.glob("*.txt"))]
    )
    write_csv(artifacts / f"{OUTPUT_PREFIX}_stage1_sha_manifest.csv", sha_manifest)
    metrics_path = (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "metrics.stage1.runtime.json"
    )
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "stage1_completed",
            "route": "ensemble",
            "metric": "rmse",
            "cv": evaluation["pooled"]["stage1_rmse"],
            "stage1": summary,
            "public_lb": None,
            "private_lb": None,
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary

# %% [markdown]
# ## 12. Setup and contract preview

# %%
CONFIG = load_config()
CONTRACT_PREVIEW = validate_implementation_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "selected_stage": get_nested(CONFIG, "execution_contract.selected_stage"),
            "stage0_boosters": get_nested(
                CONFIG, "execution_contract.stage_0.boosters"
            ),
            "stage1_variants": get_nested(
                CONFIG,
                "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.active_variants",
            ),
            "stage1_model_configs": get_nested(
                CONFIG,
                "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.model_configs",
            ),
            "stage1_folds": get_nested(
                CONFIG,
                "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.outer_folds",
            ),
            "stage1_boosters": get_nested(
                CONFIG,
                "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.boosters",
            ),
            "exp226_fits": get_nested(
                CONFIG,
                "execution_contract.stage_1_if_stage_0_pass_and_separately_approved.exp226_fits",
            ),
            "message": (
                "Compact Stage 0/1 implementation is ready; execution, canonical "
                "Notebook adoption, inference, and submission remain fail-closed."
            ),
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    SELECTED_STAGE = get_nested(CONFIG, "execution_contract.selected_stage")
    if SELECTED_STAGE == "stage_0":
        run_stage0(CONFIG)
    elif SELECTED_STAGE == "stage_1":
        run_stage1(CONFIG)
    else:
        print("Implementation-only mode: no Stage 0 or Stage 1 run was started.")
