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
# # exp333 exp226 K16 segment residual offset target — Stage 0 + Stage 1
#
# Stage 0 is the completed zero-model headroom audit. Stage 1 implements the
# separately approved strict-nested exp226 base, the three target-free feature
# groups, exact K16 finite-mean aggregation, and one fixed LightGBM lgb1 model
# per saved outer fold. The expensive exp226 predictor and target-free GRWR
# generator are pinned parent-source dependencies; all split, aggregation,
# training, evaluation, gate, artifact, and SHA orchestration is visible here.
# Kaggle execution, inference, and submission remain fail-closed.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment boundary
# 2. Runtime, configuration, path, SHA, and serialization helpers
# 3. Frozen scientific and execution contract
# 4. Saved exp226 target-free input checks
# 5. Exact exp226 K16 assignment and target-free freeze
# 6. Late truth attachment and diagnostic oracle readout
# 7. Fixed Stage 0 gate and reproducibility evidence
# 8. Full Kaggle CPU orchestration and generated artifacts
# 9. Strict-nested exp226 and target-free Stage 1 feature surface
# 10. K16 segment samples and fixed LightGBM training
# 11. Stage 1 promotion gates, artifacts, and orchestration
# 12. Setup, contract preview, and guarded execution

# %% [markdown]
# ## 1. Imports and immutable Stage 0 boundary

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp333_exp226_k16_segment_residual_offset_target"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FREE_COLUMNS = ("well_id", "row_idx", "suffix_offset", "tvt_pred", "fold")
TRUTH_COLUMNS = ("well_id", "row_idx", "tvt_true")
KEY_COLUMNS = ("well_id", "row_idx")
K_SEGMENTS = 16
ALLOWED_STAGE1_GROUPS = (
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


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP333_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    candidates = (Path.cwd() / "config.yaml", experiment_dir() / "config.yaml")
    for path in candidates:
        if path.exists():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            return value
    raise FileNotFoundError("exp333 config.yaml was not found")


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
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
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


def canonical_frame_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    selected = frame.loc[:, list(columns)].copy()
    return selected.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode()


def frame_content_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return hashlib.sha256(canonical_frame_bytes(frame, columns)).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def artifact_evidence(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }
    if path.suffix == ".gz":
        record["decompressed_sha256"] = sha256_gzip_decompressed(path)
    return record


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    roots = (project_root(), Path.cwd())
    seen: set[Path] = set()
    resolved: list[Path] = []
    for raw in candidates:
        pattern = str(raw)
        has_wildcard = any(token in pattern for token in ("*", "?", "["))
        matches: list[Path] = []
        if has_wildcard and Path(pattern).is_absolute():
            matches.extend(Path(item) for item in glob.glob(pattern, recursive=True))
        elif has_wildcard:
            for root in roots:
                matches.extend(root.glob(pattern))
        else:
            path = Path(pattern)
            matches.append(path if path.is_absolute() else project_root() / path)
    
        for path in matches:
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


# %% [markdown]
# ## 3. Frozen scientific and execution contract

# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_execution_authorization: bool
) -> dict[str, Any]:
    exact = {
        "experiment.route": "ensemble",
        "implementation.enabled": True,
        "implementation.scope": "stage_0_and_stage_1_train",
        "implementation.stage_0_enabled": True,
        "implementation.stage_1_enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "segmentation.k_segments": 16,
        "segmentation.assignment": (
            "numpy_searchsorted_edges_1_to_16_side_left_clip_zero_15"
        ),
        "target.aggregation": "float64_arithmetic_mean",
        "target.sample_weight": "segment_row_count",
        "target.broadcast": "constant_to_all_rows_in_segment",
        "target.clipping": "none",
        "target.shrinkage": "none",
        "target.taper": "none",
        "target.interpolation": "none",
        "target.slope": "disabled",
        "stage_0_headroom.enabled_after_implementation_approval": True,
        "stage_0_headroom.models": 0,
        "stage_0_headroom.boosters": 0,
        "execution_contract.stage_0.variants": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.implementation_approved": True,
        "execution_contract.stage_1_implementation_approved": True,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    changed = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if changed:
        raise ValueError(f"exp333 frozen Stage 0/1 contract changed: {changed}")

    if tuple(get_nested(config, "data.exp226_oof.target_free_columns", ())) != TARGET_FREE_COLUMNS:
        raise ValueError("exp226 target-free allowlist changed")
    if tuple(get_nested(config, "data.exp226_oof.truth_columns", ())) != TRUTH_COLUMNS:
        raise ValueError("exp226 late-truth allowlist changed")
    if tuple(get_nested(config, "features.allowed_groups", ())) != (
        "projection_correction",
        "u_disagreement",
        "gr_wavelet_rotation_confidence",
    ):
        raise ValueError("future Stage 1 feature allowlist changed")
    if bool(get_nested(config, "features.include_lgb_oof_features")):
        raise ValueError("supervised LGB OOF features are forbidden")

    authorization = {
        "selected_stage": get_nested(config, "execution_contract.selected_stage"),
        "kaggle_push_approved": bool(
            get_nested(config, "execution_contract.kaggle_push_approved")
        ),
        "stage_0_run_approved": bool(
            get_nested(config, "execution_contract.stage_0_run_approved")
        ),
    }
    if require_execution_authorization and authorization != {
        "selected_stage": "stage_0",
        "kaggle_push_approved": True,
        "stage_0_run_approved": True,
    }:
        raise RuntimeError(
            "Stage 0 execution is not authorized; keep this notebook implementation-only"
        )
    return {"fixed_values": exact, "execution_authorization": authorization}


def build_stage0_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0_saved_exp226_k16_oracle_headroom",
        "route": "ensemble",
        "input": "saved_exp226_group_safe_oof",
        "segmentation": {
            "k_segments": K_SEGMENTS,
            "edges": "numpy.linspace(0, suffix_length, 17)",
            "row_coordinate": "one_based_unknown_suffix_row",
            "assignment": "searchsorted(edges[1:], t, side=left), clipped 0..15",
        },
        "oracle": {
            "target": "float64 mean(tvt_true - exp226_tvt_pred) per well K16 segment",
            "broadcast": "constant within segment",
            "offset_persisted": False,
            "prediction_persisted": False,
            "target_persisted": False,
        },
        "execution": {
            "models": 0,
            "boosters": 0,
            "trained_folds": 0,
            "parent_or_control_retraining": False,
            "stage_1_code_present": False,
        },
        "gates": get_nested(config, "stage_0_headroom.gates"),
        "fail_action": get_nested(config, "stage_0_headroom.fail_action"),
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Saved exp226 target-free input checks

# %%
def reject_forbidden_pre_freeze_columns(
    columns: Sequence[str], forbidden: Sequence[str]
) -> None:
    lower = {str(column).lower() for column in columns}
    hits = sorted(str(column) for column in forbidden if str(column).lower() in lower)
    if hits:
        raise ValueError(f"forbidden pre-freeze columns requested: {hits}")


def resolve_exp226_oof(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.exp226_oof", {})
    return resolve_existing(str(spec["filename"]), spec.get("patterns", ()))


def validate_target_free_rows(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_expected_counts: bool,
) -> pd.DataFrame:
    if tuple(frame.columns) != TARGET_FREE_COLUMNS:
        raise ValueError(f"target-free columns must be exactly {TARGET_FREE_COLUMNS}")
    clean = frame.copy()
    clean["well_id"] = clean["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        numeric = pd.to_numeric(clean[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain finite integers")
        clean[column] = numeric.astype(np.int64)
    clean["tvt_pred"] = pd.to_numeric(clean["tvt_pred"], errors="raise").astype(
        np.float64
    )
    if not np.isfinite(clean["tvt_pred"].to_numpy()).all():
        raise ValueError("saved exp226 prediction contains non-finite values")
    if clean.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("saved exp226 row keys are not unique")
    clean = clean.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    expected_offset = clean.groupby("well_id", sort=False).cumcount().to_numpy(dtype=np.int64)
    if not np.array_equal(clean["suffix_offset"].to_numpy(), expected_offset):
        raise ValueError("exp226 suffix_offset is not contiguous from zero per well")
    fold_counts = clean.groupby("well_id", sort=False)["fold"].nunique()
    if not fold_counts.eq(1).all():
        raise ValueError("each well must have exactly one saved exp226 source fold")
    if enforce_expected_counts:
        expected_rows = int(get_nested(config, "validation.expected_rows"))
        expected_wells = int(get_nested(config, "validation.expected_wells"))
        if len(clean) != expected_rows or clean["well_id"].nunique() != expected_wells:
            raise ValueError("saved exp226 row/well coverage does not match the fixed contract")
        expected_folds = set(
            int(value)
            for value in get_nested(config, "validation.technical_guards.required_outer_fold_set")
        )
        if set(clean["fold"].unique()) != expected_folds:
            raise ValueError("saved exp226 fold identity does not match 0..4")
    return clean


def load_exp226_target_free(
    path: Path, config: Mapping[str, Any], *, enforce_expected_counts: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof", {})
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("saved exp226 decompressed SHA does not match the fixed input")
    physical_columns = tuple(str(value) for value in pd.read_csv(path, nrows=0).columns)
    safe_columns = tuple(str(value) for value in spec["target_free_columns"])
    truth_columns = tuple(str(value) for value in spec["truth_columns"])
    missing = sorted(set(safe_columns).union(truth_columns) - set(physical_columns))
    if missing:
        raise ValueError(f"saved exp226 OOF is missing required columns: {missing}")
    reject_forbidden_pre_freeze_columns(
        safe_columns, tuple(str(value) for value in spec["forbidden_pre_freeze_columns"])
    )
    frame = pd.read_csv(path, usecols=list(safe_columns))
    frame = frame.loc[:, list(safe_columns)]
    frame = validate_target_free_rows(
        frame, config, enforce_expected_counts=enforce_expected_counts
    )
    evidence = {
        "name": "saved_exp226_oof_target_free",
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "decompressed_sha256": decompressed_sha,
        "physical_columns": list(physical_columns),
        "loaded_columns": list(safe_columns),
        "truth_columns_loaded_before_freeze": 0,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, evidence


# %% [markdown]
# ## 5. Exact exp226 K16 assignment and target-free freeze

# %%
def exact_k16_segment_ids(length: int, k_segments: int = K_SEGMENTS) -> np.ndarray:
    if length <= 0:
        raise ValueError("unknown suffix length must be positive")
    if k_segments != K_SEGMENTS:
        raise ValueError("exp333 Stage 0 is fixed to K16")
    edges = np.linspace(0.0, float(length), k_segments + 1)
    one_based_step = np.arange(1, length + 1, dtype=np.float64)
    return np.clip(
        np.searchsorted(edges[1:], one_based_step, side="left"),
        0,
        k_segments - 1,
    ).astype(np.int16)


def assign_k16_segments(
    target_free: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_expected_counts: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = target_free.copy()
    segment_id = np.empty(len(assigned), dtype=np.int16)
    for positions in assigned.groupby("well_id", sort=False).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        segment_id[index] = exact_k16_segment_ids(len(index))
    assigned["segment_id"] = segment_id
    counts = (
        assigned.groupby(["well_id", "fold", "segment_id"], sort=True, observed=True)
        .size()
        .rename("segment_row_count")
        .reset_index()
    )
    if counts["segment_row_count"].le(0).any():
        raise ValueError("empty K16 segments must be omitted")
    if int(assigned["segment_id"].min()) != 0 or int(assigned["segment_id"].max()) != 15:
        raise ValueError("K16 segment ids must span 0..15")
    if enforce_expected_counts:
        expected = int(get_nested(config, "validation.expected_nonempty_segments"))
        if len(counts) != expected:
            raise ValueError(f"expected {expected} non-empty K16 segments, found {len(counts)}")
        per_well = counts.groupby("well_id", sort=False).size()
        if not per_well.eq(K_SEGMENTS).all():
            raise ValueError("every exp333 well must contain all 16 K16 segments")
    return assigned, counts


def build_target_free_freeze(
    target_free: pd.DataFrame,
    segment_counts: pd.DataFrame,
    input_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fold_map = target_free[["well_id", "fold"]].drop_duplicates().sort_values("well_id")
    assignment_columns = ("well_id", "row_idx", "suffix_offset", "fold", "segment_id")
    freeze = {
        "stage": "stage0_target_free_freeze_before_truth",
        "input_decompressed_sha256": input_evidence["decompressed_sha256"],
        "target_free_content_sha256": frame_content_sha256(
            target_free, (*TARGET_FREE_COLUMNS, "segment_id")
        ),
        "fold_map_sha256": frame_content_sha256(fold_map, ("well_id", "fold")),
        "segment_assignment_sha256": frame_content_sha256(
            target_free, assignment_columns
        ),
        "segment_count_content_sha256": frame_content_sha256(
            segment_counts,
            ("well_id", "fold", "segment_id", "segment_row_count"),
        ),
        "rows": len(target_free),
        "wells": int(target_free["well_id"].nunique()),
        "segments": len(segment_counts),
        "folds": sorted(int(value) for value in target_free["fold"].unique()),
        "segment_id_min": int(target_free["segment_id"].min()),
        "segment_id_max": int(target_free["segment_id"].max()),
        "truth_columns_loaded_before_freeze": 0,
        "oracle_offsets_persisted": False,
        "oracle_predictions_persisted": False,
        "expected_rows": int(get_nested(config, "validation.expected_rows")),
        "expected_wells": int(get_nested(config, "validation.expected_wells")),
        "expected_segments": int(
            get_nested(config, "validation.expected_nonempty_segments")
        ),
    }
    freeze["target_free_contract_sha256"] = mapping_sha256(freeze)
    return freeze


# %% [markdown]
# ## 6. Late truth attachment and diagnostic oracle readout

# %%
def load_exp226_truth(path: Path, *, target_free_contract_sha256: str) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("late truth requires a frozen target-free contract SHA")
    truth = pd.read_csv(path, usecols=list(TRUTH_COLUMNS)).loc[:, list(TRUTH_COLUMNS)]
    truth["well_id"] = truth["well_id"].astype(str)
    truth["row_idx"] = pd.to_numeric(truth["row_idx"], errors="raise").astype(np.int64)
    truth["tvt_true"] = pd.to_numeric(truth["tvt_true"], errors="raise").astype(
        np.float64
    )
    if truth.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("late exp226 truth row keys are not unique")
    if not np.isfinite(truth["tvt_true"].to_numpy()).all():
        raise ValueError("late exp226 truth contains non-finite values")
    return truth.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def attach_truth_after_freeze(
    target_free: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    target_free_contract_sha256: str,
) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("late truth requires a frozen target-free contract SHA")
    joined = target_free.merge(
        truth,
        on=list(KEY_COLUMNS),
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(joined) != len(target_free) or joined["tvt_true"].isna().any():
        raise ValueError("late truth join failed full row identity coverage")
    return joined


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(truth, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64), dtype=np.float64)))


@dataclass(frozen=True)
class OracleReadout:
    base_prediction: np.ndarray
    oracle_prediction: np.ndarray
    segment_table: pd.DataFrame
    fold_metrics: pd.DataFrame
    segment_counts: pd.DataFrame
    summary: dict[str, Any]


def build_oracle_readout(
    joined: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_expected_counts: bool,
) -> OracleReadout:
    required = {*TARGET_FREE_COLUMNS, "segment_id", "tvt_true"}
    if not required.issubset(joined.columns):
        raise ValueError(f"oracle readout is missing columns: {sorted(required - set(joined))}")
    ordered = joined.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    truth = ordered["tvt_true"].to_numpy(dtype=np.float64)
    base = ordered["tvt_pred"].to_numpy(dtype=np.float64)
    residual = truth - base
    oracle_offset = np.empty(len(ordered), dtype=np.float64)
    segment_records: list[dict[str, Any]] = []
    grouped = ordered.groupby(
        ["well_id", "fold", "segment_id"], sort=True, observed=True
    ).indices
    for (well_id, fold, segment_id), positions in grouped.items():
        index = np.asarray(positions, dtype=np.int64)
        offset = float(np.mean(residual[index], dtype=np.float64))
        oracle_offset[index] = offset
        segment_records.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "segment_id": int(segment_id),
                "segment_row_count": len(index),
                "segment_mean_residual": offset,
            }
        )
    segment_table = pd.DataFrame(segment_records).sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    oracle = base + oracle_offset
    fold_records: list[dict[str, Any]] = []
    minimum_fold_gain = float(
        get_nested(config, "stage_0_headroom.gates.minimum_fold_gain_ft")
    )
    for fold in sorted(int(value) for value in ordered["fold"].unique()):
        mask = ordered["fold"].to_numpy() == fold
        base_fold = rmse(truth[mask], base[mask])
        oracle_fold = rmse(truth[mask], oracle[mask])
        gain = base_fold - oracle_fold
        fold_records.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "wells": int(ordered.loc[mask, "well_id"].nunique()),
                "segments": int(
                    segment_table.loc[segment_table["fold"].eq(fold)].shape[0]
                ),
                "exp226_rmse": base_fold,
                "k16_oracle_offset_rmse": oracle_fold,
                "rmse_gain_ft": gain,
                "minimum_required_gain_ft": minimum_fold_gain,
                "gate_pass": bool(gain >= minimum_fold_gain),
            }
        )
    fold_metrics = pd.DataFrame(fold_records)
    base_rmse = rmse(truth, base)
    oracle_rmse = rmse(truth, oracle)
    gain = base_rmse - oracle_rmse
    expected_base = float(get_nested(config, "data.exp226_oof.expected_rmse"))
    base_parity_abs = abs(base_rmse - expected_base)
    minimum_gain = float(
        get_nested(config, "stage_0_headroom.gates.minimum_rmse_gain_vs_exp226_ft")
    )
    required_improved_folds = int(
        get_nested(config, "stage_0_headroom.gates.required_improved_folds")
    )
    technical_pass = bool(
        np.isfinite(truth).all()
        and np.isfinite(base).all()
        and np.isfinite(oracle).all()
        and not ordered.duplicated(list(KEY_COLUMNS)).any()
        and (
            not enforce_expected_counts
            or len(ordered) == int(get_nested(config, "validation.expected_rows"))
        )
        and (
            not enforce_expected_counts
            or ordered["well_id"].nunique()
            == int(get_nested(config, "validation.expected_wells"))
        )
        and (
            not enforce_expected_counts
            or len(segment_table)
            == int(get_nested(config, "validation.expected_nonempty_segments"))
        )
        and base_parity_abs <= 1e-8
    )
    improved_folds = int(fold_metrics["gate_pass"].sum())
    scientific_pass = bool(
        technical_pass and gain >= minimum_gain and improved_folds == required_improved_folds
    )
    truth_frame = ordered[["well_id", "row_idx", "tvt_true"]]
    readout_frame = ordered[["well_id", "row_idx", "fold", "segment_id"]].copy()
    readout_frame["exp226_tvt_pred"] = base
    readout_frame["oracle_k16_tvt_pred"] = oracle
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "status": "completed" if technical_pass else "technical_fail",
        "rows": len(ordered),
        "wells": int(ordered["well_id"].nunique()),
        "segments": len(segment_table),
        "folds": len(fold_metrics),
        "exp226_rmse": base_rmse,
        "expected_exp226_rmse": expected_base,
        "exp226_rmse_parity_abs_ft": base_parity_abs,
        "k16_oracle_offset_rmse": oracle_rmse,
        "rmse_gain_vs_exp226_ft": gain,
        "minimum_required_gain_ft": minimum_gain,
        "minimum_required_fold_gain_ft": minimum_fold_gain,
        "improved_folds": improved_folds,
        "required_improved_folds": required_improved_folds,
        "technical_pass": technical_pass,
        "scientific_pass": scientific_pass,
        "decision": "PASS_STAGE0" if scientific_pass else "FAIL_CLOSE_BRANCH",
        "truth_content_sha256": frame_content_sha256(
            truth_frame, ("well_id", "row_idx", "tvt_true")
        ),
        "segment_target_content_sha256": frame_content_sha256(
            segment_table,
            (
                "well_id",
                "fold",
                "segment_id",
                "segment_row_count",
                "segment_mean_residual",
            ),
        ),
        "oracle_readout_content_sha256": frame_content_sha256(
            readout_frame,
            (
                "well_id",
                "row_idx",
                "fold",
                "segment_id",
                "exp226_tvt_pred",
                "oracle_k16_tvt_pred",
            ),
        ),
        "oracle_offset_persisted": False,
        "oracle_prediction_persisted": False,
        "segment_target_persisted": False,
        "models": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "stage_1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    segment_counts = segment_table.drop(columns="segment_mean_residual")
    return OracleReadout(base, oracle, segment_table, fold_metrics, segment_counts, summary)


# %% [markdown]
# ## 7. Fixed Stage 0 gate and reproducibility evidence

# %%
def evaluate_stage0_gate(
    overall_gain_ft: float,
    fold_gains_ft: Sequence[float],
    config: Mapping[str, Any],
    *,
    technical_pass: bool = True,
) -> dict[str, Any]:
    minimum_gain = float(
        get_nested(config, "stage_0_headroom.gates.minimum_rmse_gain_vs_exp226_ft")
    )
    minimum_fold_gain = float(
        get_nested(config, "stage_0_headroom.gates.minimum_fold_gain_ft")
    )
    required = int(get_nested(config, "stage_0_headroom.gates.required_improved_folds"))
    fold_passes = [bool(float(value) >= minimum_fold_gain) for value in fold_gains_ft]
    passed = bool(
        technical_pass
        and float(overall_gain_ft) >= minimum_gain
        and len(fold_passes) == required
        and sum(fold_passes) == required
    )
    return {
        "technical_pass": technical_pass,
        "overall_gain_ft": float(overall_gain_ft),
        "minimum_overall_gain_ft": minimum_gain,
        "fold_gains_ft": [float(value) for value in fold_gains_ft],
        "minimum_fold_gain_ft": minimum_fold_gain,
        "fold_passes": fold_passes,
        "required_passed_folds": required,
        "decision": "PASS_STAGE0" if passed else "FAIL_CLOSE_BRANCH",
        "stage_1_may_be_implemented": passed,
    }


def input_manifest_frame(input_evidence: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": input_evidence["name"],
                "path": input_evidence["path"],
                "bytes": input_evidence["bytes"],
                "file_sha256": input_evidence["file_sha256"],
                "decompressed_sha256": input_evidence["decompressed_sha256"],
                "rows": input_evidence["rows"],
                "wells": input_evidence["wells"],
                "loaded_columns": "|".join(input_evidence["loaded_columns"]),
                "truth_columns_loaded_before_freeze": input_evidence[
                    "truth_columns_loaded_before_freeze"
                ],
            }
        ]
    )


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and generated artifacts

# %%
def run_stage0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_execution_authorization=True)
    artifacts = output_artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)

    contract = build_stage0_contract(config)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_stage0_contract.json"
    write_json(contract_path, contract)

    exp226_path = resolve_exp226_oof(config)
    target_free, input_evidence = load_exp226_target_free(exp226_path, config)
    assigned, segment_counts = assign_k16_segments(
        target_free, config, enforce_expected_counts=True
    )
    freeze = build_target_free_freeze(assigned, segment_counts, input_evidence, config)

    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_stage0_input_manifest.csv"
    freeze_path = artifacts / f"{OUTPUT_PREFIX}_stage0_target_free_freeze.json"
    segment_counts_path = artifacts / f"{OUTPUT_PREFIX}_stage0_segment_counts.csv"
    write_csv(input_manifest_path, input_manifest_frame(input_evidence))
    write_json(freeze_path, freeze)
    write_csv(segment_counts_path, segment_counts)

    truth = load_exp226_truth(
        exp226_path,
        target_free_contract_sha256=str(freeze["target_free_contract_sha256"]),
    )
    joined = attach_truth_after_freeze(
        assigned,
        truth,
        target_free_contract_sha256=str(freeze["target_free_contract_sha256"]),
    )
    readout = build_oracle_readout(joined, config, enforce_expected_counts=True)
    gate = evaluate_stage0_gate(
        float(readout.summary["rmse_gain_vs_exp226_ft"]),
        readout.fold_metrics["rmse_gain_ft"].tolist(),
        config,
        technical_pass=bool(readout.summary["technical_pass"]),
    )
    if gate["decision"] != readout.summary["decision"]:
        raise RuntimeError("Stage 0 decision implementations disagree")

    fold_metrics_path = artifacts / f"{OUTPUT_PREFIX}_stage0_fold_metrics.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_stage0_summary.json"
    write_csv(fold_metrics_path, readout.fold_metrics)
    summary = {
        **readout.summary,
        "gate": gate,
        "contract_sha256": contract["contract_sha256"],
        "input_decompressed_sha256": input_evidence["decompressed_sha256"],
        "target_free_contract_sha256": freeze["target_free_contract_sha256"],
        "fold_map_sha256": freeze["fold_map_sha256"],
        "segment_assignment_sha256": freeze["segment_assignment_sha256"],
        "target_free_content_sha256": freeze["target_free_content_sha256"],
        "generated_artifacts_exclude_oracle_offsets_predictions_and_targets": True,
    }
    write_json(summary_path, summary)

    evidence_paths = (
        contract_path,
        input_manifest_path,
        freeze_path,
        fold_metrics_path,
        segment_counts_path,
        summary_path,
    )
    sha_manifest = pd.DataFrame(artifact_evidence(path) for path in evidence_paths)
    sha_manifest_path = artifacts / f"{OUTPUT_PREFIX}_stage0_sha_manifest.csv"
    write_csv(sha_manifest_path, sha_manifest)

    metrics_path = KAGGLE_WORKING_ROOT / "metrics.json"
    if not KAGGLE_WORKING_ROOT.exists():
        metrics_path = experiment_dir() / "metrics.stage0.runtime.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "stage0_completed",
            "route": "ensemble",
            "metric": "rmse",
            "stage0": summary,
            "cv": None,
            "public_lb": None,
            "private_lb": None,
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Strict-nested exp226 and target-free Stage 1 feature surface

# %%
def validate_stage1_contract(
    config: Mapping[str, Any], *, execution_mode: str | None = None
) -> dict[str, Any]:
    exact = {
        "experiment.route": "ensemble",
        "implementation.scope": "stage_0_and_stage_1_train",
        "implementation.stage_1_enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.n_outer_folds": 5,
        "validation.n_inner_folds": 4,
        "validation.outer_valid_parent_parity_max_abs_ft": 1e-8,
        "segmentation.k_segments": 16,
        "target.aggregation": "float64_arithmetic_mean",
        "target.sample_weight": "segment_row_count",
        "target.broadcast": "constant_to_all_rows_in_segment",
        "target.clipping": "none",
        "target.shrinkage": "none",
        "target.taper": "none",
        "target.interpolation": "none",
        "target.slope": "disabled",
        "features.include_lgb_oof_features": False,
        "features.u_projection.include_lgb_oof_features": False,
        "model.config_name": "exp228_lgb1_single_fixed",
        "model.params.num_leaves": 64,
        "model.params.min_child_samples": 40,
        "model.params.subsample": 0.474,
        "model.params.colsample_bytree": 0.393,
        "model.params.learning_rate": 0.0093,
        "model.params.n_estimators": 10000,
        "model.params.random_state": 0,
        "model.params.deterministic": True,
        "model.params.force_col_wise": True,
        "model.params.n_jobs": 8,
        "model.params.num_threads": 8,
        "model.early_stopping_rounds": 250,
        "execution_contract.stage_1_if_stage_0_pass.active_variants": 1,
        "execution_contract.stage_1_if_stage_0_pass.model_configs": 1,
        "execution_contract.stage_1_if_stage_0_pass.outer_folds": 5,
        "execution_contract.stage_1_if_stage_0_pass.boosters": 5,
        "execution_contract.stage_1_if_stage_0_pass.gpu": False,
        "execution_contract.stage_1_if_stage_0_pass.parent_control_retraining": False,
        "execution_contract.stage_1_implementation_approved": True,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    changed = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if changed:
        raise ValueError(f"exp333 frozen Stage 1 contract changed: {changed}")
    if tuple(get_nested(config, "features.allowed_groups", ())) != ALLOWED_STAGE1_GROUPS:
        raise ValueError("Stage 1 feature group allowlist changed")
    if tuple(get_nested(config, "features.structural_columns", ())) != (
        STRUCTURAL_FEATURE_COLUMNS
    ):
        raise ValueError("Stage 1 structural feature contract changed")
    if tuple(get_nested(config, "model.active_variants", ())) != (
        "k16_mean_residual_offset",
    ):
        raise ValueError("Stage 1 must contain exactly one active variant")
    if execution_mode is not None:
        selected = get_nested(config, "execution_contract.selected_stage")
        approved_key = {
            "preflight": "execution_contract.stage_1_preflight_approved",
            "train": "execution_contract.stage_1_run_approved",
        }.get(execution_mode)
        required_stage = {"preflight": "stage_1_preflight", "train": "stage_1_train"}.get(
            execution_mode
        )
        if approved_key is None or required_stage is None:
            raise ValueError(f"unknown Stage 1 execution mode: {execution_mode}")
        if not (
            selected == required_stage
            and bool(get_nested(config, "execution_contract.kaggle_push_approved"))
            and bool(get_nested(config, approved_key))
        ):
            raise RuntimeError(f"Stage 1 {execution_mode} execution is not authorized")
    return {"fixed_values": exact, "execution_mode": execution_mode}


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


def load_stage1_dependencies() -> tuple[Any, Any]:
    try:
        from inputs.exp226_source import connortynan_k16_reproduction as exp226_source
        from inputs.exp228_source import direct_residual_correction_on_exp226 as exp228_source
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stage 1 parent sources were not bootstrapped; prepare the Kaggle package first"
        ) from exc
    return exp226_source, exp228_source


def load_parent_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "inputs" / "exp226_source" / "config.yaml",
        project_root()
        / "experiments"
        / "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
        / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            value = yaml.safe_load(path.read_text()) or {}
            if isinstance(value, dict):
                return value
    raise FileNotFoundError("frozen exp226 parent config was not found")


def stable_inner_fold_manifest(
    well_ids: Sequence[str], outer_fold: int, n_inner_folds: int = 4
) -> pd.DataFrame:
    if n_inner_folds != 4:
        raise ValueError("exp333 inner split is fixed to four folds")
    records = []
    for well_id in sorted(set(map(str, well_ids))):
        digest = hashlib.sha256(
            f"exp333|outer={int(outer_fold)}|well={well_id}".encode()
        ).hexdigest()
        records.append({"well_id": well_id, "inner_digest": digest})
    manifest = pd.DataFrame(records).sort_values(
        ["inner_digest", "well_id"], kind="mergesort"
    )
    manifest["outer_fold"] = int(outer_fold)
    manifest["inner_fold"] = np.arange(len(manifest), dtype=np.int64) % n_inner_folds
    return manifest[["outer_fold", "well_id", "inner_fold", "inner_digest"]].reset_index(
        drop=True
    )


def _parent_prediction_rows(
    target_wells: Sequence[Any],
    source_wells: Sequence[Any],
    parent: Any,
    params: Any,
    *,
    outer_fold: int,
    role: str,
    inner_fold: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    fit_started = time.perf_counter()
    fields = parent.build_fields(list(source_wells), params)
    kappa = parent.fit_kappa(list(source_wells), fields, params)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    for well in target_wells:
        result = parent.predict_well(well, fields, kappa, params)
        segment_id = np.asarray(well.segid, dtype=np.int16)
        if not np.array_equal(segment_id, exact_k16_segment_ids(int(well.n))):
            raise ValueError(f"parent K16 segment parity failed for {well.wid}")
        frames.append(
            pd.DataFrame(
                {
                    "outer_fold": int(outer_fold),
                    "role": role,
                    "inner_fold": int(inner_fold),
                    "well_id": str(well.wid),
                    "row_idx": np.asarray(well.suffix_row_idx, dtype=np.int64),
                    "suffix_offset": np.arange(int(well.n), dtype=np.int64),
                    "segment_id": segment_id,
                    "tvt_pred": np.asarray(result.pred, dtype=np.float64),
                }
            )
        )
    predict_seconds = time.perf_counter() - predict_started
    if not frames:
        raise ValueError(f"strict nested fit produced no {role} predictions")
    return pd.concat(frames, ignore_index=True), {
        "fits": 1,
        "prediction_well_runs": len(target_wells),
        "fit_seconds": fit_seconds,
        "prediction_seconds": predict_seconds,
    }


def _parent_fit_only(
    source_wells: Sequence[Any], parent: Any, params: Any
) -> dict[str, float | int]:
    fit_started = time.perf_counter()
    fields = parent.build_fields(list(source_wells), params)
    parent.fit_kappa(list(source_wells), fields, params)
    return {
        "fits": 1,
        "prediction_well_runs": 0,
        "fit_seconds": time.perf_counter() - fit_started,
        "prediction_seconds": 0.0,
    }


@dataclass(frozen=True)
class NestedPredictionBundle:
    predictions: pd.DataFrame
    fold_manifest: pd.DataFrame
    timing: dict[str, Any]
    outer_valid_parity_max_abs_ft: float


def generate_strict_nested_predictions(
    wells: Sequence[Any],
    saved_exp226: pd.DataFrame,
    parent: Any,
    params: Any,
    config: Mapping[str, Any],
    *,
    target_well_ids: set[str] | None = None,
) -> NestedPredictionBundle:
    saved_fold = (
        saved_exp226[["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    well_by_id = {str(well.wid): well for well in wells}
    if set(well_by_id) != set(saved_fold):
        missing_parent = sorted(set(saved_fold) - set(well_by_id))
        missing_saved = sorted(set(well_by_id) - set(saved_fold))
        raise ValueError(
            f"parent/saved exp226 well identity mismatch: {missing_parent[:5]}, {missing_saved[:5]}"
        )
    target_ids = set(well_by_id) if target_well_ids is None else set(target_well_ids)
    if not target_ids.issubset(well_by_id):
        raise ValueError("preflight target wells are outside the saved exp226 identity")

    prediction_frames: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []
    n_outer = int(get_nested(config, "validation.n_outer_folds"))
    n_inner = int(get_nested(config, "validation.n_inner_folds"))
    for outer_fold in range(n_outer):
        outer_train_ids = sorted(
            well_id for well_id, fold in saved_fold.items() if int(fold) != outer_fold
        )
        outer_valid_ids = sorted(
            well_id for well_id, fold in saved_fold.items() if int(fold) == outer_fold
        )
        inner_manifest = stable_inner_fold_manifest(outer_train_ids, outer_fold, n_inner)
        manifests.append(inner_manifest)
        inner_by_well = inner_manifest.set_index("well_id")["inner_fold"].to_dict()
        for inner_fold in range(n_inner):
            source_ids = [
                well_id
                for well_id in outer_train_ids
                if int(inner_by_well[well_id]) != inner_fold
            ]
            valid_ids = [
                well_id
                for well_id in outer_train_ids
                if int(inner_by_well[well_id]) == inner_fold and well_id in target_ids
            ]
            if not valid_ids:
                if target_well_ids is not None:
                    timing_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "role": "inner_fit_only",
                            **_parent_fit_only(
                                [well_by_id[well_id] for well_id in source_ids],
                                parent,
                                params,
                            ),
                        }
                    )
                continue
            frame, timing = _parent_prediction_rows(
                [well_by_id[well_id] for well_id in valid_ids],
                [well_by_id[well_id] for well_id in source_ids],
                parent,
                params,
                outer_fold=outer_fold,
                role="inner_oof_train",
                inner_fold=inner_fold,
            )
            prediction_frames.append(frame)
            timing_rows.append(
                {"outer_fold": outer_fold, "inner_fold": inner_fold, "role": "inner", **timing}
            )
        selected_outer_valid = [well_id for well_id in outer_valid_ids if well_id in target_ids]
        if selected_outer_valid:
            frame, timing = _parent_prediction_rows(
                [well_by_id[well_id] for well_id in selected_outer_valid],
                [well_by_id[well_id] for well_id in outer_train_ids],
                parent,
                params,
                outer_fold=outer_fold,
                role="outer_valid",
                inner_fold=-1,
            )
            prediction_frames.append(frame)
            timing_rows.append(
                {"outer_fold": outer_fold, "inner_fold": -1, "role": "outer", **timing}
            )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["outer_fold", "role", "well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if predictions.duplicated(["outer_fold", "role", "well_id", "row_idx"]).any():
        raise ValueError("strict nested prediction keys are not unique within context")
    outer_valid = predictions.loc[predictions["role"].eq("outer_valid")]
    parity = outer_valid.merge(
        saved_exp226[["well_id", "row_idx", "fold", "tvt_pred"]],
        left_on=["well_id", "row_idx", "outer_fold"],
        right_on=["well_id", "row_idx", "fold"],
        how="left",
        validate="one_to_one",
        suffixes=("_nested", "_saved"),
    )
    if parity["tvt_pred_saved"].isna().any() or len(parity) != len(outer_valid):
        raise ValueError("outer-valid parent parity join did not cover all rows")
    max_abs = float(
        np.max(
            np.abs(
                parity["tvt_pred_nested"].to_numpy(np.float64)
                - parity["tvt_pred_saved"].to_numpy(np.float64)
            )
        )
    )
    tolerance = float(get_nested(config, "validation.outer_valid_parent_parity_max_abs_ft"))
    if max_abs > tolerance:
        raise ValueError(f"outer-valid exp226 parity failed: {max_abs} > {tolerance}")
    timing_frame = pd.DataFrame(timing_rows)
    timing_summary = {
        "fits": int(timing_frame["fits"].sum()),
        "prediction_well_runs": int(timing_frame["prediction_well_runs"].sum()),
        "fit_seconds": float(timing_frame["fit_seconds"].sum()),
        "prediction_seconds": float(timing_frame["prediction_seconds"].sum()),
        "details": timing_frame.to_dict(orient="records"),
    }
    return NestedPredictionBundle(
        predictions=predictions,
        fold_manifest=pd.concat(manifests, ignore_index=True),
        timing=timing_summary,
        outer_valid_parity_max_abs_ft=max_abs,
    )


def hashed_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)]
    digest = hashlib.sha256(canonical_json_bytes({"columns": list(columns)}))
    digest.update(pd.util.hash_pandas_object(selected, index=False).to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def load_target_free_exp072_frame(
    path: Path, config: Mapping[str, Any], *, well_ids: set[str] | None = None
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    header = [str(column) for column in pd.read_csv(path, nrows=0).columns]
    if not {"id", "well", "target"}.issubset(header):
        raise ValueError("exp072 cache schema is incomplete")
    usecols = [column for column in header if column != "target"]
    frame = pd.read_csv(path, usecols=usecols, dtype={"id": str, "well": str})
    if "target" in frame.columns:
        raise ValueError("exp072 target was loaded before the Stage 1 feature freeze")
    if well_ids is not None:
        frame = frame.loc[frame["well"].astype(str).isin(well_ids)].copy()
    feature_columns = [column for column in frame.columns if column not in {"id", "well"}]
    expected = int(get_nested(config, "data.exp072_feature_cache.expected_feature_count"))
    if len(feature_columns) != expected:
        raise ValueError(f"expected {expected} exp072 features, found {len(feature_columns)}")
    cache_spec = get_nested(config, "data.exp072_feature_cache")
    schema_path = resolve_existing(
        str(cache_spec["schema_filename"]), cache_spec["schema_patterns"]
    )
    schema = pd.read_csv(schema_path).sort_values("feature_index", kind="mergesort")
    if schema["feature"].astype(str).tolist() != feature_columns:
        raise ValueError("exp072 cache columns do not match the frozen feature schema")
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame.duplicated(["id", "well"]).any():
        raise ValueError("exp072 cache keys are not unique")
    metadata = {
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
    return frame.reset_index(drop=True), feature_columns, metadata


def _row_index_from_id(ids: pd.Series) -> np.ndarray:
    values = pd.to_numeric(
        ids.astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
    ).to_numpy(np.float64)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("exp072 id suffix is not an integer row index")
    return values.astype(np.int64)


@dataclass(frozen=True)
class Stage1FeatureSurface:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    schema: pd.DataFrame
    metadata: dict[str, Any]
    projection_summary: pd.DataFrame
    grwr_summary: pd.DataFrame


def add_target_free_anchor_columns(
    frame: pd.DataFrame, train_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for well_id in sorted(frame["well"].astype(str).unique()):
        path = train_dir / f"{well_id}__horizontal_well.csv"
        raw = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        known = raw.loc[pd.to_numeric(raw["TVT_input"], errors="coerce").notna()]
        if known.empty:
            raise ValueError(f"no known TVT_input prefix for anchor recovery: {well_id}")
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
    result["known_prefix_rows"] = result["well"].map(anchors["known_prefix_rows"]).astype(
        np.int32
    )
    if result[["anchor_md", "anchor_z0", "anchor_t0"]].isna().any().any():
        raise ValueError("target-free anchor merge produced missing values")
    delta = (
        result["last_known_tvt"].to_numpy(np.float32)
        - result["anchor_t0"].to_numpy(np.float32)
    )
    max_abs = float(np.max(np.abs(delta)))
    if max_abs > 0.05:
        raise ValueError(f"target-free anchor parity failed: max abs={max_abs}")
    return result, {
        "anchor_wells": len(anchors),
        "anchor_t0_vs_last_known_abs_max": max_abs,
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
        "raw_columns_loaded": ["MD", "Z", "TVT_input"],
        "target_columns_loaded": 0,
    }


def build_stage1_feature_surface(
    config: Mapping[str, Any],
    train_dir: Path,
    feature_source: Any,
    *,
    well_ids: set[str] | None = None,
) -> Stage1FeatureSurface:
    cache_spec = get_nested(config, "data.exp072_feature_cache")
    cache_path = resolve_existing(str(cache_spec["filename"]), cache_spec["patterns"])
    base, _base_columns, cache_meta = load_target_free_exp072_frame(
        cache_path, config, well_ids=well_ids
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
    for group in ALLOWED_STAGE1_GROUPS:
        for column in group_columns[group]:
            if column not in selected:
                selected.append(column)
                column_group[column] = group
    forbidden_exact = {
        "tvt_true",
        "target",
        "error",
        "abs_error",
        "oracle_candidate",
        "well_id",
    }
    forbidden_hits = sorted(forbidden_exact.intersection(selected))
    learned_or_selector = sorted(
        column for column in selected if column.startswith("ll_") or "selector" in column.lower()
    )
    if forbidden_hits or learned_or_selector:
        raise ValueError(
            f"forbidden Stage 1 feature columns: {forbidden_hits + learned_or_selector}"
        )
    surface = pd.DataFrame(
        {
            "well_id": base["well"].astype(str),
            "row_idx": _row_index_from_id(base["id"]),
            "md_since": pd.to_numeric(base["md_since"], errors="coerce").astype(np.float64),
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
        raise ValueError("Stage 1 row feature keys are not unique")
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
    metadata = {
        "cache": cache_meta,
        "anchor": anchor_meta,
        "grwr": grwr_meta,
        "allowed_groups": list(ALLOWED_STAGE1_GROUPS),
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
    return Stage1FeatureSurface(
        frame=surface.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True),
        feature_columns=tuple(selected),
        schema=schema,
        metadata=metadata,
        projection_summary=projection_summary,
        grwr_summary=grwr_summary,
    )


# %% [markdown]
# ## 10. K16 segment samples and fixed LightGBM training

# %%
def aggregate_stage1_segments(
    nested_rows: pd.DataFrame,
    row_surface: pd.DataFrame,
    truth: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    context = nested_rows.merge(
        row_surface,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    ).merge(truth, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    required = ["md_since", "tvt_true"]
    if context[required].isna().any().any():
        missing = context[required].columns[context[required].isna().any(axis=0)].tolist()
        raise ValueError(f"Stage 1 context has uncovered columns: {missing}")
    context = context.sort_values(["well_id", "suffix_offset"], kind="mergesort").reset_index(
        drop=True
    )
    group_keys = ["well_id", "segment_id"]
    group_index = context.groupby(group_keys, sort=True, observed=True).ngroup().to_numpy(
        np.int64
    )
    group_meta = (
        context.groupby(group_keys, sort=True, observed=True)
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

    def finite_mean(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        sums = np.bincount(group_index[finite], weights=array[finite], minlength=n_groups)
        counts = np.bincount(group_index[finite], minlength=n_groups)
        result = np.full(n_groups, np.nan, dtype=np.float64)
        np.divide(sums, counts, out=result, where=counts > 0)
        return result

    segment = group_meta.drop(columns=["segment_md_min", "segment_md_max", "exp226_pred_end"])
    segment["segment_position"] = (
        segment["segment_id"].to_numpy(np.float64) + 0.5
    ) / K_SEGMENTS
    segment["segment_md_span"] = (
        group_meta["segment_md_max"].to_numpy(np.float64)
        - group_meta["segment_md_min"].to_numpy(np.float64)
    )
    segment["exp226_pred_mean"] = finite_mean(context["tvt_pred"].to_numpy())
    segment["exp226_pred_end_minus_start"] = (
        group_meta["exp226_pred_end"].to_numpy(np.float64)
        - group_meta["exp226_pred_start"].to_numpy(np.float64)
    )
    for column in feature_columns:
        segment[column] = finite_mean(context[column].to_numpy())
    residual = context["tvt_true"].to_numpy(np.float64) - context["tvt_pred"].to_numpy(
        np.float64
    )
    segment["segment_mean_residual"] = finite_mean(residual)
    ordered = [
        "outer_fold",
        "role",
        "inner_fold",
        "well_id",
        "segment_id",
        *STRUCTURAL_FEATURE_COLUMNS[1:],
        *feature_columns,
        "segment_mean_residual",
    ]
    return segment.loc[:, ordered].sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)


def fit_stage1_fold(
    train_segments: pd.DataFrame,
    valid_segments: pd.DataFrame,
    model_features: Sequence[str],
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    model_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(get_nested(config, "model.params"))
    model = LGBMRegressor(**params)
    model.fit(
        train_segments.loc[:, list(model_features)],
        train_segments["segment_mean_residual"].to_numpy(np.float64),
        sample_weight=train_segments["segment_row_count"].to_numpy(np.float64),
        eval_set=[
            (
                valid_segments.loc[:, list(model_features)],
                valid_segments["segment_mean_residual"].to_numpy(np.float64),
            )
        ],
        eval_sample_weight=[valid_segments["segment_row_count"].to_numpy(np.float64)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(int(get_nested(config, "model.early_stopping_rounds")), verbose=True),
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
        "params": params,
    }
    return prediction, importance, record


def broadcast_valid_segment_predictions(
    valid_rows: pd.DataFrame,
    valid_segments: pd.DataFrame,
    segment_prediction: np.ndarray,
    row_surface: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    lookup = valid_segments[["well_id", "segment_id", "segment_mean_residual"]].copy()
    lookup["segment_offset_pred"] = np.asarray(segment_prediction, dtype=np.float64)
    rows = valid_rows.merge(
        lookup,
        on=["well_id", "segment_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        row_surface[["well_id", "row_idx", "md_since"]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    ).merge(truth, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if rows[["segment_offset_pred", "md_since", "tvt_true"]].isna().any().any():
        raise ValueError("Stage 1 row broadcast did not cover validation rows")
    rows["tvt_pred_stage1"] = rows["tvt_pred"] + rows["segment_offset_pred"]
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
    return rows.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 11. Stage 1 promotion gates, artifacts, and orchestration

# %%
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
    base = rmse(
        frame.loc[selected, "tvt_true"].to_numpy(),
        frame.loc[selected, "tvt_pred"].to_numpy(),
    )
    stage1 = rmse(
        frame.loc[selected, "tvt_true"].to_numpy(),
        frame.loc[selected, "tvt_pred_stage1"].to_numpy(),
    )
    return {
        "scope": label,
        "rows": int(selected.sum()),
        "wells": int(frame.loc[selected, "well_id"].nunique()),
        "exp226_rmse": base,
        "stage1_rmse": stage1,
        "delta_stage1_minus_exp226": stage1 - base,
    }


def evaluate_stage1_outputs(
    row_oof: pd.DataFrame,
    segment_oof: pd.DataFrame,
    hidden_assignment: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pooled = metric_comparison(row_oof, np.ones(len(row_oof), dtype=bool), label="pooled")
    fold_rows = []
    for fold in range(int(get_nested(config, "validation.n_outer_folds"))):
        row = metric_comparison(row_oof, row_oof["outer_fold"].eq(fold), label=f"fold_{fold}")
        valid_segments = segment_oof.loc[segment_oof["outer_fold"].eq(fold)]
        weight = valid_segments["segment_row_count"].to_numpy(np.float64)
        target = valid_segments["segment_mean_residual"].to_numpy(np.float64)
        prediction = valid_segments["segment_offset_pred"].to_numpy(np.float64)
        zero_rmse = float(np.sqrt(np.average(target * target, weights=weight)))
        predicted_rmse = float(
            np.sqrt(np.average(np.square(target - prediction), weights=weight))
        )
        row.update(
            {
                "outer_fold": fold,
                "segment_target_zero_prior_weighted_rmse": zero_rmse,
                "segment_target_model_weighted_rmse": predicted_rmse,
                "segment_target_improved": bool(predicted_rmse < zero_rmse),
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
        metric_comparison(row_oof, mask, label=name) for name, mask in bucket_masks.items()
    )
    hidden_lookup = hidden_assignment.set_index("well_id")
    hidden_rows = []
    for column in (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ):
        mask = row_oof["well_id"].map(hidden_lookup[column]).eq("valid")
        hidden_rows.append(metric_comparison(row_oof, mask, label=column))
    hidden_metrics = pd.DataFrame(hidden_rows)
    boundary_metrics = pd.DataFrame(
        [
            metric_comparison(
                row_oof, row_oof["boundary_band_pm8"], label="segment_boundary_pm8_rows"
            )
        ]
    )
    by_well_rows = []
    for well_id, part in row_oof.groupby("well_id", sort=True):
        base_score = rmse(part["tvt_true"].to_numpy(), part["tvt_pred"].to_numpy())
        new_score = rmse(
            part["tvt_true"].to_numpy(), part["tvt_pred_stage1"].to_numpy()
        )
        by_well_rows.append(
            {
                "well_id": str(well_id),
                "rows": len(part),
                "exp226_rmse": base_score,
                "stage1_rmse": new_score,
                "delta_stage1_minus_exp226": new_score - base_score,
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    base_p95 = float(by_well["exp226_rmse"].quantile(0.95))
    stage1_p95 = float(by_well["stage1_rmse"].quantile(0.95))
    worst_well_delta = float(by_well["delta_stage1_minus_exp226"].max())
    gates = get_nested(config, "promotion_gates")
    bucket_lookup = bucket_metrics.set_index("scope")
    hidden_lookup_metrics = hidden_metrics.set_index("scope")
    checks = {
        "pooled_rmse": pooled["stage1_rmse"] <= float(gates["maximum_pooled_rmse"]),
        "gain_vs_exp226": pooled["exp226_rmse"] - pooled["stage1_rmse"]
        >= float(gates["minimum_rmse_gain_vs_exp226_ft"]),
        "improved_outer_folds": int(
            (fold_metrics["delta_stage1_minus_exp226"] < 0.0).sum()
        )
        >= int(gates["minimum_improved_outer_folds_vs_exp226"]),
        "segment_target_all_folds": int(fold_metrics["segment_target_improved"].sum())
        == int(gates["segment_target_weighted_rmse_improved_folds_vs_zero_prior"]),
        "near_0_250_nonworse": float(
            bucket_lookup.loc["near_0_250", "delta_stage1_minus_exp226"]
        )
        <= float(gates["maximum_near_0_250_rmse_delta_vs_exp226_ft"]),
        "1000_plus_nonworse": float(
            bucket_lookup.loc["1000_plus", "delta_stage1_minus_exp226"]
        )
        <= float(gates["maximum_1000_plus_rmse_delta_vs_exp226_ft"]),
        "hidden_spatial_nonworse": float(
            hidden_lookup_metrics.loc[
                "verification_like_spatial_role", "delta_stage1_minus_exp226"
            ]
        )
        <= float(gates["maximum_hidden_like_spatial_rmse_delta_vs_exp226_ft"]),
        "hidden_typewell_nonworse": float(
            hidden_lookup_metrics.loc[
                "verification_like_typewell_purged_role", "delta_stage1_minus_exp226"
            ]
        )
        <= float(gates["maximum_hidden_like_typewell_purged_rmse_delta_vs_exp226_ft"]),
        "boundary_nonworse": float(
            boundary_metrics.loc[0, "delta_stage1_minus_exp226"]
        )
        <= float(gates["maximum_segment_boundary_band_rmse_delta_vs_exp226_ft"]),
        "by_well_p95_nonworse": stage1_p95 - base_p95
        <= float(gates["maximum_by_well_p95_delta_vs_exp226_ft"]),
        "worst_well_delta": worst_well_delta
        <= float(gates["maximum_worst_well_delta_vs_exp226_ft"]),
    }
    scientific_pass = all(bool(value) for value in checks.values())
    inference_candidate = bool(
        scientific_pass
        and pooled["stage1_rmse"]
        <= float(gates["inference_candidate_additional_maximum_rmse"])
    )
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
        "worst_well_delta": worst_well_delta,
        "gate_checks": checks,
        "scientific_pass": scientific_pass,
        "inference_candidate_threshold_pass": inference_candidate,
        "decision": "PASS_STAGE1" if scientific_pass else "FAIL_CLOSE_BRANCH",
    }


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


def resolve_hidden_assignment(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
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


def run_stage1_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_stage1_contract(config, execution_mode="preflight")
    parent, feature_source = load_stage1_dependencies()
    train_dir = resolve_train_dir()
    exp226_path = resolve_exp226_oof(config)
    saved, _input_evidence = load_exp226_target_free(exp226_path, config)
    params = parent.params_from_config(load_parent_config())
    wells = parent.load_train_wells(train_dir, params)
    fold_map = saved[["well_id", "fold"]].drop_duplicates()
    selected: set[str] = set()
    requested_wells = int(get_nested(config, "runtime.preflight.wells"))
    fold_count = int(get_nested(config, "validation.n_outer_folds"))
    for fold, part in fold_map.groupby("fold", sort=True):
        ordered = sorted(
            part["well_id"].astype(str),
            key=lambda well_id: hashlib.sha256(
                f"exp333|preflight|fold={int(fold)}|well={well_id}".encode()
            ).hexdigest(),
        )
        quota = requested_wells // fold_count + (
            1 if int(fold) < requested_wells % fold_count else 0
        )
        selected.update(ordered[:quota])
    if len(selected) != requested_wells:
        raise ValueError("preflight well selection did not preserve the fixed fold-balanced count")
    started = time.perf_counter()
    nested = generate_strict_nested_predictions(
        wells, saved, parent, params, config, target_well_ids=selected
    )
    nested_elapsed = time.perf_counter() - started
    feature_started = time.perf_counter()
    feature_surface = build_stage1_feature_surface(
        config, train_dir, feature_source, well_ids=selected
    )
    feature_elapsed = time.perf_counter() - feature_started
    elapsed = time.perf_counter() - started
    planned_prediction_runs = int(get_nested(config, "nested_exp226.total_prediction_well_runs"))
    projected_nested = float(nested.timing["fit_seconds"]) + float(
        nested.timing["prediction_seconds"]
    ) * planned_prediction_runs / int(nested.timing["prediction_well_runs"])
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    projected_features = feature_elapsed * expected_rows / len(feature_surface.frame)
    fixed_reserve = float(
        get_nested(
            config,
            "runtime.preflight.fixed_five_booster_and_artifact_io_reserve_seconds",
        )
    )
    projected = projected_nested + projected_features + fixed_reserve
    limit = int(get_nested(config, "runtime.preflight.required_projected_stage_1_seconds_at_most"))
    summary = {
        "stage": "stage_1_preflight",
        "status": "completed",
        "selected_wells": len(selected),
        "elapsed_seconds": elapsed,
        "nested_elapsed_seconds": nested_elapsed,
        "feature_elapsed_seconds": feature_elapsed,
        "selected_feature_rows": len(feature_surface.frame),
        "nested_timing": nested.timing,
        "outer_valid_parent_parity_max_abs_ft": nested.outer_valid_parity_max_abs_ft,
        "projected_nested_stage1_seconds": projected_nested,
        "projected_feature_stage1_seconds": projected_features,
        "fixed_five_booster_and_artifact_io_reserve_seconds": fixed_reserve,
        "projected_full_stage1_seconds": projected,
        "required_projected_stage1_seconds_at_most": limit,
        "runtime_gate_pass": projected <= limit,
        "boosters": 0,
        "models": 0,
        "full_stage1_executed": False,
    }
    artifacts = output_artifacts_dir()
    write_json(artifacts / f"{OUTPUT_PREFIX}_stage1_preflight_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def run_stage1_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_stage1_contract(config, execution_mode="train")
    artifacts = output_artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)
    parent, feature_source = load_stage1_dependencies()
    train_dir = resolve_train_dir()
    exp226_path = resolve_exp226_oof(config)
    saved, saved_evidence = load_exp226_target_free(exp226_path, config)
    params = parent.params_from_config(load_parent_config())
    wells = parent.load_train_wells(train_dir, params)
    nested = generate_strict_nested_predictions(wells, saved, parent, params, config)
    feature_surface = build_stage1_feature_surface(config, train_dir, feature_source)
    if set(feature_surface.frame["well_id"].unique()) != set(saved["well_id"].unique()):
        raise ValueError("Stage 1 feature surface does not cover saved exp226 wells")
    feature_freeze = {
        "saved_exp226_decompressed_sha256": saved_evidence["decompressed_sha256"],
        "fold_manifest_sha256": hashed_frame_sha256(
            nested.fold_manifest,
            ("outer_fold", "well_id", "inner_fold", "inner_digest"),
        ),
        "segment_assignment_sha256": hashed_frame_sha256(
            nested.predictions,
            ("outer_fold", "role", "inner_fold", "well_id", "row_idx", "segment_id"),
        ),
        "nested_exp226_prediction_sha256": hashed_frame_sha256(
            nested.predictions,
            (
                "outer_fold",
                "role",
                "inner_fold",
                "well_id",
                "row_idx",
                "tvt_pred",
            ),
        ),
        **feature_surface.metadata,
        "outer_valid_parent_parity_max_abs_ft": nested.outer_valid_parity_max_abs_ft,
        "residual_target_attached": False,
    }
    feature_freeze["feature_freeze_sha256"] = mapping_sha256(feature_freeze)
    truth = load_exp226_truth(
        exp226_path, target_free_contract_sha256=feature_freeze["feature_freeze_sha256"]
    )
    hidden, hidden_evidence = resolve_hidden_assignment(config)
    model_features = (*STRUCTURAL_FEATURE_COLUMNS, *feature_surface.feature_columns)
    row_oof_frames: list[pd.DataFrame] = []
    segment_oof_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    model_dir = artifacts / f"{OUTPUT_PREFIX}_stage1_lgb_models"
    for outer_fold in range(int(get_nested(config, "validation.n_outer_folds"))):
        train_rows = nested.predictions.loc[
            nested.predictions["outer_fold"].eq(outer_fold)
            & nested.predictions["role"].eq("inner_oof_train")
        ].copy()
        valid_rows = nested.predictions.loc[
            nested.predictions["outer_fold"].eq(outer_fold)
            & nested.predictions["role"].eq("outer_valid")
        ].copy()
        train_segments = aggregate_stage1_segments(
            train_rows,
            feature_surface.frame,
            truth,
            feature_surface.feature_columns,
        )
        valid_segments = aggregate_stage1_segments(
            valid_rows,
            feature_surface.frame,
            truth,
            feature_surface.feature_columns,
        )
        segment_prediction, importance, model_record = fit_stage1_fold(
            train_segments,
            valid_segments,
            model_features,
            config,
            outer_fold=outer_fold,
            model_dir=model_dir,
        )
        row_oof_frames.append(
            broadcast_valid_segment_predictions(
                valid_rows,
                valid_segments,
                segment_prediction,
                feature_surface.frame,
                truth,
            )
        )
        valid_output = valid_segments[
            [
                "outer_fold",
                "well_id",
                "segment_id",
                "segment_row_count",
                "segment_mean_residual",
            ]
        ].copy()
        valid_output["segment_offset_pred"] = segment_prediction
        segment_oof_frames.append(valid_output)
        importance_frames.append(importance)
        model_records.append(model_record)
    row_oof = pd.concat(row_oof_frames, ignore_index=True).sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    segment_oof = pd.concat(segment_oof_frames, ignore_index=True).sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(row_oof) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("Stage 1 OOF row coverage mismatch")
    if len(segment_oof) != int(get_nested(config, "validation.expected_nonempty_segments")):
        raise ValueError("Stage 1 OOF segment coverage mismatch")
    evaluation = evaluate_stage1_outputs(row_oof, segment_oof, hidden, config)
    importance = pd.concat(importance_frames, ignore_index=True)
    model_manifest = {
        "experiment": EXPERIMENT_NAME,
        "variant": "k16_mean_residual_offset",
        "config": "exp228_lgb1_single_fixed",
        "boosters": len(model_records),
        "models": model_records,
        "feature_columns": list(model_features),
        "feature_schema_sha256": feature_surface.metadata["row_feature_schema_sha256"],
        "feature_freeze_sha256": feature_freeze["feature_freeze_sha256"],
    }
    model_manifest_path = artifacts / f"{OUTPUT_PREFIX}_stage1_model_manifest.json"
    write_json(model_manifest_path, model_manifest)

    contract = {
        "stage": "stage_1_strict_nested_k16_segment_residual",
        "route": "ensemble",
        "variants": 1,
        "model_configs": 1,
        "outer_folds": 5,
        "boosters": 5,
        "nested_fits": 25,
        "nested_prediction_well_runs": 3865,
        "parent_control_retraining": False,
        "gpu": False,
        "inference": False,
        "submission": False,
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    input_manifest = pd.DataFrame(
        [
            {"name": "saved_exp226_oof", **saved_evidence},
            {"name": "exp072_target_free_cache", **feature_surface.metadata["cache"]},
            {"name": "hidden_like_assignment", **hidden_evidence},
        ]
    )
    paths = {
        "contract": artifacts / f"{OUTPUT_PREFIX}_stage1_contract.json",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_input_manifest.csv",
        "fold_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_fold_manifest.csv",
        "feature_schema": artifacts / f"{OUTPUT_PREFIX}_stage1_feature_schema.csv",
        "projection_summary": artifacts
        / f"{OUTPUT_PREFIX}_stage1_projection_feature_summary.csv",
        "grwr_summary": artifacts / f"{OUTPUT_PREFIX}_stage1_grwr_feature_summary.csv",
        "nested": artifacts / f"{OUTPUT_PREFIX}_stage1_nested_exp226_predictions.csv.gz",
        "segment": artifacts / f"{OUTPUT_PREFIX}_stage1_segment_predictions.csv.gz",
        "oof": artifacts / f"{OUTPUT_PREFIX}_stage1_oof_predictions.csv.gz",
        "fold_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_fold_metrics.csv",
        "bucket_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_bucket_metrics.csv",
        "hidden_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_hidden_like_metrics.csv",
        "boundary_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_boundary_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv",
        "importance": artifacts / f"{OUTPUT_PREFIX}_stage1_feature_importance.csv",
        "summary": artifacts / f"{OUTPUT_PREFIX}_stage1_summary.json",
    }
    write_json(paths["contract"], contract)
    write_csv(paths["input_manifest"], input_manifest)
    write_csv(paths["fold_manifest"], nested.fold_manifest)
    write_csv(paths["feature_schema"], feature_surface.schema)
    write_csv(paths["projection_summary"], feature_surface.projection_summary)
    write_csv(paths["grwr_summary"], feature_surface.grwr_summary)
    write_csv_gzip(paths["nested"], nested.predictions)
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
            "segment_offset_pred",
            "tvt_pred_stage1",
            "boundary_band_pm8",
        ]
    ]
    write_csv_gzip(paths["oof"], oof_output)
    write_csv(paths["fold_metrics"], evaluation["fold_metrics"])
    write_csv(paths["bucket_metrics"], evaluation["bucket_metrics"])
    write_csv(paths["hidden_metrics"], evaluation["hidden_metrics"])
    write_csv(paths["boundary_metrics"], evaluation["boundary_metrics"])
    write_csv(paths["by_well"], evaluation["by_well"])
    write_csv(paths["importance"], importance)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_1",
        "status": "completed",
        "decision": evaluation["decision"],
        "scientific_pass": evaluation["scientific_pass"],
        "inference_candidate_threshold_pass": evaluation[
            "inference_candidate_threshold_pass"
        ],
        "inference_approved": False,
        "submission_approved": False,
        "pooled": evaluation["pooled"],
        "gate_checks": evaluation["gate_checks"],
        "by_well_exp226_p95": evaluation["by_well_exp226_p95"],
        "by_well_stage1_p95": evaluation["by_well_stage1_p95"],
        "by_well_p95_delta": evaluation["by_well_p95_delta"],
        "worst_well_delta": evaluation["worst_well_delta"],
        "feature_freeze": feature_freeze,
        "nested_timing": nested.timing,
        "segment_target_sha256": hashed_frame_sha256(
            segment_oof,
            (
                "outer_fold",
                "well_id",
                "segment_id",
                "segment_row_count",
                "segment_mean_residual",
            ),
        ),
        "oof_prediction_sha256": hashed_frame_sha256(
            oof_output,
            ("well_id", "row_idx", "outer_fold", "tvt_pred_stage1"),
        ),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "boosters": 5,
        "parent_control_retraining": False,
    }
    write_json(paths["summary"], summary)
    evidence_paths = [*paths.values(), model_manifest_path, *sorted(model_dir.glob("*.txt"))]
    sha_manifest = pd.DataFrame(artifact_evidence(path) for path in evidence_paths)
    write_csv(artifacts / f"{OUTPUT_PREFIX}_stage1_sha_manifest.csv", sha_manifest)
    metrics_path = KAGGLE_WORKING_ROOT / "metrics.json"
    if not KAGGLE_WORKING_ROOT.exists():
        metrics_path = experiment_dir() / "metrics.stage1.runtime.json"
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
# ## 12. Setup, contract preview, and guarded execution

# %%
CONFIG = load_config()
CONTRACT_PREVIEW = validate_scientific_contract(
    CONFIG, require_execution_authorization=False
)
STAGE1_CONTRACT_PREVIEW = validate_stage1_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "selected_stage": get_nested(CONFIG, "execution_contract.selected_stage"),
            "stage0_models": get_nested(CONFIG, "execution_contract.stage_0.model_configs"),
            "stage0_boosters": get_nested(CONFIG, "execution_contract.stage_0.boosters"),
            "stage1_enabled": get_nested(CONFIG, "implementation.stage_1_enabled"),
            "kaggle_push_approved": get_nested(
                CONFIG, "execution_contract.kaggle_push_approved"
            ),
            "stage0_run_approved": get_nested(
                CONFIG, "execution_contract.stage_0_run_approved"
            ),
            "stage1_preflight_approved": get_nested(
                CONFIG, "execution_contract.stage_1_preflight_approved"
            ),
            "stage1_run_approved": get_nested(
                CONFIG, "execution_contract.stage_1_run_approved"
            ),
            "stage1_boosters": get_nested(
                CONFIG, "execution_contract.stage_1_if_stage_0_pass.boosters"
            ),
            "message": "Stage 1 implementation is ready; all execution remains fail-closed.",
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    SELECTED_STAGE = get_nested(CONFIG, "execution_contract.selected_stage")
    if SELECTED_STAGE == "stage_0":
        run_stage0_experiment(CONFIG)
    elif SELECTED_STAGE == "stage_1_preflight":
        run_stage1_preflight(CONFIG)
    elif SELECTED_STAGE == "stage_1_train":
        run_stage1_experiment(CONFIG)
    else:
        print("Implementation-only mode: no Stage 0 or Stage 1 execution was started.")
