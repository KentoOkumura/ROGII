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
# # exp333 exp226 K16 segment residual offset target — Stage 0
#
# This compact self-contained notebook implements only the approved zero-model
# Stage 0 headroom audit.  It freezes the saved exp226 row identity, source
# folds, prediction, and exact K16 assignment before loading suffix truth.  It
# then measures the diagnostic oracle mean residual per K16 segment.  Oracle
# offsets and oracle predictions are never written as deployable artifacts.
# Stage 1, inference, and submission remain unavailable pending separate gates
# and approvals.

# %% [markdown]
# ## Contents
# 1. Imports and immutable Stage 0 boundary
# 2. Runtime, configuration, path, SHA, and serialization helpers
# 3. Frozen scientific and execution contract
# 4. Saved exp226 target-free input checks
# 5. Exact exp226 K16 assignment and target-free freeze
# 6. Late truth attachment and diagnostic oracle readout
# 7. Fixed Stage 0 gate and reproducibility evidence
# 8. Full Kaggle CPU orchestration and generated artifacts
# 9. Setup, contract preview, and guarded execution

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
        "implementation.scope": "stage_0_only",
        "implementation.stage_0_enabled": True,
        "implementation.stage_1_enabled": False,
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
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    changed = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if changed:
        raise ValueError(f"exp333 frozen Stage 0 contract changed: {changed}")

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
# ## 9. Setup, contract preview, and guarded execution

# %%
CONFIG = load_config()
CONTRACT_PREVIEW = validate_scientific_contract(
    CONFIG, require_execution_authorization=False
)
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
            "message": "Stage 0 implementation is ready; execution is fail-closed.",
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    if get_nested(CONFIG, "execution_contract.selected_stage") == "stage_0":
        run_stage0_experiment(CONFIG)
    else:
        print("Implementation-only mode: no Stage 0 readout was executed.")
