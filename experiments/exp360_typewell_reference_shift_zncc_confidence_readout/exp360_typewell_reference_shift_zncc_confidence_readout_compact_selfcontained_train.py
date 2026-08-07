# %% [markdown]
# # exp360 Type Well reference-shift ZNCC confidence readout
#
# Stage 0 is a deterministic, zero-booster diagnostic. It scores raw finite
# horizontal GR against GR_typewell(tvt_geop + delta) with ZNCC, constructs the
# six frozen confidence families plus matched historical/permutation controls,
# freezes every target-free table and SHA, and only then reads exp264/exp226
# truth. No candidate, prediction, selector, model, HMM, inference, or
# submission is changed.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Safe raw inputs, parent controls, and Type Well ZNCC scoring
# 5. Six confidence families and stable shift-label control
# 6. Fold-wise quantile and target-free freeze
# 7. Post-freeze exp264/exp226 truth and hidden-like loaders
# 8. Block RMSE, row-weighted AUC, scopes, and fixed gate
# 9. Metrics and generated artifacts
# 10. Setup and configuration preview
# 11. Run the approved Stage 0 readout only

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp360_typewell_reference_shift_zncc_confidence_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXPECTED_SHIFTS = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)
FAMILIES = (
    "best_nonzero_minus_zero_zncc",
    "low_zero_shift_zncc",
    "zero_shift_rank",
    "absolute_top1_shift",
    "top1_shift_jump_from_previous_block",
    "three_block_sign_inconsistency",
)
PRIMARY_FAMILY = "best_nonzero_minus_zero_zncc"
SEQUENCE_FAMILIES = (
    "top1_shift_jump_from_previous_block",
    "three_block_sign_inconsistency",
)
VARIANTS = (
    "real_zncc",
    "historical_raw_gaussian",
    "stable_permutation",
)
VARIANT_PREFIX = {
    "real_zncc": "",
    "historical_raw_gaussian": "raw_gaussian_",
    "stable_permutation": "permutation_",
}
TIE_ORDER = np.asarray(
    [0.0, -2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -20.0, 20.0, -40.0, 40.0, -80.0, 80.0],
    dtype=np.float64,
)
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "actual_tvt",
    "target",
    "error",
    "abs_error",
    "tvt_pred",
    "prediction",
    "selector_compact_addonly__lgb_mean__pred_tvt",
}


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP360_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").is_dir():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd().resolve()


def load_experiment_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp360 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return experiment_dir() / "metrics.json"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column, dtype in normalized.dtypes.items():
        if isinstance(dtype, pd.StringDtype):
            normalized[column] = normalized[column].astype(object)
    return normalized


def dataframe_content_sha(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = _normalize_frame_for_hash(selected)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    root = project_root()
    found: dict[str, Path] = {}
    for raw in map(str, patterns):
        path = Path(raw)
        direct = path if path.is_absolute() else root / path
        if direct.is_file() and direct.stat().st_size > 0:
            found[str(direct.resolve())] = direct
            continue
        searches = [raw]
        if not path.is_absolute():
            searches.append(str(root / raw))
        for search in searches:
            for match in glob.glob(search, recursive=True):
                candidate = Path(match)
                if candidate.is_file() and candidate.stat().st_size > 0:
                    found[str(candidate.resolve())] = candidate
    return list(found.values())


def resolve_file(
    patterns: Sequence[str],
    *,
    label: str,
    expected_file_sha256: str | None = None,
    expected_decompressed_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for candidate in expand_existing_paths(patterns):
        row: dict[str, Any] = {
            "path": str(candidate),
            "bytes": candidate.stat().st_size,
            "file_sha256": sha256_path(candidate),
        }
        if expected_decompressed_sha256 is not None:
            row["decompressed_sha256"] = sha256_gzip_decompressed(candidate)
        evidence.append(row)
        if expected_file_sha256 is not None and row["file_sha256"] != expected_file_sha256:
            continue
        if (
            expected_decompressed_sha256 is not None
            and row["decompressed_sha256"] != expected_decompressed_sha256
        ):
            continue
        return candidate, row
    raise FileNotFoundError(
        f"Could not resolve {label} with its fixed SHA contract: {evidence[:8]}"
    )


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_gzip_decompressed(path),
    }


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    present = set(map(str, columns)).intersection(FORBIDDEN_PRE_FREEZE_COLUMNS)
    if present:
        raise ValueError(f"truth/error columns are forbidden before freeze: {sorted(present)}")


def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "ensemble":
        raise ValueError("exp360 route must remain ensemble")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp360 Stage 0 implementation must be enabled")
    family_spec = get_nested(config, "model.target_free_families") or {}
    observed_families = tuple(
        list(family_spec.get("primary", ())) + list(family_spec.get("supporting", ()))
    )
    if observed_families != FAMILIES or tuple(family_spec.get("primary", ())) != (
        PRIMARY_FAMILY,
    ):
        raise ValueError("the six-family primary/supporting contract changed")
    shifts = np.asarray(get_nested(config, "data.shifts_ft"), dtype=np.float64)
    if not np.array_equal(shifts, EXPECTED_SHIFTS):
        raise ValueError("the fixed 13-shift bank changed")
    if int(get_nested(config, "data.block_size")) != 512:
        raise ValueError("exp360 requires fixed non-overlapping H512 blocks")
    score_spec = get_nested(config, "model.score") or {}
    expected_score_contract = {
        "name": "raw_finite_zncc",
        "horizontal_gr_imputation": "none",
        "typewell_missing_gr_fill": "forward_then_backward",
        "typewell_interpolation": "linear_with_endpoint_hold",
        "minimum_finite_pairs": 32,
        "minimum_observed_std": 1.0e-6,
        "minimum_expected_std": 1.0e-6,
        "invalid_score": -1.0,
        "core_supported_requires_zero_valid": True,
        "core_supported_minimum_valid_candidates": 2,
    }
    if score_spec != expected_score_contract:
        raise ValueError(f"the fixed raw-finite ZNCC score contract changed: {score_spec}")
    if list(get_nested(config, "validation.expected_folds")) != [0, 1, 2, 3, 4]:
        raise ValueError("the five-fold readout contract changed")
    observed_tie_order = np.asarray(
        get_nested(config, "model.tie_policy.exact_tie_order_ft"), dtype=float
    ).tolist()
    if observed_tie_order != TIE_ORDER.tolist():
        raise ValueError("the deterministic exact-tie order changed")
    counts = get_nested(config, "execution_contract") or {}
    expected_counts = {
        "zncc_score_variants": 1,
        "stable_negative_controls": 1,
        "saved_historical_controls": 1,
        "core_feature_families": 6,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_beam_hmm_runs": 0,
        "parent_control_retraining": False,
    }
    if counts != expected_counts:
        raise ValueError(f"zero-booster execution contract changed: {counts}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("Stage 0 implementation is not approved")
    forbidden_true = (
        "execution.run_inference",
        "execution.create_submission",
        "inference.enabled",
        "inference.create_submission",
        "implementation.inference_enabled",
        "implementation.submission_enabled",
    )
    if any(bool(get_nested(config, key)) for key in forbidden_true):
        raise ValueError("inference and submission must remain disabled")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
        and bool(get_nested(config, "runtime.kaggle.train_run_on_push"))
    ):
        raise RuntimeError("exp360 Kaggle package/push/run is not approved")


class TruthAccessLedger:
    def __init__(self) -> None:
        self.frozen = False
        self.count_before_freeze = 0

    def mark_frozen(self) -> None:
        self.frozen = True

    def register_truth_access(self) -> None:
        if not self.frozen:
            self.count_before_freeze += 1
            raise ValueError("truth access attempted before target-free freeze")


# %% [markdown]
# ## 4. Safe raw inputs, parent controls, and Type Well ZNCC scoring


# %%
def load_exp280_target_free_scores(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp280_source")
    score_path, score_evidence = resolve_file(
        spec["score_patterns"],
        label="exp280 target-free shift score",
        expected_decompressed_sha256=str(spec["score_decompressed_sha256"]),
    )
    contract_path, contract_evidence = resolve_file(
        spec["contract_patterns"],
        label="exp280 score contract",
    )
    contract = json.loads(contract_path.read_text())
    if bool(contract.get("truth_attached")):
        raise ValueError("exp280 score contract must be truth-free")
    if contract.get("target_free_score_content_sha256") != str(spec["score_content_sha256"]):
        raise ValueError("exp280 target-free score content declaration changed")
    if contract.get("scientific_contract_sha256") != str(spec["scientific_contract_sha256"]):
        raise ValueError("exp280 scientific contract SHA changed")
    if list(map(float, contract.get("shift_bank_ft", []))) != EXPECTED_SHIFTS.tolist():
        raise ValueError("exp280 score contract shift bank changed")
    if int(contract.get("block_rows", -1)) != 512:
        raise ValueError("exp280 block contract changed")

    scores = pd.read_csv(score_path, dtype={"well_id": str})
    assert_no_forbidden_columns(scores.columns)
    required = {
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
        "shift_slot",
        "shift_ft",
        "likelihood_mean",
        "likelihood_rank",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"exp280 score table missing {missing}")
    integer_columns = (
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "shift_slot",
        "likelihood_rank",
    )
    for column in integer_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.int64)
    numeric_columns = (
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
        "shift_ft",
        "likelihood_mean",
    )
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.float64)
    scores["well_id"] = scores["well_id"].astype(str)
    scores = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)

    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(scores) != expected_blocks * len(EXPECTED_SHIFTS):
        raise ValueError("exp280 score row count changed")
    if scores["well_id"].nunique() != expected_wells:
        raise ValueError("exp280 score well count changed")
    group_size = scores.groupby(["well_id", "block_id"], sort=False).size()
    if not group_size.eq(len(EXPECTED_SHIFTS)).all() or len(group_size) != expected_blocks:
        raise ValueError("each exp280 block must contain exactly 13 shifts")
    observed_shifts = scores["shift_ft"].to_numpy(np.float64).reshape(
        expected_blocks, len(EXPECTED_SHIFTS)
    )
    expected_matrix = np.broadcast_to(EXPECTED_SHIFTS, observed_shifts.shape)
    if not np.array_equal(observed_shifts, expected_matrix):
        raise ValueError("exp280 shift order changed")
    if not np.isfinite(scores["likelihood_mean"].to_numpy(np.float64)).all():
        raise ValueError("exp280 likelihood contains non-finite values")
    return scores, [
        {"name": "exp280_target_free_shift_scores", **score_evidence},
        {
            "name": "exp280_score_contract",
            **contract_evidence,
            "declared_target_free_content_sha256": contract[
                "target_free_score_content_sha256"
            ],
            "scientific_contract_sha256": contract["scientific_contract_sha256"],
        },
    ]


def load_lineage_contracts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = get_nested(config, "data.exp340_source")
    feature_path, feature_evidence = resolve_file(
        spec["target_free_block_features_patterns"],
        label="exp340 target-free block feature lineage",
        expected_file_sha256=str(spec["target_free_block_features_sha256"]),
    )
    _summary_path, summary_evidence = resolve_file(
        spec["summary_patterns"],
        label="exp340 summary lineage",
        expected_file_sha256=str(spec["summary_sha256"]),
    )
    return [
        {"name": "exp340_target_free_block_features_lineage", **feature_evidence},
        {
            "name": "exp340_summary_lineage",
            **summary_evidence,
            "content_policy": "sha_only_not_materialized_before_target_free_freeze",
        },
    ]


def load_exp226_safe(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_source")
    path, evidence = resolve_file(
        spec["patterns"],
        label="exp226 OOF safe columns",
        expected_decompressed_sha256=str(spec["expected_decompressed_sha256"]),
    )
    safe_columns = list(map(str, spec["safe_columns"]))
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    assert_no_forbidden_columns(frame.columns)
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(
        np.float64
    )
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF has duplicate well_id/row_idx")
    if not np.isfinite(frame["tvt_geop"].to_numpy(np.float64)).all():
        raise ValueError("exp226 safe tvt_geop must be finite")
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or frame["well_id"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
        or sorted(frame["fold"].unique().tolist())
        != list(get_nested(config, "validation.expected_folds"))
    ):
        raise ValueError("exp226 safe row/well/fold contract changed")
    per_well = frame.groupby("well_id", sort=True).agg(
        fold_count=("fold", "nunique"),
        rows=("row_idx", "size"),
        suffix_min=("suffix_offset", "min"),
        suffix_max=("suffix_offset", "max"),
    )
    if (
        not per_well["fold_count"].eq(1).all()
        or not per_well["suffix_min"].eq(0).all()
        or not per_well["suffix_max"].eq(per_well["rows"] - 1).all()
    ):
        raise ValueError("exp226 per-well safe suffix/fold contract changed")
    return frame, path, {
        "name": "exp226_safe_oof",
        **evidence,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "safe_columns": safe_columns,
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_train_root(config: Mapping[str, Any]) -> Path:
    evidence: list[str] = []
    for raw in map(str, get_nested(config, "data.train_root_candidates") or ()):
        path = Path(raw)
        candidate = path if path.is_absolute() else project_root() / path
        evidence.append(str(candidate))
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"raw train root not found in {evidence}")


def load_horizontal_safe(path: Path) -> pd.DataFrame:
    safe_columns = ["MD", "GR", "TVT_input"]
    frame = pd.read_csv(path, usecols=safe_columns)[safe_columns]
    if list(frame.columns) != safe_columns or "TVT" in frame.columns:
        raise ValueError("horizontal safe loader exposed an unexpected column")
    return frame


def prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    frame = typewell[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].ffill().bfill()
    frame = frame.groupby("TVT", sort=True, as_index=False)["GR"].mean()
    values = frame[["TVT", "GR"]].to_numpy(np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("typewell requires at least two finite unique TVT/GR rows")
    return values[:, 0], values[:, 1]


def centered_normalized_correlation(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_pairs: int,
    minimum_observed_std: float,
    minimum_expected_std: float,
    invalid_score: float,
) -> tuple[float, bool, int]:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    pair_count = int(finite.sum())
    if pair_count < minimum_pairs:
        return float(invalid_score), False, pair_count
    left = left[finite]
    right = right[finite]
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_std = float(np.sqrt(np.mean(np.square(left_centered))))
    right_std = float(np.sqrt(np.mean(np.square(right_centered))))
    if left_std <= minimum_observed_std or right_std <= minimum_expected_std:
        return float(invalid_score), False, pair_count
    denominator = float(
        np.sqrt(np.sum(np.square(left_centered)) * np.sum(np.square(right_centered)))
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float(invalid_score), False, pair_count
    score = float(np.dot(left_centered, right_centered) / denominator)
    if not np.isfinite(score):
        return float(invalid_score), False, pair_count
    return float(np.clip(score, -1.0, 1.0)), True, pair_count


def tie_priority_by_slot(shifts: np.ndarray = EXPECTED_SHIFTS) -> np.ndarray:
    priority = {float(shift): index for index, shift in enumerate(TIE_ORDER)}
    if set(priority) != set(map(float, shifts)):
        raise ValueError("tie order and shift bank differ")
    return np.asarray([priority[float(shift)] for shift in shifts], dtype=np.int64)


def tie_resolved_valid_slots(
    scores: np.ndarray,
    valid: np.ndarray,
    shifts: np.ndarray = EXPECTED_SHIFTS,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    slots = np.flatnonzero(mask)
    if not len(slots):
        return slots
    priority = tie_priority_by_slot(shifts)
    order = np.lexsort((priority[slots], -values[slots]))
    return slots[order]


def stable_valid_score_permutation(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    well_id: str,
    block_id: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    output = values.copy()
    slots = np.flatnonzero(mask)
    decorated = sorted(
        slots.tolist(),
        key=lambda slot: stable_uint64(
            EXPERIMENT_NAME,
            "stable_shift_label_permutation",
            well_id,
            int(block_id),
            int(slot),
        ),
    )
    if len(decorated) > 1 and decorated == slots.tolist():
        decorated = decorated[1:] + decorated[:1]
    output[slots] = values[np.asarray(decorated, dtype=np.int64)]
    return output


def score_well_target_free_zncc(
    oof_safe: pd.DataFrame,
    horizontal_safe: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_no_forbidden_columns(oof_safe.columns)
    if "TVT" in horizontal_safe.columns:
        raise ValueError("horizontal truth is forbidden during target-free scoring")
    required_oof = {"well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"}
    required_horizontal = {"MD", "GR", "TVT_input"}
    if not required_oof.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required_oof - set(oof_safe.columns))}")
    if not required_horizontal.issubset(horizontal_safe.columns):
        missing_horizontal = sorted(
            required_horizontal - set(horizontal_safe.columns)
        )
        raise ValueError(
            f"horizontal safe frame missing {missing_horizontal}"
        )
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("ZNCC scoring requires exactly one non-empty well and fold")
    row_idx = oof["row_idx"].to_numpy(np.int64)
    suffix = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("exp226 suffix_offset must be contiguous from zero")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_safe):
        raise ValueError("exp226 row_idx falls outside the horizontal file")
    if horizontal_safe.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF must align only to the unknown suffix")

    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    shifts = np.asarray(get_nested(config, "data.shifts_ft"), dtype=np.float64)
    geop = oof["tvt_geop"].to_numpy(np.float64)
    expected = np.column_stack(
        [np.interp(geop + shift, typewell_tvt, typewell_gr) for shift in shifts]
    )
    observed = pd.to_numeric(
        horizontal_safe.iloc[row_idx]["GR"], errors="coerce"
    ).to_numpy(np.float64)
    md = pd.to_numeric(horizontal_safe["MD"], errors="raise").to_numpy(np.float64)
    known_positions = np.flatnonzero(horizontal_safe["TVT_input"].notna().to_numpy())
    if not len(known_positions):
        raise ValueError("horizontal well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    md_since = md[row_idx] - md[last_known]
    block_size = int(get_nested(config, "data.block_size"))
    block_id = suffix // block_size
    score_spec = get_nested(config, "model.score")
    minimum_pairs = int(score_spec["minimum_finite_pairs"])
    minimum_observed_std = float(score_spec["minimum_observed_std"])
    minimum_expected_std = float(score_spec["minimum_expected_std"])
    invalid_score = float(score_spec["invalid_score"])
    well_id = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        selected = block_id == block
        positions = np.flatnonzero(selected)
        for slot, shift in enumerate(shifts):
            score, valid, pair_count = centered_normalized_correlation(
                observed[selected],
                expected[selected, slot],
                minimum_pairs=minimum_pairs,
                minimum_observed_std=minimum_observed_std,
                minimum_expected_std=minimum_expected_std,
                invalid_score=invalid_score,
            )
            rows.append(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(suffix[positions[0]]),
                    "block_end_suffix_offset": int(suffix[positions[-1]]),
                    "block_start_row_idx": int(row_idx[positions[0]]),
                    "block_end_row_idx": int(row_idx[positions[-1]]),
                    "block_row_count": int(selected.sum()),
                    "md_since_min_ft": float(np.min(md_since[selected])),
                    "md_since_max_ft": float(np.max(md_since[selected])),
                    "md_since_mid_ft": float(np.mean(md_since[selected])),
                    "observed_gr_share": float(np.isfinite(observed[selected]).mean()),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "zncc": score,
                    "valid": bool(valid),
                    "finite_pair_count": pair_count,
                }
            )
    score_frame = pd.DataFrame(rows).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    return score_frame.reset_index(drop=True), {
        "well_id": well_id,
        "fold": fold,
        "horizontal_rows": len(horizontal_safe),
        "evaluation_rows": len(oof),
        "blocks": int(block_id.max() + 1),
        "last_known_row_idx": last_known,
        "observed_eval_gr_share": float(np.isfinite(observed).mean()),
        "valid_candidate_share": float(score_frame["valid"].mean()),
    }


# %% [markdown]
# ## 5. Six confidence families and stable shift-label control


# %%
def pairwise_sign_inconsistency(values: np.ndarray) -> np.ndarray:
    shifts = np.asarray(values, dtype=np.float64)
    output = np.zeros(len(shifts), dtype=np.float64)
    signs = np.sign(shifts).astype(np.int8)
    for index in range(len(signs)):
        window = signs[max(0, index - 2) : index + 1]
        nonzero = window[window != 0]
        if len(nonzero) < 2:
            output[index] = 0.0
            continue
        disagreements = 0
        pairs = 0
        for left in range(len(nonzero)):
            for right in range(left + 1, len(nonzero)):
                pairs += 1
                disagreements += int(nonzero[left] != nonzero[right])
        output[index] = disagreements / pairs
    return output


def sequence_features(top1_shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(top1_shift, dtype=np.float64)
    jump = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        jump[1:] = np.abs(np.diff(values))
    inconsistency = pairwise_sign_inconsistency(values)
    return jump, inconsistency


def _variant_block_features(
    scores: pd.DataFrame,
    *,
    score_column: str,
    valid_column: str,
    variant: str,
    support_by_block: Mapping[tuple[str, int], bool] | None = None,
    permute_valid_scores: bool = False,
) -> pd.DataFrame:
    ordered = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    zero_slot = int(np.flatnonzero(EXPECTED_SHIFTS == 0.0)[0])
    metadata_columns = [
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
    ]
    for (well_id, block_id), part in ordered.groupby(
        ["well_id", "block_id"], sort=True, observed=True
    ):
        if len(part) != len(EXPECTED_SHIFTS) or not np.array_equal(
            part["shift_ft"].to_numpy(np.float64), EXPECTED_SHIFTS
        ):
            raise ValueError("every feature block requires the fixed ordered 13 shifts")
        values = part[score_column].to_numpy(np.float64)
        valid = part[valid_column].to_numpy(bool)
        if permute_valid_scores:
            values = stable_valid_score_permutation(
                values, valid, well_id=str(well_id), block_id=int(block_id)
            )
        core_supported = bool(valid[zero_slot] and valid.sum() >= 2)
        if support_by_block is not None:
            core_supported = bool(support_by_block[(str(well_id), int(block_id))])
        row = {column: part.iloc[0][column] for column in metadata_columns}
        row["variant"] = variant
        row["core_supported"] = core_supported
        row["valid_candidate_count"] = int(valid.sum())
        row["top1_shift_ft"] = np.nan
        row["best_nonzero_minus_zero_zncc"] = np.nan
        row["low_zero_shift_zncc"] = np.nan
        row["zero_shift_rank"] = np.nan
        row["absolute_top1_shift"] = np.nan
        row["top1_shift_jump_from_previous_block"] = np.nan
        row["three_block_sign_inconsistency"] = np.nan
        row["best_zncc"] = np.nan
        row["top1_minus_top2_zncc"] = np.nan
        if core_supported:
            ranked = tie_resolved_valid_slots(values, valid)
            nonzero = np.flatnonzero(valid & ~np.isclose(EXPECTED_SHIFTS, 0.0))
            if not len(ranked) or not len(nonzero):
                raise ValueError("supported block lacks a valid ranked nonzero candidate")
            top_slot = int(ranked[0])
            zero_rank_position = int(np.flatnonzero(ranked == zero_slot)[0])
            row["top1_shift_ft"] = float(EXPECTED_SHIFTS[top_slot])
            row["best_nonzero_minus_zero_zncc"] = float(
                np.max(values[nonzero]) - values[zero_slot]
            )
            row["low_zero_shift_zncc"] = float(-values[zero_slot])
            row["zero_shift_rank"] = float(
                zero_rank_position / max(len(ranked) - 1, 1)
            )
            row["absolute_top1_shift"] = float(abs(EXPECTED_SHIFTS[top_slot]))
            row["best_zncc"] = float(values[top_slot])
            row["top1_minus_top2_zncc"] = float(
                values[top_slot] - values[int(ranked[1])]
            )
        rows.append(row)
    features = pd.DataFrame(rows).sort_values(
        ["well_id", "block_id"], kind="mergesort"
    ).reset_index(drop=True)
    for _, positions in features.groupby("well_id", sort=True).indices.items():
        all_positions = np.asarray(positions, dtype=np.int64)
        supported_positions = all_positions[
            features.iloc[all_positions]["core_supported"].to_numpy(bool)
        ]
        supported_positions = supported_positions[
            np.argsort(
                features.iloc[supported_positions]["block_id"].to_numpy(np.int64),
                kind="mergesort",
            )
        ]
        shifts = features.iloc[supported_positions]["top1_shift_ft"].to_numpy(
            np.float64
        )
        jump, inconsistency = sequence_features(shifts)
        features.loc[
            supported_positions, "top1_shift_jump_from_previous_block"
        ] = jump
        features.loc[
            supported_positions, "three_block_sign_inconsistency"
        ] = inconsistency
    prefix = VARIANT_PREFIX[variant]
    rename = {
        family: f"{prefix}risk__{family}"
        for family in FAMILIES
    }
    rename.update(
        {
            "top1_shift_ft": f"{prefix}top1_shift_ft",
            "best_zncc": f"{prefix}best_zncc",
            "top1_minus_top2_zncc": f"{prefix}top1_minus_top2_zncc",
        }
    )
    return features.rename(columns=rename)


def build_target_free_block_features(
    zncc_scores: pd.DataFrame,
    raw_scores: pd.DataFrame,
) -> pd.DataFrame:
    assert_no_forbidden_columns(zncc_scores.columns)
    assert_no_forbidden_columns(raw_scores.columns)
    real = _variant_block_features(
        zncc_scores,
        score_column="zncc",
        valid_column="valid",
        variant="real_zncc",
    )
    support_map = {
        (str(row.well_id), int(row.block_id)): bool(row.core_supported)
        for row in real[["well_id", "block_id", "core_supported"]].itertuples(index=False)
    }
    permutation = _variant_block_features(
        zncc_scores,
        score_column="zncc",
        valid_column="valid",
        variant="stable_permutation",
        support_by_block=support_map,
        permute_valid_scores=True,
    )
    raw = raw_scores.copy()
    raw["valid"] = True
    historical = _variant_block_features(
        raw,
        score_column="likelihood_mean",
        valid_column="valid",
        variant="historical_raw_gaussian",
        support_by_block=support_map,
    )
    identity_columns = [
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
    ]
    numeric_metadata_columns = [
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
    ]
    output = real.drop(columns="variant")
    for variant_frame in (historical, permutation):
        if not variant_frame[identity_columns].equals(real[identity_columns]):
            raise ValueError("real/control feature block identity differs")
        if not np.allclose(
            variant_frame[numeric_metadata_columns].to_numpy(np.float64),
            real[numeric_metadata_columns].to_numpy(np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("real/control feature block metadata differs")
        non_feature_columns = {
            *identity_columns,
            *numeric_metadata_columns,
            "observed_gr_share",
            "variant",
            "core_supported",
            "valid_candidate_count",
        }
        value_columns = [
            column
            for column in variant_frame.columns
            if column not in non_feature_columns
        ]
        output = output.merge(
            variant_frame[identity_columns + value_columns],
            on=identity_columns,
            how="left",
            validate="one_to_one",
        )
    supported = output["core_supported"].to_numpy(bool)
    required = [
        f"{VARIANT_PREFIX[variant]}risk__{family}"
        for variant in VARIANTS
        for family in FAMILIES
    ]
    if not np.isfinite(output.loc[supported, required].to_numpy(np.float64)).all():
        raise ValueError("supported real/control target-free families must be finite")
    return output.sort_values(["well_id", "block_id"], kind="mergesort").reset_index(
        drop=True
    )


# %% [markdown]
# ## 6. Fold-wise quantile and target-free freeze


# %%
def fit_fold_quantile_boundaries(
    features: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    assert_no_forbidden_columns(features.columns)
    quantiles = list(map(float, get_nested(config, "model.attribution.quantiles")))
    if quantiles != [0.25, 0.75]:
        raise ValueError("exp360 fixes Q1/Q4 at 0.25/0.75")
    rows: list[dict[str, Any]] = []
    supported = features.loc[features["core_supported"]].copy()
    for fold, part in supported.groupby("fold", sort=True):
        for variant in VARIANTS:
            prefix = VARIANT_PREFIX[variant]
            for family in FAMILIES:
                values = part[f"{prefix}risk__{family}"].to_numpy(np.float64)
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError("fold quantiles require finite supported features")
                rows.append(
                    {
                        "fold": int(fold),
                        "variant": variant,
                        "family": family,
                        "q25_risk_boundary": float(np.quantile(values, 0.25)),
                        "q75_risk_boundary": float(np.quantile(values, 0.75)),
                        "blocks": len(part),
                        "finite_coverage": float(np.isfinite(values).mean()),
                        "risk_direction": "higher_is_more_exp264_error_risk",
                    }
                )
    output = pd.DataFrame(rows).sort_values(
        ["variant", "family", "fold"], kind="mergesort"
    )
    if len(output) != len(VARIANTS) * len(FAMILIES) * 5:
        raise ValueError("fold-wise family quantile coverage changed")
    return output.reset_index(drop=True)


def attach_frozen_quartile_flags(
    features: pd.DataFrame, boundaries: pd.DataFrame
) -> pd.DataFrame:
    output = features.copy()
    supported = output["core_supported"].to_numpy(bool)
    for variant in VARIANTS:
        prefix = VARIANT_PREFIX[variant]
        for family in FAMILIES:
            selected = boundaries.loc[
                boundaries["variant"].eq(variant)
                & boundaries["family"].eq(family)
            ].set_index("fold")
            q25 = output["fold"].map(selected["q25_risk_boundary"]).to_numpy(np.float64)
            q75 = output["fold"].map(selected["q75_risk_boundary"]).to_numpy(np.float64)
            risk = output[f"{prefix}risk__{family}"].to_numpy(np.float64)
            output[f"{prefix}q1__{family}"] = supported & (risk <= q25)
            output[f"{prefix}q4__{family}"] = supported & (risk >= q75)
    return output


def freeze_target_free_bundle(
    zncc_scores: pd.DataFrame,
    features: pd.DataFrame,
    boundaries: pd.DataFrame,
    input_manifest: list[dict[str, Any]],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    assert_no_forbidden_columns(zncc_scores.columns)
    assert_no_forbidden_columns(features.columns)
    artifacts = artifact_dir()
    score_path = artifacts / f"{OUTPUT_PREFIX}_target_free_zncc_scores.parquet"
    valid_mask_path = artifacts / f"{OUTPUT_PREFIX}_valid_masks.parquet"
    feature_path = artifacts / f"{OUTPUT_PREFIX}_target_free_features.parquet"
    quantile_path = artifacts / f"{OUTPUT_PREFIX}_fold_quantile_boundaries.csv"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_score_schema.json"
    input_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.json"
    contract_path = artifacts / f"{OUTPUT_PREFIX}_contract.json"
    freeze_path = artifacts / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    score_columns = [
        "well_id",
        "fold",
        "block_id",
        "shift_slot",
        "shift_ft",
        "zncc",
        "valid",
        "finite_pair_count",
    ]
    mask_columns = [
        "well_id",
        "fold",
        "block_id",
        "shift_slot",
        "shift_ft",
        "valid",
        "finite_pair_count",
    ]
    zncc_scores.to_parquet(score_path, index=False, compression="zstd")
    zncc_scores[mask_columns].to_parquet(
        valid_mask_path, index=False, compression="zstd"
    )
    features.to_parquet(feature_path, index=False, compression="zstd")
    boundaries.to_csv(quantile_path, index=False)
    schema = {
        "score_columns": [
            {"name": column, "dtype": str(zncc_scores[column].dtype)}
            for column in score_columns
        ],
        "feature_columns": [
            {"name": column, "dtype": str(dtype)} for column, dtype in features.dtypes.items()
        ],
        "families": list(FAMILIES),
        "primary_family": PRIMARY_FAMILY,
        "variants": list(VARIANTS),
        "sequence_families": list(SEQUENCE_FAMILIES),
        "forbidden_pre_freeze_columns": sorted(FORBIDDEN_PRE_FREEZE_COLUMNS),
    }
    write_json(schema_path, schema)
    write_json(input_path, {"inputs": input_manifest})
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "ensemble",
        "truth_attached": False,
        "readout_families": list(FAMILIES),
        "primary_family": PRIMARY_FAMILY,
        "variants": list(VARIANTS),
        "shift_bank_ft": EXPECTED_SHIFTS.tolist(),
        "tie_order_ft": TIE_ORDER.tolist(),
        "block_rows": 512,
        "minimum_finite_pairs": int(
            get_nested(config, "model.score.minimum_finite_pairs")
        ),
        "minimum_observed_std": float(
            get_nested(config, "model.score.minimum_observed_std")
        ),
        "minimum_expected_std": float(
            get_nested(config, "model.score.minimum_expected_std")
        ),
        "declared_post_freeze_input_sha256": {
            "exp264_oof_file": str(
                get_nested(config, "data.exp264_source.expected_sha256")
            ),
            "exp226_oof_decompressed": str(
                get_nested(config, "data.exp226_source.expected_decompressed_sha256")
            ),
            "hidden_like_assignment_file": str(
                get_nested(config, "data.hidden_like_assignment.expected_sha256")
            ),
        },
        "quantiles": [0.25, 0.75],
        "models": 0,
        "hmm_well_runs": 0,
        "boosters": 0,
        "prediction_changes": 0,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    write_json(contract_path, contract)
    supported = features["core_supported"].to_numpy(bool)
    coverage = float(supported.mean())
    support_by_well = features.groupby("well_id", sort=True)["core_supported"].any()
    primary_boundaries = boundaries.loc[
        boundaries["variant"].eq("real_zncc")
        & boundaries["family"].eq(PRIMARY_FAMILY)
    ]
    technical_checks = {
        "expected_blocks": len(features)
        == int(get_nested(config, "validation.expected_blocks")),
        "expected_wells": features["well_id"].nunique()
        == int(get_nested(config, "validation.expected_wells")),
        "expected_folds": sorted(features["fold"].unique().tolist())
        == list(get_nested(config, "validation.expected_folds")),
        "minimum_core_supported_block_coverage": coverage
        >= float(
            get_nested(
                config,
                "model.pass_requires_all_for_primary.minimum_core_supported_block_coverage",
            )
        ),
        "all_wells_have_supported_block": bool(support_by_well.all()),
        "primary_fold_quantiles_nonoverlapping": bool(
            (
                primary_boundaries["q75_risk_boundary"]
                > primary_boundaries["q25_risk_boundary"]
            ).all()
        ),
        "truth_access_count_zero": ledger.count_before_freeze == 0,
    }
    freeze = {
        "experiment": EXPERIMENT_NAME,
        "frozen": True,
        "truth_access_count_before_freeze": ledger.count_before_freeze,
        "truth_columns_loaded_before_freeze": [],
        "blocks": len(features),
        "wells": int(features["well_id"].nunique()),
        "supported_blocks": int(supported.sum()),
        "core_supported_block_coverage": coverage,
        "wells_with_supported_block": int(support_by_well.sum()),
        "technical_checks": technical_checks,
        "technical_passed": bool(all(technical_checks.values())),
        "score_schema_sha256": dataframe_schema_sha(zncc_scores[score_columns]),
        "score_content_sha256": dataframe_content_sha(zncc_scores[score_columns]),
        "valid_mask_content_sha256": dataframe_content_sha(zncc_scores[mask_columns]),
        "feature_schema_sha256": dataframe_schema_sha(features),
        "feature_content_sha256": dataframe_content_sha(features),
        "quantile_content_sha256": dataframe_content_sha(boundaries),
        "file_sha256": {
            "target_free_zncc_scores": sha256_path(score_path),
            "valid_masks": sha256_path(valid_mask_path),
            "target_free_features": sha256_path(feature_path),
            "fold_quantile_boundaries": sha256_path(quantile_path),
            "score_schema": sha256_path(schema_path),
            "input_manifest": sha256_path(input_path),
            "contract": sha256_path(contract_path),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(freeze_path, freeze)
    ledger.mark_frozen()
    return {
        "manifest": freeze,
        "manifest_path": freeze_path,
        "score_path": score_path,
        "valid_mask_path": valid_mask_path,
        "feature_path": feature_path,
        "quantile_path": quantile_path,
        "schema_path": schema_path,
        "input_path": input_path,
        "contract_path": contract_path,
    }


def verify_freeze(freeze: Mapping[str, Any], ledger: TruthAccessLedger) -> None:
    manifest = freeze["manifest"]
    if not bool(manifest["frozen"]) or int(manifest["truth_access_count_before_freeze"]) != 0:
        raise ValueError("target-free freeze contract is invalid")
    if not ledger.frozen or ledger.count_before_freeze != 0:
        raise ValueError("truth ledger is not cleanly frozen")
    file_map = {
        "target_free_zncc_scores": freeze["score_path"],
        "valid_masks": freeze["valid_mask_path"],
        "target_free_features": freeze["feature_path"],
        "fold_quantile_boundaries": freeze["quantile_path"],
        "score_schema": freeze["schema_path"],
        "input_manifest": freeze["input_path"],
        "contract": freeze["contract_path"],
    }
    for name, path in file_map.items():
        if sha256_path(path) != manifest["file_sha256"][name]:
            raise ValueError(f"frozen file changed after freeze: {name}")


# %% [markdown]
# ## 7. Post-freeze exp264/exp226 truth and hidden-like loaders


# %%
def aggregate_exp226_blocks(path: Path, *, block_size: int) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    usecols = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "tvt_true",
        "tvt_pred",
    ]
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"well_id": str},
        chunksize=250_000,
    ):
        for column in ("row_idx", "suffix_offset", "fold"):
            chunk[column] = pd.to_numeric(chunk[column], errors="raise").astype(np.int64)
        for column in ("tvt_true", "tvt_pred"):
            chunk[column] = pd.to_numeric(chunk[column], errors="raise").astype(np.float64)
        if not np.isfinite(chunk[["tvt_true", "tvt_pred"]].to_numpy(np.float64)).all():
            raise ValueError("exp226 OOF contains non-finite values")
        chunk["block_id"] = chunk["suffix_offset"].to_numpy(np.int64) // block_size
        chunk["exp226_squared_error"] = np.square(chunk["tvt_pred"] - chunk["tvt_true"])
        grouped = chunk.groupby(["well_id", "block_id"], sort=False, observed=True)
        partials.append(
            grouped.agg(
                fold_min=("fold", "min"),
                fold_max=("fold", "max"),
                exp226_rows=("row_idx", "size"),
                exp226_first_row_idx=("row_idx", "min"),
                exp226_last_row_idx=("row_idx", "max"),
                exp226_first_suffix_offset=("suffix_offset", "min"),
                exp226_last_suffix_offset=("suffix_offset", "max"),
                exp226_squared_error_sum=("exp226_squared_error", "sum"),
                exp226_truth_sum=("tvt_true", "sum"),
                exp226_truth_min=("tvt_true", "min"),
                exp226_truth_max=("tvt_true", "max"),
            ).reset_index()
        )
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(["well_id", "block_id"], sort=True, observed=True)
    blocks = grouped.agg(
        fold_min=("fold_min", "min"),
        fold_max=("fold_max", "max"),
        exp226_rows=("exp226_rows", "sum"),
        exp226_first_row_idx=("exp226_first_row_idx", "min"),
        exp226_last_row_idx=("exp226_last_row_idx", "max"),
        exp226_first_suffix_offset=("exp226_first_suffix_offset", "min"),
        exp226_last_suffix_offset=("exp226_last_suffix_offset", "max"),
        exp226_squared_error_sum=("exp226_squared_error_sum", "sum"),
        exp226_truth_sum=("exp226_truth_sum", "sum"),
        exp226_truth_min=("exp226_truth_min", "min"),
        exp226_truth_max=("exp226_truth_max", "max"),
    ).reset_index()
    if not blocks["fold_min"].eq(blocks["fold_max"]).all():
        raise ValueError("exp226 fold changes within a block")
    blocks["fold"] = blocks.pop("fold_min").astype(np.int64)
    blocks = blocks.drop(columns="fold_max")
    if not np.array_equal(
        blocks["exp226_rows"].to_numpy(np.int64),
        (
            blocks["exp226_last_suffix_offset"]
            - blocks["exp226_first_suffix_offset"]
            + 1
        ).to_numpy(np.int64),
    ):
        raise ValueError("exp226 suffix offsets are not contiguous within a block")
    by_well = blocks.groupby("well_id", sort=True, observed=True).agg(
        rows=("exp226_rows", "sum"),
        minimum_suffix=("exp226_first_suffix_offset", "min"),
        maximum_suffix=("exp226_last_suffix_offset", "max"),
        fold_count=("fold", "nunique"),
    )
    if (
        not by_well["minimum_suffix"].eq(0).all()
        or not by_well["maximum_suffix"].eq(by_well["rows"] - 1).all()
        or not by_well["fold_count"].eq(1).all()
    ):
        raise ValueError("exp226 well-level suffix/fold contract changed")
    return blocks


def aggregate_exp264_blocks(
    path: Path,
    *,
    prediction_column: str,
    block_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    required = {
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        "md_since",
        prediction_column,
    }
    missing = sorted(required.difference(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"exp264 Stage D OOF missing {missing}")
    partials: list[pd.DataFrame] = []
    first_row_by_well: dict[str, int] = {}
    last_row_by_well: dict[str, int] = {}
    outer_fold_by_well: dict[str, int] = {}
    total_rows = 0
    total_squared_error = 0.0
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=[
            "id",
            "well",
            "outer_fold",
            "actual_tvt",
            "md_since",
            prediction_column,
        ],
    ):
        chunk = batch.to_pandas()
        chunk["id"] = chunk["id"].astype(str)
        chunk["well"] = chunk["well"].astype(str)
        split_id = chunk["id"].str.rsplit("_", n=1, expand=True)
        if not np.array_equal(
            split_id[0].to_numpy(dtype=str), chunk["well"].to_numpy(dtype=str)
        ):
            raise ValueError("exp264 ID prefix differs from well")
        chunk["row_idx"] = pd.to_numeric(split_id[1], errors="raise").astype(np.int64)
        suffix_offset = np.empty(len(chunk), dtype=np.int64)
        for well, positions in chunk.groupby("well", sort=False).indices.items():
            position_array = np.asarray(positions, dtype=np.int64)
            row_index = chunk.iloc[position_array]["row_idx"].to_numpy(np.int64)
            if len(row_index) > 1 and not np.all(np.diff(row_index) == 1):
                raise ValueError(f"exp264 rows are not consecutive within well {well}")
            if well in last_row_by_well and row_index[0] != last_row_by_well[well] + 1:
                raise ValueError(f"exp264 well {well} reappeared out of row order")
            first_row_by_well.setdefault(str(well), int(row_index[0]))
            last_row_by_well[str(well)] = int(row_index[-1])
            folds = chunk.iloc[position_array]["outer_fold"].to_numpy(np.int64)
            if len(np.unique(folds)) != 1:
                raise ValueError(f"exp264 outer fold changes within well {well}")
            previous_fold = outer_fold_by_well.setdefault(str(well), int(folds[0]))
            if previous_fold != int(folds[0]):
                raise ValueError(f"exp264 outer fold changes across batches for {well}")
            suffix_offset[position_array] = row_index - first_row_by_well[str(well)]
        chunk["suffix_offset"] = suffix_offset
        chunk["block_id"] = suffix_offset // block_size
        actual = pd.to_numeric(chunk["actual_tvt"], errors="raise").to_numpy(np.float64)
        prediction = pd.to_numeric(chunk[prediction_column], errors="raise").to_numpy(
            np.float64
        )
        md_since = pd.to_numeric(chunk["md_since"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(np.c_[actual, prediction, md_since]).all():
            raise ValueError("exp264 Stage D OOF contains non-finite values")
        error = prediction - actual
        chunk["exp264_squared_error"] = np.square(error)
        chunk["exp264_bad10"] = np.abs(error) >= 10.0
        grouped = chunk.groupby(["well", "block_id"], sort=False, observed=True)
        partials.append(
            grouped.agg(
                exp264_outer_fold_min=("outer_fold", "min"),
                exp264_outer_fold_max=("outer_fold", "max"),
                exp264_rows=("row_idx", "size"),
                exp264_first_row_idx=("row_idx", "min"),
                exp264_last_row_idx=("row_idx", "max"),
                exp264_first_suffix_offset=("suffix_offset", "min"),
                exp264_last_suffix_offset=("suffix_offset", "max"),
                min_md_since=("md_since", "min"),
                max_md_since=("md_since", "max"),
                exp264_squared_error_sum=("exp264_squared_error", "sum"),
                exp264_bad10_rows=("exp264_bad10", "sum"),
                exp264_truth_sum=("actual_tvt", "sum"),
                exp264_truth_min=("actual_tvt", "min"),
                exp264_truth_max=("actual_tvt", "max"),
            ).reset_index()
        )
        total_rows += len(chunk)
        total_squared_error += float(np.square(error).sum())
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(["well", "block_id"], sort=True, observed=True)
    blocks = grouped.agg(
        exp264_outer_fold_min=("exp264_outer_fold_min", "min"),
        exp264_outer_fold_max=("exp264_outer_fold_max", "max"),
        exp264_rows=("exp264_rows", "sum"),
        exp264_first_row_idx=("exp264_first_row_idx", "min"),
        exp264_last_row_idx=("exp264_last_row_idx", "max"),
        exp264_first_suffix_offset=("exp264_first_suffix_offset", "min"),
        exp264_last_suffix_offset=("exp264_last_suffix_offset", "max"),
        min_md_since=("min_md_since", "min"),
        max_md_since=("max_md_since", "max"),
        exp264_squared_error_sum=("exp264_squared_error_sum", "sum"),
        exp264_bad10_rows=("exp264_bad10_rows", "sum"),
        exp264_truth_sum=("exp264_truth_sum", "sum"),
        exp264_truth_min=("exp264_truth_min", "min"),
        exp264_truth_max=("exp264_truth_max", "max"),
    ).reset_index()
    blocks = blocks.rename(columns={"well": "well_id"})
    if not blocks["exp264_outer_fold_min"].eq(blocks["exp264_outer_fold_max"]).all():
        raise ValueError("exp264 outer fold changes within a final block")
    blocks["exp264_outer_fold"] = blocks.pop("exp264_outer_fold_min").astype(np.int64)
    blocks = blocks.drop(columns="exp264_outer_fold_max")
    if not np.array_equal(
        blocks["exp264_rows"].to_numpy(np.int64),
        (
            blocks["exp264_last_suffix_offset"]
            - blocks["exp264_first_suffix_offset"]
            + 1
        ).to_numpy(np.int64),
    ):
        raise ValueError("exp264 suffix offsets are not contiguous within a block")
    evidence = {
        "rows": total_rows,
        "wells": len(first_row_by_well),
        "outer_folds": sorted(set(outer_fold_by_well.values())),
        "rmse": float(np.sqrt(total_squared_error / total_rows)),
    }
    return blocks, evidence


def load_post_freeze_block_metrics(
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    verify_freeze(freeze, ledger)
    ledger.register_truth_access()
    exp226_spec = get_nested(config, "data.exp226_source")
    exp226_path, exp226_evidence = resolve_file(
        exp226_spec["patterns"],
        label="exp226 OOF",
        expected_decompressed_sha256=str(exp226_spec["expected_decompressed_sha256"]),
    )
    exp264_spec = get_nested(config, "data.exp264_source")
    exp264_path, exp264_evidence = resolve_file(
        exp264_spec["patterns"],
        label="corrected exp264 Stage D OOF",
        expected_file_sha256=str(exp264_spec["expected_sha256"]),
    )

    prediction_column = str(exp264_spec["prediction_column"])
    block_size = int(get_nested(config, "data.block_size"))
    exp226_blocks = aggregate_exp226_blocks(exp226_path, block_size=block_size)
    exp264_blocks, exp264_aggregate_evidence = aggregate_exp264_blocks(
        exp264_path,
        prediction_column=prediction_column,
        block_size=block_size,
    )
    if exp264_aggregate_evidence["outer_folds"] != [0, 1, 2, 3, 4]:
        raise ValueError("exp264 outer-fold provenance changed")
    blocks = exp226_blocks.merge(
        exp264_blocks,
        on=["well_id", "block_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not blocks["_merge"].eq("both").all():
        raise ValueError("exp264/exp226 block identity coverage failed")
    blocks = blocks.drop(columns="_merge")
    identity_checks = {
        "rows": np.array_equal(
            blocks["exp226_rows"].to_numpy(np.int64),
            blocks["exp264_rows"].to_numpy(np.int64),
        ),
        "first_row": np.array_equal(
            blocks["exp226_first_row_idx"].to_numpy(np.int64),
            blocks["exp264_first_row_idx"].to_numpy(np.int64),
        ),
        "last_row": np.array_equal(
            blocks["exp226_last_row_idx"].to_numpy(np.int64),
            blocks["exp264_last_row_idx"].to_numpy(np.int64),
        ),
        "first_suffix": np.array_equal(
            blocks["exp226_first_suffix_offset"].to_numpy(np.int64),
            blocks["exp264_first_suffix_offset"].to_numpy(np.int64),
        ),
        "last_suffix": np.array_equal(
            blocks["exp226_last_suffix_offset"].to_numpy(np.int64),
            blocks["exp264_last_suffix_offset"].to_numpy(np.int64),
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(f"exp264/exp226 block identity differs: {identity_checks}")
    truth_mean_difference = np.abs(
        blocks["exp264_truth_sum"] / blocks["exp264_rows"]
        - blocks["exp226_truth_sum"] / blocks["exp226_rows"]
    )
    truth_atol = float(get_nested(config, "validation.truth_alignment_atol_ft"))
    if float(truth_mean_difference.max()) > truth_atol:
        raise ValueError("exp264/exp226 block truth mean differs")
    blocks["rows"] = blocks["exp264_rows"].astype(np.int64)
    blocks["exp264_block_rmse"] = np.sqrt(
        blocks["exp264_squared_error_sum"] / blocks["rows"]
    )
    blocks["exp226_block_rmse"] = np.sqrt(
        blocks["exp226_squared_error_sum"] / blocks["rows"]
    )
    blocks["exp264_bad10_rate"] = blocks["exp264_bad10_rows"] / blocks["rows"]
    blocks["exp264_block_rmse_ge_10ft"] = blocks["exp264_block_rmse"] >= 10.0
    blocks["exp226_beats_exp264_by_0p25ft"] = (
        blocks["exp226_block_rmse"] + 0.25 <= blocks["exp264_block_rmse"]
    )
    blocks["exp226_benefit_ft"] = (
        blocks["exp264_block_rmse"] - blocks["exp226_block_rmse"]
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        int(blocks["rows"].sum()) != expected_rows
        or blocks["well_id"].nunique() != expected_wells
    ):
        raise ValueError("post-freeze row/well contract changed")
    observed_rmse = float(exp264_aggregate_evidence["rmse"])
    if not np.isclose(
        observed_rmse, float(exp264_spec["expected_rmse"]), rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"exp264 RMSE contract changed: {observed_rmse}")
    return blocks, [
        {"name": "exp226_oof_post_freeze", **exp226_evidence},
        {
            "name": "exp264_corrected_stage_d_v3_post_freeze",
            **exp264_evidence,
            "observed_rmse": observed_rmse,
            "aggregate_evidence": exp264_aggregate_evidence,
            "maximum_block_truth_mean_difference_ft": float(
                truth_mean_difference.max()
            ),
        },
    ]


def load_hidden_like_sets(
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    ledger: TruthAccessLedger,
    valid_wells: set[str],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    verify_freeze(freeze, ledger)
    spec = get_nested(config, "data.hidden_like_assignment")
    path, evidence = resolve_file(
        spec["patterns"],
        label="exp115 hidden-like assignment",
        expected_file_sha256=str(spec["expected_sha256"]),
    )
    frame = pd.read_csv(path, dtype={str(spec["well_column"]): str})
    well_column = str(spec["well_column"])
    required = {well_column, *map(str, spec["role_columns"].values())}
    missing = sorted(required.difference(frame.columns))
    if missing or frame[well_column].duplicated().any():
        raise ValueError(f"hidden-like assignment contract failed: missing={missing}")
    output: dict[str, set[str]] = {}
    for scope, role_column in spec["role_columns"].items():
        selected = set(
            frame.loc[frame[str(role_column)].astype(str).eq("valid"), well_column].astype(str)
        )
        if not selected or not selected.issubset(valid_wells):
            raise ValueError(f"hidden-like scope {scope} has invalid well membership")
        output[str(scope)] = selected
    return output, {"name": "exp115_hidden_like_assignment", **evidence}


# %% [markdown]
# ## 8. Block RMSE, row-weighted AUC, scopes, and fixed gate


# %%
def build_post_freeze_block_readout(
    frozen_features: pd.DataFrame,
    post_freeze_blocks: pd.DataFrame,
    hidden_sets: Mapping[str, set[str]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    readout = frozen_features.merge(
        post_freeze_blocks,
        on=["well_id", "fold", "block_id"],
        how="left",
        validate="one_to_one",
    )
    if readout["rows"].isna().any() or len(readout) != len(frozen_features):
        raise ValueError("frozen block identity does not cover post-freeze block metrics")
    identity_checks = {
        "row_count": np.array_equal(
            readout["block_row_count"].to_numpy(np.int64), readout["rows"].to_numpy(np.int64)
        ),
        "first_row": np.array_equal(
            readout["block_start_row_idx"].to_numpy(np.int64),
            readout["exp264_first_row_idx"].to_numpy(np.int64),
        ),
        "last_row": np.array_equal(
            readout["block_end_row_idx"].to_numpy(np.int64),
            readout["exp264_last_row_idx"].to_numpy(np.int64),
        ),
        "first_suffix": np.array_equal(
            readout["block_start_suffix_offset"].to_numpy(np.int64),
            readout["exp264_first_suffix_offset"].to_numpy(np.int64),
        ),
        "last_suffix": np.array_equal(
            readout["block_end_suffix_offset"].to_numpy(np.int64),
            readout["exp264_last_suffix_offset"].to_numpy(np.int64),
        ),
        "min_md_since": np.allclose(
            readout["md_since_min_ft"].to_numpy(np.float64),
            readout["min_md_since"].to_numpy(np.float64),
            rtol=0.0,
            atol=1e-5,
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(f"exp280/exp264 block identity mismatch: {identity_checks}")
    readout["distance_1000_plus"] = readout["min_md_since"] >= 1000.0
    for scope, wells in hidden_sets.items():
        readout[str(scope)] = readout["well_id"].astype(str).isin(wells)
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    if len(readout) != expected_blocks:
        raise ValueError("post-freeze block count changed")
    return readout


def weighted_block_auc(
    scores: np.ndarray, positive_counts: np.ndarray, negative_counts: np.ndarray
) -> float | None:
    score = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive_counts, dtype=np.float64)
    negative = np.asarray(negative_counts, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(positive) & np.isfinite(negative)
    if not valid.all() or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError("AUC requires aligned finite nonnegative counts")
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    if total_positive <= 0 or total_negative <= 0:
        return None
    work = pd.DataFrame(
        {"score": score, "positive": positive, "negative": negative}
    ).sort_values("score", kind="mergesort")
    tied = work.groupby("score", sort=True, observed=True)[["positive", "negative"]].sum()
    cumulative_negative = tied["negative"].cumsum().shift(fill_value=0.0)
    numerator = np.sum(
        tied["positive"].to_numpy()
        * (cumulative_negative.to_numpy() + 0.5 * tied["negative"].to_numpy())
    )
    return float(numerator / (total_positive * total_negative))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    label = np.asarray(labels, dtype=bool)
    return weighted_block_auc(scores, label.astype(float), (~label).astype(float))


def summarize_family_scope(
    frame: pd.DataFrame,
    *,
    variant: str,
    family: str,
    scope: str,
) -> dict[str, Any]:
    prefix = VARIANT_PREFIX[variant]
    risk = frame[f"{prefix}risk__{family}"].to_numpy(np.float64)
    q1 = frame[f"{prefix}q1__{family}"].to_numpy(bool)
    q4 = frame[f"{prefix}q4__{family}"].to_numpy(bool)
    positive_counts = frame["exp264_bad10_rows"].to_numpy(np.float64)
    negative_counts = frame["rows"].to_numpy(np.float64) - positive_counts
    q1_rmse = frame.loc[q1, "exp264_block_rmse"]
    q4_rmse = frame.loc[q4, "exp264_block_rmse"]
    real_auc = weighted_block_auc(risk, positive_counts, negative_counts)
    q1_rows = float(frame.loc[q1, "rows"].sum())
    q4_rows = float(frame.loc[q4, "rows"].sum())
    q1_bad = float(frame.loc[q1, "exp264_bad10_rows"].sum())
    q4_bad = float(frame.loc[q4, "exp264_bad10_rows"].sum())
    return {
        "variant": variant,
        "family": family,
        "scope": scope,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "rows": int(frame["rows"].sum()),
        "feature_finite_coverage": float(np.isfinite(risk).mean()),
        "q1_blocks": int(q1.sum()),
        "q4_blocks": int(q4.sum()),
        "q1_q4_overlap_blocks": int(np.logical_and(q1, q4).sum()),
        "q1_mean_exp264_block_rmse": float(q1_rmse.mean()) if len(q1_rmse) else np.nan,
        "q4_mean_exp264_block_rmse": float(q4_rmse.mean()) if len(q4_rmse) else np.nan,
        "q4_minus_q1_mean_exp264_block_rmse": (
            float(q4_rmse.mean() - q1_rmse.mean()) if len(q1_rmse) and len(q4_rmse) else np.nan
        ),
        "q1_median_exp264_block_rmse": float(q1_rmse.median()) if len(q1_rmse) else np.nan,
        "q4_median_exp264_block_rmse": float(q4_rmse.median()) if len(q4_rmse) else np.nan,
        "q4_minus_q1_median_exp264_block_rmse": (
            float(q4_rmse.median() - q1_rmse.median())
            if len(q1_rmse) and len(q4_rmse)
            else np.nan
        ),
        "q1_abs_error_ge_10ft_rate": q1_bad / q1_rows if q1_rows else np.nan,
        "q4_abs_error_ge_10ft_rate": q4_bad / q4_rows if q4_rows else np.nan,
        "row_weighted_abs_error_ge_10ft_auc": real_auc,
        "alias_like_failure_auc": binary_auc(
            frame["exp226_beats_exp264_by_0p25ft"].to_numpy(bool), risk
        ),
    }


def build_family_metrics(
    readout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_masks = {
        "pooled": np.ones(len(readout), dtype=bool),
        "distance_1000_plus": readout["distance_1000_plus"].to_numpy(bool),
        "hidden_like_spatial": readout["hidden_like_spatial"].to_numpy(bool),
        "hidden_like_typewell_purged": readout[
            "hidden_like_typewell_purged"
        ].to_numpy(bool),
    }
    scope_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    supported = readout["core_supported"].to_numpy(bool)
    for variant in VARIANTS:
        for family in FAMILIES:
            for scope, mask in scope_masks.items():
                part = readout.loc[supported & mask]
                if part.empty:
                    raise ValueError(f"supported scope {scope} is empty")
                scope_rows.append(
                    summarize_family_scope(
                        part, variant=variant, family=family, scope=scope
                    )
                )
            for fold, part in readout.loc[supported].groupby("fold", sort=True):
                fold_rows.append(
                    {
                        **summarize_family_scope(
                            part,
                            variant=variant,
                            family=family,
                            scope=f"fold_{int(fold)}",
                        ),
                        "fold": int(fold),
                    }
                )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def evaluate_fixed_gate(
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    boundaries: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    guards = get_nested(config, "model.pass_requires_all_for_primary")
    primary_scopes = scope_metrics.loc[
        scope_metrics["variant"].eq("real_zncc")
        & scope_metrics["family"].eq(PRIMARY_FAMILY)
    ].set_index("scope")
    primary_folds = fold_metrics.loc[
        fold_metrics["variant"].eq("real_zncc")
        & fold_metrics["family"].eq(PRIMARY_FAMILY)
    ].sort_values("fold")
    raw_scopes = scope_metrics.loc[
        scope_metrics["variant"].eq("historical_raw_gaussian")
        & scope_metrics["family"].eq(PRIMARY_FAMILY)
    ].set_index("scope")
    raw_folds = fold_metrics.loc[
        fold_metrics["variant"].eq("historical_raw_gaussian")
        & fold_metrics["family"].eq(PRIMARY_FAMILY)
    ].sort_values("fold")
    permutation_scopes = scope_metrics.loc[
        scope_metrics["variant"].eq("stable_permutation")
        & scope_metrics["family"].eq(PRIMARY_FAMILY)
    ].set_index("scope")
    permutation_folds = fold_metrics.loc[
        fold_metrics["variant"].eq("stable_permutation")
        & fold_metrics["family"].eq(PRIMARY_FAMILY)
    ].sort_values("fold")
    pooled = primary_scopes.loc["pooled"]
    pooled_auc = float(pooled["row_weighted_abs_error_ge_10ft_auc"])
    raw_pooled_auc = float(
        raw_scopes.loc["pooled", "row_weighted_abs_error_ge_10ft_auc"]
    )
    permutation_pooled_auc = float(
        permutation_scopes.loc["pooled", "row_weighted_abs_error_ge_10ft_auc"]
    )
    positive_folds = int(
        (primary_folds["q4_minus_q1_mean_exp264_block_rmse"] > 0.0).sum()
    )
    auc_above_half_folds = int(
        (primary_folds["row_weighted_abs_error_ge_10ft_auc"] > 0.5).sum()
    )
    better_than_raw_folds = int(
        (
            primary_folds["row_weighted_abs_error_ge_10ft_auc"].to_numpy(np.float64)
            > raw_folds["row_weighted_abs_error_ge_10ft_auc"].to_numpy(np.float64)
        ).sum()
    )
    better_than_permutation_folds = int(
        (
            primary_folds["row_weighted_abs_error_ge_10ft_auc"].to_numpy(np.float64)
            > permutation_folds[
                "row_weighted_abs_error_ge_10ft_auc"
            ].to_numpy(np.float64)
        ).sum()
    )
    primary_boundaries = boundaries.loc[
        boundaries["variant"].eq("real_zncc")
        & boundaries["family"].eq(PRIMARY_FAMILY)
    ]
    scientific_checks = {
        "primary_quantile_separation_all_folds": bool(
            (
                primary_boundaries["q75_risk_boundary"]
                > primary_boundaries["q25_risk_boundary"]
            ).all()
        ),
        "pooled_q4_minus_q1_mean_rmse": float(
            pooled["q4_minus_q1_mean_exp264_block_rmse"]
        )
        >= float(guards["minimum_q4_minus_q1_mean_rmse_ft"]),
        "pooled_q4_minus_q1_median_positive": float(
            pooled["q4_minus_q1_median_exp264_block_rmse"]
        )
        > 0.0,
        "positive_rmse_folds": positive_folds
        >= int(guards["minimum_positive_rmse_folds"]),
        "pooled_bad10_auc": pooled_auc >= float(guards["minimum_pooled_bad10_auc"]),
        "bad10_auc_above_half_folds": auc_above_half_folds
        >= int(guards["minimum_folds_bad10_auc_gt_half"]),
        "distance_1000_plus_positive": float(
            primary_scopes.loc[
                "distance_1000_plus", "q4_minus_q1_mean_exp264_block_rmse"
            ]
        )
        > 0.0,
        "hidden_like_spatial_positive": float(
            primary_scopes.loc[
                "hidden_like_spatial", "q4_minus_q1_mean_exp264_block_rmse"
            ]
        )
        > 0.0,
        "hidden_like_typewell_purged_positive": float(
            primary_scopes.loc[
                "hidden_like_typewell_purged",
                "q4_minus_q1_mean_exp264_block_rmse",
            ]
        )
        > 0.0,
        "pooled_auc_gain_vs_raw_gaussian": pooled_auc - raw_pooled_auc
        >= float(guards["minimum_pooled_auc_gain_vs_raw_gaussian"]),
        "folds_better_than_raw_gaussian": better_than_raw_folds
        >= int(guards["minimum_folds_better_than_raw_gaussian"]),
        "pooled_auc_gain_vs_permutation": pooled_auc - permutation_pooled_auc
        >= float(guards["minimum_pooled_auc_gain_vs_permutation"]),
        "folds_better_than_permutation": better_than_permutation_folds
        >= int(guards["minimum_folds_better_than_permutation"]),
    }
    technical_passed = bool(freeze_manifest["technical_passed"])
    scientific_passed = bool(all(scientific_checks.values()))
    gate = {
        "family": PRIMARY_FAMILY,
        "technical_passed": technical_passed,
        "scientific_checks_passed": scientific_passed,
        "passed": technical_passed and scientific_passed,
        "technical_checks": dict(freeze_manifest["technical_checks"]),
        "scientific_checks": scientific_checks,
        "positive_rmse_folds": positive_folds,
        "bad10_auc_above_half_folds": auc_above_half_folds,
        "folds_better_than_raw_gaussian": better_than_raw_folds,
        "folds_better_than_permutation": better_than_permutation_folds,
        "pooled_bad10_auc": pooled_auc,
        "raw_gaussian_pooled_bad10_auc": raw_pooled_auc,
        "permutation_pooled_bad10_auc": permutation_pooled_auc,
        "pooled_auc_gain_vs_raw_gaussian": pooled_auc - raw_pooled_auc,
        "pooled_auc_gain_vs_permutation": pooled_auc - permutation_pooled_auc,
    }
    decision = {
        "technical_passed": technical_passed,
        "scientific_passed": scientific_passed,
        "passed": bool(gate["passed"]),
        "promotable_family": PRIMARY_FAMILY,
        "action": (
            "propose_separate_addonly_ml_feature_experiment_no_prediction_change"
            if gate["passed"]
            else "close_zncc_confidence_branch_without_rescue"
        ),
    }
    return gate, decision


# %% [markdown]
# ## 9. Metrics and generated artifacts


# %%
def save_final_artifacts(
    readout: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    primary_gate: Mapping[str, Any],
    freeze: Mapping[str, Any],
    post_freeze_evidence: list[dict[str, Any]],
    decision: Mapping[str, Any],
    started_at: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = artifact_dir()
    readout_path = artifacts / f"{OUTPUT_PREFIX}_post_freeze_block_readout.csv.gz"
    scope_path = artifacts / f"{OUTPUT_PREFIX}_family_scope_metrics.csv"
    fold_path = artifacts / f"{OUTPUT_PREFIX}_family_fold_metrics.csv"
    gate_path = artifacts / f"{OUTPUT_PREFIX}_primary_gate.json"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    sha_path = artifacts / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    readout_evidence = write_csv_gzip(readout, readout_path)
    scope_metrics.to_csv(scope_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    write_json(gate_path, primary_gate)
    pooled = scope_metrics.loc[scope_metrics["scope"].eq("pooled")].to_dict(orient="records")
    sentinel_descriptive: list[dict[str, Any]] = []
    for well_id in map(
        str, get_nested(config, "validation.descriptive_sentinel_wells") or ()
    ):
        part = readout.loc[
            readout["well_id"].astype(str).eq(well_id) & readout["core_supported"]
        ]
        if part.empty:
            sentinel_descriptive.append(
                {"well_id": well_id, "supported_blocks": 0}
            )
            continue
        sentinel_descriptive.append(
            {
                "well_id": well_id,
                "supported_blocks": len(part),
                "primary_risk_mean": float(
                    part[f"risk__{PRIMARY_FAMILY}"].mean()
                ),
                "exp264_block_rmse_mean": float(part["exp264_block_rmse"].mean()),
                "exp264_bad10_row_rate": float(
                    part["exp264_bad10_rows"].sum() / part["rows"].sum()
                ),
                "top1_shift_ft_median": float(part["top1_shift_ft"].median()),
                "gate_usage": "descriptive_only",
            }
        )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0_completed_guard_passed"
            if decision["passed"]
            else "stage_0_completed_guard_failed"
        ),
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started_at,
        "rows": int(get_nested(config, "validation.expected_rows")),
        "wells": int(readout["well_id"].nunique()),
        "blocks": len(readout),
        "readout_families": len(FAMILIES),
        "score_variants": list(VARIANTS),
        "controls": 2,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "parent_control_retraining": False,
        "decision": dict(decision),
        "primary_gate": dict(primary_gate),
        "pooled_family_metrics": pooled,
        "sentinel_descriptive_only": sentinel_descriptive,
        "freeze": freeze["manifest"],
        "post_freeze_inputs": post_freeze_evidence,
        "artifacts": {
            "post_freeze_block_readout": readout_evidence,
            "family_scope_metrics": str(scope_path),
            "family_fold_metrics": str(fold_path),
            "primary_gate": str(gate_path),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(summary_path, summary)
    output_paths = {
        "contract": freeze["contract_path"],
        "input_manifest": freeze["input_path"],
        "score_schema": freeze["schema_path"],
        "target_free_zncc_scores": freeze["score_path"],
        "valid_masks": freeze["valid_mask_path"],
        "target_free_features": freeze["feature_path"],
        "fold_quantile_boundaries": freeze["quantile_path"],
        "freeze_manifest": freeze["manifest_path"],
        "post_freeze_block_readout": readout_path,
        "family_scope_metrics": scope_path,
        "family_fold_metrics": fold_path,
        "primary_gate": gate_path,
        "summary": summary_path,
    }
    sha_frame = pd.DataFrame(
        [
            {"name": name, "path": str(path), "sha256": sha256_path(path)}
            for name, path in output_paths.items()
        ]
    )
    sha_frame.to_csv(sha_path, index=False)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "ensemble",
        "stage": "stage_0",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "diagnostic": {
            "technical_passed": decision["technical_passed"],
            "scientific_passed": decision["scientific_passed"],
            "passed": decision["passed"],
            "primary_gate": dict(primary_gate),
            "score_content_sha256": freeze["manifest"]["score_content_sha256"],
            "valid_mask_content_sha256": freeze["manifest"][
                "valid_mask_content_sha256"
            ],
            "feature_content_sha256": freeze["manifest"]["feature_content_sha256"],
            "quantile_content_sha256": freeze["manifest"]["quantile_content_sha256"],
        },
        "notes": (
            "Zero-booster readout only; no prediction, inference, or submission was generated."
        ),
    }
    write_json(metrics_output_path(), metrics)
    return summary


def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started_at = time.time()
    ledger = TruthAccessLedger()
    raw_scores, pre_freeze_evidence = load_exp280_target_free_scores(config)
    pre_freeze_evidence.extend(load_lineage_contracts(config))
    safe_oof, _, exp226_safe_evidence = load_exp226_safe(config)
    pre_freeze_evidence.append(exp226_safe_evidence)
    train_root = resolve_train_root(config)
    raw_wells = sorted(
        path.name.removesuffix("__horizontal_well.csv")
        for path in train_root.glob("*__horizontal_well.csv")
    )
    expected_wells = sorted(safe_oof["well_id"].astype(str).unique().tolist())
    if raw_wells != expected_wells:
        raise ValueError("raw train well set differs from exp226 safe OOF")
    score_parts: list[pd.DataFrame] = []
    well_manifest_rows: list[dict[str, Any]] = []
    grouped_oof = safe_oof.groupby("well_id", sort=True, observed=True)
    for index, (well_id, well_oof) in enumerate(grouped_oof, start=1):
        well_id = str(well_id)
        horizontal_path = train_root / f"{well_id}__horizontal_well.csv"
        typewell_path = train_root / f"{well_id}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(typewell_path)
        horizontal_safe = load_horizontal_safe(horizontal_path)
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        well_scores, well_manifest = score_well_target_free_zncc(
            well_oof, horizontal_safe, typewell, config
        )
        well_manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        score_parts.append(well_scores)
        well_manifest_rows.append(well_manifest)
        if index % 25 == 0 or index == len(raw_wells):
            print(f"target-free ZNCC wells={index}/{len(raw_wells)}")
    zncc_scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    if len(zncc_scores) != expected_blocks * len(EXPECTED_SHIFTS):
        raise ValueError("ZNCC score bank row count changed")
    identity_columns = [
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "shift_slot",
        "shift_ft",
    ]
    if not zncc_scores[identity_columns].equals(raw_scores[identity_columns]):
        raise ValueError("ZNCC and exp280 matched-control block/shift identity differs")
    well_manifest = pd.DataFrame(well_manifest_rows).sort_values(
        "well_id", kind="mergesort"
    )
    pre_freeze_evidence.append(
        {
            "name": "raw_train_horizontal_and_typewell_files",
            "path": str(train_root),
            "wells": len(well_manifest),
            "horizontal_rows": int(well_manifest["horizontal_rows"].sum()),
            "content_sha256": dataframe_content_sha(
                well_manifest,
                ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
            ),
        }
    )
    features = build_target_free_block_features(zncc_scores, raw_scores)
    expected_well_count = int(get_nested(config, "validation.expected_wells"))
    if (
        len(features) != expected_blocks
        or features["well_id"].nunique() != expected_well_count
    ):
        raise ValueError("target-free feature block/well contract changed")
    boundaries = fit_fold_quantile_boundaries(features, config)
    features = attach_frozen_quartile_flags(features, boundaries)
    freeze = freeze_target_free_bundle(
        zncc_scores, features, boundaries, pre_freeze_evidence, config, ledger
    )
    post_freeze_blocks, post_freeze_evidence = load_post_freeze_block_metrics(
        config, freeze, ledger
    )
    hidden_sets, hidden_evidence = load_hidden_like_sets(
        config,
        freeze,
        ledger,
        set(features["well_id"].astype(str)),
    )
    post_freeze_evidence.append(hidden_evidence)
    readout = build_post_freeze_block_readout(
        features, post_freeze_blocks, hidden_sets, config
    )
    scope_metrics, fold_metrics = build_family_metrics(readout)
    primary_gate, decision = evaluate_fixed_gate(
        scope_metrics, fold_metrics, boundaries, freeze["manifest"], config
    )
    summary = save_final_artifacts(
        readout,
        scope_metrics,
        fold_metrics,
        primary_gate,
        freeze,
        post_freeze_evidence,
        decision,
        started_at,
        config,
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG)
SETUP_PREVIEW = {
    "experiment": get_nested(CONFIG, "experiment.name"),
    "route": get_nested(CONFIG, "experiment.route"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "score": get_nested(CONFIG, "model.score.name"),
    "historical_control": get_nested(CONFIG, "data.exp280_source.experiment"),
    "readout_families": list(FAMILIES),
    "primary_family": PRIMARY_FAMILY,
    "variants": list(VARIANTS),
    "folds": get_nested(CONFIG, "validation.expected_folds"),
    "expected_blocks": get_nested(CONFIG, "validation.expected_blocks"),
    "active_stage": get_nested(CONFIG, "execution.active_stage"),
    "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
    "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
    "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
    "model_configs": get_nested(CONFIG, "execution_contract.model_configs"),
    "trained_folds": get_nested(CONFIG, "execution_contract.trained_folds"),
    "boosters": get_nested(CONFIG, "execution_contract.boosters"),
    "hmm_well_runs": get_nested(CONFIG, "execution_contract.pf_beam_hmm_runs"),
}
print(json.dumps(to_jsonable(SETUP_PREVIEW), indent=2, sort_keys=True))


# %% [markdown]
# ## 11. Run the approved Stage 0 readout only
#
# The repository implementation intentionally leaves Kaggle push/run disabled.
# A later explicit approval must set all three execution switches before this
# cell can consume mounted parent outputs.


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_stage_0_experiment(CONFIG)
    POOLED_PREVIEW = pd.DataFrame(SUMMARY["pooled_family_metrics"])[
        [
            "variant",
            "family",
            "q4_minus_q1_mean_exp264_block_rmse",
            "row_weighted_abs_error_ge_10ft_auc",
        ]
    ]
    display(POOLED_PREVIEW)
