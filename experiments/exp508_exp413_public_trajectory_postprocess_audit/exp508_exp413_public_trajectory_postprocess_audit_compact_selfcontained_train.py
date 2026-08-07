# %% [markdown]
# # exp508 exp413 public trajectory postprocess audit — Stage A candidate
#
# This notebook candidate performs one deterministic, saved-OOF-only audit.
# The selectable treatment is exactly one per-well Savitzky--Golay 61/3 pass
# over the final exp413 TVT prediction. Tau-85 warmup variants are scored only
# after the primary decision is frozen. No model, selector, PF, HMM, Beam,
# router, parameter search, inference, or submission path is present.

# %% [markdown]
# ## Contents
# 1. Imports and immutable boundary
# 2. Notebook-safe paths, serialization, and hashes
# 3. Static contract and authorization guard
# 4. Frozen evidence and truth-free input checks
# 5. Truth-free candidate generation and prediction freeze
# 6. Truth-late primary metrics and all-AND gate
# 7. Report-only warmup readout
# 8. Stage A orchestration and generated artifacts
# 9. Setup and fixed stop

# %% [markdown]
# ## 1. Imports and immutable boundary

# %%
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
import yaml
from scipy.signal import savgol_filter

EXPERIMENT_NAME = "exp508_exp413_public_trajectory_postprocess_audit"
PARENT_EXPERIMENT = "exp413_scale5_likpf_full_replacement_on_exp335"
PREDICTION_COLUMN = "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
CONTROL_COLUMN = "raw_exp413_stage_d_oof"
PRIMARY_COLUMN = "sg61_p3_final_tvt"
WARMUP_COLUMN = "tau85_warmup_final_delta"
WARMUP_SG_COLUMN = "tau85_warmup_then_sg61_p3"
TRUTH_FREE_OOF_COLUMNS = (
    "id",
    "well",
    "md_since",
    "last_known_tvt",
    "outer_fold",
    PREDICTION_COLUMN,
)
TRUTH_LATE_OOF_COLUMNS = ("id", "target", "actual_tvt")
PREDICTION_FREEZE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "fold",
    "md_since",
    CONTROL_COLUMN,
    PRIMARY_COLUMN,
    WARMUP_COLUMN,
    WARMUP_SG_COLUMN,
)
REPORT_ONLY_COLUMNS = (WARMUP_COLUMN, WARMUP_SG_COLUMN)
FIXED_SCOPE_ORDER = (
    "md_0_250",
    "md_250_1000",
    "md_1000_plus",
    "hidden_like_spatial",
    "hidden_like_typewell_purged",
)
HIDDEN_ROLE_COLUMNS = {
    "hidden_like_spatial": "verification_like_spatial_role",
    "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
}
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe paths, serialization, and hashes

# %%
def locate_experiment_dir(start: Path = PACKAGE_DIR) -> Path:
    candidates = [
        start,
        start / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    for candidate in candidates:
        if (candidate / "config.yaml").is_file() and (
            candidate / "postprocess_contract.yaml"
        ).is_file():
            return candidate
    for candidate in (start, *start.parents):
        path = candidate / "experiments" / EXPERIMENT_NAME
        if (path / "config.yaml").is_file():
            return path
    raise FileNotFoundError(f"Could not locate {EXPERIMENT_NAME}")


EXPERIMENT_DIR = locate_experiment_dir()


def find_project_root(start: Path = EXPERIMENT_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


ROOT = find_project_root()


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def output_dir() -> Path:
    if is_kaggle_runtime():
        return KAGGLE_WORKING_ROOT / "artifacts"
    return EXPERIMENT_DIR / "artifacts"


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML must contain a mapping: {path}")
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame, chunk_rows: int = 100_000) -> str:
    normalized = frame.copy()
    logical_dtypes: list[str] = []
    for column in normalized.columns:
        if pd.api.types.is_object_dtype(normalized[column]) or isinstance(
            normalized[column].dtype, pd.StringDtype
        ):
            normalized[column] = normalized[column].astype(str)
            logical_dtypes.append("string")
        else:
            logical_dtypes.append(str(normalized[column].dtype))
    digest = hashlib.sha256()
    digest.update("\x1f".join(normalized.columns.astype(str)).encode("utf-8"))
    digest.update("\x1f".join(logical_dtypes).encode("utf-8"))
    for start in range(0, len(normalized), int(chunk_rows)):
        chunk = normalized.iloc[start : start + int(chunk_rows)]
        hashes = pd.util.hash_pandas_object(chunk, index=False, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def array_sha256(values: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(sha256_json({"shape": list(array.shape), "dtype": dtype}).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def write_yaml(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(to_jsonable(value), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return sha256_file(path)


def search_roots() -> list[Path]:
    return [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT, Path.cwd()]


def _expand_pattern(pattern: str) -> list[Path]:
    raw = Path(pattern)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.extend(Path(value) for value in glob.glob(pattern, recursive=True))
    else:
        for root in search_roots():
            candidates.extend(
                Path(value) for value in glob.glob(str(root / pattern), recursive=True)
            )
    return candidates


def resolve_sha_qualified_file(
    patterns: Sequence[str],
    expected_sha256: str,
    *,
    explicit_env: str | None = None,
) -> Path:
    ordered_patterns: list[str] = []
    if explicit_env and os.environ.get(explicit_env):
        ordered_patterns.append(str(os.environ[explicit_env]))
    ordered_patterns.extend(str(value) for value in patterns)
    mismatches: dict[str, str] = {}
    seen: set[str] = set()
    for pattern in ordered_patterns:
        for candidate in _expand_pattern(pattern):
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            observed = sha256_file(candidate)
            if observed == str(expected_sha256):
                return candidate
            mismatches[key] = observed
    raise FileNotFoundError(
        f"No SHA-qualified file found for {ordered_patterns}; mismatches={mismatches}"
    )


def resolve_parent_evidence_root(config: Mapping[str, Any]) -> Path:
    spec = config["data"]["exp413_oof"]
    files = dict(spec["evidence_files"])
    expected = {
        "oof": str(spec["expected_oof_sha256"]),
        "fold_metrics": str(spec["expected_fold_metrics_sha256"]),
        "scope_metrics": str(spec["expected_scope_metrics_sha256"]),
        "hidden_like_metrics": str(spec["expected_hidden_like_metrics_sha256"]),
        "by_well": str(spec["expected_by_well_sha256"]),
    }
    mismatches: list[dict[str, Any]] = []
    for pattern in spec["root_patterns"]:
        for candidate in _expand_pattern(str(pattern)):
            if not candidate.is_dir():
                continue
            paths = {label: candidate / str(filename) for label, filename in files.items()}
            if not all(path.is_file() for path in paths.values()):
                continue
            observed = {label: sha256_file(path) for label, path in paths.items()}
            if observed == expected:
                return candidate
            mismatches.append({"root": str(candidate), "sha256": observed})
    raise FileNotFoundError(f"No complete SHA-qualified exp413 Stage D root: {mismatches}")


# %% [markdown]
# ## 3. Static contract and authorization guard

# %%
def validate_static_contract(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if config["experiment"]["name"] != EXPERIMENT_NAME:
        raise ValueError("experiment name changed")
    if config["experiment"]["route"] != "ml_model":
        raise ValueError("exp508 route must remain ml_model")
    if config["lineage"]["parent"] != PARENT_EXPERIMENT:
        raise ValueError("exp508 parent changed")
    if not bool(config["authorization"]["implementation_approved"]):
        raise ValueError("implementation approval is not recorded")
    required_run_approvals = (
        "canonical_notebook_adoption_approved",
        "kaggle_package_approved",
        "kaggle_run_approved",
    )
    if not all(bool(config["authorization"][key]) for key in required_run_approvals):
        raise ValueError("canonical notebook, package, and Stage A run approvals are required")

    primary = config["postprocess"]["primary"]
    expected_primary = {
        "id": PRIMARY_COLUMN,
        "selectable": True,
        "window_length": 61,
        "polyorder": 3,
        "mode": "scipy_default_interp",
        "dtype": "float64",
        "reanchor_after_filter": False,
        "clip_after_filter": False,
        "project_after_filter": False,
    }
    observed_primary = {key: primary[key] for key in expected_primary}
    if observed_primary != expected_primary:
        raise ValueError(f"selectable primary changed: {observed_primary}")
    report_ids = tuple(item["id"] for item in config["postprocess"]["report_only"])
    if report_ids != REPORT_ONLY_COLUMNS or any(
        bool(item["selectable"]) or bool(item["may_rescue_primary"])
        for item in config["postprocess"]["report_only"]
    ):
        raise ValueError("report-only warmup contract changed")
    if contract["selectable_primary"]["id"] != PRIMARY_COLUMN:
        raise ValueError("postprocess contract primary changed")
    if int(contract["selectable_primary"]["window_length"]) != 61 or int(
        contract["selectable_primary"]["polyorder"]
    ) != 3:
        raise ValueError("postprocess contract SG parameters changed")

    cost = config["execution_contract"]
    observed_cost = {
        "scientific_primary_variants": int(cost["scientific_primary_variants"]),
        "report_only_controls": int(cost["report_only_controls"]),
        "trained_models": int(cost["trained_models"]),
        "lightgbm_configs": int(cost["lightgbm_configs"]),
        "total_boosters": int(cost["total_boosters"]),
        "hmm_runs": int(cost["hmm_runs"]),
        "pf_runs": int(cost["pf_runs"]),
        "beam_runs": int(cost["beam_runs"]),
        "parent_or_control_retraining": int(cost["parent_or_control_retraining"]),
        "gpu_runs": int(cost["gpu_runs"]),
    }
    expected_cost = {
        "scientific_primary_variants": 1,
        "report_only_controls": 2,
        "trained_models": 0,
        "lightgbm_configs": 0,
        "total_boosters": 0,
        "hmm_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "parent_or_control_retraining": 0,
        "gpu_runs": 0,
    }
    if observed_cost != expected_cost:
        raise ValueError(f"Stage A compute contract changed: {observed_cost}")

    gate = config["promotion_gate"]
    expected_gate = {
        "pooled_gain_min_ft": 0.01,
        "nonworse_folds_required": 4,
        "fixed_scope_max_delta_ft": 0.02,
        "by_well_p95_max_delta_ft": 0.25,
        "by_well_worst_max_delta_ft": 0.25,
        "first_score_row_abs_correction_p95_max_ft": 0.50,
        "first_score_row_abs_correction_max_ft": 2.00,
    }
    observed_gate = {key: gate[key] for key in expected_gate}
    if observed_gate != expected_gate or tuple(gate["fixed_scopes"]) != FIXED_SCOPE_ORDER:
        raise ValueError("promotion gate changed")
    if gate["fail_decision"] != "FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE":
        raise ValueError("fixed fail decision changed")

    forbidden = set(config["postprocess"]["prohibited"])
    required_forbidden = {
        "public_model60_likpf40_blend",
        "direct_likpf_reblend",
        "sg_window_or_polyorder_grid",
        "tau_grid",
        "row_or_well_gate",
        "public_fixed_well_shape_threshold_or_map",
        "same_oof_rescue",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("prohibited rescue contract changed")
    return {
        "status": "pass",
        "cost": observed_cost,
        "primary": observed_primary,
        "report_only": list(report_ids),
        "gate": observed_gate,
        "forbidden": sorted(forbidden),
        "contract_logical_sha256": sha256_json(contract),
    }


def require_stage_a_authorization(config: Mapping[str, Any]) -> None:
    if not is_kaggle_runtime():
        raise RuntimeError("exp508 Stage A must run on Kaggle private CPU")
    authorization = config["authorization"]
    execution = config["execution"]["stage_a"]
    if not bool(authorization["kaggle_run_approved"]):
        raise RuntimeError("exp508 Kaggle Stage A run is not approved")
    if not bool(execution["run_approved"]) or not bool(execution["enabled"]):
        raise RuntimeError("exp508 Stage A remains disabled")


def verify_public_source_and_contract(
    config: Mapping[str, Any], experiment_dir: Path = EXPERIMENT_DIR
) -> dict[str, Any]:
    source_spec = config["public_source_contract"]
    contract_path = experiment_dir / "postprocess_contract.yaml"
    observed_contract_sha = sha256_file(contract_path)
    if observed_contract_sha != str(source_spec["postprocess_contract_sha256"]):
        raise ValueError("postprocess_contract.yaml SHA mismatch")
    source_path = ROOT / str(config["lineage"]["public_source"])
    source_evidence: dict[str, Any] = {
        "path": str(source_path),
        "expected_sha256": str(source_spec["expected_sha256"]),
        "available_in_runtime": source_path.is_file(),
    }
    if source_path.is_file():
        observed_source_sha = sha256_file(source_path)
        if observed_source_sha != str(source_spec["expected_sha256"]):
            raise ValueError("public source SHA mismatch")
        text = source_path.read_text(encoding="utf-8")
        required_snippets = (
            "def make_prediction(df, model_delta, likpf):",
            "sg_win = 61",
            "sg_poly = 3",
            "tau = 85.0",
            "savgol_filter(v, wl, PP.sg_poly)",
        )
        missing = [snippet for snippet in required_snippets if snippet not in text]
        if missing:
            raise ValueError(f"public source required snippets missing: {missing}")
        source_evidence["observed_sha256"] = observed_source_sha
        source_evidence["required_snippets_passed"] = True
    return {
        "source": source_evidence,
        "postprocess_contract": {
            "path": str(contract_path),
            "sha256": observed_contract_sha,
        },
    }


# %% [markdown]
# ## 4. Frozen evidence and truth-free input checks

# %%
def verify_parent_evidence(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    spec = config["data"]["exp413_oof"]
    files = dict(spec["evidence_files"])
    expected = {
        "oof": str(spec["expected_oof_sha256"]),
        "fold_metrics": str(spec["expected_fold_metrics_sha256"]),
        "scope_metrics": str(spec["expected_scope_metrics_sha256"]),
        "hidden_like_metrics": str(spec["expected_hidden_like_metrics_sha256"]),
        "by_well": str(spec["expected_by_well_sha256"]),
    }
    evidence: dict[str, dict[str, Any]] = {}
    for label, filename in files.items():
        path = root / str(filename)
        observed = sha256_file(path)
        if observed != expected[label]:
            raise ValueError(f"exp413 {label} SHA mismatch")
        evidence[label] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return evidence


def _parse_row_idx(ids: pd.Series, wells: pd.Series) -> np.ndarray:
    split = ids.astype(str).str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2 or not split[0].astype(str).eq(wells.astype(str)).all():
        raise ValueError("exp413 id/well logical-key contract changed")
    return pd.to_numeric(split[1], errors="raise").to_numpy(np.int64)


def load_truth_free_parent_oof(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_wells: int,
    expected_folds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed_sha = sha256_file(path)
    if observed_sha != str(expected_sha256):
        raise ValueError("exp413 Stage D OOF SHA mismatch")
    available_columns = tuple(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(TRUTH_FREE_OOF_COLUMNS) - set(available_columns))
    if missing:
        raise ValueError(f"exp413 truth-free OOF columns missing: {missing}")
    frame = pd.read_parquet(path, columns=list(TRUTH_FREE_OOF_COLUMNS))
    if tuple(frame.columns) != TRUTH_FREE_OOF_COLUMNS:
        raise ValueError("truth-free OOF loader column order changed")
    frame = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "well": frame["well"].astype(str),
            "row_idx": _parse_row_idx(frame["id"], frame["well"]),
            "fold": pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy(np.int8),
            "md_since": pd.to_numeric(frame["md_since"], errors="raise").to_numpy(
                np.float64
            ),
            "last_known_tvt": pd.to_numeric(
                frame["last_known_tvt"], errors="raise"
            ).to_numpy(np.float64),
            CONTROL_COLUMN: pd.to_numeric(
                frame[PREDICTION_COLUMN], errors="raise"
            ).to_numpy(np.float64),
        }
    )
    if len(frame) != int(expected_rows):
        raise ValueError(f"exp413 row count mismatch: {len(frame)} != {expected_rows}")
    if int(frame["well"].nunique()) != int(expected_wells):
        raise ValueError("exp413 well count mismatch")
    if frame["id"].duplicated().any() or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("exp413 logical keys are duplicated")
    if sorted(int(value) for value in frame["fold"].unique()) != list(expected_folds):
        raise ValueError("exp413 fold inventory changed")
    if not frame.groupby("well", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("an exp413 well spans multiple folds")
    numeric = frame[["md_since", "last_known_tvt", CONTROL_COLUMN]].to_numpy(np.float64)
    if not np.isfinite(numeric).all() or bool((frame["md_since"] < 0.0).any()):
        raise ValueError("exp413 truth-free numeric contract failed")
    key_frame = frame[["id", "well", "row_idx"]]
    per_well_order = frame[["well", "row_idx"]]
    fold_frame = (
        frame[["well", "fold"]]
        .drop_duplicates()
        .sort_values("well", kind="stable")
        .reset_index(drop=True)
    )
    manifest = {
        "source_path": str(path),
        "source_file_sha256": observed_sha,
        "source_schema": list(available_columns),
        "loaded_columns": list(TRUTH_FREE_OOF_COLUMNS),
        "truth_or_error_columns_loaded": 0,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "logical_key_sha256": logical_frame_sha256(key_frame),
        "global_row_order_sha256": logical_frame_sha256(frame[["id"]]),
        "per_well_row_order_sha256": logical_frame_sha256(per_well_order),
        "fold_assignment_logical_sha256": logical_frame_sha256(fold_frame),
    }
    return frame, manifest


def resolve_fold_manifest(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = config["data"]["fold_manifest"]
    path = resolve_sha_qualified_file(
        spec["patterns"], str(spec["expected_sha256"]), explicit_env="EXP508_FOLD_MANIFEST"
    )
    return path, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(pd.read_csv(path))),
    }


def resolve_hidden_assignment(config: Mapping[str, Any]) -> Path:
    spec = config["data"]["hidden_like_assignment"]
    return resolve_sha_qualified_file(
        spec["patterns"], str(spec["expected_sha256"]), explicit_env="EXP508_HIDDEN_ASSIGNMENT"
    )


# %% [markdown]
# ## 5. Truth-free candidate generation and prediction freeze

# %%
def savgol_by_well(
    wells: pd.Series | Sequence[str],
    values: np.ndarray | Sequence[float],
    *,
    window_length: int = 61,
    polyorder: int = 3,
) -> tuple[np.ndarray, pd.DataFrame]:
    if int(window_length) != 61 or int(polyorder) != 3:
        raise ValueError("exp508 permits only the frozen SG61/p3 primary")
    well_series = pd.Series(wells, dtype=str).reset_index(drop=True)
    source = np.asarray(values, dtype=np.float64)
    if len(well_series) != len(source) or not np.isfinite(source).all():
        raise ValueError("invalid per-well SG input")
    output = source.copy()
    rows: list[dict[str, Any]] = []
    for well, positions in well_series.groupby(well_series, sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        vector = source[pos]
        effective_window = min(int(window_length), len(vector))
        if effective_window % 2 == 0:
            effective_window -= 1
        applied = effective_window >= int(polyorder) + 2
        if applied:
            output[pos] = savgol_filter(vector, effective_window, int(polyorder))
        rows.append(
            {
                "well": str(well),
                "rows": len(vector),
                "effective_window": effective_window,
                "filter_applied": bool(applied),
            }
        )
    if not np.isfinite(output).all():
        raise ValueError("SG primary contains non-finite values")
    return output, pd.DataFrame(rows)


def tau85_warmup(
    md_since: np.ndarray | Sequence[float],
    last_known_tvt: np.ndarray | Sequence[float],
    prediction: np.ndarray | Sequence[float],
    *,
    tau_ft: float = 85.0,
) -> np.ndarray:
    if float(tau_ft) != 85.0:
        raise ValueError("exp508 permits only the frozen report-only tau=85")
    md = np.asarray(md_since, dtype=np.float64)
    last = np.asarray(last_known_tvt, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if not (len(md) == len(last) == len(pred)) or not np.isfinite(
        np.column_stack([md, last, pred])
    ).all():
        raise ValueError("invalid tau85 warmup input")
    factor = 1.0 - np.exp(-np.maximum(md, 0.0) / 85.0)
    return last + factor * (pred - last)


def generate_truth_free_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    control = frame[CONTROL_COLUMN].to_numpy(np.float64)
    primary, short_well_audit = savgol_by_well(frame["well"], control)
    warmup = tau85_warmup(
        frame["md_since"], frame["last_known_tvt"], control, tau_ft=85.0
    )
    warmup_sg, warmup_short_well_audit = savgol_by_well(frame["well"], warmup)
    if not short_well_audit.equals(warmup_short_well_audit):
        raise ValueError("SG grouping changed between primary and report-only paths")
    predictions = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "well": frame["well"].astype(str),
            "row_idx": frame["row_idx"].to_numpy(np.int64),
            "fold": frame["fold"].to_numpy(np.int8),
            "md_since": frame["md_since"].to_numpy(np.float64),
            CONTROL_COLUMN: control,
            PRIMARY_COLUMN: primary,
            WARMUP_COLUMN: warmup,
            WARMUP_SG_COLUMN: warmup_sg,
        }
    )
    if tuple(predictions.columns) != PREDICTION_FREEZE_COLUMNS:
        raise ValueError("prediction freeze schema changed")
    if not np.isfinite(
        predictions[[CONTROL_COLUMN, PRIMARY_COLUMN, *REPORT_ONLY_COLUMNS]].to_numpy()
    ).all():
        raise ValueError("truth-free prediction contains non-finite values")
    manifest = {
        "status": "truth_free_predictions_frozen",
        "schema": list(predictions.columns),
        "truth_or_error_columns": 0,
        "rows": len(predictions),
        "wells": int(predictions["well"].nunique()),
        "prediction_content_sha256": {
            column: array_sha256(predictions[column].to_numpy(np.float64))
            for column in (CONTROL_COLUMN, PRIMARY_COLUMN, *REPORT_ONLY_COLUMNS)
        },
        "full_logical_sha256": logical_frame_sha256(predictions),
        "control_primary_equal_rows": int(
            np.equal(
                predictions[CONTROL_COLUMN].to_numpy(),
                predictions[PRIMARY_COLUMN].to_numpy(),
            ).sum()
        ),
        "short_well_contract": {
            "wells": len(short_well_audit),
            "filter_applied_wells": int(short_well_audit["filter_applied"].sum()),
            "unchanged_short_wells": int((~short_well_audit["filter_applied"]).sum()),
        },
    }
    return predictions, manifest, short_well_audit


def freeze_truth_free_predictions(
    predictions: pd.DataFrame, destination: Path
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination, index=False, compression="zstd")
    reloaded = pd.read_parquet(destination, columns=list(PREDICTION_FREEZE_COLUMNS))
    if logical_frame_sha256(reloaded) != logical_frame_sha256(predictions):
        raise ValueError("truth-free prediction parquet round-trip changed content")
    return {
        "path": str(destination),
        "file_sha256": sha256_file(destination),
        "logical_sha256": logical_frame_sha256(predictions),
        "truth_or_error_columns": 0,
    }


# %% [markdown]
# ## 6. Truth-late primary metrics and all-AND gate
#
# The functions below are called only after the truth-free parquet and its
# prediction manifest have been written. Candidate parameters and predictions
# are immutable at this point.

# %%
def load_truth_late(
    path: Path,
    frozen: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    truth_frame = pd.read_parquet(path, columns=list(TRUTH_LATE_OOF_COLUMNS))
    if tuple(truth_frame.columns) != TRUTH_LATE_OOF_COLUMNS:
        raise ValueError("truth-late loader column order changed")
    if not truth_frame["id"].astype(str).reset_index(drop=True).equals(
        frozen["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("truth-late OOF row order differs from prediction freeze")
    target = pd.to_numeric(truth_frame["target"], errors="raise").to_numpy(np.float64)
    actual = pd.to_numeric(truth_frame["actual_tvt"], errors="raise").to_numpy(np.float64)
    if not np.isfinite(np.column_stack([target, actual])).all():
        raise ValueError("truth-late target contains non-finite values")
    reconstructed = frozen["last_known_tvt"].to_numpy(np.float64) + target
    max_abs = float(np.max(np.abs(reconstructed - actual), initial=0.0))
    if max_abs > 1.0e-4:
        raise ValueError(f"actual_tvt differs from last_known_tvt + target: {max_abs}")
    return actual, {
        "loaded_after_prediction_freeze": True,
        "rows": len(actual),
        "actual_target_parity_max_abs_ft": max_abs,
        "actual_tvt_sha256": array_sha256(actual),
    }


def load_hidden_assignment_late(
    path: Path, predictions: pd.DataFrame, expected_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = sha256_file(path)
    if observed != str(expected_sha256):
        raise ValueError("hidden-like assignment SHA mismatch")
    assignment = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *HIDDEN_ROLE_COLUMNS.values()}
    missing = sorted(required - set(assignment.columns))
    if missing or assignment["well_id"].duplicated().any():
        raise ValueError(f"hidden-like assignment contract failed: {missing}")
    prediction_wells = set(predictions["well"].astype(str))
    assignment_wells = set(assignment["well_id"].astype(str))
    if not prediction_wells.issubset(assignment_wells):
        raise ValueError("hidden-like assignment does not cover exp508 wells")
    return assignment, {
        "path": str(path),
        "sha256": observed,
        "rows": len(assignment),
        "loaded_after_prediction_freeze": True,
    }


def rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def _score_pair(
    truth: np.ndarray, control: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    if not bool(mask.any()):
        raise ValueError("exp508 metric scope is empty")
    control_rmse = rmse(truth[mask], control[mask])
    candidate_rmse = rmse(truth[mask], candidate[mask])
    return control_rmse, candidate_rmse, candidate_rmse - control_rmse


def build_trajectory_diagnostics(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    all_control_second: list[np.ndarray] = []
    all_primary_second: list[np.ndarray] = []
    for well, positions in predictions.groupby("well", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        control = predictions.iloc[pos][CONTROL_COLUMN].to_numpy(np.float64)
        primary = predictions.iloc[pos][PRIMARY_COLUMN].to_numpy(np.float64)
        correction = np.abs(primary - control)
        control_second = np.diff(control, n=2)
        primary_second = np.diff(primary, n=2)
        all_control_second.append(control_second)
        all_primary_second.append(primary_second)
        rows.append(
            {
                "well": str(well),
                "rows": len(pos),
                "correction_abs_mean_ft": float(correction.mean()),
                "correction_abs_max_ft": float(correction.max(initial=0.0)),
                "control_second_difference_rms_ft": (
                    float(np.sqrt(np.mean(np.square(control_second))))
                    if len(control_second)
                    else 0.0
                ),
                "primary_second_difference_rms_ft": (
                    float(np.sqrt(np.mean(np.square(primary_second))))
                    if len(primary_second)
                    else 0.0
                ),
            }
        )
    correction = np.abs(
        predictions[PRIMARY_COLUMN].to_numpy(np.float64)
        - predictions[CONTROL_COLUMN].to_numpy(np.float64)
    )
    control_second_all = np.concatenate(all_control_second)
    primary_second_all = np.concatenate(all_primary_second)
    summary = {
        "correction_abs_ft": {
            "mean": float(correction.mean()),
            "p50": float(np.quantile(correction, 0.50)),
            "p90": float(np.quantile(correction, 0.90)),
            "p95": float(np.quantile(correction, 0.95)),
            "p99": float(np.quantile(correction, 0.99)),
            "max": float(correction.max(initial=0.0)),
        },
        "second_difference_rms_ft": {
            "control": float(np.sqrt(np.mean(np.square(control_second_all)))),
            "primary": float(np.sqrt(np.mean(np.square(primary_second_all)))),
        },
    }
    return summary, pd.DataFrame(rows)


def evaluate_primary(
    predictions: pd.DataFrame,
    truth: np.ndarray,
    assignment: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    control = predictions[CONTROL_COLUMN].to_numpy(np.float64)
    primary = predictions[PRIMARY_COLUMN].to_numpy(np.float64)
    fold = predictions["fold"].to_numpy(np.int8)
    md_since = predictions["md_since"].to_numpy(np.float64)
    pooled_control = rmse(truth, control)
    pooled_primary = rmse(truth, primary)

    fold_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        mask = fold == outer_fold
        parent, candidate, delta = _score_pair(truth, control, primary, mask)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "rows": int(mask.sum()),
                "exp413_rmse": parent,
                "sg61_p3_rmse": candidate,
                "delta_sg_minus_exp413": delta,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)

    scope_masks: dict[str, np.ndarray] = {
        "md_0_250": md_since <= 250.0,
        "md_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "md_1000_plus": md_since >= 1000.0,
    }
    assignment_by_well = assignment.set_index("well_id")
    for scope, role_column in HIDDEN_ROLE_COLUMNS.items():
        scope_masks[scope] = (
            predictions["well"]
            .astype(str)
            .map(assignment_by_well[role_column])
            .eq("valid")
            .to_numpy()
        )
    scope_rows: list[dict[str, Any]] = []
    for scope in FIXED_SCOPE_ORDER:
        mask = scope_masks[scope]
        parent, candidate, delta = _score_pair(truth, control, primary, mask)
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(predictions.loc[mask, "well"].nunique()),
                "exp413_rmse": parent,
                "sg61_p3_rmse": candidate,
                "delta_sg_minus_exp413": delta,
            }
        )
    scope_metrics = pd.DataFrame(scope_rows)

    joined = predictions[["well"]].copy()
    joined["truth"] = truth
    joined["control"] = control
    joined["primary"] = primary
    by_well_rows: list[dict[str, Any]] = []
    for well, part in joined.groupby("well", sort=True):
        parent = rmse(part["truth"], part["control"])
        candidate = rmse(part["truth"], part["primary"])
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(part),
                "exp413_rmse": parent,
                "sg61_p3_rmse": candidate,
                "delta_sg_minus_exp413": candidate - parent,
            }
        )
    by_well = pd.DataFrame(by_well_rows)

    first_positions = np.asarray(
        [positions[0] for positions in predictions.groupby("well", sort=False).indices.values()],
        dtype=np.int64,
    )
    first_rows = predictions.iloc[first_positions][["well", "id", "row_idx"]].copy()
    first_rows["exp413_prediction"] = control[first_positions]
    first_rows["sg61_p3_prediction"] = primary[first_positions]
    first_rows["abs_correction_ft"] = np.abs(primary[first_positions] - control[first_positions])
    first_rows.reset_index(drop=True, inplace=True)

    trajectory_summary, trajectory_by_well = build_trajectory_diagnostics(predictions)
    delta = by_well["delta_sg_minus_exp413"]
    first_correction = first_rows["abs_correction_ft"]
    summary = {
        "pooled": {
            "rows": len(predictions),
            "wells": int(predictions["well"].nunique()),
            "exp413_rmse": pooled_control,
            "sg61_p3_rmse": pooled_primary,
            "gain_ft": pooled_control - pooled_primary,
            "delta_sg_minus_exp413": pooled_primary - pooled_control,
        },
        "tail": {
            "delta_median_ft": float(delta.median()),
            "delta_p90_ft": float(delta.quantile(0.90)),
            "delta_p95_ft": float(delta.quantile(0.95)),
            "delta_p99_ft": float(delta.quantile(0.99)),
            "worst_well": str(by_well.loc[delta.idxmax(), "well"]),
            "worst_delta_ft": float(delta.max()),
            "worsened_wells_plus_0p25ft": int((delta > 0.25).sum()),
            "worsened_wells_plus_1ft": int((delta > 1.0).sum()),
            "worsened_wells_plus_3ft": int((delta > 3.0).sum()),
            "worsened_wells_plus_5ft": int((delta > 5.0).sum()),
        },
        "prediction_start": {
            "wells": len(first_rows),
            "abs_correction_p50_ft": float(first_correction.quantile(0.50)),
            "abs_correction_p90_ft": float(first_correction.quantile(0.90)),
            "abs_correction_p95_ft": float(first_correction.quantile(0.95)),
            "abs_correction_p99_ft": float(first_correction.quantile(0.99)),
            "abs_correction_max_ft": float(first_correction.max()),
        },
        "trajectory": trajectory_summary,
    }
    return summary, {
        "fold": fold_metrics,
        "scope": scope_metrics,
        "by_well": by_well,
        "prediction_start": first_rows,
        "trajectory_by_well": trajectory_by_well,
    }


def build_primary_gate(
    summary: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    technical_checks: Mapping[str, bool],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["promotion_gate"]
    fold_metrics = tables["fold"]
    scope_metrics = tables["scope"].set_index("scope")
    nonworse_folds = int((fold_metrics["delta_sg_minus_exp413"] <= 0.0).sum())
    maximum_scope_delta = float(
        scope_metrics.loc[list(FIXED_SCOPE_ORDER), "delta_sg_minus_exp413"].max()
    )
    tail = summary["tail"]
    prediction_start = summary["prediction_start"]
    checks = {
        "technical_all_pass": bool(technical_checks) and bool(all(technical_checks.values())),
        "pooled_gain_at_least_0p01_ft": float(summary["pooled"]["gain_ft"])
        >= float(gate["pooled_gain_min_ft"]),
        "nonworse_folds_at_least_4_of_5": nonworse_folds
        >= int(gate["nonworse_folds_required"]),
        "all_fixed_scopes_within_plus_0p02_ft": maximum_scope_delta
        <= float(gate["fixed_scope_max_delta_ft"]),
        "by_well_p95_within_plus_0p25_ft": float(tail["delta_p95_ft"])
        <= float(gate["by_well_p95_max_delta_ft"]),
        "worst_well_within_plus_0p25_ft": float(tail["worst_delta_ft"])
        <= float(gate["by_well_worst_max_delta_ft"]),
        "first_score_row_p95_within_0p50_ft": float(
            prediction_start["abs_correction_p95_ft"]
        )
        <= float(gate["first_score_row_abs_correction_p95_max_ft"]),
        "first_score_row_max_within_2p00_ft": float(
            prediction_start["abs_correction_max_ft"]
        )
        <= float(gate["first_score_row_abs_correction_max_ft"]),
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "decision": (
            "PASS_QUALIFY_SAME_EXP_INFERENCE_IMPLEMENTATION_FOR_SEPARATE_APPROVAL"
            if passed
            else str(gate["fail_decision"])
        ),
        "checks": checks,
        "technical_checks": dict(technical_checks),
        "gain_ft": float(summary["pooled"]["gain_ft"]),
        "nonworse_folds": nonworse_folds,
        "maximum_fixed_scope_delta_ft": maximum_scope_delta,
        "by_well_delta_p95_ft": float(tail["delta_p95_ft"]),
        "worst_well_delta_ft": float(tail["worst_delta_ft"]),
        "first_score_row_abs_correction_p95_ft": float(
            prediction_start["abs_correction_p95_ft"]
        ),
        "first_score_row_abs_correction_max_ft": float(
            prediction_start["abs_correction_max_ft"]
        ),
        "report_only_may_rescue_primary": False,
        "inference_automatically_approved": False,
    }


# %% [markdown]
# ## 7. Report-only warmup readout
#
# This readout receives the already-frozen primary decision SHA. Its values are
# descriptive only and cannot change the primary gate or create a new candidate.

# %%
def score_report_only(
    predictions: pd.DataFrame,
    truth: np.ndarray,
    *,
    primary_decision_sha256: str,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    folds = predictions["fold"].to_numpy(np.int8)
    for column in REPORT_ONLY_COLUMNS:
        candidate = predictions[column].to_numpy(np.float64)
        fold_metrics = []
        for outer_fold in range(5):
            mask = folds == outer_fold
            fold_metrics.append(
                {
                    "outer_fold": outer_fold,
                    "rows": int(mask.sum()),
                    "rmse": rmse(truth[mask], candidate[mask]),
                }
            )
        rows[column] = {
            "selectable": False,
            "may_rescue_primary": False,
            "pooled_rmse": rmse(truth, candidate),
            "fold_metrics": fold_metrics,
        }
    return {
        "status": "report_only_scored_after_primary_decision_freeze",
        "primary_decision_sha256": primary_decision_sha256,
        "candidates": rows,
        "selection_or_rescue_performed": False,
    }


# %% [markdown]
# ## 8. Stage A orchestration and generated artifacts

# %%
def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def run_stage_a(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    require_stage_a_authorization(config)
    artifacts_dir = output_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    source_contract_evidence = verify_public_source_and_contract(config)
    parent_root = resolve_parent_evidence_root(config)
    parent_evidence = verify_parent_evidence(parent_root, config)
    fold_manifest_path, fold_manifest_evidence = resolve_fold_manifest(config)
    del fold_manifest_path
    oof_path = parent_root / str(config["data"]["exp413_oof"]["evidence_files"]["oof"])
    frozen_input, input_manifest = load_truth_free_parent_oof(
        oof_path,
        expected_sha256=str(config["data"]["exp413_oof"]["expected_oof_sha256"]),
        expected_rows=int(config["validation"]["expected_rows"]),
        expected_wells=int(config["validation"]["expected_wells"]),
        expected_folds=[int(value) for value in config["validation"]["expected_folds"]],
    )
    input_manifest.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "environment": runtime_versions(),
            "parent_evidence": parent_evidence,
            "fold_manifest": fold_manifest_evidence,
            "source_contract": source_contract_evidence,
        }
    )
    write_json(artifacts_dir / "input_manifest.json", input_manifest)
    row_order_manifest = {
        key: input_manifest[key]
        for key in (
            "rows",
            "wells",
            "folds",
            "logical_key_sha256",
            "global_row_order_sha256",
            "per_well_row_order_sha256",
            "fold_assignment_logical_sha256",
        )
    }
    row_order_manifest["fold_manifest_file_sha256"] = fold_manifest_evidence["sha256"]
    write_json(artifacts_dir / "row_order_manifest.json", row_order_manifest)
    write_yaml(artifacts_dir / "postprocess_contract_resolved.yaml", contract)

    predictions, prediction_manifest, short_well_audit = generate_truth_free_predictions(
        frozen_input
    )
    prediction_file_evidence = freeze_truth_free_predictions(
        predictions, artifacts_dir / "trajectory_postprocess_predictions.parquet"
    )
    prediction_manifest.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "file": prediction_file_evidence,
            "contract_file_sha256": source_contract_evidence["postprocess_contract"][
                "sha256"
            ],
            "truth_attached_at_freeze": False,
            "primary_decision_computed_at_freeze": False,
        }
    )
    write_json(artifacts_dir / "prediction_manifest.json", prediction_manifest)
    short_well_audit.to_csv(artifacts_dir / "short_well_audit.csv", index=False)

    truth, truth_manifest = load_truth_late(oof_path, frozen_input)
    hidden_path = resolve_hidden_assignment(config)
    assignment, hidden_manifest = load_hidden_assignment_late(
        hidden_path,
        predictions,
        str(config["data"]["hidden_like_assignment"]["expected_sha256"]),
    )
    observed_control_rmse = rmse(truth, predictions[CONTROL_COLUMN])
    expected_control_rmse = float(config["data"]["exp413_oof"]["expected_cv"])
    if abs(observed_control_rmse - expected_control_rmse) > 1.0e-9:
        raise ValueError(
            f"exp413 saved control RMSE changed: {observed_control_rmse} != {expected_control_rmse}"
        )
    technical_checks = {
        "static_contract": STATIC_CONTRACT["status"] == "pass",
        "parent_evidence_sha": True,
        "fold_manifest_sha": fold_manifest_evidence["sha256"]
        == str(config["validation"]["fold_manifest_sha256"]),
        "truth_free_input_allowlist": input_manifest["truth_or_error_columns_loaded"] == 0,
        "key_and_row_order_frozen": all(
            bool(row_order_manifest[key])
            for key in (
                "logical_key_sha256",
                "global_row_order_sha256",
                "per_well_row_order_sha256",
                "fold_assignment_logical_sha256",
            )
        ),
        "prediction_freeze_before_truth": bool(
            truth_manifest["loaded_after_prediction_freeze"]
            and hidden_manifest["loaded_after_prediction_freeze"]
        ),
        "prediction_truth_columns_absent": prediction_manifest["truth_or_error_columns"] == 0,
        "prediction_finite": True,
        "control_rmse_parity": abs(observed_control_rmse - expected_control_rmse) <= 1.0e-9,
        "actual_target_parity": truth_manifest["actual_target_parity_max_abs_ft"] <= 1.0e-4,
        "no_model_or_physics_rerun": all(
            int(STATIC_CONTRACT["cost"][key]) == 0
            for key in (
                "trained_models",
                "total_boosters",
                "hmm_runs",
                "pf_runs",
                "beam_runs",
                "parent_or_control_retraining",
                "gpu_runs",
            )
        ),
    }
    if not all(technical_checks.values()):
        raise RuntimeError(f"exp508 technical checks failed: {technical_checks}")

    primary_summary, primary_tables = evaluate_primary(predictions, truth, assignment)
    gate = build_primary_gate(primary_summary, primary_tables, technical_checks, config)
    primary_tables["fold"].to_csv(artifacts_dir / "primary_fold_metrics.csv", index=False)
    primary_tables["scope"].to_csv(artifacts_dir / "primary_scope_metrics.csv", index=False)
    primary_tables["by_well"].to_csv(
        artifacts_dir / "primary_by_well_metrics.csv", index=False
    )
    primary_tables["prediction_start"].to_csv(
        artifacts_dir / "prediction_start_continuity.csv", index=False
    )
    primary_tables["trajectory_by_well"].to_csv(
        artifacts_dir / "trajectory_smoothness_by_well.csv", index=False
    )
    write_json(artifacts_dir / "trajectory_diagnostics.json", primary_summary["trajectory"])
    gate_sha = write_json(artifacts_dir / "promotion_gate.json", gate)
    decision_freeze = {
        "status": "primary_decision_frozen_before_report_only_scoring",
        "decision": gate["decision"],
        "promotion_gate_file_sha256": gate_sha,
        "primary_prediction_content_sha256": prediction_manifest[
            "prediction_content_sha256"
        ][PRIMARY_COLUMN],
        "report_only_scored": False,
    }
    decision_freeze_sha = write_json(
        artifacts_dir / "primary_decision_freeze.json", decision_freeze
    )

    report_only = score_report_only(
        predictions, truth, primary_decision_sha256=decision_freeze_sha
    )
    write_json(artifacts_dir / "report_only_metrics.json", report_only)
    artifact_names = (
        "input_manifest.json",
        "row_order_manifest.json",
        "postprocess_contract_resolved.yaml",
        "trajectory_postprocess_predictions.parquet",
        "prediction_manifest.json",
        "short_well_audit.csv",
        "primary_fold_metrics.csv",
        "primary_scope_metrics.csv",
        "primary_by_well_metrics.csv",
        "prediction_start_continuity.csv",
        "trajectory_smoothness_by_well.csv",
        "trajectory_diagnostics.json",
        "promotion_gate.json",
        "primary_decision_freeze.json",
        "report_only_metrics.json",
    )
    reproducibility = {
        "status": "stage_a_complete",
        "created_at": datetime.now(UTC).isoformat(),
        "seed_policy": "no_rng_fixed_source_row_order_float64",
        "environment": runtime_versions(),
        "input": input_manifest,
        "truth_late": truth_manifest,
        "hidden_assignment": hidden_manifest,
        "technical_checks": technical_checks,
        "artifact_sha256": {
            name: sha256_file(artifacts_dir / name) for name in artifact_names
        },
        "deterministic_anchor": False,
        "rerun_prediction_sha_match_required": True,
    }
    reproducibility_sha = write_json(
        artifacts_dir / "reproducibility_manifest.json", reproducibility
    )
    metrics = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_NAME,
        "status": "stage_a_complete_pass" if gate["passed"] else "stage_a_complete_fail",
        "route": "ml_model",
        "cost": STATIC_CONTRACT["cost"],
        "primary": primary_summary,
        "promotion_gate": gate,
        "report_only": report_only,
        "reproducibility_manifest_sha256": reproducibility_sha,
        "inference_implemented": False,
        "submission_created": False,
    }
    metrics_path = KAGGLE_WORKING_ROOT / "metrics.json"
    write_json(metrics_path, metrics)
    print(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False))
    return metrics


# %% [markdown]
# ## 9. Setup and fixed stop

# %%
CONFIG = read_yaml(EXPERIMENT_DIR / "config.yaml")
POSTPROCESS_CONTRACT = read_yaml(EXPERIMENT_DIR / "postprocess_contract.yaml")
STATIC_CONTRACT = validate_static_contract(CONFIG, POSTPROCESS_CONTRACT)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "parent": CONFIG["lineage"]["parent"],
            "selectable_primary": PRIMARY_COLUMN,
            "report_only": list(REPORT_ONLY_COLUMNS),
            "cost": STATIC_CONTRACT["cost"],
            "kaggle_run_approved": CONFIG["authorization"]["kaggle_run_approved"],
            "stage_a_enabled": CONFIG["execution"]["stage_a"]["enabled"],
        },
        indent=2,
    )
)

if os.environ.get("EXP508_IMPORT_ONLY", "0") == "1":
    print("EXP508_IMPORT_ONLY=1: function and contract import completed; Stage A not run.")
else:
    STAGE_A_RESULT = run_stage_a(CONFIG, POSTPROCESS_CONTRACT)
