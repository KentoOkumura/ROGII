# %% [markdown]
# # exp506 exp490 mean-reversion correction blend on exp413 — Stage A
#
# Audit one preregistered additive correction:
#
# `exp413 + lambda * (exp490 - exp357)`
#
# The scalar is fit on the other four outer folds, clipped to `[0, 0.10]`,
# and applied to the held fold.  This candidate notebook contains the complete
# zero-model Stage A implementation.  A Kaggle run remains disabled until a
# separate approval changes only the run authorization in `config.yaml`.

# %% [markdown]
# ## Contents
# 1. Imports and immutable boundary
# 2. Notebook-safe runtime, serialization, and path helpers
# 3. Frozen contract and anchor resolution
# 4. Truth-free anchor and correction loaders
# 5. Meta-fold scalar fit helpers
# 6. Primary metrics, scope, tail, and gate helpers
# 7. Report-only control and reproducibility helpers
# 8. Setup and fixed execution inventory
# 9. Resolve inputs and freeze truth-free correction
# 10. Attach truth and freeze cross-fitted primary prediction
# 11. Evaluate primary, freeze gate, then compute report-only control
# 12. Generated artifacts and fixed stop

# %% [markdown]
# ## 1. Imports and immutable boundary

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp506_exp490_mean_reversion_correction_blend_on_exp413"
ANCHOR_PREDICTION_COLUMN = "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
EXP490_PREDICTION_COLUMN = "geometry_mean_reverting_hmm"
EXP357_PREDICTION_COLUMN = "exp357_parent_prediction"
EXP490_INPUT_ALLOWLIST = (
    "well",
    "row_idx",
    "suffix_offset",
    "md_since",
    EXP490_PREDICTION_COLUMN,
    EXP357_PREDICTION_COLUMN,
)
ANCHOR_TRUTH_FREE_COLUMNS = (
    "id",
    "well",
    "md_since",
    "outer_fold",
    ANCHOR_PREDICTION_COLUMN,
)
ANCHOR_TRUTH_COLUMNS = ("id", "well", "outer_fold", "actual_tvt")
FIXED_FOLDS = (0, 1, 2, 3, 4)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP506_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)

# %% [markdown]
# ## 2. Notebook-safe runtime, serialization, and path helpers

# %%
def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd()


def find_support_file(filename: str) -> Path:
    candidates = (Path.cwd() / filename, experiment_dir() / filename)
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(path for path in Path.cwd().rglob(filename) if path.is_file())
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{filename} did not resolve uniquely: {matches}")


def runtime_output_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if in_notebook_runtime() and KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def search_roots() -> list[Path]:
    return [Path.cwd(), project_root(), KAGGLE_INPUT_ROOT, Path("/tmp/kaggle-output")]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(dict(payload)), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_gzip_payload(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame, chunk_rows: int = 100_000) -> str:
    digest = hashlib.sha256()
    descriptor = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": len(frame),
    }
    digest.update(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
    for start in range(0, len(frame), chunk_rows):
        payload = frame.iloc[start : start + chunk_rows].to_csv(
            index=False,
            header=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def prediction_sha256(
    frame: pd.DataFrame, prediction_column: str, namespace: str
) -> str:
    ordered = frame.sort_values(["well", "row_idx"], kind="mergesort")
    payload = ordered[["well", "row_idx", prediction_column]].copy()
    return sha256_json(
        {
            "namespace": namespace,
            "logical_frame_sha256": logical_frame_sha256(payload),
        }
    )


def _contains_glob(raw: str) -> bool:
    return any(token in raw for token in ("*", "?", "["))


def _expand_pattern(raw: str, roots: Sequence[Path]) -> list[Path]:
    path = Path(raw)
    if not _contains_glob(raw):
        return [path] if path.exists() else []
    matches: list[Path] = []
    if path.is_absolute():
        matches.extend(Path(path.anchor).glob(str(path)[len(path.anchor) :]))
    else:
        for root in roots:
            if root.exists():
                matches.extend(root.glob(raw))
    return matches


def resolve_file_by_sha(
    patterns: Sequence[str],
    *,
    expected_sha256: str,
    filename_if_directory: str | None = None,
    roots: Sequence[Path] | None = None,
) -> Path:
    roots = list(roots or search_roots())
    candidates: list[Path] = []
    for raw in patterns:
        for match in _expand_pattern(str(raw), roots):
            candidate = (
                match / filename_if_directory
                if match.is_dir() and filename_if_directory is not None
                else match
            )
            if candidate.is_file():
                candidates.append(candidate)
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"no input matches patterns={list(patterns)}")
    observed: dict[str, str] = {}
    for candidate in candidates:
        digest = sha256_file(candidate)
        observed[str(candidate)] = digest
        if digest == str(expected_sha256):
            return candidate
    raise ValueError(
        f"no candidate matched expected SHA {expected_sha256}; observed={observed}"
    )


CONFIG_PATH = find_support_file("config.yaml")
CONTRACT_PATH = find_support_file("ensemble_contract.yaml")
CONFIG = read_yaml(CONFIG_PATH)
CONTRACT = read_yaml(CONTRACT_PATH)
OUTPUT_DIR = runtime_output_dir()
RUN_STAGE_A = EXECUTE_NOTEBOOK and bool(
    CONFIG.get("implementation", {}).get("kaggle_run_approved", False)
)

# %% [markdown]
# ## 3. Frozen contract and anchor resolution
#
# exp497 Stage E failed its own preregistered promotion gate.  The resolution
# rule therefore selects the saved exp413 Stage D OOF before any exp506 truth,
# error, scope, or tail outcome is read.

# %%
def validate_exp506_contract(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    experiment = dict(config["experiment"])
    if experiment.get("route") != "ensemble":
        raise ValueError("exp506 route must remain ensemble")
    implementation = dict(config["implementation"])
    if not bool(implementation.get("approved")):
        raise ValueError("exp506 implementation is not approved")
    resolution = dict(config["prerequisites"]["anchor_resolution"])
    if resolution.get("resolved_anchor") != "exp413_saved_stage_d_oof":
        raise ValueError("exp506 resolved anchor must be exp413 Stage D OOF")
    terminal = dict(config["data"]["exp497_conditional_anchor"])
    if terminal.get("terminal_status") != "completed_gate_failed_closed":
        raise ValueError("exp497 terminal status is not frozen")
    if bool(terminal.get("promotion_gate_passed")):
        raise ValueError("exp497 gate must be recorded as failed")
    if terminal.get("selected_prediction") != "exp413_oof":
        raise ValueError("exp497 selected prediction must be exp413_oof")
    primary = dict(contract["primary"])
    if primary.get("prediction") != "anchor_plus_lambda_times_correction":
        raise ValueError("exp506 primary formula changed")
    if bool(primary.get("intercept")):
        raise ValueError("exp506 primary must not contain an intercept")
    if tuple(float(item) for item in primary.get("lambda_bounds", [])) != (0.0, 0.10):
        raise ValueError("exp506 lambda bounds changed")
    if primary.get("fit") != "other_four_outer_folds":
        raise ValueError("exp506 meta-fold separation changed")
    control = dict(contract["report_only_control"])
    if bool(control.get("selectable")) or bool(control.get("may_rescue_primary")):
        raise ValueError("report-only control became selectable")
    execution = dict(config["execution_contract"])
    zero_keys = (
        "trained_models",
        "total_boosters",
        "hmm_runs",
        "pf_runs",
        "beam_runs",
        "parent_or_control_retraining",
        "gpu_runs",
    )
    if any(int(execution.get(key, -1)) != 0 for key in zero_keys):
        raise ValueError("exp506 Stage A execution inventory changed")
    if int(execution.get("scientific_primary_variants", -1)) != 1:
        raise ValueError("exp506 must contain exactly one scientific primary")
    if int(execution.get("report_only_controls", -1)) != 1:
        raise ValueError("exp506 must contain exactly one report-only control")


def resolved_anchor_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_exp506_contract(config, CONTRACT)
    exp497 = dict(config["data"]["exp497_conditional_anchor"])
    exp413 = dict(config["data"]["exp413_root_anchor"])
    return {
        "resolved_before_exp506_outcome": True,
        "decision_source": "exp497_own_preregistered_all_and_gate_only",
        "exp497": {
            "source_experiment": exp497["source_experiment"],
            "kernel_version": int(exp497["stage_e_kernel_version"]),
            "terminal_status": exp497["terminal_status"],
            "promotion_gate_passed": bool(exp497["promotion_gate_passed"]),
            "selected_prediction": exp497["selected_prediction"],
            "candidate_cv": float(exp497["candidate_cv"]),
            "selected_cv": float(exp497["selected_cv"]),
            "promotion_gate_sha256": exp497["promotion_gate_sha256"],
            "reproducibility_manifest_sha256": exp497[
                "reproducibility_manifest_sha256"
            ],
            "selected_oof_logical_sha256": exp497[
                "selected_oof_logical_sha256"
            ],
        },
        "selected_anchor": {
            "id": "exp413_saved_stage_d_oof",
            "source_kernel_id": exp413["source_kernel_id"],
            "source_kernel_version": int(exp413["source_kernel_version"]),
            "file": exp413["oof_file"],
            "prediction_column": exp413["prediction_column"],
            "expected_cv": float(exp413["expected_cv"]),
            "expected_oof_sha256": exp413["expected_oof_sha256"],
        },
        "exp506_outcome_dependent_anchor_selection_allowed": False,
    }


validate_exp506_contract(CONFIG, CONTRACT)
ANCHOR_RESOLUTION = resolved_anchor_manifest(CONFIG)

# %% [markdown]
# ## 4. Truth-free anchor and correction loaders
#
# The anchor Parquet is first read without `actual_tvt`.  The exp490 gzip is
# read through the exact six-column allowlist, excluding upstream fold, truth,
# error, episode, role, scope, by-well, and gate outcomes.  Keys, fold, anchor,
# exp490, exp357, and their correction are SHA-frozen before truth is attached.

# %%
def parse_ids(ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = ids.astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("anchor IDs must have '<well>_<row_idx>' form")
    well = parts.iloc[:, 0].astype(str)
    row_idx = pd.to_numeric(parts.iloc[:, 1], errors="raise").astype(np.int64)
    return well, row_idx


def load_anchor_without_truth(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_wells: int,
    expected_cv: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed_sha = sha256_file(path)
    if observed_sha != str(expected_sha256):
        raise ValueError("exp413 anchor OOF SHA mismatch")
    frame = pd.read_parquet(path, columns=list(ANCHOR_TRUTH_FREE_COLUMNS))
    if tuple(frame.columns) != ANCHOR_TRUTH_FREE_COLUMNS:
        raise ValueError("anchor truth-free column order changed")
    id_well, row_idx = parse_ids(frame["id"])
    frame["well"] = frame["well"].astype(str)
    if not np.array_equal(id_well.to_numpy(), frame["well"].to_numpy()):
        raise ValueError("anchor ID/well parity failed")
    frame["row_idx"] = row_idx
    if frame.duplicated(["well", "row_idx"]).any() or frame["id"].duplicated().any():
        raise ValueError("anchor contains duplicate keys")
    frame["outer_fold"] = pd.to_numeric(
        frame["outer_fold"], errors="raise"
    ).astype(np.int8)
    if set(frame["outer_fold"].unique()) != set(FIXED_FOLDS):
        raise ValueError("anchor outer-fold set changed")
    if frame.groupby("well", sort=False)["outer_fold"].nunique().max() != 1:
        raise ValueError("anchor assigns a well to multiple outer folds")
    numeric = frame[["md_since", ANCHOR_PREDICTION_COLUMN]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("anchor truth-free values contain NaN/Inf")
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    frame["suffix_offset"] = (
        frame.groupby("well", sort=False).cumcount().astype(np.int64)
    )
    if len(frame) != int(expected_rows) or frame["well"].nunique() != int(expected_wells):
        raise ValueError("anchor row/well coverage mismatch")
    frame = frame.rename(columns={ANCHOR_PREDICTION_COLUMN: "anchor_prediction"})
    manifest = {
        "path": str(path),
        "file_sha256": observed_sha,
        "loaded_columns": list(ANCHOR_TRUTH_FREE_COLUMNS),
        "truth_column_loaded": False,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "folds": sorted(int(item) for item in frame["outer_fold"].unique()),
        "expected_cv": float(expected_cv),
        "logical_sha256": logical_frame_sha256(
            frame[
                [
                    "id",
                    "well",
                    "row_idx",
                    "suffix_offset",
                    "md_since",
                    "outer_fold",
                    "anchor_prediction",
                ]
            ]
        ),
    }
    return frame, manifest


def verify_anchor_evidence_files(
    anchor_path: Path, anchor_spec: Mapping[str, Any]
) -> dict[str, Any]:
    contracts = {
        "fold_metrics": (
            "stage_d_fold_metrics.csv",
            "expected_fold_metrics_sha256",
        ),
        "scope_metrics": (
            "stage_d_scope_metrics.csv",
            "expected_scope_metrics_sha256",
        ),
        "hidden_like_metrics": (
            "stage_d_hidden_like_metrics.csv",
            "expected_hidden_like_metrics_sha256",
        ),
        "by_well_metrics": (
            "stage_d_by_well.csv",
            "expected_by_well_sha256",
        ),
    }
    evidence: dict[str, Any] = {}
    for name, (filename, sha_key) in contracts.items():
        path = anchor_path.parent / filename
        if not path.is_file():
            raise FileNotFoundError(f"exp413 anchor evidence missing: {path}")
        observed = sha256_file(path)
        expected = str(anchor_spec[sha_key])
        if observed != expected:
            raise ValueError(f"exp413 {name} SHA mismatch")
        evidence[name] = {
            "path": str(path),
            "file_sha256": observed,
        }
    return evidence


def load_correction_without_truth(
    path: Path,
    *,
    expected_raw_gzip_sha256: str,
    expected_decompressed_sha256: str,
    expected_rows: int,
    expected_wells: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_sha = sha256_file(path)
    payload_sha = sha256_gzip_payload(path)
    if raw_sha != str(expected_raw_gzip_sha256):
        raise ValueError("exp490 raw gzip SHA mismatch")
    if payload_sha != str(expected_decompressed_sha256):
        raise ValueError("exp490 decompressed payload SHA mismatch")
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = sorted(set(EXP490_INPUT_ALLOWLIST) - set(header))
    if missing:
        raise ValueError(f"exp490 allowlist columns missing: {missing}")
    frame = pd.read_csv(
        path,
        usecols=list(EXP490_INPUT_ALLOWLIST),
        dtype={
            "well": str,
            "row_idx": np.int64,
            "suffix_offset": np.int64,
            "md_since": np.float64,
            EXP490_PREDICTION_COLUMN: np.float64,
            EXP357_PREDICTION_COLUMN: np.float64,
        },
    ).loc[:, list(EXP490_INPUT_ALLOWLIST)]
    if len(frame) != int(expected_rows) or frame["well"].nunique() != int(expected_wells):
        raise ValueError("exp490 correction row/well coverage mismatch")
    if frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("exp490 correction contains duplicate keys")
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    expected_suffix = frame.groupby("well", sort=False).cumcount().to_numpy(np.int64)
    if not np.array_equal(frame["suffix_offset"].to_numpy(np.int64), expected_suffix):
        raise ValueError("exp490 suffix_offset sequence mismatch")
    frame.insert(0, "id", frame["well"] + "_" + frame["row_idx"].astype(str))
    numeric = frame[
        ["md_since", EXP490_PREDICTION_COLUMN, EXP357_PREDICTION_COLUMN]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("exp490 correction allowlist contains NaN/Inf")
    frame["correction"] = (
        frame[EXP490_PREDICTION_COLUMN].to_numpy(np.float64)
        - frame[EXP357_PREDICTION_COLUMN].to_numpy(np.float64)
    )
    manifest = {
        "path": str(path),
        "raw_gzip_sha256": raw_sha,
        "decompressed_payload_sha256": payload_sha,
        "source_header_columns": header,
        "loaded_columns": list(EXP490_INPUT_ALLOWLIST),
        "loaded_column_count": len(EXP490_INPUT_ALLOWLIST),
        "forbidden_truth_error_episode_role_fold_scope_by_well_gate_columns_loaded": 0,
        "upstream_fold_loaded_or_used": False,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "logical_sha256": logical_frame_sha256(
            frame[
                [
                    "id",
                    "well",
                    "row_idx",
                    "suffix_offset",
                    "md_since",
                    EXP490_PREDICTION_COLUMN,
                    EXP357_PREDICTION_COLUMN,
                    "correction",
                ]
            ]
        ),
        "correction_prediction_sha256": prediction_sha256(
            frame, "correction", "exp506:exp490_minus_exp357:before_truth_attach"
        ),
    }
    return frame, manifest


def freeze_truth_free_components(
    anchor: pd.DataFrame, correction: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_columns = [
        "id",
        "well",
        "row_idx",
        "suffix_offset",
        "md_since",
        "outer_fold",
        "anchor_prediction",
    ]
    correction_columns = [
        "id",
        "well",
        "row_idx",
        "suffix_offset",
        "md_since",
        EXP490_PREDICTION_COLUMN,
        EXP357_PREDICTION_COLUMN,
        "correction",
    ]
    joined = anchor[anchor_columns].merge(
        correction[correction_columns],
        on=["id", "well", "row_idx"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_anchor", "_correction"),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("anchor/correction join has missing or extra keys")
    if not np.array_equal(
        joined["suffix_offset_anchor"].to_numpy(np.int64),
        joined["suffix_offset_correction"].to_numpy(np.int64),
    ):
        raise ValueError("anchor/correction suffix_offset parity failed")
    md_delta = np.abs(
        joined["md_since_anchor"].to_numpy(np.float64)
        - joined["md_since_correction"].to_numpy(np.float64)
    )
    if float(md_delta.max(initial=0.0)) > 1.0e-6:
        raise ValueError("anchor/correction md_since parity failed")
    frozen = joined.rename(
        columns={
            "suffix_offset_anchor": "suffix_offset",
            "md_since_anchor": "md_since",
        }
    )[
        [
            "id",
            "well",
            "row_idx",
            "suffix_offset",
            "md_since",
            "outer_fold",
            "anchor_prediction",
            EXP490_PREDICTION_COLUMN,
            EXP357_PREDICTION_COLUMN,
            "correction",
        ]
    ]
    frozen = frozen.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if not np.isfinite(
        frozen[
            [
                "md_since",
                "anchor_prediction",
                EXP490_PREDICTION_COLUMN,
                EXP357_PREDICTION_COLUMN,
                "correction",
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("truth-free frozen surface contains NaN/Inf")
    manifest = {
        "truth_attached": False,
        "rows": len(frozen),
        "wells": int(frozen["well"].nunique()),
        "duplicate_keys": int(frozen.duplicated(["well", "row_idx"]).sum()),
        "missing_or_extra_keys": 0,
        "suffix_offset_mismatch_rows": 0,
        "md_since_max_abs_difference": float(md_delta.max(initial=0.0)),
        "key_fold_logical_sha256": logical_frame_sha256(
            frozen[["id", "well", "row_idx", "suffix_offset", "outer_fold"]]
        ),
        "frozen_component_logical_sha256": logical_frame_sha256(frozen),
        "anchor_prediction_sha256": prediction_sha256(
            frozen, "anchor_prediction", "exp506:resolved_exp413_anchor:before_truth_attach"
        ),
        "correction_prediction_sha256": prediction_sha256(
            frozen, "correction", "exp506:exp490_minus_exp357:after_anchor_alignment"
        ),
    }
    return frozen, manifest


def attach_anchor_truth(
    path: Path,
    frozen: pd.DataFrame,
    *,
    expected_sha256: str,
    expected_anchor_rmse: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(path) != str(expected_sha256):
        raise ValueError("anchor changed between truth-free freeze and truth attach")
    truth = pd.read_parquet(path, columns=list(ANCHOR_TRUTH_COLUMNS))
    truth["id"] = truth["id"].astype(str)
    truth["well"] = truth["well"].astype(str)
    truth["outer_fold"] = pd.to_numeric(
        truth["outer_fold"], errors="raise"
    ).astype(np.int8)
    if truth["id"].duplicated().any():
        raise ValueError("anchor truth surface contains duplicate IDs")
    frame = frozen.merge(
        truth,
        on=["id", "well", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    actual = frame["actual_tvt"].to_numpy(np.float64)
    if not np.isfinite(actual).all():
        raise ValueError("truth attach produced missing/non-finite values")
    observed_rmse = rmse(actual, frame["anchor_prediction"].to_numpy(np.float64))
    if abs(observed_rmse - float(expected_anchor_rmse)) > 1.0e-9:
        raise ValueError(
            f"resolved anchor RMSE mismatch: {observed_rmse} != {expected_anchor_rmse}"
        )
    return frame, {
        "truth_loaded_after_component_freeze": True,
        "loaded_columns": list(ANCHOR_TRUTH_COLUMNS),
        "rows": len(frame),
        "anchor_rmse": observed_rmse,
        "truth_logical_sha256": logical_frame_sha256(
            frame[["id", "well", "row_idx", "outer_fold", "actual_tvt"]]
        ),
    }

# %% [markdown]
# ## 5. Meta-fold scalar fit helpers

# %%
def rmse(actual: Any, prediction: Any) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def fit_bounded_additive_weight(
    actual: Any,
    anchor: Any,
    component: Any,
    bounds: tuple[float, float] = (0.0, 0.10),
) -> tuple[float, float, float, float]:
    y = np.asarray(actual, dtype=np.float64)
    base = np.asarray(anchor, dtype=np.float64)
    delta = np.asarray(component, dtype=np.float64)
    if not (len(y) == len(base) == len(delta)) or len(y) == 0:
        raise ValueError("weight-fit arrays must have equal nonzero length")
    if not np.isfinite(np.column_stack([y, base, delta])).all():
        raise ValueError("weight-fit arrays contain NaN/Inf")
    residual = base - y
    numerator = -float(np.dot(residual, delta))
    denominator = float(np.dot(delta, delta))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("weight-fit component has zero/non-finite norm")
    unconstrained = numerator / denominator
    weight = float(np.clip(unconstrained, bounds[0], bounds[1]))
    return weight, float(unconstrained), numerator, denominator


def crossfit_additive_component(
    frame: pd.DataFrame,
    component_column: str,
    *,
    bounds: tuple[float, float] = (0.0, 0.10),
    weight_name: str = "lambda",
) -> tuple[np.ndarray, pd.DataFrame]:
    actual = frame["actual_tvt"].to_numpy(np.float64)
    anchor = frame["anchor_prediction"].to_numpy(np.float64)
    component = frame[component_column].to_numpy(np.float64)
    folds = frame["outer_fold"].to_numpy(np.int8)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for held_fold in FIXED_FOLDS:
        fit_mask = folds != held_fold
        held_mask = folds == held_fold
        if not fit_mask.any() or not held_mask.any():
            raise ValueError(f"meta fold {held_fold} has empty fit/apply rows")
        weight, unconstrained, numerator, denominator = fit_bounded_additive_weight(
            actual[fit_mask], anchor[fit_mask], component[fit_mask], bounds
        )
        prediction[held_mask] = anchor[held_mask] + weight * component[held_mask]
        rows.append(
            {
                "held_fold": held_fold,
                "fit_folds": ",".join(str(fold) for fold in FIXED_FOLDS if fold != held_fold),
                "fit_rows": int(fit_mask.sum()),
                "apply_rows": int(held_mask.sum()),
                f"{weight_name}_unconstrained": unconstrained,
                weight_name: weight,
                "numerator": numerator,
                "denominator": denominator,
                "lower_bound_hit": bool(weight <= bounds[0] + 1.0e-15),
                "upper_bound_hit": bool(weight >= bounds[1] - 1.0e-15),
                "strict_interior": bool(bounds[0] < weight < bounds[1]),
                "fit_rmse": rmse(
                    actual[fit_mask], anchor[fit_mask] + weight * component[fit_mask]
                ),
            }
        )
    if not np.isfinite(prediction).all():
        raise ValueError("cross-fitted prediction is incomplete")
    return prediction, pd.DataFrame(rows)

# %% [markdown]
# ## 6. Primary metrics, scope, tail, and gate helpers

# %%
def metric_row(
    frame: pd.DataFrame, mask: np.ndarray, scope: str
) -> dict[str, Any]:
    if not np.any(mask):
        raise ValueError(f"metric scope has no rows: {scope}")
    actual = frame.loc[mask, "actual_tvt"].to_numpy(np.float64)
    anchor = frame.loc[mask, "anchor_prediction"].to_numpy(np.float64)
    primary = frame.loc[mask, "primary_prediction"].to_numpy(np.float64)
    anchor_rmse = rmse(actual, anchor)
    primary_rmse = rmse(actual, primary)
    return {
        "scope": scope,
        "rows": int(mask.sum()),
        "wells": int(frame.loc[mask, "well"].nunique()),
        "anchor_rmse": anchor_rmse,
        "primary_rmse": primary_rmse,
        "delta_rmse_primary_minus_anchor": primary_rmse - anchor_rmse,
        "gain_anchor_minus_primary": anchor_rmse - primary_rmse,
    }


def build_primary_readouts(
    frame: pd.DataFrame,
    *,
    hidden_like_assignment_path: Path,
    hidden_like_assignment_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = metric_row(frame, np.ones(len(frame), dtype=bool), "pooled")
    fold_metrics = pd.DataFrame(
        [
            metric_row(
                frame,
                frame["outer_fold"].eq(fold).to_numpy(),
                f"outer_fold_{fold}",
            )
            | {"outer_fold": fold}
            for fold in FIXED_FOLDS
        ]
    )
    md = frame["md_since"].to_numpy(np.float64)
    scope_masks = {
        "md_0_250": md <= 250.0,
        "md_250_1000": (md > 250.0) & (md < 1000.0),
        "md_1000_plus": md >= 1000.0,
    }
    if sha256_file(hidden_like_assignment_path) != str(hidden_like_assignment_sha256):
        raise ValueError("hidden-like assignment SHA mismatch")
    assignment = pd.read_csv(
        hidden_like_assignment_path, dtype={"well_id": str}
    )
    if assignment["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    assignment = assignment.set_index("well_id")
    hidden_columns = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    for scope, column in hidden_columns.items():
        if column not in assignment.columns:
            raise ValueError(f"hidden-like assignment missing {column}")
        scope_masks[scope] = (
            frame["well"].map(assignment[column]).eq("valid").to_numpy()
        )
    scope_metrics = pd.DataFrame(
        [metric_row(frame, mask, scope) for scope, mask in scope_masks.items()]
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        anchor_rmse = rmse(group["actual_tvt"], group["anchor_prediction"])
        primary_rmse = rmse(group["actual_tvt"], group["primary_prediction"])
        well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "anchor_rmse": anchor_rmse,
                "primary_rmse": primary_rmse,
                "delta_rmse_primary_minus_anchor": primary_rmse - anchor_rmse,
            }
        )
    return pooled, fold_metrics, scope_metrics, pd.DataFrame(well_rows)


def build_primary_gate(
    *,
    config: Mapping[str, Any],
    pooled: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    weights: pd.DataFrame,
    technical_checks: Mapping[str, bool],
) -> dict[str, Any]:
    spec = dict(config["promotion_gate"])
    deltas = by_well["delta_rmse_primary_minus_anchor"].to_numpy(np.float64)
    p95 = float(np.quantile(deltas, 0.95))
    worst_pos = int(np.argmax(deltas))
    worst = float(deltas[worst_pos])
    lambda_values = weights["lambda"].to_numpy(np.float64)
    checks = {
        "technical_all_pass": bool(technical_checks) and all(technical_checks.values()),
        "pooled_gain_min": float(pooled["gain_anchor_minus_primary"])
        >= float(spec["pooled_gain_min_ft"]),
        "nonworse_folds_5_of_5": int(
            fold_metrics["delta_rmse_primary_minus_anchor"].le(0.0).sum()
        )
        >= int(spec["nonworse_folds_required"]),
        "all_fixed_scopes_nonworse": bool(
            scope_metrics["delta_rmse_primary_minus_anchor"]
            .le(float(spec["fixed_scope_max_delta_ft"]))
            .all()
        ),
        "by_well_p95_delta_le_0p25": p95
        <= float(spec["by_well_p95_max_delta_ft"]),
        "worst_well_delta_le_0p25": worst
        <= float(spec["by_well_worst_max_delta_ft"]),
        "lambda_positive_5_of_5": int(np.sum(lambda_values > 0.0))
        >= int(spec["lambda_positive_meta_folds_required"]),
        "lambda_strictly_below_upper_bound_5_of_5": int(
            np.sum(
                lambda_values
                < float(config["ensemble"]["primary"]["lambda_bounds"][1])
            )
        )
        >= int(spec["lambda_strictly_below_upper_bound_meta_folds_required"]),
        "lambda_range_le_0p05": float(np.ptp(lambda_values))
        <= float(spec["lambda_range_max"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": (
            "PASS_QUALIFY_INFERENCE_DESIGN_FOR_SEPARATE_APPROVAL"
            if passed
            else str(spec["fail_decision"])
        ),
        "checks": checks,
        "technical_checks": dict(technical_checks),
        "pooled": dict(pooled),
        "nonworse_folds": int(
            fold_metrics["delta_rmse_primary_minus_anchor"].le(0.0).sum()
        ),
        "fixed_scope_nonworse": int(
            scope_metrics["delta_rmse_primary_minus_anchor"].le(0.0).sum()
        ),
        "by_well_delta": {
            "median": float(np.quantile(deltas, 0.50)),
            "p90": float(np.quantile(deltas, 0.90)),
            "p95": p95,
            "p99": float(np.quantile(deltas, 0.99)),
            "worst": worst,
            "worst_well": str(by_well.iloc[worst_pos]["well"]),
            "worsened_well_count_plus_0p25ft": int(np.sum(deltas > 0.25)),
            "worsened_well_count_plus_1ft": int(np.sum(deltas > 1.0)),
            "worsened_well_count_plus_3ft": int(np.sum(deltas > 3.0)),
            "worsened_well_count_plus_5ft": int(np.sum(deltas > 5.0)),
        },
        "lambda": {
            "values": lambda_values.tolist(),
            "median_for_deployment": float(np.median(lambda_values)),
            "min": float(np.min(lambda_values)),
            "max": float(np.max(lambda_values)),
            "range": float(np.ptp(lambda_values)),
            "strict_interior_count": int(weights["strict_interior"].sum()),
        },
        "control_evaluated_before_gate_freeze": False,
        "fail_rescue_allowed": False,
    }

# %% [markdown]
# ## 7. Report-only control and reproducibility helpers

# %%
def _pearson(left: Any, right: Any) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def build_report_only_control(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    control_frame = frame.copy()
    control_frame["convex_component"] = (
        control_frame[EXP490_PREDICTION_COLUMN].to_numpy(np.float64)
        - control_frame["anchor_prediction"].to_numpy(np.float64)
    )
    prediction, weights = crossfit_additive_component(
        control_frame,
        "convex_component",
        bounds=(0.0, 0.10),
        weight_name="convex_weight",
    )
    actual = control_frame["actual_tvt"].to_numpy(np.float64)
    anchor = control_frame["anchor_prediction"].to_numpy(np.float64)
    exp490 = control_frame[EXP490_PREDICTION_COLUMN].to_numpy(np.float64)
    exp357 = control_frame[EXP357_PREDICTION_COLUMN].to_numpy(np.float64)
    correction = control_frame["correction"].to_numpy(np.float64)
    anchor_residual = anchor - actual
    exp490_residual = exp490 - actual
    correction_norm = float(np.linalg.norm(correction))
    residual_norm = float(np.linalg.norm(anchor_residual))
    fold_correlations = {
        str(fold): _pearson(
            anchor_residual[control_frame["outer_fold"].eq(fold).to_numpy()],
            exp490_residual[control_frame["outer_fold"].eq(fold).to_numpy()],
        )
        for fold in FIXED_FOLDS
    }
    report = {
        "selectable": False,
        "may_rescue_primary": False,
        "computed_after_primary_gate_freeze": True,
        "formula": "anchor + w * (exp490 - anchor)",
        "bounds": [0.0, 0.10],
        "crossfit_rmse": rmse(actual, prediction),
        "anchor_rmse": rmse(actual, anchor),
        "exp490_standalone_rmse": rmse(actual, exp490),
        "exp357_parent_rmse": rmse(actual, exp357),
        "weights": weights["convex_weight"].tolist(),
        "residual_pearson_anchor_vs_exp490": _pearson(
            anchor_residual, exp490_residual
        ),
        "residual_pearson_by_fold": fold_correlations,
        "anchor_residual_vs_correction": {
            "dot_product": float(np.dot(anchor_residual, correction)),
            "cosine": (
                float(np.dot(anchor_residual, correction) / (residual_norm * correction_norm))
                if residual_norm > 0.0 and correction_norm > 0.0
                else None
            ),
            "covariance": float(np.cov(anchor_residual, correction, ddof=0)[0, 1]),
            "anchor_residual_norm": residual_norm,
            "correction_norm": correction_norm,
        },
    }
    return report, weights, prediction


def build_reproducibility_manifest(
    *,
    output_dir: Path,
    source_path: Path,
    config_path: Path,
    contract_path: Path,
    anchor_resolution: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_a",
        "runtime": "kaggle_private_cpu_internet_off",
        "kernel_version": os.environ.get(
            "KAGGLE_KERNEL_VERSION_NUMBER", "unrecorded"
        ),
        "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unrecorded"),
        "seed_policy": "no_rng_fixed_fold_stable_key_order_float64_reduction",
        "deterministic_anchor": False,
        "rerun_result": "pending_independent_rerun",
        "source_sha256": sha256_file(source_path),
        "config_sha256": sha256_file(config_path),
        "contract_sha256": sha256_file(contract_path),
        "anchor_resolution_sha256": sha256_json(anchor_resolution),
        "input_manifest_sha256": sha256_json(input_manifest),
        "artifact_sha256": {
            name: sha256_file(path) for name, path in artifact_paths.items()
        },
        "output_dir": str(output_dir),
        "models": 0,
        "boosters": 0,
        "hmm_pf_beam_runs": 0,
        "gpu_runs": 0,
        "inference_or_submission_generated": False,
    }

# %% [markdown]
# ## 8. Setup and fixed execution inventory

# %%
print("experiment", EXPERIMENT_NAME)
print("route/status", CONFIG["experiment"]["route"], CONFIG["experiment"]["status"])
print("resolved anchor", ANCHOR_RESOLUTION["selected_anchor"]["id"])
print("primary", CONFIG["ensemble"]["primary"]["prediction_formula"])
print("lambda bounds", CONFIG["ensemble"]["primary"]["lambda_bounds"])
print("execution inventory", json.dumps(CONFIG["execution_contract"], indent=2))
print("Kaggle Stage A run approved", CONFIG["implementation"]["kaggle_run_approved"])

if EXECUTE_NOTEBOOK and not RUN_STAGE_A:
    print(
        "Stage A implementation is present, but execution is fail-closed until a "
        "separate Kaggle run approval updates config.yaml."
    )

# %% [markdown]
# ## 9. Resolve inputs and freeze truth-free correction

# %%
if RUN_STAGE_A:
    started_at = time.time()
    anchor_spec = CONFIG["data"]["exp413_root_anchor"]
    correction_spec = CONFIG["data"]["exp490_correction_source"]
    anchor_path = resolve_file_by_sha(
        anchor_spec["root_patterns"],
        expected_sha256=anchor_spec["expected_oof_sha256"],
        filename_if_directory=anchor_spec["oof_file"],
    )
    correction_path = resolve_file_by_sha(
        correction_spec["patterns"],
        expected_sha256=correction_spec["expected_raw_gzip_sha256"],
    )
    write_json(OUTPUT_DIR / "anchor_resolution_manifest.json", ANCHOR_RESOLUTION)
    anchor_frame, anchor_input_manifest = load_anchor_without_truth(
        anchor_path,
        expected_sha256=anchor_spec["expected_oof_sha256"],
        expected_rows=CONFIG["validation"]["expected_rows"],
        expected_wells=CONFIG["validation"]["expected_wells"],
        expected_cv=anchor_spec["expected_cv"],
    )
    anchor_input_manifest["evidence_files"] = verify_anchor_evidence_files(
        anchor_path, anchor_spec
    )
    anchor_input_manifest["declared_upstream_fold_manifest_sha256"] = CONFIG[
        "validation"
    ]["fold_manifest_sha256"]
    correction_frame, correction_input_manifest = load_correction_without_truth(
        correction_path,
        expected_raw_gzip_sha256=correction_spec["expected_raw_gzip_sha256"],
        expected_decompressed_sha256=correction_spec["expected_decompressed_sha256"],
        expected_rows=correction_spec["expected_rows"],
        expected_wells=correction_spec["expected_wells"],
    )
    frozen_frame, correction_manifest = freeze_truth_free_components(
        anchor_frame, correction_frame
    )
    write_json(OUTPUT_DIR / "correction_manifest.json", correction_manifest)
    print(json.dumps(correction_manifest, indent=2))

# %% [markdown]
# ## 10. Attach truth and freeze cross-fitted primary prediction

# %%
if RUN_STAGE_A:
    stage_a_frame, truth_manifest = attach_anchor_truth(
        anchor_path,
        frozen_frame,
        expected_sha256=anchor_spec["expected_oof_sha256"],
        expected_anchor_rmse=anchor_spec["expected_cv"],
    )
    primary_prediction, primary_weights = crossfit_additive_component(
        stage_a_frame,
        "correction",
        bounds=tuple(CONFIG["ensemble"]["primary"]["lambda_bounds"]),
        weight_name="lambda",
    )
    stage_a_frame["primary_prediction"] = primary_prediction
    lambda_by_fold = primary_weights.set_index("held_fold")["lambda"]
    stage_a_frame["lambda"] = stage_a_frame["outer_fold"].map(lambda_by_fold)
    if stage_a_frame["lambda"].isna().any():
        raise ValueError("primary lambda assignment is incomplete")
    primary_prediction_sha = prediction_sha256(
        stage_a_frame,
        "primary_prediction",
        "exp506:crossfit_primary:before_metrics_and_control",
    )
    primary_oof_path = OUTPUT_DIR / "primary_oof_predictions.parquet"
    primary_weights_path = OUTPUT_DIR / "meta_fold_weights.csv"
    stage_a_frame[
        [
            "id",
            "well",
            "row_idx",
            "suffix_offset",
            "md_since",
            "outer_fold",
            "anchor_prediction",
            EXP490_PREDICTION_COLUMN,
            EXP357_PREDICTION_COLUMN,
            "correction",
            "lambda",
            "primary_prediction",
        ]
    ].to_parquet(primary_oof_path, index=False, compression="zstd")
    primary_weights.to_csv(primary_weights_path, index=False)
    print(primary_weights.to_string(index=False))

# %% [markdown]
# ## 11. Evaluate primary, freeze gate, then compute report-only control

# %%
if RUN_STAGE_A:
    hidden_spec = CONFIG["data"]["hidden_like_assignment"]
    hidden_path = resolve_file_by_sha(
        hidden_spec["patterns"],
        expected_sha256=hidden_spec["expected_sha256"],
    )
    pooled, fold_metrics, scope_metrics, by_well = build_primary_readouts(
        stage_a_frame,
        hidden_like_assignment_path=hidden_path,
        hidden_like_assignment_sha256=hidden_spec["expected_sha256"],
    )
    technical_checks = {
        "anchor_resolution_frozen_to_exp413": ANCHOR_RESOLUTION["selected_anchor"]["id"]
        == "exp413_saved_stage_d_oof",
        "exp497_terminal_gate_failed": not ANCHOR_RESOLUTION["exp497"][
            "promotion_gate_passed"
        ],
        "anchor_file_sha_match": anchor_input_manifest["file_sha256"]
        == anchor_spec["expected_oof_sha256"],
        "anchor_fold_scope_hidden_by_well_sha_match": len(
            anchor_input_manifest["evidence_files"]
        )
        == 4,
        "exp490_raw_gzip_sha_match": correction_input_manifest["raw_gzip_sha256"]
        == correction_spec["expected_raw_gzip_sha256"],
        "exp490_payload_sha_match": correction_input_manifest[
            "decompressed_payload_sha256"
        ]
        == correction_spec["expected_decompressed_sha256"],
        "exact_six_column_allowlist": correction_input_manifest["loaded_columns"]
        == list(EXP490_INPUT_ALLOWLIST),
        "truth_late_phase_separation": correction_manifest["truth_attached"] is False
        and truth_manifest["truth_loaded_after_component_freeze"] is True,
        "key_fold_suffix_md_parity": correction_manifest["missing_or_extra_keys"] == 0
        and correction_manifest["suffix_offset_mismatch_rows"] == 0
        and correction_manifest["md_since_max_abs_difference"] <= 1.0e-6,
        "outer5_meta5": len(primary_weights) == 5
        and set(primary_weights["held_fold"]) == set(FIXED_FOLDS),
        "other_four_fold_fit_only": primary_weights["fit_folds"]
        .str.split(",")
        .map(len)
        .eq(4)
        .all(),
        "zero_model_booster_hmm_pf_beam_gpu": all(
            int(CONFIG["execution_contract"][key]) == 0
            for key in (
                "trained_models",
                "total_boosters",
                "hmm_runs",
                "pf_runs",
                "beam_runs",
                "gpu_runs",
            )
        ),
    }
    gate = build_primary_gate(
        config=CONFIG,
        pooled=pooled,
        fold_metrics=fold_metrics,
        scope_metrics=scope_metrics,
        by_well=by_well,
        weights=primary_weights,
        technical_checks=technical_checks,
    )
    fold_path = OUTPUT_DIR / "primary_fold_metrics.csv"
    scope_path = OUTPUT_DIR / "primary_scope_metrics.csv"
    by_well_path = OUTPUT_DIR / "primary_by_well.csv"
    gate_path = OUTPUT_DIR / "primary_gate.json"
    fold_metrics.to_csv(fold_path, index=False)
    scope_metrics.to_csv(scope_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    write_json(gate_path, gate)
    gate_sha_before_control = sha256_file(gate_path)

    report_only_control, control_weights, control_prediction = build_report_only_control(
        stage_a_frame
    )
    if sha256_file(gate_path) != gate_sha_before_control:
        raise RuntimeError("primary gate changed while computing report-only control")
    report_only_control["primary_gate_sha256_before_control"] = gate_sha_before_control
    report_only_control["control_prediction_sha256"] = sha256_json(
        {
            "namespace": "exp506:report_only_convex_control",
            "values_sha256": hashlib.sha256(
                np.asarray(control_prediction, dtype="<f8").tobytes()
            ).hexdigest(),
        }
    )
    report_only_control["weight_rows"] = control_weights.to_dict(orient="records")
    control_path = OUTPUT_DIR / "report_only_control.json"
    write_json(control_path, report_only_control)

    input_manifest = {
        "anchor": anchor_input_manifest,
        "correction_source": correction_input_manifest,
        "truth_attach": truth_manifest,
        "hidden_like_assignment": {
            "path": str(hidden_path),
            "file_sha256": sha256_file(hidden_path),
        },
        "phase_order": [
            "resolve_anchor_from_exp497_terminal_gate",
            "load_anchor_without_truth",
            "load_exp490_exact_six_column_allowlist",
            "freeze_key_fold_anchor_exp490_exp357_correction_sha",
            "attach_anchor_truth",
            "fit_other_four_fold_primary_weights",
            "freeze_primary_prediction",
            "compute_primary_metrics_and_freeze_gate",
            "compute_report_only_control",
        ],
    }
    input_manifest_path = OUTPUT_DIR / "input_manifest.json"
    write_json(input_manifest_path, input_manifest)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_a_complete_gate_passed"
            if gate["passed"]
            else "stage_a_complete_gate_failed_closed"
        ),
        "route": "ensemble",
        "selected_anchor": "exp413_saved_stage_d_oof",
        "anchor_cv": pooled["anchor_rmse"],
        "primary_cv": pooled["primary_rmse"],
        "gain_anchor_minus_primary": pooled["gain_anchor_minus_primary"],
        "public_lb": None,
        "private_lb": None,
        "primary_prediction_sha256": primary_prediction_sha,
        "deployment_lambda": float(np.median(primary_weights["lambda"])),
        "primary_gate": gate,
        "report_only_control": {
            "selectable": False,
            "crossfit_rmse": report_only_control["crossfit_rmse"],
            "weights": report_only_control["weights"],
        },
        "execution_contract": CONFIG["execution_contract"],
        "inference_enabled": False,
        "submission_generated": False,
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    write_json(metrics_path, metrics)
    artifact_paths = {
        "anchor_resolution_manifest": OUTPUT_DIR / "anchor_resolution_manifest.json",
        "input_manifest": input_manifest_path,
        "correction_manifest": OUTPUT_DIR / "correction_manifest.json",
        "meta_fold_weights": primary_weights_path,
        "primary_oof_predictions": primary_oof_path,
        "primary_fold_metrics": fold_path,
        "primary_scope_metrics": scope_path,
        "primary_by_well": by_well_path,
        "primary_gate": gate_path,
        "report_only_control": control_path,
        "metrics": metrics_path,
    }
    reproducibility = build_reproducibility_manifest(
        output_dir=OUTPUT_DIR,
        source_path=find_support_file(
            "exp506_exp490_mean_reversion_correction_blend_on_exp413_compact_selfcontained_train.py"
        ),
        config_path=CONFIG_PATH,
        contract_path=CONTRACT_PATH,
        anchor_resolution=ANCHOR_RESOLUTION,
        input_manifest=input_manifest,
        artifact_paths=artifact_paths,
    )
    reproducibility["elapsed_seconds"] = round(time.time() - started_at, 3)
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    write_json(reproducibility_path, reproducibility)
    print(json.dumps(to_jsonable(metrics), indent=2))

# %% [markdown]
# ## 12. Generated artifacts and fixed stop
#
# This candidate notebook never trains a model, regenerates HMM/PF/Beam, runs
# inference, or writes `submission.csv`.  A PASS only qualifies a separately
# approved inference design.  A FAIL closes the hypothesis without changing
# weight bounds, component, scope, router, gate, or anchor.

# %%
if RUN_STAGE_A:
    generated = sorted(
        [
            {
                "path": str(path.relative_to(OUTPUT_DIR)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in OUTPUT_DIR.iterdir()
            if path.is_file()
        ],
        key=lambda item: item["path"],
    )
    print(json.dumps(generated, indent=2))
    print("fixed stop: no inference, submission, or rescue path is implemented")
else:
    print("implementation-only stop: Kaggle Stage A execution was not authorized")
