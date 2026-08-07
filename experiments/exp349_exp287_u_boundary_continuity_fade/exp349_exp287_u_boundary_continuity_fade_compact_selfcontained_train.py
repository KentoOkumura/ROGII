# ---
# jupyter:
#   jupytext:
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
# # exp349 exp287 U-boundary continuity fade — train audit
#
# 保存済み exp287 OOF へ、現在 well の既知 prefix 末端だけを基準にした
# `cap=8 ft / tau=240 MD-ft` の U 境界 fade を1回だけ適用する。
# Stage A では truth / fold / hidden-like assignment を開かず candidate と診断を
# SHA freeze し、Stage B で freeze を再検証してから固定 gate を評価する。
# モデル学習、parameter search、raw-test inference、submission は行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Scientific, leakage, and execution contract
# 3. Frozen input and path helpers
# 4. Raw prefix/suffix and U-fade generation helpers
# 5. Stage A target-free generation and freeze barrier
# 6. Stage B late-truth alignment and metrics
# 7. Fixed technical/scientific decision gate
# 8. Generated artifacts and reproducibility manifest
# 9. Execution orchestration

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp349_exp287_u_boundary_continuity_fade"
PARENT_EXPERIMENT = "exp287_fold_safe_formation_74_addonly_on_exp264"
VARIANT_NAME = "u_cap8_tau240_always_on"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP349_IMPORT_ONLY", "0") == "1"

PRETRUTH_PARENT_COLUMNS = [
    "id",
    "well",
    "fold_safe_formation_74_addonly__lgb_mean__pred_tvt",
]
LATE_TRUTH_PARENT_COLUMNS = [
    "id",
    "well",
    "outer_fold",
    "actual_tvt",
    "fold_safe_formation_74_addonly__lgb_mean__pred_tvt",
]
RAW_ALLOWED_COLUMNS = ["MD", "Z", "TVT_input"]
PRETRUTH_FORBIDDEN_EXACT = {
    "TVT",
    "target",
    "actual",
    "actual_tvt",
    "truth",
    "error",
    "abs_error",
    "squared_error",
    "oracle",
    "outer_fold",
    "verification_like_spatial_role",
    "verification_like_typewell_purged_role",
}
CANDIDATE_COLUMNS = [
    "id",
    "well",
    "raw_row_index",
    "MD",
    "Z",
    "last_visible_row_index",
    "first_hidden_row_index",
    "last_visible_md",
    "u_last",
    "md_since_boundary",
    "parent_pred_tvt",
    "gap_u",
    "abs_gap_bucket",
    "gap_sign",
    "move_tvt",
    "candidate_pred_tvt",
]
DIAGNOSTIC_COLUMNS = [
    "well",
    "prefix_rows",
    "suffix_rows",
    "last_visible_row_index",
    "first_hidden_row_index",
    "last_visible_md",
    "first_hidden_md",
    "u_last",
    "gap_u_before",
    "abs_gap_bucket",
    "gap_sign",
    "clipped_gap_u",
    "first_hidden_md_since_boundary",
    "first_hidden_move_tvt",
    "gap_u_after_first_hidden",
    "max_abs_move_tvt",
    "abs_move_nonincreasing",
]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def find_config_path() -> Path:
    direct = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in direct:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"exp349 config resolution is ambiguous: {matches}")
    return matches[0].resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def canonical_csv_bytes(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> bytes:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    return selected.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="NA",
    ).encode("utf-8")


def write_canonical_csv(
    path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_csv_bytes(frame, columns)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_file_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("RMSE requires nonempty finite arrays")
    return float(np.sqrt(np.mean(delta * delta)))


# %% [markdown]
# ## 2. Scientific, leakage, and execution contract
#
# implementation-only 状態では全 run flag を false に保つ。Stage 0 は canonical
# notebook 採用・Kaggle push/run の別承認後にだけ有効化する。active variant は
# postprocess 1件であり、trained fold/model/config/booster/PF/Beam/HMM/GPU は全て0。

# %%
def validate_pretruth_columns(columns: Sequence[str], *, label: str) -> None:
    names = [str(value) for value in columns]
    duplicated = pd.Index(names)[pd.Index(names).duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{label} has duplicate columns: {duplicated}")
    forbidden = sorted(set(names).intersection(PRETRUTH_FORBIDDEN_EXACT))
    if forbidden:
        raise ValueError(f"{label} exposes pretruth-forbidden columns: {forbidden}")


def validate_scientific_contract(
    config: Mapping[str, Any], *, require_execution: bool
) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp349 route must remain ml_model")
    if nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp349 parent contract changed")
    frozen_parent = {
        "rows": int(nested(config, "validation.parent_oof_rows")),
        "wells": int(nested(config, "validation.parent_oof_wells")),
        "cv": float(nested(config, "validation.parent_oof_cv")),
        "oof_sha256": str(nested(config, "validation.parent_oof_sha256")),
        "model_manifest_sha256": str(
            nested(config, "validation.parent_model_manifest_sha256")
        ),
        "prediction_column": str(nested(config, "validation.parent_prediction_column")),
        "hidden_like_assignment_sha256": str(
            nested(config, "validation.hidden_like_assignment_sha256")
        ),
    }
    if frozen_parent != {
        "rows": 3_783_989,
        "wells": 773,
        "cv": 8.136708220359452,
        "oof_sha256": "8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913",
        "model_manifest_sha256": "419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590",
        "prediction_column": "fold_safe_formation_74_addonly__lgb_mean__pred_tvt",
        "hidden_like_assignment_sha256": (
            "5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597"
        ),
    }:
        raise ValueError(f"frozen parent/input contract changed: {frozen_parent}")
    fixed_method = {
        "cap_ft": float(nested(config, "method.cap_ft")),
        "tau_md_ft": float(nested(config, "method.tau_md_ft")),
        "application": str(nested(config, "method.application")),
        "parameter_search": bool(nested(config, "method.parameter_search")),
    }
    if fixed_method != {
        "cap_ft": 8.0,
        "tau_md_ft": 240.0,
        "application": "always_on_all_eligible_wells",
        "parameter_search": False,
    }:
        raise ValueError(f"fixed U-fade method changed: {fixed_method}")
    execution = dict(nested(config, "execution"))
    counts = {
        "postprocess_variants": int(execution.get("active_postprocess_variants", -1)),
        "reporting_folds": int(execution.get("reporting_folds", -1)),
        "trained_folds": int(execution.get("trained_folds", -1)),
        "model_configs": int(execution.get("model_configs", -1)),
        "trained_models": int(execution.get("trained_models", -1)),
        "boosters": int(execution.get("lightgbm_boosters", -1)),
        "pf_well_runs": int(execution.get("pf_well_runs", -1)),
        "beam_well_runs": int(execution.get("beam_well_runs", -1)),
        "hmm_well_runs": int(execution.get("hmm_well_runs", -1)),
    }
    expected_counts = {
        "postprocess_variants": 1,
        "reporting_folds": 5,
        "trained_folds": 0,
        "model_configs": 0,
        "trained_models": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "hmm_well_runs": 0,
    }
    if counts != expected_counts:
        raise ValueError(f"zero-training contract changed: {counts}")
    if bool(execution.get("parent_control_retraining")) or bool(execution.get("enable_gpu")):
        raise ValueError("parent retraining and GPU must remain disabled")
    forbidden_flags = [
        "run_model_training",
        "run_inference",
        "create_submission",
        "submit_to_kaggle",
    ]
    enabled_forbidden = [name for name in forbidden_flags if bool(execution.get(name, False))]
    if enabled_forbidden:
        raise ValueError(f"forbidden execution flags are enabled: {enabled_forbidden}")
    stage = str(execution.get("active_stage"))
    allowed = [str(value) for value in execution.get("allowed_stages", [])]
    if stage not in allowed:
        raise ValueError(f"active stage is not allowed: {stage}")
    if require_execution:
        if stage != "stage0_saved_oof_audit":
            raise RuntimeError("exp349 Stage 0 execution is not active")
        required_true = [
            "run_approved",
            "kaggle_push_approved",
            "run_stage_a_generation",
            "run_stage_b_evaluation",
        ]
        if not all(bool(execution.get(name, False)) for name in required_true):
            raise RuntimeError("exp349 Stage 0 approval/run flags are incomplete")
        approvals = dict(nested(config, "implementation.approvals"))
        if not bool(approvals.get("canonical_notebook_adoption")) or not bool(
            approvals.get("kaggle_stage0_run")
        ):
            raise RuntimeError("canonical notebook and Kaggle Stage 0 approvals are required")
    elif stage == "implementation_complete_no_run":
        unexpected = [
            name
            for name in [
                "run_approved",
                "kaggle_push_approved",
                "run_stage_a_generation",
                "run_stage_b_evaluation",
            ]
            if bool(execution.get(name, False))
        ]
        if unexpected:
            raise ValueError(f"implementation-only run flags must be false: {unexpected}")
    return {"stage": stage, **counts, "parent_control_retraining": False, "gpu": False}


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_NAME,
        "parent": PARENT_EXPERIMENT,
        "variant": VARIANT_NAME,
        "route": "ml_model",
        "formula": {
            "u_last": "TVT_input[last_visible] + Z[last_visible]",
            "gap_u": "parent_pred[first_hidden] + Z[first_hidden] - u_last",
            "move": "-clip(gap_u,-8,8)*exp(-md_since_boundary/240)",
            "candidate": "parent_pred + move",
            "first_hidden_distance_policy": "strict_positive_raw_md_distance_not_zeroed",
        },
        "method": dict(nested(config, "method")),
        "technical_gate": dict(nested(config, "guards.technical")),
        "scientific_gate": dict(nested(config, "guards.scientific")),
        "pass_policy": str(nested(config, "guards.pass_policy")),
        "fail_action": str(nested(config, "guards.fail_action")),
        "forbidden": list(nested(config, "guards.forbidden")),
        "truth_join_policy": str(nested(config, "reproducibility.truth_join_policy")),
        "parameter_search": False,
    }


# %% [markdown]
# ## 3. Frozen input and path helpers

# %%
def _existing_pattern_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in patterns:
        candidate = Path(str(raw))
        if candidate.exists():
            paths.append(candidate.resolve())
        if not candidate.is_absolute():
            local = PACKAGE_DIR / candidate
            if local.exists():
                paths.append(local.resolve())
    return list(dict.fromkeys(paths))


def select_first_sha_matched_file(
    candidates: Sequence[Path], *, expected_sha256: str, label: str
) -> Path:
    matches = [
        path
        for path in dict.fromkeys(Path(value).resolve() for value in candidates)
        if path.is_file() and sha256_file(path) == str(expected_sha256)
    ]
    if not matches:
        raise FileNotFoundError(f"no SHA-matched {label} was found")
    selected = matches[0]
    if len(matches) > 1:
        print(
            json.dumps(
                {
                    "label": label,
                    "sha_equivalent_match_count": len(matches),
                    "selected_path": str(selected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return selected


def resolve_sha_matched_file(
    patterns: Sequence[str],
    *,
    expected_sha256: str,
    fallback_name: str,
    label: str,
) -> Path:
    candidates = _existing_pattern_paths(patterns)
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(path.resolve() for path in KAGGLE_INPUT_ROOT.rglob(fallback_name))
    return select_first_sha_matched_file(
        candidates,
        expected_sha256=expected_sha256,
        label=label,
    )


def resolve_raw_train_dir(config: Mapping[str, Any]) -> Path:
    configured = _existing_pattern_paths(nested(config, "data.raw_train_dir_patterns"))
    for candidate in configured:
        if candidate.is_dir() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate
    fallback: list[Path] = []
    if KAGGLE_INPUT_ROOT.exists():
        fallback = sorted(
            path.resolve()
            for path in KAGGLE_INPUT_ROOT.rglob("train")
            if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None)
        )
    if len(fallback) != 1:
        raise FileNotFoundError(f"raw train directory was not unique: {fallback}")
    return fallback[0]


def load_parent_pretruth_prediction(
    path: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_sha = str(nested(config, "validation.parent_oof_sha256"))
    file_sha = verify_file_sha(path, expected_sha, "exp287 parent OOF")
    prediction_column = str(nested(config, "validation.parent_prediction_column"))
    columns = ["id", "well", prediction_column]
    validate_pretruth_columns(columns, label="parent pretruth projection")
    frame = pd.read_parquet(path, columns=columns)
    if list(frame.columns) != columns:
        raise ValueError("parent pretruth projection schema/order mismatch")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError("parent OOF id must be unique")
    prediction = frame[prediction_column].to_numpy(np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("parent prediction contains nonfinite values")
    expected_rows = int(nested(config, "validation.parent_oof_rows"))
    expected_wells = int(nested(config, "validation.parent_oof_wells"))
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError("parent pretruth row/well coverage mismatch")
    audit = {
        "path": str(path),
        "file_sha256": file_sha,
        "opened_columns": columns,
        "truth_columns_opened": [],
        "outer_fold_opened": False,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "unique_ids": int(frame["id"].nunique()),
    }
    return frame, audit


# %% [markdown]
# ## 4. Raw prefix/suffix and U-fade generation helpers
#
# raw CSV は `MD/Z/TVT_input` だけを `usecols` で読む。有限 prefix の直後が全て
# NaN suffix、MD は全行で狭義単調増加、最初の suffix 距離は正でなければならない。

# %%
def build_raw_suffix_for_well(
    well: str, frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if list(frame.columns) != RAW_ALLOWED_COLUMNS:
        raise ValueError(f"raw {well} opened unexpected columns: {frame.columns.tolist()}")
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(md).all() or not np.isfinite(z).all():
        raise ValueError(f"raw geometry contains nonfinite values for {well}")
    if len(md) < 2 or not np.all(np.diff(md) > 0.0):
        raise ValueError(f"raw MD must be strictly increasing for {well}")
    missing = np.isnan(tvt_input)
    if not missing.any() or missing.all():
        raise ValueError(f"raw well must have finite prefix and NaN suffix for {well}")
    first_hidden = int(np.flatnonzero(missing)[0])
    if first_hidden <= 0:
        raise ValueError(f"raw prefix is empty for {well}")
    if not np.isfinite(tvt_input[:first_hidden]).all() or not missing[first_hidden:].all():
        raise ValueError(f"TVT_input is not a finite prefix plus contiguous NaN suffix for {well}")
    last_visible = first_hidden - 1
    suffix_index = np.arange(first_hidden, len(frame), dtype=np.int64)
    distance = md[suffix_index] - md[last_visible]
    if not np.isfinite(distance).all() or np.any(distance <= 0.0):
        raise ValueError(f"hidden MD distance must be strictly positive for {well}")
    u_last = float(tvt_input[last_visible] + z[last_visible])
    suffix = pd.DataFrame(
        {
            "id": [f"{well}_{int(index)}" for index in suffix_index],
            "well": well,
            "raw_row_index": suffix_index,
            "MD": md[suffix_index],
            "Z": z[suffix_index],
            "last_visible_row_index": last_visible,
            "first_hidden_row_index": first_hidden,
            "last_visible_md": md[last_visible],
            "u_last": u_last,
            "md_since_boundary": distance,
        }
    )
    manifest = {
        "well": well,
        "raw_rows": len(frame),
        "prefix_rows": first_hidden,
        "suffix_rows": len(suffix),
        "last_visible_row_index": last_visible,
        "first_hidden_row_index": first_hidden,
        "first_hidden_md_since_boundary": float(distance[0]),
        "strictly_increasing_md": True,
        "finite_prefix_contiguous_nan_suffix": True,
    }
    return suffix, manifest


def load_raw_suffix_surface(
    raw_train_dir: Path,
    *,
    expected_wells: Sequence[str],
    expected_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    files = {
        path.name.removesuffix("__horizontal_well.csv"): path
        for path in sorted(Path(raw_train_dir).glob("*__horizontal_well.csv"))
    }
    expected = {str(well) for well in expected_wells}
    if set(files) != expected:
        missing = sorted(expected - set(files))
        extra = sorted(set(files) - expected)
        raise ValueError(f"raw/parent well mismatch: missing={missing[:5]}, extra={extra[:5]}")
    suffix_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    headers_with_forbidden_but_unopened = 0
    for well in sorted(expected):
        path = files[well]
        header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        if not set(RAW_ALLOWED_COLUMNS).issubset(header):
            raise ValueError(f"raw required columns are missing for {well}")
        headers_with_forbidden_but_unopened += int(
            bool(set(header).intersection(PRETRUTH_FORBIDDEN_EXACT))
        )
        frame = pd.read_csv(path, usecols=RAW_ALLOWED_COLUMNS)[RAW_ALLOWED_COLUMNS]
        opened_content_sha = hashlib.sha256(
            canonical_csv_bytes(frame, RAW_ALLOWED_COLUMNS)
        ).hexdigest()
        suffix, manifest = build_raw_suffix_for_well(well, frame)
        suffix_parts.append(suffix)
        manifest_rows.append(
            {
                **manifest,
                "source_file": path.name,
                "opened_target_free_content_sha256": opened_content_sha,
            }
        )
    surface = pd.concat(suffix_parts, ignore_index=True)
    manifest_frame = pd.DataFrame(manifest_rows).sort_values("well").reset_index(drop=True)
    if len(surface) != expected_rows or surface["id"].duplicated().any():
        raise ValueError("raw suffix row/id coverage mismatch")
    if surface["well"].nunique() != len(expected):
        raise ValueError("raw suffix well coverage mismatch")
    audit = {
        "raw_train_dir": str(raw_train_dir),
        "opened_columns": RAW_ALLOWED_COLUMNS,
        "truth_columns_opened": [],
        "outer_fold_opened": False,
        "hidden_like_assignment_opened": False,
        "headers_with_forbidden_but_unopened_columns": headers_with_forbidden_but_unopened,
        "rows": len(surface),
        "wells": int(surface["well"].nunique()),
        "all_wells_prefix_suffix_contract": True,
    }
    return surface, manifest_frame, audit


def assign_abs_gap_bucket(values: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    pairs = list(nested(config, "validation.absolute_gap_buckets_ft"))
    lower = [float(pair[0]) for pair in pairs]
    upper = [np.inf if pair[1] is None else float(pair[1]) for pair in pairs]
    if lower != [0.0, 1.0, 2.0, 4.0, 8.0] or upper != [1.0, 2.0, 4.0, 8.0, np.inf]:
        raise ValueError("absolute-gap bucket contract changed")
    labels = ["[0,1)", "[1,2)", "[2,4)", "[4,8)", "[8,+inf)"]
    bucket = pd.cut(
        np.asarray(values, dtype=np.float64),
        bins=[0.0, 1.0, 2.0, 4.0, 8.0, np.inf],
        labels=labels,
        right=False,
        include_lowest=True,
    )
    if bucket.isna().any():
        raise ValueError("absolute gap bucket assignment is incomplete")
    # pandas versions differ here: Categorical.astype(str) may already return
    # a NumPy array, while Series-backed variants expose ``to_numpy``.
    return np.asarray(bucket.astype(str), dtype=str)


def apply_fixed_u_boundary_fade(
    raw_suffix: pd.DataFrame,
    parent_pretruth: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_column = str(nested(config, "validation.parent_prediction_column"))
    if list(parent_pretruth.columns) != ["id", "well", prediction_column]:
        raise ValueError("parent pretruth schema changed")
    if len(raw_suffix) != len(parent_pretruth):
        raise ValueError("raw suffix and parent OOF row counts differ")
    aligned = raw_suffix.merge(
        parent_pretruth,
        on=["id", "well"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(aligned) != len(raw_suffix) or not aligned["id"].equals(raw_suffix["id"]):
        raise ValueError("raw suffix/OOF ID order alignment failed")
    parent = aligned[prediction_column].to_numpy(np.float64)
    if not np.isfinite(parent).all():
        raise ValueError("raw suffix has unmatched or nonfinite parent prediction")
    aligned = aligned.rename(columns={prediction_column: "parent_pred_tvt"})
    first_parent = aligned.groupby("well", sort=False)["parent_pred_tvt"].transform("first")
    first_z = aligned.groupby("well", sort=False)["Z"].transform("first")
    gap = first_parent.to_numpy(np.float64) + first_z.to_numpy(np.float64) - aligned[
        "u_last"
    ].to_numpy(np.float64)
    cap = float(nested(config, "method.cap_ft"))
    tau = float(nested(config, "method.tau_md_ft"))
    distance = aligned["md_since_boundary"].to_numpy(np.float64)
    expected_move = -np.clip(gap, -cap, cap) * np.exp(-distance / tau)
    candidate = parent + expected_move
    aligned["gap_u"] = gap
    aligned["abs_gap_bucket"] = assign_abs_gap_bucket(np.abs(gap), config)
    aligned["gap_sign"] = np.where(gap < 0.0, "negative", np.where(gap > 0.0, "positive", "zero"))
    aligned["move_tvt"] = expected_move
    aligned["candidate_pred_tvt"] = candidate
    aligned = aligned[CANDIDATE_COLUMNS]

    numeric = aligned[
        [
            "MD",
            "Z",
            "last_visible_md",
            "u_last",
            "md_since_boundary",
            "parent_pred_tvt",
            "gap_u",
            "move_tvt",
            "candidate_pred_tvt",
        ]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("target-free candidate contains nonfinite values")
    if aligned["id"].duplicated().any() or np.any(distance <= 0.0):
        raise ValueError("target-free candidate identity/distance contract failed")
    formula_move = -np.clip(aligned["gap_u"].to_numpy(np.float64), -cap, cap) * np.exp(
        -aligned["md_since_boundary"].to_numpy(np.float64) / tau
    )
    formula_parity = float(
        np.max(np.abs(formula_move - aligned["move_tvt"].to_numpy(np.float64)))
    )
    atol = float(nested(config, "guards.technical.formula_atol"))
    if formula_parity > atol:
        raise ValueError("U-fade formula parity failed")
    abs_move = np.abs(aligned["move_tvt"].to_numpy(np.float64))
    if float(abs_move.max()) > cap + atol:
        raise ValueError("U-fade move exceeded the fixed cap")
    diff = pd.Series(abs_move).groupby(aligned["well"], sort=False).diff().fillna(0.0)
    if bool((diff > atol).any()):
        raise ValueError("absolute U-fade move is not nonincreasing per well")
    nonzero = aligned["gap_u"].to_numpy(np.float64) != 0.0
    if np.any(aligned.loc[nonzero, "move_tvt"].to_numpy(np.float64) * gap[nonzero] >= 0.0):
        raise ValueError("U-fade move sign is not opposite to nonzero gap")

    first = aligned.groupby("well", sort=True, as_index=False).first()
    after_gap = (
        first["candidate_pred_tvt"].to_numpy(np.float64)
        + first["Z"].to_numpy(np.float64)
        - first["u_last"].to_numpy(np.float64)
    )
    if np.any(np.abs(after_gap) > np.abs(first["gap_u"].to_numpy(np.float64)) + atol):
        raise ValueError("first-hidden absolute U gap increased")
    counts = aligned.groupby("well", sort=True).size().rename("suffix_rows")
    diagnostics = first[
        [
            "well",
            "last_visible_row_index",
            "first_hidden_row_index",
            "last_visible_md",
            "MD",
            "u_last",
            "gap_u",
            "abs_gap_bucket",
            "gap_sign",
            "md_since_boundary",
            "move_tvt",
        ]
    ].copy()
    diagnostics = diagnostics.rename(
        columns={
            "MD": "first_hidden_md",
            "gap_u": "gap_u_before",
            "md_since_boundary": "first_hidden_md_since_boundary",
            "move_tvt": "first_hidden_move_tvt",
        }
    )
    diagnostics["prefix_rows"] = diagnostics["first_hidden_row_index"].astype(int)
    diagnostics = diagnostics.merge(counts, on="well", validate="one_to_one")
    diagnostics["clipped_gap_u"] = np.clip(
        diagnostics["gap_u_before"].to_numpy(np.float64), -cap, cap
    )
    diagnostics["gap_u_after_first_hidden"] = after_gap
    max_move = (
        aligned.assign(_abs_move=np.abs(aligned["move_tvt"].to_numpy(np.float64)))
        .groupby("well", sort=True)["_abs_move"]
        .max()
    )
    diagnostics["max_abs_move_tvt"] = diagnostics["well"].map(max_move)
    diagnostics["abs_move_nonincreasing"] = True
    diagnostics = diagnostics[DIAGNOSTIC_COLUMNS].sort_values("well").reset_index(drop=True)
    technical = {
        "raw_suffix_parent_exact_alignment": True,
        "finite_output": True,
        "maximum_abs_move_tvt": float(abs_move.max()),
        "maximum_formula_abs_difference": formula_parity,
        "abs_move_nonincreasing_per_well": True,
        "first_hidden_abs_u_gap_nonincrease": True,
        "opposite_sign_for_nonzero_gap": True,
        "rows": len(aligned),
        "wells": int(aligned["well"].nunique()),
    }
    return aligned, diagnostics, technical


# %% [markdown]
# ## 5. Stage A target-free generation and freeze barrier
#
# candidate parquet、pretruth diagnostics、raw/schema/input manifest を書き、各SHAを
# freeze manifest へ記録する。candidate/diagnostic を再読込して一致確認するまで
# `actual_tvt`、`outer_fold`、hidden-like assignment はロードしない。

# %%
def freeze_target_free_generation(
    candidate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    raw_manifest: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    parent_oof_path: Path,
    parent_model_manifest_path: Path,
    parent_audit: Mapping[str, Any],
    raw_audit: Mapping[str, Any],
    technical_audit: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    validate_pretruth_columns(candidate.columns, label="target-free candidate")
    if list(candidate.columns) != CANDIDATE_COLUMNS:
        raise ValueError("candidate output schema/order mismatch")
    if list(diagnostics.columns) != DIAGNOSTIC_COLUMNS:
        raise ValueError("diagnostic output schema/order mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "target_free_candidate.parquet"
    diagnostic_path = output_dir / "u_boundary_diagnostics_pretruth.csv"
    raw_manifest_path = output_dir / "raw_horizontal_ordered_manifest.csv"
    schema_path = output_dir / "input_schema_manifest.json"
    input_path = output_dir / "stage_a_input_manifest.json"
    candidate.to_parquet(candidate_path, index=False)
    diagnostic_sha = write_canonical_csv(diagnostic_path, diagnostics, DIAGNOSTIC_COLUMNS)
    raw_manifest_sha = write_canonical_csv(raw_manifest_path, raw_manifest)
    schema_manifest = {
        "schema_version": "1.0.0",
        "raw_opened_columns": RAW_ALLOWED_COLUMNS,
        "parent_opened_columns": list(parent_audit["opened_columns"]),
        "candidate_columns": CANDIDATE_COLUMNS,
        "candidate_dtypes": {name: str(dtype) for name, dtype in candidate.dtypes.items()},
        "diagnostic_columns": DIAGNOSTIC_COLUMNS,
        "truth_columns_opened_before_freeze": [],
        "outer_fold_opened_before_freeze": False,
        "hidden_like_assignment_opened_before_freeze": False,
    }
    write_json(schema_path, schema_manifest)
    input_manifest = {
        "schema_version": "1.0.0",
        "stage": "target_free_generation_complete_truth_not_opened",
        "parent_oof": dict(parent_audit),
        "parent_model_manifest": {
            "path": str(parent_model_manifest_path),
            "sha256": verify_file_sha(
                parent_model_manifest_path,
                str(nested(config, "validation.parent_model_manifest_sha256")),
                "exp287 model manifest",
            ),
        },
        "raw": dict(raw_audit),
        "raw_horizontal_ordered_manifest_sha256": raw_manifest_sha,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "scientific_contract_sha256": sha256_file(output_dir / "scientific_contract.json"),
        "truth_access_before_freeze_count": 0,
    }
    write_json(input_path, input_manifest)
    candidate_sha = sha256_file(candidate_path)
    readback = pd.read_parquet(
        candidate_path,
        columns=["id", "well", "move_tvt", "candidate_pred_tvt"],
    )
    readback_ids = readback["id"].astype(str).reset_index(drop=True)
    candidate_ids = candidate["id"].astype(str).reset_index(drop=True)
    if len(readback) != len(candidate) or not readback_ids.equals(candidate_ids):
        raise ValueError("candidate parquet readback identity mismatch")
    if not np.isfinite(readback[["move_tvt", "candidate_pred_tvt"]].to_numpy(np.float64)).all():
        raise ValueError("candidate parquet readback contains nonfinite values")
    diagnostic_readback = pd.read_csv(diagnostic_path, dtype={"well": str})
    diagnostic_wells = diagnostics["well"].astype(str).reset_index(drop=True)
    if (
        len(diagnostic_readback) != len(diagnostics)
        or not diagnostic_readback["well"].reset_index(drop=True).equals(diagnostic_wells)
    ):
        raise ValueError("diagnostic CSV readback identity mismatch")
    if sha256_file(diagnostic_path) != diagnostic_sha:
        raise ValueError("diagnostic CSV SHA readback mismatch")
    freeze = {
        "schema_version": "1.0.0",
        "stage": "target_free_candidate_frozen_before_truth_join",
        "variant": VARIANT_NAME,
        "rows": len(candidate),
        "wells": int(candidate["well"].nunique()),
        "truth_access_before_freeze_count": 0,
        "outer_fold_opened_before_freeze": False,
        "hidden_like_assignment_opened_before_freeze": False,
        "technical_audit": dict(technical_audit),
        "artifacts": {
            "candidate": {"file": candidate_path.name, "sha256": candidate_sha},
            "diagnostic": {"file": diagnostic_path.name, "sha256": diagnostic_sha},
            "raw_manifest": {"file": raw_manifest_path.name, "sha256": raw_manifest_sha},
            "schema_manifest": {"file": schema_path.name, "sha256": sha256_file(schema_path)},
            "stage_a_input_manifest": {
                "file": input_path.name,
                "sha256": sha256_file(input_path),
            },
            "scientific_contract": {
                "file": "scientific_contract.json",
                "sha256": sha256_file(output_dir / "scientific_contract.json"),
            },
        },
        "parent_oof_sha256": sha256_file(parent_oof_path),
        "parent_model_manifest_sha256": sha256_file(parent_model_manifest_path),
        "config_sha256": sha256_file(config_path),
        "prediction_and_diagnostic_sha_readback_verified": True,
    }
    freeze_path = output_dir / "target_free_freeze_manifest.json"
    write_json(freeze_path, freeze)
    freeze_sha = sha256_file(freeze_path)
    verify_generation_freeze(output_dir, expected_freeze_manifest_sha256=freeze_sha)
    return freeze, freeze_sha


def verify_generation_freeze(
    output_dir: Path, *, expected_freeze_manifest_sha256: str
) -> dict[str, Any]:
    freeze_path = Path(output_dir) / "target_free_freeze_manifest.json"
    verify_file_sha(freeze_path, expected_freeze_manifest_sha256, "target-free freeze manifest")
    freeze = json.loads(freeze_path.read_text())
    if int(freeze.get("truth_access_before_freeze_count", -1)) != 0:
        raise ValueError("truth was accessed before the freeze barrier")
    for label, contract in dict(freeze.get("artifacts") or {}).items():
        verify_file_sha(
            Path(output_dir) / str(contract["file"]),
            str(contract["sha256"]),
            f"frozen {label}",
        )
    return freeze


def run_stage_a(
    *,
    parent_oof_path: Path,
    parent_model_manifest_path: Path,
    raw_train_dir: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> tuple[str, dict[str, Any]]:
    parent, parent_audit = load_parent_pretruth_prediction(parent_oof_path, config)
    expected_wells = sorted(parent["well"].astype(str).unique().tolist())
    raw_suffix, raw_manifest, raw_audit = load_raw_suffix_surface(
        raw_train_dir,
        expected_wells=expected_wells,
        expected_rows=int(nested(config, "validation.parent_oof_rows")),
    )
    candidate, diagnostics, technical = apply_fixed_u_boundary_fade(
        raw_suffix,
        parent,
        config,
    )
    technical.update(
        {
            "all_wells_prefix_suffix_contract": bool(
                raw_audit["all_wells_prefix_suffix_contract"]
            ),
            "all_hidden_md_distance_positive": bool(
                (candidate["md_since_boundary"].to_numpy(np.float64) > 0.0).all()
            ),
            "pretruth_forbidden_value_access_zero": bool(
                not parent_audit["truth_columns_opened"]
                and not raw_audit["truth_columns_opened"]
                and not parent_audit["outer_fold_opened"]
                and not raw_audit["outer_fold_opened"]
                and not raw_audit["hidden_like_assignment_opened"]
            ),
        }
    )
    _, freeze_sha = freeze_target_free_generation(
        candidate,
        diagnostics,
        raw_manifest,
        config=config,
        config_path=config_path,
        parent_oof_path=parent_oof_path,
        parent_model_manifest_path=parent_model_manifest_path,
        parent_audit=parent_audit,
        raw_audit=raw_audit,
        technical_audit=technical,
        output_dir=output_dir,
    )
    evidence = {
        "parent": parent_audit,
        "raw": raw_audit,
        "technical": technical,
        "freeze_manifest_sha256": freeze_sha,
        "truth_access_before_freeze_count": 0,
    }
    return freeze_sha, evidence


# %% [markdown]
# ## 6. Stage B late-truth alignment and metrics

# %%
def load_hidden_like_assignments(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    verify_file_sha(
        path,
        str(nested(config, "validation.hidden_like_assignment_sha256")),
        "hidden-like assignment",
    )
    required = [
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]
    frame = pd.read_csv(path, usecols=required, dtype={"well_id": str})[required]
    frame = frame.rename(columns={"well_id": "well"})
    if frame["well"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    return frame


def align_late_truth(
    candidate: pd.DataFrame,
    parent_oof_path: Path,
    hidden_like_path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_file_sha(
        parent_oof_path,
        str(nested(config, "validation.parent_oof_sha256")),
        "exp287 parent OOF late-truth phase",
    )
    prediction_column = str(nested(config, "validation.parent_prediction_column"))
    parent = pd.read_parquet(parent_oof_path, columns=LATE_TRUTH_PARENT_COLUMNS)
    if list(parent.columns) != LATE_TRUTH_PARENT_COLUMNS:
        raise ValueError("parent late-truth projection schema/order mismatch")
    parent["id"] = parent["id"].astype(str)
    parent["well"] = parent["well"].astype(str)
    if parent["id"].duplicated().any():
        raise ValueError("parent late-truth id must be unique")
    parent = parent.rename(columns={prediction_column: "parent_late_pred_tvt"})
    aligned = candidate.merge(
        parent,
        on=["id", "well"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(aligned) != len(candidate) or not aligned["id"].equals(candidate["id"]):
        raise ValueError("late-truth identity/order alignment failed")
    numeric = aligned[
        [
            "parent_pred_tvt",
            "parent_late_pred_tvt",
            "actual_tvt",
            "outer_fold",
            "candidate_pred_tvt",
        ]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("late-truth aligned surface contains nonfinite values")
    parent_parity = float(
        np.max(
            np.abs(
                aligned["parent_pred_tvt"].to_numpy(np.float64)
                - aligned["parent_late_pred_tvt"].to_numpy(np.float64)
            )
        )
    )
    if parent_parity != 0.0:
        raise ValueError("Stage A/Stage B parent prediction parity failed")
    folds = sorted(aligned["outer_fold"].astype(int).unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"outer folds must be exactly 0..4: {folds}")
    fold_per_well = aligned.groupby("well")["outer_fold"].nunique()
    if not fold_per_well.eq(1).all():
        raise ValueError("a well appears in multiple outer folds")
    parent_cv = rmse(aligned["actual_tvt"], aligned["parent_pred_tvt"])
    expected_cv = float(nested(config, "validation.parent_oof_cv"))
    if abs(parent_cv - expected_cv) > float(nested(config, "guards.technical.parent_cv_atol")):
        raise ValueError(f"parent CV parity failed: {parent_cv} != {expected_cv}")
    assignment = load_hidden_like_assignments(hidden_like_path, config).set_index("well")
    for column in list(nested(config, "validation.hidden_like_scopes")):
        aligned[str(column)] = aligned["well"].map(assignment[str(column)])
        if aligned[str(column)].isna().any():
            raise ValueError(f"hidden-like assignment is incomplete for {column}")
    audit = {
        "rows": len(aligned),
        "wells": int(aligned["well"].nunique()),
        "unique_ids": int(aligned["id"].nunique()),
        "outer_folds": folds,
        "parent_cv": parent_cv,
        "stage_a_stage_b_parent_prediction_max_abs_difference": parent_parity,
        "hidden_like_assignment_sha256": sha256_file(hidden_like_path),
    }
    return aligned, audit


def metric_row(scope: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "scope": scope,
            "rows": 0,
            "wells": 0,
            "parent_rmse": None,
            "candidate_rmse": None,
            "gain_parent_minus_candidate": None,
            "delta_candidate_minus_parent": None,
        }
    parent_score = rmse(frame["actual_tvt"], frame["parent_pred_tvt"])
    candidate_score = rmse(frame["actual_tvt"], frame["candidate_pred_tvt"])
    return {
        "scope": scope,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "parent_rmse": parent_score,
        "candidate_rmse": candidate_score,
        "gain_parent_minus_candidate": parent_score - candidate_score,
        "delta_candidate_minus_parent": candidate_score - parent_score,
    }


def distance_bucket_masks(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> list[tuple[str, np.ndarray]]:
    distance = frame["md_since_boundary"].to_numpy(np.float64)
    pairs = list(nested(config, "validation.distance_buckets_md_ft"))
    expected = [
        [0.0, 64.0],
        [64.0, 128.0],
        [128.0, 240.0],
        [240.0, 480.0],
        [480.0, 1000.0],
        [1000.0, None],
    ]
    normalized = [
        [float(pair[0]), None if pair[1] is None else float(pair[1])] for pair in pairs
    ]
    if normalized != expected:
        raise ValueError("distance bucket contract changed")
    rows: list[tuple[str, np.ndarray]] = []
    for lower, upper in normalized:
        label = f"{int(lower)}+" if upper is None else f"{int(lower)}--{int(upper)}"
        mask = distance >= lower
        if upper is not None:
            mask &= distance < upper
        rows.append((label, mask))
    return rows


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        parent_score = rmse(group["actual_tvt"], group["parent_pred_tvt"])
        candidate_score = rmse(group["actual_tvt"], group["candidate_pred_tvt"])
        rows.append(
            {
                "well": str(well),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "rows": len(group),
                "gap_u": float(group["gap_u"].iloc[0]),
                "abs_gap_bucket": str(group["abs_gap_bucket"].iloc[0]),
                "parent_rmse": parent_score,
                "candidate_rmse": candidate_score,
                "delta_candidate_minus_parent": candidate_score - parent_score,
            }
        )
    return pd.DataFrame(rows)


def evaluate_fixed_candidate(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    pooled = pd.DataFrame([metric_row("pooled", frame)])
    fold = pd.DataFrame(
        [
            metric_row(
                f"outer_fold_{fold_id}", frame[frame["outer_fold"].eq(fold_id)]
            )
            for fold_id in range(5)
        ]
    )
    distance_rows = [
        metric_row(label, frame.loc[mask]) for label, mask in distance_bucket_masks(frame, config)
    ]
    boundary_mask = frame["md_since_boundary"].to_numpy(np.float64) < 240.0
    distance_rows.append(metric_row("0--240_primary", frame.loc[boundary_mask]))
    distance = pd.DataFrame(distance_rows)
    gap_order = ["[0,1)", "[1,2)", "[2,4)", "[4,8)", "[8,+inf)"]
    gap = pd.DataFrame(
        [metric_row(label, frame[frame["abs_gap_bucket"].eq(label)]) for label in gap_order]
    )
    sign = pd.DataFrame(
        [
            metric_row(label, frame[frame["gap_sign"].eq(label)])
            for label in ["negative", "zero", "positive"]
        ]
    )
    hidden_rows: list[dict[str, Any]] = []
    for column in list(nested(config, "validation.hidden_like_scopes")):
        hidden_rows.append(metric_row(str(column), frame[frame[str(column)].eq("valid")]))
    hidden = pd.DataFrame(hidden_rows)
    by_well = build_by_well_metrics(frame)

    scientific = dict(nested(config, "guards.scientific"))
    pooled_row = pooled.iloc[0]
    distance_lookup = distance.set_index("scope")
    hidden_lookup = hidden.set_index("scope")
    deltas = by_well["delta_candidate_minus_parent"]
    threshold = float(scientific["large_well_delta_threshold_ft"])
    improving_folds = int((fold["gain_parent_minus_candidate"] > 0.0).sum())
    worsened_large = int((deltas > threshold).sum())
    improved_large = int((deltas < -threshold).sum())
    checks = {
        "minimum_pooled_rmse_gain_ft": float(pooled_row["gain_parent_minus_candidate"])
        >= float(scientific["minimum_pooled_rmse_gain_ft"]),
        "minimum_improving_folds": improving_folds
        >= int(scientific["minimum_improving_folds"]),
        "minimum_boundary_0_240_rmse_gain_ft": float(
            distance_lookup.loc["0--240_primary", "gain_parent_minus_candidate"]
        )
        >= float(scientific["minimum_boundary_0_240_rmse_gain_ft"]),
        "maximum_240_480_rmse_delta_ft": float(
            distance_lookup.loc["240--480", "delta_candidate_minus_parent"]
        )
        <= float(scientific["maximum_240_480_rmse_delta_ft"]),
        "maximum_480_1000_rmse_delta_ft": float(
            distance_lookup.loc["480--1000", "delta_candidate_minus_parent"]
        )
        <= float(scientific["maximum_480_1000_rmse_delta_ft"]),
        "maximum_1000_plus_rmse_delta_ft": float(
            distance_lookup.loc["1000+", "delta_candidate_minus_parent"]
        )
        <= float(scientific["maximum_1000_plus_rmse_delta_ft"]),
        "maximum_hidden_like_spatial_rmse_delta_ft": float(
            hidden_lookup.loc[
                "verification_like_spatial_role", "delta_candidate_minus_parent"
            ]
        )
        <= float(scientific["maximum_hidden_like_spatial_rmse_delta_ft"]),
        "maximum_hidden_like_typewell_purged_rmse_delta_ft": float(
            hidden_lookup.loc[
                "verification_like_typewell_purged_role", "delta_candidate_minus_parent"
            ]
        )
        <= float(scientific["maximum_hidden_like_typewell_purged_rmse_delta_ft"]),
        "maximum_by_well_median_rmse_delta_ft": float(deltas.median())
        <= float(scientific["maximum_by_well_median_rmse_delta_ft"]),
        "maximum_by_well_p95_rmse_delta_ft": float(deltas.quantile(0.95))
        <= float(scientific["maximum_by_well_p95_rmse_delta_ft"]),
        "maximum_worst_well_rmse_delta_ft": float(deltas.max())
        <= float(scientific["maximum_worst_well_rmse_delta_ft"]),
        "large_worsened_well_count_not_above_large_improved_count": worsened_large
        <= improved_large,
    }
    summary = {
        "pooled_parent_rmse": float(pooled_row["parent_rmse"]),
        "pooled_candidate_rmse": float(pooled_row["candidate_rmse"]),
        "pooled_gain_parent_minus_candidate": float(
            pooled_row["gain_parent_minus_candidate"]
        ),
        "improving_folds": improving_folds,
        "by_well_delta_median": float(deltas.median()),
        "by_well_delta_p95": float(deltas.quantile(0.95)),
        "worst_well_delta": float(deltas.max()),
        "large_improved_wells": improved_large,
        "large_worsened_wells": worsened_large,
        "checks": checks,
        "all_scientific_checks_passed": bool(all(checks.values())),
    }
    tables = {
        "pooled_metrics.csv": pooled,
        "fold_metrics.csv": fold,
        "distance_bucket_metrics.csv": distance,
        "gap_bucket_metrics.csv": gap,
        "gap_sign_metrics.csv": sign,
        "hidden_like_metrics.csv": hidden,
    }
    return tables, by_well, summary


# %% [markdown]
# ## 7. Fixed technical/scientific decision gate

# %%
def build_decision(
    *,
    freeze: Mapping[str, Any],
    late_audit: Mapping[str, Any],
    scientific: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = int(nested(config, "validation.parent_oof_rows"))
    expected_wells = int(nested(config, "validation.parent_oof_wells"))
    stage_a = dict(freeze["technical_audit"])
    technical_checks = {
        "parent_oof_sha_match": str(freeze["parent_oof_sha256"])
        == str(nested(config, "validation.parent_oof_sha256")),
        "parent_model_manifest_sha_match": str(freeze["parent_model_manifest_sha256"])
        == str(nested(config, "validation.parent_model_manifest_sha256")),
        "row_count_parity": int(late_audit["rows"]) == expected_rows,
        "well_count_parity": int(late_audit["wells"]) == expected_wells,
        "unique_id_parity": int(late_audit["unique_ids"]) == expected_rows,
        "parent_cv_parity": abs(
            float(late_audit["parent_cv"])
            - float(nested(config, "validation.parent_oof_cv"))
        )
        <= float(nested(config, "guards.technical.parent_cv_atol")),
        "raw_suffix_parent_exact_alignment": bool(
            stage_a["raw_suffix_parent_exact_alignment"]
        ),
        "all_wells_prefix_suffix_contract": bool(
            stage_a["all_wells_prefix_suffix_contract"]
        ),
        "all_hidden_md_distance_positive": bool(
            stage_a["all_hidden_md_distance_positive"]
        ),
        "pretruth_forbidden_value_access_zero": bool(
            stage_a["pretruth_forbidden_value_access_zero"]
        ),
        "truth_access_before_freeze_zero": int(
            freeze["truth_access_before_freeze_count"]
        )
        == 0,
        "finite_output": bool(stage_a["finite_output"]),
        "maximum_abs_move_within_cap": float(stage_a["maximum_abs_move_tvt"])
        <= float(nested(config, "guards.technical.max_abs_move_ft"))
        + float(nested(config, "guards.technical.formula_atol")),
        "formula_parity": float(stage_a["maximum_formula_abs_difference"])
        <= float(nested(config, "guards.technical.formula_atol")),
        "abs_move_nonincreasing_per_well": bool(
            stage_a["abs_move_nonincreasing_per_well"]
        ),
        "first_hidden_abs_u_gap_nonincrease": bool(
            stage_a["first_hidden_abs_u_gap_nonincrease"]
        ),
        "opposite_sign_for_nonzero_gap": bool(stage_a["opposite_sign_for_nonzero_gap"]),
        "prediction_diagnostic_sha_readback": bool(
            freeze["prediction_and_diagnostic_sha_readback_verified"]
        ),
        "hidden_like_assignment_sha_match": str(
            late_audit["hidden_like_assignment_sha256"]
        )
        == str(nested(config, "validation.hidden_like_assignment_sha256")),
    }
    technical_pass = bool(all(technical_checks.values()))
    scientific_pass = bool(scientific["all_scientific_checks_passed"])
    return {
        "schema_version": "1.0.0",
        "variant": VARIANT_NAME,
        "technical_checks": technical_checks,
        "scientific_checks": dict(scientific["checks"]),
        "technical_pass": technical_pass,
        "scientific_pass": scientific_pass,
        "all_gates_passed": technical_pass and scientific_pass,
        "status": (
            "PASS_FOR_INFERENCE_REVIEW"
            if technical_pass and scientific_pass
            else "FAIL_CLOSE_NO_RESCUE"
        ),
        "failure_policy": "no cap/tau/threshold/gate/parent/distance/blend rescue",
        "inference_automatically_authorized": False,
    }


# %% [markdown]
# ## 8. Generated artifacts and reproducibility manifest

# %%
def write_evaluation_artifacts(
    *,
    tables: Mapping[str, pd.DataFrame],
    by_well: pd.DataFrame,
    decision: Mapping[str, Any],
    scientific: Mapping[str, Any],
    late_audit: Mapping[str, Any],
    freeze_sha: str,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    for name, frame in tables.items():
        write_canonical_csv(output_dir / name, frame)
    write_canonical_csv(output_dir / "by_well_metrics.csv", by_well)
    write_json(output_dir / "decision.json", dict(decision))
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": str(decision["status"]),
        "route": "ml_model",
        "variant": VARIANT_NAME,
        "parent": PARENT_EXPERIMENT,
        "parent_cv": float(scientific["pooled_parent_rmse"]),
        "candidate_cv": float(scientific["pooled_candidate_rmse"]),
        "pooled_gain_parent_minus_candidate": float(
            scientific["pooled_gain_parent_minus_candidate"]
        ),
        "improving_folds": int(scientific["improving_folds"]),
        "by_well_delta_median": float(scientific["by_well_delta_median"]),
        "by_well_delta_p95": float(scientific["by_well_delta_p95"]),
        "worst_well_delta": float(scientific["worst_well_delta"]),
        "decision": dict(decision),
        "rows": int(late_audit["rows"]),
        "wells": int(late_audit["wells"]),
        "stage_a_freeze_manifest_sha256": freeze_sha,
        "model_configs": 0,
        "trained_models": 0,
        "boosters": 0,
        "inference": False,
        "submission": False,
    }
    write_json(output_dir / "metrics.json", metrics)
    evaluation_input = {
        "schema_version": "1.0.0",
        "stage": "late_truth_evaluation_complete",
        "freeze_manifest_sha256_verified_before_truth_open": freeze_sha,
        "late_truth_audit": dict(late_audit),
    }
    write_json(output_dir / "evaluation_input_manifest.json", evaluation_input)
    artifact_names = [
        "scientific_contract.json",
        "stage_a_input_manifest.json",
        "input_schema_manifest.json",
        "raw_horizontal_ordered_manifest.csv",
        "target_free_candidate.parquet",
        "u_boundary_diagnostics_pretruth.csv",
        "target_free_freeze_manifest.json",
        "pooled_metrics.csv",
        "fold_metrics.csv",
        "distance_bucket_metrics.csv",
        "gap_bucket_metrics.csv",
        "gap_sign_metrics.csv",
        "hidden_like_metrics.csv",
        "by_well_metrics.csv",
        "decision.json",
        "metrics.json",
        "evaluation_input_manifest.json",
    ]
    artifact_sha = {name: sha256_file(output_dir / name) for name in artifact_names}
    reproducibility = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_NAME,
        "deterministic_anchor": False,
        "deterministic_transform_given_fixed_inputs": True,
        "seed_policy": str(nested(config, "reproducibility.seed_policy")),
        "stochastic_components": [],
        "stage_a_freeze_manifest_sha256": freeze_sha,
        "artifact_sha256": artifact_sha,
        "new_model_count": 0,
        "parent_model_manifest_sha256": str(
            nested(config, "validation.parent_model_manifest_sha256")
        ),
        "package_sha256": None,
        "kernel_version": None,
        "package_and_kernel_sha_status": "record_externally_after_separate_package_run_approval",
        "prediction_is_oof_postprocess_not_hidden_test": True,
        "submission_sha256": None,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    return metrics


def run_stage_b(
    *,
    parent_oof_path: Path,
    hidden_like_path: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    expected_freeze_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = verify_generation_freeze(
        output_dir,
        expected_freeze_manifest_sha256=expected_freeze_manifest_sha256,
    )
    if str(freeze["config_sha256"]) != sha256_file(find_config_path()):
        raise ValueError("config changed after target-free freeze")
    candidate = pd.read_parquet(
        output_dir / "target_free_candidate.parquet",
        columns=CANDIDATE_COLUMNS,
    )
    candidate["id"] = candidate["id"].astype(str)
    candidate["well"] = candidate["well"].astype(str)
    aligned, late_audit = align_late_truth(
        candidate,
        parent_oof_path,
        hidden_like_path,
        config,
    )
    tables, by_well, scientific = evaluate_fixed_candidate(aligned, config)
    decision = build_decision(
        freeze=freeze,
        late_audit=late_audit,
        scientific=scientific,
        config=config,
    )
    write_evaluation_artifacts(
        tables=tables,
        by_well=by_well,
        decision=decision,
        scientific=scientific,
        late_audit=late_audit,
        freeze_sha=expected_freeze_manifest_sha256,
        output_dir=output_dir,
        config=config,
    )
    return decision, {"late_truth": late_audit, "scientific": scientific}


# %% [markdown]
# ## 9. Execution orchestration

# %%
def run_stage0_saved_oof_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_scientific_contract(config, require_execution=True)
    config_path = find_config_path()
    output_dir = KAGGLE_WORKING_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    scientific_contract = build_scientific_contract(config)
    write_json(output_dir / "scientific_contract.json", scientific_contract)
    parent_oof_path = resolve_sha_matched_file(
        nested(config, "data.parent_oof_patterns"),
        expected_sha256=str(nested(config, "validation.parent_oof_sha256")),
        fallback_name="fold_safe_formation_oof_predictions.parquet",
        label="exp287 parent OOF",
    )
    parent_model_manifest_path = resolve_sha_matched_file(
        nested(config, "data.parent_model_manifest_patterns"),
        expected_sha256=str(nested(config, "validation.parent_model_manifest_sha256")),
        fallback_name="model_manifest.json",
        label="exp287 model manifest",
    )
    raw_train_dir = resolve_raw_train_dir(config)
    freeze_sha, stage_a = run_stage_a(
        parent_oof_path=parent_oof_path,
        parent_model_manifest_path=parent_model_manifest_path,
        raw_train_dir=raw_train_dir,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
    )
    # This resolver and file open occur only after the freeze manifest SHA was verified.
    hidden_like_path = resolve_sha_matched_file(
        nested(config, "data.hidden_like_assignment_patterns"),
        expected_sha256=str(nested(config, "validation.hidden_like_assignment_sha256")),
        fallback_name="exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
        label="hidden-like assignment",
    )
    decision, stage_b = run_stage_b(
        parent_oof_path=parent_oof_path,
        hidden_like_path=hidden_like_path,
        config=config,
        output_dir=output_dir,
        expected_freeze_manifest_sha256=freeze_sha,
    )
    reproducibility = json.loads(
        (output_dir / "reproducibility_manifest.json").read_text()
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "execution_contract": contract,
        "decision": decision["status"],
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_a_freeze_manifest_sha256": freeze_sha,
        "artifact_sha256": reproducibility["artifact_sha256"],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "model_training_or_submission_generated": False,
    }
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    display(pd.read_csv(output_dir / "pooled_metrics.csv"))
    display(pd.read_csv(output_dir / "fold_metrics.csv"))
    display(pd.read_csv(output_dir / "distance_bucket_metrics.csv"))
    display(pd.read_csv(output_dir / "hidden_like_metrics.csv"))
    display(pd.read_csv(output_dir / "by_well_metrics.csv").sort_values(
        "delta_candidate_minus_parent", ascending=False
    ).head(50))
    return summary


CONFIG_PATH = find_config_path()
CONFIG = read_yaml(CONFIG_PATH)
IMPLEMENTATION_CONTRACT = validate_scientific_contract(CONFIG, require_execution=False)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "parent": PARENT_EXPERIMENT,
            "route": CONFIG["experiment"]["route"],
            "status": CONFIG["experiment"]["status"],
            "config_path": str(CONFIG_PATH),
            "implementation_contract": IMPLEMENTATION_CONTRACT,
            "run_stage_a": CONFIG["execution"]["run_stage_a_generation"],
            "run_stage_b": CONFIG["execution"]["run_stage_b_evaluation"],
            "kaggle_push_approved": CONFIG["execution"]["kaggle_push_approved"],
        },
        sort_keys=True,
        indent=2,
    )
)

if not IMPORT_ONLY:
    if CONFIG["execution"]["active_stage"] == "stage0_saved_oof_audit":
        RUN_SUMMARY = run_stage0_saved_oof_audit(CONFIG)
    else:
        print(
            "Implementation-only state: no parent OOF generation/evaluation, model, "
            "inference, or submission execution was started."
        )
