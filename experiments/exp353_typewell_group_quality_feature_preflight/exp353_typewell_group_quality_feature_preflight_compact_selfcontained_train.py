# %% [markdown]
# # exp353 Type-Well group quality feature preflight
#
# This zero-booster diagnostic builds six fold-safe Type-Well group quality
# features from outer-train wells only.  The feature manifest is frozen before
# the saved exp148 by-well OOF error is opened.  The readout then tests whether
# the real group residual-sigma feature is associated with exp148 error more
# strongly than a stable group-label shuffle.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, SHA, and output helpers
# 3. Frozen scientific contract and input resolution
# 4. exp148-compatible fold and native Type-Well membership
# 5. Outer-train GR calibration and group-quality priors
# 6. Truth-free feature manifest and deterministic shuffle control
# 7. Late exp148 OOF-error attachment and fixed Stage 0 gate
# 8. Full Kaggle CPU orchestration and generated-artifact guards
# 9. Setup, configuration, and contract preview
# 10. Run the preflight and report generated artifacts

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

EXPERIMENT_NAME = "exp353_typewell_group_quality_feature_preflight"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SHA256_LENGTH = 64
FEATURE_COLUMNS = [
    "typewell_group_log_support_wells",
    "typewell_group_log_support_rows",
    "typewell_group_residual_sigma",
    "typewell_group_fit_rmse",
    "typewell_group_bias_abs_gr50",
    "typewell_group_prior_available",
]
QUALITY_COLUMNS = [
    "residual_sigma_mad",
    "fit_rmse",
    "bias_at_gr50",
]
CONTROL_NAMES = ["real_native_group", "stable_group_label_shuffle_within_fold"]


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return shell is not None


# %% [markdown]
# ## 2. Runtime, configuration, SHA, and output helpers

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
    if not isinstance(value, (str, bytes)):
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    for start in [Path.cwd(), KAGGLE_WORKING_ROOT]:
        for candidate in (start, *start.parents):
            if (candidate / "project.yml").exists():
                return candidate
    return Path.cwd()


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = [
        Path.cwd() / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp353 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_digest(*parts: Any) -> str:
    return mapping_sha256([str(part) for part in parts])


def stable_int(*parts: Any, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    return int(stable_digest(*parts)[:16], 16) % modulo


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    return mapping_sha256([(str(column), str(frame[column].dtype)) for column in frame.columns])


def dataframe_content_sha(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> str:
    work = frame.copy()
    if sort_columns:
        work = work.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    if columns:
        work = work.loc[:, list(columns)]
    digest = hashlib.sha256()
    digest.update(dataframe_schema_sha(work).encode())
    for row in work.itertuples(index=False, name=None):
        payload = json.dumps(to_jsonable(row), separators=(",", ":"), ensure_ascii=True)
        digest.update(payload.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    sort_columns: Sequence[str],
) -> dict[str, Any]:
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": len(ordered),
        "columns": len(ordered.columns),
        "schema_sha256": dataframe_schema_sha(ordered),
        "content_sha256": dataframe_content_sha(ordered),
        "raw_sha256": sha256_path(path),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }


def resolve_existing_file(
    filename: str,
    candidates: Sequence[str],
) -> Path:
    root = project_root()
    attempted: list[Path] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in [candidate, root / candidate, Path.cwd() / candidate]:
            attempted.append(path)
            if path.is_file() and path.stat().st_size > 0:
                return path
            nested = path / filename
            attempted.append(nested)
            if nested.is_file() and nested.stat().st_size > 0:
                return nested
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(
        f"could not resolve {filename}; checked {[str(path) for path in attempted]}"
    )


def train_data_dir(config: Mapping[str, Any]) -> Path:
    candidates = [
        Path(str(value))
        for value in list(get_nested(config, "data.train_root_candidates") or [])
    ]
    root = project_root()
    candidates.extend(root / path for path in candidates if not path.is_absolute())
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            [
                KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
                KAGGLE_INPUT_ROOT
                / "competitions"
                / "rogii-wellbore-geology-prediction"
                / "train",
            ]
        )
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("**/train")))
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate
    raise FileNotFoundError("ROGII train directory with horizontal wells was not found")


# %% [markdown]
# ## 3. Frozen scientific contract and input resolution

# %%
def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    fixed = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "ml_model",
        "lineage.parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "implementation.enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.strategy": "exp148_grouped_outer5_typewell_quality_preflight",
        "validation.n_folds": 5,
        "validation.control": "saved_exp148_oof_no_retrain",
        "validation.fit_scope": "outer_train_wells_only",
        "validation.truth_attachment": "after_group_prior_and_feature_content_sha_freeze",
        "model.name": "exp148_typewell_group_quality_addonly_preflight",
        "model.features.fallback": "global_outer_train_prior_with_availability_zero",
        "model.features.posthoc_column_selection": False,
        "model.calibration.typewell_group": "native_overlap_1",
        "model.calibration.minimum_peer_wells": 1,
        "model.calibration.minimum_peer_effective_rows": 32,
        "execution_contract.stage_0.preflight_variants": 1,
        "execution_contract.stage_0.negative_controls": 1,
        "execution_contract.stage_0.folds": 5,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.kaggle.enable_gpu_stage_0": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"exp353 fixes {key}={expected!r}, got {actual!r}")
    if list(get_nested(config, "model.features.add_only") or []) != FEATURE_COLUMNS:
        raise ValueError("exp353 fixes the ordered six-column feature schema")
    if list(get_nested(config, "model.stage_0.controls") or []) != CONTROL_NAMES:
        raise ValueError("exp353 fixes real native group and one stable shuffle control")
    forbidden = set(get_nested(config, "model.forbidden") or [])
    if forbidden != {
        "same_readout_feature_selection",
        "group_definition_grid",
        "calibrated_gr_value",
        "direct_tvt_correction",
        "hard_group_router",
    }:
        raise ValueError("exp353 forbidden-operation contract changed")
    expected_hash_keys = [
        "data.exp065_membership.expected_raw_sha256",
        "data.exp148_artifacts.expected_summary_raw_sha256",
        "data.exp148_artifacts.expected_by_well_raw_sha256",
    ]
    for key in expected_hash_keys:
        if len(str(get_nested(config, key) or "")) != SHA256_LENGTH:
            raise ValueError(f"exp353 requires pinned SHA256 at {key}")


def validate_run_approval(config: Mapping[str, Any]) -> None:
    if not bool(get_nested(config, "execution.run_stage_0")):
        raise RuntimeError("Stage 0 execution is disabled until separately approved")
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("Kaggle package/push/run is not approved")
    if not bool(get_nested(config, "runtime.kaggle.train_run_on_push")):
        raise RuntimeError("approved Stage 0 requires train_run_on_push=true")
    if bool(get_nested(config, "execution.run_stage_1")):
        raise RuntimeError("Stage 1 GPU training remains separately gated and disabled")


def resolve_membership_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.exp065_membership") or {})
    return resolve_existing_file(str(spec["filename"]), list(spec.get("candidates") or []))


def resolve_exp148_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    spec = dict(get_nested(config, "data.exp148_artifacts") or {})
    filenames = dict(spec.get("filenames") or {})
    roots = list(spec.get("root_candidates") or [])
    return {
        name: resolve_existing_file(str(filename), roots)
        for name, filename in sorted(filenames.items())
    }


def verify_pinned_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA mismatch: expected={expected_sha256}, actual={actual}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual,
    }


# %% [markdown]
# ## 4. exp148-compatible fold and native Type-Well membership

# %%
def list_train_wells(data_dir: Path) -> list[str]:
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv")
        for path in data_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv")
        for path in data_dir.glob("*__typewell.csv")
    }
    wells = sorted(horizontal & typewell)
    if not wells:
        raise FileNotFoundError(f"no paired train wells found under {data_dir}")
    return wells


def count_exp148_score_rows(path: Path) -> int:
    # exp148 trains on the evaluation surface only: rows where TVT_input is
    # missing.  Reading this target-free column preserves the late-error rule.
    frame = pd.read_csv(path, usecols=["TVT_input"])
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce")
    return int(tvt_input.isna().sum())


def build_row_count_manifest(wells: Sequence[str], data_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "well_id": str(well_id),
            "rows": count_exp148_score_rows(
                data_dir / f"{well_id}__horizontal_well.csv"
            ),
        }
        for well_id in sorted(map(str, wells))
    ]
    result = pd.DataFrame(rows)
    if (result["rows"] <= 0).any() or result["well_id"].duplicated().any():
        raise ValueError("row-count manifest requires one positive row count per well")
    return result


def build_exp148_fold_manifest(
    row_counts: pd.DataFrame,
    n_folds: int,
) -> pd.DataFrame:
    required = {"well_id", "rows"}
    if not required.issubset(row_counts.columns):
        raise ValueError(f"row counts missing {sorted(required - set(row_counts.columns))}")
    work = row_counts[["well_id", "rows"]].copy()
    work["well_id"] = work["well_id"].astype(str)
    work["rows"] = pd.to_numeric(work["rows"], errors="raise").astype(np.int64)
    if work["well_id"].duplicated().any() or (work["rows"] <= 0).any():
        raise ValueError("fold manifest requires unique wells and positive row counts")
    work = work.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    weights = work["rows"].to_numpy(np.int64)
    # This is sklearn GroupKFold(shuffle=False): np.unique sorts group labels,
    # then groups are distributed largest-first to the currently lightest fold.
    ordered_indices = np.argsort(weights)[::-1]
    fold_weights = np.zeros(int(n_folds), dtype=np.int64)
    group_to_fold = np.zeros(len(work), dtype=np.int64)
    for group_index in ordered_indices:
        lightest_fold = int(np.argmin(fold_weights))
        fold_weights[lightest_fold] += int(weights[group_index])
        group_to_fold[group_index] = lightest_fold
    work["fold"] = group_to_fold.astype(int)
    work["fold_rows"] = work["fold"].map(
        work.groupby("fold", sort=True)["rows"].sum().to_dict()
    )
    return work[["well_id", "rows", "fold", "fold_rows"]]


def load_native_membership(
    path: Path,
    config: Mapping[str, Any],
    expected_wells: Sequence[str],
) -> pd.DataFrame:
    verify_pinned_file(
        path,
        str(get_nested(config, "data.exp065_membership.expected_raw_sha256")),
        "exp065 membership",
    )
    source = pd.read_csv(path, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    if not required.issubset(source.columns):
        raise ValueError(f"membership missing {sorted(required - set(source.columns))}")
    calibration = dict(get_nested(config, "model.calibration") or {})
    selected = source[
        source["method"].eq(str(calibration["group_method"]))
        & source["threshold"].eq(str(calibration["group_threshold"]))
    ][["well_id", "cluster_id", "cluster_size"]].copy()
    selected = selected.rename(columns={"cluster_id": "group_id"})
    selected["cluster_size"] = pd.to_numeric(
        selected["cluster_size"], errors="raise"
    ).astype(int)
    selected["group_scheme"] = str(calibration["typewell_group"])
    selected = selected[
        ["well_id", "group_scheme", "group_id", "cluster_size"]
    ].sort_values("well_id", kind="mergesort")
    expected = set(map(str, expected_wells))
    actual = set(selected["well_id"].astype(str))
    if selected["well_id"].duplicated().any() or actual != expected:
        raise ValueError(
            f"native membership well mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return selected.reset_index(drop=True)


# %% [markdown]
# ## 5. Outer-train GR calibration and group-quality priors

# %%
@dataclass(frozen=True)
class GrPairs:
    typewell_gr: np.ndarray
    horizontal_gr: np.ndarray


def collapse_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    work = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    work = work.groupby("TVT", sort=True, as_index=False)["GR"].median()
    tvt = work["TVT"].to_numpy(np.float64)
    gr = work["GR"].to_numpy(np.float64)
    if len(tvt) < 2 or not bool((np.diff(tvt) > 0).all()):
        raise ValueError("typewell requires at least two increasing finite TVT values")
    return tvt, gr


def interpolate_no_extrapolation(
    query_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_tvt, dtype=np.float64)
    result = np.full(len(query), np.nan, dtype=np.float64)
    inside = (
        np.isfinite(query)
        & (query >= typewell_tvt[0])
        & (query <= typewell_tvt[-1])
    )
    if bool(inside.any()):
        result[inside] = np.interp(query[inside], typewell_tvt, typewell_gr)
    return result


def load_label_pairs(well_id: str, data_dir: Path) -> GrPairs:
    horizontal = pd.read_csv(
        data_dir / f"{well_id}__horizontal_well.csv",
        usecols=["GR", "TVT"],
    )
    typewell = pd.read_csv(
        data_dir / f"{well_id}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    typewell_tvt, typewell_gr = collapse_typewell(typewell)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    true_tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    paired = interpolate_no_extrapolation(true_tvt, typewell_tvt, typewell_gr)
    finite = np.isfinite(paired) & np.isfinite(horizontal_gr) & np.isfinite(true_tvt)
    return GrPairs(
        typewell_gr=paired[finite],
        horizontal_gr=horizontal_gr[finite],
    )


def robust_mad_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan
    center = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - center)))


def weighted_affine(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    root_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
    solution, *_ = np.linalg.lstsq(
        design * root_weight[:, None],
        y * root_weight,
        rcond=None,
    )
    return float(solution[0]), float(solution[1])


def fit_huber_affine_quality(
    typewell_gr: np.ndarray,
    horizontal_gr: np.ndarray,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    x_all = np.asarray(typewell_gr, dtype=np.float64)
    y_all = np.asarray(horizontal_gr, dtype=np.float64)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    x = x_all[finite]
    y = y_all[finite]
    support = int(len(x))
    minimum = int(calibration["overlap_rows_min"])
    base = {
        "support_rows": support,
        "fit_available": False,
        "bias_at_gr50": math.nan,
        "residual_sigma_mad": math.nan,
        "fit_rmse": math.nan,
        "fallback_reason": "insufficient_pairs" if support < minimum else None,
    }
    if support < minimum or float(np.std(x)) <= 1.0e-12:
        if support >= minimum:
            base["fallback_reason"] = "degenerate_typewell_gr"
        return base
    slope, intercept = weighted_affine(x, y, np.ones(support, dtype=np.float64))
    delta = float(calibration["huber_delta"])
    tolerance = float(calibration["huber_tolerance"])
    for _iteration in range(int(calibration["huber_max_iterations"])):
        residual = y - (slope * x + intercept)
        scale = robust_mad_scale(residual)
        if not np.isfinite(scale) or scale <= 1.0e-12:
            break
        threshold = delta * scale
        absolute = np.abs(residual)
        weights = np.ones(support, dtype=np.float64)
        large = absolute > threshold
        weights[large] = threshold / absolute[large]
        next_slope, next_intercept = weighted_affine(x, y, weights)
        relative = max(
            abs(next_slope - slope) / max(abs(slope), 1.0),
            abs(next_intercept - intercept) / max(abs(intercept), 1.0),
        )
        slope, intercept = next_slope, next_intercept
        if relative <= tolerance:
            break
    support_k = float(calibration["identity_shrinkage_support_k"])
    alpha = support / (support + support_k)
    slope = 1.0 + alpha * (slope - 1.0)
    intercept = alpha * intercept
    residual = y - (slope * x + intercept)
    return {
        "support_rows": support,
        "fit_available": True,
        "bias_at_gr50": slope * 50.0 + intercept - 50.0,
        "residual_sigma_mad": robust_mad_scale(residual),
        "fit_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "fallback_reason": None,
    }


def fit_outer_train_quality(
    outer_train: Sequence[str],
    data_dir: Path,
    fold: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    calibration = dict(get_nested(config, "model.calibration") or {})
    rows: list[dict[str, Any]] = []
    for index, well_id in enumerate(sorted(map(str, outer_train)), start=1):
        pairs = load_label_pairs(well_id, data_dir)
        rows.append(
            {
                "fold": int(fold),
                "well_id": well_id,
                **fit_huber_affine_quality(
                    pairs.typewell_gr,
                    pairs.horizontal_gr,
                    calibration,
                ),
            }
        )
        if index % 100 == 0:
            print(
                f"fold={fold} fitted_outer_train_wells={index}/{len(outer_train)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def stable_shuffled_group_lookup(
    source_wells: Sequence[str],
    real_lookup: Mapping[str, str],
    fold: int,
) -> dict[str, str]:
    eligible = [str(well) for well in source_wells if str(well) in real_lookup]
    ordered = sorted(eligible, key=lambda well: stable_digest("exp353_group_shuffle", fold, well))
    if len(ordered) < 2:
        return {well: str(real_lookup[well]) for well in ordered}
    labels = [str(real_lookup[well]) for well in ordered]
    offset = 1 + stable_int(
        "exp353_group_shuffle_offset",
        fold,
        modulo=len(labels) - 1,
    )
    rotated = labels[offset:] + labels[:offset]
    return dict(zip(ordered, rotated, strict=True))


def aggregate_group_quality_priors(
    well_quality: pd.DataFrame,
    source_group_by_well: Mapping[str, str],
    *,
    fold: int,
    control: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = well_quality[well_quality["well_id"].isin(source_group_by_well)].copy()
    work["group_id"] = work["well_id"].map(source_group_by_well)
    work = work[
        work["fit_available"].fillna(False) & work["group_id"].notna()
    ].copy()
    if work.empty:
        raise ValueError(f"fold={fold} control={control} has no usable outer-train quality")
    minimum_wells = int(get_nested(config, "model.calibration.minimum_peer_wells"))
    minimum_rows = int(
        get_nested(config, "model.calibration.minimum_peer_effective_rows")
    )
    rows: list[dict[str, Any]] = []
    for group_id, group in work.groupby("group_id", sort=True):
        source_wells = sorted(group["well_id"].astype(str).tolist())
        support_rows = int(group["support_rows"].sum())
        available = len(source_wells) >= minimum_wells and support_rows >= minimum_rows
        rows.append(
            {
                "fold": int(fold),
                "control": str(control),
                "group_id": str(group_id),
                "source_wells": len(source_wells),
                "support_rows": support_rows,
                "available": bool(available),
                "source_wells_sha256": mapping_sha256(source_wells),
                "residual_sigma_mad": (
                    float(group["residual_sigma_mad"].median()) if available else math.nan
                ),
                "fit_rmse": float(group["fit_rmse"].median()) if available else math.nan,
                "bias_at_gr50": (
                    float(group["bias_at_gr50"].median()) if available else math.nan
                ),
            }
        )
    available_work = work[
        np.isfinite(work[QUALITY_COLUMNS].to_numpy(np.float64)).all(axis=1)
    ].copy()
    if available_work.empty:
        raise ValueError(f"fold={fold} control={control} has no finite global fallback")
    global_prior = {
        "source_wells": int(available_work["well_id"].nunique()),
        "support_rows": int(available_work["support_rows"].sum()),
        "residual_sigma_mad": float(available_work["residual_sigma_mad"].median()),
        "fit_rmse": float(available_work["fit_rmse"].median()),
        "bias_at_gr50": float(available_work["bias_at_gr50"].median()),
        "source_wells_sha256": mapping_sha256(
            sorted(available_work["well_id"].astype(str).tolist())
        ),
    }
    return pd.DataFrame(rows), global_prior


# %% [markdown]
# ## 6. Truth-free feature manifest and deterministic shuffle control

# %%
def build_valid_feature_rows(
    outer_valid: Sequence[str],
    target_group_by_well: Mapping[str, str],
    priors: pd.DataFrame,
    global_prior: Mapping[str, Any],
    *,
    fold: int,
    control: str,
    fit_wells: Sequence[str],
) -> pd.DataFrame:
    fit_set = set(map(str, fit_wells))
    valid_set = set(map(str, outer_valid))
    overlap = sorted(fit_set & valid_set)
    if overlap:
        raise ValueError(f"outer-valid wells leaked into prior fit: {overlap[:5]}")
    available = priors[priors["available"].fillna(False)].copy()
    prior_map = {
        str(row["group_id"]): row.to_dict()
        for _, row in available.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for well_id in sorted(valid_set):
        group_id = str(target_group_by_well[well_id])
        prior = prior_map.get(group_id)
        is_available = prior is not None
        selected = prior if prior is not None else dict(global_prior)
        rows.append(
            {
                "fold": int(fold),
                "well_id": well_id,
                "group_id": group_id,
                "control": str(control),
                "fallback_reason": (
                    "exact_outer_train_group"
                    if is_available
                    else "global_outer_train_prior_unseen_group"
                ),
                "prior_source_wells_sha256": str(selected["source_wells_sha256"]),
                "fit_well_overlap": 0,
                "outer_valid_truth_rows_before_feature_freeze": 0,
                "typewell_group_log_support_wells": math.log1p(
                    int(selected["source_wells"])
                ),
                "typewell_group_log_support_rows": math.log1p(
                    int(selected["support_rows"])
                ),
                "typewell_group_residual_sigma": float(
                    selected["residual_sigma_mad"]
                ),
                "typewell_group_fit_rmse": float(selected["fit_rmse"]),
                "typewell_group_bias_abs_gr50": abs(float(selected["bias_at_gr50"])),
                "typewell_group_prior_available": float(is_available),
            }
        )
    result = pd.DataFrame(rows)
    if not np.isfinite(result[FEATURE_COLUMNS].to_numpy(np.float64)).all():
        raise ValueError(f"fold={fold} control={control} produced non-finite features")
    return result


def build_feature_manifest(
    folds: pd.DataFrame,
    membership: pd.DataFrame,
    data_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_lookup = dict(
        zip(membership["well_id"].astype(str), membership["group_id"].astype(str), strict=True)
    )
    feature_frames: list[pd.DataFrame] = []
    prior_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    for fold in range(int(get_nested(config, "validation.n_folds"))):
        outer_valid = sorted(
            folds.loc[folds["fold"].eq(fold), "well_id"].astype(str).tolist()
        )
        outer_train = sorted(
            folds.loc[~folds["fold"].eq(fold), "well_id"].astype(str).tolist()
        )
        if set(outer_train) & set(outer_valid):
            raise AssertionError("outer fold train/valid overlap")
        print(
            f"fold={fold} outer_train={len(outer_train)} outer_valid={len(outer_valid)}",
            flush=True,
        )
        quality = fit_outer_train_quality(outer_train, data_dir, fold, config)
        fit_frames.append(quality)
        control_lookups = {
            "real_native_group": {well: group_lookup[well] for well in outer_train},
            "stable_group_label_shuffle_within_fold": stable_shuffled_group_lookup(
                outer_train,
                group_lookup,
                fold,
            ),
        }
        for control in CONTROL_NAMES:
            priors, global_prior = aggregate_group_quality_priors(
                quality,
                control_lookups[control],
                fold=fold,
                control=control,
                config=config,
            )
            priors["global_source_wells"] = int(global_prior["source_wells"])
            priors["global_support_rows"] = int(global_prior["support_rows"])
            priors["global_residual_sigma_mad"] = float(
                global_prior["residual_sigma_mad"]
            )
            priors["global_fit_rmse"] = float(global_prior["fit_rmse"])
            priors["global_bias_at_gr50"] = float(global_prior["bias_at_gr50"])
            priors["global_source_wells_sha256"] = str(
                global_prior["source_wells_sha256"]
            )
            prior_frames.append(priors)
            feature_frames.append(
                build_valid_feature_rows(
                    outer_valid,
                    group_lookup,
                    priors,
                    global_prior,
                    fold=fold,
                    control=control,
                    fit_wells=outer_train,
                )
            )
    features = pd.concat(feature_frames, ignore_index=True).sort_values(
        ["control", "fold", "well_id"], kind="mergesort"
    ).reset_index(drop=True)
    priors = pd.concat(prior_frames, ignore_index=True).sort_values(
        ["control", "fold", "group_id"], kind="mergesort"
    ).reset_index(drop=True)
    fit_quality = pd.concat(fit_frames, ignore_index=True).sort_values(
        ["fold", "well_id"], kind="mergesort"
    ).reset_index(drop=True)
    expected_rows = len(folds) * len(CONTROL_NAMES)
    if len(features) != expected_rows:
        raise ValueError(f"feature manifest rows {len(features)} != {expected_rows}")
    if features.duplicated(["control", "well_id"]).any():
        raise ValueError("each control must contain exactly one row per held-out well")
    return features, priors, fit_quality


def add_freeze_hashes(features: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    result = features.copy()
    fold_hashes: dict[tuple[str, int], str] = {}
    for (control, fold), group in result.groupby(["control", "fold"], sort=True):
        fold_hashes[(str(control), int(fold))] = dataframe_content_sha(
            group.drop(
                columns=[
                    column
                    for column in [
                        "fold_feature_freeze_sha256",
                        "feature_manifest_freeze_sha256",
                    ]
                    if column in group.columns
                ]
            ),
            sort_columns=["well_id"],
        )
    result["fold_feature_freeze_sha256"] = [
        fold_hashes[(str(control), int(fold))]
        for control, fold in zip(result["control"], result["fold"], strict=True)
    ]
    manifest_sha = dataframe_content_sha(
        result,
        sort_columns=["control", "fold", "well_id"],
    )
    result["feature_manifest_freeze_sha256"] = manifest_sha
    return result, manifest_sha


# %% [markdown]
# ## 7. Late exp148 OOF-error attachment and fixed Stage 0 gate

# %%
def load_exp148_error_after_freeze(
    paths: Mapping[str, Path],
    folds: pd.DataFrame,
    feature_manifest_freeze_sha256: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(feature_manifest_freeze_sha256) != SHA256_LENGTH:
        raise ValueError("exp148 error cannot be opened without a complete feature freeze SHA")
    summary_manifest = verify_pinned_file(
        paths["summary"],
        str(get_nested(config, "data.exp148_artifacts.expected_summary_raw_sha256")),
        "exp148 summary",
    )
    by_well_manifest = verify_pinned_file(
        paths["by_well"],
        str(get_nested(config, "data.exp148_artifacts.expected_by_well_raw_sha256")),
        "exp148 by-well",
    )
    summary = json.loads(paths["summary"].read_text())
    if summary.get("experiment") != "exp148_learned_likelihood_fulltrain_addonly_on_exp092":
        raise ValueError("resolved exp148 summary belongs to another experiment")
    expected_rmse = float(get_nested(config, "data.exp148_artifacts.expected_lgb_mean_rmse"))
    actual_rmse = float(get_nested(summary, "best_lgb_mean_by_rmse_tvt.rmse_tvt"))
    if abs(actual_rmse - expected_rmse) > 1.0e-12:
        raise ValueError(f"exp148 control RMSE mismatch: {actual_rmse} != {expected_rmse}")
    frame = pd.read_csv(paths["by_well"])
    selected = frame[
        frame["variant"].eq(str(get_nested(config, "data.exp148_artifacts.variant")))
        & frame["mode"].eq(str(get_nested(config, "data.exp148_artifacts.mode")))
        & frame["model"].eq(str(get_nested(config, "data.exp148_artifacts.model")))
    ][["well", "rows", "rmse_tvt"]].copy()
    selected = selected.rename(
        columns={"well": "well_id", "rmse_tvt": "exp148_well_rmse_ft"}
    )
    selected["well_id"] = selected["well_id"].astype(str)
    selected["rows"] = pd.to_numeric(selected["rows"], errors="raise").astype(np.int64)
    selected["exp148_well_rmse_ft"] = pd.to_numeric(
        selected["exp148_well_rmse_ft"], errors="raise"
    )
    expected_wells = set(folds["well_id"].astype(str))
    actual_wells = set(selected["well_id"])
    if (
        selected["well_id"].duplicated().any()
        or actual_wells != expected_wells
        or len(selected) != int(get_nested(config, "data.exp148_artifacts.expected_wells"))
    ):
        raise ValueError("exp148 selected by-well control does not match frozen wells")
    parity = selected.merge(
        folds[["well_id", "rows", "fold"]],
        on="well_id",
        how="left",
        suffixes=("_exp148", "_raw"),
        validate="one_to_one",
    )
    if not parity["rows_exp148"].eq(parity["rows_raw"]).all():
        raise ValueError("exp148 by-well row counts differ from raw fold manifest")
    if not np.isfinite(parity["exp148_well_rmse_ft"].to_numpy(np.float64)).all():
        raise ValueError("exp148 by-well RMSE contains non-finite values")
    return parity[["well_id", "fold", "rows_raw", "exp148_well_rmse_ft"]].rename(
        columns={"rows_raw": "rows"}
    ), {
        "summary": summary_manifest,
        "by_well": by_well_manifest,
        "selected_wells": len(parity),
        "selected_rows": int(parity["rows_exp148"].sum()),
        "control_rmse_ft": actual_rmse,
        "feature_manifest_freeze_sha256": feature_manifest_freeze_sha256,
    }


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if int(finite.sum()) < 3:
        return math.nan
    value = spearmanr(x_array[finite], y_array[finite]).statistic
    return float(value) if np.isfinite(value) else math.nan


def attach_error_and_compute_readout(
    features: pd.DataFrame,
    exp148_error: pd.DataFrame,
    feature_manifest_freeze_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(feature_manifest_freeze_sha256) != SHA256_LENGTH:
        raise ValueError("late error attachment requires a complete feature freeze SHA")
    if not features["feature_manifest_freeze_sha256"].eq(
        feature_manifest_freeze_sha256
    ).all():
        raise ValueError("feature manifest is not consistently frozen")
    scored = features.merge(
        exp148_error[["well_id", "fold", "exp148_well_rmse_ft"]],
        on=["well_id", "fold"],
        how="left",
        validate="many_to_one",
    )
    if scored["exp148_well_rmse_ft"].isna().any():
        raise ValueError("late exp148 error join did not cover every feature row")
    metric_rows: list[dict[str, Any]] = []
    for control in CONTROL_NAMES:
        control_rows = scored[scored["control"].eq(control)]
        for fold_label, scope in [
            *((int(fold), part) for fold, part in control_rows.groupby("fold", sort=True)),
            ("pooled", control_rows),
        ]:
            metric_rows.append(
                {
                    "control": control,
                    "fold": fold_label,
                    "wells": len(scope),
                    "coverage": float(
                        scope["typewell_group_prior_available"].mean()
                    ),
                    "fallback_fraction": float(
                        1.0 - scope["typewell_group_prior_available"].mean()
                    ),
                    "residual_sigma_vs_exp148_well_rmse_spearman": safe_spearman(
                        scope["typewell_group_residual_sigma"],
                        scope["exp148_well_rmse_ft"],
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics["fold"] = metrics["fold"].astype(str)
    real = scored[scored["control"].eq("real_native_group")].copy()
    q1 = float(real["typewell_group_residual_sigma"].quantile(0.25))
    q3 = float(real["typewell_group_residual_sigma"].quantile(0.75))
    lower = real[real["typewell_group_residual_sigma"] <= q1]
    upper = real[real["typewell_group_residual_sigma"] >= q3]
    quartile = {
        "residual_sigma_q1_threshold": q1,
        "residual_sigma_q3_threshold": q3,
        "q1_wells": len(lower),
        "q4_wells": len(upper),
        "q1_exp148_well_rmse_mean_ft": float(lower["exp148_well_rmse_ft"].mean()),
        "q4_exp148_well_rmse_mean_ft": float(upper["exp148_well_rmse_ft"].mean()),
    }
    quartile["q4_minus_q1_exp148_well_rmse_ft"] = (
        quartile["q4_exp148_well_rmse_mean_ft"]
        - quartile["q1_exp148_well_rmse_mean_ft"]
    )
    return scored, metrics, quartile


def one_metric(
    metrics: pd.DataFrame,
    control: str,
    fold: str | int,
) -> pd.Series:
    selected = metrics[
        metrics["control"].eq(control) & metrics["fold"].astype(str).eq(str(fold))
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for {(control, fold)}, got {len(selected)}")
    return selected.iloc[0]


def evaluate_stage_0_gate(
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    quartile: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = dict(get_nested(config, "model.stage_0.pass_requires_all") or {})
    real_pooled = one_metric(metrics, "real_native_group", "pooled")
    shuffle_pooled = one_metric(
        metrics,
        "stable_group_label_shuffle_within_fold",
        "pooled",
    )
    real_folds = metrics[
        metrics["control"].eq("real_native_group")
        & ~metrics["fold"].astype(str).eq("pooled")
    ]
    real_spearman = float(
        real_pooled["residual_sigma_vs_exp148_well_rmse_spearman"]
    )
    shuffle_spearman = float(
        shuffle_pooled["residual_sigma_vs_exp148_well_rmse_spearman"]
    )
    real_minus_shuffle = real_spearman - shuffle_spearman
    positive_folds = int(
        (
            real_folds["residual_sigma_vs_exp148_well_rmse_spearman"]
            .fillna(-np.inf)
            .gt(0.0)
        ).sum()
    )
    feature_finite = bool(
        np.isfinite(features[FEATURE_COLUMNS].to_numpy(np.float64)).all()
    )
    truth_before_freeze = int(
        features["outer_valid_truth_rows_before_feature_freeze"].sum()
    )
    checks = {
        "minimum_coverage": bool(
            float(real_pooled["coverage"]) >= float(gates["minimum_coverage"])
        ),
        "maximum_fallback_fraction": bool(
            float(real_pooled["fallback_fraction"])
            <= float(gates["maximum_fallback_fraction"])
        ),
        "require_all_features_finite": feature_finite,
        "minimum_residual_sigma_vs_exp148_well_rmse_spearman": bool(
            np.isfinite(real_spearman)
            and real_spearman
            >= float(gates["minimum_residual_sigma_vs_exp148_well_rmse_spearman"])
        ),
        "minimum_positive_folds": bool(
            positive_folds >= int(gates["minimum_positive_folds"])
        ),
        "minimum_exp148_rmse_q4_minus_q1_ft": bool(
            float(quartile["q4_minus_q1_exp148_well_rmse_ft"])
            >= float(gates["minimum_exp148_rmse_q4_minus_q1_ft"])
        ),
        "minimum_real_minus_shuffle_spearman": bool(
            np.isfinite(real_minus_shuffle)
            and real_minus_shuffle
            >= float(gates["minimum_real_minus_shuffle_spearman"])
        ),
        "outer_valid_truth_before_feature_freeze_zero": truth_before_freeze == 0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "coverage": float(real_pooled["coverage"]),
        "fallback_fraction": float(real_pooled["fallback_fraction"]),
        "all_features_finite": feature_finite,
        "real_residual_sigma_vs_exp148_well_rmse_spearman": real_spearman,
        "shuffle_residual_sigma_vs_exp148_well_rmse_spearman": shuffle_spearman,
        "real_minus_shuffle_spearman": real_minus_shuffle,
        "positive_folds": positive_folds,
        "q4_minus_q1_exp148_well_rmse_ft": float(
            quartile["q4_minus_q1_exp148_well_rmse_ft"]
        ),
        "outer_valid_truth_rows_before_feature_freeze": truth_before_freeze,
        "thresholds": gates,
    }


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and generated-artifact guards

# %%
def feature_schema_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    descriptions = {
        "typewell_group_log_support_wells": (
            "log1p outer-train usable wells in selected group/global fallback"
        ),
        "typewell_group_log_support_rows": (
            "log1p outer-train fitted GR rows in selected group/global fallback"
        ),
        "typewell_group_residual_sigma": (
            "equal-well median robust residual sigma for selected group/global fallback"
        ),
        "typewell_group_fit_rmse": (
            "equal-well median robust affine fit RMSE for selected group/global fallback"
        ),
        "typewell_group_bias_abs_gr50": "absolute equal-well median affine bias at GR=50",
        "typewell_group_prior_available": (
            "1 for exact native group prior, 0 for frozen global fallback"
        ),
    }
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "dtype": "float64",
                "stage_0_role": "real_feature_and_shuffle_control",
                "stage_1_role": "reserved_add_only" if feature in FEATURE_COLUMNS else None,
                "description": descriptions[feature],
                "posthoc_selectable": False,
                "fallback": str(get_nested(config, "model.features.fallback")),
            }
            for feature in FEATURE_COLUMNS
        ]
    )


def run_stage_0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    validate_run_approval(config)
    started = time.perf_counter()
    output_dir = artifact_dir()
    data_dir = train_data_dir(config)
    wells = list_train_wells(data_dir)
    expected_wells = int(get_nested(config, "data.expected_wells"))
    if len(wells) != expected_wells:
        raise ValueError(f"raw wells {len(wells)} != expected {expected_wells}")
    row_counts = build_row_count_manifest(wells, data_dir)
    expected_rows = int(get_nested(config, "data.expected_rows"))
    if int(row_counts["rows"].sum()) != expected_rows:
        raise ValueError(
            f"raw rows {int(row_counts['rows'].sum())} != expected {expected_rows}"
        )
    folds = build_exp148_fold_manifest(
        row_counts,
        int(get_nested(config, "validation.n_folds")),
    )
    membership_path = resolve_membership_path(config)
    membership = load_native_membership(membership_path, config, wells)
    membership_with_fold = membership.merge(
        folds[["well_id", "fold"]],
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    feature_manifest, priors, fit_quality = build_feature_manifest(
        folds,
        membership,
        data_dir,
        config,
    )
    feature_manifest, feature_freeze_sha = add_freeze_hashes(feature_manifest)
    feature_schema = feature_schema_frame(config)
    feature_schema_sha = dataframe_content_sha(
        feature_schema,
        sort_columns=["feature"],
    )
    # The saved exp148 error is not resolved or opened until every fold/control
    # feature row has been materialized and frozen above.
    exp148_paths = resolve_exp148_paths(config)
    exp148_error, exp148_input = load_exp148_error_after_freeze(
        exp148_paths,
        folds,
        feature_freeze_sha,
        config,
    )
    scored, readout_metrics, quartile = attach_error_and_compute_readout(
        feature_manifest,
        exp148_error,
        feature_freeze_sha,
    )
    gate = evaluate_stage_0_gate(
        feature_manifest,
        readout_metrics,
        quartile,
        config,
    )
    runtime_seconds = float(time.perf_counter() - started)
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "train_data_dir": str(data_dir),
        "raw_wells": len(wells),
        "raw_rows": int(row_counts["rows"].sum()),
        "row_count_content_sha256": dataframe_content_sha(
            row_counts,
            sort_columns=["well_id"],
        ),
        "fold_content_sha256": dataframe_content_sha(
            folds,
            sort_columns=["fold", "well_id"],
        ),
        "membership": verify_pinned_file(
            membership_path,
            str(get_nested(config, "data.exp065_membership.expected_raw_sha256")),
            "exp065 membership",
        ),
        "membership_selected_content_sha256": dataframe_content_sha(
            membership,
            sort_columns=["well_id"],
        ),
        "exp148": exp148_input,
        "feature_schema_content_sha256": feature_schema_sha,
        "feature_manifest_freeze_sha256": feature_freeze_sha,
        "truth_attachment": str(get_nested(config, "validation.truth_attachment")),
    }
    scientific_contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "parent": str(get_nested(config, "lineage.parent")),
        "stage_0_controls": CONTROL_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "execution_contract": get_nested(config, "execution_contract.stage_0"),
        "calibration": get_nested(config, "model.calibration"),
        "gates": get_nested(config, "model.stage_0.pass_requires_all"),
        "forbidden": get_nested(config, "model.forbidden"),
        "stage_1_enabled": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    artifacts["fold_manifest"] = write_csv(
        folds,
        output_dir / f"{OUTPUT_PREFIX}_fold_manifest.csv",
        sort_columns=["fold", "well_id"],
    )
    artifacts["group_membership"] = write_csv(
        membership_with_fold,
        output_dir / f"{OUTPUT_PREFIX}_group_membership.csv",
        sort_columns=["fold", "well_id"],
    )
    artifacts["fit_quality"] = write_csv(
        fit_quality,
        output_dir / f"{OUTPUT_PREFIX}_outer_train_well_quality.csv",
        sort_columns=["fold", "well_id"],
    )
    artifacts["group_priors"] = write_csv(
        priors,
        output_dir / f"{OUTPUT_PREFIX}_group_quality_priors.csv",
        sort_columns=["control", "fold", "group_id"],
    )
    artifacts["feature_manifest"] = write_csv(
        feature_manifest,
        output_dir / f"{OUTPUT_PREFIX}_feature_manifest.csv",
        sort_columns=["control", "fold", "well_id"],
    )
    artifacts["feature_schema"] = write_csv(
        feature_schema,
        output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        sort_columns=["feature"],
    )
    artifacts["error_association"] = write_csv(
        scored,
        output_dir / f"{OUTPUT_PREFIX}_error_association.csv",
        sort_columns=["control", "fold", "well_id"],
    )
    artifacts["fold_metrics"] = write_csv(
        readout_metrics,
        output_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        sort_columns=["control", "fold"],
    )
    artifacts["gate"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_gate.json",
        gate,
    )
    artifacts["input_manifest"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_input_manifest.json",
        input_manifest,
    )
    artifacts["scientific_contract"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_scientific_contract.json",
        scientific_contract,
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage_0_passed" if gate["passed"] else "stage_0_failed",
        "route": "ml_model",
        "parent": str(get_nested(config, "lineage.parent")),
        "runtime_seconds": runtime_seconds,
        "runtime": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "execution_contract": get_nested(config, "execution_contract.stage_0"),
        "raw_wells": len(wells),
        "raw_rows": int(row_counts["rows"].sum()),
        "group_count": int(membership["group_id"].nunique()),
        "feature_rows": len(feature_manifest),
        "prior_rows": len(priors),
        "feature_manifest_freeze_sha256": feature_freeze_sha,
        "quartile_readout": quartile,
        "stage_0_gate": gate,
        "input_manifest": input_manifest,
        "artifact_manifests": artifacts,
        "forbidden_outputs": {
            "trained_models": 0,
            "boosters": 0,
            "predictions": 0,
            "inference": 0,
            "submissions": 0,
        },
    }
    artifacts["summary"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_summary.json",
        summary,
    )
    expected_artifacts = set(
        get_nested(config, "artifacts.expected_stage_0_artifacts") or []
    )
    generated = {Path(item["path"]).name for item in artifacts.values()}
    if generated != expected_artifacts:
        raise RuntimeError(
            f"generated artifact contract mismatch: "
            f"missing={sorted(expected_artifacts - generated)}, "
            f"unexpected={sorted(generated - expected_artifacts)}"
        )
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "ml_model",
        "stage": "stage_0_zero_booster_preflight",
        "cv": gate,
        "public_lb": None,
        "private_lb": None,
        "metric": "feature_stability_and_exp148_error_association",
        "runtime_seconds": runtime_seconds,
        "feature_manifest_freeze_sha256": feature_freeze_sha,
        "summary_path": str(artifacts["summary"]["path"]),
        "notes": (
            "Stage 0 trained no model or booster. A PASS does not authorize the "
            "reserved 15-booster Stage 1, inference, or submission."
        ),
    }
    write_json(metrics_output_path(), metrics_payload)
    return summary


# %% [markdown]
# ## 9. Setup, configuration, and contract preview

# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG)
CONTRACT_PREVIEW = {
    "experiment": get_nested(CONFIG, "experiment.name"),
    "route": get_nested(CONFIG, "experiment.route"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "status": get_nested(CONFIG, "experiment.status"),
    "feature_columns": get_nested(CONFIG, "model.features.add_only"),
    "stage_0_controls": get_nested(CONFIG, "model.stage_0.controls"),
    "execution_contract": get_nested(CONFIG, "execution_contract"),
    "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
    "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
    "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
}
print(json.dumps(to_jsonable(CONTRACT_PREVIEW), indent=2, sort_keys=True), flush=True)


# %% [markdown]
# ## 10. Run the preflight and report generated artifacts

# %%
SUMMARY: dict[str, Any] | None = None
if in_notebook_runtime():
    SUMMARY = run_stage_0(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY["stage_0_gate"]), indent=2, sort_keys=True))
    print("generated artifacts", flush=True)
    for artifact_name, manifest_item in SUMMARY["artifact_manifests"].items():
        print(artifact_name, manifest_item.get("path"), flush=True)
