# %% [markdown]
# # exp311 Type-Well group prefix/suffix GR calibration readout
#
# This zero-booster diagnostic asks whether robust affine/noise summaries learned
# from other wells in the same Type-Well group transfer to a held-out well's
# unknown suffix. Outer-valid TVT is unavailable until every group prior and
# negative-control prior for that fold has been frozen and hashed.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Fold, group, and raw-input checks
# 4. TVT-to-Type-Well-GR pairing and robust calibration helpers
# 5. Fold-safe group priors and deterministic negative controls
# 6. Late suffix-truth attachment and well-equal readout
# 7. Metrics, R2 readout, and fixed promotion gate
# 8. Full Kaggle CPU orchestration and generated-artifact guards
# 9. Setup, configuration, and leakage-contract preview
# 10. Run the diagnostic and report generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp311_typewell_group_prefix_suffix_gr_calibration_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP311_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp311 config not found in {[str(path) for path in candidates]}")


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


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Any) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return mapping_sha256(schema)


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    work = frame.loc[:, chosen]
    digest = hashlib.sha256()
    digest.update(dataframe_schema_sha(work).encode())
    for row in work.itertuples(index=False, name=None):
        digest.update(
            json.dumps(to_jsonable(row), separators=(",", ":"), ensure_ascii=True).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "schema_sha256": dataframe_schema_sha(frame),
        "content_sha256": dataframe_content_sha(frame),
        "raw_sha256": sha256_path(path),
    }


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "schema_sha256": dataframe_schema_sha(frame),
        "content_sha256": dataframe_content_sha(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
    }


def stable_digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def stable_int(*parts: Any, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    return int(stable_digest(*parts)[:16], 16) % modulo


def validate_scientific_contract(config: dict[str, Any]) -> None:
    expected = {
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.strategy": "sha256_well_grouped_5fold",
        "validation.n_folds": 5,
        "validation.fit_scope": "outer_train_wells_only",
        "validation.row_weighting": "equal_well_weight",
        "validation.truth_join_policy": "fit_tables_before_outer_valid_truth_join",
        "calibration.typewell_group": "native_overlap_1",
        "calibration.sensitivity_group": "exact_typewell_content_sha",
        "calibration.tvt_duplicate_reducer": "median",
        "calibration.interpolation": "linear_no_extrapolation",
        "calibration.estimator": "huber_affine_with_identity_shrinkage",
        "calibration.score_unit": "horizontal_gr_api",
        "calibration.aggregate_reducer": "equal_well_median",
        "calibration.group_fallback": "identity_no_correction",
        "execution_contract.diagnostic_variants": 1,
        "execution_contract.model_configs": 0,
        "execution_contract.folds": 5,
        "execution_contract.boosters": 0,
        "execution_contract.decoder_runs": 0,
        "runtime.num_workers": 1,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp311 contract mismatch: {key} must equal {value!r}")
    if list(get_nested(config, "validation.stress_surfaces") or []) != [
        "same_typewell_heldout_well",
        "leave_one_typewell_group_out",
        "spatial_typewell_purged",
    ]:
        raise ValueError("exp311 fixes exactly three ordered stress surfaces")
    if list(get_nested(config, "calibration.negative_controls") or []) != [
        "group_label_shuffle",
        "horizontal_gr_circular_shift",
    ]:
        raise ValueError("exp311 fixes group-shuffle and horizontal-GR-shift controls")
    if float(get_nested(config, "calibration.identity_shrinkage_support_k") or 0.0) != 200.0:
        raise ValueError("exp311 fixes identity shrinkage support k=200")
    if int(get_nested(config, "calibration.overlap_rows_min") or 0) != 32:
        raise ValueError("exp311 fixes minimum pair support at 32 rows")


# %% [markdown]
# ## 3. Fold, group, and raw-input checks


# %%
@dataclass(frozen=True)
class TargetFreeWell:
    well_id: str
    horizontal_path: Path
    typewell_path: Path
    horizontal_gr: np.ndarray
    tvt_input: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    prefix_row_idx: np.ndarray
    prefix_typewell_gr: np.ndarray
    prefix_horizontal_gr: np.ndarray
    suffix_row_idx: np.ndarray


def build_fold_manifest(wells: list[str], n_folds: int, seed: int) -> pd.DataFrame:
    if n_folds < 2 or len(wells) < n_folds:
        raise ValueError("fold count must be at least two and no larger than well count")
    ordered = sorted(set(map(str, wells)), key=lambda well: stable_digest(seed, "fold", well))
    rows = [
        {
            "well_id": well,
            "fold": int(index % n_folds),
            "sha256_order_key": stable_digest(seed, "fold", well),
        }
        for index, well in enumerate(ordered)
    ]
    frame = pd.DataFrame(rows).sort_values(["fold", "sha256_order_key"], kind="mergesort")
    if frame["well_id"].duplicated().any():
        raise ValueError("fold manifest contains duplicate wells")
    return frame.reset_index(drop=True)


def load_group_membership(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.typewell_group_assignments") or {}
    path = resolve_existing(str(spec["filename"]), [str(v) for v in spec["candidates"]])
    source = pd.read_csv(path, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    if not required.issubset(source.columns):
        raise ValueError(f"group assignments missing {sorted(required - set(source.columns))}")
    definitions = get_nested(config, "calibration.group_definitions") or {}
    outputs: list[pd.DataFrame] = []
    for name, definition in definitions.items():
        chosen = source[
            source["method"].eq(str(definition["method"]))
            & source["threshold"].eq(str(definition["threshold"]))
        ].copy()
        if chosen.empty:
            raise ValueError(f"no group assignments found for {name}")
        if chosen["well_id"].duplicated().any():
            raise ValueError(f"{name} assigns at least one well more than once")
        chosen = chosen[["well_id", "cluster_id", "cluster_size"]]
        chosen.insert(0, "group_scheme", str(name))
        chosen = chosen.rename(columns={"cluster_id": "group_id"})
        chosen["cluster_size"] = pd.to_numeric(chosen["cluster_size"], errors="raise").astype(int)
        outputs.append(chosen)
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["group_scheme", "well_id"], kind="mergesort"
    )
    metadata = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "rows": len(source),
        "selected_rows": len(result),
        "selected_wells": int(result["well_id"].nunique()),
        "schemes": sorted(result["group_scheme"].unique()),
    }
    return result.reset_index(drop=True), metadata


def load_spatial_purge_assignments(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.spatial_typewell_purged_assignments") or {}
    path = resolve_existing(str(spec["filename"]), [str(v) for v in spec["candidates"]])
    frame = pd.read_csv(path, dtype=str)
    required = {"well_id", "verification_like_typewell_purged_role"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"spatial purge assignments missing {sorted(required - set(frame.columns))}"
        )
    if frame["well_id"].duplicated().any():
        raise ValueError("spatial purge assignment requires one row per well")
    return frame[["well_id", "verification_like_typewell_purged_role"]].copy(), {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "rows": len(frame),
        "roles": frame["verification_like_typewell_purged_role"].value_counts().to_dict(),
    }


def list_train_wells(data_dir: Path) -> list[str]:
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv")
        for path in data_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv") for path in data_dir.glob("*__typewell.csv")
    }
    wells = sorted(horizontal & typewell)
    if not wells:
        raise FileNotFoundError(f"no paired train wells found under {data_dir}")
    return wells


def collapse_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"TVT", "GR"}
    if not required.issubset(typewell.columns):
        raise ValueError(f"typewell is missing {sorted(required - set(typewell.columns))}")
    work = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    work = work.groupby("TVT", sort=True, as_index=False)["GR"].median()
    tvt = work["TVT"].to_numpy(np.float64)
    gr = work["GR"].to_numpy(np.float64)
    if len(tvt) < 2 or not bool((np.diff(tvt) > 0.0).all()):
        raise ValueError("typewell requires at least two strictly increasing finite TVT values")
    return tvt, gr


def interpolate_no_extrapolation(
    query_tvt: np.ndarray, typewell_tvt: np.ndarray, typewell_gr: np.ndarray
) -> np.ndarray:
    query = np.asarray(query_tvt, dtype=np.float64)
    result = np.full(len(query), np.nan, dtype=np.float64)
    finite = np.isfinite(query)
    inside = finite & (query >= typewell_tvt[0]) & (query <= typewell_tvt[-1])
    if bool(inside.any()):
        result[inside] = np.interp(query[inside], typewell_tvt, typewell_gr)
    return result


def load_target_free_well(well_id: str, data_dir: Path) -> TargetFreeWell:
    horizontal_path = data_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = data_dir / f"{well_id}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
    if "TVT" in horizontal.columns:
        raise ValueError("target-free horizontal reader must not expose TVT")
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell_tvt, typewell_gr = collapse_typewell(typewell)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    paired = interpolate_no_extrapolation(tvt_input, typewell_tvt, typewell_gr)
    prefix_mask = np.isfinite(horizontal_gr) & np.isfinite(tvt_input) & np.isfinite(paired)
    suffix_mask = np.isfinite(horizontal_gr) & ~np.isfinite(tvt_input)
    return TargetFreeWell(
        well_id=str(well_id),
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        horizontal_gr=horizontal_gr,
        tvt_input=tvt_input,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        prefix_row_idx=np.flatnonzero(prefix_mask).astype(np.int32),
        prefix_typewell_gr=paired[prefix_mask],
        prefix_horizontal_gr=horizontal_gr[prefix_mask],
        suffix_row_idx=np.flatnonzero(suffix_mask).astype(np.int32),
    )


# %% [markdown]
# ## 4. TVT-to-Type-Well-GR pairing and robust calibration helpers


# %%
def robust_mad_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan
    center = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - center)))


def weighted_affine(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    root_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
    solution, *_ = np.linalg.lstsq(design * root_weight[:, None], y * root_weight, rcond=None)
    return float(solution[0]), float(solution[1])


def fit_huber_affine_with_identity_shrinkage(
    typewell_gr: np.ndarray,
    horizontal_gr: np.ndarray,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    x_all = np.asarray(typewell_gr, dtype=np.float64)
    y_all = np.asarray(horizontal_gr, dtype=np.float64)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    x = x_all[finite]
    y = y_all[finite]
    support = int(len(x))
    minimum = int(calibration["overlap_rows_min"])
    if support:
        identity_residual = y - x
        identity_sigma = robust_mad_scale(identity_residual)
        identity_rmse = float(np.sqrt(np.mean(identity_residual**2)))
    else:
        identity_sigma = math.nan
        identity_rmse = math.nan
    base = {
        "support_rows": support,
        "used_rows": 0,
        "raw_slope": math.nan,
        "raw_intercept": math.nan,
        "slope": 1.0,
        "intercept": 0.0,
        "bias_at_gr50": 0.0,
        "residual_sigma_mad": identity_sigma,
        "fit_rmse": identity_rmse,
        "identity_rmse": identity_rmse,
        "shrinkage_alpha": 0.0,
        "fit_available": False,
        "fallback_reason": "insufficient_pairs" if support < minimum else None,
        "huber_iterations": 0,
    }
    if support < minimum or float(np.std(x)) <= 1.0e-12:
        if support >= minimum:
            base["fallback_reason"] = "degenerate_typewell_gr"
        return base

    slope, intercept = weighted_affine(x, y, np.ones(support, dtype=np.float64))
    delta = float(calibration["huber_delta"])
    tolerance = float(calibration["huber_tolerance"])
    maximum_iterations = int(calibration["huber_max_iterations"])
    iterations = 0
    for _iteration in range(1, maximum_iterations + 1):
        iterations = _iteration
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
    shrunk_slope = 1.0 + alpha * (slope - 1.0)
    shrunk_intercept = alpha * intercept
    residual = y - (shrunk_slope * x + shrunk_intercept)
    return {
        "support_rows": support,
        "used_rows": support,
        "raw_slope": slope,
        "raw_intercept": intercept,
        "slope": shrunk_slope,
        "intercept": shrunk_intercept,
        "bias_at_gr50": shrunk_slope * 50.0 + shrunk_intercept - 50.0,
        "residual_sigma_mad": robust_mad_scale(residual),
        "fit_rmse": float(np.sqrt(np.mean(residual**2))),
        "identity_rmse": identity_rmse,
        "shrinkage_alpha": alpha,
        "fit_available": True,
        "fallback_reason": None,
        "huber_iterations": iterations,
    }


def load_truth_pairs(
    target_free: TargetFreeWell, *, rows: np.ndarray | None = None
) -> pd.DataFrame:
    truth = pd.read_csv(target_free.horizontal_path, usecols=["TVT"])
    tvt = pd.to_numeric(truth["TVT"], errors="coerce").to_numpy(np.float64)
    if len(tvt) != len(target_free.horizontal_gr):
        raise ValueError(f"horizontal row count changed for {target_free.well_id}")
    row_idx = (
        np.arange(len(tvt), dtype=np.int32) if rows is None else np.asarray(rows, dtype=np.int32)
    )
    paired = interpolate_no_extrapolation(
        tvt[row_idx], target_free.typewell_tvt, target_free.typewell_gr
    )
    horizontal_gr = target_free.horizontal_gr[row_idx]
    finite = np.isfinite(paired) & np.isfinite(horizontal_gr) & np.isfinite(tvt[row_idx])
    return pd.DataFrame(
        {
            "well_id": target_free.well_id,
            "row_idx": row_idx[finite],
            "true_tvt": tvt[row_idx][finite],
            "typewell_gr": paired[finite],
            "horizontal_gr": horizontal_gr[finite],
        }
    )


def circular_shift(
    values: np.ndarray, well_id: str, fold: int, config: dict[str, Any]
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2:
        return y.copy()
    low_fraction = float(get_nested(config, "calibration.circular_shift_min_fraction"))
    high_fraction = float(get_nested(config, "calibration.circular_shift_max_fraction"))
    low = max(1, int(math.ceil(low_fraction * len(y))))
    high = min(len(y) - 1, int(math.floor(high_fraction * len(y))))
    if high < low:
        low, high = 1, len(y) - 1
    offset = low + stable_int("horizontal_gr_circular_shift", fold, well_id, modulo=high - low + 1)
    return np.roll(y, int(offset))


def fit_outer_train_wells(
    train_wells: list[str],
    target_free: dict[str, TargetFreeWell],
    fold: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = dict(get_nested(config, "calibration") or {})
    real_rows: list[dict[str, Any]] = []
    shifted_rows: list[dict[str, Any]] = []
    for well_id in sorted(train_wells):
        pairs = load_truth_pairs(target_free[well_id])
        real = fit_huber_affine_with_identity_shrinkage(
            pairs["typewell_gr"].to_numpy(), pairs["horizontal_gr"].to_numpy(), calibration
        )
        shifted_y = circular_shift(pairs["horizontal_gr"].to_numpy(), well_id, fold, config)
        shifted = fit_huber_affine_with_identity_shrinkage(
            pairs["typewell_gr"].to_numpy(), shifted_y, calibration
        )
        real_rows.append({"fold": fold, "well_id": well_id, **real})
        shifted_rows.append({"fold": fold, "well_id": well_id, **shifted})
    return pd.DataFrame(real_rows), pd.DataFrame(shifted_rows)


# %% [markdown]
# ## 5. Fold-safe group priors and deterministic negative controls


# %%
def group_lookup(membership: pd.DataFrame, scheme: str) -> dict[str, str]:
    selected = membership[membership["group_scheme"].eq(scheme)]
    return dict(zip(selected["well_id"], selected["group_id"], strict=False))


def shuffled_group_lookup(
    train_wells: list[str], lookup: dict[str, str], fold: int
) -> dict[str, str]:
    eligible = [well for well in train_wells if well in lookup]
    ordered = sorted(eligible, key=lambda well: stable_digest("group_shuffle", fold, well))
    if len(ordered) < 2:
        return {well: lookup[well] for well in ordered}
    labels = [lookup[well] for well in ordered]
    offset = 1 + stable_int("group_shuffle_offset", fold, modulo=len(labels) - 1)
    rotated = labels[offset:] + labels[:offset]
    return dict(zip(ordered, rotated, strict=True))


def aggregate_group_priors(
    stats: pd.DataFrame,
    source_group_by_well: dict[str, str],
    *,
    fold: int,
    group_scheme: str,
    surface: str,
    control: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    work = stats[stats["well_id"].isin(source_group_by_well)].copy()
    work["group_id"] = work["well_id"].map(source_group_by_well)
    work = work[work["fit_available"].fillna(False) & work["group_id"].notna()].copy()
    value_columns = [
        "slope",
        "intercept",
        "bias_at_gr50",
        "residual_sigma_mad",
        "fit_rmse",
    ]
    rows: list[dict[str, Any]] = []
    minimum_wells = int(get_nested(config, "calibration.minimum_peer_wells") or 1)
    minimum_rows = int(get_nested(config, "calibration.minimum_peer_effective_rows") or 32)
    for group_id, group in work.groupby("group_id", sort=True):
        source_wells = sorted(group["well_id"].astype(str).tolist())
        support_rows = int(group["support_rows"].sum())
        source_well_count = len(source_wells)
        available = source_well_count >= minimum_wells and support_rows >= minimum_rows
        row: dict[str, Any] = {
            "fold": fold,
            "group_scheme": group_scheme,
            "surface": surface,
            "control": control,
            "group_id": str(group_id),
            "source_wells": source_well_count,
            "support_rows": support_rows,
            "available": available,
            "source_wells_sha256": mapping_sha256(source_wells),
        }
        for column in value_columns:
            row[column] = float(group[column].median()) if available else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def surface_wells(
    surface: str,
    outer_train: list[str],
    outer_valid: list[str],
    purge_roles: dict[str, str],
) -> tuple[list[str], list[str]]:
    if surface in {"same_typewell_heldout_well", "leave_one_typewell_group_out"}:
        return sorted(outer_train), sorted(outer_valid)
    if surface == "spatial_typewell_purged":
        train = [well for well in outer_train if purge_roles.get(well) == "train"]
        valid = [well for well in outer_valid if purge_roles.get(well) == "valid"]
        return sorted(train), sorted(valid)
    raise ValueError(f"unknown stress surface: {surface}")


def prior_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    return {str(row["group_id"]): row.to_dict() for _, row in frame.iterrows()}


# %% [markdown]
# ## 6. Late suffix-truth attachment and well-equal readout


# %%
def attach_suffix_truth_after_freeze(
    target_free: TargetFreeWell,
    *,
    freeze_sha256: str,
) -> pd.DataFrame:
    if len(str(freeze_sha256)) != 64:
        raise ValueError("outer-valid truth requires a complete frozen-prior SHA256")
    return load_truth_pairs(target_free, rows=target_free.suffix_row_idx)


def identity_prior(reason: str) -> dict[str, Any]:
    return {
        "slope": 1.0,
        "intercept": 0.0,
        "bias_at_gr50": 0.0,
        "residual_sigma_mad": math.nan,
        "fit_rmse": math.nan,
        "source_wells": 0,
        "support_rows": 0,
        "available": False,
        "fallback_reason": reason,
    }


def score_suffix_well(
    suffix_pairs: pd.DataFrame,
    prefix_stats: dict[str, Any],
    group_prior: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    x = suffix_pairs["typewell_gr"].to_numpy(np.float64)
    y = suffix_pairs["horizontal_gr"].to_numpy(np.float64)
    actual = fit_huber_affine_with_identity_shrinkage(x, y, calibration)
    identity_error = y - x
    prefix_error = y - (float(prefix_stats["slope"]) * x + float(prefix_stats["intercept"]))
    transfer_error = y - (float(group_prior["slope"]) * x + float(group_prior["intercept"]))
    identity_rmse = float(np.sqrt(np.mean(identity_error**2)))
    prefix_rmse = float(np.sqrt(np.mean(prefix_error**2)))
    transfer_rmse = float(np.sqrt(np.mean(transfer_error**2)))
    return {
        "suffix_rows": len(suffix_pairs),
        "identity_suffix_gr_rmse": identity_rmse,
        "prefix_suffix_gr_rmse": prefix_rmse,
        "transfer_suffix_gr_rmse": transfer_rmse,
        "transfer_gain_vs_identity": identity_rmse - transfer_rmse,
        "transfer_delta_vs_identity": transfer_rmse - identity_rmse,
        "transfer_gain_vs_prefix": prefix_rmse - transfer_rmse,
        "actual_suffix_slope": actual["slope"],
        "actual_suffix_intercept": actual["intercept"],
        "actual_suffix_bias_at_gr50": actual["bias_at_gr50"],
        "actual_suffix_residual_sigma_mad": actual["residual_sigma_mad"],
        "actual_suffix_fit_rmse": actual["fit_rmse"],
        "actual_suffix_fit_available": actual["fit_available"],
    }


def prefix_calibration_frame(
    target_free: dict[str, TargetFreeWell], config: dict[str, Any]
) -> pd.DataFrame:
    calibration = dict(get_nested(config, "calibration") or {})
    rows = []
    for well_id in sorted(target_free):
        item = target_free[well_id]
        stats = fit_huber_affine_with_identity_shrinkage(
            item.prefix_typewell_gr, item.prefix_horizontal_gr, calibration
        )
        rows.append({"well_id": well_id, **stats})
    return pd.DataFrame(rows)


def pair_artifact_for_well(
    target_free: TargetFreeWell, suffix_pairs: pd.DataFrame, fold: int
) -> pd.DataFrame:
    prefix = pd.DataFrame(
        {
            "fold": fold,
            "well_id": target_free.well_id,
            "row_idx": target_free.prefix_row_idx,
            "partition": "known_prefix",
            "typewell_gr": target_free.prefix_typewell_gr,
            "horizontal_gr": target_free.prefix_horizontal_gr,
            "truth_attached_after_freeze": False,
        }
    )
    suffix = suffix_pairs[["well_id", "row_idx", "typewell_gr", "horizontal_gr"]].copy()
    suffix.insert(0, "fold", fold)
    suffix.insert(3, "partition", "evaluation_suffix")
    suffix["truth_attached_after_freeze"] = True
    return pd.concat([prefix, suffix], ignore_index=True)


# %% [markdown]
# ## 7. Metrics, R2 readout, and fixed promotion gate


# %%
def r2_equal_well(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if len(y) < 2:
        return math.nan, len(y)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 1.0e-12:
        return math.nan, len(y)
    return float(1.0 - np.sum((y - p) ** 2) / denominator), len(y)


def aggregate_suffix_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    keys = ["group_scheme", "surface", "control"]
    rows: list[dict[str, Any]] = []
    for key, group in scored.groupby(keys, sort=True, dropna=False):
        for fold_label, scope in [
            *((int(fold), part) for fold, part in group.groupby("fold", sort=True)),
            ("pooled", group),
        ]:
            if scope.empty:
                continue
            identity = np.sqrt(np.mean(scope["identity_suffix_gr_rmse"].to_numpy() ** 2))
            prefix = np.sqrt(np.mean(scope["prefix_suffix_gr_rmse"].to_numpy() ** 2))
            transfer = np.sqrt(np.mean(scope["transfer_suffix_gr_rmse"].to_numpy() ** 2))
            rows.append(
                {
                    "group_scheme": key[0],
                    "surface": key[1],
                    "control": key[2],
                    "fold": fold_label,
                    "wells": len(scope),
                    "available_wells": int(scope["prior_available"].sum()),
                    "availability_rate": float(scope["prior_available"].mean()),
                    "suffix_rows": int(scope["suffix_rows"].sum()),
                    "identity_suffix_gr_rmse": float(identity),
                    "prefix_suffix_gr_rmse": float(prefix),
                    "transfer_suffix_gr_rmse": float(transfer),
                    "transfer_gain_vs_identity": float(identity - transfer),
                    "transfer_gain_vs_prefix": float(prefix - transfer),
                    "worst_well_gr_rmse_delta": float(scope["transfer_delta_vs_identity"].max()),
                    "improved_wells": int((scope["transfer_gain_vs_identity"] > 0.0).sum()),
                    "worse_wells": int((scope["transfer_gain_vs_identity"] < 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_r2_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["group_scheme", "surface", "control"]
    fields = [
        ("residual_sigma_mad", "actual_suffix_residual_sigma_mad"),
        ("fit_rmse", "actual_suffix_fit_rmse"),
        ("bias_at_gr50", "actual_suffix_bias_at_gr50"),
        ("slope", "actual_suffix_slope"),
        ("intercept", "actual_suffix_intercept"),
    ]
    for key, group in scored.groupby(keys, sort=True, dropna=False):
        for fold_label, scope in [
            *((int(fold), part) for fold, part in group.groupby("fold", sort=True)),
            ("pooled", group),
        ]:
            available = scope[scope["prior_available"]].copy()
            for predicted_column, actual_column in fields:
                value, wells = r2_equal_well(
                    available[actual_column].to_numpy(), available[predicted_column].to_numpy()
                )
                rows.append(
                    {
                        "group_scheme": key[0],
                        "surface": key[1],
                        "control": key[2],
                        "fold": fold_label,
                        "statistic": predicted_column,
                        "wells": wells,
                        "r2_equal_well": value,
                    }
                )
    return pd.DataFrame(rows)


def one_metric_row(
    metrics: pd.DataFrame,
    *,
    scheme: str,
    surface: str,
    control: str,
    fold: str | int,
) -> pd.Series:
    selected = metrics[
        metrics["group_scheme"].eq(scheme)
        & metrics["surface"].eq(surface)
        & metrics["control"].eq(control)
        & metrics["fold"].astype(str).eq(str(fold))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one metric row for {(scheme, surface, control, fold)}, got {len(selected)}"
        )
    return selected.iloc[0]


def evaluate_promotion_gate(
    metrics: pd.DataFrame,
    r2_metrics: pd.DataFrame,
    scored: pd.DataFrame,
    freeze_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    scheme = str(get_nested(config, "calibration.typewell_group"))
    surface = str(get_nested(config, "validation.primary_surface"))
    real = one_metric_row(metrics, scheme=scheme, surface=surface, control="real", fold="pooled")
    shuffled = one_metric_row(
        metrics, scheme=scheme, surface=surface, control="group_label_shuffle", fold="pooled"
    )
    r2_pool = r2_metrics[
        r2_metrics["group_scheme"].eq(scheme)
        & r2_metrics["surface"].eq(surface)
        & r2_metrics["control"].eq("real")
        & r2_metrics["fold"].astype(str).eq("pooled")
    ].set_index("statistic")
    noise_r2 = float(r2_pool.loc["residual_sigma_mad", "r2_equal_well"])
    fit_r2 = float(r2_pool.loc["fit_rmse", "r2_equal_well"])
    folds = metrics[
        metrics["group_scheme"].eq(scheme)
        & metrics["surface"].eq(surface)
        & metrics["control"].eq("real")
        & ~metrics["fold"].astype(str).eq("pooled")
    ]
    folds_improved = int((folds["transfer_gain_vs_identity"] > 0.0).sum())
    gates = get_nested(config, "promotion_gates") or {}
    checks = {
        "outer_valid_truth_before_freeze_zero": bool(
            (freeze_manifest["outer_valid_truth_rows_before_freeze"] == 0).all()
        ),
        "group_loo_noise_r2": bool(noise_r2 >= float(gates["group_loo_r2_noise_min"])),
        "group_loo_fit_rmse_r2": bool(fit_r2 >= float(gates["group_loo_r2_fit_rmse_min"])),
        "suffix_gr_rmse_gain": bool(
            float(real["transfer_gain_vs_identity"])
            >= float(gates["calibrated_suffix_gr_rmse_gain_min"])
        ),
        "folds_improved": bool(folds_improved >= int(gates["folds_improved_min"])),
        "real_minus_shuffled_gain": bool(
            float(real["transfer_gain_vs_identity"]) - float(shuffled["transfer_gain_vs_identity"])
            >= float(gates["real_minus_shuffled_gr_gain_min"])
        ),
        "worst_well_delta": bool(
            float(real["worst_well_gr_rmse_delta"]) <= float(gates["worst_well_gr_rmse_delta_max"])
        ),
        "minimum_scored_wells": bool(
            int(real["wells"]) >= int(get_nested(config, "validation.minimum_scored_wells"))
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "primary_scheme": scheme,
        "primary_surface": surface,
        "score_unit": get_nested(config, "calibration.score_unit"),
        "noise_r2": noise_r2,
        "fit_rmse_r2": fit_r2,
        "suffix_gr_rmse_gain": float(real["transfer_gain_vs_identity"]),
        "real_minus_shuffled_gr_gain": float(real["transfer_gain_vs_identity"])
        - float(shuffled["transfer_gain_vs_identity"]),
        "folds_improved": folds_improved,
        "worst_well_gr_rmse_delta": float(real["worst_well_gr_rmse_delta"]),
        "scored_wells": int(real["wells"]),
        "available_wells": int(real["available_wells"]),
    }


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and generated-artifact guards


# %%
def run_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    started = time.perf_counter()
    output_dir = artifact_dir()
    data_dir = train_data_dir(config)
    wells = list_train_wells(data_dir)
    membership, membership_source = load_group_membership(config)
    purge, purge_source = load_spatial_purge_assignments(config)
    required_schemes = list((get_nested(config, "calibration.group_definitions") or {}).keys())
    for scheme in required_schemes:
        assigned = set(membership.loc[membership["group_scheme"].eq(scheme), "well_id"].astype(str))
        missing = sorted(set(wells) - assigned)
        if missing:
            raise ValueError(f"{scheme} is missing {len(missing)} raw train wells")
    purge_roles = dict(
        zip(purge["well_id"], purge["verification_like_typewell_purged_role"], strict=False)
    )
    seed = int(get_nested(config, "reproducibility.seed") or 42)
    folds = build_fold_manifest(wells, int(get_nested(config, "validation.n_folds")), seed)

    target_free: dict[str, TargetFreeWell] = {}
    input_rows: list[dict[str, Any]] = []
    for index, well_id in enumerate(sorted(wells), start=1):
        if index == 1 or index % 50 == 0 or index == len(wells):
            print(f"target-free input [{index}/{len(wells)}] well={well_id}", flush=True)
        item = load_target_free_well(well_id, data_dir)
        target_free[well_id] = item
        input_rows.append(
            {
                "well_id": well_id,
                "horizontal_rows": len(item.horizontal_gr),
                "prefix_pairs": len(item.prefix_row_idx),
                "suffix_observed_gr_rows": len(item.suffix_row_idx),
                "horizontal_sha256": sha256_path(item.horizontal_path),
                "typewell_sha256": sha256_path(item.typewell_path),
            }
        )
    input_manifest = pd.DataFrame(input_rows).sort_values("well_id", kind="mergesort")
    prefix_stats = prefix_calibration_frame(target_free, config)
    prefix_by_well = {str(row["well_id"]): row.to_dict() for _, row in prefix_stats.iterrows()}
    membership_lookups = {scheme: group_lookup(membership, scheme) for scheme in required_schemes}

    group_prior_frames: list[pd.DataFrame] = []
    scored_rows: list[dict[str, Any]] = []
    freeze_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []
    suffix_pairs_by_well: dict[str, pd.DataFrame] = {}
    calibration = dict(get_nested(config, "calibration") or {})
    surfaces = list(get_nested(config, "validation.stress_surfaces") or [])

    for fold in range(int(get_nested(config, "validation.n_folds"))):
        outer_valid = sorted(folds.loc[folds["fold"].eq(fold), "well_id"].astype(str))
        outer_train = sorted(set(wells) - set(outer_valid))
        print(
            f"fold={fold} outer_train={len(outer_train)} outer_valid={len(outer_valid)}",
            flush=True,
        )
        train_stats, shifted_train_stats = fit_outer_train_wells(
            outer_train, target_free, fold, config
        )
        fold_specs: list[dict[str, Any]] = []
        for scheme in required_schemes:
            lookup = membership_lookups[scheme]
            for surface in surfaces:
                allowed_train, valid_wells = surface_wells(
                    surface, outer_train, outer_valid, purge_roles
                )
                if surface == "leave_one_typewell_group_out":
                    real_prior = pd.DataFrame()
                else:
                    source_lookup = {well: lookup[well] for well in allowed_train if well in lookup}
                    real_prior = aggregate_group_priors(
                        train_stats[train_stats["well_id"].isin(allowed_train)],
                        source_lookup,
                        fold=fold,
                        group_scheme=scheme,
                        surface=surface,
                        control="real",
                        config=config,
                    )
                fold_specs.append(
                    {
                        "scheme": scheme,
                        "surface": surface,
                        "control": "real",
                        "valid_wells": valid_wells,
                        "lookup": lookup,
                        "priors": real_prior,
                    }
                )
                if not real_prior.empty:
                    group_prior_frames.append(real_prior)

            primary_surface = str(get_nested(config, "validation.primary_surface"))
            allowed_train, valid_wells = surface_wells(
                primary_surface, outer_train, outer_valid, purge_roles
            )
            shuffled_lookup = shuffled_group_lookup(allowed_train, lookup, fold)
            shuffled_prior = aggregate_group_priors(
                train_stats[train_stats["well_id"].isin(allowed_train)],
                shuffled_lookup,
                fold=fold,
                group_scheme=scheme,
                surface=primary_surface,
                control="group_label_shuffle",
                config=config,
            )
            shifted_lookup = {well: lookup[well] for well in allowed_train if well in lookup}
            shifted_prior = aggregate_group_priors(
                shifted_train_stats[shifted_train_stats["well_id"].isin(allowed_train)],
                shifted_lookup,
                fold=fold,
                group_scheme=scheme,
                surface=primary_surface,
                control="horizontal_gr_circular_shift",
                config=config,
            )
            group_prior_frames.extend([shuffled_prior, shifted_prior])
            fold_specs.extend(
                [
                    {
                        "scheme": scheme,
                        "surface": primary_surface,
                        "control": "group_label_shuffle",
                        "valid_wells": valid_wells,
                        "lookup": lookup,
                        "priors": shuffled_prior,
                    },
                    {
                        "scheme": scheme,
                        "surface": primary_surface,
                        "control": "horizontal_gr_circular_shift",
                        "valid_wells": valid_wells,
                        "lookup": lookup,
                        "priors": shifted_prior,
                    },
                ]
            )

        freeze_payload = []
        for spec in fold_specs:
            priors = spec["priors"]
            freeze_payload.append(
                {
                    "scheme": spec["scheme"],
                    "surface": spec["surface"],
                    "control": spec["control"],
                    "valid_wells_sha256": mapping_sha256(spec["valid_wells"]),
                    "prior_content_sha256": (
                        dataframe_content_sha(priors) if not priors.empty else mapping_sha256([])
                    ),
                }
            )
        freeze_sha = mapping_sha256(freeze_payload)
        freeze_rows.append(
            {
                "fold": fold,
                "outer_train_wells": len(outer_train),
                "outer_valid_wells": len(outer_valid),
                "outer_train_truth_wells_used": len(train_stats),
                "outer_valid_truth_rows_before_freeze": 0,
                "freeze_sha256": freeze_sha,
            }
        )

        for well_id in outer_valid:
            suffix_pairs = attach_suffix_truth_after_freeze(
                target_free[well_id], freeze_sha256=freeze_sha
            )
            suffix_pairs_by_well[well_id] = suffix_pairs
            pair_frames.append(pair_artifact_for_well(target_free[well_id], suffix_pairs, fold))

        minimum_suffix = int(
            get_nested(config, "validation.minimum_scored_suffix_rows_per_well") or 32
        )
        for spec in fold_specs:
            lookup = spec["lookup"]
            priors = prior_map(spec["priors"])
            for well_id in spec["valid_wells"]:
                suffix_pairs = suffix_pairs_by_well[well_id]
                if len(suffix_pairs) < minimum_suffix:
                    continue
                group_id = lookup.get(well_id)
                if spec["surface"] == "leave_one_typewell_group_out":
                    prior = identity_prior("typewell_group_left_out")
                elif group_id is None or group_id not in priors:
                    prior = identity_prior("group_prior_unavailable")
                else:
                    prior = dict(priors[group_id])
                    prior["fallback_reason"] = None
                score = score_suffix_well(suffix_pairs, prefix_by_well[well_id], prior, calibration)
                scored_rows.append(
                    {
                        "fold": fold,
                        "well_id": well_id,
                        "group_scheme": spec["scheme"],
                        "surface": spec["surface"],
                        "control": spec["control"],
                        "group_id": group_id,
                        "prior_available": bool(prior["available"]),
                        "fallback_reason": prior.get("fallback_reason"),
                        "source_wells": int(prior["source_wells"]),
                        "support_rows": int(prior["support_rows"]),
                        "slope": float(prior["slope"]),
                        "intercept": float(prior["intercept"]),
                        "bias_at_gr50": float(prior["bias_at_gr50"]),
                        "residual_sigma_mad": float(prior["residual_sigma_mad"]),
                        "fit_rmse": float(prior["fit_rmse"]),
                        "freeze_sha256": freeze_sha,
                        **score,
                    }
                )

    group_priors = pd.concat(group_prior_frames, ignore_index=True)
    scored = pd.DataFrame(scored_rows).sort_values(
        ["group_scheme", "surface", "control", "fold", "well_id"], kind="mergesort"
    )
    freeze_manifest = pd.DataFrame(freeze_rows).sort_values("fold", kind="mergesort")
    pair_table = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["well_id", "row_idx", "partition"], kind="mergesort"
    )
    metrics = aggregate_suffix_metrics(scored)
    r2_metrics = compute_r2_metrics(scored)
    promotion = evaluate_promotion_gate(metrics, r2_metrics, scored, freeze_manifest, config)

    membership_output = membership.merge(folds[["well_id", "fold"]], on="well_id", how="left")
    feature_schema = pd.DataFrame(
        [
            {
                "artifact": "pair_table",
                "column": column,
                "dtype": str(pair_table[column].dtype),
                "uses_outer_valid_truth": column in {"typewell_gr", "truth_attached_after_freeze"},
                "freeze_policy": "suffix values materialized only after fold prior SHA freeze",
            }
            for column in pair_table.columns
        ]
        + [
            {
                "artifact": "suffix_by_well",
                "column": column,
                "dtype": str(scored[column].dtype),
                "uses_outer_valid_truth": column.startswith("actual_suffix_")
                or "suffix_gr_rmse" in column
                or column.startswith("transfer_gain")
                or column.startswith("transfer_delta"),
                "freeze_policy": "score-only after fold prior SHA freeze",
            }
            for column in scored.columns
        ]
    )

    artifact_frames = {
        "fold_manifest": folds,
        "group_membership": membership_output,
        "prefix_calibration": prefix_stats,
        "group_priors": group_priors,
        "suffix_by_well": scored,
        "surface_metrics": metrics,
        "r2_metrics": r2_metrics,
        "feature_schema": feature_schema,
    }
    artifact_manifests: dict[str, Any] = {}
    for name, frame in artifact_frames.items():
        path = output_dir / f"{OUTPUT_PREFIX}_{name}.csv"
        artifact_manifests[name] = write_csv(frame, path)
    pair_path = output_dir / f"{OUTPUT_PREFIX}_pair_table.csv.gz"
    artifact_manifests["pair_table"] = write_csv_gzip(pair_table, pair_path)

    runtime_seconds = time.perf_counter() - started
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_readout"
        if promotion["passed"]
        else "completed_gate_failed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_only": True,
            "internet_enabled": False,
            "kaggle_kernel_version": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "execution_contract": get_nested(config, "execution_contract"),
        "score_unit": get_nested(config, "calibration.score_unit"),
        "rows": int(sum(len(item.horizontal_gr) for item in target_free.values())),
        "wells": len(target_free),
        "prefix_pairs": int(sum(len(item.prefix_row_idx) for item in target_free.values())),
        "suffix_pairs": int(sum(len(frame) for frame in suffix_pairs_by_well.values())),
        "truth_boundary": {
            "policy": get_nested(config, "validation.truth_join_policy"),
            "outer_valid_truth_rows_before_freeze": int(
                freeze_manifest["outer_valid_truth_rows_before_freeze"].sum()
            ),
            "fold_freeze_sha256": dict(
                zip(
                    freeze_manifest["fold"].astype(str),
                    freeze_manifest["freeze_sha256"],
                    strict=True,
                )
            ),
        },
        "input_manifests": {
            "raw_wells_content_sha256": dataframe_content_sha(input_manifest),
            "raw_wells_schema_sha256": dataframe_schema_sha(input_manifest),
            "group_membership_source": membership_source,
            "spatial_purge_source": purge_source,
        },
        "promotion": promotion,
        "artifact_manifests": artifact_manifests,
        "forbidden_outputs": {
            "models": 0,
            "boosters": 0,
            "decoders": 0,
            "predictions": 0,
            "submission": False,
        },
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    summary["artifact_manifests"]["summary"] = {
        "path": str(summary_path),
        "raw_sha256": sha256_path(summary_path),
    }
    expected = set(get_nested(config, "artifacts.expected_train_artifacts") or [])
    generated = {Path(item["path"]).name for item in summary["artifact_manifests"].values()}
    generated.add(summary_path.name)
    if generated != expected:
        raise RuntimeError(
            f"generated artifact contract mismatch: missing={sorted(expected - generated)} "
            f"unexpected={sorted(generated - expected)}"
        )
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_readout"
        if promotion["passed"]
        else "completed_gate_failed",
        "route": "pf_beam",
        "cv": promotion,
        "public_lb": None,
        "private_lb": None,
        "metric": "group_transfer_suffix_gr_reconstruction",
        "score_unit": get_nested(config, "calibration.score_unit"),
        "runtime_seconds": runtime_seconds,
        "summary_path": str(summary_path),
        "notes": "No model, decoder, inference, or submission was run.",
    }
    write_json(metrics_output_path(), metrics_payload)
    return summary


# %% [markdown]
# ## 9. Setup, configuration, and leakage-contract preview


# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG)
CONTRACT_PREVIEW = {
    "experiment": get_nested(CONFIG, "experiment.name"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "route": get_nested(CONFIG, "experiment.route"),
    "implementation_scope": get_nested(CONFIG, "implementation.scope"),
    "group_definitions": get_nested(CONFIG, "calibration.group_definitions"),
    "stress_surfaces": get_nested(CONFIG, "validation.stress_surfaces"),
    "negative_controls": get_nested(CONFIG, "calibration.negative_controls"),
    "score_unit": get_nested(CONFIG, "calibration.score_unit"),
    "truth_join_policy": get_nested(CONFIG, "validation.truth_join_policy"),
    "execution_contract": get_nested(CONFIG, "execution_contract"),
    "inference_enabled": get_nested(CONFIG, "inference.enabled"),
    "submission_enabled": get_nested(CONFIG, "inference.create_submission"),
}
print(json.dumps(CONTRACT_PREVIEW, indent=2, sort_keys=True), flush=True)


# %% [markdown]
# ## 10. Run the diagnostic and report generated artifacts


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_diagnostic(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY["promotion"]), indent=2, sort_keys=True), flush=True)
    print("generated artifacts", flush=True)
    for artifact_name, manifest in SUMMARY["artifact_manifests"].items():
        print(artifact_name, manifest.get("path"), flush=True)
else:
    SUMMARY = None
