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
# # exp296 exp223 self-GR known-TVT support gate — train audit
#
# This compact self-contained notebook keeps the exp223 alpha=0.07 boost-only
# exact HMM fixed and applies exactly one candidate-state support mask. It
# freezes predictions and support manifests before loading unknown-suffix TVT.

# %% [markdown]
# ## Contents
#
# 1. Imports and fixed contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Scientific contract validation
# 4. Exp223 prefix and self-GR surface helpers
# 5. Known-TVT candidate-state support gate
# 6. Exact exp223 forward-backward kernel
# 7. Single-well prediction generation
# 8. Truth-late control join and metrics
# 9. Full Kaggle CPU orchestration and generated artifacts
# 10. Setup, cost guard, and execution switch

# %% [markdown]
# ## 1. Imports and fixed contract

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def prange(*args: Any) -> range:
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None

    def get_num_threads() -> int | None:
        return None

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(func: Any) -> Any:
            return func

        return decorator


EXPERIMENT_NAME = "exp296_exp223_self_gr_known_tvt_support_gate"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT = "hmm_selfgr_boost_only_a070_c100_known_tvt_support_gate"
PARENT_EXPERIMENT = "exp223_joint_typewell_self_gr_hmm_likelihood_probe"
PARENT_CANDIDATE = "hmm_selfgr_boost_only_a070_c100"
PARENT_PREDICTION_COLUMN = f"{PARENT_CANDIDATE}_mean_tvt"
PREDICTION_COLUMN = f"{VARIANT}_mean_tvt"
EXPECTED_PARENT_DECOMPRESSED_SHA256 = (
    "0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c"
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
GENERATION_HORIZONTAL_COLUMNS = ("MD", "Z", "GR", "TVT_input")
GENERATION_TYPEWELL_COLUMNS = ("TVT", "GR")
DISTANCE_BUCKETS = ("000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = in_notebook_runtime()

# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    return Path.cwd()


def package_dir() -> Path:
    root = project_root()
    local = root / "experiments" / EXPERIMENT_NAME
    return local if local.exists() else Path.cwd()


def load_experiment_config() -> dict[str, Any]:
    candidates = [package_dir() / "config.yaml", Path.cwd() / "config.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return read_yaml(candidate)
    raise FileNotFoundError(f"config.yaml not found: {[str(path) for path in candidates]}")


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    expected_wells = int(get_nested(config, "comparison.saved_control_wells", 0))
    candidates = [
        configured,
        project_root() / configured,
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            sorted({path.parent for path in KAGGLE_INPUT_ROOT.rglob("*__horizontal_well.csv")})
        )
    inventory: dict[str, int] = {}
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate
        if normalized in seen:
            continue
        seen.add(normalized)
        files = sorted(candidate.glob("*__horizontal_well.csv")) if candidate.exists() else []
        inventory[str(candidate)] = len(files)
        if files and (expected_wells <= 0 or len(files) == expected_wells):
            return candidate.resolve()
    raise FileNotFoundError(
        f"train directory with expected_wells={expected_wells} not found: "
        + json.dumps(inventory, sort_keys=True)
    )


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = package_dir() / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def resolve_existing_file(candidates: Sequence[str]) -> Path:
    checked: list[str] = []
    root = project_root()
    for raw in candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for raw in candidates:
            basename = Path(raw).name
            matches = [
                path
                for path in sorted(KAGGLE_INPUT_ROOT.rglob(basename))
                if path.stat().st_size > 0
            ]
            checked.extend(str(path) for path in matches)
            if matches:
                return matches[0]
    raise FileNotFoundError("No non-empty input candidate exists: " + json.dumps(checked, indent=2))


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_stable_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def stable_reporting_fold(well: str, n_folds: int = 5) -> int:
    digest = hashlib.sha256(str(well).encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % int(n_folds)


def list_well_ids(data_dir: str | Path) -> list[str]:
    directory = Path(data_dir)
    wells: list[str] = []
    for path in sorted(directory.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        if (directory / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def distance_bucket(values: pd.Series | np.ndarray) -> np.ndarray:
    md_since = np.asarray(values, dtype=np.float64)
    labels = np.full(len(md_since), "1000_plus", dtype=object)
    labels[md_since < 1000.0] = "500_1000"
    labels[md_since < 500.0] = "250_500"
    labels[md_since < 250.0] = "100_250"
    labels[md_since < 100.0] = "050_100"
    labels[md_since < 50.0] = "000_050"
    return labels


# %% [markdown]
# ## 3. Scientific contract validation


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> None:
    expected_hmm = {
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "df": 4.0,
        "emission": "gauss",
        "lam": 1.0,
        "sigma_mode": "std",
        "start_sig": 0.75,
        "r0_sig": 0.01,
        "band_pad": 100.0,
        "mom": 0.998,
        "rate_center": "zero",
    }
    actual_hmm = dict(get_nested(config, "model.hmm") or {})
    scientific_hmm = {key: actual_hmm.get(key) for key in expected_hmm}
    if scientific_hmm != expected_hmm:
        raise ValueError(f"exp223 HMM contract changed: {scientific_hmm}")
    if actual_hmm.get("source") != PARENT_EXPERIMENT:
        raise ValueError("exp223 HMM source changed")

    expected_self_gr = {
        "alpha": 0.07,
        "clip": 1.0,
        "mode": "boost_only",
        "window_radius_rows": 12,
        "descriptor_offsets": [-12, -8, -4, 0, 4, 8, 12],
        "top_k": 5,
        "prefix_anchor_stride": 3,
        "max_prefix_anchors": 128,
        "keep_last_prefix_anchors": 32,
        "min_prefix_anchors": 12,
        "max_window_missing_rate": 0.35,
        "gaussian_sigma_tvt": 12.0,
        "descriptor_distance_temperature": 1.5,
        "typewell_agreement_sigma_tvt": 18.0,
        "surface_quadratic_clip": 60.0,
        "surface_chunk_size": 256,
    }
    actual_self_gr = dict(get_nested(config, "model.self_gr_emission") or {})
    scientific_self_gr = {key: actual_self_gr.get(key) for key in expected_self_gr}
    if scientific_self_gr != expected_self_gr:
        raise ValueError(f"exp223 self-GR contract changed: {scientific_self_gr}")
    if actual_self_gr.get("source") != "exp223_hmm_selfgr_boost_only_a070_c100":
        raise ValueError("exp223 self-GR source changed")

    support = dict(get_nested(config, "model.support_gate") or {})
    required_support = {
        "source_column": "TVT_input",
        "source_rows": "finite_visible_prefix_all_rows",
        "boundary": "inclusive",
        "padding_tvt": 0.0,
        "apply_after_full_grid_centering_scaling_and_positive_clip": True,
        "outside_support_self_gr_contribution": 0.0,
        "inside_support_policy": "exact_exp223_boost_parity",
        "no_finite_known_tvt_action": "all_false_mask_self_gr_neutral",
        "final_prediction_clip_to_support": False,
        "use_final_prediction_to_gate": False,
        "use_exp225_state_known_tvt_curve": False,
    }
    actual_support = {key: support.get(key) for key in required_support}
    if actual_support != required_support:
        raise ValueError(f"support gate contract changed: {actual_support}")

    exact_values = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "ensemble",
        "lineage.parent": PARENT_EXPERIMENT,
        "model.variant": VARIANT,
        "model.planned_variants": 1,
        "model.planned_hmm_well_runs": 773,
        "model.lightgbm_configs": 0,
        "model.trained_folds": 0,
        "model.boosters": 0,
        "comparison.saved_control_candidate": PARENT_CANDIDATE,
        "comparison.saved_control_expected_decompressed_sha256": (
            EXPECTED_PARENT_DECOMPRESSED_SHA256
        ),
        "execution.run_control": False,
        "execution.run_inference": False,
        "execution.write_submission": False,
    }
    for key, expected in exact_values.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"contract mismatch {key}: expected={expected!r} actual={actual!r}")
    if not bool(get_nested(config, "execution.implementation")):
        raise ValueError("implementation flag must be true after implementation")
    if require_run_approval:
        if not bool(get_nested(config, "execution.run_variant")):
            raise RuntimeError("execution.run_variant is false")
        if not bool(get_nested(config, "execution.kaggle_cpu_push_approved")):
            raise RuntimeError("Kaggle CPU push is not approved")


# %% [markdown]
# ## 4. Exp223 prefix and self-GR surface helpers


# %%
def load_generation_well(well: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(data_dir)
    horizontal = pd.read_csv(
        directory / f"{well}__horizontal_well.csv",
        usecols=list(GENERATION_HORIZONTAL_COLUMNS),
    )
    typewell = (
        pd.read_csv(
            directory / f"{well}__typewell.csv",
            usecols=list(GENERATION_TYPEWELL_COLUMNS),
        )
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    if tuple(horizontal.columns) != GENERATION_HORIZONTAL_COLUMNS:
        horizontal = horizontal.loc[:, list(GENERATION_HORIZONTAL_COLUMNS)]
    return horizontal, typewell


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    tail_n: int = 30,
) -> tuple[float, float, float, float]:
    known = horizontal[horizontal["TVT_input"].notna()]
    if known.empty:
        return 1.0, 0.0, 30.0, 0.0
    known_gr = known["GR"].to_numpy(np.float64)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    valid = np.isfinite(known_gr) & np.isfinite(typewell_at_known)
    if valid.sum() >= 20 and np.std(typewell_at_known[valid]) > 1e-6:
        cal_a, cal_b = np.polyfit(typewell_at_known[valid], known_gr[valid], 1)
    elif valid.any():
        cal_a = 1.0
        cal_b = float(np.nanmean(known_gr) - np.nanmean(typewell_at_known))
    else:
        cal_a = 1.0
        cal_b = 0.0

    residual = known_gr[valid] - (cal_a * typewell_at_known[valid] + cal_b)
    if valid.sum() > 20:
        sigma = float(
            np.clip(
                1.4826 * np.median(np.abs(residual - np.median(residual))),
                8.0,
                60.0,
            )
        )
    else:
        sigma = 30.0

    tail = known.tail(tail_n)
    dtvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    dz = np.diff(tail["Z"].to_numpy(np.float64))
    dmd = np.diff(tail["MD"].to_numpy(np.float64))
    mask = dmd > 0
    init_rate = float(np.median((dtvt + dz)[mask] / dmd[mask])) if mask.sum() >= 3 else 0.0
    return float(cal_a), float(cal_b), sigma, init_rate


def _safe_interp_gr(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    fill_value = (
        float(np.nanmedian(series.to_numpy(dtype=np.float64))) if series.notna().any() else 0.0
    )
    return series.interpolate(limit_direction="both").fillna(fill_value).to_numpy(dtype=np.float64)


def build_gr_window_descriptors(
    horizontal: pd.DataFrame,
    *,
    radius: int,
    offsets: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(gr_raw).astype(np.float64)
    gr = _safe_interp_gr(gr_raw)
    series = pd.Series(gr)
    window = int(2 * radius + 1)
    roll_mean = series.rolling(window=window, center=True, min_periods=max(3, radius // 2)).mean()
    roll_std = series.rolling(window=window, center=True, min_periods=max(3, radius // 2)).std(
        ddof=0
    )
    mean = (
        roll_mean.interpolate(limit_direction="both")
        .fillna(float(np.mean(gr)))
        .to_numpy(np.float64)
    )
    std = (
        roll_std.interpolate(limit_direction="both")
        .fillna(float(np.std(gr) if np.std(gr) > 1e-6 else 1.0))
        .to_numpy(np.float64)
    )
    std = np.clip(std, 1.0, None)
    missing_rate = 1.0 - (
        pd.Series(finite)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )

    descriptors: list[np.ndarray] = []
    for offset in offsets:
        shifted = (
            pd.Series(gr).shift(-int(offset)).interpolate(limit_direction="both").bfill().ffill()
        )
        descriptors.append(((shifted.to_numpy(np.float64) - mean) / std).astype(np.float64))
    global_std = float(np.std(gr) if np.std(gr) > 1e-6 else 1.0)
    descriptors.append(((mean - float(np.mean(gr))) / global_std).astype(np.float64))
    descriptors.append(np.log1p(std).astype(np.float64))
    if radius > 0:
        left = (
            pd.Series(gr)
            .shift(radius)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(np.float64)
        )
        right = (
            pd.Series(gr)
            .shift(-radius)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(np.float64)
        )
        descriptors.append(((right - left) / (2.0 * radius * std)).astype(np.float64))
    matrix = np.vstack(descriptors).T
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix.astype(np.float32), np.clip(missing_rate, 0.0, 1.0).astype(np.float32)


def select_prefix_anchor_indices(
    known_indices: np.ndarray,
    *,
    radius: int,
    stride: int,
    max_anchors: int,
    keep_last: int,
) -> np.ndarray:
    if len(known_indices) == 0:
        return np.array([], dtype=np.int64)
    last_known = int(np.max(known_indices))
    usable = known_indices[known_indices <= last_known - int(radius)]
    if len(usable) == 0:
        usable = known_indices
    selected = usable[:: max(1, int(stride))]
    if keep_last > 0:
        selected = np.unique(np.concatenate([selected, usable[-int(keep_last) :]])).astype(np.int64)
    if max_anchors > 0 and len(selected) > max_anchors:
        take = np.linspace(0, len(selected) - 1, int(max_anchors)).round().astype(np.int64)
        selected = selected[take]
    return selected.astype(np.int64)


def _empty_self_gr_surface(n_eval: int, n_grid: int, anchor_count: int = 0) -> dict[str, Any]:
    zero_surface = np.zeros((n_eval, n_grid), dtype=np.float32)
    zero_vector = np.zeros(n_eval, dtype=np.float32)
    return {
        "centered_logl": zero_surface,
        "quality": zero_vector,
        "peak_tvt": zero_vector.astype(np.float64),
        "peak_gap": zero_vector,
        "typewell_agreement": zero_vector,
        "valid": zero_vector,
        "prefix_anchor_count": int(anchor_count),
    }


def build_self_gr_likelihood_surface(
    horizontal: pd.DataFrame,
    eval_index: np.ndarray,
    grid: np.ndarray,
    typewell_peak_tvt: np.ndarray,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    radius = int(config.get("window_radius_rows", 12))
    offsets = [
        int(value) for value in (config.get("descriptor_offsets") or [-12, -8, -4, 0, 4, 8, 12])
    ]
    top_k = max(1, int(config.get("top_k", 5)))
    stride = max(1, int(config.get("prefix_anchor_stride", 3)))
    max_anchors = int(config.get("max_prefix_anchors", 128))
    keep_last = int(config.get("keep_last_prefix_anchors", 32))
    min_anchors = max(1, int(config.get("min_prefix_anchors", 12)))
    max_missing_rate = float(config.get("max_window_missing_rate", 0.35))
    sigma_tvt = max(1e-6, float(config.get("gaussian_sigma_tvt", 12.0)))
    distance_temperature = max(1e-6, float(config.get("descriptor_distance_temperature", 1.5)))
    agreement_sigma = max(1e-6, float(config.get("typewell_agreement_sigma_tvt", 18.0)))
    surface_clip = float(config.get("surface_quadratic_clip", 60.0))
    chunk_size = max(1, int(config.get("surface_chunk_size", 256)))
    n_eval, n_grid = len(eval_index), len(grid)
    if n_eval == 0 or n_grid == 0:
        return _empty_self_gr_surface(n_eval, n_grid)

    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    anchor_indices = select_prefix_anchor_indices(
        known_indices,
        radius=radius,
        stride=stride,
        max_anchors=max_anchors,
        keep_last=keep_last,
    )
    if len(anchor_indices) < min_anchors:
        return _empty_self_gr_surface(n_eval, n_grid, len(anchor_indices))

    descriptors, missing_rate = build_gr_window_descriptors(
        horizontal, radius=radius, offsets=offsets
    )
    anchor_indices = anchor_indices[missing_rate[anchor_indices] <= max_missing_rate]
    if len(anchor_indices) < min_anchors:
        return _empty_self_gr_surface(n_eval, n_grid, len(anchor_indices))

    anchor_desc = descriptors[anchor_indices].astype(np.float32)
    anchor_tvt = tvt_input[anchor_indices].astype(np.float64)
    eval_desc = descriptors[eval_index].astype(np.float32)
    eval_missing = missing_rate[eval_index].astype(np.float32)
    prefix_coverage_quality = float(
        np.clip(len(anchor_indices) / max(float(min_anchors), 1.0), 0.0, 1.0)
    )
    centered = np.zeros((n_eval, n_grid), dtype=np.float32)
    quality = np.zeros(n_eval, dtype=np.float32)
    peak_tvt = np.full(n_eval, np.nan, dtype=np.float64)
    peak_gap = np.zeros(n_eval, dtype=np.float32)
    agreement = np.zeros(n_eval, dtype=np.float32)
    valid = np.zeros(n_eval, dtype=np.float32)
    k_eff = min(top_k, len(anchor_indices))
    eps = 1e-6
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        desc = eval_desc[start:end]
        diff = desc[:, None, :] - anchor_desc[None, :, :]
        cost = np.mean(diff * diff, axis=2)
        if k_eff < cost.shape[1]:
            top_idx_unsorted = np.argpartition(cost, kth=k_eff - 1, axis=1)[:, :k_eff]
        else:
            top_idx_unsorted = np.tile(np.arange(cost.shape[1]), (cost.shape[0], 1))
        top_cost_unsorted = np.take_along_axis(cost, top_idx_unsorted, axis=1)
        order = np.argsort(top_cost_unsorted, axis=1)
        top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
        top_cost = np.take_along_axis(top_cost_unsorted, order, axis=1)
        centers = anchor_tvt[top_idx]
        rel_cost = top_cost - top_cost[:, :1]
        weights = np.exp(-rel_cost / (2.0 * distance_temperature**2))
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), eps, None)
        z_score = (grid[None, None, :] - centers[:, :, None]) / sigma_tvt
        component_ll = np.log(np.clip(weights, eps, None))[:, :, None] - 0.5 * np.minimum(
            z_score * z_score,
            surface_clip,
        )
        best = np.max(component_ll, axis=1)
        log_likelihood = best + np.log(
            np.clip(np.exp(component_ll - best[:, None, :]).sum(axis=1), eps, None)
        )
        centered_chunk = log_likelihood - np.mean(log_likelihood, axis=1, keepdims=True)
        scale = np.std(centered_chunk, axis=1, keepdims=True)
        centered[start:end] = (centered_chunk / np.clip(scale, 0.25, None)).astype(np.float32)

        cost_q75 = np.quantile(cost, 0.75, axis=1)
        sharpness = np.clip((cost_q75 - top_cost[:, 0]) / np.clip(cost_q75, eps, None), 0.0, 1.0)
        gap = (
            top_cost[:, 1] - top_cost[:, 0]
            if k_eff >= 2
            else np.zeros(end - start, dtype=np.float32)
        )
        gap_quality = np.clip(gap / max(distance_temperature, eps), 0.0, 1.0)
        peak = centers[:, 0]
        agree = np.exp(-0.5 * ((peak - typewell_peak_tvt[start:end]) / agreement_sigma) ** 2)
        miss_quality = np.clip(1.0 - eval_missing[start:end], 0.0, 1.0)
        row_quality = (
            prefix_coverage_quality
            * miss_quality
            * (0.25 + 0.75 * sharpness)
            * (0.25 + 0.75 * gap_quality)
            * (0.15 + 0.85 * agree)
        )
        row_valid = (
            np.isfinite(peak)
            & np.isfinite(row_quality)
            & (eval_missing[start:end] <= max_missing_rate)
        )
        quality[start:end] = np.where(row_valid, np.clip(row_quality, 0.0, 1.0), 0.0).astype(
            np.float32
        )
        peak_tvt[start:end] = peak
        peak_gap[start:end] = gap.astype(np.float32)
        agreement[start:end] = agree.astype(np.float32)
        valid[start:end] = row_valid.astype(np.float32)

    return {
        "centered_logl": centered,
        "quality": quality,
        "peak_tvt": peak_tvt,
        "peak_gap": peak_gap,
        "typewell_agreement": agreement,
        "valid": valid,
        "prefix_anchor_count": int(len(anchor_indices)),
    }


# %% [markdown]
# ## 5. Known-TVT candidate-state support gate


# %%
def build_candidate_state_support_mask(
    grid: np.ndarray,
    tvt_input: pd.Series | np.ndarray,
) -> tuple[np.ndarray, float, float]:
    states = np.asarray(grid, dtype=np.float64)
    known = np.asarray(tvt_input, dtype=np.float64)
    finite = known[np.isfinite(known)]
    if finite.size == 0:
        return np.zeros(states.shape, dtype=bool), float("nan"), float("nan")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    support = (states >= lower) & (states <= upper)
    return support.astype(bool, copy=False), lower, upper


def apply_known_tvt_support_gate(
    exp223_boost: np.ndarray,
    grid: np.ndarray,
    tvt_input: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    boost = np.asarray(exp223_boost)
    if boost.ndim != 2:
        raise ValueError(f"exp223 boost must be 2D, got shape={boost.shape}")
    support, lower, upper = build_candidate_state_support_mask(grid, tvt_input)
    if boost.shape[1] != len(support):
        raise ValueError(f"boost/grid shape mismatch: boost={boost.shape} grid={len(support)}")
    gated = boost.copy()
    gated[:, ~support] = np.array(0.0, dtype=boost.dtype)
    if support.any() and not np.array_equal(gated[:, support], boost[:, support]):
        raise AssertionError("inside-support boost parity failed")
    if (~support).any() and np.max(np.abs(gated[:, ~support])) != 0.0:
        raise AssertionError("outside-support boost is not exact zero")
    return gated, support, lower, upper


def support_mask_identity(
    grid: np.ndarray,
    support: np.ndarray,
    known_tvt_min: float,
    known_tvt_max: float,
) -> dict[str, Any]:
    return {
        "grid_size": int(len(grid)),
        "grid_min": float(grid[0]) if len(grid) else None,
        "grid_max": float(grid[-1]) if len(grid) else None,
        "grid_sha256": sha256_array(np.asarray(grid, dtype=np.float64)),
        "support_state_count": int(np.sum(support)),
        "support_state_rate": float(np.mean(support)) if len(support) else 0.0,
        "support_mask_sha256": sha256_array(np.asarray(support, dtype=np.uint8)),
        "known_tvt_min": known_tvt_min,
        "known_tvt_max": known_tvt_max,
    }


# %% [markdown]
# ## 6. Exact exp223 forward-backward kernel


# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb(
    emission,
    delta_md,
    delta_z,
    step,
    rates,
    sigma_rate,
    sigma_position,
    start_position,
    start_sigma,
    initial_rate,
    initial_rate_sigma,
    likelihood_weight,
    momentum,
):
    """Amerhu/exp209 exact forward-backward over (TVT position, dip-rate)."""
    time_count, position_count = emission.shape
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    negative = np.float32(-1e18)
    alpha = np.full((time_count, position_count, rate_count), negative, np.float32)
    previous = np.full((position_count, rate_count), negative, np.float32)
    for position_index in range(position_count):
        delta_position = (position_index - start_position) * step
        start_log_probability = -0.5 * (delta_position / start_sigma) ** 2
        if start_log_probability < -60.0:
            continue
        for rate_index in range(rate_count):
            standardized_rate = (rates[rate_index] - initial_rate) / initial_rate_sigma
            previous[position_index, rate_index] = np.float32(
                start_log_probability - 0.5 * standardized_rate * standardized_rate
            )

    intermediate = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count):
        sigma_rate_step = sigma_rate * np.sqrt(delta_md[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((rate_count, 3))
        for rate_index in range(rate_count):
            mean_rate_move = (
                -(1.0 - momentum) * rates[rate_index] * delta_md[time_index] / rate_step
            )
            probability_plus = 0.5 * (rate_variance_cells + mean_rate_move)
            probability_minus = 0.5 * (rate_variance_cells - mean_rate_move)
            if probability_plus < 1e-12:
                probability_plus = 1e-12
            if probability_minus < 1e-12:
                probability_minus = 1e-12
            probability_total = probability_plus + probability_minus
            if probability_total > 0.9:
                probability_plus *= 0.9 / probability_total
                probability_minus *= 0.9 / probability_total
            rate_log_kernel[rate_index, 0] = np.log(probability_minus)
            rate_log_kernel[rate_index, 1] = np.log(1.0 - probability_plus - probability_minus)
            rate_log_kernel[rate_index, 2] = np.log(probability_plus)

        for position_index in prange(position_count):
            for next_rate in range(rate_count):
                best = negative
                lower = next_rate - 1 if next_rate - 1 >= 0 else 0
                upper = next_rate + 1 if next_rate + 1 <= rate_count - 1 else rate_count - 1
                for rate_index in range(lower, upper + 1):
                    value = (
                        previous[position_index, rate_index]
                        + rate_log_kernel[
                            rate_index,
                            next_rate - rate_index + 1,
                        ]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for rate_index in range(lower, upper + 1):
                        total += np.exp(
                            previous[position_index, rate_index]
                            + rate_log_kernel[rate_index, next_rate - rate_index + 1]
                            - best
                        )
                    intermediate[position_index, next_rate] = np.float32(best + np.log(total))
                else:
                    intermediate[position_index, next_rate] = negative

        effective_position_sigma = sigma_position if sigma_position > 0.35 * step else 0.35 * step
        for next_rate in range(rate_count):
            mean = rates[next_rate] * delta_md[time_index] - delta_z[time_index]
            center = int(np.floor(mean / step + 0.5))
            position_log_kernel = np.empty(5)
            for kernel_index in range(5):
                delta = (center - 2 + kernel_index) * step - mean
                position_log_kernel[kernel_index] = -0.5 * (delta / effective_position_sigma) ** 2
            kernel_max = position_log_kernel[0]
            for kernel_index in range(1, 5):
                if position_log_kernel[kernel_index] > kernel_max:
                    kernel_max = position_log_kernel[kernel_index]
            kernel_sum = 0.0
            for kernel_index in range(5):
                kernel_sum += np.exp(position_log_kernel[kernel_index] - kernel_max)
            log_normalizer = kernel_max + np.log(kernel_sum)
            for kernel_index in range(5):
                position_log_kernel[kernel_index] -= log_normalizer
            for next_position in prange(position_count):
                best = negative
                for kernel_index in range(5):
                    previous_position = next_position - (center - 2 + kernel_index)
                    if previous_position < 0 or previous_position >= position_count:
                        continue
                    value = (
                        intermediate[previous_position, next_rate]
                        + position_log_kernel[kernel_index]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        previous_position = next_position - (center - 2 + kernel_index)
                        if previous_position < 0 or previous_position >= position_count:
                            continue
                        total += np.exp(
                            intermediate[previous_position, next_rate]
                            + position_log_kernel[kernel_index]
                            - best
                        )
                    current[next_position, next_rate] = np.float32(
                        best
                        + np.log(total)
                        + likelihood_weight * emission[time_index, next_position]
                    )
                else:
                    current[next_position, next_rate] = negative
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                alpha[time_index, position_index, rate_index] = current[position_index, rate_index]
                previous[position_index, rate_index] = current[position_index, rate_index]

    best = np.float32(negative)
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            if alpha[time_count - 1, position_index, rate_index] > best:
                best = alpha[time_count - 1, position_index, rate_index]
    total = 0.0
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            total += np.exp(alpha[time_count - 1, position_index, rate_index] - best)
    log_likelihood = float(best) + np.log(total)

    posterior_position = np.zeros((time_count, position_count))
    beta_next = np.zeros((position_count, rate_count), np.float32)
    best = negative
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            value = (
                alpha[time_count - 1, position_index, rate_index]
                + beta_next[position_index, rate_index]
            )
            if value > best:
                best = value
    total = 0.0
    for position_index in range(position_count):
        accumulated = 0.0
        for rate_index in range(rate_count):
            accumulated += np.exp(
                alpha[time_count - 1, position_index, rate_index]
                + beta_next[position_index, rate_index]
                - best
            )
        posterior_position[time_count - 1, position_index] = accumulated
        total += accumulated
    for position_index in range(position_count):
        posterior_position[time_count - 1, position_index] /= total

    beta_current = np.empty((position_count, rate_count), np.float32)
    beta_intermediate = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count - 1, 0, -1):
        sigma_rate_step = sigma_rate * np.sqrt(delta_md[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((rate_count, 3))
        for rate_index in range(rate_count):
            mean_rate_move = (
                -(1.0 - momentum) * rates[rate_index] * delta_md[time_index] / rate_step
            )
            probability_plus = 0.5 * (rate_variance_cells + mean_rate_move)
            probability_minus = 0.5 * (rate_variance_cells - mean_rate_move)
            if probability_plus < 1e-12:
                probability_plus = 1e-12
            if probability_minus < 1e-12:
                probability_minus = 1e-12
            probability_total = probability_plus + probability_minus
            if probability_total > 0.9:
                probability_plus *= 0.9 / probability_total
                probability_minus *= 0.9 / probability_total
            rate_log_kernel[rate_index, 0] = np.log(probability_minus)
            rate_log_kernel[rate_index, 1] = np.log(1.0 - probability_plus - probability_minus)
            rate_log_kernel[rate_index, 2] = np.log(probability_plus)
        effective_position_sigma = sigma_position if sigma_position > 0.35 * step else 0.35 * step
        for next_rate in range(rate_count):
            mean = rates[next_rate] * delta_md[time_index] - delta_z[time_index]
            center = int(np.floor(mean / step + 0.5))
            position_log_kernel = np.empty(5)
            for kernel_index in range(5):
                delta = (center - 2 + kernel_index) * step - mean
                position_log_kernel[kernel_index] = -0.5 * (delta / effective_position_sigma) ** 2
            kernel_max = position_log_kernel[0]
            for kernel_index in range(1, 5):
                if position_log_kernel[kernel_index] > kernel_max:
                    kernel_max = position_log_kernel[kernel_index]
            kernel_sum = 0.0
            for kernel_index in range(5):
                kernel_sum += np.exp(position_log_kernel[kernel_index] - kernel_max)
            log_normalizer = kernel_max + np.log(kernel_sum)
            for kernel_index in range(5):
                position_log_kernel[kernel_index] -= log_normalizer
            for previous_position in prange(position_count):
                best = negative
                for kernel_index in range(5):
                    next_position = previous_position + (center - 2 + kernel_index)
                    if next_position < 0 or next_position >= position_count:
                        continue
                    value = (
                        position_log_kernel[kernel_index]
                        + likelihood_weight * emission[time_index, next_position]
                        + beta_next[next_position, next_rate]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        next_position = previous_position + (center - 2 + kernel_index)
                        if next_position < 0 or next_position >= position_count:
                            continue
                        total += np.exp(
                            position_log_kernel[kernel_index]
                            + likelihood_weight * emission[time_index, next_position]
                            + beta_next[next_position, next_rate]
                            - best
                        )
                    beta_intermediate[previous_position, next_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    beta_intermediate[previous_position, next_rate] = negative

        for position_index in prange(position_count):
            for rate_index in range(rate_count):
                best = negative
                lower = rate_index - 1 if rate_index - 1 >= 0 else 0
                upper = rate_index + 1 if rate_index + 1 <= rate_count - 1 else rate_count - 1
                for next_rate in range(lower, upper + 1):
                    value = (
                        rate_log_kernel[rate_index, next_rate - rate_index + 1]
                        + beta_intermediate[position_index, next_rate]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for next_rate in range(lower, upper + 1):
                        total += np.exp(
                            rate_log_kernel[rate_index, next_rate - rate_index + 1]
                            + beta_intermediate[position_index, next_rate]
                            - best
                        )
                    beta_current[position_index, rate_index] = np.float32(best + np.log(total))
                else:
                    beta_current[position_index, rate_index] = negative

        best = negative
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                value = (
                    alpha[time_index - 1, position_index, rate_index]
                    + beta_current[
                        position_index,
                        rate_index,
                    ]
                )
                if value > best:
                    best = value
        total = 0.0
        for position_index in range(position_count):
            accumulated = 0.0
            for rate_index in range(rate_count):
                accumulated += np.exp(
                    alpha[time_index - 1, position_index, rate_index]
                    + beta_current[position_index, rate_index]
                    - best
                )
            posterior_position[time_index - 1, position_index] = accumulated
            total += accumulated
        for position_index in range(position_count):
            posterior_position[time_index - 1, position_index] /= total
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                beta_next[position_index, rate_index] = beta_current[position_index, rate_index]
    return posterior_position, log_likelihood


def _hmm_start_context(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, bool]:
    known = horizontal[horizontal["TVT_input"].notna()]
    eval_rows = horizontal[horizontal["TVT_input"].isna()]
    if eval_rows.empty:
        return known, eval_rows, float("nan"), float("nan"), float("nan"), False
    if not known.empty:
        last = known.iloc[-1]
        return (
            known,
            eval_rows,
            float(last["TVT_input"]),
            float(last["MD"]),
            float(last["Z"]),
            False,
        )
    md = horizontal["MD"].to_numpy(np.float64)
    z_values = horizontal["Z"].to_numpy(np.float64)
    first_step = float(max(md[1] - md[0], 1.0)) if len(md) > 1 else 1.0
    return (
        known,
        eval_rows,
        float(np.median(typewell_tvt)),
        float(md[0] - first_step),
        float(z_values[0]),
        True,
    )


def run_hmm2_known_tvt_support_gate(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    step: float = 0.35,
    n_rates: int = 41,
    rate_span: float = 0.10,
    sig_r: float = 0.002,
    sig_p: float = 0.02,
    df: float = 4.0,
    emission: str = "gauss",
    lam: float = 1.0,
    sigma_mode: str = "std",
    start_sig: float = 0.75,
    r0_sig: float = 0.01,
    band_pad: float = 100.0,
    mom: float = 0.998,
    rate_center: str = "zero",
    self_gr_config: Mapping[str, Any] | None = None,
    self_gr_alpha: float = 0.07,
    self_gr_clip: float = 1.0,
    self_gr_mode: str = "boost_only",
    return_debug_matrices: bool = False,
) -> dict[str, Any]:
    if self_gr_mode != "boost_only":
        raise ValueError("exp296 fixes exp223 boost_only mode")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known, eval_rows, last_tvt, last_md, last_z, no_known_fallback = _hmm_start_context(
        horizontal,
        typewell_tvt,
    )
    if eval_rows.empty:
        return {
            "mean_eval": np.array([], dtype=np.float64),
            "std_eval": np.array([], dtype=np.float64),
            "ev_index": np.array([], dtype=np.int64),
            "grid": np.array([], dtype=np.float64),
            "loglik": 0.0,
            "last_known_tvt": last_tvt,
            "last_known_md": last_md,
            "no_known_tvt_fallback": no_known_fallback,
        }

    cal_a, cal_b, robust_sigma, init_rate = prefix_stats(horizontal, typewell_tvt, typewell_gr)
    if sigma_mode == "std" and not known.empty:
        typewell_at_known = np.interp(
            known["TVT_input"].to_numpy(np.float64), typewell_tvt, typewell_gr
        )
        gr_residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
        gr_sigma = float(np.clip(np.nanstd(gr_residual), 10.0, 60.0))
        cal_a_use, cal_b_use = 1.0, 0.0
    elif sigma_mode == "std":
        gr_sigma = 30.0
        cal_a_use, cal_b_use = 1.0, 0.0
    else:
        gr_sigma = robust_sigma
        cal_a_use, cal_b_use = cal_a, cal_b

    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - band_pad)
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + band_pad)
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = cal_a_use * np.interp(grid, typewell_tvt, typewell_gr) + cal_b_use
    eval_index = eval_rows.index.to_numpy(np.int64)
    md = eval_rows["MD"].to_numpy(np.float64)
    z_values = eval_rows["Z"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[eval_index]
    )
    delta_md = np.maximum(np.diff(np.concatenate([[last_md], md])), 1.0)
    delta_z = np.diff(np.concatenate([[last_z], z_values]))
    z_score = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    if emission == "t":
        base_emission = (-0.5 * (df + 1.0) * np.log1p(z_score**2 / df)).astype(np.float32)
    else:
        base_emission = (-0.5 * np.minimum(z_score**2, 600.0)).astype(np.float32)

    typewell_peak_tvt = grid[np.argmax(base_emission, axis=1)]
    self_surface = build_self_gr_likelihood_surface(
        horizontal,
        eval_index,
        grid,
        typewell_peak_tvt,
        self_gr_config,
    )
    centered_self_ll = np.asarray(self_surface["centered_logl"], dtype=np.float32)
    quality_self = np.asarray(self_surface["quality"], dtype=np.float32)
    if centered_self_ll.shape != base_emission.shape:
        raise ValueError(
            "self-GR surface shape mismatch: "
            f"expected={base_emission.shape} actual={centered_self_ll.shape}"
        )
    exp223_boost = np.clip(centered_self_ll, 0.0, float(self_gr_clip)).astype(
        np.float32, copy=False
    )
    gated_boost, support, known_tvt_min, known_tvt_max = apply_known_tvt_support_gate(
        exp223_boost,
        grid,
        horizontal["TVT_input"].to_numpy(np.float64),
    )
    contribution = (
        np.float32(float(self_gr_alpha)) * quality_self[:, None].astype(np.float32) * gated_boost
    ).astype(np.float32)
    emission_ll = (base_emission + contribution).astype(np.float32)

    if rate_center == "zero":
        span = max(rate_span, abs(init_rate) + 0.04)
        rates = np.linspace(-span, span, n_rates, dtype=np.float64)
    else:
        rates = init_rate + np.linspace(-rate_span, rate_span, n_rates, dtype=np.float64)
    start_position = float((last_tvt - grid_min) / step)
    posterior, log_likelihood = _hmm2_fb(
        emission_ll,
        delta_md.astype(np.float64),
        delta_z.astype(np.float64),
        float(step),
        rates,
        float(sig_r),
        float(sig_p),
        start_position,
        float(start_sig),
        float(init_rate),
        float(r0_sig),
        float(lam),
        float(mom),
    )
    mean = posterior @ grid
    variance = posterior @ (grid**2) - mean**2
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    posterior_outside = (
        posterior[:, ~support].sum(axis=1) if (~support).any() else np.zeros(len(mean))
    )
    outside_boost_max = float(np.max(np.abs(gated_boost[:, ~support]))) if (~support).any() else 0.0
    inside_boost_delta_max = (
        float(np.max(np.abs(gated_boost[:, support] - exp223_boost[:, support])))
        if support.any()
        else 0.0
    )
    outside_contribution_max = (
        float(np.max(np.abs(contribution[:, ~support]))) if (~support).any() else 0.0
    )
    identity = support_mask_identity(grid, support, known_tvt_min, known_tvt_max)
    result: dict[str, Any] = {
        "mean_eval": mean,
        "std_eval": standard_deviation,
        "loglik": float(log_likelihood),
        "ev_index": eval_index,
        "grid": grid,
        "last_known_tvt": last_tvt,
        "last_known_md": last_md,
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "cal_a": cal_a,
        "cal_b": cal_b,
        "self_gr_quality": quality_self,
        "self_gr_peak_tvt": np.asarray(self_surface["peak_tvt"], dtype=np.float64),
        "self_gr_peak_gap": np.asarray(self_surface["peak_gap"], dtype=np.float32),
        "self_gr_typewell_agreement": np.asarray(
            self_surface["typewell_agreement"], dtype=np.float32
        ),
        "self_gr_valid": np.asarray(self_surface["valid"], dtype=np.float32),
        "self_gr_prefix_anchor_count": int(self_surface["prefix_anchor_count"]),
        "known_tvt_min": known_tvt_min,
        "known_tvt_max": known_tvt_max,
        "support_state_count": int(identity["support_state_count"]),
        "support_state_rate": float(identity["support_state_rate"]),
        "support_mask_sha256": identity["support_mask_sha256"],
        "grid_sha256": identity["grid_sha256"],
        "posterior_outside_support_mass": posterior_outside,
        "outside_support_boost_max_abs": outside_boost_max,
        "inside_support_boost_delta_max_abs": inside_boost_delta_max,
        "outside_support_contribution_max_abs": outside_contribution_max,
        "no_known_tvt_fallback": no_known_fallback,
    }
    if return_debug_matrices:
        result.update(
            {
                "base_emission_ll": base_emission,
                "exp223_boost": exp223_boost,
                "gated_boost": gated_boost,
                "gated_emission_ll": emission_ll,
                "support_mask": support,
                "posterior": posterior,
            }
        )
    return result


# %% [markdown]
# ## 7. Single-well prediction generation


# %%
HMM_RUNTIME_KEYS = {
    "step",
    "n_rates",
    "rate_span",
    "sig_r",
    "sig_p",
    "df",
    "emission",
    "lam",
    "sigma_mode",
    "start_sig",
    "r0_sig",
    "band_pad",
    "mom",
    "rate_center",
}


def hmm_runtime_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(get_nested(config, "model.hmm") or {})
    values.pop("source", None)
    unexpected = sorted(set(values) - HMM_RUNTIME_KEYS)
    if unexpected:
        raise ValueError(f"unexpected HMM runtime config keys: {unexpected}")
    return values


def build_prediction_rows_for_well(
    well: str,
    data_dir: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal, typewell = load_generation_well(well, data_dir)
    result = run_hmm2_known_tvt_support_gate(
        horizontal,
        typewell,
        **hmm_runtime_kwargs(config),
        self_gr_config=dict(get_nested(config, "model.self_gr_emission") or {}),
        self_gr_alpha=float(get_nested(config, "model.self_gr_emission.alpha")),
        self_gr_clip=float(get_nested(config, "model.self_gr_emission.clip")),
        self_gr_mode=str(get_nested(config, "model.self_gr_emission.mode")),
    )
    eval_index = np.asarray(result["ev_index"], dtype=np.int64)
    if len(eval_index) == 0:
        return pd.DataFrame(), {
            "well": well,
            "status": "skipped_no_eval_rows",
            "rows": 0,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    prediction = np.asarray(result["mean_eval"], dtype=np.float64)
    standard_deviation = np.asarray(result["std_eval"], dtype=np.float64)
    posterior_outside = np.asarray(result["posterior_outside_support_mass"], dtype=np.float64)
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(standard_deviation).all()
        and np.isfinite(posterior_outside).all()
    ):
        raise ValueError(f"non-finite exp296 output for well={well}")
    last_known_tvt = float(result["last_known_tvt"])
    last_known_md = float(result["last_known_md"])
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row_index)}" for row_index in eval_index],
            "well": str(well),
            "row_index": eval_index,
            "reporting_fold": stable_reporting_fold(str(well), 5),
            "last_known_tvt": last_known_tvt,
            "md_since": horizontal.loc[eval_index, "MD"].to_numpy(np.float64) - last_known_md,
            "known_tvt_min": float(result["known_tvt_min"]),
            "known_tvt_max": float(result["known_tvt_max"]),
            PREDICTION_COLUMN: prediction,
            f"{VARIANT}_std": standard_deviation,
            f"{VARIANT}_loglik": float(result["loglik"]),
            "posterior_outside_support_mass": posterior_outside,
            "self_gr_quality": np.asarray(result["self_gr_quality"], dtype=np.float32),
            "self_gr_peak_tvt": np.asarray(result["self_gr_peak_tvt"], dtype=np.float64),
            "self_gr_peak_gap": np.asarray(result["self_gr_peak_gap"], dtype=np.float32),
            "self_gr_typewell_agreement": np.asarray(
                result["self_gr_typewell_agreement"],
                dtype=np.float32,
            ),
            "self_gr_valid": np.asarray(result["self_gr_valid"], dtype=np.float32),
        }
    )
    meta = {
        "well": str(well),
        "status": "ok",
        "rows": int(len(frame)),
        "reporting_fold": stable_reporting_fold(str(well), 5),
        "known_tvt_min": float(result["known_tvt_min"]),
        "known_tvt_max": float(result["known_tvt_max"]),
        "grid_size": int(len(result["grid"])),
        "grid_min": float(result["grid"][0]),
        "grid_max": float(result["grid"][-1]),
        "grid_sha256": str(result["grid_sha256"]),
        "support_state_count": int(result["support_state_count"]),
        "support_state_rate": float(result["support_state_rate"]),
        "support_mask_sha256": str(result["support_mask_sha256"]),
        "outside_support_boost_max_abs": float(result["outside_support_boost_max_abs"]),
        "inside_support_boost_delta_max_abs": float(result["inside_support_boost_delta_max_abs"]),
        "outside_support_contribution_max_abs": float(
            result["outside_support_contribution_max_abs"]
        ),
        "posterior_outside_support_mass_mean": float(np.mean(posterior_outside)),
        "posterior_outside_support_mass_p95": float(np.quantile(posterior_outside, 0.95)),
        "self_gr_prefix_anchor_count": int(result["self_gr_prefix_anchor_count"]),
        "self_gr_quality_mean": float(
            np.mean(np.asarray(result["self_gr_quality"], dtype=np.float64))
        ),
        "self_gr_valid_rate": float(np.mean(np.asarray(result["self_gr_valid"], dtype=np.float64))),
        "no_known_tvt_fallback": bool(result["no_known_tvt_fallback"]),
        "prediction_finite_rate": float(np.mean(np.isfinite(prediction))),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return frame, meta


def prediction_schema(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column_index": np.arange(len(frame.columns), dtype=np.int32),
            "column": frame.columns.astype(str),
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "truth_available_before_freeze": False,
        }
    )


def load_unknown_suffix_truth(data_dir: str | Path, wells: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    directory = Path(data_dir)
    for well in wells:
        horizontal = pd.read_csv(
            directory / f"{well}__horizontal_well.csv",
            usecols=["TVT", "TVT_input"],
        )
        eval_mask = horizontal["TVT_input"].isna().to_numpy()
        row_index = horizontal.index.to_numpy(np.int64)[eval_mask]
        truth = pd.to_numeric(horizontal.loc[eval_mask, "TVT"], errors="coerce").to_numpy(
            np.float64
        )
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(index)}" for index in row_index],
                    "well": str(well),
                    "row_index": row_index,
                    "true_tvt": truth,
                }
            )
        )
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if result.empty or result["id"].duplicated().any() or not np.isfinite(result["true_tvt"]).all():
        raise ValueError("invalid truth-late unknown-suffix frame")
    return result


def load_saved_exp223_control(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = list(get_nested(config, "comparison.saved_control_candidates") or [])
    if not candidates:
        filename = str(get_nested(config, "comparison.saved_control_feature_filename"))
        candidates = [filename]
    path = resolve_existing_file(candidates)
    actual_sha = sha256_gzip_decompressed(path)
    expected_sha = str(get_nested(config, "comparison.saved_control_expected_decompressed_sha256"))
    if actual_sha != expected_sha:
        raise ValueError(f"saved exp223 decompressed SHA mismatch: {actual_sha} != {expected_sha}")
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        PARENT_PREDICTION_COLUMN,
        f"{PARENT_CANDIDATE}_std",
    ]
    control = pd.read_csv(path, usecols=required, dtype={"id": str, "well": str})
    if control["id"].duplicated().any():
        raise ValueError("saved exp223 control has duplicate ids")
    control["control_true_tvt"] = pd.to_numeric(
        control["last_known_tvt"], errors="coerce"
    ).to_numpy(np.float64) + pd.to_numeric(control["target"], errors="coerce").to_numpy(np.float64)
    control = control.rename(
        columns={
            "well": "control_well",
            "last_known_tvt": "control_last_known_tvt",
            "md_since": "control_md_since",
            PARENT_PREDICTION_COLUMN: "exp223_control_tvt",
            f"{PARENT_CANDIDATE}_std": "exp223_control_std",
        }
    )
    manifest = {
        "kind": "saved_exp223_control",
        "path": str(path),
        "rows": int(len(control)),
        "wells": int(control["control_well"].nunique()),
        "raw_gzip_sha256": sha256_path(path),
        "decompressed_sha256": actual_sha,
        "expected_decompressed_sha256": expected_sha,
        "sha_exact": True,
    }
    return control, manifest


def attach_truth_and_control_after_freeze(
    frozen_predictions: pd.DataFrame,
    truth: pd.DataFrame,
    control: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_ids = frozen_predictions["id"].astype(str)
    truth_ids = truth["id"].astype(str)
    control_ids = control["id"].astype(str)
    row_identity_exact = len(prediction_ids) == len(truth_ids) == len(control_ids) and set(
        prediction_ids
    ) == set(truth_ids) == set(control_ids)
    if not row_identity_exact:
        raise ValueError(
            "row identity mismatch prediction/truth/control="
            f"{len(prediction_ids)}/{len(truth_ids)}/{len(control_ids)}"
        )
    merged = frozen_predictions.merge(
        truth,
        on=["id", "well", "row_index"],
        how="inner",
        validate="one_to_one",
    ).merge(control, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(frozen_predictions):
        raise ValueError("truth/control merge lost rows")
    if not (merged["well"].astype(str) == merged["control_well"].astype(str)).all():
        raise ValueError("saved control well identity mismatch")
    truth_delta = np.abs(
        merged["true_tvt"].to_numpy(np.float64) - merged["control_true_tvt"].to_numpy(np.float64)
    )
    if float(np.max(truth_delta)) > 0.001:
        raise ValueError(f"saved control truth parity failed: max_abs={float(np.max(truth_delta))}")
    merged["true_tvt_inside_known_range"] = (
        np.isfinite(merged["known_tvt_min"].to_numpy(np.float64))
        & np.isfinite(merged["known_tvt_max"].to_numpy(np.float64))
        & (merged["true_tvt"] >= merged["known_tvt_min"])
        & (merged["true_tvt"] <= merged["known_tvt_max"])
    )
    merged["distance_bucket"] = distance_bucket(merged["md_since"])
    audit = {
        "row_identity_exact": True,
        "rows": int(len(merged)),
        "wells": int(merged["well"].nunique()),
        "control_truth_max_abs_delta": float(np.max(truth_delta)),
        "control_well_identity_exact": True,
    }
    return merged, audit


# %% [markdown]
# ## 8. Truth-late control join and metrics


# %%
def score_prediction(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    if not finite.any():
        return {"rows": 0, "rmse": float("nan"), "mae": float("nan"), "within10": float("nan")}
    error = prediction[finite] - truth[finite]
    return {
        "rows": int(finite.sum()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def metric_rows(frame: pd.DataFrame, *, scope: str, scope_value: str) -> list[dict[str, Any]]:
    truth = frame["true_tvt"].to_numpy(np.float64)
    candidates = {
        PARENT_CANDIDATE: frame["exp223_control_tvt"].to_numpy(np.float64),
        VARIANT: frame[PREDICTION_COLUMN].to_numpy(np.float64),
    }
    rows: list[dict[str, Any]] = []
    control_metric = score_prediction(truth, candidates[PARENT_CANDIDATE])
    for candidate, prediction in candidates.items():
        metric = score_prediction(truth, prediction)
        rows.append(
            {
                "scope": scope,
                "scope_value": scope_value,
                "candidate": candidate,
                **metric,
                "delta_rmse_vs_exp223": float(metric["rmse"] - control_metric["rmse"]),
                "delta_mae_vs_exp223": float(metric["mae"] - control_metric["mae"]),
                "delta_within10_vs_exp223": float(metric["within10"] - control_metric["within10"]),
            }
        )
    return rows


def overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(metric_rows(frame, scope="overall", scope_value="all"))


def grouped_metrics(frame: pd.DataFrame, column: str, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, sort=True):
        rows.extend(metric_rows(group, scope=scope, scope_value=str(value)))
    return pd.DataFrame(rows)


def known_range_scope_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inside = frame["true_tvt_inside_known_range"].to_numpy(bool)
    for name, mask in (("inside", inside), ("outside", ~inside)):
        if mask.any():
            rows.extend(
                metric_rows(frame.loc[mask], scope="true_tvt_known_range", scope_value=name)
            )
    return pd.DataFrame(rows)


def by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        control_rmse = rmse(truth, group["exp223_control_tvt"].to_numpy(np.float64))
        candidate_rmse = rmse(truth, group[PREDICTION_COLUMN].to_numpy(np.float64))
        rows.append(
            {
                "well": str(well),
                "reporting_fold": int(group["reporting_fold"].iloc[0]),
                "rows": int(len(group)),
                "exp223_rmse": control_rmse,
                "exp296_rmse": candidate_rmse,
                "delta_rmse_vs_exp223": candidate_rmse - control_rmse,
                "true_tvt_outside_known_range_rate": float(
                    np.mean(~group["true_tvt_inside_known_range"].to_numpy(bool))
                ),
                "posterior_outside_support_mass_mean": float(
                    group["posterior_outside_support_mass"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def step_delta_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = {
        PARENT_CANDIDATE: ("exp223_control_tvt", "control_last_known_tvt"),
        VARIANT: (PREDICTION_COLUMN, "last_known_tvt"),
    }
    ordered = frame.sort_values(["well", "row_index"], kind="mergesort")
    for candidate, (prediction_column, anchor_column) in candidates.items():
        deltas: list[np.ndarray] = []
        for _, group in ordered.groupby("well", sort=False):
            prediction = group[prediction_column].to_numpy(np.float64)
            previous = np.empty(len(group), dtype=np.float64)
            previous[0] = float(group[anchor_column].iloc[0])
            if len(group) > 1:
                previous[1:] = prediction[:-1]
            deltas.append(np.abs(prediction - previous))
        values = np.concatenate(deltas) if deltas else np.array([], dtype=np.float64)
        rows.append(
            {
                "candidate": candidate,
                "rows": int(len(values)),
                "abs_step_delta_mean": float(np.mean(values)),
                "abs_step_delta_p95": float(np.quantile(values, 0.95)),
                "abs_step_delta_p99": float(np.quantile(values, 0.99)),
            }
        )
    result = pd.DataFrame(rows)
    control_p99 = float(
        result.loc[result["candidate"] == PARENT_CANDIDATE, "abs_step_delta_p99"].iloc[0]
    )
    result["delta_p99_vs_exp223"] = result["abs_step_delta_p99"] - control_p99
    return result


def hidden_like_metrics(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hidden = dict(get_nested(config, "data.hidden_like") or {})
    if not bool(hidden.get("enabled")):
        raise ValueError("hidden-like readout must remain enabled")
    path = resolve_existing_file(list(hidden.get("fold_assignment_candidates") or []))
    assignments = pd.read_csv(path, dtype={"well_id": str})
    rows: list[dict[str, Any]] = []
    for subgroup, role_column in (hidden.get("valid_role_columns") or {}).items():
        if role_column not in assignments.columns:
            raise ValueError(f"hidden-like assignment missing column={role_column}")
        valid_wells = set(
            assignments.loc[assignments[role_column].astype(str) == "valid", "well_id"].astype(str)
        )
        subset = frame.loc[frame["well"].astype(str).isin(valid_wells)]
        if subset.empty:
            raise ValueError(f"hidden-like subgroup selected zero rows: {subgroup}")
        rows.extend(metric_rows(subset, scope="hidden_like", scope_value=str(subgroup)))
    manifest = {
        "kind": "hidden_like_assignments",
        "path": str(path),
        "rows": int(len(assignments)),
        "sha256": sha256_path(path),
    }
    return pd.DataFrame(rows), manifest


def metric_value(
    table: pd.DataFrame,
    *,
    candidate: str,
    scope_value: str | None = None,
    column: str = "rmse",
) -> float:
    selected = table[table["candidate"] == candidate]
    if scope_value is not None:
        selected = selected[selected["scope_value"].astype(str) == str(scope_value)]
    if len(selected) != 1:
        return float("nan")
    return float(selected[column].iloc[0])


def delta_vs_exp223(table: pd.DataFrame, scope_value: str | None = None) -> float:
    return metric_value(
        table,
        candidate=VARIANT,
        scope_value=scope_value,
        column="delta_rmse_vs_exp223",
    )


def _gate(name: str, actual: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "actual": to_jsonable(actual),
        "operator": operator,
        "threshold": to_jsonable(threshold),
        "passed": bool(passed),
    }


def evaluate_hard_gates(
    *,
    generated: pd.DataFrame,
    well_manifest: pd.DataFrame,
    overall: pd.DataFrame,
    folds: pd.DataFrame,
    distance: pd.DataFrame,
    scopes: pd.DataFrame,
    hidden: pd.DataFrame,
    by_well: pd.DataFrame,
    steps: pd.DataFrame,
    join_audit: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "validation.hard_gates.technical") or {})
    performance_config = dict(get_nested(config, "validation.hard_gates.performance") or {})
    finite_rate = float(np.mean(np.isfinite(generated[PREDICTION_COLUMN].to_numpy(np.float64))))
    outside_contribution_max = float(well_manifest["outside_support_contribution_max_abs"].max())
    inside_boost_delta_max = float(well_manifest["inside_support_boost_delta_max_abs"].max())
    technical = [
        _gate(
            "input_wells",
            int(generated["well"].nunique()),
            "==",
            int(technical_config["input_wells"]),
            int(generated["well"].nunique()) == int(technical_config["input_wells"]),
        ),
        _gate(
            "finite_prediction_coverage",
            finite_rate,
            ">=",
            float(technical_config["finite_prediction_coverage_min"]),
            finite_rate >= float(technical_config["finite_prediction_coverage_min"]),
        ),
        _gate(
            "saved_exp223_row_identity_exact",
            bool(join_audit["row_identity_exact"]),
            "==",
            True,
            bool(join_audit["row_identity_exact"]),
        ),
        _gate(
            "saved_exp223_decompressed_sha_exact",
            bool(control_manifest["sha_exact"]),
            "==",
            True,
            bool(control_manifest["sha_exact"]),
        ),
        _gate(
            "outside_support_self_gr_contribution_max_abs",
            outside_contribution_max,
            "==",
            float(technical_config["outside_support_self_gr_contribution_max_abs"]),
            outside_contribution_max
            == float(technical_config["outside_support_self_gr_contribution_max_abs"]),
        ),
        _gate(
            "inside_support_boost_delta_vs_exp223_max_abs",
            inside_boost_delta_max,
            "==",
            float(technical_config["inside_support_boost_delta_vs_exp223_max_abs"]),
            inside_boost_delta_max
            == float(technical_config["inside_support_boost_delta_vs_exp223_max_abs"]),
        ),
        _gate("base_and_selfgr_config_parity", True, "==", True, True),
        _gate("unknown_suffix_truth_access_before_freeze", 0, "<=", 0, True),
        _gate("parent_control_retraining", 0, "<=", 0, True),
        _gate("lightgbm_configs", 0, "<=", 0, True),
        _gate("trained_folds", 0, "<=", 0, True),
        _gate("boosters", 0, "<=", 0, True),
    ]

    pooled_delta = delta_vs_exp223(overall)
    fold_candidate = folds[folds["candidate"] == VARIANT]
    improving_folds = int((fold_candidate["delta_rmse_vs_exp223"] < 0.0).sum())
    outside_delta = delta_vs_exp223(scopes, "outside")
    inside_delta = delta_vs_exp223(scopes, "inside")
    distance_1000_delta = delta_vs_exp223(distance, "1000_plus")
    hidden_spatial_delta = delta_vs_exp223(hidden, "verification_like_spatial")
    hidden_typewell_delta = delta_vs_exp223(hidden, "verification_like_typewell_purged")
    candidate_p95 = float(np.quantile(by_well["exp296_rmse"], 0.95))
    control_p95 = float(np.quantile(by_well["exp223_rmse"], 0.95))
    p95_delta = candidate_p95 - control_p95
    worst_delta = float(by_well["delta_rmse_vs_exp223"].max())
    step_delta_p99 = float(steps.loc[steps["candidate"] == VARIANT, "delta_p99_vs_exp223"].iloc[0])
    performance = [
        _gate(
            "pooled_rmse_delta_vs_exp223",
            pooled_delta,
            "<=",
            float(performance_config["pooled_rmse_delta_vs_exp223_max"]),
            pooled_delta <= float(performance_config["pooled_rmse_delta_vs_exp223_max"]),
        ),
        _gate(
            "improving_reporting_folds",
            improving_folds,
            ">=",
            int(performance_config["improving_reporting_folds_min"]),
            improving_folds >= int(performance_config["improving_reporting_folds_min"]),
        ),
        _gate(
            "true_tvt_outside_known_range_rmse_delta",
            outside_delta,
            "<=",
            float(performance_config["true_tvt_outside_known_range_rmse_delta_max"]),
            math.isfinite(outside_delta)
            and outside_delta
            <= float(performance_config["true_tvt_outside_known_range_rmse_delta_max"]),
        ),
        _gate(
            "true_tvt_inside_known_range_rmse_delta",
            inside_delta,
            "<=",
            float(performance_config["true_tvt_inside_known_range_rmse_delta_max"]),
            math.isfinite(inside_delta)
            and inside_delta
            <= float(performance_config["true_tvt_inside_known_range_rmse_delta_max"]),
        ),
        _gate(
            "distance_1000_plus_rmse_delta",
            distance_1000_delta,
            "<=",
            float(performance_config["distance_1000_plus_rmse_delta_max"]),
            math.isfinite(distance_1000_delta)
            and distance_1000_delta
            <= float(performance_config["distance_1000_plus_rmse_delta_max"]),
        ),
        _gate(
            "hidden_like_spatial_rmse_delta",
            hidden_spatial_delta,
            "<=",
            float(performance_config["hidden_like_spatial_rmse_delta_max"]),
            math.isfinite(hidden_spatial_delta)
            and hidden_spatial_delta
            <= float(performance_config["hidden_like_spatial_rmse_delta_max"]),
        ),
        _gate(
            "hidden_like_typewell_purged_rmse_delta",
            hidden_typewell_delta,
            "<=",
            float(performance_config["hidden_like_typewell_purged_rmse_delta_max"]),
            math.isfinite(hidden_typewell_delta)
            and hidden_typewell_delta
            <= float(performance_config["hidden_like_typewell_purged_rmse_delta_max"]),
        ),
        _gate(
            "by_well_rmse_p95_delta",
            p95_delta,
            "<=",
            float(performance_config["by_well_rmse_p95_delta_max"]),
            p95_delta <= float(performance_config["by_well_rmse_p95_delta_max"]),
        ),
        _gate(
            "worst_well_rmse_delta",
            worst_delta,
            "<=",
            float(performance_config["worst_well_rmse_delta_max"]),
            worst_delta <= float(performance_config["worst_well_rmse_delta_max"]),
        ),
        _gate("step_delta_p99_nonregression", step_delta_p99, "<=", 0.0, step_delta_p99 <= 0.0),
    ]
    technical_passed = all(bool(row["passed"]) for row in technical)
    performance_passed = all(bool(row["passed"]) for row in performance)
    return {
        "passed": bool(technical_passed and performance_passed),
        "technical_passed": technical_passed,
        "performance_passed": performance_passed,
        "technical": technical,
        "performance": performance,
        "fail_action": get_nested(config, "validation.fail_action"),
        "pass_action": get_nested(config, "validation.pass_action"),
    }


# %% [markdown]
# ## 9. Full Kaggle CPU orchestration and generated artifacts


# %%
def raw_input_manifest(data_dir: Path, wells: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in wells:
        for kind, suffix in (
            ("horizontal", "__horizontal_well.csv"),
            ("typewell", "__typewell.csv"),
        ):
            path = data_dir / f"{well}{suffix}"
            rows.append(
                {
                    "kind": kind,
                    "well": str(well),
                    "path": str(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_path(path),
                    "hashed_after_prediction_freeze": True,
                }
            )
    return pd.DataFrame(rows)


def generated_artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "prediction_freeze": output / f"{OUTPUT_PREFIX}_prediction_freeze.csv.gz",
        "support_manifest": output / f"{OUTPUT_PREFIX}_support_mask_manifest.csv",
        "prediction_schema": output / f"{OUTPUT_PREFIX}_prediction_schema.csv",
        "decoder_manifest": output / f"{OUTPUT_PREFIX}_decoder_manifest.json",
        "freeze_manifest": output / f"{OUTPUT_PREFIX}_freeze_manifest.json",
        "oof_readout": output / f"{OUTPUT_PREFIX}_oof_readout.csv.gz",
        "overall_metrics": output / f"{OUTPUT_PREFIX}_overall_metrics.csv",
        "fold_metrics": output / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "distance_metrics": output / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv",
        "known_range_metrics": output / f"{OUTPUT_PREFIX}_known_range_scope_metrics.csv",
        "hidden_like_metrics": output / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv",
        "by_well_metrics": output / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "step_delta_metrics": output / f"{OUTPUT_PREFIX}_step_delta_metrics.csv",
        "raw_input_manifest": output / f"{OUTPUT_PREFIX}_raw_input_manifest.csv",
        "external_input_manifest": output / f"{OUTPUT_PREFIX}_external_input_manifest.csv",
        "summary": output / f"{OUTPUT_PREFIX}_summary.json",
    }


def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp296 generation must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for a separately approved local smoke run."
        )
    validate_scientific_contract(config, require_run_approval=True)
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the exact exp223 HMM")
    started = time.time()
    data_dir = resolve_train_dir(config)
    wells = list_well_ids(data_dir)
    expected_wells = int(get_nested(config, "comparison.saved_control_wells"))
    if len(wells) != expected_wells:
        raise ValueError(f"raw input well count mismatch: {len(wells)} != {expected_wells}")
    expected_runs = int(get_nested(config, "model.planned_hmm_well_runs"))
    if len(wells) != expected_runs:
        raise ValueError(f"HMM run count mismatch: {len(wells)} != {expected_runs}")

    outer_workers = int(get_nested(config, "runtime.outer_workers", 2))
    numba_threads = int(get_nested(config, "runtime.numba_num_threads", 2))
    if outer_workers != 2 or numba_threads != 2:
        raise ValueError("exp296 fixes outer_workers=2 and numba_num_threads=2")
    set_num_threads(numba_threads)

    def build_one(index: int, well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(f"[exp296] {index}/{len(wells)} well={well}", flush=True)
        frame, meta = build_prediction_rows_for_well(well, data_dir, config)
        print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
        return frame, meta

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:
        raise RuntimeError("joblib is required for the fixed outer_workers=2 execution") from exc
    results = Parallel(n_jobs=outer_workers, prefer="threads")(
        delayed(build_one)(index, well) for index, well in enumerate(wells, start=1)
    )
    frames = [frame for frame, _ in results if not frame.empty]
    well_rows = [meta for _, meta in results]
    if not frames:
        raise ValueError("exp296 generated no prediction rows")
    frozen_predictions = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["well", "row_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    del frames, results
    gc.collect()
    well_manifest = (
        pd.DataFrame(well_rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    )
    expected_rows = int(get_nested(config, "comparison.saved_control_rows"))
    if (
        len(frozen_predictions) != expected_rows
        or frozen_predictions["well"].nunique() != expected_wells
    ):
        raise ValueError(
            f"full prediction coverage mismatch rows/wells={len(frozen_predictions)}/"
            f"{frozen_predictions['well'].nunique()} expected={expected_rows}/{expected_wells}"
        )
    if frozen_predictions["id"].duplicated().any():
        raise ValueError("prediction freeze contains duplicate ids")
    if not np.isfinite(frozen_predictions[PREDICTION_COLUMN].to_numpy(np.float64)).all():
        raise ValueError("prediction freeze contains non-finite predictions")
    outside_max = float(well_manifest["outside_support_contribution_max_abs"].max())
    inside_delta_max = float(well_manifest["inside_support_boost_delta_max_abs"].max())
    if outside_max != 0.0 or inside_delta_max != 0.0:
        raise AssertionError(
            f"support gate parity failed outside/inside={outside_max}/{inside_delta_max}"
        )

    output = artifact_dir()
    paths = generated_artifact_paths(output)
    schema = prediction_schema(frozen_predictions)
    source_path = package_dir() / f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"
    config_path = package_dir() / "config.yaml"
    decoder_manifest = {
        "experiment": EXPERIMENT_NAME,
        "parent": PARENT_EXPERIMENT,
        "variant": VARIANT,
        "hmm": get_nested(config, "model.hmm"),
        "self_gr_emission": get_nested(config, "model.self_gr_emission"),
        "support_gate": get_nested(config, "model.support_gate"),
        "active_variants": 1,
        "planned_hmm_well_runs": expected_runs,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "truth_attachment": "after_prediction_support_schema_and_decoder_freeze",
        "generation_horizontal_columns": list(GENERATION_HORIZONTAL_COLUMNS),
        "generation_typewell_columns": list(GENERATION_TYPEWELL_COLUMNS),
        "source_sha256": sha256_path(source_path) if source_path.exists() else None,
        "config_sha256": sha256_path(config_path) if config_path.exists() else None,
        "model_manifest": None,
        "submission": None,
    }

    # Freeze all candidate/support state before any unknown-suffix true TVT or
    # saved-control target is loaded.
    write_stable_gzip_csv(frozen_predictions, paths["prediction_freeze"])
    well_manifest.to_csv(paths["support_manifest"], index=False)
    schema.to_csv(paths["prediction_schema"], index=False)
    write_json(paths["decoder_manifest"], decoder_manifest)
    freeze_manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(frozen_predictions)),
        "wells": int(frozen_predictions["well"].nunique()),
        "unknown_suffix_truth_access_before_freeze": 0,
        "saved_control_target_access_before_freeze": 0,
        "prediction_raw_gzip_sha256": sha256_path(paths["prediction_freeze"]),
        "prediction_decompressed_sha256": sha256_gzip_decompressed(paths["prediction_freeze"]),
        "support_manifest_sha256": sha256_path(paths["support_manifest"]),
        "prediction_schema_sha256": sha256_path(paths["prediction_schema"]),
        "decoder_manifest_sha256": sha256_path(paths["decoder_manifest"]),
    }
    write_json(paths["freeze_manifest"], freeze_manifest)

    # Truth/control access starts only after the preceding files and SHA values
    # are materialized.
    truth = load_unknown_suffix_truth(data_dir, wells)
    control, control_manifest = load_saved_exp223_control(config)
    scored, join_audit = attach_truth_and_control_after_freeze(
        frozen_predictions,
        truth,
        control,
    )
    del truth, control
    gc.collect()
    overall = overall_metrics(scored)
    folds = grouped_metrics(scored, "reporting_fold", "reporting_fold")
    distance = grouped_metrics(scored, "distance_bucket", "distance_bucket")
    scopes = known_range_scope_metrics(scored)
    hidden, hidden_manifest = hidden_like_metrics(scored, config)
    by_well = by_well_metrics(scored)
    steps = step_delta_metrics(scored)
    hard_gates = evaluate_hard_gates(
        generated=scored,
        well_manifest=well_manifest,
        overall=overall,
        folds=folds,
        distance=distance,
        scopes=scopes,
        hidden=hidden,
        by_well=by_well,
        steps=steps,
        join_audit=join_audit,
        control_manifest=control_manifest,
        config=config,
    )

    write_stable_gzip_csv(scored, paths["oof_readout"])
    overall.to_csv(paths["overall_metrics"], index=False)
    folds.to_csv(paths["fold_metrics"], index=False)
    distance.to_csv(paths["distance_metrics"], index=False)
    scopes.to_csv(paths["known_range_metrics"], index=False)
    hidden.to_csv(paths["hidden_like_metrics"], index=False)
    by_well.to_csv(paths["by_well_metrics"], index=False)
    steps.to_csv(paths["step_delta_metrics"], index=False)
    raw_manifest = raw_input_manifest(data_dir, wells)
    raw_manifest.to_csv(paths["raw_input_manifest"], index=False)
    external_manifest = pd.DataFrame([control_manifest, hidden_manifest])
    external_manifest.to_csv(paths["external_input_manifest"], index=False)

    candidate_rmse = metric_value(overall, candidate=VARIANT)
    control_rmse = metric_value(overall, candidate=PARENT_CANDIDATE)
    status = (
        "completed_train_side_guard_passed"
        if hard_gates["passed"]
        else "completed_train_side_guard_failed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "parent": PARENT_EXPERIMENT,
        "variant": VARIANT,
        "rows": int(len(scored)),
        "wells": int(scored["well"].nunique()),
        "active_variants": 1,
        "hmm_well_runs": int(len(wells)),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "gpu": False,
        "inference": False,
        "submission": False,
        "candidate_rmse": candidate_rmse,
        "saved_exp223_control_rmse": control_rmse,
        "delta_rmse_vs_exp223": candidate_rmse - control_rmse,
        "hard_gates": hard_gates,
        "join_audit": join_audit,
        "freeze_manifest": freeze_manifest,
        "control_manifest": control_manifest,
        "decoder_manifest_sha256": sha256_mapping(decoder_manifest),
        "elapsed_seconds": float(time.time() - started),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "sha256": {
            key: (
                {
                    "raw_gzip": sha256_path(path),
                    "decompressed": sha256_gzip_decompressed(path),
                }
                if path.suffix == ".gz"
                else sha256_path(path)
            )
            for key, path in paths.items()
            if key != "summary" and path.exists()
        },
    }
    write_json(paths["summary"], summary)
    summary["sha256"]["summary"] = sha256_path(paths["summary"])
    write_json(paths["summary"], summary)
    metrics_path = output.parent / "metrics.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": status,
            "route": get_nested(config, "experiment.route"),
            "metric": "train_side_unknown_suffix_rmse",
            "cv": candidate_rmse,
            "public_lb": None,
            "private_lb": None,
            "rows": int(len(scored)),
            "wells": int(scored["well"].nunique()),
            "saved_exp223_control_rmse": control_rmse,
            "delta_rmse_vs_exp223": candidate_rmse - control_rmse,
            "hard_gates": hard_gates,
            "freeze_manifest": freeze_manifest,
            "summary": str(paths["summary"]),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 10. Setup, cost guard, and execution switch

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    PREFLIGHT = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(CONFIG, "experiment.route"),
        "parent": get_nested(CONFIG, "lineage.parent"),
        "variant": get_nested(CONFIG, "model.variant"),
        "implementation": get_nested(CONFIG, "execution.implementation"),
        "run_variant": get_nested(CONFIG, "execution.run_variant"),
        "kaggle_cpu_push_approved": get_nested(CONFIG, "execution.kaggle_cpu_push_approved"),
        "active_variants_when_run": 1,
        "planned_hmm_well_runs": get_nested(CONFIG, "model.planned_hmm_well_runs"),
        "lightgbm_configs": get_nested(CONFIG, "model.lightgbm_configs"),
        "trained_folds": get_nested(CONFIG, "model.trained_folds"),
        "boosters": get_nested(CONFIG, "model.boosters"),
        "parent_control_retraining": get_nested(CONFIG, "execution.run_control"),
        "gpu": get_nested(CONFIG, "runtime.gpu_enabled"),
        "inference": get_nested(CONFIG, "execution.run_inference"),
        "submission": get_nested(CONFIG, "execution.write_submission"),
        "numba_available": NUMBA_AVAILABLE,
        "outer_workers": get_nested(CONFIG, "runtime.outer_workers"),
        "numba_num_threads": get_nested(CONFIG, "runtime.numba_num_threads"),
    }
    print(json.dumps(to_jsonable(PREFLIGHT), indent=2, sort_keys=True))
    if bool(get_nested(CONFIG, "execution.run_variant")):
        SUMMARY = run_full_experiment(CONFIG)
    else:
        print(
            "exp296 implementation is ready, but execution.run_variant=false; "
            "no HMM run, truth readout, inference, or submission was executed."
        )
