# %% [markdown]
# # exp304 GR-denoiser emission separability readout
#
# This target-free, zero-booster diagnostic compares raw GR with three fixed
# denoisers before any HMM/PF/Beam decode. It freezes every denoised series and
# fixed-shift emission score before attaching true TVT for block-level MRR/top3.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Exp226 cache and raw-well input checks
# 4. Target-free GR denoisers and solver diagnostics
# 5. Fixed shift scoring and target-free freeze helpers
# 6. Late truth attachment and block separability readout
# 7. Fold/scope metrics, distortion summaries, and fixed promotion gate
# 8. Full Kaggle CPU orchestration and artifact guards
# 9. Setup, configuration, and target-free contract preview
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_NAME = "exp304_gr_denoiser_emission_separability_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP304_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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
        if (candidate / "project.yml").exists():
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
    raise FileNotFoundError(f"exp304 config not found in {[str(path) for path in candidates]}")


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
            KAGGLE_INPUT_ROOT
            / "competitions"
            / "rogii-wellbore-geology-prediction"
            / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def rank_descending(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking requires one finite score per shift")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int16)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int16)
    return ranks


def validate_scientific_contract(config: dict[str, Any]) -> None:
    expected_shifts = [
        -80.0,
        -40.0,
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        0.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ]
    shifts = [float(value) for value in get_nested(config, "audit.shift_bank_ft") or []]
    emission = get_nested(config, "audit.emission") or {}
    promotion = get_nested(config, "validation.promotion") or {}
    denoisers = get_nested(config, "audit.denoisers") or {}
    variants = [str(value) for value in get_nested(config, "model.active_audit_variants")]
    if shifts != expected_shifts:
        raise ValueError("exp304 fixes the approved 13-value shift bank")
    if int(get_nested(config, "audit.block_rows") or 0) != 512:
        raise ValueError("exp304 fixes non-overlapping 512-row blocks")
    if (
        get_nested(config, "audit.block_policy")
        != "non_overlapping_from_suffix_start_keep_short_tail"
    ):
        raise ValueError("exp304 fixes the non-overlapping short-tail block policy")
    if get_nested(config, "audit.score_aggregation") != "mean_row_log_likelihood":
        raise ValueError("exp304 fixes mean row log-likelihood aggregation")
    if get_nested(config, "audit.tie_policy") != "config_shift_bank_order":
        raise ValueError("exp304 fixes config-order tie resolution")
    if emission.get("kind") != "exp209_gaussian_gr":
        raise ValueError("exp304 fixes the exp209 Gaussian GR emission")
    if [float(value) for value in emission.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("exp304 fixes GR sigma clip [10, 60]")
    if float(emission.get("log_likelihood_clip", 0.0)) != 600.0:
        raise ValueError("exp304 fixes Gaussian log-likelihood clip 600")
    if bool(emission.get("recompute_sigma_from_denoised_gr", True)):
        raise ValueError("exp304 requires one shared raw-GR sigma for all variants")
    if variants != ["raw", "robust_rts", "swt_db4_l3", "l1_trend"]:
        raise ValueError("exp304 fixes raw plus exactly three denoiser variants")
    robust = denoisers.get("robust_rts", {})
    wavelet = denoisers.get("swt_db4_l3", {})
    trend = denoisers.get("l1_trend", {})
    if (
        float(robust.get("measurement_df", 0.0)) != 4.0
        or int(robust.get("maximum_iterations", 0)) != 8
        or float(robust.get("relative_mean_tolerance", 0.0)) != 1.0e-6
    ):
        raise ValueError("exp304 fixes Student-t RTS df/iterations/tolerance")
    if wavelet.get("wavelet") != "db4" or int(wavelet.get("level", 0)) != 3:
        raise ValueError("exp304 fixes stationary db4 level-3 wavelet denoising")
    if (
        wavelet.get("padding_side") != "right_preserve_original_row_alignment"
        or wavelet.get("fallback") != "forbidden_technical_fail"
    ):
        raise ValueError("exp304 fixes right reflection padding and no SWT fallback")
    if (
        float(trend.get("rho", 0.0)) != 1.0
        or int(trend.get("maximum_iterations", 0)) != 500
        or float(trend.get("absolute_tolerance", 0.0)) != 1.0e-4
        or float(trend.get("relative_tolerance", 0.0)) != 1.0e-4
    ):
        raise ValueError("exp304 fixes second-order L1 trend ADMM settings")
    if (
        float(promotion.get("minimum_pooled_mrr_absolute_gain_vs_raw", 0.0))
        != 0.01
        or float(promotion.get("minimum_pooled_top3_absolute_gain_vs_raw", 0.0))
        != 0.01
        or int(promotion.get("minimum_folds_mrr_improved_vs_raw", 0)) != 4
        or int(promotion.get("minimum_folds_top3_improved_vs_raw", 0)) != 4
        or int(promotion.get("minimum_folds_real_mrr_above_shuffled", 0)) != 5
        or int(promotion.get("minimum_folds_real_top3_above_shuffled", 0)) != 5
    ):
        raise ValueError("exp304 fixes the preregistered pooled/fold/shuffled gates")
    expected_zero = {
        "model.lightgbm_config_count": 0,
        "model.trained_fold_count": 0,
        "model.booster_count": 0,
        "model.hmm_decode_count": 0,
        "model.pf_run_count": 0,
        "model.beam_run_count": 0,
        "execution.total_boosters": 0,
        "execution.hmm_well_runs": 0,
        "execution.pf_well_runs": 0,
        "execution.beam_well_runs": 0,
    }
    for key, expected in expected_zero.items():
        if int(get_nested(config, key) or 0) != expected:
            raise ValueError(f"exp304 requires {key}={expected}")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("exp304 forbids inference and submission")
    fixed_values = {
        "experiment.route": "pf_beam",
        "model.active_variant_count": 4,
        "execution.active_audit_variants": 4,
        "validation.n_folds": 5,
        "runtime.num_workers": 1,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
    }
    for key, expected in fixed_values.items():
        if get_nested(config, key) != expected:
            raise ValueError(f"exp304 contract mismatch {key}={expected!r}")


# %% [markdown]
# ## 3. Exp226 cache and raw-well input checks


# %%
def load_exp226_safe(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_decompressed_sha = sha256_gzip_decompressed(path)
    expected_decompressed_sha = str(spec["expected_decompressed_sha256"])
    if actual_decompressed_sha != expected_decompressed_sha:
        raise ValueError(
            "exp226 decompressed SHA mismatch: "
            f"{actual_decompressed_sha} != {expected_decompressed_sha}"
        )
    safe_columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF has duplicate well_id/row_idx")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 tvt_geop must be finite")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if len(frame) != expected_rows or frame["well_id"].nunique() != expected_wells:
        raise ValueError("exp226 row/well coverage does not match the fixed contract")
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold set does not match the fixed contract")
    fold_counts = frame.groupby("well_id")["fold"].nunique()
    if not bool((fold_counts == 1).all()):
        raise ValueError("each exp226 well must belong to exactly one fold")
    manifest = {
        "name": "exp226_group_safe_oof_safe_columns",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": actual_decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "safe_columns": safe_columns,
    }
    return frame, path, manifest


def load_hidden_like_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"enabled": False}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignments missing {sorted(required - set(frame.columns))}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments require one row per well")
    manifest = {
        "name": "exp115_hidden_like_fold_assignments",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, manifest


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader must not expose TVT")
    if set(frame.columns) != {"MD", "GR", "TVT_input"}:
        raise ValueError("target-free horizontal reader exposed unexpected columns")
    return frame


# %% [markdown]
# ## 4. Target-free GR denoisers and solver diagnostics
#
# The three candidate denoisers operate separately within each horizontal well
# (MD order) and paired typewell (TVT order). They only see the common
# interpolated GR series. Any unavailable library, non-finite output, or
# non-converged solver is recorded as a technical failure for that method;
# parameters are never changed and no fallback smoother is substituted.


# %%
def robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0
    center = float(np.median(finite))
    return float(np.median(np.abs(finite - center)) / 0.67448975)


def normalized_coordinate(coordinate: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("denoiser coordinate must have at least two finite values")
    positive = np.diff(values)
    positive = positive[positive > 0.0]
    if not len(positive):
        raise ValueError("denoiser coordinate needs a positive spacing")
    spacing = float(np.median(positive))
    return (values - values[0]) / spacing, spacing


def second_divided_differences(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    u = np.asarray(coordinate, dtype=np.float64)
    left = np.diff(u[:-1])
    right = np.diff(u[1:])
    valid = (left > 0.0) & (right > 0.0) & ((u[2:] - u[:-2]) > 0.0)
    if not bool(valid.any()):
        return np.empty(0, dtype=np.float64)
    first_left = np.diff(y[:-1])[valid] / left[valid]
    first_right = np.diff(y[1:])[valid] / right[valid]
    span = (u[2:] - u[:-2])[valid]
    return 2.0 * (first_right - first_left) / span


def _student_t_rts_pass(
    observations: np.ndarray,
    coordinate: np.ndarray,
    *,
    measurement_variance: float,
    acceleration_variance: float,
    weights: np.ndarray,
    initial_slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observations, dtype=np.float64)
    u = np.asarray(coordinate, dtype=np.float64)
    n = len(y)
    filtered_mean = np.empty((n, 2), dtype=np.float64)
    predicted_mean = np.empty((n, 2), dtype=np.float64)
    filtered_cov = np.empty((n, 2, 2), dtype=np.float64)
    predicted_cov = np.empty((n, 2, 2), dtype=np.float64)
    state = np.array([y[0], initial_slope], dtype=np.float64)
    covariance = np.diag(
        [measurement_variance, measurement_variance + acceleration_variance]
    ).astype(np.float64)
    identity = np.eye(2, dtype=np.float64)

    for index in range(n):
        if index:
            dt = float(u[index] - u[index - 1])
            if dt < 0.0:
                raise ValueError("RTS coordinate must be non-decreasing")
            transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
            process = acceleration_variance * np.array(
                [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
                dtype=np.float64,
            )
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        predicted_mean[index] = state
        predicted_cov[index] = covariance
        effective_r = measurement_variance / float(weights[index])
        innovation_variance = float(covariance[0, 0] + effective_r)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise FloatingPointError("RTS innovation variance is not positive finite")
        gain = covariance[:, 0] / innovation_variance
        state = state + gain * (y[index] - state[0])
        covariance = (identity - np.outer(gain, np.array([1.0, 0.0]))) @ covariance
        covariance = 0.5 * (covariance + covariance.T)
        filtered_mean[index] = state
        filtered_cov[index] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_cov = filtered_cov.copy()
    for index in range(n - 2, -1, -1):
        dt = float(u[index + 1] - u[index])
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        smoother_gain = np.linalg.solve(
            predicted_cov[index + 1], transition @ filtered_cov[index]
        ).T
        smoothed_mean[index] = filtered_mean[index] + smoother_gain @ (
            smoothed_mean[index + 1] - predicted_mean[index + 1]
        )
        smoothed_cov[index] = filtered_cov[index] + smoother_gain @ (
            smoothed_cov[index + 1] - predicted_cov[index + 1]
        ) @ smoother_gain.T
        smoothed_cov[index] = 0.5 * (smoothed_cov[index] + smoothed_cov[index].T)
    return smoothed_mean, smoothed_cov


def robust_rts_smooth(
    values: np.ndarray,
    coordinate: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("robust RTS requires a finite one-dimensional series")
    u, coordinate_spacing = normalized_coordinate(coordinate)
    floor = float(spec["finite_scale_floor"])
    measurement_std = max(robust_scale(np.diff(y)) / math.sqrt(2.0), floor)
    acceleration_std = max(robust_scale(second_divided_differences(y, u)), floor)
    positive = np.diff(u) > 0.0
    slopes = np.diff(y)[positive] / np.diff(u)[positive]
    initial_slope = float(np.median(slopes)) if len(slopes) else 0.0
    measurement_variance = measurement_std**2
    acceleration_variance = acceleration_std**2
    degrees_of_freedom = float(spec["measurement_df"])
    maximum_iterations = int(spec["maximum_iterations"])
    tolerance = float(spec["relative_mean_tolerance"])
    weights = np.ones(len(y), dtype=np.float64)
    previous_level: np.ndarray | None = None
    relative_change = math.inf
    converged = False
    smoothed_mean = np.column_stack([y, np.full(len(y), initial_slope)])
    smoothed_cov = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], len(y), axis=0)

    for iteration in range(1, maximum_iterations + 1):
        smoothed_mean, smoothed_cov = _student_t_rts_pass(
            y,
            u,
            measurement_variance=measurement_variance,
            acceleration_variance=acceleration_variance,
            weights=weights,
            initial_slope=initial_slope,
        )
        level = smoothed_mean[:, 0]
        if previous_level is not None:
            relative_change = float(
                np.linalg.norm(level - previous_level)
                / max(np.linalg.norm(previous_level), 1.0e-12)
            )
            if relative_change <= tolerance:
                converged = True
                break
        normalized_residual_sq = ((y - level) / measurement_std) ** 2
        weights = (degrees_of_freedom + 1.0) / (
            degrees_of_freedom + normalized_residual_sq
        )
        if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
            raise FloatingPointError("RTS Student-t weights are not positive finite")
        previous_level = level.copy()

    level = smoothed_mean[:, 0]
    posterior_variance = np.maximum(smoothed_cov[:, 0, 0], 0.0)
    finite = bool(np.isfinite(level).all() and np.isfinite(posterior_variance).all())
    status = {
        "converged": converged,
        "iterations": iteration,
        "relative_mean_change": relative_change,
        "measurement_std": measurement_std,
        "measurement_variance": measurement_variance,
        "acceleration_std": acceleration_std,
        "acceleration_variance": acceleration_variance,
        "measurement_df": degrees_of_freedom,
        "coordinate_spacing": coordinate_spacing,
        "initial_slope": initial_slope,
        "minimum_weight": float(np.min(weights)),
        "finite_output": finite,
    }
    return level, posterior_variance, status


def swt_db4_l3_smooth(
    values: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("SWT requires a finite series with at least two rows")
    if str(spec["wavelet"]) != "db4" or int(spec["level"]) != 3:
        raise ValueError("exp304 only permits stationary db4 level-3 wavelets")
    try:
        import pywt
    except ImportError as error:
        raise RuntimeError(
            "PyWavelets is unavailable; swt_db4_l3 must technical-fail without fallback"
        ) from error
    level = int(spec["level"])
    multiple = 2**level
    pad_right = (-len(y)) % multiple
    padded = np.pad(y, (0, pad_right), mode="reflect") if pad_right else y.copy()
    coefficients = pywt.swt(
        padded,
        str(spec["wavelet"]),
        level=level,
        trim_approx=False,
        norm=False,
    )
    thresholds: list[float] = []
    thresholded: list[tuple[np.ndarray, np.ndarray]] = []
    universal_factor = math.sqrt(2.0 * math.log(len(y)))
    for approximation, detail in coefficients:
        sigma = robust_scale(np.asarray(detail, dtype=np.float64))
        threshold = float(sigma * universal_factor)
        thresholds.append(threshold)
        thresholded.append(
            (
                np.asarray(approximation, dtype=np.float64),
                np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0),
            )
        )
    reconstructed = np.asarray(
        pywt.iswt(thresholded, str(spec["wavelet"]), norm=False), dtype=np.float64
    )[: len(y)]
    status = {
        "converged": bool(np.isfinite(reconstructed).all()),
        "iterations": 1,
        "pad_right_rows": pad_right,
        "thresholds": thresholds,
        "pywavelets_version": str(pywt.__version__),
        "finite_output": bool(np.isfinite(reconstructed).all()),
    }
    return reconstructed, status


def second_difference(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x[:-2] - 2.0 * x[1:-1] + x[2:]


def second_difference_transpose(values: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.zeros(size, dtype=np.float64)
    output[:-2] += source
    output[1:-1] -= 2.0 * source
    output[2:] += source
    return output


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.maximum(np.abs(array) - float(threshold), 0.0)


def l1_trend_smooth(
    values: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("L1 trend filtering requires a finite series with at least two rows")
    noise_std = robust_scale(np.diff(y)) / math.sqrt(2.0)
    regularization = float(noise_std * math.sqrt(2.0 * math.log(len(y))))
    if len(y) < 3 or regularization == 0.0:
        return y.copy(), {
            "converged": True,
            "iterations": 0,
            "lambda": regularization,
            "rho": float(spec["rho"]),
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "finite_output": True,
        }
    try:
        from scipy.linalg import cholesky_banded, cho_solve_banded
    except ImportError as error:
        raise RuntimeError(
            "SciPy is unavailable; l1_trend must technical-fail without fallback"
        ) from error

    rho = float(spec["rho"])
    maximum_iterations = int(spec["maximum_iterations"])
    absolute_tolerance = float(spec["absolute_tolerance"])
    relative_tolerance = float(spec["relative_tolerance"])
    n = len(y)
    m = n - 2
    diagonal = np.ones(n, dtype=np.float64)
    diagonal[:-2] += rho
    diagonal[1:-1] += 4.0 * rho
    diagonal[2:] += rho
    first_upper = np.zeros(n - 1, dtype=np.float64)
    first_upper[:-1] -= 2.0 * rho
    first_upper[1:] -= 2.0 * rho
    second_upper = np.full(n - 2, rho, dtype=np.float64)
    banded = np.zeros((3, n), dtype=np.float64)
    banded[2] = diagonal
    banded[1, 1:] = first_upper
    banded[0, 2:] = second_upper
    factor = cholesky_banded(banded, lower=False, check_finite=False)

    x = y.copy()
    z = second_difference(x)
    dual = np.zeros(m, dtype=np.float64)
    converged = False
    primal_norm = math.inf
    dual_norm = math.inf
    for iteration in range(1, maximum_iterations + 1):
        right_hand_side = y + rho * second_difference_transpose(z - dual, n)
        x = cho_solve_banded((factor, False), right_hand_side, check_finite=False)
        d2x = second_difference(x)
        previous_z = z.copy()
        z = soft_threshold(d2x + dual, regularization / rho)
        dual = dual + d2x - z
        primal_norm = float(np.linalg.norm(d2x - z))
        dual_norm = float(
            np.linalg.norm(rho * second_difference_transpose(z - previous_z, n))
        )
        primal_tolerance = math.sqrt(m) * absolute_tolerance + relative_tolerance * max(
            np.linalg.norm(d2x), np.linalg.norm(z)
        )
        dual_tolerance = math.sqrt(n) * absolute_tolerance + relative_tolerance * float(
            np.linalg.norm(rho * second_difference_transpose(dual, n))
        )
        if primal_norm <= primal_tolerance and dual_norm <= dual_tolerance:
            converged = True
            break
    status = {
        "converged": converged,
        "iterations": iteration,
        "lambda": regularization,
        "rho": rho,
        "primal_residual": primal_norm,
        "dual_residual": dual_norm,
        "finite_output": bool(np.isfinite(x).all()),
    }
    return np.asarray(x, dtype=np.float64), status


def absolute_gradient(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinate, dtype=np.float64)
    if len(y) != len(x) or len(y) < 2:
        raise ValueError("gradient inputs must be aligned with at least two rows")
    spacing = np.diff(x)
    positive = spacing[spacing > 0.0]
    if not len(positive):
        raise ValueError("gradient coordinate requires positive spacing")
    safe_spacing = np.where(spacing > 0.0, spacing, float(np.median(positive)))
    interval = np.abs(np.diff(y) / safe_spacing)
    result = np.empty(len(y), dtype=np.float64)
    result[0] = interval[0]
    result[-1] = interval[-1]
    if len(y) > 2:
        result[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return result


def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free GR preparation forbids horizontal TVT")
    required_horizontal = {"MD", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal_without_truth.columns):
        missing = sorted(required_horizontal - set(horizontal_without_truth.columns))
        raise ValueError(
            f"horizontal missing {missing}"
        )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")

    horizontal = horizontal_without_truth.copy()
    horizontal["MD"] = pd.to_numeric(horizontal["MD"], errors="raise")
    horizontal_gr_observed = pd.to_numeric(horizontal["GR"], errors="coerce")
    horizontal_original_missing = horizontal_gr_observed.isna().to_numpy(bool)
    if not np.isfinite(horizontal["MD"].to_numpy(np.float64)).all():
        raise ValueError("horizontal MD must be finite")
    if bool((np.diff(horizontal["MD"].to_numpy(np.float64)) < 0.0).any()):
        raise ValueError("horizontal rows must already be in non-decreasing MD order")

    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    typewell_original_missing = tw["GR"].isna().to_numpy(bool)
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy()).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    if bool((np.diff(typewell_tvt) < 0.0).any()):
        raise ValueError("typewell TVT must be non-decreasing after stable sort")

    known = horizontal.loc[horizontal["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("well requires at least four known-prefix rows")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "audit.emission.sigma_clip")
    ]
    gr_sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(gr_sigma):
        raise ValueError("known-prefix raw-GR residual sigma is not finite")

    gr_fill = float(np.nanmean(typewell_gr))
    horizontal_gr = (
        horizontal_gr_observed.interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    if not np.isfinite(horizontal_gr).all():
        raise ValueError("common horizontal GR interpolation must be finite")
    known_positions = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
    return {
        "horizontal_coordinate": horizontal["MD"].to_numpy(np.float64),
        "horizontal_gr": horizontal_gr,
        "horizontal_original_missing": horizontal_original_missing,
        "typewell_coordinate": typewell_tvt,
        "typewell_gr": typewell_gr,
        "typewell_original_missing": typewell_original_missing,
        "gr_sigma": gr_sigma,
        "known_rows": len(known),
        "last_known_row_idx": int(known_positions[-1]),
        "known_residual_mean": float(np.mean(residual)),
        "known_residual_std_unclipped": float(np.std(residual)),
    }


def _denoise_one(
    variant: str,
    values: np.ndarray,
    coordinate: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if variant == "raw":
        output = np.asarray(values, dtype=np.float64).copy()
        return output, np.full(len(output), np.nan), {
            "converged": True,
            "iterations": 0,
            "finite_output": bool(np.isfinite(output).all()),
        }
    spec = get_nested(config, f"audit.denoisers.{variant}") or {}
    if variant == "robust_rts":
        return robust_rts_smooth(values, coordinate, spec)
    if variant == "swt_db4_l3":
        output, status = swt_db4_l3_smooth(values, spec)
        return output, np.full(len(output), np.nan), status
    if variant == "l1_trend":
        output, status = l1_trend_smooth(values, spec)
        return output, np.full(len(output), np.nan), status
    raise ValueError(f"unknown exp304 denoiser {variant}")


def denoise_prepared_inputs(
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    variants = [str(value) for value in get_nested(config, "model.active_audit_variants")]
    outputs: dict[str, dict[str, np.ndarray]] = {}
    status_rows: list[dict[str, Any]] = []
    for variant in variants:
        outputs[variant] = {}
        for series_kind in ("horizontal", "typewell"):
            values = np.asarray(prepared[f"{series_kind}_gr"], dtype=np.float64)
            coordinate = np.asarray(
                prepared[f"{series_kind}_coordinate"], dtype=np.float64
            )
            try:
                smoothed, variance, details = _denoise_one(
                    variant, values, coordinate, config
                )
                length_match = len(smoothed) == len(values)
                finite_output = bool(
                    length_match
                    and np.isfinite(smoothed).all()
                    and (
                        variant != "robust_rts"
                        or (
                            len(variance) == len(values)
                            and np.isfinite(variance).all()
                        )
                    )
                )
                passed = bool(
                    length_match
                    and finite_output
                    and bool(details.get("converged", False))
                )
                error = ""
            except Exception as exception:
                smoothed = np.full(len(values), np.nan)
                variance = np.full(len(values), np.nan)
                details = {
                    "converged": False,
                    "iterations": 0,
                    "finite_output": False,
                }
                length_match = True
                finite_output = False
                passed = False
                error = f"{type(exception).__name__}: {exception}"[:500]
            outputs[variant][series_kind] = np.asarray(smoothed, dtype=np.float64)
            outputs[variant][f"{series_kind}_variance"] = np.asarray(
                variance, dtype=np.float64
            )
            status_rows.append(
                {
                    "variant": variant,
                    "series_kind": series_kind,
                    "rows": len(values),
                    "finite_input": bool(np.isfinite(values).all()),
                    "length_match": length_match,
                    "finite_output": finite_output,
                    "converged": bool(details.get("converged", False)),
                    "technical_pass": passed,
                    "error": error,
                    **{
                        f"solver_{key}": value
                        for key, value in details.items()
                        if key not in {"finite_output", "converged"}
                    },
                }
            )
    return outputs, pd.DataFrame(status_rows)


def build_denoised_series_frame(
    well_id: str,
    prepared: dict[str, Any],
    outputs: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> pd.DataFrame:
    variants = [str(value) for value in get_nested(config, "model.active_audit_variants")]
    parts: list[pd.DataFrame] = []
    for series_kind in ("horizontal", "typewell"):
        coordinate = np.asarray(
            prepared[f"{series_kind}_coordinate"], dtype=np.float64
        )
        frame = pd.DataFrame(
            {
                "series_kind": series_kind,
                "well_id": str(well_id),
                "position": np.arange(len(coordinate), dtype=np.int64),
                "coordinate": coordinate,
                "original_missing": np.asarray(
                    prepared[f"{series_kind}_original_missing"], dtype=bool
                ),
            }
        )
        for variant in variants:
            frame[f"{variant}_gr"] = outputs[variant][series_kind]
        frame["robust_rts_posterior_variance"] = outputs["robust_rts"][
            f"{series_kind}_variance"
        ]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def distortion_metric_rows(
    well_id: str,
    prepared: dict[str, Any],
    outputs: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = [
        str(value)
        for value in get_nested(config, "model.active_audit_variants")
        if str(value) != "raw"
    ]
    for series_kind in ("horizontal", "typewell"):
        raw = np.asarray(prepared[f"{series_kind}_gr"], dtype=np.float64)
        coordinate = np.asarray(
            prepared[f"{series_kind}_coordinate"], dtype=np.float64
        )
        raw_gradient = absolute_gradient(raw, coordinate)
        edge_threshold = float(np.quantile(raw_gradient, 0.90))
        edge_mask = raw_gradient >= edge_threshold
        raw_energy = float(np.mean((raw - np.mean(raw)) ** 2))
        for variant in variants:
            smooth = np.asarray(outputs[variant][series_kind], dtype=np.float64)
            finite = bool(len(smooth) == len(raw) and np.isfinite(smooth).all())
            if finite:
                smooth_gradient = absolute_gradient(smooth, coordinate)
                correlation = (
                    float(np.corrcoef(raw, smooth)[0, 1])
                    if np.std(raw) > 0.0 and np.std(smooth) > 0.0
                    else float(np.allclose(raw, smooth))
                )
                edge_ratio = float(
                    np.mean(smooth_gradient[edge_mask])
                    / max(np.mean(raw_gradient[edge_mask]), 1.0e-12)
                )
                row = {
                    "raw_smoothed_mae": float(np.mean(np.abs(raw - smooth))),
                    "raw_smoothed_correlation": correlation,
                    "detail_energy_ratio": float(
                        np.mean((raw - smooth) ** 2) / max(raw_energy, 1.0e-12)
                    ),
                    "sharp_edge_attenuation": float(1.0 - edge_ratio),
                    "output_finite_coverage": 1.0,
                }
            else:
                row = {
                    "raw_smoothed_mae": np.nan,
                    "raw_smoothed_correlation": np.nan,
                    "detail_energy_ratio": np.nan,
                    "sharp_edge_attenuation": np.nan,
                    "output_finite_coverage": float(np.isfinite(smooth).mean()),
                }
            rows.append(
                {
                    "well_id": str(well_id),
                    "series_kind": series_kind,
                    "variant": variant,
                    "rows": len(raw),
                    **row,
                }
            )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 5. Fixed shift scoring and target-free freeze helpers


# %%
def score_variant_target_free(
    oof_safe: pd.DataFrame,
    prepared: dict[str, Any],
    outputs: dict[str, dict[str, np.ndarray]],
    variant: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = oof["row_idx"].to_numpy(np.int64)
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    shifts = np.asarray(get_nested(config, "audit.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "audit.block_rows"))
    geop = oof["tvt_geop"].to_numpy(np.float64)
    candidate_tvt = geop[:, None] + shifts[None, :]
    typewell_tvt = np.asarray(prepared["typewell_coordinate"], dtype=np.float64)
    typewell_signal = np.asarray(outputs[variant]["typewell"], dtype=np.float64)
    expected_gr = np.empty_like(candidate_tvt)
    for slot in range(len(shifts)):
        expected_gr[:, slot] = np.interp(
            candidate_tvt[:, slot], typewell_tvt, typewell_signal
        )
    horizontal_signal = np.asarray(outputs[variant]["horizontal"], dtype=np.float64)[
        row_idx
    ]
    clip_value = float(get_nested(config, "audit.emission.log_likelihood_clip"))
    zscore = (horizontal_signal[:, None] - expected_gr) / float(prepared["gr_sigma"])
    log_likelihood = -0.5 * np.minimum(zscore**2, clip_value)
    if not np.isfinite(log_likelihood).all():
        raise ValueError(f"{variant} target-free shift likelihood must be finite")

    md = np.asarray(prepared["horizontal_coordinate"], dtype=np.float64)
    last_known = int(prepared["last_known_row_idx"])
    md_since = md[row_idx] - md[last_known]
    original_missing = np.asarray(
        prepared["horizontal_original_missing"], dtype=bool
    )[row_idx]
    raw_typewell_gradient = absolute_gradient(
        np.asarray(prepared["typewell_gr"], dtype=np.float64), typewell_tvt
    )
    gradient_at_geop = np.interp(geop, typewell_tvt, raw_typewell_gradient)
    block_id = suffix_offset // block_rows
    native = (candidate_tvt >= typewell_tvt.min()) & (
        candidate_tvt <= typewell_tvt.max()
    )
    extension = float(get_nested(config, "audit.typewell_extension_ft"))
    extended = (candidate_tvt >= typewell_tvt.min() - extension) & (
        candidate_tvt <= typewell_tvt.max() + extension
    )

    well = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    shuffle_seed = int(get_nested(config, "audit.shuffled_control.seed"))
    rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        mask = block_id == block
        scores = log_likelihood[mask].mean(axis=0)
        score_sums = log_likelihood[mask].sum(axis=0)
        ranks = rank_descending(scores)
        rng = np.random.default_rng(
            stable_seed(EXPERIMENT_NAME, shuffle_seed, well, int(block))
        )
        shuffled_scores = scores[rng.permutation(len(scores))]
        shuffled_ranks = rank_descending(shuffled_scores)
        block_positions = np.flatnonzero(mask)
        for slot, shift in enumerate(shifts):
            rows.append(
                {
                    "variant": variant,
                    "well_id": well,
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(suffix_offset[block_positions[0]]),
                    "block_end_suffix_offset": int(suffix_offset[block_positions[-1]]),
                    "block_start_row_idx": int(row_idx[block_positions[0]]),
                    "block_end_row_idx": int(row_idx[block_positions[-1]]),
                    "block_row_count": int(mask.sum()),
                    "md_since_min_ft": float(np.min(md_since[mask])),
                    "md_since_max_ft": float(np.max(md_since[mask])),
                    "md_since_mid_ft": float(np.mean(md_since[mask])),
                    "original_gr_missing_share": float(original_missing[mask].mean()),
                    "typewell_abs_gradient_mean": float(
                        np.mean(gradient_at_geop[mask])
                    ),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "likelihood_mean": float(scores[slot]),
                    "likelihood_sum": float(score_sums[slot]),
                    "likelihood_rank": int(ranks[slot]),
                    "shuffled_likelihood_mean": float(shuffled_scores[slot]),
                    "shuffled_likelihood_rank": int(shuffled_ranks[slot]),
                    "native_typewell_coverage": float(native[mask, slot].mean()),
                    "extended_typewell_coverage": float(extended[mask, slot].mean()),
                }
            )
    score_frame = pd.DataFrame(rows).sort_values(
        ["variant", "well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    manifest = {
        "variant": variant,
        "well_id": well,
        "fold": fold,
        "evaluation_rows": len(oof),
        "blocks": int(block_id.max() + 1),
        "known_rows": int(prepared["known_rows"]),
        "last_known_row_idx": last_known,
        "gr_sigma": float(prepared["gr_sigma"]),
        "known_residual_mean": float(prepared["known_residual_mean"]),
        "known_residual_std_unclipped": float(
            prepared["known_residual_std_unclipped"]
        ),
        "original_eval_gr_missing_share": float(original_missing.mean()),
        "score_finite_coverage": float(np.isfinite(log_likelihood).mean()),
    }
    return score_frame.reset_index(drop=True), manifest


def score_well_target_free(
    oof_safe: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forbidden = set(
        str(value)
        for value in get_nested(config, "data.exp226_oof.forbidden_score_columns")
    )
    leaked = sorted(forbidden.intersection(oof_safe.columns))
    if leaked:
        raise ValueError(f"target-free score input contains forbidden exp226 columns: {leaked}")
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free score input contains horizontal TVT")
    required_oof = {"well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"}
    if not required_oof.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required_oof - set(oof_safe.columns))}")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("score_well_target_free requires one non-empty well and fold")
    row_idx = oof["row_idx"].to_numpy(np.int64)
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("exp226 suffix_offset must be contiguous from zero")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_without_truth):
        raise ValueError("exp226 row_idx is outside the raw horizontal frame")
    if horizontal_without_truth.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF rows must align only to unknown-suffix rows")

    prepared = prepare_gr_inputs(horizontal_without_truth, typewell, config)
    outputs, solver_status = denoise_prepared_inputs(prepared, config)
    well_id = str(oof["well_id"].iloc[0])
    solver_status.insert(0, "well_id", well_id)
    score_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for variant in [str(value) for value in get_nested(config, "model.active_audit_variants")]:
        variant_status = solver_status.loc[solver_status["variant"] == variant]
        if len(variant_status) == 2 and bool(variant_status["technical_pass"].all()):
            part, manifest = score_variant_target_free(
                oof, prepared, outputs, variant, config
            )
            score_parts.append(part)
            manifest_rows.append(manifest)
    if not score_parts or "raw" not in set(pd.concat(score_parts)["variant"]):
        raise RuntimeError("raw target-free scoring must always succeed")
    scores = pd.concat(score_parts, ignore_index=True)
    series = build_denoised_series_frame(well_id, prepared, outputs, config)
    distortion = distortion_metric_rows(well_id, prepared, outputs, config)
    manifests = pd.DataFrame(manifest_rows)
    return scores, series, solver_status, distortion, manifests


def attach_target_free_scope_flags(scores: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    raw_blocks = (
        scores.loc[scores["variant"] == "raw"]
        .drop_duplicates(["well_id", "block_id"])
        .sort_values(["well_id", "block_id"], kind="mergesort")
    )
    if raw_blocks.empty:
        raise ValueError("raw target-free blocks are required for scope flags")
    threshold = float(raw_blocks["typewell_abs_gradient_mean"].quantile(0.90))
    flags = raw_blocks[["well_id", "block_id", "typewell_abs_gradient_mean"]].copy()
    flags["sharp_edge_block"] = flags["typewell_abs_gradient_mean"] >= threshold
    output = scores.drop(columns=["sharp_edge_block"], errors="ignore").merge(
        flags[["well_id", "block_id", "sharp_edge_block"]],
        on=["well_id", "block_id"],
        how="left",
        validate="many_to_one",
    )
    if output["sharp_edge_block"].isna().any():
        raise ValueError("sharp-edge target-free flag failed block identity coverage")
    output["sharp_edge_block"] = output["sharp_edge_block"].astype(bool)
    return output.sort_values(
        ["variant", "well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True), threshold


class DeterministicGzipCsvWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="", mode="wb", fileobj=self._raw, compresslevel=6, mtime=0
        )
        self._decompressed_digest = hashlib.sha256()
        self._header_written = False
        self.rows = 0
        self.closed = False

    def append(self, frame: pd.DataFrame) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed gzip CSV writer")
        payload = frame.to_csv(index=False, header=not self._header_written).encode()
        self._gzip.write(payload)
        self._decompressed_digest.update(payload)
        self._header_written = True
        self.rows += len(frame)

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self._gzip.close()
            self._raw.close()
            self.closed = True
        return {
            "path": str(self.path),
            "rows": self.rows,
            "raw_sha256": sha256_path(self.path),
            "decompressed_sha256": self._decompressed_digest.hexdigest(),
            "content_sha256": self._decompressed_digest.hexdigest(),
        }


class DenoisedSeriesContentHasher:
    def __init__(self, variants: list[str]):
        self.variants = variants
        self.digests = {variant: hashlib.sha256() for variant in variants}
        self.header_written = {variant: False for variant in variants}

    def update(self, frame: pd.DataFrame) -> None:
        identity = [
            "series_kind",
            "well_id",
            "position",
            "coordinate",
            "original_missing",
        ]
        for variant in self.variants:
            value_columns = [f"{variant}_gr"]
            if variant == "robust_rts":
                value_columns.append("robust_rts_posterior_variance")
            selected = frame[identity + value_columns].copy()
            selected = selected.rename(columns={f"{variant}_gr": "gr"})
            payload = selected.to_csv(
                index=False, header=not self.header_written[variant]
            ).encode()
            self.digests[variant].update(payload)
            self.header_written[variant] = True

    def hexdigests(self) -> dict[str, str]:
        return {variant: digest.hexdigest() for variant, digest in self.digests.items()}


# %% [markdown]
# ## 6. Late truth attachment and block separability readout


# %%
def load_exp226_truth(
    path: Path,
    config: dict[str, Any],
    *,
    frozen_evidence: dict[str, str],
) -> pd.DataFrame:
    required = {
        "scientific_contract_sha256",
        "denoised_gr_content_sha256",
        "target_free_score_content_sha256",
    }
    if not required.issubset(frozen_evidence) or any(
        not frozen_evidence[key] for key in required
    ):
        raise ValueError("truth attachment requires complete frozen target-free evidence")
    spec = get_nested(config, "data.exp226_oof") or {}
    truth_columns = [str(value) for value in spec["truth_columns"]]
    frame = pd.read_csv(path, usecols=truth_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(
        frame["tvt_true"]
    ).all():
        raise ValueError("exp226 truth readout rows must be unique and finite")
    return frame


def sign_match(selected_shift: float, nearest_shift: float) -> bool:
    return bool(np.sign(float(selected_shift)) == np.sign(float(nearest_shift)))


def build_truth_readout(
    target_free_scores: pd.DataFrame,
    oof_safe: pd.DataFrame,
    truth: pd.DataFrame,
    config: dict[str, Any],
    valid_variants: list[str],
) -> pd.DataFrame:
    forbidden = {"tvt_true", "error", "abs_error", "formation"}
    leaked = forbidden.intersection(target_free_scores.columns)
    if leaked:
        raise ValueError(f"target-free score table contains forbidden truth columns {leaked}")
    if len(truth) != len(oof_safe):
        raise ValueError("truth and safe OOF row counts must match before attachment")
    merged = oof_safe.merge(truth, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    if len(merged) != len(oof_safe) or merged["tvt_true"].isna().any():
        raise ValueError("truth attachment failed row identity coverage")
    merged = merged.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    shifts = np.asarray(get_nested(config, "audit.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "audit.block_rows"))
    rows: list[dict[str, Any]] = []

    for well, well_frame in merged.groupby("well_id", sort=True):
        well_frame = well_frame.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        fold = int(well_frame["fold"].iloc[0])
        block_id = well_frame["suffix_offset"].to_numpy(np.int64) // block_rows
        for block in np.unique(block_id):
            mask = block_id == block
            block_frame = well_frame.loc[mask]
            true_tvt = block_frame["tvt_true"].to_numpy(np.float64)
            geop = block_frame["tvt_geop"].to_numpy(np.float64)
            errors = geop[:, None] + shifts[None, :] - true_tvt[:, None]
            candidate_rmse = np.sqrt(np.mean(errors**2, axis=0))
            nearest_slot = int(np.argmin(candidate_rmse))
            continuous_optimal_shift = float(np.mean(true_tvt - geop))
            nearest_shift = float(shifts[nearest_slot])
            for variant in valid_variants:
                block_scores = target_free_scores.loc[
                    (target_free_scores["variant"] == variant)
                    & (target_free_scores["well_id"].astype(str) == str(well))
                    & (target_free_scores["block_id"] == int(block))
                ].sort_values("shift_slot", kind="mergesort")
                if len(block_scores) != len(shifts) or not np.array_equal(
                    block_scores["shift_ft"].to_numpy(np.float64), shifts
                ):
                    raise ValueError(
                        f"target-free score bank misalignment for {variant}/{well}/{block}"
                    )
                real_rank = int(block_scores["likelihood_rank"].iloc[nearest_slot])
                shuffled_rank = int(
                    block_scores["shuffled_likelihood_rank"].iloc[nearest_slot]
                )
                top1_slot = int(
                    np.argmin(block_scores["likelihood_rank"].to_numpy(np.int64))
                )
                shuffled_top1_slot = int(
                    np.argmin(
                        block_scores["shuffled_likelihood_rank"].to_numpy(np.int64)
                    )
                )
                likelihood = block_scores["likelihood_mean"].to_numpy(np.float64)
                ordered_likelihood = np.sort(likelihood)[::-1]
                other = np.delete(likelihood, nearest_slot)
                top1_shift = float(shifts[top1_slot])
                shuffled_top1_shift = float(shifts[shuffled_top1_slot])
                score_meta = block_scores.iloc[0]
                rows.append(
                    {
                        "variant": variant,
                        "well_id": str(well),
                        "fold": fold,
                        "block_id": int(block),
                        "block_start_row_idx": int(block_frame["row_idx"].iloc[0]),
                        "block_end_row_idx": int(block_frame["row_idx"].iloc[-1]),
                        "block_row_count": len(block_frame),
                        "md_since_min_ft": float(score_meta["md_since_min_ft"]),
                        "md_since_max_ft": float(score_meta["md_since_max_ft"]),
                        "md_since_mid_ft": float(score_meta["md_since_mid_ft"]),
                        "original_gr_missing_share": float(
                            score_meta["original_gr_missing_share"]
                        ),
                        "typewell_abs_gradient_mean": float(
                            score_meta["typewell_abs_gradient_mean"]
                        ),
                        "sharp_edge_block": bool(score_meta["sharp_edge_block"]),
                        "continuous_optimal_shift_ft": continuous_optimal_shift,
                        "nearest_shift_ft": nearest_shift,
                        "nearest_shift_slot": nearest_slot,
                        "nearest_shift_rank": real_rank,
                        "nearest_shift_shuffled_rank": shuffled_rank,
                        "top1_hit": bool(real_rank == 1),
                        "top3_hit": bool(real_rank <= 3),
                        "mrr": float(1.0 / real_rank),
                        "shuffled_top1_hit": bool(shuffled_rank == 1),
                        "shuffled_top3_hit": bool(shuffled_rank <= 3),
                        "shuffled_mrr": float(1.0 / shuffled_rank),
                        "top1_shift_ft": top1_shift,
                        "shuffled_top1_shift_ft": shuffled_top1_shift,
                        "sign_match": sign_match(top1_shift, nearest_shift),
                        "shuffled_sign_match": sign_match(
                            shuffled_top1_shift, nearest_shift
                        ),
                        "likelihood_top1_margin": float(
                            ordered_likelihood[0] - ordered_likelihood[1]
                        ),
                        "truth_minus_best_decoy_gap": float(
                            likelihood[nearest_slot] - np.max(other)
                        ),
                        "bank_range_covered": bool(
                            shifts.min() <= continuous_optimal_shift <= shifts.max()
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["variant", "well_id", "block_id"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 7. Fold/scope metrics, distortion summaries, and fixed promotion gate


# %%
def readout_metric_row(
    frame: pd.DataFrame,
    *,
    variant: str,
    scope: str,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"{variant}/{scope} selected zero blocks")
    return {
        "variant": variant,
        "scope": scope,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "top1_rate": float(frame["top1_hit"].mean()),
        "top3_rate": float(frame["top3_hit"].mean()),
        "mrr": float(frame["mrr"].mean()),
        "mean_rank": float(frame["nearest_shift_rank"].mean()),
        "sign_accuracy": float(frame["sign_match"].mean()),
        "shuffled_top1_rate": float(frame["shuffled_top1_hit"].mean()),
        "shuffled_top3_rate": float(frame["shuffled_top3_hit"].mean()),
        "shuffled_mrr": float(frame["shuffled_mrr"].mean()),
        "shuffled_mean_rank": float(frame["nearest_shift_shuffled_rank"].mean()),
        "shuffled_sign_accuracy": float(frame["shuffled_sign_match"].mean()),
        "top1_lift_vs_shuffled": float(
            frame["top1_hit"].mean() - frame["shuffled_top1_hit"].mean()
        ),
        "top3_lift_vs_shuffled": float(
            frame["top3_hit"].mean() - frame["shuffled_top3_hit"].mean()
        ),
        "mrr_lift_vs_shuffled": float(
            frame["mrr"].mean() - frame["shuffled_mrr"].mean()
        ),
        "sign_lift_vs_shuffled": float(
            frame["sign_match"].mean() - frame["shuffled_sign_match"].mean()
        ),
        "truth_minus_best_decoy_gap_mean": float(
            frame["truth_minus_best_decoy_gap"].mean()
        ),
        "bank_range_coverage": float(frame["bank_range_covered"].mean()),
    }


def build_scope_and_fold_metrics(
    readout: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    role_columns = get_nested(config, "data.hidden_like.role_columns") or {}
    hidden_by_well = hidden_assignments.set_index("well_id")
    long_tail_min = float(get_nested(config, "audit.scopes.long_tail_min_md_since_ft"))

    for variant, variant_frame in readout.groupby("variant", sort=False):
        scopes: dict[str, pd.Series] = {
            "overall": pd.Series(True, index=variant_frame.index),
            "md_since_1000_plus": variant_frame["md_since_mid_ft"] >= long_tail_min,
            "original_gr_missing": variant_frame["original_gr_missing_share"] > 0.0,
            "original_gr_observed": variant_frame["original_gr_missing_share"] == 0.0,
            "typewell_gr_abs_gradient_top10pct": variant_frame[
                "sharp_edge_block"
            ].astype(bool),
        }
        for scope_name, role_column in role_columns.items():
            valid_wells = set(
                hidden_by_well.index[
                    hidden_by_well[str(role_column)].astype(str) == "valid"
                ].astype(str)
            )
            scopes[str(scope_name)] = variant_frame["well_id"].astype(str).isin(
                valid_wells
            )
        for scope_name, mask in scopes.items():
            scope_rows.append(
                readout_metric_row(
                    variant_frame.loc[mask],
                    variant=str(variant),
                    scope=scope_name,
                )
            )
        for fold, part in variant_frame.groupby("fold", sort=True):
            row = readout_metric_row(
                part, variant=str(variant), scope=f"fold_{int(fold)}"
            )
            row["fold"] = int(fold)
            fold_rows.append(row)
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def summarize_distortion(distortion: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "raw_smoothed_mae",
        "raw_smoothed_correlation",
        "detail_energy_ratio",
        "sharp_edge_attenuation",
        "output_finite_coverage",
    ]
    rows: list[dict[str, Any]] = []
    for (variant, series_kind), part in distortion.groupby(
        ["variant", "series_kind"], sort=True
    ):
        weights = part["rows"].to_numpy(np.float64)
        row: dict[str, Any] = {
            "variant": str(variant),
            "series_kind": str(series_kind),
            "groups": len(part),
            "rows": int(weights.sum()),
        }
        for column in numeric:
            values = part[column].to_numpy(np.float64)
            finite = np.isfinite(values)
            row[column] = (
                float(np.average(values[finite], weights=weights[finite]))
                if bool(finite.any())
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_technical_pass(
    scores: pd.DataFrame,
    solver_status: pd.DataFrame,
    config: dict[str, Any],
    *,
    frozen_evidence: dict[str, str],
) -> dict[str, Any]:
    variants = [str(value) for value in get_nested(config, "model.active_audit_variants")]
    shifts = [float(value) for value in get_nested(config, "audit.shift_bank_ft")]
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    raw_scores = scores.loc[scores["variant"] == "raw"]
    raw_blocks = raw_scores[["well_id", "fold", "block_id"]].drop_duplicates()
    raw_block_identity = set(
        raw_blocks.itertuples(index=False, name=None)
    )
    expected_blocks = len(raw_blocks)
    per_variant: dict[str, Any] = {}
    for variant in variants:
        status = solver_status.loc[solver_status["variant"] == variant]
        variant_scores = scores.loc[scores["variant"] == variant]
        score_blocks = variant_scores[
            ["well_id", "fold", "block_id"]
        ].drop_duplicates()
        score_block_identity = set(
            score_blocks.itertuples(index=False, name=None)
        )
        candidate_groups = variant_scores.groupby(
            ["well_id", "fold", "block_id"], sort=False
        )
        candidate_identity = bool(
            len(variant_scores)
            and all(
                len(part) == len(shifts)
                and np.array_equal(
                    part.sort_values("shift_slot", kind="mergesort")[
                        "shift_ft"
                    ].to_numpy(np.float64),
                    np.asarray(shifts, dtype=np.float64),
                )
                for _, part in candidate_groups
            )
        )
        expected_status_identity = {
            (str(well), series_kind)
            for well in scores.loc[scores["variant"] == "raw", "well_id"].unique()
            for series_kind in ("horizontal", "typewell")
        }
        actual_status_identity = set(
            status[["well_id", "series_kind"]].itertuples(index=False, name=None)
        )
        score_finite = bool(
            len(variant_scores)
            and np.isfinite(
                variant_scores[
                    ["likelihood_mean", "shuffled_likelihood_mean"]
                ].to_numpy(np.float64)
            ).all()
        )
        checks = {
            "solver_status_rows": len(status) == expected_wells * 2,
            "solver_status_identity": actual_status_identity
            == expected_status_identity,
            "all_series_technical_pass": bool(
                len(status) and status["technical_pass"].all()
            ),
            "score_block_identity": score_block_identity == raw_block_identity,
            "score_candidate_identity": candidate_identity
            and len(variant_scores) == expected_blocks * len(shifts),
            "score_finite": score_finite,
        }
        per_variant[variant] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "solver_failures": int(
                len(status) - int(status["technical_pass"].sum())
            ),
            "score_blocks": len(score_blocks),
            "score_rows": len(variant_scores),
        }
    common_checks = {
        "raw_row_block_fold_identity": bool(per_variant["raw"]["passed"]),
        "frozen_scientific_contract_sha": bool(
            frozen_evidence.get("scientific_contract_sha256")
        ),
        "frozen_denoised_gr_content_sha": bool(
            frozen_evidence.get("denoised_gr_content_sha256")
        ),
        "frozen_target_free_score_content_sha": bool(
            frozen_evidence.get("target_free_score_content_sha256")
        ),
    }
    valid_denoisers = [
        variant
        for variant in variants
        if variant != "raw" and per_variant[variant]["passed"]
    ]
    return {
        "passed": bool(all(common_checks.values()) and len(valid_denoisers) > 0),
        "common_passed": bool(all(common_checks.values())),
        "common_checks": common_checks,
        "per_variant": per_variant,
        "valid_denoisers": valid_denoisers,
        "expected_wells": expected_wells,
        "expected_blocks": expected_blocks,
    }


def _metric_record(
    metrics: pd.DataFrame,
    variant: str,
    scope: str,
) -> pd.Series | None:
    part = metrics.loc[
        (metrics["variant"] == variant) & (metrics["scope"] == scope)
    ]
    return None if part.empty else part.iloc[0]


def evaluate_quality_gate(
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    technical: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    promotion = get_nested(config, "validation.promotion") or {}
    raw_overall = _metric_record(scope_metrics, "raw", "overall")
    if raw_overall is None:
        raise ValueError("raw overall scope metrics are required")
    per_variant: dict[str, Any] = {}
    for variant in technical["valid_denoisers"]:
        overall = _metric_record(scope_metrics, variant, "overall")
        if overall is None:
            raise ValueError(f"{variant} overall metrics are missing")
        variant_folds = fold_metrics.loc[fold_metrics["variant"] == variant].sort_values(
            "fold"
        )
        raw_folds = fold_metrics.loc[fold_metrics["variant"] == "raw"].sort_values(
            "fold"
        )
        paired_folds = variant_folds.merge(
            raw_folds,
            on="fold",
            suffixes=("", "_raw"),
            validate="one_to_one",
        )
        scope_checks: dict[str, bool] = {}
        for scope in promotion["require_scope_mrr_top3_non_degradation"]:
            candidate = _metric_record(scope_metrics, variant, str(scope))
            control = _metric_record(scope_metrics, "raw", str(scope))
            scope_checks[str(scope)] = bool(
                candidate is not None
                and control is not None
                and float(candidate["mrr"]) >= float(control["mrr"])
                and float(candidate["top3_rate"]) >= float(control["top3_rate"])
            )
        mrr_gain = float(overall["mrr"] - raw_overall["mrr"])
        top3_gain = float(overall["top3_rate"] - raw_overall["top3_rate"])
        top1_gain = float(overall["top1_rate"] - raw_overall["top1_rate"])
        gap_gain = float(
            overall["truth_minus_best_decoy_gap_mean"]
            - raw_overall["truth_minus_best_decoy_gap_mean"]
        )
        folds_mrr_improved = int(
            (paired_folds["mrr"] > paired_folds["mrr_raw"]).sum()
        )
        folds_top3_improved = int(
            (paired_folds["top3_rate"] > paired_folds["top3_rate_raw"]).sum()
        )
        folds_real_mrr_above_shuffled = int(
            (variant_folds["mrr"] > variant_folds["shuffled_mrr"]).sum()
        )
        folds_real_top3_above_shuffled = int(
            (
                variant_folds["top3_rate"]
                > variant_folds["shuffled_top3_rate"]
            ).sum()
        )
        checks = {
            "pooled_mrr_gain": mrr_gain
            >= float(promotion["minimum_pooled_mrr_absolute_gain_vs_raw"]),
            "pooled_top3_gain": top3_gain
            >= float(promotion["minimum_pooled_top3_absolute_gain_vs_raw"]),
            "pooled_top1_non_degradation": (
                top1_gain >= 0.0
                if bool(promotion["require_pooled_top1_non_degradation"])
                else True
            ),
            "fold_mrr_improvement": folds_mrr_improved
            >= int(promotion["minimum_folds_mrr_improved_vs_raw"]),
            "fold_top3_improvement": folds_top3_improved
            >= int(promotion["minimum_folds_top3_improved_vs_raw"]),
            "required_scopes_non_degraded": bool(all(scope_checks.values())),
            "real_mrr_above_shuffled": folds_real_mrr_above_shuffled
            >= int(promotion["minimum_folds_real_mrr_above_shuffled"]),
            "real_top3_above_shuffled": folds_real_top3_above_shuffled
            >= int(promotion["minimum_folds_real_top3_above_shuffled"]),
            "truth_minus_best_decoy_gap_improved": (
                gap_gain > 0.0
                if bool(promotion["require_truth_minus_best_decoy_gap_improvement"])
                else True
            ),
        }
        per_variant[variant] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "scope_checks": scope_checks,
            "pooled_mrr_gain_vs_raw": mrr_gain,
            "pooled_top3_gain_vs_raw": top3_gain,
            "pooled_top1_gain_vs_raw": top1_gain,
            "truth_minus_best_decoy_gap_gain_vs_raw": gap_gain,
            "folds_mrr_improved_vs_raw": folds_mrr_improved,
            "folds_top3_improved_vs_raw": folds_top3_improved,
            "folds_real_mrr_above_shuffled": folds_real_mrr_above_shuffled,
            "folds_real_top3_above_shuffled": folds_real_top3_above_shuffled,
        }

    passed = [
        variant for variant, result in per_variant.items() if bool(result["passed"])
    ]
    selected: str | None = None
    if passed:
        maximum_gain = max(
            float(per_variant[variant]["pooled_mrr_gain_vs_raw"])
            for variant in passed
        )
        tolerance = float(promotion["selection_tie_tolerance"])
        tied = {
            variant
            for variant in passed
            if maximum_gain
            - float(per_variant[variant]["pooled_mrr_gain_vs_raw"])
            <= tolerance
        }
        tie_order = [str(value) for value in promotion["selection_tie_order"]]
        selected = next(variant for variant in tie_order if variant in tied)
    return {
        "passed": selected is not None,
        "per_variant": per_variant,
        "passing_variants": passed,
        "selected_denoiser": selected,
        "failure_action": promotion["failure_action"],
    }


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and artifact guards


# %%
def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "lineage": get_nested(config, "lineage"),
        "safe_columns": get_nested(config, "data.exp226_oof.safe_columns"),
        "forbidden_score_columns": get_nested(
            config, "data.exp226_oof.forbidden_score_columns"
        ),
        "shift_bank_ft": get_nested(config, "audit.shift_bank_ft"),
        "block_rows": get_nested(config, "audit.block_rows"),
        "block_policy": get_nested(config, "audit.block_policy"),
        "score_aggregation": get_nested(config, "audit.score_aggregation"),
        "tie_policy": get_nested(config, "audit.tie_policy"),
        "common_input": get_nested(config, "audit.common_input"),
        "emission": get_nested(config, "audit.emission"),
        "denoisers": get_nested(config, "audit.denoisers"),
        "shuffled_control": get_nested(config, "audit.shuffled_control"),
        "scope_contract": get_nested(config, "audit.scopes"),
        "promotion": get_nested(config, "validation.promotion"),
        "implementation_details": {
            "wavelet_reflection_padding_side": "right",
            "sharp_edge_scope": (
                "global_top10_percent_of_raw_typewell_abs_gradient_block_mean_at_tvt_geop"
            ),
            "denoiser_failure_policy": "method_technical_fail_without_fallback",
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "blas_threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = None
    try:
        import pywt

        versions["pywavelets"] = pywt.__version__
    except ImportError:
        versions["pywavelets"] = None
    return versions


def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp304 readout must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp304 Kaggle CPU execution is not approved")
    validate_scientific_contract(config)
    started = time.time()
    artifacts = artifact_dir()
    safe_oof, exp226_path, exp226_manifest = load_exp226_safe(config)
    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    raw_dir = train_data_dir(config)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    if len(raw_wells) != expected_wells or set(raw_wells) != set(
        safe_oof["well_id"].unique()
    ):
        raise ValueError("raw train and exp226 well sets do not match")

    scientific_contract = build_scientific_contract(config)
    scientific_contract_path = (
        artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    )
    write_json(scientific_contract_path, scientific_contract)
    variants = [str(value) for value in get_nested(config, "model.active_audit_variants")]
    series_writer = DeterministicGzipCsvWriter(
        artifacts / f"{OUTPUT_PREFIX}_denoised_gr_series.csv.gz"
    )
    series_hasher = DenoisedSeriesContentHasher(variants)
    score_parts: list[pd.DataFrame] = []
    solver_parts: list[pd.DataFrame] = []
    distortion_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    raw_file_rows: list[dict[str, Any]] = []
    progress_every = 25

    for index, well in enumerate(raw_wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        horizontal_safe = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        scores, series, statuses, distortion, manifests = score_well_target_free(
            safe_oof.loc[safe_oof["well_id"] == well],
            horizontal_safe,
            typewell,
            config,
        )
        score_parts.append(scores)
        solver_parts.append(statuses)
        distortion_parts.append(distortion)
        manifest_parts.append(manifests)
        series_writer.append(series)
        series_hasher.update(series)
        raw_file_rows.append(
            {
                "well_id": well,
                "horizontal_path": str(horizontal_path),
                "horizontal_rows": len(horizontal_safe),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_rows": len(typewell),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        if index % progress_every == 0 or index == len(raw_wells):
            print(f"denoise/target-free scoring wells={index}/{len(raw_wells)}")

    series_artifact = series_writer.close()
    scores = pd.concat(score_parts, ignore_index=True)
    scores, sharp_edge_threshold = attach_target_free_scope_flags(scores)
    solver_status = pd.concat(solver_parts, ignore_index=True).sort_values(
        ["variant", "well_id", "series_kind"], kind="mergesort"
    )
    distortion_by_group = pd.concat(distortion_parts, ignore_index=True)
    distortion = summarize_distortion(distortion_by_group)
    score_manifest = pd.concat(manifest_parts, ignore_index=True)
    raw_file_manifest = pd.DataFrame(raw_file_rows).sort_values(
        "well_id", kind="mergesort"
    )

    target_free_score_content_sha = dataframe_content_sha(scores)
    if not target_free_score_content_sha or not series_artifact["decompressed_sha256"]:
        raise RuntimeError("failed to freeze target-free denoised GR or score SHA")
    score_artifact = write_csv_gzip(
        scores,
        artifacts / f"{OUTPUT_PREFIX}_target_free_shift_scores.csv.gz",
    )
    frozen_evidence = {
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "denoised_gr_content_sha256": series_artifact["decompressed_sha256"],
        "target_free_score_content_sha256": target_free_score_content_sha,
    }

    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "exp226_oof": exp226_manifest,
        "hidden_like": hidden_manifest,
        "raw_train": {
            "path": str(raw_dir),
            "wells": len(raw_file_manifest),
            "horizontal_rows": int(raw_file_manifest["horizontal_rows"].sum()),
            "typewell_rows": int(raw_file_manifest["typewell_rows"].sum()),
            "well_file_identity_content_sha256": dataframe_content_sha(
                raw_file_manifest,
                [
                    "well_id",
                    "horizontal_raw_sha256",
                    "typewell_raw_sha256",
                ],
            ),
            "well_files": raw_file_manifest.to_dict(orient="records"),
        },
    }
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.json"
    write_json(input_manifest_path, input_manifest)
    denoised_manifest = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "series_artifact": series_artifact,
        "variant_content_sha256": series_hasher.hexdigests(),
        "sharp_edge_threshold": sharp_edge_threshold,
        "solver_status_rows": len(solver_status),
        "score_manifest_rows": len(score_manifest),
        "silent_fallback_count": 0,
    }
    denoised_manifest_path = artifacts / f"{OUTPUT_PREFIX}_denoised_gr_manifest.json"
    write_json(denoised_manifest_path, denoised_manifest)

    technical = evaluate_technical_pass(
        scores,
        solver_status,
        config,
        frozen_evidence=frozen_evidence,
    )
    if not technical["common_passed"]:
        raise RuntimeError("exp304 common raw identity/freeze technical contract failed")
    valid_variants = ["raw", *technical["valid_denoisers"]]

    # Horizontal true TVT is first read here, after denoised series and all scores
    # are frozen and content-hashed.
    truth = load_exp226_truth(
        exp226_path,
        config,
        frozen_evidence=frozen_evidence,
    )
    readout = build_truth_readout(
        scores,
        safe_oof,
        truth,
        config,
        valid_variants,
    )
    scope_metrics, fold_metrics = build_scope_and_fold_metrics(
        readout, hidden_assignments, config
    )
    quality = evaluate_quality_gate(scope_metrics, fold_metrics, technical, config)

    readout_artifact = write_csv_gzip(
        readout,
        artifacts / f"{OUTPUT_PREFIX}_block_readout.csv.gz",
    )
    csv_outputs = {
        "fold_metrics": fold_metrics,
        "scope_metrics": scope_metrics,
        "distortion_metrics": distortion,
        "solver_status": solver_status,
    }
    output_paths: dict[str, Path] = {}
    for name, frame in csv_outputs.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = path

    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"].copy()
    selected = quality["selected_denoiser"]
    if selected is not None:
        status = "train_side_readout_completed_quality_passed"
        decision = "permit_one_separate_tempered_exact_hmm_experiment"
    elif technical["valid_denoisers"]:
        status = "train_side_readout_completed_quality_failed"
        decision = "close_reserved_followups_2_3_4_without_rescue_grid"
    else:
        status = "train_side_readout_completed_no_valid_denoiser"
        decision = "close_reserved_followups_2_3_4_without_rescue_grid"
    output_sha = {
        name: sha256_path(path)
        for name, path in {
            **output_paths,
            "scientific_contract": scientific_contract_path,
            "denoised_manifest": denoised_manifest_path,
            "input_manifest": input_manifest_path,
        }.items()
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "rows": len(safe_oof),
        "wells": int(safe_oof["well_id"].nunique()),
        "blocks": int(
            scores.loc[scores["variant"] == "raw", ["well_id", "block_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "shift_candidates": len(get_nested(config, "audit.shift_bank_ft")),
        "active_audit_variants": len(variants),
        "valid_denoisers": technical["valid_denoisers"],
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "overall": overall.to_dict(orient="records"),
        "technical_gate": technical,
        "quality_gate": quality,
        "selected_denoiser": selected,
        "truth_attachment": {
            "stage": (
                "after_scientific_contract_denoised_gr_and_all_target_free_scores_frozen"
            ),
            **frozen_evidence,
        },
        "sharp_edge_scope": {
            "definition": (
                "global_top10_percent_of_raw_typewell_abs_gradient_block_mean_at_tvt_geop"
            ),
            "threshold": sharp_edge_threshold,
        },
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "artifacts": {
            "scientific_contract": str(scientific_contract_path),
            "denoised_gr_manifest": str(denoised_manifest_path),
            "denoised_gr_series": series_artifact,
            "target_free_scores": score_artifact,
            "block_readout": readout_artifact,
            "file_sha256": output_sha,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "model_sha256": None,
        "prediction_sha256": None,
        "submission_sha256": None,
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "selected_denoiser": selected,
        "promotion_decision": "passed" if selected is not None else "failed",
        "diagnostic": {
            "overall": overall.to_dict(orient="records"),
            "technical_gate": technical,
            "quality_gate": quality,
            **frozen_evidence,
        },
        "notes": "No prediction, model, HMM, PF, Beam, inference, or submission is produced.",
    }
    write_json(metrics_output_path(), metrics)
    print(overall.to_string(index=False))
    print(json.dumps(to_jsonable(quality), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup, configuration, and target-free contract preview


# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "methodology_parent": get_nested(
                    CONFIG, "lineage.methodology_parent"
                ),
                "emission_reference": get_nested(
                    CONFIG, "lineage.emission_reference"
                ),
                "stage": get_nested(CONFIG, "execution.stage"),
                "variants": get_nested(CONFIG, "model.active_audit_variants"),
                "shift_bank_ft": get_nested(CONFIG, "audit.shift_bank_ft"),
                "block_rows": get_nested(CONFIG, "audit.block_rows"),
                "saved_fold_strata": get_nested(CONFIG, "validation.n_folds"),
                "lightgbm_configs": get_nested(
                    CONFIG, "execution.lightgbm_config_count"
                ),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "hmm_well_runs": get_nested(CONFIG, "execution.hmm_well_runs"),
                "pf_well_runs": get_nested(CONFIG, "execution.pf_well_runs"),
                "beam_well_runs": get_nested(CONFIG, "execution.beam_well_runs"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
                "kaggle_push_approved": get_nested(
                    CONFIG, "execution.kaggle_push_approved"
                ),
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 10. Run the diagnostic and report generated artifacts


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP304_SUMMARY = run_full_experiment(CONFIG)
