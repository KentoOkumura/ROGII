from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


EXPERIMENT_NAME = "exp225_state_known_tvt_self_gr_hmm_emission"
OUTPUT_PREFIX = "exp225_state_known_tvt_self_gr_hmm_emission"
VARIANT = "state_known_tvt_self_gr_hmm_emission"
META_COLUMNS = ["id", "well", "target"]
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
BASE_FEATURE_COLUMNS = [
    "last_known_tvt",
    "md_since",
]
BASE_HMM_FEATURE_COLUMNS = [
    "hmm_mean_tvt",
    "hmm_mean_d",
    "hmm_std",
    "hmm_loglik",
    "hmm_grid_step",
    "hmm_grid_size",
    "hmm_prefix_sigma",
    "hmm_prefix_ir",
    "hmm_cal_a",
    "hmm_cal_b",
    "hmm_finite",
]
FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *BASE_HMM_FEATURE_COLUMNS]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_well_ids(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    wells: list[str] = []
    for path in sorted(data_dir.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        if (data_dir / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def resolve_existing_file(root: str | Path, candidates: list[str]) -> Path:
    root = Path(root)
    checked: list[str] = []
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
            if not basename:
                continue
            matches = [path for path in sorted(KAGGLE_INPUT_ROOT.rglob(basename)) if path.stat().st_size > 0]
            checked.extend(str(path) for path in matches)
            if matches:
                return matches[0]
    raise FileNotFoundError("No non-empty candidate path exists: " + json.dumps(checked, indent=2))


def sanitize_token(value: Any) -> str:
    text = str(value).lower().replace(".", "p").replace("-", "m")
    keep = []
    for char in text:
        keep.append(char if char.isalnum() else "_")
    compact = "_".join(part for part in "".join(keep).split("_") if part)
    return compact or "x"


def self_gr_variant_name(method: str, alpha: float, clip_value: float, mode: str) -> str:
    alpha_token = f"a{int(round(float(alpha) * 1000)):03d}"
    clip_token = f"c{int(round(float(clip_value) * 100)):03d}"
    return f"hmm_selfgr_{sanitize_token(method)}_{sanitize_token(mode)}_{alpha_token}_{clip_token}"


def self_gr_variant_feature_columns(variants: list[dict[str, Any]]) -> list[str]:
    columns = list(BASE_FEATURE_COLUMNS)
    if variants:
        columns.extend(
            [
                "self_gr_quality",
                "self_gr_peak_tvt",
                "self_gr_peak_gap",
                "self_gr_typewell_agreement",
                "self_gr_valid",
                "self_gr_state_valid_rate",
            ]
        )
    for variant in variants:
        name = str(variant["name"])
        columns.extend(
            [
                f"{name}_mean_tvt",
                f"{name}_mean_d",
                f"{name}_std",
                f"{name}_loglik",
                f"{name}_finite",
            ]
        )
    return columns


def prepare_self_gr_emission_variants(config: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return [], []
    alpha_values = [float(v) for v in (config.get("alpha_grid") or [])]
    clip_values = [float(v) for v in (config.get("clip_grid") or [])]
    modes = [str(v) for v in (config.get("modes") or [])]
    if not alpha_values or not clip_values or not modes:
        raise ValueError("self_gr_emission.enabled requires alpha_grid, clip_grid, and modes")
    method = str(config.get("method") or "state_known_tvt_curve")
    if method not in {"state_known_tvt_curve", "descriptor_motif"}:
        raise ValueError(f"unsupported self-GR emission method: {method}")

    max_variants = config.get("max_variants")
    max_variants = int(max_variants) if max_variants is not None else None
    runtime_variants: list[dict[str, Any]] = []
    summary_variants: list[dict[str, Any]] = []
    for alpha in alpha_values:
        for clip_value in clip_values:
            for mode in modes:
                if mode not in {"boost_only", "symmetric"}:
                    raise ValueError(f"unsupported self-GR emission mode: {mode}")
                name = self_gr_variant_name(method, alpha, clip_value, mode)
                variant = {
                    "name": name,
                    "method": method,
                    "alpha": float(alpha),
                    "clip": float(clip_value),
                    "mode": mode,
                }
                runtime_variants.append(variant)
                summary_variants.append(
                    {
                        "name": name,
                        "method": method,
                        "alpha": float(alpha),
                        "clip": float(clip_value),
                        "mode": mode,
                    }
                )
                if max_variants is not None and len(runtime_variants) >= max_variants:
                    return runtime_variants, summary_variants
    return runtime_variants, summary_variants


def load_well(well: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    horizontal = pd.read_csv(data_dir / f"{well}__horizontal_well.csv")
    typewell = (
        pd.read_csv(data_dir / f"{well}__typewell.csv")
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    return horizontal, typewell


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    tail_n: int = 30,
) -> tuple[float, float, float, float]:
    """Amerhu affine GR calibration, residual sigma, and tail U-rate."""
    known = horizontal[horizontal["TVT_input"].notna()]
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
    fill_value = float(np.nanmedian(series.to_numpy(dtype=np.float64))) if series.notna().any() else 0.0
    return (
        series.interpolate(limit_direction="both")
        .fillna(fill_value)
        .to_numpy(dtype=np.float64)
    )


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
    roll_std = series.rolling(window=window, center=True, min_periods=max(3, radius // 2)).std(ddof=0)
    mean = roll_mean.interpolate(limit_direction="both").fillna(float(np.mean(gr))).to_numpy(np.float64)
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
        shifted = pd.Series(gr).shift(-int(offset)).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill")
        descriptors.append(((shifted.to_numpy(np.float64) - mean) / std).astype(np.float64))
    global_std = float(np.std(gr) if np.std(gr) > 1e-6 else 1.0)
    descriptors.append(((mean - float(np.mean(gr))) / global_std).astype(np.float64))
    descriptors.append(np.log1p(std).astype(np.float64))
    if radius > 0:
        left = pd.Series(gr).shift(radius).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill").to_numpy(np.float64)
        right = pd.Series(gr).shift(-radius).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill").to_numpy(np.float64)
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
    stride = max(1, int(stride))
    selected = usable[::stride]
    if keep_last > 0:
        selected = np.unique(np.concatenate([selected, usable[-int(keep_last) :]])).astype(np.int64)
    if max_anchors > 0 and len(selected) > max_anchors:
        take = np.linspace(0, len(selected) - 1, int(max_anchors)).round().astype(np.int64)
        selected = selected[take]
    return selected.astype(np.int64)


def build_self_gr_likelihood_surface(
    horizontal: pd.DataFrame,
    eval_index: np.ndarray,
    grid: np.ndarray,
    typewell_peak_tvt: np.ndarray,
    config: dict[str, Any] | None,
) -> dict[str, np.ndarray | float | int]:
    config = config or {}
    radius = int(config.get("window_radius_rows", 12))
    offsets = [int(v) for v in (config.get("descriptor_offsets") or [-12, -8, -4, 0, 4, 8, 12])]
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

    n_eval = len(eval_index)
    n_grid = len(grid)
    zero_surface = np.zeros((n_eval, n_grid), dtype=np.float32)
    zero_vector = np.zeros(n_eval, dtype=np.float32)
    if n_eval == 0 or n_grid == 0:
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "prefix_anchor_count": 0,
        }

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
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "prefix_anchor_count": int(len(anchor_indices)),
        }

    descriptors, missing_rate = build_gr_window_descriptors(horizontal, radius=radius, offsets=offsets)
    anchor_mask = missing_rate[anchor_indices] <= max_missing_rate
    anchor_indices = anchor_indices[anchor_mask]
    if len(anchor_indices) < min_anchors:
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "prefix_anchor_count": int(len(anchor_indices)),
        }

    anchor_desc = descriptors[anchor_indices].astype(np.float32)
    anchor_tvt = tvt_input[anchor_indices].astype(np.float64)
    eval_desc = descriptors[eval_index].astype(np.float32)
    eval_missing = missing_rate[eval_index].astype(np.float32)
    prefix_coverage_quality = float(np.clip(len(anchor_indices) / max(float(min_anchors), 1.0), 0.0, 1.0))

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
        z = (grid[None, None, :] - centers[:, :, None]) / sigma_tvt
        component_ll = np.log(np.clip(weights, eps, None))[:, :, None] - 0.5 * np.minimum(z * z, surface_clip)
        best = np.max(component_ll, axis=1)
        ll = best + np.log(np.clip(np.exp(component_ll - best[:, None, :]).sum(axis=1), eps, None))
        ll_centered = ll - np.mean(ll, axis=1, keepdims=True)
        ll_scale = np.std(ll_centered, axis=1, keepdims=True)
        ll_centered = ll_centered / np.clip(ll_scale, 0.25, None)
        centered[start:end] = ll_centered.astype(np.float32)

        cost_q75 = np.quantile(cost, 0.75, axis=1)
        sharpness = np.clip((cost_q75 - top_cost[:, 0]) / np.clip(cost_q75, eps, None), 0.0, 1.0)
        if k_eff >= 2:
            gap = top_cost[:, 1] - top_cost[:, 0]
        else:
            gap = np.zeros(end - start, dtype=np.float32)
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
        quality[start:end] = np.where(row_valid, np.clip(row_quality, 0.0, 1.0), 0.0).astype(np.float32)
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
        "state_valid_rate": np.zeros(n_eval, dtype=np.float32),
        "prefix_anchor_count": int(len(anchor_indices)),
    }


def robust_mad_sigma(values: np.ndarray, *, default: float, floor: float, cap: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) >= 3:
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = float(np.nanstd(finite))
        if np.isfinite(sigma) and sigma > 1e-6:
            return float(np.clip(sigma, floor, cap))
    return float(np.clip(default, floor, cap))


def build_state_known_tvt_self_gr_likelihood_surface(
    horizontal: pd.DataFrame,
    eval_index: np.ndarray,
    eval_gr: np.ndarray,
    grid: np.ndarray,
    typewell_peak_tvt: np.ndarray,
    config: dict[str, Any] | None,
) -> dict[str, np.ndarray | float | int]:
    config = config or {}
    min_points = max(2, int(config.get("min_curve_points", 12)))
    min_tvt_span = max(1e-6, float(config.get("min_tvt_span", 20.0)))
    min_state_count = max(1, int(config.get("min_state_count", 5)))
    sigma_floor = max(1e-6, float(config.get("gr_sigma_floor", 10.0)))
    sigma_cap = max(sigma_floor, float(config.get("gr_sigma_cap", 60.0)))
    sigma_default = float(config.get("gr_sigma_default", 25.0))
    smooth_window = max(1, int(config.get("curve_smooth_window", 9)))
    agreement_sigma = max(1e-6, float(config.get("typewell_agreement_sigma_tvt", 18.0)))
    surface_clip = float(config.get("surface_quadratic_clip", 60.0))
    gap_temperature = max(1e-6, float(config.get("peak_gap_temperature", 0.25)))

    n_eval = len(eval_index)
    n_grid = len(grid)
    zero_surface = np.zeros((n_eval, n_grid), dtype=np.float32)
    zero_vector = np.zeros(n_eval, dtype=np.float32)
    if n_eval == 0 or n_grid == 0:
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "state_valid_rate": zero_vector,
            "prefix_anchor_count": 0,
            "prefix_tvt_min": None,
            "prefix_tvt_max": None,
            "prefix_tvt_span": 0.0,
            "self_gr_sigma": sigma_default,
        }

    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    known_mask = np.isfinite(tvt_input) & np.isfinite(gr_raw)
    if int(known_mask.sum()) < min_points:
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "state_valid_rate": zero_vector,
            "prefix_anchor_count": int(known_mask.sum()),
            "prefix_tvt_min": None,
            "prefix_tvt_max": None,
            "prefix_tvt_span": 0.0,
            "self_gr_sigma": sigma_default,
        }

    curve_frame = (
        pd.DataFrame({"tvt": tvt_input[known_mask], "gr": gr_raw[known_mask]})
        .sort_values("tvt", kind="mergesort")
        .groupby("tvt", as_index=False)["gr"]
        .mean()
    )
    if len(curve_frame) < min_points:
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "state_valid_rate": zero_vector,
            "prefix_anchor_count": int(len(curve_frame)),
            "prefix_tvt_min": None,
            "prefix_tvt_max": None,
            "prefix_tvt_span": 0.0,
            "self_gr_sigma": sigma_default,
        }

    curve_tvt = curve_frame["tvt"].to_numpy(np.float64)
    curve_gr_raw = curve_frame["gr"].to_numpy(np.float64)
    smooth_window = min(smooth_window, len(curve_gr_raw))
    if smooth_window > 2 and smooth_window % 2 == 0:
        smooth_window -= 1
    if smooth_window > 1:
        curve_gr = (
            pd.Series(curve_gr_raw)
            .rolling(window=smooth_window, center=True, min_periods=max(2, smooth_window // 3))
            .median()
            .interpolate(limit_direction="both")
            .fillna(pd.Series(curve_gr_raw).median())
            .to_numpy(np.float64)
        )
    else:
        curve_gr = curve_gr_raw

    prefix_tvt_min = float(curve_tvt.min())
    prefix_tvt_max = float(curve_tvt.max())
    prefix_tvt_span = prefix_tvt_max - prefix_tvt_min
    state_mask = (grid >= prefix_tvt_min) & (grid <= prefix_tvt_max)
    state_count = int(state_mask.sum())
    state_valid_rate = float(state_count / max(n_grid, 1))
    if prefix_tvt_span < min_tvt_span or state_count < min_state_count:
        return {
            "centered_logl": zero_surface,
            "quality": zero_vector,
            "peak_tvt": zero_vector.astype(np.float64),
            "peak_gap": zero_vector,
            "typewell_agreement": zero_vector,
            "valid": zero_vector,
            "state_valid_rate": np.full(n_eval, state_valid_rate, dtype=np.float32),
            "prefix_anchor_count": int(len(curve_tvt)),
            "prefix_tvt_min": prefix_tvt_min,
            "prefix_tvt_max": prefix_tvt_max,
            "prefix_tvt_span": float(prefix_tvt_span),
            "self_gr_sigma": sigma_default,
        }

    residual_for_sigma = curve_gr_raw - curve_gr
    sigma_gr = float(
        config.get("gr_sigma")
        or robust_mad_sigma(
            residual_for_sigma,
            default=sigma_default,
            floor=sigma_floor,
            cap=sigma_cap,
        )
    )
    sigma_gr = float(np.clip(sigma_gr, sigma_floor, sigma_cap))

    valid_grid = grid[state_mask]
    expected_gr = np.interp(valid_grid, curve_tvt, curve_gr)
    eval_gr = np.asarray(eval_gr, dtype=np.float64)
    finite_eval = np.isfinite(eval_gr)
    z = (eval_gr[:, None] - expected_gr[None, :]) / sigma_gr
    ll_valid = -0.5 * np.minimum(z * z, surface_clip)
    row_mean = np.nanmean(ll_valid, axis=1, keepdims=True)
    centered_valid = ll_valid - row_mean
    row_scale = np.nanstd(centered_valid, axis=1, keepdims=True)
    centered_valid = centered_valid / np.clip(row_scale, 0.25, None)
    centered_valid[~np.isfinite(centered_valid)] = 0.0
    centered = np.zeros((n_eval, n_grid), dtype=np.float32)
    centered[:, state_mask] = centered_valid.astype(np.float32)

    best_local = np.argmax(ll_valid, axis=1)
    peak_tvt = valid_grid[best_local].astype(np.float64)
    if state_count >= 2:
        top2 = np.partition(ll_valid, -2, axis=1)[:, -2:]
        top2 = np.sort(top2, axis=1)
        peak_gap = (top2[:, 1] - top2[:, 0]).astype(np.float32)
    else:
        peak_gap = np.zeros(n_eval, dtype=np.float32)
    agreement = np.exp(-0.5 * ((peak_tvt - typewell_peak_tvt) / agreement_sigma) ** 2).astype(np.float32)
    coverage_quality = float(np.clip(len(curve_tvt) / max(float(min_points), 1.0), 0.0, 1.0))
    span_quality = float(np.clip(prefix_tvt_span / min_tvt_span, 0.0, 1.0))
    state_quality = float(np.clip(state_valid_rate / max(float(config.get("min_state_valid_rate", 0.02)), 1e-6), 0.0, 1.0))
    gap_quality = np.clip(peak_gap / gap_temperature, 0.0, 1.0)
    row_valid = finite_eval & np.isfinite(peak_tvt)
    quality = (
        coverage_quality
        * span_quality
        * state_quality
        * (0.25 + 0.75 * gap_quality)
        * (0.15 + 0.85 * agreement)
    )
    quality = np.where(row_valid, np.clip(quality, 0.0, 1.0), 0.0).astype(np.float32)

    return {
        "centered_logl": centered,
        "quality": quality,
        "peak_tvt": peak_tvt,
        "peak_gap": peak_gap.astype(np.float32),
        "typewell_agreement": agreement.astype(np.float32),
        "valid": row_valid.astype(np.float32),
        "state_valid_rate": np.full(n_eval, state_valid_rate, dtype=np.float32),
        "prefix_anchor_count": int(len(curve_tvt)),
        "prefix_tvt_min": prefix_tvt_min,
        "prefix_tvt_max": prefix_tvt_max,
        "prefix_tvt_span": float(prefix_tvt_span),
        "self_gr_sigma": sigma_gr,
    }


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb(
    em,
    dm,
    dz,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    """Amerhu exact forward-backward over joint state (TVT position, dip-rate)."""
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    alpha = np.full((t_count, p_count, r_count), neg, np.float32)

    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)

    tmp = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = 0.5 * (rate_var_cells + mean_rate_move)
            p_minus = 0.5 * (rate_var_cells - mean_rate_move)
            if p_plus < 1e-12:
                p_plus = 1e-12
            if p_minus < 1e-12:
                p_minus = 1e-12
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)

        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = r2 - 1 if r2 - 1 >= 0 else 0
                k1 = r2 + 1 if r2 + 1 <= r_count - 1 else r_count - 1
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best)
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = sig_p if sig_p > 0.35 * sp else 0.35 * sp
        for r2 in range(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = position_log_kernel[0]
            for k_i in range(1, 5):
                if position_log_kernel[k_i] > kernel_max:
                    kernel_max = position_log_kernel[k_i]
            kernel_sum = 0.0
            for k_i in range(5):
                kernel_sum += np.exp(position_log_kernel[k_i] - kernel_max)
            log_norm = kernel_max + np.log(kernel_sum)
            for k_i in range(5):
                position_log_kernel[k_i] -= log_norm
            for p2 in prange(p_count):
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if p1 < 0 or p1 >= p_count:
                        continue
                    value = tmp[p1, r2] + position_log_kernel[k_i]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if p1 < 0 or p1 >= p_count:
                            continue
                        total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    cur[p2, r2] = np.float32(best + np.log(total) + lam * em[t_i, p2])
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]

    best = np.float32(neg)
    for p_i in range(p_count):
        for r_i in range(r_count):
            if alpha[t_count - 1, p_i, r_i] > best:
                best = alpha[t_count - 1, p_i, r_i]
    total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            total += np.exp(alpha[t_count - 1, p_i, r_i] - best)
    loglik = float(best) + np.log(total)

    post_p = np.zeros((t_count, p_count))
    beta_next = np.zeros((p_count, r_count), np.float32)

    best = neg
    for p_i in range(p_count):
        for r_i in range(r_count):
            value = alpha[t_count - 1, p_i, r_i] + beta_next[p_i, r_i]
            if value > best:
                best = value
    total = 0.0
    for p_i in range(p_count):
        acc = 0.0
        for r_i in range(r_count):
            acc += np.exp(alpha[t_count - 1, p_i, r_i] + beta_next[p_i, r_i] - best)
        post_p[t_count - 1, p_i] = acc
        total += acc
    for p_i in range(p_count):
        post_p[t_count - 1, p_i] /= total

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = 0.5 * (rate_var_cells + mean_rate_move)
            p_minus = 0.5 * (rate_var_cells - mean_rate_move)
            if p_plus < 1e-12:
                p_plus = 1e-12
            if p_minus < 1e-12:
                p_minus = 1e-12
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        sigma_position = sig_p if sig_p > 0.35 * sp else 0.35 * sp
        for r2 in range(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = position_log_kernel[0]
            for k_i in range(1, 5):
                if position_log_kernel[k_i] > kernel_max:
                    kernel_max = position_log_kernel[k_i]
            kernel_sum = 0.0
            for k_i in range(5):
                kernel_sum += np.exp(position_log_kernel[k_i] - kernel_max)
            log_norm = kernel_max + np.log(kernel_sum)
            for k_i in range(5):
                position_log_kernel[k_i] -= log_norm
            for p1 in prange(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if p2 < 0 or p2 >= p_count:
                        continue
                    value = position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if p2 < 0 or p2 >= p_count:
                            continue
                        total += np.exp(position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2] - best)
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg

        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = r_i - 1 if r_i - 1 >= 0 else 0
                k1 = r_i + 1 if r_i + 1 <= r_count - 1 else r_count - 1
                for r2 in range(k0, k1 + 1):
                    value = rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best)
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg

        best = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                value = alpha[t_i - 1, p_i, r_i] + beta_cur[p_i, r_i]
                if value > best:
                    best = value
        total = 0.0
        for p_i in range(p_count):
            acc = 0.0
            for r_i in range(r_count):
                acc += np.exp(alpha[t_i - 1, p_i, r_i] + beta_cur[p_i, r_i] - best)
            post_p[t_i - 1, p_i] = acc
            total += acc
        for p_i in range(p_count):
            post_p[t_i - 1, p_i] /= total
        for p_i in range(p_count):
            for r_i in range(r_count):
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    return post_p, loglik


def run_hmm2(
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
    return_post: bool = False,
    self_gr_config: dict[str, Any] | None = None,
    self_gr_alpha: float = 0.0,
    self_gr_clip: float = 1.0,
    self_gr_mode: str = "boost_only",
) -> dict[str, Any]:
    """Second-order HMM smoother, kept close to amerhu's public notebook."""
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal[horizontal["TVT_input"].notna()]
    eval_rows = horizontal[horizontal["TVT_input"].isna()]
    out = horizontal["TVT_input"].to_numpy(np.float64).copy()
    if len(eval_rows) == 0:
        return {
            "pred": out,
            "std_eval": np.array([], dtype=np.float64),
            "loglik": 0.0,
            "ev_index": np.array([], dtype=np.int64),
            "grid": np.array([], dtype=np.float64),
            "mean_eval": np.array([], dtype=np.float64),
            "prefix_sigma": None,
            "prefix_ir": None,
            "cal_a": None,
            "cal_b": None,
        }

    cal_a, cal_b, robust_sigma, init_rate = prefix_stats(horizontal, typewell_tvt, typewell_gr)
    if sigma_mode == "std":
        typewell_at_known = np.interp(known["TVT_input"].to_numpy(np.float64), typewell_tvt, typewell_gr)
        gr_residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
        gr_sigma = float(np.clip(np.nanstd(gr_residual), 10.0, 60.0))
        cal_a_use, cal_b_use = 1.0, 0.0
    else:
        gr_sigma = robust_sigma
        cal_a_use, cal_b_use = cal_a, cal_b

    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - band_pad)
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + band_pad)
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = cal_a_use * np.interp(grid, typewell_tvt, typewell_gr) + cal_b_use

    md = eval_rows["MD"].to_numpy(np.float64)
    z = eval_rows["Z"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[eval_rows.index]
    )
    dm = np.maximum(
        np.diff(np.concatenate([[float(last["MD"])], md])),
        1.0,
    )
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))

    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    if emission == "t":
        emission_ll = (-0.5 * (df + 1.0) * np.log1p(zscore**2 / df)).astype(np.float32)
    else:
        emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)

    self_surface: dict[str, np.ndarray | float | int] | None = None
    if self_gr_config is not None and float(self_gr_alpha) != 0.0:
        typewell_peak_tvt = grid[np.argmax(emission_ll, axis=1)]
        self_method = str(self_gr_config.get("method") or "state_known_tvt_curve")
        if self_method == "state_known_tvt_curve":
            self_surface = build_state_known_tvt_self_gr_likelihood_surface(
                horizontal,
                eval_rows.index.to_numpy(np.int64),
                gr,
                grid,
                typewell_peak_tvt,
                self_gr_config,
            )
        elif self_method == "descriptor_motif":
            self_surface = build_self_gr_likelihood_surface(
                horizontal,
                eval_rows.index.to_numpy(np.int64),
                grid,
                typewell_peak_tvt,
                self_gr_config,
            )
        else:
            raise ValueError(f"unsupported self-GR emission method: {self_method}")
        centered_self_ll = np.asarray(self_surface["centered_logl"], dtype=np.float32)
        quality_self = np.asarray(self_surface["quality"], dtype=np.float32)
        if centered_self_ll.shape != emission_ll.shape:
            raise ValueError(f"self-GR surface shape mismatch: expected {emission_ll.shape} got {centered_self_ll.shape}")
        clip_value = float(self_gr_clip)
        if self_gr_mode == "boost_only":
            self_boost = np.clip(centered_self_ll, 0.0, clip_value)
        elif self_gr_mode == "symmetric":
            self_boost = np.clip(centered_self_ll, -clip_value, clip_value)
        else:
            raise ValueError(f"unsupported self-GR mode: {self_gr_mode}")
        emission_ll = (
            emission_ll
            + np.float32(float(self_gr_alpha)) * quality_self[:, None].astype(np.float32) * self_boost
        ).astype(np.float32)

    if rate_center == "zero":
        span = max(rate_span, abs(init_rate) + 0.04)
        rates = np.linspace(-span, span, n_rates, dtype=np.float64)
    else:
        rates = init_rate + np.linspace(-rate_span, rate_span, n_rates, dtype=np.float64)
    start_p = float((last_tvt - grid_min) / step)

    post_p, loglik = _hmm2_fb(
        emission_ll,
        dm.astype(np.float64),
        dz.astype(np.float64),
        float(step),
        rates,
        float(sig_r),
        float(sig_p),
        start_p,
        float(start_sig),
        float(init_rate),
        float(r0_sig),
        float(lam),
        float(mom),
    )
    mean = post_p @ grid
    var = post_p @ (grid**2) - mean**2
    std = np.sqrt(np.maximum(var, 0.0))
    out[eval_rows.index] = mean
    result: dict[str, Any] = {
        "pred": out,
        "std_eval": std,
        "loglik": float(loglik),
        "ev_index": eval_rows.index.to_numpy(np.int64),
        "grid": grid,
        "mean_eval": mean,
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "cal_a": cal_a,
        "cal_b": cal_b,
        "self_gr_alpha": float(self_gr_alpha),
        "self_gr_clip": float(self_gr_clip),
        "self_gr_mode": self_gr_mode,
    }
    if self_surface is not None:
        result["self_gr_quality"] = np.asarray(self_surface["quality"], dtype=np.float32)
        result["self_gr_peak_tvt"] = np.asarray(self_surface["peak_tvt"], dtype=np.float64)
        result["self_gr_peak_gap"] = np.asarray(self_surface["peak_gap"], dtype=np.float32)
        result["self_gr_typewell_agreement"] = np.asarray(self_surface["typewell_agreement"], dtype=np.float32)
        result["self_gr_valid"] = np.asarray(self_surface["valid"], dtype=np.float32)
        result["self_gr_state_valid_rate"] = np.asarray(self_surface["state_valid_rate"], dtype=np.float32)
        result["self_gr_prefix_anchor_count"] = int(self_surface["prefix_anchor_count"])
        result["self_gr_prefix_tvt_min"] = self_surface.get("prefix_tvt_min")
        result["self_gr_prefix_tvt_max"] = self_surface.get("prefix_tvt_max")
        result["self_gr_prefix_tvt_span"] = self_surface.get("prefix_tvt_span")
        result["self_gr_sigma"] = self_surface.get("self_gr_sigma")
    if return_post:
        result["post"] = post_p
        result["md_eval"] = md
    return result


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if column in {"id", "well"}:
            frame[column] = frame[column].astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame


def build_hmm_rows_for_well(
    well: str,
    data_dir: str | Path,
    hmm_config: dict[str, Any],
    self_gr_variants: list[dict[str, Any]] | None = None,
    self_gr_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    self_gr_variants = self_gr_variants or []
    feature_columns = self_gr_variant_feature_columns(self_gr_variants) if self_gr_variants else FEATURE_COLUMNS
    horizontal, typewell = load_well(well, data_dir)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if int(eval_mask.sum()) == 0:
        return pd.DataFrame(columns=[*META_COLUMNS, *feature_columns]), {
            "well": well,
            "status": "skipped_no_eval_rows",
            "rows": 0,
        }
    if horizontal["TVT"].isna().all():
        return pd.DataFrame(columns=[*META_COLUMNS, *feature_columns]), {
            "well": well,
            "status": "skipped_no_train_tvt",
            "rows": 0,
        }

    started = time.time()
    eval_index = horizontal.index[eval_mask].to_numpy(np.int64)
    eval_ids = [f"{well}_{int(i)}" for i in eval_index]
    known = horizontal.loc[known_mask]
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_md = float(last["MD"])
    true_tvt = horizontal.loc[eval_index, "TVT"].to_numpy(np.float64)
    md_since = horizontal.loc[eval_index, "MD"].to_numpy(np.float64) - last_md
    frame_data: dict[str, Any] = {
        "id": eval_ids,
        "well": well,
        "target": true_tvt - last_tvt,
        "last_known_tvt": last_tvt,
        "md_since": md_since,
    }
    variant_meta: list[dict[str, Any]] = []
    self_diag_written = False
    if self_gr_variants:
        for variant in self_gr_variants:
            name = str(variant["name"])
            result = run_hmm2(
                horizontal,
                typewell,
                **hmm_config,
                self_gr_config=self_gr_config,
                self_gr_alpha=float(variant["alpha"]),
                self_gr_clip=float(variant["clip"]),
                self_gr_mode=str(variant["mode"]),
            )
            hmm_mean = np.asarray(result["mean_eval"], dtype=np.float64)
            hmm_std = np.asarray(result["std_eval"], dtype=np.float64)
            loglik = float(result["loglik"])
            if not self_diag_written:
                frame_data["self_gr_quality"] = np.asarray(result.get("self_gr_quality"), dtype=np.float32)
                frame_data["self_gr_peak_tvt"] = np.asarray(result.get("self_gr_peak_tvt"), dtype=np.float64)
                frame_data["self_gr_peak_gap"] = np.asarray(result.get("self_gr_peak_gap"), dtype=np.float32)
                frame_data["self_gr_typewell_agreement"] = np.asarray(
                    result.get("self_gr_typewell_agreement"),
                    dtype=np.float32,
                )
                frame_data["self_gr_valid"] = np.asarray(result.get("self_gr_valid"), dtype=np.float32)
                frame_data["self_gr_state_valid_rate"] = np.asarray(
                    result.get("self_gr_state_valid_rate", np.zeros_like(hmm_mean)),
                    dtype=np.float32,
                )
                self_diag_written = True
            frame_data[f"{name}_mean_tvt"] = hmm_mean
            frame_data[f"{name}_mean_d"] = hmm_mean - last_tvt
            frame_data[f"{name}_std"] = hmm_std
            frame_data[f"{name}_loglik"] = loglik
            frame_data[f"{name}_finite"] = np.isfinite(hmm_mean).astype(np.float32)
            variant_meta.append(
                {
                    "name": name,
                    "loglik": loglik,
                    "grid_size": int(len(result["grid"])),
                    "rmse": rmse(true_tvt, hmm_mean),
                    "std_mean": float(np.mean(hmm_std)),
                    "std_p90": float(np.quantile(hmm_std, 0.90)),
                    "self_gr_quality_mean": float(np.mean(np.asarray(result.get("self_gr_quality"), dtype=np.float32))),
                    "self_gr_valid_rate": float(np.mean(np.asarray(result.get("self_gr_valid"), dtype=np.float32))),
                    "self_gr_state_valid_rate": float(
                        np.mean(np.asarray(result.get("self_gr_state_valid_rate"), dtype=np.float32))
                    ),
                    "self_gr_prefix_anchor_count": int(result.get("self_gr_prefix_anchor_count") or 0),
                    "self_gr_prefix_tvt_min": result.get("self_gr_prefix_tvt_min"),
                    "self_gr_prefix_tvt_max": result.get("self_gr_prefix_tvt_max"),
                    "self_gr_prefix_tvt_span": result.get("self_gr_prefix_tvt_span"),
                    "self_gr_sigma": result.get("self_gr_sigma"),
                }
            )
    else:
        result = run_hmm2(horizontal, typewell, **hmm_config)
        hmm_mean = np.asarray(result["mean_eval"], dtype=np.float64)
        hmm_std = np.asarray(result["std_eval"], dtype=np.float64)
        loglik = float(result["loglik"])
        frame_data.update(
            {
                "hmm_mean_tvt": hmm_mean,
                "hmm_mean_d": hmm_mean - last_tvt,
                "hmm_std": hmm_std,
                "hmm_loglik": loglik,
                "hmm_grid_step": float(hmm_config["step"]),
                "hmm_grid_size": int(len(result["grid"])),
                "hmm_prefix_sigma": float(result["prefix_sigma"]),
                "hmm_prefix_ir": float(result["prefix_ir"]),
                "hmm_cal_a": float(result["cal_a"]),
                "hmm_cal_b": float(result["cal_b"]),
                "hmm_finite": np.isfinite(hmm_mean).astype(np.float32),
            }
        )
        variant_meta.append(
            {
                "name": "hmm_mean_tvt",
                "loglik": loglik,
                "grid_size": int(len(result["grid"])),
                "rmse": rmse(true_tvt, hmm_mean),
                "std_mean": float(np.mean(hmm_std)),
                "std_p90": float(np.quantile(hmm_std, 0.90)),
            }
        )
    frame = pd.DataFrame(frame_data)
    finite = bool(np.isfinite(frame[feature_columns].to_numpy(np.float64)).all())
    best_variant = min(variant_meta, key=lambda row: row["rmse"]) if variant_meta else {}
    meta = {
        "well": well,
        "status": "ok" if finite else "non_finite",
        "rows": int(len(frame)),
        "elapsed_seconds": round(time.time() - started, 3),
        "variant_count": int(len(variant_meta)),
        "best_variant": best_variant.get("name"),
        "best_variant_rmse": best_variant.get("rmse"),
        "variant_metrics": variant_meta,
    }
    return _numeric_frame(frame), meta


def run_train_feature_cache(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    hmm_config: dict[str, Any],
    self_gr_emission_config: dict[str, Any] | None = None,
    output_prefix: str = OUTPUT_PREFIX,
    max_wells: int | None = None,
    fast: bool = False,
    numba_num_threads: int | None = None,
    outer_workers: int = 1,
) -> dict[str, Any]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "numba is required to run the exact HMM smoother. "
            "The local import guard exists for validation-only environments; "
            "run the full notebook on Kaggle or install numba in the runtime."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if numba_num_threads:
        set_num_threads(int(numba_num_threads))
    effective_numba_num_threads = get_num_threads()
    self_gr_variants, self_gr_variant_summary = prepare_self_gr_emission_variants(self_gr_emission_config)
    feature_columns = self_gr_variant_feature_columns(self_gr_variants) if self_gr_variants else FEATURE_COLUMNS

    data_dir = Path(data_dir)
    wells = list_well_ids(data_dir)
    if max_wells is None:
        env_max = int(os.environ.get("N_WELLS", "0") or "0")
        max_wells = env_max or None
    if fast and max_wells is None:
        max_wells = 3
    if max_wells is not None:
        wells = wells[: int(max_wells)]

    outer_workers = max(1, int(outer_workers or 1))

    def build_one(idx: int, well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(f"[{idx}/{len(wells)}] self-GR emission HMM smoother well={well}", flush=True)
        frame, meta = build_hmm_rows_for_well(
            well,
            data_dir,
            hmm_config,
            self_gr_variants=self_gr_variants,
            self_gr_config=self_gr_emission_config,
        )
        meta["outer_workers"] = outer_workers
        print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
        return frame, meta

    started = time.time()
    if outer_workers > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise RuntimeError("joblib is required when feature_cache.hmm.outer_workers > 1") from exc
        results = Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(build_one)(idx, well) for idx, well in enumerate(wells, start=1)
        )
        frames = [frame for frame, _ in results]
        well_meta = [meta for _, meta in results]
    else:
        frames = []
        well_meta = []
        for idx, well in enumerate(wells, start=1):
            frame, meta = build_one(idx, well)
            frames.append(frame)
            well_meta.append(meta)

    if not frames:
        raise ValueError("No wells were selected for HMM feature cache generation")
    train_frame = pd.concat(frames, ignore_index=True)
    if train_frame.empty:
        raise ValueError("HMM feature cache is empty")
    numeric_values = train_frame[[*["target"], *feature_columns]].to_numpy(np.float32)
    if not np.isfinite(numeric_values).all():
        raise ValueError("HMM feature cache contains non-finite numeric values")

    train_path = output_dir / f"{output_prefix}_{VARIANT}_train_features.csv.gz"
    schema_path = output_dir / f"{output_prefix}_feature_schema.csv"
    well_summary_path = output_dir / f"{output_prefix}_by_well_generation_summary.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    train_frame[[*META_COLUMNS, *feature_columns]].to_csv(
        train_path,
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {
            "variant": VARIANT,
            "feature_index": np.arange(len(feature_columns), dtype=np.int32),
            "feature": feature_columns,
        }
    ).to_csv(schema_path, index=False)
    well_summary = pd.DataFrame(well_meta)
    well_summary.to_csv(well_summary_path, index=False)

    ok_meta = [row for row in well_meta if row.get("status") == "ok"]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_feature_cache_completed",
        "mode": "state_known_tvt_self_gr_hmm_emission_train_feature_cache",
        "source_notebook": "amerhu/rogii-wellbore-geology-exact-hmm-smoother",
        "variant": VARIANT,
        "rows": int(len(train_frame)),
        "wells": int(train_frame["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "hmm_config": hmm_config,
        "self_gr_emission": {
            "enabled": bool(self_gr_variants),
            "variant_count": int(len(self_gr_variants)),
            "variants": self_gr_variant_summary,
            "surface_config": self_gr_emission_config,
        },
        "output_prefix": output_prefix,
        "outer_workers": outer_workers,
        "numba_num_threads_requested": int(numba_num_threads) if numba_num_threads else None,
        "numba_num_threads": int(effective_numba_num_threads) if effective_numba_num_threads else None,
        "well_generation": {
            "selected_wells": int(len(wells)),
            "ok_wells": int(len(ok_meta)),
            "skipped_wells": int(len(well_meta) - len(ok_meta)),
            "mean_elapsed_seconds_per_ok_well": (
                float(np.mean([row["elapsed_seconds"] for row in ok_meta])) if ok_meta else None
            ),
            "mean_best_variant_rmse_train_side": (
                float(np.mean([row["best_variant_rmse"] for row in ok_meta])) if ok_meta else None
            ),
        },
        "outputs": {
            "train_features": train_path.name,
            "feature_schema": schema_path.name,
            "by_well_generation_summary": well_summary_path.name,
            "summary": summary_path.name,
        },
        "sha256": {
            "train_features_gzip": sha256_path(train_path),
            "train_features_decompressed": sha256_gzip_decompressed(train_path),
            "feature_schema": sha256_path(schema_path),
            "by_well_generation_summary": sha256_path(well_summary_path),
        },
        "notes": [
            "This cache is generated from raw train horizontal/typewell files only.",
            "Self-GR likelihood uses full horizontal GR and finite TVT_input prefix only; true TVT is not used to build the surface or tune alpha/clip.",
            "Self-GR likelihood is neutral for candidate TVT states outside the known-prefix TVT_input range.",
            "Unknown-suffix train TVT is used only for target and generation summary metrics.",
            "No test feature, model, inference, or submission output is generated.",
            "exp072 likpf_mean and HMM/self-GR variants are compared only by the separate direct comparison helper.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary
