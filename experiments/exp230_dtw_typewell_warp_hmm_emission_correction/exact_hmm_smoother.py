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


EXPERIMENT_NAME = "exp230_dtw_typewell_warp_hmm_emission_correction"
OUTPUT_PREFIX = "exp230_dtw_typewell_warp_hmm_emission_correction"
VARIANT = "dtw_typewell_warp_hmm_emission_correction"
META_COLUMNS = ["id", "well", "target"]
BASE_FEATURE_COLUMNS = [
    "last_known_tvt",
    "md_since",
    "hmm_grid_step",
    "hmm_grid_size",
    "hmm_prefix_sigma",
    "hmm_prefix_ir",
    "hmm_cal_a",
    "hmm_cal_b",
    "dtw_available",
    "dtw_cost",
    "dtw_path_slope",
    "dtw_anchor_abs_error",
    "dtw_confidence",
    "dtw_stochastic_std",
    "dtw_stochastic_cv",
]


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


def sanitize_token(value: Any) -> str:
    text = str(value).lower().replace(".", "p").replace("-", "m")
    keep = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(keep).split("_") if part) or "x"


def stable_int_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def dtw_variant_name(variant: dict[str, Any]) -> str:
    if variant.get("name"):
        return str(variant["name"])
    alpha = int(round(float(variant.get("alpha", 0.0)) * 1000))
    sigma = int(round(float(variant.get("sigma_base", 12.0)) * 100))
    return f"hmm_dtw_a{alpha:03d}_s{sigma:04d}"


def prepare_dtw_variants(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    config = dict(config or {})
    if not bool(config.get("enabled", False)):
        return []
    active = list(config.get("active_variants") or [])
    variants: list[dict[str, Any]] = []
    if active:
        for raw in active:
            variant = {**config, **dict(raw or {})}
            variant["name"] = dtw_variant_name(variant)
            variants.append(variant)
    else:
        for alpha in config.get("alpha_grid") or [0.05]:
            variant = {**config, "alpha": float(alpha)}
            variant["name"] = dtw_variant_name(variant)
            variants.append(variant)
    max_variants = config.get("max_variants")
    if max_variants is not None:
        variants = variants[: int(max_variants)]
    return variants


def dtw_feature_columns(variants: list[dict[str, Any]]) -> list[str]:
    columns = list(BASE_FEATURE_COLUMNS)
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


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)
    fill = float(np.nanmedian(values[finite]))
    clean = np.where(finite, values, fill)
    center = float(np.median(clean))
    scale = float(1.4826 * np.median(np.abs(clean - center)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(clean))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return np.clip((clean - center) / scale, -8.0, 8.0)


def _sample_positions(n: int, max_points: int, seed: int | None = None) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    count = min(int(max_points), n)
    if count <= 1:
        return np.array([0], dtype=np.int64)
    pos = np.linspace(0, n - 1, count, dtype=np.float64)
    if seed is not None and count > 2:
        rng = np.random.default_rng(seed)
        step = max(1.0, (n - 1) / max(count - 1, 1))
        jitter = rng.uniform(-0.35 * step, 0.35 * step, size=count)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        pos = np.clip(pos + jitter, 0.0, float(n - 1))
    idx = np.unique(np.rint(pos).astype(np.int64))
    idx = np.unique(np.concatenate([idx, np.array([0, n - 1], dtype=np.int64)]))
    return idx


@njit(cache=True, nogil=True)
def _constrained_dtw_path(a: np.ndarray, b: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(a)
    m = len(b)
    inf = 1e30
    dp = np.full((n, m), inf, np.float64)
    move = np.full((n, m), -1, np.int8)
    if n == 0 or m == 0:
        return np.array([-1], dtype=np.int64), np.array([-1], dtype=np.int64), inf
    for i in range(n):
        center = 0.0
        if n > 1:
            center = i * (m - 1) / (n - 1)
        lo = int(max(0, math.floor(center - band)))
        hi = int(min(m - 1, math.ceil(center + band)))
        for j in range(lo, hi + 1):
            cost = (a[i] - b[j]) ** 2
            if i == 0 and j == 0:
                dp[i, j] = cost
                move[i, j] = 0
                continue
            best = inf
            best_move = -1
            if i > 0 and j > 0 and dp[i - 1, j - 1] < best:
                best = dp[i - 1, j - 1]
                best_move = 0
            if i > 0 and dp[i - 1, j] < best:
                best = dp[i - 1, j]
                best_move = 1
            if j > 0 and dp[i, j - 1] < best:
                best = dp[i, j - 1]
                best_move = 2
            if best < inf / 2:
                dp[i, j] = best + cost
                move[i, j] = best_move
    if not np.isfinite(dp[n - 1, m - 1]) or move[n - 1, m - 1] < 0:
        return np.array([-1], dtype=np.int64), np.array([-1], dtype=np.int64), inf
    path_i = np.empty(n + m, dtype=np.int64)
    path_j = np.empty(n + m, dtype=np.int64)
    length = 0
    i = n - 1
    j = m - 1
    while True:
        path_i[length] = i
        path_j[length] = j
        length += 1
        if i == 0 and j == 0:
            break
        step = move[i, j]
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            break
    out_i = np.empty(length, dtype=np.int64)
    out_j = np.empty(length, dtype=np.int64)
    for k in range(length):
        out_i[k] = path_i[length - 1 - k]
        out_j[k] = path_j[length - 1 - k]
    return out_i, out_j, float(dp[n - 1, m - 1] / max(length, 1))


def _dtw_anchor(
    eval_gr: np.ndarray,
    type_gr: np.ndarray,
    grid: np.ndarray,
    *,
    max_points: int,
    band_fraction: float,
    min_band: int,
    seed: int | None = None,
) -> tuple[np.ndarray, float, float]:
    eval_idx = _sample_positions(len(eval_gr), max_points, seed=seed)
    type_idx = _sample_positions(len(type_gr), max_points, seed=None if seed is None else seed + 17)
    if len(eval_idx) < 2 or len(type_idx) < 2:
        return np.full(len(eval_gr), np.nan, dtype=np.float64), np.inf, np.nan
    a = _robust_standardize(eval_gr[eval_idx]).astype(np.float64)
    b = _robust_standardize(type_gr[type_idx]).astype(np.float64)
    band = max(int(min_band), int(round(max(len(a), len(b)) * float(band_fraction))))
    path_i, path_j, cost = _constrained_dtw_path(a, b, band)
    if len(path_i) == 1 and int(path_i[0]) < 0:
        return np.full(len(eval_gr), np.nan, dtype=np.float64), np.inf, np.nan
    path_eval_pos = eval_idx[path_i]
    path_tvt = grid[type_idx[path_j]]
    grouped_pos: list[int] = []
    grouped_tvt: list[float] = []
    for pos in np.unique(path_eval_pos):
        mask = path_eval_pos == pos
        grouped_pos.append(int(pos))
        grouped_tvt.append(float(np.mean(path_tvt[mask])))
    anchor = np.interp(
        np.arange(len(eval_gr), dtype=np.float64),
        np.asarray(grouped_pos, dtype=np.float64),
        np.asarray(grouped_tvt, dtype=np.float64),
    )
    if len(grouped_pos) >= 2:
        slope = float((grouped_tvt[-1] - grouped_tvt[0]) / max(grouped_pos[-1] - grouped_pos[0], 1))
    else:
        slope = np.nan
    return anchor, float(cost), slope


def build_dtw_emission_signal(
    *,
    well: str,
    eval_gr: np.ndarray,
    type_gr_grid: np.ndarray,
    grid: np.ndarray,
    last_tvt: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_points = int(config.get("max_points", 384))
    band_fraction = float(config.get("band_fraction", 0.16))
    min_band = int(config.get("min_band", 12))
    stochastic_k = int(config.get("stochastic_k", 3))
    sigma_base = float(config.get("sigma_base", 12.0))
    sigma_floor = float(config.get("sigma_floor", 6.0))
    sigma_cap = float(config.get("sigma_cap", 35.0))
    stochastic_sigma_scale = float(config.get("stochastic_sigma_scale", 1.0))
    emission_clip = float(config.get("emission_clip", 120.0))
    cost_scale = float(config.get("confidence_cost_scale", 1.5))
    anchor_scale = float(config.get("confidence_anchor_scale", 30.0))
    stochastic_scale = float(config.get("confidence_stochastic_scale", 20.0))

    anchor, cost, slope = _dtw_anchor(
        eval_gr,
        type_gr_grid,
        grid,
        max_points=max_points,
        band_fraction=band_fraction,
        min_band=min_band,
    )
    available = bool(np.isfinite(anchor).all() and np.isfinite(cost))
    if not available:
        zeros = np.zeros((len(eval_gr), len(grid)), dtype=np.float32)
        return {
            "available": 0.0,
            "ll": zeros,
            "cost": np.nan,
            "path_slope": np.nan,
            "anchor_abs_error": np.nan,
            "confidence": 0.0,
            "stochastic_std": np.nan,
            "stochastic_cv": np.nan,
        }

    jitter_anchors: list[np.ndarray] = []
    for idx in range(stochastic_k):
        seed = stable_int_seed(EXPERIMENT_NAME, well, "dtw", idx)
        jitter_anchor, _, _ = _dtw_anchor(
            eval_gr,
            type_gr_grid,
            grid,
            max_points=max_points,
            band_fraction=band_fraction,
            min_band=min_band,
            seed=seed,
        )
        if np.isfinite(jitter_anchor).all():
            jitter_anchors.append(jitter_anchor)
    if jitter_anchors:
        stack = np.vstack([anchor, *jitter_anchors])
        row_std = np.nanstd(stack, axis=0)
        stochastic_std = float(np.nanmean(row_std))
        stochastic_cv = float(stochastic_std / max(float(np.nanstd(anchor)), 1e-6))
    else:
        row_std = np.zeros(len(anchor), dtype=np.float64)
        stochastic_std = 0.0
        stochastic_cv = 0.0

    anchor_abs_error = float(abs(anchor[0] - last_tvt)) if len(anchor) else np.nan
    confidence = math.exp(-cost_scale * float(cost))
    if np.isfinite(anchor_abs_error):
        confidence *= math.exp(-anchor_abs_error / max(anchor_scale, 1e-6))
    confidence *= math.exp(-stochastic_std / max(stochastic_scale, 1e-6))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    sigma = np.clip(sigma_base + stochastic_sigma_scale * row_std, sigma_floor, sigma_cap)
    zscore = (grid[None, :] - anchor[:, None]) / sigma[:, None]
    ll = (-0.5 * np.minimum(zscore**2, emission_clip) * confidence).astype(np.float32)
    return {
        "available": 1.0,
        "ll": ll,
        "cost": float(cost),
        "path_slope": slope,
        "anchor_abs_error": anchor_abs_error,
        "confidence": confidence,
        "stochastic_std": stochastic_std,
        "stochastic_cv": stochastic_cv,
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
    dtw_emission: dict[str, Any] | None = None,
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

    dtw_signal: dict[str, Any] | None = None
    if dtw_emission is not None and bool(dtw_emission.get("enabled", True)):
        alpha = float(dtw_emission.get("alpha", 0.0))
        if alpha != 0.0:
            dtw_signal = build_dtw_emission_signal(
                well=str(dtw_emission.get("well", "unknown")),
                eval_gr=gr,
                type_gr_grid=gr_grid,
                grid=grid,
                last_tvt=last_tvt,
                config=dtw_emission,
            )
            emission_ll = (emission_ll + np.float32(alpha) * dtw_signal["ll"]).astype(np.float32)

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
        "dtw_signal": dtw_signal,
    }
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
    dtw_variants: list[dict[str, Any]],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    if not dtw_variants:
        raise ValueError("At least one DTW HMM variant must be configured")
    first_variant = {**dtw_variants[0], "well": well}
    result = run_hmm2(horizontal, typewell, **hmm_config, dtw_emission=first_variant)
    eval_index = result["ev_index"]
    known = horizontal.loc[known_mask]
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_md = float(last["MD"])
    true_tvt = horizontal.loc[eval_index, "TVT"].to_numpy(np.float64)
    md_since = horizontal.loc[eval_index, "MD"].to_numpy(np.float64) - last_md
    dtw_signal = result.get("dtw_signal") or {}
    frame_data: dict[str, Any] = {
        "id": [f"{well}_{int(i)}" for i in eval_index],
        "well": well,
        "target": true_tvt - last_tvt,
        "last_known_tvt": last_tvt,
        "md_since": md_since,
        "hmm_grid_step": float(hmm_config["step"]),
        "hmm_grid_size": int(len(result["grid"])),
        "hmm_prefix_sigma": float(result["prefix_sigma"]),
        "hmm_prefix_ir": float(result["prefix_ir"]),
        "hmm_cal_a": float(result["cal_a"]),
        "hmm_cal_b": float(result["cal_b"]),
        "dtw_available": float(dtw_signal.get("available", 0.0)),
        "dtw_cost": float(dtw_signal.get("cost", np.nan)),
        "dtw_path_slope": float(dtw_signal.get("path_slope", np.nan)),
        "dtw_anchor_abs_error": float(dtw_signal.get("anchor_abs_error", np.nan)),
        "dtw_confidence": float(dtw_signal.get("confidence", 0.0)),
        "dtw_stochastic_std": float(dtw_signal.get("stochastic_std", np.nan)),
        "dtw_stochastic_cv": float(dtw_signal.get("stochastic_cv", np.nan)),
    }
    variant_meta: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(dtw_variants):
        variant_result = result if variant_index == 0 else run_hmm2(
            horizontal,
            typewell,
            **hmm_config,
            dtw_emission={**variant, "well": well},
        )
        name = str(variant["name"])
        hmm_mean = np.asarray(variant_result["mean_eval"], dtype=np.float64)
        hmm_std = np.asarray(variant_result["std_eval"], dtype=np.float64)
        loglik = float(variant_result["loglik"])
        frame_data[f"{name}_mean_tvt"] = hmm_mean
        frame_data[f"{name}_mean_d"] = hmm_mean - last_tvt
        frame_data[f"{name}_std"] = hmm_std
        frame_data[f"{name}_loglik"] = loglik
        frame_data[f"{name}_finite"] = np.isfinite(hmm_mean).astype(np.float32)
        variant_meta.append(
            {
                "name": name,
                "alpha": float(variant.get("alpha", 0.0)),
                "loglik": loglik,
                "grid_size": int(len(variant_result["grid"])),
                "hmm_rmse": rmse(true_tvt, hmm_mean),
                "hmm_std_mean": float(np.mean(hmm_std)),
                "hmm_std_p90": float(np.quantile(hmm_std, 0.90)),
            }
        )
    frame = pd.DataFrame(frame_data)
    finite = bool(np.isfinite(frame[feature_columns].to_numpy(np.float64)).all())
    best_variant = min(variant_meta, key=lambda row: row["hmm_rmse"]) if variant_meta else {}
    meta = {
        "well": well,
        "status": "ok" if finite else "non_finite",
        "rows": int(len(frame)),
        "elapsed_seconds": round(time.time() - started, 3),
        "variant_count": int(len(variant_meta)),
        "best_variant": best_variant.get("name"),
        "best_variant_rmse": best_variant.get("hmm_rmse"),
        "variant_metrics": variant_meta,
        "grid_size": int(len(result["grid"])),
        "dtw_available": float(dtw_signal.get("available", 0.0)),
        "dtw_cost": float(dtw_signal.get("cost", np.nan)),
        "dtw_confidence": float(dtw_signal.get("confidence", 0.0)),
        "dtw_anchor_abs_error": float(dtw_signal.get("anchor_abs_error", np.nan)),
        "dtw_stochastic_std": float(dtw_signal.get("stochastic_std", np.nan)),
    }
    return _numeric_frame(frame), meta


def run_train_feature_cache(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    hmm_config: dict[str, Any],
    dtw_config: dict[str, Any],
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
    dtw_variants = prepare_dtw_variants(dtw_config)
    if not dtw_variants:
        raise ValueError("dtw_emission.enabled must define at least one active variant")
    feature_columns = dtw_feature_columns(dtw_variants)

    def build_one(idx: int, well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(f"[{idx}/{len(wells)}] DTW-emission HMM smoother well={well}", flush=True)
        frame, meta = build_hmm_rows_for_well(well, data_dir, hmm_config, dtw_variants, feature_columns)
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
        "mode": "dtw_typewell_warp_hmm_emission_train_feature_cache",
        "source_notebook": "amerhu/rogii-wellbore-geology-exact-hmm-smoother",
        "variant": VARIANT,
        "dtw_variants": [
            {key: value for key, value in variant.items() if key not in {"ll"}}
            for variant in dtw_variants
        ],
        "rows": int(len(train_frame)),
        "wells": int(train_frame["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "hmm_config": hmm_config,
        "dtw_config": dtw_config,
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
            "mean_dtw_confidence": (
                float(np.mean([row["dtw_confidence"] for row in ok_meta])) if ok_meta else None
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
            "Full horizontal GR is used only to build a weak DTW auxiliary emission; true tail TVT is not used in generation.",
            "Unknown-suffix train TVT is used only for target and generation summary metrics.",
            "No test feature, model, inference, or submission output is generated.",
            "DTW path TVT is not written as a selectable prediction candidate.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary
