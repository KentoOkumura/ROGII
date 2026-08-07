from __future__ import annotations

import gzip
import hashlib
import json
import math
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


EXPERIMENT_NAME = "exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation"
OUTPUT_PREFIX = "exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation"
VARIANT = "raw_hmm_missing_gr_neutral"
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
            matches = [
                path
                for path in sorted(KAGGLE_INPUT_ROOT.rglob(basename))
                if path.stat().st_size > 0
            ]
            checked.extend(str(path) for path in matches)
            if matches:
                return matches[0]
    raise FileNotFoundError("No non-empty candidate path exists: " + json.dumps(checked, indent=2))




def load_well(well: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    horizontal = pd.read_csv(data_dir / f"{well}__horizontal_well.csv")
    typewell = (
        pd.read_csv(data_dir / f"{well}__typewell.csv").sort_values("TVT").reset_index(drop=True)
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


def build_gr_emission_loglik(
    gr: np.ndarray,
    gr_grid: np.ndarray,
    gr_sigma: float,
    *,
    emission: str,
    df: float,
    raw_gr_missing: np.ndarray | None = None,
    mask_missing_gr: bool = False,
) -> np.ndarray:
    """Build the exp209 raw typewell-GR unary and optionally neutralize missing rows.

    A neutralized row contributes exactly zero GR log-likelihood for every TVT
    state. The HMM transition remains active; exp269 does not pass any additional
    self-GR or LightGBM unary to ``run_hmm2``.
    """
    gr = np.asarray(gr, dtype=np.float64)
    gr_grid = np.asarray(gr_grid, dtype=np.float64)
    zscore = (gr[:, None] - gr_grid[None, :]) / float(gr_sigma)
    if emission == "t":
        emission_ll = (-0.5 * (float(df) + 1.0) * np.log1p(zscore**2 / float(df))).astype(
            np.float32
        )
    else:
        emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    if mask_missing_gr:
        if raw_gr_missing is None:
            raise ValueError("mask_missing_gr=True requires raw_gr_missing")
        missing = np.asarray(raw_gr_missing, dtype=bool)
        if len(missing) != len(emission_ll):
            raise ValueError(
                f"raw_gr_missing length mismatch: expected {len(emission_ll)} got {len(missing)}"
            )
        emission_ll[missing, :] = np.float32(0.0)
    return emission_ll


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
                        total += np.exp(
                            position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2] - best
                        )
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
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best
                        )
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
    mask_missing_gr: bool = False,
) -> dict[str, Any]:
    """Second-order raw typewell-GR HMM kept at the exp209 control settings."""
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
            "raw_gr_missing_eval": np.array([], dtype=bool),
            "masked_gr_rows": 0,
        }

    cal_a, cal_b, robust_sigma, init_rate = prefix_stats(horizontal, typewell_tvt, typewell_gr)
    if sigma_mode == "std":
        typewell_at_known = np.interp(
            known["TVT_input"].to_numpy(np.float64), typewell_tvt, typewell_gr
        )
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
    raw_gr_eval = pd.to_numeric(horizontal.loc[eval_rows.index, "GR"], errors="coerce").to_numpy(
        np.float64
    )
    raw_gr_missing = ~np.isfinite(raw_gr_eval)
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

    emission_ll = build_gr_emission_loglik(
        gr,
        gr_grid,
        gr_sigma,
        emission=emission,
        df=df,
        raw_gr_missing=raw_gr_missing,
        mask_missing_gr=mask_missing_gr,
    )

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
        "raw_gr_missing_eval": raw_gr_missing,
        "masked_gr_rows": int(raw_gr_missing.sum()) if mask_missing_gr else 0,
        "mask_missing_gr": bool(mask_missing_gr),
    }
    if return_post:
        result["post"] = post_p
        result["md_eval"] = md
    return result


def run_raw_hmm_missing_gr_neutral(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    **hmm_config: Any,
) -> dict[str, Any]:
    """Run the exp269 scientific path with no auxiliary observation unary."""

    forbidden = {
        "lgb_tvt",
        "lgb_sigma",
        "lgb_lambda",
        "lgb_emission_clip",
        "mask_missing_gr",
    }.intersection(hmm_config)
    if forbidden:
        raise ValueError(f"exp269 raw-HMM wrapper forbids override keys: {sorted(forbidden)}")
    return run_hmm2(
        horizontal,
        typewell,
        **hmm_config,
        mask_missing_gr=True,
    )
