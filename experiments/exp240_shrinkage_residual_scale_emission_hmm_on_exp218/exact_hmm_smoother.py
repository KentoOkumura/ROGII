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


EXPERIMENT_NAME = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"
OUTPUT_PREFIX = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"
VARIANT = "shrinkage_residual_scale_emission_hmm"
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


def lgb_variant_name(source: str, sigma: float, weight: float) -> str:
    sigma_token = f"s{int(round(float(sigma) * 100)):04d}"
    weight_token = f"l{int(round(float(weight) * 1000)):04d}"
    return f"hmm_lgb_{sanitize_token(source)}_{sigma_token}_{weight_token}"


def lgb_band_variant_name(source: str, sigma_floor: float, sigma_cap: float, weight: float) -> str:
    floor_token = f"sf{int(round(float(sigma_floor) * 100)):04d}"
    cap_token = f"sc{int(round(float(sigma_cap) * 100)):04d}"
    weight_token = f"l{int(round(float(weight) * 1000)):04d}"
    return f"hmm_lgb_{sanitize_token(source)}_band_{floor_token}_{cap_token}_{weight_token}"


def lgb_variant_feature_columns(variants: list[dict[str, Any]]) -> list[str]:
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


def load_lgb_prediction_source(root: str | Path, source_name: str, source_config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_existing_file(root, list(source_config.get("candidates") or []))
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    id_column = str(source_config.get("id_column") or "id")
    pred_column = source_config.get("prediction_column")
    if pred_column is None:
        if "pred_tvt" in columns:
            pred_column = "pred_tvt"
        elif "tvt" in columns:
            pred_column = "tvt"
        else:
            raise ValueError(f"{source_name} prediction source needs pred_tvt or tvt column: {path}")
    pred_column = str(pred_column)
    sigma_column = source_config.get("sigma_column")
    sigma_column = str(sigma_column) if sigma_column is not None else None
    if id_column not in columns or pred_column not in columns:
        raise ValueError(f"{source_name} source missing columns id={id_column!r} pred={pred_column!r}: {path}")
    if sigma_column is not None and sigma_column not in columns:
        raise ValueError(f"{source_name} source missing configured sigma column {sigma_column!r}: {path}")

    usecols = [id_column, pred_column]
    if sigma_column is not None:
        usecols.append(sigma_column)
    model_filter = source_config.get("model_filter")
    if model_filter is not None and "model" in columns:
        usecols.append("model")
    frame = pd.read_csv(path, usecols=usecols, dtype={id_column: str})
    if model_filter is not None and "model" in frame.columns:
        frame = frame[frame["model"].astype(str) == str(model_filter)].copy()
    frame = frame.rename(columns={id_column: "id", pred_column: "pred_tvt"})
    frame["pred_tvt"] = pd.to_numeric(frame["pred_tvt"], errors="coerce")
    if sigma_column is not None:
        frame = frame.rename(columns={sigma_column: "sigma_tvt"})
        frame["sigma_tvt"] = pd.to_numeric(frame["sigma_tvt"], errors="coerce")
    before = len(frame)
    required = ["id", "pred_tvt", *([] if sigma_column is None else ["sigma_tvt"])]
    frame = frame.dropna(subset=required)
    duplicated = int(frame["id"].duplicated().sum())
    if duplicated:
        raise ValueError(f"{source_name} prediction source has duplicated ids after filtering: {duplicated}")
    series = frame.set_index("id")["pred_tvt"].astype(np.float64)
    sigma_series = None
    if sigma_column is not None:
        sigma_series = frame.set_index("id")["sigma_tvt"].astype(np.float64)
        if not np.isfinite(sigma_series.to_numpy()).all() or float(sigma_series.min()) <= 0.0:
            raise ValueError(f"{source_name} sigma column must be finite and positive")
    meta = {
        "source": source_name,
        "path": str(path),
        "rows_loaded": int(len(series)),
        "rows_before_dropna": int(before),
        "prediction_column": pred_column,
        "sigma_column": sigma_column,
        "model_filter": model_filter,
        "prediction_min": float(series.min()) if len(series) else None,
        "prediction_max": float(series.max()) if len(series) else None,
    }
    if sigma_series is not None:
        meta.update(
            {
                "sigma_min": float(sigma_series.min()),
                "sigma_mean": float(sigma_series.mean()),
                "sigma_p90": float(sigma_series.quantile(0.90)),
                "sigma_max": float(sigma_series.max()),
            }
        )
    return {
        "predictions": series,
        "sigmas": sigma_series,
        "meta": meta,
    }


def load_lgb_prediction_series(root: str | Path, source_name: str, source_config: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    source = load_lgb_prediction_source(root, source_name, source_config)
    return source["predictions"], source["meta"]


def prepare_lgb_emission_variants(root: str | Path, config: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return [], []
    sources_config = config.get("sources") or {}
    active_sources = list(config.get("active_sources") or sources_config.keys())
    lambda_values = [float(v) for v in (config.get("lambda_grid") or [])]
    if not active_sources or not lambda_values:
        raise ValueError("lgb_emission.enabled requires active_sources and lambda_grid")

    max_variants = config.get("max_variants")
    max_variants = int(max_variants) if max_variants is not None else None
    sigma_values = [float(v) for v in (config.get("sigma_grid") or [])]
    sigma_floor = float(config.get("sigma_floor", min(sigma_values) if sigma_values else 6.0))
    sigma_cap = float(config.get("sigma_cap", max(sigma_values) if sigma_values else 30.0))
    sigma_floor_grid = [float(v) for v in (config.get("sigma_floor_grid") or [sigma_floor])]
    sigma_cap_grid = [float(v) for v in (config.get("sigma_cap_grid") or [sigma_cap])]
    emission_clip = float(config.get("emission_clip", 600.0))
    runtime_variants: list[dict[str, Any]] = []
    summary_variants: list[dict[str, Any]] = []
    loaded_sources: dict[str, dict[str, Any]] = {}

    for source in active_sources:
        if source not in sources_config:
            raise KeyError(f"active LGB source is not configured: {source}")
        if source not in loaded_sources:
            loaded_sources[source] = load_lgb_prediction_source(root, source, sources_config[source])
        source_payload = loaded_sources[source]
        predictions = source_payload["predictions"]
        source_meta = source_payload["meta"]
        source_sigmas = source_payload.get("sigmas")
        if source_sigmas is not None:
            for floor in sigma_floor_grid:
                for cap in sigma_cap_grid:
                    if float(floor) <= 0.0 or float(cap) < float(floor):
                        raise ValueError(f"invalid residual-scale sigma floor/cap: floor={floor} cap={cap}")
                    sigma_eff = source_sigmas.clip(lower=float(floor), upper=float(cap)).astype(np.float64)
                    for weight in lambda_values:
                        name = lgb_band_variant_name(source, float(floor), float(cap), float(weight))
                        variant = {
                            "name": name,
                            "source": source,
                            "predictions": predictions,
                            "sigma": sigma_eff,
                            "sigma_mode": "rowwise_source_column",
                            "sigma_floor": float(floor),
                            "sigma_cap": float(cap),
                            "lambda": float(weight),
                            "emission_clip": emission_clip,
                        }
                        runtime_variants.append(variant)
                        summary_variants.append(
                            {
                                "name": name,
                                "source": source,
                                "sigma_mode": "rowwise_source_column",
                                "sigma_floor": float(floor),
                                "sigma_cap": float(cap),
                                "sigma_source_min": float(source_sigmas.min()),
                                "sigma_source_mean": float(source_sigmas.mean()),
                                "sigma_source_p90": float(source_sigmas.quantile(0.90)),
                                "sigma_source_max": float(source_sigmas.max()),
                                "sigma_effective_min": float(sigma_eff.min()),
                                "sigma_effective_mean": float(sigma_eff.mean()),
                                "sigma_effective_p90": float(sigma_eff.quantile(0.90)),
                                "sigma_effective_max": float(sigma_eff.max()),
                                "lambda": float(weight),
                                "emission_clip": emission_clip,
                                "source_meta": source_meta,
                            }
                        )
                        if max_variants is not None and len(runtime_variants) >= max_variants:
                            return runtime_variants, summary_variants
            continue
        if not sigma_values:
            raise ValueError(f"{source} has no sigma_column, so lgb_emission.sigma_grid is required")
        for sigma in sigma_values:
            for weight in lambda_values:
                sigma_eff = float(np.clip(float(sigma), sigma_floor, sigma_cap))
                name = lgb_variant_name(source, sigma_eff, float(weight))
                variant = {
                    "name": name,
                    "source": source,
                    "predictions": predictions,
                    "sigma": sigma_eff,
                    "lambda": float(weight),
                    "emission_clip": emission_clip,
                }
                runtime_variants.append(variant)
                summary_variants.append(
                    {
                        "name": name,
                        "source": source,
                        "sigma": sigma_eff,
                        "lambda": float(weight),
                        "emission_clip": emission_clip,
                        "source_meta": source_meta,
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
    lgb_tvt: np.ndarray | None = None,
    lgb_sigma: float | np.ndarray | None = None,
    lgb_lambda: float = 0.0,
    lgb_emission_clip: float = 600.0,
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

    if lgb_tvt is not None and float(lgb_lambda) != 0.0:
        lgb_obs = np.asarray(lgb_tvt, dtype=np.float64)
        if len(lgb_obs) != len(eval_rows):
            raise ValueError(f"lgb_tvt length mismatch: expected {len(eval_rows)} got {len(lgb_obs)}")
        if not np.isfinite(lgb_obs).all():
            raise ValueError("lgb_tvt contains non-finite values")
        sigma_obj = lgb_sigma if lgb_sigma is not None else 0.0
        if np.isscalar(sigma_obj):
            sigma = float(sigma_obj)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"lgb_sigma must be positive, got {lgb_sigma}")
            lgb_zscore = (grid[None, :] - lgb_obs[:, None]) / sigma
        else:
            sigma_values = np.asarray(sigma_obj, dtype=np.float64)
            if len(sigma_values) != len(eval_rows):
                raise ValueError(
                    f"lgb_sigma length mismatch: expected {len(eval_rows)} got {len(sigma_values)}"
                )
            if not np.isfinite(sigma_values).all() or float(np.min(sigma_values)) <= 0.0:
                raise ValueError("row-wise lgb_sigma must be finite and positive")
            lgb_zscore = (grid[None, :] - lgb_obs[:, None]) / sigma_values[:, None]
        lgb_ll = (-0.5 * np.minimum(lgb_zscore**2, float(lgb_emission_clip))).astype(np.float32)
        emission_ll = (emission_ll + np.float32(float(lgb_lambda)) * lgb_ll).astype(np.float32)

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
        "lgb_sigma": lgb_sigma,
        "lgb_lambda": float(lgb_lambda),
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


def lgb_predictions_for_eval_ids(variant: dict[str, Any], ids: list[str]) -> np.ndarray:
    series = variant["predictions"]
    values = series.reindex(ids)
    missing = int(values.isna().sum())
    if missing:
        example = values[values.isna()].index[:5].tolist()
        raise ValueError(f"{variant['name']} missing {missing} LGB predictions, examples={example}")
    return values.to_numpy(np.float64)


def lgb_sigmas_for_eval_ids(variant: dict[str, Any], ids: list[str]) -> float | np.ndarray:
    sigma = variant["sigma"]
    if hasattr(sigma, "reindex"):
        values = sigma.reindex(ids)
        missing = int(values.isna().sum())
        if missing:
            example = values[values.isna()].index[:5].tolist()
            raise ValueError(f"{variant['name']} missing {missing} LGB sigma values, examples={example}")
        sigma_values = values.to_numpy(np.float64)
        if not np.isfinite(sigma_values).all() or float(np.min(sigma_values)) <= 0.0:
            raise ValueError(f"{variant['name']} has non-positive or non-finite sigma values")
        return sigma_values
    return float(sigma)


def build_hmm_rows_for_well(
    well: str,
    data_dir: str | Path,
    hmm_config: dict[str, Any],
    lgb_variants: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lgb_variants = lgb_variants or []
    feature_columns = lgb_variant_feature_columns(lgb_variants) if lgb_variants else FEATURE_COLUMNS
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
    if lgb_variants:
        for variant in lgb_variants:
            name = str(variant["name"])
            lgb_tvt = lgb_predictions_for_eval_ids(variant, eval_ids)
            result = run_hmm2(
                horizontal,
                typewell,
                **hmm_config,
                lgb_tvt=lgb_tvt,
                lgb_sigma=lgb_sigmas_for_eval_ids(variant, eval_ids),
                lgb_lambda=float(variant["lambda"]),
                lgb_emission_clip=float(variant["emission_clip"]),
            )
            hmm_mean = np.asarray(result["mean_eval"], dtype=np.float64)
            hmm_std = np.asarray(result["std_eval"], dtype=np.float64)
            loglik = float(result["loglik"])
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


def build_hmm_inference_rows_for_well(
    well: str,
    data_dir: str | Path,
    hmm_config: dict[str, Any],
    lgb_variants: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not lgb_variants:
        raise ValueError("HMM inference requires at least one LGB emission variant")
    horizontal, typewell = load_well(well, data_dir)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if int(eval_mask.sum()) == 0:
        return pd.DataFrame(columns=["id", "well"]), {
            "well": well,
            "status": "skipped_no_eval_rows",
            "rows": 0,
        }
    if int(known_mask.sum()) == 0:
        raise ValueError(f"{well} has no known TVT_input prefix rows")

    started = time.time()
    eval_index = horizontal.index[eval_mask].to_numpy(np.int64)
    eval_ids = [f"{well}_{int(i)}" for i in eval_index]
    known = horizontal.loc[known_mask]
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_md = float(last["MD"])
    md_since = horizontal.loc[eval_index, "MD"].to_numpy(np.float64) - last_md
    frame_data: dict[str, Any] = {
        "id": eval_ids,
        "well": well,
        "last_known_tvt": last_tvt,
        "md_since": md_since,
    }
    variant_meta: list[dict[str, Any]] = []
    for variant in lgb_variants:
        name = str(variant["name"])
        lgb_tvt = lgb_predictions_for_eval_ids(variant, eval_ids)
        result = run_hmm2(
            horizontal,
            typewell,
            **hmm_config,
            lgb_tvt=lgb_tvt,
            lgb_sigma=lgb_sigmas_for_eval_ids(variant, eval_ids),
            lgb_lambda=float(variant["lambda"]),
            lgb_emission_clip=float(variant["emission_clip"]),
        )
        hmm_mean = np.asarray(result["mean_eval"], dtype=np.float64)
        hmm_std = np.asarray(result["std_eval"], dtype=np.float64)
        loglik = float(result["loglik"])
        frame_data[f"{name}_lgb_tvt"] = lgb_tvt
        frame_data[f"{name}_mean_tvt"] = hmm_mean
        frame_data[f"{name}_mean_d"] = hmm_mean - last_tvt
        frame_data[f"{name}_std"] = hmm_std
        frame_data[f"{name}_loglik"] = loglik
        frame_data[f"{name}_finite"] = np.isfinite(hmm_mean).astype(np.float32)
        variant_meta.append(
            {
                "name": name,
                "source": variant.get("source"),
                "loglik": loglik,
                "grid_size": int(len(result["grid"])),
                "std_mean": float(np.mean(hmm_std)),
                "std_p90": float(np.quantile(hmm_std, 0.90)),
                "lgb_pred_min": float(np.min(lgb_tvt)),
                "lgb_pred_max": float(np.max(lgb_tvt)),
                "lgb_pred_mean": float(np.mean(lgb_tvt)),
            }
        )

    frame = pd.DataFrame(frame_data)
    numeric_columns = [column for column in frame.columns if column not in {"id", "well"}]
    finite = bool(np.isfinite(frame[numeric_columns].to_numpy(np.float64)).all())
    meta = {
        "well": well,
        "status": "ok" if finite else "non_finite",
        "rows": int(len(frame)),
        "elapsed_seconds": round(time.time() - started, 3),
        "variant_count": int(len(variant_meta)),
        "variant_metrics": variant_meta,
    }
    return _numeric_frame(frame), meta


def run_lgb_emission_hmm_inference(
    *,
    root: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    hmm_config: dict[str, Any],
    lgb_emission_config: dict[str, Any],
    output_prefix: str = OUTPUT_PREFIX,
    selected_candidate: str | None = None,
    strict_sample_ids: bool = True,
    submission_target_column: str = "tvt",
    max_wells: int | None = None,
    fast: bool = False,
    numba_num_threads: int | None = None,
    outer_workers: int = 1,
) -> dict[str, Any]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "numba is required to run the exact HMM smoother. "
            "Run the full inference notebook on Kaggle or install numba in the runtime."
        )

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    sample_submission_path = Path(sample_submission_path)
    if numba_num_threads:
        set_num_threads(int(numba_num_threads))
    effective_numba_num_threads = get_num_threads()

    lgb_variants, lgb_variant_summary = prepare_lgb_emission_variants(root, lgb_emission_config)
    if not lgb_variants:
        raise ValueError("run_lgb_emission_hmm_inference requires lgb_emission.enabled variants")
    if selected_candidate is None:
        selected_candidate = str(lgb_variants[0]["name"])

    data_dir = Path(data_dir)
    wells = list_well_ids(data_dir)
    if max_wells is None:
        env_max = int(os.environ.get("N_WELLS", "0") or "0")
        max_wells = env_max or None
    if fast and max_wells is None:
        max_wells = 3
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    if not wells:
        raise ValueError(f"No test wells found in {data_dir}")

    outer_workers = max(1, int(outer_workers or 1))

    def build_one(idx: int, well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(f"[{idx}/{len(wells)}] LGB-emission HMM inference well={well}", flush=True)
        frame, meta = build_hmm_inference_rows_for_well(
            well,
            data_dir,
            hmm_config,
            lgb_variants=lgb_variants,
        )
        meta["outer_workers"] = outer_workers
        print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
        return frame, meta

    if outer_workers > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise RuntimeError("joblib is required when inference outer_workers > 1") from exc
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

    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        raise ValueError("HMM inference produced no eval rows")
    inference_frame = pd.concat(non_empty_frames, ignore_index=True)
    if inference_frame["id"].duplicated().any():
        duplicated = int(inference_frame["id"].duplicated().sum())
        raise ValueError(f"HMM inference produced duplicated ids: {duplicated}")
    numeric_columns = [column for column in inference_frame.columns if column not in {"id", "well"}]
    if not np.isfinite(inference_frame[numeric_columns].to_numpy(np.float32)).all():
        raise ValueError("HMM inference feature frame contains non-finite numeric values")

    mean_col = f"{selected_candidate}_mean_tvt"
    delta_col = f"{selected_candidate}_mean_d"
    std_col = f"{selected_candidate}_std"
    loglik_col = f"{selected_candidate}_loglik"
    lgb_col = f"{selected_candidate}_lgb_tvt"
    missing_candidate_columns = [
        column
        for column in [mean_col, delta_col, std_col, loglik_col, lgb_col]
        if column not in inference_frame.columns
    ]
    if missing_candidate_columns:
        available = [str(variant["name"]) for variant in lgb_variants]
        raise ValueError(
            f"selected_candidate={selected_candidate!r} is not available. "
            f"missing={missing_candidate_columns}, available={available}"
        )

    predictions = pd.DataFrame(
        {
            "id": inference_frame["id"].astype(str).to_numpy(),
            "well": inference_frame["well"].astype(str).to_numpy(),
            "candidate": selected_candidate,
            "last_known_tvt": inference_frame["last_known_tvt"].to_numpy(np.float32),
            "md_since": inference_frame["md_since"].to_numpy(np.float32),
            "lgb_pred_tvt": inference_frame[lgb_col].to_numpy(np.float32),
            "pred_delta": inference_frame[delta_col].to_numpy(np.float32),
            "pred_tvt": inference_frame[mean_col].to_numpy(np.float32),
            "hmm_std": inference_frame[std_col].to_numpy(np.float32),
            "hmm_loglik": inference_frame[loglik_col].to_numpy(np.float32),
        }
    )

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    if sample["id"].duplicated().any():
        duplicated = int(sample["id"].duplicated().sum())
        raise ValueError(f"sample submission has duplicated ids: {duplicated}")
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    pred_map = predictions.set_index("id")["pred_tvt"]
    mapped = sample["id"].astype(str).map(pred_map)
    missing_mask = mapped.isna()
    missing_ids = sample.loc[missing_mask, "id"].astype(str).head(20).tolist()
    extra_ids = sorted(set(predictions["id"].astype(str)) - set(sample["id"].astype(str)))
    if strict_sample_ids and missing_ids:
        raise ValueError(
            f"HMM inference predictions do not cover sample_submission ids: "
            f"missing_count={int(missing_mask.sum())}, examples={missing_ids}"
        )
    fallback = float(predictions["pred_tvt"].mean())
    sample[target_column] = mapped.fillna(fallback).astype("float64")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(submission_path, index=False)

    feature_path = output_dir / f"{output_prefix}_{VARIANT}_inference_test_features.csv.gz"
    prediction_path = output_dir / f"{output_prefix}_inference_test_predictions.csv.gz"
    well_summary_path = output_dir / f"{output_prefix}_inference_by_well_generation_summary.csv"
    feature_schema_path = output_dir / f"{output_prefix}_inference_feature_schema.csv"
    metrics_path = output_dir / f"{output_prefix}_inference_metrics.csv"
    summary_path = output_dir / f"{output_prefix}_inference_summary.json"

    inference_frame.to_csv(feature_path, index=False, compression="gzip")
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    pd.DataFrame(well_meta).to_csv(well_summary_path, index=False)
    pd.DataFrame(
        {
            "variant": VARIANT,
            "feature_index": np.arange(len(numeric_columns), dtype=np.int32),
            "feature": numeric_columns,
        }
    ).to_csv(feature_schema_path, index=False)

    ok_meta = [row for row in well_meta if row.get("status") == "ok"]
    metrics = {
        "selected_candidate": selected_candidate,
        "test_rows": int(len(inference_frame)),
        "test_wells": int(inference_frame["well"].nunique()),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "extra_prediction_ids": int(len(extra_ids)),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "hmm_std_mean": float(predictions["hmm_std"].mean()),
        "hmm_std_p90": float(predictions["hmm_std"].quantile(0.90)),
        "lgb_pred_min": float(predictions["lgb_pred_tvt"].min()),
        "lgb_pred_max": float(predictions["lgb_pred_tvt"].max()),
        "lgb_pred_mean": float(predictions["lgb_pred_tvt"].mean()),
        "outer_workers": int(outer_workers),
        "numba_num_threads": (
            int(effective_numba_num_threads) if effective_numba_num_threads else None
        ),
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_completed",
        "mode": "lgb_emission_hmm_current_test_inference",
        "variant": VARIANT,
        "selected_candidate": selected_candidate,
        "rows": int(len(inference_frame)),
        "wells": int(inference_frame["well"].nunique()),
        "hmm_config": hmm_config,
        "lgb_emission": {
            "enabled": True,
            "variant_count": int(len(lgb_variants)),
            "variants": lgb_variant_summary,
        },
        "well_generation": {
            "selected_wells": int(len(wells)),
            "ok_wells": int(len(ok_meta)),
            "skipped_wells": int(len(well_meta) - len(ok_meta)),
            "mean_elapsed_seconds_per_ok_well": (
                float(np.mean([row["elapsed_seconds"] for row in ok_meta])) if ok_meta else None
            ),
        },
        "sample_submission": {
            "path": str(sample_submission_path),
            "target_column": target_column,
            "missing_prediction_ids": int(missing_mask.sum()),
            "missing_prediction_examples": missing_ids,
            "extra_prediction_ids": int(len(extra_ids)),
            "extra_prediction_examples": extra_ids[:20],
            "strict_sample_ids": bool(strict_sample_ids),
        },
        "metrics": metrics,
        "outputs": {
            "submission": str(submission_path),
            "inference_features": feature_path.name,
            "predictions": prediction_path.name,
            "by_well_generation_summary": well_summary_path.name,
            "feature_schema": feature_schema_path.name,
            "metrics": metrics_path.name,
            "summary": summary_path.name,
        },
        "sha256": {
            "submission": sha256_path(submission_path),
            "inference_features_gzip": sha256_path(feature_path),
            "inference_features_decompressed": sha256_gzip_decompressed(feature_path),
            "predictions_gzip": sha256_path(prediction_path),
            "predictions_decompressed": sha256_gzip_decompressed(prediction_path),
            "by_well_generation_summary": sha256_path(well_summary_path),
            "feature_schema": sha256_path(feature_schema_path),
            "metrics": sha256_path(metrics_path),
        },
        "notes": [
            "Current-test LightGBM predictions are used as Gaussian observation centers.",
            "Visible sample test rows are not used for model selection or score interpretation.",
            "submission.csv is ordered by sample_submission.csv and strict id coverage is enforced by default.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def run_train_feature_cache(
    *,
    root: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    hmm_config: dict[str, Any],
    lgb_emission_config: dict[str, Any] | None = None,
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
    lgb_variants, lgb_variant_summary = prepare_lgb_emission_variants(root, lgb_emission_config)
    feature_columns = lgb_variant_feature_columns(lgb_variants) if lgb_variants else FEATURE_COLUMNS

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
        print(f"[{idx}/{len(wells)}] LGB-emission HMM smoother well={well}", flush=True)
        frame, meta = build_hmm_rows_for_well(well, data_dir, hmm_config, lgb_variants=lgb_variants)
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
        "mode": "shrinkage_residual_scale_emission_hmm_train_feature_cache",
        "source_notebook": "amerhu/rogii-wellbore-geology-exact-hmm-smoother",
        "variant": VARIANT,
        "rows": int(len(train_frame)),
        "wells": int(train_frame["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "hmm_config": hmm_config,
        "lgb_emission": {
            "enabled": bool(lgb_variants),
            "variant_count": int(len(lgb_variants)),
            "variants": lgb_variant_summary,
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
            "Saved exp218 OOF predictions are immutable Gaussian observation centers.",
            "The selected stage uses either scalar sigma=20 or one predeclared variance-shrinkage alpha.",
            "Each row's true TVT residual is excluded from the model that produced its sigma; true TVT is not used to tune lambda.",
            "Unknown-suffix train TVT is used only for target and generation summary metrics.",
            "No test feature, model, inference, or submission output is generated.",
            "exp072 likpf_mean and LGB baselines are compared only by the separate direct comparison helper.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary
