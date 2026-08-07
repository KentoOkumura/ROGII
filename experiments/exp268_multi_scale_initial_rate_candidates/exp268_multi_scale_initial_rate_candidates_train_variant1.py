# %% [markdown]
# # exp268 multi-scale initial-rate candidates
#
# Exact-HMM candidate-bank audit. The canonical `train` notebook aggregates two
# target-free well shards. `train_variant0` and `train_variant1` generate the
# same four predeclared candidates on disjoint well sets.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Known-prefix rate and HMM input helpers
# 4. Exact HMM forward-backward kernel
# 5. Multi-scale shard generation
# 6. Candidate-bank audit helpers
# 7. Setup and execution contract
# 8. Input preflight
# 9. Generate a shard or aggregate the candidate bank
# 10. Metrics and generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from itertools import combinations
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


EXPERIMENT_NAME = "exp268_multi_scale_initial_rate_candidates"
OUTPUT_PREFIX = EXPERIMENT_NAME
RUN_KIND_OVERRIDE = "shard1"
WINDOWS = (32, 64, 128, 256)
WINDOW_CANDIDATES = {window: f"hmm_ir_w{window}" for window in WINDOWS}
CONTROL_CANDIDATE = "hmm_ir_tail30"
LIKPF_REFERENCE = "exp072_likpf_mean"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers

# %%


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
    candidates = [
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp268 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        slug = "rogii-wellbore-geology-prediction"
        direct_candidates = (
            KAGGLE_INPUT_ROOT / slug / "train",
            KAGGLE_INPUT_ROOT / "competitions" / slug / "train",
        )
        for direct in direct_candidates:
            if direct.exists() and list(direct.glob("*__horizontal_well.csv")):
                return direct
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if list(candidate.glob("*__horizontal_well.csv")):
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


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


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_well_shard(well: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    key = f"exp268::well_shard::{well}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % shard_count


def resolve_existing(filename: str, candidates: list[str]) -> Path:
    checked: list[str] = []
    root = project_root()
    for raw in candidates:
        candidate = Path(raw)
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename)):
            checked.append(str(path))
            if path.exists() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"Could not resolve {filename}; checked={checked}")


def require_decompressed_sha(path: Path, expected: str | None) -> str:
    actual = sha256_gzip_decompressed(path)
    if expected and actual != expected:
        raise ValueError(
            f"decompressed SHA mismatch for {path}: expected={expected} actual={actual}"
        )
    return actual


# %% [markdown]
# ## 3. Known-prefix rate and HMM input helpers

# %%


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


def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int]:
    """Median surface increment per MD from a fixed known-prefix tail window."""
    if window_rows <= 1:
        raise ValueError("window_rows must be greater than one")
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    valid_steps = int(valid.sum())
    if valid_steps < int(min_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    if not np.isfinite(rate):
        rate = float(fallback_rate)
    return rate, int(len(tail)), valid_steps


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    tail_n: int = 30,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, float, float, float, int, int]:
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

    init_rate, effective_rows, valid_steps = robust_initial_rate(
        known,
        tail_n,
        min_valid_steps=min_valid_steps,
        fallback_rate=fallback_rate,
    )
    return (
        float(cal_a),
        float(cal_b),
        sigma,
        init_rate,
        effective_rows,
        valid_steps,
    )


# %% [markdown]
# ## 4. Exact HMM forward-backward kernel
#
# The numerical kernel and fixed grammar follow exp209. Only `r0`, computed from
# a predeclared known-prefix window, changes between the four new candidates.

# %%


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
    initial_rate_window_rows: int = 30,
    initial_rate_min_valid_steps: int = 3,
    initial_rate_fallback: float = 0.0,
    return_post: bool = False,
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

    (
        cal_a,
        cal_b,
        robust_sigma,
        init_rate,
        effective_rate_rows,
        valid_rate_steps,
    ) = prefix_stats(
        horizontal,
        typewell_tvt,
        typewell_gr,
        tail_n=initial_rate_window_rows,
        min_valid_steps=initial_rate_min_valid_steps,
        fallback_rate=initial_rate_fallback,
    )
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
        "initial_rate_window_rows": int(initial_rate_window_rows),
        "initial_rate_effective_rows": int(effective_rate_rows),
        "initial_rate_valid_steps": int(valid_rate_steps),
        "cal_a": cal_a,
        "cal_b": cal_b,
    }
    if return_post:
        result["post"] = post_p
        result["md_eval"] = md
    return result


# %% [markdown]
# ## 5. Multi-scale shard generation
#
# Each shard notebook executes every predeclared window on its target-free well
# subset. The generator passes a frame without the `TVT` target column into the
# HMM and attaches true TVT only after all four paths are frozen.

# %%
def fixed_hmm_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.hmm") or {}
    keys = (
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
    )
    missing = [key for key in keys if key not in hmm]
    if missing:
        raise ValueError(f"model.hmm is missing fixed keys: {missing}")
    return {key: hmm[key] for key in keys}


def validate_scientific_contract(config: dict[str, Any]) -> None:
    windows = tuple(int(value) for value in get_nested(config, "model.initial_rate.windows") or [])
    candidates = tuple(
        str(value) for value in get_nested(config, "model.initial_rate.candidate_columns") or []
    )
    if windows != WINDOWS:
        raise ValueError(f"exp268 windows must be exactly {WINDOWS}; received={windows}")
    expected_candidates = tuple(WINDOW_CANDIDATES[window] for window in WINDOWS)
    if candidates != expected_candidates:
        raise ValueError(
            f"candidate columns must be {expected_candidates}; received={candidates}"
        )
    if bool(get_nested(config, "model.control.regenerate")):
        raise ValueError("saved exp209 tail-30 control must not be regenerated")
    if bool(get_nested(config, "audit.persist_oracle_predictions")):
        raise ValueError("oracle predictions must remain transient diagnostics")
    if bool(get_nested(config, "audit.persist_candidate_mean")):
        raise ValueError("candidate mean is forbidden in exp268")
    if bool(get_nested(config, "audit.persist_selector")):
        raise ValueError("selector fitting or persistence is forbidden in exp268")
    if int(get_nested(config, "execution.lightgbm_config_count") or 0) != 0:
        raise ValueError("LightGBM configs are forbidden in exp268")
    if int(get_nested(config, "execution.total_boosters") or 0) != 0:
        raise ValueError("boosters are forbidden in exp268")
    if bool(get_nested(config, "execution.control_or_parent_retraining")):
        raise ValueError("parent/control regeneration is forbidden in exp268")
    if bool(get_nested(config, "execution.gpu")):
        raise ValueError("exp268 is CPU-only")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("inference and submission are disabled in exp268")


def build_multiscale_rows_for_well(
    well: str,
    data_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizontal_path = data_dir / f"{well}__horizontal_well.csv"
    typewell_path = data_dir / f"{well}__typewell.csv"
    horizontal, typewell = load_well(well, data_dir)
    required_horizontal = {"MD", "Z", "GR", "TVT_input", "TVT"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        missing = sorted(required_horizontal - set(horizontal.columns))
        raise ValueError(f"{horizontal_path} missing columns: {missing}")
    if not required_typewell.issubset(typewell.columns):
        missing = sorted(required_typewell - set(typewell.columns))
        raise ValueError(f"{typewell_path} missing columns: {missing}")

    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if int(eval_mask.sum()) == 0:
        return pd.DataFrame(), {"well": well, "status": "skipped_no_eval_rows", "rows": 0}
    known = horizontal.loc[known_mask]
    if len(known) < 4:
        return pd.DataFrame(), {"well": well, "status": "skipped_short_prefix", "rows": 0}

    # The HMM never receives the evaluation target column.
    generation_horizontal = horizontal.drop(columns=["TVT"]).copy()
    hmm_kwargs = fixed_hmm_kwargs(config)
    initial_rate = get_nested(config, "model.initial_rate") or {}
    min_valid_steps = int(initial_rate.get("min_valid_steps", 3))
    fallback_rate = float(initial_rate.get("fallback_rate", 0.0))
    started = time.time()
    results: dict[int, dict[str, Any]] = {}
    for window in WINDOWS:
        results[window] = run_hmm2(
            generation_horizontal,
            typewell,
            **hmm_kwargs,
            initial_rate_window_rows=window,
            initial_rate_min_valid_steps=min_valid_steps,
            initial_rate_fallback=fallback_rate,
        )

    eval_index = np.asarray(results[WINDOWS[0]]["ev_index"], dtype=np.int64)
    for window in WINDOWS[1:]:
        if not np.array_equal(eval_index, np.asarray(results[window]["ev_index"], dtype=np.int64)):
            raise RuntimeError(f"candidate eval-index mismatch for well={well} window={window}")

    # Candidate paths are frozen above. Target attachment starts here.
    true_tvt = pd.to_numeric(horizontal.loc[eval_index, "TVT"], errors="coerce").to_numpy(
        np.float64
    )
    last_known = known.iloc[-1]
    last_known_tvt = float(last_known["TVT_input"])
    last_known_md = float(last_known["MD"])
    md_since = (
        pd.to_numeric(horizontal.loc[eval_index, "MD"], errors="coerce").to_numpy(np.float64)
        - last_known_md
    )
    if not np.isfinite(true_tvt).all():
        raise ValueError(f"non-finite evaluation target for well={well}")

    payload: dict[str, Any] = {
        "id": [f"{well}_{int(row_index)}" for row_index in eval_index],
        "well": str(well),
        "row_idx": eval_index,
        "true_tvt": true_tvt.astype(np.float32),
        "last_known_tvt": np.float32(last_known_tvt),
        "md_since": md_since.astype(np.float32),
        "prefix_rows": np.int32(len(known)),
    }
    meta: dict[str, Any] = {
        "well": str(well),
        "status": "ok",
        "rows": int(len(eval_index)),
        "prefix_rows": int(len(known)),
        "horizontal_sha256": sha256_path(horizontal_path),
        "typewell_sha256": sha256_path(typewell_path),
    }
    for window in WINDOWS:
        candidate = WINDOW_CANDIDATES[window]
        result = results[window]
        mean = np.asarray(result["mean_eval"], dtype=np.float64)
        std = np.asarray(result["std_eval"], dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"non-finite HMM output for well={well} window={window}")
        payload[candidate] = mean.astype(np.float32)
        payload[f"{candidate}_std"] = std.astype(np.float32)
        payload[f"initial_rate_w{window}"] = np.float32(result["prefix_ir"])
        meta[f"initial_rate_w{window}"] = float(result["prefix_ir"])
        meta[f"effective_rate_rows_w{window}"] = int(result["initial_rate_effective_rows"])
        meta[f"valid_rate_steps_w{window}"] = int(result["initial_rate_valid_steps"])
        meta[f"loglik_w{window}"] = float(result["loglik"])
        meta[f"rmse_w{window}"] = rmse(true_tvt, mean)
    frame = pd.DataFrame(payload)
    numeric = frame.drop(columns=["id", "well"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"shard rows contain non-finite values for well={well}")
    meta["elapsed_seconds"] = float(time.time() - started)
    return frame, meta


def run_shard_generation(config: dict[str, Any], shard_index: int) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp268 shard generation must run on Kaggle. Set EXPERIMENT_ALLOW_LOCAL=1 "
            "only for an explicitly approved local smoke run."
        )
    validate_scientific_contract(config)
    shard_count = int(get_nested(config, "execution.shard_count") or 0)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid shard index {shard_index} for shard_count={shard_count}")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exact-HMM shard generation")
    requested_threads = int(get_nested(config, "execution.numba_num_threads") or 1)
    set_num_threads(requested_threads)
    effective_threads = get_num_threads()
    data_dir = train_data_dir(config)
    wells = list_well_ids(data_dir)
    selected = [well for well in wells if stable_well_shard(well, shard_count) == shard_index]
    configured_max = get_nested(config, "execution.max_wells_per_shard")
    env_max = int(os.environ.get("EXPERIMENT_MAX_WELLS", "0") or "0")
    max_wells = env_max or (int(configured_max) if configured_max is not None else None)
    if max_wells is not None:
        selected = selected[:max_wells]
    if not selected:
        raise ValueError(f"no wells selected for shard {shard_index}")

    outer_workers = int(get_nested(config, "execution.outer_workers") or 1)
    started = time.time()

    def build(index: int, well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(
            f"[shard {shard_index}] {index}/{len(selected)} well={well}",
            flush=True,
        )
        frame, meta = build_multiscale_rows_for_well(well, data_dir, config)
        print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
        return frame, meta

    if outer_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(build)(index, well) for index, well in enumerate(selected, start=1)
        )
    else:
        results = [build(index, well) for index, well in enumerate(selected, start=1)]
    frames = [frame for frame, _ in results if not frame.empty]
    well_rows = [meta for _, meta in results]
    if not frames:
        raise RuntimeError(f"shard {shard_index} produced no candidate rows")
    output = pd.concat(frames, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    )
    if output["id"].duplicated().any():
        raise RuntimeError(f"shard {shard_index} contains duplicate ids")

    artifacts = artifact_dir()
    output_path = artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}_schema.csv"
    well_path = artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}_by_well.csv"
    input_path = artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}_input_manifest.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}_summary.json"
    output.to_csv(output_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "column_index": np.arange(len(output.columns), dtype=np.int32),
            "column": output.columns,
            "dtype": [str(output[column].dtype) for column in output.columns],
        }
    ).to_csv(schema_path, index=False)
    well_frame = pd.DataFrame(well_rows).sort_values("well")
    well_frame.to_csv(well_path, index=False)
    well_frame[
        ["well", "horizontal_sha256", "typewell_sha256"]
    ].to_csv(input_path, index=False)
    candidate_matrix = output[[WINDOW_CANDIDATES[window] for window in WINDOWS]].to_numpy(
        np.float32
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "shard_generation_completed",
        "run_kind": f"shard{shard_index}",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_policy": get_nested(config, "execution.shard_policy"),
        "selected_wells": len(selected),
        "ok_wells": int((well_frame["status"] == "ok").sum()),
        "rows": int(len(output)),
        "candidate_windows": WINDOWS,
        "candidate_columns": [WINDOW_CANDIDATES[window] for window in WINDOWS],
        "outer_workers": outer_workers,
        "numba_threads_requested": requested_threads,
        "numba_threads_effective": effective_threads,
        "lightgbm_configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent_control_regeneration": False,
        "elapsed_seconds": float(time.time() - started),
        "prediction_content_sha256": array_bundle_sha256(
            row_idx=output["row_idx"].to_numpy(np.int64),
            candidates=candidate_matrix,
        ),
        "artifacts": {
            "rows": str(output_path),
            "schema": str(schema_path),
            "by_well": str(well_path),
            "input_manifest": str(input_path),
            "summary": str(summary_path),
        },
        "sha256": {
            "rows_raw_gzip": sha256_path(output_path),
            "rows_decompressed": sha256_gzip_decompressed(output_path),
            "schema": sha256_path(schema_path),
            "by_well": sha256_path(well_path),
            "input_manifest": sha256_path(input_path),
        },
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    metrics_path = artifacts.parent / "metrics.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "shard_generation_completed",
            "run_kind": f"shard{shard_index}",
            "rows": len(output),
            "wells": int(output["well"].nunique()),
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "prediction_content_sha256": summary["prediction_content_sha256"],
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 6. Candidate-bank audit helpers
#
# Oracle paths are constructed only inside metric functions, scored, and then
# discarded. They are never added to the persisted candidate frame.

# %%
def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def score_prediction(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not valid.all() or not valid.any():
        raise ValueError("metric input contains non-finite rows")
    error = prediction - truth
    return {
        "rows": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def distance_bucket(values: pd.Series) -> np.ndarray:
    md = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    labels = np.full(len(md), "1000_plus", dtype=object)
    labels[md < 1000.0] = "500_1000"
    labels[md < 500.0] = "250_500"
    labels[md < 250.0] = "100_250"
    labels[md < 100.0] = "050_100"
    labels[md < 50.0] = "000_050"
    return labels


def prefix_length_bucket(values: pd.Series) -> np.ndarray:
    rows = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    labels = np.full(len(rows), "256_plus", dtype=object)
    labels[rows < 256] = "128_255"
    labels[rows < 128] = "064_127"
    labels[rows < 64] = "000_063"
    return labels


def oracle_prediction(
    frame: pd.DataFrame,
    candidates: list[str],
    scope: str,
    block_rows: int | None = None,
) -> np.ndarray:
    output = np.full(len(frame), np.nan, dtype=np.float32)
    for _, group in frame.groupby("well", sort=False):
        ordered = group.sort_values("row_idx")
        positions = ordered.index.to_numpy(np.int64)
        truth = numeric_array(ordered, "true_tvt")
        values = ordered[candidates].to_numpy(np.float64)
        if scope == "row":
            winners = np.argmin(np.abs(values - truth[:, None]), axis=1)
            selected = values[np.arange(len(values)), winners]
        else:
            selected = np.empty(len(values), dtype=np.float64)
            chunk_size = len(values) if scope == "whole_well" else int(block_rows or 0)
            if chunk_size <= 0:
                raise ValueError(f"invalid oracle scope={scope} block_rows={block_rows}")
            for start in range(0, len(values), chunk_size):
                stop = min(start + chunk_size, len(values))
                loss = (values[start:stop] - truth[start:stop, None]) ** 2
                winner = int(np.argmin(np.mean(loss, axis=0)))
                selected[start:stop] = values[start:stop, winner]
        output[positions] = selected.astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError(f"oracle output contains non-finite values for scope={scope}")
    return output


def compute_oracle_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    banks = get_nested(config, "candidate_bank.oracle_banks") or {}
    block_sizes = [int(value) for value in get_nested(config, "audit.oracle_block_rows") or []]
    scopes: list[tuple[str, str, int | None]] = [("row", "row", None)]
    scopes.extend((f"block_{size}", "block", size) for size in block_sizes)
    scopes.append(("whole_well", "whole_well", None))
    truth = numeric_array(frame, "true_tvt")
    rows: list[dict[str, Any]] = []
    for bank, raw_candidates in banks.items():
        candidates = [str(value) for value in raw_candidates]
        missing = [candidate for candidate in candidates if candidate not in frame.columns]
        if missing:
            raise ValueError(f"oracle bank {bank} missing candidates: {missing}")
        for scope_name, scope, block_size in scopes:
            prediction = oracle_prediction(frame, candidates, scope, block_size)
            rows.append(
                {
                    "bank": str(bank),
                    "scope": scope_name,
                    "candidate_count": len(candidates),
                    **score_prediction(prediction, truth),
                }
            )
            del prediction
    metrics = pd.DataFrame(rows)
    control = metrics.loc[
        metrics["bank"] == "tail30_control", ["scope", "rmse"]
    ].rename(columns={"rmse": "tail30_control_oracle_rmse"})
    metrics = metrics.merge(control, on="scope", how="left", validate="many_to_one")
    metrics["delta_rmse_vs_tail30_control"] = (
        metrics["rmse"] - metrics["tail30_control_oracle_rmse"]
    )
    return metrics


def compute_unique_best(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    truth = numeric_array(frame, "true_tvt")
    rows: list[dict[str, Any]] = []
    banks = get_nested(config, "candidate_bank.oracle_banks") or {}
    for bank in ("initial_rate_5", "likpf_plus_initial_rate_5"):
        candidates = [str(value) for value in banks[bank]]
        values = frame[candidates].to_numpy(np.float64)
        errors = np.abs(values - truth[:, None])
        minimum = np.min(errors, axis=1)
        tie_count = np.sum(
            np.isclose(errors, minimum[:, None], rtol=0.0, atol=1.0e-6), axis=1
        )
        winner = np.argmin(errors, axis=1)
        for index, candidate in enumerate(candidates):
            best = winner == index
            unique = best & (tie_count == 1)
            rows.append(
                {
                    "bank": bank,
                    "scope": "row",
                    "candidate": candidate,
                    "best_rows": int(best.sum()),
                    "best_rate": float(np.mean(best)),
                    "unique_best_rows": int(unique.sum()),
                    "unique_best_rate": float(np.mean(unique)),
                }
            )
        for candidate in candidates:
            whole_best = 0
            whole_unique = 0
            for _, group in frame.groupby("well", sort=False):
                group_truth = numeric_array(group, "true_tvt")
                losses = np.mean(
                    (group[candidates].to_numpy(np.float64) - group_truth[:, None]) ** 2,
                    axis=0,
                )
                best_loss = float(np.min(losses))
                tied = np.isclose(losses, best_loss, rtol=0.0, atol=1.0e-10)
                candidate_index = candidates.index(candidate)
                if tied[candidate_index]:
                    whole_best += 1
                    whole_unique += int(tied.sum() == 1)
            rows.append(
                {
                    "bank": bank,
                    "scope": "whole_well",
                    "candidate": candidate,
                    "best_rows": whole_best,
                    "best_rate": whole_best / frame["well"].nunique(),
                    "unique_best_rows": whole_unique,
                    "unique_best_rate": whole_unique / frame["well"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def compute_rate_and_duplicate_diagnostics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate_columns = [f"initial_rate_w{window}" for window in WINDOWS]
    candidate_columns = [WINDOW_CANDIDATES[window] for window in WINDOWS]
    rate_atol = float(get_nested(config, "audit.rate_duplicate_atol") or 0.0)
    path_atol = float(get_nested(config, "audit.path_duplicate_atol_ft") or 0.0)
    rate_rows: list[dict[str, Any]] = []
    pair_accumulator: dict[tuple[str, str], dict[str, Any]] = {
        pair: {
            "wells": 0,
            "rate_duplicate_wells": 0,
            "path_duplicate_wells": 0,
            "max_path_abs_diff": 0.0,
            "mean_path_abs_diff_sum": 0.0,
        }
        for pair in combinations(candidate_columns, 2)
    }
    for well, group in frame.groupby("well", sort=False):
        rates = np.asarray([float(group[column].iloc[0]) for column in rate_columns])
        rate_row: dict[str, Any] = {
            "well": str(well),
            "prefix_rows": int(group["prefix_rows"].iloc[0]),
            "rate_min": float(np.min(rates)),
            "rate_max": float(np.max(rates)),
            "rate_spread": float(np.max(rates) - np.min(rates)),
            "unique_rate_count": int(
                len({round(float(value) / max(rate_atol, 1.0e-15)) for value in rates})
            ),
        }
        for window, value in zip(WINDOWS, rates, strict=True):
            rate_row[f"initial_rate_w{window}"] = float(value)
        rate_rows.append(rate_row)
        rate_map = {
            WINDOW_CANDIDATES[window]: float(group[f"initial_rate_w{window}"].iloc[0])
            for window in WINDOWS
        }
        for left, right in combinations(candidate_columns, 2):
            accumulator = pair_accumulator[(left, right)]
            difference = np.abs(numeric_array(group, left) - numeric_array(group, right))
            accumulator["wells"] += 1
            accumulator["rate_duplicate_wells"] += int(
                abs(rate_map[left] - rate_map[right]) <= rate_atol
            )
            accumulator["path_duplicate_wells"] += int(float(np.max(difference)) <= path_atol)
            accumulator["max_path_abs_diff"] = max(
                float(accumulator["max_path_abs_diff"]), float(np.max(difference))
            )
            accumulator["mean_path_abs_diff_sum"] += float(np.mean(difference))
    duplicate_rows: list[dict[str, Any]] = []
    for (left, right), accumulator in pair_accumulator.items():
        wells = int(accumulator["wells"])
        duplicate_rows.append(
            {
                "candidate_left": left,
                "candidate_right": right,
                "wells": wells,
                "rate_duplicate_wells": int(accumulator["rate_duplicate_wells"]),
                "rate_duplicate_rate": accumulator["rate_duplicate_wells"] / wells,
                "path_duplicate_wells": int(accumulator["path_duplicate_wells"]),
                "path_duplicate_rate": accumulator["path_duplicate_wells"] / wells,
                "mean_well_path_abs_diff": accumulator["mean_path_abs_diff_sum"] / wells,
                "max_path_abs_diff": accumulator["max_path_abs_diff"],
            }
        )
    return pd.DataFrame(rate_rows), pd.DataFrame(duplicate_rows)


def strict_same_ids(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    if len(reference) != len(other):
        raise ValueError(f"{label} row mismatch: reference={len(reference)} other={len(other)}")
    reference_ids = reference["id"].astype(str).to_numpy()
    other_ids = other["id"].astype(str).to_numpy()
    if np.array_equal(reference_ids, other_ids):
        return other.reset_index(drop=True)
    other_indexed = other.set_index(other["id"].astype(str), drop=False)
    if other_indexed.index.duplicated().any():
        raise ValueError(f"{label} contains duplicate ids")
    missing = np.setdiff1d(reference_ids, other_indexed.index.to_numpy(), assume_unique=False)
    extra = np.setdiff1d(other_indexed.index.to_numpy(), reference_ids, assume_unique=False)
    if len(missing) or len(extra):
        raise ValueError(f"{label} id mismatch: missing={len(missing)} extra={len(extra)}")
    return other_indexed.loc[reference_ids].reset_index(drop=True)


def load_shards(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for spec in get_nested(config, "data.shard_outputs") or []:
        path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
        frame = pd.read_csv(path, dtype={"id": str, "well": str})
        frames.append(frame)
        manifest.append(
            {
                "role": f"shard{int(spec['shard_index'])}",
                "path": str(path),
                "bytes": path.stat().st_size,
                "raw_sha256": sha256_path(path),
                "decompressed_sha256": sha256_gzip_decompressed(path),
                "rows": len(frame),
                "wells": frame["well"].nunique(),
            }
        )
    if len(frames) != int(get_nested(config, "execution.shard_count") or 0):
        raise ValueError("all configured shards must be present")
    combined = pd.concat(frames, ignore_index=True)
    if combined["id"].duplicated().any():
        raise ValueError("shard union contains duplicate ids")
    combined = combined.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    return combined, manifest


def compute_direct_metrics(
    frame: pd.DataFrame, candidates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = numeric_array(frame, "true_tvt")
    control = numeric_array(frame, CONTROL_CANDIDATE)
    likpf = numeric_array(frame, LIKPF_REFERENCE)
    overall_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    distance = distance_bucket(frame["md_since"])
    prefix = prefix_length_bucket(frame["prefix_rows"])
    control_rmse = rmse(truth, control)
    likpf_rmse = rmse(truth, likpf)
    for candidate in candidates:
        prediction = numeric_array(frame, candidate)
        row = {"candidate": candidate, **score_prediction(prediction, truth)}
        row["delta_rmse_vs_tail30"] = row["rmse"] - control_rmse
        row["delta_rmse_vs_exp072_likpf_mean"] = row["rmse"] - likpf_rmse
        overall_rows.append(row)
        for bucket in (
            "000_050",
            "050_100",
            "100_250",
            "250_500",
            "500_1000",
            "1000_plus",
        ):
            mask = distance == bucket
            if mask.any():
                distance_rows.append(
                    {
                        "candidate": candidate,
                        "bucket": bucket,
                        **score_prediction(prediction[mask], truth[mask]),
                        "tail30_rmse": rmse(truth[mask], control[mask]),
                    }
                )
        for bucket in ("000_063", "064_127", "128_255", "256_plus"):
            mask = prefix == bucket
            if mask.any():
                prefix_rows.append(
                    {
                        "candidate": candidate,
                        "prefix_bucket": bucket,
                        **score_prediction(prediction[mask], truth[mask]),
                        "tail30_rmse": rmse(truth[mask], control[mask]),
                    }
                )
        for well, group in frame.groupby("well", sort=False):
            group_truth = numeric_array(group, "true_tvt")
            group_prediction = numeric_array(group, candidate)
            group_control = numeric_array(group, CONTROL_CANDIDATE)
            metric = score_prediction(group_prediction, group_truth)
            by_well_rows.append(
                {
                    "candidate": candidate,
                    "well": str(well),
                    **metric,
                    "tail30_rmse": rmse(group_truth, group_control),
                    "delta_rmse_vs_tail30": metric["rmse"] - rmse(group_truth, group_control),
                }
            )
    overall = pd.DataFrame(overall_rows)
    by_well = pd.DataFrame(by_well_rows)
    worst = (
        by_well.groupby("candidate", as_index=False)["delta_rmse_vs_tail30"]
        .max()
        .rename(columns={"delta_rmse_vs_tail30": "max_well_regression_vs_tail30"})
    )
    overall = overall.merge(worst, on="candidate", how="left", validate="one_to_one")
    return overall, pd.DataFrame(distance_rows), pd.DataFrame(prefix_rows), by_well


def compute_hidden_like_metrics(
    frame: pd.DataFrame, candidates: list[str], config: dict[str, Any]
) -> tuple[pd.DataFrame, Path]:
    hidden = get_nested(config, "data.hidden_like") or {}
    if not bool(hidden.get("enabled", False)):
        return pd.DataFrame(), Path()
    filename = Path(str(hidden["fold_assignment_candidates"][0])).name
    path = resolve_existing(filename, [str(value) for value in hidden["fold_assignment_candidates"]])
    assignments = pd.read_csv(path, dtype={"well_id": str})
    truth = numeric_array(frame, "true_tvt")
    rows: list[dict[str, Any]] = []
    for subgroup, role_column in (hidden.get("valid_role_columns") or {}).items():
        if role_column not in assignments.columns:
            raise ValueError(f"hidden-like assignment missing role column {role_column}")
        wells = set(
            assignments.loc[
                assignments[role_column].astype(str) == "valid", "well_id"
            ].astype(str)
        )
        mask = frame["well"].astype(str).isin(wells).to_numpy()
        if not mask.any():
            raise ValueError(f"hidden-like subgroup {subgroup} selected zero rows")
        for candidate in candidates:
            rows.append(
                {
                    "subgroup": str(subgroup),
                    "candidate": candidate,
                    **score_prediction(numeric_array(frame, candidate)[mask], truth[mask]),
                }
            )
    return pd.DataFrame(rows), path


def run_aggregate(config: dict[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    started = time.time()
    shards, input_manifest = load_shards(config)
    expected_rows = int(get_nested(config, "validation.expected_rows") or 0)
    expected_wells = int(get_nested(config, "validation.expected_wells") or 0)
    if len(shards) != expected_rows or shards["well"].nunique() != expected_wells:
        raise ValueError(
            f"shard coverage mismatch: rows={len(shards)}/{expected_rows} "
            f"wells={shards['well'].nunique()}/{expected_wells}"
        )

    control_spec = get_nested(config, "data.exp209_hmm_control") or {}
    control_path = resolve_existing(
        str(control_spec["filename"]), [str(value) for value in control_spec["candidates"]]
    )
    control_sha = require_decompressed_sha(
        control_path, str(control_spec.get("expected_decompressed_sha256") or "")
    )
    control = pd.read_csv(
        control_path,
        usecols=[
            "id",
            "well",
            "target",
            "last_known_tvt",
            "md_since",
            "hmm_mean_tvt",
            "hmm_prefix_ir",
        ],
        dtype={"id": str, "well": str},
    ).sort_values(["well", "id"], kind="mergesort").reset_index(drop=True)
    # Sort the shard by the same stable key before strict alignment.
    shards = shards.sort_values(["well", "id"], kind="mergesort").reset_index(drop=True)
    shards = strict_same_ids(control, shards, "exp268 shard union")

    exp072_spec = get_nested(config, "data.exp072_cache") or {}
    exp072_path = resolve_existing(
        str(exp072_spec["filename"]), [str(value) for value in exp072_spec["candidates"]]
    )
    exp072_sha = require_decompressed_sha(
        exp072_path, str(exp072_spec.get("expected_decompressed_sha256") or "")
    )
    exp072 = pd.read_csv(
        exp072_path,
        usecols=["id", "well", "target", "last_known_tvt", "md_since", "likpf_mean_d"],
        dtype={"id": str, "well": str},
    ).sort_values(["well", "id"], kind="mergesort").reset_index(drop=True)
    exp072 = strict_same_ids(control, exp072, "exp072 reference")

    true_tvt = numeric_array(control, "last_known_tvt") + numeric_array(control, "target")
    if np.max(np.abs(true_tvt - numeric_array(shards, "true_tvt"))) > 1.0e-3:
        raise ValueError("shard true TVT does not match saved exp209 target surface")
    if not np.array_equal(control["well"].astype(str), shards["well"].astype(str)):
        raise ValueError("shard well labels do not match exp209 control")

    frame = pd.DataFrame(
        {
            "id": control["id"].astype(str),
            "well": control["well"].astype(str),
            "row_idx": pd.to_numeric(shards["row_idx"], errors="raise").astype(np.int64),
            "true_tvt": true_tvt.astype(np.float64),
            "last_known_tvt": numeric_array(control, "last_known_tvt"),
            "md_since": numeric_array(control, "md_since"),
            "prefix_rows": pd.to_numeric(shards["prefix_rows"], errors="raise").astype(np.int32),
            LIKPF_REFERENCE: numeric_array(exp072, "last_known_tvt")
            + numeric_array(exp072, "likpf_mean_d"),
            CONTROL_CANDIDATE: numeric_array(control, "hmm_mean_tvt"),
            "initial_rate_tail30": numeric_array(control, "hmm_prefix_ir"),
        }
    )
    for window in WINDOWS:
        candidate = WINDOW_CANDIDATES[window]
        frame[candidate] = numeric_array(shards, candidate)
        frame[f"initial_rate_w{window}"] = numeric_array(shards, f"initial_rate_w{window}")
    direct_candidates = [
        str(value) for value in get_nested(config, "candidate_bank.direct_candidates") or []
    ]
    forbidden = [
        column
        for column in frame.columns
        if "oracle" in column.lower()
        or "candidate_mean" in column.lower()
        or "selector" in column.lower()
    ]
    if forbidden:
        raise RuntimeError(f"forbidden deployable columns found: {forbidden}")

    overall, distance, prefix, by_well = compute_direct_metrics(frame, direct_candidates)
    hidden, hidden_path = compute_hidden_like_metrics(frame, direct_candidates, config)
    rate_diagnostics, duplicates = compute_rate_and_duplicate_diagnostics(frame, config)
    unique_best = compute_unique_best(frame, config)
    oracle_metrics = compute_oracle_metrics(frame, config)

    artifacts = artifact_dir()
    paths = {
        "candidate_metrics": artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
        "distance_bucket_metrics": artifacts / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv",
        "prefix_length_metrics": artifacts / f"{OUTPUT_PREFIX}_prefix_length_metrics.csv",
        "hidden_like_metrics": artifacts / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_by_well.csv",
        "rate_diagnostics": artifacts / f"{OUTPUT_PREFIX}_rate_diagnostics.csv",
        "duplicate_diagnostics": artifacts / f"{OUTPUT_PREFIX}_duplicate_diagnostics.csv",
        "unique_best": artifacts / f"{OUTPUT_PREFIX}_unique_best.csv",
        "oracle_scope_metrics": artifacts / f"{OUTPUT_PREFIX}_oracle_scope_metrics.csv",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "summary": artifacts / f"{OUTPUT_PREFIX}_summary.json",
    }
    overall.sort_values("candidate").to_csv(paths["candidate_metrics"], index=False)
    distance.to_csv(paths["distance_bucket_metrics"], index=False)
    prefix.to_csv(paths["prefix_length_metrics"], index=False)
    hidden.to_csv(paths["hidden_like_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    rate_diagnostics.to_csv(paths["rate_diagnostics"], index=False)
    duplicates.to_csv(paths["duplicate_diagnostics"], index=False)
    unique_best.to_csv(paths["unique_best"], index=False)
    oracle_metrics.to_csv(paths["oracle_scope_metrics"], index=False)
    input_manifest.extend(
        [
            {
                "role": "exp209_tail30_control",
                "path": str(control_path),
                "bytes": control_path.stat().st_size,
                "raw_sha256": sha256_path(control_path),
                "decompressed_sha256": control_sha,
                "rows": len(control),
                "wells": control["well"].nunique(),
            },
            {
                "role": "exp072_likpf_reference",
                "path": str(exp072_path),
                "bytes": exp072_path.stat().st_size,
                "raw_sha256": sha256_path(exp072_path),
                "decompressed_sha256": exp072_sha,
                "rows": len(exp072),
                "wells": exp072["well"].nunique(),
            },
            {
                "role": "hidden_like_assignments",
                "path": str(hidden_path),
                "bytes": hidden_path.stat().st_size,
                "raw_sha256": sha256_path(hidden_path),
                "decompressed_sha256": None,
                "rows": None,
                "wells": None,
            },
        ]
    )
    pd.DataFrame(input_manifest).to_csv(paths["input_manifest"], index=False)

    candidate_matrix = frame[direct_candidates].to_numpy(np.float32)
    prediction_sha = array_bundle_sha256(
        row_idx=frame["row_idx"].to_numpy(np.int64), candidates=candidate_matrix
    )
    tail30_row = overall.loc[overall["candidate"] == CONTROL_CANDIDATE].iloc[0].to_dict()
    best_diagnostic = overall.sort_values(["rmse", "candidate"]).iloc[0].to_dict()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_candidate_bank_audit_pending_review",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "candidate_windows": WINDOWS,
        "direct_candidates": direct_candidates,
        "saved_control_regenerated": False,
        "candidate_mean_persisted": False,
        "oracle_prediction_persisted": False,
        "selector_persisted": False,
        "tail30_control": tail30_row,
        "best_direct_candidate_target_side_diagnostic_only": best_diagnostic,
        "prediction_content_sha256": prediction_sha,
        "rate_spread": {
            "mean": float(rate_diagnostics["rate_spread"].mean()),
            "median": float(rate_diagnostics["rate_spread"].median()),
            "p90": float(rate_diagnostics["rate_spread"].quantile(0.90)),
            "zero_rate_spread_wells": int((rate_diagnostics["rate_spread"] <= 1.0e-12).sum()),
        },
        "oracle_scope_metrics": oracle_metrics.to_dict("records"),
        "runtime_seconds": float(time.time() - started),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "sha256": {
            key: sha256_path(path)
            for key, path in paths.items()
            if key != "summary" and path.exists()
        },
    }
    write_json(paths["summary"], summary)
    summary["sha256"]["summary"] = sha256_path(paths["summary"])
    write_json(paths["summary"], summary)
    metrics_path = artifacts.parent / "metrics.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "completed_train_side_candidate_bank_audit_pending_review",
            "metric": "rmse_tvt",
            "cv": tail30_row["rmse"],
            "public_lb": None,
            "private_lb": None,
            "rows": len(frame),
            "wells": frame["well"].nunique(),
            "tail30_control": tail30_row,
            "best_direct_candidate_target_side_diagnostic_only": best_diagnostic,
            "prediction_content_sha256": prediction_sha,
            "summary": str(paths["summary"]),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 7. Setup and execution contract

# %%
EXECUTE_NOTEBOOK = os.environ.get("EXP268_IMPORT_ONLY", "0") != "1"
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "run_kind": RUN_KIND_OVERRIDE,
                "candidate_windows": WINDOWS,
                "saved_tail30_control_regenerated": False,
                "active_hmm_variants": get_nested(CONFIG, "execution.active_hmm_variants"),
                "well_shards": get_nested(CONFIG, "execution.shard_count"),
                "lightgbm_configs": 0,
                "folds": 0,
                "boosters": 0,
                "gpu": False,
                "inference": False,
                "submission": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 8. Input preflight

# %%
if EXECUTE_NOTEBOOK:
    if RUN_KIND_OVERRIDE in {"shard0", "shard1"}:
        DATA_DIR = train_data_dir(CONFIG)
        if not DATA_DIR.exists() or not list_well_ids(DATA_DIR):
            raise FileNotFoundError(f"raw train well pairs not found: {DATA_DIR}")
        print(f"Raw train input: {DATA_DIR}; well pairs={len(list_well_ids(DATA_DIR))}")
    elif RUN_KIND_OVERRIDE == "aggregate":
        print("Aggregate preflight resolves two shard caches, exp209 control, exp072 reference, and exp115 folds.")
    else:
        raise ValueError(f"unsupported RUN_KIND_OVERRIDE={RUN_KIND_OVERRIDE}")


# %% [markdown]
# ## 9. Generate a shard or aggregate the candidate bank

# %%
if EXECUTE_NOTEBOOK:
    if RUN_KIND_OVERRIDE == "shard0":
        RUN_SUMMARY = run_shard_generation(CONFIG, shard_index=0)
    elif RUN_KIND_OVERRIDE == "shard1":
        RUN_SUMMARY = run_shard_generation(CONFIG, shard_index=1)
    else:
        RUN_SUMMARY = run_aggregate(CONFIG)


# %% [markdown]
# ## 10. Metrics and generated artifacts

# %%
if EXECUTE_NOTEBOOK:
    print("Run completed:")
    print(json.dumps(to_jsonable(RUN_SUMMARY), indent=2, sort_keys=True))
    print("Generated files:")
    for generated_path in sorted(artifact_dir().glob(f"{OUTPUT_PREFIX}*")):
        print(f"- {generated_path.name}")
