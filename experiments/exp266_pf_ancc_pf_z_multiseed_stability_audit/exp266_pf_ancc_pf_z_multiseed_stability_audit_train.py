# %% [markdown]
# # exp266 PF ANCC / PF-Z multiseed stability audit
#
# Exact exp072 replay with one original and 63 new stable seeds per algorithm.
# The notebook freezes all paths before using true TVT for diagnostic readout.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, and SHA helpers
# 3. Input cache and reference helpers
# 4. Exact exp072 PF kernels
# 5. Per-well replay helpers
# 6. Diagnostic and artifact helpers
# 7. Setup and fixed execution contract
# 8. Input checks and reference assembly
# 9. Original-seed exact parity phase
# 10. Full multiseed generation
# 11. Stability, convergence, and occurrence-condition readout
# 12. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from numba import njit

EXPERIMENT_NAME = "exp266_pf_ancc_pf_z_multiseed_stability_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME


# %% [markdown]
# ## 2. Runtime, configuration, and SHA helpers


# %%
def is_kaggle_runtime() -> bool:
    return Path("/kaggle/input").exists() and Path("/kaggle/working").exists()


def find_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path("/kaggle/working/config.yaml"),
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in sorted(Path.cwd().rglob("config.yaml")):
        try:
            data = yaml.safe_load(candidate.read_text()) or {}
        except Exception:
            continue
        if (data.get("experiment") or {}).get("name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError(f"config.yaml for {EXPERIMENT_NAME} was not found")


def load_config() -> tuple[dict[str, Any], Path]:
    path = find_config_path()
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    return value, path


def nested(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def sha256_path(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_content(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        [(str(column), str(frame[column].dtype)) for column in frame.columns],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_competition_root(config: dict[str, Any]) -> Path:
    if is_kaggle_runtime():
        direct = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
        if (direct / "train").is_dir():
            return direct
        for candidate in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
            if (candidate.parent / "train").is_dir():
                return candidate.parent
        raise FileNotFoundError("Kaggle competition train directory was not found")
    return Path(str(nested(config, "data.raw_dir", "data/raw")))


def resolve_artifact(filename: str, local_candidates: list[str] | None = None) -> Path:
    candidates = [Path(value) for value in (local_candidates or [])]
    if is_kaggle_runtime():
        candidates.extend(sorted(Path("/kaggle/input").rglob(filename)))
    else:
        candidates.extend(sorted(Path.cwd().rglob(filename)))
        candidates.extend(sorted(Path("/tmp").rglob(filename)))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"artifact not found: {filename}")


def require_kaggle_or_explicit_local() -> None:
    if is_kaggle_runtime():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "Kaggle Notebook is the source of truth. Set EXPERIMENT_ALLOW_LOCAL=1 "
        "only for an approved smoke run."
    )


# %% [markdown]
# ## 3. Input cache and reference helpers


# %%
def assert_file_sha(
    path: Path,
    *,
    expected_raw: str | None,
    expected_decompressed: str | None,
) -> dict[str, Any]:
    raw = sha256_path(path)
    decompressed = sha256_gzip_content(path) if path.suffix == ".gz" else raw
    if expected_raw and raw != expected_raw:
        raise RuntimeError(f"raw SHA mismatch for {path.name}: {raw} != {expected_raw}")
    if expected_decompressed and decompressed != expected_decompressed:
        raise RuntimeError(
            f"decompressed SHA mismatch for {path.name}: {decompressed} != {expected_decompressed}"
        )
    return {
        "kind": "reference_cache",
        "path": str(path),
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "raw_sha256": raw,
        "decompressed_sha256": decompressed,
    }


def id_row_index(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()


def pooled_reference_by_well(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[
        ["well", "target_tvt", "pf_ancc", "pf_z", "likpf_mean_tvt", "hmm_mean_tvt", "exp226_tvt"]
    ].copy()
    pred_columns = {
        "pf_ancc": "pf_ancc",
        "pf_z": "pf_z",
        "likpf": "likpf_mean_tvt",
        "hmm": "hmm_mean_tvt",
        "exp226": "exp226_tvt",
    }
    for name, column in pred_columns.items():
        error = work[column].to_numpy(np.float64) - work["target_tvt"].to_numpy(np.float64)
        work[f"{name}_sse"] = error * error
        work[f"{name}_ae"] = np.abs(error)
    aggregations: dict[str, tuple[str, str]] = {"rows": ("well", "size")}
    for name in pred_columns:
        aggregations[f"{name}_sse"] = (f"{name}_sse", "sum")
        aggregations[f"{name}_mae"] = (f"{name}_ae", "mean")
    by_well = work.groupby("well", sort=True).agg(**aggregations).reset_index()
    for name in pred_columns:
        by_well[f"{name}_rmse"] = np.sqrt(
            by_well[f"{name}_sse"].to_numpy(np.float64) / by_well["rows"].to_numpy(np.float64)
        )
        by_well.drop(columns=[f"{name}_sse"], inplace=True)
    return by_well


def load_reference_surface(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    exp072_path = resolve_artifact(
        str(nested(config, "data.exp072.filename")),
        [
            "experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
        ],
    )
    exp209_path = resolve_artifact(
        str(nested(config, "data.exp209.filename")),
        [
            "/tmp/exp209_blend_audit/artifacts/exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz",
        ],
    )
    exp226_path = resolve_artifact(
        str(nested(config, "data.exp226.filename")),
        [
            "/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/train_v1/artifacts/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz",
        ],
    )

    manifests = [
        assert_file_sha(
            exp072_path,
            expected_raw=str(nested(config, "data.exp072.expected_sha256")),
            expected_decompressed=str(nested(config, "data.exp072.expected_decompressed_sha256")),
        ),
        assert_file_sha(
            exp209_path,
            expected_raw=str(nested(config, "data.exp209.expected_sha256")),
            expected_decompressed=str(nested(config, "data.exp209.expected_decompressed_sha256")),
        ),
        assert_file_sha(
            exp226_path,
            expected_raw=None,
            expected_decompressed=str(nested(config, "data.exp226.expected_decompressed_sha256")),
        ),
    ]

    base = pd.read_csv(
        exp072_path,
        usecols=[
            "id",
            "well",
            "target",
            "last_known_tvt",
            "md_since",
            "pf_ancc",
            "pf_z",
            "likpf_mean_d",
        ],
        dtype={
            "id": "string",
            "well": "string",
            "pf_ancc": "float32",
            "pf_z": "float32",
        },
    )
    base["id"] = base["id"].astype(str)
    base["well"] = base["well"].astype(str)
    if base["pf_ancc"].dtype != np.dtype(np.float32) or base["pf_z"].dtype != np.dtype(
        np.float32
    ):
        raise RuntimeError("exp072 PF reference columns must retain their original float32 dtype")
    id_well = base["id"].str.rsplit("_", n=1).str[0]
    identity_mismatch = id_well.ne(base["well"])
    if identity_mismatch.any():
        mismatch = base.loc[identity_mismatch, ["id", "well"]].head(10)
        raise RuntimeError(
            "exp072 well/id identity mismatch after string-preserving CSV read: "
            f"{mismatch.to_dict(orient='records')}"
        )
    base["row_idx"] = id_row_index(base["id"])
    base["target_tvt"] = base["last_known_tvt"].to_numpy(np.float64) + base["target"].to_numpy(
        np.float64
    )
    base["likpf_mean_tvt"] = base["last_known_tvt"].to_numpy(np.float64) + base[
        "likpf_mean_d"
    ].to_numpy(np.float64)

    hmm = pd.read_csv(
        exp209_path,
        usecols=["id", "hmm_mean_tvt"],
        dtype={"id": "string"},
    )
    hmm["id"] = hmm["id"].astype(str)
    exp226 = pd.read_csv(
        exp226_path,
        usecols=["well_id", "row_idx", "tvt_pred"],
        dtype={"well_id": "string"},
    )
    exp226["well_id"] = exp226["well_id"].astype(str)
    exp226["id"] = exp226["well_id"] + "_" + exp226["row_idx"].astype(np.int64).astype(str)
    exp226.rename(columns={"tvt_pred": "exp226_tvt"}, inplace=True)

    if (
        base["id"].duplicated().any()
        or hmm["id"].duplicated().any()
        or exp226["id"].duplicated().any()
    ):
        raise RuntimeError("duplicate reference id detected")
    merged = base.merge(hmm, on="id", how="left", validate="one_to_one")
    merged = merged.merge(exp226[["id", "exp226_tvt"]], on="id", how="left", validate="one_to_one")
    required = ["hmm_mean_tvt", "exp226_tvt", "target_tvt", "pf_ancc", "pf_z", "likpf_mean_tvt"]
    if merged[required].isna().any().any():
        raise RuntimeError(
            f"reference join has missing values: {merged[required].isna().sum().to_dict()}"
        )

    expected_rows = int(nested(config, "validation.expected_rows"))
    expected_wells = int(nested(config, "validation.expected_wells"))
    if len(merged) != expected_rows or merged["well"].nunique() != expected_wells:
        raise RuntimeError(
            f"reference coverage mismatch: rows={len(merged)}, wells={merged['well'].nunique()}"
        )
    by_well = pooled_reference_by_well(merged)
    margin = float(nested(config, "validation.strong_margin_ft", 2.0))
    good_side = np.maximum(by_well["pf_ancc_rmse"], by_well["exp226_rmse"])
    bad_side = np.minimum(by_well["hmm_rmse"], by_well["likpf_rmse"])
    by_well["strict_original_phenotype"] = good_side < bad_side
    by_well["strong_original_phenotype"] = good_side + margin < bad_side
    by_well["original_family_margin_ft"] = bad_side - good_side
    return merged, by_well, manifests


# %% [markdown]
# ## 4. Exact exp072 PF kernels
#
# These equations and operation order are copied from exp072. Numba cache is
# disabled only for notebook safety; the numerical path is unchanged.

# %%
PF_N = 600
ANCC_N = 600
PF_MOM = 0.993
PF_VN = 0.005
PF_PN = 0.01
PF_GR_SIG_MIN = 10.0
PF_GR_SIG_MAX = 60.0
PF_GR_SIG_DEF = 30.0
PF_GR_WIN = 5
PF_GR_WT = 0.3
PF_RESAMP = 0.5
PF_ROUGH_P = 0.2
PF_ROUGH_V = 0.003
ANCC_ALPHA = 0.998
ANCC_RN = 0.002
ANCC_PN = 0.005
ANCC_IS = 0.3
ANCC_RP = 0.1
ANCC_RR = 0.001


@njit(cache=False, nogil=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=False, nogil=True)
def _resamp(pos, aux, w, n_particles, rough_position, rough_aux):
    cum = np.zeros(n_particles + 1)
    for j in range(n_particles):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / n_particles)
    new_pos = np.empty(n_particles)
    new_aux = np.empty(n_particles)
    cursor = 0
    for j in range(n_particles):
        u = u0 + j / n_particles
        while cursor < n_particles - 1 and cum[cursor + 1] < u:
            cursor += 1
        new_pos[j] = pos[cursor] + rough_position * np.random.randn()
        new_aux[j] = aux[cursor] + rough_aux * np.random.randn()
    return new_pos, new_aux


@njit(cache=False, nogil=True)
def _pf_ancc(
    md_v,
    z_v,
    gr_v,
    grid,
    vmin,
    step,
    gr_sigma,
    last_surface,
    initial_rate,
    n_particles,
    alpha,
    rate_noise,
    position_noise,
    init_spread,
    rough_position,
    rough_rate,
    resample_threshold,
):
    pos = np.empty(n_particles)
    rate = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for j in range(n_particles):
        pos[j] = last_surface + init_spread * np.random.randn()
        rate[j] = initial_rate + 0.01 * np.random.randn()
    predictions = np.empty(len(md_v))
    stds = np.empty(len(md_v))
    previous_md = md_v[0] - 1.0
    for i in range(len(md_v)):
        delta_md = md_v[i] - previous_md
        delta_md = max(delta_md, 1.0)
        for j in range(n_particles):
            rate[j] = alpha * rate[j] + rate_noise * np.random.randn()
            pos[j] += rate[j] * delta_md + position_noise * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(grid) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            weight_sum = 0.0
            for j in range(n_particles):
                expected_gr = _interp1(grid, pos[j] - z_v[i], vmin, step)
                delta = (gr_v[i] - expected_gr) / gr_sigma
                likelihood = max(
                    np.exp(-0.5 * delta * delta) if delta * delta < 600.0 else 0.0,
                    1e-300,
                )
                weights[j] *= likelihood
                weight_sum += weights[j]
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles
        neff_inverse = 0.0
        for j in range(n_particles):
            neff_inverse += weights[j] * weights[j]
        if 1.0 / neff_inverse < resample_threshold * n_particles:
            pos, rate = _resamp(
                pos,
                rate,
                weights,
                n_particles,
                rough_position,
                rough_rate,
            )
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles
        estimate = 0.0
        for j in range(n_particles):
            estimate += weights[j] * (pos[j] - z_v[i])
        predictions[i] = estimate
        variance = 0.0
        for j in range(n_particles):
            variance += weights[j] * (pos[j] - z_v[i] - estimate) ** 2
        stds[i] = variance**0.5
        previous_md = md_v[i]
    return predictions, stds


@njit(cache=False, nogil=True)
def _pf_z(
    md_v,
    z_v,
    gr_v,
    gr_smoothed_v,
    primary_grid,
    smooth_grid,
    vmin,
    step,
    gr_sigma,
    initial_position,
    initial_velocity,
    beta,
    intercept,
    z_sigma,
    n_particles,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_position,
    rough_velocity,
    resample_threshold,
):
    pos = np.empty(n_particles)
    velocity = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for j in range(n_particles):
        pos[j] = initial_position + 0.5 * np.random.randn()
        velocity[j] = initial_velocity + 0.02 * np.random.randn()
    predictions = np.empty(len(md_v))
    stds = np.empty(len(md_v))
    previous_md = md_v[0] - 1.0
    previous_z = z_v[0] - 1.0
    for i in range(len(md_v)):
        delta_md = md_v[i] - previous_md
        delta_md = max(delta_md, 1.0)
        z_rate = (z_v[i] - previous_z) / delta_md
        expected_velocity = beta * z_rate + intercept
        for j in range(n_particles):
            velocity[j] = momentum * velocity[j] + velocity_noise * np.random.randn()
            pos[j] += velocity[j] * delta_md + position_noise * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(primary_grid) * step + 50.0)
        if not np.isnan(gr_v[i]):
            weight_sum = 0.0
            for j in range(n_particles):
                expected_primary = _interp1(primary_grid, pos[j], vmin, step)
                delta_primary = (gr_v[i] - expected_primary) / gr_sigma
                primary_likelihood = max(
                    np.exp(-0.5 * delta_primary * delta_primary)
                    if delta_primary * delta_primary < 600.0
                    else 0.0,
                    1e-300,
                )
                if not np.isnan(gr_smoothed_v[i]):
                    expected_smooth = _interp1(smooth_grid, pos[j], vmin, step)
                    delta_smooth = (gr_smoothed_v[i] - expected_smooth) / (gr_sigma * 1.5)
                    smooth_likelihood = max(
                        np.exp(-0.5 * delta_smooth * delta_smooth)
                        if delta_smooth * delta_smooth < 600.0
                        else 0.0,
                        1e-300,
                    )
                    likelihood = (
                        1.0 - gr_smooth_weight
                    ) * primary_likelihood + gr_smooth_weight * smooth_likelihood
                else:
                    likelihood = primary_likelihood
                likelihood = max(likelihood, 1e-300)
                weights[j] *= likelihood
                weight_sum += weights[j]
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles
        velocity_weight_sum = 0.0
        for j in range(n_particles):
            delta_velocity = (velocity[j] - expected_velocity) / max(z_sigma * 2.0, 0.005)
            velocity_likelihood = max(
                np.exp(-0.5 * delta_velocity * delta_velocity)
                if delta_velocity * delta_velocity < 600.0
                else 0.0,
                1e-300,
            )
            weights[j] *= velocity_likelihood
            velocity_weight_sum += weights[j]
        if velocity_weight_sum > 0.0:
            for j in range(n_particles):
                weights[j] /= velocity_weight_sum
        else:
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles
        neff_inverse = 0.0
        for j in range(n_particles):
            neff_inverse += weights[j] * weights[j]
        if 1.0 / neff_inverse < resample_threshold * n_particles:
            pos, velocity = _resamp(
                pos,
                velocity,
                weights,
                n_particles,
                rough_position,
                rough_velocity,
            )
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles
        estimate = 0.0
        for j in range(n_particles):
            estimate += weights[j] * pos[j]
        predictions[i] = estimate
        variance = 0.0
        for j in range(n_particles):
            variance += weights[j] * (pos[j] - estimate) ** 2
        stds[i] = variance**0.5
        previous_md = md_v[i]
        previous_z = z_v[i]
    return predictions, stds


@njit(cache=False, nogil=True)
def _pf_ancc_seeded(
    seed,
    md_v,
    z_v,
    gr_v,
    grid,
    vmin,
    step,
    gr_sigma,
    last_surface,
    initial_rate,
    n_particles,
    alpha,
    rate_noise,
    position_noise,
    init_spread,
    rough_position,
    rough_rate,
    resample_threshold,
):
    np.random.seed(seed)
    return _pf_ancc(
        md_v,
        z_v,
        gr_v,
        grid,
        vmin,
        step,
        gr_sigma,
        last_surface,
        initial_rate,
        n_particles,
        alpha,
        rate_noise,
        position_noise,
        init_spread,
        rough_position,
        rough_rate,
        resample_threshold,
    )


@njit(cache=False, nogil=True)
def _pf_z_seeded(
    seed,
    md_v,
    z_v,
    gr_v,
    gr_smoothed_v,
    primary_grid,
    smooth_grid,
    vmin,
    step,
    gr_sigma,
    initial_position,
    initial_velocity,
    beta,
    intercept,
    z_sigma,
    n_particles,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_position,
    rough_velocity,
    resample_threshold,
):
    np.random.seed(seed)
    return _pf_z(
        md_v,
        z_v,
        gr_v,
        gr_smoothed_v,
        primary_grid,
        smooth_grid,
        vmin,
        step,
        gr_sigma,
        initial_position,
        initial_velocity,
        beta,
        intercept,
        z_sigma,
        n_particles,
        momentum,
        velocity_noise,
        position_noise,
        gr_smooth_weight,
        rough_position,
        rough_velocity,
        resample_threshold,
    )


def _grid(tvt: np.ndarray, gr: np.ndarray, step: float = 0.2) -> tuple[np.ndarray, float, float]:
    minimum = float(tvt.min())
    maximum = float(tvt.max())
    tvt_grid = np.arange(minimum, maximum + step, step)
    return np.interp(tvt_grid, tvt, gr).astype(np.float64), minimum, float(step)


def _gr_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    known = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(known) < 20:
        return float(PF_GR_SIG_DEF)
    return float(
        np.clip(
            np.std(known.GR.values - np.interp(known.TVT_input.values, tw_tvt, tw_gr)),
            PF_GR_SIG_MIN,
            PF_GR_SIG_MAX,
        )
    )


# %% [markdown]
# ## 5. Per-well replay helpers


# %%
def read_well(train_dir: Path, well: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"raw train files missing for {well}")
    hw = pd.read_csv(horizontal_path)
    tw = pd.read_csv(typewell_path).sort_values("TVT")
    return hw, tw, horizontal_path, typewell_path


def prepare_ancc_args(
    hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray
) -> tuple[tuple[Any, ...], dict[str, float]]:
    gr_sigma = _gr_sigma(hw, tw_tvt, tw_gr)
    known = hw[hw.TVT_input.notna()]
    evaluation = hw[hw.TVT_input.isna()]
    if evaluation.empty or known.empty:
        raise RuntimeError("PF ANCC requires known prefix and evaluation rows")
    last_surface = float(known.TVT_input.iloc[-1] + known.Z.iloc[-1])
    tail = known.tail(30)
    delta_tvt = np.diff(tail.TVT_input.values)
    delta_z = np.diff(tail.Z.values)
    delta_md = np.diff(tail.MD.values)
    valid = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[valid] / delta_md[valid]))
        if valid.sum() >= 3
        else 0.0
    )
    grid, minimum, step = _grid(tw_tvt, tw_gr)
    args = (
        evaluation.MD.values.astype(np.float64),
        evaluation.Z.values.astype(np.float64),
        evaluation.GR.values.astype(np.float64),
        grid,
        minimum,
        step,
        gr_sigma,
        last_surface,
        initial_rate,
        ANCC_N,
        ANCC_ALPHA,
        ANCC_RN,
        ANCC_PN,
        ANCC_IS,
        ANCC_RP,
        ANCC_RR,
        PF_RESAMP,
    )
    return args, {"gr_sigma": gr_sigma, "ancc_initial_rate": initial_rate}


def prepare_pf_z_args(
    hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray
) -> tuple[tuple[Any, ...], dict[str, float]]:
    gr_sigma = _gr_sigma(hw, tw_tvt, tw_gr)
    tw_smooth = (
        pd.Series(tw_gr)
        .rolling(PF_GR_WIN, center=True, min_periods=1)
        .mean()
        .values.astype(np.float32)
    )
    known = hw[hw.TVT_input.notna()]
    evaluation = hw[hw.TVT_input.isna()]
    if evaluation.empty or known.empty:
        raise RuntimeError("PF-Z requires known prefix and evaluation rows")
    delta_z = np.diff(known.Z.values)
    delta_tvt = np.diff(known.TVT_input.values)
    delta_md = np.diff(known.MD.values)
    valid = delta_md > 0
    if valid.sum() >= 10:
        z_rate = delta_z[valid] / delta_md[valid]
        tvt_rate = delta_tvt[valid] / delta_md[valid]
        design = np.column_stack([z_rate, np.ones_like(z_rate)])
        coefficient, _, _, _ = np.linalg.lstsq(design, tvt_rate, rcond=None)
        beta = float(coefficient[0])
        intercept = float(coefficient[1])
        z_sigma = max(
            float(np.std(tvt_rate - (coefficient[0] * z_rate + coefficient[1]))),
            0.001,
        )
    else:
        beta, intercept, z_sigma = -1.0, 0.0, 0.1
    tail = known.tail(20)
    tail_delta_tvt = np.diff(tail.TVT_input.values)
    tail_delta_md = np.diff(tail.MD.values)
    tail_valid = tail_delta_md > 0
    initial_velocity = (
        float(np.median(tail_delta_tvt[tail_valid] / tail_delta_md[tail_valid]))
        if tail_valid.sum() >= 3
        else 0.0
    )
    primary_grid, minimum, step = _grid(tw_tvt, tw_gr)
    smooth_grid, _, _ = _grid(tw_tvt, tw_smooth)
    gr_smoothed = hw.GR.rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    args = (
        evaluation.MD.values.astype(np.float64),
        evaluation.Z.values.astype(np.float64),
        evaluation.GR.values.astype(np.float64),
        gr_smoothed.loc[evaluation.index].values.astype(np.float64),
        primary_grid,
        smooth_grid,
        minimum,
        step,
        gr_sigma,
        float(known.TVT_input.iloc[-1]),
        initial_velocity,
        beta,
        intercept,
        z_sigma,
        PF_N,
        PF_MOM,
        PF_VN,
        PF_PN,
        PF_GR_WT,
        PF_ROUGH_P,
        PF_ROUGH_V,
        PF_RESAMP,
    )
    return args, {
        "pf_z_beta": beta,
        "pf_z_intercept": intercept,
        "pf_z_sigma": z_sigma,
        "pf_z_initial_velocity": initial_velocity,
    }


def run_original_seed_task(train_dir: Path, well: str) -> dict[str, Any]:
    hw, tw, horizontal_path, typewell_path = read_well(train_dir, well)
    evaluation = hw[hw.TVT_input.isna()]
    known = hw[hw.TVT_input.notna()]
    tw_tvt = tw.TVT.to_numpy(np.float32)
    tw_gr = tw.GR.to_numpy(np.float32)
    ancc_args, ancc_quality = prepare_ancc_args(hw, tw_tvt, tw_gr)
    pf_z_args, pf_z_quality = prepare_pf_z_args(hw, tw_tvt, tw_gr)
    ancc_seed = stable_seed("pf_ancc", well)
    pf_z_seed = stable_seed("pf_z", well)
    ancc_pred, ancc_std = _pf_ancc_seeded(ancc_seed, *ancc_args)
    pf_z_pred, pf_z_std = _pf_z_seeded(pf_z_seed, *pf_z_args)
    quality = {
        "well": well,
        "known_rows": int(len(known)),
        "eval_rows": int(len(evaluation)),
        "eval_md_span": float(evaluation.MD.max() - evaluation.MD.min()),
        "typewell_rows": int(len(tw)),
        "typewell_tvt_range": float(np.ptp(tw_tvt)),
        "horizontal_sha256": sha256_path(horizontal_path),
        "typewell_sha256": sha256_path(typewell_path),
        **ancc_quality,
        **pf_z_quality,
    }
    return {
        "well": well,
        "row_idx": evaluation.index.to_numpy(np.int64),
        "pf_ancc": ancc_pred.astype(np.float32),
        "pf_ancc_std": ancc_std.astype(np.float32),
        "pf_z": pf_z_pred.astype(np.float32),
        "pf_z_std": pf_z_std.astype(np.float32),
        "quality": quality,
    }


def seed_vector(algorithm: str, well: str, seed_count: int) -> np.ndarray:
    original = stable_seed(algorithm, well)
    additional = [
        stable_seed(EXPERIMENT_NAME, "train", algorithm, well, seed_index)
        for seed_index in range(1, seed_count)
    ]
    values = np.asarray([original, *additional], dtype=np.int64)
    if len(np.unique(values)) != seed_count:
        raise RuntimeError(f"seed collision for {algorithm}/{well}")
    return values


def distance_bucket_masks(md_since: np.ndarray, edges: list[float]) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        labels[f"rmse_{int(left)}_{int(right)}"] = (md_since >= left) & (md_since < right)
    labels[f"rmse_{int(edges[-1])}_plus"] = md_since >= edges[-1]
    return labels


def path_metric_record(
    *,
    well: str,
    algorithm: str,
    prediction: np.ndarray,
    target: np.ndarray,
    original_prediction: np.ndarray,
    md_since: np.ndarray,
    bucket_masks: dict[str, np.ndarray],
    mean_particle_std: float,
    seed_index: int | None = None,
    seed: int | None = None,
    aggregation: str | None = None,
    seed_count: int | None = None,
) -> dict[str, Any]:
    pred = prediction.astype(np.float64)
    error = pred - target
    absolute = np.abs(error)
    record: dict[str, Any] = {
        "well": well,
        "algorithm": algorithm,
        "rows": int(len(target)),
        "squared_error_sum": float(np.sum(error * error)),
        "absolute_error_sum": float(np.sum(absolute)),
        "error_sum": float(np.sum(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(error)),
        "endpoint_error": float(error[-1]),
        "endpoint_abs_error": float(absolute[-1]),
        "endpoint_sign": int(np.sign(error[-1])),
        "path_vs_original_rmse": float(
            np.sqrt(np.mean((pred - original_prediction.astype(np.float64)) ** 2))
        ),
        "mean_particle_std": float(mean_particle_std),
    }
    if seed_index is not None:
        record.update(
            {
                "seed_index": int(seed_index),
                "seed": int(seed),
                "is_original_seed": bool(seed_index == 0),
            }
        )
    if aggregation is not None:
        record.update({"aggregation": aggregation, "seed_count": int(seed_count)})
    for name, mask in bucket_masks.items():
        record[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if int(mask.sum()) else np.nan
    return record


def aggregate_path(paths: np.ndarray, aggregation: str, trim_fraction: float) -> np.ndarray:
    if aggregation == "mean":
        return paths.mean(axis=0)
    if aggregation == "median":
        return np.median(paths, axis=0)
    if aggregation == "trimmed_mean_10pct":
        ordered = np.sort(paths, axis=0)
        trim = int(math.floor(len(paths) * trim_fraction))
        if trim == 0:
            return ordered.mean(axis=0)
        return ordered[trim : len(paths) - trim].mean(axis=0)
    raise ValueError(f"unknown aggregation: {aggregation}")


def run_multiseed_task(
    *,
    ordinal: int,
    train_dir: Path,
    original: dict[str, Any],
    seed_count: int,
    nested_counts: list[int],
    aggregations: list[str],
    trim_fraction: float,
    bucket_edges: list[float],
    detailed_wells: set[str],
    progress_every: int,
) -> dict[str, Any]:
    well = str(original["well"])
    hw, tw, _, _ = read_well(train_dir, well)
    evaluation = hw[hw.TVT_input.isna()]
    known = hw[hw.TVT_input.notna()]
    target = evaluation.TVT.to_numpy(np.float64)
    md_since = evaluation.MD.to_numpy(np.float64) - float(known.MD.iloc[-1])
    row_idx = evaluation.index.to_numpy(np.int64)
    if not np.array_equal(row_idx, original["row_idx"]):
        raise RuntimeError(f"evaluation row index changed for {well}")
    tw_tvt = tw.TVT.to_numpy(np.float32)
    tw_gr = tw.GR.to_numpy(np.float32)
    ancc_args, _ = prepare_ancc_args(hw, tw_tvt, tw_gr)
    pf_z_args, _ = prepare_pf_z_args(hw, tw_tvt, tw_gr)
    bucket_masks = distance_bucket_masks(md_since, bucket_edges)

    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    detailed_frames: list[pd.DataFrame] = []
    for algorithm, args, kernel, original_key, original_std_key in [
        ("pf_ancc", ancc_args, _pf_ancc_seeded, "pf_ancc", "pf_ancc_std"),
        ("pf_z", pf_z_args, _pf_z_seeded, "pf_z", "pf_z_std"),
    ]:
        seeds = seed_vector(algorithm, well, seed_count)
        paths = np.empty((seed_count, len(evaluation)), dtype=np.float32)
        particle_stds = np.empty((seed_count, len(evaluation)), dtype=np.float32)
        paths[0] = original[original_key]
        particle_stds[0] = original[original_std_key]
        for seed_index in range(1, seed_count):
            prediction, particle_std = kernel(int(seeds[seed_index]), *args)
            paths[seed_index] = prediction.astype(np.float32)
            particle_stds[seed_index] = particle_std.astype(np.float32)
        for seed_index in range(seed_count):
            seed_rows.append(
                path_metric_record(
                    well=well,
                    algorithm=algorithm,
                    prediction=paths[seed_index],
                    target=target,
                    original_prediction=paths[0],
                    md_since=md_since,
                    bucket_masks=bucket_masks,
                    mean_particle_std=float(particle_stds[seed_index].mean()),
                    seed_index=seed_index,
                    seed=int(seeds[seed_index]),
                )
            )
        for count in nested_counts:
            subset = paths[:count]
            for aggregation in aggregations:
                prediction = aggregate_path(subset, aggregation, trim_fraction)
                aggregate_rows.append(
                    path_metric_record(
                        well=well,
                        algorithm=algorithm,
                        prediction=prediction,
                        target=target,
                        original_prediction=paths[0],
                        md_since=md_since,
                        bucket_masks=bucket_masks,
                        mean_particle_std=float(particle_stds[:count].mean()),
                        aggregation=aggregation,
                        seed_count=count,
                    )
                )
        if well in detailed_wells:
            new_paths = paths[1:]
            detail = pd.DataFrame(
                {
                    "id": [f"{well}_{int(index)}" for index in row_idx],
                    "well": well,
                    "row_idx": row_idx,
                    "md_since": md_since.astype(np.float32),
                    "target_tvt": target.astype(np.float32),
                    "algorithm": algorithm,
                    "original_seed_path": paths[0],
                    "new_seed_q10": np.quantile(new_paths, 0.10, axis=0).astype(np.float32),
                    "new_seed_q25": np.quantile(new_paths, 0.25, axis=0).astype(np.float32),
                    "new_seed_median": np.median(new_paths, axis=0).astype(np.float32),
                    "new_seed_q75": np.quantile(new_paths, 0.75, axis=0).astype(np.float32),
                    "new_seed_q90": np.quantile(new_paths, 0.90, axis=0).astype(np.float32),
                    "new_seed_mean": new_paths.mean(axis=0).astype(np.float32),
                    "new_seed_std": new_paths.std(axis=0).astype(np.float32),
                    "all_seed_trimmed_mean": aggregate_path(
                        paths, "trimmed_mean_10pct", trim_fraction
                    ).astype(np.float32),
                }
            )
            detailed_frames.append(detail)
    if ordinal % progress_every == 0:
        print(
            f"multiseed progress ordinal={ordinal} well={well} rows={len(evaluation)}",
            flush=True,
        )
    return {
        "seed_metrics": pd.DataFrame(seed_rows),
        "aggregate_metrics": pd.DataFrame(aggregate_rows),
        "detailed": pd.concat(detailed_frames, ignore_index=True) if detailed_frames else None,
    }


# %% [markdown]
# ## 6. Diagnostic and artifact helpers


# %%
def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def build_well_stability_summary(
    seed_metrics: pd.DataFrame,
    reference_by_well: pd.DataFrame,
    quality: pd.DataFrame,
    thresholds: list[float],
    strong_margin: float,
) -> pd.DataFrame:
    reference_lookup = reference_by_well.set_index("well")
    rows: list[dict[str, Any]] = []
    for (well, algorithm), group in seed_metrics.groupby(["well", "algorithm"], sort=True):
        reference = reference_lookup.loc[well]
        original = group.loc[group["seed_index"] == 0].iloc[0]
        new = group.loc[group["seed_index"] > 0].copy()
        new_rmse = new["rmse"].to_numpy(np.float64)
        original_rmse = float(original["rmse"])
        record: dict[str, Any] = {
            "well": well,
            "algorithm": algorithm,
            "rows": int(original["rows"]),
            "new_seed_count": int(len(new)),
            "original_rmse": original_rmse,
            "original_rmse_lower_tail_percentile": float(np.mean(new_rmse <= original_rmse)),
            "new_seed_rmse_mean": float(new_rmse.mean()),
            "new_seed_rmse_std": float(new_rmse.std()),
            "new_seed_rmse_q05": float(np.quantile(new_rmse, 0.05)),
            "new_seed_rmse_q10": float(np.quantile(new_rmse, 0.10)),
            "new_seed_rmse_q25": float(np.quantile(new_rmse, 0.25)),
            "new_seed_rmse_median": float(np.quantile(new_rmse, 0.50)),
            "new_seed_rmse_q75": float(np.quantile(new_rmse, 0.75)),
            "new_seed_rmse_q90": float(np.quantile(new_rmse, 0.90)),
            "new_seed_rmse_q95": float(np.quantile(new_rmse, 0.95)),
            "new_seed_endpoint_error_mean": float(new["endpoint_error"].mean()),
            "new_seed_endpoint_error_std": float(new["endpoint_error"].std(ddof=0)),
            "new_seed_original_endpoint_sign_agreement": float(
                np.mean(new["endpoint_sign"].to_numpy() == int(original["endpoint_sign"]))
            ),
            "new_seed_path_vs_original_rmse_mean": float(new["path_vs_original_rmse"].mean()),
            "new_seed_particle_std_mean": float(new["mean_particle_std"].mean()),
            "pf_ancc_reference_rmse": float(reference["pf_ancc_rmse"]),
            "pf_z_reference_rmse": float(reference["pf_z_rmse"]),
            "exp226_reference_rmse": float(reference["exp226_rmse"]),
            "hmm_reference_rmse": float(reference["hmm_rmse"]),
            "likpf_reference_rmse": float(reference["likpf_rmse"]),
            "strict_original_phenotype": bool(reference["strict_original_phenotype"]),
            "strong_original_phenotype": bool(reference["strong_original_phenotype"]),
            "original_family_margin_ft": float(reference["original_family_margin_ft"]),
            "new_seed_beats_exp226_rate": float(
                np.mean(new_rmse < float(reference["exp226_rmse"]))
            ),
            "new_seed_beats_hmm_rate": float(np.mean(new_rmse < float(reference["hmm_rmse"]))),
            "new_seed_beats_likpf_rate": float(np.mean(new_rmse < float(reference["likpf_rmse"]))),
            "new_seed_strong_margin_vs_hmm_likpf_rate": float(
                np.mean(
                    new_rmse + strong_margin
                    < min(float(reference["hmm_rmse"]), float(reference["likpf_rmse"]))
                )
            ),
        }
        for threshold in thresholds:
            successes = int(np.sum(new_rmse <= threshold))
            low, high = wilson_interval(successes, len(new))
            key = str(threshold).replace(".", "p")
            record[f"new_seed_rmse_le_{key}_count"] = successes
            record[f"new_seed_rmse_le_{key}_rate"] = successes / len(new)
            record[f"new_seed_rmse_le_{key}_wilson_low"] = low
            record[f"new_seed_rmse_le_{key}_wilson_high"] = high
        rows.append(record)
    summary = pd.DataFrame(rows)
    return summary.merge(quality, on="well", how="left", validate="many_to_one")


def pooled_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, sort=True)
        .agg(
            rows=("rows", "sum"),
            wells=("well", "nunique"),
            squared_error_sum=("squared_error_sum", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            error_sum=("error_sum", "sum"),
            median_well_rmse=("rmse", "median"),
            mean_well_rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    grouped["rmse"] = np.sqrt(grouped["squared_error_sum"] / grouped["rows"])
    grouped["mae"] = grouped["absolute_error_sum"] / grouped["rows"]
    grouped["bias"] = grouped["error_sum"] / grouped["rows"]
    return grouped


def build_occurrence_group_summary(well_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "original_rmse",
        "original_rmse_lower_tail_percentile",
        "new_seed_rmse_mean",
        "new_seed_rmse_std",
        "new_seed_rmse_median",
        "new_seed_beats_exp226_rate",
        "new_seed_beats_hmm_rate",
        "new_seed_beats_likpf_rate",
        "new_seed_strong_margin_vs_hmm_likpf_rate",
        "new_seed_original_endpoint_sign_agreement",
    ]
    return (
        well_summary.groupby(["algorithm", "strong_original_phenotype"], sort=True)
        .agg(
            wells=("well", "nunique"),
            **{f"{column}_mean": (column, "mean") for column in metric_columns},
            **{f"{column}_median": (column, "median") for column in metric_columns},
        )
        .reset_index()
    )


def build_feature_associations(well_summary: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "known_rows",
        "eval_rows",
        "eval_md_span",
        "typewell_rows",
        "typewell_tvt_range",
        "gr_sigma",
        "ancc_initial_rate",
        "pf_z_beta",
        "pf_z_intercept",
        "pf_z_sigma",
        "pf_z_initial_velocity",
        "original_family_margin_ft",
        "exp226_reference_rmse",
        "hmm_reference_rmse",
        "likpf_reference_rmse",
    ]
    outcome_columns = [
        "original_rmse_lower_tail_percentile",
        "new_seed_rmse_mean",
        "new_seed_rmse_std",
        "new_seed_rmse_median",
        "new_seed_strong_margin_vs_hmm_likpf_rate",
        "new_seed_original_endpoint_sign_agreement",
    ]
    rows: list[dict[str, Any]] = []
    for algorithm, group in well_summary.groupby("algorithm", sort=True):
        for feature in feature_columns:
            for outcome in outcome_columns:
                pair = group[[feature, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "algorithm": algorithm,
                        "feature": feature,
                        "outcome": outcome,
                        "wells": int(len(pair)),
                        "spearman": float(pair[feature].corr(pair[outcome], method="spearman"))
                        if len(pair) >= 3
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_feature_buckets(well_summary: pd.DataFrame) -> pd.DataFrame:
    features = [
        "known_rows",
        "eval_rows",
        "eval_md_span",
        "typewell_tvt_range",
        "gr_sigma",
        "pf_z_sigma",
        "original_family_margin_ft",
    ]
    rows: list[dict[str, Any]] = []
    for algorithm, algorithm_group in well_summary.groupby("algorithm", sort=True):
        for feature in features:
            work = (
                algorithm_group[
                    [
                        feature,
                        "well",
                        "strong_original_phenotype",
                        "original_rmse_lower_tail_percentile",
                        "new_seed_rmse_std",
                        "new_seed_rmse_median",
                        "new_seed_strong_margin_vs_hmm_likpf_rate",
                    ]
                ]
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[feature])
            )
            if work[feature].nunique() < 2:
                continue
            work = work.copy()
            work["bucket"] = pd.qcut(work[feature], q=5, duplicates="drop")
            for bucket, group in work.groupby("bucket", observed=True):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "feature": feature,
                        "bucket": str(bucket),
                        "wells": int(group["well"].nunique()),
                        "strong_rate": float(group["strong_original_phenotype"].mean()),
                        "original_percentile_mean": float(
                            group["original_rmse_lower_tail_percentile"].mean()
                        ),
                        "new_seed_rmse_std_mean": float(group["new_seed_rmse_std"].mean()),
                        "new_seed_rmse_median_mean": float(group["new_seed_rmse_median"].mean()),
                        "strong_margin_reproduction_rate_mean": float(
                            group["new_seed_strong_margin_vs_hmm_likpf_rate"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path, *, gzip_output: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        frame.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=False)
    return path


def artifact_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "filename": path.name,
                "bytes": int(path.stat().st_size),
                "raw_sha256": sha256_path(path),
                "decompressed_sha256": sha256_gzip_content(path)
                if path.suffix == ".gz"
                else sha256_path(path),
            }
        )
    return rows


# %% [markdown]
# ## 7. Setup and fixed execution contract

# %%
require_kaggle_or_explicit_local()
CONFIG, CONFIG_PATH = load_config()
COMPETITION_ROOT = find_competition_root(CONFIG)
TRAIN_DIR = COMPETITION_ROOT / "train"
OUTPUT_ROOT = (
    Path("/kaggle/working") if is_kaggle_runtime() else Path.cwd() / "experiments" / EXPERIMENT_NAME
)
ARTIFACT_DIR = OUTPUT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED_COUNT = int(nested(CONFIG, "model.pf.seed_count"))
NESTED_COUNTS = [int(value) for value in nested(CONFIG, "model.pf.nested_seed_counts")]
AGGREGATIONS = [str(value) for value in nested(CONFIG, "model.pf.aggregations")]
TRIM_FRACTION = float(nested(CONFIG, "model.pf.trim_fraction"))
NUM_WORKERS = int(nested(CONFIG, "runtime.num_workers"))
PROGRESS_EVERY = int(nested(CONFIG, "execution.progress_every_wells"))
BUCKET_EDGES = [float(value) for value in nested(CONFIG, "validation.distance_bucket_edges_ft")]
THRESHOLDS = [float(value) for value in nested(CONFIG, "validation.success_thresholds_ft")]
STRONG_MARGIN = float(nested(CONFIG, "validation.strong_margin_ft"))
FOCUS_WELL = str(nested(CONFIG, "validation.focus_well"))

if SEED_COUNT != 64 or NESTED_COUNTS != [1, 4, 8, 16, 32, 64]:
    raise RuntimeError("seed contract changed from the approved 64-seed nested design")
if nested(CONFIG, "execution.active_variant_count") != 2:
    raise RuntimeError("active PF variant count must be exactly 2")
if any(
    int(nested(CONFIG, key, -1)) != 0
    for key in [
        "execution.lightgbm_config_count",
        "execution.fold_count",
        "execution.total_boosters",
    ]
):
    raise RuntimeError("exp266 must run with 0 configs, 0 folds, and 0 boosters")

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": nested(CONFIG, "experiment.route"),
            "config_path": str(CONFIG_PATH),
            "competition_root": str(COMPETITION_ROOT),
            "artifact_dir": str(ARTIFACT_DIR),
            "algorithms": nested(CONFIG, "model.algorithms"),
            "particles": nested(CONFIG, "model.pf.particles"),
            "seed_count": SEED_COUNT,
            "nested_counts": NESTED_COUNTS,
            "aggregations": AGGREGATIONS,
            "workers": NUM_WORKERS,
            "lightgbm_configs": 0,
            "folds": 0,
            "boosters": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
        indent=2,
    )
)


# %% [markdown]
# ## 8. Input checks and reference assembly

# %%
RUN_START = time.time()
REFERENCE_ROWS, REFERENCE_BY_WELL, INPUT_MANIFEST_CACHE = load_reference_surface(CONFIG)
WELLS = sorted(REFERENCE_BY_WELL["well"].astype(str).tolist())
STRONG_WELLS = set(
    REFERENCE_BY_WELL.loc[REFERENCE_BY_WELL["strong_original_phenotype"], "well"].astype(str)
)
if FOCUS_WELL not in WELLS:
    raise RuntimeError(f"focus well {FOCUS_WELL} is missing")
if len(WELLS) != int(nested(CONFIG, "validation.expected_wells")):
    raise RuntimeError("well coverage mismatch")
print(
    f"reference rows={len(REFERENCE_ROWS):,} wells={len(WELLS)} "
    f"strict={int(REFERENCE_BY_WELL['strict_original_phenotype'].sum())} "
    f"strong={len(STRONG_WELLS)} focus={FOCUS_WELL}",
    flush=True,
)


# %% [markdown]
# ## 9. Original-seed exact parity phase

# %%
warm_md = np.linspace(1.0, 8.0, 8, dtype=np.float64)
warm_z = np.zeros(8, dtype=np.float64)
warm_gr = np.full(8, 50.0, dtype=np.float64)
warm_grid = np.linspace(45.0, 55.0, 100, dtype=np.float64)
_pf_ancc_seeded(
    12345,
    warm_md,
    warm_z,
    warm_gr,
    warm_grid,
    45.0,
    0.1,
    20.0,
    50.0,
    0.0,
    8,
    ANCC_ALPHA,
    ANCC_RN,
    ANCC_PN,
    ANCC_IS,
    ANCC_RP,
    ANCC_RR,
    PF_RESAMP,
)
_pf_z_seeded(
    12345,
    warm_md,
    warm_z,
    warm_gr,
    warm_gr,
    warm_grid,
    warm_grid,
    45.0,
    0.1,
    20.0,
    50.0,
    0.0,
    -1.0,
    0.0,
    0.1,
    8,
    PF_MOM,
    PF_VN,
    PF_PN,
    PF_GR_WT,
    PF_ROUGH_P,
    PF_ROUGH_V,
    PF_RESAMP,
)
print("exact PF kernels compiled", flush=True)

PARITY_START = time.time()
ORIGINAL_RESULTS = Parallel(n_jobs=NUM_WORKERS, prefer="threads")(
    delayed(run_original_seed_task)(TRAIN_DIR, well) for well in WELLS
)
original_frames: list[pd.DataFrame] = []
for result in ORIGINAL_RESULTS:
    well = str(result["well"])
    original_frames.append(
        pd.DataFrame(
            {
                "id": [f"{well}_{int(index)}" for index in result["row_idx"]],
                "well": well,
                "row_idx": result["row_idx"],
                "replay_pf_ancc": result["pf_ancc"],
                "replay_pf_z": result["pf_z"],
            }
        )
    )
ORIGINAL_FRAME = pd.concat(original_frames, ignore_index=True)
parity_join = REFERENCE_ROWS[["id", "pf_ancc", "pf_z"]].merge(
    ORIGINAL_FRAME[["id", "replay_pf_ancc", "replay_pf_z"]],
    on="id",
    how="left",
    validate="one_to_one",
)
if parity_join[["replay_pf_ancc", "replay_pf_z"]].isna().any().any():
    raise RuntimeError("original replay coverage is incomplete")
parity_records: list[dict[str, Any]] = []
for algorithm in ["pf_ancc", "pf_z"]:
    difference = parity_join[f"replay_{algorithm}"].to_numpy(np.float64) - parity_join[
        algorithm
    ].to_numpy(np.float64)
    parity_records.append(
        {
            "algorithm": algorithm,
            "rows": int(len(difference)),
            "mean_abs_diff": float(np.mean(np.abs(difference))),
            "rmse_diff": float(np.sqrt(np.mean(difference * difference))),
            "max_abs_diff": float(np.max(np.abs(difference))),
            "nonzero_rows": int(np.count_nonzero(difference)),
            "exact_parity": bool(np.count_nonzero(difference) == 0),
        }
    )
PARITY_SUMMARY = pd.DataFrame(parity_records)
PARITY_PATH = write_csv(PARITY_SUMMARY, ARTIFACT_DIR / "original_parity_summary.csv")
print(PARITY_SUMMARY.to_string(index=False), flush=True)
if not bool(PARITY_SUMMARY["exact_parity"].all()):
    raise RuntimeError("original-seed exact parity failed; multiseed phase is blocked")
PARITY_SECONDS = time.time() - PARITY_START


# %% [markdown]
# ## 10. Full multiseed generation

# %%
MULTISEED_START = time.time()
MULTISEED_RESULTS = Parallel(n_jobs=NUM_WORKERS, prefer="threads")(
    delayed(run_multiseed_task)(
        ordinal=ordinal,
        train_dir=TRAIN_DIR,
        original=original,
        seed_count=SEED_COUNT,
        nested_counts=NESTED_COUNTS,
        aggregations=AGGREGATIONS,
        trim_fraction=TRIM_FRACTION,
        bucket_edges=BUCKET_EDGES,
        detailed_wells=STRONG_WELLS,
        progress_every=PROGRESS_EVERY,
    )
    for ordinal, original in enumerate(ORIGINAL_RESULTS)
)
SEED_METRICS = pd.concat(
    [result["seed_metrics"] for result in MULTISEED_RESULTS], ignore_index=True
)
AGGREGATE_METRICS = pd.concat(
    [result["aggregate_metrics"] for result in MULTISEED_RESULTS], ignore_index=True
)
detailed_parts = [
    result["detailed"] for result in MULTISEED_RESULTS if result["detailed"] is not None
]
DETAILED_STRONG_PATHS = (
    pd.concat(detailed_parts, ignore_index=True) if detailed_parts else pd.DataFrame()
)
MULTISEED_SECONDS = time.time() - MULTISEED_START

expected_seed_rows = len(WELLS) * 2 * SEED_COUNT
expected_aggregate_rows = len(WELLS) * 2 * len(NESTED_COUNTS) * len(AGGREGATIONS)
if len(SEED_METRICS) != expected_seed_rows:
    raise RuntimeError(f"seed metric row mismatch: {len(SEED_METRICS)} != {expected_seed_rows}")
if len(AGGREGATE_METRICS) != expected_aggregate_rows:
    raise RuntimeError(
        f"aggregate metric row mismatch: {len(AGGREGATE_METRICS)} != {expected_aggregate_rows}"
    )
if not np.isfinite(SEED_METRICS[["rmse", "mae", "bias", "endpoint_error"]].to_numpy()).all():
    raise RuntimeError("nonfinite seed metric detected")
print(
    f"multiseed complete seed_rows={len(SEED_METRICS):,} "
    f"aggregate_rows={len(AGGREGATE_METRICS):,} "
    f"detailed_rows={len(DETAILED_STRONG_PATHS):,} seconds={MULTISEED_SECONDS:.3f}",
    flush=True,
)


# %% [markdown]
# ## 11. Stability, convergence, and occurrence-condition readout

# %%
QUALITY = pd.DataFrame([result["quality"] for result in ORIGINAL_RESULTS])
WELL_STABILITY = build_well_stability_summary(
    SEED_METRICS,
    REFERENCE_BY_WELL,
    QUALITY,
    THRESHOLDS,
    STRONG_MARGIN,
)
GLOBAL_SEED_METRICS = pooled_metrics(SEED_METRICS, ["algorithm", "seed_index"])
GLOBAL_AGGREGATE_METRICS = pooled_metrics(
    AGGREGATE_METRICS, ["algorithm", "seed_count", "aggregation"]
)
OCCURRENCE_GROUP_SUMMARY = build_occurrence_group_summary(WELL_STABILITY)
OCCURRENCE_ASSOCIATIONS = build_feature_associations(WELL_STABILITY)
OCCURRENCE_BUCKETS = build_feature_buckets(WELL_STABILITY)

focus_rows = WELL_STABILITY.loc[WELL_STABILITY["well"] == FOCUS_WELL].copy()
if len(focus_rows) != 2:
    raise RuntimeError(f"focus-well algorithm coverage mismatch: {len(focus_rows)}")
FOCUS_SUMMARY = focus_rows.to_dict(orient="records")
print("focus well stability:")
print(
    focus_rows[
        [
            "well",
            "algorithm",
            "original_rmse",
            "original_rmse_lower_tail_percentile",
            "new_seed_rmse_median",
            "new_seed_rmse_q10",
            "new_seed_rmse_q90",
            "new_seed_rmse_le_5p0_rate",
            "new_seed_rmse_le_10p0_rate",
            "new_seed_strong_margin_vs_hmm_likpf_rate",
            "new_seed_original_endpoint_sign_agreement",
        ]
    ].to_string(index=False),
    flush=True,
)


# %% [markdown]
# ## 12. Metrics and generated artifacts

# %%
REFERENCE_PATH = write_csv(REFERENCE_BY_WELL, ARTIFACT_DIR / "reference_by_well.csv")
SEED_PATH = write_csv(SEED_METRICS, ARTIFACT_DIR / "seed_by_well.csv.gz", gzip_output=True)
AGGREGATE_PATH = write_csv(
    AGGREGATE_METRICS,
    ARTIFACT_DIR / "aggregate_by_well.csv.gz",
    gzip_output=True,
)
WELL_PATH = write_csv(WELL_STABILITY, ARTIFACT_DIR / "well_stability_summary.csv")
GLOBAL_SEED_PATH = write_csv(GLOBAL_SEED_METRICS, ARTIFACT_DIR / "global_seed_metrics.csv")
GLOBAL_AGGREGATE_PATH = write_csv(
    GLOBAL_AGGREGATE_METRICS, ARTIFACT_DIR / "global_aggregate_metrics.csv"
)
GROUP_PATH = write_csv(OCCURRENCE_GROUP_SUMMARY, ARTIFACT_DIR / "occurrence_group_summary.csv")
ASSOCIATION_PATH = write_csv(
    OCCURRENCE_ASSOCIATIONS, ARTIFACT_DIR / "occurrence_feature_associations.csv"
)
BUCKET_PATH = write_csv(OCCURRENCE_BUCKETS, ARTIFACT_DIR / "occurrence_feature_buckets.csv")
DETAIL_PATH = write_csv(
    DETAILED_STRONG_PATHS,
    ARTIFACT_DIR / "detailed_strong_paths.csv.gz",
    gzip_output=True,
)

input_manifest_rows = list(INPUT_MANIFEST_CACHE)
for quality_row in QUALITY.to_dict(orient="records"):
    input_manifest_rows.extend(
        [
            {
                "kind": "raw_horizontal",
                "well": quality_row["well"],
                "filename": f"{quality_row['well']}__horizontal_well.csv",
                "raw_sha256": quality_row["horizontal_sha256"],
                "decompressed_sha256": quality_row["horizontal_sha256"],
            },
            {
                "kind": "raw_typewell",
                "well": quality_row["well"],
                "filename": f"{quality_row['well']}__typewell.csv",
                "raw_sha256": quality_row["typewell_sha256"],
                "decompressed_sha256": quality_row["typewell_sha256"],
            },
        ]
    )
INPUT_MANIFEST = pd.DataFrame(input_manifest_rows)
INPUT_MANIFEST_PATH = write_csv(INPUT_MANIFEST, ARTIFACT_DIR / "input_manifest.csv")

artifact_paths = [
    PARITY_PATH,
    REFERENCE_PATH,
    SEED_PATH,
    AGGREGATE_PATH,
    WELL_PATH,
    GLOBAL_SEED_PATH,
    GLOBAL_AGGREGATE_PATH,
    GROUP_PATH,
    ASSOCIATION_PATH,
    BUCKET_PATH,
    DETAIL_PATH,
    INPUT_MANIFEST_PATH,
]
ARTIFACT_MANIFEST = pd.DataFrame(artifact_manifest_rows(artifact_paths))
ARTIFACT_MANIFEST["schema_sha256"] = [
    schema_sha(frame)
    for frame in [
        PARITY_SUMMARY,
        REFERENCE_BY_WELL,
        SEED_METRICS,
        AGGREGATE_METRICS,
        WELL_STABILITY,
        GLOBAL_SEED_METRICS,
        GLOBAL_AGGREGATE_METRICS,
        OCCURRENCE_GROUP_SUMMARY,
        OCCURRENCE_ASSOCIATIONS,
        OCCURRENCE_BUCKETS,
        DETAILED_STRONG_PATHS,
        INPUT_MANIFEST,
    ]
]
ARTIFACT_MANIFEST_PATH = write_csv(ARTIFACT_MANIFEST, ARTIFACT_DIR / "artifact_manifest.csv")

TOTAL_SECONDS = time.time() - RUN_START
summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_multiseed_stability_audit",
    "created_at": datetime.now(UTC).isoformat(),
    "route": "pf_beam",
    "runtime": {
        "total_seconds": TOTAL_SECONDS,
        "parity_seconds": PARITY_SECONDS,
        "multiseed_seconds": MULTISEED_SECONDS,
        "num_workers": NUM_WORKERS,
        "gpu": False,
    },
    "execution": {
        "rows": int(len(REFERENCE_ROWS)),
        "wells": int(len(WELLS)),
        "strong_wells": int(len(STRONG_WELLS)),
        "algorithms": ["pf_ancc", "pf_z"],
        "particles": 600,
        "seed_count": SEED_COUNT,
        "new_seed_count": SEED_COUNT - 1,
        "active_variants": 2,
        "lightgbm_configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "inference": False,
        "submission": False,
    },
    "parity": PARITY_SUMMARY.to_dict(orient="records"),
    "focus_well": FOCUS_SUMMARY,
    "global_original_seed": GLOBAL_SEED_METRICS.loc[GLOBAL_SEED_METRICS["seed_index"] == 0].to_dict(
        orient="records"
    ),
    "global_aggregate_metrics": GLOBAL_AGGREGATE_METRICS.to_dict(orient="records"),
    "occurrence_groups": OCCURRENCE_GROUP_SUMMARY.to_dict(orient="records"),
    "input_manifest_sha256": sha256_path(INPUT_MANIFEST_PATH),
    "artifact_manifest_sha256": sha256_path(ARTIFACT_MANIFEST_PATH),
    "config_sha256": sha256_path(CONFIG_PATH),
    "notes": [
        "All seed paths and fixed aggregations were generated before target-side "
        "stability labels were computed.",
        "Model, inference, and submission SHA are not applicable to this train-side diagnostic.",
    ],
}
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"
SUMMARY_PATH.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n")
(OUTPUT_ROOT / "metrics.json").write_text(
    json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n"
)

print(
    json.dumps(
        {
            "status": summary["status"],
            "runtime": summary["runtime"],
            "execution": summary["execution"],
            "parity": summary["parity"],
            "focus_well": summary["focus_well"],
            "summary_path": str(SUMMARY_PATH),
            "artifact_manifest": str(ARTIFACT_MANIFEST_PATH),
        },
        indent=2,
        default=to_jsonable,
    ),
    flush=True,
)
