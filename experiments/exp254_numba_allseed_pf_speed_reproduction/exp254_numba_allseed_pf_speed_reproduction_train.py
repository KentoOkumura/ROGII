# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp254 Numba all-seed PF speed reproduction
#
# exp243 v3でexact parityが確認されたexp072 likelihood-PFを固定し、Python側の
# legacy seed loop、Numba all-seed loop、保存済みseed bankからのwarm candidate
# generationを分離して速度・parity・決定性を監査する。
#
# このnotebookはruntime基盤専用であり、true TVT、error、oracle、CV/LB、selector、
# inference、submissionを扱わない。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime, configuration, SHA, and memory helpers
# 3. Fixed exp243 input and representative-well helpers
# 4. Exact exp243 PF input assembly
# 5. Legacy single-seed and Numba all-seed PF kernels
# 6. Warm candidate and cache helpers
# 7. Setup and fixed benchmark contract
# 8. Input preflight and target-free well selection
# 9. JIT cold timing and PF benchmark execution
# 10. Parity, determinism, projections, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from numba import njit

# %% [markdown]
# ## 2. Runtime, configuration, SHA, and memory helpers

# %%
EXPERIMENT_NAME = "exp254_numba_allseed_pf_speed_reproduction"
PACKAGE_DIR = Path.cwd()
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found; checked={candidates}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def output_experiment_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT
    return ROOT / "experiments" / EXPERIMENT_NAME


def require_authorized_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "Kaggle Notebook execution is authoritative. Local execution requires "
        "the explicit EXPERIMENT_ALLOW_LOCAL=1 debug opt-in."
    )


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def exp072_stable_seed(*parts: Any, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def timed_call(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    implementation: str,
    measurement_kind: str,
    well: str | None,
    eval_rows: int | None,
    seed_count: int | None,
    candidate_spec_count: int | None,
    call: Callable[[], Any],
) -> Any:
    rss_before = peak_rss_mb()
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    rss_after = peak_rss_mb()
    rows.append(
        {
            "stage": stage,
            "implementation": implementation,
            "measurement_kind": measurement_kind,
            "well": well,
            "eval_rows": eval_rows,
            "seed_count": seed_count,
            "candidate_spec_count": candidate_spec_count,
            "seconds": float(elapsed),
            "peak_rss_mb_before": float(rss_before),
            "peak_rss_mb_after": float(rss_after),
        }
    )
    return result


def scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent": get_nested(config, "lineage.parent"),
        "row_candidates_sha": get_nested(
            config, "data.exp243_row_candidates.expected_sha256"
        ),
        "cluster_summary_sha": get_nested(
            config, "data.exp243_cluster_summary.expected_sha256"
        ),
        "pf_runtime": get_nested(config, "model.runtime"),
        "benchmark": get_nested(config, "model.benchmark"),
        "parity_tolerance": get_nested(config, "audit.exact_parity_tolerance"),
        "saved_tolerance": get_nested(
            config, "audit.saved_exp243_float32_tolerance"
        ),
    }


# %% [markdown]
# ## 3. Fixed exp243 input and representative-well helpers

# %%
@dataclass(frozen=True)
class ArtifactMeta:
    path: Path
    bytes: int
    sha_kind: str
    sha256: str


@dataclass(frozen=True)
class ReferenceRows:
    well: str
    row_idx: np.ndarray
    saved_replay_mean: np.ndarray


def resolve_artifact(root: Path, spec: dict[str, Any]) -> ArtifactMeta:
    checked: list[str] = []
    for value in spec.get("paths") or []:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        checked.append(str(path))
        if not path.exists() or path.stat().st_size <= 0:
            continue
        sha_kind = str(spec.get("sha_kind") or "raw")
        actual = sha256_path(path, decompressed=sha_kind == "decompressed")
        expected = str(spec.get("expected_sha256") or "")
        if expected and actual != expected:
            raise RuntimeError(
                f"Input SHA mismatch for {path}: actual={actual} expected={expected}"
            )
        return ArtifactMeta(
            path=path,
            bytes=int(path.stat().st_size),
            sha_kind=sha_kind,
            sha256=actual,
        )
    raise FileNotFoundError(f"Artifact {spec.get('filename')} not found; checked={checked}")


def find_train_dir(root: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir") or "data/raw/train"))
    candidates = [
        configured if configured.is_absolute() else root / configured,
        Path("/kaggle/input/rogii-wellbore-geology-prediction/train"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"),
    ]
    for candidate in candidates:
        if candidate.exists() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    raise FileNotFoundError(f"Raw train directory not found; checked={candidates}")


def load_cluster_summary(
    meta: ArtifactMeta, config: dict[str, Any]
) -> pd.DataFrame:
    frame = pd.read_csv(meta.path, low_memory=False)
    required = ["well", "k", "eval_rows", "seed_base"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"cluster summary missing columns: {missing}")
    frame["well"] = frame["well"].astype(str)
    k8 = frame.loc[pd.to_numeric(frame["k"], errors="coerce") == 8, required].copy()
    if k8["well"].duplicated().any():
        raise ValueError("exp243 K8 cluster summary has duplicate wells")
    k8["eval_rows"] = pd.to_numeric(k8["eval_rows"], errors="raise").astype(np.int64)
    k8["seed_base"] = pd.to_numeric(k8["seed_base"], errors="raise").astype(np.int64)
    k8 = k8.sort_values("well").reset_index(drop=True)
    expected_wells = int(get_nested(config, "data.expected_wells"))
    if len(k8) != expected_wells:
        raise ValueError(f"expected {expected_wells} exp243 wells, found {len(k8)}")
    return k8


def select_representative_wells(
    cluster_summary: pd.DataFrame, quantiles: list[float]
) -> pd.DataFrame:
    available = cluster_summary.copy()
    selected: list[dict[str, Any]] = []
    for quantile in quantiles:
        target = int(
            np.quantile(
                cluster_summary["eval_rows"].to_numpy(np.int64),
                float(quantile),
                method="nearest",
            )
        )
        ranked = available.assign(
            distance_to_quantile=(available["eval_rows"] - target).abs()
        ).sort_values(["distance_to_quantile", "eval_rows", "well"])
        if ranked.empty:
            raise RuntimeError("Could not select distinct representative wells")
        row = ranked.iloc[0]
        selected.append(
            {
                "selection_kind": "fixed_eval_length_quantile",
                "length_quantile": float(quantile),
                "quantile_target_eval_rows": target,
                "well": str(row["well"]),
                "eval_rows": int(row["eval_rows"]),
                "seed_base_from_exp243": int(row["seed_base"]),
            }
        )
        available = available.loc[available["well"] != row["well"]].copy()
    return pd.DataFrame(selected)


def load_saved_reference_rows(
    meta: ArtifactMeta,
    selected: pd.DataFrame,
) -> dict[str, ReferenceRows]:
    required = ["well", "row_idx", "pf_replay_likpf_mean"]
    header = pd.read_csv(meta.path, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"exp243 row candidates missing columns: {missing}")
    selected_wells = set(selected["well"].astype(str))
    pieces: dict[str, list[pd.DataFrame]] = {well: [] for well in selected_wells}
    for chunk in pd.read_csv(
        meta.path,
        usecols=required,
        dtype={"well": str},
        chunksize=200_000,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["well"].isin(selected_wells)]
        if chunk.empty:
            continue
        for well, group in chunk.groupby("well", sort=False):
            pieces[str(well)].append(group[["row_idx", "pf_replay_likpf_mean"]].copy())
    references: dict[str, ReferenceRows] = {}
    expected_rows = selected.set_index("well")["eval_rows"].to_dict()
    for well in sorted(selected_wells):
        if not pieces[well]:
            raise ValueError(f"exp243 row candidates do not contain selected well {well}")
        frame = pd.concat(pieces[well], ignore_index=True)
        frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
        frame["pf_replay_likpf_mean"] = pd.to_numeric(
            frame["pf_replay_likpf_mean"], errors="raise"
        ).astype(np.float32)
        frame = frame.sort_values("row_idx").reset_index(drop=True)
        if frame["row_idx"].duplicated().any():
            raise ValueError(f"duplicate exp243 row_idx for {well}")
        if len(frame) != int(expected_rows[well]):
            raise ValueError(
                f"exp243 row count mismatch for {well}: {len(frame)} != {expected_rows[well]}"
            )
        references[well] = ReferenceRows(
            well=well,
            row_idx=frame["row_idx"].to_numpy(np.int64),
            saved_replay_mean=frame["pf_replay_likpf_mean"].to_numpy(np.float32),
        )
    return references


# %% [markdown]
# ## 4. Exact exp243 PF input assembly

# %%
@dataclass(frozen=True)
class PfInput:
    well: str
    md: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    gr_grid: np.ndarray
    grid_min: float
    grid_step: float
    gr_sigma: float
    last_surface: float
    initial_rate: float
    seed_base: int
    saved_replay_mean: np.ndarray
    horizontal_path: Path
    typewell_path: Path
    horizontal_sha256: str
    typewell_sha256: str
    pf_input_sha256: str


def numeric64(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def initial_surface_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(30)
    tvt = numeric64(tail, "TVT_input")
    z = numeric64(tail, "Z")
    md = numeric64(tail, "MD")
    dm = np.diff(md)
    ds = np.diff(tvt) + np.diff(z)
    finite = np.isfinite(dm) & np.isfinite(ds) & (dm > 0.0)
    if int(finite.sum()) < 3:
        return 0.0
    return float(np.median(ds[finite] / dm[finite]))


def build_pf_input(
    well: str,
    reference: ReferenceRows,
    exp243_seed_base: int,
    train_dir: Path,
    config: dict[str, Any],
) -> PfInput:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"raw horizontal/typewell input missing for {well}")
    horizontal_sha = sha256_path(horizontal_path)
    typewell_sha = sha256_path(typewell_path)
    required_horizontal = ["MD", "Z", "GR", "TVT_input"]
    horizontal_header = pd.read_csv(horizontal_path, nrows=0).columns.tolist()
    missing_horizontal = [
        column for column in required_horizontal if column not in horizontal_header
    ]
    if missing_horizontal:
        raise ValueError(f"{horizontal_path} missing {missing_horizontal}")
    horizontal = pd.read_csv(
        horizontal_path, usecols=required_horizontal, low_memory=False
    )
    typewell = pd.read_csv(typewell_path, low_memory=False).sort_values("TVT").reset_index(
        drop=True
    )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError(f"{typewell_path} must contain TVT and GR")
    eval_index = reference.row_idx
    if len(eval_index) == 0 or np.any(eval_index < 0) or np.any(eval_index >= len(horizontal)):
        raise ValueError(f"invalid exp243 evaluation row index for {well}")
    masked = horizontal.iloc[: int(eval_index[-1]) + 1].copy()
    masked.loc[eval_index, "TVT_input"] = np.nan
    typewell_tvt = numeric64(typewell, "TVT")
    typewell_gr_series = pd.to_numeric(typewell["GR"], errors="coerce")
    typewell_gr_mean = float(typewell_gr_series.mean())
    typewell_gr = typewell_gr_series.fillna(typewell_gr_mean).to_numpy(np.float64)
    horizontal_gr_series = pd.to_numeric(masked["GR"], errors="coerce")
    horizontal_gr = (
        horizontal_gr_series.interpolate(limit_direction="both")
        .fillna(typewell_gr_mean)
        .to_numpy(np.float64)
    )
    known = masked.loc[masked["TVT_input"].notna()]
    known_tvt = numeric64(known, "TVT_input")
    known_gr = (
        pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    )
    typewell_gr_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    runtime = get_nested(config, "model.runtime") or {}
    sigma = float(
        np.clip(
            np.nanstd(known_gr - typewell_gr_at_known),
            float(runtime.get("gr_sigma_min", 10.0)),
            float(runtime.get("gr_sigma_max", 60.0)),
        )
    )
    prefix = masked.iloc[: int(eval_index[0])]
    known_prefix = prefix.loc[prefix["TVT_input"].notna()]
    if len(known_prefix) < 3:
        raise ValueError(f"insufficient known prefix for {well}")
    last_prefix = known_prefix.iloc[-1]
    last_surface = float(last_prefix["TVT_input"]) + float(last_prefix["Z"])
    initial_rate = initial_surface_velocity(known_prefix)
    grid_step = float(runtime.get("grid_step", 0.2))
    grid_min = float(np.nanmin(typewell_tvt))
    grid_max = float(np.nanmax(typewell_tvt))
    tvt_grid = np.arange(grid_min, grid_max + grid_step, grid_step, dtype=np.float64)
    gr_grid = np.interp(tvt_grid, typewell_tvt, typewell_gr).astype(np.float64)
    seed_base = exp072_stable_seed("likpf", "train", well)
    if seed_base != int(exp243_seed_base):
        raise ValueError(
            f"exp243 seed base mismatch for {well}: computed={seed_base} "
            f"saved={exp243_seed_base}"
        )
    md = numeric64(masked.loc[eval_index], "MD")
    z = numeric64(masked.loc[eval_index], "Z")
    gr = horizontal_gr[eval_index].astype(np.float64)
    scalars = np.asarray(
        [grid_min, grid_step, sigma, last_surface, initial_rate, float(seed_base)],
        dtype=np.float64,
    )
    pf_input_sha = array_bundle_sha256(
        md=md,
        z=z,
        gr=gr,
        gr_grid=gr_grid,
        scalars=scalars,
    )
    return PfInput(
        well=well,
        md=md,
        z=z,
        gr=gr,
        gr_grid=gr_grid,
        grid_min=grid_min,
        grid_step=grid_step,
        gr_sigma=max(sigma, 1.0e-6),
        last_surface=last_surface,
        initial_rate=initial_rate,
        seed_base=seed_base,
        saved_replay_mean=reference.saved_replay_mean,
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        horizontal_sha256=horizontal_sha,
        typewell_sha256=typewell_sha,
        pf_input_sha256=pf_input_sha,
    )


# %% [markdown]
# ## 5. Legacy single-seed and Numba all-seed PF kernels

# %%
@njit(cache=False, nogil=True)
def _interp1(grid: np.ndarray, value: float, vmin: float, step: float) -> float:
    x = (value - vmin) / step
    index = int(np.floor(x))
    if index <= 0:
        return grid[0]
    if index >= len(grid) - 1:
        return grid[-1]
    fraction = x - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=False, nogil=True)
def _legacy_single_seed_pf(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gg: np.ndarray,
    vmin: float,
    step: float,
    gs: float,
    last_surface: float,
    init_rate: float,
    n_particles: int,
    seed: int,
    momentum: float,
    velocity_noise: float,
    position_noise: float,
    resample_pos_noise: float,
    resample_velocity_noise: float,
    resample_threshold: float,
    init_spread: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    n_rows = len(md_v)
    prediction = np.empty(n_rows, dtype=np.float64)
    ess_by_row = np.zeros(n_rows, dtype=np.float64)
    resampled_by_row = np.zeros(n_rows, dtype=np.float64)
    tmax = vmin + len(gg) * step
    np.random.seed(seed)
    pos = np.empty(n_particles, dtype=np.float64)
    rate = np.empty(n_particles, dtype=np.float64)
    weights = np.empty(n_particles, dtype=np.float64)
    for particle_index in range(n_particles):
        pos[particle_index] = last_surface + init_spread * np.random.randn()
        rate[particle_index] = init_rate + 0.01 * np.random.randn()
        weights[particle_index] = 1.0 / n_particles
    log_likelihood = 0.0
    previous_md = md_v[0] - 1.0
    for row_index in range(n_rows):
        delta_md = md_v[row_index] - previous_md
        if delta_md < 1.0:
            delta_md = 1.0
        for particle_index in range(n_particles):
            rate[particle_index] = (
                momentum * rate[particle_index] + velocity_noise * np.random.randn()
            )
            pos[particle_index] += (
                rate[particle_index] * delta_md + position_noise * np.random.randn()
            )
            tvt_particle = pos[particle_index] - z_v[row_index]
            if tvt_particle < vmin - 100.0:
                tvt_particle = vmin - 100.0
            if tvt_particle > tmax + 100.0:
                tvt_particle = tmax + 100.0
            pos[particle_index] = tvt_particle + z_v[row_index]
        average_likelihood = 0.0
        for particle_index in range(n_particles):
            expected_gr = _interp1(
                gg, pos[particle_index] - z_v[row_index], vmin, step
            )
            residual = (gr_v[row_index] - expected_gr) / gs
            residual2 = residual * residual
            if residual2 > 600.0:
                residual2 = 600.0
            likelihood = np.exp(-0.5 * residual2)
            if likelihood < 1.0e-300:
                likelihood = 1.0e-300
            average_likelihood += weights[particle_index] * likelihood
            weights[particle_index] *= likelihood
        if average_likelihood < 1.0e-300:
            average_likelihood = 1.0e-300
        log_likelihood += np.log(average_likelihood)
        weight_sum = 0.0
        for particle_index in range(n_particles):
            weight_sum += weights[particle_index]
        if weight_sum > 0.0:
            for particle_index in range(n_particles):
                weights[particle_index] /= weight_sum
        else:
            for particle_index in range(n_particles):
                weights[particle_index] = 1.0 / n_particles
        inverse_ess = 0.0
        for particle_index in range(n_particles):
            inverse_ess += weights[particle_index] * weights[particle_index]
        ess = 1.0 / inverse_ess
        ess_by_row[row_index] = ess
        if ess < resample_threshold * n_particles:
            cumulative = np.empty(n_particles, dtype=np.float64)
            cumulative_weight = 0.0
            for particle_index in range(n_particles):
                cumulative_weight += weights[particle_index]
                cumulative[particle_index] = cumulative_weight
            draw0 = np.random.uniform(0.0, 1.0 / n_particles)
            new_pos = np.empty(n_particles, dtype=np.float64)
            new_rate = np.empty(n_particles, dtype=np.float64)
            cumulative_index = 0
            for particle_index in range(n_particles):
                draw = draw0 + particle_index / n_particles
                while (
                    cumulative_index < n_particles - 1
                    and cumulative[cumulative_index] < draw
                ):
                    cumulative_index += 1
                new_pos[particle_index] = (
                    pos[cumulative_index] + resample_pos_noise * np.random.randn()
                )
                new_rate[particle_index] = (
                    rate[cumulative_index]
                    + resample_velocity_noise * np.random.randn()
                )
            for particle_index in range(n_particles):
                pos[particle_index] = new_pos[particle_index]
                rate[particle_index] = new_rate[particle_index]
                weights[particle_index] = 1.0 / n_particles
            resampled_by_row[row_index] = 1.0
        estimate = 0.0
        for particle_index in range(n_particles):
            estimate += weights[particle_index] * (
                pos[particle_index] - z_v[row_index]
            )
        prediction[row_index] = estimate
        previous_md = md_v[row_index]
    return prediction, log_likelihood, ess_by_row, resampled_by_row


@njit(cache=False, nogil=True)
def _exp243_numba_allseed_pf(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gg: np.ndarray,
    vmin: float,
    step: float,
    gs: float,
    last_surface: float,
    init_rate: float,
    n_particles: int,
    n_seeds: int,
    seed_base: int,
    momentum: float,
    velocity_noise: float,
    position_noise: float,
    resample_pos_noise: float,
    resample_velocity_noise: float,
    resample_threshold: float,
    init_spread: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(md_v)
    predictions = np.empty((n_seeds, n_rows), dtype=np.float64)
    log_likelihoods = np.empty(n_seeds, dtype=np.float64)
    ess_accum = np.zeros(n_rows, dtype=np.float64)
    resampled_accum = np.zeros(n_rows, dtype=np.float64)
    tmax = vmin + len(gg) * step
    for seed_index in range(n_seeds):
        np.random.seed(seed_base + seed_index)
        pos = np.empty(n_particles, dtype=np.float64)
        rate = np.empty(n_particles, dtype=np.float64)
        weights = np.empty(n_particles, dtype=np.float64)
        for particle_index in range(n_particles):
            pos[particle_index] = last_surface + init_spread * np.random.randn()
            rate[particle_index] = init_rate + 0.01 * np.random.randn()
            weights[particle_index] = 1.0 / n_particles
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row_index in range(n_rows):
            delta_md = md_v[row_index] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle_index in range(n_particles):
                rate[particle_index] = (
                    momentum * rate[particle_index]
                    + velocity_noise * np.random.randn()
                )
                pos[particle_index] += (
                    rate[particle_index] * delta_md
                    + position_noise * np.random.randn()
                )
                tvt_particle = pos[particle_index] - z_v[row_index]
                if tvt_particle < vmin - 100.0:
                    tvt_particle = vmin - 100.0
                if tvt_particle > tmax + 100.0:
                    tvt_particle = tmax + 100.0
                pos[particle_index] = tvt_particle + z_v[row_index]
            average_likelihood = 0.0
            for particle_index in range(n_particles):
                expected_gr = _interp1(
                    gg, pos[particle_index] - z_v[row_index], vmin, step
                )
                residual = (gr_v[row_index] - expected_gr) / gs
                residual2 = residual * residual
                if residual2 > 600.0:
                    residual2 = 600.0
                likelihood = np.exp(-0.5 * residual2)
                if likelihood < 1.0e-300:
                    likelihood = 1.0e-300
                average_likelihood += weights[particle_index] * likelihood
                weights[particle_index] *= likelihood
            if average_likelihood < 1.0e-300:
                average_likelihood = 1.0e-300
            log_likelihood += np.log(average_likelihood)
            weight_sum = 0.0
            for particle_index in range(n_particles):
                weight_sum += weights[particle_index]
            if weight_sum > 0.0:
                for particle_index in range(n_particles):
                    weights[particle_index] /= weight_sum
            else:
                for particle_index in range(n_particles):
                    weights[particle_index] = 1.0 / n_particles
            inverse_ess = 0.0
            for particle_index in range(n_particles):
                inverse_ess += weights[particle_index] * weights[particle_index]
            ess = 1.0 / inverse_ess
            ess_accum[row_index] += ess
            if ess < resample_threshold * n_particles:
                cumulative = np.empty(n_particles, dtype=np.float64)
                cumulative_weight = 0.0
                for particle_index in range(n_particles):
                    cumulative_weight += weights[particle_index]
                    cumulative[particle_index] = cumulative_weight
                draw0 = np.random.uniform(0.0, 1.0 / n_particles)
                new_pos = np.empty(n_particles, dtype=np.float64)
                new_rate = np.empty(n_particles, dtype=np.float64)
                cumulative_index = 0
                for particle_index in range(n_particles):
                    draw = draw0 + particle_index / n_particles
                    while (
                        cumulative_index < n_particles - 1
                        and cumulative[cumulative_index] < draw
                    ):
                        cumulative_index += 1
                    new_pos[particle_index] = (
                        pos[cumulative_index]
                        + resample_pos_noise * np.random.randn()
                    )
                    new_rate[particle_index] = (
                        rate[cumulative_index]
                        + resample_velocity_noise * np.random.randn()
                    )
                for particle_index in range(n_particles):
                    pos[particle_index] = new_pos[particle_index]
                    rate[particle_index] = new_rate[particle_index]
                    weights[particle_index] = 1.0 / n_particles
                resampled_accum[row_index] += 1.0
            estimate = 0.0
            for particle_index in range(n_particles):
                estimate += weights[particle_index] * (
                    pos[particle_index] - z_v[row_index]
                )
            predictions[seed_index, row_index] = estimate
            previous_md = md_v[row_index]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        ess_accum / n_seeds,
        resampled_accum / n_seeds,
    )


def pf_arguments(pf_input: PfInput, config: dict[str, Any]) -> tuple[Any, ...]:
    runtime = get_nested(config, "model.runtime") or {}
    return (
        pf_input.md,
        pf_input.z,
        pf_input.gr,
        pf_input.gr_grid,
        pf_input.grid_min,
        pf_input.grid_step,
        pf_input.gr_sigma,
        pf_input.last_surface,
        pf_input.initial_rate,
        int(runtime.get("particles", 500)),
    )


def pf_tail_arguments(config: dict[str, Any]) -> tuple[float, ...]:
    runtime = get_nested(config, "model.runtime") or {}
    return (
        float(runtime.get("momentum", 0.998)),
        float(runtime.get("velocity_noise", 0.002)),
        float(runtime.get("position_noise", 0.005)),
        float(runtime.get("resample_pos_noise", 0.10)),
        float(runtime.get("resample_velocity_noise", 0.001)),
        float(runtime.get("resample_threshold", 0.5)),
        float(runtime.get("init_spread", 4.5)),
    )


def run_legacy_seed_loop(
    pf_input: PfInput, n_seeds: int, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.empty((n_seeds, len(pf_input.md)), dtype=np.float64)
    log_likelihoods = np.empty(n_seeds, dtype=np.float64)
    ess_accum = np.zeros(len(pf_input.md), dtype=np.float64)
    resampled_accum = np.zeros(len(pf_input.md), dtype=np.float64)
    args = pf_arguments(pf_input, config)
    tail = pf_tail_arguments(config)
    for seed_index in range(n_seeds):
        prediction, log_likelihood, ess, resampled = _legacy_single_seed_pf(
            *args, pf_input.seed_base + seed_index, *tail
        )
        predictions[seed_index] = prediction
        log_likelihoods[seed_index] = log_likelihood
        ess_accum += ess
        resampled_accum += resampled
    return (
        predictions,
        log_likelihoods,
        ess_accum / n_seeds,
        resampled_accum / n_seeds,
    )


def run_numba_allseed(
    pf_input: PfInput, n_seeds: int, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _exp243_numba_allseed_pf(
        *pf_arguments(pf_input, config),
        n_seeds,
        pf_input.seed_base,
        *pf_tail_arguments(config),
    )


def compile_pf_kernels(timing_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    runtime = get_nested(config, "model.runtime") or {}
    md = np.asarray([1.0, 2.0], dtype=np.float64)
    z = np.zeros(2, dtype=np.float64)
    gr = np.asarray([100.0, 101.0], dtype=np.float64)
    grid = np.linspace(90.0, 110.0, 32, dtype=np.float64)
    common = (
        md,
        z,
        gr,
        grid,
        0.0,
        1.0,
        30.0,
        50.0,
        0.0,
        4,
    )
    tail = (
        float(runtime.get("momentum", 0.998)),
        float(runtime.get("velocity_noise", 0.002)),
        float(runtime.get("position_noise", 0.005)),
        float(runtime.get("resample_pos_noise", 0.10)),
        float(runtime.get("resample_velocity_noise", 0.001)),
        float(runtime.get("resample_threshold", 0.5)),
        float(runtime.get("init_spread", 4.5)),
    )
    timed_call(
        timing_rows,
        stage="jit_cold_compile_dependency",
        implementation="shared_interp1",
        measurement_kind="measured_compile_plus_synthetic_execution",
        well=None,
        eval_rows=1,
        seed_count=None,
        candidate_spec_count=None,
        call=lambda: _interp1(grid, 1.5, 0.0, 1.0),
    )
    timed_call(
        timing_rows,
        stage="jit_cold_compile",
        implementation="legacy_single_seed_numba_kernel",
        measurement_kind="measured_compile_plus_synthetic_execution",
        well=None,
        eval_rows=2,
        seed_count=1,
        candidate_spec_count=None,
        call=lambda: _legacy_single_seed_pf(*common, 12345, *tail),
    )
    timed_call(
        timing_rows,
        stage="jit_cold_compile",
        implementation="numba_allseed_kernel",
        measurement_kind="measured_compile_plus_synthetic_execution",
        well=None,
        eval_rows=2,
        seed_count=1,
        candidate_spec_count=None,
        call=lambda: _exp243_numba_allseed_pf(*common, 1, 12345, *tail),
    )


# %% [markdown]
# ## 6. Warm candidate and cache helpers

# %%
@dataclass(frozen=True)
class CandidateSpec:
    candidate_index: int
    aggregation: str
    temperature: float | None
    subset_count: int
    subset_offset: int
    subset_stride: int
    seed_indices: tuple[int, ...]


def make_candidate_specs(
    max_count: int, n_seeds: int, config: dict[str, Any]
) -> list[CandidateSpec]:
    benchmark = get_nested(config, "model.benchmark") or {}
    subset_counts = [int(value) for value in benchmark["subset_seed_counts"]]
    temperatures = [float(value) for value in benchmark["temperatures"]]
    strides = [int(value) for value in benchmark["subset_strides"]]
    rules = [str(value) for value in benchmark["aggregation_rules"]]
    full_seed_indices = tuple(range(n_seeds))
    specs: list[CandidateSpec] = [
        CandidateSpec(
            candidate_index=0,
            aggregation="mean",
            temperature=None,
            subset_count=n_seeds,
            subset_offset=0,
            subset_stride=1,
            seed_indices=full_seed_indices,
        )
    ]
    seen: set[tuple[Any, ...]] = {("mean", None, full_seed_indices)}
    attempt = 0
    while len(specs) < max_count:
        subset_count = min(subset_counts[attempt % len(subset_counts)], n_seeds)
        temperature = temperatures[(attempt // len(subset_counts)) % len(temperatures)]
        aggregation = rules[
            (attempt // (len(subset_counts) * len(temperatures))) % len(rules)
        ]
        stride = strides[
            (attempt // (len(subset_counts) * len(temperatures) * len(rules)))
            % len(strides)
        ]
        offset = (
            attempt
            // (
                len(subset_counts)
                * len(temperatures)
                * len(rules)
                * len(strides)
            )
        ) % n_seeds
        seed_indices = tuple(
            sorted((offset + stride * index) % n_seeds for index in range(subset_count))
        )
        effective_temperature = temperature if aggregation == "likelihood_weighted" else None
        key = (aggregation, effective_temperature, seed_indices)
        if key not in seen:
            seen.add(key)
            specs.append(
                CandidateSpec(
                    candidate_index=len(specs),
                    aggregation=aggregation,
                    temperature=effective_temperature,
                    subset_count=len(seed_indices),
                    subset_offset=offset,
                    subset_stride=stride,
                    seed_indices=seed_indices,
                )
            )
        attempt += 1
        if attempt > 1_000_000:
            raise RuntimeError("Could not construct the fixed candidate-spec bank")
    return specs


def candidate_specs_frame(specs: list[CandidateSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_index": spec.candidate_index,
                "aggregation": spec.aggregation,
                "temperature": spec.temperature,
                "subset_count": spec.subset_count,
                "subset_offset": spec.subset_offset,
                "subset_stride": spec.subset_stride,
                "seed_indices": " ".join(str(index) for index in spec.seed_indices),
            }
            for spec in specs
        ]
    )


def aggregate_candidates(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    specs: list[CandidateSpec],
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
    outputs = np.empty((len(specs), values.shape[1]), dtype=np.float64)
    for output_index, spec in enumerate(specs):
        indices = np.asarray(spec.seed_indices, dtype=np.int64)
        subset = values[indices]
        if spec.aggregation == "mean":
            outputs[output_index] = subset.mean(axis=0)
        elif spec.aggregation == "likelihood_weighted":
            if spec.temperature is None or spec.temperature <= 0.0:
                raise ValueError("likelihood-weighted candidate needs positive temperature")
            logits = likelihoods[indices] / spec.temperature
            logits -= np.max(logits)
            weights = np.exp(logits)
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                weights = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
            else:
                weights /= weight_sum
            outputs[output_index] = weights @ subset
        else:
            raise ValueError(f"unknown aggregation rule: {spec.aggregation}")
    return outputs


def save_seed_bank(
    path: Path, predictions: np.ndarray, log_likelihoods: np.ndarray
) -> None:
    np.savez(
        path,
        predictions=np.asarray(predictions, dtype=np.float64),
        log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
    )


def load_seed_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        predictions = np.asarray(payload["predictions"], dtype=np.float64)
        log_likelihoods = np.asarray(payload["log_likelihoods"], dtype=np.float64)
    return predictions, log_likelihoods


def load_and_validate_probe_summary(
    root: Path, config: dict[str, Any], contract_sha: str
) -> dict[str, Any]:
    full = get_nested(config, "execution.full_workload") or {}
    expected_sha = str(full.get("probe_summary_expected_sha256") or "")
    if not expected_sha:
        raise RuntimeError(
            "full_workload is fail-closed until probe_summary_expected_sha256 is pinned"
        )
    spec = {
        "filename": "exp254 probe summary",
        "paths": full.get("probe_summary_candidates") or [],
        "sha_kind": "raw",
        "expected_sha256": expected_sha,
    }
    meta = resolve_artifact(root, spec)
    payload = json.loads(meta.path.read_text())
    if bool(full.get("require_probe_passed", True)) and not bool(
        get_nested(payload, "guards.probe_passed")
    ):
        raise RuntimeError("Pinned probe summary did not pass every required guard")
    if str(payload.get("scientific_contract_sha256")) != contract_sha:
        raise RuntimeError("Pinned probe summary scientific contract SHA mismatch")
    return {**payload, "pinned_probe_summary_sha256": meta.sha256}


# %% [markdown]
# ## 7. Setup and fixed benchmark contract

# %%
require_authorized_runtime()
config_path = find_config_path()
config = read_yaml(config_path)
experiment_dir = output_experiment_dir()
artifacts_dir = experiment_dir / "artifacts"
cache_dir = artifacts_dir / "seed_bank_cache"
artifacts_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)
started_wall = time.time()
mode = str(get_nested(config, "execution.mode") or "probe")
if mode not in {"probe", "full_workload"}:
    raise ValueError(f"execution.mode must be probe or full_workload, got {mode}")

benchmark = get_nested(config, "model.benchmark") or {}
seed_counts = [int(value) for value in benchmark["seed_counts"]]
candidate_spec_counts = [int(value) for value in benchmark["candidate_spec_counts"]]
if seed_counts != [1, 4, 16, 32, 64, 128]:
    raise ValueError("seed count grid must remain [1, 4, 16, 32, 64, 128]")
if candidate_spec_counts != [1, 10, 100, 300]:
    raise ValueError("candidate spec grid must remain [1, 10, 100, 300]")
if int(get_nested(config, "model.runtime.particles")) != 500:
    raise ValueError("exp254 fixes particles=500")
if int(get_nested(config, "model.runtime.seed_count")) != 128:
    raise ValueError("exp254 fixes seed_count=128")
if int(benchmark.get("python_processes", 1)) != 1:
    raise ValueError("exp254 requires one Python process")
numba.set_num_threads(int(benchmark.get("numba_threads", 1)))
if numba.get_num_threads() != 1:
    raise RuntimeError(f"Numba thread guard failed: {numba.get_num_threads()}")

contract = scientific_contract(config)
contract_sha = sha256_json(contract)
pinned_probe_summary: dict[str, Any] | None = None
if mode == "full_workload":
    pinned_probe_summary = load_and_validate_probe_summary(ROOT, config, contract_sha)

display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "mode": mode,
        "parent": get_nested(config, "lineage.parent"),
        "particles": get_nested(config, "model.runtime.particles"),
        "seed_counts": seed_counts,
        "candidate_spec_counts": candidate_spec_counts,
        "processes": benchmark.get("python_processes"),
        "numba_threads": numba.get_num_threads(),
        "GPU": False,
        "LightGBM configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent/control retraining": False,
        "scientific_contract_sha256": contract_sha,
        "target_usage": "none; target/error/oracle columns are never loaded",
    }
)

# %% [markdown]
# ## 8. Input preflight and target-free well selection

# %%
row_meta = resolve_artifact(ROOT, get_nested(config, "data.exp243_row_candidates"))
cluster_meta = resolve_artifact(ROOT, get_nested(config, "data.exp243_cluster_summary"))
cluster_summary = load_cluster_summary(cluster_meta, config)
train_dir = find_train_dir(ROOT, config)
if mode == "probe":
    quantiles = [
        float(value)
        for value in get_nested(
            config, "model.benchmark.representative_length_quantiles"
        )
    ]
    selected_wells = select_representative_wells(cluster_summary, quantiles)
else:
    selected_wells = cluster_summary.rename(
        columns={"seed_base": "seed_base_from_exp243"}
    ).copy()
    selected_wells["selection_kind"] = "full_fixed_exp243_workload"
    selected_wells["length_quantile"] = np.nan
    selected_wells["quantile_target_eval_rows"] = selected_wells["eval_rows"]
    selected_wells = selected_wells[
        [
            "selection_kind",
            "length_quantile",
            "quantile_target_eval_rows",
            "well",
            "eval_rows",
            "seed_base_from_exp243",
        ]
    ]
saved_references = load_saved_reference_rows(row_meta, selected_wells)

print("Fixed target-free workload")
display(selected_wells.head(30))
display(
    {
        "selected_wells": len(selected_wells),
        "selected_eval_rows": int(selected_wells["eval_rows"].sum()),
        "full_exp243_eval_rows": int(cluster_summary["eval_rows"].sum()),
        "exp243_row_candidates_sha": row_meta.sha256,
        "exp243_cluster_summary_sha": cluster_meta.sha256,
        "raw_train_dir": str(train_dir),
    }
)

# %% [markdown]
# ## 9. JIT cold timing and PF benchmark execution

# %%
timing_rows: list[dict[str, Any]] = []
parity_rows: list[dict[str, Any]] = []
cache_manifest_rows: list[dict[str, Any]] = []
input_manifest_rows: list[dict[str, Any]] = [
    {
        "input_kind": "exp243_row_candidates",
        "well": None,
        "path": str(row_meta.path),
        "bytes": row_meta.bytes,
        "sha_kind": row_meta.sha_kind,
        "sha256": row_meta.sha256,
    },
    {
        "input_kind": "exp243_cluster_summary",
        "well": None,
        "path": str(cluster_meta.path),
        "bytes": cluster_meta.bytes,
        "sha_kind": cluster_meta.sha_kind,
        "sha256": cluster_meta.sha256,
    },
]
compile_pf_kernels(timing_rows, config)
max_seed_count = max(seed_counts)
max_candidate_count = max(candidate_spec_counts)
candidate_specs = make_candidate_specs(max_candidate_count, max_seed_count, config)
candidate_spec_table = candidate_specs_frame(candidate_specs)
legacy_elapsed_total = 0.0
legacy_budget = float(benchmark.get("legacy_wall_time_budget_seconds", 14400))

for workload_index, selected_row in selected_wells.reset_index(drop=True).iterrows():
    well = str(selected_row["well"])
    pf_input = build_pf_input(
        well,
        saved_references[well],
        int(selected_row["seed_base_from_exp243"]),
        train_dir,
        config,
    )
    input_manifest_rows.extend(
        [
            {
                "input_kind": "raw_horizontal",
                "well": well,
                "path": str(pf_input.horizontal_path),
                "bytes": int(pf_input.horizontal_path.stat().st_size),
                "sha_kind": "raw",
                "sha256": pf_input.horizontal_sha256,
            },
            {
                "input_kind": "raw_typewell",
                "well": well,
                "path": str(pf_input.typewell_path),
                "bytes": int(pf_input.typewell_path.stat().st_size),
                "sha_kind": "raw",
                "sha256": pf_input.typewell_sha256,
            },
            {
                "input_kind": "assembled_pf_input",
                "well": well,
                "path": None,
                "bytes": int(
                    pf_input.md.nbytes
                    + pf_input.z.nbytes
                    + pf_input.gr.nbytes
                    + pf_input.gr_grid.nbytes
                ),
                "sha_kind": "array_content",
                "sha256": pf_input.pf_input_sha256,
            },
        ]
    )
    print(
        f"[workload] {workload_index + 1}/{len(selected_wells)} well={well} "
        f"eval_rows={len(pf_input.md)} mode={mode}",
        flush=True,
    )
    if mode == "full_workload":
        allseed = timed_call(
            timing_rows,
            stage="pf_core",
            implementation="numba_allseed",
            measurement_kind="measured_full_workload",
            well=well,
            eval_rows=len(pf_input.md),
            seed_count=max_seed_count,
            candidate_spec_count=None,
            call=lambda current=pf_input: run_numba_allseed(
                current, max_seed_count, config
            ),
        )
        predictions, log_likelihoods, ess_mean, resampled_rate = allseed
        saved_exact = np.array_equal(
            predictions.mean(axis=0).astype(np.float32),
            pf_input.saved_replay_mean,
        )
        parity_rows.append(
            {
                "well": well,
                "eval_rows": len(pf_input.md),
                "seed_count": max_seed_count,
                "comparison": "allseed_vs_saved_exp243_mean_float32",
                "trajectory_exact": None,
                "log_likelihood_exact": None,
                "final_mean_exact": bool(saved_exact),
                "ess_exact": None,
                "resampling_exact": None,
                "max_abs_trajectory_diff": None,
                "max_abs_log_likelihood_diff": None,
                "max_abs_final_mean_diff": float(
                    np.max(
                        np.abs(
                            predictions.mean(axis=0).astype(np.float32)
                            - pf_input.saved_replay_mean
                        )
                    )
                ),
                "legacy_sha256": None,
                "allseed_sha256": array_bundle_sha256(
                    predictions=predictions, log_likelihoods=log_likelihoods
                ),
            }
        )
        warm = timed_call(
            timing_rows,
            stage="warm_candidate_generation",
            implementation="cached_seed_bank_aggregation",
            measurement_kind="measured_full_workload",
            well=well,
            eval_rows=len(pf_input.md),
            seed_count=max_seed_count,
            candidate_spec_count=max_candidate_count,
            call=lambda p=predictions, ll=log_likelihoods: aggregate_candidates(
                p, ll, candidate_specs
            ),
        )
        parity_rows.append(
            {
                "well": well,
                "eval_rows": len(pf_input.md),
                "seed_count": max_seed_count,
                "comparison": "full_warm_candidate_content",
                "trajectory_exact": None,
                "log_likelihood_exact": None,
                "final_mean_exact": None,
                "ess_exact": None,
                "resampling_exact": None,
                "max_abs_trajectory_diff": None,
                "max_abs_log_likelihood_diff": None,
                "max_abs_final_mean_diff": None,
                "legacy_sha256": None,
                "allseed_sha256": array_bundle_sha256(candidates=warm),
            }
        )
        del predictions, log_likelihoods, ess_mean, resampled_rate, warm, allseed
        continue

    max_allseed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for seed_count in seed_counts:
        if legacy_elapsed_total >= legacy_budget:
            parity_rows.append(
                {
                    "well": well,
                    "eval_rows": len(pf_input.md),
                    "seed_count": seed_count,
                    "comparison": "legacy_vs_allseed",
                    "trajectory_exact": False,
                    "log_likelihood_exact": False,
                    "final_mean_exact": False,
                    "ess_exact": False,
                    "resampling_exact": False,
                    "max_abs_trajectory_diff": None,
                    "max_abs_log_likelihood_diff": None,
                    "max_abs_final_mean_diff": None,
                    "legacy_sha256": None,
                    "allseed_sha256": None,
                    "status": "legacy_wall_time_budget_exhausted",
                }
            )
            continue
        legacy_started = time.perf_counter()
        legacy = timed_call(
            timing_rows,
            stage="pf_core",
            implementation="python_legacy_seed_loop",
            measurement_kind="measured_probe",
            well=well,
            eval_rows=len(pf_input.md),
            seed_count=seed_count,
            candidate_spec_count=None,
            call=lambda count=seed_count, current=pf_input: run_legacy_seed_loop(
                current, count, config
            ),
        )
        legacy_elapsed_total += time.perf_counter() - legacy_started
        allseed = timed_call(
            timing_rows,
            stage="pf_core",
            implementation="numba_allseed",
            measurement_kind="measured_probe",
            well=well,
            eval_rows=len(pf_input.md),
            seed_count=seed_count,
            candidate_spec_count=None,
            call=lambda count=seed_count, current=pf_input: run_numba_allseed(
                current, count, config
            ),
        )
        legacy_predictions, legacy_likelihoods, legacy_ess, legacy_resampled = legacy
        all_predictions, all_likelihoods, all_ess, all_resampled = allseed
        legacy_mean = legacy_predictions.mean(axis=0)
        allseed_mean = all_predictions.mean(axis=0)
        legacy_sha = array_bundle_sha256(
            predictions=legacy_predictions, log_likelihoods=legacy_likelihoods
        )
        allseed_sha = array_bundle_sha256(
            predictions=all_predictions, log_likelihoods=all_likelihoods
        )
        trajectory_exact = np.array_equal(legacy_predictions, all_predictions)
        likelihood_exact = np.array_equal(legacy_likelihoods, all_likelihoods)
        mean_exact = np.array_equal(legacy_mean, allseed_mean)
        ess_exact = np.array_equal(legacy_ess, all_ess)
        resampling_exact = np.array_equal(legacy_resampled, all_resampled)
        parity_rows.append(
            {
                "well": well,
                "eval_rows": len(pf_input.md),
                "seed_count": seed_count,
                "comparison": "legacy_vs_allseed",
                "trajectory_exact": bool(trajectory_exact),
                "log_likelihood_exact": bool(likelihood_exact),
                "final_mean_exact": bool(mean_exact),
                "ess_exact": bool(ess_exact),
                "resampling_exact": bool(resampling_exact),
                "max_abs_trajectory_diff": float(
                    np.max(np.abs(legacy_predictions - all_predictions))
                ),
                "max_abs_log_likelihood_diff": float(
                    np.max(np.abs(legacy_likelihoods - all_likelihoods))
                ),
                "max_abs_final_mean_diff": float(
                    np.max(np.abs(legacy_mean - allseed_mean))
                ),
                "legacy_sha256": legacy_sha,
                "allseed_sha256": allseed_sha,
                "status": "completed",
            }
        )
        if seed_count == max_seed_count:
            max_allseed = allseed
            saved_mean = allseed_mean.astype(np.float32)
            saved_exact = np.array_equal(saved_mean, pf_input.saved_replay_mean)
            parity_rows.append(
                {
                    "well": well,
                    "eval_rows": len(pf_input.md),
                    "seed_count": seed_count,
                    "comparison": "allseed_vs_saved_exp243_mean_float32",
                    "trajectory_exact": None,
                    "log_likelihood_exact": None,
                    "final_mean_exact": bool(saved_exact),
                    "ess_exact": None,
                    "resampling_exact": None,
                    "max_abs_trajectory_diff": None,
                    "max_abs_log_likelihood_diff": None,
                    "max_abs_final_mean_diff": float(
                        np.max(np.abs(saved_mean - pf_input.saved_replay_mean))
                    ),
                    "legacy_sha256": None,
                    "allseed_sha256": allseed_sha,
                    "status": "completed",
                }
            )
        del legacy, legacy_predictions, legacy_likelihoods, legacy_ess, legacy_resampled
        if seed_count != max_seed_count:
            del allseed, all_predictions, all_likelihoods, all_ess, all_resampled
    if max_allseed is None:
        raise RuntimeError(f"max-seed allseed benchmark did not complete for {well}")
    predictions, log_likelihoods, ess_mean, resampled_rate = max_allseed
    repeat_allseed = timed_call(
        timing_rows,
        stage="pf_core_repeat",
        implementation="numba_allseed",
        measurement_kind="measured_probe_repeat",
        well=well,
        eval_rows=len(pf_input.md),
        seed_count=max_seed_count,
        candidate_spec_count=None,
        call=lambda current=pf_input: run_numba_allseed(current, max_seed_count, config),
    )
    repeat_predictions, repeat_likelihoods, repeat_ess, repeat_resampled = repeat_allseed
    original_sha = array_bundle_sha256(
        predictions=predictions, log_likelihoods=log_likelihoods
    )
    repeat_sha = array_bundle_sha256(
        predictions=repeat_predictions, log_likelihoods=repeat_likelihoods
    )
    parity_rows.append(
        {
            "well": well,
            "eval_rows": len(pf_input.md),
            "seed_count": max_seed_count,
            "comparison": "allseed_repeat",
            "trajectory_exact": bool(np.array_equal(predictions, repeat_predictions)),
            "log_likelihood_exact": bool(
                np.array_equal(log_likelihoods, repeat_likelihoods)
            ),
            "final_mean_exact": bool(
                np.array_equal(
                    predictions.mean(axis=0), repeat_predictions.mean(axis=0)
                )
            ),
            "ess_exact": bool(np.array_equal(ess_mean, repeat_ess)),
            "resampling_exact": bool(
                np.array_equal(resampled_rate, repeat_resampled)
            ),
            "max_abs_trajectory_diff": float(
                np.max(np.abs(predictions - repeat_predictions))
            ),
            "max_abs_log_likelihood_diff": float(
                np.max(np.abs(log_likelihoods - repeat_likelihoods))
            ),
            "max_abs_final_mean_diff": float(
                np.max(
                    np.abs(
                        predictions.mean(axis=0) - repeat_predictions.mean(axis=0)
                    )
                )
            ),
            "legacy_sha256": original_sha,
            "allseed_sha256": repeat_sha,
            "status": "completed",
        }
    )
    cache_path = cache_dir / f"{well}_seed_bank.npz"
    timed_call(
        timing_rows,
        stage="cache_write",
        implementation="npz_uncompressed",
        measurement_kind="measured_probe",
        well=well,
        eval_rows=len(pf_input.md),
        seed_count=max_seed_count,
        candidate_spec_count=None,
        call=lambda path=cache_path, p=predictions, ll=log_likelihoods: save_seed_bank(
            path, p, ll
        ),
    )
    cache_predictions, cache_likelihoods = timed_call(
        timing_rows,
        stage="cache_read",
        implementation="npz_uncompressed",
        measurement_kind="measured_probe",
        well=well,
        eval_rows=len(pf_input.md),
        seed_count=max_seed_count,
        candidate_spec_count=None,
        call=lambda path=cache_path: load_seed_bank(path),
    )
    cache_content_sha = array_bundle_sha256(
        predictions=cache_predictions, log_likelihoods=cache_likelihoods
    )
    cache_roundtrip_exact = bool(
        np.array_equal(predictions, cache_predictions)
        and np.array_equal(log_likelihoods, cache_likelihoods)
    )
    cache_manifest_rows.append(
        {
            "well": well,
            "path": str(cache_path),
            "file_bytes": int(cache_path.stat().st_size),
            "file_sha256": sha256_path(cache_path),
            "content_sha256_before": original_sha,
            "content_sha256_after": cache_content_sha,
            "predictions_shape": "x".join(str(value) for value in predictions.shape),
            "predictions_dtype": str(predictions.dtype),
            "log_likelihoods_shape": "x".join(
                str(value) for value in log_likelihoods.shape
            ),
            "log_likelihoods_dtype": str(log_likelihoods.dtype),
            "roundtrip_exact": cache_roundtrip_exact,
        }
    )
    parity_rows.append(
        {
            "well": well,
            "eval_rows": len(pf_input.md),
            "seed_count": max_seed_count,
            "comparison": "cache_roundtrip",
            "trajectory_exact": bool(np.array_equal(predictions, cache_predictions)),
            "log_likelihood_exact": bool(
                np.array_equal(log_likelihoods, cache_likelihoods)
            ),
            "final_mean_exact": bool(
                np.array_equal(
                    predictions.mean(axis=0), cache_predictions.mean(axis=0)
                )
            ),
            "ess_exact": None,
            "resampling_exact": None,
            "max_abs_trajectory_diff": float(
                np.max(np.abs(predictions - cache_predictions))
            ),
            "max_abs_log_likelihood_diff": float(
                np.max(np.abs(log_likelihoods - cache_likelihoods))
            ),
            "max_abs_final_mean_diff": float(
                np.max(
                    np.abs(
                        predictions.mean(axis=0) - cache_predictions.mean(axis=0)
                    )
                )
            ),
            "legacy_sha256": original_sha,
            "allseed_sha256": cache_content_sha,
            "status": "completed",
        }
    )
    max_warm: np.ndarray | None = None
    max_warm_sha: str | None = None
    for spec_count in candidate_spec_counts:
        warm = timed_call(
            timing_rows,
            stage="warm_candidate_generation",
            implementation="cached_seed_bank_aggregation",
            measurement_kind="measured_probe",
            well=well,
            eval_rows=len(pf_input.md),
            seed_count=max_seed_count,
            candidate_spec_count=spec_count,
            call=partial(
                aggregate_candidates,
                cache_predictions,
                cache_likelihoods,
                candidate_specs[:spec_count],
            ),
        )
        warm_sha = array_bundle_sha256(candidates=warm)
        parity_rows.append(
            {
                "well": well,
                "eval_rows": len(pf_input.md),
                "seed_count": max_seed_count,
                "candidate_spec_count": spec_count,
                "comparison": "warm_candidate_content",
                "trajectory_exact": None,
                "log_likelihood_exact": None,
                "final_mean_exact": None,
                "ess_exact": None,
                "resampling_exact": None,
                "max_abs_trajectory_diff": None,
                "max_abs_log_likelihood_diff": None,
                "max_abs_final_mean_diff": None,
                "legacy_sha256": None,
                "allseed_sha256": warm_sha,
                "status": "completed",
            }
        )
        if spec_count == max_candidate_count:
            max_warm = warm
            max_warm_sha = warm_sha
        else:
            del warm
    if max_warm is None or max_warm_sha is None:
        raise RuntimeError(f"max warm candidate benchmark did not complete for {well}")
    repeat_warm = timed_call(
        timing_rows,
        stage="warm_candidate_repeat",
        implementation="cached_seed_bank_aggregation",
        measurement_kind="measured_probe_repeat",
        well=well,
        eval_rows=len(pf_input.md),
        seed_count=max_seed_count,
        candidate_spec_count=max_candidate_count,
        call=partial(
            aggregate_candidates,
            cache_predictions,
            cache_likelihoods,
            candidate_specs,
        ),
    )
    repeat_warm_sha = array_bundle_sha256(candidates=repeat_warm)
    parity_rows.append(
        {
            "well": well,
            "eval_rows": len(pf_input.md),
            "seed_count": max_seed_count,
            "candidate_spec_count": max_candidate_count,
            "comparison": "warm_candidate_repeat",
            "trajectory_exact": None,
            "log_likelihood_exact": None,
            "final_mean_exact": bool(np.array_equal(max_warm, repeat_warm)),
            "ess_exact": None,
            "resampling_exact": None,
            "max_abs_trajectory_diff": None,
            "max_abs_log_likelihood_diff": None,
            "max_abs_final_mean_diff": float(
                np.max(np.abs(max_warm - repeat_warm))
            ),
            "legacy_sha256": max_warm_sha,
            "allseed_sha256": repeat_warm_sha,
            "status": "completed",
        }
    )
    del (
        predictions,
        log_likelihoods,
        ess_mean,
        resampled_rate,
        repeat_allseed,
        repeat_predictions,
        repeat_likelihoods,
        repeat_ess,
        repeat_resampled,
        max_warm,
        repeat_warm,
    )

# %% [markdown]
# ## 10. Parity, determinism, projections, and generated artifacts

# %%
timings = pd.DataFrame(timing_rows)
parity = pd.DataFrame(parity_rows)
cache_manifest = pd.DataFrame(cache_manifest_rows)
input_manifest = pd.DataFrame(input_manifest_rows)
speedup_rows: list[dict[str, Any]] = []
if mode == "probe":
    legacy_times = timings.loc[
        (timings["stage"] == "pf_core")
        & (timings["implementation"] == "python_legacy_seed_loop")
        & (timings["measurement_kind"] == "measured_probe")
    ].set_index(["well", "seed_count"])["seconds"]
    allseed_times = timings.loc[
        (timings["stage"] == "pf_core")
        & (timings["implementation"] == "numba_allseed")
        & (timings["measurement_kind"] == "measured_probe")
    ].set_index(["well", "seed_count"])["seconds"]
    for key in legacy_times.index.intersection(allseed_times.index):
        legacy_seconds = float(legacy_times.loc[key])
        allseed_seconds = float(allseed_times.loc[key])
        speedup_rows.append(
            {
                "comparison": "legacy_seed_loop_over_numba_allseed",
                "well": key[0],
                "seed_count": int(key[1]),
                "candidate_spec_count": None,
                "numerator_seconds": legacy_seconds,
                "denominator_seconds": allseed_seconds,
                "ratio": legacy_seconds / allseed_seconds,
                "interpretation": "measured_same_pf_body_speed_ratio",
            }
        )
    warm_times = timings.loc[
        (timings["stage"] == "warm_candidate_generation")
        & (timings["implementation"] == "cached_seed_bank_aggregation")
        & (timings["measurement_kind"] == "measured_probe")
        & (timings["candidate_spec_count"] == max_candidate_count)
    ].set_index("well")["seconds"]
    allseed_128 = allseed_times.loc[
        allseed_times.index.get_level_values("seed_count") == max_seed_count
    ]
    for (well, _), allseed_seconds_value in allseed_128.items():
        if well not in warm_times.index:
            continue
        allseed_seconds = float(allseed_seconds_value)
        warm_seconds = float(warm_times.loc[well])
        speedup_rows.append(
            {
                "comparison": "allseed_pf_core_over_300candidate_warm_generation",
                "well": well,
                "seed_count": max_seed_count,
                "candidate_spec_count": max_candidate_count,
                "numerator_seconds": allseed_seconds,
                "denominator_seconds": warm_seconds,
                "ratio": allseed_seconds / warm_seconds,
                "interpretation": (
                    "measured_compute_ratio_only_not_a_300x_ensemble_speed_claim"
                ),
            }
        )
speedups = pd.DataFrame(speedup_rows)
projection_rows: list[dict[str, Any]] = []
if mode == "probe":
    representative_rows = int(selected_wells["eval_rows"].sum())
    full_rows = int(cluster_summary["eval_rows"].sum())
    projection_specs = [
        ("pf_core", "python_legacy_seed_loop", max_seed_count, None),
        ("pf_core", "numba_allseed", max_seed_count, None),
        (
            "warm_candidate_generation",
            "cached_seed_bank_aggregation",
            max_seed_count,
            max_candidate_count,
        ),
        ("cache_write", "npz_uncompressed", max_seed_count, None),
        ("cache_read", "npz_uncompressed", max_seed_count, None),
    ]
    for stage, implementation, seed_count, spec_count in projection_specs:
        mask = (
            (timings["stage"] == stage)
            & (timings["implementation"] == implementation)
            & (timings["measurement_kind"] == "measured_probe")
            & (timings["seed_count"] == seed_count)
        )
        if spec_count is not None:
            mask &= timings["candidate_spec_count"] == spec_count
        measured = timings.loc[mask]
        if measured.empty:
            continue
        measured_seconds = float(measured["seconds"].sum())
        seconds_per_eval_row = measured_seconds / representative_rows
        projection_rows.append(
            {
                "stage": stage,
                "implementation": implementation,
                "seed_count": seed_count,
                "candidate_spec_count": spec_count,
                "measurement_kind": "projection_from_three_fixed_length_quantile_wells",
                "measured_wells": int(len(selected_wells)),
                "measured_eval_rows": representative_rows,
                "measured_seconds": measured_seconds,
                "seconds_per_eval_row": seconds_per_eval_row,
                "projection_target_wells": int(
                    benchmark.get("projection_target_wells", 773)
                ),
                "projection_target_eval_rows": full_rows,
                "projected_seconds": seconds_per_eval_row * full_rows,
                "is_measured_full_runtime": False,
            }
        )
projections = pd.DataFrame(projection_rows)

if mode == "probe":
    legacy_rows = parity.loc[parity["comparison"] == "legacy_vs_allseed"]
    saved_rows = parity.loc[
        parity["comparison"] == "allseed_vs_saved_exp243_mean_float32"
    ]
    repeat_rows = parity.loc[parity["comparison"] == "allseed_repeat"]
    cache_rows = parity.loc[parity["comparison"] == "cache_roundtrip"]
    warm_repeat_rows = parity.loc[parity["comparison"] == "warm_candidate_repeat"]
    guards = {
        "legacy_vs_allseed_trajectory_exact": bool(
            len(legacy_rows) == len(selected_wells) * len(seed_counts)
            and legacy_rows["trajectory_exact"].fillna(False).all()
        ),
        "legacy_vs_allseed_log_likelihood_exact": bool(
            len(legacy_rows) == len(selected_wells) * len(seed_counts)
            and legacy_rows["log_likelihood_exact"].fillna(False).all()
        ),
        "legacy_vs_allseed_final_mean_exact": bool(
            len(legacy_rows) == len(selected_wells) * len(seed_counts)
            and legacy_rows["final_mean_exact"].fillna(False).all()
        ),
        "allseed_repeat_sha_exact": bool(
            len(repeat_rows) == len(selected_wells)
            and (repeat_rows["legacy_sha256"] == repeat_rows["allseed_sha256"]).all()
        ),
        "saved_exp243_mean_float32_exact": bool(
            len(saved_rows) == len(selected_wells)
            and saved_rows["final_mean_exact"].fillna(False).all()
        ),
        "cache_roundtrip_exact": bool(
            len(cache_rows) == len(selected_wells)
            and cache_rows["trajectory_exact"].fillna(False).all()
            and cache_rows["log_likelihood_exact"].fillna(False).all()
        ),
        "warm_candidate_repeat_sha_exact": bool(
            len(warm_repeat_rows) == len(selected_wells)
            and (
                warm_repeat_rows["legacy_sha256"]
                == warm_repeat_rows["allseed_sha256"]
            ).all()
        ),
    }
    guards["probe_passed"] = bool(all(guards.values()))
else:
    saved_rows = parity.loc[
        parity["comparison"] == "allseed_vs_saved_exp243_mean_float32"
    ]
    guards = {
        "pinned_probe_passed": bool(
            get_nested(pinned_probe_summary or {}, "guards.probe_passed")
        ),
        "saved_exp243_mean_float32_exact": bool(
            len(saved_rows) == len(selected_wells)
            and saved_rows["final_mean_exact"].fillna(False).all()
        ),
    }
    guards["full_workload_passed"] = bool(all(guards.values()))

outputs = get_nested(config, "audit.outputs") or {}
artifact_paths = {
    "selected_wells": artifacts_dir / outputs["probe_wells_filename"],
    "parity": artifacts_dir / outputs["parity_filename"],
    "timings": artifacts_dir / outputs["timings_filename"],
    "speedups": artifacts_dir / outputs["speedups_filename"],
    "candidate_specs": artifacts_dir / outputs["candidate_specs_filename"],
    "cache_manifest": artifacts_dir / outputs["cache_manifest_filename"],
    "projections": artifacts_dir / outputs["projections_filename"],
    "input_manifest": artifacts_dir / outputs["input_manifest_filename"],
    "summary": artifacts_dir / outputs["summary_filename"],
}
selected_wells.to_csv(artifact_paths["selected_wells"], index=False)
parity.to_csv(artifact_paths["parity"], index=False)
timings.to_csv(artifact_paths["timings"], index=False)
speedups.to_csv(artifact_paths["speedups"], index=False)
candidate_spec_table.to_csv(artifact_paths["candidate_specs"], index=False)
cache_manifest.to_csv(artifact_paths["cache_manifest"], index=False)
projections.to_csv(artifact_paths["projections"], index=False)
input_manifest.to_csv(artifact_paths["input_manifest"], index=False)

measured_full_seconds = None
if mode == "full_workload":
    measured_full_seconds = float(
        timings.loc[
            timings["measurement_kind"] == "measured_full_workload", "seconds"
        ].sum()
    )
projection_allseed_warm_seconds = None
projection_within_three_minutes = None
if mode == "probe" and not projections.empty:
    mask = projections["stage"].isin(["pf_core", "warm_candidate_generation"]) & (
        projections["implementation"]
        .isin(["numba_allseed", "cached_seed_bank_aggregation"])
    )
    projection_allseed_warm_seconds = float(projections.loc[mask, "projected_seconds"].sum())
    projection_within_three_minutes = bool(
        guards["probe_passed"] and projection_allseed_warm_seconds <= 180.0
    )
two_to_three_minute_hypothesis_supported = None
if mode == "full_workload":
    two_to_three_minute_hypothesis_supported = bool(
        guards["full_workload_passed"]
        and measured_full_seconds is not None
        and measured_full_seconds <= 180.0
    )

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": (
        "completed_probe_passed"
        if mode == "probe" and guards["probe_passed"]
        else "completed_probe_guard_failed"
        if mode == "probe"
        else "completed_full_workload_passed"
        if guards["full_workload_passed"]
        else "completed_full_workload_guard_failed"
    ),
    "route": "pf_beam",
    "mode": mode,
    "created_at_utc": datetime.now(UTC).isoformat(),
    "wall_runtime_seconds": float(time.time() - started_wall),
    "measured_full_workload_seconds": measured_full_seconds,
    "selected_wells": int(len(selected_wells)),
    "selected_eval_rows": int(selected_wells["eval_rows"].sum()),
    "full_exp243_wells": int(len(cluster_summary)),
    "full_exp243_eval_rows": int(cluster_summary["eval_rows"].sum()),
    "particles": int(get_nested(config, "model.runtime.particles")),
    "seed_counts": seed_counts,
    "candidate_spec_counts": candidate_spec_counts,
    "model_configs": 0,
    "folds": 0,
    "boosters": 0,
    "processes": 1,
    "numba_threads": int(numba.get_num_threads()),
    "peak_rss_mb": float(peak_rss_mb()),
    "scientific_contract": contract,
    "scientific_contract_sha256": contract_sha,
    "guards": guards,
    "speedup_comparisons": speedups.to_dict("records"),
    "projection_allseed_plus_warm_seconds": projection_allseed_warm_seconds,
    "projection_within_three_minutes": projection_within_three_minutes,
    "two_to_three_minute_hypothesis_supported": two_to_three_minute_hypothesis_supported,
    "two_to_three_minute_hypothesis_decision_basis": (
        "measured_full_workload"
        if mode == "full_workload"
        else "not_decided_from_probe_projection"
    ),
    "projection_warning": (
        "Probe projection is not a measured 773-well runtime. Full workload is required "
        "before accepting or rejecting the 2-3 minute claim."
        if mode == "probe"
        else None
    ),
    "pinned_probe_summary": pinned_probe_summary,
    "input_artifacts": {
        "exp243_row_candidates": to_jsonable(row_meta.__dict__),
        "exp243_cluster_summary": to_jsonable(cluster_meta.__dict__),
    },
    "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    "cache_content_sha_primary": True,
    "target_usage": "none",
    "inference": False,
    "submission": False,
}
write_json(artifact_paths["summary"], summary)
artifact_sha = {
    key: sha256_path(path)
    for key, path in artifact_paths.items()
    if key != "summary"
}
artifact_sha["summary"] = sha256_path(artifact_paths["summary"])
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "route": "pf_beam",
    "metric": "runtime_seconds_with_exact_parity_guards",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "mode": mode,
    "model_configs": 0,
    "folds": 0,
    "boosters": 0,
    "selected_wells": summary["selected_wells"],
    "selected_eval_rows": summary["selected_eval_rows"],
    "wall_runtime_seconds": summary["wall_runtime_seconds"],
    "measured_full_workload_seconds": measured_full_seconds,
    "peak_rss_mb": summary["peak_rss_mb"],
    "guards": guards,
    "scientific_contract_sha256": contract_sha,
    "artifact_sha256": artifact_sha,
    "artifacts": summary["artifacts"],
    "inference": False,
    "submission": False,
    "notes": (
        "Runtime foundation audit only. Probe projections are not measured full runtime; "
        "no target, model training, inference, or submission."
    ),
}
metrics_path = experiment_dir / "metrics.json"
write_json(metrics_path, metrics)

print("Parity and determinism guards")
display(pd.DataFrame([guards]))
print("Timing summary")
display(
    timings.groupby(
        ["stage", "implementation", "measurement_kind"], dropna=False
    )["seconds"]
    .agg(["count", "sum", "mean", "min", "max"])
    .reset_index()
)
if not projections.empty:
    print("773-well projections (not measured full runtime)")
    display(projections)
print("Generated artifacts")
print(json.dumps(to_jsonable(summary["artifacts"]), indent=2))
print("Artifact SHA256")
print(json.dumps(artifact_sha, indent=2))
print(
    json.dumps(
        {
            "status": summary["status"],
            "mode": mode,
            "wall_runtime_seconds": summary["wall_runtime_seconds"],
            "peak_rss_mb": summary["peak_rss_mb"],
            "guards": guards,
            "metrics_path": str(metrics_path),
        },
        indent=2,
    )
)
