# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp238 OOF with all 128 likelihood-PF seed paths
#
# This is a plotting-only companion to `exp238-oof-selector-confidence-probe`.
# For every train well it replays the exact exp072 likelihood-PF seed bank and
# plots all 128 seed trajectories with transparency. True TVT and the saved
# exp238 `lgb_mean` OOF stay opaque and above the PF bundle.
#
# The PF paths are diagnostic only. They are not blended into exp238, no model is
# trained, and no submission is generated. True TVT is used only after the PF
# trajectories have been frozen, for plot overlays and metric readouts.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Path resolution and input contract
# 3. Saved truth, exp072 mean, and exp238 OOF
# 4. Exact exp072 likelihood-PF replay helpers
# 5. Plot and metric helpers
# 6. Replay and plot all wells
# 7. Summary and generated outputs

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from joblib import Parallel, delayed
from numba import njit

EXPERIMENT_NAME = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
NOTEBOOK_KIND = "oof_likpf_128_paths_probe"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_{NOTEBOOK_KIND}"
PFBEAM_FILENAME = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP238_OOF_FILENAME = f"{EXPERIMENT_NAME}_final_oof_predictions.csv.gz"
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
READ_CHUNKSIZE = 300_000


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment_summary.md").exists() and (
            candidate / "experiments"
        ).exists():
            return candidate
    return current


REPO_ROOT = find_repo_root(Path.cwd())
EXP_DIR = REPO_ROOT / "experiments" / EXPERIMENT_NAME
if not EXP_DIR.exists() and Path.cwd().name == EXPERIMENT_NAME:
    EXP_DIR = Path.cwd()


def load_probe_config() -> tuple[dict[str, Any], Path]:
    candidates = [Path.cwd() / "config.yaml", EXP_DIR / "config.yaml"]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        probe = config.get("audit", {}).get(NOTEBOOK_KIND)
        if isinstance(probe, dict):
            return probe, path
    raise FileNotFoundError(
        f"audit.{NOTEBOOK_KIND} was not found in: {[str(path) for path in candidates]}"
    )


PROBE_CONFIG, PROBE_CONFIG_PATH = load_probe_config()
PF_PARTICLES = int(PROBE_CONFIG["particles"])
PF_SEEDS = int(PROBE_CONFIG["seed_count"])
PF_PATH_ALPHA = float(PROBE_CONFIG["path_alpha"])
PF_PATH_LINEWIDTH = float(PROBE_CONFIG["path_linewidth"])
PF_PATH_COLOR = str(PROBE_CONFIG["path_color"])
TRUTH_COLOR = str(PROBE_CONFIG["truth_color"])
LGB_OOF_COLOR = str(PROBE_CONFIG["lgb_oof_color"])
MAX_POINTS_PER_PLOT = int(PROBE_CONFIG["max_points_per_plot"])
N_JOBS = min(int(PROBE_CONFIG["n_jobs"]), os.cpu_count() or 1)
BATCH_WELLS = int(PROBE_CONFIG["batch_wells"])
ZIP_PLOTS = bool(PROBE_CONFIG["zip_plots"])
TVT_AXIS_INVERTED = bool(PROBE_CONFIG["tvt_axis_inverted"])
SAVED_MEAN_PARITY_ATOL = float(PROBE_CONFIG["saved_mean_parity_atol"])
MAX_PLOTS_ENV = os.environ.get("EXPERIMENT_MAX_PLOTS")
MAX_PLOTS = int(MAX_PLOTS_ENV) if MAX_PLOTS_ENV else None
N_JOBS_ENV = os.environ.get("EXPERIMENT_N_JOBS")
if N_JOBS_ENV:
    N_JOBS = min(int(N_JOBS_ENV), os.cpu_count() or 1)

if PF_PARTICLES != 500 or PF_SEEDS != 128:
    raise ValueError("The exp072 replay contract must remain 500 particles x 128 seeds")
if not 0.0 < PF_PATH_ALPHA < 1.0:
    raise ValueError("PF path alpha must be between 0 and 1")
if N_JOBS < 1 or BATCH_WELLS < 1:
    raise ValueError("n_jobs and batch_wells must be positive")

print("Experiment:", EXPERIMENT_NAME)
print("Notebook kind:", NOTEBOOK_KIND)
print("Route: ml_model (PF is a diagnostic overlay only)")
print("Probe config:", PROBE_CONFIG_PATH)
print("PF replay:", {"particles": PF_PARTICLES, "seeds": PF_SEEDS})
print("PF plot style:", {"alpha": PF_PATH_ALPHA, "linewidth": PF_PATH_LINEWIDTH})
print("Parallelism:", {"n_jobs": N_JOBS, "batch_wells": BATCH_WELLS})
print("Debug max plots override:", MAX_PLOTS)
print("Model fit / submission generation: 0 / 0")

# %% [markdown]
# ## 2. Path resolution and input contract

# %%
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
ARTIFACTS_DIR = (
    KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else EXP_DIR
) / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_input(
    *,
    filename: str,
    local_candidates: list[Path],
    preferred_slugs: list[str],
) -> Path:
    local = _existing(local_candidates)
    if local is not None:
        return local

    if KAGGLE_INPUT_ROOT.exists():
        preferred_roots = [KAGGLE_INPUT_ROOT / slug for slug in preferred_slugs]
        preferred_roots.extend(
            KAGGLE_INPUT_ROOT / "notebooks" / "kentookumura" / slug
            for slug in preferred_slugs
        )
        generic_roots = [
            path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir()
        ]
        seen: set[Path] = set()
        for root in [*preferred_roots, *generic_roots]:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]

    checked = "\n".join(str(path) for path in local_candidates)
    raise FileNotFoundError(
        f"{filename} not found. Checked local paths:\n{checked}\n"
        f"Kaggle slugs: {preferred_slugs}"
    )


def is_raw_data_root(path: Path) -> bool:
    train_dir = path / "train"
    return (
        train_dir.is_dir()
        and next(train_dir.glob("*__horizontal_well.csv"), None) is not None
        and next(train_dir.glob("*__typewell.csv"), None) is not None
    )


def resolve_raw_data_root() -> Path:
    candidates = [
        REPO_ROOT / "data" / "raw",
        Path.cwd() / "data" / "raw",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction",
    ]
    for path in candidates:
        if is_raw_data_root(path):
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for sample_path in sorted(KAGGLE_INPUT_ROOT.rglob("sample_submission.csv")):
            if is_raw_data_root(sample_path.parent):
                return sample_path.parent
    raise FileNotFoundError("ROGII raw train/typewell directory was not found")


pfbeam_path = resolve_input(
    filename=PFBEAM_FILENAME,
    local_candidates=[
        REPO_ROOT
        / "experiments"
        / "exp072_exp063_full_replay_feature_cache"
        / "artifacts"
        / PFBEAM_FILENAME,
        Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train/artifacts")
        / PFBEAM_FILENAME,
    ],
    preferred_slugs=["exp072-exp063-full-replay-feature-cache-train"],
)
exp238_oof_path = resolve_input(
    filename=EXP238_OOF_FILENAME,
    local_candidates=[
        EXP_DIR / "artifacts" / EXP238_OOF_FILENAME,
        EXP_DIR / "kaggle" / "output" / "train_v5" / "artifacts" / EXP238_OOF_FILENAME,
        Path("/tmp/kaggle-output")
        / EXPERIMENT_NAME
        / "train_v5"
        / "artifacts"
        / EXP238_OOF_FILENAME,
    ],
    preferred_slugs=["exp238-nested-rank-slot-exp218-train"],
)
raw_data_root = resolve_raw_data_root()

print("Artifacts dir:", ARTIFACTS_DIR)
print("exp072 cache:", pfbeam_path)
print("exp238 OOF:", exp238_oof_path)
print("Raw data root:", raw_data_root)

# %% [markdown]
# ## 3. Saved truth, exp072 mean, and exp238 OOF

# %%
def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    if decompressed and path.suffix == ".gz":
        stream_context = gzip.open(path, "rb")
    else:
        stream_context = path.open("rb")
    with stream_context as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse_values(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(prediction[valid] - truth[valid]))))


def read_base_frame(path: Path) -> pd.DataFrame:
    needed = {
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "likpf_mean_d",
    }
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(needed.difference(header))
    if missing:
        raise ValueError(f"exp072 cache is missing columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in needed,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    if frame.duplicated(["id", "well"]).any():
        raise ValueError("exp072 cache has duplicate id/well rows")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in needed.difference({"id", "well"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if not np.isfinite(
        frame[["target", "last_known_tvt", "md_since", "likpf_mean_d"]].to_numpy(
            np.float32
        )
    ).all():
        raise ValueError("exp072 base columns contain non-finite values")
    frame["true_tvt"] = (
        frame["last_known_tvt"].to_numpy(np.float32)
        + frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    frame["saved_likpf_mean_tvt"] = (
        frame["last_known_tvt"].to_numpy(np.float32)
        + frame["likpf_mean_d"].to_numpy(np.float32)
    ).astype(np.float32)
    return frame


def read_order_aligned_column(
    path: Path,
    base: pd.DataFrame,
    *,
    source_name: str,
    value_column: str,
) -> np.ndarray:
    usecols = ["id", "well", value_column]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(usecols).difference(header))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")
    output = np.full(len(base), np.nan, dtype=np.float32)
    base_ids = base["id"].to_numpy(dtype=str)
    base_wells = base["well"].to_numpy(dtype=str)
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"id": str, "well": str},
        chunksize=READ_CHUNKSIZE,
        low_memory=False,
    ):
        stop = offset + len(chunk)
        if stop > len(base):
            raise ValueError(f"{source_name} contains more rows than the exp072 cache")
        if not np.array_equal(chunk["id"].astype(str).to_numpy(), base_ids[offset:stop]):
            raise ValueError(f"{source_name} id order differs at row {offset}")
        if not np.array_equal(
            chunk["well"].astype(str).to_numpy(), base_wells[offset:stop]
        ):
            raise ValueError(f"{source_name} well order differs at row {offset}")
        output[offset:stop] = pd.to_numeric(
            chunk[value_column], errors="coerce"
        ).to_numpy(np.float32)
        offset = stop
    if offset != len(base):
        raise ValueError(f"{source_name} row count {offset} != base row count {len(base)}")
    if not np.isfinite(output).all():
        raise ValueError(f"{source_name} contains non-finite {value_column}")
    return output


base = read_base_frame(pfbeam_path)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise ValueError(
        {
            "message": "exp238 PF plot base row/well contract changed",
            "rows": len(base),
            "wells": int(base["well"].nunique()),
            "expected_rows": EXPECTED_ROWS,
            "expected_wells": EXPECTED_WELLS,
        }
    )
base["exp238_lgb_mean_oof_tvt"] = read_order_aligned_column(
    exp238_oof_path,
    base,
    source_name="exp238 final OOF",
    value_column="lgb_mean_pred_tvt",
)

print("Base rows / wells:", len(base), "/", int(base["well"].nunique()))
print(
    "exp238 lgb_mean OOF RMSE:",
    rmse_values(
        base["true_tvt"].to_numpy(), base["exp238_lgb_mean_oof_tvt"].to_numpy()
    ),
)
display(base.head())

# %% [markdown]
# ## 4. Exact exp072 likelihood-PF replay helpers
#
# Only the functions and fixed constants needed for the exp072 likelihood-PF are
# carried into this notebook. The state update does not receive true TVT.

# %%
def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


@njit(nogil=True)
def _interp1(grid: np.ndarray, value: float, vmin: float, step: float) -> float:
    index = int((value - vmin) / step)
    if index < 0:
        return grid[0]
    last = len(grid) - 1
    if index >= last:
        return grid[last]
    fraction = (value - vmin) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(nogil=True)
def _pf_lik_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gg: np.ndarray,
    vmin: float,
    step: float,
    gs: float,
    ls: float,
    ir: float,
    n_particles: int,
    n_seeds: int,
    seed_base: int,
    momentum: float,
    velocity_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(md_v)
    predictions = np.empty((n_seeds, n_rows))
    log_likelihoods = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for seed_index in range(n_seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(n_particles)
        rate = np.empty(n_particles)
        weights = np.ones(n_particles) / n_particles
        for particle in range(n_particles):
            position[particle] = ls + initial_spread * np.random.randn()
            rate[particle] = ir + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row_index in range(n_rows):
            md_step = md_v[row_index] - previous_md
            if md_step < 1.0:
                md_step = 1.0
            for particle in range(n_particles):
                rate[particle] = (
                    momentum * rate[particle] + velocity_noise * np.random.randn()
                )
                position[particle] += (
                    rate[particle] * md_step + position_noise * np.random.randn()
                )
                particle_tvt = position[particle] - z_v[row_index]
                if particle_tvt < vmin - 100.0:
                    particle_tvt = vmin - 100.0
                if particle_tvt > tmax + 100.0:
                    particle_tvt = tmax + 100.0
                position[particle] = particle_tvt + z_v[row_index]
            average_likelihood = 0.0
            for particle in range(n_particles):
                expected_gr = _interp1(
                    gg, position[particle] - z_v[row_index], vmin, step
                )
                residual = (gr_v[row_index] - expected_gr) / gs
                squared_residual = residual * residual
                if squared_residual > 600.0:
                    squared_residual = 600.0
                likelihood = np.exp(-0.5 * squared_residual)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)
            weight_sum = 0.0
            for particle in range(n_particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(n_particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(n_particles):
                    weights[particle] = 1.0 / n_particles
            inverse_effective_n = 0.0
            for particle in range(n_particles):
                inverse_effective_n += weights[particle] * weights[particle]
            effective_n = 1.0 / inverse_effective_n
            if effective_n < resample_fraction * n_particles:
                cumulative = np.empty(n_particles)
                cumulative_weight = 0.0
                for particle in range(n_particles):
                    cumulative_weight += weights[particle]
                    cumulative[particle] = cumulative_weight
                first_u = np.random.uniform(0.0, 1.0 / n_particles)
                new_position = np.empty(n_particles)
                new_rate = np.empty(n_particles)
                cumulative_index = 0
                for particle in range(n_particles):
                    u_value = first_u + particle / n_particles
                    while (
                        cumulative_index < n_particles - 1
                        and cumulative[cumulative_index] < u_value
                    ):
                        cumulative_index += 1
                    new_position[particle] = (
                        position[cumulative_index] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = (
                        rate[cumulative_index] + rough_rate * np.random.randn()
                    )
                for particle in range(n_particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / n_particles
            estimate = 0.0
            for particle in range(n_particles):
                estimate += weights[particle] * (
                    position[particle] - z_v[row_index]
                )
            predictions[seed_index, row_index] = estimate
            previous_md = md_v[row_index]
        log_likelihoods[seed_index] = log_likelihood
    return predictions, log_likelihoods


def typewell_grid(
    typewell_tvt: np.ndarray, typewell_gr: np.ndarray, step: float = 0.2
) -> tuple[np.ndarray, float, float]:
    minimum_tvt = float(typewell_tvt.min())
    maximum_tvt = float(typewell_tvt.max())
    grid_tvt = np.arange(minimum_tvt, maximum_tvt + step, step)
    grid_gr = np.interp(grid_tvt, typewell_tvt, typewell_gr).astype(np.float64)
    return grid_gr, minimum_tvt, float(step)


def replay_one_well(well_id: str) -> dict[str, Any]:
    horizontal_path = raw_data_root / "train" / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_data_root / "train" / f"{well_id}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"raw files missing for well {well_id}")
    horizontal = pd.read_csv(horizontal_path, low_memory=False)
    typewell = pd.read_csv(typewell_path, low_memory=False).sort_values("TVT")
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError(f"well {well_id}: horizontal columns are incomplete")
    if not required_typewell.issubset(typewell.columns):
        raise ValueError(f"well {well_id}: typewell columns are incomplete")

    known = horizontal.loc[horizontal["TVT_input"].notna()]
    evaluation = horizontal.loc[horizontal["TVT_input"].isna()]
    if known.empty or evaluation.empty:
        raise ValueError(f"well {well_id}: known/evaluation split is empty")
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(
        np.float64
    )
    typewell_gr_series = pd.to_numeric(typewell["GR"], errors="coerce")
    typewell_gr = typewell_gr_series.fillna(typewell_gr_series.mean()).to_numpy(
        np.float64
    )
    if not np.isfinite(typewell_tvt).all() or not np.isfinite(typewell_gr).all():
        raise ValueError(f"well {well_id}: typewell contains non-finite values")

    last_known = known.iloc[-1]
    last_position = float(last_known["TVT_input"]) + float(last_known["Z"])
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(
        np.float64
    )
    known_tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    typewell_gr_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    gr_sigma = float(np.clip(np.nanstd(known_gr - typewell_gr_at_known), 10.0, 60.0))

    tail = known.tail(30)
    delta_tvt = np.diff(
        pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    )
    delta_z = np.diff(pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64))
    delta_md = np.diff(
        pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    )
    positive_md = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[positive_md] / delta_md[positive_md]))
        if positive_md.sum() >= 3
        else 0.0
    )
    grid_gr, grid_minimum_tvt, grid_step = typewell_grid(typewell_tvt, typewell_gr)
    full_gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
    )
    evaluation_gr = full_gr.loc[evaluation.index].to_numpy(np.float64)
    evaluation_md = pd.to_numeric(evaluation["MD"], errors="coerce").to_numpy(
        np.float64
    )
    evaluation_z = pd.to_numeric(evaluation["Z"], errors="coerce").to_numpy(
        np.float64
    )
    if not (
        np.isfinite(evaluation_gr).all()
        and np.isfinite(evaluation_md).all()
        and np.isfinite(evaluation_z).all()
    ):
        raise ValueError(f"well {well_id}: PF evaluation inputs contain non-finite values")

    seed_base = stable_seed("likpf", "train", well_id)
    paths, log_likelihoods = _pf_lik_allseeds(
        evaluation_md,
        evaluation_z,
        evaluation_gr,
        grid_gr,
        grid_minimum_tvt,
        grid_step,
        gr_sigma,
        last_position,
        initial_rate,
        PF_PARTICLES,
        PF_SEEDS,
        seed_base,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    if paths.shape != (PF_SEEDS, len(evaluation)):
        raise ValueError(
            f"well {well_id}: PF shape {paths.shape} != {(PF_SEEDS, len(evaluation))}"
        )
    return {
        "well": well_id,
        "raw_row_indices": evaluation.index.to_numpy(np.int64),
        "paths": paths,
        "log_likelihoods": log_likelihoods,
        "seed_base": seed_base,
        "gr_sigma": gr_sigma,
        "initial_rate": initial_rate,
    }


# Compile once before thread-parallel well replay.
_warm_md = np.asarray([1.0, 2.0], dtype=np.float64)
_warm_zero = np.zeros(2, dtype=np.float64)
_warm_gr = np.full(2, 50.0, dtype=np.float64)
_warm_grid = np.linspace(45.0, 55.0, 64, dtype=np.float64)
_pf_lik_allseeds(
    _warm_md,
    _warm_zero,
    _warm_gr,
    _warm_grid,
    0.0,
    0.2,
    20.0,
    1.0,
    0.0,
    8,
    2,
    1,
    0.998,
    0.002,
    0.005,
    0.1,
    0.001,
    0.5,
    4.5,
)
print("Numba likelihood-PF replay helper compiled")

# %% [markdown]
# ## 5. Plot and metric helpers

# %%
def downsample_positions(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=np.int64))


def plot_one_well(
    well_id: str,
    group: pd.DataFrame,
    pf_paths: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    full_truth = group["true_tvt"].to_numpy(np.float64)
    full_lgb_oof = group["exp238_lgb_mean_oof_tvt"].to_numpy(np.float64)
    positions = downsample_positions(len(group), MAX_POINTS_PER_PLOT)
    x = group["md_since"].to_numpy(np.float64)[positions]
    truth = full_truth[positions]
    lgb_oof = full_lgb_oof[positions]
    plotted_paths = pf_paths[:, positions]

    fig, axis = plt.subplots(figsize=(16, 8))
    for seed_index in range(PF_SEEDS):
        axis.plot(
            x,
            plotted_paths[seed_index],
            color=PF_PATH_COLOR,
            linewidth=PF_PATH_LINEWIDTH,
            alpha=PF_PATH_ALPHA,
            zorder=1,
        )
    axis.plot(
        x,
        truth,
        color=TRUTH_COLOR,
        linewidth=2.5,
        alpha=1.0,
        label="true TVT",
        zorder=5,
    )
    axis.plot(
        x,
        lgb_oof,
        color=LGB_OOF_COLOR,
        linewidth=2.1,
        alpha=1.0,
        label="exp238 lgb_mean OOF",
        zorder=4,
    )
    if TVT_AXIS_INVERTED:
        axis.invert_yaxis()
    axis.set_xlabel("MD since prediction start (ft)")
    axis.set_ylabel("TVT (ft; depth increases downward)")
    axis.grid(True, color="#e2e8f0", linewidth=0.7, alpha=0.8)
    axis.set_title(
        f"{well_id} | all {PF_SEEDS} exp072 likelihood-PF seed paths | "
        f"PF alpha={PF_PATH_ALPHA:g}\n"
        f"exp238 OOF RMSE {rmse_values(full_truth, full_lgb_oof):.2f} | "
        f"PF seed-mean RMSE "
        f"{rmse_values(full_truth, pf_paths.mean(axis=0)):.2f}"
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=PF_PATH_COLOR,
            linewidth=1.5,
            alpha=0.65,
            label=f"{PF_SEEDS} likelihood-PF seed paths",
        ),
        Line2D([0], [0], color=TRUTH_COLOR, linewidth=2.5, label="true TVT"),
        Line2D(
            [0],
            [0],
            color=LGB_OOF_COLOR,
            linewidth=2.1,
            label="exp238 lgb_mean OOF",
        ),
    ]
    axis.legend(handles=handles, loc="best", fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Diagnostic only: PF seeds are not selected or blended into exp238; true TVT is "
        "used only for the overlay and metrics.",
        fontsize=8,
        color="#7f1d1d",
    )
    fig.tight_layout(rect=[0.0, 0.035, 1.0, 1.0])
    fig.savefig(output_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_plot_well(
    replay: dict[str, Any],
    group: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    well_id = str(replay["well"])
    raw_row_indices = np.asarray(replay["raw_row_indices"], dtype=np.int64)
    expected_ids = np.asarray(
        [f"{well_id}_{row_index}" for row_index in raw_row_indices], dtype=str
    )
    group_ids = group["id"].astype(str).to_numpy()
    if not np.array_equal(expected_ids, group_ids):
        raise ValueError(f"well {well_id}: raw evaluation row IDs differ from exp072 cache")

    pf_paths = np.asarray(replay["paths"], dtype=np.float64)
    truth = group["true_tvt"].to_numpy(np.float64)
    lgb_oof = group["exp238_lgb_mean_oof_tvt"].to_numpy(np.float64)
    last_known_tvt = group["last_known_tvt"].to_numpy(np.float32)
    saved_delta = group["likpf_mean_d"].to_numpy(np.float32)
    regenerated_mean_float32 = pf_paths.mean(axis=0).astype(np.float32)
    regenerated_delta = (regenerated_mean_float32 - last_known_tvt).astype(np.float32)
    parity_abs = np.abs(regenerated_delta.astype(np.float64) - saved_delta.astype(np.float64))
    parity_max_abs = float(parity_abs.max())
    parity_mean_abs = float(parity_abs.mean())
    parity_exact = bool(np.array_equal(regenerated_delta, saved_delta))
    if parity_max_abs > SAVED_MEAN_PARITY_ATOL:
        raise ValueError(
            f"well {well_id}: replay/saved likpf_mean_d max abs {parity_max_abs} "
            f"> {SAVED_MEAN_PARITY_ATOL}"
        )

    seed_rmse = np.sqrt(np.mean(np.square(pf_paths - truth[None, :]), axis=1))
    plot_one_well(well_id, group, pf_paths, output_path)
    return {
        "well": well_id,
        "rows": int(len(group)),
        "plotted_rows": int(min(len(group), MAX_POINTS_PER_PLOT)),
        "pf_seed_count": int(pf_paths.shape[0]),
        "pf_particles_per_seed": PF_PARTICLES,
        "seed_base": int(replay["seed_base"]),
        "gr_sigma": float(replay["gr_sigma"]),
        "initial_rate": float(replay["initial_rate"]),
        "exp238_lgb_mean_oof_rmse": rmse_values(truth, lgb_oof),
        "pf_seed_mean_rmse": rmse_values(truth, regenerated_mean_float32),
        "pf_seed_rmse_min": float(seed_rmse.min()),
        "pf_seed_rmse_p50": float(np.quantile(seed_rmse, 0.50)),
        "pf_seed_rmse_max": float(seed_rmse.max()),
        "saved_mean_parity_exact": parity_exact,
        "saved_mean_parity_max_abs": parity_max_abs,
        "saved_mean_parity_mean_abs": parity_mean_abs,
        "pf_squared_error_sum": float(
            np.square(regenerated_mean_float32.astype(np.float64) - truth).sum()
        ),
        "lgb_squared_error_sum": float(np.square(lgb_oof - truth).sum()),
        "plot_path": str(output_path),
    }

# %% [markdown]
# ## 6. Replay and plot all wells

# %%
all_wells = sorted(base["well"].dropna().astype(str).unique().tolist())
plot_wells = all_wells[:MAX_PLOTS] if MAX_PLOTS is not None else all_wells
indices_by_well = base.groupby("well", sort=False).indices
run_started = time.time()
plot_rows: list[dict[str, Any]] = []

for batch_start in range(0, len(plot_wells), BATCH_WELLS):
    batch = plot_wells[batch_start : batch_start + BATCH_WELLS]
    replay_batch = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(replay_one_well)(well_id) for well_id in batch
    )
    for replay in replay_batch:
        well_id = str(replay["well"])
        positions = indices_by_well[well_id]
        group = base.iloc[positions].copy()
        plot_path = PLOTS_DIR / f"{well_id}.png"
        plot_rows.append(evaluate_and_plot_well(replay, group, plot_path))
    completed = min(batch_start + len(batch), len(plot_wells))
    print(
        f"replayed and plotted {completed}/{len(plot_wells)} wells | "
        f"elapsed {time.time() - run_started:.1f}s",
        flush=True,
    )

manifest = pd.DataFrame(plot_rows)
if len(manifest) != len(plot_wells):
    raise ValueError(f"manifest wells {len(manifest)} != requested wells {len(plot_wells)}")
if manifest["well"].nunique() != len(plot_wells):
    raise ValueError("manifest well IDs are not unique")
if not manifest["pf_seed_count"].eq(PF_SEEDS).all():
    raise ValueError("not every well plot received all 128 PF seed paths")
if float(manifest["saved_mean_parity_max_abs"].max()) > SAVED_MEAN_PARITY_ATOL:
    raise ValueError("global saved mean parity guard failed")

manifest_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plot_manifest.csv"
manifest.to_csv(manifest_path, index=False)
zip_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots.zip"
if ZIP_PLOTS:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for plot_path in sorted(PLOTS_DIR.glob("*.png")):
            archive.write(plot_path, arcname=plot_path.name)

display(manifest.head(10))
display(
    manifest.sort_values("pf_seed_rmse_p50", ascending=False)[
        [
            "well",
            "rows",
            "exp238_lgb_mean_oof_rmse",
            "pf_seed_mean_rmse",
            "pf_seed_rmse_min",
            "pf_seed_rmse_p50",
            "pf_seed_rmse_max",
            "saved_mean_parity_max_abs",
        ]
    ].head(20)
)

# %% [markdown]
# ## 7. Summary and generated outputs

# %%
processed_rows = int(manifest["rows"].sum())
global_metrics = {
    "rows": processed_rows,
    "wells": int(len(manifest)),
    "exp238_lgb_mean_oof_rmse": float(
        np.sqrt(manifest["lgb_squared_error_sum"].sum() / processed_rows)
    ),
    "pf_seed_mean_rmse": float(
        np.sqrt(manifest["pf_squared_error_sum"].sum() / processed_rows)
    ),
    "saved_mean_parity_exact_all_wells": bool(
        manifest["saved_mean_parity_exact"].all()
    ),
    "saved_mean_parity_max_abs": float(
        manifest["saved_mean_parity_max_abs"].max()
    ),
    "saved_mean_parity_mean_abs_weighted": float(
        np.average(
            manifest["saved_mean_parity_mean_abs"], weights=manifest["rows"]
        )
    ),
}
artifact_sha = {
    "plot_manifest": sha256_path(manifest_path),
    "plots_zip": sha256_path(zip_path) if ZIP_PLOTS else None,
}
summary = {
    "status": "diagnostic_notebook_completed_not_submitted",
    "created_at_utc": datetime.now(UTC).isoformat(),
    "experiment": EXPERIMENT_NAME,
    "route": "ml_model",
    "notebook": f"{EXPERIMENT_NAME}_{NOTEBOOK_KIND}.ipynb",
    "reference_notebook": "kentookumura/exp238-oof-selector-confidence-probe",
    "reference_script_version_id": 335655690,
    "scope": {
        "plot_wells": len(plot_wells),
        "all_wells": len(all_wells),
        "rows": processed_rows,
        "max_plots_override": MAX_PLOTS,
        "runtime_seconds": round(time.time() - run_started, 3),
    },
    "pf_replay_contract": {
        "family": str(PROBE_CONFIG["pf_family"]),
        "split": str(PROBE_CONFIG["split"]),
        "particles_per_seed": PF_PARTICLES,
        "seed_count": PF_SEEDS,
        "seed_key": "stable_seed('likpf', 'train', well_id) + seed_index",
        "seed_indices": [0, PF_SEEDS - 1],
        "per_well_stable_seed": True,
        "parallel_unit": "well",
        "paths_persisted": False,
        "truth_used_in_pf": False,
    },
    "plot_contract": {
        "pf_path_count": PF_SEEDS,
        "pf_path_alpha": PF_PATH_ALPHA,
        "pf_path_linewidth": PF_PATH_LINEWIDTH,
        "pf_path_color": PF_PATH_COLOR,
        "truth_color": TRUTH_COLOR,
        "lgb_oof_color": LGB_OOF_COLOR,
        "truth_and_lgb_opaque": True,
        "max_points_per_plot": MAX_POINTS_PER_PLOT,
        "tvt_axis_inverted": TVT_AXIS_INVERTED,
    },
    "global_metrics": global_metrics,
    "inputs": {
        "exp072_cache": str(pfbeam_path),
        "exp072_cache_sha256_decompressed": sha256_path(
            pfbeam_path, decompressed=True
        ),
        "exp238_final_oof": str(exp238_oof_path),
        "exp238_final_oof_sha256_decompressed": sha256_path(
            exp238_oof_path, decompressed=True
        ),
        "raw_data_root": str(raw_data_root),
    },
    "artifacts": {
        "plot_directory": str(PLOTS_DIR),
        "plot_manifest": str(manifest_path),
        "plots_zip": str(zip_path) if ZIP_PLOTS else None,
        "summary": str(ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"),
    },
    "artifact_sha256": artifact_sha,
    "execution_contract": {
        "model_fits": 0,
        "lightgbm_boosters": 0,
        "submission_generated": False,
        "competition_submitted": False,
        "device": "cpu",
        "internet_required": False,
    },
}
summary_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(json.dumps(global_metrics, indent=2))
print("Generated outputs:")
for name, path in summary["artifacts"].items():
    print(f"- {name}: {path}")
print("Summary SHA256:", sha256_path(summary_path))
