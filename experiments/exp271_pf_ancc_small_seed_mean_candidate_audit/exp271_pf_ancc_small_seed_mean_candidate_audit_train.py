# %% [markdown]
# # exp268 PF ANCC small-seed mean candidate audit
#
# exp266で固定した先頭8 seedのPF ANCCを再生成し、4/8 seed mean pathを
# exp263 core-12 candidate bankへ加える価値を0-boosterで監査する。

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, and SHA helpers
# 3. Canonical input and upstream parity helpers
# 4. Exact exp266 PF ANCC kernel
# 5. Fixed small-seed candidate generation helpers
# 6. Exp263 core-bank loading helpers
# 7. Candidate and oracle diagnostic helpers
# 8. Setup and fixed execution contract
# 9. Input checks and target-free PF generation
# 10. Upstream parity and core-bank assembly
# 11. Candidate, oracle, and disagreement readout
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
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from numba import njit

EXPERIMENT_NAME = "exp271_pf_ancc_small_seed_mean_candidate_audit"
UPSTREAM_SEED_NAMESPACE = "exp266_pf_ancc_pf_z_multiseed_stability_audit"


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


def seed_vector(well: str, seed_count: int = 8) -> np.ndarray:
    values = [stable_seed("pf_ancc", well)]
    values.extend(
        stable_seed(UPSTREAM_SEED_NAMESPACE, "train", "pf_ancc", well, seed_index)
        for seed_index in range(1, seed_count)
    )
    seeds = np.asarray(values, dtype=np.int64)
    if len(seeds) != seed_count or len(np.unique(seeds)) != seed_count:
        raise RuntimeError(f"seed contract failed for {well}")
    return seeds


def sha256_path(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_gzip_content(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def require_kaggle_or_explicit_local() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "Kaggle Notebook is authoritative. Set EXPERIMENT_ALLOW_LOCAL=1 only for "
        "an explicitly approved local smoke run."
    )


def find_competition_root(config: dict[str, Any]) -> Path:
    if is_kaggle_runtime():
        direct = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
        if (direct / "train").is_dir():
            return direct
        for sample in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
            if (sample.parent / "train").is_dir():
                return sample.parent
        raise FileNotFoundError("Kaggle competition train directory was not found")
    return Path(str(nested(config, "data.raw_dir", "data/raw")))


def resolve_file_by_sha(filename: str, expected_sha256: str) -> Path:
    roots = [Path.cwd(), Path("/tmp")]
    if is_kaggle_runtime():
        roots.insert(0, Path("/kaggle/input"))
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(filename)):
            key = str(path.resolve())
            if key in seen or not path.is_file() or path.stat().st_size == 0:
                continue
            seen.add(key)
            if sha256_path(path) == expected_sha256:
                return path
    raise FileNotFoundError(f"{filename} with expected SHA {expected_sha256} was not found")


def resolve_file(filename: str, local_candidates: Iterable[str] = ()) -> Path:
    candidates = [Path(value) for value in local_candidates]
    roots = [Path.cwd(), Path("/tmp")]
    if is_kaggle_runtime():
        roots.insert(0, Path("/kaggle/input"))
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.rglob(filename)))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(filename)


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


def artifact_manifest_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "filename": path.name,
                "bytes": int(path.stat().st_size),
                "raw_sha256": sha256_path(path),
                "decompressed_sha256": (
                    sha256_gzip_content(path) if path.suffix == ".gz" else sha256_path(path)
                ),
            }
        )
    return rows


# %% [markdown]
# ## 3. Canonical input and upstream parity helpers

# %%
def id_row_index(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()


def load_exp072_reference(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    filename = str(nested(config, "data.exp072.filename"))
    path = resolve_file(
        filename,
        [
            "experiments/exp072_exp063_full_replay_feature_cache/artifacts/"
            + filename,
        ],
    )
    raw_sha = sha256_path(path)
    decompressed_sha = sha256_gzip_content(path)
    expected_raw = str(nested(config, "data.exp072.expected_sha256"))
    expected_decompressed = str(nested(config, "data.exp072.expected_decompressed_sha256"))
    if raw_sha != expected_raw or decompressed_sha != expected_decompressed:
        raise RuntimeError("exp072 canonical cache SHA guard failed")

    frame = pd.read_csv(
        path,
        usecols=["id", "well", "md_since", "pf_ancc"],
        dtype={"id": "string", "well": "string", "pf_ancc": "float32"},
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = id_row_index(frame["id"])
    id_well = frame["id"].str.rsplit("_", n=1).str[0]
    if not id_well.equals(frame["well"]):
        raise RuntimeError("exp072 id/well identity guard failed")
    if frame["id"].duplicated().any() or frame["pf_ancc"].dtype != np.dtype(np.float32):
        raise RuntimeError("exp072 identity or float32 PF contract failed")
    expected_rows = int(nested(config, "validation.expected_rows"))
    expected_wells = int(nested(config, "validation.expected_wells"))
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise RuntimeError("exp072 coverage guard failed")
    manifest = {
        "kind": "exp072_canonical_cache",
        "path": str(path),
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "schema_sha256_loaded_subset": schema_sha(frame),
    }
    return frame, path, manifest


def load_exp266_aggregate(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest_filename = str(nested(config, "data.exp266.artifact_manifest_filename"))
    expected_manifest_sha = str(
        nested(config, "data.exp266.expected_artifact_manifest_sha256")
    )
    manifest_path = resolve_file_by_sha(manifest_filename, expected_manifest_sha)
    manifest = pd.read_csv(manifest_path)
    aggregate_filename = str(nested(config, "data.exp266.aggregate_filename"))
    matched = manifest.loc[manifest["filename"].astype(str) == aggregate_filename]
    if len(matched) != 1:
        raise RuntimeError("exp266 aggregate manifest row is not unique")
    aggregate_path = resolve_file(
        aggregate_filename,
        [str(manifest_path.parent / aggregate_filename)],
    )
    row = matched.iloc[0]
    raw_sha = sha256_path(aggregate_path)
    decompressed_sha = sha256_gzip_content(aggregate_path)
    if raw_sha != str(row["raw_sha256"]) or decompressed_sha != str(
        row["decompressed_sha256"]
    ):
        raise RuntimeError("exp266 aggregate artifact SHA guard failed")
    aggregate = pd.read_csv(aggregate_path, dtype={"well": "string"})
    aggregate["well"] = aggregate["well"].astype(str)
    selected = aggregate.loc[
        (aggregate["algorithm"] == "pf_ancc")
        & (aggregate["aggregation"] == "mean")
        & (aggregate["seed_count"].isin([4, 8]))
    ].copy()
    if len(selected) != 2 * int(nested(config, "validation.expected_wells")):
        raise RuntimeError("exp266 mean4/mean8 aggregate coverage guard failed")
    manifests = [
        {
            "kind": "exp266_artifact_manifest",
            "path": str(manifest_path),
            "filename": manifest_path.name,
            "bytes": int(manifest_path.stat().st_size),
            "raw_sha256": sha256_path(manifest_path),
            "decompressed_sha256": sha256_path(manifest_path),
        },
        {
            "kind": "exp266_aggregate_by_well",
            "path": str(aggregate_path),
            "filename": aggregate_path.name,
            "bytes": int(aggregate_path.stat().st_size),
            "raw_sha256": raw_sha,
            "decompressed_sha256": decompressed_sha,
        },
    ]
    return selected, manifests


# %% [markdown]
# ## 4. Exact exp266 PF ANCC kernel
#
# exp072/exp266の演算順を保ち、Numba cacheだけnotebook安全性のため無効化する。

# %%
ANCC_N = 600
PF_GR_SIG_MIN = 10.0
PF_GR_SIG_MAX = 60.0
PF_GR_SIG_DEF = 30.0
PF_RESAMP = 0.5
ANCC_ALPHA = 0.998
ANCC_RN = 0.002
ANCC_PN = 0.005
ANCC_IS = 0.3
ANCC_RP = 0.1
ANCC_RR = 0.001


@njit(cache=False, nogil=True)
def _interp1(grid, value, minimum, step):
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    last = len(grid) - 1
    if index >= last:
        return grid[last]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=False, nogil=True)
def _resamp(position, rate, weights, n_particles, rough_position, rough_rate):
    cumulative = np.zeros(n_particles + 1)
    for index in range(n_particles):
        cumulative[index + 1] = cumulative[index] + weights[index]
    start = np.random.uniform(0.0, 1.0 / n_particles)
    new_position = np.empty(n_particles)
    new_rate = np.empty(n_particles)
    cursor = 0
    for index in range(n_particles):
        threshold = start + index / n_particles
        while cursor < n_particles - 1 and cumulative[cursor + 1] < threshold:
            cursor += 1
        new_position[index] = position[cursor] + rough_position * np.random.randn()
        new_rate[index] = rate[cursor] + rough_rate * np.random.randn()
    return new_position, new_rate


@njit(cache=False, nogil=True)
def _pf_ancc(
    md_values,
    z_values,
    gr_values,
    grid,
    minimum,
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
    position = np.empty(n_particles)
    rate = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for particle in range(n_particles):
        position[particle] = last_surface + init_spread * np.random.randn()
        rate[particle] = initial_rate + 0.01 * np.random.randn()
    predictions = np.empty(len(md_values))
    standard_deviations = np.empty(len(md_values))
    previous_md = md_values[0] - 1.0
    for row in range(len(md_values)):
        delta_md = md_values[row] - previous_md
        delta_md = max(delta_md, 1.0)
        for particle in range(n_particles):
            rate[particle] = alpha * rate[particle] + rate_noise * np.random.randn()
            position[particle] += (
                rate[particle] * delta_md + position_noise * np.random.randn()
            )
            tvt = position[particle] - z_values[row]
            tvt = max(tvt, minimum - 50.0)
            tvt = min(tvt, minimum + len(grid) * step + 50.0)
            position[particle] = tvt + z_values[row]
        if not np.isnan(gr_values[row]):
            weight_sum = 0.0
            for particle in range(n_particles):
                expected_gr = _interp1(
                    grid, position[particle] - z_values[row], minimum, step
                )
                delta = (gr_values[row] - expected_gr) / gr_sigma
                likelihood = max(
                    np.exp(-0.5 * delta * delta) if delta * delta < 600.0 else 0.0,
                    1e-300,
                )
                weights[particle] *= likelihood
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(n_particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(n_particles):
                    weights[particle] = 1.0 / n_particles
        neff_inverse = 0.0
        for particle in range(n_particles):
            neff_inverse += weights[particle] * weights[particle]
        if 1.0 / neff_inverse < resample_threshold * n_particles:
            position, rate = _resamp(
                position,
                rate,
                weights,
                n_particles,
                rough_position,
                rough_rate,
            )
            for particle in range(n_particles):
                weights[particle] = 1.0 / n_particles
        estimate = 0.0
        for particle in range(n_particles):
            estimate += weights[particle] * (position[particle] - z_values[row])
        predictions[row] = estimate
        variance = 0.0
        for particle in range(n_particles):
            variance += weights[particle] * (
                position[particle] - z_values[row] - estimate
            ) ** 2
        standard_deviations[row] = variance**0.5
        previous_md = md_values[row]
    return predictions, standard_deviations


@njit(cache=False, nogil=True)
def _pf_ancc_seeded(
    seed,
    md_values,
    z_values,
    gr_values,
    grid,
    minimum,
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
        md_values,
        z_values,
        gr_values,
        grid,
        minimum,
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


def _grid(
    tvt: np.ndarray, gr: np.ndarray, step: float = 0.2
) -> tuple[np.ndarray, float, float]:
    minimum = float(tvt.min())
    maximum = float(tvt.max())
    tvt_grid = np.arange(minimum, maximum + step, step)
    return np.interp(tvt_grid, tvt, gr).astype(np.float64), minimum, float(step)


def _gr_sigma(horizontal: pd.DataFrame, type_tvt: np.ndarray, type_gr: np.ndarray) -> float:
    known = horizontal[horizontal.TVT_input.notna() & horizontal.GR.notna()]
    if len(known) < 20:
        return float(PF_GR_SIG_DEF)
    residual = known.GR.values - np.interp(known.TVT_input.values, type_tvt, type_gr)
    return float(np.clip(np.std(residual), PF_GR_SIG_MIN, PF_GR_SIG_MAX))


# %% [markdown]
# ## 5. Fixed small-seed candidate generation helpers

# %%
def prepare_ancc_args(
    horizontal: pd.DataFrame, type_tvt: np.ndarray, type_gr: np.ndarray
) -> tuple[tuple[Any, ...], dict[str, float]]:
    known = horizontal[horizontal.TVT_input.notna()]
    evaluation = horizontal[horizontal.TVT_input.isna()]
    if evaluation.empty or known.empty:
        raise RuntimeError("PF ANCC requires a known prefix and evaluation rows")
    gr_sigma = _gr_sigma(horizontal, type_tvt, type_gr)
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
    grid, minimum, step = _grid(type_tvt, type_gr)
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
    return args, {"gr_sigma": gr_sigma, "initial_rate": initial_rate}


def read_well(train_dir: Path, well: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"raw train files are missing for {well}")
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path).sort_values("TVT")
    return horizontal, typewell, horizontal_path, typewell_path


def load_raw_targets_after_candidate_freeze(
    train_dir: Path,
    wells: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for well in wells:
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=["TVT", "TVT_input"])
        evaluation = horizontal.loc[horizontal["TVT_input"].isna(), ["TVT"]]
        frames.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in evaluation.index],
                    "well": well,
                    "row_idx": evaluation.index.to_numpy(np.int64),
                    "target_tvt": evaluation["TVT"].to_numpy(np.float64),
                }
            )
        )
    targets = pd.concat(frames, ignore_index=True)
    if targets["id"].duplicated().any() or not np.isfinite(
        targets["target_tvt"].to_numpy(np.float64)
    ).all():
        raise RuntimeError("raw target identity/finite guard failed")
    return targets


def generate_well_candidate(
    train_dir: Path,
    well: str,
    *,
    seed_count: int = 8,
    progress_ordinal: int = 0,
    progress_every: int = 16,
) -> dict[str, Any]:
    horizontal, typewell, horizontal_path, typewell_path = read_well(train_dir, well)
    evaluation = horizontal[horizontal.TVT_input.isna()]
    type_tvt = typewell.TVT.to_numpy(np.float32)
    type_gr = typewell.GR.to_numpy(np.float32)
    args, quality = prepare_ancc_args(horizontal, type_tvt, type_gr)
    seeds = seed_vector(well, seed_count)
    paths = np.empty((seed_count, len(evaluation)), dtype=np.float32)
    particle_stds = np.empty((seed_count, len(evaluation)), dtype=np.float32)
    for seed_index, seed in enumerate(seeds):
        prediction, particle_std = _pf_ancc_seeded(int(seed), *args)
        paths[seed_index] = prediction.astype(np.float32)
        particle_stds[seed_index] = particle_std.astype(np.float32)
    mean4 = paths[:4].mean(axis=0).astype(np.float32)
    mean8 = paths.mean(axis=0).astype(np.float32)
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in evaluation.index],
            "well": well,
            "row_idx": evaluation.index.to_numpy(np.int64),
            "pf_ancc_seed0": paths[0],
            "pf_ancc_seed_mean_4": mean4,
            "pf_ancc_seed_mean_8": mean8,
            "pf_ancc_seed_std_4": paths[:4].std(axis=0).astype(np.float32),
            "pf_ancc_seed_std_8": paths.std(axis=0).astype(np.float32),
            "pf_ancc_particle_std_mean_4": particle_stds[:4].mean(axis=0).astype(np.float32),
            "pf_ancc_particle_std_mean_8": particle_stds.mean(axis=0).astype(np.float32),
            "pf_ancc_mean8_minus_mean4": (mean8 - mean4).astype(np.float32),
        }
    )
    quality_row = {
        "well": well,
        "known_rows": int(horizontal.TVT_input.notna().sum()),
        "evaluation_rows": int(len(evaluation)),
        "typewell_rows": int(len(typewell)),
        "horizontal_sha256": sha256_path(horizontal_path),
        "typewell_sha256": sha256_path(typewell_path),
        "horizontal_path": str(horizontal_path),
        "typewell_path": str(typewell_path),
        "seed0": int(seeds[0]),
        "seed7": int(seeds[-1]),
        **quality,
    }
    if progress_ordinal % progress_every == 0:
        print(
            f"PF ANCC progress ordinal={progress_ordinal} well={well} "
            f"rows={len(evaluation):,}",
            flush=True,
        )
    return {"frame": frame, "quality": quality_row}


def generate_all_candidates(
    train_dir: Path,
    wells: list[str],
    *,
    seed_count: int,
    workers: int,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = Parallel(n_jobs=workers, prefer="threads")(
        delayed(generate_well_candidate)(
            train_dir,
            well,
            seed_count=seed_count,
            progress_ordinal=ordinal,
            progress_every=progress_every,
        )
        for ordinal, well in enumerate(wells)
    )
    frame = pd.concat([item["frame"] for item in results], ignore_index=True)
    quality = pd.DataFrame([item["quality"] for item in results])
    if frame["id"].duplicated().any() or not np.isfinite(
        frame.filter(like="pf_ancc_").to_numpy(np.float64)
    ).all():
        raise RuntimeError("generated candidate identity/finite guard failed")
    return frame, quality


def exp266_parity_readout(
    generated_with_target: pd.DataFrame,
    exp266_aggregate: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed_count, column in [(4, "pf_ancc_seed_mean_4"), (8, "pf_ancc_seed_mean_8")]:
        work = generated_with_target[["well", "target_tvt", column]].copy()
        error = work[column].to_numpy(np.float64) - work["target_tvt"].to_numpy(np.float64)
        work["squared_error"] = error * error
        actual = (
            work.groupby("well", sort=True)
            .agg(rows=("well", "size"), squared_error_sum=("squared_error", "sum"))
            .reset_index()
        )
        actual["actual_rmse"] = np.sqrt(
            actual["squared_error_sum"].to_numpy(np.float64)
            / actual["rows"].to_numpy(np.float64)
        )
        expected = exp266_aggregate.loc[
            exp266_aggregate["seed_count"].astype(int) == seed_count,
            ["well", "rows", "rmse"],
        ].rename(columns={"rows": "expected_rows", "rmse": "expected_rmse"})
        joined = actual.merge(expected, on="well", how="left", validate="one_to_one")
        if joined["expected_rmse"].isna().any() or not np.array_equal(
            joined["rows"].to_numpy(), joined["expected_rows"].to_numpy()
        ):
            raise RuntimeError(f"exp266 seed_count={seed_count} coverage parity failed")
        joined["abs_rmse_diff"] = np.abs(
            joined["actual_rmse"].to_numpy(np.float64)
            - joined["expected_rmse"].to_numpy(np.float64)
        )
        max_diff = float(joined["abs_rmse_diff"].max())
        if max_diff > tolerance:
            raise RuntimeError(
                f"exp266 seed_count={seed_count} RMSE parity failed: {max_diff} > {tolerance}"
            )
        rows.append(
            {
                "candidate": column,
                "seed_count": seed_count,
                "wells": len(joined),
                "max_abs_per_well_rmse_diff": max_diff,
                "mean_abs_per_well_rmse_diff": float(joined["abs_rmse_diff"].mean()),
                "rows_exact": True,
                "passed": True,
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 6. Exp263 core-bank loading helpers

# %%
def resolve_exp263_cache_root(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    filename = str(nested(config, "data.exp263.manifest_filename"))
    expected_sha = str(nested(config, "data.exp263.expected_manifest_sha256"))
    path = resolve_file_by_sha(filename, expected_sha)
    manifest = json.loads(path.read_text())
    required = {
        "schema_version": str(nested(config, "data.exp263.expected_schema_version")),
        "rows": int(nested(config, "validation.expected_rows")),
        "wells": int(nested(config, "validation.expected_wells")),
        "core_candidates": int(nested(config, "data.exp263.expected_core_candidates")),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"exp263 manifest contract failed for {key}")
    return path.parent, manifest


def load_exp263_core_bank(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[dict[str, Any]]]:
    root, manifest = resolve_exp263_cache_root(config)
    candidate_ids = [str(value) for value in nested(config, "data.exp263.core_candidate_ids")]
    manifest_ids = list((manifest.get("candidate_value_partitions") or {}).keys())
    if set(candidate_ids) != set(manifest_ids) or len(candidate_ids) != 12:
        raise RuntimeError("exp263 core candidate inventory changed")
    expected_rows = int(nested(config, "validation.expected_rows"))
    values = np.empty((expected_rows, len(candidate_ids)), dtype=np.float32)
    identity: pd.DataFrame | None = None
    input_rows: list[dict[str, Any]] = [
        {
            "kind": "exp263_cache_manifest",
            "path": str(root / "cache_manifest.json"),
            "filename": "cache_manifest.json",
            "bytes": int((root / "cache_manifest.json").stat().st_size),
            "raw_sha256": sha256_path(root / "cache_manifest.json"),
            "decompressed_sha256": sha256_path(root / "cache_manifest.json"),
        }
    ]
    for candidate_index, candidate_id in enumerate(candidate_ids):
        partition_rows = manifest["candidate_value_partitions"][candidate_id]
        paths = sorted((root / "candidate_values" / candidate_id).glob("fold=*/*.parquet"))
        if len(paths) != 5 or len(partition_rows) != 5:
            raise RuntimeError(f"exp263 partition count changed for {candidate_id}")
        frames: list[pd.DataFrame] = []
        for path, expected in zip(paths, partition_rows, strict=True):
            file_sha = sha256_path(path)
            if file_sha != str(expected["file_sha256"]):
                raise RuntimeError(f"exp263 partition SHA failed for {candidate_id}/{path}")
            frame = pd.read_parquet(
                path,
                columns=[
                    "id",
                    "well",
                    "well_row_idx",
                    "outer_fold",
                    "md_since",
                    "candidate_tvt",
                    "candidate_available",
                    "coverage_valid",
                ],
            )
            frames.append(frame)
            input_rows.append(
                {
                    "kind": "exp263_candidate_partition",
                    "candidate": candidate_id,
                    "path": str(path),
                    "filename": path.name,
                    "bytes": int(path.stat().st_size),
                    "raw_sha256": file_sha,
                    "decompressed_sha256": str(expected["content_sha256"]),
                }
            )
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != expected_rows:
            raise RuntimeError(f"exp263 row coverage changed for {candidate_id}")
        if not combined["candidate_available"].astype(bool).all() or not combined[
            "coverage_valid"
        ].astype(bool).all():
            raise RuntimeError(f"exp263 candidate coverage is incomplete for {candidate_id}")
        current_identity = combined[
            ["id", "well", "well_row_idx", "outer_fold", "md_since"]
        ].copy()
        current_identity["id"] = current_identity["id"].astype(str)
        current_identity["well"] = current_identity["well"].astype(str)
        if identity is None:
            identity = current_identity
        elif not identity.equals(current_identity):
            raise RuntimeError(f"exp263 candidate identity mismatch for {candidate_id}")
        candidate_values = combined["candidate_tvt"].to_numpy(np.float32)
        if not np.isfinite(candidate_values).all():
            raise RuntimeError(f"exp263 nonfinite candidate values for {candidate_id}")
        values[:, candidate_index] = candidate_values
    if identity is None:
        raise RuntimeError("exp263 identity was not loaded")
    order = np.lexsort(
        (
            identity["well_row_idx"].to_numpy(np.int64),
            identity["well"].astype(str).to_numpy(),
        )
    )
    identity = identity.iloc[order].reset_index(drop=True)
    values = values[order]
    return identity, values, candidate_ids, input_rows


# %% [markdown]
# ## 7. Candidate and oracle diagnostic helpers

# %%
DISTANCE_BUCKETS = (
    ("000_050", 0.0, 50.0),
    ("050_100", 50.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1000", 500.0, 1000.0),
    ("1000_plus", 1000.0, math.inf),
)


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(error)),
        "within10": float(np.mean(absolute <= 10.0)),
    }


def hidden_like_masks(
    config: dict[str, Any], wells: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bundled = Path(str(nested(config, "data.hidden_like_assignments.bundled_path")))
    path = resolve_file(
        str(nested(config, "data.hidden_like_assignments.filename")),
        [str(bundled)],
    )
    raw_sha = sha256_path(path)
    if raw_sha != str(nested(config, "data.hidden_like_assignments.expected_sha256")):
        raise RuntimeError("hidden-like assignment SHA guard failed")
    assignments = pd.read_csv(path, dtype={"well_id": "string"}).set_index("well_id")
    spatial = assignments["verification_like_spatial_role"].to_dict()
    typewell = assignments["verification_like_typewell_purged_role"].to_dict()
    masks = {
        "verification_like_spatial": np.asarray(
            [spatial.get(str(well)) == "valid" for well in wells]
        ),
        "verification_like_typewell_purged": np.asarray(
            [typewell.get(str(well)) == "valid" for well in wells]
        ),
    }
    manifest = {
        "kind": "hidden_like_assignments",
        "path": str(path),
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "raw_sha256": raw_sha,
        "decompressed_sha256": raw_sha,
    }
    return masks, manifest


def readout_masks(
    md_since: np.ndarray, hidden_masks: dict[str, np.ndarray]
) -> dict[tuple[str, str], np.ndarray]:
    masks: dict[tuple[str, str], np.ndarray] = {
        ("overall", "all"): np.ones(len(md_since), dtype=bool)
    }
    for name, lower, upper in DISTANCE_BUCKETS:
        masks[("distance_bucket", name)] = (md_since >= lower) & (md_since < upper)
    masks.update({("hidden_like", name): mask for name, mask in hidden_masks.items()})
    return masks


def metric_rows_for_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    masks: dict[tuple[str, str], np.ndarray],
    **labels: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (readout_scope, readout_value), mask in masks.items():
        count = int(mask.sum())
        metrics = (
            regression_metrics(prediction[mask], target[mask])
            if count
            else {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "within10": np.nan}
        )
        rows.append(
            {
                **labels,
                "readout_scope": readout_scope,
                "readout_value": readout_value,
                "rows": count,
                **metrics,
            }
        )
    return rows


def well_spans(wells: np.ndarray) -> list[tuple[str, int, int]]:
    if len(wells) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(wells[1:] != wells[:-1]) + 1]
    stops = np.r_[starts[1:], len(wells)]
    return [
        (str(wells[start]), int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    ]


def oracle_prediction(
    values: np.ndarray,
    target: np.ndarray,
    wells: np.ndarray,
    candidate_indices: list[int],
    candidate_names: list[str],
    *,
    scope: str,
    core_count: int,
    tie_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = np.empty(len(target), dtype=np.float32)
    selected_units = {name: 0 for name in candidate_names}
    selected_rows = {name: 0 for name in candidate_names}
    unique_units = {name: 0 for name in candidate_names[core_count:]}
    unique_rows = {name: 0 for name in candidate_names[core_count:]}
    unit_count = 0

    def assign(indices: np.ndarray) -> None:
        nonlocal unit_count
        block = values[indices][:, candidate_indices].astype(np.float64)
        truth = target[indices].astype(np.float64)
        rmse = np.sqrt(np.mean((block - truth[:, None]) ** 2, axis=0))
        local = int(np.argmin(rmse))
        global_index = candidate_indices[local]
        name = candidate_names[global_index]
        prediction[indices] = values[indices, global_index]
        selected_units[name] += 1
        selected_rows[name] += len(indices)
        unit_count += 1
        if global_index >= core_count:
            core_best = float(np.min(rmse[:core_count]))
            if float(rmse[local]) + tie_tolerance < core_best:
                unique_units[name] += 1
                unique_rows[name] += len(indices)

    if scope == "row":
        block = values[:, candidate_indices].astype(np.float64)
        absolute = np.abs(block - target.astype(np.float64)[:, None])
        local_best = np.argmin(absolute, axis=1)
        global_best = np.asarray(candidate_indices, dtype=np.int16)[local_best]
        prediction[:] = values[np.arange(len(values)), global_best]
        unit_count = len(values)
        core_best = np.min(absolute[:, :core_count], axis=1)
        for global_index in candidate_indices:
            name = candidate_names[global_index]
            selected = global_best == global_index
            count = int(selected.sum())
            selected_units[name] = count
            selected_rows[name] = count
            if global_index >= core_count:
                unique = selected & (
                    absolute[:, candidate_indices.index(global_index)] + tie_tolerance < core_best
                )
                unique_count = int(unique.sum())
                unique_units[name] = unique_count
                unique_rows[name] = unique_count
    else:
        block_size = None
        if scope.startswith("block_"):
            block_size = int(scope.split("_", 1)[1])
        elif scope != "whole_well":
            raise ValueError(f"unknown oracle scope: {scope}")
        for _, start, stop in well_spans(wells):
            if block_size is None:
                assign(np.arange(start, stop, dtype=np.int64))
            else:
                for left in range(start, stop, block_size):
                    assign(np.arange(left, min(left + block_size, stop), dtype=np.int64))
    summary = {
        "unit_count": unit_count,
        "selected_units": selected_units,
        "selected_rows": selected_rows,
        "unique_best_units": unique_units,
        "unique_best_rows": unique_rows,
    }
    return prediction, summary


def by_well_metrics(
    prediction: np.ndarray, target: np.ndarray, wells: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, start, stop in well_spans(wells):
        metrics = regression_metrics(prediction[start:stop], target[start:stop])
        rows.append({"well": well, "rows": stop - start, **metrics})
    return pd.DataFrame(rows)


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = pd.Series(left).rank(method="average")
    b = pd.Series(right).rank(method="average")
    return float(a.corr(b))


def disagreement_readout(
    generated: pd.DataFrame,
    target: np.ndarray,
    core_values: np.ndarray,
    quantiles: int,
    tie_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean4 = generated["pf_ancc_seed_mean_4"].to_numpy(np.float64)
    mean8 = generated["pf_ancc_seed_mean_8"].to_numpy(np.float64)
    std4 = generated["pf_ancc_seed_std_4"].to_numpy(np.float64)
    std8 = generated["pf_ancc_seed_std_8"].to_numpy(np.float64)
    error4 = np.abs(mean4 - target)
    error8 = np.abs(mean8 - target)
    core_best = np.min(np.abs(core_values.astype(np.float64) - target[:, None]), axis=1)
    unique8 = error8 + tie_tolerance < core_best
    unique4 = error4 + tie_tolerance < core_best
    metrics = pd.DataFrame(
        [
            {
                "metric": "mean8_minus_mean4",
                "mean_abs": float(np.mean(np.abs(mean8 - mean4))),
                "rmse": float(np.sqrt(np.mean((mean8 - mean4) ** 2))),
                "max_abs": float(np.max(np.abs(mean8 - mean4))),
                "spearman_seed_std_vs_abs_error": np.nan,
                "unique_best_rate": np.nan,
            },
            {
                "metric": "seed_mean_4",
                "mean_abs": float(np.mean(error4)),
                "rmse": float(np.sqrt(np.mean((mean4 - target) ** 2))),
                "max_abs": float(np.max(error4)),
                "spearman_seed_std_vs_abs_error": spearman_rank_correlation(std4, error4),
                "unique_best_rate": float(np.mean(unique4)),
            },
            {
                "metric": "seed_mean_8",
                "mean_abs": float(np.mean(error8)),
                "rmse": float(np.sqrt(np.mean((mean8 - target) ** 2))),
                "max_abs": float(np.max(error8)),
                "spearman_seed_std_vs_abs_error": spearman_rank_correlation(std8, error8),
                "unique_best_rate": float(np.mean(unique8)),
            },
        ]
    )
    bucket = pd.DataFrame(
        {
            "seed_std_8": std8,
            "abs_error_4": error4,
            "abs_error_8": error8,
            "unique_best_4": unique4,
            "unique_best_8": unique8,
            "mean8_abs_minus_mean4": np.abs(mean8 - mean4),
        }
    )
    bucket["std_quantile"] = pd.qcut(
        bucket["seed_std_8"], q=quantiles, labels=False, duplicates="drop"
    )
    buckets = (
        bucket.groupby("std_quantile", dropna=False)
        .agg(
            rows=("seed_std_8", "size"),
            seed_std_8_mean=("seed_std_8", "mean"),
            seed_std_8_min=("seed_std_8", "min"),
            seed_std_8_max=("seed_std_8", "max"),
            abs_error_4_mean=("abs_error_4", "mean"),
            abs_error_8_mean=("abs_error_8", "mean"),
            unique_best_4_rate=("unique_best_4", "mean"),
            unique_best_8_rate=("unique_best_8", "mean"),
            mean8_abs_minus_mean4=("mean8_abs_minus_mean4", "mean"),
        )
        .reset_index()
    )
    return metrics, buckets


# %% [markdown]
# ## 8. Setup and fixed execution contract

# %%
def validate_execution_contract(config: dict[str, Any]) -> None:
    expected = {
        "execution.active_variant_count": 1,
        "execution.pf_dynamics_variants": 1,
        "execution.generated_seed_count": 8,
        "execution.lightgbm_config_count": 0,
        "execution.fold_count": 0,
        "execution.total_boosters": 0,
    }
    for key, value in expected.items():
        if int(nested(config, key, -1)) != value:
            raise RuntimeError(f"execution contract changed for {key}")
    if nested(config, "execution.parent_control_retraining") is not False:
        raise RuntimeError("parent/control retraining must remain disabled")
    if nested(config, "execution.inference_enabled") is not False or nested(
        config, "execution.submission_enabled"
    ) is not False:
        raise RuntimeError("inference/submission must remain disabled")
    if str(nested(config, "reproducibility.upstream_seed_namespace")) != (
        UPSTREAM_SEED_NAMESPACE
    ):
        raise RuntimeError("upstream seed namespace changed")


# %% [markdown]
# ## 9. Input checks and target-free PF generation
#
# exp072 identityとraw trainを確認し、targetを渡さずに固定8 seed pathを生成・保存する。

# %% [markdown]
# ## 10. Upstream parity and core-bank assembly
#
# seed 0をexp072、mean4/mean8 per-well RMSEをexp266へ照合してから、exp263 core-12を読む。

# %% [markdown]
# ## 11. Candidate, oracle, and disagreement readout
#
# standalone、row/block/well oracle、distance/hidden-like、seed disagreementを固定scopeで測る。

# %% [markdown]
# ## 12. Metrics and generated artifacts
#
# target-free candidate path、readout、input/artifact manifest、SHA、summaryを保存する。

# %%
def run_experiment() -> dict[str, Any]:
    require_kaggle_or_explicit_local()
    config, config_path = load_config()
    validate_execution_contract(config)
    competition_root = find_competition_root(config)
    train_dir = competition_root / "train"
    output_root = (
        Path("/kaggle/working")
        if is_kaggle_runtime()
        else Path.cwd() / "experiments" / EXPERIMENT_NAME
    )
    artifact_dir = output_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    workers = int(nested(config, "runtime.num_workers"))
    seed_count = int(nested(config, "execution.generated_seed_count"))
    progress_every = int(nested(config, "execution.progress_every_wells"))
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": nested(config, "experiment.route"),
                "config": str(config_path),
                "competition_root": str(competition_root),
                "PF dynamics variants": 1,
                "particles": ANCC_N,
                "generated seeds": seed_count,
                "aggregate seed counts": [4, 8],
                "LightGBM configs": 0,
                "folds": 0,
                "boosters": 0,
                "parent/control retraining": False,
                "GPU": False,
                "inference": False,
                "submission": False,
            },
            indent=2,
        ),
        flush=True,
    )

    reference, _, exp072_manifest = load_exp072_reference(config)
    wells = sorted(reference["well"].unique().tolist())
    generation_started = time.perf_counter()
    generated, quality = generate_all_candidates(
        train_dir,
        wells,
        seed_count=seed_count,
        workers=workers,
        progress_every=progress_every,
    )
    generation_seconds = time.perf_counter() - generation_started
    expected_rows = int(nested(config, "validation.expected_rows"))
    if len(generated) != expected_rows or generated["well"].nunique() != len(wells):
        raise RuntimeError("generated candidate coverage guard failed")

    # Candidate path artifact is target-free by construction and is written before target join.
    candidate_filename = str(nested(config, "audit.outputs.candidate_paths"))
    candidate_path = write_csv(
        generated,
        artifact_dir / candidate_filename,
        gzip_output=True,
    )

    # 10. Upstream parity and core-bank assembly

    raw_targets = load_raw_targets_after_candidate_freeze(train_dir, wells)
    reference_indexed = reference.set_index("id", verify_integrity=True)
    generated_indexed = generated.set_index("id", verify_integrity=True)
    raw_target_indexed = raw_targets.set_index("id", verify_integrity=True)
    if not (
        set(reference_indexed.index)
        == set(generated_indexed.index)
        == set(raw_target_indexed.index)
    ):
        raise RuntimeError("generated/raw-target/exp072 identity sets differ")
    generated_with_target = generated_indexed.loc[reference_indexed.index].reset_index()
    generated_with_target["target_tvt"] = raw_target_indexed.loc[
        reference_indexed.index, "target_tvt"
    ].to_numpy(np.float64)
    generated_with_target["md_since"] = reference["md_since"].to_numpy(np.float32)
    generated_with_target["exp072_pf_ancc"] = reference["pf_ancc"].to_numpy(np.float32)
    seed0_diff = np.abs(
        generated_with_target["pf_ancc_seed0"].to_numpy(np.float32).astype(np.float64)
        - generated_with_target["exp072_pf_ancc"].to_numpy(np.float32).astype(np.float64)
    )
    seed0_max_abs = float(seed0_diff.max(initial=0.0))
    if seed0_max_abs > float(nested(config, "validation.parity_tolerance_ft")):
        raise RuntimeError(f"seed0 exp072 parity failed: max_abs={seed0_max_abs}")

    exp266_aggregate, exp266_manifests = load_exp266_aggregate(config)
    exp266_parity = exp266_parity_readout(
        generated_with_target,
        exp266_aggregate,
        float(nested(config, "validation.aggregate_rmse_tolerance_ft")),
    )

    identity, core_values, core_names, exp263_manifests = load_exp263_core_bank(config)
    aligned_reference = reference_indexed.loc[identity["id"].astype(str)].reset_index()
    aligned_generated = generated_indexed.loc[identity["id"].astype(str)].reset_index()
    aligned_target = raw_target_indexed.loc[identity["id"].astype(str)].reset_index()
    if not np.array_equal(
        aligned_reference["well"].astype(str).to_numpy(),
        identity["well"].astype(str).to_numpy(),
    ) or not np.array_equal(
        aligned_reference["row_idx"].to_numpy(np.int64),
        identity["well_row_idx"].to_numpy(np.int64),
    ):
        raise RuntimeError("exp263 identity does not align to exp072")
    if not np.allclose(
        aligned_reference["md_since"].to_numpy(np.float64),
        identity["md_since"].to_numpy(np.float64),
        rtol=0.0,
        atol=1e-4,
        equal_nan=True,
    ):
        raise RuntimeError("exp263 md_since does not align to exp072")

    target = aligned_target["target_tvt"].to_numpy(np.float64)
    md_since = aligned_reference["md_since"].to_numpy(np.float64)
    row_wells = identity["well"].astype(str).to_numpy()
    mean4 = aligned_generated["pf_ancc_seed_mean_4"].to_numpy(np.float32)
    mean8 = aligned_generated["pf_ancc_seed_mean_8"].to_numpy(np.float32)
    all_names = [*core_names, "pf_ancc_seed_mean_4", "pf_ancc_seed_mean_8"]
    all_values = np.column_stack([core_values, mean4, mean8]).astype(np.float32)
    hidden_masks, hidden_manifest = hidden_like_masks(config, row_wells)
    masks = readout_masks(md_since, hidden_masks)

    # 11. Candidate, oracle, and disagreement readout

    standalone_rows: list[dict[str, Any]] = []
    direct_by_well_rows: list[pd.DataFrame] = []
    standalone_predictions = {
        "exp072_pf_ancc_seed0": aligned_reference["pf_ancc"].to_numpy(np.float32),
        "pf_ancc_seed_mean_4": mean4,
        "pf_ancc_seed_mean_8": mean8,
    }
    for candidate, prediction in standalone_predictions.items():
        standalone_rows.extend(
            metric_rows_for_prediction(
                prediction,
                target,
                masks,
                candidate=candidate,
            )
        )
        frame = by_well_metrics(prediction, target, row_wells)
        frame.insert(0, "configuration", candidate)
        frame.insert(1, "oracle_scope", "direct")
        frame["baseline_core12_rmse"] = np.nan
        frame["delta_vs_core12"] = np.nan
        direct_by_well_rows.append(frame)
    standalone_metrics = pd.DataFrame(standalone_rows)

    expected_global = nested(config, "validation.expected_global_rmse")
    for candidate in ["pf_ancc_seed_mean_4", "pf_ancc_seed_mean_8"]:
        actual = float(
            standalone_metrics.loc[
                (standalone_metrics["candidate"] == candidate)
                & (standalone_metrics["readout_scope"] == "overall"),
                "rmse",
            ].iloc[0]
        )
        expected = float(expected_global[candidate])
        if abs(actual - expected) > float(
            nested(config, "validation.aggregate_rmse_tolerance_ft")
        ):
            raise RuntimeError(f"global RMSE parity failed for {candidate}: {actual} != {expected}")

    config_additions = nested(config, "audit.bank_configurations")
    scopes = [str(value) for value in nested(config, "audit.scopes")]
    tie_tolerance = float(nested(config, "validation.tie_tolerance_ft"))
    oracle_rows: list[dict[str, Any]] = []
    oracle_by_well_rows: list[pd.DataFrame] = []
    core_indices = list(range(len(core_names)))
    for scope in scopes:
        baseline_prediction, _ = oracle_prediction(
            all_values,
            target,
            row_wells,
            core_indices,
            all_names,
            scope=scope,
            core_count=len(core_names),
            tie_tolerance=tie_tolerance,
        )
        baseline_by_well = by_well_metrics(baseline_prediction, target, row_wells).rename(
            columns={"rmse": "baseline_core12_rmse"}
        )
        for configuration, additions in config_additions.items():
            addition_names = [str(value) for value in additions]
            indices = [*core_indices, *(all_names.index(name) for name in addition_names)]
            prediction, selection = oracle_prediction(
                all_values,
                target,
                row_wells,
                indices,
                all_names,
                scope=scope,
                core_count=len(core_names),
                tie_tolerance=tie_tolerance,
            )
            current_by_well = by_well_metrics(prediction, target, row_wells)
            joined = current_by_well.merge(
                baseline_by_well[["well", "baseline_core12_rmse"]],
                on="well",
                how="left",
                validate="one_to_one",
            )
            joined["delta_vs_core12"] = (
                joined["rmse"] - joined["baseline_core12_rmse"]
            )
            joined.insert(0, "configuration", configuration)
            joined.insert(1, "oracle_scope", scope)
            oracle_by_well_rows.append(joined)
            regression = joined["delta_vs_core12"].to_numpy(np.float64)
            labels = {
                "configuration": configuration,
                "oracle_scope": scope,
                "candidate_count": len(indices),
                "unit_count": int(selection["unit_count"]),
                "selected_units_json": json.dumps(selection["selected_units"], sort_keys=True),
                "selected_rows_json": json.dumps(selection["selected_rows"], sort_keys=True),
                "unique_best_units_json": json.dumps(
                    selection["unique_best_units"], sort_keys=True
                ),
                "unique_best_rows_json": json.dumps(
                    selection["unique_best_rows"], sort_keys=True
                ),
                "wells_improved_vs_core12": int(np.sum(regression < -1e-12)),
                "wells_worsened_vs_core12": int(np.sum(regression > 1e-12)),
                "max_well_regression_rmse": float(np.max(regression)),
                "max_well_improvement_rmse": float(np.min(regression)),
            }
            oracle_rows.extend(
                metric_rows_for_prediction(prediction, target, masks, **labels)
            )

    oracle_metrics = pd.DataFrame(oracle_rows)
    oracle_by_well = pd.concat(
        [*direct_by_well_rows, *oracle_by_well_rows], ignore_index=True
    )
    disagreement_metrics, disagreement_buckets = disagreement_readout(
        aligned_generated,
        target,
        core_values,
        int(nested(config, "audit.disagreement_quantiles")),
        tie_tolerance,
    )

    # 12. Metrics and generated artifacts

    output_names = nested(config, "audit.outputs")
    parity_path = write_csv(
        exp266_parity, artifact_dir / str(output_names["exp266_parity"])
    )
    standalone_path = write_csv(
        standalone_metrics, artifact_dir / str(output_names["standalone_metrics"])
    )
    oracle_path = write_csv(
        oracle_metrics, artifact_dir / str(output_names["oracle_metrics"])
    )
    by_well_path = write_csv(
        oracle_by_well,
        artifact_dir / str(output_names["oracle_by_well"]),
        gzip_output=True,
    )
    disagreement_path = write_csv(
        disagreement_metrics,
        artifact_dir / str(output_names["seed_disagreement_metrics"]),
    )
    disagreement_bucket_path = write_csv(
        disagreement_buckets,
        artifact_dir / str(output_names["seed_disagreement_buckets"]),
    )

    input_rows = [exp072_manifest, *exp266_manifests, *exp263_manifests, hidden_manifest]
    for item in quality.to_dict(orient="records"):
        input_rows.extend(
            [
                {
                    "kind": "raw_horizontal",
                    "well": item["well"],
                    "path": item["horizontal_path"],
                    "filename": Path(item["horizontal_path"]).name,
                    "bytes": Path(item["horizontal_path"]).stat().st_size,
                    "raw_sha256": item["horizontal_sha256"],
                    "decompressed_sha256": item["horizontal_sha256"],
                },
                {
                    "kind": "raw_typewell",
                    "well": item["well"],
                    "path": item["typewell_path"],
                    "filename": Path(item["typewell_path"]).name,
                    "bytes": Path(item["typewell_path"]).stat().st_size,
                    "raw_sha256": item["typewell_sha256"],
                    "decompressed_sha256": item["typewell_sha256"],
                },
            ]
        )
    input_manifest = pd.DataFrame(input_rows)
    input_manifest_path = write_csv(
        input_manifest, artifact_dir / str(output_names["input_manifest"])
    )

    artifact_paths = [
        candidate_path,
        parity_path,
        standalone_path,
        oracle_path,
        by_well_path,
        disagreement_path,
        disagreement_bucket_path,
        input_manifest_path,
    ]
    artifact_manifest = pd.DataFrame(artifact_manifest_rows(artifact_paths))
    artifact_manifest["schema_sha256"] = [
        schema_sha(frame)
        for frame in [
            generated,
            exp266_parity,
            standalone_metrics,
            oracle_metrics,
            oracle_by_well,
            disagreement_metrics,
            disagreement_buckets,
            input_manifest,
        ]
    ]
    artifact_manifest_path = write_csv(
        artifact_manifest, artifact_dir / str(output_names["artifact_manifest"])
    )

    total_seconds = time.perf_counter() - started
    overall_standalone = standalone_metrics.loc[
        standalone_metrics["readout_scope"] == "overall"
    ].set_index("candidate")["rmse"].to_dict()
    overall_oracle = oracle_metrics.loc[
        oracle_metrics["readout_scope"] == "overall"
    ][
        [
            "configuration",
            "oracle_scope",
            "rmse",
            "wells_improved_vs_core12",
            "wells_worsened_vs_core12",
            "max_well_regression_rmse",
            "unique_best_rows_json",
        ]
    ].to_dict(orient="records")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_candidate_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "route": "pf_beam",
        "runtime": {
            "total_seconds": total_seconds,
            "generation_seconds": generation_seconds,
            "num_workers": workers,
            "gpu": False,
        },
        "execution": {
            "rows": len(identity),
            "wells": int(pd.Series(row_wells).nunique()),
            "pf_dynamics_variants": 1,
            "particles": ANCC_N,
            "generated_seed_count": seed_count,
            "aggregate_seed_counts": [4, 8],
            "lightgbm_configs": 0,
            "folds": 0,
            "boosters": 0,
            "parent_control_retraining": False,
            "inference": False,
            "submission": False,
        },
        "target": {
            "source": "raw_horizontal_evaluation_TVT_float64_after_candidate_freeze",
            "rows": len(raw_targets),
        },
        "parity": {
            "seed0_max_abs_vs_exp072": seed0_max_abs,
            "exp266_mean4_mean8": exp266_parity.to_dict(orient="records"),
        },
        "standalone_overall_rmse": overall_standalone,
        "oracle_overall": overall_oracle,
        "seed_disagreement": disagreement_metrics.to_dict(orient="records"),
        "candidate_path": {
            "path": str(candidate_path),
            "raw_sha256": sha256_path(candidate_path),
            "decompressed_sha256": sha256_gzip_content(candidate_path),
            "schema_sha256": schema_sha(generated),
        },
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "config_sha256": sha256_path(config_path),
        "model_sha": "not_applicable_no_training",
        "submission_sha": "not_applicable_no_submission",
        "notes": [
            "Candidate generation was frozen before target join.",
            "Evaluation target is raw horizontal TVT, matching the exp266 aggregate contract.",
            "Oracle headroom does not establish a deployable selector.",
            "Raw-test inference and submission are disabled.",
        ],
    }
    summary_path = artifact_dir / str(output_names["summary"])
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n")
    (output_root / "metrics.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "runtime": summary["runtime"],
                "execution": summary["execution"],
                "parity": summary["parity"],
                "standalone_overall_rmse": summary["standalone_overall_rmse"],
                "candidate_path": summary["candidate_path"],
                "artifacts": {
                    "summary": str(summary_path),
                    "manifest": str(artifact_manifest_path),
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


# %%
if __name__ == "__main__":
    RUN_SUMMARY = run_experiment()
