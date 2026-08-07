from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp142_trajectory_aware_pf_transition_prior"
EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
EXP072_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


@dataclass(frozen=True)
class ExistingCandidate:
    name: str
    source_column: str
    transform: str
    role: str


@dataclass(frozen=True)
class StrictPfZFit:
    beta: float
    intercept: float
    zsig: float
    initial_velocity: float
    gr_sigma: float
    known_rows: int
    eval_rows: int


@dataclass(frozen=True)
class TrajectoryVariant:
    name: str
    transition_strength: float
    z_accel_gain: float
    prefix_slope_strength: float
    velocity_noise_gain: float
    position_noise_gain: float
    likelihood_sigma_gain: float
    curve_clip: float


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            EXP072_ARTIFACTS / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def _as_float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def _read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _source_for_candidate(header: set[str], base_name: str) -> ExistingCandidate:
    if base_name == "pf_z":
        if "pf_z" not in header:
            raise ValueError("exp072 feature cache is missing required baseline column: pf_z")
        return ExistingCandidate("exp072_pf_z", "pf_z", "absolute", "exp072_pf_z")
    delta = f"{base_name}_d"
    if delta in header:
        return ExistingCandidate(f"exp072_{base_name}", delta, "base_plus_delta", "exp072_likpf")
    if base_name in header:
        return ExistingCandidate(f"exp072_{base_name}", base_name, "absolute", "exp072_likpf")
    raise ValueError(f"exp072 feature cache is missing required baseline column: {base_name}(_d)")


def build_existing_candidate_specs(
    header: list[str],
    scales: list[float],
) -> list[ExistingCandidate]:
    header_set = set(header)
    specs = [
        _source_for_candidate(header_set, "pf_z"),
        _source_for_candidate(header_set, "pf_ancc"),
        _source_for_candidate(header_set, "beam_mean"),
        _source_for_candidate(header_set, "likpf_mean"),
    ]
    for scale in scales:
        base_name = f"likpf_scale_{scale:g}"
        if f"{base_name}_d" in header_set or base_name in header_set:
            specs.append(_source_for_candidate(header_set, base_name))
    return specs


def read_exp072_cache(
    config: dict[str, Any],
    specs: list[ExistingCandidate],
    *,
    max_rows: int | None,
    max_wells: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = get_nested(config, "data.exp072_train_feature_cache_local")
    source = find_artifact(EXP072_TRAIN_FEATURES, explicit)
    required = {"id", "well", "target", "last_known_tvt", "md_since"}
    required.update(spec.source_column for spec in specs)
    frame = pd.read_csv(
        source,
        usecols=sorted(required),
        nrows=max_rows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if max_wells is not None:
        keep = sorted(frame["well"].unique())[: int(max_wells)]
        frame = frame[frame["well"].isin(keep)].reset_index(drop=True)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame.empty:
        raise ValueError("exp072 feature cache read produced zero rows")
    schema_path: Path | None
    try:
        schema_path = find_artifact(
            EXP072_FEATURE_SCHEMA,
            get_nested(config, "data.exp072_feature_schema_local"),
        )
    except FileNotFoundError:
        schema_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=True)
        if source.suffix == ".gz"
        else None,
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    return frame, metadata


@njit(cache=True)
def _interp1(grid, value, vmin, step):
    i = int((value - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (value - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _grid_interp(tvt, gr, step):
    tmin = tvt[0]
    tmax = tvt[-1]
    size = int(np.ceil((tmax - tmin) / step)) + 1
    out = np.empty(size)
    k = 0
    for i in range(size):
        x = tmin + i * step
        while k < len(tvt) - 2 and tvt[k + 1] < x:
            k += 1
        denom = tvt[k + 1] - tvt[k]
        if denom <= 1e-9:
            out[i] = gr[k]
        else:
            a = (x - tvt[k]) / denom
            out[i] = gr[k] * (1.0 - a) + gr[k + 1] * a
    return out, tmin, step


@njit(cache=True, nogil=True)
def _resamp_exp072(pos, aux, weights, n_particles, rough_pos, rough_vel):
    cum = np.zeros(n_particles + 1)
    for j in range(n_particles):
        cum[j + 1] = cum[j] + weights[j]
    out_pos = np.empty(n_particles)
    out_aux = np.empty(n_particles)
    u0 = np.random.uniform(0.0, 1.0 / n_particles)
    cursor = 0
    for j in range(n_particles):
        u = u0 + j / n_particles
        while cursor < n_particles - 1 and cum[cursor + 1] < u:
            cursor += 1
        out_pos[j] = pos[cursor] + rough_pos * np.random.randn()
        out_aux[j] = aux[cursor] + rough_vel * np.random.randn()
    return out_pos, out_aux


@njit(cache=True, nogil=True)
def _strict_pf_z_seeded_exp072(
    seed,
    md_v,
    z_v,
    gr_v,
    gr_sm_v,
    gg_p,
    gg_s,
    vmin,
    step,
    gr_sigma,
    initial_pos,
    initial_velocity,
    beta,
    intercept,
    zsig,
    n_particles,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_pos,
    rough_vel,
    resample_threshold,
):
    np.random.seed(seed)
    pos = np.empty(n_particles)
    vel = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for j in range(n_particles):
        pos[j] = initial_pos + 0.5 * np.random.randn()
        vel[j] = initial_velocity + 0.02 * np.random.randn()
    preds = np.empty(len(md_v))
    stds = np.empty(len(md_v))
    prev_md = md_v[0] - 1.0
    prev_z = z_v[0] - 1.0
    log_lik = 0.0
    for i in range(len(md_v)):
        dm = md_v[i] - prev_md
        if dm < 1.0:
            dm = 1.0
        dzd = (z_v[i] - prev_z) / dm
        expected_velocity = beta * dzd + intercept
        for j in range(n_particles):
            vel[j] = momentum * vel[j] + velocity_noise * np.random.randn()
            pos[j] += vel[j] * dm + position_noise * np.random.randn()
            if pos[j] < vmin - 50.0:
                pos[j] = vmin - 50.0
            if pos[j] > vmin + len(gg_p) * step + 50.0:
                pos[j] = vmin + len(gg_p) * step + 50.0
        if not np.isnan(gr_v[i]):
            weight_sum = 0.0
            avg_lik = 0.0
            for j in range(n_particles):
                expected_raw = _interp1(gg_p, pos[j], vmin, step)
                raw_delta = (gr_v[i] - expected_raw) / gr_sigma
                raw_like = (
                    max(np.exp(-0.5 * raw_delta * raw_delta), 1e-300)
                    if raw_delta * raw_delta < 600.0
                    else 1e-300
                )
                if not np.isnan(gr_sm_v[i]):
                    expected_smooth = _interp1(gg_s, pos[j], vmin, step)
                    smooth_delta = (gr_sm_v[i] - expected_smooth) / (gr_sigma * 1.5)
                    smooth_like = (
                        max(np.exp(-0.5 * smooth_delta * smooth_delta), 1e-300)
                        if smooth_delta * smooth_delta < 600.0
                        else 1e-300
                    )
                    lik = (1.0 - gr_smooth_weight) * raw_like + gr_smooth_weight * smooth_like
                else:
                    lik = raw_like
                if lik < 1e-300:
                    lik = 1e-300
                avg_lik += weights[j] * lik
                weights[j] *= lik
                weight_sum += weights[j]
            if avg_lik < 1e-300:
                avg_lik = 1e-300
            log_lik += np.log(avg_lik)
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles
        velocity_weight_sum = 0.0
        velocity_avg_lik = 0.0
        for j in range(n_particles):
            velocity_delta = (vel[j] - expected_velocity) / max(zsig * 2.0, 0.005)
            velocity_like = (
                max(np.exp(-0.5 * velocity_delta * velocity_delta), 1e-300)
                if velocity_delta * velocity_delta < 600.0
                else 1e-300
            )
            velocity_avg_lik += weights[j] * velocity_like
            weights[j] *= velocity_like
            velocity_weight_sum += weights[j]
        if velocity_avg_lik < 1e-300:
            velocity_avg_lik = 1e-300
        log_lik += np.log(velocity_avg_lik)
        if velocity_weight_sum > 0.0:
            for j in range(n_particles):
                weights[j] /= velocity_weight_sum
        else:
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles
        neff_denom = 0.0
        for j in range(n_particles):
            neff_denom += weights[j] * weights[j]
        if 1.0 / neff_denom < resample_threshold * n_particles:
            pos, vel = _resamp_exp072(pos, vel, weights, n_particles, rough_pos, rough_vel)
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles
        weighted_mean = 0.0
        for j in range(n_particles):
            weighted_mean += weights[j] * pos[j]
        preds[i] = weighted_mean
        variance = 0.0
        for j in range(n_particles):
            variance += weights[j] * (pos[j] - weighted_mean) ** 2
        stds[i] = variance**0.5
        prev_md = md_v[i]
        prev_z = z_v[i]
    return preds, stds, log_lik


@njit(cache=True, nogil=True)
def _trajectory_pf_z_seeded(
    seed,
    md_v,
    z_v,
    gr_v,
    gr_sm_v,
    dzd_v,
    d2z_v,
    gg_p,
    gg_s,
    vmin,
    step,
    gr_sigma,
    initial_pos,
    initial_velocity,
    beta,
    intercept,
    zsig,
    prefix_tvt_slope,
    n_particles,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_pos,
    rough_vel,
    resample_threshold,
    transition_strength,
    z_accel_gain,
    prefix_slope_strength,
    velocity_noise_gain,
    position_noise_gain,
    likelihood_sigma_gain,
    curve_clip,
    collapse_std_threshold,
):
    np.random.seed(seed)
    pos = np.empty(n_particles)
    vel = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for j in range(n_particles):
        pos[j] = initial_pos + 0.5 * np.random.randn()
        vel[j] = initial_velocity + 0.02 * np.random.randn()
    preds = np.empty(len(md_v))
    stds = np.empty(len(md_v))
    prev_md = md_v[0] - 1.0
    log_lik = 0.0
    neff_sum = 0.0
    neff_min = n_particles * 1.0
    resample_count = 0
    collapse_count = 0
    upper = vmin + len(gg_p) * step + 50.0
    lower = vmin - 50.0

    for i in range(len(md_v)):
        dm = md_v[i] - prev_md
        if dm < 1.0:
            dm = 1.0
        base_velocity = beta * dzd_v[i] + intercept
        curve = abs(d2z_v[i]) * 100.0
        if curve > curve_clip:
            curve = curve_clip
        target_velocity = (
            base_velocity
            + z_accel_gain * d2z_v[i] * 100.0
            + prefix_slope_strength * (prefix_tvt_slope - base_velocity)
        )
        if target_velocity < -2.0:
            target_velocity = -2.0
        if target_velocity > 2.0:
            target_velocity = 2.0
        velocity_scale = 1.0 + velocity_noise_gain * curve
        position_scale = 1.0 + position_noise_gain * curve
        keep = momentum * (1.0 - transition_strength)
        if keep < 0.0:
            keep = 0.0
        if keep > momentum:
            keep = momentum

        for j in range(n_particles):
            vel[j] = (
                keep * vel[j]
                + (1.0 - keep) * target_velocity
                + velocity_noise * velocity_scale * np.random.randn()
            )
            pos[j] += vel[j] * dm + position_noise * position_scale * np.random.randn()
            if pos[j] < lower:
                pos[j] = lower
            if pos[j] > upper:
                pos[j] = upper

        if not np.isnan(gr_v[i]):
            weight_sum = 0.0
            avg_lik = 0.0
            for j in range(n_particles):
                expected_raw = _interp1(gg_p, pos[j], vmin, step)
                raw_delta = (gr_v[i] - expected_raw) / gr_sigma
                raw_like = (
                    max(np.exp(-0.5 * raw_delta * raw_delta), 1e-300)
                    if raw_delta * raw_delta < 600.0
                    else 1e-300
                )
                if not np.isnan(gr_sm_v[i]):
                    expected_smooth = _interp1(gg_s, pos[j], vmin, step)
                    smooth_delta = (gr_sm_v[i] - expected_smooth) / (gr_sigma * 1.5)
                    smooth_like = (
                        max(np.exp(-0.5 * smooth_delta * smooth_delta), 1e-300)
                        if smooth_delta * smooth_delta < 600.0
                        else 1e-300
                    )
                    lik = (1.0 - gr_smooth_weight) * raw_like + gr_smooth_weight * smooth_like
                else:
                    lik = raw_like
                if lik < 1e-300:
                    lik = 1e-300
                avg_lik += weights[j] * lik
                weights[j] *= lik
                weight_sum += weights[j]
            if avg_lik < 1e-300:
                avg_lik = 1e-300
            log_lik += np.log(avg_lik)
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles

        velocity_weight_sum = 0.0
        velocity_avg_lik = 0.0
        velocity_sigma = max(zsig * 2.0 * (1.0 + likelihood_sigma_gain * curve), 0.005)
        for j in range(n_particles):
            velocity_delta = (vel[j] - target_velocity) / velocity_sigma
            velocity_like = (
                max(np.exp(-0.5 * velocity_delta * velocity_delta), 1e-300)
                if velocity_delta * velocity_delta < 600.0
                else 1e-300
            )
            velocity_avg_lik += weights[j] * velocity_like
            weights[j] *= velocity_like
            velocity_weight_sum += weights[j]
        if velocity_avg_lik < 1e-300:
            velocity_avg_lik = 1e-300
        log_lik += np.log(velocity_avg_lik)
        if velocity_weight_sum > 0.0:
            for j in range(n_particles):
                weights[j] /= velocity_weight_sum
        else:
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles

        neff_denom = 0.0
        for j in range(n_particles):
            neff_denom += weights[j] * weights[j]
        neff = 1.0 / neff_denom
        neff_sum += neff
        if neff < neff_min:
            neff_min = neff
        if neff < resample_threshold * n_particles:
            pos, vel = _resamp_exp072(pos, vel, weights, n_particles, rough_pos, rough_vel)
            resample_count += 1
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles

        weighted_mean = 0.0
        for j in range(n_particles):
            weighted_mean += weights[j] * pos[j]
        preds[i] = weighted_mean
        variance = 0.0
        for j in range(n_particles):
            variance += weights[j] * (pos[j] - weighted_mean) ** 2
        stds[i] = variance**0.5
        if stds[i] < collapse_std_threshold:
            collapse_count += 1
        prev_md = md_v[i]

    n_rows = max(len(md_v), 1)
    diagnostics = np.empty(5)
    diagnostics[0] = neff_sum / (n_rows * n_particles)
    diagnostics[1] = neff_min / n_particles
    diagnostics[2] = resample_count
    diagnostics[3] = collapse_count / n_rows
    diagnostics[4] = np.mean(stds)
    return preds, stds, log_lik, diagnostics


@njit(cache=True, nogil=True)
def _strict_pf_z_allseeds(
    md_v,
    z_v,
    gr_v,
    gr_sm_v,
    gg_p,
    gg_s,
    vmin,
    step,
    gr_sigma,
    initial_pos,
    initial_velocity,
    beta,
    intercept,
    zsig,
    n_particles,
    n_seeds,
    seed_vector,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_pos,
    rough_vel,
    resample_threshold,
):
    n = len(md_v)
    preds = np.empty((n_seeds, n))
    stds = np.empty((n_seeds, n))
    liks = np.empty(n_seeds)
    upper = vmin + len(gg_p) * step + 50.0
    lower = vmin - 50.0
    for seed_offset in range(n_seeds):
        np.random.seed(seed_vector[seed_offset])
        pos = np.empty(n_particles)
        vel = np.empty(n_particles)
        weights = np.ones(n_particles) / n_particles
        for j in range(n_particles):
            pos[j] = initial_pos + 0.5 * np.random.randn()
            vel[j] = initial_velocity + 0.02 * np.random.randn()
        log_lik = 0.0
        prev_md = md_v[0] - 1.0
        prev_z = z_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0
            dzd = (z_v[i] - prev_z) / dm
            expected_velocity = beta * dzd + intercept
            for j in range(n_particles):
                vel[j] = momentum * vel[j] + velocity_noise * np.random.randn()
                pos[j] += vel[j] * dm + position_noise * np.random.randn()
                if pos[j] < lower:
                    pos[j] = lower
                if pos[j] > upper:
                    pos[j] = upper

            avg_lik = 0.0
            if not np.isnan(gr_v[i]):
                for j in range(n_particles):
                    expected_raw = _interp1(gg_p, pos[j], vmin, step)
                    raw_delta = (gr_v[i] - expected_raw) / gr_sigma
                    raw_dd = raw_delta * raw_delta
                    if raw_dd > 600.0:
                        raw_like = 0.0
                    else:
                        raw_like = np.exp(-0.5 * raw_dd)
                    if raw_like < 1e-300:
                        raw_like = 1e-300
                    if not np.isnan(gr_sm_v[i]):
                        expected_smooth = _interp1(gg_s, pos[j], vmin, step)
                        smooth_delta = (gr_sm_v[i] - expected_smooth) / (gr_sigma * 1.5)
                        smooth_dd = smooth_delta * smooth_delta
                        if smooth_dd > 600.0:
                            smooth_like = 0.0
                        else:
                            smooth_like = np.exp(-0.5 * smooth_dd)
                        if smooth_like < 1e-300:
                            smooth_like = 1e-300
                        combined = (
                            1.0 - gr_smooth_weight
                        ) * raw_like + gr_smooth_weight * smooth_like
                    else:
                        combined = raw_like
                    if combined < 1e-300:
                        combined = 1e-300
                    avg_lik += weights[j] * combined
                    weights[j] *= combined
            if avg_lik < 1e-300:
                avg_lik = 1e-300
            log_lik += np.log(avg_lik)

            weight_sum = 0.0
            for j in range(n_particles):
                weight_sum += weights[j]
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles

            velocity_lik_sum = 0.0
            for j in range(n_particles):
                dv = (vel[j] - expected_velocity) / max(zsig * 2.0, 0.005)
                dv2 = dv * dv
                if dv2 > 600.0:
                    velocity_like = 0.0
                else:
                    velocity_like = np.exp(-0.5 * dv2)
                if velocity_like < 1e-300:
                    velocity_like = 1e-300
                velocity_lik_sum += weights[j] * velocity_like
                weights[j] *= velocity_like
            if velocity_lik_sum < 1e-300:
                velocity_lik_sum = 1e-300
            log_lik += np.log(velocity_lik_sum)
            weight_sum = 0.0
            for j in range(n_particles):
                weight_sum += weights[j]
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles

            neff_denom = 0.0
            for j in range(n_particles):
                neff_denom += weights[j] * weights[j]
            neff = 1.0 / neff_denom
            if neff < resample_threshold * n_particles:
                cum = np.empty(n_particles)
                c = 0.0
                for j in range(n_particles):
                    c += weights[j]
                    cum[j] = c
                u0 = np.random.uniform(0.0, 1.0 / n_particles)
                new_pos = np.empty(n_particles)
                new_rate = np.empty(n_particles)
                cursor = 0
                for j in range(n_particles):
                    u = u0 + j / n_particles
                    while cursor < n_particles - 1 and cum[cursor] < u:
                        cursor += 1
                    new_pos[j] = pos[cursor] + rough_pos * np.random.randn()
                    new_rate[j] = vel[cursor] + rough_vel * np.random.randn()
                for j in range(n_particles):
                    pos[j] = new_pos[j]
                    vel[j] = new_rate[j]
                    weights[j] = 1.0 / n_particles

            est = 0.0
            for j in range(n_particles):
                est += weights[j] * pos[j]
            preds[seed_offset, i] = est
            var = 0.0
            for j in range(n_particles):
                diff = pos[j] - est
                var += weights[j] * diff * diff
            stds[seed_offset, i] = var**0.5
            prev_md = md_v[i]
            prev_z = z_v[i]
        liks[seed_offset] = log_lik
    return preds, stds, liks


def _typewell_grids(
    tw: pd.DataFrame,
    step: float,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    tw = tw.sort_values("TVT")
    tvt = _as_float_array(tw, "TVT")
    gr = pd.to_numeric(tw["GR"], errors="coerce")
    gr = gr.interpolate(limit_direction="both").fillna(float(gr.mean())).to_numpy(np.float64)
    mask = np.isfinite(tvt) & np.isfinite(gr)
    tvt = tvt[mask]
    gr = gr[mask]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    if len(tvt) < 2:
        raise ValueError("typewell grid needs at least two finite rows")
    smooth_gr = pd.Series(gr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float64)
    grid, vmin, actual_step = _grid_interp(
        tvt.astype(np.float64),
        gr.astype(np.float64),
        float(step),
    )
    smooth_grid, _, _ = _grid_interp(
        tvt.astype(np.float64),
        smooth_gr.astype(np.float64),
        float(step),
    )
    return (
        grid.astype(np.float64),
        smooth_grid.astype(np.float64),
        float(vmin),
        float(actual_step),
        tvt,
        gr,
    )


def _grid_exp072(
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    step: float,
) -> tuple[np.ndarray, float, float]:
    tmin = float(np.nanmin(tw_tvt))
    tmax = float(np.nanmax(tw_tvt))
    tvt_grid = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig_exp072(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> float:
    known = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(known) < 20:
        return float(config.get("gr_sigma_default", 30.0))
    return float(
        np.clip(
            np.std(known["GR"].values - np.interp(known["TVT_input"].values, tw_tvt, tw_gr)),
            float(config.get("gr_sigma_min", 10.0)),
            float(config.get("gr_sigma_max", 60.0)),
        )
    )


def _fit_strict_pf_z_params(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> StrictPfZFit:
    known = hw[hw["TVT_input"].notna()].copy()
    eval_rows = int(hw["TVT_input"].isna().sum())
    if len(known) < 2:
        return StrictPfZFit(-1.0, 0.0, 0.1, 0.0, 30.0, int(len(known)), eval_rows)

    md = _as_float_array(known, "MD")
    z = _as_float_array(known, "Z")
    tvt = _as_float_array(known, "TVT_input")
    dz = np.diff(z)
    dvt = np.diff(tvt)
    dm = np.diff(md)
    valid = (dm > 0) & np.isfinite(dz) & np.isfinite(dvt)
    if int(valid.sum()) >= 10:
        z_rate = dz[valid] / dm[valid]
        tvt_rate = dvt[valid] / dm[valid]
        design = np.column_stack([z_rate, np.ones_like(z_rate)])
        coeff, _, _, _ = np.linalg.lstsq(design, tvt_rate, rcond=None)
        fitted = design @ coeff
        beta = float(coeff[0])
        intercept = float(coeff[1])
        zsig = max(float(np.std(tvt_rate - fitted)), 0.001)
    else:
        beta, intercept, zsig = -1.0, 0.0, 0.1

    tail = known.tail(20)
    tail_md = _as_float_array(tail, "MD")
    tail_tvt = _as_float_array(tail, "TVT_input")
    tail_dm = np.diff(tail_md)
    tail_dvt = np.diff(tail_tvt)
    tail_valid = (tail_dm > 0) & np.isfinite(tail_dvt)
    initial_velocity = (
        float(np.median(tail_dvt[tail_valid] / tail_dm[tail_valid]))
        if int(tail_valid.sum()) >= 3
        else 0.0
    )

    gr_known = known[known["GR"].notna()]
    if len(gr_known) >= 20:
        expected = np.interp(
            gr_known["TVT_input"].to_numpy(np.float64),
            tw_tvt.astype(np.float64),
            tw_gr.astype(np.float64),
        )
        observed = gr_known["GR"].to_numpy(np.float64)
        gr_sigma = float(
            np.clip(
                np.nanstd(observed - expected),
                float(config.get("gr_sigma_min", 10.0)),
                float(config.get("gr_sigma_max", 60.0)),
            )
        )
    else:
        gr_sigma = float(config.get("gr_sigma_default", 30.0))
    return StrictPfZFit(
        beta=beta,
        intercept=intercept,
        zsig=zsig,
        initial_velocity=initial_velocity,
        gr_sigma=gr_sigma,
        known_rows=int(len(known)),
        eval_rows=eval_rows,
    )


def _seed_vector_for_well(
    well: str,
    n_seeds: int,
    seed_root: int,
) -> np.ndarray:
    return np.array(
        [
            stable_seed(OUTPUT_PREFIX, "strict_pf_z", seed_root, well, seed_index)
            for seed_index in range(n_seeds)
        ],
        dtype=np.int64,
    )


def _trajectory_seed_vector_for_well(
    well: str,
    variant: str,
    n_seeds: int,
    seed_root: int,
) -> np.ndarray:
    return np.array(
        [
            stable_seed(OUTPUT_PREFIX, "trajectory_pf_z", seed_root, well, variant, seed_index)
            for seed_index in range(n_seeds)
        ],
        dtype=np.int64,
    )


def parse_trajectory_variants(config: dict[str, Any]) -> list[TrajectoryVariant]:
    variants: list[TrajectoryVariant] = []
    for raw in config.get("transition_variants", []):
        variants.append(
            TrajectoryVariant(
                name=str(raw["name"]),
                transition_strength=float(raw.get("transition_strength", 0.08)),
                z_accel_gain=float(raw.get("z_accel_gain", 0.0)),
                prefix_slope_strength=float(raw.get("prefix_slope_strength", 0.0)),
                velocity_noise_gain=float(raw.get("velocity_noise_gain", 0.0)),
                position_noise_gain=float(raw.get("position_noise_gain", 0.0)),
                likelihood_sigma_gain=float(raw.get("likelihood_sigma_gain", 0.0)),
                curve_clip=float(raw.get("curve_clip", 3.0)),
            )
        )
    if not variants:
        raise ValueError("model.trajectory_pf_z.transition_variants must define variants")
    return variants


def _tail_slope(values: np.ndarray, md: np.ndarray, n: int) -> float:
    if len(values) < 2:
        return 0.0
    values = values[-n:]
    md = md[-n:]
    finite = np.isfinite(values) & np.isfinite(md)
    if int(finite.sum()) < 2:
        return 0.0
    x = md[finite]
    y = values[finite]
    centered = x - float(x.mean())
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(centered, y - float(y.mean())) / denom)


def _trajectory_arrays(
    eval_frame: pd.DataFrame,
    known: pd.DataFrame,
    *,
    tail_rows: int,
) -> dict[str, np.ndarray | float]:
    md_v = _as_float_array(eval_frame, "MD")
    z_v = _as_float_array(eval_frame, "Z")
    known_md = _as_float_array(known, "MD")
    known_z = _as_float_array(known, "Z")
    known_tvt = _as_float_array(known, "TVT_input")
    last_md = float(known_md[-1])
    last_z = float(known_z[-1])
    all_md = np.concatenate([np.array([last_md], dtype=np.float64), md_v])
    all_z = np.concatenate([np.array([last_z], dtype=np.float64), z_v])
    dm = np.diff(all_md)
    dm = np.where(dm < 1.0, 1.0, dm)
    dzd = np.diff(all_z) / dm
    dzd_prev = np.concatenate([np.array([dzd[0]], dtype=np.float64), dzd[:-1]])
    d2z = (dzd - dzd_prev) / dm
    return {
        "dzd_v": np.nan_to_num(dzd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64),
        "d2z_v": np.nan_to_num(d2z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64),
        "prefix_tvt_slope": _tail_slope(known_tvt, known_md, tail_rows),
        "prefix_z_slope": _tail_slope(known_z, known_md, tail_rows),
    }


def run_strict_pf_z_for_well(
    well: str,
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    row_indices: np.ndarray,
    config: dict[str, Any],
    seed_root: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strict_config = config.get("strict_pf_z", config)
    trajectory_config = config.get("trajectory_pf_z", {})
    eval_frame = hw.iloc[row_indices].copy()
    known = hw[hw["TVT_input"].notna()].copy()
    if known.empty:
        raise ValueError(f"No finite TVT_input prefix rows for well {well}")
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy(np.float32)
    tw_gr = tw_sorted["GR"].to_numpy(np.float32)
    step = float(strict_config.get("gr_grid_step", 0.2))
    grid, vmin, actual_step = _grid_exp072(tw_tvt, tw_gr, step)
    tw_smooth_gr = (
        pd.Series(tw_gr)
        .rolling(int(strict_config.get("gr_smooth_window", 5)), center=True, min_periods=1)
        .mean()
        .values.astype(np.float32)
    )
    smooth_grid, _, _ = _grid_exp072(tw_tvt, tw_smooth_gr, step)
    fit = _fit_strict_pf_z_params(hw, tw_tvt, tw_gr, strict_config)
    gr_sigma = _gr_sig_exp072(hw, tw_tvt, tw_gr, strict_config)
    last = known.iloc[-1]
    n_seeds = int(strict_config.get("n_seeds", 64))
    scales = [float(value) for value in strict_config.get("scales", [3.0, 5.0, 8.0, 12.0])]
    parity_seed = stable_seed("pf_z", well)
    seed_vector = _seed_vector_for_well(well, n_seeds, seed_root)
    all_seeds = np.concatenate([np.array([parity_seed], dtype=np.int64), seed_vector])
    gr_sm = (
        hw["GR"]
        .rolling(
            int(strict_config.get("gr_smooth_window", 5)),
            center=True,
            min_periods=1,
        )
        .mean()
    )
    md_v = _as_float_array(eval_frame, "MD")
    z_v = _as_float_array(eval_frame, "Z")
    gr_v = _as_float_array(eval_frame, "GR")
    gr_sm_v = gr_sm.loc[eval_frame.index].values.astype(np.float64)
    preds = np.empty((len(all_seeds), len(eval_frame)), dtype=np.float64)
    stds = np.empty((len(all_seeds), len(eval_frame)), dtype=np.float64)
    liks = np.empty(len(all_seeds), dtype=np.float64)
    for seed_idx, seed in enumerate(all_seeds):
        seed_pred, seed_std, seed_lik = _strict_pf_z_seeded_exp072(
            int(seed),
            md_v,
            z_v,
            gr_v,
            gr_sm_v,
            grid,
            smooth_grid,
            vmin,
            actual_step,
            float(gr_sigma),
            float(last["TVT_input"]),
            float(fit.initial_velocity),
            float(fit.beta),
            float(fit.intercept),
            float(fit.zsig),
            int(strict_config.get("n_particles", 500)),
            float(strict_config.get("momentum", 0.998)),
            float(strict_config.get("velocity_noise", 0.002)),
            float(strict_config.get("position_noise", 0.005)),
            float(strict_config.get("gr_smooth_weight", 0.3)),
            float(strict_config.get("rough_position", 0.2)),
            float(strict_config.get("rough_velocity", 0.003)),
            float(strict_config.get("resample_threshold", 0.5)),
        )
        preds[seed_idx] = seed_pred
        stds[seed_idx] = seed_std
        liks[seed_idx] = seed_lik
    out = pd.DataFrame({"id": [f"{well}_{int(idx)}" for idx in row_indices]})
    out["strict_pf_z_parity_seed"] = preds[0].astype(np.float32)
    ms_preds = preds[1:]
    ms_stds = stds[1:]
    ms_liks = liks[1:]
    out["pf_z_ms_mean"] = ms_preds.mean(axis=0).astype(np.float32)
    out["pf_z_ms_std"] = ms_preds.std(axis=0).astype(np.float32)
    out["pf_z_ms_particle_std_mean"] = ms_stds.mean(axis=0).astype(np.float32)
    best_idx = int(np.argmax(ms_liks)) if len(ms_liks) else 0
    out["pf_z_ms_best_lik_seed"] = ms_preds[best_idx].astype(np.float32)
    centered_lik = ms_liks - float(ms_liks.max())
    for scale in scales:
        weights = np.exp(centered_lik / float(scale))
        weights /= weights.sum()
        out[f"pf_z_ms_scale_{scale:g}"] = (
            (weights[:, None] * ms_preds).sum(axis=0).astype(np.float32)
        )

    trajectory_variants = parse_trajectory_variants(trajectory_config)
    trajectory_scales = [
        float(value)
        for value in trajectory_config.get("scales", strict_config.get("scales", [3.0]))
    ]
    trajectory_features = _trajectory_arrays(
        eval_frame,
        known,
        tail_rows=int(trajectory_config.get("prefix_tail_rows", 50)),
    )
    dzd_v = trajectory_features["dzd_v"]
    d2z_v = trajectory_features["d2z_v"]
    prefix_tvt_slope = float(trajectory_features["prefix_tvt_slope"])
    prefix_z_slope = float(trajectory_features["prefix_z_slope"])
    out["md"] = md_v.astype(np.float32)
    out["z"] = z_v.astype(np.float32)
    out["dzdmd"] = np.asarray(dzd_v, dtype=np.float32)
    out["d2zdmd2"] = np.asarray(d2z_v, dtype=np.float32)
    out["prefix_tvt_slope"] = np.float32(prefix_tvt_slope)
    out["prefix_z_slope"] = np.float32(prefix_z_slope)

    trajectory_quality_rows: list[dict[str, Any]] = []
    trajectory_n_seeds = int(trajectory_config.get("n_seeds", n_seeds))
    trajectory_n_particles = int(
        trajectory_config.get("n_particles", strict_config.get("n_particles", 500))
    )
    for variant in trajectory_variants:
        variant_seed_vector = _trajectory_seed_vector_for_well(
            well,
            variant.name,
            trajectory_n_seeds,
            seed_root,
        )
        variant_preds = np.empty((len(variant_seed_vector), len(eval_frame)), dtype=np.float64)
        variant_stds = np.empty_like(variant_preds)
        variant_liks = np.empty(len(variant_seed_vector), dtype=np.float64)
        variant_diags = np.empty((len(variant_seed_vector), 5), dtype=np.float64)
        for seed_idx, seed in enumerate(variant_seed_vector):
            seed_pred, seed_std, seed_lik, seed_diag = _trajectory_pf_z_seeded(
                int(seed),
                md_v,
                z_v,
                gr_v,
                gr_sm_v,
                np.asarray(dzd_v, dtype=np.float64),
                np.asarray(d2z_v, dtype=np.float64),
                grid,
                smooth_grid,
                vmin,
                actual_step,
                float(gr_sigma),
                float(last["TVT_input"]),
                float(fit.initial_velocity),
                float(fit.beta),
                float(fit.intercept),
                float(fit.zsig),
                prefix_tvt_slope,
                trajectory_n_particles,
                float(trajectory_config.get("momentum", strict_config.get("momentum", 0.993))),
                float(
                    trajectory_config.get(
                        "velocity_noise",
                        strict_config.get("velocity_noise", 0.005),
                    )
                ),
                float(
                    trajectory_config.get(
                        "position_noise",
                        strict_config.get("position_noise", 0.01),
                    )
                ),
                float(
                    trajectory_config.get(
                        "gr_smooth_weight",
                        strict_config.get("gr_smooth_weight", 0.3),
                    )
                ),
                float(
                    trajectory_config.get(
                        "rough_position",
                        strict_config.get("rough_position", 0.2),
                    )
                ),
                float(
                    trajectory_config.get(
                        "rough_velocity",
                        strict_config.get("rough_velocity", 0.003),
                    )
                ),
                float(
                    trajectory_config.get(
                        "resample_threshold",
                        strict_config.get("resample_threshold", 0.5),
                    )
                ),
                variant.transition_strength,
                variant.z_accel_gain,
                variant.prefix_slope_strength,
                variant.velocity_noise_gain,
                variant.position_noise_gain,
                variant.likelihood_sigma_gain,
                variant.curve_clip,
                float(trajectory_config.get("collapse_std_threshold", 0.25)),
            )
            variant_preds[seed_idx] = seed_pred
            variant_stds[seed_idx] = seed_std
            variant_liks[seed_idx] = seed_lik
            variant_diags[seed_idx] = seed_diag
        centered_lik = variant_liks - float(variant_liks.max())
        mean_name = f"traj_pf_{variant.name}_mean"
        out[mean_name] = variant_preds.mean(axis=0).astype(np.float32)
        out[f"traj_pf_{variant.name}_seed_std"] = variant_preds.std(axis=0).astype(np.float32)
        out[f"traj_pf_{variant.name}_particle_std_mean"] = variant_stds.mean(axis=0).astype(
            np.float32
        )
        best_idx = int(np.argmax(variant_liks)) if len(variant_liks) else 0
        out[f"traj_pf_{variant.name}_best_lik_seed"] = variant_preds[best_idx].astype(np.float32)
        for scale in trajectory_scales:
            weights = np.exp(centered_lik / float(scale))
            weights /= weights.sum()
            out[f"traj_pf_{variant.name}_scale_{scale:g}"] = (
                (weights[:, None] * variant_preds).sum(axis=0).astype(np.float32)
            )
        trajectory_quality_rows.append(
            {
                "variant": variant.name,
                "n_seeds": int(len(variant_seed_vector)),
                "n_particles": int(trajectory_n_particles),
                "seed_min": int(variant_seed_vector.min()) if len(variant_seed_vector) else None,
                "seed_max": int(variant_seed_vector.max()) if len(variant_seed_vector) else None,
                "lik_best_per_row": float(variant_liks.max() / max(len(row_indices), 1)),
                "lik_std": float(variant_liks.std()),
                "mean_neff_frac": float(variant_diags[:, 0].mean()),
                "min_neff_frac": float(variant_diags[:, 1].min()),
                "mean_resample_count": float(variant_diags[:, 2].mean()),
                "mean_collapse_rate": float(variant_diags[:, 3].mean()),
                "mean_particle_std": float(variant_diags[:, 4].mean()),
            }
        )
    quality = {
        "well": well,
        "known_rows": fit.known_rows,
        "eval_rows": fit.eval_rows,
        "beta": fit.beta,
        "intercept": fit.intercept,
        "zsig": fit.zsig,
        "initial_velocity": fit.initial_velocity,
        "gr_sigma": gr_sigma,
        "parity_seed": int(parity_seed),
        "seed_min": int(seed_vector.min()) if len(seed_vector) else None,
        "seed_max": int(seed_vector.max()) if len(seed_vector) else None,
        "lik_best_per_row": float(ms_liks.max() / max(len(row_indices), 1)),
        "lik_std": float(ms_liks.std()),
        "parity_lik_per_row": float(liks[0] / max(len(row_indices), 1)),
        "prefix_tvt_slope": prefix_tvt_slope,
        "prefix_z_slope": prefix_z_slope,
        "mean_abs_dzdmd": float(np.mean(np.abs(np.asarray(dzd_v, dtype=np.float64)))),
        "mean_abs_d2zdmd2": float(np.mean(np.abs(np.asarray(d2z_v, dtype=np.float64)))),
        "trajectory_variants": trajectory_quality_rows,
    }
    return out, quality


def warm_up_strict_pf_z_kernel() -> None:
    md = np.linspace(1.0, 5.0, 5, dtype=np.float64)
    z = np.zeros(5, dtype=np.float64)
    gr = np.full(5, 50.0, dtype=np.float64)
    grid = np.linspace(45.0, 55.0, 100, dtype=np.float64)
    _strict_pf_z_seeded_exp072(
        12345,
        md,
        z,
        gr,
        gr,
        grid,
        grid,
        45.0,
        0.1,
        20.0,
        50.0,
        0.0,
        -1.0,
        0.0,
        0.1,
        8,
        0.993,
        0.005,
        0.01,
        0.3,
        0.2,
        0.003,
        0.5,
    )


def run_strict_pf_z_task(
    well: str,
    group: pd.DataFrame,
    *,
    train_dir: Path,
    model_config: dict[str, Any],
    seed_root: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"missing raw train files for {well}")
    input_sha = {
        str(horizontal_path): sha256_path(horizontal_path),
        str(typewell_path): sha256_path(typewell_path),
    }
    hw = pd.read_csv(horizontal_path)
    tw = pd.read_csv(typewell_path)
    row_indices = group["row_idx"].to_numpy(np.int32)
    strict_frame, quality = run_strict_pf_z_for_well(
        str(well),
        hw,
        tw,
        row_indices=row_indices,
        config=model_config,
        seed_root=seed_root,
    )
    return strict_frame, quality, input_sha


def materialize_existing_candidates(
    frame: pd.DataFrame,
    specs: list[ExistingCandidate],
) -> pd.DataFrame:
    out = frame[["id", "well", "target", "last_known_tvt", "md_since"]].copy()
    out["row_idx"] = _row_indices_from_ids(out["id"])
    last_known = _as_float_array(out, "last_known_tvt")
    out["target_tvt"] = last_known + _as_float_array(out, "target")
    for spec in specs:
        values = _as_float_array(frame, spec.source_column)
        if spec.transform == "base_plus_delta":
            out[spec.name] = last_known + values
        elif spec.transform == "absolute":
            out[spec.name] = values
        else:
            raise ValueError(f"unknown transform: {spec.transform}")
    return out


def _metric_row(frame: pd.DataFrame, candidate: str, thresholds: list[float]) -> dict[str, Any]:
    error = _as_float_array(frame, candidate) - _as_float_array(frame, "target_tvt")
    abs_error = np.abs(error)
    row: dict[str, Any] = {
        "candidate": candidate,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "bias": float(np.mean(error)),
    }
    for threshold in thresholds:
        key = str(threshold).replace(".", "p")
        row[f"within_{key}ft"] = float(np.mean(abs_error <= threshold))
    return row


def _smoothness_by_well(frame: pd.DataFrame, candidate_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidate_names:
        for well, group in frame.sort_values(["well", "row_idx"]).groupby("well", sort=False):
            pred = _as_float_array(group, candidate)
            step = np.diff(pred)
            error = pred - _as_float_array(group, "target_tvt")
            rows.append(
                {
                    "well": str(well),
                    "candidate": candidate,
                    "rows": int(len(group)),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "mean_abs_step": float(np.mean(np.abs(step))) if len(step) else 0.0,
                    "p95_abs_step": float(np.quantile(np.abs(step), 0.95)) if len(step) else 0.0,
                    "p95_abs_acceleration": float(np.quantile(np.abs(np.diff(step)), 0.95))
                    if len(step) > 1
                    else 0.0,
                    "path_switch_count": int(np.sum(np.abs(step) > 25.0)) if len(step) else 0,
                }
            )
    return pd.DataFrame(rows)


def _bucket_metrics(
    frame: pd.DataFrame,
    candidate_names: list[str],
    thresholds: list[float],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bucket_defs = {
        "md_since": config.get("bucket_edges_md_since", [0, 50, 100, 250, 500, 1000]),
        "row_idx": config.get("bucket_edges_row_idx", [0, 100, 250, 500, 1000]),
    }
    if "dzdmd" in frame.columns:
        bucket_defs["abs_dzdmd"] = config.get("bucket_edges_abs_dzdmd", [0, 0.1, 0.25, 0.5, 1.0])
        frame = frame.copy()
        frame["abs_dzdmd"] = np.abs(_as_float_array(frame, "dzdmd")).astype(np.float32)
    if "d2zdmd2" in frame.columns:
        bucket_defs["abs_d2zdmd2"] = config.get(
            "bucket_edges_abs_d2zdmd2",
            [0, 0.001, 0.003, 0.01, 0.03],
        )
        frame = frame.copy()
        frame["abs_d2zdmd2"] = np.abs(_as_float_array(frame, "d2zdmd2")).astype(np.float32)
    for bucket_column, edges in bucket_defs.items():
        edges = [float(value) for value in edges]
        bins = [-np.inf, *edges[1:], np.inf]

        def edge_label(value: float) -> str:
            return f"{value:g}".replace("-", "m").replace(".", "p")

        labels = [
            f"{edge_label(edges[i])}_{edge_label(edges[i + 1])}" for i in range(len(edges) - 1)
        ]
        labels.append(f"{edge_label(edges[-1])}_plus")
        buckets = pd.cut(
            pd.to_numeric(frame[bucket_column], errors="coerce"),
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        for candidate in candidate_names:
            for bucket, group in frame.groupby(buckets, observed=False):
                if len(group) == 0:
                    continue
                metric = _metric_row(group, candidate, thresholds)
                metric["bucket_family"] = bucket_column
                metric["bucket"] = str(bucket)
                rows.append(metric)
    return pd.DataFrame(rows)


def _candidate_long(frame: pd.DataFrame, candidate_names: list[str]) -> pd.DataFrame:
    rows = []
    base_cols = ["id", "well", "row_idx", "md_since", "last_known_tvt", "target", "target_tvt"]
    for candidate in candidate_names:
        item = frame[base_cols].copy()
        item["candidate"] = candidate
        item["pred_tvt"] = frame[candidate].to_numpy(np.float32)
        item["abs_error"] = np.abs(
            item["pred_tvt"].to_numpy(np.float32) - item["target_tvt"].to_numpy(np.float32)
        )
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.time()
    config = config or load_config()
    paths = ExperimentPaths()
    audit_config = get_nested(config, "audit") or {}
    strict_config = get_nested(config, "model.strict_pf_z") or {}
    trajectory_config = get_nested(config, "model.trajectory_pf_z") or {}
    model_config = {"strict_pf_z": strict_config, "trajectory_pf_z": trajectory_config}
    thresholds = [float(value) for value in audit_config.get("thresholds_ft", [1, 2, 5, 10])]
    max_rows = audit_config.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None
    max_wells = audit_config.get("max_wells")
    max_wells = int(max_wells) if max_wells is not None else None

    cache_path = find_artifact(
        EXP072_TRAIN_FEATURES,
        get_nested(config, "data.exp072_train_feature_cache_local"),
    )
    header = _read_header(cache_path)
    scales = [float(value) for value in strict_config.get("scales", [3.0, 5.0, 8.0, 12.0])]
    trajectory_variants = parse_trajectory_variants(trajectory_config)
    trajectory_scales = [
        float(value)
        for value in trajectory_config.get("scales", strict_config.get("scales", [3.0]))
    ]
    existing_specs = build_existing_candidate_specs(header, scales)
    cache_frame, cache_metadata = read_exp072_cache(
        config,
        existing_specs,
        max_rows=max_rows,
        max_wells=max_wells,
    )
    candidate_wide = materialize_existing_candidates(cache_frame, existing_specs)
    row_lookup = candidate_wide[["id", "well", "row_idx"]].copy()

    train_dir = paths.train_data_dir
    seed_root = int(get_nested(config, "reproducibility.seed") or 42)
    num_workers = int(get_nested(config, "runtime.num_workers") or 1)
    tasks = [(str(well), group.copy()) for well, group in row_lookup.groupby("well", sort=True)]
    warm_up_strict_pf_z_kernel()
    if num_workers == 1:
        task_results = [
            run_strict_pf_z_task(
                well,
                group,
                train_dir=train_dir,
                model_config=model_config,
                seed_root=seed_root,
            )
            for well, group in tasks
        ]
    else:
        task_results = Parallel(n_jobs=num_workers, prefer="threads")(
            delayed(run_strict_pf_z_task)(
                well,
                group,
                train_dir=train_dir,
                model_config=model_config,
                seed_root=seed_root,
            )
            for well, group in tasks
        )
    strict_frames = [frame for frame, _, _ in task_results]
    raw_quality_rows = [quality for _, quality, _ in task_results]
    quality_rows = [
        {key: value for key, value in quality.items() if key != "trajectory_variants"}
        for quality in raw_quality_rows
    ]
    trajectory_quality_rows: list[dict[str, Any]] = []
    for quality in raw_quality_rows:
        for row in quality.get("trajectory_variants", []):
            trajectory_quality_rows.append(
                {
                    "well": quality["well"],
                    "known_rows": quality["known_rows"],
                    "eval_rows": quality["eval_rows"],
                    **row,
                }
            )
    input_sha: dict[str, str] = {}
    for _, _, sha_values in task_results:
        input_sha.update(sha_values)

    strict_all = pd.concat(strict_frames, ignore_index=True)
    candidate_wide = candidate_wide.merge(
        strict_all,
        on="id",
        how="left",
        validate="one_to_one",
    )
    if candidate_wide.isna().any().any():
        missing = candidate_wide.columns[candidate_wide.isna().any()].tolist()
        raise ValueError(f"candidate_wide contains missing values in columns: {missing}")
    candidate_wide["pf_z_ms_delta_vs_pf_z"] = (
        candidate_wide["pf_z_ms_mean"].to_numpy(np.float64)
        - candidate_wide["exp072_pf_z"].to_numpy(np.float64)
    ).astype(np.float32)
    candidate_wide["pf_z_ms_delta_vs_likpf_mean"] = (
        candidate_wide["pf_z_ms_mean"].to_numpy(np.float64)
        - candidate_wide["exp072_likpf_mean"].to_numpy(np.float64)
    ).astype(np.float32)
    parity_diff = candidate_wide[
        ["id", "well", "row_idx", "exp072_pf_z", "strict_pf_z_parity_seed"]
    ].copy()
    parity_diff["diff"] = (
        parity_diff["strict_pf_z_parity_seed"].to_numpy(np.float64)
        - parity_diff["exp072_pf_z"].to_numpy(np.float64)
    ).astype(np.float32)
    parity_diff["abs_diff"] = np.abs(parity_diff["diff"].to_numpy(np.float32))
    parity_summary = {
        "rows": int(len(parity_diff)),
        "wells": int(parity_diff["well"].nunique()),
        "max_abs_diff": float(parity_diff["abs_diff"].max()),
        "mean_abs_diff": float(parity_diff["abs_diff"].mean()),
        "p95_abs_diff": float(parity_diff["abs_diff"].quantile(0.95)),
        "rmse_diff": float(np.sqrt(np.mean(parity_diff["diff"].to_numpy(np.float64) ** 2))),
    }
    parity_pass = parity_summary["max_abs_diff"] <= float(
        model_config.get("parity_abs_max_tolerance", 1e-4)
    ) and parity_summary["rmse_diff"] <= float(model_config.get("parity_rmse_tolerance", 1e-5))
    if (
        bool(model_config.get("require_parity_for_full", True))
        and not parity_pass
        and max_wells is None
        and max_rows is None
    ):
        raise ValueError(f"strict pf_z parity failed on full run: {parity_summary}")
    strict_candidates = [
        "strict_pf_z_parity_seed",
        "pf_z_ms_mean",
        *[f"pf_z_ms_scale_{scale:g}" for scale in scales],
        "pf_z_ms_best_lik_seed",
    ]
    trajectory_candidates: list[str] = []
    for variant in trajectory_variants:
        trajectory_candidates.extend(
            [
                f"traj_pf_{variant.name}_mean",
                *[f"traj_pf_{variant.name}_scale_{scale:g}" for scale in trajectory_scales],
                f"traj_pf_{variant.name}_best_lik_seed",
            ]
        )
    existing_candidates = [spec.name for spec in existing_specs]
    candidate_names = [*existing_candidates, *strict_candidates, *trajectory_candidates]
    numeric_values = candidate_wide[
        ["target_tvt", "last_known_tvt", "target", "md_since", *candidate_names]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("candidate_wide contains non-finite numeric values")

    candidate_metrics = (
        pd.DataFrame([_metric_row(candidate_wide, name, thresholds) for name in candidate_names])
        .sort_values("rmse", kind="stable")
        .reset_index(drop=True)
    )
    by_well = _smoothness_by_well(candidate_wide, candidate_names)
    bucket_metrics = _bucket_metrics(candidate_wide, candidate_names, thresholds, audit_config)
    candidate_long = (
        _candidate_long(candidate_wide, candidate_names)
        if bool(audit_config.get("save_candidate_long", False))
        else None
    )
    quality = pd.DataFrame(quality_rows)
    trajectory_quality = pd.DataFrame(trajectory_quality_rows)

    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    quality_path = output_dir / f"{OUTPUT_PREFIX}_strict_pf_z_quality.csv"
    trajectory_quality_path = output_dir / f"{OUTPUT_PREFIX}_trajectory_pf_z_quality.csv"
    parity_path = output_dir / f"{OUTPUT_PREFIX}_parity_diff.csv.gz"
    wide_path = output_dir / f"{OUTPUT_PREFIX}_candidate_wide.csv.gz"
    long_path = output_dir / f"{OUTPUT_PREFIX}_candidate_long.csv.gz"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    quality.to_csv(quality_path, index=False)
    trajectory_quality.to_csv(trajectory_quality_path, index=False)
    parity_diff.to_csv(parity_path, index=False, compression="gzip")
    candidate_wide.to_csv(wide_path, index=False, compression="gzip")
    if candidate_long is not None:
        candidate_long.to_csv(long_path, index=False, compression="gzip")

    output_files = {
        "candidate_metrics": str(metrics_path),
        "bucket_metrics": str(bucket_path),
        "by_well": str(by_well_path),
        "strict_pf_z_quality": str(quality_path),
        "trajectory_pf_z_quality": str(trajectory_quality_path),
        "parity_diff": str(parity_path),
        "candidate_wide": str(wide_path),
    }
    if long_path.exists():
        output_files["candidate_long"] = str(long_path)
    output_sha: dict[str, Any] = {}
    for key, value in output_files.items():
        path = Path(value)
        output_sha[key] = {"sha256": sha256_path(path)}
        if path.suffix == ".gz":
            output_sha[key]["decompressed_sha256"] = sha256_path(path, decompressed=True)

    best = candidate_metrics.iloc[0].to_dict()
    strict_best = (
        candidate_metrics[candidate_metrics["candidate"].isin(strict_candidates)].iloc[0].to_dict()
    )
    strict_ms_best = (
        candidate_metrics[
            candidate_metrics["candidate"].isin(
                ["pf_z_ms_mean", *[f"pf_z_ms_scale_{scale:g}" for scale in scales]]
            )
        ]
        .iloc[0]
        .to_dict()
    )
    trajectory_best = (
        candidate_metrics[candidate_metrics["candidate"].isin(trajectory_candidates)]
        .iloc[0]
        .to_dict()
    )
    exp072_likpf_mean = (
        candidate_metrics[candidate_metrics["candidate"].eq("exp072_likpf_mean")].iloc[0].to_dict()
    )
    exp072_pf_z = (
        candidate_metrics[candidate_metrics["candidate"].eq("exp072_pf_z")].iloc[0].to_dict()
    )
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_not_run" if max_rows or max_wells else "completed_train_side_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_sec": float(time.time() - start),
        "rows": int(len(candidate_wide)),
        "wells": int(candidate_wide["well"].nunique()),
        "candidate_names": candidate_names,
        "strict_pf_z_candidates": strict_candidates,
        "trajectory_pf_z_candidates": trajectory_candidates,
        "existing_candidates": existing_candidates,
        "missing_exp072_scale_candidates": [
            f"exp072_likpf_scale_{scale:g}"
            for scale in scales
            if f"exp072_likpf_scale_{scale:g}" not in existing_candidates
        ],
        "parity_summary": parity_summary,
        "parity_pass": bool(parity_pass),
        "best_candidate": to_jsonable(best),
        "strict_pf_z_best_candidate": to_jsonable(strict_best),
        "strict_pf_z_multiseed_best_candidate": to_jsonable(strict_ms_best),
        "trajectory_pf_z_best_candidate": to_jsonable(trajectory_best),
        "exp072_likpf_mean": to_jsonable(exp072_likpf_mean),
        "exp072_pf_z": to_jsonable(exp072_pf_z),
        "rmse_delta_strict_ms_best_minus_exp072_likpf_mean": float(
            strict_ms_best["rmse"] - exp072_likpf_mean["rmse"]
        ),
        "rmse_delta_strict_ms_best_minus_exp072_pf_z": float(
            strict_ms_best["rmse"] - exp072_pf_z["rmse"]
        ),
        "rmse_delta_trajectory_best_minus_exp072_likpf_mean": float(
            trajectory_best["rmse"] - exp072_likpf_mean["rmse"]
        ),
        "rmse_delta_trajectory_best_minus_exp072_pf_z": float(
            trajectory_best["rmse"] - exp072_pf_z["rmse"]
        ),
        "exp072_cache": cache_metadata,
        "input_file_sha256": input_sha,
        "output_files": output_files,
        "output_sha256": output_sha,
        "config_subset": {
            "strict_pf_z": strict_config,
            "trajectory_pf_z": trajectory_config,
            "audit": audit_config,
            "reproducibility": get_nested(config, "reproducibility"),
        },
    }
    summary["output_files"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.max_rows is not None:
        config.setdefault("audit", {})["max_rows"] = int(args.max_rows)
    if args.max_wells is not None:
        config.setdefault("audit", {})["max_wells"] = int(args.max_wells)
    if args.n_seeds is not None:
        config.setdefault("model", {}).setdefault("strict_pf_z", {})["n_seeds"] = int(args.n_seeds)
        config.setdefault("model", {}).setdefault("trajectory_pf_z", {})["n_seeds"] = int(
            args.n_seeds
        )
    if args.n_particles is not None:
        config.setdefault("model", {}).setdefault("strict_pf_z", {})["n_particles"] = int(
            args.n_particles
        )
        config.setdefault("model", {}).setdefault("trajectory_pf_z", {})["n_particles"] = int(
            args.n_particles
        )
    summary = run_audit(config)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
