"""Run preregistered paired PF counterfactuals on exp410 sentinel wells."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


os.environ["EXP410_IMPORT_ONLY"] = "1"
import exp410_likpf_particle_resampling_basin_audit_compact_selfcontained_train as base  # noqa: E402


EXPERIMENT = "exp410_likpf_particle_resampling_basin_audit"
PREFIX = f"{EXPERIMENT}_counterfactual"
SENTINEL_FILENAME = "pf_counterfactual_sentinel_wells.csv"
SENTINEL_MANIFEST_FILENAME = "pf_counterfactual_sentinel_manifest.json"
EXPECTED_VARIANTS = (
    "baseline",
    "momentum_one",
    "zero_process_noise",
    "process_noise_x3",
    "init_spread_x3",
    "gr_sigma_x1p3",
    "gr_sigma_x3",
    "gr_emission_near_disabled",
    "resample_threshold_0p1",
    "resampling_disabled",
    "roughening_x10",
    "clamp_margin_x2",
)
EXPECTED_SENTINELS = 12
EXPECTED_SHARDS = 4
PF_INTERP1 = base._interp1


@dataclass(frozen=True)
class ParticleModeRun:
    seed_predictions: np.ndarray
    seed_particle_modes: np.ndarray
    log_likelihoods: np.ndarray
    ess_mean: np.ndarray
    resampled_seed_fraction: np.ndarray


@base.njit(cache=True, nogil=True)
def _exp072_baseline_with_particle_mode(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gr_grid: np.ndarray,
    vmin: float,
    step: float,
    gr_sigma: float,
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
    mode_bin_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact exp072 state/RNG update plus a read-only weighted mode readout."""

    n_rows = len(md_v)
    predictions = np.empty((n_seeds, n_rows), dtype=np.float64)
    particle_modes = np.empty((n_seeds, n_rows), dtype=np.float64)
    log_likelihoods = np.empty(n_seeds, dtype=np.float64)
    ess_accum = np.zeros(n_rows, dtype=np.float64)
    resampled_accum = np.zeros(n_rows, dtype=np.float64)
    tmax = vmin + len(gr_grid) * step
    histogram_min = vmin - 100.0 - mode_bin_width
    histogram_max = tmax + 100.0 + mode_bin_width
    n_mode_bins = (
        int(np.ceil((histogram_max - histogram_min) / mode_bin_width)) + 2
    )

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
        histogram0 = np.zeros(n_mode_bins, dtype=np.float64)
        histogram1 = np.zeros(n_mode_bins, dtype=np.float64)
        shifted_histogram_min = histogram_min - 0.5 * mode_bin_width
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
                expected_gr = PF_INTERP1(
                    gr_grid,
                    pos[particle_index] - z_v[row_index],
                    vmin,
                    step,
                )
                residual = (gr_v[row_index] - expected_gr) / gr_sigma
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

            # Read-only target-free particle-mode calculation after the exact
            # estimate is fixed. It adds no RNG calls and mutates no PF state.
            for bin_index in range(n_mode_bins):
                histogram0[bin_index] = 0.0
                histogram1[bin_index] = 0.0
            for particle_index in range(n_particles):
                tvt_particle = pos[particle_index] - z_v[row_index]
                bin0 = int((tvt_particle - histogram_min) / mode_bin_width)
                bin1 = int(
                    (tvt_particle - shifted_histogram_min) / mode_bin_width
                )
                if bin0 < 0:
                    bin0 = 0
                if bin0 >= n_mode_bins:
                    bin0 = n_mode_bins - 1
                if bin1 < 0:
                    bin1 = 0
                if bin1 >= n_mode_bins:
                    bin1 = n_mode_bins - 1
                histogram0[bin0] += weights[particle_index]
                histogram1[bin1] += weights[particle_index]
            best_histogram = 0
            best_bin = 0
            best_mass = histogram0[0]
            for bin_index in range(1, n_mode_bins):
                if histogram0[bin_index] > best_mass:
                    best_mass = histogram0[bin_index]
                    best_bin = bin_index
            for bin_index in range(n_mode_bins):
                if histogram1[bin_index] > best_mass:
                    best_mass = histogram1[bin_index]
                    best_bin = bin_index
                    best_histogram = 1
            mode_sum = 0.0
            mode_mass = 0.0
            for particle_index in range(n_particles):
                tvt_particle = pos[particle_index] - z_v[row_index]
                if best_histogram == 0:
                    bin_index = int(
                        (tvt_particle - histogram_min) / mode_bin_width
                    )
                else:
                    bin_index = int(
                        (tvt_particle - shifted_histogram_min)
                        / mode_bin_width
                    )
                if bin_index < 0:
                    bin_index = 0
                if bin_index >= n_mode_bins:
                    bin_index = n_mode_bins - 1
                if bin_index == best_bin:
                    mode_sum += weights[particle_index] * tvt_particle
                    mode_mass += weights[particle_index]
            particle_modes[seed_index, row_index] = (
                mode_sum / mode_mass if mode_mass > 0.0 else estimate
            )
            previous_md = md_v[row_index]
        log_likelihoods[seed_index] = log_likelihood

    denominator = float(n_seeds)
    return (
        predictions,
        particle_modes,
        log_likelihoods,
        ess_accum / denominator,
        resampled_accum / denominator,
    )


def run_baseline_with_particle_mode(
    prepared: Any,
    config: Mapping[str, Any],
) -> tuple[ParticleModeRun, dict[str, Any]]:
    runtime = base.get_nested(config, "model.runtime")
    split = str(base.get_nested(config, "model.replay.split_key"))
    seed_base = base.exp072_stable_seed("likpf", split, prepared.well)
    values = _exp072_baseline_with_particle_mode(
        prepared.md,
        prepared.z,
        prepared.gr,
        prepared.gr_grid,
        prepared.grid_min,
        prepared.grid_step,
        prepared.gr_sigma,
        prepared.last_surface,
        prepared.initial_surface_rate,
        int(runtime["particles"]),
        int(runtime["seed_count"]),
        seed_base,
        float(runtime["momentum"]),
        float(runtime["velocity_noise"]),
        float(runtime["position_noise"]),
        float(runtime["resample_pos_noise"]),
        float(runtime["resample_velocity_noise"]),
        float(runtime["resample_threshold"]),
        float(runtime["init_spread"]),
        float(
            base.get_nested(
                config,
                "audit.counterfactual.particle_mode_readout.histogram_bin_width_ft",
            )
        ),
    )
    run = ParticleModeRun(*values)
    replay = (
        run.seed_predictions.mean(axis=0, dtype=np.float64)
        .astype(np.float32)
        .astype(np.float64)
    )
    difference = replay - prepared.fixed_prediction
    diagnostics = {
        "parity_max_abs_ft": float(np.max(np.abs(difference))),
        "ess_mean": float(np.mean(run.ess_mean)),
        "resampling_rate": float(np.mean(run.resampled_seed_fraction)),
        "best_log_likelihood_per_row": float(
            np.max(run.log_likelihoods) / len(prepared.row_idx)
        ),
        "log_likelihood_std": float(np.std(run.log_likelihoods)),
    }
    return run, diagnostics


def sentinel_path(filename: str) -> Path:
    candidates = (
        base.PACKAGE_DIR / "assets" / filename,
        base.find_project_root()
        / "experiments"
        / EXPERIMENT
        / "assets"
        / filename,
        Path("/kaggle/working/assets") / filename,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"counterfactual asset missing: {filename}")


def prediction_sha(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype=np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()


def variant_config(
    config: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> tuple[dict[str, Any], float, float]:
    result = copy.deepcopy(dict(config))
    runtime = result["model"]["runtime"]
    gr_sigma_multiplier = float(changes.get("gr_sigma_multiplier", 1.0))
    clamp_margin_multiplier = float(
        changes.get("clamp_margin_multiplier", 1.0)
    )
    for key, value in changes.items():
        if key in {"gr_sigma_multiplier", "clamp_margin_multiplier"}:
            continue
        if key.endswith("_multiplier"):
            runtime_key = key.removesuffix("_multiplier")
            runtime[runtime_key] = float(runtime[runtime_key]) * float(value)
        else:
            runtime[key] = float(value)
    return result, gr_sigma_multiplier, clamp_margin_multiplier


def extend_clamp_input(
    prepared: Any,
    multiplier: float,
) -> Any:
    if multiplier == 1.0:
        return prepared
    if multiplier < 1.0:
        raise RuntimeError("counterfactual clamp multiplier must be >= 1")
    extra_ft = 100.0 * (multiplier - 1.0)
    extra_bins = int(round(extra_ft / float(prepared.grid_step)))
    if extra_bins <= 0:
        return prepared
    left = np.full(extra_bins, prepared.gr_grid[0], dtype=np.float64)
    right = np.full(extra_bins, prepared.gr_grid[-1], dtype=np.float64)
    gr_grid = np.concatenate((left, prepared.gr_grid, right))
    grid_min = float(prepared.grid_min) - extra_bins * float(
        prepared.grid_step
    )
    return replace(
        prepared,
        gr_grid=gr_grid,
        grid_min=grid_min,
        clamp_min_tvt=float(prepared.clamp_min_tvt) - extra_ft,
        clamp_max_tvt=float(prepared.clamp_max_tvt) + extra_ft,
    )


def metric_row(
    *,
    well: str,
    scope: str,
    readout: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    prediction_content_sha256: str,
    target_free: bool,
    oracle: bool,
    offline_suffix_readout: bool,
    pf_variant: str,
) -> dict[str, Any]:
    error = prediction - truth
    squared_error = error * error
    return {
        "well": well,
        "scope": scope,
        "readout": readout,
        "pf_variant": pf_variant,
        "target_free": target_free,
        "oracle": oracle,
        "offline_suffix_readout": offline_suffix_readout,
        "rows": int(len(error)),
        "sse": float(squared_error.sum()),
        "rmse_ft": float(np.sqrt(np.mean(squared_error))),
        "mae_ft": float(np.mean(np.abs(error))),
        "bias_ft": float(np.mean(error)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
        "max_abs_error_ft": float(np.max(np.abs(error))),
        "prediction_content_sha256": prediction_content_sha256,
    }


def add_baseline_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[
        metrics["readout"] == "baseline_arithmetic_seed_mean",
        ["well", "scope", "sse", "rmse_ft"],
    ].rename(
        columns={
            "sse": "baseline_sse",
            "rmse_ft": "baseline_rmse_ft",
        }
    )
    result = metrics.merge(
        baseline,
        on=["well", "scope"],
        how="left",
        validate="many_to_one",
    )
    result["sse_delta_vs_baseline"] = result["sse"] - result["baseline_sse"]
    result["rmse_delta_vs_baseline_ft"] = (
        result["rmse_ft"] - result["baseline_rmse_ft"]
    )
    result["sse_relative_vs_baseline"] = result["sse"] / np.maximum(
        result["baseline_sse"], 1.0e-12
    )
    return result


def episode_indices(
    row_idx: np.ndarray,
    episodes: pd.DataFrame,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    union = np.zeros(len(row_idx), dtype=bool)
    parts: list[tuple[str, np.ndarray]] = []
    for episode in episodes.itertuples(index=False):
        index = np.flatnonzero(
            (row_idx >= int(episode.start_row_idx))
            & (row_idx < int(episode.end_row_idx_exclusive))
        )
        if len(index) != int(episode.rows):
            raise RuntimeError(
                f"{episode.episode_id}: counterfactual episode coverage changed"
            )
        union[index] = True
        parts.append((str(episode.episode_id), index))
    return np.flatnonzero(union), parts


def aggregate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    result = (
        metrics.groupby(
            [
                "scope",
                "readout",
                "pf_variant",
                "target_free",
                "oracle",
                "offline_suffix_readout",
            ],
            observed=False,
            sort=True,
        )
        .agg(
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            sse=("sse", "sum"),
            baseline_sse=("baseline_sse", "sum"),
        )
        .reset_index()
    )
    result["rmse_ft"] = np.sqrt(result["sse"] / result["rows"])
    result["baseline_rmse_ft"] = np.sqrt(
        result["baseline_sse"] / result["rows"]
    )
    result["sse_delta_vs_baseline"] = result["sse"] - result["baseline_sse"]
    result["rmse_delta_vs_baseline_ft"] = (
        result["rmse_ft"] - result["baseline_rmse_ft"]
    )
    result["sse_relative_vs_baseline"] = result["sse"] / np.maximum(
        result["baseline_sse"], 1.0e-12
    )
    return result.sort_values(
        ["scope", "rmse_ft", "readout"], kind="stable"
    )


def main() -> dict[str, Any]:
    started = time.perf_counter()
    config = base.load_config()
    base.set_num_threads(int(base.get_nested(config, "execution.numba_num_threads")))
    shard_index = int(os.environ.get("EXP410_COUNTERFACTUAL_SHARD_INDEX", "0"))
    if not 0 <= shard_index < EXPECTED_SHARDS:
        raise RuntimeError(f"invalid counterfactual shard: {shard_index}")
    configured_variants = base.get_nested(config, "audit.counterfactual.variants")
    if tuple(configured_variants) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"counterfactual variants changed: {tuple(configured_variants)}"
        )

    sentinel_asset_path = sentinel_path(SENTINEL_FILENAME)
    sentinel_manifest_path = sentinel_path(SENTINEL_MANIFEST_FILENAME)
    sentinels = pd.read_csv(sentinel_asset_path, dtype={"well": str})
    if (
        len(sentinels) != EXPECTED_SENTINELS
        or sentinels["well"].duplicated().any()
        or set(sentinels["counterfactual_shard_index"].astype(int))
        != set(range(EXPECTED_SHARDS))
    ):
        raise RuntimeError("counterfactual sentinel contract changed")
    selected = sentinels.loc[
        sentinels["counterfactual_shard_index"].astype(int) == shard_index
    ].sort_values("selection_order", kind="stable")
    counterfactual_preflight = (
        os.environ.get("EXP410_COUNTERFACTUAL_PREFLIGHT", "0") == "1"
    )
    if counterfactual_preflight:
        if shard_index != 0:
            raise RuntimeError("counterfactual preflight must use shard 0")
        selected = selected.head(1).copy()
    if selected.empty:
        raise RuntimeError(f"counterfactual shard {shard_index} is empty")

    target_wells, fixed_episodes, fixed_asset_meta = base.load_fixed_assets(config)
    all_target_set = set(target_wells["well"].astype(str))
    fixed_control, source_meta = base.load_fixed_prediction_control(
        config, all_target_set
    )
    train_dir = base.resolve_train_dir(config)
    selected_set = set(selected["well"])
    selected_episodes = fixed_episodes.loc[
        fixed_episodes["well"].isin(selected_set)
    ].copy()

    print(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "stage": (
                    "counterfactual_preflight"
                    if counterfactual_preflight
                    else "counterfactual"
                ),
                "shard_index": shard_index,
                "selected_wells": len(selected),
                "selected_suffix_rows": int(selected["suffix_rows"].sum()),
                "variants": list(EXPECTED_VARIANTS),
                "pf_well_runs": int(len(selected) * len(EXPECTED_VARIANTS)),
                "particles": int(base.get_nested(config, "model.runtime.particles")),
                "seeds": int(base.get_nested(config, "model.runtime.seed_count")),
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

    metric_rows: list[dict[str, Any]] = []
    episode_metric_rows: list[dict[str, Any]] = []
    row_prediction_parts: list[pd.DataFrame] = []
    run_manifest_rows: list[dict[str, Any]] = []
    parity_max = 0.0

    for well_number, selected_row in enumerate(
        selected.itertuples(index=False), start=1
    ):
        well = str(selected_row.well)
        well_episodes = selected_episodes.loc[
            selected_episodes["well"] == well
        ].sort_values("start_row_idx", kind="stable")
        cache_rows = fixed_control.loc[fixed_control["well"] == well].copy()
        prepared = base.prepare_well(
            well=well,
            cache_rows=cache_rows,
            well_episodes=well_episodes,
            train_dir=train_dir,
            config=config,
        )
        episode_union_index, episode_parts = episode_indices(
            prepared.row_idx, well_episodes
        )
        zero_diagnostic = np.zeros_like(prepared.diagnostic_mask)
        replay_input = replace(prepared, diagnostic_mask=zero_diagnostic)
        all_index = np.arange(len(prepared.row_idx), dtype=np.int64)
        predictions: dict[str, np.ndarray] = {}
        readout_meta: dict[str, dict[str, Any]] = {}

        for variant_name, changes in configured_variants.items():
            run_started = time.perf_counter()
            (
                active_config,
                gr_sigma_multiplier,
                clamp_margin_multiplier,
            ) = variant_config(config, changes)
            active_input = replace(
                replay_input,
                gr_sigma=float(prepared.gr_sigma) * gr_sigma_multiplier,
            )
            active_input = extend_clamp_input(
                active_input, clamp_margin_multiplier
            )
            if variant_name == "baseline":
                run, diagnostics = run_baseline_with_particle_mode(
                    active_input, active_config
                )
            else:
                run, diagnostics = base.run_pf_audit(
                    active_input, active_config
                )
            arithmetic = (
                run.seed_predictions.mean(axis=0, dtype=np.float64)
                .astype(np.float32)
                .astype(np.float64)
            )
            best_loglik_index = int(np.argmax(run.log_likelihoods))
            shifted_loglik = (
                run.log_likelihoods - np.max(run.log_likelihoods)
            )
            likelihood_weights = np.exp(np.maximum(shifted_loglik, -745.0))
            likelihood_weights /= likelihood_weights.sum()
            likelihood_effective_seed_count = float(
                1.0 / np.sum(likelihood_weights * likelihood_weights)
            )
            readout_name = f"{variant_name}_arithmetic_seed_mean"
            predictions[readout_name] = arithmetic
            readout_meta[readout_name] = {
                "target_free": True,
                "oracle": False,
                "offline_suffix_readout": False,
                "pf_variant": variant_name,
            }
            if variant_name == "baseline":
                parity = float(
                    np.max(np.abs(arithmetic - prepared.fixed_prediction))
                )
                parity_max = max(parity_max, parity)
                if parity != 0.0:
                    raise RuntimeError(f"{well}: baseline parity changed: {parity}")
                seed_median = (
                    np.median(run.seed_predictions, axis=0)
                    .astype(np.float32)
                    .astype(np.float64)
                )
                best_loglik = (
                    run.seed_predictions[best_loglik_index]
                    .astype(np.float32)
                    .astype(np.float64)
                )
                likelihood_weighted = (
                    np.average(
                        run.seed_predictions,
                        axis=0,
                        weights=likelihood_weights,
                    )
                    .astype(np.float32)
                    .astype(np.float64)
                )
                particle_mode_seed_mean = (
                    run.seed_particle_modes.mean(
                        axis=0, dtype=np.float64
                    )
                    .astype(np.float32)
                    .astype(np.float64)
                )
                particle_mode_seed_median = (
                    np.median(run.seed_particle_modes, axis=0)
                    .astype(np.float32)
                    .astype(np.float64)
                )
                oracle_index = np.argmin(
                    np.abs(run.seed_predictions - prepared.truth[None, :]),
                    axis=0,
                )
                oracle = (
                    run.seed_predictions[
                        oracle_index,
                        np.arange(len(prepared.truth), dtype=np.int64),
                    ]
                    .astype(np.float32)
                    .astype(np.float64)
                )
                baseline_readouts = {
                    "baseline_seed_median": (
                        seed_median,
                        True,
                        False,
                        False,
                    ),
                    "baseline_best_loglik_seed": (
                        best_loglik,
                        True,
                        False,
                        True,
                    ),
                    "baseline_loglik_weighted_seed_mean": (
                        likelihood_weighted,
                        True,
                        False,
                        True,
                    ),
                    "baseline_oracle_best_seed_per_row": (
                        oracle,
                        False,
                        True,
                        False,
                    ),
                    "baseline_particle_mode_seed_mean": (
                        particle_mode_seed_mean,
                        True,
                        False,
                        False,
                    ),
                    "baseline_particle_mode_seed_median": (
                        particle_mode_seed_median,
                        True,
                        False,
                        False,
                    ),
                }
                for block_index in range(4):
                    start = block_index * 32
                    stop = start + 32
                    block_mean = (
                        run.seed_predictions[start:stop].mean(
                            axis=0, dtype=np.float64
                        )
                        .astype(np.float32)
                        .astype(np.float64)
                    )
                    baseline_readouts[
                        f"baseline_seed_block{block_index}_mean"
                    ] = (
                        block_mean,
                        True,
                        False,
                        False,
                    )
                for name, (
                    prediction,
                    target_free,
                    oracle_flag,
                    offline,
                ) in baseline_readouts.items():
                    predictions[name] = prediction
                    readout_meta[name] = {
                        "target_free": target_free,
                        "oracle": oracle_flag,
                        "offline_suffix_readout": offline,
                        "pf_variant": "baseline_readout",
                    }
            run_elapsed = time.perf_counter() - run_started
            run_manifest_rows.append(
                {
                    "well": well,
                    "shard_index": shard_index,
                    "variant": variant_name,
                    "suffix_rows": int(len(prepared.row_idx)),
                    "episode_rows": int(len(episode_union_index)),
                    "elapsed_seconds": float(run_elapsed),
                    "gr_sigma": float(active_input.gr_sigma),
                    "clamp_min_tvt": float(active_input.clamp_min_tvt),
                    "clamp_max_tvt": float(active_input.clamp_max_tvt),
                    "parity_max_abs_ft_vs_fixed": float(
                        diagnostics["parity_max_abs_ft"]
                    ),
                    "ess_mean": float(diagnostics["ess_mean"]),
                    "resampling_rate": float(diagnostics["resampling_rate"]),
                    "best_log_likelihood_per_row": float(
                        diagnostics["best_log_likelihood_per_row"]
                    ),
                    "log_likelihood_std": float(
                        diagnostics["log_likelihood_std"]
                    ),
                    "best_log_likelihood_seed_index": best_loglik_index,
                    "max_likelihood_seed_weight": float(
                        np.max(likelihood_weights)
                    ),
                    "likelihood_effective_seed_count": (
                        likelihood_effective_seed_count
                    ),
                    "peak_rss_gb_after": float(base.max_rss_gb()),
                }
            )
            elapsed = time.perf_counter() - started
            hard_runtime = float(
                base.get_nested(config, "execution.hard_runtime_seconds")
            )
            hard_rss = float(
                base.get_nested(config, "execution.hard_peak_rss_gb")
            )
            if elapsed > hard_runtime:
                raise RuntimeError(
                    f"counterfactual hard runtime guard exceeded: {elapsed:.1f}s"
                )
            if base.max_rss_gb() > hard_rss:
                raise RuntimeError(
                    "counterfactual hard RSS guard exceeded: "
                    f"{base.max_rss_gb():.3f}GB"
                )
            del run
            gc.collect()

        for readout, prediction in predictions.items():
            meta = readout_meta[readout]
            sha = prediction_sha(prediction)
            for scope, index in (
                ("all_suffix_rows", all_index),
                ("fixed_episode_rows", episode_union_index),
            ):
                metric_rows.append(
                    metric_row(
                        well=well,
                        scope=scope,
                        readout=readout,
                        prediction=prediction[index],
                        truth=prepared.truth[index],
                        prediction_content_sha256=sha,
                        target_free=bool(meta["target_free"]),
                        oracle=bool(meta["oracle"]),
                        offline_suffix_readout=bool(
                            meta["offline_suffix_readout"]
                        ),
                        pf_variant=str(meta["pf_variant"]),
                    )
                )
            for episode_id, index in episode_parts:
                row = metric_row(
                    well=well,
                    scope="single_fixed_episode",
                    readout=readout,
                    prediction=prediction[index],
                    truth=prepared.truth[index],
                    prediction_content_sha256=sha,
                    target_free=bool(meta["target_free"]),
                    oracle=bool(meta["oracle"]),
                    offline_suffix_readout=bool(
                        meta["offline_suffix_readout"]
                    ),
                    pf_variant=str(meta["pf_variant"]),
                )
                row["episode_id"] = episode_id
                episode_metric_rows.append(row)

        row_part = pd.DataFrame(
            {
                "well": well,
                "id": prepared.ids[episode_union_index],
                "row_idx": prepared.row_idx[episode_union_index],
                "true_tvt": prepared.truth[episode_union_index],
                "fixed_likpf_mean": prepared.fixed_prediction[
                    episode_union_index
                ],
            }
        )
        for readout, prediction in predictions.items():
            row_part[readout] = prediction[episode_union_index].astype(np.float32)
        row_prediction_parts.append(row_part)
        print(
            f"[{well_number:02d}/{len(selected):02d}] {well}: "
            f"{len(EXPECTED_VARIANTS)} variants, "
            f"suffix={len(prepared.row_idx)}, episodes={len(well_episodes)}, "
            f"peak_rss={base.max_rss_gb():.3f}GB"
        )
        del predictions, prepared, replay_input, cache_rows
        gc.collect()

    metrics = add_baseline_deltas(pd.DataFrame(metric_rows))
    episode_metrics = pd.DataFrame(episode_metric_rows)
    episode_baseline = episode_metrics.loc[
        episode_metrics["readout"] == "baseline_arithmetic_seed_mean",
        ["episode_id", "sse", "rmse_ft"],
    ].rename(
        columns={
            "sse": "baseline_sse",
            "rmse_ft": "baseline_rmse_ft",
        }
    )
    episode_metrics = episode_metrics.merge(
        episode_baseline,
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    episode_metrics["sse_delta_vs_baseline"] = (
        episode_metrics["sse"] - episode_metrics["baseline_sse"]
    )
    episode_metrics["rmse_delta_vs_baseline_ft"] = (
        episode_metrics["rmse_ft"] - episode_metrics["baseline_rmse_ft"]
    )
    episode_metrics["sse_relative_vs_baseline"] = episode_metrics[
        "sse"
    ] / np.maximum(episode_metrics["baseline_sse"], 1.0e-12)
    rows = pd.concat(row_prediction_parts, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="stable"
    )
    expected_episode_rows = int(selected_episodes["rows"].sum())
    if (
        metrics["well"].nunique() != len(selected)
        or episode_metrics["episode_id"].nunique() != len(selected_episodes)
        or len(rows) != expected_episode_rows
        or rows[["well", "row_idx"]].duplicated().any()
    ):
        raise RuntimeError("strict counterfactual selected coverage failed")
    run_manifest = pd.DataFrame(run_manifest_rows).sort_values(
        ["well", "variant"], kind="stable"
    )
    aggregate = aggregate_summary(metrics)

    output = base.output_dir()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "variant_metrics": output / f"{PREFIX}_variant_metrics.csv",
        "aggregate_summary": output / f"{PREFIX}_aggregate_summary.csv",
        "episode_metrics": output / f"{PREFIX}_episode_metrics.csv",
        "episode_row_predictions": output
        / f"{PREFIX}_episode_row_predictions.csv.gz",
        "run_manifest": output / f"{PREFIX}_run_manifest.csv",
    }
    metrics.to_csv(paths["variant_metrics"], index=False)
    aggregate.to_csv(paths["aggregate_summary"], index=False)
    episode_metrics.to_csv(paths["episode_metrics"], index=False)
    rows.to_csv(
        paths["episode_row_predictions"],
        index=False,
        compression="gzip",
    )
    run_manifest.to_csv(paths["run_manifest"], index=False)

    elapsed = time.perf_counter() - started
    summary = {
        "experiment": EXPERIMENT,
        "stage": (
            "counterfactual_preflight"
            if counterfactual_preflight
            else "counterfactual"
        ),
        "status": "complete",
        "shard_index": shard_index,
        "counts": {
            "wells": int(len(selected)),
            "suffix_rows": int(selected["suffix_rows"].sum()),
            "episodes": int(len(selected_episodes)),
            "episode_rows": int(selected_episodes["rows"].sum()),
            "pf_variants": len(EXPECTED_VARIANTS),
            "pf_well_runs": int(len(run_manifest)),
            "episode_prediction_rows": int(len(rows)),
        },
        "parity": {
            "baseline_max_abs_ft": parity_max,
            "failed_wells": 0,
        },
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "sum_variant_well_seconds": float(
                run_manifest["elapsed_seconds"].sum()
            ),
            "peak_rss_gb": float(base.max_rss_gb()),
        },
        "fixed_assets": {
            "sentinels": str(sentinel_asset_path),
            "sentinels_sha256": base.sha256_path(sentinel_asset_path),
            "sentinel_manifest": str(sentinel_manifest_path),
            "sentinel_manifest_sha256": base.sha256_path(
                sentinel_manifest_path
            ),
            "persistent_assets": fixed_asset_meta,
            "fixed_prediction_source": source_meta,
        },
        "implementation": {
            "config_path": str(base.config_path()),
            "config_sha256": base.sha256_path(base.config_path()),
            "baseline_kernel_path": str(Path(base.__file__)),
            "baseline_kernel_sha256": base.sha256_path(Path(base.__file__)),
            "counterfactual_runner_path": str(Path(__file__)),
            "counterfactual_runner_sha256": base.sha256_path(Path(__file__)),
            "variant_contract_sha256": hashlib.sha256(
                json.dumps(
                    configured_variants,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": base.sha256_path(path),
                "decompressed_sha256": (
                    base.sha256_path(path, decompressed=True)
                    if path.suffix == ".gz"
                    else None
                ),
            }
            for name, path in paths.items()
        },
        "guards": {
            "strict_selected_well_coverage": True,
            "strict_episode_coverage": True,
            "baseline_persisted_prediction_parity_zero": parity_max == 0.0,
            "truth_not_used_by_pf_dynamics": True,
            "no_lightgbm_fold_booster_gpu_inference_or_submission": True,
        },
    }
    summary_path = output / f"{PREFIX}_summary.json"
    base.write_json(summary_path, summary)
    base.write_json(
        base.metrics_path(),
        {
            "experiment": EXPERIMENT,
            "route": "pf_beam",
            "status": "counterfactual_shard_complete",
            "metric": "paired_mechanism_intervention_not_candidate_cv",
            "shard_index": shard_index,
            "counterfactual_preflight": counterfactual_preflight,
            "counts": summary["counts"],
            "parity": summary["parity"],
            "runtime": summary["runtime"],
            "guards": summary["guards"],
        },
    )
    print("Counterfactual aggregate summary")
    print(aggregate.to_string(index=False))
    print(json.dumps(base.to_jsonable(summary), indent=2, sort_keys=True))
    return summary


if os.environ.get("EXP410_COUNTERFACTUAL_IMPORT_ONLY", "0") != "1":
    RESULT = main()
else:
    RESULT = None
