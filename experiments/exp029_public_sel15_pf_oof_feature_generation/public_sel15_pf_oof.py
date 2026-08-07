from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested, load_config
from sklearn.model_selection import GroupKFold

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]


@dataclass(frozen=True)
class ParticleFilterResult:
    prediction: np.ndarray
    log_likelihood: float
    mean_effective_particles: float
    min_effective_particles: float
    resample_count: int
    gr_sigma: float
    initial_rate: float


@dataclass(frozen=True)
class EnsembleResult:
    predictions_by_scale: dict[float, np.ndarray]
    seed_mean: np.ndarray
    seed_std: np.ndarray
    log_likelihoods: np.ndarray
    diagnostics: list[ParticleFilterResult]


@dataclass(frozen=True)
class BeamResult:
    mean: np.ndarray
    std: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray
    final_cost_mean: float
    final_cost_min: float


def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = get_nested(config, key)
    return default if value is None else value


def load_well(data_dir: Path, well_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(data_dir / f"{well_id}__typewell.csv")
    return horizontal, typewell


def list_train_wells(train_dir: Path) -> list[str]:
    return sorted(path.name.split("__")[0] for path in train_dir.glob("*__horizontal_well.csv"))


def choose_wells(wells: list[str], config: dict[str, Any]) -> list[str]:
    max_wells = config_value(config, "runtime.debug_n_wells", None)
    if max_wells in {"", "null", "None"}:
        max_wells = None
    if max_wells is None:
        return wells
    return wells[: int(max_wells)]


def assign_group_folds(wells: list[str], n_folds: int) -> dict[str, int]:
    if not wells:
        return {}
    if len(wells) == 1:
        return {wells[0]: 0}
    n_splits = min(max(2, int(n_folds)), len(wells))
    dummy_x = np.arange(len(wells))
    groups = np.asarray(wells)
    out: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid_idx) in enumerate(splitter.split(dummy_x, groups=groups)):
        for idx in valid_idx:
            out[wells[int(idx)]] = fold
    return out


def distance_bucket(eval_step: np.ndarray) -> np.ndarray:
    bins = np.asarray([49, 249, 999, 2499], dtype=float)
    labels = np.asarray(["rows_0_49", "rows_50_249", "rows_250_999", "rows_1000_2499"])
    bucket_idx = np.searchsorted(bins, eval_step, side="right")
    clipped_idx = np.minimum(bucket_idx, len(labels) - 1)
    bucket = labels[clipped_idx]
    return np.where(bucket_idx >= len(labels), "rows_2500_plus", bucket)


def gr_fill_values(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    tw_s = typewell.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr_raw = tw_s["GR"].to_numpy(dtype=float)
    fallback = float(np.nanmean(tw_gr_raw)) if np.isfinite(np.nanmean(tw_gr_raw)) else 0.0
    tw_gr = pd.Series(tw_gr_raw).interpolate(limit_direction="both").fillna(fallback)
    return tw_tvt, tw_gr.to_numpy(dtype=float), fallback


def make_pseudo_hidden(
    horizontal: pd.DataFrame,
    cutoff_fraction: float,
    min_known_rows: int,
    min_eval_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    if "TVT" not in horizontal.columns:
        raise ValueError("train horizontal well must contain TVT for pseudo-hidden scoring")
    n_rows = len(horizontal)
    cutoff_row = int(round(float(cutoff_fraction) * n_rows))
    cutoff_row = max(int(min_known_rows), min(cutoff_row, n_rows - int(min_eval_rows)))
    if cutoff_row <= 0 or cutoff_row >= n_rows:
        raise ValueError(f"invalid pseudo-hidden cutoff {cutoff_row} for {n_rows} rows")

    pseudo = horizontal.copy()
    if "TVT_input" not in pseudo.columns:
        pseudo["TVT_input"] = pseudo["TVT"]
    pseudo["TVT_input"] = pseudo["TVT_input"].where(pseudo.index < cutoff_row, np.nan)
    eval_indices = np.arange(cutoff_row, n_rows, dtype=int)
    return pseudo, eval_indices, cutoff_row


def run_particle_filter(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    n_particles: int,
    seed: int,
) -> ParticleFilterResult:
    tw_tvt, tw_gr, tw_gr_mean = gr_fill_values(typewell)
    known = horizontal[horizontal["TVT_input"].notna()]
    eval_frame = horizontal[horizontal["TVT_input"].isna()]
    out_values = horizontal["TVT_input"].to_numpy(dtype=float).copy()
    if eval_frame.empty:
        return ParticleFilterResult(
            out_values,
            0.0,
            float(n_particles),
            float(n_particles),
            0,
            0.0,
            0.0,
        )
    if known.empty:
        fallback = float(np.nanmean(tw_tvt)) if np.isfinite(np.nanmean(tw_tvt)) else 0.0
        out_values[eval_frame.index.to_numpy()] = fallback
        return ParticleFilterResult(out_values, -1e9, 0.0, 0.0, 0, 0.0, 0.0)

    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])

    known_gr = (
        known["GR"].interpolate(limit_direction="both").fillna(tw_gr_mean).to_numpy(dtype=float)
    )
    tw_at_known = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known_gr - tw_at_known), 10.0, 60.0))

    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(dtype=float))
    dz = np.diff(tail["Z"].to_numpy(dtype=float))
    dm = np.diff(tail["MD"].to_numpy(dtype=float))
    valid_step = dm > 0
    initial_rate = (
        float(np.median((dt + dz)[valid_step] / dm[valid_step]))
        if valid_step.sum() >= 3
        else 0.0
    )

    rng = np.random.default_rng(seed)
    n = int(n_particles)
    position = last_tvt + last_z + 3.0 * rng.standard_normal(n)
    rate = initial_rate + 0.01 * rng.standard_normal(n)
    weights = np.ones(n, dtype=float) / n

    momentum = 0.998
    velocity_noise = 0.002
    position_noise = 0.005
    resample_position_noise = 0.1
    resample_rate_noise = 0.001
    resample_threshold = 0.5

    md_values = eval_frame["MD"].to_numpy(dtype=float)
    z_values = eval_frame["Z"].to_numpy(dtype=float)
    gr_interp = horizontal["GR"].interpolate(limit_direction="both").fillna(tw_gr_mean)
    gr_values = gr_interp.to_numpy(dtype=float)[eval_frame.index.to_numpy(dtype=int)]

    predictions = np.empty(len(eval_frame), dtype=float)
    effective_values: list[float] = []
    log_likelihood = 0.0
    resample_count = 0
    previous_md = last_md

    for row_idx in range(len(eval_frame)):
        md_step = max(float(md_values[row_idx] - previous_md), 1.0)
        rate = momentum * rate + velocity_noise * rng.standard_normal(n)
        position = position + rate * md_step + position_noise * rng.standard_normal(n)
        tvt_particles = position - z_values[row_idx]
        tvt_particles = np.clip(tvt_particles, tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)
        position = tvt_particles + z_values[row_idx]

        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        normalized_delta = (gr_values[row_idx] - expected_gr) / gr_sigma
        likelihood = np.exp(-0.5 * np.minimum(normalized_delta**2, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        average_likelihood = float((weights * likelihood).sum())
        log_likelihood += math.log(max(average_likelihood, 1e-300))

        weights = weights * likelihood
        weight_sum = float(weights.sum())
        weights = weights / weight_sum if weight_sum > 0 else np.ones(n, dtype=float) / n
        effective_n = float(1.0 / np.square(weights).sum())
        effective_values.append(effective_n)

        if effective_n < resample_threshold * n:
            cumulative = np.cumsum(weights)
            u0 = rng.uniform(0.0, 1.0 / n)
            indices = np.clip(np.searchsorted(cumulative, u0 + np.arange(n) / n), 0, n - 1)
            position = position[indices] + resample_position_noise * rng.standard_normal(n)
            rate = rate[indices] + resample_rate_noise * rng.standard_normal(n)
            weights = np.ones(n, dtype=float) / n
            resample_count += 1

        predictions[row_idx] = float(np.dot(weights, position - z_values[row_idx]))
        previous_md = md_values[row_idx]

    out_values[eval_frame.index.to_numpy(dtype=int)] = predictions
    effective = np.asarray(effective_values, dtype=float)
    return ParticleFilterResult(
        prediction=out_values,
        log_likelihood=float(log_likelihood),
        mean_effective_particles=float(effective.mean()) if effective.size else float(n),
        min_effective_particles=float(effective.min()) if effective.size else float(n),
        resample_count=int(resample_count),
        gr_sigma=gr_sigma,
        initial_rate=initial_rate,
    )


def run_pf_ensemble_scales(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    scales: list[float],
    n_particles: int,
    n_seeds: int,
    seed_offset: int,
) -> EnsembleResult:
    diagnostics = [
        run_particle_filter(horizontal, typewell, n_particles=n_particles, seed=seed_offset + seed)
        for seed in range(int(n_seeds))
    ]
    prediction_array = np.stack([item.prediction for item in diagnostics], axis=0)
    log_likelihoods = np.asarray([item.log_likelihood for item in diagnostics], dtype=float)
    centered = log_likelihoods - np.nanmax(log_likelihoods)

    predictions_by_scale: dict[float, np.ndarray] = {}
    for scale in scales:
        weights = np.exp(centered / float(scale))
        weights /= weights.sum()
        predictions_by_scale[float(scale)] = (weights[:, None] * prediction_array).sum(axis=0)

    return EnsembleResult(
        predictions_by_scale=predictions_by_scale,
        seed_mean=prediction_array.mean(axis=0),
        seed_std=prediction_array.std(axis=0),
        log_likelihoods=log_likelihoods,
        diagnostics=diagnostics,
    )


def smooth_centered(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) <= max(3, 2 * radius + 1):
        return values.copy()
    return (
        pd.Series(values)
        .rolling(window=2 * radius + 1, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def beam_search(
    horizontal_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_tvt: float,
    *,
    beam_size: int,
    move_cost: float,
    error_scale: float,
    smooth_radius: int,
) -> tuple[np.ndarray, float]:
    n_steps = len(horizontal_gr)
    n_typewell = len(tw_tvt)
    if n_steps == 0:
        return np.asarray([last_tvt], dtype=float), 0.0

    smoothed_gr = smooth_centered(horizontal_gr, smooth_radius)
    start_idx = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    move_penalty = move_cost * np.asarray([2.0, 1.0, 0.0, 1.0, 2.0], dtype=float)

    beam_indices = np.full(beam_size, start_idx, dtype=np.int64)
    beam_cost = np.full(beam_size, np.inf, dtype=float)
    beam_cost[0] = 0.0
    active = 1
    result = np.zeros(n_steps, dtype=float)

    for step, gr_value in enumerate(smoothed_gr):
        next_idx = beam_indices[:active, None] + moves[None, :]
        clipped_idx = np.clip(next_idx, 0, n_typewell - 1)
        valid = (next_idx >= 0) & (next_idx < n_typewell)

        gr_error = np.square(gr_value - tw_gr[clipped_idx]) / error_scale
        total = beam_cost[:active, None] + gr_error + move_penalty[None, :]
        total = np.where(valid, total, np.inf)

        flat_idx = next_idx.ravel()[valid.ravel()]
        flat_cost = total.ravel()[valid.ravel()]
        order = np.argsort(flat_cost)
        sorted_idx = flat_idx[order]
        sorted_cost = flat_cost[order]
        _, first_seen = np.unique(sorted_idx, return_index=True)
        unique_idx = sorted_idx[first_seen]
        unique_cost = sorted_cost[first_seen]

        kept = min(beam_size, len(unique_idx))
        top = np.argpartition(unique_cost, kept - 1)[:kept]
        top = top[np.argsort(unique_cost[top])]
        beam_indices[:kept] = unique_idx[top]
        beam_cost[:kept] = unique_cost[top]
        if kept < beam_size:
            beam_indices[kept:] = beam_indices[kept - 1]
            beam_cost[kept:] = np.inf
        active = kept
        result[step] = tw_tvt[beam_indices[0]]

    return result, float(beam_cost[0])


def run_beam_ensemble(horizontal: pd.DataFrame, typewell: pd.DataFrame) -> BeamResult:
    tw_tvt, tw_gr, tw_gr_mean = gr_fill_values(typewell)
    known = horizontal[horizontal["TVT_input"].notna()]
    eval_frame = horizontal[horizontal["TVT_input"].isna()]
    out_template = horizontal["TVT_input"].to_numpy(dtype=float).copy()
    if eval_frame.empty:
        return BeamResult(
            out_template,
            np.zeros_like(out_template),
            out_template,
            out_template,
            0.0,
            0.0,
        )
    if known.empty:
        fallback = float(np.nanmean(tw_tvt)) if np.isfinite(np.nanmean(tw_tvt)) else 0.0
        out_template[eval_frame.index.to_numpy()] = fallback
        return BeamResult(
            out_template,
            np.zeros_like(out_template),
            out_template,
            out_template,
            0.0,
            0.0,
        )

    gr_all = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(tw_gr_mean)
        .to_numpy(dtype=float)
    )
    hgr = gr_all[eval_frame.index.to_numpy(dtype=int)]
    last_tvt = float(known.iloc[-1]["TVT_input"])
    member_predictions = []
    final_costs = []

    for beam_size, move_cost, error_scale, smooth_radius in BEAM_CONFIGS:
        eval_prediction, final_cost = beam_search(
            hgr,
            tw_tvt,
            tw_gr,
            last_tvt,
            beam_size=beam_size,
            move_cost=move_cost,
            error_scale=error_scale,
            smooth_radius=smooth_radius,
        )
        full_prediction = out_template.copy()
        full_prediction[eval_frame.index.to_numpy(dtype=int)] = eval_prediction
        member_predictions.append(full_prediction)
        final_costs.append(final_cost)

    members = np.stack(member_predictions, axis=0)
    final_cost_array = np.asarray(final_costs, dtype=float)
    return BeamResult(
        mean=members.mean(axis=0),
        std=members.std(axis=0),
        min_value=members.min(axis=0),
        max_value=members.max(axis=0),
        final_cost_mean=float(final_cost_array.mean()),
        final_cost_min=float(final_cost_array.min()),
    )


def selector_well_code(horizontal: pd.DataFrame) -> tuple[int, str, float, float]:
    eval_mask = horizontal["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = horizontal.loc[eval_mask, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT), n_eval, z_span


def parse_selector_variant(name: str) -> tuple[float, float, float]:
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(
    variant: str,
    pf_by_scale: dict[float, np.ndarray],
    beam_prediction: np.ndarray,
    last_known_tvt: float,
) -> np.ndarray:
    scale, beam_weight, hold_weight = parse_selector_variant(variant)
    base = pf_by_scale.get(scale, pf_by_scale[8.0])
    pred = (1.0 - beam_weight) * base + beam_weight * beam_prediction
    return (1.0 - hold_weight) * pred + hold_weight * last_known_tvt


def load_reference_oof(path_value: Any) -> pd.DataFrame:
    if path_value in {None, "", "TODO"}:
        return pd.DataFrame(columns=["id", "exp026_oof"])
    path = Path(str(path_value))
    if not path.exists():
        return pd.DataFrame(columns=["id", "exp026_oof"])
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        return pd.DataFrame(columns=["id", "exp026_oof"])
    for candidate in ("exp026_oof", "prediction", "pred", "tvt", "TVT"):
        if candidate in frame.columns:
            return frame[["id", candidate]].rename(columns={candidate: "exp026_oof"})
    return pd.DataFrame(columns=["id", "exp026_oof"])


def feature_output_path(paths: ExperimentPaths, config: dict[str, Any]) -> Path:
    compression = str(config_value(config, "runtime.output_compression", "") or "").lower()
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    return paths.features_dir / f"public_sel15_pf_oof_features{suffix}"


def to_csv_compression(config: dict[str, Any]) -> str | None:
    compression = str(config_value(config, "runtime.output_compression", "") or "").lower()
    return "gzip" if compression == "gzip" else None


def likelihood_summary(log_likelihoods: np.ndarray, scale: float) -> dict[str, float]:
    centered = log_likelihoods - np.nanmax(log_likelihoods)
    weights = np.exp(centered / float(scale))
    weights /= weights.sum()
    sorted_weights = np.sort(weights)[::-1]
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-300))))
    return {
        "pf_lik_best": float(np.nanmax(log_likelihoods)),
        "pf_lik_mean": float(np.nanmean(log_likelihoods)),
        "pf_lik_std": float(np.nanstd(log_likelihoods)),
        "pf_lik_gap_best_second": float(np.sort(log_likelihoods)[-1] - np.sort(log_likelihoods)[-2])
        if len(log_likelihoods) >= 2
        else 0.0,
        "pf_weight_entropy": entropy,
        "pf_weight_top": float(sorted_weights[0]) if len(sorted_weights) else 1.0,
        "pf_weight_best_second_gap": float(sorted_weights[0] - sorted_weights[1])
        if len(sorted_weights) >= 2
        else 1.0,
        "pf_effective_seeds": float(1.0 / np.square(weights).sum()),
    }


def build_feature_rows(
    *,
    well_id: str,
    fold: int,
    cutoff_fraction: float,
    cutoff_row: int,
    horizontal: pd.DataFrame,
    eval_indices: np.ndarray,
    pf_result: EnsembleResult,
    beam_result: BeamResult,
    exp026_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    code, variant, selector_n_eval, selector_z_span = selector_well_code(horizontal)
    selected_scale, selector_beam_weight, selector_hold_weight = parse_selector_variant(variant)
    last_known_tvt = float(horizontal.loc[horizontal["TVT_input"].notna(), "TVT_input"].iloc[-1])
    selector_prediction = apply_selector_variant(
        variant,
        pf_result.predictions_by_scale,
        beam_result.mean,
        last_known_tvt,
    )
    selected_pf = pf_result.predictions_by_scale.get(
        selected_scale,
        pf_result.predictions_by_scale[8.0],
    )
    eval_step = np.arange(len(eval_indices), dtype=float)
    ids = [f"{well_id}_{idx}" for idx in eval_indices]

    feature_frame = pd.DataFrame(
        {
            "id": ids,
            "well_id": well_id,
            "fold": int(fold),
            "pseudo_cutoff_fraction": float(cutoff_fraction),
            "cutoff_row": int(cutoff_row),
            "row_idx": eval_indices.astype(int),
            "prefix_length": int(cutoff_row),
            "eval_step": eval_step,
            "eval_fraction": eval_step / max(float(len(eval_indices) - 1), 1.0),
            "distance_bucket": distance_bucket(eval_step),
            "MD": horizontal.loc[eval_indices, "MD"].to_numpy(dtype=float),
            "X": horizontal.loc[eval_indices, "X"].to_numpy(dtype=float),
            "Y": horizontal.loc[eval_indices, "Y"].to_numpy(dtype=float),
            "Z": horizontal.loc[eval_indices, "Z"].to_numpy(dtype=float),
            "GR": horizontal.loc[eval_indices, "GR"].to_numpy(dtype=float),
            "gr_isna": horizontal.loc[eval_indices, "GR"].isna().astype(int).to_numpy(),
            "gr_prefix_availability": float(horizontal.loc[: cutoff_row - 1, "GR"].notna().mean()),
            "gr_eval_availability": float(horizontal.loc[eval_indices, "GR"].notna().mean()),
            "target_tvt": horizontal.loc[eval_indices, "TVT"].to_numpy(dtype=float),
            "last_anchor_tvt": last_known_tvt,
            "pf_pred": selector_prediction[eval_indices],
            "pf_selected_scale_pred": selected_pf[eval_indices],
            "pf_scale_3": pf_result.predictions_by_scale[3.0][eval_indices],
            "pf_scale_5": pf_result.predictions_by_scale[5.0][eval_indices],
            "pf_scale_8": pf_result.predictions_by_scale[8.0][eval_indices],
            "pf_scale_12": pf_result.predictions_by_scale[12.0][eval_indices],
            "pf_seed_mean": pf_result.seed_mean[eval_indices],
            "pf_seed_std": pf_result.seed_std[eval_indices],
            "beam_pred": beam_result.mean[eval_indices],
            "beam_spread": beam_result.std[eval_indices],
            "beam_min": beam_result.min_value[eval_indices],
            "beam_max": beam_result.max_value[eval_indices],
            "selector_code": int(code),
            "selector_variant": variant,
            "selector_scale": selected_scale,
            "selector_beam_weight": selector_beam_weight,
            "selector_hold_weight": selector_hold_weight,
            "selector_n_eval": selector_n_eval,
            "selector_z_span": selector_z_span,
            "beam_final_cost_mean": beam_result.final_cost_mean,
            "beam_final_cost_min": beam_result.final_cost_min,
        }
    )

    for key, value in likelihood_summary(pf_result.log_likelihoods, selected_scale).items():
        feature_frame[key] = value
    diag = pf_result.diagnostics
    feature_frame["pf_gr_sigma_mean"] = float(np.mean([item.gr_sigma for item in diag]))
    feature_frame["pf_initial_rate_mean"] = float(np.mean([item.initial_rate for item in diag]))
    feature_frame["pf_mean_effective_particles"] = float(
        np.mean([item.mean_effective_particles for item in diag])
    )
    feature_frame["pf_min_effective_particles"] = float(
        np.min([item.min_effective_particles for item in diag])
    )
    feature_frame["pf_resample_count_mean"] = float(np.mean([item.resample_count for item in diag]))

    feature_frame = feature_frame.merge(exp026_oof, on="id", how="left")
    feature_frame["pf_pred_minus_last_anchor"] = (
        feature_frame["pf_pred"] - feature_frame["last_anchor_tvt"]
    )
    feature_frame["pf_pred_minus_exp026_oof"] = (
        feature_frame["pf_pred"] - feature_frame["exp026_oof"]
    )
    feature_frame["abs_pf_pred_minus_exp026_oof"] = feature_frame["pf_pred_minus_exp026_oof"].abs()
    feature_frame["pf_beam_diff"] = feature_frame["pf_pred"] - feature_frame["beam_pred"]
    feature_frame["abs_pf_beam_diff"] = feature_frame["pf_beam_diff"].abs()
    feature_frame["pf_error"] = feature_frame["pf_pred"] - feature_frame["target_tvt"]
    feature_frame["last_anchor_error"] = (
        feature_frame["last_anchor_tvt"] - feature_frame["target_tvt"]
    )
    feature_frame["beam_error"] = feature_frame["beam_pred"] - feature_frame["target_tvt"]

    summary = {
        "well_id": well_id,
        "fold": int(fold),
        "cutoff_fraction": float(cutoff_fraction),
        "cutoff_row": int(cutoff_row),
        "rows": int(len(feature_frame)),
        "selector_variant": variant,
        "selector_code": int(code),
        "selector_n_eval": float(selector_n_eval),
        "selector_z_span": float(selector_z_span),
        "pf_rmse": rmse(feature_frame["target_tvt"], feature_frame["pf_pred"]),
        "last_anchor_rmse": rmse(feature_frame["target_tvt"], feature_frame["last_anchor_tvt"]),
        "beam_rmse": rmse(feature_frame["target_tvt"], feature_frame["beam_pred"]),
        "pf_lik_best": float(feature_frame["pf_lik_best"].iloc[0]),
        "pf_weight_entropy": float(feature_frame["pf_weight_entropy"].iloc[0]),
        "beam_final_cost_min": float(beam_result.final_cost_min),
    }
    return feature_frame, summary


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.nanmean(np.square(diff))))


def run_feature_generation(paths: ExperimentPaths, config: dict[str, Any]) -> dict[str, Any]:
    paths.ensure_output_dirs()
    train_dir = paths.train_data_dir
    wells = choose_wells(list_train_wells(train_dir), config)
    folds = assign_group_folds(wells, int(config_value(config, "validation.n_folds", 5)))

    pf_config = config_value(config, "model.public_sel15_pf", {})
    cutoff_fractions = [
        float(value) for value in config_value(config, "model.cutoff_fractions", [0.65])
    ]
    scales = [float(value) for value in pf_config.get("selector_scales", [3.0, 5.0, 8.0, 12.0])]
    n_particles = int(pf_config.get("n_particles", 250))
    n_seeds = int(pf_config.get("n_seeds", 16))
    min_known_rows = int(config_value(config, "model.min_known_rows", 500))
    min_eval_rows = int(config_value(config, "model.min_eval_rows", 200))
    seed_offset = int(config_value(config, "validation.seed", 42)) * 1000
    exp026_oof = load_reference_oof(config_value(config, "data.reference_oof_path", None))

    feature_path = feature_output_path(paths, config)
    well_summary_path = paths.artifacts_dir / "public_sel15_pf_oof_well_summary.csv"
    metrics_path = paths.metrics_path
    for stale_path in paths.features_dir.glob("public_sel15_pf_oof_features.csv*"):
        stale_path.unlink()
    csv_compression = to_csv_compression(config)

    summaries: list[dict[str, Any]] = []
    total_rows = 0
    header = True
    squared_error_sums = {"pf": 0.0, "last_anchor": 0.0, "beam": 0.0}

    for well_position, well_id in enumerate(wells):
        horizontal, typewell = load_well(train_dir, well_id)
        for cutoff_position, cutoff_fraction in enumerate(cutoff_fractions):
            pseudo, eval_indices, cutoff_row = make_pseudo_hidden(
                horizontal,
                cutoff_fraction,
                min_known_rows=min_known_rows,
                min_eval_rows=min_eval_rows,
            )
            scenario_seed_offset = seed_offset + 100 * well_position + 10 * cutoff_position
            pf_result = run_pf_ensemble_scales(
                pseudo,
                typewell,
                scales=scales,
                n_particles=n_particles,
                n_seeds=n_seeds,
                seed_offset=scenario_seed_offset,
            )
            beam_result = run_beam_ensemble(pseudo, typewell)
            feature_frame, summary = build_feature_rows(
                well_id=well_id,
                fold=folds[well_id],
                cutoff_fraction=cutoff_fraction,
                cutoff_row=cutoff_row,
                horizontal=pseudo,
                eval_indices=eval_indices,
                pf_result=pf_result,
                beam_result=beam_result,
                exp026_oof=exp026_oof,
            )
            feature_frame.to_csv(
                feature_path,
                mode="a",
                header=header,
                index=False,
                compression=csv_compression,
            )
            header = False
            summaries.append(summary)
            total_rows += len(feature_frame)
            squared_error_sums["pf"] += float(np.square(feature_frame["pf_error"]).sum())
            squared_error_sums["last_anchor"] += float(
                np.square(feature_frame["last_anchor_error"]).sum()
            )
            squared_error_sums["beam"] += float(np.square(feature_frame["beam_error"]).sum())
            print(
                f"{well_id} cutoff={cutoff_fraction:.2f} rows={len(feature_frame)} "
                f"pf_rmse={summary['pf_rmse']:.4f} hold_rmse={summary['last_anchor_rmse']:.4f}"
            )

    well_summary = pd.DataFrame(summaries)
    well_summary.to_csv(well_summary_path, index=False)
    metrics = {
        "experiment": config["experiment"]["name"],
        "status": "feature_generation_implemented",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "mode": "smoke" if config_value(config, "runtime.debug_n_wells", None) else "full",
        "wells": len(wells),
        "cutoffs": cutoff_fractions,
        "rows": int(total_rows),
        "n_particles": n_particles,
        "n_seeds": n_seeds,
        "pf_rmse": math.sqrt(squared_error_sums["pf"] / total_rows) if total_rows else None,
        "last_anchor_rmse": math.sqrt(squared_error_sums["last_anchor"] / total_rows)
        if total_rows
        else None,
        "beam_rmse": math.sqrt(squared_error_sums["beam"] / total_rows) if total_rows else None,
        "feature_file": feature_path.as_posix(),
        "well_summary_file": well_summary_path.as_posix(),
        "reference_oof_rows": int(len(exp026_oof)),
        "notes": "OOF-like public sel15 PF/Beam feature artifact; no meta model fitted yet.",
    }
    with metrics_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
        fp.write("\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate public sel15 PF OOF-like features.")
    parser.add_argument("--allow-local", action="store_true", help="Allow local smoke execution.")
    parser.add_argument("--debug-n-wells", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--cutoffs", default=None, help="Comma-separated cutoff fractions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_config()
    if not args.allow_local:
        paths.require_kaggle_runtime()
    if args.debug_n_wells is not None:
        config.setdefault("runtime", {})["debug_n_wells"] = args.debug_n_wells
    pf_config = config.setdefault("model", {}).setdefault("public_sel15_pf", {})
    if args.n_seeds is not None:
        pf_config["n_seeds"] = args.n_seeds
    if args.n_particles is not None:
        pf_config["n_particles"] = args.n_particles
    if args.cutoffs:
        config["model"]["cutoff_fractions"] = [float(value) for value in args.cutoffs.split(",")]

    metrics = run_feature_generation(paths, config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
