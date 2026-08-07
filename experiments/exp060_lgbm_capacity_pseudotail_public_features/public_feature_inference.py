from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pseudo_tail_augmentation import test_files, train_files
from public_feature_model_audit import (
    VariantSpec,
    add_derived_features,
    add_ncc_gr_match_features,
    apply_prediction_policy,
    bucket_codes,
    choose_train_indices_for_variant,
    compute_ncc_gr_match_group,
    feature_variants,
    get_nested,
    load_features,
    make_estimator,
    resolve_feature_path,
    stable_fold,
    transformed_features,
)
from settings import ExperimentPaths

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
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)
PF_N_PARTICLES = 250
PF_N_SEEDS = 16
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


def tvt_from_contacts(
    hw_tr: pd.DataFrame,
    tw_tr: pd.DataFrame,
    ref_col: str = "EGFDU",
) -> pd.Series:
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = str(tw_g["Geology"].iloc[0])
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


def selector_well_code(hw: pd.DataFrame) -> tuple[int, str, float, float]:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].values.astype(float)
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
    name: str,
    pf_by_scale: dict[str, np.ndarray],
    tvt_beam: np.ndarray,
    last_known_tvt: float,
) -> np.ndarray:
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale["pf_scale_8"])
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    return (1.0 - hold_weight) * pred + hold_weight * last_known_tvt


def run_particle_filter_diag(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr_raw = tw_s["GR"].values.astype(float)
    tw_gr_mean = float(np.nanmean(tw_gr_raw)) if np.isfinite(np.nanmean(tw_gr_raw)) else 0.0
    tw_gr = (
        pd.Series(tw_gr_raw)
        .interpolate(limit_direction="both")
        .fillna(tw_gr_mean)
        .values.astype(float)
    )

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    out_vals = hw["TVT_input"].values.astype(float).copy()
    if len(ev) == 0:
        return {
            "prediction": out_vals,
            "log_likelihood": 0.0,
            "mean_effective_particles": float(n_particles),
            "min_effective_particles": float(n_particles),
            "resample_count": 0,
            "gr_sigma": 0.0,
            "initial_rate": 0.0,
        }
    if len(kn) == 0:
        fallback = float(np.nanmean(tw_tvt)) if np.isfinite(np.nanmean(tw_tvt)) else 0.0
        out_vals[list(ev.index)] = fallback
        return {
            "prediction": out_vals,
            "log_likelihood": -1e9,
            "mean_effective_particles": 0.0,
            "min_effective_particles": 0.0,
            "resample_count": 0,
            "gr_sigma": 0.0,
            "initial_rate": 0.0,
        }

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])
    known_gr = (
        kn["GR"].interpolate(limit_direction="both").fillna(tw_gr_mean).values.astype(float)
    )
    tw_at_k = np.interp(kn["TVT_input"].values.astype(float), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known_gr - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    moving = dm > 0
    initial_rate = float(np.median((dt + dz)[moving] / dm[moving])) if moving.sum() >= 3 else 0.0

    n = int(n_particles)
    rng = np.random.default_rng(seed)
    pos = last_tvt + last_z + 3.0 * rng.standard_normal(n)
    rate = initial_rate + 0.01 * rng.standard_normal(n)
    weight = np.ones(n) / n

    md_eval = ev["MD"].values.astype(float)
    z_eval = ev["Z"].values.astype(float)
    gr_eval = (
        hw["GR"]
        .interpolate(limit_direction="both")
        .fillna(tw_gr_mean)
        .values.astype(float)[ev.index]
    )
    result = np.empty(len(ev), dtype=float)
    effective_values: list[float] = []
    log_likelihood = 0.0
    resample_count = 0
    prev_md = last_md

    for idx in range(len(ev)):
        md_step = max(md_eval[idx] - prev_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(n)
        pos = pos + rate * md_step + 0.005 * rng.standard_normal(n)
        tvt_particles = np.clip(pos - z_eval[idx], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_particles + z_eval[idx]

        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        delta = (gr_eval[idx] - expected_gr) / gr_sigma
        likelihood = np.exp(-0.5 * np.minimum(delta**2, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        avg_likelihood = float((weight * likelihood).sum())
        log_likelihood += np.log(max(avg_likelihood, 1e-300))
        weight = weight * likelihood
        weight_sum = weight.sum()
        weight = weight / weight_sum if weight_sum > 0 else np.ones(n) / n

        n_eff = float(1.0 / (weight**2).sum())
        effective_values.append(n_eff)
        if n_eff < 0.5 * n:
            cumulative = np.cumsum(weight)
            start = rng.uniform(0, 1.0 / n)
            selected = np.clip(np.searchsorted(cumulative, start + np.arange(n) / n), 0, n - 1)
            pos = pos[selected] + 0.1 * rng.standard_normal(n)
            rate = rate[selected] + 0.001 * rng.standard_normal(n)
            weight = np.ones(n) / n
            resample_count += 1

        result[idx] = float(np.dot(weight, pos - z_eval[idx]))
        prev_md = md_eval[idx]

    out_vals[list(ev.index)] = result
    effective = np.asarray(effective_values, dtype=float)
    return {
        "prediction": out_vals,
        "log_likelihood": float(log_likelihood),
        "mean_effective_particles": float(effective.mean()) if effective.size else float(n),
        "min_effective_particles": float(effective.min()) if effective.size else float(n),
        "resample_count": int(resample_count),
        "gr_sigma": gr_sigma,
        "initial_rate": initial_rate,
    }


def run_pf_lik_ensemble_scales_diag(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int = PF_N_PARTICLES,
    n_seeds: int = PF_N_SEEDS,
    scales: tuple[float, ...] = SELECTOR_SCALES,
) -> dict[str, Any]:
    diagnostics = [
        run_particle_filter_diag(hw, tw, n_particles=n_particles, seed=42 + seed)
        for seed in range(int(n_seeds))
    ]
    pred_arr = np.stack([item["prediction"] for item in diagnostics], 0)
    likelihoods = np.asarray([item["log_likelihood"] for item in diagnostics], dtype=float)
    centered = likelihoods - np.nanmax(likelihoods)
    predictions_by_scale: dict[str, np.ndarray] = {}
    for scale in scales:
        weights = np.exp(centered / float(scale))
        weights /= weights.sum()
        predictions_by_scale[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(0)
    return {
        "predictions_by_scale": predictions_by_scale,
        "seed_mean": pred_arr.mean(0),
        "seed_std": pred_arr.std(0),
        "log_likelihoods": likelihoods,
        "diagnostics": diagnostics,
    }


def smooth_gr(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) <= max(3, 2 * radius + 1):
        return values.copy()
    try:
        from scipy.signal import savgol_filter

        window = min(2 * radius + 1, len(values) if len(values) % 2 == 1 else len(values) - 1)
        return savgol_filter(values, window, min(2, window - 1))
    except ImportError:
        return (
            pd.Series(values, dtype="float64")
            .rolling(2 * radius + 1, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )


def beam_search_with_cost(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_tvt: float,
    *,
    beam_size: int,
    move_cost: float,
    error_scale: float,
    radius: int,
) -> tuple[np.ndarray, float]:
    n_rows = len(hgr)
    n_tvt = len(tw_tvt)
    if n_rows == 0:
        return np.asarray([last_tvt], dtype=float), 0.0
    smoothed = smooth_gr(hgr, radius)
    start_idx = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    penalties = move_cost * np.asarray([2.0, 1.0, 0.0, 1.0, 2.0])
    beam_idx = np.full(int(beam_size), start_idx, dtype=np.int64)
    beam_cost = np.full(int(beam_size), np.inf)
    beam_cost[0] = 0.0
    active = 1
    result = np.zeros(n_rows)

    for step, gr_value in enumerate(smoothed):
        next_idx = beam_idx[:active, None] + moves[None, :]
        clipped = np.clip(next_idx, 0, n_tvt - 1)
        valid = (next_idx >= 0) & (next_idx < n_tvt)
        gr_error = (gr_value - tw_gr[clipped]) ** 2 / error_scale
        total = beam_cost[:active, None] + gr_error + penalties[None, :]
        total = np.where(valid, total, np.inf)
        flat_idx = next_idx.ravel()[valid.ravel()]
        flat_cost = total.ravel()[valid.ravel()]
        order = np.argsort(flat_cost)
        flat_idx = flat_idx[order]
        flat_cost = flat_cost[order]
        _, first = np.unique(flat_idx, return_index=True)
        unique_idx = flat_idx[first]
        unique_cost = flat_cost[first]
        kept = min(int(beam_size), len(unique_idx))
        top = np.argpartition(unique_cost, min(kept - 1, len(unique_cost) - 1))[:kept]
        top = top[np.argsort(unique_cost[top])]
        beam_idx[:kept] = unique_idx[top]
        beam_cost[:kept] = unique_cost[top]
        if kept < beam_size:
            beam_idx[kept:] = beam_idx[kept - 1]
            beam_cost[kept:] = np.inf
        active = kept
        result[step] = tw_tvt[beam_idx[0]]
    return result, float(beam_cost[0])


def run_beam_ensemble_diag(hw: pd.DataFrame, tw: pd.DataFrame) -> dict[str, Any]:
    known = hw[hw["TVT_input"].notna()]
    eval_rows = hw[hw["TVT_input"].isna()]
    template = hw["TVT_input"].values.astype(float).copy()
    if len(eval_rows) == 0:
        return {
            "mean": template,
            "std": np.zeros_like(template),
            "min": template,
            "max": template,
            "final_cost_mean": 0.0,
            "final_cost_min": 0.0,
        }
    if len(known) == 0:
        fallback = np.nanmean(template) if np.isfinite(np.nanmean(template)) else 0.0
        template[list(eval_rows.index)] = fallback
        return {
            "mean": template,
            "std": np.zeros_like(template),
            "min": template,
            "max": template,
            "final_cost_mean": 0.0,
            "final_cost_min": 0.0,
        }

    last_tvt = float(known.iloc[-1]["TVT_input"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr_raw = tw_s["GR"].values.astype(float)
    tw_gr_mean = float(np.nanmean(tw_gr_raw)) if np.isfinite(np.nanmean(tw_gr_raw)) else 0.0
    tw_gr = (
        pd.Series(tw_gr_raw)
        .interpolate(limit_direction="both")
        .fillna(tw_gr_mean)
        .values.astype(float)
    )
    gr_all = (
        hw["GR"].interpolate(limit_direction="both").fillna(tw_gr_mean).values.astype(float)
    )
    hgr = gr_all[eval_rows.index]
    members: list[np.ndarray] = []
    costs: list[float] = []
    for beam_size, move_cost, error_scale, radius in BEAM_CONFIGS:
        eval_pred, cost = beam_search_with_cost(
            hgr,
            tw_tvt,
            tw_gr,
            last_tvt,
            beam_size=beam_size,
            move_cost=move_cost,
            error_scale=error_scale,
            radius=radius,
        )
        full = template.copy()
        full[list(eval_rows.index)] = eval_pred
        members.append(full)
        costs.append(cost)
    arr = np.stack(members, 0)
    cost_values = np.asarray(costs, dtype=float)
    return {
        "mean": arr.mean(0),
        "std": arr.std(0),
        "min": arr.min(0),
        "max": arr.max(0),
        "final_cost_mean": float(cost_values.mean()),
        "final_cost_min": float(cost_values.min()),
    }


def likelihood_summary(log_likelihoods: np.ndarray, scale: float) -> dict[str, float]:
    likelihoods = np.asarray(log_likelihoods, dtype=float)
    centered = likelihoods - np.nanmax(likelihoods)
    weights = np.exp(centered / float(scale))
    weights /= weights.sum()
    sorted_weights = np.sort(weights)[::-1]
    return {
        "pf_lik_best": float(np.nanmax(likelihoods)),
        "pf_lik_mean": float(np.nanmean(likelihoods)),
        "pf_lik_std": float(np.nanstd(likelihoods)),
        "pf_lik_gap_best_second": (
            float(np.sort(likelihoods)[-1] - np.sort(likelihoods)[-2])
            if len(likelihoods) >= 2
            else 0.0
        ),
        "pf_weight_entropy": float(-np.sum(weights * np.log(np.maximum(weights, 1e-300)))),
        "pf_weight_top": float(sorted_weights[0]) if len(sorted_weights) else 1.0,
        "pf_weight_best_second_gap": (
            float(sorted_weights[0] - sorted_weights[1]) if len(sorted_weights) >= 2 else 1.0
        ),
        "pf_effective_seeds": float(1.0 / np.square(weights).sum()),
    }


def summarize_pf_diagnostics(pf_result: dict[str, Any]) -> dict[str, float]:
    diagnostics = pf_result.get("diagnostics", [])
    if not diagnostics:
        return {
            "pf_mean_effective_particles": 0.0,
            "pf_min_effective_particles": 0.0,
            "pf_resample_count_mean": 0.0,
            "pf_gr_sigma_mean": 0.0,
            "pf_initial_rate_mean": 0.0,
        }
    diag = pd.DataFrame(diagnostics)
    return {
        "pf_mean_effective_particles": float(
            diag.get("mean_effective_particles", pd.Series([0.0])).mean()
        ),
        "pf_min_effective_particles": float(
            diag.get("min_effective_particles", pd.Series([0.0])).min()
        ),
        "pf_resample_count_mean": float(diag.get("resample_count", pd.Series([0.0])).mean()),
        "pf_gr_sigma_mean": float(diag.get("gr_sigma", pd.Series([0.0])).mean()),
        "pf_initial_rate_mean": float(diag.get("initial_rate", pd.Series([0.0])).mean()),
    }


def fallback_pf_result(hw: pd.DataFrame, tvt_pf: np.ndarray) -> dict[str, Any]:
    return {
        "predictions_by_scale": {
            f"pf_scale_{scale:g}": tvt_pf.copy() for scale in SELECTOR_SCALES
        },
        "seed_mean": tvt_pf.copy(),
        "seed_std": np.zeros_like(tvt_pf, dtype=float),
        "log_likelihoods": np.asarray([0.0], dtype=float),
        "diagnostics": [
            {
                "gr_sigma": 0.0,
                "initial_rate": 0.0,
                "mean_effective_particles": 0.0,
                "min_effective_particles": 0.0,
                "resample_count": 0,
            }
        ],
    }


def build_hidden_feature_frame(
    *,
    well_id: str,
    hw: pd.DataFrame,
    pf_result: dict[str, Any],
    beam_result: dict[str, Any],
    tvt_selector: np.ndarray,
    selector_variant: str,
    last_known_tvt: float,
) -> pd.DataFrame:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    eval_indices = np.flatnonzero(eval_mask).astype(int)
    if len(eval_indices) == 0:
        return pd.DataFrame()

    cutoff_row = int(eval_indices[0])
    selected_scale, _, _ = parse_selector_variant(selector_variant)
    scale_key = f"pf_scale_{selected_scale:g}"
    pf_by_scale = pf_result["predictions_by_scale"]
    selected_pf = pf_by_scale.get(scale_key, pf_by_scale["pf_scale_8"])
    eval_step = np.arange(len(eval_indices), dtype=float)
    likelihood = likelihood_summary(
        pf_result.get("log_likelihoods", np.asarray([0.0])), selected_scale
    )
    diagnostics = summarize_pf_diagnostics(pf_result)
    beam_mean = beam_result.get("mean", tvt_selector)
    beam_std = beam_result.get("std", np.zeros_like(beam_mean, dtype=float))
    beam_min = beam_result.get("min", beam_mean)
    beam_max = beam_result.get("max", beam_mean)

    frame = pd.DataFrame(
        {
            "well_id": well_id,
            "pseudo_cutoff_fraction": cutoff_row / max(float(len(hw)), 1.0),
            "cutoff_row": cutoff_row,
            "row_idx": eval_indices,
            "prefix_length": int(hw["TVT_input"].notna().sum()),
            "eval_step": eval_step,
            "eval_fraction": eval_step / max(float(len(eval_indices) - 1), 1.0),
            "MD": hw.loc[eval_indices, "MD"].to_numpy(dtype=float),
            "X": hw.loc[eval_indices, "X"].to_numpy(dtype=float),
            "Y": hw.loc[eval_indices, "Y"].to_numpy(dtype=float),
            "Z": hw.loc[eval_indices, "Z"].to_numpy(dtype=float),
            "gr_isna": hw.loc[eval_indices, "GR"].isna().astype(int).to_numpy(),
            "gr_prefix_availability": (
                float(hw.loc[: max(cutoff_row - 1, 0), "GR"].notna().mean())
                if cutoff_row > 0
                else 0.0
            ),
            "gr_eval_availability": float(hw.loc[eval_indices, "GR"].notna().mean()),
            "last_anchor_tvt": last_known_tvt,
            "pf_pred": tvt_selector[eval_indices],
            "beam_pred": beam_mean[eval_indices],
            "pf_selected_scale_pred": selected_pf[eval_indices],
            "pf_scale_3": pf_by_scale["pf_scale_3"][eval_indices],
            "pf_scale_5": pf_by_scale["pf_scale_5"][eval_indices],
            "pf_scale_8": pf_by_scale["pf_scale_8"][eval_indices],
            "pf_scale_12": pf_by_scale["pf_scale_12"][eval_indices],
            "pf_seed_mean": pf_result["seed_mean"][eval_indices],
            "pf_seed_std": pf_result["seed_std"][eval_indices],
            "pf_beam_diff": tvt_selector[eval_indices] - beam_mean[eval_indices],
            "abs_pf_beam_diff": np.abs(tvt_selector[eval_indices] - beam_mean[eval_indices]),
            "beam_spread": beam_std[eval_indices],
            "beam_min": beam_min[eval_indices],
            "beam_max": beam_max[eval_indices],
            "beam_final_cost_mean": float(beam_result.get("final_cost_mean", 0.0)),
            "beam_final_cost_min": float(beam_result.get("final_cost_min", 0.0)),
        }
    )
    for key, value in likelihood.items():
        frame[key] = value
    for key, value in diagnostics.items():
        frame[key] = value
    return add_derived_features(frame)


def add_hidden_ncc_gr_match_features(
    frame: pd.DataFrame,
    *,
    horizontal_path: Path,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    ncc_columns = [
        "ncc_sc_ens_delta",
        "ncc_score_mean",
        "ncc_score_std",
        "ncc_score_gap_best_second",
        "ncc_best_scale",
        "ncc_sc_trust",
        "ncc_sc_range",
        "ncc_sc_ens_vs_public_beam",
        "abs_ncc_sc_ens_vs_public_beam",
        "ncc_sc_ens_vs_pf",
        "abs_ncc_sc_ens_vs_pf",
        "gr_match_prefix_rmse",
        "gr_vs_tw_anchor",
        "gr_vs_sc_ens",
        "gr_vs_public_beam",
        "gr_vs_pf",
    ]
    for column in ncc_columns:
        frame[column] = 0.0

    cutoff_row = int(frame["cutoff_row"].iloc[0])
    eval_indices, generated, summary = compute_ncc_gr_match_group(
        horizontal_path=horizontal_path,
        cutoff_row=cutoff_row,
    )
    if not generated:
        return frame

    by_row = pd.DataFrame(
        {
            key: value
            for key, value in generated.items()
            if key not in {"typewell_tvt", "typewell_gr"}
        },
        index=eval_indices,
    )
    aligned = by_row.reindex(frame["row_idx"].to_numpy(dtype=int))
    if aligned.isna().any().any():
        raise ValueError(f"missing hidden NCC rows for {horizontal_path.name}")

    last_anchor = frame["last_anchor_tvt"].to_numpy(dtype=float)
    public_beam = frame["beam_pred"].to_numpy(dtype=float)
    public_pf = frame["pf_pred"].to_numpy(dtype=float)
    sc8 = aligned["sc8"].to_numpy(dtype=float)
    sc15 = aligned["sc15"].to_numpy(dtype=float)
    sc25 = aligned["sc25"].to_numpy(dtype=float)
    sc_ens = aligned["sc_ens"].to_numpy(dtype=float)
    score_matrix = aligned[["sc8_score", "sc15_score", "sc25_score"]].to_numpy(dtype=float)
    sorted_scores = np.sort(score_matrix, axis=1)
    score_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
    path_matrix = np.column_stack([sc8, sc15, sc25])
    eval_gr = aligned["eval_gr"].to_numpy(dtype=float)
    typewell_tvt = generated["typewell_tvt"]
    typewell_gr = generated["typewell_gr"]
    scales = np.asarray([8.0, 15.0, 25.0], dtype=float)

    frame["ncc_sc_ens_delta"] = sc_ens - last_anchor
    frame["ncc_score_mean"] = score_matrix.mean(axis=1)
    frame["ncc_score_std"] = score_matrix.std(axis=1)
    frame["ncc_score_gap_best_second"] = score_gap
    frame["ncc_best_scale"] = scales[np.argmax(score_matrix, axis=1)]
    frame["ncc_sc_trust"] = float(np.clip(float(summary.get("known_len", 0.0)) / 200.0, 0.0, 0.6))
    frame["ncc_sc_range"] = path_matrix.max(axis=1) - path_matrix.min(axis=1)
    frame["ncc_sc_ens_vs_public_beam"] = sc_ens - public_beam
    frame["abs_ncc_sc_ens_vs_public_beam"] = np.abs(sc_ens - public_beam)
    frame["ncc_sc_ens_vs_pf"] = sc_ens - public_pf
    frame["abs_ncc_sc_ens_vs_pf"] = np.abs(sc_ens - public_pf)
    frame["gr_match_prefix_rmse"] = float(summary["prefix_rmse"])

    def gr_residual(tvt_values: np.ndarray | float) -> np.ndarray:
        return eval_gr - np.interp(tvt_values, typewell_tvt, typewell_gr).astype(float)

    frame["gr_vs_tw_anchor"] = gr_residual(last_anchor)
    frame["gr_vs_sc_ens"] = gr_residual(sc_ens)
    frame["gr_vs_public_beam"] = gr_residual(public_beam)
    frame["gr_vs_pf"] = gr_residual(public_pf)

    for column in ncc_columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            frame[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return frame


def selected_inference_spec(config: dict[str, Any]) -> VariantSpec:
    selected_name = str(
        get_nested(
            config,
            "inference.selected_variant",
            "lgbm_capacity_public_core_spatial_multicutoff",
        )
    )
    for spec in feature_variants(config):
        if spec.name == selected_name:
            return spec
    raise ValueError(f"selected inference variant not found: {selected_name}")


def fit_final_candidate_model(
    paths: ExperimentPaths,
    config: dict[str, Any],
    *,
    max_wells: int | None = None,
) -> tuple[Any, VariantSpec, pd.DataFrame, dict[str, Any]]:
    feature_path = resolve_feature_path(paths, get_nested(config, "data.feature_path"))
    allowed_wells = None
    if max_wells is not None:
        allowed_wells = {path.stem.split("__")[0] for path in train_files(paths, max_wells)}
    frame, loaded_columns = load_features(feature_path, config, allowed_wells=allowed_wells)
    frame = add_ncc_gr_match_features(
        frame,
        config=config,
        paths=paths,
        max_wells=max_wells,
    )
    spec = selected_inference_spec(config)
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    bucket_code_values = bucket_codes(frame["eval_step"].to_numpy(dtype=float), buckets)
    train_idx, policy_summary = choose_train_indices_for_variant(
        frame=frame,
        train_mask=np.ones(len(frame), dtype=bool),
        bucket_code_values=bucket_code_values,
        spec=spec,
        config=config,
        max_train_rows_override=None,
        seed=int(get_nested(config, "model.training.seed", 42)) + stable_fold(spec.name, 997),
    )
    x_train = transformed_features(frame, spec.feature_columns).iloc[train_idx]
    y_train = (
        frame["target_tvt"].to_numpy(dtype=float) - frame["last_anchor_tvt"].to_numpy(dtype=float)
    )[train_idx]
    model = make_estimator(config, spec, seed=int(get_nested(config, "model.training.seed", 42)))
    model.fit(x_train, y_train)
    info = {
        "feature_path": str(feature_path),
        "loaded_columns": loaded_columns,
        "artifact_rows": int(len(frame)),
        "train_rows": int(len(train_idx)),
        "feature_count": int(len(spec.feature_columns)),
        "variant": spec.name,
        "candidate": f"{spec.name}_raw",
        "training_policy": policy_summary,
    }
    return model, spec, pd.DataFrame(), info

def predict_hidden_frame(
    *,
    frame: pd.DataFrame,
    path: Path,
    model: Any,
    spec: VariantSpec,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    frame = add_hidden_ncc_gr_match_features(frame, horizontal_path=path)
    missing_features = sorted(set(spec.feature_columns) - set(frame.columns))
    if missing_features:
        raise ValueError(f"inference feature frame is missing columns: {missing_features}")
    residual = model.predict(transformed_features(frame, spec.feature_columns)).astype(float)
    pred = apply_prediction_policy(
        residual=residual,
        last_anchor=frame["last_anchor_tvt"].to_numpy(dtype=float),
        config=config,
    )
    return pred, residual


def generate_public_feature_submission(
    paths: ExperimentPaths,
    config: dict[str, Any],
    *,
    max_wells: int | None = None,
) -> dict[str, Any]:
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()
    model, spec, foldout_source_rows, model_info = fit_final_candidate_model(
        paths,
        config,
        max_wells=max_wells,
    )

    sample = pd.read_csv(paths.sample_submission_path)
    id_column = str(get_nested(config, "data.id_column", "id"))
    target_column = str(get_nested(config, "data.submission_target_column", "tvt"))
    sample["well"] = sample[id_column].astype(str).str[:8]
    sample["row_idx"] = sample[id_column].astype(str).str[9:].astype(int)
    train_path_by_well = {path.stem.split("__")[0]: path for path in train_files(paths, None)}
    test_path_by_well = {path.stem.split("__")[0]: path for path in test_files(paths)}

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    for well_id in sorted(test_path_by_well):
        print(f"Processing {well_id}...")
        hw_te = pd.read_csv(test_path_by_well[well_id])
        tw_te = pd.read_csv(
            paths.test_data_dir / f"{well_id}__typewell.csv",
        )
        tvt_phys = None
        hw_tr = None
        tw_tr = None
        if well_id in train_path_by_well:
            try:
                hw_tr = pd.read_csv(train_path_by_well[well_id])
                tw_tr = pd.read_csv(paths.train_data_dir / f"{well_id}__typewell.csv")
                hw_te["TVT_input"] = hw_tr["TVT_input"].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
                print("  Physical branch OK")
            except Exception as exc:
                print(f"  Physical branch failed: {exc}")
                tvt_phys = None

        selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(
            hw_te
        )
        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            pf_result = run_pf_lik_ensemble_scales_diag(hw_te, tw_ref)
            pf_by_scale = pf_result["predictions_by_scale"]
            tvt_pf = pf_by_scale["pf_scale_8"]
            print(f"  PF {PF_N_SEEDS}-seed lik-ensemble OK")
        except Exception as exc:
            print(f"  PF failed: {exc}")
            last_known = hw_te["TVT_input"].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            tvt_pf = hw_te["TVT_input"].fillna(last_val).values.astype(float)
            pf_result = fallback_pf_result(hw_te, tvt_pf)
            pf_by_scale = pf_result["predictions_by_scale"]

        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            beam_result = run_beam_ensemble_diag(hw_te, tw_ref)
            tvt_beam = beam_result["mean"]
            print("  Beam 14-config ensemble OK")
        except Exception as exc:
            print(f"  Beam failed: {exc}")
            tvt_beam = tvt_pf.copy()
            beam_result = {
                "mean": tvt_beam,
                "std": np.zeros_like(tvt_beam),
                "final_cost_mean": 0.0,
                "final_cost_min": 0.0,
            }

        last_known = hw_te["TVT_input"].dropna()
        last_known_tvt = (
            float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
        )
        tvt_selector = apply_selector_variant(
            selector_variant,
            pf_by_scale,
            tvt_beam,
            last_known_tvt,
        )
        pred_by_row: dict[int, float] = {}
        residual_by_row: dict[int, float] = {}
        if tvt_phys is None:
            feature_frame = build_hidden_feature_frame(
                well_id=well_id,
                hw=hw_te,
                pf_result=pf_result,
                beam_result=beam_result,
                tvt_selector=tvt_selector,
                selector_variant=selector_variant,
                last_known_tvt=last_known_tvt,
            )
            pred, residual = predict_hidden_frame(
                frame=feature_frame,
                path=test_path_by_well[well_id],
                model=model,
                spec=spec,
                config=config,
            )
            row_indices = feature_frame["row_idx"].to_numpy(dtype=int)
            pred_by_row = {
                int(idx): float(value) for idx, value in zip(row_indices, pred, strict=True)
            }
            residual_by_row = {
                int(idx): float(value) for idx, value in zip(row_indices, residual, strict=True)
            }
            print(f"  Public-feature LGBM correction OK rows={len(feature_frame)}")

        well_sample = sample[sample["well"] == well_id]
        for _, row in well_sample.iterrows():
            row_idx = int(row["row_idx"])
            if tvt_phys is not None:
                original_val = float(tvt_phys.iloc[row_idx])
                corrected_val = original_val
                source = "physical_visible"
                residual_pred = 0.0
            else:
                original_val = float(tvt_selector[row_idx])
                corrected_val = float(pred_by_row.get(row_idx, original_val))
                source = "lgbm_capacity_public_core_spatial_multicutoff_raw_hidden"
                residual_pred = float(residual_by_row.get(row_idx, 0.0))
            rows.append({id_column: row[id_column], target_column: corrected_val})
            audit_row = {
                "id": row[id_column],
                "well": well_id,
                "row_idx": row_idx,
                "source": source,
                "selector_variant": selector_variant,
                "selector_code": selector_code,
                "selector_n_eval": selector_n_eval,
                "selector_z_span": selector_z_span,
                "last_known_tvt": last_known_tvt,
                "original_tvt": original_val,
                "corrected_tvt": corrected_val,
                "predicted_lgbm_residual": residual_pred,
                "diff": corrected_val - original_val,
            }
            audit_rows.append(audit_row)
        well_rows.append(
            {
                "well": well_id,
                "rows": int(len(well_sample)),
                "branch": "physical_visible" if tvt_phys is not None else "model_hidden",
                "selector_variant": selector_variant,
                "selector_code": selector_code,
                "selector_n_eval": selector_n_eval,
                "selector_z_span": selector_z_span,
            }
        )
        print(f"  Added {len(well_sample)} rows")

    submission = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if len(submission) != len(sample):
        raise ValueError(f"submission row mismatch: got {len(submission)} expected {len(sample)}")
    missing_ids = sorted(set(sample[id_column]) - set(submission[id_column]))
    if missing_ids:
        raise ValueError(f"missing submission ids: {missing_ids[:5]}")
    submission = sample[[id_column]].merge(submission, on=id_column, how="left")
    if not np.isfinite(submission[target_column].to_numpy(dtype=float)).all():
        raise ValueError("submission contains non-finite predictions")

    anchor_submission = audit[["id", "original_tvt"]].rename(
        columns={"original_tvt": target_column}
    )
    changed = audit[audit["diff"].abs() > 1e-12].copy()
    summary = {
        "experiment": str(get_nested(config, "experiment.name")),
        "status": "inference_completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "base": "exp027_public_replay_needless090_sel15_spread3_visible_branch",
        "ml_route_anchor": "exp054_pseudo_tail_seed_bagging_inference_submit",
        "candidate": f"{spec.name}_raw_hidden",
        "rows": int(len(submission)),
        "changed_rows": int(len(changed)),
        "changed_wells": int(changed["well"].nunique()) if len(changed) else 0,
        "diff_min": float(audit["diff"].min()) if len(audit) else 0.0,
        "diff_max": float(audit["diff"].max()) if len(audit) else 0.0,
        "diff_mean": float(audit["diff"].mean()) if len(audit) else 0.0,
        "diff_abs_mean": float(audit["diff"].abs().mean()) if len(audit) else 0.0,
        "diff_rmse": float(np.sqrt(np.mean(np.square(audit["diff"])))) if len(audit) else 0.0,
        "original_min": (
            float(anchor_submission[target_column].min()) if len(anchor_submission) else 0.0
        ),
        "original_max": (
            float(anchor_submission[target_column].max()) if len(anchor_submission) else 0.0
        ),
        "corrected_min": float(submission[target_column].min()) if len(submission) else 0.0,
        "corrected_max": float(submission[target_column].max()) if len(submission) else 0.0,
        "pf_n_particles": PF_N_PARTICLES,
        "pf_n_seeds": PF_N_SEEDS,
        "final_model": model_info,
    }

    submission.to_csv(paths.submission_path, index=False)
    anchor_submission.to_csv(
        paths.output_root / "public_feature_original_selector_submission.csv",
        index=False,
    )
    audit.to_csv(paths.output_root / "lgbm_public_feature_corrected_diff.csv", index=False)
    pd.DataFrame(well_rows).to_csv(
        paths.artifacts_dir / "public_feature_inference_wells.csv",
        index=False,
    )
    foldout_source_rows.to_csv(
        paths.artifacts_dir / "public_feature_inference_source_summary.csv",
        index=False,
    )
    (paths.output_root / "lgbm_public_feature_corrected_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (paths.artifacts_dir / "public_feature_inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
