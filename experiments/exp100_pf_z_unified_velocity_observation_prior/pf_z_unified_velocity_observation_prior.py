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
from numba import njit
from settings import ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp100_pf_z_unified_velocity_observation_prior"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_xy_velocity: bool
    use_prefix_velocity: bool
    use_gr_calibration: bool


@dataclass(frozen=True)
class WellFit:
    beta_z: float
    beta_xy: float
    intercept: float
    velocity_sigma: float
    initial_velocity: float
    prefix_velocity_tail: float
    prefix_velocity_all: float
    gr_cal_a: float
    gr_cal_b: float
    gr_cal_sigma: float
    gr_sigma: float
    known_rows: int
    eval_rows: int


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


def _as_float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


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
def _resamp(pos, vel, w, n_particles, rough_pos, rough_vel):
    cum = np.zeros(n_particles + 1)
    for j in range(n_particles):
        cum[j + 1] = cum[j] + w[j]
    out_pos = np.empty(n_particles)
    out_vel = np.empty(n_particles)
    u0 = np.random.uniform(0.0, 1.0 / n_particles)
    cursor = 0
    for j in range(n_particles):
        u = u0 + j / n_particles
        while cursor < n_particles - 1 and cum[cursor + 1] < u:
            cursor += 1
        out_pos[j] = pos[cursor] + rough_pos * np.random.randn()
        out_vel[j] = vel[cursor] + rough_vel * np.random.randn()
    return out_pos, out_vel


@njit(cache=True)
def _pf_z_unified(
    md_v,
    z_v,
    x_v,
    y_v,
    gr_v,
    gr_sm_v,
    gg_raw,
    gg_smooth,
    vmin,
    step,
    gr_sigma,
    initial_pos,
    initial_velocity,
    beta_z,
    beta_xy,
    intercept,
    velocity_sigma,
    prefix_velocity,
    gr_cal_a,
    gr_cal_b,
    gr_cal_sigma,
    use_xy_velocity,
    use_prefix_velocity,
    use_gr_calibration,
    n_particles,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_pos,
    rough_vel,
    resample_threshold,
    velocity_sigma_multiplier,
    xy_velocity_sigma_multiplier,
    prefix_velocity_blend_weight,
    gr_calibration_sigma_multiplier,
):
    pos = np.empty(n_particles)
    vel = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for j in range(n_particles):
        pos[j] = initial_pos + 0.5 * np.random.randn()
        vel[j] = initial_velocity + 0.02 * np.random.randn()

    n_rows = len(md_v)
    pred = np.empty(n_rows)
    std = np.empty(n_rows)
    prev_md = md_v[0] - 1.0
    prev_z = z_v[0] - 1.0
    prev_x = x_v[0]
    prev_y = y_v[0]
    upper = vmin + (len(gg_raw) - 1) * step + 50.0
    lower = vmin - 50.0

    for i in range(n_rows):
        dm = md_v[i] - prev_md
        if dm < 1.0:
            dm = 1.0
        dzd = (z_v[i] - prev_z) / dm
        dxy = ((x_v[i] - prev_x) ** 2 + (y_v[i] - prev_y) ** 2) ** 0.5 / dm
        expected_velocity = beta_z * dzd + intercept
        sigma = velocity_sigma * velocity_sigma_multiplier
        if use_xy_velocity:
            expected_velocity += beta_xy * dxy
            sigma = max(sigma, velocity_sigma * xy_velocity_sigma_multiplier)
        if use_prefix_velocity:
            expected_velocity = (
                (1.0 - prefix_velocity_blend_weight) * expected_velocity
                + prefix_velocity_blend_weight * prefix_velocity
            )
            sigma = max(sigma, abs(prefix_velocity) * 0.10 + 0.01)

        for j in range(n_particles):
            vel[j] = momentum * vel[j] + velocity_noise * np.random.randn()
            pos[j] += vel[j] * dm + position_noise * np.random.randn()
            if pos[j] < lower:
                pos[j] = lower
            if pos[j] > upper:
                pos[j] = upper

        if not np.isnan(gr_v[i]):
            weight_sum = 0.0
            obs_sigma = gr_sigma
            if use_gr_calibration:
                obs_sigma = max(gr_sigma, gr_cal_sigma * gr_calibration_sigma_multiplier)
            for j in range(n_particles):
                expected_raw = _interp1(gg_raw, pos[j], vmin, step)
                if use_gr_calibration:
                    expected_raw = gr_cal_a * expected_raw + gr_cal_b
                raw_delta = (gr_v[i] - expected_raw) / max(obs_sigma, 1e-6)
                if raw_delta * raw_delta < 600.0:
                    raw_like = np.exp(-0.5 * raw_delta * raw_delta)
                else:
                    raw_like = 0.0
                raw_like = max(raw_like, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    expected_smooth = _interp1(gg_smooth, pos[j], vmin, step)
                    if use_gr_calibration:
                        expected_smooth = gr_cal_a * expected_smooth + gr_cal_b
                    smooth_delta = (gr_sm_v[i] - expected_smooth) / max(obs_sigma * 1.5, 1e-6)
                    if smooth_delta * smooth_delta < 600.0:
                        smooth_like = np.exp(-0.5 * smooth_delta * smooth_delta)
                    else:
                        smooth_like = 0.0
                    smooth_like = max(smooth_like, 1e-300)
                    like = (1.0 - gr_smooth_weight) * raw_like + gr_smooth_weight * smooth_like
                else:
                    like = raw_like
                weights[j] *= max(like, 1e-300)
                weight_sum += weights[j]
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles

        velocity_weight_sum = 0.0
        for j in range(n_particles):
            dv = (vel[j] - expected_velocity) / max(sigma, 0.005)
            if dv * dv < 600.0:
                velocity_like = np.exp(-0.5 * dv * dv)
            else:
                velocity_like = 0.0
            weights[j] *= max(velocity_like, 1e-300)
            velocity_weight_sum += weights[j]
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
            pos, vel = _resamp(pos, vel, weights, n_particles, rough_pos, rough_vel)
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles

        mean = 0.0
        for j in range(n_particles):
            mean += weights[j] * pos[j]
        pred[i] = mean
        var = 0.0
        for j in range(n_particles):
            var += weights[j] * (pos[j] - mean) ** 2
        std[i] = var**0.5
        prev_md = md_v[i]
        prev_z = z_v[i]
        prev_x = x_v[i]
        prev_y = y_v[i]
    return pred, std


@njit(cache=True)
def _pf_z_unified_seeded(
    seed,
    md_v,
    z_v,
    x_v,
    y_v,
    gr_v,
    gr_sm_v,
    gg_raw,
    gg_smooth,
    vmin,
    step,
    gr_sigma,
    initial_pos,
    initial_velocity,
    beta_z,
    beta_xy,
    intercept,
    velocity_sigma,
    prefix_velocity,
    gr_cal_a,
    gr_cal_b,
    gr_cal_sigma,
    use_xy_velocity,
    use_prefix_velocity,
    use_gr_calibration,
    n_particles,
    momentum,
    velocity_noise,
    position_noise,
    gr_smooth_weight,
    rough_pos,
    rough_vel,
    resample_threshold,
    velocity_sigma_multiplier,
    xy_velocity_sigma_multiplier,
    prefix_velocity_blend_weight,
    gr_calibration_sigma_multiplier,
):
    np.random.seed(seed)
    return _pf_z_unified(
        md_v,
        z_v,
        x_v,
        y_v,
        gr_v,
        gr_sm_v,
        gg_raw,
        gg_smooth,
        vmin,
        step,
        gr_sigma,
        initial_pos,
        initial_velocity,
        beta_z,
        beta_xy,
        intercept,
        velocity_sigma,
        prefix_velocity,
        gr_cal_a,
        gr_cal_b,
        gr_cal_sigma,
        use_xy_velocity,
        use_prefix_velocity,
        use_gr_calibration,
        n_particles,
        momentum,
        velocity_noise,
        position_noise,
        gr_smooth_weight,
        rough_pos,
        rough_vel,
        resample_threshold,
        velocity_sigma_multiplier,
        xy_velocity_sigma_multiplier,
        prefix_velocity_blend_weight,
        gr_calibration_sigma_multiplier,
    )


def _grid(tvt: np.ndarray, gr: np.ndarray, step: float) -> tuple[np.ndarray, float, float]:
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if int(mask.sum()) < 2:
        raise ValueError("typewell grid needs at least two finite TVT/GR rows")
    tvt = tvt[mask]
    gr = gr[mask]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    tmin = float(tvt.min())
    tmax = float(tvt.max())
    grid_tvt = np.arange(tmin, tmax + step, step)
    grid_gr = np.interp(grid_tvt, tvt, gr).astype(np.float64)
    return grid_gr, tmin, float(step)


def _gr_sigma(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> float:
    known = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(known) < 20:
        return float(config["gr_sigma_default"])
    expected = np.interp(
        known["TVT_input"].to_numpy(np.float64),
        tw_tvt.astype(np.float64),
        tw_gr.astype(np.float64),
    )
    sigma = float(np.nanstd(known["GR"].to_numpy(np.float64) - expected))
    return float(np.clip(sigma, config["gr_sigma_min"], config["gr_sigma_max"]))


def _robust_slope(delta_value: np.ndarray, delta_md: np.ndarray) -> float:
    mask = (delta_md > 0) & np.isfinite(delta_value) & np.isfinite(delta_md)
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.nanmedian(delta_value[mask] / delta_md[mask]))


def _fit_well_priors(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> WellFit:
    known = hw[hw["TVT_input"].notna()].copy()
    eval_rows = int(hw["TVT_input"].isna().sum())
    if len(known) < 5:
        return WellFit(
            beta_z=-1.0,
            beta_xy=0.0,
            intercept=0.0,
            velocity_sigma=0.10,
            initial_velocity=0.0,
            prefix_velocity_tail=0.0,
            prefix_velocity_all=0.0,
            gr_cal_a=1.0,
            gr_cal_b=0.0,
            gr_cal_sigma=float(config["gr_sigma_default"]),
            gr_sigma=float(config["gr_sigma_default"]),
            known_rows=int(len(known)),
            eval_rows=eval_rows,
        )

    md = _as_float_array(known, "MD")
    z = _as_float_array(known, "Z")
    x = _as_float_array(known, "X")
    y = _as_float_array(known, "Y")
    tvt = _as_float_array(known, "TVT_input")
    dmd = np.diff(md)
    dvt = np.diff(tvt)
    dz = np.diff(z)
    dxy = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    valid = (dmd > 0) & np.isfinite(dvt) & np.isfinite(dz) & np.isfinite(dxy)
    if int(valid.sum()) >= 10:
        response = dvt[valid] / dmd[valid]
        z_rate = dz[valid] / dmd[valid]
        xy_rate = dxy[valid] / dmd[valid]
        design = np.column_stack([z_rate, xy_rate, np.ones_like(z_rate)])
        coeff, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        fitted = design @ coeff
        beta_z = float(coeff[0])
        beta_xy = float(coeff[1])
        intercept = float(coeff[2])
        velocity_sigma = max(float(np.nanstd(response - fitted)), 0.001)
    else:
        beta_z = -1.0
        beta_xy = 0.0
        intercept = 0.0
        velocity_sigma = 0.10

    tail_rows = int(config["prefix_tail_rows"])
    tail = known.tail(tail_rows)
    all_velocity = _robust_slope(dvt, dmd)
    tail_velocity = _robust_slope(
        np.diff(_as_float_array(tail, "TVT_input")),
        np.diff(_as_float_array(tail, "MD")),
    )
    init_tail = known.tail(20)
    initial_velocity = _robust_slope(
        np.diff(_as_float_array(init_tail, "TVT_input")),
        np.diff(_as_float_array(init_tail, "MD")),
    )
    gr_sigma = _gr_sigma(hw, tw_tvt, tw_gr, config)
    cal_rows = known[known["GR"].notna()]
    if len(cal_rows) >= 20:
        type_gr = np.interp(
            cal_rows["TVT_input"].to_numpy(np.float64),
            tw_tvt.astype(np.float64),
            tw_gr.astype(np.float64),
        )
        observed = cal_rows["GR"].to_numpy(np.float64)
        mask = np.isfinite(type_gr) & np.isfinite(observed)
        if int(mask.sum()) >= 20:
            design = np.column_stack([type_gr[mask], np.ones(int(mask.sum()))])
            coeff, _, _, _ = np.linalg.lstsq(design, observed[mask], rcond=None)
            cal_a = float(np.clip(coeff[0], 0.20, 3.00))
            cal_b = float(np.clip(coeff[1], -250.0, 250.0))
            resid = observed[mask] - (cal_a * type_gr[mask] + cal_b)
            cal_sigma = float(np.clip(np.nanstd(resid), 5.0, 80.0))
        else:
            cal_a, cal_b, cal_sigma = 1.0, 0.0, gr_sigma
    else:
        cal_a, cal_b, cal_sigma = 1.0, 0.0, gr_sigma

    return WellFit(
        beta_z=beta_z,
        beta_xy=beta_xy,
        intercept=intercept,
        velocity_sigma=velocity_sigma,
        initial_velocity=initial_velocity,
        prefix_velocity_tail=tail_velocity,
        prefix_velocity_all=all_velocity,
        gr_cal_a=cal_a,
        gr_cal_b=cal_b,
        gr_cal_sigma=cal_sigma,
        gr_sigma=gr_sigma,
        known_rows=int(len(known)),
        eval_rows=eval_rows,
    )


def _parse_variants(config: dict[str, Any]) -> list[VariantSpec]:
    variants = []
    for item in config:
        variants.append(
            VariantSpec(
                name=str(item["name"]),
                use_xy_velocity=bool(item.get("use_xy_velocity", False)),
                use_prefix_velocity=bool(item.get("use_prefix_velocity", False)),
                use_gr_calibration=bool(item.get("use_gr_calibration", False)),
            )
        )
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate variant names: {names}")
    if "pf_z_control" not in names:
        raise ValueError("variants must include pf_z_control")
    return variants


def _read_well_ids(train_dir: Path, max_wells: int | None) -> list[str]:
    ids = sorted(path.name.split("__", 1)[0] for path in train_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        ids = ids[: int(max_wells)]
    if not ids:
        raise FileNotFoundError(f"No train horizontal well files found under {train_dir}")
    return ids


def _run_variant_for_well(
    *,
    well: str,
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    fit: WellFit,
    variant: VariantSpec,
    pf_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    eval_frame = hw[hw["TVT_input"].isna()]
    if len(eval_frame) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    tw_tvt = _as_float_array(tw, "TVT")
    tw_gr = _as_float_array(tw, "GR")
    smooth_tw_gr = (
        pd.Series(tw_gr, dtype="float64")
        .rolling(int(pf_config["gr_rolling_window"]), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )
    gg_raw, vmin, step = _grid(tw_tvt, tw_gr, float(pf_config["gr_grid_step"]))
    gg_smooth, _, _ = _grid(tw_tvt, smooth_tw_gr, float(pf_config["gr_grid_step"]))
    gr_smoothed = (
        pd.Series(_as_float_array(hw, "GR"), dtype="float64")
        .rolling(int(pf_config["gr_rolling_window"]), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )
    known = hw[hw["TVT_input"].notna()]
    prefix_velocity = (
        0.70 * fit.prefix_velocity_tail + 0.30 * fit.prefix_velocity_all
    )
    pred, std = _pf_z_unified_seeded(
        int(seed),
        _as_float_array(eval_frame, "MD"),
        _as_float_array(eval_frame, "Z"),
        _as_float_array(eval_frame, "X"),
        _as_float_array(eval_frame, "Y"),
        _as_float_array(eval_frame, "GR"),
        gr_smoothed[eval_frame.index].astype(np.float64),
        gg_raw,
        gg_smooth,
        vmin,
        step,
        float(fit.gr_sigma),
        float(known["TVT_input"].iloc[-1]),
        float(fit.initial_velocity),
        float(fit.beta_z),
        float(fit.beta_xy),
        float(fit.intercept),
        float(fit.velocity_sigma),
        float(prefix_velocity),
        float(fit.gr_cal_a),
        float(fit.gr_cal_b),
        float(fit.gr_cal_sigma),
        bool(variant.use_xy_velocity),
        bool(variant.use_prefix_velocity),
        bool(variant.use_gr_calibration),
        int(pf_config["n_particles"]),
        float(pf_config["momentum"]),
        float(pf_config["velocity_noise"]),
        float(pf_config["position_noise"]),
        float(pf_config["gr_smooth_weight"]),
        float(pf_config["rough_position"]),
        float(pf_config["rough_velocity"]),
        float(pf_config["resample_threshold"]),
        float(pf_config["velocity_sigma_multiplier"]),
        float(pf_config["xy_velocity_sigma_multiplier"]),
        float(pf_config["prefix_velocity_blend_weight"]),
        float(pf_config["gr_calibration_sigma_multiplier"]),
    )
    return pred.astype(np.float32), std.astype(np.float32)


def _metric_row(values: pd.DataFrame, pred_column: str, thresholds: list[float]) -> dict[str, Any]:
    error = values[pred_column].to_numpy(np.float64) - values["target_tvt"].to_numpy(np.float64)
    abs_error = np.abs(error)
    row = {
        "variant": pred_column.replace("_pred_tvt", ""),
        "rows": int(len(values)),
        "wells": int(values["well"].nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "bias": float(np.mean(error)),
        "std": float(values[pred_column.replace("_pred_tvt", "_std_tvt")].mean()),
    }
    for threshold in thresholds:
        key = str(threshold).replace(".", "p")
        row[f"within_{key}ft"] = float(np.mean(abs_error <= threshold))
    return row


def _smoothness_by_well(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    pred_col = f"{variant}_pred_tvt"
    rows: list[dict[str, Any]] = []
    for well, group in frame.sort_values(["well", "eval_rank"]).groupby("well", sort=False):
        pred = group[pred_col].to_numpy(np.float64)
        if len(pred) <= 1:
            mean_abs_step = 0.0
            p95_abs_step = 0.0
            p95_abs_accel = 0.0
            switch_count = 0
        else:
            step = np.diff(pred)
            abs_step = np.abs(step)
            mean_abs_step = float(np.mean(abs_step))
            p95_abs_step = float(np.quantile(abs_step, 0.95))
            accel = np.diff(step)
            p95_abs_accel = float(np.quantile(np.abs(accel), 0.95)) if len(accel) else 0.0
            switch_count = int(np.sum(abs_step > 25.0))
        error = pred - group["target_tvt"].to_numpy(np.float64)
        rows.append(
            {
                "well": str(well),
                "variant": variant,
                "rows": int(len(group)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "mean_abs_step": mean_abs_step,
                "p95_abs_step": p95_abs_step,
                "p95_abs_acceleration": p95_abs_accel,
                "path_switch_count": switch_count,
            }
        )
    return pd.DataFrame(rows)


def _bucket_metrics(
    frame: pd.DataFrame,
    variants: list[VariantSpec],
    thresholds: list[float],
    config: dict[str, Any],
) -> pd.DataFrame:
    bucket_defs = {
        "md_since": config.get("bucket_edges_md_since", [0, 50, 100, 250, 500, 1000]),
        "eval_rank": config.get("bucket_edges_eval_rank", [0, 100, 250, 500, 1000]),
    }
    rows: list[dict[str, Any]] = []
    for bucket_column, edges in bucket_defs.items():
        edges = [float(value) for value in edges]
        bins = [-np.inf, *edges[1:], np.inf]
        labels = []
        lower = edges[0]
        for upper in edges[1:]:
            labels.append(f"{int(lower):04d}_{int(upper):04d}")
            lower = upper
        labels.append(f"{int(edges[-1]):04d}_plus")
        buckets = pd.cut(
            pd.to_numeric(frame[bucket_column], errors="coerce"),
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        for variant in variants:
            pred_col = f"{variant.name}_pred_tvt"
            for bucket, group in frame.groupby(buckets, observed=False):
                if len(group) == 0:
                    continue
                metric = _metric_row(group, pred_col, thresholds)
                metric["bucket_family"] = bucket_column
                metric["bucket"] = str(bucket)
                rows.append(metric)
    return pd.DataFrame(rows)


def _build_candidate_long(
    frame: pd.DataFrame,
    variants: list[VariantSpec],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for variant in variants:
        pred_col = f"{variant.name}_pred_tvt"
        std_col = f"{variant.name}_std_tvt"
        item = frame[
            ["id", "well", "row_idx", "eval_rank", "md_since", "target_tvt"]
        ].copy()
        item["variant"] = variant.name
        item["pred_tvt"] = frame[pred_col].to_numpy(np.float32)
        item["std_tvt"] = frame[std_col].to_numpy(np.float32)
        item["abs_error"] = np.abs(
            item["pred_tvt"].to_numpy(np.float32) - item["target_tvt"].to_numpy(np.float32)
        )
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.time()
    config = config or load_config()
    paths = ExperimentPaths()
    pf_config = get_nested(config, "model.pf_z") or {}
    variants = _parse_variants(get_nested(config, "model.variants") or [])
    audit_config = get_nested(config, "audit") or {}
    thresholds = [float(value) for value in audit_config.get("thresholds_ft", [1, 2, 5, 10])]
    max_wells = audit_config.get("max_wells")
    if max_wells is None:
        max_wells = pf_config.get("max_wells")
    max_wells = int(max_wells) if max_wells is not None else None
    train_dir = paths.train_data_dir
    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    well_ids = _read_well_ids(train_dir, max_wells)
    candidate_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    input_files: dict[str, str] = {}
    seed_base = int(get_nested(config, "reproducibility.seed") or config["validation"]["seed"])
    for well in well_ids:
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        typewell_path = train_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(f"missing typewell file for {well}: {typewell_path}")
        input_files[str(horizontal_path)] = sha256_path(horizontal_path)
        input_files[str(typewell_path)] = sha256_path(typewell_path)
        hw = pd.read_csv(horizontal_path)
        tw = pd.read_csv(typewell_path).sort_values("TVT").reset_index(drop=True)
        eval_mask = hw["TVT_input"].isna()
        if not bool(eval_mask.any()):
            continue
        if "TVT" not in hw.columns:
            raise ValueError(f"{horizontal_path} does not contain train-side true TVT")
        tw_tvt = _as_float_array(tw, "TVT")
        tw_gr = _as_float_array(tw, "GR")
        fit = _fit_well_priors(hw, tw_tvt, tw_gr, pf_config)
        eval_frame = hw.loc[eval_mask].copy()
        known = hw.loc[~eval_mask]
        row_idx = eval_frame.index.to_numpy(np.int32)
        base = pd.DataFrame(
            {
                "id": [f"{well}_{int(idx)}" for idx in row_idx],
                "well": well,
                "row_idx": row_idx,
                "eval_rank": np.arange(len(eval_frame), dtype=np.int32),
                "md": _as_float_array(eval_frame, "MD").astype(np.float32),
                "md_since": (
                    _as_float_array(eval_frame, "MD") - float(known["MD"].iloc[-1])
                ).astype(np.float32),
                "target_tvt": _as_float_array(eval_frame, "TVT").astype(np.float32),
                "last_known_tvt": np.full(
                    len(eval_frame), float(known["TVT_input"].iloc[-1]), dtype=np.float32
                ),
            }
        )
        for variant in variants:
            seed = stable_seed(OUTPUT_PREFIX, seed_base, variant.name, well)
            pred, std = _run_variant_for_well(
                well=well,
                hw=hw,
                tw=tw,
                fit=fit,
                variant=variant,
                pf_config=pf_config,
                seed=seed,
            )
            if len(pred) != len(base):
                raise ValueError(f"variant {variant.name} produced wrong row count for {well}")
            base[f"{variant.name}_pred_tvt"] = pred
            base[f"{variant.name}_std_tvt"] = std
        candidate_frames.append(base)
        fit_rows.append(
            {
                "well": well,
                "known_rows": fit.known_rows,
                "eval_rows": fit.eval_rows,
                "beta_z": fit.beta_z,
                "beta_xy": fit.beta_xy,
                "intercept": fit.intercept,
                "velocity_sigma": fit.velocity_sigma,
                "initial_velocity": fit.initial_velocity,
                "prefix_velocity_tail": fit.prefix_velocity_tail,
                "prefix_velocity_all": fit.prefix_velocity_all,
                "gr_cal_a": fit.gr_cal_a,
                "gr_cal_b": fit.gr_cal_b,
                "gr_cal_sigma": fit.gr_cal_sigma,
                "gr_sigma": fit.gr_sigma,
            }
        )

    if not candidate_frames:
        raise ValueError("No evaluation rows were produced")
    candidate_wide = pd.concat(candidate_frames, ignore_index=True)
    if not np.isfinite(
        candidate_wide.drop(columns=["id", "well"]).to_numpy(np.float64)
    ).all():
        raise ValueError("candidate_wide contains non-finite values")

    variant_metrics = pd.DataFrame(
        [
            _metric_row(candidate_wide, f"{variant.name}_pred_tvt", thresholds)
            for variant in variants
        ]
    ).sort_values("rmse", kind="stable")
    by_well = pd.concat(
        [_smoothness_by_well(candidate_wide, variant.name) for variant in variants],
        ignore_index=True,
    )
    bucket_metrics = _bucket_metrics(candidate_wide, variants, thresholds, audit_config)
    candidate_long = _build_candidate_long(candidate_wide, variants)
    fit_summary = pd.DataFrame(fit_rows)

    variant_metrics_path = output_dir / f"{OUTPUT_PREFIX}_variant_metrics.csv"
    bucket_metrics_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    fit_summary_path = output_dir / f"{OUTPUT_PREFIX}_well_fit_summary.csv"
    candidate_wide_path = output_dir / f"{OUTPUT_PREFIX}_candidate_wide.csv.gz"
    candidate_long_path = output_dir / f"{OUTPUT_PREFIX}_candidate_long.csv.gz"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"

    variant_metrics.to_csv(variant_metrics_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    fit_summary.to_csv(fit_summary_path, index=False)
    candidate_wide.to_csv(candidate_wide_path, index=False, compression="gzip")
    if bool(audit_config.get("save_candidate_long", True)):
        candidate_long.to_csv(candidate_long_path, index=False, compression="gzip")

    best = variant_metrics.iloc[0].to_dict()
    control = variant_metrics[variant_metrics["variant"] == "pf_z_control"].iloc[0].to_dict()
    best_by_well = (
        by_well.sort_values("rmse", ascending=False)
        .groupby("variant", sort=False)
        .head(5)
        .reset_index(drop=True)
    )
    output_files = {
        "variant_metrics": str(variant_metrics_path),
        "bucket_metrics": str(bucket_metrics_path),
        "by_well": str(by_well_path),
        "well_fit_summary": str(fit_summary_path),
        "candidate_wide": str(candidate_wide_path),
    }
    if candidate_long_path.exists():
        output_files["candidate_long"] = str(candidate_long_path)

    file_sha: dict[str, Any] = {}
    for key, file_name in output_files.items():
        path = Path(file_name)
        file_sha[key] = {"sha256": sha256_path(path)}
        if path.suffix == ".gz":
            file_sha[key]["decompressed_sha256"] = sha256_path(path, decompressed=True)

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_train_side_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_sec": float(time.time() - start),
        "rows": int(len(candidate_wide)),
        "wells": int(candidate_wide["well"].nunique()),
        "variants": [variant.name for variant in variants],
        "best_variant": to_jsonable(best),
        "control": to_jsonable(control),
        "rmse_delta_best_minus_control": float(best["rmse"] - control["rmse"]),
        "within10_delta_best_minus_control": float(
            best.get("within_10p0ft", np.nan) - control.get("within_10p0ft", np.nan)
        ),
        "worst_wells_top5_by_variant": to_jsonable(best_by_well.to_dict(orient="records")),
        "input_file_sha256": input_files,
        "output_files": output_files,
        "output_sha256": file_sha,
        "config_subset": {
            "pf_z": pf_config,
            "audit": audit_config,
            "reproducibility": get_nested(config, "reproducibility"),
        },
    }
    summary["output_files"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.max_wells is not None:
        config.setdefault("audit", {})["max_wells"] = int(args.max_wells)
    if args.n_particles is not None:
        config.setdefault("model", {}).setdefault("pf_z", {})["n_particles"] = int(args.n_particles)
    summary = run_audit(config)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
