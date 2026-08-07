from __future__ import annotations

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
from settings import ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp186_typewell_late_range_pfbeam_generation_soft_prior"
EXPERIMENT_NAME = "exp186_typewell_late_range_pfbeam_generation_soft_prior"


@dataclass(frozen=True)
class SoftPriorSpec:
    name: str
    weak_pct: float
    strong_pct: float
    weak_penalty: float
    strong_penalty: float
    known_last_pct_threshold: float
    known_last_multiplier: float


@dataclass(frozen=True)
class PrefixHoldout:
    well: str
    masked: pd.DataFrame
    typewell: pd.DataFrame
    eval_index: np.ndarray
    true_tvt: np.ndarray
    last_known_tvt: float
    last_known_md: float
    known_last_pct: float
    typewell_min: float
    typewell_span: float
    status: dict[str, Any]


@dataclass(frozen=True)
class PfRun:
    preds: np.ndarray
    log_likelihoods: np.ndarray
    ess_mean_by_row: np.ndarray
    resampled_by_row: np.ndarray
    seed_weights: np.ndarray


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
        return float(value) if np.isfinite(float(value)) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    key = "::".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def fill_numeric(values: pd.Series | np.ndarray, fallback: float) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def smooth(values: np.ndarray, window: int, fallback: float) -> np.ndarray:
    if window <= 1:
        return fill_numeric(values, fallback)
    return (
        pd.Series(values, dtype="float64")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(fallback)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(float(values[idx - 1]) - target) <= abs(float(values[idx]) - target):
        return idx - 1
    return idx


def interpolate_typewell_gr(tvt: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> np.ndarray:
    return np.interp(tvt, tw_tvt, tw_gr).astype(np.float32)


def parse_soft_prior_specs(config: dict[str, Any]) -> list[SoftPriorSpec]:
    specs: list[SoftPriorSpec] = []
    for raw in get_nested(config, "model.soft_prior_variants") or []:
        specs.append(
            SoftPriorSpec(
                name=str(raw["name"]),
                weak_pct=float(raw.get("weak_pct", 0.70)),
                strong_pct=float(raw.get("strong_pct", 0.50)),
                weak_penalty=float(raw.get("weak_penalty", 0.0)),
                strong_penalty=float(raw.get("strong_penalty", 0.0)),
                known_last_pct_threshold=float(raw.get("known_last_pct_threshold", 0.75)),
                known_last_multiplier=float(raw.get("known_last_multiplier", 0.0)),
            )
        )
    if not specs:
        raise ValueError("model.soft_prior_variants must define at least one variant")
    if specs[0].name != "no_prior":
        raise ValueError("first soft prior variant must be no_prior for baseline comparisons")
    return specs


def typewell_pct(
    tvt: np.ndarray | float,
    typewell_min: np.ndarray | float,
    typewell_span: np.ndarray | float,
) -> np.ndarray:
    span = np.maximum(np.asarray(typewell_span, dtype=np.float64), 1e-6)
    return (np.asarray(tvt, dtype=np.float64) - np.asarray(typewell_min, dtype=np.float64)) / span


def soft_prior_penalty(
    tvt: np.ndarray,
    *,
    spec: SoftPriorSpec,
    typewell_min: float,
    typewell_span: float,
    known_last_pct: float,
) -> np.ndarray:
    if spec.weak_penalty == 0.0 and spec.strong_penalty == 0.0:
        return np.zeros_like(tvt, dtype=np.float64)
    pct = typewell_pct(tvt, typewell_min, typewell_span)
    weak_width = max(spec.weak_pct - spec.strong_pct, 1e-6)
    weak = np.clip((spec.weak_pct - pct) / weak_width, 0.0, 1.0) * spec.weak_penalty
    strong = np.maximum(spec.strong_pct - pct, 0.0) / max(spec.strong_pct, 1e-6)
    strong = strong * spec.strong_penalty
    if spec.known_last_pct_threshold < 1.0:
        late = np.clip(
            (known_last_pct - spec.known_last_pct_threshold)
            / max(1.0 - spec.known_last_pct_threshold, 1e-6),
            0.0,
            1.0,
        )
    else:
        late = 0.0
    return (weak + strong) * (1.0 + spec.known_last_multiplier * late)


def gr_sigma(
    prefix: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> float:
    runtime = get_nested(config, "model.runtime") or {}
    finite = prefix["TVT_input"].notna() & prefix["GR"].notna()
    if int(finite.sum()) < 20:
        return float(runtime.get("gr_sigma_default", 30.0))
    residual = pd.to_numeric(prefix.loc[finite, "GR"], errors="coerce").to_numpy(
        np.float64
    ) - np.interp(
        pd.to_numeric(prefix.loc[finite, "TVT_input"], errors="coerce").to_numpy(np.float64),
        tw_tvt,
        tw_gr,
    )
    return float(
        np.clip(
            np.nanstd(residual),
            float(runtime.get("gr_sigma_min", 10.0)),
            float(runtime.get("gr_sigma_max", 60.0)),
        )
    )


def initial_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(30)
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dm = np.diff(md)
    dt = np.diff(tvt)
    finite = np.isfinite(dm) & np.isfinite(dt) & (dm > 0.0)
    if int(finite.sum()) < 3:
        return 0.0
    return float(np.median(dt[finite] / dm[finite]))


def systematic_resample(
    rng: np.random.Generator,
    pos: np.ndarray,
    vel: np.ndarray,
    weights: np.ndarray,
    pos_noise: float,
    vel_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pos)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    positions = rng.uniform(0.0, 1.0 / n) + np.arange(n, dtype=np.float64) / n
    idx = np.searchsorted(cdf, positions, side="left")
    return (
        pos[idx] + pos_noise * rng.standard_normal(n),
        vel[idx] + vel_noise * rng.standard_normal(n),
    )


def run_pf_for_holdout(
    holdout: PrefixHoldout,
    spec: SoftPriorSpec,
    config: dict[str, Any],
) -> PfRun:
    runtime = get_nested(config, "model.runtime") or {}
    n_particles = int(runtime.get("particles", 260))
    seed_count = int(runtime.get("seed_count", 10))
    temperature = float(runtime.get("likelihood_temperature", 6.0))
    resample_threshold = float(runtime.get("resample_threshold", 0.5))
    init_spread = float(runtime.get("init_spread", 4.5))
    velocity_noise = float(runtime.get("velocity_noise", 0.002))
    position_noise = float(runtime.get("position_noise", 0.005))
    resample_pos_noise = float(runtime.get("resample_pos_noise", 0.10))
    resample_velocity_noise = float(runtime.get("resample_velocity_noise", 0.001))

    hw = holdout.masked
    tw = holdout.typewell.sort_values("TVT")
    tw_tvt = numeric_array(tw, "TVT").astype(np.float64)
    tw_gr = fill_numeric(tw["GR"], float(np.nanmean(numeric_array(tw, "GR")))).astype(np.float64)
    eval_rows = hw.loc[holdout.eval_index].copy()
    md = numeric_array(eval_rows, "MD").astype(np.float64)
    gr = fill_numeric(eval_rows["GR"], float(np.nanmean(tw_gr))).astype(np.float64)
    prefix = hw.loc[: int(holdout.eval_index[0]) - 1]
    sigma = gr_sigma(prefix, tw_tvt, tw_gr, config)
    init_vel = initial_velocity(prefix)
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())

    preds = np.empty((seed_count, len(eval_rows)), dtype=np.float32)
    log_likelihoods = np.empty(seed_count, dtype=np.float64)
    ess_accum = np.zeros(len(eval_rows), dtype=np.float64)
    resampled_accum = np.zeros(len(eval_rows), dtype=np.float64)

    for seed_index in range(seed_count):
        rng = np.random.default_rng(stable_seed(EXPERIMENT_NAME, holdout.well, "pf", seed_index))
        pos = holdout.last_known_tvt + init_spread * rng.standard_normal(n_particles)
        vel = init_vel + 0.01 * rng.standard_normal(n_particles)
        weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
        prev_md = holdout.last_known_md
        log_lik = 0.0
        for row_pos, (row_md, row_gr) in enumerate(zip(md, gr, strict=True)):
            delta_md = max(float(row_md - prev_md), 1.0)
            vel = 0.998 * vel + velocity_noise * rng.standard_normal(n_particles)
            pos = pos + vel * delta_md + position_noise * rng.standard_normal(n_particles)
            pos = np.clip(pos, tmin - 100.0, tmax + 100.0)

            expected_gr = np.interp(pos, tw_tvt, tw_gr)
            residual = (float(row_gr) - expected_gr) / max(sigma, 1e-6)
            residual2 = np.minimum(residual * residual, 600.0)
            penalty = soft_prior_penalty(
                pos,
                spec=spec,
                typewell_min=holdout.typewell_min,
                typewell_span=holdout.typewell_span,
                known_last_pct=holdout.known_last_pct,
            )
            likelihood = np.exp(-0.5 * residual2 - penalty)
            likelihood = np.maximum(likelihood, 1e-300)
            avg_likelihood = float(np.dot(weights, likelihood))
            log_lik += float(np.log(max(avg_likelihood, 1e-300)))

            weights = weights * likelihood
            weight_sum = float(weights.sum())
            if weight_sum > 0.0 and np.isfinite(weight_sum):
                weights = weights / weight_sum
            else:
                weights.fill(1.0 / n_particles)

            ess = 1.0 / max(float(np.dot(weights, weights)), 1e-300)
            ess_accum[row_pos] += ess
            if ess < resample_threshold * n_particles:
                pos, vel = systematic_resample(
                    rng,
                    pos,
                    vel,
                    weights,
                    resample_pos_noise,
                    resample_velocity_noise,
                )
                weights.fill(1.0 / n_particles)
                resampled_accum[row_pos] += 1.0

            preds[seed_index, row_pos] = np.float32(np.dot(weights, pos))
            prev_md = float(row_md)
        log_likelihoods[seed_index] = log_lik

    centered = log_likelihoods - float(np.max(log_likelihoods))
    seed_weights = np.exp(centered / max(temperature, 1e-6))
    seed_weights = seed_weights / max(float(seed_weights.sum()), 1e-300)
    return PfRun(
        preds=preds,
        log_likelihoods=log_likelihoods,
        ess_mean_by_row=(ess_accum / seed_count).astype(np.float32),
        resampled_by_row=(resampled_accum / seed_count).astype(np.float32),
        seed_weights=seed_weights.astype(np.float32),
    )


def beam_search_for_holdout(
    holdout: PrefixHoldout,
    spec: SoftPriorSpec,
    config: dict[str, Any],
) -> np.ndarray:
    beam_cfg = get_nested(config, "model.beam") or {}
    beam_size = int(beam_cfg.get("beam_size", 14))
    move_radius = int(beam_cfg.get("move_radius", 2))
    move_cost = float(beam_cfg.get("move_cost", 16.0))
    error_scale = float(beam_cfg.get("error_scale", 120.0))
    smooth_window = int(beam_cfg.get("smooth_window", 5))

    hw = holdout.masked
    tw = holdout.typewell.sort_values("TVT")
    tw_tvt = numeric_array(tw, "TVT").astype(np.float64)
    tw_gr = fill_numeric(tw["GR"], float(np.nanmean(numeric_array(tw, "GR")))).astype(np.float64)
    tw_gr = smooth(tw_gr, smooth_window, float(np.nanmean(tw_gr))).astype(np.float64)
    eval_rows = hw.loc[holdout.eval_index].copy()
    gr = smooth(numeric_array(eval_rows, "GR"), smooth_window, float(np.nanmean(tw_gr))).astype(
        np.float64
    )

    start_idx = nearest_index(tw_tvt, holdout.last_known_tvt)
    active: dict[int, tuple[float, list[int]]] = {start_idx: (0.0, [])}
    for row_gr in gr:
        candidates: dict[int, tuple[float, list[int]]] = {}
        for idx, (cost, path) in active.items():
            for delta in range(-move_radius, move_radius + 1):
                next_idx = int(np.clip(idx + delta, 0, len(tw_tvt) - 1))
                gr_cost = ((float(row_gr) - float(tw_gr[next_idx])) ** 2) / max(
                    error_scale,
                    1e-6,
                )
                prior_cost = float(
                    soft_prior_penalty(
                        np.asarray([tw_tvt[next_idx]], dtype=np.float64),
                        spec=spec,
                        typewell_min=holdout.typewell_min,
                        typewell_span=holdout.typewell_span,
                        known_last_pct=holdout.known_last_pct,
                    )[0]
                )
                total = cost + gr_cost + move_cost * abs(delta) + prior_cost
                previous = candidates.get(next_idx)
                if previous is None or total < previous[0]:
                    candidates[next_idx] = (total, [*path, next_idx])
        kept = sorted(candidates.items(), key=lambda item: item[1][0])[:beam_size]
        active = {idx: value for idx, value in kept}
    if not active:
        return np.full(len(eval_rows), holdout.last_known_tvt, dtype=np.float32)
    _, (_, best_path) = min(active.items(), key=lambda item: item[1][0])
    return tw_tvt[np.asarray(best_path, dtype=np.int64)].astype(np.float32)


def list_wells(train_dir: Path, config: dict[str, Any]) -> list[str]:
    holdout_cfg = get_nested(config, "model.prefix_holdout") or {}
    include = [str(value) for value in holdout_cfg.get("well_include", []) if value]
    if include:
        wells = include
    else:
        wells = sorted(
            path.name.removesuffix("__horizontal_well.csv")
            for path in train_dir.glob("*__horizontal_well.csv")
        )
    max_wells = holdout_cfg.get("max_wells")
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    if not wells:
        raise FileNotFoundError(f"No horizontal well files found under {train_dir}")
    return wells


def build_holdout_for_well(
    well: str,
    train_dir: Path,
    config: dict[str, Any],
) -> PrefixHoldout | None:
    holdout_cfg = get_nested(config, "model.prefix_holdout") or {}
    min_known = int(holdout_cfg.get("min_known_prefix_rows", 160))
    holdout_rows = int(holdout_cfg.get("holdout_rows", 192))
    min_eval = int(holdout_cfg.get("min_eval_rows", 64))
    max_eval = int(holdout_cfg.get("max_eval_rows_per_well", holdout_rows))

    hw_path = train_dir / f"{well}__horizontal_well.csv"
    tw_path = train_dir / f"{well}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        return None
    horizontal = pd.read_csv(hw_path, low_memory=False)
    typewell = pd.read_csv(tw_path, low_memory=False).sort_values("TVT").reset_index(drop=True)
    if len(typewell) < 3:
        return None

    known_mask = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    known_idx = np.flatnonzero(known_mask)
    if len(known_idx) < min_known + min_eval:
        return None

    eval_count = min(holdout_rows, len(known_idx) - min_known, max_eval)
    if eval_count < min_eval:
        return None
    eval_index = known_idx[-eval_count:].astype(np.int64)
    prefix_end = int(eval_index[0])
    last_known_idx = int(known_idx[np.searchsorted(known_idx, prefix_end) - 1])
    masked = horizontal.iloc[: int(eval_index[-1]) + 1].copy()
    masked.loc[eval_index, "TVT_input"] = np.nan

    tw_tvt = numeric_array(typewell, "TVT").astype(np.float64)
    typewell_min = float(np.nanmin(tw_tvt))
    typewell_max = float(np.nanmax(tw_tvt))
    typewell_span = max(typewell_max - typewell_min, 1e-6)
    last_known_tvt = float(horizontal.loc[last_known_idx, "TVT_input"])
    last_known_md = float(horizontal.loc[last_known_idx, "MD"])
    known_last_pct = float((last_known_tvt - typewell_min) / typewell_span)
    truth = pd.to_numeric(horizontal.loc[eval_index, "TVT_input"], errors="coerce").to_numpy(
        np.float32
    )
    status = {
        "well": well,
        "status": "ok",
        "known_rows": int(len(known_idx)),
        "eval_rows": int(len(eval_index)),
        "last_known_idx": int(last_known_idx),
        "last_known_tvt": last_known_tvt,
        "known_last_pct": known_last_pct,
        "typewell_min": typewell_min,
        "typewell_max": typewell_max,
    }
    return PrefixHoldout(
        well=well,
        masked=masked,
        typewell=typewell,
        eval_index=eval_index,
        true_tvt=truth,
        last_known_tvt=last_known_tvt,
        last_known_md=last_known_md,
        known_last_pct=known_last_pct,
        typewell_min=typewell_min,
        typewell_span=typewell_span,
        status=status,
    )


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    return {
        "rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "within10": float(np.mean(np.abs(err) <= 10.0)),
        "bias": float(np.mean(err)),
    }


def distance_bucket(md_since: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(pd.Series(md_since), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def path_jump_rate(pred: np.ndarray, threshold_ft: float) -> float | None:
    finite = np.isfinite(pred)
    if finite.sum() < 2:
        return None
    diffs = np.abs(np.diff(pred[finite].astype(np.float64)))
    return float(np.mean(diffs > threshold_ft))


def candidate_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "well",
        "row_idx",
        "id",
        "true_tvt",
        "last_known_tvt",
        "last_known_md",
        "md_since",
        "known_last_pct",
        "typewell_min",
        "typewell_span",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and not column.endswith("_diag")
    ]


def compute_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_no_prior_lik_mean")
    baseline = score_prediction(numeric_array(frame, primary), true) if primary in frame else None
    threshold = float(get_nested(config, "audit.path_jump_threshold_ft") or 8.0)
    rows: list[dict[str, Any]] = []
    for column in candidate_columns(frame):
        pred = numeric_array(frame, column)
        score = score_prediction(pred, true)
        delta = None
        if baseline and score["rmse"] is not None and baseline["rmse"] is not None:
            delta = float(score["rmse"] - baseline["rmse"])
        rows.append(
            {
                "candidate": column,
                "is_oracle_diagnostic": bool("_oracle" in column),
                **score,
                "delta_rmse_vs_primary_baseline": delta,
                "path_jump_rate": path_jump_rate(pred, threshold),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"], na_position="last")


def compute_bucket_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    buckets = distance_bucket(frame["md_since"])
    rows: list[dict[str, Any]] = []
    for column in candidate_columns(frame):
        pred = numeric_array(frame, column)
        for bucket in pd.Series(buckets).cat.categories:
            mask = np.asarray(buckets == bucket, dtype=bool)
            if not mask.any():
                continue
            rows.append(
                {
                    "candidate": column,
                    "distance_bucket": str(bucket),
                    **score_prediction(pred[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_no_prior_lik_mean")
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        base_rmse = None
        if primary in group:
            base_rmse = score_prediction(numeric_array(group, primary), true)["rmse"]
        for column in candidate_columns(group):
            score = score_prediction(numeric_array(group, column), true)
            delta = None
            if base_rmse is not None and score["rmse"] is not None:
                delta = float(score["rmse"] - base_rmse)
            rows.append(
                {
                    "well": str(well),
                    "candidate": column,
                    **score,
                    "delta_rmse_vs_primary_baseline": delta,
                }
            )
    return pd.DataFrame(rows)


def compute_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": numeric_array(frame, "md_since") <= 50.0,
        "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
        "late_prefix_ge_0p75": numeric_array(frame, "known_last_pct") >= 0.75,
        "candidate_pct_baseline_lt_0p70": typewell_pct(
            numeric_array(frame, "pf_no_prior_lik_mean"),
            numeric_array(frame, "typewell_min"),
            numeric_array(frame, "typewell_span"),
        )
        < 0.70,
    }
    rows: list[dict[str, Any]] = []
    true = numeric_array(frame, "true_tvt")
    for group_name, mask in groups.items():
        if not mask.any():
            continue
        for column in candidate_columns(frame):
            rows.append(
                {
                    "group": group_name,
                    "candidate": column,
                    **score_prediction(numeric_array(frame, column)[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def add_worst_well_regression(candidate_metrics: pd.DataFrame, by_well: pd.DataFrame) -> None:
    if by_well.empty:
        candidate_metrics["max_well_regression_vs_primary"] = np.nan
        return
    max_regression = by_well.groupby("candidate", observed=True)[
        "delta_rmse_vs_primary_baseline"
    ].max()
    candidate_metrics["max_well_regression_vs_primary"] = candidate_metrics["candidate"].map(
        max_regression
    )


def build_row_frame_for_holdout(
    holdout: PrefixHoldout,
    pf_outputs: dict[str, PfRun],
    beam_outputs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eval_rows = holdout.masked.loc[holdout.eval_index]
    row_frame = pd.DataFrame(
        {
            "well": holdout.well,
            "row_idx": holdout.eval_index.astype(np.int32),
            "id": [f"{holdout.well}_{int(idx)}" for idx in holdout.eval_index],
            "true_tvt": holdout.true_tvt.astype(np.float32),
            "last_known_tvt": np.float32(holdout.last_known_tvt),
            "last_known_md": np.float32(holdout.last_known_md),
            "md_since": numeric_array(eval_rows, "MD") - np.float32(holdout.last_known_md),
            "known_last_pct": np.float32(holdout.known_last_pct),
            "typewell_min": np.float32(holdout.typewell_min),
            "typewell_span": np.float32(holdout.typewell_span),
        }
    )
    top_k = int(get_nested(config, "model.runtime.topk_oracle") or 3)
    diagnostics: list[dict[str, Any]] = []
    for variant_name, run in pf_outputs.items():
        weighted = (run.seed_weights[:, None] * run.preds).sum(axis=0)
        best_idx = int(np.argmax(run.log_likelihoods))
        top_idx = np.argsort(run.log_likelihoods)[::-1][:top_k]
        oracle = np.empty(len(row_frame), dtype=np.float32)
        truth = row_frame["true_tvt"].to_numpy(np.float32)
        for i in range(len(row_frame)):
            seed_values = run.preds[top_idx, i]
            oracle[i] = seed_values[np.argmin(np.abs(seed_values - truth[i]))]
        row_frame[f"pf_{variant_name}_lik_mean"] = weighted.astype(np.float32)
        row_frame[f"pf_{variant_name}_best_seed"] = run.preds[best_idx].astype(np.float32)
        row_frame[f"pf_{variant_name}_top{top_k}_oracle"] = oracle
        row_frame[f"pf_{variant_name}_ess_mean_diag"] = run.ess_mean_by_row.astype(np.float32)
        row_frame[f"pf_{variant_name}_resampled_rate_diag"] = run.resampled_by_row.astype(
            np.float32
        )
        diagnostics.append(
            {
                "well": holdout.well,
                "variant": variant_name,
                "seed_count": int(run.preds.shape[0]),
                "rows": int(run.preds.shape[1]),
                "log_likelihood_mean": float(np.mean(run.log_likelihoods)),
                "log_likelihood_std": float(np.std(run.log_likelihoods)),
                "ess_mean": float(np.mean(run.ess_mean_by_row)),
                "resampling_rate": float(np.mean(run.resampled_by_row)),
                "seed_weight_max": float(np.max(run.seed_weights)),
            }
        )
    for variant_name, pred in beam_outputs.items():
        row_frame[f"beam_{variant_name}_top1"] = pred.astype(np.float32)
    return row_frame, diagnostics


def run_soft_prior_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    train_dir = paths.train_data_dir
    specs = parse_soft_prior_specs(config)
    row_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for well in list_wells(train_dir, config):
        holdout = build_holdout_for_well(well, train_dir, config)
        if holdout is None:
            status_rows.append({"well": well, "status": "skipped"})
            continue
        pf_outputs: dict[str, PfRun] = {}
        beam_outputs: dict[str, np.ndarray] = {}
        for spec in specs:
            pf_outputs[spec.name] = run_pf_for_holdout(holdout, spec, config)
            beam_outputs[spec.name] = beam_search_for_holdout(holdout, spec, config)
        frame, diag = build_row_frame_for_holdout(holdout, pf_outputs, beam_outputs, config)
        row_frames.append(frame)
        diagnostics.extend(diag)
        status_rows.append(holdout.status)

    if not row_frames:
        raise RuntimeError("No prefix holdout rows were generated.")

    row_frame = pd.concat(row_frames, ignore_index=True)
    pf_diagnostics = pd.DataFrame(diagnostics)
    well_status = pd.DataFrame(status_rows)
    candidate_metrics = compute_candidate_metrics(row_frame, config)
    bucket_metrics = compute_bucket_metrics(row_frame)
    by_well = compute_by_well(row_frame, config)
    group_metrics = compute_group_metrics(row_frame)
    add_worst_well_regression(candidate_metrics, by_well)

    artifacts = paths.artifacts_dir
    candidate_metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_metrics_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_metrics_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    pf_diagnostics_path = artifacts / f"{OUTPUT_PREFIX}_pf_diagnostics.csv"
    well_status_path = artifacts / f"{OUTPUT_PREFIX}_well_status.csv"
    row_candidates_path = artifacts / f"{OUTPUT_PREFIX}_row_candidates.csv.gz"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_metrics_path, index=False)
    pf_diagnostics.to_csv(pf_diagnostics_path, index=False)
    well_status.to_csv(well_status_path, index=False)
    row_frame.to_csv(row_candidates_path, index=False, compression="gzip")

    best_row = (
        candidate_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    deployable_metrics = candidate_metrics[~candidate_metrics["is_oracle_diagnostic"]]
    best_non_oracle_row = (
        deployable_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit_pending_interpretation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - started),
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "variants": [spec.name for spec in specs],
        "primary_baseline": get_nested(config, "audit.primary_baseline"),
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "pf_diagnostics_summary": (
            pf_diagnostics.groupby("variant", observed=True)
            .agg(
                wells=("well", "nunique"),
                ess_mean=("ess_mean", "mean"),
                resampling_rate=("resampling_rate", "mean"),
                log_likelihood_mean=("log_likelihood_mean", "mean"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "artifacts": {
            "candidate_metrics": str(candidate_metrics_path),
            "bucket_metrics": str(bucket_metrics_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_metrics_path),
            "pf_diagnostics": str(pf_diagnostics_path),
            "well_status": str(well_status_path),
            "row_candidates": str(row_candidates_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit_pending_interpretation",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "artifacts": summary["artifacts"],
        "sha256": {
            "candidate_metrics": sha256_path(candidate_metrics_path),
            "bucket_metrics": sha256_path(bucket_metrics_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_metrics_path),
            "pf_diagnostics": sha256_path(pf_diagnostics_path),
            "well_status": sha256_path(well_status_path),
            "row_candidates_raw_gzip": sha256_path(row_candidates_path),
            "row_candidates_decompressed": sha256_path(row_candidates_path, decompressed=True),
            "summary": sha256_path(summary_path),
        },
        "notes": "Train-side prefix holdout PF/Beam soft-prior audit only; no model or submission.",
    }
    write_json(paths.metrics_path, metrics)
    return {
        "summary": summary,
        "candidate_metrics": candidate_metrics,
        "bucket_metrics": bucket_metrics,
        "by_well": by_well,
        "group_metrics": group_metrics,
        "pf_diagnostics": pf_diagnostics,
        "well_status": well_status,
        "row_frame": row_frame,
    }


def main() -> dict[str, Any]:
    return run_soft_prior_audit()


if __name__ == "__main__":
    result = main()
    print(json.dumps(to_jsonable(result["summary"]), indent=2, sort_keys=True))
