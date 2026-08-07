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
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp104_pf_z_seedbag_scale_cache"
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
class XyRateFit:
    beta_z: float
    beta_xy: float
    intercept: float
    rate_sigma: float
    initial_rate: float
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
def _pf_z_seedbag_allseeds(
    md_v,
    z_v,
    x_v,
    y_v,
    gr_v,
    gg,
    vmin,
    step,
    gr_sigma,
    last_state,
    last_md,
    last_z,
    last_x,
    last_y,
    initial_rate,
    beta_z,
    beta_xy,
    intercept,
    rate_sigma,
    n_particles,
    n_seeds,
    seed_base,
    momentum,
    velocity_noise,
    position_noise,
    rough_position,
    rough_rate,
    resample_threshold,
    init_spread,
    rate_likelihood_power,
):
    n = len(md_v)
    preds = np.empty((n_seeds, n))
    liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for seed_offset in range(n_seeds):
        np.random.seed(seed_base + seed_offset)
        pos = np.empty(n_particles)
        rate = np.empty(n_particles)
        weights = np.ones(n_particles) / n_particles
        for j in range(n_particles):
            pos[j] = last_state + init_spread * np.random.randn()
            rate[j] = initial_rate + 0.01 * np.random.randn()
        log_lik = 0.0
        prev_md = last_md
        prev_z = last_z
        prev_x = last_x
        prev_y = last_y
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0
            dzd = (z_v[i] - prev_z) / dm
            dxyd = ((x_v[i] - prev_x) ** 2 + (y_v[i] - prev_y) ** 2) ** 0.5 / dm
            expected_rate = beta_z * dzd + beta_xy * dxyd + intercept
            for j in range(n_particles):
                rate[j] = momentum * rate[j] + velocity_noise * np.random.randn()
                pos[j] += rate[j] * dm + position_noise * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.0:
                    tvt_j = vmin - 100.0
                if tvt_j > tmax + 100.0:
                    tvt_j = tmax + 100.0
                pos[j] = tvt_j + z_v[i]

            avg_lik = 0.0
            for j in range(n_particles):
                expected_gr = _interp1(gg, pos[j] - z_v[i], vmin, step)
                gr_delta = (gr_v[i] - expected_gr) / gr_sigma
                gr_dd = gr_delta * gr_delta
                if gr_dd > 600.0:
                    gr_dd = 600.0
                gr_like = np.exp(-0.5 * gr_dd)
                rate_delta = (rate[j] - expected_rate) / rate_sigma
                rate_dd = rate_delta * rate_delta
                if rate_dd > 600.0:
                    rate_dd = 600.0
                rate_like = np.exp(-0.5 * rate_dd)
                combined = gr_like * (rate_like**rate_likelihood_power)
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
                    new_pos[j] = pos[cursor] + rough_position * np.random.randn()
                    new_rate[j] = rate[cursor] + rough_rate * np.random.randn()
                for j in range(n_particles):
                    pos[j] = new_pos[j]
                    rate[j] = new_rate[j]
                    weights[j] = 1.0 / n_particles

            est = 0.0
            for j in range(n_particles):
                est += weights[j] * (pos[j] - z_v[i])
            preds[seed_offset, i] = est
            prev_md = md_v[i]
            prev_z = z_v[i]
            prev_x = x_v[i]
            prev_y = y_v[i]
        liks[seed_offset] = log_lik
    return preds, liks


def _typewell_grid(
    tw: pd.DataFrame,
    step: float,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
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
    grid, vmin, actual_step = _grid_interp(
        tvt.astype(np.float64),
        gr.astype(np.float64),
        float(step),
    )
    return grid.astype(np.float64), float(vmin), float(actual_step), tvt, gr


def _fit_xy_rate_prior(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> XyRateFit:
    known = hw[hw["TVT_input"].notna()].copy()
    eval_rows = int(hw["TVT_input"].isna().sum())
    default_sigma = float(config.get("rate_sigma_default", 0.10))
    if len(known) < 5:
        return XyRateFit(-1.0, 0.0, 0.0, default_sigma, 0.0, 30.0, int(len(known)), eval_rows)

    md = _as_float_array(known, "MD")
    z = _as_float_array(known, "Z")
    x = _as_float_array(known, "X")
    y = _as_float_array(known, "Y")
    tvt = _as_float_array(known, "TVT_input")
    state = tvt + z
    dm = np.diff(md)
    response = np.diff(state)
    dz = np.diff(z)
    dxy = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    valid = (dm > 0) & np.isfinite(response) & np.isfinite(dz) & np.isfinite(dxy)
    if int(valid.sum()) >= int(config.get("min_fit_rows", 10)):
        y_rate = response[valid] / dm[valid]
        z_rate = dz[valid] / dm[valid]
        xy_rate = dxy[valid] / dm[valid]
        design = np.column_stack([z_rate, xy_rate, np.ones_like(z_rate)])
        coeff, _, _, _ = np.linalg.lstsq(design, y_rate, rcond=None)
        fitted = design @ coeff
        beta_z = float(np.clip(coeff[0], -5.0, 5.0))
        beta_xy = float(np.clip(coeff[1], -5.0, 5.0))
        intercept = float(np.clip(coeff[2], -5.0, 5.0))
        rate_sigma = float(np.clip(np.nanstd(y_rate - fitted), 0.005, 1.0))
    else:
        beta_z, beta_xy, intercept, rate_sigma = -1.0, 0.0, 0.0, default_sigma

    tail = known.tail(int(config.get("initial_rate_tail_rows", 30)))
    tail_md = _as_float_array(tail, "MD")
    tail_state = _as_float_array(tail, "TVT_input") + _as_float_array(tail, "Z")
    tail_dm = np.diff(tail_md)
    tail_ds = np.diff(tail_state)
    tail_valid = (tail_dm > 0) & np.isfinite(tail_ds)
    initial_rate = (
        float(np.median(tail_ds[tail_valid] / tail_dm[tail_valid]))
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
        gr_sigma = float(np.clip(np.nanstd(observed - expected), 10.0, 60.0))
    else:
        gr_sigma = float(config.get("gr_sigma_default", 30.0))
    return XyRateFit(
        beta_z=beta_z,
        beta_xy=beta_xy,
        intercept=intercept,
        rate_sigma=max(rate_sigma * float(config.get("rate_sigma_multiplier", 2.5)), 0.005),
        initial_rate=initial_rate,
        gr_sigma=gr_sigma,
        known_rows=int(len(known)),
        eval_rows=eval_rows,
    )


def run_pf_z_seedbag_for_well(
    well: str,
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    row_indices: np.ndarray,
    config: dict[str, Any],
    seed_base: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_frame = hw.iloc[row_indices].copy()
    known = hw[hw["TVT_input"].notna()].copy()
    if known.empty:
        raise ValueError(f"No finite TVT_input prefix rows for well {well}")
    grid, vmin, step, tw_tvt, tw_gr = _typewell_grid(tw, float(config.get("gr_grid_step", 0.1)))
    fit = _fit_xy_rate_prior(hw, tw_tvt, tw_gr, config)
    last = known.iloc[-1]
    gr_series = pd.to_numeric(hw["GR"], errors="coerce")
    gr_fallback = float(gr_series[hw["TVT_input"].notna()].mean())
    if not np.isfinite(gr_fallback):
        gr_fallback = float(np.nanmean(tw_gr))
    gr_all = (
        gr_series.interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(gr_fallback)
        .to_numpy(np.float64)
    )
    n_seeds = int(config.get("n_seeds", 128))
    scales = [float(value) for value in config.get("scales", [3.0, 5.0, 8.0, 12.0])]
    preds, liks = _pf_z_seedbag_allseeds(
        _as_float_array(eval_frame, "MD"),
        _as_float_array(eval_frame, "Z"),
        _as_float_array(eval_frame, "X"),
        _as_float_array(eval_frame, "Y"),
        gr_all[row_indices].astype(np.float64),
        grid,
        vmin,
        step,
        float(fit.gr_sigma),
        float(last["TVT_input"] + last["Z"]),
        float(last["MD"]),
        float(last["Z"]),
        float(last["X"]),
        float(last["Y"]),
        float(fit.initial_rate),
        float(fit.beta_z),
        float(fit.beta_xy),
        float(fit.intercept),
        float(fit.rate_sigma),
        int(config.get("n_particles", 500)),
        n_seeds,
        int(seed_base),
        float(config.get("momentum", 0.998)),
        float(config.get("velocity_noise", 0.002)),
        float(config.get("position_noise", 0.005)),
        float(config.get("rough_position", 0.1)),
        float(config.get("rough_rate", 0.001)),
        float(config.get("resample_threshold", 0.5)),
        float(config.get("init_spread", 4.5)),
        float(config.get("rate_likelihood_power", 1.0)),
    )
    out = pd.DataFrame({"id": [f"{well}_{int(idx)}" for idx in row_indices]})
    out["pf_z_seedbag_mean"] = preds.mean(axis=0).astype(np.float32)
    out["pf_z_seedbag_seed_std"] = preds.std(axis=0).astype(np.float32)
    centered_lik = liks - float(liks.max())
    for scale in scales:
        weights = np.exp(centered_lik / float(scale))
        weights /= weights.sum()
        out[f"pf_z_seedbag_scale_{scale:g}"] = (weights[:, None] * preds).sum(axis=0).astype(
            np.float32
        )
    quality = {
        "well": well,
        "known_rows": fit.known_rows,
        "eval_rows": fit.eval_rows,
        "beta_z": fit.beta_z,
        "beta_xy": fit.beta_xy,
        "intercept": fit.intercept,
        "rate_sigma": fit.rate_sigma,
        "initial_rate": fit.initial_rate,
        "gr_sigma": fit.gr_sigma,
        "seed_base": int(seed_base),
        "lik_best_per_row": float(liks.max() / max(len(row_indices), 1)),
        "lik_std": float(liks.std()),
    }
    return out, quality


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
    for bucket_column, edges in bucket_defs.items():
        edges = [float(value) for value in edges]
        bins = [-np.inf, *edges[1:], np.inf]
        labels = [f"{int(edges[i]):04d}_{int(edges[i + 1]):04d}" for i in range(len(edges) - 1)]
        labels.append(f"{int(edges[-1]):04d}_plus")
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
            item["pred_tvt"].to_numpy(np.float32)
            - item["target_tvt"].to_numpy(np.float32)
        )
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.time()
    config = config or load_config()
    paths = ExperimentPaths()
    audit_config = get_nested(config, "audit") or {}
    model_config = get_nested(config, "model.pf_z_seedbag") or {}
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
    scales = [float(value) for value in model_config.get("scales", [3.0, 5.0, 8.0, 12.0])]
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
    seedbag_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    input_sha: dict[str, str] = {}
    seed_root = int(get_nested(config, "reproducibility.seed") or 42)
    for well, group in row_lookup.groupby("well", sort=True):
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        typewell_path = train_dir / f"{well}__typewell.csv"
        if not horizontal_path.exists() or not typewell_path.exists():
            raise FileNotFoundError(f"missing raw train files for {well}")
        input_sha[str(horizontal_path)] = sha256_path(horizontal_path)
        input_sha[str(typewell_path)] = sha256_path(typewell_path)
        hw = pd.read_csv(horizontal_path)
        tw = pd.read_csv(typewell_path)
        row_indices = group["row_idx"].to_numpy(np.int32)
        seed = stable_seed(OUTPUT_PREFIX, "pf_z_seedbag", seed_root, well)
        seedbag_frame, quality = run_pf_z_seedbag_for_well(
            str(well),
            hw,
            tw,
            row_indices=row_indices,
            config=model_config,
            seed_base=seed,
        )
        seedbag_frames.append(seedbag_frame)
        quality_rows.append(quality)

    seedbag_all = pd.concat(seedbag_frames, ignore_index=True)
    candidate_wide = candidate_wide.merge(
        seedbag_all,
        on="id",
        how="left",
        validate="one_to_one",
    )
    if candidate_wide.isna().any().any():
        missing = candidate_wide.columns[candidate_wide.isna().any()].tolist()
        raise ValueError(f"candidate_wide contains missing values in columns: {missing}")
    seedbag_candidates = [
        "pf_z_seedbag_mean",
        *[f"pf_z_seedbag_scale_{scale:g}" for scale in scales],
    ]
    existing_candidates = [spec.name for spec in existing_specs]
    candidate_names = [*existing_candidates, *seedbag_candidates]
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
    candidate_long = _candidate_long(candidate_wide, candidate_names)
    quality = pd.DataFrame(quality_rows)

    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    quality_path = output_dir / f"{OUTPUT_PREFIX}_pf_z_seedbag_quality.csv"
    wide_path = output_dir / f"{OUTPUT_PREFIX}_candidate_wide.csv.gz"
    long_path = output_dir / f"{OUTPUT_PREFIX}_candidate_long.csv.gz"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    quality.to_csv(quality_path, index=False)
    candidate_wide.to_csv(wide_path, index=False, compression="gzip")
    if bool(audit_config.get("save_candidate_long", True)):
        candidate_long.to_csv(long_path, index=False, compression="gzip")

    output_files = {
        "candidate_metrics": str(metrics_path),
        "bucket_metrics": str(bucket_path),
        "by_well": str(by_well_path),
        "pf_z_seedbag_quality": str(quality_path),
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
    seedbag_best = (
        candidate_metrics[candidate_metrics["candidate"].isin(seedbag_candidates)]
        .iloc[0]
        .to_dict()
    )
    exp072_likpf_mean = candidate_metrics[
        candidate_metrics["candidate"].eq("exp072_likpf_mean")
    ].iloc[0].to_dict()
    exp072_pf_z = (
        candidate_metrics[candidate_metrics["candidate"].eq("exp072_pf_z")]
        .iloc[0]
        .to_dict()
    )
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_not_run" if max_rows or max_wells else "completed_train_side_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_sec": float(time.time() - start),
        "rows": int(len(candidate_wide)),
        "wells": int(candidate_wide["well"].nunique()),
        "candidate_names": candidate_names,
        "pf_z_seedbag_candidates": seedbag_candidates,
        "existing_candidates": existing_candidates,
        "missing_exp072_scale_candidates": [
            f"exp072_likpf_scale_{scale:g}"
            for scale in scales
            if f"exp072_likpf_scale_{scale:g}" not in existing_candidates
        ],
        "best_candidate": to_jsonable(best),
        "pf_z_seedbag_best_candidate": to_jsonable(seedbag_best),
        "exp072_likpf_mean": to_jsonable(exp072_likpf_mean),
        "exp072_pf_z": to_jsonable(exp072_pf_z),
        "rmse_delta_pf_z_seedbag_best_minus_exp072_likpf_mean": float(
            seedbag_best["rmse"] - exp072_likpf_mean["rmse"]
        ),
        "rmse_delta_pf_z_seedbag_best_minus_exp072_pf_z": float(
            seedbag_best["rmse"] - exp072_pf_z["rmse"]
        ),
        "exp072_cache": cache_metadata,
        "input_file_sha256": input_sha,
        "output_files": output_files,
        "output_sha256": output_sha,
        "config_subset": {
            "pf_z_seedbag": model_config,
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
        config.setdefault("model", {}).setdefault("pf_z_seedbag", {})["n_seeds"] = int(
            args.n_seeds
        )
    if args.n_particles is not None:
        config.setdefault("model", {}).setdefault("pf_z_seedbag", {})["n_particles"] = int(
            args.n_particles
        )
    summary = run_audit(config)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
