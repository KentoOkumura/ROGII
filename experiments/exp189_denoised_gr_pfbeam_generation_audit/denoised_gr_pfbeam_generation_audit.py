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
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp189_denoised_gr_pfbeam_generation_audit"
EXPERIMENT_NAME = "exp189_denoised_gr_pfbeam_generation_audit"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


@dataclass(frozen=True)
class GrFilterSpec:
    name: str
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class PrefixHoldout:
    well: str
    masked: pd.DataFrame
    typewell: pd.DataFrame
    eval_index: np.ndarray
    eval_ids: np.ndarray
    true_tvt: np.ndarray
    target_delta: np.ndarray
    last_known_tvt: float
    last_known_md: float
    cache_md_since: np.ndarray | None
    reference_candidates: pd.DataFrame
    status: dict[str, Any]


@dataclass(frozen=True)
class PfRun:
    preds: np.ndarray
    log_likelihoods: np.ndarray
    ess_mean_by_row: np.ndarray
    resampled_by_row: np.ndarray
    seed_weights: np.ndarray
    sigma: float
    filter_metadata: dict[str, Any]


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


def find_existing_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path | None:
    paths: list[Path] = []
    if explicit_path is not None:
        paths.append(Path(explicit_path))
    if candidates:
        paths.extend(Path(item) for item in candidates)
    paths.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("artifacts") / filename,
            Path("experiments")
            / "exp072_exp063_full_replay_feature_cache"
            / "artifacts"
            / filename,
        ]
    )
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in KAGGLE_INPUT_ROOT.glob(f"**/{filename}"):
            if path.exists() and path.stat().st_size > 0:
                return path
    return None


def require_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    path = find_existing_path(filename=filename, explicit_path=explicit_path, candidates=candidates)
    if path is None:
        checked = [str(explicit_path)] if explicit_path is not None else []
        checked.extend(str(item) for item in candidates or [])
        raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked: {checked}")
    return path


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int64)


def reference_candidate_columns(config: dict[str, Any], header: list[str]) -> list[str]:
    requested = [
        str(value) for value in get_nested(config, "data.exp072_reference_candidates") or []
    ]
    return [column for column in requested if column in header]


def read_exp072_eval_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=FULL_REPLAY_TRAIN_FEATURES,
        explicit_path=get_nested(config, "data.exp072_train_feature_cache_local"),
    )
    required = ["id", "well", "target", "last_known_tvt"]
    optional = ["md_since", "eval_len", "known_len"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    references = reference_candidate_columns(config, header)
    usecols = required + [column for column in optional if column in header] + references
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=usecols,
        nrows=None if max_rows is None else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["row_idx"] = row_indices_from_ids(frame["id"]).astype(np.int32)
    frame["true_tvt"] = (
        numeric_array(frame, "last_known_tvt") + numeric_array(frame, "target")
    ).astype(np.float32)

    schema_path = find_existing_path(
        filename=FULL_REPLAY_FEATURE_SCHEMA,
        explicit_path=get_nested(config, "data.exp072_feature_schema_local"),
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
        "reference_candidates_present": references,
        "max_rows": None if max_rows is None else int(max_rows),
    }
    return frame, metadata


def fill_numeric(values: pd.Series | np.ndarray, fallback: float) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(np.float32)
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def savgol_or_rolling_mean(
    values: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if len(values) < window or window <= int(polyorder):
        effective = max(3, min(len(values), window))
        return rolling_mean(values, effective), {
            "effective_kind": "rolling_mean_short_series",
            "window": int(effective),
        }
    try:
        from scipy.signal import savgol_filter

        return (
            savgol_filter(
                values.astype(np.float64),
                window_length=window,
                polyorder=int(polyorder),
                mode="interp",
            ).astype(np.float32),
            {"effective_kind": "savgol", "window": window, "polyorder": int(polyorder)},
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle image packages.
        return rolling_mean(values, window), {
            "effective_kind": "rolling_mean_fallback",
            "window": window,
            "polyorder": int(polyorder),
            "fallback_reason": type(exc).__name__,
        }


def parse_filter_specs(config: dict[str, Any]) -> list[GrFilterSpec]:
    specs: list[GrFilterSpec] = []
    for raw in get_nested(config, "model.gr_filters") or []:
        item = dict(raw)
        name = str(item.pop("name"))
        kind = str(item.pop("kind", "raw"))
        specs.append(GrFilterSpec(name=name, kind=kind, params=item))
    if not specs:
        raise ValueError("model.gr_filters must define at least one filter")
    if specs[0].name != "raw" or specs[0].kind != "raw":
        raise ValueError("first model.gr_filters entry must be raw baseline")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate GR filter names: {names}")
    return specs


def apply_gr_filter(values: np.ndarray, spec: GrFilterSpec) -> tuple[np.ndarray, dict[str, Any]]:
    fallback = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    filled = fill_numeric(values, fallback)
    if spec.kind == "raw":
        return filled.astype(np.float32), {"effective_kind": "raw"}
    if spec.kind == "rolling_median":
        window = int(spec.params.get("window", 11))
        return rolling_median(filled, window), {
            "effective_kind": "rolling_median",
            "window": window,
        }
    if spec.kind == "rolling_mean":
        window = int(spec.params.get("window", 21))
        return rolling_mean(filled, window), {
            "effective_kind": "rolling_mean",
            "window": window,
        }
    if spec.kind == "savgol":
        return savgol_or_rolling_mean(
            filled,
            window=int(spec.params.get("window", 31)),
            polyorder=int(spec.params.get("polyorder", 2)),
        )
    raise ValueError(f"unknown GR filter kind: {spec.kind}")


def select_target_wells(
    validation_frame: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    surface = get_nested(config, "model.validation_surface") or {}
    include = [str(value) for value in surface.get("well_include", []) if value]
    min_eval = int(surface.get("min_eval_rows", 64))
    max_target_wells = surface.get("max_target_wells")
    summary = (
        validation_frame.groupby("well", sort=False)
        .agg(
            eval_rows=("id", "size"),
            max_md_since=("md_since", "max")
            if "md_since" in validation_frame.columns
            else ("row_idx", "max"),
            mean_md_since=("md_since", "mean")
            if "md_since" in validation_frame.columns
            else ("row_idx", "mean"),
            known_len=("known_len", "max")
            if "known_len" in validation_frame.columns
            else ("row_idx", "min"),
            eval_len=("eval_len", "max")
            if "eval_len" in validation_frame.columns
            else ("id", "size"),
        )
        .reset_index()
    )
    summary["has_horizontal"] = summary["well"].map(
        lambda well: (train_dir / f"{well}__horizontal_well.csv").exists()
    )
    summary["has_typewell"] = summary["well"].map(
        lambda well: (train_dir / f"{well}__typewell.csv").exists()
    )
    summary = summary[
        (summary["eval_rows"] >= min_eval) & summary["has_horizontal"] & summary["has_typewell"]
    ].copy()
    if include:
        selected = summary[summary["well"].isin(include)].copy()
    else:
        mode = str(surface.get("selection_mode", "long_md_since"))
        if mode == "eval_rows":
            selected = summary.sort_values(
                ["eval_rows", "max_md_since", "well"],
                ascending=[False, False, True],
            )
        else:
            selected = summary.sort_values(
                ["max_md_since", "eval_rows", "well"],
                ascending=[False, False, True],
            )
        if max_target_wells is not None:
            selected = selected.head(int(max_target_wells))
    return selected.reset_index(drop=True)


def build_eval_zone_for_well(
    well: str,
    eval_cache_rows: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> PrefixHoldout | None:
    validation_cfg = get_nested(config, "model.validation_surface") or {}
    min_known = int(validation_cfg.get("min_known_prefix_rows", 160))
    min_eval = int(validation_cfg.get("min_eval_rows", 64))

    hw_path = train_dir / f"{well}__horizontal_well.csv"
    tw_path = train_dir / f"{well}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists() or eval_cache_rows.empty:
        return None
    horizontal = pd.read_csv(hw_path, low_memory=False)
    typewell = pd.read_csv(tw_path, low_memory=False).sort_values("TVT").reset_index(drop=True)
    if len(typewell) < 3:
        return None

    cache_rows = eval_cache_rows.copy()
    if "row_idx" not in cache_rows.columns:
        cache_rows["row_idx"] = row_indices_from_ids(cache_rows["id"]).astype(np.int32)
    cache_rows = cache_rows.sort_values("row_idx").reset_index(drop=True)
    eval_index = pd.to_numeric(cache_rows["row_idx"], errors="coerce").to_numpy(np.int64)
    valid_index = (eval_index >= 0) & (eval_index < len(horizontal))
    if not valid_index.all():
        cache_rows = cache_rows.loc[valid_index].reset_index(drop=True)
        eval_index = eval_index[valid_index]
    if len(eval_index) < min_eval:
        return None

    known_mask = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    known_idx = np.flatnonzero(known_mask)
    if len(known_idx) < min_known:
        return None
    prefix_end = int(eval_index[0])
    known_before = known_idx[known_idx < prefix_end]
    if len(known_before) < min_known:
        return None
    last_known_idx = int(known_before[-1])

    masked = horizontal.iloc[: int(eval_index[-1]) + 1].copy()
    masked.loc[eval_index, "TVT_input"] = np.nan

    target_delta = numeric_array(cache_rows, "target")
    last_known_values = numeric_array(cache_rows, "last_known_tvt")
    last_known_tvt = float(last_known_values[0])
    truth = (last_known_values + target_delta).astype(np.float32)
    last_known_md = float(horizontal.loc[last_known_idx, "MD"])
    md_since = numeric_array(cache_rows, "md_since") if "md_since" in cache_rows.columns else None
    references = reference_candidate_columns(config, cache_rows.columns.tolist())
    status = {
        "well": well,
        "status": "ok",
        "validation_surface": "exp072_TVT_input_missing_equivalent_rows",
        "known_rows": int(len(known_idx)),
        "eval_rows": int(len(eval_index)),
        "last_known_idx": int(last_known_idx),
        "last_known_tvt": last_known_tvt,
        "last_known_md": last_known_md,
        "max_md_since": float(np.nanmax(md_since)) if md_since is not None else None,
        "reference_candidates_present": references,
        "raw_eval_tvt_input_missing_rate": float(
            pd.to_numeric(horizontal.loc[eval_index, "TVT_input"], errors="coerce")
            .isna()
            .mean()
        ),
    }
    return PrefixHoldout(
        well=well,
        masked=masked,
        typewell=typewell,
        eval_index=eval_index,
        eval_ids=cache_rows["id"].astype(str).to_numpy(),
        true_tvt=truth,
        target_delta=target_delta,
        last_known_tvt=last_known_tvt,
        last_known_md=last_known_md,
        cache_md_since=md_since,
        reference_candidates=cache_rows[references].copy() if references else pd.DataFrame(),
        status=status,
    )


def gr_sigma(
    *,
    horizontal: pd.DataFrame,
    filtered_horizontal_gr: np.ndarray,
    eval_start: int,
    tw_tvt: np.ndarray,
    filtered_typewell_gr: np.ndarray,
    config: dict[str, Any],
) -> float:
    runtime = get_nested(config, "model.runtime") or {}
    prefix = horizontal.iloc[:eval_start]
    finite = prefix["TVT_input"].notna() & prefix["GR"].notna()
    if int(finite.sum()) < 20:
        return float(runtime.get("gr_sigma_default", 30.0))
    prefix_pos = prefix.index.to_numpy(np.int64)
    tvt = pd.to_numeric(prefix.loc[finite, "TVT_input"], errors="coerce").to_numpy(np.float64)
    gr = filtered_horizontal_gr[prefix_pos[finite.to_numpy()]].astype(np.float64)
    residual = gr - np.interp(tvt, tw_tvt, filtered_typewell_gr)
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


def filtered_observations(
    holdout: PrefixHoldout,
    spec: GrFilterSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    hw = holdout.masked
    tw = holdout.typewell.sort_values("TVT").reset_index(drop=True)
    tw_tvt = numeric_array(tw, "TVT").astype(np.float64)
    raw_tw_gr = fill_numeric(tw["GR"], float(np.nanmean(numeric_array(tw, "GR")))).astype(
        np.float64
    )
    raw_hw_gr = fill_numeric(hw["GR"], float(np.nanmean(raw_tw_gr))).astype(np.float64)
    filtered_tw_gr, tw_meta = apply_gr_filter(raw_tw_gr, spec)
    filtered_hw_gr, hw_meta = apply_gr_filter(raw_hw_gr, spec)
    metadata = {
        "filter": spec.name,
        "kind": spec.kind,
        "params": spec.params,
        "typewell_filter": tw_meta,
        "horizontal_filter": hw_meta,
    }
    return (
        tw_tvt,
        filtered_tw_gr.astype(np.float64),
        filtered_hw_gr.astype(np.float64),
        raw_hw_gr,
        metadata,
    )


def run_pf_for_holdout(
    holdout: PrefixHoldout,
    spec: GrFilterSpec,
    config: dict[str, Any],
) -> PfRun:
    runtime = get_nested(config, "model.runtime") or {}
    n_particles = int(runtime.get("particles", 240))
    seed_count = int(runtime.get("seed_count", 8))
    temperature = float(runtime.get("likelihood_temperature", 6.0))
    resample_threshold = float(runtime.get("resample_threshold", 0.5))
    init_spread = float(runtime.get("init_spread", 4.5))
    velocity_noise = float(runtime.get("velocity_noise", 0.002))
    position_noise = float(runtime.get("position_noise", 0.005))
    resample_pos_noise = float(runtime.get("resample_pos_noise", 0.10))
    resample_velocity_noise = float(runtime.get("resample_velocity_noise", 0.001))

    hw = holdout.masked
    tw_tvt, tw_gr, hw_gr, _, metadata = filtered_observations(holdout, spec)
    eval_rows = hw.loc[holdout.eval_index].copy()
    md = numeric_array(eval_rows, "MD").astype(np.float64)
    gr = hw_gr[holdout.eval_index].astype(np.float64)
    prefix = hw.iloc[: int(holdout.eval_index[0])]
    sigma = gr_sigma(
        horizontal=hw,
        filtered_horizontal_gr=hw_gr,
        eval_start=int(holdout.eval_index[0]),
        tw_tvt=tw_tvt,
        filtered_typewell_gr=tw_gr,
        config=config,
    )
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
            likelihood = np.exp(-0.5 * residual2)
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
        sigma=sigma,
        filter_metadata=metadata,
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(float(values[idx - 1]) - target) <= abs(float(values[idx]) - target):
        return idx - 1
    return idx


def beam_search_for_holdout(
    holdout: PrefixHoldout,
    spec: GrFilterSpec,
    config: dict[str, Any],
) -> np.ndarray:
    beam_cfg = get_nested(config, "model.beam") or {}
    beam_size = int(beam_cfg.get("beam_size", 14))
    move_radius = int(beam_cfg.get("move_radius", 2))
    move_cost = float(beam_cfg.get("move_cost", 16.0))
    error_scale = float(beam_cfg.get("error_scale", 120.0))
    post_smooth_window = int(beam_cfg.get("post_smooth_window", 1))

    tw_tvt, tw_gr, hw_gr, _, _ = filtered_observations(holdout, spec)
    if post_smooth_window > 1:
        tw_gr = rolling_mean(tw_gr, post_smooth_window).astype(np.float64)
        hw_gr = rolling_mean(hw_gr, post_smooth_window).astype(np.float64)
    gr = hw_gr[holdout.eval_index].astype(np.float64)

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
                total = cost + gr_cost + move_cost * abs(delta)
                previous = candidates.get(next_idx)
                if previous is None or total < previous[0]:
                    candidates[next_idx] = (total, [*path, next_idx])
        kept = sorted(candidates.items(), key=lambda item: item[1][0])[:beam_size]
        active = {idx: value for idx, value in kept}
    if not active:
        return np.full(len(holdout.eval_index), holdout.last_known_tvt, dtype=np.float32)
    _, (_, best_path) = min(active.items(), key=lambda item: item[1][0])
    return tw_tvt[np.asarray(best_path, dtype=np.int64)].astype(np.float32)


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


def candidate_columns(frame: pd.DataFrame) -> list[str]:
    excluded_suffixes = ("_diag", "_name")
    prefixes = ("pf_", "beam_", "exp072_", "oracle_")
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes) and not column.endswith(excluded_suffixes)
    ]


def is_oracle_candidate(column: str) -> bool:
    return "_oracle" in column or column.startswith("oracle_")


def path_jump_rate_for_candidate(
    frame: pd.DataFrame,
    column: str,
    threshold_ft: float,
) -> float | None:
    jumps = 0
    total = 0
    for _, group in frame.sort_values(["well", "row_idx"]).groupby("well", sort=False):
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        diffs = np.abs(np.diff(values[finite]))
        jumps += int(np.sum(diffs > threshold_ft))
        total += int(len(diffs))
    if total == 0:
        return None
    return float(jumps / total)


def compute_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_raw_lik_mean")
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
                "is_oracle_diagnostic": is_oracle_candidate(column),
                **score,
                "delta_rmse_vs_primary_baseline": delta,
                "path_jump_rate": path_jump_rate_for_candidate(frame, column, threshold),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"], na_position="last")


def compute_filter_delta_metrics(frame: pd.DataFrame, filters: list[GrFilterSpec]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    specs = [
        ("pf_lik_mean", "pf_raw_lik_mean", "pf_{name}_lik_mean"),
        ("pf_best_seed", "pf_raw_best_seed", "pf_{name}_best_seed"),
        ("pf_top3_oracle", "pf_raw_top3_oracle", "pf_{name}_top3_oracle"),
        ("beam_top1", "beam_raw_top1", "beam_{name}_top1"),
    ]
    rows: list[dict[str, Any]] = []
    for family, raw_column, template in specs:
        if raw_column not in frame:
            continue
        raw_score = score_prediction(numeric_array(frame, raw_column), true)
        raw_values = numeric_array(frame, raw_column)
        for spec in filters:
            if spec.name == "raw":
                continue
            column = template.format(name=spec.name)
            if column not in frame:
                continue
            score = score_prediction(numeric_array(frame, column), true)
            values = numeric_array(frame, column)
            diff = values.astype(np.float64) - raw_values.astype(np.float64)
            finite = np.isfinite(diff)
            delta = None
            if score["rmse"] is not None and raw_score["rmse"] is not None:
                delta = float(score["rmse"] - raw_score["rmse"])
            rows.append(
                {
                    "family": family,
                    "filter": spec.name,
                    "filter_kind": spec.kind,
                    "candidate": column,
                    "raw_candidate": raw_column,
                    **score,
                    "raw_rmse": raw_score["rmse"],
                    "delta_rmse_vs_raw_family": delta,
                    "row_abs_diff_mean_vs_raw": (
                        float(np.mean(np.abs(diff[finite]))) if finite.any() else None
                    ),
                    "row_diff_rmse_vs_raw": (
                        float(np.sqrt(np.mean(diff[finite] * diff[finite])))
                        if finite.any()
                        else None
                    ),
                    "changed_rows_vs_raw": int(np.sum(np.abs(diff[finite]) > 1.0e-6))
                    if finite.any()
                    else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["delta_rmse_vs_raw_family", "candidate"],
        na_position="last",
    )


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
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_raw_lik_mean")
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
                    "eval_rows": int(len(group)),
                    "max_md_since": float(pd.to_numeric(group["md_since"], errors="coerce").max()),
                    **score,
                    "delta_rmse_vs_primary_baseline": delta,
                }
            )
    return pd.DataFrame(rows)


def compute_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    md_since = numeric_array(frame, "md_since")
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": md_since <= 50.0,
        "mid_250_1000": (md_since >= 250.0) & (md_since < 1000.0),
        "longtail_1000_plus": md_since >= 1000.0,
        "very_longtail_2000_plus": md_since >= 2000.0,
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


def rowwise_oracle(
    frame: pd.DataFrame,
    columns: list[str],
    output_name: str,
) -> None:
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return
    true = numeric_array(frame, "true_tvt").astype(np.float64)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    err = np.abs(values - true[:, None])
    all_nan = ~np.isfinite(err).any(axis=1)
    err[~np.isfinite(err)] = np.inf
    best_pos = np.argmin(err, axis=1)
    pred = values[np.arange(len(frame)), best_pos]
    pred[all_nan] = np.nan
    frame[output_name] = pred.astype(np.float32)
    frame[f"{output_name}_name"] = np.asarray(columns, dtype=object)[best_pos]


def build_row_frame_for_holdout(
    holdout: PrefixHoldout,
    filters: list[GrFilterSpec],
    pf_outputs: dict[str, PfRun],
    beam_outputs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eval_rows = holdout.masked.loc[holdout.eval_index]
    md_since = (
        holdout.cache_md_since
        if holdout.cache_md_since is not None
        else numeric_array(eval_rows, "MD") - np.float32(holdout.last_known_md)
    )
    row_frame = pd.DataFrame(
        {
            "well": holdout.well,
            "row_idx": holdout.eval_index.astype(np.int32),
            "id": holdout.eval_ids.astype(str),
            "target": holdout.target_delta.astype(np.float32),
            "true_tvt": holdout.true_tvt.astype(np.float32),
            "last_known_tvt": np.float32(holdout.last_known_tvt),
            "last_known_md": np.float32(holdout.last_known_md),
            "md_since": md_since.astype(np.float32),
        }
    )
    for column in holdout.reference_candidates.columns:
        row_frame[f"exp072_{column}"] = pd.to_numeric(
            holdout.reference_candidates[column],
            errors="coerce",
        ).to_numpy(np.float32)

    top_k = int(get_nested(config, "model.runtime.topk_oracle") or 3)
    diagnostics: list[dict[str, Any]] = []
    for spec in filters:
        run = pf_outputs[spec.name]
        weighted = (run.seed_weights[:, None] * run.preds).sum(axis=0)
        best_idx = int(np.argmax(run.log_likelihoods))
        top_idx = np.argsort(run.log_likelihoods)[::-1][:top_k]
        oracle = np.empty(len(row_frame), dtype=np.float32)
        truth = row_frame["true_tvt"].to_numpy(np.float32)
        for i in range(len(row_frame)):
            seed_values = run.preds[top_idx, i]
            oracle[i] = seed_values[np.argmin(np.abs(seed_values - truth[i]))]
        row_frame[f"pf_{spec.name}_lik_mean"] = weighted.astype(np.float32)
        row_frame[f"pf_{spec.name}_best_seed"] = run.preds[best_idx].astype(np.float32)
        row_frame[f"pf_{spec.name}_top{top_k}_oracle"] = oracle
        row_frame[f"pf_{spec.name}_ess_mean_diag"] = run.ess_mean_by_row.astype(np.float32)
        row_frame[f"pf_{spec.name}_resampled_rate_diag"] = run.resampled_by_row.astype(np.float32)
        diagnostics.append(
            {
                "well": holdout.well,
                "filter": spec.name,
                "filter_kind": spec.kind,
                "filter_params": json.dumps(to_jsonable(spec.params), sort_keys=True),
                "seed_count": int(run.preds.shape[0]),
                "rows": int(run.preds.shape[1]),
                "gr_sigma": float(run.sigma),
                "log_likelihood_mean": float(np.mean(run.log_likelihoods)),
                "log_likelihood_std": float(np.std(run.log_likelihoods)),
                "ess_mean": float(np.mean(run.ess_mean_by_row)),
                "resampling_rate": float(np.mean(run.resampled_by_row)),
                "seed_weight_max": float(np.max(run.seed_weights)),
                "filter_metadata": json.dumps(to_jsonable(run.filter_metadata), sort_keys=True),
            }
        )
    for spec in filters:
        row_frame[f"beam_{spec.name}_top1"] = beam_outputs[spec.name].astype(np.float32)

    filter_candidate_cols = []
    smoothed_candidate_cols = []
    for spec in filters:
        cols = [f"pf_{spec.name}_lik_mean", f"beam_{spec.name}_top1"]
        filter_candidate_cols.extend(cols)
        if spec.name != "raw":
            smoothed_candidate_cols.extend(cols)
    rowwise_oracle(row_frame, filter_candidate_cols, "oracle_best_filter_candidate")
    rowwise_oracle(row_frame, smoothed_candidate_cols, "oracle_best_smoothed_candidate")
    return row_frame, diagnostics


def selector_headroom_summary(candidate_metrics: pd.DataFrame, primary: str) -> dict[str, Any]:
    rows = candidate_metrics.set_index("candidate")
    primary_rmse = rows.loc[primary, "rmse"] if primary in rows.index else np.nan
    out: dict[str, Any] = {"primary": primary, "primary_rmse": float(primary_rmse)}
    for candidate in ["oracle_best_filter_candidate", "oracle_best_smoothed_candidate"]:
        if candidate in rows.index:
            rmse = rows.loc[candidate, "rmse"]
            out[candidate] = {
                "rmse": float(rmse),
                "delta_rmse_vs_primary": float(rmse - primary_rmse)
                if np.isfinite(primary_rmse) and np.isfinite(rmse)
                else None,
            }
    return out


def run_denoised_gr_pfbeam_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    train_dir = paths.train_data_dir
    filters = parse_filter_specs(config)
    validation_frame, validation_meta = read_exp072_eval_cache(config)
    target_wells = select_target_wells(validation_frame, train_dir, config)
    target_well_set = set(target_wells["well"].astype(str).tolist())
    validation_frame = validation_frame[validation_frame["well"].isin(target_well_set)].copy()
    validation_rows_by_well = {
        str(well): group.copy() for well, group in validation_frame.groupby("well", sort=False)
    }

    row_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for _, target_row in target_wells.iterrows():
        well = str(target_row["well"])
        holdout = build_eval_zone_for_well(
            well,
            validation_rows_by_well.get(well, pd.DataFrame()),
            train_dir,
            config,
        )
        if holdout is None:
            status_rows.append({"well": well, "status": "skipped_no_valid_exp072_eval_zone"})
            continue
        pf_outputs: dict[str, PfRun] = {}
        beam_outputs: dict[str, np.ndarray] = {}
        for spec in filters:
            pf_outputs[spec.name] = run_pf_for_holdout(holdout, spec, config)
            beam_outputs[spec.name] = beam_search_for_holdout(holdout, spec, config)
        frame, diag = build_row_frame_for_holdout(
            holdout,
            filters,
            pf_outputs,
            beam_outputs,
            config,
        )
        row_frames.append(frame)
        diagnostics.extend(diag)
        status = dict(holdout.status)
        status.update(
            {
                "filters": ",".join(spec.name for spec in filters),
                "filter_count": int(len(filters)),
            }
        )
        status_rows.append(status)

    if not row_frames:
        raise RuntimeError("No denoised-GR PF/Beam audit rows were generated.")

    row_frame = pd.concat(row_frames, ignore_index=True)
    pf_diagnostics = pd.DataFrame(diagnostics)
    well_status = pd.DataFrame(status_rows)
    candidate_metrics = compute_candidate_metrics(row_frame, config)
    filter_delta_metrics = compute_filter_delta_metrics(row_frame, filters)
    bucket_metrics = compute_bucket_metrics(row_frame)
    by_well = compute_by_well(row_frame, config)
    group_metrics = compute_group_metrics(row_frame)
    add_worst_well_regression(candidate_metrics, by_well)

    artifacts = paths.artifacts_dir
    candidate_metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    filter_delta_path = artifacts / f"{OUTPUT_PREFIX}_filter_delta_metrics.csv"
    bucket_metrics_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_metrics_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    pf_diagnostics_path = artifacts / f"{OUTPUT_PREFIX}_pf_diagnostics.csv"
    target_wells_path = artifacts / f"{OUTPUT_PREFIX}_target_wells.csv"
    well_status_path = artifacts / f"{OUTPUT_PREFIX}_well_status.csv"
    row_candidates_path = artifacts / f"{OUTPUT_PREFIX}_row_candidates.csv.gz"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    filter_delta_metrics.to_csv(filter_delta_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_metrics_path, index=False)
    pf_diagnostics.to_csv(pf_diagnostics_path, index=False)
    target_wells.to_csv(target_wells_path, index=False)
    well_status.to_csv(well_status_path, index=False)
    row_frame.to_csv(row_candidates_path, index=False, compression="gzip")

    best_row = (
        candidate_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    deployable_metrics = candidate_metrics[~candidate_metrics["is_oracle_diagnostic"]]
    best_non_oracle_row = (
        deployable_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    best_smoothed_delta = (
        filter_delta_metrics[~filter_delta_metrics["candidate"].str.contains("_oracle")]
        .sort_values(["delta_rmse_vs_raw_family", "candidate"], na_position="last")
        .head(1)
        .to_dict("records")
    )
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_raw_lik_mean")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - started),
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "target_wells_selected": int(len(target_wells)),
        "filters": [to_jsonable(spec.__dict__) for spec in filters],
        "primary_baseline": primary,
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "best_smoothed_delta_vs_raw_family": best_smoothed_delta[0]
        if best_smoothed_delta
        else None,
        "selector_headroom": selector_headroom_summary(candidate_metrics, primary),
        "validation_surface": validation_meta,
        "pf_diagnostics_summary": (
            pf_diagnostics.groupby("filter", observed=True)
            .agg(
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                ess_mean=("ess_mean", "mean"),
                resampling_rate=("resampling_rate", "mean"),
                log_likelihood_mean=("log_likelihood_mean", "mean"),
                gr_sigma=("gr_sigma", "mean"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "artifacts": {
            "candidate_metrics": str(candidate_metrics_path),
            "filter_delta_metrics": str(filter_delta_path),
            "bucket_metrics": str(bucket_metrics_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_metrics_path),
            "pf_diagnostics": str(pf_diagnostics_path),
            "target_wells": str(target_wells_path),
            "well_status": str(well_status_path),
            "row_candidates": str(row_candidates_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "validation_surface": validation_meta,
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "best_smoothed_delta_vs_raw_family": summary["best_smoothed_delta_vs_raw_family"],
        "selector_headroom": summary["selector_headroom"],
        "artifacts": summary["artifacts"],
        "sha256": {
            "candidate_metrics": sha256_path(candidate_metrics_path),
            "filter_delta_metrics": sha256_path(filter_delta_path),
            "bucket_metrics": sha256_path(bucket_metrics_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_metrics_path),
            "pf_diagnostics": sha256_path(pf_diagnostics_path),
            "target_wells": sha256_path(target_wells_path),
            "well_status": sha256_path(well_status_path),
            "row_candidates_raw_gzip": sha256_path(row_candidates_path),
            "row_candidates_decompressed": sha256_path(row_candidates_path, decompressed=True),
            "summary": sha256_path(summary_path),
        },
        "notes": (
            "Train-side exp072-aligned denoised-GR PF/Beam audit only; "
            "no model, inference, or submission."
        ),
    }
    write_json(paths.metrics_path, metrics)
    return {
        "summary": summary,
        "candidate_metrics": candidate_metrics,
        "filter_delta_metrics": filter_delta_metrics,
        "bucket_metrics": bucket_metrics,
        "by_well": by_well,
        "group_metrics": group_metrics,
        "pf_diagnostics": pf_diagnostics,
        "target_wells": target_wells,
        "well_status": well_status,
        "row_frame": row_frame,
    }


def main() -> dict[str, Any]:
    return run_denoised_gr_pfbeam_audit()


if __name__ == "__main__":
    result = main()
    print(json.dumps(to_jsonable(result["summary"]), indent=2, sort_keys=True))
