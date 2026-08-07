from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from public_notebook_replay_audit import (
    configure_public_runtime,
    build_well,
    init_imputers,
    lik_pf,
    stable_seed,
)
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp169_tvt_input_pfbeam_offset_calibration"
EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
EXP072_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_columns: tuple[str, ...]
    transform: str
    role: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    base_candidate: str
    offset_source: str
    estimator: str
    alpha: float
    clip_ft: float
    near_guard_ft: float
    fade_end_ft: float
    max_iqr_ft: float
    min_rows: int
    use_slope: bool


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
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return values.astype(np.int32)


def float_tag(value: float) -> str:
    text = f"{value:.5g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def parse_candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for raw in get_nested(config, "audit.candidates") or []:
        source_columns = raw.get("source_columns")
        if source_columns is None:
            source_columns = [raw["source_column"]]
        specs.append(
            CandidateSpec(
                name=str(raw["name"]),
                source_columns=tuple(str(column) for column in source_columns),
                transform=str(raw["transform"]),
                role=str(raw.get("role", "candidate")),
            )
        )
    if not specs:
        raise ValueError("audit.candidates must define at least one candidate")
    return specs


def resolve_source_columns(header: list[str], specs: list[CandidateSpec]) -> dict[str, str]:
    header_set = set(header)
    resolved: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for spec in specs:
        match = next((column for column in spec.source_columns if column in header_set), None)
        if match is None:
            missing[spec.name] = list(spec.source_columns)
        else:
            resolved[spec.name] = match
    if missing:
        detail = "; ".join(f"{name}: {columns}" for name, columns in missing.items())
        raise ValueError(f"feature cache is missing candidate source columns: {detail}")
    return resolved


def read_tail_feature_cache(
    config: dict[str, Any],
    specs: list[CandidateSpec],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    explicit = get_nested(config, "data.exp072_train_feature_cache_local")
    source = find_artifact(EXP072_TRAIN_FEATURES, explicit)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    resolved_sources = resolve_source_columns(header, specs)
    required = {"id", "well", "target", "last_known_tvt", "md_since"}
    optional = {"eval_len", "last_anchor_tvt", "pf_ancc_std", "likpf_mean_d"}
    usecols = sorted(required | optional.intersection(header) | set(resolved_sources.values()))
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=usecols,
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    max_wells = get_nested(config, "audit.max_wells")
    if max_wells not in {None, "null"}:
        keep_wells = sorted(frame["well"].unique().tolist())[: int(max_wells)]
        frame = frame[frame["well"].isin(keep_wells)].reset_index(drop=True)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["row_idx"] = row_indices_from_ids(frame["id"])
    schema_path: Path | None = None
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
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
        "resolved_sources": resolved_sources,
    }
    return frame, metadata, resolved_sources


def materialize_tail_candidates(
    frame: pd.DataFrame,
    specs: list[CandidateSpec],
    resolved_sources: dict[str, str],
) -> list[str]:
    frame["true_tvt"] = numeric_array(frame, "last_known_tvt") + numeric_array(frame, "target")
    last_known = numeric_array(frame, "last_known_tvt")
    names: list[str] = []
    for spec in specs:
        values = numeric_array(frame, resolved_sources[spec.name])
        source_column = resolved_sources[spec.name]
        if spec.transform == "absolute":
            pred = values
        elif spec.transform == "base_plus_delta":
            pred = last_known + values if source_column.endswith("_d") else values
        else:
            raise ValueError(f"unsupported transform for {spec.name}: {spec.transform}")
        frame[spec.name] = pred.astype(np.float32)
        names.append(spec.name)
    disagreement_cols = [c for c in ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"] if c in frame]
    if len(disagreement_cols) >= 2:
        frame["candidate_disagreement_std"] = frame[disagreement_cols].std(axis=1).astype(np.float32)
        frame["candidate_disagreement_range"] = (
            frame[disagreement_cols].max(axis=1) - frame[disagreement_cols].min(axis=1)
        ).astype(np.float32)
    return names


def huber_location(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)].astype(np.float64)
    if len(finite) == 0:
        return float("nan")
    med = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    scale = max(float(q75 - q25) / 1.349, 1.0)
    clipped = med + np.clip(finite - med, -1.5 * scale, 1.5 * scale)
    return float(np.mean(clipped))


def robust_slope_per_1000md(md_since: np.ndarray, offset: np.ndarray, clip_abs: float) -> float:
    finite = np.isfinite(md_since) & np.isfinite(offset)
    if finite.sum() < 8:
        return 0.0
    x = md_since[finite].astype(np.float64)
    y = offset[finite].astype(np.float64)
    x = x - np.median(x)
    denom = float(np.dot(x, x))
    if denom <= 1e-9:
        return 0.0
    slope = float(np.dot(x, y - np.median(y)) / denom * 1000.0)
    return float(np.clip(slope, -clip_abs, clip_abs))


def candidate_abs_from_replay(frame: pd.DataFrame, candidate: str) -> np.ndarray | None:
    last = numeric_array(frame, "last_known_tvt")
    if candidate == "pf_ancc" and "pf_ancc" in frame:
        return numeric_array(frame, "pf_ancc")
    if candidate == "pf_z" and "pf_z" in frame:
        return numeric_array(frame, "pf_z")
    if candidate == "beam_mean" and "beam_mean_d" in frame:
        return last + numeric_array(frame, "beam_mean_d")
    if candidate == "sc_ens" and "sc_ens_d" in frame:
        return last + numeric_array(frame, "sc_ens_d")
    if candidate == "hyb" and "hyb_d" in frame:
        return last + numeric_array(frame, "hyb_d")
    if candidate == "likpf_mean" and "likpf_mean" in frame:
        return numeric_array(frame, "likpf_mean")
    return None


def replay_prefix_holdout_for_well(
    well: str,
    paths: ExperimentPaths,
    config: dict[str, Any],
    temp_root: Path,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    cfg = get_nested(config, "model.offset_calibration") or {}
    holdout_rows = int(cfg.get("prefix_holdout_rows", 256))
    min_known = int(cfg.get("min_known_prefix_rows", 80))
    min_cal = int(cfg.get("min_calibration_rows", 32))
    use_likpf = bool(get_nested(config, "model.replay_runtime.use_likpf_prefix"))

    hw_path = paths.train_data_dir / f"{well}__horizontal_well.csv"
    tw_path = paths.train_data_dir / f"{well}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        return None, {"well": well, "status": "missing_raw_files"}
    horizontal = pd.read_csv(hw_path, low_memory=False)
    known = np.flatnonzero(pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy())
    if len(known) < min_known + min_cal:
        return None, {"well": well, "status": "too_short_prefix", "known_rows": int(len(known))}
    start = max(min_known, len(known) - holdout_rows)
    holdout_idx = known[start:]
    if len(holdout_idx) < min_cal:
        return None, {"well": well, "status": "too_few_holdout_rows", "holdout_rows": int(len(holdout_idx))}

    masked = horizontal.copy()
    masked.loc[holdout_idx, "TVT_input"] = np.nan
    temp_hw = temp_root / f"{well}__horizontal_well.csv"
    masked.to_csv(temp_hw, index=False)

    replay = build_well(temp_hw, tw_path, is_train=True)
    if replay is None or len(replay) == 0:
        return None, {"well": well, "status": "replay_failed", "holdout_rows": int(len(holdout_idx))}
    replay = replay.copy()
    replay["row_idx"] = row_indices_from_ids(replay["id"])

    if use_likpf:
        tw = pd.read_csv(tw_path).sort_values("TVT")
        out, ev_index, _ = lik_pf(
            masked,
            tw,
            seed_base=stable_seed("exp169_likpf_prefix", well),
        )
        if len(ev_index) == len(replay) and "pf_mean" in out:
            replay["likpf_mean"] = np.asarray(out["pf_mean"], dtype=np.float32)

    holdout_set = set(int(v) for v in holdout_idx)
    replay = replay[replay["row_idx"].isin(holdout_set)].reset_index(drop=True)
    if len(replay) == 0:
        return None, {"well": well, "status": "no_holdout_rows_after_replay"}
    truth = horizontal.loc[replay["row_idx"].to_numpy(np.int64), "TVT_input"].to_numpy(np.float32)
    replay["prefix_true_tvt"] = truth
    replay["prefix_md"] = horizontal.loc[replay["row_idx"].to_numpy(np.int64), "MD"].to_numpy(np.float32)
    replay["prefix_anchor_idx"] = int(known[start - 1])
    anchor_md = float(horizontal.loc[int(known[start - 1]), "MD"])
    replay["prefix_md_since"] = replay["prefix_md"].to_numpy(np.float32) - np.float32(anchor_md)
    return replay, {
        "well": well,
        "status": "ok",
        "known_rows": int(len(known)),
        "holdout_rows": int(len(holdout_idx)),
        "replay_rows": int(len(replay)),
    }


def summarize_prefix_offsets(
    prefix_frame: pd.DataFrame,
    candidate_names: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recent_rows = int(get_nested(config, "model.offset_calibration.recent_rows") or 64)
    slope_clip = float(get_nested(config, "model.offset_calibration.slope_clip_ft_per_1000md") or 25.0)

    for well, group in prefix_frame.groupby("well", sort=False):
        candidate_offsets: dict[str, np.ndarray] = {}
        md_since = numeric_array(group, "prefix_md_since")
        true = numeric_array(group, "prefix_true_tvt")
        for candidate in candidate_names:
            pred = candidate_abs_from_replay(group, candidate)
            if pred is None:
                continue
            offset = pred - true
            finite = np.isfinite(offset)
            if not finite.any():
                continue
            candidate_offsets[candidate] = offset.astype(np.float32)
            recent_offset = offset[-recent_rows:] if len(offset) > recent_rows else offset
            q25, q75 = np.nanpercentile(offset[finite], [25.0, 75.0])
            err = offset[finite].astype(np.float64)
            rows.append(
                {
                    "well": str(well),
                    "offset_source": "self",
                    "candidate": candidate,
                    "rows": int(finite.sum()),
                    "coverage": float(finite.mean()),
                    "offset_median": float(np.nanmedian(offset)),
                    "offset_huber": huber_location(offset),
                    "offset_recent_median": float(np.nanmedian(recent_offset)),
                    "offset_iqr": float(q75 - q25),
                    "offset_slope_ft_per_1000md": robust_slope_per_1000md(md_since, offset, slope_clip),
                    "prefix_rmse": float(np.sqrt(np.mean(err * err))),
                    "prefix_mae": float(np.mean(np.abs(err))),
                    "prefix_bias": float(np.mean(err)),
                }
            )
        aggregate_names = [name for name in ["pf_ancc", "beam_mean", "likpf_mean"] if name in candidate_offsets]
        if aggregate_names:
            stacked = np.stack([candidate_offsets[name] for name in aggregate_names], axis=1)
            aggregate_offset = np.nanmedian(stacked, axis=1).astype(np.float32)
            finite = np.isfinite(aggregate_offset)
            q25, q75 = np.nanpercentile(aggregate_offset[finite], [25.0, 75.0])
            err = aggregate_offset[finite].astype(np.float64)
            rows.append(
                {
                    "well": str(well),
                    "offset_source": "pfbeam_median",
                    "candidate": "pfbeam_median",
                    "rows": int(finite.sum()),
                    "coverage": float(finite.mean()),
                    "offset_median": float(np.nanmedian(aggregate_offset)),
                    "offset_huber": huber_location(aggregate_offset),
                    "offset_recent_median": float(np.nanmedian(aggregate_offset[-recent_rows:])),
                    "offset_iqr": float(q75 - q25),
                    "offset_slope_ft_per_1000md": robust_slope_per_1000md(
                        md_since,
                        aggregate_offset,
                        slope_clip,
                    ),
                    "prefix_rmse": float(np.sqrt(np.mean(err * err))),
                    "prefix_mae": float(np.mean(np.abs(err))),
                    "prefix_bias": float(np.mean(err)),
                }
            )
    return pd.DataFrame(rows)


def build_prefix_offsets(
    tail_frame: pd.DataFrame,
    candidate_names: list[str],
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_cfg = get_nested(config, "model.replay_runtime") or {}
    configure_public_runtime(
        data_dir=paths.raw_data_dir,
        output_dir=paths.artifacts_dir,
        n_jobs=int(replay_cfg.get("n_jobs", 2)),
        pf_seeds=int(replay_cfg.get("pf_seeds", 32)),
        pf_particles=int(replay_cfg.get("pf_particles", 300)),
    )
    train_wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in paths.train_data_dir.glob("*__horizontal_well.csv")
    )
    init_imputers(train_wells)

    prefix_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    wells = sorted(tail_frame["well"].unique().tolist())
    with tempfile.TemporaryDirectory(prefix="exp169_prefix_holdout_") as tmp:
        temp_root = Path(tmp)
        for well in wells:
            replay, status = replay_prefix_holdout_for_well(well, paths, config, temp_root)
            status_rows.append(status)
            if replay is not None:
                prefix_frames.append(replay)
    if not prefix_frames:
        raise RuntimeError("No prefix holdout replay rows were generated; cannot estimate offsets.")
    prefix_frame = pd.concat(prefix_frames, ignore_index=True)
    offsets = summarize_prefix_offsets(prefix_frame, candidate_names, config)
    return prefix_frame, offsets, pd.DataFrame(status_rows)


def parse_variants(config: dict[str, Any]) -> list[VariantSpec]:
    cfg = get_nested(config, "model.offset_calibration.correction_grid") or {}
    variants: list[VariantSpec] = []
    for base in [str(v) for v in cfg.get("base_candidates", ["likpf_mean"])]:
        for source in [str(v) for v in cfg.get("offset_sources", ["self"])]:
            for estimator in [str(v) for v in cfg.get("estimators", ["median"])]:
                for alpha in [float(v) for v in cfg.get("alphas", [0.5])]:
                    for clip in [float(v) for v in cfg.get("clip_ft", [10.0])]:
                        for guard in [float(v) for v in cfg.get("near_guard_ft", [50.0])]:
                            for fade in [float(v) for v in cfg.get("fade_end_ft", [250.0])]:
                                for max_iqr in [float(v) for v in cfg.get("max_iqr_ft", [20.0])]:
                                    for min_rows in [int(v) for v in cfg.get("min_rows", [32])]:
                                        for use_slope in [bool(v) for v in cfg.get("use_slope", [False])]:
                                            name = (
                                                f"off_{base}_{source}_{estimator}"
                                                f"_a{float_tag(alpha)}_c{float_tag(clip)}"
                                                f"_g{float_tag(guard)}_f{float_tag(fade)}"
                                                f"_iqr{float_tag(max_iqr)}_n{min_rows}"
                                                f"_{'slope' if use_slope else 'const'}"
                                            )
                                            variants.append(
                                                VariantSpec(
                                                    name=name,
                                                    base_candidate=base,
                                                    offset_source=source,
                                                    estimator=estimator,
                                                    alpha=alpha,
                                                    clip_ft=clip,
                                                    near_guard_ft=guard,
                                                    fade_end_ft=fade,
                                                    max_iqr_ft=max_iqr,
                                                    min_rows=min_rows,
                                                    use_slope=use_slope,
                                                )
                                            )
    return variants


def md_fade_gate(md_since: np.ndarray, near_guard_ft: float, fade_end_ft: float) -> np.ndarray:
    denom = max(fade_end_ft - near_guard_ft, 1.0)
    return np.clip((md_since - near_guard_ft) / denom, 0.0, 1.0).astype(np.float32)


def offset_lookup_table(offsets: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in offsets.to_dict("records"):
        key = (str(row["well"]), str(row["offset_source"]), str(row["candidate"]))
        lookup[key] = row
    return lookup


def apply_variant(
    frame: pd.DataFrame,
    variant: VariantSpec,
    offset_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> np.ndarray:
    if variant.base_candidate not in frame.columns:
        raise ValueError(f"base candidate not found: {variant.base_candidate}")
    base = numeric_array(frame, variant.base_candidate)
    md_since = numeric_array(frame, "md_since")
    pred = base.copy()
    source_candidate = (
        variant.base_candidate if variant.offset_source == "self" else "pfbeam_median"
    )
    estimator_column = "offset_median" if variant.estimator == "median" else "offset_huber"

    for well, group in frame.groupby("well", sort=False):
        key = (str(well), variant.offset_source, source_candidate)
        stats = offset_lookup.get(key)
        if not stats:
            continue
        rows_value = stats.get("rows")
        iqr_value = stats.get("offset_iqr")
        offset_raw = stats.get(estimator_column)
        rows = int(rows_value) if rows_value is not None and np.isfinite(rows_value) else 0
        iqr = float(iqr_value) if iqr_value is not None and np.isfinite(iqr_value) else np.inf
        offset_value = (
            float(offset_raw) if offset_raw is not None and np.isfinite(offset_raw) else np.nan
        )
        if rows < variant.min_rows or iqr > variant.max_iqr_ft or not np.isfinite(offset_value):
            continue
        positions = group.index.to_numpy(np.int64)
        local_md = md_since[positions]
        correction = np.full(len(positions), offset_value, dtype=np.float32)
        if variant.use_slope:
            slope = float(stats.get("offset_slope_ft_per_1000md") or 0.0)
            correction = correction + (slope * local_md / 1000.0).astype(np.float32)
        correction = np.clip(correction, -variant.clip_ft, variant.clip_ft)
        gate = md_fade_gate(local_md, variant.near_guard_ft, variant.fade_end_ft)
        pred[positions] = base[positions] - variant.alpha * gate * correction
    return pred.astype(np.float32)


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {"rows": 0, "coverage": 0.0, "rmse": None, "mae": None, "within10": None, "bias": None}
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    return {
        "rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "within10": float(np.mean(np.abs(err) <= 10.0)),
        "bias": float(np.mean(err)),
    }


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(pd.Series(values), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def score_columns(frame: pd.DataFrame, columns: list[str], primary_baseline: str) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    baseline_score = None
    if primary_baseline in frame.columns:
        baseline_score = score_prediction(numeric_array(frame, primary_baseline), true)
    rows: list[dict[str, Any]] = []
    for column in columns:
        score = score_prediction(numeric_array(frame, column), true)
        delta = None
        if baseline_score and score["rmse"] is not None and baseline_score["rmse"] is not None:
            delta = float(score["rmse"] - baseline_score["rmse"])
        rows.append({"candidate": column, **score, "delta_rmse_vs_primary_baseline": delta})
    return (
        pd.DataFrame(rows)
        .sort_values(["rmse", "candidate"], na_position="last")
        .reset_index(drop=True)
    )


def compute_bucket_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    buckets = distance_bucket(frame["md_since"])
    rows: list[dict[str, Any]] = []
    for column in columns:
        pred = numeric_array(frame, column)
        for bucket, idx in pd.Series(buckets).groupby(buckets, observed=False).groups.items():
            positions = np.asarray(list(idx), dtype=np.int64)
            if len(positions) == 0:
                continue
            score = score_prediction(pred[positions], true[positions])
            rows.append({"candidate": column, "distance_bucket": str(bucket), **score})
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, columns: list[str], primary_baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        base_rmse = None
        if primary_baseline in group.columns:
            base_rmse = score_prediction(numeric_array(group, primary_baseline), true)["rmse"]
        for column in columns:
            score = score_prediction(numeric_array(group, column), true)
            delta = None
            if base_rmse is not None and score["rmse"] is not None:
                delta = float(score["rmse"] - base_rmse)
            rows.append({"well": str(well), "candidate": column, **score, "delta_rmse_vs_primary_baseline": delta})
    return pd.DataFrame(rows)


def compute_group_metrics(frame: pd.DataFrame, columns: list[str], config: dict[str, Any]) -> pd.DataFrame:
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": numeric_array(frame, "md_since") <= 50.0,
        "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
    }
    disagreement = numeric_array(frame, "candidate_disagreement_std")
    if np.isfinite(disagreement).any():
        groups["candidate_disagreement_top_quartile"] = disagreement >= np.nanquantile(disagreement, 0.75)
    for well in get_nested(config, "audit.representative_wells") or []:
        groups[f"representative_{well}"] = frame["well"].astype(str).to_numpy() == str(well)

    rows: list[dict[str, Any]] = []
    true = numeric_array(frame, "true_tvt")
    for group_name, mask in groups.items():
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue
        for column in columns:
            score = score_prediction(numeric_array(frame, column)[positions], true[positions])
            rows.append({"group": group_name, "candidate": column, **score})
    return pd.DataFrame(rows)


def max_well_regression(by_well: pd.DataFrame) -> pd.Series:
    return by_well.groupby("candidate", observed=True)["delta_rmse_vs_primary_baseline"].max()


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": OUTPUT_PREFIX,
            "feature_index": np.arange(len(columns), dtype=int),
            "feature": columns,
        }
    ).to_csv(path, index=False)


def run_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    specs = parse_candidate_specs(config)
    frame, feature_meta, resolved_sources = read_tail_feature_cache(config, specs)
    candidate_names = materialize_tail_candidates(frame, specs, resolved_sources)
    prefix_frame, prefix_offsets, prefix_status = build_prefix_offsets(
        frame,
        candidate_names,
        paths,
        config,
    )

    lookup = offset_lookup_table(prefix_offsets)
    variants = parse_variants(config)
    variant_columns: list[str] = []
    for variant in variants:
        if variant.base_candidate not in frame.columns:
            continue
        frame[variant.name] = apply_variant(frame, variant, lookup)
        variant_columns.append(variant.name)

    primary_baseline = str(get_nested(config, "audit.primary_baseline") or "likpf_mean")
    score_columns_all = [*candidate_names, *variant_columns]
    candidate_metrics = score_columns(frame, score_columns_all, primary_baseline)
    bucket_metrics = compute_bucket_metrics(frame, score_columns_all)
    by_well = compute_by_well(frame, score_columns_all, primary_baseline)
    group_metrics = compute_group_metrics(frame, score_columns_all, config)
    regressions = max_well_regression(by_well)
    candidate_metrics["max_well_regression_vs_primary"] = candidate_metrics["candidate"].map(regressions)

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    prefix_offsets_path = artifacts / f"{OUTPUT_PREFIX}_prefix_offsets.csv"
    prefix_status_path = artifacts / f"{OUTPUT_PREFIX}_prefix_status.csv"
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_path, index=False)
    prefix_offsets.to_csv(prefix_offsets_path, index=False)
    prefix_status.to_csv(prefix_status_path, index=False)

    top_k = int(get_nested(config, "audit.save_oof_top_k") or 10)
    top_variants = [
        str(value)
        for value in candidate_metrics[candidate_metrics["candidate"].isin(variant_columns)]
        .head(top_k)["candidate"]
        .tolist()
    ]
    keep_columns = [
        "id",
        "well",
        "row_idx",
        "target",
        "true_tvt",
        "last_known_tvt",
        "md_since",
        "eval_len",
        "candidate_disagreement_std",
        "candidate_disagreement_range",
        *candidate_names,
        *top_variants,
    ]
    keep_columns = list(dict.fromkeys([column for column in keep_columns if column in frame.columns]))
    frame[keep_columns].to_csv(oof_path, index=False, compression="gzip")
    write_feature_schema(schema_path, keep_columns)

    best = candidate_metrics.iloc[0].to_dict() if len(candidate_metrics) else {}
    primary = (
        candidate_metrics[candidate_metrics["candidate"] == primary_baseline].iloc[0].to_dict()
        if (candidate_metrics["candidate"] == primary_baseline).any()
        else {}
    )
    best_variant = (
        candidate_metrics[candidate_metrics["candidate"].isin(variant_columns)].iloc[0].to_dict()
        if len(variant_columns)
        else {}
    )
    ok_prefix = prefix_status[prefix_status["status"].eq("ok")] if "status" in prefix_status else pd.DataFrame()
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "feature_cache": feature_meta,
        "prefix_replay": {
            "status_counts": prefix_status["status"].value_counts().to_dict(),
            "ok_wells": int(len(ok_prefix)),
            "prefix_rows": int(len(prefix_frame)),
            "offset_rows": int(len(prefix_offsets)),
        },
        "candidate_names": candidate_names,
        "variant_count": len(variant_columns),
        "primary_baseline": to_jsonable(primary),
        "best_candidate": to_jsonable(best),
        "best_offset_variant": to_jsonable(best_variant),
        "decision": {
            "recommendation": "diagnostic_only_until_kaggle_result_review",
            "reason": (
                "TVT_input prefix offset correction is a train-side audit. "
                "Boundary discontinuity, raw-test parity, and worst-well regressions "
                "must be reviewed before any inference port."
            ),
        },
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_path),
            "prefix_offsets": str(prefix_offsets_path),
            "prefix_status": str(prefix_status_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_path),
            "prefix_offsets": sha256_path(prefix_offsets_path),
            "prefix_status": sha256_path(prefix_status_path),
            "oof_predictions_raw": sha256_path(oof_path),
            "oof_predictions_decompressed": sha256_path(oof_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")

    best_delta = float(best_variant.get("delta_rmse_vs_primary_baseline") or 0.0)
    metrics_status = (
        "completed_train_side_audit_candidate"
        if best_delta < 0.0
        else "completed_train_side_rejected_no_submit"
    )
    metrics_note = (
        "Train-side TVT_input PF/Beam offset calibration audit completed. "
        "Best offset variant improved the primary baseline; review guards before any inference port."
        if best_delta < 0.0
        else "Train-side TVT_input PF/Beam offset calibration audit completed. "
        "Best offset variant did not improve the primary baseline; no inference port or submission."
    )
    metrics_json = {
        "experiment": OUTPUT_PREFIX,
        "status": metrics_status,
        "metric": "rmse",
        "cv": float(primary["rmse"]) if primary.get("rmse") is not None else None,
        "public_lb": None,
        "private_lb": None,
        "primary_baseline": to_jsonable(primary),
        "best_offset_variant": to_jsonable(best_variant),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "summary_path": str(summary_path),
        "notes": metrics_note,
    }
    paths.metrics_path.write_text(json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_offset_variant"]), indent=2, sort_keys=True))
