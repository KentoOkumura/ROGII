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

try:
    from numba import njit
except ImportError:  # Local static validation may not have numba installed.

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]
        return lambda func: func


OUTPUT_PREFIX = "exp146_tvt_plus_z_beam_smoothness_penalty"
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
class BeamVariant:
    name: str
    beam_size: int
    move_cost: float
    error_scale: float
    smooth_radius: int
    u_abs_cost: float
    u_abs_scale: float
    u_slope_cost: float
    u_curve_cost: float


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


def read_feature_cache(
    config: dict[str, Any],
    specs: list[CandidateSpec],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    explicit = get_nested(config, "data.exp072_train_feature_cache_local")
    source = find_artifact(EXP072_TRAIN_FEATURES, explicit)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    resolved_sources = resolve_source_columns(header, specs)
    required = {"id", "well", "target", "last_known_tvt", "md_since"}
    optional = {"eval_len", "beam_std_d", "pf_ancc_std"}
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


def materialize_candidates(
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
    return names


def smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    series = pd.Series(values, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius > 0:
        series = series.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return series.to_numpy(np.float64)


def nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(values[idx - 1] - target) <= abs(values[idx] - target):
        return idx - 1
    return idx


@njit(cache=True)
def beam_u_smooth_jit(
    smooth_gr_v,
    md_v,
    z_v,
    typewell_tvt,
    typewell_gr,
    start_index,
    start_tvt,
    start_md,
    start_z,
    beam_size,
    move_cost,
    error_scale,
    u_abs_cost,
    u_abs_scale,
    u_slope_cost,
    u_curve_cost,
):
    n = len(smooth_gr_v)
    nt = len(typewell_gr)
    max_candidates = beam_size * 5
    beam_idx = np.zeros(beam_size, np.int64)
    beam_idx[0] = start_index
    beam_cost = np.full(beam_size, 1e30)
    beam_cost[0] = 0.0
    beam_u_slope = np.zeros(beam_size)
    active = np.int64(1)
    hist_idx = np.zeros((n, beam_size), np.int64)
    hist_parent = np.zeros((n, beam_size), np.int64)
    cand_idx = np.zeros(max_candidates, np.int64)
    cand_parent = np.zeros(max_candidates, np.int64)
    cand_cost = np.full(max_candidates, 1e30)
    cand_u_slope = np.zeros(max_candidates)
    u0 = start_tvt + start_z
    prev_md = start_md
    prev_z = start_z

    for step in range(n):
        dm = md_v[step] - prev_md
        if dm < 1.0:
            dm = 1.0
        dz = z_v[step] - prev_z
        candidate_count = np.int64(0)
        for bi in range(active):
            idx = beam_idx[bi]
            cost = beam_cost[bi]
            prev_u_slope = beam_u_slope[bi]
            prev_tvt = typewell_tvt[idx]
            for move in range(-2, 3):
                next_idx = idx + move
                if next_idx < 0 or next_idx >= nt:
                    continue
                next_tvt = typewell_tvt[next_idx]
                dtvt = next_tvt - prev_tvt
                u_value = next_tvt + z_v[step] - u0
                u_slope = (dtvt + dz) / dm
                u_curve = u_slope - prev_u_slope
                gr_delta = smooth_gr_v[step] - typewell_gr[next_idx]
                move_abs = move if move >= 0 else -move
                u_abs_norm = u_value / u_abs_scale
                total = (
                    cost
                    + (gr_delta * gr_delta) / error_scale
                    + move_cost * move_abs
                    + u_abs_cost * u_abs_norm * u_abs_norm
                    + u_slope_cost * u_slope * u_slope
                    + u_curve_cost * u_curve * u_curve
                )
                found = np.int64(-1)
                for ci in range(candidate_count):
                    if cand_idx[ci] == next_idx:
                        found = ci
                        break
                if found >= 0:
                    if total < cand_cost[found]:
                        cand_cost[found] = total
                        cand_parent[found] = bi
                        cand_u_slope[found] = u_slope
                elif candidate_count < max_candidates:
                    cand_idx[candidate_count] = next_idx
                    cand_cost[candidate_count] = total
                    cand_parent[candidate_count] = bi
                    cand_u_slope[candidate_count] = u_slope
                    candidate_count += 1
        kept = min(beam_size, candidate_count)
        for i in range(kept):
            best = i
            for j in range(i + 1, candidate_count):
                if cand_cost[j] < cand_cost[best]:
                    best = j
            if best != i:
                cand_idx[i], cand_idx[best] = cand_idx[best], cand_idx[i]
                cand_cost[i], cand_cost[best] = cand_cost[best], cand_cost[i]
                cand_parent[i], cand_parent[best] = cand_parent[best], cand_parent[i]
                cand_u_slope[i], cand_u_slope[best] = cand_u_slope[best], cand_u_slope[i]
        hist_idx[step, :kept] = cand_idx[:kept]
        hist_parent[step, :kept] = cand_parent[:kept]
        beam_idx[:kept] = cand_idx[:kept]
        beam_cost[:kept] = cand_cost[:kept]
        beam_u_slope[:kept] = cand_u_slope[:kept]
        active = kept
        prev_md = md_v[step]
        prev_z = z_v[step]

    best = np.int64(0)
    for bi in range(1, active):
        if beam_cost[bi] < beam_cost[best]:
            best = bi
    path = np.zeros(n, np.int64)
    cursor = best
    for step in range(n - 1, -1, -1):
        path[step] = hist_idx[step, cursor]
        cursor = hist_parent[step, cursor]
    return path, beam_cost[best]


def parse_beam_variants(config: dict[str, Any]) -> list[BeamVariant]:
    variants: list[BeamVariant] = []
    for raw in get_nested(config, "model.beam_smoothness.variants") or []:
        variants.append(
            BeamVariant(
                name=str(raw["name"]),
                beam_size=int(raw.get("beam_size", 32)),
                move_cost=float(raw.get("move_cost", 15.0)),
                error_scale=float(raw.get("error_scale", 100.0)),
                smooth_radius=int(raw.get("smooth_radius", 0)),
                u_abs_cost=float(raw.get("u_abs_cost", 0.0)),
                u_abs_scale=float(raw.get("u_abs_scale", 100.0)),
                u_slope_cost=float(raw.get("u_slope_cost", 0.0)),
                u_curve_cost=float(raw.get("u_curve_cost", 0.0)),
            )
        )
    if not variants:
        raise ValueError("model.beam_smoothness.variants must define at least one variant")
    return variants


def generate_well_beams(
    well: str,
    group: pd.DataFrame,
    train_dir: Path,
    variants: list[BeamVariant],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"missing raw train files for {well}")
    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "GR", "TVT_input"])
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"]).sort_values("TVT")
    row_idx = group["row_idx"].to_numpy(np.int64)
    if row_idx.max(initial=-1) >= len(horizontal):
        raise IndexError(f"row_idx exceeds horizontal rows for {well}: {row_idx.max()}")
    known = horizontal[horizontal["TVT_input"].notna()]
    if known.empty:
        raise ValueError(f"well {well} has no finite TVT_input prefix")
    anchor = known.iloc[-1]
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    typewell_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    finite_typewell = np.isfinite(typewell_tvt) & np.isfinite(typewell_gr)
    typewell_tvt = typewell_tvt[finite_typewell]
    typewell_gr = typewell_gr[finite_typewell]
    order = np.argsort(typewell_tvt)
    typewell_tvt = typewell_tvt[order]
    typewell_gr = typewell_gr[order]
    if len(typewell_tvt) < 3:
        raise ValueError(f"well {well} typewell needs at least three finite rows")

    eval_frame = horizontal.iloc[row_idx]
    gr_v = pd.to_numeric(eval_frame["GR"], errors="coerce").to_numpy(np.float32)
    md_v = pd.to_numeric(eval_frame["MD"], errors="coerce").to_numpy(np.float64)
    z_v = pd.to_numeric(eval_frame["Z"], errors="coerce").to_numpy(np.float64)
    fallback_gr = float(np.nanmean(typewell_gr))
    start_tvt = float(anchor["TVT_input"])
    start_md = float(anchor["MD"])
    start_z = float(anchor["Z"])
    start_index = nearest_index(typewell_tvt, start_tvt)

    out = pd.DataFrame({"id": group["id"].to_numpy(str), "well": str(well), "row_idx": row_idx})
    out["md"] = md_v.astype(np.float32)
    out["z"] = z_v.astype(np.float32)
    out["u_reference"] = (out["z"].to_numpy(np.float32) + start_tvt + 0.0 - start_z).astype(
        np.float32
    )
    quality_rows: list[dict[str, Any]] = []
    for variant in variants:
        smooth_gr_v = smooth_gr(gr_v, fallback_gr, variant.smooth_radius)
        path_indices, cost = beam_u_smooth_jit(
            smooth_gr_v.astype(np.float64),
            md_v.astype(np.float64),
            z_v.astype(np.float64),
            typewell_tvt.astype(np.float64),
            typewell_gr.astype(np.float64),
            int(start_index),
            float(start_tvt),
            float(start_md),
            float(start_z),
            int(variant.beam_size),
            float(variant.move_cost),
            float(variant.error_scale),
            float(variant.u_abs_cost),
            float(variant.u_abs_scale),
            float(variant.u_slope_cost),
            float(variant.u_curve_cost),
        )
        pred = typewell_tvt[path_indices].astype(np.float32)
        out[variant.name] = pred
        if len(pred) > 1:
            du_slope = np.gradient(pred.astype(np.float64) + z_v.astype(np.float64), md_v)
        else:
            du_slope = np.zeros(len(pred), dtype=np.float64)
        quality_rows.append(
            {
                "well": str(well),
                "variant": variant.name,
                "rows": int(len(pred)),
                "cost": float(cost),
                "cost_per_row": float(cost / max(len(pred), 1)),
                "mean_abs_step": float(np.mean(np.abs(np.diff(pred)))) if len(pred) > 1 else 0.0,
                "p95_abs_step": float(np.quantile(np.abs(np.diff(pred)), 0.95))
                if len(pred) > 1
                else 0.0,
                "mean_abs_du_slope": float(np.mean(np.abs(du_slope))),
                "p95_abs_du_slope": float(np.quantile(np.abs(du_slope), 0.95)),
            }
        )
    input_sha = {
        str(horizontal_path): sha256_path(horizontal_path),
        str(typewell_path): sha256_path(typewell_path),
    }
    return out, quality_rows, input_sha


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {"rows": 0, "coverage": 0.0, "rmse": None, "mae": None, "within10": None}
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    abs_err = np.abs(err)
    return {
        "rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(abs_err)),
        "within10": float(np.mean(abs_err <= 10.0)),
        "bias": float(np.mean(err)),
    }


def score_columns(frame: pd.DataFrame, columns: list[str], primary_baseline: str) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    baseline = (
        score_prediction(numeric_array(frame, primary_baseline), true)
        if primary_baseline in frame.columns
        else None
    )
    rows: list[dict[str, Any]] = []
    for column in columns:
        score = score_prediction(numeric_array(frame, column), true)
        delta = None
        if baseline and score["rmse"] is not None and baseline["rmse"] is not None:
            delta = float(score["rmse"] - baseline["rmse"])
        rows.append({"candidate": column, **score, "delta_rmse_vs_primary_baseline": delta})
    return (
        pd.DataFrame(rows)
        .sort_values(["rmse", "candidate"], na_position="last")
        .reset_index(drop=True)
    )


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(pd.Series(values), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def bucket_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    true = numeric_array(frame, "true_tvt")
    buckets = distance_bucket(frame["md_since"])
    for column in columns:
        pred = numeric_array(frame, column)
        for bucket, idx in pd.Series(buckets).groupby(buckets, observed=False).groups.items():
            positions = np.asarray(list(idx), dtype=np.int64)
            if len(positions) == 0:
                continue
            rows.append(
                {
                    "candidate": column,
                    "bucket_family": "md_since",
                    "bucket": str(bucket),
                    **score_prediction(pred[positions], true[positions]),
                }
            )
    return pd.DataFrame(rows)


def by_well_metrics(frame: pd.DataFrame, columns: list[str], primary_baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        baseline_rmse = None
        if primary_baseline in group.columns:
            baseline_rmse = score_prediction(numeric_array(group, primary_baseline), true)["rmse"]
        for column in columns:
            score = score_prediction(numeric_array(group, column), true)
            delta = None
            if baseline_rmse is not None and score["rmse"] is not None:
                delta = float(score["rmse"] - baseline_rmse)
            rows.append(
                {
                    "well": str(well),
                    "candidate": column,
                    **score,
                    "delta_rmse_vs_primary_baseline": delta,
                }
            )
    return pd.DataFrame(rows)


def group_metrics(frame: pd.DataFrame, columns: list[str], config: dict[str, Any]) -> pd.DataFrame:
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": numeric_array(frame, "md_since") <= 50.0,
        "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
    }
    beam_disagreement = (
        np.abs(numeric_array(frame, "beam_mean") - numeric_array(frame, "likpf_mean"))
        if {"beam_mean", "likpf_mean"}.issubset(frame.columns)
        else np.zeros(len(frame), dtype=np.float32)
    )
    if np.isfinite(beam_disagreement).any():
        groups["beam_likpf_gap_top_quartile"] = beam_disagreement >= np.nanquantile(
            beam_disagreement,
            0.75,
        )
    for well in get_nested(config, "audit.representative_wells") or []:
        groups[f"representative_{well}"] = frame["well"].astype(str).to_numpy() == str(well)
    true = numeric_array(frame, "true_tvt")
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue
        for column in columns:
            rows.append(
                {
                    "group": group_name,
                    "candidate": column,
                    **score_prediction(numeric_array(frame, column)[positions], true[positions]),
                }
            )
    return pd.DataFrame(rows)


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
    frame, feature_meta, resolved_sources = read_feature_cache(config, specs)
    baseline_candidates = materialize_candidates(frame, specs, resolved_sources)
    variants = parse_beam_variants(config)
    variant_names = [variant.name for variant in variants]

    generated_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    input_sha: dict[str, str] = {}
    for well, group in frame[["id", "well", "row_idx"]].groupby("well", sort=True):
        generated, quality, sha_values = generate_well_beams(
            str(well),
            group.sort_values("row_idx", kind="stable"),
            paths.train_data_dir,
            variants,
        )
        generated_frames.append(generated)
        quality_rows.extend(quality)
        input_sha.update(sha_values)

    generated_all = pd.concat(generated_frames, ignore_index=True)
    frame = frame.merge(generated_all, on=["id", "well", "row_idx"], how="left", validate="1:1")
    if frame[variant_names].isna().any().any():
        missing = frame[variant_names].columns[frame[variant_names].isna().any()].tolist()
        raise ValueError(f"generated beam candidates contain missing values: {missing}")

    primary_baseline = str(get_nested(config, "audit.primary_baseline") or "likpf_mean")
    score_names = [*baseline_candidates, *variant_names]
    candidate_metrics = score_columns(frame, score_names, primary_baseline)
    bucket = bucket_metrics(frame, score_names)
    by_well = by_well_metrics(frame, score_names, primary_baseline)
    groups = group_metrics(frame, score_names, config)
    regressions = by_well.groupby("candidate", observed=True)[
        "delta_rmse_vs_primary_baseline"
    ].max()
    candidate_metrics["max_well_regression_vs_primary"] = candidate_metrics["candidate"].map(
        regressions
    )
    quality_frame = pd.DataFrame(quality_rows)

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    quality_path = artifacts / f"{OUTPUT_PREFIX}_beam_quality.csv"
    wide_path = artifacts / f"{OUTPUT_PREFIX}_candidate_wide.csv.gz"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    groups.to_csv(group_path, index=False)
    quality_frame.to_csv(quality_path, index=False)
    keep_columns = [
        "id",
        "well",
        "row_idx",
        "target",
        "true_tvt",
        "last_known_tvt",
        "md_since",
        "md",
        "z",
        *baseline_candidates,
        *variant_names,
    ]
    keep_columns = [column for column in keep_columns if column in frame.columns]
    frame[keep_columns].to_csv(wide_path, index=False, compression="gzip")

    best = candidate_metrics.iloc[0].to_dict()
    primary = (
        candidate_metrics[candidate_metrics["candidate"].eq(primary_baseline)].iloc[0].to_dict()
    )
    best_variant = (
        candidate_metrics[candidate_metrics["candidate"].isin(variant_names)].iloc[0].to_dict()
    )
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - start),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "feature_cache": feature_meta,
        "input_sha256": input_sha,
        "baseline_candidates": baseline_candidates,
        "beam_variant_names": variant_names,
        "primary_baseline": to_jsonable(primary),
        "best_candidate": to_jsonable(best),
        "best_beam_smoothness_variant": to_jsonable(best_variant),
        "decision": {
            "recommendation": "diagnostic_only_until_kaggle_result_review",
            "reason": (
                "This reruns Beam search with TVT+Z / dTVT+dZ penalties. "
                "Raw-test parity and near-prefix / worst-well guards must be reviewed "
                "before any inference port."
            ),
        },
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_path),
            "beam_quality": str(quality_path),
            "candidate_wide": str(wide_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_path),
            "beam_quality": sha256_path(quality_path),
            "candidate_wide_raw": sha256_path(wide_path),
            "candidate_wide_decompressed": sha256_path(wide_path, decompressed=True),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    best_delta = best_variant["delta_rmse_vs_primary_baseline"]
    metrics_json = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_not_run",
        "metric": "rmse",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "primary_baseline": to_jsonable(primary),
        "best_beam_smoothness_variant": to_jsonable(best_variant),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "summary_path": str(summary_path),
        "notes": (
            "Best generated Beam variant improved primary baseline; review guards before inference."
            if best_delta is not None and float(best_delta) < 0.0
            else "Best generated Beam variant did not improve primary baseline; no "
            "inference port until review."
        ),
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(
        json.dumps(
            to_jsonable(result["best_beam_smoothness_variant"]),
            indent=2,
            sort_keys=True,
        )
    )
