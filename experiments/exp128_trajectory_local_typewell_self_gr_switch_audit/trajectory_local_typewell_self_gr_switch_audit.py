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

OUTPUT_PREFIX = "exp128_trajectory_local_typewell_self_gr_switch_audit"
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)


@dataclass
class WellContext:
    well: str
    row_index: np.ndarray
    md: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    gr: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    anchor_row_index: int
    anchor_md: float
    anchor_tvt: float
    eval_start_row_index: int
    eval_row_index: np.ndarray
    eval_md_since: np.ndarray
    raw_rows: int


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
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("artifacts") / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "artifacts"
            / filename,
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


def parse_row_index(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce")
    if values.isna().any():
        examples = ids[values.isna()].head(5).tolist()
        raise ValueError(f"Could not parse row index from id examples: {examples}")
    return values.astype(np.int32).to_numpy()


def read_feature_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = get_nested(config, "data.exp099_train_feature_cache_local")
    source = find_artifact(EXP099_FEATURE_CACHE, explicit)
    base_candidates = list(get_nested(config, "model.base_candidates") or ["likpf_mean"])
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "last_anchor_tvt",
        "eval_len",
        "md_since",
        *base_candidates,
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=required,
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["row_index"] = parse_row_index(frame["id"])
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["true_tvt"] = frame["last_known_tvt"] + frame["target"]
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(
            EXP099_FEATURE_SCHEMA,
            get_nested(config, "data.exp099_train_feature_schema_local"),
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
    }
    return frame, metadata


def read_typewell(typewell_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not typewell_path.exists():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    raw = pd.read_csv(typewell_path, usecols=lambda col: col in {"TVT", "GR"})
    if not {"TVT", "GR"} <= set(raw.columns):
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    raw = raw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if raw.empty:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    tvt = pd.to_numeric(raw["TVT"], errors="coerce").to_numpy(np.float32)
    gr = pd.to_numeric(raw["GR"], errors="coerce").to_numpy(np.float32)
    finite = np.isfinite(tvt) & np.isfinite(gr)
    tvt = tvt[finite]
    gr = gr[finite]
    if len(tvt) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    unique_tvt, unique_idx = np.unique(tvt, return_index=True)
    return unique_tvt.astype(np.float32), gr[unique_idx].astype(np.float32)


def read_horizontal_contexts(
    frame: pd.DataFrame,
    train_dir: Path,
    *,
    max_wells: int | None = None,
) -> tuple[dict[str, WellContext], dict[str, Any]]:
    contexts: dict[str, WellContext] = {}
    wells = sorted(frame["well"].astype(str).unique())
    if max_wells is not None:
        wells = wells[:max_wells]
    for well in wells:
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            continue
        raw = pd.read_csv(horizontal_path, usecols=["MD", "TVT", "TVT_input", "GR"])
        raw_rows = len(raw)
        if raw_rows == 0:
            continue
        well_frame = frame[frame["well"] == well].sort_values("row_index")
        eval_row_index = well_frame["row_index"].to_numpy(np.int32)
        if eval_row_index.size == 0:
            continue
        eval_start = int(np.nanmin(eval_row_index))
        tvt_input = pd.to_numeric(raw["TVT_input"], errors="coerce").to_numpy(np.float32)
        finite_prefix = np.flatnonzero(np.isfinite(tvt_input[:eval_start]))
        if not finite_prefix.size:
            continue
        anchor_row = int(finite_prefix[-1])
        tvt = pd.to_numeric(raw["TVT"], errors="coerce").to_numpy(np.float32)
        md = pd.to_numeric(raw["MD"], errors="coerce").to_numpy(np.float32)
        gr = pd.to_numeric(raw["GR"], errors="coerce").to_numpy(np.float32)
        typewell_tvt, typewell_gr = read_typewell(train_dir / f"{well}__typewell.csv")
        contexts[well] = WellContext(
            well=well,
            row_index=np.arange(raw_rows, dtype=np.int32),
            md=md,
            tvt=tvt,
            tvt_input=tvt_input,
            gr=gr,
            typewell_tvt=typewell_tvt,
            typewell_gr=typewell_gr,
            anchor_row_index=anchor_row,
            anchor_md=float(md[anchor_row]),
            anchor_tvt=float(tvt_input[anchor_row]),
            eval_start_row_index=eval_start,
            eval_row_index=eval_row_index,
            eval_md_since=well_frame["md_since"].to_numpy(np.float32),
            raw_rows=raw_rows,
        )
    metadata = {
        "train_dir": str(train_dir),
        "requested_wells": int(frame["well"].nunique()),
        "loaded_wells": int(len(contexts)),
        "loaded_raw_rows": int(sum(ctx.raw_rows for ctx in contexts.values())),
        "loaded_typewells": int(
            sum(1 for ctx in contexts.values() if len(ctx.typewell_tvt) >= 2)
        ),
    }
    return contexts, metadata


def candidate_centers(start: int, stop: int, radius: int, stride: int) -> np.ndarray:
    left = max(start, radius)
    right = min(stop - radius, stop)
    if right <= left:
        return np.array([], dtype=np.int32)
    centers = np.arange(left, right, stride, dtype=np.int32)
    if centers.size == 0:
        centers = np.array([(left + right) // 2], dtype=np.int32)
    return centers


def normalized_window(
    values: np.ndarray,
    indices: np.ndarray,
    min_valid_fraction: float,
) -> np.ndarray | None:
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(values):
        return None
    window = values[indices].astype(np.float32, copy=True)
    finite = np.isfinite(window)
    if float(finite.mean()) < min_valid_fraction or finite.sum() < 3:
        return None
    mean = float(np.nanmean(window))
    filled = np.where(finite, window, mean)
    centered = filled - float(filled.mean())
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-6:
        return None
    return (centered / norm).astype(np.float32)


def prefix_windows(
    context: WellContext,
    radius: int,
    stride: int,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = candidate_centers(0, context.eval_start_row_index, radius, stride)
    windows: list[np.ndarray] = []
    valid_centers: list[int] = []
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    for center in centers:
        window = normalized_window(context.gr, center + offsets, min_valid_fraction)
        if window is None:
            continue
        if not np.isfinite(context.tvt_input[center]):
            continue
        windows.append(window)
        valid_centers.append(int(center))
    if not windows:
        return np.empty((0, 2 * radius + 1), dtype=np.float32), np.array([], dtype=np.int32)
    return np.stack(windows).astype(np.float32), np.asarray(valid_centers, dtype=np.int32)


def local_prefix_slope(context: WellContext, centers: np.ndarray, radius: int) -> np.ndarray:
    slopes = np.full(len(centers), np.nan, dtype=np.float32)
    for i, center in enumerate(centers):
        left = max(int(center) - radius, 0)
        right = min(int(center) + radius, context.eval_start_row_index - 1)
        prefix_idx = np.arange(left, right + 1, dtype=np.int32)
        finite = np.isfinite(context.tvt_input[prefix_idx]) & np.isfinite(context.md[prefix_idx])
        if finite.sum() < 3:
            continue
        x = context.md[prefix_idx][finite].astype(np.float64)
        y = context.tvt_input[prefix_idx][finite].astype(np.float64)
        if float(np.nanmax(x) - np.nanmin(x)) <= 1e-6:
            continue
        slopes[i] = float(np.polyfit(x, y, 1)[0])
    return slopes


def row_positions_for_center(row_index: np.ndarray, center: int, radius: int) -> np.ndarray:
    return np.flatnonzero(np.abs(row_index.astype(np.int64) - int(center)) <= radius)


def typewell_cost_for_center(
    context: WellContext,
    well_frame: pd.DataFrame,
    candidate_values: np.ndarray,
    center: int,
    radius: int,
    min_valid_fraction: float,
) -> float:
    if len(context.typewell_tvt) < 2:
        return np.nan
    positions = row_positions_for_center(well_frame["row_index"].to_numpy(np.int32), center, radius)
    if len(positions) < 3:
        return np.nan
    row_idx = well_frame["row_index"].to_numpy(np.int32)[positions]
    query = normalized_window(context.gr, row_idx, min_valid_fraction)
    if query is None:
        return np.nan
    candidate_tvt = candidate_values[positions].astype(np.float32)
    finite = np.isfinite(candidate_tvt)
    if float(finite.mean()) < min_valid_fraction or finite.sum() < 3:
        return np.nan
    interp_gr = np.interp(
        candidate_tvt.astype(np.float64),
        context.typewell_tvt.astype(np.float64),
        context.typewell_gr.astype(np.float64),
        left=np.nan,
        right=np.nan,
    ).astype(np.float32)
    reference = normalized_window(
        interp_gr,
        np.arange(len(interp_gr), dtype=np.int32),
        min_valid_fraction,
    )
    if reference is None:
        return np.nan
    return float(np.sqrt(np.mean(np.square(query - reference))))


def self_match_for_center(
    context: WellContext,
    query_center: int,
    query_md: float,
    prefix_matrix: np.ndarray,
    prefix_centers: np.ndarray,
    prefix_slopes: np.ndarray,
    *,
    radius: int,
    min_valid_fraction: float,
) -> dict[str, float]:
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    query = normalized_window(context.gr, int(query_center) + offsets, min_valid_fraction)
    if query is None or len(prefix_matrix) == 0:
        return {
            "self_cost": np.nan,
            "self_score": np.nan,
            "self_center": np.nan,
            "self_prior_tvt": np.nan,
            "self_delta_from_anchor": np.nan,
        }
    sims = prefix_matrix @ query
    best_idx = int(np.argmax(sims))
    best_center = int(prefix_centers[best_idx])
    best_score = float(sims[best_idx])
    self_cost = float(np.sqrt(max(0.0, 2.0 - 2.0 * best_score)))
    slope = float(prefix_slopes[best_idx])
    if not np.isfinite(slope):
        slope = 0.0
    prior_tvt = float(
        context.tvt_input[best_center]
        + slope * (float(query_md) - context.md[best_center])
    )
    return {
        "self_cost": self_cost,
        "self_score": best_score,
        "self_center": float(best_center),
        "self_prior_tvt": prior_tvt,
        "self_delta_from_anchor": prior_tvt - float(context.anchor_tvt),
    }


def interpolate_samples(
    target_row_index: np.ndarray,
    sample_row_index: np.ndarray,
    sample_values: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(sample_row_index) & np.isfinite(sample_values)
    if finite.sum() < 2:
        return np.full(len(target_row_index), np.nan, dtype=np.float32)
    x = sample_row_index[finite].astype(np.float64)
    y = sample_values[finite].astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    if len(unique_x) < 2:
        return np.full(len(target_row_index), np.nan, dtype=np.float32)
    return np.interp(
        target_row_index.astype(np.float64),
        unique_x,
        y[unique_idx],
        left=np.nan,
        right=np.nan,
    ).astype(np.float32)


def generate_local_switch_features_for_well(
    well_frame: pd.DataFrame,
    context: WellContext,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_cfg = get_nested(config, "model.local_switch") or {}
    radius = int(local_cfg.get("window_radius_rows", 32))
    query_stride = int(local_cfg.get("query_stride_rows", 8))
    prefix_stride = int(local_cfg.get("prefix_stride_rows", 4))
    min_valid_fraction = float(local_cfg.get("min_gr_valid_fraction", 0.70))
    typewell_candidate = str(local_cfg.get("typewell_cost_candidate", "likpf_mean"))
    row_index = well_frame["row_index"].to_numpy(np.int32)
    centers = candidate_centers(
        int(row_index.min()),
        int(row_index.max()) + 1,
        radius,
        query_stride,
    )
    prefix_matrix, prefix_centers = prefix_windows(
        context,
        radius=radius,
        stride=prefix_stride,
        min_valid_fraction=min_valid_fraction,
    )
    prefix_slopes = local_prefix_slope(context, prefix_centers, radius)
    candidate_values = numeric_array(well_frame, typewell_candidate)
    center_rows: list[dict[str, Any]] = []
    for center in centers:
        row_pos = int(np.argmin(np.abs(row_index.astype(np.int64) - int(center))))
        query_md = float(context.md[int(center)]) if 0 <= int(center) < len(context.md) else np.nan
        typewell_cost = typewell_cost_for_center(
            context,
            well_frame,
            candidate_values,
            int(center),
            radius,
            min_valid_fraction,
        )
        self_match = self_match_for_center(
            context,
            int(center),
            query_md,
            prefix_matrix,
            prefix_centers,
            prefix_slopes,
            radius=radius,
            min_valid_fraction=min_valid_fraction,
        )
        center_rows.append(
            {
                "well": context.well,
                "center_row_index": int(center),
                "nearest_oof_row_index": int(row_index[row_pos]),
                "typewell_cost": typewell_cost,
                **self_match,
                "cost_gap_typewell_minus_self": typewell_cost - self_match["self_cost"]
                if np.isfinite(typewell_cost) and np.isfinite(self_match["self_cost"])
                else np.nan,
                "prefix_window_count": int(len(prefix_centers)),
            }
        )
    if center_rows:
        centers_df = pd.DataFrame(center_rows)
    else:
        centers_df = pd.DataFrame(
            columns=[
                "well",
                "center_row_index",
                "nearest_oof_row_index",
                "typewell_cost",
                "self_cost",
                "self_score",
                "self_center",
                "self_prior_tvt",
                "self_delta_from_anchor",
                "cost_gap_typewell_minus_self",
                "prefix_window_count",
            ]
        )
    out = pd.DataFrame({"well": context.well, "row_index": row_index})
    for column in [
        "typewell_cost",
        "self_cost",
        "self_score",
        "self_center",
        "self_prior_tvt",
        "self_delta_from_anchor",
        "cost_gap_typewell_minus_self",
        "prefix_window_count",
    ]:
        out[column] = interpolate_samples(
            row_index,
            centers_df["center_row_index"].to_numpy(np.float32)
            if len(centers_df)
            else np.array([], dtype=np.float32),
            pd.to_numeric(centers_df[column], errors="coerce").to_numpy(np.float32)
            if len(centers_df)
            else np.array([], dtype=np.float32),
        )
    return out, centers_df


def generate_local_switch_features(
    frame: pd.DataFrame,
    contexts: dict[str, WellContext],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_wells = get_nested(config, "audit.max_wells")
    wells = sorted(frame["well"].unique())
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    feature_parts: list[pd.DataFrame] = []
    center_parts: list[pd.DataFrame] = []
    for well in wells:
        if well not in contexts:
            continue
        well_frame = frame[frame["well"] == well].sort_values("row_index").reset_index(drop=True)
        features, centers = generate_local_switch_features_for_well(
            well_frame,
            contexts[well],
            config,
        )
        feature_parts.append(features)
        center_parts.append(centers)
    feature_frame = (
        pd.concat(feature_parts, ignore_index=True)
        if feature_parts
        else pd.DataFrame(columns=["well", "row_index"])
    )
    center_frame = (
        pd.concat(center_parts, ignore_index=True)
        if center_parts
        else pd.DataFrame(columns=["well", "center_row_index"])
    )
    rename = {
        column: f"local_{column}"
        for column in feature_frame.columns
        if column not in {"well", "row_index"}
    }
    return feature_frame.rename(columns=rename), center_frame


def add_switch_candidates(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    local_cfg = get_nested(config, "model.local_switch") or {}
    base_candidates = list(get_nested(config, "model.base_candidates") or ["likpf_mean"])
    switch_gaps = [float(value) for value in local_cfg.get("switch_gap_thresholds", [0.20, 0.35])]
    blend_gaps = [float(value) for value in local_cfg.get("blend_gap_thresholds", [0.15, 0.25])]
    blend_weights = [float(value) for value in local_cfg.get("blend_weights", [0.25, 0.50])]
    max_self_cost = float(local_cfg.get("max_self_cost", 1.20))
    min_prefix_windows = float(local_cfg.get("min_prefix_window_count", 8))
    candidate_columns = [candidate for candidate in base_candidates if candidate in frame.columns]
    source_by_candidate = {candidate: "baseline" for candidate in candidate_columns}
    self_prior = numeric_array(frame, "local_self_prior_tvt")
    cost_gap = numeric_array(frame, "local_cost_gap_typewell_minus_self")
    self_cost = numeric_array(frame, "local_self_cost")
    prefix_count = numeric_array(frame, "local_prefix_window_count")
    valid_self = (
        np.isfinite(self_prior)
        & np.isfinite(cost_gap)
        & np.isfinite(self_cost)
        & (self_cost <= max_self_cost)
        & (prefix_count >= min_prefix_windows)
    )
    if np.isfinite(self_prior).any():
        frame["self_gr_prefix_prior_tvt"] = self_prior.astype(np.float32)
        candidate_columns.append("self_gr_prefix_prior_tvt")
        source_by_candidate["self_gr_prefix_prior_tvt"] = "self_gr_prefix_prior"
    for base in base_candidates:
        if base not in frame.columns:
            continue
        base_values = numeric_array(frame, base)
        for gap in switch_gaps:
            gap_tag = str(gap).replace(".", "p")
            switch_mask = valid_self & (cost_gap >= gap)
            name = f"{base}_local_self_gr_switch_gap{gap_tag}"
            pred = base_values.copy()
            pred[switch_mask] = self_prior[switch_mask]
            frame[name] = pred.astype(np.float32)
            frame[f"{name}_switch_flag"] = switch_mask.astype(np.int8)
            candidate_columns.append(name)
            source_by_candidate[name] = "local_self_gr_hard_switch"
        for gap in blend_gaps:
            gap_tag = str(gap).replace(".", "p")
            active = valid_self & (cost_gap >= gap)
            for weight in blend_weights:
                weight_tag = str(weight).replace(".", "p")
                name = f"{base}_local_self_gr_blend_gap{gap_tag}_w{weight_tag}"
                gate = np.zeros(len(frame), dtype=np.float32)
                denom = max(gap, 1e-6)
                gate[active] = np.clip((cost_gap[active] - gap) / denom, 0.0, 1.0) * weight
                pred = base_values.copy()
                positive_gate = gate > 0.0
                pred[positive_gate] = (
                    (1.0 - gate[positive_gate]) * base_values[positive_gate]
                    + gate[positive_gate] * self_prior[positive_gate]
                )
                frame[name] = pred.astype(np.float32)
                frame[f"{name}_gate"] = gate.astype(np.float32)
                candidate_columns.append(name)
                source_by_candidate[name] = "local_self_gr_soft_blend"
    seen: set[str] = set()
    deduped = []
    for candidate in candidate_columns:
        if candidate in frame.columns and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return frame, deduped, source_by_candidate


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def compute_metrics(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    *,
    source_by_candidate: dict[str, str],
) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt").astype(np.float64)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_columns:
        pred = numeric_array(frame, candidate).astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        if not mask.any():
            continue
        error = pred[mask] - true[mask]
        switch_cols = [c for c in [f"{candidate}_switch_flag", f"{candidate}_gate"] if c in frame]
        rows.append(
            {
                "candidate": candidate,
                "source": source_by_candidate.get(candidate, "unknown"),
                "rows": int(mask.sum()),
                "coverage": float(mask.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "within10": float(np.mean(np.abs(error) <= 10.0)),
                "bias": float(np.mean(error)),
                "mean_switch_or_gate": float(
                    np.nanmean(pd.concat([frame[col] for col in switch_cols], axis=1).sum(axis=1))
                )
                if switch_cols
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)


def compute_bucket_metrics(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    work = frame[["true_tvt", "md_since"] + candidate_columns].copy()
    work["distance_bucket"] = distance_bucket(work["md_since"])
    rows: list[dict[str, Any]] = []
    true = numeric_array(work, "true_tvt").astype(np.float64)
    for candidate in candidate_columns:
        pred = numeric_array(work, candidate).astype(np.float64)
        for bucket, idx in work.groupby("distance_bucket", observed=False).groups.items():
            positions = np.array(list(idx), dtype=np.int64)
            mask = np.isfinite(true[positions]) & np.isfinite(pred[positions])
            if not mask.any():
                continue
            error = pred[positions][mask] - true[positions][mask]
            rows.append(
                {
                    "candidate": candidate,
                    "distance_bucket": str(bucket),
                    "rows": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "within10": float(np.mean(np.abs(error) <= 10.0)),
                    "bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_name = "likpf_mean" if "likpf_mean" in frame.columns else candidate_columns[0]
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt").astype(np.float64)
        baseline_pred = numeric_array(group, baseline_name).astype(np.float64)
        baseline_mask = np.isfinite(true) & np.isfinite(baseline_pred)
        baseline_rmse = (
            float(np.sqrt(np.mean((baseline_pred[baseline_mask] - true[baseline_mask]) ** 2)))
            if baseline_mask.any()
            else np.nan
        )
        for candidate in candidate_columns:
            pred = numeric_array(group, candidate).astype(np.float64)
            mask = np.isfinite(true) & np.isfinite(pred)
            if not mask.any():
                continue
            error = pred[mask] - true[mask]
            rmse = float(np.sqrt(np.mean(error**2)))
            rows.append(
                {
                    "well": str(well),
                    "candidate": candidate,
                    "rows": int(mask.sum()),
                    "rmse": rmse,
                    "delta_vs_likpf_rmse": rmse - baseline_rmse
                    if np.isfinite(baseline_rmse)
                    else np.nan,
                    "mae": float(np.mean(np.abs(error))),
                    "within10": float(np.mean(np.abs(error) <= 10.0)),
                    "bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def compute_switch_signal_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "likpf_mean" not in frame.columns:
        return pd.DataFrame(rows)
    true_error = numeric_array(frame, "true_tvt") - numeric_array(frame, "likpf_mean")
    for signal in [
        "local_typewell_cost",
        "local_self_cost",
        "local_self_score",
        "local_cost_gap_typewell_minus_self",
        "local_self_prior_tvt",
    ]:
        if signal not in frame.columns:
            continue
        values = numeric_array(frame, signal)
        compare_values = (
            values - numeric_array(frame, "likpf_mean")
            if signal == "local_self_prior_tvt"
            else values
        )
        mask = np.isfinite(compare_values) & np.isfinite(true_error)
        corr = (
            float(np.corrcoef(compare_values[mask], true_error[mask])[0, 1])
            if mask.sum() >= 2
            else np.nan
        )
        rows.append(
            {
                "signal": signal,
                "rows": int(mask.sum()),
                "coverage": float(mask.mean()),
                "corr_with_likpf_error": corr,
                "mean": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                "p05": float(np.nanpercentile(values, 5)) if np.isfinite(values).any() else np.nan,
                "p50": float(np.nanpercentile(values, 50)) if np.isfinite(values).any() else np.nan,
                "p95": float(np.nanpercentile(values, 95)) if np.isfinite(values).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": "trajectory_local_typewell_self_gr_switch_audit",
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

    frame, feature_meta = read_feature_cache(config)
    contexts, context_meta = read_horizontal_contexts(
        frame,
        paths.train_data_dir,
        max_wells=get_nested(config, "audit.max_wells"),
    )
    feature_frame, window_diagnostics = generate_local_switch_features(frame, contexts, config)
    work = frame.merge(feature_frame, on=["well", "row_index"], how="inner")
    work, candidate_columns, source_by_candidate = add_switch_candidates(work, config)

    candidate_metrics = compute_metrics(
        work,
        candidate_columns,
        source_by_candidate=source_by_candidate,
    )
    bucket_metrics = compute_bucket_metrics(work, candidate_columns)
    by_well = compute_by_well(work, candidate_columns)
    signal_metrics = compute_switch_signal_metrics(work)

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    signal_path = artifacts / f"{OUTPUT_PREFIX}_signal_metrics.csv"
    window_path = artifacts / f"{OUTPUT_PREFIX}_window_diagnostics.csv"
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    signal_metrics.to_csv(signal_path, index=False)
    window_diagnostics.to_csv(window_path, index=False)
    keep_columns = [
        "id",
        "well",
        "row_index",
        "target",
        "true_tvt",
        "last_known_tvt",
        "last_anchor_tvt",
        "md_since",
        "eval_len",
        *[column for column in work.columns if column.startswith("local_")],
        *candidate_columns,
        *[
            column
            for column in work.columns
            if column.endswith("_switch_flag") or column.endswith("_gate")
        ],
    ]
    keep_columns = list(
        dict.fromkeys([column for column in keep_columns if column in work.columns])
    )
    work[keep_columns].to_csv(oof_path, index=False, compression="gzip")
    write_feature_schema(schema_path, keep_columns)

    best = candidate_metrics.iloc[0].to_dict() if len(candidate_metrics) else {}
    baseline = candidate_metrics[candidate_metrics["candidate"] == "likpf_mean"]
    baseline_row = baseline.iloc[0].to_dict() if len(baseline) else {}
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(work)),
        "wells": int(work["well"].nunique()),
        "feature_cache": feature_meta,
        "horizontal_contexts": context_meta,
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": (
            float(best["rmse"] - baseline_row["rmse"]) if best and baseline_row else None
        ),
        "switch_signal": {
            "finite_self_prior_rate": float(
                np.isfinite(numeric_array(work, "local_self_prior_tvt")).mean()
            )
            if "local_self_prior_tvt" in work
            else 0.0,
            "mean_cost_gap_typewell_minus_self": float(
                np.nanmean(numeric_array(work, "local_cost_gap_typewell_minus_self"))
            )
            if "local_cost_gap_typewell_minus_self" in work
            else None,
        },
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "signal_metrics": str(signal_path),
            "window_diagnostics": str(window_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "signal_metrics": sha256_path(signal_path),
            "window_diagnostics": sha256_path(window_path),
            "oof_predictions_raw": sha256_path(oof_path),
            "oof_predictions_decompressed": sha256_path(oof_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    metrics_json = {
        "status": "implemented_pending_kaggle_train",
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": summary["delta_best_minus_likpf_rmse"],
        "rows": int(len(work)),
        "wells": int(work["well"].nunique()),
        "summary_path": str(summary_path),
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_candidate"]), indent=2, sort_keys=True))
