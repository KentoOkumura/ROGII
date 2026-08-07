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

OUTPUT_PREFIX = "exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit"
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"


@dataclass(frozen=True)
class GroupMethod:
    name: str
    method: str
    threshold: str


@dataclass
class WellContext:
    well: str
    row_index: np.ndarray
    md: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    gr: np.ndarray
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


def stable_int(*parts: str, modulo: int | None = None) -> int:
    key = "|".join(parts).encode()
    value = int(hashlib.sha256(key).hexdigest()[:16], 16)
    return value if modulo is None else value % modulo


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
            / "exp065_typewell_supertype_cluster_cv_audit"
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
    return ids.astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32).to_numpy()


def read_feature_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = get_nested(config, "data.exp099_train_feature_cache_local")
    source = find_artifact(EXP099_FEATURE_CACHE, explicit)
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "last_anchor_tvt",
        "pf_ancc",
        "beam_mean",
        "likpf_mean",
        "sc_ens",
        "hyb",
        "eval_len",
        "md_since",
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


def read_cluster_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = get_nested(config, "data.exp065_cluster_assignments_local")
    source = find_artifact(EXP065_CLUSTER_ASSIGNMENTS, explicit)
    frame = pd.read_csv(source, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    frame["cluster_size"] = (
        pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, metadata


def parse_group_methods(config: dict[str, Any]) -> list[GroupMethod]:
    raw_methods = get_nested(config, "model.gr_transfer.group_methods") or []
    methods = [
        GroupMethod(
            name=str(raw["name"]),
            method=str(raw["method"]),
            threshold=str(raw["threshold"]),
        )
        for raw in raw_methods
    ]
    if not methods:
        raise ValueError("model.gr_transfer.group_methods must not be empty")
    return methods


def make_group_lookup(
    assignments: pd.DataFrame,
    method: GroupMethod,
    *,
    min_cluster_size: int,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    subset = assignments[
        (assignments["method"].astype(str) == method.method)
        & (assignments["threshold"].astype(str) == method.threshold)
        & (assignments["cluster_size"] >= min_cluster_size)
    ].copy()
    well_to_cluster = dict(zip(subset["well_id"], subset["cluster_id"], strict=False))
    cluster_to_wells = {
        cluster: sorted(group["well_id"].astype(str).tolist())
        for cluster, group in subset.groupby("cluster_id", sort=False)
    }
    cluster_sizes = {cluster: len(wells) for cluster, wells in cluster_to_wells.items()}
    return well_to_cluster, cluster_to_wells, cluster_sizes


def groupkfold_wells(wells: np.ndarray, n_folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    wells = np.array(sorted(map(str, wells)))
    rng = np.random.default_rng(seed)
    shuffled = wells.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    all_wells = set(wells.tolist())
    return [(all_wells.difference(set(valid.tolist())), set(valid.tolist())) for valid in folds]


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
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, usecols=["MD", "TVT", "TVT_input", "GR"])
        raw_rows = len(raw)
        row_index = np.arange(raw_rows, dtype=np.int32)
        well_frame = frame[frame["well"] == well]
        eval_row_index = well_frame["row_index"].to_numpy(np.int32)
        eval_md_since = well_frame["md_since"].to_numpy(np.float32)
        eval_start = int(np.nanmin(eval_row_index))
        anchor_row = max(eval_start - 1, 0)
        tvt_input = pd.to_numeric(raw["TVT_input"], errors="coerce").to_numpy(np.float32)
        finite_prefix = np.flatnonzero(np.isfinite(tvt_input[:eval_start]))
        if finite_prefix.size:
            anchor_row = int(finite_prefix[-1])
        tvt = pd.to_numeric(raw["TVT"], errors="coerce").to_numpy(np.float32)
        md = pd.to_numeric(raw["MD"], errors="coerce").to_numpy(np.float32)
        gr = pd.to_numeric(raw["GR"], errors="coerce").to_numpy(np.float32)
        contexts[well] = WellContext(
            well=well,
            row_index=row_index,
            md=md,
            tvt=tvt,
            tvt_input=tvt_input,
            gr=gr,
            anchor_row_index=anchor_row,
            anchor_md=float(md[anchor_row]) if 0 <= anchor_row < raw_rows else np.nan,
            anchor_tvt=float(tvt_input[anchor_row]) if 0 <= anchor_row < raw_rows else np.nan,
            eval_start_row_index=eval_start,
            eval_row_index=eval_row_index,
            eval_md_since=eval_md_since,
            raw_rows=raw_rows,
        )
    metadata = {
        "train_dir": str(train_dir),
        "requested_wells": int(len(frame["well"].unique())),
        "loaded_wells": int(len(contexts)),
        "loaded_raw_rows": int(sum(ctx.raw_rows for ctx in contexts.values())),
    }
    return contexts, metadata


def candidate_centers(start: int, stop: int, radius: int, stride: int) -> np.ndarray:
    left = max(start, radius)
    right = max(left, stop - radius)
    centers = np.arange(left, right, stride, dtype=np.int32)
    if centers.size == 0 and right > left:
        centers = np.array([(left + right) // 2], dtype=np.int32)
    return centers


def normalized_windows(
    gr: np.ndarray,
    centers: np.ndarray,
    radius: int,
    *,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = 2 * radius + 1
    out = np.zeros((len(centers), width), dtype=np.float32)
    valid = np.zeros(len(centers), dtype=bool)
    valid_frac = np.zeros(len(centers), dtype=np.float32)
    for i, center in enumerate(centers):
        left = int(center) - radius
        right = int(center) + radius + 1
        if left < 0 or right > len(gr):
            continue
        window = gr[left:right].astype(np.float32, copy=True)
        finite = np.isfinite(window)
        frac = float(finite.mean())
        valid_frac[i] = frac
        if frac < min_valid_fraction or finite.sum() < 3:
            continue
        mean = float(np.nanmean(window))
        filled = np.where(finite, window, mean)
        centered = filled - float(filled.mean())
        norm = float(np.linalg.norm(centered))
        if norm <= 1e-6:
            continue
        out[i] = centered / norm
        valid[i] = True
    return out[valid], centers[valid], valid_frac[valid]


def local_slope(context: WellContext, centers: np.ndarray, radius: int) -> np.ndarray:
    left = np.maximum(centers - radius, 0)
    right = np.minimum(centers + radius, len(context.md) - 1)
    dmd = context.md[right] - context.md[left]
    dtvt = context.tvt[right] - context.tvt[left]
    slope = np.divide(
        dtvt, dmd, out=np.full(len(centers), np.nan, dtype=np.float32), where=np.abs(dmd) > 1e-6
    )
    return slope.astype(np.float32)


def weighted_mean(
    values: np.ndarray, scores: np.ndarray, min_score: float
) -> tuple[float, int, float]:
    finite = np.isfinite(values) & np.isfinite(scores) & (scores >= min_score)
    if not finite.any():
        return np.nan, 0, np.nan
    selected_values = values[finite]
    selected_scores = scores[finite]
    weights = np.maximum(selected_scores - min_score + 1e-3, 1e-3)
    return (
        float(np.average(selected_values, weights=weights)),
        int(finite.sum()),
        float(np.max(selected_scores)),
    )


def interpolate_samples(
    target_md_since: np.ndarray,
    sample_md_since: np.ndarray,
    sample_values: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(sample_md_since) & np.isfinite(sample_values)
    if finite.sum() < 2:
        return np.full(len(target_md_since), np.nan, dtype=np.float32)
    x = sample_md_since[finite].astype(np.float64)
    y = sample_values[finite].astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if len(x) < 2:
        return np.full(len(target_md_since), np.nan, dtype=np.float32)
    return np.interp(target_md_since.astype(np.float64), x, y, left=np.nan, right=np.nan).astype(
        np.float32
    )


def match_query_to_source(
    query: WellContext,
    source: WellContext,
    query_centers: np.ndarray,
    *,
    radius: int,
    source_stride: int,
    min_valid_fraction: float,
    chunk_size: int,
    random_control: bool,
) -> dict[str, np.ndarray]:
    source_stop = max(source.eval_start_row_index, 0)
    source_centers = candidate_centers(0, source_stop, radius, source_stride)
    if source_centers.size < 2:
        n = len(query_centers)
        return {
            "score": np.full(n, np.nan, dtype=np.float32),
            "offset_delta": np.full(n, np.nan, dtype=np.float32),
            "slope_delta": np.full(n, np.nan, dtype=np.float32),
            "path_delta": np.full(n, np.nan, dtype=np.float32),
        }
    source_windows, source_centers, _ = normalized_windows(
        source.gr,
        source_centers,
        radius,
        min_valid_fraction=min_valid_fraction,
    )
    query_windows, valid_query_centers, _ = normalized_windows(
        query.gr,
        query_centers,
        radius,
        min_valid_fraction=min_valid_fraction,
    )
    n_query = len(query_centers)
    score = np.full(n_query, np.nan, dtype=np.float32)
    offset_delta = np.full(n_query, np.nan, dtype=np.float32)
    slope_delta = np.full(n_query, np.nan, dtype=np.float32)
    path_delta = np.full(n_query, np.nan, dtype=np.float32)
    if len(source_windows) == 0 or len(query_windows) == 0:
        return {
            "score": score,
            "offset_delta": offset_delta,
            "slope_delta": slope_delta,
            "path_delta": path_delta,
        }

    query_pos = {int(center): pos for pos, center in enumerate(query_centers)}
    valid_positions = np.array(
        [query_pos[int(center)] for center in valid_query_centers], dtype=np.int32
    )
    source_slope = local_slope(source, source_centers, radius)
    source_delta = source.tvt[source_centers] - np.float32(source.anchor_tvt)
    source_md_offset = source.md[source_centers] - np.float32(source.anchor_md)
    query_md = np.interp(valid_query_centers, query.row_index, query.md).astype(np.float32)
    query_md_since = query_md - np.float32(query.anchor_md)

    if random_control:
        for local_i, pos in enumerate(valid_positions):
            source_i = stable_int(
                query.well, source.well, str(int(query_centers[pos])), modulo=len(source_centers)
            )
            score[pos] = 0.0
            slope = source_slope[source_i]
            offset_delta[pos] = source_delta[source_i]
            slope_delta[pos] = query_md_since[local_i] * slope if np.isfinite(slope) else np.nan
            path_delta[pos] = (
                source_delta[source_i]
                + (query_md_since[local_i] - source_md_offset[source_i]) * slope
                if np.isfinite(slope)
                else np.nan
            )
        return {
            "score": score,
            "offset_delta": offset_delta,
            "slope_delta": slope_delta,
            "path_delta": path_delta,
        }

    for start in range(0, len(query_windows), chunk_size):
        stop = min(start + chunk_size, len(query_windows))
        sims = query_windows[start:stop] @ source_windows.T
        best_idx = np.argmax(sims, axis=1)
        best_score = sims[np.arange(stop - start), best_idx]
        positions = valid_positions[start:stop]
        score[positions] = best_score.astype(np.float32)
        slope = source_slope[best_idx]
        offset = source_delta[best_idx]
        md_offset = source_md_offset[best_idx]
        qmd_since = query_md_since[start:stop]
        offset_delta[positions] = offset.astype(np.float32)
        slope_delta[positions] = (qmd_since * slope).astype(np.float32)
        path_delta[positions] = (offset + (qmd_since - md_offset) * slope).astype(np.float32)
    return {
        "score": score,
        "offset_delta": offset_delta,
        "slope_delta": slope_delta,
        "path_delta": path_delta,
    }


def aggregate_source_matches(
    query: WellContext,
    sources: list[WellContext],
    *,
    mode: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    transfer_cfg = get_nested(config, "model.gr_transfer") or {}
    radius = int(transfer_cfg.get("window_radius_rows", 32))
    query_stride = int(transfer_cfg.get("query_stride_rows", 8))
    source_stride = int(transfer_cfg.get("source_stride_rows", 8))
    min_valid_fraction = float(transfer_cfg.get("min_gr_valid_fraction", 0.7))
    min_score = float(transfer_cfg.get("min_match_score", 0.15))
    top_k_sources = int(transfer_cfg.get("top_k_sources", 5))
    chunk_size = int(transfer_cfg.get("chunk_size", 192))
    max_sources = int(transfer_cfg.get("max_source_wells", 12))
    random_control = mode == "same_typewell_random_control"
    selected_sources = sources[:max_sources]
    query_centers = candidate_centers(
        int(np.nanmin(query.eval_row_index)),
        int(np.nanmax(query.eval_row_index)) + 1,
        radius,
        query_stride,
    )
    if query_centers.size == 0 or len(selected_sources) == 0:
        return pd.DataFrame({"row_index": query.eval_row_index})

    per_source = [
        match_query_to_source(
            query,
            source,
            query_centers,
            radius=radius,
            source_stride=source_stride,
            min_valid_fraction=min_valid_fraction,
            chunk_size=chunk_size,
            random_control=random_control,
        )
        for source in selected_sources
    ]
    sample_offset = np.full(len(query_centers), np.nan, dtype=np.float32)
    sample_slope = np.full(len(query_centers), np.nan, dtype=np.float32)
    sample_path = np.full(len(query_centers), np.nan, dtype=np.float32)
    sample_score = np.full(len(query_centers), np.nan, dtype=np.float32)
    sample_count = np.zeros(len(query_centers), dtype=np.int16)
    sample_source_count = np.full(len(query_centers), len(selected_sources), dtype=np.int16)

    for i in range(len(query_centers)):
        scores = np.array([item["score"][i] for item in per_source], dtype=np.float32)
        order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:top_k_sources]
        selected_scores = scores[order]
        for key, target in [
            ("offset_delta", sample_offset),
            ("slope_delta", sample_slope),
            ("path_delta", sample_path),
        ]:
            values = np.array([per_source[j][key][i] for j in order], dtype=np.float32)
            target[i], count, best_score = weighted_mean(
                values, selected_scores, min_score if not random_control else -1.0
            )
            sample_count[i] = max(sample_count[i], count)
            if np.isfinite(best_score):
                sample_score[i] = (
                    best_score
                    if not np.isfinite(sample_score[i])
                    else max(sample_score[i], best_score)
                )

    target_md_since = query.eval_md_since.astype(np.float32)
    sample_md = np.interp(query_centers, query.row_index, query.md).astype(np.float32) - np.float32(
        query.anchor_md
    )
    offset_delta = interpolate_samples(target_md_since, sample_md, sample_offset)
    slope_delta = interpolate_samples(target_md_since, sample_md, sample_slope)
    path_delta = interpolate_samples(target_md_since, sample_md, sample_path)
    match_score = interpolate_samples(target_md_since, sample_md, sample_score)
    match_count = interpolate_samples(target_md_since, sample_md, sample_count.astype(np.float32))
    source_count = interpolate_samples(
        target_md_since, sample_md, sample_source_count.astype(np.float32)
    )
    return pd.DataFrame(
        {
            "row_index": query.eval_row_index,
            f"{mode}_offset_delta": offset_delta,
            f"{mode}_slope_delta": slope_delta,
            f"{mode}_path_delta": path_delta,
            f"{mode}_match_score": match_score,
            f"{mode}_match_count": match_count,
            f"{mode}_source_wells": source_count,
        }
    )


def source_wells_for_query(
    query_well: str,
    train_wells: set[str],
    well_to_cluster: dict[str, str],
    cluster_to_wells: dict[str, list[str]],
    contexts: dict[str, WellContext],
    *,
    mode: str,
    all_context_wells: list[str],
) -> list[WellContext]:
    cluster = well_to_cluster.get(query_well)
    if mode in {"same_typewell_gr_match", "same_typewell_random_control"}:
        if cluster is None:
            return []
        wells = [
            well
            for well in cluster_to_wells.get(cluster, [])
            if well in train_wells and well in contexts and well != query_well
        ]
    elif mode == "different_typewell_gr_match":
        wells = [
            well
            for well in all_context_wells
            if well in train_wells and well in contexts and well_to_cluster.get(well) != cluster
        ]
        start = (
            stable_int(query_well, "different_typewell", modulo=max(len(wells), 1)) if wells else 0
        )
        wells = wells[start:] + wells[:start]
    else:
        raise ValueError(f"unknown transfer mode: {mode}")
    return [contexts[well] for well in wells]


def generate_transfer_features_for_method(
    frame: pd.DataFrame,
    contexts: dict[str, WellContext],
    assignments: pd.DataFrame,
    method: GroupMethod,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transfer_cfg = get_nested(config, "model.gr_transfer") or {}
    min_cluster_size = int(transfer_cfg.get("min_cluster_size", 2))
    modes = list(transfer_cfg.get("modes", ["same_typewell_gr_match"]))
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    seed = int(get_nested(config, "validation.seed") or 42)
    well_to_cluster, cluster_to_wells, cluster_sizes = make_group_lookup(
        assignments,
        method,
        min_cluster_size=min_cluster_size,
    )
    splits = groupkfold_wells(frame["well"].unique(), n_folds, seed)
    outputs: list[pd.DataFrame] = []
    all_context_wells = sorted(contexts)
    mode_source_rows: list[dict[str, Any]] = []
    for fold, (train_wells, valid_wells) in enumerate(splits):
        for query_well in sorted(valid_wells):
            if query_well not in contexts:
                continue
            query = contexts[query_well]
            query_out = pd.DataFrame({"well": query_well, "row_index": query.eval_row_index})
            for mode in modes:
                sources = source_wells_for_query(
                    query_well,
                    train_wells,
                    well_to_cluster,
                    cluster_to_wells,
                    contexts,
                    mode=mode,
                    all_context_wells=all_context_wells,
                )
                mode_source_rows.append(
                    {
                        "fold": fold,
                        "well": query_well,
                        "mode": mode,
                        "source_wells": len(sources),
                        "cluster": well_to_cluster.get(query_well),
                        "cluster_size": cluster_sizes.get(well_to_cluster.get(query_well, ""), 0),
                    }
                )
                mode_out = aggregate_source_matches(query, sources, mode=mode, config=config)
                query_out = query_out.merge(mode_out, on="row_index", how="left")
            outputs.append(query_out)
    if not outputs:
        return pd.DataFrame(columns=["well", "row_index"]), {"source_summary": []}
    transfer = pd.concat(outputs, ignore_index=True)
    rename = {
        column: f"{method.name}_{column}"
        for column in transfer.columns
        if column not in {"well", "row_index"}
    }
    transfer = transfer.rename(columns=rename)
    source_summary = pd.DataFrame(mode_source_rows)
    metadata = {
        "method": method.__dict__,
        "source_summary": source_summary.to_dict(orient="records"),
        "mean_source_wells_by_mode": (
            source_summary.groupby("mode")["source_wells"].mean().to_dict()
            if len(source_summary)
            else {}
        ),
    }
    return transfer, metadata


def add_transfer_candidates(
    frame: pd.DataFrame,
    methods: list[GroupMethod],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    transfer_cfg = get_nested(config, "model.gr_transfer") or {}
    base_candidates = list(transfer_cfg.get("base_candidates", ["likpf_mean"]))
    alphas = [float(value) for value in transfer_cfg.get("correction_alphas", [0.1, 0.2])]
    clips = [float(value) for value in transfer_cfg.get("correction_clip_ft", [10.0, 20.0])]
    modes = list(transfer_cfg.get("modes", ["same_typewell_gr_match"]))
    delta_kinds = list(transfer_cfg.get("delta_kinds", ["path", "slope", "offset"]))
    candidate_columns: list[str] = list(transfer_cfg.get("score_baselines", []))
    source_by_candidate = {candidate: "baseline" for candidate in candidate_columns}
    new_columns: dict[str, np.ndarray] = {}

    for method in methods:
        for mode in modes:
            for delta_kind in delta_kinds:
                delta_col = f"{method.name}_{mode}_{delta_kind}_delta"
                score_col = f"{method.name}_{mode}_match_score"
                count_col = f"{method.name}_{mode}_match_count"
                if delta_col not in frame.columns:
                    continue
                prior_col = f"{method.name}_{mode}_{delta_kind}_tvt"
                prior_values = numeric_array(frame, "last_known_tvt") + numeric_array(
                    frame, delta_col
                )
                new_columns[prior_col] = prior_values.astype(np.float32)
                candidate_columns.append(prior_col)
                source_by_candidate[prior_col] = f"{method.name}:{mode}"
                for base in base_candidates:
                    if base not in frame.columns:
                        continue
                    base_values = numeric_array(frame, base)
                    diff = prior_values - base_values
                    valid = (
                        np.isfinite(diff)
                        & np.isfinite(numeric_array(frame, score_col))
                        & np.isfinite(numeric_array(frame, count_col))
                        & (numeric_array(frame, count_col) >= 1.0)
                    )
                    for alpha in alphas:
                        alpha_tag = str(alpha).replace(".", "p")
                        for clip in clips:
                            clip_tag = (
                                str(int(clip))
                                if float(clip).is_integer()
                                else str(clip).replace(".", "p")
                            )
                            name = (
                                f"{method.name}_{mode}_{delta_kind}_{base}"
                                f"_corr_a{alpha_tag}_c{clip_tag}"
                            )
                            corrected = base_values.copy()
                            corrected[valid] = base_values[valid] + alpha * np.clip(
                                diff[valid], -clip, clip
                            )
                            new_columns[name] = corrected.astype(np.float32)
                            candidate_columns.append(name)
                            source_by_candidate[name] = f"{method.name}:{mode}"

    if new_columns:
        frame = pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)

    seen: set[str] = set()
    deduped = []
    for candidate in candidate_columns:
        if candidate in frame.columns and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return frame, deduped, source_by_candidate


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
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
        rows.append(
            {
                "candidate": candidate,
                "source": source_by_candidate.get(candidate, "baseline"),
                "rows": int(mask.sum()),
                "coverage": float(mask.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "within10": float(np.mean(np.abs(error) <= 10.0)),
                "bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)


def compute_bucket_metrics(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    work = frame[["true_tvt", "md_since"] + candidate_columns].copy()
    work["distance_bucket"] = _distance_bucket(work["md_since"])
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
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt").astype(np.float64)
        baseline_pred = (
            numeric_array(group, "likpf_mean").astype(np.float64) if "likpf_mean" in group else None
        )
        baseline_rmse = None
        if baseline_pred is not None:
            mask = np.isfinite(true) & np.isfinite(baseline_pred)
            if mask.any():
                baseline_rmse = float(np.sqrt(np.mean((baseline_pred[mask] - true[mask]) ** 2)))
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
                    if baseline_rmse is not None
                    else np.nan,
                    "mae": float(np.mean(np.abs(error))),
                    "within10": float(np.mean(np.abs(error) <= 10.0)),
                    "bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def compute_signal_metrics(
    frame: pd.DataFrame, methods: list[GroupMethod], config: dict[str, Any]
) -> pd.DataFrame:
    transfer_cfg = get_nested(config, "model.gr_transfer") or {}
    modes = list(transfer_cfg.get("modes", ["same_typewell_gr_match"]))
    delta_kinds = list(transfer_cfg.get("delta_kinds", ["path", "slope", "offset"]))
    rows: list[dict[str, Any]] = []
    true_minus_base = numeric_array(frame, "true_tvt") - numeric_array(frame, "likpf_mean")
    for method in methods:
        for mode in modes:
            for delta_kind in delta_kinds:
                prior_col = f"{method.name}_{mode}_{delta_kind}_tvt"
                score_col = f"{method.name}_{mode}_match_score"
                count_col = f"{method.name}_{mode}_match_count"
                if prior_col not in frame.columns:
                    continue
                prior_minus_base = numeric_array(frame, prior_col) - numeric_array(
                    frame, "likpf_mean"
                )
                mask = np.isfinite(prior_minus_base) & np.isfinite(true_minus_base)
                corr = (
                    float(np.corrcoef(prior_minus_base[mask], true_minus_base[mask])[0, 1])
                    if mask.sum() >= 2
                    else np.nan
                )
                rows.append(
                    {
                        "method": method.name,
                        "mode": mode,
                        "delta_kind": delta_kind,
                        "rows": int(mask.sum()),
                        "coverage": float(mask.mean()),
                        "prior_minus_likpf_corr_with_error": corr,
                        "sign_match": float(
                            np.mean(
                                np.sign(prior_minus_base[mask]) == np.sign(true_minus_base[mask])
                            )
                        )
                        if mask.any()
                        else np.nan,
                        "mean_match_score": float(np.nanmean(numeric_array(frame, score_col)))
                        if score_col in frame
                        else np.nan,
                        "mean_match_count": float(np.nanmean(numeric_array(frame, count_col)))
                        if count_col in frame
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": "same_typewell_other_horizontal_prefix_gr_transfer_audit",
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
    assignments, cluster_meta = read_cluster_assignments(config)
    methods = parse_group_methods(config)
    contexts, context_meta = read_horizontal_contexts(
        frame,
        paths.train_data_dir,
        max_wells=get_nested(config, "audit.max_wells"),
    )

    work = frame.copy()
    transfer_meta = []
    for method in methods:
        transfer, metadata = generate_transfer_features_for_method(
            work,
            contexts,
            assignments,
            method,
            config,
        )
        transfer_meta.append(metadata)
        if len(transfer):
            work = work.merge(transfer, on=["well", "row_index"], how="left")

    work, candidate_columns, source_by_candidate = add_transfer_candidates(work, methods, config)
    candidate_metrics = compute_metrics(
        work, candidate_columns, source_by_candidate=source_by_candidate
    )
    bucket_metrics = compute_bucket_metrics(work, candidate_columns)
    by_well = compute_by_well(work, candidate_columns)
    signal_metrics = compute_signal_metrics(work, methods, config)

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    signal_path = artifacts / f"{OUTPUT_PREFIX}_signal_metrics.csv"
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    signal_metrics.to_csv(signal_path, index=False)
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
        *[
            column
            for column in work.columns
            if "_same_typewell_" in column
            or "_different_typewell_" in column
            or column.endswith("_match_score")
            or column.endswith("_match_count")
            or column.endswith("_source_wells")
        ],
        *candidate_columns,
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
        "cluster_assignments": cluster_meta,
        "horizontal_contexts": context_meta,
        "group_methods": [method.__dict__ for method in methods],
        "transfer_metadata": to_jsonable(transfer_meta),
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": (
            float(best["rmse"] - baseline_row["rmse"]) if best and baseline_row else None
        ),
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "signal_metrics": str(signal_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "signal_metrics": sha256_path(signal_path),
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
