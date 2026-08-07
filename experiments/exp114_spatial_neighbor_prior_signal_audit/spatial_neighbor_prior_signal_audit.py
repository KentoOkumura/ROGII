from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp114_spatial_neighbor_prior_signal_audit"
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"


@dataclass(frozen=True)
class ClusterMethod:
    name: str
    method: str
    threshold: str


@dataclass(frozen=True)
class NeighborVariant:
    name: str
    features: tuple[str, ...]
    top_k: int
    require_same_typewell_group: bool


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


def parse_row_idx(ids: pd.Series) -> np.ndarray:
    values = ids.astype(str).str.rsplit("_", n=1).str[-1]
    return pd.to_numeric(values, errors="raise").to_numpy(np.int32)


def float_tag(value: float) -> str:
    text = f"{float(value):.5g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def read_feature_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP099_FEATURE_CACHE,
        get_nested(config, "data.exp099_train_feature_cache_local"),
    )
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "last_anchor_tvt",
        "md_since",
        "eval_len",
    ]
    optional = ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    usecols = required + [column for column in optional if column in header]
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
    frame["row_idx"] = parse_row_idx(frame["id"])
    for column in frame.columns:
        if column not in {"id", "well", "row_idx"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["true_tvt"] = frame["last_known_tvt"] + frame["target"]
    frame["true_delta_from_anchor"] = frame["true_tvt"] - frame["last_known_tvt"]

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


def read_cluster_assignments(
    config: dict[str, Any],
    method: ClusterMethod,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    source = find_artifact(
        EXP065_CLUSTER_ASSIGNMENTS,
        get_nested(config, "data.exp065_cluster_assignments_local"),
    )
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
    subset = frame[
        (frame["method"].astype(str) == method.method)
        & (frame["threshold"].astype(str) == method.threshold)
    ].copy()
    well_to_cluster = dict(zip(subset["well_id"], subset["cluster_id"], strict=False))
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "selected_method": method.__dict__,
        "selected_rows": int(len(subset)),
        "selected_clusters": int(subset["cluster_id"].nunique()),
    }
    return frame, metadata, well_to_cluster


def parse_cluster_method(config: dict[str, Any]) -> ClusterMethod:
    raw = get_nested(config, "model.neighbor_prior.cluster_method") or {}
    return ClusterMethod(
        name=str(raw.get("name", "native_overlap_0p999")),
        method=str(raw.get("method", "native_overlap")),
        threshold=str(raw.get("threshold", "0.999")),
    )


def parse_variants(config: dict[str, Any]) -> list[NeighborVariant]:
    raw_variants = get_nested(config, "model.neighbor_prior.variants") or []
    variants: list[NeighborVariant] = []
    for raw in raw_variants:
        features = tuple(str(item) for item in raw.get("features", []))
        if not features:
            raise ValueError(f"neighbor variant has no features: {raw}")
        variants.append(
            NeighborVariant(
                name=str(raw["name"]),
                features=features,
                top_k=int(raw.get("top_k", 8)),
                require_same_typewell_group=bool(raw.get("require_same_typewell_group", False)),
            )
        )
    if not variants:
        raise ValueError("model.neighbor_prior.variants must not be empty")
    return variants


def circular_abs_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(math.atan2(math.sin(a - b), math.cos(a - b))))


def safe_span(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def path_tortuosity(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if finite.sum() < 2:
        return float("nan")
    pts = np.column_stack([x[finite], y[finite], z[finite]]).astype(np.float64)
    step = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)).sum()
    chord = float(np.sqrt(np.sum((pts[-1] - pts[0]) ** 2)))
    if chord <= 0.0:
        return float("nan")
    return float(step / chord)


def augment_geometry(
    frame: pd.DataFrame,
    paths: ExperimentPaths,
    well_to_cluster: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    md = np.full(len(frame), np.nan, dtype=np.float32)
    x = np.full(len(frame), np.nan, dtype=np.float32)
    y = np.full(len(frame), np.nan, dtype=np.float32)
    z = np.full(len(frame), np.nan, dtype=np.float32)
    gr = np.full(len(frame), np.nan, dtype=np.float32)
    last_md = np.full(len(frame), np.nan, dtype=np.float32)
    last_x = np.full(len(frame), np.nan, dtype=np.float32)
    last_y = np.full(len(frame), np.nan, dtype=np.float32)
    last_z = np.full(len(frame), np.nan, dtype=np.float32)
    last_row_idx = np.full(len(frame), -1, dtype=np.int32)
    summary_rows: list[dict[str, Any]] = []
    missing_wells: list[str] = []
    md_since_diffs: list[float] = []

    for well, group in frame.groupby("well", sort=False):
        horizontal_path = paths.train_data_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            missing_wells.append(str(well))
            continue
        horizontal = pd.read_csv(
            horizontal_path,
            usecols=["MD", "X", "Y", "Z", "GR", "TVT_input"],
            low_memory=False,
        )
        row_idx = group["row_idx"].to_numpy(np.int64)
        if row_idx.max(initial=-1) >= len(horizontal):
            raise IndexError(f"row_idx exceeds horizontal rows for {well}: {row_idx.max()}")
        positions = group.index.to_numpy(np.int64)
        md_values = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float32)
        x_values = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(np.float32)
        y_values = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(np.float32)
        z_values = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float32)
        gr_values = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float32)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
        known_positions = np.flatnonzero(np.isfinite(tvt_input))
        if len(known_positions) == 0:
            missing_wells.append(str(well))
            continue
        anchor_idx = int(known_positions[-1])

        md[positions] = md_values[row_idx]
        x[positions] = x_values[row_idx]
        y[positions] = y_values[row_idx]
        z[positions] = z_values[row_idx]
        gr[positions] = gr_values[row_idx]
        last_md[positions] = md_values[anchor_idx]
        last_x[positions] = x_values[anchor_idx]
        last_y[positions] = y_values[anchor_idx]
        last_z[positions] = z_values[anchor_idx]
        last_row_idx[positions] = anchor_idx

        cache_md_since = pd.to_numeric(group["md_since"], errors="coerce").to_numpy(np.float32)
        geom_md_since = md_values[row_idx] - md_values[anchor_idx]
        finite = np.isfinite(cache_md_since) & np.isfinite(geom_md_since)
        if finite.any():
            md_since_diffs.append(
                float(np.nanmax(np.abs(cache_md_since[finite] - geom_md_since[finite])))
            )

        segment_start = int(np.nanmin(row_idx))
        segment_end = int(np.nanmax(row_idx))
        segment_idx = np.arange(segment_start, segment_end + 1, dtype=np.int64)
        sx = x_values[segment_idx]
        sy = y_values[segment_idx]
        sz = z_values[segment_idx]
        smd = md_values[segment_idx]
        start_idx = segment_idx[0]
        end_idx = segment_idx[-1]
        dx = float(x_values[end_idx] - x_values[start_idx])
        dy = float(y_values[end_idx] - y_values[start_idx])
        dz = float(z_values[end_idx] - z_values[start_idx])
        dmd = float(md_values[end_idx] - md_values[start_idx])
        azimuth = float(math.atan2(dy, dx)) if np.isfinite(dx) and np.isfinite(dy) else float("nan")
        local_end = min(anchor_idx + 10, len(horizontal) - 1)
        ldx = float(x_values[local_end] - x_values[anchor_idx])
        ldy = float(y_values[local_end] - y_values[anchor_idx])
        local_azimuth = (
            float(math.atan2(ldy, ldx)) if np.isfinite(ldx) and np.isfinite(ldy) else azimuth
        )
        prefix_tvt = tvt_input[known_positions]
        summary_rows.append(
            {
                "well": str(well),
                "typewell_cluster": well_to_cluster.get(str(well), ""),
                "rows": int(len(group)),
                "anchor_row_idx": anchor_idx,
                "eval_start_row_idx": segment_start,
                "eval_end_row_idx": segment_end,
                "centroid_x": float(np.nanmean(sx)),
                "centroid_y": float(np.nanmean(sy)),
                "centroid_z": float(np.nanmean(sz)),
                "start_x": float(x_values[start_idx]),
                "start_y": float(y_values[start_idx]),
                "start_z": float(z_values[start_idx]),
                "end_x": float(x_values[end_idx]),
                "end_y": float(y_values[end_idx]),
                "end_z": float(z_values[end_idx]),
                "bbox_dx": safe_span(sx),
                "bbox_dy": safe_span(sy),
                "bbox_dz": safe_span(sz),
                "md_span": float(np.nanmax(smd) - np.nanmin(smd)),
                "z_span": dz,
                "dz_dmd": dz / dmd if abs(dmd) > 1.0e-9 else float("nan"),
                "azimuth": azimuth,
                "azimuth_sin": float(math.sin(azimuth)) if np.isfinite(azimuth) else float("nan"),
                "azimuth_cos": float(math.cos(azimuth)) if np.isfinite(azimuth) else float("nan"),
                "local_azimuth": local_azimuth,
                "local_azimuth_sin": (
                    float(math.sin(local_azimuth)) if np.isfinite(local_azimuth) else float("nan")
                ),
                "local_azimuth_cos": (
                    float(math.cos(local_azimuth)) if np.isfinite(local_azimuth) else float("nan")
                ),
                "tortuosity": path_tortuosity(sx, sy, sz),
                "prefix_tvt_range": safe_span(prefix_tvt),
                "last_md": float(md_values[anchor_idx]),
                "last_x": float(x_values[anchor_idx]),
                "last_y": float(y_values[anchor_idx]),
                "last_z": float(z_values[anchor_idx]),
            }
        )

    out = frame.copy()
    out["md"] = md
    out["x"] = x
    out["y"] = y
    out["z"] = z
    out["horizontal_gr"] = gr
    out["last_md"] = last_md
    out["last_x"] = last_x
    out["last_y"] = last_y
    out["last_z"] = last_z
    out["last_row_idx"] = last_row_idx
    out["md_delta"] = out["md"] - out["last_md"]
    out["z_delta"] = out["z"] - out["last_z"]
    summaries = pd.DataFrame(summary_rows)
    metadata = {
        "missing_wells": missing_wells,
        "missing_well_count": len(missing_wells),
        "max_abs_md_since_diff_vs_cache": max(md_since_diffs) if md_since_diffs else None,
        "md_delta_finite_rate": float(np.isfinite(out["md_delta"].to_numpy(np.float32)).mean()),
        "xy_finite_rate": float(
            (
                np.isfinite(out["x"].to_numpy(np.float32))
                & np.isfinite(out["y"].to_numpy(np.float32))
            ).mean()
        ),
        "horizontal_gr_finite_rate": float(
            np.isfinite(out["horizontal_gr"].to_numpy(np.float32)).mean()
        ),
        "summary_wells": int(len(summaries)),
    }
    return out, summaries, metadata


def groupkfold_wells(wells: np.ndarray, n_folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    wells = np.array(sorted(map(str, wells)))
    rng = np.random.default_rng(seed)
    shuffled = wells.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    all_wells = set(wells.tolist())
    splits: list[tuple[set[str], set[str]]] = []
    for valid in folds:
        valid_set = set(valid.tolist())
        splits.append((all_wells.difference(valid_set), valid_set))
    return splits


def build_well_arrays(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for well, group in frame.groupby("well", sort=False):
        order = np.argsort(pd.to_numeric(group["md_since"], errors="coerce").to_numpy(np.float32))
        arrays[str(well)] = {
            "index": group.index.to_numpy(np.int64)[order],
            "md_since": numeric_array(group, "md_since")[order],
            "true_delta": numeric_array(group, "true_delta_from_anchor")[order],
        }
    return arrays


def interp_neighbor_delta(
    query_md: np.ndarray,
    neighbor_md: np.ndarray,
    neighbor_delta: np.ndarray,
    *,
    require_in_range: bool,
) -> np.ndarray:
    finite = np.isfinite(neighbor_md) & np.isfinite(neighbor_delta)
    if finite.sum() < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    x = neighbor_md[finite].astype(np.float64)
    y = neighbor_delta[finite].astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if len(x) < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    left = np.nan if require_in_range else float(y[0])
    right = np.nan if require_in_range else float(y[-1])
    return np.interp(query_md.astype(np.float64), x, y, left=left, right=right).astype(np.float32)


def standardized_matrix(
    summaries: pd.DataFrame,
    wells: list[str],
    features: tuple[str, ...],
    *,
    mean: pd.Series | None = None,
    std: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    subset = summaries.set_index("well").reindex(wells)
    values = subset.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0, ddof=0).replace(0.0, np.nan)
    filled = values.fillna(mean).fillna(0.0)
    scaled = (filled - mean.fillna(0.0)) / std.fillna(1.0)
    return scaled.to_numpy(np.float64), mean, std


def select_neighbors(
    summaries: pd.DataFrame,
    train_wells: set[str],
    query_well: str,
    variant: NeighborVariant,
) -> tuple[list[str], np.ndarray]:
    summary_by_well = summaries.set_index("well", drop=False)
    if query_well not in summary_by_well.index:
        return [], np.array([], dtype=np.float64)
    train_candidates = [well for well in sorted(train_wells) if well in summary_by_well.index]
    query_cluster = str(summary_by_well.loc[query_well, "typewell_cluster"])
    if variant.require_same_typewell_group:
        train_candidates = [
            well
            for well in train_candidates
            if query_cluster
            and str(summary_by_well.loc[well, "typewell_cluster"]) == query_cluster
        ]
    if len(train_candidates) == 0:
        return [], np.array([], dtype=np.float64)

    train_matrix, mean, std = standardized_matrix(summaries, train_candidates, variant.features)
    query_matrix, _, _ = standardized_matrix(
        summaries,
        [query_well],
        variant.features,
        mean=mean,
        std=std,
    )
    distances = np.sqrt(np.sum((train_matrix - query_matrix[0]) ** 2, axis=1))
    order = np.argsort(distances)
    keep = order[: max(variant.top_k, 1)]
    return [train_candidates[int(i)] for i in keep], distances[keep]


def generate_prior_for_variant(
    frame: pd.DataFrame,
    summaries: pd.DataFrame,
    variant: NeighborVariant,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_cfg = get_nested(config, "model.neighbor_prior") or {}
    min_neighbor_wells = int(prior_cfg.get("min_neighbor_wells", 2))
    min_row_neighbor_values = int(prior_cfg.get("min_row_neighbor_values", 1))
    require_in_range = bool(prior_cfg.get("require_in_range", True))
    distance_epsilon = float(prior_cfg.get("distance_epsilon", 1.0e-6))
    distance_power = float(prior_cfg.get("distance_power", 1.0))
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    seed = int(get_nested(config, "validation.seed") or 42)

    well_arrays = build_well_arrays(frame)
    summary_by_well = summaries.set_index("well", drop=False)
    prior_delta = np.full(len(frame), np.nan, dtype=np.float32)
    prior_std = np.full(len(frame), np.nan, dtype=np.float32)
    prior_count = np.zeros(len(frame), dtype=np.int16)
    prior_neighbor_wells = np.zeros(len(frame), dtype=np.int16)
    prior_distance_min = np.full(len(frame), np.nan, dtype=np.float32)
    prior_distance_mean = np.full(len(frame), np.nan, dtype=np.float32)
    prior_same_typewell_share = np.full(len(frame), np.nan, dtype=np.float32)
    prior_azimuth_mismatch = np.full(len(frame), np.nan, dtype=np.float32)
    prior_dz_dmd_mismatch = np.full(len(frame), np.nan, dtype=np.float32)
    neighbor_rows: list[dict[str, Any]] = []

    splits = groupkfold_wells(frame["well"].unique(), n_folds, seed)
    for fold, (train_wells, valid_wells) in enumerate(splits):
        for well in sorted(valid_wells):
            if well not in well_arrays or well not in summary_by_well.index:
                continue
            neighbors, distances = select_neighbors(summaries, train_wells, well, variant)
            if len(neighbors) < min_neighbor_wells:
                continue
            query = well_arrays[well]
            row_idx = query["index"]
            query_md = query["md_since"]
            neighbor_values: list[np.ndarray] = []
            usable_neighbors: list[str] = []
            usable_distances: list[float] = []
            for neighbor, distance in zip(neighbors, distances, strict=False):
                if neighbor not in well_arrays:
                    continue
                neighbor_data = well_arrays[neighbor]
                values = interp_neighbor_delta(
                    query_md,
                    neighbor_data["md_since"],
                    neighbor_data["true_delta"],
                    require_in_range=require_in_range,
                )
                if np.isfinite(values).any():
                    neighbor_values.append(values)
                    usable_neighbors.append(neighbor)
                    usable_distances.append(float(distance))
            if len(usable_neighbors) < min_neighbor_wells:
                continue
            stacked = np.vstack(neighbor_values).astype(np.float32)
            dist_arr = np.asarray(usable_distances, dtype=np.float64)
            weights = 1.0 / np.power(dist_arr + distance_epsilon, distance_power)
            finite = np.isfinite(stacked)
            weighted = np.where(finite, stacked.astype(np.float64) * weights[:, None], 0.0)
            weight_sum = np.where(finite, weights[:, None], 0.0).sum(axis=0)
            counts = finite.sum(axis=0)
            valid_rows = (counts >= min_row_neighbor_values) & (weight_sum > 0.0)
            if valid_rows.any():
                prior_delta[row_idx[valid_rows]] = (
                    weighted[:, valid_rows].sum(axis=0) / weight_sum[valid_rows]
                ).astype(np.float32)
                prior_std[row_idx[valid_rows]] = np.nanstd(
                    stacked[:, valid_rows],
                    axis=0,
                ).astype(np.float32)
            prior_count[row_idx] = counts.astype(np.int16)
            prior_neighbor_wells[row_idx] = len(usable_neighbors)
            prior_distance_min[row_idx] = float(np.nanmin(dist_arr))
            prior_distance_mean[row_idx] = float(np.nanmean(dist_arr))

            query_summary = summary_by_well.loc[well]
            neighbor_summary = summary_by_well.loc[usable_neighbors]
            same_typewell = (
                neighbor_summary["typewell_cluster"].astype(str).to_numpy()
                == str(query_summary["typewell_cluster"])
            )
            azimuth_mismatch = [
                circular_abs_diff(float(query_summary["azimuth"]), float(value))
                for value in neighbor_summary["azimuth"].to_numpy()
            ]
            dz_mismatch = np.abs(
                pd.to_numeric(neighbor_summary["dz_dmd"], errors="coerce").to_numpy(np.float64)
                - float(query_summary["dz_dmd"])
            )
            same_share = float(np.mean(same_typewell)) if len(same_typewell) else float("nan")
            az_mismatch_mean = float(np.nanmean(azimuth_mismatch))
            dz_mismatch_mean = float(np.nanmean(dz_mismatch))
            prior_same_typewell_share[row_idx] = same_share
            prior_azimuth_mismatch[row_idx] = az_mismatch_mean
            prior_dz_dmd_mismatch[row_idx] = dz_mismatch_mean
            neighbor_rows.append(
                {
                    "variant": variant.name,
                    "fold": fold,
                    "query_well": well,
                    "query_typewell_cluster": str(query_summary["typewell_cluster"]),
                    "neighbor_wells": " ".join(usable_neighbors),
                    "neighbor_count": int(len(usable_neighbors)),
                    "distance_min": float(np.nanmin(dist_arr)),
                    "distance_mean": float(np.nanmean(dist_arr)),
                    "same_typewell_share": same_share,
                    "azimuth_mismatch_mean": az_mismatch_mean,
                    "dz_dmd_mismatch_mean": dz_mismatch_mean,
                }
            )

    out = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            f"{variant.name}_prior_delta": prior_delta,
            f"{variant.name}_prior_tvt": numeric_array(frame, "last_known_tvt") + prior_delta,
            f"{variant.name}_prior_std": prior_std,
            f"{variant.name}_prior_count": prior_count,
            f"{variant.name}_neighbor_wells": prior_neighbor_wells,
            f"{variant.name}_distance_min": prior_distance_min,
            f"{variant.name}_distance_mean": prior_distance_mean,
            f"{variant.name}_same_typewell_share": prior_same_typewell_share,
            f"{variant.name}_azimuth_mismatch": prior_azimuth_mismatch,
            f"{variant.name}_dz_dmd_mismatch": prior_dz_dmd_mismatch,
        }
    )
    return out, pd.DataFrame(neighbor_rows)


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    pred_values = pred.astype(np.float64)
    true_values = true.astype(np.float64)
    mask = np.isfinite(pred_values) & np.isfinite(true_values)
    if not mask.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    error = pred_values[mask] - true_values[mask]
    return {
        "rows": int(mask.sum()),
        "coverage": float(mask.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
        "bias": float(np.mean(error)),
    }


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
    method_by_candidate: dict[str, str],
) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    rows: list[dict[str, Any]] = []
    for candidate in candidate_columns:
        if candidate not in frame.columns:
            continue
        score = score_prediction(numeric_array(frame, candidate), true)
        rows.append(
            {
                "candidate": candidate,
                "variant": method_by_candidate.get(candidate, "baseline"),
                "candidate_type": (
                    "baseline" if method_by_candidate.get(candidate, "baseline") == "baseline"
                    else "spatial_neighbor_prior"
                ),
                **score,
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"], na_position="last")


def compute_signal_metrics(
    frame: pd.DataFrame,
    variants: list[NeighborVariant],
    base_candidates: list[str],
) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt").astype(np.float64)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        prior_col = f"{variant.name}_prior_tvt"
        if prior_col not in frame.columns:
            continue
        prior = numeric_array(frame, prior_col).astype(np.float64)
        for base in base_candidates:
            if base not in frame.columns:
                continue
            base_values = numeric_array(frame, base).astype(np.float64)
            base_error = true - base_values
            prior_delta = prior - base_values
            mask = np.isfinite(base_error) & np.isfinite(prior_delta)
            if not mask.any():
                continue
            corr = (
                float(np.corrcoef(base_error[mask], prior_delta[mask])[0, 1])
                if mask.sum() > 1 and np.nanstd(prior_delta[mask]) > 0.0
                else None
            )
            nonzero = mask & (np.abs(base_error) > 1.0e-9) & (np.abs(prior_delta) > 1.0e-9)
            sign_match = (
                float(np.mean(np.sign(base_error[nonzero]) == np.sign(prior_delta[nonzero])))
                if nonzero.any()
                else None
            )
            prior_error = prior - true
            base_pred_error = base_values - true
            rows.append(
                {
                    "variant": variant.name,
                    "base_candidate": base,
                    "rows": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "corr_true_minus_base_vs_prior_minus_base": corr,
                    "sign_match_rate": sign_match,
                    "prior_beats_base_rate": float(
                        np.mean(np.abs(prior_error[mask]) < np.abs(base_pred_error[mask]))
                    ),
                    "mean_prior_minus_base": float(np.mean(prior_delta[mask])),
                    "mean_abs_prior_minus_base": float(np.mean(np.abs(prior_delta[mask]))),
                    "p95_abs_prior_minus_base": float(
                        np.quantile(np.abs(prior_delta[mask]), 0.95)
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_bucket_metrics(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    work = frame[["true_tvt", "md_since", *candidate_columns]].copy()
    work["distance_bucket"] = _distance_bucket(work["md_since"])
    rows: list[dict[str, Any]] = []
    true = numeric_array(work, "true_tvt")
    for candidate in candidate_columns:
        if candidate not in work.columns:
            continue
        pred = numeric_array(work, candidate)
        for bucket, idx in work.groupby("distance_bucket", observed=False).groups.items():
            positions = np.array(list(idx), dtype=np.int64)
            score = score_prediction(pred[positions], true[positions])
            if score["rows"] == 0:
                continue
            rows.append({"candidate": candidate, "distance_bucket": str(bucket), **score})
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        for candidate in candidate_columns:
            if candidate not in group.columns:
                continue
            score = score_prediction(numeric_array(group, candidate), true)
            if score["rows"] == 0:
                continue
            rows.append({"well": str(well), "candidate": candidate, **score})
    return pd.DataFrame(rows)


def add_corrected_candidates(
    frame: pd.DataFrame,
    variants: list[NeighborVariant],
    config: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    prior_cfg = get_nested(config, "model.neighbor_prior") or {}
    base_candidates = [str(value) for value in prior_cfg.get("base_candidates", ["likpf_mean"])]
    alphas = [float(value) for value in prior_cfg.get("correction_alphas", [0.1])]
    clips = [float(value) for value in prior_cfg.get("correction_clip_ft", [20.0])]
    candidate_columns = [
        str(column)
        for column in prior_cfg.get("score_baselines", [])
        if str(column) in frame.columns
    ]
    method_by_candidate = {candidate: "baseline" for candidate in candidate_columns}

    for variant in variants:
        prior_col = f"{variant.name}_prior_tvt"
        if prior_col not in frame.columns:
            continue
        candidate_columns.append(prior_col)
        method_by_candidate[prior_col] = variant.name
        prior_values = numeric_array(frame, prior_col)
        for base in base_candidates:
            if base not in frame.columns:
                continue
            base_values = numeric_array(frame, base)
            diff = prior_values - base_values
            for alpha in alphas:
                for clip in clips:
                    name = (
                        f"{variant.name}_{base}_corr_a{float_tag(alpha)}_c{float_tag(clip)}"
                    )
                    corrected = base_values.copy()
                    valid = np.isfinite(base_values) & np.isfinite(diff)
                    corrected[valid] = base_values[valid] + alpha * np.clip(
                        diff[valid],
                        -clip,
                        clip,
                    )
                    frame[name] = corrected.astype(np.float32)
                    candidate_columns.append(name)
                    method_by_candidate[name] = variant.name

    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidate_columns:
        if candidate in frame.columns and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped, method_by_candidate


def write_feature_schema(path: Path, columns: list[str]) -> None:
    schema = pd.DataFrame(
        {
            "variant": "spatial_neighbor_prior_signal_audit",
            "feature_index": np.arange(len(columns), dtype=int),
            "feature": columns,
        }
    )
    schema.to_csv(path, index=False)


def run_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    cluster_method = parse_cluster_method(config)
    _, cluster_meta, well_to_cluster = read_cluster_assignments(config, cluster_method)
    variants = parse_variants(config)
    frame, feature_meta = read_feature_cache(config)
    frame, well_summaries, geometry_meta = augment_geometry(frame, paths, well_to_cluster)

    prior_frames: list[pd.DataFrame] = []
    neighbor_summary_frames: list[pd.DataFrame] = []
    for variant in variants:
        prior, neighbor_summary = generate_prior_for_variant(
            frame,
            well_summaries,
            variant,
            config,
        )
        prior_frames.append(prior)
        neighbor_summary_frames.append(neighbor_summary)

    work = frame.copy()
    for prior in prior_frames:
        extra_cols = [column for column in prior.columns if column not in {"id", "well"}]
        work = work.merge(prior[["id", "well", *extra_cols]], on=["id", "well"], how="left")

    candidate_columns, method_by_candidate = add_corrected_candidates(work, variants, config)
    prior_cfg = get_nested(config, "model.neighbor_prior") or {}
    base_candidates = [
        str(column)
        for column in prior_cfg.get("base_candidates", ["likpf_mean"])
        if str(column) in work.columns
    ]
    candidate_metrics = compute_metrics(
        work,
        candidate_columns,
        method_by_candidate=method_by_candidate,
    )
    signal_metrics = compute_signal_metrics(work, variants, base_candidates)
    bucket_metrics = compute_bucket_metrics(work, candidate_columns)
    by_well = compute_by_well(work, candidate_columns)
    neighbor_summary = (
        pd.concat(neighbor_summary_frames, ignore_index=True, sort=False)
        if neighbor_summary_frames
        else pd.DataFrame()
    )

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    signal_path = artifacts / f"{OUTPUT_PREFIX}_signal_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    neighbor_path = artifacts / f"{OUTPUT_PREFIX}_neighbor_summary.csv"
    well_summary_path = artifacts / f"{OUTPUT_PREFIX}_well_geometry_summary.csv"
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    signal_metrics.to_csv(signal_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    neighbor_summary.to_csv(neighbor_path, index=False)
    well_summaries.to_csv(well_summary_path, index=False)

    diagnostic_cols = [
        column
        for column in work.columns
        if column.endswith("_prior_tvt")
        or column.endswith("_prior_delta")
        or column.endswith("_prior_std")
        or column.endswith("_prior_count")
        or column.endswith("_neighbor_wells")
        or column.endswith("_distance_min")
        or column.endswith("_distance_mean")
        or column.endswith("_same_typewell_share")
        or column.endswith("_azimuth_mismatch")
        or column.endswith("_dz_dmd_mismatch")
    ]
    keep_columns = [
        "id",
        "well",
        "row_idx",
        "target",
        "true_tvt",
        "last_known_tvt",
        "last_anchor_tvt",
        "md_since",
        "eval_len",
        "md",
        "x",
        "y",
        "z",
        "last_md",
        "last_x",
        "last_y",
        "last_z",
        "md_delta",
        "z_delta",
        *diagnostic_cols,
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
    prior_coverage = {
        variant.name: {
            "prior_valid_rate": float(
                np.isfinite(numeric_array(work, f"{variant.name}_prior_tvt")).mean()
            ),
            "mean_row_neighbor_count": float(
                np.mean(numeric_array(work, f"{variant.name}_prior_count"))
            ),
            "mean_neighbor_wells": float(
                np.mean(numeric_array(work, f"{variant.name}_neighbor_wells"))
            ),
        }
        for variant in variants
        if f"{variant.name}_prior_tvt" in work.columns
    }
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(work)),
        "wells": int(work["well"].nunique()),
        "feature_cache": feature_meta,
        "cluster_assignments": cluster_meta,
        "geometry": geometry_meta,
        "variants": [variant.__dict__ for variant in variants],
        "prior_coverage": prior_coverage,
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": (
            float(best["rmse"] - baseline_row["rmse"]) if best and baseline_row else None
        ),
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "signal_metrics": str(signal_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "neighbor_summary": str(neighbor_path),
            "well_geometry_summary": str(well_summary_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "signal_metrics": sha256_path(signal_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "neighbor_summary": sha256_path(neighbor_path),
            "well_geometry_summary": sha256_path(well_summary_path),
            "oof_predictions_raw": sha256_path(oof_path),
            "oof_predictions_decompressed": sha256_path(oof_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    metrics_json = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_pending_kaggle_train",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": summary["delta_best_minus_likpf_rmse"],
        "rows": int(len(work)),
        "wells": int(work["well"].nunique()),
        "summary_path": str(summary_path),
        "notes": "Implemented train-side audit. Awaiting Kaggle train execution.",
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_candidate"]), indent=2, sort_keys=True))
