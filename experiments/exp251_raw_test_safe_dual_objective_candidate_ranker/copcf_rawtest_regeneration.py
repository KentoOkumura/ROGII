from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hmm_exp226_candidate_selector_on_exp183 as parent
import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested


@dataclass(frozen=True)
class TypewellSeries:
    well: str
    gr_quantized: np.ndarray


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def _row_indices(ids: pd.Series) -> np.ndarray:
    values = ids.astype(str).str.rsplit("_", n=1).str[-1]
    return pd.to_numeric(values, errors="raise").to_numpy(np.int32)


def _safe_span(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite) - np.min(finite)) if len(finite) else float("nan")


def _path_tortuosity(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if finite.sum() < 2:
        return float("nan")
    points = np.column_stack([x[finite], y[finite], z[finite]]).astype(np.float64)
    path = float(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)).sum())
    chord = float(np.sqrt(np.sum((points[-1] - points[0]) ** 2)))
    return path / chord if chord > 0.0 else float("nan")


def _circular_abs_diff(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(abs(math.atan2(math.sin(left - right), math.cos(left - right))))


def _robust_scale(values: np.ndarray, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(floor)
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    median = float(np.median(finite))
    iqr_scale = float(q75 - q25) / 1.349
    mad_scale = float(np.median(np.abs(finite - median)) * 1.4826)
    return max(iqr_scale, mad_scale, float(floor))


def _interp_neighbor_delta(
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
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if len(x) < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    left = np.nan if require_in_range else float(y[0])
    right = np.nan if require_in_range else float(y[-1])
    return np.interp(query_md.astype(np.float64), x, y, left=left, right=right).astype(np.float32)


def _load_source_curves(
    config: dict[str, Any],
    excluded_wells: set[str],
    train_reference_frame: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    source = parent.find_artifact(
        parent.DEFAULT_TRAIN_FEATURE_CACHE,
        get_nested(config, "data.exp099_train_feature_cache_local"),
    )
    required = ["well", "md_since", "target", "last_known_tvt"]
    if train_reference_frame is None:
        frame = pd.read_csv(
            source,
            usecols=required,
            dtype={"well": str},
            low_memory=False,
        )
    else:
        missing = sorted(set(required).difference(train_reference_frame.columns))
        if missing:
            raise ValueError(f"train reference frame lacks prior-source columns: {missing}")
        frame = train_reference_frame.loc[:, required]
    frame = frame.loc[~frame["well"].astype(str).isin(excluded_wells)].copy()
    frame["well"] = frame["well"].astype(str)
    frame["true_delta"] = pd.to_numeric(frame["target"], errors="coerce")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for well, group in frame.groupby("well", sort=False):
        md = pd.to_numeric(group["md_since"], errors="coerce").to_numpy(np.float32)
        delta = pd.to_numeric(group["true_delta"], errors="coerce").to_numpy(np.float32)
        order = np.argsort(md)
        arrays[str(well)] = {"md_since": md[order], "true_delta": delta[order]}
    overlap = sorted(excluded_wells.intersection(arrays))
    if overlap:
        raise AssertionError(f"raw-test wells leaked into prior source arrays: {overlap}")
    compute_sha = bool(get_nested(config, "rawtest_copcf.compute_source_content_sha"))
    return arrays, {
        "path": str(source),
        "sha256": parent.sha256_path(source) if compute_sha else None,
        "decompressed_sha256": (
            parent.sha256_path(source, decompressed=True) if compute_sha else None
        ),
        "source_wells": len(arrays),
        "source_rows": int(len(frame)),
        "excluded_rawtest_wells": sorted(excluded_wells),
    }


def _load_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = parent.find_artifact(
        parent.EXP065_CLUSTER_ASSIGNMENTS,
        get_nested(config, "data.exp065_cluster_assignments_local"),
    )
    frame = pd.read_csv(source, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cluster assignments lack columns: {missing}")
    return frame, {"path": str(source), "sha256": parent.sha256_path(source)}


def _load_typewell(path: Path, well: str) -> TypewellSeries:
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    return TypewellSeries(
        well=str(well),
        gr_quantized=np.rint(frame["GR"].to_numpy(np.float64) * 100.0).astype(np.int64),
    )


def _rolling_hashes(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) < width:
        return np.empty(0, dtype=np.uint64)
    base = 1_000_003
    offset = 2_147_483_647
    mask = (1 << 64) - 1
    hashes = np.empty(len(values) - width + 1, dtype=np.uint64)
    power = 1
    for _ in range(width - 1):
        power = (power * base) & mask
    current = 0
    for idx in range(width):
        current = (current * base + int(values[idx]) + offset) & mask
    hashes[0] = current
    for idx in range(width, len(values)):
        left = int(values[idx - width]) + offset
        right = int(values[idx]) + offset
        current = ((current - left * power) * base + right) & mask
        hashes[idx - width + 1] = current
    return hashes


def _overlap_stats(
    train: TypewellSeries,
    query: TypewellSeries,
    lag_query_minus_train: int,
) -> dict[str, Any] | None:
    lag = int(lag_query_minus_train)
    train_start = max(0, -lag)
    query_start = train_start + lag
    overlap = min(
        len(train.gr_quantized) - train_start,
        len(query.gr_quantized) - query_start,
    )
    if overlap <= 0:
        return None
    left = train.gr_quantized[train_start : train_start + overlap]
    right = query.gr_quantized[query_start : query_start + overlap]
    return {
        "source_well": train.well,
        "row_lag_query_minus_source": lag,
        "overlap_rows": int(overlap),
        "overlap_fraction_shorter": float(
            overlap / max(min(len(train.gr_quantized), len(query.gr_quantized)), 1)
        ),
        "exact_match_rate": float(np.mean(left == right)),
    }


def _typewell_matches(
    query: TypewellSeries,
    source_series: dict[str, TypewellSeries],
    source_hashes: dict[str, np.ndarray],
    *,
    kgram_rows: int,
    min_kgram_hits: int,
    min_overlap_rows: int,
    min_overlap_fraction_shorter: float,
) -> list[dict[str, Any]]:
    query_hashes = _rolling_hashes(query.gr_quantized, kgram_rows)
    query_positions: dict[int, list[int]] = defaultdict(list)
    for position, value in enumerate(query_hashes):
        query_positions[int(value)].append(position)
    matches: list[dict[str, Any]] = []
    for well, train in source_series.items():
        lag_hits: Counter[int] = Counter()
        for train_position, value in enumerate(source_hashes[well]):
            for query_position in query_positions.get(int(value), ()):
                lag_hits[query_position - train_position] += 1
        if not lag_hits:
            continue
        for lag, hits in lag_hits.most_common(4):
            if hits < min_kgram_hits:
                continue
            stats = _overlap_stats(train, query, lag)
            if stats is None:
                continue
            if stats["overlap_rows"] < min_overlap_rows:
                continue
            if stats["overlap_fraction_shorter"] < min_overlap_fraction_shorter:
                continue
            stats["kgram_hits"] = int(hits)
            matches.append(stats)
            break
    return matches


def _assign_test_clusters(
    *,
    query_wells: list[str],
    source_wells: set[str],
    assignments: pd.DataFrame,
    train_dir: Path,
    test_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, str | None]], dict[str, Any]]:
    settings = get_nested(config, "rawtest_copcf.typewell_matching") or {}
    kgram_rows = int(settings.get("kgram_rows", 64))
    min_kgram_hits = int(settings.get("min_kgram_hits", 1))
    min_overlap_rows = int(settings.get("min_overlap_rows", 200))
    min_overlap_fraction = float(settings.get("min_overlap_fraction_shorter", 0.8))
    method = str(settings.get("method", "native_overlap"))
    threshold_map = {
        str(key): str(value)
        for key, value in (
            get_nested(config, "rawtest_copcf.typewell_prior_thresholds") or {}
        ).items()
    }
    thresholds = sorted(
        set(threshold_map.values())
        | {str(get_nested(config, "cluster.assignment_threshold") or "1")}
    )
    method_rows = assignments.loc[assignments["method"] == method].copy()
    allowed_source_wells = set(method_rows["well_id"].astype(str)).intersection(source_wells)
    source_series = {
        well: _load_typewell(train_dir / f"{well}__typewell.csv", well)
        for well in sorted(allowed_source_wells)
        if (train_dir / f"{well}__typewell.csv").exists()
    }
    forbidden_typewell_sources = sorted(set(query_wells).intersection(source_series))
    if forbidden_typewell_sources:
        raise AssertionError(
            f"raw-test wells leaked into typewell matching sources: {forbidden_typewell_sources}"
        )
    source_hashes = {
        well: _rolling_hashes(item.gr_quantized, kgram_rows) for well, item in source_series.items()
    }
    cluster_maps: dict[str, dict[str, str | None]] = {threshold: {} for threshold in thresholds}
    query_meta: dict[str, Any] = {}
    for query_well in query_wells:
        path = test_dir / f"{query_well}__typewell.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw-test typewell is missing: {path}")
        query = _load_typewell(path, query_well)
        matches = _typewell_matches(
            query,
            source_series,
            source_hashes,
            kgram_rows=kgram_rows,
            min_kgram_hits=min_kgram_hits,
            min_overlap_rows=min_overlap_rows,
            min_overlap_fraction_shorter=min_overlap_fraction,
        )
        query_meta[query_well] = {"candidate_match_count": len(matches), "thresholds": {}}
        for threshold in thresholds:
            subset = method_rows.loc[method_rows["threshold"].astype(str) == threshold]
            well_to_cluster = dict(
                zip(subset["well_id"].astype(str), subset["cluster_id"].astype(str), strict=False)
            )
            required_rate = float(threshold)
            eligible = [
                item
                for item in matches
                if float(item["exact_match_rate"]) + 1.0e-12 >= required_rate
                and item["source_well"] in well_to_cluster
            ]
            eligible.sort(
                key=lambda item: (
                    float(item["exact_match_rate"]),
                    int(item["overlap_rows"]),
                    int(item["kgram_hits"]),
                ),
                reverse=True,
            )
            clusters = sorted({well_to_cluster[item["source_well"]] for item in eligible})
            if len(clusters) > 1:
                raise ValueError(
                    f"raw-test typewell {query_well} bridges multiple "
                    f"{method}/{threshold} clusters: {clusters}"
                )
            cluster = clusters[0] if clusters else None
            cluster_maps[threshold][query_well] = cluster
            best = eligible[0] if eligible else None
            query_meta[query_well]["thresholds"][threshold] = {
                "cluster_id": cluster,
                "eligible_source_matches": len(eligible),
                "best_match": best,
            }
    if bool(get_nested(config, "rawtest_copcf.require_cluster_assignment")):
        missing_assignments = {
            threshold: sorted(well for well, cluster in mapping.items() if cluster is None)
            for threshold, mapping in cluster_maps.items()
            if any(cluster is None for cluster in mapping.values())
        }
        if missing_assignments:
            raise ValueError(
                f"raw-test typewell cluster assignment is incomplete: {missing_assignments}"
            )
    return cluster_maps, {
        "method": method,
        "source_typewell_count": len(source_series),
        "threshold_map": threshold_map,
        "query_wells": query_meta,
    }


def _query_geometry(
    frame: pd.DataFrame,
    test_dir: Path,
    typewell_clusters: dict[str, str | None],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        path = test_dir / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "GR", "TVT_input"])
        for column in horizontal.columns:
            horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
        row_idx = _row_indices(group["id"])
        if len(row_idx) == 0 or row_idx.max() >= len(horizontal):
            raise IndexError(f"raw-test row index is outside horizontal input for {well}")
        known = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
        if len(known) == 0:
            raise ValueError(f"raw-test well {well} has no known TVT_input prefix")
        anchor_idx = int(known[-1])
        segment = np.arange(int(row_idx.min()), int(row_idx.max()) + 1, dtype=np.int32)
        md = horizontal["MD"].to_numpy(np.float64)
        x = horizontal["X"].to_numpy(np.float64)
        y = horizontal["Y"].to_numpy(np.float64)
        z = horizontal["Z"].to_numpy(np.float64)
        sx, sy, sz, smd = x[segment], y[segment], z[segment], md[segment]
        start_idx, end_idx = int(segment[0]), int(segment[-1])
        dx = float(x[end_idx] - x[start_idx])
        dy = float(y[end_idx] - y[start_idx])
        dz = float(z[end_idx] - z[start_idx])
        dmd = float(md[end_idx] - md[start_idx])
        azimuth = float(math.atan2(dy, dx))
        local_end = min(anchor_idx + 10, len(horizontal) - 1)
        local_azimuth = float(
            math.atan2(y[local_end] - y[anchor_idx], x[local_end] - x[anchor_idx])
        )
        prefix_tvt = horizontal["TVT_input"].to_numpy(np.float64)[known]
        rows.append(
            {
                "well": str(well),
                "typewell_cluster": typewell_clusters.get(str(well)) or "",
                "centroid_x": float(np.nanmean(sx)),
                "centroid_y": float(np.nanmean(sy)),
                "centroid_z": float(np.nanmean(sz)),
                "start_x": float(x[start_idx]),
                "start_y": float(y[start_idx]),
                "start_z": float(z[start_idx]),
                "end_x": float(x[end_idx]),
                "end_y": float(y[end_idx]),
                "end_z": float(z[end_idx]),
                "bbox_dx": _safe_span(sx),
                "bbox_dy": _safe_span(sy),
                "bbox_dz": _safe_span(sz),
                "md_span": _safe_span(smd),
                "z_span": dz,
                "dz_dmd": dz / dmd if abs(dmd) > 1.0e-9 else float("nan"),
                "azimuth": azimuth,
                "azimuth_sin": float(math.sin(azimuth)),
                "azimuth_cos": float(math.cos(azimuth)),
                "local_azimuth": local_azimuth,
                "local_azimuth_sin": float(math.sin(local_azimuth)),
                "local_azimuth_cos": float(math.cos(local_azimuth)),
                "tortuosity": _path_tortuosity(sx, sy, sz),
                "prefix_tvt_range": _safe_span(prefix_tvt),
                "last_md": float(md[anchor_idx]),
                "last_x": float(x[anchor_idx]),
                "last_y": float(y[anchor_idx]),
                "last_z": float(z[anchor_idx]),
            }
        )
    return pd.DataFrame(rows)


def _load_train_geometry(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = parent.find_artifact(
        parent.EXP114_WELL_GEOMETRY,
        get_nested(config, "data.exp114_well_geometry_local"),
    )
    frame = pd.read_csv(source, dtype={"well": str})
    if frame["well"].duplicated().any():
        raise ValueError("exp114 geometry summary contains duplicate wells")
    return frame, {"path": str(source), "sha256": parent.sha256_path(source)}


def _cluster_features(
    *,
    query_geometry: pd.DataFrame,
    train_geometry: pd.DataFrame,
    assignments: pd.DataFrame,
    query_clusters: dict[str, str | None],
    excluded_wells: set[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method = str(get_nested(config, "cluster.assignment_method") or "native_overlap")
    threshold = str(get_nested(config, "cluster.assignment_threshold") or "1")
    assignment = assignments.loc[
        (assignments["method"] == method) & (assignments["threshold"].astype(str) == threshold),
        ["well_id", "cluster_id"],
    ].copy()
    reference = train_geometry.loc[~train_geometry["well"].astype(str).isin(excluded_wells)].copy()
    reference = reference.merge(
        assignment.rename(columns={"well_id": "well"}),
        on="well",
        how="left",
        validate="one_to_one",
    )
    reference["cluster_id"] = reference["cluster_id"].astype("string")
    min_cluster_size = int(get_nested(config, "cluster.min_cluster_size") or 2)
    scale_floor = float(get_nested(config, "cluster.robust_scale_floor_ft") or 250.0)
    counts = reference["cluster_id"].value_counts(dropna=True)
    valid_ids = set(counts.loc[counts >= min_cluster_size].index.astype(str))
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, group in reference.loc[
        reference["cluster_id"].astype(str).isin(valid_ids)
    ].groupby("cluster_id", sort=False):
        x = _numeric(group, "centroid_x").astype(np.float64)
        y = _numeric(group, "centroid_y").astype(np.float64)
        center_x = float(np.nanmedian(x))
        center_y = float(np.nanmedian(y))
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        cluster_rows.append(
            {
                "cluster_id": str(cluster_id),
                "center_x": center_x,
                "center_y": center_y,
                "distance_median": float(np.nanmedian(distance)),
                "distance_scale": _robust_scale(distance, scale_floor),
                "cluster_size": int(len(group)),
            }
        )
    stats = pd.DataFrame(cluster_rows).set_index("cluster_id", drop=False)
    centers = stats[["center_x", "center_y"]].to_numpy(np.float64)
    center_ids = stats["cluster_id"].astype(str).to_numpy()
    ref_xy = (
        reference[["centroid_x", "centroid_y"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    ref_clusters = reference["cluster_id"].astype("string").to_numpy()
    nearby_k_values = [
        int(value) for value in get_nested(config, "cluster.nearby_k_values") or [5, 8, 12]
    ]
    majority_min_share = float(get_nested(config, "cluster.nearby_majority_min_share") or 0.5)
    rows: list[dict[str, Any]] = []
    for query in query_geometry.to_dict(orient="records"):
        well = str(query["well"])
        cluster_id = query_clusters.get(well)
        x, y = float(query["centroid_x"]), float(query["centroid_y"])
        valid = cluster_id is not None and str(cluster_id) in stats.index
        own_distance = float("nan")
        own_z = float("nan")
        nearest_other_distance = float("nan")
        if valid:
            item = stats.loc[str(cluster_id)]
            own_distance = float(np.hypot(x - item["center_x"], y - item["center_y"]))
            own_z = float((own_distance - item["distance_median"]) / item["distance_scale"])
            distances = np.sqrt(np.sum((centers - np.array([x, y])) ** 2, axis=1))
            distances[center_ids == str(cluster_id)] = np.inf
            if np.isfinite(distances).any():
                nearest_other_distance = float(np.min(distances))
        row: dict[str, Any] = {
            "well": well,
            "cluster_id": cluster_id,
            "cluster_size": int(stats.loc[str(cluster_id), "cluster_size"]) if valid else 0,
            "copcf_cluster_feature_valid": float(valid),
            "copcf_own_cluster_dist": own_distance,
            "copcf_own_cluster_dist_z": own_z,
            "copcf_nearest_other_cluster_dist": nearest_other_distance,
            "copcf_nearest_other_closer": float(
                np.isfinite(nearest_other_distance)
                and np.isfinite(own_distance)
                and nearest_other_distance < own_distance
            ),
        }
        distances = np.sqrt(np.sum((ref_xy - np.array([x, y])) ** 2, axis=1))
        for k in nearby_k_values:
            nearest = np.argsort(distances)[: min(k, len(reference))]
            values = [str(ref_clusters[idx]) for idx in nearest if not pd.isna(ref_clusters[idx])]
            if values:
                value_counts = pd.Series(values).value_counts()
                majority_cluster = str(value_counts.index[0])
                majority_count = int(value_counts.iloc[0])
                share = float(majority_count / len(values))
                differs = bool(
                    valid and majority_cluster != str(cluster_id) and share >= majority_min_share
                )
            else:
                majority_count, share, differs = 0, 0.0, False
            row[f"copcf_nearby_majority_count_k{k}"] = majority_count
            row[f"copcf_nearby_majority_share_k{k}"] = share
            row[f"copcf_nearby_majority_diff_k{k}"] = float(differs)
        rows.append(row)
    return pd.DataFrame(rows), {
        "reference_wells": int(len(reference)),
        "clusters": int(len(stats)),
        "excluded_rawtest_wells": sorted(excluded_wells),
    }


def _typewell_prior(
    *,
    frame: pd.DataFrame,
    source_arrays: dict[str, dict[str, np.ndarray]],
    assignments: pd.DataFrame,
    query_clusters: dict[str, str | None],
    threshold: str,
    prior: parent.PriorSpec,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method = str(
        (get_nested(config, "rawtest_copcf.typewell_matching") or {}).get(
            "method", "native_overlap"
        )
    )
    subset = assignments.loc[
        (assignments["method"] == method) & (assignments["threshold"].astype(str) == threshold),
        ["well_id", "cluster_id"],
    ]
    cluster_to_wells = (
        subset.groupby("cluster_id")["well_id"]
        .apply(lambda values: sorted(map(str, values)))
        .to_dict()
    )
    require_in_range = bool(get_nested(config, "rawtest_copcf.prior.require_in_range"))
    min_row_values = int(get_nested(config, "rawtest_copcf.prior.min_row_neighbor_values") or 1)
    rows: list[pd.DataFrame] = []
    neighbor_meta: dict[str, Any] = {}
    for well, group in frame.groupby("well", sort=False):
        cluster = query_clusters.get(str(well))
        neighbors = [
            source_well
            for source_well in cluster_to_wells.get(str(cluster), [])
            if source_well in source_arrays and source_well != str(well)
        ]
        query_md = _numeric(group, "md_since")
        values = [
            _interp_neighbor_delta(
                query_md,
                source_arrays[source_well]["md_since"],
                source_arrays[source_well]["true_delta"],
                require_in_range=require_in_range,
            )
            for source_well in neighbors
        ]
        stacked = np.vstack(values) if values else np.empty((0, len(group)), dtype=np.float32)
        counts = (
            np.isfinite(stacked).sum(axis=0) if len(stacked) else np.zeros(len(group), dtype=int)
        )
        valid = counts >= min_row_values
        delta = np.full(len(group), np.nan, dtype=np.float32)
        std = np.full(len(group), np.nan, dtype=np.float32)
        if valid.any():
            delta[valid] = np.nanmedian(stacked[:, valid], axis=0).astype(np.float32)
            std[valid] = np.nanstd(stacked[:, valid], axis=0).astype(np.float32)
        item = pd.DataFrame({"id": group["id"].astype(str), "well": str(well)})
        prefix = f"copcf_{prior.family}_"
        item[prefix + prior.prior_tvt] = _numeric(group, "last_known_tvt") + delta
        if prior.prior_std:
            item[prefix + prior.prior_std] = std
        if prior.prior_count:
            item[prefix + prior.prior_count] = counts.astype(np.float32)
        if prior.neighbor_wells:
            item[prefix + prior.neighbor_wells] = np.float32(len(neighbors))
        rows.append(item)
        neighbor_meta[str(well)] = {
            "cluster_id": cluster,
            "source_wells": neighbors,
            "source_well_count": len(neighbors),
            "valid_rate": float(valid.mean()),
        }
    return pd.concat(rows, ignore_index=True), neighbor_meta


def _standardized_matrix(
    summaries: pd.DataFrame,
    wells: list[str],
    features: list[str],
    *,
    mean: pd.Series | None = None,
    std: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    values = (
        summaries.set_index("well").reindex(wells)[features].apply(pd.to_numeric, errors="coerce")
    )
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0, ddof=0).replace(0.0, np.nan)
    scaled = (values.fillna(mean).fillna(0.0) - mean.fillna(0.0)) / std.fillna(1.0)
    return scaled.to_numpy(np.float64), mean, std


def _spatial_prior(
    *,
    frame: pd.DataFrame,
    source_arrays: dict[str, dict[str, np.ndarray]],
    train_geometry: pd.DataFrame,
    query_geometry: pd.DataFrame,
    prior: parent.PriorSpec,
    variant: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = [str(value) for value in variant["features"]]
    top_k = int(variant.get("top_k", 8))
    require_in_range = bool(get_nested(config, "rawtest_copcf.prior.require_in_range"))
    min_row_values = int(get_nested(config, "rawtest_copcf.prior.min_row_neighbor_values") or 1)
    distance_epsilon = float(get_nested(config, "rawtest_copcf.prior.distance_epsilon") or 1.0e-6)
    distance_power = float(get_nested(config, "rawtest_copcf.prior.distance_power") or 1.0)
    source_wells = sorted(set(train_geometry["well"].astype(str)).intersection(source_arrays))
    train_geometry = train_geometry.loc[
        train_geometry["well"].astype(str).isin(source_wells)
    ].copy()
    rows: list[pd.DataFrame] = []
    neighbor_meta: dict[str, Any] = {}
    query_index = query_geometry.set_index("well", drop=False)
    for well, group in frame.groupby("well", sort=False):
        query_well = str(well)
        if query_well not in query_index.index:
            raise ValueError(f"raw-test geometry is missing query well {query_well}")
        train_matrix, mean, std = _standardized_matrix(train_geometry, source_wells, features)
        query_matrix, _, _ = _standardized_matrix(
            query_geometry, [query_well], features, mean=mean, std=std
        )
        distances = np.sqrt(np.sum((train_matrix - query_matrix[0]) ** 2, axis=1))
        keep = np.argsort(distances)[: min(top_k, len(source_wells))]
        neighbors = [source_wells[int(idx)] for idx in keep]
        neighbor_distances = distances[keep].astype(np.float64)
        query_md = _numeric(group, "md_since")
        values = [
            _interp_neighbor_delta(
                query_md,
                source_arrays[source_well]["md_since"],
                source_arrays[source_well]["true_delta"],
                require_in_range=require_in_range,
            )
            for source_well in neighbors
        ]
        stacked = np.vstack(values).astype(np.float32)
        finite = np.isfinite(stacked)
        weights = 1.0 / np.power(neighbor_distances + distance_epsilon, distance_power)
        weight_sum = np.where(finite, weights[:, None], 0.0).sum(axis=0)
        counts = finite.sum(axis=0)
        valid = (counts >= min_row_values) & (weight_sum > 0.0)
        delta = np.full(len(group), np.nan, dtype=np.float32)
        delta[valid] = (
            np.where(finite[:, valid], stacked[:, valid] * weights[:, None], 0.0).sum(axis=0)
            / weight_sum[valid]
        ).astype(np.float32)
        prior_std = np.full(len(group), np.nan, dtype=np.float32)
        prior_std[valid] = np.nanstd(stacked[:, valid], axis=0).astype(np.float32)
        query_azimuth = float(query_index.loc[query_well, "azimuth"])
        source_index = train_geometry.set_index("well")
        azimuth_mismatch = float(
            np.nanmean(
                [
                    _circular_abs_diff(query_azimuth, float(source_index.loc[item, "azimuth"]))
                    for item in neighbors
                ]
            )
        )
        item = pd.DataFrame({"id": group["id"].astype(str), "well": query_well})
        prefix = f"copcf_{prior.family}_"
        item[prefix + prior.prior_tvt] = _numeric(group, "last_known_tvt") + delta
        if prior.prior_std:
            item[prefix + prior.prior_std] = prior_std
        if prior.prior_count:
            item[prefix + prior.prior_count] = counts.astype(np.float32)
        if prior.neighbor_wells:
            item[prefix + prior.neighbor_wells] = np.float32(len(neighbors))
        if prior.distance_mean:
            item[prefix + prior.distance_mean] = np.float32(np.mean(neighbor_distances))
        if prior.azimuth_mismatch:
            item[prefix + prior.azimuth_mismatch] = np.float32(azimuth_mismatch)
        rows.append(item)
        neighbor_meta[query_well] = {
            "source_wells": neighbors,
            "source_distances": neighbor_distances.tolist(),
            "valid_rate": float(valid.mean()),
        }
    return pd.concat(rows, ignore_index=True), neighbor_meta


def _add_base_confidence_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    priors = parent.parse_prior_specs(config)
    gates = parent.parse_cluster_gates(config)
    generated: dict[str, np.ndarray] = {}
    cluster_columns = [
        "copcf_cluster_feature_valid",
        "copcf_own_cluster_dist",
        "copcf_own_cluster_dist_z",
        "copcf_nearest_other_cluster_dist",
        "copcf_nearest_other_closer",
    ]
    for k in get_nested(config, "cluster.nearby_k_values") or [5, 8, 12]:
        cluster_columns.extend(
            [
                f"copcf_nearby_majority_count_k{int(k)}",
                f"copcf_nearby_majority_share_k{int(k)}",
                f"copcf_nearby_majority_diff_k{int(k)}",
            ]
        )
    for column in cluster_columns:
        generated[column] = _numeric(frame, column)
    for gate in gates:
        mask = np.ones(len(frame), dtype=bool)
        raw = dict(gate.raw)

        def condition(spec: dict[str, Any]) -> np.ndarray:
            result = np.ones(len(frame), dtype=bool)
            if spec.get("own_cluster_dist_z_gt") is not None:
                result &= _numeric(frame, "copcf_own_cluster_dist_z") > float(
                    spec["own_cluster_dist_z_gt"]
                )
            if bool(spec.get("nearest_other_closer", False)):
                result &= _numeric(frame, "copcf_nearest_other_closer") > 0.5
            if bool(spec.get("nearby_majority_diff", False)):
                result &= (
                    _numeric(frame, f"copcf_nearby_majority_diff_k{int(spec.get('nearby_k', 8))}")
                    > 0.5
                )
            return result

        valid_cluster = _numeric(frame, "copcf_cluster_feature_valid") > 0.5
        if raw.get("all_rows"):
            mask = np.ones(len(frame), dtype=bool)
        elif raw.get("any_of"):
            options = [condition(dict(item)) for item in raw["any_of"]]
            mask = valid_cluster & np.logical_or.reduce(options)
        else:
            mask = valid_cluster & condition(raw)
        gate_col = f"copcf_gate_{gate.name}"
        frame[gate_col] = mask.astype(np.float32)
        generated[gate_col] = frame[gate_col].to_numpy(np.float32)
        ratio_col = f"copcf_well_gate_ratio_{gate.name}"
        frame[ratio_col] = frame.groupby("well", observed=True)[gate_col].transform("mean")
        generated[ratio_col] = frame[ratio_col].to_numpy(np.float32)
    gate_columns = [f"copcf_gate_{gate.name}" for gate in gates]
    frame["copcf_any_configured_gate"] = frame[gate_columns].max(axis=1).astype(np.float32)
    generated["copcf_any_configured_gate"] = frame["copcf_any_configured_gate"].to_numpy(np.float32)
    settings = get_nested(config, "ranker.cluster_prior_features") or {}
    lookup = {prior.name: prior for prior in priors}
    typewell = lookup[str(settings.get("primary_typewell_prior"))]
    spatial = lookup[str(settings.get("primary_spatial_prior"))]
    typewell_col = parent.prefixed_prior_column(typewell, typewell.prior_tvt)
    spatial_col = parent.prefixed_prior_column(spatial, spatial.prior_tvt)
    delta = _numeric(frame, typewell_col) - _numeric(frame, spatial_col)
    frame["copcf_typewell_spatial_prior_delta"] = delta
    frame["copcf_typewell_spatial_prior_abs_delta"] = np.abs(delta)
    generated["copcf_typewell_spatial_prior_delta"] = delta
    generated["copcf_typewell_spatial_prior_abs_delta"] = np.abs(delta)
    for prior in priors:
        prior_tvt_col = parent.prefixed_prior_column(prior, prior.prior_tvt)
        prior_std_col = parent.prefixed_prior_column(prior, prior.prior_std)
        prior_count_col = parent.prefixed_prior_column(prior, prior.prior_count)
        neighbor_col = parent.prefixed_prior_column(prior, prior.neighbor_wells)
        valid = np.isfinite(_numeric(frame, prior_tvt_col))
        name = f"copcf_{prior.name}_valid_prior"
        frame[name] = valid.astype(np.float32)
        generated[name] = frame[name].to_numpy(np.float32)
        for suffix, source_column in [
            ("prior_std", prior_std_col),
            ("prior_count", prior_count_col),
            ("neighbor_wells", neighbor_col),
        ]:
            if source_column is None:
                continue
            output = f"copcf_{prior.name}_{suffix}"
            frame[output] = _numeric(frame, source_column)
            generated[output] = frame[output].to_numpy(np.float32)
    return frame, list(generated)


def attach_regenerated_copcf(
    frame: pd.DataFrame,
    config: dict[str, Any],
    paths: ExperimentPaths,
    *,
    train_reference_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    settings = get_nested(config, "rawtest_copcf") or {}
    if not bool(settings.get("enabled", False)):
        return frame, [], {"enabled": False}
    query_wells = sorted(frame["well"].astype(str).unique())
    excluded_wells = set(query_wells)
    source_arrays, source_meta = _load_source_curves(config, excluded_wells, train_reference_frame)
    assignments, assignment_meta = _load_assignments(config)
    cluster_maps, typewell_meta = _assign_test_clusters(
        query_wells=query_wells,
        source_wells=set(source_arrays),
        assignments=assignments,
        train_dir=paths.train_data_dir,
        test_dir=paths.test_data_dir,
        config=config,
    )
    threshold_map = {
        str(key): str(value)
        for key, value in (settings.get("typewell_prior_thresholds") or {}).items()
    }
    spatial_threshold = str(settings.get("spatial_typewell_threshold", "0.999"))
    query_geometry = _query_geometry(
        frame,
        paths.test_data_dir,
        cluster_maps.get(spatial_threshold, {}),
    )
    train_geometry, geometry_meta = _load_train_geometry(config)
    train_geometry = train_geometry.loc[
        ~train_geometry["well"].astype(str).isin(excluded_wells)
    ].copy()
    forbidden_geometry_sources = sorted(
        excluded_wells.intersection(train_geometry["well"].astype(str))
    )
    if forbidden_geometry_sources:
        raise AssertionError(
            f"raw-test wells leaked into geometry references: {forbidden_geometry_sources}"
        )
    primary_threshold = str(get_nested(config, "cluster.assignment_threshold") or "1")
    cluster_frame, cluster_meta = _cluster_features(
        query_geometry=query_geometry,
        train_geometry=train_geometry,
        assignments=assignments,
        query_clusters=cluster_maps[primary_threshold],
        excluded_wells=excluded_wells,
        config=config,
    )
    out = frame.merge(cluster_frame, on="well", how="left", validate="many_to_one")
    if len(out) != len(frame):
        raise ValueError("raw-test cluster features changed row count")
    priors = parent.parse_prior_specs(config)
    spatial_variants = {
        str(item["name"]): dict(item) for item in settings.get("spatial_variants", [])
    }
    prior_meta: dict[str, Any] = {}
    for prior in priors:
        if prior.family == "typewell":
            threshold = threshold_map[prior.name]
            prior_frame, item_meta = _typewell_prior(
                frame=out,
                source_arrays=source_arrays,
                assignments=assignments,
                query_clusters=cluster_maps[threshold],
                threshold=threshold,
                prior=prior,
                config=config,
            )
        else:
            prior_frame, item_meta = _spatial_prior(
                frame=out,
                source_arrays=source_arrays,
                train_geometry=train_geometry,
                query_geometry=query_geometry,
                prior=prior,
                variant=spatial_variants[prior.name.removeprefix("spatial_")],
                config=config,
            )
        extra_columns = [column for column in prior_frame if column not in {"id", "well"}]
        out = out.merge(
            prior_frame[["id", "well", *extra_columns]],
            on=["id", "well"],
            how="left",
            validate="one_to_one",
        )
        if len(out) != len(frame):
            raise ValueError(f"raw-test prior {prior.name} changed row count")
        prior_meta[prior.name] = item_meta
    out, generated_columns = _add_base_confidence_features(out, config)
    expected_base_features = int(settings.get("expected_generated_base_feature_count", 0))
    if expected_base_features > 0 and len(generated_columns) != expected_base_features:
        raise ValueError(
            "unexpected regenerated copcf base feature count: "
            f"expected={expected_base_features} actual={len(generated_columns)}"
        )
    forbidden_sources = sorted(
        {
            source_well
            for item in prior_meta.values()
            for well_meta in item.values()
            for source_well in well_meta.get("source_wells", [])
            if source_well in excluded_wells
        }
    )
    if forbidden_sources:
        raise AssertionError(f"raw-test wells leaked into regenerated prior: {forbidden_sources}")
    return (
        out,
        generated_columns,
        {
            "enabled": True,
            "source_curves": source_meta,
            "assignments": assignment_meta,
            "typewell_mapping": typewell_meta,
            "geometry": geometry_meta,
            "cluster": cluster_meta,
            "priors": prior_meta,
            "generated_base_feature_count": len(generated_columns),
            "generated_base_features": generated_columns,
            "rawtest_well_source_exclusion": "pass",
        },
    )


__all__ = ["attach_regenerated_copcf"]
