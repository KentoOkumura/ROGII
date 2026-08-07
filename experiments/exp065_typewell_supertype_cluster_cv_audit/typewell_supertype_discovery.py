from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

TYPEWELL_SUFFIX = "__typewell.csv"


@dataclass(frozen=True)
class TypewellRecord:
    well_id: str
    path: Path
    exact_hash: str
    n_rows: int
    tvt_min: float
    tvt_max: float
    gr_min: float
    gr_max: float
    gr_mean: float
    gr_std: float
    gr_missing_rate: float
    signature_valid: bool


@dataclass(frozen=True)
class NativeTypewellSeries:
    well_id: str
    tvt: np.ndarray
    gr: np.ndarray
    gr_quantized: np.ndarray
    median_tvt_step: float


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def cfg(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    value = get_nested(config, dotted_key)
    return default if value is None else value


def well_id_from_typewell_path(path: Path) -> str:
    return path.name.removesuffix(TYPEWELL_SUFFIX)


def exact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - mean) / std
    return out.astype(np.float32)


def build_signature(
    frame: pd.DataFrame,
    n_points: int,
    smooth_window: int,
    min_valid_fraction: float,
) -> tuple[np.ndarray, bool]:
    work = frame[["TVT", "GR"]].copy()
    work["TVT"] = pd.to_numeric(work["TVT"], errors="coerce")
    work["GR"] = pd.to_numeric(work["GR"], errors="coerce")
    valid_fraction = float(work["GR"].notna().mean()) if len(work) else 0.0
    work = work.dropna(subset=["TVT", "GR"])
    if len(work) < 4 or valid_fraction < min_valid_fraction:
        return np.zeros(n_points, dtype=np.float32), False

    work = work.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    tvt = work["TVT"].to_numpy(dtype=np.float64)
    gr = work["GR"].to_numpy(dtype=np.float64)
    tvt_span = float(tvt[-1] - tvt[0])
    if tvt_span <= 1e-8:
        return np.zeros(n_points, dtype=np.float32), False

    x = (tvt - tvt[0]) / tvt_span
    grid = np.linspace(0.0, 1.0, n_points)
    resampled = np.interp(grid, x, gr)
    resampled = smooth(resampled, smooth_window)
    return zscore(resampled), True


def load_typewell_records(
    train_dir: Path,
    n_points: int,
    smooth_window: int,
    min_valid_fraction: float,
    max_wells: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    paths = sorted(train_dir.glob(f"*{TYPEWELL_SUFFIX}"))
    if max_wells is not None:
        paths = paths[:max_wells]
    if not paths:
        raise FileNotFoundError(f"No typewell CSV files found in {train_dir}")

    records: list[TypewellRecord] = []
    signatures: list[np.ndarray] = []
    for path in paths:
        frame = pd.read_csv(path)
        gr = pd.to_numeric(frame.get("GR"), errors="coerce")
        tvt = pd.to_numeric(frame.get("TVT"), errors="coerce")
        signature, valid = build_signature(frame, n_points, smooth_window, min_valid_fraction)
        records.append(
            TypewellRecord(
                well_id=well_id_from_typewell_path(path),
                path=path,
                exact_hash=exact_hash(path),
                n_rows=int(len(frame)),
                tvt_min=float(tvt.min()),
                tvt_max=float(tvt.max()),
                gr_min=float(gr.min()),
                gr_max=float(gr.max()),
                gr_mean=float(gr.mean()),
                gr_std=float(gr.std()),
                gr_missing_rate=float(gr.isna().mean()),
                signature_valid=valid,
            )
        )
        signatures.append(signature)

    index = pd.DataFrame([record.__dict__ for record in records])
    hash_sizes = index["exact_hash"].value_counts()
    index["exact_group_id"] = "exact_" + index["exact_hash"].astype(str)
    index["exact_group_size"] = index["exact_hash"].map(hash_sizes).astype(int)
    return index, np.vstack(signatures).astype(np.float32)


def load_native_typewell_series(data_dir: Path, well_ids: list[str]) -> list[NativeTypewellSeries]:
    series: list[NativeTypewellSeries] = []
    for well_id in well_ids:
        frame = pd.read_csv(data_dir / f"{well_id}{TYPEWELL_SUFFIX}")
        work = frame[["TVT", "GR"]].copy()
        work["TVT"] = pd.to_numeric(work["TVT"], errors="coerce")
        work["GR"] = pd.to_numeric(work["GR"], errors="coerce")
        work = work.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        tvt = work["TVT"].to_numpy(dtype=np.float64)
        gr = work["GR"].to_numpy(dtype=np.float64)
        if len(tvt) >= 2:
            median_step = float(np.median(np.diff(tvt)))
        else:
            median_step = float("nan")
        series.append(
            NativeTypewellSeries(
                well_id=well_id,
                tvt=tvt,
                gr=gr,
                gr_quantized=np.rint(gr * 100.0).astype(np.int64),
                median_tvt_step=median_step,
            )
        )
    return series


def rolling_hashes(values: np.ndarray, k: int) -> np.ndarray:
    n = len(values)
    if n < k:
        return np.empty(0, dtype=np.uint64)

    base = 1_000_003
    offset = 2_147_483_647
    mask = (1 << 64) - 1
    hashes = np.empty(n - k + 1, dtype=np.uint64)
    current = 0
    power = 1
    for _ in range(k - 1):
        power = (power * base) & mask

    adjusted = values.astype(np.int64, copy=False)
    for idx in range(k):
        current = (current * base + int(adjusted[idx]) + offset) & mask
    hashes[0] = current

    for idx in range(k, n):
        left = int(adjusted[idx - k]) + offset
        right = int(adjusted[idx]) + offset
        current = ((current - left * power) * base + right) & mask
        hashes[idx - k + 1] = current
    return hashes


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def native_overlap_stats(
    left: NativeTypewellSeries,
    right: NativeTypewellSeries,
    row_lag_right_minus_left: int,
) -> dict[str, Any] | None:
    lag = int(row_lag_right_minus_left)
    left_start = max(0, -lag)
    right_start = left_start + lag
    overlap_rows = min(len(left.gr) - left_start, len(right.gr) - right_start)
    if overlap_rows <= 0:
        return None

    left_end = left_start + overlap_rows
    right_end = right_start + overlap_rows
    left_gr = left.gr[left_start:left_end]
    right_gr = right.gr[right_start:right_end]
    left_q = left.gr_quantized[left_start:left_end]
    right_q = right.gr_quantized[right_start:right_end]
    exact_mask = left_q == right_q

    diff = left_gr - right_gr
    if overlap_rows >= 2 and np.std(left_gr) > 1e-8 and np.std(right_gr) > 1e-8:
        corr = float(np.corrcoef(left_gr, right_gr)[0, 1])
    else:
        corr = float("nan")

    left_tvt = left.tvt[left_start:left_end]
    right_tvt = right.tvt[right_start:right_end]
    overlap_fraction_shorter = overlap_rows / float(min(len(left.gr), len(right.gr)))
    overlap_fraction_left = overlap_rows / float(len(left.gr))
    overlap_fraction_right = overlap_rows / float(len(right.gr))
    row_lag_ft_equivalent = (
        float(lag * np.nanmedian([left.median_tvt_step, right.median_tvt_step]))
        if np.isfinite(left.median_tvt_step) or np.isfinite(right.median_tvt_step)
        else float("nan")
    )

    relation = "partial_overlap"
    if overlap_fraction_left >= 0.999 and overlap_fraction_right >= 0.999:
        relation = "same_length_overlap"
    elif overlap_fraction_left >= 0.999:
        relation = "left_contained_in_right"
    elif overlap_fraction_right >= 0.999:
        relation = "right_contained_in_left"

    return {
        "well_id_a": left.well_id,
        "well_id_b": right.well_id,
        "row_lag_b_minus_a": lag,
        "row_lag_ft_equivalent": round(row_lag_ft_equivalent, 6),
        "overlap_rows": int(overlap_rows),
        "overlap_fraction_shorter": round(overlap_fraction_shorter, 6),
        "overlap_fraction_a": round(overlap_fraction_left, 6),
        "overlap_fraction_b": round(overlap_fraction_right, 6),
        "exact_match_rows": int(exact_mask.sum()),
        "exact_match_rate": round(float(exact_mask.mean()), 9),
        "longest_exact_run_rows": int(longest_true_run(exact_mask)),
        "gr_corr": round(corr, 9) if np.isfinite(corr) else None,
        "gr_mae": round(float(np.mean(np.abs(diff))), 9),
        "gr_max_abs_diff": round(float(np.max(np.abs(diff))), 9),
        "tvt_delta_b_minus_a_median": round(float(np.median(right_tvt - left_tvt)), 9),
        "tvt_delta_b_minus_a_min": round(float(np.min(right_tvt - left_tvt)), 9),
        "tvt_delta_b_minus_a_max": round(float(np.max(right_tvt - left_tvt)), 9),
        "a_start_row": int(left_start),
        "a_end_row_exclusive": int(left_end),
        "b_start_row": int(right_start),
        "b_end_row_exclusive": int(right_end),
        "a_prefix_rows": int(left_start),
        "b_prefix_rows": int(right_start),
        "a_suffix_rows": int(len(left.gr) - left_end),
        "b_suffix_rows": int(len(right.gr) - right_end),
        "a_tvt_start": round(float(left_tvt[0]), 6),
        "a_tvt_end": round(float(left_tvt[-1]), 6),
        "b_tvt_start": round(float(right_tvt[0]), 6),
        "b_tvt_end": round(float(right_tvt[-1]), 6),
        "relation": relation,
    }


def discover_native_overlap_pairs(
    series: list[NativeTypewellSeries],
    kgram_rows: int,
    max_hash_occurrences: int,
    min_kgram_hits: int,
    min_overlap_rows: int,
    min_overlap_fraction_shorter: float,
) -> pd.DataFrame:
    hash_to_occurrences: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for series_idx, item in enumerate(series):
        hashes = rolling_hashes(item.gr_quantized, kgram_rows)
        for pos, value in enumerate(hashes):
            hash_to_occurrences[int(value)].append((series_idx, pos))

    lag_hits: dict[tuple[int, int, int], int] = defaultdict(int)
    for occurrences in hash_to_occurrences.values():
        if len(occurrences) < 2 or len(occurrences) > max_hash_occurrences:
            continue
        for left_pos in range(len(occurrences) - 1):
            left_idx, left_row = occurrences[left_pos]
            for right_idx, right_row in occurrences[left_pos + 1 :]:
                if left_idx == right_idx:
                    continue
                if left_idx < right_idx:
                    pair_key = (left_idx, right_idx, right_row - left_row)
                else:
                    pair_key = (right_idx, left_idx, left_row - right_row)
                lag_hits[pair_key] += 1

    rows: list[dict[str, Any]] = []
    for (left_idx, right_idx, lag), hits in lag_hits.items():
        if hits < min_kgram_hits:
            continue
        stats = native_overlap_stats(series[left_idx], series[right_idx], lag)
        if stats is None:
            continue
        if int(stats["overlap_rows"]) < min_overlap_rows:
            continue
        if float(stats["overlap_fraction_shorter"]) < min_overlap_fraction_shorter:
            continue
        stats["kgram_hits"] = int(hits)
        rows.append(stats)

    if not rows:
        return pd.DataFrame(
            columns=[
                "well_id_a",
                "well_id_b",
                "row_lag_b_minus_a",
                "overlap_rows",
                "overlap_fraction_shorter",
                "exact_match_rate",
                "kgram_hits",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "exact_match_rate",
            "overlap_rows",
            "overlap_fraction_shorter",
            "kgram_hits",
            "well_id_a",
            "well_id_b",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std


def shifted_ncc(
    signatures: np.ndarray,
    max_shift: int,
    min_overlap_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_wells, n_points = signatures.shape
    best = np.full((n_wells, n_wells), -np.inf, dtype=np.float32)
    best_shift = np.zeros((n_wells, n_wells), dtype=np.int16)
    min_overlap = max(4, int(math.ceil(n_points * min_overlap_fraction)))

    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            left = signatures[:, : n_points + shift]
            right = signatures[:, -shift:]
        elif shift > 0:
            left = signatures[:, shift:]
            right = signatures[:, : n_points - shift]
        else:
            left = signatures
            right = signatures

        overlap = left.shape[1]
        if overlap < min_overlap:
            continue
        left_norm = normalize_rows(left)
        right_norm = normalize_rows(right)
        corr = (left_norm @ right_norm.T) / float(overlap)
        update = corr > best
        best[update] = corr[update]
        best_shift[update] = shift

    np.fill_diagonal(best, -np.inf)
    return best, best_shift


def candidate_pairs_from_ncc(
    well_ids: list[str],
    best: np.ndarray,
    best_shift: np.ndarray,
    top_k: int,
    min_similarity: float,
) -> pd.DataFrame:
    pair_keys: set[tuple[int, int]] = set()
    n_wells = len(well_ids)
    for i in range(n_wells):
        row = np.maximum(best[i], best[:, i])
        row[i] = -np.inf
        if top_k > 0:
            top_idx = np.argpartition(row, -min(top_k, n_wells - 1))[-top_k:]
            for j in top_idx:
                if np.isfinite(row[j]):
                    pair_keys.add((min(i, int(j)), max(i, int(j))))
        for j in np.flatnonzero(row >= min_similarity):
            pair_keys.add((min(i, int(j)), max(i, int(j))))

    rows: list[dict[str, Any]] = []
    for i, j in sorted(pair_keys):
        if i == j:
            continue
        if best[i, j] >= best[j, i]:
            similarity = float(best[i, j])
            shift = int(best_shift[i, j])
        else:
            similarity = float(best[j, i])
            shift = -int(best_shift[j, i])
        rows.append(
            {
                "well_id_a": well_ids[i],
                "well_id_b": well_ids[j],
                "shifted_ncc_similarity": round(similarity, 6),
                "best_shift_points": shift,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["shifted_ncc_similarity", "well_id_a", "well_id_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def constrained_dtw_distance(left: np.ndarray, right: np.ndarray, band: int) -> float:
    n = len(left)
    m = len(right)
    band = max(band, abs(n - m))
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr.fill(np.inf)
        start = max(1, i - band)
        end = min(m, i + band)
        for j in range(start, end + 1):
            cost = abs(float(left[i - 1] - right[j - 1]))
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev

    distance = float(prev[m]) / float(max(n, m))
    return distance


def add_dtw_scores(
    pairs: pd.DataFrame,
    well_ids: list[str],
    signatures: np.ndarray,
    band_fraction: float,
    max_pairs: int,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.assign(dtw_distance=[], dtw_similarity=[])
    id_to_idx = {well_id: idx for idx, well_id in enumerate(well_ids)}
    band = max(1, int(round(signatures.shape[1] * band_fraction)))
    work = pairs.head(max_pairs).copy()
    distances: list[float] = []
    similarities: list[float] = []
    for row in work.itertuples(index=False):
        left = signatures[id_to_idx[row.well_id_a]]
        right = signatures[id_to_idx[row.well_id_b]]
        distance = constrained_dtw_distance(left, right, band)
        distances.append(round(distance, 6))
        similarities.append(round(1.0 / (1.0 + distance), 6))
    work["dtw_distance"] = distances
    work["dtw_similarity"] = similarities
    return work.sort_values(
        ["dtw_similarity", "shifted_ncc_similarity", "well_id_a", "well_id_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def connected_components_from_edges(
    well_ids: list[str],
    edges: pd.DataFrame,
    similarity_column: str,
    threshold: float,
) -> dict[str, list[str]]:
    uf = UnionFind(well_ids)
    if not edges.empty:
        for row in edges.itertuples(index=False):
            if float(getattr(row, similarity_column)) >= threshold:
                uf.union(str(row.well_id_a), str(row.well_id_b))

    groups: dict[str, list[str]] = defaultdict(list)
    for well_id in well_ids:
        groups[uf.find(well_id)].append(well_id)
    return {min(members): sorted(members) for members in groups.values()}


def exact_components(index: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for _, part in index.groupby("exact_hash", sort=True):
        members = sorted(part["well_id"].astype(str).tolist())
        groups[min(members)] = members
    return groups


def threshold_label(value: float) -> str:
    return f"{value:.6g}"


def component_rows(
    method: str,
    threshold: str,
    groups: dict[str, list[str]],
    exact_size_by_well: dict[str, int],
    max_members_preview: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sorted_groups = sorted(groups.values(), key=lambda values: (-len(values), values[0]))

    for cluster_idx, members in enumerate(sorted_groups, start=1):
        cluster_id = f"{method}_{threshold}_cluster_{cluster_idx:04d}"
        exact_duplicate_wells = sum(
            1 for well_id in members if exact_size_by_well.get(well_id, 1) > 1
        )
        preview_members = members[:max_members_preview]
        summary_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "representative_well_id": members[0],
                "exact_duplicate_wells": exact_duplicate_wells,
                "members": ",".join(preview_members),
                "members_truncated": len(members) > len(preview_members),
            }
        )
        for well_id in members:
            assignment_rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "cluster_id": cluster_id,
                    "well_id": well_id,
                    "cluster_size": len(members),
                    "representative_well_id": members[0],
                }
            )
    return assignment_rows, summary_rows


def build_cluster_outputs(
    index: pd.DataFrame,
    ncc_pairs: pd.DataFrame,
    dtw_pairs: pd.DataFrame,
    native_pairs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    well_ids = index["well_id"].astype(str).tolist()
    exact_size_by_well = dict(zip(index["well_id"], index["exact_group_size"], strict=True))
    max_members_preview = int(cfg(config, "discovery.reporting.max_members_preview", 80))

    assignment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    definitions: list[tuple[str, str, dict[str, list[str]]]] = [
        ("exact_hash", "byte_hash", exact_components(index))
    ]

    for threshold in cfg(config, "discovery.shifted_ncc.cluster_thresholds", [0.8]):
        groups = connected_components_from_edges(
            well_ids,
            ncc_pairs,
            "shifted_ncc_similarity",
            float(threshold),
        )
        definitions.append(("shifted_ncc", threshold_label(float(threshold)), groups))

    if not dtw_pairs.empty:
        for threshold in cfg(config, "discovery.dtw.cluster_similarity_thresholds", [0.5]):
            groups = connected_components_from_edges(
                well_ids,
                dtw_pairs,
                "dtw_similarity",
                float(threshold),
            )
            definitions.append(("dtw", threshold_label(float(threshold)), groups))

    if not native_pairs.empty:
        for threshold in cfg(
            config,
            "discovery.native_overlap.exact_match_rate_thresholds",
            [1.0],
        ):
            groups = connected_components_from_edges(
                well_ids,
                native_pairs,
                "exact_match_rate",
                float(threshold),
            )
            definitions.append(("native_overlap", threshold_label(float(threshold)), groups))

    for method, threshold, groups in definitions:
        rows, summaries = component_rows(
            method,
            threshold,
            groups,
            exact_size_by_well,
            max_members_preview,
        )
        assignment_rows.extend(rows)
        summary_rows.extend(summaries)
        sizes = [len(members) for members in groups.values()]
        metric_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "unique_groups": len(groups),
                "multi_well_groups": int(sum(size > 1 for size in sizes)),
                "wells_in_multi_well_groups": int(sum(size for size in sizes if size > 1)),
                "max_group_size": int(max(sizes) if sizes else 0),
            }
        )

    assignments = pd.DataFrame(assignment_rows)
    summaries = pd.DataFrame(summary_rows).sort_values(
        ["method", "threshold", "cluster_size", "cluster_id"],
        ascending=[True, True, False, True],
    )
    return assignments, summaries, metric_rows


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_discovery(max_wells: int | None = None, skip_dtw: bool = False) -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()

    split = str(cfg(config, "discovery.split", "train"))
    data_dir = paths.train_data_dir if split == "train" else paths.test_data_dir
    n_points = int(cfg(config, "discovery.signature.n_points", 160))
    smooth_window = int(cfg(config, "discovery.signature.smooth_window", 5))
    min_valid_fraction = float(cfg(config, "discovery.signature.min_valid_gr_fraction", 0.5))

    index, signatures = load_typewell_records(
        data_dir,
        n_points=n_points,
        smooth_window=smooth_window,
        min_valid_fraction=min_valid_fraction,
        max_wells=max_wells,
    )
    well_ids = index["well_id"].astype(str).tolist()

    best, best_shift = shifted_ncc(
        signatures,
        max_shift=int(cfg(config, "discovery.shifted_ncc.max_shift", 32)),
        min_overlap_fraction=float(cfg(config, "discovery.shifted_ncc.min_overlap_fraction", 0.7)),
    )
    ncc_pairs = candidate_pairs_from_ncc(
        well_ids,
        best,
        best_shift,
        top_k=int(cfg(config, "discovery.shifted_ncc.top_k_per_well", 20)),
        min_similarity=float(cfg(config, "discovery.shifted_ncc.candidate_min_similarity", 0.62)),
    )

    dtw_enabled = bool(cfg(config, "discovery.dtw.enabled", True)) and not skip_dtw
    if dtw_enabled:
        dtw_pairs = add_dtw_scores(
            ncc_pairs,
            well_ids,
            signatures,
            band_fraction=float(cfg(config, "discovery.dtw.band_fraction", 0.12)),
            max_pairs=int(cfg(config, "discovery.dtw.max_pairs", 20000)),
        )
    else:
        dtw_pairs = pd.DataFrame(
            columns=[
                "well_id_a",
                "well_id_b",
                "shifted_ncc_similarity",
                "best_shift_points",
                "dtw_distance",
                "dtw_similarity",
            ]
        )

    native_enabled = bool(cfg(config, "discovery.native_overlap.enabled", True))
    if native_enabled:
        native_series = load_native_typewell_series(data_dir, well_ids)
        native_pairs = discover_native_overlap_pairs(
            native_series,
            kgram_rows=int(cfg(config, "discovery.native_overlap.kgram_rows", 64)),
            max_hash_occurrences=int(
                cfg(config, "discovery.native_overlap.max_hash_occurrences", 200)
            ),
            min_kgram_hits=int(cfg(config, "discovery.native_overlap.min_kgram_hits", 1)),
            min_overlap_rows=int(cfg(config, "discovery.native_overlap.min_overlap_rows", 200)),
            min_overlap_fraction_shorter=float(
                cfg(config, "discovery.native_overlap.min_overlap_fraction_shorter", 0.8)
            ),
        )
    else:
        native_pairs = pd.DataFrame(
            columns=[
                "well_id_a",
                "well_id_b",
                "row_lag_b_minus_a",
                "overlap_rows",
                "overlap_fraction_shorter",
                "exact_match_rate",
                "kgram_hits",
            ]
        )

    assignments, summaries, cluster_metrics = build_cluster_outputs(
        index,
        ncc_pairs,
        dtw_pairs,
        native_pairs,
        config,
    )

    artifact_paths = {
        "well_index": paths.artifacts_dir / "typewell_well_index.csv",
        "ncc_pairs": paths.artifacts_dir / "typewell_shifted_ncc_pairs.csv",
        "dtw_pairs": paths.artifacts_dir / "typewell_dtw_pairs.csv",
        "native_overlap_pairs": paths.artifacts_dir / "typewell_native_overlap_pairs.csv",
        "cluster_assignments": paths.artifacts_dir / "common_typewell_cluster_assignments.csv",
        "cluster_summary": paths.artifacts_dir / "common_typewell_cluster_summary.csv",
        "cluster_metrics": paths.artifacts_dir / "common_typewell_cluster_metrics.csv",
        "signatures": paths.features_dir / "typewell_gr_signatures.npy",
    }
    write_csv(index, artifact_paths["well_index"])
    write_csv(ncc_pairs, artifact_paths["ncc_pairs"])
    write_csv(dtw_pairs, artifact_paths["dtw_pairs"])
    write_csv(native_pairs, artifact_paths["native_overlap_pairs"])
    write_csv(assignments, artifact_paths["cluster_assignments"])
    write_csv(summaries, artifact_paths["cluster_summary"])
    cluster_metrics_frame = pd.DataFrame(cluster_metrics)
    write_csv(cluster_metrics_frame, artifact_paths["cluster_metrics"])
    np.save(artifact_paths["signatures"], signatures)

    target_groups = int(cfg(config, "discovery.reporting.target_unique_groups", 57))
    near_target = cluster_metrics_frame.assign(
        distance_to_target=lambda frame: (frame["unique_groups"] - target_groups).abs()
    ).sort_values(["distance_to_target", "method", "threshold"])
    near_target_records = near_target.head(5).to_dict(orient="records")

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "purpose": "common_typewell_discovery",
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "wells": int(len(index)),
        "valid_signatures": int(index["signature_valid"].sum()),
        "exact_unique_groups": int(index["exact_hash"].nunique()),
        "exact_duplicate_wells": int((index["exact_group_size"] > 1).sum()),
        "ncc_pair_rows": int(len(ncc_pairs)),
        "dtw_pair_rows": int(len(dtw_pairs)),
        "native_overlap_pair_rows": int(len(native_pairs)),
        "native_exact_containment_pair_rows": int(
            (
                (native_pairs["exact_match_rate"] >= 1.0)
                & (native_pairs["overlap_fraction_shorter"] >= 0.999)
            ).sum()
        )
        if not native_pairs.empty
        else 0,
        "target_unique_groups_reference": target_groups,
        "nearest_thresholds_to_target_group_count": near_target_records,
        "cluster_metrics": cluster_metrics,
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "group_count_similarity_diagnostic",
        "max_wells": max_wells,
        "notes": "No CV audit or submission candidate selection is performed.",
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover common typewell candidate groups.")
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--skip-dtw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_discovery(max_wells=args.max_wells, skip_dtw=args.skip_dtw)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
