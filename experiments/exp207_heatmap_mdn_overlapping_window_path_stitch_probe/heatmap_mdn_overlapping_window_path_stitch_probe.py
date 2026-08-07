from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from settings import (
    EXPERIMENT_NAME,
    KAGGLE_INPUT_ROOT,
    ROOT,
    ExperimentPaths,
    get_nested,
    load_config,
)


OUTPUT_PREFIX = "exp207_heatmap_mdn_overlapping_window_path_stitch_probe"
DEFAULT_EXP202_PREFIX = "exp202_heatmap_mdn_candidate_generator_probe"
DEFAULT_PATH_NPZ = f"{DEFAULT_EXP202_PREFIX}_heatmap_candidate_paths_top10.npz"
DEFAULT_PATH_SAMPLES = f"{DEFAULT_EXP202_PREFIX}_heatmap_candidate_path_samples.csv.gz"
DEFAULT_EXP099_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)


@dataclass(frozen=True)
class Segment:
    well: str
    path_npz_sample_index: int
    row_center: int
    rank: int
    center_score: float
    score_prob: float
    center_tvt: float
    rows: np.ndarray
    tvt: np.ndarray
    step_abs_mean: float
    step_abs_max: float


@dataclass(frozen=True)
class BeamState:
    total_cost: float
    score_cost: float
    smoothness_cost: float
    overlap_cost: float
    boundary_cost: float
    rank_switch_cost: float
    assignments: tuple[Segment, ...]
    overlap_rows_total: int
    gap_count: int
    last_segment: Segment | None

    @staticmethod
    def empty() -> "BeamState":
        return BeamState(
            total_cost=0.0,
            score_cost=0.0,
            smoothness_cost=0.0,
            overlap_cost=0.0,
            boundary_cost=0.0,
            rank_switch_cost=0.0,
            assignments=(),
            overlap_rows_total=0,
            gap_count=0,
            last_segment=None,
        )


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    if decompressed and path.suffix == ".gz":
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip")


def resolve_config_reference(config: dict[str, Any], value: Any) -> Any:
    if isinstance(value, str):
        nested = get_nested(config, value)
        if nested is not None:
            return nested
    return value


def direct_path_candidates(path_value: Any) -> list[Path]:
    if path_value is None:
        return []
    raw = Path(str(path_value))
    if raw.is_absolute():
        return [raw]
    return [
        ROOT / raw,
        Path.cwd() / raw,
        raw,
    ]


def find_artifact(path_value: Any, *, fallback_name: str) -> Path:
    for candidate in direct_path_candidates(path_value):
        if candidate.exists():
            return candidate

    search_names = []
    if path_value is not None:
        search_names.append(Path(str(path_value)).name)
    search_names.append(fallback_name)

    search_roots = [
        ROOT / "experiments",
        ROOT / "data",
        KAGGLE_INPUT_ROOT,
        Path.cwd(),
    ]
    seen_roots: set[Path] = set()
    for root in search_roots:
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        for name in dict.fromkeys(search_names):
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Could not find artifact {path_value!r} or {fallback_name!r}")


def load_candidate_path_inputs(
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    path_cfg = get_nested(config, "stitching.inputs") or {}
    npz_path = find_artifact(
        resolve_config_reference(config, path_cfg.get("path_npz")),
        fallback_name=DEFAULT_PATH_NPZ,
    )
    samples_path = find_artifact(
        resolve_config_reference(config, path_cfg.get("path_samples")),
        fallback_name=DEFAULT_PATH_SAMPLES,
    )

    with np.load(npz_path) as loaded:
        required_keys = {
            "sample_id",
            "center_tvt",
            "score",
            "pred_tvt_path",
            "horizontal_row_index",
            "horizontal_offsets",
        }
        missing = sorted(required_keys.difference(loaded.files))
        if missing:
            raise ValueError(f"{npz_path} is missing keys: {missing}")
        arrays = {key: loaded[key] for key in required_keys}

    sample_usecols = [
        "path_npz_sample_index",
        "sample_id",
        "id",
        "split",
        "well",
        "fold_index",
        "row_center",
        "prefix_end",
        "horizontal_window_rows",
        "last_known_tvt",
        "prior_center_tvt",
        "md_since_prefix",
        "z_since_prefix",
        "distance_bucket",
        "score_entropy",
        "score_top3_mass",
        "score_top5_mass",
        "top1_top2_score_margin",
        "top1_top3_score_margin",
    ]
    sample_header = pd.read_csv(samples_path, nrows=0).columns.tolist()
    missing_sample_cols = sorted(set(sample_usecols).difference(sample_header))
    if missing_sample_cols:
        raise ValueError(f"{samples_path} is missing columns: {missing_sample_cols}")
    samples = pd.read_csv(
        samples_path,
        usecols=sample_usecols,
        dtype={"id": str, "well": str, "split": str, "distance_bucket": str},
        low_memory=False,
    )
    samples["well"] = samples["well"].astype(str)
    samples["id"] = samples["id"].astype(str)
    for column in samples.columns:
        if column not in {"id", "well", "split", "distance_bucket"}:
            samples[column] = pd.to_numeric(samples[column], errors="coerce")

    forbidden = {"true_center_tvt", "target_in_grid", "center_abs_error"}
    leaked = sorted(forbidden.intersection(samples.columns))
    if leaked:
        raise ValueError(f"target-derived sample columns entered stitch inputs: {leaked}")

    meta = {
        "path_npz": str(npz_path),
        "path_npz_sha256": sha256_path(npz_path),
        "path_samples": str(samples_path),
        "path_samples_csv_gz_sha256": sha256_path(samples_path),
        "path_samples_csv_decompressed_sha256": sha256_path(
            samples_path,
            decompressed=samples_path.suffix == ".gz",
        ),
        "samples": int(len(samples)),
        "wells": int(samples["well"].nunique()),
        "topk": int(arrays["pred_tvt_path"].shape[1]),
        "horizon": int(arrays["pred_tvt_path"].shape[2]),
        "horizontal_offsets_min": int(np.nanmin(arrays["horizontal_offsets"])),
        "horizontal_offsets_max": int(np.nanmax(arrays["horizontal_offsets"])),
    }
    return arrays, samples, meta


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def step_metrics(values: np.ndarray) -> tuple[float, float]:
    diffs = np.abs(np.diff(values.astype(np.float32)))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 0.0, 0.0
    return float(np.mean(diffs)), float(np.max(diffs))


def segments_for_sample(
    *,
    sample: pd.Series,
    arrays: dict[str, np.ndarray],
    topk: int,
) -> list[Segment]:
    sample_index = int(sample["path_npz_sample_index"])
    rows = arrays["horizontal_row_index"][sample_index].astype(np.int32)
    score_values = arrays["score"][sample_index, :topk].astype(np.float32)
    finite_scores = np.where(np.isfinite(score_values) & (score_values > 0), score_values, 0.0)
    score_sum = float(np.sum(finite_scores))
    if score_sum <= 0.0:
        score_prob = np.full(topk, 1.0 / float(topk), dtype=np.float32)
    else:
        score_prob = finite_scores / score_sum

    segments: list[Segment] = []
    for rank in range(1, topk + 1):
        rank_index = rank - 1
        tvt_path = arrays["pred_tvt_path"][sample_index, rank_index].astype(np.float32)
        valid = np.isfinite(tvt_path) & np.isfinite(rows)
        if not np.any(valid):
            continue
        segment_rows = rows[valid].astype(np.int32)
        segment_tvt = tvt_path[valid].astype(np.float32)
        order = np.argsort(segment_rows)
        segment_rows = segment_rows[order]
        segment_tvt = segment_tvt[order]
        step_mean, step_max = step_metrics(segment_tvt)
        center_tvt = float(arrays["center_tvt"][sample_index, rank_index])
        center_score = float(score_values[rank_index])
        segments.append(
            Segment(
                well=str(sample["well"]),
                path_npz_sample_index=sample_index,
                row_center=int(sample["row_center"]),
                rank=rank,
                center_score=center_score,
                score_prob=float(score_prob[rank_index]),
                center_tvt=center_tvt,
                rows=segment_rows,
                tvt=segment_tvt,
                step_abs_mean=step_mean,
                step_abs_max=step_max,
            )
        )
    return segments


def adjacent_overlap_abs(prev_segment: Segment, segment: Segment) -> tuple[int, float]:
    rows, prev_idx, cur_idx = np.intersect1d(
        prev_segment.rows,
        segment.rows,
        assume_unique=False,
        return_indices=True,
    )
    if len(rows) == 0:
        return 0, float("nan")
    diff = np.abs(prev_segment.tvt[prev_idx] - segment.tvt[cur_idx])
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return int(len(rows)), float("nan")
    return int(len(rows)), float(np.mean(diff))


def boundary_gap_abs(prev_segment: Segment, segment: Segment) -> tuple[int, float]:
    prev_last_row = int(prev_segment.rows[-1])
    cur_first_row = int(segment.rows[0])
    gap_rows = max(0, cur_first_row - prev_last_row - 1)
    if gap_rows <= 0:
        return 0, 0.0
    return gap_rows, float(abs(float(prev_segment.tvt[-1]) - float(segment.tvt[0])))


def add_segment_to_state(
    state: BeamState,
    segment: Segment,
    weights: dict[str, float],
) -> tuple[BeamState, dict[str, Any]]:
    eps = float(weights.get("score_eps", 1e-6))
    score_cost = float(weights.get("score", 1.0)) * -math.log(max(segment.score_prob, eps))
    score_cost += float(weights.get("rank", 0.05)) * float(segment.rank - 1)
    smoothness_cost = float(weights.get("smoothness", 0.01)) * segment.step_abs_mean

    overlap_rows = 0
    overlap_abs = float("nan")
    overlap_cost = 0.0
    gap_rows = 0
    gap_abs = 0.0
    boundary_cost = 0.0
    rank_switch_cost = 0.0
    if state.last_segment is not None:
        overlap_rows, overlap_abs = adjacent_overlap_abs(state.last_segment, segment)
        if overlap_rows > 0 and np.isfinite(overlap_abs):
            overlap_cost = float(weights.get("overlap", 0.04)) * overlap_abs
        else:
            gap_rows, gap_abs = boundary_gap_abs(state.last_segment, segment)
            if gap_rows > 0:
                boundary_cost = float(weights.get("boundary", 0.02)) * gap_abs
        if state.last_segment.rank != segment.rank:
            rank_switch_cost = float(weights.get("rank_switch", 0.02))

    increment = score_cost + smoothness_cost + overlap_cost + boundary_cost + rank_switch_cost
    next_state = BeamState(
        total_cost=state.total_cost + increment,
        score_cost=state.score_cost + score_cost,
        smoothness_cost=state.smoothness_cost + smoothness_cost,
        overlap_cost=state.overlap_cost + overlap_cost,
        boundary_cost=state.boundary_cost + boundary_cost,
        rank_switch_cost=state.rank_switch_cost + rank_switch_cost,
        assignments=(*state.assignments, segment),
        overlap_rows_total=state.overlap_rows_total + overlap_rows,
        gap_count=state.gap_count + int(gap_rows > 0),
        last_segment=segment,
    )
    assignment = {
        "path_npz_sample_index": int(segment.path_npz_sample_index),
        "row_center": int(segment.row_center),
        "rank": int(segment.rank),
        "center_score": float(segment.center_score),
        "score_prob": float(segment.score_prob),
        "segment_step_abs_mean_ft": float(segment.step_abs_mean),
        "segment_step_abs_max_ft": float(segment.step_abs_max),
        "overlap_row_count": int(overlap_rows),
        "overlap_abs_mean_ft": overlap_abs,
        "gap_row_count": int(gap_rows),
        "gap_boundary_abs_ft": gap_abs,
        "incremental_cost": float(increment),
    }
    return next_state, assignment


def stitch_well(
    well_samples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    topk: int,
    beam_width: int,
    output_topn: int,
    weights: dict[str, float],
) -> tuple[list[BeamState], list[dict[str, Any]], dict[str, Any]]:
    samples = well_samples.sort_values("row_center").reset_index(drop=True)
    states: list[tuple[BeamState, tuple[dict[str, Any], ...]]] = [(BeamState.empty(), ())]
    source_rows: set[int] = set()
    center_values: list[int] = []
    for _, sample in samples.iterrows():
        center_values.append(int(sample["row_center"]))
        sample_segments = segments_for_sample(sample=sample, arrays=arrays, topk=topk)
        if not sample_segments:
            continue
        for row in sample_segments[0].rows.tolist():
            source_rows.add(int(row))
        candidates: list[tuple[BeamState, tuple[dict[str, Any], ...]]] = []
        for state, assignments in states:
            for segment in sample_segments:
                next_state, assignment = add_segment_to_state(state, segment, weights)
                candidates.append((next_state, (*assignments, assignment)))
        candidates.sort(key=lambda item: item[0].total_cost)
        states = candidates[:beam_width]

    selected = states[:output_topn]
    assignment_rows: list[dict[str, Any]] = []
    for candidate_index, (state, assignments) in enumerate(selected, start=1):
        for window_order, assignment in enumerate(assignments, start=1):
            row = dict(assignment)
            row.update(
                {
                    "well": str(samples["well"].iloc[0]),
                    "stitched_candidate": f"stitched_path{candidate_index}",
                    "stitched_candidate_rank": candidate_index,
                    "window_order": window_order,
                    "total_cost": float(state.total_cost),
                }
            )
            assignment_rows.append(row)

    centers = np.asarray(center_values, dtype=np.int32)
    gaps = np.diff(np.sort(centers)) if len(centers) > 1 else np.asarray([], dtype=np.int32)
    horizon = int(arrays["pred_tvt_path"].shape[2])
    source_meta = {
        "well": str(samples["well"].iloc[0]) if len(samples) else "",
        "source_window_count": int(len(samples)),
        "source_row_coverage_count": int(len(source_rows)),
        "min_center_gap_rows": int(np.min(gaps)) if len(gaps) else None,
        "max_center_gap_rows": int(np.max(gaps)) if len(gaps) else None,
        "overlap_center_pair_count": int(np.sum(gaps < horizon)) if len(gaps) else 0,
        "gap_center_pair_count": int(np.sum(gaps >= horizon)) if len(gaps) else 0,
    }
    return [state for state, _ in selected], assignment_rows, source_meta


def replay_state_path(
    state: BeamState,
    *,
    well: str,
    candidate_name: str,
) -> dict[str, list[Any]]:
    row_sum: dict[int, float] = {}
    row_weight: dict[int, float] = {}
    row_count: dict[int, int] = {}
    for segment in state.assignments:
        weight = max(float(segment.score_prob), 1e-6)
        for row, tvt in zip(segment.rows.tolist(), segment.tvt.tolist(), strict=False):
            row_int = int(row)
            row_sum[row_int] = row_sum.get(row_int, 0.0) + float(tvt) * weight
            row_weight[row_int] = row_weight.get(row_int, 0.0) + weight
            row_count[row_int] = row_count.get(row_int, 0) + 1

    columns: dict[str, list[Any]] = {
        "id": [],
        "well": [],
        "row_index": [],
        "stitched_candidate": [],
        "stitched_candidate_rank": [],
        "stitched_tvt": [],
        "source_window_count": [],
        "source_weight_sum": [],
        "total_cost": [],
    }
    candidate_rank = int(candidate_name.replace("stitched_path", ""))
    for row in sorted(row_sum):
        weight = row_weight[row]
        columns["id"].append(f"{well}_{row}")
        columns["well"].append(well)
        columns["row_index"].append(row)
        columns["stitched_candidate"].append(candidate_name)
        columns["stitched_candidate_rank"].append(candidate_rank)
        columns["stitched_tvt"].append(row_sum[row] / weight if weight > 0 else np.nan)
        columns["source_window_count"].append(row_count[row])
        columns["source_weight_sum"].append(weight)
        columns["total_cost"].append(float(state.total_cost))
    return columns


def extend_columns(base: dict[str, list[Any]], extra: dict[str, list[Any]]) -> None:
    for key, values in extra.items():
        base.setdefault(key, []).extend(values)


def stitch_all_wells(
    samples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    max_wells: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stitch_cfg = get_nested(config, "stitching") or {}
    topk = int(stitch_cfg.get("local_topk", 10))
    beam_width = int(stitch_cfg.get("beam_width", 6))
    output_topn = int(stitch_cfg.get("output_topn", 3))
    weights = stitch_cfg.get("score_weights") or {}
    max_windows_per_well = stitch_cfg.get("max_windows_per_well")
    topk = min(topk, int(arrays["pred_tvt_path"].shape[1]))

    path_columns: dict[str, list[Any]] = {}
    assignment_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    well_names = sorted(samples["well"].astype(str).unique().tolist())
    if max_wells is not None:
        well_names = well_names[:max_wells]

    for well_index, well in enumerate(well_names, start=1):
        well_samples = samples.loc[samples["well"].astype(str) == well].copy()
        if max_windows_per_well is not None:
            well_samples = well_samples.sort_values("row_center").head(int(max_windows_per_well))
        states, assignments, source_meta = stitch_well(
            well_samples,
            arrays,
            topk=topk,
            beam_width=beam_width,
            output_topn=output_topn,
            weights=weights,
        )
        for candidate_index, state in enumerate(states, start=1):
            extend_columns(
                path_columns,
                replay_state_path(
                    state,
                    well=str(well),
                    candidate_name=f"stitched_path{candidate_index}",
                ),
            )
        assignment_rows.extend(assignments)
        source_meta["well_order"] = well_index
        source_rows.append(source_meta)

    path_rows = pd.DataFrame(path_columns)
    assignments = pd.DataFrame(assignment_rows)
    source_coverage = pd.DataFrame(source_rows)
    meta = {
        "wells_processed": int(len(well_names)),
        "path_rows": int(len(path_rows)),
        "assignment_rows": int(len(assignments)),
        "source_coverage_rows": int(len(source_coverage)),
        "local_topk": int(topk),
        "beam_width": int(beam_width),
        "output_topn": int(output_topn),
        "max_wells": max_wells,
    }
    return path_rows, assignments, source_coverage, meta


def load_candidate_cache(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    eval_cfg = get_nested(config, "candidate_union") or {}
    source_path = find_artifact(
        resolve_config_reference(config, eval_cfg.get("source_cache")),
        fallback_name=DEFAULT_EXP099_CACHE,
    )
    id_col = str(eval_cfg.get("id_column", "id"))
    target_col = str(eval_cfg.get("target_delta_column", "target"))
    last_col = str(eval_cfg.get("last_known_tvt_column", "last_known_tvt"))
    distance_col = str(eval_cfg.get("distance_column", "md_since"))
    requested = [str(value) for value in eval_cfg.get("existing_candidates", [])]
    required = [str(value) for value in eval_cfg.get("required_existing_candidates", [])]

    header = pd.read_csv(source_path, nrows=0).columns.tolist()
    available = [column for column in requested if column in header]
    missing_required = sorted(column for column in required if column not in header)
    if missing_required:
        raise ValueError(f"{source_path} missing required candidates: {missing_required}")
    usecols = [id_col, "well", target_col, last_col, *available]
    if distance_col in header:
        usecols.append(distance_col)
    usecols = list(dict.fromkeys(usecols))
    frame = pd.read_csv(source_path, usecols=usecols, dtype={id_col: str, "well": str})
    frame[id_col] = frame[id_col].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {id_col, "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["true_tvt"] = frame[last_col] + frame[target_col]
    rename = {id_col: "id"}
    if distance_col in frame.columns:
        rename[distance_col] = "md_since"
    frame = frame.rename(columns=rename)
    meta = {
        "path": str(source_path),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "source_csv_gz_sha256": sha256_path(source_path),
        "source_csv_decompressed_sha256": sha256_path(
            source_path,
            decompressed=source_path.suffix == ".gz",
        ),
        "available_existing_candidates": available,
        "missing_existing_candidates": sorted(set(requested).difference(available)),
    }
    return frame, available, meta


def min_abs_error(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    errors = np.abs(values.astype(np.float32) - truth[:, None].astype(np.float32))
    errors[~np.isfinite(values)] = np.nan
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid = np.isfinite(errors).any(axis=1)
    if np.any(valid):
        result[valid] = np.nanmin(errors[valid], axis=1)
    return result


def pairwise_min(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stacked = np.vstack([left, right]).T
    result = np.full(len(left), np.nan, dtype=np.float32)
    valid = np.isfinite(stacked).any(axis=1)
    if np.any(valid):
        result[valid] = np.nanmin(stacked[valid], axis=1)
    return result


def oracle_metric_row(
    *,
    candidate_set: str,
    topk: int,
    candidate_count: int,
    error: np.ndarray,
    within_ft: float,
    existing_error: np.ndarray | None = None,
) -> dict[str, Any]:
    valid = np.isfinite(error)
    row: dict[str, Any] = {
        "candidate_set": candidate_set,
        "topk": int(topk),
        "rows": int(valid.sum()),
        "candidate_count": int(candidate_count),
        "oracle_rmse": None,
        "oracle_mae": None,
        "within10": None,
        "new_best_candidate_rate": None,
        "oracle_rmse_delta_vs_existing": None,
        "within_delta_vs_existing": None,
    }
    if np.any(valid):
        err = error[valid].astype(np.float64)
        row["oracle_rmse"] = float(np.sqrt(np.mean(err * err)))
        row["oracle_mae"] = float(np.mean(err))
        row["within10"] = float(np.mean(err <= within_ft))
    if existing_error is not None:
        both = np.isfinite(error) & np.isfinite(existing_error)
        if np.any(both):
            existing_row = oracle_metric_row(
                candidate_set="existing_reference",
                topk=0,
                candidate_count=0,
                error=existing_error[both],
                within_ft=within_ft,
            )
            row["oracle_rmse_delta_vs_existing"] = (
                float(row["oracle_rmse"] - existing_row["oracle_rmse"])
                if row["oracle_rmse"] is not None
                and existing_row["oracle_rmse"] is not None
                else None
            )
            row["within_delta_vs_existing"] = (
                float(row["within10"] - existing_row["within10"])
                if row["within10"] is not None and existing_row["within10"] is not None
                else None
            )
            row["new_best_candidate_rate"] = float(
                np.mean(error[both] + 1e-6 < existing_error[both])
            )
    return row


def assign_distance_bucket(values: pd.Series, buckets: list[list[float]]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    labels = pd.Series(["unknown"] * len(numeric), index=values.index, dtype=object)
    for low, high in buckets:
        label = f"{int(low)}_{int(high)}" if high < 1_000_000 else f"{int(low)}_plus"
        mask = (numeric >= float(low)) & (numeric < float(high))
        labels.loc[mask] = label
    return labels


def evaluate_union(
    path_rows: pd.DataFrame,
    cache: pd.DataFrame,
    existing_candidates: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_cfg = get_nested(config, "candidate_union") or {}
    topk_values = [int(value) for value in eval_cfg.get("topk_values", [1, 3])]
    within_ft = float(eval_cfg.get("within_ft", 10.0))
    if path_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"status": "empty_stitched_path_rows"}

    wide = path_rows.pivot_table(
        index=["id", "well", "row_index"],
        columns="stitched_candidate",
        values="stitched_tvt",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    stitched_cols = sorted(
        [column for column in wide.columns if str(column).startswith("stitched_path")],
        key=lambda name: int(str(name).replace("stitched_path", "")),
    )
    merged = wide.merge(cache, on="id", how="inner", suffixes=("_stitched", ""))
    if "well" not in merged and "well_stitched" in merged.columns:
        merged = merged.rename(columns={"well_stitched": "well"})
    if merged.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"status": "stitched_cache_join_empty"}

    truth = merged["true_tvt"].to_numpy(np.float32)
    existing_error = min_abs_error(merged[existing_candidates].to_numpy(np.float32), truth)
    metric_rows = [
        oracle_metric_row(
            candidate_set="existing_union_on_stitched_rows",
            topk=0,
            candidate_count=len(existing_candidates),
            error=existing_error,
            within_ft=within_ft,
        )
    ]
    max_topk = min(max(topk_values), len(stitched_cols))
    errors_by_topk: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for topk in topk_values:
        use_cols = stitched_cols[: min(topk, len(stitched_cols))]
        if not use_cols:
            continue
        stitched_error = min_abs_error(merged[use_cols].to_numpy(np.float32), truth)
        union_error = pairwise_min(existing_error, stitched_error)
        errors_by_topk[topk] = (stitched_error, union_error)
        metric_rows.append(
            oracle_metric_row(
                candidate_set=f"stitched_only_top{topk}",
                topk=topk,
                candidate_count=len(use_cols),
                error=stitched_error,
                within_ft=within_ft,
                existing_error=existing_error,
            )
        )
        metric_rows.append(
            oracle_metric_row(
                candidate_set=f"existing_plus_stitched_top{topk}",
                topk=topk,
                candidate_count=len(existing_candidates) + len(use_cols),
                error=union_error,
                within_ft=within_ft,
                existing_error=existing_error,
            )
        )

    metrics_df = pd.DataFrame(metric_rows)
    coverage_df = (
        merged.groupby("well", observed=True)
        .agg(
            covered_rows=("id", "nunique"),
            row_index_min=("row_index", "min"),
            row_index_max=("row_index", "max"),
        )
        .reset_index()
    )
    cache_rows_by_well = cache.groupby("well", observed=True)["id"].nunique().rename("cache_rows")
    coverage_df = coverage_df.merge(cache_rows_by_well, on="well", how="left")
    coverage_df["coverage_rate_vs_cache"] = (
        coverage_df["covered_rows"] / coverage_df["cache_rows"].replace(0, np.nan)
    )

    distance_rows: list[dict[str, Any]] = []
    buckets = eval_cfg.get("distance_buckets") or [
        [0, 50],
        [50, 100],
        [100, 250],
        [250, 500],
        [500, 1000],
        [1000, 1000000000],
    ]
    merged["distance_bucket"] = assign_distance_bucket(merged["md_since"], buckets)
    topk_for_bucket = max_topk
    if topk_for_bucket in errors_by_topk:
        stitched_error, union_error = errors_by_topk[topk_for_bucket]
        metric_context = merged[["well", "distance_bucket"]].copy()
        metric_context["existing_error"] = existing_error
        metric_context["stitched_error"] = stitched_error
        metric_context["union_error"] = union_error
        for bucket, group in metric_context.groupby("distance_bucket", observed=True):
            idx = group.index.to_numpy()
            distance_rows.append(
                {
                    "distance_bucket": bucket,
                    "rows": int(len(group)),
                    "existing_oracle_rmse": oracle_metric_row(
                        candidate_set="existing",
                        topk=0,
                        candidate_count=len(existing_candidates),
                        error=existing_error[idx],
                        within_ft=within_ft,
                    )["oracle_rmse"],
                    "stitched_oracle_rmse": oracle_metric_row(
                        candidate_set="stitched",
                        topk=topk_for_bucket,
                        candidate_count=topk_for_bucket,
                        error=stitched_error[idx],
                        within_ft=within_ft,
                    )["oracle_rmse"],
                    "union_oracle_rmse": oracle_metric_row(
                        candidate_set="union",
                        topk=topk_for_bucket,
                        candidate_count=len(existing_candidates) + topk_for_bucket,
                        error=union_error[idx],
                        within_ft=within_ft,
                    )["oracle_rmse"],
                    "new_best_candidate_rate": float(
                        np.mean(stitched_error[idx] + 1e-6 < existing_error[idx])
                    ),
                }
            )
    distance_df = pd.DataFrame(distance_rows)

    by_well_rows: list[dict[str, Any]] = []
    if topk_for_bucket in errors_by_topk:
        stitched_error, union_error = errors_by_topk[topk_for_bucket]
        context = merged[["well", "id"]].copy()
        context["existing_error"] = existing_error
        context["stitched_error"] = stitched_error
        context["union_error"] = union_error
        for well, group in context.groupby("well", observed=True):
            existing_rmse = oracle_metric_row(
                candidate_set="existing",
                topk=0,
                candidate_count=len(existing_candidates),
                error=group["existing_error"].to_numpy(np.float32),
                within_ft=within_ft,
            )["oracle_rmse"]
            union_rmse = oracle_metric_row(
                candidate_set="union",
                topk=topk_for_bucket,
                candidate_count=len(existing_candidates) + topk_for_bucket,
                error=group["union_error"].to_numpy(np.float32),
                within_ft=within_ft,
            )["oracle_rmse"]
            by_well_rows.append(
                {
                    "well": well,
                    "rows": int(len(group)),
                    "existing_oracle_rmse": existing_rmse,
                    "union_oracle_rmse": union_rmse,
                    "rmse_delta": float(union_rmse - existing_rmse)
                    if existing_rmse is not None and union_rmse is not None
                    else None,
                    "new_best_candidate_rate": float(
                        np.mean(
                            group["stitched_error"].to_numpy(np.float32) + 1e-6
                            < group["existing_error"].to_numpy(np.float32)
                        )
                    ),
                }
            )
    by_well_df = pd.DataFrame(by_well_rows)
    if not by_well_df.empty:
        by_well_df = by_well_df.sort_values(["rmse_delta", "well"], ascending=[True, True])

    summary = {
        "status": "evaluated",
        "stitched_rows": int(len(path_rows)),
        "stitched_wide_rows": int(len(wide)),
        "merged_rows": int(len(merged)),
        "cache_rows": int(len(cache)),
        "row_coverage_rate_vs_cache": float(len(merged) / len(cache)) if len(cache) else None,
        "stitched_candidate_columns": stitched_cols,
        "existing_candidates": existing_candidates,
        "topk_values": topk_values,
    }
    return metrics_df, distance_df, by_well_df, coverage_df, summary


def summarize_physicality(
    path_rows: pd.DataFrame,
    assignments: pd.DataFrame,
    source_coverage: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not source_coverage.empty:
        summary.update(
            {
                "source_wells": int(len(source_coverage)),
                "source_windows_per_well_mean": float(
                    source_coverage["source_window_count"].mean()
                ),
                "source_overlap_wells": int(
                    (source_coverage["overlap_center_pair_count"] > 0).sum()
                ),
                "source_overlap_pair_count": int(
                    source_coverage["overlap_center_pair_count"].sum()
                ),
                "source_gap_pair_count": int(source_coverage["gap_center_pair_count"].sum()),
                "source_row_coverage_count_mean": float(
                    source_coverage["source_row_coverage_count"].mean()
                ),
            }
        )
    if not assignments.empty:
        best = assignments.loc[assignments["stitched_candidate_rank"] == 1].copy()
        rank_dist = (
            best["rank"].value_counts(normalize=True).sort_index().rename("rate").reset_index()
        )
        rank_dist.columns = ["rank", "rate"]
        overlap = assignments.loc[assignments["overlap_row_count"] > 0, "overlap_abs_mean_ft"]
        gap = assignments.loc[assignments["gap_row_count"] > 0, "gap_boundary_abs_ft"]
        summary.update(
            {
                "best_path_rank_distribution": rank_dist.to_dict(orient="records"),
                "assignment_overlap_rows_total": int(assignments["overlap_row_count"].sum()),
                "assignment_overlap_abs_mean_ft": float(overlap.mean())
                if len(overlap)
                else None,
                "assignment_gap_boundary_abs_mean_ft": float(gap.mean()) if len(gap) else None,
                "assignment_gap_boundary_abs_p95_ft": float(gap.quantile(0.95))
                if len(gap)
                else None,
            }
        )
    if not path_rows.empty:
        path_metrics: list[dict[str, Any]] = []
        for (well, candidate), group in path_rows.groupby(
            ["well", "stitched_candidate"],
            observed=True,
        ):
            ordered = group.sort_values("row_index")
            tvt = ordered["stitched_tvt"].to_numpy(np.float32)
            rows = ordered["row_index"].to_numpy(np.int32)
            diffs = np.abs(np.diff(tvt))
            row_gaps = np.diff(rows)
            curvature = np.abs(np.diff(tvt, n=2)) if len(tvt) >= 3 else np.asarray([])
            path_metrics.append(
                {
                    "well": well,
                    "stitched_candidate": candidate,
                    "rows": int(len(ordered)),
                    "path_step_abs_mean_ft": float(np.nanmean(diffs)) if len(diffs) else 0.0,
                    "path_step_abs_p95_ft": float(np.nanpercentile(diffs, 95))
                    if len(diffs)
                    else 0.0,
                    "curvature_abs_mean_ft": float(np.nanmean(curvature))
                    if len(curvature)
                    else 0.0,
                    "row_gap_count": int(np.sum(row_gaps > 1)) if len(row_gaps) else 0,
                    "overlap_row_rate": float(
                        np.mean(ordered["source_window_count"].to_numpy(np.float32) > 1)
                    ),
                }
            )
        path_metrics_df = pd.DataFrame(path_metrics)
        summary.update(
            {
                "stitched_path_step_abs_mean_ft": float(
                    path_metrics_df["path_step_abs_mean_ft"].mean()
                ),
                "stitched_path_step_abs_p95_ft": float(
                    path_metrics_df["path_step_abs_p95_ft"].quantile(0.95)
                ),
                "stitched_curvature_abs_mean_ft": float(
                    path_metrics_df["curvature_abs_mean_ft"].mean()
                ),
                "stitched_row_gap_count_total": int(path_metrics_df["row_gap_count"].sum()),
                "stitched_overlap_row_rate_mean": float(path_metrics_df["overlap_row_rate"].mean()),
            }
        )
    return summary


def run_stitch_probe(
    *,
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
    max_wells: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    config = config or load_config()
    paths = paths or ExperimentPaths()
    paths.ensure_output_dirs()

    arrays, samples, path_meta = load_candidate_path_inputs(config)
    if debug and max_wells is None:
        max_wells = int(get_nested(config, "stitching.debug_max_wells") or 3)

    path_rows, assignments, source_coverage, stitch_meta = stitch_all_wells(
        samples,
        arrays,
        config,
        max_wells=max_wells,
    )
    cache, existing_candidates, cache_meta = load_candidate_cache(config)
    metrics_df, distance_df, by_well_df, coverage_df, eval_summary = evaluate_union(
        path_rows,
        cache,
        existing_candidates,
        config,
    )
    physical_summary = summarize_physicality(path_rows, assignments, source_coverage)

    path_rows_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_stitched_path_rows.csv.gz"
    assignments_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_stitched_window_assignments.csv.gz"
    source_coverage_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_source_window_coverage.csv"
    coverage_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_stitched_coverage_by_well.csv"
    metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_metrics.csv"
    distance_path = (
        paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_distance_bucket_metrics.csv"
    )
    by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_by_well.csv"
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"

    gzip_csv(path_rows, path_rows_path)
    gzip_csv(assignments, assignments_path)
    source_coverage.to_csv(source_coverage_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    distance_df.to_csv(distance_path, index=False)
    by_well_df.to_csv(by_well_path, index=False)

    output_paths = {
        "stitched_path_rows": str(path_rows_path),
        "stitched_window_assignments": str(assignments_path),
        "source_window_coverage": str(source_coverage_path),
        "stitched_coverage_by_well": str(coverage_path),
        "candidate_union_metrics": str(metrics_path),
        "candidate_union_distance_bucket_metrics": str(distance_path),
        "candidate_union_by_well": str(by_well_path),
        "summary": str(summary_path),
    }
    output_sha = {
        "stitched_path_rows_csv_gz_sha256": sha256_path(path_rows_path),
        "stitched_path_rows_csv_decompressed_sha256": sha256_path(
            path_rows_path,
            decompressed=True,
        ),
        "stitched_window_assignments_csv_gz_sha256": sha256_path(assignments_path),
        "stitched_window_assignments_csv_decompressed_sha256": sha256_path(
            assignments_path,
            decompressed=True,
        ),
        "source_window_coverage_csv_sha256": sha256_path(source_coverage_path),
        "candidate_union_metrics_csv_sha256": sha256_path(metrics_path),
        "candidate_union_distance_bucket_metrics_csv_sha256": sha256_path(distance_path),
        "candidate_union_by_well_csv_sha256": sha256_path(by_well_path),
    }

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "debug_completed" if debug else "implemented_diagnostic_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "debug": bool(debug),
        "max_wells": max_wells,
        "route": get_nested(config, "experiment.route"),
        "path_inputs": path_meta,
        "candidate_cache": cache_meta,
        "stitching": stitch_meta,
        "evaluation": eval_summary,
        "physicality": physical_summary,
        "metrics": metrics_df.to_dict(orient="records"),
        "distance_metrics": distance_df.to_dict(orient="records"),
        "output_paths": output_paths,
        "output_sha256": output_sha,
        "leakage_guard": {
            "stitch_score_uses_target": False,
            "target_columns_read_for_stitch_score": [],
            "target_usage": "candidate cache target is used only after stitched paths are fixed",
        },
    }
    write_json(summary_path, summary)

    experiment_metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "debug_completed" if debug else "kaggle_train_diagnostic_completed",
        "route": get_nested(config, "experiment.route"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "key_idea": get_nested(config, "lineage.diff_summary"),
        "parent": get_nested(config, "lineage.parent"),
        "debug": bool(debug),
        "max_wells": max_wells,
        "summary_path": str(summary_path),
        "candidate_union_metrics": metrics_df.to_dict(orient="records"),
        "physicality": physical_summary,
        "coverage": eval_summary,
        "input_sha256": {
            "path_npz": path_meta.get("path_npz_sha256"),
            "path_samples_decompressed": path_meta.get("path_samples_csv_decompressed_sha256"),
            "candidate_cache_decompressed": cache_meta.get("source_csv_decompressed_sha256"),
        },
        "output_sha256": output_sha,
        "notes": [
            "This is a CPU diagnostic over cached exp202 local path artifacts.",
            (
                "The current exp202 path artifact is sparse validation-sample output; "
                "overlap coverage is reported explicitly."
            ),
            (
                "No direct TVT replacement, softmax averaging, PF weight replacement, "
                "inference, or submission is performed."
            ),
        ],
    }
    write_json(paths.metrics_path, experiment_metrics)
    return summary
