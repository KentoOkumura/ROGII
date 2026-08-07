from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    logical_frame_sha256,
    sha256_file,
    sha256_json,
    to_jsonable,
    write_json,
)

METADATA_COLUMNS = [
    "block_key",
    "well",
    "block_id",
    "outer_fold",
    "rows",
    "well_row_idx_start",
    "well_row_idx_end",
]
FEATURE_PREFIXES = ("context__", "raw__", "confidence__", "pair__", "bank__")


@dataclass(frozen=True)
class PrimitiveFold:
    base: pd.DataFrame
    values: np.ndarray
    available: np.ndarray
    confidence: dict[str, pd.DataFrame]
    primitive_ids: list[str]


def primitive_ids(contract: Mapping[str, Any]) -> list[str]:
    return [str(item["id"]) for item in contract.get("primitives", [])]


def primitive_family_map(contract: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["id"]): str(item["family"]) for item in contract.get("primitives", [])}


def pair_ids(ids: Sequence[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations([str(item) for item in ids], 2))


def validate_regime_contract(contract: Mapping[str, Any]) -> None:
    ids = primitive_ids(contract)
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("regime contract requires exactly six unique primitive candidates")
    families = primitive_family_map(contract)
    if set(families) != set(ids) or any(not family for family in families.values()):
        raise ValueError("every primitive must have a non-empty family")
    expected_pairs = int(contract.get("pair_policy", {}).get("expected_count", -1))
    if len(pair_ids(ids)) != expected_pairs:
        raise ValueError("pair count does not match regime contract")
    policy = contract.get("feature_policy", {})
    if str(policy.get("candidate_absolute_value")) != "forbidden":
        raise ValueError("candidate absolute values must be forbidden")
    if str(policy.get("target_or_error_derived")) != "forbidden":
        raise ValueError("target/error-derived regime features must be forbidden")


def _read_partition(root: Path, kind: str, candidate_id: str, fold: int) -> pd.DataFrame:
    paths = sorted((Path(root) / kind / candidate_id / f"fold={fold}").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing {kind}/{candidate_id}/fold={fold}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _assert_key_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise ValueError(f"candidate row count mismatch: {len(left)} != {len(right)}")
    for column in KEY_COLUMNS:
        left_values = left[column].to_numpy()
        right_values = right[column].to_numpy()
        if column == "md_since":
            equal = np.array_equal(left_values, right_values, equal_nan=True)
        else:
            equal = np.array_equal(left_values, right_values)
        if not equal:
            raise ValueError(f"candidate key mismatch in {column}")


def load_primitive_fold(
    cache_root: Path,
    contract: Mapping[str, Any],
    fold: int,
) -> PrimitiveFold:
    """Load only the six primitive paths; formula candidates are deliberately excluded."""

    validate_regime_contract(contract)
    ids = primitive_ids(contract)
    value_frames: dict[str, pd.DataFrame] = {}
    confidence: dict[str, pd.DataFrame] = {}
    for candidate_id in ids:
        values = _read_partition(cache_root, "candidate_values", candidate_id, fold)
        values = values.sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)
        conf = _read_partition(cache_root, "candidate_confidence", candidate_id, fold)
        conf = conf.sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)
        _assert_key_alignment(values, conf)
        value_frames[candidate_id] = values
        confidence[candidate_id] = conf

    first = value_frames[ids[0]]
    for candidate_id in ids[1:]:
        _assert_key_alignment(first, value_frames[candidate_id])
    base = first[KEY_COLUMNS].copy()
    values = np.column_stack(
        [
            pd.to_numeric(value_frames[candidate_id]["candidate_tvt"], errors="coerce").to_numpy(
                np.float32
            )
            for candidate_id in ids
        ]
    )
    available = np.column_stack(
        [
            value_frames[candidate_id]["candidate_available"].astype(bool).to_numpy()
            for candidate_id in ids
        ]
    ) & np.isfinite(values)
    return PrimitiveFold(base, values, available, confidence, ids)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def _safe_mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if len(finite) else math.nan


def _safe_std(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.std(finite)) if len(finite) else math.nan


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, quantile)) if len(finite) else math.nan


def _safe_range(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.max(finite) - np.min(finite)) if len(finite) else math.nan


def _safe_end_minus_start(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite[-1] - finite[0]) if len(finite) else math.nan


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return math.nan
    x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)[finite]
    y = values[finite]
    x_centered = x - np.mean(x)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y - np.mean(y)) / denominator)


def _slope_change(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 4:
        return 0.0
    midpoint = len(values) // 2
    first = _slope(values[:midpoint])
    second = _slope(values[midpoint:])
    return float(second - first) if np.isfinite(first) and np.isfinite(second) else math.nan


def _first_diff_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    left_diff = np.diff(np.asarray(left, dtype=np.float64))
    right_diff = np.diff(np.asarray(right, dtype=np.float64))
    finite = np.isfinite(left_diff) & np.isfinite(right_diff)
    if finite.sum() < 2:
        return 0.0
    left_diff = left_diff[finite]
    right_diff = right_diff[finite]
    if np.std(left_diff) <= 1e-12 or np.std(right_diff) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_diff, right_diff)[0, 1])


def _sign_metrics(gap: np.ndarray) -> tuple[float, float]:
    signs = np.sign(_finite(gap))
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0, 1.0
    changes = signs[1:] != signs[:-1]
    return float(np.mean(changes)), float(np.mean(~changes))


def _divergence_expansion_ratio(gap: np.ndarray) -> float:
    absolute = np.abs(np.asarray(gap, dtype=np.float64))
    if len(absolute) == 0:
        return math.nan
    quarter = max(len(absolute) // 4, 1)
    start = _safe_mean(absolute[:quarter])
    end = _safe_mean(absolute[-quarter:])
    if not np.isfinite(start) or not np.isfinite(end):
        return math.nan
    if start <= 1e-6 and end <= 1e-6:
        return 1.0
    return float(np.clip(end / max(start, 1e-6), 0.0, 1000.0))


def _pair_features(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    gap = left - right
    absolute = np.abs(gap)
    zero_crossing_rate, sign_persistence = _sign_metrics(gap)
    finite_gap = _finite(gap)
    return {
        "gap_mean": _safe_mean(gap),
        "gap_end": float(finite_gap[-1]) if len(finite_gap) else math.nan,
        "gap_std": _safe_std(gap),
        "abs_gap_mean": _safe_mean(absolute),
        "abs_gap_p90": _safe_quantile(absolute, 0.90),
        "abs_gap_max": float(np.max(_finite(absolute))) if len(_finite(absolute)) else math.nan,
        "gap_slope": _slope(gap),
        "gap_slope_change": _slope_change(gap),
        "first_diff_correlation": _first_diff_correlation(left, right),
        "zero_crossing_rate": zero_crossing_rate,
        "sign_persistence": sign_persistence,
        "divergence_expansion_ratio": _divergence_expansion_ratio(gap),
    }


def _bank_features(values: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("bank matrix must be rows by candidates")
    row_ranges = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)
    rank_switch_rate = 0.0
    if len(matrix) > 1:
        ranks = np.argsort(np.argsort(matrix, axis=1, kind="stable"), axis=1, kind="stable")
        rank_switch_rate = float(np.mean(np.any(ranks[1:] != ranks[:-1], axis=1)))
    filled = matrix.copy()
    for column in range(filled.shape[1]):
        finite = np.isfinite(filled[:, column])
        fill = float(np.median(filled[finite, column])) if finite.any() else 0.0
        filled[~finite, column] = fill
    centered = filled - np.mean(filled, axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    singular_energy = np.square(singular_values)
    shares = singular_energy / max(float(np.sum(singular_energy)), 1e-12)
    nonzero = shares[shares > 1e-12]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if len(nonzero) else 1.0
    return {
        "range_mean": _safe_mean(row_ranges),
        "range_p90": _safe_quantile(row_ranges, 0.90),
        "range_max": float(np.max(_finite(row_ranges))) if len(_finite(row_ranges)) else math.nan,
        "rank_switch_rate": rank_switch_rate,
        "singular_value_1_share": float(shares[0]) if len(shares) else 1.0,
        "singular_value_2_share": float(shares[1]) if len(shares) > 1 else 0.0,
        "effective_rank": effective_rank,
    }


def _raw_block_features(raw: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for column in columns:
        values = (
            pd.to_numeric(raw[column], errors="coerce").to_numpy(np.float64)
            if column in raw.columns
            else np.full(len(raw), np.nan, dtype=np.float64)
        )
        prefix = f"raw__{column.lower()}"
        features[f"{prefix}__mean"] = _safe_mean(values)
        features[f"{prefix}__std"] = _safe_std(values)
        features[f"{prefix}__range"] = _safe_range(values)
        features[f"{prefix}__end_minus_start"] = _safe_end_minus_start(values)
        features[f"{prefix}__finite_fraction"] = float(np.isfinite(values).mean())
    return features


def _confidence_block_features(
    confidence: Mapping[str, pd.DataFrame],
    primitive_names: Sequence[str],
    positions: np.ndarray,
    slots: Sequence[str],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for candidate_id in primitive_names:
        frame = confidence[candidate_id].iloc[positions]
        valid = (
            frame["confidence_valid"].astype(bool).to_numpy()
            if "confidence_valid" in frame
            else np.zeros(len(frame), dtype=bool)
        )
        features[f"confidence__{candidate_id}__valid_fraction"] = float(np.mean(valid))
        for slot in slots:
            values = (
                pd.to_numeric(frame[slot], errors="coerce").to_numpy(np.float64)
                if slot in frame
                else np.full(len(frame), np.nan, dtype=np.float64)
            )
            features[f"confidence__{candidate_id}__{slot}__median"] = _safe_quantile(values, 0.50)
    return features


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith(FEATURE_PREFIXES))


def assert_target_free_feature_schema(
    columns: Sequence[str], forbidden_tokens: Sequence[str]
) -> None:
    forbidden_segments = {str(token).lower() for token in forbidden_tokens}
    hits = [
        column for column in columns if forbidden_segments.intersection(column.lower().split("__"))
    ]
    if hits:
        raise ValueError(f"forbidden regime features: {hits[:20]}")
    if not any(column.startswith("pair__") for column in columns):
        raise ValueError("pairwise candidate-difference features are missing")


def build_block_fingerprints(
    bundle: PrimitiveFold,
    raw_train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target-free fixed-block fingerprints and a row-to-block map."""

    feature_cfg = dict(config.get("features", {}))
    block_size = int(feature_cfg.get("block_size", 512))
    raw_cfg = dict(feature_cfg.get("raw_context", {}))
    raw_columns = [str(item) for item in raw_cfg.get("horizontal_numeric_allowlist", [])]
    forbidden_raw = {str(item).lower() for item in raw_cfg.get("forbidden_columns", [])}
    if forbidden_raw.intersection(column.lower() for column in raw_columns):
        raise ValueError("raw allowlist contains a forbidden target column")
    slots = [str(item) for item in feature_cfg.get("confidence_slots", [])]
    ids = bundle.primitive_ids
    expected_pairs = int(config.get("guards", {}).get("technical", {}).get("expected_pairs", 15))
    if len(pair_ids(ids)) != expected_pairs:
        raise ValueError("unexpected primitive pair count")

    records: list[dict[str, Any]] = []
    row_maps: list[pd.DataFrame] = []
    base = bundle.base.reset_index(drop=True)
    for well, raw_positions in base.groupby("well", sort=False).indices.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        well_base = base.iloc[positions].reset_index(drop=True)
        row_indices = pd.to_numeric(well_base["well_row_idx"], errors="raise").to_numpy(np.int64)
        raw_path = Path(raw_train_dir) / f"{well}__horizontal_well.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw = pd.read_csv(raw_path)
        if row_indices.min(initial=0) < 0 or row_indices.max(initial=-1) >= len(raw):
            raise ValueError(f"raw row index out of bounds for well={well}")
        selected_raw = raw.iloc[row_indices].reset_index(drop=True)
        local_block_ids = np.arange(len(positions), dtype=np.int32) // block_size
        fold_values = bundle.values[positions]
        fold_available = bundle.available[positions]

        row_map = well_base[["id", "well", "well_row_idx", "outer_fold"]].copy()
        row_map["block_id"] = local_block_ids
        row_map["block_key"] = [f"{well}::{block_id:04d}" for block_id in local_block_ids]
        row_maps.append(row_map)

        for block_id in np.unique(local_block_ids):
            local = np.flatnonzero(local_block_ids == block_id)
            absolute_positions = positions[local]
            values = fold_values[local].astype(np.float64, copy=True)
            values[~fold_available[local]] = np.nan
            record: dict[str, Any] = {
                "block_key": f"{well}::{int(block_id):04d}",
                "well": str(well),
                "block_id": int(block_id),
                "outer_fold": int(well_base.iloc[local[0]]["outer_fold"]),
                "rows": int(len(local)),
                "well_row_idx_start": int(row_indices[local[0]]),
                "well_row_idx_end": int(row_indices[local[-1]]),
                "context__evaluation_progress_start": float(local[0] / max(len(positions), 1)),
                "context__evaluation_progress_end": float((local[-1] + 1) / max(len(positions), 1)),
                "context__block_row_fraction": float(len(local) / block_size),
                "context__md_since_start": float(
                    pd.to_numeric(well_base.iloc[local[0]]["md_since"], errors="coerce")
                ),
                "context__md_since_end": float(
                    pd.to_numeric(well_base.iloc[local[-1]]["md_since"], errors="coerce")
                ),
            }
            record.update(_raw_block_features(selected_raw.iloc[local], raw_columns))
            record.update(
                _confidence_block_features(bundle.confidence, ids, absolute_positions, slots)
            )
            for left_index, right_index in itertools.combinations(range(len(ids)), 2):
                pair_name = f"{ids[left_index]}__vs__{ids[right_index]}"
                for metric, value in _pair_features(
                    values[:, left_index], values[:, right_index]
                ).items():
                    record[f"pair__{pair_name}__{metric}"] = value
            for metric, value in _bank_features(values).items():
                record[f"bank__{metric}"] = value
            records.append(record)

    blocks = (
        pd.DataFrame.from_records(records)
        .sort_values(["outer_fold", "well", "block_id"], kind="stable")
        .reset_index(drop=True)
    )
    row_map = (
        pd.concat(row_maps, ignore_index=True)
        .sort_values(["outer_fold", "well", "well_row_idx"], kind="stable")
        .reset_index(drop=True)
    )
    columns = feature_columns(blocks)
    forbidden_tokens = [str(item) for item in feature_cfg.get("forbidden_feature_tokens", [])]
    assert_target_free_feature_schema(columns, forbidden_tokens)
    if blocks["block_key"].duplicated().any():
        raise ValueError("duplicate block keys")
    if row_map["id"].duplicated().any():
        raise ValueError("duplicate row ids")
    return blocks, row_map


def _canonical_label_map(centers: np.ndarray) -> dict[int, int]:
    order = sorted(
        range(len(centers)),
        key=lambda index: tuple(np.round(centers[index], decimals=10).tolist()),
    )
    return {old: canonical for canonical, old in enumerate(order)}


def _soft_membership(distances: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    logits = -np.asarray(distances, dtype=np.float64) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)


def fit_outer_fold_regimes(
    blocks: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = feature_columns(blocks)
    forbidden = [
        str(item) for item in config.get("features", {}).get("forbidden_feature_tokens", [])
    ]
    assert_target_free_feature_schema(columns, forbidden)
    regime_cfg = dict(config.get("regime", {}))
    n_clusters = int(regime_cfg.get("n_clusters", 3))
    n_init = int(regime_cfg.get("kmeans_n_init", 20))
    quantile_range = tuple(float(item) for item in regime_cfg.get("quantile_range", [25, 75]))
    seed = int(config.get("reproducibility", {}).get("seed", 42))
    folds = sorted(int(item) for item in blocks["outer_fold"].unique())
    assignments: list[pd.DataFrame] = []
    centroid_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []

    for fold in folds:
        train_mask = blocks["outer_fold"].to_numpy() != fold
        valid_mask = ~train_mask
        train_raw = blocks.loc[train_mask, columns].apply(pd.to_numeric, errors="coerce")
        valid_raw = blocks.loc[valid_mask, columns].apply(pd.to_numeric, errors="coerce")
        medians = train_raw.median(axis=0, skipna=True).fillna(0.0)
        train_imputed = train_raw.fillna(medians).to_numpy(np.float64)
        valid_imputed = valid_raw.fillna(medians).to_numpy(np.float64)
        scaler = RobustScaler(quantile_range=quantile_range)
        train_scaled = scaler.fit_transform(train_imputed)
        valid_scaled = scaler.transform(valid_imputed)

        primary = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=seed + fold,
            algorithm="lloyd",
        ).fit(train_scaled)
        audit = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=seed + 10000 + fold,
            algorithm="lloyd",
        ).fit(train_scaled)
        original_centers = scaler.inverse_transform(primary.cluster_centers_)
        canonical = _canonical_label_map(original_centers)
        canonical_order = [old for old, _ in sorted(canonical.items(), key=lambda item: item[1])]
        canonical_scaled_centers = primary.cluster_centers_[canonical_order]
        canonical_original_centers = original_centers[canonical_order]

        primary_distances = cdist(valid_scaled, canonical_scaled_centers)
        primary_labels = np.argmin(primary_distances, axis=1).astype(np.int16)
        nearest_train = np.min(primary.transform(train_scaled), axis=1)
        positive_nearest = nearest_train[np.isfinite(nearest_train) & (nearest_train > 0)]
        temperature = float(np.median(positive_nearest)) if len(positive_nearest) else 1.0
        probabilities = _soft_membership(primary_distances, temperature)
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
        entropy /= max(math.log(n_clusters), 1e-12)

        primary_rows, audit_columns = linear_sum_assignment(
            cdist(primary.cluster_centers_, audit.cluster_centers_)
        )
        audit_to_primary = {
            int(audit_label): int(primary_label)
            for primary_label, audit_label in zip(primary_rows, audit_columns, strict=True)
        }
        audit_raw_labels = audit.predict(valid_scaled)
        audit_labels = np.asarray(
            [canonical[audit_to_primary[int(label)]] for label in audit_raw_labels],
            dtype=np.int16,
        )
        agreement = audit_labels == primary_labels

        output = blocks.loc[valid_mask, METADATA_COLUMNS].copy().reset_index(drop=True)
        output["regime"] = primary_labels
        output["regime_distance"] = np.min(primary_distances, axis=1).astype(np.float32)
        output["regime_entropy"] = entropy.astype(np.float32)
        output["assignment_audit_regime"] = audit_labels
        output["assignment_stable"] = agreement
        for regime in range(n_clusters):
            output[f"regime_probability_{regime}"] = probabilities[:, regime].astype(np.float32)
        assignments.append(output)

        for regime in range(n_clusters):
            record: dict[str, Any] = {
                "outer_fold": fold,
                "regime": regime,
                "temperature": temperature,
                "outer_train_blocks": int(train_mask.sum()),
                "outer_valid_blocks": int(valid_mask.sum()),
            }
            record.update(
                {
                    column: float(canonical_original_centers[regime, index])
                    for index, column in enumerate(columns)
                }
            )
            centroid_records.append(record)
        stability_records.append(
            {
                "outer_fold": fold,
                "outer_valid_blocks": int(valid_mask.sum()),
                "centroid_matched_assignment_agreement": float(np.mean(agreement)),
            }
        )

    assignment_frame = (
        pd.concat(assignments, ignore_index=True)
        .sort_values(["outer_fold", "well", "block_id"], kind="stable")
        .reset_index(drop=True)
    )
    centroids = pd.DataFrame.from_records(centroid_records)
    stability = pd.DataFrame.from_records(stability_records)
    if len(assignment_frame) != len(blocks):
        raise ValueError("OOF regime assignment does not cover every block")
    return assignment_frame, centroids, stability


def expand_row_assignments(
    row_map: pd.DataFrame,
    block_assignments: pd.DataFrame,
) -> pd.DataFrame:
    assignment_columns = [
        "block_key",
        "regime",
        "regime_distance",
        "regime_entropy",
        "assignment_stable",
        *sorted(
            column
            for column in block_assignments.columns
            if column.startswith("regime_probability_")
        ),
    ]
    expanded = row_map.merge(
        block_assignments[assignment_columns], on="block_key", how="left", validate="many_to_one"
    )
    if expanded["regime"].isna().any():
        raise ValueError("row-to-block regime expansion has missing assignments")
    expanded["regime"] = expanded["regime"].astype(np.int16)
    return expanded


def evaluate_regime_separability(
    block_assignments: pd.DataFrame,
    row_assignments: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ids = primitive_ids(contract)
    families = primitive_family_map(contract)
    required = {
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "candidate_id",
        "pred_abs_error",
        "actual_abs_error",
    }
    missing = sorted(required - set(candidate_scores.columns))
    if missing:
        raise ValueError(f"exp264 candidate score is incomplete: missing={missing}")
    scores = candidate_scores[candidate_scores["candidate_id"].isin(ids)].copy()
    if "candidate_available" in scores:
        scores = scores[scores["candidate_available"].astype(bool)].copy()
    if scores.empty or set(scores["candidate_id"].unique()) != set(ids):
        raise ValueError("exp264 score does not contain all six primitive candidates")
    join_columns = ["id", "well", "well_row_idx", "outer_fold"]
    assignment_columns = join_columns + ["block_key", "regime"]
    merged = scores.merge(
        row_assignments[assignment_columns],
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    if merged["regime"].isna().any():
        raise ValueError("candidate score rows are not fully covered by regime assignments")
    merged["candidate_family"] = merged["candidate_id"].map(families)
    merged["actual_squared_error"] = np.square(
        pd.to_numeric(merged["actual_abs_error"], errors="coerce")
    )
    merged["calibration_error"] = pd.to_numeric(
        merged["pred_abs_error"], errors="coerce"
    ) - pd.to_numeric(merged["actual_abs_error"], errors="coerce")
    if not np.isfinite(merged["actual_squared_error"]).all():
        raise ValueError("candidate score contains non-finite actual errors")

    grouped = merged.groupby(["outer_fold", "regime", "candidate_id", "candidate_family"])
    metrics = grouped.agg(
        rows=("id", "size"),
        wells=("well", "nunique"),
        actual_mae=("actual_abs_error", "mean"),
        actual_mse=("actual_squared_error", "mean"),
        predicted_abs_error_mean=("pred_abs_error", "mean"),
        calibration_bias=("calibration_error", "mean"),
    ).reset_index()
    metrics["actual_rmse"] = np.sqrt(metrics.pop("actual_mse"))
    pooled = (
        merged.groupby(["regime", "candidate_id", "candidate_family"])
        .agg(
            rows=("id", "size"),
            wells=("well", "nunique"),
            actual_mae=("actual_abs_error", "mean"),
            actual_mse=("actual_squared_error", "mean"),
            predicted_abs_error_mean=("pred_abs_error", "mean"),
            calibration_bias=("calibration_error", "mean"),
        )
        .reset_index()
    )
    pooled["actual_rmse"] = np.sqrt(pooled.pop("actual_mse"))
    pooled["outer_fold"] = "all"
    metrics["outer_fold"] = metrics["outer_fold"].astype(str)
    metrics = pd.concat([metrics, pooled[metrics.columns]], ignore_index=True)
    calibration = metrics[
        [
            "outer_fold",
            "regime",
            "candidate_id",
            "candidate_family",
            "rows",
            "predicted_abs_error_mean",
            "actual_mae",
            "calibration_bias",
        ]
    ].copy()

    regime_cfg = dict(config.get("regime", {}))
    n_clusters = int(regime_cfg.get("n_clusters", 3))
    folds = sorted(int(item) for item in block_assignments["outer_fold"].unique())
    occupancy = (
        block_assignments.groupby(["outer_fold", "regime"])
        .agg(blocks=("block_key", "size"), wells=("well", "nunique"))
        .reindex(
            pd.MultiIndex.from_product([folds, range(n_clusters)], names=["outer_fold", "regime"])
        )
        .fillna(0)
        .reset_index()
    )
    fold_blocks = occupancy.groupby("outer_fold")["blocks"].transform("sum")
    occupancy["block_share"] = occupancy["blocks"] / np.maximum(fold_blocks, 1)
    occupancy_cfg = dict(config.get("guards", {}).get("occupancy", {}))
    occupancy["guard_pass"] = (
        occupancy["wells"] >= int(occupancy_cfg["min_wells_per_regime_fold"])
    ) & (occupancy["block_share"] >= float(occupancy_cfg["min_block_share_per_regime_fold"]))
    pass_counts = occupancy.groupby("regime")["guard_pass"].sum()
    occupancy_guard = bool(
        (pass_counts >= int(occupancy_cfg["min_passing_folds_per_regime"])).all()
    )

    stability_value = float(
        np.average(
            stability["centroid_matched_assignment_agreement"],
            weights=stability["outer_valid_blocks"],
        )
    )
    stability_guard = stability_value >= float(
        config.get("guards", {}).get("stability", {})["min_centroid_matched_assignment_agreement"]
    )
    pooled_only = pooled.copy()
    winners = pooled_only.loc[pooled_only.groupby("regime")["actual_rmse"].idxmin()].sort_values(
        "regime"
    )
    distinct_best_families = int(winners["candidate_family"].nunique())
    regime_bias = merged.groupby("regime")["calibration_error"].mean()
    calibration_bias_range = float(regime_bias.max() - regime_bias.min())
    separability_cfg = dict(config.get("guards", {}).get("separability", {}))
    family_guard = distinct_best_families >= int(
        separability_cfg["min_distinct_best_candidate_families"]
    )
    calibration_guard = calibration_bias_range >= float(
        separability_cfg["min_global_calibration_bias_range_ft"]
    )
    separability_guard = bool(family_guard or calibration_guard)
    summary = {
        "rows_scored": int(merged["id"].nunique()),
        "score_rows_candidate_long": int(len(merged)),
        "regime_count": n_clusters,
        "occupancy_guard_pass": occupancy_guard,
        "stability": {
            "centroid_matched_assignment_agreement": stability_value,
            "guard_pass": bool(stability_guard),
        },
        "separability": {
            "best_candidate_by_regime": [
                {
                    "regime": int(row.regime),
                    "candidate_id": str(row.candidate_id),
                    "candidate_family": str(row.candidate_family),
                    "actual_rmse": float(row.actual_rmse),
                }
                for row in winners.itertuples(index=False)
            ],
            "distinct_best_candidate_families": distinct_best_families,
            "family_guard_pass": bool(family_guard),
            "global_calibration_bias_range_ft": calibration_bias_range,
            "calibration_guard_pass": bool(calibration_guard),
            "guard_pass": separability_guard,
        },
        "stage0_guard_pass": bool(occupancy_guard and stability_guard and separability_guard),
    }
    return occupancy, metrics, {**summary, "calibration": calibration}


def evaluate_regime_separability_from_parquet(
    block_assignments: pd.DataFrame,
    row_assignments: pd.DataFrame,
    candidate_score_path: Path,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    stability: pd.DataFrame,
    *,
    batch_size: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stream the large exp264 candidate-long artifact and retain only aggregates."""

    import pyarrow.parquet as pq

    path = Path(candidate_score_path)
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    required = {
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "candidate_id",
        "pred_abs_error",
        "actual_abs_error",
    }
    missing = sorted(required - set(parquet.schema.names))
    if missing:
        raise ValueError(f"exp264 candidate score is incomplete: missing={missing}")
    columns = sorted(required)
    if "candidate_available" in parquet.schema.names:
        columns.append("candidate_available")

    ids = primitive_ids(contract)
    families = primitive_family_map(contract)
    lookup = row_assignments.set_index("id")[["regime", "outer_fold"]]
    if not lookup.index.is_unique:
        raise ValueError("row assignment id must be unique")
    fold_stats: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "wells": set(),
            "actual_abs_sum": 0.0,
            "actual_sq_sum": 0.0,
            "predicted_sum": 0.0,
            "calibration_sum": 0.0,
        }
    )
    pooled_stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "wells": set(),
            "actual_abs_sum": 0.0,
            "actual_sq_sum": 0.0,
            "predicted_sum": 0.0,
            "calibration_sum": 0.0,
        }
    )
    regime_calibration: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    rows_by_candidate: dict[str, int] = defaultdict(int)
    batches = 0

    def update_stats(target: dict[str, Any], group: pd.DataFrame) -> None:
        actual = group["actual_abs_error"].to_numpy(np.float64)
        predicted = group["pred_abs_error"].to_numpy(np.float64)
        target["rows"] += int(len(group))
        target["wells"].update(group["well"].astype(str).unique().tolist())
        target["actual_abs_sum"] += float(np.sum(actual))
        target["actual_sq_sum"] += float(np.dot(actual, actual))
        target["predicted_sum"] += float(np.sum(predicted))
        target["calibration_sum"] += float(np.sum(predicted - actual))

    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        batches += 1
        frame = batch.to_pandas()
        frame = frame[frame["candidate_id"].isin(ids)].copy()
        if "candidate_available" in frame:
            frame = frame[frame["candidate_available"].astype(bool)].copy()
        if frame.empty:
            continue
        mapped = lookup.reindex(frame["id"].to_numpy())
        if mapped["regime"].isna().any():
            raise ValueError("candidate score batch has ids without regime assignment")
        mapped_fold = mapped["outer_fold"].to_numpy(np.int16)
        score_fold = pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy(np.int16)
        if not np.array_equal(mapped_fold, score_fold):
            raise ValueError("candidate score outer_fold does not match row assignment")
        frame["regime"] = mapped["regime"].to_numpy(np.int16)
        frame["outer_fold"] = score_fold
        frame["actual_abs_error"] = pd.to_numeric(frame["actual_abs_error"], errors="coerce")
        frame["pred_abs_error"] = pd.to_numeric(frame["pred_abs_error"], errors="coerce")
        if not np.isfinite(
            frame[["actual_abs_error", "pred_abs_error"]].to_numpy(np.float64)
        ).all():
            raise ValueError("candidate score contains non-finite errors")
        for candidate_id, count in frame["candidate_id"].value_counts().items():
            rows_by_candidate[str(candidate_id)] += int(count)
        for (fold, regime, candidate_id), group in frame.groupby(
            ["outer_fold", "regime", "candidate_id"], sort=False
        ):
            update_stats(fold_stats[(int(fold), int(regime), str(candidate_id))], group)
            update_stats(pooled_stats[(int(regime), str(candidate_id))], group)
            calibration = group["pred_abs_error"].to_numpy(np.float64) - group[
                "actual_abs_error"
            ].to_numpy(np.float64)
            regime_calibration[int(regime)][0] += float(np.sum(calibration))
            regime_calibration[int(regime)][1] += float(len(calibration))

    if set(rows_by_candidate) != set(ids):
        raise ValueError("exp264 score does not contain all six primitive candidates")

    metric_records: list[dict[str, Any]] = []
    for (fold, regime, candidate_id), values in sorted(fold_stats.items()):
        count = int(values["rows"])
        metric_records.append(
            {
                "outer_fold": str(fold),
                "regime": regime,
                "candidate_id": candidate_id,
                "candidate_family": families[candidate_id],
                "rows": count,
                "wells": len(values["wells"]),
                "actual_mae": values["actual_abs_sum"] / count,
                "predicted_abs_error_mean": values["predicted_sum"] / count,
                "calibration_bias": values["calibration_sum"] / count,
                "actual_rmse": math.sqrt(values["actual_sq_sum"] / count),
            }
        )
    pooled_records: list[dict[str, Any]] = []
    for (regime, candidate_id), values in sorted(pooled_stats.items()):
        count = int(values["rows"])
        pooled_records.append(
            {
                "outer_fold": "all",
                "regime": regime,
                "candidate_id": candidate_id,
                "candidate_family": families[candidate_id],
                "rows": count,
                "wells": len(values["wells"]),
                "actual_mae": values["actual_abs_sum"] / count,
                "predicted_abs_error_mean": values["predicted_sum"] / count,
                "calibration_bias": values["calibration_sum"] / count,
                "actual_rmse": math.sqrt(values["actual_sq_sum"] / count),
            }
        )
    metrics = pd.DataFrame.from_records(metric_records + pooled_records)
    calibration = metrics[
        [
            "outer_fold",
            "regime",
            "candidate_id",
            "candidate_family",
            "rows",
            "predicted_abs_error_mean",
            "actual_mae",
            "calibration_bias",
        ]
    ].copy()

    n_clusters = int(config.get("regime", {}).get("n_clusters", 3))
    folds = sorted(int(item) for item in block_assignments["outer_fold"].unique())
    occupancy = (
        block_assignments.groupby(["outer_fold", "regime"])
        .agg(blocks=("block_key", "size"), wells=("well", "nunique"))
        .reindex(
            pd.MultiIndex.from_product([folds, range(n_clusters)], names=["outer_fold", "regime"])
        )
        .fillna(0)
        .reset_index()
    )
    fold_blocks = occupancy.groupby("outer_fold")["blocks"].transform("sum")
    occupancy["block_share"] = occupancy["blocks"] / np.maximum(fold_blocks, 1)
    occupancy_cfg = dict(config.get("guards", {}).get("occupancy", {}))
    occupancy["guard_pass"] = (
        occupancy["wells"] >= int(occupancy_cfg["min_wells_per_regime_fold"])
    ) & (occupancy["block_share"] >= float(occupancy_cfg["min_block_share_per_regime_fold"]))
    pass_counts = occupancy.groupby("regime")["guard_pass"].sum()
    occupancy_guard = bool(
        (pass_counts >= int(occupancy_cfg["min_passing_folds_per_regime"])).all()
    )
    stability_value = float(
        np.average(
            stability["centroid_matched_assignment_agreement"],
            weights=stability["outer_valid_blocks"],
        )
    )
    stability_guard = stability_value >= float(
        config.get("guards", {}).get("stability", {})["min_centroid_matched_assignment_agreement"]
    )
    pooled_frame = pd.DataFrame.from_records(pooled_records)
    winners = pooled_frame.loc[pooled_frame.groupby("regime")["actual_rmse"].idxmin()].sort_values(
        "regime"
    )
    distinct_best_families = int(winners["candidate_family"].nunique())
    bias_by_regime = {
        regime: values[0] / max(values[1], 1.0) for regime, values in regime_calibration.items()
    }
    calibration_bias_range = float(max(bias_by_regime.values()) - min(bias_by_regime.values()))
    separability_cfg = dict(config.get("guards", {}).get("separability", {}))
    family_guard = distinct_best_families >= int(
        separability_cfg["min_distinct_best_candidate_families"]
    )
    calibration_guard = calibration_bias_range >= float(
        separability_cfg["min_global_calibration_bias_range_ft"]
    )
    separability_guard = bool(family_guard or calibration_guard)
    summary = {
        "rows_scored": int(len(row_assignments)),
        "score_rows_candidate_long": int(sum(rows_by_candidate.values())),
        "score_rows_by_candidate": dict(sorted(rows_by_candidate.items())),
        "parquet_batches": batches,
        "regime_count": n_clusters,
        "occupancy_guard_pass": occupancy_guard,
        "stability": {
            "centroid_matched_assignment_agreement": stability_value,
            "guard_pass": bool(stability_guard),
        },
        "separability": {
            "best_candidate_by_regime": [
                {
                    "regime": int(row.regime),
                    "candidate_id": str(row.candidate_id),
                    "candidate_family": str(row.candidate_family),
                    "actual_rmse": float(row.actual_rmse),
                }
                for row in winners.itertuples(index=False)
            ],
            "distinct_best_candidate_families": distinct_best_families,
            "family_guard_pass": bool(family_guard),
            "global_calibration_bias_range_ft": calibration_bias_range,
            "calibration_guard_pass": bool(calibration_guard),
            "guard_pass": separability_guard,
        },
        "stage0_guard_pass": bool(occupancy_guard and stability_guard and separability_guard),
        "calibration": calibration,
    }
    return occupancy, metrics, summary


def save_stage0_artifacts(
    output_dir: Path,
    blocks: pd.DataFrame,
    block_assignments: pd.DataFrame,
    row_assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    stability: pd.DataFrame,
    occupancy: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    input_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "blocks": output_dir / "pairwise_block_fingerprint.parquet",
        "block_assignments": output_dir / "regime_block_assignments.parquet",
        "row_assignments": output_dir / "regime_row_assignments.parquet",
        "centroids": output_dir / "regime_centroids.csv",
        "stability": output_dir / "regime_assignment_stability.csv",
        "occupancy": output_dir / "regime_occupancy.csv",
        "candidate_metrics": output_dir / "regime_candidate_metrics.csv",
        "calibration": output_dir / "regime_candidate_calibration.csv",
    }
    blocks.to_parquet(paths["blocks"], index=False)
    block_assignments.to_parquet(paths["block_assignments"], index=False)
    row_assignments.to_parquet(paths["row_assignments"], index=False)
    centroids.to_csv(paths["centroids"], index=False)
    stability.to_csv(paths["stability"], index=False)
    occupancy.to_csv(paths["occupancy"], index=False)
    candidate_metrics.to_csv(paths["candidate_metrics"], index=False)
    calibration.to_csv(paths["calibration"], index=False)

    columns = feature_columns(blocks)
    schema = {
        "feature_count": len(columns),
        "feature_columns": columns,
        "pair_count": len(pair_ids(primitive_ids(contract))),
        "primitive_ids": primitive_ids(contract),
        "forbidden_feature_hits": [],
        "feature_schema_sha256": sha256_json(columns),
        "block_fingerprint_logical_sha256": logical_frame_sha256(blocks),
    }
    write_json(output_dir / "regime_feature_schema.json", schema)
    clean_summary = {key: value for key, value in summary.items() if key != "calibration"}
    clean_summary.update(
        {
            "blocks": int(len(blocks)),
            "rows": int(len(row_assignments)),
            "wells": int(row_assignments["well"].nunique()),
            "folds": int(row_assignments["outer_fold"].nunique()),
            "feature_count": len(columns),
            "primitive_count": len(primitive_ids(contract)),
            "pair_count": len(pair_ids(primitive_ids(contract))),
            "stage0_compute": {
                "variants": 0,
                "lightgbm_configs": 0,
                "folds_trained": 0,
                "boosters": 0,
                "parent_control_retraining": False,
            },
        }
    )
    write_json(output_dir / "stage0_summary.json", clean_summary)
    reproducibility = {
        "seed": int(config.get("reproducibility", {}).get("seed", 42)),
        "regime_contract_sha256": sha256_json(contract),
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "block_fingerprint_logical_sha256": schema["block_fingerprint_logical_sha256"],
        "input_evidence": to_jsonable(dict(input_evidence)),
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "deterministic_anchor": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    return clean_summary


def load_candidate_scores(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    scores = pd.read_parquet(path)
    if scores.empty:
        raise ValueError("exp264 candidate score artifact is empty")
    return scores


def read_contract(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"contract must be a mapping: {path}")
    validate_regime_contract(value)
    return value


def summary_for_display(summary: Mapping[str, Any]) -> str:
    return json.dumps(to_jsonable(dict(summary)), ensure_ascii=False, indent=2)
