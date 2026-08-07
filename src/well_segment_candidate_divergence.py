from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from src.candidate_pairwise_regime import (
    PrimitiveFold,
    pair_ids,
    primitive_family_map,
    primitive_ids,
)
from src.candidate_selector_pipeline import (
    logical_frame_sha256,
    sha256_file,
    sha256_json,
    to_jsonable,
    write_json,
)

SEGMENTS = ("early", "middle", "late")
SEMANTIC_CLUSTERS = ("low", "middle", "high")
SIGNATURE_METRICS = (
    "bank_range_mean",
    "bank_range_p90",
    "effective_rank",
    "rank_switch_rate",
    "pair_abs_gap_mean",
    "pair_abs_gap_p90",
)


def signature_feature_columns() -> list[str]:
    return [
        f"segment__{segment}__{metric}"
        for segment in SEGMENTS
        for metric in SIGNATURE_METRICS
    ]


def assert_target_free_signature_schema(
    columns: Sequence[str], forbidden_tokens: Sequence[str]
) -> None:
    expected = signature_feature_columns()
    if list(columns) != expected:
        raise ValueError(f"signature schema mismatch: expected={expected}, actual={list(columns)}")
    forbidden = {str(token).lower() for token in forbidden_tokens}
    hits = [
        column for column in columns if forbidden.intersection(column.lower().split("__"))
    ]
    if hits:
        raise ValueError(f"forbidden well signature features: {hits}")


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else math.nan


def _effective_rank(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        return math.nan
    filled = values.copy()
    for column in range(filled.shape[1]):
        finite = np.isfinite(filled[:, column])
        if not finite.any():
            return math.nan
        filled[~finite, column] = float(np.median(filled[finite, column]))
    centered = filled - np.mean(filled, axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular_values)
    total = float(np.sum(energy))
    if total <= 1e-12:
        return 1.0
    shares = energy / total
    nonzero = shares[shares > 1e-12]
    return float(np.exp(-np.sum(nonzero * np.log(nonzero))))


def _rank_switch_rate(matrix: np.ndarray) -> tuple[float, int]:
    values = np.asarray(matrix, dtype=np.float64)
    if len(values) < 2:
        return 0.0, 0
    complete = np.isfinite(values).all(axis=1)
    eligible = complete[1:] & complete[:-1]
    if not eligible.any():
        return math.nan, 0
    ranks = np.argsort(np.argsort(values, axis=1, kind="stable"), axis=1, kind="stable")
    changed = np.any(ranks[1:] != ranks[:-1], axis=1)
    return float(np.mean(changed[eligible])), int(eligible.sum())


def _segment_metrics(matrix: np.ndarray) -> tuple[dict[str, float], dict[str, int | bool]]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("segment candidate matrix must have six columns")
    finite = np.isfinite(values)
    row_ranges = np.full(len(values), np.nan, dtype=np.float64)
    usable_range_rows = finite.sum(axis=1) >= 2
    if usable_range_rows.any():
        selected = values[usable_range_rows]
        row_ranges[usable_range_rows] = np.nanmax(selected, axis=1) - np.nanmin(
            selected, axis=1
        )
    pair_values = [
        np.abs(values[:, left] - values[:, right])
        for left, right in itertools.combinations(range(6), 2)
    ]
    flattened_pairs = (
        np.concatenate(pair_values) if pair_values else np.asarray([], dtype=np.float64)
    )
    rank_switch_rate, eligible_transitions = _rank_switch_rate(values)
    metrics = {
        "bank_range_mean": _safe_mean(row_ranges),
        "bank_range_p90": _safe_quantile(row_ranges, 0.90),
        "effective_rank": _effective_rank(values),
        "rank_switch_rate": rank_switch_rate,
        "pair_abs_gap_mean": _safe_mean(flattened_pairs),
        "pair_abs_gap_p90": _safe_quantile(flattened_pairs, 0.90),
    }
    coverage: dict[str, int | bool] = {
        "segment_rows": int(len(values)),
        "complete_candidate_rows": int(finite.all(axis=1).sum()),
        "range_eligible_rows": int(usable_range_rows.sum()),
        "rank_eligible_transitions": eligible_transitions,
        "finite_pair_values": int(np.isfinite(flattened_pairs).sum()),
        "fallback_required": bool(
            len(values) == 0
            or not finite.all()
            or any(not np.isfinite(value) for value in metrics.values())
        ),
    }
    return metrics, coverage


def build_well_segment_signatures(
    bundle: PrimitiveFold, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse one outer-fold candidate cache to one target-free 18-feature row per well."""

    if len(bundle.primitive_ids) != 6 or bundle.values.shape[1] != 6:
        raise ValueError("well signature requires exactly six primitive candidates")
    feature_cfg = dict(config.get("features", {}))
    boundaries = [float(item) for item in feature_cfg.get("segment_boundaries", [])]
    if boundaries != [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]:
        raise ValueError(f"segment boundaries must be fixed thirds: {boundaries}")
    base = bundle.base.reset_index(drop=True)
    if len(base) != len(bundle.values) or bundle.available.shape != bundle.values.shape:
        raise ValueError("primitive bundle shape mismatch")
    feature_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    for well, raw_positions in base.groupby("well", sort=False).indices.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        well_base = base.iloc[positions].sort_values("well_row_idx", kind="stable")
        positions = well_base.index.to_numpy(np.int64)
        folds = pd.to_numeric(well_base["outer_fold"], errors="raise").unique()
        if len(folds) != 1:
            raise ValueError(f"well spans outer folds: {well}")
        values = bundle.values[positions].astype(np.float64, copy=True)
        values[~bundle.available[positions]] = np.nan
        eval_rows = len(values)
        progress = np.arange(eval_rows, dtype=np.float64) / max(eval_rows, 1)
        segment_index = np.minimum(np.floor(progress * 3.0).astype(np.int8), 2)
        record: dict[str, Any] = {
            "well": str(well),
            "outer_fold": int(folds[0]),
            "eval_rows": int(eval_rows),
        }
        for index, segment in enumerate(SEGMENTS):
            selected = segment_index == index
            metrics, coverage = _segment_metrics(values[selected])
            for metric, value in metrics.items():
                record[f"segment__{segment}__{metric}"] = value
            coverage_records.append(
                {
                    "well": str(well),
                    "outer_fold": int(folds[0]),
                    "segment": segment,
                    "progress_start": boundaries[index],
                    "progress_end": boundaries[index + 1],
                    **coverage,
                }
            )
        feature_records.append(record)
    signatures = pd.DataFrame.from_records(feature_records).sort_values(
        ["outer_fold", "well"], kind="stable"
    ).reset_index(drop=True)
    coverage = pd.DataFrame.from_records(coverage_records).sort_values(
        ["outer_fold", "well", "segment"], kind="stable"
    ).reset_index(drop=True)
    columns = signature_feature_columns()
    assert_target_free_signature_schema(
        columns, [str(item) for item in feature_cfg.get("forbidden_feature_tokens", [])]
    )
    if signatures["well"].duplicated().any():
        raise ValueError("well signature contains duplicate wells")
    if len(coverage) != len(signatures) * len(SEGMENTS):
        raise ValueError("segment coverage is incomplete")
    return signatures, coverage


def _soft_membership(distances: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    logits = -np.asarray(distances, dtype=np.float64) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)


def _cluster_profiles(
    signatures: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    merged = signatures.merge(
        assignments[["well", "outer_fold", "semantic_cluster"]],
        on=["well", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    records: list[dict[str, Any]] = []
    for (fold, cluster), group in merged.groupby(
        ["outer_fold", "semantic_cluster"], sort=False
    ):
        for segment in SEGMENTS:
            record: dict[str, Any] = {
                "outer_fold": int(fold),
                "semantic_cluster": str(cluster),
                "segment": segment,
                "wells": int(len(group)),
            }
            for metric in SIGNATURE_METRICS:
                record[metric] = float(
                    group[f"segment__{segment}__{metric}"].median(skipna=True)
                )
            records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["outer_fold", "semantic_cluster", "segment"], kind="stable"
    ).reset_index(drop=True)


def fit_outer_fold_clusters(
    signatures: pd.DataFrame,
    coverage: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    columns = signature_feature_columns()
    assert_target_free_signature_schema(
        columns,
        [str(item) for item in config.get("features", {}).get("forbidden_feature_tokens", [])],
    )
    cluster_cfg = dict(config.get("cluster", {}))
    n_clusters = int(cluster_cfg.get("n_clusters", 3))
    if n_clusters != 3:
        raise ValueError("well divergence audit fixes KMeans K=3")
    quantile_range = tuple(float(item) for item in cluster_cfg.get("quantile_range", []))
    clip_low, clip_high = [float(item) for item in cluster_cfg.get("scaled_clip", [])]
    n_init = int(cluster_cfg.get("kmeans_n_init", 20))
    seed = int(config.get("reproducibility", {}).get("seed", 42))
    folds = sorted(int(item) for item in signatures["outer_fold"].unique())
    assignments: list[pd.DataFrame] = []
    centroid_records: list[dict[str, Any]] = []
    preprocessor_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = signatures["outer_fold"].to_numpy() != fold
        valid_mask = ~train_mask
        train_raw = signatures.loc[train_mask, columns].apply(pd.to_numeric, errors="coerce")
        valid_raw = signatures.loc[valid_mask, columns].apply(pd.to_numeric, errors="coerce")
        medians = train_raw.median(axis=0, skipna=True).fillna(0.0)
        train_imputed = train_raw.fillna(medians).to_numpy(np.float64)
        valid_imputed = valid_raw.fillna(medians).to_numpy(np.float64)
        scaler = RobustScaler(quantile_range=quantile_range)
        train_scaled = np.clip(scaler.fit_transform(train_imputed), clip_low, clip_high)
        valid_scaled = np.clip(scaler.transform(valid_imputed), clip_low, clip_high)
        primary = KMeans(
            n_clusters=3,
            n_init=n_init,
            random_state=seed + fold,
            algorithm="lloyd",
        ).fit(train_scaled)
        audit = KMeans(
            n_clusters=3,
            n_init=n_init,
            random_state=seed + 10000 + fold,
            algorithm="lloyd",
        ).fit(train_scaled)
        raw_scores = np.mean(primary.cluster_centers_, axis=1)
        semantic_order = np.argsort(raw_scores, kind="stable")
        raw_to_semantic = {
            int(raw): int(semantic) for semantic, raw in enumerate(semantic_order)
        }
        ordered_centers = primary.cluster_centers_[semantic_order]
        distances = cdist(valid_scaled, ordered_centers)
        semantic_labels = np.argmin(distances, axis=1).astype(np.int8)
        nearest_train = np.min(primary.transform(train_scaled), axis=1)
        positive = nearest_train[np.isfinite(nearest_train) & (nearest_train > 0)]
        temperature = float(np.median(positive)) if len(positive) else 1.0
        probabilities = _soft_membership(distances, temperature)
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
        entropy /= math.log(3.0)

        primary_rows, audit_columns = linear_sum_assignment(
            cdist(primary.cluster_centers_, audit.cluster_centers_)
        )
        audit_to_primary = {
            int(audit_label): int(primary_label)
            for primary_label, audit_label in zip(primary_rows, audit_columns, strict=True)
        }
        audit_raw = audit.predict(valid_scaled)
        audit_semantic = np.asarray(
            [raw_to_semantic[audit_to_primary[int(label)]] for label in audit_raw],
            dtype=np.int8,
        )
        agreement = audit_semantic == semantic_labels

        output = signatures.loc[valid_mask, ["well", "outer_fold", "eval_rows"]].copy()
        output = output.reset_index(drop=True)
        output["cluster_index"] = semantic_labels
        output["semantic_cluster"] = [SEMANTIC_CLUSTERS[index] for index in semantic_labels]
        output["cluster_distance"] = np.min(distances, axis=1).astype(np.float32)
        output["cluster_entropy"] = entropy.astype(np.float32)
        output["assignment_stable"] = agreement
        output["audit_cluster_index"] = audit_semantic
        for index, name in enumerate(SEMANTIC_CLUSTERS):
            output[f"cluster_probability_{name}"] = probabilities[:, index].astype(np.float32)
        assignments.append(output)

        original_centers = scaler.inverse_transform(ordered_centers)
        for semantic, name in enumerate(SEMANTIC_CLUSTERS):
            record: dict[str, Any] = {
                "outer_fold": fold,
                "cluster_index": semantic,
                "semantic_cluster": name,
                "scaled_divergence_index": float(np.mean(ordered_centers[semantic])),
                "temperature": temperature,
                "outer_train_wells": int(train_mask.sum()),
                "outer_valid_wells": int(valid_mask.sum()),
            }
            record.update(
                {
                    column: float(original_centers[semantic, index])
                    for index, column in enumerate(columns)
                }
            )
            centroid_records.append(record)
        preprocessor_records.append(
            {
                "outer_fold": fold,
                "feature_columns": columns,
                "outer_train_medians": medians.to_dict(),
                "robust_scaler_center": scaler.center_.tolist(),
                "robust_scaler_scale": scaler.scale_.tolist(),
                "scaled_clip": [clip_low, clip_high],
                "primary_centers_semantic_order": ordered_centers.tolist(),
                "temperature": temperature,
                "primary_seed": seed + fold,
                "audit_seed": seed + 10000 + fold,
            }
        )
        stability_records.append(
            {
                "outer_fold": fold,
                "outer_valid_wells": int(valid_mask.sum()),
                "centroid_matched_assignment_agreement": float(np.mean(agreement)),
            }
        )

    assignment_frame = pd.concat(assignments, ignore_index=True).sort_values(
        ["outer_fold", "well"], kind="stable"
    ).reset_index(drop=True)
    centroids = pd.DataFrame.from_records(centroid_records)
    stability = pd.DataFrame.from_records(stability_records)
    profiles = _cluster_profiles(signatures, assignment_frame)
    fold_occupancy = (
        assignment_frame.groupby(["outer_fold", "semantic_cluster"])
        .agg(wells=("well", "size"))
        .reindex(
            pd.MultiIndex.from_product(
                [folds, SEMANTIC_CLUSTERS], names=["outer_fold", "semantic_cluster"]
            )
        )
        .fillna(0)
        .reset_index()
    )
    fold_occupancy["well_share"] = fold_occupancy["wells"] / np.maximum(
        fold_occupancy.groupby("outer_fold")["wells"].transform("sum"), 1
    )
    occupancy_cfg = dict(config.get("guards", {}).get("occupancy", {}))
    fold_occupancy["guard_pass"] = fold_occupancy["wells"] >= int(
        occupancy_cfg["min_wells_per_cluster_outer_valid"]
    )
    pooled = (
        assignment_frame.groupby("semantic_cluster")
        .agg(wells=("well", "size"))
        .reindex(SEMANTIC_CLUSTERS)
        .fillna(0)
        .reset_index()
    )
    pooled["outer_fold"] = "all"
    pooled["well_share"] = pooled["wells"] / max(int(pooled["wells"].sum()), 1)
    pooled["guard_pass"] = pooled["wells"] >= int(
        occupancy_cfg["min_wells_per_cluster_pooled"]
    )
    fold_occupancy["outer_fold"] = fold_occupancy["outer_fold"].astype(str)
    occupancy = pd.concat(
        [fold_occupancy, pooled[fold_occupancy.columns]], ignore_index=True
    )

    profile_fold_pass: dict[int, bool] = {}
    for fold in folds:
        selected = profiles[profiles["outer_fold"] == fold]
        fold_pass = True
        for segment in SEGMENTS:
            values = selected[selected["segment"] == segment].set_index("semantic_cluster")[
                "bank_range_mean"
            ]
            fold_pass &= bool(
                set(values.index) == set(SEMANTIC_CLUSTERS)
                and values["low"] < values["middle"] < values["high"]
            )
        profile_fold_pass[fold] = bool(fold_pass)
    centroid_score = centroids.pivot(
        index="outer_fold", columns="semantic_cluster", values="scaled_divergence_index"
    )
    semantic_consistency = bool(
        centroid_score["low"].max() < centroid_score["middle"].min()
        and centroid_score["middle"].max() < centroid_score["high"].min()
    )
    technical_cfg = dict(config.get("guards", {}).get("technical", {}))
    technical_guard = bool(
        len(signatures) == int(technical_cfg["expected_wells"])
        and int(signatures["eval_rows"].sum())
        == int(technical_cfg.get("expected_rows", signatures["eval_rows"].sum()))
        and len(folds) == int(technical_cfg["expected_folds"])
        and len(columns) == int(technical_cfg["expected_features"])
    )
    coverage_summary = {
        "rows": int(len(coverage)),
        "wells_with_all_three_segments": int(
            coverage.groupby("well")["segment"].nunique().eq(3).sum()
        ),
        "fallback_wells": int(coverage.loc[coverage["fallback_required"], "well"].nunique()),
        "fallback_segments": int(coverage["fallback_required"].sum()),
    }
    occupancy_guard = bool(occupancy["guard_pass"].all())
    stability_threshold = float(
        config.get("guards", {}).get("stability", {})[
            "min_centroid_matched_assignment_agreement_each_fold"
        ]
    )
    stability_guard = bool(
        (stability["centroid_matched_assignment_agreement"] >= stability_threshold).all()
    )
    profile_guard = bool(all(profile_fold_pass.values()))
    structure_guard = bool(
        technical_guard
        and occupancy_guard
        and stability_guard
        and profile_guard
        and semantic_consistency
    )
    summary = {
        "technical_guard_pass": technical_guard,
        "technical": {
            "rows": int(signatures["eval_rows"].sum()),
            "wells": int(len(signatures)),
            "folds": int(len(folds)),
            "features": int(len(columns)),
            "forbidden_feature_hits": 0,
        },
        "coverage": coverage_summary,
        "occupancy_guard_pass": occupancy_guard,
        "stability_guard_pass": stability_guard,
        "divergence_profile_guard_pass": profile_guard,
        "divergence_profile_fold_pass": profile_fold_pass,
        "semantic_label_consistency_guard_pass": semantic_consistency,
        "structure_guard_pass": structure_guard,
    }
    return (
        assignment_frame,
        centroids,
        preprocessor_records,
        stability,
        occupancy,
        profiles,
        summary,
    )


def evaluate_post_assignment_scores_from_parquet(
    assignments: pd.DataFrame,
    candidate_score_path: Path,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    structure_summary: Mapping[str, Any],
    *,
    batch_size: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Open exp264 labels only after OOF well assignments are fixed and stream aggregates."""

    import pyarrow.parquet as pq

    path = Path(candidate_score_path)
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    required = {
        "well",
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
    lookup = assignments.set_index("well")[["outer_fold", "semantic_cluster"]]
    if not lookup.index.is_unique:
        raise ValueError("well cluster assignment must be unique")

    def new_stats() -> dict[str, Any]:
        return {"rows": 0, "actual_sum": 0.0, "predicted_sum": 0.0, "wells": set()}

    fold_candidate: dict[tuple[int, str, str], dict[str, Any]] = defaultdict(new_stats)
    pooled_candidate: dict[tuple[str, str], dict[str, Any]] = defaultdict(new_stats)
    fold_cluster: dict[tuple[int, str], dict[str, Any]] = defaultdict(new_stats)
    pooled_cluster: dict[str, dict[str, Any]] = defaultdict(new_stats)
    well_cluster: dict[tuple[str, str], dict[str, Any]] = defaultdict(new_stats)
    rows_by_candidate: dict[str, int] = defaultdict(int)
    batches = 0

    def update(target: dict[str, Any], group: pd.DataFrame) -> None:
        actual = group["actual_abs_error"].to_numpy(np.float64)
        predicted = group["pred_abs_error"].to_numpy(np.float64)
        target["rows"] += int(len(group))
        target["actual_sum"] += float(np.sum(actual))
        target["predicted_sum"] += float(np.sum(predicted))
        target["wells"].update(group["well"].astype(str).unique().tolist())

    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        batches += 1
        frame = batch.to_pandas()
        frame["well"] = frame["well"].astype(str)
        frame = frame[frame["candidate_id"].isin(ids)].copy()
        if "candidate_available" in frame:
            frame = frame[frame["candidate_available"].astype(bool)].copy()
        if frame.empty:
            continue
        mapped = lookup.reindex(frame["well"].to_numpy())
        if mapped["semantic_cluster"].isna().any():
            raise ValueError("candidate score has wells without frozen cluster assignment")
        score_fold = pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy(np.int16)
        mapped_fold = mapped["outer_fold"].to_numpy(np.int16)
        if not np.array_equal(score_fold, mapped_fold):
            raise ValueError("candidate score outer_fold does not match cluster assignment")
        frame["outer_fold"] = score_fold
        frame["semantic_cluster"] = mapped["semantic_cluster"].to_numpy(str)
        frame["actual_abs_error"] = pd.to_numeric(
            frame["actual_abs_error"], errors="coerce"
        )
        frame["pred_abs_error"] = pd.to_numeric(frame["pred_abs_error"], errors="coerce")
        if not np.isfinite(
            frame[["actual_abs_error", "pred_abs_error"]].to_numpy(np.float64)
        ).all():
            raise ValueError("candidate score contains non-finite values")
        for candidate_id, count in frame["candidate_id"].value_counts().items():
            rows_by_candidate[str(candidate_id)] += int(count)
        for key, group in frame.groupby(
            ["outer_fold", "semantic_cluster", "candidate_id"], sort=False
        ):
            fold, cluster, candidate_id = key
            update(fold_candidate[(int(fold), str(cluster), str(candidate_id))], group)
            update(pooled_candidate[(str(cluster), str(candidate_id))], group)
        for (fold, cluster), group in frame.groupby(
            ["outer_fold", "semantic_cluster"], sort=False
        ):
            update(fold_cluster[(int(fold), str(cluster))], group)
            update(pooled_cluster[str(cluster)], group)
        for (cluster, well), group in frame.groupby(
            ["semantic_cluster", "well"], sort=False
        ):
            update(well_cluster[(str(cluster), str(well))], group)
    if set(rows_by_candidate) != set(ids):
        raise ValueError("exp264 score does not contain all six primitive candidates")
    technical_cfg = dict(config.get("guards", {}).get("technical", {}))
    expected_score_rows = technical_cfg.get("expected_rows")
    if expected_score_rows is not None:
        incomplete = {
            candidate_id: count
            for candidate_id, count in rows_by_candidate.items()
            if count != int(expected_score_rows)
        }
        if incomplete:
            raise ValueError(
                "exp264 primitive score row coverage mismatch: "
                f"expected={expected_score_rows}, actual={incomplete}"
            )

    def metric_record(
        outer_fold: str, cluster: str, candidate_id: str, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        count = int(values["rows"])
        return {
            "outer_fold": outer_fold,
            "semantic_cluster": cluster,
            "candidate_id": candidate_id,
            "candidate_family": families[candidate_id],
            "rows": count,
            "wells": len(values["wells"]),
            "actual_mae": float(values["actual_sum"]) / count,
            "predicted_abs_error_mean": float(values["predicted_sum"]) / count,
            "calibration_bias": (
                float(values["predicted_sum"]) - float(values["actual_sum"])
            )
            / count,
        }

    metric_records = [
        metric_record(str(fold), cluster, candidate_id, values)
        for (fold, cluster, candidate_id), values in sorted(fold_candidate.items())
    ]
    metric_records.extend(
        metric_record("all", cluster, candidate_id, values)
        for (cluster, candidate_id), values in sorted(pooled_candidate.items())
    )
    metrics = pd.DataFrame.from_records(metric_records)
    calibration = metrics[
        [
            "outer_fold",
            "semantic_cluster",
            "candidate_id",
            "candidate_family",
            "rows",
            "wells",
            "predicted_abs_error_mean",
            "actual_mae",
            "calibration_bias",
        ]
    ].copy()
    well_records = []
    for (cluster, well), values in sorted(well_cluster.items()):
        count = int(values["rows"])
        well_records.append(
            {
                "semantic_cluster": cluster,
                "well": well,
                "rows": count,
                "actual_mae": float(values["actual_sum"]) / count,
                "predicted_abs_error_mean": float(values["predicted_sum"]) / count,
                "calibration_bias": (
                    float(values["predicted_sum"]) - float(values["actual_sum"])
                )
                / count,
            }
        )
    well_metrics = pd.DataFrame.from_records(well_records)
    expected_wells = technical_cfg.get("expected_wells")
    if expected_wells is not None and len(well_metrics) != int(expected_wells):
        raise ValueError(
            "exp264 primitive score well coverage mismatch: "
            f"expected={expected_wells}, actual={len(well_metrics)}"
        )

    fold_winners: dict[int, tuple[str, str, str]] = {}
    for fold in sorted(int(item) for item in assignments["outer_fold"].unique()):
        fold_metrics = metrics[metrics["outer_fold"] == str(fold)]
        winners = []
        for cluster in SEMANTIC_CLUSTERS:
            selected = fold_metrics[fold_metrics["semantic_cluster"] == cluster]
            if selected.empty:
                raise ValueError(f"missing score metrics for fold={fold}, cluster={cluster}")
            winners.append(str(selected.loc[selected["actual_mae"].idxmin(), "candidate_id"]))
        fold_winners[fold] = tuple(winners)  # type: ignore[assignment]
    score_cfg = dict(config.get("guards", {}).get("score", {}))
    min_consistent_folds = int(score_cfg.get("min_consistent_folds", 4))
    min_distinct_winners = int(
        score_cfg.get("min_distinct_candidates_in_modal_winner_pattern", 2)
    )
    pattern_counts = Counter(fold_winners.values())
    modal_pattern, modal_count = pattern_counts.most_common(1)[0]
    winner_guard = bool(
        modal_count >= min_consistent_folds
        and len(set(modal_pattern)) >= min_distinct_winners
    )

    calibration_differences: dict[int, float] = {}
    for fold in sorted(int(item) for item in assignments["outer_fold"].unique()):
        low = fold_cluster[(fold, "low")]
        high = fold_cluster[(fold, "high")]
        low_bias = (float(low["predicted_sum"]) - float(low["actual_sum"])) / int(
            low["rows"]
        )
        high_bias = (float(high["predicted_sum"]) - float(high["actual_sum"])) / int(
            high["rows"]
        )
        calibration_differences[fold] = high_bias - low_bias
    epsilon = float(score_cfg.get("calibration_epsilon", 1e-9))
    signs = [
        int(np.sign(value))
        for value in calibration_differences.values()
        if abs(value) > epsilon
    ]
    sign_counts = Counter(signs)
    modal_sign_count = sign_counts.most_common(1)[0][1] if sign_counts else 0
    calibration_guard = bool(modal_sign_count >= min_consistent_folds)

    cluster_actual = {
        cluster: float(values["actual_sum"]) / int(values["rows"])
        for cluster, values in pooled_cluster.items()
    }
    worst_cluster = max(cluster_actual, key=cluster_actual.get)
    worst_wells = well_metrics[well_metrics["semantic_cluster"] == worst_cluster]
    worst_well = str(worst_wells.loc[worst_wells["actual_mae"].idxmax(), "well"])
    removed = well_cluster[(worst_cluster, worst_well)]
    source = pooled_cluster[worst_cluster]
    remaining_rows = int(source["rows"]) - int(removed["rows"])
    if remaining_rows <= 0:
        worst_cluster_after_removal = "none"
    else:
        after_removal = dict(cluster_actual)
        after_removal[worst_cluster] = (
            float(source["actual_sum"]) - float(removed["actual_sum"])
        ) / remaining_rows
        worst_cluster_after_removal = max(after_removal, key=after_removal.get)
    single_well_guard = worst_cluster_after_removal == worst_cluster
    score_guard = bool((winner_guard or calibration_guard) and single_well_guard)
    stage_a_guard = bool(structure_summary.get("structure_guard_pass", False) and score_guard)
    summary = {
        "parquet_batches": batches,
        "score_rows_candidate_long": int(sum(rows_by_candidate.values())),
        "score_rows_by_candidate": dict(sorted(rows_by_candidate.items())),
        "candidate_winner": {
            "fold_patterns": {str(key): list(value) for key, value in fold_winners.items()},
            "modal_pattern": list(modal_pattern),
            "modal_fold_count": modal_count,
            "guard_pass": winner_guard,
        },
        "calibration_direction": {
            "high_minus_low_by_fold": calibration_differences,
            "modal_nonzero_sign_fold_count": modal_sign_count,
            "guard_pass": calibration_guard,
        },
        "worst_cluster_single_well": {
            "cluster_actual_mae": cluster_actual,
            "worst_cluster": worst_cluster,
            "highest_error_well": worst_well,
            "worst_cluster_after_removing_highest_error_well": worst_cluster_after_removal,
            "guard_pass": single_well_guard,
        },
        "score_separability_guard_pass": score_guard,
        "stage_a_guard_pass": stage_a_guard,
    }
    return metrics, calibration, well_metrics, summary


def save_stage_a_artifacts(
    output_dir: Path,
    signatures: pd.DataFrame,
    coverage: pd.DataFrame,
    assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    preprocessors: Sequence[Mapping[str, Any]],
    stability: pd.DataFrame,
    occupancy: pd.DataFrame,
    profiles: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    well_score_metrics: pd.DataFrame,
    structure_summary: Mapping[str, Any],
    score_summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    input_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "signatures": output_dir / "well_segment_signatures.parquet",
        "coverage": output_dir / "well_segment_coverage.csv",
        "assignments": output_dir / "well_cluster_assignments.parquet",
        "centroids": output_dir / "well_cluster_centroids.csv",
        "stability": output_dir / "well_cluster_assignment_stability.csv",
        "occupancy": output_dir / "well_cluster_occupancy.csv",
        "profiles": output_dir / "well_cluster_divergence_profiles.csv",
        "candidate_metrics": output_dir / "well_cluster_candidate_metrics.csv",
        "calibration": output_dir / "well_cluster_candidate_calibration.csv",
        "well_score_metrics": output_dir / "well_cluster_score_by_well.csv",
    }
    signatures.to_parquet(paths["signatures"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    assignments.to_parquet(paths["assignments"], index=False)
    centroids.to_csv(paths["centroids"], index=False)
    stability.to_csv(paths["stability"], index=False)
    occupancy.to_csv(paths["occupancy"], index=False)
    profiles.to_csv(paths["profiles"], index=False)
    candidate_metrics.to_csv(paths["candidate_metrics"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    well_score_metrics.to_csv(paths["well_score_metrics"], index=False)
    preprocessor_payload = {"folds": list(preprocessors)}
    write_json(output_dir / "well_cluster_preprocessors.json", preprocessor_payload)

    columns = signature_feature_columns()
    schema = {
        "feature_count": len(columns),
        "feature_columns": columns,
        "segments": list(SEGMENTS),
        "metrics_per_segment": list(SIGNATURE_METRICS),
        "primitive_ids": primitive_ids(contract),
        "pair_count": len(pair_ids(primitive_ids(contract))),
        "forbidden_feature_hits": [],
        "feature_schema_sha256": sha256_json(columns),
        "well_signature_logical_sha256": logical_frame_sha256(signatures),
    }
    write_json(output_dir / "well_signature_feature_schema.json", schema)
    summary = {
        **to_jsonable(dict(structure_summary)),
        **to_jsonable(dict(score_summary)),
        "wells": int(len(signatures)),
        "folds": int(signatures["outer_fold"].nunique()),
        "feature_count": len(columns),
        "stage_a_compute": {
            "variants": 0,
            "lightgbm_configs": 0,
            "folds_trained": 0,
            "boosters": 0,
            "parent_control_retraining": False,
        },
    }
    write_json(output_dir / "stage_a_summary.json", summary)
    reproducibility = {
        "seed": int(config.get("reproducibility", {}).get("seed", 42)),
        "candidate_contract_sha256": sha256_json(contract),
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "well_signature_logical_sha256": schema["well_signature_logical_sha256"],
        "assignment_logical_sha256": logical_frame_sha256(assignments),
        "preprocessor_sha256": sha256_json(preprocessor_payload),
        "input_evidence": to_jsonable(dict(input_evidence)),
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "deterministic_anchor": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    return summary


def summary_for_display(summary: Mapping[str, Any]) -> str:
    return json.dumps(to_jsonable(dict(summary)), ensure_ascii=False, indent=2)
