from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from baseline import build_drift_feature_frame, config_get, predict_drift, well_id_from_path
from pseudo_tail_augmentation import (
    collect_training_rows,
    feature_importance_rows,
    fit_model,
    get_nested,
    load_yaml,
    sample_weights,
    source_summary_rows,
    train_files,
)
from settings import ExperimentPaths
from sklearn.model_selection import GroupKFold

STAT_FIELDS = ("n", "target2", "pred2", "pred_target")


@dataclass(frozen=True)
class Bucket:
    name: str
    max_step: float


@dataclass
class CandidateAggregate:
    name: str
    method: str
    params: dict[str, Any]
    selectable: bool
    n: int = 0
    sse: float = 0.0
    fold_sse: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    fold_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    well_fold_sse: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    well_fold_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    bucket_sse: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    bucket_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))

    @property
    def rmse(self) -> float:
        return rmse_from_sse(self.sse, self.n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit postprocess on pseudo-tail OOF.")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke limit")
    return parser.parse_args()


def rmse_from_sse(sse: float, n_rows: int | float) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(max(0.0, float(sse)) / float(n_rows))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % n_folds


def empty_stats() -> dict[str, float]:
    return {"n": 0.0, "target2": 0.0, "pred2": 0.0, "pred_target": 0.0}


def subtract_stats(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {
        field: float(left.get(field, 0.0)) - float(right.get(field, 0.0)) for field in STAT_FIELDS
    }


def stats_sse(stats: dict[str, float], alpha: float) -> float:
    return (
        float(stats["target2"])
        - 2.0 * alpha * float(stats["pred_target"])
        + alpha * alpha * float(stats["pred2"])
    )


def stats_rmse(stats: dict[str, float], alpha: float) -> float:
    return rmse_from_sse(stats_sse(stats, alpha), stats["n"])


def fit_alpha(stats: dict[str, float], *, alpha_min: float, alpha_max: float) -> float:
    denom = float(stats["pred2"])
    if denom <= 0:
        return 1.0
    alpha = float(stats["pred_target"]) / denom
    return float(np.clip(alpha, alpha_min, alpha_max))


def load_buckets(config: dict[str, Any]) -> list[Bucket]:
    raw_buckets = get_nested(config, "audit.distance_buckets", [])
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError("audit.distance_buckets must be a non-empty list")
    return [
        Bucket(name=str(item["name"]), max_step=float(item["max_step"])) for item in raw_buckets
    ]


def bucket_codes(eval_step: np.ndarray, buckets: list[Bucket]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        mask = (eval_step > previous_max) & (eval_step <= bucket.max_step)
        codes[mask] = idx
        previous_max = bucket.max_step
    return codes


def bucket_alpha_array(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    alpha = np.ones(eval_step.shape, dtype=float)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        alpha[mask] = float(bucket.get("alpha", 1.0))
        previous_max = max_step
    return alpha


def candidate_prediction(
    candidate: dict[str, Any],
    *,
    raw: np.ndarray,
    anchor: np.ndarray,
    eval_step: np.ndarray,
) -> np.ndarray:
    method = str(candidate["method"])
    if method == "raw":
        return raw.copy()
    if method == "last_anchor":
        return anchor.copy()
    if method == "distance_bucket_shrink":
        alpha = bucket_alpha_array(eval_step, list(candidate.get("buckets", [])))
        return anchor + alpha * (raw - anchor)
    raise ValueError(f"unsupported candidate method: {method}")


def make_candidate_aggregates(
    config: dict[str, Any],
    *,
    n_folds: int,
    well_holdout_folds: int,
    n_buckets: int,
) -> dict[str, CandidateAggregate]:
    candidates = list(get_nested(config, "audit.fixed_candidates", []))
    if not candidates:
        raise ValueError("audit.fixed_candidates must be a non-empty list")
    aggregates: dict[str, CandidateAggregate] = {}
    for candidate in candidates:
        name = str(candidate["name"])
        aggregates[name] = CandidateAggregate(
            name=name,
            method=str(candidate["method"]),
            params={
                key: value for key, value in candidate.items() if key not in {"name", "method"}
            },
            selectable=bool(candidate.get("selectable", True)),
            fold_sse=np.zeros(n_folds, dtype=float),
            fold_n=np.zeros(n_folds, dtype=float),
            well_fold_sse=np.zeros(well_holdout_folds, dtype=float),
            well_fold_n=np.zeros(well_holdout_folds, dtype=float),
            bucket_sse=np.zeros(n_buckets, dtype=float),
            bucket_n=np.zeros(n_buckets, dtype=float),
        )
    return aggregates


def add_candidate_stats(
    aggregate: CandidateAggregate,
    *,
    pred: np.ndarray,
    y_true: np.ndarray,
    fold: int,
    well_fold: int,
    bucket_code_values: np.ndarray,
    n_buckets: int,
) -> None:
    diff2 = np.square(pred - y_true)
    aggregate.n += int(diff2.size)
    aggregate.sse += float(diff2.sum())
    aggregate.fold_sse[fold] += float(diff2.sum())
    aggregate.fold_n[fold] += float(diff2.size)
    aggregate.well_fold_sse[well_fold] += float(diff2.sum())
    aggregate.well_fold_n[well_fold] += float(diff2.size)
    aggregate.bucket_sse += np.bincount(bucket_code_values, weights=diff2, minlength=n_buckets)
    aggregate.bucket_n += np.bincount(bucket_code_values, minlength=n_buckets).astype(float)


def add_residual_stats(
    stats: dict[str, float],
    *,
    raw: np.ndarray,
    anchor: np.ndarray,
    y_true: np.ndarray,
) -> None:
    pred_resid = raw - anchor
    true_resid = y_true - anchor
    stats["n"] += float(y_true.size)
    stats["target2"] += float(np.square(true_resid).sum())
    stats["pred2"] += float(np.square(pred_resid).sum())
    stats["pred_target"] += float((pred_resid * true_resid).sum())


def aggregate_predictions(
    *,
    config: dict[str, Any],
    candidate_aggs: dict[str, CandidateAggregate],
    total_by_bucket: dict[int, dict[str, float]],
    by_fold_bucket: dict[tuple[int, int], dict[str, float]],
    by_well_fold_bucket: dict[tuple[int, int], dict[str, float]],
    buckets: list[Bucket],
    fold: int,
    well_id: str,
    raw: np.ndarray,
    anchor: np.ndarray,
    y_true: np.ndarray,
    eval_step: np.ndarray,
    well_holdout_folds: int,
) -> None:
    bucket_code_values = bucket_codes(eval_step, buckets)
    well_fold = stable_fold(well_id, well_holdout_folds)
    n_buckets = len(buckets)
    fixed_candidates = list(get_nested(config, "audit.fixed_candidates", []))

    for candidate in fixed_candidates:
        pred = candidate_prediction(candidate, raw=raw, anchor=anchor, eval_step=eval_step)
        add_candidate_stats(
            candidate_aggs[str(candidate["name"])],
            pred=pred,
            y_true=y_true,
            fold=fold,
            well_fold=well_fold,
            bucket_code_values=bucket_code_values,
            n_buckets=n_buckets,
        )

    for bucket_id in range(n_buckets):
        mask = bucket_code_values == bucket_id
        if not bool(mask.any()):
            continue
        add_residual_stats(
            total_by_bucket[bucket_id], raw=raw[mask], anchor=anchor[mask], y_true=y_true[mask]
        )
        add_residual_stats(
            by_fold_bucket[(fold, bucket_id)],
            raw=raw[mask],
            anchor=anchor[mask],
            y_true=y_true[mask],
        )
        add_residual_stats(
            by_well_fold_bucket[(well_fold, bucket_id)],
            raw=raw[mask],
            anchor=anchor[mask],
            y_true=y_true[mask],
        )


def build_selection_audit(
    *,
    name: str,
    stats_by_name: dict[str, CandidateAggregate],
    selectable_names: list[str],
    raw_name: str,
    attr_sse: str,
    attr_n: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_stats = stats_by_name[raw_name]
    raw_holdout_sse = getattr(raw_stats, attr_sse)
    raw_holdout_n = getattr(raw_stats, attr_n)

    rows: list[dict[str, Any]] = []
    selected_total_sse = 0.0
    raw_total_sse = 0.0
    selected_total_n = 0.0
    for fold_idx in range(len(raw_holdout_n)):
        if raw_holdout_n[fold_idx] <= 0:
            continue
        train_scores: dict[str, float] = {}
        for candidate_name in selectable_names:
            stats = stats_by_name[candidate_name]
            holdout_sse = float(getattr(stats, attr_sse)[fold_idx])
            holdout_n = float(getattr(stats, attr_n)[fold_idx])
            train_scores[candidate_name] = rmse_from_sse(
                stats.sse - holdout_sse,
                stats.n - holdout_n,
            )
        selected = min(train_scores, key=train_scores.get)
        selected_stats = stats_by_name[selected]
        selected_sse = float(getattr(selected_stats, attr_sse)[fold_idx])
        selected_n = float(getattr(selected_stats, attr_n)[fold_idx])
        raw_sse = float(raw_holdout_sse[fold_idx])
        raw_n = float(raw_holdout_n[fold_idx])
        selected_total_sse += selected_sse
        raw_total_sse += raw_sse
        selected_total_n += selected_n
        rows.append(
            {
                "audit": name,
                "holdout_fold": fold_idx,
                "selected_candidate": selected,
                "train_rmse": round(train_scores[selected], 6),
                "holdout_rmse": round(rmse_from_sse(selected_sse, selected_n), 6),
                "holdout_raw_rmse": round(rmse_from_sse(raw_sse, raw_n), 6),
                "holdout_delta_vs_raw": round(
                    rmse_from_sse(selected_sse, selected_n) - rmse_from_sse(raw_sse, raw_n),
                    6,
                ),
                "rows": int(selected_n),
            }
        )
    selected_rmse = rmse_from_sse(selected_total_sse, selected_total_n)
    raw_rmse = rmse_from_sse(raw_total_sse, selected_total_n)
    return (
        {
            "candidate": name,
            "rmse": round(selected_rmse, 6),
            "raw_holdout_rmse": round(raw_rmse, 6),
            "delta_vs_raw": round(selected_rmse - raw_rmse, 6),
            "rows": int(selected_total_n),
        },
        rows,
    )


def score_bucket_alphas(
    eval_by_bucket: dict[int, dict[str, float]],
    alphas: dict[int, float],
) -> tuple[float, float, int]:
    total_sse = 0.0
    total_n = 0.0
    for bucket_id, stats in eval_by_bucket.items():
        total_n += float(stats["n"])
        total_sse += stats_sse(stats, alphas[bucket_id])
    return rmse_from_sse(total_sse, total_n), total_sse, int(total_n)


def build_alpha_holdout(
    *,
    name: str,
    total_by_bucket: dict[int, dict[str, float]],
    holdout_by_fold_bucket: dict[tuple[int, int], dict[str, float]],
    folds: list[int],
    buckets: list[Bucket],
    alpha_min: float,
    alpha_max: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_sse = 0.0
    total_n = 0
    alpha_rows: list[dict[str, Any]] = []
    for fold in folds:
        eval_by_bucket: dict[int, dict[str, float]] = {}
        alphas: dict[int, float] = {}
        for bucket_id, bucket in enumerate(buckets):
            eval_stats = holdout_by_fold_bucket[(fold, bucket_id)]
            train_stats = subtract_stats(total_by_bucket[bucket_id], eval_stats)
            alpha = fit_alpha(train_stats, alpha_min=alpha_min, alpha_max=alpha_max)
            eval_by_bucket[bucket_id] = eval_stats
            alphas[bucket_id] = alpha
            alpha_rows.append(
                {
                    "audit": name,
                    "holdout_fold": fold,
                    "bucket": bucket.name,
                    "alpha": alpha,
                    "train_rows": int(train_stats["n"]),
                    "eval_rows": int(eval_stats["n"]),
                    "eval_rmse": round(stats_rmse(eval_stats, alpha), 6),
                    "eval_raw_rmse": round(stats_rmse(eval_stats, 1.0), 6),
                    "eval_anchor_rmse": round(stats_rmse(eval_stats, 0.0), 6),
                }
            )
        _, fold_sse, fold_n = score_bucket_alphas(eval_by_bucket, alphas)
        total_sse += fold_sse
        total_n += fold_n
    rmse = rmse_from_sse(total_sse, total_n)
    raw_stats = combine_stats(total_by_bucket)
    raw_rmse = stats_rmse(raw_stats, 1.0)
    return (
        {
            "candidate": name,
            "rmse": round(rmse, 6),
            "delta_vs_raw": round(rmse - raw_rmse, 6),
            "rows": total_n,
        },
        alpha_rows,
    )


def combine_stats(by_bucket: dict[int, dict[str, float]]) -> dict[str, float]:
    total = empty_stats()
    for stats in by_bucket.values():
        for stat_field in STAT_FIELDS:
            total[stat_field] += float(stats.get(stat_field, 0.0))
    return total


def predict_valid_and_aggregate(
    *,
    files: list[Path],
    config: dict[str, Any],
    model: Any,
    fold: int,
    candidate_aggs: dict[str, CandidateAggregate],
    total_by_bucket: dict[int, dict[str, float]],
    by_fold_bucket: dict[tuple[int, int], dict[str, float]],
    by_well_fold_bucket: dict[tuple[int, int], dict[str, float]],
    buckets: list[Bucket],
    well_holdout_folds: int,
) -> tuple[float, int]:
    target_column = str(config_get(config, "data.target_column", "TVT"))
    fold_sse = 0.0
    fold_n = 0
    for path in files:
        df = pd.read_csv(path)
        frame = build_drift_feature_frame(df, config, include_target=True)
        if frame.target_residual is None or frame.eval_indices.size == 0:
            continue
        raw = predict_drift(frame, model, config)
        anchor = frame.baseline_prediction.astype(float)
        y_true = df.loc[frame.eval_indices, target_column].to_numpy(dtype=float)
        eval_step = frame.features["eval_step"].to_numpy(dtype=float)
        well_id = well_id_from_path(path)
        aggregate_predictions(
            config=config,
            candidate_aggs=candidate_aggs,
            total_by_bucket=total_by_bucket,
            by_fold_bucket=by_fold_bucket,
            by_well_fold_bucket=by_well_fold_bucket,
            buckets=buckets,
            fold=fold,
            well_id=well_id,
            raw=raw,
            anchor=anchor,
            y_true=y_true,
            eval_step=eval_step,
            well_holdout_folds=well_holdout_folds,
        )
        diff = raw - y_true
        fold_sse += float(np.square(diff).sum())
        fold_n += int(diff.size)
    return fold_sse, fold_n


def selected_variant(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    variants = list(get_nested(config, "audit.training_variants.variants", []))
    selected_name = str(get_nested(config, "audit.training_variants.selected_variant"))
    selected_index = int(get_nested(config, "audit.selected_variant_index", 0))
    for idx, variant in enumerate(variants):
        if str(variant["name"]) == selected_name:
            return selected_index if selected_index == idx else idx, variant
    raise ValueError(f"selected variant not found: {selected_name}")


def run_audit(
    *,
    files: list[Path],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(config_get(config, "validation.seed", 42))
    n_folds = int(config_get(config, "validation.n_folds", 5))
    groups = np.asarray([well_id_from_path(path) for path in files])
    splitter = GroupKFold(n_splits=n_folds)
    buckets = load_buckets(config)
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    alpha_min = float(get_nested(config, "audit.alpha_clip.min", 0.2))
    alpha_max = float(get_nested(config, "audit.alpha_clip.max", 1.15))
    variant_index, variant = selected_variant(config)
    variant_name = str(variant["name"])
    max_rows_per_well = int(config_get(config, "model.training.max_train_rows_per_well", 800))
    max_rows_total = int(config_get(config, "model.training.max_train_rows_per_fold", 300000))

    candidate_aggs = make_candidate_aggregates(
        config,
        n_folds=n_folds,
        well_holdout_folds=well_holdout_folds,
        n_buckets=len(buckets),
    )
    total_by_bucket: dict[int, dict[str, float]] = defaultdict(empty_stats)
    by_fold_bucket: dict[tuple[int, int], dict[str, float]] = defaultdict(empty_stats)
    by_well_fold_bucket: dict[tuple[int, int], dict[str, float]] = defaultdict(empty_stats)
    source_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(files, groups=groups)):
        train_paths = [files[index] for index in train_idx]
        valid_paths = [files[index] for index in valid_idx]
        rng = np.random.default_rng(seed + fold * 1009 + variant_index)
        train = collect_training_rows(
            train_paths,
            config,
            rng,
            variant,
            max_rows_per_well=max_rows_per_well,
            max_rows_total=max_rows_total,
        )
        source_rows.extend(source_summary_rows(train, variant=variant_name, fold=fold))
        weights = sample_weights(train, variant)
        if str(variant.get("weight_profile", "uniform")) == "uniform":
            weights = None
        print(f"Fold {fold}: fitting {variant_name} on {len(train):,} sampled rows")
        model = fit_model(train, config, seed=seed + fold, weights=weights)
        importance_rows.extend(
            feature_importance_rows(model, config, variant=variant_name, fold=fold, segment="all")
        )
        fold_sse, fold_n = predict_valid_and_aggregate(
            files=valid_paths,
            config=config,
            model=model,
            fold=fold,
            candidate_aggs=candidate_aggs,
            total_by_bucket=total_by_bucket,
            by_fold_bucket=by_fold_bucket,
            by_well_fold_bucket=by_well_fold_bucket,
            buckets=buckets,
            well_holdout_folds=well_holdout_folds,
        )
        fold_rmse = rmse_from_sse(fold_sse, fold_n)
        fold_rows.append(
            {
                "variant": variant_name,
                "fold": fold,
                "rmse": round(fold_rmse, 6),
                "rows": fold_n,
                "train_rows": len(train),
                "valid_wells": len(valid_paths),
            }
        )
        print(f"Fold {fold}: raw RMSE={fold_rmse:.6f} rows={fold_n:,}")

    raw_name = "raw_pseudo_tail"
    raw_rmse = candidate_aggs[raw_name].rmse
    metric_rows: list[dict[str, Any]] = []
    for item in candidate_aggs.values():
        metric_rows.append(
            {
                "candidate": item.name,
                "method": item.method,
                "rmse": round(item.rmse, 6),
                "delta_vs_raw": round(item.rmse - raw_rmse, 6),
                "rows": item.n,
                "selectable": item.selectable,
                "params_json": json.dumps(item.params, sort_keys=True, separators=(",", ":")),
            }
        )
    metric_rows = sorted(metric_rows, key=lambda row: row["rmse"])

    selectable_names = [name for name, item in candidate_aggs.items() if item.selectable]
    original_selection, original_selection_rows = build_selection_audit(
        name="leave_one_original_fold_out_fixed_candidate_selection",
        stats_by_name=candidate_aggs,
        selectable_names=selectable_names,
        raw_name=raw_name,
        attr_sse="fold_sse",
        attr_n="fold_n",
    )
    well_selection, well_selection_rows = build_selection_audit(
        name="well_hash_holdout_fixed_candidate_selection",
        stats_by_name=candidate_aggs,
        selectable_names=selectable_names,
        raw_name=raw_name,
        attr_sse="well_fold_sse",
        attr_n="well_fold_n",
    )

    in_sample_alphas = {
        bucket_id: fit_alpha(total_by_bucket[bucket_id], alpha_min=alpha_min, alpha_max=alpha_max)
        for bucket_id in range(len(buckets))
    }
    in_sample_rmse, _, _ = score_bucket_alphas(total_by_bucket, in_sample_alphas)
    metric_rows.append(
        {
            "candidate": "same_oof_bucket_alpha_fit",
            "method": "bucket_alpha_fit",
            "rmse": round(in_sample_rmse, 6),
            "delta_vs_raw": round(in_sample_rmse - raw_rmse, 6),
            "rows": candidate_aggs[raw_name].n,
            "selectable": False,
            "params_json": "{}",
        }
    )
    original_alpha, original_alpha_rows = build_alpha_holdout(
        name="leave_one_original_fold_out_bucket_alpha_fit",
        total_by_bucket=total_by_bucket,
        holdout_by_fold_bucket=by_fold_bucket,
        folds=list(range(n_folds)),
        buckets=buckets,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    well_alpha, well_alpha_rows = build_alpha_holdout(
        name="well_hash_holdout_bucket_alpha_fit",
        total_by_bucket=total_by_bucket,
        holdout_by_fold_bucket=by_well_fold_bucket,
        folds=list(range(well_holdout_folds)),
        buckets=buckets,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    metric_rows.extend([original_alpha, well_alpha, original_selection, well_selection])

    bucket_rows: list[dict[str, Any]] = []
    raw_agg = candidate_aggs[raw_name]
    for item in candidate_aggs.values():
        for bucket_id, bucket in enumerate(buckets):
            bucket_rows.append(
                {
                    "candidate": item.name,
                    "method": item.method,
                    "bucket": bucket.name,
                    "max_step": bucket.max_step,
                    "rmse": round(
                        rmse_from_sse(item.bucket_sse[bucket_id], item.bucket_n[bucket_id]), 6
                    ),
                    "raw_rmse": round(
                        rmse_from_sse(raw_agg.bucket_sse[bucket_id], raw_agg.bucket_n[bucket_id]),
                        6,
                    ),
                    "rows": int(item.bucket_n[bucket_id]),
                }
            )
    for bucket_id, bucket in enumerate(buckets):
        stats = total_by_bucket[bucket_id]
        bucket_rows.append(
            {
                "candidate": "same_oof_bucket_alpha_fit",
                "method": "bucket_alpha_fit",
                "bucket": bucket.name,
                "max_step": bucket.max_step,
                "rmse": round(stats_rmse(stats, in_sample_alphas[bucket_id]), 6),
                "raw_rmse": round(stats_rmse(stats, 1.0), 6),
                "rows": int(stats["n"]),
                "alpha": in_sample_alphas[bucket_id],
            }
        )

    alpha_rows: list[dict[str, Any]] = []
    for bucket_id, bucket in enumerate(buckets):
        stats = total_by_bucket[bucket_id]
        alpha_rows.append(
            {
                "audit": "same_oof_bucket_alpha_fit",
                "holdout_fold": "all",
                "bucket": bucket.name,
                "alpha": in_sample_alphas[bucket_id],
                "train_rows": int(stats["n"]),
                "eval_rows": int(stats["n"]),
                "eval_rmse": round(stats_rmse(stats, in_sample_alphas[bucket_id]), 6),
                "eval_raw_rmse": round(stats_rmse(stats, 1.0), 6),
                "eval_anchor_rmse": round(stats_rmse(stats, 0.0), 6),
            }
        )
    alpha_rows.extend(original_alpha_rows)
    alpha_rows.extend(well_alpha_rows)

    supported_methods: list[dict[str, Any]] = []
    if original_alpha["rmse"] < raw_rmse and well_alpha["rmse"] < raw_rmse:
        supported_methods.append(
            {
                "method": "leave_one_original_fold_out_bucket_alpha_fit",
                "original_fold_cv": float(original_alpha["rmse"]),
                "well_hash_cv": float(well_alpha["rmse"]),
            }
        )
    if original_selection["rmse"] < raw_rmse and well_selection["rmse"] < raw_rmse:
        selected_fixed = "fixed_candidate_selection"
        selected_names = {
            str(row["selected_candidate"]) for row in original_selection_rows + well_selection_rows
        }
        if len(selected_names) == 1:
            selected_fixed = selected_names.pop()
        supported_methods.append(
            {
                "method": selected_fixed,
                "original_fold_cv": float(original_selection["rmse"]),
                "well_hash_cv": float(well_selection["rmse"]),
            }
        )
    best_supported = min(
        supported_methods,
        key=lambda item: (item["original_fold_cv"], item["well_hash_cv"]),
        default=None,
    )
    clean_postprocess_supported = best_supported is not None
    selected_clean_cv = (
        float(best_supported["original_fold_cv"]) if best_supported is not None else raw_rmse
    )
    selected_method = (
        str(best_supported["method"]) if best_supported is not None else "raw_pseudo_tail"
    )

    summary = {
        "experiment": "exp025_pseudo_tail_postprocess_cv_audit",
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "source_variant": variant_name,
        "parent_best_cv": get_nested(config, "audit.parent_best_cv"),
        "parent_public_lb": get_nested(config, "audit.parent_public_lb"),
        "raw_pseudo_tail_cv": round(raw_rmse, 6),
        "same_oof_bucket_alpha_fit_cv": round(in_sample_rmse, 6),
        "leave_one_original_fold_out_bucket_alpha_fit_cv": original_alpha["rmse"],
        "well_hash_holdout_bucket_alpha_fit_cv": well_alpha["rmse"],
        "leave_one_original_fold_out_fixed_selection_cv": original_selection["rmse"],
        "well_hash_holdout_fixed_selection_cv": well_selection["rmse"],
        "selected_clean_cv": round(selected_clean_cv, 6),
        "selected_method": selected_method,
        "supported_methods": supported_methods,
        "clean_postprocess_supported": clean_postprocess_supported,
        "cv": round(selected_clean_cv, 6),
        "public_lb": None,
        "metric": "rmse",
        "artifact_rows": {
            "pseudo_tail_postprocess_metrics": len(metric_rows),
            "pseudo_tail_postprocess_selection": len(original_selection_rows + well_selection_rows),
            "pseudo_tail_postprocess_alphas": len(alpha_rows),
            "pseudo_tail_postprocess_bucket_summary": len(bucket_rows),
            "pseudo_tail_postprocess_fold_metrics": len(fold_rows),
            "pseudo_tail_source_summary": len(source_rows),
            "pseudo_tail_feature_importance": len(importance_rows),
        },
        "metrics": metric_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "pseudo_tail_postprocess_metrics.csv", index=False
    )
    pd.DataFrame(original_selection_rows + well_selection_rows).to_csv(
        output_dir / "pseudo_tail_postprocess_selection.csv", index=False
    )
    pd.DataFrame(alpha_rows).to_csv(output_dir / "pseudo_tail_postprocess_alphas.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "pseudo_tail_postprocess_bucket_summary.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "pseudo_tail_postprocess_fold_metrics.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(output_dir / "pseudo_tail_source_summary.csv", index=False)
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(
            output_dir / "pseudo_tail_feature_importance.csv", index=False
        )
    (output_dir / "pseudo_tail_postprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    ExperimentPaths().metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_yaml(Path(__file__).with_name("config.yaml"))
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    files = train_files(paths, args.max_wells)
    summary = run_audit(files=files, config=config, output_dir=output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
