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
    with_pseudo_cutoff,
)
from settings import ExperimentPaths
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Bucket:
    name: str
    max_step: float


@dataclass
class CandidateStats:
    name: str
    method: str
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
    parser = argparse.ArgumentParser(description="Audit test-time prefix calibration candidates.")
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


def bucket_alpha_array(eval_step: np.ndarray, bucket_config: list[dict[str, Any]]) -> np.ndarray:
    alpha = np.ones(eval_step.shape, dtype=float)
    previous_max = -np.inf
    for bucket in bucket_config:
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        alpha[mask] = float(bucket.get("alpha", 1.0))
        previous_max = max_step
    return alpha


def exp026_bucket_shrink(
    *,
    raw: np.ndarray,
    anchor: np.ndarray,
    eval_step: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    buckets = list(get_nested(config, "postprocess.methods.distance_bucket_shrink.buckets", []))
    if not buckets:
        return raw.copy()
    alpha = bucket_alpha_array(eval_step, buckets)
    return anchor + alpha * (raw - anchor)


def choose_calibration_cutoffs(df: pd.DataFrame, config: dict[str, Any]) -> list[int]:
    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    if known_indices.size == 0:
        return []

    original_last_known = int(known_indices[-1])
    min_prefix_rows = int(get_nested(config, "audit.calibration.min_prefix_rows", 200))
    min_calibration_rows = int(get_nested(config, "audit.calibration.min_calibration_rows", 80))
    min_cutoff = max(0, min_prefix_rows - 1)
    max_cutoff = original_last_known - min_calibration_rows
    if max_cutoff < min_cutoff:
        return []

    quantiles = list(get_nested(config, "audit.calibration.cutoff_quantiles", [0.35, 0.55, 0.75]))
    cutoffs: list[int] = []
    for quantile in quantiles:
        clipped = float(np.clip(float(quantile), 0.0, 1.0))
        cutoff = int(round(min_cutoff + clipped * (max_cutoff - min_cutoff)))
        cutoff = int(np.clip(cutoff, min_cutoff, max_cutoff))
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    return cutoffs


def fit_alpha(pred_residual: np.ndarray, true_residual: np.ndarray, config: dict[str, Any]) -> float:
    valid = np.isfinite(pred_residual) & np.isfinite(true_residual)
    if not bool(valid.any()):
        return 1.0
    denom = float(np.square(pred_residual[valid]).sum())
    if denom <= 0.0:
        return 1.0
    alpha = float((pred_residual[valid] * true_residual[valid]).sum()) / denom
    alpha_min = float(get_nested(config, "audit.calibration.alpha_clip.min", 0.2))
    alpha_max = float(get_nested(config, "audit.calibration.alpha_clip.max", 1.15))
    return float(np.clip(alpha, alpha_min, alpha_max))


def clipped_correction(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    max_abs = float(get_nested(config, "audit.calibration.max_abs_correction", 25.0))
    return np.clip(values, -max_abs, max_abs)


def calibration_candidates_for_cutoff(
    *,
    df: pd.DataFrame,
    cutoff_index: int,
    full_raw: np.ndarray,
    full_anchor: np.ndarray,
    full_eval_step: np.ndarray,
    full_control: np.ndarray,
    model: Any,
    config: dict[str, Any],
    buckets: list[Bucket],
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    target_column = str(config_get(config, "data.target_column", "TVT"))
    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    if known_indices.size == 0:
        return {}, None
    original_last_known = int(known_indices[-1])

    pseudo_df = with_pseudo_cutoff(df, cutoff_index)
    pseudo_frame = build_drift_feature_frame(pseudo_df, config, include_target=True)
    if pseudo_frame.eval_indices.size == 0:
        return {}, None

    pseudo_raw = predict_drift(pseudo_frame, model, config)
    pseudo_anchor = pseudo_frame.baseline_prediction.astype(float)
    pseudo_eval_step = pseudo_frame.features["eval_step"].to_numpy(dtype=float)
    pseudo_control = exp026_bucket_shrink(
        raw=pseudo_raw,
        anchor=pseudo_anchor,
        eval_step=pseudo_eval_step,
        config=config,
    )
    pseudo_eval_indices = pseudo_frame.eval_indices.astype(int)
    cal_mask = pseudo_eval_indices <= original_last_known
    if not bool(cal_mask.any()):
        return {}, None

    y_cal = df.loc[pseudo_eval_indices[cal_mask], target_column].to_numpy(dtype=float)
    pred_cal = pseudo_control[cal_mask]
    raw_cal = pseudo_raw[cal_mask]
    anchor_cal = pseudo_anchor[cal_mask]
    cal_eval_step = pseudo_eval_step[cal_mask]
    cal_error = y_cal - pred_cal
    finite = np.isfinite(cal_error) & np.isfinite(pred_cal) & np.isfinite(y_cal)
    if int(finite.sum()) < int(get_nested(config, "audit.calibration.min_finite_rows", 30)):
        return {}, None

    cal_error = cal_error[finite]
    raw_cal = raw_cal[finite]
    anchor_cal = anchor_cal[finite]
    cal_eval_step = cal_eval_step[finite]
    y_cal = y_cal[finite]

    score_offset = float(original_last_known - cutoff_index + 1)
    x_score = score_offset + full_eval_step
    correction_by_candidate: dict[str, np.ndarray] = {}

    bias = float(np.mean(cal_error))
    correction_by_candidate["prefix_bias_add"] = full_control + clipped_correction(
        np.full(full_control.shape, bias, dtype=float), config
    )

    x_cal = cal_eval_step.astype(float)
    if x_cal.size >= 2 and float(np.nanstd(x_cal)) > 0.0:
        slope, intercept = np.polyfit(x_cal, cal_error, deg=1)
        max_abs_slope = float(get_nested(config, "audit.calibration.max_abs_error_slope", 0.03))
        slope = float(np.clip(slope, -max_abs_slope, max_abs_slope))
        linear_correction = intercept + slope * x_score
    else:
        slope = 0.0
        intercept = bias
        linear_correction = np.full(full_control.shape, bias, dtype=float)
    correction_by_candidate["prefix_error_slope"] = full_control + clipped_correction(
        linear_correction, config
    )

    global_alpha = fit_alpha(raw_cal - anchor_cal, y_cal - anchor_cal, config)
    correction_by_candidate["prefix_global_residual_shrink"] = (
        full_anchor + global_alpha * (full_raw - full_anchor)
    )

    cal_bucket_codes = bucket_codes(cal_eval_step, buckets)
    full_bucket_codes = bucket_codes(full_eval_step, buckets)
    bucket_alpha = np.full(len(buckets), global_alpha, dtype=float)
    for bucket_id in range(len(buckets)):
        mask = cal_bucket_codes == bucket_id
        if bool(mask.any()):
            bucket_alpha[bucket_id] = fit_alpha(
                raw_cal[mask] - anchor_cal[mask],
                y_cal[mask] - anchor_cal[mask],
                config,
            )
    correction_by_candidate["prefix_distance_bucket_shrink"] = (
        full_anchor + bucket_alpha[full_bucket_codes] * (full_raw - full_anchor)
    )

    near_rows = int(get_nested(config, "audit.calibration.near_continuity_rows", 32))
    near_error = float(np.median(cal_error[-near_rows:]))
    tau = float(get_nested(config, "audit.calibration.near_continuity_tau_rows", 80.0))
    decay = np.exp(-np.maximum(full_eval_step, 0.0) / max(tau, 1.0))
    correction_by_candidate["prefix_near_continuity_decay"] = full_control + clipped_correction(
        near_error * decay, config
    )

    prefix_range = float(np.nanmax(tvt_input[known_indices]) - np.nanmin(tvt_input[known_indices]))
    cal_pred_range = float(np.nanmax(pred_cal[finite]) - np.nanmin(pred_cal[finite]))
    summary_row = {
        "cutoff_index": cutoff_index,
        "original_last_known_index": original_last_known,
        "calibration_rows": int(finite.sum()),
        "calibration_eval_step_min": int(np.nanmin(cal_eval_step)),
        "calibration_eval_step_max": int(np.nanmax(cal_eval_step)),
        "calibration_rmse": round(rmse_from_sse(float(np.square(cal_error).sum()), len(cal_error)), 6),
        "calibration_bias": round(bias, 6),
        "calibration_error_slope": round(float(slope), 8),
        "global_alpha": round(float(global_alpha), 8),
        "near_error": round(near_error, 6),
        "prefix_tvt_range": round(prefix_range, 6),
        "calibration_pred_range": round(cal_pred_range, 6),
        "bucket_alpha_json": json.dumps(
            {buckets[idx].name: round(float(value), 8) for idx, value in enumerate(bucket_alpha)},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    return correction_by_candidate, summary_row


def selected_variant(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    variants = list(get_nested(config, "audit.training_variants.variants", []))
    selected_name = str(get_nested(config, "audit.training_variants.selected_variant"))
    selected_index = int(get_nested(config, "audit.selected_variant_index", 4))
    for idx, variant in enumerate(variants):
        if str(variant["name"]) == selected_name:
            return selected_index if selected_index == idx else idx, variant
    raise ValueError(f"selected variant not found: {selected_name}")


def add_stats(
    stats: CandidateStats,
    *,
    pred: np.ndarray,
    y_true: np.ndarray,
    fold: int,
    well_fold: int,
    bucket_code_values: np.ndarray,
    n_buckets: int,
) -> None:
    mask = np.isfinite(pred) & np.isfinite(y_true)
    if not bool(mask.any()):
        return
    diff2 = np.square(pred[mask] - y_true[mask])
    bucket_code_values = bucket_code_values[mask]
    stats.n += int(diff2.size)
    stats.sse += float(diff2.sum())
    stats.fold_sse[fold] += float(diff2.sum())
    stats.fold_n[fold] += float(diff2.size)
    stats.well_fold_sse[well_fold] += float(diff2.sum())
    stats.well_fold_n[well_fold] += float(diff2.size)
    stats.bucket_sse += np.bincount(bucket_code_values, weights=diff2, minlength=n_buckets)
    stats.bucket_n += np.bincount(bucket_code_values, minlength=n_buckets).astype(float)


def average_candidate_predictions(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    names = sorted(set().union(*(part.keys() for part in parts)))
    averaged: dict[str, np.ndarray] = {}
    for name in names:
        arrays = [part[name] for part in parts if name in part]
        if arrays:
            averaged[name] = np.mean(np.vstack(arrays), axis=0)
    return averaged


def predict_valid_well_candidates(
    *,
    path: Path,
    model: Any,
    config: dict[str, Any],
    buckets: list[Bucket],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_column = str(config_get(config, "data.target_column", "TVT"))
    df = pd.read_csv(path)
    full_frame = build_drift_feature_frame(df, config, include_target=True)
    if full_frame.target_residual is None or full_frame.eval_indices.size == 0:
        return {}, np.asarray([], dtype=float), np.asarray([], dtype=np.int16), []

    full_raw = predict_drift(full_frame, model, config)
    full_anchor = full_frame.baseline_prediction.astype(float)
    full_eval_step = full_frame.features["eval_step"].to_numpy(dtype=float)
    full_control = exp026_bucket_shrink(
        raw=full_raw,
        anchor=full_anchor,
        eval_step=full_eval_step,
        config=config,
    )
    y_true = df.loc[full_frame.eval_indices, target_column].to_numpy(dtype=float)
    bucket_code_values = bucket_codes(full_eval_step, buckets)

    candidates: dict[str, np.ndarray] = {
        "raw_pseudo_tail": full_raw,
        "exp026_bucket_shrink_control": full_control,
    }
    cutoff_prediction_parts: list[dict[str, np.ndarray]] = []
    cutoff_rows: list[dict[str, Any]] = []
    well_id = well_id_from_path(path)
    for cutoff_index in choose_calibration_cutoffs(df, config):
        part, cutoff_summary = calibration_candidates_for_cutoff(
            df=df,
            cutoff_index=cutoff_index,
            full_raw=full_raw,
            full_anchor=full_anchor,
            full_eval_step=full_eval_step,
            full_control=full_control,
            model=model,
            config=config,
            buckets=buckets,
        )
        if not part or cutoff_summary is None:
            continue
        cutoff_prediction_parts.append(part)
        cutoff_summary["well_id"] = well_id
        cutoff_rows.append(cutoff_summary)

    candidates.update(average_candidate_predictions(cutoff_prediction_parts))
    for name in (
        "prefix_bias_add",
        "prefix_error_slope",
        "prefix_global_residual_shrink",
        "prefix_distance_bucket_shrink",
        "prefix_near_continuity_decay",
    ):
        if name not in candidates:
            candidates[name] = full_control.copy()
    return candidates, y_true, bucket_code_values, cutoff_rows


def selection_audit(
    *,
    name: str,
    stats_by_name: dict[str, CandidateStats],
    candidate_names: list[str],
    control_name: str,
    attr_sse: str,
    attr_n: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control = stats_by_name[control_name]
    control_sse = getattr(control, attr_sse)
    control_n = getattr(control, attr_n)

    rows: list[dict[str, Any]] = []
    selected_total_sse = 0.0
    control_total_sse = 0.0
    selected_total_n = 0.0
    for fold_idx in range(len(control_n)):
        if control_n[fold_idx] <= 0:
            continue
        train_scores: dict[str, float] = {}
        for candidate_name in candidate_names:
            candidate = stats_by_name[candidate_name]
            holdout_sse = float(getattr(candidate, attr_sse)[fold_idx])
            holdout_n = float(getattr(candidate, attr_n)[fold_idx])
            train_scores[candidate_name] = rmse_from_sse(
                candidate.sse - holdout_sse,
                candidate.n - holdout_n,
            )
        selected = min(train_scores, key=train_scores.get)
        selected_stats = stats_by_name[selected]
        selected_sse = float(getattr(selected_stats, attr_sse)[fold_idx])
        selected_n = float(getattr(selected_stats, attr_n)[fold_idx])
        fold_control_sse = float(control_sse[fold_idx])
        fold_control_n = float(control_n[fold_idx])
        selected_total_sse += selected_sse
        control_total_sse += fold_control_sse
        selected_total_n += selected_n
        rows.append(
            {
                "audit": name,
                "holdout_fold": fold_idx,
                "selected_candidate": selected,
                "train_rmse": round(train_scores[selected], 6),
                "holdout_rmse": round(rmse_from_sse(selected_sse, selected_n), 6),
                "holdout_control_rmse": round(rmse_from_sse(fold_control_sse, fold_control_n), 6),
                "holdout_delta_vs_control": round(
                    rmse_from_sse(selected_sse, selected_n)
                    - rmse_from_sse(fold_control_sse, fold_control_n),
                    6,
                ),
                "rows": int(selected_n),
            }
        )
    selected_rmse = rmse_from_sse(selected_total_sse, selected_total_n)
    control_rmse = rmse_from_sse(control_total_sse, selected_total_n)
    return (
        {
            "candidate": name,
            "method": "heldout_candidate_selection",
            "rmse": round(selected_rmse, 6),
            "control_rmse": round(control_rmse, 6),
            "delta_vs_control": round(selected_rmse - control_rmse, 6),
            "rows": int(selected_total_n),
        },
        rows,
    )


def run_audit(
    *,
    files: list[Path],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(config_get(config, "validation.seed", 42))
    n_folds = int(config_get(config, "validation.n_folds", 5))
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    groups = np.asarray([well_id_from_path(path) for path in files])
    splitter = GroupKFold(n_splits=n_folds)
    buckets = load_buckets(config)
    variant_index, variant = selected_variant(config)
    variant_name = str(variant["name"])
    max_rows_per_well = int(config_get(config, "model.training.max_train_rows_per_well", 800))
    max_rows_total = int(config_get(config, "model.training.max_train_rows_per_fold", 300000))

    candidate_methods = {
        "raw_pseudo_tail": "raw",
        "exp026_bucket_shrink_control": "fixed_exp014_bucket_shrink",
        "prefix_bias_add": "test_time_prefix_bias",
        "prefix_error_slope": "test_time_prefix_error_slope",
        "prefix_global_residual_shrink": "test_time_prefix_global_alpha",
        "prefix_distance_bucket_shrink": "test_time_prefix_bucket_alpha",
        "prefix_near_continuity_decay": "test_time_near_continuity",
    }
    stats_by_name = {
        name: CandidateStats(
            name=name,
            method=method,
            fold_sse=np.zeros(n_folds, dtype=float),
            fold_n=np.zeros(n_folds, dtype=float),
            well_fold_sse=np.zeros(well_holdout_folds, dtype=float),
            well_fold_n=np.zeros(well_holdout_folds, dtype=float),
            bucket_sse=np.zeros(len(buckets), dtype=float),
            bucket_n=np.zeros(len(buckets), dtype=float),
        )
        for name, method in candidate_methods.items()
    }
    source_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []

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

        fold_sse = 0.0
        fold_n = 0
        for path in valid_paths:
            well_id = well_id_from_path(path)
            well_fold = stable_fold(well_id, well_holdout_folds)
            candidates, y_true, bucket_code_values, well_cutoff_rows = predict_valid_well_candidates(
                path=path,
                model=model,
                config=config,
                buckets=buckets,
            )
            if not candidates:
                continue
            for row in well_cutoff_rows:
                row["fold"] = fold
                row["well_hash_fold"] = well_fold
            cutoff_rows.extend(well_cutoff_rows)
            for name, pred in candidates.items():
                add_stats(
                    stats_by_name[name],
                    pred=pred,
                    y_true=y_true,
                    fold=fold,
                    well_fold=well_fold,
                    bucket_code_values=bucket_code_values,
                    n_buckets=len(buckets),
                )
            control_pred = candidates["exp026_bucket_shrink_control"]
            fold_sse += float(np.square(control_pred - y_true).sum())
            fold_n += int(y_true.size)
        fold_rows.append(
            {
                "fold": fold,
                "control_rmse": round(rmse_from_sse(fold_sse, fold_n), 6),
                "rows": fold_n,
                "train_rows": len(train),
                "valid_wells": len(valid_paths),
            }
        )
        print(f"Fold {fold}: control RMSE={rmse_from_sse(fold_sse, fold_n):.6f} rows={fold_n:,}")

    control_name = "exp026_bucket_shrink_control"
    control_rmse = stats_by_name[control_name].rmse
    metric_rows: list[dict[str, Any]] = []
    for item in stats_by_name.values():
        metric_rows.append(
            {
                "candidate": item.name,
                "method": item.method,
                "rmse": round(item.rmse, 6),
                "control_rmse": round(control_rmse, 6),
                "delta_vs_control": round(item.rmse - control_rmse, 6),
                "rows": item.n,
            }
        )
    metric_rows = sorted(metric_rows, key=lambda row: row["rmse"])

    candidate_names = [name for name in stats_by_name if name != "raw_pseudo_tail"]
    original_selection, original_selection_rows = selection_audit(
        name="leave_one_original_fold_out_prefix_candidate_selection",
        stats_by_name=stats_by_name,
        candidate_names=candidate_names,
        control_name=control_name,
        attr_sse="fold_sse",
        attr_n="fold_n",
    )
    well_selection, well_selection_rows = selection_audit(
        name="well_hash_holdout_prefix_candidate_selection",
        stats_by_name=stats_by_name,
        candidate_names=candidate_names,
        control_name=control_name,
        attr_sse="well_fold_sse",
        attr_n="well_fold_n",
    )
    metric_rows.extend([original_selection, well_selection])

    bucket_rows: list[dict[str, Any]] = []
    control_stats = stats_by_name[control_name]
    for item in stats_by_name.values():
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
                    "control_rmse": round(
                        rmse_from_sse(
                            control_stats.bucket_sse[bucket_id],
                            control_stats.bucket_n[bucket_id],
                        ),
                        6,
                    ),
                    "rows": int(item.bucket_n[bucket_id]),
                }
            )

    fold_metric_rows: list[dict[str, Any]] = []
    for item in stats_by_name.values():
        for fold in range(n_folds):
            fold_metric_rows.append(
                {
                    "candidate": item.name,
                    "method": item.method,
                    "fold": fold,
                    "rmse": round(rmse_from_sse(item.fold_sse[fold], item.fold_n[fold]), 6),
                    "rows": int(item.fold_n[fold]),
                }
            )

    well_holdout_rows: list[dict[str, Any]] = []
    for item in stats_by_name.values():
        for fold in range(well_holdout_folds):
            well_holdout_rows.append(
                {
                    "candidate": item.name,
                    "method": item.method,
                    "well_hash_fold": fold,
                    "rmse": round(
                        rmse_from_sse(item.well_fold_sse[fold], item.well_fold_n[fold]), 6
                    ),
                    "rows": int(item.well_fold_n[fold]),
                }
            )

    best_fixed = min(
        [row for row in metric_rows if row["candidate"] in candidate_names],
        key=lambda row: row["rmse"],
    )
    clean_supported = (
        original_selection["rmse"] < control_rmse and well_selection["rmse"] < control_rmse
    )
    selected_method = (
        "heldout_prefix_candidate_selection" if clean_supported else "exp026_bucket_shrink_control"
    )
    selected_cv = float(original_selection["rmse"] if clean_supported else control_rmse)

    summary = {
        "experiment": "exp036_test_time_prefix_calibration_audit",
        "status": "implemented",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "source_variant": variant_name,
        "control_candidate": control_name,
        "control_cv": round(control_rmse, 6),
        "raw_pseudo_tail_cv": round(stats_by_name["raw_pseudo_tail"].rmse, 6),
        "best_same_oof_candidate": best_fixed["candidate"],
        "best_same_oof_cv": best_fixed["rmse"],
        "best_same_oof_delta_vs_control": best_fixed["delta_vs_control"],
        "leave_one_original_fold_out_selection_cv": original_selection["rmse"],
        "well_hash_holdout_selection_cv": well_selection["rmse"],
        "clean_prefix_calibration_supported": clean_supported,
        "selected_method": selected_method,
        "cv": round(selected_cv, 6),
        "public_lb": None,
        "metric": "rmse",
        "artifact_rows": {
            "prefix_calibration_candidate_metrics": len(metric_rows),
            "prefix_calibration_bucket_summary": len(bucket_rows),
            "prefix_calibration_fold_metrics": len(fold_metric_rows),
            "prefix_calibration_well_holdout_metrics": len(well_holdout_rows),
            "prefix_calibration_cutoff_summary": len(cutoff_rows),
            "prefix_calibration_selection": len(original_selection_rows + well_selection_rows),
            "pseudo_tail_source_summary": len(source_rows),
            "pseudo_tail_feature_importance": len(importance_rows),
        },
        "metrics": metric_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "prefix_calibration_candidate_metrics.csv", index=False
    )
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "prefix_calibration_bucket_summary.csv", index=False
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        output_dir / "prefix_calibration_fold_metrics.csv", index=False
    )
    pd.DataFrame(well_holdout_rows).to_csv(
        output_dir / "prefix_calibration_well_holdout_metrics.csv", index=False
    )
    pd.DataFrame(cutoff_rows).to_csv(
        output_dir / "prefix_calibration_cutoff_summary.csv", index=False
    )
    pd.DataFrame(original_selection_rows + well_selection_rows).to_csv(
        output_dir / "prefix_calibration_selection.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(output_dir / "pseudo_tail_source_summary.csv", index=False)
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(
            output_dir / "pseudo_tail_feature_importance.csv", index=False
        )
    (output_dir / "prefix_calibration_summary.json").write_text(
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
