from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser = argparse.ArgumentParser(description="Audit test-time prefix online training.")
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


def selected_variant(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    variants = list(get_nested(config, "audit.training_variants.variants", []))
    selected_name = str(get_nested(config, "audit.training_variants.selected_variant"))
    selected_index = int(get_nested(config, "audit.selected_variant_index", 4))
    for idx, variant in enumerate(variants):
        if str(variant["name"]) == selected_name:
            return selected_index if selected_index == idx else idx, variant
    raise ValueError(f"selected variant not found: {selected_name}")


def choose_online_cutoffs(df: pd.DataFrame, config: dict[str, Any]) -> list[int]:
    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    if known_indices.size == 0:
        return []

    original_last_known = int(known_indices[-1])
    min_prefix_rows = int(get_nested(config, "audit.online_training.min_prefix_rows", 200))
    min_online_rows = int(get_nested(config, "audit.online_training.min_online_rows", 80))
    min_cutoff = max(0, min_prefix_rows - 1)
    max_cutoff = original_last_known - min_online_rows
    if max_cutoff < min_cutoff:
        return []

    n_aug_splits = int(get_nested(config, "audit.online_training.n_aug_splits", 1))
    quantiles = list(get_nested(config, "audit.online_training.cutoff_quantiles", [0.55]))
    if not quantiles:
        quantiles = np.linspace(0.45, 0.75, max(1, n_aug_splits)).tolist()
    cutoffs: list[int] = []
    for quantile in quantiles[:n_aug_splits]:
        clipped = float(np.clip(float(quantile), 0.0, 1.0))
        cutoff = int(round(min_cutoff + clipped * (max_cutoff - min_cutoff)))
        cutoff = int(np.clip(cutoff, min_cutoff, max_cutoff))
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    return cutoffs


def online_rows_from_well(
    *,
    path: Path,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    df = pd.read_csv(path)
    target_column = str(config_get(config, "data.target_column", "TVT"))
    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    if known_indices.size == 0:
        return None, []
    original_last_known = int(known_indices[-1])
    max_rows_per_well = int(get_nested(config, "audit.online_training.max_rows_per_well", 220))
    min_finite_rows = int(get_nested(config, "audit.online_training.min_finite_rows", 30))
    source_kind = str(
        get_nested(config, "audit.online_training.online_source_kind", "test_time_prefix_online")
    )
    well_id = well_id_from_path(path)

    parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for cutoff_index in choose_online_cutoffs(df, config):
        pseudo_df = with_pseudo_cutoff(df, cutoff_index)
        frame = build_drift_feature_frame(pseudo_df, config, include_target=True)
        if frame.target_residual is None or frame.eval_indices.size == 0:
            continue
        visible_mask = frame.eval_indices.astype(int) <= original_last_known
        finite_mask = np.isfinite(frame.target_residual)
        selected = np.flatnonzero(visible_mask & finite_mask)
        if selected.size < min_finite_rows:
            continue
        if max_rows_per_well > 0 and selected.size > max_rows_per_well:
            selected = rng.choice(selected, size=max_rows_per_well, replace=False)
            selected = np.sort(selected)

        part = frame.features.iloc[selected].copy()
        part["target_residual"] = frame.target_residual[selected]
        part["well_id"] = well_id
        part["source_kind"] = source_kind
        part["pseudo_cutoff_index"] = int(cutoff_index)
        parts.append(part)

        eval_indices = frame.eval_indices[selected].astype(int)
        target_values = df.loc[eval_indices, target_column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "well_id": well_id,
                "cutoff_index": int(cutoff_index),
                "original_last_known_index": original_last_known,
                "online_rows": int(len(part)),
                "online_eval_step_min": int(part["eval_step"].min()),
                "online_eval_step_max": int(part["eval_step"].max()),
                "online_target_mean": round(float(np.nanmean(target_values)), 6),
                "online_target_std": round(float(np.nanstd(target_values)), 6),
            }
        )

    if not parts:
        return None, summary_rows
    return pd.concat(parts, ignore_index=True), summary_rows


def collect_online_rows(
    *,
    valid_paths: list[Path],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for path in valid_paths:
        part, summary_rows = online_rows_from_well(path=path, config=config, rng=rng)
        rows.extend(summary_rows)
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return None, rows

    online = pd.concat(parts, ignore_index=True)
    max_rows_per_fold = int(get_nested(config, "audit.online_training.max_rows_per_fold", 45000))
    if max_rows_per_fold > 0 and len(online) > max_rows_per_fold:
        selected = rng.choice(len(online), size=max_rows_per_fold, replace=False)
        online = online.iloc[selected].reset_index(drop=True)
    return online, rows


def base_training_weights(train: pd.DataFrame, variant: dict[str, Any]) -> np.ndarray | None:
    weights = sample_weights(train, variant)
    if str(variant.get("weight_profile", "uniform")) == "uniform":
        return None
    return weights


def combined_training_weights(
    *,
    base_train: pd.DataFrame,
    online_train: pd.DataFrame,
    variant: dict[str, Any],
    online_weight: float,
) -> np.ndarray:
    base_weights = base_training_weights(base_train, variant)
    if base_weights is None:
        base_weights = np.ones(len(base_train), dtype=float)
    online_weights = np.full(len(online_train), float(online_weight), dtype=float)
    weights = np.concatenate([base_weights, online_weights])
    mean_weight = float(np.mean(weights)) if weights.size else 1.0
    if np.isfinite(mean_weight) and mean_weight > 0.0:
        weights = weights / mean_weight
    return weights


def online_weight_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = list(get_nested(config, "audit.online_training.weights", []))
    if not raw:
        return [{"name": "online_weight_0_10", "online_weight": 0.10}]
    return [
        {"name": str(item["name"]), "online_weight": float(item["online_weight"])} for item in raw
    ]


def predict_valid_well(
    *,
    path: Path,
    model: Any,
    config: dict[str, Any],
    buckets: list[Bucket],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_column = str(config_get(config, "data.target_column", "TVT"))
    df = pd.read_csv(path)
    frame = build_drift_feature_frame(df, config, include_target=True)
    if frame.target_residual is None or frame.eval_indices.size == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=np.int16),
        )
    raw = predict_drift(frame, model, config)
    anchor = frame.baseline_prediction.astype(float)
    eval_step = frame.features["eval_step"].to_numpy(dtype=float)
    pred = exp026_bucket_shrink(raw=raw, anchor=anchor, eval_step=eval_step, config=config)
    y_true = df.loc[frame.eval_indices, target_column].to_numpy(dtype=float)
    return pred, y_true, raw, bucket_codes(eval_step, buckets)


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
            score = rmse_from_sse(candidate.sse - holdout_sse, candidate.n - holdout_n)
            train_scores[candidate_name] = score if np.isfinite(score) else float("inf")
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
    weight_candidates = online_weight_candidates(config)

    candidate_methods = {
        "raw_pseudo_tail": "raw",
        "exp026_bucket_shrink_control": "fixed_exp014_bucket_shrink",
    }
    for candidate in weight_candidates:
        candidate_methods[str(candidate["name"])] = "test_time_prefix_online_training"

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
    online_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(files, groups=groups)):
        train_paths = [files[index] for index in train_idx]
        valid_paths = [files[index] for index in valid_idx]
        rng = np.random.default_rng(seed + fold * 1009 + variant_index)
        base_train = collect_training_rows(
            train_paths,
            config,
            rng,
            variant,
            max_rows_per_well=max_rows_per_well,
            max_rows_total=max_rows_total,
        )
        source_rows.extend(source_summary_rows(base_train, variant=variant_name, fold=fold))
        base_weights = base_training_weights(base_train, variant)
        print(f"Fold {fold}: fitting {variant_name} control on {len(base_train):,} sampled rows")
        base_model = fit_model(base_train, config, seed=seed + fold, weights=base_weights)
        importance_rows.extend(
            feature_importance_rows(
                base_model,
                config,
                variant=variant_name,
                fold=fold,
                segment="control",
            )
        )

        online_rng = np.random.default_rng(seed + fold * 2003 + variant_index)
        online_train, fold_online_rows = collect_online_rows(
            valid_paths=valid_paths,
            config=config,
            rng=online_rng,
        )
        for row in fold_online_rows:
            row["fold"] = fold
        online_rows.extend(fold_online_rows)
        online_models: dict[str, Any] = {}
        if online_train is not None and not online_train.empty:
            for candidate in weight_candidates:
                candidate_name = str(candidate["name"])
                online_weight = float(candidate["online_weight"])
                combined = pd.concat([base_train, online_train], ignore_index=True)
                combined_weights = combined_training_weights(
                    base_train=base_train,
                    online_train=online_train,
                    variant=variant,
                    online_weight=online_weight,
                )
                print(
                    f"Fold {fold}: fitting {candidate_name} with "
                    f"{len(online_train):,} online rows at weight={online_weight:g}"
                )
                online_models[candidate_name] = fit_model(
                    combined,
                    config,
                    seed=seed + fold + 10000 + int(round(online_weight * 1000)),
                    weights=combined_weights,
                )
                importance_rows.extend(
                    feature_importance_rows(
                        online_models[candidate_name],
                        config,
                        variant=variant_name,
                        fold=fold,
                        segment=candidate_name,
                    )
                )
        else:
            print(f"Fold {fold}: no usable online rows; online candidates fall back to control")

        fold_sse = 0.0
        fold_n = 0
        for path in valid_paths:
            well_id = well_id_from_path(path)
            well_fold = stable_fold(well_id, well_holdout_folds)
            control_pred, y_true, raw_pred, bucket_code_values = predict_valid_well(
                path=path,
                model=base_model,
                config=config,
                buckets=buckets,
            )
            if y_true.size == 0:
                continue
            add_stats(
                stats_by_name["raw_pseudo_tail"],
                pred=raw_pred,
                y_true=y_true,
                fold=fold,
                well_fold=well_fold,
                bucket_code_values=bucket_code_values,
                n_buckets=len(buckets),
            )
            add_stats(
                stats_by_name["exp026_bucket_shrink_control"],
                pred=control_pred,
                y_true=y_true,
                fold=fold,
                well_fold=well_fold,
                bucket_code_values=bucket_code_values,
                n_buckets=len(buckets),
            )
            for candidate in weight_candidates:
                candidate_name = str(candidate["name"])
                model = online_models.get(candidate_name)
                if model is None:
                    candidate_pred = control_pred
                else:
                    candidate_pred, _, _, _ = predict_valid_well(
                        path=path,
                        model=model,
                        config=config,
                        buckets=buckets,
                    )
                add_stats(
                    stats_by_name[candidate_name],
                    pred=candidate_pred,
                    y_true=y_true,
                    fold=fold,
                    well_fold=well_fold,
                    bucket_code_values=bucket_code_values,
                    n_buckets=len(buckets),
                )

            fold_sse += float(np.square(control_pred - y_true).sum())
            fold_n += int(y_true.size)
        fold_rows.append(
            {
                "fold": fold,
                "control_rmse": round(rmse_from_sse(fold_sse, fold_n), 6),
                "rows": fold_n,
                "base_train_rows": len(base_train),
                "online_rows": 0 if online_train is None else len(online_train),
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
        name="leave_one_original_fold_out_online_candidate_selection",
        stats_by_name=stats_by_name,
        candidate_names=candidate_names,
        control_name=control_name,
        attr_sse="fold_sse",
        attr_n="fold_n",
    )
    well_selection, well_selection_rows = selection_audit(
        name="well_hash_holdout_online_candidate_selection",
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

    online_candidate_names = [str(item["name"]) for item in weight_candidates]
    best_fixed = min(
        [row for row in metric_rows if row["candidate"] in [control_name, *online_candidate_names]],
        key=lambda row: row["rmse"],
    )
    clean_supported = (
        original_selection["rmse"] < control_rmse and well_selection["rmse"] < control_rmse
    )
    selected_method = (
        "heldout_prefix_online_training_selection" if clean_supported else control_name
    )
    selected_cv = float(original_selection["rmse"] if clean_supported else control_rmse)

    summary = {
        "experiment": "exp037_test_time_prefix_online_training_audit",
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
        "clean_prefix_online_training_supported": clean_supported,
        "selected_method": selected_method,
        "cv": round(selected_cv, 6),
        "public_lb": None,
        "metric": "rmse",
        "rules_risk": "organizer approval for test-time online training was not confirmed",
        "artifact_rows": {
            "prefix_online_training_candidate_metrics": len(metric_rows),
            "prefix_online_training_bucket_summary": len(bucket_rows),
            "prefix_online_training_fold_metrics": len(fold_metric_rows),
            "prefix_online_training_well_holdout_metrics": len(well_holdout_rows),
            "prefix_online_training_online_rows": len(online_rows),
            "prefix_online_training_selection": len(original_selection_rows + well_selection_rows),
            "pseudo_tail_source_summary": len(source_rows),
            "pseudo_tail_feature_importance": len(importance_rows),
        },
        "metrics": metric_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "prefix_online_training_candidate_metrics.csv", index=False
    )
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "prefix_online_training_bucket_summary.csv", index=False
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        output_dir / "prefix_online_training_fold_metrics.csv", index=False
    )
    pd.DataFrame(well_holdout_rows).to_csv(
        output_dir / "prefix_online_training_well_holdout_metrics.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "prefix_online_training_fold_summary.csv", index=False
    )
    pd.DataFrame(online_rows).to_csv(
        output_dir / "prefix_online_training_online_rows.csv", index=False
    )
    pd.DataFrame(original_selection_rows + well_selection_rows).to_csv(
        output_dir / "prefix_online_training_selection.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(output_dir / "pseudo_tail_source_summary.csv", index=False)
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(
            output_dir / "pseudo_tail_feature_importance.csv", index=False
        )
    (output_dir / "prefix_online_training_summary.json").write_text(
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
