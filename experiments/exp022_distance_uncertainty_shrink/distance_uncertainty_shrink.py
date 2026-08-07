from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from baseline import (
    HORIZONTAL_SUFFIX,
    active_feature_columns,
    build_drift_feature_frame,
    config_get,
    distance_bucket_alphas,
    make_drift_model,
    optional_positive_int,
    predict_drift,
    well_id_from_path,
)
from baseline import (
    postprocess_predictions as baseline_postprocess_predictions,
)
from settings import ExperimentPaths
from sklearn.model_selection import GroupKFold

EXPERIMENT_NAME = "exp022_distance_uncertainty_shrink"


@dataclass
class MetricBucket:
    sse: float = 0.0
    n: int = 0
    wells: set[str] = field(default_factory=set)

    def add(self, pred: np.ndarray, true: np.ndarray, well_ids: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=float)
        true = np.asarray(true, dtype=float)
        mask = np.isfinite(pred) & np.isfinite(true)
        if not bool(mask.any()):
            return
        diff = pred[mask] - true[mask]
        self.sse += float(np.square(diff).sum())
        self.n += int(mask.sum())
        self.wells.update(str(value) for value in np.asarray(well_ids, dtype=object)[mask])

    @property
    def rmse(self) -> float:
        return rmse_from_sse(self.sse, self.n)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def rmse_from_sse(sse: float, n_rows: int | float) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(max(0.0, float(sse)) / float(n_rows))


def finite_float(value: float, digits: int = 6) -> float | None:
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def experiment_config() -> dict[str, Any]:
    return load_yaml(Path(__file__).with_name("config.yaml"))


def train_files(paths: ExperimentPaths, *, max_wells: int | None = None) -> list[Path]:
    files = sorted(paths.train_data_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if max_wells is not None:
        files = files[:max_wells]
    if not files:
        raise ValueError(f"No train horizontal well CSVs found in {paths.train_data_dir}")
    return files


def test_files(paths: ExperimentPaths) -> list[Path]:
    files = sorted(paths.test_data_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not files:
        raise ValueError(f"No test horizontal well CSVs found in {paths.test_data_dir}")
    return files


def selected_training_variant(config: dict[str, Any]) -> dict[str, Any]:
    selected_name = str(
        config_get(config, "audit.training_variants.selected_variant", "near_down_far_up_lightgbm")
    )
    variants = config_get(config, "audit.training_variants.variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("audit.training_variants.variants must be a non-empty list")
    for variant in variants:
        if isinstance(variant, dict) and str(variant.get("name")) == selected_name:
            return variant
    raise ValueError(f"selected training variant not found: {selected_name}")


def sample_weights(frame: pd.DataFrame, variant: dict[str, Any]) -> np.ndarray:
    profile = str(variant.get("weight_profile", "uniform"))
    eval_step = frame["eval_step"].to_numpy(dtype=float)
    weights = np.ones(eval_step.shape, dtype=float)

    if profile in {"near_downweight", "near_down_far_up"}:
        previous_max = -np.inf
        for rule in variant.get("weights", []):
            max_step = float(rule["max_step"])
            mask = (eval_step > previous_max) & (eval_step <= max_step)
            weights[mask] = float(rule["weight"])
            previous_max = max_step

    if profile in {"far_upweight", "near_down_far_up"}:
        c_value = float(variant.get("far_upweight_c", 0.5))
        reference = max(1.0, float(variant.get("reference_step", 2500.0)))
        far_multiplier = 1.0 + c_value * np.log1p(np.maximum(eval_step, 0.0)) / np.log1p(reference)
        weights *= np.minimum(far_multiplier, float(variant.get("max_weight", 2.0)))

    mean_weight = float(np.mean(weights)) if weights.size else 1.0
    if np.isfinite(mean_weight) and mean_weight > 0:
        weights = weights / mean_weight
    return weights


def collect_training_rows(
    files: list[Path],
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    max_rows_per_well: int | None,
    max_rows_total: int | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        frame = build_drift_feature_frame(df, config, include_target=True)
        if frame.target_residual is None:
            raise ValueError("target_residual is required")
        valid = np.flatnonzero(np.isfinite(frame.target_residual))
        if max_rows_per_well is not None and valid.size > max_rows_per_well:
            valid = rng.choice(valid, size=max_rows_per_well, replace=False)
        if valid.size == 0:
            continue
        part = frame.features.iloc[valid].copy()
        part["target_residual"] = frame.target_residual[valid]
        part["well_id"] = well_id_from_path(path)
        parts.append(part)

    if not parts:
        raise ValueError("no finite weighted training rows were collected")
    train = pd.concat(parts, ignore_index=True)
    if max_rows_total is not None and len(train) > max_rows_total:
        selected = rng.choice(len(train), size=max_rows_total, replace=False)
        train = train.iloc[selected].reset_index(drop=True)
    return train


def fit_weighted_model(
    train: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
    *,
    seed: int,
) -> Any:
    columns = active_feature_columns(config)
    model = make_drift_model(config, random_state=seed)
    weights = None
    if str(variant.get("weight_profile", "uniform")) != "uniform":
        weights = sample_weights(train, variant)
    model.fit(
        train[columns],
        train["target_residual"].to_numpy(dtype=float),
        sample_weight=weights,
    )
    return model


def fit_weighted_model_from_files(
    files: list[Path],
    config: dict[str, Any],
    variant: dict[str, Any],
    *,
    seed: int,
    max_rows_total: int | None,
    max_rows_per_well: int | None,
) -> tuple[Any, int]:
    train = collect_training_rows(
        files,
        config,
        np.random.default_rng(seed),
        max_rows_per_well=max_rows_per_well,
        max_rows_total=max_rows_total,
    )
    return fit_weighted_model(train, config, variant, seed=seed), int(len(train))


def bucket_labels(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    labels = np.full(eval_step.shape, str(buckets[-1]["name"]), dtype=object)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        labels[mask] = str(bucket["name"])
        previous_max = max_step
    return labels


def candidate_methods(config: dict[str, Any]) -> list[str]:
    raw = config_get(config, "postprocess.candidate_methods", ["raw", "distance_bucket_shrink"])
    if not isinstance(raw, list) or not raw:
        raise ValueError("postprocess.candidate_methods must be a non-empty list")
    return [str(method) for method in raw]


def candidate_name(method: str) -> str:
    if method in {"raw", "raw_lightgbm_no_gr"}:
        return "weighted_raw"
    return f"weighted_{method}"


def normalized_log(values: np.ndarray, reference: float) -> np.ndarray:
    reference = max(1.0, float(reference))
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    return np.clip(np.log1p(np.maximum(values, 0.0)) / math.log1p(reference), 0.0, 1.5)


def uncertainty_score(
    raw_pred: np.ndarray,
    frame: Any,
    params: dict[str, Any],
) -> np.ndarray:
    features = frame.features
    eval_step = features["eval_step"].to_numpy(dtype=float)
    eval_row_count = np.maximum(features["eval_row_count"].to_numpy(dtype=float), 1.0)
    anchor = np.asarray(frame.baseline_prediction, dtype=float)
    residual_abs = np.abs(np.asarray(raw_pred, dtype=float) - anchor)
    gr_values = features["gr"].to_numpy(dtype=float)
    delta_z_abs = np.abs(features["delta_z"].to_numpy(dtype=float))

    raw_weights = params.get("weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError("uncertainty shrink weights must be a mapping")
    weights = {str(key): float(value) for key, value in raw_weights.items()}
    total_weight = sum(max(0.0, value) for value in weights.values())
    if total_weight <= 0.0:
        raise ValueError("uncertainty shrink weights must contain a positive value")

    tail_progress = np.clip(eval_step / np.maximum(eval_row_count - 1.0, 1.0), 0.0, 1.0)
    components = {
        "distance": normalized_log(eval_step, float(params.get("reference_step", 2500.0))),
        "tail_progress": tail_progress,
        "gr_missing": (~np.isfinite(gr_values)).astype(float),
        "z_span": normalized_log(delta_z_abs, float(params.get("z_reference", 800.0))),
        "raw_residual_abs": normalized_log(
            residual_abs,
            float(params.get("residual_reference", 40.0)),
        ),
    }

    score = np.zeros(eval_step.shape, dtype=float)
    for name, weight in weights.items():
        if name not in components:
            raise ValueError(f"unknown uncertainty component: {name}")
        score += max(0.0, weight) * components[name]
    return np.clip(score / total_weight, 0.0, 1.5)


def uncertainty_shrink_predictions(
    raw_pred: np.ndarray,
    frame: Any,
    config: dict[str, Any],
    *,
    params: dict[str, Any],
) -> np.ndarray:
    raw_pred = np.asarray(raw_pred, dtype=float)
    if raw_pred.size == 0:
        return raw_pred.copy()

    anchor = np.asarray(frame.baseline_prediction, dtype=float)
    residual = raw_pred - anchor
    eval_step = frame.features["eval_step"].to_numpy(dtype=float)
    base_method = str(params.get("base_method", "raw"))
    if base_method == "distance_bucket_shrink":
        bucket_params = config_get(config, "postprocess.methods.distance_bucket_shrink", {})
        buckets = bucket_params.get("buckets", []) if isinstance(bucket_params, dict) else []
        if not isinstance(buckets, list):
            raise ValueError("postprocess distance_bucket_shrink buckets must be a list")
        base_alpha = distance_bucket_alphas(eval_step, buckets)
    elif base_method == "raw":
        base_alpha = np.ones(raw_pred.shape, dtype=float)
    else:
        raise ValueError(f"unsupported uncertainty base_method: {base_method}")

    score = uncertainty_score(raw_pred, frame, params)
    shrink_floor = float(params.get("shrink_floor", 0.65))
    strength = float(params.get("shrink_strength", 0.75))
    shrink = 1.0 / (1.0 + strength * score)
    shrink = np.maximum(shrink, shrink_floor)
    alpha = np.clip(
        base_alpha * shrink,
        float(params.get("min_alpha", 0.10)),
        float(params.get("max_alpha", 1.15)),
    )
    return anchor + alpha * residual


def apply_postprocess_predictions(
    raw_pred: np.ndarray,
    frame: Any,
    config: dict[str, Any],
    *,
    method: str | None = None,
) -> np.ndarray:
    method = method or str(config_get(config, "postprocess.selected_method", "raw"))
    params = config_get(config, f"postprocess.methods.{method}", {})
    if not isinstance(params, dict):
        raise ValueError(f"postprocess.methods.{method} must be a mapping")
    if method.startswith("uncertainty_shrink"):
        return uncertainty_shrink_predictions(raw_pred, frame, config, params=params)
    return baseline_postprocess_predictions(raw_pred, frame, config, method=method, params=params)


def predict_candidates(
    raw_pred: np.ndarray,
    frame: Any,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for method in candidate_methods(config):
        predictions[candidate_name(method)] = apply_postprocess_predictions(
            raw_pred,
            frame,
            config,
            method=method,
        )
    return predictions


def score_oof_candidates(
    oof: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    buckets = list(config_get(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")

    y_true = oof["y_true"].to_numpy(dtype=float)
    well_ids = oof["well_id"].to_numpy(dtype=object)
    labels = bucket_labels(oof["eval_step"].to_numpy(dtype=float), buckets)
    metric_buckets: dict[tuple[str, str], MetricBucket] = defaultdict(MetricBucket)

    prediction_columns = [column for column in oof.columns if column.startswith("y_pred_")]
    for column in prediction_columns:
        candidate = column.removeprefix("y_pred_")
        pred = oof[column].to_numpy(dtype=float)
        metric_buckets[("overall", candidate)].add(pred, y_true, well_ids)
        for bucket_name in sorted(set(labels)):
            mask = labels == bucket_name
            metric_buckets[(bucket_name, candidate)].add(pred[mask], y_true[mask], well_ids[mask])

    raw_by_segment = {
        segment: bucket.rmse
        for (segment, candidate), bucket in metric_buckets.items()
        if candidate == "weighted_raw"
    }
    rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    for (segment, candidate), bucket in sorted(metric_buckets.items()):
        raw_rmse = raw_by_segment.get(segment, float("nan"))
        row = {
            "segment": segment,
            "candidate": candidate,
            "rmse": finite_float(bucket.rmse),
            "weighted_raw_rmse": finite_float(raw_rmse),
            "delta_vs_weighted_raw": finite_float(bucket.rmse - raw_rmse),
            "rows": bucket.n,
            "wells": len(bucket.wells),
        }
        rows.append(row)
        if segment != "overall":
            distance_rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "weighted_postprocess_metrics.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(
        output_dir / "weighted_postprocess_distance_metrics.csv", index=False
    )
    overall = {
        row["candidate"]: row["rmse"]
        for row in rows
        if row["segment"] == "overall" and row["rmse"] is not None
    }
    return rows, distance_rows, overall


def run_weighted_oof_cv(
    files: list[Path],
    config: dict[str, Any],
    output_dir: Path,
    *,
    max_wells: int | None = None,
) -> dict[str, Any]:
    if max_wells is not None:
        files = files[:max_wells]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config_get(config, "validation.seed", 42))
    n_folds = int(config_get(config, "validation.n_folds", 5))
    max_rows_per_well = optional_positive_int(
        config_get(config, "model.training.max_train_rows_per_well", 800)
    )
    max_rows_total = optional_positive_int(
        config_get(config, "model.training.max_train_rows_per_fold", 300000)
    )
    variant = selected_training_variant(config)
    groups = np.asarray([well_id_from_path(path) for path in files])
    splitter = GroupKFold(n_splits=n_folds)

    oof_parts: list[pd.DataFrame] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(files, groups=groups)):
        train_paths = [files[index] for index in train_idx]
        valid_paths = [files[index] for index in valid_idx]
        train = collect_training_rows(
            train_paths,
            config,
            np.random.default_rng(seed + fold),
            max_rows_per_well=max_rows_per_well,
            max_rows_total=max_rows_total,
        )
        model = fit_weighted_model(train, config, variant, seed=seed + fold)
        print(
            f"Fold {fold}: fitted {variant['name']} on {len(train):,} rows; "
            f"valid wells={len(valid_paths)}"
        )

        for path in valid_paths:
            well_id = well_id_from_path(path)
            df = pd.read_csv(path)
            frame = build_drift_feature_frame(df, config, include_target=True)
            if frame.target_residual is None or frame.eval_indices.size == 0:
                continue
            raw_pred = predict_drift(frame, model, config)
            candidates = predict_candidates(raw_pred, frame, config)
            part = pd.DataFrame(
                {
                    "fold": fold,
                    "well_id": well_id,
                    "row_index": frame.eval_indices.astype(int),
                    "eval_step": frame.features["eval_step"].to_numpy(dtype=float),
                    "eval_progress": frame.features["eval_progress"].to_numpy(dtype=float),
                    "eval_row_count": frame.features["eval_row_count"].to_numpy(dtype=float),
                    "known_row_count": frame.features["known_row_count"].to_numpy(dtype=float),
                    "delta_z_abs": np.abs(frame.features["delta_z"].to_numpy(dtype=float)),
                    "delta_xyz": frame.features["delta_xyz"].to_numpy(dtype=float),
                    "gr_missing": (
                        ~np.isfinite(frame.features["gr"].to_numpy(dtype=float))
                    ).astype(int),
                    "last_anchor": frame.baseline_prediction.astype(float),
                    "raw_residual_abs": np.abs(
                        raw_pred.astype(float) - frame.baseline_prediction.astype(float)
                    ),
                    "y_true": df.loc[
                        frame.eval_indices, str(config_get(config, "data.target_column", "TVT"))
                    ].to_numpy(dtype=float),
                }
            )
            for name, values in candidates.items():
                part[f"y_pred_{name}"] = values.astype(float)
            oof_parts.append(part)

    if not oof_parts:
        raise ValueError("no weighted OOF predictions were generated")
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "weighted_oof_predictions.csv", index=False)
    metric_rows, distance_rows, overall = score_oof_candidates(oof, config, output_dir)
    best_candidate = min(overall, key=overall.get) if overall else None
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "oof_cv_completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "selected_training_variant": str(variant["name"]),
        "selected_postprocess_method": str(
            config_get(config, "postprocess.selected_method", "raw")
        ),
        "parent_best_cv": config_get(config, "audit.parent_best_cv", None),
        "weighted_postprocess_overall": overall,
        "best_candidate": best_candidate,
        "best_candidate_cv": overall.get(best_candidate) if best_candidate else None,
        "artifact_rows": {
            "weighted_oof_predictions": int(len(oof)),
            "weighted_postprocess_metrics": len(metric_rows),
            "weighted_postprocess_distance_metrics": len(distance_rows),
        },
    }
    (output_dir / "distance_uncertainty_shrink_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def write_metrics_from_summary(paths: ExperimentPaths, summary: dict[str, Any]) -> None:
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "updated_at": summary["updated_at"],
        "cv": summary.get("best_candidate_cv"),
        "public_lb": None,
        "parent_best_cv": summary.get("parent_best_cv"),
        "selected_training_variant": summary.get("selected_training_variant"),
        "selected_postprocess_method": summary.get("selected_postprocess_method"),
        "weighted_postprocess_overall": summary.get("weighted_postprocess_overall", {}),
        "best_candidate": summary.get("best_candidate"),
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def generate_weighted_submission(
    paths: ExperimentPaths,
    config: dict[str, Any],
    *,
    max_wells: int | None = None,
) -> dict[str, Any]:
    variant = selected_training_variant(config)
    seed = int(config_get(config, "validation.seed", 42))
    max_rows_per_well = optional_positive_int(
        config_get(config, "model.training.max_train_rows_per_well", 800)
    )
    max_rows_total = optional_positive_int(
        config_get(config, "model.training.max_train_rows_final", 450000)
    )
    model_files = train_files(paths, max_wells=max_wells)
    model, n_train_rows = fit_weighted_model_from_files(
        model_files,
        config,
        variant,
        seed=seed,
        max_rows_total=max_rows_total,
        max_rows_per_well=max_rows_per_well,
    )

    selected_method = str(config_get(config, "postprocess.selected_method", "raw"))
    predictions: dict[str, float] = {}
    well_summaries: list[dict[str, object]] = []
    for path in test_files(paths):
        well_id = well_id_from_path(path)
        df = pd.read_csv(path)
        frame = build_drift_feature_frame(df, config, include_target=False)
        raw_pred = predict_drift(frame, model, config)
        y_pred = apply_postprocess_predictions(raw_pred, frame, config, method=selected_method)
        for row_index, value in zip(frame.eval_indices, y_pred, strict=True):
            predictions[f"{well_id}_{int(row_index)}"] = float(value)
        well_summaries.append(
            {
                "well_id": well_id,
                "n_rows": int(len(df)),
                "n_eval": int(frame.eval_indices.size),
                "last_known_index": int(frame.last_known_index),
                "last_known_tvt": float(frame.last_known_tvt),
                "recent_slope": float(frame.recent_slope),
                "training_variant": str(variant["name"]),
                "postprocess": selected_method,
            }
        )

    sample_submission = pd.read_csv(paths.sample_submission_path)
    id_column = str(config_get(config, "data.id_column", "id"))
    target_column = str(config_get(config, "data.submission_target_column", "tvt"))
    missing_ids = sorted(set(sample_submission[id_column]) - set(predictions))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"missing predictions for {len(missing_ids)} sample ids: {preview}")

    output = sample_submission.copy()
    output[target_column] = output[id_column].map(predictions).astype(float)
    output.to_csv(paths.submission_path, index=False)
    pd.DataFrame(well_summaries).to_csv(
        paths.artifacts_dir / "weighted_inference_well_summaries.csv", index=False
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "submission_path": str(paths.submission_path),
        "predicted_rows": int(len(predictions)),
        "train_wells": int(len(model_files)),
        "train_rows": int(n_train_rows),
        "training_variant": str(variant["name"]),
        "postprocess": selected_method,
    }
    (paths.artifacts_dir / "weighted_inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
