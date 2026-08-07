from __future__ import annotations

import argparse
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
    predict_drift,
    predict_from_prefix,
    well_id_from_path,
)
from settings import ExperimentPaths
from sklearn.model_selection import GroupKFold

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
RAW_USECOLS = [
    "variant",
    "fold",
    "well_id",
    "row_index",
    "eval_step",
    "eval_row_count",
    "last_anchor",
    "y_true",
    "y_pred",
]


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


@dataclass
class ResidualBucket:
    n: int = 0
    wells: set[str] = field(default_factory=set)
    error_sum: float = 0.0
    error_sq_sum: float = 0.0
    pred_residual_sum: float = 0.0
    pred_residual_sq_sum: float = 0.0
    true_residual_sum: float = 0.0
    true_residual_sq_sum: float = 0.0

    def add(
        self,
        *,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        last_anchor: np.ndarray,
        well_ids: np.ndarray,
    ) -> None:
        mask = np.isfinite(y_pred) & np.isfinite(y_true) & np.isfinite(last_anchor)
        if not bool(mask.any()):
            return
        error = y_pred[mask] - y_true[mask]
        pred_residual = y_pred[mask] - last_anchor[mask]
        true_residual = y_true[mask] - last_anchor[mask]
        self.n += int(mask.sum())
        self.wells.update(str(value) for value in np.asarray(well_ids, dtype=object)[mask])
        self.error_sum += float(error.sum())
        self.error_sq_sum += float(np.square(error).sum())
        self.pred_residual_sum += float(pred_residual.sum())
        self.pred_residual_sq_sum += float(np.square(pred_residual).sum())
        self.true_residual_sum += float(true_residual.sum())
        self.true_residual_sq_sum += float(np.square(true_residual).sum())

    def row(self, bucket: str) -> dict[str, Any]:
        error_mean = safe_mean(self.error_sum, self.n)
        pred_mean = safe_mean(self.pred_residual_sum, self.n)
        true_mean = safe_mean(self.true_residual_sum, self.n)
        return {
            "bucket": bucket,
            "rows": self.n,
            "wells": len(self.wells),
            "raw_rmse": round(rmse_from_sse(self.error_sq_sum, self.n), 6),
            "raw_bias": round(error_mean, 6),
            "raw_error_std": round(safe_std(self.error_sq_sum, self.error_sum, self.n), 6),
            "pred_residual_mean": round(pred_mean, 6),
            "pred_residual_std": round(
                safe_std(self.pred_residual_sq_sum, self.pred_residual_sum, self.n), 6
            ),
            "true_residual_mean": round(true_mean, 6),
            "true_residual_std": round(
                safe_std(self.true_residual_sq_sum, self.true_residual_sum, self.n), 6
            ),
            "pred_minus_true_residual_mean": round(pred_mean - true_mean, 6),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pseudo-tail residual augmentation.")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke limit")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only run the existing OOF distance audit.",
    )
    args, _ = parser.parse_known_args()
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def rmse_from_sse(sse: float, n_rows: int | float) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(max(0.0, float(sse)) / float(n_rows))


def safe_mean(total: float, n_rows: int) -> float:
    return float(total) / float(n_rows) if n_rows > 0 else float("nan")


def safe_std(square_total: float, total: float, n_rows: int) -> float:
    if n_rows <= 0:
        return float("nan")
    mean = float(total) / float(n_rows)
    variance = max(0.0, float(square_total) / float(n_rows) - mean * mean)
    return math.sqrt(variance)


def resolve_existing_path(
    path_value: str | Path,
    *,
    required_name: str | None = None,
    preferred_substring: str | None = None,
) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(required_name or path.name))
        if preferred_substring:
            preferred = [item for item in matches if preferred_substring in str(item)]
            if preferred:
                return preferred[0]
        if matches:
            return matches[0]
    return path


def load_raw_oof(config: dict[str, Any]) -> pd.DataFrame:
    raw_path = resolve_existing_path(
        str(get_nested(config, "audit.raw_oof_path")),
        required_name="row_oof_predictions.csv",
        preferred_substring="exp013-model-diversity-or-postprocess-train",
    )
    raw_variant = str(get_nested(config, "audit.raw_oof_variant", "lightgbm_no_gr"))
    chunk_rows = int(get_nested(config, "audit.chunk_rows", 250000))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_path, usecols=RAW_USECOLS, chunksize=chunk_rows):
        chunk = chunk[chunk["variant"] == raw_variant]
        if chunk.empty:
            continue
        frames.append(chunk.drop(columns=["variant"]))
    if not frames:
        raise ValueError(f"No raw OOF rows for variant={raw_variant!r} in {raw_path}")
    frame = pd.concat(frames, ignore_index=True)
    frame["row_index"] = frame["row_index"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame.sort_values(["well_id", "row_index"]).reset_index(drop=True)


def train_files(paths: ExperimentPaths, max_wells: int | None) -> list[Path]:
    files = sorted(paths.train_data_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if max_wells is not None:
        files = files[:max_wells]
    if not files:
        raise ValueError(f"No train horizontal well CSVs found in {paths.train_data_dir}")
    return files


def add_recent_linear(oof: pd.DataFrame, files: list[Path], config: dict[str, Any]) -> pd.DataFrame:
    recent_parts: list[pd.DataFrame] = []
    for path in files:
        well_id = well_id_from_path(path)
        df = pd.read_csv(path)
        prefix = predict_from_prefix(df, config)
        recent = prefix.predictions.get("recent_linear")
        if recent is None:
            continue
        recent_parts.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "row_index": prefix.eval_indices.astype(int),
                    "recent_linear": recent.astype(float),
                }
            )
        )
    if not recent_parts:
        oof["recent_linear"] = np.nan
        return oof
    recent_frame = pd.concat(recent_parts, ignore_index=True)
    return oof.merge(recent_frame, on=["well_id", "row_index"], how="left", validate="1:1")


def bucket_labels(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    labels = np.full(eval_step.shape, str(buckets[-1]["name"]), dtype=object)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        labels[mask] = str(bucket["name"])
        previous_max = max_step
    return labels


def candidate_predictions(oof: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = oof["y_pred"].to_numpy(dtype=float)
    anchor = oof["last_anchor"].to_numpy(dtype=float)
    candidates = {
        "raw_lightgbm_no_gr": raw,
        "last_anchor": anchor,
        "recent_linear": oof["recent_linear"].to_numpy(dtype=float),
    }
    buckets = list(get_nested(config, "audit.postprocess.exp014_heldout_bucket_shrink", []))
    if buckets:
        alphas = distance_bucket_alphas(oof["eval_step"].to_numpy(dtype=float), buckets)
        candidates["exp014_bucket_shrink_params"] = anchor + alphas * (raw - anchor)
    return candidates


def audit_existing_oof(
    oof: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")

    candidates = candidate_predictions(oof, config)
    y_true = oof["y_true"].to_numpy(dtype=float)
    well_ids = oof["well_id"].to_numpy(dtype=object)
    labels = bucket_labels(oof["eval_step"].to_numpy(dtype=float), buckets)
    candidate_metrics: dict[tuple[str, str], MetricBucket] = defaultdict(MetricBucket)
    residual_buckets: dict[str, ResidualBucket] = defaultdict(ResidualBucket)

    for candidate_name, pred in candidates.items():
        candidate_metrics[("overall", candidate_name)].add(pred, y_true, well_ids)
        for bucket_name in sorted(set(labels)):
            mask = labels == bucket_name
            candidate_metrics[(bucket_name, candidate_name)].add(
                pred[mask], y_true[mask], well_ids[mask]
            )

    raw = candidates["raw_lightgbm_no_gr"]
    last_anchor = candidates["last_anchor"]
    residual_buckets["overall"].add(
        y_pred=raw, y_true=y_true, last_anchor=last_anchor, well_ids=well_ids
    )
    for bucket_name in sorted(set(labels)):
        mask = labels == bucket_name
        residual_buckets[bucket_name].add(
            y_pred=raw[mask],
            y_true=y_true[mask],
            last_anchor=last_anchor[mask],
            well_ids=well_ids[mask],
        )

    raw_by_segment = {
        segment: bucket.rmse
        for (segment, candidate), bucket in candidate_metrics.items()
        if candidate == "raw_lightgbm_no_gr"
    }
    metric_rows: list[dict[str, Any]] = []
    for (segment, candidate), bucket in sorted(candidate_metrics.items()):
        raw_rmse = raw_by_segment.get(segment, float("nan"))
        metric_rows.append(
            {
                "segment": segment,
                "candidate": candidate,
                "rmse": round(bucket.rmse, 6),
                "raw_rmse": round(raw_rmse, 6),
                "delta_vs_raw": round(bucket.rmse - raw_rmse, 6),
                "rows": bucket.n,
                "wells": len(bucket.wells),
            }
        )
    residual_rows = [bucket.row(name) for name, bucket in sorted(residual_buckets.items())]

    pd.DataFrame(metric_rows).to_csv(output_dir / "distance_candidate_metrics.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(
        output_dir / "distance_residual_bucket_summary.csv", index=False
    )

    overall = {row["candidate"]: row["rmse"] for row in metric_rows if row["segment"] == "overall"}
    return metric_rows, residual_rows, overall


def choose_pseudo_cutoffs(
    df: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> list[int]:
    cutoffs_per_well = int(variant.get("cutoffs_per_well", 0))
    if cutoffs_per_well <= 0:
        return []

    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    if known_indices.size == 0:
        return []

    original_last_known = int(known_indices[-1])
    min_prefix_rows = int(get_nested(config, "audit.pseudo_tail.min_prefix_rows", 200))
    min_pseudo_eval_rows = int(get_nested(config, "audit.pseudo_tail.min_pseudo_eval_rows", 250))
    min_cutoff = max(0, min_prefix_rows - 1)
    max_cutoff = original_last_known - min_pseudo_eval_rows
    if max_cutoff < min_cutoff:
        return []

    quantiles = variant.get("cutoff_quantiles")
    if not quantiles:
        quantiles = np.linspace(0.45, 0.85, cutoffs_per_well).tolist()
    quantiles = [float(value) for value in quantiles][:cutoffs_per_well]

    cutoffs: list[int] = []
    for quantile in quantiles:
        quantile = float(np.clip(quantile, 0.0, 1.0))
        cutoff = int(round(min_cutoff + quantile * (max_cutoff - min_cutoff)))
        cutoff = int(np.clip(cutoff, min_cutoff, max_cutoff))
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    return cutoffs


def with_pseudo_cutoff(df: pd.DataFrame, cutoff_index: int) -> pd.DataFrame:
    pseudo = df.copy()
    tvt_input = pseudo["TVT_input"].to_numpy(dtype=float)
    tvt_input[cutoff_index + 1 :] = np.nan
    pseudo["TVT_input"] = tvt_input
    return pseudo


def sample_frame_rows(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    max_rows: int | None,
) -> pd.DataFrame:
    if max_rows is not None and len(frame) > max_rows:
        selected = rng.choice(len(frame), size=max_rows, replace=False)
        return frame.iloc[selected].reset_index(drop=True)
    return frame.reset_index(drop=True)


def training_part_from_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    well_id: str,
    source_kind: str,
    cutoff_index: int | None,
    max_rows: int | None,
) -> pd.DataFrame | None:
    frame = build_drift_feature_frame(df, config, include_target=True)
    if frame.target_residual is None:
        raise ValueError("target_residual is required")
    valid = np.flatnonzero(np.isfinite(frame.target_residual))
    if valid.size == 0:
        return None
    part = frame.features.iloc[valid].copy()
    part["target_residual"] = frame.target_residual[valid]
    part["well_id"] = well_id
    part["source_kind"] = source_kind
    part["pseudo_cutoff_index"] = -1 if cutoff_index is None else int(cutoff_index)
    return sample_frame_rows(part, rng, max_rows=max_rows)


def distance_balance_rows(
    train: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
    rng: np.random.Generator,
    *,
    max_rows_total: int | None,
) -> pd.DataFrame:
    if not bool(variant.get("distance_balanced", False)):
        if max_rows_total is not None and len(train) > max_rows_total:
            selected = rng.choice(len(train), size=max_rows_total, replace=False)
            train = train.iloc[selected].reset_index(drop=True)
        return train

    buckets = list(get_nested(config, "audit.distance_buckets", []))
    labels = bucket_labels(train["eval_step"].to_numpy(dtype=float), buckets)
    bucket_cap = int(variant.get("balanced_rows_per_bucket", 0))
    if bucket_cap <= 0:
        bucket_cap = max(1, int((max_rows_total or len(train)) / max(1, len(buckets))))

    selected_parts: list[pd.DataFrame] = []
    for bucket_name in [str(bucket["name"]) for bucket in buckets]:
        indices = np.flatnonzero(labels == bucket_name)
        if indices.size == 0:
            continue
        if indices.size > bucket_cap:
            indices = rng.choice(indices, size=bucket_cap, replace=False)
        selected_parts.append(train.iloc[indices])

    if not selected_parts:
        return train.iloc[[]].reset_index(drop=True)
    balanced = pd.concat(selected_parts, ignore_index=True)
    if max_rows_total is not None and len(balanced) > max_rows_total:
        selected = rng.choice(len(balanced), size=max_rows_total, replace=False)
        balanced = balanced.iloc[selected].reset_index(drop=True)
    order = rng.permutation(len(balanced))
    return balanced.iloc[order].reset_index(drop=True)


def source_summary_rows(train: pd.DataFrame, *, variant: str, fold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if train.empty:
        return rows
    for (source_kind, cutoff_index), part in train.groupby(
        ["source_kind", "pseudo_cutoff_index"], dropna=False
    ):
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "source_kind": str(source_kind),
                "pseudo_cutoff_index": int(cutoff_index),
                "rows": int(len(part)),
                "wells": int(part["well_id"].nunique()),
                "eval_step_mean": round(float(part["eval_step"].mean()), 6),
                "eval_step_max": round(float(part["eval_step"].max()), 6),
            }
        )
    return rows


def collect_training_rows(
    files: list[Path],
    config: dict[str, Any],
    rng: np.random.Generator,
    variant: dict[str, Any],
    *,
    max_rows_per_well: int | None,
    max_rows_total: int | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        well_id = well_id_from_path(path)
        original = training_part_from_frame(
            df,
            config,
            rng,
            well_id=well_id,
            source_kind="original_tail",
            cutoff_index=None,
            max_rows=max_rows_per_well,
        )
        if original is not None:
            parts.append(original)

        pseudo_max_rows = variant.get("max_rows_per_pseudo_tail")
        pseudo_max_rows = int(pseudo_max_rows) if pseudo_max_rows is not None else max_rows_per_well
        for cutoff_index in choose_pseudo_cutoffs(df, config, variant):
            pseudo_df = with_pseudo_cutoff(df, cutoff_index)
            pseudo = training_part_from_frame(
                pseudo_df,
                config,
                rng,
                well_id=well_id,
                source_kind="pseudo_tail",
                cutoff_index=cutoff_index,
                max_rows=pseudo_max_rows,
            )
            if pseudo is not None:
                parts.append(pseudo)
    if not parts:
        raise ValueError("no finite training rows were collected")
    train = pd.concat(parts, ignore_index=True)
    return distance_balance_rows(
        train,
        config,
        variant,
        rng,
        max_rows_total=max_rows_total,
    )


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


def fit_model(
    train: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    weights: np.ndarray | None = None,
) -> Any:
    columns = active_feature_columns(config)
    model = make_drift_model(config, random_state=seed)
    if weights is None:
        model.fit(train[columns], train["target_residual"].to_numpy(dtype=float))
    else:
        model.fit(
            train[columns],
            train["target_residual"].to_numpy(dtype=float),
            sample_weight=weights,
        )
    return model


def feature_importance_rows(
    model: Any,
    config: dict[str, Any],
    *,
    variant: str,
    fold: int,
    segment: str,
) -> list[dict[str, Any]]:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return []
    return [
        {
            "variant": variant,
            "fold": fold,
            "segment": segment,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in zip(active_feature_columns(config), importance, strict=False)
    ]


def segment_masks(eval_step: np.ndarray, segments: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    previous_max = -np.inf
    for segment in segments:
        max_step = float(segment["max_step"])
        name = str(segment["name"])
        masks[name] = (eval_step > previous_max) & (eval_step <= max_step)
        previous_max = max_step
    return masks


def predict_valid_files(
    files: list[Path],
    config: dict[str, Any],
    model: Any,
    *,
    variant_name: str,
    fold: int,
    bucket_config: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int]:
    metric_buckets: dict[str, MetricBucket] = defaultdict(MetricBucket)
    for path in files:
        df = pd.read_csv(path)
        frame = build_drift_feature_frame(df, config, include_target=True)
        if frame.target_residual is None or frame.eval_indices.size == 0:
            continue
        pred = predict_drift(frame, model, config)
        y_true = df.loc[frame.eval_indices, str(config_get(config, "data.target_column", "TVT"))]
        y_true = y_true.to_numpy(dtype=float)
        well_ids = np.full(y_true.shape, well_id_from_path(path), dtype=object)
        labels = bucket_labels(frame.features["eval_step"].to_numpy(dtype=float), bucket_config)
        metric_buckets["overall"].add(pred, y_true, well_ids)
        for bucket_name in sorted(set(labels)):
            mask = labels == bucket_name
            metric_buckets[bucket_name].add(pred[mask], y_true[mask], well_ids[mask])

    rows: list[dict[str, Any]] = []
    overall_rmse = metric_buckets["overall"].rmse
    for segment, bucket in sorted(metric_buckets.items()):
        rows.append(
            {
                "variant": variant_name,
                "fold": fold,
                "segment": segment,
                "rmse": round(bucket.rmse, 6),
                "overall_rmse": round(overall_rmse, 6),
                "rows": bucket.n,
                "wells": len(bucket.wells),
            }
        )
    return rows, metric_buckets["overall"].sse, metric_buckets["overall"].n


def predict_valid_files_segmented(
    files: list[Path],
    config: dict[str, Any],
    models: dict[str, Any],
    segments: list[dict[str, Any]],
    *,
    variant_name: str,
    fold: int,
    bucket_config: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int]:
    metric_buckets: dict[str, MetricBucket] = defaultdict(MetricBucket)
    for path in files:
        df = pd.read_csv(path)
        frame = build_drift_feature_frame(df, config, include_target=True)
        if frame.target_residual is None or frame.eval_indices.size == 0:
            continue
        y_true = df.loc[frame.eval_indices, str(config_get(config, "data.target_column", "TVT"))]
        y_true = y_true.to_numpy(dtype=float)
        pred = np.full(y_true.shape, np.nan, dtype=float)
        eval_step = frame.features["eval_step"].to_numpy(dtype=float)
        for segment, mask in segment_masks(eval_step, segments).items():
            if not bool(mask.any()):
                continue
            segment_pred = predict_drift(frame, models[segment], config)
            pred[mask] = segment_pred[mask]

        well_ids = np.full(y_true.shape, well_id_from_path(path), dtype=object)
        labels = bucket_labels(eval_step, bucket_config)
        metric_buckets["overall"].add(pred, y_true, well_ids)
        for bucket_name in sorted(set(labels)):
            mask = labels == bucket_name
            metric_buckets[bucket_name].add(pred[mask], y_true[mask], well_ids[mask])

    rows: list[dict[str, Any]] = []
    overall_rmse = metric_buckets["overall"].rmse
    for segment, bucket in sorted(metric_buckets.items()):
        rows.append(
            {
                "variant": variant_name,
                "fold": fold,
                "segment": segment,
                "rmse": round(bucket.rmse, 6),
                "overall_rmse": round(overall_rmse, 6),
                "rows": bucket.n,
                "wells": len(bucket.wells),
            }
        )
    return rows, metric_buckets["overall"].sse, metric_buckets["overall"].n


def run_training_variants(
    files: list[Path],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    variants = list(get_nested(config, "audit.training_variants.variants", []))
    if not variants:
        return [], [], [], {}

    seed = int(config_get(config, "validation.seed", 42))
    n_folds = int(config_get(config, "validation.n_folds", 5))
    groups = np.asarray([well_id_from_path(path) for path in files])
    splitter = GroupKFold(n_splits=n_folds)
    bucket_config = list(get_nested(config, "audit.distance_buckets", []))
    max_rows_per_well = int(config_get(config, "model.training.max_train_rows_per_well", 800))
    max_rows_total = int(config_get(config, "model.training.max_train_rows_per_fold", 300000))

    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    overall_sse: dict[str, float] = defaultdict(float)
    overall_rows: dict[str, int] = defaultdict(int)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(files, groups=groups)):
        train_paths = [files[index] for index in train_idx]
        valid_paths = [files[index] for index in valid_idx]

        for variant_index, variant in enumerate(variants):
            name = str(variant["name"])
            kind = str(variant.get("kind", "sample_weight"))
            rng = np.random.default_rng(seed + fold * 1009 + variant_index)
            train = collect_training_rows(
                train_paths,
                config,
                rng,
                variant,
                max_rows_per_well=max_rows_per_well,
                max_rows_total=max_rows_total,
            )
            source_rows.extend(source_summary_rows(train, variant=name, fold=fold))
            print(f"Fold {fold} fitting {name} on {len(train):,} sampled rows")

            if kind == "segment_models":
                models: dict[str, Any] = {}
                segments = list(variant.get("segments", []))
                masks = segment_masks(train["eval_step"].to_numpy(dtype=float), segments)
                for segment, mask in masks.items():
                    segment_train = train.loc[mask].reset_index(drop=True)
                    if segment_train.empty:
                        raise ValueError(f"{name} has no training rows for segment={segment}")
                    model = fit_model(segment_train, config, seed=seed + fold)
                    models[segment] = model
                    importance_rows.extend(
                        feature_importance_rows(
                            model, config, variant=name, fold=fold, segment=segment
                        )
                    )
                rows, sse, n_rows = predict_valid_files_segmented(
                    valid_paths,
                    config,
                    models,
                    segments,
                    variant_name=name,
                    fold=fold,
                    bucket_config=bucket_config,
                )
            else:
                weights = sample_weights(train, variant)
                if str(variant.get("weight_profile", "uniform")) == "uniform":
                    weights = None
                model = fit_model(train, config, seed=seed + fold, weights=weights)
                importance_rows.extend(
                    feature_importance_rows(model, config, variant=name, fold=fold, segment="all")
                )
                rows, sse, n_rows = predict_valid_files(
                    valid_paths,
                    config,
                    model,
                    variant_name=name,
                    fold=fold,
                    bucket_config=bucket_config,
                )

            metric_rows.extend(rows)
            overall_sse[name] += sse
            overall_rows[name] += n_rows
            print(f"Fold {fold} {name} RMSE={rmse_from_sse(sse, n_rows):.6f}")

    overall = {
        name: round(rmse_from_sse(overall_sse[name], overall_rows[name]), 6)
        for name in sorted(overall_sse)
    }
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(
            output_dir / "pseudo_tail_training_metrics.csv", index=False
        )
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(
            output_dir / "pseudo_tail_feature_importance.csv", index=False
        )
    if source_rows:
        pd.DataFrame(source_rows).to_csv(output_dir / "pseudo_tail_source_summary.csv", index=False)
    return metric_rows, importance_rows, source_rows, overall


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_yaml(Path(__file__).with_name("config.yaml"))
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = train_files(paths, args.max_wells)
    oof = load_raw_oof(config)
    if args.max_wells is not None:
        allowed_wells = {well_id_from_path(path) for path in files}
        oof = oof[oof["well_id"].isin(allowed_wells)].reset_index(drop=True)
    oof = add_recent_linear(oof, files, config)

    candidate_rows, residual_rows, oof_overall = audit_existing_oof(oof, config, output_dir)
    training_enabled = bool(get_nested(config, "audit.training_variants.enabled", True))
    training_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    training_overall: dict[str, float] = {}
    if training_enabled and not args.skip_training:
        training_rows, importance_rows, source_rows, training_overall = run_training_variants(
            files, config, output_dir
        )

    best_training_variant = None
    if training_overall:
        best_training_variant = min(training_overall, key=training_overall.get)

    summary = {
        "experiment": "exp023_pseudo_tail_distance_augmentation",
        "status": "completed" if training_overall else "oof_audit_completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "max_wells": args.max_wells,
        "raw_anchor_cv": get_nested(config, "audit.raw_clean_cv"),
        "heldout_postprocess_cv": get_nested(config, "audit.heldout_postprocess_cv"),
        "weighted_clean_cv": get_nested(config, "audit.weighted_clean_cv"),
        "weighted_postprocess_cv": get_nested(config, "audit.weighted_postprocess_cv"),
        "oof_candidate_overall": oof_overall,
        "training_variant_overall": training_overall,
        "best_training_variant": best_training_variant,
        "best_training_cv": training_overall.get(best_training_variant)
        if best_training_variant
        else None,
        "artifact_rows": {
            "distance_candidate_metrics": len(candidate_rows),
            "distance_residual_bucket_summary": len(residual_rows),
            "pseudo_tail_training_metrics": len(training_rows),
            "pseudo_tail_feature_importance": len(importance_rows),
            "pseudo_tail_source_summary": len(source_rows),
        },
    }
    (output_dir / "pseudo_tail_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    metrics = {
        "experiment": "exp023_pseudo_tail_distance_augmentation",
        "status": summary["status"],
        "updated_at": summary["updated_at"],
        "cv": summary["best_training_cv"],
        "public_lb": None,
        "raw_anchor_cv": summary["raw_anchor_cv"],
        "heldout_postprocess_cv": summary["heldout_postprocess_cv"],
        "weighted_clean_cv": summary["weighted_clean_cv"],
        "weighted_postprocess_cv": summary["weighted_postprocess_cv"],
        "best_training_variant": best_training_variant,
        "training_variant_overall": training_overall,
        "oof_candidate_overall": oof_overall,
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
