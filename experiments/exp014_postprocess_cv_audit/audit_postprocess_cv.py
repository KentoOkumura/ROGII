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
import yaml
from settings import ExperimentPaths

USECOLS = [
    "variant",
    "fold",
    "well_id",
    "eval_step",
    "last_anchor",
    "y_true",
    "y_pred",
]
STAT_FIELDS = ("n", "target2", "pred2", "pred_target")


@dataclass(frozen=True)
class Bucket:
    name: str
    max_step: float
    alpha: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exp013 postprocess alpha CV.")
    parser.add_argument("--oof", default=None, help="Path to row_oof_predictions.csv")
    parser.add_argument("--variant", default=None, help="OOF variant to audit")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    return parser.parse_args()


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_local_config() -> dict[str, Any]:
    with Path(__file__).with_name("config.yaml").open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return value


def load_buckets(config: dict[str, Any], key: str) -> list[Bucket]:
    raw_buckets = get_nested(config, key, [])
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError(f"{key} must be a non-empty list")
    buckets: list[Bucket] = []
    for item in raw_buckets:
        if not isinstance(item, dict):
            raise ValueError(f"{key} entries must be mappings")
        buckets.append(
            Bucket(
                name=str(item["name"]),
                max_step=float(item["max_step"]),
                alpha=float(item["alpha"]) if "alpha" in item else None,
            )
        )
    return buckets


def empty_stats() -> dict[str, float]:
    return {"n": 0.0, "target2": 0.0, "pred2": 0.0, "pred_target": 0.0}


def add_stats(left: dict[str, float], right: dict[str, float]) -> None:
    for field in STAT_FIELDS:
        left[field] += float(right.get(field, 0.0))


def subtract_stats(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {
        field: float(left.get(field, 0.0)) - float(right.get(field, 0.0))
        for field in STAT_FIELDS
    }


def stats_sse(stats: dict[str, float], alpha: float) -> float:
    return (
        float(stats["target2"])
        - 2.0 * alpha * float(stats["pred_target"])
        + alpha * alpha * float(stats["pred2"])
    )


def stats_rmse(stats: dict[str, float], alpha: float) -> float:
    n = float(stats["n"])
    if n <= 0:
        return float("nan")
    return math.sqrt(max(0.0, stats_sse(stats, alpha)) / n)


def fit_alpha(stats: dict[str, float], *, alpha_min: float, alpha_max: float) -> float:
    denom = float(stats["pred2"])
    if denom <= 0:
        return 1.0
    alpha = float(stats["pred_target"]) / denom
    return float(np.clip(alpha, alpha_min, alpha_max))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % n_folds


def bucket_codes(eval_step: pd.Series, buckets: list[Bucket]) -> np.ndarray:
    values = eval_step.to_numpy(dtype=float)
    codes = np.full(values.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        mask = (values > previous_max) & (values <= bucket.max_step)
        codes[mask] = idx
        previous_max = bucket.max_step
    return codes


def aggregate_chunk(
    frame: pd.DataFrame,
    *,
    buckets: list[Bucket],
    well_holdout_folds: int,
    total_by_bucket: dict[int, dict[str, float]],
    by_cv_fold_bucket: dict[tuple[int, int], dict[str, float]],
    by_well_fold_bucket: dict[tuple[int, int], dict[str, float]],
) -> None:
    frame = frame.copy()
    frame["bucket_id"] = bucket_codes(frame["eval_step"], buckets)
    frame["well_audit_fold"] = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_holdout_folds)
    )

    pred_resid = frame["y_pred"].to_numpy(dtype=float) - frame["last_anchor"].to_numpy(dtype=float)
    true_resid = frame["y_true"].to_numpy(dtype=float) - frame["last_anchor"].to_numpy(dtype=float)
    frame["_target2"] = true_resid * true_resid
    frame["_pred2"] = pred_resid * pred_resid
    frame["_pred_target"] = pred_resid * true_resid

    for bucket_id, group in frame.groupby("bucket_id", sort=False):
        stats = total_by_bucket[int(bucket_id)]
        stats["n"] += float(len(group))
        stats["target2"] += float(group["_target2"].sum())
        stats["pred2"] += float(group["_pred2"].sum())
        stats["pred_target"] += float(group["_pred_target"].sum())

    for (fold, bucket_id), group in frame.groupby(["fold", "bucket_id"], sort=False):
        stats = by_cv_fold_bucket[(int(fold), int(bucket_id))]
        stats["n"] += float(len(group))
        stats["target2"] += float(group["_target2"].sum())
        stats["pred2"] += float(group["_pred2"].sum())
        stats["pred_target"] += float(group["_pred_target"].sum())

    for (fold, bucket_id), group in frame.groupby(["well_audit_fold", "bucket_id"], sort=False):
        stats = by_well_fold_bucket[(int(fold), int(bucket_id))]
        stats["n"] += float(len(group))
        stats["target2"] += float(group["_target2"].sum())
        stats["pred2"] += float(group["_pred2"].sum())
        stats["pred_target"] += float(group["_pred_target"].sum())


def combine_bucket_stats(by_bucket: dict[int, dict[str, float]]) -> dict[str, float]:
    total = empty_stats()
    for stats in by_bucket.values():
        add_stats(total, stats)
    return total


def score_bucket_alphas(
    eval_by_bucket: dict[int, dict[str, float]],
    alphas: dict[int, float],
) -> tuple[float, float]:
    n = 0.0
    sse = 0.0
    for bucket_id, stats in eval_by_bucket.items():
        n += float(stats["n"])
        sse += stats_sse(stats, alphas[bucket_id])
    return math.sqrt(max(0.0, sse) / n), sse


def fixed_alpha_metrics(
    *,
    name: str,
    by_bucket: dict[int, dict[str, float]],
    alphas: dict[int, float],
    raw_rmse: float,
) -> dict[str, Any]:
    rmse, _ = score_bucket_alphas(by_bucket, alphas)
    return {
        "candidate": name,
        "rmse": round(rmse, 6),
        "delta_vs_raw": round(rmse - raw_rmse, 6),
        "rows": int(sum(stats["n"] for stats in by_bucket.values())),
    }


def build_cv_holdout(
    *,
    total_by_bucket: dict[int, dict[str, float]],
    holdout_by_fold_bucket: dict[tuple[int, int], dict[str, float]],
    folds: list[int],
    buckets: list[Bucket],
    alpha_min: float,
    alpha_max: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_sse = 0.0
    total_n = 0.0
    alpha_rows: list[dict[str, Any]] = []
    for fold in folds:
        eval_by_bucket: dict[int, dict[str, float]] = {}
        alphas: dict[int, float] = {}
        fold_sse = 0.0
        fold_n = 0.0
        for bucket_id, bucket in enumerate(buckets):
            eval_stats = holdout_by_fold_bucket[(fold, bucket_id)]
            train_stats = subtract_stats(total_by_bucket[bucket_id], eval_stats)
            alpha = fit_alpha(train_stats, alpha_min=alpha_min, alpha_max=alpha_max)
            sse = stats_sse(eval_stats, alpha)
            fold_sse += sse
            fold_n += eval_stats["n"]
            eval_by_bucket[bucket_id] = eval_stats
            alphas[bucket_id] = alpha
            alpha_rows.append(
                {
                    "audit_fold": fold,
                    "bucket": bucket.name,
                    "alpha": alpha,
                    "train_rows": int(train_stats["n"]),
                    "eval_rows": int(eval_stats["n"]),
                    "eval_rmse": stats_rmse(eval_stats, alpha),
                    "eval_raw_rmse": stats_rmse(eval_stats, 1.0),
                    "eval_anchor_rmse": stats_rmse(eval_stats, 0.0),
                }
            )
        total_sse += fold_sse
        total_n += fold_n
    rmse = math.sqrt(max(0.0, total_sse) / total_n)
    return {"rmse": rmse, "rows": int(total_n)}, alpha_rows


def write_outputs(
    *,
    output_dir: Path,
    metrics: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_dir / "postprocess_cv_audit_metrics.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(output_dir / "postprocess_cv_audit_alphas.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "postprocess_cv_audit_bucket_summary.csv",
        index=False,
    )
    with (output_dir / "postprocess_cv_audit_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_local_config()
    buckets = load_buckets(config, "audit.distance_buckets")
    fixed_buckets = load_buckets(config, "audit.exp013_fixed_buckets")

    oof_path = Path(args.oof or get_nested(config, "audit.source_oof_predictions"))
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    variant = args.variant or str(get_nested(config, "audit.source_variant", "lightgbm_no_gr"))
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    chunk_rows = int(get_nested(config, "audit.chunk_rows", 250000))
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    alpha_min = float(get_nested(config, "audit.alpha_clip.min", 0.2))
    alpha_max = float(get_nested(config, "audit.alpha_clip.max", 1.15))

    total_by_bucket: dict[int, dict[str, float]] = defaultdict(empty_stats)
    by_cv_fold_bucket: dict[tuple[int, int], dict[str, float]] = defaultdict(empty_stats)
    by_well_fold_bucket: dict[tuple[int, int], dict[str, float]] = defaultdict(empty_stats)
    cv_folds: set[int] = set()
    rows_seen = 0

    for chunk in pd.read_csv(oof_path, usecols=USECOLS, chunksize=chunk_rows):
        chunk = chunk[chunk["variant"] == variant]
        if chunk.empty:
            continue
        cv_folds.update(int(value) for value in chunk["fold"].unique())
        rows_seen += len(chunk)
        aggregate_chunk(
            chunk,
            buckets=buckets,
            well_holdout_folds=well_holdout_folds,
            total_by_bucket=total_by_bucket,
            by_cv_fold_bucket=by_cv_fold_bucket,
            by_well_fold_bucket=by_well_fold_bucket,
        )

    if rows_seen == 0:
        raise ValueError(f"No rows found for variant={variant!r} in {oof_path}")

    total_stats = combine_bucket_stats(total_by_bucket)
    raw_rmse = stats_rmse(total_stats, 1.0)
    anchor_rmse = stats_rmse(total_stats, 0.0)

    in_sample_alphas = {
        bucket_id: fit_alpha(stats, alpha_min=alpha_min, alpha_max=alpha_max)
        for bucket_id, stats in total_by_bucket.items()
    }
    fixed_alphas = {
        idx: float(bucket.alpha if bucket.alpha is not None else 1.0)
        for idx, bucket in enumerate(fixed_buckets)
    }

    metrics: list[dict[str, Any]] = [
        {
            "candidate": "raw_lightgbm_no_gr",
            "rmse": round(raw_rmse, 6),
            "delta_vs_raw": 0.0,
            "rows": int(total_stats["n"]),
        },
        {
            "candidate": "last_anchor",
            "rmse": round(anchor_rmse, 6),
            "delta_vs_raw": round(anchor_rmse - raw_rmse, 6),
            "rows": int(total_stats["n"]),
        },
        fixed_alpha_metrics(
            name="exp013_fixed_bucket_alphas",
            by_bucket=total_by_bucket,
            alphas=fixed_alphas,
            raw_rmse=raw_rmse,
        ),
        fixed_alpha_metrics(
            name="in_sample_bucket_refit",
            by_bucket=total_by_bucket,
            alphas=in_sample_alphas,
            raw_rmse=raw_rmse,
        ),
    ]

    cv_summary, cv_alpha_rows = build_cv_holdout(
        total_by_bucket=total_by_bucket,
        holdout_by_fold_bucket=by_cv_fold_bucket,
        folds=sorted(cv_folds),
        buckets=buckets,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    metrics.append(
        {
            "candidate": "leave_one_original_fold_out_bucket_fit",
            "rmse": round(cv_summary["rmse"], 6),
            "delta_vs_raw": round(cv_summary["rmse"] - raw_rmse, 6),
            "rows": cv_summary["rows"],
        }
    )

    well_summary, well_alpha_rows = build_cv_holdout(
        total_by_bucket=total_by_bucket,
        holdout_by_fold_bucket=by_well_fold_bucket,
        folds=list(range(well_holdout_folds)),
        buckets=buckets,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    metrics.append(
        {
            "candidate": "well_bucket_holdout_fit",
            "rmse": round(well_summary["rmse"], 6),
            "delta_vs_raw": round(well_summary["rmse"] - raw_rmse, 6),
            "rows": well_summary["rows"],
        }
    )

    bucket_rows: list[dict[str, Any]] = []
    for bucket_id, bucket in enumerate(buckets):
        stats = total_by_bucket[bucket_id]
        bucket_rows.append(
            {
                "bucket": bucket.name,
                "max_step": bucket.max_step,
                "rows": int(stats["n"]),
                "last_anchor_rmse": round(stats_rmse(stats, 0.0), 6),
                "raw_rmse": round(stats_rmse(stats, 1.0), 6),
                "in_sample_alpha": in_sample_alphas[bucket_id],
                "in_sample_rmse": round(stats_rmse(stats, in_sample_alphas[bucket_id]), 6),
                "exp013_fixed_alpha": fixed_alphas[bucket_id],
                "exp013_fixed_rmse": round(stats_rmse(stats, fixed_alphas[bucket_id]), 6),
            }
        )

    alpha_rows: list[dict[str, Any]] = []
    for bucket_id, bucket in enumerate(buckets):
        alpha_rows.append(
            {
                "audit": "in_sample_bucket_refit",
                "audit_fold": "all",
                "bucket": bucket.name,
                "alpha": in_sample_alphas[bucket_id],
                "train_rows": int(total_by_bucket[bucket_id]["n"]),
                "eval_rows": int(total_by_bucket[bucket_id]["n"]),
                "eval_rmse": stats_rmse(total_by_bucket[bucket_id], in_sample_alphas[bucket_id]),
                "eval_raw_rmse": stats_rmse(total_by_bucket[bucket_id], 1.0),
                "eval_anchor_rmse": stats_rmse(total_by_bucket[bucket_id], 0.0),
            }
        )
    alpha_rows.extend(
        {"audit": "leave_one_original_fold_out_bucket_fit", **row} for row in cv_alpha_rows
    )
    alpha_rows.extend({"audit": "well_bucket_holdout_fit", **row} for row in well_alpha_rows)

    loo_rmse = cv_summary["rmse"]
    well_rmse = well_summary["rmse"]
    clean_postprocess_supported = bool(loo_rmse < raw_rmse and well_rmse < raw_rmse)
    status = "completed"
    selected_clean_cv = loo_rmse if clean_postprocess_supported else raw_rmse

    summary = {
        "experiment": "exp014_postprocess_cv_audit",
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "source_oof_predictions": str(oof_path),
        "source_variant": variant,
        "rows": int(total_stats["n"]),
        "raw_clean_cv": round(raw_rmse, 6),
        "last_anchor_cv": round(anchor_rmse, 6),
        "exp013_fixed_bucket_alphas_cv": metrics[2]["rmse"],
        "in_sample_bucket_refit_cv": metrics[3]["rmse"],
        "leave_one_original_fold_out_bucket_fit_cv": round(loo_rmse, 6),
        "well_bucket_holdout_fit_cv": round(well_rmse, 6),
        "selected_clean_cv": round(selected_clean_cv, 6),
        "clean_postprocess_supported": clean_postprocess_supported,
        "metric": "rmse",
        "notes": (
            "Held-out alpha audits support a constrained distance bucket shrink candidate"
            if clean_postprocess_supported
            else (
                "Keep raw LightGBM no-GR as clean CV; "
                "exp013 bucket shrink remains OOF-fit/LB anchor"
            )
        ),
        "metrics": metrics,
    }

    write_outputs(
        output_dir=output_dir,
        metrics=metrics,
        alpha_rows=alpha_rows,
        bucket_rows=bucket_rows,
        summary=summary,
    )
    with (Path(__file__).with_name("metrics.json")).open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
