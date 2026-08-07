from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested, load_config
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class WellFrame:
    well: str
    path: Path
    frame: pd.DataFrame
    prefix_indices: np.ndarray
    eval_indices: np.ndarray
    first_prefix_md: float
    last_prefix_md: float
    last_prefix_tvt: float
    linear_intercept: float
    linear_slope: float


@dataclass(frozen=True)
class ContextStats:
    source_wells: int
    source_rows: int
    bias_median: float
    bias_mean: float
    slope_intercept: float
    slope_u: float
    residual_scale: float
    scale_alpha: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-test prefix label context.")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke limit")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def rmse_from_errors(errors: np.ndarray) -> float:
    if errors.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean(np.square(errors)))))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(n_folds)


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def linear_fit(md: np.ndarray, tvt: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(md) & np.isfinite(tvt)
    if finite.sum() < 2 or np.nanstd(md[finite]) <= 0:
        return float(np.nanmedian(tvt[finite])), 0.0
    slope, intercept = np.polyfit(md[finite].astype(float), tvt[finite].astype(float), deg=1)
    return float(intercept), float(slope)


def read_well(path: Path, config: dict[str, Any]) -> WellFrame | None:
    usecols = ["MD", "TVT", "TVT_input"]
    frame = pd.read_csv(path, usecols=usecols)
    for column in usecols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    tvt_input = frame["TVT_input"].to_numpy(dtype=float)
    target = frame["TVT"].to_numpy(dtype=float)
    prefix_indices = np.flatnonzero(np.isfinite(tvt_input))
    eval_indices = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(target))

    min_prefix_rows = int(get_nested(config, "audit.min_prefix_rows") or 30)
    min_eval_rows = int(get_nested(config, "audit.min_eval_rows") or 30)
    if prefix_indices.size < min_prefix_rows or eval_indices.size < min_eval_rows:
        return None

    prefix_md = frame["MD"].to_numpy(dtype=float)[prefix_indices]
    prefix_tvt = tvt_input[prefix_indices]
    intercept, slope = linear_fit(prefix_md, prefix_tvt)
    return WellFrame(
        well=well_id_from_path(path),
        path=path,
        frame=frame,
        prefix_indices=prefix_indices,
        eval_indices=eval_indices,
        first_prefix_md=float(prefix_md[0]),
        last_prefix_md=float(prefix_md[-1]),
        last_prefix_tvt=float(prefix_tvt[-1]),
        linear_intercept=intercept,
        linear_slope=slope,
    )


def load_wells(train_dir: Path, config: dict[str, Any], max_wells: int | None) -> list[WellFrame]:
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        paths = paths[: int(max_wells)]
    wells = [read_well(path, config) for path in paths]
    loaded = [well for well in wells if well is not None]
    if len(loaded) < 2:
        raise ValueError(f"Need at least 2 usable wells, found {len(loaded)} in {train_dir}")
    return loaded


def normalized_u(well: WellFrame, indices: np.ndarray) -> np.ndarray:
    md = well.frame["MD"].to_numpy(dtype=float)[indices]
    denom = max(1.0, abs(well.last_prefix_md - well.first_prefix_md))
    return (md - well.last_prefix_md) / denom


def linear_prediction(well: WellFrame, indices: np.ndarray) -> np.ndarray:
    md = well.frame["MD"].to_numpy(dtype=float)[indices]
    return well.linear_intercept + well.linear_slope * md


def hold_prediction(well: WellFrame, indices: np.ndarray) -> np.ndarray:
    return np.full(indices.shape, well.last_prefix_tvt, dtype=float)


def prefix_context_rows(well: WellFrame, config: dict[str, Any]) -> pd.DataFrame:
    max_rows = int(get_nested(config, "audit.context_prefix_tail_rows") or 500)
    indices = well.prefix_indices[-max_rows:]
    pred_hold = hold_prediction(well, indices)
    residual = well.frame["TVT_input"].to_numpy(dtype=float)[indices] - pred_hold
    return pd.DataFrame(
        {
            "well": well.well,
            "u": normalized_u(well, indices),
            "residual": residual,
        }
    )


def robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    q25, q75 = np.nanpercentile(finite, [25, 75])
    iqr_scale = (q75 - q25) / 1.349 if q75 > q25 else 0.0
    std_scale = float(np.nanstd(finite))
    if iqr_scale > 0:
        return float(iqr_scale)
    return std_scale


def clip_array(values: np.ndarray, max_abs: float) -> np.ndarray:
    if not np.isfinite(max_abs) or max_abs <= 0:
        return values
    return np.clip(values, -max_abs, max_abs)


def fit_context_stats(source_wells: list[WellFrame], config: dict[str, Any]) -> ContextStats:
    frames = [prefix_context_rows(well, config) for well in source_wells]
    context = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if context.empty:
        return ContextStats(0, 0, 0.0, 0.0, 0.0, 0.0, float("nan"), 1.0)

    residual = pd.to_numeric(context["residual"], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(context["u"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(residual) & np.isfinite(u)
    residual = residual[finite]
    u = u[finite]
    if residual.size == 0:
        return ContextStats(len(source_wells), 0, 0.0, 0.0, 0.0, 0.0, float("nan"), 1.0)

    clip_abs = float(get_nested(config, "audit.context_residual_clip") or 60.0)
    clipped_residual = clip_array(residual, clip_abs)
    bias_median = float(np.nanmedian(clipped_residual))
    bias_mean = float(np.nanmean(clipped_residual))
    scale = robust_scale(clipped_residual)

    if clipped_residual.size >= 2 and np.nanstd(u) > 0:
        slope_u, intercept = np.polyfit(u, clipped_residual, deg=1)
    else:
        intercept, slope_u = bias_median, 0.0

    scale_reference = float(get_nested(config, "audit.scale_reference") or 10.0)
    min_alpha = float(get_nested(config, "audit.scale_min_alpha") or 0.25)
    max_alpha = float(get_nested(config, "audit.scale_max_alpha") or 1.0)
    if np.isfinite(scale) and scale >= 0:
        alpha = scale_reference / (scale_reference + scale)
    else:
        alpha = 1.0
    alpha = float(np.clip(alpha, min_alpha, max_alpha))
    return ContextStats(
        source_wells=len(source_wells),
        source_rows=int(clipped_residual.size),
        bias_median=bias_median,
        bias_mean=bias_mean,
        slope_intercept=float(intercept),
        slope_u=float(slope_u),
        residual_scale=float(scale),
        scale_alpha=alpha,
    )


def prediction_candidates(
    well: WellFrame,
    indices: np.ndarray,
    context: ContextStats,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    hold = hold_prediction(well, indices)
    linear = linear_prediction(well, indices)
    u = normalized_u(well, indices)
    max_correction = float(get_nested(config, "audit.max_context_correction") or 25.0)
    bias = float(np.clip(context.bias_median, -max_correction, max_correction))
    slope_correction = clip_array(context.slope_intercept + context.slope_u * u, max_correction)
    bias_hold = hold + bias
    slope_hold = hold + slope_correction
    scale_slope_hold = hold + context.scale_alpha * slope_correction
    bias_scale_hold = hold + context.scale_alpha * bias
    return {
        "hold_prefix_control": hold,
        "self_linear_prefix_control": linear,
        "cross_batch_bias_hold": bias_hold,
        "cross_batch_slope_hold": slope_hold,
        "cross_batch_scale_slope_hold": scale_slope_hold,
        "cross_batch_bias_scale_hold": bias_scale_hold,
    }


def bucket_name(eval_step: np.ndarray) -> np.ndarray:
    bins = np.array([0, 50, 250, 500, 1000, 2500, np.inf], dtype=float)
    names = np.array(["0_50", "51_250", "251_500", "501_1000", "1001_2500", "2501_plus"])
    codes = np.digitize(eval_step.astype(float), bins[1:-1], right=True)
    return names[codes]


def metrics_from_errors(errors: np.ndarray) -> dict[str, Any]:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {"rows": 0, "rmse": None, "mae": None, "bias": None, "within10": None}
    return {
        "rows": int(finite.size),
        "rmse": rmse_from_errors(finite),
        "mae": float(np.nanmean(np.abs(finite))),
        "bias": float(np.nanmean(finite)),
        "within10": float(np.nanmean(np.abs(finite) <= 10.0)),
    }


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: key for column, key in zip(group_cols, keys, strict=True)}
        item.update(metrics_from_errors(group["error"].to_numpy(dtype=float)))
        rows.append(item)
    return pd.DataFrame(rows)


def selection_readout(candidate_metrics: pd.DataFrame) -> dict[str, Any]:
    fold_rows = candidate_metrics.copy()
    candidates = sorted(fold_rows["candidate"].unique())
    folds = sorted(fold_rows["fold"].unique())
    selected: list[dict[str, Any]] = []
    total_sse = 0.0
    total_rows = 0
    for fold in folds:
        train = fold_rows[fold_rows["fold"] != fold]
        valid = fold_rows[fold_rows["fold"] == fold]
        score_items = []
        for candidate in candidates:
            subset = train[train["candidate"] == candidate]
            rows = int(subset["rows"].sum())
            if rows <= 0:
                rmse = float("nan")
            else:
                rmse = math.sqrt(float(((subset["rmse"] ** 2) * subset["rows"]).sum()) / rows)
            score_items.append((candidate, rmse))
        best_candidate = min(
            score_items,
            key=lambda item: (math.inf if np.isnan(item[1]) else item[1], item[0]),
        )[0]
        valid_best = valid[valid["candidate"] == best_candidate]
        rows = int(valid_best["rows"].sum())
        sse = float(((valid_best["rmse"] ** 2) * valid_best["rows"]).sum())
        total_sse += sse
        total_rows += rows
        selected.append(
            {
                "fold": int(fold),
                "selected_candidate": best_candidate,
                "selection_train_rmse": dict(score_items).get(best_candidate),
                "valid_rows": rows,
                "valid_rmse": math.sqrt(sse / rows) if rows > 0 else None,
            }
        )
    return {
        "fold_selection_rows": selected,
        "fold_selection_rmse": math.sqrt(total_sse / total_rows) if total_rows > 0 else None,
        "fold_selection_n": total_rows,
    }


def run_audit(
    *,
    paths: ExperimentPaths,
    config: dict[str, Any],
    output_dir: Path,
    max_wells: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = load_wells(paths.train_data_dir, config, max_wells)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    if len(wells) < n_folds:
        n_folds = max(2, len(wells))
    groups = np.array([well.well for well in wells])
    splitter = GroupKFold(n_splits=n_folds)

    row_metrics: list[pd.DataFrame] = []
    context_rows: list[dict[str, Any]] = []
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(wells)), groups=groups)):
        valid_wells = [wells[int(index)] for index in valid_index]
        for target in valid_wells:
            source_wells = [well for well in valid_wells if well.well != target.well]
            context = fit_context_stats(source_wells, config)
            eval_indices = target.eval_indices
            true = target.frame["TVT"].to_numpy(dtype=float)[eval_indices]
            eval_step = np.arange(1, eval_indices.size + 1, dtype=int)
            candidates = prediction_candidates(target, eval_indices, context, config)
            context_rows.append(
                {
                    "fold": fold,
                    "well": target.well,
                    "eval_rows": int(eval_indices.size),
                    "prefix_rows": int(target.prefix_indices.size),
                    "source_wells": context.source_wells,
                    "source_rows": context.source_rows,
                    "bias_median": context.bias_median,
                    "bias_mean": context.bias_mean,
                    "slope_intercept": context.slope_intercept,
                    "slope_u": context.slope_u,
                    "residual_scale": context.residual_scale,
                    "scale_alpha": context.scale_alpha,
                }
            )
            pieces = []
            buckets = bucket_name(eval_step)
            for candidate, pred in candidates.items():
                pieces.append(
                    pd.DataFrame(
                        {
                            "fold": fold,
                            "well": target.well,
                            "candidate": candidate,
                            "eval_step": eval_step,
                            "eval_bucket": buckets,
                            "error": pred - true,
                        }
                    )
                )
            row_metrics.append(pd.concat(pieces, ignore_index=True))

    errors = pd.concat(row_metrics, ignore_index=True)
    overall = summarize_group(errors, ["candidate"]).sort_values("rmse").reset_index(drop=True)
    by_fold = summarize_group(errors, ["fold", "candidate"]).sort_values(["fold", "rmse"])
    by_well = summarize_group(errors, ["well", "candidate"]).sort_values(["well", "rmse"])
    by_bucket = summarize_group(errors, ["eval_bucket", "candidate"]).sort_values(
        ["eval_bucket", "rmse"]
    )
    context_frame = pd.DataFrame(context_rows)
    fold_candidate = by_fold[["fold", "candidate", "rows", "rmse"]].copy()
    selection = selection_readout(fold_candidate)

    overall_path = output_dir / "cross_test_prefix_label_candidate_metrics.csv"
    fold_path = output_dir / "cross_test_prefix_label_fold_metrics.csv"
    well_path = output_dir / "cross_test_prefix_label_by_well.csv"
    bucket_path = output_dir / "cross_test_prefix_label_bucket_metrics.csv"
    context_path = output_dir / "cross_test_prefix_label_context_stats.csv"
    overall.to_csv(overall_path, index=False)
    by_fold.to_csv(fold_path, index=False)
    by_well.to_csv(well_path, index=False)
    by_bucket.to_csv(bucket_path, index=False)
    context_frame.to_csv(context_path, index=False)

    best = overall.iloc[0].to_dict()
    baseline = overall[overall["candidate"] == "self_linear_prefix_control"].iloc[0].to_dict()
    hold = overall[overall["candidate"] == "hold_prefix_control"].iloc[0].to_dict()
    summary = {
        "experiment": paths.experiment_name,
        "status": "audit_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "debug_max_wells": max_wells,
        "n_wells": len(wells),
        "n_folds": n_folds,
        "best_candidate": to_jsonable(best),
        "self_linear_baseline": to_jsonable(baseline),
        "hold_baseline": to_jsonable(hold),
        "delta_vs_self_linear": (
            float(best["rmse"] - baseline["rmse"])
            if np.isfinite(best["rmse"]) and np.isfinite(baseline["rmse"])
            else None
        ),
        "delta_vs_hold": (
            float(best["rmse"] - hold["rmse"])
            if np.isfinite(best["rmse"]) and np.isfinite(hold["rmse"])
            else None
        ),
        "selection": to_jsonable(selection),
        "artifacts": {
            "candidate_metrics": str(overall_path),
            "fold_metrics": str(fold_path),
            "by_well": str(well_path),
            "bucket_metrics": str(bucket_path),
            "context_stats": str(context_path),
        },
        "notes": [
            "This is a rules-risk diagnostic; it uses other validation wells' finite "
            "TVT_input prefix labels.",
            "No inference notebook should produce a submission from this audit without "
            "organizer/rules approval.",
        ],
    }
    summary_path = output_dir / "cross_test_prefix_label_summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n")
    return summary


def write_metrics(paths: ExperimentPaths, summary: dict[str, Any]) -> None:
    metrics = {
        "experiment": paths.experiment_name,
        "status": summary["status"],
        "updated_at": datetime.now(UTC).isoformat(),
        "cv": summary["best_candidate"]["rmse"],
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "key_idea": (
            "Diagnose whether other same-batch wells' visible TVT_input prefix labels "
            "support batch-level bias/slope/scale correction."
        ),
        "notes": "Diagnostic only; no inference port or submit from this result alone.",
        "summary": summary,
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_config()
    paths.ensure_output_dirs()
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    summary = run_audit(paths=paths, config=config, output_dir=output_dir, max_wells=args.max_wells)
    write_metrics(paths, summary)
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
