from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from exact_hmm_smoother import (
    load_lgb_prediction_series,
    resolve_existing_file,
    sha256_path,
    to_jsonable,
)
from settings import ExperimentPaths, get_nested, load_config


EXPERIMENT_NAME = "exp234_crossfitted_residual_scale_emission_hmm_on_exp218"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def exp072_column(candidate: str) -> str:
    if candidate in {"pf_ancc", "pf_z"}:
        return candidate
    return f"{candidate}_d"


def exp072_prediction(frame: pd.DataFrame, candidate: str) -> np.ndarray:
    column = exp072_column(candidate)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if column.endswith("_d"):
        values = values + pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float64)
    return values


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def within(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= threshold))


def distance_bucket(md_since: pd.Series) -> pd.Series:
    values = pd.to_numeric(md_since, errors="coerce").to_numpy(np.float64)
    labels = np.full(len(values), "1000_plus", dtype=object)
    labels[values < 1000.0] = "500_1000"
    labels[values < 500.0] = "250_500"
    labels[values < 250.0] = "100_250"
    labels[values < 100.0] = "050_100"
    labels[values < 50.0] = "000_050"
    return pd.Series(labels, index=md_since.index)


def metric_row(
    candidate: str,
    true_tvt: np.ndarray,
    pred: np.ndarray,
    rows: int,
    baselines: dict[str, np.ndarray],
) -> dict[str, Any]:
    pred_rmse = rmse(true_tvt, pred)
    row: dict[str, Any] = {
        "candidate": candidate,
        "rows": int(rows),
        "rmse": pred_rmse,
        "mae": mae(true_tvt, pred),
        "bias": bias(true_tvt, pred),
        "within10": within(true_tvt, pred, 10.0),
    }
    for baseline_name, baseline_pred in baselines.items():
        baseline_rmse = rmse(true_tvt, baseline_pred)
        row[f"delta_rmse_vs_{baseline_name}"] = pred_rmse - baseline_rmse
        row[f"delta_mae_vs_{baseline_name}"] = mae(true_tvt, pred) - mae(true_tvt, baseline_pred)
        row[f"delta_within10_vs_{baseline_name}"] = within(true_tvt, pred, 10.0) - within(
            true_tvt,
            baseline_pred,
            10.0,
        )
    return row


def align_frames(baseline: pd.DataFrame, hmm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if len(baseline) == len(hmm):
        mismatch = int((baseline["id"].astype(str).to_numpy() != hmm["id"].astype(str).to_numpy()).sum())
        if mismatch == 0:
            return baseline.reset_index(drop=True), hmm.reset_index(drop=True), 0
    merged = baseline.merge(hmm, on="id", how="inner", suffixes=("_exp072", "_hmm"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(hmm):
        raise ValueError(
            f"id set mismatch after merge: baseline={len(baseline)} hmm={len(hmm)} merged={len(merged)}"
        )
    baseline_cols = [col for col in baseline.columns if col != "id"]
    hmm_cols = [col for col in hmm.columns if col != "id"]
    baseline_aligned = pd.DataFrame({"id": merged["id"]})
    hmm_aligned = pd.DataFrame({"id": merged["id"]})
    for col in baseline_cols:
        baseline_aligned[col] = merged[f"{col}_exp072"] if f"{col}_exp072" in merged else merged[col]
    for col in hmm_cols:
        hmm_aligned[col] = merged[f"{col}_hmm"] if f"{col}_hmm" in merged else merged[col]
    return baseline_aligned, hmm_aligned, -1


def compute_uncertainty_bins(
    candidate: str,
    std_values: np.ndarray,
    true_tvt: np.ndarray,
    pred: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "hmm_std": np.asarray(std_values, dtype=np.float64),
            "abs_error": np.abs(pred - true_tvt),
            "sq_error": (pred - true_tvt) ** 2,
        }
    )
    work["bin"] = pd.qcut(work["hmm_std"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_id, group in work.groupby("bin", sort=True):
        rows.append(
            {
                "candidate": candidate,
                "hmm_std_bin": int(bin_id),
                "rows": int(len(group)),
                "hmm_std_min": float(group["hmm_std"].min()),
                "hmm_std_mean": float(group["hmm_std"].mean()),
                "hmm_std_max": float(group["hmm_std"].max()),
                "abs_error_mean": float(group["abs_error"].mean()),
                "rmse": float(np.sqrt(group["sq_error"].mean())),
            }
        )
    return pd.DataFrame(rows)


def compute_step_delta_rates(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = frame[["well", "last_known_tvt"]].copy()
    for name, pred in predictions.items():
        base[name] = pred
    for name in predictions:
        deltas: list[np.ndarray] = []
        for _, group in base.groupby("well", sort=False):
            pred = group[name].to_numpy(np.float64)
            prev = np.empty(len(group), dtype=np.float64)
            prev[0] = float(group["last_known_tvt"].iloc[0])
            if len(group) > 1:
                prev[1:] = pred[:-1]
            deltas.append(np.abs(pred - prev))
        abs_delta = np.concatenate(deltas) if deltas else np.array([], dtype=np.float64)
        row: dict[str, Any] = {
            "candidate": name,
            "rows": int(len(abs_delta)),
            "abs_step_delta_mean": float(np.mean(abs_delta)) if len(abs_delta) else None,
            "abs_step_delta_p95": float(np.quantile(abs_delta, 0.95)) if len(abs_delta) else None,
            "abs_step_delta_p99": float(np.quantile(abs_delta, 0.99)) if len(abs_delta) else None,
        }
        for threshold in thresholds:
            key = f"rate_abs_step_delta_gt_{str(threshold).replace('.', 'p')}"
            row[key] = float(np.mean(abs_delta > threshold)) if len(abs_delta) else None
        rows.append(row)
    return pd.DataFrame(rows)


def load_hidden_like_masks(paths: ExperimentPaths, config: dict[str, Any], wells: pd.Series) -> dict[str, np.ndarray]:
    hidden_config = get_nested(config, "comparison.hidden_like") or {}
    if not bool(hidden_config.get("enabled", False)):
        return {}
    path = resolve_existing_file(paths.root, list(hidden_config.get("fold_assignment_candidates") or []))
    frame = pd.read_csv(path, dtype={"well_id": str})
    masks: dict[str, np.ndarray] = {}
    well_values = wells.astype(str).to_numpy()
    for split_name, role_column in (hidden_config.get("valid_role_columns") or {}).items():
        if role_column not in frame.columns:
            continue
        valid_wells = set(frame.loc[frame[role_column].astype(str) == "valid", "well_id"].astype(str))
        masks[str(split_name)] = np.isin(well_values, list(valid_wells))
    return masks


def load_lgb_prediction_baselines(paths: ExperimentPaths, config: dict[str, Any], ids: pd.Series) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    lgb_config = get_nested(config, "lgb_emission") or {}
    sources_config = lgb_config.get("sources") or {}
    compare_sources = list((get_nested(config, "comparison.lgb_baseline_sources") or lgb_config.get("active_sources") or []))
    predictions: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for source in compare_sources:
        if source not in sources_config:
            raise KeyError(f"LGB comparison source is not configured: {source}")
        series, meta = load_lgb_prediction_series(paths.root, source, sources_config[source])
        aligned = series.reindex(ids.astype(str))
        missing = int(aligned.isna().sum())
        if missing:
            example = aligned[aligned.isna()].index[:5].tolist()
            raise ValueError(f"{source} missing {missing} predictions for comparison ids, examples={example}")
        name = f"{source}"
        predictions[name] = aligned.to_numpy(np.float64)
        metadata.append(meta)
    return predictions, metadata


def run_direct_comparison(
    *,
    baseline_frame: pd.DataFrame | None = None,
    baseline_source: str | None = None,
    hmm_frame: pd.DataFrame | None = None,
    hmm_source: str | None = None,
) -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    comparison = get_nested(config, "comparison") or {}
    output_prefix = str(comparison.get("output_prefix") or "exp234_crossfitted_residual_scale_hmm")
    baseline_candidates = list(comparison.get("baseline_candidate_columns") or ["likpf_mean"])
    thresholds = [float(v) for v in (comparison.get("step_delta_thresholds") or [0.08, 0.10, 0.20])]

    baseline_needed = {"id", "well", "target", "last_known_tvt", "md_since"}
    for candidate in baseline_candidates:
        baseline_needed.add(exp072_column(candidate))
    if baseline_frame is None:
        baseline_path = resolve_existing_file(paths.root, list(comparison.get("baseline_feature_cache") or []))
        baseline = pd.read_csv(
            baseline_path,
            usecols=sorted(baseline_needed),
            dtype={"id": str, "well": str},
        )
        baseline_source = str(baseline_path)
        baseline_load_mode = "csv_gzip"
    else:
        missing = sorted(baseline_needed.difference(baseline_frame.columns))
        if missing:
            raise ValueError(f"baseline frame is missing required columns: {missing}")
        baseline = baseline_frame.loc[:, sorted(baseline_needed)].copy()
        baseline["id"] = baseline["id"].astype(str)
        baseline["well"] = baseline["well"].astype(str)
        baseline_source = baseline_source or "in_memory_exp072_full_cache"
        baseline_load_mode = "in_memory"

    if hmm_frame is None:
        hmm_path = resolve_existing_file(paths.root, list(comparison.get("hmm_feature_cache") or []))
        hmm = pd.read_csv(hmm_path, dtype={"id": str, "well": str})
        hmm_source = str(hmm_path)
        hmm_load_mode = "csv_gzip"
    else:
        hmm = hmm_frame.copy()
        hmm["id"] = hmm["id"].astype(str)
        hmm["well"] = hmm["well"].astype(str)
        hmm_source = hmm_source or "in_memory_lgb_emission_hmm_cache"
        hmm_load_mode = "in_memory"

    baseline, hmm, id_mismatches = align_frames(baseline, hmm)
    true_tvt = (
        pd.to_numeric(baseline["last_known_tvt"], errors="coerce").to_numpy(np.float64)
        + pd.to_numeric(baseline["target"], errors="coerce").to_numpy(np.float64)
    )

    predictions: dict[str, np.ndarray] = {}
    for candidate in baseline_candidates:
        predictions[f"exp072_{candidate}"] = exp072_prediction(baseline, candidate)
    lgb_predictions, lgb_metadata = load_lgb_prediction_baselines(paths, config, baseline["id"])
    predictions.update(lgb_predictions)

    hmm_mean_columns = [col for col in hmm.columns if col.endswith("_mean_tvt")]
    if not hmm_mean_columns:
        raise ValueError("No HMM prediction columns ending with _mean_tvt were found")
    for column in hmm_mean_columns:
        predictions[column.removesuffix("_mean_tvt")] = pd.to_numeric(hmm[column], errors="coerce").to_numpy(np.float64)

    primary_baseline_names = list(comparison.get("primary_baselines") or [])
    baselines = {
        name: predictions[name]
        for name in primary_baseline_names
        if name in predictions
    }
    if not baselines:
        for fallback_name in ("exp148_lgb_mean", "exp072_likpf_mean"):
            if fallback_name in predictions:
                baselines[fallback_name] = predictions[fallback_name]

    bucket = distance_bucket(baseline["md_since"])
    hidden_masks = load_hidden_like_masks(paths, config, baseline["well"])

    overall_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    for name, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"Prediction contains non-finite values: {name}")
        overall_rows.append(metric_row(name, true_tvt, pred, len(baseline), baselines))
        for bucket_name in ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]:
            mask = bucket.to_numpy() == bucket_name
            if not np.any(mask):
                continue
            row = metric_row(
                name,
                true_tvt[mask],
                pred[mask],
                int(mask.sum()),
                {base_name: base_pred[mask] for base_name, base_pred in baselines.items()},
            )
            row["bucket"] = bucket_name
            bucket_rows.append(row)
        for subgroup_name, mask in hidden_masks.items():
            if not np.any(mask):
                continue
            row = metric_row(
                name,
                true_tvt[mask],
                pred[mask],
                int(mask.sum()),
                {base_name: base_pred[mask] for base_name, base_pred in baselines.items()},
            )
            row["subgroup"] = subgroup_name
            subgroup_rows.append(row)
        work = pd.DataFrame(
            {
                "well": baseline["well"].astype(str),
                "true_tvt": true_tvt,
                "pred": pred,
            }
        )
        for base_name, base_pred in baselines.items():
            work[base_name] = base_pred
        for well, group in work.groupby("well", sort=False):
            row = {
                "candidate": name,
                "well": well,
                "rows": int(len(group)),
                "rmse": rmse(group["true_tvt"].to_numpy(), group["pred"].to_numpy()),
            }
            for base_name in baselines:
                base_rmse = rmse(group["true_tvt"].to_numpy(), group[base_name].to_numpy())
                row[f"{base_name}_rmse"] = base_rmse
                row[f"delta_rmse_vs_{base_name}"] = row["rmse"] - base_rmse
            by_well_rows.append(row)

    overall = pd.DataFrame(overall_rows).sort_values("rmse")
    bucket_metrics = pd.DataFrame(bucket_rows)
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    by_well = pd.DataFrame(by_well_rows)

    uncertainty_frames: list[pd.DataFrame] = []
    for column in hmm_mean_columns:
        candidate = column.removesuffix("_mean_tvt")
        std_column = f"{candidate}_std"
        if std_column not in hmm.columns:
            continue
        uncertainty_frames.append(
            compute_uncertainty_bins(
                candidate,
                pd.to_numeric(hmm[std_column], errors="coerce").to_numpy(np.float64),
                true_tvt,
                predictions[candidate],
                int(comparison.get("uncertainty_bins", 10)),
            )
        )
    uncertainty_bins = pd.concat(uncertainty_frames, ignore_index=True) if uncertainty_frames else pd.DataFrame()
    step_delta_rates = compute_step_delta_rates(baseline, predictions, thresholds)

    enriched_path = paths.artifacts_dir / f"{output_prefix}_enriched_predictions.csv.gz"
    enriched = pd.DataFrame(
        {
            "id": baseline["id"].astype(str),
            "well": baseline["well"].astype(str),
            "target": baseline["target"],
            "last_known_tvt": baseline["last_known_tvt"],
            "md_since": baseline["md_since"],
        }
    )
    for name, pred in predictions.items():
        if name.startswith("hmm_lgb") or name in lgb_predictions:
            enriched[name] = pred
    if bool(comparison.get("write_enriched_cache", False)):
        enriched.to_csv(enriched_path, index=False, compression="gzip")

    overall_path = paths.artifacts_dir / f"{output_prefix}_overall_metrics.csv"
    bucket_path = paths.artifacts_dir / f"{output_prefix}_distance_bucket_metrics.csv"
    subgroup_path = paths.artifacts_dir / f"{output_prefix}_hidden_like_metrics.csv"
    by_well_path = paths.artifacts_dir / f"{output_prefix}_by_well_delta.csv"
    uncertainty_path = paths.artifacts_dir / f"{output_prefix}_hmm_std_calibration.csv"
    step_delta_path = paths.artifacts_dir / f"{output_prefix}_step_delta_rates.csv"
    summary_path = paths.artifacts_dir / f"{output_prefix}_summary.json"
    overall.to_csv(overall_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    subgroup_metrics.to_csv(subgroup_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    uncertainty_bins.to_csv(uncertainty_path, index=False)
    step_delta_rates.to_csv(step_delta_path, index=False)

    best = overall.iloc[0].to_dict() if len(overall) else {}
    hmm_overall = overall[overall["candidate"].astype(str).str.startswith("hmm_lgb")]
    best_hmm = hmm_overall.iloc[0].to_dict() if len(hmm_overall) else {}
    summary = {
        "experiment": EXPERIMENT_NAME,
        "baseline": baseline_source,
        "baseline_load_mode": baseline_load_mode,
        "hmm_feature_cache": hmm_source,
        "hmm_load_mode": hmm_load_mode,
        "lgb_prediction_sources": lgb_metadata,
        "rows_checked": int(len(baseline)),
        "unique_wells": int(baseline["well"].nunique()),
        "id_mismatches": id_mismatches,
        "prediction_candidates": list(predictions.keys()),
        "primary_baselines": list(baselines.keys()),
        "reference_train_side_metrics": dict(comparison.get("reference_train_side_metrics") or {}),
        "best_candidate": best,
        "best_hmm_lgb_candidate": best_hmm,
        "overall_metrics": overall.to_dict(orient="records"),
        "hidden_like_metrics_available": bool(len(subgroup_metrics)),
        "artifacts": {
            "overall_metrics": str(overall_path),
            "distance_bucket_metrics": str(bucket_path),
            "hidden_like_metrics": str(subgroup_path),
            "by_well_delta": str(by_well_path),
            "hmm_std_calibration": str(uncertainty_path),
            "step_delta_rates": str(step_delta_path),
            "summary": str(summary_path),
            "enriched_cache": str(enriched_path) if bool(comparison.get("write_enriched_cache", False)) else None,
        },
        "sha256": {
            "overall_metrics": sha256_path(overall_path),
            "distance_bucket_metrics": sha256_path(bucket_path),
            "hidden_like_metrics": sha256_path(subgroup_path),
            "by_well_delta": sha256_path(by_well_path),
            "hmm_std_calibration": sha256_path(uncertainty_path),
            "step_delta_rates": sha256_path(step_delta_path),
        },
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_direct_comparison()
