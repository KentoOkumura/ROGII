from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def find_first_existing(candidates: list[str | Path], label: str) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        checked.append(str(path))
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No existing {label} path. Checked: {checked}")


def candidate_paths(config: dict[str, Any], key: str) -> list[str]:
    value = get_nested(config, f"audit.candidates.{key}")
    if not isinstance(value, list) or not value:
        raise KeyError(f"audit.candidates.{key} must be a non-empty list")
    return [str(item) for item in value]


def read_csv_columns(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if usecols is None:
        return pd.read_csv(path)
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    selected = [column for column in usecols if column in available]
    missing = sorted(set(usecols) - available)
    if missing:
        print(f"[warn] missing columns in {path.name}: {missing}", flush=True)
    return pd.read_csv(path, usecols=selected)


def load_policy_predictions(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    baseline_policy = str(get_nested(config, "audit.baseline_policy"))
    compare_policy = str(get_nested(config, "audit.compare_policy"))
    max_rows = get_nested(config, "audit.max_rows")
    policies = [baseline_policy, compare_policy]

    usecols = [
        "id",
        "well",
        "policy",
        "target_tvt",
        "last_known_tvt",
        "target_delta",
        "pred_delta",
        "pred_tvt",
    ]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        selected = chunk[chunk["policy"].isin(policies)].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"No rows for policies={policies} in {path}")
    frame = pd.concat(chunks, ignore_index=True)
    if max_rows:
        # Keep paired policy rows for the same ids.
        id_subset = (
            frame.loc[frame["policy"] == baseline_policy, "id"]
            .drop_duplicates()
            .head(int(max_rows))
        )
        frame = frame[frame["id"].isin(set(id_subset))].copy()

    required = {
        "id",
        "well",
        "policy",
        "target_tvt",
        "last_known_tvt",
        "target_delta",
        "pred_delta",
        "pred_tvt",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing policy prediction columns: {missing}")
    return frame


def pivot_policy_predictions(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    baseline_policy = str(get_nested(config, "audit.baseline_policy"))
    compare_policy = str(get_nested(config, "audit.compare_policy"))

    base = frame[frame["policy"] == baseline_policy].copy()
    compare = frame[frame["policy"] == compare_policy].copy()
    if base.empty:
        raise ValueError(f"No baseline policy rows: {baseline_policy}")
    if compare.empty:
        raise ValueError(f"No compare policy rows: {compare_policy}")

    keep = ["id", "pred_delta", "pred_tvt"]
    compare = compare[keep].rename(
        columns={"pred_delta": "compare_pred_delta", "pred_tvt": "compare_pred_tvt"}
    )
    base = base.rename(
        columns={"pred_delta": "baseline_pred_delta", "pred_tvt": "baseline_pred_tvt"}
    )
    merged = base.merge(compare, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(base):
        raise ValueError(
            f"Policy row mismatch: baseline rows={len(base)} joined rows={len(merged)}"
        )

    merged["baseline_error"] = merged["baseline_pred_tvt"] - merged["target_tvt"]
    merged["compare_error"] = merged["compare_pred_tvt"] - merged["target_tvt"]
    merged["baseline_abs_error"] = merged["baseline_error"].abs()
    merged["compare_abs_error"] = merged["compare_error"].abs()
    merged["baseline_squared_error"] = np.square(merged["baseline_error"])
    merged["compare_squared_error"] = np.square(merged["compare_error"])
    merged["squared_error_delta"] = (
        merged["compare_squared_error"] - merged["baseline_squared_error"]
    )
    return merged


def load_feature_importance(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    top_n = int(get_nested(config, "audit.top_features") or 30)
    frame = pd.read_csv(path)
    required = {"feature", "gain_mean", "split_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing feature importance columns: {missing}")
    frame = frame.sort_values(["gain_mean", "split_mean"], ascending=False).reset_index(drop=True)
    frame["importance_rank"] = np.arange(1, len(frame) + 1)
    return frame.head(top_n).copy()


def selected_feature_columns(importance: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    required = get_nested(config, "audit.required_feature_columns") or []
    if not isinstance(required, list):
        required = []
    columns = ["id", "well"]
    for feature in list(importance["feature"]) + [str(value) for value in required]:
        if feature not in columns:
            columns.append(feature)
    return columns


def add_distance_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    distance_column = "md_since" if "md_since" in out.columns else "md_from_ps"
    distance = pd.to_numeric(out.get(distance_column), errors="coerce")
    bins = [-np.inf, 50, 250, 1000, 2500, np.inf]
    labels = ["0000_0050", "0050_0250", "0250_1000", "1000_2500", "2500_plus"]
    out["distance_bucket"] = pd.cut(distance, bins=bins, labels=labels)
    out["distance_bucket"] = out["distance_bucket"].astype("string").fillna("unknown")
    return out


def aggregate_policy_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    baseline_policy = str(get_nested(config, "audit.baseline_policy"))
    compare_policy = str(get_nested(config, "audit.compare_policy"))
    rows = []
    for policy, error_col, abs_col, sq_col in [
        (baseline_policy, "baseline_error", "baseline_abs_error", "baseline_squared_error"),
        (compare_policy, "compare_error", "compare_abs_error", "compare_squared_error"),
    ]:
        rows.append(
            {
                "policy": policy,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "rmse_tvt": rmse(frame[error_col]),
                "mae_tvt": float(frame[abs_col].mean()),
                "error_mean": float(frame[error_col].mean()),
                "sse": float(frame[sq_col].sum()),
            }
        )
    metrics = pd.DataFrame(rows)
    baseline_rmse = float(metrics.loc[metrics["policy"] == baseline_policy, "rmse_tvt"].iloc[0])
    metrics["rmse_delta_vs_baseline"] = metrics["rmse_tvt"] - baseline_rmse
    return metrics


def quantile_metrics_for_feature(
    frame: pd.DataFrame,
    feature: str,
    importance_row: pd.Series,
    quantile_bins: int,
) -> pd.DataFrame:
    values = pd.to_numeric(frame[feature], errors="coerce")
    valid = frame.loc[values.notna()].copy()
    valid["_feature_value"] = values.loc[values.notna()].astype(float)
    if valid.empty:
        return pd.DataFrame()
    try:
        valid["_feature_bin"] = pd.qcut(
            valid["_feature_value"],
            q=quantile_bins,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = valid.groupby("_feature_bin", observed=False)
    for bin_id, (bin_value, group) in enumerate(grouped, start=1):
        if group.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "importance_rank": int(importance_row["importance_rank"]),
                "gain_mean": float(importance_row["gain_mean"]),
                "split_mean": float(importance_row["split_mean"]),
                "bin_id": bin_id,
                "feature_bin": str(bin_value),
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
                "feature_min": float(group["_feature_value"].min()),
                "feature_max": float(group["_feature_value"].max()),
                "feature_mean": float(group["_feature_value"].mean()),
                "baseline_rmse": rmse(group["baseline_error"]),
                "compare_rmse": rmse(group["compare_error"]),
                "baseline_mae": float(group["baseline_abs_error"].mean()),
                "compare_mae": float(group["compare_abs_error"].mean()),
                "baseline_error_mean": float(group["baseline_error"].mean()),
                "compare_error_mean": float(group["compare_error"].mean()),
                "sse_delta": float(group["squared_error_delta"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["compare_rmse_delta"] = out["compare_rmse"] - out["baseline_rmse"]
        out["compare_mae_delta"] = out["compare_mae"] - out["baseline_mae"]
    return out


def feature_correlations(
    frame: pd.DataFrame,
    features: list[str],
    sample_rows: int | None,
) -> pd.DataFrame:
    if sample_rows and len(frame) > sample_rows:
        work = frame.sample(n=int(sample_rows), random_state=42)
    else:
        work = frame

    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(work[feature], errors="coerce")
        valid = values.notna()
        if int(valid.sum()) < 3:
            continue
        x = values.loc[valid]
        abs_error = work.loc[valid, "baseline_abs_error"]
        signed_error = work.loc[valid, "baseline_error"]
        squared_error = work.loc[valid, "baseline_squared_error"]
        rows.append(
            {
                "feature": feature,
                "rows": int(valid.sum()),
                "pearson_abs_error": float(x.corr(abs_error, method="pearson")),
                "spearman_abs_error": float(x.corr(abs_error, method="spearman")),
                "pearson_signed_error": float(x.corr(signed_error, method="pearson")),
                "spearman_signed_error": float(x.corr(signed_error, method="spearman")),
                "pearson_squared_error": float(x.corr(squared_error, method="pearson")),
                "spearman_squared_error": float(x.corr(squared_error, method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def build_feature_readout(
    frame: pd.DataFrame,
    importance: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [feature for feature in importance["feature"] if feature in frame.columns]
    quantile_bins = int(get_nested(config, "audit.quantile_bins") or 5)
    sample_rows = get_nested(config, "audit.correlation_sample_rows")
    sample_rows = int(sample_rows) if sample_rows else None

    quantile_frames = []
    for _, importance_row in importance.iterrows():
        feature = str(importance_row["feature"])
        if feature not in frame.columns:
            continue
        quantile_frames.append(
            quantile_metrics_for_feature(frame, feature, importance_row, quantile_bins)
        )
    quantiles = (
        pd.concat([item for item in quantile_frames if not item.empty], ignore_index=True)
        if quantile_frames
        else pd.DataFrame()
    )

    correlations = feature_correlations(frame, features, sample_rows)
    summary_rows: list[dict[str, Any]] = []
    global_abs_error = float(frame["baseline_abs_error"].mean())
    global_rmse = rmse(frame["baseline_error"])
    for _, importance_row in importance.iterrows():
        feature = str(importance_row["feature"])
        feature_quantiles = (
            quantiles[quantiles["feature"] == feature]
            if not quantiles.empty
            else pd.DataFrame()
        )
        if feature_quantiles.empty:
            worst_bin = pd.Series(dtype=object)
            high_bin = pd.Series(dtype=object)
        else:
            worst_bin = feature_quantiles.sort_values("baseline_mae", ascending=False).iloc[0]
            high_bin = feature_quantiles.sort_values("bin_id", ascending=False).iloc[0]
        corr_row = (
            correlations[correlations["feature"] == feature].iloc[0]
            if not correlations.empty and feature in set(correlations["feature"])
            else pd.Series(dtype=object)
        )
        summary_rows.append(
            {
                "feature": feature,
                "importance_rank": int(importance_row["importance_rank"]),
                "gain_mean": float(importance_row["gain_mean"]),
                "split_mean": float(importance_row["split_mean"]),
                "gain_std": float(importance_row.get("gain_std", np.nan)),
                "folds": int(importance_row.get("folds", 0)),
                "models": int(importance_row.get("models", 0)),
                "pearson_abs_error": float(corr_row.get("pearson_abs_error", np.nan)),
                "spearman_abs_error": float(corr_row.get("spearman_abs_error", np.nan)),
                "pearson_squared_error": float(corr_row.get("pearson_squared_error", np.nan)),
                "spearman_squared_error": float(corr_row.get("spearman_squared_error", np.nan)),
                "worst_bin": str(worst_bin.get("feature_bin", "")),
                "worst_bin_rows": int(worst_bin.get("rows", 0) or 0),
                "worst_bin_baseline_mae": float(worst_bin.get("baseline_mae", np.nan)),
                "worst_bin_baseline_rmse": float(worst_bin.get("baseline_rmse", np.nan)),
                "worst_bin_mae_lift_vs_global": float(
                    worst_bin.get("baseline_mae", np.nan) - global_abs_error
                ),
                "worst_bin_rmse_lift_vs_global": float(
                    worst_bin.get("baseline_rmse", np.nan) - global_rmse
                ),
                "worst_bin_compare_rmse_delta": float(
                    worst_bin.get("compare_rmse_delta", np.nan)
                ),
                "high_bin": str(high_bin.get("feature_bin", "")),
                "high_bin_baseline_mae": float(high_bin.get("baseline_mae", np.nan)),
                "high_bin_baseline_rmse": float(high_bin.get("baseline_rmse", np.nan)),
                "high_bin_mae_lift_vs_global": float(
                    high_bin.get("baseline_mae", np.nan) - global_abs_error
                ),
                "high_bin_compare_rmse_delta": float(high_bin.get("compare_rmse_delta", np.nan)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return summary, quantiles


def build_well_summary(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("id", "size"),
        "baseline_rmse": ("baseline_error", lambda value: rmse(value)),
        "compare_rmse": ("compare_error", lambda value: rmse(value)),
        "baseline_mae": ("baseline_abs_error", "mean"),
        "compare_mae": ("compare_abs_error", "mean"),
        "baseline_error_mean": ("baseline_error", "mean"),
        "sse_delta": ("squared_error_delta", "sum"),
    }
    for feature in features[:12]:
        aggregations[f"{feature}_mean"] = (feature, "mean")
        aggregations[f"{feature}_std"] = (feature, "std")
    out = frame.groupby("well", observed=False).agg(**aggregations).reset_index()
    out["compare_rmse_delta"] = out["compare_rmse"] - out["baseline_rmse"]
    return out.sort_values("baseline_rmse", ascending=False)


def write_plots(
    output_dir: Path,
    prefix: str,
    feature_summary: pd.DataFrame,
) -> dict[str, str | None]:
    artifacts: dict[str, str | None] = {
        "feature_error_lift_plot": None,
        "feature_error_correlation_plot": None,
    }
    if feature_summary.empty:
        return artifacts

    top = feature_summary.sort_values("gain_mean", ascending=False).head(20).copy()
    lift_path = output_dir / f"{prefix}_feature_error_lift_top20.png"
    fig_height = max(5.0, 0.32 * len(top))
    plt.figure(figsize=(10, fig_height))
    plt.barh(top["feature"], top["worst_bin_mae_lift_vs_global"], color="#4c78a8")
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Worst quantile MAE lift vs global baseline")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(lift_path, dpi=160)
    plt.close()
    artifacts["feature_error_lift_plot"] = lift_path.name

    corr_path = output_dir / f"{prefix}_feature_error_correlation_top20.png"
    corr = top.copy()
    corr["spearman_abs_error"] = corr["spearman_abs_error"].replace([np.inf, -np.inf], np.nan)
    plt.figure(figsize=(10, fig_height))
    plt.barh(corr["feature"], corr["spearman_abs_error"], color="#f58518")
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Spearman correlation with baseline absolute error")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=160)
    plt.close()
    artifacts["feature_error_correlation_plot"] = corr_path.name
    return artifacts


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    return value


def run_oof_feature_importance_error_readout(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    paths = ExperimentPaths()
    config = paths.config
    paths.ensure_output_dirs()
    out_dir = Path(output_dir) if output_dir else paths.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(
        get_nested(config, "audit.output_prefix")
        or "exp086_oof_feature_importance_error_readout"
    )

    prediction_path = find_first_existing(
        candidate_paths(config, "exp077_policy_predictions"),
        "exp077 policy predictions",
    )
    metrics_path = find_first_existing(
        candidate_paths(config, "exp077_policy_metrics"),
        "exp077 policy metrics",
    )
    importance_path = find_first_existing(
        candidate_paths(config, "exp077_feature_importance_mean"),
        "exp077 feature importance mean",
    )
    feature_cache_path = find_first_existing(
        candidate_paths(config, "exp072_feature_cache"),
        "exp072 feature cache",
    )

    policy_predictions = load_policy_predictions(prediction_path, config)
    frame = pivot_policy_predictions(policy_predictions, config)
    importance = load_feature_importance(importance_path, config)
    feature_columns = selected_feature_columns(importance, config)
    features = read_csv_columns(feature_cache_path, feature_columns)

    merge_columns = [column for column in features.columns if column != "well"]
    frame = frame.merge(features[merge_columns], on="id", how="left", validate="one_to_one")
    if importance.empty:
        missing_features = 0
    else:
        first_feature = str(importance["feature"].head(1).iloc[0])
        missing_features = int(frame[first_feature].isna().sum())
    frame = add_distance_bucket(frame)

    policy_metrics = aggregate_policy_metrics(frame, config)
    feature_summary, feature_quantiles = build_feature_readout(frame, importance, config)
    well_summary = build_well_summary(
        frame,
        [feature for feature in importance["feature"] if feature in frame.columns],
    )

    policy_metrics_path = out_dir / f"{prefix}_policy_metrics.csv"
    feature_summary_path = out_dir / f"{prefix}_feature_summary.csv"
    feature_quantiles_path = out_dir / f"{prefix}_feature_quantile_metrics.csv"
    well_summary_path = out_dir / f"{prefix}_well_summary.csv"

    policy_metrics.to_csv(policy_metrics_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)
    feature_quantiles.to_csv(feature_quantiles_path, index=False)
    well_summary.to_csv(well_summary_path, index=False)
    plots = write_plots(out_dir, prefix, feature_summary)

    top_lift = (
        feature_summary.sort_values("worst_bin_mae_lift_vs_global", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    top_corr = (
        feature_summary.reindex(
            feature_summary["spearman_abs_error"].abs().sort_values(ascending=False).index
        )
        .head(10)
        .to_dict(orient="records")
    )

    summary = {
        "experiment": "exp086_oof_feature_importance_error_readout",
        "status": "readout_completed",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "readout_parent": "exp077_full_replay_postprocess_guard",
        "mode": get_nested(config, "audit.mode"),
        "inputs": {
            "policy_predictions": str(prediction_path),
            "policy_predictions_sha256": sha256_file(prediction_path),
            "policy_metrics": str(metrics_path),
            "policy_metrics_sha256": sha256_file(metrics_path),
            "feature_importance_mean": str(importance_path),
            "feature_importance_mean_sha256": sha256_file(importance_path),
            "feature_cache": str(feature_cache_path),
            "feature_cache_sha256": sha256_file(feature_cache_path),
        },
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "features_read": int(len(feature_columns) - 2),
        "missing_top_feature_rows": missing_features,
        "baseline_policy": get_nested(config, "audit.baseline_policy"),
        "compare_policy": get_nested(config, "audit.compare_policy"),
        "best_policy_metric": policy_metrics.sort_values("rmse_tvt").iloc[0].to_dict(),
        "top_error_lift_features": top_lift,
        "top_error_correlation_features": top_corr,
        "artifacts": {
            "policy_metrics": policy_metrics_path.name,
            "feature_summary": feature_summary_path.name,
            "feature_quantile_metrics": feature_quantiles_path.name,
            "well_summary": well_summary_path.name,
            **plots,
            "summary": f"{prefix}_summary.json",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(clean_json(summary), indent=2))
    print(json.dumps(clean_json(summary), indent=2), flush=True)
    return summary


def main() -> None:
    run_oof_feature_importance_error_readout()


if __name__ == "__main__":
    main()
