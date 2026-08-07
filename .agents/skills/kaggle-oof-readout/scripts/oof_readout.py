#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        config = yaml.safe_load(fp) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def first_existing(candidates: list[str], label: str) -> Path:
    checked = []
    for item in candidates:
        path = Path(item)
        checked.append(str(path))
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No usable {label} path. Checked: {checked}")


def read_policy_rows(path: Path, policies: list[str], chunksize: int) -> pd.DataFrame:
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
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        selected = chunk[chunk["policy"].isin(policies)]
        if not selected.empty:
            chunks.append(selected.copy())
    if not chunks:
        raise ValueError(f"No policy rows found for {policies} in {path}")
    return pd.concat(chunks, ignore_index=True)


def pivot_policy_rows(frame: pd.DataFrame, baseline: str, compare: str) -> pd.DataFrame:
    base = frame[frame["policy"] == baseline].copy()
    comp = frame[frame["policy"] == compare].copy()
    if base.empty or comp.empty:
        raise ValueError(f"Missing baseline or compare rows: {baseline}, {compare}")
    comp = comp[["id", "pred_delta", "pred_tvt"]].rename(
        columns={"pred_delta": "compare_pred_delta", "pred_tvt": "compare_pred_tvt"}
    )
    base = base.rename(
        columns={"pred_delta": "baseline_pred_delta", "pred_tvt": "baseline_pred_tvt"}
    )
    out = base.merge(comp, on="id", how="inner", validate="one_to_one")
    out["baseline_error"] = out["baseline_pred_tvt"] - out["target_tvt"]
    out["compare_error"] = out["compare_pred_tvt"] - out["target_tvt"]
    out["baseline_abs_error"] = out["baseline_error"].abs()
    out["compare_abs_error"] = out["compare_error"].abs()
    out["baseline_squared_error"] = np.square(out["baseline_error"])
    out["compare_squared_error"] = np.square(out["compare_error"])
    out["squared_error_delta"] = out["compare_squared_error"] - out["baseline_squared_error"]
    return out


def selected_features(config: dict[str, Any], importance: pd.DataFrame) -> list[str]:
    top_n = int(nested(config, "readout.top_features", 30))
    required = nested(config, "readout.required_feature_columns", []) or []
    ranked = (
        importance.sort_values(["gain_mean", "split_mean"], ascending=False)
        .head(top_n)["feature"]
        .astype(str)
        .tolist()
    )
    columns = ["id", "well"]
    for column in ranked + [str(item) for item in required]:
        if column not in columns:
            columns.append(column)
    return columns


def read_feature_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    usecols = [column for column in columns if column in available]
    missing = sorted(set(columns) - available)
    if missing:
        print(f"[warn] missing feature columns: {missing}", flush=True)
    return pd.read_csv(path, usecols=usecols)


def policy_metrics(frame: pd.DataFrame, baseline: str, compare: str) -> pd.DataFrame:
    rows = []
    for policy, err, abs_err, sq_err in [
        (baseline, "baseline_error", "baseline_abs_error", "baseline_squared_error"),
        (compare, "compare_error", "compare_abs_error", "compare_squared_error"),
    ]:
        rows.append(
            {
                "policy": policy,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "rmse_tvt": rmse(frame[err]),
                "mae_tvt": float(frame[abs_err].mean()),
                "error_mean": float(frame[err].mean()),
                "sse": float(frame[sq_err].sum()),
            }
        )
    out = pd.DataFrame(rows)
    base_rmse = float(out.loc[out["policy"] == baseline, "rmse_tvt"].iloc[0])
    out["rmse_delta_vs_baseline"] = out["rmse_tvt"] - base_rmse
    return out


def feature_quantiles(
    frame: pd.DataFrame,
    importance: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_mae = float(frame["baseline_abs_error"].mean())
    global_rmse = rmse(frame["baseline_error"])
    for _, item in importance.iterrows():
        feature = str(item["feature"])
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = frame.loc[values.notna()].copy()
        if valid.empty:
            continue
        valid["_value"] = values.loc[values.notna()].astype(float)
        try:
            valid["_bin"] = pd.qcut(valid["_value"], q=bins, duplicates="drop")
        except ValueError:
            continue
        for bin_id, (label, group) in enumerate(valid.groupby("_bin", observed=False), start=1):
            if group.empty:
                continue
            baseline_rmse = rmse(group["baseline_error"])
            compare_rmse = rmse(group["compare_error"])
            baseline_mae = float(group["baseline_abs_error"].mean())
            rows.append(
                {
                    "feature": feature,
                    "importance_rank": int(item["importance_rank"]),
                    "gain_mean": float(item["gain_mean"]),
                    "split_mean": float(item["split_mean"]),
                    "bin_id": bin_id,
                    "feature_bin": str(label),
                    "rows": int(len(group)),
                    "wells": int(group["well"].nunique()),
                    "baseline_rmse": baseline_rmse,
                    "compare_rmse": compare_rmse,
                    "baseline_mae": baseline_mae,
                    "compare_mae": float(group["compare_abs_error"].mean()),
                    "baseline_error_mean": float(group["baseline_error"].mean()),
                    "compare_error_mean": float(group["compare_error"].mean()),
                    "mae_lift_vs_global": baseline_mae - global_mae,
                    "rmse_lift_vs_global": baseline_rmse - global_rmse,
                    "compare_rmse_delta": compare_rmse - baseline_rmse,
                    "sse_delta": float(group["squared_error_delta"].sum()),
                }
            )
    return pd.DataFrame(rows)


def feature_summary(
    frame: pd.DataFrame,
    importance: pd.DataFrame,
    quantiles: pd.DataFrame,
    sample_rows: int,
) -> pd.DataFrame:
    sample = frame.sample(n=sample_rows, random_state=42) if len(frame) > sample_rows else frame
    rows = []
    for _, item in importance.iterrows():
        feature = str(item["feature"])
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(sample[feature], errors="coerce")
        valid = values.notna()
        q = quantiles[quantiles["feature"] == feature]
        worst = q.sort_values("baseline_mae", ascending=False).head(1)
        worst_row = worst.iloc[0] if not worst.empty else pd.Series(dtype=object)
        rows.append(
            {
                "feature": feature,
                "importance_rank": int(item["importance_rank"]),
                "gain_mean": float(item["gain_mean"]),
                "split_mean": float(item["split_mean"]),
                "spearman_abs_error": float(
                    values.loc[valid].corr(
                        sample.loc[valid, "baseline_abs_error"],
                        method="spearman",
                    )
                )
                if int(valid.sum()) >= 3
                else float("nan"),
                "pearson_abs_error": float(
                    values.loc[valid].corr(
                        sample.loc[valid, "baseline_abs_error"],
                        method="pearson",
                    )
                )
                if int(valid.sum()) >= 3
                else float("nan"),
                "worst_bin": str(worst_row.get("feature_bin", "")),
                "worst_bin_rows": int(worst_row.get("rows", 0) or 0),
                "worst_bin_baseline_mae": float(worst_row.get("baseline_mae", np.nan)),
                "worst_bin_baseline_rmse": float(worst_row.get("baseline_rmse", np.nan)),
                "worst_bin_mae_lift_vs_global": float(
                    worst_row.get("mae_lift_vs_global", np.nan)
                ),
                "worst_bin_compare_rmse_delta": float(
                    worst_row.get("compare_rmse_delta", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def well_summary(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, Any]] = {
        "rows": ("id", "size"),
        "baseline_rmse": ("baseline_error", rmse),
        "compare_rmse": ("compare_error", rmse),
        "baseline_mae": ("baseline_abs_error", "mean"),
        "compare_mae": ("compare_abs_error", "mean"),
        "sse_delta": ("squared_error_delta", "sum"),
    }
    for feature in features[:12]:
        aggregations[f"{feature}_mean"] = (feature, "mean")
        aggregations[f"{feature}_std"] = (feature, "std")
    out = frame.groupby("well", observed=False).agg(**aggregations).reset_index()
    out["compare_rmse_delta"] = out["compare_rmse"] - out["baseline_rmse"]
    return out.sort_values("baseline_rmse", ascending=False)


def write_plots(output_dir: Path, prefix: str, summary: pd.DataFrame) -> dict[str, str | None]:
    import matplotlib.pyplot as plt

    result: dict[str, str | None] = {"error_lift_plot": None, "error_correlation_plot": None}
    if summary.empty:
        return result
    top = summary.sort_values("worst_bin_mae_lift_vs_global", ascending=False).head(20)
    height = max(5.0, 0.32 * len(top))
    lift_path = output_dir / f"{prefix}_feature_error_lift_top20.png"
    plt.figure(figsize=(10, height))
    plt.barh(top["feature"], top["worst_bin_mae_lift_vs_global"])
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(lift_path, dpi=160)
    plt.close()
    result["error_lift_plot"] = lift_path.name

    corr_order = summary["spearman_abs_error"].abs().sort_values(ascending=False).index
    corr = summary.reindex(corr_order).head(20)
    corr_path = output_dir / f"{prefix}_feature_error_correlation_top20.png"
    plt.figure(figsize=(10, height))
    plt.barh(corr["feature"], corr["spearman_abs_error"])
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(corr_path, dpi=160)
    plt.close()
    result["error_correlation_plot"] = corr_path.name
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = read_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(nested(config, "readout.output_prefix", "oof_readout"))
    baseline = str(nested(config, "readout.baseline_policy"))
    compare = str(nested(config, "readout.compare_policy"))

    prediction_path = first_existing(
        nested(config, "paths.policy_predictions"),
        "policy predictions",
    )
    importance_path = first_existing(
        nested(config, "paths.feature_importance"),
        "feature importance",
    )
    feature_cache_path = first_existing(nested(config, "paths.feature_cache"), "feature cache")

    policy_rows = read_policy_rows(
        prediction_path,
        [baseline, compare],
        int(nested(config, "readout.chunksize", 1_000_000)),
    )
    frame = pivot_policy_rows(policy_rows, baseline, compare)

    importance = pd.read_csv(importance_path)
    importance = importance.sort_values(
        ["gain_mean", "split_mean"],
        ascending=False,
    ).reset_index(drop=True)
    importance["importance_rank"] = np.arange(1, len(importance) + 1)
    importance = importance.head(int(nested(config, "readout.top_features", 30))).copy()

    feature_columns = selected_features(config, importance)
    features = read_feature_cache(feature_cache_path, feature_columns)
    frame = frame.merge(
        features[[column for column in features.columns if column != "well"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    metrics = policy_metrics(frame, baseline, compare)
    quantiles = feature_quantiles(
        frame,
        importance,
        int(nested(config, "readout.quantile_bins", 5)),
    )
    summary = feature_summary(
        frame,
        importance,
        quantiles,
        int(nested(config, "readout.correlation_sample_rows", 500_000)),
    )
    well_features = [
        str(item)
        for item in importance["feature"]
        if item in frame.columns
    ]
    wells = well_summary(frame, well_features)

    metrics_path = args.output_dir / f"{prefix}_policy_metrics.csv"
    quantiles_path = args.output_dir / f"{prefix}_feature_quantile_metrics.csv"
    summary_path = args.output_dir / f"{prefix}_feature_summary.csv"
    wells_path = args.output_dir / f"{prefix}_well_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    quantiles.to_csv(quantiles_path, index=False)
    summary.to_csv(summary_path, index=False)
    wells.to_csv(wells_path, index=False)
    plots = write_plots(args.output_dir, prefix, summary)

    run_summary = {
        "status": "readout_completed",
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "baseline_policy": baseline,
        "compare_policy": compare,
        "best_policy_metric": metrics.sort_values("rmse_tvt").iloc[0].to_dict(),
        "inputs": {
            "policy_predictions": str(prediction_path),
            "policy_predictions_sha256": sha256_file(prediction_path),
            "feature_importance": str(importance_path),
            "feature_importance_sha256": sha256_file(importance_path),
            "feature_cache": str(feature_cache_path),
            "feature_cache_sha256": sha256_file(feature_cache_path),
        },
        "artifacts": {
            "policy_metrics": metrics_path.name,
            "feature_quantile_metrics": quantiles_path.name,
            "feature_summary": summary_path.name,
            "well_summary": wells_path.name,
            **plots,
        },
    }
    (args.output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(clean_json(run_summary), indent=2)
    )
    print(json.dumps(clean_json(run_summary), indent=2))


if __name__ == "__main__":
    main()
