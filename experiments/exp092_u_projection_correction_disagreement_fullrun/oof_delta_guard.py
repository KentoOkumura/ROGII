from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EXP073_PREDICTIONS = Path(
    "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay"
    "/train_v2/artifacts/exp063_full_replay_repro_guard_predictions.csv.gz"
)
DEFAULT_EXP077_PREDICTIONS = Path(
    "/tmp/kaggle-output/exp077-full-replay-postprocess-guard-train-v1"
    "/artifacts/exp077_full_replay_postprocess_guard_predictions.csv.gz"
)
DEFAULT_EXP092_PREDICTIONS = (
    Path("experiments")
    / "exp092_u_projection_correction_disagreement_fullrun"
    / "kaggle"
    / "output"
    / "train"
    / "artifacts"
    / "exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz"
)
DEFAULT_FEATURE_CACHE = Path(
    "/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1/artifacts/"
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
DEFAULT_OUTPUT_DIR = (
    Path("experiments")
    / "exp092_u_projection_correction_disagreement_fullrun"
    / "artifacts"
    / "oof_delta_guard"
)

MODEL_SPECS = {
    "exp073_lgb_mean": {
        "path_key": "exp073",
        "selector_col": "model",
        "selector_value": "lgb_mean",
    },
    "exp077_policy": {
        "path_key": "exp077",
        "selector_col": "policy",
        "selector_value": "longtail_likpf_tiny_gate_w006",
    },
    "exp092_lgb1": {
        "path_key": "exp092",
        "selector_col": "model",
        "selector_value": "lgb1",
    },
    "exp092_lgb2": {
        "path_key": "exp092",
        "selector_col": "model",
        "selector_value": "lgb2",
    },
    "exp092_lgb_mean": {
        "path_key": "exp092",
        "selector_col": "model",
        "selector_value": "lgb_mean",
    },
    "exp092_lgb0": {
        "path_key": "exp092",
        "selector_col": "model",
        "selector_value": "lgb0",
    },
}

PATH_MODELS = {
    "pred_exp073": MODEL_SPECS["exp073_lgb_mean"],
    "pred_exp077_policy": MODEL_SPECS["exp077_policy"],
    "pred_exp092_lgb1": MODEL_SPECS["exp092_lgb1"],
    "pred_exp092_lgb2": MODEL_SPECS["exp092_lgb2"],
    "pred_exp092_lgb_mean": MODEL_SPECS["exp092_lgb_mean"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_path(path: Path, label: str) -> Path:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return path


def has_nonempty_path(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def parse_tail_rank(ids: pd.Series) -> pd.Series:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="raise").astype("int32")


def distance_bucket(values: pd.Series) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
            labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
            include_lowest=True,
        )
        .astype("string")
        .fillna("unknown")
    )


def tail_rank_bucket(values: pd.Series) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=[-np.inf, 99, 249, 499, 999, np.inf],
            labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
            include_lowest=True,
        )
        .astype("string")
        .fillna("unknown")
    )


def empty_stat() -> dict[str, Any]:
    return {"rows": 0, "sse": 0.0, "sae": 0.0, "wells": set()}


def update_stat(store: dict[Any, dict[str, Any]], key: Any, subset: pd.DataFrame) -> None:
    stat = store.setdefault(key, empty_stat())
    errors = pd.to_numeric(subset["pred_tvt"], errors="raise").to_numpy(np.float64) - pd.to_numeric(
        subset["target_tvt"], errors="raise"
    ).to_numpy(np.float64)
    stat["rows"] += int(errors.size)
    stat["sse"] += float(np.dot(errors, errors))
    stat["sae"] += float(np.abs(errors).sum())
    stat["wells"].update(subset["well"].astype(str).unique())


def update_grouped_stats(
    stores: dict[str, dict[Any, dict[str, Any]]],
    selected: pd.DataFrame,
) -> None:
    update_stat(stores["overall"], "__overall__", selected)
    for well, subset in selected.groupby("well", sort=False, observed=True):
        update_stat(stores["by_well"], str(well), subset)
    for bucket, subset in selected.groupby("distance_bucket", sort=False, observed=True):
        update_stat(stores["distance_bucket"], str(bucket), subset)
    for bucket, subset in selected.groupby("tail_rank_bucket", sort=False, observed=True):
        update_stat(stores["tail_rank_bucket"], str(bucket), subset)


def aggregate_model_metrics(
    *,
    path: Path,
    selector_col: str,
    selector_value: str,
    chunksize: int = 750_000,
) -> dict[str, dict[Any, dict[str, Any]]]:
    usecols = ["id", "well", selector_col, "target_tvt", "pred_tvt"]
    stores: dict[str, dict[Any, dict[str, Any]]] = {
        "overall": {},
        "by_well": {},
        "distance_bucket": {},
        "tail_rank_bucket": {},
    }
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        selected = chunk[chunk[selector_col].eq(selector_value)].copy()
        if selected.empty:
            continue
        selected["tail_rank"] = parse_tail_rank(selected["id"])
        selected["distance_bucket"] = distance_bucket(selected["tail_rank"])
        selected["tail_rank_bucket"] = tail_rank_bucket(selected["tail_rank"])
        update_grouped_stats(stores, selected)
    if "__overall__" not in stores["overall"]:
        raise ValueError(f"No rows found for {selector_col}={selector_value} in {path}")
    return stores


def stat_to_metrics(stat: dict[str, Any]) -> dict[str, float | int]:
    rows = int(stat["rows"])
    return {
        "rows": rows,
        "wells": len(stat["wells"]),
        "rmse": float(np.sqrt(stat["sse"] / rows)) if rows else float("nan"),
        "mae": float(stat["sae"] / rows) if rows else float("nan"),
        "sse": float(stat["sse"]),
    }


def combine_group_metrics(
    all_stats: dict[str, dict[str, dict[Any, dict[str, Any]]]],
    group_kind: str,
    *,
    group_col: str | None = None,
) -> pd.DataFrame:
    keys: set[Any] = set()
    for model_stats in all_stats.values():
        keys.update(model_stats[group_kind].keys())
    rows: list[dict[str, Any]] = []
    for key in sorted(keys):
        base_stat = all_stats["exp073_lgb_mean"][group_kind].get(key)
        if base_stat is None:
            base_stat = next(
                model_stats[group_kind][key]
                for model_stats in all_stats.values()
                if key in model_stats[group_kind]
            )
        record: dict[str, Any] = {
            "rows": int(base_stat["rows"]),
            "wells": len(base_stat["wells"]),
        }
        if group_col is not None:
            record[group_col] = key
        for model_name, model_stats in all_stats.items():
            stat = model_stats[group_kind].get(key)
            if stat is None:
                record[f"{model_name}_rows"] = 0
                record[f"{model_name}_rmse"] = float("nan")
                record[f"{model_name}_mae"] = float("nan")
                record[f"{model_name}_sse"] = float("nan")
                continue
            metrics = stat_to_metrics(stat)
            record[f"{model_name}_rows"] = metrics["rows"]
            record[f"{model_name}_rmse"] = metrics["rmse"]
            record[f"{model_name}_mae"] = metrics["mae"]
            record[f"{model_name}_sse"] = metrics["sse"]
        add_deltas(record)
        rows.append(record)
    columns = ([group_col] if group_col is not None else []) + [
        col for col in rows[0] if col != group_col
    ]
    return pd.DataFrame(rows)[columns]


def add_deltas(record: dict[str, Any]) -> None:
    record["exp092_lgb1_rmse_delta_vs_exp073"] = (
        record["exp092_lgb1_rmse"] - record["exp073_lgb_mean_rmse"]
    )
    record["exp092_lgb1_rmse_delta_vs_exp077"] = (
        record["exp092_lgb1_rmse"] - record["exp077_policy_rmse"]
    )
    record["exp092_lgb_mean_rmse_delta_vs_exp073"] = (
        record["exp092_lgb_mean_rmse"] - record["exp073_lgb_mean_rmse"]
    )
    record["exp092_lgb_mean_rmse_delta_vs_exp077"] = (
        record["exp092_lgb_mean_rmse"] - record["exp077_policy_rmse"]
    )
    record["exp092_lgb1_sse_delta_vs_exp073"] = (
        record["exp092_lgb1_sse"] - record["exp073_lgb_mean_sse"]
    )
    record["exp092_lgb1_sse_delta_vs_exp077"] = (
        record["exp092_lgb1_sse"] - record["exp077_policy_sse"]
    )


def load_path_frame(
    *,
    path: Path,
    selector_col: str,
    selector_value: str,
    pred_col: str,
    chunksize: int = 750_000,
) -> pd.DataFrame:
    usecols = ["id", "well", selector_col, "pred_tvt"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        selected = chunk[chunk[selector_col].eq(selector_value)].copy()
        if selected.empty:
            continue
        selected["tail_rank"] = parse_tail_rank(selected["id"])
        selected[pred_col] = pd.to_numeric(selected["pred_tvt"], errors="raise").astype("float32")
        chunks.append(selected[["well", "tail_rank", pred_col]])
    if not chunks:
        raise ValueError(f"No rows found for {selector_col}={selector_value} in {path}")
    frame = pd.concat(chunks, ignore_index=True)
    frame["well"] = frame["well"].astype("category")
    frame["tail_rank"] = frame["tail_rank"].astype("int32")
    return frame.sort_values(["well", "tail_rank"]).reset_index(drop=True)


def step_stats(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, subset in frame.groupby("well", sort=False, observed=True):
        step = subset[pred_col].diff().abs().dropna()
        rows.append(
            {
                "well": str(well),
                "rows": int(len(subset)),
                f"{pred_col}_step_abs_p95": (
                    float(step.quantile(0.95)) if not step.empty else float("nan")
                ),
                f"{pred_col}_step_abs_max": float(step.max()) if not step.empty else float("nan"),
                f"{pred_col}_step_abs_ge10": int((step >= 10.0).sum()),
                f"{pred_col}_step_abs_ge25": int((step >= 25.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def correction_stats(lgb1: pd.DataFrame, base: pd.DataFrame, base_col: str) -> pd.DataFrame:
    merged = lgb1.merge(base, on=["well", "tail_rank"], how="inner", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    prefix = "lgb1_minus_" + re.sub(r"^pred_", "", base_col)
    for well, subset in merged.groupby("well", sort=False, observed=True):
        correction = subset["pred_exp092_lgb1"] - subset[base_col]
        correction_step = correction.diff().abs().dropna()
        rows.append(
            {
                "well": str(well),
                f"{prefix}_correction_abs_mean": float(correction.abs().mean()),
                f"{prefix}_correction_abs_p95": float(correction.abs().quantile(0.95)),
                f"{prefix}_correction_step_abs_p95": (
                    float(correction_step.quantile(0.95))
                    if not correction_step.empty
                    else float("nan")
                ),
                f"{prefix}_correction_step_abs_max": (
                    float(correction_step.max()) if not correction_step.empty else float("nan")
                ),
                f"{prefix}_correction_step_abs_ge5": int((correction_step >= 5.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def path_continuity(paths: dict[str, Path]) -> pd.DataFrame:
    lgb1_spec = PATH_MODELS["pred_exp092_lgb1"]
    lgb1 = load_path_frame(
        path=paths[lgb1_spec["path_key"]],
        selector_col=lgb1_spec["selector_col"],
        selector_value=lgb1_spec["selector_value"],
        pred_col="pred_exp092_lgb1",
    )
    continuity = step_stats(lgb1, "pred_exp092_lgb1")
    for pred_col, spec in PATH_MODELS.items():
        if pred_col == "pred_exp092_lgb1":
            continue
        frame = load_path_frame(
            path=paths[spec["path_key"]],
            selector_col=spec["selector_col"],
            selector_value=spec["selector_value"],
            pred_col=pred_col,
        )
        continuity = continuity.merge(
            step_stats(frame, pred_col).drop(columns=["rows"]),
            on="well",
            how="outer",
        )
        if pred_col in {"pred_exp073", "pred_exp077_policy"}:
            continuity = continuity.merge(correction_stats(lgb1, frame, pred_col), on="well")
        del frame
    return continuity


def summarize_guard(
    overall: pd.DataFrame,
    by_well: pd.DataFrame,
    buckets: pd.DataFrame,
) -> dict[str, Any]:
    one = overall.iloc[0].to_dict()
    well_lgb1_vs_exp077 = by_well["exp092_lgb1_rmse_delta_vs_exp077"]
    well_lgb1_vs_exp073 = by_well["exp092_lgb1_rmse_delta_vs_exp073"]
    near_buckets = buckets[
        buckets["bucket_family"].eq("distance_bucket")
        & buckets["bucket"].isin(["000_050", "050_100", "100_250"])
    ]
    long_bucket = buckets[
        buckets["bucket_family"].eq("distance_bucket") & buckets["bucket"].eq("1000_plus")
    ]
    return {
        "rows": int(one["rows"]),
        "wells": int(one["wells"]),
        "overall": {
            "exp073_lgb_mean_rmse": float(one["exp073_lgb_mean_rmse"]),
            "exp077_policy_rmse": float(one["exp077_policy_rmse"]),
            "exp092_lgb1_rmse": float(one["exp092_lgb1_rmse"]),
            "exp092_lgb2_rmse": float(one["exp092_lgb2_rmse"]),
            "exp092_lgb_mean_rmse": float(one["exp092_lgb_mean_rmse"]),
            "exp092_lgb1_delta_vs_exp073": float(one["exp092_lgb1_rmse_delta_vs_exp073"]),
            "exp092_lgb1_delta_vs_exp077": float(one["exp092_lgb1_rmse_delta_vs_exp077"]),
            "exp092_lgb_mean_delta_vs_exp073": float(one["exp092_lgb_mean_rmse_delta_vs_exp073"]),
            "exp092_lgb_mean_delta_vs_exp077": float(one["exp092_lgb_mean_rmse_delta_vs_exp077"]),
        },
        "well_delta": {
            "lgb1_improved_vs_exp077_wells": int((well_lgb1_vs_exp077 < 0).sum()),
            "lgb1_worsened_vs_exp077_wells": int((well_lgb1_vs_exp077 > 0).sum()),
            "lgb1_max_regression_vs_exp077": float(well_lgb1_vs_exp077.max()),
            "lgb1_max_improvement_vs_exp077": float(well_lgb1_vs_exp077.min()),
            "lgb1_improved_vs_exp073_wells": int((well_lgb1_vs_exp073 < 0).sum()),
            "lgb1_worsened_vs_exp073_wells": int((well_lgb1_vs_exp073 > 0).sum()),
            "lgb1_max_regression_vs_exp073": float(well_lgb1_vs_exp073.max()),
            "lgb1_max_improvement_vs_exp073": float(well_lgb1_vs_exp073.min()),
        },
        "near_row_guard": near_buckets[
            [
                "bucket",
                "rows",
                "exp092_lgb1_rmse_delta_vs_exp077",
                "exp092_lgb1_rmse_delta_vs_exp073",
                "exp092_lgb_mean_rmse_delta_vs_exp077",
                "exp092_lgb_mean_rmse_delta_vs_exp073",
            ]
        ].to_dict("records"),
        "long_tail_guard": long_bucket[
            [
                "bucket",
                "rows",
                "exp092_lgb1_rmse_delta_vs_exp077",
                "exp092_lgb1_rmse_delta_vs_exp073",
                "exp092_lgb_mean_rmse_delta_vs_exp077",
                "exp092_lgb_mean_rmse_delta_vs_exp073",
            ]
        ].to_dict("records"),
        "guard_thresholds": {
            "max_well_regression_warn": 0.25,
            "near_row_regression_warn": 0.05,
        },
    }


def run_guard(
    *,
    exp073_path: Path = DEFAULT_EXP073_PREDICTIONS,
    exp077_path: Path = DEFAULT_EXP077_PREDICTIONS,
    exp092_path: Path = DEFAULT_EXP092_PREDICTIONS,
    feature_cache_path: Path = DEFAULT_FEATURE_CACHE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "exp073": ensure_path(exp073_path, "exp073 predictions"),
        "exp077": ensure_path(exp077_path, "exp077 predictions"),
        "exp092": ensure_path(exp092_path, "exp092 predictions"),
    }
    feature_cache_available = has_nonempty_path(feature_cache_path)

    all_stats: dict[str, dict[str, dict[Any, dict[str, Any]]]] = {}
    for model_name, spec in MODEL_SPECS.items():
        print(f"aggregating {model_name}", flush=True)
        all_stats[model_name] = aggregate_model_metrics(
            path=paths[spec["path_key"]],
            selector_col=spec["selector_col"],
            selector_value=spec["selector_value"],
        )

    print("combining aggregate tables", flush=True)
    overall = combine_group_metrics(all_stats, "overall")
    by_well = combine_group_metrics(all_stats, "by_well", group_col="well").sort_values(
        "exp092_lgb1_rmse_delta_vs_exp077", ascending=False
    )
    by_distance = combine_group_metrics(all_stats, "distance_bucket", group_col="bucket")
    by_distance.insert(0, "bucket_family", "distance_bucket")
    by_tail = combine_group_metrics(all_stats, "tail_rank_bucket", group_col="bucket")
    by_tail.insert(0, "bucket_family", "tail_rank_bucket")
    buckets = pd.concat([by_distance, by_tail], ignore_index=True)

    print("checking path continuity", flush=True)
    continuity = path_continuity(paths)
    summary = summarize_guard(overall, by_well, buckets)
    summary.update(
        {
            "experiment": "exp092_oof_delta_guard",
            "status": "completed",
            "context_mode": (
                "feature_cache_available_but_row_rank_used"
                if feature_cache_available
                else "tail_rank_fallback_feature_cache_missing_or_empty"
            ),
            "inputs": {
                "exp073_predictions": str(exp073_path),
                "exp073_predictions_sha256": sha256_file(exp073_path),
                "exp073_predictions_decompressed_sha256": sha256_gzip_decompressed(exp073_path),
                "exp077_predictions": str(exp077_path),
                "exp077_predictions_sha256": sha256_file(exp077_path),
                "exp077_predictions_decompressed_sha256": sha256_gzip_decompressed(exp077_path),
                "exp092_predictions": str(exp092_path),
                "exp092_predictions_sha256": sha256_file(exp092_path),
                "exp092_predictions_decompressed_sha256": sha256_gzip_decompressed(exp092_path),
                "feature_cache": str(feature_cache_path),
                "feature_cache_available": feature_cache_available,
                "feature_cache_sha256": (
                    sha256_file(feature_cache_path) if feature_cache_available else None
                ),
                "feature_cache_decompressed_sha256": (
                    sha256_gzip_decompressed(feature_cache_path)
                    if feature_cache_available
                    else None
                ),
            },
            "artifacts": {
                "overall": "exp092_oof_delta_guard_overall.csv",
                "by_well": "exp092_oof_delta_guard_by_well.csv",
                "bucket": "exp092_oof_delta_guard_bucket.csv",
                "path_continuity": "exp092_oof_delta_guard_path_continuity.csv",
                "summary": "exp092_oof_delta_guard_summary.json",
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    overall.to_csv(output_dir / "exp092_oof_delta_guard_overall.csv", index=False)
    by_well.to_csv(output_dir / "exp092_oof_delta_guard_by_well.csv", index=False)
    buckets.to_csv(output_dir / "exp092_oof_delta_guard_bucket.csv", index=False)
    continuity.to_csv(output_dir / "exp092_oof_delta_guard_path_continuity.csv", index=False)
    with (output_dir / "exp092_oof_delta_guard_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_guard()
