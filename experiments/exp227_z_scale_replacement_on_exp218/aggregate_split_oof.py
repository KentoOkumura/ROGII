from __future__ import annotations

import gzip
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "exp227_z_scale_replacement_on_exp218"
OUTPUT_PREFIX = EXPERIMENT
VARIANT = "z_scale_replacement"
MODE = "cpu_deterministic_threads8"
SPLITS = ["lgb0", "lgb1", "lgb2"]
CHUNK_SIZE = 250_000
PREDICTION_USECOLS = [
    "id",
    "well",
    "variant",
    "mode",
    "model",
    "last_known_tvt",
    "target",
    "target_tvt",
    "pred_target",
    "pred_tvt",
]
REFERENCE_RMSE_TVT = {
    "exp218_parent_lgb_mean": 8.475793751656624,
    "exp148_feature_surface_lgb_mean": 8.50128118189582,
    "exp224_addonly_lgb_mean": 8.538687041980328,
}


def sha256_gzip_decompressed(path: Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = pd.to_numeric(ids.astype(str).str.extract(r"_(\d+)$", expand=False), errors="coerce")
    return pd.cut(
        ranks.fillna(-1).to_numpy(np.int32),
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def split_artifacts_dir(base_dir: Path, split: str) -> Path:
    return base_dir / "kaggle" / "output" / f"train_{split}_v1" / "artifacts"


def prediction_path(base_dir: Path, split: str) -> Path:
    return split_artifacts_dir(base_dir, split) / f"{OUTPUT_PREFIX}_predictions.csv.gz"


def iter_selected_prediction_chunks(path: Path):
    reader = pd.read_csv(
        path,
        usecols=PREDICTION_USECOLS,
        dtype={"id": str, "well": str},
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        selected = chunk[
            chunk["variant"].astype(str).eq(VARIANT)
            & chunk["mode"].astype(str).eq(MODE)
            & chunk["model"].astype(str).eq("lgb_mean")
        ].copy()
        if not selected.empty:
            yield selected.reset_index(drop=True)


def load_metric_row(base_dir: Path, split: str) -> dict[str, Any]:
    path = split_artifacts_dir(base_dir, split) / f"{OUTPUT_PREFIX}_metrics.csv"
    metrics = pd.read_csv(path)
    row = metrics[
        metrics["variant"].astype(str).eq(VARIANT)
        & metrics["mode"].astype(str).eq(MODE)
        & metrics["model"].astype(str).eq("lgb_mean")
        & metrics["fold"].astype(str).eq("pooled")
    ]
    if row.empty:
        raise ValueError(f"{split}: pooled lgb_mean metric row not found: {path}")
    data = row.iloc[0].to_dict()
    data["split"] = split
    return jsonable(data)


def validate_aligned(base: pd.DataFrame, other: pd.DataFrame, split: str, chunk_index: int) -> None:
    if len(base) != len(other):
        raise ValueError(f"{split} chunk {chunk_index}: row count mismatch {len(base)} vs {len(other)}")
    for column in ["id", "well"]:
        if not base[column].to_numpy().tolist() == other[column].to_numpy().tolist():
            raise ValueError(f"{split} chunk {chunk_index}: {column} order mismatch")
    for column in ["last_known_tvt", "target", "target_tvt"]:
        if not np.allclose(
            base[column].to_numpy(np.float64),
            other[column].to_numpy(np.float64),
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"{split} chunk {chunk_index}: {column} value mismatch")


def grouped_error_stats(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    work = frame[[group_col, "error_tvt"]].copy()
    work["error_abs"] = np.abs(work["error_tvt"].to_numpy(np.float64))
    work["error_sq"] = np.square(work["error_tvt"].to_numpy(np.float64))
    return (
        work.groupby(group_col, observed=True)
        .agg(
            rows=("error_tvt", "size"),
            error_sum=("error_tvt", "sum"),
            error_abs_sum=("error_abs", "sum"),
            error_sq_sum=("error_sq", "sum"),
        )
        .reset_index()
    )


def combine_error_stats(frames: list[pd.DataFrame], group_col: str) -> pd.DataFrame:
    combined = (
        pd.concat(frames, ignore_index=True)
        .groupby(group_col, observed=True, as_index=False)
        .agg(
            rows=("rows", "sum"),
            error_sum=("error_sum", "sum"),
            error_abs_sum=("error_abs_sum", "sum"),
            error_sq_sum=("error_sq_sum", "sum"),
        )
    )
    combined["rmse_tvt"] = np.sqrt(combined["error_sq_sum"] / combined["rows"])
    combined["error_mean"] = combined["error_sum"] / combined["rows"]
    combined["error_abs_mean"] = combined["error_abs_sum"] / combined["rows"]
    return combined.drop(columns=["error_sum", "error_abs_sum", "error_sq_sum"])


def add_metric_accumulator(acc: dict[str, Any], frame: pd.DataFrame) -> None:
    tvt_error = frame["pred_tvt"].to_numpy(np.float64) - frame["target_tvt"].to_numpy(np.float64)
    target_error = frame["pred_target"].to_numpy(np.float64) - frame["target"].to_numpy(np.float64)
    acc["rows"] += int(len(frame))
    acc["wells"].update(frame["well"].astype(str).unique().tolist())
    acc["sum_sq_tvt"] += float(np.square(tvt_error).sum())
    acc["sum_sq_target"] += float(np.square(target_error).sum())


def aggregate_predictions(
    base_dir: Path,
    output_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    readers = [iter_selected_prediction_chunks(prediction_path(base_dir, split)) for split in SPLITS]
    sentinel = object()
    total_rows = 0
    wells: set[str] = set()
    sum_sq_tvt = 0.0
    sum_sq_target = 0.0
    value_chunks: list[np.ndarray] = []
    by_well_parts: list[pd.DataFrame] = []
    tail_bucket_parts: list[pd.DataFrame] = []
    split_acc = {
        split: {"rows": 0, "wells": set(), "sum_sq_tvt": 0.0, "sum_sq_target": 0.0}
        for split in SPLITS
    }

    prediction_hasher = hashlib.sha256()
    prediction_hasher.update(f"{VARIANT}/{MODE}/lgb_mean/pooled/tvt".encode("utf-8"))

    target_columns = [f"pred_target_{split}" for split in SPLITS]
    tvt_columns = [f"pred_tvt_{split}" for split in SPLITS]
    output_columns = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target",
        "target_tvt",
        *target_columns,
        *tvt_columns,
        "pred_target",
        "pred_tvt",
        "pred_tvt_direct_mean",
        "pred_tvt_formula_minus_direct_mean",
    ]

    with gzip.open(output_path, "wt", newline="") as fp:
        write_header = True
        for chunk_index, chunks in enumerate(zip_longest(*readers, fillvalue=sentinel)):
            if any(chunk is sentinel for chunk in chunks):
                raise ValueError(f"split prediction readers ended at different times near chunk {chunk_index}")
            split_chunks = dict(zip(SPLITS, chunks, strict=True))
            base = split_chunks[SPLITS[0]]
            for split, frame in split_chunks.items():
                add_metric_accumulator(split_acc[split], frame)
                if split != SPLITS[0]:
                    validate_aligned(base, frame, split, chunk_index)

            out = base[["id", "well", "variant", "mode", "model", "last_known_tvt", "target", "target_tvt"]].copy()
            pred_targets = []
            pred_tvts = []
            for split in SPLITS:
                out[f"pred_target_{split}"] = split_chunks[split]["pred_target"].to_numpy(np.float32)
                out[f"pred_tvt_{split}"] = split_chunks[split]["pred_tvt"].to_numpy(np.float32)
                pred_targets.append(out[f"pred_target_{split}"].to_numpy(np.float32))
                pred_tvts.append(out[f"pred_tvt_{split}"].to_numpy(np.float32))

            pred_target = np.mean(np.vstack(pred_targets), axis=0).astype(np.float32)
            pred_tvt_direct_mean = np.mean(np.vstack(pred_tvts), axis=0).astype(np.float32)
            pred_tvt = (out["last_known_tvt"].to_numpy(np.float32) + pred_target).astype(np.float32)
            out["pred_target"] = pred_target
            out["pred_tvt"] = pred_tvt
            out["pred_tvt_direct_mean"] = pred_tvt_direct_mean
            out["pred_tvt_formula_minus_direct_mean"] = (pred_tvt - pred_tvt_direct_mean).astype(np.float32)

            error_tvt = out["pred_tvt"].to_numpy(np.float64) - out["target_tvt"].to_numpy(np.float64)
            error_target = out["pred_target"].to_numpy(np.float64) - out["target"].to_numpy(np.float64)
            out["error_tvt"] = error_tvt

            total_rows += int(len(out))
            wells.update(out["well"].astype(str).unique().tolist())
            sum_sq_tvt += float(np.square(error_tvt).sum())
            sum_sq_target += float(np.square(error_target).sum())
            value_chunks.append(pred_tvt.copy())
            for raw_id in out["id"].astype(str).to_numpy():
                prediction_hasher.update(raw_id.encode("utf-8"))
                prediction_hasher.update(b"\0")

            by_well_parts.append(grouped_error_stats(out, "well"))
            bucket_frame = out[["id", "error_tvt"]].copy()
            bucket_frame["bucket"] = tail_rank_bucket(out["id"])
            tail_bucket_parts.append(grouped_error_stats(bucket_frame, "bucket"))

            out[output_columns].to_csv(fp, index=False, header=write_header)
            write_header = False

    for values in value_chunks:
        prediction_hasher.update(values.astype(np.float32).tobytes())

    by_well = combine_error_stats(by_well_parts, "well").sort_values("rmse_tvt", ascending=False)
    by_well.insert(0, "model", "lgb_mean")
    by_well.insert(0, "mode", MODE)
    by_well.insert(0, "variant", VARIANT)

    tail_bucket = combine_error_stats(tail_bucket_parts, "bucket")
    tail_bucket.insert(0, "model", "lgb_mean")
    tail_bucket.insert(0, "mode", MODE)
    tail_bucket.insert(0, "variant", VARIANT)
    tail_bucket.insert(3, "bucket_family", "tail_rank_bucket")

    split_sources = {}
    for split, acc in split_acc.items():
        rows = int(acc["rows"])
        split_sources[split] = {
            "split": split,
            "path": str(prediction_path(base_dir, split)),
            "rows": rows,
            "wells": int(len(acc["wells"])),
            "rmse_tvt_from_predictions": float(np.sqrt(acc["sum_sq_tvt"] / rows)),
            "rmse_target_from_predictions": float(np.sqrt(acc["sum_sq_target"] / rows)),
            "decompressed_sha256": sha256_gzip_decompressed(prediction_path(base_dir, split)),
        }

    aggregate = {
        "rows": int(total_rows),
        "wells": int(len(wells)),
        "aggregate_rmse_tvt": float(np.sqrt(sum_sq_tvt / total_rows)),
        "aggregate_rmse_target": float(np.sqrt(sum_sq_target / total_rows)),
        "prediction_sha256": prediction_hasher.hexdigest(),
        "prediction_sources": [split_sources[split] for split in SPLITS],
    }
    return aggregate, by_well, tail_bucket


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    aggregate_dir = base_dir / "kaggle" / "output" / "train_split_aggregate_v1" / "artifacts"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{OUTPUT_PREFIX}_split_aggregate"
    paths = {
        "metrics": aggregate_dir / f"{prefix}_metrics.csv",
        "by_well": aggregate_dir / f"{prefix}_by_well.csv",
        "bucket_metrics_tail_rank_only": aggregate_dir / f"{prefix}_bucket_metrics.csv",
        "split_individual_bucket_metrics": aggregate_dir / f"{OUTPUT_PREFIX}_split_individual_bucket_metrics.csv",
        "feature_importance": aggregate_dir / f"{prefix}_feature_importance.csv",
        "feature_importance_mean": aggregate_dir / f"{prefix}_feature_importance_mean.csv",
        "predictions": aggregate_dir / f"{prefix}_predictions.csv.gz",
        "summary": aggregate_dir / f"{prefix}_summary.json",
    }

    aggregate, by_well, tail_bucket = aggregate_predictions(base_dir, paths["predictions"])
    split_metric_rows = [load_metric_row(base_dir, split) for split in SPLITS]

    split_bucket_frames = []
    importance_frames = []
    for split in SPLITS:
        artifacts_dir = split_artifacts_dir(base_dir, split)
        bucket = pd.read_csv(artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
        bucket.insert(0, "split_source", split)
        split_bucket_frames.append(bucket)

        importance = pd.read_csv(artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance.csv")
        importance.insert(0, "split_source", split)
        importance_frames.append(importance)

    split_individual_bucket = pd.concat(split_bucket_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    importance_mean = (
        importance.groupby(["variant", "mode", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_max=("importance", "max"),
            records=("importance", "size"),
        )
        .sort_values("importance_mean", ascending=False)
    )

    metrics = pd.DataFrame(
        [
            {
                "variant": VARIANT,
                "mode": MODE,
                "model": "lgb_mean",
                "fold": "pooled",
                "rows": aggregate["rows"],
                "train_rows": None,
                "features": int(split_metric_rows[0].get("features", 0)),
                "feature_groups": split_metric_rows[0].get("feature_groups"),
                "best_iteration": None,
                "rmse_tvt": aggregate["aggregate_rmse_tvt"],
                "rmse_target": aggregate["aggregate_rmse_target"],
                "prediction_sha256": aggregate["prediction_sha256"],
                "model_file": None,
                "model_sha256": None,
            }
        ]
    )

    metrics.to_csv(paths["metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    tail_bucket.to_csv(paths["bucket_metrics_tail_rank_only"], index=False)
    split_individual_bucket.to_csv(paths["split_individual_bucket_metrics"], index=False)
    importance.to_csv(paths["feature_importance"], index=False)
    importance_mean.to_csv(paths["feature_importance_mean"], index=False)

    aggregate_rmse_tvt = float(aggregate["aggregate_rmse_tvt"])
    rejected = aggregate_rmse_tvt > REFERENCE_RMSE_TVT["exp218_parent_lgb_mean"]
    summary = {
        "experiment": EXPERIMENT,
        "status": "split_aggregate_completed_rejected" if rejected else "split_aggregate_completed_candidate",
        "variant": VARIANT,
        "mode": MODE,
        "model": "lgb_mean",
        "rows": aggregate["rows"],
        "wells": aggregate["wells"],
        "aggregate_rmse_tvt": aggregate_rmse_tvt,
        "aggregate_rmse_target": aggregate["aggregate_rmse_target"],
        "prediction_sha256": aggregate["prediction_sha256"],
        "reference_rmse_tvt": REFERENCE_RMSE_TVT,
        "delta_vs_exp218_rmse_tvt": aggregate_rmse_tvt - REFERENCE_RMSE_TVT["exp218_parent_lgb_mean"],
        "delta_vs_exp148_rmse_tvt": aggregate_rmse_tvt - REFERENCE_RMSE_TVT["exp148_feature_surface_lgb_mean"],
        "delta_vs_exp224_addonly_rmse_tvt": aggregate_rmse_tvt - REFERENCE_RMSE_TVT["exp224_addonly_lgb_mean"],
        "split_rmse_tvt": {
            row["split"]: row["rmse_tvt_from_predictions"] for row in aggregate["prediction_sources"]
        },
        "split_rmse_target": {
            row["split"]: row["rmse_target_from_predictions"] for row in aggregate["prediction_sources"]
        },
        "split_metric_rows": split_metric_rows,
        "prediction_sources": aggregate["prediction_sources"],
        "full_train_coverage_pass": {
            row["split"]: row["rows"] == aggregate["rows"] and row["wells"] == aggregate["wells"]
            for row in aggregate["prediction_sources"]
        },
        "top_features_by_mean_importance": importance_mean.head(20).to_dict(orient="records"),
        "worst_wells_top5": by_well.head(5)[
            ["well", "rows", "rmse_tvt", "error_mean", "error_abs_mean"]
        ].to_dict(orient="records"),
        "tail_rank_bucket_metrics": tail_bucket.to_dict(orient="records"),
        "split_distance_bucket_rmse_tvt": split_individual_bucket[
            split_individual_bucket["bucket_family"].astype(str).eq("distance_bucket")
        ][["split_source", "bucket", "rows", "rmse_tvt", "error_abs_mean"]].to_dict(orient="records"),
        "bucket_readout_note": (
            "Aggregate bucket metrics are exact for tail_rank_bucket only. "
            "distance_bucket aggregate was not recomputed locally because md_since is not present "
            "in the split prediction archives; per-split distance_bucket CSVs are concatenated as "
            "split_individual_bucket_metrics."
        ),
        "decision": "reject_no_inference_no_submit" if rejected else "candidate_needs_review",
        "decision_reason": (
            "3-config OOF RMSE is worse than exp218 parent."
            if rejected
            else "3-config OOF RMSE improves over exp218 parent; review before inference/submit."
        ),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "summary"},
    }
    paths["summary"].write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
