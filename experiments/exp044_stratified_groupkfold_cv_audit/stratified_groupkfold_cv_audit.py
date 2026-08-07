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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


@dataclass
class MetricAccumulator:
    sse: float = 0.0
    n: int = 0
    wells: set[str] = field(default_factory=set)

    def add(self, y_true: np.ndarray, y_pred: np.ndarray, wells: np.ndarray) -> None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not mask.any():
            return
        diff = y_pred[mask] - y_true[mask]
        self.sse += float(np.dot(diff, diff))
        self.n += int(mask.sum())
        self.wells.update(str(value) for value in wells[mask])

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sse / self.n) if self.n else float("nan")

    def row(self, **keys: Any) -> dict[str, Any]:
        return {
            **keys,
            "rmse": round(self.rmse, 6) if self.n else None,
            "rows": self.n,
            "wells": len(self.wells),
        }


def well_id_from_path(path: Path) -> str:
    return path.name.split(HORIZONTAL_SUFFIX)[0]


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        return float("nan")
    diff = y_pred.loc[mask].to_numpy(dtype=float) - y_true.loc[mask].to_numpy(dtype=float)
    return float(math.sqrt(np.mean(diff * diff)))


def qbin(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index, dtype=object)
    try:
        bins = pd.qcut(values.rank(method="first"), q=min(n_bins, values.notna().sum()), labels=False)
    except ValueError:
        bins = pd.cut(values, bins=min(n_bins, values.nunique(dropna=True)), labels=False)
    out = bins.astype("Int64").astype(str)
    out = out.where(values.notna(), "missing")
    return prefix + "_" + out


def fixed_azimuth_bin(series: pd.Series, n_bins: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    bins = pd.cut(values, bins=edges, include_lowest=True, labels=False)
    out = bins.astype("Int64").astype(str)
    out = out.where(values.notna(), "missing")
    return "az_" + out


def summarize_well(path: Path) -> dict[str, Any]:
    well_id = well_id_from_path(path)
    df = pd.read_csv(path)
    eval_mask = df["TVT_input"].isna()
    known_mask = df["TVT_input"].notna()
    eval_indices = np.flatnonzero(eval_mask.to_numpy())
    known_indices = np.flatnonzero(known_mask.to_numpy())

    if len(eval_indices):
        first_eval = int(eval_indices[0])
        last_eval = int(eval_indices[-1])
    else:
        first_eval = -1
        last_eval = -1

    prefix_known = df.loc[: max(first_eval - 1, 0), "TVT_input"] if first_eval > 0 else df["TVT_input"].iloc[:0]
    if prefix_known.notna().any():
        last_known_tvt = float(prefix_known.dropna().iloc[-1])
        median_known_tvt = float(prefix_known.dropna().median())
    elif len(known_indices):
        known_tvt = df.loc[known_mask, "TVT_input"]
        last_known_tvt = float(known_tvt.iloc[-1])
        median_known_tvt = float(known_tvt.median())
    else:
        last_known_tvt = float("nan")
        median_known_tvt = float("nan")

    dx = float(df["X"].iloc[-1] - df["X"].iloc[0])
    dy = float(df["Y"].iloc[-1] - df["Y"].iloc[0])
    signed_azimuth = math.degrees(math.atan2(dy, dx))

    return {
        "well_id": well_id,
        "n_rows": int(len(df)),
        "eval_rows": int(eval_mask.sum()),
        "known_rows": int(known_mask.sum()),
        "first_eval_row": first_eval,
        "last_eval_row": last_eval,
        "eval_length": int(eval_mask.sum()),
        "prefix_length": int((np.arange(len(df)) < first_eval).sum()) if first_eval >= 0 else int(len(df)),
        "centroid_x": float(df["X"].mean()),
        "centroid_y": float(df["Y"].mean()),
        "start_x": float(df["X"].iloc[0]),
        "start_y": float(df["Y"].iloc[0]),
        "end_x": float(df["X"].iloc[-1]),
        "end_y": float(df["Y"].iloc[-1]),
        "delta_x": dx,
        "delta_y": dy,
        "signed_azimuth_deg": signed_azimuth,
        "median_known_tvt": median_known_tvt,
        "last_known_tvt": last_known_tvt,
        "median_full_tvt": float(df["TVT"].median()) if "TVT" in df else float("nan"),
        "gr_coverage": float(df["GR"].notna().mean()) if "GR" in df else float("nan"),
        "gr_missing_rate": float(df["GR"].isna().mean()) if "GR" in df else float("nan"),
        "md_min": float(df["MD"].min()),
        "md_max": float(df["MD"].max()),
        "z_min": float(df["Z"].min()),
        "z_max": float(df["Z"].max()),
    }


def build_well_metadata(paths: ExperimentPaths, config: dict[str, Any], max_wells: int | None) -> pd.DataFrame:
    train_files = sorted(paths.train_data_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if max_wells is not None:
        train_files = train_files[:max_wells]
    if not train_files:
        raise FileNotFoundError(f"No train files found in {paths.train_data_dir}")

    rows = [summarize_well(path) for path in train_files]
    meta = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    bin_config = get_nested(config, "audit.binning") or {}
    meta["azimuth_bin"] = fixed_azimuth_bin(
        meta["signed_azimuth_deg"], int(bin_config.get("azimuth_bins", 4))
    )
    meta["tvt_bin"] = qbin(
        meta["median_known_tvt"].fillna(meta["median_full_tvt"]),
        int(bin_config.get("tvt_bins", 4)),
        "tvt",
    )
    meta["x_bin"] = qbin(meta["centroid_x"], int(bin_config.get("spatial_x_bins", 3)), "x")
    meta["y_bin"] = qbin(meta["centroid_y"], int(bin_config.get("spatial_y_bins", 3)), "y")
    meta["spatial_bin"] = meta["x_bin"].astype(str) + "__" + meta["y_bin"].astype(str)
    meta["eval_length_bin"] = qbin(
        meta["eval_length"], int(bin_config.get("eval_length_bins", 3)), "eval_len"
    )
    meta["gr_bin"] = qbin(meta["gr_coverage"], int(bin_config.get("gr_bins", 3)), "gr")
    meta["strat_label_full"] = (
        meta["azimuth_bin"].astype(str)
        + "|"
        + meta["tvt_bin"].astype(str)
        + "|"
        + meta["spatial_bin"].astype(str)
        + "|"
        + meta["eval_length_bin"].astype(str)
        + "|"
        + meta["gr_bin"].astype(str)
    )
    return meta


def collapse_rare_labels(labels: pd.Series, n_folds: int) -> pd.Series:
    collapsed = labels.astype(str).copy()
    while True:
        counts = collapsed.value_counts()
        rare_labels = counts[counts < n_folds].index
        if len(rare_labels) == 0:
            return collapsed
        if len(rare_labels) == len(counts):
            return pd.Series(["rare"] * len(collapsed), index=collapsed.index, dtype=object)
        collapsed = collapsed.where(~collapsed.isin(rare_labels), "rare")


def composite_label(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    label = frame[columns[0]].astype(str).copy()
    for column in columns[1:]:
        label = label + "|" + frame[column].astype(str)
    return label


def choose_stratification_labels(meta: pd.DataFrame, n_folds: int) -> tuple[pd.Series, str]:
    candidate_bases = [
        ["azimuth_bin", "tvt_bin", "spatial_bin", "eval_length_bin", "gr_bin"],
        ["azimuth_bin", "tvt_bin", "spatial_bin"],
        ["azimuth_bin", "tvt_bin", "eval_length_bin"],
        ["spatial_bin", "tvt_bin"],
        ["azimuth_bin", "tvt_bin"],
        ["spatial_bin"],
        ["tvt_bin"],
    ]
    for columns in candidate_bases:
        labels = composite_label(meta, columns)
        counts = labels.value_counts()
        if labels.nunique() > 1 and (counts >= n_folds).any():
            return collapse_rare_labels(labels, n_folds), "+".join(columns)
    return pd.Series(["all"] * len(meta), index=meta.index, dtype=object), "all"


def assign_folds(meta: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    seed = int(get_nested(config, "validation.seed") or 42)
    out = meta.copy()
    out["strat_label"], label_basis = choose_stratification_labels(out, n_folds)
    out["strat_label_basis"] = label_basis

    x = out[["well_id"]]
    groups = out["well_id"].to_numpy()
    out["groupkfold_fold"] = -1
    out["stratified_groupkfold_fold"] = -1

    for fold, (_, valid_idx) in enumerate(GroupKFold(n_splits=n_folds).split(x, groups=groups)):
        out.loc[out.index[valid_idx], "groupkfold_fold"] = fold

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(x, y=out["strat_label"], groups=groups)):
        out.loc[out.index[valid_idx], "stratified_groupkfold_fold"] = fold

    return out


def fold_balance_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_col in ["groupkfold_fold", "stratified_groupkfold_fold"]:
        for fold, part in assignments.groupby(fold_col, sort=True):
            rows.append(
                {
                    "fold_system": fold_col.replace("_fold", ""),
                    "fold": int(fold),
                    "wells": int(len(part)),
                    "eval_rows": int(part["eval_rows"].sum()),
                    "known_rows": int(part["known_rows"].sum()),
                    "mean_eval_length": float(part["eval_length"].mean()),
                    "mean_centroid_x": float(part["centroid_x"].mean()),
                    "mean_centroid_y": float(part["centroid_y"].mean()),
                    "mean_signed_azimuth_deg": float(part["signed_azimuth_deg"].mean()),
                    "mean_median_known_tvt": float(part["median_known_tvt"].mean()),
                    "mean_gr_coverage": float(part["gr_coverage"].mean()),
                    "strat_labels": int(part["strat_label"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def bucket_distribution(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bucket_cols = [
        "azimuth_bin",
        "tvt_bin",
        "spatial_bin",
        "eval_length_bin",
        "gr_bin",
        "strat_label",
    ]
    for fold_col in ["groupkfold_fold", "stratified_groupkfold_fold"]:
        fold_system = fold_col.replace("_fold", "")
        for bucket_col in bucket_cols:
            for (fold, bucket), part in assignments.groupby([fold_col, bucket_col], sort=True):
                rows.append(
                    {
                        "fold_system": fold_system,
                        "bucket_column": bucket_col,
                        "bucket": str(bucket),
                        "fold": int(fold),
                        "wells": int(len(part)),
                        "eval_rows": int(part["eval_rows"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def resolve_source_path(path_value: str) -> Path | None:
    path = Path(path_value)
    if path.exists():
        return path
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(path.name))
        if matches:
            return matches[0]
    return None


def distance_bucket(eval_step: pd.Series, buckets: list[dict[str, Any]]) -> pd.Series:
    values = pd.to_numeric(eval_step, errors="coerce")
    labels = pd.Series([str(buckets[-1]["name"])] * len(values), index=values.index, dtype=object)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        mask = (values > previous_max) & (values <= max_step)
        labels.loc[mask] = str(bucket["name"])
        previous_max = max_step
    return labels


def distance_bucket_alphas(eval_step: pd.Series, buckets: list[dict[str, Any]]) -> np.ndarray:
    values = pd.to_numeric(eval_step, errors="coerce").to_numpy(dtype=float)
    alphas = np.ones(len(values), dtype=float)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        alpha = float(bucket["alpha"])
        mask = (values > previous_max) & (values <= max_step)
        alphas[mask] = alpha
        previous_max = max_step
    return alphas


def candidate_arrays(chunk: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    candidates = {
        "raw": chunk["y_pred"].to_numpy(dtype=float),
        "last_anchor": chunk["last_anchor"].to_numpy(dtype=float),
    }
    buckets = get_nested(config, "audit.postprocess_bucket_shrink.buckets") or []
    if buckets:
        alphas = distance_bucket_alphas(chunk["eval_step"], list(buckets))
        anchor = chunk["last_anchor"].to_numpy(dtype=float)
        raw = chunk["y_pred"].to_numpy(dtype=float)
        name = str(get_nested(config, "audit.postprocess_bucket_shrink.name") or "bucket_shrink")
        candidates[name] = anchor + alphas * (raw - anchor)
    return candidates


def update_segment_metrics(
    metrics: dict[tuple[str, str, str, str], MetricAccumulator],
    chunk: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    y_true = chunk["y_true"].to_numpy(dtype=float)
    wells = chunk["well_id"].to_numpy(dtype=object)
    candidates = candidate_arrays(chunk, config)

    segment_specs = [
        ("overall", pd.Series(["all"] * len(chunk), index=chunk.index, dtype=object)),
        ("original_fold", chunk["fold"].astype(str)),
        ("groupkfold_fold", chunk["groupkfold_fold"].astype(str)),
        ("stratified_groupkfold_fold", chunk["stratified_groupkfold_fold"].astype(str)),
        ("distance_bucket", chunk["distance_bucket"].astype(str)),
        ("azimuth_bin", chunk["azimuth_bin"].astype(str)),
        ("tvt_bin", chunk["tvt_bin"].astype(str)),
        ("spatial_bin", chunk["spatial_bin"].astype(str)),
        ("eval_length_bin", chunk["eval_length_bin"].astype(str)),
        ("gr_bin", chunk["gr_bin"].astype(str)),
        (
            "stratified_fold_x_distance_bucket",
            chunk["stratified_groupkfold_fold"].astype(str) + "|" + chunk["distance_bucket"].astype(str),
        ),
    ]

    for candidate_name, pred in candidates.items():
        for segment_type, labels in segment_specs:
            for segment_value, idx in labels.groupby(labels).groups.items():
                idx_array = np.asarray(list(idx), dtype=int)
                metrics[(candidate_name, segment_type, str(segment_value), "rmse")].add(
                    y_true[idx_array], pred[idx_array], wells[idx_array]
                )


def audit_oof_sources(
    assignments: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    allow_missing_oof: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = get_nested(config, "audit.oof_sources") or []
    if not sources:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merge_cols = [
        "well_id",
        "groupkfold_fold",
        "stratified_groupkfold_fold",
        "azimuth_bin",
        "tvt_bin",
        "spatial_bin",
        "eval_length_bin",
        "gr_bin",
    ]
    assignment_small = assignments[merge_cols].copy()
    distance_buckets = list(get_nested(config, "audit.distance_buckets") or [])
    chunk_rows = int(get_nested(config, "audit.chunk_rows") or 250000)
    skipped_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_well_rows: list[dict[str, Any]] = []

    for source in sources:
        source_name = str(source["name"])
        source_path = resolve_source_path(str(source["path"]))
        if source_path is None:
            row = {"source": source_name, "path": str(source["path"]), "reason": "missing"}
            skipped_rows.append(row)
            if allow_missing_oof:
                continue
            raise FileNotFoundError(row)

        variant = source.get("variant")
        accumulators: dict[tuple[str, str, str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
        well_accumulators: dict[tuple[str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
        rows_read = 0
        rows_used = 0

        for chunk in pd.read_csv(source_path, chunksize=chunk_rows):
            rows_read += len(chunk)
            if variant is not None and "variant" in chunk.columns:
                chunk = chunk[chunk["variant"].astype(str) == str(variant)].copy()
            required = {"well_id", "fold", "eval_step", "last_anchor", "y_true", "y_pred"}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")
            if chunk.empty:
                continue

            chunk = chunk.merge(assignment_small, on="well_id", how="inner", validate="many_to_one")
            if chunk.empty:
                continue
            if distance_buckets:
                chunk["distance_bucket"] = distance_bucket(chunk["eval_step"], distance_buckets)
            else:
                chunk["distance_bucket"] = "all"
            rows_used += len(chunk)

            update_segment_metrics(accumulators, chunk, config)
            candidates = candidate_arrays(chunk, config)
            y_true = chunk["y_true"].to_numpy(dtype=float)
            wells = chunk["well_id"].to_numpy(dtype=object)
            for candidate_name, pred in candidates.items():
                for well_id, idx in chunk.groupby("well_id", sort=False).groups.items():
                    idx_array = np.asarray(list(idx), dtype=int)
                    well_accumulators[(candidate_name, str(well_id))].add(
                        y_true[idx_array], pred[idx_array], wells[idx_array]
                    )

        for (candidate, segment_type, segment, _), bucket in sorted(accumulators.items()):
            all_metric_rows.append(
                bucket.row(
                    source=source_name,
                    variant=variant,
                    candidate=candidate,
                    segment_type=segment_type,
                    segment=segment,
                )
            )
        for (candidate, well_id), bucket in sorted(well_accumulators.items()):
            all_well_rows.append(
                bucket.row(source=source_name, variant=variant, candidate=candidate, well_id=well_id)
            )
        skipped_rows.append(
            {
                "source": source_name,
                "path": str(source_path),
                "variant": variant,
                "reason": "processed",
                "rows_read": rows_read,
                "rows_used": rows_used,
            }
        )

    metrics_df = pd.DataFrame(all_metric_rows)
    wells_df = pd.DataFrame(all_well_rows)
    source_df = pd.DataFrame(skipped_rows)
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "stratified_oof_segment_metrics.csv", index=False)
    if not wells_df.empty:
        wells_df.to_csv(output_dir / "stratified_oof_well_metrics.csv", index=False)
    source_df.to_csv(output_dir / "oof_source_status.csv", index=False)
    return metrics_df, wells_df, source_df


def run_audit(
    max_wells: int | None = None,
    allow_missing_oof: bool = True,
    skip_oof: bool = False,
) -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    output_dir = paths.artifacts_dir

    metadata = build_well_metadata(paths, config, max_wells=max_wells)
    assignments = assign_folds(metadata, config)
    fold_summary = fold_balance_summary(assignments)
    bucket_summary = bucket_distribution(assignments)

    assignments.to_csv(output_dir / "well_metadata_stratified_folds.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_balance_summary.csv", index=False)
    bucket_summary.to_csv(output_dir / "fold_bucket_distribution.csv", index=False)

    if skip_oof:
        oof_metrics = pd.DataFrame()
        well_metrics = pd.DataFrame()
        source_status = pd.DataFrame(
            [{"source": "all", "reason": "skip_oof requested", "rows_read": 0, "rows_used": 0}]
        )
        source_status.to_csv(output_dir / "oof_source_status.csv", index=False)
    else:
        oof_metrics, well_metrics, source_status = audit_oof_sources(
            assignments, config, output_dir, allow_missing_oof=allow_missing_oof
        )

    overall_rows = []
    if not oof_metrics.empty:
        overall_rows = (
            oof_metrics[oof_metrics["segment_type"] == "overall"]
            .sort_values(["source", "candidate"])
            .to_dict(orient="records")
        )

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "seed": get_nested(config, "validation.seed"),
        "n_folds": get_nested(config, "validation.n_folds"),
        "n_wells": int(len(assignments)),
        "strat_labels": int(assignments["strat_label"].nunique()),
        "artifacts": {
            "well_metadata_stratified_folds": "artifacts/well_metadata_stratified_folds.csv",
            "fold_balance_summary": "artifacts/fold_balance_summary.csv",
            "fold_bucket_distribution": "artifacts/fold_bucket_distribution.csv",
            "oof_source_status": "artifacts/oof_source_status.csv",
            "stratified_oof_segment_metrics": "artifacts/stratified_oof_segment_metrics.csv"
            if not oof_metrics.empty
            else None,
            "stratified_oof_well_metrics": "artifacts/stratified_oof_well_metrics.csv"
            if not well_metrics.empty
            else None,
        },
        "oof_sources": source_status.to_dict(orient="records"),
        "oof_overall": overall_rows,
        "notes": (
            "Diagnostic only. StratifiedGroupKFold is a stress-report split and does not "
            "replace the primary GroupKFold CV."
        ),
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified well GroupKFold audit.")
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--strict-oof", action="store_true", help="Fail if configured OOF is missing.")
    parser.add_argument("--skip-oof", action="store_true", help="Only build well metadata and folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_audit(
        max_wells=args.max_wells,
        allow_missing_oof=not args.strict_oof,
        skip_oof=args.skip_oof,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
