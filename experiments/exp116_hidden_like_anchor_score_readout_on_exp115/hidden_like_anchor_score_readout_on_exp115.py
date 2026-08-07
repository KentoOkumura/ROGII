from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from settings import (
    EXPERIMENT_NAME,
    ExperimentPaths,
    allow_local_notebook_execution,
    get_nested,
    is_kaggle_runtime,
    load_config,
)

OUTPUT_PREFIX = EXPERIMENT_NAME
ROLE_VALUE = "valid"
EPS = 1e-12
CSV_CHUNK_ROWS = 400_000
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


@dataclass(frozen=True)
class SourceStatus:
    source: str
    display_name: str
    kind: str
    status: str
    path: str | None
    rows: int
    candidate_count: int
    note: str
    size_bytes: int | None = None
    sha256: str | None = None
    decompressed_sha256: str | None = None


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    h = hashlib.sha256()
    opener = gzip.open if decompressed else open
    mode = "rb"
    with opener(path, mode) as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_gzip_path(path: Path) -> bool:
    return "".join(path.suffixes[-2:]) == ".csv.gz" or path.suffix == ".gz"


def resolve_path(paths: ExperimentPaths, value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return path
    return paths.root / path


def first_existing_path(paths: ExperimentPaths, candidates: list[Any]) -> Path | None:
    for candidate in candidates:
        path = resolve_path(paths, candidate)
        if path.exists() and path.is_file():
            return path
        if is_kaggle_runtime():
            basename = path.name
            matches = sorted(
                candidate for candidate in KAGGLE_INPUT_ROOT.rglob(basename) if candidate.is_file()
            )
            if matches:
                return matches[0]
    return None


def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def load_required_csv(
    paths: ExperimentPaths,
    candidates: list[Any],
    label: str,
) -> tuple[pd.DataFrame, Path]:
    path = first_existing_path(paths, candidates)
    if path is None:
        rendered = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"{label} not found. Candidates: {rendered}")
    return pd.read_csv(path), path


def metric_record(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            rows=("squared_error", "size"),
            wells=("well", "nunique"),
            sse=("squared_error", "sum"),
            abs_error_sum=("abs_error", "sum"),
        )
        .reset_index()
    )
    grouped["rmse_tvt"] = np.sqrt(grouped["sse"] / grouped["rows"].clip(lower=1))
    grouped["error_abs_mean"] = grouped["abs_error_sum"] / grouped["rows"].clip(lower=1)
    return grouped.drop(columns=["abs_error_sum"])


def error_sum_record(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            rows=("squared_error", "size"),
            sse=("squared_error", "sum"),
            abs_error_sum=("abs_error", "sum"),
        )
        .reset_index()
    )


def weighted_metric_record(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            rows=("rows", "sum"),
            wells=("well", "nunique"),
            sse=("sse", "sum"),
            abs_error_sum=("abs_error_sum", "sum"),
        )
        .reset_index()
    )
    grouped["rmse_tvt"] = np.sqrt(grouped["sse"] / grouped["rows"].clip(lower=1))
    grouped["error_abs_mean"] = grouped["abs_error_sum"] / grouped["rows"].clip(lower=1)
    return grouped.drop(columns=["abs_error_sum"])


def combine_error_sums(
    parts: list[pd.DataFrame],
    group_cols: list[str],
    *,
    add_wells: bool,
) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    frame = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            rows=("rows", "sum"),
            sse=("sse", "sum"),
            abs_error_sum=("abs_error_sum", "sum"),
        )
        .reset_index()
    )
    if add_wells:
        grouped["wells"] = 1 if "well" in grouped.columns else np.nan
    grouped["rmse_tvt"] = np.sqrt(grouped["sse"] / grouped["rows"].clip(lower=1))
    grouped["error_abs_mean"] = grouped["abs_error_sum"] / grouped["rows"].clip(lower=1)
    return grouped


def candidate_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["source"].astype(str)
        + "::"
        + frame["variant"].astype(str)
        + "::"
        + frame["mode"].astype(str)
        + "::"
        + frame["model"].astype(str)
    )


def parse_row_index(ids: pd.Series) -> pd.Series:
    suffix = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(suffix, errors="coerce")


def eval_rank_bucket(values: pd.Series, edges: list[int | float]) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    clean_edges = [float(x) for x in edges]
    bins = clean_edges + [math.inf]
    labels: list[str] = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        if math.isinf(right):
            labels.append(f"{int(left):04d}_inf")
        else:
            labels.append(f"{int(left):04d}_{int(right):03d}")
    return pd.cut(numeric, bins=bins, labels=labels, right=False, include_lowest=True)


def typewell_group_size_bin(values: pd.Series) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = [0, 1, 2, 3, 5, 10, math.inf]
    labels = ["size_1", "size_2", "size_3", "size_4_5", "size_6_10", "size_11_plus"]
    return pd.cut(numeric, bins=bins, labels=labels, right=True, include_lowest=True)


def normalize_meta(fold_assignments: pd.DataFrame, well_metadata: pd.DataFrame) -> pd.DataFrame:
    if "well_id" not in fold_assignments.columns:
        raise ValueError("fold assignments must contain well_id")
    if "well_id" not in well_metadata.columns:
        raise ValueError("well metadata must contain well_id")
    assignment_cols = [
        col for col in fold_assignments.columns if col == "well_id" or col.endswith("_role")
    ]
    drop_cols = [
        col for col in assignment_cols if col != "well_id" and col in well_metadata.columns
    ]
    metadata = well_metadata.drop(columns=drop_cols, errors="ignore")
    meta = metadata.merge(
        fold_assignments[assignment_cols],
        on="well_id",
        how="left",
        validate="1:1",
    )
    if "typewell_group_size" in meta.columns:
        meta["typewell_group_size_bin"] = typewell_group_size_bin(meta["typewell_group_size"])
    return meta


def add_context(frame: pd.DataFrame, meta: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.merge(meta, left_on="well", right_on="well_id", how="left", validate="m:1")
    out["row_index"] = parse_row_index(out["id"]) if "id" in out.columns else np.nan
    if "first_eval_row" in out.columns:
        out["eval_rank"] = out["row_index"] - pd.to_numeric(out["first_eval_row"], errors="coerce")
    else:
        out["eval_rank"] = np.nan
    edges = list(
        get_nested(config, "readout.bucket_edges_eval_rank") or [0, 50, 100, 250, 500, 1000]
    )
    out["eval_rank_bucket"] = eval_rank_bucket(out["eval_rank"], edges)
    return out


def source_input_status(
    source: dict[str, Any],
    status: str,
    path: Path | None,
    rows: int,
    candidate_count: int,
    note: str,
) -> SourceStatus:
    sha = None
    decompressed_sha = None
    size = None
    if path is not None and path.exists():
        size = path.stat().st_size
        sha = sha256_path(path)
        if is_gzip_path(path):
            decompressed_sha = sha256_path(path, decompressed=True)
    return SourceStatus(
        source=str(source.get("name")),
        display_name=str(source.get("display_name", source.get("name"))),
        kind=str(source.get("kind")),
        status=status,
        path=str(path) if path is not None else None,
        rows=rows,
        candidate_count=candidate_count,
        note=note,
        size_bytes=size,
        sha256=sha,
        decompressed_sha256=decompressed_sha,
    )


def normalize_row_predictions(
    source: dict[str, Any],
    path: Path,
    *,
    keep_wells: set[str] | None = None,
) -> pd.DataFrame:
    id_col = str(source.get("id_col", "id"))
    well_col = str(source.get("well_col", "well"))
    target_col = str(source.get("target_col", "target_tvt"))
    pred_col = str(source.get("prediction_col", "pred_tvt"))
    header = read_csv_header(path)
    required = [id_col, well_col, target_col, pred_col]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path} missing required row prediction columns: {missing}")
    optional = [col for col in ["variant", "mode", "model", "last_known_tvt"] if col in header]
    read_cols = required + optional
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=read_cols, chunksize=CSV_CHUNK_ROWS, low_memory=False):
        if keep_wells is not None:
            chunk = chunk.loc[chunk[well_col].astype(str).isin(keep_wells)]
        if not chunk.empty:
            chunks.append(chunk)
    if chunks:
        frame = pd.concat(chunks, ignore_index=True)
    else:
        frame = pd.DataFrame(columns=read_cols)
    frame = frame.rename(
        columns={
            id_col: "id",
            well_col: "well",
            target_col: "target_tvt",
            pred_col: "pred_tvt",
        }
    )
    frame["source"] = str(source["name"])
    frame["display_name"] = str(source.get("display_name", source["name"]))
    frame["input_kind"] = "row_predictions"
    if "variant" not in frame.columns:
        frame["variant"] = str(source.get("variant_value", source["name"]))
    if "mode" not in frame.columns:
        frame["mode"] = str(source.get("mode_value", "default"))
    if "model" not in frame.columns:
        frame["model"] = str(source.get("model_value", "prediction"))
    if source.get("variant_value") and "variant" not in optional:
        frame["variant"] = str(source["variant_value"])
    frame["target_tvt"] = pd.to_numeric(frame["target_tvt"], errors="coerce")
    frame["pred_tvt"] = pd.to_numeric(frame["pred_tvt"], errors="coerce")

    if source.get("include_model_mean", False):
        mean_name = str(source.get("model_mean_name", "model_mean"))
        group_cols = ["id", "well", "variant", "mode", "target_tvt"]
        if "last_known_tvt" in frame.columns:
            group_cols.append("last_known_tvt")
        mean_frame = frame.groupby(group_cols, dropna=False, observed=True, as_index=False).agg(
            pred_tvt=("pred_tvt", "mean")
        )
        mean_frame["source"] = str(source["name"])
        mean_frame["display_name"] = str(source.get("display_name", source["name"]))
        mean_frame["input_kind"] = "row_predictions"
        mean_frame["model"] = mean_name
        frame = pd.concat([frame, mean_frame[frame.columns]], ignore_index=True, copy=False)

    frame["candidate_key"] = candidate_key(frame)
    return frame


def normalize_by_well_metrics(source: dict[str, Any], path: Path) -> pd.DataFrame:
    well_col = str(source.get("well_col", "well"))
    rows_col = str(source.get("rows_col", "rows"))
    rmse_col = str(source.get("rmse_col", "rmse_tvt"))
    mae_col = str(source.get("mae_col", "error_abs_mean"))
    header = read_csv_header(path)
    required = [well_col, rows_col, rmse_col]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path} missing required by-well columns: {missing}")
    optional = [
        col
        for col in ["variant", "mode", "model", mae_col]
        if col in header and col not in required
    ]
    frame = pd.read_csv(path, usecols=required + optional, low_memory=False)
    frame = frame.rename(
        columns={
            well_col: "well",
            rows_col: "rows",
            rmse_col: "rmse_tvt",
            mae_col: "error_abs_mean",
        }
    )
    frame["source"] = str(source["name"])
    frame["display_name"] = str(source.get("display_name", source["name"]))
    frame["input_kind"] = "by_well_metrics"
    if "variant" not in frame.columns:
        frame["variant"] = str(source.get("variant_value", source["name"]))
    if "mode" not in frame.columns:
        frame["mode"] = str(source.get("mode_value", "default"))
    if "model" not in frame.columns:
        frame["model"] = str(source.get("model_value", "prediction"))
    if source.get("variant_value") and "variant" not in optional:
        frame["variant"] = str(source["variant_value"])
    frame["rows"] = pd.to_numeric(frame["rows"], errors="coerce").fillna(0).astype(int)
    frame["rmse_tvt"] = pd.to_numeric(frame["rmse_tvt"], errors="coerce")
    if "error_abs_mean" not in frame.columns:
        frame["error_abs_mean"] = np.nan
    frame["error_abs_mean"] = pd.to_numeric(frame["error_abs_mean"], errors="coerce")
    frame["sse"] = (frame["rmse_tvt"] ** 2) * frame["rows"]
    frame["abs_error_sum"] = frame["error_abs_mean"].fillna(0) * frame["rows"]
    frame["candidate_key"] = candidate_key(frame)
    return frame


def split_role_column(split_variant: str) -> str:
    return f"{split_variant}_role"


def filter_split(frame: pd.DataFrame, split_variant: str) -> pd.DataFrame:
    role_col = split_role_column(split_variant)
    if role_col not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame.loc[frame[role_col].astype(str) == ROLE_VALUE].copy()


def score_row_source(
    source_frame: pd.DataFrame,
    meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = add_context(source_frame, meta, config)
    context = context[np.isfinite(context["target_tvt"]) & np.isfinite(context["pred_tvt"])].copy()
    context["error"] = context["pred_tvt"] - context["target_tvt"]
    context["squared_error"] = context["error"] ** 2
    context["abs_error"] = context["error"].abs()
    splits = list(get_nested(config, "readout.exp115.split_variants") or [])
    base_cols = [
        "split_variant",
        "source",
        "display_name",
        "variant",
        "mode",
        "model",
        "candidate_key",
        "input_kind",
    ]
    overall_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    well_parts: list[pd.DataFrame] = []
    bucket_cols = list(get_nested(config, "readout.bucket_columns") or [])
    for split in splits:
        split_frame = filter_split(context, split)
        if split_frame.empty:
            continue
        split_frame["split_variant"] = split
        overall_parts.append(metric_record(split_frame, base_cols))
        well_group_cols = base_cols + ["well"]
        well_parts.append(metric_record(split_frame, well_group_cols))
        for bucket_col in bucket_cols:
            if bucket_col not in split_frame.columns:
                continue
            bucket_frame = split_frame.copy()
            bucket_frame["bucket_family"] = bucket_col
            bucket_frame["bucket"] = bucket_frame[bucket_col].astype(str)
            bucket_parts.append(
                metric_record(bucket_frame, base_cols + ["bucket_family", "bucket"])
            )
    return (
        pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame(),
        pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame(),
        pd.concat(well_parts, ignore_index=True) if well_parts else pd.DataFrame(),
    )


def normalize_row_chunk(source: dict[str, Any], chunk: pd.DataFrame) -> pd.DataFrame:
    id_col = str(source.get("id_col", "id"))
    well_col = str(source.get("well_col", "well"))
    target_col = str(source.get("target_col", "target_tvt"))
    pred_col = str(source.get("prediction_col", "pred_tvt"))
    frame = chunk.rename(
        columns={
            id_col: "id",
            well_col: "well",
            target_col: "target_tvt",
            pred_col: "pred_tvt",
        }
    )
    frame["source"] = str(source["name"])
    frame["display_name"] = str(source.get("display_name", source["name"]))
    frame["input_kind"] = "row_predictions"
    if "variant" not in frame.columns:
        frame["variant"] = str(source.get("variant_value", source["name"]))
    if "mode" not in frame.columns:
        frame["mode"] = str(source.get("mode_value", "default"))
    if "model" not in frame.columns:
        frame["model"] = str(source.get("model_value", "prediction"))
    frame["target_tvt"] = pd.to_numeric(frame["target_tvt"], errors="coerce")
    frame["pred_tvt"] = pd.to_numeric(frame["pred_tvt"], errors="coerce")
    frame["candidate_key"] = candidate_key(frame)
    return frame


def score_row_source_streaming(
    source: dict[str, Any],
    path: Path,
    meta: pd.DataFrame,
    config: dict[str, Any],
    keep_wells: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    id_col = str(source.get("id_col", "id"))
    well_col = str(source.get("well_col", "well"))
    target_col = str(source.get("target_col", "target_tvt"))
    pred_col = str(source.get("prediction_col", "pred_tvt"))
    header = read_csv_header(path)
    required = [id_col, well_col, target_col, pred_col]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path} missing required row prediction columns: {missing}")
    optional = [col for col in ["variant", "mode", "model", "last_known_tvt"] if col in header]
    read_cols = required + optional

    base_cols = [
        "split_variant",
        "source",
        "display_name",
        "variant",
        "mode",
        "model",
        "candidate_key",
        "input_kind",
    ]
    well_cols = base_cols + ["well"]
    bucket_well_cols = base_cols + ["bucket_family", "bucket", "well"]
    splits = list(get_nested(config, "readout.exp115.split_variants") or [])
    bucket_cols = list(get_nested(config, "readout.bucket_columns") or [])

    by_well_parts: list[pd.DataFrame] = []
    bucket_well_parts: list[pd.DataFrame] = []
    candidate_keys: set[str] = set()
    loaded_rows = 0

    for chunk in pd.read_csv(path, usecols=read_cols, chunksize=CSV_CHUNK_ROWS, low_memory=False):
        chunk = chunk.loc[chunk[well_col].astype(str).isin(keep_wells)]
        if chunk.empty:
            continue
        source_frame = normalize_row_chunk(source, chunk)
        source_frame = source_frame[
            np.isfinite(source_frame["target_tvt"]) & np.isfinite(source_frame["pred_tvt"])
        ].copy()
        if source_frame.empty:
            continue
        loaded_rows += int(len(source_frame))
        candidate_keys.update(source_frame["candidate_key"].astype(str).unique().tolist())
        context = add_context(source_frame, meta, config)
        context["error"] = context["pred_tvt"] - context["target_tvt"]
        context["squared_error"] = context["error"] ** 2
        context["abs_error"] = context["error"].abs()
        for split in splits:
            split_frame = filter_split(context, split)
            if split_frame.empty:
                continue
            split_frame["split_variant"] = split
            by_well_parts.append(error_sum_record(split_frame, well_cols))
            for bucket_col in bucket_cols:
                if bucket_col not in split_frame.columns:
                    continue
                bucket_frame = split_frame.copy()
                bucket_frame["bucket_family"] = bucket_col
                bucket_frame["bucket"] = bucket_frame[bucket_col].astype(str)
                bucket_well_parts.append(error_sum_record(bucket_frame, bucket_well_cols))

    by_well = combine_error_sums(by_well_parts, well_cols, add_wells=True)
    if not by_well.empty and "wells" in by_well.columns:
        by_well["wells"] = 1
    overall = weighted_metric_record(by_well, base_cols) if not by_well.empty else pd.DataFrame()
    bucket_by_well = combine_error_sums(bucket_well_parts, bucket_well_cols, add_wells=True)
    if not bucket_by_well.empty and "wells" in bucket_by_well.columns:
        bucket_by_well["wells"] = 1
    bucket = (
        weighted_metric_record(bucket_by_well, base_cols + ["bucket_family", "bucket"])
        if not bucket_by_well.empty
        else pd.DataFrame()
    )
    return overall, bucket, by_well, loaded_rows, len(candidate_keys)


def score_by_well_source(
    source_frame: pd.DataFrame,
    meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = source_frame.merge(
        meta,
        left_on="well",
        right_on="well_id",
        how="left",
        validate="m:1",
    )
    splits = list(get_nested(config, "readout.exp115.split_variants") or [])
    base_cols = [
        "split_variant",
        "source",
        "display_name",
        "variant",
        "mode",
        "model",
        "candidate_key",
        "input_kind",
    ]
    overall_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    well_parts: list[pd.DataFrame] = []
    bucket_cols = [
        col
        for col in list(get_nested(config, "readout.bucket_columns") or [])
        if col != "eval_rank_bucket"
    ]
    for split in splits:
        split_frame = filter_split(context, split)
        if split_frame.empty:
            continue
        split_frame["split_variant"] = split
        overall_parts.append(weighted_metric_record(split_frame, base_cols))
        well_parts.append(
            split_frame[
                base_cols
                + [
                    "well",
                    "rows",
                    "sse",
                    "rmse_tvt",
                    "error_abs_mean",
                ]
            ].copy()
        )
        for bucket_col in bucket_cols:
            if bucket_col not in split_frame.columns:
                continue
            bucket_frame = split_frame.copy()
            bucket_frame["bucket_family"] = bucket_col
            bucket_frame["bucket"] = bucket_frame[bucket_col].astype(str)
            bucket_parts.append(
                weighted_metric_record(
                    bucket_frame,
                    base_cols + ["bucket_family", "bucket"],
                )
            )
    return (
        pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame(),
        pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame(),
        pd.concat(well_parts, ignore_index=True) if well_parts else pd.DataFrame(),
    )


def build_worst_well_delta(by_well: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if by_well.empty:
        return pd.DataFrame()
    baseline = dict(get_nested(config, "readout.baseline_candidate") or {})
    required = ["source", "variant", "mode", "model"]
    if any(key not in baseline for key in required):
        return pd.DataFrame()
    mask = np.ones(len(by_well), dtype=bool)
    for key in required:
        mask &= by_well[key].astype(str).to_numpy() == str(baseline[key])
    base = by_well.loc[mask, ["split_variant", "well", "rows", "rmse_tvt", "sse"]].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.rename(
        columns={
            "rows": "baseline_rows",
            "rmse_tvt": "baseline_rmse_tvt",
            "sse": "baseline_sse",
        }
    )
    merged = by_well.merge(base, on=["split_variant", "well"], how="inner")
    merged["rmse_delta_vs_baseline"] = merged["rmse_tvt"] - merged["baseline_rmse_tvt"]
    merged["sse_delta_vs_baseline"] = merged["sse"] - merged["baseline_sse"]
    cols = [
        "split_variant",
        "source",
        "display_name",
        "variant",
        "mode",
        "model",
        "candidate_key",
        "well",
        "rows",
        "baseline_rows",
        "rmse_tvt",
        "baseline_rmse_tvt",
        "rmse_delta_vs_baseline",
        "sse_delta_vs_baseline",
        "input_kind",
    ]
    return merged[cols].sort_values(
        ["split_variant", "candidate_key", "rmse_delta_vs_baseline"],
        ascending=[True, True, False],
    )


def sort_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_candidates = [
        "split_variant",
        "source",
        "variant",
        "mode",
        "model",
        "bucket_family",
        "bucket",
        "well",
    ]
    cols = [col for col in sort_candidates if col in frame.columns]
    return frame.sort_values(cols).reset_index(drop=True) if cols else frame.reset_index(drop=True)


def write_outputs(
    paths: ExperimentPaths,
    config: dict[str, Any],
    inventory: list[SourceStatus],
    overall: pd.DataFrame,
    bucket: pd.DataFrame,
    by_well: pd.DataFrame,
    worst_delta: pd.DataFrame,
    split_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_prefix = str(get_nested(config, "readout.output_prefix") or OUTPUT_PREFIX)
    files = {
        "source_inventory": paths.artifacts_dir / f"{output_prefix}_source_inventory.csv",
        "overall_metrics": paths.artifacts_dir / f"{output_prefix}_overall_metrics.csv",
        "bucket_metrics": paths.artifacts_dir / f"{output_prefix}_bucket_metrics.csv",
        "by_well": paths.artifacts_dir / f"{output_prefix}_by_well.csv",
        "worst_well_delta": paths.artifacts_dir / f"{output_prefix}_worst_well_delta.csv",
        "summary": paths.artifacts_dir / f"{output_prefix}_summary.json",
    }
    inventory_df = pd.DataFrame([status.__dict__ for status in inventory])
    sort_output(inventory_df).to_csv(files["source_inventory"], index=False)
    sort_output(overall).to_csv(files["overall_metrics"], index=False)
    sort_output(bucket).to_csv(files["bucket_metrics"], index=False)
    sort_output(by_well).to_csv(files["by_well"], index=False)
    sort_output(worst_delta).to_csv(files["worst_well_delta"], index=False)

    best_rows: list[dict[str, Any]] = []
    if not overall.empty:
        for split, group in overall.groupby("split_variant", observed=True):
            best = group.sort_values("rmse_tvt").head(5)
            for _, row in best.iterrows():
                best_rows.append(
                    {
                        "split_variant": split,
                        "candidate_key": row["candidate_key"],
                        "rmse_tvt": float(row["rmse_tvt"]),
                        "rows": int(row["rows"]),
                        "wells": int(row["wells"]),
                        "input_kind": row["input_kind"],
                    }
                )

    summary = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": get_nested(config, "readout.mode"),
        "retrain_on_exp115_split": False,
        "split_inputs": split_inputs,
        "source_status": [status.__dict__ for status in inventory],
        "loaded_sources": int(sum(status.status == "loaded" for status in inventory)),
        "missing_sources": [status.source for status in inventory if status.status != "loaded"],
        "best_overall_by_split": best_rows,
        "artifacts": {key: str(path) for key, path in files.items()},
    }
    files["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    artifact_sha256 = {
        key: sha256_path(path, decompressed=is_gzip_path(path))
        for key, path in files.items()
        if path.exists()
    }
    summary["artifact_sha256"] = artifact_sha256
    files["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "implemented_kaggle_readout" if is_kaggle_runtime() else "implemented_local_readout"
        ),
        "metric": "rmse",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "readout": {
            "loaded_sources": summary["loaded_sources"],
            "missing_sources": summary["missing_sources"],
            "best_overall_by_split": best_rows,
            "artifacts": {key: str(path) for key, path in files.items()},
            "artifact_sha256": artifact_sha256,
        },
        "notes": [
            "No new model was trained.",
            (
                "Scores are stress readouts on exp115 hidden-like valid wells, "
                "not exact hidden LB estimates."
            ),
        ],
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return summary


def run_readout(*, allow_local: bool = False, max_sources: int | None = None) -> dict[str, Any]:
    if not is_kaggle_runtime() and not allow_local and not allow_local_notebook_execution():
        raise RuntimeError("Local execution requires --allow-local or EXPERIMENT_ALLOW_LOCAL=1.")

    paths = ExperimentPaths()
    config = load_config()
    fold_assignments, fold_path = load_required_csv(
        paths,
        list(get_nested(config, "readout.exp115.fold_assignments_path_candidates") or []),
        "exp115 fold assignments",
    )
    holdout_wells, holdout_path = load_required_csv(
        paths,
        list(get_nested(config, "readout.exp115.holdout_wells_path_candidates") or []),
        "exp115 holdout wells",
    )
    well_metadata, meta_path = load_required_csv(
        paths,
        list(get_nested(config, "readout.exp115.well_metadata_path_candidates") or []),
        "exp115 well metadata",
    )
    meta = normalize_meta(fold_assignments, well_metadata)
    keep_wells: set[str] = set()
    for split in list(get_nested(config, "readout.exp115.split_variants") or []):
        role_col = split_role_column(split)
        if role_col in meta.columns:
            valid = meta.loc[meta[role_col].astype(str) == ROLE_VALUE, "well_id"].astype(str)
            keep_wells.update(valid.tolist())
    split_inputs = {
        "fold_assignments": {
            "path": str(fold_path),
            "rows": int(len(fold_assignments)),
            "sha256": sha256_path(fold_path),
        },
        "holdout_wells": {
            "path": str(holdout_path),
            "rows": int(len(holdout_wells)),
            "sha256": sha256_path(holdout_path),
        },
        "well_metadata": {
            "path": str(meta_path),
            "rows": int(len(well_metadata)),
            "sha256": sha256_path(meta_path),
        },
    }

    sources = list(get_nested(config, "readout.prediction_sources") or [])
    if max_sources is None:
        configured_max = get_nested(config, "readout.max_sources")
        max_sources = int(configured_max) if configured_max is not None else None
    if max_sources is not None:
        sources = sources[:max_sources]

    inventory: list[SourceStatus] = []
    overall_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    by_well_parts: list[pd.DataFrame] = []

    for source in sources:
        path = first_existing_path(paths, list(source.get("path_candidates") or []))
        if path is None:
            inventory.append(
                source_input_status(source, "missing", None, 0, 0, "no path candidate exists")
            )
            continue
        try:
            kind = str(source.get("kind"))
            if kind == "row_predictions":
                overall, bucket, by_well, source_rows, candidate_count = score_row_source_streaming(
                    source,
                    path,
                    meta,
                    config,
                    keep_wells,
                )
            elif kind == "by_well_metrics":
                source_frame = normalize_by_well_metrics(source, path)
                overall, bucket, by_well = score_by_well_source(source_frame, meta, config)
                source_rows = int(len(source_frame))
                candidate_count = (
                    int(source_frame["candidate_key"].nunique())
                    if "candidate_key" in source_frame
                    else 0
                )
            else:
                raise ValueError(f"Unsupported source kind: {kind}")
        except (EmptyDataError, OSError, ValueError) as exc:
            inventory.append(source_input_status(source, "error", path, 0, 0, str(exc)))
            continue
        inventory.append(
            source_input_status(source, "loaded", path, source_rows, candidate_count, "ok")
        )
        overall_parts.append(overall)
        bucket_parts.append(bucket)
        by_well_parts.append(by_well)

    overall_all = pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame()
    bucket_all = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame()
    by_well_all = pd.concat(by_well_parts, ignore_index=True) if by_well_parts else pd.DataFrame()
    worst_delta = build_worst_well_delta(by_well_all, config)
    return write_outputs(
        paths,
        config,
        inventory,
        overall_all,
        bucket_all,
        by_well_all,
        worst_delta,
        split_inputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score existing anchors on exp115 hidden-like holdout."
    )
    parser.add_argument("--allow-local", action="store_true", help="Allow local smoke execution.")
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Limit number of configured sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_readout(allow_local=args.allow_local, max_sources=args.max_sources)
    print(json.dumps(summary, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
