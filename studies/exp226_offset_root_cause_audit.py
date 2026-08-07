"""Comprehensive root-cause audit for exp226's persistent TVT offsets.

This is a read-only diagnostic over the saved, group-safe exp226 OOF artifact.
It does not train a model, regenerate exp226, write a submission, or use truth
before any prediction is made.  Truth is used only for retrospective readout.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP226_NAME = "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
DEFAULT_ARTIFACT_DIR = (
    Path("/tmp/kaggle-output")
    / EXP226_NAME
    / "train_v1"
    / "artifacts"
)
DEFAULT_OUTPUT_DIR = ROOT / "studies" / "exp226_offset_root_cause_audit_20260727"
EXPECTED_OOF_ROWS = 3_783_989
EXPECTED_OOF_DECOMPRESSED_SHA256 = (
    "709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609"
)
K_SEGMENTS = 16
PERSISTENT_THRESHOLD_FT = 10.0
PERSISTENT_MIN_ROWS = 128


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(error)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(error[finite]))))


def mae(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(error)
    if not finite.any():
        return float("nan")
    return float(np.mean(np.abs(error[finite])))


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def metric_row(name: str, error: np.ndarray) -> dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    return {
        "name": name,
        "rows": int(len(values)),
        "rmse": rmse(values),
        "mae": mae(values),
        "bias_pred_minus_true": float(np.mean(values)),
        "error_std": float(np.std(values)),
        "within_5": float(np.mean(np.abs(values) <= 5.0)),
        "within_10": float(np.mean(np.abs(values) <= 10.0)),
        "within_25": float(np.mean(np.abs(values) <= 25.0)),
    }


def exact_k16_segment_id(offset: np.ndarray, n_rows: int) -> np.ndarray:
    edges = np.linspace(0, n_rows, K_SEGMENTS + 1)
    step_index = np.asarray(offset, dtype=np.float64) + 1.0
    return np.clip(
        np.searchsorted(edges[1:], step_index, side="left"),
        0,
        K_SEGMENTS - 1,
    ).astype(np.int8)


def add_k16_segments(oof: pd.DataFrame) -> pd.DataFrame:
    segment = np.empty(len(oof), dtype=np.int8)
    for positions in oof.groupby("well_id", sort=False).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        offset = oof["suffix_offset"].to_numpy(dtype=np.int64)[index]
        if not np.array_equal(offset, np.arange(len(index), dtype=np.int64)):
            raise ValueError("OOF suffix offsets are not contiguous and zero-based")
        segment[index] = exact_k16_segment_id(offset, len(index))
    oof["k16_segment"] = segment
    return oof


def load_inputs(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    oof_path = artifact_dir / f"{EXP226_NAME}_train_oof_predictions.csv.gz"
    by_well_path = artifact_dir / f"{EXP226_NAME}_by_well_metrics.csv"
    kappa_path = artifact_dir / f"{EXP226_NAME}_kappa_by_fold.csv"
    for path in (oof_path, by_well_path, kappa_path):
        if not path.exists():
            raise FileNotFoundError(path)

    decompressed_sha = sha256_decompressed_gzip(oof_path)
    if decompressed_sha != EXPECTED_OOF_DECOMPRESSED_SHA256:
        raise ValueError(
            "exp226 OOF decompressed SHA mismatch: "
            f"{decompressed_sha} != {EXPECTED_OOF_DECOMPRESSED_SHA256}"
        )
    columns = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_true",
        "tvt_pred",
        "tvt_geop",
        "gr_delta",
        "error",
        "fold",
    ]
    oof = pd.read_csv(
        oof_path,
        usecols=columns,
        dtype={
            "well_id": "string",
            "row_idx": "int32",
            "suffix_offset": "int32",
            "tvt_true": "float64",
            "tvt_pred": "float64",
            "tvt_geop": "float64",
            "gr_delta": "float64",
            "error": "float64",
            "fold": "int8",
        },
    )
    if len(oof) != EXPECTED_OOF_ROWS:
        raise ValueError(f"unexpected exp226 OOF rows: {len(oof)}")
    expected_error = oof["tvt_pred"] - oof["tvt_true"]
    if float(np.max(np.abs(expected_error - oof["error"]))) > 1e-9:
        raise ValueError("saved exp226 error column is inconsistent")
    if oof[["tvt_true", "tvt_pred", "tvt_geop", "gr_delta"]].isna().any().any():
        raise ValueError("exp226 OOF has non-finite required values")
    if oof.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 OOF has duplicate well/row keys")

    oof["tvt_pre_u"] = oof["tvt_geop"] + oof["gr_delta"]
    oof["u_adjust"] = oof["tvt_pred"] - oof["tvt_pre_u"]
    oof["error_geop"] = oof["tvt_geop"] - oof["tvt_true"]
    oof["error_pre_u"] = oof["tvt_pre_u"] - oof["tvt_true"]
    oof = add_k16_segments(oof)

    by_well = pd.read_csv(by_well_path)
    kappa = pd.read_csv(kappa_path)
    manifest = {
        "oof_path": str(oof_path),
        "oof_file_sha256": sha256_file(oof_path),
        "oof_decompressed_sha256": decompressed_sha,
        "by_well_path": str(by_well_path),
        "by_well_sha256": sha256_file(by_well_path),
        "kappa_path": str(kappa_path),
        "kappa_sha256": sha256_file(kappa_path),
        "rows": int(len(oof)),
        "wells": int(oof["well_id"].nunique()),
        "folds": sorted(int(value) for value in oof["fold"].unique()),
    }
    return oof, by_well, kappa, manifest


def stage_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stages = {
        "geometry_tvt_geop": oof["error_geop"].to_numpy(dtype=np.float64),
        "geometry_plus_gr_pre_u": oof["error_pre_u"].to_numpy(dtype=np.float64),
        "final_post_u": oof["error"].to_numpy(dtype=np.float64),
    }
    rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    same_well = (
        oof["well_id"].to_numpy()[1:] == oof["well_id"].to_numpy()[:-1]
    )
    true = oof["tvt_true"].to_numpy(dtype=np.float64)
    for name, error in stages.items():
        row = metric_row(name, error)
        prediction = true + error
        increment_error = np.diff(prediction)[same_well] - np.diff(true)[same_well]
        row.update(
            {
                "increment_rows": int(len(increment_error)),
                "increment_rmse": rmse(increment_error),
                "increment_mae": mae(increment_error),
                "increment_bias": float(np.mean(increment_error)),
            }
        )
        rows.append(row)

        temporary = pd.DataFrame(
            {
                "well_id": oof["well_id"],
                "fold": oof["fold"],
                "error": error,
            }
        )
        for (well_id, fold), part in temporary.groupby(
            ["well_id", "fold"], sort=False, observed=True
        ):
            well_rows.append(
                {
                    "stage": name,
                    "well_id": str(well_id),
                    "fold": int(fold),
                    "rows": int(len(part)),
                    "rmse": rmse(part["error"].to_numpy()),
                    "mae": mae(part["error"].to_numpy()),
                    "bias": float(part["error"].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(well_rows)


def component_effects(
    oof: pd.DataFrame, well_stage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geo = oof["error_geop"].to_numpy(dtype=np.float64)
    pre_u = oof["error_pre_u"].to_numpy(dtype=np.float64)
    final = oof["error"].to_numpy(dtype=np.float64)
    gr = oof["gr_delta"].to_numpy(dtype=np.float64)
    u_adjust = oof["u_adjust"].to_numpy(dtype=np.float64)
    rows = [
        {
            "component": "gr_delta",
            "adjustment_mean": float(np.mean(gr)),
            "adjustment_abs_mean": float(np.mean(np.abs(gr))),
            "adjustment_abs_p95": float(np.quantile(np.abs(gr), 0.95)),
            "adjustment_cap_3p99_fraction": float(np.mean(np.abs(gr) >= 3.99)),
            "row_abs_error_improvement_fraction": float(
                np.mean(np.abs(pre_u) < np.abs(geo))
            ),
            "row_abs_error_worsening_fraction": float(
                np.mean(np.abs(pre_u) > np.abs(geo))
            ),
            "sse_before": float(np.sum(np.square(geo))),
            "sse_after": float(np.sum(np.square(pre_u))),
            "sse_change_after_minus_before": float(
                np.sum(np.square(pre_u)) - np.sum(np.square(geo))
            ),
            "adjustment_vs_needed_correction_pearson": pearson(gr, -geo),
        },
        {
            "component": "u_projection",
            "adjustment_mean": float(np.mean(u_adjust)),
            "adjustment_abs_mean": float(np.mean(np.abs(u_adjust))),
            "adjustment_abs_p95": float(np.quantile(np.abs(u_adjust), 0.95)),
            "adjustment_cap_3p99_fraction": float(
                np.mean(np.abs(u_adjust) >= 3.99)
            ),
            "row_abs_error_improvement_fraction": float(
                np.mean(np.abs(final) < np.abs(pre_u))
            ),
            "row_abs_error_worsening_fraction": float(
                np.mean(np.abs(final) > np.abs(pre_u))
            ),
            "sse_before": float(np.sum(np.square(pre_u))),
            "sse_after": float(np.sum(np.square(final))),
            "sse_change_after_minus_before": float(
                np.sum(np.square(final)) - np.sum(np.square(pre_u))
            ),
            "adjustment_vs_needed_correction_pearson": pearson(
                u_adjust, -pre_u
            ),
        },
    ]

    pivot = well_stage.pivot(
        index=["well_id", "fold"], columns="stage", values="rmse"
    ).reset_index()
    pivot["gr_rmse_delta"] = (
        pivot["geometry_plus_gr_pre_u"] - pivot["geometry_tvt_geop"]
    )
    pivot["u_rmse_delta"] = (
        pivot["final_post_u"] - pivot["geometry_plus_gr_pre_u"]
    )
    fold_rows: list[dict[str, float]] = []
    for fold, part in oof.groupby("fold", sort=True, observed=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "geometry_tvt_geop": rmse(part["error_geop"].to_numpy()),
                "geometry_plus_gr_pre_u": rmse(
                    part["error_pre_u"].to_numpy()
                ),
                "final_post_u": rmse(part["error"].to_numpy()),
            }
        )
    fold_stage = pd.DataFrame(fold_rows)
    fold_stage["gr_rmse_delta"] = (
        fold_stage["geometry_plus_gr_pre_u"]
        - fold_stage["geometry_tvt_geop"]
    )
    fold_stage["u_rmse_delta"] = (
        fold_stage["final_post_u"]
        - fold_stage["geometry_plus_gr_pre_u"]
    )
    well_component = pd.DataFrame(
        [
            {
                "component": "gr_delta",
                "wells_improved": int((pivot["gr_rmse_delta"] < 0).sum()),
                "wells_worsened": int((pivot["gr_rmse_delta"] > 0).sum()),
                "well_rmse_delta_median": float(
                    pivot["gr_rmse_delta"].median()
                ),
                "folds_improved": int(
                    (fold_stage["gr_rmse_delta"] < 0).sum()
                ),
            },
            {
                "component": "u_projection",
                "wells_improved": int((pivot["u_rmse_delta"] < 0).sum()),
                "wells_worsened": int((pivot["u_rmse_delta"] > 0).sum()),
                "well_rmse_delta_median": float(
                    pivot["u_rmse_delta"].median()
                ),
                "folds_improved": int(
                    (fold_stage["u_rmse_delta"] < 0).sum()
                ),
            },
        ]
    )
    return pd.DataFrame(rows), well_component


def suffix_bucket_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    absolute_edges = [0, 50, 100, 250, 500, 1000, 2000, np.inf]
    absolute_labels = [
        "0000_0050",
        "0050_0100",
        "0100_0250",
        "0250_0500",
        "0500_1000",
        "1000_2000",
        "2000_plus",
    ]
    frame = oof[
        [
            "well_id",
            "suffix_offset",
            "error_geop",
            "error_pre_u",
            "error",
        ]
    ].copy()
    frame["absolute_bucket"] = pd.cut(
        frame["suffix_offset"],
        bins=absolute_edges,
        labels=absolute_labels,
        right=False,
    )
    well_length = frame.groupby("well_id", sort=False, observed=True)[
        "suffix_offset"
    ].transform("max") + 1
    normalized = frame["suffix_offset"] / np.maximum(well_length - 1, 1)
    frame["normalized_decile"] = np.minimum(
        np.floor(normalized * 10).astype(np.int8), 9
    )
    stages = {
        "geometry_tvt_geop": "error_geop",
        "geometry_plus_gr_pre_u": "error_pre_u",
        "final_post_u": "error",
    }
    rows: list[dict[str, Any]] = []
    for bucket_kind in ("absolute_bucket", "normalized_decile"):
        for bucket, part in frame.groupby(
            bucket_kind, observed=True, sort=True
        ):
            for stage, error_column in stages.items():
                row = metric_row(stage, part[error_column].to_numpy())
                row.update(
                    {
                        "bucket_kind": bucket_kind,
                        "bucket": str(bucket),
                        "stage": stage,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def grouped_sufficient_statistics(
    frame: pd.DataFrame, keys: list[str], x_column: str
) -> pd.DataFrame:
    temporary = frame[keys + [x_column, "error"]].copy()
    temporary["error_sq"] = np.square(temporary["error"])
    temporary["x_sq"] = np.square(temporary[x_column].astype(np.float64))
    temporary["x_error"] = (
        temporary[x_column].astype(np.float64) * temporary["error"]
    )
    grouped = temporary.groupby(keys, sort=False, observed=True)
    result = grouped.agg(
        rows=("error", "size"),
        error_sum=("error", "sum"),
        error_sq_sum=("error_sq", "sum"),
        x_sum=(x_column, "sum"),
        x_sq_sum=("x_sq", "sum"),
        x_error_sum=("x_error", "sum"),
        start_error=("error", "first"),
        end_error=("error", "last"),
    ).reset_index()
    result["mean_error"] = result["error_sum"] / result["rows"]
    centered_y_sse = (
        result["error_sq_sum"]
        - np.square(result["error_sum"]) / result["rows"]
    )
    centered_x_ss = (
        result["x_sq_sum"] - np.square(result["x_sum"]) / result["rows"]
    )
    centered_xy = (
        result["x_error_sum"]
        - result["x_sum"] * result["error_sum"] / result["rows"]
    )
    result["mean_corrected_sse"] = np.maximum(centered_y_sse, 0.0)
    result["start_corrected_sse"] = np.maximum(
        result["error_sq_sum"]
        - 2.0 * result["start_error"] * result["error_sum"]
        + result["rows"] * np.square(result["start_error"]),
        0.0,
    )
    affine_gain = np.divide(
        np.square(centered_xy),
        centered_x_ss,
        out=np.zeros(len(result), dtype=np.float64),
        where=centered_x_ss > 0,
    )
    result["affine_corrected_sse"] = np.maximum(
        centered_y_sse - affine_gain, 0.0
    )
    result["error_slope_per_row"] = np.divide(
        centered_xy,
        centered_x_ss,
        out=np.zeros(len(result), dtype=np.float64),
        where=centered_x_ss > 0,
    )
    return result


def oracle_quotient_metrics(
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_error = oof["error"].to_numpy(dtype=np.float64)
    base_sse = float(np.sum(np.square(base_error)))
    total_rows = int(len(oof))
    rows: list[dict[str, Any]] = []

    def add(name: str, corrected_sse: float, groups: int) -> None:
        corrected_sse = max(float(corrected_sse), 0.0)
        rows.append(
            {
                "quotient": name,
                "groups": int(groups),
                "rows": total_rows,
                "rmse_after_oracle_removal": math.sqrt(
                    corrected_sse / total_rows
                ),
                "mse_fraction_remaining": corrected_sse / base_sse,
                "mse_fraction_explained": 1.0 - corrected_sse / base_sse,
            }
        )

    global_sse = float(
        np.sum(np.square(base_error - np.mean(base_error)))
    )
    add("global_mean_bias", global_sse, 1)

    frame = oof[
        ["well_id", "suffix_offset", "k16_segment", "error"]
    ].copy()
    well = grouped_sufficient_statistics(
        frame, ["well_id"], "suffix_offset"
    )
    add("well_mean_offset", well["mean_corrected_sse"].sum(), len(well))
    add("well_start_reanchor", well["start_corrected_sse"].sum(), len(well))
    add("well_affine_offset_slope", well["affine_corrected_sse"].sum(), len(well))

    segment = grouped_sufficient_statistics(
        frame, ["well_id", "k16_segment"], "suffix_offset"
    )
    add(
        "k16_segment_mean_offset",
        segment["mean_corrected_sse"].sum(),
        len(segment),
    )
    add(
        "k16_segment_start_reanchor",
        segment["start_corrected_sse"].sum(),
        len(segment),
    )
    add(
        "k16_segment_affine_offset_slope",
        segment["affine_corrected_sse"].sum(),
        len(segment),
    )

    for block_size in (64, 128, 256, 512, 1024):
        block_column = f"block_{block_size}"
        frame[block_column] = (
            frame["suffix_offset"].to_numpy(dtype=np.int64) // block_size
        )
        block = grouped_sufficient_statistics(
            frame, ["well_id", block_column], "suffix_offset"
        )
        add(
            f"h{block_size}_block_mean_offset",
            block["mean_corrected_sse"].sum(),
            len(block),
        )
        add(
            f"h{block_size}_block_affine_offset_slope",
            block["affine_corrected_sse"].sum(),
            len(block),
        )

    for stage, error_column in (
        ("geometry_tvt_geop", "error_geop"),
        ("geometry_plus_gr_pre_u", "error_pre_u"),
        ("final_post_u", "error"),
    ):
        stage_frame = oof[
            ["well_id", "suffix_offset", "k16_segment", error_column]
        ].rename(columns={error_column: "error"})
        stage_segment = grouped_sufficient_statistics(
            stage_frame, ["well_id", "k16_segment"], "suffix_offset"
        )
        stage_sse = float(np.sum(np.square(stage_frame["error"])))
        corrected_sse = float(stage_segment["mean_corrected_sse"].sum())
        rows.append(
            {
                "quotient": f"{stage}__k16_segment_mean_offset",
                "groups": int(len(stage_segment)),
                "rows": total_rows,
                "rmse_after_oracle_removal": math.sqrt(
                    corrected_sse / total_rows
                ),
                "mse_fraction_remaining": corrected_sse / stage_sse,
                "mse_fraction_explained": 1.0 - corrected_sse / stage_sse,
            }
        )
    return pd.DataFrame(rows), segment


def segment_persistence(segment: pd.DataFrame) -> pd.DataFrame:
    ordered = segment.sort_values(["well_id", "k16_segment"]).copy()
    ordered["previous_end_error"] = ordered.groupby(
        "well_id", sort=False, observed=True
    )["end_error"].shift(1)
    ordered["previous_mean_error"] = ordered.groupby(
        "well_id", sort=False, observed=True
    )["mean_error"].shift(1)
    valid_previous = ordered["previous_end_error"].notna()
    weights = ordered.loc[valid_previous, "rows"].to_numpy(dtype=np.float64)
    previous_prediction_error = (
        ordered.loc[valid_previous, "mean_error"].to_numpy(dtype=np.float64)
        - ordered.loc[valid_previous, "previous_end_error"].to_numpy(
            dtype=np.float64
        )
    )
    weighted_previous_rmse = math.sqrt(
        float(np.sum(weights * np.square(previous_prediction_error)))
        / float(np.sum(weights))
    )
    boundary_jump = (
        ordered.loc[valid_previous, "start_error"]
        - ordered.loc[valid_previous, "previous_end_error"]
    )
    within_drift = ordered["end_error"] - ordered["start_error"]
    rows = [
        {
            "metric": "segment_mean_vs_start_error_pearson",
            "value": pearson(ordered["mean_error"], ordered["start_error"]),
        },
        {
            "metric": "segment_mean_vs_previous_end_error_pearson",
            "value": pearson(
                ordered.loc[valid_previous, "mean_error"],
                ordered.loc[valid_previous, "previous_end_error"],
            ),
        },
        {
            "metric": "segment_mean_vs_previous_mean_error_pearson",
            "value": pearson(
                ordered.loc[valid_previous, "mean_error"],
                ordered.loc[valid_previous, "previous_mean_error"],
            ),
        },
        {
            "metric": "previous_end_as_segment_mean_weighted_rmse",
            "value": weighted_previous_rmse,
        },
        {
            "metric": "boundary_error_jump_abs_median",
            "value": float(np.median(np.abs(boundary_jump))),
        },
        {
            "metric": "boundary_error_jump_abs_p95",
            "value": float(np.quantile(np.abs(boundary_jump), 0.95)),
        },
        {
            "metric": "within_segment_error_drift_abs_median",
            "value": float(np.median(np.abs(within_drift))),
        },
        {
            "metric": "within_segment_error_drift_abs_p95",
            "value": float(np.quantile(np.abs(within_drift), 0.95)),
        },
        {
            "metric": "segments_abs_mean_error_ge_10_fraction",
            "value": float(np.mean(np.abs(ordered["mean_error"]) >= 10.0)),
        },
        {
            "metric": "segment_start_end_same_sign_fraction",
            "value": float(
                np.mean(
                    np.sign(ordered["start_error"])
                    == np.sign(ordered["end_error"])
                )
            ),
        },
    ]
    return pd.DataFrame(rows)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def persistent_episode_readout(
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, Any]] = []
    total_sse = float(np.sum(np.square(oof["error"].to_numpy(dtype=np.float64))))
    for well_id, part in oof.groupby("well_id", sort=False, observed=True):
        part = part.sort_values("suffix_offset")
        offset = part["suffix_offset"].to_numpy(dtype=np.int64)
        if not np.array_equal(offset, np.arange(len(part), dtype=np.int64)):
            raise ValueError(f"non-contiguous suffix for well {well_id}")
        final_error = part["error"].to_numpy(dtype=np.float64)
        geo_error = part["error_geop"].to_numpy(dtype=np.float64)
        pre_u_error = part["error_pre_u"].to_numpy(dtype=np.float64)
        gr_delta = part["gr_delta"].to_numpy(dtype=np.float64)
        u_adjust = part["u_adjust"].to_numpy(dtype=np.float64)
        segment = part["k16_segment"].to_numpy(dtype=np.int8)
        segment_starts = np.where(np.r_[True, segment[1:] != segment[:-1]])[0]
        for start, end in contiguous_true_runs(
            np.abs(final_error) >= PERSISTENT_THRESHOLD_FT
        ):
            if end - start < PERSISTENT_MIN_ROWS:
                continue
            sl = slice(start, end)
            before = max(0, start - 64)
            denominator = max(start - before, 1)
            if abs(geo_error[start]) >= PERSISTENT_THRESHOLD_FT:
                crossing_component = "geometry_already_over_threshold"
            elif abs(pre_u_error[start]) >= PERSISTENT_THRESHOLD_FT:
                crossing_component = "gr_crossed_threshold"
            else:
                crossing_component = "u_projection_crossed_threshold"
            episode_rows.append(
                {
                    "well_id": str(well_id),
                    "fold": int(part["fold"].iloc[0]),
                    "start_suffix_offset": int(start),
                    "end_suffix_offset_exclusive": int(end),
                    "rows": int(end - start),
                    "signed_error_median": float(np.median(final_error[sl])),
                    "abs_error_mean": float(np.mean(np.abs(final_error[sl]))),
                    "abs_error_max": float(np.max(np.abs(final_error[sl]))),
                    "episode_sse": float(np.sum(np.square(final_error[sl]))),
                    "geometry_episode_rmse": rmse(geo_error[sl]),
                    "pre_u_episode_rmse": rmse(pre_u_error[sl]),
                    "final_episode_rmse": rmse(final_error[sl]),
                    "crossing_component": crossing_component,
                    "onset_error": float(final_error[start]),
                    "onset_geometry_error": float(geo_error[start]),
                    "onset_gr_delta": float(gr_delta[start]),
                    "onset_u_adjust": float(u_adjust[start]),
                    "onset_one_row_error_change": float(
                        final_error[start]
                        - (final_error[start - 1] if start else 0.0)
                    ),
                    "pre_onset_64row_error_rate": float(
                        (final_error[start] - final_error[before])
                        / denominator
                    ),
                    "distance_to_k16_boundary_rows": int(
                        np.min(np.abs(segment_starts - start))
                    ),
                    "gr_cap_fraction": float(
                        np.mean(np.abs(gr_delta[sl]) >= 3.99)
                    ),
                    "u_worsens_episode_sse": bool(
                        np.sum(np.square(final_error[sl]))
                        > np.sum(np.square(pre_u_error[sl]))
                    ),
                    "gr_worsens_episode_sse": bool(
                        np.sum(np.square(pre_u_error[sl]))
                        > np.sum(np.square(geo_error[sl]))
                    ),
                }
            )
    episodes = pd.DataFrame(episode_rows)
    if episodes.empty:
        raise ValueError("no persistent exp226 offset episodes found")

    summary_rows = [
        {"metric": "episodes", "value": int(len(episodes))},
        {
            "metric": "wells_with_episode",
            "value": int(episodes["well_id"].nunique()),
        },
        {
            "metric": "episode_sse_fraction_of_total",
            "value": float(episodes["episode_sse"].sum() / total_sse),
        },
        {
            "metric": "episode_rows",
            "value": int(episodes["rows"].sum()),
        },
        {
            "metric": "episode_rows_fraction_of_total",
            "value": float(episodes["rows"].sum() / len(oof)),
        },
        {
            "metric": "geometry_already_over_threshold_fraction",
            "value": float(
                np.mean(
                    episodes["crossing_component"]
                    == "geometry_already_over_threshold"
                )
            ),
        },
        {
            "metric": "gr_crossed_threshold_fraction",
            "value": float(
                np.mean(episodes["crossing_component"] == "gr_crossed_threshold")
            ),
        },
        {
            "metric": "u_projection_crossed_threshold_fraction",
            "value": float(
                np.mean(
                    episodes["crossing_component"]
                    == "u_projection_crossed_threshold"
                )
            ),
        },
        {
            "metric": "gr_worsens_episode_sse_fraction",
            "value": float(episodes["gr_worsens_episode_sse"].mean()),
        },
        {
            "metric": "u_worsens_episode_sse_fraction",
            "value": float(episodes["u_worsens_episode_sse"].mean()),
        },
        {
            "metric": "onset_distance_to_k16_boundary_median",
            "value": float(episodes["distance_to_k16_boundary_rows"].median()),
        },
        {
            "metric": "onset_distance_to_k16_boundary_p90",
            "value": float(
                episodes["distance_to_k16_boundary_rows"].quantile(0.9)
            ),
        },
        {
            "metric": "onset_one_row_change_abs_median",
            "value": float(
                episodes["onset_one_row_error_change"].abs().median()
            ),
        },
        {
            "metric": "pre_onset_64row_rate_abs_median",
            "value": float(
                episodes["pre_onset_64row_error_rate"].abs().median()
            ),
        },
    ]
    return episodes, pd.DataFrame(summary_rows)


def raw_well_features(raw_train_dir: Path, well_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_train_dir.glob("*__horizontal_well.csv")):
        well_id = path.name.split("__")[0]
        if well_id not in well_ids:
            continue
        frame = pd.read_csv(
            path,
            usecols=["X", "Y", "Z", "TVT", "TVT_input", "GR"],
        )
        tvt_input = frame["TVT_input"].to_numpy(dtype=np.float64)
        known = np.where(np.isfinite(tvt_input))[0]
        if not len(known):
            raise ValueError(f"well {well_id} has no TVT_input anchor")
        s = int(known.max())
        suffix = frame.iloc[s + 1 :]
        prefix = frame.iloc[: s + 1]
        tvt = frame["TVT"].to_numpy(dtype=np.float64)
        z = frame["Z"].to_numpy(dtype=np.float64)
        x = frame["X"].to_numpy(dtype=np.float64)
        y = frame["Y"].to_numpy(dtype=np.float64)
        u_true = tvt[s + 1 :] + z[s + 1 :]
        xy_step = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)))
        rows.append(
            {
                "well_id": well_id,
                "known_prefix_rows": int(s + 1),
                "raw_suffix_rows": int(len(suffix)),
                "suffix_gr_missing_fraction": float(suffix["GR"].isna().mean()),
                "prefix_gr_missing_fraction": float(prefix["GR"].isna().mean()),
                "suffix_z_range": float(np.ptp(z[s + 1 :])),
                "suffix_tvt_range": float(np.ptp(tvt[s + 1 :])),
                "suffix_u_true_range": float(np.ptp(u_true)),
                "suffix_u_true_end_minus_start": float(u_true[-1] - u_true[0]),
                "true_tvt_end_minus_anchor": float(tvt[-1] - tvt[s]),
                "true_first_suffix_minus_anchor": float(tvt[s + 1] - tvt[s]),
                "z_first_suffix_minus_anchor": float(z[s + 1] - z[s]),
                "whole_xy_path_length": float(np.sum(xy_step)),
                "suffix_xy_displacement": float(
                    np.hypot(x[-1] - x[s], y[-1] - y[s])
                ),
            }
        )
    result = pd.DataFrame(rows)
    missing = well_ids - set(result["well_id"])
    if missing:
        raise ValueError(f"raw train wells missing: {sorted(missing)[:5]}")
    return result


def build_well_readout(
    oof: pd.DataFrame,
    by_well: pd.DataFrame,
    raw_features: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well_id, part in oof.groupby("well_id", sort=False, observed=True):
        error = part["error"].to_numpy(dtype=np.float64)
        offset = part["suffix_offset"].to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(offset)), offset])
        intercept, slope = np.linalg.lstsq(design, error, rcond=None)[0]
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(part["fold"].iloc[0]),
                "oof_rows": int(len(part)),
                "final_rmse_recomputed": rmse(error),
                "final_mae_recomputed": mae(error),
                "final_bias_recomputed": float(np.mean(error)),
                "final_abs_bias": float(abs(np.mean(error))),
                "final_first_error": float(error[0]),
                "final_end_error": float(error[-1]),
                "final_error_slope_per_row": float(slope),
                "geometry_rmse": rmse(part["error_geop"].to_numpy()),
                "pre_u_rmse": rmse(part["error_pre_u"].to_numpy()),
                "gr_rmse_delta": rmse(part["error_pre_u"].to_numpy())
                - rmse(part["error_geop"].to_numpy()),
                "u_rmse_delta": rmse(error)
                - rmse(part["error_pre_u"].to_numpy()),
                "gr_cap_fraction_rows": float(
                    np.mean(np.abs(part["gr_delta"]) >= 3.99)
                ),
                "u_adjust_abs_mean": float(np.mean(np.abs(part["u_adjust"]))),
                "well_affine_intercept": float(intercept),
            }
        )
    well = pd.DataFrame(rows)
    by_well_subset = by_well.drop(
        columns=[
            column
            for column in ("label", "rows", "rmse", "mae", "bias", "fold")
            if column in by_well.columns
        ]
    )
    well = well.merge(
        by_well_subset,
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    well = well.merge(
        raw_features,
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    episode_well = (
        episodes.groupby("well_id", sort=False, observed=True)
        .agg(
            persistent_episode_count=("well_id", "size"),
            persistent_episode_rows=("rows", "sum"),
            persistent_episode_sse=("episode_sse", "sum"),
        )
        .reset_index()
    )
    well = well.merge(
        episode_well, on="well_id", how="left", validate="one_to_one"
    )
    for column in (
        "persistent_episode_count",
        "persistent_episode_rows",
        "persistent_episode_sse",
    ):
        well[column] = well[column].fillna(0)
    if well.isna().any().any():
        missing_columns = well.columns[well.isna().any()].tolist()
        raise ValueError(f"well readout has missing columns: {missing_columns}")
    if not np.array_equal(
        well["oof_rows"].to_numpy(dtype=np.int64),
        well["raw_suffix_rows"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("raw/OOF suffix-row parity failed")
    return well


def driver_correlations(well: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "final_rmse_recomputed",
        "final_abs_bias",
        "final_error_slope_per_row",
        "persistent_episode_sse",
        "gr_rmse_delta",
        "u_rmse_delta",
    ]
    drivers = [
        "unknown_rows",
        "known_prefix_rows",
        "donor_dist_min",
        "donor_dist_max",
        "gate_segments",
        "delta_abs_median",
        "delta_abs_max",
        "end_minus_anchor",
        "suffix_gr_missing_fraction",
        "prefix_gr_missing_fraction",
        "suffix_z_range",
        "suffix_tvt_range",
        "suffix_u_true_range",
        "suffix_u_true_end_minus_start",
        "true_tvt_end_minus_anchor",
        "whole_xy_path_length",
        "suffix_xy_displacement",
        "gr_cap_fraction_rows",
        "u_adjust_abs_mean",
    ]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for driver in drivers:
            rows.append(
                {
                    "outcome": outcome,
                    "driver": driver,
                    "spearman": spearman(well[outcome], well[driver]),
                    "abs_spearman": abs(
                        spearman(well[outcome], well[driver])
                    ),
                    "wells": int(len(well)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["outcome", "abs_spearman"], ascending=[True, False]
    )


def driver_quantile_contrasts(well: pd.DataFrame) -> pd.DataFrame:
    drivers = [
        "donor_dist_min",
        "donor_dist_max",
        "unknown_rows",
        "gate_segments",
        "delta_abs_median",
        "suffix_gr_missing_fraction",
        "suffix_u_true_range",
        "suffix_xy_displacement",
    ]
    rows: list[dict[str, Any]] = []
    for driver in drivers:
        ranked = well[driver].rank(method="first", pct=True)
        for label, mask in (
            ("bottom_quartile", ranked <= 0.25),
            ("top_quartile", ranked > 0.75),
        ):
            part = well[mask]
            rows.append(
                {
                    "driver": driver,
                    "quartile": label,
                    "wells": int(len(part)),
                    "driver_median": float(part[driver].median()),
                    "well_rmse_median": float(
                        part["final_rmse_recomputed"].median()
                    ),
                    "well_abs_bias_median": float(
                        part["final_abs_bias"].median()
                    ),
                    "persistent_episode_well_fraction": float(
                        np.mean(part["persistent_episode_count"] > 0)
                    ),
                    "gr_well_rmse_delta_median": float(
                        part["gr_rmse_delta"].median()
                    ),
                    "u_well_rmse_delta_median": float(
                        part["u_rmse_delta"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def kappa_stability(kappa: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for term, part in kappa.groupby("term", sort=False):
        values = part["value"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "term": str(term),
                "folds": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.ptp(values)),
                "relative_std_abs_mean": float(
                    np.std(values) / max(abs(np.mean(values)), 1e-12)
                ),
                "signs": ",".join(str(int(value)) for value in np.sign(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("range", ascending=False)


def source_port_contract() -> pd.DataFrame:
    """Record equations/constants manually checked against the saved source."""

    rows = [
        (
            "segment_count",
            "16",
            "16",
            "match",
        ),
        (
            "local_linear_k_bandwidth",
            "50 / 500",
            "50 / 500",
            "match",
        ),
        (
            "kappa_distance_bins",
            "0,750,1500,2500,4000,inf",
            "0,750,1500,2500,4000,inf",
            "match",
        ),
        (
            "kappa_regimes",
            "0,1000,1500,2000",
            "0,1000,1500,2000",
            "match",
        ),
        (
            "geometry_anchor",
            "last finite TVT_input",
            "last finite TVT_input",
            "match",
        ),
        (
            "geometry_path",
            "anchor + cumulative design @ global kappa",
            "anchor + cumulative design @ global kappa",
            "match",
        ),
        (
            "gr_window_stride_cap",
            "500 / 125 / +/-4 ft",
            "500 / 125 / +/-4 ft",
            "match",
        ),
        (
            "u_projection",
            "degree 4 / beta .75 / 4 iterations",
            "degree 4 / beta .75 / 4 iterations",
            "match",
        ),
        (
            "external_v7_v8",
            "optional weight-dependent",
            "disabled by experiment contract",
            "intentional_scope_difference",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "item",
            "public_source",
            "exp226_port",
            "status",
        ],
    )


def numeric_source_port_parity(
    public_source_path: Path,
    port_source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not public_source_path.exists():
        raise FileNotFoundError(public_source_path)
    if not port_source_path.exists():
        raise FileNotFoundError(port_source_path)

    selected_functions = {
        "segment_geometry",
        "fit_coeffs",
        "local_linear",
        "kernel_mean",
        "build_columns",
        "affine_cal",
        "emissions",
        "decode",
        "gr_correction",
        "project_u",
    }
    public_tree = ast.parse(public_source_path.read_text(encoding="utf-8"))
    safe_nodes: list[ast.stmt] = []
    for node in public_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in selected_functions:
                safe_nodes.append(node)
        elif isinstance(node, ast.Assign):
            safe_nodes.append(node)
    safe_module = ast.Module(body=safe_nodes, type_ignores=[])
    ast.fix_missing_locations(safe_module)
    public_namespace: dict[str, Any] = {"np": np, "pd": pd}
    exec(
        compile(safe_module, str(public_source_path), "exec"),
        public_namespace,
    )

    spec = importlib.util.spec_from_file_location(
        "exp226_port_for_parity", port_source_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import exp226 port: {port_source_path}")
    port = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    params = port.K16Params()

    rng = np.random.default_rng(20260727)
    s = 20
    n = 257
    total = s + 1 + n
    x = np.cumsum(rng.normal(4.0, 0.5, total))
    y = np.cumsum(rng.normal(2.0, 0.5, total))
    ndz = rng.normal(0.01, 0.02, n)
    cumulative_z = np.cumsum(ndz)
    r0 = cumulative_z + np.cumsum(rng.normal(0.0, 0.015, n))

    comparisons: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    public_segment = public_namespace["segment_geometry"](x, y, s, n)
    port_segment = port.segment_geometry(x, y, s, n, params)
    comparisons.append(
        (
            "segment_geometry",
            np.concatenate([np.asarray(value).ravel() for value in public_segment]),
            np.concatenate([np.asarray(value).ravel() for value in port_segment]),
            1e-12,
        )
    )

    for rho in (0.0, 10.0):
        comparisons.append(
            (
                f"fit_coeffs_rho_{rho:g}",
                public_namespace["fit_coeffs"](r0, cumulative_z, n, rho),
                port.fit_coeffs(r0, cumulative_z, n, params, rho),
                1e-10,
            )
        )

    field_rows = 800
    field = np.column_stack(
        [
            rng.uniform(-4000, 4000, field_rows),
            rng.uniform(-4000, 4000, field_rows),
            rng.normal(0, 0.2, field_rows),
            np.arange(field_rows) % 80,
        ]
    )
    mids = np.column_stack(
        [rng.uniform(-2000, 2000, 16), rng.uniform(-2000, 2000, 16)]
    )
    public_local = public_namespace["local_linear"](field, 7, mids, 1000.0)
    port_local = port.local_linear(field, 7, mids, params, 1000.0)
    comparisons.append(
        (
            "local_linear",
            np.concatenate([np.asarray(value).ravel() for value in public_local]),
            np.concatenate([np.asarray(value).ravel() for value in port_local]),
            1e-10,
        )
    )
    comparisons.append(
        (
            "kernel_mean",
            public_namespace["kernel_mean"](field, 7, mids, 1000.0),
            port.kernel_mean(field, 7, mids, 1000.0),
            1e-10,
        )
    )

    segid, _, proj, _ = public_segment
    raw = rng.normal(0, 0.05, K_SEGMENTS)
    smooth = rng.normal(0, 0.05, K_SEGMENTS)
    donor_distance = rng.uniform(100, 6000, K_SEGMENTS)
    sub = (
        rng.normal(0, 0.05, K_SEGMENTS),
        rng.random(n) > 0.5,
    )
    comparisons.append(
        (
            "build_columns",
            public_namespace["build_columns"](
                ndz,
                segid,
                proj,
                raw,
                smooth,
                donor_distance,
                sub,
            ),
            port.build_columns(
                ndz,
                segid,
                proj,
                raw,
                smooth,
                donor_distance,
                params,
                sub,
            ),
            1e-10,
        )
    )

    affine_x = rng.normal(80, 15, 500)
    affine_y = 1.2 * affine_x + 3.0 + rng.normal(0, 2, 500)
    comparisons.append(
        (
            "affine_cal",
            np.asarray(public_namespace["affine_cal"](affine_x, affine_y)),
            np.asarray(port.affine_cal(affine_x, affine_y)),
            1e-10,
        )
    )

    projection_pred = np.linspace(10_000, 10_100, 1000) + rng.normal(
        0, 1, 1000
    )
    projection_z = np.linspace(-100, -250, 1000)
    comparisons.append(
        (
            "project_u",
            public_namespace["project_u"](projection_pred, projection_z),
            port.project_u(projection_pred, projection_z, params),
            1e-9,
        )
    )

    typewell_tvt = np.linspace(11_500, 12_500, 2001)
    typewell_gr = (
        80
        + 15 * np.sin((typewell_tvt - 11_500) / 7)
        + 5 * np.cos((typewell_tvt - 11_500) / 23)
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    known_tvt = np.linspace(11_600, 11_900, 800)
    known_gr = np.interp(known_tvt, typewell_tvt, typewell_gr) + rng.normal(
        0, 3, len(known_tvt)
    )
    evaluation_rows = 1000
    geop = np.linspace(11_900, 12_200, evaluation_rows)
    lateral_gr = np.interp(
        geop + 2.0, typewell_tvt, typewell_gr
    ) + rng.normal(0, 3, evaluation_rows)
    relpath = np.linspace(0, 300, evaluation_rows)
    comparisons.append(
        (
            "gr_correction",
            public_namespace["gr_correction"](
                typewell,
                known_tvt,
                known_gr,
                lateral_gr,
                geop,
                relpath,
                evaluation_rows,
            ),
            port.gr_correction(
                typewell,
                known_tvt,
                known_gr,
                lateral_gr,
                geop,
                relpath,
                evaluation_rows,
                params,
            ),
            1e-9,
        )
    )

    rows: list[dict[str, Any]] = []
    for item, public_value, port_value, tolerance in comparisons:
        public_array = np.asarray(public_value, dtype=np.float64)
        port_array = np.asarray(port_value, dtype=np.float64)
        if public_array.shape != port_array.shape:
            max_abs_diff = float("inf")
        else:
            max_abs_diff = float(
                np.max(np.abs(public_array - port_array), initial=0.0)
            )
        rows.append(
            {
                "item": item,
                "public_shape": str(public_array.shape),
                "port_shape": str(port_array.shape),
                "max_abs_diff": max_abs_diff,
                "tolerance": tolerance,
                "status": "PASS" if max_abs_diff <= tolerance else "FAIL",
            }
        )
    result = pd.DataFrame(rows)
    if not result["status"].eq("PASS").all():
        raise ValueError(
            "numeric source-port parity failed: "
            + ",".join(result.loc[result["status"] != "PASS", "item"])
        )
    manifest = {
        "public_source_path": str(public_source_path),
        "public_source_sha256": sha256_file(public_source_path),
        "port_source_path": str(port_source_path),
        "port_source_sha256": sha256_file(port_source_path),
        "selected_functions": sorted(selected_functions),
        "synthetic_seed": 20260727,
        "all_pass": True,
    }
    return result, manifest


def empty_metric_accumulator() -> dict[str, float]:
    return {
        "rows": 0.0,
        "sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "within_5": 0.0,
        "within_10": 0.0,
        "within_25": 0.0,
    }


def update_metric_accumulator(
    accumulator: dict[str, float], values: np.ndarray
) -> None:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    accumulator["rows"] += len(values)
    accumulator["sum"] += float(np.sum(values))
    accumulator["abs_sum"] += float(np.sum(np.abs(values)))
    accumulator["sq_sum"] += float(np.sum(np.square(values)))
    accumulator["within_5"] += float(np.sum(np.abs(values) <= 5.0))
    accumulator["within_10"] += float(np.sum(np.abs(values) <= 10.0))
    accumulator["within_25"] += float(np.sum(np.abs(values) <= 25.0))


def finalize_metric_accumulator(
    name: str, accumulator: dict[str, float]
) -> dict[str, Any]:
    count = float(accumulator["rows"])
    if count <= 0:
        raise ValueError(f"empty metric accumulator: {name}")
    mean = accumulator["sum"] / count
    variance = max(accumulator["sq_sum"] / count - mean * mean, 0.0)
    return {
        "name": name,
        "rows": int(count),
        "rmse": math.sqrt(accumulator["sq_sum"] / count),
        "mae": accumulator["abs_sum"] / count,
        "bias_pred_minus_true": mean,
        "error_std": math.sqrt(variance),
        "within_5": accumulator["within_5"] / count,
        "within_10": accumulator["within_10"] / count,
        "within_25": accumulator["within_25"] / count,
    }


def array_sufficient_statistics(
    error: np.ndarray, x: np.ndarray
) -> dict[str, float]:
    error = np.asarray(error, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    count = len(error)
    error_sum = float(np.sum(error))
    error_sq_sum = float(np.sum(np.square(error)))
    x_sum = float(np.sum(x))
    x_sq_sum = float(np.sum(np.square(x)))
    x_error_sum = float(np.sum(x * error))
    centered_y_sse = max(error_sq_sum - error_sum * error_sum / count, 0.0)
    centered_x_ss = max(x_sq_sum - x_sum * x_sum / count, 0.0)
    centered_xy = x_error_sum - x_sum * error_sum / count
    affine_gain = centered_xy * centered_xy / centered_x_ss if centered_x_ss else 0.0
    start_error = float(error[0])
    return {
        "rows": float(count),
        "error_sum": error_sum,
        "error_sq_sum": error_sq_sum,
        "x_sum": x_sum,
        "x_sq_sum": x_sq_sum,
        "x_error_sum": x_error_sum,
        "start_error": start_error,
        "end_error": float(error[-1]),
        "mean_error": error_sum / count,
        "mean_corrected_sse": centered_y_sse,
        "start_corrected_sse": max(
            error_sq_sum
            - 2.0 * start_error * error_sum
            + count * start_error * start_error,
            0.0,
        ),
        "affine_corrected_sse": max(centered_y_sse - affine_gain, 0.0),
        "error_slope_per_row": centered_xy / centered_x_ss
        if centered_x_ss
        else 0.0,
    }


def iter_oof_wells(oof_path: Path, chunksize: int = 100_000):
    columns = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_true",
        "tvt_pred",
        "tvt_geop",
        "gr_delta",
        "error",
        "fold",
    ]
    dtype = {
        "well_id": "string",
        "row_idx": "int32",
        "suffix_offset": "int32",
        "tvt_true": "float64",
        "tvt_pred": "float64",
        "tvt_geop": "float64",
        "gr_delta": "float64",
        "error": "float64",
        "fold": "int8",
    }
    pending: pd.DataFrame | None = None
    seen: set[str] = set()
    reader = pd.read_csv(
        oof_path,
        usecols=columns,
        dtype=dtype,
        chunksize=chunksize,
    )
    for chunk in reader:
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = None
        last_well = str(chunk["well_id"].iloc[-1])
        pending = chunk[chunk["well_id"] == last_well].copy()
        complete = chunk[chunk["well_id"] != last_well]
        for well_id, part in complete.groupby(
            "well_id", sort=False, observed=True
        ):
            key = str(well_id)
            if key in seen:
                raise ValueError(f"OOF well rows are not contiguous: {key}")
            seen.add(key)
            yield key, part.reset_index(drop=True)
    if pending is not None:
        well_id = str(pending["well_id"].iloc[0])
        if well_id in seen:
            raise ValueError(f"OOF well rows are not contiguous: {well_id}")
        yield well_id, pending.reset_index(drop=True)


def episode_rows_for_well(
    well_id: str,
    fold: int,
    final_error: np.ndarray,
    geo_error: np.ndarray,
    pre_u_error: np.ndarray,
    gr_delta: np.ndarray,
    u_adjust: np.ndarray,
    segment: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    segment_starts = np.where(np.r_[True, segment[1:] != segment[:-1]])[0]
    for start, end in contiguous_true_runs(
        np.abs(final_error) >= PERSISTENT_THRESHOLD_FT
    ):
        if end - start < PERSISTENT_MIN_ROWS:
            continue
        window = slice(start, end)
        before = max(0, start - 64)
        denominator = max(start - before, 1)
        if abs(geo_error[start]) >= PERSISTENT_THRESHOLD_FT:
            crossing_component = "geometry_already_over_threshold"
        elif abs(pre_u_error[start]) >= PERSISTENT_THRESHOLD_FT:
            crossing_component = "gr_crossed_threshold"
        else:
            crossing_component = "u_projection_crossed_threshold"
        rows.append(
            {
                "well_id": well_id,
                "fold": fold,
                "start_suffix_offset": int(start),
                "end_suffix_offset_exclusive": int(end),
                "rows": int(end - start),
                "signed_error_median": float(np.median(final_error[window])),
                "abs_error_mean": float(np.mean(np.abs(final_error[window]))),
                "abs_error_max": float(np.max(np.abs(final_error[window]))),
                "episode_sse": float(np.sum(np.square(final_error[window]))),
                "geometry_episode_rmse": rmse(geo_error[window]),
                "pre_u_episode_rmse": rmse(pre_u_error[window]),
                "final_episode_rmse": rmse(final_error[window]),
                "crossing_component": crossing_component,
                "onset_error": float(final_error[start]),
                "onset_geometry_error": float(geo_error[start]),
                "onset_gr_delta": float(gr_delta[start]),
                "onset_u_adjust": float(u_adjust[start]),
                "onset_one_row_error_change": float(
                    final_error[start]
                    - (final_error[start - 1] if start else 0.0)
                ),
                "pre_onset_64row_error_rate": float(
                    (final_error[start] - final_error[before]) / denominator
                ),
                "distance_to_k16_boundary_rows": int(
                    np.min(np.abs(segment_starts - start))
                ),
                "gr_cap_fraction": float(
                    np.mean(np.abs(gr_delta[window]) >= 3.99)
                ),
                "u_worsens_episode_sse": bool(
                    np.sum(np.square(final_error[window]))
                    > np.sum(np.square(pre_u_error[window]))
                ),
                "gr_worsens_episode_sse": bool(
                    np.sum(np.square(pre_u_error[window]))
                    > np.sum(np.square(geo_error[window]))
                ),
            }
        )
    return rows


def episode_summary_frame(
    episodes: pd.DataFrame, total_sse: float, total_rows: int
) -> pd.DataFrame:
    if episodes.empty:
        raise ValueError("no persistent exp226 offset episodes found")
    return pd.DataFrame(
        [
            {"metric": "episodes", "value": int(len(episodes))},
            {
                "metric": "wells_with_episode",
                "value": int(episodes["well_id"].nunique()),
            },
            {
                "metric": "episode_sse_fraction_of_total",
                "value": float(episodes["episode_sse"].sum() / total_sse),
            },
            {
                "metric": "episode_rows",
                "value": int(episodes["rows"].sum()),
            },
            {
                "metric": "episode_rows_fraction_of_total",
                "value": float(episodes["rows"].sum() / total_rows),
            },
            {
                "metric": "geometry_already_over_threshold_fraction",
                "value": float(
                    np.mean(
                        episodes["crossing_component"]
                        == "geometry_already_over_threshold"
                    )
                ),
            },
            {
                "metric": "gr_crossed_threshold_fraction",
                "value": float(
                    np.mean(
                        episodes["crossing_component"]
                        == "gr_crossed_threshold"
                    )
                ),
            },
            {
                "metric": "u_projection_crossed_threshold_fraction",
                "value": float(
                    np.mean(
                        episodes["crossing_component"]
                        == "u_projection_crossed_threshold"
                    )
                ),
            },
            {
                "metric": "gr_worsens_episode_sse_fraction",
                "value": float(episodes["gr_worsens_episode_sse"].mean()),
            },
            {
                "metric": "u_worsens_episode_sse_fraction",
                "value": float(episodes["u_worsens_episode_sse"].mean()),
            },
            {
                "metric": "onset_distance_to_k16_boundary_median",
                "value": float(
                    episodes["distance_to_k16_boundary_rows"].median()
                ),
            },
            {
                "metric": "onset_distance_to_k16_boundary_p90",
                "value": float(
                    episodes["distance_to_k16_boundary_rows"].quantile(0.9)
                ),
            },
            {
                "metric": "onset_one_row_change_abs_median",
                "value": float(
                    episodes["onset_one_row_error_change"].abs().median()
                ),
            },
            {
                "metric": "pre_onset_64row_rate_abs_median",
                "value": float(
                    episodes["pre_onset_64row_error_rate"].abs().median()
                ),
            },
        ]
    )


def stream_oof_audit(
    oof_path: Path,
) -> dict[str, pd.DataFrame | int | float | set[int] | set[str]]:
    stage_names = [
        "geometry_tvt_geop",
        "geometry_plus_gr_pre_u",
        "final_post_u",
    ]
    stage_acc = {name: empty_metric_accumulator() for name in stage_names}
    increment_acc = {name: empty_metric_accumulator() for name in stage_names}
    fold_stage_acc = {
        fold: {name: empty_metric_accumulator() for name in stage_names}
        for fold in range(5)
    }
    suffix_acc: dict[tuple[str, str, str], dict[str, float]] = {}
    quotient_sse: dict[str, float] = {}
    quotient_groups: dict[str, int] = {}
    quotient_base: dict[str, float] = {}
    well_stage_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    gr_abs_parts: list[np.ndarray] = []
    u_abs_parts: list[np.ndarray] = []
    component_acc = {
        "gr_delta": {
            "sum": 0.0,
            "abs_sum": 0.0,
            "rows": 0.0,
            "cap_rows": 0.0,
            "improved_rows": 0.0,
            "worsened_rows": 0.0,
            "sse_before": 0.0,
            "sse_after": 0.0,
            "dot_needed": 0.0,
            "sum_adjust_sq": 0.0,
            "sum_needed_sq": 0.0,
            "sum_needed": 0.0,
        },
        "u_projection": {
            "sum": 0.0,
            "abs_sum": 0.0,
            "rows": 0.0,
            "cap_rows": 0.0,
            "improved_rows": 0.0,
            "worsened_rows": 0.0,
            "sse_before": 0.0,
            "sse_after": 0.0,
            "dot_needed": 0.0,
            "sum_adjust_sq": 0.0,
            "sum_needed_sq": 0.0,
            "sum_needed": 0.0,
        },
    }
    total_rows = 0
    folds: set[int] = set()
    wells: set[str] = set()

    absolute_edges = np.asarray([0, 50, 100, 250, 500, 1000, 2000, np.inf])
    absolute_labels = [
        "0000_0050",
        "0050_0100",
        "0100_0250",
        "0250_0500",
        "0500_1000",
        "1000_2000",
        "2000_plus",
    ]

    for well_id, part in iter_oof_wells(oof_path):
        fold_values = part["fold"].unique()
        if len(fold_values) != 1:
            raise ValueError(f"well {well_id} spans OOF folds")
        fold = int(fold_values[0])
        offset = part["suffix_offset"].to_numpy(dtype=np.int64)
        if not np.array_equal(offset, np.arange(len(part), dtype=np.int64)):
            raise ValueError(f"well {well_id} suffix offsets are not contiguous")
        if part.duplicated(["row_idx"]).any():
            raise ValueError(f"well {well_id} has duplicate row_idx")
        true = part["tvt_true"].to_numpy(dtype=np.float64)
        final_prediction = part["tvt_pred"].to_numpy(dtype=np.float64)
        geop = part["tvt_geop"].to_numpy(dtype=np.float64)
        gr_delta = part["gr_delta"].to_numpy(dtype=np.float64)
        saved_error = part["error"].to_numpy(dtype=np.float64)
        if not np.isfinite(
            np.column_stack([true, final_prediction, geop, gr_delta])
        ).all():
            raise ValueError(f"well {well_id} has non-finite OOF values")
        if float(np.max(np.abs(saved_error - (final_prediction - true)))) > 1e-9:
            raise ValueError(f"well {well_id} saved error parity failed")
        pre_u = geop + gr_delta
        u_adjust = final_prediction - pre_u
        errors = {
            "geometry_tvt_geop": geop - true,
            "geometry_plus_gr_pre_u": pre_u - true,
            "final_post_u": saved_error,
        }
        segment = exact_k16_segment_id(offset, len(part))

        for name, error in errors.items():
            update_metric_accumulator(stage_acc[name], error)
            update_metric_accumulator(fold_stage_acc[fold][name], error)
            if len(error) > 1:
                update_metric_accumulator(increment_acc[name], np.diff(error))
            well_row = metric_row(name, error)
            well_row.update(
                {"stage": name, "well_id": well_id, "fold": fold}
            )
            well_stage_rows.append(well_row)

        geo_error = errors["geometry_tvt_geop"]
        pre_u_error = errors["geometry_plus_gr_pre_u"]
        final_error = errors["final_post_u"]
        for component, adjustment, before, after, collection in (
            ("gr_delta", gr_delta, geo_error, pre_u_error, gr_abs_parts),
            (
                "u_projection",
                u_adjust,
                pre_u_error,
                final_error,
                u_abs_parts,
            ),
        ):
            acc = component_acc[component]
            needed = -before
            acc["sum"] += float(np.sum(adjustment))
            acc["abs_sum"] += float(np.sum(np.abs(adjustment)))
            acc["rows"] += len(adjustment)
            acc["cap_rows"] += float(np.sum(np.abs(adjustment) >= 3.99))
            acc["improved_rows"] += float(
                np.sum(np.abs(after) < np.abs(before))
            )
            acc["worsened_rows"] += float(
                np.sum(np.abs(after) > np.abs(before))
            )
            acc["sse_before"] += float(np.sum(np.square(before)))
            acc["sse_after"] += float(np.sum(np.square(after)))
            acc["dot_needed"] += float(np.sum(adjustment * needed))
            acc["sum_adjust_sq"] += float(np.sum(np.square(adjustment)))
            acc["sum_needed_sq"] += float(np.sum(np.square(needed)))
            acc["sum_needed"] += float(np.sum(needed))
            collection.append(np.abs(adjustment).astype(np.float32))

        absolute_index = np.searchsorted(
            absolute_edges[1:], offset, side="right"
        )
        normalized_decile = np.minimum(
            np.floor(offset / max(len(offset) - 1, 1) * 10).astype(np.int8),
            9,
        )
        for bucket_kind, assignments, labels in (
            ("absolute_bucket", absolute_index, absolute_labels),
            (
                "normalized_decile",
                normalized_decile,
                [str(value) for value in range(10)],
            ),
        ):
            for bucket_index in np.unique(assignments):
                mask = assignments == bucket_index
                bucket = labels[int(bucket_index)]
                for stage_name, error in errors.items():
                    key = (bucket_kind, bucket, stage_name)
                    if key not in suffix_acc:
                        suffix_acc[key] = empty_metric_accumulator()
                    update_metric_accumulator(suffix_acc[key], error[mask])

        final_well_stats = array_sufficient_statistics(final_error, offset)
        for quotient, field in (
            ("well_mean_offset", "mean_corrected_sse"),
            ("well_start_reanchor", "start_corrected_sse"),
            ("well_affine_offset_slope", "affine_corrected_sse"),
        ):
            quotient_sse[quotient] = quotient_sse.get(quotient, 0.0) + float(
                final_well_stats[field]
            )
            quotient_groups[quotient] = quotient_groups.get(quotient, 0) + 1

        for segment_id in range(K_SEGMENTS):
            mask = segment == segment_id
            stats = array_sufficient_statistics(final_error[mask], offset[mask])
            segment_rows.append(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "k16_segment": segment_id,
                    **stats,
                }
            )
            for quotient, field in (
                ("k16_segment_mean_offset", "mean_corrected_sse"),
                ("k16_segment_start_reanchor", "start_corrected_sse"),
                (
                    "k16_segment_affine_offset_slope",
                    "affine_corrected_sse",
                ),
            ):
                quotient_sse[quotient] = quotient_sse.get(
                    quotient, 0.0
                ) + float(stats[field])
                quotient_groups[quotient] = (
                    quotient_groups.get(quotient, 0) + 1
                )
            for stage_name, error in errors.items():
                stage_stats = array_sufficient_statistics(
                    error[mask], offset[mask]
                )
                quotient = f"{stage_name}__k16_segment_mean_offset"
                quotient_sse[quotient] = quotient_sse.get(
                    quotient, 0.0
                ) + float(stage_stats["mean_corrected_sse"])
                quotient_groups[quotient] = (
                    quotient_groups.get(quotient, 0) + 1
                )

        for block_size in (64, 128, 256, 512, 1024):
            block_id = offset // block_size
            for block in np.unique(block_id):
                mask = block_id == block
                stats = array_sufficient_statistics(
                    final_error[mask], offset[mask]
                )
                for suffix_name, field in (
                    ("mean_offset", "mean_corrected_sse"),
                    ("affine_offset_slope", "affine_corrected_sse"),
                ):
                    quotient = f"h{block_size}_block_{suffix_name}"
                    quotient_sse[quotient] = quotient_sse.get(
                        quotient, 0.0
                    ) + float(stats[field])
                    quotient_groups[quotient] = (
                        quotient_groups.get(quotient, 0) + 1
                    )

        episode_rows.extend(
            episode_rows_for_well(
                well_id,
                fold,
                final_error,
                geo_error,
                pre_u_error,
                gr_delta,
                u_adjust,
                segment,
            )
        )
        design = np.column_stack([np.ones(len(offset)), offset])
        intercept, slope = np.linalg.lstsq(
            design, final_error, rcond=None
        )[0]
        well_rows.append(
            {
                "well_id": well_id,
                "fold": fold,
                "oof_rows": int(len(part)),
                "final_rmse_recomputed": rmse(final_error),
                "final_mae_recomputed": mae(final_error),
                "final_bias_recomputed": float(np.mean(final_error)),
                "final_abs_bias": float(abs(np.mean(final_error))),
                "final_first_error": float(final_error[0]),
                "final_end_error": float(final_error[-1]),
                "final_error_slope_per_row": float(slope),
                "geometry_rmse": rmse(geo_error),
                "pre_u_rmse": rmse(pre_u_error),
                "gr_rmse_delta": rmse(pre_u_error) - rmse(geo_error),
                "u_rmse_delta": rmse(final_error) - rmse(pre_u_error),
                "gr_cap_fraction_rows": float(
                    np.mean(np.abs(gr_delta) >= 3.99)
                ),
                "u_adjust_abs_mean": float(np.mean(np.abs(u_adjust))),
                "well_affine_intercept": float(intercept),
            }
        )
        total_rows += len(part)
        folds.add(fold)
        wells.add(well_id)
        if len(wells) % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "stream_oof",
                        "wells": len(wells),
                        "rows": total_rows,
                    }
                ),
                flush=True,
            )

    if total_rows != EXPECTED_OOF_ROWS:
        raise ValueError(f"unexpected streamed OOF rows: {total_rows}")
    if len(wells) != 773 or folds != set(range(5)):
        raise ValueError(
            f"unexpected streamed OOF well/folds: {len(wells)} / {folds}"
        )

    stage_rows: list[dict[str, Any]] = []
    for name in stage_names:
        row = finalize_metric_accumulator(name, stage_acc[name])
        increment = finalize_metric_accumulator(
            f"{name}_increment", increment_acc[name]
        )
        row.update(
            {
                "increment_rows": increment["rows"],
                "increment_rmse": increment["rmse"],
                "increment_mae": increment["mae"],
                "increment_bias": increment["bias_pred_minus_true"],
            }
        )
        stage_rows.append(row)
    stage = pd.DataFrame(stage_rows)

    effect_rows: list[dict[str, Any]] = []
    for component, abs_parts in (
        ("gr_delta", gr_abs_parts),
        ("u_projection", u_abs_parts),
    ):
        acc = component_acc[component]
        adjustment_mean = acc["sum"] / acc["rows"]
        needed_mean = acc["sum_needed"] / acc["rows"]
        covariance = (
            acc["dot_needed"] / acc["rows"]
            - adjustment_mean * needed_mean
        )
        adjustment_variance = (
            acc["sum_adjust_sq"] / acc["rows"] - adjustment_mean**2
        )
        needed_variance = (
            acc["sum_needed_sq"] / acc["rows"] - needed_mean**2
        )
        denominator = math.sqrt(
            max(adjustment_variance, 0.0) * max(needed_variance, 0.0)
        )
        absolute = np.concatenate(abs_parts)
        effect_rows.append(
            {
                "component": component,
                "adjustment_mean": adjustment_mean,
                "adjustment_abs_mean": acc["abs_sum"] / acc["rows"],
                "adjustment_abs_p95": float(np.quantile(absolute, 0.95)),
                "adjustment_cap_3p99_fraction": acc["cap_rows"]
                / acc["rows"],
                "row_abs_error_improvement_fraction": acc["improved_rows"]
                / acc["rows"],
                "row_abs_error_worsening_fraction": acc["worsened_rows"]
                / acc["rows"],
                "sse_before": acc["sse_before"],
                "sse_after": acc["sse_after"],
                "sse_change_after_minus_before": acc["sse_after"]
                - acc["sse_before"],
                "adjustment_vs_needed_correction_pearson": covariance
                / denominator
                if denominator
                else float("nan"),
            }
        )
    effects = pd.DataFrame(effect_rows)

    well_core = pd.DataFrame(well_rows)
    well_effect_rows: list[dict[str, Any]] = []
    for component, delta_column in (
        ("gr_delta", "gr_rmse_delta"),
        ("u_projection", "u_rmse_delta"),
    ):
        fold_improved = 0
        for fold in range(5):
            before_name = (
                "geometry_tvt_geop"
                if component == "gr_delta"
                else "geometry_plus_gr_pre_u"
            )
            after_name = (
                "geometry_plus_gr_pre_u"
                if component == "gr_delta"
                else "final_post_u"
            )
            before = finalize_metric_accumulator(
                before_name, fold_stage_acc[fold][before_name]
            )["rmse"]
            after = finalize_metric_accumulator(
                after_name, fold_stage_acc[fold][after_name]
            )["rmse"]
            fold_improved += int(after < before)
        well_effect_rows.append(
            {
                "component": component,
                "wells_improved": int((well_core[delta_column] < 0).sum()),
                "wells_worsened": int((well_core[delta_column] > 0).sum()),
                "well_rmse_delta_median": float(
                    well_core[delta_column].median()
                ),
                "folds_improved": fold_improved,
            }
        )

    suffix_rows: list[dict[str, Any]] = []
    for (bucket_kind, bucket, stage_name), acc in suffix_acc.items():
        row = finalize_metric_accumulator(stage_name, acc)
        row.update(
            {
                "bucket_kind": bucket_kind,
                "bucket": bucket,
                "stage": stage_name,
            }
        )
        suffix_rows.append(row)
    suffix = pd.DataFrame(suffix_rows).sort_values(
        ["bucket_kind", "bucket", "stage"]
    )

    final_sse = stage_acc["final_post_u"]["sq_sum"]
    quotient_sse["global_mean_bias"] = final_sse - (
        stage_acc["final_post_u"]["sum"] ** 2 / total_rows
    )
    quotient_groups["global_mean_bias"] = 1
    standard_quotients = [
        "global_mean_bias",
        "well_mean_offset",
        "well_start_reanchor",
        "well_affine_offset_slope",
        "k16_segment_mean_offset",
        "k16_segment_start_reanchor",
        "k16_segment_affine_offset_slope",
    ]
    standard_quotients += [
        f"h{block_size}_block_{kind}"
        for block_size in (64, 128, 256, 512, 1024)
        for kind in ("mean_offset", "affine_offset_slope")
    ]
    for quotient in standard_quotients:
        quotient_base[quotient] = final_sse
    for stage_name in stage_names:
        quotient_base[f"{stage_name}__k16_segment_mean_offset"] = stage_acc[
            stage_name
        ]["sq_sum"]
    quotient_rows: list[dict[str, Any]] = []
    for quotient in standard_quotients + [
        f"{stage_name}__k16_segment_mean_offset"
        for stage_name in stage_names
    ]:
        corrected_sse = max(quotient_sse[quotient], 0.0)
        base_sse = quotient_base[quotient]
        quotient_rows.append(
            {
                "quotient": quotient,
                "groups": quotient_groups[quotient],
                "rows": total_rows,
                "rmse_after_oracle_removal": math.sqrt(
                    corrected_sse / total_rows
                ),
                "mse_fraction_remaining": corrected_sse / base_sse,
                "mse_fraction_explained": 1.0
                - corrected_sse / base_sse,
            }
        )
    quotient = pd.DataFrame(quotient_rows)
    segment_frame = pd.DataFrame(segment_rows)
    persistence = segment_persistence(segment_frame)
    episodes = pd.DataFrame(episode_rows)
    episode_summary = episode_summary_frame(
        episodes, final_sse, total_rows
    )
    return {
        "stage": stage,
        "well_stage": pd.DataFrame(well_stage_rows),
        "effects": effects,
        "well_effects": pd.DataFrame(well_effect_rows),
        "suffix": suffix,
        "quotient": quotient,
        "segment": segment_frame,
        "persistence": persistence,
        "episodes": episodes,
        "episode_summary": episode_summary,
        "well_core": well_core,
        "rows": total_rows,
        "folds": folds,
        "wells": wells,
    }


def complete_well_readout(
    well_core: pd.DataFrame,
    by_well: pd.DataFrame,
    raw_features: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    by_well_subset = by_well.drop(
        columns=[
            column
            for column in ("label", "rows", "rmse", "mae", "bias", "fold")
            if column in by_well.columns
        ]
    )
    well = well_core.merge(
        by_well_subset,
        on="well_id",
        how="left",
        validate="one_to_one",
    ).merge(
        raw_features,
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    episode_well = (
        episodes.groupby("well_id", sort=False, observed=True)
        .agg(
            persistent_episode_count=("well_id", "size"),
            persistent_episode_rows=("rows", "sum"),
            persistent_episode_sse=("episode_sse", "sum"),
        )
        .reset_index()
    )
    well = well.merge(
        episode_well, on="well_id", how="left", validate="one_to_one"
    )
    for column in (
        "persistent_episode_count",
        "persistent_episode_rows",
        "persistent_episode_sse",
    ):
        well[column] = well[column].fillna(0)
    if well.isna().any().any():
        missing_columns = well.columns[well.isna().any()].tolist()
        raise ValueError(f"well readout has missing columns: {missing_columns}")
    if not np.array_equal(
        well["oof_rows"].to_numpy(dtype=np.int64),
        well["raw_suffix_rows"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("raw/OOF suffix-row parity failed")
    return well


def build_summary(
    stage: pd.DataFrame,
    effects: pd.DataFrame,
    quotient: pd.DataFrame,
    persistence: pd.DataFrame,
    episode_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    kappa: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    stage_index = stage.set_index("name")
    quotient_index = quotient.set_index("quotient")
    episode_index = episode_summary.set_index("metric")["value"]
    persistence_index = persistence.set_index("metric")["value"]
    top_rmse_drivers = (
        correlations[correlations["outcome"] == "final_rmse_recomputed"]
        .head(8)[["driver", "spearman"]]
        .to_dict(orient="records")
    )
    return {
        "study": "exp226_offset_root_cause_audit_20260727",
        "status": "read_only_oof_audit_complete",
        "input_manifest": manifest,
        "definitions": {
            "error_sign": "prediction_minus_true",
            "persistent_threshold_ft": PERSISTENT_THRESHOLD_FT,
            "persistent_min_rows": PERSISTENT_MIN_ROWS,
            "k_segments": K_SEGMENTS,
        },
        "stage_rmse": {
            name: float(stage_index.loc[name, "rmse"])
            for name in stage_index.index
        },
        "stage_increment_rmse": {
            name: float(stage_index.loc[name, "increment_rmse"])
            for name in stage_index.index
        },
        "component_effects": effects.to_dict(orient="records"),
        "offset_structure": {
            "global_bias_ft": float(
                stage_index.loc[
                    "final_post_u", "bias_pred_minus_true"
                ]
            ),
            "global_bias_mse_fraction_explained": float(
                quotient_index.loc[
                    "global_mean_bias", "mse_fraction_explained"
                ]
            ),
            "well_mean_oracle_rmse": float(
                quotient_index.loc[
                    "well_mean_offset", "rmse_after_oracle_removal"
                ]
            ),
            "well_mean_mse_fraction_explained": float(
                quotient_index.loc[
                    "well_mean_offset", "mse_fraction_explained"
                ]
            ),
            "k16_mean_oracle_rmse": float(
                quotient_index.loc[
                    "k16_segment_mean_offset",
                    "rmse_after_oracle_removal",
                ]
            ),
            "k16_mean_mse_fraction_explained": float(
                quotient_index.loc[
                    "k16_segment_mean_offset", "mse_fraction_explained"
                ]
            ),
            "k16_affine_oracle_rmse": float(
                quotient_index.loc[
                    "k16_segment_affine_offset_slope",
                    "rmse_after_oracle_removal",
                ]
            ),
            "segment_mean_vs_start_error_pearson": float(
                persistence_index["segment_mean_vs_start_error_pearson"]
            ),
            "segment_mean_vs_previous_end_error_pearson": float(
                persistence_index[
                    "segment_mean_vs_previous_end_error_pearson"
                ]
            ),
        },
        "persistent_offsets": {
            key: jsonable(value)
            for key, value in episode_index.to_dict().items()
        },
        "top_well_level_rmse_drivers": top_rmse_drivers,
        "kappa_max_fold_range": float(kappa["range"].max()),
        "causal_conclusion": [
            (
                "The exact boundary TVT anchor is not globally shifted; error is "
                "small near the boundary and grows with suffix distance."
            ),
            (
                "Small local increment/rate errors from spatially transferred K16 "
                "geometry are cumulatively integrated without a new absolute TVT "
                "anchor."
            ),
            (
                "The accumulated error is then inherited as a nearly constant "
                "offset by later K16 segments; segment-mean removal explains most "
                "squared error."
            ),
            (
                "Sparse capped GR correction and robust U projection modify the "
                "error but are not the primary source when geometry already crossed "
                "the persistent-offset threshold."
            ),
            (
                "Large donor distance and geometry extrapolation are risk "
                "amplifiers; global bias, fold identity, file ordering, and a simple "
                "source-port constant mismatch are not sufficient explanations."
            ),
        ],
        "non_causal_or_not_supported": [
            "single global calibration bias",
            "wrong last-known-row anchor",
            "submission row ordering",
            "one bad CV fold",
            "K=16 alone",
            "always-on prefix offset predictability",
            "GR correction as the sole origin",
            "U projection as the sole origin",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    parser.add_argument(
        "--raw-train-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="Update only the numeric public-source/port parity evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parity_only:
        summary_path = args.output_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                "--parity-only requires an existing completed summary"
            )
        parity, parity_manifest = numeric_source_port_parity(
            Path(
                "/tmp/kaggle-notebooks/connortynan-k16-versioned/"
                "rogii-k16-spline-kernel-knn-adaptive-kappa.py"
            ),
            ROOT
            / "experiments"
            / EXP226_NAME
            / "connortynan_k16_reproduction.py",
        )
        parity_path = args.output_dir / "source_port_numeric_parity.csv"
        parity.to_csv(parity_path, index=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["input_manifest"][
            "source_port_numeric_parity"
        ] = parity_manifest
        summary["output_sha256"][
            parity_path.name
        ] = sha256_file(parity_path)
        write_json(summary_path, summary)
        print(
            json.dumps(
                {
                    "phase": "parity_only_complete",
                    "all_pass": bool(parity["status"].eq("PASS").all()),
                    "output": str(parity_path),
                },
                sort_keys=True,
            )
        )
        return

    oof_path = args.artifact_dir / f"{EXP226_NAME}_train_oof_predictions.csv.gz"
    by_well_path = args.artifact_dir / f"{EXP226_NAME}_by_well_metrics.csv"
    kappa_path = args.artifact_dir / f"{EXP226_NAME}_kappa_by_fold.csv"
    for path in (oof_path, by_well_path, kappa_path):
        if not path.exists():
            raise FileNotFoundError(path)
    decompressed_sha = sha256_decompressed_gzip(oof_path)
    if decompressed_sha != EXPECTED_OOF_DECOMPRESSED_SHA256:
        raise ValueError(
            "exp226 OOF decompressed SHA mismatch: "
            f"{decompressed_sha} != {EXPECTED_OOF_DECOMPRESSED_SHA256}"
        )
    by_well = pd.read_csv(by_well_path)
    kappa_input = pd.read_csv(kappa_path)
    streamed = stream_oof_audit(oof_path)
    manifest = {
        "oof_path": str(oof_path),
        "oof_file_sha256": sha256_file(oof_path),
        "oof_decompressed_sha256": decompressed_sha,
        "by_well_path": str(by_well_path),
        "by_well_sha256": sha256_file(by_well_path),
        "kappa_path": str(kappa_path),
        "kappa_sha256": sha256_file(kappa_path),
        "rows": int(streamed["rows"]),
        "wells": int(len(streamed["wells"])),
        "folds": sorted(int(value) for value in streamed["folds"]),
        "read_mode": "chunked_well_contiguous_stream",
    }
    print(json.dumps({"phase": "loaded", **manifest}, sort_keys=True))

    stage = streamed["stage"]
    well_stage = streamed["well_stage"]
    effects = streamed["effects"]
    well_effects = streamed["well_effects"]
    suffix = streamed["suffix"]
    quotient = streamed["quotient"]
    segment = streamed["segment"]
    persistence = streamed["persistence"]
    episodes = streamed["episodes"]
    episode_summary = streamed["episode_summary"]
    raw_features = raw_well_features(
        args.raw_train_dir, set(streamed["wells"])
    )
    well = complete_well_readout(
        streamed["well_core"], by_well, raw_features, episodes
    )
    correlations = driver_correlations(well)
    quantiles = driver_quantile_contrasts(well)
    kappa = kappa_stability(kappa_input)
    port = source_port_contract()
    parity, parity_manifest = numeric_source_port_parity(
        Path(
            "/tmp/kaggle-notebooks/connortynan-k16-versioned/"
            "rogii-k16-spline-kernel-knn-adaptive-kappa.py"
        ),
        ROOT
        / "experiments"
        / EXP226_NAME
        / "connortynan_k16_reproduction.py",
    )
    manifest["source_port_numeric_parity"] = parity_manifest

    outputs = {
        "stage_metrics.csv": stage,
        "well_stage_metrics.csv": well_stage,
        "component_effects.csv": effects,
        "well_component_effects.csv": well_effects,
        "suffix_bucket_metrics.csv": suffix,
        "oracle_quotient_metrics.csv": quotient,
        "k16_segment_statistics.csv": segment,
        "segment_persistence_metrics.csv": persistence,
        "persistent_offset_episodes.csv": episodes,
        "persistent_offset_summary.csv": episode_summary,
        "well_root_cause_readout.csv": well,
        "well_driver_correlations.csv": correlations,
        "driver_quantile_contrasts.csv": quantiles,
        "kappa_fold_stability.csv": kappa,
        "source_port_contract.csv": port,
        "source_port_numeric_parity.csv": parity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    summary = build_summary(
        stage,
        effects,
        quotient,
        persistence,
        episode_summary,
        correlations,
        kappa,
        manifest,
    )
    summary["output_sha256"] = {
        filename: sha256_file(args.output_dir / filename)
        for filename in outputs
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "phase": "complete",
                "output_dir": str(args.output_dir),
                "stage_rmse": summary["stage_rmse"],
                "offset_structure": summary["offset_structure"],
                "episodes": summary["persistent_offsets"]["episodes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
