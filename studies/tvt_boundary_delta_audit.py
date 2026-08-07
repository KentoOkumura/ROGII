#!/usr/bin/env python3
"""Audit TVT deltas around the TVT_input/evaluation boundary.

This is a diagnostic study, not a submission candidate.  It measures whether
TVT changes immediately before the known-prefix cutoff are informative for the
hidden/evaluation zone after the cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX_WINDOWS: tuple[int | str, ...] = (1, 5, 10, 30, 50, 100, 200, "all")
FUTURE_WINDOWS: tuple[int | str, ...] = (1, 5, 10, 30, 50, 100, 250, 500, 1000, 2000, "end")
STEP_BUCKETS: tuple[tuple[str, int, float], ...] = (
    ("000_050", 1, 50),
    ("050_100", 51, 100),
    ("100_250", 101, 250),
    ("250_500", 251, 500),
    ("500_1000", 501, 1000),
    ("1000_2000", 1001, 2000),
    ("2000_plus", 2001, np.inf),
)
VALUE_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class PredictionAccum:
    n: int = 0
    sum_y2: float = 0.0
    sum_pred2: float = 0.0
    sum_pred_y: float = 0.0
    sum_raw_err2: float = 0.0
    sum_raw_abs_err: float = 0.0
    within_5: int = 0
    within_10: int = 0

    def update(self, y: np.ndarray, pred: np.ndarray) -> None:
        mask = np.isfinite(y) & np.isfinite(pred)
        if not mask.any():
            return
        yy = y[mask].astype(np.float64, copy=False)
        pp = pred[mask].astype(np.float64, copy=False)
        err = pp - yy
        abs_err = np.abs(err)
        self.n += int(yy.size)
        self.sum_y2 += float(np.dot(yy, yy))
        self.sum_pred2 += float(np.dot(pp, pp))
        self.sum_pred_y += float(np.dot(pp, yy))
        self.sum_raw_err2 += float(np.dot(err, err))
        self.sum_raw_abs_err += float(abs_err.sum())
        self.within_5 += int((abs_err <= 5.0).sum())
        self.within_10 += int((abs_err <= 10.0).sum())

    def as_dict(self) -> dict[str, float | int]:
        if self.n == 0:
            return {
                "rows": 0,
                "raw_rmse": np.nan,
                "raw_mae": np.nan,
                "raw_within5": np.nan,
                "raw_within10": np.nan,
                "optimal_alpha": np.nan,
                "optimal_alpha_clipped_0_1": np.nan,
                "optimal_shrink_rmse": np.nan,
                "clipped_shrink_rmse": np.nan,
                "anchor_rmse": np.nan,
            }
        raw_rmse = math.sqrt(self.sum_raw_err2 / self.n)
        raw_mae = self.sum_raw_abs_err / self.n
        anchor_rmse = math.sqrt(self.sum_y2 / self.n)
        if self.sum_pred2 > 1e-12:
            alpha = self.sum_pred_y / self.sum_pred2
            clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
            shrink_sse = (
                self.sum_y2
                - 2.0 * alpha * self.sum_pred_y
                + alpha * alpha * self.sum_pred2
            )
            clipped_sse = (
                self.sum_y2
                - 2.0 * clipped_alpha * self.sum_pred_y
                + clipped_alpha * clipped_alpha * self.sum_pred2
            )
            shrink_rmse = math.sqrt(max(shrink_sse, 0.0) / self.n)
            clipped_rmse = math.sqrt(max(clipped_sse, 0.0) / self.n)
        else:
            alpha = np.nan
            clipped_alpha = np.nan
            shrink_rmse = np.nan
            clipped_rmse = np.nan
        return {
            "rows": self.n,
            "raw_rmse": raw_rmse,
            "raw_mae": raw_mae,
            "raw_within5": self.within_5 / self.n,
            "raw_within10": self.within_10 / self.n,
            "optimal_alpha": alpha,
            "optimal_alpha_clipped_0_1": clipped_alpha,
            "optimal_shrink_rmse": shrink_rmse,
            "clipped_shrink_rmse": clipped_rmse,
            "anchor_rmse": anchor_rmse,
            "raw_delta_rmse_vs_anchor": raw_rmse - anchor_rmse,
            "clipped_shrink_delta_rmse_vs_anchor": clipped_rmse - anchor_rmse
            if np.isfinite(clipped_rmse)
            else np.nan,
        }


def window_name(window: int | str) -> str:
    return str(window)


def read_horizontal_files(train_dir: Path) -> list[Path]:
    return sorted(train_dir.glob("*__horizontal_well.csv"))


def step_bucket_ids(steps: np.ndarray) -> np.ndarray:
    ids = np.full(steps.shape, -1, dtype=np.int16)
    for i, (_label, lo, hi) in enumerate(STEP_BUCKETS):
        ids[(steps >= lo) & (steps <= hi)] = i
    return ids


def safe_endpoint_slope(md: np.ndarray, tvt: np.ndarray, start_idx: int, end_idx: int) -> float:
    if start_idx < 0 or end_idx <= start_idx:
        return np.nan
    denom = float(md[end_idx] - md[start_idx])
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return np.nan
    delta = float(tvt[end_idx] - tvt[start_idx])
    if not np.isfinite(delta):
        return np.nan
    return delta / denom


def safe_median_step_slope(md: np.ndarray, tvt: np.ndarray, start_idx: int, end_idx: int) -> float:
    if start_idx < 0 or end_idx <= start_idx:
        return np.nan
    dmd = np.diff(md[start_idx : end_idx + 1])
    dtvt = np.diff(tvt[start_idx : end_idx + 1])
    mask = np.isfinite(dmd) & np.isfinite(dtvt) & (np.abs(dmd) > 1e-9)
    if mask.sum() == 0:
        return np.nan
    return float(np.median(dtvt[mask] / dmd[mask]))


def describe_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        row: dict[str, float | int | str] = {"metric": column, "count": int(values.size)}
        if values.size == 0:
            for key in ["mean", "std", "min", "max"]:
                row[key] = np.nan
            for q in VALUE_QUANTILES:
                row[f"p{int(q * 100):02d}"] = np.nan
        else:
            row.update(
                {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
            qs = np.quantile(values, VALUE_QUANTILES)
            for q, value in zip(VALUE_QUANTILES, qs, strict=True):
                row[f"p{int(q * 100):02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_group_values(
    frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, float | int | str] = dict(zip(group_columns, keys, strict=True))
        row["rows"] = int(len(group))
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            row[f"{column}_count"] = int(values.size)
            if values.size == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
                row[f"{column}_min"] = np.nan
                row[f"{column}_max"] = np.nan
                for q in VALUE_QUANTILES:
                    row[f"{column}_p{int(q * 100):02d}"] = np.nan
            else:
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                row[f"{column}_min"] = float(values.min())
                row[f"{column}_max"] = float(values.max())
                qs = np.quantile(values, VALUE_QUANTILES)
                for q, value in zip(VALUE_QUANTILES, qs, strict=True):
                    row[f"{column}_p{int(q * 100):02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def add_prefix_to_columns(frame: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{prefix}{column}" for column in frame.columns if column not in keep}
    )


def summarize_boundary_relative_steps(
    before_frame: pd.DataFrame, after_frame: pd.DataFrame
) -> pd.DataFrame:
    value_columns = ["step_tvt_delta", "abs_step_tvt_delta"]
    before_summary = summarize_group_values(before_frame, ["relative_step"], value_columns)
    after_summary = summarize_group_values(after_frame, ["relative_step"], value_columns)
    before_summary = add_prefix_to_columns(before_summary, "before_", {"relative_step"})
    after_summary = add_prefix_to_columns(after_summary, "after_", {"relative_step"})
    return before_summary.merge(after_summary, on="relative_step", how="outer").sort_values(
        "relative_step"
    )


def summarize_eval_rows(row_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in row_frame.groupby("step_bucket", sort=False):
        row: dict[str, float | int | str] = {"step_bucket": bucket, "rows": len(group)}
        for column in ["target_delta", "abs_target_delta", "future_slope", "md_since"]:
            values = group[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_p50"] = np.nan
                row[f"{column}_p90"] = np.nan
                row[f"{column}_p95"] = np.nan
                row[f"{column}_p99"] = np.nan
            else:
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_p50"] = float(np.quantile(values, 0.50))
                row[f"{column}_p90"] = float(np.quantile(values, 0.90))
                row[f"{column}_p95"] = float(np.quantile(values, 0.95))
                row[f"{column}_p99"] = float(np.quantile(values, 0.99))
        rows.append(row)
    return pd.DataFrame(rows)


def horizon_prediction_metrics(horizon_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, prefix_window), group in horizon_frame.groupby(["horizon", "prefix_window"]):
        y = group["target_delta"].to_numpy(dtype=np.float64)
        p = group["pred_delta"].to_numpy(dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        if y.size == 0:
            continue
        err = p - y
        anchor_rmse = math.sqrt(float(np.dot(y, y)) / y.size)
        raw_rmse = math.sqrt(float(np.dot(err, err)) / y.size)
        raw_mae = float(np.abs(err).mean())
        pred2 = float(np.dot(p, p))
        pred_y = float(np.dot(p, y))
        alpha = pred_y / pred2 if pred2 > 1e-12 else np.nan
        clipped_alpha = float(np.clip(alpha, 0.0, 1.0)) if np.isfinite(alpha) else np.nan
        if np.isfinite(clipped_alpha):
            clipped_err = clipped_alpha * p - y
            clipped_rmse = math.sqrt(float(np.dot(clipped_err, clipped_err)) / y.size)
        else:
            clipped_rmse = np.nan
        rows.append(
            {
                "horizon": horizon,
                "prefix_window": prefix_window,
                "wells": int(y.size),
                "anchor_rmse": anchor_rmse,
                "raw_rmse": raw_rmse,
                "raw_mae": raw_mae,
                "raw_delta_rmse_vs_anchor": raw_rmse - anchor_rmse,
                "optimal_alpha": alpha,
                "optimal_alpha_clipped_0_1": clipped_alpha,
                "clipped_shrink_rmse": clipped_rmse,
                "clipped_shrink_delta_rmse_vs_anchor": clipped_rmse - anchor_rmse
                if np.isfinite(clipped_rmse)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def slope_correlations(well_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prefix_cols = [c for c in well_frame.columns if c.startswith("prefix_endpoint_slope_")]
    future_cols = [c for c in well_frame.columns if c.startswith("future_endpoint_slope_")]
    for prefix_col in prefix_cols:
        for future_col in future_cols:
            pair = well_frame[[prefix_col, future_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3:
                continue
            x = pair[prefix_col]
            y = pair[future_col]
            nonzero = (x.abs() > 1e-9) & (y.abs() > 1e-9)
            rows.append(
                {
                    "prefix": prefix_col.replace("prefix_endpoint_slope_", ""),
                    "future": future_col.replace("future_endpoint_slope_", ""),
                    "wells": int(len(pair)),
                    "pearson": float(x.corr(y, method="pearson")),
                    "spearman": float(x.corr(y, method="spearman")),
                    "sign_match_rate": float((np.sign(x[nonzero]) == np.sign(y[nonzero])).mean())
                    if nonzero.any()
                    else np.nan,
                    "prefix_abs_median": float(x.abs().median()),
                    "future_abs_median": float(y.abs().median()),
                }
            )
    return pd.DataFrame(rows)


def update_prediction_accums(
    accums: dict[tuple[str, str], PredictionAccum],
    target_delta: np.ndarray,
    md_since: np.ndarray,
    bucket_ids: np.ndarray,
    prefix_slopes: dict[str, float],
) -> None:
    for prefix_name, slope in prefix_slopes.items():
        if not np.isfinite(slope):
            continue
        pred = slope * md_since
        accums.setdefault((prefix_name, "all"), PredictionAccum()).update(target_delta, pred)
        for bucket_id, (bucket_label, _lo, _hi) in enumerate(STEP_BUCKETS):
            mask = bucket_ids == bucket_id
            if mask.any():
                accums.setdefault((prefix_name, bucket_label), PredictionAccum()).update(
                    target_delta[mask], pred[mask]
                )


def analyze(train_dir: Path, output_dir: Path) -> dict[str, int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizontal_files = read_horizontal_files(train_dir)

    well_records: list[dict[str, float | int | str]] = []
    horizon_records: list[dict[str, float | int | str]] = []
    row_frames: list[pd.DataFrame] = []
    before_step_frames: list[pd.DataFrame] = []
    pred_accums: dict[tuple[str, str], PredictionAccum] = {}
    skipped = 0

    for path in horizontal_files:
        well_id = path.name.replace("__horizontal_well.csv", "")
        df = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        tvt_input = df["TVT_input"].to_numpy(dtype=np.float64)
        tvt = df["TVT"].to_numpy(dtype=np.float64)
        md = df["MD"].to_numpy(dtype=np.float64)
        known_idx = np.flatnonzero(np.isfinite(tvt_input))
        if known_idx.size < 3:
            skipped += 1
            continue
        cutoff_idx = int(known_idx[-1])
        eval_idx = np.flatnonzero((np.arange(len(df)) > cutoff_idx) & np.isfinite(tvt))
        if eval_idx.size == 0:
            skipped += 1
            continue

        last_md = float(md[cutoff_idx])
        last_tvt = float(tvt_input[cutoff_idx])
        first_eval_idx = int(eval_idx[0])
        record: dict[str, float | int | str] = {
            "well_id": well_id,
            "rows_total": int(len(df)),
            "known_rows": int(known_idx.size),
            "eval_rows": int(eval_idx.size),
            "cutoff_idx": cutoff_idx,
            "last_known_md": last_md,
            "last_known_tvt": last_tvt,
            "first_eval_md_since": float(md[first_eval_idx] - last_md),
            "boundary_delta_after_1": float(tvt[first_eval_idx] - last_tvt),
            "boundary_slope_after_1": safe_endpoint_slope(md, tvt, cutoff_idx, first_eval_idx),
        }
        if cutoff_idx > 0:
            record["boundary_delta_before_1"] = float(
                tvt_input[cutoff_idx] - tvt_input[cutoff_idx - 1]
            )
            record["boundary_slope_before_1"] = safe_endpoint_slope(
                md, tvt_input, cutoff_idx - 1, cutoff_idx
            )
            record["boundary_delta_after_minus_before_1"] = (
                record["boundary_delta_after_1"] - record["boundary_delta_before_1"]
            )

        prefix_slopes: dict[str, float] = {}
        for window in PREFIX_WINDOWS:
            if window == "all":
                start_idx = int(known_idx[0])
            else:
                start_idx = cutoff_idx - int(window)
            prefix_name = window_name(window)
            endpoint_slope = safe_endpoint_slope(md, tvt_input, start_idx, cutoff_idx)
            median_slope = safe_median_step_slope(md, tvt_input, start_idx, cutoff_idx)
            prefix_slopes[prefix_name] = endpoint_slope
            record[f"prefix_endpoint_slope_{prefix_name}"] = endpoint_slope
            record[f"prefix_median_step_slope_{prefix_name}"] = median_slope
            if np.isfinite(endpoint_slope):
                record[f"prefix_delta_{prefix_name}"] = endpoint_slope * (
                    md[cutoff_idx] - md[start_idx]
                )

        future_slope_by_horizon: dict[str, float] = {}
        for horizon in FUTURE_WINDOWS:
            if horizon == "end":
                end_eval_idx = int(eval_idx[-1])
            else:
                h = int(horizon)
                if eval_idx.size < h:
                    continue
                end_eval_idx = int(eval_idx[h - 1])
            horizon_name = window_name(horizon)
            target_delta = float(tvt[end_eval_idx] - last_tvt)
            md_since = float(md[end_eval_idx] - last_md)
            future_slope = safe_endpoint_slope(md, tvt, cutoff_idx, end_eval_idx)
            future_slope_by_horizon[horizon_name] = future_slope
            record[f"future_delta_{horizon_name}"] = target_delta
            record[f"future_abs_delta_{horizon_name}"] = abs(target_delta)
            record[f"future_md_since_{horizon_name}"] = md_since
            record[f"future_endpoint_slope_{horizon_name}"] = future_slope
            for prefix_name, prefix_slope in prefix_slopes.items():
                if not np.isfinite(prefix_slope):
                    continue
                horizon_records.append(
                    {
                        "well_id": well_id,
                        "horizon": horizon_name,
                        "prefix_window": prefix_name,
                        "target_delta": target_delta,
                        "md_since": md_since,
                        "prefix_slope": prefix_slope,
                        "future_slope": future_slope,
                        "pred_delta": prefix_slope * md_since,
                    }
                )

        steps = np.arange(1, eval_idx.size + 1, dtype=np.int32)
        md_since = md[eval_idx] - last_md
        target_delta = tvt[eval_idx] - last_tvt
        previous_tvt = np.empty(eval_idx.size, dtype=np.float64)
        previous_tvt[0] = last_tvt
        previous_tvt[1:] = tvt[eval_idx[:-1]]
        step_tvt_delta = tvt[eval_idx] - previous_tvt
        with np.errstate(divide="ignore", invalid="ignore"):
            future_slope = target_delta / md_since
        bucket_ids = step_bucket_ids(steps)
        bucket_labels = pd.Categorical.from_codes(
            bucket_ids,
            categories=[bucket_label for bucket_label, _lo, _hi in STEP_BUCKETS],
        )
        row_frames.append(
            pd.DataFrame(
                {
                    "eval_step": steps,
                    "step_bucket": bucket_labels,
                    "md_since": md_since.astype(np.float32),
                    "target_delta": target_delta.astype(np.float32),
                    "abs_target_delta": np.abs(target_delta).astype(np.float32),
                    "step_tvt_delta": step_tvt_delta.astype(np.float32),
                    "abs_step_tvt_delta": np.abs(step_tvt_delta).astype(np.float32),
                    "future_slope": future_slope.astype(np.float32),
                }
            )
        )

        before_transition_count = max(int(known_idx.size) - 1, 0)
        if before_transition_count > 0:
            relative_before_steps = np.arange(1, before_transition_count + 1, dtype=np.int32)
            current_idx = known_idx[-relative_before_steps]
            previous_idx = known_idx[-relative_before_steps - 1]
            before_step_tvt_delta = tvt_input[current_idx] - tvt_input[previous_idx]
            before_step_frames.append(
                pd.DataFrame(
                    {
                        "relative_step": relative_before_steps,
                        "step_tvt_delta": before_step_tvt_delta.astype(np.float32),
                        "abs_step_tvt_delta": np.abs(before_step_tvt_delta).astype(np.float32),
                    }
                )
            )
        update_prediction_accums(pred_accums, target_delta, md_since, bucket_ids, prefix_slopes)

        # Short after-window changes, symmetric with prefix windows where possible.
        for window in (5, 10, 30, 50, 100, 200):
            if eval_idx.size >= window:
                end_eval_idx = int(eval_idx[window - 1])
                record[f"boundary_delta_after_{window}"] = float(tvt[end_eval_idx] - last_tvt)
                record[f"boundary_slope_after_{window}"] = safe_endpoint_slope(
                    md, tvt, cutoff_idx, end_eval_idx
                )
            before_start = cutoff_idx - window
            if before_start >= 0:
                record[f"boundary_delta_before_{window}"] = float(
                    tvt_input[cutoff_idx] - tvt_input[before_start]
                )
                record[f"boundary_slope_before_{window}"] = safe_endpoint_slope(
                    md, tvt_input, before_start, cutoff_idx
                )
                if f"boundary_delta_after_{window}" in record:
                    record[f"boundary_delta_after_minus_before_{window}"] = (
                        record[f"boundary_delta_after_{window}"]
                        - record[f"boundary_delta_before_{window}"]
                    )

        well_records.append(record)

    well_frame = pd.DataFrame(well_records)
    row_frame = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    before_step_frame = (
        pd.concat(before_step_frames, ignore_index=True) if before_step_frames else pd.DataFrame()
    )
    horizon_frame = pd.DataFrame(horizon_records)

    well_frame.to_csv(output_dir / "well_delta_summary.csv", index=False)
    eval_step_summary = pd.DataFrame()
    if not row_frame.empty:
        summarize_eval_rows(row_frame).to_csv(
            output_dir / "eval_row_delta_bucket_summary.csv", index=False
        )
        eval_step_summary = summarize_group_values(
            row_frame,
            ["eval_step"],
            ["step_tvt_delta", "abs_step_tvt_delta", "target_delta", "abs_target_delta"],
        )
        eval_step_summary.to_csv(output_dir / "eval_step_tvt_delta_summary.csv", index=False)
        summarize_group_values(
            row_frame,
            ["step_bucket"],
            ["step_tvt_delta", "abs_step_tvt_delta", "target_delta", "abs_target_delta"],
        ).to_csv(output_dir / "eval_step_tvt_delta_bucket_summary.csv", index=False)
    if not before_step_frame.empty and not eval_step_summary.empty:
        before_summary = summarize_group_values(
            before_step_frame, ["relative_step"], ["step_tvt_delta", "abs_step_tvt_delta"]
        )
        before_summary = add_prefix_to_columns(before_summary, "before_", {"relative_step"})
        after_summary = eval_step_summary.rename(columns={"eval_step": "relative_step"})
        after_summary = after_summary[
            [
                "relative_step",
                "rows",
                *[
                    column
                    for column in after_summary.columns
                    if column.startswith("step_tvt_delta_")
                ],
                *[
                    column
                    for column in after_summary.columns
                    if column.startswith("abs_step_tvt_delta_")
                ],
            ]
        ]
        after_summary = add_prefix_to_columns(after_summary, "after_", {"relative_step"})
        before_summary.merge(after_summary, on="relative_step", how="outer").sort_values(
            "relative_step"
        ).to_csv(output_dir / "boundary_relative_step_tvt_delta_summary.csv", index=False)
    horizon_prediction_metrics(horizon_frame).to_csv(
        output_dir / "horizon_prefix_slope_prediction_metrics.csv", index=False
    )
    slope_correlations(well_frame).to_csv(
        output_dir / "prefix_future_slope_correlations.csv", index=False
    )

    boundary_columns = [
        c
        for c in well_frame.columns
        if c.startswith("boundary_delta_")
        or c.startswith("boundary_slope_")
        or c.startswith("future_delta_")
        or c.startswith("future_abs_delta_")
    ]
    describe_columns(well_frame, boundary_columns).to_csv(
        output_dir / "boundary_delta_describe.csv", index=False
    )

    pred_rows = []
    for (prefix_name, bucket_label), accum in sorted(pred_accums.items()):
        row = {
            "prefix_window": prefix_name,
            "step_bucket": bucket_label,
        }
        row.update(accum.as_dict())
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(
        output_dir / "row_prefix_slope_prediction_metrics.csv", index=False
    )

    summary = {
        "train_dir": str(train_dir),
        "output_dir": str(output_dir),
        "horizontal_files": len(horizontal_files),
        "usable_wells": len(well_frame),
        "skipped_wells": skipped,
        "eval_rows": int(len(row_frame)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(output_dir, summary)
    return summary


def markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "(empty)\n"
    shown = frame.head(max_rows).copy()
    columns = list(shown.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return "nan"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_readme(output_dir: Path, summary: dict[str, int | str]) -> None:
    boundary = pd.read_csv(output_dir / "boundary_delta_describe.csv")
    eval_bucket = pd.read_csv(output_dir / "eval_row_delta_bucket_summary.csv")
    step_delta = pd.read_csv(output_dir / "eval_step_tvt_delta_summary.csv")
    step_delta_bucket = pd.read_csv(output_dir / "eval_step_tvt_delta_bucket_summary.csv")
    boundary_relative = pd.read_csv(output_dir / "boundary_relative_step_tvt_delta_summary.csv")

    selected_boundary = boundary[
        boundary["metric"].isin(
            [
                "boundary_delta_before_1",
                "boundary_delta_after_1",
                "boundary_delta_after_minus_before_1",
                "boundary_delta_before_50",
                "boundary_delta_after_50",
                "boundary_delta_after_minus_before_50",
                "future_delta_end",
                "future_abs_delta_end",
            ]
        )
    ]

    exact_step_cols = [
        "eval_step",
        "rows",
        "step_tvt_delta_mean",
        "step_tvt_delta_std",
        "step_tvt_delta_p01",
        "step_tvt_delta_p05",
        "step_tvt_delta_p50",
        "step_tvt_delta_p95",
        "step_tvt_delta_p99",
        "abs_step_tvt_delta_p50",
        "abs_step_tvt_delta_p95",
        "abs_step_tvt_delta_p99",
    ]
    selected_steps = step_delta[
        step_delta["eval_step"].isin(list(range(1, 21)) + [50, 100, 250, 500, 1000, 2000])
    ]
    selected_steps = selected_steps[exact_step_cols]

    bucket_cols = [
        "step_bucket",
        "rows",
        "step_tvt_delta_mean",
        "step_tvt_delta_std",
        "step_tvt_delta_p01",
        "step_tvt_delta_p05",
        "step_tvt_delta_p50",
        "step_tvt_delta_p95",
        "step_tvt_delta_p99",
        "abs_step_tvt_delta_p50",
        "abs_step_tvt_delta_p95",
        "abs_step_tvt_delta_p99",
    ]
    selected_step_bucket = step_delta_bucket[bucket_cols]

    relative_cols = [
        "relative_step",
        "before_rows",
        "before_step_tvt_delta_mean",
        "before_step_tvt_delta_p50",
        "before_step_tvt_delta_p95",
        "before_step_tvt_delta_p99",
        "after_rows",
        "after_step_tvt_delta_mean",
        "after_step_tvt_delta_p50",
        "after_step_tvt_delta_p95",
        "after_step_tvt_delta_p99",
    ]
    selected_relative = boundary_relative[
        boundary_relative["relative_step"].isin(
            list(range(1, 21)) + [50, 100, 250, 500, 1000, 2000]
        )
    ][relative_cols]

    text = f"""# TVT Boundary Delta Audit

## Scope

- Input: `{summary["train_dir"]}`
- Usable wells: {summary["usable_wells"]}
- Evaluation rows: {summary["eval_rows"]}
- This is a train-side diagnostic for per-step TVT changes around the `TVT_input` cutoff.

Definitions:

- `step_tvt_delta`: per-row change, `TVT[current] - TVT[previous]`.
- `eval_step=1`: `TVT[first_eval] - last_known_TVT`.
- `target_delta`: cumulative change from `last_known_TVT`; included only as context.

## Evaluation Step TVT Delta

{markdown_table(selected_steps, max_rows=40)}

## Evaluation Step TVT Delta By Bucket

{markdown_table(selected_step_bucket)}

## Boundary Before/After Step Delta

This compares the same relative distance from the cutoff.  `relative_step=1`
means the last known step before the cutoff and the first evaluation step after
the cutoff.

{markdown_table(selected_relative, max_rows=40)}

## Boundary Delta Summary

These are cumulative deltas over fixed windows and are retained for context.

{markdown_table(selected_boundary)}

## Cumulative Evaluation Delta By Step Bucket

{markdown_table(eval_bucket)}

## Files

- `well_delta_summary.csv`
- `boundary_delta_describe.csv`
- `eval_row_delta_bucket_summary.csv`
- `eval_step_tvt_delta_summary.csv`
- `eval_step_tvt_delta_bucket_summary.csv`
- `boundary_relative_step_tvt_delta_summary.csv`
- `prefix_future_slope_correlations.csv`
- `horizon_prefix_slope_prediction_metrics.csv`
- `row_prefix_slope_prediction_metrics.csv`
- `summary.json`
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("studies/tvt_boundary_delta_audit_20260705")
    )
    args = parser.parse_args()
    summary = analyze(args.train_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
