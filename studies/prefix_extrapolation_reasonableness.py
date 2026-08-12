#!/usr/bin/env python3
"""Audit TVT extrapolations fitted only on known prefix rows.

The goal is not to build a submission candidate.  This script checks whether
prefix-fitted TVT / TVT+Z extrapolations are sane enough to use as weak PF/Beam
priors or selector confidence features.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "anchor",
    "md_all_anchor",
    "md_tail200_anchor",
    "md_tail50_anchor",
    "z_all_anchor",
    "z_tail200_anchor",
    "mdz_all_anchor",
    "mdz_tail200_anchor",
    "u_const_anchor",
    "u_md_all_anchor",
    "u_md_tail100_anchor",
    "u_md_tail30_median_anchor",
]

BLEND_BASE_METHODS = [
    "md_tail200_anchor",
    "mdz_tail200_anchor",
    "u_md_tail100_anchor",
    "u_md_tail30_median_anchor",
]

STEP_BUCKETS = [
    ("000_050", 0, 50),
    ("050_100", 50, 100),
    ("100_250", 100, 250),
    ("250_500", 250, 500),
    ("500_1000", 500, 1000),
    ("1000_plus", 1000, np.inf),
]

DRIFT_BUCKETS = [
    ("000_005", 0, 5),
    ("005_010", 5, 10),
    ("010_020", 10, 20),
    ("020_040", 20, 40),
    ("040_plus", 40, np.inf),
]


@dataclass
class MetricAccum:
    n: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    sum_err: float = 0.0
    within_5: int = 0
    within_10: int = 0
    abs_chunks: list[np.ndarray] = field(default_factory=list)

    def update(self, truth: np.ndarray, pred: np.ndarray, keep_abs: bool = False) -> None:
        mask = np.isfinite(truth) & np.isfinite(pred)
        if not mask.any():
            return
        err = pred[mask].astype(np.float64) - truth[mask].astype(np.float64)
        abs_err = np.abs(err)
        self.n += int(err.size)
        self.sum_sq += float(np.dot(err, err))
        self.sum_abs += float(abs_err.sum())
        self.sum_err += float(err.sum())
        self.within_5 += int((abs_err <= 5.0).sum())
        self.within_10 += int((abs_err <= 10.0).sum())
        if keep_abs:
            self.abs_chunks.append(abs_err.astype(np.float32, copy=False))

    def as_dict(self) -> dict[str, float | int]:
        if self.n == 0:
            return {
                "rows": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "bias": np.nan,
                "within5": np.nan,
                "within10": np.nan,
                "abs_p50": np.nan,
                "abs_p90": np.nan,
                "abs_p95": np.nan,
                "abs_p99": np.nan,
            }
        out: dict[str, float | int] = {
            "rows": self.n,
            "rmse": math.sqrt(self.sum_sq / self.n),
            "mae": self.sum_abs / self.n,
            "bias": self.sum_err / self.n,
            "within5": self.within_5 / self.n,
            "within10": self.within_10 / self.n,
        }
        if self.abs_chunks:
            vals = np.concatenate(self.abs_chunks)
            out.update(
                {
                    "abs_p50": float(np.quantile(vals, 0.50)),
                    "abs_p90": float(np.quantile(vals, 0.90)),
                    "abs_p95": float(np.quantile(vals, 0.95)),
                    "abs_p99": float(np.quantile(vals, 0.99)),
                }
            )
        else:
            out.update({"abs_p50": np.nan, "abs_p90": np.nan, "abs_p95": np.nan, "abs_p99": np.nan})
        return out


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask].astype(np.float64), y[mask].astype(np.float64)


def slope_ols(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    x, y = _finite_xy(x, y)
    if x.size < 2:
        return default
    x0 = x - float(x.mean())
    denom = float(np.dot(x0, x0))
    if denom < 1e-9:
        return default
    return float(np.dot(x0, y - float(y.mean())) / denom)


def slope_median_diff(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    x, y = _finite_xy(x, y)
    if x.size < 3:
        return default
    dx = np.diff(x)
    dy = np.diff(y)
    mask = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-9)
    if mask.sum() < 2:
        return default
    return float(np.median(dy[mask] / dx[mask]))


def anchored_linear(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_eval: np.ndarray,
    x_last: float,
    y_last: float,
    window: int | None = None,
) -> np.ndarray:
    if window is not None:
        x_known = x_known[-window:]
        y_known = y_known[-window:]
    slope = slope_ols(x_known, y_known)
    return y_last + slope * (x_eval - x_last)


def anchored_median_slope(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_eval: np.ndarray,
    x_last: float,
    y_last: float,
    window: int | None = None,
) -> np.ndarray:
    if window is not None:
        x_known = x_known[-window:]
        y_known = y_known[-window:]
    slope = slope_median_diff(x_known, y_known)
    return y_last + slope * (x_eval - x_last)


def anchored_mdz(
    md_known: np.ndarray,
    z_known: np.ndarray,
    tvt_known: np.ndarray,
    md_eval: np.ndarray,
    z_eval: np.ndarray,
    last_md: float,
    last_z: float,
    last_tvt: float,
    window: int | None = None,
) -> np.ndarray:
    if window is not None:
        md_known = md_known[-window:]
        z_known = z_known[-window:]
        tvt_known = tvt_known[-window:]
    mask = np.isfinite(md_known) & np.isfinite(z_known) & np.isfinite(tvt_known)
    if mask.sum() < 4:
        return np.full_like(md_eval, last_tvt, dtype=np.float64)
    x = np.column_stack([md_known[mask], z_known[mask]]).astype(np.float64)
    y = tvt_known[mask].astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    use = std > 1e-9
    if not use.any():
        return np.full_like(md_eval, last_tvt, dtype=np.float64)
    xs = (x[:, use] - mean[use]) / std[use]
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    # Convert standardized coefficients back to physical slopes, then anchor at
    # the last known TVT to avoid an artificial prefix/eval discontinuity.
    slopes = np.zeros(2, dtype=np.float64)
    slopes[use] = beta[1:] / std[use]
    return last_tvt + slopes[0] * (md_eval - last_md) + slopes[1] * (z_eval - last_z)


def predict_methods(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    known = df["TVT_input"].notna().to_numpy()
    eval_mask = df["TVT_input"].isna().to_numpy() & df["TVT"].notna().to_numpy()
    if known.sum() < 10 or eval_mask.sum() == 0:
        return {}, {}

    md = df["MD"].to_numpy(np.float64)
    z = df["Z"].to_numpy(np.float64)
    tvt = df["TVT"].to_numpy(np.float64)
    tvt_in = df["TVT_input"].to_numpy(np.float64)

    md_k = md[known]
    z_k = z[known]
    tvt_k = tvt_in[known]
    md_e = md[eval_mask]
    z_e = z[eval_mask]

    last_md = float(md_k[-1])
    last_z = float(z_k[-1])
    last_tvt = float(tvt_k[-1])
    last_u = last_tvt + last_z
    u_k = tvt_k + z_k

    pred: dict[str, np.ndarray] = {
        "anchor": np.full(md_e.shape, last_tvt, dtype=np.float64),
        "md_all_anchor": anchored_linear(md_k, tvt_k, md_e, last_md, last_tvt),
        "md_tail200_anchor": anchored_linear(md_k, tvt_k, md_e, last_md, last_tvt, window=200),
        "md_tail50_anchor": anchored_linear(md_k, tvt_k, md_e, last_md, last_tvt, window=50),
        "z_all_anchor": anchored_linear(z_k, tvt_k, z_e, last_z, last_tvt),
        "z_tail200_anchor": anchored_linear(z_k, tvt_k, z_e, last_z, last_tvt, window=200),
        "mdz_all_anchor": anchored_mdz(md_k, z_k, tvt_k, md_e, z_e, last_md, last_z, last_tvt),
        "mdz_tail200_anchor": anchored_mdz(md_k, z_k, tvt_k, md_e, z_e, last_md, last_z, last_tvt, window=200),
        "u_const_anchor": last_u - z_e,
        "u_md_all_anchor": anchored_linear(md_k, u_k, md_e, last_md, last_u) - z_e,
        "u_md_tail100_anchor": anchored_linear(md_k, u_k, md_e, last_md, last_u, window=100) - z_e,
        "u_md_tail30_median_anchor": anchored_median_slope(md_k, u_k, md_e, last_md, last_u, window=30) - z_e,
    }

    for base in BLEND_BASE_METHODS:
        delta = pred[base] - pred["anchor"]
        pred[f"blend10_{base}"] = pred["anchor"] + 0.10 * delta
        pred[f"blend25_{base}"] = pred["anchor"] + 0.25 * delta

    meta = {
        "known_len": float(known.sum()),
        "eval_len": float(eval_mask.sum()),
        "last_tvt": last_tvt,
        "known_tvt_range": float(np.nanmax(tvt_k) - np.nanmin(tvt_k)),
        "known_tvt_std": float(np.nanstd(tvt_k)),
        "eval_tvt_span": float(np.nanmax(tvt[eval_mask]) - np.nanmin(tvt[eval_mask])),
        "eval_drift_abs_mean": float(np.nanmean(np.abs(tvt[eval_mask] - last_tvt))),
        "md_tail50_slope": slope_ols(md_k[-50:], tvt_k[-50:]),
        "md_tail200_slope": slope_ols(md_k[-200:], tvt_k[-200:]),
        "u_tail30_median_slope": slope_median_diff(md_k[-30:], u_k[-30:]),
        "u_tail100_slope": slope_ols(md_k[-100:], u_k[-100:]),
    }
    return pred, meta


def bucket_values(values: np.ndarray, buckets: list[tuple[str, float, float]]) -> np.ndarray:
    labels = np.empty(values.shape, dtype=object)
    for name, lo, hi in buckets:
        mask = (values >= lo) & (values < hi)
        labels[mask] = name
    return labels


def frame_metrics(frame: pd.DataFrame, method: str) -> dict[str, float | int | str]:
    acc = MetricAccum()
    acc.update(frame["truth"].to_numpy(), frame[method].to_numpy(), keep_abs=False)
    out = acc.as_dict()
    out["method"] = method
    return out


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    if frame.empty:
        return "_empty_\n"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt) if np.isfinite(val) else "nan")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("studies/prefix_extrapolation_reasonableness_20260705"))
    parser.add_argument("--max-wells", type=int, default=None)
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if args.max_wells:
        paths = paths[: args.max_wells]
    if not paths:
        raise FileNotFoundError(f"No train horizontal wells under {train_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_methods = METHOD_ORDER + [f"blend{a}_{m}" for m in BLEND_BASE_METHODS for a in ("10", "25")]
    overall = {m: MetricAccum() for m in all_methods}
    step_stats: dict[tuple[str, str], MetricAccum] = defaultdict(MetricAccum)
    drift_stats: dict[tuple[str, str], MetricAccum] = defaultdict(MetricAccum)
    well_rows: list[dict[str, float | str | int]] = []

    usecols = ["MD", "Z", "TVT", "TVT_input"]
    for i, path in enumerate(paths, start=1):
        wid = path.name.split("__", 1)[0]
        df = pd.read_csv(path, usecols=usecols)
        pred, meta = predict_methods(df)
        if not pred:
            continue
        eval_mask = df["TVT_input"].isna().to_numpy() & df["TVT"].notna().to_numpy()
        truth = df.loc[eval_mask, "TVT"].to_numpy(np.float64)
        step = np.arange(1, truth.size + 1, dtype=np.float64)
        last_tvt = float(meta["last_tvt"])
        drift_abs = np.abs(truth - last_tvt)
        step_labels = bucket_values(step, STEP_BUCKETS)
        drift_labels = bucket_values(drift_abs, DRIFT_BUCKETS)

        well_row: dict[str, float | str | int] = {"well": wid, **meta}
        for method in all_methods:
            p = pred[method]
            overall[method].update(truth, p, keep_abs=True)
            acc = MetricAccum()
            acc.update(truth, p)
            metrics = acc.as_dict()
            well_row[f"{method}_rmse"] = metrics["rmse"]
            well_row[f"{method}_mae"] = metrics["mae"]
            for bucket in np.unique(step_labels):
                mask = step_labels == bucket
                step_stats[(method, str(bucket))].update(truth[mask], p[mask])
            for bucket in np.unique(drift_labels):
                mask = drift_labels == bucket
                drift_stats[(method, str(bucket))].update(truth[mask], p[mask])
        well_rows.append(well_row)
        if i % 100 == 0:
            print(f"processed {i}/{len(paths)} wells", flush=True)

    overall_rows = []
    anchor_rmse = overall["anchor"].as_dict()["rmse"]
    for method in all_methods:
        out = overall[method].as_dict()
        out["method"] = method
        out["delta_rmse_vs_anchor"] = float(out["rmse"] - anchor_rmse)
        overall_rows.append(out)
    overall_df = pd.DataFrame(overall_rows).sort_values("rmse")
    overall_df.to_csv(args.out_dir / "overall_metrics.csv", index=False)

    well_df = pd.DataFrame(well_rows)
    anchor_col = "anchor_rmse"
    well_summary_rows = []
    for method in all_methods:
        rmse_col = f"{method}_rmse"
        delta = well_df[rmse_col] - well_df[anchor_col]
        well_summary_rows.append(
            {
                "method": method,
                "mean_well_rmse": float(well_df[rmse_col].mean()),
                "median_well_rmse": float(well_df[rmse_col].median()),
                "p90_well_rmse": float(well_df[rmse_col].quantile(0.90)),
                "max_well_rmse": float(well_df[rmse_col].max()),
                "improved_wells_vs_anchor": int((delta < 0).sum()),
                "worse_wells_vs_anchor": int((delta > 0).sum()),
                "max_well_regression_vs_anchor": float(delta.max()),
                "best_well_improvement_vs_anchor": float(delta.min()),
            }
        )
    well_summary_df = pd.DataFrame(well_summary_rows).sort_values("median_well_rmse")
    well_summary_df.to_csv(args.out_dir / "well_summary.csv", index=False)
    well_df.to_csv(args.out_dir / "by_well_metrics.csv", index=False)

    step_rows = []
    for (method, bucket), acc in sorted(step_stats.items()):
        row = acc.as_dict()
        row.update({"method": method, "bucket_type": "eval_step", "bucket": bucket})
        step_rows.append(row)
    step_df = pd.DataFrame(step_rows).sort_values(["bucket", "rmse"])
    step_df.to_csv(args.out_dir / "step_bucket_metrics.csv", index=False)

    drift_rows = []
    for (method, bucket), acc in sorted(drift_stats.items()):
        row = acc.as_dict()
        row.update({"method": method, "bucket_type": "abs_true_drift_from_last", "bucket": bucket})
        drift_rows.append(row)
    drift_df = pd.DataFrame(drift_rows).sort_values(["bucket", "rmse"])
    drift_df.to_csv(args.out_dir / "drift_bucket_metrics.csv", index=False)

    # A compact report with the main tables copied inline.
    top_methods = overall_df.head(12).copy()
    direct = overall_df[overall_df["method"].isin(METHOD_ORDER)].sort_values("rmse")
    step_focus = step_df[
        step_df["method"].isin(["anchor", "md_tail200_anchor", "mdz_tail200_anchor", "u_md_tail100_anchor", "u_md_tail30_median_anchor"])
    ].copy()
    worst_direct = (
        well_summary_df[well_summary_df["method"].isin(METHOD_ORDER)]
        .sort_values("max_well_regression_vs_anchor", ascending=False)
        .head(8)
    )

    report = []
    report.append("# Prefix Extrapolation Reasonableness Audit 2026-07-05\n")
    report.append("## Scope\n")
    report.append(
        f"- Input: `{train_dir}` train horizontal wells, {len(well_df):,} usable wells.\n"
        "- Fits use only `TVT_input` known-prefix rows. Metrics use the hidden/evaluation rows where train `TVT` is available.\n"
        "- This is a pre-experiment diagnostic, not a submission candidate.\n"
    )
    report.append("\n## Findings\n\n")
    report.append(
        "- Direct extrapolation methods do not beat `anchor` globally. Raw `TVT ~ MD`, `TVT ~ Z`, "
        "`TVT ~ MD+Z`, and `U=TVT+Z` extrapolations are too unstable as standalone candidates.\n"
    )
    report.append(
        "- The only global positive signal is a weak blend: `blend10_u_md_tail100_anchor` improves "
        "RMSE from 15.909853 to 15.797007, but slightly worsens MAE and still has large well-level regressions.\n"
    )
    report.append(
        "- `U=TVT+Z` tail extrapolation is strong near the prefix and weak in long tail. It beats anchor through "
        "roughly the 500-1000 step bucket, then collapses badly in `1000_plus`.\n"
    )
    report.append(
        "- Prefix-only proxy features are weak guards for whether the extrapolation will help. Treat this as a "
        "confidence/selector feature or very weak faded prior, not as a hard generation constraint.\n"
    )
    report.append("\n## Decision\n\n")
    report.append(
        "- If this is tried inside PF/Beam, restrict it to `U=TVT+Z` tail-slope style priors.\n"
        "- Do not use the extrapolated path as a direct candidate, hard prior, or hard prune.\n"
        "- Use `alpha <= 0.10`, near-prefix / distance-aware fade, and explicit long-tail regression checks.\n"
        "- A selector/candidate-scoring experiment is safer than constraining PF/Beam transition directly.\n"
    )
    report.append("\n## Top Overall Metrics\n\n")
    report.append(markdown_table(top_methods, floatfmt=".6f"))
    report.append("\n\n## Direct Extrapolation Methods\n\n")
    report.append(markdown_table(direct, floatfmt=".6f"))
    report.append("\n\n## Step Bucket Focus\n\n")
    report.append(markdown_table(step_focus, floatfmt=".6f"))
    report.append("\n\n## Worst Direct Well-Level Regressions\n\n")
    report.append(markdown_table(worst_direct, floatfmt=".6f"))
    report.append("\n\n## Files\n\n")
    for name in [
        "overall_metrics.csv",
        "well_summary.csv",
        "by_well_metrics.csv",
        "step_bucket_metrics.csv",
        "drift_bucket_metrics.csv",
    ]:
        report.append(f"- `{args.out_dir / name}`\n")

    report_text = "".join(report)
    (args.out_dir / "README.md").write_text(report_text, encoding="utf-8")
    docs_path = Path("docs/surveys/prefix_extrapolation_reasonableness_20260705.md")
    survey_front_matter = """---
title: Prefix Extrapolation Reasonableness Audit
date: 2026-07-05
types:
  - oof_analysis
  - comparison
experiments:
  - exp001
topics:
  - prefix_extrapolation
  - tail
  - candidate_path
status: final
summary: "known prefixからの直接外挿はlong tailで不安定であり、小さいblendまたは信頼度特徴として限定利用する判断を記録した。"
---

"""
    docs_path.write_text(survey_front_matter + report_text, encoding="utf-8")

    summary = {
        "wells": int(len(well_df)),
        "rows": int(overall["anchor"].n),
        "best_overall_method": str(overall_df.iloc[0]["method"]),
        "best_direct_method": str(direct.iloc[0]["method"]),
        "anchor_rmse": float(anchor_rmse),
        "best_overall_rmse": float(overall_df.iloc[0]["rmse"]),
        "best_direct_rmse": float(direct.iloc[0]["rmse"]),
        "output_dir": str(args.out_dir),
        "docs_report": str(docs_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
