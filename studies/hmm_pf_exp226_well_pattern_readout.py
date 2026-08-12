#!/usr/bin/env python3
"""Build a repeatable HMM/PF/exp226 by-well contrast readout.

This is a diagnostic study. It joins existing OOF/by-well artifacts and raw
train wells, then writes reusable CSV/JSON summaries under studies/.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HMM_CANDIDATE = "hmm_selfgr_boost_only_a070_c100"
HMM_ALT_CANDIDATE = "hmm_selfgr_boost_only_a150_c100"
LIKPF_CANDIDATE = "exp072_likpf_mean"
PF_ANCC_CANDIDATE = "exp072_pf_ancc"
PF_Z_CANDIDATE = "exp072_pf_z"
BEAM_CANDIDATE = "exp072_beam_mean"

DEFAULT_OUTPUT = Path("studies/hmm_pf_exp226_well_pattern_readout_20260712")
FORM_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

FEATURE_SUMMARY_COLUMNS = [
    "x_mean",
    "y_mean",
    "unknown_rows",
    "last_known_row",
    "z_mean_eval",
    "z_range_eval",
    "tvt_range_eval",
    "formation_abs_slope_mean",
    "formation_abs_slope_max",
    "formation_slope_std",
    "gr_missing_prefix",
    "gr_missing_eval",
    "gr_longest_nan_eval",
    "gr_std_eval",
    "gr_slope_change",
    "gr_half_mean_diff",
    "gr_quarter_max_jump",
    "hmm_std_mean",
    "hmm_std_p90",
    "self_gr_quality_mean",
    "self_gr_valid_rate",
    "pf_std_mean",
    "beam_spread_mean",
    "likpf_abs_delta_mean",
    "donor_dist_min",
    "donor_dist_max",
    "delta_abs_median",
    "delta_abs_max",
    "end_minus_anchor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bad-rmse", type=float, default=30.0)
    parser.add_argument("--good-rmse", type=float, default=10.0)
    parser.add_argument("--neighbor-k", type=int, default=8)
    parser.add_argument(
        "--skip-hmm-row-bias",
        action="store_true",
        help="Skip reading the large HMM row feature file for HMM bias.",
    )
    return parser.parse_args()


def file_info(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        out.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return out


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact is missing: {path}\n"
            "If it is a Kaggle output, download it first with `kaggle kernels output`."
        )
    return path


def safe_nanmean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def safe_nanstd(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan


def longest_nan_run(values: np.ndarray) -> int:
    mask = ~np.isfinite(values)
    best = 0
    cur = 0
    for value in mask:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def lin_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 20:
        return np.nan
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def pct_of_all(series: pd.Series, value: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(value):
        return np.nan
    return float((values <= value).mean())


def read_exp226_by_well(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path))
    frame = frame.rename(
        columns={
            "well_id": "well",
            "rows": "exp226_rows",
            "rmse": "exp226_rmse",
            "mae": "exp226_mae",
            "bias": "exp226_bias",
            "within10": "exp226_within10",
            "within25": "exp226_within25",
        }
    )
    keep = [
        "well",
        "exp226_rows",
        "last_known_row",
        "unknown_rows",
        "exp226_rmse",
        "exp226_mae",
        "exp226_bias",
        "exp226_within10",
        "exp226_within25",
        "fold",
        "donor_dist_min",
        "donor_dist_max",
        "delta_abs_median",
        "delta_abs_max",
        "end_minus_anchor",
        "gate_segments",
    ]
    return frame[keep].copy()


def read_hmm_by_well(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(require_file(path))
    wide = raw.pivot(index="well", columns="candidate", values="rmse").reset_index()
    wide = wide.rename(
        columns={
            HMM_CANDIDATE: "hmm_rmse",
            HMM_ALT_CANDIDATE: "hmm_a150_rmse",
            LIKPF_CANDIDATE: "likpf_rmse",
            PF_ANCC_CANDIDATE: "pf_ancc_rmse",
            PF_Z_CANDIDATE: "pf_z_rmse",
            BEAM_CANDIDATE: "beam_mean_rmse",
        }
    )
    return wide.copy()


def read_hmm_generation(path: Path) -> pd.DataFrame:
    gen = pd.read_csv(require_file(path))
    rows: list[dict[str, Any]] = []
    for _, row in gen.iterrows():
        metrics = ast.literal_eval(row["variant_metrics"]) if isinstance(row["variant_metrics"], str) else []
        out: dict[str, Any] = {
            "well": row["well"],
            "hmm_best_variant": row["best_variant"],
            "hmm_best_rmse": row["best_variant_rmse"],
            "hmm_status": row.get("status"),
        }
        for item in metrics:
            if item.get("name") != HMM_CANDIDATE:
                continue
            out.update(
                {
                    "hmm_std_mean": item.get("std_mean"),
                    "hmm_std_p90": item.get("std_p90"),
                    "self_gr_quality_mean": item.get("self_gr_quality_mean"),
                    "self_gr_valid_rate": item.get("self_gr_valid_rate"),
                    "self_gr_prefix_anchor_count": item.get("self_gr_prefix_anchor_count"),
                    "hmm_loglik": item.get("loglik"),
                    "hmm_grid_size": item.get("grid_size"),
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def read_hmm_row_bias(path: Path) -> pd.DataFrame:
    require_file(path)
    usecols = ["well", "target", "last_known_tvt", f"{HMM_CANDIDATE}_mean_tvt"]
    acc: dict[str, dict[str, float]] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        pred = pd.to_numeric(chunk[f"{HMM_CANDIDATE}_mean_tvt"], errors="coerce").to_numpy(dtype=float)
        truth = (
            pd.to_numeric(chunk["last_known_tvt"], errors="coerce").to_numpy(dtype=float)
            + pd.to_numeric(chunk["target"], errors="coerce").to_numpy(dtype=float)
        )
        err = pred - truth
        mask = np.isfinite(err)
        work = pd.DataFrame({"well": chunk["well"].to_numpy(), "err": err, "sq": err * err})
        work = work.loc[mask]
        grouped = work.groupby("well", sort=False).agg(n=("err", "size"), sum_err=("err", "sum"), sum_sq=("sq", "sum"))
        for well, row in grouped.iterrows():
            item = acc.setdefault(str(well), {"n": 0.0, "sum_err": 0.0, "sum_sq": 0.0})
            item["n"] += float(row["n"])
            item["sum_err"] += float(row["sum_err"])
            item["sum_sq"] += float(row["sum_sq"])
    rows = []
    for well, item in acc.items():
        n = item["n"]
        rows.append(
            {
                "well": well,
                "hmm_bias": item["sum_err"] / n if n else np.nan,
                "hmm_row_rmse": math.sqrt(item["sum_sq"] / n) if n else np.nan,
                "hmm_row_count": int(n),
            }
        )
    return pd.DataFrame(rows)


def read_pf_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path)).rename(columns={"well_id": "well"})
    keep = [
        "well",
        "likpf_mean_bias",
        "pf_ancc_bias",
        "pf_z_bias",
        "beam_mean_bias",
        "pf_std_mean",
        "beam_spread_mean",
        "likpf_abs_delta_mean",
        "prefix_length",
        "eval_length",
        "ml_rmse",
    ]
    return frame[[col for col in keep if col in frame.columns]].copy()


def read_positions(path: Path) -> pd.DataFrame:
    pos = pd.read_csv(require_file(path)).rename(columns={"well_id": "well"})
    pos = pos.loc[pos["split"] == "train"].copy()
    keep = [
        "well",
        "x_mean",
        "y_mean",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
        "exact_typewell_group",
        "exact_typewell_group_size",
        "horizontal_well_path",
    ]
    return pos[keep].copy()


def raw_eval_features(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        well = row["well"]
        path = root / "data" / "raw" / "train" / f"{well}__horizontal_well.csv"
        if not path.exists() and isinstance(row.get("horizontal_well_path"), str):
            path = root / row["horizontal_well_path"]
        require_file(path)
        raw = pd.read_csv(path)
        last_known = int(row["last_known_row"])
        unknown_rows = int(row["unknown_rows"])
        prefix = raw.iloc[: last_known + 1].copy()
        eval_frame = raw.iloc[last_known + 1 :].copy()
        if len(eval_frame) != unknown_rows:
            eval_frame = raw.iloc[-unknown_rows:].copy()
            prefix = raw.iloc[: len(raw) - len(eval_frame)].copy()

        gr_eval = pd.to_numeric(eval_frame["GR"], errors="coerce").to_numpy(dtype=float)
        gr_prefix = pd.to_numeric(prefix["GR"], errors="coerce").to_numpy(dtype=float)
        halves = np.array_split(gr_eval, 2)
        quarters = np.array_split(gr_eval, 4)
        half_means = [safe_nanmean(part) for part in halves]
        quarter_means = [safe_nanmean(part) for part in quarters]
        quarter_diffs = [
            abs(quarter_means[i + 1] - quarter_means[i])
            for i in range(3)
            if np.isfinite(quarter_means[i]) and np.isfinite(quarter_means[i + 1])
        ]
        slope_first = lin_slope(halves[0])
        slope_second = lin_slope(halves[1])

        md = pd.to_numeric(eval_frame["MD"], errors="coerce").to_numpy(dtype=float)
        marker_slopes: list[float] = []
        marker_abs_slopes: list[float] = []
        for col in FORM_COLS:
            if col not in eval_frame:
                continue
            values = pd.to_numeric(eval_frame[col], errors="coerce").to_numpy(dtype=float)
            dmd = np.diff(md)
            dy = np.diff(values)
            mask = np.isfinite(dmd) & np.isfinite(dy) & (np.abs(dmd) > 1e-9)
            if mask.any():
                slopes = dy[mask] / dmd[mask]
                marker_slopes.extend(slopes.tolist())
                marker_abs_slopes.extend(np.abs(slopes).tolist())

        z = pd.to_numeric(eval_frame["Z"], errors="coerce").to_numpy(dtype=float)
        tvt = pd.to_numeric(eval_frame["TVT"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "well": well,
                "z_mean_eval": safe_nanmean(z),
                "z_min_eval": float(np.nanmin(z)),
                "z_max_eval": float(np.nanmax(z)),
                "z_range_eval": float(np.nanmax(z) - np.nanmin(z)),
                "tvt_mean_eval": safe_nanmean(tvt),
                "tvt_range_eval": float(np.nanmax(tvt) - np.nanmin(tvt)),
                "formation_abs_slope_mean": safe_nanmean(marker_abs_slopes),
                "formation_abs_slope_max": float(np.nanmax(marker_abs_slopes)) if marker_abs_slopes else np.nan,
                "formation_slope_std": safe_nanstd(marker_slopes),
                "gr_missing_prefix": float(np.mean(~np.isfinite(gr_prefix))),
                "gr_missing_eval": float(np.mean(~np.isfinite(gr_eval))),
                "gr_longest_nan_eval": longest_nan_run(gr_eval),
                "gr_mean_eval": safe_nanmean(gr_eval),
                "gr_std_eval": safe_nanstd(gr_eval),
                "gr_half_mean_diff": float(abs(half_means[1] - half_means[0]))
                if all(np.isfinite(half_means))
                else np.nan,
                "gr_quarter_max_jump": float(max(quarter_diffs)) if quarter_diffs else np.nan,
                "gr_slope_first_half": slope_first,
                "gr_slope_second_half": slope_second,
                "gr_slope_change": float(abs(slope_second - slope_first))
                if np.isfinite(slope_first) and np.isfinite(slope_second)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gr_best_rmse"] = out[["hmm_rmse", "likpf_rmse", "pf_ancc_rmse"]].min(axis=1)
    out["gr_worst_rmse"] = out[["hmm_rmse", "likpf_rmse", "pf_ancc_rmse"]].max(axis=1)
    out["pf_best_rmse"] = out[["likpf_rmse", "pf_ancc_rmse"]].min(axis=1)
    out["pf_worst_rmse"] = out[["likpf_rmse", "pf_ancc_rmse"]].max(axis=1)
    out["hmm_likpf_abs_gap"] = (out["hmm_rmse"] - out["likpf_rmse"]).abs()
    out["hmm_exp226_abs_gap"] = (out["hmm_rmse"] - out["exp226_rmse"]).abs()
    out["likpf_exp226_abs_gap"] = (out["likpf_rmse"] - out["exp226_rmse"]).abs()
    out["pf_ancc_exp226_abs_gap"] = (out["pf_ancc_rmse"] - out["exp226_rmse"]).abs()
    return out


def category_masks(frame: pd.DataFrame, bad: float, good: float) -> dict[str, pd.Series]:
    return {
        "pf_bad_hmm_good": (frame["likpf_rmse"] >= bad) & (frame["hmm_rmse"] <= good),
        "hmm_bad_pf_good": (frame["hmm_rmse"] >= bad) & (frame["likpf_rmse"] <= good),
        "hmm_bad_exp226_good": (frame["hmm_rmse"] >= bad) & (frame["exp226_rmse"] <= good),
        "likpf_bad_exp226_good": (frame["likpf_rmse"] >= bad) & (frame["exp226_rmse"] <= good),
        "pf_ancc_bad_exp226_good": (frame["pf_ancc_rmse"] >= bad) & (frame["exp226_rmse"] <= good),
        "any_gr_bad_exp226_good": (frame["gr_worst_rmse"] >= bad) & (frame["exp226_rmse"] <= good),
        "hmm_and_pf_bad_exp226_good": (
            (frame["hmm_rmse"] >= bad) & (frame["pf_best_rmse"] >= bad) & (frame["exp226_rmse"] <= good)
        ),
        "exp226_bad_hmm_good": (frame["exp226_rmse"] >= bad) & (frame["hmm_rmse"] <= good),
        "exp226_bad_likpf_good": (frame["exp226_rmse"] >= bad) & (frame["likpf_rmse"] <= good),
        "exp226_bad_pf_ancc_good": (frame["exp226_rmse"] >= bad) & (frame["pf_ancc_rmse"] <= good),
        "exp226_bad_any_gr_good": (frame["exp226_rmse"] >= bad) & (frame["gr_best_rmse"] <= good),
        "exp226_bad_hmm_and_pf_good": (
            (frame["exp226_rmse"] >= bad)
            & (frame["hmm_rmse"] <= good)
            & (frame["pf_best_rmse"] <= good)
        ),
    }


def add_neighbor_columns(frame: pd.DataFrame, masks: dict[str, pd.Series], k: int) -> pd.DataFrame:
    out = frame.copy()
    coords = out[["well", "x_mean", "y_mean"]].dropna().copy()
    xy = coords[["x_mean", "y_mean"]].to_numpy(dtype=float)
    wells = coords["well"].astype(str).tolist()
    well_to_idx = {well: i for i, well in enumerate(wells)}
    mask_sets = {name: set(out.loc[mask, "well"].astype(str)) for name, mask in masks.items()}
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        well = str(row["well"])
        if well not in well_to_idx:
            continue
        idx = well_to_idx[well]
        dist = np.sqrt(((xy - xy[idx]) ** 2).sum(axis=1))
        order = np.argsort(dist)
        neighbor_ids = [wells[j] for j in order if wells[j] != well][:k]
        neighbor_frame = out.loc[out["well"].isin(neighbor_ids)]
        item: dict[str, Any] = {
            "well": well,
            "nearest8_min_dist": float(dist[order[1]]) if len(order) > 1 else np.nan,
            "nearest8_med_dist": float(np.median([dist[well_to_idx[n]] for n in neighbor_ids]))
            if neighbor_ids
            else np.nan,
            "nearest8_wells": " ".join(neighbor_ids),
            "same_typewell_nearest8_count": int(
                (neighbor_frame["exact_typewell_group"] == row["exact_typewell_group"]).sum()
            ),
        }
        for name, values in mask_sets.items():
            item[f"neighbor_{name}_count"] = sum(neighbor in values for neighbor in neighbor_ids)
        for col in ["exp226_rmse", "hmm_rmse", "likpf_rmse", "pf_ancc_rmse", "gr_best_rmse", "gr_worst_rmse"]:
            item[f"neighbor_median_{col}"] = (
                float(neighbor_frame[col].median()) if len(neighbor_frame) else np.nan
            )
        rows.append(item)
    return out.merge(pd.DataFrame(rows), on="well", how="left")


def make_category_wells(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    keep = [
        "well",
        "x_mean",
        "y_mean",
        "exact_typewell_group",
        "exp226_rmse",
        "hmm_rmse",
        "hmm_bias",
        "likpf_rmse",
        "pf_ancc_rmse",
        "gr_best_rmse",
        "gr_worst_rmse",
        "exp226_bias",
        "likpf_mean_bias",
        "pf_ancc_bias",
        "hmm_std_mean",
        "self_gr_valid_rate",
        "self_gr_quality_mean",
        "gr_missing_eval",
        "formation_abs_slope_mean",
        "z_mean_eval",
        "z_range_eval",
        "tvt_range_eval",
        "donor_dist_min",
        "donor_dist_max",
        "delta_abs_median",
        "end_minus_anchor",
        "nearest8_min_dist",
        "nearest8_med_dist",
        "same_typewell_nearest8_count",
        "nearest8_wells",
    ]
    keep = [col for col in keep if col in frame.columns]
    rows = []
    for name, mask in masks.items():
        sub = frame.loc[mask, keep].copy()
        if sub.empty:
            continue
        sub.insert(0, "category", name)
        sub.insert(1, "category_count", len(sub))
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["category", *keep])


def make_category_summary(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "exp226_rmse",
        "hmm_rmse",
        "likpf_rmse",
        "pf_ancc_rmse",
        "gr_best_rmse",
        "gr_worst_rmse",
        "exp226_bias",
        "hmm_bias",
        "likpf_mean_bias",
        "pf_ancc_bias",
        "gr_missing_eval",
        "hmm_std_mean",
        "self_gr_valid_rate",
        "donor_dist_min",
        "donor_dist_max",
        "z_range_eval",
        "tvt_range_eval",
        "formation_abs_slope_mean",
    ]
    for name, mask in masks.items():
        sub = frame.loc[mask]
        row: dict[str, Any] = {"category": name, "well_count": int(len(sub))}
        for col in metric_cols:
            if col not in sub:
                continue
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_feature_summary(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        sub = frame.loc[mask]
        for col in FEATURE_SUMMARY_COLUMNS:
            if col not in frame:
                continue
            values = pd.to_numeric(sub[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else np.nan
            rows.append(
                {
                    "category": name,
                    "well_count": int(len(sub)),
                    "feature": col,
                    "group_median": median,
                    "all_median": float(pd.to_numeric(frame[col], errors="coerce").median()),
                    "group_median_percentile_vs_all": pct_of_all(frame[col], median),
                }
            )
    return pd.DataFrame(rows)


def make_typewell_context(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        wells = set(frame.loc[mask, "well"])
        if not wells:
            continue
        groups = set(frame.loc[frame["well"].isin(wells), "exact_typewell_group"].dropna())
        for group_name, group in frame.loc[frame["exact_typewell_group"].isin(groups)].groupby(
            "exact_typewell_group"
        ):
            rows.append(
                {
                    "category": name,
                    "exact_typewell_group": group_name,
                    "group_wells": int(len(group)),
                    "category_wells_in_group": int(group["well"].isin(wells).sum()),
                    "exp226_rmse_median": float(group["exp226_rmse"].median()),
                    "hmm_rmse_median": float(group["hmm_rmse"].median()),
                    "likpf_rmse_median": float(group["likpf_rmse"].median()),
                    "pf_ancc_rmse_median": float(group["pf_ancc_rmse"].median()),
                    "gr_missing_eval_median": float(group["gr_missing_eval"].median()),
                    "formation_abs_slope_mean_median": float(group["formation_abs_slope_mean"].median()),
                }
            )
    return pd.DataFrame(rows)


def build_joined(root: Path, paths: dict[str, Path], skip_hmm_row_bias: bool) -> pd.DataFrame:
    exp226 = read_exp226_by_well(paths["exp226_by_well"])
    hmm = read_hmm_by_well(paths["hmm_by_well"])
    hmm_gen = read_hmm_generation(paths["hmm_generation"])
    pf = read_pf_map(paths["pf_map"])
    pos = read_positions(paths["position_summary"])

    joined = (
        exp226.merge(hmm, on="well", how="inner")
        .merge(hmm_gen, on="well", how="left")
        .merge(pf, on="well", how="left")
        .merge(pos, on="well", how="left")
    )
    if not skip_hmm_row_bias:
        joined = joined.merge(read_hmm_row_bias(paths["hmm_row_features"]), on="well", how="left")
    raw_features = raw_eval_features(root, joined)
    joined = joined.merge(raw_features, on="well", how="left")
    return add_derived_columns(joined)


def write_readme(
    output_dir: Path,
    joined: pd.DataFrame,
    category_summary: pd.DataFrame,
    bad: float,
    good: float,
) -> None:
    count = dict(zip(category_summary["category"], category_summary["well_count"], strict=False))
    lines = [
        "# HMM / PF / exp226 well pattern readout 2026-07-12",
        "",
        "既存 OOF / by-well artifact を well 単位で結合した diagnostic study。",
        "新規学習、提出候補、anchor 更新ではない。",
        "",
        "## Inputs",
        "",
        "- HMM: exp223 `hmm_selfgr_boost_only_a070_c100`",
        "- PF primary: exp072 `likPF_mean`",
        "- pure PF: exp072 `pf_ancc`",
        "- exp226: train OOF by-well metrics",
        f"- 大外し閾値: RMSE >= {bad:g}",
        f"- 当たり閾値: RMSE <= {good:g}",
        "",
        "## Key Counts",
        "",
        "| category | wells |",
        "| --- | ---: |",
    ]
    for category in [
        "pf_bad_hmm_good",
        "hmm_bad_pf_good",
        "hmm_bad_exp226_good",
        "likpf_bad_exp226_good",
        "pf_ancc_bad_exp226_good",
        "any_gr_bad_exp226_good",
        "hmm_and_pf_bad_exp226_good",
        "exp226_bad_any_gr_good",
        "exp226_bad_hmm_good",
        "exp226_bad_likpf_good",
        "exp226_bad_hmm_and_pf_good",
    ]:
        lines.append(f"| `{category}` | {int(count.get(category, 0))} |")
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "- `hmm_and_pf_bad_exp226_good` は strict 条件では 0 本。",
            "- `hmm_bad_exp226_good` は GR 欠損、低 self-GR valid rate、高 HMM std が目立つ。",
            "- `likpf_bad_exp226_good` は GR 欠損ではなく PF/likPF branch offset が主因に見える。",
            "- `exp226_bad_any_gr_good` は donor 距離が大きく、GR 欠損が少ない。z/geometry donor が外れ、GR 系が補正したケース。",
            "- 直接 replacement / global fixed blend ではなく、confidence / selector feature として使うのが安全。",
            "",
            "## Outputs",
            "",
            "- `joined_well_summary.csv`: 773 well の結合表。",
            "- `category_wells.csv`: 条件に該当した well の long table。",
            "- `category_summary.csv`: 条件別の主要メトリクス集計。",
            "- `feature_summary.csv`: 条件別 feature median と全体 percentile。",
            "- `typewell_context.csv`: 該当 well が属する typewell group の文脈。",
            "- `source_manifest.json`: 入力 artifact と実行条件。",
            "",
            "## Source Docs",
            "",
            "- `docs/surveys/hmm_pf_exp226_well_pattern_readout_20260712.md` に人間向けの解釈を記録。",
            "",
            f"Rows in `joined_well_summary.csv`: {len(joined)}",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "hmm_by_well": Path(
            "/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/"
            "exp223_joint_typewell_self_gr_hmm_likelihood_probe_by_well_delta.csv"
        ),
        "hmm_generation": Path(
            "/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/"
            "exp223_joint_typewell_self_gr_hmm_likelihood_probe_by_well_generation_summary.csv"
        ),
        "hmm_row_features": Path(
            "/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/"
            "exp223_joint_typewell_self_gr_hmm_likelihood_probe_joint_typewell_self_gr_hmm_likelihood_probe_train_features.csv.gz"
        ),
        "pf_map": root / "studies/pf_beam_disagreement_error_map/pf_beam_disagreement_well_map.csv",
        "position_summary": root
        / "studies/typewell_position_groups/native_overlap_1_well_position_typewell_summary.csv",
        "exp226_by_well": Path(
            "/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/"
            "train_v1/artifacts/"
            "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_by_well_metrics.csv"
        ),
    }

    joined = build_joined(root, paths, skip_hmm_row_bias=args.skip_hmm_row_bias)
    masks = category_masks(joined, bad=args.bad_rmse, good=args.good_rmse)
    joined = add_neighbor_columns(joined, masks, k=args.neighbor_k)
    masks = category_masks(joined, bad=args.bad_rmse, good=args.good_rmse)

    category_wells = make_category_wells(joined, masks)
    category_summary = make_category_summary(joined, masks)
    feature_summary = make_feature_summary(joined, masks)
    typewell_context = make_typewell_context(joined, masks)

    joined.to_csv(output_dir / "joined_well_summary.csv", index=False)
    category_wells.to_csv(output_dir / "category_wells.csv", index=False)
    category_summary.to_csv(output_dir / "category_summary.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    typewell_context.to_csv(output_dir / "typewell_context.csv", index=False)

    manifest = {
        "study": "hmm_pf_exp226_well_pattern_readout_20260712",
        "bad_rmse": args.bad_rmse,
        "good_rmse": args.good_rmse,
        "neighbor_k": args.neighbor_k,
        "rows": int(len(joined)),
        "wells": int(joined["well"].nunique()),
        "skip_hmm_row_bias": bool(args.skip_hmm_row_bias),
        "inputs": {name: file_info(path) for name, path in paths.items()},
        "outputs": {
            "joined_well_summary": "joined_well_summary.csv",
            "category_wells": "category_wells.csv",
            "category_summary": "category_summary.csv",
            "feature_summary": "feature_summary.csv",
            "typewell_context": "typewell_context.csv",
            "readme": "README.md",
        },
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, joined, category_summary, args.bad_rmse, args.good_rmse)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(joined)}, sort_keys=True))


if __name__ == "__main__":
    main()
