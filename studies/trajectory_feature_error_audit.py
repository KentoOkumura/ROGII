from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exp010 trajectory feature RMSE deltas at well level."
    )
    parser.add_argument(
        "--well-metrics",
        type=Path,
        default=ROOT
        / "experiments"
        / "exp010_trajectory_drift_ablation"
        / "artifacts"
        / "well_metrics.csv",
        help="exp010 well_metrics.csv path.",
    )
    parser.add_argument(
        "--router-tags",
        type=Path,
        default=ROOT
        / "experiments"
        / "exp006_hard_well_router_diagnostic"
        / "artifacts"
        / "router_diagnostic_well_tags.csv",
        help="exp006 router diagnostic tags with GR-missing and hard-well columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments"
        / "exp010_trajectory_drift_ablation"
        / "artifacts"
        / "trajectory_feature_error_audit",
        help="Directory for audit outputs.",
    )
    parser.add_argument(
        "--meaningful-delta",
        type=float,
        default=0.25,
        help="Per-well RMSE delta threshold for better/hurt labels.",
    )
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def variant_slice(well_metrics: pd.DataFrame, variant: str) -> pd.DataFrame:
    subset = well_metrics.loc[well_metrics["variant"] == variant].copy()
    if subset.empty:
        available = sorted(well_metrics["variant"].dropna().astype(str).unique())
        raise ValueError(f"variant not found: {variant}; available={available}")
    duplicated = subset["well_id"].duplicated()
    if duplicated.any():
        examples = subset.loc[duplicated, "well_id"].head(5).tolist()
        raise ValueError(f"variant {variant} has duplicated well rows: {examples}")
    return subset.set_index("well_id").sort_index()


def weighted_rmse(frame: pd.DataFrame, column: str) -> float | None:
    rows = numeric(frame["n_eval"]).to_numpy(dtype=float)
    rmse = numeric(frame[column]).to_numpy(dtype=float)
    valid = np.isfinite(rows) & np.isfinite(rmse) & (rows > 0)
    if not valid.any():
        return None
    return float(sqrt(np.sum(rows[valid] * rmse[valid] * rmse[valid]) / np.sum(rows[valid])))


def finite_float(value: Any, digits: int = 6) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return round(parsed, digits)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def classify_delta(delta: pd.Series, threshold: float) -> pd.Series:
    values = numeric(delta)
    labels = np.full(len(values), "similar", dtype=object)
    labels[values <= -threshold] = "trajectory_better"
    labels[values >= threshold] = "trajectory_hurt"
    labels[values >= 2.0] = "trajectory_hurt_major"
    labels[values >= 5.0] = "trajectory_hurt_severe"
    labels[~np.isfinite(values.to_numpy(dtype=float))] = "unknown"
    return pd.Series(labels, index=delta.index)


def make_bin(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    values = numeric(series)
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True).astype("object")


def load_router_tags(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    tags = pd.read_csv(path)
    keep_columns = [
        "well_id",
        "condition_prefix_gr_missing_rate",
        "condition_eval_gr_missing_rate",
        "condition_prefix_fraction",
        "condition_eval_row_count",
        "condition_trajectory_abs_dz_dmd",
        "condition_trajectory_delta_xy",
        "condition_trajectory_delta_z",
        "candidate_router_bucket",
        "gr_weak_any",
        "gr_weak_all",
        "gr_strong",
        "short_prefix",
        "long_eval",
        "steep_trajectory",
        "large_gr_shift",
        "hard_exp002_failure",
        "public_like_keep_all_gr",
        "hard_no_gr_candidate",
    ]
    return tags[[column for column in keep_columns if column in tags.columns]].copy()


def build_audit_frame(
    well_metrics: pd.DataFrame,
    router_tags: pd.DataFrame,
    meaningful_delta: float,
) -> pd.DataFrame:
    control = variant_slice(well_metrics, "control_exp003_no_gr")
    exp002 = variant_slice(well_metrics, "control_exp002_all")
    direction = variant_slice(well_metrics, "trajectory_direction_no_gr")
    slope = variant_slice(well_metrics, "trajectory_slope_no_gr")
    full = variant_slice(well_metrics, "trajectory_full_no_gr")
    full_all = variant_slice(well_metrics, "trajectory_full_all")

    base_columns = [
        "fold",
        "n_rows",
        "n_known",
        "n_eval",
        "last_known_index",
        "last_known_md",
        "last_known_tvt",
        "recent_slope",
        "anchor_azimuth_sin",
        "anchor_azimuth_cos",
        "anchor_dz_dmd_minus_prefix_dz_dmd",
        "anchor_dxy_dmd_minus_prefix_dxy_dmd",
        "target_residual_mean",
        "last_anchor_rmse",
    ]
    frame = control[base_columns].copy()
    frame["exp003_no_gr_rmse"] = numeric(control["drift_hgb_rmse"])
    frame["exp002_all_rmse"] = numeric(exp002["drift_hgb_rmse"])
    frame["trajectory_direction_rmse"] = numeric(direction["drift_hgb_rmse"])
    frame["trajectory_slope_rmse"] = numeric(slope["drift_hgb_rmse"])
    frame["trajectory_full_rmse"] = numeric(full["drift_hgb_rmse"])
    frame["trajectory_full_all_rmse"] = numeric(full_all["drift_hgb_rmse"])

    frame["direction_delta_vs_exp003"] = (
        frame["trajectory_direction_rmse"] - frame["exp003_no_gr_rmse"]
    )
    frame["slope_delta_vs_exp003"] = frame["trajectory_slope_rmse"] - frame["exp003_no_gr_rmse"]
    frame["full_delta_vs_exp003"] = frame["trajectory_full_rmse"] - frame["exp003_no_gr_rmse"]
    frame["full_all_delta_vs_exp002"] = (
        frame["trajectory_full_all_rmse"] - frame["exp002_all_rmse"]
    )
    frame["full_sse_delta_vs_exp003"] = (
        frame["trajectory_full_rmse"] * frame["trajectory_full_rmse"]
        - frame["exp003_no_gr_rmse"] * frame["exp003_no_gr_rmse"]
    ) * numeric(frame["n_eval"])
    frame["direction_sse_delta_vs_exp003"] = (
        frame["trajectory_direction_rmse"] * frame["trajectory_direction_rmse"]
        - frame["exp003_no_gr_rmse"] * frame["exp003_no_gr_rmse"]
    ) * numeric(frame["n_eval"])
    frame.index.name = "well_id"
    frame = frame.reset_index()

    if not router_tags.empty:
        frame = frame.merge(router_tags, on="well_id", how="left", validate="one_to_one")

    frame["full_delta_label"] = classify_delta(frame["full_delta_vs_exp003"], meaningful_delta)
    frame["direction_delta_label"] = classify_delta(
        frame["direction_delta_vs_exp003"], meaningful_delta
    )
    frame["eval_length_bin"] = make_bin(
        frame["n_eval"],
        [-np.inf, 3500.0, 5500.0, np.inf],
        ["short_eval", "mid_eval", "long_eval"],
    )
    frame["prefix_fraction_bin"] = make_bin(
        frame.get("condition_prefix_fraction", pd.Series(index=frame.index, dtype=float)),
        [-np.inf, 0.23, 0.30, np.inf],
        ["short_prefix", "mid_prefix", "long_prefix"],
    )
    max_gr_missing = pd.concat(
        [
            numeric(frame.get("condition_prefix_gr_missing_rate", pd.Series(index=frame.index))),
            numeric(frame.get("condition_eval_gr_missing_rate", pd.Series(index=frame.index))),
        ],
        axis=1,
    ).max(axis=1)
    frame["max_gr_missing_rate"] = max_gr_missing
    frame["gr_missing_bin"] = make_bin(
        max_gr_missing,
        [-np.inf, 0.20, 0.40, np.inf],
        ["low_gr_missing", "mid_gr_missing", "high_gr_missing"],
    )
    frame["trajectory_abs_dz_dmd_bin"] = make_bin(
        frame.get("condition_trajectory_abs_dz_dmd", pd.Series(index=frame.index, dtype=float)),
        [-np.inf, 0.025, 0.04, np.inf],
        ["flat_trajectory", "mid_trajectory", "steep_trajectory"],
    )
    frame["abs_recent_slope_bin"] = make_bin(
        numeric(frame["recent_slope"]).abs(),
        [-np.inf, 0.005, 0.015, np.inf],
        ["flat_recent_tvt", "mid_recent_tvt", "steep_recent_tvt"],
    )
    return frame


def summarize_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for value, subset in frame.groupby(group_column, dropna=False):
        record = {
            "group": group_column,
            "value": "missing" if pd.isna(value) else str(value),
            "wells": int(len(subset)),
            "eval_rows": int(numeric(subset["n_eval"]).sum()),
            "exp003_cv": finite_float(weighted_rmse(subset, "exp003_no_gr_rmse")),
            "trajectory_full_cv": finite_float(weighted_rmse(subset, "trajectory_full_rmse")),
            "trajectory_direction_cv": finite_float(
                weighted_rmse(subset, "trajectory_direction_rmse")
            ),
            "full_mean_delta": finite_float(subset["full_delta_vs_exp003"].mean()),
            "full_median_delta": finite_float(subset["full_delta_vs_exp003"].median()),
            "full_hurt_rate": finite_float(
                (subset["full_delta_label"].astype(str).str.startswith("trajectory_hurt")).mean()
            ),
            "full_better_rate": finite_float(
                (subset["full_delta_label"] == "trajectory_better").mean()
            ),
            "direction_mean_delta": finite_float(subset["direction_delta_vs_exp003"].mean()),
            "direction_hurt_rate": finite_float(
                (
                    subset["direction_delta_label"].astype(str).str.startswith("trajectory_hurt")
                ).mean()
            ),
            "max_gr_missing_mean": finite_float(subset["max_gr_missing_rate"].mean()),
            "eval_length_mean": finite_float(subset["n_eval"].mean()),
            "abs_dz_dmd_mean": finite_float(
                numeric(
                    subset.get("condition_trajectory_abs_dz_dmd", pd.Series(dtype=float))
                ).mean()
            ),
        }
        records.append(record)
    return pd.DataFrame(records)


def build_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "fold",
        "eval_length_bin",
        "prefix_fraction_bin",
        "gr_missing_bin",
        "trajectory_abs_dz_dmd_bin",
        "abs_recent_slope_bin",
        "candidate_router_bucket",
        "hard_no_gr_candidate",
        "public_like_keep_all_gr",
        "gr_weak_any",
    ]
    summaries = [summarize_group(frame, column) for column in group_columns if column in frame]
    return pd.concat(summaries, ignore_index=True)


def top_wells(frame: pd.DataFrame, n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "well_id",
        "fold",
        "n_eval",
        "exp003_no_gr_rmse",
        "trajectory_full_rmse",
        "full_delta_vs_exp003",
        "direction_delta_vs_exp003",
        "slope_delta_vs_exp003",
        "full_sse_delta_vs_exp003",
        "candidate_router_bucket",
        "max_gr_missing_rate",
        "condition_prefix_fraction",
        "condition_trajectory_abs_dz_dmd",
        "recent_slope",
        "target_residual_mean",
        "last_anchor_rmse",
    ]
    available = [column for column in columns if column in frame.columns]
    hurts = frame.sort_values("full_sse_delta_vs_exp003", ascending=False)[available].head(n)
    helps = frame.sort_values("full_sse_delta_vs_exp003", ascending=True)[available].head(n)
    return hurts, helps


def build_metrics(frame: pd.DataFrame, group_summary: pd.DataFrame) -> dict[str, Any]:
    full_hurt = frame["full_delta_label"].astype(str).str.startswith("trajectory_hurt")
    direction_hurt = frame["direction_delta_label"].astype(str).str.startswith("trajectory_hurt")
    metrics: dict[str, Any] = {
        "source": "exp010_trajectory_drift_ablation well_metrics.csv",
        "n_wells": int(len(frame)),
        "eval_rows": int(numeric(frame["n_eval"]).sum()),
        "exp003_no_gr_cv": finite_float(weighted_rmse(frame, "exp003_no_gr_rmse")),
        "trajectory_direction_cv": finite_float(weighted_rmse(frame, "trajectory_direction_rmse")),
        "trajectory_slope_cv": finite_float(weighted_rmse(frame, "trajectory_slope_rmse")),
        "trajectory_full_cv": finite_float(weighted_rmse(frame, "trajectory_full_rmse")),
        "exp002_all_cv": finite_float(weighted_rmse(frame, "exp002_all_rmse")),
        "trajectory_full_all_cv": finite_float(weighted_rmse(frame, "trajectory_full_all_rmse")),
        "full_delta_mean": finite_float(frame["full_delta_vs_exp003"].mean()),
        "full_delta_median": finite_float(frame["full_delta_vs_exp003"].median()),
        "full_hurt_wells": int(full_hurt.sum()),
        "full_hurt_rate": finite_float(full_hurt.mean()),
        "full_better_wells": int((frame["full_delta_label"] == "trajectory_better").sum()),
        "direction_hurt_wells": int(direction_hurt.sum()),
        "direction_hurt_rate": finite_float(direction_hurt.mean()),
        "largest_sse_hurts": frame.sort_values(
            "full_sse_delta_vs_exp003", ascending=False
        )["well_id"].head(10).tolist(),
        "largest_sse_helps": frame.sort_values("full_sse_delta_vs_exp003")["well_id"]
        .head(10)
        .tolist(),
    }

    focus_groups = group_summary.loc[
        (group_summary["group"] == "candidate_router_bucket")
        | (group_summary["group"] == "gr_missing_bin")
        | (group_summary["group"] == "trajectory_abs_dz_dmd_bin")
        | (group_summary["group"] == "eval_length_bin")
    ].copy()
    focus_groups["full_cv_delta"] = (
        numeric(focus_groups["trajectory_full_cv"]) - numeric(focus_groups["exp003_cv"])
    )
    metrics["worst_groups_by_full_cv_delta"] = focus_groups.sort_values(
        "full_cv_delta", ascending=False
    ).head(8).to_dict(orient="records")
    return json_ready(metrics)


def render_report(metrics: dict[str, Any], group_summary: pd.DataFrame) -> str:
    focus = group_summary.copy()
    focus["full_cv_delta"] = numeric(focus["trajectory_full_cv"]) - numeric(focus["exp003_cv"])
    worst = focus.sort_values("full_cv_delta", ascending=False).head(8)

    lines = [
        "# trajectory_feature_error_audit",
        "",
        "## Summary",
        "",
        f"- Wells audited: {metrics['n_wells']}",
        f"- Eval rows: {metrics['eval_rows']}",
        f"- exp003 no-GR CV: {metrics['exp003_no_gr_cv']}",
        f"- trajectory_direction_no_gr CV: {metrics['trajectory_direction_cv']}",
        f"- trajectory_slope_no_gr CV: {metrics['trajectory_slope_cv']}",
        f"- trajectory_full_no_gr CV: {metrics['trajectory_full_cv']}",
        f"- trajectory_full hurt wells: {metrics['full_hurt_wells']} "
        f"({metrics['full_hurt_rate']})",
        f"- trajectory_full better wells: {metrics['full_better_wells']}",
        "",
        "## Worst Groups By Full-Trajectory CV Delta",
        "",
        "| Group | Value | Wells | exp003 CV | trajectory_full CV | Delta | Hurt rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in worst.to_dict(orient="records"):
        delta = finite_float(row.get("full_cv_delta"))
        lines.append(
            f"| {row['group']} | {row['value']} | {row['wells']} | "
            f"{row['exp003_cv']} | {row['trajectory_full_cv']} | {delta} | "
            f"{row['full_hurt_rate']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Full trajectory features hurt more wells than they help and are not safe "
            "as an add-only feature group.",
            "- Direction-only is less damaging than slope/full, so trajectory geometry "
            "should not be discarded entirely.",
            "- Use these outputs to design a router or feature gate before trying "
            "tracker divergence features.",
            "",
            "## Output Files",
            "",
            "- `trajectory_audit_well_deltas.csv`",
            "- `trajectory_audit_group_summary.csv`",
            "- `trajectory_audit_top_hurts.csv`",
            "- `trajectory_audit_top_helps.csv`",
            "- `trajectory_audit_metrics.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    well_metrics = pd.read_csv(args.well_metrics)
    router_tags = load_router_tags(args.router_tags)
    frame = build_audit_frame(well_metrics, router_tags, args.meaningful_delta)
    group_summary = build_group_summary(frame)
    hurts, helps = top_wells(frame)
    metrics = build_metrics(frame, group_summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "trajectory_audit_well_deltas.csv", index=False)
    group_summary.to_csv(output_dir / "trajectory_audit_group_summary.csv", index=False)
    hurts.to_csv(output_dir / "trajectory_audit_top_hurts.csv", index=False)
    helps.to_csv(output_dir / "trajectory_audit_top_helps.csv", index=False)
    (output_dir / "trajectory_audit_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "trajectory_audit_report.md").write_text(render_report(metrics, group_summary))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
