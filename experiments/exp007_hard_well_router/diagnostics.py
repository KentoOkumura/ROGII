from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from baseline import config_get
from settings import ExperimentPaths, load_config


def finite_float(value: Any, digits: int = 6) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return round(parsed, digits)


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


def weighted_rmse(rows: pd.Series | np.ndarray, rmse: pd.Series | np.ndarray) -> float | None:
    row_values = np.asarray(rows, dtype=float)
    rmse_values = np.asarray(rmse, dtype=float)
    valid = np.isfinite(row_values) & np.isfinite(rmse_values) & (row_values > 0)
    if not valid.any():
        return None
    sse = np.sum(row_values[valid] * rmse_values[valid] * rmse_values[valid])
    count = np.sum(row_values[valid])
    if count <= 0:
        return None
    return float(sqrt(sse / count))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def effect_label(delta: pd.Series, threshold: float, alternate_name: str) -> pd.Series:
    values = numeric(delta)
    labels = np.full(len(values), "similar", dtype=object)
    labels[values <= -threshold] = f"{alternate_name}_better"
    labels[values >= threshold] = "all_gr_better"
    labels[~np.isfinite(values.to_numpy(dtype=float))] = "unknown"
    return pd.Series(labels, index=delta.index)


def diagnostic_thresholds(config: dict[str, Any]) -> dict[str, float]:
    defaults = {
        "prefix_gr_missing_rate": 0.35,
        "eval_gr_missing_rate": 0.40,
        "prefix_fraction_short": 0.23,
        "eval_row_count_long": 5700.0,
        "trajectory_abs_dz_dmd_high": 0.04,
        "gr_delta_abs_mean_high": 15.0,
    }
    raw = config_get(config, "diagnostic.thresholds", {})
    if isinstance(raw, dict):
        for key in defaults:
            if key in raw and raw[key] is not None:
                defaults[key] = float(raw[key])
    return defaults


def attach_variant_columns(
    frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    frame[f"{prefix}_rmse"] = numeric(variant_frame["drift_hgb_rmse"])
    if "gr_gate_weight" in variant_frame.columns:
        frame[f"{prefix}_gate_weight"] = numeric(variant_frame["gr_gate_weight"]).fillna(0.0)


def build_router_frame(
    well_metrics: pd.DataFrame,
    config: dict[str, Any],
    exp003_delta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    control_variant = str(config_get(config, "diagnostic.control_variant", "control_exp002_all"))
    no_gr_variant = str(config_get(config, "diagnostic.no_gr_variant", "control_exp003_no_gr"))
    guarded_variant = str(
        config_get(config, "diagnostic.guarded_variant", "gate_low_gr_strict_hard")
    )
    comparison_variant = str(
        config_get(config, "diagnostic.comparison_variant", "control_exp004_low_gr_any_hard")
    )
    threshold = float(config_get(config, "diagnostic.meaningful_rmse_delta", 0.25))
    thresholds = diagnostic_thresholds(config)

    control = variant_slice(well_metrics, control_variant)
    no_gr = variant_slice(well_metrics, no_gr_variant)
    guarded = variant_slice(well_metrics, guarded_variant)
    comparison = variant_slice(well_metrics, comparison_variant)

    condition_columns = [column for column in control.columns if column.startswith("condition_")]
    base_columns = [
        "fold",
        "n_rows",
        "n_known",
        "n_eval",
        "last_known_index",
        "last_known_md",
        "last_known_tvt",
        "recent_slope",
        "target_residual_mean",
        "last_anchor_rmse",
        *condition_columns,
    ]
    frame = control[base_columns].copy()
    frame["exp002_rmse"] = numeric(control["drift_hgb_rmse"])
    attach_variant_columns(frame, no_gr, prefix="exp003")
    attach_variant_columns(frame, comparison, prefix="exp004")
    attach_variant_columns(frame, guarded, prefix="exp005")
    frame.index.name = "well_id"
    frame = frame.reset_index()

    frame["rmse_delta_003_minus_002"] = frame["exp003_rmse"] - frame["exp002_rmse"]
    frame["rmse_delta_005_minus_002"] = frame["exp005_rmse"] - frame["exp002_rmse"]
    frame["rmse_delta_004_minus_002"] = frame["exp004_rmse"] - frame["exp002_rmse"]
    frame["sse_delta_003_minus_002"] = (
        frame["exp003_rmse"] * frame["exp003_rmse"]
        - frame["exp002_rmse"] * frame["exp002_rmse"]
    ) * frame["n_eval"]
    frame["sse_delta_005_minus_002"] = (
        frame["exp005_rmse"] * frame["exp005_rmse"]
        - frame["exp002_rmse"] * frame["exp002_rmse"]
    ) * frame["n_eval"]

    frame["exp002_vs_anchor"] = np.where(
        frame["exp002_rmse"] < frame["last_anchor_rmse"],
        "exp002_better_than_anchor",
        "exp002_worse_than_anchor",
    )
    frame["no_gr_effect"] = effect_label(
        frame["rmse_delta_003_minus_002"], threshold, "no_gr"
    )
    frame["guarded_effect"] = effect_label(
        frame["rmse_delta_005_minus_002"], threshold, "guarded"
    )

    prefix_missing = numeric(frame["condition_prefix_gr_missing_rate"])
    eval_missing = numeric(frame["condition_eval_gr_missing_rate"])
    prefix_fraction = numeric(frame["condition_prefix_fraction"])
    eval_rows = numeric(frame["condition_eval_row_count"])
    abs_dz_dmd = numeric(frame["condition_trajectory_abs_dz_dmd"])
    gr_delta = numeric(frame["condition_gr_delta_abs_mean"])

    frame["gr_weak_any"] = (prefix_missing >= thresholds["prefix_gr_missing_rate"]) | (
        eval_missing >= thresholds["eval_gr_missing_rate"]
    )
    frame["gr_weak_all"] = (prefix_missing >= thresholds["prefix_gr_missing_rate"]) & (
        eval_missing >= thresholds["eval_gr_missing_rate"]
    )
    frame["gr_strong"] = ~frame["gr_weak_any"]
    frame["short_prefix"] = prefix_fraction <= thresholds["prefix_fraction_short"]
    frame["long_eval"] = eval_rows >= thresholds["eval_row_count_long"]
    frame["steep_trajectory"] = abs_dz_dmd >= thresholds["trajectory_abs_dz_dmd_high"]
    frame["large_gr_shift"] = gr_delta >= thresholds["gr_delta_abs_mean_high"]
    frame["hard_exp002_failure"] = (
        frame["exp002_vs_anchor"] == "exp002_worse_than_anchor"
    ) | (frame["exp002_rmse"] >= frame["exp002_rmse"].quantile(0.75))
    frame["no_gr_improved"] = frame["rmse_delta_003_minus_002"] <= -threshold
    frame["no_gr_hurt"] = frame["rmse_delta_003_minus_002"] >= threshold
    frame["guarded_improved"] = frame["rmse_delta_005_minus_002"] <= -threshold
    frame["public_like_keep_all_gr"] = frame["gr_strong"] & frame["no_gr_hurt"]
    frame["hard_no_gr_candidate"] = frame["no_gr_improved"] & (
        frame["gr_weak_any"] | frame["hard_exp002_failure"]
    )

    conditions = [
        frame["public_like_keep_all_gr"],
        frame["hard_no_gr_candidate"],
        frame["guarded_improved"],
    ]
    choices = [
        "public_like_keep_all_gr",
        "hard_no_gr_candidate",
        "guarded_candidate",
    ]
    frame["candidate_router_bucket"] = np.select(conditions, choices, default="ambiguous")

    if exp003_delta is not None and not exp003_delta.empty:
        merge_columns = [
            "well_id",
            "rmse_delta_003_minus_002",
            "sse_delta_003_minus_002",
            "exp002_vs_anchor",
        ]
        available_columns = [column for column in merge_columns if column in exp003_delta.columns]
        artifact_delta = exp003_delta[available_columns].copy()
        rename_map = {
            column: f"exp003_artifact_{column}"
            for column in available_columns
            if column != "well_id"
        }
        frame = frame.merge(artifact_delta.rename(columns=rename_map), on="well_id", how="left")

    return frame


def summarize_groups(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "candidate_router_bucket",
        "no_gr_effect",
        "guarded_effect",
        "exp002_vs_anchor",
        "gr_weak_any",
        "gr_weak_all",
        "short_prefix",
        "long_eval",
        "steep_trajectory",
    ]
    records: list[dict[str, Any]] = []
    for group_column in group_columns:
        if group_column not in frame.columns:
            continue
        for group_value, part in frame.groupby(group_column, dropna=False):
            records.append(
                {
                    "group": group_column,
                    "value": str(group_value),
                    "wells": int(len(part)),
                    "eval_rows": int(numeric(part["n_eval"]).sum()),
                    "exp002_cv": finite_float(weighted_rmse(part["n_eval"], part["exp002_rmse"])),
                    "exp003_cv": finite_float(weighted_rmse(part["n_eval"], part["exp003_rmse"])),
                    "exp005_cv": finite_float(weighted_rmse(part["n_eval"], part["exp005_rmse"])),
                    "mean_no_gr_delta": finite_float(part["rmse_delta_003_minus_002"].mean()),
                    "mean_guarded_delta": finite_float(part["rmse_delta_005_minus_002"].mean()),
                    "exp002_worse_anchor_rate": finite_float(
                        (part["exp002_vs_anchor"] == "exp002_worse_than_anchor").mean()
                    ),
                    "prefix_gr_missing_mean": finite_float(
                        part["condition_prefix_gr_missing_rate"].mean()
                    ),
                    "eval_gr_missing_mean": finite_float(
                        part["condition_eval_gr_missing_rate"].mean()
                    ),
                    "prefix_fraction_mean": finite_float(part["condition_prefix_fraction"].mean()),
                    "eval_row_count_mean": finite_float(part["condition_eval_row_count"].mean()),
                }
            )
    return pd.DataFrame(records)


def evaluate_rule(
    frame: pd.DataFrame,
    *,
    rule: str,
    mask: pd.Series,
    alternate_column: str,
    inference_safe: bool,
    route_to: str,
) -> dict[str, Any]:
    selected = mask.fillna(False).astype(bool)
    routed_rmse = frame["exp002_rmse"].where(~selected, frame[alternate_column])
    part = frame.loc[selected]
    threshold_win = float(config_get_placeholder(frame, "meaningful_rmse_delta", 0.0))
    routed_cv = weighted_rmse(frame["n_eval"], routed_rmse)
    all_gr_cv = weighted_rmse(frame["n_eval"], frame["exp002_rmse"])
    return {
        "rule": rule,
        "inference_safe": bool(inference_safe),
        "route_to": route_to,
        "selected_wells": int(selected.sum()),
        "selected_eval_rows": int(numeric(frame.loc[selected, "n_eval"]).sum()),
        "cv": finite_float(routed_cv),
        "delta_vs_all_gr": finite_float(
            None if routed_cv is None or all_gr_cv is None else routed_cv - all_gr_cv
        ),
        "selected_no_gr_win_rate": finite_float(
            (part["rmse_delta_003_minus_002"] < 0).mean() if len(part) else None
        ),
        "selected_no_gr_mean_delta": finite_float(
            part["rmse_delta_003_minus_002"].mean() if len(part) else None
        ),
        "selected_no_gr_hurt_rate": finite_float(
            (part["rmse_delta_003_minus_002"] >= threshold_win).mean() if len(part) else None
        ),
        "selected_exp002_worse_anchor_rate": finite_float(
            (part["exp002_vs_anchor"] == "exp002_worse_than_anchor").mean()
            if len(part)
            else None
        ),
        "selected_prefix_gr_missing_mean": finite_float(
            part["condition_prefix_gr_missing_rate"].mean() if len(part) else None
        ),
        "selected_eval_gr_missing_mean": finite_float(
            part["condition_eval_gr_missing_rate"].mean() if len(part) else None
        ),
        "selected_prefix_fraction_mean": finite_float(
            part["condition_prefix_fraction"].mean() if len(part) else None
        ),
    }


def config_get_placeholder(frame: pd.DataFrame, key: str, default: float) -> float:
    value = frame.attrs.get(key, default)
    return float(value if value is not None else default)


def evaluate_router_rules(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    threshold = float(config_get(config, "diagnostic.meaningful_rmse_delta", 0.25))
    frame.attrs["meaningful_rmse_delta"] = threshold
    low_gr_any = frame["gr_weak_any"]
    low_gr_all = frame["gr_weak_all"]
    long_eval_steep = frame["long_eval"] & frame["steep_trajectory"]
    short_prefix_low_gr = frame["short_prefix"] & frame["gr_weak_any"]
    large_gr_shift_low_gr = frame["large_gr_shift"] & frame["gr_weak_any"]
    exp004_gate = numeric(frame.get("exp004_gate_weight", pd.Series(0.0, index=frame.index))) > 0
    exp005_gate = numeric(frame.get("exp005_gate_weight", pd.Series(0.0, index=frame.index))) > 0

    records = [
        evaluate_rule(
            frame,
            rule="default_all_gr",
            mask=pd.Series(False, index=frame.index),
            alternate_column="exp003_rmse",
            inference_safe=True,
            route_to="exp002_all_gr",
        ),
        evaluate_rule(
            frame,
            rule="low_gr_any_to_no_gr",
            mask=low_gr_any,
            alternate_column="exp003_rmse",
            inference_safe=True,
            route_to="exp003_no_gr",
        ),
        evaluate_rule(
            frame,
            rule="low_gr_all_to_no_gr",
            mask=low_gr_all,
            alternate_column="exp003_rmse",
            inference_safe=True,
            route_to="exp003_no_gr",
        ),
        evaluate_rule(
            frame,
            rule="short_prefix_low_gr_to_no_gr",
            mask=short_prefix_low_gr,
            alternate_column="exp003_rmse",
            inference_safe=True,
            route_to="exp003_no_gr",
        ),
        evaluate_rule(
            frame,
            rule="low_gr_or_long_eval_steep_to_no_gr",
            mask=low_gr_all | long_eval_steep | large_gr_shift_low_gr,
            alternate_column="exp003_rmse",
            inference_safe=True,
            route_to="exp003_no_gr",
        ),
        evaluate_rule(
            frame,
            rule="exp004_any_gate_to_guarded",
            mask=exp004_gate,
            alternate_column="exp004_rmse",
            inference_safe=True,
            route_to="exp004_any_guarded",
        ),
        evaluate_rule(
            frame,
            rule="exp005_strict_gate_to_guarded",
            mask=exp005_gate,
            alternate_column="exp005_rmse",
            inference_safe=True,
            route_to="exp005_strict_guarded",
        ),
    ]

    oracle_rmse = np.minimum(frame["exp002_rmse"], frame["exp003_rmse"])
    oracle_selected = frame["exp003_rmse"] < frame["exp002_rmse"]
    oracle_record = evaluate_rule(
        frame,
        rule="oracle_best_of_all_gr_no_gr",
        mask=oracle_selected,
        alternate_column="exp003_rmse",
        inference_safe=False,
        route_to="target-leaking oracle",
    )
    oracle_record["cv"] = finite_float(weighted_rmse(frame["n_eval"], oracle_rmse))
    oracle_record["delta_vs_all_gr"] = finite_float(
        weighted_rmse(frame["n_eval"], oracle_rmse)
        - weighted_rmse(frame["n_eval"], frame["exp002_rmse"])
    )
    records.append(oracle_record)
    return pd.DataFrame(records).sort_values(["inference_safe", "cv"], ascending=[False, True])


def build_metrics(
    frame: pd.DataFrame,
    rules: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    inference_safe_rules = rules.loc[rules["inference_safe"]].copy()
    best_rule = None
    if not inference_safe_rules.empty:
        best_rule = json_ready(inference_safe_rules.sort_values("cv").iloc[0].to_dict())

    return {
        "n_wells": int(len(frame)),
        "rows": int(numeric(frame["n_eval"]).sum()),
        "all_gr_cv": finite_float(weighted_rmse(frame["n_eval"], frame["exp002_rmse"])),
        "no_gr_cv": finite_float(weighted_rmse(frame["n_eval"], frame["exp003_rmse"])),
        "guarded_cv": finite_float(weighted_rmse(frame["n_eval"], frame["exp005_rmse"])),
        "comparison_guarded_cv": finite_float(
            weighted_rmse(frame["n_eval"], frame["exp004_rmse"])
        ),
        "meaningful_rmse_delta": finite_float(
            config_get(config, "diagnostic.meaningful_rmse_delta", 0.25)
        ),
        "thresholds": diagnostic_thresholds(config),
        "bucket_counts": {
            str(key): int(value)
            for key, value in frame["candidate_router_bucket"].value_counts().items()
        },
        "best_inference_safe_rule": best_rule,
        "output_files": {
            "well_tags": str(config_get(config, "diagnostic.outputs.well_tags", "")),
            "condition_summary": str(
                config_get(config, "diagnostic.outputs.condition_summary", "")
            ),
            "candidate_rules": str(config_get(config, "diagnostic.outputs.candidate_rules", "")),
        },
    }


def write_router_diagnostic_artifacts(
    well_metrics: pd.DataFrame,
    config: dict[str, Any],
    artifacts_dir: Path,
    exp003_delta: pd.DataFrame | None = None,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frame = build_router_frame(well_metrics, config, exp003_delta=exp003_delta)
    summary = summarize_groups(frame)
    rules = evaluate_router_rules(frame, config)
    metrics = build_metrics(frame, rules, config)

    well_tags_name = str(
        config_get(config, "diagnostic.outputs.well_tags", "router_diagnostic_well_tags.csv")
    )
    summary_name = str(
        config_get(config, "diagnostic.outputs.condition_summary", "router_condition_summary.csv")
    )
    rules_name = str(
        config_get(config, "diagnostic.outputs.candidate_rules", "router_candidate_rules.csv")
    )
    metrics_name = str(
        config_get(config, "diagnostic.outputs.metrics", "router_diagnostic_metrics.json")
    )

    frame.to_csv(artifacts_dir / well_tags_name, index=False)
    summary.to_csv(artifacts_dir / summary_name, index=False)
    rules.to_csv(artifacts_dir / rules_name, index=False)
    (artifacts_dir / metrics_name).write_text(json.dumps(metrics, indent=2) + "\n")

    return {
        "well_tags": frame,
        "condition_summary": summary,
        "candidate_rules": rules,
        "metrics": metrics,
    }


def resolve_root_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-well router diagnostics.")
    parser.add_argument("--well-metrics", default=None, help="Source exp005-style well_metrics.csv")
    parser.add_argument("--exp003-delta", default=None, help="Optional exp003 well delta CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()

    default_well_metrics = config_get(config, "diagnostic.source_artifacts.exp005_well_metrics", "")
    default_exp003_delta = config_get(config, "diagnostic.source_artifacts.exp003_well_delta", "")
    well_metrics_path = resolve_root_path(paths.root, args.well_metrics or default_well_metrics)
    exp003_delta_path = resolve_root_path(paths.root, args.exp003_delta or default_exp003_delta)

    if well_metrics_path is None or not well_metrics_path.exists():
        raise FileNotFoundError(f"well metrics source not found: {well_metrics_path}")
    well_metrics = pd.read_csv(well_metrics_path)
    exp003_delta = None
    if exp003_delta_path is not None and exp003_delta_path.exists():
        exp003_delta = pd.read_csv(exp003_delta_path)

    outputs = write_router_diagnostic_artifacts(
        well_metrics,
        config,
        paths.artifacts_dir,
        exp003_delta=exp003_delta,
    )
    rules = outputs["candidate_rules"]
    print(rules.to_string(index=False))
    print("Router diagnostic metrics:", paths.artifacts_dir / "router_diagnostic_metrics.json")


if __name__ == "__main__":
    main()
