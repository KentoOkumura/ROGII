from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hidden_branch_surrogate_audit import (
    add_generated_features,
    branch_diff_rows,
    bucket_codes,
    build_controls,
    candidate_metric_rows,
    choose_train_indices,
    file_by_well,
    generate_exp026_anchor_predictions,
    get_nested,
    load_features,
    load_local_config,
    load_or_build_metadata,
    resolve_feature_path,
    segment_rows,
    split_systems,
    well_metric_rows,
)
from meta_stack_audit import build_feature_matrix
from settings import ExperimentPaths
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GateSpec:
    name: str
    base_column: str
    candidate_column: str
    estimator: str
    params: dict[str, Any]
    target: str
    max_weight: float
    min_abs_candidate_delta: float
    win_margin: float
    max_train_rows: int | None
    max_train_rows_per_bucket: int | None
    seed: int
    features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit clipped PF/Beam gate-only candidates.")
    parser.add_argument("--features", default=None, help="Path to exp029 feature CSV")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke well limit")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Override gate row caps")
    parser.add_argument(
        "--base-estimator",
        choices=["LGBMRegressor", "HistGradientBoostingRegressor"],
        default=None,
        help="Override exp026 anchor estimator for local smoke checks.",
    )
    return parser.parse_args()


def safe_name(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def fixed_gate_prediction(
    frame: pd.DataFrame,
    *,
    base_column: str,
    candidate_column: str,
    weight: float | np.ndarray,
) -> np.ndarray:
    base = frame[base_column].to_numpy(dtype=float)
    candidate = frame[candidate_column].to_numpy(dtype=float)
    w = np.asarray(weight, dtype=float)
    return base + w * (candidate - base)


def gate_target(
    frame: pd.DataFrame,
    *,
    base_column: str,
    candidate_column: str,
    target: str,
    max_weight: float,
    min_abs_candidate_delta: float,
    win_margin: float,
) -> np.ndarray:
    y_true = frame["target_tvt"].to_numpy(dtype=float)
    base = frame[base_column].to_numpy(dtype=float)
    candidate = frame[candidate_column].to_numpy(dtype=float)
    delta = candidate - base
    active = np.abs(delta) >= float(min_abs_candidate_delta)

    if target == "optimal_weight":
        weights = np.zeros(len(frame), dtype=float)
        weights[active] = (y_true[active] - base[active]) / delta[active]
        return np.clip(weights, 0.0, float(max_weight))

    if target == "candidate_wins":
        base_abs = np.abs(base - y_true)
        candidate_abs = np.abs(candidate - y_true)
        wins = (candidate_abs + float(win_margin) < base_abs) & active
        return np.where(wins, float(max_weight), 0.0).astype(float)

    raise ValueError(f"unsupported gate target: {target}")


def make_gate_estimator(spec: GateSpec, split: int) -> Pipeline:
    seed = spec.seed + split * 1009
    if spec.estimator == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(**spec.params)),
            ]
        )
    if spec.estimator == "hist_gradient_boosting":
        params = {
            "loss": "squared_error",
            "learning_rate": 0.04,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 200,
            "l2_regularization": 5.0,
            "early_stopping": True,
            "validation_fraction": 0.12,
            "n_iter_no_change": 10,
            "random_state": seed,
        }
        params.update(spec.params)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(**params)),
            ]
        )
    raise ValueError(f"unsupported gate estimator: {spec.estimator}")


def gate_specs(config: dict[str, Any], max_train_rows_override: int | None) -> list[GateSpec]:
    default_features = [str(value) for value in get_nested(config, "model.features", [])]
    specs: list[GateSpec] = []
    for item in get_nested(config, "audit.gate_models", []):
        features = item.get("features", "*")
        if features == "*":
            feature_columns = default_features
        else:
            feature_columns = [str(value) for value in features]
        max_train_rows = item.get("max_train_rows")
        if max_train_rows_override is not None:
            max_train_rows = max_train_rows_override
        specs.append(
            GateSpec(
                name=str(item["name"]),
                base_column=str(
                    item.get("base_column", get_nested(config, "model.base_prediction"))
                ),
                candidate_column=str(item["candidate_column"]),
                estimator=str(item.get("estimator", "ridge")),
                params=dict(item.get("params") or {}),
                target=str(item.get("target", "optimal_weight")),
                max_weight=float(item.get("max_weight", 0.2)),
                min_abs_candidate_delta=float(item.get("min_abs_candidate_delta", 0.0)),
                win_margin=float(item.get("win_margin", 0.0)),
                max_train_rows=None if max_train_rows is None else int(max_train_rows),
                max_train_rows_per_bucket=(
                    None
                    if item.get("max_train_rows_per_bucket") is None
                    else int(item["max_train_rows_per_bucket"])
                ),
                seed=int(item.get("seed", get_nested(config, "validation.seed", 42))),
                features=feature_columns,
            )
        )
    return specs


def build_fixed_gate_predictions(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    stats: list[dict[str, Any]] = []
    for item in get_nested(config, "audit.fixed_gates", []):
        name = str(item["name"])
        weight = float(item["weight"])
        pred = fixed_gate_prediction(
            frame,
            base_column=str(item["base_column"]),
            candidate_column=str(item["candidate_column"]),
            weight=weight,
        )
        predictions[name] = pred
        stats.append(
            {
                "candidate": name,
                "gate_type": "fixed",
                "base_column": str(item["base_column"]),
                "candidate_column": str(item["candidate_column"]),
                "target": "fixed",
                "weight_min": weight,
                "weight_mean": weight,
                "weight_max": weight,
                "max_weight": weight,
                "active_rows": int(len(frame)),
            }
        )
    return predictions, stats


def cross_fit_gate_predictions(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    max_train_rows_override: int | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    stats: list[dict[str, Any]] = []
    for spec in gate_specs(config, max_train_rows_override):
        x_matrix = build_feature_matrix(frame, spec.features)
        target = gate_target(
            frame,
            base_column=spec.base_column,
            candidate_column=spec.candidate_column,
            target=spec.target,
            max_weight=spec.max_weight,
            min_abs_candidate_delta=spec.min_abs_candidate_delta,
            win_margin=spec.win_margin,
        )
        weights = np.full(len(frame), np.nan, dtype=float)
        train_rows_by_split: dict[int, int] = {}
        for split in sorted(int(value) for value in np.unique(split_codes)):
            valid_mask = split_codes == split
            train_idx = choose_train_indices(
                ~valid_mask,
                bucket_code_values,
                max_rows=spec.max_train_rows,
                max_rows_per_bucket=spec.max_train_rows_per_bucket,
                seed=spec.seed + split * 7919,
            )
            valid_idx = np.flatnonzero(valid_mask)
            estimator = make_gate_estimator(spec, split)
            estimator.fit(x_matrix[train_idx], target[train_idx])
            weights[valid_idx] = estimator.predict(x_matrix[valid_idx])
            train_rows_by_split[split] = int(len(train_idx))

        if not np.isfinite(weights).all():
            raise ValueError(f"{spec.name}: non-finite gate weights")
        weights = np.clip(weights, 0.0, spec.max_weight)
        candidate_delta = (
            frame[spec.candidate_column].to_numpy(dtype=float)
            - frame[spec.base_column].to_numpy(dtype=float)
        )
        weights[np.abs(candidate_delta) < spec.min_abs_candidate_delta] = 0.0
        predictions[spec.name] = fixed_gate_prediction(
            frame,
            base_column=spec.base_column,
            candidate_column=spec.candidate_column,
            weight=weights,
        )
        stats.append(
            {
                "candidate": spec.name,
                "gate_type": "learned",
                "base_column": spec.base_column,
                "candidate_column": spec.candidate_column,
                "target": spec.target,
                "weight_min": round(float(np.min(weights)), 8),
                "weight_mean": round(float(np.mean(weights)), 8),
                "weight_max": round(float(np.max(weights)), 8),
                "max_weight": spec.max_weight,
                "active_rows": int(np.count_nonzero(weights > 1e-9)),
                "train_rows_by_split": json.dumps(train_rows_by_split, sort_keys=True),
            }
        )
        print(
            json.dumps(
                {
                    "gate_model": spec.name,
                    "target": spec.target,
                    "weight_mean": round(float(np.mean(weights)), 8),
                    "weight_max": round(float(np.max(weights)), 8),
                    "train_rows_by_split": train_rows_by_split,
                },
                sort_keys=True,
            )
        )
    return predictions, stats


def build_gate_predictions(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    max_train_rows_override: int | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = build_controls(frame, config)
    fixed_predictions, fixed_stats = build_fixed_gate_predictions(frame, config)
    learned_predictions, learned_stats = cross_fit_gate_predictions(
        frame=frame,
        config=config,
        split_codes=split_codes,
        bucket_code_values=bucket_code_values,
        max_train_rows_override=max_train_rows_override,
    )
    predictions.update(fixed_predictions)
    predictions.update(learned_predictions)
    return predictions, fixed_stats + learned_stats


def run_one_audit(
    *,
    audit: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    paths: ExperimentPaths,
    split_codes: np.ndarray,
    split_labels: list[Any],
    bucket_code_values: np.ndarray,
    bucket_labels: list[str],
    max_wells: int | None,
    max_train_rows_override: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp026_pred, source_rows = generate_exp026_anchor_predictions(
        audit=audit,
        frame=frame,
        config=config,
        paths=paths,
        split_codes=split_codes,
        max_wells=max_wells,
    )
    audit_frame = add_generated_features(frame, exp026_pred)
    predictions, gate_stats = build_gate_predictions(
        frame=audit_frame,
        config=config,
        split_codes=split_codes,
        bucket_code_values=bucket_code_values,
        max_train_rows_override=max_train_rows_override,
    )

    reference_name = str(get_nested(config, "audit.reference_control"))
    if reference_name not in predictions:
        raise ValueError(f"reference candidate is missing: {reference_name}")
    reference_names = [
        str(value) for value in get_nested(config, "audit.primary_reference_controls")
    ]
    diff_references = sorted(
        set(reference_names + [reference_name, "visible_train_oracle_surrogate"])
    )

    metrics = pd.DataFrame(
        candidate_metric_rows(
            audit=audit,
            frame=audit_frame,
            predictions=predictions,
            reference_names=reference_names,
        )
    )
    segments = pd.DataFrame(
        segment_rows(
            audit=audit,
            frame=audit_frame,
            predictions=predictions,
            reference=predictions[reference_name],
            split_codes=split_codes,
            split_labels=split_labels,
            bucket_code_values=bucket_code_values,
            bucket_labels=bucket_labels,
        )
    )
    diffs = pd.DataFrame(
        branch_diff_rows(
            audit=audit,
            frame=audit_frame,
            predictions=predictions,
            references=diff_references,
        )
    )
    wells = well_metric_rows(
        audit=audit,
        frame=audit_frame,
        predictions=predictions,
        reference=predictions[reference_name],
    )
    source = pd.DataFrame(source_rows)
    gate_frame = pd.DataFrame(gate_stats)
    gate_frame.insert(0, "audit", audit)
    return metrics, segments, diffs, wells, source, gate_frame


def run_audit(
    paths: ExperimentPaths,
    config: dict[str, Any],
    feature_path: Path,
    output_dir: Path | None = None,
    *,
    max_wells: int | None = None,
    max_train_rows_override: int | None = None,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_dir = output_dir or paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_wells = set(file_by_well(paths, max_wells)) if max_wells is not None else None
    frame, loaded_columns = load_features(feature_path, config, allowed_wells=allowed_wells)
    metadata = load_or_build_metadata(paths, config, max_wells=max_wells)
    merge_columns = [
        "well_id",
        "groupkfold_fold",
        "stratified_groupkfold_fold",
        "azimuth_bin",
        "tvt_bin",
        "spatial_bin",
        "eval_length_bin",
        "gr_bin",
        "strat_label",
    ]
    frame = frame.merge(
        metadata[[column for column in merge_columns if column in metadata.columns]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if frame["stratified_groupkfold_fold"].isna().any():
        raise ValueError("Missing stratified fold metadata for at least one feature row")

    buckets = list(get_nested(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")
    bucket_code_values = bucket_codes(frame["eval_step"].to_numpy(dtype=float), buckets)
    bucket_labels = [str(bucket["name"]) for bucket in buckets]

    metric_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []
    diff_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    split_map = split_systems(frame, metadata, config)
    for audit_name, (codes, labels) in split_map.items():
        metrics, segments, diffs, wells, source, gate_stats = run_one_audit(
            audit=audit_name,
            frame=frame,
            config=config,
            paths=paths,
            split_codes=codes,
            split_labels=labels,
            bucket_code_values=bucket_code_values,
            bucket_labels=bucket_labels,
            max_wells=max_wells,
            max_train_rows_override=max_train_rows_override,
        )
        metric_frames.append(metrics)
        segment_frames.append(segments)
        diff_frames.append(diffs)
        well_frames.append(wells)
        source_frames.append(source)
        gate_frames.append(gate_stats)

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["audit", "rmse"])
    segments = pd.concat(segment_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "segment_type", "segment"]
    )
    diffs = pd.concat(diff_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "reference"]
    )
    wells = pd.concat(well_frames, ignore_index=True).sort_values(["audit", "candidate", "well_id"])
    source_summary = pd.concat(source_frames, ignore_index=True)
    gate_stats = pd.concat(gate_frames, ignore_index=True).sort_values(["audit", "candidate"])

    best_by_audit = (
        metrics[~metrics["candidate"].eq("visible_train_oracle_surrogate")]
        .sort_values(["audit", "rmse"])
        .groupby("audit", sort=True)
        .head(1)
        .to_dict(orient="records")
    )
    stopped_direct = {
        str(item["name"]): {
            "known_public_lb": item.get("known_public_lb"),
            "public_lb_delta_vs_exp027": item.get("public_lb_delta_vs_exp027"),
        }
        for item in get_nested(config, "audit.stopped_direct_branches", [])
    }
    summary = {
        "experiment": "exp047_public_pf_beam_gate_only_audit",
        "status": (
            "completed"
            if max_wells is None and output_dir.resolve() == paths.artifacts_dir.resolve()
            else "smoke_completed"
        ),
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "feature_file": feature_path.as_posix(),
        "loaded_columns": loaded_columns,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "max_wells": max_wells,
        "metric": "rmse",
        "reference_control": get_nested(config, "audit.reference_control"),
        "split_systems": list(split_map),
        "best_by_audit": best_by_audit,
        "stopped_direct_branch_outcomes": stopped_direct,
        "notes": (
            "Gate-only audit: PF/Beam predictions may only move the exp026 anchor by a clipped "
            "weight in base + w * (candidate - base). No free TVT residual branch is trained."
        ),
    }

    metrics.to_csv(output_dir / "public_pf_beam_gate_only_metrics.csv", index=False)
    segments.to_csv(output_dir / "public_pf_beam_gate_only_segment_metrics.csv", index=False)
    diffs.to_csv(output_dir / "public_pf_beam_gate_only_diff_metrics.csv", index=False)
    wells.to_csv(output_dir / "public_pf_beam_gate_only_well_metrics.csv", index=False)
    gate_stats.to_csv(output_dir / "public_pf_beam_gate_only_gate_stats.csv", index=False)
    source_summary.to_csv(
        output_dir / "public_pf_beam_gate_only_exp026_source_summary.csv",
        index=False,
    )
    metadata.to_csv(output_dir / "public_pf_beam_gate_only_well_metadata.csv", index=False)
    with (output_dir / "public_pf_beam_gate_only_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")
    if output_dir.resolve() == paths.artifacts_dir.resolve():
        with paths.metrics_path.open("w") as fp:
            json.dump(summary, fp, indent=2)
            fp.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_local_config()
    if args.base_estimator is not None:
        config.setdefault("model", {}).setdefault("drift_model", {})["estimator"] = (
            args.base_estimator
        )
    feature_path = resolve_feature_path(
        paths,
        args.features or get_nested(config, "data.feature_path"),
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = paths.root / output_dir
    summary = run_audit(
        paths,
        config,
        feature_path,
        output_dir=output_dir,
        max_wells=args.max_wells,
        max_train_rows_override=args.max_train_rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
