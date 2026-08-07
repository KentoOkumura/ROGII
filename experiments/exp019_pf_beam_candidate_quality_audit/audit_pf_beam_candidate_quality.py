from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from settings import ExperimentPaths

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
SCALE_NAMES = ("s3", "s5", "s8", "s12")
RAW_USECOLS = [
    "variant",
    "fold",
    "well_id",
    "row_index",
    "eval_step",
    "eval_row_count",
    "last_anchor",
    "y_true",
    "y_pred",
]


@dataclass
class MetricBucket:
    sse: float = 0.0
    n: int = 0
    wells: set[str] = field(default_factory=set)

    def add(self, pred: np.ndarray, true: np.ndarray, well_id: str) -> None:
        mask = np.isfinite(pred) & np.isfinite(true)
        if not bool(mask.any()):
            return
        diff = pred[mask] - true[mask]
        self.sse += float(np.square(diff).sum())
        self.n += int(mask.sum())
        self.wells.add(well_id)

    @property
    def rmse(self) -> float:
        return rmse_from_sse(self.sse, self.n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exp015 PF/beam candidate quality.")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke limit")
    args, _ = parser.parse_known_args()
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def resolve_existing_path(
    path_value: str | Path,
    *,
    required_name: str | None = None,
    preferred_substring: str | None = None,
) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    if KAGGLE_INPUT_ROOT.exists():
        if required_name:
            matches = sorted(KAGGLE_INPUT_ROOT.rglob(required_name))
        else:
            matches = sorted(KAGGLE_INPUT_ROOT.rglob(path.name))
        if preferred_substring:
            preferred = [item for item in matches if preferred_substring in str(item)]
            if preferred:
                return preferred[0]
        if matches:
            return matches[0]
    return path


def rmse_from_sse(sse: float, n_rows: int | float) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(max(0.0, float(sse)) / float(n_rows))


def load_exp015_baseline(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("exp015_baseline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import exp015 baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_raw_oof(config: dict[str, Any]) -> pd.DataFrame:
    raw_path = resolve_existing_path(
        str(get_nested(config, "audit.raw_oof_path")),
        required_name="row_oof_predictions.csv",
        preferred_substring="exp013-model-diversity-or-postprocess-train",
    )
    raw_variant = str(get_nested(config, "audit.raw_oof_variant", "lightgbm_no_gr"))
    chunk_rows = int(get_nested(config, "audit.chunk_rows", 250000))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_path, usecols=RAW_USECOLS, chunksize=chunk_rows):
        chunk = chunk[chunk["variant"] == raw_variant]
        if chunk.empty:
            continue
        frames.append(chunk.drop(columns=["variant"]))
    if not frames:
        raise ValueError(f"No raw OOF rows for variant={raw_variant!r} in {raw_path}")
    frame = pd.concat(frames, ignore_index=True)
    frame["row_index"] = frame["row_index"].astype(int)
    return frame.set_index(["well_id", "row_index"], verify_integrity=True).sort_index()


def load_exp015_model_well_metrics(config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_existing_path(
        str(get_nested(config, "audit.exp015_well_metrics_path")),
        required_name="well_metrics.csv",
        preferred_substring="exp015-public-pf-beam-scale-selector-features-train",
    )
    frame = pd.read_csv(path)
    keep = frame[frame["variant"].isin(["control_lightgbm_no_gr", "pf_beam_no_gr"])].copy()
    pivot = keep.pivot_table(
        index=["well_id", "fold"],
        columns="variant",
        values="drift_model_rmse",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "control_lightgbm_no_gr": "exp015_control_model_rmse",
            "pf_beam_no_gr": "exp015_pf_feature_model_rmse",
        }
    )
    pivot["exp015_pf_feature_delta_vs_control"] = (
        pivot["exp015_pf_feature_model_rmse"] - pivot["exp015_control_model_rmse"]
    )
    return pivot


def distance_bucket_labels(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    labels = np.full(eval_step.shape, str(buckets[-1]["name"]), dtype=object)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        labels[mask] = str(bucket["name"])
        previous_max = max_step
    return labels


def add_group_metrics(
    metrics: dict[tuple[str, str, str], MetricBucket],
    *,
    segment: str,
    group: str,
    candidates: dict[str, np.ndarray],
    y_true: np.ndarray,
    mask: np.ndarray,
    well_id: str,
) -> None:
    if not bool(mask.any()):
        return
    for name, pred in candidates.items():
        metrics[(segment, group, name)].add(pred[mask], y_true[mask], well_id)


def metric_row(
    *,
    segment: str,
    group: str,
    candidate: str,
    bucket: MetricBucket,
    raw_rmse: float,
) -> dict[str, Any]:
    value = bucket.rmse
    return {
        "segment": segment,
        "group": group,
        "candidate": candidate,
        "rmse": round(value, 6),
        "raw_rmse": round(raw_rmse, 6),
        "delta_vs_raw": round(value - raw_rmse, 6),
        "rows": bucket.n,
        "wells": len(bucket.wells),
    }


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_yaml(Path(__file__).with_name("config.yaml"))
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exp015_module = load_exp015_baseline(
        resolve_existing_path(str(get_nested(config, "audit.exp015_baseline_py")))
    )
    pf_config = load_yaml(
        resolve_existing_path(str(get_nested(config, "audit.exp015_config_path")))
    )
    set_nested(pf_config, "model.pf_beam.enabled", True)

    configured_train_dir = Path(str(get_nested(config, "audit.train_dir", "data/raw/train")))
    train_dir = configured_train_dir if configured_train_dir.exists() else paths.train_data_dir
    files = sorted(train_dir.glob(f"*{exp015_module.HORIZONTAL_SUFFIX}"))
    if args.max_wells is not None:
        files = files[: args.max_wells]
    if not files:
        raise ValueError(f"No train horizontal well CSVs found in {train_dir}")

    raw_oof = load_raw_oof(config)
    exp015_well_metrics = load_exp015_model_well_metrics(config)
    exp015_well_by_id = {
        str(row.well_id): row for row in exp015_well_metrics.itertuples(index=False)
    }

    buckets = list(get_nested(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")
    high_gr_missing = float(get_nested(config, "audit.high_gr_missing_rate", 0.50))
    long_eval_rows = int(get_nested(config, "audit.long_eval_rows", 5700))
    high_z_span = float(get_nested(config, "audit.high_z_span", 120.0))
    steep_trajectory_abs_dz_dmd = float(
        get_nested(config, "audit.steep_trajectory_abs_dz_dmd", 0.035)
    )
    high_confidence = float(get_nested(config, "audit.high_confidence", 0.08))
    low_confidence = float(get_nested(config, "audit.low_confidence", 0.02))

    metrics: dict[tuple[str, str, str], MetricBucket] = defaultdict(MetricBucket)
    scale_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for file_idx, path in enumerate(files, start=1):
        well_id = exp015_module.well_id_from_path(path)
        df = pd.read_csv(path)
        typewell_df = exp015_module.read_typewell_for_horizontal_path(path)
        try:
            frame = exp015_module.build_drift_feature_frame(
                df,
                pf_config,
                include_target=True,
                typewell_df=typewell_df,
            )
        except Exception as exc:
            skipped.append(f"{well_id}: feature build failed: {exc}")
            continue
        if frame.target_residual is None or frame.eval_indices.size == 0:
            skipped.append(f"{well_id}: no evaluation rows")
            continue

        try:
            raw_part = raw_oof.loc[(well_id, frame.eval_indices.astype(int)), :]
        except KeyError:
            skipped.append(f"{well_id}: missing raw OOF rows")
            continue
        raw_part = raw_part.reset_index()
        y_true = np.asarray(frame.baseline_prediction + frame.target_residual, dtype=float)
        if len(raw_part) != y_true.size:
            skipped.append(f"{well_id}: raw OOF row count mismatch")
            continue

        features = frame.features
        last_known_tvt = features["last_known_tvt"].to_numpy(dtype=float)
        anchor = np.asarray(frame.baseline_prediction, dtype=float)
        raw_pred = raw_part["y_pred"].to_numpy(dtype=float)
        recent_pred = last_known_tvt + features["pf_beam_recent_delta_tvt"].to_numpy(dtype=float)
        mean_pred = last_known_tvt + features["pf_beam_mean_delta_tvt"].to_numpy(dtype=float)
        best_pred = last_known_tvt + features["pf_beam_best_delta_tvt"].to_numpy(dtype=float)
        hold_weight = features["pf_beam_hold_weight"].to_numpy(dtype=float)

        candidates: dict[str, np.ndarray] = {
            "raw_lightgbm_no_gr": raw_pred,
            "last_anchor": anchor,
            "recent_linear": recent_pred,
            "pf_mean": mean_pred,
            "pf_best": best_pred,
            "pf_hold_mean_blend": hold_weight * anchor + (1.0 - hold_weight) * mean_pred,
            "pf_hold_best_blend": hold_weight * anchor + (1.0 - hold_weight) * best_pred,
        }
        for scale in SCALE_NAMES:
            candidates[f"pf_{scale}"] = features[f"pf_beam_{scale}_pred_tvt"].to_numpy(
                dtype=float
            )

        eval_step = features["eval_step"].to_numpy(dtype=float)
        bucket_labels = distance_bucket_labels(eval_step, buckets)
        eval_gr = df.loc[frame.eval_indices, "GR"].to_numpy(dtype=float)
        eval_z = df.loc[frame.eval_indices, "Z"].to_numpy(dtype=float)
        prefix_gr = df.loc[: frame.last_known_index, "GR"].to_numpy(dtype=float)
        prefix_gr_missing_rate = float(np.isnan(prefix_gr).mean()) if prefix_gr.size else 1.0
        eval_gr_missing_rate = float(np.isnan(eval_gr).mean()) if eval_gr.size else 1.0
        z_span = float(np.nanmax(eval_z) - np.nanmin(eval_z)) if eval_z.size else 0.0
        trajectory_abs_dz_dmd = float(
            np.nanmean(np.abs(features["anchor_dz_dmd"].to_numpy(dtype=float)))
        )
        best_scale_values = features["pf_beam_best_scale"].to_numpy(dtype=float)
        confidence = features["pf_beam_confidence"].to_numpy(dtype=float)
        availability = features["pf_beam_available"].to_numpy(dtype=float)

        add_group_metrics(
            metrics,
            segment="all",
            group="all",
            candidates=candidates,
            y_true=y_true,
            mask=np.ones(y_true.shape, dtype=bool),
            well_id=well_id,
        )
        for bucket in buckets:
            label = str(bucket["name"])
            add_group_metrics(
                metrics,
                segment="distance_bucket",
                group=label,
                candidates=candidates,
                y_true=y_true,
                mask=bucket_labels == label,
                well_id=well_id,
            )
        well_group_masks = {
            "high_gr_missing": eval_gr_missing_rate >= high_gr_missing,
            "long_eval": y_true.size >= long_eval_rows,
            "high_z_span": z_span >= high_z_span,
            "steep_trajectory": trajectory_abs_dz_dmd >= steep_trajectory_abs_dz_dmd,
            "pf_available": float(np.nanmean(availability)) > 0.5,
        }
        for group, include in well_group_masks.items():
            if include:
                add_group_metrics(
                    metrics,
                    segment="well_condition",
                    group=group,
                    candidates=candidates,
                    y_true=y_true,
                    mask=np.ones(y_true.shape, dtype=bool),
                    well_id=well_id,
                )
        for scale in sorted(np.unique(best_scale_values[np.isfinite(best_scale_values)])):
            add_group_metrics(
                metrics,
                segment="best_scale",
                group=f"scale_{scale:g}",
                candidates=candidates,
                y_true=y_true,
                mask=best_scale_values == scale,
                well_id=well_id,
            )
        add_group_metrics(
            metrics,
            segment="confidence",
            group="low_confidence",
            candidates=candidates,
            y_true=y_true,
            mask=confidence <= low_confidence,
            well_id=well_id,
        )
        add_group_metrics(
            metrics,
            segment="confidence",
            group="high_confidence",
            candidates=candidates,
            y_true=y_true,
            mask=confidence >= high_confidence,
            well_id=well_id,
        )

        well_record: dict[str, Any] = {
            "well_id": well_id,
            "fold": int(raw_part["fold"].iloc[0]),
            "n_eval": int(y_true.size),
            "prefix_gr_missing_rate": round(prefix_gr_missing_rate, 6),
            "eval_gr_missing_rate": round(eval_gr_missing_rate, 6),
            "z_span": round(z_span, 6),
            "trajectory_abs_dz_dmd": round(trajectory_abs_dz_dmd, 6),
            "pf_available_rate": round(float(np.nanmean(availability)), 6),
            "pf_confidence_mean": round(float(np.nanmean(confidence)), 6),
            "pf_hold_weight_mean": round(float(np.nanmean(hold_weight)), 6),
            "pf_range_delta_tvt_mean": round(
                float(np.nanmean(features["pf_beam_range_delta_tvt"])), 6
            ),
            "pf_best_scale_mode": float(pd.Series(best_scale_values).mode().iloc[0]),
        }
        for name, pred in candidates.items():
            well_record[f"{name}_rmse"] = round(
                rmse_from_sse(float(np.square(pred - y_true).sum()), y_true.size), 6
            )
        if well_id in exp015_well_by_id:
            source_row = exp015_well_by_id[well_id]
            well_record["exp015_control_model_rmse"] = round(
                float(source_row.exp015_control_model_rmse), 6
            )
            well_record["exp015_pf_feature_model_rmse"] = round(
                float(source_row.exp015_pf_feature_model_rmse), 6
            )
            well_record["exp015_pf_feature_delta_vs_control"] = round(
                float(source_row.exp015_pf_feature_delta_vs_control), 6
            )
        well_record["pf_best_delta_vs_raw"] = round(
            well_record["pf_best_rmse"] - well_record["raw_lightgbm_no_gr_rmse"], 6
        )
        well_record["pf_hold_best_delta_vs_raw"] = round(
            well_record["pf_hold_best_blend_rmse"] - well_record["raw_lightgbm_no_gr_rmse"],
            6,
        )
        well_rows.append(well_record)

        for scale in SCALE_NAMES:
            pred = candidates[f"pf_{scale}"]
            scale_rows.append(
                {
                    "well_id": well_id,
                    "scale": scale,
                    "rmse": round(
                        rmse_from_sse(float(np.square(pred - y_true).sum()), y_true.size), 6
                    ),
                    "score_mean": round(float(np.nanmean(features[f"pf_beam_{scale}_score"])), 6),
                    "dtw_cost_mean": round(
                        float(np.nanmean(features[f"pf_beam_{scale}_dtw_cost"])), 6
                    ),
                    "gr_abs_error_mean": round(
                        float(np.nanmean(features[f"pf_beam_{scale}_gr_abs_error"])), 6
                    ),
                    "minus_recent_mean": round(
                        float(np.nanmean(features[f"pf_beam_{scale}_minus_recent"])), 6
                    ),
                    "selected_rate": round(
                        float(np.mean(best_scale_values == float(scale[1:]))), 6
                    ),
                }
            )

        if file_idx % 100 == 0:
            print(f"processed {file_idx}/{len(files)} wells")

    raw_keys = {
        (segment, group): bucket.rmse
        for (segment, group, candidate), bucket in metrics.items()
        if candidate == "raw_lightgbm_no_gr"
    }
    metric_rows = [
        metric_row(
            segment=segment,
            group=group,
            candidate=candidate,
            bucket=bucket,
            raw_rmse=raw_keys[(segment, group)],
        )
        for (segment, group, candidate), bucket in metrics.items()
    ]
    metric_rows = sorted(metric_rows, key=lambda row: (row["segment"], row["group"], row["rmse"]))
    metric_df = pd.DataFrame(metric_rows)
    well_df = pd.DataFrame(well_rows)
    scale_df = pd.DataFrame(scale_rows)
    if well_df.empty:
        raise ValueError("No wells were audited")

    top_hurt_help = pd.concat(
        [
            well_df.sort_values("exp015_pf_feature_delta_vs_control", ascending=False)
            .head(25)
            .assign(list_name="pf_feature_model_top_hurt"),
            well_df.sort_values("exp015_pf_feature_delta_vs_control", ascending=True)
            .head(25)
            .assign(list_name="pf_feature_model_top_help"),
            well_df.sort_values("pf_best_delta_vs_raw", ascending=False)
            .head(25)
            .assign(list_name="pf_best_direct_top_hurt"),
            well_df.sort_values("pf_best_delta_vs_raw", ascending=True)
            .head(25)
            .assign(list_name="pf_best_direct_top_help"),
        ],
        ignore_index=True,
    )

    all_metrics = metric_df.query("segment == 'all' and group == 'all'")
    raw_cv = float(all_metrics[all_metrics["candidate"] == "raw_lightgbm_no_gr"]["rmse"].iloc[0])
    best_direct = all_metrics.sort_values("rmse").iloc[0]
    feature_delta = float(well_df["exp015_pf_feature_delta_vs_control"].mean())
    summary = {
        "experiment": "exp019_pf_beam_candidate_quality_audit",
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": "exp015_public_pf_beam_scale_selector_features",
        "rows": int(all_metrics["rows"].max()),
        "wells": int(well_df["well_id"].nunique()),
        "skipped": skipped,
        "raw_lightgbm_no_gr_cv": round(raw_cv, 6),
        "best_direct_candidate": str(best_direct["candidate"]),
        "best_direct_candidate_cv": float(best_direct["rmse"]),
        "best_direct_delta_vs_raw": float(best_direct["delta_vs_raw"]),
        "exp015_pf_feature_model_mean_well_delta_vs_control": round(feature_delta, 6),
        "pf_feature_model_supported": feature_delta < 0.0,
        "direct_pf_supported": float(best_direct["delta_vs_raw"]) < 0.0,
        "metric": "rmse",
        "notes": (
            "PF/beam candidate quality is diagnostic only; do not retrain or submit unless "
            "direct candidates beat raw in stable groups."
        ),
        "artifacts": [
            "pf_beam_candidate_metrics.csv",
            "pf_beam_well_deltas.csv",
            "pf_beam_scale_diagnostics.csv",
            "pf_beam_top_hurt_help.csv",
            "pf_beam_candidate_quality_summary.json",
        ],
    }

    metric_df.to_csv(output_dir / "pf_beam_candidate_metrics.csv", index=False)
    well_df.to_csv(output_dir / "pf_beam_well_deltas.csv", index=False)
    scale_df.to_csv(output_dir / "pf_beam_scale_diagnostics.csv", index=False)
    top_hurt_help.to_csv(output_dir / "pf_beam_top_hurt_help.csv", index=False)
    with (output_dir / "pf_beam_candidate_quality_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    with Path(__file__).with_name("metrics.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
