from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from settings import ExperimentPaths

BASE_USECOLS = {
    "id",
    "well_id",
    "fold",
    "row_idx",
    "eval_step",
    "distance_bucket",
    "target_tvt",
    "last_anchor_tvt",
    "pf_pred",
    "pf_selected_scale_pred",
    "pf_scale_3",
    "pf_scale_5",
    "pf_scale_8",
    "pf_scale_12",
    "pf_seed_mean",
    "pf_seed_std",
    "beam_pred",
    "beam_spread",
    "abs_pf_beam_diff",
    "pf_lik_gap_best_second",
    "pf_weight_entropy",
    "gr_eval_availability",
    "selector_n_eval",
    "selector_z_span",
}


@dataclass(frozen=True)
class CandidateStats:
    name: str
    method: str
    params: dict[str, Any]
    selectable: bool
    rows: int
    sse: float
    fold_sse: np.ndarray
    fold_n: np.ndarray
    well_fold_sse: np.ndarray
    well_fold_n: np.ndarray
    bucket_sse: np.ndarray
    bucket_n: np.ndarray
    fold_bucket_sse: np.ndarray
    fold_bucket_n: np.ndarray
    well_fold_bucket_sse: np.ndarray
    well_fold_bucket_n: np.ndarray

    @property
    def rmse(self) -> float:
        return rmse_from_sse(self.sse, self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit public sel15 PF candidate selection.")
    parser.add_argument("--features", default=None, help="Path to exp029 feature CSV")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    return parser.parse_args()


def load_local_config() -> dict[str, Any]:
    with Path(__file__).with_name("config.yaml").open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def rmse_from_sse(sse: float, n_rows: int | float) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(max(0.0, float(sse)) / float(n_rows))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % n_folds


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def columns_for_candidate(candidate: dict[str, Any]) -> set[str]:
    method = str(candidate.get("method", ""))
    columns: set[str] = set()
    if method == "column":
        columns.add(str(candidate["column"]))
    elif method == "blend":
        columns.update(str(key) for key in dict(candidate.get("weights") or {}).keys())
    elif method == "confidence_fallback":
        columns.add(str(candidate["base"]))
        columns.add(str(candidate["fallback"]))
        thresholds = dict(candidate.get("thresholds") or {})
        if "max_pf_seed_std" in thresholds:
            columns.add("pf_seed_std")
        if "max_abs_pf_beam_diff" in thresholds:
            columns.add("abs_pf_beam_diff")
        if "max_beam_spread" in thresholds:
            columns.add("beam_spread")
        if "min_pf_lik_gap_best_second" in thresholds:
            columns.add("pf_lik_gap_best_second")
        if "max_pf_weight_entropy" in thresholds:
            columns.add("pf_weight_entropy")
        if "min_gr_eval_availability" in thresholds:
            columns.add("gr_eval_availability")
    else:
        raise ValueError(f"unsupported candidate method: {method}")
    return columns


def required_columns(config: dict[str, Any]) -> list[str]:
    columns = set(BASE_USECOLS)
    for candidate in list(get_nested(config, "audit.candidates", [])):
        columns.update(columns_for_candidate(dict(candidate)))
    return sorted(columns)


def load_features(path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    usecols = required_columns(config)
    chunk_rows = int(get_nested(config, "runtime.chunk_rows", 500000))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in usecols, chunksize=chunk_rows):
        frames.append(chunk)
    if not frames:
        raise ValueError(f"No feature rows found in {path}")
    frame = pd.concat(frames, ignore_index=True)
    missing = sorted(set(usecols) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature file is missing required columns: {missing}")
    frame = frame.sort_values(["fold", "well_id", "row_idx"], kind="mergesort")
    return frame.reset_index(drop=True), usecols


def bucket_codes(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        codes[mask] = idx
        previous_max = max_step
    return codes


def column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(column)
    return frame[column].to_numpy(dtype=float)


def candidate_prediction(frame: pd.DataFrame, candidate: dict[str, Any]) -> np.ndarray:
    method = str(candidate["method"])
    if method == "column":
        return column_values(frame, str(candidate["column"])).copy()
    if method == "blend":
        weights = dict(candidate.get("weights") or {})
        total_weight = float(sum(float(value) for value in weights.values()))
        if total_weight <= 0:
            raise ValueError(f"blend candidate {candidate['name']} has non-positive weights")
        output = np.zeros(len(frame), dtype=float)
        for column, weight in weights.items():
            output += column_values(frame, str(column)) * (float(weight) / total_weight)
        return output
    if method == "confidence_fallback":
        base = column_values(frame, str(candidate["base"]))
        fallback = column_values(frame, str(candidate["fallback"]))
        thresholds = dict(candidate.get("thresholds") or {})
        use_base = np.ones(len(frame), dtype=bool)
        if "max_pf_seed_std" in thresholds:
            use_base &= column_values(frame, "pf_seed_std") <= float(thresholds["max_pf_seed_std"])
        if "max_abs_pf_beam_diff" in thresholds:
            use_base &= column_values(frame, "abs_pf_beam_diff") <= float(
                thresholds["max_abs_pf_beam_diff"]
            )
        if "max_beam_spread" in thresholds:
            use_base &= column_values(frame, "beam_spread") <= float(thresholds["max_beam_spread"])
        if "min_pf_lik_gap_best_second" in thresholds:
            use_base &= column_values(frame, "pf_lik_gap_best_second") >= float(
                thresholds["min_pf_lik_gap_best_second"]
            )
        if "max_pf_weight_entropy" in thresholds:
            use_base &= column_values(frame, "pf_weight_entropy") <= float(
                thresholds["max_pf_weight_entropy"]
            )
        if "min_gr_eval_availability" in thresholds:
            use_base &= column_values(frame, "gr_eval_availability") >= float(
                thresholds["min_gr_eval_availability"]
            )
        return np.where(use_base, base, fallback)
    raise ValueError(f"unsupported candidate method: {method}")


def finite_prediction(name: str, pred: np.ndarray) -> np.ndarray:
    if pred.shape[0] == 0:
        raise ValueError(f"{name}: empty prediction")
    if not np.isfinite(pred).all():
        raise ValueError(f"{name}: prediction contains non-finite values")
    return pred


def bincount2d(
    primary_codes: np.ndarray,
    secondary_codes: np.ndarray,
    weights: np.ndarray | None,
    primary_count: int,
    secondary_count: int,
) -> np.ndarray:
    combined = primary_codes.astype(np.int64) * secondary_count + secondary_codes.astype(np.int64)
    counts = np.bincount(
        combined,
        weights=weights,
        minlength=primary_count * secondary_count,
    )
    return counts.reshape(primary_count, secondary_count)


def aggregate_candidate(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    *,
    y_true: np.ndarray,
    fold_codes: np.ndarray,
    fold_count: int,
    well_fold_codes: np.ndarray,
    well_fold_count: int,
    bucket_code_values: np.ndarray,
    bucket_count: int,
) -> tuple[CandidateStats, np.ndarray]:
    name = str(candidate["name"])
    pred = finite_prediction(name, candidate_prediction(frame, candidate))
    diff2 = np.square(pred - y_true)
    stats = CandidateStats(
        name=name,
        method=str(candidate["method"]),
        params={key: value for key, value in candidate.items() if key not in {"name", "method"}},
        selectable=bool(candidate.get("selectable", True)),
        rows=int(len(frame)),
        sse=float(diff2.sum()),
        fold_sse=np.bincount(fold_codes, weights=diff2, minlength=fold_count),
        fold_n=np.bincount(fold_codes, minlength=fold_count).astype(float),
        well_fold_sse=np.bincount(well_fold_codes, weights=diff2, minlength=well_fold_count),
        well_fold_n=np.bincount(well_fold_codes, minlength=well_fold_count).astype(float),
        bucket_sse=np.bincount(bucket_code_values, weights=diff2, minlength=bucket_count),
        bucket_n=np.bincount(bucket_code_values, minlength=bucket_count).astype(float),
        fold_bucket_sse=bincount2d(
            fold_codes,
            bucket_code_values,
            diff2,
            fold_count,
            bucket_count,
        ),
        fold_bucket_n=bincount2d(
            fold_codes,
            bucket_code_values,
            None,
            fold_count,
            bucket_count,
        ),
        well_fold_bucket_sse=bincount2d(
            well_fold_codes,
            bucket_code_values,
            diff2,
            well_fold_count,
            bucket_count,
        ),
        well_fold_bucket_n=bincount2d(
            well_fold_codes,
            bucket_code_values,
            None,
            well_fold_count,
            bucket_count,
        ),
    )
    return stats, pred


def build_selection_audit(
    *,
    name: str,
    stats_by_name: dict[str, CandidateStats],
    selectable_names: list[str],
    control_name: str,
    attr_sse: str,
    attr_n: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control = stats_by_name[control_name]
    control_sse = getattr(control, attr_sse)
    control_n = getattr(control, attr_n)
    split_count = len(control_sse)
    rows: list[dict[str, Any]] = []
    selected_sse = 0.0
    selected_n = 0.0

    for split in range(split_count):
        best_name = control_name
        best_train_rmse = float("inf")
        for candidate_name in selectable_names:
            item = stats_by_name[candidate_name]
            item_sse = getattr(item, attr_sse)
            item_n = getattr(item, attr_n)
            train_sse = float(item.sse - item_sse[split])
            train_n = float(item.rows - item_n[split])
            train_rmse = rmse_from_sse(train_sse, train_n)
            if train_rmse < best_train_rmse:
                best_train_rmse = train_rmse
                best_name = candidate_name
        chosen = stats_by_name[best_name]
        chosen_sse = getattr(chosen, attr_sse)
        chosen_n = getattr(chosen, attr_n)
        valid_sse = float(chosen_sse[split])
        valid_n = float(chosen_n[split])
        selected_sse += valid_sse
        selected_n += valid_n
        control_valid_rmse = rmse_from_sse(float(control_sse[split]), float(control_n[split]))
        selected_valid_rmse = rmse_from_sse(valid_sse, valid_n)
        rows.append(
            {
                "audit": name,
                "split": split,
                "selected_candidate": best_name,
                "train_rmse": round(best_train_rmse, 6),
                "valid_rmse": round(selected_valid_rmse, 6),
                "control_valid_rmse": round(control_valid_rmse, 6),
                "delta_vs_control": round(selected_valid_rmse - control_valid_rmse, 6),
                "rows": int(valid_n),
            }
        )

    summary = {
        "candidate": name,
        "rmse": round(rmse_from_sse(selected_sse, selected_n), 6),
        "rows": int(selected_n),
        "control": control_name,
    }
    return summary, rows


def build_bucket_selection_audit(
    *,
    name: str,
    stats_by_name: dict[str, CandidateStats],
    selectable_names: list[str],
    control_name: str,
    attr_sse: str,
    attr_n: str,
    buckets: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control = stats_by_name[control_name]
    control_sse = getattr(control, attr_sse)
    control_n = getattr(control, attr_n)
    split_count, bucket_count = control_sse.shape
    rows: list[dict[str, Any]] = []
    selected_sse = 0.0
    selected_n = 0.0

    for split in range(split_count):
        for bucket_idx in range(bucket_count):
            best_name = control_name
            best_train_rmse = float("inf")
            for candidate_name in selectable_names:
                item = stats_by_name[candidate_name]
                item_sse = getattr(item, attr_sse)
                item_n = getattr(item, attr_n)
                train_sse = float(item.bucket_sse[bucket_idx] - item_sse[split, bucket_idx])
                train_n = float(item.bucket_n[bucket_idx] - item_n[split, bucket_idx])
                train_rmse = rmse_from_sse(train_sse, train_n)
                if train_rmse < best_train_rmse:
                    best_train_rmse = train_rmse
                    best_name = candidate_name
            chosen = stats_by_name[best_name]
            chosen_sse = getattr(chosen, attr_sse)
            chosen_n = getattr(chosen, attr_n)
            valid_sse = float(chosen_sse[split, bucket_idx])
            valid_n = float(chosen_n[split, bucket_idx])
            selected_sse += valid_sse
            selected_n += valid_n
            control_valid_rmse = rmse_from_sse(
                float(control_sse[split, bucket_idx]),
                float(control_n[split, bucket_idx]),
            )
            selected_valid_rmse = rmse_from_sse(valid_sse, valid_n)
            rows.append(
                {
                    "audit": name,
                    "split": split,
                    "bucket": str(buckets[bucket_idx]["name"]),
                    "selected_candidate": best_name,
                    "train_rmse": round(best_train_rmse, 6),
                    "valid_rmse": round(selected_valid_rmse, 6),
                    "control_valid_rmse": round(control_valid_rmse, 6),
                    "delta_vs_control": round(selected_valid_rmse - control_valid_rmse, 6),
                    "rows": int(valid_n),
                }
            )

    summary = {
        "candidate": name,
        "rmse": round(rmse_from_sse(selected_sse, selected_n), 6),
        "rows": int(selected_n),
        "control": control_name,
    }
    return summary, rows


def build_well_metrics(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    control_name: str,
    y_true: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = frame[["well_id", "fold"]].copy()
    control_diff2 = np.square(predictions[control_name] - y_true)
    control_well = (
        base.assign(diff2=control_diff2)
        .groupby(["well_id", "fold"], sort=False)
        .agg(control_sse=("diff2", "sum"), rows=("diff2", "size"))
        .reset_index()
    )
    for name, pred in predictions.items():
        diff2 = np.square(pred - y_true)
        item = (
            base.assign(diff2=diff2)
            .groupby(["well_id", "fold"], sort=False)
            .agg(sse=("diff2", "sum"), rows=("diff2", "size"))
            .reset_index()
        )
        item = item.merge(control_well, on=["well_id", "fold"], suffixes=("", "_control"))
        item["candidate"] = name
        item["rmse"] = np.sqrt(item["sse"] / item["rows"])
        item["control_rmse"] = np.sqrt(item["control_sse"] / item["rows"])
        item["delta_vs_control"] = item["rmse"] - item["control_rmse"]
        rows.append(
            item[
                [
                    "candidate",
                    "well_id",
                    "fold",
                    "rows",
                    "rmse",
                    "control_rmse",
                    "delta_vs_control",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def run_audit(
    paths: ExperimentPaths,
    config: dict[str, Any],
    feature_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_dir = output_dir or paths.artifacts_dir
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    candidates = [dict(item) for item in list(get_nested(config, "audit.candidates", []))]
    raw_name = str(get_nested(config, "audit.raw_candidate", "public_pf_selector"))
    control_names = [
        str(value) for value in get_nested(config, "audit.control_candidates", [raw_name])
    ]
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    if not buckets or not candidates:
        raise ValueError("audit.distance_buckets and audit.candidates must be non-empty")

    frame, loaded_columns = load_features(feature_path, config)
    y_true = frame["target_tvt"].to_numpy(dtype=float)
    folds = sorted(int(value) for value in frame["fold"].unique())
    fold_map = {fold: idx for idx, fold in enumerate(folds)}
    fold_codes = frame["fold"].map(fold_map).to_numpy(dtype=np.int16)
    well_fold_codes = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_holdout_folds)
    ).to_numpy(dtype=np.int16)
    bucket_code_values = bucket_codes(frame["eval_step"].to_numpy(dtype=float), buckets)

    stats: list[CandidateStats] = []
    predictions: dict[str, np.ndarray] = {}
    for candidate in candidates:
        item, pred = aggregate_candidate(
            frame,
            candidate,
            y_true=y_true,
            fold_codes=fold_codes,
            fold_count=len(folds),
            well_fold_codes=well_fold_codes,
            well_fold_count=well_holdout_folds,
            bucket_code_values=bucket_code_values,
            bucket_count=len(buckets),
        )
        stats.append(item)
        predictions[item.name] = pred

    stats_by_name = {item.name: item for item in stats}
    if raw_name not in stats_by_name:
        raise ValueError(f"raw candidate {raw_name!r} was not evaluated")
    for control_name in control_names:
        if control_name not in stats_by_name:
            raise ValueError(f"control candidate {control_name!r} was not evaluated")

    raw_rmse = stats_by_name[raw_name].rmse
    required_control_rmse = min(stats_by_name[name].rmse for name in control_names)
    selectable_names = [item.name for item in stats if item.selectable]

    metric_rows = []
    for item in stats:
        well_frame = build_well_metrics(
            frame,
            {item.name: predictions[item.name], raw_name: predictions[raw_name]},
            control_name=raw_name,
            y_true=y_true,
        )
        item_wells = well_frame[well_frame["candidate"] == item.name]
        metric_rows.append(
            {
                "candidate": item.name,
                "method": item.method,
                "rmse": round(item.rmse, 6),
                "delta_vs_raw": round(item.rmse - raw_rmse, 6),
                "rows": item.rows,
                "selectable": item.selectable,
                "well_wins_vs_raw": int((item_wells["delta_vs_control"] < 0).sum()),
                "well_losses_vs_raw": int((item_wells["delta_vs_control"] > 0).sum()),
                "params_json": params_json(item.params),
            }
        )
    metric_rows = sorted(metric_rows, key=lambda row: row["rmse"])

    selection_summaries: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    original_summary, original_rows = build_selection_audit(
        name="leave_one_original_fold_out_candidate_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        control_name=raw_name,
        attr_sse="fold_sse",
        attr_n="fold_n",
    )
    well_summary, well_rows = build_selection_audit(
        name="well_hash_holdout_candidate_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        control_name=raw_name,
        attr_sse="well_fold_sse",
        attr_n="well_fold_n",
    )
    original_bucket_summary, original_bucket_rows = build_bucket_selection_audit(
        name="leave_one_original_fold_out_bucket_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        control_name=raw_name,
        attr_sse="fold_bucket_sse",
        attr_n="fold_bucket_n",
        buckets=buckets,
    )
    well_bucket_summary, well_bucket_rows = build_bucket_selection_audit(
        name="well_hash_holdout_bucket_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        control_name=raw_name,
        attr_sse="well_fold_bucket_sse",
        attr_n="well_fold_bucket_n",
        buckets=buckets,
    )
    selection_summaries.extend(
        [original_summary, well_summary, original_bucket_summary, well_bucket_summary]
    )
    selection_rows.extend(original_rows + well_rows + original_bucket_rows + well_bucket_rows)

    bucket_rows: list[dict[str, Any]] = []
    for item in stats:
        for bucket_idx, bucket in enumerate(buckets):
            raw_bucket_rmse = rmse_from_sse(
                stats_by_name[raw_name].bucket_sse[bucket_idx],
                stats_by_name[raw_name].bucket_n[bucket_idx],
            )
            item_bucket_rmse = rmse_from_sse(item.bucket_sse[bucket_idx], item.bucket_n[bucket_idx])
            bucket_rows.append(
                {
                    "candidate": item.name,
                    "method": item.method,
                    "bucket": str(bucket["name"]),
                    "max_step": float(bucket["max_step"]),
                    "rmse": round(item_bucket_rmse, 6),
                    "raw_rmse": round(raw_bucket_rmse, 6),
                    "delta_vs_raw": round(item_bucket_rmse - raw_bucket_rmse, 6),
                    "rows": int(item.bucket_n[bucket_idx]),
                }
            )

    well_metrics = build_well_metrics(frame, predictions, control_name=raw_name, y_true=y_true)
    same_oof_best = metric_rows[0]
    global_supported = (
        original_summary["rmse"] < required_control_rmse
        and well_summary["rmse"] < required_control_rmse
    )
    bucket_supported = (
        original_bucket_summary["rmse"] < required_control_rmse
        and well_bucket_summary["rmse"] < required_control_rmse
    )
    selector_supported = bool(global_supported or bucket_supported)
    selected_clean_cv = min(
        original_summary["rmse"] if global_supported else raw_rmse,
        original_bucket_summary["rmse"] if bucket_supported else raw_rmse,
    )

    summary = {
        "experiment": "exp030_public_sel15_pf_candidate_selector",
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "feature_file": feature_path.as_posix(),
        "loaded_columns": loaded_columns,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "raw_candidate": raw_name,
        "control_candidates": control_names,
        "raw_clean_cv": round(raw_rmse, 6),
        "required_control_cv": round(required_control_rmse, 6),
        "best_same_oof_candidate": same_oof_best["candidate"],
        "best_same_oof_cv": same_oof_best["rmse"],
        "leave_one_original_fold_out_selection_cv": original_summary["rmse"],
        "well_hash_holdout_selection_cv": well_summary["rmse"],
        "leave_one_original_fold_out_bucket_selection_cv": original_bucket_summary["rmse"],
        "well_hash_holdout_bucket_selection_cv": well_bucket_summary["rmse"],
        "selected_clean_cv": round(float(selected_clean_cv), 6),
        "selector_supported": selector_supported,
        "global_selection_supported": global_supported,
        "bucket_selection_supported": bucket_supported,
        "metric": "rmse",
        "metrics": metric_rows,
        "selection_metrics": selection_summaries,
        "notes": (
            "Global fixed candidate selection passed original-fold and well-hash checks; "
            "bucket/hard confidence selectors are diagnostic only."
            if selector_supported
            else "Keep public_pf_selector as clean CV; candidate selection is diagnostic only."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "candidate_selector_metrics.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "candidate_selector_bucket_metrics.csv",
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "candidate_selector_selection.csv",
        index=False,
    )
    well_metrics.to_csv(output_dir / "candidate_selector_well_metrics.csv", index=False)
    with (output_dir / "candidate_selector_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")
    with paths.metrics_path.open("w") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_local_config()
    feature_path = Path(args.features or get_nested(config, "data.feature_path"))
    if not feature_path.is_absolute():
        feature_path = paths.root / feature_path
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = paths.root / output_dir
    summary = run_audit(paths, config, feature_path, output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
