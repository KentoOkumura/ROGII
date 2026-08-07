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

USECOLS = [
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

    @property
    def rmse(self) -> float:
        return rmse_from_sse(self.sse, self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit public-style postprocess candidates.")
    parser.add_argument("--oof", default=None, help="Path to row_oof_predictions.csv")
    parser.add_argument("--variant", default=None, help="OOF variant to audit")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    return parser.parse_args()


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_local_config() -> dict[str, Any]:
    with Path(__file__).with_name("config.yaml").open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return value


def rmse_from_sse(sse: float, n_rows: int | float) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(max(0.0, float(sse)) / float(n_rows))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % n_folds


def load_oof(path: Path, *, variant: str, chunk_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunk_rows):
        chunk = chunk[chunk["variant"] == variant]
        if not chunk.empty:
            frames.append(chunk.drop(columns=["variant"]))
    if not frames:
        raise ValueError(f"No rows found for variant={variant!r} in {path}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["well_id", "eval_step", "row_index"], kind="mergesort")
    return frame.reset_index(drop=True)


def bucket_codes(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        codes[mask] = idx
        previous_max = max_step
    return codes


def _odd_window(value: int, n_rows: int) -> int:
    window = max(1, int(value))
    if window % 2 == 0:
        window += 1
    if n_rows > 0 and window > n_rows:
        window = n_rows if n_rows % 2 == 1 else max(1, n_rows - 1)
    return max(1, window)


def smooth_1d(values: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    window = _odd_window(window, values.size)
    if window < 3:
        return values.copy()
    polyorder = min(max(1, int(polyorder)), window - 1)
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return (
            pd.Series(values, dtype="float64")
            .rolling(window, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
    return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")


def smooth_by_well(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for indices in frame.groupby("well_id", sort=False).indices.values():
        output[indices] = smooth_1d(values[indices], window=window, polyorder=polyorder)
    return output


def distance_bucket_alpha(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    alpha = np.ones(eval_step.shape, dtype=float)
    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        alpha[mask] = float(bucket.get("alpha", 1.0))
        previous_max = max_step
    return alpha


def candidate_prediction(
    frame: pd.DataFrame,
    *,
    method: str,
    params: dict[str, Any],
    anchor: np.ndarray,
    raw: np.ndarray,
    eval_step: np.ndarray,
) -> np.ndarray:
    residual = raw - anchor
    if method == "raw":
        return raw.copy()
    if method == "last_anchor":
        return anchor.copy()
    if method == "sg_smooth":
        smoothed = smooth_by_well(
            frame,
            raw,
            window=int(params.get("window", 21)),
            polyorder=int(params.get("polyorder", 2)),
        )
        blend = float(params.get("blend", 1.0))
        return raw * (1.0 - blend) + smoothed * blend
    if method == "fade_in":
        fade_rows = max(1.0, float(params.get("fade_rows", 50)))
        start_alpha = float(params.get("start_alpha", 0.20))
        end_alpha = float(params.get("end_alpha", 1.0))
        progress = np.clip(eval_step / fade_rows, 0.0, 1.0)
        alpha = start_alpha + (end_alpha - start_alpha) * progress
        return anchor + alpha * residual
    if method == "hold_blend":
        hold_rows = max(1.0, float(params.get("hold_rows", 50)))
        start_weight = float(params.get("start_weight", 0.80))
        hold_weight = start_weight * np.clip(1.0 - eval_step / hold_rows, 0.0, 1.0)
        return anchor * hold_weight + raw * (1.0 - hold_weight)
    if method == "alpha_tau":
        tau = max(1.0, float(params.get("tau", 50)))
        alpha_min = float(params.get("alpha_min", 0.20))
        alpha_max = float(params.get("alpha_max", 1.0))
        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - np.exp(-eval_step / tau))
        return anchor + alpha * residual
    if method == "sg_then_fade":
        smoothed = smooth_by_well(
            frame,
            raw,
            window=int(params.get("window", 21)),
            polyorder=int(params.get("polyorder", 2)),
        )
        smooth_residual = smoothed - anchor
        fade_rows = max(1.0, float(params.get("fade_rows", 50)))
        start_alpha = float(params.get("start_alpha", 0.20))
        end_alpha = float(params.get("end_alpha", 1.0))
        progress = np.clip(eval_step / fade_rows, 0.0, 1.0)
        alpha = start_alpha + (end_alpha - start_alpha) * progress
        return anchor + alpha * smooth_residual
    if method == "distance_bucket_shrink":
        alpha = distance_bucket_alpha(eval_step, list(params.get("buckets", [])))
        return anchor + alpha * residual
    raise ValueError(f"unsupported postprocess method: {method}")


def aggregate_candidate(
    frame: pd.DataFrame,
    *,
    candidate: dict[str, Any],
    y_true: np.ndarray,
    anchor: np.ndarray,
    raw: np.ndarray,
    eval_step: np.ndarray,
    fold_codes: np.ndarray,
    fold_count: int,
    well_fold_codes: np.ndarray,
    well_fold_count: int,
    bucket_code_values: np.ndarray,
    bucket_count: int,
) -> CandidateStats:
    params = dict(candidate.get("params") or {})
    method = str(candidate["method"])
    pred = candidate_prediction(
        frame,
        method=method,
        params=params,
        anchor=anchor,
        raw=raw,
        eval_step=eval_step,
    )
    diff2 = np.square(pred - y_true)
    rows = int(diff2.size)
    return CandidateStats(
        name=str(candidate["name"]),
        method=method,
        params=params,
        selectable=bool(candidate.get("selectable", True)),
        rows=rows,
        sse=float(diff2.sum()),
        fold_sse=np.bincount(fold_codes, weights=diff2, minlength=fold_count),
        fold_n=np.bincount(fold_codes, minlength=fold_count).astype(float),
        well_fold_sse=np.bincount(well_fold_codes, weights=diff2, minlength=well_fold_count),
        well_fold_n=np.bincount(well_fold_codes, minlength=well_fold_count).astype(float),
        bucket_sse=np.bincount(bucket_code_values, weights=diff2, minlength=bucket_count),
        bucket_n=np.bincount(bucket_code_values, minlength=bucket_count).astype(float),
    )


def build_selection_audit(
    *,
    name: str,
    stats_by_name: dict[str, CandidateStats],
    selectable_names: list[str],
    raw_name: str,
    attr_sse: str,
    attr_n: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_stats = stats_by_name[raw_name]
    holdout_sse = getattr(raw_stats, attr_sse)
    holdout_n = getattr(raw_stats, attr_n)
    total_sse = {candidate: stats_by_name[candidate].sse for candidate in selectable_names}
    total_n = {candidate: float(stats_by_name[candidate].rows) for candidate in selectable_names}

    rows: list[dict[str, Any]] = []
    selected_total_sse = 0.0
    raw_total_sse = 0.0
    selected_total_n = 0.0
    for fold_idx in range(len(holdout_n)):
        if holdout_n[fold_idx] <= 0:
            continue
        train_scores: dict[str, float] = {}
        for candidate in selectable_names:
            stats = stats_by_name[candidate]
            candidate_holdout_sse = getattr(stats, attr_sse)[fold_idx]
            candidate_holdout_n = getattr(stats, attr_n)[fold_idx]
            train_scores[candidate] = rmse_from_sse(
                total_sse[candidate] - float(candidate_holdout_sse),
                total_n[candidate] - float(candidate_holdout_n),
            )
        selected = min(train_scores, key=train_scores.get)
        selected_stats = stats_by_name[selected]
        selected_sse = float(getattr(selected_stats, attr_sse)[fold_idx])
        selected_n = float(getattr(selected_stats, attr_n)[fold_idx])
        raw_sse = float(holdout_sse[fold_idx])
        raw_n = float(holdout_n[fold_idx])
        selected_total_sse += selected_sse
        raw_total_sse += raw_sse
        selected_total_n += selected_n
        rows.append(
            {
                "audit": name,
                "holdout_fold": fold_idx,
                "selected_candidate": selected,
                "train_rmse": round(train_scores[selected], 6),
                "holdout_rmse": round(rmse_from_sse(selected_sse, selected_n), 6),
                "holdout_raw_rmse": round(rmse_from_sse(raw_sse, raw_n), 6),
                "holdout_delta_vs_raw": round(
                    rmse_from_sse(selected_sse, selected_n) - rmse_from_sse(raw_sse, raw_n),
                    6,
                ),
                "rows": int(selected_n),
            }
        )
    selected_rmse = rmse_from_sse(selected_total_sse, selected_total_n)
    raw_rmse = rmse_from_sse(raw_total_sse, selected_total_n)
    summary = {
        "candidate": name,
        "rmse": round(selected_rmse, 6),
        "raw_holdout_rmse": round(raw_rmse, 6),
        "delta_vs_raw": round(selected_rmse - raw_rmse, 6),
        "rows": int(selected_total_n),
    }
    return summary, rows


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_local_config()
    oof_path = Path(args.oof or get_nested(config, "audit.source_oof_predictions"))
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    variant = args.variant or str(get_nested(config, "audit.source_variant", "lightgbm_no_gr"))
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    chunk_rows = int(get_nested(config, "audit.chunk_rows", 250000))
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    candidates = list(get_nested(config, "audit.candidates", []))
    if not candidates:
        raise ValueError("audit.candidates must be a non-empty list")

    frame = load_oof(oof_path, variant=variant, chunk_rows=chunk_rows)
    y_true = frame["y_true"].to_numpy(dtype=float)
    anchor = frame["last_anchor"].to_numpy(dtype=float)
    raw = frame["y_pred"].to_numpy(dtype=float)
    eval_step = frame["eval_step"].to_numpy(dtype=float)

    folds = sorted(int(value) for value in frame["fold"].unique())
    fold_map = {fold: idx for idx, fold in enumerate(folds)}
    fold_codes = frame["fold"].map(fold_map).to_numpy(dtype=np.int16)
    well_fold_codes = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_holdout_folds)
    ).to_numpy(dtype=np.int16)
    bucket_code_values = bucket_codes(eval_step, buckets)

    stats: list[CandidateStats] = []
    for candidate in candidates:
        stats.append(
            aggregate_candidate(
                frame,
                candidate=candidate,
                y_true=y_true,
                anchor=anchor,
                raw=raw,
                eval_step=eval_step,
                fold_codes=fold_codes,
                fold_count=len(folds),
                well_fold_codes=well_fold_codes,
                well_fold_count=well_holdout_folds,
                bucket_code_values=bucket_code_values,
                bucket_count=len(buckets),
            )
        )

    stats_by_name = {item.name: item for item in stats}
    raw_name = "raw_lightgbm_no_gr"
    raw_rmse = stats_by_name[raw_name].rmse
    selectable_names = [item.name for item in stats if item.selectable]

    metric_rows = []
    for item in stats:
        metric_rows.append(
            {
                "candidate": item.name,
                "method": item.method,
                "rmse": round(item.rmse, 6),
                "delta_vs_raw": round(item.rmse - raw_rmse, 6),
                "rows": item.rows,
                "selectable": item.selectable,
                "params_json": params_json(item.params),
            }
        )
    metric_rows = sorted(metric_rows, key=lambda row: row["rmse"])

    original_summary, original_rows = build_selection_audit(
        name="leave_one_original_fold_out_candidate_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        raw_name=raw_name,
        attr_sse="fold_sse",
        attr_n="fold_n",
    )
    well_summary, well_rows = build_selection_audit(
        name="well_hash_holdout_candidate_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        raw_name=raw_name,
        attr_sse="well_fold_sse",
        attr_n="well_fold_n",
    )

    bucket_rows: list[dict[str, Any]] = []
    for item in stats:
        for bucket_idx, bucket in enumerate(buckets):
            bucket_rows.append(
                {
                    "candidate": item.name,
                    "method": item.method,
                    "bucket": str(bucket["name"]),
                    "max_step": float(bucket["max_step"]),
                    "rmse": round(
                        rmse_from_sse(item.bucket_sse[bucket_idx], item.bucket_n[bucket_idx]),
                        6,
                    ),
                    "raw_rmse": round(
                        rmse_from_sse(
                            stats_by_name[raw_name].bucket_sse[bucket_idx],
                            stats_by_name[raw_name].bucket_n[bucket_idx],
                        ),
                        6,
                    ),
                    "rows": int(item.bucket_n[bucket_idx]),
                }
            )

    best_same_oof = metric_rows[0]
    clean_candidates = [original_summary, well_summary]
    selected_clean_cv = (
        original_summary["rmse"]
        if original_summary["rmse"] < raw_rmse and well_summary["rmse"] < raw_rmse
        else raw_rmse
    )
    clean_postprocess_supported = selected_clean_cv < raw_rmse

    summary = {
        "experiment": "exp016_public_postprocess_ablation",
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "source_oof_predictions": str(oof_path),
        "source_variant": variant,
        "rows": int(len(frame)),
        "raw_clean_cv": round(raw_rmse, 6),
        "best_same_oof_candidate": best_same_oof["candidate"],
        "best_same_oof_cv": best_same_oof["rmse"],
        "leave_one_original_fold_out_selection_cv": original_summary["rmse"],
        "well_hash_holdout_selection_cv": well_summary["rmse"],
        "selected_clean_cv": round(selected_clean_cv, 6),
        "clean_postprocess_supported": clean_postprocess_supported,
        "metric": "rmse",
        "notes": (
            "Fold-held candidate selection supports public-style postprocess over raw."
            if clean_postprocess_supported
            else (
                "Keep raw LightGBM no-GR as clean CV; "
                "public-style candidates are diagnostics only."
            )
        ),
        "metrics": metric_rows,
        "selection_metrics": clean_candidates,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "public_postprocess_ablation_metrics.csv", index=False
    )
    pd.DataFrame(original_rows + well_rows).to_csv(
        output_dir / "public_postprocess_ablation_selection.csv", index=False
    )
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "public_postprocess_ablation_bucket_summary.csv", index=False
    )
    with (output_dir / "public_postprocess_ablation_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    with (Path(__file__).with_name("metrics.json")).open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
