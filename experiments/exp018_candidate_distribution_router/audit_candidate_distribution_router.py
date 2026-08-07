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

KEY_COLS = ["fold", "well_id", "row_id", "row_index", "eval_step"]
PRIMARY_USECOLS = [
    "variant",
    "fold",
    "well_id",
    "row_id",
    "row_index",
    "eval_step",
    "eval_row_count",
    "last_anchor",
    "y_true",
    "y_pred",
]
SECONDARY_USECOLS = ["variant", *KEY_COLS, "y_pred"]


@dataclass(frozen=True)
class RouterStats:
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
    notes: str = ""

    @property
    def rmse(self) -> float:
        return rmse_from_sse(self.sse, self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate-distribution routers.")
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


def load_variant(
    path: Path,
    *,
    variant: str,
    pred_name: str,
    chunk_rows: int,
    primary: bool,
) -> pd.DataFrame:
    usecols = PRIMARY_USECOLS if primary else SECONDARY_USECOLS
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows):
        chunk = chunk[chunk["variant"] == variant]
        if chunk.empty:
            continue
        chunk = chunk.drop(columns=["variant"])
        chunk = chunk.rename(columns={"y_pred": f"pred_{pred_name}"})
        frames.append(chunk)
    if not frames:
        raise ValueError(f"No rows for variant={variant!r} in {path}")
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)


def load_candidate_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    chunk_rows = int(get_nested(config, "audit.chunk_rows", 250000))
    source_candidates = list(get_nested(config, "audit.source_candidates", []))
    if not source_candidates:
        raise ValueError("audit.source_candidates must be non-empty")
    primary_items = [item for item in source_candidates if item.get("role") == "primary"]
    if len(primary_items) != 1:
        raise ValueError("Exactly one source candidate must have role=primary")

    loaded: list[str] = []
    skipped: list[str] = []
    primary = primary_items[0]
    frame = load_variant(
        Path(primary["path"]),
        variant=str(primary["variant"]),
        pred_name=str(primary["name"]),
        chunk_rows=chunk_rows,
        primary=True,
    )
    loaded.append(str(primary["name"]))

    for item in source_candidates:
        name = str(item["name"])
        if name == primary["name"]:
            continue
        path = Path(item["path"])
        optional = bool(item.get("optional", False))
        if not path.exists():
            if optional:
                skipped.append(f"{name}: missing {path}")
                continue
            raise FileNotFoundError(f"Required OOF candidate missing: {path}")
        candidate = load_variant(
            path,
            variant=str(item["variant"]),
            pred_name=name,
            chunk_rows=chunk_rows,
            primary=False,
        )
        before_rows = len(frame)
        frame = frame.merge(candidate, on=KEY_COLS, how="left", validate="one_to_one")
        missing = int(frame[f"pred_{name}"].isna().sum())
        if missing:
            if optional:
                frame = frame.drop(columns=[f"pred_{name}"])
                skipped.append(f"{name}: {missing} unmatched rows")
                continue
            raise ValueError(f"Required OOF candidate {name} has {missing} unmatched rows")
        if len(frame) != before_rows:
            raise ValueError(f"Merge changed row count for {name}: {before_rows} -> {len(frame)}")
        loaded.append(name)
    return frame, loaded, skipped


def bucket_codes(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        codes[mask] = idx
        previous_max = max_step
    return codes


def candidate_pred(
    name: str,
    *,
    frame: pd.DataFrame,
    anchor: np.ndarray,
    available: set[str],
) -> np.ndarray:
    if name == "last_anchor":
        return anchor.copy()
    if name not in available:
        raise KeyError(name)
    return frame[f"pred_{name}"].to_numpy(dtype=float).copy()


def weighted_blend_pred(
    weights: dict[str, Any],
    *,
    frame: pd.DataFrame,
    anchor: np.ndarray,
    available: set[str],
) -> np.ndarray:
    total_weight = float(sum(float(value) for value in weights.values()))
    if total_weight <= 0:
        raise ValueError("blend weights must sum to a positive value")
    output = np.zeros(anchor.shape, dtype=float)
    for name, weight in weights.items():
        output += candidate_pred(str(name), frame=frame, anchor=anchor, available=available) * (
            float(weight) / total_weight
        )
    return output


def router_prediction(
    router: dict[str, Any],
    *,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    anchor: np.ndarray,
    eval_step: np.ndarray,
    bucket_code_values: np.ndarray,
    buckets: list[dict[str, Any]],
    available: set[str],
) -> tuple[np.ndarray, str]:
    method = str(router["method"])
    params = dict(router.get("params") or {})
    if method == "fixed":
        pred = candidate_pred(
            str(params["candidate"]),
            frame=frame,
            anchor=anchor,
            available=available,
        )
        return (
            pred,
            "",
        )
    if method == "weighted_blend":
        return (
            weighted_blend_pred(
                dict(params["weights"]), frame=frame, anchor=anchor, available=available
            ),
            "",
        )
    if method == "distance_router":
        default_name = str(params.get("default", "raw_lightgbm_no_gr"))
        output = candidate_pred(default_name, frame=frame, anchor=anchor, available=available)
        previous_max = -np.inf
        for rule in sorted(list(params.get("rules", [])), key=lambda item: float(item["max_step"])):
            max_step = float(rule["max_step"])
            mask = (eval_step > previous_max) & (eval_step <= max_step)
            if "blend" in rule:
                output[mask] = weighted_blend_pred(
                    dict(rule["blend"]), frame=frame, anchor=anchor, available=available
                )[mask]
            else:
                output[mask] = candidate_pred(
                    str(rule["candidate"]), frame=frame, anchor=anchor, available=available
                )[mask]
            previous_max = max_step
        return output, ""
    if method == "disagreement_damped":
        primary = candidate_pred(
            str(params["primary"]),
            frame=frame,
            anchor=anchor,
            available=available,
        )
        refs = [
            candidate_pred(str(name), frame=frame, anchor=anchor, available=available)
            for name in params.get("references", [])
            if str(name) in available
        ]
        if not refs:
            raise KeyError("no available references for disagreement_damped")
        stacked = np.vstack([primary, *refs])
        spread = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
        threshold = float(params.get("threshold", 12.0))
        shrink = float(params.get("shrink", 0.70))
        min_step = float(params.get("min_step", 0))
        output = primary.copy()
        mask = (spread >= threshold) & (eval_step >= min_step)
        output[mask] = anchor[mask] + shrink * (primary[mask] - anchor[mask])
        return output, f"damped_rows={int(mask.sum())}"
    if method == "bucket_oracle":
        names = [str(name) for name in params.get("candidates", []) if str(name) in available]
        if "last_anchor" in params.get("candidates", []):
            names = ["last_anchor", *[name for name in names if name != "last_anchor"]]
        if not names:
            raise KeyError("no available candidates for bucket_oracle")
        output = np.empty(anchor.shape, dtype=float)
        choices: list[str] = []
        for bucket_idx, bucket in enumerate(buckets):
            mask = bucket_code_values == bucket_idx
            best_name = min(
                names,
                key=lambda name: float(
                    np.square(
                        candidate_pred(
                            name,
                            frame=frame,
                            anchor=anchor,
                            available=available,
                        )[mask]
                        - y_true[mask],
                    ).sum()
                ),
            )
            best_pred = candidate_pred(
                best_name,
                frame=frame,
                anchor=anchor,
                available=available,
            )
            output[mask] = best_pred[mask]
            choices.append(f"{bucket['name']}={best_name}")
        return output, ";".join(choices)
    raise ValueError(f"unsupported router method: {method}")


def aggregate_router(
    frame: pd.DataFrame,
    *,
    router: dict[str, Any],
    y_true: np.ndarray,
    anchor: np.ndarray,
    eval_step: np.ndarray,
    fold_codes: np.ndarray,
    fold_count: int,
    well_fold_codes: np.ndarray,
    well_fold_count: int,
    bucket_code_values: np.ndarray,
    bucket_count: int,
    buckets: list[dict[str, Any]],
    available: set[str],
) -> RouterStats | None:
    try:
        pred, notes = router_prediction(
            router,
            frame=frame,
            y_true=y_true,
            anchor=anchor,
            eval_step=eval_step,
            bucket_code_values=bucket_code_values,
            buckets=buckets,
            available=available,
        )
    except KeyError:
        return None
    diff2 = np.square(pred - y_true)
    return RouterStats(
        name=str(router["name"]),
        method=str(router["method"]),
        params=dict(router.get("params") or {}),
        selectable=bool(router.get("selectable", True)),
        rows=int(diff2.size),
        sse=float(diff2.sum()),
        fold_sse=np.bincount(fold_codes, weights=diff2, minlength=fold_count),
        fold_n=np.bincount(fold_codes, minlength=fold_count).astype(float),
        well_fold_sse=np.bincount(well_fold_codes, weights=diff2, minlength=well_fold_count),
        well_fold_n=np.bincount(well_fold_codes, minlength=well_fold_count).astype(float),
        bucket_sse=np.bincount(bucket_code_values, weights=diff2, minlength=bucket_count),
        bucket_n=np.bincount(bucket_code_values, minlength=bucket_count).astype(float),
        notes=notes,
    )


def build_selection_audit(
    *,
    name: str,
    stats_by_name: dict[str, RouterStats],
    selectable_names: list[str],
    raw_name: str,
    attr_sse: str,
    attr_n: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_stats = stats_by_name[raw_name]
    holdout_sse = getattr(raw_stats, attr_sse)
    holdout_n = getattr(raw_stats, attr_n)
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
                stats.sse - float(candidate_holdout_sse),
                stats.rows - float(candidate_holdout_n),
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
                "selected_router": selected,
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
    return (
        {
            "candidate": name,
            "rmse": round(selected_rmse, 6),
            "raw_holdout_rmse": round(raw_rmse, 6),
            "delta_vs_raw": round(selected_rmse - raw_rmse, 6),
            "rows": int(selected_total_n),
        },
        rows,
    )


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    config = load_local_config()
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    routers = list(get_nested(config, "audit.routers", []))
    raw_name = str(get_nested(config, "audit.raw_candidate", "raw_lightgbm_no_gr"))
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    if not buckets or not routers:
        raise ValueError("audit.distance_buckets and audit.routers must be non-empty")

    frame, loaded_candidates, skipped_candidates = load_candidate_frame(config)
    available = set(loaded_candidates)
    available.add("last_anchor")
    y_true = frame["y_true"].to_numpy(dtype=float)
    anchor = frame["last_anchor"].to_numpy(dtype=float)
    eval_step = frame["eval_step"].to_numpy(dtype=float)
    folds = sorted(int(value) for value in frame["fold"].unique())
    fold_map = {fold: idx for idx, fold in enumerate(folds)}
    fold_codes = frame["fold"].map(fold_map).to_numpy(dtype=np.int16)
    well_fold_codes = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_holdout_folds)
    ).to_numpy(dtype=np.int16)
    bucket_code_values = bucket_codes(eval_step, buckets)

    stats: list[RouterStats] = []
    skipped_routers: list[str] = []
    for router in routers:
        item = aggregate_router(
            frame,
            router=router,
            y_true=y_true,
            anchor=anchor,
            eval_step=eval_step,
            fold_codes=fold_codes,
            fold_count=len(folds),
            well_fold_codes=well_fold_codes,
            well_fold_count=well_holdout_folds,
            bucket_code_values=bucket_code_values,
            bucket_count=len(buckets),
            buckets=buckets,
            available=available,
        )
        if item is None:
            skipped_routers.append(str(router["name"]))
            continue
        stats.append(item)
    stats_by_name = {item.name: item for item in stats}
    if raw_name not in stats_by_name:
        raise ValueError(f"raw router {raw_name!r} was not evaluated")
    raw_rmse = stats_by_name[raw_name].rmse
    selectable_names = [item.name for item in stats if item.selectable]

    metric_rows = [
        {
            "router": item.name,
            "method": item.method,
            "rmse": round(item.rmse, 6),
            "delta_vs_raw": round(item.rmse - raw_rmse, 6),
            "rows": item.rows,
            "selectable": item.selectable,
            "params_json": params_json(item.params),
            "notes": item.notes,
        }
        for item in stats
    ]
    metric_rows = sorted(metric_rows, key=lambda row: row["rmse"])

    original_summary, original_rows = build_selection_audit(
        name="leave_one_original_fold_out_router_selection",
        stats_by_name=stats_by_name,
        selectable_names=selectable_names,
        raw_name=raw_name,
        attr_sse="fold_sse",
        attr_n="fold_n",
    )
    well_summary, well_rows = build_selection_audit(
        name="well_hash_holdout_router_selection",
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
                    "router": item.name,
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
    selected_clean_cv = (
        original_summary["rmse"]
        if original_summary["rmse"] < raw_rmse and well_summary["rmse"] < raw_rmse
        else raw_rmse
    )
    router_supported = selected_clean_cv < raw_rmse
    summary = {
        "experiment": "exp018_candidate_distribution_router",
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "rows": int(len(frame)),
        "loaded_candidates": loaded_candidates,
        "skipped_candidates": skipped_candidates,
        "skipped_routers": skipped_routers,
        "raw_clean_cv": round(raw_rmse, 6),
        "best_same_oof_router": best_same_oof["router"],
        "best_same_oof_cv": best_same_oof["rmse"],
        "leave_one_original_fold_out_selection_cv": original_summary["rmse"],
        "well_hash_holdout_selection_cv": well_summary["rmse"],
        "selected_clean_cv": round(selected_clean_cv, 6),
        "router_supported": router_supported,
        "metric": "rmse",
        "notes": (
            "Fold-held router selection supports routing over raw."
            if router_supported
            else "Keep raw LightGBM no-GR as clean CV; routers are diagnostics only."
        ),
        "metrics": metric_rows,
        "selection_metrics": [original_summary, well_summary],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "candidate_router_metrics.csv", index=False)
    pd.DataFrame(original_rows + well_rows).to_csv(
        output_dir / "candidate_router_selection.csv", index=False
    )
    pd.DataFrame(bucket_rows).to_csv(
        output_dir / "candidate_router_bucket_summary.csv", index=False
    )
    with (output_dir / "candidate_router_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    with (Path(__file__).with_name("metrics.json")).open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
