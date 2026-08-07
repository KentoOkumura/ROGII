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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_USECOLS = {
    "id",
    "well_id",
    "fold",
    "eval_step",
    "distance_bucket",
    "target_tvt",
    "last_anchor_tvt",
    "pf_pred",
    "beam_pred",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: str
    params: dict[str, Any]
    residual_shrink_values: tuple[float, ...]
    residual_clip_values: tuple[float | None, ...]
    target_clip: float | None
    max_train_rows: int | None
    max_train_rows_per_bucket: int | None
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit public sel15 PF residual correction.")
    parser.add_argument("--features", default=None, help="Path to exp029 feature CSV")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Override model row cap")
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % n_folds


def safe_name(value: float | None) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p").replace("-", "m")


def required_columns(config: dict[str, Any]) -> list[str]:
    features = {str(value) for value in get_nested(config, "model.features", [])}
    controls = {
        "last_anchor_tvt",
        "pf_pred",
        "beam_pred",
        "target_tvt",
    }
    return sorted(BASE_USECOLS | features | controls)


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
    frame = frame.sort_values(["fold", "well_id", "eval_step"], kind="mergesort").reset_index(
        drop=True
    )
    if not np.isfinite(frame["target_tvt"].to_numpy(dtype=float)).all():
        raise ValueError("target_tvt contains non-finite values")
    return frame, usecols


def resolve_feature_path(paths: ExperimentPaths, configured_path: str | Path) -> Path:
    feature_path = Path(configured_path)
    if not feature_path.is_absolute():
        feature_path = paths.root / feature_path
    if feature_path.exists():
        return feature_path

    kaggle_input_root = Path("/kaggle/input")
    if kaggle_input_root.exists():
        matches = sorted(kaggle_input_root.rglob(Path(configured_path).name))
        if matches:
            return matches[0]
    return feature_path


def bucket_codes(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        codes[mask] = idx
        previous_max = max_step
    return codes


def build_feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = frame[features].to_numpy(dtype=np.float32, copy=True)
    return matrix


def model_specs(config: dict[str, Any], max_train_rows_override: int | None) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for item in get_nested(config, "model.candidates", []):
        params = dict(item.get("params") or {})
        max_train_rows = item.get("max_train_rows")
        if max_train_rows_override is not None:
            max_train_rows = max_train_rows_override
        specs.append(
            ModelSpec(
                name=str(item["name"]),
                estimator=str(item["estimator"]),
                params=params,
                residual_shrink_values=tuple(
                    float(value) for value in item.get("residual_shrink_values", [1.0])
                ),
                residual_clip_values=tuple(
                    None if value is None else float(value)
                    for value in item.get("residual_clip_values", [None])
                ),
                target_clip=(
                    None if item.get("target_clip") is None else float(item.get("target_clip"))
                ),
                max_train_rows=(
                    None if max_train_rows is None else int(max_train_rows)
                ),
                max_train_rows_per_bucket=(
                    None
                    if item.get("max_train_rows_per_bucket") is None
                    else int(item.get("max_train_rows_per_bucket"))
                ),
                seed=int(item.get("seed", get_nested(config, "validation.seed", 42))),
            )
        )
    if not specs:
        raise ValueError("model.candidates must be non-empty")
    return specs


def make_estimator(spec: ModelSpec, split: int) -> Pipeline:
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
            "learning_rate": 0.06,
            "max_iter": 140,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 80,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "validation_fraction": 0.12,
            "n_iter_no_change": 12,
            "random_state": seed,
        }
        params.update(spec.params)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(**params)),
            ]
        )
    raise ValueError(f"unsupported estimator: {spec.estimator}")


def choose_train_indices(
    train_mask: np.ndarray,
    bucket_code_values: np.ndarray,
    *,
    max_rows: int | None,
    max_rows_per_bucket: int | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train_mask)
    if max_rows_per_bucket is not None:
        selected_parts: list[np.ndarray] = []
        for bucket in np.unique(bucket_code_values[train_idx]):
            bucket_idx = train_idx[bucket_code_values[train_idx] == bucket]
            if len(bucket_idx) > max_rows_per_bucket:
                bucket_idx = rng.choice(bucket_idx, size=max_rows_per_bucket, replace=False)
            selected_parts.append(np.asarray(bucket_idx, dtype=np.int64))
        train_idx = np.concatenate(selected_parts)
    if max_rows is not None and len(train_idx) > max_rows:
        train_idx = rng.choice(train_idx, size=max_rows, replace=False)
    return np.asarray(np.sort(train_idx), dtype=np.int64)


def control_predictions(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for item in get_nested(config, "audit.controls", []):
        name = str(item["name"])
        method = str(item["method"])
        if method == "column":
            output[name] = frame[str(item["column"])].to_numpy(dtype=float)
        elif method == "blend":
            weights = dict(item.get("weights") or {})
            total = float(sum(float(value) for value in weights.values()))
            if total <= 0:
                raise ValueError(f"control {name} has non-positive weights")
            pred = np.zeros(len(frame), dtype=float)
            for column, weight in weights.items():
                pred += frame[str(column)].to_numpy(dtype=float) * (float(weight) / total)
            output[name] = pred
        else:
            raise ValueError(f"unsupported control method: {method}")
    return output


def aggregate_global(
    *,
    audit: str,
    candidate: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    reference_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    score = rmse(y_true, pred)
    row = {
        "audit": audit,
        "candidate": candidate,
        "rmse": round(score, 6),
        "rows": int(len(y_true)),
    }
    for ref_name, ref_pred in reference_predictions.items():
        row[f"delta_vs_{ref_name}"] = round(score - rmse(y_true, ref_pred), 6)
    return row


def aggregate_by_code(
    *,
    audit: str,
    candidate: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    codes: np.ndarray,
    code_name: str,
    labels: list[Any],
    reference: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diff2 = np.square(pred - y_true)
    ref_diff2 = np.square(reference - y_true)
    for code, label in enumerate(labels):
        mask = codes == code
        n_rows = int(mask.sum())
        score = rmse_from_sse(float(diff2[mask].sum()), n_rows)
        ref_score = rmse_from_sse(float(ref_diff2[mask].sum()), n_rows)
        rows.append(
            {
                "audit": audit,
                "candidate": candidate,
                code_name: label,
                "rmse": round(score, 6),
                "reference_rmse": round(ref_score, 6),
                "delta_vs_reference": round(score - ref_score, 6),
                "rows": n_rows,
            }
        )
    return rows


def aggregate_well_metrics(
    *,
    audit: str,
    candidate: str,
    frame: pd.DataFrame,
    pred: np.ndarray,
    y_true: np.ndarray,
    reference: np.ndarray,
) -> pd.DataFrame:
    base = frame[["well_id", "fold"]].copy()
    diff2 = np.square(pred - y_true)
    ref_diff2 = np.square(reference - y_true)
    grouped = (
        base.assign(diff2=diff2, ref_diff2=ref_diff2)
        .groupby(["well_id", "fold"], sort=False)
        .agg(sse=("diff2", "sum"), ref_sse=("ref_diff2", "sum"), rows=("diff2", "size"))
        .reset_index()
    )
    grouped["audit"] = audit
    grouped["candidate"] = candidate
    grouped["rmse"] = np.sqrt(grouped["sse"] / grouped["rows"])
    grouped["reference_rmse"] = np.sqrt(grouped["ref_sse"] / grouped["rows"])
    grouped["delta_vs_reference"] = grouped["rmse"] - grouped["reference_rmse"]
    return grouped[
        [
            "audit",
            "candidate",
            "well_id",
            "fold",
            "rows",
            "rmse",
            "reference_rmse",
            "delta_vs_reference",
        ]
    ]


def cross_fit_predictions(
    *,
    audit: str,
    frame: pd.DataFrame,
    x_matrix: np.ndarray,
    y_residual: np.ndarray,
    base_pred: np.ndarray,
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    specs: list[ModelSpec],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    unique_splits = sorted(int(value) for value in np.unique(split_codes))
    for spec in specs:
        residual_oof = np.full(len(frame), np.nan, dtype=float)
        train_rows_by_split: dict[int, int] = {}
        for split in unique_splits:
            valid_mask = split_codes == split
            train_mask = ~valid_mask
            train_idx = choose_train_indices(
                train_mask,
                bucket_code_values,
                max_rows=spec.max_train_rows,
                max_rows_per_bucket=spec.max_train_rows_per_bucket,
                seed=spec.seed + split * 7919,
            )
            valid_idx = np.flatnonzero(valid_mask)
            y_train = y_residual[train_idx]
            if spec.target_clip is not None:
                y_train = np.clip(y_train, -spec.target_clip, spec.target_clip)
            estimator = make_estimator(spec, split)
            estimator.fit(x_matrix[train_idx], y_train)
            residual_oof[valid_idx] = estimator.predict(x_matrix[valid_idx])
            train_rows_by_split[split] = int(len(train_idx))

        if not np.isfinite(residual_oof).all():
            raise ValueError(f"{audit}/{spec.name}: non-finite residual predictions")

        for shrink in spec.residual_shrink_values:
            for clip_value in spec.residual_clip_values:
                residual = residual_oof.copy()
                if clip_value is not None:
                    residual = np.clip(residual, -clip_value, clip_value)
                name = (
                    f"{spec.name}_shrink{safe_name(shrink)}_clip{safe_name(clip_value)}"
                )
                predictions[name] = base_pred + shrink * residual
        print(
            json.dumps(
                {
                    "audit": audit,
                    "model": spec.name,
                    "train_rows_by_split": train_rows_by_split,
                },
                sort_keys=True,
            )
        )
    return predictions


def run_audit(
    paths: ExperimentPaths,
    config: dict[str, Any],
    feature_path: Path,
    output_dir: Path | None = None,
    max_train_rows_override: int | None = None,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_dir = output_dir or paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    buckets = list(get_nested(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")
    features = [str(value) for value in get_nested(config, "model.features", [])]
    specs = model_specs(config, max_train_rows_override)

    frame, loaded_columns = load_features(feature_path, config)
    y_true = frame["target_tvt"].to_numpy(dtype=float)
    base_pred = frame["pf_pred"].to_numpy(dtype=float)
    y_residual = y_true - base_pred
    x_matrix = build_feature_matrix(frame, features)

    original_folds = sorted(int(value) for value in frame["fold"].unique())
    original_fold_map = {fold: idx for idx, fold in enumerate(original_folds)}
    original_fold_codes = frame["fold"].map(original_fold_map).to_numpy(dtype=np.int16)
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    well_hash_codes = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_holdout_folds)
    ).to_numpy(dtype=np.int16)
    bucket_code_values = bucket_codes(frame["eval_step"].to_numpy(dtype=float), buckets)
    bucket_labels = [str(bucket["name"]) for bucket in buckets]

    controls = control_predictions(frame, config)
    required_control_names = [
        str(value)
        for value in get_nested(config, "audit.required_controls", ["public_pf_selector"])
    ]
    for name in required_control_names:
        if name not in controls:
            raise ValueError(f"required control {name!r} is not defined")
    reference_name = str(get_nested(config, "audit.reference_control", "public_pf_selector"))
    if reference_name not in controls:
        raise ValueError(f"reference control {reference_name!r} is not defined")
    reference_pred = controls[reference_name]
    required_control_cv = min(rmse(y_true, controls[name]) for name in required_control_names)

    predictions_by_audit: dict[str, dict[str, np.ndarray]] = {
        "control": controls,
        "leave_one_original_fold_out": cross_fit_predictions(
            audit="leave_one_original_fold_out",
            frame=frame,
            x_matrix=x_matrix,
            y_residual=y_residual,
            base_pred=base_pred,
            split_codes=original_fold_codes,
            bucket_code_values=bucket_code_values,
            specs=specs,
        ),
        "well_hash_holdout": cross_fit_predictions(
            audit="well_hash_holdout",
            frame=frame,
            x_matrix=x_matrix,
            y_residual=y_residual,
            base_pred=base_pred,
            split_codes=well_hash_codes,
            bucket_code_values=bucket_code_values,
            specs=specs,
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    well_metric_frames: list[pd.DataFrame] = []

    for audit, predictions in predictions_by_audit.items():
        for candidate, pred in predictions.items():
            metric_rows.append(
                aggregate_global(
                    audit=audit,
                    candidate=candidate,
                    pred=pred,
                    y_true=y_true,
                    reference_predictions={name: controls[name] for name in required_control_names},
                )
            )
            bucket_rows.extend(
                aggregate_by_code(
                    audit=audit,
                    candidate=candidate,
                    pred=pred,
                    y_true=y_true,
                    codes=bucket_code_values,
                    code_name="bucket",
                    labels=bucket_labels,
                    reference=reference_pred,
                )
            )
            if audit == "leave_one_original_fold_out":
                split_rows.extend(
                    aggregate_by_code(
                        audit=audit,
                        candidate=candidate,
                        pred=pred,
                        y_true=y_true,
                        codes=original_fold_codes,
                        code_name="split",
                        labels=original_folds,
                        reference=reference_pred,
                    )
                )
            elif audit == "well_hash_holdout":
                split_rows.extend(
                    aggregate_by_code(
                        audit=audit,
                        candidate=candidate,
                        pred=pred,
                        y_true=y_true,
                        codes=well_hash_codes,
                        code_name="split",
                        labels=list(range(well_holdout_folds)),
                        reference=reference_pred,
                    )
                )
            well_metric_frames.append(
                aggregate_well_metrics(
                    audit=audit,
                    candidate=candidate,
                    frame=frame,
                    pred=pred,
                    y_true=y_true,
                    reference=reference_pred,
                )
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["audit", "rmse"]).reset_index(drop=True)
    bucket_metrics = pd.DataFrame(bucket_rows).sort_values(
        ["audit", "candidate", "bucket"]
    )
    split_metrics = pd.DataFrame(split_rows).sort_values(["audit", "candidate", "split"])
    well_metrics = pd.concat(well_metric_frames, ignore_index=True)

    original = metrics[metrics["audit"] == "leave_one_original_fold_out"]
    well_hash = metrics[metrics["audit"] == "well_hash_holdout"]
    original_by_candidate = dict(zip(original["candidate"], original["rmse"], strict=False))
    well_hash_by_candidate = dict(zip(well_hash["candidate"], well_hash["rmse"], strict=False))
    supported: list[dict[str, Any]] = []
    for candidate, original_rmse in original_by_candidate.items():
        if candidate not in well_hash_by_candidate:
            continue
        well_rmse = float(well_hash_by_candidate[candidate])
        original_rmse = float(original_rmse)
        if original_rmse < required_control_cv and well_rmse < required_control_cv:
            supported.append(
                {
                    "candidate": candidate,
                    "original_fold_rmse": round(original_rmse, 6),
                    "well_hash_rmse": round(well_rmse, 6),
                    "worst_holdout_rmse": round(max(original_rmse, well_rmse), 6),
                }
            )
    supported = sorted(supported, key=lambda row: row["worst_holdout_rmse"])

    best_original = original.iloc[0].to_dict()
    best_well_hash = well_hash.iloc[0].to_dict()
    selected = supported[0] if supported else None
    selected_clean_cv = (
        float(selected["original_fold_rmse"])
        if selected is not None
        else rmse(y_true, reference_pred)
    )

    summary = {
        "experiment": "exp032_public_sel15_pf_residual_correction",
        "status": "completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "feature_file": feature_path.as_posix(),
        "loaded_columns": loaded_columns,
        "feature_columns": features,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "metric": "rmse",
        "target": "target_tvt_minus_pf_pred",
        "reference_control": reference_name,
        "required_controls": required_control_names,
        "required_control_cv": round(required_control_cv, 6),
        "public_pf_selector_cv": round(rmse(y_true, controls["public_pf_selector"]), 6),
        "pf090_hold010_cv": round(rmse(y_true, controls["pf090_hold010"]), 6),
        "best_original_fold_candidate": best_original["candidate"],
        "best_original_fold_cv": round(float(best_original["rmse"]), 6),
        "best_well_hash_candidate": best_well_hash["candidate"],
        "best_well_hash_cv": round(float(best_well_hash["rmse"]), 6),
        "supported_candidates": supported,
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_clean_cv": round(float(selected_clean_cv), 6),
        "residual_model_supported": selected is not None,
        "model_specs": [spec.__dict__ for spec in specs],
        "notes": (
            "Residual correction beats required controls in both holdout audits."
            if selected is not None
            else (
                "Residual correction is diagnostic only; no candidate beat required controls "
                "in both holdout audits."
            )
        ),
    }

    metrics.to_csv(output_dir / "residual_correction_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / "residual_correction_bucket_metrics.csv", index=False)
    split_metrics.to_csv(output_dir / "residual_correction_split_metrics.csv", index=False)
    well_metrics.to_csv(output_dir / "residual_correction_well_metrics.csv", index=False)
    with (output_dir / "residual_correction_summary.json").open("w") as fp:
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
        max_train_rows_override=args.max_train_rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
