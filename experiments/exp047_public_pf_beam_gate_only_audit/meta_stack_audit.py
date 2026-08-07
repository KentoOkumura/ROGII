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
from baseline import (
    build_drift_feature_frame,
    config_get,
    postprocess_predictions,
    predict_drift,
    well_id_from_path,
)
from pseudo_tail_augmentation import fit_pseudo_tail_model_from_files, train_files
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
    "pseudo_cutoff_fraction",
    "cutoff_row",
    "row_idx",
    "eval_step",
    "target_tvt",
    "last_anchor_tvt",
    "pf_pred",
    "beam_pred",
}


@dataclass(frozen=True)
class MetaModelSpec:
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
    parser = argparse.ArgumentParser(description="Audit public sel15 PF meta stack.")
    parser.add_argument("--features", default=None, help="Path to exp029 feature CSV")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke well limit")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Override meta row cap")
    parser.add_argument(
        "--base-estimator",
        choices=["LGBMRegressor", "HistGradientBoostingRegressor"],
        default=None,
        help="Override pseudo-tail base estimator for local smoke checks.",
    )
    parser.add_argument(
        "--skip-hgb",
        action="store_true",
        help="Run only fixed blends and ridge meta candidates.",
    )
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
    generated = {
        "exp026_pred",
        "pf090_hold010",
        "exp026_minus_pf",
        "abs_exp026_minus_pf",
        "exp026_minus_last_anchor",
    }
    return sorted(BASE_USECOLS | controls | (features - generated))


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


def load_features(
    path: Path,
    config: dict[str, Any],
    *,
    allowed_wells: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    usecols = required_columns(config)
    chunk_rows = int(get_nested(config, "runtime.chunk_rows", 500000))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in usecols, chunksize=chunk_rows):
        if allowed_wells is not None:
            chunk = chunk[chunk["well_id"].astype(str).isin(allowed_wells)]
        if not chunk.empty:
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


def bucket_codes(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        codes[mask] = idx
        previous_max = max_step
    return codes


def selected_training_variant(config: dict[str, Any]) -> dict[str, Any]:
    selected = str(get_nested(config, "audit.training_variants.selected_variant"))
    variants = list(get_nested(config, "audit.training_variants.variants", []))
    for variant in variants:
        if isinstance(variant, dict) and str(variant.get("name")) == selected:
            return variant
    raise ValueError(f"selected pseudo-tail variant not found: {selected}")


def file_by_well(paths: ExperimentPaths, max_wells: int | None) -> dict[str, Path]:
    files = train_files(paths, max_wells)
    return {well_id_from_path(path): path for path in files}


def split_code_by_well(frame: pd.DataFrame, split_codes: np.ndarray) -> dict[str, int]:
    tmp = frame[["well_id"]].copy()
    tmp["split_code"] = split_codes
    grouped = tmp.groupby("well_id", sort=False)["split_code"].nunique()
    bad = grouped[grouped != 1]
    if not bad.empty:
        preview = ", ".join(str(value) for value in bad.index[:5])
        raise ValueError(f"split codes must be constant per well; bad wells: {preview}")
    return (
        tmp.drop_duplicates("well_id")
        .set_index("well_id")["split_code"]
        .astype(int)
        .to_dict()
    )


def predict_well_cutoff(
    *,
    path: Path,
    cutoff_row: int,
    row_indices: np.ndarray,
    model: Any,
    config: dict[str, Any],
) -> np.ndarray:
    df = pd.read_csv(path)
    pseudo = df.copy()
    tvt_input = pseudo["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input[int(cutoff_row) :] = np.nan
    pseudo["TVT_input"] = tvt_input
    frame = build_drift_feature_frame(pseudo, config, include_target=False)
    raw_pred = predict_drift(frame, model, config)
    method = str(get_nested(config, "postprocess.selected_method", "distance_bucket_shrink"))
    pred = postprocess_predictions(raw_pred, frame, config, method=method)
    by_row = pd.Series(pred, index=frame.eval_indices.astype(int))
    output = by_row.reindex(row_indices.astype(int)).to_numpy(dtype=float)
    if not np.isfinite(output).all():
        missing = row_indices[~np.isfinite(output)]
        preview = ", ".join(str(int(value)) for value in missing[:5])
        raise ValueError(f"missing exp026 predictions for {path.name} rows: {preview}")
    return output


def generate_exp026_anchor_predictions(
    *,
    audit: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    paths: ExperimentPaths,
    split_codes: np.ndarray,
    max_wells: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    variant = selected_training_variant(config)
    seed = int(config_get(config, "validation.seed", 42))
    max_rows_per_well = int(config_get(config, "model.training.max_train_rows_per_well", 800))
    max_rows_total = int(config_get(config, "model.training.max_train_rows_per_fold", 300000))
    path_by_well = file_by_well(paths, max_wells)
    split_by_well = split_code_by_well(frame, split_codes)
    missing_wells = sorted(set(frame["well_id"].astype(str)) - set(path_by_well))
    if missing_wells:
        preview = ", ".join(missing_wells[:5])
        raise ValueError(f"feature rows reference wells without local train files: {preview}")

    output = np.full(len(frame), np.nan, dtype=float)
    source_rows: list[dict[str, Any]] = []
    unique_splits = sorted(int(value) for value in np.unique(split_codes))
    all_paths = list(path_by_well.values())
    for split in unique_splits:
        train_paths = [
            path
            for path in all_paths
            if split_by_well.get(well_id_from_path(path), -1) != split
        ]
        if not train_paths:
            raise ValueError(f"{audit} split {split} has no pseudo-tail train wells")
        model, n_train_rows, split_source_rows = fit_pseudo_tail_model_from_files(
            train_paths,
            config,
            variant,
            seed=seed + split * 1009,
            max_rows_total=max_rows_total,
            max_rows_per_well=max_rows_per_well,
        )
        for row in split_source_rows:
            row = dict(row)
            row["audit"] = audit
            row["split"] = split
            row["train_rows_total"] = n_train_rows
            source_rows.append(row)

        valid_positions = np.flatnonzero(split_codes == split)
        valid = frame.iloc[valid_positions]
        for (well_id, cutoff_row), part in valid.groupby(["well_id", "cutoff_row"], sort=False):
            positions = part.index.to_numpy(dtype=int)
            output[positions] = predict_well_cutoff(
                path=path_by_well[str(well_id)],
                cutoff_row=int(cutoff_row),
                row_indices=part["row_idx"].to_numpy(dtype=int),
                model=model,
                config=config,
            )
        print(
            json.dumps(
                {
                    "audit": audit,
                    "split": split,
                    "pseudo_tail_train_rows": n_train_rows,
                    "valid_rows": int(len(valid_positions)),
                },
                sort_keys=True,
            )
        )

    if not np.isfinite(output).all():
        raise ValueError(f"{audit}: non-finite exp026 anchor predictions")
    return output, source_rows


def add_generated_features(frame: pd.DataFrame, exp026_pred: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["exp026_pred"] = exp026_pred
    out["pf090_hold010"] = 0.90 * out["pf_pred"].astype(float) + 0.10 * out[
        "last_anchor_tvt"
    ].astype(float)
    out["exp026_minus_pf"] = out["exp026_pred"].astype(float) - out["pf_pred"].astype(float)
    out["abs_exp026_minus_pf"] = np.abs(out["exp026_minus_pf"].to_numpy(dtype=float))
    out["exp026_minus_last_anchor"] = out["exp026_pred"].astype(float) - out[
        "last_anchor_tvt"
    ].astype(float)
    return out


def meta_specs(
    config: dict[str, Any],
    *,
    max_train_rows_override: int | None,
    skip_hgb: bool,
) -> list[MetaModelSpec]:
    specs: list[MetaModelSpec] = []
    for item in get_nested(config, "model.candidates", []):
        estimator = str(item["estimator"])
        if skip_hgb and estimator == "hist_gradient_boosting":
            continue
        max_train_rows = item.get("max_train_rows")
        if max_train_rows_override is not None:
            max_train_rows = max_train_rows_override
        specs.append(
            MetaModelSpec(
                name=str(item["name"]),
                estimator=estimator,
                params=dict(item.get("params") or {}),
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
                max_train_rows=None if max_train_rows is None else int(max_train_rows),
                max_train_rows_per_bucket=(
                    None
                    if item.get("max_train_rows_per_bucket") is None
                    else int(item.get("max_train_rows_per_bucket"))
                ),
                seed=int(item.get("seed", get_nested(config, "validation.seed", 42))),
            )
        )
    return specs


def make_estimator(spec: MetaModelSpec, split: int) -> Pipeline:
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
            "max_leaf_nodes": 31,
            "min_samples_leaf": 160,
            "l2_regularization": 3.0,
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


def build_feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"meta feature columns missing after generation: {missing}")
    return frame[features].to_numpy(dtype=np.float32, copy=True)


def weighted_sum(frame: pd.DataFrame, weights: dict[str, Any]) -> np.ndarray:
    total = float(sum(float(value) for value in weights.values()))
    if total <= 0:
        raise ValueError("blend weights must have positive sum")
    pred = np.zeros(len(frame), dtype=float)
    for column, weight in weights.items():
        pred += frame[str(column)].to_numpy(dtype=float) * (float(weight) / total)
    return pred


def control_predictions(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for item in get_nested(config, "audit.controls", []):
        name = str(item["name"])
        method = str(item["method"])
        if method == "column":
            output[name] = frame[str(item["column"])].to_numpy(dtype=float)
        elif method == "blend":
            output[name] = weighted_sum(frame, dict(item.get("weights") or {}))
        else:
            raise ValueError(f"unsupported control method: {method}")
    return output


def fixed_blend_predictions(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for item in get_nested(config, "audit.fixed_blends", []):
        output[str(item["name"])] = weighted_sum(frame, dict(item.get("weights") or {}))
    return output


def cross_fit_meta_predictions(
    *,
    audit: str,
    frame: pd.DataFrame,
    x_matrix: np.ndarray,
    y_residual: np.ndarray,
    base_pred: np.ndarray,
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    specs: list[MetaModelSpec],
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
                name = f"{spec.name}_shrink{safe_name(shrink)}_clip{safe_name(clip_value)}"
                predictions[name] = base_pred + shrink * residual
        print(
            json.dumps(
                {"audit": audit, "model": spec.name, "train_rows_by_split": train_rows_by_split},
                sort_keys=True,
            )
        )
    return predictions


def aggregate_global(
    *,
    audit: str,
    candidate: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    reference_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    score = rmse(y_true, pred)
    row = {"audit": audit, "candidate": candidate, "rmse": round(score, 6), "rows": len(y_true)}
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
    specs: list[MetaModelSpec],
    max_wells: int | None,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    exp026_pred, source_rows = generate_exp026_anchor_predictions(
        audit=audit,
        frame=frame,
        config=config,
        paths=paths,
        split_codes=split_codes,
        max_wells=max_wells,
    )
    audit_frame = add_generated_features(frame, exp026_pred)
    y_true = audit_frame["target_tvt"].to_numpy(dtype=float)
    base_pred = audit_frame["exp026_pred"].to_numpy(dtype=float)
    y_residual = y_true - base_pred
    features = [str(value) for value in get_nested(config, "model.features", [])]
    x_matrix = build_feature_matrix(audit_frame, features)

    controls = control_predictions(audit_frame, config)
    fixed_blends = fixed_blend_predictions(audit_frame, config)
    meta_predictions = cross_fit_meta_predictions(
        audit=audit,
        frame=audit_frame,
        x_matrix=x_matrix,
        y_residual=y_residual,
        base_pred=base_pred,
        split_codes=split_codes,
        bucket_code_values=bucket_code_values,
        specs=specs,
    )
    predictions = {
        "control": controls,
        "fixed_blend": fixed_blends,
        "meta": meta_predictions,
    }

    default_required = ["exp026_pseudo_tail_bucket_shrink"]
    required_control_names = [
        str(value) for value in get_nested(config, "audit.required_controls", default_required)
    ]
    for name in required_control_names:
        if name not in controls:
            raise ValueError(f"required control {name!r} is not defined")
    reference_name = str(
        get_nested(config, "audit.reference_control", "exp026_pseudo_tail_bucket_shrink")
    )
    if reference_name not in controls:
        raise ValueError(f"reference control {reference_name!r} is not defined")
    reference_pred = controls[reference_name]

    metric_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    well_metric_frames: list[pd.DataFrame] = []
    flat_predictions: dict[str, np.ndarray] = {}
    for group_name, group_predictions in predictions.items():
        for candidate, pred in group_predictions.items():
            full_candidate = candidate if group_name == "control" else candidate
            flat_predictions[full_candidate] = pred
            metric_rows.append(
                aggregate_global(
                    audit=audit,
                    candidate=full_candidate,
                    pred=pred,
                    y_true=y_true,
                    reference_predictions={name: controls[name] for name in required_control_names},
                )
            )
            bucket_rows.extend(
                aggregate_by_code(
                    audit=audit,
                    candidate=full_candidate,
                    pred=pred,
                    y_true=y_true,
                    codes=bucket_code_values,
                    code_name="bucket",
                    labels=bucket_labels,
                    reference=reference_pred,
                )
            )
            split_rows.extend(
                aggregate_by_code(
                    audit=audit,
                    candidate=full_candidate,
                    pred=pred,
                    y_true=y_true,
                    codes=split_codes,
                    code_name="split",
                    labels=split_labels,
                    reference=reference_pred,
                )
            )
            well_metric_frames.append(
                aggregate_well_metrics(
                    audit=audit,
                    candidate=full_candidate,
                    frame=audit_frame,
                    pred=pred,
                    y_true=y_true,
                    reference=reference_pred,
                )
            )

    return (
        predictions,
        pd.DataFrame(metric_rows),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(split_rows),
        pd.concat(well_metric_frames, ignore_index=True),
        source_rows,
    )


def run_audit(
    paths: ExperimentPaths,
    config: dict[str, Any],
    feature_path: Path,
    output_dir: Path | None = None,
    *,
    max_wells: int | None = None,
    max_train_rows_override: int | None = None,
    skip_hgb: bool = False,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_dir = output_dir or paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_wells = None
    if max_wells is not None:
        allowed_wells = set(file_by_well(paths, max_wells))

    frame, loaded_columns = load_features(feature_path, config, allowed_wells=allowed_wells)
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")
    bucket_code_values = bucket_codes(frame["eval_step"].to_numpy(dtype=float), buckets)
    bucket_labels = [str(bucket["name"]) for bucket in buckets]
    specs = meta_specs(
        config,
        max_train_rows_override=max_train_rows_override,
        skip_hgb=skip_hgb,
    )

    original_folds = sorted(int(value) for value in frame["fold"].unique())
    original_fold_map = {fold: idx for idx, fold in enumerate(original_folds)}
    original_fold_codes = frame["fold"].map(original_fold_map).to_numpy(dtype=np.int16)
    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    well_hash_codes = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_holdout_folds)
    ).to_numpy(dtype=np.int16)

    audit_outputs = {
        "leave_one_original_fold_out": (original_fold_codes, original_folds),
        "well_hash_holdout": (well_hash_codes, list(range(well_holdout_folds))),
    }
    metric_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for audit_name, (split_codes, split_labels) in audit_outputs.items():
        _, metrics, bucket_metrics, split_metrics, well_metrics, audit_source_rows = run_one_audit(
            audit=audit_name,
            frame=frame,
            config=config,
            paths=paths,
            split_codes=split_codes,
            split_labels=split_labels,
            bucket_code_values=bucket_code_values,
            bucket_labels=bucket_labels,
            specs=specs,
            max_wells=max_wells,
        )
        metric_frames.append(metrics)
        bucket_frames.append(bucket_metrics)
        split_frames.append(split_metrics)
        well_frames.append(well_metrics)
        source_rows.extend(audit_source_rows)

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["audit", "rmse"])
    bucket_metrics = pd.concat(bucket_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "bucket"]
    )
    split_metrics = pd.concat(split_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "split"]
    )
    well_metrics = pd.concat(well_frames, ignore_index=True)
    source_summary = pd.DataFrame(source_rows)

    default_required = ["exp026_pseudo_tail_bucket_shrink"]
    required_control_names = [
        str(value) for value in get_nested(config, "audit.required_controls", default_required)
    ]
    required_control_by_audit = {}
    for audit_name, audit_metrics in metrics.groupby("audit"):
        controls = audit_metrics[audit_metrics["candidate"].isin(required_control_names)]
        required_control_by_audit[str(audit_name)] = round(float(controls["rmse"].min()), 6)

    original = metrics[metrics["audit"] == "leave_one_original_fold_out"]
    well_hash = metrics[metrics["audit"] == "well_hash_holdout"]
    original_by_candidate = dict(zip(original["candidate"], original["rmse"], strict=False))
    well_hash_by_candidate = dict(zip(well_hash["candidate"], well_hash["rmse"], strict=False))
    supported: list[dict[str, Any]] = []
    for candidate, original_rmse in original_by_candidate.items():
        if candidate in required_control_names or candidate not in well_hash_by_candidate:
            continue
        well_rmse = float(well_hash_by_candidate[candidate])
        original_rmse = float(original_rmse)
        if (
            original_rmse < required_control_by_audit["leave_one_original_fold_out"]
            and well_rmse < required_control_by_audit["well_hash_holdout"]
        ):
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
    summary = {
        "experiment": "exp034_public_sel15_pf_meta_stack",
        "status": "completed" if selected is not None else "implemented_no_supported_candidate_yet",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "lineage.parent"),
        "feature_file": feature_path.as_posix(),
        "loaded_columns": loaded_columns,
        "feature_columns": [str(value) for value in get_nested(config, "model.features", [])],
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "max_wells": max_wells,
        "metric": "rmse",
        "target": "target_tvt_minus_exp026_anchor",
        "parent_exp026_clean_cv": get_nested(config, "audit.parent_exp026_clean_cv"),
        "reference_control": get_nested(config, "audit.reference_control"),
        "required_controls": required_control_names,
        "required_control_by_audit": required_control_by_audit,
        "best_original_fold_candidate": best_original["candidate"],
        "best_original_fold_cv": round(float(best_original["rmse"]), 6),
        "best_well_hash_candidate": best_well_hash["candidate"],
        "best_well_hash_cv": round(float(best_well_hash["rmse"]), 6),
        "supported_candidates": supported,
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_clean_cv": None if selected is None else selected["original_fold_rmse"],
        "meta_stack_supported": selected is not None,
        "model_specs": [spec.__dict__ for spec in specs],
        "notes": (
            "A meta-stack candidate beat required controls in both holdout audits."
            if selected is not None
            else (
                "No meta-stack candidate beat the exp026/PF required controls "
                "in both holdout audits."
            )
        ),
    }

    metrics.to_csv(output_dir / "meta_stack_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / "meta_stack_bucket_metrics.csv", index=False)
    split_metrics.to_csv(output_dir / "meta_stack_split_metrics.csv", index=False)
    well_metrics.to_csv(output_dir / "meta_stack_well_metrics.csv", index=False)
    source_summary.to_csv(output_dir / "meta_stack_exp026_source_summary.csv", index=False)
    with (output_dir / "meta_stack_summary.json").open("w") as fp:
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
        skip_hgb=args.skip_hgb,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
