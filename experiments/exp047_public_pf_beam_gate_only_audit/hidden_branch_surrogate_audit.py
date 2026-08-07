from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from meta_stack_audit import (
    MetaModelSpec,
    add_generated_features,
    bucket_codes,
    build_feature_matrix,
    control_predictions,
    file_by_well,
    generate_exp026_anchor_predictions,
    get_nested,
    load_features,
    make_estimator,
    resolve_feature_path,
    rmse,
    rmse_from_sse,
    stable_fold,
)
from settings import ExperimentPaths
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

HORIZONTAL_SUFFIX = "__horizontal_well.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit hidden-branch candidates on train-side surrogate rows."
    )
    parser.add_argument("--features", default=None, help="Path to exp029 feature CSV")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke well limit")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Override residual/meta row caps",
    )
    parser.add_argument(
        "--base-estimator",
        choices=["LGBMRegressor", "HistGradientBoostingRegressor"],
        default=None,
        help="Override exp026 anchor estimator for local smoke checks.",
    )
    return parser.parse_args()


def load_local_config() -> dict[str, Any]:
    with Path(__file__).with_name("config.yaml").open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return value


def safe_name(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p").replace("-", "m")


def weighted_sum(frame: pd.DataFrame, weights: dict[str, Any]) -> np.ndarray:
    total = float(sum(float(value) for value in weights.values()))
    if total <= 0:
        raise ValueError("blend weights must have positive sum")
    pred = np.zeros(len(frame), dtype=float)
    for column, weight in weights.items():
        pred += frame[str(column)].to_numpy(dtype=float) * (float(weight) / total)
    return pred


def qbin(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index, dtype=object)
    q = max(1, min(n_bins, int(values.notna().sum()), int(values.nunique(dropna=True))))
    try:
        bins = pd.qcut(values.rank(method="first"), q=q, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(values, bins=q, labels=False, include_lowest=True)
    out = bins.astype("Int64").astype(str)
    out = out.where(values.notna(), "missing")
    return prefix + "_" + out


def fixed_azimuth_bin(series: pd.Series, n_bins: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    bins = pd.cut(values, bins=edges, include_lowest=True, labels=False)
    out = bins.astype("Int64").astype(str)
    out = out.where(values.notna(), "missing")
    return "az_" + out


def well_id_from_path(path: Path) -> str:
    return path.name.split(HORIZONTAL_SUFFIX)[0]


def summarize_well(path: Path) -> dict[str, Any]:
    well_id = well_id_from_path(path)
    df = pd.read_csv(path)
    eval_mask = df["TVT_input"].isna()
    known_mask = df["TVT_input"].notna()
    eval_indices = np.flatnonzero(eval_mask.to_numpy())
    first_eval = int(eval_indices[0]) if len(eval_indices) else -1
    prefix_known = (
        df.loc[: max(first_eval - 1, 0), "TVT_input"]
        if first_eval > 0
        else df["TVT_input"].iloc[:0]
    )
    known_tvt = df.loc[known_mask, "TVT_input"]
    if prefix_known.notna().any():
        last_known_tvt = float(prefix_known.dropna().iloc[-1])
        median_known_tvt = float(prefix_known.dropna().median())
    elif known_tvt.notna().any():
        last_known_tvt = float(known_tvt.iloc[-1])
        median_known_tvt = float(known_tvt.median())
    else:
        last_known_tvt = float("nan")
        median_known_tvt = float("nan")

    dx = float(df["X"].iloc[-1] - df["X"].iloc[0])
    dy = float(df["Y"].iloc[-1] - df["Y"].iloc[0])
    return {
        "well_id": well_id,
        "n_rows": int(len(df)),
        "eval_rows": int(eval_mask.sum()),
        "known_rows": int(known_mask.sum()),
        "first_eval_row": first_eval,
        "prefix_length": int(first_eval) if first_eval >= 0 else int(len(df)),
        "eval_length": int(eval_mask.sum()),
        "centroid_x": float(df["X"].mean()),
        "centroid_y": float(df["Y"].mean()),
        "signed_azimuth_deg": math.degrees(math.atan2(dy, dx)),
        "median_known_tvt": median_known_tvt,
        "last_known_tvt": last_known_tvt,
        "median_full_tvt": float(df["TVT"].median()) if "TVT" in df else float("nan"),
        "gr_coverage": float(df["GR"].notna().mean()) if "GR" in df else float("nan"),
    }


def collapse_rare_labels(labels: pd.Series, n_folds: int) -> pd.Series:
    collapsed = labels.astype(str).copy()
    while True:
        counts = collapsed.value_counts()
        rare_labels = counts[counts < n_folds].index
        if len(rare_labels) == 0:
            return collapsed
        if len(rare_labels) == len(counts):
            return pd.Series(["rare"] * len(collapsed), index=collapsed.index, dtype=object)
        collapsed = collapsed.where(~collapsed.isin(rare_labels), "rare")


def build_well_metadata(paths: ExperimentPaths, max_wells: int | None) -> pd.DataFrame:
    train_paths = list(file_by_well(paths, max_wells).values())
    if not train_paths:
        raise FileNotFoundError(f"No train files found in {paths.train_data_dir}")
    meta = pd.DataFrame(summarize_well(path) for path in train_paths)
    meta = meta.sort_values("well_id").reset_index(drop=True)
    meta["azimuth_bin"] = fixed_azimuth_bin(meta["signed_azimuth_deg"], 4)
    meta["tvt_bin"] = qbin(meta["median_known_tvt"].fillna(meta["median_full_tvt"]), 4, "tvt")
    meta["x_bin"] = qbin(meta["centroid_x"], 3, "x")
    meta["y_bin"] = qbin(meta["centroid_y"], 3, "y")
    meta["spatial_bin"] = meta["x_bin"].astype(str) + "__" + meta["y_bin"].astype(str)
    meta["eval_length_bin"] = qbin(meta["eval_length"], 3, "eval_len")
    meta["gr_bin"] = qbin(meta["gr_coverage"], 3, "gr")
    label = (
        meta["azimuth_bin"].astype(str)
        + "|"
        + meta["tvt_bin"].astype(str)
        + "|"
        + meta["spatial_bin"].astype(str)
        + "|"
        + meta["eval_length_bin"].astype(str)
        + "|"
        + meta["gr_bin"].astype(str)
    )
    return meta.assign(strat_label=label)


def assign_metadata_folds(meta: pd.DataFrame, requested_folds: int, seed: int) -> pd.DataFrame:
    out = meta.copy()
    n_folds = min(int(requested_folds), int(len(out)))
    if n_folds < 2:
        out["groupkfold_fold"] = 0
        out["stratified_groupkfold_fold"] = 0
        out["strat_label"] = "all"
        return out

    x = out[["well_id"]]
    groups = out["well_id"].to_numpy()
    out["groupkfold_fold"] = -1
    out["stratified_groupkfold_fold"] = -1
    for fold, (_, valid_idx) in enumerate(GroupKFold(n_splits=n_folds).split(x, groups=groups)):
        out.loc[out.index[valid_idx], "groupkfold_fold"] = fold

    y = collapse_rare_labels(out["strat_label"], n_folds)
    out["strat_label"] = y
    try:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_idx) in enumerate(splitter.split(x, y=y, groups=groups)):
            out.loc[out.index[valid_idx], "stratified_groupkfold_fold"] = fold
    except ValueError:
        out["stratified_groupkfold_fold"] = out["groupkfold_fold"]
        out["strat_label"] = "fallback_groupkfold"
    return out


def load_or_build_metadata(
    paths: ExperimentPaths,
    config: dict[str, Any],
    max_wells: int | None,
) -> pd.DataFrame:
    configured = get_nested(config, "data.optional_stratified_metadata_path")
    if configured and max_wells is None:
        path = Path(str(configured))
        if not path.is_absolute():
            path = paths.root / path
        if path.exists():
            return pd.read_csv(path)
    seed = int(get_nested(config, "validation.seed", 42))
    n_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    return assign_metadata_folds(build_well_metadata(paths, max_wells), n_folds, seed)


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
        parts: list[np.ndarray] = []
        for bucket in np.unique(bucket_code_values[train_idx]):
            bucket_idx = train_idx[bucket_code_values[train_idx] == bucket]
            if len(bucket_idx) > max_rows_per_bucket:
                bucket_idx = rng.choice(bucket_idx, size=max_rows_per_bucket, replace=False)
            parts.append(np.asarray(bucket_idx, dtype=np.int64))
        train_idx = np.concatenate(parts)
    if max_rows is not None and len(train_idx) > max_rows:
        train_idx = rng.choice(train_idx, size=max_rows, replace=False)
    return np.asarray(np.sort(train_idx), dtype=np.int64)


def branch_model_spec(branch: dict[str, Any], max_train_rows_override: int | None) -> MetaModelSpec:
    max_train_rows = branch.get("max_train_rows")
    if max_train_rows_override is not None:
        max_train_rows = max_train_rows_override
    return MetaModelSpec(
        name=str(branch["name"]),
        estimator=str(branch.get("estimator", "ridge")),
        params=dict(branch.get("params") or {}),
        residual_shrink_values=(float(branch.get("residual_shrink", 1.0)),),
        residual_clip_values=(
            None if branch.get("residual_clip") is None else float(branch["residual_clip"]),
        ),
        target_clip=None if branch.get("target_clip") is None else float(branch["target_clip"]),
        max_train_rows=None if max_train_rows is None else int(max_train_rows),
        max_train_rows_per_bucket=(
            None
            if branch.get("max_train_rows_per_bucket") is None
            else int(branch["max_train_rows_per_bucket"])
        ),
        seed=int(branch.get("seed", 42)),
    )


def cross_fit_branch(
    *,
    branch: dict[str, Any],
    frame: pd.DataFrame,
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    max_train_rows_override: int | None,
) -> np.ndarray:
    spec = branch_model_spec(branch, max_train_rows_override)
    feature_columns = [str(value) for value in branch.get("features", [])]
    x_matrix = build_feature_matrix(frame, feature_columns)
    base = frame[str(branch["base_column"])].to_numpy(dtype=float)
    y_residual = frame["target_tvt"].to_numpy(dtype=float) - base
    output_residual = np.full(len(frame), np.nan, dtype=float)

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
        y_train = y_residual[train_idx]
        if spec.target_clip is not None:
            y_train = np.clip(y_train, -spec.target_clip, spec.target_clip)
        estimator = make_estimator(spec, split)
        estimator.fit(x_matrix[train_idx], y_train)
        output_residual[valid_idx] = estimator.predict(x_matrix[valid_idx])

    if not np.isfinite(output_residual).all():
        raise ValueError(f"{branch['name']}: non-finite residual predictions")
    residual = output_residual
    clip_value = spec.residual_clip_values[0]
    if clip_value is not None:
        residual = np.clip(residual, -clip_value, clip_value)
    return base + spec.residual_shrink_values[0] * residual


def build_controls(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    controls = control_predictions(frame, config)
    controls["visible_train_oracle_surrogate"] = frame["target_tvt"].to_numpy(dtype=float)
    return controls


def build_branch_predictions(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    max_train_rows_override: int | None,
) -> dict[str, np.ndarray]:
    predictions = build_controls(frame, config)
    branches = [dict(item) for item in get_nested(config, "audit.submitted_hidden_branches", [])]
    pending_aliases: list[dict[str, Any]] = []
    for branch in branches:
        kind = str(branch.get("kind"))
        name = str(branch["name"])
        if kind == "alias":
            pending_aliases.append(branch)
        elif kind in {"pf_residual", "exp026_meta_residual"}:
            predictions[name] = cross_fit_branch(
                branch=branch,
                frame=frame,
                split_codes=split_codes,
                bucket_code_values=bucket_code_values,
                max_train_rows_override=max_train_rows_override,
            )
        else:
            raise ValueError(f"unsupported branch kind for {name}: {kind}")

    for branch in pending_aliases:
        name = str(branch["name"])
        source = str(branch["source"])
        if source not in predictions:
            raise ValueError(f"alias {name} references missing source {source}")
        predictions[name] = predictions[source].copy()
    return predictions


def segment_rows(
    *,
    audit: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    reference: np.ndarray,
    split_codes: np.ndarray,
    split_labels: list[Any],
    bucket_code_values: np.ndarray,
    bucket_labels: list[str],
) -> list[dict[str, Any]]:
    segment_specs: list[tuple[str, pd.Series]] = [
        ("overall", pd.Series(["all"] * len(frame), index=frame.index, dtype=object)),
        (
            "audit_split",
            pd.Series([str(split_labels[int(code)]) for code in split_codes], index=frame.index),
        ),
        (
            "distance_bucket",
            pd.Series(
                [bucket_labels[int(code)] for code in bucket_code_values],
                index=frame.index,
            ),
        ),
    ]
    for column in [
        "groupkfold_fold",
        "stratified_groupkfold_fold",
        "azimuth_bin",
        "tvt_bin",
        "spatial_bin",
        "eval_length_bin",
        "gr_bin",
        "strat_label",
    ]:
        if column in frame.columns:
            segment_specs.append((column, frame[column].astype(str)))

    y_true = frame["target_tvt"].to_numpy(dtype=float)
    ref_score_by_segment: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for segment_type, labels in segment_specs:
        for segment, idx in labels.groupby(labels, sort=True).groups.items():
            idx_array = np.asarray(list(idx), dtype=int)
            ref_score_by_segment[(segment_type, str(segment))] = rmse(
                y_true[idx_array],
                reference[idx_array],
            )

    wells = frame["well_id"].astype(str)
    for candidate, pred in predictions.items():
        for segment_type, labels in segment_specs:
            for segment, idx in labels.groupby(labels, sort=True).groups.items():
                idx_array = np.asarray(list(idx), dtype=int)
                score = rmse(y_true[idx_array], pred[idx_array])
                ref_score = ref_score_by_segment[(segment_type, str(segment))]
                rows.append(
                    {
                        "audit": audit,
                        "candidate": candidate,
                        "segment_type": segment_type,
                        "segment": str(segment),
                        "rmse": round(score, 6),
                        "reference_rmse": round(ref_score, 6),
                        "delta_vs_reference": round(score - ref_score, 6),
                        "rows": int(len(idx_array)),
                        "wells": int(wells.iloc[idx_array].nunique()),
                    }
                )
    return rows


def candidate_metric_rows(
    *,
    audit: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    reference_names: list[str],
) -> list[dict[str, Any]]:
    y_true = frame["target_tvt"].to_numpy(dtype=float)
    reference_scores = {
        name: rmse(y_true, predictions[name])
        for name in reference_names
        if name in predictions
    }
    rows: list[dict[str, Any]] = []
    for candidate, pred in predictions.items():
        score = rmse(y_true, pred)
        row: dict[str, Any] = {
            "audit": audit,
            "candidate": candidate,
            "rmse": round(score, 6),
            "rows": int(len(frame)),
            "wells": int(frame["well_id"].nunique()),
            "pred_min": round(float(np.min(pred)), 6),
            "pred_max": round(float(np.max(pred)), 6),
            "pred_mean": round(float(np.mean(pred)), 6),
        }
        for ref_name, ref_score in reference_scores.items():
            row[f"delta_vs_{ref_name}"] = round(score - ref_score, 6)
        rows.append(row)
    return rows


def branch_diff_rows(
    *,
    audit: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    references: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wells = frame["well_id"].astype(str).to_numpy()
    for candidate, pred in predictions.items():
        for ref_name in references:
            if ref_name not in predictions or ref_name == candidate:
                continue
            diff = pred - predictions[ref_name]
            changed = np.abs(diff) > 1e-9
            rows.append(
                {
                    "audit": audit,
                    "candidate": candidate,
                    "reference": ref_name,
                    "diff_rmse": round(rmse_from_sse(float(np.dot(diff, diff)), len(diff)), 6),
                    "mean_abs_diff": round(float(np.mean(np.abs(diff))), 6),
                    "max_abs_diff": round(float(np.max(np.abs(diff))), 6),
                    "changed_rows": int(changed.sum()),
                    "changed_wells": int(len(set(wells[changed]))),
                    "rows": int(len(diff)),
                    "wells": int(len(set(wells))),
                }
            )
    return rows


def well_metric_rows(
    *,
    audit: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    reference: np.ndarray,
) -> pd.DataFrame:
    base = frame[["well_id", "fold"]].copy()
    y_true = frame["target_tvt"].to_numpy(dtype=float)
    ref_diff2 = np.square(reference - y_true)
    ref_well = (
        base.assign(ref_diff2=ref_diff2)
        .groupby(["well_id", "fold"], sort=False)
        .agg(ref_sse=("ref_diff2", "sum"), rows=("ref_diff2", "size"))
        .reset_index()
    )
    frames: list[pd.DataFrame] = []
    for candidate, pred in predictions.items():
        diff2 = np.square(pred - y_true)
        item = (
            base.assign(diff2=diff2)
            .groupby(["well_id", "fold"], sort=False)
            .agg(sse=("diff2", "sum"), rows=("diff2", "size"))
            .reset_index()
        )
        item = item.merge(ref_well, on=["well_id", "fold"], suffixes=("", "_ref"))
        item["audit"] = audit
        item["candidate"] = candidate
        item["rmse"] = np.sqrt(item["sse"] / item["rows"])
        item["reference_rmse"] = np.sqrt(item["ref_sse"] / item["rows"])
        item["delta_vs_reference"] = item["rmse"] - item["reference_rmse"]
        frames.append(
            item[
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
        )
    return pd.concat(frames, ignore_index=True)


def split_systems(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, list[Any]]]:
    original_folds = sorted(int(value) for value in frame["fold"].unique())
    original_fold_map = {fold: idx for idx, fold in enumerate(original_folds)}
    systems: dict[str, tuple[np.ndarray, list[Any]]] = {
        "leave_one_original_fold_out": (
            frame["fold"].map(original_fold_map).to_numpy(dtype=np.int16),
            original_folds,
        )
    }

    well_holdout_folds = int(get_nested(config, "audit.well_holdout_folds", 5))
    systems["well_hash_holdout"] = (
        frame["well_id"]
        .map(lambda value: stable_fold(str(value), well_holdout_folds))
        .to_numpy(dtype=np.int16),
        list(range(well_holdout_folds)),
    )

    strat_map = (
        metadata[["well_id", "stratified_groupkfold_fold"]]
        .drop_duplicates("well_id")
        .set_index("well_id")["stratified_groupkfold_fold"]
        .astype(int)
        .to_dict()
    )
    strat_codes = frame["well_id"].map(lambda value: strat_map[str(value)]).to_numpy(dtype=np.int16)
    systems["stratified_groupkfold_holdout"] = (
        strat_codes,
        sorted(int(value) for value in np.unique(strat_codes)),
    )
    requested = [str(value) for value in get_nested(config, "audit.split_systems", systems.keys())]
    return {name: systems[name] for name in requested}


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
) -> tuple[
    pd.DataFrame,
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
    predictions = build_branch_predictions(
        frame=audit_frame,
        config=config,
        split_codes=split_codes,
        bucket_code_values=bucket_code_values,
        max_train_rows_override=max_train_rows_override,
    )

    reference_name = str(
        get_nested(config, "audit.reference_control", "exp026_pseudo_tail_bucket_shrink")
    )
    if reference_name not in predictions:
        raise ValueError(f"reference candidate is missing: {reference_name}")
    reference_names = [
        str(value)
        for value in get_nested(
            config,
            "audit.primary_reference_controls",
            ["exp026_pseudo_tail_bucket_shrink", "public_pf_selector", "pf090_hold010"],
        )
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
    return metrics, segments, diffs, wells, source, []


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
    for audit_name, (codes, labels) in split_systems(frame, metadata, config).items():
        metrics, segments, diffs, wells, source, _ = run_one_audit(
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

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["audit", "rmse"])
    segments = pd.concat(segment_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "segment_type", "segment"]
    )
    diffs = pd.concat(diff_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "reference"]
    )
    wells = pd.concat(well_frames, ignore_index=True).sort_values(["audit", "candidate", "well_id"])
    source_summary = pd.concat(source_frames, ignore_index=True)

    reference_name = str(
        get_nested(config, "audit.reference_control", "exp026_pseudo_tail_bucket_shrink")
    )
    best_by_audit = (
        metrics[~metrics["candidate"].eq("visible_train_oracle_surrogate")]
        .sort_values(["audit", "rmse"])
        .groupby("audit", sort=True)
        .head(1)
        .to_dict(orient="records")
    )
    known_outcomes = {
        str(item["name"]): {
            "known_public_lb": item.get("known_public_lb"),
            "public_lb_delta_vs_exp027": item.get("public_lb_delta_vs_exp027"),
        }
        for item in get_nested(config, "audit.submitted_hidden_branches", [])
        if "known_public_lb" in item
    }
    known_failed = [
        name
        for name, outcome in known_outcomes.items()
        if outcome.get("public_lb_delta_vs_exp027") is not None
        and float(outcome["public_lb_delta_vs_exp027"]) > 0
    ]
    summary = {
        "experiment": "exp046_hidden_branch_surrogate_audit",
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
        "reference_control": reference_name,
        "split_systems": list(split_systems(frame, metadata, config)),
        "best_by_audit": best_by_audit,
        "known_hidden_branch_outcomes": known_outcomes,
        "known_hidden_branch_failures": known_failed,
        "public_sample_blind_spot": (
            "visible_train_oracle_surrogate is a diagnostic only; public sample changed_rows=0 "
            "does not validate hidden-branch behavior."
        ),
        "notes": (
            "This audit creates surrogate hidden-branch predictions on train pseudo-hidden rows "
            "and records diff/range/segment red flags. It does not create a submission."
        ),
    }

    metrics.to_csv(output_dir / "hidden_branch_surrogate_metrics.csv", index=False)
    segments.to_csv(output_dir / "hidden_branch_surrogate_segment_metrics.csv", index=False)
    diffs.to_csv(output_dir / "hidden_branch_surrogate_diff_metrics.csv", index=False)
    wells.to_csv(output_dir / "hidden_branch_surrogate_well_metrics.csv", index=False)
    source_summary.to_csv(
        output_dir / "hidden_branch_surrogate_exp026_source_summary.csv",
        index=False,
    )
    metadata.to_csv(output_dir / "hidden_branch_surrogate_well_metadata.csv", index=False)
    with (output_dir / "hidden_branch_surrogate_summary.json").open("w") as fp:
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
