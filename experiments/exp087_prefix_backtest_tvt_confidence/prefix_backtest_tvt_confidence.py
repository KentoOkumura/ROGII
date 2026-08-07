from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    well_column: str
    target_column: str
    target_delta_column: str | None
    base_column: str | None
    x_column: str
    cutoff_column: str | None
    selected_cutoff: float | None
    candidate_columns: list[dict[str, str]]
    confidence_columns: list[str]
    disagreement_columns: dict[str, str]


@dataclass(frozen=True)
class ConfidenceModel:
    feature_columns: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(n_folds)


def find_source_path(root: Path, source_config: dict[str, Any]) -> Path:
    local_path = source_config.get("local_path")
    if local_path:
        candidate = Path(str(local_path))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate

    slug = str(source_config.get("kaggle_source_slug") or "").strip()
    relative_paths = [str(item) for item in source_config.get("relative_paths", [])]
    if KAGGLE_INPUT_ROOT.exists():
        input_roots: list[Path] = []
        if slug:
            input_roots.append(KAGGLE_INPUT_ROOT / slug)
        input_roots.extend(path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir())
        seen: set[Path] = set()
        for input_root in input_roots:
            if input_root in seen or not input_root.exists():
                continue
            seen.add(input_root)
            for relative_path in relative_paths:
                candidate = input_root / relative_path
                if candidate.exists():
                    return candidate
            for filename in (
                "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
                "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv",
            ):
                matches = sorted(input_root.rglob(filename))
                if matches:
                    return matches[0]

    raise FileNotFoundError(
        "PF/Beam source artifact not found. Checked local_path and Kaggle input source "
        f"for source={source_config.get('name')!r}."
    )


def build_source_spec(config: dict[str, Any], root: Path) -> SourceSpec:
    source_name = str(get_nested(config, "audit.source_name") or "")
    sources = get_nested(config, "data.pfbeam_sources") or []
    source_config = next((item for item in sources if item.get("name") == source_name), None)
    if source_config is None:
        raise ValueError(f"data.pfbeam_sources does not contain audit.source_name={source_name!r}")

    selected_cutoff = source_config.get("selected_cutoff")
    return SourceSpec(
        name=str(source_config["name"]),
        path=find_source_path(root, source_config),
        well_column=str(source_config.get("well_column", "well_id")),
        target_column=str(source_config.get("target_column", "target_tvt")),
        target_delta_column=source_config.get("target_delta_column"),
        base_column=source_config.get("base_column"),
        x_column=str(source_config.get("x_column", "MD")),
        cutoff_column=source_config.get("cutoff_column"),
        selected_cutoff=float(selected_cutoff) if selected_cutoff is not None else None,
        candidate_columns=list(source_config.get("candidate_columns", [])),
        confidence_columns=list(source_config.get("confidence_columns", [])),
        disagreement_columns=dict(source_config.get("disagreement_columns", {})),
    )


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def materialize_tvt_columns(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    if source.target_column not in frame:
        if not source.base_column or not source.target_delta_column:
            raise ValueError(
                f"{source.target_column!r} is missing and base/target delta columns are not set."
            )
        missing = [
            col for col in (source.base_column, source.target_delta_column) if col not in frame
        ]
        if missing:
            raise ValueError(f"Cannot materialize true TVT; missing columns: {missing}")
        frame[source.target_column] = numeric_series(frame, source.base_column) + numeric_series(
            frame, source.target_delta_column
        )

    base = numeric_series(frame, source.base_column) if source.base_column else None
    for spec in source.candidate_columns:
        name = spec.get("name")
        source_column = spec.get("source_column") or name
        transform = spec.get("transform", "absolute")
        if not name or not source_column or source_column not in frame:
            continue
        values = numeric_series(frame, source_column)
        if transform == "base_plus_delta":
            if base is None:
                raise ValueError(f"Candidate {name!r} requires base_plus_delta without base.")
            frame[name] = base + values
        elif transform == "absolute":
            frame[name] = values
        else:
            raise ValueError(f"Unsupported candidate transform for {name!r}: {transform!r}")

    if {"pf_ancc", "beam_mean"}.issubset(frame.columns):
        diff = numeric_series(frame, "pf_ancc") - numeric_series(frame, "beam_mean")
        frame["pf_ancc_vs_beam_mean"] = diff
        frame["pf_ancc_vs_beam_mean_abs"] = diff.abs()
    return frame


def source_usecols(source: SourceSpec) -> list[str]:
    columns = {"id", source.well_column, source.target_column, source.x_column}
    for column in (source.target_delta_column, source.base_column, source.cutoff_column):
        if column:
            columns.add(column)
    for spec in source.candidate_columns:
        name = spec.get("name")
        source_column = spec.get("source_column") or name
        if name:
            columns.add(name)
        if source_column:
            columns.add(source_column)
    columns.update(source.confidence_columns)
    columns.update(source.disagreement_columns.values())
    return sorted(columns)


def downcast_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if column in {"id"}:
            continue
        if frame[column].dtype == object:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() > 0:
                frame[column] = converted
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], downcast="float")
        elif pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], downcast="integer")
    return frame


def read_source_frame(source: SourceSpec, *, debug_max_wells: int | None = None) -> pd.DataFrame:
    dtype = {"id": str, source.well_column: str}
    available_columns = pd.read_csv(source.path, nrows=0).columns.tolist()
    requested_columns = source_usecols(source)
    usecols = [column for column in requested_columns if column in available_columns]
    missing_required = [
        column
        for column in (source.well_column, source.target_delta_column, source.base_column)
        if column and column not in usecols
    ]
    if missing_required:
        raise ValueError(f"Source frame missing required columns: {missing_required}")

    frame = pd.read_csv(source.path, dtype=dtype, usecols=usecols, low_memory=False)
    frame[source.well_column] = frame[source.well_column].astype(str)
    frame = downcast_numeric_columns(frame)
    frame[source.well_column] = frame[source.well_column].astype(str)
    frame = materialize_tvt_columns(frame, source)
    if (
        source.cutoff_column
        and source.selected_cutoff is not None
        and source.cutoff_column in frame
    ):
        cutoff_values = numeric_series(frame, source.cutoff_column)
        frame = frame[np.isclose(cutoff_values, float(source.selected_cutoff), atol=1e-9)].copy()
    if debug_max_wells is not None and debug_max_wells > 0:
        wells = sorted(frame[source.well_column].dropna().astype(str).unique())[
            : int(debug_max_wells)
        ]
        frame = frame[frame[source.well_column].astype(str).isin(wells)].copy()
    return frame


def rmse(pred: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> float:
    pred_values = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float)
    delta = pred_values - target_values
    finite = np.isfinite(delta)
    if not bool(finite.any()):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(delta[finite]))))


def mae(pred: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> float:
    pred_values = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float)
    delta = pred_values - target_values
    finite = np.isfinite(delta)
    if not bool(finite.any()):
        return float("nan")
    return float(np.mean(np.abs(delta[finite])))


def add_confidence_features(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    frame = frame.copy()
    frame["beam_std_abs"] = numeric_series(frame, "beam_std_d").abs()
    frame["pf_beam_abs"] = numeric_series(frame, "pf_ancc") - numeric_series(frame, "beam_mean")
    frame["pf_beam_abs"] = frame["pf_beam_abs"].abs()
    frame["pf_likpf_abs"] = numeric_series(frame, "pf_ancc") - numeric_series(frame, "likpf_mean")
    frame["pf_likpf_abs"] = frame["pf_likpf_abs"].abs()
    frame["beam_likpf_abs"] = numeric_series(frame, "beam_mean") - numeric_series(
        frame, "likpf_mean"
    )
    frame["beam_likpf_abs"] = frame["beam_likpf_abs"].abs()
    frame["likpf_delta_abs"] = numeric_series(frame, "likpf_mean_d").abs()
    known_len = numeric_series(frame, "known_len").replace(0, np.nan)
    frame["eval_to_known_ratio"] = numeric_series(frame, "eval_len") / known_len
    if source.x_column in frame:
        frame[source.x_column] = numeric_series(frame, source.x_column)
    return frame


def assign_backtest_phase(
    frame: pd.DataFrame, source: SourceSpec, *, calibration_fraction: float
) -> pd.DataFrame:
    frame = frame.copy()
    x_column = source.x_column if source.x_column in frame else None
    phase = pd.Series("holdout", index=frame.index, dtype=object)
    row_fraction = pd.Series(np.nan, index=frame.index, dtype=float)

    for _, group in frame.groupby(source.well_column, sort=False):
        if x_column:
            order = group.sort_values(x_column).index
        else:
            order = group.index
        n_rows = len(order)
        if n_rows <= 1:
            frac = np.zeros(n_rows, dtype=float)
        else:
            frac = np.linspace(0.0, 1.0, n_rows)
        row_fraction.loc[order] = frac
        phase.loc[order[frac <= calibration_fraction]] = "calibration"

    frame["prefix_backtest_fraction"] = row_fraction
    frame["prefix_backtest_phase"] = phase
    return frame


def distance_bucket(values: pd.Series, bucket_config: list[dict[str, Any]]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.Series("missing", index=values.index, dtype=object)
    previous_max = -np.inf
    for bucket in bucket_config:
        max_value = float(bucket["max_value"])
        name = str(bucket["name"])
        mask = (numeric > previous_max) & (numeric <= max_value)
        labels.loc[mask] = name
        previous_max = max_value
    return labels


def model_matrix(
    frame: pd.DataFrame, feature_columns: list[str], medians: np.ndarray
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for idx, column in enumerate(feature_columns):
        values = numeric_series(frame, column).to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, medians[idx])
        columns.append(values)
    if not columns:
        return np.zeros((len(frame), 0), dtype=float)
    return np.vstack(columns).T.astype(float)


def fit_confidence_model(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    ridge_lambda: float,
) -> ConfidenceModel:
    medians = np.array(
        [
            float(numeric_series(frame, column).median())
            if bool(np.isfinite(numeric_series(frame, column)).any())
            else 0.0
            for column in feature_columns
        ],
        dtype=float,
    )
    x = model_matrix(frame, feature_columns, medians)
    means = np.nanmean(x, axis=0) if x.size else np.zeros(0, dtype=float)
    scales = np.nanstd(x, axis=0) if x.size else np.zeros(0, dtype=float)
    scales = np.where(scales > 1e-9, scales, 1.0)
    x_scaled = (x - means) / scales
    design = np.column_stack([np.ones(len(x_scaled), dtype=float), x_scaled])

    y = np.log1p(numeric_series(frame, target_column).abs().to_numpy(dtype=float))
    finite = np.isfinite(y) & np.isfinite(design).all(axis=1)
    if int(finite.sum()) < max(2, len(feature_columns) + 1):
        coefficients = np.zeros(len(feature_columns) + 1, dtype=float)
        coefficients[0] = (
            float(np.nanmedian(y[np.isfinite(y)]))
            if bool(np.isfinite(y).any())
            else 0.0
        )
        return ConfidenceModel(feature_columns, medians, means, scales, coefficients)

    x_fit = design[finite]
    y_fit = y[finite]
    penalty = np.eye(x_fit.shape[1], dtype=float) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x_fit.T @ x_fit + penalty, x_fit.T @ y_fit)
    return ConfidenceModel(feature_columns, medians, means, scales, coefficients)


def predict_expected_error(
    model: ConfidenceModel, frame: pd.DataFrame, *, clip: float
) -> np.ndarray:
    x = model_matrix(frame, model.feature_columns, model.medians)
    x_scaled = (x - model.means) / model.scales
    design = np.column_stack([np.ones(len(x_scaled), dtype=float), x_scaled])
    pred_log = design @ model.coefficients
    pred = np.expm1(np.clip(pred_log, 0.0, math.log1p(float(clip))))
    return np.clip(pred, 0.0, float(clip))


def candidate_metrics(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = frame[source.target_column]
    for spec in source.candidate_columns:
        name = spec.get("name")
        if not name or name not in frame:
            continue
        rows.append(
            {
                "candidate": name,
                "role": spec.get("role"),
                "rows": int(numeric_series(frame, name).notna().sum()),
                "rmse": rmse(frame[name], target),
                "mae": mae(frame[name], target),
                "bias": float(
                    (
                        numeric_series(frame, name)
                        - numeric_series(frame, source.target_column)
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_group(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    primary_error_col: str,
    expected_col: str,
    high_error_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys, strict=False)}
        errors = numeric_series(group, primary_error_col).abs()
        expected = numeric_series(group, expected_col)
        row.update(
            {
                "rows": int(len(group)),
                "wells": int(group["well_id"].nunique()) if "well_id" in group else None,
                "observed_mae": float(errors.mean()),
                "observed_rmse_abs_error": float(np.sqrt(np.nanmean(np.square(errors)))),
                "expected_error_mean": float(expected.mean()),
                "expected_error_median": float(expected.median()),
                "high_error_rate": float(
                    pd.to_numeric(group[high_error_col], errors="coerce").mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def signal_correlations(
    frame: pd.DataFrame, *, feature_columns: list[str], primary_abs_error_col: str
) -> pd.DataFrame:
    target = numeric_series(frame, primary_abs_error_col).abs()
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        values = numeric_series(frame, column)
        valid = np.isfinite(values) & np.isfinite(target)
        if int(valid.sum()) < 3:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(np.corrcoef(values[valid], target[valid])[0, 1])
            spearman = float(values[valid].rank().corr(target[valid].rank()))
        rows.append(
            {
                "signal": column,
                "rows": int(valid.sum()),
                "pearson_abs_error": pearson,
                "spearman_abs_error": spearman,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_abs_error", key=lambda s: s.abs(), ascending=False
    )


def assign_confidence_bins(
    frame: pd.DataFrame, *, expected_col: str, n_bins: int
) -> pd.Series:
    values = numeric_series(frame, expected_col)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series("missing", index=frame.index, dtype=object)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.nanquantile(finite.to_numpy(dtype=float), quantiles))
    if len(edges) <= 2:
        median = float(finite.median())
        return pd.Series(
            np.where(values <= median, "bin_0_low", "bin_1_high"),
            index=frame.index,
            dtype=object,
        )
    labels = [
        f"bin_{idx}_{'low' if idx == 0 else 'high' if idx == len(edges) - 2 else 'mid'}"
        for idx in range(len(edges) - 1)
    ]
    return (
        pd.cut(values, bins=edges, labels=labels, include_lowest=True)
        .astype(object)
        .fillna("missing")
    )


def build_audit_frame(
    frame: pd.DataFrame, source: SourceSpec, config: dict[str, Any]
) -> pd.DataFrame:
    params = get_nested(config, "model.params") or {}
    primary = str(params.get("primary_candidate", "pf_ancc"))
    if primary not in frame:
        raise ValueError(f"Primary candidate {primary!r} is missing from source frame.")
    if source.target_column not in frame:
        raise ValueError(f"Target column {source.target_column!r} is missing.")

    frame = add_confidence_features(frame, source)
    frame = assign_backtest_phase(
        frame,
        source,
        calibration_fraction=float(params.get("calibration_fraction", 0.35)),
    )
    x_column = source.x_column if source.x_column in frame else "prefix_backtest_fraction"
    frame["distance_bucket"] = distance_bucket(
        numeric_series(frame, x_column), list(params.get("distance_buckets", []))
    )
    frame["well_id"] = frame[source.well_column].astype(str)
    frame["primary_candidate"] = primary
    frame["primary_error"] = numeric_series(frame, primary) - numeric_series(
        frame, source.target_column
    )
    frame["primary_abs_error"] = frame["primary_error"].abs()
    frame["well_hash_fold"] = frame["well_id"].map(
        lambda value: stable_fold(str(value), int(params.get("well_hash_folds", 5)))
    )
    return frame


def fold_safe_confidence_predictions(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    params = get_nested(config, "model.params") or {}
    feature_columns = [
        column for column in params.get("feature_columns", []) if column in frame
    ]
    folds = int(params.get("well_hash_folds", 5))
    ridge_lambda = float(params.get("ridge_lambda", 1.0))
    clip = float(params.get("expected_error_clip", 250.0))
    min_train_rows = int(params.get("min_train_rows", 2000))
    min_holdout_rows = int(params.get("min_holdout_rows", 200))

    frame = frame.copy()
    frame["expected_tvt_error"] = np.nan
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in range(folds):
        train_mask = (frame["well_hash_fold"] != fold) & (
            frame["prefix_backtest_phase"] == "calibration"
        )
        holdout_mask = frame["well_hash_fold"] == fold
        train = frame.loc[train_mask].copy()
        holdout = frame.loc[holdout_mask].copy()
        if len(train) < min_train_rows or len(holdout) < min_holdout_rows:
            continue

        model = fit_confidence_model(
            train,
            feature_columns=feature_columns,
            target_column="primary_abs_error",
            ridge_lambda=ridge_lambda,
        )
        frame.loc[holdout.index, "expected_tvt_error"] = predict_expected_error(
            model, holdout, clip=clip
        )
        model_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "train_wells": int(train["well_id"].nunique()),
                "holdout_rows": int(len(holdout)),
                "holdout_wells": int(holdout["well_id"].nunique()),
                "feature_columns": feature_columns,
                "coefficients": model.coefficients,
                "means": model.means,
                "scales": model.scales,
                "medians": model.medians,
            }
        )

        for phase, phase_group in holdout.groupby("prefix_backtest_phase", sort=True):
            fold_rows.append(
                {
                    "fold": fold,
                    "phase": phase,
                    "rows": int(len(phase_group)),
                    "wells": int(phase_group["well_id"].nunique()),
                    "observed_mae": float(phase_group["primary_abs_error"].mean()),
                    "expected_mae": float(phase_group["expected_tvt_error"].mean()),
                    "expected_vs_abs_error_pearson": float(
                        numeric_series(phase_group, "expected_tvt_error").corr(
                            numeric_series(phase_group, "primary_abs_error")
                        )
                    ),
                }
            )

    return frame, pd.DataFrame(fold_rows), model_rows


def run_audit(
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    debug: bool = False,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_prefix = str(
        get_nested(config, "audit.output_prefix") or "prefix_backtest_tvt_confidence"
    )
    params = get_nested(config, "model.params") or {}
    source = build_source_spec(config, paths.root)
    debug_max_wells = int(get_nested(config, "audit.debug_max_wells") or 12) if debug else None
    frame = read_source_frame(source, debug_max_wells=debug_max_wells)
    if frame.empty:
        raise ValueError("Source frame is empty after filtering.")

    audit_frame = build_audit_frame(frame, source, config)
    predictions, fold_metrics, model_rows = fold_safe_confidence_predictions(audit_frame, config)
    valid_predictions = predictions[
        np.isfinite(numeric_series(predictions, "expected_tvt_error"))
    ].copy()
    if valid_predictions.empty:
        raise ValueError("No fold-safe confidence predictions were generated.")

    high_error_q = float(params.get("high_error_quantile", 0.8))
    high_threshold = float(valid_predictions["primary_abs_error"].quantile(high_error_q))
    valid_predictions["high_error"] = valid_predictions["primary_abs_error"] >= high_threshold
    valid_predictions["confidence_bin"] = assign_confidence_bins(
        valid_predictions,
        expected_col="expected_tvt_error",
        n_bins=int(params.get("confidence_bins", 5)),
    )
    expected_threshold = float(valid_predictions["expected_tvt_error"].quantile(high_error_q))
    valid_predictions["unstable_flag"] = (
        valid_predictions["expected_tvt_error"] >= expected_threshold
    )

    candidate_metrics_df = candidate_metrics(audit_frame, source)
    bin_metrics = summarize_group(
        valid_predictions,
        ["confidence_bin"],
        primary_error_col="primary_abs_error",
        expected_col="expected_tvt_error",
        high_error_col="high_error",
    )
    bucket_metrics = summarize_group(
        valid_predictions,
        ["prefix_backtest_phase", "distance_bucket"],
        primary_error_col="primary_abs_error",
        expected_col="expected_tvt_error",
        high_error_col="high_error",
    )
    phase_metrics = summarize_group(
        valid_predictions,
        ["prefix_backtest_phase"],
        primary_error_col="primary_abs_error",
        expected_col="expected_tvt_error",
        high_error_col="high_error",
    )
    correlations = signal_correlations(
        valid_predictions,
        feature_columns=[
            column for column in params.get("feature_columns", []) if column in valid_predictions
        ],
        primary_abs_error_col="primary_abs_error",
    )

    prediction_columns = [
        "id",
        "well_id",
        source.x_column if source.x_column in valid_predictions else "prefix_backtest_fraction",
        "prefix_backtest_fraction",
        "prefix_backtest_phase",
        "distance_bucket",
        "well_hash_fold",
        "primary_candidate",
        source.target_column,
        str(params.get("primary_candidate", "pf_ancc")),
        "primary_error",
        "primary_abs_error",
        "expected_tvt_error",
        "confidence_bin",
        "unstable_flag",
        "high_error",
    ]
    prediction_columns = [column for column in prediction_columns if column in valid_predictions]

    candidate_metrics_path = paths.artifacts_dir / f"{output_prefix}_candidate_metrics.csv"
    bin_metrics_path = paths.artifacts_dir / f"{output_prefix}_confidence_bin_metrics.csv"
    bucket_metrics_path = paths.artifacts_dir / f"{output_prefix}_bucket_metrics.csv"
    phase_metrics_path = paths.artifacts_dir / f"{output_prefix}_phase_metrics.csv"
    fold_metrics_path = paths.artifacts_dir / f"{output_prefix}_fold_metrics.csv"
    correlations_path = paths.artifacts_dir / f"{output_prefix}_signal_correlations.csv"
    predictions_path = paths.artifacts_dir / f"{output_prefix}_predictions.csv.gz"
    summary_path = paths.artifacts_dir / f"{output_prefix}_summary.json"

    candidate_metrics_df.to_csv(candidate_metrics_path, index=False)
    bin_metrics.to_csv(bin_metrics_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    phase_metrics.to_csv(phase_metrics_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    if bool(get_nested(config, "audit.write_row_predictions")):
        valid_predictions[prediction_columns].to_csv(
            predictions_path, index=False, compression="gzip"
        )

    ordered_bins = bin_metrics.sort_values("expected_error_mean")
    low_error = (
        float(ordered_bins["observed_mae"].iloc[0])
        if not ordered_bins.empty
        else float("nan")
    )
    high_error = (
        float(ordered_bins["observed_mae"].iloc[-1])
        if not ordered_bins.empty
        else float("nan")
    )
    lift = high_error / low_error if np.isfinite(low_error) and low_error > 0 else float("nan")
    expected_corr = float(
        numeric_series(valid_predictions, "expected_tvt_error").corr(
            numeric_series(valid_predictions, "primary_abs_error")
        )
    )

    compressed = source.path.suffix == ".gz"
    summary = {
        "experiment": get_nested(config, "experiment.name"),
        "status": "debug_completed" if debug else "implemented",
        "created_at": datetime.now(UTC).isoformat(),
        "debug": debug,
        "source": {
            "name": source.name,
            "path": str(source.path),
            "sha256": sha256_path(source.path),
            "decompressed_sha256": sha256_path(source.path, decompressed=True)
            if compressed
            else None,
            "rows_after_filter": int(len(frame)),
            "wells_after_filter": int(frame[source.well_column].nunique()),
        },
        "audit": {
            "primary_candidate": str(params.get("primary_candidate", "pf_ancc")),
            "rows_scored": int(len(valid_predictions)),
            "wells_scored": int(valid_predictions["well_id"].nunique()),
            "calibration_fraction": float(params.get("calibration_fraction", 0.35)),
            "well_hash_folds": int(params.get("well_hash_folds", 5)),
            "high_error_quantile": high_error_q,
            "high_error_threshold": high_threshold,
            "unstable_expected_error_threshold": expected_threshold,
        },
        "metrics": {
            "primary_candidate_rmse": rmse(
                valid_predictions[str(params.get("primary_candidate", "pf_ancc"))],
                valid_predictions[source.target_column],
            ),
            "primary_candidate_mae": float(valid_predictions["primary_abs_error"].mean()),
            "expected_error_abs_error_pearson": expected_corr,
            "top_vs_bottom_confidence_bin_observed_mae_lift": lift,
            "unstable_flag_rate": float(valid_predictions["unstable_flag"].mean()),
            "unstable_flag_high_error_rate": float(
                valid_predictions.loc[valid_predictions["unstable_flag"], "high_error"].mean()
            ),
            "stable_flag_high_error_rate": float(
                valid_predictions.loc[~valid_predictions["unstable_flag"], "high_error"].mean()
            ),
        },
        "outputs": {
            "candidate_metrics_csv": str(candidate_metrics_path),
            "confidence_bin_metrics_csv": str(bin_metrics_path),
            "bucket_metrics_csv": str(bucket_metrics_path),
            "phase_metrics_csv": str(phase_metrics_path),
            "fold_metrics_csv": str(fold_metrics_path),
            "signal_correlations_csv": str(correlations_path),
            "predictions_csv_gz": str(predictions_path) if predictions_path.exists() else None,
            "summary_json": str(summary_path),
        },
        "model_rows": model_rows,
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    shutil.copyfile(summary_path, paths.metrics_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    if not args.allow_local:
        paths.require_kaggle_runtime()
    summary = run_audit(config=load_config(), paths=paths, debug=args.debug)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
