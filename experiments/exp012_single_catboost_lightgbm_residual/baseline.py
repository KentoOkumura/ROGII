from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
PREFIX_STRATEGIES = {"last_anchor", "recent_linear"}
DEFAULT_DRIFT_STRATEGY = "drift_model"
FEATURE_COLUMNS = [
    "md",
    "x",
    "y",
    "z",
    "gr",
    "row_index",
    "eval_step",
    "eval_progress",
    "eval_row_count",
    "known_row_count",
    "delta_md",
    "delta_md_abs",
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_xy",
    "delta_xyz",
    "anchor_dz_dmd",
    "step_dz_dmd",
    "last_known_md",
    "last_known_tvt",
    "last_known_x",
    "last_known_y",
    "last_known_z",
    "last_known_gr",
    "recent_tvt_slope",
    "prefix_md_span",
    "prefix_gr_mean",
    "prefix_gr_std",
    "prefix_gr_slope",
    "gr_delta_last",
    "gr_delta_prefix_mean",
    "gr_roll_short_mean",
    "gr_roll_long_mean",
    "gr_roll_long_std",
]
GR_ROLL_FEATURE_COLUMNS = [
    "gr_roll_short_mean",
    "gr_roll_long_mean",
    "gr_roll_long_std",
]
GR_FEATURE_COLUMNS = [
    "gr",
    "last_known_gr",
    "prefix_gr_mean",
    "prefix_gr_std",
    "prefix_gr_slope",
    "gr_delta_last",
    "gr_delta_prefix_mean",
    *GR_ROLL_FEATURE_COLUMNS,
]
DEFAULT_FEATURE_SETS = {
    "all": FEATURE_COLUMNS,
    "no_gr_roll": [column for column in FEATURE_COLUMNS if column not in GR_ROLL_FEATURE_COLUMNS],
    "no_gr_signal": [column for column in FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
}


@dataclass(frozen=True)
class PrefixPrediction:
    eval_indices: np.ndarray
    predictions: dict[str, np.ndarray]
    last_known_index: int
    last_known_md: float
    last_known_tvt: float
    recent_slope: float


@dataclass(frozen=True)
class DriftFeatureFrame:
    eval_indices: np.ndarray
    features: pd.DataFrame
    baseline_prediction: np.ndarray
    target_residual: np.ndarray | None
    last_known_index: int
    last_known_md: float
    last_known_tvt: float
    recent_slope: float


def config_get(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def well_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(HORIZONTAL_SUFFIX):
        raise ValueError(f"not a horizontal well CSV: {path}")
    return name.removesuffix(HORIZONTAL_SUFFIX)


def strategy_names(config: dict[str, Any]) -> list[str]:
    raw = config_get(config, "model.strategies", ["last_anchor"])
    if not isinstance(raw, list) or not raw:
        raise ValueError("model.strategies must be a non-empty list")
    return [str(strategy) for strategy in raw]


def primary_strategy(config: dict[str, Any]) -> str:
    strategy = str(config_get(config, "model.primary_strategy", "last_anchor"))
    if strategy not in strategy_names(config):
        raise ValueError(f"primary strategy is not listed in model.strategies: {strategy}")
    return strategy


def drift_strategy(config: dict[str, Any]) -> str:
    return str(config_get(config, "model.drift_strategy", DEFAULT_DRIFT_STRATEGY))


def feature_set_name(config: dict[str, Any]) -> str:
    return str(config_get(config, "model.feature_set", "all"))


def active_feature_columns(config: dict[str, Any]) -> list[str]:
    name = feature_set_name(config)
    configured_sets = config_get(config, "model.feature_sets", {})
    configured_columns = None
    if isinstance(configured_sets, dict):
        configured_columns = configured_sets.get(name)

    raw_columns = (
        configured_columns if configured_columns is not None else DEFAULT_FEATURE_SETS.get(name)
    )
    if raw_columns is None:
        raise ValueError(f"unknown model.feature_set: {name}")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError(f"model.feature_sets.{name} must be a non-empty list")

    columns: list[str] = []
    for column in raw_columns:
        column = str(column)
        if column not in columns:
            columns.append(column)

    unknown_columns = sorted(set(columns) - set(FEATURE_COLUMNS))
    if unknown_columns:
        raise ValueError(f"unknown feature columns in feature set {name}: {unknown_columns}")
    return columns


def optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def finite_mean(values: np.ndarray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def finite_std(values: np.ndarray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.std(finite))


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(max(1, window), min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def trailing_std(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(max(2, window), min_periods=2)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def estimate_prefix_slope(
    md: np.ndarray,
    values: np.ndarray,
    last_known_index: int,
    *,
    window: int,
    max_abs_slope: float | None = None,
    shrink: float = 1.0,
) -> float:
    start = max(0, last_known_index - window + 1)
    md_window = md[start : last_known_index + 1]
    value_window = values[start : last_known_index + 1]
    finite = np.isfinite(md_window) & np.isfinite(value_window)
    md_window = md_window[finite]
    value_window = value_window[finite]
    if md_window.size < 2:
        return 0.0

    delta_md = np.diff(md_window)
    delta_value = np.diff(value_window)
    valid = np.isfinite(delta_md) & np.isfinite(delta_value) & (delta_md != 0.0)
    if not valid.any():
        return 0.0

    slope = float(np.median(delta_value[valid] / delta_md[valid]))
    if not np.isfinite(slope):
        return 0.0
    if max_abs_slope is not None and max_abs_slope > 0:
        slope = float(np.clip(slope, -max_abs_slope, max_abs_slope))
    return slope * shrink


def estimate_recent_slope(
    md: np.ndarray,
    tvt_input: np.ndarray,
    last_known_index: int,
    config: dict[str, Any],
) -> float:
    return estimate_prefix_slope(
        md,
        tvt_input,
        last_known_index,
        window=int(config_get(config, "model.params.recent_slope_window", 200)),
        max_abs_slope=float(config_get(config, "model.params.max_abs_recent_slope", 0.08)),
        shrink=float(config_get(config, "model.params.recent_slope_shrink", 0.5)),
    )


def predict_from_prefix(df: pd.DataFrame, config: dict[str, Any]) -> PrefixPrediction:
    required_columns = {"MD", "TVT_input"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    md = df["MD"].to_numpy(dtype=float)
    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    known_mask = np.isfinite(tvt_input)
    eval_indices = np.flatnonzero(~known_mask)
    known_indices = np.flatnonzero(known_mask)

    if known_indices.size == 0:
        raise ValueError("TVT_input has no known prefix rows")

    last_known_index = int(known_indices[-1])
    if eval_indices.size and int(eval_indices.min()) <= last_known_index:
        raise ValueError("TVT_input missing rows are not a tail block")

    last_known_md = float(md[last_known_index])
    last_known_tvt = float(tvt_input[last_known_index])
    eval_md = md[eval_indices]
    recent_slope = estimate_recent_slope(md, tvt_input, last_known_index, config)

    predictions: dict[str, np.ndarray] = {
        "last_anchor": np.full(eval_indices.size, last_known_tvt, dtype=float)
    }
    configured_drift_strategy = drift_strategy(config)
    for strategy in strategy_names(config):
        if strategy == "last_anchor":
            continue
        if strategy == "recent_linear":
            predictions[strategy] = last_known_tvt + recent_slope * (eval_md - last_known_md)
        elif strategy == configured_drift_strategy:
            continue
        else:
            raise ValueError(f"unknown prediction strategy: {strategy}")

    return PrefixPrediction(
        eval_indices=eval_indices,
        predictions=predictions,
        last_known_index=last_known_index,
        last_known_md=last_known_md,
        last_known_tvt=last_known_tvt,
        recent_slope=recent_slope,
    )


def build_drift_feature_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_target: bool,
) -> DriftFeatureFrame:
    required_columns = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    prefix = predict_from_prefix(df, config)
    eval_indices = prefix.eval_indices
    baseline_prediction = prefix.predictions["last_anchor"]
    if eval_indices.size == 0:
        empty_features = pd.DataFrame(columns=FEATURE_COLUMNS)
        target_residual = np.asarray([], dtype=float) if include_target else None
        return DriftFeatureFrame(
            eval_indices=eval_indices,
            features=empty_features,
            baseline_prediction=baseline_prediction,
            target_residual=target_residual,
            last_known_index=prefix.last_known_index,
            last_known_md=prefix.last_known_md,
            last_known_tvt=prefix.last_known_tvt,
            recent_slope=prefix.recent_slope,
        )

    md = df["MD"].to_numpy(dtype=float)
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    gr = df["GR"].to_numpy(dtype=float)

    last_index = prefix.last_known_index
    n_eval = int(eval_indices.size)
    eval_step = np.arange(n_eval, dtype=float)
    eval_progress = eval_step / max(n_eval - 1, 1)

    eval_md = md[eval_indices]
    eval_x = x[eval_indices]
    eval_y = y[eval_indices]
    eval_z = z[eval_indices]
    eval_gr = gr[eval_indices]

    last_x = float(x[last_index])
    last_y = float(y[last_index])
    last_z = float(z[last_index])
    last_gr = float(gr[last_index])

    delta_md = eval_md - prefix.last_known_md
    delta_x = eval_x - last_x
    delta_y = eval_y - last_y
    delta_z = eval_z - last_z
    delta_xy = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    delta_xyz = np.sqrt(delta_xy * delta_xy + delta_z * delta_z)

    step_md = np.diff(md, prepend=np.nan)
    step_z = np.diff(z, prepend=np.nan)
    step_dz_dmd = safe_divide(step_z, step_md)

    prefix_gr = gr[: last_index + 1]
    prefix_gr_mean = finite_mean(prefix_gr)
    prefix_gr_std = finite_std(prefix_gr)
    prefix_gr_slope = estimate_prefix_slope(
        md,
        gr,
        last_index,
        window=int(config_get(config, "model.params.gr_slope_window", 200)),
        max_abs_slope=float(config_get(config, "model.params.max_abs_gr_slope", 2.0)),
    )

    gr_roll_short_window = int(config_get(config, "model.params.gr_roll_short", 25))
    gr_roll_long_window = int(config_get(config, "model.params.gr_roll_long", 100))
    gr_roll_short = trailing_mean(gr, gr_roll_short_window)
    gr_roll_long = trailing_mean(gr, gr_roll_long_window)
    gr_roll_long_std = trailing_std(gr, gr_roll_long_window)

    known_row_count = float(last_index + 1)
    prefix_md_span = float(prefix.last_known_md - md[0]) if len(md) else 0.0
    features = pd.DataFrame(
        {
            "md": eval_md,
            "x": eval_x,
            "y": eval_y,
            "z": eval_z,
            "gr": eval_gr,
            "row_index": eval_indices.astype(float),
            "eval_step": eval_step,
            "eval_progress": eval_progress,
            "eval_row_count": np.full(n_eval, float(n_eval), dtype=float),
            "known_row_count": np.full(n_eval, known_row_count, dtype=float),
            "delta_md": delta_md,
            "delta_md_abs": np.abs(delta_md),
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "delta_xy": delta_xy,
            "delta_xyz": delta_xyz,
            "anchor_dz_dmd": safe_divide(delta_z, delta_md),
            "step_dz_dmd": step_dz_dmd[eval_indices],
            "last_known_md": np.full(n_eval, prefix.last_known_md, dtype=float),
            "last_known_tvt": np.full(n_eval, prefix.last_known_tvt, dtype=float),
            "last_known_x": np.full(n_eval, last_x, dtype=float),
            "last_known_y": np.full(n_eval, last_y, dtype=float),
            "last_known_z": np.full(n_eval, last_z, dtype=float),
            "last_known_gr": np.full(n_eval, last_gr, dtype=float),
            "recent_tvt_slope": np.full(n_eval, prefix.recent_slope, dtype=float),
            "prefix_md_span": np.full(n_eval, prefix_md_span, dtype=float),
            "prefix_gr_mean": np.full(n_eval, prefix_gr_mean, dtype=float),
            "prefix_gr_std": np.full(n_eval, prefix_gr_std, dtype=float),
            "prefix_gr_slope": np.full(n_eval, prefix_gr_slope, dtype=float),
            "gr_delta_last": eval_gr - last_gr,
            "gr_delta_prefix_mean": eval_gr - prefix_gr_mean,
            "gr_roll_short_mean": gr_roll_short[eval_indices],
            "gr_roll_long_mean": gr_roll_long[eval_indices],
            "gr_roll_long_std": gr_roll_long_std[eval_indices],
        }
    )[FEATURE_COLUMNS]

    target_residual = None
    target_column = str(config_get(config, "data.target_column", "TVT"))
    if include_target:
        if target_column not in df.columns:
            raise ValueError(f"missing target column for training: {target_column}")
        y_true = df.loc[eval_indices, target_column].to_numpy(dtype=float)
        target_residual = y_true - baseline_prediction

    return DriftFeatureFrame(
        eval_indices=eval_indices,
        features=features,
        baseline_prediction=baseline_prediction,
        target_residual=target_residual,
        last_known_index=prefix.last_known_index,
        last_known_md=prefix.last_known_md,
        last_known_tvt=prefix.last_known_tvt,
        recent_slope=prefix.recent_slope,
    )


def make_drift_model(config: dict[str, Any], *, random_state: int | None = None) -> Any:
    seed = int(config_get(config, "validation.seed", 42)) if random_state is None else random_state
    estimator = str(
        config_get(config, "model.drift_model.estimator", "HistGradientBoostingRegressor")
    )
    if estimator == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(
            max_iter=int(config_get(config, "model.drift_model.params.max_iter", 180)),
            learning_rate=float(config_get(config, "model.drift_model.params.learning_rate", 0.05)),
            max_leaf_nodes=int(config_get(config, "model.drift_model.params.max_leaf_nodes", 31)),
            min_samples_leaf=int(
                config_get(config, "model.drift_model.params.min_samples_leaf", 80)
            ),
            l2_regularization=float(
                config_get(config, "model.drift_model.params.l2_regularization", 0.05)
            ),
            max_bins=int(config_get(config, "model.drift_model.params.max_bins", 255)),
            early_stopping=False,
            random_state=seed,
        )
    if estimator == "LGBMRegressor":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "model.drift_model.estimator=LGBMRegressor requires lightgbm in the runtime"
            ) from exc
        return LGBMRegressor(
            n_estimators=int(config_get(config, "model.drift_model.params.n_estimators", 800)),
            learning_rate=float(config_get(config, "model.drift_model.params.learning_rate", 0.03)),
            num_leaves=int(config_get(config, "model.drift_model.params.num_leaves", 31)),
            min_child_samples=int(
                config_get(config, "model.drift_model.params.min_child_samples", 80)
            ),
            subsample=float(config_get(config, "model.drift_model.params.subsample", 0.9)),
            colsample_bytree=float(
                config_get(config, "model.drift_model.params.colsample_bytree", 0.9)
            ),
            reg_lambda=float(config_get(config, "model.drift_model.params.reg_lambda", 0.1)),
            objective="regression",
            random_state=seed,
            n_jobs=int(config_get(config, "model.drift_model.params.n_jobs", 2)),
            verbosity=-1,
        )
    if estimator == "CatBoostRegressor":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "model.drift_model.estimator=CatBoostRegressor requires catboost in the runtime"
            ) from exc
        return CatBoostRegressor(
            iterations=int(config_get(config, "model.drift_model.params.iterations", 800)),
            learning_rate=float(config_get(config, "model.drift_model.params.learning_rate", 0.03)),
            depth=int(config_get(config, "model.drift_model.params.depth", 8)),
            l2_leaf_reg=float(config_get(config, "model.drift_model.params.l2_leaf_reg", 3.0)),
            loss_function="RMSE",
            random_seed=seed,
            thread_count=int(config_get(config, "model.drift_model.params.thread_count", 2)),
            allow_writing_files=False,
            verbose=False,
        )
    raise ValueError(f"unsupported model.drift_model.estimator: {estimator}")


def sample_training_rows(
    frame: DriftFeatureFrame,
    rng: np.random.Generator,
    max_rows: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if frame.target_residual is None:
        raise ValueError("target_residual is required for training")

    valid_indices = np.flatnonzero(np.isfinite(frame.target_residual))
    if max_rows is not None and valid_indices.size > max_rows:
        valid_indices = rng.choice(valid_indices, size=max_rows, replace=False)
    if valid_indices.size == 0:
        return frame.features.iloc[[]], np.asarray([], dtype=float)
    return frame.features.iloc[valid_indices], frame.target_residual[valid_indices]


def fit_drift_model_from_files(
    files: Iterable[Path],
    config: dict[str, Any],
    *,
    seed: int,
    max_rows_total: int | None,
    max_rows_per_well: int | None,
) -> tuple[Any, int]:
    rng = np.random.default_rng(seed)
    x_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []

    for path in files:
        df = pd.read_csv(path)
        frame = build_drift_feature_frame(df, config, include_target=True)
        x_part, y_part = sample_training_rows(frame, rng, max_rows_per_well)
        if y_part.size:
            x_parts.append(x_part)
            y_parts.append(y_part)

    if not y_parts:
        raise ValueError("no finite residual training rows were collected")

    x_train = pd.concat(x_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    if max_rows_total is not None and y_train.size > max_rows_total:
        selected = rng.choice(y_train.size, size=max_rows_total, replace=False)
        x_train = x_train.iloc[selected].reset_index(drop=True)
        y_train = y_train[selected]

    model_columns = active_feature_columns(config)
    model = make_drift_model(config, random_state=seed)
    model.fit(x_train[model_columns], y_train)
    return model, int(y_train.size)


def predict_drift(frame: DriftFeatureFrame, model: Any, config: dict[str, Any]) -> np.ndarray:
    if frame.features.empty:
        return np.asarray([], dtype=float)
    model_columns = active_feature_columns(config)
    residual = model.predict(frame.features[model_columns]).astype(float)
    residual *= float(config_get(config, "model.params.residual_shrink", 1.0))

    max_abs_residual = float(config_get(config, "model.params.max_abs_residual", 0.0))
    if max_abs_residual > 0:
        residual = np.clip(residual, -max_abs_residual, max_abs_residual)
    return frame.baseline_prediction + residual
