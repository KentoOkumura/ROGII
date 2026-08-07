from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
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
PF_BEAM_SCALE_NAMES = ("s3", "s5", "s8", "s12")
PF_BEAM_SUMMARY_FEATURE_COLUMNS = [
    "pf_beam_available",
    "pf_beam_best_score",
    "pf_beam_second_score",
    "pf_beam_confidence",
    "pf_beam_ambiguity",
    "pf_beam_score_entropy",
    "pf_beam_best_scale",
    "pf_beam_best_shift_tvt",
    "pf_beam_best_slope_scale",
    "pf_beam_best_dtw_cost",
    "pf_beam_hold_weight",
    "pf_beam_recent_delta_tvt",
    "pf_beam_best_delta_tvt",
    "pf_beam_mean_delta_tvt",
    "pf_beam_std_delta_tvt",
    "pf_beam_range_delta_tvt",
    "pf_beam_best_minus_recent",
    "pf_beam_mean_minus_recent",
    "pf_beam_max_pairwise_delta",
    "pf_beam_eval_length_selector",
    "pf_beam_z_span_selector",
]
PF_BEAM_SCALE_FEATURE_COLUMNS = [
    f"pf_beam_{name}_{suffix}"
    for name in PF_BEAM_SCALE_NAMES
    for suffix in (
        "score",
        "shift_tvt",
        "slope_scale",
        "dtw_cost",
        "pred_tvt",
        "delta_tvt",
        "minus_recent",
        "typewell_gr",
        "gr_abs_error",
    )
]
PF_BEAM_FEATURE_COLUMNS = [
    *PF_BEAM_SUMMARY_FEATURE_COLUMNS,
    *PF_BEAM_SCALE_FEATURE_COLUMNS,
]
BASE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
DTW_DWT_SCALE_NAMES = ("w16", "w32", "w64", "w128")
DTW_DWT_SUMMARY_FEATURE_COLUMNS = [
    "dtw_dwt_available",
    "dtw_dwt_best_cost",
    "dtw_dwt_second_cost",
    "dtw_dwt_cost_margin",
    "dtw_dwt_best_shift_tvt",
    "dtw_dwt_best_ncc",
    "dtw_dwt_best_dtw_cost",
    "dtw_dwt_best_slope",
    "dtw_dwt_confidence",
    "dtw_dwt_candidate_count",
    "dtw_dwt_recent_delta_tvt",
    "dtw_dwt_best_delta_tvt",
    "dtw_dwt_best_minus_recent",
    "dtw_dwt_eval_energy_mean",
    "dtw_dwt_type_energy_mean",
    "dtw_dwt_energy_abs_error_mean",
    "dtw_dwt_smooth_abs_error_mean",
    "dtw_dwt_eval_length_selector",
]
DTW_DWT_SCALE_FEATURE_COLUMNS = [
    f"dtw_dwt_{name}_{suffix}"
    for name in DTW_DWT_SCALE_NAMES
    for suffix in (
        "eval_energy",
        "type_energy",
        "energy_abs_error",
        "smooth_abs_error",
    )
]
DTW_DWT_ROW_FEATURE_COLUMNS = [
    "dtw_dwt_best_pred_tvt",
    "dtw_dwt_best_typewell_gr",
    "dtw_dwt_best_gr_abs_error",
]
DTW_DWT_FEATURE_COLUMNS = [
    *DTW_DWT_SUMMARY_FEATURE_COLUMNS,
    *DTW_DWT_ROW_FEATURE_COLUMNS,
    *DTW_DWT_SCALE_FEATURE_COLUMNS,
]
FEATURE_COLUMNS = [*FEATURE_COLUMNS, *PF_BEAM_FEATURE_COLUMNS, *DTW_DWT_FEATURE_COLUMNS]
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
    "all": BASE_FEATURE_COLUMNS,
    "no_gr_roll": [
        column for column in BASE_FEATURE_COLUMNS if column not in GR_ROLL_FEATURE_COLUMNS
    ],
    "no_gr_signal": [column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
    "all_plus_pf_beam": [*BASE_FEATURE_COLUMNS, *PF_BEAM_FEATURE_COLUMNS],
    "no_gr_signal_plus_pf_beam": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *PF_BEAM_FEATURE_COLUMNS,
    ],
    "all_plus_dtw_dwt": [*BASE_FEATURE_COLUMNS, *DTW_DWT_FEATURE_COLUMNS],
    "no_gr_signal_plus_dtw_dwt": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *DTW_DWT_FEATURE_COLUMNS,
    ],
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


def typewell_path_for_horizontal_path(path: Path) -> Path:
    return path.with_name(path.name.replace(HORIZONTAL_SUFFIX, TYPEWELL_SUFFIX))


def read_typewell_for_horizontal_path(path: Path) -> pd.DataFrame | None:
    typewell_path = typewell_path_for_horizontal_path(path)
    if not typewell_path.exists():
        return None
    return pd.read_csv(typewell_path)


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


def centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    min_periods = min(window, max(3, window // 3))
    return (
        pd.Series(values, dtype="float64")
        .rolling(window, min_periods=min_periods, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def interpolate_finite_by_index(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if finite.all():
        return values.astype(float)
    if not finite.any():
        return np.full(values.shape, np.nan, dtype=float)
    indices = np.arange(values.size, dtype=float)
    return np.interp(indices, indices[finite], values[finite]).astype(float)


def finite_pair_ncc(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_valid_fraction: float,
) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    min_valid = max(5, int(np.ceil(left.size * min_valid_fraction)))
    if int(valid.sum()) < min_valid:
        return np.nan

    left_centered = left[valid] - float(np.mean(left[valid]))
    right_centered = right[valid] - float(np.mean(right[valid]))
    denominator = float(
        np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered))
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan
    return float(np.clip(np.sum(left_centered * right_centered) / denominator, -1.0, 1.0))


def clean_typewell_curve(typewell_df: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray] | None:
    if typewell_df is None or not {"TVT", "GR"}.issubset(typewell_df.columns):
        return None

    type_tvt = typewell_df["TVT"].to_numpy(dtype=float)
    type_gr = typewell_df["GR"].to_numpy(dtype=float)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    if int(finite.sum()) < 5:
        return None

    type_tvt = type_tvt[finite]
    type_gr = type_gr[finite]
    order = np.argsort(type_tvt)
    type_tvt = type_tvt[order]
    type_gr = type_gr[order]
    unique_tvt, unique_indices = np.unique(type_tvt, return_index=True)
    if unique_tvt.size < 5:
        return None
    return unique_tvt.astype(float), type_gr[unique_indices].astype(float)


def interpolate_typewell_gr(
    query_tvt: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
) -> np.ndarray:
    return np.interp(query_tvt, type_tvt, type_gr, left=np.nan, right=np.nan).astype(float)


def downsample_pair(
    left: np.ndarray,
    right: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size <= max_points:
        return left, right
    indices = np.linspace(0, left.size - 1, max_points).round().astype(int)
    return left[indices], right[indices]


def normalized_banded_dtw_cost(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_points: int,
    band_fraction: float,
) -> float:
    left, right = downsample_pair(left, right, max_points)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 5:
        return np.nan
    left = left[valid]
    right = right[valid]
    left_scale = finite_std(left, default=1.0)
    right_scale = finite_std(right, default=1.0)
    left = (left - finite_mean(left)) / max(left_scale, 1e-6)
    right = (right - finite_mean(right)) / max(right_scale, 1e-6)

    n = int(left.size)
    band = max(2, int(np.ceil(n * band_fraction)))
    previous = np.full(n + 1, np.inf, dtype=float)
    current = np.full(n + 1, np.inf, dtype=float)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current.fill(np.inf)
        j_start = max(1, i - band)
        j_end = min(n, i + band)
        for j in range(j_start, j_end + 1):
            cost = abs(left[i - 1] - right[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    cost = float(previous[n] / n)
    return cost if np.isfinite(cost) else np.nan


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


def pf_beam_enabled(config: dict[str, Any]) -> bool:
    return bool(config_get(config, "model.pf_beam.enabled", False))


def pf_beam_candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = [
        {
            "name": "s3",
            "scale": 3.0,
            "slope_scale": 0.375,
            "window": 25,
            "radius": 90.0,
            "shift_step": 15.0,
        },
        {
            "name": "s5",
            "scale": 5.0,
            "slope_scale": 0.625,
            "window": 35,
            "radius": 120.0,
            "shift_step": 15.0,
        },
        {
            "name": "s8",
            "scale": 8.0,
            "slope_scale": 1.0,
            "window": 50,
            "radius": 160.0,
            "shift_step": 20.0,
        },
        {
            "name": "s12",
            "scale": 12.0,
            "slope_scale": 1.5,
            "window": 75,
            "radius": 220.0,
            "shift_step": 20.0,
        },
    ]
    raw_specs = config_get(config, "model.pf_beam.candidates", defaults)
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("model.pf_beam.candidates must be a non-empty list")

    specs: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"model.pf_beam.candidates[{index}] must be a mapping")
        name = str(raw_spec.get("name") or "")
        if name not in PF_BEAM_SCALE_NAMES:
            raise ValueError(
                f"unsupported PF/beam candidate name {name!r}; "
                f"expected one of {list(PF_BEAM_SCALE_NAMES)}"
            )
        specs.append(
            {
                "name": name,
                "scale": float(raw_spec.get("scale", name.removeprefix("s"))),
                "slope_scale": float(raw_spec.get("slope_scale", 1.0)),
                "window": int(raw_spec.get("window", 50)),
                "radius": float(raw_spec.get("radius", 160.0)),
                "shift_step": float(raw_spec.get("shift_step", 20.0)),
            }
        )
    return specs


def empty_pf_beam_features(n_rows: int) -> pd.DataFrame:
    data = {column: np.full(n_rows, np.nan, dtype=float) for column in PF_BEAM_FEATURE_COLUMNS}
    zero_columns = [
        "pf_beam_available",
        "pf_beam_best_score",
        "pf_beam_second_score",
        "pf_beam_confidence",
        "pf_beam_ambiguity",
        "pf_beam_score_entropy",
        "pf_beam_best_scale",
        "pf_beam_best_shift_tvt",
        "pf_beam_best_slope_scale",
        "pf_beam_best_dtw_cost",
        "pf_beam_hold_weight",
        "pf_beam_recent_delta_tvt",
        "pf_beam_best_delta_tvt",
        "pf_beam_mean_delta_tvt",
        "pf_beam_std_delta_tvt",
        "pf_beam_range_delta_tvt",
        "pf_beam_best_minus_recent",
        "pf_beam_mean_minus_recent",
        "pf_beam_max_pairwise_delta",
        "pf_beam_eval_length_selector",
        "pf_beam_z_span_selector",
    ]
    for column in zero_columns:
        data[column] = np.zeros(n_rows, dtype=float)
    for name in PF_BEAM_SCALE_NAMES:
        for suffix in (
            "score",
            "shift_tvt",
            "slope_scale",
            "dtw_cost",
            "delta_tvt",
            "minus_recent",
            "gr_abs_error",
        ):
            data[f"pf_beam_{name}_{suffix}"] = np.zeros(n_rows, dtype=float)
    return pd.DataFrame(data, columns=PF_BEAM_FEATURE_COLUMNS)


def effective_pf_beam_prior_slope(prefix: PrefixPrediction, config: dict[str, Any]) -> float:
    slope = float(prefix.recent_slope)
    min_abs_slope = float(config_get(config, "model.pf_beam.min_abs_prior_slope", 0.005))
    if abs(slope) >= min_abs_slope:
        return slope

    fallback = abs(float(config_get(config, "model.pf_beam.fallback_abs_prior_slope", 0.03)))
    sign = -1.0 if slope < 0.0 else 1.0
    return sign * fallback


def score_pf_beam_candidate(
    *,
    eval_md: np.ndarray,
    eval_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    prefix: PrefixPrediction,
    spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    window = int(spec["window"])
    radius = float(spec["radius"])
    shift_step = max(float(spec["shift_step"]), 1.0)
    slope_scale = float(spec["slope_scale"])
    min_valid_fraction = float(config_get(config, "model.pf_beam.min_valid_fraction", 0.30))

    eval_gr_filled = interpolate_finite_by_index(eval_gr)
    eval_smooth = centered_mean(eval_gr_filled, window)
    shifts = np.arange(-radius, radius + shift_step * 0.5, shift_step, dtype=float)
    base_slope = effective_pf_beam_prior_slope(prefix, config) * slope_scale
    delta_md = eval_md - prefix.last_known_md
    prior_tvt = prefix.last_known_tvt + base_slope * delta_md

    best_score = -np.inf
    second_score = -np.inf
    best_shift = np.nan
    best_tvt = np.full(eval_md.shape, np.nan, dtype=float)
    best_type_gr = np.full(eval_md.shape, np.nan, dtype=float)
    for shift in shifts:
        candidate_tvt = prior_tvt + shift
        sampled_gr = interpolate_typewell_gr(candidate_tvt, type_tvt, type_gr)
        sampled_gr = interpolate_finite_by_index(sampled_gr)
        sampled_smooth = centered_mean(sampled_gr, window)
        score = finite_pair_ncc(
            eval_smooth,
            sampled_smooth,
            min_valid_fraction=min_valid_fraction,
        )
        if not np.isfinite(score):
            continue
        if score > best_score:
            second_score = best_score
            best_score = score
            best_shift = float(shift)
            best_tvt = candidate_tvt.astype(float)
            best_type_gr = sampled_gr.astype(float)
        elif score > second_score:
            second_score = score

    if not np.isfinite(best_score):
        best_score = 0.0
        second_score = 0.0
        best_shift = 0.0
        best_tvt = prefix.last_known_tvt + np.zeros(eval_md.shape, dtype=float)
        best_type_gr = np.full(eval_md.shape, np.nan, dtype=float)
    elif not np.isfinite(second_score):
        second_score = best_score

    dtw_cost = normalized_banded_dtw_cost(
        eval_smooth,
        centered_mean(best_type_gr, window),
        max_points=int(config_get(config, "model.pf_beam.dtw_max_points", 160)),
        band_fraction=float(config_get(config, "model.pf_beam.dtw_band_fraction", 0.12)),
    )
    return {
        "score": float(best_score),
        "second_score": float(second_score),
        "shift_tvt": float(best_shift),
        "slope_scale": slope_scale,
        "scale": float(spec["scale"]),
        "pred_tvt": best_tvt,
        "typewell_gr": best_type_gr,
        "dtw_cost": float(dtw_cost) if np.isfinite(dtw_cost) else np.nan,
    }


def build_pf_beam_features(
    *,
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    typewell_df: pd.DataFrame | None,
    prefix: PrefixPrediction,
    eval_indices: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    n_eval = int(eval_indices.size)
    features = empty_pf_beam_features(n_eval)
    if n_eval == 0 or not pf_beam_enabled(config):
        return features

    curve = clean_typewell_curve(typewell_df)
    if curve is None:
        return features
    type_tvt, type_gr = curve

    eval_md = md[eval_indices]
    eval_z = z[eval_indices]
    eval_gr = gr[eval_indices]
    delta_md = eval_md - prefix.last_known_md
    recent_tvt = prefix.last_known_tvt + prefix.recent_slope * delta_md

    records: dict[str, dict[str, Any]] = {}
    pred_matrix: list[np.ndarray] = []
    scores: list[float] = []
    for spec in pf_beam_candidate_specs(config):
        name = str(spec["name"])
        record = score_pf_beam_candidate(
            eval_md=eval_md,
            eval_gr=eval_gr,
            type_tvt=type_tvt,
            type_gr=type_gr,
            prefix=prefix,
            spec=spec,
            config=config,
        )
        records[name] = record
        pred_tvt = np.asarray(record["pred_tvt"], dtype=float)
        typewell_gr = np.asarray(record["typewell_gr"], dtype=float)
        delta_tvt = pred_tvt - prefix.last_known_tvt
        pred_matrix.append(pred_tvt)
        scores.append(float(record["score"]))

        features[f"pf_beam_{name}_score"] = record["score"]
        features[f"pf_beam_{name}_shift_tvt"] = record["shift_tvt"]
        features[f"pf_beam_{name}_slope_scale"] = record["slope_scale"]
        features[f"pf_beam_{name}_dtw_cost"] = record["dtw_cost"]
        features[f"pf_beam_{name}_pred_tvt"] = pred_tvt
        features[f"pf_beam_{name}_delta_tvt"] = delta_tvt
        features[f"pf_beam_{name}_minus_recent"] = pred_tvt - recent_tvt
        features[f"pf_beam_{name}_typewell_gr"] = typewell_gr
        features[f"pf_beam_{name}_gr_abs_error"] = np.abs(eval_gr - typewell_gr)

    if not records:
        return features

    _, best_record = max(records.items(), key=lambda item: float(item[1]["score"]))
    all_scores = np.asarray(scores, dtype=float)
    sorted_scores = np.sort(all_scores[np.isfinite(all_scores)])[::-1]
    best_score = float(best_record["score"])
    second_score = float(sorted_scores[1]) if sorted_scores.size > 1 else best_score
    confidence = max(0.0, best_score - second_score)

    score_weights = np.exp(np.clip(all_scores - np.nanmax(all_scores), -20.0, 20.0))
    if not np.isfinite(score_weights).any() or float(np.nansum(score_weights)) <= 0.0:
        score_weights = np.ones_like(all_scores, dtype=float)
    score_weights = score_weights / float(np.nansum(score_weights))
    entropy = -float(np.nansum(score_weights * np.log(np.clip(score_weights, 1e-12, 1.0))))

    predictions = np.vstack(pred_matrix)
    mean_pred = np.nanmean(predictions, axis=0)
    std_pred = np.nanstd(predictions, axis=0)
    range_pred = np.nanmax(predictions, axis=0) - np.nanmin(predictions, axis=0)
    best_pred = np.asarray(best_record["pred_tvt"], dtype=float)
    recent_delta = recent_tvt - prefix.last_known_tvt
    mean_delta = mean_pred - prefix.last_known_tvt
    best_delta = best_pred - prefix.last_known_tvt

    z_span = float(np.nanmax(eval_z) - np.nanmin(eval_z)) if eval_z.size else 0.0
    eval_length_threshold = float(config_get(config, "model.pf_beam.long_eval_rows", 5700.0))
    z_span_threshold = float(config_get(config, "model.pf_beam.high_z_span", 120.0))
    hold_weight = np.clip(
        1.0 - confidence * float(config_get(config, "model.pf_beam.confidence_hold_scale", 2.0)),
        0.0,
        1.0,
    )

    features["pf_beam_available"] = 1.0
    features["pf_beam_best_score"] = best_score
    features["pf_beam_second_score"] = second_score
    features["pf_beam_confidence"] = confidence
    features["pf_beam_ambiguity"] = max(0.0, 1.0 - confidence)
    features["pf_beam_score_entropy"] = entropy
    features["pf_beam_best_scale"] = float(best_record["scale"])
    features["pf_beam_best_shift_tvt"] = float(best_record["shift_tvt"])
    features["pf_beam_best_slope_scale"] = float(best_record["slope_scale"])
    features["pf_beam_best_dtw_cost"] = best_record["dtw_cost"]
    features["pf_beam_hold_weight"] = float(hold_weight)
    features["pf_beam_recent_delta_tvt"] = recent_delta
    features["pf_beam_best_delta_tvt"] = best_delta
    features["pf_beam_mean_delta_tvt"] = mean_delta
    features["pf_beam_std_delta_tvt"] = std_pred
    features["pf_beam_range_delta_tvt"] = range_pred
    features["pf_beam_best_minus_recent"] = best_pred - recent_tvt
    features["pf_beam_mean_minus_recent"] = mean_pred - recent_tvt
    features["pf_beam_max_pairwise_delta"] = range_pred
    features["pf_beam_eval_length_selector"] = float(n_eval >= eval_length_threshold)
    features["pf_beam_z_span_selector"] = float(z_span >= z_span_threshold)
    return features[PF_BEAM_FEATURE_COLUMNS]


def dtw_dwt_enabled(config: dict[str, Any]) -> bool:
    return bool(config_get(config, "model.dtw_dwt.enabled", False))


def dtw_dwt_scale_windows(config: dict[str, Any]) -> dict[str, int]:
    raw = config_get(config, "model.dtw_dwt.scale_windows", None)
    defaults = {"w16": 16, "w32": 32, "w64": 64, "w128": 128}
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError("model.dtw_dwt.scale_windows must be a mapping")

    windows: dict[str, int] = {}
    for name in DTW_DWT_SCALE_NAMES:
        value = int(raw.get(name, defaults[name]))
        windows[name] = max(3, value)
    return windows


def empty_dtw_dwt_features(n_rows: int) -> pd.DataFrame:
    data = {column: np.full(n_rows, np.nan, dtype=float) for column in DTW_DWT_FEATURE_COLUMNS}
    zero_columns = [
        "dtw_dwt_available",
        "dtw_dwt_best_cost",
        "dtw_dwt_second_cost",
        "dtw_dwt_cost_margin",
        "dtw_dwt_best_shift_tvt",
        "dtw_dwt_best_ncc",
        "dtw_dwt_best_dtw_cost",
        "dtw_dwt_best_slope",
        "dtw_dwt_confidence",
        "dtw_dwt_candidate_count",
        "dtw_dwt_recent_delta_tvt",
        "dtw_dwt_best_delta_tvt",
        "dtw_dwt_best_minus_recent",
        "dtw_dwt_eval_energy_mean",
        "dtw_dwt_type_energy_mean",
        "dtw_dwt_energy_abs_error_mean",
        "dtw_dwt_smooth_abs_error_mean",
        "dtw_dwt_eval_length_selector",
        "dtw_dwt_best_gr_abs_error",
    ]
    for column in zero_columns:
        data[column] = np.zeros(n_rows, dtype=float)
    for name in DTW_DWT_SCALE_NAMES:
        for suffix in (
            "eval_energy",
            "type_energy",
            "energy_abs_error",
            "smooth_abs_error",
        ):
            data[f"dtw_dwt_{name}_{suffix}"] = np.zeros(n_rows, dtype=float)
    return pd.DataFrame(data, columns=DTW_DWT_FEATURE_COLUMNS)


def rolling_detail_energy(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    filled = interpolate_finite_by_index(values)
    if not np.isfinite(filled).any():
        return (
            np.full(values.shape, np.nan, dtype=float),
            np.full(values.shape, np.nan, dtype=float),
        )
    smooth = centered_mean(filled, window)
    detail = filled - smooth
    energy = centered_mean(detail * detail, window)
    return smooth.astype(float), energy.astype(float)


def standardize_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = finite_std(values, default=1.0)
    return (values - finite_mean(values)) / max(scale, 1e-6)


def score_dtw_dwt_candidate(
    *,
    eval_md: np.ndarray,
    eval_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    prefix: PrefixPrediction,
    shift: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    delta_md = eval_md - prefix.last_known_md
    base_slope = effective_pf_beam_prior_slope(prefix, config)
    candidate_tvt = prefix.last_known_tvt + base_slope * delta_md + shift
    sampled_gr = interpolate_typewell_gr(candidate_tvt, type_tvt, type_gr)
    sampled_gr = interpolate_finite_by_index(sampled_gr)

    scale_rows: list[dict[str, np.ndarray]] = []
    scale_costs: list[float] = []
    smooth_abs_errors: list[np.ndarray] = []
    energy_abs_errors: list[np.ndarray] = []
    min_valid_fraction = float(config_get(config, "model.dtw_dwt.min_valid_fraction", 0.30))
    for window in dtw_dwt_scale_windows(config).values():
        eval_smooth, eval_energy = rolling_detail_energy(eval_gr, window)
        type_smooth, type_energy = rolling_detail_energy(sampled_gr, window)
        valid = (
            np.isfinite(eval_smooth)
            & np.isfinite(type_smooth)
            & np.isfinite(eval_energy)
            & np.isfinite(type_energy)
        )
        min_valid = max(5, int(np.ceil(eval_gr.size * min_valid_fraction)))
        if int(valid.sum()) < min_valid:
            continue

        smooth_error = np.abs(standardize_finite(eval_smooth) - standardize_finite(type_smooth))
        energy_error = np.abs(
            standardize_finite(eval_energy) - standardize_finite(type_energy)
        )
        scale_rows.append(
            {
                "eval_smooth": eval_smooth,
                "type_smooth": type_smooth,
                "eval_energy": eval_energy,
                "type_energy": type_energy,
                "smooth_abs_error": smooth_error,
                "energy_abs_error": energy_error,
            }
        )
        smooth_abs_errors.append(smooth_error)
        energy_abs_errors.append(energy_error)
        scale_costs.append(float(np.nanmean(smooth_error[valid] + 0.5 * energy_error[valid])))

    if not scale_costs:
        return {
            "cost": np.inf,
            "shift": shift,
            "candidate_tvt": candidate_tvt,
            "sampled_gr": sampled_gr,
            "ncc": np.nan,
            "dtw_cost": np.nan,
            "scale_rows": [],
        }

    eval_smooth_main = scale_rows[min(1, len(scale_rows) - 1)]["eval_smooth"]
    type_smooth_main = scale_rows[min(1, len(scale_rows) - 1)]["type_smooth"]
    ncc = finite_pair_ncc(
        eval_smooth_main,
        type_smooth_main,
        min_valid_fraction=min_valid_fraction,
    )
    dtw_cost = normalized_banded_dtw_cost(
        eval_smooth_main,
        type_smooth_main,
        max_points=int(config_get(config, "model.dtw_dwt.dtw_max_points", 192)),
        band_fraction=float(config_get(config, "model.dtw_dwt.dtw_band_fraction", 0.10)),
    )
    cost = float(np.mean(scale_costs))
    if np.isfinite(dtw_cost):
        cost += float(config_get(config, "model.dtw_dwt.dtw_cost_weight", 0.15)) * dtw_cost
    if np.isfinite(ncc):
        cost += float(config_get(config, "model.dtw_dwt.ncc_cost_weight", 0.10)) * (1.0 - ncc)
    return {
        "cost": cost,
        "shift": shift,
        "candidate_tvt": candidate_tvt,
        "sampled_gr": sampled_gr,
        "ncc": ncc,
        "dtw_cost": dtw_cost,
        "scale_rows": scale_rows,
    }


def build_dtw_dwt_features(
    *,
    md: np.ndarray,
    gr: np.ndarray,
    typewell_df: pd.DataFrame | None,
    prefix: PrefixPrediction,
    eval_indices: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    n_eval = int(eval_indices.size)
    features = empty_dtw_dwt_features(n_eval)
    if n_eval == 0 or not dtw_dwt_enabled(config):
        return features

    curve = clean_typewell_curve(typewell_df)
    if curve is None:
        return features
    type_tvt, type_gr = curve

    eval_md = md[eval_indices]
    eval_gr = gr[eval_indices]
    delta_md = eval_md - prefix.last_known_md
    recent_tvt = prefix.last_known_tvt + prefix.recent_slope * delta_md
    radius = float(config_get(config, "model.dtw_dwt.shift_radius", 180.0))
    step = max(float(config_get(config, "model.dtw_dwt.shift_step", 15.0)), 1.0)
    shifts = np.arange(-radius, radius + step * 0.5, step, dtype=float)

    records = [
        score_dtw_dwt_candidate(
            eval_md=eval_md,
            eval_gr=eval_gr,
            type_tvt=type_tvt,
            type_gr=type_gr,
            prefix=prefix,
            shift=float(shift),
            config=config,
        )
        for shift in shifts
    ]
    finite_records = [record for record in records if np.isfinite(float(record["cost"]))]
    if not finite_records:
        return features

    finite_records.sort(key=lambda record: float(record["cost"]))
    best_record = finite_records[0]
    best_cost = float(best_record["cost"])
    second_cost = float(finite_records[1]["cost"]) if len(finite_records) > 1 else best_cost
    margin = max(0.0, second_cost - best_cost)
    confidence = margin / (1.0 + abs(best_cost))

    best_pred = np.asarray(best_record["candidate_tvt"], dtype=float)
    best_type_gr = np.asarray(best_record["sampled_gr"], dtype=float)
    best_delta = best_pred - prefix.last_known_tvt
    best_slope = finite_mean(safe_divide(best_delta, delta_md))
    scale_rows = list(best_record["scale_rows"])
    windows = list(dtw_dwt_scale_windows(config))
    eval_energy_stack: list[np.ndarray] = []
    type_energy_stack: list[np.ndarray] = []
    energy_error_stack: list[np.ndarray] = []
    smooth_error_stack: list[np.ndarray] = []
    for name, row in zip(windows, scale_rows, strict=False):
        eval_energy = np.asarray(row["eval_energy"], dtype=float)
        type_energy = np.asarray(row["type_energy"], dtype=float)
        energy_error = np.asarray(row["energy_abs_error"], dtype=float)
        smooth_error = np.asarray(row["smooth_abs_error"], dtype=float)
        eval_energy_stack.append(eval_energy)
        type_energy_stack.append(type_energy)
        energy_error_stack.append(energy_error)
        smooth_error_stack.append(smooth_error)
        features[f"dtw_dwt_{name}_eval_energy"] = eval_energy
        features[f"dtw_dwt_{name}_type_energy"] = type_energy
        features[f"dtw_dwt_{name}_energy_abs_error"] = energy_error
        features[f"dtw_dwt_{name}_smooth_abs_error"] = smooth_error

    eval_length_threshold = float(config_get(config, "model.dtw_dwt.long_eval_rows", 5700.0))
    features["dtw_dwt_available"] = 1.0
    features["dtw_dwt_best_cost"] = best_cost
    features["dtw_dwt_second_cost"] = second_cost
    features["dtw_dwt_cost_margin"] = margin
    features["dtw_dwt_best_shift_tvt"] = float(best_record["shift"])
    features["dtw_dwt_best_ncc"] = (
        float(best_record["ncc"]) if np.isfinite(float(best_record["ncc"])) else 0.0
    )
    features["dtw_dwt_best_dtw_cost"] = (
        float(best_record["dtw_cost"])
        if np.isfinite(float(best_record["dtw_cost"]))
        else best_cost
    )
    features["dtw_dwt_best_slope"] = best_slope
    features["dtw_dwt_confidence"] = confidence
    features["dtw_dwt_candidate_count"] = float(len(finite_records))
    features["dtw_dwt_recent_delta_tvt"] = recent_tvt - prefix.last_known_tvt
    features["dtw_dwt_best_delta_tvt"] = best_delta
    features["dtw_dwt_best_minus_recent"] = best_pred - recent_tvt
    features["dtw_dwt_eval_energy_mean"] = np.nanmean(eval_energy_stack, axis=0)
    features["dtw_dwt_type_energy_mean"] = np.nanmean(type_energy_stack, axis=0)
    features["dtw_dwt_energy_abs_error_mean"] = np.nanmean(energy_error_stack, axis=0)
    features["dtw_dwt_smooth_abs_error_mean"] = np.nanmean(smooth_error_stack, axis=0)
    features["dtw_dwt_eval_length_selector"] = float(n_eval >= eval_length_threshold)
    features["dtw_dwt_best_pred_tvt"] = best_pred
    features["dtw_dwt_best_typewell_gr"] = best_type_gr
    features["dtw_dwt_best_gr_abs_error"] = np.abs(eval_gr - best_type_gr)
    return features[DTW_DWT_FEATURE_COLUMNS]


def build_drift_feature_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_target: bool,
    typewell_df: pd.DataFrame | None = None,
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
    base_features = pd.DataFrame(
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
    )[BASE_FEATURE_COLUMNS]
    pf_beam_features = build_pf_beam_features(
        md=md,
        z=z,
        gr=gr,
        typewell_df=typewell_df,
        prefix=prefix,
        eval_indices=eval_indices,
        config=config,
    )
    dtw_dwt_features = build_dtw_dwt_features(
        md=md,
        gr=gr,
        typewell_df=typewell_df,
        prefix=prefix,
        eval_indices=eval_indices,
        config=config,
    )
    features = pd.concat(
        [
            base_features.reset_index(drop=True),
            pf_beam_features.reset_index(drop=True),
            dtw_dwt_features.reset_index(drop=True),
        ],
        axis=1,
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
        typewell_df = read_typewell_for_horizontal_path(path)
        frame = build_drift_feature_frame(
            df,
            config,
            include_target=True,
            typewell_df=typewell_df,
        )
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


def _odd_window(value: int, n_rows: int) -> int:
    window = max(1, int(value))
    if window % 2 == 0:
        window += 1
    if n_rows > 0 and window > n_rows:
        window = n_rows if n_rows % 2 == 1 else max(1, n_rows - 1)
    return max(1, window)


def smooth_prediction(values: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
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


def distance_bucket_alphas(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    eval_step = np.asarray(eval_step, dtype=float)
    alphas = np.ones(eval_step.shape, dtype=float)
    if not buckets:
        return alphas

    previous_max = -np.inf
    for bucket in buckets:
        max_step = float(bucket.get("max_step", np.inf))
        alpha = float(bucket.get("alpha", 1.0))
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        alphas[mask] = alpha
        previous_max = max_step
    return alphas


def postprocess_predictions(
    raw_prediction: np.ndarray,
    frame: DriftFeatureFrame,
    config: dict[str, Any],
    *,
    method: str | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    raw_prediction = np.asarray(raw_prediction, dtype=float)
    if raw_prediction.size == 0:
        return raw_prediction.copy()

    method = method or str(config_get(config, "postprocess.selected_method", "raw"))
    params = params or config_get(config, f"postprocess.methods.{method}", {})
    if not isinstance(params, dict):
        raise ValueError(f"postprocess.methods.{method} must be a mapping")

    anchor = np.asarray(frame.baseline_prediction, dtype=float)
    residual = raw_prediction - anchor
    eval_step = frame.features["eval_step"].to_numpy(dtype=float)

    if method in {"raw", "raw_lightgbm_no_gr"}:
        return raw_prediction.copy()
    if method == "sg_smooth":
        smoothed = smooth_prediction(
            raw_prediction,
            window=int(params.get("window", 21)),
            polyorder=int(params.get("polyorder", 2)),
        )
        blend = float(params.get("blend", 1.0))
        return raw_prediction * (1.0 - blend) + smoothed * blend
    if method == "global_residual_shrink":
        alpha = float(params.get("alpha", 1.0))
        return anchor + alpha * residual
    if method == "near_anchor_damping":
        near_rows = max(1.0, float(params.get("near_rows", 50)))
        near_alpha = float(params.get("near_alpha", 0.20))
        far_alpha = float(params.get("far_alpha", 1.0))
        progress = np.clip(eval_step / near_rows, 0.0, 1.0)
        alpha = near_alpha + (far_alpha - near_alpha) * progress
        return anchor + alpha * residual
    if method == "distance_bucket_shrink":
        buckets = params.get("buckets", [])
        if not isinstance(buckets, list):
            raise ValueError("postprocess distance_bucket_shrink buckets must be a list")
        alpha = distance_bucket_alphas(eval_step, buckets)
        return anchor + alpha * residual

    raise ValueError(f"unsupported postprocess method: {method}")
