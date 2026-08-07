from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
PREFIX_STRATEGIES = {"last_anchor", "recent_linear"}
DEFAULT_DRIFT_STRATEGY = "drift_hgb"
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
BASE_FEATURE_COLUMNS = [
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
TRAJECTORY_DIRECTION_FEATURE_COLUMNS = [
    "anchor_azimuth_sin",
    "anchor_azimuth_cos",
    "prefix_azimuth_sin",
    "prefix_azimuth_cos",
    "step_azimuth_sin",
    "step_azimuth_cos",
    "turn_from_prefix_sin",
    "turn_from_prefix_cos",
    "delta_along_final_axis",
    "delta_cross_final_axis",
    "delta_x_sign",
    "delta_y_sign",
    "delta_z_sign",
]
TRAJECTORY_SLOPE_FEATURE_COLUMNS = [
    "anchor_dx_dmd",
    "anchor_dy_dmd",
    "anchor_dxy_dmd",
    "anchor_dz_dxy",
    "anchor_inclination_sin",
    "anchor_horizontal_fraction",
    "step_dx_dmd",
    "step_dy_dmd",
    "step_dxy_dmd",
    "step_dz_dxy",
    "step_inclination_sin",
    "step_horizontal_fraction",
    "prefix_dx_dmd",
    "prefix_dy_dmd",
    "prefix_dz_dmd",
    "prefix_dxy_dmd",
]
TRAJECTORY_INTERACTION_FEATURE_COLUMNS = [
    "recent_tvt_slope_x_delta_md",
    "recent_tvt_slope_x_anchor_dz_dmd",
    "anchor_dz_dmd_minus_prefix_dz_dmd",
    "anchor_dxy_dmd_minus_prefix_dxy_dmd",
    "eval_progress_x_anchor_dz_dmd",
    "eval_progress_x_delta_xy",
]
TRAJECTORY_DRIFT_FEATURE_COLUMNS = [
    *TRAJECTORY_DIRECTION_FEATURE_COLUMNS,
    *TRAJECTORY_SLOPE_FEATURE_COLUMNS,
    *TRAJECTORY_INTERACTION_FEATURE_COLUMNS,
]
NCC_CANDIDATE_NAMES = ("w25_r150", "w75_r300")
NCC_SUMMARY_FEATURE_COLUMNS = [
    "gr_ncc_match_available",
    "gr_ncc_best_score",
    "gr_ncc_second_score",
    "gr_ncc_confidence",
    "gr_ncc_ambiguity",
    "gr_ncc_best_shift_tvt",
    "gr_ncc_best_direction",
    "gr_ncc_best_window",
    "gr_ncc_best_radius",
    "gr_ncc_matched_tvt",
    "gr_ncc_delta_tvt_from_anchor",
    "gr_ncc_typewell_gr",
    "gr_ncc_gr_delta",
    "gr_ncc_abs_gr_delta",
    "gr_ncc_prior_typewell_gr",
    "gr_ncc_prior_gr_delta",
]
NCC_CANDIDATE_FEATURE_COLUMNS = [
    f"gr_ncc_{name}_{suffix}"
    for name in NCC_CANDIDATE_NAMES
    for suffix in ("score", "shift_tvt", "confidence", "direction")
]
NCC_FEATURE_COLUMNS = [*NCC_SUMMARY_FEATURE_COLUMNS, *NCC_CANDIDATE_FEATURE_COLUMNS]
FORMATION_GUIDE_SUMMARY_FEATURE_COLUMNS = [
    "formation_guide_available",
    "formation_guide_min_abs_dz",
    "formation_guide_signed_dz_to_nearest",
    "formation_guide_nearest_index",
    "formation_guide_mean_abs_dz",
    "formation_guide_surface_span",
]
FORMATION_GUIDE_PER_FORMATION_FEATURE_COLUMNS = [
    f"formation_{name.lower()}_{suffix}"
    for name in FORMATION_COLUMNS
    for suffix in ("surface_z", "dz", "abs_dz")
]
FORMATION_GUIDE_FEATURE_COLUMNS = [
    *FORMATION_GUIDE_SUMMARY_FEATURE_COLUMNS,
    *FORMATION_GUIDE_PER_FORMATION_FEATURE_COLUMNS,
]
FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *TRAJECTORY_DRIFT_FEATURE_COLUMNS,
    *NCC_FEATURE_COLUMNS,
    *FORMATION_GUIDE_FEATURE_COLUMNS,
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
    "all": BASE_FEATURE_COLUMNS,
    "no_gr_roll": [
        column for column in BASE_FEATURE_COLUMNS if column not in GR_ROLL_FEATURE_COLUMNS
    ],
    "no_gr_signal": [column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
    "all_plus_trajectory_direction": [
        *BASE_FEATURE_COLUMNS,
        *TRAJECTORY_DIRECTION_FEATURE_COLUMNS,
    ],
    "no_gr_signal_plus_trajectory_direction": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *TRAJECTORY_DIRECTION_FEATURE_COLUMNS,
    ],
    "all_plus_trajectory_slope": [
        *BASE_FEATURE_COLUMNS,
        *TRAJECTORY_SLOPE_FEATURE_COLUMNS,
    ],
    "no_gr_signal_plus_trajectory_slope": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *TRAJECTORY_SLOPE_FEATURE_COLUMNS,
    ],
    "all_plus_trajectory_drift": [
        *BASE_FEATURE_COLUMNS,
        *TRAJECTORY_DRIFT_FEATURE_COLUMNS,
    ],
    "no_gr_signal_plus_trajectory_drift": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *TRAJECTORY_DRIFT_FEATURE_COLUMNS,
    ],
    "all_plus_gr_ncc": [*BASE_FEATURE_COLUMNS, *NCC_FEATURE_COLUMNS],
    "no_gr_signal_plus_gr_ncc": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *NCC_FEATURE_COLUMNS,
    ],
    "all_plus_formation_guide": [*BASE_FEATURE_COLUMNS, *FORMATION_GUIDE_FEATURE_COLUMNS],
    "no_gr_signal_plus_formation_guide": [
        *[column for column in BASE_FEATURE_COLUMNS if column not in GR_FEATURE_COLUMNS],
        *FORMATION_GUIDE_FEATURE_COLUMNS,
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
    well_features: dict[str, float]
    last_known_index: int
    last_known_md: float
    last_known_tvt: float
    recent_slope: float


@dataclass(frozen=True)
class GrGatedModelBundle:
    base_model: Any
    alternate_model: Any
    base_config: dict[str, Any]
    alternate_config: dict[str, Any]
    n_train_rows_base: int
    n_train_rows_alternate: int


@dataclass(frozen=True)
class HardRouterModelBundle:
    all_gr_model: Any
    no_gr_model: Any
    all_gr_config: dict[str, Any]
    no_gr_config: dict[str, Any]
    guarded_config: dict[str, Any]
    n_train_rows_all_gr: int
    n_train_rows_no_gr: int


@dataclass(frozen=True)
class FormationGuideModel:
    formation_columns: tuple[str, ...]
    xy_mean: np.ndarray
    xy_scale: np.ndarray
    model: Any

    def predict_surfaces(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        coordinates = np.column_stack([x, y]).astype(float)
        scaled = (coordinates - self.xy_mean) / self.xy_scale
        return self.model.predict(scaled).astype(float)


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
    return path.with_name(path.name.replace(HORIZONTAL_SUFFIX, "__typewell.csv"))


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


def config_with_feature_set(config: dict[str, Any], feature_set: str) -> dict[str, Any]:
    updated = deepcopy(config)
    model_config = updated.setdefault("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model_config["feature_set"] = feature_set
    return updated


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


def finite_rate(mask: np.ndarray, default: float = 0.0) -> float:
    if mask.size == 0:
        return default
    return float(np.mean(mask))


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def safe_scalar_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return default
    return float(numerator / denominator)


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


def gr_ncc_enabled(config: dict[str, Any]) -> bool:
    return bool(config_get(config, "model.gr_ncc.enabled", False))


def ncc_candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = [
        {"name": "w25_r150", "window": 25, "radius": 150.0, "shift_step": 10.0},
        {"name": "w75_r300", "window": 75, "radius": 300.0, "shift_step": 20.0},
    ]
    raw_specs = config_get(config, "model.gr_ncc.candidates", defaults)
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("model.gr_ncc.candidates must be a non-empty list")

    specs: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"model.gr_ncc.candidates[{index}] must be a mapping")
        name = str(raw_spec.get("name") or "")
        if name not in NCC_CANDIDATE_NAMES:
            raise ValueError(
                f"unsupported GR NCC candidate name {name!r}; "
                f"expected one of {list(NCC_CANDIDATE_NAMES)}"
            )
        specs.append(
            {
                "name": name,
                "window": int(raw_spec.get("window", 25)),
                "radius": float(raw_spec.get("radius", 150.0)),
                "shift_step": float(raw_spec.get("shift_step", 10.0)),
            }
        )
    return specs


def empty_gr_ncc_features(n_rows: int) -> pd.DataFrame:
    data = {column: np.full(n_rows, np.nan, dtype=float) for column in NCC_FEATURE_COLUMNS}
    zero_columns = [
        "gr_ncc_match_available",
        "gr_ncc_best_score",
        "gr_ncc_second_score",
        "gr_ncc_confidence",
        "gr_ncc_ambiguity",
        "gr_ncc_best_direction",
        "gr_ncc_best_window",
        "gr_ncc_best_radius",
    ]
    for column in zero_columns:
        data[column] = np.zeros(n_rows, dtype=float)
    for name in NCC_CANDIDATE_NAMES:
        data[f"gr_ncc_{name}_score"] = np.zeros(n_rows, dtype=float)
        data[f"gr_ncc_{name}_confidence"] = np.zeros(n_rows, dtype=float)
        data[f"gr_ncc_{name}_direction"] = np.zeros(n_rows, dtype=float)
    return pd.DataFrame(data, columns=NCC_FEATURE_COLUMNS)


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


def effective_ncc_prior_slope(prefix: PrefixPrediction, config: dict[str, Any]) -> float:
    slope = float(prefix.recent_slope)
    min_abs_slope = float(config_get(config, "model.gr_ncc.min_abs_prior_slope", 0.005))
    if abs(slope) >= min_abs_slope:
        return slope

    fallback = abs(float(config_get(config, "model.gr_ncc.fallback_abs_prior_slope", 0.03)))
    sign = -1.0 if slope < 0.0 else 1.0
    return sign * fallback


def score_gr_ncc_candidate(
    *,
    eval_md: np.ndarray,
    eval_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    prefix: PrefixPrediction,
    spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    window = int(spec["window"])
    radius = float(spec["radius"])
    shift_step = max(float(spec["shift_step"]), 1.0)
    min_valid_fraction = float(config_get(config, "model.gr_ncc.min_valid_fraction", 0.30))
    allow_reverse = bool(config_get(config, "model.gr_ncc.allow_reverse", True))

    eval_smooth = centered_mean(eval_gr, window)
    shifts = np.arange(-radius, radius + shift_step * 0.5, shift_step, dtype=float)
    directions = (1.0, -1.0) if allow_reverse else (1.0,)
    base_slope = effective_ncc_prior_slope(prefix, config)
    delta_md = eval_md - prefix.last_known_md

    best_score = -np.inf
    second_score = -np.inf
    best_shift = np.nan
    best_direction = 0.0
    for direction in directions:
        prior_tvt = prefix.last_known_tvt + base_slope * direction * delta_md
        for shift in shifts:
            sampled_gr = interpolate_typewell_gr(prior_tvt + shift, type_tvt, type_gr)
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
                best_direction = float(direction)
            elif score > second_score:
                second_score = score

    if not np.isfinite(best_score):
        best_score = 0.0
        second_score = 0.0
        best_shift = np.nan
        best_direction = 0.0
    elif not np.isfinite(second_score):
        second_score = best_score

    confidence = max(0.0, float(best_score - second_score))
    return {
        "score": float(best_score),
        "second_score": float(second_score),
        "confidence": confidence,
        "shift_tvt": float(best_shift),
        "direction": float(best_direction),
        "window": float(window),
        "radius": float(radius),
    }


def build_gr_ncc_features(
    *,
    md: np.ndarray,
    gr: np.ndarray,
    typewell_df: pd.DataFrame | None,
    prefix: PrefixPrediction,
    eval_indices: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    n_eval = int(eval_indices.size)
    features = empty_gr_ncc_features(n_eval)
    if n_eval == 0 or not gr_ncc_enabled(config):
        return features

    curve = clean_typewell_curve(typewell_df)
    if curve is None:
        return features
    type_tvt, type_gr = curve

    eval_md = md[eval_indices]
    eval_gr = gr[eval_indices]
    records: dict[str, dict[str, float]] = {}
    for spec in ncc_candidate_specs(config):
        record = score_gr_ncc_candidate(
            eval_md=eval_md,
            eval_gr=eval_gr,
            type_tvt=type_tvt,
            type_gr=type_gr,
            prefix=prefix,
            spec=spec,
            config=config,
        )
        name = str(spec["name"])
        records[name] = record
        features[f"gr_ncc_{name}_score"] = record["score"]
        features[f"gr_ncc_{name}_shift_tvt"] = record["shift_tvt"]
        features[f"gr_ncc_{name}_confidence"] = record["confidence"]
        features[f"gr_ncc_{name}_direction"] = record["direction"]

    best_name, best_record = max(records.items(), key=lambda item: item[1]["score"])
    all_scores = sorted((record["score"] for record in records.values()), reverse=True)
    second_score = best_record["second_score"]
    if len(all_scores) > 1:
        second_score = max(second_score, all_scores[1])
    confidence = max(0.0, float(best_record["score"] - second_score))

    if best_record["direction"] == 0.0 or not np.isfinite(best_record["shift_tvt"]):
        return features

    base_slope = effective_ncc_prior_slope(prefix, config)
    delta_md = eval_md - prefix.last_known_md
    prior_tvt = prefix.last_known_tvt + base_slope * best_record["direction"] * delta_md
    matched_tvt = prior_tvt + best_record["shift_tvt"]
    typewell_gr = interpolate_typewell_gr(matched_tvt, type_tvt, type_gr)
    prior_typewell_gr = interpolate_typewell_gr(prior_tvt, type_tvt, type_gr)

    features["gr_ncc_match_available"] = 1.0
    features["gr_ncc_best_score"] = best_record["score"]
    features["gr_ncc_second_score"] = second_score
    features["gr_ncc_confidence"] = confidence
    features["gr_ncc_ambiguity"] = max(0.0, 1.0 - confidence)
    features["gr_ncc_best_shift_tvt"] = best_record["shift_tvt"]
    features["gr_ncc_best_direction"] = best_record["direction"]
    features["gr_ncc_best_window"] = best_record["window"]
    features["gr_ncc_best_radius"] = best_record["radius"]
    features["gr_ncc_matched_tvt"] = matched_tvt
    features["gr_ncc_delta_tvt_from_anchor"] = matched_tvt - prefix.last_known_tvt
    features["gr_ncc_typewell_gr"] = typewell_gr
    features["gr_ncc_gr_delta"] = eval_gr - typewell_gr
    features["gr_ncc_abs_gr_delta"] = np.abs(eval_gr - typewell_gr)
    features["gr_ncc_prior_typewell_gr"] = prior_typewell_gr
    features["gr_ncc_prior_gr_delta"] = eval_gr - prior_typewell_gr
    return features[NCC_FEATURE_COLUMNS]


def formation_guide_enabled(config: dict[str, Any]) -> bool:
    return bool(config_get(config, "model.formation_guide.enabled", False))


def formation_guide_columns(config: dict[str, Any]) -> tuple[str, ...]:
    raw_columns = config_get(
        config,
        "model.formation_guide.formation_columns",
        list(FORMATION_COLUMNS),
    )
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("model.formation_guide.formation_columns must be a non-empty list")

    columns = tuple(str(column) for column in raw_columns)
    unsupported = sorted(set(columns) - set(FORMATION_COLUMNS))
    if unsupported:
        raise ValueError(f"unsupported formation guide columns: {unsupported}")
    return columns


def empty_formation_guide_features(n_rows: int) -> pd.DataFrame:
    data = {
        column: np.full(n_rows, np.nan, dtype=float)
        for column in FORMATION_GUIDE_FEATURE_COLUMNS
    }
    data["formation_guide_available"] = np.zeros(n_rows, dtype=float)
    data["formation_guide_nearest_index"] = np.full(n_rows, -1.0, dtype=float)
    return pd.DataFrame(data, columns=FORMATION_GUIDE_FEATURE_COLUMNS)


def sample_formation_rows(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    rng: np.random.Generator,
    max_rows_per_well: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    required_columns = {"X", "Y", *columns}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        return np.empty((0, 2), dtype=float), np.empty((0, len(columns)), dtype=float)

    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    surfaces = df.loc[:, list(columns)].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(surfaces).all(axis=1)
    indices = np.flatnonzero(valid)
    if max_rows_per_well is not None and indices.size > max_rows_per_well:
        indices = rng.choice(indices, size=max_rows_per_well, replace=False)
    if indices.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, len(columns)), dtype=float)
    coordinates = np.column_stack([x[indices], y[indices]]).astype(float)
    return coordinates, surfaces[indices].astype(float)


def fit_formation_guide_from_files(
    files: Iterable[Path],
    config: dict[str, Any],
    *,
    seed: int,
) -> FormationGuideModel | None:
    if not formation_guide_enabled(config):
        return None

    columns = formation_guide_columns(config)
    rng = np.random.default_rng(seed)
    max_rows_total = optional_positive_int(
        config_get(config, "model.formation_guide.max_rows_total", 200000)
    )
    max_rows_per_well = optional_positive_int(
        config_get(config, "model.formation_guide.max_rows_per_well", 300)
    )

    coordinate_parts: list[np.ndarray] = []
    surface_parts: list[np.ndarray] = []
    for path in files:
        coordinates, surfaces = sample_formation_rows(
            pd.read_csv(path),
            columns=columns,
            rng=rng,
            max_rows_per_well=max_rows_per_well,
        )
        if coordinates.size:
            coordinate_parts.append(coordinates)
            surface_parts.append(surfaces)

    if not coordinate_parts:
        return None

    coordinates = np.vstack(coordinate_parts)
    surfaces = np.vstack(surface_parts)
    if max_rows_total is not None and coordinates.shape[0] > max_rows_total:
        selected = rng.choice(coordinates.shape[0], size=max_rows_total, replace=False)
        coordinates = coordinates[selected]
        surfaces = surfaces[selected]

    xy_mean = np.mean(coordinates, axis=0)
    xy_scale = np.std(coordinates, axis=0)
    xy_scale = np.where(np.isfinite(xy_scale) & (xy_scale > 0.0), xy_scale, 1.0)
    scaled_coordinates = (coordinates - xy_mean) / xy_scale

    n_neighbors = min(
        int(config_get(config, "model.formation_guide.n_neighbors", 24)),
        int(scaled_coordinates.shape[0]),
    )
    if n_neighbors < 1:
        return None
    weights = str(config_get(config, "model.formation_guide.weights", "distance"))
    if weights not in {"uniform", "distance"}:
        raise ValueError("model.formation_guide.weights must be uniform or distance")

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(scaled_coordinates, surfaces)
    return FormationGuideModel(
        formation_columns=columns,
        xy_mean=xy_mean,
        xy_scale=xy_scale,
        model=model,
    )


def build_formation_guide_features(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    eval_indices: np.ndarray,
    formation_guide: FormationGuideModel | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    n_eval = int(eval_indices.size)
    features = empty_formation_guide_features(n_eval)
    if n_eval == 0 or not formation_guide_enabled(config) or formation_guide is None:
        return features

    eval_x = x[eval_indices]
    eval_y = y[eval_indices]
    eval_z = z[eval_indices]
    surfaces = formation_guide.predict_surfaces(eval_x, eval_y)
    if surfaces.shape != (n_eval, len(formation_guide.formation_columns)):
        raise ValueError("formation guide prediction shape does not match configured columns")

    dz = eval_z.reshape(-1, 1) - surfaces
    abs_dz = np.abs(dz)
    finite_abs_dz = np.where(np.isfinite(abs_dz), abs_dz, np.inf)
    nearest_index = np.argmin(finite_abs_dz, axis=1)
    row_index = np.arange(n_eval)
    nearest_abs_dz = finite_abs_dz[row_index, nearest_index]
    nearest_abs_dz = np.where(np.isfinite(nearest_abs_dz), nearest_abs_dz, np.nan)
    nearest_signed_dz = dz[row_index, nearest_index]
    surface_span = np.nanmax(surfaces, axis=1) - np.nanmin(surfaces, axis=1)

    features["formation_guide_available"] = 1.0
    features["formation_guide_min_abs_dz"] = nearest_abs_dz
    features["formation_guide_signed_dz_to_nearest"] = nearest_signed_dz
    features["formation_guide_nearest_index"] = nearest_index.astype(float)
    features["formation_guide_mean_abs_dz"] = np.nanmean(abs_dz, axis=1)
    features["formation_guide_surface_span"] = surface_span

    for column_index, column in enumerate(formation_guide.formation_columns):
        prefix = f"formation_{column.lower()}"
        features[f"{prefix}_surface_z"] = surfaces[:, column_index]
        features[f"{prefix}_dz"] = dz[:, column_index]
        features[f"{prefix}_abs_dz"] = abs_dz[:, column_index]

    return features[FORMATION_GUIDE_FEATURE_COLUMNS]


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


def build_well_condition_features(
    *,
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    prefix: PrefixPrediction,
) -> dict[str, float]:
    eval_indices = prefix.eval_indices
    last_index = prefix.last_known_index
    prefix_gr = gr[: last_index + 1]
    eval_gr = gr[eval_indices]
    prefix_gr_mean = finite_mean(prefix_gr)

    if eval_indices.size:
        eval_md = md[eval_indices]
        final_index = int(eval_indices[-1])
        trajectory_delta_x = float(x[final_index] - x[last_index])
        trajectory_delta_y = float(y[final_index] - y[last_index])
        trajectory_delta_z = float(z[final_index] - z[last_index])
        trajectory_delta_xy = float(
            np.sqrt(
                trajectory_delta_x * trajectory_delta_x
                + trajectory_delta_y * trajectory_delta_y
            )
        )
        trajectory_delta_xyz = float(
            np.sqrt(
                trajectory_delta_xy * trajectory_delta_xy
                + trajectory_delta_z * trajectory_delta_z
            )
        )
        final_delta_md = float(md[final_index] - prefix.last_known_md)
        trajectory_abs_dz_dmd = (
            abs(trajectory_delta_z / final_delta_md) if final_delta_md != 0.0 else np.nan
        )
        eval_md_span = float(np.nanmax(eval_md) - np.nanmin(eval_md))
        gr_delta_abs_mean = finite_mean(np.abs(eval_gr - prefix_gr_mean))
        eval_gr_std = finite_std(eval_gr)
    else:
        trajectory_delta_xy = 0.0
        trajectory_delta_z = 0.0
        trajectory_delta_xyz = 0.0
        trajectory_abs_dz_dmd = 0.0
        eval_md_span = 0.0
        gr_delta_abs_mean = 0.0
        eval_gr_std = 0.0

    n_rows = int(md.size)
    known_row_count = int(last_index + 1)
    eval_row_count = int(eval_indices.size)
    prefix_md_span = float(prefix.last_known_md - md[0]) if md.size else 0.0
    return {
        "n_rows": float(n_rows),
        "known_row_count": float(known_row_count),
        "eval_row_count": float(eval_row_count),
        "prefix_fraction": float(known_row_count / n_rows) if n_rows else 0.0,
        "prefix_md_span": prefix_md_span,
        "eval_md_span": eval_md_span,
        "recent_slope": float(prefix.recent_slope),
        "abs_recent_slope": float(abs(prefix.recent_slope)),
        "prefix_gr_missing_rate": finite_rate(~np.isfinite(prefix_gr)),
        "eval_gr_missing_rate": finite_rate(~np.isfinite(eval_gr)),
        "prefix_gr_std": finite_std(prefix_gr),
        "eval_gr_std": eval_gr_std,
        "gr_delta_abs_mean": gr_delta_abs_mean,
        "trajectory_delta_xy": trajectory_delta_xy,
        "trajectory_delta_z": float(abs(trajectory_delta_z)),
        "trajectory_delta_xyz": trajectory_delta_xyz,
        "trajectory_abs_dz_dmd": float(trajectory_abs_dz_dmd),
    }


def build_drift_feature_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_target: bool,
    typewell_df: pd.DataFrame | None = None,
    formation_guide: FormationGuideModel | None = None,
) -> DriftFeatureFrame:
    required_columns = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    prefix = predict_from_prefix(df, config)
    eval_indices = prefix.eval_indices
    baseline_prediction = prefix.predictions["last_anchor"]

    md = df["MD"].to_numpy(dtype=float)
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    gr = df["GR"].to_numpy(dtype=float)
    well_features = build_well_condition_features(md=md, x=x, y=y, z=z, gr=gr, prefix=prefix)

    if eval_indices.size == 0:
        empty_features = pd.DataFrame(columns=FEATURE_COLUMNS)
        target_residual = np.asarray([], dtype=float) if include_target else None
        return DriftFeatureFrame(
            eval_indices=eval_indices,
            features=empty_features,
            baseline_prediction=baseline_prediction,
            target_residual=target_residual,
            well_features=well_features,
            last_known_index=prefix.last_known_index,
            last_known_md=prefix.last_known_md,
            last_known_tvt=prefix.last_known_tvt,
            recent_slope=prefix.recent_slope,
        )

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
    anchor_dz_dmd = safe_divide(delta_z, delta_md)
    anchor_dx_dmd = safe_divide(delta_x, delta_md)
    anchor_dy_dmd = safe_divide(delta_y, delta_md)
    anchor_dxy_dmd = safe_divide(delta_xy, delta_md)
    anchor_dz_dxy = safe_divide(delta_z, delta_xy)
    anchor_azimuth_sin = safe_divide(delta_y, delta_xy)
    anchor_azimuth_cos = safe_divide(delta_x, delta_xy)
    anchor_inclination_sin = safe_divide(delta_z, delta_xyz)
    anchor_horizontal_fraction = safe_divide(delta_xy, delta_xyz)

    step_md = np.diff(md, prepend=np.nan)
    step_x = np.diff(x, prepend=np.nan)
    step_y = np.diff(y, prepend=np.nan)
    step_z = np.diff(z, prepend=np.nan)
    step_xy = np.sqrt(step_x * step_x + step_y * step_y)
    step_xyz = np.sqrt(step_xy * step_xy + step_z * step_z)
    step_dx_dmd = safe_divide(step_x, step_md)
    step_dy_dmd = safe_divide(step_y, step_md)
    step_dxy_dmd = safe_divide(step_xy, step_md)
    step_dz_dmd = safe_divide(step_z, step_md)
    step_dz_dxy = safe_divide(step_z, step_xy)
    step_azimuth_sin = safe_divide(step_y, step_xy)
    step_azimuth_cos = safe_divide(step_x, step_xy)
    step_inclination_sin = safe_divide(step_z, step_xyz)
    step_horizontal_fraction = safe_divide(step_xy, step_xyz)

    trajectory_window = int(
        config_get(
            config,
            "model.params.trajectory_window",
            config_get(config, "model.params.recent_slope_window", 200),
        )
    )
    prefix_start = max(0, last_index - max(1, trajectory_window) + 1)
    prefix_delta_md = float(md[last_index] - md[prefix_start])
    prefix_delta_x = float(x[last_index] - x[prefix_start])
    prefix_delta_y = float(y[last_index] - y[prefix_start])
    prefix_delta_z = float(z[last_index] - z[prefix_start])
    prefix_delta_xy = float(
        np.sqrt(prefix_delta_x * prefix_delta_x + prefix_delta_y * prefix_delta_y)
    )
    prefix_dx_dmd = safe_scalar_divide(prefix_delta_x, prefix_delta_md)
    prefix_dy_dmd = safe_scalar_divide(prefix_delta_y, prefix_delta_md)
    prefix_dz_dmd = safe_scalar_divide(prefix_delta_z, prefix_delta_md)
    prefix_dxy_dmd = safe_scalar_divide(prefix_delta_xy, prefix_delta_md)
    prefix_azimuth_sin = safe_scalar_divide(prefix_delta_y, prefix_delta_xy)
    prefix_azimuth_cos = safe_scalar_divide(prefix_delta_x, prefix_delta_xy)

    final_delta_x = float(eval_x[-1] - last_x)
    final_delta_y = float(eval_y[-1] - last_y)
    final_delta_xy = float(np.sqrt(final_delta_x * final_delta_x + final_delta_y * final_delta_y))
    final_unit_x = safe_scalar_divide(final_delta_x, final_delta_xy)
    final_unit_y = safe_scalar_divide(final_delta_y, final_delta_xy)
    delta_along_final_axis = delta_x * final_unit_x + delta_y * final_unit_y
    delta_cross_final_axis = delta_x * final_unit_y - delta_y * final_unit_x
    turn_from_prefix_sin = (
        prefix_azimuth_cos * anchor_azimuth_sin
        - prefix_azimuth_sin * anchor_azimuth_cos
    )
    turn_from_prefix_cos = (
        prefix_azimuth_cos * anchor_azimuth_cos
        + prefix_azimuth_sin * anchor_azimuth_sin
    )

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
            "anchor_dz_dmd": anchor_dz_dmd,
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
    trajectory_features = pd.DataFrame(
        {
            "anchor_azimuth_sin": anchor_azimuth_sin,
            "anchor_azimuth_cos": anchor_azimuth_cos,
            "prefix_azimuth_sin": np.full(n_eval, prefix_azimuth_sin, dtype=float),
            "prefix_azimuth_cos": np.full(n_eval, prefix_azimuth_cos, dtype=float),
            "step_azimuth_sin": step_azimuth_sin[eval_indices],
            "step_azimuth_cos": step_azimuth_cos[eval_indices],
            "turn_from_prefix_sin": turn_from_prefix_sin,
            "turn_from_prefix_cos": turn_from_prefix_cos,
            "delta_along_final_axis": delta_along_final_axis,
            "delta_cross_final_axis": delta_cross_final_axis,
            "delta_x_sign": np.sign(delta_x),
            "delta_y_sign": np.sign(delta_y),
            "delta_z_sign": np.sign(delta_z),
            "anchor_dx_dmd": anchor_dx_dmd,
            "anchor_dy_dmd": anchor_dy_dmd,
            "anchor_dxy_dmd": anchor_dxy_dmd,
            "anchor_dz_dxy": anchor_dz_dxy,
            "anchor_inclination_sin": anchor_inclination_sin,
            "anchor_horizontal_fraction": anchor_horizontal_fraction,
            "step_dx_dmd": step_dx_dmd[eval_indices],
            "step_dy_dmd": step_dy_dmd[eval_indices],
            "step_dxy_dmd": step_dxy_dmd[eval_indices],
            "step_dz_dxy": step_dz_dxy[eval_indices],
            "step_inclination_sin": step_inclination_sin[eval_indices],
            "step_horizontal_fraction": step_horizontal_fraction[eval_indices],
            "prefix_dx_dmd": np.full(n_eval, prefix_dx_dmd, dtype=float),
            "prefix_dy_dmd": np.full(n_eval, prefix_dy_dmd, dtype=float),
            "prefix_dz_dmd": np.full(n_eval, prefix_dz_dmd, dtype=float),
            "prefix_dxy_dmd": np.full(n_eval, prefix_dxy_dmd, dtype=float),
            "recent_tvt_slope_x_delta_md": prefix.recent_slope * delta_md,
            "recent_tvt_slope_x_anchor_dz_dmd": prefix.recent_slope * anchor_dz_dmd,
            "anchor_dz_dmd_minus_prefix_dz_dmd": anchor_dz_dmd - prefix_dz_dmd,
            "anchor_dxy_dmd_minus_prefix_dxy_dmd": anchor_dxy_dmd - prefix_dxy_dmd,
            "eval_progress_x_anchor_dz_dmd": eval_progress * anchor_dz_dmd,
            "eval_progress_x_delta_xy": eval_progress * delta_xy,
        }
    )[TRAJECTORY_DRIFT_FEATURE_COLUMNS]
    ncc_features = build_gr_ncc_features(
        md=md,
        gr=gr,
        typewell_df=typewell_df,
        prefix=prefix,
        eval_indices=eval_indices,
        config=config,
    )
    formation_features = build_formation_guide_features(
        x=x,
        y=y,
        z=z,
        eval_indices=eval_indices,
        formation_guide=formation_guide,
        config=config,
    )
    features = pd.concat(
        [
            base_features.reset_index(drop=True),
            trajectory_features.reset_index(drop=True),
            ncc_features.reset_index(drop=True),
            formation_features.reset_index(drop=True),
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
        well_features=well_features,
        last_known_index=prefix.last_known_index,
        last_known_md=prefix.last_known_md,
        last_known_tvt=prefix.last_known_tvt,
        recent_slope=prefix.recent_slope,
    )


def make_drift_model(config: dict[str, Any], *, random_state: int | None = None) -> Any:
    seed = int(config_get(config, "validation.seed", 42)) if random_state is None else random_state
    return HistGradientBoostingRegressor(
        max_iter=int(config_get(config, "model.drift_model.params.max_iter", 180)),
        learning_rate=float(config_get(config, "model.drift_model.params.learning_rate", 0.05)),
        max_leaf_nodes=int(config_get(config, "model.drift_model.params.max_leaf_nodes", 31)),
        min_samples_leaf=int(config_get(config, "model.drift_model.params.min_samples_leaf", 80)),
        l2_regularization=float(
            config_get(config, "model.drift_model.params.l2_regularization", 0.05)
        ),
        max_bins=int(config_get(config, "model.drift_model.params.max_bins", 255)),
        early_stopping=False,
        random_state=seed,
    )


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
    files = list(files)
    rng = np.random.default_rng(seed)
    formation_guide = fit_formation_guide_from_files(files, config, seed=seed)
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
            formation_guide=formation_guide,
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
    model.formation_guide_ = formation_guide
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


def gr_gating_enabled(config: dict[str, Any]) -> bool:
    return bool(config_get(config, "model.gr_gating.enabled", False))


def gr_gating_feature_sets(config: dict[str, Any]) -> tuple[str, str]:
    base_feature_set = str(config_get(config, "model.gr_gating.base_feature_set", "all"))
    alternate_feature_set = str(
        config_get(config, "model.gr_gating.alternate_feature_set", "no_gr_signal")
    )
    return base_feature_set, alternate_feature_set


def hard_router_enabled(config: dict[str, Any]) -> bool:
    return bool(config_get(config, "model.hard_router.enabled", False))


def hard_router_feature_sets(config: dict[str, Any]) -> tuple[str, str]:
    all_gr_feature_set = str(config_get(config, "model.hard_router.all_gr_feature_set", "all"))
    no_gr_feature_set = str(
        config_get(config, "model.hard_router.no_gr_feature_set", "no_gr_signal")
    )
    return all_gr_feature_set, no_gr_feature_set


def compare_rule_value(left: float, operator: str, right: float) -> bool:
    if not np.isfinite(left):
        return False
    if operator == ">=":
        return left >= right
    if operator == ">":
        return left > right
    if operator == "<=":
        return left <= right
    if operator == "<":
        return left < right
    if operator in {"==", "="}:
        return left == right
    raise ValueError(f"unsupported gr_gating rule operator: {operator}")


def gr_gate_rule_matches(frame: DriftFeatureFrame, config: dict[str, Any]) -> list[bool]:
    raw_rules = config_get(config, "model.gr_gating.rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("model.gr_gating.rules must be a list")

    matches: list[bool] = []
    for index, rule in enumerate(raw_rules):
        if not isinstance(rule, dict):
            raise ValueError(f"model.gr_gating.rules[{index}] must be a mapping")
        feature_name = str(rule.get("feature") or "")
        if feature_name not in frame.well_features:
            raise ValueError(f"unknown gr_gating feature: {feature_name}")
        value = float(frame.well_features[feature_name])
        if bool(rule.get("abs", False)):
            value = abs(value)
        operator = str(rule.get("op", ">="))
        threshold = float(rule.get("value"))
        matches.append(compare_rule_value(value, operator, threshold))
    return matches


def gr_gate_weight(frame: DriftFeatureFrame, config: dict[str, Any]) -> float:
    if not gr_gating_enabled(config):
        return 0.0

    rule_matches = gr_gate_rule_matches(frame, config)
    combine = str(config_get(config, "model.gr_gating.combine", "any"))
    if not rule_matches:
        matched = False
    elif combine == "any":
        matched = any(rule_matches)
    elif combine == "all":
        matched = all(rule_matches)
    else:
        raise ValueError(f"unsupported model.gr_gating.combine: {combine}")

    default_weight = float(config_get(config, "model.gr_gating.default_weight", 0.0))
    alternate_weight = float(config_get(config, "model.gr_gating.alternate_weight", 1.0))
    weight = alternate_weight if matched else default_weight
    return float(np.clip(weight, 0.0, 1.0))


def hard_router_thresholds(config: dict[str, Any]) -> dict[str, float]:
    thresholds = {
        "prefix_gr_missing_rate": 0.35,
        "eval_gr_missing_rate": 0.40,
        "prefix_fraction_short": 0.23,
        "eval_row_count_long": 5700.0,
        "trajectory_abs_dz_dmd_high": 0.04,
        "gr_delta_abs_mean_high": 15.0,
    }
    raw_thresholds = config_get(config, "model.hard_router.thresholds", {})
    if isinstance(raw_thresholds, dict):
        for key in thresholds:
            if raw_thresholds.get(key) is not None:
                thresholds[key] = float(raw_thresholds[key])
    return thresholds


def hard_router_condition_flags(
    frame: DriftFeatureFrame,
    config: dict[str, Any],
) -> dict[str, bool]:
    thresholds = hard_router_thresholds(config)
    features = frame.well_features

    prefix_low_gr = compare_rule_value(
        float(features.get("prefix_gr_missing_rate", np.nan)),
        ">=",
        thresholds["prefix_gr_missing_rate"],
    )
    eval_low_gr = compare_rule_value(
        float(features.get("eval_gr_missing_rate", np.nan)),
        ">=",
        thresholds["eval_gr_missing_rate"],
    )
    short_prefix = compare_rule_value(
        float(features.get("prefix_fraction", np.nan)),
        "<=",
        thresholds["prefix_fraction_short"],
    )
    long_eval = compare_rule_value(
        float(features.get("eval_row_count", np.nan)),
        ">=",
        thresholds["eval_row_count_long"],
    )
    steep_trajectory = compare_rule_value(
        float(features.get("trajectory_abs_dz_dmd", np.nan)),
        ">=",
        thresholds["trajectory_abs_dz_dmd_high"],
    )
    large_gr_shift = compare_rule_value(
        float(features.get("gr_delta_abs_mean", np.nan)),
        ">=",
        thresholds["gr_delta_abs_mean_high"],
    )

    gr_weak_any = prefix_low_gr or eval_low_gr
    gr_weak_all = prefix_low_gr and eval_low_gr
    return {
        "prefix_low_gr": prefix_low_gr,
        "eval_low_gr": eval_low_gr,
        "gr_weak_any": gr_weak_any,
        "gr_weak_all": gr_weak_all,
        "gr_strong": not gr_weak_any,
        "short_prefix": short_prefix,
        "long_eval": long_eval,
        "steep_trajectory": steep_trajectory,
        "large_gr_shift": large_gr_shift,
        "short_prefix_low_gr": short_prefix and gr_weak_any,
        "large_gr_shift_low_gr": large_gr_shift and gr_weak_any,
        "long_eval_steep": long_eval and steep_trajectory,
        "low_gr_or_long_eval_steep": gr_weak_all
        or (long_eval and steep_trajectory)
        or (large_gr_shift and gr_weak_any),
    }


def hard_router_conditions_match(
    frame: DriftFeatureFrame,
    config: dict[str, Any],
    route_config: dict[str, Any],
) -> bool:
    raw_conditions = route_config.get("conditions", route_config.get("when", []))
    if isinstance(raw_conditions, str):
        condition_names = [raw_conditions]
    elif isinstance(raw_conditions, list):
        condition_names = [str(value) for value in raw_conditions]
    elif raw_conditions:
        raise ValueError("hard_router route conditions must be a string or list")
    else:
        condition_names = []

    flags = hard_router_condition_flags(frame, config)
    condition_matches: list[bool] = []
    for condition_name in condition_names:
        if condition_name not in flags:
            raise ValueError(f"unknown hard_router condition: {condition_name}")
        condition_matches.append(flags[condition_name])

    raw_rules = route_config.get("rules", [])
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError("hard_router route rules must be a list")

    for index, rule in enumerate(raw_rules):
        if not isinstance(rule, dict):
            raise ValueError(f"hard_router route rule #{index} must be a mapping")
        feature_name = str(rule.get("feature") or "")
        if feature_name not in frame.well_features:
            raise ValueError(f"unknown hard_router feature: {feature_name}")
        value = float(frame.well_features[feature_name])
        if bool(rule.get("abs", False)):
            value = abs(value)
        condition_matches.append(
            compare_rule_value(value, str(rule.get("op", ">=")), float(rule.get("value")))
        )

    if not condition_matches:
        return False

    combine = str(route_config.get("combine", route_config.get("condition_combine", "all")))
    if combine == "all":
        return all(condition_matches)
    if combine == "any":
        return any(condition_matches)
    raise ValueError(f"unsupported hard_router route combine: {combine}")


def hard_router_route(frame: DriftFeatureFrame, config: dict[str, Any]) -> str:
    if not hard_router_enabled(config):
        return "all_gr"

    routes = config_get(config, "model.hard_router.routes", [])
    if not isinstance(routes, list):
        raise ValueError("model.hard_router.routes must be a list")

    allowed_routes = {"all_gr", "no_gr", "guarded"}
    for index, route_config in enumerate(routes):
        if not isinstance(route_config, dict):
            raise ValueError(f"model.hard_router.routes[{index}] must be a mapping")
        route = str(route_config.get("route") or "")
        if route not in allowed_routes:
            raise ValueError(f"unsupported hard_router route: {route}")
        if hard_router_conditions_match(frame, config, route_config):
            return route

    default_route = str(config_get(config, "model.hard_router.default_route", "all_gr"))
    if default_route not in allowed_routes:
        raise ValueError(f"unsupported hard_router default route: {default_route}")
    return default_route


def hard_router_guarded_config(config: dict[str, Any]) -> dict[str, Any]:
    all_gr_feature_set, no_gr_feature_set = hard_router_feature_sets(config)
    guarded_config = config_with_feature_set(config, all_gr_feature_set)
    model_config = guarded_config.setdefault("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")

    guarded_gr_gating = config_get(config, "model.hard_router.guarded_gr_gating", None)
    if guarded_gr_gating is None:
        guarded_gr_gating = {
            "enabled": True,
            "base_feature_set": all_gr_feature_set,
            "alternate_feature_set": no_gr_feature_set,
            "combine": "all",
            "default_weight": 0.0,
            "alternate_weight": 1.0,
            "rules": [
                {"feature": "prefix_gr_missing_rate", "op": ">=", "value": 0.35},
                {"feature": "eval_gr_missing_rate", "op": ">=", "value": 0.40},
            ],
        }
    model_config["gr_gating"] = deepcopy(guarded_gr_gating)
    return guarded_config


def hard_router_guarded_weight(frame: DriftFeatureFrame, config: dict[str, Any]) -> float:
    return gr_gate_weight(frame, hard_router_guarded_config(config))


def fit_gr_gated_models_from_files(
    files: Iterable[Path],
    config: dict[str, Any],
    *,
    seed: int,
    max_rows_total: int | None,
    max_rows_per_well: int | None,
) -> GrGatedModelBundle:
    base_feature_set, alternate_feature_set = gr_gating_feature_sets(config)
    base_config = config_with_feature_set(config, base_feature_set)
    alternate_config = config_with_feature_set(config, alternate_feature_set)
    files = list(files)
    base_model, n_train_rows_base = fit_drift_model_from_files(
        files,
        base_config,
        seed=seed,
        max_rows_total=max_rows_total,
        max_rows_per_well=max_rows_per_well,
    )
    alternate_model, n_train_rows_alternate = fit_drift_model_from_files(
        files,
        alternate_config,
        seed=seed,
        max_rows_total=max_rows_total,
        max_rows_per_well=max_rows_per_well,
    )
    return GrGatedModelBundle(
        base_model=base_model,
        alternate_model=alternate_model,
        base_config=base_config,
        alternate_config=alternate_config,
        n_train_rows_base=n_train_rows_base,
        n_train_rows_alternate=n_train_rows_alternate,
    )


def fit_hard_router_models_from_files(
    files: Iterable[Path],
    config: dict[str, Any],
    *,
    seed: int,
    max_rows_total: int | None,
    max_rows_per_well: int | None,
) -> HardRouterModelBundle:
    all_gr_feature_set, no_gr_feature_set = hard_router_feature_sets(config)
    all_gr_config = config_with_feature_set(config, all_gr_feature_set)
    no_gr_config = config_with_feature_set(config, no_gr_feature_set)
    guarded_config = hard_router_guarded_config(config)
    files = list(files)
    all_gr_model, n_train_rows_all_gr = fit_drift_model_from_files(
        files,
        all_gr_config,
        seed=seed,
        max_rows_total=max_rows_total,
        max_rows_per_well=max_rows_per_well,
    )
    no_gr_model, n_train_rows_no_gr = fit_drift_model_from_files(
        files,
        no_gr_config,
        seed=seed,
        max_rows_total=max_rows_total,
        max_rows_per_well=max_rows_per_well,
    )
    return HardRouterModelBundle(
        all_gr_model=all_gr_model,
        no_gr_model=no_gr_model,
        all_gr_config=all_gr_config,
        no_gr_config=no_gr_config,
        guarded_config=guarded_config,
        n_train_rows_all_gr=n_train_rows_all_gr,
        n_train_rows_no_gr=n_train_rows_no_gr,
    )


def fit_variant_model_from_files(
    files: Iterable[Path],
    config: dict[str, Any],
    *,
    seed: int,
    max_rows_total: int | None,
    max_rows_per_well: int | None,
) -> tuple[Any, int]:
    if hard_router_enabled(config):
        bundle = fit_hard_router_models_from_files(
            files,
            config,
            seed=seed,
            max_rows_total=max_rows_total,
            max_rows_per_well=max_rows_per_well,
        )
        return bundle, int(bundle.n_train_rows_all_gr + bundle.n_train_rows_no_gr)
    if gr_gating_enabled(config):
        bundle = fit_gr_gated_models_from_files(
            files,
            config,
            seed=seed,
            max_rows_total=max_rows_total,
            max_rows_per_well=max_rows_per_well,
        )
        return bundle, int(bundle.n_train_rows_base + bundle.n_train_rows_alternate)
    return fit_drift_model_from_files(
        files,
        config,
        seed=seed,
        max_rows_total=max_rows_total,
        max_rows_per_well=max_rows_per_well,
    )


def predict_gr_gated(
    frame: DriftFeatureFrame,
    model_bundle: GrGatedModelBundle,
    config: dict[str, Any],
) -> np.ndarray:
    base_prediction = predict_drift(frame, model_bundle.base_model, model_bundle.base_config)
    weight = gr_gate_weight(frame, config)
    if weight <= 0.0:
        return base_prediction
    alternate_prediction = predict_drift(
        frame,
        model_bundle.alternate_model,
        model_bundle.alternate_config,
    )
    return base_prediction + weight * (alternate_prediction - base_prediction)


def predict_hard_router(
    frame: DriftFeatureFrame,
    model_bundle: HardRouterModelBundle,
    config: dict[str, Any],
) -> np.ndarray:
    all_gr_prediction = predict_drift(
        frame,
        model_bundle.all_gr_model,
        model_bundle.all_gr_config,
    )
    route = hard_router_route(frame, config)
    if route == "all_gr":
        return all_gr_prediction

    no_gr_prediction = predict_drift(
        frame,
        model_bundle.no_gr_model,
        model_bundle.no_gr_config,
    )
    if route == "no_gr":
        return no_gr_prediction
    if route == "guarded":
        weight = gr_gate_weight(frame, model_bundle.guarded_config)
        return all_gr_prediction + weight * (no_gr_prediction - all_gr_prediction)
    raise ValueError(f"unsupported hard_router route: {route}")


def predict_variant_drift(
    frame: DriftFeatureFrame,
    model: Any,
    config: dict[str, Any],
) -> np.ndarray:
    if isinstance(model, HardRouterModelBundle):
        return predict_hard_router(frame, model, config)
    if isinstance(model, GrGatedModelBundle):
        return predict_gr_gated(frame, model, config)
    return predict_drift(frame, model, config)
