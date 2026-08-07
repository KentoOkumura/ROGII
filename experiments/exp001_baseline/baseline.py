from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HORIZONTAL_SUFFIX = "__horizontal_well.csv"


@dataclass(frozen=True)
class PrefixPrediction:
    eval_indices: np.ndarray
    predictions: dict[str, np.ndarray]
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


def estimate_recent_slope(
    md: np.ndarray,
    tvt_input: np.ndarray,
    last_known_index: int,
    config: dict[str, Any],
) -> float:
    window = int(config_get(config, "model.params.recent_slope_window", 200))
    max_abs_slope = float(config_get(config, "model.params.max_abs_recent_slope", 0.08))
    shrink = float(config_get(config, "model.params.recent_slope_shrink", 0.5))

    start = max(0, last_known_index - window + 1)
    md_window = md[start : last_known_index + 1]
    tvt_window = tvt_input[start : last_known_index + 1]
    finite = np.isfinite(md_window) & np.isfinite(tvt_window)
    md_window = md_window[finite]
    tvt_window = tvt_window[finite]
    if md_window.size < 2:
        return 0.0

    delta_md = np.diff(md_window)
    delta_tvt = np.diff(tvt_window)
    valid = np.isfinite(delta_md) & np.isfinite(delta_tvt) & (delta_md != 0.0)
    if not valid.any():
        return 0.0

    slope = float(np.median(delta_tvt[valid] / delta_md[valid]))
    if not np.isfinite(slope):
        return 0.0
    if max_abs_slope > 0:
        slope = float(np.clip(slope, -max_abs_slope, max_abs_slope))
    return slope * shrink


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

    predictions: dict[str, np.ndarray] = {}
    for strategy in strategy_names(config):
        if strategy == "last_anchor":
            predictions[strategy] = np.full(eval_indices.size, last_known_tvt, dtype=float)
        elif strategy == "recent_linear":
            predictions[strategy] = last_known_tvt + recent_slope * (eval_md - last_known_md)
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
