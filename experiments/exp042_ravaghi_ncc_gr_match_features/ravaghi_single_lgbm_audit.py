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
    distance_bucket_alphas,
    postprocess_predictions,
    predict_drift,
    well_id_from_path,
)
from pseudo_tail_augmentation import fit_pseudo_tail_model_from_files, train_files
from settings import ExperimentPaths

BASE_USECOLS = {
    "id",
    "well_id",
    "fold",
    "pseudo_cutoff_fraction",
    "cutoff_row",
    "row_idx",
    "prefix_length",
    "eval_step",
    "eval_fraction",
    "target_tvt",
    "last_anchor_tvt",
    "pf_pred",
    "beam_pred",
}
DERIVED_FEATURE_DEPENDENCIES = {
    "pf_ancc_delta": {"pf_pred", "last_anchor_tvt"},
    "pf_ancc_abs_delta": {"pf_pred", "last_anchor_tvt"},
    "pf_ancc_surface_delta": {"pf_pred", "last_anchor_tvt", "Z", "well_id", "cutoff_row"},
    "pf_ancc_delta_per_eval_step": {"pf_pred", "last_anchor_tvt", "eval_step"},
    "pf_vs_last_anchor": {"pf_pred", "last_anchor_tvt"},
    "abs_pf_vs_last_anchor": {"pf_pred", "last_anchor_tvt"},
    "z_delta_from_cutoff": {"Z", "well_id", "cutoff_row"},
    "md_delta_from_cutoff": {"MD", "well_id", "cutoff_row"},
    "pf_z_delta_proxy": {"Z", "well_id", "cutoff_row"},
    "pf_z_proxy_delta_from_last_anchor": {"Z", "well_id", "cutoff_row"},
    "pf_vs_z_proxy": {"pf_pred", "last_anchor_tvt", "Z", "well_id", "cutoff_row"},
    "abs_pf_vs_z_proxy": {"pf_pred", "last_anchor_tvt", "Z", "well_id", "cutoff_row"},
    "pf_ancc_std": {"pf_seed_std"},
    "pf_scale_mean_delta": {
        "pf_scale_3",
        "pf_scale_5",
        "pf_scale_8",
        "pf_scale_12",
        "last_anchor_tvt",
    },
    "pf_scale_std": {"pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"},
    "pf_scale_range": {"pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"},
    "pf_selected_vs_scale_mean": {
        "pf_selected_scale_pred",
        "pf_scale_3",
        "pf_scale_5",
        "pf_scale_8",
        "pf_scale_12",
    },
    "pf_scale_range_to_seed_std": {
        "pf_scale_3",
        "pf_scale_5",
        "pf_scale_8",
        "pf_scale_12",
        "pf_seed_std",
    },
    "pf_likelihood_margin": {"pf_lik_gap_best_second"},
    "beam_exact_cons_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_loose_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_vcons_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_sm5_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_vloose_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_mid_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_stiff_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_mean_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_median_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_ref_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_std": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_range": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_min_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_max_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "beam_exact_cons_sm5_gap": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_cost_mean": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_cost_min": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_cost_std": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_cost_gap_best_second": {"well_id", "cutoff_row", "row_idx"},
    "beam_exact_mean_vs_public_beam": {
        "well_id",
        "cutoff_row",
        "row_idx",
        "beam_pred",
    },
    "abs_beam_exact_mean_vs_public_beam": {
        "well_id",
        "cutoff_row",
        "row_idx",
        "beam_pred",
    },
    "beam_exact_ref_vs_public_beam": {
        "well_id",
        "cutoff_row",
        "row_idx",
        "beam_pred",
    },
    "abs_beam_exact_ref_vs_public_beam": {
        "well_id",
        "cutoff_row",
        "row_idx",
        "beam_pred",
    },
    "beam_exact_mean_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "abs_beam_exact_mean_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "beam_exact_ref_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "abs_beam_exact_ref_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "ncc_sc8_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "ncc_sc15_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "ncc_sc25_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "ncc_sc_cons_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "ncc_sc_ens_delta": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "ncc_sc8_score": {"well_id", "cutoff_row", "row_idx"},
    "ncc_sc15_score": {"well_id", "cutoff_row", "row_idx"},
    "ncc_sc25_score": {"well_id", "cutoff_row", "row_idx"},
    "ncc_score_mean": {"well_id", "cutoff_row", "row_idx"},
    "ncc_score_std": {"well_id", "cutoff_row", "row_idx"},
    "ncc_score_min": {"well_id", "cutoff_row", "row_idx"},
    "ncc_score_max": {"well_id", "cutoff_row", "row_idx"},
    "ncc_score_gap_best_second": {"well_id", "cutoff_row", "row_idx"},
    "ncc_best_scale": {"well_id", "cutoff_row", "row_idx"},
    "ncc_sc_trust": {"well_id", "cutoff_row", "row_idx"},
    "ncc_sc_range": {"well_id", "cutoff_row", "row_idx"},
    "ncc_sc_std": {"well_id", "cutoff_row", "row_idx"},
    "ncc_sc_ens_vs_public_beam": {"well_id", "cutoff_row", "row_idx", "beam_pred"},
    "abs_ncc_sc_ens_vs_public_beam": {
        "well_id",
        "cutoff_row",
        "row_idx",
        "beam_pred",
    },
    "ncc_sc_ens_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "abs_ncc_sc_ens_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "ncc_sc_cons_vs_public_beam": {"well_id", "cutoff_row", "row_idx", "beam_pred"},
    "abs_ncc_sc_cons_vs_public_beam": {
        "well_id",
        "cutoff_row",
        "row_idx",
        "beam_pred",
    },
    "ncc_sc_cons_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "abs_ncc_sc_cons_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "gr_match_prefix_rmse": {"well_id", "cutoff_row", "row_idx"},
    "gr_match_cal_a": {"well_id", "cutoff_row", "row_idx"},
    "gr_match_cal_b": {"well_id", "cutoff_row", "row_idx"},
    "gr_vs_tw_anchor": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "gr_vs_sc_ens": {"well_id", "cutoff_row", "row_idx"},
    "gr_vs_public_beam": {"well_id", "cutoff_row", "row_idx", "beam_pred"},
    "gr_vs_pf": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "gr_vs_anchor_m25": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "gr_vs_anchor_p25": {"well_id", "cutoff_row", "row_idx", "last_anchor_tvt"},
    "gr_vs_sc_m25": {"well_id", "cutoff_row", "row_idx"},
    "gr_vs_sc_p25": {"well_id", "cutoff_row", "row_idx"},
    "gr_vs_beam_m25": {"well_id", "cutoff_row", "row_idx", "beam_pred"},
    "gr_vs_beam_p25": {"well_id", "cutoff_row", "row_idx", "beam_pred"},
    "gr_vs_pf_m25": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
    "gr_vs_pf_p25": {"well_id", "cutoff_row", "row_idx", "pf_pred"},
}
DERIVED_FEATURE_COLUMNS = set(DERIVED_FEATURE_DEPENDENCIES)
BEAM_EXACT_CONFIGS = (
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
)
BEAM_EXACT_TAGS = tuple(config[-1] for config in BEAM_EXACT_CONFIGS)
BEAM_EXACT_FEATURE_COLUMNS = {
    column
    for column in DERIVED_FEATURE_COLUMNS
    if column.startswith(("beam_exact", "abs_beam_exact"))
}
NCC_GR_MATCH_FEATURE_COLUMNS = {
    column
    for column in DERIVED_FEATURE_COLUMNS
    if column.startswith(("ncc_", "abs_ncc_", "gr_match_", "gr_vs_"))
}
LOCAL_GENERATED_FEATURE_COLUMNS = BEAM_EXACT_FEATURE_COLUMNS | NCC_GR_MATCH_FEATURE_COLUMNS
SIMPLE_DERIVED_FEATURE_COLUMNS = DERIVED_FEATURE_COLUMNS - LOCAL_GENERATED_FEATURE_COLUMNS
EXCLUDED_FEATURE_COLUMNS = {
    "pf_error",
    "last_anchor_error",
    "beam_error",
    "exp026_oof",
    "pf_pred_minus_exp026_oof",
    "abs_pf_pred_minus_exp026_oof",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    feature_columns: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Ravaghi NCC/GR match single LGBM features.")
    parser.add_argument("--features", default=None, help="Path to exp029 feature CSV")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional smoke well limit")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Override per-split candidate model row cap",
    )
    parser.add_argument(
        "--base-estimator",
        choices=["LGBMRegressor", "HistGradientBoostingRegressor"],
        default=None,
        help="Override candidate estimator for local smoke checks.",
    )
    parser.add_argument(
        "--skip-exp026-control",
        action="store_true",
        help="Skip regenerated exp026 control and run only single-LGBM feature variants.",
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


def feature_variants(config: dict[str, Any]) -> list[VariantSpec]:
    families = get_nested(config, "model.feature_sets", {})
    if not isinstance(families, dict):
        raise ValueError("model.feature_sets must be a mapping")

    specs: list[VariantSpec] = []
    for variant in get_nested(config, "model.variants", []):
        name = str(variant["name"])
        columns: list[str] = []
        for family_name in variant.get("feature_families", []):
            raw_columns = families.get(str(family_name))
            if not isinstance(raw_columns, list) or not raw_columns:
                raise ValueError(f"unknown or empty feature family: {family_name}")
            for column in raw_columns:
                column = str(column)
                if column in EXCLUDED_FEATURE_COLUMNS:
                    raise ValueError(f"excluded column configured as a feature: {column}")
                if column not in columns:
                    columns.append(column)
        if not columns:
            raise ValueError(f"variant {name} has no feature columns")
        specs.append(VariantSpec(name=name, feature_columns=tuple(columns)))
    if not specs:
        raise ValueError("model.variants must define at least one variant")
    return specs


def configured_feature_columns(config: dict[str, Any]) -> set[str]:
    columns: set[str] = set()
    for spec in feature_variants(config):
        columns.update(spec.feature_columns)
    return columns


def required_columns(config: dict[str, Any]) -> list[str]:
    columns = set(BASE_USECOLS)
    for spec in feature_variants(config):
        for column in spec.feature_columns:
            if column in DERIVED_FEATURE_DEPENDENCIES:
                columns.update(DERIVED_FEATURE_DEPENDENCIES[column])
            else:
                columns.add(column)
    return sorted(columns - EXCLUDED_FEATURE_COLUMNS)


def file_by_well(paths: ExperimentPaths, max_wells: int | None) -> dict[str, Path]:
    files = train_files(paths, max_wells)
    return {well_id_from_path(path): path for path in files}


def load_features(
    path: Path,
    config: dict[str, Any],
    *,
    allowed_wells: set[str] | None,
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
    frame = add_derived_features(frame)
    configured = configured_feature_columns(config)
    missing_features = sorted(
        configured - set(frame.columns) - LOCAL_GENERATED_FEATURE_COLUMNS
    )
    if missing_features:
        raise ValueError(f"Feature frame is missing configured features: {missing_features}")
    frame = frame.sort_values(["fold", "well_id", "eval_step"], kind="mergesort").reset_index(
        drop=True
    )
    if not np.isfinite(frame["target_tvt"].to_numpy(dtype=float)).all():
        raise ValueError("target_tvt contains non-finite values")
    return frame, usecols


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    group_keys = ["well_id", "cutoff_row"]
    first_z = frame.groupby(group_keys, sort=False)["Z"].transform("first").astype(float)
    first_md = frame.groupby(group_keys, sort=False)["MD"].transform("first").astype(float)

    pf_pred = frame["pf_pred"].to_numpy(dtype=float)
    last_anchor = frame["last_anchor_tvt"].to_numpy(dtype=float)
    z_values = frame["Z"].to_numpy(dtype=float)
    md_values = frame["MD"].to_numpy(dtype=float)
    z_delta = z_values - first_z.to_numpy(dtype=float)
    md_delta = md_values - first_md.to_numpy(dtype=float)
    pf_delta = pf_pred - last_anchor
    pf_z_delta = -z_delta
    pf_z_proxy = last_anchor + pf_z_delta

    scale_columns = ["pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"]
    has_scale_columns = set(scale_columns).issubset(frame.columns)
    if has_scale_columns:
        scale_values = frame[scale_columns].to_numpy(dtype=float)
        scale_mean = np.nanmean(scale_values, axis=1)
        scale_std = np.nanstd(scale_values, axis=1)
        scale_range = np.nanmax(scale_values, axis=1) - np.nanmin(scale_values, axis=1)
    else:
        scale_mean = np.zeros(len(frame), dtype=float)
        scale_std = np.zeros(len(frame), dtype=float)
        scale_range = np.zeros(len(frame), dtype=float)
    seed_std = (
        frame["pf_seed_std"].to_numpy(dtype=float)
        if "pf_seed_std" in frame.columns
        else np.zeros(len(frame), dtype=float)
    )
    eval_step = frame["eval_step"].to_numpy(dtype=float)

    frame["pf_ancc_delta"] = pf_delta
    frame["pf_ancc_abs_delta"] = np.abs(pf_delta)
    frame["pf_ancc_surface_delta"] = (pf_pred + z_values) - (
        last_anchor + first_z.to_numpy(dtype=float)
    )
    frame["pf_ancc_delta_per_eval_step"] = pf_delta / np.maximum(eval_step + 1.0, 1.0)
    frame["pf_vs_last_anchor"] = pf_delta
    frame["abs_pf_vs_last_anchor"] = np.abs(pf_delta)
    frame["z_delta_from_cutoff"] = z_delta
    frame["md_delta_from_cutoff"] = md_delta
    frame["pf_z_delta_proxy"] = pf_z_delta
    frame["pf_z_proxy_delta_from_last_anchor"] = pf_z_delta
    frame["pf_vs_z_proxy"] = pf_pred - pf_z_proxy
    frame["abs_pf_vs_z_proxy"] = np.abs(frame["pf_vs_z_proxy"].to_numpy(dtype=float))
    frame["pf_ancc_std"] = seed_std
    frame["pf_scale_mean_delta"] = scale_mean - last_anchor
    frame["pf_scale_std"] = scale_std
    frame["pf_scale_range"] = scale_range
    if "pf_selected_scale_pred" in frame.columns and has_scale_columns:
        frame["pf_selected_vs_scale_mean"] = (
            frame["pf_selected_scale_pred"].to_numpy(dtype=float) - scale_mean
        )
    else:
        frame["pf_selected_vs_scale_mean"] = 0.0
    frame["pf_scale_range_to_seed_std"] = scale_range / np.maximum(seed_std, 1e-6)
    if "pf_lik_gap_best_second" in frame.columns:
        frame["pf_likelihood_margin"] = frame["pf_lik_gap_best_second"].to_numpy(dtype=float)
    else:
        frame["pf_likelihood_margin"] = 0.0

    for column in SIMPLE_DERIVED_FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            frame[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return frame


def smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64").interpolate(limit_direction="both")
    series = series.fillna(float(fallback))
    if radius <= 0:
        return series.to_numpy(dtype=float)
    return (
        series.rolling(2 * int(radius) + 1, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(values[idx - 1] - target) <= abs(values[idx] - target):
        return idx - 1
    return idx


def beam_search_exact(
    horizontal_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    start_tvt: float,
    beam_size: int,
    move_cost: float,
    error_scale: float,
    smooth_radius: int,
) -> tuple[np.ndarray, float]:
    n_rows = len(horizontal_gr)
    if n_rows == 0:
        return np.asarray([], dtype=float), 0.0
    if len(typewell_tvt) == 0:
        return np.full(n_rows, start_tvt, dtype=float), 0.0

    smoothed_gr = smooth_gr(horizontal_gr, float(np.nanmean(typewell_gr)), smooth_radius)
    moves = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    move_penalty = move_cost * np.abs(moves).astype(float)
    beam_indices = np.full(int(beam_size), nearest_index(typewell_tvt, start_tvt), dtype=np.int64)
    beam_cost = np.full(int(beam_size), np.inf, dtype=float)
    beam_cost[0] = 0.0
    active = 1
    output = np.empty(n_rows, dtype=float)

    for step, gr_value in enumerate(smoothed_gr):
        next_idx = beam_indices[:active, None] + moves[None, :]
        valid = (next_idx >= 0) & (next_idx < len(typewell_tvt))
        clipped = np.clip(next_idx, 0, len(typewell_tvt) - 1)
        gr_error = np.square(gr_value - typewell_gr[clipped]) / float(error_scale)
        total = beam_cost[:active, None] + gr_error + move_penalty[None, :]
        total = np.where(valid, total, np.inf)

        flat_idx = next_idx.ravel()
        flat_cost = total.ravel()
        keep = np.isfinite(flat_cost)
        flat_idx = flat_idx[keep]
        flat_cost = flat_cost[keep]
        if len(flat_idx) == 0:
            output[step:] = typewell_tvt[beam_indices[0]]
            break

        best_by_idx: dict[int, float] = {}
        for idx, cost in zip(flat_idx, flat_cost, strict=False):
            idx = int(idx)
            cost = float(cost)
            previous = best_by_idx.get(idx)
            if previous is None or cost < previous:
                best_by_idx[idx] = cost
        ordered = sorted(best_by_idx.items(), key=lambda item: item[1])[: int(beam_size)]
        active = len(ordered)
        beam_indices[:active] = [idx for idx, _ in ordered]
        beam_cost[:active] = [cost for _, cost in ordered]
        if active < int(beam_size):
            beam_indices[active:] = beam_indices[active - 1]
            beam_cost[active:] = np.inf
        output[step] = typewell_tvt[beam_indices[0]]
    return output, float(np.nanmin(beam_cost[:active]))


def typewell_path_from_horizontal(path: Path, well_id: str) -> Path:
    return path.with_name(f"{well_id}__typewell.csv")


def compute_beam_exact_group(
    *,
    horizontal_path: Path,
    cutoff_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    well_id = well_id_from_path(horizontal_path)
    typewell_path = typewell_path_from_horizontal(horizontal_path, well_id)
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    if cutoff_row <= 0 or cutoff_row >= len(horizontal):
        raise ValueError(f"invalid cutoff_row={cutoff_row} for {horizontal_path.name}")

    tvt_input = horizontal["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input[int(cutoff_row) :] = np.nan
    horizontal = horizontal.copy()
    horizontal["TVT_input"] = tvt_input
    known = horizontal.iloc[: int(cutoff_row)][
        horizontal.iloc[: int(cutoff_row)]["TVT_input"].notna()
    ]
    eval_indices = np.arange(int(cutoff_row), len(horizontal), dtype=int)
    if known.empty or len(eval_indices) == 0:
        empty = np.empty((0, len(BEAM_EXACT_TAGS)), dtype=float)
        return eval_indices, empty, np.full(len(BEAM_EXACT_TAGS), np.nan, dtype=float)

    sorted_typewell = typewell.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if sorted_typewell.empty:
        empty = np.full(
            (len(eval_indices), len(BEAM_EXACT_TAGS)),
            float(known["TVT_input"].iloc[-1]),
        )
        return eval_indices, empty, np.full(len(BEAM_EXACT_TAGS), np.nan, dtype=float)

    typewell_tvt = sorted_typewell["TVT"].to_numpy(dtype=float)
    typewell_gr = sorted_typewell["GR"].to_numpy(dtype=float)
    horizontal_gr = (
        horizontal["GR"]
        .astype(float)
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(dtype=float)[eval_indices]
    )
    start_tvt = float(known["TVT_input"].iloc[-1])
    predictions: list[np.ndarray] = []
    costs: list[float] = []
    for beam_size, move_cost, error_scale, smooth_radius, _tag in BEAM_EXACT_CONFIGS:
        pred, cost = beam_search_exact(
            horizontal_gr,
            typewell_tvt,
            typewell_gr,
            start_tvt=start_tvt,
            beam_size=beam_size,
            move_cost=move_cost,
            error_scale=error_scale,
            smooth_radius=smooth_radius,
        )
        predictions.append(pred)
        costs.append(cost)
    return eval_indices, np.column_stack(predictions), np.asarray(costs, dtype=float)


def add_beam_exact_features(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    max_wells: int | None,
) -> pd.DataFrame:
    if not (configured_feature_columns(config) & BEAM_EXACT_FEATURE_COLUMNS):
        return frame

    frame = frame.copy()
    path_by_well = file_by_well(paths, max_wells)
    for column in BEAM_EXACT_FEATURE_COLUMNS:
        frame[column] = 0.0

    for (well_id, cutoff_row), part in frame.groupby(["well_id", "cutoff_row"], sort=False):
        horizontal_path = path_by_well.get(str(well_id))
        if horizontal_path is None:
            raise ValueError(f"missing train file for beam exact features: {well_id}")
        eval_indices, beam_matrix, costs = compute_beam_exact_group(
            horizontal_path=horizontal_path,
            cutoff_row=int(cutoff_row),
        )
        by_row = pd.DataFrame(beam_matrix, index=eval_indices, columns=BEAM_EXACT_TAGS)
        aligned = by_row.reindex(part["row_idx"].to_numpy(dtype=int))
        if aligned.isna().any().any():
            raise ValueError(f"missing exact beam rows for {well_id} cutoff={cutoff_row}")

        positions = part.index.to_numpy(dtype=int)
        last_anchor = part["last_anchor_tvt"].to_numpy(dtype=float)
        public_beam = part["beam_pred"].to_numpy(dtype=float)
        public_pf = part["pf_pred"].to_numpy(dtype=float)
        values = aligned.to_numpy(dtype=float)
        values_by_tag = dict(zip(BEAM_EXACT_TAGS, values.T, strict=True))
        for tag in BEAM_EXACT_TAGS:
            frame.loc[positions, f"beam_exact_{tag}_delta"] = values_by_tag[tag] - last_anchor

        mean_pred = values.mean(axis=1)
        median_pred = np.median(values, axis=1)
        ref_pred = 0.5 * (values_by_tag["cons"] + values_by_tag["sm5"])
        min_pred = values.min(axis=1)
        max_pred = values.max(axis=1)
        cost_values = np.nan_to_num(costs, nan=0.0, posinf=0.0, neginf=0.0)
        sorted_costs = np.sort(cost_values)
        cost_gap = (
            sorted_costs[1] - sorted_costs[0] if len(sorted_costs) >= 2 else 0.0
        )

        frame.loc[positions, "beam_exact_mean_delta"] = mean_pred - last_anchor
        frame.loc[positions, "beam_exact_median_delta"] = median_pred - last_anchor
        frame.loc[positions, "beam_exact_ref_delta"] = ref_pred - last_anchor
        frame.loc[positions, "beam_exact_std"] = values.std(axis=1)
        frame.loc[positions, "beam_exact_range"] = max_pred - min_pred
        frame.loc[positions, "beam_exact_min_delta"] = min_pred - last_anchor
        frame.loc[positions, "beam_exact_max_delta"] = max_pred - last_anchor
        frame.loc[positions, "beam_exact_cons_sm5_gap"] = np.abs(
            values_by_tag["cons"] - values_by_tag["sm5"]
        )
        frame.loc[positions, "beam_exact_cost_mean"] = float(cost_values.mean())
        frame.loc[positions, "beam_exact_cost_min"] = float(cost_values.min())
        frame.loc[positions, "beam_exact_cost_std"] = float(cost_values.std())
        frame.loc[positions, "beam_exact_cost_gap_best_second"] = float(cost_gap)
        frame.loc[positions, "beam_exact_mean_vs_public_beam"] = mean_pred - public_beam
        frame.loc[positions, "abs_beam_exact_mean_vs_public_beam"] = np.abs(
            mean_pred - public_beam
        )
        frame.loc[positions, "beam_exact_ref_vs_public_beam"] = ref_pred - public_beam
        frame.loc[positions, "abs_beam_exact_ref_vs_public_beam"] = np.abs(
            ref_pred - public_beam
        )
        frame.loc[positions, "beam_exact_mean_vs_pf"] = mean_pred - public_pf
        frame.loc[positions, "abs_beam_exact_mean_vs_pf"] = np.abs(mean_pred - public_pf)
        frame.loc[positions, "beam_exact_ref_vs_pf"] = ref_pred - public_pf
        frame.loc[positions, "abs_beam_exact_ref_vs_pf"] = np.abs(ref_pred - public_pf)
        print(
            json.dumps(
                {
                    "beam_exact": "generated",
                    "well_id": str(well_id),
                    "cutoff_row": int(cutoff_row),
                    "rows": int(len(part)),
                    "cost_min": round(float(cost_values.min()), 6),
                },
                sort_keys=True,
            )
        )

    missing_features = sorted(configured_feature_columns(config) - set(frame.columns))
    if missing_features:
        raise ValueError(f"Feature frame is missing configured features: {missing_features}")
    for column in BEAM_EXACT_FEATURE_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            frame[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return frame


def affine_calibration(
    prefix_gr: np.ndarray,
    prefix_typewell_gr: np.ndarray,
    *,
    min_points: int = 20,
) -> tuple[float, float]:
    valid = np.isfinite(prefix_gr) & np.isfinite(prefix_typewell_gr)
    if valid.sum() < min_points or np.nanstd(prefix_typewell_gr[valid]) < 1e-6:
        if valid.any():
            return 1.0, float(np.nanmean(prefix_gr[valid]) - np.nanmean(prefix_typewell_gr[valid]))
        return 1.0, 0.0
    slope, intercept = np.polyfit(prefix_typewell_gr[valid], prefix_gr[valid], 1)
    return float(slope), float(intercept)


def multi_scale_ncc(
    prefix_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    eval_gr: np.ndarray,
    *,
    half_windows: tuple[int, ...] = (8, 15, 25),
    stride: int = 3,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    n_eval = len(eval_gr)
    for half_window in half_windows:
        window = 2 * int(half_window) + 1
        n_prefix = len(prefix_gr)
        if n_prefix < window + 1 or n_eval == 0:
            outputs.append(
                (
                    np.full(n_eval, float(prefix_tvt[-1]), dtype=np.float32),
                    np.zeros(n_eval, dtype=np.float32),
                )
            )
            continue

        known_gr = (
            pd.Series(prefix_gr, dtype="float32")
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float32)
        )
        hidden_gr = (
            pd.Series(eval_gr, dtype="float32")
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float32)
        )
        starts = np.arange(0, n_prefix - window + 1, int(stride), dtype=np.int32)
        if len(starts) == 0:
            outputs.append(
                (
                    np.full(n_eval, float(prefix_tvt[-1]), dtype=np.float32),
                    np.zeros(n_eval, dtype=np.float32),
                )
            )
            continue

        window_offsets = np.arange(window, dtype=np.int32)
        candidates = known_gr[starts[:, None] + window_offsets[None, :]].astype(np.float32)
        candidates_norm = (candidates - candidates.mean(axis=1, keepdims=True)) / (
            candidates.std(axis=1, keepdims=True) + 1e-6
        )
        padded_hidden = np.pad(hidden_gr, int(half_window), mode="edge")
        hidden = padded_hidden[
            np.arange(n_eval)[:, None] + window_offsets[None, :]
        ].astype(np.float32)
        hidden_norm = (hidden - hidden.mean(axis=1, keepdims=True)) / (
            hidden.std(axis=1, keepdims=True) + 1e-6
        )
        ncc = hidden_norm @ candidates_norm.T / float(window)
        best = ncc.argmax(axis=1)
        score = ncc.max(axis=1).astype(np.float32)
        matched_tvt = prefix_tvt[
            np.clip(starts[best] + int(half_window), 0, n_prefix - 1)
        ].astype(np.float32)
        outputs.append((matched_tvt, score))

    tvt_matrix = np.stack([output[0] for output in outputs], axis=1)
    score_matrix = np.stack([output[1] for output in outputs], axis=1)
    score_weights = np.exp(3.0 * score_matrix)
    score_weights /= score_weights.sum(axis=1, keepdims=True) + 1e-9
    ensemble = (tvt_matrix * score_weights).sum(axis=1).astype(np.float32)
    return outputs, ensemble


def compute_ncc_gr_match_group(
    *,
    horizontal_path: Path,
    cutoff_row: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    well_id = well_id_from_path(horizontal_path)
    typewell_path = typewell_path_from_horizontal(horizontal_path, well_id)
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    if cutoff_row <= 0 or cutoff_row >= len(horizontal):
        raise ValueError(f"invalid cutoff_row={cutoff_row} for {horizontal_path.name}")

    tvt_input = horizontal["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input[int(cutoff_row) :] = np.nan
    horizontal = horizontal.copy()
    horizontal["TVT_input"] = tvt_input
    known = horizontal[horizontal["TVT_input"].notna()]
    eval_mask = horizontal["TVT_input"].isna().to_numpy()
    eval_indices = np.flatnonzero(eval_mask).astype(int)
    if known.empty or len(eval_indices) == 0:
        return eval_indices, {}, {"available": 0.0, "prefix_rmse": 0.0, "cal_a": 1.0, "cal_b": 0.0}

    sorted_typewell = typewell.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if sorted_typewell.empty:
        empty = np.zeros(len(eval_indices), dtype=np.float32)
        return (
            eval_indices,
            {
                "sc8": empty.copy(),
                "sc15": empty.copy(),
                "sc25": empty.copy(),
                "sc8_score": empty.copy(),
                "sc15_score": empty.copy(),
                "sc25_score": empty.copy(),
                "sc_ens": empty.copy(),
                "eval_gr": empty.copy(),
            },
            {"available": 0.0, "prefix_rmse": 0.0, "cal_a": 1.0, "cal_b": 0.0},
        )

    typewell_tvt = sorted_typewell["TVT"].to_numpy(dtype=np.float32)
    typewell_gr = sorted_typewell["GR"].to_numpy(dtype=np.float32)
    fallback_gr = float(np.nanmean(typewell_gr))
    full_gr = (
        horizontal["GR"]
        .astype(float)
        .interpolate(limit_direction="both")
        .fillna(fallback_gr)
        .to_numpy(dtype=np.float32)
    )
    prefix_gr = full_gr[: len(known)]
    prefix_tvt = known["TVT_input"].to_numpy(dtype=np.float32)
    eval_gr = full_gr[eval_indices]

    ncc_outputs, sc_ens = multi_scale_ncc(prefix_gr, prefix_tvt, eval_gr)
    (sc8, sc8_score), (sc15, sc15_score), (sc25, sc25_score) = ncc_outputs
    prefix_typewell_gr = np.interp(prefix_tvt, typewell_tvt, typewell_gr).astype(np.float32)
    prefix_rmse = float(np.sqrt(np.mean(np.square(prefix_gr - prefix_typewell_gr))))
    cal_a, cal_b = affine_calibration(prefix_gr, prefix_typewell_gr)
    return (
        eval_indices,
        {
            "sc8": sc8,
            "sc15": sc15,
            "sc25": sc25,
            "sc8_score": sc8_score,
            "sc15_score": sc15_score,
            "sc25_score": sc25_score,
            "sc_ens": sc_ens,
            "eval_gr": eval_gr,
            "typewell_tvt": typewell_tvt,
            "typewell_gr": typewell_gr,
        },
        {
            "available": 1.0,
            "prefix_rmse": prefix_rmse,
            "cal_a": cal_a,
            "cal_b": cal_b,
            "known_len": float(len(known)),
        },
    )


def add_ncc_gr_match_features(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    max_wells: int | None,
) -> pd.DataFrame:
    if not (configured_feature_columns(config) & NCC_GR_MATCH_FEATURE_COLUMNS):
        return frame

    frame = frame.copy()
    path_by_well = file_by_well(paths, max_wells)
    for column in NCC_GR_MATCH_FEATURE_COLUMNS:
        frame[column] = 0.0

    scales = np.asarray([8.0, 15.0, 25.0], dtype=float)
    for (well_id, cutoff_row), part in frame.groupby(["well_id", "cutoff_row"], sort=False):
        horizontal_path = path_by_well.get(str(well_id))
        if horizontal_path is None:
            raise ValueError(f"missing train file for NCC features: {well_id}")
        eval_indices, generated, summary = compute_ncc_gr_match_group(
            horizontal_path=horizontal_path,
            cutoff_row=int(cutoff_row),
        )
        if not generated:
            continue
        by_row = pd.DataFrame(
            {
                key: value
                for key, value in generated.items()
                if key not in {"typewell_tvt", "typewell_gr"}
            },
            index=eval_indices,
        )
        aligned = by_row.reindex(part["row_idx"].to_numpy(dtype=int))
        if aligned.isna().any().any():
            raise ValueError(f"missing NCC rows for {well_id} cutoff={cutoff_row}")

        positions = part.index.to_numpy(dtype=int)
        last_anchor = part["last_anchor_tvt"].to_numpy(dtype=float)
        public_beam = part["beam_pred"].to_numpy(dtype=float)
        public_pf = part["pf_pred"].to_numpy(dtype=float)
        sc8 = aligned["sc8"].to_numpy(dtype=float)
        sc15 = aligned["sc15"].to_numpy(dtype=float)
        sc25 = aligned["sc25"].to_numpy(dtype=float)
        sc_ens = aligned["sc_ens"].to_numpy(dtype=float)
        sc_cons = (sc8 + sc15 + sc25) / 3.0
        score_matrix = aligned[["sc8_score", "sc15_score", "sc25_score"]].to_numpy(dtype=float)
        sorted_scores = np.sort(score_matrix, axis=1)
        score_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
        path_matrix = np.column_stack([sc8, sc15, sc25])
        eval_gr = aligned["eval_gr"].to_numpy(dtype=float)
        typewell_tvt = generated["typewell_tvt"]
        typewell_gr = generated["typewell_gr"]

        frame.loc[positions, "ncc_sc8_delta"] = sc8 - last_anchor
        frame.loc[positions, "ncc_sc15_delta"] = sc15 - last_anchor
        frame.loc[positions, "ncc_sc25_delta"] = sc25 - last_anchor
        frame.loc[positions, "ncc_sc_cons_delta"] = sc_cons - last_anchor
        frame.loc[positions, "ncc_sc_ens_delta"] = sc_ens - last_anchor
        frame.loc[positions, "ncc_sc8_score"] = score_matrix[:, 0]
        frame.loc[positions, "ncc_sc15_score"] = score_matrix[:, 1]
        frame.loc[positions, "ncc_sc25_score"] = score_matrix[:, 2]
        frame.loc[positions, "ncc_score_mean"] = score_matrix.mean(axis=1)
        frame.loc[positions, "ncc_score_std"] = score_matrix.std(axis=1)
        frame.loc[positions, "ncc_score_min"] = score_matrix.min(axis=1)
        frame.loc[positions, "ncc_score_max"] = score_matrix.max(axis=1)
        frame.loc[positions, "ncc_score_gap_best_second"] = score_gap
        frame.loc[positions, "ncc_best_scale"] = scales[np.argmax(score_matrix, axis=1)]
        frame.loc[positions, "ncc_sc_trust"] = float(
            np.clip(float(summary.get("known_len", 0.0)) / 200.0, 0.0, 0.6)
        )
        frame.loc[positions, "ncc_sc_range"] = path_matrix.max(axis=1) - path_matrix.min(axis=1)
        frame.loc[positions, "ncc_sc_std"] = path_matrix.std(axis=1)
        frame.loc[positions, "ncc_sc_ens_vs_public_beam"] = sc_ens - public_beam
        frame.loc[positions, "abs_ncc_sc_ens_vs_public_beam"] = np.abs(sc_ens - public_beam)
        frame.loc[positions, "ncc_sc_ens_vs_pf"] = sc_ens - public_pf
        frame.loc[positions, "abs_ncc_sc_ens_vs_pf"] = np.abs(sc_ens - public_pf)
        frame.loc[positions, "ncc_sc_cons_vs_public_beam"] = sc_cons - public_beam
        frame.loc[positions, "abs_ncc_sc_cons_vs_public_beam"] = np.abs(sc_cons - public_beam)
        frame.loc[positions, "ncc_sc_cons_vs_pf"] = sc_cons - public_pf
        frame.loc[positions, "abs_ncc_sc_cons_vs_pf"] = np.abs(sc_cons - public_pf)
        frame.loc[positions, "gr_match_prefix_rmse"] = float(summary["prefix_rmse"])
        frame.loc[positions, "gr_match_cal_a"] = float(summary["cal_a"])
        frame.loc[positions, "gr_match_cal_b"] = float(summary["cal_b"])

        def gr_residual(
            tvt_values: np.ndarray | float,
            *,
            current_gr: np.ndarray = eval_gr,
            current_typewell_tvt: np.ndarray = typewell_tvt,
            current_typewell_gr: np.ndarray = typewell_gr,
        ) -> np.ndarray:
            return current_gr - np.interp(
                tvt_values,
                current_typewell_tvt,
                current_typewell_gr,
            ).astype(float)

        frame.loc[positions, "gr_vs_tw_anchor"] = gr_residual(last_anchor)
        frame.loc[positions, "gr_vs_sc_ens"] = gr_residual(sc_ens)
        frame.loc[positions, "gr_vs_public_beam"] = gr_residual(public_beam)
        frame.loc[positions, "gr_vs_pf"] = gr_residual(public_pf)
        frame.loc[positions, "gr_vs_anchor_m25"] = gr_residual(last_anchor - 25.0)
        frame.loc[positions, "gr_vs_anchor_p25"] = gr_residual(last_anchor + 25.0)
        frame.loc[positions, "gr_vs_sc_m25"] = gr_residual(sc_ens - 25.0)
        frame.loc[positions, "gr_vs_sc_p25"] = gr_residual(sc_ens + 25.0)
        frame.loc[positions, "gr_vs_beam_m25"] = gr_residual(public_beam - 25.0)
        frame.loc[positions, "gr_vs_beam_p25"] = gr_residual(public_beam + 25.0)
        frame.loc[positions, "gr_vs_pf_m25"] = gr_residual(public_pf - 25.0)
        frame.loc[positions, "gr_vs_pf_p25"] = gr_residual(public_pf + 25.0)
        print(
            json.dumps(
                {
                    "ncc_gr_match": "generated",
                    "well_id": str(well_id),
                    "cutoff_row": int(cutoff_row),
                    "rows": int(len(part)),
                    "prefix_rmse": round(float(summary["prefix_rmse"]), 6),
                    "score_mean": round(float(score_matrix.mean()), 6),
                },
                sort_keys=True,
            )
        )

    missing_features = sorted(configured_feature_columns(config) - set(frame.columns))
    if missing_features:
        raise ValueError(f"Feature frame is missing configured features: {missing_features}")
    for column in NCC_GR_MATCH_FEATURE_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            frame[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return frame


def bucket_codes(eval_step: np.ndarray, buckets: list[dict[str, Any]]) -> np.ndarray:
    codes = np.full(eval_step.shape, len(buckets) - 1, dtype=np.int16)
    previous_max = -np.inf
    for idx, bucket in enumerate(buckets):
        max_step = float(bucket["max_step"])
        mask = (eval_step > previous_max) & (eval_step <= max_step)
        codes[mask] = idx
        previous_max = max_step
    return codes


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


def make_estimator(config: dict[str, Any], *, seed: int) -> Any:
    estimator = str(get_nested(config, "model.estimator", "LGBMRegressor"))
    params = dict(get_nested(config, "model.params", {}) or {})
    if estimator == "LGBMRegressor":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LGBMRegressor requires lightgbm in the runtime") from exc
        params.setdefault("objective", "regression")
        params.setdefault("random_state", seed)
        params.setdefault("verbosity", -1)
        return LGBMRegressor(**params)
    if estimator == "HistGradientBoostingRegressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        translated = {
            "loss": "squared_error",
            "max_iter": int(params.pop("max_iter", params.pop("n_estimators", 120))),
            "learning_rate": float(params.pop("learning_rate", 0.04)),
            "max_leaf_nodes": int(params.pop("max_leaf_nodes", params.pop("num_leaves", 31))),
            "min_samples_leaf": int(
                params.pop("min_samples_leaf", params.pop("min_child_samples", 80))
            ),
            "l2_regularization": float(
                params.pop("l2_regularization", params.pop("reg_lambda", 0.1))
            ),
            "early_stopping": False,
            "random_state": seed,
        }
        return HistGradientBoostingRegressor(**translated)
    raise ValueError(f"unsupported model.estimator: {estimator}")


def transformed_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"feature columns missing: {missing}")
    return frame.loc[:, list(columns)].copy().astype("float32")


def apply_prediction_policy(
    *,
    residual: np.ndarray,
    last_anchor: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    residual *= float(get_nested(config, "model.prediction.residual_shrink", 1.0))
    max_abs_residual = float(get_nested(config, "model.prediction.max_abs_residual", 0.0))
    if max_abs_residual > 0:
        residual = np.clip(residual, -max_abs_residual, max_abs_residual)
    return np.asarray(last_anchor, dtype=float) + residual


def postprocess_bucket_shrink(
    pred: np.ndarray,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    buckets = list(get_nested(config, "postprocess.methods.distance_bucket_shrink.buckets", []))
    alphas = distance_bucket_alphas(frame["eval_step"].to_numpy(dtype=float), buckets)
    anchor = frame["last_anchor_tvt"].to_numpy(dtype=float)
    return anchor + alphas * (np.asarray(pred, dtype=float) - anchor)


def cross_fit_variant_predictions(
    *,
    audit: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    split_codes: np.ndarray,
    bucket_code_values: np.ndarray,
    specs: list[VariantSpec],
    max_train_rows_override: int | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    y_residual = frame["target_tvt"].to_numpy(dtype=float) - frame["last_anchor_tvt"].to_numpy(
        dtype=float
    )
    last_anchor = frame["last_anchor_tvt"].to_numpy(dtype=float)
    max_rows = max_train_rows_override
    if max_rows is None:
        max_rows = int(get_nested(config, "model.training.max_train_rows_per_split", 300000))
    max_rows_per_bucket = int(get_nested(config, "model.training.max_train_rows_per_bucket", 60000))
    seed = int(get_nested(config, "model.training.seed", 42))
    unique_splits = sorted(int(value) for value in np.unique(split_codes))

    predictions: dict[str, np.ndarray] = {}
    train_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for spec in specs:
        raw_oof = np.full(len(frame), np.nan, dtype=float)
        x_matrix = transformed_features(frame, spec.feature_columns)
        for split in unique_splits:
            valid_mask = split_codes == split
            train_idx = choose_train_indices(
                ~valid_mask,
                bucket_code_values,
                max_rows=max_rows,
                max_rows_per_bucket=max_rows_per_bucket,
                seed=seed + split * 7919 + stable_fold(spec.name, 997),
            )
            valid_idx = np.flatnonzero(valid_mask)
            model = make_estimator(config, seed=seed + split * 1009)
            model.fit(x_matrix.iloc[train_idx], y_residual[train_idx])
            raw_oof[valid_idx] = apply_prediction_policy(
                residual=model.predict(x_matrix.iloc[valid_idx]),
                last_anchor=last_anchor[valid_idx],
                config=config,
            )
            train_rows.append(
                {
                    "audit": audit,
                    "variant": spec.name,
                    "split": split,
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                    "features": int(len(spec.feature_columns)),
                }
            )
            if hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)
                for column, importance in zip(spec.feature_columns, importances, strict=False):
                    importance_rows.append(
                        {
                            "audit": audit,
                            "variant": spec.name,
                            "split": split,
                            "feature": column,
                            "importance": float(importance),
                        }
                    )
            print(
                json.dumps(
                    {
                        "audit": audit,
                        "variant": spec.name,
                        "split": split,
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                    },
                    sort_keys=True,
                )
            )
        if not np.isfinite(raw_oof).all():
            raise ValueError(f"{audit}/{spec.name}: non-finite OOF predictions")
        predictions[f"{spec.name}_raw"] = raw_oof
        predictions[f"{spec.name}_bucket_shrink"] = postprocess_bucket_shrink(
            raw_oof,
            frame,
            config,
        )
    return predictions, train_rows, importance_rows


def selected_exp026_training_variant(config: dict[str, Any]) -> dict[str, Any]:
    selected = str(get_nested(config, "audit.exp026_training.training_variants.selected_variant"))
    variants = list(get_nested(config, "audit.exp026_training.training_variants.variants", []))
    for variant in variants:
        if isinstance(variant, dict) and str(variant.get("name")) == selected:
            return variant
    raise ValueError(f"selected exp026 pseudo-tail variant not found: {selected}")


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


def predict_exp026_well_cutoff(
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
    pred = postprocess_predictions(raw_pred, frame, config, method="distance_bucket_shrink")
    by_row = pd.Series(pred, index=frame.eval_indices.astype(int))
    output = by_row.reindex(row_indices.astype(int)).to_numpy(dtype=float)
    if not np.isfinite(output).all():
        missing = row_indices[~np.isfinite(output)]
        preview = ", ".join(str(int(value)) for value in missing[:5])
        raise ValueError(f"missing exp026 control predictions for {path.name} rows: {preview}")
    return output


def exp026_control_config(config: dict[str, Any]) -> dict[str, Any]:
    exp026_config = dict(get_nested(config, "audit.exp026_training", {}) or {})
    exp026_config["data"] = dict(config.get("data", {}) or {})
    exp026_config["validation"] = dict(config.get("validation", {}) or {})
    exp026_config["postprocess"] = dict(config.get("postprocess", {}) or {})
    exp026_config["audit"] = {
        "pseudo_tail": get_nested(config, "audit.exp026_training.pseudo_tail", {}),
        "distance_buckets": get_nested(config, "audit.distance_buckets", []),
    }
    return exp026_config


def generate_exp026_control(
    *,
    audit: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    paths: ExperimentPaths,
    split_codes: np.ndarray,
    max_wells: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    control_config = exp026_control_config(config)
    variant = selected_exp026_training_variant(config)
    seed = int(config_get(control_config, "validation.seed", 42))
    max_rows_per_well = int(
        get_nested(config, "audit.exp026_training.training.max_train_rows_per_well", 800)
    )
    max_rows_total = int(
        get_nested(config, "audit.exp026_training.training.max_train_rows_per_fold", 300000)
    )
    path_by_well = file_by_well(paths, max_wells)
    split_by_well = split_code_by_well(frame, split_codes)
    missing_wells = sorted(set(frame["well_id"].astype(str)) - set(path_by_well))
    if missing_wells:
        preview = ", ".join(missing_wells[:5])
        raise ValueError(f"feature rows reference wells without local train files: {preview}")

    output = np.full(len(frame), np.nan, dtype=float)
    source_rows: list[dict[str, Any]] = []
    all_paths = list(path_by_well.values())
    for split in sorted(int(value) for value in np.unique(split_codes)):
        train_paths = [
            path
            for path in all_paths
            if split_by_well.get(well_id_from_path(path), -1) != split
        ]
        model, n_train_rows, split_source_rows = fit_pseudo_tail_model_from_files(
            train_paths,
            control_config,
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
            output[positions] = predict_exp026_well_cutoff(
                path=path_by_well[str(well_id)],
                cutoff_row=int(cutoff_row),
                row_indices=part["row_idx"].to_numpy(dtype=int),
                model=model,
                config=control_config,
            )
        print(
            json.dumps(
                {
                    "audit": audit,
                    "control": "exp026_regenerated_bucket_shrink",
                    "split": split,
                    "pseudo_tail_train_rows": n_train_rows,
                    "valid_rows": int(len(valid_positions)),
                },
                sort_keys=True,
            )
        )
    if not np.isfinite(output).all():
        raise ValueError(f"{audit}: non-finite exp026 control predictions")
    return output, source_rows


def control_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "last_anchor": frame["last_anchor_tvt"].to_numpy(dtype=float),
        "public_pf_selector": frame["pf_pred"].to_numpy(dtype=float),
        "pf090_hold010": 0.90 * frame["pf_pred"].to_numpy(dtype=float)
        + 0.10 * frame["last_anchor_tvt"].to_numpy(dtype=float),
        "beam": frame["beam_pred"].to_numpy(dtype=float),
    }


def aggregate_global(
    *,
    audit: str,
    candidate: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    references: dict[str, np.ndarray],
) -> dict[str, Any]:
    score = rmse(y_true, pred)
    row: dict[str, Any] = {
        "audit": audit,
        "candidate": candidate,
        "rmse": round(score, 6),
        "rows": int(len(y_true)),
    }
    for ref_name, ref_pred in references.items():
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
                code_name: label,
                "candidate": candidate,
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
    specs: list[VariantSpec],
    max_wells: int | None,
    max_train_rows_override: int | None,
    skip_exp026_control: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = frame["target_tvt"].to_numpy(dtype=float)
    predictions = control_predictions(frame)
    exp026_source = pd.DataFrame()
    if (
        bool(get_nested(config, "audit.include_exp026_regenerated_control", True))
        and not skip_exp026_control
    ):
        exp026_pred, source_rows = generate_exp026_control(
            audit=audit,
            frame=frame,
            config=config,
            paths=paths,
            split_codes=split_codes,
            max_wells=max_wells,
        )
        predictions["exp026_regenerated_bucket_shrink"] = exp026_pred
        exp026_source = pd.DataFrame(source_rows)

    candidate_predictions, train_rows, importance_rows = cross_fit_variant_predictions(
        audit=audit,
        frame=frame,
        config=config,
        split_codes=split_codes,
        bucket_code_values=bucket_code_values,
        specs=specs,
        max_train_rows_override=max_train_rows_override,
    )
    predictions.update(candidate_predictions)

    reference_name = str(
        get_nested(config, "audit.reference_control", "base_geometry_bucket_shrink")
    )
    if reference_name not in predictions:
        raise ValueError(f"reference control {reference_name!r} was not generated")
    reference = predictions[reference_name]
    references = {
        name: pred
        for name, pred in predictions.items()
        if name in {reference_name, "exp026_regenerated_bucket_shrink", "public_pf_selector"}
    }

    metric_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    well_frames: list[pd.DataFrame] = []
    for candidate, pred in predictions.items():
        metric_rows.append(
            aggregate_global(
                audit=audit,
                candidate=candidate,
                pred=pred,
                y_true=y_true,
                references=references,
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
                reference=reference,
            )
        )
        split_rows.extend(
            aggregate_by_code(
                audit=audit,
                candidate=candidate,
                pred=pred,
                y_true=y_true,
                codes=split_codes,
                code_name="split",
                labels=split_labels,
                reference=reference,
            )
        )
        well_frames.append(
            aggregate_well_metrics(
                audit=audit,
                candidate=candidate,
                frame=frame,
                pred=pred,
                y_true=y_true,
                reference=reference,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(split_rows),
        pd.concat(well_frames, ignore_index=True),
        pd.DataFrame(train_rows),
        pd.DataFrame(importance_rows),
        exp026_source,
    )


def summarize_support(metrics: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    reference_name = str(
        get_nested(config, "audit.reference_control", "base_geometry_bucket_shrink")
    )
    original = metrics[metrics["audit"] == "leave_one_original_fold_out"]
    well_hash = metrics[metrics["audit"] == "well_hash_holdout"]
    original_by_candidate = dict(zip(original["candidate"], original["rmse"], strict=False))
    well_hash_by_candidate = dict(zip(well_hash["candidate"], well_hash["rmse"], strict=False))
    required = set(str(value) for value in get_nested(config, "audit.required_controls", []))
    supported: list[dict[str, Any]] = []
    reference_original = float(original_by_candidate[reference_name])
    reference_well_hash = float(well_hash_by_candidate[reference_name])
    for candidate, original_rmse in original_by_candidate.items():
        if candidate in required or candidate not in well_hash_by_candidate:
            continue
        if not candidate.endswith("_bucket_shrink"):
            continue
        well_rmse = float(well_hash_by_candidate[candidate])
        original_rmse = float(original_rmse)
        if original_rmse < reference_original and well_rmse < reference_well_hash:
            supported.append(
                {
                    "candidate": candidate,
                    "original_fold_rmse": round(original_rmse, 6),
                    "well_hash_rmse": round(well_rmse, 6),
                    "delta_vs_reference_original": round(original_rmse - reference_original, 6),
                    "delta_vs_reference_well_hash": round(well_rmse - reference_well_hash, 6),
                    "worst_holdout_rmse": round(max(original_rmse, well_rmse), 6),
                }
            )
    return sorted(supported, key=lambda row: row["worst_holdout_rmse"])


def run_audit(
    paths: ExperimentPaths,
    config: dict[str, Any],
    feature_path: Path,
    output_dir: Path | None = None,
    *,
    max_wells: int | None = None,
    max_train_rows_override: int | None = None,
    skip_exp026_control: bool = False,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_dir = output_dir or paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    path_by_well = file_by_well(paths, max_wells) if max_wells is not None else None
    allowed_wells = set(path_by_well) if path_by_well is not None else None
    frame, loaded_columns = load_features(feature_path, config, allowed_wells=allowed_wells)
    frame = add_beam_exact_features(frame, config=config, paths=paths, max_wells=max_wells)
    frame = add_ncc_gr_match_features(frame, config=config, paths=paths, max_wells=max_wells)
    buckets = list(get_nested(config, "audit.distance_buckets", []))
    if not buckets:
        raise ValueError("audit.distance_buckets must be non-empty")
    bucket_code_values = bucket_codes(frame["eval_step"].to_numpy(dtype=float), buckets)
    bucket_labels = [str(bucket["name"]) for bucket in buckets]
    specs = feature_variants(config)

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
    train_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    exp026_source_frames: list[pd.DataFrame] = []
    for audit_name, (split_codes, split_labels) in audit_outputs.items():
        (
            metrics,
            bucket_metrics,
            split_metrics,
            well_metrics,
            train_rows,
            importance_rows,
            exp026_source,
        ) = run_one_audit(
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
            max_train_rows_override=max_train_rows_override,
            skip_exp026_control=skip_exp026_control,
        )
        metric_frames.append(metrics)
        bucket_frames.append(bucket_metrics)
        split_frames.append(split_metrics)
        well_frames.append(well_metrics)
        train_frames.append(train_rows)
        importance_frames.append(importance_rows)
        if not exp026_source.empty:
            exp026_source_frames.append(exp026_source)

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["audit", "rmse"])
    bucket_metrics = pd.concat(bucket_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "bucket"]
    )
    split_metrics = pd.concat(split_frames, ignore_index=True).sort_values(
        ["audit", "candidate", "split"]
    )
    well_metrics = pd.concat(well_frames, ignore_index=True)
    train_summary = pd.concat(train_frames, ignore_index=True)
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        if any(not frame.empty for frame in importance_frames)
        else pd.DataFrame(columns=["audit", "variant", "split", "feature", "importance"])
    )
    exp026_source = (
        pd.concat(exp026_source_frames, ignore_index=True)
        if exp026_source_frames
        else pd.DataFrame()
    )
    supported = summarize_support(metrics, config)

    original = metrics[metrics["audit"] == "leave_one_original_fold_out"]
    well_hash = metrics[metrics["audit"] == "well_hash_holdout"]
    best_original = original.iloc[0].to_dict()
    best_well_hash = well_hash.iloc[0].to_dict()
    selected = supported[0] if supported else None
    summary = {
        "experiment": str(get_nested(config, "experiment.name")),
        "status": "completed" if selected is not None else "implemented_no_supported_candidate_yet",
        "updated_at": datetime.now(UTC).isoformat(),
        "source_experiment": get_nested(config, "data.feature_path"),
        "feature_file": feature_path.as_posix(),
        "loaded_columns": loaded_columns,
        "feature_variants": [
            {"name": spec.name, "features": list(spec.feature_columns)} for spec in specs
        ],
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "max_wells": max_wells,
        "metric": "rmse",
        "target": "target_tvt_minus_last_anchor_tvt",
        "parent_exp026_clean_cv": get_nested(config, "audit.parent_exp026_clean_cv"),
        "reference_control": get_nested(config, "audit.reference_control"),
        "best_original_fold_candidate": best_original["candidate"],
        "best_original_fold_cv": round(float(best_original["rmse"]), 6),
        "best_well_hash_candidate": best_well_hash["candidate"],
        "best_well_hash_cv": round(float(best_well_hash["rmse"]), 6),
        "supported_candidates": supported,
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_clean_cv": None if selected is None else selected["original_fold_rmse"],
        "single_lgbm_feature_supported": selected is not None,
        "notes": (
            "A Ravaghi NCC/GR match feature candidate beat the base single-LGBM reference "
            "in both holdout audits."
            if selected is not None
            else (
                "No Ravaghi NCC/GR match feature candidate beat the base single-LGBM "
                "reference in both holdout audits."
            )
        ),
    }

    metrics.to_csv(output_dir / "single_lgbm_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / "single_lgbm_bucket_metrics.csv", index=False)
    split_metrics.to_csv(output_dir / "single_lgbm_split_metrics.csv", index=False)
    well_metrics.to_csv(output_dir / "single_lgbm_well_metrics.csv", index=False)
    train_summary.to_csv(output_dir / "single_lgbm_train_summary.csv", index=False)
    importance.to_csv(output_dir / "single_lgbm_feature_importance.csv", index=False)
    if not exp026_source.empty:
        exp026_source.to_csv(output_dir / "single_lgbm_exp026_source_summary.csv", index=False)
    with (output_dir / "single_lgbm_summary.json").open("w") as fp:
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
        config.setdefault("model", {})["estimator"] = args.base_estimator
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
        skip_exp026_control=args.skip_exp026_control,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
