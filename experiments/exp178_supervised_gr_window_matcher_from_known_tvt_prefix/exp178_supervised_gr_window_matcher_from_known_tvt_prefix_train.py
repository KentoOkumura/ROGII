# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp178 supervised GR window matcher from known TVT prefix train
#
# Train-side smoke for learning a GR window match scorer from observed
# `TVT_input` prefix alignments. This does not create a TVT replacement,
# inference branch, or submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. GR window pair dataset helpers
# 4. Supervised matcher and diagnostics
# 5. Setup and input checks
# 6. Run 1-fold row-cap smoke
# 7. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
OUTPUT_PREFIX = EXPERIMENT_NAME
GR_FEATURE_NAMES = [
    "raw_abs",
    "window_mae",
    "window_rmse",
    "window_ncc",
    "z_mae",
    "derivative_mae",
    "energy_abs",
    "missing_mean",
    "combo_score",
]
CONTEXT_FEATURES = [
    "candidate_minus_last_known_tvt",
    "candidate_abs_minus_last_known_tvt",
    "candidate_tvt_pct",
    "candidate_outside_typewell_ft",
    "md_to_last_known",
    "abs_md_to_last_known",
    "z_to_last_known",
    "abs_z_to_last_known",
    "prefix_fraction",
    "known_prefix_rows",
    "row_gr_missing_window",
    "typewell_gr_missing_window",
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: str, modulo: int | None = None) -> int:
    payload = "::".join(parts).encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value % int(modulo) if modulo else value


def list_train_wells(train_dir: Path, max_wells: int | None) -> list[str]:
    wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in train_dir.glob("*__horizontal_well.csv")
    )
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    return wells


def finite_mean(values: pd.Series | np.ndarray, default: float = 0.0) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(np.float32)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if len(finite) else float(default)


def fill_numeric_series(series: pd.Series, fallback: float) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(float(fallback))
        .to_numpy(np.float32)
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def nearest_indices(sorted_tvt: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_tvt, values, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - values) < np.abs(sorted_tvt[left] - values)
    return np.where(choose_right, right, left).astype(np.int32)


def gather_windows(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    indices = np.clip(centers[:, None] + offsets.astype(np.int32), 0, len(series) - 1)
    return series[indices].astype(np.float32)


def standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True) + 1e-6
    return (centered / scale).astype(np.float32)


def window_feature_frame(
    *,
    horizontal_gr: np.ndarray,
    horizontal_missing: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_missing: np.ndarray,
    row_idx: np.ndarray,
    candidate_idx: np.ndarray,
    window_offsets: np.ndarray,
    derivative_step: int,
    prefix: str,
) -> pd.DataFrame:
    h_window = gather_windows(horizontal_gr, row_idx, window_offsets)
    t_window = gather_windows(typewell_gr, candidate_idx, window_offsets)
    h_center = horizontal_gr[row_idx]
    t_center = typewell_gr[candidate_idx]
    h_norm = standardize_rows(h_window)
    t_norm = standardize_rows(t_window)
    ncc = np.mean(h_norm * t_norm, axis=1)
    h_derivative = np.gradient(horizontal_gr).astype(np.float32)
    t_derivative = np.gradient(typewell_gr).astype(np.float32)
    h_d = gather_windows(h_derivative, row_idx, window_offsets)
    t_d = gather_windows(t_derivative, candidate_idx, window_offsets)
    h_missing = gather_windows(horizontal_missing.astype(np.float32), row_idx, window_offsets)
    t_missing = gather_windows(typewell_missing.astype(np.float32), candidate_idx, window_offsets)

    raw_abs = np.abs(t_center - h_center)
    window_mae = np.mean(np.abs(t_window - h_window), axis=1)
    window_rmse = np.sqrt(np.mean(np.square(t_window - h_window), axis=1))
    z_mae = np.mean(np.abs(t_norm - h_norm), axis=1)
    derivative_mae = np.mean(np.abs(t_d - h_d), axis=1)
    energy_abs = np.abs(
        np.sqrt(np.mean(np.square(t_d), axis=1)) - np.sqrt(np.mean(np.square(h_d), axis=1))
    )
    missing_mean = 0.5 * h_missing.mean(axis=1) + 0.5 * t_missing.mean(axis=1)

    combo_cost = (
        0.20 * np.clip(raw_abs / 18.0, 0.0, 5.0)
        + 0.25 * np.clip(window_mae / 18.0, 0.0, 5.0)
        + 0.20 * np.clip(z_mae, 0.0, 5.0)
        + 0.20 * np.clip(derivative_mae / max(float(derivative_step), 1.0), 0.0, 5.0)
        + 0.10 * np.clip(energy_abs / max(float(derivative_step), 1.0), 0.0, 5.0)
        + 0.05 * np.clip(missing_mean, 0.0, 1.0)
    )
    data = {
        f"{prefix}_raw_abs": raw_abs,
        f"{prefix}_window_mae": window_mae,
        f"{prefix}_window_rmse": window_rmse,
        f"{prefix}_window_ncc": ncc,
        f"{prefix}_z_mae": z_mae,
        f"{prefix}_derivative_mae": derivative_mae,
        f"{prefix}_energy_abs": energy_abs,
        f"{prefix}_missing_mean": missing_mean,
        f"{prefix}_combo_score": np.exp(-combo_cost),
    }
    return pd.DataFrame({key: np.asarray(value, dtype=np.float32) for key, value in data.items()})


# %% [markdown]
# ## 3. GR window pair dataset helpers

# %%
@dataclass(frozen=True)
class WellArrays:
    well: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame
    horizontal_gr: np.ndarray
    horizontal_missing: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    typewell_missing: np.ndarray
    prefix_end: int
    last_known_tvt: float
    last_known_md: float
    last_known_z: float


def read_well_arrays(well: str, train_dir: Path, smoothing_window: int) -> WellArrays | None:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        return None
    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "GR", "TVT_input"])
    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        return None
    prefix_end = int(known[-1] + 1)
    last_idx = int(known[-1])
    horizontal_fallback = finite_mean(
        horizontal["GR"].iloc[:prefix_end],
        finite_mean(horizontal["GR"]),
    )
    typewell_fallback = finite_mean(typewell["GR"])
    horizontal_missing = pd.to_numeric(horizontal["GR"], errors="coerce").isna().to_numpy()
    typewell_missing = pd.to_numeric(typewell["GR"], errors="coerce").isna().to_numpy()
    horizontal_gr = rolling_mean(
        fill_numeric_series(horizontal["GR"], horizontal_fallback),
        smoothing_window,
    )
    typewell_gr = rolling_mean(
        fill_numeric_series(typewell["GR"], typewell_fallback),
        smoothing_window,
    )
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    finite_typewell = np.isfinite(typewell_tvt)
    if finite_typewell.sum() < 4:
        return None
    if not finite_typewell.all():
        keep = np.flatnonzero(finite_typewell)
        typewell_tvt = typewell_tvt[keep]
        typewell_gr = typewell_gr[keep]
        typewell_missing = typewell_missing[keep]
        typewell = typewell.iloc[keep].reset_index(drop=True)
    return WellArrays(
        well=well,
        horizontal=horizontal,
        typewell=typewell,
        horizontal_gr=horizontal_gr,
        horizontal_missing=horizontal_missing,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        typewell_missing=typewell_missing,
        prefix_end=prefix_end,
        last_known_tvt=float(tvt_input[last_idx]),
        last_known_md=float(pd.to_numeric(horizontal["MD"], errors="coerce").iloc[last_idx]),
        last_known_z=float(pd.to_numeric(horizontal["Z"], errors="coerce").iloc[last_idx]),
    )


def selected_prefix_rows(arrays: WellArrays, rows_per_well: int, row_margin: int) -> np.ndarray:
    tvt_input = pd.to_numeric(arrays.horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input))
    eligible = known[(known >= int(row_margin)) & (known < arrays.prefix_end - int(row_margin))]
    if len(eligible) == 0:
        eligible = known
    if len(eligible) > int(rows_per_well):
        positions = np.linspace(0, len(eligible) - 1, int(rows_per_well)).round().astype(np.int64)
        eligible = eligible[positions]
    return eligible.astype(np.int32)


def hard_decoy_offsets(
    arrays: WellArrays,
    row_idx: np.ndarray,
    true_tvt: np.ndarray,
    scan_offsets: np.ndarray,
    window_offsets: np.ndarray,
    derivative_step: int,
) -> np.ndarray:
    if len(scan_offsets) == 0:
        return np.zeros(len(row_idx), dtype=np.float32)
    candidate_tvt = (true_tvt[:, None] + scan_offsets[None, :]).astype(np.float32)
    candidate_idx = nearest_indices(arrays.typewell_tvt, candidate_tvt.reshape(-1))
    row_repeat = np.repeat(row_idx, len(scan_offsets))
    features = window_feature_frame(
        horizontal_gr=arrays.horizontal_gr,
        horizontal_missing=arrays.horizontal_missing,
        typewell_gr=arrays.typewell_gr,
        typewell_missing=arrays.typewell_missing,
        row_idx=row_repeat,
        candidate_idx=candidate_idx,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
        prefix="scan",
    )
    score = features["scan_combo_score"].to_numpy(np.float32).reshape(
        len(row_idx),
        len(scan_offsets),
    )
    best = score.argmax(axis=1)
    return scan_offsets[best].astype(np.float32)


def build_pairs_for_well(
    well: str,
    train_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    pair_cfg = get_nested(config, "pair_dataset") or {}
    smoothing_window = int(pair_cfg.get("gr_smoothing_window", 5))
    rows_per_well = int(pair_cfg.get("rows_per_well", 64))
    row_margin = int(pair_cfg.get("row_margin", 32))
    derivative_step = int(pair_cfg.get("derivative_step", 3))
    window_offsets = np.asarray(
        pair_cfg.get("window_offsets", [-24, -12, -6, 0, 6, 12, 24]),
        dtype=np.int32,
    )
    negative_offsets = np.asarray(
        pair_cfg.get("negative_offsets_ft", [-100, -50, -25, -15, 15, 25, 50, 100]),
        dtype=np.float32,
    )
    scan_offsets = np.asarray(pair_cfg.get("hard_decoy_scan_offsets_ft", []), dtype=np.float32)

    arrays = read_well_arrays(well, train_dir, smoothing_window)
    if arrays is None:
        return None, {"well": well, "status": "missing_or_invalid_raw"}
    row_idx = selected_prefix_rows(arrays, rows_per_well, row_margin)
    if len(row_idx) == 0:
        return None, {"well": well, "status": "no_eligible_prefix_rows"}

    tvt_input = pd.to_numeric(arrays.horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    md = pd.to_numeric(arrays.horizontal["MD"], errors="coerce").to_numpy(np.float32)
    z = pd.to_numeric(arrays.horizontal["Z"], errors="coerce").to_numpy(np.float32)
    true_tvt = tvt_input[row_idx]
    hard_offsets = hard_decoy_offsets(
        arrays,
        row_idx=row_idx,
        true_tvt=true_tvt,
        scan_offsets=scan_offsets,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
    )

    candidate_offsets = [np.zeros(len(row_idx), dtype=np.float32)]
    pair_kinds = ["positive"]
    labels = [np.ones(len(row_idx), dtype=np.int8)]
    for offset in negative_offsets:
        candidate_offsets.append(np.full(len(row_idx), float(offset), dtype=np.float32))
        pair_kinds.append(f"decoy_{float(offset):+.0f}ft".replace("+", "p").replace("-", "m"))
        labels.append(np.zeros(len(row_idx), dtype=np.int8))
    if len(scan_offsets):
        candidate_offsets.append(hard_offsets)
        pair_kinds.append("hard_local_decoy")
        labels.append(np.zeros(len(row_idx), dtype=np.int8))

    offset_matrix = np.column_stack(candidate_offsets).astype(np.float32)
    label_matrix = np.column_stack(labels).astype(np.int8)
    n_rows, n_candidates = offset_matrix.shape
    candidate_tvt = (true_tvt[:, None] + offset_matrix).astype(np.float32)
    candidate_idx = nearest_indices(arrays.typewell_tvt, candidate_tvt.reshape(-1))
    row_repeat = np.repeat(row_idx, n_candidates)

    real_features = window_feature_frame(
        horizontal_gr=arrays.horizontal_gr,
        horizontal_missing=arrays.horizontal_missing,
        typewell_gr=arrays.typewell_gr,
        typewell_missing=arrays.typewell_missing,
        row_idx=row_repeat,
        candidate_idx=candidate_idx,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
        prefix="real",
    )
    roll = (
        stable_int(
            "exp178_shuffled_typewell_gr",
            well,
            modulo=max(len(arrays.typewell_gr) - 1, 1),
        )
        + 1
    )
    shuffled_features = window_feature_frame(
        horizontal_gr=arrays.horizontal_gr,
        horizontal_missing=arrays.horizontal_missing,
        typewell_gr=np.roll(arrays.typewell_gr, int(roll)),
        typewell_missing=np.roll(arrays.typewell_missing, int(roll)),
        row_idx=row_repeat,
        candidate_idx=candidate_idx,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
        prefix="shuf",
    )

    typewell_min = float(np.nanmin(arrays.typewell_tvt))
    typewell_max = float(np.nanmax(arrays.typewell_tvt))
    typewell_span = max(typewell_max - typewell_min, 1.0)
    row_md = md[row_idx]
    row_z = z[row_idx]
    row_gr_missing = (
        gather_windows(arrays.horizontal_missing.astype(np.float32), row_idx, window_offsets)
        .mean(axis=1)
        .astype(np.float32)
    )
    typewell_gr_missing = (
        gather_windows(arrays.typewell_missing.astype(np.float32), candidate_idx, window_offsets)
        .mean(axis=1)
        .astype(np.float32)
    )
    outside = np.maximum(0.0, typewell_min - candidate_tvt.reshape(-1)) + np.maximum(
        0.0,
        candidate_tvt.reshape(-1) - typewell_max,
    )

    pairs = pd.DataFrame(
        {
            "anchor_id": [f"{well}_{int(idx)}" for idx in np.repeat(row_idx, n_candidates)],
            "pair_id": [
                f"{well}_{int(idx)}_{kind}"
                for idx in row_idx
                for kind in pair_kinds
            ],
            "well": well,
            "row_idx": row_repeat.astype(np.int32),
            "pair_kind": np.tile(pair_kinds, n_rows),
            "label_within_10ft": label_matrix.reshape(-1).astype(np.int8),
            "abs_tvt_offset": np.abs(offset_matrix.reshape(-1)).astype(np.float32),
            "signed_tvt_offset": offset_matrix.reshape(-1).astype(np.float32),
            "true_prefix_tvt": np.repeat(true_tvt, n_candidates).astype(np.float32),
            "candidate_tvt": candidate_tvt.reshape(-1).astype(np.float32),
            "candidate_minus_last_known_tvt": (
                candidate_tvt.reshape(-1) - arrays.last_known_tvt
            ).astype(np.float32),
            "candidate_abs_minus_last_known_tvt": np.abs(
                candidate_tvt.reshape(-1) - arrays.last_known_tvt
            ).astype(np.float32),
            "candidate_tvt_pct": (
                (candidate_tvt.reshape(-1) - typewell_min) / typewell_span
            ).astype(np.float32),
            "candidate_outside_typewell_ft": outside.astype(np.float32),
            "md_to_last_known": (
                np.repeat(row_md, n_candidates).astype(np.float32) - arrays.last_known_md
            ).astype(np.float32),
            "abs_md_to_last_known": np.abs(
                np.repeat(row_md, n_candidates).astype(np.float32) - arrays.last_known_md
            ).astype(np.float32),
            "z_to_last_known": (
                np.repeat(row_z, n_candidates).astype(np.float32) - arrays.last_known_z
            ).astype(np.float32),
            "abs_z_to_last_known": np.abs(
                np.repeat(row_z, n_candidates).astype(np.float32) - arrays.last_known_z
            ).astype(np.float32),
            "prefix_fraction": (
                np.repeat(row_idx, n_candidates) / max(arrays.prefix_end - 1, 1)
            ).astype(np.float32),
            "known_prefix_rows": np.full(
                n_rows * n_candidates,
                arrays.prefix_end,
                dtype=np.float32,
            ),
            "row_gr_missing_window": np.repeat(row_gr_missing, n_candidates).astype(np.float32),
            "typewell_gr_missing_window": typewell_gr_missing.astype(np.float32),
        }
    )
    pairs = pd.concat([pairs, real_features, shuffled_features], axis=1)
    status = {
        "well": well,
        "status": "ok",
        "prefix_rows": int(arrays.prefix_end),
        "selected_prefix_rows": int(n_rows),
        "pairs": int(len(pairs)),
        "typewell_rows": int(len(arrays.typewell_tvt)),
        "horizontal_gr_missing_rate": float(arrays.horizontal_missing.mean()),
        "typewell_gr_missing_rate": float(arrays.typewell_missing.mean()),
        "shuffled_roll": int(roll),
    }
    return pairs, status


def cap_pair_dataset(pair_frame: pd.DataFrame, max_pairs: int | None) -> pd.DataFrame:
    if max_pairs is None or len(pair_frame) <= int(max_pairs):
        return pair_frame.reset_index(drop=True)
    anchors = pair_frame[["anchor_id", "well", "row_idx"]].drop_duplicates().sort_values(
        ["well", "row_idx"]
    )
    pairs_per_anchor = int(pair_frame.groupby("anchor_id", sort=False).size().median())
    keep_anchor_count = max(1, int(max_pairs) // max(pairs_per_anchor, 1))
    keep_positions = (
        np.linspace(0, len(anchors) - 1, min(keep_anchor_count, len(anchors)))
        .round()
        .astype(np.int64)
    )
    keep_anchors = set(anchors.iloc[keep_positions]["anchor_id"].astype(str))
    return pair_frame[pair_frame["anchor_id"].isin(keep_anchors)].reset_index(drop=True)


def build_pair_dataset(
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_cfg = get_nested(config, "pair_dataset") or {}
    max_wells = pair_cfg.get("max_wells")
    max_pairs = pair_cfg.get("max_pairs")
    wells = list_train_wells(
        paths.train_data_dir,
        None if max_wells in {None, "null"} else int(max_wells),
    )
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    for well in wells:
        frame, status = build_pairs_for_well(well, paths.train_data_dir, config)
        statuses.append(status)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No supervised GR window pair rows were generated.")
    pair_frame = pd.concat(frames, ignore_index=True)
    pair_frame = cap_pair_dataset(
        pair_frame,
        None if max_pairs in {None, "null"} else int(max_pairs),
    )
    return pair_frame, pd.DataFrame(statuses)


# %% [markdown]
# ## 4. Supervised matcher and diagnostics

# %%
def feature_sets() -> dict[str, list[str]]:
    real_features = [f"real_{name}" for name in GR_FEATURE_NAMES] + CONTEXT_FEATURES
    shuffled_features = [f"shuf_{name}" for name in GR_FEATURE_NAMES] + CONTEXT_FEATURES
    no_gr_features = CONTEXT_FEATURES
    return {
        "real_gr_logistic": real_features,
        "shuffled_gr_logistic": shuffled_features,
        "no_gr_logistic": no_gr_features,
    }


def split_train_valid(
    pair_frame: pd.DataFrame,
    n_folds: int,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    anchor_groups = pair_frame[["anchor_id", "well"]].drop_duplicates().reset_index(drop=True)
    wells = anchor_groups["well"].to_numpy()
    split_count = min(int(n_folds), int(pd.Series(wells).nunique()))
    if split_count < 2:
        raise ValueError("Need at least two wells for GroupKFold smoke.")
    splitter = GroupKFold(n_splits=split_count)
    folds = list(splitter.split(anchor_groups, groups=wells))
    fold_index = int(fold_index) % split_count
    train_anchor_idx, valid_anchor_idx = folds[fold_index]
    valid_anchors = set(anchor_groups.iloc[valid_anchor_idx]["anchor_id"].astype(str))
    valid_mask = pair_frame["anchor_id"].astype(str).isin(valid_anchors).to_numpy()
    train_mask = ~valid_mask
    meta = {
        "n_folds": int(split_count),
        "fold_index": int(fold_index),
        "train_pairs": int(train_mask.sum()),
        "valid_pairs": int(valid_mask.sum()),
        "train_wells": int(pair_frame.loc[train_mask, "well"].nunique()),
        "valid_wells": int(pair_frame.loc[valid_mask, "well"].nunique()),
        "train_anchors": int(pair_frame.loc[train_mask, "anchor_id"].nunique()),
        "valid_anchors": int(pair_frame.loc[valid_mask, "anchor_id"].nunique()),
    }
    return train_mask, valid_mask, meta


def make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_regressor(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    l2_regularization=0.01,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def safe_auc(labels: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, score))


def safe_logloss(labels: np.ndarray, prob: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(log_loss(labels, np.clip(prob, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def add_handcrafted_scores(valid: pd.DataFrame) -> pd.DataFrame:
    out = valid.copy()
    out["real_combo_score"] = np.clip(out["real_combo_score"].to_numpy(np.float32), 1e-6, 1.0)
    out["shuffled_combo_score"] = np.clip(out["shuf_combo_score"].to_numpy(np.float32), 1e-6, 1.0)
    distance = np.abs(out["candidate_minus_last_known_tvt"].to_numpy(np.float32))
    out["no_gr_distance_prior_score"] = np.exp(-distance / 250.0).astype(np.float32)
    return out


def fit_models(
    pair_frame: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_sets()
    train = pair_frame.loc[train_mask].copy()
    valid = pair_frame.loc[valid_mask].copy()
    y_train = train["label_within_10ft"].to_numpy(np.int8)
    valid = add_handcrafted_scores(valid)
    importance_rows: list[dict[str, Any]] = []

    for model_name, columns in features.items():
        clf = make_classifier(seed)
        clf.fit(train[columns], y_train)
        prob = clf.predict_proba(valid[columns])[:, 1].astype(np.float32)
        valid[f"{model_name}_prob"] = np.clip(prob, 1e-6, 1.0 - 1e-6)
        coef = clf.named_steps["model"].coef_[0]
        for feature, value in zip(columns, coef, strict=False):
            importance_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "coefficient": float(value),
                    "abs_coefficient": float(abs(value)),
                }
            )

    real_columns = features["real_gr_logistic"]
    reg = make_regressor(seed)
    reg.fit(train[real_columns], train["abs_tvt_offset"].to_numpy(np.float32))
    expected_error = np.maximum(0.0, reg.predict(valid[real_columns])).astype(np.float32)
    valid["real_gr_expected_error"] = expected_error
    valid["real_gr_expected_error_score"] = (-expected_error).astype(np.float32)
    return valid, pd.DataFrame(importance_rows)


def pair_score_metrics(valid: pd.DataFrame) -> pd.DataFrame:
    labels = valid["label_within_10ft"].to_numpy(np.int8)
    score_specs = [
        ("real_gr_logistic", "real_gr_logistic_prob", "probability"),
        ("shuffled_gr_logistic", "shuffled_gr_logistic_prob", "probability"),
        ("no_gr_logistic", "no_gr_logistic_prob", "probability"),
        ("real_combo_descriptor", "real_combo_score", "probability_proxy"),
        ("shuffled_combo_descriptor", "shuffled_combo_score", "probability_proxy"),
        ("no_gr_distance_prior", "no_gr_distance_prior_score", "probability_proxy"),
        ("real_gr_expected_error", "real_gr_expected_error_score", "negative_expected_error"),
    ]
    rows: list[dict[str, Any]] = []
    for name, column, score_type in score_specs:
        score = valid[column].to_numpy(np.float32)
        row = {
            "score_name": name,
            "score_type": score_type,
            "rows": int(len(valid)),
            "positive_rate": float(labels.mean()),
            "auc": safe_auc(labels, score),
            "logloss": safe_logloss(labels, score) if "prob" in score_type else None,
            "score_mean_positive": (
                float(score[labels == 1].mean()) if (labels == 1).any() else None
            ),
            "score_mean_negative": (
                float(score[labels == 0].mean()) if (labels == 0).any() else None
            ),
        }
        if name == "real_gr_expected_error":
            row["expected_error_mae"] = float(
                mean_absolute_error(valid["abs_tvt_offset"].to_numpy(np.float32), -score)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auc", "score_name"], ascending=[False, True])


def rank_metrics(valid: pd.DataFrame, topk_values: list[int]) -> pd.DataFrame:
    score_columns = {
        "real_gr_logistic": "real_gr_logistic_prob",
        "shuffled_gr_logistic": "shuffled_gr_logistic_prob",
        "no_gr_logistic": "no_gr_logistic_prob",
        "real_combo_descriptor": "real_combo_score",
        "shuffled_combo_descriptor": "shuffled_combo_score",
        "no_gr_distance_prior": "no_gr_distance_prior_score",
        "real_gr_expected_error": "real_gr_expected_error_score",
    }
    rows: list[dict[str, Any]] = []
    for score_name, column in score_columns.items():
        ranked = valid.sort_values(["anchor_id", column], ascending=[True, False])
        top1 = ranked.groupby("anchor_id", sort=False).head(1)
        for topk in topk_values:
            selected = ranked.groupby("anchor_id", sort=False).head(int(topk))
            per_anchor = selected.groupby("anchor_id", sort=False)["abs_tvt_offset"].min()
            coverage = float((per_anchor <= 10.0).mean())
            rows.append(
                {
                    "score_name": score_name,
                    "topk": int(topk),
                    "anchors": int(per_anchor.shape[0]),
                    "within10_topk_coverage": coverage,
                    "top1_within10_rate": float((top1["abs_tvt_offset"] <= 10.0).mean()),
                    "top1_abs_offset_mean": float(top1["abs_tvt_offset"].mean()),
                    "top1_abs_offset_p90": float(top1["abs_tvt_offset"].quantile(0.90)),
                    "top1_pair_kind_top": str(top1["pair_kind"].mode().iloc[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["topk", "within10_topk_coverage", "top1_abs_offset_mean"],
        ascending=[True, False, True],
    )


def by_well_rank_metrics(valid: pd.DataFrame) -> pd.DataFrame:
    score_columns = {
        "real_gr_logistic": "real_gr_logistic_prob",
        "shuffled_gr_logistic": "shuffled_gr_logistic_prob",
        "no_gr_logistic": "no_gr_logistic_prob",
        "real_combo_descriptor": "real_combo_score",
        "real_gr_expected_error": "real_gr_expected_error_score",
    }
    rows: list[dict[str, Any]] = []
    for score_name, column in score_columns.items():
        top1 = (
            valid.sort_values(["anchor_id", column], ascending=[True, False])
            .groupby("anchor_id", sort=False)
            .head(1)
        )
        for well, group in top1.groupby("well", sort=False):
            rows.append(
                {
                    "score_name": score_name,
                    "well": str(well),
                    "anchors": int(len(group)),
                    "top1_within10_rate": float((group["abs_tvt_offset"] <= 10.0).mean()),
                    "top1_abs_offset_mean": float(group["abs_tvt_offset"].mean()),
                    "top1_abs_offset_p90": float(group["abs_tvt_offset"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["score_name", "top1_within10_rate", "top1_abs_offset_mean"]
    )


def summarize_decision(
    pair_metrics: pd.DataFrame,
    ranks: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = get_nested(config, "decision_thresholds") or {}
    min_auc_margin = float(thresholds.get("min_auc_margin_vs_shuffled", 0.02))
    min_top1_margin = float(thresholds.get("min_top1_margin_vs_no_gr", 0.03))
    metric_by_name = {str(row["score_name"]): row.to_dict() for _, row in pair_metrics.iterrows()}
    rank_top1 = ranks[ranks["topk"].eq(1)]
    rank_by_name = {str(row["score_name"]): row.to_dict() for _, row in rank_top1.iterrows()}
    real_auc = metric_by_name.get("real_gr_logistic", {}).get("auc")
    shuf_auc = metric_by_name.get("shuffled_gr_logistic", {}).get("auc")
    no_gr_top1 = rank_by_name.get("no_gr_logistic", {}).get("top1_within10_rate")
    real_top1 = rank_by_name.get("real_gr_logistic", {}).get("top1_within10_rate")
    auc_margin = None if real_auc is None or shuf_auc is None else float(real_auc - shuf_auc)
    top1_margin = (
        None if real_top1 is None or no_gr_top1 is None else float(real_top1 - no_gr_top1)
    )
    supported = (
        auc_margin is not None
        and auc_margin >= min_auc_margin
        and top1_margin is not None
        and top1_margin >= min_top1_margin
    )
    return {
        "min_auc_margin_vs_shuffled": min_auc_margin,
        "min_top1_margin_vs_no_gr": min_top1_margin,
        "real_auc": to_jsonable(real_auc),
        "shuffled_auc": to_jsonable(shuf_auc),
        "auc_margin_vs_shuffled": to_jsonable(auc_margin),
        "real_top1_within10": to_jsonable(real_top1),
        "no_gr_top1_within10": to_jsonable(no_gr_top1),
        "top1_margin_vs_no_gr": to_jsonable(top1_margin),
        "real_gr_beats_negative_controls": bool(supported),
        "recommendation": (
            "learned_gr_window_matcher_supported_for_followup_feature_generation"
            if supported
            else "diagnostic_only_until_real_gr_beats_shuffled_and_no_gr_controls"
        ),
    }


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": OUTPUT_PREFIX,
            "feature_index": np.arange(len(columns), dtype=np.int32),
            "feature": columns,
        }
    ).to_csv(path, index=False)


# %% [markdown]
# ## 5. Setup and input checks

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

pair_cfg = get_nested(config, "pair_dataset") or {}
model_cfg = get_nested(config, "model") or {}
runtime_cfg = get_nested(config, "runtime.kaggle") or {}
seed = int(get_nested(config, "reproducibility.seed") or 42)

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Status:", get_nested(config, "experiment.status"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Train dir:", paths.train_data_dir)
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_cfg.get("enable_gpu"))
print("Max wells:", pair_cfg.get("max_wells"))
print("Rows per well:", pair_cfg.get("rows_per_well"))
print("Negative offsets:", pair_cfg.get("negative_offsets_ft"))
print("Hard decoy scan offsets:", pair_cfg.get("hard_decoy_scan_offsets_ft"))
print("Estimator:", model_cfg.get("estimator"))
print("LightGBM configs: 0 folds: 0 boosters: 0 control retraining: none")

display(
    {
        "expected_train_artifacts": get_nested(config, "audit.expected_train_artifacts"),
        "leakage_policy": get_nested(config, "validation.leakage_policy"),
    }
)

# %% [markdown]
# ## 6. Run 1-fold row-cap smoke

# %%
start_time = time.time()
pair_frame, well_status = build_pair_dataset(paths, config)
split_cfg = get_nested(config, "validation") or {}
train_mask, valid_mask, split_meta = split_train_valid(
    pair_frame,
    n_folds=int(split_cfg.get("n_folds", 5)),
    fold_index=int(split_cfg.get("fold_index", 0)),
)
valid_predictions, importance = fit_models(
    pair_frame,
    train_mask=train_mask,
    valid_mask=valid_mask,
    seed=seed,
)
pair_metrics = pair_score_metrics(valid_predictions)
topk_values = [int(v) for v in get_nested(config, "audit.topk_values") or [1, 2, 3]]
ranks = rank_metrics(valid_predictions, topk_values)
by_well = by_well_rank_metrics(valid_predictions)
decision = summarize_decision(pair_metrics, ranks, config)

display(split_meta)
display(pair_metrics)
display(ranks.head(30))
display(decision)

# %% [markdown]
# ## 7. Metrics and generated artifacts

# %%
artifacts = paths.artifacts_dir
pair_path = artifacts / f"{OUTPUT_PREFIX}_pair_features.csv.gz"
valid_path = artifacts / f"{OUTPUT_PREFIX}_validation_predictions.csv.gz"
pair_metrics_path = artifacts / f"{OUTPUT_PREFIX}_pair_metrics.csv"
rank_metrics_path = artifacts / f"{OUTPUT_PREFIX}_rank_metrics.csv"
by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well_rank_metrics.csv"
well_status_path = artifacts / f"{OUTPUT_PREFIX}_well_status.csv"
importance_path = artifacts / f"{OUTPUT_PREFIX}_logistic_coefficients.csv"
schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

pair_frame.to_csv(pair_path, index=False, compression="gzip")
valid_predictions.to_csv(valid_path, index=False, compression="gzip")
pair_metrics.to_csv(pair_metrics_path, index=False)
ranks.to_csv(rank_metrics_path, index=False)
by_well.to_csv(by_well_path, index=False)
well_status.to_csv(well_status_path, index=False)
importance.sort_values(["model", "abs_coefficient"], ascending=[True, False]).to_csv(
    importance_path,
    index=False,
)
all_feature_columns = sorted(
    set(CONTEXT_FEATURES)
    | {f"real_{name}" for name in GR_FEATURE_NAMES}
    | {f"shuf_{name}" for name in GR_FEATURE_NAMES}
)
write_feature_schema(schema_path, all_feature_columns)

runtime_seconds = time.time() - start_time
summary = {
    "experiment": OUTPUT_PREFIX,
    "created_at": datetime.now(UTC).isoformat(),
    "runtime_seconds": runtime_seconds,
    "rows": int(len(pair_frame)),
    "anchors": int(pair_frame["anchor_id"].nunique()),
    "wells": int(pair_frame["well"].nunique()),
    "split": split_meta,
    "pair_dataset": {
        "positive_rate": float(pair_frame["label_within_10ft"].mean()),
        "pair_kind_counts": pair_frame["pair_kind"].value_counts().to_dict(),
        "well_status_counts": well_status["status"].value_counts().to_dict(),
    },
    "pair_metrics": {
        str(row["score_name"]): to_jsonable(row.to_dict()) for _, row in pair_metrics.iterrows()
    },
    "rank_top1": {
        str(row["score_name"]): to_jsonable(row.to_dict())
        for _, row in ranks[ranks["topk"].eq(1)].iterrows()
    },
    "decision": decision,
    "artifacts": {
        "pair_features": str(pair_path),
        "validation_predictions": str(valid_path),
        "pair_metrics": str(pair_metrics_path),
        "rank_metrics": str(rank_metrics_path),
        "by_well_rank_metrics": str(by_well_path),
        "well_status": str(well_status_path),
        "logistic_coefficients": str(importance_path),
        "feature_schema": str(schema_path),
        "summary": str(summary_path),
    },
    "sha256": {
        "pair_features_raw": sha256_path(pair_path),
        "pair_features_decompressed": sha256_path(pair_path, decompressed=True),
        "validation_predictions_raw": sha256_path(valid_path),
        "validation_predictions_decompressed": sha256_path(valid_path, decompressed=True),
        "pair_metrics": sha256_path(pair_metrics_path),
        "rank_metrics": sha256_path(rank_metrics_path),
        "by_well_rank_metrics": sha256_path(by_well_path),
        "well_status": sha256_path(well_status_path),
        "logistic_coefficients": sha256_path(importance_path),
        "feature_schema": sha256_path(schema_path),
    },
}
summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")

metrics_json = {
    "experiment": OUTPUT_PREFIX,
    "status": "completed_train_side_smoke"
    if decision["real_gr_beats_negative_controls"]
    else "completed_train_side_diagnostic_no_submit",
    "metric": "pair_auc",
    "cv": decision["real_auc"],
    "public_lb": None,
    "private_lb": None,
    "rows": int(len(pair_frame)),
    "wells": int(pair_frame["well"].nunique()),
    "anchors": int(pair_frame["anchor_id"].nunique()),
    "decision": decision,
    "summary_path": str(summary_path),
    "notes": (
        "Known-prefix supervised GR window matcher smoke completed. "
        "This is pair-level diagnostic output only; no direct TVT replacement, "
        "inference port, or submission is selected."
    ),
}
paths.metrics_path.write_text(
    json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
)

print("runtime_seconds:", runtime_seconds)
print("summary:", summary_path)
print("metrics_json:", paths.metrics_path)
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
