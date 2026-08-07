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
# # exp202 heatmap mdn candidate generator probe train
#
# Train-side GPU diagnostic for turning the discussion-699853 CNN/SDF/MTP
# heatmap model into a PF/Beam candidate generator. It trains a fold-safe
# K-path heatmap head, saves topK TVT candidates, and measures whether adding
# them to the existing PF/Beam candidate union improves oracle headroom. It
# does not create an inference branch or submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and reproducibility helpers
# 3. Run spec and fold helpers
# 4. Well loading and fold-safe sample index helpers
# 5. Heatmap dataset
# 6. CNN/MTP model and training helpers
# 7. Setup and input checks
# 8. Run GPU specs
# 9. Candidate-union readout, metrics, SHA, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## 2. Runtime and reproducibility helpers

# %%
OUTPUT_PREFIX = EXPERIMENT_NAME
TOPK_VALUES = [1, 3, 5, 10]
BASE_CHANNEL_SCHEMA = [
    ("typewell_gr_heatmap", "Typewell GR sampled at the target-free TVT grid."),
    ("horizontal_gr_heatmap", "Horizontal GR sampled at the horizontal row window."),
    ("typewell_minus_horizontal_gr", "Pairwise GR difference heatmap."),
    (
        "tvt_history_sdf_from_observed_tvt_input_prefix",
        "Target-free SDF history: grid_tvt - observed TVT_input where prefix is known.",
    ),
    ("observed_tvt_input_mask", "1 where TVT_input is observed in the horizontal window."),
]
GEOMETRY_CHANNEL_SCHEMA = [
    ("sin_dmd_dz_tangent", "sin(arctan(dMD/dZ)) sampled on horizontal rows."),
    ("cos_dmd_dz_tangent", "cos(arctan(dMD/dZ)) sampled on horizontal rows."),
    ("sin_dx_dy_direction", "sin(horizontal XY direction angle) sampled on horizontal rows."),
    ("cos_dx_dy_direction", "cos(horizontal XY direction angle) sampled on horizontal rows."),
    ("prefix_distance_prior", "Clipped MD distance from observed prefix anchor."),
    ("row_location_prior", "Clipped row distance from observed prefix anchor."),
]


def channel_schema_for(channel_set: str) -> list[tuple[str, str]]:
    if channel_set == "base":
        return list(BASE_CHANNEL_SCHEMA)
    if channel_set == "geometry":
        return list(BASE_CHANNEL_SCHEMA) + list(GEOMETRY_CHANNEL_SCHEMA)
    raise ValueError(f"Unknown channel_set: {channel_set}")


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
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
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


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_cuda_device(config: dict[str, Any]) -> torch.device:
    require_cuda = bool(get_nested(config, "model.training.require_cuda"))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires a Kaggle GPU runtime. "
            "CPU fallback is disabled by config."
        )
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        min_major = int(get_nested(config, "model.training.min_cuda_capability_major") or 0)
        if capability[0] < min_major:
            raise RuntimeError(
                "The allocated GPU is incompatible with this Kaggle PyTorch build: "
                f"device={torch.cuda.get_device_name(0)!r}, capability={capability}, "
                f"required_major>={min_major}. Re-push with machine_shape=NvidiaTeslaT4."
            )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finite_float_array(series: pd.Series | None, fallback: float = 0.0, length: int | None = None) -> np.ndarray:
    if series is None:
        if length is None:
            raise ValueError("length is required when series is None")
        return np.full(length, fallback, dtype=np.float32)
    values = pd.to_numeric(series, errors="coerce")
    values = values.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return values.to_numpy(np.float32)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=np.float32)
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        std = float(np.std(finite))
        scale = std if std > 1e-6 else 1.0
    return np.clip((values - median) / scale, -8.0, 8.0).astype(np.float32)


def fill_and_scale_gr(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(series, errors="coerce")
    missing = raw.isna().to_numpy(np.float32)
    filled = raw.interpolate(limit_direction="both").ffill().bfill()
    fallback = float(filled.dropna().median()) if filled.notna().any() else 0.0
    filled = filled.fillna(fallback).to_numpy(np.float32)
    return robust_zscore(filled), missing


def safe_angle_sin_cos(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.arctan2(numerator.astype(np.float32), denominator.astype(np.float32))
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def nearest_grid_indices(grid_tvt: np.ndarray, truth_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(grid_tvt, truth_tvt, side="left")
    left = np.clip(positions - 1, 0, len(grid_tvt) - 1)
    right = np.clip(positions, 0, len(grid_tvt) - 1)
    choose_right = np.abs(grid_tvt[right] - truth_tvt) < np.abs(grid_tvt[left] - truth_tvt)
    index = np.where(choose_right, right, left).astype(np.int64)
    distance = np.abs(grid_tvt[index] - truth_tvt).astype(np.float32)
    return index, distance


def gzip_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def clean_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


# %% [markdown]
# ## 3. Run spec and fold helpers

# %%
@dataclass(frozen=True)
class RunSpec:
    name: str
    variant: str
    channel_set: str
    fold_indices: tuple[int, ...]
    horizontal_window_rows: int
    typewell_window_bins: int
    tvt_grid_half_width_ft: float
    history_scale_ft: float


@dataclass(frozen=True)
class RunPlanItem:
    run_spec: str
    variant: str
    channel_set: str
    fold_index: int
    horizontal_window_rows: int
    typewell_window_bins: int
    tvt_grid_half_width_ft: float
    history_scale_ft: float


def resolve_run_specs(config: dict[str, Any]) -> list[RunSpec]:
    training = get_nested(config, "model.training") or {}
    raw_specs = get_nested(config, "model.active_run_specs") or []
    if not raw_specs:
        raise ValueError("model.active_run_specs must not be empty")

    default_fold_indices = tuple(int(v) for v in (get_nested(config, "validation.active_fold_indices") or [0]))
    resolved: list[RunSpec] = []
    seen_names: set[str] = set()
    for raw_spec in raw_specs:
        spec = dict(raw_spec)
        name = clean_name(str(spec["name"]))
        if name in seen_names:
            raise ValueError(f"Duplicate run spec name: {name}")
        seen_names.add(name)
        fold_indices = tuple(int(v) for v in spec.get("fold_indices", default_fold_indices))
        if not fold_indices:
            raise ValueError(f"Run spec {name} has no fold indices")
        variant = str(spec.get("variant", "real_gr"))
        if variant not in {"real_gr", "shuffled_gr", "no_gr"}:
            raise ValueError(f"Unexpected variant for {name}: {variant}")
        channel_set = str(spec.get("channel_set", "base"))
        channel_schema_for(channel_set)
        resolved.append(
            RunSpec(
                name=name,
                variant=variant,
                channel_set=channel_set,
                fold_indices=fold_indices,
                horizontal_window_rows=int(
                    spec.get(
                        "horizontal_window_rows",
                        training.get("default_horizontal_window_rows", 128),
                    )
                ),
                typewell_window_bins=int(
                    spec.get(
                        "typewell_window_bins",
                        training.get("default_typewell_window_bins", 64),
                    )
                ),
                tvt_grid_half_width_ft=float(
                    spec.get(
                        "tvt_grid_half_width_ft",
                        training.get("default_tvt_grid_half_width_ft", 192.0),
                    )
                ),
                history_scale_ft=float(
                    spec.get(
                        "history_scale_ft",
                        training.get("default_history_scale_ft", 200.0),
                    )
                ),
            )
        )
    return resolved


def expand_run_plan(run_specs: list[RunSpec]) -> list[RunPlanItem]:
    plan: list[RunPlanItem] = []
    for spec in run_specs:
        for fold_index in spec.fold_indices:
            plan.append(
                RunPlanItem(
                    run_spec=spec.name,
                    variant=spec.variant,
                    channel_set=spec.channel_set,
                    fold_index=int(fold_index),
                    horizontal_window_rows=spec.horizontal_window_rows,
                    typewell_window_bins=spec.typewell_window_bins,
                    tvt_grid_half_width_ft=spec.tvt_grid_half_width_ft,
                    history_scale_ft=spec.history_scale_ft,
                )
            )
    return plan


def split_wells(
    wells: list[str],
    config: dict[str, Any],
    *,
    fold_index: int,
) -> tuple[list[str], list[str]]:
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    groups = np.asarray(wells)
    dummy_x = np.zeros((len(wells), 1), dtype=np.float32)
    dummy_y = np.zeros(len(wells), dtype=np.float32)
    splits = list(GroupKFold(n_splits=n_folds).split(dummy_x, dummy_y, groups=groups))
    train_idx, valid_idx = splits[int(fold_index)]
    train_wells = [wells[index] for index in train_idx]
    valid_wells = [wells[index] for index in valid_idx]

    training = get_nested(config, "model.training") or {}
    max_train_wells = training.get("max_train_wells")
    max_valid_wells = training.get("max_valid_wells")
    if max_train_wells is not None:
        train_wells = train_wells[: int(max_train_wells)]
    if max_valid_wells is not None:
        valid_wells = valid_wells[: int(max_valid_wells)]
    return train_wells, valid_wells


def run_plan_dataframe(run_plan: list[RunPlanItem]) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in run_plan])


# %% [markdown]
# ## 4. Well loading and fold-safe sample index helpers

# %%
@dataclass(frozen=True)
class WellArrays:
    well: str
    horizontal_rows: int
    typewell_rows: int
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    horizontal_gr: np.ndarray
    horizontal_gr_missing: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    typewell_gr_shuffled: np.ndarray
    sin_dmd_dz: np.ndarray
    cos_dmd_dz: np.ndarray
    sin_dx_dy: np.ndarray
    cos_dx_dy: np.ndarray
    prefix_end: int
    last_known_tvt: float
    last_known_z: float
    last_known_md: float


@dataclass(frozen=True)
class WindowSample:
    sample_id: int
    id: str
    run_spec: str
    fold_index: int
    variant: str
    channel_set: str
    split: str
    well: str
    row_center: int
    prefix_end: int
    horizontal_window_rows: int
    typewell_window_bins: int
    tvt_grid_half_width_ft: float
    history_scale_ft: float
    last_known_tvt: float
    prior_center_tvt: float
    true_center_tvt: float
    md_since_prefix: float
    z_since_prefix: float
    center_target_in_grid: bool
    label_fraction: float


@dataclass
class CandidatePathOutput:
    sample_id: np.ndarray
    mode_index: np.ndarray
    center_bin: np.ndarray
    center_tvt: np.ndarray
    score: np.ndarray
    pred_tvt_path: np.ndarray
    pred_bin_path: np.ndarray
    true_tvt_path: np.ndarray
    tvt_input_path: np.ndarray
    md_path: np.ndarray
    z_path: np.ndarray
    horizontal_row_index: np.ndarray
    horizontal_offsets: np.ndarray

    @classmethod
    def empty(
        cls,
        *,
        topk: int,
        horizon: int,
        horizontal_offsets: np.ndarray | None = None,
    ) -> "CandidatePathOutput":
        offsets = (
            np.asarray(horizontal_offsets, dtype=np.int32)
            if horizontal_offsets is not None
            else np.empty((horizon,), dtype=np.int32)
        )
        return cls(
            sample_id=np.empty((0,), dtype=np.int64),
            mode_index=np.empty((0, topk), dtype=np.int16),
            center_bin=np.empty((0, topk), dtype=np.int16),
            center_tvt=np.empty((0, topk), dtype=np.float32),
            score=np.empty((0, topk), dtype=np.float32),
            pred_tvt_path=np.empty((0, topk, horizon), dtype=np.float32),
            pred_bin_path=np.empty((0, topk, horizon), dtype=np.int16),
            true_tvt_path=np.empty((0, horizon), dtype=np.float32),
            tvt_input_path=np.empty((0, horizon), dtype=np.float32),
            md_path=np.empty((0, horizon), dtype=np.float32),
            z_path=np.empty((0, horizon), dtype=np.float32),
            horizontal_row_index=np.empty((0, horizon), dtype=np.int32),
            horizontal_offsets=offsets,
        )


def list_train_wells(train_dir: Path, max_wells: int | None) -> list[str]:
    wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in train_dir.glob("*__horizontal_well.csv")
    )
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    return wells


def read_well_arrays(well: str, train_dir: Path, seed: int) -> WellArrays | None:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        return None

    h = pd.read_csv(horizontal_path)
    t = pd.read_csv(typewell_path)
    required_h = {"MD", "Z", "TVT", "TVT_input", "GR"}
    required_t = {"TVT", "GR"}
    if not required_h.issubset(h.columns) or not required_t.issubset(t.columns):
        return None

    tvt_input_raw = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input_raw))
    if len(known) < 16:
        return None
    prefix_end = int(known[-1])
    if prefix_end >= len(h) - 16:
        return None

    t = t.sort_values("TVT").reset_index(drop=True)
    typewell_tvt = finite_float_array(t["TVT"])
    typewell_gr, _ = fill_and_scale_gr(t["GR"])
    if len(typewell_tvt) < 32:
        return None

    roll = stable_int(EXPERIMENT_NAME, "shuffle-gr", well, str(seed), modulo=len(typewell_gr))
    typewell_gr_shuffled = np.roll(typewell_gr, int(roll)).astype(np.float32)

    horizontal_gr, horizontal_gr_missing = fill_and_scale_gr(h["GR"])
    md = finite_float_array(h["MD"])
    x = finite_float_array(h["X"] if "X" in h.columns else None, length=len(h))
    y = finite_float_array(h["Y"] if "Y" in h.columns else None, length=len(h))
    z = finite_float_array(h["Z"])
    tvt = finite_float_array(h["TVT"])
    tvt_input = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(np.float32)

    dmd = np.gradient(md).astype(np.float32)
    dz = np.gradient(z).astype(np.float32)
    dx = np.gradient(x).astype(np.float32)
    dy = np.gradient(y).astype(np.float32)
    sin_dmd_dz, cos_dmd_dz = safe_angle_sin_cos(dmd, dz)
    sin_dx_dy, cos_dx_dy = safe_angle_sin_cos(dy, dx)

    return WellArrays(
        well=well,
        horizontal_rows=len(h),
        typewell_rows=len(t),
        md=md,
        x=x,
        y=y,
        z=z,
        tvt=tvt,
        tvt_input=tvt_input,
        horizontal_gr=horizontal_gr,
        horizontal_gr_missing=horizontal_gr_missing,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        typewell_gr_shuffled=typewell_gr_shuffled,
        sin_dmd_dz=sin_dmd_dz,
        cos_dmd_dz=cos_dmd_dz,
        sin_dx_dy=sin_dx_dy,
        cos_dx_dy=cos_dx_dy,
        prefix_end=prefix_end,
        last_known_tvt=float(tvt_input[prefix_end]),
        last_known_z=float(z[prefix_end]),
        last_known_md=float(md[prefix_end]),
    )


def sample_rows_for_well(arrays: WellArrays, samples_per_well: int, max_tail_rows: int) -> np.ndarray:
    tail_start = arrays.prefix_end + 1
    tail_stop = min(arrays.horizontal_rows - 1, arrays.prefix_end + int(max_tail_rows))
    if tail_stop <= tail_start:
        return np.array([], dtype=np.int32)
    count = min(int(samples_per_well), int(tail_stop - tail_start + 1))
    rows = np.linspace(tail_start, tail_stop, count)
    return np.unique(np.rint(rows).astype(np.int32))


def sample_label_status(
    arrays: WellArrays,
    row_center: int,
    horizontal_offsets: np.ndarray,
    grid_offsets_tvt: np.ndarray,
    target_tolerance: float,
) -> tuple[float, float, bool, float]:
    prior_center = arrays.last_known_tvt - (float(arrays.z[row_center]) - arrays.last_known_z)
    grid_tvt = prior_center + grid_offsets_tvt
    h_idx = np.clip(row_center + horizontal_offsets, 0, arrays.horizontal_rows - 1)
    _, target_distance = nearest_grid_indices(grid_tvt, arrays.tvt[h_idx])
    center_position = int(np.flatnonzero(horizontal_offsets == 0)[0])
    center_distance = float(target_distance[center_position])
    center_target_in_grid = bool(center_distance <= float(target_tolerance))
    label_fraction = float(np.mean(target_distance <= float(target_tolerance)))
    true_center_tvt = float(arrays.tvt[row_center])
    return prior_center, true_center_tvt, center_target_in_grid, label_fraction


def build_sample_index_for_plan(
    *,
    arrays_by_well: dict[str, WellArrays],
    train_wells: list[str],
    valid_wells: list[str],
    config: dict[str, Any],
    plan_item: RunPlanItem,
    sample_id_start: int,
) -> pd.DataFrame:
    training = get_nested(config, "model.training") or {}
    horizontal_window = int(plan_item.horizontal_window_rows)
    horizontal_offsets = np.arange(-(horizontal_window // 2), horizontal_window // 2, dtype=np.int32)
    grid_bins = int(plan_item.typewell_window_bins)
    grid_half_width = float(plan_item.tvt_grid_half_width_ft)
    grid_offsets_tvt = np.linspace(-grid_half_width, grid_half_width, grid_bins).astype(np.float32)
    target_tolerance = float(training.get("center_target_tolerance_ft", 10.0))
    min_label_fraction = float(training.get("min_label_fraction", 0.35))
    max_tail_rows = int(training.get("max_tail_rows", 2048))

    sample_rows: list[dict[str, Any]] = []
    sample_id = int(sample_id_start)
    split_specs = [
        ("train", train_wells, int(training.get("train_samples_per_well", 20)), True),
        ("valid", valid_wells, int(training.get("valid_samples_per_well", 14)), False),
    ]
    for split, wells, samples_per_well, filter_for_training in split_specs:
        for well in wells:
            arrays = arrays_by_well[well]
            for row_center in sample_rows_for_well(arrays, samples_per_well, max_tail_rows):
                prior_center, true_center, center_in_grid, label_fraction = sample_label_status(
                    arrays,
                    int(row_center),
                    horizontal_offsets,
                    grid_offsets_tvt,
                    target_tolerance,
                )
                if filter_for_training and (
                    not center_in_grid or label_fraction < min_label_fraction
                ):
                    continue
                item = WindowSample(
                    sample_id=sample_id,
                    id=f"{well}_{int(row_center)}",
                    run_spec=plan_item.run_spec,
                    fold_index=int(plan_item.fold_index),
                    variant=plan_item.variant,
                    channel_set=plan_item.channel_set,
                    split=split,
                    well=well,
                    row_center=int(row_center),
                    prefix_end=arrays.prefix_end,
                    horizontal_window_rows=horizontal_window,
                    typewell_window_bins=grid_bins,
                    tvt_grid_half_width_ft=grid_half_width,
                    history_scale_ft=float(plan_item.history_scale_ft),
                    last_known_tvt=arrays.last_known_tvt,
                    prior_center_tvt=float(prior_center),
                    true_center_tvt=float(true_center),
                    md_since_prefix=float(arrays.md[row_center] - arrays.last_known_md),
                    z_since_prefix=float(arrays.z[row_center] - arrays.last_known_z),
                    center_target_in_grid=center_in_grid,
                    label_fraction=float(label_fraction),
                )
                sample_rows.append(asdict(item))
                sample_id += 1
    return pd.DataFrame(sample_rows)


# %% [markdown]
# ## 5. Heatmap dataset

# %%
class HeatmapWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        sample_index: pd.DataFrame,
        arrays_by_well: dict[str, WellArrays],
        plan_item: RunPlanItem,
        config: dict[str, Any],
        split: str,
    ) -> None:
        mask = (
            (sample_index["run_spec"] == plan_item.run_spec)
            & (sample_index["fold_index"] == int(plan_item.fold_index))
            & (sample_index["split"] == split)
        )
        self.sample_index = sample_index.loc[mask].reset_index(drop=True)
        self.arrays_by_well = arrays_by_well
        self.plan_item = plan_item
        self.variant = plan_item.variant
        self.channel_set = plan_item.channel_set
        horizontal_window = int(plan_item.horizontal_window_rows)
        self.horizontal_offsets = np.arange(
            -(horizontal_window // 2),
            horizontal_window // 2,
            dtype=np.int32,
        )
        grid_bins = int(plan_item.typewell_window_bins)
        grid_half_width = float(plan_item.tvt_grid_half_width_ft)
        self.grid_offsets_tvt = np.linspace(-grid_half_width, grid_half_width, grid_bins).astype(
            np.float32
        )
        self.history_scale_ft = float(plan_item.history_scale_ft)
        training = get_nested(config, "model.training") or {}
        self.max_tail_rows = int(training.get("max_tail_rows", 2048))
        self.target_tolerance = float(training.get("center_target_tolerance_ft", 10.0))
        self.center_position = int(np.flatnonzero(self.horizontal_offsets == 0)[0])

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.sample_index.iloc[index]
        arrays = self.arrays_by_well[str(row["well"])]
        row_center = int(row["row_center"])
        h_idx = np.clip(row_center + self.horizontal_offsets, 0, arrays.horizontal_rows - 1)
        grid_tvt = float(row["prior_center_tvt"]) + self.grid_offsets_tvt

        t_gr_source = (
            arrays.typewell_gr_shuffled if self.variant == "shuffled_gr" else arrays.typewell_gr
        )
        t_gr = np.interp(grid_tvt, arrays.typewell_tvt, t_gr_source).astype(np.float32)
        h_gr = arrays.horizontal_gr[h_idx].astype(np.float32)
        if self.variant == "no_gr":
            t_gr = np.zeros_like(t_gr, dtype=np.float32)
            h_gr = np.zeros_like(h_gr, dtype=np.float32)

        t_heatmap = np.broadcast_to(t_gr.reshape(1, -1), (len(h_idx), len(grid_tvt)))
        h_heatmap = np.broadcast_to(h_gr.reshape(-1, 1), (len(h_idx), len(grid_tvt)))
        diff = t_heatmap - h_heatmap

        observed_tvt = arrays.tvt_input[h_idx]
        mask = np.isfinite(observed_tvt).astype(np.float32)
        observed_safe = np.where(np.isfinite(observed_tvt), observed_tvt, 0.0).astype(np.float32)
        history = (grid_tvt.reshape(1, -1) - observed_safe.reshape(-1, 1)) / self.history_scale_ft
        history = np.clip(history * mask.reshape(-1, 1), -8.0, 8.0)
        mask_heatmap = np.broadcast_to(mask.reshape(-1, 1), history.shape)

        channels = [
            t_heatmap,
            h_heatmap,
            diff,
            history,
            mask_heatmap,
        ]
        if self.channel_set == "geometry":
            prefix_distance = np.clip(
                (arrays.md[h_idx] - arrays.last_known_md) / max(float(self.history_scale_ft), 1.0),
                -4.0,
                8.0,
            ).astype(np.float32)
            row_location = np.clip(
                (h_idx.astype(np.float32) - float(arrays.prefix_end)) / max(float(self.max_tail_rows), 1.0),
                -1.0,
                2.0,
            ).astype(np.float32)
            geom_vectors = [
                arrays.sin_dmd_dz[h_idx],
                arrays.cos_dmd_dz[h_idx],
                arrays.sin_dx_dy[h_idx],
                arrays.cos_dx_dy[h_idx],
                prefix_distance,
                row_location,
            ]
            channels.extend(
                np.broadcast_to(vector.reshape(-1, 1), history.shape)
                for vector in geom_vectors
            )

        image = np.stack(channels, axis=0).astype(np.float32)

        target_idx, target_distance = nearest_grid_indices(grid_tvt, arrays.tvt[h_idx])
        target_mask = (target_distance <= self.target_tolerance).astype(np.float32)
        center_target_idx = int(target_idx[self.center_position])
        center_target_distance = float(target_distance[self.center_position])

        return {
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
            "image": torch.from_numpy(image),
            "target_idx": torch.from_numpy(target_idx.astype(np.int64)),
            "target_mask": torch.from_numpy(target_mask.astype(np.float32)),
            "grid_tvt": torch.from_numpy(grid_tvt.astype(np.float32)),
            "true_center_tvt": torch.tensor(float(row["true_center_tvt"]), dtype=torch.float32),
            "prior_center_tvt": torch.tensor(float(row["prior_center_tvt"]), dtype=torch.float32),
            "center_target_idx": torch.tensor(center_target_idx, dtype=torch.long),
            "center_target_distance": torch.tensor(center_target_distance, dtype=torch.float32),
            "center_target_in_grid": torch.tensor(
                bool(row["center_target_in_grid"]), dtype=torch.bool
            ),
            "horizontal_row_index": torch.from_numpy(h_idx.astype(np.int32)),
            "horizontal_md": torch.from_numpy(arrays.md[h_idx].astype(np.float32)),
            "horizontal_z": torch.from_numpy(arrays.z[h_idx].astype(np.float32)),
            "true_tvt_path": torch.from_numpy(arrays.tvt[h_idx].astype(np.float32)),
            "tvt_input_path": torch.from_numpy(arrays.tvt_input[h_idx].astype(np.float32)),
        }


# %% [markdown]
# ## 6. CNN/MTP model and training helpers

# %%
class HeatmapMTPNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        path_modes: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
        use_group_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(in_channels)
        padding = int(kernel_size) // 2
        for width in channels:
            width = int(width)
            layers.append(nn.Conv2d(current, width, kernel_size=kernel_size, padding=padding))
            if use_group_norm:
                groups = 8 if width % 8 == 0 else 1
                layers.append(nn.GroupNorm(groups, width))
            else:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            current = width
        self.backbone = nn.Sequential(*layers)
        self.path_head = nn.Conv2d(current, int(path_modes), kernel_size=1)
        self.mode_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(current, int(path_modes)),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.backbone(image)
        path_logits = self.path_head(feature)
        mode_logits = self.mode_head(feature)
        return path_logits, mode_logits


def make_model(config: dict[str, Any], *, in_channels: int) -> HeatmapMTPNet:
    arch = get_nested(config, "model.architecture") or {}
    return HeatmapMTPNet(
        in_channels=int(in_channels),
        path_modes=int(arch.get("path_modes", 10)),
        channels=[int(value) for value in arch.get("channels", [32, 64, 64])],
        kernel_size=int(arch.get("kernel_size", 3)),
        dropout=float(arch.get("dropout", 0.05)),
        use_group_norm=bool(arch.get("use_group_norm", True)),
    )


def closest_mode_loss(
    path_logits: torch.Tensor,
    mode_logits: torch.Tensor,
    target_idx: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, path_modes, h_size, t_size = path_logits.shape
    flat_logits = path_logits.reshape(batch_size * path_modes, h_size, t_size).permute(0, 2, 1)
    flat_target = target_idx[:, None, :].expand(batch_size, path_modes, h_size).reshape(
        batch_size * path_modes,
        h_size,
    )
    flat_mask = target_mask[:, None, :].expand(batch_size, path_modes, h_size).reshape(
        batch_size * path_modes,
        h_size,
    )
    ce = F.cross_entropy(flat_logits, flat_target, reduction="none")
    counts = flat_mask.sum(dim=1).clamp_min(1.0)
    per_mode_loss = (ce * flat_mask).sum(dim=1) / counts
    per_mode_loss = per_mode_loss.reshape(batch_size, path_modes)
    best_mode = per_mode_loss.argmin(dim=1)
    chosen_path_loss = per_mode_loss.gather(1, best_mode[:, None]).squeeze(1).mean()
    mode_loss = F.cross_entropy(mode_logits, best_mode)
    return chosen_path_loss + 0.2 * mode_loss, best_mode


def make_loader(
    dataset: HeatmapWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    plan_item: RunPlanItem,
    loss: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "run_spec": plan_item.run_spec,
        "variant": plan_item.variant,
        "channel_set": plan_item.channel_set,
        "fold_index": int(plan_item.fold_index),
        "horizontal_window_rows": int(plan_item.horizontal_window_rows),
        "typewell_window_bins": int(plan_item.typewell_window_bins),
        "tvt_grid_half_width_ft": float(plan_item.tvt_grid_half_width_ft),
        "history_scale_ft": float(plan_item.history_scale_ft),
        "loss": float(loss),
        "samples": int(len(predictions)),
        "target_in_grid_rate": float(predictions["target_in_grid"].mean())
        if len(predictions)
        else np.nan,
    }
    for topk in TOPK_VALUES:
        within = predictions[f"top{topk}_within10"].astype(float)
        best_error = predictions[f"top{topk}_best_abs_error"].astype(float)
        metrics[f"top{topk}_within10_center_rate"] = float(within.mean())
        metrics[f"top{topk}_oracle_center_rmse"] = float(
            math.sqrt(float(np.nanmean(np.square(best_error))))
        )
        in_grid = predictions["target_in_grid"].astype(bool)
        metrics[f"top{topk}_within10_center_rate_in_grid"] = (
            float(within.loc[in_grid].mean()) if in_grid.any() else np.nan
        )
    metrics["top1_center_rmse"] = float(
        math.sqrt(float(np.nanmean(np.square(predictions["pred_top1_abs_error"].astype(float)))))
    )
    metrics["path_step_abs_mean_ft"] = float(
        np.nanmean(predictions["path_step_abs_mean_ft"].astype(float))
    )
    metrics["path_step_abs_p95_ft"] = float(
        np.nanpercentile(predictions["path_step_abs_mean_ft"].astype(float), 95)
    )
    return metrics


@torch.no_grad()
def evaluate_model(
    *,
    model: HeatmapMTPNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    plan_item: RunPlanItem,
    collect_paths: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, CandidatePathOutput]:
    model.eval()
    losses: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    center_position = loader.dataset.center_position  # type: ignore[attr-defined]
    horizontal_offsets = loader.dataset.horizontal_offsets.astype(np.int32)  # type: ignore[attr-defined]
    max_topk = max(TOPK_VALUES)
    horizon = int(len(horizontal_offsets))

    path_sample_ids: list[np.ndarray] = []
    path_mode_indices: list[np.ndarray] = []
    path_center_bins: list[np.ndarray] = []
    path_center_tvts: list[np.ndarray] = []
    path_scores: list[np.ndarray] = []
    pred_tvt_paths: list[np.ndarray] = []
    pred_bin_paths: list[np.ndarray] = []
    true_tvt_paths: list[np.ndarray] = []
    tvt_input_paths: list[np.ndarray] = []
    md_paths: list[np.ndarray] = []
    z_paths: list[np.ndarray] = []
    row_index_paths: list[np.ndarray] = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_idx = batch["target_idx"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        path_logits, mode_logits = model(image)
        loss, _ = closest_mode_loss(path_logits, mode_logits, target_idx, target_mask)
        losses.append(float(loss.detach().cpu()))

        mode_prob = torch.softmax(mode_logits, dim=1)
        center_logits = path_logits[:, :, center_position, :]
        center_prob = torch.softmax(center_logits, dim=2)
        center_score, center_idx = center_prob.max(dim=2)
        combined_score = mode_prob * center_score
        order = combined_score.argsort(dim=1, descending=True)
        full_path_idx = torch.softmax(path_logits, dim=3).argmax(dim=3)

        grid_tvt = batch["grid_tvt"].cpu().numpy()
        true_tvt = batch["true_center_tvt"].cpu().numpy()
        sample_ids = batch["sample_id"].cpu().numpy()
        target_in_grid = batch["center_target_in_grid"].cpu().numpy().astype(bool)
        center_idx_np = center_idx.cpu().numpy()
        order_np = order.cpu().numpy()
        score_np = combined_score.cpu().numpy()
        full_path_idx_np = full_path_idx.cpu().numpy()
        if collect_paths:
            batch_mode_index = np.full((len(sample_ids), max_topk), -1, dtype=np.int16)
            batch_center_bin = np.full((len(sample_ids), max_topk), -1, dtype=np.int16)
            batch_center_tvt = np.full((len(sample_ids), max_topk), np.nan, dtype=np.float32)
            batch_score = np.full((len(sample_ids), max_topk), np.nan, dtype=np.float32)
            batch_pred_tvt_path = np.full(
                (len(sample_ids), max_topk, horizon), np.nan, dtype=np.float32
            )
            batch_pred_bin_path = np.full(
                (len(sample_ids), max_topk, horizon), -1, dtype=np.int16
            )

        for row_index in range(len(sample_ids)):
            candidate_tvts: list[float] = []
            candidate_scores: list[float] = []
            seen_bins: set[int] = set()
            for mode_index in order_np[row_index].tolist():
                pred_idx = int(center_idx_np[row_index, mode_index])
                if pred_idx in seen_bins:
                    continue
                seen_bins.add(pred_idx)
                candidate_tvts.append(float(grid_tvt[row_index, pred_idx]))
                candidate_scores.append(float(score_np[row_index, mode_index]))
                rank_index = len(candidate_tvts) - 1
                if collect_paths:
                    path_bins = full_path_idx_np[row_index, mode_index, :].astype(np.int16)
                    batch_mode_index[row_index, rank_index] = int(mode_index)
                    batch_center_bin[row_index, rank_index] = int(pred_idx)
                    batch_center_tvt[row_index, rank_index] = float(grid_tvt[row_index, pred_idx])
                    batch_score[row_index, rank_index] = float(score_np[row_index, mode_index])
                    batch_pred_tvt_path[row_index, rank_index, :] = grid_tvt[row_index, path_bins]
                    batch_pred_bin_path[row_index, rank_index, :] = path_bins
                if len(candidate_tvts) >= max(TOPK_VALUES):
                    break
            while len(candidate_tvts) < max(TOPK_VALUES):
                candidate_tvts.append(float("nan"))
                candidate_scores.append(float("nan"))

            best_mode = int(order_np[row_index, 0])
            path_bins = full_path_idx_np[row_index, best_mode, :]
            pred_path_tvt = grid_tvt[row_index, path_bins]
            path_step = np.abs(np.diff(pred_path_tvt.astype(np.float32)))
            errors = np.abs(np.asarray(candidate_tvts, dtype=np.float32) - float(true_tvt[row_index]))
            score_values = np.asarray(candidate_scores, dtype=np.float32)
            finite_scores = score_values[np.isfinite(score_values)]
            score_sum = float(np.sum(finite_scores)) if len(finite_scores) else 0.0
            if score_sum > 0.0:
                score_prob = finite_scores / score_sum
                score_entropy = float(-np.sum(score_prob * np.log(np.maximum(score_prob, 1e-12))))
                score_top3_mass = float(np.sum(score_prob[: min(3, len(score_prob))]))
                score_top5_mass = float(np.sum(score_prob[: min(5, len(score_prob))]))
            else:
                score_entropy = float("nan")
                score_top3_mass = float("nan")
                score_top5_mass = float("nan")
            record: dict[str, Any] = {
                "run_spec": plan_item.run_spec,
                "variant": plan_item.variant,
                "channel_set": plan_item.channel_set,
                "fold_index": int(plan_item.fold_index),
                "horizontal_window_rows": int(plan_item.horizontal_window_rows),
                "typewell_window_bins": int(plan_item.typewell_window_bins),
                "sample_id": int(sample_ids[row_index]),
                "true_center_tvt": float(true_tvt[row_index]),
                "target_in_grid": bool(target_in_grid[row_index]),
                "best_mode": best_mode,
                "path_step_abs_mean_ft": float(np.nanmean(path_step)) if len(path_step) else 0.0,
                "path_step_abs_max_ft": float(np.nanmax(path_step)) if len(path_step) else 0.0,
                "score_entropy": score_entropy,
                "score_top3_mass": score_top3_mass,
                "score_top5_mass": score_top5_mass,
                "top1_top2_score_margin": float(score_values[0] - score_values[1])
                if len(score_values) > 1 and np.isfinite(score_values[0]) and np.isfinite(score_values[1])
                else np.nan,
                "top1_top3_score_margin": float(score_values[0] - score_values[2])
                if len(score_values) > 2 and np.isfinite(score_values[0]) and np.isfinite(score_values[2])
                else np.nan,
            }
            for rank, (pred_tvt, pred_score) in enumerate(
                zip(candidate_tvts, candidate_scores, strict=False),
                start=1,
            ):
                record[f"pred_top{rank}_tvt"] = pred_tvt
                record[f"pred_top{rank}_score"] = pred_score
                record[f"pred_top{rank}_abs_error"] = (
                    float(errors[rank - 1]) if np.isfinite(errors[rank - 1]) else np.nan
                )
            for topk in TOPK_VALUES:
                best_error = float(np.nanmin(errors[:topk]))
                record[f"top{topk}_best_abs_error"] = best_error
                record[f"top{topk}_within10"] = bool(best_error <= 10.0)
            prediction_rows.append(record)

        if collect_paths:
            path_sample_ids.append(sample_ids.astype(np.int64))
            path_mode_indices.append(batch_mode_index)
            path_center_bins.append(batch_center_bin)
            path_center_tvts.append(batch_center_tvt)
            path_scores.append(batch_score)
            pred_tvt_paths.append(batch_pred_tvt_path)
            pred_bin_paths.append(batch_pred_bin_path)
            true_tvt_paths.append(batch["true_tvt_path"].cpu().numpy().astype(np.float32))
            tvt_input_paths.append(batch["tvt_input_path"].cpu().numpy().astype(np.float32))
            md_paths.append(batch["horizontal_md"].cpu().numpy().astype(np.float32))
            z_paths.append(batch["horizontal_z"].cpu().numpy().astype(np.float32))
            row_index_paths.append(batch["horizontal_row_index"].cpu().numpy().astype(np.int32))

    predictions = pd.DataFrame(prediction_rows)
    mean_loss = float(np.mean(losses)) if losses else np.nan
    metrics = summarize_predictions(predictions, plan_item=plan_item, loss=mean_loss)
    if collect_paths and path_sample_ids:
        path_output = CandidatePathOutput(
            sample_id=np.concatenate(path_sample_ids, axis=0),
            mode_index=np.concatenate(path_mode_indices, axis=0),
            center_bin=np.concatenate(path_center_bins, axis=0),
            center_tvt=np.concatenate(path_center_tvts, axis=0),
            score=np.concatenate(path_scores, axis=0),
            pred_tvt_path=np.concatenate(pred_tvt_paths, axis=0),
            pred_bin_path=np.concatenate(pred_bin_paths, axis=0),
            true_tvt_path=np.concatenate(true_tvt_paths, axis=0),
            tvt_input_path=np.concatenate(tvt_input_paths, axis=0),
            md_path=np.concatenate(md_paths, axis=0),
            z_path=np.concatenate(z_paths, axis=0),
            horizontal_row_index=np.concatenate(row_index_paths, axis=0),
            horizontal_offsets=horizontal_offsets,
        )
    else:
        path_output = CandidatePathOutput.empty(
            topk=max_topk,
            horizon=horizon,
            horizontal_offsets=horizontal_offsets,
        )
    return metrics, predictions, path_output


def train_run_fold(
    *,
    plan_item: RunPlanItem,
    arrays_by_well: dict[str, WellArrays],
    sample_index: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, CandidatePathOutput, pd.DataFrame, Path]:
    seed = int(get_nested(config, "reproducibility.seed") or 42)
    training = get_nested(config, "model.training") or {}
    batch_size = int(get_nested(config, "runtime.batch_size") or 32)
    epochs = int(training.get("epochs", 4))

    train_dataset = HeatmapWindowDataset(
        sample_index=sample_index,
        arrays_by_well=arrays_by_well,
        plan_item=plan_item,
        config=config,
        split="train",
    )
    valid_dataset = HeatmapWindowDataset(
        sample_index=sample_index,
        arrays_by_well=arrays_by_well,
        plan_item=plan_item,
        config=config,
        split="valid",
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError(f"Empty dataset for {plan_item.run_spec} fold {plan_item.fold_index}")

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(training.get("dataloader_shuffle", True)),
        seed=stable_int(
            EXPERIMENT_NAME,
            plan_item.run_spec,
            str(plan_item.fold_index),
            "train-loader",
            str(seed),
            modulo=2**31 - 1,
        ),
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=stable_int(
            EXPERIMENT_NAME,
            plan_item.run_spec,
            str(plan_item.fold_index),
            "valid-loader",
            str(seed),
            modulo=2**31 - 1,
        ),
    )

    in_channels = len(channel_schema_for(plan_item.channel_set))
    model = make_model(config, in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    grad_clip = float(training.get("gradient_clip_norm", 1.0))

    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -np.inf
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target_idx = batch["target_idx"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            path_logits, mode_logits = model(image)
            loss, _ = closest_mode_loss(path_logits, mode_logits, target_idx, target_mask)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        valid_metrics, _, _ = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            plan_item=plan_item,
        )
        valid_top3 = float(valid_metrics["top3_within10_center_rate"])
        if valid_top3 > best_metric:
            best_metric = valid_top3
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history_rows.append(
            {
                "run_spec": plan_item.run_spec,
                "variant": plan_item.variant,
                "channel_set": plan_item.channel_set,
                "fold_index": int(plan_item.fold_index),
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                "valid_loss": valid_metrics["loss"],
                "valid_top1_within10_center_rate": valid_metrics[
                    "top1_within10_center_rate"
                ],
                "valid_top3_within10_center_rate": valid_metrics[
                    "top3_within10_center_rate"
                ],
                "valid_target_in_grid_rate": valid_metrics["target_in_grid_rate"],
                "elapsed_sec": float(time.time() - start_time),
            }
        )
        print(
            f"{plan_item.run_spec} fold={plan_item.fold_index} epoch {epoch}/{epochs}: "
            f"train_loss={history_rows[-1]['train_loss']:.4f} "
            f"valid_top3={valid_top3:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, predictions, path_output = evaluate_model(
        model=model,
        loader=valid_loader,
        device=device,
        plan_item=plan_item,
        collect_paths=True,
    )
    final_metrics["best_valid_top3_within10_center_rate"] = float(best_metric)
    final_metrics["train_samples"] = int(len(train_dataset))
    final_metrics["valid_samples"] = int(len(valid_dataset))
    final_metrics["epochs"] = epochs
    final_metrics["elapsed_sec"] = float(time.time() - start_time)

    model_stem = f"{OUTPUT_PREFIX}_{plan_item.run_spec}_fold{int(plan_item.fold_index)}"
    model_path = output_dir / f"{model_stem}_model.pt"
    torch.save(
        {
            "experiment": EXPERIMENT_NAME,
            "run_spec": asdict(plan_item),
            "state_dict": model.state_dict(),
            "config_model": get_nested(config, "model"),
            "metrics": final_metrics,
        },
        model_path,
    )
    return final_metrics, predictions, path_output, pd.DataFrame(history_rows), model_path


# %% [markdown]
# ## 7. Setup and input checks

# %%
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
seed = int(get_nested(config, "reproducibility.seed") or 42)
set_reproducibility(seed)
device = require_cuda_device(config)

print("Experiment:", EXPERIMENT_NAME)
print("Device:", device)
print("Torch:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("Train dir:", paths.train_data_dir)
print("Artifacts dir:", paths.artifacts_dir)

training_config = get_nested(config, "model.training") or {}
run_specs = resolve_run_specs(config)
run_plan = expand_run_plan(run_specs)
run_plan_df = run_plan_dataframe(run_plan)

print("Training config:", json.dumps(to_jsonable(training_config), indent=2, sort_keys=True))
print("Active run specs:", len(run_specs))
print("CNN models to train:", len(run_plan))
display(run_plan_df)

# %%
max_wells = training_config.get("max_wells")
all_wells = list_train_wells(paths.train_data_dir, int(max_wells) if max_wells is not None else None)
if len(all_wells) < int(get_nested(config, "validation.n_folds") or 5):
    raise RuntimeError("Not enough train wells for configured GroupKFold.")

arrays_by_well: dict[str, WellArrays] = {}
for well in all_wells:
    arrays = read_well_arrays(well, paths.train_data_dir, seed)
    if arrays is not None:
        arrays_by_well[well] = arrays

usable_wells = [well for well in all_wells if well in arrays_by_well]
if len(usable_wells) < int(get_nested(config, "validation.n_folds") or 5):
    raise RuntimeError("No enough usable wells after loading horizontal/typewell files.")

sample_frames: list[pd.DataFrame] = []
sample_id_start = 0
fold_well_rows: list[dict[str, Any]] = []
for plan_item in run_plan:
    train_wells, valid_wells = split_wells(usable_wells, config, fold_index=plan_item.fold_index)
    train_wells = [well for well in train_wells if well in arrays_by_well]
    valid_wells = [well for well in valid_wells if well in arrays_by_well]
    fold_well_rows.append(
        {
            "run_spec": plan_item.run_spec,
            "fold_index": int(plan_item.fold_index),
            "train_wells": len(train_wells),
            "valid_wells": len(valid_wells),
        }
    )
    frame = build_sample_index_for_plan(
        arrays_by_well=arrays_by_well,
        train_wells=train_wells,
        valid_wells=valid_wells,
        config=config,
        plan_item=plan_item,
        sample_id_start=sample_id_start,
    )
    if frame.empty:
        raise RuntimeError(f"Sample index is empty for {plan_item.run_spec} fold {plan_item.fold_index}.")
    sample_id_start = int(frame["sample_id"].max()) + 1
    sample_frames.append(frame)

sample_index = pd.concat(sample_frames, ignore_index=True)
sample_index_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_sample_index.csv.gz"
gzip_csv(sample_index, sample_index_path)

sample_overview = (
    sample_index.groupby(["run_spec", "fold_index", "split"], as_index=False)
    .agg(
        samples=("sample_id", "count"),
        wells=("well", "nunique"),
        target_in_grid_rate=("center_target_in_grid", "mean"),
        label_fraction_mean=("label_fraction", "mean"),
        md_since_prefix_mean=("md_since_prefix", "mean"),
    )
)
fold_well_df = pd.DataFrame(fold_well_rows)
display(sample_overview)
display(fold_well_df.head(20))
display(sample_index.head())

# %% [markdown]
# ## 8. Run GPU specs

# %%
metrics_rows: list[dict[str, Any]] = []
prediction_frames: list[pd.DataFrame] = []
path_outputs: list[CandidatePathOutput] = []
history_frames: list[pd.DataFrame] = []
model_manifest: dict[str, Any] = {
    "experiment": EXPERIMENT_NAME,
    "created_at": datetime.now(UTC).isoformat(),
    "device": str(device),
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "models": {},
}

for plan_item in run_plan:
    print(
        "=== Training "
        f"run_spec={plan_item.run_spec} fold={plan_item.fold_index} "
        f"variant={plan_item.variant} channel_set={plan_item.channel_set} ==="
    )
    metrics, predictions, path_output, history, model_path = train_run_fold(
        plan_item=plan_item,
        arrays_by_well=arrays_by_well,
        sample_index=sample_index,
        config=config,
        device=device,
        output_dir=paths.artifacts_dir,
    )
    metrics_rows.append(metrics)
    prediction_frames.append(predictions)
    path_outputs.append(path_output)
    history_frames.append(history)
    manifest_key = f"{plan_item.run_spec}/fold{int(plan_item.fold_index)}"
    model_manifest["models"][manifest_key] = {
        "path": str(model_path),
        "sha256": sha256_path(model_path),
        "bytes": model_path.stat().st_size,
        "run_spec": asdict(plan_item),
        "metrics": to_jsonable(metrics),
    }
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))

# %% [markdown]
# ## 9. Metrics, SHA, and generated artifacts

# %%
fold_metrics_df = pd.DataFrame(metrics_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)
history_df = pd.concat(history_frames, ignore_index=True)

sample_columns = [
    "sample_id",
    "id",
    "split",
    "well",
    "row_center",
    "prefix_end",
    "last_known_tvt",
    "prior_center_tvt",
    "true_center_tvt",
    "md_since_prefix",
    "z_since_prefix",
    "label_fraction",
]
predictions_df = predictions_df.merge(
    sample_index[sample_columns],
    on="sample_id",
    how="left",
    suffixes=("", "_sample"),
)

distance_buckets = get_nested(config, "audit.distance_buckets") or [
    [0, 250],
    [250, 500],
    [500, 1000],
    [1000, 2000],
    [2000, 1000000000],
]


def distance_bucket_label(value: float) -> str:
    abs_value = abs(float(value))
    for lower, upper in distance_buckets:
        lower_f = float(lower)
        upper_f = float(upper)
        if lower_f <= abs_value < upper_f:
            if upper_f >= 1e8:
                return f"{int(lower_f)}_plus"
            return f"{int(lower_f)}_{int(upper_f)}"
    return "unknown"


predictions_df["distance_bucket"] = predictions_df["md_since_prefix"].map(distance_bucket_label)


def stack_path_outputs(outputs: list[CandidatePathOutput]) -> CandidatePathOutput:
    non_empty = [output for output in outputs if len(output.sample_id) > 0]
    if not non_empty:
        topk = max(TOPK_VALUES)
        horizon = int(get_nested(config, "model.training.default_horizontal_window_rows") or 128)
        horizontal_offsets = np.arange(-(horizon // 2), horizon - horizon // 2, dtype=np.int32)
        return CandidatePathOutput.empty(
            topk=topk,
            horizon=horizon,
            horizontal_offsets=horizontal_offsets,
        )
    horizontal_offsets = non_empty[0].horizontal_offsets
    for output in non_empty[1:]:
        if output.pred_tvt_path.shape[1:] != non_empty[0].pred_tvt_path.shape[1:]:
            raise ValueError("Candidate path outputs have inconsistent topK/horizon shapes.")
        if not np.array_equal(output.horizontal_offsets, horizontal_offsets):
            raise ValueError("Candidate path outputs have inconsistent horizontal offsets.")
    return CandidatePathOutput(
        sample_id=np.concatenate([output.sample_id for output in non_empty], axis=0),
        mode_index=np.concatenate([output.mode_index for output in non_empty], axis=0),
        center_bin=np.concatenate([output.center_bin for output in non_empty], axis=0),
        center_tvt=np.concatenate([output.center_tvt for output in non_empty], axis=0),
        score=np.concatenate([output.score for output in non_empty], axis=0),
        pred_tvt_path=np.concatenate([output.pred_tvt_path for output in non_empty], axis=0),
        pred_bin_path=np.concatenate([output.pred_bin_path for output in non_empty], axis=0),
        true_tvt_path=np.concatenate([output.true_tvt_path for output in non_empty], axis=0),
        tvt_input_path=np.concatenate([output.tvt_input_path for output in non_empty], axis=0),
        md_path=np.concatenate([output.md_path for output in non_empty], axis=0),
        z_path=np.concatenate([output.z_path for output in non_empty], axis=0),
        horizontal_row_index=np.concatenate(
            [output.horizontal_row_index for output in non_empty], axis=0
        ),
        horizontal_offsets=horizontal_offsets,
    )


def path_step_mean(paths_array: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(paths_array.astype(np.float32), axis=2))
    valid = np.isfinite(diffs)
    counts = valid.sum(axis=2)
    sums = np.where(valid, diffs, 0.0).sum(axis=2)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float32), where=counts > 0)


def path_step_max(paths_array: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(paths_array.astype(np.float32), axis=2))
    valid = np.isfinite(diffs)
    result = np.full(diffs.shape[:2], np.nan, dtype=np.float32)
    if diffs.shape[2] == 0:
        return result
    masked = np.where(valid, diffs, -np.inf)
    max_values = masked.max(axis=2)
    has_valid = valid.any(axis=2)
    result[has_valid] = max_values[has_valid]
    return result


candidate_path_output = stack_path_outputs(path_outputs)
candidate_path_sample_df = pd.DataFrame(
    {
        "path_npz_sample_index": np.arange(len(candidate_path_output.sample_id), dtype=np.int64),
        "sample_id": candidate_path_output.sample_id.astype(np.int64),
    }
)
candidate_path_sample_df = candidate_path_sample_df.merge(
    predictions_df[
        [
            "sample_id",
            "id",
            "split",
            "well",
            "fold_index",
            "row_center",
            "prefix_end",
            "horizontal_window_rows",
            "typewell_window_bins",
            "last_known_tvt",
            "prior_center_tvt",
            "true_center_tvt",
            "md_since_prefix",
            "z_since_prefix",
            "distance_bucket",
            "score_entropy",
            "score_top3_mass",
            "score_top5_mass",
            "top1_top2_score_margin",
            "top1_top3_score_margin",
        ]
    ],
    on="sample_id",
    how="left",
)

path_count, path_topk, path_horizon = candidate_path_output.pred_tvt_path.shape
rank_index = np.tile(np.arange(1, path_topk + 1, dtype=np.int16), path_count)
sample_index_repeated = np.repeat(np.arange(path_count, dtype=np.int64), path_topk)
sample_id_repeated = np.repeat(candidate_path_output.sample_id.astype(np.int64), path_topk)
step_mean = path_step_mean(candidate_path_output.pred_tvt_path)
step_max = path_step_max(candidate_path_output.pred_tvt_path)
true_center = candidate_path_sample_df["true_center_tvt"].to_numpy(np.float32)
center_abs_error = np.abs(candidate_path_output.center_tvt - true_center[:, None])
candidate_path_rank_df = pd.DataFrame(
    {
        "path_npz_sample_index": sample_index_repeated,
        "sample_id": sample_id_repeated,
        "rank": rank_index.astype(np.int16),
        "mode_index": candidate_path_output.mode_index.reshape(-1).astype(np.int16),
        "center_bin": candidate_path_output.center_bin.reshape(-1).astype(np.int16),
        "center_pred_tvt": candidate_path_output.center_tvt.reshape(-1).astype(np.float32),
        "center_score": candidate_path_output.score.reshape(-1).astype(np.float32),
        "center_abs_error": center_abs_error.reshape(-1).astype(np.float32),
        "path_step_abs_mean_ft": step_mean.reshape(-1).astype(np.float32),
        "path_step_abs_max_ft": step_max.reshape(-1).astype(np.float32),
    }
)
candidate_path_rank_df = candidate_path_rank_df.merge(
    candidate_path_sample_df[
        [
            "path_npz_sample_index",
            "id",
            "well",
            "fold_index",
            "row_center",
            "distance_bucket",
        ]
    ],
    on="path_npz_sample_index",
    how="left",
)


def resolve_config_reference(config: dict[str, Any], value: Any) -> Any:
    if isinstance(value, str):
        nested = get_nested(config, value)
        if nested is not None:
            return nested
    return value


def find_optional_artifact(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    raw_path = Path(str(path_value))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(paths.root / raw_path)
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        filename = raw_path.name
        for candidate in sorted(input_root.glob(f"**/{filename}")):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    return None


def oracle_metrics(errors: np.ndarray, within_ft: float) -> dict[str, float]:
    finite = errors[np.isfinite(errors)]
    if len(finite) == 0:
        return {
            "oracle_rmse": float("nan"),
            "oracle_mae": float("nan"),
            "within_rate": float("nan"),
        }
    return {
        "oracle_rmse": float(math.sqrt(float(np.mean(np.square(finite))))),
        "oracle_mae": float(np.mean(np.abs(finite))),
        "within_rate": float(np.mean(finite <= within_ft)),
    }


def min_abs_error(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmin(np.abs(values.astype(np.float32) - truth[:, None].astype(np.float32)), axis=1)


def build_candidate_union_readout(
    *,
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    union_cfg = get_nested(config, "candidate_union") or {}
    if not bool(union_cfg.get("enabled", True)):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"status": "disabled"},
        )

    heatmap_run_spec = str(union_cfg.get("heatmap_run_spec", "candidate_real_w128_b64_fullfold"))
    heatmap = predictions.loc[predictions["run_spec"] == heatmap_run_spec].copy()
    if heatmap.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"status": "missing_heatmap_run_spec", "heatmap_run_spec": heatmap_run_spec},
        )
    heatmap = heatmap.loc[heatmap["split"] == "valid"].copy() if "split" in heatmap.columns else heatmap
    heatmap = heatmap.sort_values(["well", "row_center"]).reset_index(drop=True)

    max_topk = max(int(value) for value in union_cfg.get("topk_values", TOPK_VALUES))
    heatmap_candidate_cols = [f"pred_top{rank}_tvt" for rank in range(1, max_topk + 1)]
    heatmap_score_cols = [f"pred_top{rank}_score" for rank in range(1, max_topk + 1)]
    keep_cols = [
        "id",
        "well",
        "sample_id",
        "fold_index",
        "row_center",
        "last_known_tvt",
        "prior_center_tvt",
        "true_center_tvt",
        "md_since_prefix",
        "distance_bucket",
        "path_step_abs_mean_ft",
        "path_step_abs_max_ft",
        "score_entropy",
        "score_top3_mass",
        "score_top5_mass",
        "top1_top2_score_margin",
        "top1_top3_score_margin",
    ]
    heatmap_candidates = heatmap[
        [col for col in keep_cols + heatmap_candidate_cols + heatmap_score_cols if col in heatmap.columns]
    ].copy()

    source_ref = resolve_config_reference(config, union_cfg.get("source_cache"))
    schema_ref = resolve_config_reference(config, union_cfg.get("source_schema"))
    source_path = find_optional_artifact(source_ref)
    schema_path = find_optional_artifact(schema_ref)
    if source_path is None:
        return (
            heatmap_candidates,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"status": "candidate_cache_missing", "requested": str(source_ref)},
        )

    id_col = str(union_cfg.get("id_column", "id"))
    target_delta_col = str(union_cfg.get("target_delta_column", "target"))
    last_col = str(union_cfg.get("last_known_tvt_column", "last_known_tvt"))
    distance_col = str(union_cfg.get("distance_column", "md_since"))
    requested_candidates = [str(value) for value in union_cfg.get("existing_candidates", [])]
    required_candidates = [str(value) for value in union_cfg.get("required_existing_candidates", [])]

    header = pd.read_csv(source_path, nrows=0).columns.tolist()
    available_candidates = [col for col in requested_candidates if col in header]
    missing_candidates = [col for col in requested_candidates if col not in header]
    missing_required = [col for col in required_candidates if col not in header]
    required_columns = [id_col, target_delta_col, last_col] + available_candidates
    if distance_col in header:
        required_columns.append(distance_col)
    required_columns = list(dict.fromkeys(required_columns))
    candidate_cache = pd.read_csv(
        source_path,
        usecols=required_columns,
        dtype={id_col: str},
        low_memory=False,
    )
    candidate_cache[id_col] = candidate_cache[id_col].astype(str)
    for col in candidate_cache.columns:
        if col != id_col:
            candidate_cache[col] = pd.to_numeric(candidate_cache[col], errors="coerce").astype(np.float32)
    candidate_cache["cache_true_tvt"] = candidate_cache[last_col] + candidate_cache[target_delta_col]
    rename_map = {id_col: "id"}
    if distance_col in candidate_cache.columns:
        rename_map[distance_col] = "cache_md_since"
    candidate_cache = candidate_cache.rename(columns=rename_map)

    merged = heatmap.merge(candidate_cache, on="id", how="inner", suffixes=("", "_cache"))
    if merged.empty or not available_candidates:
        return (
            heatmap_candidates,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "candidate_cache_join_empty_or_no_candidates",
                "source_path": str(source_path),
                "rows_heatmap": int(len(heatmap)),
                "available_candidates": available_candidates,
                "missing_candidates": missing_candidates,
                "missing_required_candidates": missing_required,
            },
        )

    truth = merged["true_center_tvt"].to_numpy(np.float32)
    cache_truth = merged["cache_true_tvt"].to_numpy(np.float32)
    truth_abs_diff = np.abs(truth - cache_truth)
    existing_values = merged[available_candidates].to_numpy(np.float32)
    existing_error = min_abs_error(existing_values, truth)

    within_ft = float(union_cfg.get("within_ft", 10.0))
    metric_rows: list[dict[str, Any]] = []
    base = oracle_metrics(existing_error, within_ft)
    metric_rows.append(
        {
            "candidate_set": "existing_union",
            "topk": 0,
            "rows": int(len(merged)),
            "candidate_count": int(len(available_candidates)),
            "new_best_candidate_rate": 0.0,
            "oracle_rmse_delta_vs_existing": 0.0,
            "within_delta_vs_existing": 0.0,
            **base,
        }
    )
    best_heatmap_topk_for_group: dict[str, np.ndarray] = {}
    for topk in [int(value) for value in union_cfg.get("topk_values", TOPK_VALUES)]:
        hm_cols = [f"pred_top{rank}_tvt" for rank in range(1, topk + 1) if f"pred_top{rank}_tvt" in merged]
        if not hm_cols:
            continue
        hm_values = merged[hm_cols].to_numpy(np.float32)
        hm_error = min_abs_error(hm_values, truth)
        union_error = np.minimum(existing_error, hm_error)
        best_heatmap_topk_for_group[f"top{topk}"] = union_error
        hm_metrics = oracle_metrics(hm_error, within_ft)
        union_metrics = oracle_metrics(union_error, within_ft)
        metric_rows.append(
            {
                "candidate_set": f"heatmap_only_top{topk}",
                "topk": topk,
                "rows": int(len(merged)),
                "candidate_count": topk,
                "new_best_candidate_rate": float(np.mean(hm_error + 1e-6 < existing_error)),
                "oracle_rmse_delta_vs_existing": float(hm_metrics["oracle_rmse"] - base["oracle_rmse"]),
                "within_delta_vs_existing": float(hm_metrics["within_rate"] - base["within_rate"]),
                **hm_metrics,
            }
        )
        metric_rows.append(
            {
                "candidate_set": f"existing_plus_heatmap_top{topk}",
                "topk": topk,
                "rows": int(len(merged)),
                "candidate_count": int(len(available_candidates) + topk),
                "new_best_candidate_rate": float(np.mean(hm_error + 1e-6 < existing_error)),
                "oracle_rmse_delta_vs_existing": float(union_metrics["oracle_rmse"] - base["oracle_rmse"]),
                "within_delta_vs_existing": float(union_metrics["within_rate"] - base["within_rate"]),
                **union_metrics,
            }
        )

    union_metrics = pd.DataFrame(metric_rows)
    top10_error = best_heatmap_topk_for_group.get("top10")
    by_well_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    if top10_error is not None:
        merged = merged.assign(existing_error=existing_error, heatmap_union_top10_error=top10_error)
        for well, group in merged.groupby("well", dropna=False):
            by_well_rows.append(
                {
                    "well": well,
                    "rows": int(len(group)),
                    "existing_oracle_rmse": oracle_metrics(
                        group["existing_error"].to_numpy(np.float32), within_ft
                    )["oracle_rmse"],
                    "heatmap_union_top10_oracle_rmse": oracle_metrics(
                        group["heatmap_union_top10_error"].to_numpy(np.float32), within_ft
                    )["oracle_rmse"],
                    "new_best_candidate_rate": float(
                        np.mean(group["heatmap_union_top10_error"] + 1e-6 < group["existing_error"])
                    ),
                }
            )
        for bucket, group in merged.groupby("distance_bucket", dropna=False):
            base_bucket = oracle_metrics(group["existing_error"].to_numpy(np.float32), within_ft)
            union_bucket = oracle_metrics(
                group["heatmap_union_top10_error"].to_numpy(np.float32), within_ft
            )
            bucket_rows.append(
                {
                    "distance_bucket": bucket,
                    "rows": int(len(group)),
                    "existing_oracle_rmse": base_bucket["oracle_rmse"],
                    "heatmap_union_top10_oracle_rmse": union_bucket["oracle_rmse"],
                    "oracle_rmse_delta": float(
                        union_bucket["oracle_rmse"] - base_bucket["oracle_rmse"]
                    ),
                    "existing_within10": base_bucket["within_rate"],
                    "heatmap_union_top10_within10": union_bucket["within_rate"],
                    "new_best_candidate_rate": float(
                        np.mean(group["heatmap_union_top10_error"] + 1e-6 < group["existing_error"])
                    ),
                }
            )

    summary = {
        "status": "ok",
        "source_path": str(source_path),
        "source_sha256": sha256_path(source_path),
        "source_decompressed_sha256": sha256_path(source_path, decompressed=source_path.suffix == ".gz"),
        "schema_path": str(schema_path) if schema_path is not None else None,
        "schema_sha256": sha256_path(schema_path) if schema_path is not None else None,
        "rows_heatmap": int(len(heatmap)),
        "rows_joined": int(len(merged)),
        "available_candidates": available_candidates,
        "missing_candidates": missing_candidates,
        "missing_required_candidates": missing_required,
        "truth_abs_diff_max": float(np.nanmax(truth_abs_diff)) if len(truth_abs_diff) else np.nan,
        "truth_abs_diff_mean": float(np.nanmean(truth_abs_diff)) if len(truth_abs_diff) else np.nan,
    }
    return (
        heatmap_candidates,
        union_metrics,
        pd.DataFrame(by_well_rows),
        pd.DataFrame(bucket_rows),
        summary,
    )


(
    heatmap_candidates_df,
    candidate_union_metrics_df,
    candidate_union_by_well_df,
    candidate_union_distance_df,
    candidate_union_summary,
) = build_candidate_union_readout(predictions=predictions_df, config=config)

well_rows: list[dict[str, Any]] = []
for keys, group in predictions_df.groupby(
    ["run_spec", "variant", "channel_set", "fold_index", "well"],
    dropna=False,
):
    run_spec, variant, channel_set, fold_index, well = keys
    well_rows.append(
        {
            "run_spec": run_spec,
            "variant": variant,
            "channel_set": channel_set,
            "fold_index": int(fold_index),
            "well": well,
            "samples": int(len(group)),
            "top3_within10_center_rate": float(group["top3_within10"].astype(float).mean()),
            "top10_within10_center_rate": float(group["top10_within10"].astype(float).mean()),
            "top3_oracle_center_rmse": float(
                math.sqrt(float(np.nanmean(np.square(group["top3_best_abs_error"].astype(float)))))
            ),
        }
    )
well_metrics_df = pd.DataFrame(well_rows)

distance_rows: list[dict[str, Any]] = []
for keys, group in predictions_df.groupby(
    ["run_spec", "variant", "channel_set", "fold_index", "distance_bucket"],
    dropna=False,
):
    run_spec, variant, channel_set, fold_index, bucket = keys
    distance_rows.append(
        {
            "run_spec": run_spec,
            "variant": variant,
            "channel_set": channel_set,
            "fold_index": int(fold_index),
            "distance_bucket": bucket,
            "samples": int(len(group)),
            "top3_within10_center_rate": float(group["top3_within10"].astype(float).mean()),
            "top10_within10_center_rate": float(group["top10_within10"].astype(float).mean()),
            "top3_oracle_center_rmse": float(
                math.sqrt(float(np.nanmean(np.square(group["top3_best_abs_error"].astype(float)))))
            ),
        }
    )
distance_metrics_df = pd.DataFrame(distance_rows)

aggregate_rows: list[dict[str, Any]] = []
weighted_columns = [
    "target_in_grid_rate",
    "top1_within10_center_rate",
    "top3_within10_center_rate",
    "top5_within10_center_rate",
    "top10_within10_center_rate",
    "top1_oracle_center_rmse",
    "top3_oracle_center_rmse",
    "top5_oracle_center_rmse",
    "top10_oracle_center_rmse",
    "top1_center_rmse",
    "path_step_abs_mean_ft",
    "path_step_abs_p95_ft",
]
for keys, group in fold_metrics_df.groupby(
    [
        "run_spec",
        "variant",
        "channel_set",
        "horizontal_window_rows",
        "typewell_window_bins",
    ],
    dropna=False,
):
    run_spec, variant, channel_set, horizontal_window_rows, typewell_window_bins = keys
    weights = group["valid_samples"].astype(float).to_numpy()
    row: dict[str, Any] = {
        "run_spec": run_spec,
        "variant": variant,
        "channel_set": channel_set,
        "horizontal_window_rows": int(horizontal_window_rows),
        "typewell_window_bins": int(typewell_window_bins),
        "folds_completed": int(group["fold_index"].nunique()),
        "valid_samples": int(group["valid_samples"].sum()),
        "train_samples": int(group["train_samples"].sum()),
        "epochs": int(group["epochs"].max()),
        "elapsed_sec": float(group["elapsed_sec"].sum()),
    }
    for column in weighted_columns:
        values = group[column].astype(float).to_numpy()
        finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        row[column] = float(np.average(values[finite], weights=weights[finite])) if finite.any() else np.nan
    run_wells = well_metrics_df.loc[well_metrics_df["run_spec"] == run_spec]
    row["worst_well_top3_within10_center_rate"] = (
        float(run_wells["top3_within10_center_rate"].min()) if len(run_wells) else np.nan
    )
    aggregate_rows.append(row)

metrics_df = pd.DataFrame(aggregate_rows)

def metric_for_run_spec(run_spec: str, column: str) -> float | None:
    values = metrics_df.loc[metrics_df["run_spec"] == run_spec, column]
    if values.empty:
        return None
    value = float(values.iloc[0])
    return value if np.isfinite(value) else None


def weighted_fold_metric(run_spec: str, column: str, folds: list[int]) -> float | None:
    group = fold_metrics_df.loc[
        (fold_metrics_df["run_spec"] == run_spec)
        & (fold_metrics_df["fold_index"].isin(folds))
    ]
    if group.empty:
        return None
    values = group[column].astype(float).to_numpy()
    weights = group["valid_samples"].astype(float).to_numpy()
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return None
    return float(np.average(values[finite], weights=weights[finite]))


key_metrics: dict[str, Any] = {
    "heatmap_run_spec": get_nested(config, "candidate_union.heatmap_run_spec"),
    "heatmap_top3_within10_center_rate": metric_for_run_spec(
        str(get_nested(config, "candidate_union.heatmap_run_spec")),
        "top3_within10_center_rate",
    ),
    "heatmap_top10_within10_center_rate": metric_for_run_spec(
        str(get_nested(config, "candidate_union.heatmap_run_spec")),
        "top10_within10_center_rate",
    ),
    "candidate_union_status": candidate_union_summary.get("status"),
}
if not candidate_union_metrics_df.empty:
    for candidate_set in [
        "existing_union",
        "heatmap_only_top10",
        "existing_plus_heatmap_top10",
    ]:
        row = candidate_union_metrics_df.loc[
            candidate_union_metrics_df["candidate_set"] == candidate_set
        ]
        if row.empty:
            continue
        first = row.iloc[0]
        key_metrics[f"{candidate_set}_oracle_rmse"] = float(first["oracle_rmse"])
        key_metrics[f"{candidate_set}_within_rate"] = float(first["within_rate"])
        key_metrics[f"{candidate_set}_new_best_candidate_rate"] = float(
            first["new_best_candidate_rate"]
        )
        key_metrics[f"{candidate_set}_oracle_rmse_delta_vs_existing"] = float(
            first["oracle_rmse_delta_vs_existing"]
        )
        key_metrics[f"{candidate_set}_within_delta_vs_existing"] = float(
            first["within_delta_vs_existing"]
        )

metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv"
fold_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv"
well_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_well_metrics.csv"
distance_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv"
predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_validation_predictions.csv.gz"
heatmap_candidates_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_heatmap_candidates.csv.gz"
candidate_path_npz_path = (
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_heatmap_candidate_paths_top10.npz"
)
candidate_path_samples_path = (
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_heatmap_candidate_path_samples.csv.gz"
)
candidate_path_rank_index_path = (
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_heatmap_candidate_path_rank_index.csv.gz"
)
candidate_union_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_metrics.csv"
candidate_union_by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_by_well.csv"
candidate_union_distance_path = (
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_distance_bucket_metrics.csv"
)
history_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_training_history.csv"
schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
run_spec_manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_run_spec_manifest.json"
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"

metrics_df.to_csv(metrics_path, index=False)
fold_metrics_df.to_csv(fold_metrics_path, index=False)
well_metrics_df.to_csv(well_metrics_path, index=False)
distance_metrics_df.to_csv(distance_metrics_path, index=False)
gzip_csv(predictions_df, predictions_path)
gzip_csv(heatmap_candidates_df, heatmap_candidates_path)
gzip_csv(candidate_path_sample_df, candidate_path_samples_path)
gzip_csv(candidate_path_rank_df, candidate_path_rank_index_path)
np.savez_compressed(
    candidate_path_npz_path,
    sample_id=candidate_path_output.sample_id,
    mode_index=candidate_path_output.mode_index,
    center_bin=candidate_path_output.center_bin,
    center_tvt=candidate_path_output.center_tvt,
    score=candidate_path_output.score,
    pred_tvt_path=candidate_path_output.pred_tvt_path,
    pred_bin_path=candidate_path_output.pred_bin_path,
    true_tvt_path=candidate_path_output.true_tvt_path,
    tvt_input_path=candidate_path_output.tvt_input_path,
    md_path=candidate_path_output.md_path,
    z_path=candidate_path_output.z_path,
    horizontal_row_index=candidate_path_output.horizontal_row_index,
    horizontal_offsets=candidate_path_output.horizontal_offsets,
)
candidate_union_metrics_df.to_csv(candidate_union_metrics_path, index=False)
candidate_union_by_well_df.to_csv(candidate_union_by_well_path, index=False)
candidate_union_distance_df.to_csv(candidate_union_distance_path, index=False)
history_df.to_csv(history_path, index=False)

schema_rows: list[dict[str, Any]] = []
active_channel_sets = sorted({spec.channel_set for spec in run_specs})
for channel_set in active_channel_sets:
    for index, (channel, description) in enumerate(channel_schema_for(channel_set)):
        schema_rows.append(
            {
                "channel_set": channel_set,
                "channel_index": index,
                "channel": channel,
                "description": description,
            }
        )
pd.DataFrame(schema_rows).to_csv(schema_path, index=False)
run_spec_manifest = {
    "experiment": EXPERIMENT_NAME,
    "run_specs": [asdict(spec) for spec in run_specs],
    "run_plan": [asdict(item) for item in run_plan],
    "cnn_model_count": len(run_plan),
}
run_spec_manifest_path.write_text(
    json.dumps(to_jsonable(run_spec_manifest), indent=2, sort_keys=True) + "\n"
)
manifest_path.write_text(json.dumps(to_jsonable(model_manifest), indent=2, sort_keys=True) + "\n")

artifact_sha = {
    "sample_index_csv_gz_sha256": sha256_path(sample_index_path),
    "sample_index_csv_decompressed_sha256": sha256_path(sample_index_path, decompressed=True),
    "validation_predictions_csv_gz_sha256": sha256_path(predictions_path),
    "validation_predictions_csv_decompressed_sha256": sha256_path(
        predictions_path,
        decompressed=True,
    ),
    "heatmap_candidates_csv_gz_sha256": sha256_path(heatmap_candidates_path),
    "heatmap_candidates_csv_decompressed_sha256": (
        sha256_path(heatmap_candidates_path, decompressed=True)
        if heatmap_candidates_path.suffix == ".gz"
        else None
    ),
    "heatmap_candidate_paths_npz_sha256": sha256_path(candidate_path_npz_path),
    "heatmap_candidate_path_samples_csv_gz_sha256": sha256_path(candidate_path_samples_path),
    "heatmap_candidate_path_samples_csv_decompressed_sha256": sha256_path(
        candidate_path_samples_path,
        decompressed=True,
    ),
    "heatmap_candidate_path_rank_index_csv_gz_sha256": sha256_path(
        candidate_path_rank_index_path
    ),
    "heatmap_candidate_path_rank_index_csv_decompressed_sha256": sha256_path(
        candidate_path_rank_index_path,
        decompressed=True,
    ),
    "candidate_union_metrics_csv_sha256": sha256_path(candidate_union_metrics_path),
    "candidate_union_by_well_csv_sha256": sha256_path(candidate_union_by_well_path),
    "candidate_union_distance_bucket_metrics_csv_sha256": sha256_path(
        candidate_union_distance_path
    ),
    "metrics_csv_sha256": sha256_path(metrics_path),
    "fold_metrics_csv_sha256": sha256_path(fold_metrics_path),
    "well_metrics_csv_sha256": sha256_path(well_metrics_path),
    "distance_bucket_metrics_csv_sha256": sha256_path(distance_metrics_path),
    "training_history_csv_sha256": sha256_path(history_path),
    "feature_schema_csv_sha256": sha256_path(schema_path),
    "run_spec_manifest_json_sha256": sha256_path(run_spec_manifest_path),
    "model_manifest_json_sha256": sha256_path(manifest_path),
}

candidate_path_summary = {
    "status": "saved_for_plotting",
    "format": "npz_plus_index_csv",
    "topk": int(path_topk),
    "horizon": int(path_horizon),
    "samples": int(path_count),
    "paths": int(path_count * path_topk),
    "npz_path": str(candidate_path_npz_path),
    "sample_index_path": str(candidate_path_samples_path),
    "rank_index_path": str(candidate_path_rank_index_path),
    "note": (
        "Local 128-row paths for deduplicated center-row topK candidates; "
        "no full-well trajectory stitching."
    ),
}

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_gpu_probe",
    "created_at": datetime.now(UTC).isoformat(),
    "seed": seed,
    "device": str(device),
    "torch_version": torch.__version__,
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "wells": {
        "all_selected": len(all_wells),
        "loaded": len(arrays_by_well),
        "usable": len(usable_wells),
    },
    "sample_overview": sample_overview.to_dict(orient="records"),
    "run_plan": run_plan_df.to_dict(orient="records"),
    "metrics": metrics_df.to_dict(orient="records"),
    "fold_metrics": fold_metrics_df.to_dict(orient="records"),
    "key_metrics": key_metrics,
    "candidate_union": candidate_union_summary,
    "candidate_union_metrics": candidate_union_metrics_df.to_dict(orient="records"),
    "candidate_path_output": candidate_path_summary,
    "artifact_sha": artifact_sha,
    "model_manifest": model_manifest,
    "reproducibility": {
        "deterministic_anchor": False,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "num_workers": int(get_nested(config, "runtime.num_workers") or 0),
        "cpu_fallback": False,
    },
}
summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
artifact_sha["summary_json_sha256"] = sha256_path(summary_path)
summary["artifact_sha"] = artifact_sha
summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")

metrics_json = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_gpu_probe",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "metric": "union_top10_oracle_rmse_delta",
    "summary": {
        "metrics": metrics_df.to_dict(orient="records"),
        "fold_metrics": fold_metrics_df.to_dict(orient="records"),
        "candidate_union_metrics": candidate_union_metrics_df.to_dict(orient="records"),
        "key_metrics": key_metrics,
        "candidate_union": candidate_union_summary,
        "candidate_path_output": candidate_path_summary,
        "artifact_sha": artifact_sha,
        "cuda_device_name": summary["cuda_device_name"],
    },
    "notes": "Train-side GPU diagnostic only; no submission.",
}
paths.metrics_path.write_text(json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n")

display(metrics_df)
display(fold_metrics_df)
display(candidate_union_metrics_df)
display(candidate_union_distance_df)
display(distance_metrics_df.head(30))
display(candidate_path_sample_df.head(10))
display(candidate_path_rank_df.head(30))
display(well_metrics_df.sort_values("top3_within10_center_rate").head(20))
display(history_df.tail(10))
print("Saved summary:", summary_path)
print(json.dumps(to_jsonable(key_metrics), indent=2, sort_keys=True))
print(json.dumps(to_jsonable(candidate_union_summary), indent=2, sort_keys=True))
print(json.dumps(to_jsonable(artifact_sha), indent=2, sort_keys=True))
