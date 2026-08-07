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
# # exp215 mtp full tail heatmap path generator probe train
#
# Train-side GPU diagnostic for a learned MTP full-tail heatmap path generator.
# This keeps the exp202 5-channel heatmap input, but replaces the grid-cell
# classifier with a continuous `path_pred [K, L]` and learned `path_logit [K]`
# head. It writes full-grid candidate paths for downstream selector audits.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and reproducibility helpers
# 3. Run plan and well helpers
# 4. Heatmap dataset
# 5. Continuous MTP model and loss
# 6. Training and dense full-tail prediction
# 7. Full-grid aggregation and candidate-union readout
# 8. Setup and execution
# 9. Metrics, SHA, and generated artifacts

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
        "Input-only prefix history SDF channel; this experiment does not train an SDF head.",
    ),
    ("observed_tvt_input_mask", "Observed TVT_input mask in the horizontal window."),
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
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == ".gz" else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


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
    training = get_nested(config, "model.training") or {}
    require_cuda = bool(training.get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("exp215 requires a Kaggle GPU runtime.")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        min_major = int(training.get("min_cuda_capability_major", 7))
        if capability[0] < min_major:
            raise RuntimeError(
                "Allocated GPU is incompatible with this PyTorch build: "
                f"{torch.cuda.get_device_name(0)!r}, capability={capability}."
            )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finite_float_array(
    series: pd.Series | None,
    fallback: float = 0.0,
    length: int | None = None,
) -> np.ndarray:
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


def nearest_grid_indices(grid_tvt: np.ndarray, truth_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(grid_tvt, truth_tvt, side="left")
    left = np.clip(positions - 1, 0, len(grid_tvt) - 1)
    right = np.clip(positions, 0, len(grid_tvt) - 1)
    choose_right = np.abs(grid_tvt[right] - truth_tvt) < np.abs(grid_tvt[left] - truth_tvt)
    index = np.where(choose_right, right, left).astype(np.int64)
    distance = np.abs(grid_tvt[index] - truth_tvt).astype(np.float32)
    return index, distance


def clean_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


# %% [markdown]
# ## 3. Run plan and well helpers

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
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    typewell_gr_shuffled: np.ndarray
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
    path_logit: np.ndarray
    path_prob: np.ndarray
    pred_tvt_path: np.ndarray
    pred_bin_path: np.ndarray
    weighted_tvt_path: np.ndarray
    true_tvt_path: np.ndarray
    target_mask: np.ndarray
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
            path_logit=np.empty((0, topk), dtype=np.float32),
            path_prob=np.empty((0, topk), dtype=np.float32),
            pred_tvt_path=np.empty((0, topk, horizon), dtype=np.float32),
            pred_bin_path=np.empty((0, topk, horizon), dtype=np.int16),
            weighted_tvt_path=np.empty((0, horizon), dtype=np.float32),
            true_tvt_path=np.empty((0, horizon), dtype=np.float32),
            target_mask=np.empty((0, horizon), dtype=np.float32),
            tvt_input_path=np.empty((0, horizon), dtype=np.float32),
            md_path=np.empty((0, horizon), dtype=np.float32),
            z_path=np.empty((0, horizon), dtype=np.float32),
            horizontal_row_index=np.empty((0, horizon), dtype=np.int32),
            horizontal_offsets=offsets,
        )


def channel_schema_for(channel_set: str) -> list[tuple[str, str]]:
    if channel_set != "base":
        raise ValueError(f"exp215 supports only the base 5-channel input, got {channel_set}")
    return list(BASE_CHANNEL_SCHEMA)


def resolve_run_specs(config: dict[str, Any]) -> list[RunSpec]:
    raw_specs = get_nested(config, "model.active_run_specs") or []
    default_folds = tuple(int(v) for v in (get_nested(config, "validation.active_fold_indices") or [0]))
    specs: list[RunSpec] = []
    for raw_spec in raw_specs:
        spec = dict(raw_spec)
        channel_set = str(spec.get("channel_set", "base"))
        channel_schema_for(channel_set)
        specs.append(
            RunSpec(
                name=clean_name(str(spec["name"])),
                variant=str(spec.get("variant", "real_gr")),
                channel_set=channel_set,
                fold_indices=tuple(int(v) for v in spec.get("fold_indices", default_folds)),
                horizontal_window_rows=int(spec.get("horizontal_window_rows", 128)),
                typewell_window_bins=int(spec.get("typewell_window_bins", 64)),
                tvt_grid_half_width_ft=float(spec.get("tvt_grid_half_width_ft", 192.0)),
                history_scale_ft=float(spec.get("history_scale_ft", 200.0)),
            )
        )
    if not specs:
        raise ValueError("model.active_run_specs must not be empty")
    return specs


def expand_run_plan(specs: list[RunSpec]) -> list[RunPlanItem]:
    plan: list[RunPlanItem] = []
    for spec in specs:
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
    training = get_nested(config, "model.training") or {}
    train_wells = [wells[index] for index in train_idx]
    valid_wells = [wells[index] for index in valid_idx]
    if training.get("max_train_wells") is not None:
        train_wells = train_wells[: int(training["max_train_wells"])]
    if training.get("max_valid_wells") is not None:
        valid_wells = valid_wells[: int(training["max_valid_wells"])]
    return train_wells, valid_wells


def list_train_wells(train_dir: Path, max_wells: int | None) -> list[str]:
    wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in train_dir.glob("*__horizontal_well.csv")
    )
    return wells[: int(max_wells)] if max_wells is not None else wells


def read_well_arrays(well: str, train_dir: Path, seed: int) -> WellArrays | None:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        return None
    h = pd.read_csv(horizontal_path)
    t = pd.read_csv(typewell_path)
    if not {"MD", "Z", "TVT", "TVT_input", "GR"}.issubset(h.columns):
        return None
    if not {"TVT", "GR"}.issubset(t.columns):
        return None
    tvt_input = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input))
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
    horizontal_gr, _ = fill_and_scale_gr(h["GR"])
    md = finite_float_array(h["MD"])
    x = finite_float_array(h["X"] if "X" in h.columns else None, length=len(h))
    y = finite_float_array(h["Y"] if "Y" in h.columns else None, length=len(h))
    z = finite_float_array(h["Z"])
    tvt = finite_float_array(h["TVT"])
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
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        typewell_gr_shuffled=typewell_gr_shuffled,
        prefix_end=prefix_end,
        last_known_tvt=float(tvt_input[prefix_end]),
        last_known_z=float(z[prefix_end]),
        last_known_md=float(md[prefix_end]),
    )


def tail_stop_for_well(arrays: WellArrays, max_tail_rows: int | None) -> int:
    if max_tail_rows is None:
        return int(arrays.horizontal_rows - 1)
    return int(min(arrays.horizontal_rows - 1, arrays.prefix_end + int(max_tail_rows)))


def sparse_rows_for_well(
    arrays: WellArrays,
    *,
    samples_per_well: int,
    max_tail_rows: int | None,
) -> np.ndarray:
    tail_start = arrays.prefix_end + 1
    tail_stop = tail_stop_for_well(arrays, max_tail_rows)
    if tail_stop <= tail_start:
        return np.array([], dtype=np.int32)
    count = min(int(samples_per_well), int(tail_stop - tail_start + 1))
    rows = np.linspace(tail_start, tail_stop, count)
    return np.unique(np.rint(rows).astype(np.int32))


def dense_rows_for_well(
    arrays: WellArrays,
    *,
    stride: int,
    max_tail_rows: int | None,
    include_tail_stop: bool,
) -> np.ndarray:
    tail_start = arrays.prefix_end + 1
    tail_stop = tail_stop_for_well(arrays, max_tail_rows)
    if tail_stop <= tail_start:
        return np.array([], dtype=np.int32)
    rows = np.arange(tail_start, tail_stop + 1, int(stride), dtype=np.int32)
    if include_tail_stop and (len(rows) == 0 or int(rows[-1]) != int(tail_stop)):
        rows = np.unique(np.concatenate([rows, np.asarray([tail_stop], dtype=np.int32)]))
    return rows.astype(np.int32)


def prior_center_tvt(arrays: WellArrays, row_center: int) -> float:
    return float(arrays.last_known_tvt - (float(arrays.z[row_center]) - arrays.last_known_z))


def sample_label_status(
    arrays: WellArrays,
    row_center: int,
    horizontal_offsets: np.ndarray,
    grid_offsets_tvt: np.ndarray,
    target_tolerance: float,
) -> tuple[float, float, bool, float]:
    prior_center = prior_center_tvt(arrays, row_center)
    grid_tvt = prior_center + grid_offsets_tvt
    h_idx = np.clip(row_center + horizontal_offsets, 0, arrays.horizontal_rows - 1)
    _, target_distance = nearest_grid_indices(grid_tvt, arrays.tvt[h_idx])
    center_position = int(np.flatnonzero(horizontal_offsets == 0)[0])
    center_distance = float(target_distance[center_position])
    return (
        prior_center,
        float(arrays.tvt[row_center]),
        bool(center_distance <= float(target_tolerance)),
        float(np.mean(target_distance <= float(target_tolerance))),
    )


def build_sample_rows(
    *,
    arrays_by_well: dict[str, WellArrays],
    wells: list[str],
    config: dict[str, Any],
    plan_item: RunPlanItem,
    split: str,
    sample_id_start: int,
    dense: bool,
) -> pd.DataFrame:
    training = get_nested(config, "model.training") or {}
    generation = get_nested(config, "path_generation") or {}
    horizontal_window = int(plan_item.horizontal_window_rows)
    horizontal_offsets = np.arange(-(horizontal_window // 2), horizontal_window // 2, dtype=np.int32)
    grid_offsets = np.linspace(
        -float(plan_item.tvt_grid_half_width_ft),
        float(plan_item.tvt_grid_half_width_ft),
        int(plan_item.typewell_window_bins),
    ).astype(np.float32)
    target_tolerance = float(training.get("center_target_tolerance_ft", 10.0))
    max_tail_rows = training.get("max_tail_rows")
    max_tail_rows = None if max_tail_rows is None else int(max_tail_rows)
    rows: list[dict[str, Any]] = []
    sample_id = int(sample_id_start)
    for well in wells:
        arrays = arrays_by_well[well]
        if dense:
            row_centers = dense_rows_for_well(
                arrays,
                stride=int(generation.get("row_center_stride", 64)),
                max_tail_rows=max_tail_rows,
                include_tail_stop=parse_bool(generation.get("include_tail_stop"), True),
            )
        else:
            key = "train_samples_per_well" if split == "train" else "valid_samples_per_well"
            row_centers = sparse_rows_for_well(
                arrays,
                samples_per_well=int(training.get(key, 24)),
                max_tail_rows=max_tail_rows,
            )
        for row_center in row_centers:
            prior_center, true_center, center_in_grid, label_fraction = sample_label_status(
                arrays,
                int(row_center),
                horizontal_offsets,
                grid_offsets,
                target_tolerance,
            )
            rows.append(
                asdict(
                    WindowSample(
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
                        typewell_window_bins=int(plan_item.typewell_window_bins),
                        tvt_grid_half_width_ft=float(plan_item.tvt_grid_half_width_ft),
                        history_scale_ft=float(plan_item.history_scale_ft),
                        last_known_tvt=arrays.last_known_tvt,
                        prior_center_tvt=float(prior_center),
                        true_center_tvt=float(true_center),
                        md_since_prefix=float(arrays.md[row_center] - arrays.last_known_md),
                        z_since_prefix=float(arrays.z[row_center] - arrays.last_known_z),
                        center_target_in_grid=center_in_grid,
                        label_fraction=float(label_fraction),
                    )
                )
            )
            sample_id += 1
    return pd.DataFrame(rows)


def build_sample_index_for_plan(
    *,
    arrays_by_well: dict[str, WellArrays],
    train_wells: list[str],
    valid_wells: list[str],
    config: dict[str, Any],
    plan_item: RunPlanItem,
    sample_id_start: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    next_id = int(sample_id_start)
    for split, wells, dense in [
        ("train", train_wells, False),
        ("valid", valid_wells, False),
        ("valid_dense", valid_wells, True),
    ]:
        frame = build_sample_rows(
            arrays_by_well=arrays_by_well,
            wells=wells,
            config=config,
            plan_item=plan_item,
            split=split,
            sample_id_start=next_id,
            dense=dense,
        )
        if not frame.empty:
            next_id = int(frame["sample_id"].max()) + 1
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# %% [markdown]
# ## 4. Heatmap dataset

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
        horizontal_window = int(plan_item.horizontal_window_rows)
        self.horizontal_offsets = np.arange(
            -(horizontal_window // 2),
            horizontal_window // 2,
            dtype=np.int32,
        )
        self.grid_offsets_tvt = np.linspace(
            -float(plan_item.tvt_grid_half_width_ft),
            float(plan_item.tvt_grid_half_width_ft),
            int(plan_item.typewell_window_bins),
        ).astype(np.float32)
        self.history_scale_ft = float(plan_item.history_scale_ft)
        self.path_scale_ft = float((get_nested(config, "model.loss") or {}).get("path_scale_ft", 256.0))
        self.center_position = int(np.flatnonzero(self.horizontal_offsets == 0)[0])

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.sample_index.iloc[index]
        arrays = self.arrays_by_well[str(row["well"])]
        row_center = int(row["row_center"])
        h_idx = np.clip(row_center + self.horizontal_offsets, 0, arrays.horizontal_rows - 1)
        grid_tvt = float(row["prior_center_tvt"]) + self.grid_offsets_tvt

        t_gr_source = arrays.typewell_gr_shuffled if self.variant == "shuffled_gr" else arrays.typewell_gr
        t_gr = np.interp(grid_tvt, arrays.typewell_tvt, t_gr_source).astype(np.float32)
        h_gr = arrays.horizontal_gr[h_idx].astype(np.float32)
        if self.variant == "no_gr":
            t_gr = np.zeros_like(t_gr, dtype=np.float32)
            h_gr = np.zeros_like(h_gr, dtype=np.float32)

        t_heatmap = np.broadcast_to(t_gr.reshape(1, -1), (len(h_idx), len(grid_tvt)))
        h_heatmap = np.broadcast_to(h_gr.reshape(-1, 1), (len(h_idx), len(grid_tvt)))
        diff = t_heatmap - h_heatmap
        observed_tvt = arrays.tvt_input[h_idx]
        obs_mask = np.isfinite(observed_tvt).astype(np.float32)
        observed_safe = np.where(np.isfinite(observed_tvt), observed_tvt, 0.0).astype(np.float32)
        history = (grid_tvt.reshape(1, -1) - observed_safe.reshape(-1, 1)) / self.history_scale_ft
        history = np.clip(history * obs_mask.reshape(-1, 1), -8.0, 8.0)
        mask_heatmap = np.broadcast_to(obs_mask.reshape(-1, 1), history.shape)
        image = np.stack([t_heatmap, h_heatmap, diff, history, mask_heatmap], axis=0)

        true_tvt_path = arrays.tvt[h_idx].astype(np.float32)
        target_mask = np.isfinite(true_tvt_path).astype(np.float32)
        target_path = ((true_tvt_path - float(row["prior_center_tvt"])) / self.path_scale_ft).astype(np.float32)
        target_path = np.where(np.isfinite(target_path), target_path, 0.0).astype(np.float32)
        center_idx, center_dist = nearest_grid_indices(grid_tvt, true_tvt_path)
        return {
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
            "image": torch.from_numpy(image.astype(np.float32)),
            "target_path": torch.from_numpy(target_path.astype(np.float32)),
            "target_mask": torch.from_numpy(target_mask.astype(np.float32)),
            "grid_tvt": torch.from_numpy(grid_tvt.astype(np.float32)),
            "prior_center_tvt": torch.tensor(float(row["prior_center_tvt"]), dtype=torch.float32),
            "true_center_tvt": torch.tensor(float(row["true_center_tvt"]), dtype=torch.float32),
            "center_target_idx": torch.tensor(int(center_idx[self.center_position]), dtype=torch.long),
            "center_target_distance": torch.tensor(float(center_dist[self.center_position]), dtype=torch.float32),
            "horizontal_row_index": torch.from_numpy(h_idx.astype(np.int32)),
            "horizontal_md": torch.from_numpy(arrays.md[h_idx].astype(np.float32)),
            "horizontal_z": torch.from_numpy(arrays.z[h_idx].astype(np.float32)),
            "true_tvt_path": torch.from_numpy(true_tvt_path.astype(np.float32)),
            "tvt_input_path": torch.from_numpy(arrays.tvt_input[h_idx].astype(np.float32)),
        }


# %% [markdown]
# ## 5. Continuous MTP model and loss

# %%
class FullTailMTPNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        path_modes: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
        use_group_norm: bool,
        max_residual_norm: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(in_channels)
        padding = int(kernel_size) // 2
        for width_value in channels:
            width = int(width_value)
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
        self.max_residual_norm = float(max_residual_norm)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.backbone(image)
        path_map = self.path_head(feature)
        path_pred = torch.tanh(path_map.mean(dim=3)) * self.max_residual_norm
        path_logit = self.mode_head(feature)
        return path_pred, path_logit


def make_model(config: dict[str, Any], *, in_channels: int) -> FullTailMTPNet:
    arch = get_nested(config, "model.architecture") or {}
    loss_cfg = get_nested(config, "model.loss") or {}
    path_scale_ft = float(loss_cfg.get("path_scale_ft", 256.0))
    max_residual_ft = float(arch.get("max_residual_ft", 768.0))
    return FullTailMTPNet(
        in_channels=int(in_channels),
        path_modes=int(arch.get("path_modes", 10)),
        channels=[int(value) for value in arch.get("channels", [32, 64, 64, 64])],
        kernel_size=int(arch.get("kernel_size", 3)),
        dropout=float(arch.get("dropout", 0.05)),
        use_group_norm=bool(arch.get("use_group_norm", True)),
        max_residual_norm=max_residual_ft / path_scale_ft,
    )


def closest_mode_path_loss(
    path_pred: torch.Tensor,
    path_logit: torch.Tensor,
    target_path: torch.Tensor,
    target_mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_cfg = get_nested(config, "model.loss") or {}
    alpha = float(loss_cfg.get("mode_ce_alpha", 0.2))
    smoothness_alpha = float(loss_cfg.get("smoothness_alpha", 0.0))
    loss_type = str(loss_cfg.get("regression", "smooth_l1"))
    diff = path_pred - target_path[:, None, :]
    mask = target_mask[:, None, :].clamp_min(0.0)
    if loss_type == "mse":
        per_step = diff.square()
    elif loss_type == "mae":
        per_step = diff.abs()
    else:
        per_step = F.smooth_l1_loss(path_pred, target_path[:, None, :].expand_as(path_pred), reduction="none")
    counts = mask.sum(dim=2).clamp_min(1.0)
    per_mode = (per_step * mask).sum(dim=2) / counts
    best_mode = per_mode.argmin(dim=1)
    chosen_reg = per_mode.gather(1, best_mode[:, None]).squeeze(1).mean()
    mode_loss = F.cross_entropy(path_logit, best_mode)
    total = chosen_reg + alpha * mode_loss
    if smoothness_alpha > 0:
        chosen_path = path_pred[torch.arange(path_pred.shape[0], device=path_pred.device), best_mode]
        smoothness = torch.diff(chosen_path, dim=1).abs().mean()
        total = total + smoothness_alpha * smoothness
    return total, best_mode, per_mode


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


def rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(math.sqrt(float(np.mean(np.square(finite))))) if len(finite) else float("nan")


def masked_path_rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff2 = np.square(pred.astype(np.float32) - truth[:, None, :].astype(np.float32))
    valid = mask[:, None, :].astype(bool) & np.isfinite(diff2)
    sums = np.where(valid, diff2, 0.0).sum(axis=2)
    counts = valid.sum(axis=2)
    return np.sqrt(np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0))


def summarize_predictions(predictions: pd.DataFrame, *, loss: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "loss": float(loss),
        "samples": int(len(predictions)),
        "rank1_center_rmse": rmse(predictions["pred_top1_abs_error"].to_numpy(np.float32)),
        "weighted_center_rmse": rmse(predictions["weighted_center_abs_error"].to_numpy(np.float32)),
        "rank1_path_rmse": rmse(predictions["pred_top1_path_rmse"].to_numpy(np.float32)),
        "weighted_path_rmse": rmse(predictions["weighted_path_rmse"].to_numpy(np.float32)),
        "path_step_abs_mean_ft": float(np.nanmean(predictions["path_step_abs_mean_ft"])),
    }
    for topk in TOPK_VALUES:
        if f"top{topk}_best_abs_error" not in predictions:
            continue
        best_error = predictions[f"top{topk}_best_abs_error"].to_numpy(np.float32)
        metrics[f"top{topk}_oracle_center_rmse"] = rmse(best_error)
        metrics[f"top{topk}_within10_center_rate"] = float(np.nanmean(best_error <= 10.0))
        if f"top{topk}_best_path_rmse" in predictions:
            metrics[f"top{topk}_oracle_path_rmse"] = float(
                np.nanmean(predictions[f"top{topk}_best_path_rmse"].to_numpy(np.float32))
            )
    return metrics


@torch.no_grad()
def evaluate_model(
    *,
    model: FullTailMTPNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: dict[str, Any],
    device: torch.device,
    plan_item: RunPlanItem,
    collect_paths: bool,
) -> tuple[dict[str, Any], pd.DataFrame, CandidatePathOutput]:
    model.eval()
    loss_cfg = get_nested(config, "model.loss") or {}
    path_scale_ft = float(loss_cfg.get("path_scale_ft", 256.0))
    topk = max(TOPK_VALUES)
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    center_position = loader.dataset.center_position  # type: ignore[attr-defined]
    horizontal_offsets = loader.dataset.horizontal_offsets.astype(np.int32)  # type: ignore[attr-defined]
    horizon = int(len(horizontal_offsets))
    collected: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "mode_index": [],
        "center_bin": [],
        "center_tvt": [],
        "path_logit": [],
        "path_prob": [],
        "pred_tvt_path": [],
        "pred_bin_path": [],
        "weighted_tvt_path": [],
        "true_tvt_path": [],
        "target_mask": [],
        "tvt_input_path": [],
        "md_path": [],
        "z_path": [],
        "horizontal_row_index": [],
    }
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_path = batch["target_path"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        path_pred, path_logit = model(image)
        loss, _, _ = closest_mode_path_loss(path_pred, path_logit, target_path, target_mask, config)
        losses.append(float(loss.detach().cpu()))
        prob = torch.softmax(path_logit, dim=1)
        order = prob.argsort(dim=1, descending=True)

        prior = batch["prior_center_tvt"].cpu().numpy().astype(np.float32)
        pred_path_all = prior[:, None, None] + path_pred.detach().cpu().numpy().astype(np.float32) * path_scale_ft
        prob_np = prob.detach().cpu().numpy().astype(np.float32)
        logit_np = path_logit.detach().cpu().numpy().astype(np.float32)
        order_np = order.detach().cpu().numpy().astype(np.int64)
        sample_ids = batch["sample_id"].cpu().numpy().astype(np.int64)
        grid_tvt = batch["grid_tvt"].cpu().numpy().astype(np.float32)
        true_center = batch["true_center_tvt"].cpu().numpy().astype(np.float32)
        true_path = batch["true_tvt_path"].cpu().numpy().astype(np.float32)
        mask_np = batch["target_mask"].cpu().numpy().astype(np.float32)
        weighted_path = np.sum(pred_path_all * prob_np[:, :, None], axis=1)
        path_rmse = masked_path_rmse(pred_path_all, true_path, mask_np)
        weighted_path_rmse = masked_path_rmse(
            weighted_path[:, None, :],
            true_path,
            mask_np,
        )[:, 0]

        batch_size = len(sample_ids)
        top_mode = np.full((batch_size, topk), -1, dtype=np.int16)
        top_center_bin = np.full((batch_size, topk), -1, dtype=np.int16)
        top_center_tvt = np.full((batch_size, topk), np.nan, dtype=np.float32)
        top_logit = np.full((batch_size, topk), np.nan, dtype=np.float32)
        top_prob = np.full((batch_size, topk), np.nan, dtype=np.float32)
        top_path = np.full((batch_size, topk, horizon), np.nan, dtype=np.float32)
        top_bin_path = np.full((batch_size, topk, horizon), -1, dtype=np.int16)

        for i in range(batch_size):
            ordered_modes = order_np[i, :topk]
            center_values: list[float] = []
            center_errors: list[float] = []
            rank_path_rmse: list[float] = []
            for rank_index, mode_index in enumerate(ordered_modes):
                mode_int = int(mode_index)
                pred_path = pred_path_all[i, mode_int, :]
                center_value = float(pred_path[center_position])
                center_bin, _ = nearest_grid_indices(grid_tvt[i], pred_path)
                top_mode[i, rank_index] = mode_int
                top_center_bin[i, rank_index] = int(center_bin[center_position])
                top_center_tvt[i, rank_index] = center_value
                top_logit[i, rank_index] = float(logit_np[i, mode_int])
                top_prob[i, rank_index] = float(prob_np[i, mode_int])
                top_path[i, rank_index, :] = pred_path
                top_bin_path[i, rank_index, :] = center_bin.astype(np.int16)
                center_values.append(center_value)
                center_errors.append(abs(center_value - float(true_center[i])))
                rank_path_rmse.append(float(path_rmse[i, mode_int]))

            score_values = top_prob[i]
            finite_scores = score_values[np.isfinite(score_values)]
            entropy = float(-np.sum(finite_scores * np.log(np.maximum(finite_scores, 1e-12))))
            step_abs = np.abs(np.diff(top_path[i, 0, :].astype(np.float32)))
            record: dict[str, Any] = {
                "run_spec": plan_item.run_spec,
                "variant": plan_item.variant,
                "channel_set": plan_item.channel_set,
                "fold_index": int(plan_item.fold_index),
                "sample_id": int(sample_ids[i]),
                "weighted_center_tvt": float(weighted_path[i, center_position]),
                "weighted_center_abs_error": abs(float(weighted_path[i, center_position]) - float(true_center[i])),
                "weighted_path_rmse": float(weighted_path_rmse[i]),
                "path_prob_entropy": entropy,
                "path_prob_top3_mass": float(np.nansum(score_values[:3])),
                "path_prob_top5_mass": float(np.nansum(score_values[:5])),
                "top1_top2_path_prob_margin": float(score_values[0] - score_values[1])
                if len(score_values) > 1 and np.isfinite(score_values[1])
                else np.nan,
                "path_step_abs_mean_ft": float(np.nanmean(step_abs)) if len(step_abs) else 0.0,
                "path_step_abs_max_ft": float(np.nanmax(step_abs)) if len(step_abs) else 0.0,
            }
            for rank, (center_value, center_error, one_path_rmse) in enumerate(
                zip(center_values, center_errors, rank_path_rmse, strict=False),
                start=1,
            ):
                record[f"pred_top{rank}_tvt"] = float(center_value)
                record[f"pred_top{rank}_score"] = float(score_values[rank - 1])
                record[f"pred_top{rank}_abs_error"] = float(center_error)
                record[f"pred_top{rank}_path_rmse"] = float(one_path_rmse)
            for value in TOPK_VALUES:
                record[f"top{value}_best_abs_error"] = float(np.nanmin(center_errors[:value]))
                record[f"top{value}_within10"] = bool(record[f"top{value}_best_abs_error"] <= 10.0)
                record[f"top{value}_best_path_rmse"] = float(np.nanmin(rank_path_rmse[:value]))
            rows.append(record)

        if collect_paths:
            collected["sample_id"].append(sample_ids)
            collected["mode_index"].append(top_mode)
            collected["center_bin"].append(top_center_bin)
            collected["center_tvt"].append(top_center_tvt)
            collected["path_logit"].append(top_logit)
            collected["path_prob"].append(top_prob)
            collected["pred_tvt_path"].append(top_path)
            collected["pred_bin_path"].append(top_bin_path)
            collected["weighted_tvt_path"].append(weighted_path.astype(np.float32))
            collected["true_tvt_path"].append(true_path.astype(np.float32))
            collected["target_mask"].append(mask_np.astype(np.float32))
            collected["tvt_input_path"].append(batch["tvt_input_path"].cpu().numpy().astype(np.float32))
            collected["md_path"].append(batch["horizontal_md"].cpu().numpy().astype(np.float32))
            collected["z_path"].append(batch["horizontal_z"].cpu().numpy().astype(np.float32))
            collected["horizontal_row_index"].append(batch["horizontal_row_index"].cpu().numpy().astype(np.int32))

    predictions = pd.DataFrame(rows)
    metrics = summarize_predictions(predictions, loss=float(np.mean(losses)) if losses else np.nan)
    if not collect_paths or not collected["sample_id"]:
        return metrics, predictions, CandidatePathOutput.empty(topk=topk, horizon=horizon, horizontal_offsets=horizontal_offsets)
    output = CandidatePathOutput(
        sample_id=np.concatenate(collected["sample_id"], axis=0),
        mode_index=np.concatenate(collected["mode_index"], axis=0),
        center_bin=np.concatenate(collected["center_bin"], axis=0),
        center_tvt=np.concatenate(collected["center_tvt"], axis=0),
        path_logit=np.concatenate(collected["path_logit"], axis=0),
        path_prob=np.concatenate(collected["path_prob"], axis=0),
        pred_tvt_path=np.concatenate(collected["pred_tvt_path"], axis=0),
        pred_bin_path=np.concatenate(collected["pred_bin_path"], axis=0),
        weighted_tvt_path=np.concatenate(collected["weighted_tvt_path"], axis=0),
        true_tvt_path=np.concatenate(collected["true_tvt_path"], axis=0),
        target_mask=np.concatenate(collected["target_mask"], axis=0),
        tvt_input_path=np.concatenate(collected["tvt_input_path"], axis=0),
        md_path=np.concatenate(collected["md_path"], axis=0),
        z_path=np.concatenate(collected["z_path"], axis=0),
        horizontal_row_index=np.concatenate(collected["horizontal_row_index"], axis=0),
        horizontal_offsets=horizontal_offsets,
    )
    return metrics, predictions, output


# %% [markdown]
# ## 6. Training and dense full-tail prediction

# %%
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
    dense_dataset = HeatmapWindowDataset(
        sample_index=sample_index,
        arrays_by_well=arrays_by_well,
        plan_item=plan_item,
        config=config,
        split="valid_dense",
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(dense_dataset) == 0:
        raise RuntimeError(f"Empty dataset for {plan_item.run_spec} fold {plan_item.fold_index}")
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=parse_bool(training.get("dataloader_shuffle"), True),
        seed=stable_int(EXPERIMENT_NAME, plan_item.run_spec, str(plan_item.fold_index), "train", str(seed), modulo=2**31 - 1),
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=stable_int(EXPERIMENT_NAME, plan_item.run_spec, str(plan_item.fold_index), "valid", str(seed), modulo=2**31 - 1),
    )
    dense_loader = make_loader(
        dense_dataset,
        batch_size=int(get_nested(config, "path_generation.batch_size") or batch_size),
        shuffle=False,
        seed=stable_int(EXPERIMENT_NAME, plan_item.run_spec, str(plan_item.fold_index), "dense", str(seed), modulo=2**31 - 1),
    )
    model = make_model(config, in_channels=len(channel_schema_for(plan_item.channel_set))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    grad_clip = float(training.get("gradient_clip_norm", 1.0))
    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target_path = batch["target_path"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            path_pred, path_logit = model(image)
            loss, _, _ = closest_mode_path_loss(path_pred, path_logit, target_path, target_mask, config)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        valid_metrics, _, _ = evaluate_model(
            model=model,
            loader=valid_loader,
            config=config,
            device=device,
            plan_item=plan_item,
            collect_paths=False,
        )
        valid_score = float(valid_metrics["weighted_center_rmse"])
        if valid_score < best_score:
            best_score = valid_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history_rows.append(
            {
                "run_spec": plan_item.run_spec,
                "fold_index": int(plan_item.fold_index),
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                "valid_weighted_center_rmse": valid_metrics["weighted_center_rmse"],
                "valid_rank1_center_rmse": valid_metrics["rank1_center_rmse"],
                "valid_top10_oracle_center_rmse": valid_metrics.get("top10_oracle_center_rmse"),
                "elapsed_sec": float(time.time() - start),
            }
        )
        print(
            f"{plan_item.run_spec} fold={plan_item.fold_index} epoch={epoch}/{epochs} "
            f"train_loss={history_rows[-1]['train_loss']:.4f} "
            f"weighted_rmse={valid_score:.4f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    sparse_metrics, _, _ = evaluate_model(
        model=model,
        loader=valid_loader,
        config=config,
        device=device,
        plan_item=plan_item,
        collect_paths=False,
    )
    dense_metrics, dense_predictions, path_output = evaluate_model(
        model=model,
        loader=dense_loader,
        config=config,
        device=device,
        plan_item=plan_item,
        collect_paths=True,
    )
    final_metrics = {
        **{f"sparse_{key}": value for key, value in sparse_metrics.items()},
        **{f"dense_{key}": value for key, value in dense_metrics.items()},
        "run_spec": plan_item.run_spec,
        "variant": plan_item.variant,
        "channel_set": plan_item.channel_set,
        "fold_index": int(plan_item.fold_index),
        "horizontal_window_rows": int(plan_item.horizontal_window_rows),
        "typewell_window_bins": int(plan_item.typewell_window_bins),
        "train_samples": int(len(train_dataset)),
        "valid_samples": int(len(valid_dataset)),
        "dense_samples": int(len(dense_dataset)),
        "epochs": int(epochs),
        "best_sparse_weighted_center_rmse": float(best_score),
        "elapsed_sec": float(time.time() - start),
    }
    model_path = output_dir / f"{OUTPUT_PREFIX}_{plan_item.run_spec}_fold{int(plan_item.fold_index)}_model.pt"
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
    return final_metrics, dense_predictions, path_output, pd.DataFrame(history_rows), model_path


def stack_path_outputs(outputs: list[CandidatePathOutput]) -> CandidatePathOutput:
    non_empty = [output for output in outputs if len(output.sample_id) > 0]
    if not non_empty:
        return CandidatePathOutput.empty(topk=max(TOPK_VALUES), horizon=128)
    first = non_empty[0]
    return CandidatePathOutput(
        sample_id=np.concatenate([x.sample_id for x in non_empty], axis=0),
        mode_index=np.concatenate([x.mode_index for x in non_empty], axis=0),
        center_bin=np.concatenate([x.center_bin for x in non_empty], axis=0),
        center_tvt=np.concatenate([x.center_tvt for x in non_empty], axis=0),
        path_logit=np.concatenate([x.path_logit for x in non_empty], axis=0),
        path_prob=np.concatenate([x.path_prob for x in non_empty], axis=0),
        pred_tvt_path=np.concatenate([x.pred_tvt_path for x in non_empty], axis=0),
        pred_bin_path=np.concatenate([x.pred_bin_path for x in non_empty], axis=0),
        weighted_tvt_path=np.concatenate([x.weighted_tvt_path for x in non_empty], axis=0),
        true_tvt_path=np.concatenate([x.true_tvt_path for x in non_empty], axis=0),
        target_mask=np.concatenate([x.target_mask for x in non_empty], axis=0),
        tvt_input_path=np.concatenate([x.tvt_input_path for x in non_empty], axis=0),
        md_path=np.concatenate([x.md_path for x in non_empty], axis=0),
        z_path=np.concatenate([x.z_path for x in non_empty], axis=0),
        horizontal_row_index=np.concatenate([x.horizontal_row_index for x in non_empty], axis=0),
        horizontal_offsets=first.horizontal_offsets,
    )


# %% [markdown]
# ## 7. Full-grid aggregation and candidate-union readout

# %%
def read_candidate_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    union_cfg = get_nested(config, "candidate_union") or {}
    source_value = union_cfg.get("source_cache") or get_nested(config, "data.exp099_train_feature_cache_local")
    if isinstance(source_value, str):
        nested_value = get_nested(config, source_value)
        if nested_value is not None:
            source_value = nested_value
    source_path = Path(str(source_value))
    if not source_path.is_absolute():
        candidates = [Path.cwd() / source_path, paths.root / source_path, source_path]
    else:
        candidates = [source_path]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(input_root.glob(f"**/{source_path.name}")))
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Candidate cache not found: {source_value}")
    id_col = str(union_cfg.get("id_column", "id"))
    target_col = str(union_cfg.get("target_delta_column", "target"))
    last_col = str(union_cfg.get("last_known_tvt_column", "last_known_tvt"))
    distance_col = str(union_cfg.get("distance_column", "md_since"))
    existing = [str(x) for x in union_cfg.get("existing_candidates", [])]
    header = pd.read_csv(resolved, nrows=0).columns.tolist()
    available = [col for col in existing if col in header]
    usecols = [id_col, "well", target_col, last_col, distance_col] + available
    usecols = [col for col in dict.fromkeys(usecols) if col in header]
    cache = pd.read_csv(resolved, usecols=usecols, dtype={id_col: str, "well": str}, low_memory=False)
    cache = cache.rename(columns={id_col: "id", target_col: "target", last_col: "last_known_tvt", distance_col: "md_since"})
    for col in cache.columns:
        if col not in {"id", "well"}:
            cache[col] = pd.to_numeric(cache[col], errors="coerce").astype(np.float32)
    cache["true_tvt"] = cache["last_known_tvt"] + cache["target"]
    cache["row_index"] = cache["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32)
    summary = {
        "source_path": str(resolved),
        "source_sha256": sha256_path(resolved),
        "source_decompressed_sha256": sha256_path(resolved, decompressed=resolved.suffix == ".gz"),
        "rows": int(len(cache)),
        "wells": int(cache["well"].nunique()),
        "available_candidates": available,
    }
    return cache, summary


def triangular_weights(offsets: np.ndarray) -> np.ndarray:
    half = max(float(np.max(np.abs(offsets))), 1.0)
    weights = 1.0 - np.abs(offsets.astype(np.float32)) / (half + 1.0)
    return np.clip(weights, 0.05, 1.0).astype(np.float32)


def fill_missing_by_interp(row_index: np.ndarray, values: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = values.astype(np.float32).copy()
    fallback = ~covered
    if covered.any() and fallback.any():
        result[fallback] = np.interp(
            row_index[fallback].astype(np.float32),
            row_index[covered].astype(np.float32),
            result[covered].astype(np.float32),
        ).astype(np.float32)
    return result, fallback.astype(bool)


def aggregate_full_grid_paths(
    *,
    path_output: CandidatePathOutput,
    sample_frame: pd.DataFrame,
    candidate_cache: pd.DataFrame,
    topk: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_lookup = sample_frame.set_index("sample_id", drop=False)
    sample_meta = sample_lookup.loc[path_output.sample_id].reset_index(drop=True)
    topk = min(int(topk), int(path_output.pred_tvt_path.shape[1]))
    weights_by_offset = triangular_weights(path_output.horizontal_offsets)
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for well, cache_well in candidate_cache.groupby("well", sort=True):
        cache_well = cache_well.sort_values("row_index").reset_index(drop=True)
        row_index = cache_well["row_index"].to_numpy(np.int32)
        max_row = int(max(row_index.max(), 0)) + 1
        sample_mask = sample_meta["well"].to_numpy(str) == str(well)
        sample_indices = np.flatnonzero(sample_mask)
        if len(sample_indices) == 0:
            base = cache_well[["id", "well", "row_index", "md_since", "true_tvt"]].copy()
            for rank in range(1, topk + 1):
                part = base.copy()
                part["path_rank"] = rank
                part["tvt_pred"] = np.nan
                part["path_logit"] = np.nan
                part["path_prob"] = np.nan
                part["weighted_tvt_pred"] = np.nan
                part["source_window_count"] = 0
                part["coverage_flag"] = False
                part["fallback_flag"] = True
                part["candidate_cost"] = np.nan
                frames.append(part)
            continue
        weighted_sum = np.zeros((topk, max_row), dtype=np.float64)
        weight_sum = np.zeros((topk, max_row), dtype=np.float64)
        prob_sum = np.zeros((topk, max_row), dtype=np.float64)
        logit_sum = np.zeros((topk, max_row), dtype=np.float64)
        count = np.zeros((topk, max_row), dtype=np.int32)
        weighted_path_sum = np.zeros(max_row, dtype=np.float64)
        weighted_path_weight = np.zeros(max_row, dtype=np.float64)
        weighted_path_count = np.zeros(max_row, dtype=np.int32)
        for sample_pos in sample_indices:
            rows = path_output.horizontal_row_index[sample_pos].astype(np.int32)
            valid = (rows >= 0) & (rows < max_row)
            if not valid.any():
                continue
            rows_valid = rows[valid]
            base_weight = weights_by_offset[valid].astype(np.float64)
            np.add.at(
                weighted_path_sum,
                rows_valid,
                path_output.weighted_tvt_path[sample_pos, valid].astype(np.float64) * base_weight,
            )
            np.add.at(weighted_path_weight, rows_valid, base_weight)
            np.add.at(weighted_path_count, rows_valid, 1)
            for rank_index in range(topk):
                prob = float(path_output.path_prob[sample_pos, rank_index])
                if not np.isfinite(prob):
                    continue
                weights = base_weight * max(prob, 1e-6)
                np.add.at(
                    weighted_sum[rank_index],
                    rows_valid,
                    path_output.pred_tvt_path[sample_pos, rank_index, valid].astype(np.float64) * weights,
                )
                np.add.at(weight_sum[rank_index], rows_valid, weights)
                np.add.at(prob_sum[rank_index], rows_valid, prob * base_weight)
                np.add.at(logit_sum[rank_index], rows_valid, float(path_output.path_logit[sample_pos, rank_index]) * base_weight)
                np.add.at(count[rank_index], rows_valid, 1)
        weighted_pred = np.divide(
            weighted_path_sum,
            weighted_path_weight,
            out=np.full(max_row, np.nan, dtype=np.float64),
            where=weighted_path_weight > 0,
        ).astype(np.float32)
        weighted_pred, weighted_fallback = fill_missing_by_interp(
            np.arange(max_row, dtype=np.int32),
            weighted_pred,
            weighted_path_weight > 0,
        )
        base = cache_well[["id", "well", "row_index", "md_since", "true_tvt"]].copy()
        for rank_index in range(topk):
            pred = np.divide(
                weighted_sum[rank_index],
                weight_sum[rank_index],
                out=np.full(max_row, np.nan, dtype=np.float64),
                where=weight_sum[rank_index] > 0,
            ).astype(np.float32)
            pred, fallback_all = fill_missing_by_interp(
                np.arange(max_row, dtype=np.int32),
                pred,
                weight_sum[rank_index] > 0,
            )
            prob_mean = np.divide(
                prob_sum[rank_index],
                np.maximum(count[rank_index], 1),
                out=np.full(max_row, np.nan, dtype=np.float64),
                where=count[rank_index] > 0,
            ).astype(np.float32)
            logit_mean = np.divide(
                logit_sum[rank_index],
                np.maximum(count[rank_index], 1),
                out=np.full(max_row, np.nan, dtype=np.float64),
                where=count[rank_index] > 0,
            ).astype(np.float32)
            covered_all = weight_sum[rank_index] > 0
            part = base.copy()
            part["path_rank"] = int(rank_index + 1)
            part["tvt_pred"] = pred[row_index].astype(np.float32)
            part["path_logit"] = logit_mean[row_index].astype(np.float32)
            part["path_prob"] = prob_mean[row_index].astype(np.float32)
            part["weighted_tvt_pred"] = weighted_pred[row_index].astype(np.float32)
            part["source_window_count"] = count[rank_index, row_index].astype(np.int32)
            part["coverage_flag"] = covered_all[row_index].astype(bool)
            part["fallback_flag"] = fallback_all[row_index] | weighted_fallback[row_index]
            part["candidate_cost"] = -np.log(np.maximum(part["path_prob"].to_numpy(np.float32), 1e-6))
            frames.append(part)
            coverage_rows.append(
                {
                    "well": well,
                    "path_rank": int(rank_index + 1),
                    "cache_rows": int(len(row_index)),
                    "source_covered_rows": int(np.sum(covered_all[row_index])),
                    "fallback_rows": int(np.sum(fallback_all[row_index])),
                }
            )
    full_grid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    unique_rows = candidate_cache[["id"]].drop_duplicates()
    covered_unique = full_grid.loc[full_grid["coverage_flag"], ["id"]].drop_duplicates()
    fallback_unique = full_grid.loc[full_grid["fallback_flag"], ["id"]].drop_duplicates()
    summary = {
        "rows": int(len(full_grid)),
        "unique_row_ids": int(full_grid["id"].nunique()) if len(full_grid) else 0,
        "wells": int(full_grid["well"].nunique()) if len(full_grid) else 0,
        "path_ranks": int(full_grid["path_rank"].nunique()) if len(full_grid) else 0,
        "row_coverage_rate_vs_cache": float(len(covered_unique) / max(len(unique_rows), 1)),
        "fallback_unique_row_rate": float(len(fallback_unique) / max(len(unique_rows), 1)),
        "duplicate_key_rows": int(full_grid.duplicated(["id", "path_rank"]).sum()) if len(full_grid) else 0,
        "null_required_value_count": int(
            full_grid[["id", "well", "path_rank", "tvt_pred"]].isna().sum().sum()
        ) if len(full_grid) else 0,
        "coverage_by_well_rank": coverage_rows,
    }
    return full_grid, summary


def oracle_metrics(errors: np.ndarray, within_ft: float) -> dict[str, float]:
    finite = errors[np.isfinite(errors)]
    if len(finite) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "within": float("nan")}
    return {
        "rmse": float(math.sqrt(float(np.mean(np.square(finite))))),
        "mae": float(np.mean(np.abs(finite))),
        "within": float(np.mean(finite <= within_ft)),
    }


def min_abs_error(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmin(np.abs(values.astype(np.float32) - truth[:, None].astype(np.float32)), axis=1)


def evaluate_candidate_union(
    *,
    full_grid: pd.DataFrame,
    candidate_cache: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    union_cfg = get_nested(config, "candidate_union") or {}
    within_ft = float(union_cfg.get("within_ft", 10.0))
    existing = [col for col in union_cfg.get("existing_candidates", []) if col in candidate_cache.columns]
    if full_grid.empty or not existing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "missing_inputs"}
    learned_wide = full_grid.pivot(index="id", columns="path_rank", values="tvt_pred")
    learned_wide.columns = [f"mtp_rank{int(col)}" for col in learned_wide.columns]
    weighted = full_grid.sort_values("path_rank").drop_duplicates("id")[["id", "weighted_tvt_pred", "coverage_flag", "fallback_flag"]]
    merged = candidate_cache.merge(learned_wide.reset_index(), on="id", how="inner").merge(weighted, on="id", how="left")
    truth = merged["true_tvt"].to_numpy(np.float32)
    existing_error = min_abs_error(merged[existing].to_numpy(np.float32), truth)
    learned_cols = [col for col in merged.columns if col.startswith("mtp_rank")]
    learned_error = min_abs_error(merged[learned_cols].to_numpy(np.float32), truth)
    weighted_error = np.abs(merged["weighted_tvt_pred"].to_numpy(np.float32) - truth)
    union_error = np.minimum(existing_error, learned_error)
    metric_rows = []
    for name, error, count in [
        ("existing_union", existing_error, len(existing)),
        ("learned_mtp_topk", learned_error, len(learned_cols)),
        ("learned_mtp_weighted", weighted_error, 1),
        ("existing_plus_learned_mtp_topk", union_error, len(existing) + len(learned_cols)),
    ]:
        metrics = oracle_metrics(error, within_ft)
        metric_rows.append(
            {
                "candidate_set": name,
                "rows": int(len(merged)),
                "candidate_count": int(count),
                "oracle_rmse": metrics["rmse"],
                "oracle_mae": metrics["mae"],
                "within10": metrics["within"],
                "new_best_candidate_rate": float(np.mean(learned_error + 1e-6 < existing_error))
                if "learned" in name or "plus" in name
                else 0.0,
                "oracle_rmse_delta_vs_existing": float(metrics["rmse"] - oracle_metrics(existing_error, within_ft)["rmse"]),
                "within_delta_vs_existing": float(metrics["within"] - oracle_metrics(existing_error, within_ft)["within"]),
            }
        )
    merged = merged.assign(existing_error=existing_error, learned_error=learned_error, union_error=union_error)
    by_well_rows = []
    for well, group in merged.groupby("well", dropna=False):
        base = oracle_metrics(group["existing_error"].to_numpy(np.float32), within_ft)
        plus = oracle_metrics(group["union_error"].to_numpy(np.float32), within_ft)
        by_well_rows.append(
            {
                "well": well,
                "rows": int(len(group)),
                "existing_oracle_rmse": base["rmse"],
                "existing_plus_learned_oracle_rmse": plus["rmse"],
                "oracle_rmse_delta": float(plus["rmse"] - base["rmse"]),
                "fallback_row_rate": float(group["fallback_flag"].astype(float).mean()),
            }
        )
    bucket_rows = []
    buckets = union_cfg.get("distance_buckets") or [[0, 50], [50, 100], [100, 250], [250, 500], [500, 1000], [1000, 1000000000]]

    def bucket_label(value: float) -> str:
        abs_value = abs(float(value))
        for lower, upper in buckets:
            if float(lower) <= abs_value < float(upper):
                return f"{int(lower)}_plus" if float(upper) >= 1e8 else f"{int(lower)}_{int(upper)}"
        return "unknown"

    merged["distance_bucket"] = merged["md_since"].map(bucket_label)
    for bucket, group in merged.groupby("distance_bucket", dropna=False):
        base = oracle_metrics(group["existing_error"].to_numpy(np.float32), within_ft)
        plus = oracle_metrics(group["union_error"].to_numpy(np.float32), within_ft)
        bucket_rows.append(
            {
                "distance_bucket": bucket,
                "rows": int(len(group)),
                "existing_oracle_rmse": base["rmse"],
                "existing_plus_learned_oracle_rmse": plus["rmse"],
                "oracle_rmse_delta": float(plus["rmse"] - base["rmse"]),
                "fallback_row_rate": float(group["fallback_flag"].astype(float).mean()),
            }
        )
    summary = {
        "status": "ok",
        "rows_joined": int(len(merged)),
        "existing_candidates": existing,
        "learned_candidates": learned_cols,
        "fallback_row_rate": float(merged["fallback_flag"].astype(float).mean()),
        "source_covered_row_rate": float(merged["coverage_flag"].astype(float).mean()),
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(by_well_rows), pd.DataFrame(bucket_rows), summary


def path_step_mean(paths_array: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(paths_array.astype(np.float32), axis=2))
    return np.nanmean(diffs, axis=2)


# %% [markdown]
# ## 8. Setup and execution

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

run_specs = resolve_run_specs(config)
run_plan = expand_run_plan(run_specs)
run_plan_df = pd.DataFrame([asdict(item) for item in run_plan])
print("Run plan:")
display(run_plan_df)
print("Training config:")
print(json.dumps(to_jsonable(get_nested(config, "model.training") or {}), indent=2, sort_keys=True))

# %%
training_cfg = get_nested(config, "model.training") or {}
max_wells = training_cfg.get("max_wells")
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
    raise RuntimeError("Not enough usable wells after raw file loading.")

sample_frames: list[pd.DataFrame] = []
fold_well_rows: list[dict[str, Any]] = []
sample_id_start = 0
for plan_item in run_plan:
    train_wells, valid_wells = split_wells(usable_wells, config, fold_index=plan_item.fold_index)
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
        raise RuntimeError(f"Sample index empty for fold {plan_item.fold_index}")
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
display(sample_overview)
display(pd.DataFrame(fold_well_rows))
display(sample_index.head())

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
        f"variant={plan_item.variant} ==="
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
dense_predictions_df = pd.concat(prediction_frames, ignore_index=True)
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
dense_predictions_df = dense_predictions_df.merge(
    sample_index[sample_columns],
    on="sample_id",
    how="left",
    suffixes=("", "_sample"),
)
candidate_path_output = stack_path_outputs(path_outputs)
candidate_cache, candidate_cache_summary = read_candidate_cache(config)
full_grid_topk = int((get_nested(config, "full_grid") or {}).get("primary_topk", 5))
full_grid_df, full_grid_summary = aggregate_full_grid_paths(
    path_output=candidate_path_output,
    sample_frame=dense_predictions_df,
    candidate_cache=candidate_cache,
    topk=full_grid_topk,
)
candidate_union_metrics_df, candidate_union_by_well_df, candidate_union_distance_df, candidate_union_summary = evaluate_candidate_union(
    full_grid=full_grid_df,
    candidate_cache=candidate_cache,
    config=config,
)

aggregate_rows: list[dict[str, Any]] = []
for keys, group in fold_metrics_df.groupby(["run_spec", "variant", "channel_set"], dropna=False):
    run_spec, variant, channel_set = keys
    weights = group["dense_samples"].astype(float).to_numpy()
    row: dict[str, Any] = {
        "run_spec": run_spec,
        "variant": variant,
        "channel_set": channel_set,
        "folds_completed": int(group["fold_index"].nunique()),
        "train_samples": int(group["train_samples"].sum()),
        "valid_samples": int(group["valid_samples"].sum()),
        "dense_samples": int(group["dense_samples"].sum()),
        "elapsed_sec": float(group["elapsed_sec"].sum()),
    }
    for column in [
        "sparse_weighted_center_rmse",
        "sparse_rank1_center_rmse",
        "sparse_top10_oracle_center_rmse",
        "dense_weighted_center_rmse",
        "dense_rank1_center_rmse",
        "dense_top10_oracle_center_rmse",
        "dense_weighted_path_rmse",
        "dense_rank1_path_rmse",
    ]:
        values = group[column].astype(float).to_numpy()
        finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        row[column] = float(np.average(values[finite], weights=weights[finite])) if finite.any() else np.nan
    aggregate_rows.append(row)
metrics_df = pd.DataFrame(aggregate_rows)

path_count, path_topk, path_horizon = candidate_path_output.pred_tvt_path.shape
candidate_path_sample_df = pd.DataFrame(
    {
        "path_npz_sample_index": np.arange(len(candidate_path_output.sample_id), dtype=np.int64),
        "sample_id": candidate_path_output.sample_id.astype(np.int64),
    }
).merge(
    dense_predictions_df[
        [
            "sample_id",
            "id",
            "split",
            "well",
            "fold_index",
            "row_center",
            "prefix_end",
            "last_known_tvt",
            "prior_center_tvt",
            "true_center_tvt",
            "md_since_prefix",
            "z_since_prefix",
            "path_prob_entropy",
            "path_prob_top3_mass",
            "path_prob_top5_mass",
        ]
    ],
    on="sample_id",
    how="left",
)
rank_index = np.tile(np.arange(1, path_topk + 1, dtype=np.int16), path_count)
sample_index_repeated = np.repeat(np.arange(path_count, dtype=np.int64), path_topk)
sample_id_repeated = np.repeat(candidate_path_output.sample_id.astype(np.int64), path_topk)
step_mean = path_step_mean(candidate_path_output.pred_tvt_path)
candidate_path_rank_df = pd.DataFrame(
    {
        "path_npz_sample_index": sample_index_repeated,
        "sample_id": sample_id_repeated,
        "rank": rank_index.astype(np.int16),
        "mode_index": candidate_path_output.mode_index.reshape(-1).astype(np.int16),
        "center_bin": candidate_path_output.center_bin.reshape(-1).astype(np.int16),
        "center_pred_tvt": candidate_path_output.center_tvt.reshape(-1).astype(np.float32),
        "path_logit": candidate_path_output.path_logit.reshape(-1).astype(np.float32),
        "path_prob": candidate_path_output.path_prob.reshape(-1).astype(np.float32),
        "path_step_abs_mean_ft": step_mean.reshape(-1).astype(np.float32),
    }
).merge(
    candidate_path_sample_df[["path_npz_sample_index", "id", "well", "fold_index", "row_center"]],
    on="path_npz_sample_index",
    how="left",
)

metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv"
fold_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv"
predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_dense_validation_predictions.csv.gz"
full_grid_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_full_grid_candidate_paths.csv.gz"
candidate_path_npz_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_window_candidate_paths_top10.npz"
candidate_path_samples_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_window_path_samples.csv.gz"
candidate_path_rank_index_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_window_path_rank_index.csv.gz"
candidate_union_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_metrics.csv"
candidate_union_by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_by_well.csv"
candidate_union_distance_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_union_distance_bucket_metrics.csv"
history_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_training_history.csv"
schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
run_spec_manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_run_spec_manifest.json"
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"

metrics_df.to_csv(metrics_path, index=False)
fold_metrics_df.to_csv(fold_metrics_path, index=False)
gzip_csv(dense_predictions_df, predictions_path)
gzip_csv(full_grid_df, full_grid_path)
gzip_csv(candidate_path_sample_df, candidate_path_samples_path)
gzip_csv(candidate_path_rank_df, candidate_path_rank_index_path)
candidate_union_metrics_df.to_csv(candidate_union_metrics_path, index=False)
candidate_union_by_well_df.to_csv(candidate_union_by_well_path, index=False)
candidate_union_distance_df.to_csv(candidate_union_distance_path, index=False)
history_df.to_csv(history_path, index=False)
pd.DataFrame(
    [
        {"channel_index": idx, "channel": name, "description": desc}
        for idx, (name, desc) in enumerate(channel_schema_for("base"))
    ]
).to_csv(schema_path, index=False)
write_json(
    run_spec_manifest_path,
    {
        "experiment": EXPERIMENT_NAME,
        "run_specs": [asdict(spec) for spec in run_specs],
        "run_plan": [asdict(item) for item in run_plan],
        "cnn_model_count": len(run_plan),
    },
)
write_json(manifest_path, model_manifest)
np.savez_compressed(
    candidate_path_npz_path,
    sample_id=candidate_path_output.sample_id,
    mode_index=candidate_path_output.mode_index,
    center_bin=candidate_path_output.center_bin,
    center_tvt=candidate_path_output.center_tvt,
    path_logit=candidate_path_output.path_logit,
    path_prob=candidate_path_output.path_prob,
    pred_tvt_path=candidate_path_output.pred_tvt_path,
    pred_bin_path=candidate_path_output.pred_bin_path,
    weighted_tvt_path=candidate_path_output.weighted_tvt_path,
    true_tvt_path=candidate_path_output.true_tvt_path,
    target_mask=candidate_path_output.target_mask,
    tvt_input_path=candidate_path_output.tvt_input_path,
    md_path=candidate_path_output.md_path,
    z_path=candidate_path_output.z_path,
    horizontal_row_index=candidate_path_output.horizontal_row_index,
    horizontal_offsets=candidate_path_output.horizontal_offsets,
)

artifact_sha = {
    "sample_index_csv_decompressed_sha256": sha256_path(sample_index_path, decompressed=True),
    "dense_predictions_csv_decompressed_sha256": sha256_path(predictions_path, decompressed=True),
    "full_grid_candidate_paths_csv_decompressed_sha256": sha256_path(full_grid_path, decompressed=True),
    "window_candidate_paths_npz_sha256": sha256_path(candidate_path_npz_path),
    "window_path_samples_csv_decompressed_sha256": sha256_path(candidate_path_samples_path, decompressed=True),
    "window_path_rank_index_csv_decompressed_sha256": sha256_path(candidate_path_rank_index_path, decompressed=True),
    "candidate_union_metrics_csv_sha256": sha256_path(candidate_union_metrics_path),
    "candidate_union_by_well_csv_sha256": sha256_path(candidate_union_by_well_path),
    "candidate_union_distance_bucket_metrics_csv_sha256": sha256_path(candidate_union_distance_path),
    "metrics_csv_sha256": sha256_path(metrics_path),
    "fold_metrics_csv_sha256": sha256_path(fold_metrics_path),
    "training_history_csv_sha256": sha256_path(history_path),
    "feature_schema_csv_sha256": sha256_path(schema_path),
    "run_spec_manifest_json_sha256": sha256_path(run_spec_manifest_path),
    "model_manifest_json_sha256": sha256_path(manifest_path),
}
key_metrics: dict[str, Any] = {
    "full_grid_source_coverage": full_grid_summary.get("row_coverage_rate_vs_cache"),
    "full_grid_fallback_unique_row_rate": full_grid_summary.get("fallback_unique_row_rate"),
    "candidate_union_status": candidate_union_summary.get("status"),
}
if not candidate_union_metrics_df.empty:
    for candidate_set in [
        "existing_union",
        "learned_mtp_topk",
        "learned_mtp_weighted",
        "existing_plus_learned_mtp_topk",
    ]:
        row = candidate_union_metrics_df.loc[candidate_union_metrics_df["candidate_set"] == candidate_set]
        if row.empty:
            continue
        first = row.iloc[0]
        key_metrics[f"{candidate_set}_oracle_rmse"] = float(first["oracle_rmse"])
        key_metrics[f"{candidate_set}_within10"] = float(first["within10"])
        key_metrics[f"{candidate_set}_oracle_rmse_delta_vs_existing"] = float(first["oracle_rmse_delta_vs_existing"])

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "implemented_train_side_gpu_probe_not_run_locally",
    "created_at": datetime.now(UTC).isoformat(),
    "seed": seed,
    "device": str(device),
    "torch_version": torch.__version__,
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "wells": {"all_selected": len(all_wells), "loaded": len(arrays_by_well), "usable": len(usable_wells)},
    "sample_overview": sample_overview.to_dict(orient="records"),
    "run_plan": run_plan_df.to_dict(orient="records"),
    "metrics": metrics_df.to_dict(orient="records"),
    "fold_metrics": fold_metrics_df.to_dict(orient="records"),
    "candidate_cache": candidate_cache_summary,
    "full_grid": full_grid_summary,
    "candidate_union": candidate_union_summary,
    "candidate_union_metrics": candidate_union_metrics_df.to_dict(orient="records"),
    "key_metrics": key_metrics,
    "artifact_sha": artifact_sha,
    "model_manifest": model_manifest,
    "reproducibility": {
        "deterministic_anchor": False,
        "torch_deterministic_algorithms": True,
        "num_workers": 0,
        "uses_input_history_sdf_channel": True,
        "trains_sdf_head_or_sdf_loss": False,
    },
}
write_json(summary_path, summary)
artifact_sha["summary_json_sha256"] = sha256_path(summary_path)
summary["artifact_sha"] = artifact_sha
write_json(summary_path, summary)

metrics_json = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_gpu_probe_supported_no_submit",
    "cv": key_metrics.get("existing_plus_learned_mtp_topk_oracle_rmse"),
    "public_lb": None,
    "private_lb": None,
    "metric": "existing_plus_learned_mtp_topk_oracle_rmse",
    "key_idea": (
        "Learned MTP full-tail heatmap path generator fixed exp212 fallback "
        "coverage and improved existing PF/Beam candidate-union oracle headroom, "
        "but learned-only weighted path is too weak for direct replacement."
    ),
    "summary": {
        "metrics": metrics_df.to_dict(orient="records"),
        "fold_metrics": fold_metrics_df.to_dict(orient="records"),
        "candidate_union_metrics": candidate_union_metrics_df.to_dict(orient="records"),
        "key_metrics": key_metrics,
        "full_grid": full_grid_summary,
        "candidate_union": candidate_union_summary,
        "artifact_sha": artifact_sha,
    },
    "notes": "Train-side GPU diagnostic only; no inference or submission.",
}
paths.metrics_path.write_text(json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n")

display(metrics_df)
display(fold_metrics_df)
display(candidate_union_metrics_df)
display(candidate_union_distance_df)
display(candidate_path_sample_df.head(10))
display(candidate_path_rank_df.head(20))
display(full_grid_df.head(20))
display(history_df.tail(10))
print("Saved summary:", summary_path)
print(json.dumps(to_jsonable(key_metrics), indent=2, sort_keys=True))
print(json.dumps(to_jsonable(full_grid_summary), indent=2, sort_keys=True))
print(json.dumps(to_jsonable(candidate_union_summary), indent=2, sort_keys=True))
print(json.dumps(to_jsonable(artifact_sha), indent=2, sort_keys=True))
