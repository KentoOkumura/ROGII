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
# # exp179 cnn sdf mtp heatmap probe train
#
# Train-side GPU diagnostic for the discussion-699853 5-channel CNN/SDF/MTP
# heatmap idea. This notebook does not create an inference branch or
# submission. It asks one narrow question: does real GR improve topK path
# coverage over shuffled-GR and no-GR controls when window centers are
# target-free?

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and reproducibility helpers
# 3. Well loading and fold-safe sample index helpers
# 4. Heatmap dataset
# 5. CNN/MTP model and training helpers
# 6. Setup and input checks
# 7. Run GPU variants
# 8. Metrics, SHA, and generated artifacts

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
CHANNEL_SCHEMA = [
    ("typewell_gr_heatmap", "Typewell GR sampled at the target-free TVT grid."),
    ("horizontal_gr_heatmap", "Horizontal GR sampled at the horizontal row window."),
    ("typewell_minus_horizontal_gr", "Pairwise GR difference heatmap."),
    (
        "tvt_history_sdf_from_observed_tvt_input_prefix",
        "Target-free SDF history: grid_tvt - observed TVT_input where prefix is known.",
    ),
    ("observed_tvt_input_mask", "1 where TVT_input is observed in the horizontal window."),
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


def finite_float_array(series: pd.Series, fallback: float = 0.0) -> np.ndarray:
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
        scale = float(np.std(finite)) if float(np.std(finite)) > 1e-6 else 1.0
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


def gzip_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


# %% [markdown]
# ## 3. Well loading and fold-safe sample index helpers

# %%
@dataclass(frozen=True)
class WellArrays:
    well: str
    horizontal_rows: int
    typewell_rows: int
    md: np.ndarray
    z: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    horizontal_gr: np.ndarray
    horizontal_gr_missing: np.ndarray
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
    split: str
    well: str
    row_center: int
    prefix_end: int
    last_known_tvt: float
    prior_center_tvt: float
    true_center_tvt: float
    md_since_prefix: float
    z_since_prefix: float
    center_target_in_grid: bool
    label_fraction: float


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
    z = finite_float_array(h["Z"])
    tvt = finite_float_array(h["TVT"])
    tvt_input = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(np.float32)

    return WellArrays(
        well=well,
        horizontal_rows=len(h),
        typewell_rows=len(t),
        md=md,
        z=z,
        tvt=tvt,
        tvt_input=tvt_input,
        horizontal_gr=horizontal_gr,
        horizontal_gr_missing=horizontal_gr_missing,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        typewell_gr_shuffled=typewell_gr_shuffled,
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
    target_idx, target_distance = nearest_grid_indices(grid_tvt, arrays.tvt[h_idx])
    center_position = int(np.flatnonzero(horizontal_offsets == 0)[0])
    center_distance = float(target_distance[center_position])
    center_target_in_grid = bool(center_distance <= float(target_tolerance))
    label_fraction = float(np.mean(target_distance <= float(target_tolerance)))
    true_center_tvt = float(arrays.tvt[row_center])
    return prior_center, true_center_tvt, center_target_in_grid, label_fraction


def build_sample_index(
    *,
    arrays_by_well: dict[str, WellArrays],
    train_wells: list[str],
    valid_wells: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    training = get_nested(config, "model.training") or {}
    horizontal_window = int(training.get("horizontal_window_rows", 128))
    horizontal_offsets = np.arange(-(horizontal_window // 2), horizontal_window // 2, dtype=np.int32)
    grid_bins = int(training.get("typewell_window_bins", 64))
    grid_half_width = float(training.get("tvt_grid_half_width_ft", 192.0))
    grid_offsets_tvt = np.linspace(-grid_half_width, grid_half_width, grid_bins).astype(np.float32)
    target_tolerance = float(training.get("center_target_tolerance_ft", 10.0))
    min_label_fraction = float(training.get("min_label_fraction", 0.35))
    max_tail_rows = int(training.get("max_tail_rows", 2048))

    sample_rows: list[dict[str, Any]] = []
    sample_id = 0
    split_specs = [
        ("train", train_wells, int(training.get("train_samples_per_well", 24)), True),
        ("valid", valid_wells, int(training.get("valid_samples_per_well", 16)), False),
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
                    split=split,
                    well=well,
                    row_center=int(row_center),
                    prefix_end=arrays.prefix_end,
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


def split_wells(wells: list[str], config: dict[str, Any]) -> tuple[list[str], list[str]]:
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    fold_index = int(get_nested(config, "validation.fold_index") or 0)
    groups = np.asarray(wells)
    dummy_x = np.zeros((len(wells), 1), dtype=np.float32)
    dummy_y = np.zeros(len(wells), dtype=np.float32)
    splits = list(GroupKFold(n_splits=n_folds).split(dummy_x, dummy_y, groups=groups))
    train_idx, valid_idx = splits[fold_index]
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


# %% [markdown]
# ## 4. Heatmap dataset

# %%
class HeatmapWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        sample_index: pd.DataFrame,
        arrays_by_well: dict[str, WellArrays],
        config: dict[str, Any],
        split: str,
        variant: str,
    ) -> None:
        self.sample_index = sample_index.loc[sample_index["split"] == split].reset_index(drop=True)
        self.arrays_by_well = arrays_by_well
        self.variant = variant
        training = get_nested(config, "model.training") or {}
        horizontal_window = int(training.get("horizontal_window_rows", 128))
        self.horizontal_offsets = np.arange(
            -(horizontal_window // 2),
            horizontal_window // 2,
            dtype=np.int32,
        )
        grid_bins = int(training.get("typewell_window_bins", 64))
        grid_half_width = float(training.get("tvt_grid_half_width_ft", 192.0))
        self.grid_offsets_tvt = np.linspace(-grid_half_width, grid_half_width, grid_bins).astype(
            np.float32
        )
        self.history_scale_ft = float(training.get("history_scale_ft", 200.0))
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
        history = history * mask.reshape(-1, 1)
        mask_heatmap = np.broadcast_to(mask.reshape(-1, 1), history.shape)

        image = np.stack(
            [
                t_heatmap,
                h_heatmap,
                diff,
                history,
                mask_heatmap,
            ],
            axis=0,
        ).astype(np.float32)

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
        }


# %% [markdown]
# ## 5. CNN/MTP model and training helpers

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
        current = in_channels
        padding = kernel_size // 2
        for width in channels:
            layers.append(nn.Conv2d(current, int(width), kernel_size=kernel_size, padding=padding))
            if use_group_norm:
                groups = 8 if int(width) % 8 == 0 else 1
                layers.append(nn.GroupNorm(groups, int(width)))
            else:
                layers.append(nn.BatchNorm2d(int(width)))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            current = int(width)
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


def make_model(config: dict[str, Any]) -> HeatmapMTPNet:
    arch = get_nested(config, "model.architecture") or {}
    return HeatmapMTPNet(
        in_channels=len(CHANNEL_SCHEMA),
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
        batch_size * path_modes, h_size
    )
    flat_mask = target_mask[:, None, :].expand(batch_size, path_modes, h_size).reshape(
        batch_size * path_modes, h_size
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


@torch.no_grad()
def evaluate_model(
    *,
    model: HeatmapMTPNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    variant: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    center_position = loader.dataset.center_position  # type: ignore[attr-defined]

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_idx = batch["target_idx"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        path_logits, mode_logits = model(image)
        loss, _ = closest_mode_loss(path_logits, mode_logits, target_idx, target_mask)
        losses.append(float(loss.detach().cpu()))

        mode_prob = torch.softmax(mode_logits, dim=1)
        center_logits = path_logits[:, :, center_position, :]
        path_prob = torch.softmax(center_logits, dim=2)
        path_score, path_idx = path_prob.max(dim=2)
        combined_score = mode_prob * path_score
        order = combined_score.argsort(dim=1, descending=True)

        grid_tvt = batch["grid_tvt"].cpu().numpy()
        true_tvt = batch["true_center_tvt"].cpu().numpy()
        sample_ids = batch["sample_id"].cpu().numpy()
        target_in_grid = batch["center_target_in_grid"].cpu().numpy().astype(bool)
        path_idx_np = path_idx.cpu().numpy()
        order_np = order.cpu().numpy()
        score_np = combined_score.cpu().numpy()

        for row_index in range(len(sample_ids)):
            candidate_tvts: list[float] = []
            candidate_scores: list[float] = []
            seen_pairs: set[int] = set()
            for mode_index in order_np[row_index].tolist():
                pred_idx = int(path_idx_np[row_index, mode_index])
                if pred_idx in seen_pairs:
                    continue
                seen_pairs.add(pred_idx)
                candidate_tvts.append(float(grid_tvt[row_index, pred_idx]))
                candidate_scores.append(float(score_np[row_index, mode_index]))
                if len(candidate_tvts) >= max(TOPK_VALUES):
                    break
            while len(candidate_tvts) < max(TOPK_VALUES):
                candidate_tvts.append(float("nan"))
                candidate_scores.append(float("nan"))

            errors = np.abs(np.asarray(candidate_tvts, dtype=np.float32) - float(true_tvt[row_index]))
            record: dict[str, Any] = {
                "variant": variant,
                "sample_id": int(sample_ids[row_index]),
                "true_center_tvt": float(true_tvt[row_index]),
                "target_in_grid": bool(target_in_grid[row_index]),
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

    predictions = pd.DataFrame(prediction_rows)
    metrics: dict[str, Any] = {
        "variant": variant,
        "loss": float(np.mean(losses)) if losses else np.nan,
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
    return metrics, predictions


def train_variant(
    *,
    variant: str,
    arrays_by_well: dict[str, WellArrays],
    sample_index: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path]:
    seed = int(get_nested(config, "reproducibility.seed") or 42)
    training = get_nested(config, "model.training") or {}
    batch_size = int(get_nested(config, "runtime.batch_size") or 32)
    epochs = int(training.get("epochs", 5))

    train_dataset = HeatmapWindowDataset(
        sample_index=sample_index,
        arrays_by_well=arrays_by_well,
        config=config,
        split="train",
        variant=variant,
    )
    valid_dataset = HeatmapWindowDataset(
        sample_index=sample_index,
        arrays_by_well=arrays_by_well,
        config=config,
        split="valid",
        variant=variant,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(training.get("dataloader_shuffle", True)),
        seed=stable_int(EXPERIMENT_NAME, variant, "train-loader", str(seed), modulo=2**31 - 1),
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=stable_int(EXPERIMENT_NAME, variant, "valid-loader", str(seed), modulo=2**31 - 1),
    )

    model = make_model(config).to(device)
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

        valid_metrics, _ = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            variant=variant,
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
                "variant": variant,
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
            f"{variant} epoch {epoch}/{epochs}: "
            f"train_loss={history_rows[-1]['train_loss']:.4f} "
            f"valid_top3={valid_top3:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, predictions = evaluate_model(
        model=model,
        loader=valid_loader,
        device=device,
        variant=variant,
    )
    final_metrics["best_valid_top3_within10_center_rate"] = float(best_metric)
    final_metrics["train_samples"] = int(len(train_dataset))
    final_metrics["valid_samples"] = int(len(valid_dataset))
    final_metrics["epochs"] = epochs
    final_metrics["elapsed_sec"] = float(time.time() - start_time)

    model_path = output_dir / f"{OUTPUT_PREFIX}_{variant}_model.pt"
    torch.save(
        {
            "experiment": EXPERIMENT_NAME,
            "variant": variant,
            "state_dict": model.state_dict(),
            "config_model": get_nested(config, "model"),
            "metrics": final_metrics,
        },
        model_path,
    )
    return final_metrics, predictions, pd.DataFrame(history_rows), model_path


# %% [markdown]
# ## 6. Setup and input checks

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
active_variants = list(get_nested(config, "model.active_variants") or [])
print("Active variants:", active_variants)
print("Training config:", json.dumps(to_jsonable(training_config), indent=2, sort_keys=True))

# %%
max_wells = training_config.get("max_wells")
all_wells = list_train_wells(paths.train_data_dir, int(max_wells) if max_wells is not None else None)
train_wells, valid_wells = split_wells(all_wells, config)

arrays_by_well: dict[str, WellArrays] = {}
for well in sorted(set(train_wells + valid_wells)):
    arrays = read_well_arrays(well, paths.train_data_dir, seed)
    if arrays is not None:
        arrays_by_well[well] = arrays

train_wells = [well for well in train_wells if well in arrays_by_well]
valid_wells = [well for well in valid_wells if well in arrays_by_well]
if not train_wells or not valid_wells:
    raise RuntimeError("No train/valid wells were usable for exp179.")

sample_index = build_sample_index(
    arrays_by_well=arrays_by_well,
    train_wells=train_wells,
    valid_wells=valid_wells,
    config=config,
)
if sample_index.empty:
    raise RuntimeError("Sample index is empty.")

sample_index_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_sample_index.csv.gz"
gzip_csv(sample_index, sample_index_path)

sample_overview = (
    sample_index.groupby("split")
    .agg(
        samples=("sample_id", "count"),
        wells=("well", "nunique"),
        target_in_grid_rate=("center_target_in_grid", "mean"),
        label_fraction_mean=("label_fraction", "mean"),
        md_since_prefix_mean=("md_since_prefix", "mean"),
    )
    .reset_index()
)
display(sample_overview)
display(sample_index.head())

# %% [markdown]
# ## 7. Run GPU variants

# %%
metrics_rows: list[dict[str, Any]] = []
prediction_frames: list[pd.DataFrame] = []
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

for variant in active_variants:
    if variant not in {"real_gr", "shuffled_gr", "no_gr"}:
        raise ValueError(f"Unexpected variant: {variant}")
    print(f"=== Training variant: {variant} ===")
    metrics, predictions, history, model_path = train_variant(
        variant=variant,
        arrays_by_well=arrays_by_well,
        sample_index=sample_index,
        config=config,
        device=device,
        output_dir=paths.artifacts_dir,
    )
    metrics_rows.append(metrics)
    prediction_frames.append(predictions)
    history_frames.append(history)
    model_manifest["models"][variant] = {
        "path": str(model_path),
        "sha256": sha256_path(model_path),
        "bytes": model_path.stat().st_size,
        "metrics": to_jsonable(metrics),
    }
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))

# %% [markdown]
# ## 8. Metrics, SHA, and generated artifacts

# %%
metrics_df = pd.DataFrame(metrics_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)
history_df = pd.concat(history_frames, ignore_index=True)

sample_columns = [
    "sample_id",
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

metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv"
predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_validation_predictions.csv.gz"
history_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_training_history.csv"
schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"

metrics_df.to_csv(metrics_path, index=False)
gzip_csv(predictions_df, predictions_path)
history_df.to_csv(history_path, index=False)
pd.DataFrame(
    [
        {"channel_index": index, "channel": channel, "description": description}
        for index, (channel, description) in enumerate(CHANNEL_SCHEMA)
    ]
).to_csv(schema_path, index=False)
manifest_path.write_text(json.dumps(to_jsonable(model_manifest), indent=2, sort_keys=True) + "\n")

if {"real_gr", "shuffled_gr", "no_gr"}.issubset(set(metrics_df["variant"])):
    by_variant = metrics_df.set_index("variant")
    real_top3 = float(by_variant.loc["real_gr", "top3_within10_center_rate"])
    shuffled_top3 = float(by_variant.loc["shuffled_gr", "top3_within10_center_rate"])
    no_gr_top3 = float(by_variant.loc["no_gr", "top3_within10_center_rate"])
    metrics_df["real_minus_shuffled_top3_margin"] = real_top3 - shuffled_top3
    metrics_df["real_minus_no_gr_top3_margin"] = real_top3 - no_gr_top3
    metrics_df.to_csv(metrics_path, index=False)

artifact_sha = {
    "sample_index_csv_gz_sha256": sha256_path(sample_index_path),
    "sample_index_csv_decompressed_sha256": sha256_path(sample_index_path, decompressed=True),
    "validation_predictions_csv_gz_sha256": sha256_path(predictions_path),
    "validation_predictions_csv_decompressed_sha256": sha256_path(
        predictions_path,
        decompressed=True,
    ),
    "metrics_csv_sha256": sha256_path(metrics_path),
    "training_history_csv_sha256": sha256_path(history_path),
    "feature_schema_csv_sha256": sha256_path(schema_path),
    "model_manifest_json_sha256": sha256_path(manifest_path),
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
        "train": len(train_wells),
        "valid": len(valid_wells),
    },
    "sample_overview": sample_overview.to_dict(orient="records"),
    "active_variants": active_variants,
    "metrics": metrics_df.to_dict(orient="records"),
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
    "metric": "top3_within10_center_rate",
    "summary": {
        "metrics": metrics_df.to_dict(orient="records"),
        "artifact_sha": artifact_sha,
        "cuda_device_name": summary["cuda_device_name"],
    },
    "notes": "Train-side GPU diagnostic only; no submission.",
}
paths.metrics_path.write_text(json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n")

display(metrics_df)
display(history_df.tail(10))
print("Saved summary:", summary_path)
print(json.dumps(to_jsonable(artifact_sha), indent=2, sort_keys=True))
