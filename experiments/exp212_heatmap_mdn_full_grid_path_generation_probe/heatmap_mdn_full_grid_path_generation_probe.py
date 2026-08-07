from __future__ import annotations

import gzip
import hashlib
import json
import math
from types import SimpleNamespace
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    class _NoGrad:
        def __call__(self, func: Any | None = None) -> Any:
            if func is None:
                return self
            return func

        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> bool:
            return False

    class Dataset:  # type: ignore[no-redef]
        @classmethod
        def __class_getitem__(cls, item: Any) -> type["Dataset"]:
            return cls

    DataLoader = None  # type: ignore[assignment]
    nn = SimpleNamespace(Module=object)  # type: ignore[assignment]
    torch = SimpleNamespace(Tensor=object, no_grad=_NoGrad)  # type: ignore[assignment]

from settings import (
    EXPERIMENT_NAME,
    KAGGLE_INPUT_ROOT,
    ROOT,
    ExperimentPaths,
    get_nested,
    load_config,
)


OUTPUT_PREFIX = EXPERIMENT_NAME
DEFAULT_EXP202_PREFIX = "exp202_heatmap_mdn_candidate_generator_probe"
DEFAULT_EXP202_MODEL_MANIFEST = f"{DEFAULT_EXP202_PREFIX}_model_manifest.json"
DEFAULT_EXP208_PREFIX = "exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe"
DEFAULT_PATH_NPZ = f"{DEFAULT_EXP208_PREFIX}_dense_candidate_paths_top10.npz"
DEFAULT_PATH_SAMPLES = f"{DEFAULT_EXP208_PREFIX}_dense_path_samples.csv.gz"
DEFAULT_EXP099_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)

BASE_CHANNEL_SCHEMA = [
    ("typewell_gr_heatmap", "Typewell GR sampled at the target-free TVT grid."),
    ("horizontal_gr_heatmap", "Horizontal GR sampled at the horizontal row window."),
    ("typewell_minus_horizontal_gr", "Pairwise GR difference heatmap."),
    (
        "tvt_history_sdf_from_observed_tvt_input_prefix",
        "Target-free SDF history from observed TVT_input prefix.",
    ),
    ("observed_tvt_input_mask", "Observed TVT_input mask in the horizontal window."),
]


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
            tvt_input_path=np.empty((0, horizon), dtype=np.float32),
            md_path=np.empty((0, horizon), dtype=np.float32),
            z_path=np.empty((0, horizon), dtype=np.float32),
            horizontal_row_index=np.empty((0, horizon), dtype=np.int32),
            horizontal_offsets=offsets,
        )


@dataclass(frozen=True)
class Segment:
    well: str
    path_npz_sample_index: int
    row_center: int
    rank: int
    center_score: float
    score_prob: float
    center_tvt: float
    rows: np.ndarray
    tvt: np.ndarray
    step_abs_mean: float
    step_abs_max: float


@dataclass(frozen=True)
class BeamState:
    total_cost: float
    score_cost: float
    smoothness_cost: float
    overlap_cost: float
    boundary_cost: float
    rank_switch_cost: float
    assignments: tuple[Segment, ...]
    overlap_rows_total: int
    gap_count: int
    last_segment: Segment | None

    @staticmethod
    def empty() -> "BeamState":
        return BeamState(
            total_cost=0.0,
            score_cost=0.0,
            smoothness_cost=0.0,
            overlap_cost=0.0,
            boundary_cost=0.0,
            rank_switch_cost=0.0,
            assignments=(),
            overlap_rows_total=0,
            gap_count=0,
            last_segment=None,
        )


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    if decompressed and path.suffix == ".gz":
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def resolve_config_reference(config: dict[str, Any], value: Any) -> Any:
    if isinstance(value, str):
        nested = get_nested(config, value)
        if nested is not None:
            return nested
    return value


def direct_path_candidates(path_value: Any) -> list[Path]:
    if path_value is None:
        return []
    raw = Path(str(path_value))
    if raw.is_absolute():
        return [raw]
    return [
        ROOT / raw,
        Path.cwd() / raw,
        raw,
    ]


def find_artifact(path_value: Any, *, fallback_name: str) -> Path:
    for candidate in direct_path_candidates(path_value):
        if candidate.exists():
            return candidate

    search_names = []
    if path_value is not None:
        search_names.append(Path(str(path_value)).name)
    search_names.append(fallback_name)

    search_roots = [
        ROOT / "experiments",
        ROOT / "data",
        KAGGLE_INPUT_ROOT,
        Path.cwd(),
    ]
    seen_roots: set[Path] = set()
    for root in search_roots:
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        for name in dict.fromkeys(search_names):
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Could not find artifact {path_value!r} or {fallback_name!r}")


def channel_schema_for(channel_set: str) -> list[tuple[str, str]]:
    if channel_set == "base":
        return list(BASE_CHANNEL_SCHEMA)
    raise ValueError(f"Unsupported exp208 channel_set: {channel_set}")


def stable_int(*parts: str, modulo: int | None = None) -> int:
    payload = "::".join(parts).encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value % int(modulo) if modulo else value


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
    filled_values = filled.fillna(fallback).to_numpy(np.float32)
    return robust_zscore(filled_values), missing


def safe_angle_sin_cos(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.arctan2(numerator.astype(np.float32), denominator.astype(np.float32))
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def nearest_grid_indices(
    grid_tvt: np.ndarray,
    truth_tvt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(grid_tvt, truth_tvt, side="left")
    left = np.clip(positions - 1, 0, len(grid_tvt) - 1)
    right = np.clip(positions, 0, len(grid_tvt) - 1)
    choose_right = np.abs(grid_tvt[right] - truth_tvt) < np.abs(
        grid_tvt[left] - truth_tvt
    )
    index = np.where(choose_right, right, left).astype(np.int64)
    distance = np.abs(grid_tvt[index] - truth_tvt).astype(np.float32)
    return index, distance


def clean_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def resolve_run_plan(config: dict[str, Any]) -> list[RunPlanItem]:
    raw_specs = get_nested(config, "model.active_run_specs") or []
    if not raw_specs:
        raise ValueError("model.active_run_specs must not be empty")
    default_fold_indices = tuple(
        int(value) for value in (get_nested(config, "validation.active_fold_indices") or [0])
    )
    plan: list[RunPlanItem] = []
    for raw_spec in raw_specs:
        spec = dict(raw_spec)
        name = clean_name(str(spec["name"]))
        channel_set = str(spec.get("channel_set", "base"))
        channel_schema_for(channel_set)
        for fold_index in spec.get("fold_indices", default_fold_indices):
            plan.append(
                RunPlanItem(
                    run_spec=name,
                    variant=str(spec.get("variant", "real_gr")),
                    channel_set=channel_set,
                    fold_index=int(fold_index),
                    horizontal_window_rows=int(spec.get("horizontal_window_rows", 128)),
                    typewell_window_bins=int(spec.get("typewell_window_bins", 64)),
                    tvt_grid_half_width_ft=float(
                        spec.get("tvt_grid_half_width_ft", 192.0)
                    ),
                    history_scale_ft=float(spec.get("history_scale_ft", 200.0)),
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
    max_valid_wells = get_nested(config, "model.training.max_valid_wells")
    if max_valid_wells is not None:
        valid_wells = valid_wells[: int(max_valid_wells)]
    return train_wells, valid_wells


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


def dense_rows_for_well(
    arrays: WellArrays,
    *,
    stride: int,
    max_tail_rows: int,
    include_tail_stop: bool,
) -> np.ndarray:
    tail_start = arrays.prefix_end + 1
    tail_stop = min(arrays.horizontal_rows - 1, arrays.prefix_end + int(max_tail_rows))
    if tail_stop <= tail_start:
        return np.array([], dtype=np.int32)
    rows = np.arange(tail_start, tail_stop + 1, int(stride), dtype=np.int32)
    if include_tail_stop and (len(rows) == 0 or int(rows[-1]) != int(tail_stop)):
        rows = np.unique(np.concatenate([rows, np.asarray([tail_stop], dtype=np.int32)]))
    return rows.astype(np.int32)


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


def build_dense_valid_sample_index(
    *,
    arrays_by_well: dict[str, WellArrays],
    valid_wells: list[str],
    config: dict[str, Any],
    plan_item: RunPlanItem,
    sample_id_start: int,
) -> pd.DataFrame:
    generation = get_nested(config, "path_generation") or {}
    training = get_nested(config, "model.training") or {}
    horizontal_window = int(plan_item.horizontal_window_rows)
    horizontal_offsets = np.arange(
        -(horizontal_window // 2),
        horizontal_window // 2,
        dtype=np.int32,
    )
    grid_offsets_tvt = np.linspace(
        -float(plan_item.tvt_grid_half_width_ft),
        float(plan_item.tvt_grid_half_width_ft),
        int(plan_item.typewell_window_bins),
    ).astype(np.float32)
    target_tolerance = float(training.get("center_target_tolerance_ft", 10.0))
    stride = int(generation.get("row_center_stride", 64))
    include_tail_stop = bool(generation.get("include_tail_stop", True))
    max_tail_rows = int(training.get("max_tail_rows", 2048))

    rows: list[dict[str, Any]] = []
    sample_id = int(sample_id_start)
    for well in valid_wells:
        arrays = arrays_by_well[well]
        for row_center in dense_rows_for_well(
            arrays,
            stride=stride,
            max_tail_rows=max_tail_rows,
            include_tail_stop=include_tail_stop,
        ):
            prior_center, true_center, center_in_grid, label_fraction = sample_label_status(
                arrays,
                int(row_center),
                horizontal_offsets,
                grid_offsets_tvt,
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
                        split="valid",
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


class HeatmapWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        sample_index: pd.DataFrame,
        arrays_by_well: dict[str, WellArrays],
        plan_item: RunPlanItem,
    ) -> None:
        self.sample_index = sample_index.reset_index(drop=True)
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
        self.grid_offsets_tvt = np.linspace(
            -float(plan_item.tvt_grid_half_width_ft),
            float(plan_item.tvt_grid_half_width_ft),
            int(plan_item.typewell_window_bins),
        ).astype(np.float32)
        self.history_scale_ft = float(plan_item.history_scale_ft)
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
        observed_safe = np.where(np.isfinite(observed_tvt), observed_tvt, 0.0).astype(
            np.float32
        )
        history = (
            grid_tvt.reshape(1, -1) - observed_safe.reshape(-1, 1)
        ) / self.history_scale_ft
        history = np.clip(history * mask.reshape(-1, 1), -8.0, 8.0)
        mask_heatmap = np.broadcast_to(mask.reshape(-1, 1), history.shape)
        image = np.stack([t_heatmap, h_heatmap, diff, history, mask_heatmap], axis=0)

        return {
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
            "image": torch.from_numpy(image.astype(np.float32)),
            "grid_tvt": torch.from_numpy(grid_tvt.astype(np.float32)),
            "horizontal_row_index": torch.from_numpy(h_idx.astype(np.int32)),
            "horizontal_md": torch.from_numpy(arrays.md[h_idx].astype(np.float32)),
            "horizontal_z": torch.from_numpy(arrays.z[h_idx].astype(np.float32)),
            "tvt_input_path": torch.from_numpy(arrays.tvt_input[h_idx].astype(np.float32)),
        }


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


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(get_nested(config, "path_generation.device") or "cpu").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Unexpected exp202 model artifact format: {path}")
    return payload


def exp202_model_path_for_fold(config: dict[str, Any], plan_item: RunPlanItem) -> Path:
    artifact_dir_value = resolve_config_reference(
        config,
        get_nested(config, "data.exp202_artifact_dir_local"),
    )
    fallback_name = (
        f"{DEFAULT_EXP202_PREFIX}_{plan_item.run_spec}_fold"
        f"{int(plan_item.fold_index)}_model.pt"
    )
    if artifact_dir_value is not None:
        artifact_dir = Path(str(artifact_dir_value))
        candidates = [
            artifact_dir / fallback_name,
            ROOT / artifact_dir / fallback_name,
            Path.cwd() / artifact_dir / fallback_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return find_artifact(fallback_name, fallback_name=fallback_name)


def exp202_manifest_path(config: dict[str, Any]) -> Path | None:
    value = resolve_config_reference(config, get_nested(config, "data.exp202_model_manifest_local"))
    try:
        return find_artifact(value, fallback_name=DEFAULT_EXP202_MODEL_MANIFEST)
    except FileNotFoundError:
        return None


def make_loader(
    dataset: HeatmapWindowDataset,
    *,
    batch_size: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_candidate_paths(
    *,
    model: HeatmapMTPNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    plan_item: RunPlanItem,
    topk: int,
) -> tuple[pd.DataFrame, CandidatePathOutput]:
    model.eval()
    center_position = loader.dataset.center_position  # type: ignore[attr-defined]
    horizontal_offsets = loader.dataset.horizontal_offsets.astype(np.int32)  # type: ignore[attr-defined]
    horizon = int(len(horizontal_offsets))
    prediction_rows: list[dict[str, Any]] = []
    path_sample_ids: list[np.ndarray] = []
    path_mode_indices: list[np.ndarray] = []
    path_center_bins: list[np.ndarray] = []
    path_center_tvts: list[np.ndarray] = []
    path_scores: list[np.ndarray] = []
    pred_tvt_paths: list[np.ndarray] = []
    pred_bin_paths: list[np.ndarray] = []
    tvt_input_paths: list[np.ndarray] = []
    md_paths: list[np.ndarray] = []
    z_paths: list[np.ndarray] = []
    row_index_paths: list[np.ndarray] = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        path_logits, mode_logits = model(image)
        mode_prob = torch.softmax(mode_logits, dim=1)
        center_logits = path_logits[:, :, center_position, :]
        center_prob = torch.softmax(center_logits, dim=2)
        center_score, center_idx = center_prob.max(dim=2)
        combined_score = mode_prob * center_score
        order = combined_score.argsort(dim=1, descending=True)
        full_path_idx = torch.softmax(path_logits, dim=3).argmax(dim=3)

        grid_tvt = batch["grid_tvt"].cpu().numpy()
        sample_ids = batch["sample_id"].cpu().numpy()
        center_idx_np = center_idx.cpu().numpy()
        order_np = order.cpu().numpy()
        score_np = combined_score.cpu().numpy()
        full_path_idx_np = full_path_idx.cpu().numpy()
        batch_size = len(sample_ids)

        batch_mode_index = np.full((batch_size, topk), -1, dtype=np.int16)
        batch_center_bin = np.full((batch_size, topk), -1, dtype=np.int16)
        batch_center_tvt = np.full((batch_size, topk), np.nan, dtype=np.float32)
        batch_score = np.full((batch_size, topk), np.nan, dtype=np.float32)
        batch_pred_tvt_path = np.full((batch_size, topk, horizon), np.nan, dtype=np.float32)
        batch_pred_bin_path = np.full((batch_size, topk, horizon), -1, dtype=np.int16)

        for row_index in range(batch_size):
            candidate_scores: list[float] = []
            seen_bins: set[int] = set()
            for mode_index in order_np[row_index].tolist():
                pred_idx = int(center_idx_np[row_index, mode_index])
                if pred_idx in seen_bins:
                    continue
                seen_bins.add(pred_idx)
                rank_index = len(candidate_scores)
                path_bins = full_path_idx_np[row_index, mode_index, :].astype(np.int16)
                batch_mode_index[row_index, rank_index] = int(mode_index)
                batch_center_bin[row_index, rank_index] = int(pred_idx)
                batch_center_tvt[row_index, rank_index] = float(grid_tvt[row_index, pred_idx])
                batch_score[row_index, rank_index] = float(score_np[row_index, mode_index])
                batch_pred_tvt_path[row_index, rank_index, :] = grid_tvt[row_index, path_bins]
                batch_pred_bin_path[row_index, rank_index, :] = path_bins
                candidate_scores.append(float(score_np[row_index, mode_index]))
                if len(candidate_scores) >= topk:
                    break

            score_values = batch_score[row_index]
            finite_scores = score_values[np.isfinite(score_values)]
            score_sum = float(np.sum(finite_scores)) if len(finite_scores) else 0.0
            if score_sum > 0.0:
                score_prob = finite_scores / score_sum
                entropy = float(-np.sum(score_prob * np.log(np.maximum(score_prob, 1e-12))))
                top3_mass = float(np.sum(score_prob[: min(3, len(score_prob))]))
                top5_mass = float(np.sum(score_prob[: min(5, len(score_prob))]))
            else:
                entropy = float("nan")
                top3_mass = float("nan")
                top5_mass = float("nan")
            record: dict[str, Any] = {
                "run_spec": plan_item.run_spec,
                "variant": plan_item.variant,
                "channel_set": plan_item.channel_set,
                "fold_index": int(plan_item.fold_index),
                "sample_id": int(sample_ids[row_index]),
                "score_entropy": entropy,
                "score_top3_mass": top3_mass,
                "score_top5_mass": top5_mass,
                "top1_top2_score_margin": float(score_values[0] - score_values[1])
                if len(score_values) > 1
                and np.isfinite(score_values[0])
                and np.isfinite(score_values[1])
                else np.nan,
                "top1_top3_score_margin": float(score_values[0] - score_values[2])
                if len(score_values) > 2
                and np.isfinite(score_values[0])
                and np.isfinite(score_values[2])
                else np.nan,
            }
            for rank in range(1, topk + 1):
                record[f"pred_top{rank}_tvt"] = float(batch_center_tvt[row_index, rank - 1])
                record[f"pred_top{rank}_score"] = float(batch_score[row_index, rank - 1])
            prediction_rows.append(record)

        path_sample_ids.append(sample_ids.astype(np.int64))
        path_mode_indices.append(batch_mode_index)
        path_center_bins.append(batch_center_bin)
        path_center_tvts.append(batch_center_tvt)
        path_scores.append(batch_score)
        pred_tvt_paths.append(batch_pred_tvt_path)
        pred_bin_paths.append(batch_pred_bin_path)
        tvt_input_paths.append(batch["tvt_input_path"].cpu().numpy().astype(np.float32))
        md_paths.append(batch["horizontal_md"].cpu().numpy().astype(np.float32))
        z_paths.append(batch["horizontal_z"].cpu().numpy().astype(np.float32))
        row_index_paths.append(batch["horizontal_row_index"].cpu().numpy().astype(np.int32))

    predictions = pd.DataFrame(prediction_rows)
    if not path_sample_ids:
        path_output = CandidatePathOutput.empty(
            topk=topk,
            horizon=horizon,
            horizontal_offsets=horizontal_offsets,
        )
    else:
        path_output = CandidatePathOutput(
            sample_id=np.concatenate(path_sample_ids, axis=0),
            mode_index=np.concatenate(path_mode_indices, axis=0),
            center_bin=np.concatenate(path_center_bins, axis=0),
            center_tvt=np.concatenate(path_center_tvts, axis=0),
            score=np.concatenate(path_scores, axis=0),
            pred_tvt_path=np.concatenate(pred_tvt_paths, axis=0),
            pred_bin_path=np.concatenate(pred_bin_paths, axis=0),
            tvt_input_path=np.concatenate(tvt_input_paths, axis=0),
            md_path=np.concatenate(md_paths, axis=0),
            z_path=np.concatenate(z_paths, axis=0),
            horizontal_row_index=np.concatenate(row_index_paths, axis=0),
            horizontal_offsets=horizontal_offsets,
        )
    return predictions, path_output


def stack_path_outputs(
    outputs: list[CandidatePathOutput],
    *,
    topk: int,
    horizon: int,
) -> CandidatePathOutput:
    non_empty = [output for output in outputs if len(output.sample_id) > 0]
    if not non_empty:
        offsets = np.arange(-(horizon // 2), horizon - horizon // 2, dtype=np.int32)
        return CandidatePathOutput.empty(topk=topk, horizon=horizon, horizontal_offsets=offsets)
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
        tvt_input_path=np.concatenate([output.tvt_input_path for output in non_empty], axis=0),
        md_path=np.concatenate([output.md_path for output in non_empty], axis=0),
        z_path=np.concatenate([output.z_path for output in non_empty], axis=0),
        horizontal_row_index=np.concatenate(
            [output.horizontal_row_index for output in non_empty],
            axis=0,
        ),
        horizontal_offsets=horizontal_offsets,
    )


def path_step_mean(paths_array: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(paths_array.astype(np.float32), axis=2))
    valid = np.isfinite(diffs)
    counts = valid.sum(axis=2)
    sums = np.where(valid, diffs, 0.0).sum(axis=2)
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float32),
        where=counts > 0,
    )


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


def build_path_sample_frame(
    sample_index: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    sample_columns = [
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
        "md_since_prefix",
        "z_since_prefix",
    ]
    prediction_columns = [
        "sample_id",
        "score_entropy",
        "score_top3_mass",
        "score_top5_mass",
        "top1_top2_score_margin",
        "top1_top3_score_margin",
    ]
    frame = sample_index[sample_columns].merge(
        predictions[prediction_columns],
        on="sample_id",
        how="left",
    )
    frame.insert(0, "path_npz_sample_index", np.arange(len(frame), dtype=np.int64))
    frame["distance_bucket"] = "unknown"
    return frame


def build_path_rank_frame(path_output: CandidatePathOutput, samples: pd.DataFrame) -> pd.DataFrame:
    path_count, path_topk, _ = path_output.pred_tvt_path.shape
    rank_index = np.tile(np.arange(1, path_topk + 1, dtype=np.int16), path_count)
    sample_index_repeated = np.repeat(np.arange(path_count, dtype=np.int64), path_topk)
    sample_id_repeated = np.repeat(path_output.sample_id.astype(np.int64), path_topk)
    step_mean = path_step_mean(path_output.pred_tvt_path)
    step_max = path_step_max(path_output.pred_tvt_path)
    rank_frame = pd.DataFrame(
        {
            "path_npz_sample_index": sample_index_repeated,
            "sample_id": sample_id_repeated,
            "rank": rank_index.astype(np.int16),
            "mode_index": path_output.mode_index.reshape(-1).astype(np.int16),
            "center_bin": path_output.center_bin.reshape(-1).astype(np.int16),
            "center_pred_tvt": path_output.center_tvt.reshape(-1).astype(np.float32),
            "center_score": path_output.score.reshape(-1).astype(np.float32),
            "path_step_abs_mean_ft": step_mean.reshape(-1).astype(np.float32),
            "path_step_abs_max_ft": step_max.reshape(-1).astype(np.float32),
        }
    )
    return rank_frame.merge(
        samples[["path_npz_sample_index", "id", "well", "fold_index", "row_center"]],
        on="path_npz_sample_index",
        how="left",
    )


def generate_dense_path_inputs(
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    debug: bool,
    max_wells_per_fold: int | None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    seed = int(get_nested(config, "reproducibility.seed") or 42)
    generation = get_nested(config, "path_generation") or {}
    training = get_nested(config, "model.training") or {}
    batch_size = int(get_nested(config, "runtime.batch_size") or 64)
    topk = int(generation.get("topk", 10))
    device = choose_device(config)
    run_plan = resolve_run_plan(config)

    max_wells = training.get("max_wells")
    all_wells = list_train_wells(paths.train_data_dir, int(max_wells) if max_wells else None)
    if len(all_wells) < int(get_nested(config, "validation.n_folds") or 5):
        raise RuntimeError("Not enough train wells for configured GroupKFold.")

    arrays_by_well: dict[str, WellArrays] = {}
    for well in all_wells:
        arrays = read_well_arrays(well, paths.train_data_dir, seed)
        if arrays is not None:
            arrays_by_well[well] = arrays
    usable_wells = [well for well in all_wells if well in arrays_by_well]
    if len(usable_wells) < int(get_nested(config, "validation.n_folds") or 5):
        raise RuntimeError("Not enough usable wells after loading train files.")

    if debug and max_wells_per_fold is None:
        max_wells_per_fold = int(generation.get("debug_max_wells_per_fold") or 2)

    sample_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    path_outputs: list[CandidatePathOutput] = []
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    sample_id_start = 0

    for plan_item in run_plan:
        _, valid_wells = split_wells(usable_wells, config, fold_index=plan_item.fold_index)
        valid_wells = [well for well in valid_wells if well in arrays_by_well]
        if max_wells_per_fold is not None:
            valid_wells = valid_wells[: int(max_wells_per_fold)]
        sample_index = build_dense_valid_sample_index(
            arrays_by_well=arrays_by_well,
            valid_wells=valid_wells,
            config=config,
            plan_item=plan_item,
            sample_id_start=sample_id_start,
        )
        if sample_index.empty:
            continue
        sample_id_start = int(sample_index["sample_id"].max()) + 1

        dataset = HeatmapWindowDataset(
            sample_index=sample_index,
            arrays_by_well=arrays_by_well,
            plan_item=plan_item,
        )
        loader = make_loader(dataset, batch_size=batch_size)
        model_path = exp202_model_path_for_fold(config, plan_item)
        payload = load_model_payload(model_path, device)
        model = make_model(config, in_channels=len(channel_schema_for(plan_item.channel_set)))
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        predictions, path_output = predict_candidate_paths(
            model=model,
            loader=loader,
            device=device,
            plan_item=plan_item,
            topk=topk,
        )
        sample_frames.append(sample_index)
        prediction_frames.append(predictions)
        path_outputs.append(path_output)
        model_rows.append(
            {
                "run_spec": plan_item.run_spec,
                "fold_index": int(plan_item.fold_index),
                "model_path": str(model_path),
                "model_sha256": sha256_path(model_path),
                "model_bytes": model_path.stat().st_size,
                "samples": int(len(sample_index)),
                "valid_wells": int(len(valid_wells)),
            }
        )
        fold_rows.append(
            {
                "run_spec": plan_item.run_spec,
                "fold_index": int(plan_item.fold_index),
                "valid_wells": int(len(valid_wells)),
                "samples": int(len(sample_index)),
                "row_center_stride": int(generation.get("row_center_stride", 64)),
            }
        )

    if not sample_frames:
        raise RuntimeError("No dense path samples were generated.")
    sample_index_all = pd.concat(sample_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    horizon = int(get_nested(config, "model.architecture.path_horizon") or 128)
    path_output = stack_path_outputs(path_outputs, topk=topk, horizon=horizon)
    path_samples = build_path_sample_frame(sample_index_all, predictions_all)
    rank_index = build_path_rank_frame(path_output, path_samples)

    dense_sample_index_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_dense_sample_index.csv.gz"
    predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_dense_validation_predictions.csv.gz"
    path_samples_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_dense_path_samples.csv.gz"
    rank_index_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_dense_path_rank_index.csv.gz"
    path_npz_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_dense_candidate_paths_top{topk}.npz"
    model_manifest_path = exp202_manifest_path(config)

    gzip_csv(sample_index_all, dense_sample_index_path)
    gzip_csv(predictions_all, predictions_path)
    gzip_csv(path_samples, path_samples_path)
    gzip_csv(rank_index, rank_index_path)
    np.savez_compressed(
        path_npz_path,
        sample_id=path_output.sample_id,
        mode_index=path_output.mode_index,
        center_bin=path_output.center_bin,
        center_tvt=path_output.center_tvt,
        score=path_output.score,
        pred_tvt_path=path_output.pred_tvt_path,
        pred_bin_path=path_output.pred_bin_path,
        tvt_input_path=path_output.tvt_input_path,
        md_path=path_output.md_path,
        z_path=path_output.z_path,
        horizontal_row_index=path_output.horizontal_row_index,
        horizontal_offsets=path_output.horizontal_offsets,
    )

    arrays = {
        "sample_id": path_output.sample_id,
        "center_tvt": path_output.center_tvt,
        "score": path_output.score,
        "pred_tvt_path": path_output.pred_tvt_path,
        "horizontal_row_index": path_output.horizontal_row_index,
        "horizontal_offsets": path_output.horizontal_offsets,
    }
    path_meta = {
        "generated": True,
        "device": str(device),
        "debug": bool(debug),
        "max_wells_per_fold": max_wells_per_fold,
        "samples": int(len(path_samples)),
        "wells": int(path_samples["well"].nunique()),
        "topk": int(topk),
        "horizon": int(path_output.pred_tvt_path.shape[2]),
        "row_center_stride": int(generation.get("row_center_stride", 64)),
        "include_tail_stop": bool(generation.get("include_tail_stop", True)),
        "dense_sample_index_path": str(dense_sample_index_path),
        "dense_sample_index_csv_gz_sha256": sha256_path(dense_sample_index_path),
        "dense_sample_index_csv_decompressed_sha256": sha256_path(
            dense_sample_index_path,
            decompressed=True,
        ),
        "dense_predictions_path": str(predictions_path),
        "dense_predictions_csv_decompressed_sha256": sha256_path(
            predictions_path,
            decompressed=True,
        ),
        "path_npz": str(path_npz_path),
        "path_npz_sha256": sha256_path(path_npz_path),
        "path_samples": str(path_samples_path),
        "path_samples_csv_decompressed_sha256": sha256_path(
            path_samples_path,
            decompressed=True,
        ),
        "rank_index_path": str(rank_index_path),
        "rank_index_csv_decompressed_sha256": sha256_path(
            rank_index_path,
            decompressed=True,
        ),
        "fold_generation": fold_rows,
        "models": model_rows,
        "exp202_model_manifest_path": str(model_manifest_path) if model_manifest_path else None,
        "exp202_model_manifest_sha256": sha256_path(model_manifest_path)
        if model_manifest_path
        else None,
    }
    return arrays, path_samples, path_meta


def load_candidate_path_inputs(
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    path_cfg = get_nested(config, "stitching.inputs") or {}
    npz_path = find_artifact(
        resolve_config_reference(config, path_cfg.get("path_npz")),
        fallback_name=DEFAULT_PATH_NPZ,
    )
    samples_path = find_artifact(
        resolve_config_reference(config, path_cfg.get("path_samples")),
        fallback_name=DEFAULT_PATH_SAMPLES,
    )

    with np.load(npz_path) as loaded:
        required_keys = {
            "sample_id",
            "center_tvt",
            "score",
            "pred_tvt_path",
            "horizontal_row_index",
            "horizontal_offsets",
        }
        missing = sorted(required_keys.difference(loaded.files))
        if missing:
            raise ValueError(f"{npz_path} is missing keys: {missing}")
        arrays = {key: loaded[key] for key in required_keys}

    sample_usecols = [
        "path_npz_sample_index",
        "sample_id",
        "id",
        "split",
        "well",
        "fold_index",
        "row_center",
        "prefix_end",
        "horizontal_window_rows",
        "last_known_tvt",
        "prior_center_tvt",
        "md_since_prefix",
        "z_since_prefix",
        "distance_bucket",
        "score_entropy",
        "score_top3_mass",
        "score_top5_mass",
        "top1_top2_score_margin",
        "top1_top3_score_margin",
    ]
    sample_header = pd.read_csv(samples_path, nrows=0).columns.tolist()
    missing_sample_cols = sorted(set(sample_usecols).difference(sample_header))
    if missing_sample_cols:
        raise ValueError(f"{samples_path} is missing columns: {missing_sample_cols}")
    samples = pd.read_csv(
        samples_path,
        usecols=sample_usecols,
        dtype={"id": str, "well": str, "split": str, "distance_bucket": str},
        low_memory=False,
    )
    samples["well"] = samples["well"].astype(str)
    samples["id"] = samples["id"].astype(str)
    for column in samples.columns:
        if column not in {"id", "well", "split", "distance_bucket"}:
            samples[column] = pd.to_numeric(samples[column], errors="coerce")

    forbidden = {"true_center_tvt", "target_in_grid", "center_abs_error"}
    leaked = sorted(forbidden.intersection(samples.columns))
    if leaked:
        raise ValueError(f"target-derived sample columns entered stitch inputs: {leaked}")

    meta = {
        "path_npz": str(npz_path),
        "path_npz_sha256": sha256_path(npz_path),
        "path_samples": str(samples_path),
        "path_samples_csv_gz_sha256": sha256_path(samples_path),
        "path_samples_csv_decompressed_sha256": sha256_path(
            samples_path,
            decompressed=samples_path.suffix == ".gz",
        ),
        "samples": int(len(samples)),
        "wells": int(samples["well"].nunique()),
        "topk": int(arrays["pred_tvt_path"].shape[1]),
        "horizon": int(arrays["pred_tvt_path"].shape[2]),
        "horizontal_offsets_min": int(np.nanmin(arrays["horizontal_offsets"])),
        "horizontal_offsets_max": int(np.nanmax(arrays["horizontal_offsets"])),
    }
    return arrays, samples, meta


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def step_metrics(values: np.ndarray) -> tuple[float, float]:
    diffs = np.abs(np.diff(values.astype(np.float32)))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 0.0, 0.0
    return float(np.mean(diffs)), float(np.max(diffs))


def segments_for_sample(
    *,
    sample: pd.Series,
    arrays: dict[str, np.ndarray],
    topk: int,
) -> list[Segment]:
    sample_index = int(sample["path_npz_sample_index"])
    rows = arrays["horizontal_row_index"][sample_index].astype(np.int32)
    score_values = arrays["score"][sample_index, :topk].astype(np.float32)
    finite_scores = np.where(np.isfinite(score_values) & (score_values > 0), score_values, 0.0)
    score_sum = float(np.sum(finite_scores))
    if score_sum <= 0.0:
        score_prob = np.full(topk, 1.0 / float(topk), dtype=np.float32)
    else:
        score_prob = finite_scores / score_sum

    segments: list[Segment] = []
    for rank in range(1, topk + 1):
        rank_index = rank - 1
        tvt_path = arrays["pred_tvt_path"][sample_index, rank_index].astype(np.float32)
        valid = np.isfinite(tvt_path) & np.isfinite(rows)
        if not np.any(valid):
            continue
        segment_rows = rows[valid].astype(np.int32)
        segment_tvt = tvt_path[valid].astype(np.float32)
        order = np.argsort(segment_rows)
        segment_rows = segment_rows[order]
        segment_tvt = segment_tvt[order]
        step_mean, step_max = step_metrics(segment_tvt)
        center_tvt = float(arrays["center_tvt"][sample_index, rank_index])
        center_score = float(score_values[rank_index])
        segments.append(
            Segment(
                well=str(sample["well"]),
                path_npz_sample_index=sample_index,
                row_center=int(sample["row_center"]),
                rank=rank,
                center_score=center_score,
                score_prob=float(score_prob[rank_index]),
                center_tvt=center_tvt,
                rows=segment_rows,
                tvt=segment_tvt,
                step_abs_mean=step_mean,
                step_abs_max=step_max,
            )
        )
    return segments


def adjacent_overlap_abs(prev_segment: Segment, segment: Segment) -> tuple[int, float]:
    rows, prev_idx, cur_idx = np.intersect1d(
        prev_segment.rows,
        segment.rows,
        assume_unique=False,
        return_indices=True,
    )
    if len(rows) == 0:
        return 0, float("nan")
    diff = np.abs(prev_segment.tvt[prev_idx] - segment.tvt[cur_idx])
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return int(len(rows)), float("nan")
    return int(len(rows)), float(np.mean(diff))


def boundary_gap_abs(prev_segment: Segment, segment: Segment) -> tuple[int, float]:
    prev_last_row = int(prev_segment.rows[-1])
    cur_first_row = int(segment.rows[0])
    gap_rows = max(0, cur_first_row - prev_last_row - 1)
    if gap_rows <= 0:
        return 0, 0.0
    return gap_rows, float(abs(float(prev_segment.tvt[-1]) - float(segment.tvt[0])))


def add_segment_to_state(
    state: BeamState,
    segment: Segment,
    weights: dict[str, float],
) -> tuple[BeamState, dict[str, Any]]:
    eps = float(weights.get("score_eps", 1e-6))
    score_cost = float(weights.get("score", 1.0)) * -math.log(max(segment.score_prob, eps))
    score_cost += float(weights.get("rank", 0.05)) * float(segment.rank - 1)
    smoothness_cost = float(weights.get("smoothness", 0.01)) * segment.step_abs_mean

    overlap_rows = 0
    overlap_abs = float("nan")
    overlap_cost = 0.0
    gap_rows = 0
    gap_abs = 0.0
    boundary_cost = 0.0
    rank_switch_cost = 0.0
    if state.last_segment is not None:
        overlap_rows, overlap_abs = adjacent_overlap_abs(state.last_segment, segment)
        if overlap_rows > 0 and np.isfinite(overlap_abs):
            overlap_cost = float(weights.get("overlap", 0.04)) * overlap_abs
        else:
            gap_rows, gap_abs = boundary_gap_abs(state.last_segment, segment)
            if gap_rows > 0:
                boundary_cost = float(weights.get("boundary", 0.02)) * gap_abs
        if state.last_segment.rank != segment.rank:
            rank_switch_cost = float(weights.get("rank_switch", 0.02))

    increment = score_cost + smoothness_cost + overlap_cost + boundary_cost + rank_switch_cost
    next_state = BeamState(
        total_cost=state.total_cost + increment,
        score_cost=state.score_cost + score_cost,
        smoothness_cost=state.smoothness_cost + smoothness_cost,
        overlap_cost=state.overlap_cost + overlap_cost,
        boundary_cost=state.boundary_cost + boundary_cost,
        rank_switch_cost=state.rank_switch_cost + rank_switch_cost,
        assignments=(*state.assignments, segment),
        overlap_rows_total=state.overlap_rows_total + overlap_rows,
        gap_count=state.gap_count + int(gap_rows > 0),
        last_segment=segment,
    )
    assignment = {
        "path_npz_sample_index": int(segment.path_npz_sample_index),
        "row_center": int(segment.row_center),
        "rank": int(segment.rank),
        "center_score": float(segment.center_score),
        "score_prob": float(segment.score_prob),
        "segment_step_abs_mean_ft": float(segment.step_abs_mean),
        "segment_step_abs_max_ft": float(segment.step_abs_max),
        "overlap_row_count": int(overlap_rows),
        "overlap_abs_mean_ft": overlap_abs,
        "gap_row_count": int(gap_rows),
        "gap_boundary_abs_ft": gap_abs,
        "incremental_cost": float(increment),
    }
    return next_state, assignment


def stitch_well(
    well_samples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    topk: int,
    beam_width: int,
    output_topn: int,
    weights: dict[str, float],
) -> tuple[list[BeamState], list[dict[str, Any]], dict[str, Any]]:
    samples = well_samples.sort_values("row_center").reset_index(drop=True)
    states: list[tuple[BeamState, tuple[dict[str, Any], ...]]] = [(BeamState.empty(), ())]
    source_rows: set[int] = set()
    center_values: list[int] = []
    for _, sample in samples.iterrows():
        center_values.append(int(sample["row_center"]))
        sample_segments = segments_for_sample(sample=sample, arrays=arrays, topk=topk)
        if not sample_segments:
            continue
        for row in sample_segments[0].rows.tolist():
            source_rows.add(int(row))
        candidates: list[tuple[BeamState, tuple[dict[str, Any], ...]]] = []
        for state, assignments in states:
            for segment in sample_segments:
                next_state, assignment = add_segment_to_state(state, segment, weights)
                candidates.append((next_state, (*assignments, assignment)))
        candidates.sort(key=lambda item: item[0].total_cost)
        states = candidates[:beam_width]

    selected = states[:output_topn]
    assignment_rows: list[dict[str, Any]] = []
    for candidate_index, (state, assignments) in enumerate(selected, start=1):
        for window_order, assignment in enumerate(assignments, start=1):
            row = dict(assignment)
            row.update(
                {
                    "well": str(samples["well"].iloc[0]),
                    "stitched_candidate": f"stitched_path{candidate_index}",
                    "stitched_candidate_rank": candidate_index,
                    "window_order": window_order,
                    "total_cost": float(state.total_cost),
                }
            )
            assignment_rows.append(row)

    centers = np.asarray(center_values, dtype=np.int32)
    gaps = np.diff(np.sort(centers)) if len(centers) > 1 else np.asarray([], dtype=np.int32)
    horizon = int(arrays["pred_tvt_path"].shape[2])
    source_meta = {
        "well": str(samples["well"].iloc[0]) if len(samples) else "",
        "source_window_count": int(len(samples)),
        "source_row_coverage_count": int(len(source_rows)),
        "min_center_gap_rows": int(np.min(gaps)) if len(gaps) else None,
        "max_center_gap_rows": int(np.max(gaps)) if len(gaps) else None,
        "overlap_center_pair_count": int(np.sum(gaps < horizon)) if len(gaps) else 0,
        "gap_center_pair_count": int(np.sum(gaps >= horizon)) if len(gaps) else 0,
    }
    return [state for state, _ in selected], assignment_rows, source_meta


def replay_state_path(
    state: BeamState,
    *,
    well: str,
    candidate_name: str,
) -> dict[str, list[Any]]:
    row_sum: dict[int, float] = {}
    row_weight: dict[int, float] = {}
    row_count: dict[int, int] = {}
    for segment in state.assignments:
        weight = max(float(segment.score_prob), 1e-6)
        for row, tvt in zip(segment.rows.tolist(), segment.tvt.tolist(), strict=False):
            row_int = int(row)
            row_sum[row_int] = row_sum.get(row_int, 0.0) + float(tvt) * weight
            row_weight[row_int] = row_weight.get(row_int, 0.0) + weight
            row_count[row_int] = row_count.get(row_int, 0) + 1

    columns: dict[str, list[Any]] = {
        "id": [],
        "well": [],
        "row_index": [],
        "stitched_candidate": [],
        "stitched_candidate_rank": [],
        "stitched_tvt": [],
        "source_window_count": [],
        "source_weight_sum": [],
        "total_cost": [],
    }
    candidate_rank = int(candidate_name.replace("stitched_path", ""))
    for row in sorted(row_sum):
        weight = row_weight[row]
        columns["id"].append(f"{well}_{row}")
        columns["well"].append(well)
        columns["row_index"].append(row)
        columns["stitched_candidate"].append(candidate_name)
        columns["stitched_candidate_rank"].append(candidate_rank)
        columns["stitched_tvt"].append(row_sum[row] / weight if weight > 0 else np.nan)
        columns["source_window_count"].append(row_count[row])
        columns["source_weight_sum"].append(weight)
        columns["total_cost"].append(float(state.total_cost))
    return columns


def extend_columns(base: dict[str, list[Any]], extra: dict[str, list[Any]]) -> None:
    for key, values in extra.items():
        base.setdefault(key, []).extend(values)


def stitch_all_wells(
    samples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    max_wells: int | None = None,
    local_topk: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stitch_cfg = get_nested(config, "stitching") or {}
    topk = int(local_topk if local_topk is not None else stitch_cfg.get("local_topk", 10))
    beam_width = int(stitch_cfg.get("beam_width", 6))
    output_topn = int(stitch_cfg.get("output_topn", 3))
    weights = stitch_cfg.get("score_weights") or {}
    max_windows_per_well = stitch_cfg.get("max_windows_per_well")
    topk = min(topk, int(arrays["pred_tvt_path"].shape[1]))

    path_columns: dict[str, list[Any]] = {}
    assignment_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    well_names = sorted(samples["well"].astype(str).unique().tolist())
    if max_wells is not None:
        well_names = well_names[:max_wells]

    for well_index, well in enumerate(well_names, start=1):
        well_samples = samples.loc[samples["well"].astype(str) == well].copy()
        if max_windows_per_well is not None:
            well_samples = well_samples.sort_values("row_center").head(int(max_windows_per_well))
        states, assignments, source_meta = stitch_well(
            well_samples,
            arrays,
            topk=topk,
            beam_width=beam_width,
            output_topn=output_topn,
            weights=weights,
        )
        for candidate_index, state in enumerate(states, start=1):
            extend_columns(
                path_columns,
                replay_state_path(
                    state,
                    well=str(well),
                    candidate_name=f"stitched_path{candidate_index}",
                ),
            )
        assignment_rows.extend(assignments)
        source_meta["well_order"] = well_index
        source_rows.append(source_meta)

    path_rows = pd.DataFrame(path_columns)
    assignments = pd.DataFrame(assignment_rows)
    source_coverage = pd.DataFrame(source_rows)
    meta = {
        "wells_processed": int(len(well_names)),
        "path_rows": int(len(path_rows)),
        "assignment_rows": int(len(assignments)),
        "source_coverage_rows": int(len(source_coverage)),
        "local_topk": int(topk),
        "beam_width": int(beam_width),
        "output_topn": int(output_topn),
        "max_wells": max_wells,
    }
    return path_rows, assignments, source_coverage, meta


def load_candidate_cache(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    eval_cfg = get_nested(config, "candidate_union") or {}
    source_path = find_artifact(
        resolve_config_reference(config, eval_cfg.get("source_cache")),
        fallback_name=DEFAULT_EXP099_CACHE,
    )
    id_col = str(eval_cfg.get("id_column", "id"))
    target_col = str(eval_cfg.get("target_delta_column", "target"))
    last_col = str(eval_cfg.get("last_known_tvt_column", "last_known_tvt"))
    distance_col = str(eval_cfg.get("distance_column", "md_since"))
    requested = [str(value) for value in eval_cfg.get("existing_candidates", [])]
    required = [str(value) for value in eval_cfg.get("required_existing_candidates", [])]

    header = pd.read_csv(source_path, nrows=0).columns.tolist()
    available = [column for column in requested if column in header]
    missing_required = sorted(column for column in required if column not in header)
    if missing_required:
        raise ValueError(f"{source_path} missing required candidates: {missing_required}")
    usecols = [id_col, "well", target_col, last_col, *available]
    if distance_col in header:
        usecols.append(distance_col)
    usecols = list(dict.fromkeys(usecols))
    frame = pd.read_csv(source_path, usecols=usecols, dtype={id_col: str, "well": str})
    frame[id_col] = frame[id_col].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {id_col, "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["true_tvt"] = frame[last_col] + frame[target_col]
    rename = {id_col: "id"}
    if distance_col in frame.columns:
        rename[distance_col] = "md_since"
    frame = frame.rename(columns=rename)
    frame["row_index"] = row_index_from_id(frame["id"])
    meta = {
        "path": str(source_path),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "source_csv_gz_sha256": sha256_path(source_path),
        "source_csv_decompressed_sha256": sha256_path(
            source_path,
            decompressed=source_path.suffix == ".gz",
        ),
        "available_existing_candidates": available,
        "missing_existing_candidates": sorted(set(requested).difference(available)),
    }
    return frame, available, meta


def min_abs_error(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    errors = np.abs(values.astype(np.float32) - truth[:, None].astype(np.float32))
    errors[~np.isfinite(values)] = np.nan
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid = np.isfinite(errors).any(axis=1)
    if np.any(valid):
        result[valid] = np.nanmin(errors[valid], axis=1)
    return result


def pairwise_min(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stacked = np.vstack([left, right]).T
    result = np.full(len(left), np.nan, dtype=np.float32)
    valid = np.isfinite(stacked).any(axis=1)
    if np.any(valid):
        result[valid] = np.nanmin(stacked[valid], axis=1)
    return result


def oracle_metric_row(
    *,
    candidate_set: str,
    topk: int,
    candidate_count: int,
    error: np.ndarray,
    within_ft: float,
    existing_error: np.ndarray | None = None,
) -> dict[str, Any]:
    valid = np.isfinite(error)
    row: dict[str, Any] = {
        "candidate_set": candidate_set,
        "topk": int(topk),
        "rows": int(valid.sum()),
        "candidate_count": int(candidate_count),
        "oracle_rmse": None,
        "oracle_mae": None,
        "within10": None,
        "new_best_candidate_rate": None,
        "oracle_rmse_delta_vs_existing": None,
        "within_delta_vs_existing": None,
    }
    if np.any(valid):
        err = error[valid].astype(np.float64)
        row["oracle_rmse"] = float(np.sqrt(np.mean(err * err)))
        row["oracle_mae"] = float(np.mean(err))
        row["within10"] = float(np.mean(err <= within_ft))
    if existing_error is not None:
        both = np.isfinite(error) & np.isfinite(existing_error)
        if np.any(both):
            existing_row = oracle_metric_row(
                candidate_set="existing_reference",
                topk=0,
                candidate_count=0,
                error=existing_error[both],
                within_ft=within_ft,
            )
            row["oracle_rmse_delta_vs_existing"] = (
                float(row["oracle_rmse"] - existing_row["oracle_rmse"])
                if row["oracle_rmse"] is not None
                and existing_row["oracle_rmse"] is not None
                else None
            )
            row["within_delta_vs_existing"] = (
                float(row["within10"] - existing_row["within10"])
                if row["within10"] is not None and existing_row["within10"] is not None
                else None
            )
            row["new_best_candidate_rate"] = float(
                np.mean(error[both] + 1e-6 < existing_error[both])
            )
    return row


def assign_distance_bucket(values: pd.Series, buckets: list[list[float]]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    labels = pd.Series(["unknown"] * len(numeric), index=values.index, dtype=object)
    for low, high in buckets:
        label = f"{int(low)}_{int(high)}" if high < 1_000_000 else f"{int(low)}_plus"
        mask = (numeric >= float(low)) & (numeric < float(high))
        labels.loc[mask] = label
    return labels


def row_index_from_id(values: pd.Series) -> pd.Series:
    row_index = pd.to_numeric(
        values.astype(str).str.rsplit("_", n=1).str[-1],
        errors="coerce",
    )
    if row_index.isna().any():
        examples = values.loc[row_index.isna()].head(5).tolist()
        raise ValueError(f"Could not parse row_index from ids: {examples}")
    return row_index.astype(np.int64)


def evaluate_union(
    path_rows: pd.DataFrame,
    cache: pd.DataFrame,
    existing_candidates: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_cfg = get_nested(config, "candidate_union") or {}
    topk_values = [int(value) for value in eval_cfg.get("topk_values", [1, 3])]
    within_ft = float(eval_cfg.get("within_ft", 10.0))
    if path_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"status": "empty_stitched_path_rows"}

    if {"path_rank", "tvt_pred"}.issubset(path_rows.columns):
        eval_rows = path_rows[["id", "well", "row_index", "path_rank", "tvt_pred"]].copy()
        eval_rows["stitched_candidate"] = (
            "stitched_path"
            + pd.to_numeric(eval_rows["path_rank"], errors="coerce").astype("Int64").astype(str)
        )
        value_column = "tvt_pred"
    else:
        eval_rows = path_rows.copy()
        value_column = "stitched_tvt"

    wide = eval_rows.pivot_table(
        index=["id", "well", "row_index"],
        columns="stitched_candidate",
        values=value_column,
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    stitched_cols = sorted(
        [column for column in wide.columns if str(column).startswith("stitched_path")],
        key=lambda name: int(str(name).replace("stitched_path", "")),
    )
    merged = wide.merge(cache, on="id", how="inner", suffixes=("_stitched", ""))
    if "well" not in merged and "well_stitched" in merged.columns:
        merged = merged.rename(columns={"well_stitched": "well"})
    if merged.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"status": "stitched_cache_join_empty"}

    truth = merged["true_tvt"].to_numpy(np.float32)
    existing_error = min_abs_error(merged[existing_candidates].to_numpy(np.float32), truth)
    metric_rows = [
        oracle_metric_row(
            candidate_set="existing_union_on_stitched_rows",
            topk=0,
            candidate_count=len(existing_candidates),
            error=existing_error,
            within_ft=within_ft,
        )
    ]
    max_topk = min(max(topk_values), len(stitched_cols))
    errors_by_topk: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for topk in topk_values:
        use_cols = stitched_cols[: min(topk, len(stitched_cols))]
        if not use_cols:
            continue
        stitched_error = min_abs_error(merged[use_cols].to_numpy(np.float32), truth)
        union_error = pairwise_min(existing_error, stitched_error)
        errors_by_topk[topk] = (stitched_error, union_error)
        metric_rows.append(
            oracle_metric_row(
                candidate_set=f"stitched_only_top{topk}",
                topk=topk,
                candidate_count=len(use_cols),
                error=stitched_error,
                within_ft=within_ft,
                existing_error=existing_error,
            )
        )
        metric_rows.append(
            oracle_metric_row(
                candidate_set=f"existing_plus_stitched_top{topk}",
                topk=topk,
                candidate_count=len(existing_candidates) + len(use_cols),
                error=union_error,
                within_ft=within_ft,
                existing_error=existing_error,
            )
        )

    metrics_df = pd.DataFrame(metric_rows)
    coverage_df = (
        merged.groupby("well", observed=True)
        .agg(
            covered_rows=("id", "nunique"),
            row_index_min=("row_index", "min"),
            row_index_max=("row_index", "max"),
        )
        .reset_index()
    )
    cache_rows_by_well = cache.groupby("well", observed=True)["id"].nunique().rename("cache_rows")
    coverage_df = coverage_df.merge(cache_rows_by_well, on="well", how="left")
    coverage_df["coverage_rate_vs_cache"] = (
        coverage_df["covered_rows"] / coverage_df["cache_rows"].replace(0, np.nan)
    )

    distance_rows: list[dict[str, Any]] = []
    buckets = eval_cfg.get("distance_buckets") or [
        [0, 50],
        [50, 100],
        [100, 250],
        [250, 500],
        [500, 1000],
        [1000, 1000000000],
    ]
    merged["distance_bucket"] = assign_distance_bucket(merged["md_since"], buckets)
    topk_for_bucket = max_topk
    if topk_for_bucket in errors_by_topk:
        stitched_error, union_error = errors_by_topk[topk_for_bucket]
        metric_context = merged[["well", "distance_bucket"]].copy()
        metric_context["existing_error"] = existing_error
        metric_context["stitched_error"] = stitched_error
        metric_context["union_error"] = union_error
        for bucket, group in metric_context.groupby("distance_bucket", observed=True):
            idx = group.index.to_numpy()
            distance_rows.append(
                {
                    "distance_bucket": bucket,
                    "rows": int(len(group)),
                    "existing_oracle_rmse": oracle_metric_row(
                        candidate_set="existing",
                        topk=0,
                        candidate_count=len(existing_candidates),
                        error=existing_error[idx],
                        within_ft=within_ft,
                    )["oracle_rmse"],
                    "stitched_oracle_rmse": oracle_metric_row(
                        candidate_set="stitched",
                        topk=topk_for_bucket,
                        candidate_count=topk_for_bucket,
                        error=stitched_error[idx],
                        within_ft=within_ft,
                    )["oracle_rmse"],
                    "union_oracle_rmse": oracle_metric_row(
                        candidate_set="union",
                        topk=topk_for_bucket,
                        candidate_count=len(existing_candidates) + topk_for_bucket,
                        error=union_error[idx],
                        within_ft=within_ft,
                    )["oracle_rmse"],
                    "new_best_candidate_rate": float(
                        np.mean(stitched_error[idx] + 1e-6 < existing_error[idx])
                    ),
                }
            )
    distance_df = pd.DataFrame(distance_rows)

    by_well_rows: list[dict[str, Any]] = []
    if topk_for_bucket in errors_by_topk:
        stitched_error, union_error = errors_by_topk[topk_for_bucket]
        context = merged[["well", "id"]].copy()
        context["existing_error"] = existing_error
        context["stitched_error"] = stitched_error
        context["union_error"] = union_error
        for well, group in context.groupby("well", observed=True):
            existing_rmse = oracle_metric_row(
                candidate_set="existing",
                topk=0,
                candidate_count=len(existing_candidates),
                error=group["existing_error"].to_numpy(np.float32),
                within_ft=within_ft,
            )["oracle_rmse"]
            union_rmse = oracle_metric_row(
                candidate_set="union",
                topk=topk_for_bucket,
                candidate_count=len(existing_candidates) + topk_for_bucket,
                error=group["union_error"].to_numpy(np.float32),
                within_ft=within_ft,
            )["oracle_rmse"]
            by_well_rows.append(
                {
                    "well": well,
                    "rows": int(len(group)),
                    "existing_oracle_rmse": existing_rmse,
                    "union_oracle_rmse": union_rmse,
                    "rmse_delta": float(union_rmse - existing_rmse)
                    if existing_rmse is not None and union_rmse is not None
                    else None,
                    "new_best_candidate_rate": float(
                        np.mean(
                            group["stitched_error"].to_numpy(np.float32) + 1e-6
                            < group["existing_error"].to_numpy(np.float32)
                        )
                    ),
                }
            )
    by_well_df = pd.DataFrame(by_well_rows)
    if not by_well_df.empty:
        by_well_df = by_well_df.sort_values(["rmse_delta", "well"], ascending=[True, True])

    summary = {
        "status": "evaluated",
        "stitched_rows": int(len(path_rows)),
        "stitched_wide_rows": int(len(wide)),
        "merged_rows": int(len(merged)),
        "cache_rows": int(len(cache)),
        "row_coverage_rate_vs_cache": float(len(merged) / len(cache)) if len(cache) else None,
        "stitched_candidate_columns": stitched_cols,
        "existing_candidates": existing_candidates,
        "topk_values": topk_values,
    }
    return metrics_df, distance_df, by_well_df, coverage_df, summary


def summarize_physicality(
    path_rows: pd.DataFrame,
    assignments: pd.DataFrame,
    source_coverage: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not source_coverage.empty:
        summary.update(
            {
                "source_wells": int(len(source_coverage)),
                "source_windows_per_well_mean": float(
                    source_coverage["source_window_count"].mean()
                ),
                "source_overlap_wells": int(
                    (source_coverage["overlap_center_pair_count"] > 0).sum()
                ),
                "source_overlap_pair_count": int(
                    source_coverage["overlap_center_pair_count"].sum()
                ),
                "source_gap_pair_count": int(source_coverage["gap_center_pair_count"].sum()),
                "source_row_coverage_count_mean": float(
                    source_coverage["source_row_coverage_count"].mean()
                ),
            }
        )
    if not assignments.empty:
        best = assignments.loc[assignments["stitched_candidate_rank"] == 1].copy()
        rank_dist = (
            best["rank"].value_counts(normalize=True).sort_index().rename("rate").reset_index()
        )
        rank_dist.columns = ["rank", "rate"]
        overlap = assignments.loc[assignments["overlap_row_count"] > 0, "overlap_abs_mean_ft"]
        gap = assignments.loc[assignments["gap_row_count"] > 0, "gap_boundary_abs_ft"]
        summary.update(
            {
                "best_path_rank_distribution": rank_dist.to_dict(orient="records"),
                "assignment_overlap_rows_total": int(assignments["overlap_row_count"].sum()),
                "assignment_overlap_abs_mean_ft": float(overlap.mean())
                if len(overlap)
                else None,
                "assignment_gap_boundary_abs_mean_ft": float(gap.mean()) if len(gap) else None,
                "assignment_gap_boundary_abs_p95_ft": float(gap.quantile(0.95))
                if len(gap)
                else None,
            }
        )
    if not path_rows.empty:
        path_metrics: list[dict[str, Any]] = []
        for (well, candidate), group in path_rows.groupby(
            ["well", "stitched_candidate"],
            observed=True,
        ):
            ordered = group.sort_values("row_index")
            tvt = ordered["stitched_tvt"].to_numpy(np.float32)
            rows = ordered["row_index"].to_numpy(np.int32)
            diffs = np.abs(np.diff(tvt))
            row_gaps = np.diff(rows)
            curvature = np.abs(np.diff(tvt, n=2)) if len(tvt) >= 3 else np.asarray([])
            path_metrics.append(
                {
                    "well": well,
                    "stitched_candidate": candidate,
                    "rows": int(len(ordered)),
                    "path_step_abs_mean_ft": float(np.nanmean(diffs)) if len(diffs) else 0.0,
                    "path_step_abs_p95_ft": float(np.nanpercentile(diffs, 95))
                    if len(diffs)
                    else 0.0,
                    "curvature_abs_mean_ft": float(np.nanmean(curvature))
                    if len(curvature)
                    else 0.0,
                    "row_gap_count": int(np.sum(row_gaps > 1)) if len(row_gaps) else 0,
                    "overlap_row_rate": float(
                        np.mean(ordered["source_window_count"].to_numpy(np.float32) > 1)
                    ),
                }
            )
        path_metrics_df = pd.DataFrame(path_metrics)
        summary.update(
            {
                "stitched_path_step_abs_mean_ft": float(
                    path_metrics_df["path_step_abs_mean_ft"].mean()
                ),
                "stitched_path_step_abs_p95_ft": float(
                    path_metrics_df["path_step_abs_p95_ft"].quantile(0.95)
                ),
                "stitched_curvature_abs_mean_ft": float(
                    path_metrics_df["curvature_abs_mean_ft"].mean()
                ),
                "stitched_row_gap_count_total": int(path_metrics_df["row_gap_count"].sum()),
                "stitched_overlap_row_rate_mean": float(path_metrics_df["overlap_row_rate"].mean()),
            }
        )
    return summary


def candidate_assignment_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "well",
                "stitched_candidate",
                "local_rank_mix",
                "local_rank_mean",
                "assignment_gap_flag",
                "candidate_score",
                "candidate_cost",
                "assignment_windows",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (well, candidate), group in assignments.groupby(
        ["well", "stitched_candidate"],
        observed=True,
    ):
        rank_counts = group["rank"].value_counts(normalize=True).sort_index()
        rank_mix = ";".join(f"r{int(rank)}:{rate:.6f}" for rank, rate in rank_counts.items())
        total_cost = pd.to_numeric(group["total_cost"], errors="coerce").dropna()
        candidate_cost = float(total_cost.iloc[0]) if len(total_cost) else np.nan
        candidate_score = (
            float(1.0 / (1.0 + max(candidate_cost, 0.0)))
            if np.isfinite(candidate_cost)
            else np.nan
        )
        rows.append(
            {
                "well": str(well),
                "stitched_candidate": str(candidate),
                "local_rank_mix": rank_mix,
                "local_rank_mean": float(pd.to_numeric(group["rank"], errors="coerce").mean()),
                "assignment_gap_flag": bool(
                    (pd.to_numeric(group["gap_row_count"], errors="coerce").fillna(0) > 0).any()
                ),
                "candidate_score": candidate_score,
                "candidate_cost": candidate_cost,
                "assignment_windows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def add_path_geometry_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["path_step_abs"] = pd.Series(dtype=np.float32)
        frame["curvature_abs"] = pd.Series(dtype=np.float32)
        frame["row_gap_after_previous"] = pd.Series(dtype=np.int32)
        return frame

    pieces: list[pd.DataFrame] = []
    for (_, _), group in frame.groupby(["well", "path_rank"], observed=True):
        ordered = group.sort_values("row_index").copy()
        tvt = pd.to_numeric(ordered["tvt_pred"], errors="coerce").to_numpy(np.float32)
        rows = pd.to_numeric(ordered["row_index"], errors="coerce").to_numpy(np.float64)
        step = np.zeros(len(ordered), dtype=np.float32)
        curvature = np.zeros(len(ordered), dtype=np.float32)
        row_gap = np.zeros(len(ordered), dtype=np.int32)
        if len(ordered) >= 2:
            step[1:] = np.abs(np.diff(tvt)).astype(np.float32)
            raw_gap = np.diff(rows)
            row_gap[1:] = np.where(np.isfinite(raw_gap), raw_gap > 1, False).astype(np.int32)
        if len(ordered) >= 3:
            curvature[2:] = np.abs(np.diff(tvt, n=2)).astype(np.float32)
        ordered["path_step_abs"] = step
        ordered["curvature_abs"] = curvature
        ordered["row_gap_after_previous"] = row_gap
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True)


def build_full_grid_path_table(
    path_rows: pd.DataFrame,
    assignments: pd.DataFrame,
    cache: pd.DataFrame,
    *,
    local_topk: int,
) -> pd.DataFrame:
    if path_rows.empty:
        return pd.DataFrame()

    required = {
        "id",
        "well",
        "row_index",
        "stitched_candidate",
        "stitched_candidate_rank",
        "stitched_tvt",
        "source_window_count",
        "source_weight_sum",
        "total_cost",
    }
    missing = sorted(required.difference(path_rows.columns))
    if missing:
        raise ValueError(f"stitched path rows missing required columns: {missing}")

    source = path_rows.copy()
    source["id"] = source["id"].astype(str)
    source["well"] = source["well"].astype(str)
    source = source.rename(
        columns={
            "id": "row_id",
            "stitched_candidate_rank": "path_rank",
            "stitched_tvt": "tvt_pred",
            "source_weight_sum": "overlap_weight",
        }
    )
    source.insert(0, "id", source["row_id"])
    source["path_rank"] = pd.to_numeric(source["path_rank"], errors="coerce").astype("Int64")
    source["source_window_count"] = pd.to_numeric(
        source["source_window_count"],
        errors="coerce",
    ).astype("Int64")
    source["source_local_topk"] = int(local_topk)

    assignment_summary = candidate_assignment_summary(assignments)
    source = source.merge(
        assignment_summary,
        on=["well", "stitched_candidate"],
        how="left",
        validate="many_to_one",
    )

    cache_columns = ["id", "well", "row_index"]
    if "md_since" in cache.columns:
        cache_columns.append("md_since")
    if "last_known_tvt" in cache.columns:
        cache_columns.append("last_known_tvt")
    cache_grid = cache[cache_columns].drop_duplicates("id").copy()
    cache_grid["id"] = cache_grid["id"].astype(str)
    cache_grid["well"] = cache_grid["well"].astype(str)
    cache_grid["row_index"] = pd.to_numeric(cache_grid["row_index"], errors="coerce")
    if cache_grid["row_index"].isna().any():
        raise ValueError("candidate cache grid contains null row_index values")
    cache_grid["row_index"] = cache_grid["row_index"].astype(np.int64)

    source_wells = sorted(source["well"].dropna().astype(str).unique().tolist())
    cache_grid = cache_grid.loc[cache_grid["well"].isin(source_wells)].copy()
    source["row_index"] = pd.to_numeric(source["row_index"], errors="coerce").astype(np.int64)
    source["tvt_pred"] = pd.to_numeric(source["tvt_pred"], errors="coerce")
    source["overlap_weight"] = pd.to_numeric(source["overlap_weight"], errors="coerce")
    source["candidate_score"] = pd.to_numeric(source["candidate_score"], errors="coerce")
    source["candidate_cost"] = pd.to_numeric(source["candidate_cost"], errors="coerce")
    source["assignment_gap_flag"] = source["assignment_gap_flag"].fillna(False).astype(bool)
    source["local_rank_mix"] = source["local_rank_mix"].fillna("")

    rank_values = [
        int(value)
        for value in sorted(pd.to_numeric(source["path_rank"], errors="coerce").dropna().unique())
    ]
    pieces: list[pd.DataFrame] = []
    meta_columns = [
        "assignment_gap_flag",
        "local_rank_mix",
        "candidate_score",
        "candidate_cost",
        "source_local_topk",
        "stitched_candidate",
        "local_rank_mean",
        "assignment_windows",
    ]
    for well in source_wells:
        grid = cache_grid.loc[cache_grid["well"] == well].sort_values("row_index").copy()
        if grid.empty:
            continue
        grid_rows = grid["row_index"].to_numpy(np.float64)
        for rank in rank_values:
            rank_source = source.loc[
                (source["well"] == well)
                & (pd.to_numeric(source["path_rank"], errors="coerce") == rank)
            ].copy()
            out = grid.copy()
            out["row_id"] = out["id"]
            out["path_rank"] = int(rank)
            out["source_window_count"] = 0
            out["overlap_weight"] = 0.0
            out["coverage_flag"] = False
            out["fallback_flag"] = True
            out["fill_method"] = "no_source_last_known"

            if rank_source.empty:
                fallback = (
                    pd.to_numeric(out.get("last_known_tvt"), errors="coerce")
                    if "last_known_tvt" in out.columns
                    else pd.Series(np.nan, index=out.index)
                )
                out["tvt_pred"] = fallback.interpolate(limit_direction="both").ffill().bfill()
                out["assignment_gap_flag"] = False
                out["local_rank_mix"] = ""
                out["candidate_score"] = np.nan
                out["candidate_cost"] = np.nan
                out["source_local_topk"] = int(local_topk)
                out["stitched_candidate"] = f"stitched_path{rank}"
                out["local_rank_mean"] = np.nan
                out["assignment_windows"] = 0
                pieces.append(out)
                continue

            rank_source = (
                rank_source.sort_values("row_index")
                .drop_duplicates(["well", "path_rank", "row_index"], keep="first")
                .copy()
            )
            valid_source = rank_source.loc[
                np.isfinite(rank_source["row_index"].to_numpy(np.float64))
                & np.isfinite(rank_source["tvt_pred"].to_numpy(np.float64))
            ].copy()
            if valid_source.empty:
                raise ValueError(f"No finite source predictions for well={well} rank={rank}")

            source_rows = valid_source["row_index"].to_numpy(np.float64)
            source_tvt = valid_source["tvt_pred"].to_numpy(np.float64)
            order = np.argsort(source_rows)
            source_rows = source_rows[order]
            source_tvt = source_tvt[order]
            out["tvt_pred"] = np.interp(grid_rows, source_rows, source_tvt).astype(np.float32)

            source_row_set = set(int(value) for value in source_rows.tolist())
            coverage = out["row_index"].astype(int).isin(source_row_set)
            out["coverage_flag"] = coverage.to_numpy(bool)
            out["fallback_flag"] = ~out["coverage_flag"]
            out["fill_method"] = "interpolated"
            out.loc[out["coverage_flag"], "fill_method"] = "source_window"
            out.loc[out["row_index"] < int(source_rows[0]), "fill_method"] = "left_extrapolated"
            out.loc[out["row_index"] > int(source_rows[-1]), "fill_method"] = "right_extrapolated"

            direct_cols = [
                "row_index",
                "source_window_count",
                "overlap_weight",
            ]
            direct = valid_source[direct_cols].drop_duplicates("row_index")
            out = out.merge(
                direct,
                on="row_index",
                how="left",
                suffixes=("", "_source"),
                validate="many_to_one",
            )
            out["source_window_count"] = pd.to_numeric(
                out["source_window_count_source"].fillna(out["source_window_count"]),
                errors="coerce",
            ).fillna(0)
            out["overlap_weight"] = pd.to_numeric(
                out["overlap_weight_source"].fillna(out["overlap_weight"]),
                errors="coerce",
            ).fillna(0.0)
            out = out.drop(columns=["source_window_count_source", "overlap_weight_source"])

            meta = rank_source.iloc[0]
            for column in meta_columns:
                out[column] = meta[column] if column in meta.index else np.nan
            out["assignment_gap_flag"] = bool(out["assignment_gap_flag"].iloc[0])
            out["local_rank_mix"] = out["local_rank_mix"].fillna("")
            pieces.append(out)

    if not pieces:
        return pd.DataFrame()

    table = pd.concat(pieces, ignore_index=True)
    table["md_since"] = pd.to_numeric(table.get("md_since"), errors="coerce")
    table["md_from_ps"] = table["md_since"]
    table["source_window_count"] = pd.to_numeric(
        table["source_window_count"],
        errors="coerce",
    ).fillna(0).astype("Int64")
    table["overlap_weight"] = pd.to_numeric(table["overlap_weight"], errors="coerce").fillna(0.0)
    table["tvt_pred"] = pd.to_numeric(table["tvt_pred"], errors="coerce")
    table["candidate_score"] = pd.to_numeric(table["candidate_score"], errors="coerce")
    table["candidate_cost"] = pd.to_numeric(table["candidate_cost"], errors="coerce")
    table["assignment_gap_flag"] = table["assignment_gap_flag"].fillna(False).astype(bool)
    table["coverage_flag"] = table["coverage_flag"].fillna(False).astype(bool)
    table["fallback_flag"] = table["fallback_flag"].fillna(True).astype(bool)
    table["local_rank_mix"] = table["local_rank_mix"].fillna("")
    table = add_path_geometry_columns(table)

    columns = [
        "id",
        "well",
        "row_id",
        "row_index",
        "md_since",
        "md_from_ps",
        "path_rank",
        "tvt_pred",
        "source_window_count",
        "overlap_weight",
        "assignment_gap_flag",
        "local_rank_mix",
        "path_step_abs",
        "curvature_abs",
        "candidate_score",
        "coverage_flag",
        "fallback_flag",
        "fill_method",
        "candidate_cost",
        "source_local_topk",
        "stitched_candidate",
        "local_rank_mean",
        "assignment_windows",
        "row_gap_after_previous",
    ]
    return table[columns].sort_values(["well", "path_rank", "row_index"]).reset_index(drop=True)


def full_path_schema_frame(table: pd.DataFrame) -> pd.DataFrame:
    roles = {
        "id": "join key copied from the exp099 feature-cache row grid",
        "well": "well identifier",
        "row_id": "competition row id",
        "row_index": "numeric row suffix within well",
        "md_since": "distance from known prefix sourced from exp099 cache",
        "md_from_ps": "legacy alias of md_since for exp210-style readers",
        "path_rank": "full-grid candidate rank",
        "tvt_pred": "heatmap MDN full-grid candidate TVT",
        "source_window_count": "count of local windows directly covering this row",
        "overlap_weight": "sum of source local score probabilities at source-covered rows",
        "assignment_gap_flag": "candidate-level flag for stitched window gaps",
        "local_rank_mix": "candidate-level local path rank distribution",
        "path_step_abs": "absolute first difference of tvt_pred inside full grid path",
        "curvature_abs": "absolute second difference of tvt_pred inside full grid path",
        "candidate_score": "target-free score from stitch cost, higher is better",
        "coverage_flag": "true when the row is directly covered by a source local window",
        "fallback_flag": "true when tvt_pred was filled by interpolation or endpoint extrapolation",
        "fill_method": "source_window, interpolated, left_extrapolated, or right_extrapolated",
        "candidate_cost": "target-free total stitch cost, lower is better",
        "source_local_topk": "local topK used by the stitch beam",
        "stitched_candidate": "internal stitched candidate name",
        "local_rank_mean": "mean local candidate rank used by this full grid path",
        "assignment_windows": "number of local windows assigned to this full grid path",
        "row_gap_after_previous": "row gap diagnostic inside full grid path",
    }
    rows = []
    for column in table.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(table[column].dtype),
                "nullable": bool(table[column].isna().any()),
                "target_derived": False,
                "role": roles.get(column, ""),
            }
        )
    return pd.DataFrame(rows)


def contract_metrics_frame(
    full_path: pd.DataFrame,
    cache: pd.DataFrame,
    config: dict[str, Any],
    *,
    local_topk: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract_cfg = get_nested(config, "full_path_contract") or {}
    required_columns = [str(value) for value in contract_cfg.get("required_columns", [])]
    missing_columns = sorted(set(required_columns).difference(full_path.columns))
    present_columns = sorted(set(required_columns).intersection(full_path.columns))
    duplicate_keys = (
        int(full_path.duplicated(["well", "row_id", "path_rank"]).sum())
        if {"well", "row_id", "path_rank"}.issubset(full_path.columns)
        else None
    )
    null_required = (
        int(full_path[present_columns].isna().sum().sum()) if present_columns else None
    )
    unique_ids = int(full_path["id"].nunique()) if "id" in full_path else 0
    if "well" in full_path and "well" in cache:
        grid_wells = set(full_path["well"].astype(str).unique().tolist())
        cache_for_grid = cache.loc[cache["well"].astype(str).isin(grid_wells)]
    else:
        cache_for_grid = cache
    cache_ids = int(cache_for_grid["id"].nunique()) if "id" in cache_for_grid else 0
    all_cache_ids = int(cache["id"].nunique()) if "id" in cache else 0
    path_count = int(full_path[["well", "path_rank"]].drop_duplicates().shape[0])
    rows_by_rank = (
        full_path.groupby("path_rank", observed=True)["id"].nunique().to_dict()
        if "path_rank" in full_path
        else {}
    )
    source_covered_ids = (
        int(full_path.loc[full_path["coverage_flag"], "id"].nunique())
        if {"coverage_flag", "id"}.issubset(full_path.columns)
        else 0
    )
    fallback_ids = (
        int(full_path.loc[full_path["fallback_flag"], "id"].nunique())
        if {"fallback_flag", "id"}.issubset(full_path.columns)
        else 0
    )
    fill_method_rows = (
        full_path["fill_method"].value_counts(dropna=False).sort_index().to_dict()
        if "fill_method" in full_path
        else {}
    )
    rows = [
        {
            "metric": "required_columns_present",
            "value": len(missing_columns) == 0,
            "detail": ",".join(missing_columns),
        },
        {
            "metric": "rows",
            "value": int(len(full_path)),
            "detail": "",
        },
        {
            "metric": "wells",
            "value": int(full_path["well"].nunique()) if "well" in full_path else 0,
            "detail": "",
        },
        {
            "metric": "unique_row_ids",
            "value": unique_ids,
            "detail": "",
        },
        {
            "metric": "row_coverage_rate_vs_cache",
            "value": float(unique_ids / cache_ids) if cache_ids else None,
            "detail": "",
        },
        {
            "metric": "row_coverage_rate_vs_all_cache",
            "value": float(unique_ids / all_cache_ids) if all_cache_ids else None,
            "detail": "",
        },
        {
            "metric": "source_covered_unique_row_ids",
            "value": source_covered_ids,
            "detail": "",
        },
        {
            "metric": "source_coverage_rate_vs_grid",
            "value": float(source_covered_ids / unique_ids) if unique_ids else None,
            "detail": "",
        },
        {
            "metric": "fallback_unique_row_ids",
            "value": fallback_ids,
            "detail": "",
        },
        {
            "metric": "fallback_unique_row_rate",
            "value": float(fallback_ids / unique_ids) if unique_ids else None,
            "detail": "",
        },
        {
            "metric": "path_count",
            "value": path_count,
            "detail": "",
        },
        {
            "metric": "duplicate_key_rows",
            "value": duplicate_keys,
            "detail": "key=well,row_id,path_rank",
        },
        {
            "metric": "null_required_value_count",
            "value": null_required,
            "detail": ",".join(required_columns),
        },
        {
            "metric": "assignment_gap_flag_rate",
            "value": float(full_path["assignment_gap_flag"].mean())
            if "assignment_gap_flag" in full_path and len(full_path)
            else None,
            "detail": "",
        },
        {
            "metric": "row_gap_after_previous_count",
            "value": int(full_path["row_gap_after_previous"].sum())
            if "row_gap_after_previous" in full_path
            else None,
            "detail": "",
        },
        {
            "metric": "source_window_count_mean",
            "value": float(full_path["source_window_count"].mean())
            if "source_window_count" in full_path and len(full_path)
            else None,
            "detail": "",
        },
        {
            "metric": "direct_source_overlap_row_rate",
            "value": float((full_path["source_window_count"] > 1).mean())
            if "source_window_count" in full_path and len(full_path)
            else None,
            "detail": "",
        },
        {
            "metric": "path_step_abs_mean",
            "value": float(full_path["path_step_abs"].mean())
            if "path_step_abs" in full_path and len(full_path)
            else None,
            "detail": "",
        },
        {
            "metric": "path_step_abs_p95",
            "value": float(full_path["path_step_abs"].quantile(0.95))
            if "path_step_abs" in full_path and len(full_path)
            else None,
            "detail": "",
        },
        {
            "metric": "curvature_abs_mean",
            "value": float(full_path["curvature_abs"].mean())
            if "curvature_abs" in full_path and len(full_path)
            else None,
            "detail": "",
        },
        {
            "metric": "rows_by_rank",
            "value": None,
            "detail": json.dumps({str(k): int(v) for k, v in rows_by_rank.items()}, sort_keys=True),
        },
        {
            "metric": "fill_method_rows",
            "value": None,
            "detail": json.dumps({str(k): int(v) for k, v in fill_method_rows.items()}),
        },
        {
            "metric": "local_topk",
            "value": int(local_topk),
            "detail": "",
        },
    ]
    summary = {
        "required_columns_present": len(missing_columns) == 0,
        "missing_required_columns": missing_columns,
        "duplicate_key_rows": duplicate_keys,
        "null_required_value_count": null_required,
        "rows": int(len(full_path)),
        "wells": int(full_path["well"].nunique()) if "well" in full_path else 0,
        "unique_row_ids": unique_ids,
        "cache_unique_row_ids": cache_ids,
        "all_cache_unique_row_ids": all_cache_ids,
        "row_coverage_rate_vs_cache": float(unique_ids / cache_ids) if cache_ids else None,
        "row_coverage_rate_vs_all_cache": float(unique_ids / all_cache_ids)
        if all_cache_ids
        else None,
        "source_covered_unique_row_ids": source_covered_ids,
        "source_coverage_rate_vs_grid": float(source_covered_ids / unique_ids)
        if unique_ids
        else None,
        "fallback_unique_row_ids": fallback_ids,
        "fallback_unique_row_rate": float(fallback_ids / unique_ids) if unique_ids else None,
        "fill_method_rows": {str(k): int(v) for k, v in fill_method_rows.items()},
        "path_count": path_count,
        "rows_by_rank": {str(k): int(v) for k, v in rows_by_rank.items()},
        "local_topk": int(local_topk),
    }
    return pd.DataFrame(rows), summary


def run_stitch_probe(
    *,
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
    max_wells: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    config = config or load_config()
    paths = paths or ExperimentPaths()
    paths.ensure_output_dirs()

    if debug and max_wells is None:
        max_wells = int(get_nested(config, "stitching.debug_max_wells") or 3)

    path_cfg = get_nested(config, "path_generation") or {}
    if path_cfg.get("source_mode") == "cached_exp208_dense_artifacts":
        arrays, samples, path_meta = load_candidate_path_inputs(config)
        path_meta["source_mode"] = "cached_exp208_dense_artifacts"
    else:
        arrays, samples, path_meta = generate_dense_path_inputs(
            config=config,
            paths=paths,
            debug=debug,
            max_wells_per_fold=max_wells if debug else None,
        )
        path_meta["source_mode"] = "generated_dense_paths"
    cache, existing_candidates, cache_meta = load_candidate_cache(config)

    stitch_cfg = get_nested(config, "stitching") or {}
    local_topk_values = [
        int(value)
        for value in stitch_cfg.get(
            "local_topk_values",
            [stitch_cfg.get("local_topk", 10)],
        )
    ]
    primary_local_topk = int(stitch_cfg.get("primary_local_topk", max(local_topk_values)))
    topk_summaries: list[dict[str, Any]] = []
    primary_metrics: list[dict[str, Any]] = []
    primary_physicality: dict[str, Any] = {}
    primary_coverage: dict[str, Any] = {}
    primary_output_sha: dict[str, Any] = {}
    primary_output_paths: dict[str, str] = {}

    for local_topk in local_topk_values:
        label = f"localtopk{local_topk}"
        path_rows, assignments, source_coverage, stitch_meta = stitch_all_wells(
            samples,
            arrays,
            config,
            max_wells=max_wells if debug else None,
            local_topk=local_topk,
        )
        physical_summary = summarize_physicality(path_rows, assignments, source_coverage)
        full_path = build_full_grid_path_table(
            path_rows,
            assignments,
            cache,
            local_topk=local_topk,
        )
        metrics_df, distance_df, by_well_df, coverage_df, eval_summary = evaluate_union(
            full_path,
            cache,
            existing_candidates,
            config,
        )
        full_schema_df = full_path_schema_frame(full_path)
        contract_df, contract_summary = contract_metrics_frame(
            full_path,
            cache,
            config,
            local_topk=local_topk,
        )

        path_rows_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_stitched_path_rows.csv.gz"
        assignments_path = (
            paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_stitched_window_assignments.csv.gz"
        )
        source_coverage_path = (
            paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_source_window_coverage.csv"
        )
        coverage_path = (
            paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_stitched_coverage_by_well.csv"
        )
        metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_candidate_union_metrics.csv"
        distance_path = (
            paths.artifacts_dir
            / f"{OUTPUT_PREFIX}_{label}_candidate_union_distance_bucket_metrics.csv"
        )
        by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_candidate_union_by_well.csv"
        full_path_path = (
            paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_full_grid_candidate_paths.csv.gz"
        )
        full_schema_path = (
            paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_full_grid_path_schema.csv"
        )
        contract_path = (
            paths.artifacts_dir / f"{OUTPUT_PREFIX}_{label}_full_grid_contract_metrics.csv"
        )

        gzip_csv(path_rows, path_rows_path)
        gzip_csv(assignments, assignments_path)
        gzip_csv(full_path, full_path_path)
        source_coverage.to_csv(source_coverage_path, index=False)
        coverage_df.to_csv(coverage_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        distance_df.to_csv(distance_path, index=False)
        by_well_df.to_csv(by_well_path, index=False)
        full_schema_df.to_csv(full_schema_path, index=False)
        contract_df.to_csv(contract_path, index=False)

        output_paths = {
            "stitched_path_rows": str(path_rows_path),
            "stitched_window_assignments": str(assignments_path),
            "full_grid_candidate_paths": str(full_path_path),
            "full_grid_path_schema": str(full_schema_path),
            "full_grid_contract_metrics": str(contract_path),
            "source_window_coverage": str(source_coverage_path),
            "stitched_coverage_by_well": str(coverage_path),
            "candidate_union_metrics": str(metrics_path),
            "candidate_union_distance_bucket_metrics": str(distance_path),
            "candidate_union_by_well": str(by_well_path),
        }
        output_sha = {
            "stitched_path_rows_csv_gz_sha256": sha256_path(path_rows_path),
            "stitched_path_rows_csv_decompressed_sha256": sha256_path(
                path_rows_path,
                decompressed=True,
            ),
            "stitched_window_assignments_csv_gz_sha256": sha256_path(assignments_path),
            "stitched_window_assignments_csv_decompressed_sha256": sha256_path(
                assignments_path,
                decompressed=True,
            ),
            "full_grid_candidate_paths_csv_gz_sha256": sha256_path(full_path_path),
            "full_grid_candidate_paths_csv_decompressed_sha256": sha256_path(
                full_path_path,
                decompressed=True,
            ),
            "full_grid_path_schema_csv_sha256": sha256_path(full_schema_path),
            "full_grid_contract_metrics_csv_sha256": sha256_path(contract_path),
            "source_window_coverage_csv_sha256": sha256_path(source_coverage_path),
            "candidate_union_metrics_csv_sha256": sha256_path(metrics_path),
            "candidate_union_distance_bucket_metrics_csv_sha256": sha256_path(distance_path),
            "candidate_union_by_well_csv_sha256": sha256_path(by_well_path),
        }
        topk_summary = {
            "local_topk": int(local_topk),
            "label": label,
            "stitching": stitch_meta,
            "evaluation": eval_summary,
            "physicality": physical_summary,
            "full_grid_contract": contract_summary,
            "full_grid_contract_metrics": contract_df.to_dict(orient="records"),
            "full_path_contract": contract_summary,
            "full_path_contract_metrics": contract_df.to_dict(orient="records"),
            "metrics": metrics_df.to_dict(orient="records"),
            "distance_metrics": distance_df.to_dict(orient="records"),
            "output_paths": output_paths,
            "output_sha256": output_sha,
        }
        topk_summaries.append(topk_summary)
        if local_topk == primary_local_topk:
            primary_metrics = metrics_df.to_dict(orient="records")
            primary_physicality = physical_summary
            primary_coverage = eval_summary
            primary_output_sha = output_sha
            primary_output_paths = output_paths

    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "debug_completed" if debug else "implemented_diagnostic_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "debug": bool(debug),
        "max_wells": max_wells,
        "primary_local_topk": int(primary_local_topk),
        "route": get_nested(config, "experiment.route"),
        "path_inputs": path_meta,
        "candidate_cache": cache_meta,
        "topk_summaries": topk_summaries,
        "output_paths": {"summary": str(summary_path), **primary_output_paths},
        "output_sha256": primary_output_sha,
        "leakage_guard": {
            "stitch_score_uses_target": False,
            "target_columns_read_for_stitch_score": [],
            "target_usage": (
                "true TVT is used only for sample diagnostics and candidate-cache "
                "oracle readout after stitched paths are fixed"
            ),
        },
    }
    write_json(summary_path, summary)

    experiment_metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "debug_completed" if debug else "kaggle_train_diagnostic_completed",
        "route": get_nested(config, "experiment.route"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "key_idea": get_nested(config, "lineage.diff_summary"),
        "parent": get_nested(config, "lineage.parent"),
        "debug": bool(debug),
        "max_wells": max_wells,
        "primary_local_topk": int(primary_local_topk),
        "summary_path": str(summary_path),
        "candidate_union_metrics": primary_metrics,
        "physicality": primary_physicality,
        "coverage": primary_coverage,
        "topk_summaries": topk_summaries,
        "input_sha256": {
            "path_npz": path_meta.get("path_npz_sha256"),
            "path_samples_decompressed": path_meta.get("path_samples_csv_decompressed_sha256"),
            "exp202_model_manifest": path_meta.get("exp202_model_manifest_sha256"),
            "candidate_cache_decompressed": cache_meta.get("source_csv_decompressed_sha256"),
        },
        "output_sha256": primary_output_sha,
        "notes": [
            (
                "This is a CPU diagnostic that formats cached exp208 dense stride "
                "local paths into a selector-facing full-grid path contract."
            ),
            (
                "No direct TVT replacement, softmax averaging, PF weight replacement, "
                "inference, or submission is performed."
            ),
            (
                "This is not a deterministic submission anchor because it depends on "
                "upstream GPU-trained exp202 weights and creates no submission."
            ),
        ],
    }
    write_json(paths.metrics_path, experiment_metrics)
    return summary
