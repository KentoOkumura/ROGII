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
# # exp249 segment-local negative-space GR corridor audit — train
#
# This deterministic, no-training PF/Beam diagnostic rebuilds the GR mismatch
# surface inside overlapping local windows. Components, path anchors, and
# history reset for every segment. Fixed candidates are never pruned, averaged,
# selected, changed, or sent to hidden-test inference.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input resolution
# 3. Candidate cache and raw-well contracts
# 4. Segment-local mismatch surfaces and components
# 5. Path diagnostics and overlap-safe readouts
# 6. Stage 0 preview helpers
# 7. Stage 1 metric and generated-file helpers
# 8. Synthetic contract checks
# 9. Setup and input checks
# 10. Stage 0 / Stage 1 orchestration
# 11. Metrics, guards, SHA, and generated files

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from scipy import ndimage

EXPERIMENT_NAME = "exp249_segment_local_negative_space_gr_corridor_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    handle: Any
    if decompressed and path.suffix == ".gz":
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(mapping: dict[str, str]) -> str:
    payload = "\n".join(f"{key}\t{mapping[key]}" for key in sorted(mapping)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


class DeterministicGzipCsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(fileobj=self.raw, mode="wb", mtime=0)
        self.text = io.TextIOWrapper(self.compressed, encoding="utf-8", newline="")
        self.wrote_header = False

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty and self.wrote_header:
            return
        frame.to_csv(
            self.text,
            index=False,
            header=not self.wrote_header,
            lineterminator="\n",
        )
        self.wrote_header = True

    def close(self) -> None:
        if not self.text.closed:
            self.text.flush()
            self.text.close()
        if not self.raw.closed:
            self.raw.flush()
            self.raw.close()

    def __enter__(self) -> DeterministicGzipCsvWriter:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


# %% [markdown]
# ## 2. Configuration and input resolution

# %%
def find_repo_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_repo_root()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"Could not resolve config.yaml for {EXPERIMENT_NAME}")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_authoritative_runtime() -> None:
    if is_kaggle_runtime():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
        raise RuntimeError(
            "Kaggle Notebook is authoritative. Local execution requires "
            "EXPERIMENT_ALLOW_LOCAL=1 and is debug-only."
        )


def output_dir() -> Path:
    if is_kaggle_runtime():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if is_kaggle_runtime():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return ROOT / "experiments" / EXPERIMENT_NAME / "metrics.json"


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(nested(config, "data.train_dir", "data/raw/train")))
    local = configured if configured.is_absolute() else ROOT / configured
    suffix = str(nested(config, "data.horizontal_suffix", "__horizontal_well.csv"))
    if local.exists() and any(local.glob(f"*{suffix}")):
        return local
    if KAGGLE_INPUT_ROOT.exists():
        for source in sorted(KAGGLE_INPUT_ROOT.iterdir()):
            candidate = source / "train"
            if candidate.is_dir() and any(candidate.glob(f"*{suffix}")):
                return candidate
        for match in sorted(KAGGLE_INPUT_ROOT.rglob(f"*{suffix}")):
            if match.parent.name == "train":
                return match.parent
    raise FileNotFoundError("Could not resolve raw train directory")


def resolve_file(config: dict[str, Any], path_key: str, filename_key: str) -> Path:
    candidates: list[Path] = []
    for value in nested(config, path_key, []) or []:
        path = Path(str(value))
        candidates.append(path if path.is_absolute() else ROOT / path)
    filename = str(nested(config, filename_key))
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.rglob(filename)))
    usable = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    if not usable:
        raise FileNotFoundError(f"Could not resolve {filename}; candidates={candidates[:8]}")
    preferred_tokens = [
        "exp072-exp063-full-replay-feature-cache-train",
        "exp115-hidden-like-spatial-holdout-from-ppt-train",
    ]
    usable.sort(
        key=lambda path: (
            -sum(token in str(path) for token in preferred_tokens),
            len(str(path)),
            str(path),
        )
    )
    return usable[0]


# %% [markdown]
# ## 3. Candidate cache and raw-well contracts

# %%
@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_column: str
    transform: str


def candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    rows = nested(config, "audit.candidates", []) or []
    specs = [
        CandidateSpec(
            name=str(row["name"]),
            source_column=str(row["source_column"]),
            transform=str(row["transform"]),
        )
        for row in rows
    ]
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Candidate specs must be non-empty with unique names")
    return specs


def required_cache_columns(specs: list[CandidateSpec]) -> list[str]:
    columns = {
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "eval_len",
    }
    columns.update(spec.source_column for spec in specs)
    return sorted(columns)


def prepare_cache_chunk(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    out["well"] = out["well"].astype(str)
    split = out["id"].str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("Candidate cache id must have <well>_<row_index> form")
    if not np.array_equal(split[0].to_numpy(str), out["well"].to_numpy(str)):
        raise ValueError("Candidate cache id prefix does not match well")
    out["row_index"] = pd.to_numeric(split[1], errors="raise").astype(np.int32)
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def iter_candidate_cache_wells(
    cache_path: Path,
    specs: list[CandidateSpec],
    chunksize: int,
) -> Iterator[tuple[str, pd.DataFrame]]:
    usecols = required_cache_columns(specs)
    header = pd.read_csv(cache_path, nrows=0).columns.tolist()
    missing = [column for column in usecols if column not in header]
    if missing:
        raise ValueError(f"Candidate cache is missing columns: {missing}")
    numeric = [column for column in usecols if column not in {"id", "well"}]
    pending: pd.DataFrame | None = None
    last_completed: str | None = None
    reader = pd.read_csv(
        cache_path,
        usecols=usecols,
        chunksize=int(chunksize),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    for raw_chunk in reader:
        chunk = prepare_cache_chunk(raw_chunk, numeric)
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        if not chunk["well"].is_monotonic_increasing:
            raise ValueError("Candidate cache must be globally sorted by well")
        last_well = str(chunk["well"].iloc[-1])
        complete = chunk.loc[chunk["well"] != last_well]
        for well, group in complete.groupby("well", sort=False):
            well = str(well)
            if last_completed is not None and well <= last_completed:
                raise ValueError("Candidate cache well blocks are not unique and sorted")
            yield well, group.reset_index(drop=True)
            last_completed = well
        pending = chunk.loc[chunk["well"] == last_well].reset_index(drop=True)
    if pending is not None and len(pending):
        well = str(pending["well"].iloc[0])
        if last_completed is not None and well <= last_completed:
            raise ValueError("Final candidate cache well block is out of order")
        yield well, pending.reset_index(drop=True)


def load_hidden_like_roles(path: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame["well_id"].duplicated().any():
        raise ValueError(f"Invalid hidden-like assignment; missing={missing}")
    groups = {
        "verification_like_spatial": set(
            frame.loc[frame["verification_like_spatial_role"] == "valid", "well_id"]
        ),
        "verification_like_typewell_purged": set(
            frame.loc[
                frame["verification_like_typewell_purged_role"] == "valid", "well_id"
            ]
        ),
    }
    return frame, groups


def load_raw_well(
    well: str,
    cache: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    horizontal_suffix = str(
        nested(config, "data.horizontal_suffix", "__horizontal_well.csv")
    )
    typewell_suffix = str(nested(config, "data.typewell_suffix", "__typewell.csv"))
    horizontal_path = train_dir / f"{well}{horizontal_suffix}"
    typewell_path = train_dir / f"{well}{typewell_suffix}"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"Raw well pair is missing for {well}")
    md_column = str(nested(config, "data.md_column", "MD"))
    z_column = str(nested(config, "data.z_column", "Z"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    target_column = str(nested(config, "data.target_column", "TVT"))
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=[md_column, z_column, gr_column, target_column, input_column],
    )
    typewell = pd.read_csv(typewell_path, usecols=[target_column, gr_column])
    for column in [md_column, z_column, gr_column, target_column, input_column]:
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    for column in [target_column, gr_column]:
        typewell[column] = pd.to_numeric(typewell[column], errors="coerce")
    known = np.flatnonzero(np.isfinite(horizontal[input_column].to_numpy(float)))
    if not len(known):
        raise ValueError(f"{well} has no known TVT_input prefix")
    prefix_end = int(known[-1])
    if not np.array_equal(known, np.arange(prefix_end + 1)):
        raise ValueError(f"{well} TVT_input is not a contiguous prefix")
    row_index = cache["row_index"].to_numpy(np.int32)
    expected = np.arange(prefix_end + 1, len(horizontal), dtype=np.int32)
    if not np.array_equal(row_index, expected):
        raise ValueError(
            f"{well} cache rows do not match official evaluation tail: "
            f"cache={len(row_index)} expected={len(expected)}"
        )
    target = horizontal[target_column].to_numpy(float)[row_index]
    cache_target = cache["last_known_tvt"].to_numpy(float) + cache["target"].to_numpy(float)
    max_delta = float(np.nanmax(np.abs(target - cache_target)))
    if not np.isfinite(max_delta) or max_delta > 0.15:
        raise ValueError(f"{well} raw/cache truth mismatch: max_abs={max_delta}")
    return horizontal, typewell, prefix_end


def materialize_candidates(
    cache: pd.DataFrame,
    specs: list[CandidateSpec],
) -> dict[str, np.ndarray]:
    base = cache["last_known_tvt"].to_numpy(float)
    candidates: dict[str, np.ndarray] = {}
    for spec in specs:
        value = cache[spec.source_column].to_numpy(float)
        if spec.transform == "absolute":
            candidates[spec.name] = value
        elif spec.transform == "base_plus_delta":
            candidates[spec.name] = base + value
        else:
            raise ValueError(f"Unsupported candidate transform: {spec.transform}")
    return candidates


# %% [markdown]
# ## 4. Segment-local mismatch surfaces and components

# %%
@dataclass(frozen=True)
class SegmentWindow:
    segment_id: str
    center_row: int
    pixel_row_index: np.ndarray
    audit_pixel_positions: np.ndarray
    grid_tvt: np.ndarray
    prior_center_tvt: float


@dataclass(frozen=True)
class LocalSurface:
    signed_mismatch: np.ndarray
    absolute_mismatch: np.ndarray
    barrier: np.ndarray
    supported_rows: np.ndarray
    raw_high_fraction: np.ndarray
    component_labels: np.ndarray
    horizontal_center: float
    horizontal_scale: float
    typewell_center: float
    typewell_scale: float
    grid_step_ft: float
    min_tvt_thickness_bins: int


@dataclass(frozen=True)
class PathDiagnostics:
    in_grid: np.ndarray
    endpoint_forbidden: np.ndarray
    component_known: np.ndarray
    anchor_component: int
    component_transition: np.ndarray
    edge_crossing: np.ndarray
    instantaneous_violation: np.ndarray
    valid_within_segment: np.ndarray
    violation_run_length: np.ndarray
    endpoint_abs_mismatch: np.ndarray


def fill_series(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    filled = series.interpolate(limit_direction="both").ffill().bfill()
    fallback = float(filled.dropna().median()) if filled.notna().any() else 0.0
    return filled.fillna(fallback).to_numpy(float)


def robust_scale_local(values: np.ndarray, clip: float) -> tuple[np.ndarray, float, float]:
    filled = fill_series(values)
    finite = filled[np.isfinite(filled)]
    center = float(np.median(finite)) if len(finite) else 0.0
    if len(finite):
        q25, q75 = np.percentile(finite, [25, 75])
        scale = float(q75 - q25)
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(finite)) if len(finite) else 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    scaled = np.clip((filled - center) / scale, -clip, clip).astype(np.float32)
    return scaled, center, scale


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(np.float32)
    )


def segment_centers(tail_start: int, tail_stop: int, config: dict[str, Any]) -> np.ndarray:
    stride = int(nested(config, "audit.segment.row_center_stride"))
    if stride <= 0 or tail_stop < tail_start:
        raise ValueError("Invalid segment center contract")
    centers = np.arange(tail_start, tail_stop + 1, stride, dtype=np.int32)
    if bool(nested(config, "audit.segment.include_tail_stop")) and (
        not len(centers) or centers[-1] != tail_stop
    ):
        centers = np.append(centers, np.int32(tail_stop))
    return np.unique(centers)


def build_segment_window(
    well: str,
    center_row: int,
    horizontal: pd.DataFrame,
    prefix_end: int,
    config: dict[str, Any],
) -> SegmentWindow:
    window_rows = int(nested(config, "audit.segment.horizontal_window_rows"))
    bins = int(nested(config, "audit.segment.typewell_window_bins"))
    half_width = float(nested(config, "audit.segment.tvt_grid_half_width_ft"))
    if window_rows <= 0 or window_rows % 2 != 0 or bins < 2:
        raise ValueError("Local window rows must be positive/even and bins >= 2")
    offsets = np.arange(-(window_rows // 2), window_rows // 2, dtype=np.int32)
    pixel_rows = np.clip(center_row + offsets, 0, len(horizontal) - 1)
    _, first_positions = np.unique(pixel_rows, return_index=True)
    first_mask = np.zeros(len(pixel_rows), dtype=bool)
    first_mask[first_positions] = True
    audit_positions = np.flatnonzero(first_mask & (pixel_rows > prefix_end))
    z_column = str(nested(config, "data.z_column", "Z"))
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    z = horizontal[z_column].to_numpy(float)
    tvt_input = horizontal[input_column].to_numpy(float)
    last_known_tvt = float(tvt_input[prefix_end])
    last_known_z = float(z[prefix_end])
    prior = last_known_tvt - (float(z[center_row]) - last_known_z)
    grid_tvt = np.linspace(prior - half_width, prior + half_width, bins, dtype=np.float32)
    return SegmentWindow(
        segment_id=f"{well}_{int(center_row):07d}",
        center_row=int(center_row),
        pixel_row_index=pixel_rows,
        audit_pixel_positions=audit_positions,
        grid_tvt=grid_tvt,
        prior_center_tvt=float(prior),
    )


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def interval_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] < right[0]:
        return int(right[0] - left[1] - 1)
    if right[1] < left[0]:
        return int(left[0] - right[1] - 1)
    return 0


def label_free_components(
    barrier: np.ndarray,
    supported: np.ndarray,
    pixel_row_index: np.ndarray,
    shift_bins: int,
) -> np.ndarray:
    intervals_by_row: list[list[tuple[int, int, int]]] = []
    parents: list[int] = []

    def add_node() -> int:
        node = len(parents)
        parents.append(node)
        return node

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for position in range(len(supported)):
        current: list[tuple[int, int, int]] = []
        if supported[position]:
            current = [
                (start, stop, add_node())
                for start, stop in contiguous_true_segments(~barrier[position])
            ]
        if (
            position > 0
            and supported[position]
            and supported[position - 1]
            and pixel_row_index[position] == pixel_row_index[position - 1] + 1
        ):
            for start, stop, node in current:
                for previous_start, previous_stop, previous_node in intervals_by_row[-1]:
                    if interval_distance(
                        (start, stop), (previous_start, previous_stop)
                    ) <= shift_bins:
                        union(node, previous_node)
        intervals_by_row.append(current)

    labels = np.full(barrier.shape, -1, dtype=np.int32)
    root_to_label: dict[int, int] = {}
    for position, intervals in enumerate(intervals_by_row):
        for start, stop, node in intervals:
            root = find(node)
            label = root_to_label.setdefault(root, len(root_to_label))
            labels[position, start : stop + 1] = label
    return labels


def build_local_surface(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    segment: SegmentWindow,
    config: dict[str, Any],
) -> LocalSurface:
    gr_column = str(nested(config, "data.gr_column", "GR"))
    target_column = str(nested(config, "data.target_column", "TVT"))
    clip = float(nested(config, "audit.normalization.robust_clip"))
    horizontal_raw_all = horizontal[gr_column].to_numpy(float)
    horizontal_raw = horizontal_raw_all[segment.pixel_row_index]
    horizontal_z, horizontal_center, horizontal_scale = robust_scale_local(
        horizontal_raw, clip
    )
    typewell_clean = (
        typewell[[target_column, gr_column]]
        .dropna()
        .groupby(target_column, as_index=False)[gr_column]
        .mean()
        .sort_values(target_column)
    )
    if len(typewell_clean) < 16:
        raise ValueError("Typewell has fewer than 16 finite TVT/GR rows")
    grid_gr = np.interp(
        segment.grid_tvt,
        typewell_clean[target_column].to_numpy(float),
        typewell_clean[gr_column].to_numpy(float),
    )
    typewell_z, typewell_center, typewell_scale = robust_scale_local(grid_gr, clip)
    smooth_window = int(nested(config, "audit.gr_surface.smooth_window_rows"))
    horizontal_smooth = rolling_median(horizontal_z, smooth_window)
    typewell_smooth = rolling_median(typewell_z, smooth_window)
    signed = typewell_z[None, :] - horizontal_z[:, None]
    smooth_signed = typewell_smooth[None, :] - horizontal_smooth[:, None]
    absolute = np.abs(signed)
    high = (
        absolute >= float(nested(config, "audit.gr_surface.raw_abs_difference_threshold"))
    ) & (
        np.abs(smooth_signed)
        >= float(nested(config, "audit.gr_surface.smooth_abs_difference_threshold"))
    )
    high_fraction = high.mean(axis=1).astype(np.float32)
    flat_std = (
        pd.Series(horizontal_z)
        .rolling(
            int(nested(config, "audit.gr_surface.flat_window_rows")),
            center=True,
            min_periods=int(nested(config, "audit.gr_surface.flat_min_periods")),
        )
        .std()
        .fillna(0.0)
        .to_numpy(float)
    )
    max_fraction = float(
        nested(config, "audit.barrier.max_forbidden_fraction_per_row")
    )
    supported = (
        np.isfinite(horizontal_raw)
        & (flat_std >= float(nested(config, "audit.gr_surface.flat_std_threshold_z")))
        & (high_fraction <= max_fraction)
    )
    grid_step = float(np.median(np.diff(segment.grid_tvt)))
    min_tvt_bins = max(
        1,
        int(
            math.ceil(
                float(nested(config, "audit.barrier.min_tvt_thickness_ft"))
                / max(grid_step, 1e-6)
            )
        ),
    )
    structure = np.ones(
        (
            int(nested(config, "audit.barrier.min_row_persistence")),
            min_tvt_bins,
        ),
        dtype=bool,
    )
    barrier = ndimage.binary_opening(high, structure=structure).astype(bool)
    barrier[~supported] = False
    supported &= barrier.mean(axis=1) <= max_fraction
    barrier[~supported] = False
    shift_bins = int(
        math.ceil(
            float(nested(config, "audit.barrier.max_corridor_shift_ft_per_row"))
            / max(grid_step, 1e-6)
        )
    )
    labels = label_free_components(
        barrier,
        supported,
        segment.pixel_row_index,
        shift_bins,
    )
    return LocalSurface(
        signed_mismatch=signed.astype(np.float32),
        absolute_mismatch=absolute.astype(np.float32),
        barrier=barrier,
        supported_rows=supported,
        raw_high_fraction=high_fraction,
        component_labels=labels,
        horizontal_center=horizontal_center,
        horizontal_scale=horizontal_scale,
        typewell_center=typewell_center,
        typewell_scale=typewell_scale,
        grid_step_ft=grid_step,
        min_tvt_thickness_bins=min_tvt_bins,
    )


# %% [markdown]
# ## 5. Path diagnostics and overlap-safe readouts

# %%
def nearest_state_indices(grid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    positions = np.searchsorted(grid, values, side="left")
    left = np.clip(positions - 1, 0, len(grid) - 1)
    right = np.clip(positions, 0, len(grid) - 1)
    choose_right = np.abs(grid[right] - values) < np.abs(grid[left] - values)
    indices = np.where(choose_right, right, left).astype(np.int32)
    in_grid = np.isfinite(values) & (values >= float(grid[0])) & (values <= float(grid[-1]))
    return indices, in_grid


def consecutive_run_length(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mask), dtype=np.int16)
    current = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        current = current + 1 if value else 0
        result[index] = current
    return result


def diagnose_path(
    values: np.ndarray,
    segment: SegmentWindow,
    surface: LocalSurface,
) -> PathDiagnostics:
    indices, in_grid = nearest_state_indices(segment.grid_tvt, values)
    positions = np.arange(len(values), dtype=np.int32)
    endpoint_forbidden = (
        in_grid & surface.supported_rows & surface.barrier[positions, indices]
    )
    component_ids = np.full(len(values), -1, dtype=np.int32)
    component_lookup = in_grid & surface.supported_rows & ~endpoint_forbidden
    component_ids[component_lookup] = surface.component_labels[
        positions[component_lookup], indices[component_lookup]
    ]
    component_known = component_ids >= 0
    known_positions = np.flatnonzero(component_known)
    anchor_component = int(component_ids[known_positions[0]]) if len(known_positions) else -1
    component_transition = component_known & (component_ids != anchor_component)
    edge_crossing = np.zeros(len(values), dtype=bool)
    for position in range(1, len(values)):
        if segment.pixel_row_index[position] != segment.pixel_row_index[position - 1] + 1:
            continue
        if not (in_grid[position - 1] and in_grid[position]):
            continue
        if not (surface.supported_rows[position - 1] and surface.supported_rows[position]):
            continue
        low = int(min(indices[position - 1], indices[position]))
        high = int(max(indices[position - 1], indices[position])) + 1
        edge_crossing[position] = bool(
            surface.barrier[position - 1 : position + 1, low:high].any()
        )
    instantaneous = endpoint_forbidden | edge_crossing | component_transition
    invalid_history = np.maximum.accumulate(instantaneous.astype(np.int8)).astype(bool)
    endpoint_mismatch = np.full(len(values), np.nan, dtype=np.float32)
    endpoint_mismatch[in_grid] = surface.absolute_mismatch[
        positions[in_grid], indices[in_grid]
    ]
    return PathDiagnostics(
        in_grid=in_grid,
        endpoint_forbidden=endpoint_forbidden,
        component_known=component_known,
        anchor_component=anchor_component,
        component_transition=component_transition,
        edge_crossing=edge_crossing,
        instantaneous_violation=instantaneous,
        valid_within_segment=~invalid_history,
        violation_run_length=consecutive_run_length(instantaneous),
        endpoint_abs_mismatch=endpoint_mismatch,
    )


def path_values_for_segment(
    tail_values: np.ndarray,
    segment: SegmentWindow,
    tail_start: int,
) -> np.ndarray:
    values = np.full(len(segment.pixel_row_index), np.nan, dtype=float)
    valid = segment.pixel_row_index >= tail_start
    tail_positions = segment.pixel_row_index[valid] - tail_start
    inside = (tail_positions >= 0) & (tail_positions < len(tail_values))
    valid_positions = np.flatnonzero(valid)[inside]
    values[valid_positions] = tail_values[tail_positions[inside]]
    return values


def overlap_coverage(
    segments: list[SegmentWindow],
    tail_start: int,
    tail_stop: int,
) -> np.ndarray:
    coverage = np.zeros(tail_stop - tail_start + 1, dtype=np.int16)
    for segment in segments:
        rows = segment.pixel_row_index[segment.audit_pixel_positions]
        coverage[rows - tail_start] += 1
    if (coverage <= 0).any():
        missing = np.flatnonzero(coverage <= 0)[:10] + tail_start
        raise AssertionError(f"Segment contract leaves evaluation rows uncovered: {missing}")
    return coverage


def distance_bucket(values: np.ndarray, config: dict[str, Any]) -> pd.Categorical:
    edges = [float(value) for value in nested(config, "audit.distance_buckets.edges")]
    labels = [str(value) for value in nested(config, "audit.distance_buckets.labels")]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True)


def build_path_view_frame(
    *,
    well: str,
    path_name: str,
    is_truth: bool,
    values: np.ndarray,
    truth_values: np.ndarray,
    diagnostics: PathDiagnostics,
    segment: SegmentWindow,
    surface: LocalSurface,
    cache: pd.DataFrame,
    coverage: np.ndarray,
    tail_start: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    pixel = segment.audit_pixel_positions
    global_rows = segment.pixel_row_index[pixel]
    tail_positions = global_rows - tail_start
    prediction = values[pixel]
    truth = truth_values[pixel]
    error = np.abs(prediction - truth)
    if is_truth:
        error = np.zeros(len(pixel), dtype=float)
    threshold = float(nested(config, "audit.bad_candidate_threshold_ft"))
    boundary_margin = int(nested(config, "audit.segment.boundary_margin_rows"))
    if is_truth:
        is_bad = np.zeros(len(pixel), dtype=bool)
        is_good = np.zeros(len(pixel), dtype=bool)
    else:
        is_bad = error > threshold
        is_good = error <= threshold
    frame = pd.DataFrame(
        {
            "id": cache["id"].to_numpy(str)[tail_positions],
            "well": well,
            "row_index": global_rows,
            "md_since": cache["md_since"].to_numpy(float)[tail_positions],
            "segment_id": segment.segment_id,
            "center_row": segment.center_row,
            "pixel_position": pixel,
            "path": path_name,
            "is_truth": is_truth,
            "prediction": prediction,
            "true_tvt": truth,
            "abs_error": error,
            "is_bad": is_bad,
            "is_good": is_good,
            "view_weight": 1.0 / coverage[tail_positions].astype(float),
            "is_boundary": (pixel < boundary_margin)
            | (pixel >= len(segment.pixel_row_index) - boundary_margin),
            "in_grid": diagnostics.in_grid[pixel],
            "endpoint_forbidden": diagnostics.endpoint_forbidden[pixel],
            "component_known": diagnostics.component_known[pixel],
            "component_transition": diagnostics.component_transition[pixel],
            "edge_crossing": diagnostics.edge_crossing[pixel],
            "instantaneous_violation": diagnostics.instantaneous_violation[pixel],
            "valid_within_segment": diagnostics.valid_within_segment[pixel],
            "violation_run_length": diagnostics.violation_run_length[pixel],
            "endpoint_abs_mismatch": diagnostics.endpoint_abs_mismatch[pixel],
            "barrier_fraction": surface.barrier.mean(axis=1)[pixel],
            "anchor_component": diagnostics.anchor_component,
        }
    )
    frame["distance_bucket"] = distance_bucket(frame["md_since"].to_numpy(float), config)
    return frame


def build_well_views(
    well: str,
    cache: pd.DataFrame,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prefix_end: int,
    candidates: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[SegmentWindow]]:
    tail_start = prefix_end + 1
    tail_stop = len(horizontal) - 1
    centers = segment_centers(tail_start, tail_stop, config)
    segments = [
        build_segment_window(well, int(center), horizontal, prefix_end, config)
        for center in centers
    ]
    coverage = overlap_coverage(segments, tail_start, tail_stop)
    target_column = str(nested(config, "data.target_column", "TVT"))
    truth_full = horizontal[target_column].to_numpy(float)
    view_frames: list[pd.DataFrame] = []
    segment_summaries: list[dict[str, Any]] = []
    path_tail_values = {"truth": truth_full[tail_start:]}
    path_tail_values.update(candidates)
    for segment in segments:
        surface = build_local_surface(horizontal, typewell, segment, config)
        for path_name, tail_values in path_tail_values.items():
            path_values = path_values_for_segment(tail_values, segment, tail_start)
            diagnostics = diagnose_path(path_values, segment, surface)
            frame = build_path_view_frame(
                well=well,
                path_name=path_name,
                is_truth=path_name == "truth",
                values=path_values,
                truth_values=truth_full[segment.pixel_row_index],
                diagnostics=diagnostics,
                segment=segment,
                surface=surface,
                cache=cache,
                coverage=coverage,
                tail_start=tail_start,
                config=config,
            )
            view_frames.append(frame)
            weight = frame["view_weight"].to_numpy(float)
            signal = frame["instantaneous_violation"].to_numpy(bool)
            segment_summaries.append(
                {
                    "well": well,
                    "segment_id": segment.segment_id,
                    "center_row": segment.center_row,
                    "path": path_name,
                    "is_truth": path_name == "truth",
                    "rows": len(frame),
                    "weight_sum": float(weight.sum()),
                    "signal_weight": float(weight[signal].sum()),
                    "endpoint_forbidden_rows": int(frame["endpoint_forbidden"].sum()),
                    "edge_crossing_rows": int(frame["edge_crossing"].sum()),
                    "component_transition_rows": int(frame["component_transition"].sum()),
                    "max_violation_run_length": int(frame["violation_run_length"].max()),
                    "mean_endpoint_abs_mismatch": float(
                        frame["endpoint_abs_mismatch"].mean()
                    ),
                    "mean_candidate_relative_barrier_exposure": float(
                        frame["barrier_fraction"].mean()
                    ),
                    "bad_rows": int(frame["is_bad"].sum()),
                    "good_rows": int(frame["is_good"].sum()),
                }
            )
    views = pd.concat(view_frames, ignore_index=True)
    expected_weight = len(cache) * len(path_tail_values)
    if not np.isclose(views["view_weight"].sum(), expected_weight, atol=1e-6):
        raise AssertionError("Inverse overlap coverage weights do not sum to unique rows")
    return views, pd.DataFrame(segment_summaries), segments


# %% [markdown]
# ## 6. Stage 0 preview helpers

# %%
def select_preview_blocks(
    cache_path: Path,
    specs: list[CandidateSpec],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    preferred = [str(value) for value in nested(config, "audit.stage0.preferred_preview_wells")]
    fallback_count = int(nested(config, "audit.stage0.fallback_preview_well_count"))
    selected: dict[str, pd.DataFrame] = {}
    fallback: list[tuple[str, pd.DataFrame]] = []
    for well, cache in iter_candidate_cache_wells(
        cache_path,
        specs,
        int(nested(config, "audit.cache_chunksize")),
    ):
        if len(fallback) < fallback_count:
            fallback.append((well, cache))
        if well in preferred:
            selected[well] = cache
        if all(well_id in selected for well_id in preferred) and len(fallback) >= fallback_count:
            break
    for well, cache in fallback:
        selected.setdefault(well, cache)
    return {well: selected[well] for well in sorted(selected)}


def stage0_center_indices(centers: np.ndarray, config: dict[str, Any]) -> list[int]:
    requested = [str(value) for value in nested(config, "audit.stage0.center_positions_per_well")]
    mapping = {"first": 0, "middle": len(centers) // 2, "last": len(centers) - 1}
    indices = [mapping[value] for value in requested]
    return sorted(set(indices))


def plot_preview(
    *,
    well: str,
    segment: SegmentWindow,
    surface: LocalSurface,
    horizontal: pd.DataFrame,
    prefix_end: int,
    candidates: dict[str, np.ndarray],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    clip = float(nested(config, "audit.normalization.signed_display_clip"))
    extent = [
        float(segment.grid_tvt[0]),
        float(segment.grid_tvt[-1]),
        len(segment.pixel_row_index) - 0.5,
        -0.5,
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    signed_image = axes[0].imshow(
        surface.signed_mismatch,
        cmap="coolwarm",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=extent,
    )
    figure.colorbar(signed_image, ax=axes[0], label="typewell_z - horizontal_z")
    absolute_image = axes[1].imshow(
        surface.absolute_mismatch,
        cmap="Blues_r",
        vmin=0.0,
        vmax=clip,
        aspect="auto",
        extent=extent,
    )
    overlay = np.ma.masked_where(~surface.barrier, surface.barrier)
    axes[1].imshow(
        overlay,
        cmap="Reds",
        alpha=0.55,
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    figure.colorbar(absolute_image, ax=axes[1], label="absolute local mismatch")
    target_column = str(nested(config, "data.target_column", "TVT"))
    truth = horizontal[target_column].to_numpy(float)[segment.pixel_row_index]
    y = np.arange(len(segment.pixel_row_index))
    for axis in axes:
        axis.plot(truth, y, color="white", linewidth=1.8, label="truth (scoring only)")
        for name, tail_values in candidates.items():
            values = path_values_for_segment(tail_values, segment, prefix_end + 1)
            axis.plot(values, y, linewidth=0.8, alpha=0.75, label=name)
        axis.set_xlabel("TVT (ft)")
        axis.set_ylabel("local pixel row; label shows horizontal row")
        axis.set_yticks(np.linspace(0, len(y) - 1, 9).astype(int))
        axis.set_yticklabels(segment.pixel_row_index[axis.get_yticks().astype(int)])
        axis.axhline(len(y) // 2, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Signed mismatch: exp202 axis contract")
    axes[1].set_title("Absolute mismatch with red ridge barrier")
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    figure.suptitle(
        f"{well} center={segment.center_row} prior={segment.prior_center_tvt:.2f} "
        f"shape={surface.signed_mismatch.shape}"
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_stage0(
    config: dict[str, Any],
    cache_path: Path,
    train_dir: Path,
    specs: list[CandidateSpec],
    out_dir: Path,
) -> dict[str, Any]:
    preview_blocks = select_preview_blocks(cache_path, specs, config)
    if not preview_blocks:
        raise ValueError("No preview wells were resolved")
    metadata_rows: list[dict[str, Any]] = []
    preview_files: list[Path] = []
    raw_shas: dict[str, str] = {}
    for well, cache in preview_blocks.items():
        horizontal, typewell, prefix_end = load_raw_well(well, cache, train_dir, config)
        suffixes = [
            str(nested(config, "data.horizontal_suffix", "__horizontal_well.csv")),
            str(nested(config, "data.typewell_suffix", "__typewell.csv")),
        ]
        for suffix in suffixes:
            path = train_dir / f"{well}{suffix}"
            raw_shas[path.name] = sha256_path(path)
        candidates = materialize_candidates(cache, specs)
        centers = segment_centers(prefix_end + 1, len(horizontal) - 1, config)
        for center_index in stage0_center_indices(centers, config):
            segment = build_segment_window(
                well,
                int(centers[center_index]),
                horizontal,
                prefix_end,
                config,
            )
            surface = build_local_surface(horizontal, typewell, segment, config)
            png_path = out_dir / f"preview_{segment.segment_id}.png"
            plot_preview(
                well=well,
                segment=segment,
                surface=surface,
                horizontal=horizontal,
                prefix_end=prefix_end,
                candidates=candidates,
                output_path=png_path,
                config=config,
            )
            preview_files.append(png_path)
            metadata_rows.append(
                {
                    "well": well,
                    "segment_id": segment.segment_id,
                    "center_row": segment.center_row,
                    "pixel_rows": len(segment.pixel_row_index),
                    "unique_horizontal_rows": int(np.unique(segment.pixel_row_index).size),
                    "typewell_bins": len(segment.grid_tvt),
                    "grid_tvt_min": float(segment.grid_tvt[0]),
                    "grid_tvt_max": float(segment.grid_tvt[-1]),
                    "grid_step_ft": surface.grid_step_ft,
                    "prior_center_tvt": segment.prior_center_tvt,
                    "horizontal_center": surface.horizontal_center,
                    "horizontal_iqr_scale": surface.horizontal_scale,
                    "typewell_center": surface.typewell_center,
                    "typewell_iqr_scale": surface.typewell_scale,
                    "supported_row_rate": float(surface.supported_rows.mean()),
                    "barrier_cell_rate": float(surface.barrier.mean()),
                    "component_count": int(
                        np.unique(surface.component_labels[surface.component_labels >= 0]).size
                    ),
                    "png_filename": png_path.name,
                    "png_sha256": sha256_path(png_path),
                }
            )
    metadata = pd.DataFrame(metadata_rows).sort_values(["well", "center_row"])
    metadata_path = out_dir / str(
        nested(config, "audit.outputs.preview_pixel_metadata_filename")
    )
    metadata.to_csv(metadata_path, index=False)
    manifest_path = out_dir / str(nested(config, "audit.outputs.preview_manifest_filename"))
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "mode": "stage0_preview",
        "decision": "manual_parity_confirmation_required",
        "manual_parity_confirmed": bool(
            nested(config, "audit.stage0.manual_parity_confirmed")
        ),
        "contract": {
            "horizontal_window_rows": nested(
                config, "audit.segment.horizontal_window_rows"
            ),
            "typewell_window_bins": nested(config, "audit.segment.typewell_window_bins"),
            "tvt_grid_half_width_ft": nested(
                config, "audit.segment.tvt_grid_half_width_ft"
            ),
            "row_center_stride": nested(config, "audit.segment.row_center_stride"),
            "normalization_scope": nested(config, "audit.normalization.scope"),
        },
        "preview_wells": sorted(preview_blocks),
        "preview_count": len(metadata),
        "generated_files": {
            path.name: sha256_path(path) for path in [*preview_files, metadata_path]
        },
        "inputs": {
            "candidate_cache_raw_sha256": sha256_path(cache_path),
            "candidate_cache_decompressed_sha256": sha256_path(
                cache_path, decompressed=True
            ),
            "raw_preview_file_inventory_sha256": sha256_mapping(raw_shas),
            "config_sha256": sha256_path(find_config_path()),
        },
    }
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_filename": manifest_path.name}


# %% [markdown]
# ## 7. Stage 1 metric and generated-file helpers

# %%
METRIC_COLUMNS = [
    "total_weight",
    "flag_weight",
    "bad_weight",
    "good_weight",
    "flag_bad_weight",
    "flag_good_weight",
    "endpoint_weight",
    "crossing_weight",
    "transition_weight",
]


def metric_contribution(
    frame: pd.DataFrame,
    group_name: str,
    mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = frame.loc[mask]
    for path, group in selected.groupby("path", sort=False):
        weight = group["view_weight"].to_numpy(float)
        flag = group["instantaneous_violation"].to_numpy(bool)
        bad = group["is_bad"].to_numpy(bool)
        good = group["is_good"].to_numpy(bool)
        rows.append(
            {
                "group": group_name,
                "path": path,
                "is_truth": bool(group["is_truth"].iloc[0]),
                "total_weight": float(weight.sum()),
                "flag_weight": float(weight[flag].sum()),
                "bad_weight": float(weight[bad].sum()),
                "good_weight": float(weight[good].sum()),
                "flag_bad_weight": float(weight[flag & bad].sum()),
                "flag_good_weight": float(weight[flag & good].sum()),
                "endpoint_weight": float(
                    weight[group["endpoint_forbidden"].to_numpy(bool)].sum()
                ),
                "crossing_weight": float(
                    weight[group["edge_crossing"].to_numpy(bool)].sum()
                ),
                "transition_weight": float(
                    weight[group["component_transition"].to_numpy(bool)].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def add_all_candidate_rows(contributions: pd.DataFrame) -> pd.DataFrame:
    candidate = contributions.loc[~contributions["is_truth"]]
    if candidate.empty:
        return contributions
    grouped = candidate.groupby("group", as_index=False)[METRIC_COLUMNS].sum()
    grouped["path"] = "__all_candidates__"
    grouped["is_truth"] = False
    columns = ["group", "path", "is_truth", *METRIC_COLUMNS]
    return pd.concat([contributions, grouped[columns]], ignore_index=True)


def finalize_metric_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        contributions.groupby(["group", "path", "is_truth"], as_index=False)[
            METRIC_COLUMNS
        ]
        .sum()
        .sort_values(["group", "path"])
    )
    grouped["flag_rate"] = grouped["flag_weight"] / grouped["total_weight"].clip(lower=1e-12)
    grouped["base_bad_rate"] = grouped["bad_weight"] / grouped["total_weight"].clip(
        lower=1e-12
    )
    grouped["flag_bad_precision"] = grouped["flag_bad_weight"] / grouped[
        "flag_weight"
    ].clip(lower=1e-12)
    grouped["bad_precision_lift"] = grouped["flag_bad_precision"] / grouped[
        "base_bad_rate"
    ].clip(lower=1e-12)
    grouped["good_false_alert_rate"] = grouped["flag_good_weight"] / grouped[
        "good_weight"
    ].clip(lower=1e-12)
    grouped["bad_recall"] = grouped["flag_bad_weight"] / grouped["bad_weight"].clip(
        lower=1e-12
    )
    grouped["endpoint_rate"] = grouped["endpoint_weight"] / grouped[
        "total_weight"
    ].clip(lower=1e-12)
    grouped["crossing_rate"] = grouped["crossing_weight"] / grouped[
        "total_weight"
    ].clip(lower=1e-12)
    grouped["component_transition_rate"] = grouped["transition_weight"] / grouped[
        "total_weight"
    ].clip(lower=1e-12)
    return grouped


def overlap_contribution(frame: pd.DataFrame) -> pd.DataFrame:
    flag_columns = [
        "instantaneous_violation",
        "endpoint_forbidden",
        "edge_crossing",
        "component_transition",
    ]
    grouped = frame.groupby(["path", "id"], sort=False)[flag_columns].agg(["min", "max", "count"])
    rows: list[dict[str, Any]] = []
    for path, path_group in grouped.groupby(level=0, sort=False):
        counts = path_group[("instantaneous_violation", "count")].to_numpy(int)
        overlap = counts >= 2
        row: dict[str, Any] = {
            "path": path,
            "overlap_rows": int(overlap.sum()),
            "overlap_views": int(counts[overlap].sum()),
        }
        for column in flag_columns:
            minimum = path_group[(column, "min")].to_numpy(bool)
            maximum = path_group[(column, "max")].to_numpy(bool)
            disagreement = overlap & (minimum != maximum)
            row[f"{column}_disagreement_rows"] = int(disagreement.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_overlap(contributions: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in contributions.columns if column != "path"]
    result = contributions.groupby("path", as_index=False)[numeric].sum()
    for column in [
        "instantaneous_violation",
        "endpoint_forbidden",
        "edge_crossing",
        "component_transition",
    ]:
        result[f"{column}_disagreement_rate"] = result[
            f"{column}_disagreement_rows"
        ] / result["overlap_rows"].clip(lower=1)
    candidate = result.loc[result["path"] != "truth", numeric].sum().to_dict()
    candidate["path"] = "__all_candidates__"
    candidate["instantaneous_violation_disagreement_rate"] = (
        candidate["instantaneous_violation_disagreement_rows"]
        / max(candidate["overlap_rows"], 1)
    )
    candidate["endpoint_forbidden_disagreement_rate"] = (
        candidate["endpoint_forbidden_disagreement_rows"]
        / max(candidate["overlap_rows"], 1)
    )
    candidate["edge_crossing_disagreement_rate"] = (
        candidate["edge_crossing_disagreement_rows"]
        / max(candidate["overlap_rows"], 1)
    )
    candidate["component_transition_disagreement_rate"] = (
        candidate["component_transition_disagreement_rows"]
        / max(candidate["overlap_rows"], 1)
    )
    return pd.concat([result, pd.DataFrame([candidate])], ignore_index=True)


def boundary_contribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (path, boundary), group in frame.groupby(["path", "is_boundary"], sort=False):
        weight = group["view_weight"].to_numpy(float)
        flag = group["instantaneous_violation"].to_numpy(bool)
        good = group["is_good"].to_numpy(bool)
        rows.append(
            {
                "path": path,
                "region": "boundary" if boundary else "core",
                "total_weight": float(weight.sum()),
                "flag_weight": float(weight[flag].sum()),
                "good_weight": float(weight[good].sum()),
                "flag_good_weight": float(weight[flag & good].sum()),
            }
        )
    return pd.DataFrame(rows)


def finalize_boundary(contributions: pd.DataFrame) -> pd.DataFrame:
    numeric = ["total_weight", "flag_weight", "good_weight", "flag_good_weight"]
    result = contributions.groupby(["path", "region"], as_index=False)[numeric].sum()
    result["flag_rate"] = result["flag_weight"] / result["total_weight"].clip(lower=1e-12)
    result["good_false_alert_rate"] = result["flag_good_weight"] / result[
        "good_weight"
    ].clip(lower=1e-12)
    candidate = result.loc[result["path"] != "truth"].groupby("region", as_index=False)[
        numeric
    ].sum()
    candidate["path"] = "__all_candidates__"
    candidate["flag_rate"] = candidate["flag_weight"] / candidate["total_weight"].clip(
        lower=1e-12
    )
    candidate["good_false_alert_rate"] = candidate["flag_good_weight"] / candidate[
        "good_weight"
    ].clip(lower=1e-12)
    result = pd.concat([result, candidate[result.columns]], ignore_index=True)
    pivot = result.pivot(index="path", columns="region", values="good_false_alert_rate")
    delta = (pivot.get("boundary", 0.0) - pivot.get("core", 0.0)).rename(
        "boundary_core_good_false_alert_delta"
    )
    return result.merge(delta, on="path", how="left")


def metric_groups_for_well(
    views: pd.DataFrame,
    well: str,
    hidden_groups: dict[str, set[str]],
    config: dict[str, Any],
) -> list[pd.DataFrame]:
    contributions = [
        metric_contribution(views, "overall", np.ones(len(views), dtype=bool))
    ]
    for label in [str(value) for value in nested(config, "audit.distance_buckets.labels")]:
        mask = views["distance_bucket"].astype(str).to_numpy() == label
        if mask.any():
            contributions.append(metric_contribution(views, f"distance:{label}", mask))
    for name, wells in hidden_groups.items():
        if well in wells:
            contributions.append(
                metric_contribution(views, f"hidden_like:{name}", np.ones(len(views), bool))
            )
    return contributions


def by_well_readout(well: str, contribution: pd.DataFrame) -> dict[str, Any]:
    finalized = finalize_metric_contributions(add_all_candidate_rows(contribution))
    overall = finalized.loc[finalized["group"] == "overall"].set_index("path")
    truth = overall.loc["truth"]
    candidates = overall.loc["__all_candidates__"]
    return {
        "well": well,
        "truth_false_alert_rate": float(truth["flag_rate"]),
        "candidate_flag_rate": float(candidates["flag_rate"]),
        "candidate_good_false_alert_rate": float(candidates["good_false_alert_rate"]),
        "candidate_bad_precision_lift": float(candidates["bad_precision_lift"]),
        "candidate_bad_recall": float(candidates["bad_recall"]),
    }


def evaluate_stage1_guards(
    group_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    boundary_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    overall = group_metrics.loc[group_metrics["group"] == "overall"].set_index("path")
    truth_rate = float(overall.loc["truth", "flag_rate"])
    candidate = overall.loc["__all_candidates__"]
    overlap = overlap_metrics.set_index("path").loc["__all_candidates__"]
    boundary = boundary_metrics.set_index("path").loc["__all_candidates__"].iloc[0]
    hidden_truth = group_metrics.loc[
        group_metrics["group"].str.startswith("hidden_like:")
        & (group_metrics["path"] == "truth"),
        "flag_rate",
    ]
    values = {
        "truth_instantaneous_false_alert_rate": (
            truth_rate,
            float(nested(config, "audit.guards.max_truth_instantaneous_false_alert_rate")),
            "max",
        ),
        "good_candidate_false_alert_rate": (
            float(candidate["good_false_alert_rate"]),
            float(nested(config, "audit.guards.max_good_candidate_false_alert_rate")),
            "max",
        ),
        "bad_candidate_precision_lift": (
            float(candidate["bad_precision_lift"]),
            float(nested(config, "audit.guards.min_bad_candidate_precision_lift")),
            "min",
        ),
        "overlap_view_disagreement_rate": (
            float(overlap["instantaneous_violation_disagreement_rate"]),
            float(nested(config, "audit.guards.max_overlap_view_disagreement_rate")),
            "max",
        ),
        "boundary_core_false_alert_delta": (
            float(boundary["boundary_core_good_false_alert_delta"]),
            float(nested(config, "audit.guards.max_boundary_core_false_alert_delta")),
            "max",
        ),
        "hidden_like_truth_false_alert_rate": (
            float(hidden_truth.max()) if len(hidden_truth) else np.nan,
            float(nested(config, "audit.guards.max_hidden_like_truth_false_alert_rate")),
            "max",
        ),
        "worst_well_truth_false_alert_rate": (
            float(by_well["truth_false_alert_rate"].max()),
            float(nested(config, "audit.guards.max_worst_well_truth_false_alert_rate")),
            "max",
        ),
    }
    guards: dict[str, dict[str, Any]] = {}
    for name, (value, limit, direction) in values.items():
        passed = bool(value <= limit) if direction == "max" else bool(value >= limit)
        guards[name] = {
            "value": value,
            "limit": limit,
            "direction": direction,
            "pass": passed,
        }
    return guards


def run_stage1(
    config: dict[str, Any],
    cache_path: Path,
    hidden_path: Path,
    train_dir: Path,
    specs: list[CandidateSpec],
    out_dir: Path,
) -> dict[str, Any]:
    if not bool(nested(config, "audit.stage0.manual_parity_confirmed")) or not bool(
        nested(config, "audit.stage1.enabled_after_stage0_confirmation")
    ):
        raise RuntimeError(
            "Stage 1 is blocked until Stage 0 pixel/axis/normalization parity is "
            "manually confirmed in config."
        )
    hidden_frame, hidden_groups = load_hidden_like_roles(hidden_path)
    output_names = nested(config, "audit.outputs")
    segment_path = out_dir / str(output_names["segment_path_summary_filename"])
    event_path = out_dir / str(output_names["flagged_events_filename"])
    metric_contributions: list[pd.DataFrame] = []
    overlap_contributions: list[pd.DataFrame] = []
    boundary_contributions: list[pd.DataFrame] = []
    by_well_rows: list[dict[str, Any]] = []
    raw_shas: dict[str, str] = {}
    processed_rows = 0
    processed_wells = 0
    max_wells = nested(config, "audit.max_wells")
    with DeterministicGzipCsvWriter(segment_path) as segment_writer, DeterministicGzipCsvWriter(
        event_path
    ) as event_writer:
        for well, cache in iter_candidate_cache_wells(
            cache_path,
            specs,
            int(nested(config, "audit.cache_chunksize")),
        ):
            if max_wells is not None and processed_wells >= int(max_wells):
                break
            horizontal, typewell, prefix_end = load_raw_well(well, cache, train_dir, config)
            for suffix in [
                str(nested(config, "data.horizontal_suffix", "__horizontal_well.csv")),
                str(nested(config, "data.typewell_suffix", "__typewell.csv")),
            ]:
                path = train_dir / f"{well}{suffix}"
                raw_shas[path.name] = sha256_path(path)
            candidates = materialize_candidates(cache, specs)
            views, segment_summary, _ = build_well_views(
                well,
                cache,
                horizontal,
                typewell,
                prefix_end,
                candidates,
                config,
            )
            segment_writer.write(segment_summary)
            event_columns = [
                "id",
                "well",
                "row_index",
                "md_since",
                "distance_bucket",
                "segment_id",
                "center_row",
                "pixel_position",
                "path",
                "is_truth",
                "abs_error",
                "is_bad",
                "is_good",
                "view_weight",
                "is_boundary",
                "endpoint_forbidden",
                "edge_crossing",
                "component_transition",
                "violation_run_length",
                "endpoint_abs_mismatch",
                "barrier_fraction",
            ]
            event_writer.write(views.loc[views["instantaneous_violation"], event_columns])
            well_metrics = metric_groups_for_well(
                views,
                well,
                hidden_groups,
                config,
            )
            metric_contributions.extend(well_metrics)
            overall_contribution = well_metrics[0]
            by_well_rows.append(by_well_readout(well, overall_contribution))
            overlap_contributions.append(overlap_contribution(views))
            boundary_contributions.append(boundary_contribution(views))
            processed_rows += len(cache)
            processed_wells += 1
            if processed_wells % 25 == 0:
                print(f"processed wells={processed_wells} rows={processed_rows}")
    for writer_path in [segment_path, event_path]:
        if not writer_path.exists() or writer_path.stat().st_size == 0:
            raise AssertionError(f"Expected non-empty gzip output: {writer_path}")
    contributions = add_all_candidate_rows(pd.concat(metric_contributions, ignore_index=True))
    group_metrics = finalize_metric_contributions(contributions)
    candidate_metrics = group_metrics.loc[
        (group_metrics["group"] == "overall") & ~group_metrics["is_truth"]
    ].reset_index(drop=True)
    overlap_metrics = finalize_overlap(pd.concat(overlap_contributions, ignore_index=True))
    boundary_metrics = finalize_boundary(pd.concat(boundary_contributions, ignore_index=True))
    by_well = pd.DataFrame(by_well_rows).sort_values(
        "truth_false_alert_rate", ascending=False
    )
    paths = {
        "candidate_metrics": out_dir / str(output_names["candidate_metrics_filename"]),
        "group_metrics": out_dir / str(output_names["group_metrics_filename"]),
        "overlap_metrics": out_dir / str(output_names["overlap_metrics_filename"]),
        "boundary_metrics": out_dir / str(output_names["boundary_metrics_filename"]),
        "by_well": out_dir / str(output_names["by_well_filename"]),
    }
    candidate_metrics.to_csv(paths["candidate_metrics"], index=False)
    group_metrics.to_csv(paths["group_metrics"], index=False)
    overlap_metrics.to_csv(paths["overlap_metrics"], index=False)
    boundary_metrics.to_csv(paths["boundary_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    guards = evaluate_stage1_guards(
        group_metrics,
        overlap_metrics,
        boundary_metrics,
        by_well,
        config,
    )
    output_shas = {
        segment_path.name: {
            "raw_sha256": sha256_path(segment_path),
            "decompressed_sha256": sha256_path(segment_path, decompressed=True),
        },
        event_path.name: {
            "raw_sha256": sha256_path(event_path),
            "decompressed_sha256": sha256_path(event_path, decompressed=True),
        },
    }
    output_shas.update(
        {path.name: {"sha256": sha256_path(path)} for path in paths.values()}
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": "stage1_full_audit",
        "decision": (
            "segment_local_audit_guard_passed"
            if all(item["pass"] for item in guards.values())
            else "segment_local_audit_guard_failed"
        ),
        "processed_wells": processed_wells,
        "processed_rows": processed_rows,
        "hidden_assignment_rows": len(hidden_frame),
        "guards": guards,
        "inputs": {
            "candidate_cache_raw_sha256": sha256_path(cache_path),
            "candidate_cache_decompressed_sha256": sha256_path(
                cache_path, decompressed=True
            ),
            "hidden_like_sha256": sha256_path(hidden_path),
            "raw_file_inventory_sha256": sha256_mapping(raw_shas),
            "config_sha256": sha256_path(find_config_path()),
        },
        "outputs": output_shas,
    }


# %% [markdown]
# ## 8. Synthetic contract checks

# %%
def run_synthetic_contract_checks(config: dict[str, Any]) -> None:
    barrier = np.zeros((8, 10), dtype=bool)
    barrier[:, 4:6] = True
    supported = np.ones(8, dtype=bool)
    rows = np.arange(8, dtype=np.int32)
    labels = label_free_components(barrier, supported, rows, shift_bins=1)
    if np.unique(labels[:, :4]).size != 1 or np.unique(labels[:, 6:]).size != 1:
        raise AssertionError("Synthetic free corridors must each be connected")
    if labels[0, 2] == labels[0, 7]:
        raise AssertionError("Synthetic red ridge must separate blue corridors")
    segment = SegmentWindow(
        segment_id="synthetic_0000000",
        center_row=0,
        pixel_row_index=rows,
        audit_pixel_positions=rows,
        grid_tvt=np.arange(10, dtype=np.float32),
        prior_center_tvt=4.5,
    )
    surface = LocalSurface(
        signed_mismatch=barrier.astype(np.float32) * 3.0,
        absolute_mismatch=barrier.astype(np.float32) * 3.0,
        barrier=barrier,
        supported_rows=supported,
        raw_high_fraction=barrier.mean(axis=1).astype(np.float32),
        component_labels=labels,
        horizontal_center=0.0,
        horizontal_scale=1.0,
        typewell_center=0.0,
        typewell_scale=1.0,
        grid_step_ft=1.0,
        min_tvt_thickness_bins=2,
    )
    stay_left = diagnose_path(np.full(8, 2.0), segment, surface)
    if stay_left.instantaneous_violation.any():
        raise AssertionError("Path staying in one local corridor must remain unflagged")
    jump_right = diagnose_path(np.array([2, 2, 2, 2, 7, 7, 7, 7], float), segment, surface)
    if not jump_right.edge_crossing[4] or not jump_right.component_transition[4:].all():
        raise AssertionError("Ridge jump must produce crossing and component transition")
    fresh_right = diagnose_path(np.full(8, 7.0), segment, surface)
    if fresh_right.instantaneous_violation.any():
        raise AssertionError("A fresh segment must reset component anchor/history")
    if jump_right.violation_run_length[-1] != 4:
        raise AssertionError("Within-segment persistence length is incorrect")
    segment_a = SegmentWindow("a", 1, rows, np.array([0, 1, 2, 3]), np.arange(10), 4.5)
    segment_b = SegmentWindow("b", 5, rows, np.array([2, 3, 4, 5]), np.arange(10), 4.5)
    coverage = overlap_coverage([segment_a, segment_b], 0, 5)
    if not np.array_equal(coverage, np.array([1, 1, 2, 2, 1, 1])):
        raise AssertionError("Synthetic overlap coverage is incorrect")
    inverse_sum = np.zeros(6, dtype=float)
    for synthetic in [segment_a, segment_b]:
        covered = synthetic.pixel_row_index[synthetic.audit_pixel_positions]
        inverse_sum[covered] += 1.0 / coverage[covered]
    if not np.allclose(inverse_sum, 1.0):
        raise AssertionError("Inverse coverage weights must sum to one per unique row")
    if int(nested(config, "audit.segment.horizontal_window_rows")) != 128:
        raise AssertionError("Stage 0 contract requires 128 horizontal rows")
    if int(nested(config, "audit.segment.typewell_window_bins")) != 64:
        raise AssertionError("Stage 0 contract requires 64 typewell bins")
    if int(nested(config, "audit.segment.row_center_stride")) != 64:
        raise AssertionError("Stage 0 contract requires stride 64")


# %% [markdown]
# ## 9. Setup and input checks

# %%
CONFIG = load_config()
require_authoritative_runtime()
OUT_DIR = output_dir()
SPECS = candidate_specs(CONFIG)
ACTIVE_MODE = str(nested(CONFIG, "audit.active_mode"))
ALLOWED_MODES = {str(value) for value in nested(CONFIG, "audit.allowed_modes")}
if ACTIVE_MODE not in ALLOWED_MODES:
    raise ValueError(f"Unsupported active mode: {ACTIVE_MODE}")
if bool(nested(CONFIG, "model.parent_control_retraining")):
    raise ValueError("Parent/control retraining must remain false")
if any(
    int(nested(CONFIG, key)) != 0
    for key in [
        "model.lightgbm_config_count",
        "model.fold_training_count",
        "model.booster_count",
    ]
):
    raise ValueError("This diagnostic must run with zero model configs/folds/boosters")

CACHE_PATH = resolve_file(
    CONFIG,
    "data.candidate_cache_paths",
    "data.candidate_cache_filename",
)
TRAIN_DIR = resolve_train_dir(CONFIG)
HIDDEN_PATH: Path | None = None
if ACTIVE_MODE == "stage1_full_audit":
    HIDDEN_PATH = resolve_file(
        CONFIG,
        "data.hidden_like_paths",
        "data.hidden_like_filename",
    )

run_synthetic_contract_checks(CONFIG)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", nested(CONFIG, "experiment.route"))
print("Active mode:", ACTIVE_MODE)
print("Parent:", nested(CONFIG, "lineage.parent"))
print("Window contract:", nested(CONFIG, "audit.segment"))
print("Normalization:", nested(CONFIG, "audit.normalization"))
print("Candidate cache:", CACHE_PATH)
print("Train dir:", TRAIN_DIR)
print(
    "Cost: diagnostic modes=1 LightGBM configs=0 folds=0 boosters=0 "
    "parent/control retraining=false GPU=false"
)

# %% [markdown]
# ## 10. Stage 0 / Stage 1 orchestration

# %%
STARTED_AT = time.time()
if ACTIVE_MODE == "stage0_preview":
    SUMMARY = run_stage0(CONFIG, CACHE_PATH, TRAIN_DIR, SPECS, OUT_DIR)
elif ACTIVE_MODE == "stage1_full_audit":
    if HIDDEN_PATH is None:
        raise AssertionError("Stage 1 requires hidden-like assignment")
    SUMMARY = run_stage1(
        CONFIG,
        CACHE_PATH,
        HIDDEN_PATH,
        TRAIN_DIR,
        SPECS,
        OUT_DIR,
    )
else:
    raise AssertionError(f"Unreachable mode: {ACTIVE_MODE}")
SUMMARY["runtime_seconds"] = time.time() - STARTED_AT
SUMMARY["route"] = nested(CONFIG, "experiment.route")
SUMMARY["cost"] = {
    "active_diagnostic_modes": 1,
    "lightgbm_configs": 0,
    "folds": 0,
    "boosters": 0,
    "parent_control_retraining": False,
}
SUMMARY["forbidden_actions"] = [
    "No overlap OR/AND/majority aggregation.",
    "No candidate pruning, averaging, selection, or value changes.",
    "No oracle/error-based segment or threshold selection.",
    "No HMM/PF/Beam edge cuts, raw-test inference, or submission.",
]

# %% [markdown]
# ## 11. Metrics, guards, SHA, and generated files

# %%
summary_path = OUT_DIR / str(nested(CONFIG, "audit.outputs.summary_filename"))
write_json(summary_path, SUMMARY)
METRICS = {
    "experiment": EXPERIMENT_NAME,
    "status": SUMMARY["decision"],
    "route": nested(CONFIG, "experiment.route"),
    "metric": "segment_local_negative_space_risk_audit",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "active_mode": ACTIVE_MODE,
    "runtime_seconds": SUMMARY["runtime_seconds"],
    "decision": SUMMARY["decision"],
    "guards": SUMMARY.get("guards"),
    "summary_filename": summary_path.name,
    "summary_sha256": sha256_path(summary_path),
    "cost": SUMMARY["cost"],
    "notes": (
        "Stage 0 requires manual image parity confirmation before Stage 1."
        if ACTIVE_MODE == "stage0_preview"
        else "Stage 1 is train-side diagnostic only; no candidate was changed."
    ),
}
write_json(metrics_output_path(), METRICS)
print(json.dumps(to_jsonable(METRICS), indent=2, sort_keys=True))
if ACTIVE_MODE == "stage0_preview":
    metadata_path = OUT_DIR / str(
        nested(CONFIG, "audit.outputs.preview_pixel_metadata_filename")
    )
    display(pd.read_csv(metadata_path))
else:
    candidate_path = OUT_DIR / str(nested(CONFIG, "audit.outputs.candidate_metrics_filename"))
    display(pd.read_csv(candidate_path))
