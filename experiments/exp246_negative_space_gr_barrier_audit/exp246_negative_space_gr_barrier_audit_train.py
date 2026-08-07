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
# # exp246 negative-space GR barrier audit — train
#
# This is a deterministic, no-training PF/Beam path diagnostic. It constructs
# conservative high-GR-mismatch ridges on a horizontal-row × typewell-TVT grid,
# identifies corridors reachable from the known-prefix TVT anchor, and measures
# whether fixed candidate paths cross those ridges. It never changes a candidate
# value, fits a model, runs hidden-test inference, or creates a submission.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input resolution
# 3. Candidate-cache streaming and raw-well contracts
# 4. GR mismatch surface and barrier construction
# 5. Anchor corridor and edge-crossing diagnostics
# 6. Metric aggregation and generated-file helpers
# 7. Synthetic contract checks
# 8. Setup and input checks
# 9. Full train-side audit orchestration
# 10. Metrics, guards, SHA, and generated files

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

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from scipy import ndimage

EXPERIMENT_NAME = "exp246_negative_space_gr_barrier_audit"
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
    if isinstance(value, (np.integer,)):
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
    if decompressed and path.suffix == ".gz":
        handle: Any = gzip.open(path, "rb")
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
        # GzipFile does not close a caller-owned fileobj. Flush and close the
        # underlying buffer so an immediate decompressed-content SHA read sees
        # the gzip trailer and every compressed byte.
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


def resolve_file(
    config: dict[str, Any],
    path_key: str,
    filename_key: str,
) -> Path:
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


def well_id_from_path(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected well filename: {path.name}")
    return path.name[: -len(suffix)]


# %% [markdown]
# ## 3. Candidate-cache streaming and raw-well contracts

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
    if not specs:
        raise ValueError("No candidate specs are configured")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Candidate names must be unique")
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
    if out[["id", "well", "row_index"]].isna().any().any():
        raise ValueError("Candidate cache key contains missing values")
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
            raise ValueError("Candidate cache must be globally sorted by well for streaming")
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
    if missing:
        raise ValueError(f"Hidden-like assignment is missing columns: {missing}")
    if frame["well_id"].duplicated().any():
        raise ValueError("Hidden-like assignment contains duplicate well_id")
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
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    horizontal_suffix = str(nested(config, "data.horizontal_suffix"))
    typewell_suffix = str(nested(config, "data.typewell_suffix"))
    horizontal_path = train_dir / f"{well}{horizontal_suffix}"
    typewell_path = train_dir / f"{well}{typewell_suffix}"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"Raw well pair is missing for {well}")
    md_column = str(nested(config, "data.md_column", "MD"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    target_column = str(nested(config, "data.target_column", "TVT"))
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=[md_column, gr_column, target_column, input_column],
    )
    typewell = pd.read_csv(typewell_path, usecols=[target_column, gr_column])
    for column in [md_column, gr_column, target_column, input_column]:
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
            f"{well} cache rows do not exactly match the official evaluation tail: "
            f"cache={len(row_index)} expected={len(expected)}"
        )
    true_raw = horizontal[target_column].to_numpy(float)[row_index]
    true_cache = (
        cache["last_known_tvt"].to_numpy(float) + cache["target"].to_numpy(float)
    )
    max_true_delta = float(np.nanmax(np.abs(true_raw - true_cache)))
    if not np.isfinite(max_true_delta) or max_true_delta > 0.15:
        raise ValueError(f"{well} raw/cache truth mismatch: max_abs={max_true_delta}")
    return horizontal, typewell, row_index, prefix_end


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
# ## 4. GR mismatch surface and barrier construction

# %%
@dataclass(frozen=True)
class BarrierSurface:
    grid_tvt: np.ndarray
    barrier: np.ndarray
    supported_rows: np.ndarray
    raw_high_fraction: np.ndarray
    actual_grid_step_ft: float
    min_tvt_thickness_bins: int


def fill_series(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    filled = series.interpolate(limit_direction="both").ffill().bfill()
    fallback = float(filled.dropna().median()) if filled.notna().any() else 0.0
    return filled.fillna(fallback).to_numpy(float)


def robust_zscore(values: np.ndarray, clip: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.zeros_like(values, dtype=np.float32)
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return np.clip((values - median) / scale, -clip, clip).astype(np.float32)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )


def build_state_grid(typewell_tvt: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    finite = np.asarray(typewell_tvt, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        raise ValueError("Typewell TVT has insufficient finite values")
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    if span <= 0:
        raise ValueError("Typewell TVT span must be positive")
    requested_step = float(nested(config, "audit.barrier.tvt_grid_step_ft"))
    min_bins = int(nested(config, "audit.barrier.min_state_bins"))
    max_bins = int(nested(config, "audit.barrier.max_state_bins"))
    bins = int(math.floor(span / requested_step)) + 1
    bins = int(np.clip(bins, min_bins, max_bins))
    return np.linspace(low, high, bins, dtype=np.float32)


def build_barrier_surface(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_index: np.ndarray,
    config: dict[str, Any],
) -> BarrierSurface:
    gr_column = str(nested(config, "data.gr_column", "GR"))
    target_column = str(nested(config, "data.target_column", "TVT"))
    horizontal_raw = horizontal[gr_column].to_numpy(float)
    horizontal_filled = fill_series(horizontal_raw)
    typewell_clean = (
        typewell[[target_column, gr_column]]
        .dropna()
        .groupby(target_column, as_index=False)[gr_column]
        .mean()
        .sort_values(target_column)
    )
    if len(typewell_clean) < 16:
        raise ValueError("Typewell has fewer than 16 finite TVT/GR rows")
    typewell_tvt = typewell_clean[target_column].to_numpy(float)
    typewell_gr = typewell_clean[gr_column].to_numpy(float)
    grid_tvt = build_state_grid(typewell_tvt, config)
    grid_gr = np.interp(grid_tvt, typewell_tvt, typewell_gr)

    smooth_window = int(nested(config, "audit.gr_surface.smooth_window_rows"))
    clip = float(nested(config, "audit.gr_surface.robust_clip"))
    horizontal_smooth = rolling_median(horizontal_filled, smooth_window)
    grid_smooth = rolling_median(grid_gr, smooth_window)
    h_raw_z = robust_zscore(horizontal_filled, clip)[row_index]
    h_smooth_z = robust_zscore(horizontal_smooth, clip)[row_index]
    t_raw_z = robust_zscore(grid_gr, clip)
    t_smooth_z = robust_zscore(grid_smooth, clip)

    raw_difference = np.abs(h_raw_z[:, None] - t_raw_z[None, :])
    smooth_difference = np.abs(h_smooth_z[:, None] - t_smooth_z[None, :])
    raw_threshold = float(
        nested(config, "audit.gr_surface.raw_abs_difference_threshold")
    )
    smooth_threshold = float(
        nested(config, "audit.gr_surface.smooth_abs_difference_threshold")
    )
    high = (raw_difference >= raw_threshold) & (smooth_difference >= smooth_threshold)
    raw_high_fraction = high.mean(axis=1).astype(np.float32)

    original_finite = np.isfinite(horizontal_raw[row_index])
    flat_window = int(nested(config, "audit.gr_surface.flat_window_rows"))
    flat_min_periods = int(nested(config, "audit.gr_surface.flat_min_periods"))
    flat_std = (
        pd.Series(horizontal_filled)
        .rolling(flat_window, center=True, min_periods=flat_min_periods)
        .std()
        .fillna(0.0)
        .to_numpy(float)[row_index]
    )
    flat_threshold = float(nested(config, "audit.gr_surface.flat_std_threshold_gr"))
    max_fraction = float(
        nested(config, "audit.barrier.max_forbidden_fraction_per_row")
    )
    supported = original_finite & (flat_std >= flat_threshold) & (raw_high_fraction <= max_fraction)

    actual_step = float(np.median(np.diff(grid_tvt)))
    min_md = int(nested(config, "audit.barrier.min_md_persistence_rows"))
    min_tvt_ft = float(nested(config, "audit.barrier.min_tvt_thickness_ft"))
    min_tvt_bins = max(1, int(math.ceil(min_tvt_ft / max(actual_step, 1e-6))))
    structure = np.ones((min_md, min_tvt_bins), dtype=bool)
    barrier = ndimage.binary_opening(high, structure=structure).astype(bool)
    barrier[~supported, :] = False
    supported &= barrier.mean(axis=1) <= max_fraction
    barrier[~supported, :] = False
    return BarrierSurface(
        grid_tvt=grid_tvt,
        barrier=barrier,
        supported_rows=supported,
        raw_high_fraction=raw_high_fraction,
        actual_grid_step_ft=actual_step,
        min_tvt_thickness_bins=min_tvt_bins,
    )


# %% [markdown]
# ## 5. Anchor corridor and edge-crossing diagnostics

# %%
@dataclass(frozen=True)
class CorridorReadout:
    reachable: np.ndarray
    component_known_rows: np.ndarray
    segment_count_by_row: np.ndarray
    anchored_segment_count_by_row: np.ndarray
    anchor_found: bool
    anchor_row_position: int | None


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int8), (1, 1))
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


def distance_to_interval(index: int, segment: tuple[int, int]) -> int:
    if segment[0] <= index <= segment[1]:
        return 0
    return min(abs(index - segment[0]), abs(index - segment[1]))


def build_anchor_corridor(
    surface: BarrierSurface,
    row_index: np.ndarray,
    last_known_tvt: float,
    config: dict[str, Any],
) -> CorridorReadout:
    n_rows, n_states = surface.barrier.shape
    reachable = np.zeros((n_rows, n_states), dtype=bool)
    known_rows = np.zeros(n_rows, dtype=bool)
    segment_count = np.zeros(n_rows, dtype=np.int16)
    anchored_count = np.zeros(n_rows, dtype=np.int16)
    anchor_index = int(np.argmin(np.abs(surface.grid_tvt - float(last_known_tvt))))
    anchor_search_rows = int(nested(config, "audit.barrier.anchor_search_rows"))
    anchor_radius_ft = float(nested(config, "audit.barrier.anchor_radius_ft"))
    anchor_radius_bins = int(
        math.ceil(anchor_radius_ft / max(surface.actual_grid_step_ft, 1e-6))
    )
    max_shift_ft = float(
        nested(config, "audit.barrier.max_corridor_shift_ft_per_row")
    )
    max_gap = int(nested(config, "audit.barrier.max_unsupported_bridge_rows"))

    previous_row: int | None = None
    previous_segments: list[tuple[int, int]] = []
    previous_anchored: list[bool] = []
    anchor_found = False
    anchor_row_position: int | None = None
    chain_active = True

    for position in np.flatnonzero(surface.supported_rows):
        position = int(position)
        segments = contiguous_true_segments(~surface.barrier[position])
        segment_count[position] = len(segments)
        current_anchored = [False] * len(segments)
        if not anchor_found:
            if position <= anchor_search_rows and segments:
                distances = [distance_to_interval(anchor_index, segment) for segment in segments]
                selected = int(np.argmin(distances))
                if distances[selected] <= anchor_radius_bins:
                    current_anchored[selected] = True
                    anchor_found = True
                    anchor_row_position = position
            elif position > anchor_search_rows:
                chain_active = False
        elif chain_active and previous_row is not None:
            row_gap = int(row_index[position] - row_index[previous_row])
            if row_gap > max_gap:
                chain_active = False
            else:
                shift_bins = int(
                    math.ceil(
                        max_shift_ft * max(row_gap, 1)
                        / max(surface.actual_grid_step_ft, 1e-6)
                    )
                )
                for current_index, current_segment in enumerate(segments):
                    current_anchored[current_index] = any(
                        was_anchored
                        and interval_distance(previous_segment, current_segment) <= shift_bins
                        for previous_segment, was_anchored in zip(
                            previous_segments,
                            previous_anchored,
                            strict=True,
                        )
                    )
        if anchor_found and chain_active:
            known_rows[position] = True
            for segment, is_anchored in zip(segments, current_anchored, strict=True):
                if is_anchored:
                    reachable[position, segment[0] : segment[1] + 1] = True
            anchored_count[position] = int(sum(current_anchored))
        previous_row = position
        previous_segments = segments
        previous_anchored = current_anchored

    return CorridorReadout(
        reachable=reachable,
        component_known_rows=known_rows,
        segment_count_by_row=segment_count,
        anchored_segment_count_by_row=anchored_count,
        anchor_found=anchor_found,
        anchor_row_position=anchor_row_position,
    )


@dataclass(frozen=True)
class PathDiagnostics:
    in_grid: np.ndarray
    endpoint_forbidden: np.ndarray
    component_known: np.ndarray
    anchor_member: np.ndarray
    edge_crossing: np.ndarray
    instantaneous_violation: np.ndarray
    valid_after_history: np.ndarray


def nearest_state_indices(grid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    positions = np.searchsorted(grid, values, side="left")
    left = np.clip(positions - 1, 0, len(grid) - 1)
    right = np.clip(positions, 0, len(grid) - 1)
    choose_right = np.abs(grid[right] - values) < np.abs(grid[left] - values)
    indices = np.where(choose_right, right, left).astype(np.int32)
    in_grid = np.isfinite(values) & (values >= float(grid[0])) & (values <= float(grid[-1]))
    return indices, in_grid


def diagnose_path(
    values: np.ndarray,
    surface: BarrierSurface,
    corridor: CorridorReadout,
    row_index: np.ndarray,
) -> PathDiagnostics:
    indices, in_grid = nearest_state_indices(surface.grid_tvt, values)
    positions = np.arange(len(values), dtype=np.int32)
    endpoint_forbidden = (
        in_grid
        & surface.supported_rows
        & surface.barrier[positions, indices]
    )
    component_known = in_grid & corridor.component_known_rows
    anchor_member = np.ones(len(values), dtype=bool)
    anchor_member[component_known] = corridor.reachable[
        positions[component_known], indices[component_known]
    ]
    edge_crossing = np.zeros(len(values), dtype=bool)
    for position in range(1, len(values)):
        if row_index[position] != row_index[position - 1] + 1:
            continue
        if not (in_grid[position - 1] and in_grid[position]):
            continue
        if not (
            surface.supported_rows[position - 1]
            and surface.supported_rows[position]
        ):
            continue
        low = int(min(indices[position - 1], indices[position]))
        high = int(max(indices[position - 1], indices[position])) + 1
        edge_crossing[position] = bool(
            surface.barrier[position - 1 : position + 1, low:high].any()
        )
    instantaneous = endpoint_forbidden | (component_known & ~anchor_member) | edge_crossing
    invalid_history = np.maximum.accumulate(instantaneous.astype(np.int8)).astype(bool)
    return PathDiagnostics(
        in_grid=in_grid,
        endpoint_forbidden=endpoint_forbidden,
        component_known=component_known,
        anchor_member=anchor_member,
        edge_crossing=edge_crossing,
        instantaneous_violation=instantaneous,
        valid_after_history=~invalid_history,
    )


# %% [markdown]
# ## 6. Metric aggregation and generated-file helpers

# %%
def safe_rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else np.nan


def distance_bucket(values: np.ndarray, config: dict[str, Any]) -> pd.Categorical:
    edges = [float(value) for value in nested(config, "audit.distance_buckets.edges")]
    labels = [str(value) for value in nested(config, "audit.distance_buckets.labels")]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True)


def build_row_audit(
    well: str,
    cache: pd.DataFrame,
    horizontal: pd.DataFrame,
    row_index: np.ndarray,
    surface: BarrierSurface,
    corridor: CorridorReadout,
    candidates: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, PathDiagnostics]]:
    target_column = str(nested(config, "data.target_column", "TVT"))
    true_tvt = horizontal[target_column].to_numpy(float)[row_index]
    truth = diagnose_path(true_tvt, surface, corridor, row_index)
    audit = pd.DataFrame(
        {
            "id": cache["id"].astype(str).to_numpy(),
            "well": well,
            "row_index": row_index,
            "md_since": cache["md_since"].to_numpy(float),
            "barrier_supported": surface.supported_rows.astype(np.int8),
            "barrier_fraction": surface.barrier.mean(axis=1).astype(np.float32),
            "truth_in_grid": truth.in_grid.astype(np.int8),
            "truth_endpoint_forbidden": truth.endpoint_forbidden.astype(np.int8),
            "truth_component_known": truth.component_known.astype(np.int8),
            "truth_anchor_member": truth.anchor_member.astype(np.int8),
            "truth_edge_crossing": truth.edge_crossing.astype(np.int8),
            "truth_instantaneous_violation": truth.instantaneous_violation.astype(np.int8),
            "truth_valid_after_history": truth.valid_after_history.astype(np.int8),
        }
    )
    audit["distance_bucket"] = distance_bucket(audit["md_since"].to_numpy(float), config)
    diagnostics: dict[str, PathDiagnostics] = {}
    errors: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for name, values in candidates.items():
        diagnostic = diagnose_path(values, surface, corridor, row_index)
        diagnostics[name] = diagnostic
        error = np.abs(values - true_tvt)
        errors.append(error)
        valid.append(diagnostic.valid_after_history)
        audit[f"{name}_abs_error"] = error.astype(np.float32)
        audit[f"{name}_endpoint_forbidden"] = diagnostic.endpoint_forbidden.astype(np.int8)
        audit[f"{name}_anchor_member"] = diagnostic.anchor_member.astype(np.int8)
        audit[f"{name}_edge_crossing"] = diagnostic.edge_crossing.astype(np.int8)
        audit[f"{name}_valid_after_history"] = diagnostic.valid_after_history.astype(np.int8)
    error_matrix = np.column_stack(errors)
    valid_matrix = np.column_stack(valid)
    union_before = np.nanmin(error_matrix, axis=1)
    strict_matrix = np.where(valid_matrix, error_matrix, np.nan)
    finite_survivor = np.isfinite(strict_matrix).any(axis=1)
    union_after = np.full(len(audit), np.nan, dtype=float)
    union_after[finite_survivor] = np.nanmin(strict_matrix[finite_survivor], axis=1)
    union_fallback = np.where(finite_survivor, union_after, union_before)
    audit["survivor_count"] = valid_matrix.sum(axis=1).astype(np.int8)
    audit["union_oracle_before_abs_error"] = union_before.astype(np.float32)
    audit["union_oracle_after_strict_abs_error"] = union_after.astype(np.float32)
    audit["union_oracle_after_fallback_abs_error"] = union_fallback.astype(np.float32)
    return audit, diagnostics


def group_contribution(
    frame: pd.DataFrame,
    scope: str,
    label: str,
) -> dict[str, Any]:
    before = frame["union_oracle_before_abs_error"].to_numpy(float)
    after = frame["union_oracle_after_strict_abs_error"].to_numpy(float)
    fallback = frame["union_oracle_after_fallback_abs_error"].to_numpy(float)
    component_known = frame["truth_component_known"].to_numpy(bool)
    anchor_member = frame["truth_anchor_member"].to_numpy(bool)
    return {
        "scope": scope,
        "group": label,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "barrier_supported_rows": int(frame["barrier_supported"].sum()),
        "truth_in_grid_rows": int(frame["truth_in_grid"].sum()),
        "truth_endpoint_forbidden_rows": int(frame["truth_endpoint_forbidden"].sum()),
        "truth_component_known_rows": int(component_known.sum()),
        "truth_anchor_member_known_rows": int((component_known & anchor_member).sum()),
        "truth_edge_crossing_rows": int(frame["truth_edge_crossing"].sum()),
        "truth_instantaneous_violation_rows": int(
            frame["truth_instantaneous_violation"].sum()
        ),
        "truth_valid_after_history_rows": int(frame["truth_valid_after_history"].sum()),
        "no_survivor_rows": int((frame["survivor_count"] == 0).sum()),
        "union_before_sse": float(np.nansum(np.square(before))),
        "union_before_count": int(np.isfinite(before).sum()),
        "union_after_strict_sse": float(np.nansum(np.square(after))),
        "union_after_strict_count": int(np.isfinite(after).sum()),
        "union_after_fallback_sse": float(np.nansum(np.square(fallback))),
        "union_after_fallback_count": int(np.isfinite(fallback).sum()),
    }


def candidate_contribution(
    frame: pd.DataFrame,
    candidate: str,
    scope: str,
    label: str,
    bad_threshold: float,
) -> dict[str, Any]:
    error = frame[f"{candidate}_abs_error"].to_numpy(float)
    valid = frame[f"{candidate}_valid_after_history"].to_numpy(bool)
    pruned = ~valid
    bad = error > bad_threshold
    good = np.isfinite(error) & ~bad
    return {
        "scope": scope,
        "group": label,
        "candidate": candidate,
        "rows": int(len(frame)),
        "error_sse": float(np.nansum(np.square(error))),
        "error_count": int(np.isfinite(error).sum()),
        "pruned_rows": int(pruned.sum()),
        "bad_rows": int(bad.sum()),
        "good_rows": int(good.sum()),
        "pruned_bad_rows": int((pruned & bad).sum()),
        "pruned_good_rows": int((pruned & good).sum()),
        "endpoint_forbidden_rows": int(frame[f"{candidate}_endpoint_forbidden"].sum()),
        "edge_crossing_rows": int(frame[f"{candidate}_edge_crossing"].sum()),
    }


def finalize_group_metrics(contributions: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in contributions.columns if column not in {"scope", "group"}]
    result = contributions.groupby(["scope", "group"], as_index=False)[numeric].sum()
    result["barrier_supported_rate"] = result["barrier_supported_rows"] / result["rows"]
    result["truth_endpoint_forbidden_rate"] = (
        result["truth_endpoint_forbidden_rows"] / result["truth_in_grid_rows"].clip(lower=1)
    )
    result["truth_anchor_component_survival_rate"] = (
        result["truth_anchor_member_known_rows"]
        / result["truth_component_known_rows"].clip(lower=1)
    )
    result["truth_edge_crossing_rate"] = result["truth_edge_crossing_rows"] / result["rows"]
    result["truth_instantaneous_violation_rate"] = (
        result["truth_instantaneous_violation_rows"] / result["rows"]
    )
    result["truth_valid_after_history_rate"] = (
        result["truth_valid_after_history_rows"] / result["rows"]
    )
    result["no_survivor_rate"] = result["no_survivor_rows"] / result["rows"]
    for prefix in ["union_before", "union_after_strict", "union_after_fallback"]:
        result[f"{prefix}_rmse"] = np.sqrt(
            result[f"{prefix}_sse"] / result[f"{prefix}_count"].clip(lower=1)
        )
    result["union_after_fallback_delta_rmse"] = (
        result["union_after_fallback_rmse"] - result["union_before_rmse"]
    )
    return result


def finalize_candidate_metrics(contributions: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope", "group", "candidate"]
    numeric = [column for column in contributions.columns if column not in set(keys)]
    result = contributions.groupby(keys, as_index=False)[numeric].sum()
    result["candidate_rmse"] = np.sqrt(
        result["error_sse"] / result["error_count"].clip(lower=1)
    )
    result["prune_rate"] = result["pruned_rows"] / result["rows"]
    result["bad_candidate_prune_precision"] = (
        result["pruned_bad_rows"] / result["pruned_rows"].clip(lower=1)
    )
    result["bad_candidate_recall"] = (
        result["pruned_bad_rows"] / result["bad_rows"].clip(lower=1)
    )
    result["good_candidate_false_prune_rate"] = (
        result["pruned_good_rows"] / result["good_rows"].clip(lower=1)
    )
    result["endpoint_forbidden_rate"] = result["endpoint_forbidden_rows"] / result["rows"]
    result["edge_crossing_rate"] = result["edge_crossing_rows"] / result["rows"]
    return result


def by_well_row(
    well: str,
    audit: pd.DataFrame,
    surface: BarrierSurface,
    corridor: CorridorReadout,
) -> dict[str, Any]:
    before = safe_rmse(audit["union_oracle_before_abs_error"].to_numpy(float))
    fallback = safe_rmse(audit["union_oracle_after_fallback_abs_error"].to_numpy(float))
    known = audit["truth_component_known"].to_numpy(bool)
    member = audit["truth_anchor_member"].to_numpy(bool)
    return {
        "well": well,
        "rows": int(len(audit)),
        "grid_bins": int(surface.barrier.shape[1]),
        "grid_step_ft": float(surface.actual_grid_step_ft),
        "barrier_supported_rate": float(audit["barrier_supported"].mean()),
        "barrier_cell_rate": float(surface.barrier.mean()),
        "anchor_found": bool(corridor.anchor_found),
        "anchor_row_position": corridor.anchor_row_position,
        "component_known_rate": float(known.mean()),
        "truth_anchor_component_survival_rate": (
            float(member[known].mean()) if known.any() else np.nan
        ),
        "truth_endpoint_forbidden_rate": float(audit["truth_endpoint_forbidden"].mean()),
        "truth_edge_crossing_rate": float(audit["truth_edge_crossing"].mean()),
        "truth_valid_after_history_rate": float(audit["truth_valid_after_history"].mean()),
        "no_survivor_rate": float((audit["survivor_count"] == 0).mean()),
        "union_before_rmse": before,
        "union_after_fallback_rmse": fallback,
        "union_after_fallback_delta_rmse": fallback - before,
    }


# %% [markdown]
# ## 7. Synthetic contract checks

# %%
def run_synthetic_contract_checks(config: dict[str, Any]) -> None:
    grid = np.arange(10, dtype=np.float32) * 4.0
    barrier = np.zeros((6, 10), dtype=bool)
    barrier[:, 4:6] = True
    surface = BarrierSurface(
        grid_tvt=grid,
        barrier=barrier,
        supported_rows=np.ones(6, dtype=bool),
        raw_high_fraction=barrier.mean(axis=1).astype(np.float32),
        actual_grid_step_ft=4.0,
        min_tvt_thickness_bins=2,
    )
    rows = np.arange(6, dtype=np.int32)
    corridor = build_anchor_corridor(surface, rows, last_known_tvt=8.0, config=config)
    if not corridor.anchor_found:
        raise AssertionError("Synthetic anchor must be found")
    if corridor.reachable[:, 7].any():
        raise AssertionError("Separated right corridor must not be anchor-reachable")
    stay_left = diagnose_path(np.full(6, 8.0), surface, corridor, rows)
    if not stay_left.valid_after_history.all():
        raise AssertionError("Path staying in the anchor corridor must remain valid")
    jump_right = diagnose_path(
        np.array([8.0, 8.0, 28.0, 28.0, 28.0, 28.0]),
        surface,
        corridor,
        rows,
    )
    if not jump_right.edge_crossing[2]:
        raise AssertionError("Synthetic jump over the red ridge must cross an edge barrier")
    if jump_right.valid_after_history[2:].any():
        raise AssertionError("Hard diagnostic validity must remain false after first crossing")
    unsupported = BarrierSurface(
        grid_tvt=grid,
        barrier=np.zeros((2, 10), dtype=bool),
        supported_rows=np.zeros(2, dtype=bool),
        raw_high_fraction=np.ones(2, dtype=np.float32),
        actual_grid_step_ft=4.0,
        min_tvt_thickness_bins=2,
    )
    neutral_corridor = build_anchor_corridor(
        unsupported,
        np.arange(2, dtype=np.int32),
        last_known_tvt=8.0,
        config=config,
    )
    neutral_path = diagnose_path(
        np.array([8.0, 28.0]),
        unsupported,
        neutral_corridor,
        np.arange(2, dtype=np.int32),
    )
    if not neutral_path.valid_after_history.all():
        raise AssertionError("Unsupported rows must remain neutral, not hard walls")


# %% [markdown]
# ## 8. Setup and input checks

# %%
require_authoritative_runtime()
START_TIME = time.time()
CONFIG = load_config()
if nested(CONFIG, "experiment.route") != "pf_beam":
    raise ValueError("exp246 route must be pf_beam")
if nested(CONFIG, "model.active_variants") != ["diagnostic_only"]:
    raise ValueError("exp246 must run exactly one diagnostic_only variant")
if any(
    int(nested(CONFIG, key, -1)) != 0
    for key in [
        "model.lightgbm_config_count",
        "model.fold_training_count",
        "model.booster_count",
    ]
):
    raise ValueError("exp246 must train zero configs/folds/boosters")
if bool(nested(CONFIG, "model.parent_control_retraining", True)):
    raise ValueError("exp246 must not retrain its parent/control")
if bool(nested(CONFIG, "runtime.kaggle.enable_gpu", True)):
    raise ValueError("exp246 must run CPU-only")
if bool(nested(CONFIG, "inference.enabled", True)):
    raise ValueError("exp246 hidden-test inference must remain disabled")

TRAIN_DIR = resolve_train_dir(CONFIG)
CACHE_PATH = resolve_file(
    CONFIG,
    "data.candidate_cache_paths",
    "data.candidate_cache_filename",
)
HIDDEN_PATH = resolve_file(
    CONFIG,
    "data.hidden_like_paths",
    "data.hidden_like_filename",
)
OUTPUT_DIR = output_dir()
SPECS = candidate_specs(CONFIG)
RUN_SYNTHETIC = run_synthetic_contract_checks(CONFIG)
HIDDEN_FRAME, HIDDEN_GROUPS = load_hidden_like_roles(HIDDEN_PATH)

print("Experiment:", EXPERIMENT_NAME)
print("Route / mode:", nested(CONFIG, "experiment.route"), nested(CONFIG, "audit.mode"))
print("Active variants / boosters:", nested(CONFIG, "model.active_variants"), 0)
print("Train dir:", TRAIN_DIR)
print("Candidate cache:", CACHE_PATH, CACHE_PATH.stat().st_size)
print("Hidden-like assignment:", HIDDEN_PATH, len(HIDDEN_FRAME))
print("Output dir:", OUTPUT_DIR)
print("Candidate specs:", [spec.__dict__ for spec in SPECS])
print(
    "Barrier config:",
    {
        "raw_threshold": nested(CONFIG, "audit.gr_surface.raw_abs_difference_threshold"),
        "smooth_threshold": nested(CONFIG, "audit.gr_surface.smooth_abs_difference_threshold"),
        "md_persistence": nested(CONFIG, "audit.barrier.min_md_persistence_rows"),
        "tvt_thickness_ft": nested(CONFIG, "audit.barrier.min_tvt_thickness_ft"),
        "grid_cap": nested(CONFIG, "audit.barrier.max_state_bins"),
    },
)

# %% [markdown]
# ## 9. Full train-side audit orchestration

# %%
candidate_cache_raw_sha = sha256_path(CACHE_PATH)
candidate_cache_content_sha = sha256_path(CACHE_PATH, decompressed=True)
hidden_sha = sha256_path(HIDDEN_PATH)
config_sha = sha256_path(find_config_path())

row_filename = str(nested(CONFIG, "audit.outputs.row_audit_filename"))
row_path = OUTPUT_DIR / row_filename
group_contributions: list[dict[str, Any]] = []
candidate_contributions: list[dict[str, Any]] = []
by_well_rows: list[dict[str, Any]] = []
barrier_well_rows: list[dict[str, Any]] = []
raw_file_shas: dict[str, str] = {}
chunksize = int(nested(CONFIG, "audit.cache_chunksize"))
max_wells_value = nested(CONFIG, "audit.max_wells")
max_wells = int(max_wells_value) if max_wells_value is not None else None
bad_threshold = float(nested(CONFIG, "audit.bad_candidate_threshold_ft"))

processed_wells = 0
processed_rows = 0
with DeterministicGzipCsvWriter(row_path) as row_writer:
    for well, cache in iter_candidate_cache_wells(CACHE_PATH, SPECS, chunksize):
        if max_wells is not None and processed_wells >= max_wells:
            break
        horizontal, typewell, row_index, prefix_end = load_raw_well(
            well,
            cache,
            TRAIN_DIR,
            CONFIG,
        )
        horizontal_path = TRAIN_DIR / f"{well}{nested(CONFIG, 'data.horizontal_suffix')}"
        typewell_path = TRAIN_DIR / f"{well}{nested(CONFIG, 'data.typewell_suffix')}"
        raw_file_shas[horizontal_path.name] = sha256_path(horizontal_path)
        raw_file_shas[typewell_path.name] = sha256_path(typewell_path)

        surface = build_barrier_surface(horizontal, typewell, row_index, CONFIG)
        last_known = float(cache["last_known_tvt"].iloc[0])
        if float(cache["last_known_tvt"].max() - cache["last_known_tvt"].min()) > 0.15:
            raise ValueError(f"{well} last_known_tvt is not constant within tolerance")
        corridor = build_anchor_corridor(surface, row_index, last_known, CONFIG)
        candidates = materialize_candidates(cache, SPECS)
        audit, _ = build_row_audit(
            well,
            cache,
            horizontal,
            row_index,
            surface,
            corridor,
            candidates,
            CONFIG,
        )
        row_writer.write(audit)

        scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", audit)]
        for label, subset in audit.groupby("distance_bucket", observed=True):
            scopes.append(("distance", str(label), subset))
        for label, wells in HIDDEN_GROUPS.items():
            if well in wells:
                scopes.append(("hidden_like", label, audit))
        for scope, label, subset in scopes:
            group_contributions.append(group_contribution(subset, scope, label))
            for spec in SPECS:
                candidate_contributions.append(
                    candidate_contribution(
                        subset,
                        spec.name,
                        scope,
                        label,
                        bad_threshold,
                    )
                )

        well_row = by_well_row(well, audit, surface, corridor)
        well_row["prefix_end"] = prefix_end
        by_well_rows.append(well_row)
        barrier_well_rows.append(
            {
                "well": well,
                "rows": len(audit),
                "grid_bins": surface.barrier.shape[1],
                "grid_tvt_min": float(surface.grid_tvt[0]),
                "grid_tvt_max": float(surface.grid_tvt[-1]),
                "grid_step_ft": surface.actual_grid_step_ft,
                "min_tvt_thickness_bins": surface.min_tvt_thickness_bins,
                "supported_rows": int(surface.supported_rows.sum()),
                "barrier_cells": int(surface.barrier.sum()),
                "barrier_cell_rate": float(surface.barrier.mean()),
                "mean_segments_supported": (
                    float(corridor.segment_count_by_row[surface.supported_rows].mean())
                    if surface.supported_rows.any()
                    else np.nan
                ),
                "mean_anchored_segments_known": (
                    float(
                        corridor.anchored_segment_count_by_row[
                            corridor.component_known_rows
                        ].mean()
                    )
                    if corridor.component_known_rows.any()
                    else np.nan
                ),
                "anchor_found": corridor.anchor_found,
                "anchor_row_position": corridor.anchor_row_position,
                "component_known_rows": int(corridor.component_known_rows.sum()),
            }
        )
        processed_wells += 1
        processed_rows += len(audit)
        if processed_wells == 1 or processed_wells % 25 == 0:
            print(
                f"[{processed_wells}] well={well} rows={len(audit)} "
                f"grid={surface.barrier.shape[1]} supported={surface.supported_rows.mean():.4f} "
                f"barrier={surface.barrier.mean():.4f} anchor={corridor.anchor_found}"
            )

if processed_wells == 0 or processed_rows == 0:
    raise RuntimeError("No wells/rows were processed")
if not row_writer.raw.closed:
    raise AssertionError("Row-audit gzip file buffer must be closed before hashing")

group_metrics = finalize_group_metrics(pd.DataFrame(group_contributions))
candidate_metrics = finalize_candidate_metrics(pd.DataFrame(candidate_contributions))
by_well = pd.DataFrame(by_well_rows).sort_values("well").reset_index(drop=True)
barrier_well = pd.DataFrame(barrier_well_rows).sort_values("well").reset_index(drop=True)

group_path = OUTPUT_DIR / str(nested(CONFIG, "audit.outputs.group_metrics_filename"))
candidate_path = OUTPUT_DIR / str(
    nested(CONFIG, "audit.outputs.candidate_metrics_filename")
)
by_well_path = OUTPUT_DIR / str(nested(CONFIG, "audit.outputs.by_well_filename"))
barrier_well_path = OUTPUT_DIR / str(
    nested(CONFIG, "audit.outputs.barrier_well_filename")
)
group_metrics.to_csv(group_path, index=False)
candidate_metrics.to_csv(candidate_path, index=False)
by_well.to_csv(by_well_path, index=False)
barrier_well.to_csv(barrier_well_path, index=False)

# %% [markdown]
# ## 10. Metrics, guards, SHA, and generated files

# %%
overall = group_metrics.loc[
    (group_metrics["scope"] == "overall") & (group_metrics["group"] == "all")
].iloc[0]
overall_candidate = candidate_metrics.loc[
    (candidate_metrics["scope"] == "overall")
    & (candidate_metrics["group"] == "all")
]
weighted_good_false_prune = float(
    overall_candidate["pruned_good_rows"].sum()
    / max(int(overall_candidate["good_rows"].sum()), 1)
)
worst_well_oracle_delta = float(by_well["union_after_fallback_delta_rmse"].max())
guards = {
    "true_path_forbidden_rate": {
        "value": float(overall["truth_instantaneous_violation_rate"]),
        "limit": float(nested(CONFIG, "audit.guards.max_true_path_forbidden_rate")),
        "pass": bool(
            overall["truth_instantaneous_violation_rate"]
            <= float(nested(CONFIG, "audit.guards.max_true_path_forbidden_rate"))
        ),
    },
    "true_anchor_component_survival_rate": {
        "value": float(overall["truth_anchor_component_survival_rate"]),
        "limit": float(
            nested(CONFIG, "audit.guards.min_true_anchor_component_survival_rate")
        ),
        "pass": bool(
            overall["truth_anchor_component_survival_rate"]
            >= float(
                nested(CONFIG, "audit.guards.min_true_anchor_component_survival_rate")
            )
        ),
    },
    "good_candidate_false_prune_rate": {
        "value": weighted_good_false_prune,
        "limit": float(
            nested(CONFIG, "audit.guards.max_good_candidate_false_prune_rate")
        ),
        "pass": bool(
            weighted_good_false_prune
            <= float(nested(CONFIG, "audit.guards.max_good_candidate_false_prune_rate"))
        ),
    },
    "union_oracle_rmse_delta_ft": {
        "value": float(overall["union_after_fallback_delta_rmse"]),
        "limit": float(nested(CONFIG, "audit.guards.max_union_oracle_rmse_delta_ft")),
        "pass": bool(
            overall["union_after_fallback_delta_rmse"]
            <= float(nested(CONFIG, "audit.guards.max_union_oracle_rmse_delta_ft"))
        ),
    },
    "worst_well_union_oracle_delta_ft": {
        "value": worst_well_oracle_delta,
        "limit": float(
            nested(CONFIG, "audit.guards.max_worst_well_union_oracle_delta_ft")
        ),
        "pass": bool(
            worst_well_oracle_delta
            <= float(nested(CONFIG, "audit.guards.max_worst_well_union_oracle_delta_ft"))
        ),
    },
}
guard_pass = bool(all(item["pass"] for item in guards.values()))

output_paths = [row_path, group_path, candidate_path, by_well_path, barrier_well_path]
output_sha = {
    path.name: {
        "raw_sha256": sha256_path(path),
        "decompressed_content_sha256": (
            sha256_path(path, decompressed=True) if path.suffix == ".gz" else None
        ),
    }
    for path in output_paths
}
summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "implemented_kaggle_train_complete" if is_kaggle_runtime() else "debug_complete",
    "decision": (
        "eligible_for_separate_hard_edge_cut_stage"
        if guard_pass
        else "diagnostic_only_guard_failed"
    ),
    "route": nested(CONFIG, "experiment.route"),
    "mode": nested(CONFIG, "audit.mode"),
    "runtime_seconds": time.time() - START_TIME,
    "processed_wells": processed_wells,
    "processed_rows": processed_rows,
    "active_variant_count": 1,
    "lightgbm_config_count": 0,
    "fold_training_count": 0,
    "booster_count": 0,
    "parent_control_retraining": False,
    "guard_pass": guard_pass,
    "guards": guards,
    "overall": overall.to_dict(),
    "overall_candidate_metrics": overall_candidate.to_dict(orient="records"),
    "hidden_like_metrics": group_metrics.loc[
        group_metrics["scope"] == "hidden_like"
    ].to_dict(orient="records"),
    "distance_metrics": group_metrics.loc[
        group_metrics["scope"] == "distance"
    ].to_dict(orient="records"),
    "worst_well_union_oracle_delta_ft": worst_well_oracle_delta,
    "input": {
        "candidate_cache": str(CACHE_PATH),
        "candidate_cache_raw_sha256": candidate_cache_raw_sha,
        "candidate_cache_decompressed_content_sha256": candidate_cache_content_sha,
        "hidden_like_assignment": str(HIDDEN_PATH),
        "hidden_like_assignment_sha256": hidden_sha,
        "raw_train_dir": str(TRAIN_DIR),
        "raw_file_count": len(raw_file_shas),
        "raw_file_inventory_sha256": sha256_mapping(raw_file_shas),
        "config_sha256": config_sha,
    },
    "outputs": output_sha,
    "reproducibility": nested(CONFIG, "reproducibility"),
    "prohibitions": [
        "No barrier or corridor parameter uses evaluation-tail truth/error/oracle.",
        "No candidate value or prediction was changed.",
        "No model was trained and no HMM/PF/Beam process was rerun.",
        "No hidden-test inference or submission was created.",
    ],
}
summary_path = OUTPUT_DIR / str(nested(CONFIG, "audit.outputs.summary_filename"))
write_json(summary_path, summary)
metrics_payload = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "metric": "negative_space_barrier_safety_audit",
    "primary_metrics": {
        "guard_pass": guard_pass,
        "true_path_forbidden_rate": guards["true_path_forbidden_rate"]["value"],
        "true_path_endpoint_forbidden_rate": float(
            overall["truth_endpoint_forbidden_rate"]
        ),
        "true_path_edge_crossing_rate": float(overall["truth_edge_crossing_rate"]),
        "true_anchor_component_survival_rate": guards[
            "true_anchor_component_survival_rate"
        ]["value"],
        "good_candidate_false_prune_rate": weighted_good_false_prune,
        "union_oracle_before_rmse": float(overall["union_before_rmse"]),
        "union_oracle_after_fallback_rmse": float(
            overall["union_after_fallback_rmse"]
        ),
        "worst_well_union_oracle_delta_ft": worst_well_oracle_delta,
    },
    "runtime": {
        "cpu_only": True,
        "active_variant_count": 1,
        "lightgbm_config_count": 0,
        "fold_training_count": 0,
        "booster_count": 0,
    },
    "summary_path": str(summary_path),
    "notes": "Train-side diagnostic only; no inference or submission.",
}
write_json(metrics_output_path(), metrics_payload)

print("\nPrimary group metrics")
display(group_metrics)
print("\nPrimary candidate metrics")
display(overall_candidate)
print("\nWorst wells by oracle delta")
display(by_well.nlargest(20, "union_after_fallback_delta_rmse"))
print("\nGuards")
display(pd.DataFrame([{"guard": key, **value} for key, value in guards.items()]))
print("Summary:", summary_path)
print("Metrics:", metrics_output_path())
print("Guard pass:", guard_pass)
