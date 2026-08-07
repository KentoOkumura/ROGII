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
# # exp250 segment-local negative-space GR corridor audit — train
#
# This deterministic, no-training PF/Beam diagnostic keeps the saved exp072
# candidates fixed. It divides each official evaluation tail into MD-local
# overlap segments, constructs real and shuffled-GR mismatch costs, and audits
# directed minimum-bottleneck corridors. Target TVT enters only after every
# segment, path, threshold, and corridor is fixed.

# %% [markdown]
# ## Contents
# 1. Imports and deterministic generated-file helpers
# 2. Configuration, runtime, and input resolution
# 3. Raw inventory and candidate-cache contract
# 4. Well-wide GR normalization and MD segment construction
# 5. Directed minimum-bottleneck DP and corridor reachability
# 6. Candidate and truth readout helpers
# 7. Stage 0 deterministic selection and plots
# 8. Stage 1 metrics, overlap, groups, and guards
# 9. Synthetic DAG / DP contract
# 10. Setup, input preflight, and execution
# 11. Metrics, SHA, and generated files

# %% [markdown]
# ## 1. Imports and deterministic generated-file helpers

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

EXPERIMENT_NAME = "exp250_segment_local_negative_space_gr_corridor_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EPS = 1.0e-12


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
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def stable_hex(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
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
# ## 2. Configuration, runtime, and input resolution

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
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"Could not resolve config.yaml for {EXPERIMENT_NAME}")


CONFIG_PATH = find_config_path()


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
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
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if is_kaggle_runtime()
        else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if is_kaggle_runtime():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return ROOT / "experiments" / EXPERIMENT_NAME / "metrics.json"


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(nested(config, "data.train_dir", "data/raw/train")))
    local = configured if configured.is_absolute() else ROOT / configured
    suffix = str(nested(config, "data.horizontal_suffix"))
    if local.exists() and any(local.glob(f"*{suffix}")):
        return local
    if KAGGLE_INPUT_ROOT.exists():
        for match in sorted(KAGGLE_INPUT_ROOT.rglob(f"*{suffix}")):
            if match.parent.name == "train":
                return match.parent
    raise FileNotFoundError("Could not resolve raw train directory")


def resolve_file(
    config: dict[str, Any],
    *,
    paths_key: str,
    filename_key: str,
) -> Path:
    filename = str(nested(config, filename_key))
    candidates: list[Path] = []
    for value in nested(config, paths_key, []) or []:
        path = Path(str(value))
        candidates.append(path if path.is_absolute() else ROOT / path)
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.rglob(filename)))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve {filename}; checked={candidates}")


# %% [markdown]
# ## 3. Raw inventory and candidate-cache contract

# %%
@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_column: str
    transform: str


@dataclass(frozen=True)
class RawInventoryRecord:
    well: str
    horizontal_path: Path
    typewell_path: Path
    row_count: int
    prefix_end: int
    evaluation_start: int
    prefix_md: float
    prefix_tvt: float
    evaluation_md: np.ndarray
    tail_span_ft: float
    tail_gr_missing_fraction: float
    tail_gr_iqr: float
    stable_order: str


@dataclass(frozen=True)
class CachePreflight:
    row_count: int
    well_count: int
    selected_frames: dict[str, pd.DataFrame]


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
    if len(specs) != 5 or len({spec.name for spec in specs}) != 5:
        raise ValueError("The fixed candidate contract requires five unique families")
    return specs


def required_cache_columns(specs: list[CandidateSpec]) -> list[str]:
    columns = {"id", "well", "last_known_tvt", "md_since"}
    columns.update(spec.source_column for spec in specs)
    return sorted(columns)


def prepare_cache_chunk(
    frame: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
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
                raise ValueError("Candidate-cache well blocks are not unique and sorted")
            yield well, group.reset_index(drop=True)
            last_completed = well
        pending = chunk.loc[chunk["well"] == last_well].reset_index(drop=True)
    if pending is not None and len(pending):
        well = str(pending["well"].iloc[0])
        if last_completed is not None and well <= last_completed:
            raise ValueError("Final candidate-cache well block is out of order")
        yield well, pending.reset_index(drop=True)


def finite_iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.nan
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    return float(q75 - q25)


def build_raw_inventory(
    train_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, RawInventoryRecord], str]:
    suffix = str(nested(config, "data.horizontal_suffix"))
    typewell_suffix = str(nested(config, "data.typewell_suffix"))
    md_col = str(nested(config, "data.md_column"))
    gr_col = str(nested(config, "data.gr_column"))
    input_col = str(nested(config, "data.input_target_column"))
    records: dict[str, RawInventoryRecord] = {}
    inventory_payload: dict[str, str] = {}
    for horizontal_path in sorted(train_dir.glob(f"*{suffix}")):
        well = horizontal_path.name[: -len(suffix)]
        typewell_path = train_dir / f"{well}{typewell_suffix}"
        if not typewell_path.exists():
            raise FileNotFoundError(f"Missing typewell file for {well}")
        frame = pd.read_csv(
            horizontal_path,
            usecols=[md_col, gr_col, input_col],
        )
        for column in [md_col, gr_col, input_col]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        md = frame[md_col].to_numpy(float)
        if not np.all(np.isfinite(md)) or np.any(np.diff(md) < 0):
            raise ValueError(f"{well} MD must be finite and nondecreasing")
        known = np.flatnonzero(np.isfinite(frame[input_col].to_numpy(float)))
        if not len(known):
            raise ValueError(f"{well} has no known TVT_input prefix")
        prefix_end = int(known[-1])
        if not np.array_equal(known, np.arange(prefix_end + 1)):
            raise ValueError(f"{well} TVT_input must be a contiguous prefix")
        evaluation_start = prefix_end + 1
        if evaluation_start >= len(frame):
            raise ValueError(f"{well} has no official evaluation tail")
        tail_md = md[evaluation_start:].copy()
        tail_gr = frame[gr_col].to_numpy(float)[evaluation_start:]
        finite_gr = tail_gr[np.isfinite(tail_gr)]
        missing_fraction = float(1.0 - len(finite_gr) / len(tail_gr))
        tail_iqr = finite_iqr(tail_gr)
        records[well] = RawInventoryRecord(
            well=well,
            horizontal_path=horizontal_path,
            typewell_path=typewell_path,
            row_count=len(frame),
            prefix_end=prefix_end,
            evaluation_start=evaluation_start,
            prefix_md=float(md[prefix_end]),
            prefix_tvt=float(frame[input_col].iloc[prefix_end]),
            evaluation_md=tail_md,
            tail_span_ft=float(tail_md[-1] - tail_md[0]),
            tail_gr_missing_fraction=missing_fraction,
            tail_gr_iqr=tail_iqr,
            stable_order=stable_hex(EXPERIMENT_NAME, well, "stage0"),
        )
        inventory_payload[horizontal_path.name] = str(horizontal_path.stat().st_size)
        inventory_payload[typewell_path.name] = str(typewell_path.stat().st_size)
    if not records:
        raise ValueError("Raw inventory is empty")
    return records, sha256_mapping(inventory_payload)


def validate_cache_group(
    well: str,
    frame: pd.DataFrame,
    record: RawInventoryRecord,
    specs: list[CandidateSpec],
) -> None:
    if frame["id"].duplicated().any() or frame["row_index"].duplicated().any():
        raise ValueError(f"{well} candidate cache contains duplicate keys")
    expected_rows = np.arange(
        record.evaluation_start,
        record.row_count,
        dtype=np.int32,
    )
    actual_rows = frame["row_index"].to_numpy(np.int32)
    if not np.array_equal(actual_rows, expected_rows):
        raise ValueError(f"{well} candidate-cache row coverage/order mismatch")
    expected_ids = np.array([f"{well}_{row}" for row in expected_rows], dtype=object)
    if not np.array_equal(frame["id"].to_numpy(object), expected_ids):
        raise ValueError(f"{well} candidate-cache id coverage mismatch")
    numeric_columns = ["last_known_tvt", "md_since"]
    numeric_columns.extend(spec.source_column for spec in specs)
    values = frame[numeric_columns].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{well} candidate cache contains non-finite required values")
    last_known = frame["last_known_tvt"].to_numpy(float)
    if (
        np.max(np.abs(last_known - record.prefix_tvt)) > 0.15
        or np.max(last_known) - np.min(last_known) > 1.0e-6
    ):
        raise ValueError(f"{well} last_known_tvt does not match the raw prefix")
    md_since = frame["md_since"].to_numpy(float)
    if np.any(md_since < -EPS) or np.any(np.diff(md_since) < -EPS):
        raise ValueError(f"{well} md_since must be finite and nondecreasing")


def preflight_candidate_cache(
    cache_path: Path,
    specs: list[CandidateSpec],
    chunksize: int,
    inventory: dict[str, RawInventoryRecord],
    selected_wells: set[str],
) -> CachePreflight:
    selected_frames: dict[str, pd.DataFrame] = {}
    seen: set[str] = set()
    row_count = 0
    for well, frame in iter_candidate_cache_wells(cache_path, specs, chunksize):
        if well not in inventory:
            raise ValueError(f"Candidate cache has unknown well {well}")
        validate_cache_group(well, frame, inventory[well], specs)
        seen.add(well)
        row_count += len(frame)
        if well in selected_wells:
            selected_frames[well] = frame.copy()
    missing = sorted(set(inventory).difference(seen))
    extra = sorted(seen.difference(inventory))
    if missing or extra:
        raise ValueError(f"Candidate-cache well coverage mismatch: missing={missing} extra={extra}")
    if set(selected_frames) != selected_wells:
        raise ValueError("Stage 0 selected wells are missing from candidate cache")
    return CachePreflight(
        row_count=row_count,
        well_count=len(seen),
        selected_frames=selected_frames,
    )


def load_hidden_like_roles(path: Path) -> dict[str, set[str]]:
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame["well_id"].duplicated().any():
        raise ValueError(f"Invalid hidden-like assignment; missing={missing}")
    return {
        "verification_like_spatial": set(
            frame.loc[
                frame["verification_like_spatial_role"] == "valid",
                "well_id",
            ]
        ),
        "verification_like_typewell_purged": set(
            frame.loc[
                frame["verification_like_typewell_purged_role"] == "valid",
                "well_id",
            ]
        ),
    }


def materialize_candidates(
    cache: pd.DataFrame,
    specs: list[CandidateSpec],
) -> dict[str, np.ndarray]:
    base = cache["last_known_tvt"].to_numpy(float)
    candidates: dict[str, np.ndarray] = {}
    for spec in specs:
        value = cache[spec.source_column].to_numpy(float)
        if spec.transform == "absolute":
            candidates[spec.name] = value.copy()
        elif spec.transform == "base_plus_delta":
            candidates[spec.name] = base + value
        else:
            raise ValueError(f"Unsupported candidate transform: {spec.transform}")
    return candidates


# %% [markdown]
# ## 4. Well-wide GR normalization and MD segment construction

# %%
@dataclass(frozen=True)
class NormalizationStats:
    center: float
    scale: float
    fallback: str


@dataclass(frozen=True)
class RawWell:
    well: str
    md: np.ndarray
    z: np.ndarray
    truth_tvt: np.ndarray
    tvt_input: np.ndarray
    horizontal_z: np.ndarray
    typewell_tvt: np.ndarray
    typewell_z_real: np.ndarray
    typewell_z_shuffled: np.ndarray
    prefix_end: int
    evaluation_start: int
    last_known_tvt: float
    last_known_z: float
    cache: pd.DataFrame
    candidates: dict[str, np.ndarray]
    horizontal_norm: NormalizationStats
    typewell_norm: NormalizationStats
    shuffle_shift: int


@dataclass(frozen=True)
class SegmentGrid:
    segment_id: int
    is_first: bool
    start_md: float
    end_md: float
    column_left_md: np.ndarray
    column_right_md: np.ndarray
    column_center_md: np.ndarray
    representative_rows: np.ndarray
    horizontal_values: np.ndarray
    horizontal_supported: np.ndarray
    state_grid_tvt: np.ndarray
    typewell_supported: np.ndarray
    prior_tvt: float
    short_segment_unsupported: bool


def robust_zscore_well(
    values: np.ndarray,
    clip: float,
) -> tuple[np.ndarray, NormalizationStats]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (
            np.full(values.shape, np.nan, dtype=float),
            NormalizationStats(math.nan, 1.0, "no_finite_values"),
        )
    center = float(np.median(finite))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    scale = float(q75 - q25)
    fallback = "iqr"
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(finite))
        fallback = "std"
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
        fallback = "one"
    out = (values - center) / scale
    out = np.where(np.isfinite(out), np.clip(out, -clip, clip), np.nan)
    return out, NormalizationStats(center, scale, fallback)


def stable_circular_shift(
    length: int,
    well: str,
    config: dict[str, Any],
) -> int:
    if length < 2:
        return 0
    low_fraction = float(nested(config, "audit.control.shift_fraction_min"))
    high_fraction = float(nested(config, "audit.control.shift_fraction_max"))
    seed = int(nested(config, "audit.control.seed"))
    low = max(1, int(math.ceil(length * low_fraction)))
    high = min(length - 1, int(math.floor(length * high_fraction)))
    if high < low:
        low = high = max(1, min(length - 1, length // 2))
    digest = stable_hex(EXPERIMENT_NAME, well, f"seed={seed}")
    return low + int(digest, 16) % (high - low + 1)


def load_raw_well(
    record: RawInventoryRecord,
    cache: pd.DataFrame,
    specs: list[CandidateSpec],
    config: dict[str, Any],
) -> RawWell:
    md_col = str(nested(config, "data.md_column"))
    z_col = str(nested(config, "data.z_column"))
    gr_col = str(nested(config, "data.gr_column"))
    target_col = str(nested(config, "data.target_column"))
    input_col = str(nested(config, "data.input_target_column"))
    horizontal = pd.read_csv(
        record.horizontal_path,
        usecols=[md_col, z_col, gr_col, target_col, input_col],
    )
    typewell = pd.read_csv(
        record.typewell_path,
        usecols=[target_col, gr_col],
    )
    for column in [md_col, z_col, gr_col, target_col, input_col]:
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    for column in [target_col, gr_col]:
        typewell[column] = pd.to_numeric(typewell[column], errors="coerce")
    validate_cache_group(record.well, cache, record, specs)
    clip = float(nested(config, "audit.normalization.robust_clip"))
    horizontal_z, horizontal_norm = robust_zscore_well(
        horizontal[gr_col].to_numpy(float),
        clip,
    )
    typewell = (
        typewell.loc[
            np.isfinite(typewell[target_col]) & np.isfinite(typewell[gr_col]),
            [target_col, gr_col],
        ]
        .groupby(target_col, as_index=False, sort=True)[gr_col]
        .mean()
        .sort_values(target_col, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(typewell) < 2 or np.any(np.diff(typewell[target_col].to_numpy(float)) <= 0):
        raise ValueError(f"{record.well} has insufficient sorted typewell support")
    typewell_z, typewell_norm = robust_zscore_well(
        typewell[gr_col].to_numpy(float),
        clip,
    )
    shift = stable_circular_shift(len(typewell_z), record.well, config)
    shuffled = np.roll(typewell_z, shift)
    return RawWell(
        well=record.well,
        md=horizontal[md_col].to_numpy(float),
        z=horizontal[z_col].to_numpy(float),
        truth_tvt=horizontal[target_col].to_numpy(float),
        tvt_input=horizontal[input_col].to_numpy(float),
        horizontal_z=horizontal_z,
        typewell_tvt=typewell[target_col].to_numpy(float),
        typewell_z_real=typewell_z,
        typewell_z_shuffled=shuffled,
        prefix_end=record.prefix_end,
        evaluation_start=record.evaluation_start,
        last_known_tvt=float(horizontal[input_col].iloc[record.prefix_end]),
        last_known_z=float(horizontal[z_col].iloc[record.prefix_end]),
        cache=cache.reset_index(drop=True),
        candidates=materialize_candidates(cache, specs),
        horizontal_norm=horizontal_norm,
        typewell_norm=typewell_norm,
        shuffle_shift=shift,
    )


def segment_start_values(
    evaluation_start_md: float,
    tail_end_md: float,
    config: dict[str, Any],
) -> list[float]:
    length_ft = float(nested(config, "audit.segment.length_ft"))
    stride_ft = float(nested(config, "audit.segment.stride_ft"))
    span = float(tail_end_md - evaluation_start_md)
    if span < length_ft - EPS:
        return [float(evaluation_start_md)]
    starts: list[float] = []
    start = float(evaluation_start_md)
    while start + length_ft <= tail_end_md + EPS:
        starts.append(start)
        start += stride_ft
    if bool(nested(config, "audit.segment.right_align_final_segment")):
        starts.append(float(tail_end_md - length_ft))
    deduplicated: list[float] = []
    for value in sorted(starts):
        if not deduplicated or abs(value - deduplicated[-1]) > 1.0e-8:
            deduplicated.append(value)
    return deduplicated


def interpolate_z_at_md(md: np.ndarray, z: np.ndarray, query: float) -> float:
    mask = np.isfinite(md) & np.isfinite(z)
    if mask.sum() < 2:
        raise ValueError("Horizontal Z interpolation requires at least two points")
    x = md[mask]
    y = z[mask]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        sums = np.bincount(inverse, weights=y)
        counts = np.bincount(inverse)
        y = sums / counts
        x = unique_x
    if query < x[0] - EPS or query > x[-1] + EPS:
        raise ValueError("Segment midpoint lies outside horizontal geometry")
    return float(np.interp(query, x, y))


def build_segment_grid(
    raw: RawWell,
    segment_id: int,
    start_md: float,
    config: dict[str, Any],
) -> SegmentGrid:
    length_ft = float(nested(config, "audit.segment.length_ft"))
    bin_ft = float(nested(config, "audit.segment.horizontal_bin_ft"))
    min_points = int(nested(config, "audit.segment.min_gr_points_per_bin"))
    min_columns = int(nested(config, "audit.segment.min_topology_columns"))
    tail_end_md = float(raw.md[-1])
    end_md = min(start_md + length_ft, tail_end_md)
    span = max(0.0, end_md - start_md)
    n_columns = max(1, int(math.ceil(span / bin_ft)))
    left = start_md + np.arange(n_columns, dtype=float) * bin_ft
    right = np.minimum(left + bin_ft, end_md)
    centers = (left + right) / 2.0
    horizontal_values = np.full(n_columns, np.nan, dtype=float)
    representative_rows = np.full(n_columns, -1, dtype=np.int32)
    evaluation_rows = np.arange(raw.evaluation_start, len(raw.md), dtype=np.int32)
    eval_md = raw.md[evaluation_rows]
    in_segment = (eval_md >= start_md - EPS) & (eval_md <= end_md + EPS)
    rows = evaluation_rows[in_segment]
    row_md = raw.md[rows]
    raw_bins = np.floor((row_md - start_md) / bin_ft).astype(int)
    raw_bins = np.where(raw_bins == n_columns, n_columns - 1, raw_bins)
    valid_bins = (raw_bins >= 0) & (raw_bins < n_columns)
    rows = rows[valid_bins]
    raw_bins = raw_bins[valid_bins]
    for column in range(n_columns):
        column_rows = rows[raw_bins == column]
        if not len(column_rows):
            continue
        distances = np.abs(raw.md[column_rows] - centers[column])
        representative_rows[column] = int(column_rows[np.argmin(distances)])
        finite_values = raw.horizontal_z[column_rows]
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) >= min_points:
            horizontal_values[column] = float(np.median(finite_values))
    horizontal_supported = np.isfinite(horizontal_values)
    midpoint_md = (start_md + end_md) / 2.0
    midpoint_z = interpolate_z_at_md(raw.md, raw.z, midpoint_md)
    prior_tvt = raw.last_known_tvt - (midpoint_z - raw.last_known_z)
    half_width = float(nested(config, "audit.typewell_grid.half_width_ft"))
    state_step = float(nested(config, "audit.typewell_grid.state_step_ft"))
    state_count = int(nested(config, "audit.typewell_grid.state_count"))
    expected_count = int(round(2.0 * half_width / state_step)) + 1
    if state_count != expected_count:
        raise ValueError("Typewell grid state_count does not match half-width/step")
    state_offsets = (
        np.arange(state_count, dtype=float) - (state_count - 1) / 2.0
    ) * state_step
    state_grid = prior_tvt + state_offsets
    typewell_supported = (
        (state_grid >= raw.typewell_tvt[0] - EPS)
        & (state_grid <= raw.typewell_tvt[-1] + EPS)
    )
    return SegmentGrid(
        segment_id=segment_id,
        is_first=segment_id == 0,
        start_md=float(start_md),
        end_md=float(end_md),
        column_left_md=left,
        column_right_md=right,
        column_center_md=centers,
        representative_rows=representative_rows,
        horizontal_values=horizontal_values,
        horizontal_supported=horizontal_supported,
        state_grid_tvt=state_grid,
        typewell_supported=typewell_supported,
        prior_tvt=float(prior_tvt),
        short_segment_unsupported=n_columns < min_columns,
    )


def build_all_segments(raw: RawWell, config: dict[str, Any]) -> list[SegmentGrid]:
    starts = segment_start_values(
        float(raw.md[raw.evaluation_start]),
        float(raw.md[-1]),
        config,
    )
    return [
        build_segment_grid(raw, segment_id, start_md, config)
        for segment_id, start_md in enumerate(starts)
    ]


# %% [markdown]
# ## 5. Directed minimum-bottleneck DP and corridor reachability

# %%
@dataclass(frozen=True)
class DpCore:
    exists: bool
    tau_star: float
    cumulative_cost: float
    path_x: np.ndarray
    path_y: np.ndarray
    gap_edge_used: bool


@dataclass(frozen=True)
class GraphAnalysis:
    exists: bool
    tau_star: float
    cumulative_cost: float
    path_x: np.ndarray
    path_y: np.ndarray
    path_tvt: np.ndarray
    gap_edge_used: bool
    corridor: np.ndarray
    second_tau_star: float
    second_tau_gap: float
    source_states: np.ndarray


@dataclass(frozen=True)
class SegmentAnalysis:
    variant: str
    cost: np.ndarray
    support: np.ndarray
    spanning: GraphAnalysis
    anchored: GraphAnalysis | None
    primary: GraphAnalysis
    primary_graph: str


def predecessor_nodes(
    x_index: int,
    y_index: int,
    horizontal_supported: np.ndarray,
    n_states: int,
) -> list[tuple[int, int, bool]]:
    predecessors: list[tuple[int, int, bool]] = []
    if (
        x_index >= 1
        and horizontal_supported[x_index]
        and horizontal_supported[x_index - 1]
    ):
        for delta in (-1, 0, 1):
            previous_y = y_index - delta
            if 0 <= previous_y < n_states:
                predecessors.append((x_index - 1, previous_y, False))
    if (
        x_index >= 2
        and horizontal_supported[x_index]
        and not horizontal_supported[x_index - 1]
        and horizontal_supported[x_index - 2]
    ):
        for delta in (-2, -1, 0, 1, 2):
            previous_y = y_index - delta
            if 0 <= previous_y < n_states:
                predecessors.append((x_index - 2, previous_y, True))
    return predecessors


def successor_nodes(
    x_index: int,
    y_index: int,
    horizontal_supported: np.ndarray,
    n_states: int,
) -> list[tuple[int, int, bool]]:
    successors: list[tuple[int, int, bool]] = []
    n_columns = len(horizontal_supported)
    if (
        x_index + 1 < n_columns
        and horizontal_supported[x_index]
        and horizontal_supported[x_index + 1]
    ):
        for delta in (-1, 0, 1):
            next_y = y_index + delta
            if 0 <= next_y < n_states:
                successors.append((x_index + 1, next_y, False))
    if (
        x_index + 2 < n_columns
        and horizontal_supported[x_index]
        and not horizontal_supported[x_index + 1]
        and horizontal_supported[x_index + 2]
    ):
        for delta in (-2, -1, 0, 1, 2):
            next_y = y_index + delta
            if 0 <= next_y < n_states:
                successors.append((x_index + 2, next_y, True))
    return successors


def solve_minimum_bottleneck_dp(
    cost: np.ndarray,
    support: np.ndarray,
    horizontal_supported: np.ndarray,
    source_states: np.ndarray,
) -> DpCore:
    n_columns, n_states = cost.shape
    if (
        support.shape != cost.shape
        or source_states.shape != (n_states,)
        or n_columns == 0
    ):
        raise ValueError("Invalid DAG shapes")
    bottleneck = np.full(cost.shape, np.inf, dtype=float)
    cumulative = np.full(cost.shape, np.inf, dtype=float)
    previous_x = np.full(cost.shape, -1, dtype=np.int32)
    previous_y = np.full(cost.shape, -1, dtype=np.int32)
    previous_gap = np.zeros(cost.shape, dtype=bool)
    initial_states = np.flatnonzero(source_states & support[0])
    for state in initial_states:
        bottleneck[0, state] = cost[0, state]
        cumulative[0, state] = cost[0, state]
    for x_index in range(1, n_columns):
        for y_index in np.flatnonzero(support[x_index]):
            node_cost = float(cost[x_index, y_index])
            for px, py, used_gap in predecessor_nodes(
                x_index,
                int(y_index),
                horizontal_supported,
                n_states,
            ):
                if not support[px, py] or not np.isfinite(bottleneck[px, py]):
                    continue
                proposed_bottleneck = max(float(bottleneck[px, py]), node_cost)
                proposed_cumulative = float(cumulative[px, py]) + node_cost
                better = proposed_bottleneck < bottleneck[x_index, y_index] - EPS
                tied = (
                    abs(proposed_bottleneck - bottleneck[x_index, y_index]) <= EPS
                    and proposed_cumulative < cumulative[x_index, y_index] - EPS
                )
                if better or tied:
                    bottleneck[x_index, y_index] = proposed_bottleneck
                    cumulative[x_index, y_index] = proposed_cumulative
                    previous_x[x_index, y_index] = px
                    previous_y[x_index, y_index] = py
                    previous_gap[x_index, y_index] = used_gap
    sink_states = np.flatnonzero(support[-1] & np.isfinite(bottleneck[-1]))
    if not len(sink_states):
        return DpCore(
            False,
            math.nan,
            math.nan,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            False,
        )
    sink = min(
        (int(state) for state in sink_states),
        key=lambda state: (
            float(bottleneck[-1, state]),
            float(cumulative[-1, state]),
            state,
        ),
    )
    path_x: list[int] = []
    path_y: list[int] = []
    gap_used = False
    x_index = n_columns - 1
    y_index = sink
    while x_index >= 0:
        path_x.append(x_index)
        path_y.append(y_index)
        gap_used = gap_used or bool(previous_gap[x_index, y_index])
        px = int(previous_x[x_index, y_index])
        py = int(previous_y[x_index, y_index])
        if px < 0:
            break
        x_index, y_index = px, py
    path_x_array = np.asarray(path_x[::-1], dtype=np.int32)
    path_y_array = np.asarray(path_y[::-1], dtype=np.int32)
    if not len(path_x_array) or path_x_array[0] != 0 or path_x_array[-1] != n_columns - 1:
        raise AssertionError("DP reconstruction did not connect source to sink")
    return DpCore(
        True,
        float(bottleneck[-1, sink]),
        float(cumulative[-1, sink]),
        path_x_array,
        path_y_array,
        gap_used,
    )


def build_near_optimal_corridor(
    cost: np.ndarray,
    support: np.ndarray,
    horizontal_supported: np.ndarray,
    source_states: np.ndarray,
    tau_star: float,
    slack: float,
) -> np.ndarray:
    n_columns, n_states = cost.shape
    allowed = support & (cost <= tau_star + slack + EPS)
    forward = np.zeros(cost.shape, dtype=bool)
    forward[0] = source_states & allowed[0]
    for x_index in range(1, n_columns):
        for y_index in np.flatnonzero(allowed[x_index]):
            forward[x_index, y_index] = any(
                forward[px, py]
                for px, py, _ in predecessor_nodes(
                    x_index,
                    int(y_index),
                    horizontal_supported,
                    n_states,
                )
                if allowed[px, py]
            )
    backward = np.zeros(cost.shape, dtype=bool)
    backward[-1] = allowed[-1]
    for x_index in range(n_columns - 2, -1, -1):
        for y_index in np.flatnonzero(allowed[x_index]):
            backward[x_index, y_index] = any(
                backward[nx, ny]
                for nx, ny, _ in successor_nodes(
                    x_index,
                    int(y_index),
                    horizontal_supported,
                    n_states,
                )
                if allowed[nx, ny]
            )
    corridor = allowed & forward & backward
    if not corridor.any():
        raise AssertionError("A valid primary path must be contained in its corridor")
    return corridor


def empty_graph(source_states: np.ndarray, shape: tuple[int, int]) -> GraphAnalysis:
    return GraphAnalysis(
        exists=False,
        tau_star=math.nan,
        cumulative_cost=math.nan,
        path_x=np.array([], dtype=np.int32),
        path_y=np.array([], dtype=np.int32),
        path_tvt=np.array([], dtype=float),
        gap_edge_used=False,
        corridor=np.zeros(shape, dtype=bool),
        second_tau_star=math.nan,
        second_tau_gap=math.nan,
        source_states=source_states.copy(),
    )


def analyze_graph(
    cost: np.ndarray,
    support: np.ndarray,
    horizontal_supported: np.ndarray,
    state_grid_tvt: np.ndarray,
    source_states: np.ndarray,
    config: dict[str, Any],
) -> GraphAnalysis:
    core = solve_minimum_bottleneck_dp(
        cost,
        support,
        horizontal_supported,
        source_states,
    )
    if not core.exists:
        return empty_graph(source_states, cost.shape)
    slack = float(nested(config, "audit.graph.corridor_slack_cost"))
    corridor = build_near_optimal_corridor(
        cost,
        support,
        horizontal_supported,
        source_states,
        core.tau_star,
        slack,
    )
    masked_support = support.copy()
    radius = int(nested(config, "audit.graph.second_path_mask_radius_states"))
    for x_index, y_index in zip(core.path_x, core.path_y, strict=True):
        lower = max(0, int(y_index) - radius)
        upper = min(cost.shape[1], int(y_index) + radius + 1)
        masked_support[int(x_index), lower:upper] = False
    second = solve_minimum_bottleneck_dp(
        cost,
        masked_support,
        horizontal_supported,
        source_states,
    )
    second_tau = second.tau_star if second.exists else math.nan
    second_gap = second_tau - core.tau_star if second.exists else math.nan
    return GraphAnalysis(
        exists=True,
        tau_star=core.tau_star,
        cumulative_cost=core.cumulative_cost,
        path_x=core.path_x,
        path_y=core.path_y,
        path_tvt=state_grid_tvt[core.path_y],
        gap_edge_used=core.gap_edge_used,
        corridor=corridor,
        second_tau_star=float(second_tau),
        second_tau_gap=float(second_gap),
        source_states=source_states.copy(),
    )


def build_cost_surface(
    raw: RawWell,
    segment: SegmentGrid,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == "real_gr":
        typewell_z = raw.typewell_z_real
    elif variant == "shuffled_typewell_gr":
        typewell_z = raw.typewell_z_shuffled
    else:
        raise ValueError(f"Unknown surface variant: {variant}")
    interpolated = np.interp(
        segment.state_grid_tvt,
        raw.typewell_tvt,
        typewell_z,
        left=np.nan,
        right=np.nan,
    )
    state_supported = segment.typewell_supported & np.isfinite(interpolated)
    support = segment.horizontal_supported[:, None] & state_supported[None, :]
    cost = np.abs(segment.horizontal_values[:, None] - interpolated[None, :])
    cost = np.where(support, cost, np.inf)
    return cost, support


def analyze_segment_surface(
    raw: RawWell,
    segment: SegmentGrid,
    variant: str,
    config: dict[str, Any],
) -> SegmentAnalysis:
    cost, support = build_cost_surface(raw, segment, variant)
    span_sources = support[0].copy()
    if segment.short_segment_unsupported:
        spanning = empty_graph(span_sources, cost.shape)
    else:
        spanning = analyze_graph(
            cost,
            support,
            segment.horizontal_supported,
            segment.state_grid_tvt,
            span_sources,
            config,
        )
    anchored: GraphAnalysis | None = None
    if segment.is_first:
        radius = float(nested(config, "audit.graph.anchor_radius_ft"))
        anchor_sources = (
            support[0]
            & (np.abs(segment.state_grid_tvt - raw.last_known_tvt) <= radius + EPS)
        )
        anchored = (
            empty_graph(anchor_sources, cost.shape)
            if segment.short_segment_unsupported
            else analyze_graph(
                cost,
                support,
                segment.horizontal_supported,
                segment.state_grid_tvt,
                anchor_sources,
                config,
            )
        )
        primary = anchored
        primary_graph = "anchored"
    else:
        primary = spanning
        primary_graph = "spanning"
    return SegmentAnalysis(
        variant=variant,
        cost=cost,
        support=support,
        spanning=spanning,
        anchored=anchored,
        primary=primary,
        primary_graph=primary_graph,
    )


# %% [markdown]
# ## 6. Candidate and truth readout helpers

# %%
@dataclass(frozen=True)
class PathReadout:
    summary: dict[str, Any]
    endpoint_supported: np.ndarray
    corridor_inside: np.ndarray
    candidate_values: np.ndarray
    state_indices: np.ndarray
    evaluated: np.ndarray
    bad_rows: np.ndarray
    md_since: np.ndarray


def path_values_on_grid(
    raw: RawWell,
    segment: SegmentGrid,
    values: np.ndarray,
) -> np.ndarray:
    out = np.full(len(segment.column_center_md), np.nan, dtype=float)
    valid = segment.representative_rows >= raw.evaluation_start
    cache_indices = segment.representative_rows[valid] - raw.evaluation_start
    out[valid] = values[cache_indices]
    return out


def raw_values_on_grid(
    raw: RawWell,
    segment: SegmentGrid,
    values: np.ndarray,
) -> np.ndarray:
    out = np.full(len(segment.column_center_md), np.nan, dtype=float)
    valid = segment.representative_rows >= 0
    out[valid] = values[segment.representative_rows[valid]]
    return out


def nearest_state_indices(
    state_grid: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(state_grid, values)
    indices = np.clip(indices, 1, len(state_grid) - 1)
    left = state_grid[indices - 1]
    right = state_grid[indices]
    choose_left = np.abs(values - left) <= np.abs(values - right)
    indices = np.where(choose_left, indices - 1, indices).astype(np.int32)
    finite = np.isfinite(values)
    in_range = finite & (values >= state_grid[0] - EPS) & (values <= state_grid[-1] + EPS)
    indices = np.where(in_range, indices, -1).astype(np.int32)
    return indices, in_range


def count_boolean_runs(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))


def transition_is_graph_edge(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    horizontal_supported: np.ndarray,
) -> bool:
    if x1 == x0 + 1:
        return (
            horizontal_supported[x0]
            and horizontal_supported[x1]
            and abs(y1 - y0) <= 1
        )
    if x1 == x0 + 2:
        return (
            horizontal_supported[x0]
            and not horizontal_supported[x0 + 1]
            and horizontal_supported[x1]
            and abs(y1 - y0) <= 2
        )
    return False


def build_path_readout(
    raw: RawWell,
    segment: SegmentGrid,
    analysis: SegmentAnalysis,
    values: np.ndarray,
    *,
    truth_values: np.ndarray | None,
    bad_threshold_ft: float,
) -> PathReadout:
    candidate_values = path_values_on_grid(raw, segment, values)
    state_indices, in_range = nearest_state_indices(
        segment.state_grid_tvt,
        candidate_values,
    )
    endpoint_supported = np.zeros(len(candidate_values), dtype=bool)
    corridor_inside = np.zeros(len(candidate_values), dtype=bool)
    node_costs = np.full(len(candidate_values), np.nan, dtype=float)
    corridor_distances = np.full(len(candidate_values), np.nan, dtype=float)
    primary = analysis.primary
    if primary.exists:
        for x_index, y_index in enumerate(state_indices):
            if not in_range[x_index] or y_index < 0:
                continue
            if not analysis.support[x_index, y_index]:
                continue
            endpoint_supported[x_index] = True
            corridor_inside[x_index] = bool(primary.corridor[x_index, y_index])
            node_costs[x_index] = float(analysis.cost[x_index, y_index])
            corridor_states = np.flatnonzero(primary.corridor[x_index])
            if len(corridor_states):
                corridor_distances[x_index] = float(
                    np.min(
                        np.abs(
                            segment.state_grid_tvt[corridor_states]
                            - candidate_values[x_index]
                        )
                    )
                )
    supported_count = int(endpoint_supported.sum())
    inside_count = int((endpoint_supported & corridor_inside).sum())
    inside_fraction = (
        float(inside_count / supported_count) if supported_count else math.nan
    )
    outside_fraction = 1.0 - inside_fraction if supported_count else math.nan
    threshold = (
        primary.tau_star
        + float(nested(CONFIG, "audit.graph.corridor_slack_cost"))
        if primary.exists
        else math.nan
    )
    finite_node_costs = node_costs[np.isfinite(node_costs)]
    excess = (
        np.maximum(finite_node_costs - threshold, 0.0)
        if np.isfinite(threshold)
        else np.array([], dtype=float)
    )
    corridor_distance_values = corridor_distances[np.isfinite(corridor_distances)]
    valid_positions = np.flatnonzero(endpoint_supported)
    transition_valid: list[bool] = []
    crossing_count = 0
    for left_index, right_index in zip(
        valid_positions[:-1],
        valid_positions[1:],
        strict=True,
    ):
        left_y = int(state_indices[left_index])
        right_y = int(state_indices[right_index])
        edge_valid = transition_is_graph_edge(
            int(left_index),
            left_y,
            int(right_index),
            right_y,
            segment.horizontal_supported,
        )
        transition_valid.append(
            edge_valid
            and corridor_inside[left_index]
            and corridor_inside[right_index]
        )
        if edge_valid and corridor_inside[left_index] != corridor_inside[right_index]:
            crossing_count += 1
    entry_to_exit = bool(
        supported_count > 0
        and valid_positions[0] == 0
        and valid_positions[-1] == len(candidate_values) - 1
        and corridor_inside[valid_positions].all()
        and all(transition_valid)
    )
    outside_mask = endpoint_supported & ~corridor_inside
    ridge_event_count = count_boolean_runs(outside_mask)
    singleton_gap_count = 0
    for x_index in range(1, len(segment.horizontal_supported) - 1):
        if (
            not segment.horizontal_supported[x_index]
            and segment.horizontal_supported[x_index - 1]
            and segment.horizontal_supported[x_index + 1]
        ):
            singleton_gap_count += 1
    truth_grid = (
        raw_values_on_grid(raw, segment, truth_values)
        if truth_values is not None
        else np.full(len(candidate_values), np.nan, dtype=float)
    )
    evaluated = (
        endpoint_supported
        & np.isfinite(candidate_values)
        & np.isfinite(truth_grid)
    )
    bad_rows = evaluated & (
        np.abs(candidate_values - truth_grid) > bad_threshold_ft
    )
    md_since = path_values_on_grid(
        raw,
        segment,
        raw.cache["md_since"].to_numpy(float),
    )
    summary = {
        "endpoint_supported_count": supported_count,
        "endpoint_supported_fraction": float(
            supported_count / len(candidate_values)
        ),
        "corridor_inside_count": inside_count,
        "corridor_inside_fraction": inside_fraction,
        "corridor_outside_fraction": outside_fraction,
        "corridor_distance_median_ft": (
            float(np.median(corridor_distance_values))
            if len(corridor_distance_values)
            else math.nan
        ),
        "corridor_distance_p90_ft": (
            float(np.quantile(corridor_distance_values, 0.90))
            if len(corridor_distance_values)
            else math.nan
        ),
        "corridor_distance_max_ft": (
            float(np.max(corridor_distance_values))
            if len(corridor_distance_values)
            else math.nan
        ),
        "candidate_node_cost_mean": (
            float(np.mean(finite_node_costs)) if len(finite_node_costs) else math.nan
        ),
        "candidate_node_cost_p90": (
            float(np.quantile(finite_node_costs, 0.90))
            if len(finite_node_costs)
            else math.nan
        ),
        "candidate_node_cost_max": (
            float(np.max(finite_node_costs)) if len(finite_node_costs) else math.nan
        ),
        "candidate_bottleneck": (
            float(np.max(finite_node_costs)) if len(finite_node_costs) else math.nan
        ),
        "candidate_bottleneck_minus_tau": (
            float(np.max(finite_node_costs) - primary.tau_star)
            if len(finite_node_costs) and primary.exists
            else math.nan
        ),
        "mean_excess_cost": float(np.mean(excess)) if len(excess) else math.nan,
        "entry_to_exit_reachable": entry_to_exit,
        "corridor_crossing_count": int(crossing_count),
        "ridge_event_count": int(ridge_event_count),
        "gap_edge_exposure_count": int(singleton_gap_count),
        "unsupported_fraction": float(1.0 - supported_count / len(candidate_values)),
        "evaluated_row_count": int(evaluated.sum()),
        "bad_row_count": int(bad_rows.sum()),
        "good_row_count": int(evaluated.sum() - bad_rows.sum()),
        "bad_row_fraction": (
            float(bad_rows.sum() / evaluated.sum())
            if evaluated.any()
            else math.nan
        ),
    }
    return PathReadout(
        summary=summary,
        endpoint_supported=endpoint_supported,
        corridor_inside=corridor_inside,
        candidate_values=candidate_values,
        state_indices=state_indices,
        evaluated=evaluated,
        bad_rows=bad_rows,
        md_since=md_since,
    )


def segment_metric_row(
    raw: RawWell,
    segment: SegmentGrid,
    analysis: SegmentAnalysis,
) -> dict[str, Any]:
    anchored = analysis.anchored
    return {
        "well": raw.well,
        "segment_id": segment.segment_id,
        "variant": analysis.variant,
        "start_md": segment.start_md,
        "end_md": segment.end_md,
        "column_count": len(segment.column_center_md),
        "horizontal_supported_fraction": float(
            np.mean(segment.horizontal_supported)
        ),
        "typewell_supported_fraction": float(
            np.mean(segment.typewell_supported)
        ),
        "short_segment_unsupported": segment.short_segment_unsupported,
        "prior_tvt": segment.prior_tvt,
        "primary_graph": analysis.primary_graph,
        "primary_path_exists": analysis.primary.exists,
        "tau_star": analysis.primary.tau_star,
        "cumulative_cost": analysis.primary.cumulative_cost,
        "tau_second_star": analysis.primary.second_tau_star,
        "tau_second_gap": analysis.primary.second_tau_gap,
        "gap_edge_used": analysis.primary.gap_edge_used,
        "corridor_node_count": int(analysis.primary.corridor.sum()),
        "tau_span_star": analysis.spanning.tau_star,
        "tau_anchor_star": anchored.tau_star if anchored is not None else math.nan,
        "shuffle_shift": raw.shuffle_shift if analysis.variant != "real_gr" else 0,
        "horizontal_norm_center": raw.horizontal_norm.center,
        "horizontal_norm_scale": raw.horizontal_norm.scale,
        "horizontal_norm_fallback": raw.horizontal_norm.fallback,
        "typewell_norm_center": raw.typewell_norm.center,
        "typewell_norm_scale": raw.typewell_norm.scale,
        "typewell_norm_fallback": raw.typewell_norm.fallback,
    }


# %% [markdown]
# ## 7. Stage 0 deterministic selection and plots

# %%
def select_stage0_wells(
    inventory: dict[str, RawInventoryRecord],
    config: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    stage0 = nested(config, "audit.stage0")
    selected: list[str] = []
    roles: dict[str, str] = {}

    def take(
        ordered: list[RawInventoryRecord],
        count: int,
        role: str,
    ) -> None:
        for record in ordered:
            if record.well in roles:
                continue
            selected.append(record.well)
            roles[record.well] = role
            if sum(value == role for value in roles.values()) >= count:
                break

    records = list(inventory.values())
    take(
        sorted(
            records,
            key=lambda record: (
                -record.tail_gr_missing_fraction,
                record.stable_order,
            ),
        ),
        int(stage0["high_gr_missing_wells"]),
        "high_gr_missing",
    )
    take(
        sorted(
            records,
            key=lambda record: (
                record.tail_gr_iqr
                if np.isfinite(record.tail_gr_iqr)
                else math.inf,
                record.stable_order,
            ),
        ),
        int(stage0["flat_gr_wells"]),
        "flat_gr",
    )
    take(
        sorted(
            records,
            key=lambda record: (-record.tail_span_ft, record.stable_order),
        ),
        int(stage0["long_tail_wells"]),
        "long_tail",
    )
    take(
        sorted(records, key=lambda record: record.stable_order),
        int(stage0["stable_normal_wells"]),
        "stable_sha_normal",
    )
    expected = sum(
        int(stage0[key])
        for key in [
            "stable_normal_wells",
            "high_gr_missing_wells",
            "flat_gr_wells",
            "long_tail_wells",
        ]
    )
    if len(selected) != expected:
        raise ValueError("Stage 0 could not select the fixed number of unique wells")
    return selected, roles


def stage0_segment_indices(
    segments: list[SegmentGrid],
    config: dict[str, Any],
) -> list[int]:
    positions = list(nested(config, "audit.stage0.segment_positions"))
    lookup = {
        "first": 0,
        "middle": len(segments) // 2,
        "last": len(segments) - 1,
    }
    indices: list[int] = []
    for position in positions:
        index = lookup[str(position)]
        if index not in indices:
            indices.append(index)
    return indices


def state_edges(state_grid: np.ndarray) -> np.ndarray:
    if len(state_grid) < 2:
        raise ValueError("State grid requires at least two states")
    step = float(np.median(np.diff(state_grid)))
    return np.concatenate(
        [[state_grid[0] - step / 2.0], state_grid + step / 2.0]
    )


def plot_stage0_segment(
    raw: RawWell,
    segment: SegmentGrid,
    analyses: dict[str, SegmentAnalysis],
    candidate_readouts: dict[str, dict[str, PathReadout]],
    truth_readouts: dict[str, PathReadout],
    out_path: Path,
    config: dict[str, Any],
) -> None:
    variants = ["real_gr", "shuffled_typewell_gr"]
    candidate_colors = {
        "pf_ancc": "#1f77b4",
        "beam_mean": "#ff7f0e",
        "likpf_mean": "#2ca02c",
        "sc_ens": "#9467bd",
        "hyb": "#8c564b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    x_edges = np.concatenate(
        [segment.column_left_md, [segment.column_right_md[-1]]]
    )
    y_edges = state_edges(segment.state_grid_tvt)
    vmin = float(nested(config, "audit.stage0.plot_vmin"))
    vmax = float(nested(config, "audit.stage0.plot_vmax"))
    for axis, variant in zip(axes, variants, strict=True):
        analysis = analyses[variant]
        masked_cost = np.ma.masked_where(~analysis.support, analysis.cost)
        axis.pcolormesh(
            x_edges,
            y_edges,
            masked_cost.T,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        corridor_x, corridor_y = np.where(analysis.primary.corridor)
        if len(corridor_x):
            axis.scatter(
                segment.column_center_md[corridor_x],
                segment.state_grid_tvt[corridor_y],
                s=7,
                color="#00aa55",
                alpha=0.35,
                label="near-optimal corridor",
            )
        if analysis.primary.exists:
            axis.plot(
                segment.column_center_md[analysis.primary.path_x],
                analysis.primary.path_tvt,
                color="black",
                linewidth=2.0,
                label="primary bottleneck path",
            )
        for candidate, readout in candidate_readouts[variant].items():
            axis.plot(
                segment.column_center_md,
                readout.candidate_values,
                color=candidate_colors[candidate],
                linewidth=1.1,
                alpha=0.9,
                label=candidate,
            )
        truth = truth_readouts[variant]
        axis.plot(
            segment.column_center_md,
            truth.candidate_values,
            color="#ff00aa",
            linewidth=1.5,
            linestyle="--",
            label="truth evaluation overlay",
        )
        source_y = segment.state_grid_tvt[analysis.primary.source_states]
        if len(source_y):
            axis.scatter(
                np.full(len(source_y), segment.column_center_md[0]),
                source_y,
                marker=">",
                color="#ffff00",
                edgecolor="black",
                s=25,
                label="source",
            )
        sink_states = np.flatnonzero(analysis.support[-1])
        if len(sink_states):
            axis.scatter(
                np.full(len(sink_states), segment.column_center_md[-1]),
                segment.state_grid_tvt[sink_states],
                marker="|",
                color="#ffffff",
                s=18,
                label="sink support",
            )
        axis.scatter(
            [segment.column_center_md[0]],
            [raw.last_known_tvt],
            marker="*",
            color="#ffff00",
            edgecolor="black",
            s=120,
            label="last known anchor",
        )
        axis.set_title(
            f"{variant}\n"
            f"{analysis.primary_graph} tau={analysis.primary.tau_star:.3f} "
            f"second-gap={analysis.primary.second_tau_gap:.3f}"
        )
        axis.set_xlabel("Horizontal MD (ft), increasing left to right")
        axis.grid(False)
    axes[0].set_ylabel("Typewell TVT (ft), increasing bottom to top")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=5,
        fontsize=8,
    )
    fig.suptitle(
        f"{raw.well} segment {segment.segment_id}: "
        f"MD {segment.start_md:.2f}–{segment.end_md:.2f}; "
        f"flat-Z prior {segment.prior_tvt:.2f}"
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_stage0(
    selected_wells: list[str],
    roles: dict[str, str],
    selected_frames: dict[str, pd.DataFrame],
    inventory: dict[str, RawInventoryRecord],
    specs: list[CandidateSpec],
    out_dir: Path,
    config: dict[str, Any],
    input_sha: dict[str, str],
) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    bad_threshold = float(nested(config, "audit.bad_candidate_threshold_ft"))
    plot_dir = out_dir / "stage0_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for well in selected_wells:
        raw = load_raw_well(inventory[well], selected_frames[well], specs, config)
        segments = build_all_segments(raw, config)
        for segment_index in stage0_segment_indices(segments, config):
            segment = segments[segment_index]
            analyses = {
                variant: analyze_segment_surface(raw, segment, variant, config)
                for variant in ["real_gr", "shuffled_typewell_gr"]
            }
            if not np.array_equal(
                analyses["real_gr"].support,
                analyses["shuffled_typewell_gr"].support,
            ):
                raise AssertionError("Real/shuffled support parity failed")
            if not np.array_equal(
                analyses["real_gr"].primary.source_states,
                analyses["shuffled_typewell_gr"].primary.source_states,
            ):
                raise AssertionError("Real/shuffled source parity failed")
            candidate_readouts: dict[str, dict[str, PathReadout]] = {}
            truth_readouts: dict[str, PathReadout] = {}
            truth_tail = raw.truth_tvt[raw.evaluation_start :]
            for variant, analysis in analyses.items():
                candidate_readouts[variant] = {
                    spec.name: build_path_readout(
                        raw,
                        segment,
                        analysis,
                        raw.candidates[spec.name],
                        truth_values=raw.truth_tvt,
                        bad_threshold_ft=bad_threshold,
                    )
                    for spec in specs
                }
                truth_readouts[variant] = build_path_readout(
                    raw,
                    segment,
                    analysis,
                    truth_tail,
                    truth_values=None,
                    bad_threshold_ft=bad_threshold,
                )
            for spec in specs:
                if not np.array_equal(
                    candidate_readouts["real_gr"][spec.name].candidate_values,
                    candidate_readouts["shuffled_typewell_gr"][
                        spec.name
                    ].candidate_values,
                    equal_nan=True,
                ):
                    raise AssertionError("Real/shuffled candidate parity failed")
            filename = (
                f"{well}_segment_{segment.segment_id:04d}_"
                f"md_{segment.start_md:.2f}_{segment.end_md:.2f}.png"
            )
            plot_path = plot_dir / filename
            plot_stage0_segment(
                raw,
                segment,
                analyses,
                candidate_readouts,
                truth_readouts,
                plot_path,
                config,
            )
            plot_sha = sha256_path(plot_path)
            for variant, analysis in analyses.items():
                manifest_rows.append(
                    {
                        "well": well,
                        "selection_role": roles[well],
                        "segment_id": segment.segment_id,
                        "segment_position": (
                            "first"
                            if segment.segment_id == 0
                            else (
                                "last"
                                if segment.segment_id == len(segments) - 1
                                else "middle"
                            )
                        ),
                        "variant": variant,
                        "start_md": segment.start_md,
                        "end_md": segment.end_md,
                        "column_count": len(segment.column_center_md),
                        "state_count": len(segment.state_grid_tvt),
                        "horizontal_bin_ft": nested(
                            config,
                            "audit.segment.horizontal_bin_ft",
                        ),
                        "state_step_ft": nested(
                            config,
                            "audit.typewell_grid.state_step_ft",
                        ),
                        "prior_tvt": segment.prior_tvt,
                        "source_state_count": int(
                            analysis.primary.source_states.sum()
                        ),
                        "sink_state_count": int(analysis.support[-1].sum()),
                        "path_exists": analysis.primary.exists,
                        "tau_star": analysis.primary.tau_star,
                        "corridor_node_count": int(
                            analysis.primary.corridor.sum()
                        ),
                        "horizontal_support_sha": hashlib.sha256(
                            segment.horizontal_supported.tobytes()
                        ).hexdigest(),
                        "typewell_support_sha": hashlib.sha256(
                            segment.typewell_supported.tobytes()
                        ).hexdigest(),
                        "plot_path": str(plot_path.relative_to(out_dir)),
                        "plot_sha256": plot_sha,
                        "horizontal_source_sha256": sha256_path(
                            inventory[well].horizontal_path
                        ),
                        "typewell_source_sha256": sha256_path(
                            inventory[well].typewell_path
                        ),
                    }
                )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["well", "segment_id", "variant"],
        kind="mergesort",
    )
    manifest_path = out_dir / str(
        nested(config, "audit.outputs.stage0_plot_manifest_filename")
    )
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    preview = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_preview",
        "scientific_decision": "not_evaluated_in_stage0",
        "selected_wells": [
            {
                "well": well,
                "role": roles[well],
                "tail_span_ft": inventory[well].tail_span_ft,
                "tail_gr_missing_fraction": inventory[
                    well
                ].tail_gr_missing_fraction,
                "tail_gr_iqr": inventory[well].tail_gr_iqr,
            }
            for well in selected_wells
        ],
        "plot_count": int(manifest["plot_path"].nunique()),
        "manifest_row_count": len(manifest),
        "synthetic_contract": "passed_before_stage0",
        "manual_parity_confirmed": bool(
            nested(config, "audit.stage0.manual_parity_confirmed")
        ),
        "input_sha": input_sha,
        "manifest_sha256": sha256_path(manifest_path),
    }
    preview_path = out_dir / str(
        nested(config, "audit.outputs.stage0_preview_manifest_filename")
    )
    write_json(preview_path, preview)
    print("Stage 0 plot manifest:")
    display(manifest.head(12))
    return {
        "status": "stage0_preview_complete_manual_parity_pending",
        "well_count": len(selected_wells),
        "plot_count": int(manifest["plot_path"].nunique()),
        "manifest_path": manifest_path,
        "preview_path": preview_path,
        "generated_paths": [manifest_path, preview_path, *sorted(plot_dir.glob("*.png"))],
    }


# %% [markdown]
# ## 8. Stage 1 metrics, overlap, groups, and guards

# %%
@dataclass(frozen=True)
class SegmentBundle:
    segment: SegmentGrid
    analysis: SegmentAnalysis
    candidate_readouts: dict[str, PathReadout]
    truth_readout: PathReadout


def safe_weighted_auc(
    scores: np.ndarray,
    positive_weight: np.ndarray,
    negative_weight: np.ndarray,
) -> float:
    finite = (
        np.isfinite(scores)
        & np.isfinite(positive_weight)
        & np.isfinite(negative_weight)
    )
    scores = scores[finite]
    positive_weight = positive_weight[finite]
    negative_weight = negative_weight[finite]
    total_positive = float(positive_weight.sum())
    total_negative = float(negative_weight.sum())
    if not len(scores) or total_positive <= 0 or total_negative <= 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    positive_weight = positive_weight[order]
    negative_weight = negative_weight[order]
    contribution = 0.0
    cumulative_negative = 0.0
    index = 0
    while index < len(scores):
        stop = index + 1
        while stop < len(scores) and abs(scores[stop] - scores[index]) <= EPS:
            stop += 1
        positive_tie = float(positive_weight[index:stop].sum())
        negative_tie = float(negative_weight[index:stop].sum())
        contribution += positive_tie * (
            cumulative_negative + 0.5 * negative_tie
        )
        cumulative_negative += negative_tie
        index = stop
    return contribution / (total_positive * total_negative)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    if not len(values):
        return math.nan
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cutoff = quantile * float(weights.sum())
    index = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    return float(values[min(index, len(values) - 1)])


def safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return math.nan
    return float(pd.Series(left[finite]).corr(pd.Series(right[finite]), method=method))


def summarize_risk_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_segment_count": 0,
            "evaluated_row_weight": 0.0,
            "bad_row_weight": 0.0,
            "good_row_weight": 0.0,
            "auc": math.nan,
            "q90_risk_threshold": math.nan,
            "q90_bad_rate": math.nan,
            "overall_bad_rate": math.nan,
            "q90_bad_rate_lift": math.nan,
            "q90_good_false_alert_rate": math.nan,
        }
    scores = frame["risk"].to_numpy(float)
    bad = frame["bad_weight"].to_numpy(float)
    good = frame["good_weight"].to_numpy(float)
    total = bad + good
    threshold = weighted_quantile(scores, total, 0.90)
    high = np.isfinite(scores) & (scores >= threshold - EPS)
    high_bad = float(bad[high].sum())
    high_good = float(good[high].sum())
    bad_total = float(bad.sum())
    good_total = float(good.sum())
    evaluated_total = bad_total + good_total
    high_total = high_bad + high_good
    overall_bad_rate = bad_total / evaluated_total if evaluated_total else math.nan
    high_bad_rate = high_bad / high_total if high_total else math.nan
    lift = (
        high_bad_rate / overall_bad_rate
        if np.isfinite(overall_bad_rate) and overall_bad_rate > 0
        else math.nan
    )
    return {
        "sample_segment_count": int(len(frame)),
        "evaluated_row_weight": evaluated_total,
        "bad_row_weight": bad_total,
        "good_row_weight": good_total,
        "auc": safe_weighted_auc(scores, bad, good),
        "q90_risk_threshold": threshold,
        "q90_bad_rate": high_bad_rate,
        "overall_bad_rate": overall_bad_rate,
        "q90_bad_rate_lift": lift,
        "q90_good_false_alert_rate": (
            high_good / good_total if good_total else math.nan
        ),
    }


def distance_bucket_array(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    edges = np.asarray(nested(config, "audit.distance_buckets.edges"), dtype=float)
    labels = np.asarray(
        nested(config, "audit.distance_buckets.labels"),
        dtype=object,
    )
    bins = np.searchsorted(edges[1:], values, side="left")
    valid = np.isfinite(values) & (bins >= 0) & (bins < len(labels))
    out = np.full(len(values), "", dtype=object)
    out[valid] = labels[bins[valid]]
    return out


def segment_condition_names(
    raw: RawWell,
    segment: SegmentGrid,
    analysis: SegmentAnalysis,
    config: dict[str, Any],
) -> list[str]:
    finite_h = segment.horizontal_values[np.isfinite(segment.horizontal_values)]
    flat_threshold = float(
        nested(config, "audit.buckets.flat_horizontal_z_std_max", 0.15)
    )
    long_tail_threshold = float(
        nested(config, "audit.buckets.long_tail_min_span_ft", 1000.0)
    )
    names: list[str] = []
    if not segment.horizontal_supported.all():
        names.append("missing")
    if len(finite_h) and float(np.std(finite_h)) <= flat_threshold:
        names.append("flat")
    if segment.short_segment_unsupported or not analysis.primary.exists:
        names.append("unsupported")
    if analysis.primary.gap_edge_used:
        names.append("gap_edge")
    if raw.md[-1] - raw.md[raw.evaluation_start] >= long_tail_threshold:
        names.append("long_tail")
    return names


def risk_contribution_rows(
    raw: RawWell,
    segment: SegmentGrid,
    analysis: SegmentAnalysis,
    candidate: str,
    readout: PathReadout,
    hidden_groups: dict[str, set[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    risk = float(readout.summary["corridor_outside_fraction"])
    bucket = distance_bucket_array(readout.md_since, config)
    rows: list[dict[str, Any]] = []

    def append(group_type: str, group_name: str, mask: np.ndarray) -> None:
        evaluated = mask & readout.evaluated
        bad_weight = int((evaluated & readout.bad_rows).sum())
        good_weight = int(evaluated.sum() - bad_weight)
        rows.append(
            {
                "entity": "candidate",
                "variant": analysis.variant,
                "candidate": candidate,
                "group_type": group_type,
                "group_name": group_name,
                "well": raw.well,
                "segment_id": segment.segment_id,
                "risk": risk,
                "bad_weight": bad_weight,
                "good_weight": good_weight,
                "inside_weight": math.nan,
                "supported_weight": math.nan,
            }
        )

    all_mask = np.ones(len(readout.evaluated), dtype=bool)
    append("overall", "overall", all_mask)
    for label in nested(config, "audit.distance_buckets.labels"):
        append("distance", str(label), bucket == str(label))
    for group_name, wells in hidden_groups.items():
        if raw.well in wells:
            append("hidden_like", group_name, all_mask)
    for condition in segment_condition_names(raw, segment, analysis, config):
        append("condition", condition, all_mask)
    return rows


def truth_contribution_rows(
    raw: RawWell,
    segment: SegmentGrid,
    analysis: SegmentAnalysis,
    readout: PathReadout,
    hidden_groups: dict[str, set[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    bucket = distance_bucket_array(readout.md_since, config)
    rows: list[dict[str, Any]] = []

    def append(group_type: str, group_name: str, mask: np.ndarray) -> None:
        supported = mask & readout.endpoint_supported
        inside = supported & readout.corridor_inside
        rows.append(
            {
                "entity": "truth",
                "variant": analysis.variant,
                "candidate": "truth",
                "group_type": group_type,
                "group_name": group_name,
                "well": raw.well,
                "segment_id": segment.segment_id,
                "risk": math.nan,
                "bad_weight": math.nan,
                "good_weight": math.nan,
                "inside_weight": int(inside.sum()),
                "supported_weight": int(supported.sum()),
            }
        )

    all_mask = np.ones(len(readout.endpoint_supported), dtype=bool)
    append("overall", "overall", all_mask)
    for label in nested(config, "audit.distance_buckets.labels"):
        append("distance", str(label), bucket == str(label))
    for group_name, wells in hidden_groups.items():
        if raw.well in wells:
            append("hidden_like", group_name, all_mask)
    for condition in segment_condition_names(raw, segment, analysis, config):
        append("condition", condition, all_mask)
    return rows


def candidate_segment_row(
    raw: RawWell,
    segment: SegmentGrid,
    analysis: SegmentAnalysis,
    candidate: str,
    readout: PathReadout,
) -> dict[str, Any]:
    row = {
        "well": raw.well,
        "segment_id": segment.segment_id,
        "variant": analysis.variant,
        "candidate": candidate,
        "start_md": segment.start_md,
        "end_md": segment.end_md,
        "primary_graph": analysis.primary_graph,
        "primary_path_exists": analysis.primary.exists,
        "tau_star": analysis.primary.tau_star,
        "tau_second_gap": analysis.primary.second_tau_gap,
        "gap_edge_used": analysis.primary.gap_edge_used,
        "short_segment_unsupported": segment.short_segment_unsupported,
    }
    row.update(readout.summary)
    row["risk"] = row["corridor_outside_fraction"]
    row["bad_weight"] = row["bad_row_count"]
    row["good_weight"] = row["good_row_count"]
    return row


def corridor_coordinate_set(
    bundle: SegmentBundle,
    start: float,
    end: float,
) -> set[tuple[int, int]]:
    x_indices, y_indices = np.where(bundle.analysis.primary.corridor)
    output: set[tuple[int, int]] = set()
    for x_index, y_index in zip(x_indices, y_indices, strict=True):
        md = float(bundle.segment.column_center_md[x_index])
        if md < start - EPS or md > end + EPS:
            continue
        tvt = float(bundle.segment.state_grid_tvt[y_index])
        output.add((int(round(md / 4.0)), int(round(tvt / 4.0))))
    return output


def overlap_pair_readout(
    left: SegmentBundle,
    right: SegmentBundle,
    candidate_names: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    overlap_start = max(left.segment.start_md, right.segment.start_md)
    overlap_end = min(left.segment.end_md, right.segment.end_md)
    if overlap_end <= overlap_start + EPS:
        raise ValueError("Segments do not overlap")
    path_differences = np.array([], dtype=float)
    if left.analysis.primary.exists and right.analysis.primary.exists:
        left_md = left.segment.column_center_md[left.analysis.primary.path_x]
        right_md = right.segment.column_center_md[right.analysis.primary.path_x]
        grid = np.arange(
            overlap_start + 2.0,
            overlap_end - 2.0 + EPS,
            4.0,
        )
        if len(grid):
            left_tvt = np.interp(grid, left_md, left.analysis.primary.path_tvt)
            right_tvt = np.interp(grid, right_md, right.analysis.primary.path_tvt)
            path_differences = np.abs(left_tvt - right_tvt)
    left_corridor = corridor_coordinate_set(left, overlap_start, overlap_end)
    right_corridor = corridor_coordinate_set(right, overlap_start, overlap_end)
    union = left_corridor | right_corridor
    corridor_jaccard = (
        len(left_corridor & right_corridor) / len(union) if union else math.nan
    )
    left_risk = np.asarray(
        [
            left.candidate_readouts[name].summary[
                "corridor_outside_fraction"
            ]
            for name in candidate_names
        ],
        dtype=float,
    )
    right_risk = np.asarray(
        [
            right.candidate_readouts[name].summary[
                "corridor_outside_fraction"
            ]
            for name in candidate_names
        ],
        dtype=float,
    )
    left_event = np.asarray(
        [
            left.candidate_readouts[name].summary["ridge_event_count"] > 0
            for name in candidate_names
        ],
        dtype=bool,
    )
    right_event = np.asarray(
        [
            right.candidate_readouts[name].summary["ridge_event_count"] > 0
            for name in candidate_names
        ],
        dtype=bool,
    )
    row = {
        "scope": "segment_pair",
        "well": "",
        "variant": left.analysis.variant,
        "left_segment_id": left.segment.segment_id,
        "right_segment_id": right.segment.segment_id,
        "overlap_start_md": overlap_start,
        "overlap_end_md": overlap_end,
        "path_tvt_difference_median_ft": (
            float(np.median(path_differences))
            if len(path_differences)
            else math.nan
        ),
        "path_tvt_difference_p90_ft": (
            float(np.quantile(path_differences, 0.90))
            if len(path_differences)
            else math.nan
        ),
        "corridor_jaccard": corridor_jaccard,
        "candidate_risk_pearson": safe_correlation(
            left_risk,
            right_risk,
            method="pearson",
        ),
        "candidate_risk_spearman": safe_correlation(
            left_risk,
            right_risk,
            method="spearman",
        ),
        "candidate_event_agreement": float(np.mean(left_event == right_event)),
    }
    finite_risk = np.isfinite(left_risk) & np.isfinite(right_risk)
    return row, path_differences, left_risk[finite_risk], right_risk[finite_risk]


def build_overlap_metrics(
    bundles_by_variant: dict[str, list[SegmentBundle]],
    candidate_names: list[str],
    well: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    rows: list[dict[str, Any]] = []
    raw_values: dict[str, dict[str, list[float]]] = {
        variant: {"path": [], "left_risk": [], "right_risk": []}
        for variant in bundles_by_variant
    }
    for variant, bundles in bundles_by_variant.items():
        for left_index, left in enumerate(bundles):
            for right in bundles[left_index + 1 :]:
                if right.segment.start_md >= left.segment.end_md - EPS:
                    break
                row, path_diff, left_risk, right_risk = overlap_pair_readout(
                    left,
                    right,
                    candidate_names,
                )
                row["well"] = well
                rows.append(row)
                raw_values[variant]["path"].extend(path_diff.tolist())
                raw_values[variant]["left_risk"].extend(left_risk.tolist())
                raw_values[variant]["right_risk"].extend(right_risk.tolist())
    return rows, raw_values


def finalize_candidate_metrics(candidate_segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, variant_frame in candidate_segments.groupby("variant", sort=True):
        candidate_groups: list[tuple[str, pd.DataFrame]] = [
            (str(candidate), frame)
            for candidate, frame in variant_frame.groupby("candidate", sort=True)
        ]
        candidate_groups.append(("__pooled__", variant_frame))
        for candidate, frame in candidate_groups:
            row = {"variant": str(variant), "candidate": candidate}
            row.update(
                summarize_risk_frame(
                    frame[["risk", "bad_row_count", "good_row_count"]].rename(
                        columns={
                            "bad_row_count": "bad_weight",
                            "good_row_count": "good_weight",
                        }
                    )
                )
            )
            rows.append(row)
    metrics = pd.DataFrame(rows)
    paired = metrics.pivot(index="candidate", columns="variant")
    for candidate in metrics["candidate"].unique():
        mask = metrics["candidate"] == candidate
        try:
            real_auc = float(paired.loc[candidate, ("auc", "real_gr")])
            shuffled_auc = float(
                paired.loc[candidate, ("auc", "shuffled_typewell_gr")]
            )
            real_lift = float(
                paired.loc[
                    candidate,
                    ("q90_bad_rate_lift", "real_gr"),
                ]
            )
            shuffled_lift = float(
                paired.loc[
                    candidate,
                    ("q90_bad_rate_lift", "shuffled_typewell_gr"),
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue
        metrics.loc[mask, "real_minus_shuffled_auc"] = real_auc - shuffled_auc
        metrics.loc[
            mask,
            "real_minus_shuffled_q90_lift",
        ] = real_lift - shuffled_lift
    return metrics.sort_values(["candidate", "variant"], kind="mergesort")


def finalize_group_metrics(contributions: pd.DataFrame) -> pd.DataFrame:
    candidate_contrib = contributions.loc[
        contributions["entity"] == "candidate"
    ].copy()
    pooled = candidate_contrib.copy()
    pooled["candidate"] = "__pooled__"
    candidate_contrib = pd.concat([candidate_contrib, pooled], ignore_index=True)
    rows: list[dict[str, Any]] = []
    group_columns = [
        "entity",
        "variant",
        "candidate",
        "group_type",
        "group_name",
    ]
    for keys, frame in candidate_contrib.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row.update(summarize_risk_frame(frame))
        row["truth_corridor_coverage"] = math.nan
        rows.append(row)
    truth_contrib = contributions.loc[contributions["entity"] == "truth"]
    truth_columns = ["entity", "variant", "candidate", "group_type", "group_name"]
    for keys, frame in truth_contrib.groupby(truth_columns, sort=True):
        supported = float(frame["supported_weight"].sum())
        inside = float(frame["inside_weight"].sum())
        row = dict(zip(truth_columns, keys, strict=True))
        row.update(
            {
                "sample_segment_count": int(len(frame)),
                "evaluated_row_weight": supported,
                "bad_row_weight": math.nan,
                "good_row_weight": math.nan,
                "auc": math.nan,
                "q90_risk_threshold": math.nan,
                "q90_bad_rate": math.nan,
                "overall_bad_rate": math.nan,
                "q90_bad_rate_lift": math.nan,
                "q90_good_false_alert_rate": math.nan,
                "truth_corridor_coverage": (
                    inside / supported if supported else math.nan
                ),
            }
        )
        rows.append(row)
    metrics = pd.DataFrame(rows)
    paired_keys = ["entity", "candidate", "group_type", "group_name"]
    for keys, frame in metrics.groupby(paired_keys, sort=False):
        if set(frame["variant"]) != {"real_gr", "shuffled_typewell_gr"}:
            continue
        real = frame.loc[frame["variant"] == "real_gr"].iloc[0]
        shuffled = frame.loc[
            frame["variant"] == "shuffled_typewell_gr"
        ].iloc[0]
        mask = np.ones(len(metrics), dtype=bool)
        for column, value in zip(paired_keys, keys, strict=True):
            mask &= metrics[column].to_numpy(object) == value
        metrics.loc[mask, "real_minus_shuffled_auc"] = (
            float(real["auc"]) - float(shuffled["auc"])
            if np.isfinite(real["auc"]) and np.isfinite(shuffled["auc"])
            else math.nan
        )
        metrics.loc[mask, "real_minus_shuffled_truth_coverage"] = (
            float(real["truth_corridor_coverage"])
            - float(shuffled["truth_corridor_coverage"])
            if np.isfinite(real["truth_corridor_coverage"])
            and np.isfinite(shuffled["truth_corridor_coverage"])
            else math.nan
        )
    return metrics.sort_values(
        ["entity", "group_type", "group_name", "candidate", "variant"],
        kind="mergesort",
    )


def finalize_overlap_metrics(
    pair_rows: list[dict[str, Any]],
    raw_overlap: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    rows = list(pair_rows)
    pair_frame = pd.DataFrame(pair_rows)
    for variant, values in raw_overlap.items():
        path = np.asarray(values["path"], dtype=float)
        left_risk = np.asarray(values["left_risk"], dtype=float)
        right_risk = np.asarray(values["right_risk"], dtype=float)
        variant_pairs = (
            pair_frame.loc[pair_frame["variant"] == variant]
            if not pair_frame.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "scope": "overall",
                "well": "__all__",
                "variant": variant,
                "left_segment_id": math.nan,
                "right_segment_id": math.nan,
                "overlap_start_md": math.nan,
                "overlap_end_md": math.nan,
                "path_tvt_difference_median_ft": (
                    float(np.median(path)) if len(path) else math.nan
                ),
                "path_tvt_difference_p90_ft": (
                    float(np.quantile(path, 0.90)) if len(path) else math.nan
                ),
                "corridor_jaccard": (
                    float(variant_pairs["corridor_jaccard"].mean())
                    if not variant_pairs.empty
                    else math.nan
                ),
                "candidate_risk_pearson": safe_correlation(
                    left_risk,
                    right_risk,
                    method="pearson",
                ),
                "candidate_risk_spearman": safe_correlation(
                    left_risk,
                    right_risk,
                    method="spearman",
                ),
                "candidate_event_agreement": (
                    float(variant_pairs["candidate_event_agreement"].mean())
                    if not variant_pairs.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scope", "well", "variant", "left_segment_id"],
        kind="mergesort",
    )


def build_by_well_metrics(
    candidate_segments: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    truth_contributions: pd.DataFrame,
) -> pd.DataFrame:
    pooled_real = candidate_metrics.loc[
        (candidate_metrics["variant"] == "real_gr")
        & (candidate_metrics["candidate"] == "__pooled__")
    ]
    if len(pooled_real) != 1:
        raise ValueError("Missing pooled real candidate metric")
    q90 = float(pooled_real["q90_risk_threshold"].iloc[0])
    rows: list[dict[str, Any]] = []
    real_segments = candidate_segments.loc[
        candidate_segments["variant"] == "real_gr"
    ]
    for well, frame in real_segments.groupby("well", sort=True):
        bad = frame["bad_row_count"].to_numpy(float)
        good = frame["good_row_count"].to_numpy(float)
        risk = frame["risk"].to_numpy(float)
        high = np.isfinite(risk) & (risk >= q90 - EPS)
        summary = summarize_risk_frame(
            pd.DataFrame(
                {"risk": risk, "bad_weight": bad, "good_weight": good}
            )
        )
        row = {
            "well": str(well),
            "segment_candidate_count": len(frame),
            "pooled_real_auc": summary["auc"],
            "global_q90_risk_threshold": q90,
            "good_candidate_false_alert_rate": (
                float(good[high].sum() / good.sum())
                if good.sum() > 0
                else math.nan
            ),
            "bad_candidate_high_risk_recall": (
                float(bad[high].sum() / bad.sum())
                if bad.sum() > 0
                else math.nan
            ),
        }
        truth_well = truth_contributions.loc[
            (truth_contributions["well"] == well)
            & (truth_contributions["group_type"] == "overall")
        ]
        for variant in ["real_gr", "shuffled_typewell_gr"]:
            variant_frame = truth_well.loc[truth_well["variant"] == variant]
            supported = float(variant_frame["supported_weight"].sum())
            inside = float(variant_frame["inside_weight"].sum())
            row[f"truth_corridor_coverage_{variant}"] = (
                inside / supported if supported else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("well", kind="mergesort")


def metric_row(
    frame: pd.DataFrame,
    **conditions: str,
) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).to_numpy() == str(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one metric row for {conditions}; got {len(selected)}")
    return selected.iloc[0]


def evaluate_stage1_guards(
    candidate_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    limits = nested(config, "audit.guards")
    pooled_real = metric_row(
        candidate_metrics,
        variant="real_gr",
        candidate="__pooled__",
    )
    pooled_shuffled = metric_row(
        candidate_metrics,
        variant="shuffled_typewell_gr",
        candidate="__pooled__",
    )
    guards: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, requirement: str) -> None:
        guards.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": to_jsonable(observed),
                "requirement": requirement,
            }
        )

    pooled_auc = float(pooled_real["auc"])
    add(
        "pooled_real_auc",
        np.isfinite(pooled_auc)
        and pooled_auc >= float(limits["min_pooled_real_auc"]),
        pooled_auc,
        f">={float(limits['min_pooled_real_auc']):.2f}",
    )
    auc_delta = pooled_auc - float(pooled_shuffled["auc"])
    add(
        "pooled_real_minus_shuffled_auc",
        np.isfinite(auc_delta)
        and auc_delta >= float(limits["min_real_minus_shuffled_auc"]),
        auc_delta,
        f">={float(limits['min_real_minus_shuffled_auc']):.2f}",
    )
    family_lifts: dict[str, dict[str, float]] = {}
    for candidate in ["likpf_mean", "pf_ancc"]:
        real = metric_row(
            candidate_metrics,
            variant="real_gr",
            candidate=candidate,
        )
        shuffled = metric_row(
            candidate_metrics,
            variant="shuffled_typewell_gr",
            candidate=candidate,
        )
        real_lift = float(real["q90_bad_rate_lift"])
        shuffled_lift = float(shuffled["q90_bad_rate_lift"])
        family_lifts[candidate] = {
            "real": real_lift,
            "shuffled": shuffled_lift,
            "difference": real_lift - shuffled_lift,
        }
    lift_pass = (
        max(value["real"] for value in family_lifts.values())
        >= float(limits["min_family_q90_lift"])
        and all(
            value["difference"]
            >= float(limits["min_family_real_minus_shuffled_lift"])
            for value in family_lifts.values()
        )
    )
    add(
        "likpf_or_pfancc_q90_lift_and_control_delta",
        lift_pass,
        family_lifts,
        (
            f"one real lift>={float(limits['min_family_q90_lift']):.2f}; "
            f"both real-shuffled>="
            f"{float(limits['min_family_real_minus_shuffled_lift']):.2f}"
        ),
    )
    family_false_alert = {
        str(row["candidate"]): float(row["q90_good_false_alert_rate"])
        for _, row in candidate_metrics.loc[
            (candidate_metrics["variant"] == "real_gr")
            & (candidate_metrics["candidate"] != "__pooled__")
        ].iterrows()
    }
    overall_false_alert = float(pooled_real["q90_good_false_alert_rate"])
    add(
        "q90_good_candidate_false_alert",
        (
            overall_false_alert
            <= float(limits["max_overall_good_false_alert"])
            and all(
                value <= float(limits["max_family_good_false_alert"])
                for value in family_false_alert.values()
            )
        ),
        {
            "overall": overall_false_alert,
            "families": family_false_alert,
        },
        (
            f"overall<={float(limits['max_overall_good_false_alert']):.2f}; "
            f"each family<={float(limits['max_family_good_false_alert']):.2f}"
        ),
    )
    overlap_real = metric_row(
        overlap_metrics,
        scope="overall",
        variant="real_gr",
    )
    overlap_values = {
        "path_median_ft": float(
            overlap_real["path_tvt_difference_median_ft"]
        ),
        "path_p90_ft": float(overlap_real["path_tvt_difference_p90_ft"]),
        "risk_spearman": float(overlap_real["candidate_risk_spearman"]),
    }
    add(
        "overlap_path_and_risk_agreement",
        (
            overlap_values["path_median_ft"]
            <= float(limits["max_overlap_path_tvt_median_ft"])
            and overlap_values["path_p90_ft"]
            <= float(limits["max_overlap_path_tvt_p90_ft"])
            and overlap_values["risk_spearman"]
            >= float(limits["min_overlap_risk_spearman"])
        ),
        overlap_values,
        (
            f"median<={float(limits['max_overlap_path_tvt_median_ft']):.1f}, "
            f"p90<={float(limits['max_overlap_path_tvt_p90_ft']):.1f}, "
            f"Spearman>={float(limits['min_overlap_risk_spearman']):.2f}"
        ),
    )
    hidden_auc: dict[str, float] = {}
    hidden_pass = True
    for group_name in [
        "verification_like_spatial",
        "verification_like_typewell_purged",
    ]:
        row = metric_row(
            group_metrics,
            entity="candidate",
            variant="real_gr",
            candidate="__pooled__",
            group_type="hidden_like",
            group_name=group_name,
        )
        auc = float(row["auc"])
        hidden_auc[group_name] = auc
        hidden_pass = hidden_pass and (
            auc >= float(limits["min_hidden_like_auc"])
            and pooled_auc - auc <= float(limits["max_hidden_like_auc_drop"])
        )
    add(
        "hidden_like_auc",
        hidden_pass,
        {"overall": pooled_auc, **hidden_auc},
        (
            f"each>={float(limits['min_hidden_like_auc']):.2f} and "
            f"overall drop<={float(limits['max_hidden_like_auc_drop']):.2f}"
        ),
    )
    well_false_alert = by_well["good_candidate_false_alert_rate"].dropna()
    well_p95 = (
        float(well_false_alert.quantile(0.95))
        if len(well_false_alert)
        else math.nan
    )
    well_max = float(well_false_alert.max()) if len(well_false_alert) else math.nan
    add(
        "by_well_good_candidate_false_alert",
        (
            np.isfinite(well_p95)
            and np.isfinite(well_max)
            and well_p95
            <= float(limits["max_by_well_good_false_alert_p95"])
            and well_max <= float(limits["max_by_well_good_false_alert"])
        ),
        {"p95": well_p95, "max": well_max},
        (
            f"p95<={float(limits['max_by_well_good_false_alert_p95']):.2f}; "
            f"max<={float(limits['max_by_well_good_false_alert']):.2f}"
        ),
    )
    truth_differences: dict[str, float] = {}
    truth_pass = True
    truth_groups = [
        ("overall", "overall"),
        ("distance", "1000_plus"),
        ("hidden_like", "verification_like_spatial"),
        ("hidden_like", "verification_like_typewell_purged"),
    ]
    for group_type, group_name in truth_groups:
        real = metric_row(
            group_metrics,
            entity="truth",
            variant="real_gr",
            candidate="truth",
            group_type=group_type,
            group_name=group_name,
        )
        shuffled = metric_row(
            group_metrics,
            entity="truth",
            variant="shuffled_typewell_gr",
            candidate="truth",
            group_type=group_type,
            group_name=group_name,
        )
        difference = float(real["truth_corridor_coverage"]) - float(
            shuffled["truth_corridor_coverage"]
        )
        key = f"{group_type}:{group_name}"
        truth_differences[key] = difference
        required = (
            float(limits["min_truth_coverage_real_minus_shuffled"])
            if group_type == "overall"
            else 0.0
        )
        truth_pass = truth_pass and np.isfinite(difference) and difference >= required
    add(
        "truth_corridor_coverage_real_vs_shuffled",
        truth_pass,
        truth_differences,
        (
            f"overall>="
            f"{float(limits['min_truth_coverage_real_minus_shuffled']):.2f}; "
            "1000_plus and both hidden-like groups >=0"
        ),
    )
    decision = (
        "pass_add_only_feature_experiment_allowed"
        if all(guard["passed"] for guard in guards)
        else "fail_close_segment_local_hard_use_and_grid_search"
    )
    return guards, decision


def run_stage1(
    cache_path: Path,
    inventory: dict[str, RawInventoryRecord],
    specs: list[CandidateSpec],
    hidden_groups: dict[str, set[str]],
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not bool(nested(config, "audit.stage0.manual_parity_confirmed")):
        raise RuntimeError("Stage 1 requires Stage 0 manual parity confirmation")
    if not bool(
        nested(config, "audit.stage1.enabled_after_stage0_confirmation")
    ):
        raise RuntimeError("Stage 1 enabled_after_stage0_confirmation is false")
    chunksize = int(nested(config, "audit.cache_chunksize"))
    bad_threshold = float(nested(config, "audit.bad_candidate_threshold_ft"))
    segment_path = out_dir / str(
        nested(config, "audit.outputs.segment_metrics_filename")
    )
    candidate_segment_path = out_dir / str(
        nested(config, "audit.outputs.candidate_segment_metrics_filename")
    )
    candidate_rows_all: list[dict[str, Any]] = []
    contribution_rows_all: list[dict[str, Any]] = []
    overlap_pair_rows: list[dict[str, Any]] = []
    raw_overlap_global = {
        variant: {"path": [], "left_risk": [], "right_risk": []}
        for variant in ["real_gr", "shuffled_typewell_gr"]
    }
    processed_wells = 0
    processed_segments = 0
    with (
        DeterministicGzipCsvWriter(segment_path) as segment_writer,
        DeterministicGzipCsvWriter(candidate_segment_path) as candidate_writer,
    ):
        for well, cache in iter_candidate_cache_wells(
            cache_path,
            specs,
            chunksize,
        ):
            raw = load_raw_well(inventory[well], cache, specs, config)
            segments = build_all_segments(raw, config)
            well_segment_rows: list[dict[str, Any]] = []
            well_candidate_rows: list[dict[str, Any]] = []
            bundles_by_variant: dict[str, list[SegmentBundle]] = {
                "real_gr": [],
                "shuffled_typewell_gr": [],
            }
            truth_tail = raw.truth_tvt[raw.evaluation_start :]
            for segment in segments:
                analyses = {
                    variant: analyze_segment_surface(raw, segment, variant, config)
                    for variant in ["real_gr", "shuffled_typewell_gr"]
                }
                real = analyses["real_gr"]
                shuffled = analyses["shuffled_typewell_gr"]
                if not np.array_equal(real.support, shuffled.support):
                    raise AssertionError(f"{well} real/shuffled support mismatch")
                if not np.array_equal(
                    real.primary.source_states,
                    shuffled.primary.source_states,
                ):
                    raise AssertionError(f"{well} real/shuffled source mismatch")
                segment_candidate_rows: dict[
                    str,
                    dict[str, dict[str, Any]],
                ] = {}
                for variant, analysis in analyses.items():
                    candidate_readouts = {
                        spec.name: build_path_readout(
                            raw,
                            segment,
                            analysis,
                            raw.candidates[spec.name],
                            truth_values=raw.truth_tvt,
                            bad_threshold_ft=bad_threshold,
                        )
                        for spec in specs
                    }
                    truth_readout = build_path_readout(
                        raw,
                        segment,
                        analysis,
                        truth_tail,
                        truth_values=None,
                        bad_threshold_ft=bad_threshold,
                    )
                    segment_row = segment_metric_row(raw, segment, analysis)
                    segment_row.update(
                        {
                            "truth_endpoint_supported_fraction": truth_readout.summary[
                                "endpoint_supported_fraction"
                            ],
                            "truth_corridor_coverage": truth_readout.summary[
                                "corridor_inside_fraction"
                            ],
                            "truth_corridor_outside_fraction": truth_readout.summary[
                                "corridor_outside_fraction"
                            ],
                            "truth_corridor_crossing_count": truth_readout.summary[
                                "corridor_crossing_count"
                            ],
                        }
                    )
                    well_segment_rows.append(segment_row)
                    segment_candidate_rows[variant] = {}
                    for spec in specs:
                        readout = candidate_readouts[spec.name]
                        candidate_row = candidate_segment_row(
                            raw,
                            segment,
                            analysis,
                            spec.name,
                            readout,
                        )
                        segment_candidate_rows[variant][spec.name] = candidate_row
                        contribution_rows_all.extend(
                            risk_contribution_rows(
                                raw,
                                segment,
                                analysis,
                                spec.name,
                                readout,
                                hidden_groups,
                                config,
                            )
                        )
                    contribution_rows_all.extend(
                        truth_contribution_rows(
                            raw,
                            segment,
                            analysis,
                            truth_readout,
                            hidden_groups,
                            config,
                        )
                    )
                    bundles_by_variant[variant].append(
                        SegmentBundle(
                            segment=segment,
                            analysis=analysis,
                            candidate_readouts=candidate_readouts,
                            truth_readout=truth_readout,
                        )
                    )
                for spec in specs:
                    real_row = segment_candidate_rows["real_gr"][spec.name]
                    shuffled_row = segment_candidate_rows[
                        "shuffled_typewell_gr"
                    ][spec.name]
                    paired_metric_names = [
                        "risk",
                        "mean_excess_cost",
                        "candidate_bottleneck_minus_tau",
                        "corridor_distance_median_ft",
                        "corridor_distance_p90_ft",
                        "corridor_crossing_count",
                        "ridge_event_count",
                    ]
                    for row in [real_row, shuffled_row]:
                        for metric_name in paired_metric_names:
                            real_value = float(real_row[metric_name])
                            shuffled_value = float(shuffled_row[metric_name])
                            row[f"real_minus_shuffled_{metric_name}"] = (
                                real_value - shuffled_value
                                if np.isfinite(real_value)
                                and np.isfinite(shuffled_value)
                                else math.nan
                            )
                        well_candidate_rows.append(row)
                processed_segments += 1
            overlap_rows, overlap_raw = build_overlap_metrics(
                bundles_by_variant,
                [spec.name for spec in specs],
                well,
            )
            overlap_pair_rows.extend(overlap_rows)
            for variant in raw_overlap_global:
                for key in raw_overlap_global[variant]:
                    raw_overlap_global[variant][key].extend(
                        overlap_raw[variant][key]
                    )
            segment_writer.write(
                pd.DataFrame(well_segment_rows).sort_values(
                    ["well", "segment_id", "variant"],
                    kind="mergesort",
                )
            )
            candidate_writer.write(
                pd.DataFrame(well_candidate_rows).sort_values(
                    ["well", "segment_id", "candidate", "variant"],
                    kind="mergesort",
                )
            )
            candidate_rows_all.extend(well_candidate_rows)
            processed_wells += 1
            if processed_wells % 25 == 0:
                print(
                    f"stage1 progress wells={processed_wells} "
                    f"segments={processed_segments}"
                )
    candidate_segments = pd.DataFrame(candidate_rows_all)
    contributions = pd.DataFrame(contribution_rows_all)
    candidate_metrics = finalize_candidate_metrics(candidate_segments)
    group_metrics = finalize_group_metrics(contributions)
    overlap_metrics = finalize_overlap_metrics(
        overlap_pair_rows,
        raw_overlap_global,
    )
    truth_contributions = contributions.loc[
        contributions["entity"] == "truth"
    ]
    by_well = build_by_well_metrics(
        candidate_segments,
        candidate_metrics,
        truth_contributions,
    )
    guards, decision = evaluate_stage1_guards(
        candidate_metrics,
        group_metrics,
        overlap_metrics,
        by_well,
        config,
    )
    output_frames = {
        "candidate_metrics": (
            candidate_metrics,
            out_dir
            / str(nested(config, "audit.outputs.candidate_metrics_filename")),
        ),
        "group_metrics": (
            group_metrics,
            out_dir / str(nested(config, "audit.outputs.group_metrics_filename")),
        ),
        "overlap_metrics": (
            overlap_metrics,
            out_dir
            / str(nested(config, "audit.outputs.overlap_metrics_filename")),
        ),
        "by_well": (
            by_well,
            out_dir / str(nested(config, "audit.outputs.by_well_filename")),
        ),
    }
    generated_paths = [segment_path, candidate_segment_path]
    for _, (frame, path) in output_frames.items():
        frame.to_csv(path, index=False, lineterminator="\n")
        generated_paths.append(path)
    print("Stage 1 candidate metrics:")
    display(candidate_metrics)
    print("Stage 1 guards:")
    display(pd.DataFrame(guards))
    return {
        "status": "stage1_complete",
        "decision": decision,
        "guards": guards,
        "well_count": processed_wells,
        "segment_count": processed_segments,
        "candidate_segment_count": len(candidate_segments),
        "generated_paths": generated_paths,
        "primary_metrics": to_jsonable(
            metric_row(
                candidate_metrics,
                variant="real_gr",
                candidate="__pooled__",
            ).to_dict()
        ),
    }


# %% [markdown]
# ## 9. Synthetic DAG / DP contract

# %%
def run_synthetic_contract_checks(config: dict[str, Any]) -> None:
    state_grid = np.arange(5, dtype=float) * 4.0

    # Every generated edge moves strictly right; no left or same-column cycle exists.
    horizontal = np.array([True, True, True])
    for x_index in range(len(horizontal)):
        for y_index in range(len(state_grid)):
            for next_x, _, _ in successor_nodes(
                x_index,
                y_index,
                horizontal,
                len(state_grid),
            ):
                assert next_x > x_index

    # A component that cannot be traversed left-to-right must not become a path.
    cost = np.ones((3, 5), dtype=float)
    support = np.zeros_like(cost, dtype=bool)
    support[0, 0] = True
    support[1, 1] = True
    support[2, 4] = True
    core = solve_minimum_bottleneck_dp(
        cost,
        support,
        horizontal,
        np.ones(5, dtype=bool),
    )
    assert not core.exists

    # A broken component with two unsupported columns cannot be bridged.
    horizontal_two_gap = np.array([True, False, False, True])
    support = np.zeros((4, 5), dtype=bool)
    support[0, 2] = True
    support[3, 2] = True
    core = solve_minimum_bottleneck_dp(
        np.ones((4, 5), dtype=float),
        support,
        horizontal_two_gap,
        np.array([False, False, True, False, False]),
    )
    assert not core.exists

    # One unsupported column is bridged once with dy in [-2, 2].
    horizontal_one_gap = np.array([True, False, True])
    support = np.zeros((3, 5), dtype=bool)
    support[0, 1] = True
    support[2, 3] = True
    core = solve_minimum_bottleneck_dp(
        np.ones((3, 5), dtype=float),
        support,
        horizontal_one_gap,
        np.array([False, True, False, False, False]),
    )
    assert core.exists and core.gap_edge_used
    assert np.array_equal(core.path_x, np.array([0, 2]))

    # An anchor-outside component is available to spanning but not anchored DP.
    horizontal = np.array([True, True, True])
    support = np.zeros((3, 5), dtype=bool)
    support[:, 4] = True
    support[0, 0] = True
    unanchored = solve_minimum_bottleneck_dp(
        np.ones((3, 5), dtype=float),
        support,
        horizontal,
        support[0],
    )
    anchored = solve_minimum_bottleneck_dp(
        np.ones((3, 5), dtype=float),
        support,
        horizontal,
        np.array([True, False, False, False, False]),
    )
    assert unanchored.exists and not anchored.exists

    # Equal bottleneck paths are resolved by cumulative node cost.
    support = np.zeros((3, 5), dtype=bool)
    support[:, 0] = True
    support[:, 2] = True
    cost = np.full((3, 5), np.inf, dtype=float)
    cost[:, 0] = [2.0, 0.0, 0.0]
    cost[:, 2] = [2.0, 2.0, 2.0]
    sources = np.array([True, False, True, False, False])
    core = solve_minimum_bottleneck_dp(cost, support, horizontal, sources)
    assert core.exists
    assert np.array_equal(core.path_y, np.array([0, 0, 0]))
    corridor = build_near_optimal_corridor(
        cost,
        support,
        horizontal,
        sources,
        core.tau_star,
        0.25,
    )
    assert corridor[0, 0] and corridor[-1, 0]

    # Segment construction is MD-based, right-aligns the final full window,
    # and produces one variable-width segment for short tails.
    starts = segment_start_values(100.0, 700.0, config)
    assert np.allclose(starts, [100.0, 228.0, 356.0, 444.0])
    assert segment_start_values(100.0, 300.0, config) == [100.0]

    # Stable shuffled control remains nonzero and inside the fixed 25%-75% range.
    shift_a = stable_circular_shift(100, "synthetic", config)
    shift_b = stable_circular_shift(100, "synthetic", config)
    assert shift_a == shift_b and 25 <= shift_a <= 75
    print("Synthetic DAG/DP contract: PASS")


def validate_static_contract(config: dict[str, Any]) -> None:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("Experiment name mismatch")
    if nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp250 route must be pf_beam")
    if list(nested(config, "model.active_variants")) != [
        "real_gr",
        "shuffled_typewell_gr",
    ]:
        raise ValueError("exp250 requires exactly two diagnostic surfaces")
    count_keys = [
        "model.lightgbm_config_count",
        "model.cnn_config_count",
        "model.hmm_config_count",
        "model.fold_training_count",
        "model.booster_count",
    ]
    if any(int(nested(config, key)) != 0 for key in count_keys):
        raise ValueError("exp250 must train zero configs/folds/boosters")
    if bool(nested(config, "model.parent_control_retraining")):
        raise ValueError("exp250 must not retrain parent/control")
    if bool(nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("exp250 must run CPU-only")
    if bool(nested(config, "runtime.kaggle.enable_internet")):
        raise ValueError("exp250 must run with internet disabled")
    if int(nested(config, "runtime.num_workers")) != 1:
        raise ValueError("exp250 must use one worker")
    if bool(nested(config, "inference.enabled")) or bool(
        nested(config, "inference.create_submission")
    ):
        raise ValueError("exp250 inference/submission must remain disabled")
    fixed_values = {
        "audit.segment.length_ft": 256.0,
        "audit.segment.stride_ft": 128.0,
        "audit.segment.horizontal_bin_ft": 4.0,
        "audit.typewell_grid.half_width_ft": 256.0,
        "audit.typewell_grid.state_step_ft": 4.0,
        "audit.typewell_grid.state_count": 129,
        "audit.graph.anchor_radius_ft": 8.0,
        "audit.graph.corridor_slack_cost": 0.25,
    }
    for key, expected in fixed_values.items():
        if float(nested(config, key)) != float(expected):
            raise ValueError(f"Fixed scientific contract mismatch: {key}")
    active_mode = str(nested(config, "audit.active_mode"))
    if active_mode not in set(nested(config, "audit.allowed_modes")):
        raise ValueError(f"Unknown active mode: {active_mode}")


def output_sha_records(paths: list[Path], base_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path in sorted(set(paths), key=lambda value: str(value)):
        key = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else path.name
        record = {
            "bytes": path.stat().st_size,
            "raw_sha256": sha256_path(path),
        }
        if path.suffix == ".gz":
            record["decompressed_content_sha256"] = sha256_path(
                path,
                decompressed=True,
            )
        records[key] = record
    return records


# %% [markdown]
# ## 10. Setup, input preflight, and execution

# %%
CONFIG = load_config()
validate_static_contract(CONFIG)
run_synthetic_contract_checks(CONFIG)
require_authoritative_runtime()

ACTIVE_MODE = str(nested(CONFIG, "audit.active_mode"))
OUT_DIR = output_dir()
SPECS = candidate_specs(CONFIG)
TRAIN_DIR = resolve_train_dir(CONFIG)
CACHE_PATH = resolve_file(
    CONFIG,
    paths_key="data.candidate_cache_paths",
    filename_key="data.candidate_cache_filename",
)
HIDDEN_LIKE_PATH = resolve_file(
    CONFIG,
    paths_key="data.hidden_like_paths",
    filename_key="data.hidden_like_filename",
)

print("Experiment:", EXPERIMENT_NAME)
print("Route:", nested(CONFIG, "experiment.route"))
print("Active mode:", ACTIVE_MODE)
print("Diagnostic surfaces:", nested(CONFIG, "model.active_variants"))
print(
    "Cost contract:",
    {
        "LightGBM/CNN/HMM configs": 0,
        "folds": 0,
        "boosters": 0,
        "PF/Beam regeneration": 0,
        "parent/control retraining": False,
    },
)
print("Candidate cache:", CACHE_PATH)
print("Raw train:", TRAIN_DIR)

INVENTORY, RAW_INVENTORY_SHA = build_raw_inventory(TRAIN_DIR, CONFIG)
SELECTED_WELLS, STAGE0_SELECTION_ROLES = select_stage0_wells(INVENTORY, CONFIG)
CACHE_RAW_SHA = sha256_path(CACHE_PATH)
CACHE_DECOMPRESSED_SHA = sha256_path(CACHE_PATH, decompressed=True)
INPUT_SHA = {
    "config_sha256": sha256_path(CONFIG_PATH),
    "raw_inventory_sha256": RAW_INVENTORY_SHA,
    "candidate_cache_raw_sha256": CACHE_RAW_SHA,
    "candidate_cache_decompressed_content_sha256": CACHE_DECOMPRESSED_SHA,
    "hidden_like_assignment_sha256": sha256_path(HIDDEN_LIKE_PATH),
}
PREFLIGHT = preflight_candidate_cache(
    CACHE_PATH,
    SPECS,
    int(nested(CONFIG, "audit.cache_chunksize")),
    INVENTORY,
    set(SELECTED_WELLS) if ACTIVE_MODE == "stage0_preview" else set(),
)
if PREFLIGHT.well_count != int(nested(CONFIG, "audit.expected_well_count")):
    raise ValueError("Candidate-cache well count does not match the fixed contract")
if PREFLIGHT.row_count != int(
    nested(CONFIG, "audit.expected_candidate_row_count")
):
    raise ValueError("Candidate-cache row count does not match the fixed contract")
print(
    "Input preflight PASS:",
    {
        "raw_wells": len(INVENTORY),
        "candidate_wells": PREFLIGHT.well_count,
        "candidate_rows": PREFLIGHT.row_count,
        "cache_decompressed_sha": CACHE_DECOMPRESSED_SHA,
    },
)

# %% [markdown]
# ## 11. Metrics, SHA, and generated files

# %%
STARTED_AT = time.time()
if ACTIVE_MODE == "stage0_preview":
    RUN_RESULT = run_stage0(
        SELECTED_WELLS,
        STAGE0_SELECTION_ROLES,
        PREFLIGHT.selected_frames,
        INVENTORY,
        SPECS,
        OUT_DIR,
        CONFIG,
        INPUT_SHA,
    )
elif ACTIVE_MODE == "stage1_full_audit":
    HIDDEN_GROUPS = load_hidden_like_roles(HIDDEN_LIKE_PATH)
    RUN_RESULT = run_stage1(
        CACHE_PATH,
        INVENTORY,
        SPECS,
        HIDDEN_GROUPS,
        OUT_DIR,
        CONFIG,
    )
else:
    raise ValueError(f"Unsupported active mode: {ACTIVE_MODE}")

RUNTIME_SECONDS = time.time() - STARTED_AT
OUTPUT_SHA = output_sha_records(RUN_RESULT["generated_paths"], OUT_DIR)
SUMMARY = {
    "experiment": EXPERIMENT_NAME,
    "route": "pf_beam",
    "active_mode": ACTIVE_MODE,
    "status": RUN_RESULT["status"],
    "decision": RUN_RESULT.get(
        "decision",
        "stage0_manual_parity_required_before_stage1",
    ),
    "cost_contract": {
        "diagnostic_surfaces": 2,
        "lightgbm_cnn_hmm_configs": 0,
        "fold_training": 0,
        "boosters": 0,
        "pf_beam_likpf_regeneration": 0,
        "parent_control_retraining": False,
    },
    "fixed_scientific_contract": {
        "segment_length_ft": 256.0,
        "segment_stride_ft": 128.0,
        "horizontal_bin_ft": 4.0,
        "typewell_half_width_ft": 256.0,
        "typewell_state_step_ft": 4.0,
        "typewell_state_count": 129,
        "anchor_radius_ft": 8.0,
        "corridor_slack_cost": 0.25,
    },
    "raw_well_count": len(INVENTORY),
    "candidate_cache_well_count": PREFLIGHT.well_count,
    "candidate_cache_row_count": PREFLIGHT.row_count,
    "runtime_seconds": RUNTIME_SECONDS,
    "runtime": {
        "platform": "kaggle" if is_kaggle_runtime() else "local_debug",
        "cpu_only": True,
        "internet_disabled": True,
        "single_process": True,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    },
    "reproducibility": {
        "deterministic_prediction_or_submission_anchor": False,
        "audit_deterministic_for_fixed_inputs": True,
        "upstream_stochastic_provenance": [
            "exp072 saved PF/Beam candidate cache"
        ],
        "new_rng": False,
        "shuffled_control_seed_policy": "stable_sha_seed_42_25_to_75_percent_shift",
        "gzip_mtime": 0,
    },
    "input_sha": INPUT_SHA,
    "output_sha": OUTPUT_SHA,
    "guards": RUN_RESULT.get("guards", "not_evaluated_in_stage0"),
    "primary_metrics": RUN_RESULT.get("primary_metrics"),
}
SUMMARY_PATH = OUT_DIR / str(
    nested(CONFIG, "audit.outputs.summary_filename")
)
write_json(SUMMARY_PATH, SUMMARY)

METRICS = {
    "experiment": EXPERIMENT_NAME,
    "status": RUN_RESULT["status"],
    "route": "pf_beam",
    "metric": "segment_local_minimum_bottleneck_corridor_audit",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "active_mode": ACTIVE_MODE,
    "runtime_seconds": RUNTIME_SECONDS,
    "input_sha": INPUT_SHA,
    "summary_path": str(SUMMARY_PATH),
    "summary_sha256": sha256_path(SUMMARY_PATH),
    "decision": SUMMARY["decision"],
    "guards": SUMMARY["guards"],
    "primary_metrics": SUMMARY["primary_metrics"],
}
write_json(metrics_output_path(), METRICS)
print("Run summary:")
display(pd.DataFrame([{
    "status": METRICS["status"],
    "active_mode": ACTIVE_MODE,
    "runtime_seconds": RUNTIME_SECONDS,
    "decision": METRICS["decision"],
}]))
print("Summary:", SUMMARY_PATH)
print("Metrics:", metrics_output_path())
