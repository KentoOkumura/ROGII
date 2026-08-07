# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp301 gauge-invariant multiformation edge potential
#
# This train-side notebook reconstructs one two-dimensional scalar potential
# from within-well differences of six formation surfaces. Outer-valid wells are
# opened through a geometry-only loader. Their formation, GR, and true TVT
# columns remain inaccessible until the complete OOF prediction is frozen.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Fold-safe raw-data and comparison-input loaders
# 4. Multiformation edge identity and active-grid helpers
# 5. Bilinear constraints and deterministic sparse Huber solver
# 6. Stage 0 support audit and conditional Stage 1 OOF generation
# 7. Prediction freeze, late truth join, and direct-quality readouts
# 8. Exp293 fixed-bank H512 add-one novelty diagnostic
# 9. Generated artifacts, promotion decision, and execution

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import yaml
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

EXPERIMENT_NAME = "exp301_gauge_invariant_multiformation_edge_potential"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
QUERY_RAW_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
QUERY_SAFE_COLUMNS = ("well_id", "row_index", *QUERY_RAW_COLUMNS)
SOURCE_RAW_COLUMNS = (*QUERY_RAW_COLUMNS, *FORMATION_COLUMNS, "TVT")
FORBIDDEN_QUERY_COLUMNS = {
    "TVT",
    "GR",
    *FORMATION_COLUMNS,
    "target",
    "target_error",
    "error",
    "abs_error",
    "oracle",
    "oracle_rank",
}
EXP263_KEY_COLUMNS = ("id", "well", "well_row_idx", "outer_fold", "md_since")
EXP263_READ_COLUMNS = (
    *EXP263_KEY_COLUMNS,
    "candidate_tvt",
    "candidate_available",
    "candidate_finite",
)


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers

# %%
def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP301_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    nested = project_root() / "experiments" / EXPERIMENT_NAME
    return nested if nested.exists() else Path.cwd()


def find_config_path() -> Path:
    candidates = [Path.cwd() / "config.yaml", experiment_dir() / "config.yaml"]
    found = [path for path in candidates if path.exists()]
    if found:
        return found[0]
    matches = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp301 config.yaml was not found unambiguously")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def runtime_artifacts_dir() -> Path:
    root = KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else experiment_dir()
    path = root / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_work_dir() -> Path:
    root = KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else experiment_dir()
    path = root / ".exp301_work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    root = KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else experiment_dir()
    return root / "metrics.json"


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def array_content_sha256(*arrays: np.ndarray, context: Sequence[str] = ()) -> str:
    digest = hashlib.sha256()
    for item in context:
        digest.update(str(item).encode())
        digest.update(b"\0")
    for raw in arrays:
        array = np.ascontiguousarray(raw)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_object_dtype(output[column]):
            output[column] = output[column].astype("string")
    return output


def frame_content_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(normalized.columns), separators=(",", ":")).encode())
    digest.update(
        json.dumps([str(value) for value in normalized.dtypes], separators=(",", ":")).encode()
    )
    row_hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return json_sha256(schema)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    root = project_root()
    found: dict[str, Path] = {}
    for raw_pattern in patterns:
        raw = str(raw_pattern)
        direct = Path(raw) if Path(raw).is_absolute() else root / raw
        if direct.exists():
            found.setdefault(str(direct.resolve()), direct)
        for match in glob.glob(raw, recursive=True):
            path = Path(match)
            if path.exists():
                found.setdefault(str(path.resolve()), path)
        if not Path(raw).is_absolute():
            for match in glob.glob(str(root / raw), recursive=True):
                path = Path(match)
                if path.exists():
                    found.setdefault(str(path.resolve()), path)
    return list(found.values())


def resolve_file(
    patterns: Sequence[str], *, label: str, expected_sha256: str | None = None
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    if expected_sha256:
        matching = [path for path in candidates if sha256_file(path) == expected_sha256]
        if matching:
            return sorted(matching, key=lambda path: (len(str(path)), str(path)))[0]
        if candidates:
            evidence = {str(path): sha256_file(path) for path in candidates}
            raise ValueError(f"{label} SHA mismatch: {evidence}")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"{label} was not found: {list(patterns)}")
    raise ValueError(f"{label} resolved to multiple files: {candidates}")


def resolve_raw_directory(
    patterns: Sequence[str], *, expected_wells: set[str] | None = None
) -> tuple[Path, dict[str, Path]]:
    evidence: dict[str, dict[str, int]] = {}
    for directory in expand_existing_paths(patterns):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*__horizontal_well.csv"))
        mapping = {well_id_from_path(path): path for path in files}
        evidence[str(directory)] = {"files": len(files), "wells": len(mapping)}
        if len(mapping) != len(files):
            continue
        if expected_wells is None or set(mapping) == expected_wells:
            return directory, mapping
    raise FileNotFoundError(f"raw horizontal directory not found: {evidence}")


def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def validate_execution_contract(config: Mapping[str, Any]) -> None:
    execution = get_nested(config, "execution")
    if not bool(execution["implementation_authorized"]):
        raise ValueError("exp301 implementation is not authorized")
    expected = {
        "active_scientific_variants_if_implemented": 1,
        "outer_evaluation_folds_if_implemented": 5,
        "inner_lambda_candidates_per_outer_fold": 3,
        "lightgbm_config_count": 0,
        "trained_fold_count": 0,
        "total_boosters": 0,
        "control_or_parent_retraining": False,
        "gpu": False,
        "inference": False,
        "submission": False,
    }
    mismatches = {
        key: execution.get(key)
        for key, value in expected.items()
        if execution.get(key) != value
    }
    if mismatches:
        raise ValueError(f"execution contract mismatch: {mismatches}")
    if bool(execution["kaggle_execution_authorized"]):
        print("Kaggle execution is explicitly authorized in config.")
    else:
        print("Implementation-only state: Kaggle execution remains disabled.")


# %% [markdown]
# ## 3. Fold-safe raw-data and comparison-input loaders

# %%
@dataclass(frozen=True)
class FoldIdentity:
    by_well: dict[str, int]
    path: Path
    manifest: dict[str, Any]


def _stable_numeric_frame(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(columns))
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame = frame[list(columns)].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    frame = frame.sort_values(["MD", "row_index"], kind="mergesort").reset_index(drop=True)
    if not np.isfinite(frame[["MD", "X", "Y", "Z"]].to_numpy(np.float64)).all():
        raise ValueError(f"non-finite geometry in {path}")
    return frame


def load_source_horizontal(path: Path) -> pd.DataFrame:
    frame = _stable_numeric_frame(path, SOURCE_RAW_COLUMNS)
    if not np.isfinite(frame["TVT"].to_numpy(np.float64)).all():
        raise ValueError(f"outer-train TVT is non-finite: {path}")
    return frame


def load_query_safe_horizontal(
    path: Path, requested_columns: Sequence[str] = QUERY_RAW_COLUMNS
) -> pd.DataFrame:
    requested = tuple(str(column) for column in requested_columns)
    forbidden = set(requested) & FORBIDDEN_QUERY_COLUMNS
    if forbidden:
        raise ValueError(f"query loader rejected forbidden columns: {sorted(forbidden)}")
    if requested != QUERY_RAW_COLUMNS:
        raise ValueError(f"query loader requires exact raw allowlist: {QUERY_RAW_COLUMNS}")
    frame = _stable_numeric_frame(path, requested)
    frame.insert(0, "well_id", well_id_from_path(path))
    if tuple(frame.columns) != QUERY_SAFE_COLUMNS:
        raise ValueError(f"query-safe schema mismatch: {tuple(frame.columns)}")
    if FORBIDDEN_QUERY_COLUMNS.intersection(frame.columns):
        raise ValueError("query-safe frame contains forbidden columns")
    values = frame["TVT_input"].to_numpy(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"query well has no known TVT_input prefix: {path}")
    anchor = int(np.flatnonzero(finite)[-1])
    if not finite[: anchor + 1].all() or finite[anchor + 1 :].any():
        raise ValueError(f"TVT_input is not one contiguous prefix: {path}")
    return frame


def safe_query_content_sha256(frame: pd.DataFrame) -> str:
    if tuple(frame.columns) != QUERY_SAFE_COLUMNS:
        raise ValueError("only exact query-safe frames may be hashed")
    return frame_content_sha256(frame)


def load_fold_identity(config: Mapping[str, Any]) -> FoldIdentity:
    spec = get_nested(config, "data.exp226_oof")
    path = resolve_file(spec["patterns"], label="exp226 OOF")
    decompressed_sha = sha256_decompressed_gzip(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    columns = [str(value) for value in spec["pre_freeze_columns"]]
    if set(columns) != {"well_id", "fold"}:
        raise ValueError("pre-freeze exp226 loader must expose only well_id and fold")
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    by_well = frame.drop_duplicates().sort_values("well_id", kind="mergesort")
    if by_well["well_id"].duplicated().any():
        raise ValueError("one well maps to multiple outer folds")
    mapping = {str(row.well_id): int(row.fold) for row in by_well.itertuples(index=False)}
    manifest = {
        "phase": "pre_freeze_fold_identity",
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_content_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": len(mapping),
        "columns_read": columns,
        "forbidden_column_hits": 0,
    }
    return FoldIdentity(mapping, path, manifest)


def donor_wells_for_fold(
    fold_map: Mapping[str, int], valid_fold: int, *, excluded_names: Iterable[str] = ()
) -> list[str]:
    excluded = {str(value) for value in excluded_names}
    return sorted(
        well
        for well, fold in fold_map.items()
        if int(fold) != int(valid_fold) and well not in excluded
    )


def query_wells_for_fold(fold_map: Mapping[str, int], valid_fold: int) -> list[str]:
    return sorted(well for well, fold in fold_map.items() if int(fold) == int(valid_fold))


def inference_donor_wells(train_wells: Iterable[str], test_well: str) -> list[str]:
    return sorted(well for well in train_wells if str(well) != str(test_well))


def sample_row_indices(length: int, stride: int) -> np.ndarray:
    if length <= 0 or stride <= 0:
        raise ValueError("sample length and stride must be positive")
    indices = np.arange(0, length, stride, dtype=np.int64)
    if indices[-1] != length - 1:
        indices = np.r_[indices, np.int64(length - 1)]
    return indices


# %% [markdown]
# ## 4. Multiformation edge identity and active-grid helpers

# %%
def build_well_edges(
    well_id: str, frame: pd.DataFrame, *, stride: int, minimum_finite: int
) -> pd.DataFrame:
    indices = sample_row_indices(len(frame), stride)
    start = indices[:-1]
    end = indices[1:]
    formation = frame[list(FORMATION_COLUMNS)].to_numpy(np.float64)
    deltas = formation[end] - formation[start]
    finite = np.isfinite(deltas)
    finite_count = finite.sum(axis=1)
    masked = np.where(finite, deltas, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        response = np.nanmedian(masked, axis=1)
        mad = np.nanmedian(np.abs(masked - response[:, None]), axis=1)
    scale = np.maximum(1.4826 * mad, 0.02)
    u = frame["TVT"].to_numpy(np.float64) + frame["Z"].to_numpy(np.float64)
    output = pd.DataFrame(
        {
            "well_id": str(well_id),
            "start_row": frame["row_index"].to_numpy(np.int64)[start],
            "end_row": frame["row_index"].to_numpy(np.int64)[end],
            "x_start": frame["X"].to_numpy(np.float64)[start],
            "y_start": frame["Y"].to_numpy(np.float64)[start],
            "x_end": frame["X"].to_numpy(np.float64)[end],
            "y_end": frame["Y"].to_numpy(np.float64)[end],
            "true_delta_u": u[end] - u[start],
            "finite_formation_count": finite_count.astype(np.int8),
            "response": response,
            "scale": scale,
            "solver_eligible": finite_count >= int(minimum_finite),
        }
    )
    for position, formation_name in enumerate(FORMATION_COLUMNS):
        output[f"delta_{formation_name}"] = deltas[:, position]
    return output


def build_donor_edges(
    well_paths: Mapping[str, Path],
    donor_wells: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[np.ndarray], list[dict[str, Any]]]:
    stride = int(get_nested(config, "stage0.row_stride"))
    minimum = int(get_nested(config, "stage0.minimum_finite_formations_per_edge"))
    parts: list[pd.DataFrame] = []
    trajectories: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    for well in sorted(donor_wells):
        path = well_paths[well]
        frame = load_source_horizontal(path)
        trajectories.append(frame[["X", "Y"]].to_numpy(np.float64))
        parts.append(build_well_edges(well, frame, stride=stride, minimum_finite=minimum))
        evidence.append(
            {
                "phase": "outer_train_source",
                "well_id": well,
                "path": str(path),
                "rows": len(frame),
                "file_sha256": sha256_file(path),
                "columns_read": list(SOURCE_RAW_COLUMNS),
            }
        )
    if not parts:
        raise ValueError("outer fold has no donor edges")
    edges = pd.concat(parts, ignore_index=True)
    edges = edges.sort_values(
        ["well_id", "start_row", "end_row"], kind="mergesort"
    ).reset_index(drop=True)
    return edges, trajectories, evidence


def load_query_fold_geometry(
    well_paths: Mapping[str, Path], query_wells: Sequence[str]
) -> tuple[dict[str, pd.DataFrame], list[np.ndarray], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    trajectories: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    for well in sorted(query_wells):
        path = well_paths[well]
        frame = load_query_safe_horizontal(path)
        frames[well] = frame
        trajectories.append(frame[["X", "Y"]].to_numpy(np.float64))
        evidence.append(
            {
                "phase": "outer_valid_geometry_only",
                "well_id": well,
                "path": str(path),
                "rows": len(frame),
                "file_sha256": sha256_file(path),
                "safe_content_sha256": safe_query_content_sha256(frame),
                "columns_read": list(QUERY_RAW_COLUMNS),
                "forbidden_column_hits": 0,
            }
        )
    return frames, trajectories, evidence


def edge_identity_records(edges: pd.DataFrame, outer_fold: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    truth = edges["true_delta_u"].to_numpy(np.float64)
    for formation in (*FORMATION_COLUMNS, "median6"):
        values = (
            edges["response"].to_numpy(np.float64)
            if formation == "median6"
            else edges[f"delta_{formation}"].to_numpy(np.float64)
        )
        finite = np.isfinite(values) & np.isfinite(truth)
        error = truth[finite] - values[finite]
        records.append(
            {
                "outer_fold": int(outer_fold),
                "formation": formation,
                "edges": int(len(values)),
                "finite_edges": int(finite.sum()),
                "finite_fraction": float(finite.mean()),
                "rmse_ft": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
                "mae_ft": float(np.mean(np.abs(error))) if len(error) else math.nan,
                "bias_ft": float(np.mean(error)) if len(error) else math.nan,
                "mad_ft": (
                    float(np.median(np.abs(error - np.median(error))))
                    if len(error)
                    else math.nan
                ),
            }
        )
    return records


def _pack_grid_keys(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    x = np.asarray(ix, dtype=np.int64)
    y = np.asarray(iy, dtype=np.int64)
    if np.any((x < -(2**31)) | (x >= 2**31) | (y < -(2**31)) | (y >= 2**31)):
        raise ValueError("grid coordinate exceeds signed 32-bit packing contract")
    ux = (x + 2**31).astype(np.uint64)
    uy = (y + 2**31).astype(np.uint64)
    return (ux << np.uint64(32)) | uy


def _unpack_grid_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(keys, dtype=np.uint64)
    ix = (packed >> np.uint64(32)).astype(np.int64) - 2**31
    iy = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64) - 2**31
    return ix, iy


@dataclass(frozen=True)
class ActiveGrid:
    spacing: float
    keys: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    components: np.ndarray
    component_count: int


def _lookup_grid_keys(grid: ActiveGrid, keys: np.ndarray) -> np.ndarray:
    raw = np.asarray(keys, dtype=np.uint64)
    positions = np.searchsorted(grid.keys, raw)
    valid = positions < len(grid.keys)
    matched = np.zeros(len(raw), dtype=bool)
    matched[valid] = grid.keys[positions[valid]] == raw[valid]
    if not matched.all():
        raise ValueError(f"bilinear/regularizer basis misses {int((~matched).sum())} active nodes")
    return positions.astype(np.int64)


def build_active_grid(trajectories: Sequence[np.ndarray], spacing: float) -> ActiveGrid:
    if not trajectories:
        raise ValueError("active grid requires trajectories")
    base_keys: list[np.ndarray] = []
    for xy in trajectories:
        values = np.asarray(xy, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("trajectory geometry must be finite Nx2")
        ix = np.floor(values[:, 0] / spacing).astype(np.int64)
        iy = np.floor(values[:, 1] / spacing).astype(np.int64)
        base_keys.append(_pack_grid_keys(ix, iy))
    unique_base = np.unique(np.concatenate(base_keys))
    base_ix, base_iy = _unpack_grid_keys(unique_base)
    active_parts = [
        _pack_grid_keys(base_ix + dx, base_iy + dy)
        for dx in (-1, 0, 1, 2)
        for dy in (-1, 0, 1, 2)
    ]
    keys = np.unique(np.concatenate(active_parts))
    ix, iy = _unpack_grid_keys(keys)

    right = _pack_grid_keys(ix + 1, iy)
    up = _pack_grid_keys(ix, iy + 1)
    right_pos = np.searchsorted(keys, right)
    up_pos = np.searchsorted(keys, up)
    right_valid = right_pos < len(keys)
    up_valid = up_pos < len(keys)
    right_match = np.zeros(len(keys), dtype=bool)
    up_match = np.zeros(len(keys), dtype=bool)
    right_match[right_valid] = keys[right_pos[right_valid]] == right[right_valid]
    up_match[up_valid] = keys[up_pos[up_valid]] == up[up_valid]
    rows = np.r_[np.flatnonzero(right_match), np.flatnonzero(up_match)]
    cols = np.r_[right_pos[right_match], up_pos[up_match]]
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(rows), dtype=np.int8), (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(len(keys), len(keys)),
    ).tocsr()
    component_count, components = connected_components(
        adjacency, directed=False, return_labels=True
    )
    return ActiveGrid(
        spacing=float(spacing),
        keys=keys,
        ix=ix,
        iy=iy,
        components=components.astype(np.int32),
        component_count=int(component_count),
    )


def bilinear_basis(grid: ActiveGrid, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    ix = np.floor(x_values / grid.spacing).astype(np.int64)
    iy = np.floor(y_values / grid.spacing).astype(np.int64)
    fx = x_values / grid.spacing - ix
    fy = y_values / grid.spacing - iy
    key_parts = (
        _pack_grid_keys(ix, iy),
        _pack_grid_keys(ix + 1, iy),
        _pack_grid_keys(ix, iy + 1),
        _pack_grid_keys(ix + 1, iy + 1),
    )
    indices = np.column_stack([_lookup_grid_keys(grid, keys) for keys in key_parts])
    weights = np.column_stack(
        ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy)
    )
    return indices, weights


def evaluate_bilinear(
    grid: ActiveGrid, potential: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    indices, weights = bilinear_basis(grid, x, y)
    return np.sum(np.asarray(potential, dtype=np.float64)[indices] * weights, axis=1)


# %% [markdown]
# ## 5. Bilinear constraints and deterministic sparse Huber solver

# %%
def edge_constraint_matrix(grid: ActiveGrid, edges: pd.DataFrame) -> sp.csr_matrix:
    start_indices, start_weights = bilinear_basis(
        grid,
        edges["x_start"].to_numpy(np.float64),
        edges["y_start"].to_numpy(np.float64),
    )
    end_indices, end_weights = bilinear_basis(
        grid,
        edges["x_end"].to_numpy(np.float64),
        edges["y_end"].to_numpy(np.float64),
    )
    n_edges = len(edges)
    rows = np.repeat(np.arange(n_edges, dtype=np.int64), 8)
    columns = np.column_stack((start_indices, end_indices)).reshape(-1)
    values = np.column_stack((-start_weights, end_weights)).reshape(-1)
    matrix = sp.coo_matrix(
        (values, (rows, columns)), shape=(n_edges, len(grid.keys)), dtype=np.float64
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def build_second_difference_matrix(grid: ActiveGrid) -> sp.csr_matrix:
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    row_offset = 0

    def append_rows(nodes: np.ndarray, stencil: Sequence[tuple[int, int, float]]) -> None:
        nonlocal row_offset
        if not len(nodes):
            return
        centers_x = grid.ix[nodes]
        centers_y = grid.iy[nodes]
        indices = [
            _lookup_grid_keys(grid, _pack_grid_keys(centers_x + dx, centers_y + dy))
            for dx, dy, _ in stencil
        ]
        same_component = np.ones(len(nodes), dtype=bool)
        for values in indices:
            same_component &= grid.components[values] == grid.components[nodes]
        nodes = nodes[same_component]
        indices = [values[same_component] for values in indices]
        if not len(nodes):
            return
        local_rows = np.repeat(
            np.arange(row_offset, row_offset + len(nodes), dtype=np.int64), len(stencil)
        )
        local_cols = np.column_stack(indices).reshape(-1)
        local_values = np.tile(
            np.asarray([value for _, _, value in stencil], dtype=np.float64), len(nodes)
        )
        row_indices.append(local_rows)
        col_indices.append(local_cols)
        coefficients.append(local_values)
        row_offset += len(nodes)

    def contains(dx: int, dy: int) -> np.ndarray:
        neighbor = _pack_grid_keys(grid.ix + dx, grid.iy + dy)
        positions = np.searchsorted(grid.keys, neighbor)
        valid = positions < len(grid.keys)
        output = np.zeros(len(grid.keys), dtype=bool)
        output[valid] = grid.keys[positions[valid]] == neighbor[valid]
        return output

    xx_mask = contains(-1, 0) & contains(1, 0)
    yy_mask = contains(0, -1) & contains(0, 1)
    xy_mask = contains(1, 0) & contains(0, 1) & contains(1, 1)
    append_rows(
        np.flatnonzero(xx_mask), ((-1, 0, 1.0), (0, 0, -2.0), (1, 0, 1.0))
    )
    append_rows(
        np.flatnonzero(yy_mask), ((0, -1, 1.0), (0, 0, -2.0), (0, 1, 1.0))
    )
    append_rows(
        np.flatnonzero(xy_mask),
        ((0, 0, 1.0), (1, 0, -1.0), (0, 1, -1.0), (1, 1, 1.0)),
    )
    if not row_indices:
        return sp.csr_matrix((0, len(grid.keys)), dtype=np.float64)
    return sp.coo_matrix(
        (np.concatenate(coefficients), (np.concatenate(row_indices), np.concatenate(col_indices))),
        shape=(row_offset, len(grid.keys)),
        dtype=np.float64,
    ).tocsr()


def build_gauge_matrix(grid: ActiveGrid) -> sp.csr_matrix:
    counts = np.bincount(grid.components, minlength=grid.component_count).astype(np.float64)
    rows = grid.components.astype(np.int64)
    columns = np.arange(len(grid.keys), dtype=np.int64)
    values = 1.0 / counts[grid.components]
    return sp.coo_matrix(
        (values, (rows, columns)),
        shape=(grid.component_count, len(grid.keys)),
        dtype=np.float64,
    ).tocsr()


def donor_components(grid: ActiveGrid, matrix: sp.csr_matrix) -> set[int]:
    touched = np.unique(matrix.indices)
    return set(grid.components[touched].astype(int))


def query_component_coverage(
    grid: ActiveGrid,
    donor_matrix: sp.csr_matrix,
    query_frames: Mapping[str, pd.DataFrame],
) -> tuple[float, float, int, int]:
    donors = donor_components(grid, donor_matrix)
    query_rows = 0
    supported_rows = 0
    basis_rows = 0
    for frame in query_frames.values():
        indices, _ = bilinear_basis(
            grid, frame["X"].to_numpy(np.float64), frame["Y"].to_numpy(np.float64)
        )
        components = grid.components[indices]
        same = np.all(components == components[:, :1], axis=1)
        basis_rows += int(same.sum())
        supported = same & np.isin(components[:, 0], np.asarray(sorted(donors), dtype=np.int32))
        supported_rows += int(supported.sum())
        query_rows += len(frame)
    if query_rows == 0:
        raise ValueError("fold has no query rows")
    return basis_rows / query_rows, supported_rows / query_rows, query_rows, supported_rows


def active_component_donor_coverage(grid: ActiveGrid, matrix: sp.csr_matrix) -> float:
    return len(donor_components(grid, matrix)) / grid.component_count


def build_runtime_guard(
    grid: ActiveGrid,
    edge_matrix: sp.csr_matrix,
    regularizer: sp.csr_matrix,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    limits = get_nested(config, "solver.runtime_guards")
    values = {
        "active_nodes": len(grid.keys),
        "donor_edges": edge_matrix.shape[0],
        "regularizer_rows": regularizer.shape[0],
        "constraint_matrix_nnz": edge_matrix.nnz + regularizer.nnz,
        "components": grid.component_count,
    }
    checks = {
        "active_nodes_guard": values["active_nodes"]
        <= int(limits["maximum_active_nodes_per_outer_fold"]),
        "donor_edges_guard": values["donor_edges"]
        <= int(limits["maximum_donor_edges_per_outer_fold"]),
        "regularizer_rows_guard": values["regularizer_rows"]
        <= int(limits["maximum_regularizer_rows_per_outer_fold"]),
        "constraint_nnz_guard": values["constraint_matrix_nnz"]
        <= int(limits["maximum_constraint_matrix_nnz_per_outer_fold"]),
    }
    return {**values, **checks, "passed": bool(all(checks.values()))}


def huber_values(residual: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(np.asarray(residual, dtype=np.float64))
    return np.where(
        absolute <= delta, 0.5 * np.square(absolute), delta * (absolute - 0.5 * delta)
    )


@dataclass(frozen=True)
class SolverResult:
    potential: np.ndarray
    objective_history: tuple[float, ...]
    iterations: int
    converged: bool
    solution_content_sha256: str


def solve_huber_potential(
    matrix: sp.csr_matrix,
    response: np.ndarray,
    scale: np.ndarray,
    regularizer: sp.csr_matrix,
    gauge: sp.csr_matrix,
    *,
    lambda_value: float,
    huber_delta: float,
    maximum_iterations: int,
    relative_tolerance: float,
) -> SolverResult:
    response_values = np.asarray(response, dtype=np.float64)
    scale_values = np.asarray(scale, dtype=np.float64)
    if len(response_values) != matrix.shape[0] or len(scale_values) != matrix.shape[0]:
        raise ValueError("edge response/scale does not match constraint matrix")
    if not np.isfinite(response_values).all() or not np.isfinite(scale_values).all():
        raise ValueError("solver response and scale must be finite")
    if np.any(scale_values <= 0.0):
        raise ValueError("solver scale must be positive")
    scaled = sp.diags(1.0 / scale_values, format="csr") @ matrix
    target = response_values / scale_values
    penalty = (2.0 * float(lambda_value)) * (regularizer.T @ regularizer)
    weights = np.ones(len(target), dtype=np.float64)
    previous = math.inf
    objective_history: list[float] = []
    potential = np.zeros(matrix.shape[1], dtype=np.float64)
    converged = False
    for _ in range(int(maximum_iterations)):
        normal = scaled.T @ sp.diags(weights, format="csr") @ scaled + penalty
        rhs = scaled.T @ (weights * target)
        kkt = sp.bmat(
            [[normal, gauge.T], [gauge, None]], format="csc", dtype=np.float64
        )
        kkt_rhs = np.r_[np.asarray(rhs).ravel(), np.zeros(gauge.shape[0], dtype=np.float64)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution = spsolve(kkt, kkt_rhs, permc_spec="NATURAL", use_umfpack=False)
        potential = np.asarray(solution[: matrix.shape[1]], dtype=np.float64)
        if not np.isfinite(potential).all():
            raise ValueError("sparse solver returned non-finite potential")
        residual = scaled @ potential - target
        regularized = regularizer @ potential
        objective = float(
            huber_values(residual, huber_delta).sum()
            + float(lambda_value) * np.dot(regularized, regularized)
        )
        objective_history.append(objective)
        if math.isfinite(previous):
            relative = abs(previous - objective) / max(1.0, abs(previous))
            if relative <= float(relative_tolerance):
                converged = True
                break
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        outside = absolute > float(huber_delta)
        weights[outside] = float(huber_delta) / absolute[outside]
        previous = objective
    if not converged:
        raise RuntimeError(
            f"Huber IRLS did not converge in {maximum_iterations} iterations; "
            f"last objective={objective_history[-1] if objective_history else None}"
        )
    digest = array_content_sha256(
        potential.astype("<f8", copy=False), context=("exp301_potential", str(lambda_value))
    )
    return SolverResult(
        potential=potential,
        objective_history=tuple(objective_history),
        iterations=len(objective_history),
        converged=True,
        solution_content_sha256=digest,
    )


def stable_inner_split(well_id: str, splits: int = 3) -> int:
    digest = hashlib.sha256(str(well_id).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % int(splits)


def select_lambda_outer_train_only(
    edges: pd.DataFrame,
    matrix: sp.csr_matrix,
    regularizer: sp.csr_matrix,
    gauge: sp.csr_matrix,
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    grid: ActiveGrid | None = None,
) -> tuple[float, pd.DataFrame, int]:
    solver_cfg = get_nested(config, "solver")
    candidates = [float(value) for value in solver_cfg["lambda_candidates"]]
    split = edges["well_id"].map(stable_inner_split).to_numpy(np.int8)
    response = edges["response"].to_numpy(np.float64)
    scale = edges["scale"].to_numpy(np.float64)
    records: list[dict[str, Any]] = []
    fit_count = 0
    for lambda_value in candidates:
        total_loss = 0.0
        total_edges = 0
        for inner_fold in range(3):
            train = split != inner_fold
            valid = ~train
            if not train.any() or not valid.any():
                raise ValueError("stable inner split produced an empty train/valid edge set")
            if grid is not None and not math.isclose(
                active_component_donor_coverage(grid, matrix[train]),
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "inner edge holdout leaves an active component without a donor constraint"
                )
            result = solve_huber_potential(
                matrix[train],
                response[train],
                scale[train],
                regularizer,
                gauge,
                lambda_value=lambda_value,
                huber_delta=float(solver_cfg["huber_delta"]),
                maximum_iterations=int(solver_cfg["irls_max_iterations"]),
                relative_tolerance=float(solver_cfg["irls_relative_objective_tolerance"]),
            )
            fit_count += 1
            heldout_residual = (matrix[valid] @ result.potential - response[valid]) / scale[valid]
            loss = float(huber_values(heldout_residual, float(solver_cfg["huber_delta"])).sum())
            total_loss += loss
            total_edges += int(valid.sum())
            records.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "lambda": lambda_value,
                    "heldout_edges": int(valid.sum()),
                    "heldout_huber_loss_sum": loss,
                    "heldout_huber_loss_mean": loss / int(valid.sum()),
                    "solver_iterations": result.iterations,
                    "solution_content_sha256": result.solution_content_sha256,
                }
            )
        records.append(
            {
                "outer_fold": int(outer_fold),
                "inner_fold": "aggregate",
                "lambda": lambda_value,
                "heldout_edges": total_edges,
                "heldout_huber_loss_sum": total_loss,
                "heldout_huber_loss_mean": total_loss / total_edges,
                "solver_iterations": np.nan,
                "solution_content_sha256": None,
            }
        )
    frame = pd.DataFrame(records)
    aggregate = frame[frame["inner_fold"].eq("aggregate")].copy()
    minimum = float(aggregate["heldout_huber_loss_mean"].min())
    tied = aggregate[
        np.isclose(aggregate["heldout_huber_loss_mean"], minimum, rtol=0.0, atol=1.0e-12)
    ]
    selected = float(tied["lambda"].max())
    frame["selected"] = frame["lambda"].eq(selected)
    return selected, frame, fit_count


# %% [markdown]
# ## 6. Stage 0 support audit and conditional Stage 1 OOF generation

# %%
@dataclass
class PreparedFold:
    outer_fold: int
    edges: pd.DataFrame
    query_frames: dict[str, pd.DataFrame]
    grid: ActiveGrid
    edge_matrix: sp.csr_matrix
    regularizer: sp.csr_matrix
    gauge: sp.csr_matrix
    identity_records: list[dict[str, Any]]
    support_record: dict[str, Any]
    input_evidence: list[dict[str, Any]]


def prepare_outer_fold(
    outer_fold: int,
    fold_map: Mapping[str, int],
    well_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> PreparedFold:
    donors = donor_wells_for_fold(fold_map, outer_fold)
    queries = query_wells_for_fold(fold_map, outer_fold)
    edges_all, donor_trajectories, donor_evidence = build_donor_edges(
        well_paths, donors, config
    )
    query_frames, query_trajectories, query_evidence = load_query_fold_geometry(
        well_paths, queries
    )
    eligible = edges_all.loc[edges_all["solver_eligible"]].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError(f"outer fold {outer_fold} has no eligible donor edges")
    spacing = float(get_nested(config, "solver.grid_spacing_ft"))
    grid = build_active_grid([*donor_trajectories, *query_trajectories], spacing)
    matrix = edge_constraint_matrix(grid, eligible)
    regularizer = build_second_difference_matrix(grid)
    gauge = build_gauge_matrix(grid)
    basis_coverage, donor_coverage, query_rows, supported_rows = query_component_coverage(
        grid, matrix, query_frames
    )
    component_donor_coverage = active_component_donor_coverage(grid, matrix)
    finite_fraction = float(edges_all["solver_eligible"].mean())
    suffix_rows = int(
        sum(frame["TVT_input"].isna().sum() for frame in query_frames.values())
    )
    runtime_guard = build_runtime_guard(grid, matrix, regularizer, config)
    support_record = {
        "outer_fold": int(outer_fold),
        "donor_wells": len(donors),
        "query_wells": len(queries),
        "all_edges": len(edges_all),
        "eligible_edges": len(eligible),
        "edges_with_three_finite_fraction": finite_fraction,
        "active_nodes": len(grid.keys),
        "components": grid.component_count,
        "regularizer_rows": regularizer.shape[0],
        "constraint_nnz": matrix.nnz,
        "query_rows": query_rows,
        "supported_query_rows": supported_rows,
        "query_bilinear_basis_coverage": basis_coverage,
        "query_component_with_donor_coverage": donor_coverage,
        "active_component_with_donor_coverage": component_donor_coverage,
        "prediction_suffix_rows": suffix_rows,
        "forbidden_column_hits": 0,
        "outer_valid_truth_access_before_freeze": 0,
        **{f"runtime_{key}": value for key, value in runtime_guard.items()},
    }
    return PreparedFold(
        outer_fold=int(outer_fold),
        edges=eligible,
        query_frames=query_frames,
        grid=grid,
        edge_matrix=matrix,
        regularizer=regularizer,
        gauge=gauge,
        identity_records=edge_identity_records(edges_all, outer_fold),
        support_record=support_record,
        input_evidence=[*donor_evidence, *query_evidence],
    )


def evaluate_stage0(
    identity: pd.DataFrame,
    support: pd.DataFrame,
    fold_map: Mapping[str, int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stage0 = get_nested(config, "stage0")
    expected_rows = int(get_nested(config, "validation.expected_prediction_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    n_folds = int(get_nested(config, "validation.n_folds"))
    formation_rows = identity[identity["formation"].isin(FORMATION_COLUMNS)]
    median_rows = identity[identity["formation"].eq("median6")]
    checks = {
        "fold_inventory_exact": set(identity["outer_fold"].astype(int))
        == set(range(n_folds)),
        "well_inventory_exact": len(fold_map) == expected_wells,
        "prediction_suffix_identity_exact": int(support["prediction_suffix_rows"].sum())
        == expected_rows,
        "formation_identity_rmse_all_at_most_0p02": bool(
            (
                formation_rows["rmse_ft"]
                <= float(stage0["maximum_edge_identity_rmse_ft_each_formation"])
            ).all()
        ),
        "median6_identity_rmse_all_at_most_0p02": bool(
            (median_rows["rmse_ft"] <= float(stage0["maximum_edge_identity_rmse_ft_median6"])).all()
        ),
        "three_finite_edge_fraction_all_at_least_0p995": bool(
            (
                support["edges_with_three_finite_fraction"]
                >= float(stage0["minimum_edges_with_three_finite_formations_fraction"])
            ).all()
        ),
        "query_bilinear_basis_coverage_one": bool(
            np.allclose(
                support["query_bilinear_basis_coverage"],
                float(stage0["required_query_bilinear_basis_coverage"]),
                rtol=0.0,
                atol=0.0,
            )
        ),
        "query_component_donor_coverage_one": bool(
            np.allclose(
                support["query_component_with_donor_coverage"],
                float(stage0["required_query_component_with_donor_fraction"]),
                rtol=0.0,
                atol=0.0,
            )
        ),
        "active_component_donor_coverage_one": bool(
            np.allclose(
                support["active_component_with_donor_coverage"],
                1.0,
                rtol=0.0,
                atol=0.0,
            )
        ),
        "forbidden_column_hits_zero": int(support["forbidden_column_hits"].sum()) == 0,
        "outer_valid_truth_access_before_freeze_zero": int(
            support["outer_valid_truth_access_before_freeze"].sum()
        )
        == 0,
        "runtime_guards_all_pass": bool(support["runtime_passed"].all()),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def run_stage0(
    fold_identity: FoldIdentity,
    well_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = [fold_identity.manifest]
    for outer_fold in range(int(get_nested(config, "validation.n_folds"))):
        print(f"Stage 0 outer fold {outer_fold}: identity and support audit")
        prepared = prepare_outer_fold(outer_fold, fold_identity.by_well, well_paths, config)
        identities.extend(prepared.identity_records)
        supports.append(prepared.support_record)
        evidence.extend(prepared.input_evidence)
    identity_frame = pd.DataFrame(identities)
    support_frame = pd.DataFrame(supports)
    decision = evaluate_stage0(identity_frame, support_frame, fold_identity.by_well, config)
    return identity_frame, support_frame, evidence, decision


def predict_query_fold(
    prepared: PreparedFold, potential: np.ndarray
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well, frame in sorted(prepared.query_frames.items()):
        prefix = frame["TVT_input"].to_numpy(np.float64)
        anchor_position = int(np.flatnonzero(np.isfinite(prefix))[-1])
        suffix_positions = np.arange(anchor_position + 1, len(frame), dtype=np.int64)
        phi = evaluate_bilinear(
            prepared.grid,
            potential,
            frame["X"].to_numpy(np.float64),
            frame["Y"].to_numpy(np.float64),
        )
        z = frame["Z"].to_numpy(np.float64)
        prediction = prefix[anchor_position] + z[anchor_position] + phi - phi[anchor_position] - z
        row_index = frame["row_index"].to_numpy(np.int64)[suffix_positions]
        part = pd.DataFrame(
            {
                "id": [f"{well}_{int(value)}" for value in row_index],
                "well_id": well,
                "row_index": row_index,
                "outer_fold": int(prepared.outer_fold),
                "MD": frame["MD"].to_numpy(np.float64)[suffix_positions],
                "md_since": (
                    frame["MD"].to_numpy(np.float64)[suffix_positions]
                    - frame["MD"].to_numpy(np.float64)[anchor_position]
                ),
                "tvt_pred_exp301": prediction[suffix_positions],
            }
        )
        parts.append(part)
    if not parts:
        raise ValueError("outer fold generated no suffix predictions")
    output = pd.concat(parts, ignore_index=True)
    if output["id"].duplicated().any() or not np.isfinite(
        output["tvt_pred_exp301"].to_numpy(np.float64)
    ).all():
        raise ValueError("outer-fold prediction identity/finite guard failed")
    return output


def sparse_structure_sha256(matrix: sp.csr_matrix, label: str) -> str:
    value = matrix.tocsr()
    return array_content_sha256(
        value.indptr.astype("<i8", copy=False),
        value.indices.astype("<i8", copy=False),
        value.data.astype("<f8", copy=False),
        context=(label, str(value.shape)),
    )


def run_stage1(
    fold_identity: FoldIdentity,
    well_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], int]:
    predictions: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    solver_records: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    total_fits = 0
    solver_cfg = get_nested(config, "solver")
    for outer_fold in range(int(get_nested(config, "validation.n_folds"))):
        print(f"Stage 1 outer fold {outer_fold}: lambda selection and final solver")
        prepared = prepare_outer_fold(outer_fold, fold_identity.by_well, well_paths, config)
        selected_lambda, lambda_frame, inner_fits = select_lambda_outer_train_only(
            prepared.edges,
            prepared.edge_matrix,
            prepared.regularizer,
            prepared.gauge,
            config,
            outer_fold=outer_fold,
            grid=prepared.grid,
        )
        total_fits += inner_fits
        final = solve_huber_potential(
            prepared.edge_matrix,
            prepared.edges["response"].to_numpy(np.float64),
            prepared.edges["scale"].to_numpy(np.float64),
            prepared.regularizer,
            prepared.gauge,
            lambda_value=selected_lambda,
            huber_delta=float(solver_cfg["huber_delta"]),
            maximum_iterations=int(solver_cfg["irls_max_iterations"]),
            relative_tolerance=float(solver_cfg["irls_relative_objective_tolerance"]),
        )
        total_fits += 1
        solution_path = artifacts_dir / f"{OUTPUT_PREFIX}_fold{outer_fold}_solution.npy"
        np.save(solution_path, final.potential.astype("<f8", copy=False), allow_pickle=False)
        predictions.append(predict_query_fold(prepared, final.potential))
        lambda_frames.append(lambda_frame)
        solver_records.append(
            {
                "outer_fold": outer_fold,
                "selected_lambda": selected_lambda,
                "active_nodes": len(prepared.grid.keys),
                "components": prepared.grid.component_count,
                "donor_edges": len(prepared.edges),
                "edge_matrix_nnz": prepared.edge_matrix.nnz,
                "regularizer_rows": prepared.regularizer.shape[0],
                "regularizer_nnz": prepared.regularizer.nnz,
                "solver_iterations": final.iterations,
                "solver_converged": final.converged,
                "grid_content_sha256": array_content_sha256(
                    prepared.grid.ix.astype("<i8", copy=False),
                    prepared.grid.iy.astype("<i8", copy=False),
                    prepared.grid.components.astype("<i4", copy=False),
                    context=("active_grid", str(outer_fold)),
                ),
                "donor_edge_content_sha256": frame_content_sha256(
                    prepared.edges[
                        [
                            "well_id",
                            "start_row",
                            "end_row",
                            "response",
                            "scale",
                        ]
                    ]
                ),
                "edge_structure_sha256": sparse_structure_sha256(
                    prepared.edge_matrix, f"edge_matrix_fold_{outer_fold}"
                ),
                "regularizer_structure_sha256": sparse_structure_sha256(
                    prepared.regularizer, f"regularizer_fold_{outer_fold}"
                ),
                "solution_content_sha256": final.solution_content_sha256,
                "solution_file": str(solution_path),
                "solution_file_sha256": sha256_file(solution_path),
            }
        )
        evidence.extend(prepared.input_evidence)
    maximum_fits = int(get_nested(config, "solver.runtime_guards.maximum_total_solver_fits"))
    if total_fits > maximum_fits:
        raise ValueError(f"solver fit guard exceeded: {total_fits} > {maximum_fits}")
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["outer_fold", "well_id", "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    expected_rows = int(get_nested(config, "validation.expected_prediction_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(oof) != expected_rows or oof["well_id"].nunique() != expected_wells:
        raise ValueError("Stage 1 OOF identity contract failed")
    if oof["id"].duplicated().any() or not np.isfinite(oof["tvt_pred_exp301"]).all():
        raise ValueError("Stage 1 OOF duplicate/finite guard failed")
    return (
        oof,
        pd.concat(lambda_frames, ignore_index=True),
        pd.DataFrame(solver_records),
        evidence,
        total_fits,
    )


# %% [markdown]
# ## 7. Prediction freeze, late truth join, and direct-quality readouts

# %%
@dataclass(frozen=True)
class PredictionFreeze:
    prediction_path: Path
    file_sha256: str
    decompressed_content_sha256: str
    logical_content_sha256: str
    rows: int
    wells: int
    duplicate_rows: int
    finite_coverage: float
    truth_access_count_before_freeze: int
    manifest_path: Path


def freeze_oof_prediction(
    oof: pd.DataFrame, artifacts_dir: Path, config: Mapping[str, Any]
) -> PredictionFreeze:
    columns = [
        "id",
        "well_id",
        "row_index",
        "outer_fold",
        "MD",
        "md_since",
        "tvt_pred_exp301",
    ]
    if list(oof.columns) != columns:
        raise ValueError(f"OOF freeze schema differs from contract: {list(oof.columns)}")
    path = artifacts_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    oof.to_csv(
        path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    logical_sha = frame_content_sha256(oof)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "target_free_oof_prediction_frozen",
        "frozen_at": datetime.now(UTC).isoformat(),
        "rows": len(oof),
        "wells": int(oof["well_id"].nunique()),
        "duplicate_rows": int(oof["id"].duplicated().sum()),
        "finite_coverage": float(np.isfinite(oof["tvt_pred_exp301"]).mean()),
        "prediction_file": str(path),
        "prediction_file_sha256": sha256_file(path),
        "prediction_decompressed_content_sha256": sha256_decompressed_gzip(path),
        "prediction_logical_content_sha256": logical_sha,
        "truth_access_count_before_freeze": 0,
        "allowed_outer_valid_columns": list(
            get_nested(config, "validation.allowed_outer_valid_columns")
        ),
        "forbidden_outer_valid_before_prediction_freeze": list(
            get_nested(config, "validation.forbidden_outer_valid_before_prediction_freeze")
        ),
    }
    manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_prediction_freeze.json"
    write_json(manifest_path, payload)
    return PredictionFreeze(
        prediction_path=path,
        file_sha256=payload["prediction_file_sha256"],
        decompressed_content_sha256=payload["prediction_decompressed_content_sha256"],
        logical_content_sha256=logical_sha,
        rows=len(oof),
        wells=int(oof["well_id"].nunique()),
        duplicate_rows=int(oof["id"].duplicated().sum()),
        finite_coverage=float(np.isfinite(oof["tvt_pred_exp301"]).mean()),
        truth_access_count_before_freeze=0,
        manifest_path=manifest_path,
    )


def verify_prediction_freeze(oof: pd.DataFrame, freeze: PredictionFreeze) -> None:
    if sha256_file(freeze.prediction_path) != freeze.file_sha256:
        raise ValueError("frozen OOF file changed before truth join")
    if sha256_decompressed_gzip(freeze.prediction_path) != freeze.decompressed_content_sha256:
        raise ValueError("frozen OOF decompressed content changed before truth join")
    if frame_content_sha256(oof) != freeze.logical_content_sha256:
        raise ValueError("in-memory OOF changed before truth join")
    if freeze.truth_access_count_before_freeze != 0:
        raise ValueError("truth was accessed before OOF freeze")


def load_truth_after_prediction_freeze(
    oof: pd.DataFrame,
    freeze: PredictionFreeze,
    well_paths: Mapping[str, Path],
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    verify_prediction_freeze(oof, freeze)
    parts: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for well in sorted(oof["well_id"].astype(str).unique()):
        path = well_paths[well]
        frame = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        suffix = frame["TVT_input"].isna().to_numpy()
        rows = np.flatnonzero(suffix).astype(np.int64)
        truth = pd.to_numeric(frame.loc[suffix, "TVT"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError(f"post-freeze truth is non-finite: {path}")
        parts.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in rows],
                    "well_id": well,
                    "row_index": rows,
                    "tvt_true": truth,
                }
            )
        )
        evidence.append(
            {
                "phase": "post_freeze_truth",
                "well_id": well,
                "path": str(path),
                "rows": len(frame),
                "suffix_rows": len(rows),
                "file_sha256": sha256_file(path),
                "columns_read": ["TVT", "TVT_input"],
            }
        )
    truth_frame = pd.concat(parts, ignore_index=True)
    if truth_frame["id"].duplicated().any():
        raise ValueError("post-freeze truth identity is duplicated")
    indexer = pd.Index(truth_frame["id"].astype(str)).get_indexer(oof["id"].astype(str))
    if np.any(indexer < 0):
        raise ValueError("OOF rows are missing from post-freeze truth")
    aligned = truth_frame.iloc[indexer].reset_index(drop=True)
    if not np.array_equal(aligned["id"].to_numpy(), oof["id"].to_numpy()):
        raise ValueError("post-freeze truth alignment failed")
    return (
        aligned["tvt_true"].to_numpy(np.float64),
        evidence,
        frame_content_sha256(aligned[["id", "tvt_true"]]),
    )


def load_exp226_comparison_after_freeze(
    oof: pd.DataFrame,
    truth: np.ndarray,
    freeze: PredictionFreeze,
    fold_identity: FoldIdentity,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    verify_prediction_freeze(oof, freeze)
    spec = get_nested(config, "data.exp226_oof")
    columns = [str(value) for value in spec["post_freeze_columns"]]
    frame = pd.read_csv(fold_identity.path, usecols=columns, dtype={"well_id": str})
    frame["id"] = frame["well_id"].astype(str) + "_" + frame["row_idx"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError("exp226 comparison identity is duplicated")
    indexer = pd.Index(frame["id"]).get_indexer(oof["id"])
    if np.any(indexer < 0):
        raise ValueError("exp226 comparison lacks exp301 OOF rows")
    aligned = frame.iloc[indexer].reset_index(drop=True)
    baseline = aligned["tvt_pred"].to_numpy(np.float64)
    baseline_truth = aligned["tvt_true"].to_numpy(np.float64)
    if not np.allclose(baseline_truth, truth, rtol=0.0, atol=1.0e-8):
        raise ValueError("exp226 and raw-train truth disagree")
    if not np.array_equal(aligned["fold"].to_numpy(np.int8), oof["outer_fold"].to_numpy(np.int8)):
        raise ValueError("exp226 and exp301 fold identity disagree")
    actual_rmse = rmse(baseline, truth)
    expected_rmse = float(get_nested(config, "success_criteria.direct_quality.baseline_rmse_ft"))
    if not math.isclose(actual_rmse, expected_rmse, rel_tol=0.0, abs_tol=1.0e-3):
        raise ValueError(f"exp226 baseline RMSE parity failed: {actual_rmse} != {expected_rmse}")
    evidence = {
        "phase": "post_freeze_comparison",
        "source": "exp226_oof",
        "path": str(fold_identity.path),
        "rows": len(frame),
        "file_sha256": sha256_file(fold_identity.path),
        "decompressed_content_sha256": sha256_decompressed_gzip(fold_identity.path),
        "columns_read": columns,
        "aligned_prediction_content_sha256": array_content_sha256(
            baseline.astype("<f8", copy=False), context=("exp226_aligned",)
        ),
    }
    return baseline, evidence


def load_hidden_like_sets(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        spec["patterns"],
        label="hidden-like assignment",
        expected_sha256=str(spec["expected_file_sha256"]),
    )
    frame = pd.read_csv(path)
    well_column = str(spec["well_column"])
    output: dict[str, set[str]] = {}
    for scope, role_column in spec["role_columns"].items():
        if role_column not in frame:
            raise ValueError(f"hidden-like role missing: {role_column}")
        selected = set(frame.loc[frame[role_column].eq("valid"), well_column].astype(str))
        if selected - expected_wells:
            raise ValueError(f"hidden-like scope {scope} has unknown wells")
        output[str(scope)] = selected
    evidence = {
        "phase": "post_freeze_diagnostic",
        "source": "hidden_like_assignment",
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }
    return output, evidence


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction_values = np.asarray(prediction, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(prediction_values - truth_values))))


def metric_record(
    scope: str,
    scope_type: str,
    mask: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        raise ValueError(f"metric scope is empty: {scope}")
    candidate_rmse = rmse(prediction[selected], truth[selected])
    baseline_rmse = rmse(baseline[selected], truth[selected])
    error = prediction[selected] - truth[selected]
    return {
        "scope": scope,
        "scope_type": scope_type,
        "rows": int(selected.sum()),
        "exp301_rmse": candidate_rmse,
        "exp226_rmse": baseline_rmse,
        "delta_rmse_vs_exp226": candidate_rmse - baseline_rmse,
        "exp301_mae": float(np.mean(np.abs(error))),
        "exp301_bias": float(np.mean(error)),
    }


def build_direct_readouts(
    oof: pd.DataFrame,
    prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    hidden_sets: Mapping[str, set[str]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    folds = oof["outer_fold"].to_numpy(np.int8)
    wells = oof["well_id"].astype(str).to_numpy()
    md_since = oof["md_since"].to_numpy(np.float64)
    all_rows = np.ones(len(oof), dtype=bool)
    fold_records = [metric_record("pooled", "pooled", all_rows, prediction, baseline, truth)]
    for fold in range(int(get_nested(config, "validation.n_folds"))):
        fold_records.append(
            metric_record(f"fold_{fold}", "fold", folds == fold, prediction, baseline, truth)
        )
    subgroup_records: list[dict[str, Any]] = []
    for bucket, bounds in get_nested(config, "audit.distance_buckets_ft").items():
        lower, upper = float(bounds[0]), float(bounds[1])
        subgroup_records.append(
            metric_record(
                f"distance_{bucket}",
                "distance_bucket",
                (md_since >= lower) & (md_since < upper),
                prediction,
                baseline,
                truth,
            )
        )
    for scope, selected_wells in hidden_sets.items():
        subgroup_records.append(
            metric_record(
                scope,
                "hidden_like",
                np.isin(wells, np.asarray(sorted(selected_wells), dtype=object)),
                prediction,
                baseline,
                truth,
            )
        )

    by_well_records: list[dict[str, Any]] = []
    for well in sorted(set(wells)):
        mask = wells == well
        candidate_rmse = rmse(prediction[mask], truth[mask])
        baseline_rmse = rmse(baseline[mask], truth[mask])
        by_well_records.append(
            {
                "well_id": well,
                "outer_fold": int(folds[np.flatnonzero(mask)[0]]),
                "rows": int(mask.sum()),
                "exp301_rmse": candidate_rmse,
                "exp226_rmse": baseline_rmse,
                "delta_rmse_vs_exp226": candidate_rmse - baseline_rmse,
                "exp301_bias": float(np.mean(prediction[mask] - truth[mask])),
                "exp226_bias": float(np.mean(baseline[mask] - truth[mask])),
            }
        )
    fold_metrics = pd.DataFrame(fold_records)
    subgroup_metrics = pd.DataFrame(subgroup_records)
    by_well = pd.DataFrame(by_well_records)
    pooled = fold_metrics[fold_metrics["scope"].eq("pooled")].iloc[0]
    fold_only = fold_metrics[fold_metrics["scope_type"].eq("fold")]
    longtail = subgroup_metrics[subgroup_metrics["scope"].eq("distance_1000_plus")].iloc[0]
    hidden_spatial = subgroup_metrics[subgroup_metrics["scope"].eq("hidden_like_spatial")].iloc[0]
    hidden_typewell = subgroup_metrics[
        subgroup_metrics["scope"].eq("hidden_like_typewell_purged")
    ].iloc[0]
    exp301_p95 = float(by_well["exp301_rmse"].quantile(0.95))
    exp226_p95 = float(by_well["exp226_rmse"].quantile(0.95))
    worst_delta = float(by_well["delta_rmse_vs_exp226"].max())
    criteria = get_nested(config, "success_criteria.direct_quality")
    checks = {
        "pooled_rmse_at_most_preregistered_threshold": float(pooled["exp301_rmse"])
        <= float(criteria["maximum_pooled_rmse_ft"]),
        "improves_exp226_all_five_folds": bool((fold_only["delta_rmse_vs_exp226"] < 0.0).all()),
        "distance_1000_plus_nonregression": float(longtail["delta_rmse_vs_exp226"]) <= 0.0,
        "hidden_like_spatial_nonregression": float(hidden_spatial["delta_rmse_vs_exp226"]) <= 0.0,
        "hidden_like_typewell_purged_nonregression": float(
            hidden_typewell["delta_rmse_vs_exp226"]
        )
        <= 0.0,
        "by_well_p95_nonregression": exp301_p95 <= exp226_p95,
        "worst_well_delta_at_most_0p25": worst_delta
        <= float(criteria["maximum_worst_well_delta_ft"]),
    }
    summary = {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "pooled_exp301_rmse": float(pooled["exp301_rmse"]),
        "pooled_exp226_rmse": float(pooled["exp226_rmse"]),
        "pooled_delta_rmse": float(pooled["delta_rmse_vs_exp226"]),
        "exp301_by_well_p95_rmse": exp301_p95,
        "exp226_by_well_p95_rmse": exp226_p95,
        "worst_well_delta_rmse": worst_delta,
    }
    return fold_metrics, subgroup_metrics, by_well, summary


# %% [markdown]
# ## 8. Exp293 fixed-bank H512 add-one novelty diagnostic

# %%
@dataclass
class CandidateBank:
    keys: pd.DataFrame
    candidate_ids: tuple[str, ...]
    values: np.memmap
    values_path: Path
    manifest_path: Path
    candidate_content_sha256: str
    input_evidence: list[dict[str, Any]]


def reject_forbidden_candidate_columns(columns: Iterable[str]) -> None:
    normalized = {str(column) for column in columns}
    forbidden = normalized & {
        "TVT",
        "target",
        "true_tvt",
        "error",
        "abs_error",
        "oracle",
        "oracle_label",
        "oracle_candidate",
    }
    token_forbidden = {
        column
        for column in normalized
        if any(token in column.lower() for token in ("true_tvt", "abs_error", "oracle_label"))
    }
    if forbidden or token_forbidden:
        raise ValueError(
            "candidate partition exposes truth/oracle columns: "
            f"{sorted(forbidden | token_forbidden)}"
        )


def _artifact_path_from_manifest(manifest_path: Path, item: Mapping[str, Any]) -> Path:
    raw = str(item["path"])
    marker = "/artifacts/"
    if marker in raw:
        relative = raw.split(marker, 1)[1]
        candidate = manifest_path.parent / relative
        if candidate.exists():
            return candidate
    direct = Path(raw)
    if direct.exists():
        return direct
    suffix = Path(raw).parts[-4:]
    for root in manifest_path.parents:
        candidate = root.joinpath(*suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest partition is missing: {raw}")


def _read_candidate_partitions(
    manifest_path: Path,
    items: Sequence[Mapping[str, Any]],
    candidate_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for item in items:
        path = _artifact_path_from_manifest(manifest_path, item)
        if item.get("file_sha256") and sha256_file(path) != str(item["file_sha256"]):
            raise ValueError(f"candidate partition file SHA mismatch: {path}")
        full = pd.read_parquet(path)
        reject_forbidden_candidate_columns(full.columns)
        if int(item.get("rows", len(full))) != len(full):
            raise ValueError(f"candidate partition row mismatch: {path}")
        if item.get("schema_sha256") and frame_schema_sha256(full) != str(item["schema_sha256"]):
            raise ValueError(f"candidate partition schema SHA mismatch: {path}")
        if item.get("content_sha256") and frame_content_sha256(full) != str(item["content_sha256"]):
            raise ValueError(f"candidate partition content SHA mismatch: {path}")
        missing = set(EXP263_READ_COLUMNS) - set(full.columns)
        if missing:
            raise ValueError(f"candidate partition lacks columns: {sorted(missing)}")
        frames.append(full[list(EXP263_READ_COLUMNS)].copy())
        evidence.append(
            {
                "phase": "post_freeze_candidate_novelty",
                "source": candidate_id,
                "path": str(path),
                "rows": len(full),
                "file_sha256": sha256_file(path),
                "logical_content_sha256": frame_content_sha256(full),
                "schema_sha256": frame_schema_sha256(full),
            }
        )
    if not frames:
        raise ValueError(f"no candidate partitions for {candidate_id}")
    return pd.concat(frames, ignore_index=True), evidence


def candidate_bank_content_sha256(bank: CandidateBank, chunk_rows: int) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(bank.candidate_ids), separators=(",", ":")).encode())
    digest.update(frame_content_sha256(bank.keys[list(EXP263_KEY_COLUMNS)]).encode())
    for position, candidate_id in enumerate(bank.candidate_ids):
        digest.update(candidate_id.encode())
        for start in range(0, len(bank.keys), chunk_rows):
            end = min(start + chunk_rows, len(bank.keys))
            digest.update(
                np.asarray(bank.values[start:end, position], dtype="<f4").tobytes()
            )
    return digest.hexdigest()


def load_exp293_fixed_bank_after_freeze(
    oof: pd.DataFrame,
    freeze: PredictionFreeze,
    config: Mapping[str, Any],
    work_dir: Path,
) -> CandidateBank:
    verify_prediction_freeze(oof, freeze)
    manifest_cfg = get_nested(config, "data.exp263_manifest")
    manifest_path = resolve_file(
        manifest_cfg["patterns"],
        label="exp263 cache manifest",
        expected_sha256=str(manifest_cfg["expected_file_sha256"]),
    )
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(get_nested(config, "validation.expected_prediction_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        int(manifest.get("rows", -1)) != expected_rows
        or int(manifest.get("wells", -1)) != expected_wells
    ):
        raise ValueError("exp263 manifest row/well contract mismatch")
    if manifest.get("canonical_id_sha256") != manifest_cfg["expected_canonical_id_sha256"]:
        raise ValueError("exp263 canonical ID SHA mismatch")
    audit_cfg = get_nested(config, "candidate_novelty_audit")
    candidate_ids = tuple(str(value) for value in audit_cfg["candidate_order"])
    primitives = tuple(str(value) for value in audit_cfg["primitive_candidates"])
    values_path = work_dir / f"{OUTPUT_PREFIX}_exp293_bank.f32"
    values = np.memmap(
        values_path,
        mode="w+",
        dtype="float32",
        shape=(expected_rows, len(candidate_ids)),
    )
    values[:] = np.nan
    positions = {name: index for index, name in enumerate(candidate_ids)}
    reference: pd.DataFrame | None = None
    evidence: list[dict[str, Any]] = [
        {
            "phase": "post_freeze_candidate_novelty",
            "source": "exp263_manifest",
            "path": str(manifest_path),
            "rows": expected_rows,
            "file_sha256": sha256_file(manifest_path),
            "canonical_id_sha256": manifest.get("canonical_id_sha256"),
        }
    ]
    for candidate_id in primitives:
        items = manifest["candidate_value_partitions"].get(candidate_id)
        if not items or len(items) != int(get_nested(config, "validation.n_folds")):
            raise ValueError(f"{candidate_id} does not have five fixed partitions")
        frame, item_evidence = _read_candidate_partitions(manifest_path, items, candidate_id)
        evidence.extend(item_evidence)
        keys = frame[list(EXP263_KEY_COLUMNS)].copy()
        keys["id"] = keys["id"].astype(str)
        keys["well"] = keys["well"].astype(str)
        if reference is None:
            reference = keys
        elif not reference.equals(keys):
            raise ValueError(f"candidate key identity differs for {candidate_id}")
        available = frame["candidate_available"].astype(bool).to_numpy()
        finite_flag = frame["candidate_finite"].astype(bool).to_numpy()
        candidate = pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(np.float32)
        candidate[~(available & finite_flag)] = np.nan
        values[:, positions[candidate_id]] = candidate
    if reference is None:
        raise AssertionError("candidate bank has no reference keys")
    for output_name, parents in audit_cfg["pairs"].items():
        left, right = [str(value) for value in parents]
        values[:, positions[str(output_name)]] = (
            np.float32(0.5) * (values[:, positions[left]] + values[:, positions[right]])
        ).astype(np.float32)
    fixed = audit_cfg["fixed_formula"]["exp226_w500_50_50"]
    values[:, positions["exp226_w500_50_50"]] = (
        np.float32(fixed["exp226_k16"]) * values[:, positions["exp226_k16"]]
        + np.float32(fixed["likpf_mean"]) * values[:, positions["likpf_mean"]]
        + np.float32(fixed["exact_hmm"]) * values[:, positions["exact_hmm"]]
    ).astype(np.float32)
    values.flush()
    if not np.isfinite(values).all():
        raise ValueError("exp293 fixed bank has non-finite candidate values")
    if reference["id"].duplicated().any():
        raise ValueError("exp293 fixed bank ID is duplicated")
    bank = CandidateBank(
        keys=reference.reset_index(drop=True),
        candidate_ids=candidate_ids,
        values=values,
        values_path=values_path,
        manifest_path=manifest_path,
        candidate_content_sha256="",
        input_evidence=evidence,
    )
    bank.candidate_content_sha256 = candidate_bank_content_sha256(
        bank, int(get_nested(config, "audit.work_chunk_rows"))
    )
    return bank


def build_h512_groups(keys: pd.DataFrame, horizon: int) -> tuple[np.ndarray, pd.DataFrame]:
    wells = keys["well"].astype(str).to_numpy()
    folds = keys["outer_fold"].to_numpy(np.int8)
    if len(wells) == 0:
        raise ValueError("candidate bank keys are empty")
    starts = np.flatnonzero(np.r_[True, wells[1:] != wells[:-1]])
    ends = np.r_[starts[1:], len(wells)]
    if pd.Index(wells[starts]).duplicated().any():
        raise ValueError("candidate bank well rows are not contiguous")
    lengths = ends - starts
    well_codes = np.repeat(np.arange(len(starts), dtype=np.int32), lengths)
    within = np.arange(len(wells), dtype=np.int64) - np.repeat(starts, lengths)
    blocks_per_well = (lengths + int(horizon) - 1) // int(horizon)
    offsets = np.r_[0, np.cumsum(blocks_per_well[:-1])].astype(np.int64)
    codes = (offsets[well_codes] + within // int(horizon)).astype(np.int32)
    n_groups = int(codes.max()) + 1
    first = np.full(n_groups, len(codes), dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(codes), dtype=np.int64))
    group_frame = pd.DataFrame(
        {
            "h512_group": np.arange(n_groups, dtype=np.int32),
            "well_id": wells[first],
            "outer_fold": folds[first],
            "rows": np.bincount(codes, minlength=n_groups).astype(np.int64),
        }
    )
    return codes, group_frame


def evaluate_candidate_novelty(
    bank: CandidateBank,
    oof: pd.DataFrame,
    truth: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexer = pd.Index(oof["id"].astype(str)).get_indexer(bank.keys["id"].astype(str))
    if np.any(indexer < 0):
        raise ValueError("exp301 OOF lacks exp293 bank rows")
    aligned_oof = oof.iloc[indexer].reset_index(drop=True)
    if not np.array_equal(aligned_oof["id"].to_numpy(), bank.keys["id"].to_numpy()):
        raise ValueError("exp301/exp293 identity alignment failed")
    prediction = aligned_oof["tvt_pred_exp301"].to_numpy(np.float64)
    aligned_truth = np.asarray(truth, dtype=np.float64)[indexer]
    horizon = int(get_nested(config, "candidate_novelty_audit.block_horizon_rows"))
    codes, groups = build_h512_groups(bank.keys, horizon)
    n_groups = len(groups)
    bank_sse = np.zeros((n_groups, len(bank.candidate_ids)), dtype=np.float64)
    for position in range(len(bank.candidate_ids)):
        error_squared = np.square(
            np.asarray(bank.values[:, position], dtype=np.float64) - aligned_truth
        )
        bank_sse[:, position] = np.bincount(
            codes, weights=error_squared, minlength=n_groups
        )
    exp301_sse = np.bincount(
        codes, weights=np.square(prediction - aligned_truth), minlength=n_groups
    )
    bank_best = np.min(bank_sse, axis=1)
    add_one_best = np.minimum(bank_best, exp301_sse)
    atol = float(get_nested(config, "candidate_novelty_audit.squared_error_tie_atol"))
    unique_best = exp301_sse < (bank_best - atol)
    records: list[dict[str, Any]] = []
    for scope, mask in [
        ("pooled", np.ones(n_groups, dtype=bool)),
        *[
            (f"fold_{fold}", groups["outer_fold"].to_numpy(np.int8) == fold)
            for fold in range(int(get_nested(config, "validation.n_folds")))
        ],
    ]:
        rows = int(groups.loc[mask, "rows"].sum())
        bank_rmse = float(np.sqrt(bank_best[mask].sum() / rows)) if rows else math.nan
        add_one_rmse = float(np.sqrt(add_one_best[mask].sum() / rows)) if rows else math.nan
        records.append(
            {
                "scope": scope,
                "groups": int(mask.sum()),
                "rows": rows,
                "fixed12_h512_oracle_rmse": bank_rmse,
                "add_exp301_h512_oracle_rmse": add_one_rmse,
                "oracle_rmse_improvement": bank_rmse - add_one_rmse,
                "exp301_strict_unique_best_groups": int(unique_best[mask].sum()),
                "exp301_strict_unique_best_fraction": (
                    float(unique_best[mask].mean()) if mask.any() else math.nan
                ),
            }
        )
    metrics = pd.DataFrame(records)
    pooled = metrics[metrics["scope"].eq("pooled")].iloc[0]
    folds = metrics[metrics["scope"].str.startswith("fold_")]
    criteria = get_nested(config, "success_criteria.candidate_novelty")
    checks = {
        "h512_oracle_improvement_at_least_0p10": float(pooled["oracle_rmse_improvement"])
        >= float(criteria["minimum_h512_oracle_rmse_improvement_ft"]),
        "h512_oracle_improves_at_least_four_folds": int(
            (folds["oracle_rmse_improvement"] > 0.0).sum()
        )
        >= int(criteria["minimum_improved_outer_folds"]),
        "strict_unique_best_block_fraction_at_least_0p02": float(
            pooled["exp301_strict_unique_best_fraction"]
        )
        >= float(criteria["minimum_strict_unique_best_block_fraction"]),
        "oracle_prediction_not_persisted": not bool(
            get_nested(config, "candidate_novelty_audit.persist_oracle_prediction")
        ),
    }
    summary = {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "fixed12_h512_oracle_rmse": float(pooled["fixed12_h512_oracle_rmse"]),
        "add_exp301_h512_oracle_rmse": float(pooled["add_exp301_h512_oracle_rmse"]),
        "oracle_rmse_improvement": float(pooled["oracle_rmse_improvement"]),
        "improved_fold_count": int((folds["oracle_rmse_improvement"] > 0.0).sum()),
        "strict_unique_best_block_fraction": float(
            pooled["exp301_strict_unique_best_fraction"]
        ),
        "candidate_bank_content_sha256": bank.candidate_content_sha256,
        "oracle_prediction_persisted": False,
    }
    return metrics, summary


# %% [markdown]
# ## 9. Generated artifacts, promotion decision, and execution

# %%
def write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def build_contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "validation": get_nested(config, "validation"),
        "physics": get_nested(config, "physics"),
        "stage0": get_nested(config, "stage0"),
        "solver": get_nested(config, "solver"),
        "success_criteria": get_nested(config, "success_criteria"),
        "candidate_novelty_audit": get_nested(config, "candidate_novelty_audit"),
        "execution": get_nested(config, "execution"),
        "forbidden_actions": get_nested(config, "forbidden_actions"),
    }


def persist_stage0_outputs(
    identity: pd.DataFrame,
    support: pd.DataFrame,
    input_evidence: Sequence[Mapping[str, Any]],
    stage0_decision: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> list[Path]:
    contract_path = artifacts_dir / f"{OUTPUT_PREFIX}_contract.json"
    identity_path = artifacts_dir / f"{OUTPUT_PREFIX}_stage0_identity.csv"
    support_path = artifacts_dir / f"{OUTPUT_PREFIX}_stage0_support.csv"
    input_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv"
    stage0_summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_stage0_summary.json"
    write_json(contract_path, build_contract_payload(config))
    write_frame(identity_path, identity)
    write_frame(support_path, support)
    write_frame(input_path, pd.DataFrame(input_evidence))
    write_json(
        stage0_summary_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "stage0_pass" if stage0_decision["passed"] else "stage0_fail_branch_closed",
            "decision": dict(stage0_decision),
            "stage1_executed": False,
            "identity_content_sha256": frame_content_sha256(identity),
            "support_content_sha256": frame_content_sha256(support),
            "input_manifest_content_sha256": frame_content_sha256(pd.DataFrame(input_evidence)),
        },
    )
    return [contract_path, identity_path, support_path, input_path, stage0_summary_path]


def build_sha_manifest(paths: Sequence[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=str):
        if not path.exists() or not path.is_file():
            continue
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": (
                    sha256_decompressed_gzip(path) if path.suffix == ".gz" else None
                ),
            }
        )
    return pd.DataFrame(records)


def persist_final_outputs(
    *,
    identity: pd.DataFrame,
    support: pd.DataFrame,
    input_evidence: Sequence[Mapping[str, Any]],
    stage0_decision: Mapping[str, Any],
    oof: pd.DataFrame,
    freeze: PredictionFreeze,
    lambda_selection: pd.DataFrame,
    solver_manifest: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    novelty_metrics: pd.DataFrame,
    direct_summary: Mapping[str, Any],
    novelty_summary: Mapping[str, Any],
    truth_content_sha256: str,
    total_solver_fits: int,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    existing_paths: Sequence[Path],
) -> tuple[dict[str, Any], list[Path]]:
    paths = list(existing_paths)
    lambda_path = artifacts_dir / f"{OUTPUT_PREFIX}_lambda_selection.csv"
    solver_path = artifacts_dir / f"{OUTPUT_PREFIX}_solver_manifest.csv"
    fold_path = artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv"
    subgroup_path = artifacts_dir / f"{OUTPUT_PREFIX}_subgroup_metrics.csv"
    by_well_path = artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    novelty_path = artifacts_dir / f"{OUTPUT_PREFIX}_candidate_novelty.csv"
    input_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv"
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    write_frame(lambda_path, lambda_selection)
    write_frame(solver_path, solver_manifest)
    write_frame(fold_path, fold_metrics)
    write_frame(subgroup_path, subgroup_metrics)
    write_frame(by_well_path, by_well)
    write_frame(novelty_path, novelty_metrics)
    write_frame(input_path, pd.DataFrame(input_evidence))
    paths.extend(
        [
            freeze.prediction_path,
            freeze.manifest_path,
            lambda_path,
            solver_path,
            fold_path,
            subgroup_path,
            by_well_path,
            novelty_path,
            input_path,
        ]
    )
    stage1_technical_checks = {
        "prediction_rows_exact": freeze.rows
        == int(get_nested(config, "success_criteria.stage1_technical.expected_prediction_rows")),
        "prediction_wells_exact": freeze.wells
        == int(get_nested(config, "success_criteria.stage1_technical.expected_wells")),
        "duplicate_rows_zero": freeze.duplicate_rows
        == int(get_nested(config, "success_criteria.stage1_technical.required_duplicate_rows")),
        "finite_prediction_coverage_one": math.isclose(
            freeze.finite_coverage,
            float(
                get_nested(
                    config,
                    "success_criteria.stage1_technical.required_finite_prediction_coverage",
                )
            ),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "truth_access_before_freeze_zero": freeze.truth_access_count_before_freeze == 0,
        "solver_fit_guard": total_solver_fits
        <= int(get_nested(config, "solver.runtime_guards.maximum_total_solver_fits")),
    }
    stage1_technical_pass = bool(all(stage1_technical_checks.values()))
    final_pass = bool(
        stage0_decision["passed"]
        and stage1_technical_pass
        and direct_summary["passed"]
        and novelty_summary["passed"]
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage1_pass_eligible_for_separate_inference_review"
        if final_pass
        else "stage1_fail_branch_closed_no_same_oof_rescue",
        "created_at": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "stage0": dict(stage0_decision),
        "stage1_technical": {
            "checks": stage1_technical_checks,
            "passed": stage1_technical_pass,
        },
        "direct_quality": dict(direct_summary),
        "candidate_novelty": dict(novelty_summary),
        "promotion_passed": final_pass,
        "inference_enabled": False,
        "submission_created": False,
        "same_oof_rescue_allowed": False,
        "solver_fits": total_solver_fits,
        "prediction_logical_content_sha256": freeze.logical_content_sha256,
        "prediction_file_sha256": freeze.file_sha256,
        "prediction_decompressed_content_sha256": freeze.decompressed_content_sha256,
        "truth_content_sha256": truth_content_sha256,
        "lambda_selection_content_sha256": frame_content_sha256(lambda_selection),
        "solver_manifest_content_sha256": frame_content_sha256(solver_manifest),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "process_count": 1,
        },
    }
    write_json(summary_path, summary)
    paths.append(summary_path)
    sha_path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    write_frame(sha_path, build_sha_manifest(paths))
    paths.append(sha_path)
    return summary, paths


def write_runtime_metrics(summary: Mapping[str, Any], stage0_only: bool = False) -> None:
    if stage0_only:
        payload = {
            "experiment": EXPERIMENT_NAME,
            "status": "stage0_fail_branch_closed",
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": "rmse",
            "stage0_passed": False,
            "stage1_executed": False,
            "total_boosters": 0,
            "inference": False,
            "submission": False,
        }
    else:
        payload = {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "cv": summary["direct_quality"]["pooled_exp301_rmse"],
            "public_lb": None,
            "private_lb": None,
            "metric": "rmse",
            "stage0_passed": summary["stage0"]["passed"],
            "stage1_technical_passed": summary["stage1_technical"]["passed"],
            "direct_quality_passed": summary["direct_quality"]["passed"],
            "candidate_novelty_passed": summary["candidate_novelty"]["passed"],
            "promotion_passed": summary["promotion_passed"],
            "prediction_content_sha256": summary["prediction_logical_content_sha256"],
            "total_boosters": 0,
            "inference": False,
            "submission": False,
        }
    write_json(runtime_metrics_path(), payload)


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config)
    if not bool(get_nested(config, "execution.kaggle_execution_authorized")):
        raise RuntimeError(
            "exp301 implementation is complete, but Kaggle execution needs separate user approval"
        )
    artifacts_dir = runtime_artifacts_dir()
    work_dir = runtime_work_dir()
    fold_identity = load_fold_identity(config)
    expected_wells = set(fold_identity.by_well)
    _, well_paths = resolve_raw_directory(
        get_nested(config, "data.raw_train_dir_patterns"), expected_wells=expected_wells
    )
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Comparison anchors:", get_nested(config, "lineage.comparisons"))
    print("Execution contract: 1 variant / 5 outer folds / 3 lambdas / 0 boosters")

    identity, support, input_evidence, stage0_decision = run_stage0(
        fold_identity, well_paths, config
    )
    stage0_paths = persist_stage0_outputs(
        identity, support, input_evidence, stage0_decision, config, artifacts_dir
    )
    print("Stage 0 decision:", json.dumps(stage0_decision, indent=2))
    if not stage0_decision["passed"]:
        sha_path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"
        write_frame(sha_path, build_sha_manifest(stage0_paths))
        write_runtime_metrics(stage0_decision, stage0_only=True)
        return {
            "status": "stage0_fail_branch_closed",
            "stage0": stage0_decision,
            "stage1_executed": False,
        }

    oof, lambda_selection, solver_manifest, stage1_evidence, total_fits = run_stage1(
        fold_identity, well_paths, config, artifacts_dir
    )
    input_evidence.extend(stage1_evidence)
    freeze = freeze_oof_prediction(oof, artifacts_dir, config)
    truth, truth_evidence, truth_sha = load_truth_after_prediction_freeze(
        oof, freeze, well_paths
    )
    input_evidence.extend(truth_evidence)
    baseline, baseline_evidence = load_exp226_comparison_after_freeze(
        oof, truth, freeze, fold_identity, config
    )
    input_evidence.append(baseline_evidence)
    hidden_sets, hidden_evidence = load_hidden_like_sets(config, expected_wells)
    input_evidence.append(hidden_evidence)
    prediction = oof["tvt_pred_exp301"].to_numpy(np.float64)
    fold_metrics, subgroup_metrics, by_well, direct_summary = build_direct_readouts(
        oof, prediction, baseline, truth, hidden_sets, config
    )

    bank = load_exp293_fixed_bank_after_freeze(oof, freeze, config, work_dir)
    input_evidence.extend(bank.input_evidence)
    novelty_metrics, novelty_summary = evaluate_candidate_novelty(
        bank, oof, truth, config
    )
    summary, _ = persist_final_outputs(
        identity=identity,
        support=support,
        input_evidence=input_evidence,
        stage0_decision=stage0_decision,
        oof=oof,
        freeze=freeze,
        lambda_selection=lambda_selection,
        solver_manifest=solver_manifest,
        fold_metrics=fold_metrics,
        subgroup_metrics=subgroup_metrics,
        by_well=by_well,
        novelty_metrics=novelty_metrics,
        direct_summary=direct_summary,
        novelty_summary=novelty_summary,
        truth_content_sha256=truth_sha,
        total_solver_fits=total_fits,
        config=config,
        artifacts_dir=artifacts_dir,
        existing_paths=stage0_paths,
    )
    write_runtime_metrics(summary)
    print("Final promotion decision:", json.dumps(summary, indent=2))
    return summary


# %%
CONFIG = read_yaml(find_config_path())
validate_execution_contract(CONFIG)
print(
    {
        "experiment": get_nested(CONFIG, "experiment.name"),
        "route": get_nested(CONFIG, "experiment.route"),
        "phase": get_nested(CONFIG, "experiment.phase"),
        "implementation_authorized": get_nested(CONFIG, "execution.implementation_authorized"),
        "kaggle_execution_authorized": get_nested(CONFIG, "execution.kaggle_execution_authorized"),
        "outer_folds": get_nested(CONFIG, "validation.n_folds"),
        "lambda_candidates": get_nested(CONFIG, "solver.lambda_candidates"),
        "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
        "boosters": get_nested(CONFIG, "execution.total_boosters"),
    }
)


# %%
if EXECUTE_NOTEBOOK:
    RESULT = run_experiment(CONFIG)
