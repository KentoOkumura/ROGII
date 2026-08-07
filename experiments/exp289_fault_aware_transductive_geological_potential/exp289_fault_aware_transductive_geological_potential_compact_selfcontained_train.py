# %% [markdown]
# # exp289 fault-aware transductive geological potential — Stage 0
#
# This zero-booster notebook implements only the pre-registered Stage 0
# fault-topology association readout. Outer-valid formation columns and true
# suffix TVT are never read before target-free risk tables are frozen and
# content-hashed. Stage 1 MAP fitting is deliberately not implemented here.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Fold-safe source and target input helpers
# 4. Deterministic node sampling and cross-well graph helpers
# 5. Target-free fault-risk construction and freeze helpers
# 6. Post-freeze exp226 and formation-identity readouts
# 7. Stage 0 scientific guard and generated artifacts
# 8. Setup and contract preview
# 9. Run the fixed Kaggle CPU readout

# %% [markdown]
# ## 1. Imports and fixed experiment contract

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

EXPERIMENT_NAME = "exp289_fault_aware_transductive_geological_potential"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
SOURCE_COLUMNS = ("MD", "X", "Y", "ANCC")
TARGET_SAFE_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TARGET_FORBIDDEN_COLUMNS = {
    "TVT",
    *FORMATION_COLUMNS,
    "tvt_true",
    "tvt_pred",
    "tvt_geop",
    "gr_delta",
    "error",
    "abs_error",
    "target",
    "oracle_rank",
}
FROZEN_HASH_KEYS = ("graph_manifest", "node_risk", "well_risk")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP289_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) if path.exists() else {}
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp289 config not found: {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for value in candidates:
        candidate = Path(str(value))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    payload = [(column, str(frame[column].dtype)) for column in chosen]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def dataframe_content_sha256(
    frame: pd.DataFrame, columns: Sequence[str] | None = None, chunk_rows: int = 20_000
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for start in range(0, len(frame), chunk_rows):
        payload = frame.iloc[start : start + chunk_rows][chosen].to_csv(
            index=False,
            header=start == 0,
            float_format="%.17g",
            lineterminator="\n",
            na_rep="",
        )
        digest.update(payload.encode())
    if len(frame) == 0:
        digest.update(pd.DataFrame(columns=chosen).to_csv(index=False).encode())
    return digest.hexdigest()


def typed_array_sha256(*arrays: np.ndarray, context: Sequence[str] = ()) -> str:
    digest = hashlib.sha256()
    for item in context:
        digest.update(str(item).encode())
        digest.update(b"\0")
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_frame(frame: pd.DataFrame, path: Path, *, gzip_output: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        frame.to_csv(
            path,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        logical_sha = dataframe_content_sha256(frame)
        return {
            "path": str(path),
            "rows": len(frame),
            "raw_sha256": sha256_path(path),
            "logical_content_sha256": logical_sha,
            "schema_sha256": schema_sha256(frame),
        }
    frame.to_csv(path, index=False, float_format="%.17g")
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "logical_content_sha256": dataframe_content_sha256(frame),
        "schema_sha256": schema_sha256(frame),
    }


def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    exact = {
        "experiment.route": "pf_beam",
        "validation.n_splits": 5,
        "validation.expected_rows": 3_783_989,
        "validation.expected_wells": 773,
        "physics.graph.row_stride": 16,
        "physics.graph.spatial_neighbors": 12,
        "physics.graph.missing_source_formation_policy": (
            "exclude_well_if_all_nonfinite_fail_on_partial"
        ),
        "stages.stage0.active_variants": 1,
        "stages.stage0.ml_configs": 0,
        "stages.stage0.trained_folds": 0,
        "stages.stage0.boosters": 0,
        "stages.stage0.control_retraining": 0,
        "stages.stage0.topology_contract.primary_well_risk": "suffix_fault_risk_p90",
        "stages.stage0.topology_contract.robust_z_cutoff": 4.0,
        "stages.stage0.topology_contract.fault_risk_threshold": 0.5,
        "execution.active_audit_variants": 1,
        "execution.lightgbm_config_count": 0,
        "execution.trained_fold_count": 0,
        "execution.total_boosters": 0,
        "execution.control_or_parent_retraining": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for dotted_key, expected in exact.items():
        actual = get_nested(config, dotted_key)
        if actual != expected:
            raise ValueError(f"fixed exp289 contract mismatch: {dotted_key}={actual!r}")
    if tuple(get_nested(config, "validation.expected_folds")) != (0, 1, 2, 3, 4):
        raise ValueError("exp289 fixes the exp226 five-fold identity")
    if list(get_nested(config, "data.auxiliary_formation_audit")) != list(FORMATION_COLUMNS[1:]):
        raise ValueError("exp289 fixes the six formation identity surfaces")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp289 Stage 0 implementation must be explicitly approved")


# %% [markdown]
# ## 3. Fold-safe source and target input helpers


# %%
def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def list_horizontal_paths(raw_dir: Path) -> dict[str, Path]:
    paths = {
        well_id_from_path(path): path for path in sorted(raw_dir.glob("*__horizontal_well.csv"))
    }
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate horizontal well ids")
    return paths


def validate_target_safe_frame(frame: pd.DataFrame) -> None:
    forbidden = sorted(TARGET_FORBIDDEN_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(f"target-safe frame contains forbidden columns: {forbidden}")
    if tuple(frame.columns) != TARGET_SAFE_COLUMNS:
        raise ValueError(
            f"target-safe frame schema mismatch: {tuple(frame.columns)} != {TARGET_SAFE_COLUMNS}"
        )


def _numeric_sorted_frame(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(columns))
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"missing columns in {path}: {sorted(set(columns) - set(frame.columns))}")
    frame = frame[list(columns)].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.insert(0, "original_row", np.arange(len(frame), dtype=np.int64))
    frame = frame.sort_values(["MD", "original_row"], kind="mergesort").reset_index(drop=True)
    if not np.isfinite(frame[["MD", "X", "Y"]].to_numpy(np.float64)).all():
        raise ValueError(f"non-finite geometry in {path}")
    return frame


def load_source_horizontal(path: Path) -> pd.DataFrame:
    frame = _numeric_sorted_frame(path, SOURCE_COLUMNS)
    finite = np.isfinite(frame["ANCC"].to_numpy(np.float64))
    if finite.all():
        return frame
    if finite.any():
        raise ValueError(f"partially non-finite outer-train ANCC in {path}")
    # Some competition train wells have no formation pick for ANCC at any row.
    # They cannot act as source donors, but remain valid fold targets because the
    # target-safe reader never materializes ANCC.
    frame = frame.iloc[0:0].copy()
    return frame


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    frame = _numeric_sorted_frame(path, TARGET_SAFE_COLUMNS)
    original_rows = frame.pop("original_row").to_numpy(np.int64)
    frame.attrs["original_rows"] = original_rows
    validate_target_safe_frame(frame)
    tvt_input = frame["TVT_input"].to_numpy(np.float64)
    finite = np.isfinite(tvt_input)
    if not finite.any():
        raise ValueError(f"target well has no known TVT_input prefix: {path}")
    last_known = int(np.flatnonzero(finite)[-1])
    if not finite[: last_known + 1].all() or finite[last_known + 1 :].any():
        raise ValueError(f"TVT_input is not one contiguous prefix: {path}")
    return frame


def load_fold_identity(
    config: Mapping[str, Any],
) -> tuple[dict[str, int], Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof")
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    safe_columns = [str(value) for value in spec["fold_columns"]]
    if TARGET_FORBIDDEN_COLUMNS.intersection(safe_columns):
        raise ValueError("fold reader allowlist contains forbidden exp226 columns")
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row count mismatch")
    by_well = frame.drop_duplicates().sort_values("well_id", kind="mergesort")
    if by_well["well_id"].duplicated().any():
        raise ValueError("exp226 OOF maps one well to multiple folds")
    mapping = {str(row.well_id): int(row.fold) for row in by_well.itertuples(index=False)}
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(set(mapping.values())) != expected_folds:
        raise ValueError("exp226 OOF fold set mismatch")
    manifest = {
        "name": "exp226_oof_fold_identity",
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": len(mapping),
        "columns_read": safe_columns,
        "forbidden_column_hits": 0,
    }
    return mapping, path, manifest


# %% [markdown]
# ## 4. Deterministic node sampling and cross-well graph helpers


# %%
def turning_point_indices(x: np.ndarray, y: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    window = int(get_nested(config, "physics.graph.turning_point_window_rows"))
    threshold = float(get_nested(config, "physics.graph.turning_point_angle_deg"))
    minimum_separation = int(get_nested(config, "physics.graph.turning_point_min_separation_rows"))
    if len(x) < 2 * window + 1:
        return np.empty(0, dtype=np.int64)
    centers = np.arange(window, len(x) - window, dtype=np.int64)
    left = np.column_stack((x[centers] - x[centers - window], y[centers] - y[centers - window]))
    right = np.column_stack((x[centers + window] - x[centers], y[centers + window] - y[centers]))
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid = (left_norm > 0.0) & (right_norm > 0.0)
    cosine = np.ones(len(centers), dtype=np.float64)
    cosine[valid] = np.sum(left[valid] * right[valid], axis=1) / (
        left_norm[valid] * right_norm[valid]
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    candidates = centers[valid & (angle >= threshold)]
    selected: list[int] = []
    for index in candidates:
        if not selected or int(index) - selected[-1] >= minimum_separation:
            selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def sampled_row_indices(frame: pd.DataFrame, config: Mapping[str, Any]) -> np.ndarray:
    stride = int(get_nested(config, "physics.graph.row_stride"))
    indices: set[int] = set(range(0, len(frame), stride))
    indices.add(len(frame) - 1)
    turns = turning_point_indices(
        frame["X"].to_numpy(np.float64), frame["Y"].to_numpy(np.float64), config
    )
    indices.update(int(value) for value in turns)
    if "TVT_input" in frame:
        finite = np.isfinite(frame["TVT_input"].to_numpy(np.float64))
        if finite.any():
            indices.add(int(np.flatnonzero(finite)[-1]))
    return np.asarray(sorted(indices), dtype=np.int64)


def build_source_nodes(
    well_paths: Mapping[str, Path], wells: Sequence[str], config: Mapping[str, Any]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well in sorted(wells):
        frame = load_source_horizontal(well_paths[well])
        if frame.empty:
            continue
        indices = sampled_row_indices(frame, config)
        part = frame.iloc[indices][["original_row", "MD", "X", "Y", "ANCC"]].copy()
        part.insert(0, "well_id", str(well))
        part = part.rename(columns={"original_row": "row_idx", "ANCC": "formation"})
        parts.append(part)
    if not parts:
        raise ValueError("source graph has no wells with finite ANCC")
    output = pd.concat(parts, ignore_index=True)
    output = output.sort_values(["well_id", "MD", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if output[["well_id", "row_idx"]].duplicated().any():
        raise ValueError("duplicate source graph nodes")
    return output


def build_target_nodes(
    well_paths: Mapping[str, Path], wells: Sequence[str], config: Mapping[str, Any]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well in sorted(wells):
        frame = load_target_safe_horizontal(well_paths[well])
        indices = sampled_row_indices(frame, config)
        part = frame.iloc[indices].copy()
        validate_target_safe_frame(part)
        original_rows = np.asarray(frame.attrs["original_rows"], dtype=np.int64)
        part.insert(0, "row_idx", original_rows[indices])
        part.insert(0, "well_id", str(well))
        part["is_known_prefix"] = np.isfinite(part["TVT_input"].to_numpy(np.float64))
        # pandas concat compares attrs across inputs. The per-well original-row
        # arrays have different lengths and have already been materialized above.
        part.attrs.clear()
        parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    output = output.sort_values(["well_id", "MD", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if TARGET_FORBIDDEN_COLUMNS.intersection(output.columns):
        raise ValueError("target graph nodes contain forbidden columns")
    if output[["well_id", "row_idx"]].duplicated().any():
        raise ValueError("duplicate target graph nodes")
    return output


def robust_location_scale(values: Sequence[float], floor: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("cannot estimate robust scale from no finite values")
    location = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - location)))
    return location, max(scale, float(floor))


def standardize_xy(
    source_nodes: pd.DataFrame, target_nodes: pd.DataFrame, floor: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source_xy = source_nodes[["X", "Y"]].to_numpy(np.float64)
    target_xy = target_nodes[["X", "Y"]].to_numpy(np.float64)
    center = np.median(source_xy, axis=0)
    scale = 1.4826 * np.median(np.abs(source_xy - center), axis=0)
    scale = np.maximum(scale, float(floor))
    manifest = {
        "x_center": float(center[0]),
        "y_center": float(center[1]),
        "x_scale": float(scale[0]),
        "y_scale": float(scale[1]),
    }
    return (source_xy - center) / scale, (target_xy - center) / scale, manifest


def query_cross_well_neighbors(
    query_xy: np.ndarray,
    query_wells: Sequence[str],
    source_xy: np.ndarray,
    source_wells: Sequence[str],
    source_rows: np.ndarray,
    config: Mapping[str, Any],
    *,
    exclude_same_well: bool,
) -> tuple[np.ndarray, np.ndarray]:
    k = int(get_nested(config, "physics.graph.spatial_neighbors"))
    initial = int(get_nested(config, "physics.graph.initial_cross_well_query_neighbors"))
    maximum = int(get_nested(config, "physics.graph.maximum_cross_well_query_neighbors"))
    batch_rows = int(get_nested(config, "physics.graph.query_batch_rows"))
    if len(source_xy) <= k:
        raise ValueError("not enough source nodes for fixed kNN graph")
    tree = cKDTree(source_xy, compact_nodes=True, balanced_tree=True)
    source_well_values = np.asarray(source_wells, dtype=str)
    ordered_wells = sorted(set(source_well_values).union(str(value) for value in query_wells))
    well_rank = {well: index for index, well in enumerate(ordered_wells)}
    source_rank = np.asarray([well_rank[value] for value in source_well_values], dtype=np.int64)
    query_rank = np.asarray([well_rank[str(value)] for value in query_wells], dtype=np.int64)
    output_indices = np.empty((len(query_xy), k), dtype=np.int64)
    output_distances = np.empty((len(query_xy), k), dtype=np.float64)

    for start in range(0, len(query_xy), batch_rows):
        stop = min(start + batch_rows, len(query_xy))
        unresolved = np.arange(stop - start, dtype=np.int64)
        count = min(max(initial, 2 * k), len(source_xy))
        while len(unresolved):
            distances, indices = tree.query(query_xy[start:stop][unresolved], k=count, workers=1)
            distances = np.atleast_2d(np.asarray(distances, dtype=np.float64))
            indices = np.atleast_2d(np.asarray(indices, dtype=np.int64))
            if exclude_same_well:
                invalid = source_rank[indices] == query_rank[start:stop][unresolved, None]
                distances = distances.copy()
                distances[invalid] = np.inf
            order = np.lexsort((source_rows[indices], source_rank[indices], distances), axis=1)
            selected = order[:, :k]
            chosen_distances = np.take_along_axis(distances, selected, axis=1)
            chosen_indices = np.take_along_axis(indices, selected, axis=1)
            complete = np.isfinite(chosen_distances[:, -1])
            resolved_local = unresolved[complete]
            output_indices[start + resolved_local] = chosen_indices[complete]
            output_distances[start + resolved_local] = chosen_distances[complete]
            unresolved = unresolved[~complete]
            if len(unresolved) == 0:
                break
            if count >= min(maximum, len(source_xy)):
                raise ValueError(
                    f"could not find {k} cross-well neighbors for {len(unresolved)} nodes"
                )
            count = min(maximum, len(source_xy), count * 2)
    return output_indices, output_distances


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    threshold = 0.5 * cumulative[:, -1]
    position = np.argmax(cumulative >= threshold[:, None], axis=1)
    return sorted_values[np.arange(len(values)), position]


def neighbor_surface_and_spread(
    formation: np.ndarray, neighbor_indices: np.ndarray, neighbor_distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = formation[neighbor_indices]
    positive = neighbor_distances[neighbor_distances > 0.0]
    distance_floor = float(np.median(positive)) * 1.0e-6 if len(positive) else 1.0e-12
    weights = 1.0 / np.maximum(neighbor_distances, max(distance_floor, 1.0e-12))
    surface = weighted_median(values, weights)
    spread = weighted_median(np.abs(values - surface[:, None]), weights)
    return surface, spread


def source_jump_rates(nodes: pd.DataFrame) -> np.ndarray:
    values: list[np.ndarray] = []
    for _, part in nodes.groupby("well_id", sort=True):
        md = part["MD"].to_numpy(np.float64)
        formation = part["formation"].to_numpy(np.float64)
        if len(part) < 2:
            continue
        delta_md = np.maximum(np.abs(np.diff(md)), 1.0e-9)
        values.append(np.abs(np.diff(formation)) / delta_md)
    if not values:
        raise ValueError("no source along-well jump rates")
    return np.concatenate(values)


def build_fold_graph_calibration(
    source_nodes: pd.DataFrame,
    target_nodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    floor = float(get_nested(config, "stages.stage0.topology_contract.scale_floor"))
    source_xy, target_xy, xy_manifest = standardize_xy(source_nodes, target_nodes, floor)
    source_wells = source_nodes["well_id"].astype(str).to_numpy()
    source_rows = source_nodes["row_idx"].to_numpy(np.int64)
    source_indices, source_distances = query_cross_well_neighbors(
        source_xy,
        source_wells,
        source_xy,
        source_wells,
        source_rows,
        config,
        exclude_same_well=True,
    )
    formation = source_nodes["formation"].to_numpy(np.float64)
    source_surface, source_spread = neighbor_surface_and_spread(
        formation, source_indices, source_distances
    )
    source_residual = np.abs(formation - source_surface)
    jump_rates = source_jump_rates(source_nodes)
    residual_location, residual_scale = robust_location_scale(source_residual, floor)
    spread_location, spread_scale = robust_location_scale(source_spread, floor)
    jump_location, jump_scale = robust_location_scale(jump_rates, floor)
    calibration = {
        "source_residual_location": residual_location,
        "source_residual_scale": residual_scale,
        "source_spread_location": spread_location,
        "source_spread_scale": spread_scale,
        "source_jump_rate_location": jump_location,
        "source_jump_rate_scale": jump_scale,
        **xy_manifest,
    }
    target_wells = target_nodes["well_id"].astype(str).to_numpy()
    target_indices, target_distances = query_cross_well_neighbors(
        target_xy,
        target_wells,
        source_xy,
        source_wells,
        source_rows,
        config,
        exclude_same_well=False,
    )
    edge_manifest = {
        "source_edge_content_sha256": typed_array_sha256(
            source_indices,
            source_distances,
            context=("source_cross_well", dataframe_content_sha256(source_nodes)),
        ),
        "target_edge_content_sha256": typed_array_sha256(
            target_indices,
            target_distances,
            context=(
                "target_to_source",
                dataframe_content_sha256(source_nodes),
                dataframe_content_sha256(target_nodes),
            ),
        ),
    }
    return (
        calibration,
        target_indices,
        target_distances,
        source_indices,
        source_distances,
        edge_manifest,
    )


# %% [markdown]
# ## 5. Target-free fault-risk construction and freeze helpers


# %%
def risk_transform(values: np.ndarray, location: float, scale: float, cutoff: float) -> np.ndarray:
    z = np.maximum(0.0, (np.asarray(values, dtype=np.float64) - location) / scale)
    return 1.0 - 1.0 / (1.0 + np.square(z / cutoff))


def count_true_episodes(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def build_target_fault_risk(
    fold: int,
    source_nodes: pd.DataFrame,
    target_nodes: pd.DataFrame,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    calibration: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    formation = source_nodes["formation"].to_numpy(np.float64)
    surface, spread = neighbor_surface_and_spread(formation, neighbor_indices, neighbor_distances)
    cutoff = float(get_nested(config, "stages.stage0.topology_contract.robust_z_cutoff"))
    threshold = float(get_nested(config, "stages.stage0.topology_contract.fault_risk_threshold"))
    spread_risk = risk_transform(
        spread,
        calibration["source_spread_location"],
        calibration["source_spread_scale"],
        cutoff,
    )
    output = target_nodes[["well_id", "row_idx", "MD", "is_known_prefix"]].copy()
    output.insert(0, "fold", int(fold))
    output["donor_surface"] = surface
    output["donor_spread"] = spread
    output["donor_spread_risk"] = spread_risk
    output["trajectory_jump_rate"] = 0.0
    output["trajectory_jump_risk"] = 0.0
    output["known_prefix_misfit"] = np.nan
    output["known_prefix_misfit_risk"] = 0.0
    output["prefix_risk_baseline"] = 0.0
    output["fault_risk"] = 0.0
    output["fault_cut_weight"] = 1.0
    by_well_rows: list[dict[str, Any]] = []

    for well, indices in output.groupby("well_id", sort=True).groups.items():
        positions = np.asarray(sorted(indices), dtype=np.int64)
        part = target_nodes.iloc[positions]
        md = part["MD"].to_numpy(np.float64)
        local_surface = surface[positions]
        jump_rate = np.zeros(len(positions), dtype=np.float64)
        if len(positions) > 1:
            jump_rate[1:] = np.abs(np.diff(local_surface)) / np.maximum(np.abs(np.diff(md)), 1.0e-9)
        jump_risk = risk_transform(
            jump_rate,
            calibration["source_jump_rate_location"],
            calibration["source_jump_rate_scale"],
            cutoff,
        )
        known = part["is_known_prefix"].to_numpy(bool)
        q = part["Z"].to_numpy(np.float64) + part["TVT_input"].to_numpy(np.float64)
        if not known.any() or not np.isfinite(q[known]).all():
            raise ValueError(f"target prefix observation is invalid for {well}")
        datum = float(np.median(local_surface[known] - q[known]))
        prefix_misfit = np.full(len(positions), np.nan, dtype=np.float64)
        prefix_misfit[known] = np.abs(local_surface[known] - (q[known] + datum))
        prefix_risk = np.zeros(len(positions), dtype=np.float64)
        prefix_risk[known] = risk_transform(
            prefix_misfit[known],
            calibration["source_residual_location"],
            calibration["source_residual_scale"],
            cutoff,
        )
        prefix_baseline = float(np.quantile(prefix_risk[known], 0.90))
        total_risk = np.maximum.reduce(
            (spread_risk[positions], jump_risk, np.full(len(positions), prefix_baseline))
        )
        output.loc[positions, "trajectory_jump_rate"] = jump_rate
        output.loc[positions, "trajectory_jump_risk"] = jump_risk
        output.loc[positions, "known_prefix_misfit"] = prefix_misfit
        output.loc[positions, "known_prefix_misfit_risk"] = prefix_risk
        output.loc[positions, "prefix_risk_baseline"] = prefix_baseline
        output.loc[positions, "fault_risk"] = total_risk
        output.loc[positions, "fault_cut_weight"] = 1.0 - total_risk

        anchor_local = int(np.flatnonzero(known)[-1])
        anchor_md = float(md[anchor_local])
        suffix = np.arange(len(positions)) > anchor_local
        suffix_risk = total_risk[suffix]
        if len(suffix_risk) == 0:
            raise ValueError(f"target well has no sampled suffix nodes: {well}")
        cut_mask = suffix_risk >= threshold
        suffix_md = md[suffix] - anchor_md
        first_cut_distance = (
            float(suffix_md[np.flatnonzero(cut_mask)[0]]) if cut_mask.any() else float("nan")
        )
        by_well_rows.append(
            {
                "fold": int(fold),
                "well_id": str(well),
                "sampled_nodes": len(positions),
                "known_nodes": int(known.sum()),
                "suffix_nodes": int(suffix.sum()),
                "anchor_row_idx": int(part.iloc[anchor_local]["row_idx"]),
                "anchor_md": anchor_md,
                "estimated_prefix_datum": datum,
                "prefix_fault_risk_p90": prefix_baseline,
                "suffix_fault_risk_mean": float(np.mean(suffix_risk)),
                "suffix_fault_risk_p90": float(np.quantile(suffix_risk, 0.90)),
                "suffix_fault_risk_max": float(np.max(suffix_risk)),
                "suffix_fault_cut_count": count_true_episodes(cut_mask),
                "suffix_fault_cut_node_fraction": float(np.mean(cut_mask)),
                "first_suffix_fault_cut_md_since_anchor": first_cut_distance,
                "truth_attached": False,
                "forbidden_column_hits": 0,
            }
        )
    risk_columns = [
        "donor_surface",
        "donor_spread",
        "donor_spread_risk",
        "trajectory_jump_rate",
        "trajectory_jump_risk",
        "known_prefix_misfit_risk",
        "prefix_risk_baseline",
        "fault_risk",
        "fault_cut_weight",
    ]
    if not np.isfinite(output[risk_columns].fillna(0.0).to_numpy(np.float64)).all():
        raise ValueError("non-finite target-free fault risk")
    return output, pd.DataFrame(by_well_rows)


def freeze_target_free_outputs(
    graph_manifest: pd.DataFrame, node_risk: pd.DataFrame, well_risk: pd.DataFrame
) -> dict[str, str]:
    if bool(well_risk["truth_attached"].astype(bool).any()):
        raise ValueError("truth was attached before target-free risk freeze")
    if int(graph_manifest["target_forbidden_column_hits"].sum()) != 0:
        raise ValueError("forbidden outer-valid column reached the graph")
    ordered_graph = graph_manifest.sort_values("fold", kind="mergesort").reset_index(drop=True)
    ordered_node = node_risk.sort_values(
        ["fold", "well_id", "MD", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    ordered_well = well_risk.sort_values(["fold", "well_id"], kind="mergesort").reset_index(
        drop=True
    )
    return {
        "graph_manifest": dataframe_content_sha256(ordered_graph),
        "node_risk": dataframe_content_sha256(ordered_node),
        "well_risk": dataframe_content_sha256(ordered_well),
    }


def require_frozen_hashes(frozen_hashes: Mapping[str, str]) -> None:
    missing = [key for key in FROZEN_HASH_KEYS if not frozen_hashes.get(key)]
    if missing:
        raise ValueError(f"post-freeze readout requires frozen content SHA: {missing}")


# %% [markdown]
# ## 6. Post-freeze exp226 and formation-identity readouts


# %%
def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = pd.Series(np.asarray(left, dtype=np.float64))
    y = pd.Series(np.asarray(right, dtype=np.float64))
    finite = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    return pearson_correlation(
        x.loc[finite].rank(method="average").to_numpy(np.float64),
        y.loc[finite].rank(method="average").to_numpy(np.float64),
    )


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    score = pd.Series(np.asarray(scores, dtype=np.float64))
    finite = np.isfinite(score.to_numpy(np.float64))
    y = y[finite]
    score = score.loc[finite].reset_index(drop=True)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = score.rank(method="average").to_numpy(np.float64)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def load_exp226_bias_readout(
    oof_path: Path,
    fold_by_well: Mapping[str, int],
    well_risk: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    frozen_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_frozen_hashes(frozen_hashes)
    columns = [str(value) for value in get_nested(config, "data.exp226_oof.post_freeze_columns")]
    frame = pd.read_csv(oof_path, usecols=columns, dtype={"well_id": str})
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("post-freeze exp226 OOF row count mismatch")
    frame["error"] = pd.to_numeric(frame["error"], errors="coerce")
    if not np.isfinite(frame["error"].to_numpy(np.float64)).all():
        raise ValueError("non-finite exp226 OOF error")
    by_well = (
        frame.groupby(["well_id", "fold"], sort=True)["error"]
        .agg(rows="size", bias="mean", mse=lambda values: float(np.mean(np.square(values))))
        .reset_index()
    )
    by_well["rmse"] = np.sqrt(by_well["mse"])
    by_well["abs_bias"] = np.abs(by_well["bias"])
    by_well["abs_bias_ge_10"] = by_well["abs_bias"] >= 10.0
    for row in by_well.itertuples(index=False):
        if int(row.fold) != int(fold_by_well[str(row.well_id)]):
            raise ValueError(f"exp226 fold identity mismatch for {row.well_id}")
    merged = well_risk.merge(by_well, on=["well_id", "fold"], validate="one_to_one")
    if len(merged) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("risk/bias well coverage mismatch")
    primary = str(get_nested(config, "stages.stage0.topology_contract.primary_well_risk"))
    metric_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", merged)]
    scopes.extend((f"fold_{int(fold)}", part) for fold, part in merged.groupby("fold", sort=True))
    for scope, part in scopes:
        metric_rows.append(
            {
                "scope": scope,
                "fold": -1 if scope == "overall" else int(scope.split("_")[1]),
                "wells": len(part),
                "positive_wells": int(part["abs_bias_ge_10"].sum()),
                "primary_risk": primary,
                "auc_abs_bias_ge_10": binary_auc(part["abs_bias_ge_10"], part[primary]),
                "spearman_risk_vs_abs_bias": spearman_correlation(part[primary], part["abs_bias"]),
                "pearson_risk_vs_abs_bias": pearson_correlation(part[primary], part["abs_bias"]),
            }
        )
    return merged.sort_values(["fold", "well_id"], kind="mergesort"), pd.DataFrame(metric_rows)


def formation_metric_row(
    formation: str,
    scope: str,
    fold: int,
    errors: np.ndarray,
    sums: Mapping[str, float],
) -> dict[str, Any]:
    count = int(sums["count"])
    numerator = count * sums["sum_xy"] - sums["sum_x"] * sums["sum_y"]
    denominator = math.sqrt(
        max(count * sums["sum_xx"] - sums["sum_x"] ** 2, 0.0)
        * max(count * sums["sum_yy"] - sums["sum_y"] ** 2, 0.0)
    )
    correlation = float(numerator / denominator) if denominator > 0.0 else float("nan")
    return {
        "scope": scope,
        "fold": fold,
        "formation": formation,
        "delta_rows": count,
        "finite_coverage": float(count / int(sums["total"])),
        "identity_error_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "identity_error_mae": float(np.mean(np.abs(errors))),
        "identity_error_p95_abs": float(np.quantile(np.abs(errors), 0.95)),
        "identity_error_max_abs": float(np.max(np.abs(errors))),
        "delta_correlation": correlation,
    }


def build_formation_identity_audit(
    well_paths: Mapping[str, Path],
    fold_by_well: Mapping[str, int],
    *,
    frozen_hashes: Mapping[str, str],
) -> pd.DataFrame:
    require_frozen_hashes(frozen_hashes)
    states: dict[tuple[int, str], dict[str, Any]] = {}
    for fold in sorted(set(fold_by_well.values())):
        for formation in FORMATION_COLUMNS:
            states[(fold, formation)] = {
                "errors": [],
                "count": 0,
                "total": 0,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "sum_xx": 0.0,
                "sum_yy": 0.0,
                "sum_xy": 0.0,
            }
    columns = ["MD", "Z", "TVT", *FORMATION_COLUMNS]
    for well in sorted(well_paths):
        frame = pd.read_csv(well_paths[well], usecols=columns)
        frame.insert(0, "original_row", np.arange(len(frame), dtype=np.int64))
        frame = frame.sort_values(["MD", "original_row"], kind="mergesort")
        reference = np.diff(
            pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float64)
            + pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(np.float64)
        )
        fold = int(fold_by_well[well])
        for formation in FORMATION_COLUMNS:
            observed = np.diff(
                pd.to_numeric(frame[formation], errors="coerce").to_numpy(np.float64)
            )
            finite = np.isfinite(observed) & np.isfinite(reference)
            x = observed[finite]
            y = reference[finite]
            error = x - y
            state = states[(fold, formation)]
            state["errors"].append(error)
            state["count"] += len(error)
            state["total"] += len(reference)
            state["sum_x"] += float(x.sum())
            state["sum_y"] += float(y.sum())
            state["sum_xx"] += float(np.square(x).sum())
            state["sum_yy"] += float(np.square(y).sum())
            state["sum_xy"] += float((x * y).sum())

    rows: list[dict[str, Any]] = []
    for formation in FORMATION_COLUMNS:
        overall_errors: list[np.ndarray] = []
        overall_sums = {
            "count": 0,
            "total": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_xx": 0.0,
            "sum_yy": 0.0,
            "sum_xy": 0.0,
        }
        for fold in sorted(set(fold_by_well.values())):
            state = states[(fold, formation)]
            errors = np.concatenate(state["errors"])
            rows.append(formation_metric_row(formation, f"fold_{fold}", fold, errors, state))
            overall_errors.append(errors)
            for key in overall_sums:
                overall_sums[key] += state[key]
        rows.append(
            formation_metric_row(
                formation,
                "overall",
                -1,
                np.concatenate(overall_errors),
                overall_sums,
            )
        )
    return pd.DataFrame(rows).sort_values(["formation", "fold"], kind="mergesort")


# %% [markdown]
# ## 7. Stage 0 scientific guard and generated artifacts


# %%
def evaluate_stage0_guard(
    graph_manifest: pd.DataFrame,
    well_risk: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    overall = metrics.loc[metrics["scope"].eq("overall")]
    if len(overall) != 1:
        raise ValueError("missing unique overall Stage 0 metric")
    overall_row = overall.iloc[0]
    fold_rows = metrics.loc[metrics["fold"].ge(0)].copy()
    positive_fold = fold_rows["spearman_risk_vs_abs_bias"].gt(0.0) & fold_rows[
        "auc_abs_bias_ge_10"
    ].gt(0.5)
    guards = get_nested(config, "stages.stage0.guards")
    technical = {
        "expected_folds": sorted(graph_manifest["fold"].astype(int).tolist())
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
        "expected_wells": int(well_risk["well_id"].nunique())
        == int(get_nested(config, "validation.expected_wells")),
        "forbidden_column_hits_zero": int(graph_manifest["target_forbidden_column_hits"].sum())
        == 0,
        "source_target_overlap_zero": int(graph_manifest["source_target_overlap"].sum()) == 0,
        "source_formation_accounting_exact": bool(
            (
                graph_manifest["source_wells_with_formation"]
                + graph_manifest["source_wells_without_formation"]
                == graph_manifest["source_wells"]
            ).all()
        ),
        "risk_finite_coverage_one": bool(
            np.isfinite(well_risk["suffix_fault_risk_p90"].to_numpy(np.float64)).all()
        ),
        "truth_access_before_freeze_zero": int(
            graph_manifest["truth_access_before_risk_freeze"].sum()
        )
        == 0,
    }
    scientific = {
        "pooled_auc": float(overall_row["auc_abs_bias_ge_10"])
        >= float(guards["abs_exp226_bias_ge10_auc_min"]),
        "pooled_spearman": float(overall_row["spearman_risk_vs_abs_bias"])
        >= float(guards["pooled_spearman_min"]),
        "positive_fold_count": int(positive_fold.sum()) >= int(guards["positive_fold_count_min"]),
    }
    return {
        "passed": bool(all(technical.values()) and all(scientific.values())),
        "technical": technical,
        "scientific": scientific,
        "readout": {
            "auc_abs_exp226_bias_ge_10": float(overall_row["auc_abs_bias_ge_10"]),
            "pooled_spearman": float(overall_row["spearman_risk_vs_abs_bias"]),
            "positive_fold_count": int(positive_fold.sum()),
            "positive_folds": fold_rows.loc[positive_fold, "fold"].astype(int).tolist(),
        },
        "failure_policy": "close_without_edge_threshold_formation_or_risk_aggregation_grid",
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp289 Stage 0 must run on Kaggle; local execution requires explicit approval."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp289 Kaggle CPU execution has not been approved")
    validate_scientific_contract(config)
    started = time.time()
    raw_dir = train_data_dir(config)
    well_paths = list_horizontal_paths(raw_dir)
    fold_by_well, oof_path, oof_manifest = load_fold_identity(config)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(well_paths) != expected_wells or set(well_paths) != set(fold_by_well):
        raise ValueError("raw horizontal wells and exp226 fold identity differ")

    input_rows = [oof_manifest]
    input_rows.extend(
        {
            "name": f"horizontal_{well}",
            "path": str(path),
            "raw_sha256": sha256_path(path),
            "decompressed_sha256": "",
            "rows": -1,
            "wells": 1,
            "columns_read": "fold-dependent allowlist",
            "forbidden_column_hits": 0,
        }
        for well, path in sorted(well_paths.items())
    )
    input_manifest = pd.DataFrame(input_rows)
    node_parts: list[pd.DataFrame] = []
    well_parts: list[pd.DataFrame] = []
    graph_rows: list[dict[str, Any]] = []

    for fold in [int(value) for value in get_nested(config, "validation.expected_folds")]:
        source_wells = sorted(well for well, value in fold_by_well.items() if value != fold)
        target_wells = sorted(well for well, value in fold_by_well.items() if value == fold)
        overlap = set(source_wells).intersection(target_wells)
        if overlap:
            raise ValueError(f"fold {fold} source/target overlap")
        print(
            f"exp289 Stage 0 fold={fold} source_wells={len(source_wells)} "
            f"target_wells={len(target_wells)}"
        )
        source_nodes = build_source_nodes(well_paths, source_wells, config)
        source_wells_with_formation = sorted(source_nodes["well_id"].astype(str).unique())
        source_wells_without_formation = sorted(
            set(source_wells).difference(source_wells_with_formation)
        )
        source_missing_payload = json.dumps(
            source_wells_without_formation, separators=(",", ":")
        )
        target_nodes = build_target_nodes(well_paths, target_wells, config)
        (
            calibration,
            target_indices,
            target_distances,
            source_indices,
            source_distances,
            edge_manifest,
        ) = build_fold_graph_calibration(source_nodes, target_nodes, config)
        node_risk, well_risk = build_target_fault_risk(
            fold,
            source_nodes,
            target_nodes,
            target_indices,
            target_distances,
            calibration,
            config,
        )
        graph_rows.append(
            {
                "fold": fold,
                "source_wells": len(source_wells),
                "source_wells_with_formation": len(source_wells_with_formation),
                "source_wells_without_formation": len(source_wells_without_formation),
                "source_wells_without_formation_ids": "|".join(
                    source_wells_without_formation
                ),
                "source_wells_without_formation_sha256": hashlib.sha256(
                    source_missing_payload.encode()
                ).hexdigest(),
                "target_wells": len(target_wells),
                "source_target_overlap": len(overlap),
                "source_nodes": len(source_nodes),
                "target_nodes": len(target_nodes),
                "source_edges": int(source_indices.size),
                "target_edges": int(target_indices.size),
                "source_node_schema_sha256": schema_sha256(source_nodes),
                "source_node_content_sha256": dataframe_content_sha256(source_nodes),
                "target_safe_node_schema_sha256": schema_sha256(target_nodes),
                "target_safe_node_content_sha256": dataframe_content_sha256(target_nodes),
                **edge_manifest,
                **calibration,
                "target_forbidden_column_hits": 0,
                "truth_access_before_risk_freeze": 0,
            }
        )
        node_parts.append(node_risk)
        well_parts.append(well_risk)
        del (
            source_nodes,
            target_nodes,
            target_indices,
            target_distances,
            source_indices,
            source_distances,
        )
        gc.collect()

    graph_manifest = pd.DataFrame(graph_rows).sort_values("fold", kind="mergesort")
    node_risk = pd.concat(node_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "MD", "row_idx"], kind="mergesort"
    )
    well_risk = pd.concat(well_parts, ignore_index=True).sort_values(
        ["fold", "well_id"], kind="mergesort"
    )
    frozen_hashes = freeze_target_free_outputs(graph_manifest, node_risk, well_risk)
    outputs = artifact_dir()
    contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_fault_topology_association_readout",
        "created_at": datetime.now(UTC).isoformat(),
        "active_variants": 1,
        "ml_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "control_retraining": 0,
        "kaggle_kernel_version": 3,
        "missing_source_formation_policy": get_nested(
            config, "physics.graph.missing_source_formation_policy"
        ),
        "target_free_frozen_hashes": frozen_hashes,
        "primary_risk": get_nested(config, "stages.stage0.topology_contract.primary_well_risk"),
        "stage1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    write_json(outputs / f"{OUTPUT_PREFIX}_stage0_contract.json", contract)
    output_manifests = {
        "input_manifest": write_frame(
            input_manifest, outputs / f"{OUTPUT_PREFIX}_input_manifest.csv"
        ),
        "graph_manifest": write_frame(
            graph_manifest, outputs / f"{OUTPUT_PREFIX}_graph_manifest.csv"
        ),
        "target_free_node_risk": write_frame(
            node_risk,
            outputs / f"{OUTPUT_PREFIX}_target_free_node_risk.csv.gz",
            gzip_output=True,
        ),
        "target_free_well_risk": write_frame(
            well_risk, outputs / f"{OUTPUT_PREFIX}_target_free_well_risk.csv"
        ),
    }
    # The next two readers are the first places where suffix truth or valid
    # formation columns are allowed to enter the process.
    bias_readout, fold_metrics = load_exp226_bias_readout(
        oof_path,
        fold_by_well,
        well_risk,
        config,
        frozen_hashes=frozen_hashes,
    )
    formation_audit = build_formation_identity_audit(
        well_paths, fold_by_well, frozen_hashes=frozen_hashes
    )
    guard = evaluate_stage0_guard(graph_manifest, well_risk, fold_metrics, config)
    output_manifests.update(
        {
            "exp226_bias_readout": write_frame(
                bias_readout, outputs / f"{OUTPUT_PREFIX}_exp226_bias_readout.csv"
            ),
            "fold_metrics": write_frame(
                fold_metrics, outputs / f"{OUTPUT_PREFIX}_fold_metrics.csv"
            ),
            "formation_identity_audit": write_frame(
                formation_audit,
                outputs / f"{OUTPUT_PREFIX}_formation_identity_audit.csv",
            ),
        }
    )
    runtime_seconds = float(time.time() - started)
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_fault_topology_association_readout",
        "status": "stage0_guard_passed" if guard["passed"] else "stage0_guard_failed_branch_closed",
        "created_at": datetime.now(UTC).isoformat(),
        "rows": int(get_nested(config, "validation.expected_rows")),
        "wells": int(well_risk["well_id"].nunique()),
        "folds": int(graph_manifest["fold"].nunique()),
        "kaggle_kernel_version": 3,
        "runtime_seconds": runtime_seconds,
        "peak_rss_mb": peak_rss_mb,
        "target_free_frozen_hashes": frozen_hashes,
        "guard": guard,
        "output_manifests": output_manifests,
        "stage1_implemented": False,
        "next_action": (
            "request_separate_stage1_approval"
            if guard["passed"]
            else "close_branch_without_rescue_grid"
        ),
    }
    write_json(outputs / f"{OUTPUT_PREFIX}_stage0_summary.json", summary)
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": summary["status"],
        "stage": summary["stage"],
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "active_variants": 1,
        "ml_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "kaggle_runs": 3,
        "runtime_seconds": runtime_seconds,
        "stage0_guard": guard,
        "target_free_frozen_hashes": frozen_hashes,
        "stage1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    write_json(metrics_output_path(), metrics_payload)
    return summary


# %% [markdown]
# ## 8. Setup and contract preview

# %%
config = load_experiment_config()
validate_scientific_contract(config)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Stage:", get_nested(config, "execution.stage"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Active audit variants / ML configs / trained folds / boosters: 1 / 0 / 0 / 0")
print("Control or parent retraining:", get_nested(config, "execution.control_or_parent_retraining"))
print(
    "Primary target-free risk:",
    get_nested(config, "stages.stage0.topology_contract.primary_well_risk"),
)
print("Outer-valid safe columns:", TARGET_SAFE_COLUMNS)
print("Outer-valid forbidden columns:", sorted(TARGET_FORBIDDEN_COLUMNS))
print("Stage 1 implemented: False")
print("Inference enabled:", get_nested(config, "inference.enabled"))
print("Kaggle push approved:", get_nested(config, "execution.kaggle_push_approved"))


# %% [markdown]
# ## 9. Run the fixed Kaggle CPU readout

# %%
if EXECUTE_NOTEBOOK:
    stage0_summary = run_stage0(config)
    print(json.dumps(to_jsonable(stage0_summary["guard"]), indent=2, sort_keys=True))
    print("Stage 0 summary:", artifact_dir() / f"{OUTPUT_PREFIX}_stage0_summary.json")
else:
    print("Import-only/source validation mode: Stage 0 was not executed.")
