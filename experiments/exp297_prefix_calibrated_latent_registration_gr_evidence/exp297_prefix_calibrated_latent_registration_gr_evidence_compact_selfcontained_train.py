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
# # exp297 prefix-calibrated latent-registration GR evidence
#
# This train-side Stage-2 audit keeps the exp293 deployable12 physical paths
# fixed. It calibrates Type Well GR on the known prefix, marginalizes a fixed
# observation-registration grid and a reliable/unreliable state, freezes all
# target-free posterior evidence, and only then reads suffix TVT for an
# expected-candidate-SSE readout. It never creates a TVT prediction.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Frozen exp293 bank and raw target-free input loaders
# 4. Prefix calibration and registered forward-GR helpers
# 5. Block components, posterior, and negative control
# 6. Target-free evidence freeze and artifact writers
# 7. Post-freeze truth readout and Stage-2 decision
# 8. Setup and execution

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

EXPERIMENT_NAME = "exp297_prefix_calibrated_latent_registration_gr_evidence"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
GROUP_COLUMNS = {128: "h128_group", 256: "h256_group", 512: "h512_group"}
BLOCK_ASSIGNMENT_DTYPES = {
    "id": object,
    "well": object,
    "well_row_idx": "int32",
    "outer_fold": "int8",
    "md_since": "float32",
    "well_code": "int32",
    "h128_group": "int32",
    "h256_group": "int32",
    "h512_group": "int32",
    "whole_well_group": "int32",
}
EXPECTED_CANDIDATES = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
SAFE_CANDIDATE = "exp226_w500_50_50"
SAFE_INDEX = EXPECTED_CANDIDATES.index(SAFE_CANDIDATE)
FORBIDDEN_TARGET_FREE_COLUMNS = {
    "TVT",
    "target",
    "true_tvt",
    "error",
    "abs_error",
    "oracle",
    "oracle_candidate",
    "candidate_best",
}


# %% [markdown]
# ## 2. Runtime, path, SHA, and serialization helpers


# %%
def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = os.environ.get("EXP297_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    nested = project_root() / "experiments" / EXPERIMENT_NAME
    return nested if nested.exists() else Path.cwd()


def find_config_path() -> Path:
    for path in (Path.cwd() / "config.yaml", experiment_dir() / "config.yaml"):
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp297 config.yaml was not found unambiguously")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def reject_truth_columns(columns: Iterable[str]) -> None:
    normalized = {str(column) for column in columns}
    forbidden = normalized & FORBIDDEN_TARGET_FREE_COLUMNS
    token_forbidden = {
        column
        for column in normalized
        if any(token in column.lower() for token in ("true_tvt", "abs_error", "oracle_"))
    }
    if forbidden or token_forbidden:
        raise ValueError(
            "target-free input exposes forbidden truth/readout columns: "
            f"{sorted(forbidden | token_forbidden)}"
        )


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    string_columns = [
        column for column, dtype in frame.dtypes.items() if isinstance(dtype, pd.StringDtype)
    ]
    if not string_columns:
        return frame
    normalized = frame.copy()
    for column in string_columns:
        normalized[column] = normalized[column].astype(object)
    return normalized


def frame_content_sha256(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = _normalize_frame_for_hash(selected)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format="%.12g", lineterminator="\n")


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    found: dict[str, Path] = {}
    root = project_root()
    for raw in patterns:
        path = Path(str(raw))
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            found.setdefault(str(direct.resolve()), direct)
        for match in glob.glob(str(raw), recursive=True):
            candidate = Path(match)
            if candidate.exists():
                found.setdefault(str(candidate.resolve()), candidate)
        if not path.is_absolute():
            for match in glob.glob(str(root / path), recursive=True):
                candidate = Path(match)
                if candidate.exists():
                    found.setdefault(str(candidate.resolve()), candidate)
    return list(found.values())


def resolve_file(
    patterns: Sequence[str], *, label: str, expected_sha256: str | None = None
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    if expected_sha256:
        matching = [path for path in candidates if sha256_file(path) == expected_sha256]
        if matching:
            return sorted(matching, key=lambda item: len(str(item)))[0]
        if candidates:
            raise ValueError(
                f"{label} SHA mismatch: { {str(path): sha256_file(path) for path in candidates} }"
            )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"{label} not found from patterns: {patterns}")
    raise ValueError(f"{label} resolved to multiple files: {candidates}")


def resolve_train_dir(patterns: Sequence[str]) -> Path:
    for path in expand_existing_paths(patterns):
        if path.is_dir() and any(path.glob("*__horizontal_well.csv")):
            return path
    raise FileNotFoundError(f"raw train directory not found: {patterns}")


def artifact_dir() -> Path:
    base = KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else experiment_dir()
    path = base / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


# %% [markdown]
# ## 3. Frozen exp293 bank and raw target-free input loaders


# %%
@dataclass(frozen=True)
class WellSegment:
    well: str
    start: int
    end: int
    fold: int


@dataclass
class FrozenBank:
    keys: pd.DataFrame
    values: np.memmap
    candidate_ids: tuple[str, ...]
    segments: list[WellSegment]
    matrix_path: Path
    manifest_path: Path
    assignment_path: Path
    candidate_content_sha256: str
    input_evidence: list[dict[str, Any]]


def candidate_bank_content_sha256(
    keys: pd.DataFrame,
    values: np.ndarray,
    candidate_ids: Sequence[str],
    chunk_rows: int = 100_000,
) -> str:
    key_sha = frame_content_sha256(keys, KEY_COLUMNS)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(candidate_ids), separators=(",", ":")).encode())
    digest.update(key_sha.encode())
    for position, candidate in enumerate(candidate_ids):
        digest.update(str(candidate).encode())
        for start in range(0, len(keys), chunk_rows):
            end = min(start + chunk_rows, len(keys))
            digest.update(np.asarray(values[start:end, position], dtype="<f4").tobytes())
    return digest.hexdigest()


def build_segments(keys: pd.DataFrame) -> list[WellSegment]:
    wells = keys["well"].astype(str).to_numpy()
    if len(wells) == 0:
        raise ValueError("empty exp293 block assignment")
    starts = np.r_[0, np.flatnonzero(wells[1:] != wells[:-1]) + 1]
    ends = np.r_[starts[1:], len(wells)]
    segments: list[WellSegment] = []
    for start, end in zip(starts, ends, strict=True):
        well = str(wells[start])
        if not np.all(wells[start:end] == well):
            raise ValueError("well rows are non-contiguous")
        row_idx = keys["well_row_idx"].to_numpy(np.int64)[start:end]
        if len(row_idx) > 1 and not np.all(np.diff(row_idx) == 1):
            raise ValueError(f"non-contiguous row_idx for well={well}")
        folds = keys["outer_fold"].to_numpy(np.int64)[start:end]
        if not np.all(folds == folds[0]):
            raise ValueError(f"well spans folds: {well}")
        segments.append(WellSegment(well, int(start), int(end), int(folds[0])))
    return segments


def load_frozen_exp293_bank(config: Mapping[str, Any]) -> FrozenBank:
    spec = get_nested(config, "data.exp293")
    matrix_path = resolve_file(
        spec["candidate_matrix_patterns"],
        label="exp293 candidate matrix",
        expected_sha256=str(spec["candidate_matrix_file_sha256"]),
    )
    manifest_path = resolve_file(
        spec["bank_manifest_patterns"],
        label="exp293 bank manifest",
        expected_sha256=str(spec["bank_manifest_file_sha256"]),
    )
    assignment_path = resolve_file(
        spec["block_assignment_patterns"],
        label="exp293 block assignment",
        expected_sha256=str(spec["block_assignment_file_sha256"]),
    )
    decompressed_sha = sha256_gzip_decompressed(assignment_path)
    if decompressed_sha != str(spec["block_assignment_decompressed_sha256"]):
        raise ValueError("exp293 block assignment decompressed SHA mismatch")
    # exp293 created these keys from exp263's parquet identity frame. Preserve
    # those physical dtypes when reloading its CSV; frame_content_sha256 includes
    # dtype names, and widening int32/int8/float32 would invalidate the frozen
    # logical SHA despite identical values.
    keys = pd.read_csv(assignment_path, dtype=BLOCK_ASSIGNMENT_DTYPES)
    required = (
        set(KEY_COLUMNS)
        | set(GROUP_COLUMNS.values())
        | {
            "well_code",
            "whole_well_group",
        }
    )
    if missing := sorted(required - set(keys.columns)):
        raise ValueError(f"exp293 block assignment missing columns: {missing}")
    reject_truth_columns(keys.columns)
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(keys) != expected_rows or keys["well"].nunique() != expected_wells:
        raise ValueError("exp293 block assignment row/well contract mismatch")
    if set(keys["outer_fold"].unique()) != set(range(5)):
        raise ValueError("exp293 outer fold inventory mismatch")
    key_sha = frame_content_sha256(keys, KEY_COLUMNS)
    if key_sha != str(spec["key_content_sha256"]):
        raise ValueError(f"exp293 key content SHA mismatch: {key_sha}")
    logical_assignment_sha = frame_content_sha256(keys)
    if logical_assignment_sha != str(spec["block_assignment_logical_sha256"]):
        raise ValueError(f"exp293 block assignment logical SHA mismatch: {logical_assignment_sha}")
    manifest = json.loads(manifest_path.read_text())
    candidate_ids = tuple(str(item) for item in manifest["candidate_ids"])
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("exp293 candidate order differs from Stage-2 contract")
    expected_bytes = expected_rows * len(candidate_ids) * np.dtype("float32").itemsize
    if matrix_path.stat().st_size != expected_bytes:
        raise ValueError("exp293 candidate matrix byte size mismatch")
    values = np.memmap(
        matrix_path,
        mode="r",
        dtype="float32",
        shape=(expected_rows, len(candidate_ids)),
    )
    if not np.isfinite(values).all():
        raise ValueError("exp293 candidate matrix contains non-finite values")
    candidate_sha = candidate_bank_content_sha256(keys, values, candidate_ids)
    if candidate_sha != str(spec["candidate_content_sha256"]):
        raise ValueError(f"exp293 candidate content SHA mismatch: {candidate_sha}")
    if candidate_sha != str(manifest["candidate_content_sha256"]):
        raise ValueError("exp293 bank manifest candidate SHA mismatch")
    evidence = [
        {
            "phase": "target_free",
            "role": "exp293_candidate_matrix",
            "path": str(matrix_path),
            "rows": expected_rows,
            "wells": expected_wells,
            "file_sha256": sha256_file(matrix_path),
            "decompressed_content_sha256": None,
            "logical_content_sha256": candidate_sha,
        },
        {
            "phase": "target_free",
            "role": "exp293_bank_manifest",
            "path": str(manifest_path),
            "rows": 1,
            "wells": expected_wells,
            "file_sha256": sha256_file(manifest_path),
            "decompressed_content_sha256": None,
            "logical_content_sha256": stable_json_sha256(manifest),
        },
        {
            "phase": "target_free",
            "role": "exp293_block_assignment",
            "path": str(assignment_path),
            "rows": expected_rows,
            "wells": expected_wells,
            "file_sha256": sha256_file(assignment_path),
            "decompressed_content_sha256": decompressed_sha,
            "logical_content_sha256": logical_assignment_sha,
        },
    ]
    return FrozenBank(
        keys=keys,
        values=values,
        candidate_ids=candidate_ids,
        segments=build_segments(keys),
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        assignment_path=assignment_path,
        candidate_content_sha256=candidate_sha,
        input_evidence=evidence,
    )


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["MD", "GR", "TVT_input"]
    if missing := sorted(set(required) - set(header)):
        raise ValueError(f"{path.name} missing target-free columns: {missing}")
    frame = pd.read_csv(path, usecols=required)
    reject_truth_columns(frame.columns)
    return frame


def load_typewell(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if missing := sorted({"TVT", "GR"} - set(header)):
        raise ValueError(f"{path.name} missing Type Well columns: {missing}")
    return pd.read_csv(path, usecols=["TVT", "GR"])


def prepare_typewell_curve(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    clean = clean.sort_values("TVT", kind="mergesort")
    clean = clean.groupby("TVT", as_index=False, sort=True)["GR"].median()
    if len(clean) < 2:
        raise ValueError("typewell_fewer_than_two_unique_points")
    tvt = clean["TVT"].to_numpy(np.float64)
    gr = clean["GR"].to_numpy(np.float64)
    if not np.all(np.diff(tvt) > 0):
        raise ValueError("Type Well TVT must be strictly increasing")
    return tvt, gr


def interpolate_no_extrapolation(
    query: np.ndarray, typewell_tvt: np.ndarray, typewell_gr: np.ndarray
) -> np.ndarray:
    flat = np.asarray(query, dtype=np.float64).reshape(-1)
    values = np.interp(flat, typewell_tvt, typewell_gr, left=np.nan, right=np.nan)
    return values.reshape(np.asarray(query).shape)


def typewell_gradient_no_extrapolation(
    query: np.ndarray, typewell_tvt: np.ndarray, typewell_gr: np.ndarray
) -> np.ndarray:
    midpoint = (typewell_tvt[:-1] + typewell_tvt[1:]) / 2.0
    gradient = np.diff(typewell_gr) / np.diff(typewell_tvt)
    flat = np.asarray(query, dtype=np.float64).reshape(-1)
    values = np.interp(flat, midpoint, gradient, left=np.nan, right=np.nan)
    return values.reshape(np.asarray(query).shape)


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
    sets: dict[str, set[str]] = {}
    for name, column in spec["role_columns"].items():
        selected = set(frame.loc[frame[column].eq("valid"), well_column].astype(str))
        if unknown := selected - expected_wells:
            raise ValueError(f"hidden-like set contains unknown wells: {sorted(unknown)[:5]}")
        sets[str(name)] = selected
    evidence = {
        "phase": "target_free",
        "role": "hidden_like_assignment",
        "path": str(path),
        "rows": len(frame),
        "wells": frame[well_column].nunique(),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(frame),
    }
    return sets, evidence


# %% [markdown]
# ## 4. Prefix calibration and registered forward-GR helpers


# %%
@dataclass(frozen=True)
class CalibrationResult:
    valid: bool
    reason: str
    raw_slope: float
    slope: float
    intercept: float
    residual_scale: float
    derivative_scale: float
    pairs: int
    reference_std: float
    prefix_rmse: float


def invalid_calibration(reason: str, pairs: int = 0) -> CalibrationResult:
    return CalibrationResult(
        False,
        reason,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        pairs,
        float("nan"),
        float("nan"),
    )


def median_mad(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    median = float(np.median(finite))
    return median, float(np.median(np.abs(finite - median)))


def weighted_affine(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    design = np.column_stack([x, np.ones(len(x))]) * root[:, None]
    target = y * root
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(slope), float(intercept)


def robust_affine_calibration(
    horizontal_gr: np.ndarray,
    tvt_input: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: Mapping[str, Any],
) -> CalibrationResult:
    spec = get_nested(config, "audit.prefix_calibration")
    reference = interpolate_no_extrapolation(tvt_input, typewell_tvt, typewell_gr)
    valid = np.isfinite(horizontal_gr) & np.isfinite(tvt_input) & np.isfinite(reference)
    indices = np.flatnonzero(valid)
    maximum = int(spec["maximum_rows"])
    if len(indices) > maximum:
        indices = indices[-maximum:]
    minimum = int(spec["minimum_pairs"])
    if len(indices) < minimum:
        return invalid_calibration("prefix_pairs_below_minimum", len(indices))
    x = reference[indices]
    y = np.asarray(horizontal_gr, dtype=np.float64)[indices]
    reference_std = float(np.std(x))
    if not math.isfinite(reference_std) or reference_std < float(spec["minimum_typewell_gr_std"]):
        return invalid_calibration("prefix_typewell_gr_std_below_minimum", len(x))
    weights = np.ones(len(x), dtype=np.float64)
    slope, intercept = weighted_affine(x, y, weights)
    delta = float(spec["huber_delta"])
    for _ in range(int(spec["irls_iterations"])):
        residual = y - (slope * x + intercept)
        _, mad = median_mad(residual)
        scale = max(1.4826 * mad, 1.0e-6)
        normalized = np.abs(residual) / scale
        weights = np.ones(len(x), dtype=np.float64)
        outside = normalized > delta
        weights[outside] = delta / normalized[outside]
        slope, intercept = weighted_affine(x, y, weights)
    raw_slope = float(slope)
    low, high = [float(value) for value in spec["slope_clip"]]
    slope = float(np.clip(raw_slope, low, high))
    intercept = float(np.sum(weights * (y - slope * x)) / np.sum(weights))
    residual = y - (slope * x + intercept)
    prefix_rmse = float(np.sqrt(np.mean(residual**2)))
    if not all(math.isfinite(value) for value in (slope, intercept, prefix_rmse)):
        return invalid_calibration("non_finite_affine", len(x))
    if prefix_rmse > float(spec["maximum_prefix_rmse"]):
        return invalid_calibration("prefix_rmse_above_maximum", len(x))
    _, residual_mad = median_mad(residual)
    residual_low, residual_high = [
        float(value) for value in get_nested(config, "audit.residual_component.scale_clip")
    ]
    residual_scale = float(np.clip(1.4826 * residual_mad, residual_low, residual_high))
    _, derivative_mad = median_mad(np.diff(residual))
    derivative_low, derivative_high = [
        float(value) for value in get_nested(config, "audit.derivative_component.scale_clip")
    ]
    derivative_scale = float(np.clip(1.4826 * derivative_mad, derivative_low, derivative_high))
    if not all(math.isfinite(value) for value in (residual_scale, derivative_scale)):
        return invalid_calibration("non_finite_scale", len(x))
    return CalibrationResult(
        True,
        "ok",
        raw_slope,
        slope,
        intercept,
        residual_scale,
        derivative_scale,
        len(x),
        reference_std,
        prefix_rmse,
    )


def registration_grid(config: Mapping[str, Any]) -> np.ndarray:
    spec = get_nested(config, "audit.registration")
    values = np.arange(
        float(spec["minimum_ft"]),
        float(spec["maximum_ft"]) + 0.5 * float(spec["step_ft"]),
        float(spec["step_ft"]),
        dtype=np.float64,
    )
    if len(values) != int(spec["expected_states"]):
        raise ValueError("registration state count mismatch")
    return values


def registration_prior(deltas: np.ndarray, scale: float) -> np.ndarray:
    weights = np.exp(-np.abs(np.asarray(deltas, dtype=np.float64)) / float(scale))
    return weights / weights.sum()


def registered_forward_matrices(
    candidate_paths: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    calibration: CalibrationResult,
    deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, candidates = candidate_paths.shape
    states = candidates * len(deltas)
    reference = np.full((rows, states), np.nan, dtype=np.float32)
    derivative = np.full((max(0, rows - 1), states), np.nan, dtype=np.float32)
    for candidate in range(candidates):
        path = np.asarray(candidate_paths[:, candidate], dtype=np.float64)
        query = path[:, None] + deltas[None, :]
        raw = interpolate_no_extrapolation(query, typewell_tvt, typewell_gr)
        start = candidate * len(deltas)
        end = start + len(deltas)
        reference[:, start:end] = (calibration.slope * raw + calibration.intercept).astype(
            np.float32
        )
        if rows > 1:
            midpoint = (path[:-1] + path[1:]) / 2.0
            registered_midpoint = midpoint[:, None] + deltas[None, :]
            gradient = typewell_gradient_no_extrapolation(
                registered_midpoint, typewell_tvt, typewell_gr
            )
            derivative[:, start:end] = (
                calibration.slope * gradient * np.diff(path)[:, None]
            ).astype(np.float32)
    return reference, derivative


# %% [markdown]
# ## 5. Block components, posterior, and negative control


# %%
@dataclass(frozen=True)
class BlockSlice:
    group: int
    start: int
    end: int


@dataclass(frozen=True)
class PosteriorResult:
    joint_reliable: np.ndarray
    candidate: np.ndarray
    candidate_reliable: np.ndarray
    registration_reliable: np.ndarray
    reliable_probability: np.ndarray
    unreliable_probability: np.ndarray
    candidate_entropy: np.ndarray
    registration_entropy: np.ndarray
    candidate_mode_gap: np.ndarray
    eligible_states: np.ndarray
    log_evidence: np.ndarray


def build_block_slices(groups: np.ndarray, segment: WellSegment) -> list[BlockSlice]:
    local = np.asarray(groups[segment.start : segment.end], dtype=np.int64)
    if len(local) == 0:
        return []
    changes = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1, len(local)]
    blocks: list[BlockSlice] = []
    for left, right in zip(changes[:-1], changes[1:], strict=True):
        blocks.append(BlockSlice(int(local[left]), int(left), int(right)))
    if sum(block.end - block.start for block in blocks) != len(local):
        raise ValueError(f"block partition does not cover well={segment.well}")
    return blocks


def stable_rotation_offset(well: str, finite_count: int, config: Mapping[str, Any]) -> int:
    if finite_count <= 1:
        return 0
    spec = get_nested(config, "audit.shuffled_control")
    seed_material = (
        f"{EXPERIMENT_NAME}|{int(spec['seed'])}|{well}|finite_circular_rotation"
    ).encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    minimum = max(
        int(spec["minimum_rotation_rows"]),
        int(math.ceil(float(spec["minimum_rotation_fraction"]) * finite_count)),
    )
    minimum = min(minimum, finite_count // 2)
    allowed = np.arange(minimum, finite_count - minimum + 1, dtype=np.int64)
    if len(allowed) == 0:
        allowed = np.arange(1, finite_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return int(allowed[int(rng.integers(0, len(allowed)))])


def shuffled_preserve_nan_mask(
    observed: np.ndarray, well: str, config: Mapping[str, Any]
) -> tuple[np.ndarray, int]:
    result = np.asarray(observed, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(result))
    offset = stable_rotation_offset(well, len(finite), config)
    if offset:
        result[finite] = np.roll(result[finite], offset)
    return result, offset


def _block_sum(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    prefix = np.vstack(
        [np.zeros((1, matrix.shape[1]), dtype=np.float64), np.cumsum(matrix, axis=0)]
    )
    return prefix[ends] - prefix[starts]


def _block_median_absolute(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    result = np.full((len(starts), matrix.shape[1]), np.nan, dtype=np.float64)
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        with np.errstate(all="ignore"):
            result[position] = np.nanmedian(np.abs(matrix[start:end]), axis=0)
    return result


def block_component_matrices(
    observed: np.ndarray,
    reference: np.ndarray,
    derivative_reference: np.ndarray,
    blocks: Sequence[BlockSlice],
    calibration: CalibrationResult,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, states = reference.shape
    observed = np.asarray(observed, dtype=np.float64)
    if len(observed) != rows:
        raise ValueError("observed/reference row mismatch")
    starts = np.array([block.start for block in blocks], dtype=np.int64)
    ends = np.array([block.end for block in blocks], dtype=np.int64)
    lengths = ends - starts
    minimum_pairs = int(get_nested(config, "audit.block_coverage.minimum_pairs"))
    minimum_fraction = float(get_nested(config, "audit.block_coverage.minimum_fraction"))
    required = np.maximum(minimum_pairs, np.ceil(minimum_fraction * lengths)).astype(np.int64)

    x = np.broadcast_to(observed[:, None], (rows, states))
    y = np.asarray(reference, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    count = _block_sum(valid, starts, ends)
    sx = _block_sum(np.where(valid, x, 0.0), starts, ends)
    sy = _block_sum(np.where(valid, y, 0.0), starts, ends)
    sxx = _block_sum(np.where(valid, x * x, 0.0), starts, ends)
    syy = _block_sum(np.where(valid, y * y, 0.0), starts, ends)
    sxy = _block_sum(np.where(valid, x * y, 0.0), starts, ends)

    residual = np.where(valid, x - y, 0.0)
    df = float(get_nested(config, "audit.residual_component.degrees_of_freedom"))
    raw_point = -0.5 * (df + 1.0) * np.log1p((residual / calibration.residual_scale) ** 2 / df)
    raw_score = _block_sum(np.where(valid, raw_point, 0.0), starts, ends) / np.maximum(count, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = sxy - sx * sy / count
        x_energy = sxx - sx * sx / count
        y_energy = syy - sy * sy / count
        ncc = numerator / np.sqrt(x_energy * y_energy)
        reference_std = np.sqrt(np.maximum(y_energy / count, 0.0))

    if rows > 1:
        observed_delta = np.diff(observed)
        dref = np.asarray(derivative_reference, dtype=np.float64)
        dx = np.broadcast_to(observed_delta[:, None], dref.shape)
        dvalid = np.isfinite(dx) & np.isfinite(dref)
        derivative_starts = starts
        derivative_ends = np.maximum(starts, ends - 1)
        derivative_count = _block_sum(dvalid, derivative_starts, derivative_ends)
        derivative_abs = np.where(dvalid, np.abs(dx - dref), 0.0)
        derivative_score = -_block_sum(
            derivative_abs, derivative_starts, derivative_ends
        ) / np.maximum(derivative_count, 1.0)
        derivative_score /= calibration.derivative_scale
        forward_energy = _block_median_absolute(
            np.where(dvalid, dref, np.nan), derivative_starts, derivative_ends
        )
    else:
        derivative_count = np.zeros((len(blocks), states), dtype=np.float64)
        derivative_score = np.full((len(blocks), states), np.nan, dtype=np.float64)
        forward_energy = np.full((len(blocks), states), np.nan, dtype=np.float64)

    required_derivative = np.maximum(1, required - 1)
    eligible = (
        (count >= required[:, None])
        & (derivative_count >= required_derivative[:, None])
        & (
            reference_std
            >= float(get_nested(config, "audit.prefix_calibration.minimum_typewell_gr_std"))
        )
        & (
            forward_energy
            >= float(
                get_nested(
                    config,
                    "audit.derivative_component.minimum_median_absolute_forward_delta_gr_per_row",
                )
            )
        )
        & np.isfinite(raw_score)
        & np.isfinite(ncc)
        & np.isfinite(derivative_score)
    )
    return raw_score, ncc, derivative_score, eligible


def robust_state_zscores(
    values: np.ndarray, eligible: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    threshold = float(get_nested(config, "audit.composite.zero_mad_threshold"))
    lower, upper = [float(item) for item in get_nested(config, "audit.composite.zscore_clip")]
    for block in range(values.shape[0]):
        mask = eligible[block] & np.isfinite(values[block])
        if not mask.any():
            continue
        median, mad = median_mad(values[block, mask])
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale <= threshold:
            result[block, mask] = 0.0
        else:
            result[block, mask] = np.clip((values[block, mask] - median) / scale, lower, upper)
    return result


def logsumexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("-inf")
    maximum = float(np.max(finite))
    return maximum + float(np.log(np.exp(finite - maximum).sum()))


def entropy(probability: np.ndarray) -> float:
    values = np.asarray(probability, dtype=np.float64)
    positive = values > 0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def fallback_posterior(
    block_count: int, candidate_count: int, registration_count: int
) -> PosteriorResult:
    joint = np.zeros((block_count, candidate_count, registration_count), dtype=np.float32)
    candidate = np.zeros((block_count, candidate_count), dtype=np.float32)
    candidate[:, SAFE_INDEX] = 1.0
    candidate_reliable = np.zeros_like(candidate)
    registration = np.zeros((block_count, registration_count), dtype=np.float32)
    return PosteriorResult(
        joint,
        candidate,
        candidate_reliable,
        registration,
        np.zeros(block_count, dtype=np.float32),
        np.ones(block_count, dtype=np.float32),
        np.zeros(block_count, dtype=np.float32),
        np.zeros(block_count, dtype=np.float32),
        np.ones(block_count, dtype=np.float32),
        np.zeros(block_count, dtype=np.int32),
        np.full(block_count, -np.inf, dtype=np.float64),
    )


def posterior_from_components(
    raw: np.ndarray,
    ncc: np.ndarray,
    derivative: np.ndarray,
    eligible: np.ndarray,
    deltas: np.ndarray,
    config: Mapping[str, Any],
) -> PosteriorResult:
    blocks, states = raw.shape
    candidates = len(EXPECTED_CANDIDATES)
    registrations = len(deltas)
    if states != candidates * registrations:
        raise ValueError("posterior state shape violates candidate/register contract")
    components = [robust_state_zscores(item, eligible, config) for item in (raw, ncc, derivative)]
    weights = np.asarray(get_nested(config, "audit.composite.weights"), dtype=np.float64)
    if len(weights) != 3 or not np.isclose(weights.sum(), 1.0):
        raise ValueError("component weights must contain three values summing to one")
    score = sum(weight * component for weight, component in zip(weights, components, strict=True))
    prior = registration_prior(
        deltas, float(get_nested(config, "audit.registration.prior_scale_ft"))
    )
    log_state_prior = np.tile(np.log(prior), candidates) - math.log(candidates)
    reliable_prior = float(get_nested(config, "audit.reliability.reliable_prior"))
    unreliable_prior = float(get_nested(config, "audit.reliability.unreliable_prior"))
    outlier_log_likelihood = float(get_nested(config, "audit.reliability.outlier_log_likelihood"))
    joint = np.zeros((blocks, candidates, registrations), dtype=np.float32)
    candidate = np.zeros((blocks, candidates), dtype=np.float32)
    candidate_reliable = np.zeros_like(candidate)
    registration = np.zeros((blocks, registrations), dtype=np.float32)
    reliable_probability = np.zeros(blocks, dtype=np.float32)
    unreliable_probability = np.ones(blocks, dtype=np.float32)
    candidate_entropy = np.zeros(blocks, dtype=np.float32)
    registration_entropy = np.zeros(blocks, dtype=np.float32)
    mode_gap = np.ones(blocks, dtype=np.float32)
    eligible_states = eligible.sum(axis=1).astype(np.int32)
    log_evidence = np.full(blocks, -np.inf, dtype=np.float64)
    for block in range(blocks):
        mask = eligible[block] & np.isfinite(score[block])
        if not mask.any():
            candidate[block, SAFE_INDEX] = 1.0
            continue
        log_reliable = np.full(states, -np.inf, dtype=np.float64)
        log_reliable[mask] = math.log(reliable_prior) + log_state_prior[mask] + score[block, mask]
        reliable_log_total = logsumexp(log_reliable)
        unreliable_log = math.log(unreliable_prior) + outlier_log_likelihood
        normalizer = logsumexp(np.array([reliable_log_total, unreliable_log]))
        state_probability = np.zeros(states, dtype=np.float64)
        state_probability[mask] = np.exp(log_reliable[mask] - normalizer)
        unrel = math.exp(unreliable_log - normalizer)
        joint_block = state_probability.reshape(candidates, registrations)
        reliable_candidate = joint_block.sum(axis=1)
        reliable_registration = joint_block.sum(axis=0)
        final_candidate = reliable_candidate.copy()
        final_candidate[SAFE_INDEX] += unrel
        reliable_mass = float(reliable_candidate.sum())
        conditional_candidate = (
            reliable_candidate / reliable_mass
            if reliable_mass > 0
            else np.zeros(candidates, dtype=np.float64)
        )
        conditional_registration = (
            reliable_registration / reliable_mass
            if reliable_mass > 0
            else np.zeros(registrations, dtype=np.float64)
        )
        ordered = np.sort(final_candidate)[::-1]
        joint[block] = joint_block.astype(np.float32)
        candidate[block] = final_candidate.astype(np.float32)
        candidate_reliable[block] = reliable_candidate.astype(np.float32)
        registration[block] = reliable_registration.astype(np.float32)
        reliable_probability[block] = reliable_mass
        unreliable_probability[block] = unrel
        candidate_entropy[block] = entropy(final_candidate)
        registration_entropy[block] = entropy(conditional_registration)
        mode_gap[block] = float(ordered[0] - ordered[1])
        log_evidence[block] = normalizer
        if not np.isclose(final_candidate.sum(), 1.0, atol=1.0e-6):
            raise ValueError("candidate posterior does not sum to one")
        if not np.isclose(reliable_mass + unrel, 1.0, atol=1.0e-6):
            raise ValueError("reliability posterior does not sum to one")
        if not np.isclose(joint_block.sum(), reliable_mass, atol=1.0e-6):
            raise ValueError("joint reliable posterior mass mismatch")
        if reliable_mass > 0 and not np.isclose(conditional_candidate.sum(), 1.0, atol=1.0e-6):
            raise ValueError("conditional candidate posterior is invalid")
    return PosteriorResult(
        joint,
        candidate,
        candidate_reliable,
        registration,
        reliable_probability,
        unreliable_probability,
        candidate_entropy,
        registration_entropy,
        mode_gap,
        eligible_states,
        log_evidence,
    )


# %% [markdown]
# ## 6. Target-free evidence freeze and artifact writers


# %%
class PosteriorParquetWriters:
    def __init__(self, candidate_path: Path, registration_path: Path) -> None:
        candidate_schema = pa.schema(
            [
                ("block_position", pa.int64()),
                ("candidate_index", pa.int16()),
                ("candidate_probability", pa.float32()),
                ("reliable_candidate_probability", pa.float32()),
                ("log_evidence", pa.float64()),
            ]
        )
        registration_schema = pa.schema(
            [
                ("block_position", pa.int64()),
                ("registration_index", pa.int16()),
                ("registration_delta_ft", pa.float32()),
                ("reliable_registration_probability", pa.float32()),
            ]
        )
        self.candidate = pq.ParquetWriter(candidate_path, candidate_schema, compression="zstd")
        self.registration = pq.ParquetWriter(
            registration_path, registration_schema, compression="zstd"
        )

    def write(
        self,
        positions: np.ndarray,
        posterior: PosteriorResult,
        deltas: np.ndarray,
    ) -> None:
        blocks, candidates = posterior.candidate.shape
        registrations = len(deltas)
        if len(positions) != blocks:
            raise ValueError("posterior writer block-position mismatch")
        candidate_table = pa.table(
            {
                "block_position": np.repeat(positions, candidates).astype(np.int64),
                "candidate_index": np.tile(np.arange(candidates, dtype=np.int16), blocks),
                "candidate_probability": posterior.candidate.reshape(-1),
                "reliable_candidate_probability": posterior.candidate_reliable.reshape(-1),
                "log_evidence": np.repeat(posterior.log_evidence, candidates),
            },
            schema=self.candidate.schema,
        )
        registration_table = pa.table(
            {
                "block_position": np.repeat(positions, registrations).astype(np.int64),
                "registration_index": np.tile(np.arange(registrations, dtype=np.int16), blocks),
                "registration_delta_ft": np.tile(np.asarray(deltas, dtype=np.float32), blocks),
                "reliable_registration_probability": posterior.registration_reliable.reshape(-1),
            },
            schema=self.registration.schema,
        )
        self.candidate.write_table(candidate_table)
        self.registration.write_table(registration_table)

    def close(self) -> None:
        self.candidate.close()
        self.registration.close()


@dataclass(frozen=True)
class FrozenEvidence:
    paths: tuple[Path, ...]
    file_sha256: dict[str, str]
    contract_path: Path
    contract_file_sha256: str
    truth_access_count_before_freeze: int


@dataclass
class TargetFreeArtifacts:
    block_index: pd.DataFrame
    block_summary: pd.DataFrame
    calibration: pd.DataFrame
    input_manifest: pd.DataFrame
    joint_path: Path
    candidate_path: Path
    registration_path: Path
    freeze: FrozenEvidence


class TruthAccessGuard:
    def __init__(self) -> None:
        self.count = 0

    def mark_truth_access(self) -> None:
        self.count += 1


def resolve_raw_train_dir(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[Path, dict[str, Path]]:
    patterns = list(get_nested(config, "data.raw_train_dir_patterns"))
    horizontal_glob = str(get_nested(config, "data.raw_horizontal_glob"))
    evidence: dict[str, dict[str, int]] = {}
    for directory in [path for path in expand_existing_paths(patterns) if path.is_dir()]:
        files = sorted(directory.glob(horizontal_glob))
        mapping = {path.name.split("__horizontal_well.csv", 1)[0]: path for path in files}
        evidence[str(directory)] = {"files": len(files), "wells": len(mapping)}
        if set(mapping) == expected_wells and len(mapping) == len(files):
            return directory, mapping
    raise FileNotFoundError(
        f"raw train directory with exact exp293 well inventory was not found: {evidence}"
    )


def target_free_input_evidence(
    path: Path,
    role: str,
    frame: pd.DataFrame,
    rows: int,
    wells: int = 1,
    raw_file_is_truth_bearing: bool = False,
) -> dict[str, Any]:
    return {
        "phase": "target_free",
        "role": role,
        "path": str(path),
        "rows": int(rows),
        "wells": int(wells),
        # A raw horizontal CSV also contains suffix TVT. Reading all bytes for a
        # pre-freeze file hash would violate the strict truth boundary even if
        # the digest were never used as a feature. Its raw SHA is recorded only
        # by the post-freeze truth loader; the selected target-safe frame is
        # frozen here by logical and schema SHA.
        "file_sha256": None if raw_file_is_truth_bearing else sha256_file(path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }


def target_free_output_paths(directory: Path) -> dict[str, Path]:
    return {
        "contract": directory / f"{OUTPUT_PREFIX}_contract.json",
        "input_manifest": directory / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "calibration": directory / f"{OUTPUT_PREFIX}_calibration.csv",
        "block_index": directory / f"{OUTPUT_PREFIX}_block_index.csv.gz",
        "joint": directory / f"{OUTPUT_PREFIX}_joint_reliable_posterior.npy",
        "candidate": directory / f"{OUTPUT_PREFIX}_candidate_posterior.parquet",
        "registration": directory / f"{OUTPUT_PREFIX}_registration_posterior.parquet",
        "block_summary": directory / f"{OUTPUT_PREFIX}_block_summary.csv.gz",
    }


def total_block_control_count(bank: FrozenBank, horizons: Sequence[int]) -> int:
    count = 0
    for horizon in horizons:
        groups = bank.keys[GROUP_COLUMNS[int(horizon)]].to_numpy(np.int64)
        count += sum(len(build_block_slices(groups, segment)) for segment in bank.segments)
    return 2 * count


def block_frames(
    positions: np.ndarray,
    blocks: Sequence[BlockSlice],
    segment: WellSegment,
    horizon: int,
    control: str,
    shuffle_offset: int,
    calibration: CalibrationResult,
    posterior: PosteriorResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = np.array([block.start for block in blocks], dtype=np.int64)
    ends = np.array([block.end for block in blocks], dtype=np.int64)
    index = pd.DataFrame(
        {
            "block_position": positions,
            "well": segment.well,
            "fold": segment.fold,
            "horizon_rows": int(horizon),
            "control": control,
            "block_group": np.array([block.group for block in blocks], dtype=np.int64),
            "row_start": segment.start + starts,
            "row_end": segment.start + ends,
            "rows": ends - starts,
        }
    )
    summary = index[
        ["block_position", "well", "fold", "horizon_rows", "control", "block_group"]
    ].copy()
    summary["calibration_valid"] = calibration.valid
    summary["calibration_reason"] = calibration.reason
    summary["shuffle_offset"] = int(shuffle_offset)
    summary["eligible_states"] = posterior.eligible_states
    summary["reliable_probability"] = posterior.reliable_probability
    summary["unreliable_probability"] = posterior.unreliable_probability
    summary["candidate_entropy"] = posterior.candidate_entropy
    summary["registration_entropy"] = posterior.registration_entropy
    summary["candidate_mode_gap"] = posterior.candidate_mode_gap
    summary["log_evidence"] = posterior.log_evidence
    return index, summary


def target_free_contract(
    config: Mapping[str, Any],
    bank: FrozenBank,
    paths: Mapping[str, Path],
    output_hashes: Mapping[str, str],
    block_count: int,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "phase": "target_free_closed_before_truth",
        "contract_date": get_nested(config, "experiment.created_at"),
        "candidate_ids": list(bank.candidate_ids),
        "safe_candidate": SAFE_CANDIDATE,
        "candidate_content_sha256": bank.candidate_content_sha256,
        "registration_deltas_ft": registration_grid(config),
        "controls": ["real", "shuffle"],
        "horizons_rows": get_nested(config, "validation.horizons_rows"),
        "block_control_count": int(block_count),
        "joint_shape": [
            int(block_count),
            len(bank.candidate_ids),
            int(get_nested(config, "audit.registration.expected_states")),
        ],
        "query_formula": get_nested(config, "audit.registration.query_formula"),
        "reliable_prior": get_nested(config, "audit.reliability.reliable_prior"),
        "unreliable_prior": get_nested(config, "audit.reliability.unreliable_prior"),
        "outlier_log_likelihood": get_nested(config, "audit.reliability.outlier_log_likelihood"),
        "truth_access_count_before_freeze": 0,
        "frozen_output_file_sha256": dict(output_hashes),
        "frozen_output_paths": {key: str(path) for key, path in paths.items() if key != "contract"},
        "forbidden_outputs_absent": [
            "selected_tvt_prediction",
            "candidate_weighted_tvt",
            "registration_corrected_tvt",
            "submission",
        ],
    }


def verify_frozen_evidence(freeze: FrozenEvidence) -> None:
    if freeze.truth_access_count_before_freeze != 0:
        raise ValueError("truth was accessed before target-free evidence freeze")
    for path in freeze.paths:
        expected = freeze.file_sha256[str(path)]
        current = sha256_file(path)
        if current != expected:
            raise ValueError(f"target-free artifact changed before truth read: {path}")
    if sha256_file(freeze.contract_path) != freeze.contract_file_sha256:
        raise ValueError("target-free contract changed before truth read")


def run_target_free_phase(
    config: Mapping[str, Any],
    bank: FrozenBank,
    raw_dir: Path,
    horizontal_files: Mapping[str, Path],
    hidden_evidence: Mapping[str, Any],
    truth_guard: TruthAccessGuard,
) -> TargetFreeArtifacts:
    if truth_guard.count != 0:
        raise ValueError("truth access guard is nonzero before target-free phase")
    directory = artifact_dir()
    paths = target_free_output_paths(directory)
    horizons = [int(value) for value in get_nested(config, "validation.horizons_rows")]
    deltas = registration_grid(config)
    total_blocks = total_block_control_count(bank, horizons)
    joint = np.lib.format.open_memmap(
        paths["joint"],
        mode="w+",
        dtype="float32",
        shape=(total_blocks, len(bank.candidate_ids), len(deltas)),
    )
    writers = PosteriorParquetWriters(paths["candidate"], paths["registration"])
    block_indices: list[pd.DataFrame] = []
    block_summaries: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    input_evidence = list(bank.input_evidence) + [dict(hidden_evidence)]
    position = 0
    try:
        for segment in bank.segments:
            horizontal_path = horizontal_files[segment.well]
            typewell_path = raw_dir / f"{segment.well}__typewell.csv"
            if not typewell_path.exists():
                raise FileNotFoundError(f"Type Well missing for {segment.well}: {typewell_path}")
            horizontal = load_target_safe_horizontal(horizontal_path)
            typewell = load_typewell(typewell_path)
            input_evidence.extend(
                [
                    target_free_input_evidence(
                        horizontal_path,
                        "raw_horizontal_target_safe",
                        horizontal,
                        len(horizontal),
                        raw_file_is_truth_bearing=True,
                    ),
                    target_free_input_evidence(
                        typewell_path, "raw_typewell", typewell, len(typewell)
                    ),
                ]
            )
            row_idx = bank.keys["well_row_idx"].to_numpy(np.int64)[segment.start : segment.end]
            if len(row_idx) == 0 or row_idx.min() < 0 or row_idx.max() >= len(horizontal):
                raise ValueError(f"suffix row_idx out of range for well={segment.well}")
            observed = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)[
                row_idx
            ]
            typewell_tvt, typewell_gr = prepare_typewell_curve(typewell)
            calibration = robust_affine_calibration(
                pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64),
                pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64),
                typewell_tvt,
                typewell_gr,
                config,
            )
            shuffled, shuffle_offset = shuffled_preserve_nan_mask(observed, segment.well, config)
            calibration_rows.append(
                {
                    "well": segment.well,
                    "fold": segment.fold,
                    **asdict(calibration),
                    "shuffle_offset": shuffle_offset,
                    "suffix_rows": len(observed),
                    "suffix_finite_gr": int(np.isfinite(observed).sum()),
                }
            )
            candidate_paths = np.asarray(bank.values[segment.start : segment.end], dtype=np.float32)
            reference: np.ndarray | None = None
            derivative_reference: np.ndarray | None = None
            if calibration.valid:
                reference, derivative_reference = registered_forward_matrices(
                    candidate_paths,
                    typewell_tvt,
                    typewell_gr,
                    calibration,
                    deltas,
                )
            for horizon in horizons:
                groups = bank.keys[GROUP_COLUMNS[horizon]].to_numpy(np.int64)
                blocks = build_block_slices(groups, segment)
                for control, values in (("real", observed), ("shuffle", shuffled)):
                    if calibration.valid:
                        assert reference is not None and derivative_reference is not None
                        raw, ncc, derivative, eligible = block_component_matrices(
                            values,
                            reference,
                            derivative_reference,
                            blocks,
                            calibration,
                            config,
                        )
                        posterior = posterior_from_components(
                            raw, ncc, derivative, eligible, deltas, config
                        )
                    else:
                        posterior = fallback_posterior(
                            len(blocks), len(bank.candidate_ids), len(deltas)
                        )
                    positions = np.arange(position, position + len(blocks), dtype=np.int64)
                    joint[positions] = posterior.joint_reliable
                    writers.write(positions, posterior, deltas)
                    index_frame, summary_frame = block_frames(
                        positions,
                        blocks,
                        segment,
                        horizon,
                        control,
                        shuffle_offset,
                        calibration,
                        posterior,
                    )
                    block_indices.append(index_frame)
                    block_summaries.append(summary_frame)
                    position += len(blocks)
    finally:
        writers.close()
        joint.flush()
        del joint
    if position != total_blocks:
        raise ValueError(f"target-free block count mismatch: {position} != {total_blocks}")
    block_index = pd.concat(block_indices, ignore_index=True)
    block_summary = pd.concat(block_summaries, ignore_index=True)
    calibration_frame = pd.DataFrame(calibration_rows).sort_values("well", kind="mergesort")
    input_manifest = pd.DataFrame(input_evidence)
    if block_index["block_position"].duplicated().any():
        raise ValueError("block positions are duplicated")
    if not np.array_equal(
        np.sort(block_index["block_position"].to_numpy(np.int64)),
        np.arange(total_blocks, dtype=np.int64),
    ):
        raise ValueError("block positions are not a complete dense range")
    write_csv_gzip(paths["block_index"], block_index)
    write_csv_gzip(paths["block_summary"], block_summary)
    write_csv(paths["calibration"], calibration_frame)
    write_csv(paths["input_manifest"], input_manifest)
    freeze_paths = tuple(
        paths[key]
        for key in (
            "input_manifest",
            "calibration",
            "block_index",
            "joint",
            "candidate",
            "registration",
            "block_summary",
        )
    )
    file_hashes = {str(path): sha256_file(path) for path in freeze_paths}
    contract = target_free_contract(config, bank, paths, file_hashes, total_blocks)
    write_json(paths["contract"], contract)
    freeze = FrozenEvidence(
        freeze_paths,
        file_hashes,
        paths["contract"],
        sha256_file(paths["contract"]),
        truth_guard.count,
    )
    verify_frozen_evidence(freeze)
    return TargetFreeArtifacts(
        block_index,
        block_summary,
        calibration_frame,
        input_manifest,
        paths["joint"],
        paths["candidate"],
        paths["registration"],
        freeze,
    )


# %% [markdown]
# ## 7. Post-freeze truth readout and Stage-2 decision


# %%
@dataclass
class SSEAccumulator:
    rows: int = 0
    blocks: int = 0
    anchor_sse: float = 0.0
    expected_sse: float = 0.0
    oracle_sse: float = 0.0

    def add(
        self,
        rows: int,
        anchor_sse: float,
        expected_sse: float,
        oracle_sse: float,
    ) -> None:
        self.rows += int(rows)
        self.blocks += 1
        self.anchor_sse += float(anchor_sse)
        self.expected_sse += float(expected_sse)
        self.oracle_sse += float(oracle_sse)


def load_truth_after_freeze(
    config: Mapping[str, Any],
    bank: FrozenBank,
    horizontal_files: Mapping[str, Path],
    freeze: FrozenEvidence,
    guard: TruthAccessGuard,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    verify_frozen_evidence(freeze)
    if guard.count != 0:
        raise ValueError("truth guard was incremented before post-freeze loader")
    truth = np.full(len(bank.keys), np.nan, dtype=np.float64)
    evidence: list[dict[str, Any]] = []
    for segment in bank.segments:
        path = horizontal_files[segment.well]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if "TVT" not in header:
            raise ValueError(f"truth column TVT missing: {path}")
        guard.mark_truth_access()
        values = pd.to_numeric(pd.read_csv(path, usecols=["TVT"])["TVT"], errors="coerce").to_numpy(
            np.float64
        )
        row_idx = bank.keys["well_row_idx"].to_numpy(np.int64)[segment.start : segment.end]
        selected = values[row_idx]
        if not np.isfinite(selected).all():
            raise ValueError(f"non-finite suffix truth for well={segment.well}")
        truth[segment.start : segment.end] = selected
        evidence.append(
            {
                "phase": "post_freeze_truth",
                "role": "raw_horizontal_truth",
                "path": str(path),
                "rows": len(values),
                "suffix_rows": len(selected),
                "wells": 1,
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": None,
                "logical_content_sha256": None,
                "schema_sha256": None,
            }
        )
    if not np.isfinite(truth).all():
        raise ValueError("post-freeze truth vector is incomplete")
    digest = hashlib.sha256()
    digest.update(str(get_nested(config, "data.exp293.key_content_sha256")).encode())
    digest.update(np.asarray(truth, dtype="<f8").tobytes())
    return truth, evidence, digest.hexdigest()


def candidate_posterior_from_joint(joint: np.ndarray, unreliable_probability: float) -> np.ndarray:
    probability = np.asarray(joint, dtype=np.float64).sum(axis=1)
    probability[SAFE_INDEX] += float(unreliable_probability)
    if not np.isclose(probability.sum(), 1.0, atol=2.0e-6):
        raise ValueError("frozen candidate posterior mass is invalid")
    return probability


def block_sse(truth: np.ndarray, candidates: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.full(candidates.shape[1], np.nan, dtype=np.float64)
    residual = np.asarray(candidates[mask], dtype=np.float64) - truth[mask, None]
    return np.sum(residual * residual, axis=0)


def update_sse_accumulator(
    accumulator: SSEAccumulator,
    truth: np.ndarray,
    candidates: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
) -> None:
    candidate_sse = block_sse(truth, candidates, mask)
    if not np.isfinite(candidate_sse).all():
        return
    accumulator.add(
        int(mask.sum()),
        float(candidate_sse[SAFE_INDEX]),
        float(np.dot(probability, candidate_sse)),
        float(np.min(candidate_sse)),
    )


def accumulator_row(
    accumulator: SSEAccumulator,
    *,
    control: str,
    horizon: int,
    scope: str,
    fold: int,
) -> dict[str, Any]:
    denominator = accumulator.anchor_sse - accumulator.oracle_sse
    recovery = (
        (accumulator.anchor_sse - accumulator.expected_sse) / denominator
        if denominator > 0
        else float("nan")
    )
    return {
        "control": control,
        "horizon_rows": int(horizon),
        "scope": scope,
        "fold": int(fold),
        "rows": accumulator.rows,
        "blocks": accumulator.blocks,
        "anchor_sse": accumulator.anchor_sse,
        "expected_sse": accumulator.expected_sse,
        "oracle_sse": accumulator.oracle_sse,
        "anchor_rmse": math.sqrt(accumulator.anchor_sse / accumulator.rows)
        if accumulator.rows
        else float("nan"),
        "expected_rmse": math.sqrt(accumulator.expected_sse / accumulator.rows)
        if accumulator.rows
        else float("nan"),
        "oracle_rmse": math.sqrt(accumulator.oracle_sse / accumulator.rows)
        if accumulator.rows
        else float("nan"),
        "headroom_recovery": recovery,
    }


def compute_truth_readout(
    config: Mapping[str, Any],
    bank: FrozenBank,
    artifacts: TargetFreeArtifacts,
    truth: np.ndarray,
    hidden_sets: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_index = artifacts.block_index.set_index("block_position", drop=False)
    summary = artifacts.block_summary.set_index("block_position", drop=False)
    if not block_index.index.equals(summary.index):
        raise ValueError("frozen block index/summary positions differ")
    joint = np.load(artifacts.joint_path, mmap_mode="r")
    if joint.shape[0] != len(block_index):
        raise ValueError("frozen joint posterior/block index row mismatch")
    aggregate: dict[tuple[str, int, str, int], SSEAccumulator] = {}
    subgroup: dict[str, SSEAccumulator] = {
        "md_since_1000_plus": SSEAccumulator(),
        "hidden_like_spatial": SSEAccumulator(),
        "hidden_like_typewell_purged": SSEAccumulator(),
    }
    primary_horizon = int(get_nested(config, "audit.truth_readout.subgroup_horizon_rows"))
    md_since = bank.keys["md_since"].to_numpy(np.float64)
    for position in range(len(block_index)):
        record = block_index.loc[position]
        start = int(record["row_start"])
        end = int(record["row_end"])
        control = str(record["control"])
        horizon = int(record["horizon_rows"])
        fold = int(record["fold"])
        well = str(record["well"])
        local_truth = truth[start:end]
        local_candidates = np.asarray(bank.values[start:end], dtype=np.float64)
        all_rows = np.ones(len(local_truth), dtype=bool)
        probability = candidate_posterior_from_joint(
            joint[position], float(summary.loc[position, "unreliable_probability"])
        )
        for scope, scope_fold in (("pooled", -1), ("fold", fold)):
            key = (control, horizon, scope, scope_fold)
            update_sse_accumulator(
                aggregate.setdefault(key, SSEAccumulator()),
                local_truth,
                local_candidates,
                probability,
                all_rows,
            )
        if control == "real" and horizon == primary_horizon:
            update_sse_accumulator(
                subgroup["md_since_1000_plus"],
                local_truth,
                local_candidates,
                probability,
                md_since[start:end] >= 1000.0,
            )
            if well in hidden_sets["hidden_like_spatial"]:
                update_sse_accumulator(
                    subgroup["hidden_like_spatial"],
                    local_truth,
                    local_candidates,
                    probability,
                    all_rows,
                )
            if well in hidden_sets["hidden_like_typewell_purged"]:
                update_sse_accumulator(
                    subgroup["hidden_like_typewell_purged"],
                    local_truth,
                    local_candidates,
                    probability,
                    all_rows,
                )
    metric_rows = [
        accumulator_row(
            accumulator,
            control=control,
            horizon=horizon,
            scope=scope,
            fold=fold,
        )
        for (control, horizon, scope, fold), accumulator in sorted(aggregate.items())
    ]
    subgroup_rows = []
    for name, accumulator in subgroup.items():
        row = accumulator_row(
            accumulator,
            control="real",
            horizon=primary_horizon,
            scope="subgroup",
            fold=-1,
        )
        row["subgroup"] = name
        row["anchor_nonregression"] = bool(
            accumulator.rows > 0 and accumulator.expected_sse <= accumulator.anchor_sse
        )
        subgroup_rows.append(row)
    del joint
    return pd.DataFrame(metric_rows), pd.DataFrame(subgroup_rows)


def unique_metric(
    metrics: pd.DataFrame,
    *,
    control: str,
    horizon: int,
    scope: str,
    fold: int,
) -> pd.Series:
    selected = metrics.loc[
        metrics["control"].eq(control)
        & metrics["horizon_rows"].eq(horizon)
        & metrics["scope"].eq(scope)
        & metrics["fold"].eq(fold)
    ]
    if len(selected) != 1:
        raise ValueError(f"readout metric missing/duplicated: {control}/{horizon}/{scope}/{fold}")
    return selected.iloc[0]


def stage2_decision(
    config: Mapping[str, Any],
    metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    freeze: FrozenEvidence,
) -> dict[str, Any]:
    criteria = get_nested(config, "validation.success_criteria")
    primary = int(get_nested(config, "validation.primary_horizon_rows"))
    continuity = int(get_nested(config, "validation.continuity_horizon_rows"))
    real = unique_metric(metrics, control="real", horizon=primary, scope="pooled", fold=-1)
    shuffled = unique_metric(metrics, control="shuffle", horizon=primary, scope="pooled", fold=-1)
    continuity_row = unique_metric(
        metrics, control="real", horizon=continuity, scope="pooled", fold=-1
    )
    fold_real = [
        unique_metric(metrics, control="real", horizon=primary, scope="fold", fold=fold)
        for fold in range(int(get_nested(config, "validation.n_folds")))
    ]
    fold_shuffle = [
        unique_metric(metrics, control="shuffle", horizon=primary, scope="fold", fold=fold)
        for fold in range(int(get_nested(config, "validation.n_folds")))
    ]
    subgroup_map = subgroup_metrics.set_index("subgroup")
    checks = {
        "pooled_h256_recovery": bool(
            np.isfinite(real["headroom_recovery"])
            and float(real["headroom_recovery"]) >= float(criteria["minimum_pooled_h256_recovery"])
        ),
        "positive_h256_recovery_all_folds": sum(
            bool(np.isfinite(row["headroom_recovery"]) and row["headroom_recovery"] > 0)
            for row in fold_real
        )
        >= int(criteria["require_positive_h256_recovery_folds"]),
        "real_better_than_shuffle_pooled": bool(real["expected_sse"] < shuffled["expected_sse"]),
        "real_better_than_shuffle_all_folds": sum(
            bool(real_row["expected_sse"] < shuffle_row["expected_sse"])
            for real_row, shuffle_row in zip(fold_real, fold_shuffle, strict=True)
        )
        >= int(criteria["require_real_better_than_shuffle_folds"]),
        "h512_continuity": bool(
            np.isfinite(continuity_row["headroom_recovery"])
            and np.isfinite(real["headroom_recovery"])
            and float(continuity_row["headroom_recovery"])
            >= float(real["headroom_recovery"])
            - float(criteria["maximum_h512_recovery_drop_from_h256"])
        ),
        "md_since_1000_plus_anchor_nonregression": bool(
            subgroup_map.loc["md_since_1000_plus", "anchor_nonregression"]
        ),
        "hidden_like_spatial_anchor_nonregression": bool(
            subgroup_map.loc["hidden_like_spatial", "anchor_nonregression"]
        ),
        "hidden_like_typewell_purged_anchor_nonregression": bool(
            subgroup_map.loc["hidden_like_typewell_purged", "anchor_nonregression"]
        ),
        "truth_access_before_freeze_zero": bool(freeze.truth_access_count_before_freeze == 0),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_STAGE3" if passed else "FAIL_STOP_NO_STAGE4",
        "all_checks_passed": passed,
        "checks": checks,
        "primary": {
            "h256_real_recovery": real["headroom_recovery"],
            "h256_real_expected_rmse": real["expected_rmse"],
            "h256_shuffle_expected_rmse": shuffled["expected_rmse"],
            "h512_real_recovery": continuity_row["headroom_recovery"],
        },
    }


def final_output_paths(directory: Path) -> dict[str, Path]:
    return {
        "metrics": directory / f"{OUTPUT_PREFIX}_readout_metrics.csv",
        "subgroups": directory / f"{OUTPUT_PREFIX}_subgroup_metrics.csv",
        "sha_manifest": directory / f"{OUTPUT_PREFIX}_sha_manifest.csv",
        "summary": directory / f"{OUTPUT_PREFIX}_summary.json",
        "metrics_json": directory.parent / "metrics.json",
    }


def build_sha_manifest(
    target_free: TargetFreeArtifacts,
    truth_evidence: Sequence[Mapping[str, Any]],
    truth_content_sha256: str,
    final_paths: Mapping[str, Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in target_free.freeze.paths:
        rows.append(
            {
                "phase": "target_free_frozen",
                "role": path.name,
                "path": str(path),
                "file_sha256": target_free.freeze.file_sha256[str(path)],
                "logical_content_sha256": None,
            }
        )
    rows.append(
        {
            "phase": "target_free_frozen",
            "role": target_free.freeze.contract_path.name,
            "path": str(target_free.freeze.contract_path),
            "file_sha256": target_free.freeze.contract_file_sha256,
            "logical_content_sha256": None,
        }
    )
    for item in truth_evidence:
        rows.append(
            {
                "phase": "post_freeze_truth",
                "role": item["role"],
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "logical_content_sha256": None,
            }
        )
    rows.append(
        {
            "phase": "post_freeze_truth",
            "role": "aligned_truth_vector",
            "path": "memory_only_not_persisted",
            "file_sha256": None,
            "logical_content_sha256": truth_content_sha256,
        }
    )
    for role, path in final_paths.items():
        if role == "sha_manifest" or not path.exists():
            continue
        rows.append(
            {
                "phase": "post_freeze_readout",
                "role": role,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "logical_content_sha256": None,
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. Setup and execution


# %%
def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    exact = {
        "experiment.route": "pf_beam",
        "lineage.parent": "exp293_physics_only_candidate_bank_headroom_contract",
        "validation.horizons_rows": [128, 256, 512],
        "validation.primary_horizon_rows": 256,
        "validation.continuity_horizon_rows": 512,
        "candidate_bank.order": list(EXPECTED_CANDIDATES),
        "candidate_bank.safe_candidate": SAFE_CANDIDATE,
        "candidate_bank.regenerate_candidates": False,
        "audit.prefix_calibration.maximum_rows": 512,
        "audit.prefix_calibration.minimum_pairs": 64,
        "audit.prefix_calibration.irls_iterations": 2,
        "audit.prefix_calibration.slope_clip": [0.25, 4.0],
        "audit.residual_component.degrees_of_freedom": 4.0,
        "audit.residual_component.scale_clip": [10.0, 60.0],
        "audit.registration.query_formula": "candidate_tvt_plus_delta",
        "audit.registration.minimum_ft": -20.0,
        "audit.registration.maximum_ft": 20.0,
        "audit.registration.step_ft": 2.0,
        "audit.registration.expected_states": 21,
        "audit.registration.prior_scale_ft": 10.0,
        "audit.composite.weights": [1.0 / 3.0] * 3,
        "audit.reliability.reliable_prior": 0.9,
        "audit.reliability.unreliable_prior": 0.1,
        "audit.reliability.outlier_log_likelihood": 0.0,
        "audit.shuffled_control.seed": 42,
        "candidate_bank.persist_selected_row_prediction": False,
        "audit.truth_readout.persist_truth_joined_candidate_rows": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"Stage-2 scientific contract mismatch: {mismatches}")
    priors = float(get_nested(config, "audit.reliability.reliable_prior")) + float(
        get_nested(config, "audit.reliability.unreliable_prior")
    )
    if not np.isclose(priors, 1.0):
        raise ValueError("reliable/unreliable priors must sum to one")


def validate_execution_contract(config: Mapping[str, Any], *, require_run: bool) -> None:
    fixed_zero = (
        "execution.lightgbm_config_count",
        "execution.trained_fold_count",
        "execution.total_boosters",
        "execution.hmm_pf_well_runs",
    )
    if any(int(get_nested(config, key)) != 0 for key in fixed_zero):
        raise ValueError("Stage-2 must not train a model or regenerate PF/Beam paths")
    fixed_false = (
        "execution.control_or_parent_retraining",
        "execution.gpu",
        "execution.inference",
        "execution.submission",
        "inference.enabled",
        "inference.create_submission",
    )
    if any(bool(get_nested(config, key)) for key in fixed_false):
        raise ValueError("Stage-2 execution scope enables a forbidden action")
    if require_run:
        if not bool(get_nested(config, "execution.implementation")):
            raise PermissionError("implementation is not marked complete")
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise PermissionError("Kaggle audit execution is fail-closed until separately approved")


def run_audit() -> dict[str, Any]:
    started = time.time()
    config_path = find_config_path()
    config = read_yaml(config_path)
    validate_scientific_contract(config)
    validate_execution_contract(config, require_run=True)
    bank = load_frozen_exp293_bank(config)
    expected_wells = set(bank.keys["well"].astype(str))
    raw_dir, horizontal_files = resolve_raw_train_dir(config, expected_wells)
    hidden_sets, hidden_evidence = load_hidden_like_sets(config, expected_wells)
    guard = TruthAccessGuard()
    target_free = run_target_free_phase(
        config,
        bank,
        raw_dir,
        horizontal_files,
        hidden_evidence,
        guard,
    )
    verify_frozen_evidence(target_free.freeze)
    truth, truth_evidence, truth_sha = load_truth_after_freeze(
        config,
        bank,
        horizontal_files,
        target_free.freeze,
        guard,
    )
    metrics, subgroups = compute_truth_readout(config, bank, target_free, truth, hidden_sets)
    decision = stage2_decision(config, metrics, subgroups, target_free.freeze)
    directory = artifact_dir()
    final_paths = final_output_paths(directory)
    write_csv(final_paths["metrics"], metrics)
    write_csv(final_paths["subgroups"], subgroups)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "completed",
        **decision,
        "truth_access_count_before_freeze": target_free.freeze.truth_access_count_before_freeze,
        "truth_access_count_after_freeze": guard.count,
        "candidate_content_sha256": bank.candidate_content_sha256,
        "truth_content_sha256": truth_sha,
        "rows": len(bank.keys),
        "wells": len(bank.segments),
        "block_control_count": len(target_free.block_index),
        "valid_calibration_wells": int(target_free.calibration["valid"].sum()),
        "invalid_calibration_wells": int((~target_free.calibration["valid"]).sum()),
        "elapsed_seconds": time.time() - started,
        "next_action": "stage3_only" if decision["all_checks_passed"] else "stop_no_stage4",
    }
    write_json(final_paths["summary"], summary)
    write_json(final_paths["metrics_json"], summary)
    sha_manifest = build_sha_manifest(target_free, truth_evidence, truth_sha, final_paths)
    write_csv(final_paths["sha_manifest"], sha_manifest)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


validate_scientific_contract(read_yaml(find_config_path()))
validate_execution_contract(read_yaml(find_config_path()), require_run=False)

if EXECUTE_NOTEBOOK:
    RESULT = run_audit()
