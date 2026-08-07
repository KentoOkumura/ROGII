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
# # exp405 geometry-reinjected interval semi-Markov fusion
#
# This implementation-only train candidate reads the frozen exp293 deployable12
# paths, scores block-local Type Well GR morphology without suffix TVT, applies
# an exact explicit-duration semi-Markov posterior, and freezes scores,
# posterior probabilities, and predictions before any truth or hidden-like role
# is read. The canonical notebook, Kaggle package, current-test generator,
# inference, and submission remain disabled.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Scientific contract and execution guards
# 4. Frozen exp293 bank and target-safe raw inputs
# 5. Block-local multiscale GR morphology and negative controls
# 6. Exact interval semi-Markov posterior
# 7. Posterior interpolation, prediction, and pre-truth freeze
# 8. Fixed16 resource preflight
# 9. Post-freeze truth, constrained oracle, metrics, and gates
# 10. Setup and execution

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import io
import json
import math
import os
import resource
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

EXPERIMENT_NAME = "exp405_geometry_reinjected_interval_semimarkov_fusion"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
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
GEOMETRY_CANDIDATE = "exp226_k16"
SAFE_CANDIDATE = "exp226_w500_50_50"
GEOMETRY_INDEX = EXPECTED_CANDIDATES.index(GEOMETRY_CANDIDATE)
SAFE_INDEX = EXPECTED_CANDIDATES.index(SAFE_CANDIDATE)
CONTROL_NAMES = ("real", "circular", "block_permutation")
FORBIDDEN_PRETRUTH_COLUMNS = {
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


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP405_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    nested = start / "experiments" / EXPERIMENT_NAME
    if nested.exists():
        return start
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
    raise FileNotFoundError("exp405 config.yaml was not found unambiguously")


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


def reject_pretruth_columns(columns: Iterable[str]) -> None:
    normalized = {str(column) for column in columns}
    forbidden = normalized & FORBIDDEN_PRETRUTH_COLUMNS
    forbidden |= {
        column
        for column in normalized
        if any(token in column.lower() for token in ("true_tvt", "abs_error", "oracle_"))
    }
    if forbidden:
        raise ValueError(f"pre-truth input exposes forbidden columns: {sorted(forbidden)}")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(
    path: Path, chunk_bytes: int = 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    string_columns = [
        column
        for column, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.StringDtype)
    ]
    if not string_columns:
        return frame
    normalized = frame.copy()
    for column in string_columns:
        normalized[column] = normalized[column].astype(object)
    return normalized


def frame_content_sha256(
    frame: pd.DataFrame, columns: Iterable[str] | None = None
) -> str:
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
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode()
    ).hexdigest()


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


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text,
                    index=False,
                    float_format="%.12g",
                    lineterminator="\n",
                )


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    found: dict[str, Path] = {}
    root = project_root()
    for raw_value in patterns:
        raw = str(raw_value)
        path = Path(raw)
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            found.setdefault(str(direct.resolve()), direct)
        for match in glob.glob(raw, recursive=True):
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
    patterns: Sequence[str],
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    if expected_sha256:
        matching = [path for path in candidates if sha256_file(path) == expected_sha256]
        if matching:
            return sorted(matching, key=lambda item: len(str(item)))[0]
        if candidates:
            evidence = {str(path): sha256_file(path) for path in candidates}
            raise ValueError(f"{label} SHA mismatch: {evidence}")
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


def runtime_metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return experiment_dir() / "metrics.json"


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2)


def stable_hash_int(*parts: str) -> int:
    material = "::".join(parts).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def stable_logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    if axis is None:
        finite = array[np.isfinite(array)]
        if len(finite) == 0:
            return float("-inf")
        maximum = float(np.max(finite))
        return maximum + float(np.log(np.exp(finite - maximum).sum()))
    maximum = np.max(array, axis=axis, keepdims=True)
    valid = np.isfinite(maximum)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        shifted = np.where(valid, array - maximum, -np.inf)
        total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        result = np.where(valid, maximum + np.log(total), -np.inf)
    return np.squeeze(result, axis=axis)


# %% [markdown]
# ## 3. Scientific contract and execution guards


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], run_stage: str | None = None
) -> dict[str, int]:
    expected_values = {
        "experiment.route": "pf_beam",
        "lineage.parent": "exp293_physics_only_candidate_bank_headroom_contract",
        "candidate_bank.expected_count": 12,
        "candidate_bank.geometry_candidate": GEOMETRY_CANDIDATE,
        "candidate_bank.safe_candidate": SAFE_CANDIDATE,
        "observation.base_block_rows": 256,
        "observation.typewell_query.minimum_shift_ft": -55.0,
        "observation.typewell_query.maximum_shift_ft": 55.0,
        "observation.typewell_query.step_ft": 5.0,
        "observation.typewell_query.expected_states": 23,
        "observation.typewell_query.prior_scale_ft": 20.0,
        "observation.typewell_query.add_shift_to_output_tvt": False,
        "observation.reliability.reliable_mass": 0.8,
        "observation.reliability.candidate_common_neutral_mass": 0.2,
        "semimarkov.solver": "exact_log_space_forward_backward",
        "semimarkov.minimum_duration_blocks": 2,
        "semimarkov.segment_switch_penalty.log_cost": math.log(9.0),
        "semimarkov.transition.geometry_floor_if_current_is_not_geometry": 0.1,
        "semimarkov.transition.geometry_floor_depends_on_docking": False,
        "semimarkov.prediction.use_viterbi": False,
        "semimarkov.prediction.hard_top1": False,
        "model.fitted_model": False,
        "runtime.model_configs": 0,
        "runtime.trained_folds": 0,
        "runtime.lightgbm_boosters": 0,
        "runtime.hmm_runs": 0,
        "runtime.pf_runs": 0,
        "runtime.beam_runs": 0,
        "runtime.parent_control_regeneration": False,
        "implementation.current_test_implementation_enabled": False,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
    }
    for key, expected in expected_values.items():
        actual = get_nested(config, key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(f"contract mismatch {key}: {actual} != {expected}")
        elif actual != expected:
            raise ValueError(f"contract mismatch {key}: {actual} != {expected}")
    if tuple(get_nested(config, "candidate_bank.order")) != EXPECTED_CANDIDATES:
        raise ValueError("candidate order differs from frozen exp293 order")
    if len(get_nested(config, "negative_controls.controls")) != 2:
        raise ValueError("exactly two negative controls are required")
    if not bool(get_nested(config, "implementation.implementation_approval_received")):
        raise RuntimeError("implementation is not approved")

    stage = str(run_stage or get_nested(config, "execution.run_stage"))
    if stage == "fixed16_preflight":
        if not bool(get_nested(config, "execution.run_fixed16_preflight")):
            raise RuntimeError("run_fixed16_preflight is disabled")
        if not bool(get_nested(config, "execution.fixed16_preflight_approved")):
            raise RuntimeError("fixed16 preflight is not separately approved")
        if not bool(get_nested(config, "execution.kaggle_execution_authorized")):
            raise RuntimeError("Kaggle execution is not authorized")
    elif stage == "full_saved_oof":
        if not bool(get_nested(config, "execution.run_full_saved_oof")):
            raise RuntimeError("run_full_saved_oof is disabled")
        if not bool(get_nested(config, "execution.full_saved_oof_approved")):
            raise RuntimeError("full saved OOF is not separately approved")
        if not bool(get_nested(config, "execution.fixed16_preflight_passed")):
            raise RuntimeError("fixed16 preflight has not passed")
        if not str(get_nested(config, "execution.fixed16_preflight_summary_sha256")):
            raise RuntimeError("fixed16 preflight evidence SHA is missing")
        if not bool(get_nested(config, "execution.kaggle_execution_authorized")):
            raise RuntimeError("Kaggle execution is not authorized")
    elif stage != "implementation_only":
        raise ValueError(f"unsupported run_stage: {stage}")
    return {
        "scientific_endpoints": 1,
        "negative_controls": 2,
        "reporting_folds": int(get_nested(config, "validation.n_folds")),
        "fixed_candidates": len(EXPECTED_CANDIDATES),
        "models": 0,
        "boosters": 0,
        "pf_runs": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "parent_reruns": 0,
    }


# %% [markdown]
# ## 4. Frozen exp293 bank and target-safe raw inputs


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
            digest.update(
                np.asarray(values[start:end, position], dtype="<f4").tobytes()
            )
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
        row_idx = keys["well_row_idx"].to_numpy(np.int64)[start:end]
        folds = keys["outer_fold"].to_numpy(np.int64)[start:end]
        if not np.all(wells[start:end] == well):
            raise ValueError("well rows are non-contiguous")
        if len(row_idx) > 1 and not np.all(np.diff(row_idx) == 1):
            raise ValueError(f"non-contiguous row_idx for well={well}")
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
    keys = pd.read_csv(assignment_path, dtype=BLOCK_ASSIGNMENT_DTYPES)
    required = set(KEY_COLUMNS) | {"well_code", "h256_group", "whole_well_group"}
    if missing := sorted(required - set(keys.columns)):
        raise ValueError(f"exp293 block assignment missing columns: {missing}")
    reject_pretruth_columns(keys.columns)
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(keys) != expected_rows or keys["well"].nunique() != expected_wells:
        raise ValueError("exp293 row/well contract mismatch")
    if set(keys["outer_fold"].unique()) != set(range(5)):
        raise ValueError("exp293 fold inventory mismatch")
    if frame_content_sha256(keys, KEY_COLUMNS) != str(spec["key_content_sha256"]):
        raise ValueError("exp293 key content SHA mismatch")
    logical_assignment_sha = frame_content_sha256(keys)
    if logical_assignment_sha != str(spec["block_assignment_logical_sha256"]):
        raise ValueError("exp293 block assignment logical SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    candidate_ids = tuple(str(item) for item in manifest["candidate_ids"])
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("exp293 candidate order mismatch")
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
        raise ValueError("exp293 candidate content SHA mismatch")
    if candidate_sha != str(manifest["candidate_content_sha256"]):
        raise ValueError("exp293 bank manifest candidate SHA mismatch")
    evidence = [
        {
            "phase": "pretruth",
            "role": "exp293_candidate_matrix",
            "path": str(matrix_path),
            "rows": expected_rows,
            "wells": expected_wells,
            "file_sha256": sha256_file(matrix_path),
            "decompressed_content_sha256": None,
            "logical_content_sha256": candidate_sha,
            "schema_sha256": stable_json_sha256(
                {"dtype": "float32", "shape": [expected_rows, len(candidate_ids)]}
            ),
        },
        {
            "phase": "pretruth",
            "role": "exp293_bank_manifest",
            "path": str(manifest_path),
            "rows": 1,
            "wells": expected_wells,
            "file_sha256": sha256_file(manifest_path),
            "decompressed_content_sha256": None,
            "logical_content_sha256": stable_json_sha256(manifest),
            "schema_sha256": None,
        },
        {
            "phase": "pretruth",
            "role": "exp293_block_assignment",
            "path": str(assignment_path),
            "rows": expected_rows,
            "wells": expected_wells,
            "file_sha256": sha256_file(assignment_path),
            "decompressed_content_sha256": decompressed_sha,
            "logical_content_sha256": logical_assignment_sha,
            "schema_sha256": frame_schema_sha256(keys),
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
    frame = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
    reject_pretruth_columns(frame.columns)
    return frame


def load_typewell(path: Path) -> pd.DataFrame:
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
    original = np.asarray(query)
    flat = np.asarray(query, dtype=np.float64).reshape(-1)
    values = np.interp(flat, typewell_tvt, typewell_gr, left=np.nan, right=np.nan)
    return values.reshape(original.shape)


def target_safe_evidence(
    path: Path, role: str, frame: pd.DataFrame, well: str
) -> dict[str, Any]:
    return {
        "phase": "pretruth",
        "role": role,
        "path": str(path),
        "well": well,
        "rows": len(frame),
        "wells": 1,
        # A horizontal file contains suffix TVT bytes, so its raw file SHA is
        # intentionally deferred until after prediction freeze.
        "file_sha256": sha256_file(path) if role == "raw_typewell" else None,
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }


# %% [markdown]
# ## 5. Block-local multiscale GR morphology and negative controls


# %%
@dataclass(frozen=True)
class BlockSlice:
    group: int
    start: int
    end: int


@dataclass(frozen=True)
class MorphologyScore:
    log_emission: np.ndarray
    valid_shift_count: np.ndarray
    eligible: np.ndarray
    shift_log_likelihood: np.ndarray


def build_block_slices(
    groups: np.ndarray, segment: WellSegment | None = None
) -> list[BlockSlice]:
    source = np.asarray(groups, dtype=np.int64)
    local = source if segment is None else source[segment.start : segment.end]
    if len(local) == 0:
        return []
    changes = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1, len(local)]
    blocks = [
        BlockSlice(int(local[left]), int(left), int(right))
        for left, right in zip(changes[:-1], changes[1:], strict=True)
    ]
    if sum(block.end - block.start for block in blocks) != len(local):
        raise ValueError("H256 blocks do not cover the well suffix")
    return blocks


def shift_grid(config: Mapping[str, Any]) -> np.ndarray:
    spec = get_nested(config, "observation.typewell_query")
    values = np.arange(
        float(spec["minimum_shift_ft"]),
        float(spec["maximum_shift_ft"]) + 0.5 * float(spec["step_ft"]),
        float(spec["step_ft"]),
        dtype=np.float64,
    )
    if len(values) != int(spec["expected_states"]):
        raise ValueError("registration shift inventory mismatch")
    return values


def laplace_shift_log_prior(shifts: np.ndarray, scale_ft: float) -> np.ndarray:
    log_weights = -np.abs(np.asarray(shifts, dtype=np.float64)) / float(scale_ft)
    return log_weights - float(stable_logsumexp(log_weights))


def centered_full_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("rolling input must have a row dimension")
    if window <= 0 or window % 2 == 0:
        raise ValueError("rolling window must be a positive odd integer")
    result = np.full_like(array, np.nan, dtype=np.float64)
    if window == 1:
        result[:] = array
        return result
    if len(array) < window:
        return result
    finite = np.isfinite(array)
    safe = np.where(finite, array, 0.0)
    prefix = np.concatenate(
        [np.zeros((1, *array.shape[1:]), dtype=np.float64), np.cumsum(safe, axis=0)],
        axis=0,
    )
    count_prefix = np.concatenate(
        [
            np.zeros((1, *array.shape[1:]), dtype=np.int32),
            np.cumsum(finite, axis=0, dtype=np.int32),
        ],
        axis=0,
    )
    sums = prefix[window:] - prefix[:-window]
    counts = count_prefix[window:] - count_prefix[:-window]
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts == window,
    )
    half = window // 2
    result[half : len(array) - half] = means
    return result


def robust_standardize_rows(
    values: np.ndarray, scale_clip: Sequence[float]
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(array, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(array - median), axis=0, keepdims=True)
    low, high = [float(item) for item in scale_clip]
    scale = np.clip(1.4826 * mad, low, high)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (array - median) / scale


def component_residual_mean(
    observed: np.ndarray,
    reference: np.ndarray,
    scale_clip: Sequence[float],
    cap_z2: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = robust_standardize_rows(np.asarray(observed)[:, None, None], scale_clip)
    y = robust_standardize_rows(reference, scale_clip)
    valid = np.isfinite(x) & np.isfinite(y)
    residual_z2 = np.minimum(np.square(x - y), float(cap_z2))
    count = valid.sum(axis=0)
    total = np.where(valid, residual_z2, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return mean, count


def score_block_morphology(
    observed_block: np.ndarray,
    candidate_block: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: Mapping[str, Any],
) -> MorphologyScore:
    observed = np.asarray(observed_block, dtype=np.float64)
    candidates = np.asarray(candidate_block, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[0] != len(observed):
        raise ValueError("candidate/observed block shape mismatch")
    shifts = shift_grid(config)
    query = candidates[:, :, None] + shifts[None, None, :]
    reference = interpolate_no_extrapolation(query, typewell_tvt, typewell_gr)
    components = get_nested(config, "observation.components")
    scale_clip = get_nested(config, "observation.normalization.scale_clip")
    minimum_pairs = int(
        get_nested(config, "observation.normalization.minimum_finite_pairs")
    )
    minimum_fraction = float(
        get_nested(config, "observation.normalization.minimum_pair_fraction")
    )
    required = max(minimum_pairs, int(math.ceil(minimum_fraction * len(observed))))
    cap_z2 = float(get_nested(config, "observation.residual.cap_z2"))
    weighted = np.zeros((candidates.shape[1], len(shifts)), dtype=np.float64)
    eligible_shift = np.ones(
        (candidates.shape[1], len(shifts)), dtype=bool
    )
    for name in ("raw", "rolling_21", "rolling_101"):
        window = int(components[name]["window_rows"])
        weight = float(components[name]["weight"])
        observed_component = centered_full_window_mean(observed, window)
        reference_component = centered_full_window_mean(reference, window)
        mean, count = component_residual_mean(
            observed_component,
            reference_component,
            scale_clip,
            cap_z2,
        )
        weighted += weight * np.where(np.isfinite(mean), mean, 0.0)
        eligible_shift &= count >= required
    likelihood_scale = float(
        get_nested(config, "observation.residual.log_likelihood_scale")
    )
    shift_log_likelihood = likelihood_scale * weighted
    shift_log_likelihood = np.where(
        eligible_shift, shift_log_likelihood, -np.inf
    )
    log_prior = laplace_shift_log_prior(
        shifts,
        float(get_nested(config, "observation.typewell_query.prior_scale_ft")),
    )
    marginalized = np.asarray(
        stable_logsumexp(shift_log_likelihood + log_prior[None, :], axis=1),
        dtype=np.float64,
    )
    reliable = float(get_nested(config, "observation.reliability.reliable_mass"))
    neutral = float(
        get_nested(
            config, "observation.reliability.candidate_common_neutral_mass"
        )
    )
    valid_shift_count = eligible_shift.sum(axis=1).astype(np.int16)
    eligible = valid_shift_count > 0
    reliable_probability = np.zeros(candidates.shape[1], dtype=np.float64)
    if eligible.any():
        reliable_log_normalizer = float(stable_logsumexp(marginalized[eligible]))
        reliable_probability[eligible] = np.exp(
            marginalized[eligible] - reliable_log_normalizer
        )
    else:
        reliable_probability[:] = 1.0 / candidates.shape[1]
    # The unreliable component is exactly candidate-common. This prevents a
    # candidate with no valid Type Well coverage from beating valid candidates
    # merely because capped residual log likelihoods are non-positive.
    emission_probability = (
        reliable * reliable_probability
        + neutral / candidates.shape[1]
    )
    log_emission = np.log(emission_probability)
    return MorphologyScore(
        log_emission=log_emission,
        valid_shift_count=valid_shift_count,
        eligible=eligible,
        shift_log_likelihood=shift_log_likelihood,
    )


def circular_control(
    observed: np.ndarray, well: str, config: Mapping[str, Any]
) -> tuple[np.ndarray, int]:
    result = np.asarray(observed, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(result))
    if len(finite) <= 1:
        return result, 0
    minimum_fraction = float(
        get_nested(config, "negative_controls.minimum_circular_rotation_fraction")
    )
    minimum = min(
        max(1, int(math.ceil(minimum_fraction * len(finite)))),
        len(finite) // 2,
    )
    allowed = np.arange(minimum, len(finite) - minimum + 1, dtype=np.int64)
    if len(allowed) == 0:
        allowed = np.arange(1, len(finite), dtype=np.int64)
    position = stable_hash_int(EXPERIMENT_NAME, "circular", well) % len(allowed)
    offset = int(allowed[position])
    result[finite] = np.roll(result[finite], offset)
    return result, offset


def block_permutation_control(
    observed: np.ndarray,
    blocks: Sequence[BlockSlice],
    well: str,
    block_rows: int = 256,
) -> tuple[np.ndarray, list[int]]:
    source = np.asarray(observed, dtype=np.float64)
    result = source.copy()
    full_positions = [
        position
        for position, block in enumerate(blocks)
        if block.end - block.start == block_rows
    ]
    if len(full_positions) <= 1:
        return result, full_positions
    order = sorted(
        full_positions,
        key=lambda position: stable_hash_int(
            EXPERIMENT_NAME, "block_permutation", well, str(position)
        ),
    )
    if order == full_positions:
        order = order[1:] + order[:1]
    for destination, source_position in zip(full_positions, order, strict=True):
        destination_block = blocks[destination]
        source_block = blocks[source_position]
        result[destination_block.start : destination_block.end] = source[
            source_block.start : source_block.end
        ]
    return result, order


def build_control_observations(
    observed: np.ndarray,
    blocks: Sequence[BlockSlice],
    well: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    circular, circular_offset = circular_control(observed, well, config)
    permuted, permutation = block_permutation_control(
        observed,
        blocks,
        well,
        int(get_nested(config, "observation.base_block_rows")),
    )
    return (
        {
            "real": np.asarray(observed, dtype=np.float64),
            "circular": circular,
            "block_permutation": permuted,
        },
        {
            "well": well,
            "circular_offset": circular_offset,
            "block_permutation": json.dumps(permutation, separators=(",", ":")),
            "finite_gr": int(np.isfinite(observed).sum()),
            "suffix_rows": len(observed),
        },
    )


# %% [markdown]
# ## 6. Exact interval semi-Markov posterior


# %%
@dataclass(frozen=True)
class SemiMarkovResult:
    block_posterior: np.ndarray
    log_evidence: float
    normalization_abs_error_max: float
    expected_segments: float
    transition_probability: np.ndarray


def segment_start_prior(config: Mapping[str, Any]) -> np.ndarray:
    spec = get_nested(config, "semimarkov.segment_start_prior")
    prior = np.full(
        len(EXPECTED_CANDIDATES),
        float(spec["each_other_candidate"]),
        dtype=np.float64,
    )
    prior[GEOMETRY_INDEX] = float(spec[GEOMETRY_CANDIDATE])
    prior[SAFE_INDEX] = float(spec[SAFE_CANDIDATE])
    if not math.isclose(float(prior.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"segment start prior does not sum to one: {prior.sum()}")
    if np.any(prior <= 0):
        raise ValueError("segment start prior must be positive")
    return prior


def build_switch_transition(config: Mapping[str, Any]) -> np.ndarray:
    prior = segment_start_prior(config)
    candidates = len(prior)
    geometry_floor = float(
        get_nested(
            config,
            "semimarkov.transition.geometry_floor_if_current_is_not_geometry",
        )
    )
    probability = np.zeros((candidates, candidates), dtype=np.float64)
    for current in range(candidates):
        row = prior.copy()
        row[current] = 0.0
        row /= row.sum()
        if current != GEOMETRY_INDEX and row[GEOMETRY_INDEX] < geometry_floor:
            other_mask = np.ones(candidates, dtype=bool)
            other_mask[[current, GEOMETRY_INDEX]] = False
            other_total = row[other_mask].sum()
            row[other_mask] *= (1.0 - geometry_floor) / other_total
            row[GEOMETRY_INDEX] = geometry_floor
        probability[current] = row
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-15, rtol=0.0):
        raise ValueError("switch transition rows do not sum to one")
    if np.any(np.diag(probability) != 0.0):
        raise ValueError("new segment must differ from the current candidate")
    non_geometry = np.arange(candidates) != GEOMETRY_INDEX
    if np.any(probability[non_geometry, GEOMETRY_INDEX] < geometry_floor):
        raise ValueError("geometry re-injection floor is violated")
    return probability


def allowed_segment_ends(
    start: int, block_count: int, minimum_duration: int
) -> np.ndarray:
    remaining = block_count - start
    if remaining <= 0:
        return np.empty(0, dtype=np.int64)
    if remaining < minimum_duration:
        # Only the final right-censored segment may be shorter than H512.
        return np.array([block_count], dtype=np.int64)
    return np.arange(start + minimum_duration, block_count + 1, dtype=np.int64)


def exact_interval_semimarkov(
    block_log_emission: np.ndarray, config: Mapping[str, Any]
) -> SemiMarkovResult:
    emission = np.asarray(block_log_emission, dtype=np.float64)
    if emission.ndim != 2 or emission.shape[1] != len(EXPECTED_CANDIDATES):
        raise ValueError("semi-Markov emission shape mismatch")
    if not np.isfinite(emission).all():
        raise ValueError("semi-Markov emissions must be finite")
    blocks, candidates = emission.shape
    if blocks == 0:
        raise ValueError("semi-Markov well has no H256 blocks")
    minimum_duration = int(
        get_nested(config, "semimarkov.minimum_duration_blocks")
    )
    start_log_prior = np.log(segment_start_prior(config))
    transition_probability = build_switch_transition(config)
    switch_cost = float(
        get_nested(config, "semimarkov.segment_switch_penalty.log_cost")
    )
    transition_log = np.full_like(transition_probability, -np.inf)
    positive = transition_probability > 0
    transition_log[positive] = (
        np.log(transition_probability[positive]) - switch_cost
    )
    prefix_emission = np.vstack(
        [np.zeros((1, candidates), dtype=np.float64), np.cumsum(emission, axis=0)]
    )

    forward = np.full((blocks + 1, candidates), -np.inf, dtype=np.float64)
    for start in range(blocks):
        ends = allowed_segment_ends(start, blocks, minimum_duration)
        duration_log = -math.log(len(ends))
        if start == 0:
            prefix = start_log_prior
        else:
            prefix = np.asarray(
                stable_logsumexp(
                    forward[start, :, None] + transition_log,
                    axis=0,
                ),
                dtype=np.float64,
            )
        for end in ends:
            segment_emission = prefix_emission[end] - prefix_emission[start]
            value = prefix + duration_log + segment_emission
            forward[end] = np.logaddexp(forward[end], value)
    log_evidence = float(stable_logsumexp(forward[blocks]))
    if not math.isfinite(log_evidence):
        raise ValueError("semi-Markov forward evidence is non-finite")

    # backward[start, previous_candidate] is the suffix evidence given that a
    # completed previous segment ended immediately before start.
    backward = np.full((blocks + 1, candidates), -np.inf, dtype=np.float64)
    backward[blocks] = 0.0
    for start in range(blocks - 1, 0, -1):
        ends = allowed_segment_ends(start, blocks, minimum_duration)
        duration_log = -math.log(len(ends))
        next_by_candidate = np.full(candidates, -np.inf, dtype=np.float64)
        for end in ends:
            segment_emission = prefix_emission[end] - prefix_emission[start]
            suffix = 0.0 if end == blocks else backward[end]
            if end == blocks:
                value = duration_log + segment_emission
            else:
                value = duration_log + segment_emission + suffix
            next_by_candidate = np.logaddexp(next_by_candidate, value)
        backward[start] = np.asarray(
            stable_logsumexp(
                transition_log + next_by_candidate[None, :],
                axis=1,
            ),
            dtype=np.float64,
        )

    log_block_marginal = np.full(
        (blocks, candidates), -np.inf, dtype=np.float64
    )
    expected_segments = 0.0
    for start in range(blocks):
        ends = allowed_segment_ends(start, blocks, minimum_duration)
        duration_log = -math.log(len(ends))
        if start == 0:
            prefix = start_log_prior
        else:
            prefix = np.asarray(
                stable_logsumexp(
                    forward[start, :, None] + transition_log,
                    axis=0,
                ),
                dtype=np.float64,
            )
        for end in ends:
            segment_emission = prefix_emission[end] - prefix_emission[start]
            if end == blocks:
                log_segment = (
                    prefix + duration_log + segment_emission - log_evidence
                )
            else:
                log_segment = (
                    prefix
                    + duration_log
                    + segment_emission
                    + backward[end]
                    - log_evidence
                )
            segment_probability = np.exp(log_segment)
            expected_segments += float(segment_probability.sum())
            for block in range(start, int(end)):
                log_block_marginal[block] = np.logaddexp(
                    log_block_marginal[block], log_segment
                )
    block_posterior = np.exp(log_block_marginal)
    normalization_error = np.abs(block_posterior.sum(axis=1) - 1.0)
    if not np.isfinite(block_posterior).all():
        raise ValueError("semi-Markov posterior contains non-finite values")
    return SemiMarkovResult(
        block_posterior=block_posterior,
        log_evidence=log_evidence,
        normalization_abs_error_max=float(normalization_error.max(initial=0.0)),
        expected_segments=expected_segments,
        transition_probability=transition_probability,
    )


# %% [markdown]
# ## 7. Posterior interpolation, prediction, and pre-truth freeze


# %%
@dataclass
class WellTargetFreeResult:
    segment: WellSegment
    predictions: dict[str, np.ndarray]
    geometry_mass: np.ndarray
    score: pd.DataFrame
    posterior: pd.DataFrame
    input_evidence: list[dict[str, Any]]
    role_ledger: list[dict[str, Any]]
    control_audit: dict[str, Any]
    normalization_abs_error_max: float
    row_weight_normalization_abs_error_max: float
    convex_hull_coverage: float
    interpolation_guard_passed: bool
    physical_continuity_guard_passed: bool
    expected_segments_real: float


@dataclass
class TargetFreeGeneration:
    segments: list[WellSegment]
    predictions: dict[str, np.ndarray]
    geometry_mass: np.ndarray
    score: pd.DataFrame
    posterior: pd.DataFrame
    input_evidence: pd.DataFrame
    role_ledger: pd.DataFrame
    control_audit: pd.DataFrame
    normalization_abs_error_max: float
    row_weight_normalization_abs_error_max: float
    convex_hull_coverage: float
    interpolation_guard_passed: bool
    physical_continuity_guard_passed: bool
    expected_segments_real_mean: float
    selected_global_indices: np.ndarray


@dataclass(frozen=True)
class FrozenEvidence:
    paths: tuple[Path, ...]
    file_sha256: dict[str, str]
    logical_content_sha256: dict[str, str]
    contract_path: Path
    contract_file_sha256: str
    truth_reads_before_freeze: int
    hidden_role_reads_before_freeze: int


def interpolate_block_weights(
    block_posterior: np.ndarray,
    blocks: Sequence[BlockSlice],
    row_count: int,
) -> np.ndarray:
    posterior = np.asarray(block_posterior, dtype=np.float64)
    if posterior.shape != (len(blocks), len(EXPECTED_CANDIDATES)):
        raise ValueError("block posterior/interpolation shape mismatch")
    centers = np.array(
        [(block.start + block.end - 1) / 2.0 for block in blocks],
        dtype=np.float64,
    )
    rows = np.arange(row_count, dtype=np.float64)
    weights = np.column_stack(
        [
            np.interp(
                rows,
                centers,
                posterior[:, candidate],
                left=posterior[0, candidate],
                right=posterior[-1, candidate],
            )
            for candidate in range(posterior.shape[1])
        ]
    )
    weights = np.clip(weights, 0.0, 1.0)
    total = weights.sum(axis=1, keepdims=True)
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError("interpolated posterior weights are invalid")
    return weights / total


def prediction_guards(
    candidate_values: np.ndarray,
    row_weights: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    candidates = np.asarray(candidate_values, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    values = np.asarray(prediction, dtype=np.float64)
    normalization_error = np.abs(weights.sum(axis=1) - 1.0)
    lower = candidates.min(axis=1)
    upper = candidates.max(axis=1)
    in_hull = (values >= lower - 1.0e-10) & (values <= upper + 1.0e-10)
    if len(values) <= 1:
        continuity = True
        maximum_excess = 0.0
    else:
        candidate_step = np.max(np.abs(np.diff(candidates, axis=0)), axis=1)
        weight_step = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
        candidate_range = np.ptp(candidates[:-1], axis=1)
        bound = candidate_step + 0.5 * candidate_range * weight_step
        excess = np.abs(np.diff(values)) - bound
        maximum_excess = float(excess.max(initial=0.0))
        continuity = bool(maximum_excess <= 1.0e-8)
    return {
        "finite_coverage": float(np.isfinite(values).mean()),
        "convex_hull_coverage": float(in_hull.mean()),
        "row_weight_normalization_abs_error_max": float(
            normalization_error.max(initial=0.0)
        ),
        "interpolation_guard_passed": bool(
            np.isfinite(weights).all()
            and np.all(weights >= -1.0e-12)
            and np.all(weights <= 1.0 + 1.0e-12)
            and normalization_error.max(initial=0.0) <= 1.0e-10
        ),
        "physical_continuity_guard_passed": continuity,
        "physical_continuity_maximum_bound_excess_ft": maximum_excess,
    }


def well_target_free_phase(
    segment: WellSegment,
    bank: FrozenBank,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> WellTargetFreeResult:
    horizontal_path = raw_dir / f"{segment.well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{segment.well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"raw train pair missing for well={segment.well}")
    horizontal = load_target_safe_horizontal(horizontal_path)
    typewell = load_typewell(typewell_path)
    row_idx = bank.keys["well_row_idx"].to_numpy(np.int64)[segment.start : segment.end]
    if len(row_idx) == 0 or row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise ValueError(f"suffix row index is out of range: {segment.well}")
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )[row_idx]
    if np.isfinite(visible).any():
        raise ValueError(f"exp293 rows include visible-prefix TVT: {segment.well}")
    observed = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(
        np.float64
    )[row_idx]
    typewell_tvt, typewell_gr = prepare_typewell_curve(typewell)
    groups = bank.keys["h256_group"].to_numpy(np.int64)
    blocks = build_block_slices(groups, segment)
    controls, control_audit = build_control_observations(
        observed, blocks, segment.well, config
    )
    candidates = np.asarray(
        bank.values[segment.start : segment.end], dtype=np.float64
    )
    predictions: dict[str, np.ndarray] = {}
    geometry_mass = np.full(len(observed), np.nan, dtype=np.float64)
    score_frames: list[pd.DataFrame] = []
    posterior_frames: list[pd.DataFrame] = []
    maximum_normalization_error = 0.0
    maximum_row_normalization_error = 0.0
    minimum_hull_coverage = 1.0
    interpolation_pass = True
    continuity_pass = True
    expected_segments_real = float("nan")
    for control in CONTROL_NAMES:
        block_emission = np.zeros(
            (len(blocks), len(EXPECTED_CANDIDATES)), dtype=np.float64
        )
        valid_shift_count = np.zeros_like(block_emission, dtype=np.int16)
        eligible = np.zeros_like(block_emission, dtype=bool)
        for block_position, block in enumerate(blocks):
            score = score_block_morphology(
                controls[control][block.start : block.end],
                candidates[block.start : block.end],
                typewell_tvt,
                typewell_gr,
                config,
            )
            block_emission[block_position] = score.log_emission
            valid_shift_count[block_position] = score.valid_shift_count
            eligible[block_position] = score.eligible
        posterior = exact_interval_semimarkov(block_emission, config)
        maximum_normalization_error = max(
            maximum_normalization_error,
            posterior.normalization_abs_error_max,
        )
        weights = interpolate_block_weights(
            posterior.block_posterior, blocks, len(observed)
        )
        prediction = np.sum(weights * candidates, axis=1, dtype=np.float64)
        guards = prediction_guards(candidates, weights, prediction)
        maximum_row_normalization_error = max(
            maximum_row_normalization_error,
            float(guards["row_weight_normalization_abs_error_max"]),
        )
        minimum_hull_coverage = min(
            minimum_hull_coverage, float(guards["convex_hull_coverage"])
        )
        interpolation_pass &= bool(guards["interpolation_guard_passed"])
        continuity_pass &= bool(guards["physical_continuity_guard_passed"])
        predictions[control] = prediction
        if control == "real":
            geometry_mass[:] = weights[:, GEOMETRY_INDEX]
            expected_segments_real = posterior.expected_segments
        block_index = np.repeat(np.arange(len(blocks)), len(EXPECTED_CANDIDATES))
        candidate_index = np.tile(
            np.arange(len(EXPECTED_CANDIDATES)), len(blocks)
        )
        score_frames.append(
            pd.DataFrame(
                {
                    "fold": segment.fold,
                    "well": segment.well,
                    "control": control,
                    "block_position": block_index,
                    "block_group": np.repeat(
                        [block.group for block in blocks],
                        len(EXPECTED_CANDIDATES),
                    ),
                    "row_start": np.repeat(
                        [block.start for block in blocks],
                        len(EXPECTED_CANDIDATES),
                    ),
                    "row_end": np.repeat(
                        [block.end for block in blocks],
                        len(EXPECTED_CANDIDATES),
                    ),
                    "candidate_order": candidate_index,
                    "candidate": np.asarray(EXPECTED_CANDIDATES, dtype=object)[
                        candidate_index
                    ],
                    "log_emission": block_emission.reshape(-1),
                    "valid_shift_count": valid_shift_count.reshape(-1),
                    "eligible": eligible.reshape(-1),
                }
            )
        )
        posterior_frames.append(
            pd.DataFrame(
                {
                    "fold": segment.fold,
                    "well": segment.well,
                    "control": control,
                    "block_position": block_index,
                    "block_group": np.repeat(
                        [block.group for block in blocks],
                        len(EXPECTED_CANDIDATES),
                    ),
                    "candidate_order": candidate_index,
                    "candidate": np.asarray(EXPECTED_CANDIDATES, dtype=object)[
                        candidate_index
                    ],
                    "probability": posterior.block_posterior.reshape(-1),
                    "log_evidence": posterior.log_evidence,
                    "expected_segments": posterior.expected_segments,
                    "normalization_abs_error_max": (
                        posterior.normalization_abs_error_max
                    ),
                }
            )
        )
    horizontal_safe = horizontal[["MD", "GR", "TVT_input"]].copy()
    typewell_safe = typewell[["TVT", "GR"]].copy()
    evidence = [
        target_safe_evidence(
            horizontal_path,
            "raw_horizontal_target_safe_columns",
            horizontal_safe,
            segment.well,
        ),
        target_safe_evidence(
            typewell_path,
            "raw_typewell",
            typewell_safe,
            segment.well,
        ),
    ]
    ledger = [
        {
            "phase": "pretruth",
            "well": segment.well,
            "role": "horizontal_observation",
            "path": str(horizontal_path),
            "columns_read": "MD|GR|TVT_input",
            "rows_read": len(horizontal),
            "truth_value_reads": 0,
            "hidden_role_reads": 0,
        },
        {
            "phase": "pretruth",
            "well": segment.well,
            "role": "typewell_reference_curve",
            "path": str(typewell_path),
            "columns_read": "TVT|GR",
            "rows_read": len(typewell),
            "truth_value_reads": 0,
            "hidden_role_reads": 0,
        },
    ]
    return WellTargetFreeResult(
        segment=segment,
        predictions=predictions,
        geometry_mass=geometry_mass,
        score=pd.concat(score_frames, ignore_index=True),
        posterior=pd.concat(posterior_frames, ignore_index=True),
        input_evidence=evidence,
        role_ledger=ledger,
        control_audit=control_audit,
        normalization_abs_error_max=maximum_normalization_error,
        row_weight_normalization_abs_error_max=maximum_row_normalization_error,
        convex_hull_coverage=minimum_hull_coverage,
        interpolation_guard_passed=interpolation_pass,
        physical_continuity_guard_passed=continuity_pass,
        expected_segments_real=expected_segments_real,
    )


def selected_global_indices(segments: Sequence[WellSegment]) -> np.ndarray:
    if not segments:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(
        [np.arange(segment.start, segment.end, dtype=np.int64) for segment in segments]
    )


def run_target_free_generation(
    config: Mapping[str, Any],
    bank: FrozenBank,
    raw_dir: Path,
    segments: Sequence[WellSegment],
) -> TargetFreeGeneration:
    ordered = sorted(segments, key=lambda item: (item.fold, item.well))
    workers = int(get_nested(config, "runtime.num_workers"))
    chunk_wells = int(get_nested(config, "runtime.chunk_wells"))
    results: list[WellTargetFreeResult] = []
    for start in range(0, len(ordered), chunk_wells):
        chunk = ordered[start : start + chunk_wells]
        chunk_results = Parallel(n_jobs=min(workers, len(chunk)), prefer="threads")(
            delayed(well_target_free_phase)(segment, bank, raw_dir, config)
            for segment in chunk
        )
        results.extend(chunk_results)
    results.sort(key=lambda item: (item.segment.fold, item.segment.well))
    indices = selected_global_indices([item.segment for item in results])
    row_count = len(indices)
    predictions = {
        control: np.concatenate([item.predictions[control] for item in results])
        if results
        else np.empty(0, dtype=np.float64)
        for control in CONTROL_NAMES
    }
    if any(len(values) != row_count for values in predictions.values()):
        raise ValueError("target-free prediction row count mismatch")
    return TargetFreeGeneration(
        segments=[item.segment for item in results],
        predictions=predictions,
        geometry_mass=np.concatenate([item.geometry_mass for item in results]),
        score=pd.concat([item.score for item in results], ignore_index=True),
        posterior=pd.concat([item.posterior for item in results], ignore_index=True),
        input_evidence=pd.DataFrame(
            [
                *bank.input_evidence,
                *[
                    evidence
                    for item in results
                    for evidence in item.input_evidence
                ],
            ]
        ),
        role_ledger=pd.DataFrame(
            [record for item in results for record in item.role_ledger]
        ),
        control_audit=pd.DataFrame([item.control_audit for item in results]),
        normalization_abs_error_max=max(
            (item.normalization_abs_error_max for item in results), default=0.0
        ),
        row_weight_normalization_abs_error_max=max(
            (
                item.row_weight_normalization_abs_error_max
                for item in results
            ),
            default=0.0,
        ),
        convex_hull_coverage=min(
            (item.convex_hull_coverage for item in results), default=1.0
        ),
        interpolation_guard_passed=all(
            item.interpolation_guard_passed for item in results
        ),
        physical_continuity_guard_passed=all(
            item.physical_continuity_guard_passed for item in results
        ),
        expected_segments_real_mean=float(
            np.mean([item.expected_segments_real for item in results])
        ),
        selected_global_indices=indices,
    )


def target_free_output_paths(directory: Path) -> dict[str, Path]:
    prefix = directory / OUTPUT_PREFIX
    return {
        "contract": Path(f"{prefix}_contract.json"),
        "pretruth_input": Path(f"{prefix}_pretruth_input_manifest.csv"),
        "pretruth_ledger": Path(f"{prefix}_pretruth_role_read_ledger.csv"),
        "score": Path(f"{prefix}_block_score.parquet"),
        "posterior": Path(f"{prefix}_posterior.parquet"),
        "prediction": Path(f"{prefix}_train_oof_predictions.csv.gz"),
        "control_audit": Path(f"{prefix}_negative_control_audit.csv"),
    }


def freeze_target_free_generation(
    config: Mapping[str, Any],
    bank: FrozenBank,
    generated: TargetFreeGeneration,
    directory: Path,
) -> FrozenEvidence:
    if len(generated.selected_global_indices) != len(bank.keys):
        raise ValueError("full freeze requires all exp293 rows")
    paths = target_free_output_paths(directory)
    write_csv(paths["pretruth_input"], generated.input_evidence)
    write_csv(paths["pretruth_ledger"], generated.role_ledger)
    generated.score.to_parquet(paths["score"], index=False)
    generated.posterior.to_parquet(paths["posterior"], index=False)
    keys = bank.keys.iloc[generated.selected_global_indices].reset_index(drop=True)
    prediction = keys[KEY_COLUMNS].copy()
    for control in CONTROL_NAMES:
        prediction[f"prediction_{control}"] = generated.predictions[control]
    prediction["geometry_mass_real"] = generated.geometry_mass
    write_csv_gzip(paths["prediction"], prediction)
    write_csv(paths["control_audit"], generated.control_audit)
    frozen_paths = tuple(
        paths[name]
        for name in (
            "pretruth_input",
            "pretruth_ledger",
            "score",
            "posterior",
            "prediction",
            "control_audit",
        )
    )
    file_hashes = {str(path): sha256_file(path) for path in frozen_paths}
    logical_hashes = {
        "pretruth_input": frame_content_sha256(generated.input_evidence),
        "pretruth_ledger": frame_content_sha256(generated.role_ledger),
        "score": frame_content_sha256(generated.score),
        "posterior": frame_content_sha256(generated.posterior),
        "prediction": frame_content_sha256(prediction),
        "control_audit": frame_content_sha256(generated.control_audit),
    }
    contract = {
        "experiment": EXPERIMENT_NAME,
        "phase": "target_free_scores_posterior_prediction_frozen",
        "candidate_ids": list(bank.candidate_ids),
        "candidate_content_sha256": bank.candidate_content_sha256,
        "rows": len(prediction),
        "wells": len(generated.segments),
        "folds": sorted(int(value) for value in keys["outer_fold"].unique()),
        "controls": list(CONTROL_NAMES),
        "shift_grid_ft": shift_grid(config),
        "minimum_duration_blocks": get_nested(
            config, "semimarkov.minimum_duration_blocks"
        ),
        "geometry_floor": get_nested(
            config,
            "semimarkov.transition.geometry_floor_if_current_is_not_geometry",
        ),
        "truth_reads_before_freeze": 0,
        "hidden_role_reads_before_freeze": 0,
        "frozen_file_sha256": file_hashes,
        "frozen_logical_content_sha256": logical_hashes,
        "prediction_decompressed_content_sha256": sha256_gzip_decompressed(
            paths["prediction"]
        ),
        "config_file_sha256": sha256_file(find_config_path()),
        "forbidden_outputs_absent": [
            "current_test_candidate",
            "submission",
            "viterbi_path",
            "hard_candidate_choice",
            "registration_corrected_tvt",
        ],
    }
    write_json(paths["contract"], contract)
    return FrozenEvidence(
        paths=frozen_paths,
        file_sha256=file_hashes,
        logical_content_sha256=logical_hashes,
        contract_path=paths["contract"],
        contract_file_sha256=sha256_file(paths["contract"]),
        truth_reads_before_freeze=0,
        hidden_role_reads_before_freeze=0,
    )


def verify_frozen_evidence(freeze: FrozenEvidence) -> None:
    if freeze.truth_reads_before_freeze != 0:
        raise ValueError("truth was read before target-free freeze")
    if freeze.hidden_role_reads_before_freeze != 0:
        raise ValueError("hidden-like roles were read before target-free freeze")
    for path in freeze.paths:
        if sha256_file(path) != freeze.file_sha256[str(path)]:
            raise ValueError(f"frozen artifact changed before late read: {path}")
    if sha256_file(freeze.contract_path) != freeze.contract_file_sha256:
        raise ValueError("target-free contract changed before late read")


# %% [markdown]
# ## 8. Fixed16 resource preflight


# %%
def select_fixed16_segments(
    segments: Sequence[WellSegment], config: Mapping[str, Any]
) -> list[WellSegment]:
    expected = int(
        get_nested(config, "runtime.fixed16_selector.expected_wells")
    )
    seed_key = str(get_nested(config, "runtime.fixed16_selector.seed_key"))
    by_fold: dict[int, list[WellSegment]] = {}
    for segment in segments:
        by_fold.setdefault(segment.fold, []).append(segment)
    for fold, values in by_fold.items():
        values.sort(
            key=lambda item: (
                stable_hash_int(seed_key, str(fold), item.well),
                item.well,
            )
        )
    selected: list[WellSegment] = []
    positions = {fold: 0 for fold in sorted(by_fold)}
    folds = sorted(by_fold)
    while len(selected) < expected:
        progress = False
        for fold in folds:
            position = positions[fold]
            if position < len(by_fold[fold]):
                selected.append(by_fold[fold][position])
                positions[fold] += 1
                progress = True
                if len(selected) == expected:
                    break
        if not progress:
            break
    if len(selected) != expected:
        raise ValueError(f"fixed16 selector returned {len(selected)} wells")
    if set(segment.fold for segment in selected) != set(range(5)):
        raise ValueError("fixed16 selector does not cover all folds")
    return selected


def estimate_full_peak_rss_gb(
    measured_peak_gb: float,
    expected_rows: int,
    expected_wells: int,
    candidate_count: int,
) -> float:
    # Full generation retains three float64 predictions and one geometry-mass
    # vector. Block score/posterior frames are conservatively budgeted at
    # 3 controls x 12 candidates x ceil(rows/256) x 320 bytes.
    prediction_bytes = expected_rows * 4 * 8
    block_rows = expected_rows // 256 + expected_wells
    frame_bytes = block_rows * 3 * candidate_count * 320
    safety_bytes = expected_rows * 64
    return measured_peak_gb + (
        prediction_bytes + frame_bytes + safety_bytes
    ) / (1024.0**3)


def run_fixed16_preflight(
    config: Mapping[str, Any],
    bank: FrozenBank,
    raw_dir: Path,
) -> dict[str, Any]:
    selected = select_fixed16_segments(bank.segments, config)
    started = time.perf_counter()
    generated = run_target_free_generation(
        config, bank, raw_dir, selected
    )
    elapsed = time.perf_counter() - started
    selected_rows = len(generated.selected_global_indices)
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    projection_multiplier = max(
        expected_rows / selected_rows,
        expected_wells / len(selected),
    )
    projected_seconds = elapsed * projection_multiplier
    measured_peak = peak_rss_gb()
    projected_peak = estimate_full_peak_rss_gb(
        measured_peak,
        expected_rows,
        expected_wells,
        len(EXPECTED_CANDIDATES),
    )
    normalization_limit = float(
        get_nested(
            config,
            "semimarkov.numerical.posterior_normalization_abs_error_max",
        )
    )
    gates = {
        "fixed16_well_count_exact": len(selected)
        == int(get_nested(config, "gates.technical.fixed16_resource_preflight_wells")),
        "all_folds_present": set(segment.fold for segment in selected)
        == set(range(5)),
        "candidate_bank_sha_exact": bank.candidate_content_sha256
        == str(get_nested(config, "data.exp293.candidate_content_sha256")),
        "posterior_normalization": generated.normalization_abs_error_max
        <= normalization_limit,
        "row_weight_normalization": (
            generated.row_weight_normalization_abs_error_max
            <= normalization_limit
        ),
        "finite_predictions": all(
            np.isfinite(values).all() for values in generated.predictions.values()
        ),
        "convex_hull": generated.convex_hull_coverage == 1.0,
        "block_center_interpolation": generated.interpolation_guard_passed,
        "physical_continuity": generated.physical_continuity_guard_passed,
        "truth_reads_zero": True,
        "hidden_role_reads_zero": True,
        "projected_runtime": projected_seconds
        <= float(get_nested(config, "gates.technical.projected_runtime_seconds_max")),
        "projected_peak_rss": projected_peak
        <= float(get_nested(config, "gates.technical.projected_peak_rss_gb_max")),
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "fixed16_preflight",
        "status": "technical_preflight_pass" if all(gates.values()) else "technical_error",
        "selected_wells": [
            {"well": segment.well, "fold": segment.fold} for segment in selected
        ],
        "rows": selected_rows,
        "controls": list(CONTROL_NAMES),
        "candidate_count": len(EXPECTED_CANDIDATES),
        "elapsed_seconds": elapsed,
        "projection_multiplier": projection_multiplier,
        "projected_full_seconds": projected_seconds,
        "measured_peak_rss_gb": measured_peak,
        "projected_full_peak_rss_gb": projected_peak,
        "posterior_normalization_abs_error_max": (
            generated.normalization_abs_error_max
        ),
        "row_weight_normalization_abs_error_max": (
            generated.row_weight_normalization_abs_error_max
        ),
        "candidate_content_sha256": bank.candidate_content_sha256,
        "input_manifest_logical_sha256": frame_content_sha256(
            generated.input_evidence
        ),
        "score_logical_sha256": frame_content_sha256(generated.score),
        "posterior_logical_sha256": frame_content_sha256(generated.posterior),
        "prediction_logical_sha256": stable_json_sha256(
            {
                control: hashlib.sha256(
                    np.asarray(generated.predictions[control], dtype="<f8").tobytes()
                ).hexdigest()
                for control in CONTROL_NAMES
            }
        ),
        "gates": gates,
        "all_technical_gates_passed": all(gates.values()),
        "truth_reads_before_freeze": 0,
        "hidden_role_reads_before_freeze": 0,
    }
    directory = artifact_dir()
    path = directory / f"{OUTPUT_PREFIX}_preflight_summary.json"
    write_json(path, summary)
    summary["summary_file_sha256"] = sha256_file(path)
    write_json(runtime_metrics_path(), summary)
    return summary


# %% [markdown]
# ## 9. Post-freeze truth, constrained oracle, metrics, and gates


# %%
def load_truth_after_freeze(
    bank: FrozenBank,
    generated: TargetFreeGeneration,
    raw_dir: Path,
    freeze: FrozenEvidence,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    verify_frozen_evidence(freeze)
    truth_parts: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for segment in generated.segments:
        path = raw_dir / f"{segment.well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        row_idx = bank.keys["well_row_idx"].to_numpy(np.int64)[
            segment.start : segment.end
        ]
        visible = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(
            np.float64
        )[row_idx]
        truth = pd.to_numeric(frame["TVT"], errors="raise").to_numpy(np.float64)[
            row_idx
        ]
        if np.isfinite(visible).any() or not np.isfinite(truth).all():
            raise ValueError(f"late truth contract failed for well={segment.well}")
        truth_parts.append(truth)
        selected = pd.DataFrame(
            {
                "id": bank.keys["id"].astype(str).to_numpy()[
                    segment.start : segment.end
                ],
                "true_tvt": truth,
            }
        )
        evidence.append(
            {
                "phase": "post_freeze",
                "role": "raw_horizontal_suffix_truth",
                "path": str(path),
                "well": segment.well,
                "rows": len(truth),
                "wells": 1,
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": None,
                "logical_content_sha256": frame_content_sha256(selected),
                "schema_sha256": frame_schema_sha256(selected),
            }
        )
        ledger.append(
            {
                "phase": "post_freeze",
                "well": segment.well,
                "role": "suffix_truth_readout",
                "path": str(path),
                "columns_read": "TVT|TVT_input",
                "rows_read": len(frame),
                "truth_value_reads": len(truth),
                "hidden_role_reads": 0,
            }
        )
    combined = np.concatenate(truth_parts)
    if len(combined) != len(generated.selected_global_indices):
        raise ValueError("late truth row count mismatch")
    return combined, evidence, ledger


def load_hidden_like_sets_after_freeze(
    config: Mapping[str, Any],
    expected_wells: set[str],
    freeze: FrozenEvidence,
) -> tuple[
    dict[str, set[str]],
    dict[str, Any],
    dict[str, Any],
]:
    verify_frozen_evidence(freeze)
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
        if column not in frame:
            raise ValueError(f"hidden-like role column missing: {column}")
        selected = set(frame.loc[frame[column].eq("valid"), well_column].astype(str))
        if unknown := selected - expected_wells:
            raise ValueError(f"hidden-like roles contain unknown wells: {sorted(unknown)[:5]}")
        sets[str(name)] = selected
    evidence = {
        "phase": "post_freeze",
        "role": "hidden_like_assignment",
        "path": str(path),
        "well": None,
        "rows": len(frame),
        "wells": frame[well_column].nunique(),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }
    ledger = {
        "phase": "post_freeze",
        "well": None,
        "role": "hidden_like_role_readout",
        "path": str(path),
        "columns_read": "|".join(
            [well_column, *[str(value) for value in spec["role_columns"].values()]]
        ),
        "rows_read": len(frame),
        "truth_value_reads": 0,
        "hidden_role_reads": len(frame),
    }
    return sets, evidence, ledger


def constrained_oracle_labels(
    candidate_values: np.ndarray,
    truth: np.ndarray,
    blocks: Sequence[BlockSlice],
    minimum_duration: int,
) -> np.ndarray:
    candidates = np.asarray(candidate_values, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if candidates.shape != (len(target), len(EXPECTED_CANDIDATES)):
        raise ValueError("constrained oracle input shape mismatch")
    block_loss = np.array(
        [
            np.square(candidates[block.start : block.end] - target[block.start : block.end, None]).sum(
                axis=0
            )
            for block in blocks
        ],
        dtype=np.float64,
    )
    block_count, candidate_count = block_loss.shape
    prefix_loss = np.vstack(
        [
            np.zeros((1, candidate_count), dtype=np.float64),
            np.cumsum(block_loss, axis=0),
        ]
    )
    cost = np.full((block_count + 1, candidate_count), np.inf, dtype=np.float64)
    back_start = np.full((block_count + 1, candidate_count), -1, dtype=np.int32)
    back_candidate = np.full(
        (block_count + 1, candidate_count), -1, dtype=np.int16
    )
    for start in range(block_count):
        if start == 0:
            prefix = np.zeros(candidate_count, dtype=np.float64)
            previous = np.full(candidate_count, -1, dtype=np.int16)
        else:
            prefix = np.full(candidate_count, np.inf, dtype=np.float64)
            previous = np.full(candidate_count, -1, dtype=np.int16)
            for candidate in range(candidate_count):
                allowed_previous = np.arange(candidate_count) != candidate
                masked = np.where(allowed_previous, cost[start], np.inf)
                previous[candidate] = int(np.argmin(masked))
                prefix[candidate] = masked[previous[candidate]]
        for end in allowed_segment_ends(start, block_count, minimum_duration):
            value = prefix + prefix_loss[end] - prefix_loss[start]
            improved = value < cost[end]
            cost[end, improved] = value[improved]
            back_start[end, improved] = start
            back_candidate[end, improved] = previous[improved]
    candidate = int(np.argmin(cost[block_count]))
    if not math.isfinite(float(cost[block_count, candidate])):
        raise ValueError("constrained oracle failed to tile the well")
    labels = np.full(block_count, -1, dtype=np.int16)
    end = block_count
    while end > 0:
        start = int(back_start[end, candidate])
        if start < 0:
            raise ValueError("constrained oracle backtrace is incomplete")
        labels[start:end] = candidate
        candidate = int(back_candidate[end, candidate])
        end = start
    if np.any(labels < 0):
        raise ValueError("constrained oracle left uncovered blocks")
    return labels


def constrained_oracle_prediction(
    bank: FrozenBank,
    generated: TargetFreeGeneration,
    truth: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    output: list[np.ndarray] = []
    position = 0
    groups = bank.keys["h256_group"].to_numpy(np.int64)
    minimum_duration = int(
        get_nested(config, "semimarkov.minimum_duration_blocks")
    )
    for segment in generated.segments:
        rows = segment.end - segment.start
        local_truth = truth[position : position + rows]
        candidates = np.asarray(
            bank.values[segment.start : segment.end], dtype=np.float64
        )
        blocks = build_block_slices(groups, segment)
        labels = constrained_oracle_labels(
            candidates, local_truth, blocks, minimum_duration
        )
        local = np.empty(rows, dtype=np.float64)
        for block_position, block in enumerate(blocks):
            candidate = int(labels[block_position])
            local[block.start : block.end] = candidates[
                block.start : block.end, candidate
            ]
        output.append(local)
        position += rows
    if position != len(truth):
        raise ValueError("constrained oracle truth consumption mismatch")
    return np.concatenate(output)


def rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    target = np.asarray(truth, dtype=np.float64)
    values = np.asarray(prediction, dtype=np.float64)
    selected = np.ones(len(target), dtype=bool) if mask is None else np.asarray(mask, bool)
    if not selected.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[selected] - target[selected]))))


def metric_record(
    scope: str,
    scope_type: str,
    mask: np.ndarray,
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    wells: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor_rmse = rmse(truth, predictions["anchor"], mask)
    for name, values in predictions.items():
        value = rmse(truth, values, mask)
        rows.append(
            {
                "scope": scope,
                "scope_type": scope_type,
                "prediction": name,
                "rows": int(mask.sum()),
                "wells": int(np.unique(wells[mask]).size),
                "rmse_ft": value,
                "delta_vs_anchor_ft": value - anchor_rmse,
            }
        )
    return rows


def build_readout_frames(
    bank: FrozenBank,
    generated: TargetFreeGeneration,
    truth: np.ndarray,
    oracle_prediction: np.ndarray,
    hidden_sets: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = generated.selected_global_indices
    keys = bank.keys.iloc[indices].reset_index(drop=True)
    wells = keys["well"].astype(str).to_numpy()
    folds = keys["outer_fold"].to_numpy(np.int8)
    distance = keys["md_since"].to_numpy(np.float64)
    anchor = np.asarray(bank.values[indices, SAFE_INDEX], dtype=np.float64)
    predictions = {
        "posterior_mean_real": generated.predictions["real"],
        "anchor": anchor,
        "constrained_oracle": oracle_prediction,
        "negative_circular": generated.predictions["circular"],
        "negative_block_permutation": generated.predictions["block_permutation"],
    }
    pooled = np.ones(len(truth), dtype=bool)
    fold_records: list[dict[str, Any]] = []
    fold_records.extend(
        metric_record("pooled", "pooled", pooled, truth, predictions, wells)
    )
    for fold in range(5):
        mask = folds == fold
        fold_records.extend(
            metric_record(f"fold_{fold}", "fold", mask, truth, predictions, wells)
        )
    scope_masks = {
        "near_0_250": distance <= 250.0,
        "mid_250_1000": (distance > 250.0) & (distance < 1000.0),
        "1000_plus": distance >= 1000.0,
        **{
            name: np.isin(wells, sorted(selected))
            for name, selected in hidden_sets.items()
        },
    }
    scope_records: list[dict[str, Any]] = []
    for name, mask in scope_masks.items():
        scope_records.extend(
            metric_record(name, "scope", mask, truth, predictions, wells)
        )
    by_well: list[dict[str, Any]] = []
    for well in np.unique(wells):
        mask = wells == well
        real_rmse = rmse(truth, predictions["posterior_mean_real"], mask)
        anchor_rmse = rmse(truth, anchor, mask)
        by_well.append(
            {
                "well": well,
                "fold": int(folds[np.flatnonzero(mask)[0]]),
                "rows": int(mask.sum()),
                "posterior_mean_rmse_ft": real_rmse,
                "anchor_rmse_ft": anchor_rmse,
                "delta_vs_anchor_ft": real_rmse - anchor_rmse,
                "geometry_mass_mean": float(generated.geometry_mass[mask].mean()),
            }
        )
    negative: list[dict[str, Any]] = []
    real_pooled = rmse(truth, predictions["posterior_mean_real"])
    for control, prediction_name in (
        ("circular", "negative_circular"),
        ("block_permutation", "negative_block_permutation"),
    ):
        control_pooled = rmse(truth, predictions[prediction_name])
        negative.append(
            {
                "control": control,
                "scope": "pooled",
                "fold": -1,
                "real_rmse_ft": real_pooled,
                "control_rmse_ft": control_pooled,
                "real_gain_ft": control_pooled - real_pooled,
                "real_better": real_pooled < control_pooled,
            }
        )
        for fold in range(5):
            mask = folds == fold
            real_fold = rmse(truth, predictions["posterior_mean_real"], mask)
            control_fold = rmse(truth, predictions[prediction_name], mask)
            negative.append(
                {
                    "control": control,
                    "scope": f"fold_{fold}",
                    "fold": fold,
                    "real_rmse_ft": real_fold,
                    "control_rmse_ft": control_fold,
                    "real_gain_ft": control_fold - real_fold,
                    "real_better": real_fold < control_fold,
                }
            )
    return (
        pd.DataFrame(fold_records),
        pd.DataFrame(scope_records),
        pd.DataFrame(by_well).sort_values("well", kind="mergesort"),
        pd.DataFrame(negative),
    )


def metric_lookup(
    frame: pd.DataFrame, scope: str, prediction: str, column: str = "rmse_ft"
) -> float:
    selected = frame[
        frame["scope"].eq(scope) & frame["prediction"].eq(prediction)
    ]
    if len(selected) != 1:
        raise ValueError(f"metric inventory mismatch: {scope}/{prediction}")
    return float(selected.iloc[0][column])


def evaluate_gate(
    config: Mapping[str, Any],
    bank: FrozenBank,
    generated: TargetFreeGeneration,
    freeze: FrozenEvidence,
    truth: np.ndarray,
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    negative_metrics: pd.DataFrame,
    elapsed_seconds: float,
) -> dict[str, Any]:
    real_pooled = metric_lookup(
        fold_metrics, "pooled", "posterior_mean_real"
    )
    anchor_pooled = metric_lookup(fold_metrics, "pooled", "anchor")
    oracle_pooled = metric_lookup(
        fold_metrics, "pooled", "constrained_oracle"
    )
    expected_anchor = float(
        get_nested(config, "validation.primary_control.rmse_ft")
    )
    anchor_tolerance = float(
        get_nested(config, "gates.technical.primary_control_rmse_abs_tolerance_ft")
    )
    normalization_limit = float(
        get_nested(
            config,
            "semimarkov.numerical.posterior_normalization_abs_error_max",
        )
    )
    technical = {
        "row_count_exact": len(truth)
        == int(get_nested(config, "validation.expected_rows")),
        "well_count_exact": len(generated.segments)
        == int(get_nested(config, "validation.expected_wells")),
        "fold_inventory_exact": set(
            int(value)
            for value in bank.keys.iloc[generated.selected_global_indices][
                "outer_fold"
            ].unique()
        )
        == set(range(5)),
        "candidate_count_exact": len(bank.candidate_ids)
        == int(get_nested(config, "gates.technical.candidate_count")),
        "candidate_content_sha_exact": bank.candidate_content_sha256
        == str(get_nested(config, "data.exp293.candidate_content_sha256")),
        "finite_candidate_coverage": bool(
            np.isfinite(bank.values).all()
        ),
        "finite_prediction_coverage": all(
            np.isfinite(values).all() for values in generated.predictions.values()
        ),
        "convex_hull_coverage": generated.convex_hull_coverage
        == float(get_nested(config, "gates.technical.convex_hull_coverage")),
        "truth_reads_before_freeze_zero": freeze.truth_reads_before_freeze
        <= int(get_nested(config, "gates.technical.truth_reads_before_freeze_max")),
        "hidden_role_reads_before_freeze_zero": (
            freeze.hidden_role_reads_before_freeze
            <= int(
                get_nested(
                    config, "gates.technical.hidden_role_reads_before_freeze_max"
                )
            )
        ),
        "posterior_normalization": generated.normalization_abs_error_max
        <= normalization_limit,
        "row_weight_normalization": (
            generated.row_weight_normalization_abs_error_max
            <= normalization_limit
        ),
        "block_center_interpolation": generated.interpolation_guard_passed,
        "physical_continuity": generated.physical_continuity_guard_passed,
        "anchor_rmse_parity": abs(anchor_pooled - expected_anchor)
        <= anchor_tolerance,
        "runtime": elapsed_seconds
        <= float(get_nested(config, "runtime.maximum_seconds")),
        "peak_rss": peak_rss_gb()
        <= float(get_nested(config, "runtime.maximum_peak_rss_gb")),
    }
    fold_oracle = [
        metric_lookup(fold_metrics, f"fold_{fold}", "constrained_oracle")
        < metric_lookup(fold_metrics, f"fold_{fold}", "anchor")
        for fold in range(5)
    ]
    constrained_oracle = {
        "pooled_rmse": oracle_pooled
        <= float(get_nested(config, "gates.constrained_oracle.maximum_rmse_ft")),
        "improves_anchor_all_folds": sum(fold_oracle)
        >= int(
            get_nested(
                config,
                "gates.constrained_oracle.require_improvement_vs_exp263_folds",
            )
        ),
    }
    fold_real = [
        metric_lookup(fold_metrics, f"fold_{fold}", "posterior_mean_real")
        < metric_lookup(fold_metrics, f"fold_{fold}", "anchor")
        for fold in range(5)
    ]
    delta_by_well = by_well["delta_vs_anchor_ft"].to_numpy(np.float64)
    pooled_controls = negative_metrics[negative_metrics["fold"].eq(-1)]
    control_gain = {
        str(row.control): float(row.real_gain_ft)
        for row in pooled_controls.itertuples()
    }
    control_fold_better = {
        control: int(
            negative_metrics[
                negative_metrics["control"].eq(control)
                & negative_metrics["fold"].ge(0)
            ]["real_better"].sum()
        )
        for control in ("circular", "block_permutation")
    }
    geometry_well_mean = by_well["geometry_mass_mean"].to_numpy(np.float64)
    scientific = {
        "pooled_rmse": real_pooled
        <= float(get_nested(config, "gates.scientific.maximum_pooled_rmse_ft")),
        "improves_anchor_all_folds": sum(fold_real)
        >= int(
            get_nested(
                config,
                "gates.scientific.require_improvement_vs_exp263_folds",
            )
        ),
        "1000_plus_improves": metric_lookup(
            scope_metrics, "1000_plus", "posterior_mean_real"
        )
        < metric_lookup(scope_metrics, "1000_plus", "anchor"),
        "hidden_like_spatial_improves": metric_lookup(
            scope_metrics, "hidden_like_spatial", "posterior_mean_real"
        )
        < metric_lookup(scope_metrics, "hidden_like_spatial", "anchor"),
        "hidden_like_typewell_purged_improves": metric_lookup(
            scope_metrics,
            "hidden_like_typewell_purged",
            "posterior_mean_real",
        )
        < metric_lookup(
            scope_metrics, "hidden_like_typewell_purged", "anchor"
        ),
        "by_well_delta_p95": float(np.quantile(delta_by_well, 0.95))
        <= float(
            get_nested(
                config, "gates.scientific.maximum_by_well_delta_p95_ft"
            )
        ),
        "worst_well_regression": float(delta_by_well.max(initial=-np.inf))
        <= float(
            get_nested(
                config, "gates.scientific.maximum_worst_well_regression_ft"
            )
        ),
        "real_gain_vs_circular": control_gain["circular"]
        >= float(
            get_nested(
                config,
                "gates.scientific.real_gain_vs_each_negative_control_ft_min",
            )
        ),
        "real_gain_vs_block_permutation": control_gain["block_permutation"]
        >= float(
            get_nested(
                config,
                "gates.scientific.real_gain_vs_each_negative_control_ft_min",
            )
        ),
        "real_better_circular_all_folds": control_fold_better["circular"]
        >= int(
            get_nested(
                config,
                "gates.scientific.require_real_better_than_each_control_folds",
            )
        ),
        "real_better_block_permutation_all_folds": control_fold_better[
            "block_permutation"
        ]
        >= int(
            get_nested(
                config,
                "gates.scientific.require_real_better_than_each_control_folds",
            )
        ),
        "geometry_posterior_pooled_mean": float(generated.geometry_mass.mean())
        >= float(
            get_nested(
                config,
                "gates.scientific.geometry_posterior_pooled_mean_min",
            )
        ),
        "geometry_posterior_well_median": float(
            np.median(geometry_well_mean)
        )
        >= float(
            get_nested(
                config,
                "gates.scientific.geometry_posterior_per_well_mean_median_min",
            )
        ),
        "geometry_posterior_low_mass_well_fraction": float(
            np.mean(geometry_well_mean < 0.005)
        )
        <= float(
            get_nested(
                config,
                "gates.scientific.geometry_posterior_well_fraction_below_0p005_max",
            )
        ),
        "physical_continuity_guard": generated.physical_continuity_guard_passed,
    }
    technical_pass = all(technical.values())
    oracle_pass = all(constrained_oracle.values())
    scientific_pass = all(scientific.values())
    all_pass = technical_pass and oracle_pass and scientific_pass
    if all_pass:
        decision = "current_test_implementation_eligible_pending_explicit_approval"
    elif technical_pass:
        decision = "scientific_fail_close_exp405_unlock_exp406_stage0"
    else:
        decision = "technical_error_fix_only_and_rerun_same_contract"
    payload = {
        "experiment": EXPERIMENT_NAME,
        "status": decision,
        "technical": technical,
        "constrained_oracle": constrained_oracle,
        "scientific": scientific,
        "technical_pass": technical_pass,
        "constrained_oracle_pass": oracle_pass,
        "scientific_pass": scientific_pass,
        "all_gates_pass": all_pass,
        "current_test_implementation_eligible": all_pass,
        "exp406_stage0_unlocked": technical_pass and not (oracle_pass and scientific_pass),
        "metrics": {
            "posterior_mean_rmse_ft": real_pooled,
            "anchor_rmse_ft": anchor_pooled,
            "constrained_oracle_rmse_ft": oracle_pooled,
            "delta_vs_anchor_ft": real_pooled - anchor_pooled,
            "by_well_delta_p95_ft": float(np.quantile(delta_by_well, 0.95)),
            "worst_well_regression_ft": float(delta_by_well.max()),
            "geometry_mass_pooled_mean": float(generated.geometry_mass.mean()),
            "geometry_mass_well_median": float(np.median(geometry_well_mean)),
            "geometry_low_mass_well_fraction": float(
                np.mean(geometry_well_mean < 0.005)
            ),
            "negative_control_gain_ft": control_gain,
        },
    }
    payload["decision_sha256"] = stable_json_sha256(payload)
    return payload


def persist_full_readout(
    config: Mapping[str, Any],
    bank: FrozenBank,
    generated: TargetFreeGeneration,
    freeze: FrozenEvidence,
    truth_evidence: Sequence[Mapping[str, Any]],
    truth_ledger: Sequence[Mapping[str, Any]],
    hidden_evidence: Mapping[str, Any],
    hidden_ledger: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    negative_metrics: pd.DataFrame,
    gate: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    directory = artifact_dir()
    prefix = directory / OUTPUT_PREFIX
    paths = {
        "input": Path(f"{prefix}_input_manifest.csv"),
        "ledger": Path(f"{prefix}_role_read_ledger.csv"),
        "fold": Path(f"{prefix}_fold_metrics.csv"),
        "scope": Path(f"{prefix}_scope_metrics.csv"),
        "by_well": Path(f"{prefix}_by_well.csv"),
        "geometry": Path(f"{prefix}_geometry_mass.csv"),
        "negative": Path(f"{prefix}_negative_control_metrics.csv"),
        "gate": Path(f"{prefix}_gate.json"),
        "sha": Path(f"{prefix}_sha_manifest.csv"),
        "summary": Path(f"{prefix}_summary.json"),
    }
    input_manifest = pd.concat(
        [
            generated.input_evidence,
            pd.DataFrame([*truth_evidence, dict(hidden_evidence)]),
        ],
        ignore_index=True,
    )
    role_ledger = pd.concat(
        [
            generated.role_ledger,
            pd.DataFrame([*truth_ledger, dict(hidden_ledger)]),
        ],
        ignore_index=True,
    )
    geometry = by_well[
        ["well", "fold", "rows", "geometry_mass_mean"]
    ].copy()
    write_csv(paths["input"], input_manifest)
    write_csv(paths["ledger"], role_ledger)
    write_csv(paths["fold"], fold_metrics)
    write_csv(paths["scope"], scope_metrics)
    write_csv(paths["by_well"], by_well)
    write_csv(paths["geometry"], geometry)
    write_csv(paths["negative"], negative_metrics)
    write_json(paths["gate"], gate)
    target_free_paths = target_free_output_paths(directory)
    evidence_paths = [
        freeze.contract_path,
        *freeze.paths,
        paths["input"],
        paths["ledger"],
        paths["fold"],
        paths["scope"],
        paths["by_well"],
        paths["geometry"],
        paths["negative"],
        paths["gate"],
    ]
    sha_rows = []
    for path in evidence_paths:
        sha_rows.append(
            {
                "path": str(path),
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": (
                    sha256_gzip_decompressed(path)
                    if path.suffix == ".gz"
                    else None
                ),
            }
        )
    sha_manifest = pd.DataFrame(sha_rows)
    write_csv(paths["sha"], sha_manifest)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": gate["status"],
        "route": "pf_beam",
        "rows": len(generated.selected_global_indices),
        "wells": len(generated.segments),
        "folds": 5,
        "candidate_count": len(EXPECTED_CANDIDATES),
        "controls": list(CONTROL_NAMES),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gb": peak_rss_gb(),
        "candidate_content_sha256": bank.candidate_content_sha256,
        "contract_file_sha256": freeze.contract_file_sha256,
        "score_logical_content_sha256": freeze.logical_content_sha256["score"],
        "posterior_logical_content_sha256": freeze.logical_content_sha256[
            "posterior"
        ],
        "prediction_logical_content_sha256": freeze.logical_content_sha256[
            "prediction"
        ],
        "prediction_decompressed_content_sha256": sha256_gzip_decompressed(
            target_free_paths["prediction"]
        ),
        "input_manifest_logical_content_sha256": frame_content_sha256(
            input_manifest
        ),
        "role_ledger_logical_content_sha256": frame_content_sha256(role_ledger),
        "fold_metrics_logical_content_sha256": frame_content_sha256(
            fold_metrics
        ),
        "scope_metrics_logical_content_sha256": frame_content_sha256(
            scope_metrics
        ),
        "by_well_logical_content_sha256": frame_content_sha256(by_well),
        "negative_metrics_logical_content_sha256": frame_content_sha256(
            negative_metrics
        ),
        "gate_file_sha256": sha256_file(paths["gate"]),
        "sha_manifest_file_sha256": sha256_file(paths["sha"]),
        "decision_sha256": gate["decision_sha256"],
        "gate": gate,
        "execution_count": {
            "scientific_endpoints": 1,
            "negative_controls": 2,
            "models": 0,
            "boosters": 0,
            "pf_runs": 0,
            "hmm_runs": 0,
            "beam_runs": 0,
            "parent_reruns": 0,
        },
        "current_test_implemented": False,
        "inference_implemented": False,
        "submission_created": False,
        "deterministic_anchor": False,
    }
    write_json(paths["summary"], summary)
    summary["summary_file_sha256"] = sha256_file(paths["summary"])
    write_json(runtime_metrics_path(), summary)
    return summary


def run_full_saved_oof(
    config: Mapping[str, Any],
    bank: FrozenBank,
    raw_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    generated = run_target_free_generation(
        config, bank, raw_dir, bank.segments
    )
    directory = artifact_dir()
    freeze = freeze_target_free_generation(
        config, bank, generated, directory
    )
    print("Target-free prediction frozen:", freeze.logical_content_sha256["prediction"])
    truth, truth_evidence, truth_ledger = load_truth_after_freeze(
        bank, generated, raw_dir, freeze
    )
    hidden_sets, hidden_evidence, hidden_ledger = (
        load_hidden_like_sets_after_freeze(
            config,
            set(bank.keys["well"].astype(str)),
            freeze,
        )
    )
    oracle = constrained_oracle_prediction(
        bank, generated, truth, config
    )
    fold_metrics, scope_metrics, by_well, negative_metrics = (
        build_readout_frames(
            bank,
            generated,
            truth,
            oracle,
            hidden_sets,
        )
    )
    elapsed = time.perf_counter() - started
    gate = evaluate_gate(
        config,
        bank,
        generated,
        freeze,
        truth,
        fold_metrics,
        scope_metrics,
        by_well,
        negative_metrics,
        elapsed,
    )
    return persist_full_readout(
        config,
        bank,
        generated,
        freeze,
        truth_evidence,
        truth_ledger,
        hidden_evidence,
        hidden_ledger,
        fold_metrics,
        scope_metrics,
        by_well,
        negative_metrics,
        gate,
        elapsed,
    )


# %% [markdown]
# ## 10. Setup and execution


# %%
def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(get_nested(config, "execution.run_stage"))
    counts = validate_scientific_contract(config, stage)
    print("Experiment:", get_nested(config, "experiment.name"))
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Run stage:", stage)
    print("Execution counts:", counts)
    print("Config SHA256:", sha256_file(find_config_path()))
    if stage == "implementation_only":
        return {
            "status": "implementation_complete_not_run",
            "execution_counts": counts,
            "kaggle_run": False,
            "current_test": False,
            "inference": False,
            "submission": False,
        }
    bank = load_frozen_exp293_bank(config)
    raw_dir = resolve_train_dir(get_nested(config, "data.raw_train_dir_patterns"))
    print("Frozen candidate content SHA256:", bank.candidate_content_sha256)
    print("Raw train directory:", raw_dir)
    if stage == "fixed16_preflight":
        return run_fixed16_preflight(config, bank, raw_dir)
    if stage == "full_saved_oof":
        return run_full_saved_oof(config, bank, raw_dir)
    raise AssertionError(f"unreachable stage: {stage}")


# %%
CONFIG_PATH = find_config_path()
CONFIG = read_yaml(CONFIG_PATH)
CONTRACT_COUNTS = validate_scientific_contract(CONFIG)
print("Implementation candidate ready:", EXPERIMENT_NAME)
print("Canonical notebook adopted:", get_nested(CONFIG, "implementation.canonical_notebook_adopted"))
print("Kaggle execution authorized:", get_nested(CONFIG, "execution.kaggle_execution_authorized"))
print("Planned execution:", CONTRACT_COUNTS)


# %%
if EXECUTE_NOTEBOOK:
    RUN_RESULT = run_experiment(CONFIG)
    print(json.dumps(to_jsonable(RUN_RESULT), indent=2, sort_keys=True))
