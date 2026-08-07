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
# # exp293 physics-only candidate-bank headroom contract
#
# This train-side audit freezes the twelve exp263 Stage-1 deployable physical
# paths, their row identity, and non-overlapping block assignments before truth
# is opened. It then measures row, H128/H256/H512, and whole-well oracle
# headroom without fitting a selector, regenerating a path, or persisting an
# oracle TVT prediction.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Exp263 candidate-bank reconstruction and parity checks
# 4. Block assignment and target-free freeze boundary
# 5. Post-freeze raw-train truth loader
# 6. Chunked oracle aggregation and diagnostic readouts
# 7. Support decision and generated artifacts
# 8. Setup, contract preview, and execution

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp293_physics_only_candidate_bank_headroom_contract"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
VALUE_KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
VALUE_READ_COLUMNS = VALUE_KEY_COLUMNS + [
    "last_known_tvt",
    "candidate_tvt",
    "candidate_available",
    "candidate_finite",
]
EXPECTED_CANDIDATE_ORDER = (
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
FORMULA_CANDIDATES = EXPECTED_CANDIDATE_ORDER[6:]
FORBIDDEN_CANDIDATE_COLUMNS = {
    "TVT",
    "target",
    "true_tvt",
    "error",
    "abs_error",
    "oracle",
    "oracle_label",
    "oracle_candidate",
}
EXPECTED_DOWNSTREAM_CONTRACT_SHA256 = (
    "025a81e634b9a46504314bfd9e273bc2e36ed18dd5fa744e7fd3a8b614713819"
)


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
    os.environ.get("EXP293_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    root = project_root()
    nested = root / "experiments" / EXPERIMENT_NAME
    if nested.exists():
        return nested
    return Path.cwd()


def find_config_path() -> Path:
    direct = Path.cwd() / "config.yaml"
    if direct.exists():
        return direct
    nested = experiment_dir() / "config.yaml"
    if nested.exists():
        return nested
    matches = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp293 config.yaml was not found unambiguously")


def downstream_contract_sha256() -> str:
    for path in (
        Path.cwd() / "downstream_branch_contract.md",
        experiment_dir() / "downstream_branch_contract.md",
    ):
        if not path.exists():
            continue
        actual = sha256_file(path)
        if actual != EXPECTED_DOWNSTREAM_CONTRACT_SHA256:
            raise ValueError(
                "downstream_branch_contract.md SHA mismatch: "
                f"expected {EXPECTED_DOWNSTREAM_CONTRACT_SHA256}, got {actual}"
            )
        return actual
    return EXPECTED_DOWNSTREAM_CONTRACT_SHA256


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


def runtime_artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = experiment_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_work_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / ".exp293_work"
    else:
        path = experiment_dir() / ".exp293_work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return experiment_dir() / "metrics.json"


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_gzip(
    path: Path, chunk_bytes: int = 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
    row_hashes = pd.util.hash_pandas_object(
        selected, index=False, categorize=True
    )
    digest.update(
        row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes()
    )
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    root = project_root()
    for raw_pattern in patterns:
        raw = str(raw_pattern)
        path = Path(raw)
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            found.append(direct)
            continue
        for match in glob.glob(raw, recursive=True):
            candidate = Path(match)
            if candidate.exists():
                found.append(candidate)
        if not path.is_absolute():
            for match in glob.glob(str(root / raw), recursive=True):
                candidate = Path(match)
                if candidate.exists():
                    found.append(candidate)
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(str(path.resolve()), path)
    return list(unique.values())


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


def reject_forbidden_candidate_columns(columns: Iterable[str]) -> None:
    normalized = {str(column) for column in columns}
    forbidden = normalized & FORBIDDEN_CANDIDATE_COLUMNS
    token_forbidden = {
        column
        for column in normalized
        if any(
            token in column.lower()
            for token in ("true_tvt", "abs_error", "oracle_label")
        )
    }
    if forbidden or token_forbidden:
        raise ValueError(
            "candidate partition exposes forbidden truth/readout columns: "
            f"{sorted(forbidden | token_forbidden)}"
        )


# %% [markdown]
# ## 3. Exp263 candidate-bank reconstruction and parity checks

# %%
@dataclass
class CandidateBank:
    keys: pd.DataFrame
    candidate_ids: tuple[str, ...]
    values: np.memmap
    values_path: Path
    primitive_ids: tuple[str, ...]
    manifest: dict[str, Any]
    manifest_path: Path
    key_content_sha256: str
    candidate_content_sha256: str
    coverage_by_candidate: dict[str, float]
    sample_parity: pd.DataFrame
    input_evidence: list[dict[str, Any]]


def _artifact_path_from_manifest(
    manifest_path: Path, item: Mapping[str, Any]
) -> Path:
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


def _read_manifest_partitions(
    manifest_path: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for item in items:
        path = _artifact_path_from_manifest(manifest_path, item)
        actual_file_sha = sha256_file(path)
        expected_file_sha = str(item.get("file_sha256", ""))
        if expected_file_sha and actual_file_sha != expected_file_sha:
            raise ValueError(f"{label} partition file SHA mismatch: {path}")
        full = pd.read_parquet(path)
        reject_forbidden_candidate_columns(full.columns)
        expected_rows = int(item.get("rows", len(full)))
        if len(full) != expected_rows:
            raise ValueError(f"{label} partition row mismatch: {path}")
        expected_schema = str(item.get("schema_sha256", ""))
        actual_schema = frame_schema_sha256(full)
        if expected_schema and actual_schema != expected_schema:
            raise ValueError(f"{label} partition schema SHA mismatch: {path}")
        expected_content = str(item.get("content_sha256", ""))
        actual_content = frame_content_sha256(full)
        if expected_content and actual_content != expected_content:
            raise ValueError(f"{label} partition content SHA mismatch: {path}")
        missing = set(columns) - set(full.columns)
        if missing:
            raise ValueError(f"{label} partition columns missing: {sorted(missing)}")
        frames.append(full[list(columns)].copy())
        evidence.append(
            {
                "phase": "target_free",
                "source": label,
                "path": str(path),
                "rows": len(full),
                "file_sha256": actual_file_sha,
                "decompressed_content_sha256": None,
                "logical_content_sha256": actual_content,
                "schema_sha256": actual_schema,
            }
        )
    if not frames:
        raise ValueError(f"no partitions declared for {label}")
    return pd.concat(frames, ignore_index=True), evidence


def _assert_same_keys(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} key row count mismatch")
    for column in VALUE_KEY_COLUMNS:
        left = reference[column].to_numpy()
        right = candidate[column].to_numpy()
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if column == "md_since"
            else np.array_equal(left, right)
        )
        if not equal:
            raise ValueError(f"{label} key mismatch in {column}")


def candidate_bank_content_sha256(
    bank: CandidateBank, chunk_rows: int = 100_000
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(bank.candidate_ids), separators=(",", ":")).encode())
    digest.update(bank.key_content_sha256.encode())
    for position, candidate_id in enumerate(bank.candidate_ids):
        digest.update(candidate_id.encode())
        for start in range(0, len(bank.keys), chunk_rows):
            end = min(start + chunk_rows, len(bank.keys))
            values = np.asarray(bank.values[start:end, position], dtype="<f4")
            digest.update(values.tobytes())
    return digest.hexdigest()


def _materialize_formulas(
    values: np.memmap,
    column_by_candidate: Mapping[str, int],
    config: Mapping[str, Any],
) -> None:
    pairs = get_nested(config, "candidate_bank.pairs")
    for candidate_id, weights in pairs.items():
        parents = list(weights)
        if len(parents) != 2 or any(
            not math.isclose(float(weights[parent]), 0.5) for parent in parents
        ):
            raise ValueError(f"{candidate_id} differs from fixed 50/50 contract")
        left = values[:, column_by_candidate[parents[0]]]
        right = values[:, column_by_candidate[parents[1]]]
        values[:, column_by_candidate[str(candidate_id)]] = (
            np.float32(0.5) * (left + right)
        ).astype(np.float32)

    fixed = get_nested(config, "candidate_bank.fixed_formula")
    if list(fixed) != ["exp226_w500_50_50"]:
        raise ValueError("fixed formula identity differs from exp293 contract")
    weights = fixed["exp226_w500_50_50"]
    expected = {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25}
    if weights != expected:
        raise ValueError("exp226_w500_50_50 weights differ from fixed contract")
    output = (
        np.float32(0.5) * values[:, column_by_candidate["exp226_k16"]]
        + np.float32(0.25) * values[:, column_by_candidate["likpf_mean"]]
        + np.float32(0.25) * values[:, column_by_candidate["exact_hmm"]]
    ).astype(np.float32)
    values[:, column_by_candidate["exp226_w500_50_50"]] = output
    values.flush()


def _build_sample_parity(
    bank: CandidateBank, parity_path: Path, tolerance: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample = pd.read_parquet(parity_path)
    if "id" not in sample:
        raise ValueError("exp263 small parity sample lacks id")
    missing = set(FORMULA_CANDIDATES) - set(sample.columns)
    if missing:
        raise ValueError(f"exp263 small parity formulas missing: {sorted(missing)}")
    indexer = pd.Index(bank.keys["id"].astype(str)).get_indexer(
        sample["id"].astype(str)
    )
    if np.any(indexer < 0):
        raise ValueError("exp263 small parity IDs are absent from candidate bank")
    position = {name: idx for idx, name in enumerate(bank.candidate_ids)}
    records: list[dict[str, Any]] = []
    for candidate_id in FORMULA_CANDIDATES:
        actual = np.asarray(bank.values[indexer, position[candidate_id]], dtype=np.float64)
        expected = pd.to_numeric(sample[candidate_id], errors="raise").to_numpy(
            dtype=np.float64
        )
        max_abs = float(np.max(np.abs(actual - expected), initial=0.0))
        records.append(
            {
                "check_type": "exp263_small_parity_max_abs_ft",
                "candidate_id": candidate_id,
                "actual": max_abs,
                "expected": 0.0,
                "absolute_difference": max_abs,
                "tolerance": tolerance,
                "passed": bool(max_abs <= tolerance),
            }
        )
    evidence = {
        "phase": "target_free",
        "source": "exp263_small_parity_sample",
        "path": str(parity_path),
        "rows": len(sample),
        "file_sha256": sha256_file(parity_path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(sample),
        "schema_sha256": frame_schema_sha256(sample),
    }
    return pd.DataFrame(records), evidence


def build_candidate_bank(
    config: Mapping[str, Any], work_dir: Path
) -> CandidateBank:
    manifest_cfg = get_nested(config, "data.exp263_manifest")
    manifest_path = resolve_file(
        manifest_cfg["patterns"],
        label="exp263 cache manifest",
        expected_sha256=str(manifest_cfg["expected_file_sha256"]),
    )
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = int(get_nested(config, "validation.n_folds"))
    if int(manifest.get("rows", -1)) != expected_rows:
        raise ValueError("exp263 manifest row contract mismatch")
    if int(manifest.get("wells", -1)) != expected_wells:
        raise ValueError("exp263 manifest well contract mismatch")
    if int(manifest.get("folds", -1)) != expected_folds:
        raise ValueError("exp263 manifest fold contract mismatch")
    if manifest.get("canonical_id_sha256") != manifest_cfg[
        "expected_canonical_id_sha256"
    ]:
        raise ValueError("exp263 canonical ID SHA mismatch")

    candidate_ids = tuple(get_nested(config, "candidate_bank.order"))
    if candidate_ids != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("candidate order differs from exp293 fixed contract")
    if len(candidate_ids) != int(get_nested(config, "candidate_bank.expected_count")):
        raise ValueError("candidate count differs from exp293 fixed contract")
    primitive_ids = tuple(get_nested(config, "candidate_bank.primitives"))
    if primitive_ids != EXPECTED_CANDIDATE_ORDER[:6]:
        raise ValueError("primitive order differs from exp293 fixed contract")

    values_path = work_dir / f"{OUTPUT_PREFIX}_candidate_bank.f32"
    values = np.memmap(
        values_path,
        mode="w+",
        dtype="float32",
        shape=(expected_rows, len(candidate_ids)),
    )
    values[:] = np.nan
    column_by_candidate = {name: idx for idx, name in enumerate(candidate_ids)}
    input_evidence: list[dict[str, Any]] = [
        {
            "phase": "target_free",
            "source": "exp263_manifest",
            "path": str(manifest_path),
            "rows": expected_rows,
            "file_sha256": sha256_file(manifest_path),
            "decompressed_content_sha256": None,
            "logical_content_sha256": manifest.get("canonical_id_sha256"),
            "schema_sha256": manifest.get("generation_config_sha256"),
        }
    ]
    reference_keys: pd.DataFrame | None = None
    coverage_by_candidate: dict[str, float] = {}
    for candidate_id in primitive_ids:
        items = manifest["candidate_value_partitions"].get(candidate_id)
        if not items or len(items) != expected_folds:
            raise ValueError(f"{candidate_id} must have five value partitions")
        frame, evidence = _read_manifest_partitions(
            manifest_path,
            items,
            columns=VALUE_READ_COLUMNS,
            label=f"exp263_value::{candidate_id}",
        )
        input_evidence.extend(evidence)
        if reference_keys is None:
            reference_keys = frame[VALUE_KEY_COLUMNS].copy()
            reference_keys["id"] = reference_keys["id"].astype(str)
            reference_keys["well"] = reference_keys["well"].astype(str)
        else:
            _assert_same_keys(reference_keys, frame, candidate_id)
        available = frame["candidate_available"].astype(bool).to_numpy()
        finite_flag = frame["candidate_finite"].astype(bool).to_numpy()
        candidate_values = pd.to_numeric(
            frame["candidate_tvt"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        valid = available & finite_flag & np.isfinite(candidate_values)
        candidate_values[~valid] = np.nan
        values[:, column_by_candidate[candidate_id]] = candidate_values
        coverage_by_candidate[candidate_id] = float(valid.mean())

    if reference_keys is None:
        raise AssertionError("primitive candidate loading produced no keys")
    if len(reference_keys) != expected_rows:
        raise ValueError("candidate bank total row mismatch")
    if reference_keys["well"].nunique() != expected_wells:
        raise ValueError("candidate bank total well mismatch")
    if reference_keys["id"].duplicated().any():
        raise ValueError("candidate bank IDs must be unique")
    if set(reference_keys["outer_fold"].unique()) != set(range(expected_folds)):
        raise ValueError("candidate bank outer-fold coverage mismatch")

    _materialize_formulas(values, column_by_candidate, config)
    for candidate_id in candidate_ids[6:]:
        finite = np.isfinite(values[:, column_by_candidate[candidate_id]])
        coverage_by_candidate[candidate_id] = float(finite.mean())
    if any(not math.isclose(value, 1.0) for value in coverage_by_candidate.values()):
        raise ValueError(f"candidate finite coverage is not 1.0: {coverage_by_candidate}")

    key_hash = frame_content_sha256(reference_keys[VALUE_KEY_COLUMNS])
    bank = CandidateBank(
        keys=reference_keys.reset_index(drop=True),
        candidate_ids=candidate_ids,
        values=values,
        values_path=values_path,
        primitive_ids=primitive_ids,
        manifest=manifest,
        manifest_path=manifest_path,
        key_content_sha256=key_hash,
        candidate_content_sha256="",
        coverage_by_candidate=coverage_by_candidate,
        sample_parity=pd.DataFrame(),
        input_evidence=input_evidence,
    )
    bank.candidate_content_sha256 = candidate_bank_content_sha256(
        bank, int(get_nested(config, "audit.work_chunk_rows"))
    )
    parity_path = manifest_path.parent / str(
        manifest_cfg["small_parity_filename"]
    )
    if not parity_path.exists():
        raise FileNotFoundError(f"exp263 small parity sample missing: {parity_path}")
    sample_parity, parity_evidence = _build_sample_parity(
        bank,
        parity_path,
        float(get_nested(config, "candidate_bank.formula_parity_max_abs_ft")),
    )
    if not bool(sample_parity["passed"].all()):
        raise ValueError("exp263 formula sample parity failed")
    bank.sample_parity = sample_parity
    bank.input_evidence.append(parity_evidence)
    return bank


# %% [markdown]
# ## 4. Block assignment and target-free freeze boundary

# %%
@dataclass
class GroupLayout:
    name: str
    codes: np.ndarray
    n_groups: int
    group_rows: np.ndarray
    group_well: np.ndarray
    group_fold: np.ndarray


@dataclass
class BlockAssignments:
    frame: pd.DataFrame
    well_names: np.ndarray
    well_codes: np.ndarray
    well_fold: np.ndarray
    layouts: dict[str, GroupLayout]


@dataclass(frozen=True)
class FreezeEvidence:
    contract_path: Path
    contract_file_sha256: str
    bank_manifest_path: Path
    bank_manifest_file_sha256: str
    candidate_content_sha256: str
    block_assignment_path: Path
    block_assignment_file_sha256: str
    block_assignment_decompressed_sha256: str
    target_free_input_evidence_sha256: str
    truth_access_count_before_freeze: int


def _layout_from_codes(
    name: str,
    codes: np.ndarray,
    well_codes: np.ndarray,
    row_folds: np.ndarray,
) -> GroupLayout:
    if len(codes) == 0 or np.any(codes < 0):
        raise ValueError(f"invalid group codes for {name}")
    n_groups = int(codes.max()) + 1
    rows = np.bincount(codes, minlength=n_groups).astype(np.int64)
    first = np.full(n_groups, len(codes), dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(codes), dtype=np.int64))
    if np.any(first == len(codes)):
        raise ValueError(f"group code gap for {name}")
    group_well = well_codes[first].astype(np.int32)
    group_fold = row_folds[first].astype(np.int8)
    return GroupLayout(
        name=name,
        codes=codes.astype(np.int32, copy=False),
        n_groups=n_groups,
        group_rows=rows,
        group_well=group_well,
        group_fold=group_fold,
    )


def build_block_assignments(
    keys: pd.DataFrame, horizons: Sequence[int]
) -> BlockAssignments:
    wells = keys["well"].astype(str).to_numpy()
    row_folds = pd.to_numeric(keys["outer_fold"], errors="raise").to_numpy(
        dtype=np.int8
    )
    if len(wells) == 0:
        raise ValueError("candidate keys are empty")
    segment_start_mask = np.r_[True, wells[1:] != wells[:-1]]
    starts = np.flatnonzero(segment_start_mask)
    ends = np.r_[starts[1:], len(wells)]
    segment_wells = wells[starts]
    if pd.Index(segment_wells).duplicated().any():
        raise ValueError("well rows are not contiguous in candidate bank")
    lengths = (ends - starts).astype(np.int64)
    well_codes = np.repeat(
        np.arange(len(starts), dtype=np.int32), lengths
    )
    within_well = np.arange(len(wells), dtype=np.int64) - np.repeat(starts, lengths)
    well_fold = row_folds[starts]
    for start, end, fold in zip(starts, ends, well_fold, strict=True):
        if not np.all(row_folds[start:end] == fold):
            raise ValueError("one well spans multiple outer folds")
    layouts: dict[str, GroupLayout] = {}
    for horizon in horizons:
        if int(horizon) <= 0:
            raise ValueError("block horizon must be positive")
        blocks_per_well = (lengths + int(horizon) - 1) // int(horizon)
        offsets = np.r_[0, np.cumsum(blocks_per_well[:-1])].astype(np.int64)
        codes = offsets[well_codes] + within_well // int(horizon)
        name = f"h{int(horizon)}"
        layouts[name] = _layout_from_codes(
            name, codes.astype(np.int32), well_codes, row_folds
        )
    layouts["whole_well"] = _layout_from_codes(
        "whole_well", well_codes.copy(), well_codes, row_folds
    )
    assignment = keys[VALUE_KEY_COLUMNS].copy()
    assignment["well_code"] = well_codes
    for name, layout in layouts.items():
        assignment[f"{name}_group"] = layout.codes
    return BlockAssignments(
        frame=assignment,
        well_names=np.asarray(segment_wells, dtype=object),
        well_codes=well_codes,
        well_fold=well_fold,
        layouts=layouts,
    )


def load_hidden_like_sets(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    hidden_cfg = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        hidden_cfg["patterns"],
        label="hidden-like assignment",
        expected_sha256=str(hidden_cfg["expected_file_sha256"]),
    )
    frame = pd.read_csv(path)
    well_column = str(hidden_cfg["well_column"])
    if well_column not in frame:
        raise ValueError(f"hidden-like well column missing: {well_column}")
    sets: dict[str, set[str]] = {}
    for scope, role_column in hidden_cfg["role_columns"].items():
        if role_column not in frame:
            raise ValueError(f"hidden-like role column missing: {role_column}")
        selected = set(
            frame.loc[frame[role_column].eq("valid"), well_column].astype(str)
        )
        unknown = selected - expected_wells
        if unknown:
            raise ValueError(f"hidden-like scope has unknown wells: {sorted(unknown)[:5]}")
        sets[str(scope)] = selected
    evidence = {
        "phase": "target_free",
        "source": "hidden_like_assignment",
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }
    return sets, evidence


def build_contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "candidate_order": list(get_nested(config, "candidate_bank.order")),
        "candidate_count": get_nested(config, "candidate_bank.expected_count"),
        "oracle_horizons_rows": get_nested(
            config, "validation.oracle_horizons_rows"
        ),
        "primary_horizon_rows": get_nested(
            config, "validation.primary_horizon_rows"
        ),
        "support_pass": get_nested(config, "validation.support_pass"),
        "branch_after_exp293": get_nested(
            config, "downstream.branch_after_exp293"
        ),
        "branch_after_stage2": get_nested(
            config, "downstream.branch_after_stage2"
        ),
        "stage4_return": get_nested(config, "downstream.stage4_return"),
        "execution": get_nested(config, "execution"),
        "forbidden_actions": get_nested(config, "audit.forbidden_actions"),
    }


def freeze_target_free_contract(
    bank: CandidateBank,
    assignments: BlockAssignments,
    hidden_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> FreezeEvidence:
    contract_path = artifacts_dir / f"{OUTPUT_PREFIX}.json"
    write_json(contract_path, build_contract_payload(config))

    block_path = artifacts_dir / f"{OUTPUT_PREFIX}_block_assignment.csv.gz"
    assignments.frame.to_csv(
        block_path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    block_file_sha = sha256_file(block_path)
    block_decompressed_sha = sha256_decompressed_gzip(block_path)
    block_logical_sha = frame_content_sha256(assignments.frame)

    target_free_evidence = [*bank.input_evidence, dict(hidden_evidence)]
    evidence_sha = json_sha256(target_free_evidence)
    manifest_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "target_free_candidate_bank_frozen",
        "frozen_at": datetime.now(UTC).isoformat(),
        "candidate_ids": list(bank.candidate_ids),
        "candidate_count": len(bank.candidate_ids),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "folds": sorted(int(value) for value in bank.keys["outer_fold"].unique()),
        "candidate_coverage": bank.coverage_by_candidate,
        "key_content_sha256": bank.key_content_sha256,
        "candidate_content_sha256": bank.candidate_content_sha256,
        "sample_formula_parity_content_sha256": frame_content_sha256(
            bank.sample_parity
        ),
        "block_assignment": {
            "path": str(block_path),
            "file_sha256": block_file_sha,
            "decompressed_content_sha256": block_decompressed_sha,
            "logical_content_sha256": block_logical_sha,
            "horizons_rows": list(
                get_nested(config, "audit.block_partition.horizons_rows")
            ),
            "include_final_short_block": bool(
                get_nested(config, "audit.block_partition.include_final_short_block")
            ),
        },
        "contract_file_sha256": sha256_file(contract_path),
        "config_file_sha256": sha256_file(find_config_path()),
        "downstream_branch_contract_file_sha256": downstream_contract_sha256(),
        "target_free_input_evidence_sha256": evidence_sha,
        "truth_access_count_before_freeze": 0,
        "truth_columns_loaded_before_freeze": [],
        "frozen": True,
    }
    bank_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_bank_manifest.json"
    write_json(bank_manifest_path, manifest_payload)
    return FreezeEvidence(
        contract_path=contract_path,
        contract_file_sha256=sha256_file(contract_path),
        bank_manifest_path=bank_manifest_path,
        bank_manifest_file_sha256=sha256_file(bank_manifest_path),
        candidate_content_sha256=bank.candidate_content_sha256,
        block_assignment_path=block_path,
        block_assignment_file_sha256=block_file_sha,
        block_assignment_decompressed_sha256=block_decompressed_sha,
        target_free_input_evidence_sha256=evidence_sha,
        truth_access_count_before_freeze=0,
    )


def verify_freeze_before_truth(
    bank: CandidateBank,
    freeze: FreezeEvidence,
    chunk_rows: int,
) -> None:
    if freeze.truth_access_count_before_freeze != 0:
        raise ValueError("truth was accessed before target-free freeze")
    if sha256_file(freeze.contract_path) != freeze.contract_file_sha256:
        raise ValueError("frozen contract changed before truth load")
    if sha256_file(freeze.bank_manifest_path) != freeze.bank_manifest_file_sha256:
        raise ValueError("bank manifest changed before truth load")
    if sha256_file(freeze.block_assignment_path) != freeze.block_assignment_file_sha256:
        raise ValueError("block assignment changed before truth load")
    if (
        sha256_decompressed_gzip(freeze.block_assignment_path)
        != freeze.block_assignment_decompressed_sha256
    ):
        raise ValueError("block assignment decompressed content changed")
    current_bank_sha = candidate_bank_content_sha256(bank, chunk_rows)
    if current_bank_sha != freeze.candidate_content_sha256:
        raise ValueError("candidate bank changed after target-free freeze")


# %% [markdown]
# ## 5. Post-freeze raw-train truth loader

# %%
def resolve_raw_train_dir(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[Path, list[Path]]:
    patterns = list(get_nested(config, "data.raw_train_dir_patterns"))
    horizontal_glob = str(get_nested(config, "data.raw_horizontal_glob"))
    candidates = [path for path in expand_existing_paths(patterns) if path.is_dir()]
    evidence: list[tuple[Path, list[Path], set[str]]] = []
    for directory in candidates:
        files = sorted(directory.glob(horizontal_glob))
        wells = {
            path.name.split("__horizontal_well.csv", 1)[0]
            for path in files
        }
        evidence.append((directory, files, wells))
        if wells == expected_wells and len(files) == len(expected_wells):
            return directory, files
    detail = {
        str(directory): {"files": len(files), "wells": len(wells)}
        for directory, files, wells in evidence
    }
    raise FileNotFoundError(
        "raw train directory with exact candidate-well inventory was not found: "
        f"{detail}"
    )


def load_truth_after_freeze(
    bank: CandidateBank,
    freeze: FreezeEvidence,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    chunk_rows = int(get_nested(config, "audit.work_chunk_rows"))
    verify_freeze_before_truth(bank, freeze, chunk_rows)
    expected_wells = set(bank.keys["well"].astype(str))
    raw_dir, files = resolve_raw_train_dir(config, expected_wells)
    truth_column = str(get_nested(config, "data.raw_columns.truth"))
    visible_column = str(get_nested(config, "data.raw_columns.visible_input"))
    truth_frames: list[pd.DataFrame] = []
    input_evidence: list[dict[str, Any]] = []
    for path in files:
        well = path.name.split("__horizontal_well.csv", 1)[0]
        frame = pd.read_csv(path, usecols=[truth_column, visible_column])
        visible = pd.to_numeric(frame[visible_column], errors="coerce")
        suffix_mask = visible.isna().to_numpy()
        row_idx = np.flatnonzero(suffix_mask).astype(np.int32)
        truth_values = pd.to_numeric(
            frame.loc[suffix_mask, truth_column], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(truth_values).all():
            raise ValueError(f"raw truth contains nonfinite suffix TVT: {path}")
        truth_frames.append(
            pd.DataFrame(
                {
                    "id": pd.Series(well, index=np.arange(len(row_idx)), dtype=str)
                    + "_"
                    + pd.Series(row_idx).astype(str),
                    "well": well,
                    "well_row_idx": row_idx,
                    "true_tvt": truth_values,
                }
            )
        )
        input_evidence.append(
            {
                "phase": "post_freeze_truth",
                "source": "raw_train_horizontal",
                "path": str(path),
                "rows": len(frame),
                "suffix_rows": len(row_idx),
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": None,
                "logical_content_sha256": None,
                "schema_sha256": None,
                "raw_train_dir": str(raw_dir),
            }
        )
    truth_frame = pd.concat(truth_frames, ignore_index=True)
    if truth_frame["id"].duplicated().any():
        raise ValueError("raw truth IDs are duplicated")
    if len(truth_frame) != len(bank.keys):
        raise ValueError(
            f"raw truth row mismatch: {len(truth_frame)} != {len(bank.keys)}"
        )
    truth_index = pd.Index(truth_frame["id"].astype(str))
    indexer = truth_index.get_indexer(bank.keys["id"].astype(str))
    if np.any(indexer < 0):
        raise ValueError("candidate IDs are missing from post-freeze raw truth")
    aligned = truth_frame.iloc[indexer].reset_index(drop=True)
    if not np.array_equal(
        aligned["id"].astype(str).to_numpy(),
        bank.keys["id"].astype(str).to_numpy(),
    ):
        raise ValueError("post-freeze truth identity alignment failed")
    truth = aligned["true_tvt"].to_numpy(dtype=np.float64)
    truth_sha = frame_content_sha256(aligned[["id", "true_tvt"]])
    return truth, input_evidence, truth_sha


# %% [markdown]
# ## 6. Chunked oracle aggregation and diagnostic readouts

# %%
@dataclass
class OracleState:
    row_best_sse: np.ndarray
    row_best_candidate: np.ndarray
    anchor_row_sse: np.ndarray
    candidate_total_sse: np.ndarray
    group_sse: dict[str, np.ndarray]
    longtail_group_sse: dict[str, np.ndarray]
    longtail_group_rows: dict[str, np.ndarray]


def compute_oracle_state(
    bank: CandidateBank,
    truth: np.ndarray,
    assignments: BlockAssignments,
    config: Mapping[str, Any],
) -> OracleState:
    n_rows = len(bank.keys)
    if len(truth) != n_rows or not np.isfinite(truth).all():
        raise ValueError("truth vector does not match finite candidate rows")
    n_candidates = len(bank.candidate_ids)
    chunk_rows = int(get_nested(config, "audit.work_chunk_rows"))
    row_best_sse = np.full(n_rows, np.inf, dtype=np.float64)
    row_best_candidate = np.full(n_rows, 255, dtype=np.uint8)
    anchor_row_sse = np.full(n_rows, np.nan, dtype=np.float64)
    candidate_total_sse = np.zeros(n_candidates, dtype=np.float64)
    group_sse = {
        name: np.zeros((layout.n_groups, n_candidates), dtype=np.float64)
        for name, layout in assignments.layouts.items()
    }
    longtail_mask = bank.keys["md_since"].to_numpy(dtype=np.float64) >= 1000.0
    longtail_group_sse = {
        name: np.zeros((layout.n_groups, n_candidates), dtype=np.float64)
        for name, layout in assignments.layouts.items()
    }
    longtail_group_rows = {
        name: np.bincount(
            layout.codes[longtail_mask], minlength=layout.n_groups
        ).astype(np.int64)
        for name, layout in assignments.layouts.items()
    }
    anchor_position = bank.candidate_ids.index(
        str(get_nested(config, "validation.anchor_candidate"))
    )
    for candidate_position in range(n_candidates):
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            candidate = np.asarray(
                bank.values[start:end, candidate_position], dtype=np.float64
            )
            if not np.isfinite(candidate).all():
                raise ValueError(
                    f"nonfinite candidate reached oracle: "
                    f"{bank.candidate_ids[candidate_position]}"
                )
            error_squared = np.square(candidate - truth[start:end])
            candidate_total_sse[candidate_position] += float(error_squared.sum())
            current = row_best_sse[start:end]
            better = error_squared < current
            current[better] = error_squared[better]
            row_best_candidate[start:end][better] = candidate_position
            if candidate_position == anchor_position:
                anchor_row_sse[start:end] = error_squared
            local_longtail = longtail_mask[start:end]
            for name, layout in assignments.layouts.items():
                local_codes = layout.codes[start:end]
                group_sse[name][:, candidate_position] += np.bincount(
                    local_codes,
                    weights=error_squared,
                    minlength=layout.n_groups,
                )
                if local_longtail.any():
                    longtail_group_sse[name][:, candidate_position] += np.bincount(
                        local_codes[local_longtail],
                        weights=error_squared[local_longtail],
                        minlength=layout.n_groups,
                    )
    if not np.isfinite(row_best_sse).all():
        raise ValueError("row oracle contains nonfinite SSE")
    if np.any(row_best_candidate == 255):
        raise ValueError("row oracle candidate identity is incomplete")
    if not np.isfinite(anchor_row_sse).all():
        raise ValueError("anchor row SSE is incomplete")
    return OracleState(
        row_best_sse=row_best_sse,
        row_best_candidate=row_best_candidate,
        anchor_row_sse=anchor_row_sse,
        candidate_total_sse=candidate_total_sse,
        group_sse=group_sse,
        longtail_group_sse=longtail_group_sse,
        longtail_group_rows=longtail_group_rows,
    )


def required_headroom_recovery(
    anchor_sse: float,
    oracle_sse: float,
    rows: int,
    target_rmse: float,
) -> float:
    target_sse = float(rows) * float(target_rmse) ** 2
    denominator = float(anchor_sse) - float(oracle_sse)
    if rows <= 0 or not np.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return (float(anchor_sse) - target_sse) / denominator


def row_metric_record(
    state: OracleState,
    row_mask: np.ndarray,
    *,
    scope: str,
    scope_type: str,
    target_rmse: float,
) -> dict[str, Any]:
    rows = int(row_mask.sum())
    if rows == 0:
        oracle_sse = anchor_sse = math.nan
        oracle_rmse = anchor_rmse = recovery = math.nan
    else:
        oracle_sse = float(state.row_best_sse[row_mask].sum())
        anchor_sse = float(state.anchor_row_sse[row_mask].sum())
        oracle_rmse = math.sqrt(oracle_sse / rows)
        anchor_rmse = math.sqrt(anchor_sse / rows)
        recovery = required_headroom_recovery(
            anchor_sse, oracle_sse, rows, target_rmse
        )
    return {
        "scope": scope,
        "scope_type": scope_type,
        "granularity": "row",
        "rows": rows,
        "groups": rows,
        "anchor_sse": anchor_sse,
        "anchor_rmse": anchor_rmse,
        "oracle_sse": oracle_sse,
        "oracle_rmse": oracle_rmse,
        "oracle_gain_rmse": anchor_rmse - oracle_rmse,
        "target_rmse": target_rmse,
        "required_recovery_to_6p5": recovery,
    }


def group_metric_record(
    candidate_group_sse: np.ndarray,
    group_rows: np.ndarray,
    group_mask: np.ndarray,
    anchor_position: int,
    *,
    scope: str,
    scope_type: str,
    granularity: str,
    target_rmse: float,
) -> dict[str, Any]:
    selected = group_mask & (group_rows > 0)
    rows = int(group_rows[selected].sum())
    groups = int(selected.sum())
    if rows == 0:
        oracle_sse = anchor_sse = math.nan
        oracle_rmse = anchor_rmse = recovery = math.nan
    else:
        local = candidate_group_sse[selected]
        oracle_sse = float(np.min(local, axis=1).sum())
        anchor_sse = float(local[:, anchor_position].sum())
        oracle_rmse = math.sqrt(oracle_sse / rows)
        anchor_rmse = math.sqrt(anchor_sse / rows)
        recovery = required_headroom_recovery(
            anchor_sse, oracle_sse, rows, target_rmse
        )
    return {
        "scope": scope,
        "scope_type": scope_type,
        "granularity": granularity,
        "rows": rows,
        "groups": groups,
        "anchor_sse": anchor_sse,
        "anchor_rmse": anchor_rmse,
        "oracle_sse": oracle_sse,
        "oracle_rmse": oracle_rmse,
        "oracle_gain_rmse": anchor_rmse - oracle_rmse,
        "target_rmse": target_rmse,
        "required_recovery_to_6p5": recovery,
    }


def row_choice_records(
    state: OracleState,
    row_mask: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    selected = state.row_best_candidate[row_mask]
    counts = np.bincount(selected, minlength=len(candidate_ids))
    return [
        {
            "scope": scope,
            "granularity": "row",
            "candidate_id": candidate_id,
            "choice_groups": int(counts[position]),
            "choice_rows": int(counts[position]),
        }
        for position, candidate_id in enumerate(candidate_ids)
    ]


def group_choice_records(
    candidate_group_sse: np.ndarray,
    group_rows: np.ndarray,
    group_mask: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    scope: str,
    granularity: str,
) -> list[dict[str, Any]]:
    selected = group_mask & (group_rows > 0)
    best = np.argmin(candidate_group_sse[selected], axis=1)
    rows = group_rows[selected]
    return [
        {
            "scope": scope,
            "granularity": granularity,
            "candidate_id": candidate_id,
            "choice_groups": int(np.sum(best == position)),
            "choice_rows": int(rows[best == position].sum()),
        }
        for position, candidate_id in enumerate(candidate_ids)
    ]


def build_metric_frames(
    bank: CandidateBank,
    assignments: BlockAssignments,
    state: OracleState,
    hidden_sets: Mapping[str, set[str]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(bank.keys)
    candidate_ids = bank.candidate_ids
    anchor_position = candidate_ids.index(
        str(get_nested(config, "validation.anchor_candidate"))
    )
    target_rmse = float(get_nested(config, "audit.headroom.target_rmse"))
    wells = bank.keys["well"].astype(str).to_numpy()
    folds = bank.keys["outer_fold"].to_numpy(dtype=np.int8)
    md_since = bank.keys["md_since"].to_numpy(dtype=np.float64)
    all_rows = np.ones(n_rows, dtype=bool)

    overall_records = [
        row_metric_record(
            state,
            all_rows,
            scope="overall",
            scope_type="overall",
            target_rmse=target_rmse,
        )
    ]
    choice_records = row_choice_records(
        state, all_rows, candidate_ids, scope="overall"
    )
    for name, layout in assignments.layouts.items():
        mask = np.ones(layout.n_groups, dtype=bool)
        overall_records.append(
            group_metric_record(
                state.group_sse[name],
                layout.group_rows,
                mask,
                anchor_position,
                scope="overall",
                scope_type="overall",
                granularity=name,
                target_rmse=target_rmse,
            )
        )
        choice_records.extend(
            group_choice_records(
                state.group_sse[name],
                layout.group_rows,
                mask,
                candidate_ids,
                scope="overall",
                granularity=name,
            )
        )

    fold_records: list[dict[str, Any]] = []
    for fold in range(int(get_nested(config, "validation.n_folds"))):
        scope = f"fold_{fold}"
        row_mask = folds == fold
        fold_records.append(
            row_metric_record(
                state,
                row_mask,
                scope=scope,
                scope_type="fold",
                target_rmse=target_rmse,
            )
        )
        choice_records.extend(
            row_choice_records(state, row_mask, candidate_ids, scope=scope)
        )
        for name, layout in assignments.layouts.items():
            group_mask = layout.group_fold == fold
            fold_records.append(
                group_metric_record(
                    state.group_sse[name],
                    layout.group_rows,
                    group_mask,
                    anchor_position,
                    scope=scope,
                    scope_type="fold",
                    granularity=name,
                    target_rmse=target_rmse,
                )
            )
            choice_records.extend(
                group_choice_records(
                    state.group_sse[name],
                    layout.group_rows,
                    group_mask,
                    candidate_ids,
                    scope=scope,
                    granularity=name,
                )
            )

    distance_buckets = get_nested(config, "audit.distance_buckets_ft")
    for bucket, bounds in distance_buckets.items():
        lower, upper = float(bounds[0]), float(bounds[1])
        mask = (md_since >= lower) & (md_since < upper)
        overall_records.append(
            row_metric_record(
                state,
                mask,
                scope=f"distance_{bucket}",
                scope_type="distance_bucket",
                target_rmse=target_rmse,
            )
        )

    subgroup_records: list[dict[str, Any]] = []
    longtail_mask = md_since >= 1000.0
    subgroup_records.append(
        row_metric_record(
            state,
            longtail_mask,
            scope="1000_plus",
            scope_type="risk_subgroup",
            target_rmse=target_rmse,
        )
    )
    choice_records.extend(
        row_choice_records(
            state, longtail_mask, candidate_ids, scope="1000_plus"
        )
    )
    for name in assignments.layouts:
        group_rows = state.longtail_group_rows[name]
        group_mask = group_rows > 0
        subgroup_records.append(
            group_metric_record(
                state.longtail_group_sse[name],
                group_rows,
                group_mask,
                anchor_position,
                scope="1000_plus",
                scope_type="risk_subgroup",
                granularity=name,
                target_rmse=target_rmse,
            )
        )
        choice_records.extend(
            group_choice_records(
                state.longtail_group_sse[name],
                group_rows,
                group_mask,
                candidate_ids,
                scope="1000_plus",
                granularity=name,
            )
        )

    for scope, selected_wells in hidden_sets.items():
        well_mask = np.isin(wells, np.asarray(sorted(selected_wells), dtype=object))
        subgroup_records.append(
            row_metric_record(
                state,
                well_mask,
                scope=scope,
                scope_type="risk_subgroup",
                target_rmse=target_rmse,
            )
        )
        choice_records.extend(
            row_choice_records(state, well_mask, candidate_ids, scope=scope)
        )
        selected_by_well = np.isin(
            assignments.well_names,
            np.asarray(sorted(selected_wells), dtype=object),
        )
        for name, layout in assignments.layouts.items():
            group_mask = selected_by_well[layout.group_well]
            subgroup_records.append(
                group_metric_record(
                    state.group_sse[name],
                    layout.group_rows,
                    group_mask,
                    anchor_position,
                    scope=scope,
                    scope_type="risk_subgroup",
                    granularity=name,
                    target_rmse=target_rmse,
                )
            )
            choice_records.extend(
                group_choice_records(
                    state.group_sse[name],
                    layout.group_rows,
                    group_mask,
                    candidate_ids,
                    scope=scope,
                    granularity=name,
                )
            )
    return (
        pd.DataFrame(overall_records),
        pd.DataFrame(fold_records),
        pd.DataFrame(subgroup_records),
        pd.DataFrame(choice_records),
    )


def build_by_well_metrics(
    bank: CandidateBank,
    assignments: BlockAssignments,
    state: OracleState,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    n_wells = len(assignments.well_names)
    well_rows = np.bincount(
        assignments.well_codes, minlength=n_wells
    ).astype(np.int64)
    row_oracle_sse = np.bincount(
        assignments.well_codes,
        weights=state.row_best_sse,
        minlength=n_wells,
    )
    anchor_position = bank.candidate_ids.index(
        str(get_nested(config, "validation.anchor_candidate"))
    )
    whole = assignments.layouts["whole_well"]
    anchor_group_sse = state.group_sse["whole_well"][:, anchor_position]
    anchor_sse = np.bincount(
        whole.group_well,
        weights=anchor_group_sse,
        minlength=n_wells,
    )
    output = pd.DataFrame(
        {
            "well": assignments.well_names,
            "outer_fold": assignments.well_fold,
            "rows": well_rows,
            "anchor_rmse": np.sqrt(anchor_sse / well_rows),
            "row_oracle_rmse": np.sqrt(row_oracle_sse / well_rows),
        }
    )
    for name, layout in assignments.layouts.items():
        group_best = np.min(state.group_sse[name], axis=1)
        oracle_sse = np.bincount(
            layout.group_well,
            weights=group_best,
            minlength=n_wells,
        )
        output[f"{name}_oracle_rmse"] = np.sqrt(oracle_sse / well_rows)
    return output.sort_values("well", kind="stable").reset_index(drop=True)


def build_formula_parity(
    bank: CandidateBank,
    state: OracleState,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    records = bank.sample_parity.to_dict("records")
    expected = get_nested(config, "candidate_bank.expected_oof_rmse")
    tolerance = float(
        get_nested(config, "candidate_bank.expected_oof_rmse_tolerance_ft")
    )
    actual_rmse = np.sqrt(state.candidate_total_sse / len(bank.keys))
    for position, candidate_id in enumerate(bank.candidate_ids):
        expected_value = float(expected[candidate_id])
        difference = abs(float(actual_rmse[position]) - expected_value)
        records.append(
            {
                "check_type": "exp263_oof_rmse_parity",
                "candidate_id": candidate_id,
                "actual": float(actual_rmse[position]),
                "expected": expected_value,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": bool(difference <= tolerance),
            }
        )
    return pd.DataFrame(records)


# %% [markdown]
# ## 7. Support decision and generated artifacts

# %%
def evaluate_support_decision(
    bank: CandidateBank,
    freeze: FreezeEvidence,
    oracle_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    formula_parity: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = int(get_nested(config, "validation.n_folds"))
    technical_checks = {
        "row_count_exact": len(bank.keys) == expected_rows,
        "well_count_exact": bank.keys["well"].nunique() == expected_wells,
        "fold_inventory_exact": set(bank.keys["outer_fold"].unique())
        == set(range(expected_folds)),
        "candidate_count_exact": len(bank.candidate_ids)
        == int(get_nested(config, "candidate_bank.expected_count")),
        "candidate_order_exact": bank.candidate_ids == EXPECTED_CANDIDATE_ORDER,
        "candidate_coverage_one": all(
            math.isclose(value, 1.0)
            for value in bank.coverage_by_candidate.values()
        ),
        "duplicate_id_zero": not bank.keys["id"].duplicated().any(),
        "formula_and_rmse_parity": bool(formula_parity["passed"].all()),
        "candidate_bank_frozen": freeze.bank_manifest_path.exists(),
        "truth_access_before_freeze_zero": freeze.truth_access_count_before_freeze
        == 0,
    }
    technical_passed = bool(all(technical_checks.values()))

    support_cfg = get_nested(config, "validation.support_pass")
    pooled = oracle_metrics[
        oracle_metrics["scope"].eq("overall")
        & oracle_metrics["granularity"].eq("h512")
    ]
    if len(pooled) != 1:
        raise ValueError("pooled H512 metric is not unique")
    pooled_row = pooled.iloc[0]
    fold_h512 = fold_metrics[fold_metrics["granularity"].eq("h512")].copy()
    if len(fold_h512) != expected_folds:
        raise ValueError("fold H512 metric inventory mismatch")
    recovery = float(pooled_row["required_recovery_to_6p5"])
    scientific_checks = {
        "pooled_h512_oracle_rmse_at_most_5p5": float(
            pooled_row["oracle_rmse"]
        )
        <= float(support_cfg["maximum_pooled_h512_oracle_rmse"]),
        "each_fold_h512_oracle_rmse_below_6p5": bool(
            (
                fold_h512["oracle_rmse"]
                < float(
                    support_cfg[
                        "maximum_each_fold_h512_oracle_rmse_exclusive"
                    ]
                )
            ).all()
        ),
        "h512_improves_anchor_all_folds": bool(
            (fold_h512["oracle_rmse"] < fold_h512["anchor_rmse"]).all()
        ),
        "required_recovery_finite": bool(np.isfinite(recovery)),
        "required_recovery_at_most_one": bool(
            np.isfinite(recovery)
            and recovery
            <= float(support_cfg["maximum_required_recovery_to_6p5"])
        ),
    }
    scientific_passed = bool(all(scientific_checks.values()))
    support_passed = bool(technical_passed and scientific_passed)
    branch_map = get_nested(config, "downstream.branch_after_exp293")
    next_branch = branch_map["support_pass" if support_passed else "support_fail"]

    risk_rows = subgroup_metrics[
        subgroup_metrics["granularity"].eq("h512")
    ]
    risk_flags = {
        str(row.scope): {
            "rows": int(row.rows),
            "anchor_rmse": float(row.anchor_rmse),
            "oracle_rmse": float(row.oracle_rmse),
            "oracle_improves_anchor": bool(row.oracle_rmse < row.anchor_rmse),
            "required_recovery_to_6p5": float(row.required_recovery_to_6p5),
        }
        for row in risk_rows.itertuples(index=False)
    }
    return {
        "technical_checks": technical_checks,
        "technical_passed": technical_passed,
        "scientific_checks": scientific_checks,
        "scientific_support_passed": scientific_passed,
        "support_passed": support_passed,
        "next_branch": str(next_branch),
        "primary_h512": {
            "rows": int(pooled_row["rows"]),
            "anchor_rmse": float(pooled_row["anchor_rmse"]),
            "oracle_rmse": float(pooled_row["oracle_rmse"]),
            "required_recovery_to_6p5": recovery,
        },
        "fold_h512": fold_h512[
            [
                "scope",
                "rows",
                "anchor_rmse",
                "oracle_rmse",
                "required_recovery_to_6p5",
            ]
        ].to_dict("records"),
        "diagnostic_only_risk_flags": risk_flags,
    }


def validate_execution_contract(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp293 route must remain pf_beam")
    execution = get_nested(config, "execution")
    expected_zero = [
        "lightgbm_config_count",
        "trained_fold_count",
        "total_boosters",
        "hmm_pf_well_runs",
    ]
    if any(int(execution[key]) != 0 for key in expected_zero):
        raise ValueError("exp293 must remain a zero-model, zero-PF-regeneration audit")
    if int(execution["active_audit_contracts"]) != 1:
        raise ValueError("exp293 must run exactly one fixed audit contract")
    if execution["inference"] or execution["submission"]:
        raise ValueError("exp293 inference and submission must remain disabled")
    if not bool(execution["kaggle_push_approved"]):
        raise ValueError("exp293 train push is not approved")
    if not bool(execution["canonical_train_notebook_adopted"]):
        raise ValueError("exp293 canonical train notebook is not adopted")
    if bool(execution["kaggle_inference_push_approved"]):
        raise ValueError("exp293 inference push must remain unapproved")
    if bool(execution["canonical_inference_notebook_adopted"]):
        raise ValueError("exp293 canonical inference notebook must remain unadopted")
    if not bool(execution["implementation"]):
        raise ValueError("exp293 implementation flag is not enabled")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def persist_audit_outputs(
    bank: CandidateBank,
    assignments: BlockAssignments,
    freeze: FreezeEvidence,
    hidden_evidence: Mapping[str, Any],
    truth_evidence: Sequence[Mapping[str, Any]],
    truth_content_sha256: str,
    oracle_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    choice_counts: pd.DataFrame,
    formula_parity: pd.DataFrame,
    support: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    input_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv"
    formula_path = artifacts_dir / f"{OUTPUT_PREFIX}_formula_parity.csv"
    oracle_path = artifacts_dir / f"{OUTPUT_PREFIX}_oracle_metrics.csv"
    fold_path = artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv"
    subgroup_path = artifacts_dir / f"{OUTPUT_PREFIX}_subgroup_metrics.csv"
    by_well_path = artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    choice_path = artifacts_dir / f"{OUTPUT_PREFIX}_choice_counts.csv"
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    sha_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"

    aligned_truth_evidence = {
        "phase": "post_freeze_truth",
        "source": "aligned_truth_readout",
        "path": "not_persisted",
        "rows": len(bank.keys),
        "file_sha256": None,
        "decompressed_content_sha256": None,
        "logical_content_sha256": truth_content_sha256,
        "schema_sha256": None,
    }
    input_manifest = pd.DataFrame(
        [
            *bank.input_evidence,
            dict(hidden_evidence),
            *[dict(item) for item in truth_evidence],
            aligned_truth_evidence,
        ]
    )
    _write_frame(input_manifest_path, input_manifest)
    _write_frame(formula_path, formula_parity)
    _write_frame(oracle_path, oracle_metrics)
    _write_frame(fold_path, fold_metrics)
    _write_frame(subgroup_path, subgroup_metrics)
    _write_frame(by_well_path, by_well)
    _write_frame(choice_path, choice_counts)

    readout_content_sha = json_sha256(
        {
            "oracle": frame_content_sha256(oracle_metrics),
            "fold": frame_content_sha256(fold_metrics),
            "subgroup": frame_content_sha256(subgroup_metrics),
            "by_well": frame_content_sha256(by_well),
            "choice": frame_content_sha256(choice_counts),
            "formula": frame_content_sha256(formula_parity),
        }
    )
    h512_by_well = by_well["h512_oracle_rmse"]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "audit_completed",
        "route": "pf_beam",
        "completed_at": datetime.now(UTC).isoformat(),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "candidate_count": len(bank.candidate_ids),
        "candidate_ids": list(bank.candidate_ids),
        "candidate_bank_content_sha256": bank.candidate_content_sha256,
        "block_assignment_file_sha256": freeze.block_assignment_file_sha256,
        "block_assignment_decompressed_sha256": (
            freeze.block_assignment_decompressed_sha256
        ),
        "truth_content_sha256": truth_content_sha256,
        "oracle_readout_content_sha256": readout_content_sha,
        "support": dict(support),
        "by_well_h512": {
            "p95_rmse": float(h512_by_well.quantile(0.95)),
            "worst_rmse": float(h512_by_well.max()),
            "worst_well": str(
                by_well.loc[h512_by_well.idxmax(), "well"]
            ),
        },
        "execution": {
            "active_audit_contracts": 1,
            "lightgbm_config_count": 0,
            "trained_fold_count": 0,
            "total_boosters": 0,
            "hmm_pf_well_runs": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
        "oracle_prediction_persisted": False,
    }
    write_json(summary_path, summary)

    artifact_paths = [
        freeze.contract_path,
        input_manifest_path,
        freeze.bank_manifest_path,
        formula_path,
        freeze.block_assignment_path,
        oracle_path,
        fold_path,
        subgroup_path,
        by_well_path,
        choice_path,
        summary_path,
    ]
    sha_records: list[dict[str, Any]] = []
    for path in artifact_paths:
        sha_records.append(
            {
                "artifact": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": (
                    sha256_decompressed_gzip(path)
                    if path.suffix == ".gz"
                    else None
                ),
            }
        )
    sha_manifest = pd.DataFrame(sha_records)
    _write_frame(sha_manifest_path, sha_manifest)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "implementation_complete_audit_executed",
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "candidate_bank": get_nested(config, "candidate_bank.name"),
        "candidate_count": len(bank.candidate_ids),
        "primary_horizon_rows": int(
            get_nested(config, "validation.primary_horizon_rows")
        ),
        "anchor_oof_rmse": float(
            support["primary_h512"]["anchor_rmse"]
        ),
        "h512_oracle_rmse": float(
            support["primary_h512"]["oracle_rmse"]
        ),
        "required_recovery_to_6p5": float(
            support["primary_h512"]["required_recovery_to_6p5"]
        ),
        "support_passed": bool(support["support_passed"]),
        "next_branch": str(support["next_branch"]),
        "cv": float(support["primary_h512"]["oracle_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "active_audit_contracts": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_pf_well_runs": 0,
        "oracle_readout_content_sha256": readout_content_sha,
        "sha_manifest_file_sha256": sha256_file(sha_manifest_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config)
    artifacts_dir = runtime_artifacts_dir()
    work_dir = runtime_work_dir()
    horizons = [
        int(value)
        for value in get_nested(config, "audit.block_partition.horizons_rows")
    ]
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Candidate order:", get_nested(config, "candidate_bank.order"))
    print("Oracle horizons:", ["row", *horizons, "whole_well"])
    print("Execution contract: 1 audit / 0 configs / 0 trained folds / 0 boosters")

    bank = build_candidate_bank(config, work_dir)
    assignments = build_block_assignments(bank.keys, horizons)
    hidden_sets, hidden_evidence = load_hidden_like_sets(
        config, set(bank.keys["well"].astype(str))
    )
    freeze = freeze_target_free_contract(
        bank, assignments, hidden_evidence, config, artifacts_dir
    )
    print("Target-free candidate bank frozen:", bank.candidate_content_sha256)
    print("Truth access count before freeze:", freeze.truth_access_count_before_freeze)

    truth, truth_evidence, truth_sha = load_truth_after_freeze(
        bank, freeze, config
    )
    print("Post-freeze truth attached:", truth_sha)
    state = compute_oracle_state(bank, truth, assignments, config)
    oracle_metrics, fold_metrics, subgroup_metrics, choice_counts = (
        build_metric_frames(bank, assignments, state, hidden_sets, config)
    )
    by_well = build_by_well_metrics(bank, assignments, state, config)
    formula_parity = build_formula_parity(bank, state, config)
    support = evaluate_support_decision(
        bank,
        freeze,
        oracle_metrics,
        fold_metrics,
        subgroup_metrics,
        formula_parity,
        config,
    )
    summary = persist_audit_outputs(
        bank,
        assignments,
        freeze,
        hidden_evidence,
        truth_evidence,
        truth_sha,
        oracle_metrics,
        fold_metrics,
        subgroup_metrics,
        by_well,
        choice_counts,
        formula_parity,
        support,
        config,
        artifacts_dir,
    )
    print("H512 anchor RMSE:", support["primary_h512"]["anchor_rmse"])
    print("H512 oracle RMSE:", support["primary_h512"]["oracle_rmse"])
    print(
        "Required recovery to 6.5:",
        support["primary_h512"]["required_recovery_to_6p5"],
    )
    print("Support passed:", support["support_passed"])
    print("Next branch:", support["next_branch"])
    print("Artifacts:", artifacts_dir)
    return summary


# %% [markdown]
# ## 8. Setup, contract preview, and execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Config:", CONFIG_PATH)
    print("Config SHA256:", sha256_file(CONFIG_PATH))
    print("Downstream contract SHA256:", downstream_contract_sha256())
    SUMMARY = run_audit(CONFIG)
    print(json.dumps(SUMMARY["support"], indent=2, ensure_ascii=False))
