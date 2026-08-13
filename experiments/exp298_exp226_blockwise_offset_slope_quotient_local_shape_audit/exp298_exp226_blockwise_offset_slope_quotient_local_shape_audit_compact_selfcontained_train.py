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
# # exp298 exp226 blockwise offset/slope quotient local-shape audit
#
# This train-side audit reconstructs the frozen exp293 deployable12 bank and
# exp226 geop/pre-U/post-U components without materializing truth. It freezes
# candidate values, component values, identity, folds, and the exact exp293
# H128/H256/H512 blocks before opening raw-train TVT, then removes only the
# diagnostic block offset or affine trend while measuring local-shape error.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Exp263 candidate-bank reconstruction and parity checks
# 4. Exp226 component reconstruction, block assignment, and target-free freeze
# 5. Post-freeze raw-train truth loader
# 6. Blockwise quotient aggregation and diagnostic readouts
# 7. PASS/FAIL decision and generated artifacts
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

EXPERIMENT_NAME = "exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
ARTIFACT_PREFIX = "exp298_local_shape"
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
COMPONENT_IDS = ("exp226_geop", "exp226_pre_u", "exp226_post_u")
RANK_CANDIDATE_ORDER = (*EXPECTED_CANDIDATE_ORDER, "exp226_pre_u")
READOUT_CANDIDATE_ORDER = (*RANK_CANDIDATE_ORDER, "exp226_geop", "exp226_post_u")
EXP226_ALLOWLIST = (
    "well_id",
    "fold",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "gr_delta",
    "tvt_pred",
)
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
    "f07dfb17fedb95fcb6a9df892990da3bd6e35a0108394255278473784a1fe8b8"
)
CURRENT_TO_EXECUTED_CONTRACT_TEXT = {
    "この文書、exp298のrequirements.md、": "この文書、exp298のsteering、",
    "`backlog/KAGGLE_DIRECTION.md`": "`KAGGLE_DIRECTION.md`",
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
    os.environ.get("EXP298_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
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
    raise FileNotFoundError("exp298 config.yaml was not found unambiguously")


def downstream_contract_sha256() -> str:
    for path in (
        Path.cwd() / "downstream_branch_contract.md",
        experiment_dir() / "downstream_branch_contract.md",
    ):
        if not path.exists():
            continue
        execution_contract = path.read_text()
        for current_text, executed_text in CURRENT_TO_EXECUTED_CONTRACT_TEXT.items():
            execution_contract = execution_contract.replace(current_text, executed_text)
        actual = hashlib.sha256(execution_contract.encode()).hexdigest()
        if actual != EXPECTED_DOWNSTREAM_CONTRACT_SHA256:
            raise ValueError(
                "executed downstream_branch_contract.md SHA mismatch: "
                f"expected {EXPECTED_DOWNSTREAM_CONTRACT_SHA256}, got {actual}"
            )
        return EXPECTED_DOWNSTREAM_CONTRACT_SHA256
    raise FileNotFoundError(
        "downstream_branch_contract.md is required in the exp298 runtime package"
    )


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
        path = KAGGLE_WORKING_ROOT / ".exp298_work"
    else:
        path = experiment_dir() / ".exp298_work"
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
# ## 4. Exp226 component reconstruction, block assignment, and target-free freeze

# %%
@dataclass
class ComponentBundle:
    component_ids: tuple[str, ...]
    values: np.memmap
    values_path: Path
    content_sha256: str
    source_path: Path
    source_decompressed_sha256: str
    source_physical_columns: tuple[str, ...]
    source_loaded_columns: tuple[str, ...]
    source_fold_crosswalk: dict[str, int]
    alias_max_abs_ft: float
    input_evidence: list[dict[str, Any]]


@dataclass
class GroupLayout:
    name: str
    codes: np.ndarray
    n_groups: int
    group_rows: np.ndarray
    group_well: np.ndarray
    group_fold: np.ndarray
    row_coordinate: np.ndarray


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
    component_manifest_path: Path
    component_manifest_file_sha256: str
    freeze_manifest_path: Path
    freeze_manifest_file_sha256: str
    candidate_content_sha256: str
    component_content_sha256: str
    block_assignment_path: Path
    block_assignment_file_sha256: str
    block_assignment_decompressed_sha256: str
    target_free_input_evidence_sha256: str
    truth_access_count_before_freeze: int


def resolve_gzip_by_decompressed_sha(
    patterns: Sequence[str], *, label: str, expected_sha256: str
) -> tuple[Path, str]:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    evidence: dict[str, str] = {}
    for path in candidates:
        actual = sha256_decompressed_gzip(path)
        evidence[str(path)] = actual
        if actual == expected_sha256:
            return path, actual
    if not candidates:
        raise FileNotFoundError(f"{label} not found from patterns: {patterns}")
    raise ValueError(f"{label} decompressed SHA mismatch: {evidence}")


def component_content_sha256(
    keys: pd.DataFrame,
    component_ids: Sequence[str],
    values: np.ndarray,
    chunk_rows: int = 100_000,
) -> str:
    digest = hashlib.sha256()
    digest.update(frame_content_sha256(keys[VALUE_KEY_COLUMNS]).encode())
    digest.update(json.dumps(list(component_ids), separators=(",", ":")).encode())
    for position, component_id in enumerate(component_ids):
        digest.update(str(component_id).encode())
        for start in range(0, len(keys), chunk_rows):
            end = min(start + chunk_rows, len(keys))
            chunk = np.asarray(values[start:end, position], dtype="<f8")
            digest.update(chunk.tobytes())
    return digest.hexdigest()


def load_exp226_components(
    bank: CandidateBank,
    config: Mapping[str, Any],
    work_dir: Path,
) -> ComponentBundle:
    source_cfg = get_nested(config, "data.exp226_oof")
    path, decompressed_sha = resolve_gzip_by_decompressed_sha(
        source_cfg["patterns"],
        label="exp226 OOF",
        expected_sha256=str(source_cfg["expected_decompressed_sha256"]),
    )
    header = tuple(str(column) for column in pd.read_csv(path, nrows=0).columns)
    required = tuple(str(value) for value in source_cfg["required_allowlisted_columns"])
    if required != EXP226_ALLOWLIST:
        raise ValueError("exp226 allowlist differs from the frozen component contract")
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f"exp226 OOF allowlisted columns missing: {sorted(missing)}")
    forbidden_tokens = tuple(
        str(value).lower()
        for value in source_cfg["forbidden_pre_freeze_column_patterns"]
    )
    if any(any(token in column.lower() for token in forbidden_tokens) for column in required):
        raise ValueError("exp226 allowlist contains a forbidden truth/readout column")

    source = pd.read_csv(path, usecols=list(required))
    if tuple(str(column) for column in source.columns) != required:
        source = source[list(required)]
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = int(get_nested(config, "validation.n_folds"))
    if len(source) != expected_rows:
        raise ValueError(f"exp226 OOF row mismatch: {len(source)} != {expected_rows}")
    if source["well_id"].astype(str).nunique() != expected_wells:
        raise ValueError("exp226 OOF well inventory mismatch")
    if set(pd.to_numeric(source["fold"], errors="raise").unique()) != set(
        range(expected_folds)
    ):
        raise ValueError("exp226 OOF fold inventory mismatch")

    source["well_id"] = source["well_id"].astype(str)
    source["row_idx"] = pd.to_numeric(source["row_idx"], errors="raise").astype(
        np.int32
    )
    source["suffix_offset"] = pd.to_numeric(
        source["suffix_offset"], errors="raise"
    ).astype(np.int32)
    source_id = source["well_id"] + "_" + source["row_idx"].astype(str)
    if source_id.duplicated().any():
        raise ValueError("exp226 OOF row identity is duplicated")
    expected_offset = source.groupby("well_id", sort=False).cumcount().to_numpy(
        dtype=np.int32
    )
    if not np.array_equal(source["suffix_offset"].to_numpy(), expected_offset):
        raise ValueError("exp226 suffix_offset is not contiguous from zero per well")

    bank_ids = bank.keys["id"].astype(str).to_numpy()
    source_ids = source_id.to_numpy()
    if np.array_equal(bank_ids, source_ids):
        indexer = np.arange(len(source), dtype=np.int64)
    else:
        indexer = pd.Index(source_ids).get_indexer(bank_ids)
    if np.any(indexer < 0):
        raise ValueError("exp293 candidate rows are missing from exp226 OOF")
    aligned = source.iloc[indexer].reset_index(drop=True)
    aligned_ids = aligned["well_id"] + "_" + aligned["row_idx"].astype(str)
    if not np.array_equal(aligned_ids.to_numpy(), bank_ids):
        raise ValueError("exp226 component identity alignment failed")
    aligned_source_folds = pd.to_numeric(
        aligned["fold"], errors="raise"
    ).to_numpy(
        dtype=np.int8
    )
    evaluation_folds = bank.keys["outer_fold"].to_numpy(dtype=np.int8)
    source_fold_crosswalk = {
        f"source_{source_fold}__evaluation_{evaluation_fold}": int(count)
        for (source_fold, evaluation_fold), count in pd.Series(
            list(zip(aligned_source_folds, evaluation_folds, strict=True))
        )
        .value_counts(sort=False)
        .sort_index()
        .items()
    }

    numeric = {
        column: pd.to_numeric(aligned[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        for column in ("tvt_geop", "gr_delta", "tvt_pred")
    }
    if any(not np.isfinite(values).all() for values in numeric.values()):
        raise ValueError("exp226 component input contains nonfinite values")
    values_path = work_dir / f"{ARTIFACT_PREFIX}_components.f64"
    values = np.memmap(
        values_path,
        mode="w+",
        dtype="float64",
        shape=(expected_rows, len(COMPONENT_IDS)),
    )
    values[:, 0] = numeric["tvt_geop"]
    values[:, 1] = numeric["tvt_geop"] + numeric["gr_delta"]
    values[:, 2] = numeric["tvt_pred"]
    values.flush()
    exp226_position = bank.candidate_ids.index("exp226_k16")
    alias_max_abs = float(
        np.max(
            np.abs(
                np.asarray(values[:, 2], dtype=np.float64)
                - np.asarray(bank.values[:, exp226_position], dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    alias_tolerance = float(
        get_nested(config, "component_contract.alias.maximum_abs_parity_ft")
    )
    if alias_max_abs > alias_tolerance:
        raise ValueError(
            f"exp226 post-U alias parity failed: {alias_max_abs} > {alias_tolerance}"
        )
    chunk_rows = int(get_nested(config, "audit.work_chunk_rows"))
    content_sha = component_content_sha256(
        bank.keys, COMPONENT_IDS, values, chunk_rows
    )
    evidence = {
        "phase": "target_free",
        "source": "exp226_oof_allowlist",
        "path": str(path),
        "rows": len(source),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": decompressed_sha,
        "logical_content_sha256": content_sha,
        "schema_sha256": json_sha256(
            [(column, str(source[column].dtype)) for column in required]
        ),
        "physical_columns": json.dumps(list(header), separators=(",", ":")),
        "loaded_columns": json.dumps(list(required), separators=(",", ":")),
        "truth_columns_materialized": False,
        "source_fold_crosswalk": json.dumps(
            source_fold_crosswalk, sort_keys=True, separators=(",", ":")
        ),
    }
    return ComponentBundle(
        component_ids=COMPONENT_IDS,
        values=values,
        values_path=values_path,
        content_sha256=content_sha,
        source_path=path,
        source_decompressed_sha256=decompressed_sha,
        source_physical_columns=header,
        source_loaded_columns=required,
        source_fold_crosswalk=source_fold_crosswalk,
        alias_max_abs_ft=alias_max_abs,
        input_evidence=[evidence],
    )


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
    within = np.arange(len(codes), dtype=np.int64) - first[codes]
    denominator = np.maximum(rows[codes] - 1, 1)
    coordinate = 2.0 * within.astype(np.float64) / denominator - 1.0
    coordinate[rows[codes] == 1] = 0.0
    return GroupLayout(
        name=name,
        codes=codes.astype(np.int32, copy=False),
        n_groups=n_groups,
        group_rows=rows,
        group_well=well_codes[first].astype(np.int32),
        group_fold=row_folds[first].astype(np.int8),
        row_coordinate=coordinate,
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
    starts = np.flatnonzero(np.r_[True, wells[1:] != wells[:-1]])
    ends = np.r_[starts[1:], len(wells)]
    segment_wells = wells[starts]
    if pd.Index(segment_wells).duplicated().any():
        raise ValueError("well rows are not contiguous in candidate bank")
    lengths = (ends - starts).astype(np.int64)
    well_codes = np.repeat(np.arange(len(starts), dtype=np.int32), lengths)
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
        "parent": get_nested(config, "lineage.parent"),
        "candidate_order": list(EXPECTED_CANDIDATE_ORDER),
        "rank_candidate_order": list(RANK_CANDIDATE_ORDER),
        "component_ids": list(COMPONENT_IDS),
        "primary_component": get_nested(config, "validation.primary_component"),
        "primary_horizons_rows": get_nested(
            config, "validation.primary_horizons_rows"
        ),
        "pass_requires_all": get_nested(config, "validation.pass_requires_all"),
        "quotient": get_nested(config, "audit.quotient"),
        "tie_policy": get_nested(config, "audit.tie_policy"),
        "execution": get_nested(config, "execution"),
        "forbidden_actions": get_nested(config, "audit.forbidden_actions"),
    }


def freeze_target_free_contract(
    bank: CandidateBank,
    components: ComponentBundle,
    assignments: BlockAssignments,
    hidden_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> FreezeEvidence:
    contract_path = artifacts_dir / f"{ARTIFACT_PREFIX}_contract.json"
    write_json(contract_path, build_contract_payload(config))
    block_path = artifacts_dir / f"{ARTIFACT_PREFIX}_block_assignment.csv.gz"
    assignments.frame.to_csv(
        block_path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    block_file_sha = sha256_file(block_path)
    block_decompressed_sha = sha256_decompressed_gzip(block_path)
    expected_block_sha = str(
        get_nested(
            config,
            "data.exp293_bank.expected_block_assignment_decompressed_sha256",
        )
    )
    if block_decompressed_sha != expected_block_sha:
        raise ValueError(
            "exp293 block assignment decompressed SHA mismatch: "
            f"{block_decompressed_sha} != {expected_block_sha}"
        )
    expected_bank_sha = str(
        get_nested(config, "data.exp293_bank.expected_content_sha256")
    )
    if bank.candidate_content_sha256 != expected_bank_sha:
        raise ValueError(
            "exp293 deployable12 content SHA mismatch: "
            f"{bank.candidate_content_sha256} != {expected_bank_sha}"
        )

    component_manifest = {
        "experiment": EXPERIMENT_NAME,
        "status": "target_free_components_frozen",
        "component_ids": list(components.component_ids),
        "component_formulas": {
            "exp226_geop": "float64(tvt_geop)",
            "exp226_pre_u": "float64(tvt_geop) + float64(gr_delta)",
            "exp226_post_u": "float64(tvt_pred)",
        },
        "primary_component": "exp226_pre_u",
        "rows": len(bank.keys),
        "component_content_sha256": components.content_sha256,
        "source_decompressed_sha256": components.source_decompressed_sha256,
        "source_physical_columns": list(components.source_physical_columns),
        "source_loaded_columns": list(components.source_loaded_columns),
        "source_fold_crosswalk": components.source_fold_crosswalk,
        "evaluation_fold_source": "exp293_exp263_outer_fold",
        "truth_or_error_columns_materialized": False,
        "post_u_alias": "exp226_k16",
        "post_u_alias_max_abs_ft": components.alias_max_abs_ft,
        "oracle_coefficients_persisted": False,
        "corrected_prediction_persisted": False,
    }
    component_manifest_path = (
        artifacts_dir / f"{ARTIFACT_PREFIX}_component_manifest.json"
    )
    write_json(component_manifest_path, component_manifest)

    target_free_evidence = [
        *bank.input_evidence,
        *components.input_evidence,
        dict(hidden_evidence),
    ]
    evidence_sha = json_sha256(target_free_evidence)
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "status": "pre_truth_freeze_complete",
        "frozen_at": datetime.now(UTC).isoformat(),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "folds": sorted(int(value) for value in bank.keys["outer_fold"].unique()),
        "candidate_ids": list(bank.candidate_ids),
        "candidate_bank_content_sha256": bank.candidate_content_sha256,
        "component_ids": list(components.component_ids),
        "component_content_sha256": components.content_sha256,
        "block_assignment_file_sha256": block_file_sha,
        "block_assignment_decompressed_sha256": block_decompressed_sha,
        "block_assignment_logical_sha256": frame_content_sha256(assignments.frame),
        "contract_file_sha256": sha256_file(contract_path),
        "component_manifest_file_sha256": sha256_file(component_manifest_path),
        "config_file_sha256": sha256_file(find_config_path()),
        "downstream_branch_contract_file_sha256": downstream_contract_sha256(),
        "target_free_input_evidence_sha256": evidence_sha,
        "truth_access_count_before_freeze": 0,
        "truth_columns_loaded_before_freeze": [],
        "frozen": True,
    }
    freeze_manifest_path = artifacts_dir / f"{ARTIFACT_PREFIX}_freeze_manifest.json"
    write_json(freeze_manifest_path, freeze_manifest)
    return FreezeEvidence(
        contract_path=contract_path,
        contract_file_sha256=sha256_file(contract_path),
        component_manifest_path=component_manifest_path,
        component_manifest_file_sha256=sha256_file(component_manifest_path),
        freeze_manifest_path=freeze_manifest_path,
        freeze_manifest_file_sha256=sha256_file(freeze_manifest_path),
        candidate_content_sha256=bank.candidate_content_sha256,
        component_content_sha256=components.content_sha256,
        block_assignment_path=block_path,
        block_assignment_file_sha256=block_file_sha,
        block_assignment_decompressed_sha256=block_decompressed_sha,
        target_free_input_evidence_sha256=evidence_sha,
        truth_access_count_before_freeze=0,
    )


def verify_freeze_before_truth(
    bank: CandidateBank,
    components: ComponentBundle,
    freeze: FreezeEvidence,
    chunk_rows: int,
) -> None:
    if freeze.truth_access_count_before_freeze != 0:
        raise ValueError("truth was accessed before target-free freeze")
    path_hashes = {
        freeze.contract_path: freeze.contract_file_sha256,
        freeze.component_manifest_path: freeze.component_manifest_file_sha256,
        freeze.freeze_manifest_path: freeze.freeze_manifest_file_sha256,
        freeze.block_assignment_path: freeze.block_assignment_file_sha256,
    }
    for path, expected in path_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen artifact changed before truth load: {path}")
    if sha256_decompressed_gzip(freeze.block_assignment_path) != (
        freeze.block_assignment_decompressed_sha256
    ):
        raise ValueError("block assignment decompressed content changed")
    if candidate_bank_content_sha256(bank, chunk_rows) != (
        freeze.candidate_content_sha256
    ):
        raise ValueError("candidate bank changed after target-free freeze")
    if component_content_sha256(
        bank.keys, components.component_ids, components.values, chunk_rows
    ) != freeze.component_content_sha256:
        raise ValueError("exp226 component values changed after target-free freeze")


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
        wells = {path.name.split("__horizontal_well.csv", 1)[0] for path in files}
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
    components: ComponentBundle,
    freeze: FreezeEvidence,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    chunk_rows = int(get_nested(config, "audit.work_chunk_rows"))
    verify_freeze_before_truth(bank, components, freeze, chunk_rows)
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
        raise ValueError(f"raw truth row mismatch: {len(truth_frame)} != {len(bank.keys)}")
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
# ## 6. Blockwise quotient aggregation and diagnostic readouts

# %%
@dataclass
class QuotientStats:
    rows: np.ndarray
    offset_sse: np.ndarray
    affine_sse: np.ndarray
    offset_valid: np.ndarray
    affine_valid: np.ndarray


def aggregate_quotient(
    error: np.ndarray,
    layout: GroupLayout,
    row_mask: np.ndarray | None = None,
) -> QuotientStats:
    if len(error) != len(layout.codes) or not np.isfinite(error).all():
        raise ValueError("quotient input must be finite and aligned to block rows")
    if row_mask is None:
        weight = np.ones(len(error), dtype=np.float64)
    else:
        if len(row_mask) != len(error):
            raise ValueError("quotient row mask is misaligned")
        weight = np.asarray(row_mask, dtype=np.float64)
    codes = layout.codes
    x = layout.row_coordinate
    rows = np.bincount(codes, weights=weight, minlength=layout.n_groups)
    sum_x = np.bincount(codes, weights=weight * x, minlength=layout.n_groups)
    sum_xx = np.bincount(codes, weights=weight * x * x, minlength=layout.n_groups)
    sum_e = np.bincount(codes, weights=weight * error, minlength=layout.n_groups)
    sum_ex = np.bincount(
        codes, weights=weight * error * x, minlength=layout.n_groups
    )
    sum_ee = np.bincount(
        codes, weights=weight * error * error, minlength=layout.n_groups
    )
    offset_valid = rows > 0
    offset_sse = np.full(layout.n_groups, np.nan, dtype=np.float64)
    offset_sse[offset_valid] = (
        sum_ee[offset_valid]
        - np.square(sum_e[offset_valid]) / rows[offset_valid]
    )
    determinant = rows * sum_xx - np.square(sum_x)
    affine_valid = (rows >= 2) & (determinant > 1e-12)
    affine_sse = np.full(layout.n_groups, np.nan, dtype=np.float64)
    selected = affine_valid
    explained = np.zeros(layout.n_groups, dtype=np.float64)
    explained[selected] = (
        sum_xx[selected] * np.square(sum_e[selected])
        - 2.0 * sum_x[selected] * sum_e[selected] * sum_ex[selected]
        + rows[selected] * np.square(sum_ex[selected])
    ) / determinant[selected]
    affine_sse[selected] = sum_ee[selected] - explained[selected]
    offset_sse[offset_valid] = np.maximum(offset_sse[offset_valid], 0.0)
    affine_sse[affine_valid] = np.maximum(affine_sse[affine_valid], 0.0)
    return QuotientStats(
        rows=rows,
        offset_sse=offset_sse,
        affine_sse=affine_sse,
        offset_valid=offset_valid,
        affine_valid=affine_valid,
    )


def difference_group_metrics(
    error: np.ndarray, layout: GroupLayout
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_valid = layout.codes[1:] == layout.codes[:-1]
    first_codes = layout.codes[1:][first_valid]
    first_error = np.diff(error)[first_valid]
    first_sse = np.bincount(
        first_codes,
        weights=np.square(first_error),
        minlength=layout.n_groups,
    )
    first_rows = np.bincount(first_codes, minlength=layout.n_groups).astype(np.int64)
    if len(error) < 3:
        second_valid = np.zeros(0, dtype=bool)
    else:
        second_valid = (layout.codes[2:] == layout.codes[1:-1]) & (
            layout.codes[1:-1] == layout.codes[:-2]
        )
    second_codes = layout.codes[2:][second_valid]
    second_error = (error[2:] - 2.0 * error[1:-1] + error[:-2])[second_valid]
    second_sse = np.bincount(
        second_codes,
        weights=np.square(second_error),
        minlength=layout.n_groups,
    )
    second_rows = np.bincount(second_codes, minlength=layout.n_groups).astype(
        np.int64
    )
    return first_sse, first_rows, second_sse, second_rows


def _metric_record(
    candidate_id: str,
    horizon: str,
    scope: str,
    stats: QuotientStats,
    group_mask: np.ndarray,
    first_sse: np.ndarray | None = None,
    first_rows: np.ndarray | None = None,
    second_sse: np.ndarray | None = None,
    second_rows: np.ndarray | None = None,
    group_well: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_groups = np.asarray(group_mask, dtype=bool)
    total_rows = float(stats.rows[selected_groups].sum())
    offset_groups = selected_groups & stats.offset_valid
    affine_eligible_groups = selected_groups & (stats.rows >= 2)
    affine_groups = selected_groups & stats.affine_valid
    singleton_groups = selected_groups & (stats.rows == 1)
    offset_rows = float(stats.rows[offset_groups].sum())
    affine_eligible_rows = float(stats.rows[affine_eligible_groups].sum())
    affine_rows = float(stats.rows[affine_groups].sum())
    singleton_rows = float(stats.rows[singleton_groups].sum())
    offset_sse_total = float(np.nansum(stats.offset_sse[offset_groups]))
    affine_sse_total = float(np.nansum(stats.affine_sse[affine_groups]))
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "horizon": horizon,
        "scope": scope,
        "rows": int(total_rows),
        "blocks": int(selected_groups.sum()),
        "offset_valid_blocks": int(offset_groups.sum()),
        "affine_eligible_blocks": int(affine_eligible_groups.sum()),
        "affine_valid_blocks": int(affine_groups.sum()),
        "affine_invalid_eligible_blocks": int(
            (affine_eligible_groups & ~stats.affine_valid).sum()
        ),
        "affine_excluded_singleton_blocks": int(singleton_groups.sum()),
        "affine_excluded_singleton_rows": int(singleton_rows),
        "affine_excluded_singleton_wells": (
            int(np.unique(group_well[singleton_groups]).size)
            if group_well is not None
            else None
        ),
        "offset_valid_rows": int(offset_rows),
        "affine_eligible_rows": int(affine_eligible_rows),
        "affine_valid_rows": int(affine_rows),
        "offset_valid_row_fraction": offset_rows / total_rows if total_rows else 0.0,
        "affine_valid_row_fraction": affine_rows / total_rows if total_rows else 0.0,
        "affine_valid_eligible_row_fraction": (
            affine_rows / affine_eligible_rows if affine_eligible_rows else 1.0
        ),
        "offset_quotient_sse": offset_sse_total,
        "affine_quotient_sse": affine_sse_total,
        "offset_quotient_rmse": (
            math.sqrt(offset_sse_total / offset_rows) if offset_rows else math.nan
        ),
        "affine_quotient_rmse": (
            math.sqrt(affine_sse_total / affine_rows) if affine_rows else math.nan
        ),
        "rank_eligible": candidate_id in RANK_CANDIDATE_ORDER,
    }
    if first_sse is not None and first_rows is not None:
        count = int(first_rows[selected_groups].sum())
        sse = float(first_sse[selected_groups].sum())
        record["first_difference_rows"] = count
        record["first_difference_error_rmse"] = (
            math.sqrt(sse / count) if count else math.nan
        )
    if second_sse is not None and second_rows is not None:
        count = int(second_rows[selected_groups].sum())
        sse = float(second_sse[selected_groups].sum())
        record["second_difference_rows"] = count
        record["second_difference_error_rmse"] = (
            math.sqrt(sse / count) if count else math.nan
        )
    return record


def _candidate_values(
    candidate_id: str,
    bank: CandidateBank,
    components: ComponentBundle,
) -> np.ndarray:
    if candidate_id in bank.candidate_ids:
        position = bank.candidate_ids.index(candidate_id)
        return np.asarray(bank.values[:, position], dtype=np.float64)
    position = components.component_ids.index(candidate_id)
    return np.asarray(components.values[:, position], dtype=np.float64)


def _add_stable_ranks(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    output = frame.copy()
    output["affine_quotient_rank"] = np.nan
    order = {candidate: position for position, candidate in enumerate(RANK_CANDIDATE_ORDER)}
    grouped = output.groupby(list(group_columns), sort=False, dropna=False).groups
    for indices in grouped.values():
        eligible = [
            index
            for index in indices
            if output.at[index, "candidate_id"] in order
            and np.isfinite(output.at[index, "affine_quotient_rmse"])
        ]
        eligible.sort(
            key=lambda index: (
                float(output.at[index, "affine_quotient_rmse"]),
                order[str(output.at[index, "candidate_id"])],
            )
        )
        for rank, index in enumerate(eligible, start=1):
            output.at[index, "affine_quotient_rank"] = rank
    return output


def build_quotient_readouts(
    bank: CandidateBank,
    components: ComponentBundle,
    assignments: BlockAssignments,
    truth: np.ndarray,
    hidden_sets: Mapping[str, set[str]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(truth) != len(bank.keys) or not np.isfinite(truth).all():
        raise ValueError("truth must be finite and aligned to candidate rows")
    md_since = bank.keys["md_since"].to_numpy(dtype=np.float64)
    longtail_mask = md_since >= 1000.0
    hidden_group_masks: dict[tuple[str, str], np.ndarray] = {}
    for scope, selected_wells in hidden_sets.items():
        selected_by_well = np.isin(
            assignments.well_names,
            np.asarray(sorted(selected_wells), dtype=object),
        )
        for horizon, layout in assignments.layouts.items():
            hidden_group_masks[(scope, horizon)] = selected_by_well[layout.group_well]

    pooled_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    scope_records: list[dict[str, Any]] = []
    by_well_records: list[dict[str, Any]] = []
    block_parts: list[pd.DataFrame] = []
    stats_cache: dict[tuple[str, str], QuotientStats] = {}
    difference_cache: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    for candidate_id in READOUT_CANDIDATE_ORDER:
        prediction = _candidate_values(candidate_id, bank, components)
        if not np.isfinite(prediction).all():
            raise ValueError(f"nonfinite readout candidate: {candidate_id}")
        error = truth - prediction
        for horizon, layout in assignments.layouts.items():
            stats = aggregate_quotient(error, layout)
            differences = difference_group_metrics(error, layout)
            stats_cache[(candidate_id, horizon)] = stats
            difference_cache[(candidate_id, horizon)] = differences
            all_groups = np.ones(layout.n_groups, dtype=bool)
            pooled_records.append(
                _metric_record(
                    candidate_id,
                    horizon,
                    "overall",
                    stats,
                    all_groups,
                    *differences,
                    group_well=layout.group_well,
                )
            )
            for fold in range(int(get_nested(config, "validation.n_folds"))):
                fold_records.append(
                    {
                        **_metric_record(
                            candidate_id,
                            horizon,
                            f"fold_{fold}",
                            stats,
                            layout.group_fold == fold,
                            group_well=layout.group_well,
                        ),
                        "fold": fold,
                    }
                )
            longtail_stats = aggregate_quotient(error, layout, longtail_mask)
            scope_records.append(
                _metric_record(
                    candidate_id,
                    horizon,
                    "1000_plus",
                    longtail_stats,
                    longtail_stats.rows > 0,
                    group_well=layout.group_well,
                )
            )
            for scope in hidden_sets:
                scope_records.append(
                    _metric_record(
                        candidate_id,
                        horizon,
                        scope,
                        stats,
                        hidden_group_masks[(scope, horizon)],
                        group_well=layout.group_well,
                    )
                )

            valid_sse = np.where(stats.affine_valid, stats.affine_sse, 0.0)
            valid_rows = np.where(stats.affine_valid, stats.rows, 0.0)
            eligible_rows = np.where(stats.rows >= 2, stats.rows, 0.0)
            well_sse = np.bincount(
                layout.group_well,
                weights=valid_sse,
                minlength=len(assignments.well_names),
            )
            well_valid_rows = np.bincount(
                layout.group_well,
                weights=valid_rows,
                minlength=len(assignments.well_names),
            )
            well_eligible_rows = np.bincount(
                layout.group_well,
                weights=eligible_rows,
                minlength=len(assignments.well_names),
            )
            well_total_rows = np.bincount(
                layout.group_well,
                weights=stats.rows,
                minlength=len(assignments.well_names),
            )
            for well_position, well in enumerate(assignments.well_names):
                rows = float(well_valid_rows[well_position])
                eligible = float(well_eligible_rows[well_position])
                total = float(well_total_rows[well_position])
                by_well_records.append(
                    {
                        "well": str(well),
                        "fold": int(assignments.well_fold[well_position]),
                        "candidate_id": candidate_id,
                        "horizon": horizon,
                        "rows": int(total),
                        "affine_eligible_rows": int(eligible),
                        "affine_valid_rows": int(rows),
                        "affine_valid_row_fraction": rows / total if total else 0.0,
                        "affine_valid_eligible_row_fraction": (
                            rows / eligible if eligible else 1.0
                        ),
                        "affine_excluded_singleton_rows": int(total - eligible),
                        "affine_quotient_rmse": (
                            math.sqrt(float(well_sse[well_position]) / rows)
                            if rows
                            else math.nan
                        ),
                    }
                )

    tie_atol = float(get_nested(config, "audit.tie_policy.squared_error_atol"))
    win_fraction: dict[tuple[str, str], float] = {}
    unique_fraction: dict[tuple[str, str], float] = {}
    for horizon, layout in assignments.layouts.items():
        matrix = np.column_stack(
            [stats_cache[(candidate, horizon)].affine_sse for candidate in RANK_CANDIDATE_ORDER]
        )
        valid = np.isfinite(matrix).all(axis=1)
        minimum = np.full(layout.n_groups, np.nan, dtype=np.float64)
        minimum[valid] = np.min(matrix[valid], axis=1)
        tied = np.zeros_like(matrix, dtype=bool)
        tied[valid] = matrix[valid] <= minimum[valid, None] + tie_atol
        tie_count = tied.sum(axis=1)
        for position, candidate_id in enumerate(RANK_CANDIDATE_ORDER):
            stats = stats_cache[(candidate_id, horizon)]
            first_sse, first_rows, second_sse, second_rows = difference_cache[
                (candidate_id, horizon)
            ]
            is_best = valid & tied[:, position]
            is_unique = is_best & (tie_count == 1)
            denominator = int(valid.sum())
            win_fraction[(candidate_id, horizon)] = (
                float(is_best.sum()) / denominator if denominator else 0.0
            )
            unique_fraction[(candidate_id, horizon)] = (
                float(is_unique.sum()) / denominator if denominator else 0.0
            )
            offset_rmse = np.full(layout.n_groups, np.nan, dtype=np.float64)
            affine_rmse = np.full(layout.n_groups, np.nan, dtype=np.float64)
            offset_rmse[stats.offset_valid] = np.sqrt(
                stats.offset_sse[stats.offset_valid] / stats.rows[stats.offset_valid]
            )
            affine_rmse[stats.affine_valid] = np.sqrt(
                stats.affine_sse[stats.affine_valid] / stats.rows[stats.affine_valid]
            )
            first_rmse = np.full(layout.n_groups, np.nan, dtype=np.float64)
            second_rmse = np.full(layout.n_groups, np.nan, dtype=np.float64)
            first_valid = first_rows > 0
            second_valid = second_rows > 0
            first_rmse[first_valid] = np.sqrt(first_sse[first_valid] / first_rows[first_valid])
            second_rmse[second_valid] = np.sqrt(
                second_sse[second_valid] / second_rows[second_valid]
            )
            block_parts.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidate_id,
                        "horizon": horizon,
                        "block_id": np.arange(layout.n_groups, dtype=np.int32),
                        "well": assignments.well_names[layout.group_well],
                        "fold": layout.group_fold,
                        "rows": stats.rows.astype(np.int64),
                        "affine_eligible": stats.rows >= 2,
                        "excluded_singleton": stats.rows == 1,
                        "offset_quotient_rmse": offset_rmse,
                        "affine_quotient_rmse": affine_rmse,
                        "first_difference_error_rmse": first_rmse,
                        "second_difference_error_rmse": second_rmse,
                        "is_best": is_best,
                        "is_unique_best": is_unique,
                    }
                )
            )

    pooled = _add_stable_ranks(pd.DataFrame(pooled_records), ["horizon", "scope"])
    folds = _add_stable_ranks(
        pd.DataFrame(fold_records), ["horizon", "scope", "fold"]
    )
    scopes = _add_stable_ranks(pd.DataFrame(scope_records), ["horizon", "scope"])
    pooled["block_win_fraction"] = [
        win_fraction.get((row.candidate_id, row.horizon), math.nan)
        for row in pooled.itertuples(index=False)
    ]
    pooled["strict_unique_best_block_fraction"] = [
        unique_fraction.get((row.candidate_id, row.horizon), math.nan)
        for row in pooled.itertuples(index=False)
    ]
    return (
        pooled,
        folds,
        scopes,
        pd.concat(block_parts, ignore_index=True),
        pd.DataFrame(by_well_records),
    )


# %% [markdown]
# ## 7. PASS/FAIL decision and generated artifacts

# %%
def _one_metric(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    horizon: str,
    scope: str,
    fold: int | None = None,
) -> pd.Series:
    selected = frame[
        frame["candidate_id"].eq(candidate_id)
        & frame["horizon"].eq(horizon)
        & frame["scope"].eq(scope)
    ]
    if fold is not None:
        selected = selected[selected["fold"].eq(fold)]
    if len(selected) != 1:
        raise ValueError(
            f"metric row is not unique: {candidate_id}/{horizon}/{scope}/{fold}"
        )
    return selected.iloc[0]


def evaluate_audit_decision(
    bank: CandidateBank,
    components: ComponentBundle,
    assignments: BlockAssignments,
    freeze: FreezeEvidence,
    pooled: pd.DataFrame,
    folds: pd.DataFrame,
    scopes: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = int(get_nested(config, "validation.n_folds"))
    expected_bank_sha = str(
        get_nested(config, "data.exp293_bank.expected_content_sha256")
    )
    expected_block_sha = str(
        get_nested(
            config,
            "data.exp293_bank.expected_block_assignment_decompressed_sha256",
        )
    )
    primary_rows = pooled[
        pooled["candidate_id"].isin(READOUT_CANDIDATE_ORDER)
        & pooled["horizon"].isin(["h128", "h256", "h512", "whole_well"])
    ]
    loaded_forbidden = {
        column
        for column in components.source_loaded_columns
        if any(token in column.lower() for token in ("true", "target", "error", "oracle"))
    }
    technical_checks = {
        "row_count_exact": len(bank.keys) == expected_rows,
        "well_count_exact": bank.keys["well"].nunique() == expected_wells,
        "fold_inventory_exact": set(bank.keys["outer_fold"].unique())
        == set(range(expected_folds)),
        "candidate_order_exact": bank.candidate_ids == EXPECTED_CANDIDATE_ORDER,
        "candidate_bank_content_sha_exact": bank.candidate_content_sha256
        == expected_bank_sha,
        "block_assignment_decompressed_sha_exact": (
            freeze.block_assignment_decompressed_sha256 == expected_block_sha
        ),
        "duplicate_id_zero": not bank.keys["id"].duplicated().any(),
        "candidate_finite_coverage_one": all(
            math.isclose(value, 1.0) for value in bank.coverage_by_candidate.values()
        ),
        "component_finite_coverage_one": bool(np.isfinite(components.values).all()),
        "exp226_allowlist_exact": components.source_loaded_columns == EXP226_ALLOWLIST,
        "truth_or_error_column_loaded_before_freeze_zero": not loaded_forbidden,
        "truth_access_before_freeze_zero": freeze.truth_access_count_before_freeze == 0,
        "post_u_alias_parity": components.alias_max_abs_ft
        <= float(get_nested(config, "component_contract.alias.maximum_abs_parity_ft")),
        "offset_quotient_row_coverage_one": bool(
            np.isclose(primary_rows["offset_valid_row_fraction"], 1.0).all()
        ),
        "affine_eligible_row_coverage_one": bool(
            np.isclose(
                primary_rows["affine_valid_eligible_row_fraction"], 1.0
            ).all()
        ),
        "affine_invalid_eligible_blocks_zero": bool(
            primary_rows["affine_invalid_eligible_blocks"].eq(0).all()
        ),
        "singleton_exclusion_candidate_independent": bool(
            primary_rows.groupby("horizon")[
                [
                    "affine_excluded_singleton_blocks",
                    "affine_excluded_singleton_rows",
                    "affine_excluded_singleton_wells",
                ]
            ]
            .nunique(dropna=False)
            .eq(1)
            .all()
            .all()
        ),
        "oracle_coefficients_persisted_false": True,
        "corrected_prediction_persisted_false": True,
    }
    technical_passed = bool(all(technical_checks.values()))

    pass_cfg = get_nested(config, "validation.pass_requires_all")
    pre_h256 = _one_metric(
        pooled,
        candidate_id="exp226_pre_u",
        horizon="h256",
        scope="overall",
    )
    pre_h512 = _one_metric(
        pooled,
        candidate_id="exp226_pre_u",
        horizon="h512",
        scope="overall",
    )
    post_h256 = _one_metric(
        pooled,
        candidate_id="exp226_post_u",
        horizon="h256",
        scope="overall",
    )
    post_h512 = _one_metric(
        pooled,
        candidate_id="exp226_post_u",
        horizon="h512",
        scope="overall",
    )
    maximum_rank_h256 = int(
        pass_cfg["maximum_pooled_affine_quotient_rank_h256"]
    )
    maximum_rank_h512 = int(
        pass_cfg["maximum_pooled_affine_quotient_rank_h512"]
    )
    minimum_fold_count = int(
        pass_cfg["minimum_folds_with_rank_at_most3_each_primary_horizon"]
    )
    fold_rank_counts: dict[str, int] = {}
    for horizon in ("h256", "h512"):
        selected = folds[
            folds["candidate_id"].eq("exp226_pre_u")
            & folds["horizon"].eq(horizon)
        ]
        if len(selected) != expected_folds:
            raise ValueError(f"{horizon} fold metric inventory mismatch")
        fold_rank_counts[horizon] = int(
            (selected["affine_quotient_rank"] <= 3).sum()
        )
    risk_ranks = {
        scope: int(
            _one_metric(
                scopes,
                candidate_id="exp226_pre_u",
                horizon="h512",
                scope=scope,
            )["affine_quotient_rank"]
        )
        for scope in (
            "1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        )
    }
    unique_h256 = float(pre_h256["strict_unique_best_block_fraction"])
    unique_h512 = float(pre_h512["strict_unique_best_block_fraction"])
    scientific_checks = {
        "pooled_h256_rank_at_most_3": int(pre_h256["affine_quotient_rank"])
        <= maximum_rank_h256,
        "pooled_h512_rank_at_most_3": int(pre_h512["affine_quotient_rank"])
        <= maximum_rank_h512,
        "rank1_at_one_primary_horizon": min(
            int(pre_h256["affine_quotient_rank"]),
            int(pre_h512["affine_quotient_rank"]),
        )
        == 1,
        "h256_four_of_five_folds_rank_at_most_3": fold_rank_counts["h256"]
        >= minimum_fold_count,
        "h512_four_of_five_folds_rank_at_most_3": fold_rank_counts["h512"]
        >= minimum_fold_count,
        "pre_u_nonworse_than_post_u_h256": float(
            pre_h256["affine_quotient_rmse"]
        )
        <= float(post_h256["affine_quotient_rmse"]),
        "pre_u_nonworse_than_post_u_h512": float(
            pre_h512["affine_quotient_rmse"]
        )
        <= float(post_h512["affine_quotient_rmse"]),
        "h512_1000_plus_rank_at_most_3": risk_ranks["1000_plus"]
        <= int(pass_cfg["maximum_h512_rank_distance_1000_plus"]),
        "h512_hidden_like_spatial_rank_at_most_3": risk_ranks[
            "hidden_like_spatial"
        ]
        <= int(pass_cfg["maximum_h512_rank_hidden_like_spatial"]),
        "h512_hidden_like_typewell_purged_rank_at_most_3": risk_ranks[
            "hidden_like_typewell_purged"
        ]
        <= int(pass_cfg["maximum_h512_rank_hidden_like_typewell_purged"]),
        "unique_best_block_fraction_at_least_0p05_one_primary_horizon": max(
            unique_h256, unique_h512
        )
        >= float(pass_cfg["minimum_unique_best_block_fraction_at_one_primary_horizon"]),
    }
    scientific_passed = bool(all(scientific_checks.values()))
    audit_passed = bool(technical_passed and scientific_passed)
    return {
        "technical_checks": technical_checks,
        "technical_passed": technical_passed,
        "scientific_checks": scientific_checks,
        "scientific_passed": scientific_passed,
        "audit_passed": audit_passed,
        "decision": "pass_enable_stage2_only" if audit_passed else "fail_branch_closed",
        "next_branch": (
            str(get_nested(config, "downstream.stage2_if_exp298_pass"))
            if audit_passed
            else "branch_closed_without_rescue_grid"
        ),
        "primary": {
            "h256": {
                "affine_quotient_rmse": float(pre_h256["affine_quotient_rmse"]),
                "rank": int(pre_h256["affine_quotient_rank"]),
                "post_u_rmse": float(post_h256["affine_quotient_rmse"]),
                "strict_unique_best_block_fraction": unique_h256,
            },
            "h512": {
                "affine_quotient_rmse": float(pre_h512["affine_quotient_rmse"]),
                "rank": int(pre_h512["affine_quotient_rank"]),
                "post_u_rmse": float(post_h512["affine_quotient_rmse"]),
                "strict_unique_best_block_fraction": unique_h512,
            },
        },
        "fold_rank_at_most3_counts": fold_rank_counts,
        "h512_risk_ranks": risk_ranks,
    }


def validate_execution_contract(
    config: Mapping[str, Any], *, require_kaggle_approval: bool = True
) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp298 route must remain pf_beam")
    execution = get_nested(config, "execution")
    expected_zero = [
        "lightgbm_config_count",
        "trained_fold_count",
        "total_boosters",
        "hmm_pf_well_runs",
    ]
    if any(int(execution[key]) != 0 for key in expected_zero):
        raise ValueError("exp298 must remain a zero-model, zero-PF-regeneration audit")
    if int(execution["active_audit_contracts"]) != 1:
        raise ValueError("exp298 must contain exactly one fixed audit contract")
    if int(execution["evaluation_fold_count"]) != 5:
        raise ValueError("exp298 must preserve the five-fold evaluation contract")
    if execution["inference"] or execution["submission"]:
        raise ValueError("exp298 inference and submission must remain disabled")
    if not bool(execution["implementation"]):
        raise ValueError("exp298 implementation flag is not enabled")
    if require_kaggle_approval:
        if not bool(execution["canonical_train_notebook_adopted"]):
            raise ValueError("exp298 compact train candidate is not canonically adopted")
        if not bool(execution["kaggle_push_approved"]):
            raise ValueError("exp298 Kaggle push is not approved")
        if not bool(execution["kaggle_execution_authorized"]):
            raise ValueError("exp298 Kaggle execution is not authorized")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def persist_audit_outputs(
    bank: CandidateBank,
    components: ComponentBundle,
    freeze: FreezeEvidence,
    hidden_evidence: Mapping[str, Any],
    truth_evidence: Sequence[Mapping[str, Any]],
    truth_content_sha256: str,
    pooled: pd.DataFrame,
    folds: pd.DataFrame,
    scopes: pd.DataFrame,
    blocks: pd.DataFrame,
    by_well: pd.DataFrame,
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    input_path = artifacts_dir / f"{ARTIFACT_PREFIX}_input_manifest.csv"
    pooled_path = artifacts_dir / f"{ARTIFACT_PREFIX}_pooled_metrics.csv"
    fold_path = artifacts_dir / f"{ARTIFACT_PREFIX}_fold_metrics.csv"
    scope_path = artifacts_dir / f"{ARTIFACT_PREFIX}_scope_metrics.csv"
    block_path = artifacts_dir / f"{ARTIFACT_PREFIX}_block_metrics.csv.gz"
    by_well_path = artifacts_dir / f"{ARTIFACT_PREFIX}_by_well.csv"
    summary_path = artifacts_dir / f"{ARTIFACT_PREFIX}_summary.json"
    sha_path = artifacts_dir / f"{ARTIFACT_PREFIX}_sha_manifest.csv"
    aligned_truth = {
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
            *components.input_evidence,
            dict(hidden_evidence),
            *[dict(item) for item in truth_evidence],
            aligned_truth,
        ]
    )
    _write_frame(input_path, input_manifest)
    _write_frame(pooled_path, pooled)
    _write_frame(fold_path, folds)
    _write_frame(scope_path, scopes)
    blocks.to_csv(
        block_path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    _write_frame(by_well_path, by_well)
    readout_content_sha = json_sha256(
        {
            "pooled": frame_content_sha256(pooled),
            "folds": frame_content_sha256(folds),
            "scopes": frame_content_sha256(scopes),
            "blocks": frame_content_sha256(blocks),
            "by_well": frame_content_sha256(by_well),
        }
    )
    pre_h512_wells = by_well[
        by_well["candidate_id"].eq("exp226_pre_u")
        & by_well["horizon"].eq("h512")
    ].copy()
    reference_blocks = blocks[
        blocks["candidate_id"].eq(RANK_CANDIDATE_ORDER[0])
    ]
    singleton_summary: dict[str, dict[str, int]] = {}
    for horizon, horizon_blocks in reference_blocks.groupby("horizon", sort=False):
        singleton = horizon_blocks[horizon_blocks["excluded_singleton"]]
        singleton_summary[str(horizon)] = {
            "blocks": int(len(singleton)),
            "rows": int(singleton["rows"].sum()),
            "wells": int(singleton["well"].nunique()),
        }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "audit_completed",
        "route": "pf_beam",
        "completed_at": datetime.now(UTC).isoformat(),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "rank_candidate_count": len(RANK_CANDIDATE_ORDER),
        "diagnostic_component_count": len(COMPONENT_IDS),
        "candidate_bank_content_sha256": bank.candidate_content_sha256,
        "component_content_sha256": components.content_sha256,
        "block_assignment_decompressed_sha256": (
            freeze.block_assignment_decompressed_sha256
        ),
        "truth_content_sha256": truth_content_sha256,
        "readout_content_sha256": readout_content_sha,
        "affine_singleton_policy": {
            "minimum_selected_rows": 2,
            "excluded_from_affine_metric_rank_win_unique_best": True,
            "technical_coverage_denominator": "affine_eligible_rows_only",
            "counts": singleton_summary,
        },
        "decision": dict(decision),
        "by_well_h512_pre_u": {
            "p50_rmse": float(pre_h512_wells["affine_quotient_rmse"].quantile(0.5)),
            "p95_rmse": float(pre_h512_wells["affine_quotient_rmse"].quantile(0.95)),
            "worst_rmse": float(pre_h512_wells["affine_quotient_rmse"].max()),
            "worst_well": str(
                pre_h512_wells.loc[
                    pre_h512_wells["affine_quotient_rmse"].idxmax(), "well"
                ]
            ),
        },
        "execution": {
            "active_audit_contracts": 1,
            "lightgbm_config_count": 0,
            "evaluation_fold_count": 5,
            "trained_fold_count": 0,
            "total_boosters": 0,
            "hmm_pf_well_runs": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
        "oracle_coefficients_persisted": False,
        "corrected_prediction_persisted": False,
        "selected_prediction_persisted": False,
    }
    write_json(summary_path, summary)
    artifact_paths = [
        freeze.contract_path,
        freeze.component_manifest_path,
        freeze.freeze_manifest_path,
        freeze.block_assignment_path,
        input_path,
        pooled_path,
        fold_path,
        scope_path,
        block_path,
        by_well_path,
        summary_path,
    ]
    sha_records = [
        {
            "artifact": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "decompressed_content_sha256": (
                sha256_decompressed_gzip(path) if path.suffix == ".gz" else None
            ),
        }
        for path in artifact_paths
    ]
    _write_frame(sha_path, pd.DataFrame(sha_records))
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "implementation_complete_audit_executed",
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "audit_passed": bool(decision["audit_passed"]),
        "technical_passed": bool(decision["technical_passed"]),
        "scientific_passed": bool(decision["scientific_passed"]),
        "next_branch": str(decision["next_branch"]),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "primary": decision["primary"],
        "active_audit_contracts": 1,
        "lightgbm_configs": 0,
        "evaluation_folds": 5,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_pf_well_runs": 0,
        "readout_content_sha256": readout_content_sha,
        "sha_manifest_file_sha256": sha256_file(sha_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config, require_kaggle_approval=True)
    artifacts_dir = runtime_artifacts_dir()
    work_dir = runtime_work_dir()
    horizons = [
        int(value)
        for value in get_nested(config, "audit.block_partition.horizons_rows")
    ]
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Primary component:", get_nested(config, "validation.primary_component"))
    print("Rank candidate order:", RANK_CANDIDATE_ORDER)
    print("Block horizons:", horizons)
    print(
        "Execution contract: 1 audit / 0 model configs / 5 evaluation folds / "
        "0 trained folds / 0 boosters / 0 PF-Beam reruns"
    )
    bank = build_candidate_bank(config, work_dir)
    components = load_exp226_components(bank, config, work_dir)
    assignments = build_block_assignments(bank.keys, horizons)
    hidden_sets, hidden_evidence = load_hidden_like_sets(
        config, set(bank.keys["well"].astype(str))
    )
    freeze = freeze_target_free_contract(
        bank,
        components,
        assignments,
        hidden_evidence,
        config,
        artifacts_dir,
    )
    print("Target-free candidate bank frozen:", bank.candidate_content_sha256)
    print("Target-free component bundle frozen:", components.content_sha256)
    print("Truth access count before freeze:", freeze.truth_access_count_before_freeze)
    truth, truth_evidence, truth_sha = load_truth_after_freeze(
        bank, components, freeze, config
    )
    print("Post-freeze truth attached:", truth_sha)
    pooled, folds, scopes, blocks, by_well = build_quotient_readouts(
        bank, components, assignments, truth, hidden_sets, config
    )
    decision = evaluate_audit_decision(
        bank, components, assignments, freeze, pooled, folds, scopes, config
    )
    summary = persist_audit_outputs(
        bank,
        components,
        freeze,
        hidden_evidence,
        truth_evidence,
        truth_sha,
        pooled,
        folds,
        scopes,
        blocks,
        by_well,
        decision,
        config,
        artifacts_dir,
    )
    print("H256 primary:", decision["primary"]["h256"])
    print("H512 primary:", decision["primary"]["h512"])
    print("Audit passed:", decision["audit_passed"])
    print("Next branch:", decision["next_branch"])
    print("Generated artifacts:", artifacts_dir)
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
    print(json.dumps(SUMMARY["decision"], indent=2, ensure_ascii=False))
