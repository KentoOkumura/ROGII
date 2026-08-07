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
# # exp312 Type Well group conditional GR emission-table audit
#
# This train-side, zero-booster audit keeps the exp293 deployable12 physical
# candidate paths fixed. For every outer fold it fits a hierarchical Student-t
# table from outer-train wells only, freezes candidate rank orders, and opens
# outer-valid TVT only for truth-nearest candidate-rank readout. It never runs a
# decoder, changes a candidate TVT, trains a model, or creates a submission.

# %% [markdown]
# ## Contents
# 1. Imports and immutable scientific contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Exp293 deployable12 candidate-bank reconstruction
# 4. Exp311 fold/group contract and target-free raw context
# 5. Fold-safe bins and hierarchical Student-t table
# 6. Target-free candidate scoring, controls, and freeze boundary
# 7. Late truth join, rank metrics, and promotion gate
# 8. Setup, execution orchestration, and generated artifacts

# %%
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp312_typewell_group_conditional_gr_emission_table"
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
PRIMITIVE_CANDIDATES = EXPECTED_CANDIDATE_ORDER[:6]
FORMULA_CANDIDATES = EXPECTED_CANDIDATE_ORDER[6:]
RANK_VARIANTS = ("baseline", "real", "group_label_shuffle", "tvt_shift_matched_count")
FALLBACK_LEVELS = (
    "global_unconditional",
    "global_conditional",
    "group_unconditional",
    "group_conditional",
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


# %% [markdown]
# ## 2. Runtime, path, SHA, and serialization helpers


# %%
def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = os.environ.get("EXP312_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError("exp312 config.yaml was not found unambiguously")


def runtime_artifacts_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_work_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / ".exp312_work"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / ".exp312_work"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    return (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "metrics.json"
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


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    row_hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "raw_sha256": sha256_file(path),
        "content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    root = project_root()
    for raw_pattern in patterns:
        path = Path(str(raw_pattern))
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            found.append(direct)
            continue
        for raw in (str(path), str(root / path) if not path.is_absolute() else str(path)):
            found.extend(Path(match) for match in glob.glob(raw, recursive=True))
    unique: dict[str, Path] = {}
    for path in found:
        if path.exists():
            unique.setdefault(str(path.resolve()), path)
    return list(unique.values())


def resolve_file(
    patterns: Sequence[str], *, label: str, expected_sha256: str | None = None
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


def stable_sha256_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    required = {
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.n_folds": 5,
        "validation.truth_join_policy": (
            "freeze_emission_table_and_candidate_scores_before_outer_valid_truth_join"
        ),
        "validation.parent_gate_override.enabled": True,
        "candidate_bank.expected_count": 12,
        "emission_table.fixed_df": 5,
        "emission_table.shrinkage_support_k": 200,
        "execution_contract.scientific_variants": 1,
        "execution_contract.diagnostic_controls": 2,
        "execution_contract.model_configs": 0,
        "execution_contract.folds": 5,
        "execution_contract.boosters": 0,
        "execution_contract.decoder_runs": 0,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in required.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"contract mismatch for {key}: {actual!r} != {expected!r}")
    if tuple(get_nested(config, "candidate_bank.order")) != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("candidate order differs from exp293 deployable12 contract")
    if tuple(get_nested(config, "candidate_bank.primitives")) != PRIMITIVE_CANDIDATES:
        raise ValueError("primitive candidates differ from exp293 contract")
    if list(get_nested(config, "emission_table.fallback_order")) != [
        "group_conditional",
        "group_unconditional",
        "global_conditional",
        "global_unconditional",
    ]:
        raise ValueError("fallback order differs from frozen contract")


# %% [markdown]
# ## 3. Exp293 deployable12 candidate-bank reconstruction


# %%
@dataclass
class CandidateBank:
    keys: pd.DataFrame
    candidate_ids: tuple[str, ...]
    values: np.memmap
    values_path: Path
    manifest: dict[str, Any]
    manifest_path: Path
    key_content_sha256: str
    candidate_content_sha256: str
    sample_parity: pd.DataFrame
    input_evidence: list[dict[str, Any]]


def reject_forbidden_candidate_columns(columns: Iterable[str]) -> None:
    normalized = {str(column) for column in columns}
    forbidden = normalized & FORBIDDEN_CANDIDATE_COLUMNS
    token_forbidden = {
        column
        for column in normalized
        if any(token in column.lower() for token in ("true_tvt", "abs_error", "oracle_label"))
    }
    if forbidden or token_forbidden:
        raise ValueError(
            "candidate partition exposes forbidden truth/readout columns: "
            f"{sorted(forbidden | token_forbidden)}"
        )


def _artifact_path_from_manifest(manifest_path: Path, item: Mapping[str, Any]) -> Path:
    raw = str(item["path"])
    if "/artifacts/" in raw:
        candidate = manifest_path.parent / raw.split("/artifacts/", 1)[1]
        if candidate.exists():
            return candidate
    direct = Path(raw)
    if direct.exists():
        return direct
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
        if item.get("file_sha256") and sha256_file(path) != str(item["file_sha256"]):
            raise ValueError(f"{label} partition file SHA mismatch: {path}")
        full = pd.read_parquet(path)
        reject_forbidden_candidate_columns(full.columns)
        if len(full) != int(item.get("rows", len(full))):
            raise ValueError(f"{label} partition row mismatch: {path}")
        if item.get("schema_sha256") and frame_schema_sha256(full) != str(item["schema_sha256"]):
            raise ValueError(f"{label} partition schema SHA mismatch: {path}")
        if item.get("content_sha256") and frame_content_sha256(full) != str(item["content_sha256"]):
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
                "file_sha256": sha256_file(path),
                "logical_content_sha256": frame_content_sha256(full),
                "schema_sha256": frame_schema_sha256(full),
            }
        )
    return pd.concat(frames, ignore_index=True), evidence


def _assert_same_keys(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
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


def _materialize_formulas(
    values: np.memmap, column_by_candidate: Mapping[str, int], config: Mapping[str, Any]
) -> None:
    for candidate_id, weights in get_nested(config, "candidate_bank.pairs").items():
        parents = list(weights)
        if len(parents) != 2 or any(float(weights[parent]) != 0.5 for parent in parents):
            raise ValueError(f"{candidate_id} differs from fixed 50/50 contract")
        values[:, column_by_candidate[candidate_id]] = (
            np.float32(0.5)
            * (
                values[:, column_by_candidate[parents[0]]]
                + values[:, column_by_candidate[parents[1]]]
            )
        ).astype(np.float32)
    fixed = get_nested(config, "candidate_bank.fixed_formula")
    expected = {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25}
    if fixed != {"exp226_w500_50_50": expected}:
        raise ValueError("fixed formula differs from exp293 contract")
    values[:, column_by_candidate["exp226_w500_50_50"]] = (
        np.float32(0.5) * values[:, column_by_candidate["exp226_k16"]]
        + np.float32(0.25) * values[:, column_by_candidate["likpf_mean"]]
        + np.float32(0.25) * values[:, column_by_candidate["exact_hmm"]]
    ).astype(np.float32)
    values.flush()


def candidate_bank_content_sha256(bank: CandidateBank, chunk_rows: int) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(bank.candidate_ids), separators=(",", ":")).encode())
    digest.update(bank.key_content_sha256.encode())
    for position, candidate_id in enumerate(bank.candidate_ids):
        digest.update(candidate_id.encode())
        for start in range(0, len(bank.keys), chunk_rows):
            values = np.asarray(bank.values[start : start + chunk_rows, position], dtype="<f4")
            digest.update(values.tobytes())
    return digest.hexdigest()


def build_formula_sample_parity(
    bank: CandidateBank, parity_path: Path, tolerance: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample = pd.read_parquet(parity_path)
    if "id" not in sample:
        raise ValueError("exp263 small parity sample lacks id")
    missing = set(FORMULA_CANDIDATES) - set(sample.columns)
    if missing:
        raise ValueError(f"exp263 parity formulas missing: {sorted(missing)}")
    indexer = pd.Index(bank.keys["id"].astype(str)).get_indexer(sample["id"].astype(str))
    if np.any(indexer < 0):
        raise ValueError("exp263 parity IDs are absent from deployable12 bank")
    candidate_position = {
        candidate_id: position for position, candidate_id in enumerate(bank.candidate_ids)
    }
    rows: list[dict[str, Any]] = []
    for candidate_id in FORMULA_CANDIDATES:
        actual = np.asarray(
            bank.values[indexer, candidate_position[candidate_id]], dtype=np.float64
        )
        expected = pd.to_numeric(sample[candidate_id], errors="raise").to_numpy(np.float64)
        max_abs = float(np.max(np.abs(actual - expected), initial=0.0))
        rows.append(
            {
                "candidate_id": candidate_id,
                "max_abs_ft": max_abs,
                "tolerance_ft": tolerance,
                "passed": bool(max_abs <= tolerance),
            }
        )
    evidence = {
        "phase": "target_free",
        "source": "exp263_small_parity_sample",
        "path": str(parity_path),
        "rows": len(sample),
        "file_sha256": sha256_file(parity_path),
        "logical_content_sha256": frame_content_sha256(sample),
        "schema_sha256": frame_schema_sha256(sample),
    }
    return pd.DataFrame(rows), evidence


def build_candidate_bank(config: Mapping[str, Any], work_dir: Path) -> CandidateBank:
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
    if (
        int(manifest.get("rows", -1)) != expected_rows
        or int(manifest.get("wells", -1)) != expected_wells
        or int(manifest.get("folds", -1)) != expected_folds
    ):
        raise ValueError("exp263 manifest row/well/fold contract mismatch")
    if manifest.get("canonical_id_sha256") != manifest_cfg["expected_canonical_id_sha256"]:
        raise ValueError("exp263 canonical ID SHA mismatch")

    values_path = work_dir / f"{OUTPUT_PREFIX}_candidate_bank.f32"
    values = np.memmap(
        values_path,
        mode="w+",
        dtype="float32",
        shape=(expected_rows, len(EXPECTED_CANDIDATE_ORDER)),
    )
    values[:] = np.nan
    column_by_candidate = {
        candidate_id: index for index, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER)
    }
    reference_keys: pd.DataFrame | None = None
    input_evidence: list[dict[str, Any]] = []
    for candidate_id in PRIMITIVE_CANDIDATES:
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
        candidate_values = pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        valid = available & finite_flag & np.isfinite(candidate_values)
        candidate_values[~valid] = np.nan
        values[:, column_by_candidate[candidate_id]] = candidate_values
    if reference_keys is None:
        raise AssertionError("candidate bank loading produced no keys")
    _materialize_formulas(values, column_by_candidate, config)
    if not np.isfinite(values).all():
        raise ValueError("deployable12 candidate coverage must be finite and complete")
    if len(reference_keys) != expected_rows or reference_keys["well"].nunique() != expected_wells:
        raise ValueError("candidate bank row/well count mismatch")
    if reference_keys["id"].duplicated().any():
        raise ValueError("candidate bank IDs must be unique")
    key_sha = frame_content_sha256(reference_keys[VALUE_KEY_COLUMNS])
    bank = CandidateBank(
        keys=reference_keys.reset_index(drop=True),
        candidate_ids=EXPECTED_CANDIDATE_ORDER,
        values=values,
        values_path=values_path,
        manifest=manifest,
        manifest_path=manifest_path,
        key_content_sha256=key_sha,
        candidate_content_sha256="",
        sample_parity=pd.DataFrame(),
        input_evidence=input_evidence,
    )
    bank.candidate_content_sha256 = candidate_bank_content_sha256(
        bank, int(get_nested(config, "audit.work_chunk_rows"))
    )
    parity_path = manifest_path.parent / str(manifest_cfg["small_parity_filename"])
    if not parity_path.exists():
        raise FileNotFoundError(f"exp263 small parity sample missing: {parity_path}")
    parity, parity_evidence = build_formula_sample_parity(
        bank,
        parity_path,
        float(get_nested(config, "candidate_bank.formula_parity_max_abs_ft")),
    )
    if not bool(parity["passed"].all()):
        raise ValueError("exp293 deployable12 formula parity failed")
    bank.sample_parity = parity
    bank.input_evidence.append(parity_evidence)
    return bank


# %% [markdown]
# ## 4. Exp311 fold/group contract and target-free raw context


# %%
@dataclass
class WellContext:
    well_id: str
    horizontal_path: Path
    typewell_path: Path
    horizontal_gr: np.ndarray
    horizontal_missing: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    typewell_abs_gradient: np.ndarray


@dataclass
class ParentContract:
    summary_path: Path
    fold_manifest: pd.DataFrame
    group_membership: pd.DataFrame
    group_by_well: dict[str, str]
    fold_by_well: dict[str, int]
    input_evidence: list[dict[str, Any]]


@dataclass
class TargetFreeContext:
    wells: dict[str, WellContext]
    observed_gr: np.ndarray
    missing_flag: np.ndarray
    candidate_typewell_gr: np.memmap
    candidate_abs_gradient: np.memmap
    shift_source_index: np.ndarray
    hidden_like: pd.DataFrame
    input_evidence: list[dict[str, Any]]


def collapse_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"TVT", "GR"}
    if not required.issubset(typewell.columns):
        raise ValueError(f"typewell is missing {sorted(required - set(typewell.columns))}")
    work = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    work = work.groupby("TVT", as_index=False, sort=True)["GR"].median()
    tvt = work["TVT"].to_numpy(np.float64)
    gr = work["GR"].to_numpy(np.float64)
    if len(tvt) < 2 or not np.all(np.diff(tvt) > 0):
        raise ValueError("typewell requires two strictly increasing finite TVT values")
    gradient = np.abs(np.gradient(gr, tvt))
    return tvt, gr, gradient


def interpolate_endpoint_clamp(query: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    result = np.full(query.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(query)
    result[finite] = np.interp(query[finite], x, y)
    return result


def resolve_parent_contract(config: Mapping[str, Any]) -> ParentContract:
    spec = get_nested(config, "data.exp311_parent")
    summary_path = resolve_file(
        spec["summary_candidates"],
        label="exp311 summary",
        expected_sha256=str(spec["expected_summary_sha256"]),
    )
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != spec["expected_parent_status"]:
        raise ValueError("exp311 status differs from recorded gate-failed parent")
    if summary.get("promotion", {}).get("passed") is not False:
        raise ValueError("exp311 gate failure must remain explicit")
    if summary.get("promotion", {}).get("primary_scheme") != spec["expected_primary_scheme"]:
        raise ValueError("exp311 primary group scheme mismatch")
    fold_path = summary_path.parent / str(spec["fold_manifest_filename"])
    group_path = summary_path.parent / str(spec["group_membership_filename"])
    if sha256_file(fold_path) != spec["expected_fold_manifest_sha256"]:
        raise ValueError("exp311 fold manifest SHA mismatch")
    if sha256_file(group_path) != spec["expected_group_membership_sha256"]:
        raise ValueError("exp311 group membership SHA mismatch")
    folds = pd.read_csv(fold_path, dtype={"well_id": str})
    groups = pd.read_csv(group_path, dtype={"well_id": str, "group_id": str})
    groups = groups[groups["group_scheme"] == spec["expected_primary_scheme"]].copy()
    if folds["well_id"].duplicated().any() or groups["well_id"].duplicated().any():
        raise ValueError("exp311 parent well identities must be unique")
    return ParentContract(
        summary_path=summary_path,
        fold_manifest=folds,
        group_membership=groups,
        group_by_well=dict(zip(groups["well_id"], groups["group_id"], strict=False)),
        fold_by_well=dict(zip(folds["well_id"], folds["fold"], strict=False)),
        input_evidence=[
            {
                "phase": "target_free",
                "source": "exp311_summary",
                "path": str(summary_path),
                "raw_sha256": sha256_file(summary_path),
                "rows": 1,
            },
            {
                "phase": "target_free",
                "source": "exp311_fold_manifest",
                "path": str(fold_path),
                "raw_sha256": sha256_file(fold_path),
                "rows": len(folds),
            },
            {
                "phase": "target_free",
                "source": "exp311_group_membership",
                "path": str(group_path),
                "raw_sha256": sha256_file(group_path),
                "rows": len(groups),
            },
        ],
    )


def resolve_raw_train_dir(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[Path, list[Path]]:
    patterns = list(get_nested(config, "data.raw_train_dir_patterns"))
    horizontal_glob = str(get_nested(config, "data.raw_horizontal_glob"))
    for directory in expand_existing_paths(patterns):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob(horizontal_glob))
        wells = {path.name.split("__horizontal_well.csv", 1)[0] for path in files}
        if wells == expected_wells and len(files) == len(expected_wells):
            return directory, files
    raise FileNotFoundError("raw train directory with exact 773-well inventory was not found")


def stable_shift_source_index(keys: pd.DataFrame, config: Mapping[str, Any]) -> np.ndarray:
    result = np.arange(len(keys), dtype=np.int64)
    minimum = float(
        get_nested(config, "negative_controls.tvt_shift_matched_count.minimum_fraction")
    )
    maximum = float(
        get_nested(config, "negative_controls.tvt_shift_matched_count.maximum_fraction")
    )
    for well, index in keys.groupby("well", sort=True).indices.items():
        positions = np.asarray(index, dtype=np.int64)
        n_rows = len(positions)
        low = max(1, int(math.ceil(minimum * n_rows)))
        high = max(low, int(math.floor(maximum * n_rows)))
        shift = low + stable_sha256_int(EXPERIMENT_NAME, "tvt_shift", well) % (high - low + 1)
        result[positions] = np.roll(positions, int(shift))
    return result


def shuffled_group_lookup(
    wells: Sequence[str], group_by_well: Mapping[str, str], fold: int
) -> dict[str, str]:
    ordered_wells = sorted(
        {str(well) for well in wells},
        key=lambda well: (stable_sha256_int(EXPERIMENT_NAME, "group_shuffle", fold, well), well),
    )
    labels = [str(group_by_well[well]) for well in ordered_wells]
    if len(labels) < 2:
        return dict(zip(ordered_wells, labels, strict=False))
    shift = 1 + stable_sha256_int(EXPERIMENT_NAME, "group_shift", fold) % (len(labels) - 1)
    rotated = labels[int(shift) :] + labels[: int(shift)]
    return dict(zip(ordered_wells, rotated, strict=False))


def load_hidden_like(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        spec["candidates"],
        label="exp115 hidden-like assignment",
        expected_sha256=str(spec["expected_sha256"]),
    )
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *get_nested(config, "validation.hidden_like_roles")}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignment missing {sorted(required - set(frame.columns))}")
    return frame[list(required)].copy(), {
        "phase": "target_free",
        "source": "exp115_hidden_like_assignment",
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "rows": len(frame),
    }


def build_target_free_context(
    bank: CandidateBank,
    parent: ParentContract,
    config: Mapping[str, Any],
    work_dir: Path,
) -> TargetFreeContext:
    expected_wells = set(bank.keys["well"].astype(str))
    if set(parent.group_by_well) != expected_wells or set(parent.fold_by_well) != expected_wells:
        raise ValueError("exp311 group/fold inventory differs from deployable12 wells")
    raw_dir, horizontal_files = resolve_raw_train_dir(config, expected_wells)
    raw_columns = get_nested(config, "data.raw_columns")
    wells: dict[str, WellContext] = {}
    evidence: list[dict[str, Any]] = []
    for sequence, horizontal_path in enumerate(horizontal_files, start=1):
        well = horizontal_path.name.split("__horizontal_well.csv", 1)[0]
        typewell_path = raw_dir / f"{well}__typewell.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=[raw_columns["gr"]])
        raw_gr = pd.to_numeric(horizontal[raw_columns["gr"]], errors="coerce")
        missing = raw_gr.isna().to_numpy()
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        typewell_tvt, typewell_gr, typewell_gradient = collapse_typewell(typewell)
        gr_fill = float(np.nanmean(typewell_gr))
        imputed = raw_gr.interpolate(limit_direction="both").fillna(gr_fill).to_numpy(np.float64)
        wells[well] = WellContext(
            well_id=well,
            horizontal_path=horizontal_path,
            typewell_path=typewell_path,
            horizontal_gr=imputed,
            horizontal_missing=missing,
            typewell_tvt=typewell_tvt,
            typewell_gr=typewell_gr,
            typewell_abs_gradient=typewell_gradient,
        )
        evidence.append(
            {
                "phase": "target_free",
                "source": "raw_well_context",
                "well_id": well,
                "horizontal_path": str(horizontal_path),
                "horizontal_sha256": sha256_file(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_sha256": sha256_file(typewell_path),
                "rows": len(horizontal),
            }
        )
        if sequence == 1 or sequence % 50 == 0 or sequence == len(horizontal_files):
            print(
                f"target-free context [{sequence}/{len(horizontal_files)}] well={well}", flush=True
            )

    n_rows, n_candidates = bank.values.shape
    observed_gr = np.empty(n_rows, dtype=np.float32)
    missing_flag = np.empty(n_rows, dtype=np.uint8)
    candidate_gr_path = work_dir / f"{OUTPUT_PREFIX}_candidate_typewell_gr.f32"
    candidate_grad_path = work_dir / f"{OUTPUT_PREFIX}_candidate_abs_gradient.f32"
    candidate_gr = np.memmap(
        candidate_gr_path, mode="w+", dtype="float32", shape=(n_rows, n_candidates)
    )
    candidate_grad = np.memmap(
        candidate_grad_path, mode="w+", dtype="float32", shape=(n_rows, n_candidates)
    )
    for well, index in bank.keys.groupby("well", sort=True).indices.items():
        positions = np.asarray(index, dtype=np.int64)
        row_idx = bank.keys.iloc[positions]["well_row_idx"].to_numpy(np.int64)
        context = wells[str(well)]
        if row_idx.max(initial=-1) >= len(context.horizontal_gr):
            raise ValueError(f"candidate row index exceeds raw well length: {well}")
        observed_gr[positions] = context.horizontal_gr[row_idx].astype(np.float32)
        missing_flag[positions] = context.horizontal_missing[row_idx].astype(np.uint8)
        candidate_tvt = np.asarray(bank.values[positions], dtype=np.float64)
        candidate_gr[positions] = interpolate_endpoint_clamp(
            candidate_tvt, context.typewell_tvt, context.typewell_gr
        ).astype(np.float32)
        candidate_grad[positions] = interpolate_endpoint_clamp(
            candidate_tvt, context.typewell_tvt, context.typewell_abs_gradient
        ).astype(np.float32)
    candidate_gr.flush()
    candidate_grad.flush()
    if (
        not np.isfinite(observed_gr).all()
        or not np.isfinite(candidate_gr).all()
        or not np.isfinite(candidate_grad).all()
    ):
        raise ValueError("target-free candidate emission inputs must be finite")
    hidden_like, hidden_evidence = load_hidden_like(config)
    evidence.append(hidden_evidence)
    evidence.append(
        {
            "phase": "target_free",
            "source": "candidate_emission_features",
            "path": str(candidate_gr_path),
            "rows": n_rows,
            "candidate_count": n_candidates,
            "candidate_gr_sha256": sha256_file(candidate_gr_path),
            "candidate_gradient_sha256": sha256_file(candidate_grad_path),
        }
    )
    return TargetFreeContext(
        wells=wells,
        observed_gr=observed_gr,
        missing_flag=missing_flag,
        candidate_typewell_gr=candidate_gr,
        candidate_abs_gradient=candidate_grad,
        shift_source_index=stable_shift_source_index(bank.keys, config),
        hidden_like=hidden_like,
        input_evidence=evidence,
    )


# %% [markdown]
# ## 5. Fold-safe bins and hierarchical Student-t table


# %%
def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("weighted quantile requires finite positive-weight rows")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(np.asarray(quantiles, dtype=np.float64), cumulative, values)


def weighted_location_scale(
    values: np.ndarray, weights: np.ndarray, scale_floor: float
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("emission statistic received no finite rows")
    location = float(weighted_quantile(values, weights, [0.5])[0])
    mad = float(weighted_quantile(np.abs(values - location), weights, [0.5])[0])
    scale = max(float(scale_floor), 1.4826 * mad)
    sum_w = float(weights.sum())
    effective = sum_w * sum_w / float(np.square(weights).sum())
    return {
        "location_raw": location,
        "scale_raw": scale,
        "support_rows": float(len(values)),
        "sum_weight": sum_w,
        "effective_rows": effective,
    }


def assign_bins(values: np.ndarray, internal_edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(internal_edges, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(edges).all():
        raise ValueError("bin assignment requires finite values and edges")
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def build_fit_records(
    contexts: Mapping[str, WellContext],
    train_wells: Sequence[str],
    group_lookup: Mapping[str, str],
    fold: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    truth_column = str(get_nested(config, "data.raw_columns.truth"))
    frames: list[pd.DataFrame] = []
    for well in sorted(str(item) for item in train_wells):
        context = contexts[well]
        truth = pd.to_numeric(
            pd.read_csv(context.horizontal_path, usecols=[truth_column])[truth_column],
            errors="coerce",
        ).to_numpy(np.float64)
        if len(truth) != len(context.horizontal_gr):
            raise ValueError(f"truth/context row mismatch: {well}")
        typewell_gr = interpolate_endpoint_clamp(truth, context.typewell_tvt, context.typewell_gr)
        gradient = interpolate_endpoint_clamp(
            truth, context.typewell_tvt, context.typewell_abs_gradient
        )
        residual = context.horizontal_gr - typewell_gr
        valid = (
            np.isfinite(truth)
            & np.isfinite(typewell_gr)
            & np.isfinite(gradient)
            & np.isfinite(residual)
        )
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "well_id": well,
                    "group_id": str(group_lookup[well]),
                    "typewell_gr": typewell_gr[valid],
                    "abs_gradient": gradient[valid],
                    "missing_flag": context.horizontal_missing[valid].astype(np.int8),
                    "residual": residual[valid],
                    "weight": np.full(n_valid, 1.0 / n_valid, dtype=np.float64),
                }
            )
        )
    if not frames:
        raise ValueError(f"fold {fold} produced no outer-train emission rows")
    result = pd.concat(frames, ignore_index=True)
    if set(result["well_id"]) != set(str(well) for well in train_wells):
        raise ValueError("not every outer-train well contributed emission rows")
    return result


def build_bin_edges(
    records: pd.DataFrame, fold: int, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_gr = int(get_nested(config, "emission_table.binning.typewell_gr_quantiles"))
    n_gradient = int(
        get_nested(config, "emission_table.binning.abs_typewell_gr_gradient_quantiles")
    )
    gr_edges = weighted_quantile(
        records["typewell_gr"].to_numpy(),
        records["weight"].to_numpy(),
        np.arange(1, n_gr) / n_gr,
    )
    gradient_edges = weighted_quantile(
        records["abs_gradient"].to_numpy(),
        records["weight"].to_numpy(),
        np.arange(1, n_gradient) / n_gradient,
    )
    edge_rows = [
        {"fold": fold, "axis": "typewell_gr", "edge_index": index, "edge_value": value}
        for index, value in enumerate(gr_edges, start=1)
    ] + [
        {"fold": fold, "axis": "abs_gradient", "edge_index": index, "edge_value": value}
        for index, value in enumerate(gradient_edges, start=1)
    ]
    return gr_edges, gradient_edges, pd.DataFrame(edge_rows)


def _stats_record(
    subset: pd.DataFrame,
    *,
    fold: int,
    control: str,
    level: str,
    group_id: str,
    gr_bin: int,
    gradient_bin: int,
    missing_flag: int,
    scale_floor: float,
) -> dict[str, Any]:
    stats = weighted_location_scale(
        subset["residual"].to_numpy(), subset["weight"].to_numpy(), scale_floor
    )
    return {
        "fold": fold,
        "control": control,
        "level": level,
        "group_id": group_id,
        "gr_bin": gr_bin,
        "gradient_bin": gradient_bin,
        "missing_flag": missing_flag,
        "support_wells": int(subset["well_id"].nunique()),
        **stats,
    }


def fit_hierarchical_table(
    records: pd.DataFrame,
    gr_edges: np.ndarray,
    gradient_edges: np.ndarray,
    *,
    fold: int,
    control: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    work = records.copy()
    work["gr_bin"] = assign_bins(work["typewell_gr"].to_numpy(), gr_edges)
    work["gradient_bin"] = assign_bins(work["abs_gradient"].to_numpy(), gradient_edges)
    scale_floor = float(get_nested(config, "emission_table.scale_floor_gr"))
    rows: list[dict[str, Any]] = []
    rows.append(
        _stats_record(
            work,
            fold=fold,
            control=control,
            level="global_unconditional",
            group_id="*",
            gr_bin=-1,
            gradient_bin=-1,
            missing_flag=-1,
            scale_floor=scale_floor,
        )
    )
    for keys, subset in work.groupby(
        ["gr_bin", "gradient_bin", "missing_flag"], sort=True, observed=True
    ):
        rows.append(
            _stats_record(
                subset,
                fold=fold,
                control=control,
                level="global_conditional",
                group_id="*",
                gr_bin=int(keys[0]),
                gradient_bin=int(keys[1]),
                missing_flag=int(keys[2]),
                scale_floor=scale_floor,
            )
        )
    for group_id, subset in work.groupby("group_id", sort=True, observed=True):
        rows.append(
            _stats_record(
                subset,
                fold=fold,
                control=control,
                level="group_unconditional",
                group_id=str(group_id),
                gr_bin=-1,
                gradient_bin=-1,
                missing_flag=-1,
                scale_floor=scale_floor,
            )
        )
    for keys, subset in work.groupby(
        ["group_id", "gr_bin", "gradient_bin", "missing_flag"],
        sort=True,
        observed=True,
    ):
        rows.append(
            _stats_record(
                subset,
                fold=fold,
                control=control,
                level="group_conditional",
                group_id=str(keys[0]),
                gr_bin=int(keys[1]),
                gradient_bin=int(keys[2]),
                missing_flag=int(keys[3]),
                scale_floor=scale_floor,
            )
        )
    table = pd.DataFrame(rows)
    table["location"] = table["location_raw"]
    table["scale"] = table["scale_raw"]
    table["shrinkage_alpha"] = 1.0
    k = float(get_nested(config, "emission_table.shrinkage_support_k"))
    index = {
        (
            row.level,
            row.group_id,
            int(row.gr_bin),
            int(row.gradient_bin),
            int(row.missing_flag),
        ): position
        for position, row in table.iterrows()
    }
    global_position = index[("global_unconditional", "*", -1, -1, -1)]
    level_order = ("global_conditional", "group_unconditional", "group_conditional")
    for level in level_order:
        for position in table.index[table["level"] == level]:
            row = table.loc[position]
            if level == "global_conditional":
                parent_position = global_position
            elif level == "group_unconditional":
                parent_position = global_position
            else:
                parent_position = index.get(
                    ("group_unconditional", str(row["group_id"]), -1, -1, -1),
                    index.get(
                        (
                            "global_conditional",
                            "*",
                            int(row["gr_bin"]),
                            int(row["gradient_bin"]),
                            int(row["missing_flag"]),
                        ),
                        global_position,
                    ),
                )
            parent = table.loc[parent_position]
            alpha = float(row["effective_rows"] / (row["effective_rows"] + k))
            table.loc[position, "shrinkage_alpha"] = alpha
            table.loc[position, "location"] = alpha * float(row["location_raw"]) + (
                1.0 - alpha
            ) * float(parent["location"])
            table.loc[position, "scale"] = math.exp(
                alpha * math.log(float(row["scale_raw"]))
                + (1.0 - alpha) * math.log(float(parent["scale"]))
            )
    table["available"] = table["effective_rows"] >= float(
        get_nested(config, "emission_table.min_effective_rows")
    )
    table.loc[table["level"] == "global_unconditional", "available"] = True
    return table.sort_values(
        ["fold", "control", "level", "group_id", "gr_bin", "gradient_bin", "missing_flag"]
    ).reset_index(drop=True)


@dataclass
class TableArrays:
    group_to_code: dict[str, int]
    global_location: float
    global_scale: float
    global_cond_location: np.ndarray
    global_cond_scale: np.ndarray
    global_cond_available: np.ndarray
    group_location: np.ndarray
    group_scale: np.ndarray
    group_available: np.ndarray
    group_cond_location: np.ndarray
    group_cond_scale: np.ndarray
    group_cond_available: np.ndarray


def compile_table_arrays(
    table: pd.DataFrame, group_ids: Sequence[str], n_gr_bins: int, n_gradient_bins: int
) -> TableArrays:
    group_to_code = {group_id: index for index, group_id in enumerate(sorted(set(group_ids)))}
    shape_cond = (n_gr_bins, n_gradient_bins, 2)
    global_cond_location = np.full(shape_cond, np.nan, dtype=np.float64)
    global_cond_scale = np.full(shape_cond, np.nan, dtype=np.float64)
    global_cond_available = np.zeros(shape_cond, dtype=bool)
    n_groups = len(group_to_code)
    group_location = np.full(n_groups, np.nan, dtype=np.float64)
    group_scale = np.full(n_groups, np.nan, dtype=np.float64)
    group_available = np.zeros(n_groups, dtype=bool)
    group_cond_location = np.full((n_groups, *shape_cond), np.nan, dtype=np.float64)
    group_cond_scale = np.full((n_groups, *shape_cond), np.nan, dtype=np.float64)
    group_cond_available = np.zeros((n_groups, *shape_cond), dtype=bool)
    global_row = table[table["level"] == "global_unconditional"].iloc[0]
    for row in table.itertuples(index=False):
        if row.level == "global_conditional":
            key = (int(row.gr_bin), int(row.gradient_bin), int(row.missing_flag))
            global_cond_location[key] = float(row.location)
            global_cond_scale[key] = float(row.scale)
            global_cond_available[key] = bool(row.available)
        elif row.level == "group_unconditional" and row.group_id in group_to_code:
            code = group_to_code[row.group_id]
            group_location[code] = float(row.location)
            group_scale[code] = float(row.scale)
            group_available[code] = bool(row.available)
        elif row.level == "group_conditional" and row.group_id in group_to_code:
            key = (
                group_to_code[row.group_id],
                int(row.gr_bin),
                int(row.gradient_bin),
                int(row.missing_flag),
            )
            group_cond_location[key] = float(row.location)
            group_cond_scale[key] = float(row.scale)
            group_cond_available[key] = bool(row.available)
    return TableArrays(
        group_to_code=group_to_code,
        global_location=float(global_row["location"]),
        global_scale=float(global_row["scale"]),
        global_cond_location=global_cond_location,
        global_cond_scale=global_cond_scale,
        global_cond_available=global_cond_available,
        group_location=group_location,
        group_scale=group_scale,
        group_available=group_available,
        group_cond_location=group_cond_location,
        group_cond_scale=group_cond_scale,
        group_cond_available=group_cond_available,
    )


def lookup_distribution(
    arrays: TableArrays,
    group_ids: np.ndarray,
    gr_bins: np.ndarray,
    gradient_bins: np.ndarray,
    missing_flags: np.ndarray,
    *,
    force_global: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.asarray(gr_bins).shape
    location = np.full(shape, arrays.global_location, dtype=np.float64)
    scale = np.full(shape, arrays.global_scale, dtype=np.float64)
    level = np.zeros(shape, dtype=np.uint8)
    if force_global:
        return location, scale, level
    gr_bins = np.asarray(gr_bins, dtype=np.int64)
    gradient_bins = np.asarray(gradient_bins, dtype=np.int64)
    missing_flags = np.asarray(missing_flags, dtype=np.int64)
    global_available = arrays.global_cond_available[gr_bins, gradient_bins, missing_flags]
    location[global_available] = arrays.global_cond_location[
        gr_bins[global_available], gradient_bins[global_available], missing_flags[global_available]
    ]
    scale[global_available] = arrays.global_cond_scale[
        gr_bins[global_available], gradient_bins[global_available], missing_flags[global_available]
    ]
    level[global_available] = 1
    group_codes = np.fromiter(
        (arrays.group_to_code.get(str(group_id), -1) for group_id in np.asarray(group_ids).ravel()),
        dtype=np.int32,
        count=np.asarray(group_ids).size,
    ).reshape(shape)
    known_group = group_codes >= 0
    group_available = np.zeros(shape, dtype=bool)
    group_available[known_group] = arrays.group_available[group_codes[known_group]]
    location[group_available] = arrays.group_location[group_codes[group_available]]
    scale[group_available] = arrays.group_scale[group_codes[group_available]]
    level[group_available] = 2
    conditional_available = np.zeros(shape, dtype=bool)
    conditional_available[known_group] = arrays.group_cond_available[
        group_codes[known_group],
        gr_bins[known_group],
        gradient_bins[known_group],
        missing_flags[known_group],
    ]
    location[conditional_available] = arrays.group_cond_location[
        group_codes[conditional_available],
        gr_bins[conditional_available],
        gradient_bins[conditional_available],
        missing_flags[conditional_available],
    ]
    scale[conditional_available] = arrays.group_cond_scale[
        group_codes[conditional_available],
        gr_bins[conditional_available],
        gradient_bins[conditional_available],
        missing_flags[conditional_available],
    ]
    level[conditional_available] = 3
    return location, scale, level


def student_t_log_likelihood(
    residual: np.ndarray, location: np.ndarray, scale: np.ndarray, df: float
) -> np.ndarray:
    z = (np.asarray(residual, dtype=np.float64) - location) / scale
    return -np.log(scale) - 0.5 * (df + 1.0) * np.log1p(np.square(z) / df)


# %% [markdown]
# ## 6. Target-free candidate scoring, controls, and freeze boundary


# %%
@dataclass
class FrozenFold:
    fold: int
    valid_positions: np.ndarray
    rank_path: Path
    rank_sha256: str
    table_sha256: str
    freeze_path: Path
    freeze_sha256: str
    gr_edges: np.ndarray
    gradient_edges: np.ndarray
    bin_edges: pd.DataFrame
    emission_tables: pd.DataFrame
    fallback: pd.DataFrame


def _rank_orders(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), axis=1, kind="stable").astype(np.uint8)


def score_fold_target_free(
    bank: CandidateBank,
    context: TargetFreeContext,
    parent: ParentContract,
    fold: int,
    config: Mapping[str, Any],
    work_dir: Path,
) -> FrozenFold:
    all_wells = sorted(parent.fold_by_well)
    train_wells = [well for well in all_wells if parent.fold_by_well[well] != fold]
    valid_wells = [well for well in all_wells if parent.fold_by_well[well] == fold]
    if set(train_wells) & set(valid_wells):
        raise ValueError("outer-train and outer-valid wells overlap")
    real_records = build_fit_records(context.wells, train_wells, parent.group_by_well, fold, config)
    gr_edges, gradient_edges, bin_edges = build_bin_edges(real_records, fold, config)
    shuffled_lookup = shuffled_group_lookup(all_wells, parent.group_by_well, fold)
    shuffled_records = real_records.copy()
    shuffled_records["group_id"] = shuffled_records["well_id"].map(shuffled_lookup)
    real_table = fit_hierarchical_table(
        real_records,
        gr_edges,
        gradient_edges,
        fold=fold,
        control="real",
        config=config,
    )
    shuffled_table = fit_hierarchical_table(
        shuffled_records,
        gr_edges,
        gradient_edges,
        fold=fold,
        control="group_label_shuffle",
        config=config,
    )
    tables = pd.concat([real_table, shuffled_table], ignore_index=True)
    n_gr_bins = len(gr_edges) + 1
    n_gradient_bins = len(gradient_edges) + 1
    real_arrays = compile_table_arrays(
        real_table, list(parent.group_by_well.values()), n_gr_bins, n_gradient_bins
    )
    shuffled_arrays = compile_table_arrays(
        shuffled_table, list(parent.group_by_well.values()), n_gr_bins, n_gradient_bins
    )
    valid_mask = bank.keys["well"].map(parent.fold_by_well).to_numpy(np.int16) == fold
    valid_positions = np.flatnonzero(valid_mask).astype(np.int64)
    rank_path = work_dir / f"{OUTPUT_PREFIX}_fold{fold}_rank_orders.u1"
    rank_orders = np.memmap(
        rank_path,
        mode="w+",
        dtype="uint8",
        shape=(len(valid_positions), len(RANK_VARIANTS), len(bank.candidate_ids)),
    )
    fallback_counts = np.zeros(len(FALLBACK_LEVELS), dtype=np.int64)
    chunk_rows = int(get_nested(config, "audit.work_chunk_rows"))
    df = float(get_nested(config, "emission_table.fixed_df"))
    for chunk_start in range(0, len(valid_positions), chunk_rows):
        chunk_positions = valid_positions[chunk_start : chunk_start + chunk_rows]
        local_end = chunk_start + len(chunk_positions)
        observed = context.observed_gr[chunk_positions, None].astype(np.float64)
        candidate_gr = np.asarray(context.candidate_typewell_gr[chunk_positions], dtype=np.float64)
        candidate_gradient = np.asarray(
            context.candidate_abs_gradient[chunk_positions], dtype=np.float64
        )
        missing = np.broadcast_to(
            context.missing_flag[chunk_positions, None], candidate_gr.shape
        ).astype(np.int16)
        gr_bins = assign_bins(candidate_gr, gr_edges)
        gradient_bins = assign_bins(candidate_gradient, gradient_edges)
        row_groups = bank.keys.iloc[chunk_positions]["well"].map(parent.group_by_well).to_numpy(str)
        groups = np.broadcast_to(row_groups[:, None], candidate_gr.shape)
        residual = observed - candidate_gr

        base_location, base_scale, _ = lookup_distribution(
            real_arrays, groups, gr_bins, gradient_bins, missing, force_global=True
        )
        baseline_score = student_t_log_likelihood(residual, base_location, base_scale, df)
        real_location, real_scale, levels = lookup_distribution(
            real_arrays, groups, gr_bins, gradient_bins, missing
        )
        real_score = student_t_log_likelihood(residual, real_location, real_scale, df)
        fallback_counts += np.bincount(levels.ravel(), minlength=len(FALLBACK_LEVELS))[
            : len(FALLBACK_LEVELS)
        ]

        shuffled_row_groups = (
            bank.keys.iloc[chunk_positions]["well"].map(shuffled_lookup).to_numpy(str)
        )
        shuffled_groups = np.broadcast_to(shuffled_row_groups[:, None], candidate_gr.shape)
        shuffled_location, shuffled_scale, _ = lookup_distribution(
            shuffled_arrays, shuffled_groups, gr_bins, gradient_bins, missing
        )
        shuffled_score = student_t_log_likelihood(residual, shuffled_location, shuffled_scale, df)

        shifted_positions = context.shift_source_index[chunk_positions]
        shifted_gr = np.asarray(context.candidate_typewell_gr[shifted_positions], dtype=np.float64)
        shifted_gradient = np.asarray(
            context.candidate_abs_gradient[shifted_positions], dtype=np.float64
        )
        shifted_gr_bins = assign_bins(shifted_gr, gr_edges)
        shifted_gradient_bins = assign_bins(shifted_gradient, gradient_edges)
        shifted_residual = observed - shifted_gr
        shifted_location, shifted_scale, _ = lookup_distribution(
            real_arrays,
            groups,
            shifted_gr_bins,
            shifted_gradient_bins,
            missing,
        )
        shifted_score = student_t_log_likelihood(
            shifted_residual, shifted_location, shifted_scale, df
        )
        scores = (baseline_score, real_score, shuffled_score, shifted_score)
        for variant_index, score in enumerate(scores):
            rank_orders[chunk_start:local_end, variant_index] = _rank_orders(score)
    rank_orders.flush()
    rank_sha = sha256_file(rank_path)
    table_sha = frame_content_sha256(tables)
    total_lookups = int(fallback_counts.sum())
    fallback = pd.DataFrame(
        {
            "fold": fold,
            "level_code": np.arange(len(FALLBACK_LEVELS), dtype=np.int8),
            "fallback_level": FALLBACK_LEVELS,
            "lookups": fallback_counts,
            "rate": fallback_counts / total_lookups,
        }
    )
    freeze_payload = {
        "experiment": EXPERIMENT_NAME,
        "fold": fold,
        "outer_train_wells": len(train_wells),
        "outer_valid_wells": len(valid_wells),
        "outer_valid_rows": len(valid_positions),
        "outer_valid_truth_access_count_before_freeze": 0,
        "candidate_bank_content_sha256": bank.candidate_content_sha256,
        "bin_edges_content_sha256": frame_content_sha256(bin_edges),
        "emission_table_content_sha256": table_sha,
        "rank_order_file_sha256": rank_sha,
        "rank_shape": [len(valid_positions), len(RANK_VARIANTS), len(bank.candidate_ids)],
        "rank_variants": list(RANK_VARIANTS),
        "frozen": True,
    }
    freeze_path = work_dir / f"{OUTPUT_PREFIX}_fold{fold}_freeze.json"
    write_json(freeze_path, freeze_payload)
    return FrozenFold(
        fold=fold,
        valid_positions=valid_positions,
        rank_path=rank_path,
        rank_sha256=rank_sha,
        table_sha256=table_sha,
        freeze_path=freeze_path,
        freeze_sha256=sha256_file(freeze_path),
        gr_edges=gr_edges,
        gradient_edges=gradient_edges,
        bin_edges=bin_edges,
        emission_tables=tables,
        fallback=fallback,
    )


# %% [markdown]
# ## 7. Late truth join, rank metrics, and promotion gate


# %%
@dataclass
class FoldReadout:
    fold: int
    valid_positions: np.ndarray
    wells: np.ndarray
    rank_positions: np.ndarray
    true_candidate_index: np.ndarray
    metrics: pd.DataFrame


def verify_frozen_fold(frozen: FrozenFold, bank: CandidateBank) -> None:
    if sha256_file(frozen.freeze_path) != frozen.freeze_sha256:
        raise ValueError("fold freeze manifest changed before truth join")
    payload = json.loads(frozen.freeze_path.read_text())
    if payload.get("outer_valid_truth_access_count_before_freeze") != 0:
        raise ValueError("outer-valid truth was accessed before freeze")
    if payload.get("candidate_bank_content_sha256") != bank.candidate_content_sha256:
        raise ValueError("candidate bank changed after freeze")
    if sha256_file(frozen.rank_path) != frozen.rank_sha256:
        raise ValueError("candidate rank order changed after freeze")
    if payload.get("emission_table_content_sha256") != frozen.table_sha256:
        raise ValueError("emission table SHA changed after freeze")


def true_candidate_rank_positions(
    rank_orders: np.ndarray, true_candidate_index: np.ndarray
) -> np.ndarray:
    orders = np.asarray(rank_orders, dtype=np.int16)
    truth = np.asarray(true_candidate_index, dtype=np.int16)
    if orders.ndim != 3 or truth.shape != (orders.shape[0],):
        raise ValueError("rank/truth shape mismatch")
    matches = orders == truth[:, None, None]
    if not np.all(matches.sum(axis=2) == 1):
        raise ValueError("every rank order must contain the truth-nearest candidate once")
    return matches.argmax(axis=2).astype(np.uint8)


def summarize_rank_positions(
    rank_positions: np.ndarray,
    wells: np.ndarray,
    hidden_like: pd.DataFrame,
    *,
    fold: int | str,
) -> pd.DataFrame:
    positions = np.asarray(rank_positions, dtype=np.uint8)
    wells = np.asarray(wells, dtype=str)
    if positions.shape != (len(wells), len(RANK_VARIANTS)):
        raise ValueError("rank positions and well identities are not aligned")
    hidden_index = hidden_like.set_index("well_id")
    surfaces: dict[str, np.ndarray] = {"all": np.ones(len(wells), dtype=bool)}
    for role_column in (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ):
        role = pd.Series(wells).map(hidden_index[role_column]).fillna("missing").to_numpy(str)
        surfaces[role_column] = role == "valid"
    rows: list[dict[str, Any]] = []
    for surface, mask in surfaces.items():
        if not mask.any():
            continue
        for variant_index, variant in enumerate(RANK_VARIANTS):
            selected = positions[mask, variant_index].astype(np.float64)
            rows.append(
                {
                    "fold": fold,
                    "surface": surface,
                    "variant": variant,
                    "rows": int(mask.sum()),
                    "wells": int(pd.Series(wells[mask]).nunique()),
                    "mrr": float(np.mean(1.0 / (selected + 1.0))),
                    "top3_rate": float(np.mean(selected < 3)),
                    "mean_rank": float(np.mean(selected + 1.0)),
                }
            )
    return pd.DataFrame(rows)


def load_outer_valid_truth_after_freeze(
    bank: CandidateBank,
    context: TargetFreeContext,
    parent: ParentContract,
    frozen: FrozenFold,
    config: Mapping[str, Any],
) -> FoldReadout:
    verify_frozen_fold(frozen, bank)
    valid_keys = bank.keys.iloc[frozen.valid_positions].reset_index(drop=True)
    if not np.all(valid_keys["well"].map(parent.fold_by_well).to_numpy() == frozen.fold):
        raise ValueError("late truth loader received a non-validation well")
    truth_column = str(get_nested(config, "data.raw_columns.truth"))
    true_tvt = np.full(len(valid_keys), np.nan, dtype=np.float64)
    for well, local_index in valid_keys.groupby("well", sort=True).indices.items():
        positions = np.asarray(local_index, dtype=np.int64)
        row_idx = valid_keys.iloc[positions]["well_row_idx"].to_numpy(np.int64)
        raw_truth = pd.to_numeric(
            pd.read_csv(context.wells[str(well)].horizontal_path, usecols=[truth_column])[
                truth_column
            ],
            errors="coerce",
        ).to_numpy(np.float64)
        true_tvt[positions] = raw_truth[row_idx]
    if not np.isfinite(true_tvt).all():
        raise ValueError("outer-valid suffix truth contains nonfinite values")
    candidate_values = np.asarray(bank.values[frozen.valid_positions], dtype=np.float64)
    true_candidate = np.argmin(np.abs(candidate_values - true_tvt[:, None]), axis=1).astype(
        np.uint8
    )
    rank_orders = np.memmap(
        frozen.rank_path,
        mode="r",
        dtype="uint8",
        shape=(len(valid_keys), len(RANK_VARIANTS), len(bank.candidate_ids)),
    )
    rank_positions = true_candidate_rank_positions(rank_orders, true_candidate)
    wells = valid_keys["well"].astype(str).to_numpy()
    metrics = summarize_rank_positions(rank_positions, wells, context.hidden_like, fold=frozen.fold)
    return FoldReadout(
        fold=frozen.fold,
        valid_positions=frozen.valid_positions,
        wells=wells,
        rank_positions=rank_positions,
        true_candidate_index=true_candidate,
        metrics=metrics,
    )


def metric_value(
    metrics: pd.DataFrame,
    *,
    fold: int | str,
    surface: str,
    variant: str,
    metric: str,
) -> float:
    selected = metrics[
        (metrics["fold"].astype(str) == str(fold))
        & (metrics["surface"] == surface)
        & (metrics["variant"] == variant)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"metric row is not unique: fold={fold} surface={surface} variant={variant}"
        )
    return float(selected.iloc[0][metric])


def build_fold_metric_summary(rank_metrics: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in sorted(
        int(value) for value in rank_metrics["fold"].unique() if str(value) != "pooled"
    ):
        baseline_mrr = metric_value(
            rank_metrics, fold=fold, surface="all", variant="baseline", metric="mrr"
        )
        real_mrr = metric_value(
            rank_metrics, fold=fold, surface="all", variant="real", metric="mrr"
        )
        shuffled_mrr = metric_value(
            rank_metrics,
            fold=fold,
            surface="all",
            variant="group_label_shuffle",
            metric="mrr",
        )
        shifted_mrr = metric_value(
            rank_metrics,
            fold=fold,
            surface="all",
            variant="tvt_shift_matched_count",
            metric="mrr",
        )
        baseline_top3 = metric_value(
            rank_metrics,
            fold=fold,
            surface="all",
            variant="baseline",
            metric="top3_rate",
        )
        real_top3 = metric_value(
            rank_metrics, fold=fold, surface="all", variant="real", metric="top3_rate"
        )
        fold_fallback = fallback[fallback["fold"] == fold]
        group_conditional_rate = float(
            fold_fallback.loc[fold_fallback["fallback_level"] == "group_conditional", "rate"].iloc[
                0
            ]
        )
        rows.append(
            {
                "fold": fold,
                "baseline_mrr": baseline_mrr,
                "real_mrr": real_mrr,
                "mrr_gain": real_mrr - baseline_mrr,
                "baseline_top3_rate": baseline_top3,
                "real_top3_rate": real_top3,
                "top3_gain": real_top3 - baseline_top3,
                "real_minus_shuffled_mrr": real_mrr - shuffled_mrr,
                "real_minus_tvt_shift_mrr": real_mrr - shifted_mrr,
                "lookup_fallback_rate": 1.0 - group_conditional_rate,
                "improved_both": bool(real_mrr > baseline_mrr and real_top3 > baseline_top3),
            }
        )
    return pd.DataFrame(rows)


def evaluate_promotion_gate(
    rank_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    fallback: pd.DataFrame,
    frozen_folds: Sequence[FrozenFold],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "promotion_gates")
    real_mrr = metric_value(
        rank_metrics, fold="pooled", surface="all", variant="real", metric="mrr"
    )
    baseline_mrr = metric_value(
        rank_metrics,
        fold="pooled",
        surface="all",
        variant="baseline",
        metric="mrr",
    )
    shuffled_mrr = metric_value(
        rank_metrics,
        fold="pooled",
        surface="all",
        variant="group_label_shuffle",
        metric="mrr",
    )
    shifted_mrr = metric_value(
        rank_metrics,
        fold="pooled",
        surface="all",
        variant="tvt_shift_matched_count",
        metric="mrr",
    )
    real_top3 = metric_value(
        rank_metrics,
        fold="pooled",
        surface="all",
        variant="real",
        metric="top3_rate",
    )
    baseline_top3 = metric_value(
        rank_metrics,
        fold="pooled",
        surface="all",
        variant="baseline",
        metric="top3_rate",
    )
    folds_improved = int(fold_metrics["improved_both"].sum())
    total_lookups = int(fallback["lookups"].sum())
    conditional_lookups = int(
        fallback.loc[fallback["fallback_level"] == "group_conditional", "lookups"].sum()
    )
    fallback_rate = 1.0 - conditional_lookups / total_lookups
    hidden_checks: dict[str, bool] = {}
    for surface in (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ):
        hidden_checks[surface] = bool(
            metric_value(
                rank_metrics,
                fold="pooled",
                surface=surface,
                variant="real",
                metric="mrr",
            )
            >= metric_value(
                rank_metrics,
                fold="pooled",
                surface=surface,
                variant="baseline",
                metric="mrr",
            )
            and metric_value(
                rank_metrics,
                fold="pooled",
                surface=surface,
                variant="real",
                metric="top3_rate",
            )
            >= metric_value(
                rank_metrics,
                fold="pooled",
                surface=surface,
                variant="baseline",
                metric="top3_rate",
            )
        )
    mrr_gain = real_mrr - baseline_mrr
    top3_gain = real_top3 - baseline_top3
    real_minus_shuffled = real_mrr - shuffled_mrr
    checks = {
        "truth_nearest_mrr_gain": bool(mrr_gain >= float(gates["truth_nearest_mrr_gain_min"])),
        "truth_nearest_top3_gain": bool(top3_gain >= float(gates["truth_nearest_top3_gain_min"])),
        "real_minus_shuffled_mrr": bool(
            real_minus_shuffled >= float(gates["real_minus_shuffled_mrr_min"])
        ),
        "folds_improved": bool(folds_improved >= int(gates["folds_improved_min"])),
        "hidden_like_nonregression": bool(all(hidden_checks.values())),
        "lookup_fallback_rate": bool(fallback_rate <= float(gates["lookup_fallback_rate_max"])),
        "outer_valid_truth_before_freeze_zero": bool(
            all(
                json.loads(item.freeze_path.read_text()).get(
                    "outer_valid_truth_access_count_before_freeze"
                )
                == 0
                for item in frozen_folds
            )
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mrr_gain": mrr_gain,
        "top3_gain": top3_gain,
        "real_minus_shuffled_mrr": real_minus_shuffled,
        "real_minus_tvt_shift_mrr": real_mrr - shifted_mrr,
        "folds_improved": folds_improved,
        "lookup_fallback_rate": fallback_rate,
        "baseline_mrr": baseline_mrr,
        "real_mrr": real_mrr,
        "baseline_top3_rate": baseline_top3,
        "real_top3_rate": real_top3,
        "hidden_like_checks": hidden_checks,
        "candidate_bank": "exp293_exp263_stage1_deployable12",
        "candidate_count": len(EXPECTED_CANDIDATE_ORDER),
    }


# %% [markdown]
# ## 8. Setup, execution orchestration, and generated artifacts

# %%
CONFIG: dict[str, Any] = {}
ARTIFACTS_DIR: Path | None = None
WORK_DIR: Path | None = None

if EXECUTE_NOTEBOOK:
    CONFIG = read_yaml(find_config_path())
    validate_scientific_contract(CONFIG)
    ARTIFACTS_DIR = runtime_artifacts_dir()
    WORK_DIR = runtime_work_dir()
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "candidate_bank": get_nested(CONFIG, "candidate_bank.name"),
                "candidate_count": len(EXPECTED_CANDIDATE_ORDER),
                "execution_contract": get_nested(CONFIG, "execution_contract"),
                "parent_gate_override": get_nested(CONFIG, "validation.parent_gate_override"),
                "inference_enabled": False,
                "submission_enabled": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

# %%
if EXECUTE_NOTEBOOK:
    assert ARTIFACTS_DIR is not None and WORK_DIR is not None
    started = datetime.now(UTC)
    BANK = build_candidate_bank(CONFIG, WORK_DIR)
    PARENT = resolve_parent_contract(CONFIG)
    TARGET_FREE = build_target_free_context(BANK, PARENT, CONFIG, WORK_DIR)
    print(
        json.dumps(
            {
                "candidate_rows": len(BANK.keys),
                "candidate_wells": int(BANK.keys["well"].nunique()),
                "candidate_bank_content_sha256": BANK.candidate_content_sha256,
                "parent_summary_sha256": sha256_file(PARENT.summary_path),
                "target_free_context_wells": len(TARGET_FREE.wells),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

# %%
if EXECUTE_NOTEBOOK:
    FROZEN_FOLDS: list[FrozenFold] = []
    READOUTS: list[FoldReadout] = []
    for FOLD in range(int(get_nested(CONFIG, "validation.n_folds"))):
        print(f"fold={FOLD} target-free table and candidate rank freeze", flush=True)
        FROZEN = score_fold_target_free(BANK, TARGET_FREE, PARENT, FOLD, CONFIG, WORK_DIR)
        FROZEN_FOLDS.append(FROZEN)
        print(
            f"fold={FOLD} late outer-valid truth join after freeze_sha={FROZEN.freeze_sha256}",
            flush=True,
        )
        READOUTS.append(
            load_outer_valid_truth_after_freeze(BANK, TARGET_FREE, PARENT, FROZEN, CONFIG)
        )

# %%
if EXECUTE_NOTEBOOK:
    FOLD_RANK_METRICS = pd.concat([readout.metrics for readout in READOUTS], ignore_index=True)
    ALL_WELLS = np.concatenate([readout.wells for readout in READOUTS])
    ALL_RANK_POSITIONS = np.concatenate([readout.rank_positions for readout in READOUTS], axis=0)
    POOLED_RANK_METRICS = summarize_rank_positions(
        ALL_RANK_POSITIONS, ALL_WELLS, TARGET_FREE.hidden_like, fold="pooled"
    )
    RANK_METRICS = pd.concat([FOLD_RANK_METRICS, POOLED_RANK_METRICS], ignore_index=True)
    BIN_EDGES = pd.concat([item.bin_edges for item in FROZEN_FOLDS], ignore_index=True)
    EMISSION_TABLES = pd.concat([item.emission_tables for item in FROZEN_FOLDS], ignore_index=True)
    FALLBACK = pd.concat([item.fallback for item in FROZEN_FOLDS], ignore_index=True)
    FOLD_METRICS = build_fold_metric_summary(RANK_METRICS, FALLBACK)
    HIDDEN_LIKE_METRICS = RANK_METRICS[RANK_METRICS["surface"] != "all"].reset_index(drop=True)
    PROMOTION = evaluate_promotion_gate(RANK_METRICS, FOLD_METRICS, FALLBACK, FROZEN_FOLDS, CONFIG)
    print(json.dumps(PROMOTION, indent=2, sort_keys=True), flush=True)

# %%
if EXECUTE_NOTEBOOK:
    INPUT_MANIFEST = pd.DataFrame(
        BANK.input_evidence + PARENT.input_evidence + TARGET_FREE.input_evidence
    )
    FEATURE_SCHEMA = pd.DataFrame(
        [
            {
                "feature": "candidate_tvt",
                "scope": "target_free",
                "shape": "rows_x_12",
                "description": "exp293 deployable12 fixed candidate values",
            },
            {
                "feature": "candidate_typewell_gr",
                "scope": "target_free",
                "shape": "rows_x_12",
                "description": "Type Well GR interpolated at each fixed candidate TVT",
            },
            {
                "feature": "candidate_abs_gradient",
                "scope": "target_free",
                "shape": "rows_x_12",
                "description": "absolute Type Well GR gradient at each candidate TVT",
            },
            {
                "feature": "horizontal_gr",
                "scope": "target_free",
                "shape": "rows",
                "description": "linear-imputed observed horizontal GR",
            },
            {
                "feature": "horizontal_missing_flag",
                "scope": "target_free",
                "shape": "rows",
                "description": "raw horizontal GR missingness before imputation",
            },
            {
                "feature": "group_id",
                "scope": "target_free",
                "shape": "well",
                "description": "exp311 native_overlap_1 membership",
            },
            {
                "feature": "rank_orders",
                "scope": "frozen_before_outer_valid_truth",
                "shape": "rows_x_4_x_12",
                "description": "stable candidate order for baseline, real, and two controls",
            },
            {
                "feature": "true_candidate_index",
                "scope": "late_readout_only",
                "shape": "rows",
                "description": "truth-nearest deployable12 label; never persisted as a prediction",
            },
        ]
    )
    MANIFESTS: dict[str, Any] = {}
    csv_outputs = {
        "input_manifest": INPUT_MANIFEST,
        "bin_edges": BIN_EDGES,
        "emission_table": EMISSION_TABLES,
        "lookup_fallback": FALLBACK,
        "rank_metrics": RANK_METRICS,
        "fold_metrics": FOLD_METRICS,
        "hidden_like_metrics": HIDDEN_LIKE_METRICS,
        "feature_schema": FEATURE_SCHEMA,
    }
    for NAME, FRAME in csv_outputs.items():
        PATH = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_{NAME}.csv"
        MANIFESTS[NAME] = write_csv(FRAME, PATH)
    CANDIDATE_MANIFEST_PATH = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_candidate_bank_manifest.json"
    CANDIDATE_MANIFEST = {
        "name": get_nested(CONFIG, "candidate_bank.name"),
        "source_contract": get_nested(CONFIG, "candidate_bank.source_contract"),
        "candidate_ids": list(BANK.candidate_ids),
        "rows": len(BANK.keys),
        "wells": int(BANK.keys["well"].nunique()),
        "key_content_sha256": BANK.key_content_sha256,
        "candidate_content_sha256": BANK.candidate_content_sha256,
        "exp263_manifest_path": str(BANK.manifest_path),
        "exp263_manifest_sha256": sha256_file(BANK.manifest_path),
        "formulas": {
            "pairs": get_nested(CONFIG, "candidate_bank.pairs"),
            "fixed_formula": get_nested(CONFIG, "candidate_bank.fixed_formula"),
        },
        "candidate_values_changed": False,
        "candidate_generation_runs": 0,
        "decoder_runs": 0,
        "formula_sample_parity": BANK.sample_parity.to_dict(orient="records"),
        "formula_sample_parity_passed": bool(BANK.sample_parity["passed"].all()),
    }
    write_json(CANDIDATE_MANIFEST_PATH, CANDIDATE_MANIFEST)
    MANIFESTS["candidate_bank_manifest"] = {
        "path": str(CANDIDATE_MANIFEST_PATH),
        "rows": len(BANK.keys),
        "columns": len(BANK.candidate_ids),
        "raw_sha256": sha256_file(CANDIDATE_MANIFEST_PATH),
        "content_sha256": json_sha256(CANDIDATE_MANIFEST),
    }

# %%
if EXECUTE_NOTEBOOK:
    runtime_seconds = (datetime.now(UTC) - started).total_seconds()
    SUMMARY_PATH = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"
    SUMMARY = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed" if PROMOTION["passed"] else "completed_gate_failed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "route": "pf_beam",
        "execution_contract": get_nested(CONFIG, "execution_contract"),
        "parent_gate_override": get_nested(CONFIG, "validation.parent_gate_override"),
        "parent_exp311": {
            "summary_sha256": sha256_file(PARENT.summary_path),
            "promotion_passed": False,
            "retained_failures": get_nested(
                CONFIG, "validation.parent_gate_override.retained_failures"
            ),
        },
        "candidate_bank": CANDIDATE_MANIFEST,
        "promotion": PROMOTION,
        "fold_freeze": [
            {
                "fold": item.fold,
                "freeze_sha256": item.freeze_sha256,
                "rank_order_sha256": item.rank_sha256,
                "emission_table_content_sha256": item.table_sha256,
                "outer_valid_truth_access_count_before_freeze": 0,
            }
            for item in FROZEN_FOLDS
        ],
        "artifact_manifests": MANIFESTS,
        "forbidden_outputs": {
            "models": 0,
            "boosters": 0,
            "decoders": 0,
            "candidate_generators": 0,
            "predictions": 0,
            "submission": False,
        },
        "runtime": {
            "cpu_only": True,
            "internet_enabled": False,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    write_json(SUMMARY_PATH, SUMMARY)
    METRICS = {
        "experiment": EXPERIMENT_NAME,
        "status": SUMMARY["status"],
        "route": "pf_beam",
        "metric": "truth_nearest_candidate_rank",
        "cv": PROMOTION,
        "runtime_seconds": runtime_seconds,
        "public_lb": None,
        "private_lb": None,
        "notes": "No candidate generation, model, decoder, inference, or submission was run.",
        "summary_path": str(SUMMARY_PATH),
    }
    write_json(runtime_metrics_path(), METRICS)
    print("generated artifacts", flush=True)
    for NAME, MANIFEST in MANIFESTS.items():
        print(NAME, MANIFEST["path"], flush=True)
    print("summary", SUMMARY_PATH, flush=True)
