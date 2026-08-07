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
# # exp303 multiscale K stability selectability readout
#
# This CPU-only diagnostic reads the frozen K=12/K=16/K=24 OOF predictions
# and corrected exp264 Stage C v6 candidate scores.  It freezes every
# target-free feature, empirical-percentile score, and H512 block before a
# separate loader may read TVT.  It trains no model and changes no prediction.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input contracts
# 3. Target-free input loaders
# 4. Fixed row features and outer-train score
# 5. Truth-free freeze
# 6. Post-freeze truth loader
# 7. Fixed block readout and guards
# 8. Artifacts and orchestration

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

EXP_NAME = "exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264"
OUTPUT_PREFIX = EXP_NAME
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
K_VALUES = (12, 16, 24)
PRIMARY_CANDIDATES = (
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
)
ALL_STAGE_C_CANDIDATES = PRIMARY_CANDIDATES + ("exp226_w500_50_50",)
PREDICTION_COLUMNS = ("exp226_k12_prediction", "exp226_k16_prediction", "exp226_k24_prediction")
PRIMARY_COMPONENTS = (
    "level_spread_ft",
    "slope_spread_ft",
    "boundary_weighted_jump_spread_ft",
)
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "target_tvt",
    "error",
    "abs_error",
    "actual_abs_error",
    "actual_within10",
    "oracle_candidate",
    "oracle_rank",
    "bad_well_label",
}


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "experiments" / EXP_NAME / "config.yaml").exists():
            return candidate
    return cwd


def experiment_dir() -> Path:
    root = project_root()
    candidate = root / "experiments" / EXP_NAME
    return candidate if candidate.exists() else Path.cwd().resolve()


def runtime_artifacts_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/artifacts")
    return experiment_dir() / "artifacts"


def load_config() -> dict[str, Any]:
    path = experiment_dir() / "config.yaml"
    if not path.exists():
        path = Path.cwd() / "config.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def get_nested(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        value = value[key]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    root = project_root()
    found: dict[str, Path] = {}
    for raw in map(str, patterns):
        path = Path(raw)
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            found.setdefault(str(direct.resolve()), direct)
            continue
        searches = [raw]
        if not path.is_absolute():
            searches.append(str(root / raw))
        for search in searches:
            for match in glob.glob(search, recursive=True):
                candidate = Path(match)
                if candidate.exists():
                    found.setdefault(str(candidate.resolve()), candidate)
    return list(found.values())


def resolve_file(
    patterns: Sequence[str],
    *,
    label: str,
    expected_file_sha256: str | None = None,
    expected_decompressed_sha256: str | None = None,
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    evidence: list[str] = []
    for path in candidates:
        if path.stat().st_size == 0:
            evidence.append(f"{path}:empty")
            continue
        if expected_file_sha256 is not None:
            actual = sha256_file(path)
            evidence.append(f"{path}:file={actual}")
            if actual != expected_file_sha256:
                continue
        if expected_decompressed_sha256 is not None:
            actual = sha256_decompressed_gzip(path)
            evidence.append(f"{path}:decompressed={actual}")
            if actual != expected_decompressed_sha256:
                continue
        return path
    raise FileNotFoundError(
        f"Could not resolve {label} with the fixed SHA contract: {evidence[:8]}"
    )


def resolve_directory(patterns: Sequence[str], *, label: str) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_dir()]
    for path in candidates:
        if list(path.glob("*__horizontal_well.csv")):
            return path
    raise FileNotFoundError(f"Could not resolve {label}: {candidates[:8]}")


def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    present = set(map(str, columns)) & FORBIDDEN_PRE_FREEZE_COLUMNS
    if present:
        raise ValueError(f"truth/error columns are forbidden before freeze: {sorted(present)}")


# %% [markdown]
# ## 2. Configuration and input contracts


# %%
def validate_execution_contract(config: Mapping[str, Any]) -> None:
    dependencies = get_nested(config, "dependencies.status")
    if not all(bool(value) for value in dependencies.values()):
        raise ValueError(f"exp303 dependencies are not all satisfied: {dependencies}")
    execution = config["execution"]
    expected = {
        "implementation": True,
        "implementation_authorized": True,
        "fixed_readout_variants_if_implemented": 1,
        "outer_evaluation_folds_if_implemented": 5,
        "lightgbm_config_count": 0,
        "trained_fold_count": 0,
        "total_boosters": 0,
        "parent_or_control_retraining": False,
        "candidate_regeneration": False,
        "gpu": False,
        "inference": False,
        "submission": False,
    }
    changed = {
        key: (execution.get(key), value)
        for key, value in expected.items()
        if execution.get(key) != value
    }
    if changed:
        raise ValueError(f"execution contract changed: {changed}")
    if not execution.get("kaggle_execution_authorized"):
        raise ValueError("Kaggle execution is not authorized")
    if not config["runtime"]["kaggle"]["train_run_on_push"]:
        raise ValueError("approved Kaggle execution requires train_run_on_push=true")


# %% [markdown]
# ## 3. Target-free input loaders


# %%
def read_exp302_variant(path: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "id": "string",
            "well": "string",
            "well_row_idx": "int32",
            "outer_fold": "int64",
            "variant": "string",
            "candidate_tvt": "float64",
        },
    )
    required = ["id", "well", "well_row_idx", "outer_fold", "variant", "candidate_tvt"]
    if frame.columns.tolist() != required:
        raise ValueError(f"{variant} schema changed: {frame.columns.tolist()}")
    if not frame["variant"].eq(variant).all():
        raise ValueError(f"{variant} file contains a different variant")
    if frame["id"].duplicated().any() or not np.isfinite(frame["candidate_tvt"]).all():
        raise ValueError(f"{variant} identity/finite contract failed")
    return frame


def verify_exp302_prediction_content(frame: pd.DataFrame, expected_sha256: str) -> str:
    actual = frame_content_sha256(
        frame,
        ["id", "well", "well_row_idx", "outer_fold", "variant", "candidate_tvt"],
    )
    if actual != expected_sha256:
        raise ValueError(f"exp302 prediction content SHA changed: {actual}")
    return actual


def stream_exp302_variant(
    path: Path,
    variant: str,
    expected_parsed_sha256: str | None,
    consume: Any,
    chunk_rows: int = 250_000,
) -> tuple[int, str]:
    required = ["id", "well", "well_row_idx", "outer_fold", "variant", "candidate_tvt"]
    digest = hashlib.sha256()
    total = 0
    header_written = False
    iterator = pd.read_csv(
        path,
        dtype={
            "id": "string",
            "well": "string",
            "well_row_idx": "int32",
            "outer_fold": "int64",
            "variant": "string",
            "candidate_tvt": "float64",
        },
        chunksize=int(chunk_rows),
    )
    for chunk in iterator:
        if chunk.columns.tolist() != required:
            raise ValueError(f"{variant} schema changed: {chunk.columns.tolist()}")
        if not chunk["variant"].eq(variant).all():
            raise ValueError(f"{variant} file contains a different variant")
        if chunk["id"].duplicated().any() or not np.isfinite(chunk["candidate_tvt"]).all():
            raise ValueError(f"{variant} chunk identity/finite contract failed")
        normalized = _normalize_frame_for_hash(chunk)
        if not header_written:
            digest.update("|".join(normalized.columns).encode())
            digest.update("|".join(str(dtype) for dtype in normalized.dtypes).encode())
            header_written = True
        hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
        consume(chunk, total)
        total += len(chunk)
    actual = digest.hexdigest()
    if expected_parsed_sha256 is not None and actual != expected_parsed_sha256:
        raise ValueError(f"{variant} prediction content SHA changed: {actual}")
    return total, actual


def read_k16_target_free(path: Path) -> pd.DataFrame:
    allowed = ["well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"]
    assert_no_forbidden_columns(allowed)
    frame = pd.read_csv(
        path,
        usecols=allowed,
        dtype={
            "well_id": "string",
            "row_idx": "int32",
            "suffix_offset": "int32",
            "tvt_pred": "float64",
            "fold": "int64",
        },
    )
    frame["id"] = frame["well_id"].astype(str) + "_" + frame["row_idx"].astype(str)
    if frame["id"].duplicated().any() or not np.isfinite(frame["tvt_pred"]).all():
        raise ValueError("K16 target-free projection failed identity/finite checks")
    return frame


@dataclass
class TargetFreeInputs:
    rows: pd.DataFrame
    source_paths: dict[str, Path]
    source_evidence: dict[str, Any]


def load_multiscale_predictions(config: Mapping[str, Any]) -> TargetFreeInputs:
    prediction_config = get_nested(config, "data.exp302_predictions")
    expected = prediction_config["expected"]
    source_paths: dict[str, Path] = {}
    source_evidence: dict[str, Any] = {}
    freeze_spec = prediction_config["freeze_manifest"]
    exp302_freeze_path = resolve_file(
        freeze_spec["patterns"],
        label="exp302 target-free freeze manifest",
        expected_file_sha256=str(freeze_spec["expected_file_sha256"]),
    )
    exp302_freeze = json.loads(exp302_freeze_path.read_text())
    if (
        not bool(exp302_freeze.get("frozen"))
        or int(exp302_freeze.get("evaluation_truth_access_count_before_freeze", -1)) != 0
    ):
        raise ValueError("exp302 freeze manifest is not a target-free frozen contract")
    for name in ("exp226_k12", "exp226_k24"):
        declared = exp302_freeze["variants"][name]
        spec = expected[name]
        checks = {
            "rows": int(declared["rows"]) == EXPECTED_ROWS,
            "content": str(declared["prediction_content_sha256"])
            == str(spec["prediction_content_sha256"]),
            "decompressed": str(declared["prediction_decompressed_sha256"])
            == str(spec["decompressed_sha256"]),
            "truth_state": int(declared["outer_valid_truth_state_count"]) == 0,
        }
        if not all(checks.values()):
            raise ValueError(f"exp302 frozen {name} declaration changed: {checks}")
    source_paths["exp302_freeze_manifest"] = exp302_freeze_path
    source_evidence["exp302_freeze_manifest"] = {
        "path": str(exp302_freeze_path),
        "file_sha256": sha256_file(exp302_freeze_path),
        "evaluation_truth_access_count_before_freeze": 0,
        "prediction_content_declarations_verified": True,
    }
    k12_spec = expected["exp226_k12"]
    k12_path = resolve_file(
        k12_spec["patterns"],
        label="exp226_k12",
        expected_decompressed_sha256=str(k12_spec["decompressed_sha256"]),
    )
    well_names: list[str] = []
    well_to_code: dict[str, int] = {}
    closed_wells: set[str] = set()
    current_well: str | None = None
    code_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    suffix_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    pred12_parts: list[np.ndarray] = []
    rows_seen_by_well: dict[str, int] = {}
    first_row_by_well: dict[str, int] = {}

    def consume_k12(chunk: pd.DataFrame, _: int) -> None:
        nonlocal current_well
        names = chunk["well"].astype(str).to_numpy()
        ids = chunk["id"].astype(str).to_numpy()
        row_index = chunk["well_row_idx"].to_numpy(np.int32)
        if not np.array_equal(
            ids, np.char.add(np.char.add(names.astype(str), "_"), row_index.astype(str))
        ):
            raise ValueError("K12 id is not well_row_idx identity")
        codes = np.empty(len(chunk), np.int16)
        suffix = np.empty(len(chunk), np.int32)
        run_starts = np.flatnonzero(np.r_[True, names[1:] != names[:-1]])
        run_ends = np.r_[run_starts[1:], len(names)]
        for start, end in zip(run_starts, run_ends, strict=True):
            well = str(names[start])
            if current_well != well:
                if current_well is not None:
                    closed_wells.add(current_well)
                if well in closed_wells:
                    raise ValueError(f"K12 well is not contiguous: {well}")
                current_well = well
            if well not in well_to_code:
                well_to_code[well] = len(well_names)
                well_names.append(well)
                rows_seen_by_well[well] = 0
                first_row_by_well[well] = int(row_index[start])
            code = well_to_code[well]
            base = rows_seen_by_well[well]
            codes[start:end] = code
            suffix[start:end] = np.arange(base, base + end - start, dtype=np.int32)
            expected_rows = first_row_by_well[well] + suffix[start:end]
            if not np.array_equal(row_index[start:end], expected_rows):
                raise ValueError(f"K12 row index is not contiguous: {well}")
            rows_seen_by_well[well] = base + end - start
        code_parts.append(codes)
        row_parts.append(row_index)
        suffix_parts.append(suffix)
        fold_parts.append(chunk["outer_fold"].to_numpy(np.int8))
        pred12_parts.append(chunk["candidate_tvt"].to_numpy(np.float64))

    k12_rows, k12_content_sha = stream_exp302_variant(
        k12_path,
        "exp226_k12",
        None,
        consume_k12,
    )
    well_code = np.concatenate(code_parts)
    row_index = np.concatenate(row_parts)
    suffix_offset = np.concatenate(suffix_parts)
    exp226_fold = np.concatenate(fold_parts)
    pred12 = np.concatenate(pred12_parts)
    if k12_rows != EXPECTED_ROWS or len(well_names) != EXPECTED_WELLS:
        raise ValueError("K12 canonical row/well coverage changed")
    source_paths["exp226_k12"] = k12_path
    source_evidence["exp226_k12"] = {
        "path": str(k12_path),
        "bytes": k12_path.stat().st_size,
        "decompressed_sha256": sha256_decompressed_gzip(k12_path),
        "source_manifest_prediction_content_sha256": str(k12_spec["prediction_content_sha256"]),
        "persisted_csv_parsed_content_sha256": k12_content_sha,
        "loaded_columns": ["id", "well", "well_row_idx", "outer_fold", "variant", "candidate_tvt"],
        "streamed_without_retaining_unique_id_strings": True,
    }

    k24_spec = expected["exp226_k24"]
    k24_path = resolve_file(
        k24_spec["patterns"],
        label="exp226_k24",
        expected_decompressed_sha256=str(k24_spec["decompressed_sha256"]),
    )
    pred24 = np.full(EXPECTED_ROWS, np.nan, np.float64)

    def consume_k24(chunk: pd.DataFrame, start: int) -> None:
        stop = start + len(chunk)
        names = chunk["well"].astype(str)
        codes = pd.Categorical(names, categories=well_names, ordered=True).codes
        if np.any(codes < 0):
            raise ValueError("K24 contains an unknown well")
        if not np.array_equal(codes.astype(np.int16), well_code[start:stop]):
            raise ValueError("K12/K24 well order differs")
        if not np.array_equal(chunk["well_row_idx"].to_numpy(np.int32), row_index[start:stop]):
            raise ValueError("K12/K24 row identity differs")
        if not np.array_equal(chunk["outer_fold"].to_numpy(np.int8), exp226_fold[start:stop]):
            raise ValueError("K12/K24 exp226 fold identity differs")
        pred24[start:stop] = chunk["candidate_tvt"].to_numpy(np.float64)

    k24_rows, k24_content_sha = stream_exp302_variant(
        k24_path,
        "exp226_k24",
        None,
        consume_k24,
    )
    if k24_rows != EXPECTED_ROWS or not np.isfinite(pred24).all():
        raise ValueError("K24 coverage changed")
    source_paths["exp226_k24"] = k24_path
    source_evidence["exp226_k24"] = {
        "path": str(k24_path),
        "bytes": k24_path.stat().st_size,
        "decompressed_sha256": sha256_decompressed_gzip(k24_path),
        "source_manifest_prediction_content_sha256": str(k24_spec["prediction_content_sha256"]),
        "persisted_csv_parsed_content_sha256": k24_content_sha,
        "loaded_columns": ["id", "well", "well_row_idx", "outer_fold", "variant", "candidate_tvt"],
        "streamed_without_retaining_unique_id_strings": True,
    }

    spec16 = expected["exp226_k16"]
    path16 = resolve_file(
        spec16["patterns"],
        label="exp226_k16",
        expected_decompressed_sha256=str(spec16["decompressed_sha256"]),
    )
    counts = np.bincount(well_code, minlength=len(well_names)).astype(np.int64)
    well_starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    pred16 = np.full(EXPECTED_ROWS, np.nan, np.float64)
    fold16 = np.full(EXPECTED_ROWS, -1, np.int8)
    coverage16 = np.zeros(EXPECTED_ROWS, bool)
    target_free_digest16 = hashlib.sha256()
    target_free_digest16.update(b"well_id|row_idx|suffix_offset|tvt_pred|fold")
    target_free_digest16.update(b"object|int32|int32|float64|int64")
    iterator16 = pd.read_csv(
        path16,
        usecols=["well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"],
        dtype={
            "well_id": "string",
            "row_idx": "int32",
            "suffix_offset": "int32",
            "tvt_pred": "float64",
            "fold": "int64",
        },
        chunksize=250_000,
    )
    k16_rows = 0
    for chunk in iterator16:
        assert_no_forbidden_columns(chunk.columns)
        normalized = _normalize_frame_for_hash(chunk)
        hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
        target_free_digest16.update(
            hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes()
        )
        codes = pd.Categorical(
            chunk["well_id"].astype(str), categories=well_names, ordered=True
        ).codes
        if np.any(codes < 0):
            raise ValueError("K16 contains an unknown well")
        suffix = chunk["suffix_offset"].to_numpy(np.int32)
        if np.any(suffix < 0) or np.any(suffix >= counts[codes]):
            raise ValueError("K16 suffix offset leaves its canonical well")
        positions = well_starts[codes] + suffix
        if (
            np.any(positions < 0)
            or np.any(positions >= EXPECTED_ROWS)
            or coverage16[positions].any()
        ):
            raise ValueError("K16 alignment overlaps or leaves canonical range")
        if not np.array_equal(row_index[positions], chunk["row_idx"].to_numpy(np.int32)):
            raise ValueError("K16 well-row identity differs from K12/K24")
        if not np.array_equal(well_code[positions], codes.astype(np.int16)):
            raise ValueError("K16 well identity differs from K12/K24")
        pred16[positions] = chunk["tvt_pred"].to_numpy(np.float64)
        fold16[positions] = chunk["fold"].to_numpy(np.int8)
        coverage16[positions] = True
        k16_rows += len(chunk)
    if k16_rows != EXPECTED_ROWS or not coverage16.all() or not np.isfinite(pred16).all():
        raise ValueError("K16 coverage differs from K12/K24")
    rows = pd.DataFrame(
        {
            "well_id": pd.Categorical.from_codes(well_code, categories=well_names, ordered=True),
            "row_idx": row_index,
            "suffix_offset": suffix_offset,
            "exp226_fold": exp226_fold,
            "exp226_k12_prediction": pred12,
            "exp226_k16_prediction": pred16,
            "exp226_k24_prediction": pred24,
        }
    )
    if len(rows) != EXPECTED_ROWS or rows["well_id"].nunique() != EXPECTED_WELLS:
        raise ValueError("multiscale row/well coverage changed")
    source_paths["exp226_k16"] = path16
    source_evidence["exp226_k16"] = {
        "path": str(path16),
        "bytes": path16.stat().st_size,
        "decompressed_sha256": sha256_decompressed_gzip(path16),
        "target_free_source_order_content_sha256": target_free_digest16.hexdigest(),
        "aligned_prediction_array_sha256": sha256_array(pred16),
        "aligned_fold_array_sha256": sha256_array(fold16),
        "loaded_columns": ["well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"],
        "forbidden_columns_loaded": [],
    }
    return TargetFreeInputs(rows=rows, source_paths=source_paths, source_evidence=source_evidence)


def attach_stage_c_selected_hard(
    rows: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "id",
        "well",
        "well_row_idx",
        "md_since",
        "candidate_id",
        "candidate_tvt",
        "pred_abs_error",
        "outer_fold",
        "downstream_outer_fold",
        "nested_model_count",
    ]
    assert_no_forbidden_columns(columns)
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != EXPECTED_ROWS * len(ALL_STAGE_C_CANDIDATES):
        raise ValueError("Stage C candidate-long row count changed")
    selected_prediction = np.full(EXPECTED_ROWS, np.nan, np.float64)
    selected_code = np.full(EXPECTED_ROWS, -1, np.int8)
    fold = np.full(EXPECTED_ROWS, -1, np.int8)
    md_since = np.full(EXPECTED_ROWS, np.nan, np.float64)
    processed = 0
    candidate_order = np.asarray(ALL_STAGE_C_CANDIDATES, dtype=object)
    for group_index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(group_index, columns=columns).to_pandas()
        if len(chunk) % len(candidate_order):
            raise ValueError("Stage C row group breaks candidate blocks")
        block_rows = len(chunk) // len(candidate_order)
        stop = processed + block_rows
        if stop > len(rows):
            raise ValueError("Stage C contains too many canonical rows")
        ids = chunk["id"].astype(str).to_numpy().reshape(block_rows, len(candidate_order))
        candidates = (
            chunk["candidate_id"].astype(str).to_numpy().reshape(block_rows, len(candidate_order))
        )
        if not np.all(ids == ids[:, :1]) or not np.array_equal(
            candidates, np.tile(candidate_order, (block_rows, 1))
        ):
            raise ValueError("Stage C candidate block identity/order changed")
        expected_wells = rows["well_id"].iloc[processed:stop].astype(str).to_numpy()
        expected_rows = rows["row_idx"].iloc[processed:stop].to_numpy(np.int32)
        chunk_wells = chunk["well"].astype(str).to_numpy().reshape(block_rows, len(candidate_order))
        chunk_rows = (
            chunk["well_row_idx"].to_numpy(np.int32).reshape(block_rows, len(candidate_order))
        )
        if not np.all(chunk_wells == chunk_wells[:, :1]) or not np.all(
            chunk_rows == chunk_rows[:, :1]
        ):
            raise ValueError("Stage C well/row changes within candidate block")
        if not np.array_equal(chunk_wells[:, 0], expected_wells) or not np.array_equal(
            chunk_rows[:, 0], expected_rows
        ):
            raise ValueError("Stage C canonical well-row order differs from multiscale inputs")
        folds = chunk["outer_fold"].to_numpy(np.int8).reshape(block_rows, len(candidate_order))
        downstream = (
            chunk["downstream_outer_fold"]
            .to_numpy(np.int8)
            .reshape(block_rows, len(candidate_order))
        )
        counts = (
            chunk["nested_model_count"].to_numpy(np.int8).reshape(block_rows, len(candidate_order))
        )
        if not (np.all(folds == folds[:, :1]) and np.all(downstream == downstream[:, :1])):
            raise ValueError("Stage C fold changes within candidate block")
        if not np.array_equal(folds[:, 0], downstream[:, 0]) or not np.all(counts == 4):
            raise ValueError("corrected strict nested Stage C contract failed")
        values = (
            chunk["candidate_tvt"].to_numpy(np.float64).reshape(block_rows, len(candidate_order))
        )
        scores = (
            chunk["pred_abs_error"]
            .to_numpy(np.float64)
            .reshape(block_rows, len(candidate_order))[:, : len(PRIMARY_CANDIDATES)]
        )
        if not np.isfinite(values).all() or not np.isfinite(scores).all():
            raise ValueError("Stage C target-free score surface contains non-finite values")
        codes = np.argmin(scores, axis=1).astype(np.int8)
        local = np.arange(block_rows)
        selected_prediction[processed:stop] = values[local, codes]
        selected_code[processed:stop] = codes
        fold[processed:stop] = folds[:, 0]
        distances = chunk["md_since"].to_numpy(np.float64).reshape(block_rows, len(candidate_order))
        if not np.all(distances == distances[:, :1]):
            raise ValueError("Stage C md_since changes within candidate block")
        md_since[processed:stop] = distances[:, 0]
        processed = stop
    if processed != EXPECTED_ROWS:
        raise ValueError("Stage C target-free reconstruction coverage failed")
    enriched = rows.copy()
    enriched["fold"] = fold
    enriched["MD"] = md_since
    enriched["exp264_stage_c_selected_candidate_id"] = np.asarray(PRIMARY_CANDIDATES)[selected_code]
    enriched["exp264_stage_c_selected_hard_prediction"] = selected_prediction
    well_fold_count = enriched.groupby("well_id", sort=False)["fold"].nunique()
    if not well_fold_count.eq(1).all() or not enriched["fold"].between(0, 4).all():
        raise ValueError("corrected Stage C fold is not well-group-safe")
    evidence = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "rows": len(enriched),
        "candidate_long_rows": parquet.metadata.num_rows,
        "loaded_columns": columns,
        "forbidden_columns_loaded": [],
        "selected_hard_content_sha256": frame_content_sha256(
            enriched,
            [
                "well_id",
                "row_idx",
                "fold",
                "MD",
                "exp264_stage_c_selected_candidate_id",
                "exp264_stage_c_selected_hard_prediction",
            ],
        ),
    }
    return enriched, evidence


# %% [markdown]
# ## 4. Fixed row features and outer-train score


# %%
def ols_slope_normalized(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, values - values.mean()) / denominator)


def segment_boundary_mask(n_rows: int, k_values: Sequence[int], radius: int) -> np.ndarray:
    positions = np.arange(n_rows, dtype=np.float64)
    mask = np.zeros(n_rows, dtype=bool)
    for k_value in k_values:
        edges = np.linspace(0.0, float(n_rows), int(k_value) + 1)[1:-1]
        if len(edges):
            mask |= np.min(np.abs(positions[:, None] - edges[None, :]), axis=1) <= int(radius)
    return mask


def build_fixed_row_features(
    rows: pd.DataFrame, h128: int = 128, boundary_radius: int = 8
) -> pd.DataFrame:
    required = {
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "MD",
        *PREDICTION_COLUMNS,
        "exp264_stage_c_selected_candidate_id",
        "exp264_stage_c_selected_hard_prediction",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"target-free rows are missing columns: {sorted(missing)}")
    assert_no_forbidden_columns(rows.columns)
    values = rows[list(PREDICTION_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("multiscale prediction contains non-finite values")
    features = rows[list(required)].copy()
    features = features[
        [
            "well_id",
            "row_idx",
            "suffix_offset",
            "fold",
            "MD",
            *PREDICTION_COLUMNS,
            "exp264_stage_c_selected_candidate_id",
            "exp264_stage_c_selected_hard_prediction",
        ]
    ]
    features["level_spread_ft"] = np.ptp(values, axis=1)
    features["level_std_ft"] = np.std(values, axis=1, ddof=0)
    features["k16_midpoint_deviation_ft"] = np.abs(
        values[:, 1] - 0.5 * (values[:, 0] + values[:, 2])
    )
    features["outer_asymmetry_ft"] = np.abs(values[:, 0] - values[:, 1]) - np.abs(
        values[:, 2] - values[:, 1]
    )
    side12 = np.sign(values[:, 0] - values[:, 1])
    side24 = np.sign(values[:, 2] - values[:, 1])
    features["direction_agreement"] = side12 == side24
    features["k_order_monotone"] = (
        (values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])
    ) | ((values[:, 0] >= values[:, 1]) & (values[:, 1] >= values[:, 2]))
    slope_spread = np.zeros(len(rows), np.float64)
    slope_std = np.zeros(len(rows), np.float64)
    slope_mid = np.zeros(len(rows), np.float64)
    boundary_jump = np.zeros(len(rows), np.float64)
    wells = rows["well_id"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, wells[1:] != wells[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    if pd.Index(wells[starts]).duplicated().any():
        raise ValueError("well rows are not contiguous")
    for start, end in zip(starts, ends, strict=True):
        well_values = values[start:end]
        n_rows = end - start
        expected_offsets = np.arange(n_rows, dtype=np.int32)
        if not np.array_equal(
            rows["suffix_offset"].iloc[start:end].to_numpy(np.int32), expected_offsets
        ):
            raise ValueError(f"suffix offset is not contiguous for {wells[start]}")
        for block_start in range(0, n_rows, int(h128)):
            block_end = min(block_start + int(h128), n_rows)
            slopes = np.asarray(
                [
                    ols_slope_normalized(well_values[block_start:block_end, index])
                    for index in range(3)
                ],
                dtype=np.float64,
            )
            target = slice(start + block_start, start + block_end)
            slope_spread[target] = float(np.ptp(slopes))
            slope_std[target] = float(np.std(slopes, ddof=0))
            slope_mid[target] = abs(float(slopes[1] - 0.5 * (slopes[0] + slopes[2])))
        differences = np.vstack([np.zeros((1, 3), np.float64), np.diff(well_values, axis=0)])
        jump_spread = np.ptp(differences, axis=1)
        near_boundary = segment_boundary_mask(n_rows, K_VALUES, boundary_radius)
        boundary_jump[start:end] = np.where(near_boundary, jump_spread, 0.0)
    features["slope_spread_ft"] = slope_spread
    features["slope_std_ft"] = slope_std
    features["k16_slope_midpoint_deviation_ft"] = slope_mid
    features["boundary_weighted_jump_spread_ft"] = boundary_jump
    numeric = features.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("fixed target-free feature table contains non-finite values")
    return features


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(reference) == 0 or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("empirical percentile requires finite non-empty reference")
    ordered = np.sort(reference, kind="mergesort")
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left.astype(np.float64) + right.astype(np.float64)) / (2.0 * len(ordered))


def compute_outer_train_scores(features: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fold = features["fold"].to_numpy(np.int8)
    score = np.full(len(features), np.nan, np.float64)
    preprocessor: list[dict[str, Any]] = []
    for valid_fold in range(5):
        train_mask = fold != valid_fold
        valid_mask = fold == valid_fold
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"empty outer-train/valid split: fold {valid_fold}")
        ranks = []
        for component in PRIMARY_COMPONENTS:
            reference = features.loc[train_mask, component].to_numpy(np.float64)
            values = features.loc[valid_mask, component].to_numpy(np.float64)
            ranks.append(empirical_percentile(reference, values))
            preprocessor.append(
                {
                    "valid_fold": valid_fold,
                    "component": component,
                    "reference_rows": int(len(reference)),
                    "reference_min": float(reference.min()),
                    "reference_max": float(reference.max()),
                    "reference_array_sha256": sha256_array(reference),
                    "rule": "mid_empirical_cdf_(count_lt_plus_half_count_eq)_div_n",
                }
            )
        score[valid_mask] = np.mean(np.vstack(ranks), axis=0)
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("primary target-free score is invalid")
    return score, preprocessor


def build_h512_blocks(
    features: pd.DataFrame, score: np.ndarray, horizon: int = 512
) -> pd.DataFrame:
    work = features[["well_id", "suffix_offset", "fold", "MD"]].copy()
    work["row_score"] = np.asarray(score, dtype=np.float64)
    work["block_index"] = work["suffix_offset"].to_numpy(np.int64) // int(horizon)
    grouped = work.groupby(["well_id", "block_index"], sort=False, observed=True)
    blocks = grouped.agg(
        rows=("row_score", "size"),
        fold=("fold", "first"),
        fold_count=("fold", "nunique"),
        first_suffix_offset=("suffix_offset", "min"),
        last_suffix_offset=("suffix_offset", "max"),
        min_md_since=("MD", "min"),
        max_md_since=("MD", "max"),
        multiscale_k_instability_score=("row_score", lambda value: float(np.quantile(value, 0.90))),
    ).reset_index()
    if not blocks["fold_count"].eq(1).all() or blocks.duplicated(["well_id", "block_index"]).any():
        raise ValueError("H512 block identity/fold contract failed")
    blocks = blocks.drop(columns="fold_count")
    blocks.insert(0, "block_id", np.arange(len(blocks), dtype=np.int32))
    expected_block = features.groupby("well_id", sort=False).cumcount().to_numpy(np.int64) // int(
        horizon
    )
    key_to_id = pd.Series(
        blocks["block_id"].to_numpy(np.int32),
        index=pd.MultiIndex.from_frame(blocks[["well_id", "block_index"]]),
    )
    row_index = pd.MultiIndex.from_arrays([features["well_id"], expected_block])
    row_block_id = key_to_id.reindex(row_index).to_numpy()
    if pd.isna(row_block_id).any():
        raise ValueError("H512 row assignment coverage failed")
    features["h512_block_id"] = row_block_id.astype(np.int32)
    return blocks


# %% [markdown]
# ## 5. Truth-free freeze


# %%
@dataclass
class FreezeEvidence:
    feature_path: Path
    block_path: Path
    schema_path: Path
    preprocessor_path: Path
    manifest_path: Path
    file_sha256: dict[str, str]
    feature_content_sha256: str
    block_content_sha256: str
    score_recompute_max_abs: float
    truth_access_count_before_freeze: int


@dataclass
class TruthAccessLedger:
    frozen: bool = False
    count_before_freeze: int = 0

    def mark_frozen(self) -> None:
        self.frozen = True

    def register_truth_access(self) -> None:
        if not self.frozen:
            self.count_before_freeze += 1
            raise ValueError("truth access attempted before target-free freeze")


def freeze_target_free_bundle(
    features: pd.DataFrame,
    blocks: pd.DataFrame,
    preprocessor: Sequence[Mapping[str, Any]],
    score_recompute_max_abs: float,
    input_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> FreezeEvidence:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    contract_path = artifacts_dir / f"{OUTPUT_PREFIX}_contract.json"
    input_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.json"
    schema_path = artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.json"
    feature_path = artifacts_dir / f"{OUTPUT_PREFIX}_target_free_row_features.parquet"
    block_path = artifacts_dir / f"{OUTPUT_PREFIX}_target_free_h512_blocks.parquet"
    preprocessor_path = artifacts_dir / f"{OUTPUT_PREFIX}_empirical_preprocessor_manifest.json"
    manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    write_json(
        contract_path,
        {
            "experiment": EXP_NAME,
            "validation": config["validation"],
            "feature_contract": config["feature_contract"],
            "success_criteria": config["success_criteria"],
            "execution": config["execution"],
        },
    )
    write_json(input_path, input_manifest)
    write_json(
        schema_path,
        {
            "columns": [(column, str(dtype)) for column, dtype in features.dtypes.items()],
            "schema_sha256": frame_schema_sha256(features),
            "primary_components": list(PRIMARY_COMPONENTS),
            "forbidden_columns_present": sorted(
                set(features.columns) & FORBIDDEN_PRE_FREEZE_COLUMNS
            ),
        },
    )
    write_json(preprocessor_path, {"entries": list(preprocessor)})
    features.to_parquet(feature_path, index=False, compression="zstd")
    blocks.to_parquet(block_path, index=False, compression="zstd")
    file_paths = {
        "contract": contract_path,
        "input_manifest": input_path,
        "feature_schema": schema_path,
        "row_features": feature_path,
        "h512_blocks": block_path,
        "empirical_preprocessor": preprocessor_path,
    }
    file_sha = {name: sha256_file(path) for name, path in file_paths.items()}
    feature_content_sha = frame_content_sha256(features)
    block_content_sha = frame_content_sha256(blocks)
    write_json(
        manifest_path,
        {
            "phase": "target_free_frozen_before_truth",
            "truth_access_count_before_freeze": 0,
            "truth_columns_loaded_before_freeze": [],
            "files": {
                name: {"path": str(file_paths[name]), "sha256": digest}
                for name, digest in file_sha.items()
            },
            "feature_content_sha256": feature_content_sha,
            "block_content_sha256": block_content_sha,
            "score_content_sha256": sha256_array(
                features["multiscale_k_instability_score"].to_numpy(np.float64)
            ),
            "score_recompute_max_abs": float(score_recompute_max_abs),
            "rows": len(features),
            "blocks": len(blocks),
        },
    )
    return FreezeEvidence(
        feature_path=feature_path,
        block_path=block_path,
        schema_path=schema_path,
        preprocessor_path=preprocessor_path,
        manifest_path=manifest_path,
        file_sha256=file_sha,
        feature_content_sha256=feature_content_sha,
        block_content_sha256=block_content_sha,
        score_recompute_max_abs=float(score_recompute_max_abs),
        truth_access_count_before_freeze=0,
    )


def verify_freeze(freeze: FreezeEvidence) -> None:
    named = {
        "feature_schema": freeze.schema_path,
        "row_features": freeze.feature_path,
        "h512_blocks": freeze.block_path,
        "empirical_preprocessor": freeze.preprocessor_path,
    }
    for name, path in named.items():
        if sha256_file(path) != freeze.file_sha256[name]:
            raise ValueError(f"frozen artifact changed before truth join: {name}")
    if freeze.truth_access_count_before_freeze != 0:
        raise ValueError("truth access occurred before freeze")


# %% [markdown]
# ## 6. Post-freeze truth loader


# %%
def load_truth_after_freeze(
    features: pd.DataFrame,
    config: Mapping[str, Any],
    freeze: FreezeEvidence,
    ledger: TruthAccessLedger,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    verify_freeze(freeze)
    ledger.register_truth_access()
    raw_dir = resolve_directory(
        get_nested(config, "data.raw_train_dir_patterns"), label="raw train directory"
    )
    path_by_well = {
        path.name.split("__horizontal_well.csv", 1)[0]: path
        for path in raw_dir.glob("*__horizontal_well.csv")
    }
    target = np.full(len(features), np.nan, np.float64)
    manifest_rows: list[dict[str, Any]] = []
    wells = features["well_id"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, wells[1:] != wells[:-1]])
    ends = np.r_[starts[1:], len(features)]
    for start, end in zip(starts, ends, strict=True):
        well = wells[start]
        path = path_by_well.get(well)
        if path is None:
            raise FileNotFoundError(f"raw horizontal file missing: {well}")
        raw = pd.read_csv(path, usecols=["TVT"], dtype={"TVT": "float64"})
        row_index = features["row_idx"].iloc[start:end].to_numpy(np.int64)
        if row_index.min() < 0 or row_index.max() >= len(raw):
            raise ValueError(f"raw truth row index outside file: {well}")
        target[start:end] = raw["TVT"].to_numpy(np.float64)[row_index]
        manifest_rows.append(
            {
                "phase": "post_freeze_truth",
                "well_id": well,
                "path": str(path),
                "rows_loaded": int(end - start),
                "file_sha256": sha256_file(path),
                "loaded_columns": "TVT",
            }
        )
    if not np.isfinite(target).all() or len(manifest_rows) != EXPECTED_WELLS:
        raise ValueError("post-freeze truth coverage failed")
    truth_identity = pd.DataFrame(
        {
            "well_id": features["well_id"].astype("string"),
            "row_idx": features["row_idx"].to_numpy(np.int32),
            "tvt_true": target,
        }
    )
    return target, pd.DataFrame(manifest_rows), frame_content_sha256(truth_identity)


def load_hidden_like_sets_after_freeze(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        spec["patterns"],
        label="hidden-like assignment",
        expected_file_sha256=str(spec["expected_file_sha256"]),
    )
    frame = pd.read_csv(path)
    well_column = str(spec["well_column"])
    result: dict[str, set[str]] = {}
    for scope, role_column in spec["role_columns"].items():
        selected = set(frame.loc[frame[role_column].eq("valid"), well_column].astype(str))
        unknown = selected - expected_wells
        if unknown:
            raise ValueError(
                f"hidden-like assignment contains unknown wells: {sorted(unknown)[:5]}"
            )
        result[str(scope)] = selected
    return result, {"path": str(path), "file_sha256": sha256_file(path), "rows": len(frame)}


# %% [markdown]
# ## 7. Fixed block readout and guards


# %%
def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def fixed_quintile_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "bottom_rows": 0,
            "top_rows": 0,
            "bottom_positive_rate": None,
            "top_positive_rate": None,
            "positive_rate_lift": None,
            "bottom_mean_k16_benefit_ft": None,
            "top_mean_k16_benefit_ft": None,
            "mean_k16_benefit_delta_ft": None,
        }
    scores = frame["multiscale_k_instability_score"].to_numpy(np.float64)
    q20, q80 = np.quantile(scores, [0.20, 0.80])
    bottom = frame.loc[scores <= q20]
    top = frame.loc[scores >= q80]
    bottom_rate = float(bottom["positive_label"].mean())
    top_rate = float(top["positive_label"].mean())
    lift = (
        math.inf
        if bottom_rate == 0.0 and top_rate > 0.0
        else (1.0 if bottom_rate == top_rate == 0.0 else top_rate / bottom_rate)
    )
    bottom_benefit = float(bottom["k16_benefit_ft"].mean())
    top_benefit = float(top["k16_benefit_ft"].mean())
    return {
        "q20": float(q20),
        "q80": float(q80),
        "bottom_rows": len(bottom),
        "top_rows": len(top),
        "bottom_positive_rate": bottom_rate,
        "top_positive_rate": top_rate,
        "positive_rate_lift": float(lift),
        "bottom_mean_k16_benefit_ft": bottom_benefit,
        "top_mean_k16_benefit_ft": top_benefit,
        "mean_k16_benefit_delta_ft": top_benefit - bottom_benefit,
    }


def summarize_scope(scope: str, frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["positive_label"].to_numpy(bool)
    auc = roc_auc_binary(labels, frame["multiscale_k_instability_score"].to_numpy(np.float64))
    result: dict[str, Any] = {
        "scope": scope,
        "blocks": len(frame),
        "wells": frame["well_id"].nunique(),
        "positive_blocks": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "auc": auc,
        "mean_k16_benefit_ft": float(frame["k16_benefit_ft"].mean()) if len(frame) else None,
    }
    result.update(fixed_quintile_summary(frame))
    if auc is not None:
        result["direction_pass"] = bool(auc > 0.5)
        result["direction_rule"] = "auc_gt_0p5"
    else:
        delta = result["mean_k16_benefit_delta_ft"]
        result["direction_pass"] = bool(delta is not None and delta > 0.0)
        result["direction_rule"] = "top_mean_benefit_gt_bottom_when_single_class"
    return result


def build_post_freeze_readout(
    features: pd.DataFrame,
    blocks: pd.DataFrame,
    target: np.ndarray,
    hidden_sets: Mapping[str, set[str]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    block_id = features["h512_block_id"].to_numpy(np.int32)
    n_blocks = len(blocks)
    counts = np.bincount(block_id, minlength=n_blocks).astype(np.int64)
    k16_error = features["exp226_k16_prediction"].to_numpy(np.float64) - target
    hard_error = features["exp264_stage_c_selected_hard_prediction"].to_numpy(np.float64) - target
    k16_sse = np.bincount(block_id, weights=np.square(k16_error), minlength=n_blocks)
    hard_sse = np.bincount(block_id, weights=np.square(hard_error), minlength=n_blocks)
    readout = blocks.copy()
    if not np.array_equal(readout["rows"].to_numpy(np.int64), counts):
        raise ValueError("frozen H512 block counts changed after truth join")
    readout["k16_rmse"] = np.sqrt(k16_sse / counts)
    readout["exp264_selected_hard_rmse"] = np.sqrt(hard_sse / counts)
    readout["k16_benefit_ft"] = readout["exp264_selected_hard_rmse"] - readout["k16_rmse"]
    threshold = 0.25
    configured_label = str(get_nested(config, "validation.label_positive_equation"))
    if configured_label != "rmse_k16_plus_0p25_le_rmse_exp264_selected_hard":
        raise ValueError("positive-label contract changed")
    readout["positive_label"] = readout["k16_benefit_ft"] >= threshold
    readout["distance_1000_plus"] = readout["min_md_since"] >= 1000.0
    for scope, wells in hidden_sets.items():
        readout[scope] = readout["well_id"].astype(str).isin(wells)

    scopes = {
        "pooled_h512_blocks": np.ones(len(readout), bool),
        "distance_1000_plus": readout["distance_1000_plus"].to_numpy(bool),
        "hidden_like_spatial": readout["hidden_like_spatial"].to_numpy(bool),
        "hidden_like_typewell_purged": readout["hidden_like_typewell_purged"].to_numpy(bool),
    }
    scope_readout = pd.DataFrame(
        [summarize_scope(name, readout.loc[mask]) for name, mask in scopes.items()]
    )
    fold_readout = pd.DataFrame(
        [
            summarize_scope(f"fold{fold}", readout.loc[readout["fold"].eq(fold)]) | {"fold": fold}
            for fold in range(5)
        ]
    )
    by_well = (
        readout.groupby("well_id", sort=False)
        .agg(
            blocks=("block_id", "size"),
            fold=("fold", "first"),
            instability_p90=(
                "multiscale_k_instability_score",
                lambda value: float(np.quantile(value, 0.90)),
            ),
            positive_rate=("positive_label", "mean"),
            mean_k16_benefit_ft=("k16_benefit_ft", "mean"),
            worst_k16_benefit_ft=("k16_benefit_ft", "min"),
        )
        .reset_index()
    )
    pooled = scope_readout.loc[scope_readout["scope"].eq("pooled_h512_blocks")].iloc[0]
    criteria = get_nested(config, "success_criteria.scientific_all_required")
    fold_auc_passes = int(
        sum(value is not None and float(value) > 0.5 for value in fold_readout["auc"])
    )
    subgroup_names = ["distance_1000_plus", "hidden_like_spatial", "hidden_like_typewell_purged"]
    subgroup_pass = bool(
        scope_readout.set_index("scope").loc[subgroup_names, "direction_pass"].all()
    )
    scientific_checks = {
        "pooled_h512_auc": bool(
            pooled["auc"] is not None
            and float(pooled["auc"]) >= float(criteria["minimum_pooled_h512_auc"])
        ),
        "fold_direction": fold_auc_passes >= int(criteria["minimum_folds_with_auc_above_0p5"]),
        "top_bottom_positive_rate_lift": float(pooled["positive_rate_lift"])
        >= float(criteria["minimum_top_vs_bottom_quintile_positive_rate_lift"]),
        "top_bottom_mean_benefit_delta": float(pooled["mean_k16_benefit_delta_ft"])
        >= float(criteria["minimum_top_vs_bottom_quintile_mean_k16_benefit_delta_ft"]),
        "subgroup_direction": subgroup_pass,
    }
    decision = {
        "scientific_checks": scientific_checks,
        "scientific_passed": bool(all(scientific_checks.values())),
        "folds_with_auc_above_0p5": fold_auc_passes,
        "action": "support_future_separate_addonly_selector_feature_experiment"
        if all(scientific_checks.values())
        else "close_without_rescue_grid",
    }
    return readout, scope_readout, fold_readout, by_well, decision


# %% [markdown]
# ## 8. Artifacts and orchestration


# %%
def save_final_artifacts(
    artifacts_dir: Path,
    truth_manifest: pd.DataFrame,
    block_readout: pd.DataFrame,
    scope_readout: pd.DataFrame,
    fold_readout: pd.DataFrame,
    by_well: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "post_freeze_truth_manifest": artifacts_dir
        / f"{OUTPUT_PREFIX}_post_freeze_truth_manifest.csv",
        "block_readout": artifacts_dir / f"{OUTPUT_PREFIX}_block_readout.csv",
        "scope_readout": artifacts_dir / f"{OUTPUT_PREFIX}_scope_readout.csv",
        "fold_readout": artifacts_dir / f"{OUTPUT_PREFIX}_fold_readout.csv",
        "by_well": artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "summary": artifacts_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    frames = {
        "post_freeze_truth_manifest": truth_manifest,
        "block_readout": block_readout,
        "scope_readout": scope_readout,
        "fold_readout": fold_readout,
        "by_well": by_well,
    }
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False, float_format="%.12g", lineterminator="\n")
    write_json(paths["summary"], summary)
    all_paths = sorted(
        path
        for path in artifacts_dir.iterdir()
        if path.is_file() and not path.name.endswith("_sha_manifest.csv")
    )
    sha_manifest = pd.DataFrame(
        [
            {"artifact": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in all_paths
        ]
    )
    sha_path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    sha_manifest.to_csv(sha_path, index=False, lineterminator="\n")
    return {path.name: sha256_file(path) for path in [*all_paths, sha_path]}


def main() -> None:
    config = load_config()
    validate_execution_contract(config)
    artifacts_dir = runtime_artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"experiment": config["experiment"], "execution": config["execution"]},
            indent=2,
            ensure_ascii=False,
        )
    )

    target_free = load_multiscale_predictions(config)
    stage_c_spec = get_nested(config, "data.exp264_stage_c_candidate_score")
    stage_c_path = resolve_file(
        stage_c_spec["patterns"],
        label="corrected Stage C v6 candidate score",
        expected_file_sha256=str(stage_c_spec["expected_sha256"]),
    )
    rows, stage_c_evidence = attach_stage_c_selected_hard(target_free.rows, stage_c_path)
    features = build_fixed_row_features(rows)
    score, preprocessor = compute_outer_train_scores(features)
    features["multiscale_k_instability_score"] = score
    blocks = build_h512_blocks(features, score)
    recomputed_score, recomputed_preprocessor = compute_outer_train_scores(features)
    score_recompute_max_abs = float(np.max(np.abs(score - recomputed_score)))
    if score_recompute_max_abs > float(
        get_nested(
            config, "success_criteria.technical_all_required.required_score_recompute_max_abs"
        )
    ):
        raise ValueError(f"primary score recompute mismatch: {score_recompute_max_abs}")
    if preprocessor != recomputed_preprocessor:
        raise ValueError("empirical preprocessor recompute changed")
    input_manifest = {
        "phase": "target_free",
        "truth_access_count_before_freeze": 0,
        "multiscale_predictions": target_free.source_evidence,
        "stage_c_candidate_score": stage_c_evidence,
        "rows": len(features),
        "wells": features["well_id"].nunique(),
        "folds": sorted(features["fold"].unique().tolist()),
    }
    ledger = TruthAccessLedger()
    freeze = freeze_target_free_bundle(
        features,
        blocks,
        preprocessor,
        score_recompute_max_abs,
        input_manifest,
        config,
        artifacts_dir,
    )
    verify_freeze(freeze)
    ledger.mark_frozen()
    target, truth_manifest, truth_content_sha = load_truth_after_freeze(
        features, config, freeze, ledger
    )
    hidden_sets, hidden_evidence = load_hidden_like_sets_after_freeze(
        config, set(features["well_id"].astype(str))
    )
    block_readout, scope_readout, fold_readout, by_well, decision = build_post_freeze_readout(
        features, blocks, target, hidden_sets, config
    )
    technical_checks = {
        "feature_coverage": len(features) == EXPECTED_ROWS
        and features["well_id"].nunique() == EXPECTED_WELLS,
        "duplicate_blocks_zero": not blocks.duplicated(["well_id", "block_index"]).any(),
        "truth_access_before_freeze_zero": freeze.truth_access_count_before_freeze == 0
        and ledger.count_before_freeze == 0,
        "exp302_prediction_sha_match": True,
        "exp264_candidate_score_sha_match": stage_c_evidence["file_sha256"]
        == str(stage_c_spec["expected_sha256"]),
        "score_recompute": score_recompute_max_abs
        <= float(
            get_nested(
                config, "success_criteria.technical_all_required.required_score_recompute_max_abs"
            )
        ),
    }
    summary = {
        "experiment": EXP_NAME,
        "status": "technical_pass_scientific_pass"
        if all(technical_checks.values()) and decision["scientific_passed"]
        else "technical_pass_scientific_fail"
        if all(technical_checks.values())
        else "technical_fail",
        "route": config["experiment"]["route"],
        "execution": {
            "fixed_readouts": 1,
            "evaluation_folds": 5,
            "models": 0,
            "trained_folds": 0,
            "boosters": 0,
            "candidate_regeneration": 0,
            "inference": False,
            "submission": False,
        },
        "technical_checks": technical_checks,
        "technical_passed": bool(all(technical_checks.values())),
        "scientific": decision,
        "pooled": scope_readout.loc[scope_readout["scope"].eq("pooled_h512_blocks")]
        .iloc[0]
        .to_dict(),
        "subgroups": scope_readout.loc[~scope_readout["scope"].eq("pooled_h512_blocks")].to_dict(
            orient="records"
        ),
        "folds": fold_readout.to_dict(orient="records"),
        "reproducibility": {
            "input_manifest_sha256": sha256_file(
                artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.json"
            ),
            "feature_schema_file_sha256": sha256_file(freeze.schema_path),
            "feature_content_sha256": freeze.feature_content_sha256,
            "block_content_sha256": freeze.block_content_sha256,
            "truth_content_sha256": truth_content_sha,
            "hidden_like_assignment": hidden_evidence,
            "score_recompute_max_abs": score_recompute_max_abs,
            "deterministic_anchor": False,
        },
    }
    artifact_sha = save_final_artifacts(
        artifacts_dir, truth_manifest, block_readout, scope_readout, fold_readout, by_well, summary
    )
    print(
        json.dumps(
            {"summary": summary, "artifact_sha256": artifact_sha}, indent=2, ensure_ascii=False
        )
    )
    print("Artifacts:", artifacts_dir)


# %%
if __name__ == "__main__":
    main()
