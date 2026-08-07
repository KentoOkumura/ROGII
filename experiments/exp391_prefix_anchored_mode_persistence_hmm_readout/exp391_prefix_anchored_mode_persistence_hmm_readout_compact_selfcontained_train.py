# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # exp391 prefix-anchored mode-persistence HMM readout — train
#
# This train-side, deterministic CPU readout first locates persistent
# exp209 decoder-separation events without reading suffix truth.  A separately
# authorized 16-well pass can then replay the unchanged exp209 exact HMM,
# extract all posterior-mode quantities from that same pass, and track physical
# mode identity independently of top-1/top-2 mass rank.  A full pass is disabled
# until the preflight gates and another explicit approval are recorded.

# %% [markdown]
# ## Contents
#
# 1. Imports and frozen execution contract
# 2. Runtime, path, table, and SHA helpers
# 3. Target/role read ledger and strict saved-artifact loaders
# 4. Stage A0 decoder-separation events and preflight-well freeze
# 5. Unchanged exp209 input preparation
# 6. Exact joint-state forward-backward and Viterbi kernels
# 7. Posterior peak/basin extraction and stable mode lineage
# 8. Prefix-anchor no-switch conditional decoder
# 9. Stage A1 cause labels and technical/mechanism/resource gates
# 10. Stage B truth-late scoring and safety gates
# 11. Artifact manifests and execution orchestration

# %% [markdown]
# ## 1. Imports and frozen execution contract

# %%
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - Kaggle includes numba.
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    def prange(*args: int):
        return range(*args)

    def get_num_threads() -> int:
        return 1

    def set_num_threads(_: int) -> None:
        return None


EXPERIMENT_NAME = "exp391_prefix_anchored_mode_persistence_hmm_readout"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
CANDIDATE_NAME = "prefix_anchor_no_switch_conditional_mean"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXECUTE_NOTEBOOK = os.environ.get("EXP391_IMPORT_ONLY", "0") != "1"

KEY_COLUMNS = ("well_id", "row_idx")
EXP270_SAFE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "last_known_tvt",
    "md_since",
    "prefix_rows",
    "posterior_mean",
    "marginal_map",
    "topk_path_1",
)
EXP226_SAFE_COLUMNS = (
    "well_id",
    "row_idx",
    "suffix_offset",
    "fold",
    "tvt_pred",
    "tvt_geop",
    "gr_delta",
)
EXP226_TRUTH_COLUMNS = ("well_id", "row_idx", "tvt_true")
RAW_HMM_SAFE_COLUMNS = ("MD", "Z", "GR", "TVT_input")
FIXED_FORMULA_WEIGHTS = {
    "exp226_k16": 0.50,
    "likpf_mean": 0.25,
    "exact_hmm": 0.25,
}


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name differs from the exp391 contract")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp391 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp391 parent must remain exp209")
    if not bool(get_nested(config, "execution.implementation_approved", False)):
        raise ValueError("exp391 implementation is not authorized")
    if not bool(get_nested(config, "implementation.enabled", False)):
        raise ValueError("exp391 implementation.enabled must be true")
    if bool(get_nested(config, "experiment.inference_enabled", False)):
        raise ValueError("exp391 inference must remain disabled")
    if bool(get_nested(config, "execution.inference_approved", False)):
        raise ValueError("exp391 inference approval must remain false")
    if bool(get_nested(config, "execution.submission_approved", False)):
        raise ValueError("exp391 submission approval must remain false")
    if int(get_nested(config, "execution_contract.full_scientific_variants")) != 1:
        raise ValueError("exp391 permits exactly one scientific candidate")
    zero_counts = (
        "execution_contract.model_configs",
        "execution_contract.lightgbm_configs",
        "execution_contract.trained_folds",
        "execution_contract.boosters",
        "execution_contract.pf_well_runs",
        "execution_contract.beam_well_runs",
        "execution_contract.gpu_runs",
    )
    nonzero = {key: get_nested(config, key) for key in zero_counts if get_nested(config, key) != 0}
    if nonzero:
        raise ValueError(f"model/PF/Beam/GPU counts must remain zero: {nonzero}")
    if bool(get_nested(config, "execution_contract.parent_control_retraining", True)):
        raise ValueError("parent/control retraining is forbidden")
    if bool(get_nested(config, "execution_contract.parent_control_separate_regeneration", True)):
        raise ValueError("separate parent-control regeneration is forbidden")
    if get_nested(config, "model.candidate.active_variants") != [CANDIDATE_NAME]:
        raise ValueError("the single exp391 candidate name changed")

    run_stage = str(get_nested(config, "execution.run_stage", "none"))
    if run_stage not in {"none", "stage_a0", "stage_a1", "stage_b"}:
        raise ValueError(f"unsupported execution.run_stage={run_stage!r}")
    if require_run_authorization:
        approval_key = {
            "stage_a0": "execution.stage_a0_run_approved",
            "stage_a1": "execution.stage_a1_run_approved",
            "stage_b": "execution.stage_b_run_approved",
        }.get(run_stage)
        if approval_key is None:
            raise RuntimeError("exp391 has no authorized run stage")
        if not bool(get_nested(config, approval_key, False)):
            raise RuntimeError(f"{run_stage} is not authorized")
        if run_stage == "stage_b" and not bool(get_nested(config, "stage_b.enabled", False)):
            raise RuntimeError("Stage B remains disabled")


# %% [markdown]
# ## 2. Runtime, path, table, and SHA helpers

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"exp391 config not found in {candidates}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    value = yaml.safe_load((path or config_path()).read_text())
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    return value


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def output_root() -> Path:
    if is_kaggle_runtime():
        return KAGGLE_WORKING_ROOT
    return find_project_root() / "experiments" / EXPERIMENT_NAME


def artifacts_dir() -> Path:
    path = output_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_frame(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    selected = frame if columns is None else frame[list(columns)]
    normalized = selected.reset_index(drop=True).copy()
    for column in normalized.columns:
        if isinstance(normalized[column].dtype, pd.StringDtype):
            normalized[column] = normalized[column].astype(object)
    return normalized


def logical_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = normalized_frame(frame, columns)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    payload = [(column, str(dtype)) for column, dtype in normalized_frame(frame).dtypes.items()]
    return sha256_bytes(stable_json_bytes(payload))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n")


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as zipped:
            frame.to_csv(zipped, index=False)


def memory_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if os.name != "darwin" else 1024.0 * 1024.0 * 1024.0
    return value / divisor


def resolve_unique_file(
    *,
    filename: str,
    configured_candidates: Sequence[str],
    patterns: Sequence[str],
    label: str,
) -> Path:
    root = find_project_root()
    matches: dict[str, Path] = {}
    for raw in configured_candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_dir():
            candidate = candidate / filename
        if candidate.is_file():
            matches[str(candidate.resolve())] = candidate
    search_roots = [root, Path("/tmp"), KAGGLE_INPUT_ROOT]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for pattern in patterns:
            for candidate in search_root.glob(pattern):
                if candidate.is_file() and candidate.name == filename:
                    matches[str(candidate.resolve())] = candidate
    if not matches:
        raise FileNotFoundError(f"{label}: no file named {filename!r} found")
    if len(matches) > 1:
        by_digest: dict[str, list[Path]] = {}
        for candidate in matches.values():
            digest = (
                sha256_decompressed_csv(candidate)
                if candidate.suffix == ".gz"
                else sha256_file(candidate)
            )
            by_digest.setdefault(digest, []).append(candidate)
        if len(by_digest) > 1:
            rendered = {key: [str(path) for path in value] for key, value in by_digest.items()}
            raise ValueError(f"{label}: ambiguous files with different content: {rendered}")
    return sorted(matches.values(), key=lambda item: (len(str(item)), str(item)))[0]


def require_decompressed_sha(path: Path, expected: str, label: str) -> str:
    if len(expected) != 64:
        raise ValueError(f"{label}: expected decompressed SHA is not fixed")
    actual = sha256_decompressed_csv(path)
    if actual != expected:
        raise ValueError(f"{label}: decompressed SHA mismatch: {actual} != {expected}")
    return actual


def select_train_dir(paths: Iterable[Path], expected_wells: int) -> Path:
    counts: dict[Path, int] = {}
    for path in paths:
        if path.name.endswith("__horizontal_well.csv"):
            counts[path.parent] = counts.get(path.parent, 0) + 1
    valid = sorted(path for path, count in counts.items() if count == expected_wells)
    if len(valid) != 1:
        rendered = {str(path): count for path, count in sorted(counts.items())}
        raise FileNotFoundError(
            f"expected exactly one directory with {expected_wells} horizontal wells: {rendered}"
        )
    return valid[0]


def train_data_dir(config: Mapping[str, Any]) -> Path:
    expected = int(get_nested(config, "validation.expected_wells"))
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"))
        if matches:
            return select_train_dir(matches, expected)
    local = find_project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))
    matches = sorted(local.glob("*__horizontal_well.csv"))
    if len(matches) != expected:
        raise FileNotFoundError(f"{local} has {len(matches)} horizontal wells, expected {expected}")
    return local


# %% [markdown]
# ## 3. Target/role read ledger and strict saved-artifact loaders

# %%
@dataclass
class RoleReadLedger:
    target_freeze_complete: bool = False
    target_suffix_truth_rows_before_freeze: int = 0
    error_rows_before_freeze: int = 0
    hidden_role_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    hidden_role_rows_after_freeze: int = 0
    reads: list[dict[str, Any]] = field(default_factory=list)

    def record_target_free(
        self,
        label: str,
        columns: Sequence[str],
        rows: int,
        forbidden: Sequence[str],
    ) -> None:
        overlap = sorted(set(columns).intersection(forbidden))
        if overlap:
            if not self.target_freeze_complete:
                self.target_suffix_truth_rows_before_freeze += int(rows)
            raise ValueError(f"{label}: forbidden pre-freeze columns read: {overlap}")
        self.reads.append(
            {"label": label, "phase": "target_free", "columns": list(columns), "rows": int(rows)}
        )

    def freeze(self) -> None:
        if (
            self.target_suffix_truth_rows_before_freeze
            or self.error_rows_before_freeze
            or self.hidden_role_rows_before_freeze
        ):
            raise RuntimeError("truth/error/hidden-like data was read before freeze")
        self.target_freeze_complete = True

    def record_truth_late(self, label: str, rows: int) -> None:
        if not self.target_freeze_complete:
            self.target_suffix_truth_rows_before_freeze += int(rows)
            raise RuntimeError(f"{label}: truth cannot be read before target-free SHA freeze")
        self.truth_rows_after_freeze += int(rows)
        self.reads.append({"label": label, "phase": "truth_late", "rows": int(rows)})

    def record_hidden_late(self, label: str, rows: int) -> None:
        if not self.target_freeze_complete:
            self.hidden_role_rows_before_freeze += int(rows)
            raise RuntimeError(f"{label}: hidden-like role cannot be read before freeze")
        self.hidden_role_rows_after_freeze += int(rows)
        self.reads.append({"label": label, "phase": "hidden_late", "rows": int(rows)})


def input_spec(config: Mapping[str, Any], dotted_key: str) -> Mapping[str, Any]:
    value = get_nested(config, dotted_key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{dotted_key} must contain an input mapping")
    return value


def load_exp270_target_free(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = input_spec(config, "data.exp270_mode_bank")
    filename = str(spec["aggregate_filename"])
    path = resolve_unique_file(
        filename=filename,
        configured_candidates=list(spec.get("aggregate_candidates", [])),
        patterns=list(spec.get("aggregate_patterns", [f"**/{filename}"])),
        label="exp270 aggregate",
    )
    expected_sha = str(spec["expected_aggregate_decompressed_sha256"])
    digest = require_decompressed_sha(path, expected_sha, "exp270 aggregate")
    frame = pd.read_csv(path, usecols=list(EXP270_SAFE_COLUMNS), dtype={"well": str})
    ledger.record_target_free(
        "exp270",
        list(frame.columns),
        len(frame),
        ("true_tvt_readout_only", "target", "error", "abs_error", "role"),
    )
    frame = frame.rename(columns={"well": "well_id", "topk_path_1": "global_viterbi"})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    return frame, {
        "label": "exp270_aggregate",
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": digest,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }


def load_exp209_control_target_free(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = input_spec(config, "data.exp209_control")
    filename = str(spec["filename"])
    path = resolve_unique_file(
        filename=filename,
        configured_candidates=list(spec.get("candidates", [])),
        patterns=list(spec.get("patterns", [f"**/{filename}"])),
        label="exp209 saved control",
    )
    digest = require_decompressed_sha(
        path,
        str(spec["expected_decompressed_sha256"]),
        "exp209 saved control",
    )
    header = pd.read_csv(path, nrows=0)
    prediction_column = str(spec["prediction_column"])
    well_column = "well" if "well" in header.columns else "well_id"
    frame = pd.read_csv(
        path,
        usecols=["id", well_column, prediction_column],
        dtype={well_column: str, "id": str},
    ).rename(
        columns={
            well_column: "well_id",
            prediction_column: "exp209_saved_posterior_mean",
        }
    )
    if "row_idx" not in frame.columns:
        extracted = frame["id"].str.rsplit("_", n=1, expand=True)
        if extracted.shape[1] != 2:
            raise ValueError("exp209 id cannot be parsed into well and row index")
        parsed_well = extracted[0].astype(str)
        if not np.array_equal(parsed_well.to_numpy(), frame["well_id"].astype(str).to_numpy()):
            raise ValueError("exp209 id/well identity mismatch")
        frame["row_idx"] = pd.to_numeric(extracted[1], errors="raise").astype(np.int64)
    ledger.record_target_free(
        "exp209_saved_control",
        list(frame.columns),
        len(frame),
        ("target", "TVT", "error", "abs_error", "role"),
    )
    return frame[[*KEY_COLUMNS, "exp209_saved_posterior_mean"]], {
        "label": "exp209_saved_control",
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": digest,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }


def load_exp226_target_free(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = input_spec(config, "data.exp226_k16")
    filename = str(spec["filename"])
    path = resolve_unique_file(
        filename=filename,
        configured_candidates=list(spec.get("candidates", [])),
        patterns=list(spec.get("patterns", [f"**/{filename}"])),
        label="exp226 OOF",
    )
    digest = require_decompressed_sha(
        path,
        str(spec["expected_decompressed_sha256"]),
        "exp226 OOF",
    )
    frame = pd.read_csv(path, usecols=list(EXP226_SAFE_COLUMNS), dtype={"well_id": str})
    ledger.record_target_free(
        "exp226",
        list(frame.columns),
        len(frame),
        ("tvt_true", "error", "abs_error", "target", "role"),
    )
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    frame["k16_preprojection"] = (
        pd.to_numeric(frame["tvt_geop"], errors="coerce")
        + pd.to_numeric(frame["gr_delta"], errors="coerce")
    )
    frame["k16_postprojection"] = pd.to_numeric(frame["tvt_pred"], errors="coerce")
    keep = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "k16_preprojection",
        "k16_postprojection",
    ]
    return frame[keep], {
        "label": "exp226_oof",
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": digest,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }


def resolve_exp263_cache_root(config: Mapping[str, Any]) -> Path:
    spec = input_spec(config, "data.exp263_fixed_physical_candidate")
    expected = str(spec["expected_stage0_manifest_sha256"])
    roots: list[Path] = []
    for raw in list(spec.get("cache_candidates", [])):
        path = Path(str(raw))
        if not path.is_absolute():
            path = find_project_root() / path
        if path.name == "cache_manifest.json":
            path = path.parent
        if (path / "cache_manifest.json").exists():
            roots.append(path)
    for search_root in (KAGGLE_INPUT_ROOT, Path("/tmp"), find_project_root()):
        if not search_root.exists():
            continue
        for manifest in search_root.glob("**/cache_manifest.json"):
            roots.append(manifest.parent)
    matches: list[Path] = []
    for root in sorted({str(item.resolve()): item for item in roots}.values(), key=str):
        if sha256_file(root / "cache_manifest.json") == expected:
            matches.append(root)
    if not matches:
        raise FileNotFoundError("exp263 cache with the fixed manifest SHA was not found")
    materialized = [
        root
        for root in matches
        if all((root / "candidate_values" / name).exists() for name in FIXED_FORMULA_WEIGHTS)
    ]
    if not materialized:
        raise FileNotFoundError(
            "exp263 manifest was found, but the three fixed-formula candidate partitions are absent"
        )
    content_roots = {
        tuple(
            sorted(
                path.relative_to(root).as_posix()
                for name in FIXED_FORMULA_WEIGHTS
                for path in (root / "candidate_values" / name).glob("fold=*/part-*.parquet")
            )
        ): root
        for root in materialized
    }
    if len(content_roots) > 1:
        raise ValueError("multiple structurally different exp263 fixed candidate caches found")
    return sorted(materialized, key=lambda item: (len(str(item)), str(item)))[0]


def load_candidate_partition_family(root: Path, candidate: str) -> pd.DataFrame:
    paths = sorted((root / "candidate_values" / candidate).glob("fold=*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"exp263 cache has no partitions for {candidate}")
    manifest = json.loads((root / "cache_manifest.json").read_text())
    specifications = manifest.get("candidate_value_partitions", {}).get(candidate, [])
    expected_by_suffix = {
        "/".join(Path(str(item["path"])).parts[-3:]): item
        for item in specifications
    }
    if len(expected_by_suffix) != len(paths):
        raise ValueError(
            f"exp263 {candidate} partition count differs from the fixed manifest"
        )
    for path in paths:
        suffix = "/".join(path.parts[-3:])
        specification = expected_by_suffix.get(suffix)
        if specification is None:
            raise ValueError(f"exp263 {candidate} unexpected partition: {suffix}")
        actual_sha = sha256_file(path)
        if actual_sha != str(specification["file_sha256"]):
            raise ValueError(
                f"exp263 {candidate} partition SHA mismatch for {suffix}: {actual_sha}"
            )
    columns = [
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "md_since",
        "candidate_tvt",
    ]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(
        columns={
            "id": "exp263_id",
            "well": "well_id",
            "well_row_idx": "row_idx",
            "outer_fold": "exp263_partition_fold",
            "md_since": "exp263_md_since",
            "candidate_tvt": candidate,
        }
    )
    frame["well_id"] = frame["well_id"].astype(str)
    frame["exp263_id"] = frame["exp263_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["exp263_partition_fold"] = pd.to_numeric(
        frame["exp263_partition_fold"],
        errors="raise",
    ).astype(np.int8)
    frame["exp263_md_since"] = pd.to_numeric(
        frame["exp263_md_since"],
        errors="raise",
    )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"exp263 {candidate} has duplicate row keys")
    return frame


def load_exp263_fixed_target_free(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = resolve_exp263_cache_root(config)
    identity_columns = [
        "exp263_id",
        "exp263_md_since",
        "exp263_partition_fold",
    ]
    base = load_candidate_partition_family(root, "exp226_k16")
    for candidate in ("likpf_mean", "exact_hmm"):
        frame = load_candidate_partition_family(root, candidate)
        base = base.merge(
            frame[[*KEY_COLUMNS, *identity_columns, candidate]],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
            suffixes=("", f"_{candidate}"),
        )
        for column in identity_columns:
            other_column = f"{column}_{candidate}"
            left = base[column].to_numpy()
            right = base.pop(other_column).to_numpy()
            if column == "exp263_md_since":
                equal = np.array_equal(left, right, equal_nan=True)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                raise ValueError(f"exp263 {candidate} {column} identity mismatch")
    fixed = np.zeros(len(base), dtype=np.float64)
    for candidate, weight in FIXED_FORMULA_WEIGHTS.items():
        fixed += float(weight) * pd.to_numeric(base[candidate], errors="coerce").to_numpy(
            np.float64
        )
    base["exp263_fixed_candidate"] = fixed
    ledger.record_target_free(
        "exp263",
        list(base.columns),
        len(base),
        ("target", "tvt_true", "error", "abs_error", "role"),
    )
    keep = [
        "well_id",
        "row_idx",
        *identity_columns,
        "exp226_k16",
        "likpf_mean",
        "exact_hmm",
        "exp263_fixed_candidate",
    ]
    return base[keep], {
        "label": "exp263_fixed_cache",
        "path": str(root),
        "manifest_sha256": sha256_file(root / "cache_manifest.json"),
        "rows": int(len(base)),
        "wells": int(base["well_id"].nunique()),
    }


def strict_target_free_join(
    exp270: pd.DataFrame,
    exp209: pd.DataFrame,
    exp226: pd.DataFrame,
    exp263: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for label, frame in (
        ("exp270", exp270),
        ("exp209", exp209),
        ("exp226", exp226),
        ("exp263", exp263),
    ):
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"{label} has duplicate row identity")
    joined = exp270.merge(
        exp209,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator="_join_exp209",
    )
    missing_exp209 = int((joined["_join_exp209"] != "both").sum())
    if missing_exp209:
        raise ValueError(f"exp270/exp209 strict join has {missing_exp209} missing rows")
    joined = joined.drop(columns="_join_exp209")
    joined = joined.merge(
        exp226,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator="_join_exp226",
    )
    missing_exp226 = int((joined["_join_exp226"] != "both").sum())
    if missing_exp226:
        raise ValueError(f"exp270/exp226 strict join has {missing_exp226} missing rows")
    joined = joined.drop(columns="_join_exp226")
    joined = joined.merge(
        exp263,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        suffixes=("", "_exp263"),
        indicator="_join_exp263",
    )
    missing_exp263 = int((joined["_join_exp263"] != "both").sum())
    if missing_exp263:
        raise ValueError(f"exp270/exp263 strict join has {missing_exp263} missing rows")
    joined = joined.drop(columns="_join_exp263")
    id_mismatch = int(
        (
            joined["id"].astype(str).to_numpy()
            != joined["exp263_id"].astype(str).to_numpy()
        ).sum()
    )
    if id_mismatch:
        raise ValueError(f"exp270/exp263 id mismatch rows={id_mismatch}")
    md_mismatch = int(
        (
            joined["md_since"].to_numpy(np.float64)
            != joined["exp263_md_since"].to_numpy(np.float64)
        ).sum()
    )
    if md_mismatch:
        raise ValueError(f"exp270/exp263 md_since mismatch rows={md_mismatch}")
    reporting_fold_nunique = joined.groupby("well_id", sort=True)["fold"].nunique()
    cache_fold_nunique = joined.groupby("well_id", sort=True)[
        "exp263_partition_fold"
    ].nunique()
    reporting_fold_inconsistent_wells = int((reporting_fold_nunique != 1).sum())
    cache_fold_inconsistent_wells = int((cache_fold_nunique != 1).sum())
    fold_mismatch = reporting_fold_inconsistent_wells + cache_fold_inconsistent_wells
    cache_partition_folds = sorted(
        int(value) for value in joined["exp263_partition_fold"].unique()
    )
    fold_label_agreement_fraction = float(
        (
            joined["fold"].to_numpy(np.int64)
            == joined["exp263_partition_fold"].to_numpy(np.int64)
        ).mean()
    )
    exact_parity = np.abs(
        pd.to_numeric(joined["exp209_saved_posterior_mean"], errors="coerce").to_numpy(
            np.float64
        )
        - pd.to_numeric(joined["posterior_mean"], errors="coerce").to_numpy(np.float64)
    )
    finite_columns = (
        "posterior_mean",
        "marginal_map",
        "global_viterbi",
        "k16_preprojection",
        "k16_postprojection",
        "exp263_fixed_candidate",
    )
    finite = np.column_stack(
        [np.isfinite(pd.to_numeric(joined[column], errors="coerce")) for column in finite_columns]
    )
    joined = joined.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    return joined, {
        "rows": int(len(joined)),
        "wells": int(joined["well_id"].nunique()),
        "folds": sorted(int(value) for value in joined["fold"].unique()),
        "duplicate_keys": int(joined.duplicated(list(KEY_COLUMNS)).sum()),
        "missing_joins": 0,
        "fold_mismatches": fold_mismatch,
        "reporting_fold_inconsistent_wells": reporting_fold_inconsistent_wells,
        "cache_partition_fold_inconsistent_wells": cache_fold_inconsistent_wells,
        "exp263_cache_partition_folds": cache_partition_folds,
        "exp226_exp263_fold_label_agreement_fraction": fold_label_agreement_fraction,
        "id_mismatches": id_mismatch,
        "md_since_mismatches": md_mismatch,
        "finite_path_coverage": float(finite.all(axis=1).mean()),
        "exp270_exp209_mean_max_abs_diff_ft": float(np.nanmax(exact_parity)),
    }


# %% [markdown]
# ## 4. Stage A0 decoder-separation events and preflight-well freeze

# %%
def true_runs(mask: Sequence[bool], minimum_rows: int) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(values) - 1):
            end = index if active and index == len(values) - 1 else index - 1
            if end - start + 1 >= int(minimum_rows):
                runs.append((start, end))
            start = None
    return runs


def merge_intervals(
    intervals: Sequence[tuple[int, int]],
    maximum_gap_rows: int,
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for start, end in sorted(intervals[1:]):
        gap = int(start) - merged[-1][1] - 1
        if gap < int(maximum_gap_rows):
            merged[-1][1] = max(merged[-1][1], int(end))
        else:
            merged.append([int(start), int(end)])
    return [(start, end) for start, end in merged]


def linear_slope(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if finite.sum() < 2:
        return math.nan
    x = np.arange(len(array), dtype=np.float64)[finite]
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        return 0.0
    centered_y = array[finite] - float(np.mean(array[finite]))
    return float(np.dot(centered_x, centered_y) / denominator)


def max_abs_step(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.nanmax(np.abs(np.diff(array))))


def extract_decoder_separation_events(
    joined: pd.DataFrame,
    *,
    minimum_gap_ft: float,
    minimum_rows: int,
    merge_gap_rows: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for well_id, group in joined.groupby("well_id", sort=True):
        group = group.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        mean = group["posterior_mean"].to_numpy(np.float64)
        map_path = group["marginal_map"].to_numpy(np.float64)
        viterbi = group["global_viterbi"].to_numpy(np.float64)
        map_runs = true_runs(np.abs(map_path - mean) >= minimum_gap_ft, minimum_rows)
        viterbi_runs = true_runs(np.abs(viterbi - mean) >= minimum_gap_ft, minimum_rows)
        intervals = merge_intervals(sorted(map_runs + viterbi_runs), merge_gap_rows)
        for event_index, (start, end) in enumerate(intervals):
            local = group.iloc[start : end + 1]
            row: dict[str, Any] = {
                "well_id": str(well_id),
                "event_id": f"{well_id}:event:{event_index:04d}",
                "fold": int(local["fold"].iloc[0]),
                "start_row_idx": int(local["row_idx"].iloc[0]),
                "end_row_idx": int(local["row_idx"].iloc[-1]),
                "start_suffix_offset": int(local["suffix_offset"].iloc[0]),
                "end_suffix_offset": int(local["suffix_offset"].iloc[-1]),
                "rows": int(len(local)),
                "map_gap_persistent": any(
                    left <= start and right >= end for left, right in map_runs
                ),
                "viterbi_gap_persistent": any(
                    left <= start and right >= end for left, right in viterbi_runs
                ),
            }
            for name, column in (
                ("posterior_mean", "posterior_mean"),
                ("marginal_map", "marginal_map"),
                ("global_viterbi", "global_viterbi"),
                ("k16_preprojection", "k16_preprojection"),
                ("k16_postprojection", "k16_postprojection"),
                ("exp263_fixed_candidate", "exp263_fixed_candidate"),
            ):
                values = local[column].to_numpy(np.float64)
                row[f"{name}_start"] = float(values[0])
                row[f"{name}_end"] = float(values[-1])
                row[f"{name}_max_abs_step"] = max_abs_step(values)
                row[f"{name}_ramp_slope"] = linear_slope(values)
            for name, left, right in (
                ("map_minus_mean", "marginal_map", "posterior_mean"),
                ("viterbi_minus_mean", "global_viterbi", "posterior_mean"),
                ("post_minus_pre", "k16_postprojection", "k16_preprojection"),
                ("fixed_minus_post", "exp263_fixed_candidate", "k16_postprojection"),
            ):
                difference = (
                    local[left].to_numpy(np.float64) - local[right].to_numpy(np.float64)
                )
                row[f"{name}_start"] = float(difference[0])
                row[f"{name}_end"] = float(difference[-1])
                row[f"{name}_max_abs"] = float(np.nanmax(np.abs(difference)))
                row[f"{name}_sign"] = int(np.sign(np.nanmedian(difference)))
            records.append(row)
    if not records:
        return pd.DataFrame(
            columns=[
                "well_id",
                "event_id",
                "fold",
                "start_row_idx",
                "end_row_idx",
                "rows",
            ]
        )
    return pd.DataFrame(records).sort_values(
        ["fold", "well_id", "start_row_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def well_event_severity(joined: pd.DataFrame) -> pd.DataFrame:
    frame = joined[
        [*KEY_COLUMNS, "fold", "marginal_map", "global_viterbi", "posterior_mean"]
    ].copy()
    frame["severity"] = np.maximum(
        np.abs(
            frame["marginal_map"].to_numpy(np.float64)
            - frame["posterior_mean"].to_numpy(np.float64)
        ),
        np.abs(
            frame["global_viterbi"].to_numpy(np.float64)
            - frame["posterior_mean"].to_numpy(np.float64)
        ),
    )
    return (
        frame.groupby(["fold", "well_id"], sort=True, as_index=False)["severity"]
        .max()
        .sort_values(
            ["fold", "severity", "well_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def select_preflight_wells(
    severity: pd.DataFrame,
    *,
    expected_folds: Sequence[int],
    per_fold: int,
    total_wells: int,
) -> pd.DataFrame:
    require_columns(severity, ("fold", "well_id", "severity"), "severity")
    selected: list[str] = []
    reasons: dict[str, str] = {}
    for fold in expected_folds:
        fold_rows = severity.loc[severity["fold"].eq(int(fold))].sort_values(
            ["severity", "well_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        if len(fold_rows) < per_fold:
            raise ValueError(f"fold {fold} has only {len(fold_rows)} wells")
        for ordinal, row in enumerate(fold_rows.head(per_fold).itertuples(index=False), start=1):
            well = str(row.well_id)
            if well not in selected:
                selected.append(well)
                reasons[well] = f"fold_{fold}_severity_rank_{ordinal}"
    overall = severity.sort_values(["severity", "well_id"], kind="mergesort").reset_index(drop=True)
    median_value = float(overall["severity"].median())
    median_order = overall.assign(
        median_distance=np.abs(overall["severity"].to_numpy(np.float64) - median_value)
    ).sort_values(["median_distance", "well_id"], kind="mergesort")
    for row in median_order.itertuples(index=False):
        well = str(row.well_id)
        if well not in selected:
            selected.append(well)
            reasons[well] = "global_median_severity"
            break
    if len(selected) < total_wells:
        for well in sorted(severity["well_id"].astype(str).unique()):
            if well not in selected:
                selected.append(well)
                reasons[well] = "stable_well_id_fill"
            if len(selected) == total_wells:
                break
    if len(selected) != total_wells:
        raise ValueError(f"preflight selection produced {len(selected)} wells")
    lookup = severity.drop_duplicates("well_id").set_index("well_id")
    rows = []
    for ordinal, well in enumerate(selected):
        row = lookup.loc[well]
        rows.append(
            {
                "selection_order": ordinal,
                "well_id": well,
                "fold": int(row["fold"]),
                "severity": float(row["severity"]),
                "selection_reason": reasons[well],
            }
        )
    result = pd.DataFrame(rows)
    missing_folds = sorted(set(int(value) for value in expected_folds) - set(result["fold"]))
    if missing_folds:
        raise ValueError(f"preflight selection misses folds {missing_folds}")
    return result


def validate_stage_a0_gates(
    join_summary: Mapping[str, Any],
    preflight: pd.DataFrame,
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> dict[str, bool]:
    gates = get_nested(config, "stage_a0.technical_gates")
    expected_folds = list(gates["require_folds"])
    checks = {
        "rows": int(join_summary["rows"]) == int(gates["require_rows"]),
        "wells": int(join_summary["wells"]) == int(gates["require_wells"]),
        "folds": list(join_summary["folds"]) == expected_folds,
        "duplicate_keys": int(join_summary["duplicate_keys"])
        == int(gates["require_duplicate_keys"]),
        "missing_joins": int(join_summary["missing_joins"]) == int(gates["require_missing_joins"]),
        "fold_mismatches": int(join_summary["fold_mismatches"])
        == int(gates["require_fold_mismatches"]),
        "id_mismatches": int(join_summary["id_mismatches"])
        == int(gates["require_id_mismatches"]),
        "md_since_mismatches": int(join_summary["md_since_mismatches"])
        == int(gates["require_md_since_mismatches"]),
        "cache_partition_folds": list(join_summary["exp263_cache_partition_folds"])
        == list(gates["require_exp263_cache_partition_folds"]),
        "finite_path_coverage": float(join_summary["finite_path_coverage"])
        == float(gates["require_finite_path_coverage"]),
        "exp209_mean_parity": float(join_summary["exp270_exp209_mean_max_abs_diff_ft"])
        <= float(gates["require_exp270_exp209_mean_parity_atol_ft"]),
        "preflight_wells": len(preflight)
        == int(get_nested(config, "stage_a0.preflight_selection.total_wells")),
        "preflight_fold_coverage": set(preflight["fold"]) == set(expected_folds),
        "forbidden_reads": (
            ledger.target_suffix_truth_rows_before_freeze
            + ledger.error_rows_before_freeze
            + ledger.hidden_role_rows_before_freeze
        )
        <= int(gates["maximum_truth_error_role_reads_before_freeze"]),
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise RuntimeError(f"Stage A0 technical gate failed: {failed}")
    return checks


# %% [markdown]
# ## 5. Unchanged exp209 input preparation

# %%
def list_well_ids(data_dir: Path) -> list[str]:
    wells: list[str] = []
    for path in sorted(data_dir.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        if (data_dir / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def load_target_free_well(
    well_id: str,
    data_dir: Path,
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal_path = data_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = data_dir / f"{well_id}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=list(RAW_HMM_SAFE_COLUMNS))
    ledger.record_target_free(
        f"raw_horizontal:{well_id}",
        list(horizontal.columns),
        len(horizontal),
        ("TVT", "Formation", "target", "error", "role"),
    )
    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    return horizontal, typewell


def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int]:
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    valid_steps = int(valid.sum())
    if valid_steps < int(min_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    if not np.isfinite(rate):
        rate = float(fallback_rate)
    return rate, int(len(tail)), valid_steps


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    tail_n: int = 30,
) -> tuple[float, float, float, float, int, int]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_gr = known["GR"].to_numpy(np.float64)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    valid = np.isfinite(known_gr) & np.isfinite(typewell_at_known)
    if valid.sum() >= 20 and np.std(typewell_at_known[valid]) > 1.0e-6:
        cal_a, cal_b = np.polyfit(typewell_at_known[valid], known_gr[valid], 1)
    elif valid.any():
        cal_a = 1.0
        cal_b = float(np.nanmean(known_gr) - np.nanmean(typewell_at_known))
    else:
        cal_a, cal_b = 1.0, 0.0
    residual = known_gr[valid] - (cal_a * typewell_at_known[valid] + cal_b)
    if valid.sum() > 20:
        sigma = float(
            np.clip(
                1.4826 * np.median(np.abs(residual - np.median(residual))),
                8.0,
                60.0,
            )
        )
    else:
        sigma = 30.0
    init_rate, effective_rows, valid_steps = robust_initial_rate(known, tail_n)
    return (
        float(cal_a),
        float(cal_b),
        sigma,
        init_rate,
        effective_rows,
        valid_steps,
    )


def fixed_hmm_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_hmm")
    required = (
        "step",
        "n_rates",
        "rate_span",
        "sig_r",
        "sig_p",
        "emission",
        "lam",
        "sigma_mode",
        "start_sig",
        "r0_sig",
        "band_pad",
        "momentum",
        "rate_center",
    )
    missing = [key for key in required if key not in hmm]
    if missing:
        raise ValueError(f"model.fixed_hmm is missing {missing}")
    if str(hmm["emission"]) != "gaussian":
        raise ValueError("exp391 must keep the exp209 Gaussian emission")
    return dict(hmm)


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        missing = sorted(required_horizontal - set(horizontal.columns))
        raise ValueError(f"horizontal missing {missing}")
    if not required_typewell.issubset(typewell.columns):
        raise ValueError(f"typewell missing {sorted(required_typewell - set(typewell.columns))}")
    if "TVT" in horizontal.columns:
        raise ValueError("prepare_hmm_inputs forbids suffix truth")

    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    evaluation = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4:
        raise ValueError("known prefix must contain at least four rows")
    if evaluation.empty:
        raise ValueError("well has no unknown suffix")

    cal_a, cal_b, robust_sigma, init_rate, rate_rows, valid_steps = prefix_stats(
        horizontal,
        typewell_tvt,
        typewell_gr,
        tail_n=30,
    )
    if str(hmm["sigma_mode"]) == "std":
        known_tvt = known["TVT_input"].to_numpy(np.float64)
        typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
        residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
        gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
        cal_a_use, cal_b_use = 1.0, 0.0
    else:
        gr_sigma = robust_sigma
        cal_a_use, cal_b_use = cal_a, cal_b

    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    step = float(hmm["step"])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = cal_a_use * np.interp(grid, typewell_tvt, typewell_gr) + cal_b_use

    md = evaluation["MD"].to_numpy(np.float64)
    z = evaluation["Z"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[evaluation.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    if str(hmm["rate_center"]) == "zero":
        span = max(float(hmm["rate_span"]), abs(init_rate) + 0.04)
        rates = np.linspace(-span, span, int(hmm["n_rates"]), dtype=np.float64)
    else:
        rates = init_rate + np.linspace(
            -float(hmm["rate_span"]),
            float(hmm["rate_span"]),
            int(hmm["n_rates"]),
        )
    return {
        "emission_ll": emission_ll,
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "r0": float(init_rate),
        "eval_index": evaluation.index.to_numpy(np.int64),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "cal_a": cal_a,
        "cal_b": cal_b,
    }


# %% [markdown]
# ## 6. Exact joint-state forward-backward and Viterbi kernels

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb_joint(
    em,
    allowed,
    dm,
    dz,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    """The exp209 kernel with joint posterior retained in the alpha buffer."""
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    alpha = np.full((t_count, p_count, r_count), neg, np.float32)
    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)
    tmp = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = max(r2 - 1, 0)
                k1 = min(r2 + 1, r_count - 1)
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(
                            prev[p_i, r_i]
                            + rate_log_kernel[r_i, r2 - r_i + 1]
                            - best
                        )
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p2 in range(p_count):
                if allowed[t_i, p2] == 0:
                    cur[p2, r2] = neg
                    continue
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if 0 <= p1 < p_count:
                        value = tmp[p1, r2] + position_log_kernel[k_i]
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if 0 <= p1 < p_count:
                            total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    cur[p2, r2] = np.float32(
                        best + np.log(total) + lam * em[t_i, p2]
                    )
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]

    best = np.max(alpha[t_count - 1])
    terminal_total = np.sum(np.exp(alpha[t_count - 1] - best))
    loglik = float(best) + np.log(terminal_total)
    post_p = np.zeros((t_count, p_count), np.float64)
    beta_next = np.zeros((p_count, r_count), np.float32)

    values = alpha[t_count - 1] + beta_next
    best = np.max(values)
    joint_total = np.sum(np.exp(values - best))
    for p_i in range(p_count):
        for r_i in range(r_count):
            probability = np.exp(values[p_i, r_i] - best) / joint_total
            alpha[t_count - 1, p_i, r_i] = np.float32(probability)
            post_p[t_count - 1, p_i] += probability

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count and allowed[t_i, p2] != 0:
                        value = (
                            position_log_kernel[k_i]
                            + lam * em[t_i, p2]
                            + beta_next[p2, r2]
                        )
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if 0 <= p2 < p_count and allowed[t_i, p2] != 0:
                            total += np.exp(
                                position_log_kernel[k_i]
                                + lam * em[t_i, p2]
                                + beta_next[p2, r2]
                                - best
                            )
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg
        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = max(r_i - 1, 0)
                k1 = min(r_i + 1, r_count - 1)
                for r2 in range(k0, k1 + 1):
                    value = rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1]
                            + beta_tmp[p_i, r2]
                            - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg
        values = alpha[t_i - 1] + beta_cur
        best = np.max(values)
        joint_total = np.sum(np.exp(values - best))
        for p_i in range(p_count):
            for r_i in range(r_count):
                probability = np.exp(values[p_i, r_i] - best) / joint_total
                alpha[t_i - 1, p_i, r_i] = np.float32(probability)
                post_p[t_i - 1, p_i] += probability
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    return post_p, alpha, loglik


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_viterbi(
    em,
    dm,
    dz,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    """Exact global top-1 joint-state decoder under the exp209 score."""
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    invalid = np.uint8(255)
    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)
    cur = np.full((p_count, r_count), neg, np.float32)
    backpointer = np.full((t_count, p_count, r_count), invalid, np.uint8)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p2 in range(p_count):
                best = neg
                best_code = invalid
                for position_code in range(5):
                    p1 = p2 - (b0 - 2 + position_code)
                    if p1 < 0 or p1 >= p_count:
                        continue
                    k0 = max(r2 - 1, 0)
                    k1 = min(r2 + 1, r_count - 1)
                    for r1 in range(k0, k1 + 1):
                        value = (
                            prev[p1, r1]
                            + rate_log_kernel[r1, r2 - r1 + 1]
                            + position_log_kernel[position_code]
                            + lam * em[t_i, p2]
                        )
                        if value > best:
                            best = value
                            best_code = np.uint8(position_code * 3 + (r1 - r2 + 1))
                cur[p2, r2] = best
                backpointer[t_i, p2, r2] = best_code
        for p_i in range(p_count):
            for r_i in range(r_count):
                prev[p_i, r_i] = cur[p_i, r_i]
                cur[p_i, r_i] = neg

    terminal_p = 0
    terminal_r = 0
    terminal_score = neg
    for p_i in range(p_count):
        for r_i in range(r_count):
            if prev[p_i, r_i] > terminal_score:
                terminal_score = prev[p_i, r_i]
                terminal_p = p_i
                terminal_r = r_i
    positions = np.full(t_count, -1, np.int32)
    rates_path = np.full(t_count, -1, np.int16)
    p2, r2 = terminal_p, terminal_r
    for t_i in range(t_count - 1, -1, -1):
        positions[t_i] = p2
        rates_path[t_i] = r2
        if t_i == 0:
            break
        code = backpointer[t_i, p2, r2]
        position_code = int(code) // 3
        rate_code = int(code) % 3
        r1 = r2 + rate_code - 1
        mu = rates[r2] * dm[t_i] - dz[t_i]
        b0 = int(np.floor(mu / sp + 0.5))
        p1 = p2 - (b0 - 2 + position_code)
        p2, r2 = p1, r1
    return terminal_score, positions, rates_path


# %% [markdown]
# ## 7. Posterior peak/basin extraction and stable mode lineage

# %%
@dataclass(frozen=True)
class Basin:
    row_index: int
    basin_index: int
    left_index: int
    right_index: int
    peak_index: int
    center_tvt: float
    peak_density: float
    mass: float
    conditional_mean: float
    display_rank: int
    eligible_bimodal: bool


@dataclass(frozen=True)
class TrackedBasin:
    basin: Basin
    mode_id: str
    parent_mode_id: str | None
    lineage_status: str
    transported_overlap: float


def local_peaks(values: Sequence[float], minimum_height: float) -> list[int]:
    probs = np.asarray(values, dtype=np.float64)
    if not len(probs):
        return []
    peaks: list[int] = []
    if len(probs) == 1:
        return [0] if probs[0] >= minimum_height else []
    if probs[0] >= probs[1] and probs[0] >= minimum_height:
        peaks.append(0)
    for index in range(1, len(probs) - 1):
        if (
            probs[index] >= probs[index - 1]
            and probs[index] > probs[index + 1]
            and probs[index] >= minimum_height
        ):
            peaks.append(index)
    if probs[-1] > probs[-2] and probs[-1] >= minimum_height:
        peaks.append(len(probs) - 1)
    return peaks


def extract_row_basins(
    probabilities: Sequence[float],
    grid: Sequence[float],
    config: Mapping[str, Any],
    *,
    row_index: int = 0,
) -> list[Basin]:
    probs = np.asarray(probabilities, dtype=np.float64)
    tvt = np.asarray(grid, dtype=np.float64)
    if probs.ndim != 1 or tvt.ndim != 1 or len(probs) != len(tvt):
        raise ValueError("posterior row and grid must be aligned one-dimensional arrays")
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("posterior row is not normalizable")
    probs = probs / total
    peaks = local_peaks(probs, float(config["min_peak_height"]))
    if len(peaks) < 2:
        peak = int(np.argmax(probs))
        return [
            Basin(
                row_index=row_index,
                basin_index=0,
                left_index=0,
                right_index=len(probs) - 1,
                peak_index=peak,
                center_tvt=float(tvt[peak]),
                peak_density=float(probs[peak]),
                mass=1.0,
                conditional_mean=float(np.dot(probs, tvt)),
                display_rank=1,
                eligible_bimodal=False,
            )
        ]

    ranked_peaks = sorted(peaks, key=lambda index: (-float(probs[index]), int(index)))
    first_peak, second_peak = ranked_peaks[:2]
    low_peak, high_peak = sorted((first_peak, second_peak))
    valley = low_peak + int(np.argmin(probs[low_peak : high_peak + 1]))
    lower_mass = float(np.sum(probs[: valley + 1]))
    upper_mass = float(np.sum(probs[valley + 1 :]))
    mass_by_peak = {
        low_peak: lower_mass,
        high_peak: upper_mass,
    }
    top1_mass = mass_by_peak[first_peak]
    top2_mass = mass_by_peak[second_peak]
    ratio = 0.0 if top1_mass <= 0.0 else top2_mass / top1_mass
    minimum_peak = float(min(probs[first_peak], probs[second_peak]))
    valley_density = float(probs[valley])
    valley_depth = (
        0.0 if minimum_peak <= 0.0 else 1.0 - valley_density / minimum_peak
    )
    separation = float(abs(tvt[first_peak] - tvt[second_peak]))
    eligible = (
        top2_mass >= float(config["min_top2_mass"])
        and ratio >= float(config["min_top2_to_top1_mass_ratio"])
        and separation >= float(config["min_peak_separation_ft"])
        and valley_depth >= float(config["min_valley_depth"])
    )
    if not eligible:
        peak = int(np.argmax(probs))
        return [
            Basin(
                row_index=row_index,
                basin_index=0,
                left_index=0,
                right_index=len(probs) - 1,
                peak_index=peak,
                center_tvt=float(tvt[peak]),
                peak_density=float(probs[peak]),
                mass=1.0,
                conditional_mean=float(np.dot(probs, tvt)),
                display_rank=1,
                eligible_bimodal=False,
            )
        ]

    rank_by_peak = {peak: rank for rank, peak in enumerate(ranked_peaks[:2], start=1)}
    specs = (
        (0, 0, valley, low_peak),
        (1, valley + 1, len(probs) - 1, high_peak),
    )
    basins: list[Basin] = []
    for basin_index, left, right, peak in specs:
        local_probabilities = probs[left : right + 1]
        local_grid = tvt[left : right + 1]
        mass = float(np.sum(local_probabilities))
        conditional_mean = (
            float(np.dot(local_probabilities, local_grid) / mass)
            if mass > 0.0
            else float(tvt[peak])
        )
        basins.append(
            Basin(
                row_index=row_index,
                basin_index=basin_index,
                left_index=left,
                right_index=right,
                peak_index=peak,
                center_tvt=float(tvt[peak]),
                peak_density=float(probs[peak]),
                mass=mass,
                conditional_mean=conditional_mean,
                display_rank=int(rank_by_peak[peak]),
                eligible_bimodal=True,
            )
        )
    return basins


def extract_basin_rows(
    posterior: np.ndarray,
    grid: np.ndarray,
    config: Mapping[str, Any],
) -> list[list[Basin]]:
    return [
        extract_row_basins(row, grid, config, row_index=row_index)
        for row_index, row in enumerate(np.asarray(posterior, dtype=np.float64))
    ]


def rate_transition_probabilities(
    dm: float,
    rates: np.ndarray,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    rate_step = float(rates[1] - rates[0])
    variance_cells = (float(sig_r) * math.sqrt(float(dm)) / rate_step) ** 2
    matrix = np.zeros((len(rates), len(rates)), dtype=np.float64)
    for previous_rate in range(len(rates)):
        mean_move = (
            -(1.0 - float(momentum))
            * float(rates[previous_rate])
            * float(dm)
            / rate_step
        )
        plus = max(0.5 * (variance_cells + mean_move), 1.0e-12)
        minus = max(0.5 * (variance_cells - mean_move), 1.0e-12)
        total = plus + minus
        if total > 0.9:
            plus *= 0.9 / total
            minus *= 0.9 / total
        for offset, probability in ((-1, minus), (0, 1.0 - plus - minus), (1, plus)):
            current_rate = previous_rate + offset
            if 0 <= current_rate < len(rates):
                matrix[previous_rate, current_rate] += probability
    return matrix


def position_transition_probabilities(
    dm: float,
    dz: float,
    step: float,
    rates: np.ndarray,
    sig_p: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    result: list[tuple[np.ndarray, np.ndarray]] = []
    sigma = max(float(sig_p), 0.35 * float(step))
    for rate in rates:
        mean = float(rate) * float(dm) - float(dz)
        center = int(np.floor(mean / float(step) + 0.5))
        offsets = center - 2 + np.arange(5, dtype=np.int32)
        log_probability = -0.5 * ((offsets * float(step) - mean) / sigma) ** 2
        log_probability -= float(np.max(log_probability))
        probability = np.exp(log_probability)
        probability /= probability.sum()
        result.append((offsets, probability))
    return result


def transport_mode_overlap(
    previous_joint_posterior: np.ndarray,
    previous_basins: Sequence[Basin],
    current_basins: Sequence[Basin],
    *,
    dm: float,
    dz: float,
    step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    momentum: float,
) -> np.ndarray:
    joint = np.asarray(previous_joint_posterior, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != len(rates):
        raise ValueError("joint posterior shape differs from the rate grid")
    rate_matrix = rate_transition_probabilities(dm, rates, sig_r, momentum)
    position_kernels = position_transition_probabilities(dm, dz, step, rates, sig_p)
    overlap = np.zeros((len(previous_basins), len(current_basins)), dtype=np.float64)
    for previous_index, previous_basin in enumerate(previous_basins):
        transported = np.zeros_like(joint)
        for position in range(previous_basin.left_index, previous_basin.right_index + 1):
            for previous_rate in range(len(rates)):
                source_mass = joint[position, previous_rate]
                if source_mass <= 0.0:
                    continue
                for current_rate in range(len(rates)):
                    rate_probability = rate_matrix[previous_rate, current_rate]
                    if rate_probability <= 0.0:
                        continue
                    offsets, probabilities = position_kernels[current_rate]
                    for offset, position_probability in zip(offsets, probabilities, strict=True):
                        current_position = position + int(offset)
                        if 0 <= current_position < joint.shape[0]:
                            transported[current_position, current_rate] += (
                                source_mass * rate_probability * position_probability
                            )
        for current_index, current_basin in enumerate(current_basins):
            overlap[previous_index, current_index] = float(
                transported[current_basin.left_index : current_basin.right_index + 1].sum()
            )
    return overlap


def maximum_weight_matching(
    overlap: np.ndarray,
    previous_mode_ids: Sequence[str],
    current_centers: Sequence[float],
    previous_centers: Sequence[float],
    allowance_ft: float,
) -> list[tuple[int, int, float]]:
    weights = np.asarray(overlap, dtype=np.float64)
    if weights.shape != (len(previous_mode_ids), len(current_centers)):
        raise ValueError("overlap matrix shape differs from the mode lists")
    previous_order = sorted(
        range(len(previous_mode_ids)),
        key=lambda index: str(previous_mode_ids[index]),
    )
    current_order = sorted(
        range(len(current_centers)),
        key=lambda index: (float(current_centers[index]), index),
    )
    count = min(len(previous_order), len(current_order))
    candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    if count == 0:
        return []
    if len(previous_order) <= len(current_order):
        for chosen_current in itertools.permutations(current_order, count):
            pairs = tuple(zip(previous_order, chosen_current, strict=True))
            if any(
                abs(float(previous_centers[left]) - float(current_centers[right]))
                > float(allowance_ft)
                for left, right in pairs
            ):
                continue
            score = float(sum(weights[left, right] for left, right in pairs))
            candidates.append((score, pairs))
    else:
        for chosen_previous in itertools.permutations(previous_order, count):
            pairs = tuple(zip(chosen_previous, current_order, strict=True))
            if any(
                abs(float(previous_centers[left]) - float(current_centers[right]))
                > float(allowance_ft)
                for left, right in pairs
            ):
                continue
            score = float(sum(weights[left, right] for left, right in pairs))
            candidates.append((score, pairs))
    if not candidates:
        return []
    candidates.sort(
        key=lambda item: (
            -item[0],
            tuple(
                (str(previous_mode_ids[left]), float(current_centers[right]))
                for left, right in item[1]
            ),
        )
    )
    return [
        (left, right, float(weights[left, right]))
        for left, right in candidates[0][1]
    ]


def track_mode_lineages(
    basin_rows: Sequence[Sequence[Basin]],
    transport_overlaps: Sequence[np.ndarray],
    *,
    anchor_tvt: float,
    allowance_ft: float,
    anchor_overlaps: Sequence[float] | None = None,
) -> tuple[list[list[TrackedBasin]], str, list[dict[str, Any]]]:
    if not basin_rows:
        raise ValueError("mode tracking requires at least one posterior row")
    if len(transport_overlaps) != len(basin_rows) - 1:
        raise ValueError("transport-overlap count differs from posterior rows")
    tracked: list[list[TrackedBasin]] = []
    ancestry: list[dict[str, Any]] = []
    first: list[TrackedBasin] = []
    for index, basin in enumerate(sorted(basin_rows[0], key=lambda item: item.center_tvt)):
        mode_id = f"mode_{index:03d}"
        first.append(
            TrackedBasin(
                basin=basin,
                mode_id=mode_id,
                parent_mode_id=None,
                lineage_status="anchor_row",
                transported_overlap=float(basin.mass),
            )
        )
    tracked.append(first)
    if anchor_overlaps is not None:
        if len(anchor_overlaps) != len(first):
            raise ValueError("start-prior overlap count differs from first-row basins")
        anchor_mode_id = sorted(
            zip(first, anchor_overlaps, strict=True),
            key=lambda item: (-float(item[1]), item[0].mode_id),
        )[0][0].mode_id
    else:
        anchor_mode_id = min(
            first,
            key=lambda item: (abs(item.basin.center_tvt - anchor_tvt), item.mode_id),
        ).mode_id
    next_mode_number = len(first)

    for row_index in range(1, len(basin_rows)):
        previous = tracked[-1]
        current = list(sorted(basin_rows[row_index], key=lambda item: item.center_tvt))
        matching = maximum_weight_matching(
            transport_overlaps[row_index - 1],
            [item.mode_id for item in previous],
            [item.center_tvt for item in current],
            [item.basin.center_tvt for item in previous],
            allowance_ft,
        )
        assigned_current: dict[int, tuple[int, float]] = {
            current_index: (previous_index, overlap)
            for previous_index, current_index, overlap in matching
        }
        row_tracked: list[TrackedBasin] = []
        split = len(current) > len(previous)
        merge = len(current) < len(previous)
        for current_index, basin in enumerate(current):
            if current_index in assigned_current:
                previous_index, overlap = assigned_current[current_index]
                parent = previous[previous_index]
                status = "matched"
                if split:
                    status = "split_matched"
                elif merge:
                    status = "merge_survivor"
                row_tracked.append(
                    TrackedBasin(
                        basin=basin,
                        mode_id=parent.mode_id,
                        parent_mode_id=parent.mode_id,
                        lineage_status=status,
                        transported_overlap=float(overlap),
                    )
                )
            else:
                mode_id = f"mode_{next_mode_number:03d}"
                next_mode_number += 1
                row_tracked.append(
                    TrackedBasin(
                        basin=basin,
                        mode_id=mode_id,
                        parent_mode_id=None,
                        lineage_status="split_new" if split else "unresolved_new",
                        transported_overlap=0.0,
                    )
                )
        if not matching:
            row_tracked = [
                TrackedBasin(
                    basin=item.basin,
                    mode_id=item.mode_id,
                    parent_mode_id=item.parent_mode_id,
                    lineage_status="unresolved_no_matching",
                    transported_overlap=item.transported_overlap,
                )
                for item in row_tracked
            ]
        if len({item.mode_id for item in row_tracked}) != len(row_tracked):
            raise RuntimeError(f"mode identity collision at row {row_index}")
        for item in row_tracked:
            ancestry.append(
                {
                    "row_index": row_index,
                    "mode_id": item.mode_id,
                    "parent_mode_id": item.parent_mode_id,
                    "lineage_status": item.lineage_status,
                    "transported_overlap": item.transported_overlap,
                    "basin_center_tvt": item.basin.center_tvt,
                    "basin_left_index": item.basin.left_index,
                    "basin_right_index": item.basin.right_index,
                }
            )
        tracked.append(row_tracked)
    return tracked, anchor_mode_id, ancestry


def basin_for_position(
    position_index: int,
    tracked_row: Sequence[TrackedBasin],
) -> TrackedBasin | None:
    matches = [
        item
        for item in tracked_row
        if item.basin.left_index <= position_index <= item.basin.right_index
    ]
    if len(matches) > 1:
        raise RuntimeError("position belongs to multiple posterior basins")
    return matches[0] if matches else None


def annotate_path_switches(
    position_path: Sequence[int],
    tracked_rows: Sequence[Sequence[TrackedBasin]],
    anchor_mode_id: str,
) -> pd.DataFrame:
    positions = np.asarray(position_path, dtype=np.int64)
    if len(positions) != len(tracked_rows):
        raise ValueError("path and mode rows differ")
    records: list[dict[str, Any]] = []
    previous_mode = anchor_mode_id
    switch_count = 0
    for row_index, position in enumerate(positions):
        basin = basin_for_position(int(position), tracked_rows[row_index])
        current_mode = None if basin is None else basin.mode_id
        cross_mode = current_mode is None or current_mode != previous_mode
        if cross_mode:
            switch_count += 1
        records.append(
            {
                "row_index": row_index,
                "position_index": int(position),
                "anchor_mode_id": anchor_mode_id,
                "current_mode_id": current_mode,
                "cross_mode_edge": bool(cross_mode),
                "mode_switch_count": int(switch_count),
                "lineage_status": "missing_basin" if basin is None else basin.lineage_status,
            }
        )
        if current_mode is not None:
            previous_mode = current_mode
    return pd.DataFrame(records)


def anchor_position_mask(
    tracked_rows: Sequence[Sequence[TrackedBasin]],
    anchor_mode_id: str,
    position_count: int,
) -> tuple[np.ndarray, list[int]]:
    mask = np.zeros((len(tracked_rows), position_count), dtype=np.uint8)
    unresolved_rows: list[int] = []
    for row_index, row in enumerate(tracked_rows):
        matches = [item for item in row if item.mode_id == anchor_mode_id]
        if len(matches) != 1 or matches[0].lineage_status.startswith("unresolved"):
            unresolved_rows.append(row_index)
            continue
        basin = matches[0].basin
        mask[row_index, basin.left_index : basin.right_index + 1] = 1
    return mask, unresolved_rows


def build_transport_overlaps(
    joint_posterior: np.ndarray,
    basin_rows: Sequence[Sequence[Basin]],
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> list[np.ndarray]:
    return [
        transport_mode_overlap(
            joint_posterior[row_index - 1],
            basin_rows[row_index - 1],
            basin_rows[row_index],
            dm=float(prepared["dm"][row_index]),
            dz=float(prepared["dz"][row_index]),
            step=float(hmm["step"]),
            rates=np.asarray(prepared["rates"], dtype=np.float64),
            sig_r=float(hmm["sig_r"]),
            sig_p=float(hmm["sig_p"]),
            momentum=float(hmm["momentum"]),
        )
        for row_index in range(1, len(basin_rows))
    ]


def start_prior_basin_overlap(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    first_row_basins: Sequence[Basin],
) -> np.ndarray:
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    position_index = np.arange(len(grid), dtype=np.float64)
    position_log_prior = -0.5 * (
        (
            (position_index - float(prepared["start_p"]))
            * float(hmm["step"])
            / float(hmm["start_sig"])
        )
        ** 2
    )
    rate_log_prior = -0.5 * (
        (rates - float(prepared["r0"])) / float(hmm["r0_sig"])
    ) ** 2
    joint_log_prior = position_log_prior[:, None] + rate_log_prior[None, :]
    joint_log_prior -= float(np.max(joint_log_prior))
    joint_prior = np.exp(joint_log_prior)
    joint_prior /= joint_prior.sum()
    full_basin = Basin(
        row_index=-1,
        basin_index=0,
        left_index=0,
        right_index=len(grid) - 1,
        peak_index=int(np.argmax(position_log_prior)),
        center_tvt=float(grid[int(np.argmax(position_log_prior))]),
        peak_density=1.0,
        mass=1.0,
        conditional_mean=float(np.sum(joint_prior.sum(axis=1) * grid)),
        display_rank=1,
        eligible_bimodal=False,
    )
    overlap = transport_mode_overlap(
        joint_prior,
        [full_basin],
        first_row_basins,
        dm=float(np.asarray(prepared["dm"])[0]),
        dz=float(np.asarray(prepared["dz"])[0]),
        step=float(hmm["step"]),
        rates=rates,
        sig_r=float(hmm["sig_r"]),
        sig_p=float(hmm["sig_p"]),
        momentum=float(hmm["momentum"]),
    )
    return overlap[0]


# %% [markdown]
# ## 8. Prefix-anchor no-switch conditional decoder

# %%
def hmm_common_arguments(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(hmm["step"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["r0"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["momentum"]),
    )


def posterior_row_summary(
    *,
    well_id: str,
    prepared: Mapping[str, Any],
    posterior: np.ndarray,
    grid: np.ndarray,
    basin_rows: Sequence[Sequence[Basin]],
    tracked_rows: Sequence[Sequence[TrackedBasin]],
    viterbi_positions: np.ndarray,
) -> pd.DataFrame:
    mean = np.asarray(posterior, dtype=np.float64) @ np.asarray(grid, dtype=np.float64)
    map_positions = np.argmax(posterior, axis=1).astype(np.int32)
    records: list[dict[str, Any]] = []
    for row_index, (basins, tracked) in enumerate(zip(basin_rows, tracked_rows, strict=True)):
        ranked = sorted(tracked, key=lambda item: item.basin.display_rank)
        map_basin = basin_for_position(int(map_positions[row_index]), tracked)
        viterbi_basin = basin_for_position(int(viterbi_positions[row_index]), tracked)
        record: dict[str, Any] = {
            "well_id": str(well_id),
            "row_idx": int(np.asarray(prepared["eval_index"])[row_index]),
            "suffix_offset": row_index,
            "posterior_mean_same_pass": float(mean[row_index]),
            "marginal_map_same_pass": float(grid[map_positions[row_index]]),
            "global_viterbi_same_pass": float(grid[viterbi_positions[row_index]]),
            "posterior_normalization_error": float(
                abs(np.sum(posterior[row_index]) - 1.0)
            ),
            "peak_count": len(basins),
            "eligible_bimodal": len(basins) == 2 and all(
                basin.eligible_bimodal for basin in basins
            ),
            "map_mode_id": None if map_basin is None else map_basin.mode_id,
            "viterbi_mode_id": None if viterbi_basin is None else viterbi_basin.mode_id,
        }
        for rank in (1, 2):
            key = f"top{rank}"
            if rank <= len(ranked):
                item = ranked[rank - 1]
                record[f"{key}_mode_id"] = item.mode_id
                record[f"{key}_tvt"] = item.basin.center_tvt
                record[f"{key}_mass"] = item.basin.mass
                record[f"{key}_conditional_mean"] = item.basin.conditional_mean
            else:
                record[f"{key}_mode_id"] = None
                record[f"{key}_tvt"] = math.nan
                record[f"{key}_mass"] = math.nan
                record[f"{key}_conditional_mean"] = math.nan
        records.append(record)
    return pd.DataFrame(records)


def mode_ledger_frame(
    well_id: str,
    tracked_rows: Sequence[Sequence[TrackedBasin]],
    prepared: Mapping[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    eval_index = np.asarray(prepared["eval_index"], dtype=np.int64)
    for row_index, row in enumerate(tracked_rows):
        for item in sorted(row, key=lambda value: value.basin.center_tvt):
            records.append(
                {
                    "well_id": str(well_id),
                    "row_idx": int(eval_index[row_index]),
                    "suffix_offset": row_index,
                    "mode_id": item.mode_id,
                    "parent_mode_id": item.parent_mode_id,
                    "lineage_status": item.lineage_status,
                    "transported_overlap": item.transported_overlap,
                    "basin_index": item.basin.basin_index,
                    "basin_left_index": item.basin.left_index,
                    "basin_right_index": item.basin.right_index,
                    "basin_center_tvt": item.basin.center_tvt,
                    "basin_mass": item.basin.mass,
                    "basin_conditional_mean": item.basin.conditional_mean,
                    "display_mass_rank": item.basin.display_rank,
                }
            )
    return pd.DataFrame(records)


def run_same_posterior_well(
    well_id: str,
    horizontal_target_free: pd.DataFrame,
    typewell: pd.DataFrame,
    saved_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    hmm = fixed_hmm_kwargs(config)
    prepared = prepare_hmm_inputs(horizontal_target_free, typewell, hmm)
    emission = np.asarray(prepared["emission_ll"], dtype=np.float32)
    all_allowed = np.ones(emission.shape, dtype=np.uint8)
    common = hmm_common_arguments(prepared, hmm)
    posterior, joint_posterior, log_likelihood = _hmm2_fb_joint(
        emission,
        all_allowed,
        *common,
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    _, viterbi_positions, _ = _hmm2_viterbi(emission, *common)
    basin_rows = extract_basin_rows(
        posterior,
        grid,
        get_nested(config, "model.posterior_modes"),
    )
    overlaps = build_transport_overlaps(joint_posterior, basin_rows, prepared, hmm)
    anchor_overlaps = start_prior_basin_overlap(
        prepared,
        hmm,
        basin_rows[0],
    )
    tracked_rows, anchor_mode_id, ancestry = track_mode_lineages(
        basin_rows,
        overlaps,
        anchor_tvt=float(prepared["last_known_tvt"]),
        allowance_ft=float(get_nested(config, "model.posterior_modes.mode_track_allowance_ft")),
        anchor_overlaps=anchor_overlaps,
    )
    position_mask, unresolved_rows = anchor_position_mask(
        tracked_rows,
        anchor_mode_id,
        emission.shape[1],
    )
    lineage_unresolved = bool(unresolved_rows)
    no_switch_mass = 0.0
    masked_log_likelihood = -math.inf
    if not lineage_unresolved:
        masked_posterior, _, masked_log_likelihood = _hmm2_fb_joint(
            emission,
            position_mask,
            *common,
        )
        candidate = masked_posterior @ grid
        no_switch_mass = float(
            math.exp(min(0.0, float(masked_log_likelihood) - float(log_likelihood)))
        )
        candidate_finite = bool(np.isfinite(candidate).all())
        normalization_error = float(
            np.max(np.abs(np.sum(masked_posterior, axis=1) - 1.0))
        )
        no_valid_masked_path = float(masked_log_likelihood) <= -1.0e17
        if (
            no_valid_masked_path
            or not candidate_finite
            or normalization_error
            > float(
                get_nested(
                    config,
                    "stage_a1.technical_gates.maximum_posterior_normalization_error",
                )
            )
        ):
            lineage_unresolved = True
    else:
        candidate = np.full(len(posterior), np.nan, dtype=np.float64)
        normalization_error = math.inf

    row_summary = posterior_row_summary(
        well_id=well_id,
        prepared=prepared,
        posterior=posterior,
        grid=grid,
        basin_rows=basin_rows,
        tracked_rows=tracked_rows,
        viterbi_positions=viterbi_positions,
    )
    ledger = mode_ledger_frame(well_id, tracked_rows, prepared)
    path_ledger = annotate_path_switches(
        viterbi_positions,
        tracked_rows,
        anchor_mode_id,
    )
    path_ledger["well_id"] = str(well_id)
    path_ledger["row_idx"] = np.asarray(prepared["eval_index"], dtype=np.int64)
    path_ledger["suffix_offset"] = np.arange(len(path_ledger), dtype=np.int64)

    saved = saved_rows.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        saved["row_idx"].to_numpy(np.int64),
        np.asarray(prepared["eval_index"], dtype=np.int64),
    ):
        raise ValueError(f"{well_id}: raw suffix identity differs from saved artifacts")
    parent = saved["posterior_mean"].to_numpy(np.float64)
    if lineage_unresolved:
        candidate = parent.copy()
    row_summary[CANDIDATE_NAME] = np.asarray(candidate, dtype=np.float64)
    row_summary["candidate_fail_closed"] = bool(lineage_unresolved)
    row_summary["anchor_mode_id"] = anchor_mode_id
    row_summary["no_switch_path_mass"] = no_switch_mass
    row_summary["fold"] = saved["fold"].to_numpy(np.int8)
    row_summary["md_since"] = saved["md_since"].to_numpy(np.float64)

    parity = {
        "posterior_mean_max_abs_diff_ft": float(
            np.max(
                np.abs(
                    row_summary["posterior_mean_same_pass"].to_numpy(np.float64).astype(
                        np.float32
                    )
                    - parent.astype(np.float32)
                )
            )
        ),
        "marginal_map_max_abs_diff_ft": float(
            np.max(
                np.abs(
                    row_summary["marginal_map_same_pass"].to_numpy(np.float64)
                    - saved["marginal_map"].to_numpy(np.float64)
                )
            )
        ),
        "global_viterbi_max_abs_diff_ft": float(
            np.max(
                np.abs(
                    row_summary["global_viterbi_same_pass"].to_numpy(np.float64)
                    - saved["global_viterbi"].to_numpy(np.float64)
                )
            )
        ),
    }
    runtime = time.perf_counter() - started
    return {
        "well_id": str(well_id),
        "row_summary": row_summary,
        "mode_ledger": ledger,
        "path_ledger": path_ledger,
        "ancestry": ancestry,
        "parity": parity,
        "posterior_normalization_error": float(
            np.max(np.abs(np.sum(posterior, axis=1) - 1.0))
        ),
        "candidate_normalization_error": normalization_error,
        "fail_closed": bool(lineage_unresolved),
        "unresolved_rows": [int(value) for value in unresolved_rows],
        "anchor_mode_id": anchor_mode_id,
        "baseline_log_likelihood": float(log_likelihood),
        "masked_log_likelihood": float(masked_log_likelihood),
        "no_switch_path_mass": no_switch_mass,
        "runtime_seconds": float(runtime),
        "peak_rss_gb": memory_rss_gb(),
    }


# %% [markdown]
# ## 9. Stage A1 cause labels and technical/mechanism/resource gates

# %%
def stable_ramp(values: Sequence[float], minimum_rows: int) -> bool:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < int(minimum_rows):
        return False
    differences = np.diff(array)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return False
    median_sign = np.sign(np.median(differences))
    if median_sign == 0:
        return False
    directional_fraction = float(np.mean(np.sign(differences) == median_sign))
    return directional_fraction >= 0.75 and abs(linear_slope(array)) > 0.0


def classify_event_causes(
    events: pd.DataFrame,
    posterior_rows: pd.DataFrame,
    joined: pd.DataFrame,
    *,
    minimum_rows: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    posterior_indexed = posterior_rows.set_index(["well_id", "row_idx"], drop=False)
    joined_indexed = joined.set_index(["well_id", "row_idx"], drop=False)
    for event in events.itertuples(index=False):
        key_rows = [
            (str(event.well_id), row_idx)
            for row_idx in range(int(event.start_row_idx), int(event.end_row_idx) + 1)
            if (str(event.well_id), row_idx) in posterior_indexed.index
        ]
        if not key_rows:
            continue
        posterior = posterior_indexed.loc[key_rows].reset_index(drop=True)
        source = joined_indexed.loc[key_rows].reset_index(drop=True)
        eligible = posterior["eligible_bimodal"].astype(bool).to_numpy()
        top1_ids = posterior["top1_mode_id"].astype(str).to_numpy()
        top1_switch = bool(
            np.any(eligible)
            and len(set(top1_ids[eligible])) > 1
        )
        map_modes = posterior["map_mode_id"].astype(str).to_numpy()
        viterbi_modes = posterior["viterbi_mode_id"].astype(str).to_numpy()
        map_stable = len(set(map_modes[eligible])) <= 1 if np.any(eligible) else False
        viterbi_stable = (
            len(set(viterbi_modes[eligible])) <= 1 if np.any(eligible) else False
        )
        lower = np.minimum(
            posterior["top1_conditional_mean"].to_numpy(np.float64),
            posterior["top2_conditional_mean"].to_numpy(np.float64),
        )
        upper = np.maximum(
            posterior["top1_conditional_mean"].to_numpy(np.float64),
            posterior["top2_conditional_mean"].to_numpy(np.float64),
        )
        mean = posterior["posterior_mean_same_pass"].to_numpy(np.float64)
        mean_between = np.isfinite(lower) & np.isfinite(upper) & (mean >= lower) & (mean <= upper)
        posterior_averaging = bool(
            top1_switch
            and (map_stable or viterbi_stable)
            and float(np.mean(mean_between[eligible])) >= 0.50
        )
        map_mode_switch = len(set(map_modes)) > 1
        transition = bool(
            map_mode_switch
            and stable_ramp(
                posterior["marginal_map_same_pass"].to_numpy(np.float64),
                minimum_rows,
            )
        )
        raw_jump = max(
            max_abs_step(source["marginal_map"]),
            max_abs_step(source["global_viterbi"]),
        )
        pre_slope = abs(linear_slope(source["k16_preprojection"]))
        post_slope = abs(linear_slope(source["k16_postprojection"]))
        fixed_slope = abs(linear_slope(source["exp263_fixed_candidate"]))
        k16_projection = bool(
            raw_jump >= 6.0
            and post_slope > pre_slope + 1.0e-12
            and stable_ramp(source["k16_postprojection"], minimum_rows)
        )
        fixed_blend = bool(
            fixed_slope > post_slope + 1.0e-12
            and stable_ramp(source["exp263_fixed_candidate"], minimum_rows)
        )
        unresolved = not any(
            (posterior_averaging, transition, k16_projection, fixed_blend)
        )
        records.append(
            {
                "event_id": str(event.event_id),
                "well_id": str(event.well_id),
                "fold": int(event.fold),
                "rows": int(len(posterior)),
                "eligible_bimodal_rows": int(eligible.sum()),
                "posterior_averaging_supported": posterior_averaging,
                "transition_kernel_supported": transition,
                "k16_projection_supported": k16_projection,
                "fixed_blend_supported": fixed_blend,
                "unresolved": unresolved,
            }
        )
    columns = [
        "event_id",
        "well_id",
        "fold",
        "rows",
        "eligible_bimodal_rows",
        "posterior_averaging_supported",
        "transition_kernel_supported",
        "k16_projection_supported",
        "fixed_blend_supported",
        "unresolved",
    ]
    return pd.DataFrame(records, columns=columns)


def validate_stage_a1_gates(
    results: Sequence[Mapping[str, Any]],
    causes: pd.DataFrame,
    selected_events: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical = get_nested(config, "stage_a1.technical_gates")
    mechanism = get_nested(config, "stage_a1.mechanism_gates")
    parity_columns = (
        "posterior_mean_max_abs_diff_ft",
        "marginal_map_max_abs_diff_ft",
        "global_viterbi_max_abs_diff_ft",
    )
    max_parity = max(
        float(result["parity"][column])
        for result in results
        for column in parity_columns
    )
    max_normalization = max(
        float(result["posterior_normalization_error"]) for result in results
    )
    runtime = sum(float(result["runtime_seconds"]) for result in results)
    projected_runtime = runtime * (
        int(get_nested(config, "validation.expected_wells")) / max(1, len(results))
    )
    projected_rss = max(float(result["peak_rss_gb"]) for result in results)
    eligible_events = len(causes)
    hmm_supported = (
        causes["posterior_averaging_supported"].astype(bool)
        | causes["transition_kernel_supported"].astype(bool)
        if eligible_events
        else pd.Series(dtype=bool)
    )
    supported_fraction = (
        float(hmm_supported.mean()) if eligible_events else 0.0
    )
    supported_folds = (
        int(causes.loc[hmm_supported, "fold"].nunique()) if eligible_events else 0
    )
    checks = {
        "same_pass_parity": max_parity
        <= float(technical["require_same_pass_saved_decoder_parity_atol_ft"]),
        "posterior_normalization": max_normalization
        <= float(technical["maximum_posterior_normalization_error"]),
        "mode_ledger_duplicate_keys": all(
            int(result["mode_ledger"].duplicated(["well_id", "row_idx", "mode_id"]).sum())
            <= int(technical["maximum_mode_ledger_key_duplicates"])
            for result in results
        ),
        "mode_identity_collisions": all(
            result["mode_ledger"]
            .groupby(["well_id", "row_idx", "mode_id"], sort=True)
            .size()
            .max()
            <= 1
            for result in results
        ),
        "decoder_events": len(selected_events)
        >= int(technical["minimum_decoder_separation_events"]),
        "projected_runtime": projected_runtime
        <= float(technical["maximum_projected_full_runtime_seconds"]),
        "projected_rss": projected_rss
        <= float(technical["maximum_projected_peak_rss_gb"]),
        "hmm_supported_fraction": supported_fraction
        >= float(mechanism["minimum_hmm_supported_event_fraction"]),
        "hmm_supported_fold_count": supported_folds
        >= int(mechanism["minimum_reporting_folds_with_hmm_supported_event"]),
    }
    return {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "technical_pass": bool(
            all(
                checks[key]
                for key in (
                    "same_pass_parity",
                    "posterior_normalization",
                    "mode_ledger_duplicate_keys",
                    "mode_identity_collisions",
                    "decoder_events",
                    "projected_runtime",
                    "projected_rss",
                )
            )
        ),
        "mechanism_pass": bool(
            checks["hmm_supported_fraction"] and checks["hmm_supported_fold_count"]
        ),
        "maximum_parity_abs_diff_ft": max_parity,
        "maximum_posterior_normalization_error": max_normalization,
        "hmm_supported_event_fraction": supported_fraction,
        "hmm_supported_reporting_folds": supported_folds,
        "projected_full_runtime_seconds": projected_runtime,
        "projected_peak_rss_gb": projected_rss,
    }


# %% [markdown]
# ## 10. Stage B truth-late scoring and safety gates

# %%
def rmse(truth: Sequence[float], prediction: Sequence[float]) -> float:
    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(estimate)
    if not finite.any():
        return math.nan
    return float(np.sqrt(np.mean((target[finite] - estimate[finite]) ** 2)))


def load_truth_late(
    candidate_rows: pd.DataFrame,
    data_dir: Path,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for well_id, group in candidate_rows.groupby("well_id", sort=True):
        path = data_dir / f"{well_id}__horizontal_well.csv"
        truth = pd.read_csv(path, usecols=["TVT"])
        ledger.record_truth_late(f"suffix_truth:{well_id}", len(group))
        row_index = group["row_idx"].to_numpy(np.int64)
        if row_index.max(initial=-1) >= len(truth):
            raise ValueError(f"{well_id}: candidate row index exceeds raw truth rows")
        local = group[[*KEY_COLUMNS]].copy()
        local["tvt_true"] = truth.iloc[row_index]["TVT"].to_numpy(np.float64)
        records.append(local)
    result = pd.concat(records, ignore_index=True)
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("truth-late frame contains duplicate row identity")
    return result


def resolve_hidden_like(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    spec = input_spec(config, "data.hidden_like")
    filename = str(spec["filename"])
    path = resolve_unique_file(
        filename=filename,
        configured_candidates=list(spec.get("candidates", [])),
        patterns=list(spec.get("patterns", [f"**/{filename}"])),
        label="exp115 hidden-like roles",
    )
    expected = str(spec["expected_sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"exp115 role SHA mismatch: {actual} != {expected}")
    role_columns = list(spec["role_columns"])
    frame = pd.read_csv(path, usecols=["well_id", *role_columns], dtype={"well_id": str})
    ledger.record_hidden_late("exp115_hidden_like", len(frame))
    keep = frame[["well_id"]].copy()
    for column in role_columns:
        keep[column] = frame[column].astype(str).eq("valid")
    return keep.drop_duplicates("well_id")


def score_scope(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scope: str,
    scope_value: str,
) -> dict[str, Any]:
    truth = frame.loc[mask, "tvt_true"].to_numpy(np.float64)
    parent = frame.loc[mask, "posterior_mean"].to_numpy(np.float64)
    candidate = frame.loc[mask, CANDIDATE_NAME].to_numpy(np.float64)
    parent_rmse = rmse(truth, parent)
    candidate_rmse = rmse(truth, candidate)
    return {
        "scope": scope,
        "scope_value": scope_value,
        "rows": int(mask.sum()),
        "parent_rmse": parent_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_rmse_vs_parent": candidate_rmse - parent_rmse,
        "gain_rmse_vs_parent": parent_rmse - candidate_rmse,
    }


def score_stage_b(
    candidate_rows: pd.DataFrame,
    joined: pd.DataFrame,
    cause_labels: pd.DataFrame,
    events: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_like: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = joined.merge(
        candidate_rows[
            [
                *KEY_COLUMNS,
                CANDIDATE_NAME,
                "candidate_fail_closed",
            ]
        ],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    ).merge(truth, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    frame = frame.merge(hidden_like, on="well_id", how="left", validate="many_to_one")
    role_columns = list(get_nested(config, "data.hidden_like.role_columns"))
    for column in role_columns:
        frame[column] = frame[column].fillna(False).astype(bool)

    hmm_event_ids = set(
        cause_labels.loc[
            cause_labels["posterior_averaging_supported"].astype(bool)
            | cause_labels["transition_kernel_supported"].astype(bool),
            "event_id",
        ].astype(str)
    )
    event_keys: set[tuple[str, int]] = set()
    for event in events.loc[events["event_id"].astype(str).isin(hmm_event_ids)].itertuples(
        index=False
    ):
        event_keys.update(
            (str(event.well_id), row_idx)
            for row_idx in range(int(event.start_row_idx), int(event.end_row_idx) + 1)
        )
    frame_keys = list(zip(frame["well_id"].astype(str), frame["row_idx"].astype(int), strict=True))
    hmm_event_mask = np.asarray([key in event_keys for key in frame_keys], dtype=bool)

    scopes: list[dict[str, Any]] = []
    all_mask = np.ones(len(frame), dtype=bool)
    scopes.append(score_scope(frame, all_mask, "overall", "all"))
    for fold in sorted(frame["fold"].unique()):
        mask = frame["fold"].to_numpy(np.int64) == int(fold)
        scopes.append(score_scope(frame, mask, "fold", str(int(fold))))
    scopes.append(score_scope(frame, hmm_event_mask, "hmm_supported_event", "true"))
    long_mask = frame["md_since"].to_numpy(np.float64) >= 1000.0
    scopes.append(score_scope(frame, long_mask, "distance", "1000_plus"))
    for column in role_columns:
        scopes.append(
            score_scope(
                frame,
                frame[column].to_numpy(dtype=bool),
                "hidden_like",
                column,
            )
        )
    scope_frame = pd.DataFrame(scopes)

    by_well_records = []
    for well_id, group in frame.groupby("well_id", sort=True):
        parent_rmse = rmse(group["tvt_true"], group["posterior_mean"])
        candidate_rmse = rmse(group["tvt_true"], group[CANDIDATE_NAME])
        by_well_records.append(
            {
                "well_id": str(well_id),
                "fold": int(group["fold"].iloc[0]),
                "rows": int(len(group)),
                "parent_rmse": parent_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse_vs_parent": candidate_rmse - parent_rmse,
                "fail_closed": bool(group["candidate_fail_closed"].all()),
            }
        )
    by_well = pd.DataFrame(by_well_records)

    frame["report_only_exp263_replaced"] = (
        frame["exp263_fixed_candidate"].to_numpy(np.float64)
        + 0.25
        * (
            frame[CANDIDATE_NAME].to_numpy(np.float64)
            - frame["exact_hmm"].to_numpy(np.float64)
        )
    )
    original_fixed_rmse = rmse(frame["tvt_true"], frame["exp263_fixed_candidate"])
    replaced_fixed_rmse = rmse(frame["tvt_true"], frame["report_only_exp263_replaced"])
    overall = scope_frame.loc[
        scope_frame["scope"].eq("overall") & scope_frame["scope_value"].eq("all")
    ].iloc[0]
    positive_folds = int(
        (
            scope_frame.loc[scope_frame["scope"].eq("fold"), "gain_rmse_vs_parent"]
            > 0.0
        ).sum()
    )
    event_row = scope_frame.loc[scope_frame["scope"].eq("hmm_supported_event")].iloc[0]
    long_row = scope_frame.loc[
        scope_frame["scope"].eq("distance")
        & scope_frame["scope_value"].eq("1000_plus")
    ].iloc[0]
    hidden_rows = scope_frame.loc[scope_frame["scope"].eq("hidden_like")]
    scientific = get_nested(config, "stage_b.scientific_gates")
    by_well_p95 = float(np.quantile(by_well["delta_rmse_vs_parent"], 0.95))
    worst_well = float(by_well["delta_rmse_vs_parent"].max())
    fail_fraction = float(by_well["fail_closed"].mean())
    checks = {
        "pooled_gain": float(overall["gain_rmse_vs_parent"])
        >= float(scientific["minimum_rmse_gain_vs_exp209_ft"]),
        "positive_folds": positive_folds >= int(scientific["minimum_positive_folds"]),
        "hmm_event_gain": float(event_row["gain_rmse_vs_parent"])
        >= float(scientific["minimum_hmm_supported_event_gain_ft"]),
        "distance_1000_plus": float(long_row["delta_rmse_vs_parent"])
        <= float(scientific["maximum_1000_plus_regression_ft"]),
        "hidden_like": bool(
            (
                hidden_rows["delta_rmse_vs_parent"]
                <= max(
                    float(scientific["maximum_hidden_like_spatial_regression_ft"]),
                    float(scientific["maximum_hidden_like_typewell_purged_regression_ft"]),
                )
            ).all()
        ),
        "by_well_p95": by_well_p95
        <= float(scientific["maximum_by_well_delta_p95_ft"]),
        "worst_well": worst_well
        <= float(scientific["maximum_worst_well_regression_ft"]),
        "fail_closed_fraction": fail_fraction
        <= float(scientific["maximum_fail_closed_well_fraction"]),
        "exp263_report_formula": replaced_fixed_rmse <= original_fixed_rmse,
    }
    summary = {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "pooled_parent_rmse": float(overall["parent_rmse"]),
        "pooled_candidate_rmse": float(overall["candidate_rmse"]),
        "pooled_gain_rmse_ft": float(overall["gain_rmse_vs_parent"]),
        "positive_reporting_folds": positive_folds,
        "hmm_supported_event_gain_ft": float(event_row["gain_rmse_vs_parent"]),
        "by_well_delta_p95_ft": by_well_p95,
        "worst_well_regression_ft": worst_well,
        "fail_closed_well_fraction": fail_fraction,
        "exp263_original_fixed_rmse": original_fixed_rmse,
        "exp263_report_replaced_rmse": replaced_fixed_rmse,
    }
    return scope_frame, by_well, summary


# %% [markdown]
# ## 11. Artifact manifests and execution orchestration

# %%
def decoder_contract_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "parent": PARENT_EXPERIMENT,
        "fixed_hmm": dict(get_nested(config, "model.fixed_hmm")),
        "posterior_modes": dict(get_nested(config, "model.posterior_modes")),
        "mode_identity": dict(get_nested(config, "model.mode_identity")),
        "candidate": dict(get_nested(config, "model.candidate")),
        "kernel": "_hmm2_fb_joint_exp209_equivalent_with_joint_posterior_retention",
        "viterbi": "_hmm2_viterbi_exact_joint_top1",
        "stable_order": "well,row,state,peak,basin,previous_mode_id,current_center_tvt",
        "exp236_row_artifact_reads": 0,
        "top1_top2_rank_used_as_identity": False,
    }


def run_stage_a0(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> dict[str, Any]:
    started = time.perf_counter()
    exp270, exp270_manifest = load_exp270_target_free(config, ledger)
    exp209, exp209_manifest = load_exp209_control_target_free(config, ledger)
    exp226, exp226_manifest = load_exp226_target_free(config, ledger)
    exp263, exp263_manifest = load_exp263_fixed_target_free(config, ledger)
    joined, join_summary = strict_target_free_join(exp270, exp209, exp226, exp263)
    event_config = get_nested(config, "stage_a0.decoder_separation_event")
    events = extract_decoder_separation_events(
        joined,
        minimum_gap_ft=float(event_config["minimum_abs_decoder_gap_ft"]),
        minimum_rows=int(event_config["minimum_consecutive_rows"]),
        merge_gap_rows=int(event_config["merge_event_gap_rows"]),
    )
    severity = well_event_severity(joined)
    selection = get_nested(config, "stage_a0.preflight_selection")
    preflight = select_preflight_wells(
        severity,
        expected_folds=list(get_nested(config, "validation.expected_folds")),
        per_fold=int(selection["per_fold_top_severity_wells"]),
        total_wells=int(selection["total_wells"]),
    )
    gates = validate_stage_a0_gates(join_summary, preflight, ledger, config)

    output = artifacts_dir()
    event_path = output / f"{EXPERIMENT_NAME}_stage_a0_event_manifest.csv"
    preflight_path = output / f"{EXPERIMENT_NAME}_preflight_well_manifest.csv"
    events.to_csv(event_path, index=False)
    preflight.to_csv(preflight_path, index=False)
    manifests = [exp270_manifest, exp209_manifest, exp226_manifest, exp263_manifest]
    input_path = output / f"{EXPERIMENT_NAME}_input_manifest.json"
    write_json(input_path, {"inputs": manifests})
    summary = {
        "stage": "stage_a0",
        "status": "pass",
        "join": join_summary,
        "technical_gates": gates,
        "event_count": int(len(events)),
        "event_wells": int(events["well_id"].nunique()) if len(events) else 0,
        "preflight_wells": int(len(preflight)),
        "event_manifest_logical_sha256": logical_frame_sha256(events),
        "preflight_well_manifest_logical_sha256": logical_frame_sha256(preflight),
        "input_manifest_sha256": sha256_file(input_path),
        "truth_error_hidden_reads_before_freeze": (
            ledger.target_suffix_truth_rows_before_freeze
            + ledger.error_rows_before_freeze
            + ledger.hidden_role_rows_before_freeze
        ),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_gb": memory_rss_gb(),
    }
    write_json(output / f"{EXPERIMENT_NAME}_stage_a0_summary.json", summary)
    return {
        "joined": joined,
        "events": events,
        "preflight": preflight,
        "input_manifests": manifests,
        "summary": summary,
    }


def run_hmm_scope(
    well_ids: Sequence[str],
    stage_a0: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> dict[str, Any]:
    data_dir = train_data_dir(config)
    joined = stage_a0["joined"]
    results: list[dict[str, Any]] = []
    for ordinal, well_id in enumerate(sorted(str(value) for value in well_ids), start=1):
        print(f"[exp391] HMM well {ordinal}/{len(well_ids)}: {well_id}", flush=True)
        horizontal, typewell = load_target_free_well(well_id, data_dir, ledger)
        saved = joined.loc[joined["well_id"].eq(well_id)].copy()
        result = run_same_posterior_well(
            well_id,
            horizontal,
            typewell,
            saved,
            config,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "well_id": well_id,
                    "runtime_seconds": result["runtime_seconds"],
                    "peak_rss_gb": result["peak_rss_gb"],
                    "parity": result["parity"],
                    "fail_closed": result["fail_closed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    posterior_rows = pd.concat(
        [result["row_summary"] for result in results],
        ignore_index=True,
    )
    mode_ledger = pd.concat(
        [result["mode_ledger"] for result in results],
        ignore_index=True,
    )
    path_ledger = pd.concat(
        [result["path_ledger"] for result in results],
        ignore_index=True,
    )
    return {
        "results": results,
        "posterior_rows": posterior_rows,
        "mode_ledger": mode_ledger,
        "path_ledger": path_ledger,
    }


def run_stage_a1(
    stage_a0: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> dict[str, Any]:
    preflight = stage_a0["preflight"]
    well_ids = preflight["well_id"].astype(str).tolist()
    scope = run_hmm_scope(well_ids, stage_a0, config, ledger)
    selected_events = stage_a0["events"].loc[
        stage_a0["events"]["well_id"].astype(str).isin(well_ids)
    ].copy()
    causes = classify_event_causes(
        selected_events,
        scope["posterior_rows"],
        stage_a0["joined"],
        minimum_rows=int(
            get_nested(
                config,
                "stage_a0.decoder_separation_event.minimum_consecutive_rows",
            )
        ),
    )
    gates = validate_stage_a1_gates(
        scope["results"],
        causes,
        selected_events,
        config,
    )
    output = artifacts_dir()
    posterior_path = output / f"{EXPERIMENT_NAME}_posterior_row_summary.csv.gz"
    mode_path = output / f"{EXPERIMENT_NAME}_mode_ledger.csv.gz"
    path_path = output / f"{EXPERIMENT_NAME}_viterbi_mode_path_ledger.csv.gz"
    causes_path = output / f"{EXPERIMENT_NAME}_cause_labels.csv"
    write_deterministic_gzip_csv(scope["posterior_rows"], posterior_path)
    write_deterministic_gzip_csv(scope["mode_ledger"], mode_path)
    write_deterministic_gzip_csv(scope["path_ledger"], path_path)
    causes.to_csv(causes_path, index=False)
    decoder_manifest = decoder_contract_manifest(config)
    decoder_manifest["decoder_contract_manifest_sha256"] = sha256_bytes(
        stable_json_bytes(decoder_manifest)
    )
    decoder_path = output / f"{EXPERIMENT_NAME}_decoder_contract_manifest.json"
    write_json(decoder_path, decoder_manifest)
    candidate_columns = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "md_since",
        CANDIDATE_NAME,
        "candidate_fail_closed",
        "anchor_mode_id",
        "no_switch_path_mass",
    ]
    candidate = scope["posterior_rows"][candidate_columns].copy()
    candidate_path = output / f"{EXPERIMENT_NAME}_candidate_target_free.csv.gz"
    write_deterministic_gzip_csv(candidate, candidate_path)
    summary = {
        "stage": "stage_a1",
        "status": "pass" if gates["all_pass"] else "fail_closed",
        "gates": gates,
        "wells": len(well_ids),
        "candidate": CANDIDATE_NAME,
        "posterior_row_summary_logical_sha256": logical_frame_sha256(
            scope["posterior_rows"]
        ),
        "mode_ledger_logical_sha256": logical_frame_sha256(scope["mode_ledger"]),
        "path_ledger_logical_sha256": logical_frame_sha256(scope["path_ledger"]),
        "cause_labels_logical_sha256": logical_frame_sha256(causes),
        "candidate_prediction_logical_sha256": logical_frame_sha256(candidate),
        "decoder_contract_manifest_sha256": decoder_manifest[
            "decoder_contract_manifest_sha256"
        ],
        "truth_error_hidden_reads_before_freeze": (
            ledger.target_suffix_truth_rows_before_freeze
            + ledger.error_rows_before_freeze
            + ledger.hidden_role_rows_before_freeze
        ),
    }
    write_json(output / f"{EXPERIMENT_NAME}_stage_a1_summary.json", summary)
    return {
        **scope,
        "causes": causes,
        "gates": gates,
        "candidate": candidate,
        "summary": summary,
    }


def run_stage_b(
    stage_a0: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> dict[str, Any]:
    well_ids = sorted(stage_a0["joined"]["well_id"].astype(str).unique())
    scope = run_hmm_scope(well_ids, stage_a0, config, ledger)
    causes = classify_event_causes(
        stage_a0["events"],
        scope["posterior_rows"],
        stage_a0["joined"],
        minimum_rows=int(
            get_nested(
                config,
                "stage_a0.decoder_separation_event.minimum_consecutive_rows",
            )
        ),
    )
    candidate_columns = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "md_since",
        CANDIDATE_NAME,
        "candidate_fail_closed",
        "anchor_mode_id",
        "no_switch_path_mass",
    ]
    candidate = scope["posterior_rows"][candidate_columns].copy()
    output = artifacts_dir()
    candidate_path = output / f"{EXPERIMENT_NAME}_candidate_target_free.csv.gz"
    write_deterministic_gzip_csv(candidate, candidate_path)
    candidate_sha = logical_frame_sha256(candidate)
    mode_sha = logical_frame_sha256(scope["mode_ledger"])
    ledger.freeze()
    truth = load_truth_late(candidate, train_data_dir(config), ledger)
    hidden = resolve_hidden_like(config, ledger)
    scope_metrics, by_well, gates = score_stage_b(
        candidate,
        stage_a0["joined"],
        causes,
        stage_a0["events"],
        truth,
        hidden,
        config,
    )
    scope_metrics.to_csv(output / f"{EXPERIMENT_NAME}_scope_metrics.csv", index=False)
    by_well.to_csv(output / f"{EXPERIMENT_NAME}_by_well.csv", index=False)
    write_deterministic_gzip_csv(
        scope["mode_ledger"],
        output / f"{EXPERIMENT_NAME}_mode_ledger.csv.gz",
    )
    summary = {
        "stage": "stage_b",
        "status": "pass" if gates["all_pass"] else "fail_closed",
        "candidate": CANDIDATE_NAME,
        "wells": len(well_ids),
        "rows": len(candidate),
        "gates": gates,
        "candidate_prediction_logical_sha256": candidate_sha,
        "mode_ledger_logical_sha256": mode_sha,
        "truth_rows_after_freeze": ledger.truth_rows_after_freeze,
        "hidden_role_rows_after_freeze": ledger.hidden_role_rows_after_freeze,
    }
    write_json(output / f"{EXPERIMENT_NAME}_stage_b_summary.json", summary)
    return {
        **scope,
        "candidate": candidate,
        "causes": causes,
        "scope_metrics": scope_metrics,
        "by_well": by_well,
        "gates": gates,
        "summary": summary,
    }


def write_run_metrics(
    config: Mapping[str, Any],
    stage_a0: Mapping[str, Any],
    final: Mapping[str, Any] | None,
    ledger: RoleReadLedger,
) -> None:
    run_stage = str(get_nested(config, "execution.run_stage"))
    payload = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            stage_a0["summary"]["status"]
            if final is None
            else final["summary"]["status"]
        ),
        "route": "pf_beam",
        "parent": PARENT_EXPERIMENT,
        "run_stage": run_stage,
        "candidate": CANDIDATE_NAME,
        "implementation_enabled": True,
        "inference_enabled": False,
        "stage_a0": stage_a0["summary"],
        "final_stage": None if final is None else final["summary"],
        "role_read_ledger": asdict(ledger),
        "runtime": {
            "numba_available": NUMBA_AVAILABLE,
            "numba_threads": int(get_num_threads()),
            "cpu_only": True,
            "internet_required": False,
            "peak_rss_gb": memory_rss_gb(),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output_root() / "metrics.json", payload)


def main() -> None:
    config = load_config()
    validate_execution_contract(config, require_run_authorization=True)
    if not is_kaggle_runtime() and os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
        raise RuntimeError(
            "exp391 notebook execution is Kaggle-first; local execution requires "
            "EXPERIMENT_ALLOW_LOCAL=1 and separate approval"
        )
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads", 1)))
    run_stage = str(get_nested(config, "execution.run_stage"))
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(config, "experiment.route"),
                "parent": get_nested(config, "lineage.parent"),
                "run_stage": run_stage,
                "candidate": CANDIDATE_NAME,
                "active_hmm_variants": get_nested(
                    config,
                    "stage_b.active_hmm_variants",
                ),
                "stage_a0_hmm_well_runs": get_nested(
                    config,
                    "execution_contract.stage_a0_hmm_well_runs",
                ),
                "stage_a1_hmm_well_runs": get_nested(
                    config,
                    "execution_contract.stage_a1_hmm_well_runs",
                ),
                "stage_b_hmm_well_runs": get_nested(
                    config,
                    "execution_contract.stage_b_hmm_well_runs",
                ),
                "lightgbm_configs": 0,
                "trained_folds": 0,
                "boosters": 0,
                "pf_runs": 0,
                "beam_runs": 0,
                "gpu_runs": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ledger = RoleReadLedger()
    stage_a0 = run_stage_a0(config, ledger)
    final: Mapping[str, Any] | None = None
    if run_stage == "stage_a0":
        ledger.freeze()
    elif run_stage == "stage_a1":
        final = run_stage_a1(stage_a0, config, ledger)
        ledger.freeze()
    elif run_stage == "stage_b":
        final = run_stage_b(stage_a0, config, ledger)
    else:  # guarded above; retained as a notebook-visible fail-closed branch.
        raise RuntimeError("no exp391 stage is authorized")
    write_run_metrics(config, stage_a0, final, ledger)
    print(json.dumps({"completed": run_stage, "status": "written"}, sort_keys=True))


# %% [markdown]
# The repository version remains import-only during tests.  Kaggle execution is
# possible only after the selected stage's explicit approval flag is changed.

# %%
if EXECUTE_NOTEBOOK:
    main()
