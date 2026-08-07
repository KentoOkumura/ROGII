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
# # exp435 TVT memoryless U-rate / dz-only exact HMM — Stage 0
#
# This CPU-only notebook implements two preregistered TVT-state treatments.
# `memoryless_41rate` marginalizes a fixed stationary distribution over the
# parent 41-rate support at every row. `dz_only_r0` calls the same transition
# kernel with a delta mass at zero. Rate responsibility is a row-local
# diagnostic and is never carried to the next row. The fixed32 sample is
# mechanism-only and is not CV or promotion evidence. Stage 0 execution,
# Stage 1, inference, and submission remain separately locked.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 manifest, saved parent, and target-free raw inputs
# 4. Exact exp209 HMM input preparation
# 5. TVT-only forward-backward and row-local rate marginalization
# 6. Two-treatment decoding and target-free freeze
# 7. Truth-late persistent-episode and safety readout
# 8. Stage 0 gates, generated artifacts, and metrics
# 9. Configuration preview and guarded execution

# %% [markdown]
# ## 1. Imports and immutable execution contract

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import platform
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numba import njit, prange, set_num_threads

EXPERIMENT_NAME = "exp435_tvt_memoryless_u_rate_dzonly_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
EVIDENCE_EXPERIMENT = "exp408_hmm_message_rate_basin_audit"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

FORBIDDEN_DECODER_COLUMNS = frozenset(
    {
        "TVT",
        "tvt_true",
        "error",
        "abs_error",
        "episode_id",
        "start_row_idx",
        "end_row_idx_exclusive",
        "fold",
        "role",
        "hidden_like_role",
    }
)

EXPECTED_SHARED_HMM = {
    "step": 0.35,
    "n_rates": 41,
    "rate_span": 0.10,
    "rate_step": 0.005,
    "sig_r": 0.002,
    "sig_p": 0.02,
    "df": 4.0,
    "emission": "gauss",
    "lam": 1.0,
    "sigma_mode": "std",
    "start_sig": 0.75,
    "r0_sig": 0.01,
    "band_pad": 100.0,
    "mom": 0.998,
    "rate_center": "zero",
    "position_kernel_cells": 5,
}
EXPECTED_VARIANTS = ("memoryless_41rate", "dz_only_r0")
EXPECTED_STATIONARY_SD = 0.002 / math.sqrt(1.0 - 0.998**2)


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> dict[str, int]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp435 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp435 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp435 scientific parent changed")
    if get_nested(config, "lineage.evidence_parent") != EVIDENCE_EXPERIMENT:
        raise ValueError("exp435 evidence parent changed")
    if not bool(get_nested(config, "design.implementation_authorized", False)):
        raise RuntimeError("exp435 implementation is not authorized")
    if not bool(get_nested(config, "design.canonical_notebook_adoption_authorized", False)):
        raise RuntimeError("canonical exp435 notebook adoption is not authorized")
    if bool(get_nested(config, "design.kaggle_stage_1_authorized", True)):
        raise ValueError("Stage 1 must remain disabled during Stage 0")
    if bool(get_nested(config, "design.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "design.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.enable_gpu", True)):
        raise ValueError("exp435 is CPU-only")
    if bool(get_nested(config, "runtime.enable_internet", True)):
        raise ValueError("exp435 must run with internet disabled")
    if get_nested(config, "execution.selected_stage") != "stage_0_fixed32":
        raise ValueError("selected_stage must remain stage_0_fixed32")

    expected = {
        "active_scientific_variants": 2,
        "stage_0_treatment_variants": 2,
        "stage_0_wells_per_treatment": 32,
        "stage_0_treatment_hmm_well_runs": 64,
        "parent_control_hmm_reruns_stage_0": 0,
        "stage_1_max_treatment_variants": 2,
        "stage_1_wells_per_treatment": 773,
        "stage_1_max_treatment_hmm_well_runs": 1546,
        "parent_control_hmm_reruns_stage_1": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = {key: int(get_nested(config, f"execution.{key}", -1)) for key in expected}
    if observed != expected:
        raise ValueError(f"Stage 0 execution contract changed: {observed} != {expected}")
    if bool(get_nested(config, "execution.create_submission", True)):
        raise ValueError("submission creation must remain disabled")
    if require_run_authorization:
        if not bool(get_nested(config, "design.kaggle_stage_0_authorized", False)):
            raise RuntimeError(
                "implementation approval does not authorize exp435 Stage 0 execution"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("execution.run_hmm is false")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError("execution.create_prediction is false")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    shared = get_nested(config, "model.shared_hmm")
    variants = get_nested(config, "model.active_scientific_variants")
    state = get_nested(config, "model.state_contract")
    memoryless = get_nested(config, "model.variants.memoryless_41rate")
    dz_only = get_nested(config, "model.variants.dz_only_r0")
    if shared != EXPECTED_SHARED_HMM:
        raise ValueError(f"shared exp209 HMM contract changed: {shared}")
    if tuple(variants or ()) != EXPECTED_VARIANTS:
        raise ValueError(f"scientific variants changed: {variants}")
    if state != {
        "persistent_state": "tvt_probability_distribution",
        "point_state_forbidden": True,
        "edge_variable": "u_rate",
        "edge_variable_persisted": False,
        "transition_formula": "delta_tvt_equals_u_rate_times_delta_md_minus_delta_z",
    }:
        raise ValueError(f"TVT-only state contract changed: {state}")
    if memoryless["n_edge_rates"] != 41:
        raise ValueError("memoryless treatment must retain 41 edge rates")
    if memoryless["rate_weight_family"] != (
        "zero_centered_parent_ar1_stationary_gaussian"
    ):
        raise ValueError("memoryless rate weights changed")
    if abs(float(memoryless["stationary_sd"]) - EXPECTED_STATIONARY_SD) > 1.0e-15:
        raise ValueError("stationary rate standard deviation changed")
    if bool(memoryless["uses_init_rate_as_weight_mean"]):
        raise ValueError("prefix init_rate cannot center memoryless weights")
    if not bool(memoryless["uses_init_rate_for_support_only"]):
        raise ValueError("prefix init_rate must remain support-only")
    if bool(memoryless["carries_rate_responsibility_to_next_row"]):
        raise ValueError("rate responsibility cannot persist")
    if dz_only["edge_rates"] != [0.0] or dz_only["edge_rate_weights"] != [1.0]:
        raise ValueError("dz-only delta-at-zero contract changed")
    if bool(dz_only["carries_rate_responsibility_to_next_row"]):
        raise ValueError("dz-only cannot persist a rate state")
    if get_nested(config, "validation.truth_join") != (
        "after_all_variant_predictions_and_diagnostics_freeze"
    ):
        raise ValueError("truth-late contract changed")
    return {
        "shared_hmm": shared,
        "state_contract": state,
        "active_scientific_variants": list(variants),
        "memoryless_stationary_sd": float(memoryless["stationary_sd"]),
        "memoryless_weight_mean": 0.0,
        "dz_only_rates": list(dz_only["edge_rates"]),
        "rate_responsibility_persisted": False,
        "truth_join": get_nested(config, "validation.truth_join"),
    }


# %% [markdown]
# ## 2. Notebook-safe paths, SHA, and leakage ledger


# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp435 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


def artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        target = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        target = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    return target


def metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_float_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(np.float64)
        elif pd.api.types.is_integer_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(np.int64)
        elif pd.api.types.is_bool_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(bool)
        else:
            normalized[column] = normalized[column].astype(str)
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    frame.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "logical_sha256": logical_frame_sha256(frame),
        "rows": len(frame),
    }


def write_deterministic_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw,
        compresslevel=1,
        mtime=0,
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    try:
        frame.to_csv(text, index=False, lineterminator="\n")
    finally:
        text.flush()
        text.close()
        compressed.close()
        raw.close()
    readback = pd.read_csv(path, float_precision="round_trip")
    return {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
        "logical_sha256": logical_frame_sha256(frame),
        "readback_logical_sha256": logical_frame_sha256(readback),
        "rows": len(frame),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024**3)
    return value / (1024**2)


def runtime_versions() -> dict[str, str]:
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
    }


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = (
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "assets" / filename,
        find_project_root() / local_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"bootstrap asset not found: {filename}")


@dataclass
class LeakageLedger:
    frozen_wells: set[str] = field(default_factory=set)
    scope_rows: int = 0
    target_free_rows: int = 0
    truth_rows_before_all_freeze: int = 0
    episode_rows_before_all_freeze: int = 0
    role_fold_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    episode_rows_after_all_freeze: int = 0
    role_fold_rows_after_all_freeze: int = 0
    expected_wells: int = 32

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def record_scope(self, rows: int) -> None:
        self.scope_rows += int(rows)

    def record_target_free(self, rows: int) -> None:
        self.target_free_rows += int(rows)

    def freeze(self, well: str) -> None:
        self.frozen_wells.add(str(well))

    def record_truth_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.truth_rows_before_all_freeze += int(rows)
            raise RuntimeError("truth was read before all fixed32 predictions were frozen")
        self.truth_rows_after_all_freeze += int(rows)

    def record_episode_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.episode_rows_before_all_freeze += int(rows)
            raise RuntimeError("episodes were read before all fixed32 predictions were frozen")
        self.episode_rows_after_all_freeze += int(rows)

    def record_role_fold_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.role_fold_rows_before_all_freeze += int(rows)
            raise RuntimeError(
                "role/fold identity was read before all fixed32 predictions were frozen"
            )
        self.role_fold_rows_after_all_freeze += int(rows)


# %% [markdown]
# ## 3. Fixed32 manifest, saved parent, and target-free raw inputs
#
# The manifest selects wells, but its `role` and `fold` fields are attached only
# after both HMM variants for a well are frozen. Decoder frames explicitly omit
# unknown-suffix truth, error, episode, and role columns.


# %%
def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.is_dir():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    return find_project_root() / str(get_nested(config, "data.train_dir"))


def load_fixed32_manifest(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.stage_0_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed: {observed}")
    header = pd.read_csv(
        path,
        nrows=0,
    )
    expected_columns = {
        "well",
        "role",
        "fold",
        "prefix_rows",
        "suffix_rows",
        "selection_hash",
    }
    if not expected_columns.issubset(header.columns):
        raise ValueError("fixed32 manifest schema changed")
    frame = pd.read_csv(
        path,
        usecols=["well", "prefix_rows", "suffix_rows"],
        dtype={"well": str, "matched_persistent_well": str},
    )
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 manifest must contain 32 unique wells")
    ledger.record_scope(len(frame))
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "logical_sha256": logical_frame_sha256(frame),
        "mechanism_only_not_cv_or_promotion": True,
    }


def load_fixed32_scope_after_all_freeze(
    config: Mapping[str, Any],
    execution_manifest: pd.DataFrame,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    spec = get_nested(config, "data.stage_0_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed after freeze: {observed}")
    frame = pd.read_csv(
        path,
        dtype={"well": str, "matched_persistent_well": str},
    )
    ledger.record_role_fold_late(len(frame))
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 scope must contain 32 unique wells")
    if frame["role"].value_counts().to_dict() != {"persistent": 16, "control": 16}:
        raise ValueError("fixed32 role counts changed")
    expected_fold_counts = {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}
    if frame.groupby("fold").size().to_dict() != expected_fold_counts:
        raise ValueError("fixed32 fold counts changed")
    if set(frame.loc[frame["role"].eq("persistent"), "fold"].astype(int)) != set(range(5)):
        raise ValueError("persistent fixed32 wells must cover all five folds")
    if set(frame["well"].astype(str)) != set(execution_manifest["well"].astype(str)):
        raise ValueError("fixed32 execution/scope well identity changed")
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


def parent_row_indices_from_cache_ids(frame: pd.DataFrame) -> np.ndarray:
    row_indices = np.empty(len(frame), dtype=np.int64)
    for offset, (well, identifier) in enumerate(
        zip(frame["well"].astype(str), frame["id"].astype(str), strict=True)
    ):
        prefix = f"{well}_"
        if not identifier.startswith(prefix):
            raise ValueError(f"saved parent id does not start with exact well prefix: {identifier}")
        suffix = identifier[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"saved parent id has invalid row suffix: {identifier}")
        row_indices[offset] = int(suffix)
    return row_indices


def parent_cache_ids_for_rows(well: str, row_indices: np.ndarray) -> np.ndarray:
    well = str(well)
    rows = np.asarray(row_indices, dtype=np.int64)
    if not well or rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("invalid well or row indices for parent cache ids")
    return np.asarray([f"{well}_{int(row)}" for row in rows], dtype=str)


def saved_float32_parity_max_abs_diff(
    observed: np.ndarray,
    saved: np.ndarray,
) -> float:
    observed32 = np.asarray(observed, dtype=np.float32)
    saved32 = np.asarray(saved, dtype=np.float32)
    if observed32.shape != saved32.shape:
        raise ValueError("saved exp209 parity shapes differ")
    if not (np.isfinite(observed32).all() and np.isfinite(saved32).all()):
        raise ValueError("saved exp209 parity inputs must be finite")
    return float(np.max(np.abs(observed32.astype(np.float64) - saved32.astype(np.float64))))


def candidate_parent_paths(spec: Mapping[str, Any]) -> list[Path]:
    root = find_project_root()
    filename = str(spec["filename"])
    matches: list[Path] = []
    for value in spec["candidates"]:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            matches.append(candidate)
        elif (candidate / filename).is_file():
            matches.append(candidate / filename)
    if KAGGLE_INPUT_ROOT.is_dir():
        for pattern in spec["patterns"]:
            matches.extend(KAGGLE_INPUT_ROOT.glob(str(pattern)))
    return sorted({path.resolve() for path in matches if path.is_file()})


def load_saved_parent_predictions(
    config: Mapping[str, Any],
    target_wells: set[str],
    expected_rows: int,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp209_saved_control")
    candidates = candidate_parent_paths(spec)
    expected_sha = str(spec["expected_decompressed_sha256"])
    matching = [path for path in candidates if sha256_decompressed_csv(path) == expected_sha]
    if not matching:
        raise FileNotFoundError("SHA-matching saved exp209 parent prediction cache was not found")
    path = matching[0]
    columns = ["id", "well", str(spec["prediction_column"])]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"id": str, "well": str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("saved parent contains none of the fixed32 wells")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.rename(columns={str(spec["prediction_column"]): "parent_prediction"})
    frame["row_idx"] = parent_row_indices_from_cache_ids(frame)
    frame["parent_prediction"] = pd.to_numeric(frame["parent_prediction"], errors="raise")
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if len(frame) != expected_rows:
        raise ValueError(f"saved parent rows={len(frame)}/{expected_rows}")
    if frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("saved parent keys are not unique")
    ledger.record_target_free(len(frame))
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": expected_sha,
        "rows": len(frame),
        "matching_candidates": len(matching),
    }


def load_target_free_well(
    well: str,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=lambda column: str(column) != "TVT",
    )
    forbidden = FORBIDDEN_DECODER_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: decoder input contains {sorted(forbidden)}")
    typewell = pd.read_csv(typewell_path).sort_values("TVT").reset_index(drop=True)
    ledger.record_target_free(len(horizontal) + len(typewell))
    return horizontal, typewell


# %% [markdown]
# ## 4. Exact exp209 HMM input preparation
#
# The following calibration, sigma, interpolation, grid, initial-rate, and
# emission construction are copied semantically from the exp209 exact-HMM path.
# Both variants consume the same prepared arrays.


# %%
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


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError("horizontal input schema changed")
    if not required_typewell.issubset(typewell.columns):
        raise ValueError("typewell input schema changed")
    if "TVT" in horizontal.columns:
        raise ValueError("unknown-suffix TVT reached HMM preparation")

    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(eval_rows) == 0:
        raise ValueError("expected a visible prefix and non-empty suffix")
    cal_a, cal_b, robust_sigma, init_rate, rate_rows, valid_steps = prefix_stats(
        horizontal, typewell_tvt, typewell_gr
    )
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
    gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))

    step = float(hmm["step"])
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = np.interp(grid, typewell_tvt, typewell_gr)
    md = eval_rows["MD"].to_numpy(np.float64)
    z = eval_rows["Z"].to_numpy(np.float64)
    raw_gr = eval_rows["GR"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[eval_rows.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    span = max(float(hmm["rate_span"]), abs(init_rate) + 0.04)
    rates = np.linspace(-span, span, int(hmm["n_rates"]), dtype=np.float64)
    return {
        "emission_ll": emission_ll,
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "r0": float(init_rate),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "last_known_z": float(last["Z"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "cal_a": cal_a,
        "cal_b": cal_b,
        "robust_sigma_unused": robust_sigma,
    }


# %% [markdown]
# ## 5. TVT-only forward-backward and row-local rate marginalization
#
# The only persistent message is a probability vector over the TVT grid.
# For each row, every edge rate creates a five-cell position kernel. Fixed rate
# weights marginalize those kernels before the next TVT message is formed.
# The edge-rate output records the fixed prior moments used at each row. It is
# not a filtered/smoothed rate posterior and is never an input to the following
# row. This makes the absence of a persistent rate message directly auditable.


# %%
def stationary_rate_weights(
    rates: np.ndarray,
    *,
    sig_r: float,
    mom: float,
) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 1 or len(rates) == 0 or not np.isfinite(rates).all():
        raise ValueError("rates must be a finite non-empty vector")
    if not 0.0 <= float(mom) < 1.0:
        raise ValueError("stationary weights require 0 <= mom < 1")
    stationary_sd = float(sig_r) / math.sqrt(1.0 - float(mom) ** 2)
    raw = np.exp(-0.5 * (rates / stationary_sd) ** 2)
    weights = raw / raw.sum()
    if abs(float(weights.sum()) - 1.0) > 1.0e-14:
        raise RuntimeError("stationary rate weights did not normalize")
    return weights


@njit(cache=True, nogil=True)
def position_edge_kernels(
    rates: np.ndarray,
    dm: float,
    dz: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    rate_count = len(rates)
    offsets = np.empty((rate_count, 5), np.int64)
    probabilities = np.empty((rate_count, 5), np.float64)
    sigma_position = max(sig_p, 0.35 * step)
    maximum_row_sum_error = 0.0
    for rate_index in range(rate_count):
        expected_delta = rates[rate_index] * dm - dz
        center = int(np.floor(expected_delta / step + 0.5))
        total = 0.0
        for kernel_index in range(5):
            offset = center - 2 + kernel_index
            residual = offset * step - expected_delta
            probability = np.exp(-0.5 * (residual / sigma_position) ** 2)
            offsets[rate_index, kernel_index] = offset
            probabilities[rate_index, kernel_index] = probability
            total += probability
        for kernel_index in range(5):
            probabilities[rate_index, kernel_index] /= total
        row_sum = 0.0
        for kernel_index in range(5):
            row_sum += probabilities[rate_index, kernel_index]
        maximum_row_sum_error = max(maximum_row_sum_error, abs(row_sum - 1.0))
    return offsets, probabilities, maximum_row_sum_error


@njit(cache=True, nogil=True)
def mixed_position_kernel(
    rates: np.ndarray,
    rate_weights: np.ndarray,
    dm: float,
    dz: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    rate_offsets, rate_kernels, rate_row_error = position_edge_kernels(
        rates,
        dm,
        dz,
        step,
        sig_p,
    )
    minimum_offset = int(rate_offsets.min())
    maximum_offset = int(rate_offsets.max())
    offsets = np.arange(minimum_offset, maximum_offset + 1, dtype=np.int64)
    probabilities = np.zeros(len(offsets), np.float64)
    for rate_index in range(len(rates)):
        for kernel_index in range(5):
            index = int(rate_offsets[rate_index, kernel_index] - minimum_offset)
            probabilities[index] += (
                rate_weights[rate_index] * rate_kernels[rate_index, kernel_index]
            )
    total = probabilities.sum()
    row_error = max(rate_row_error, abs(total - 1.0))
    probabilities /= total
    return offsets, probabilities, row_error


@njit(cache=True, nogil=True)
def _tvt_only_forward_backward(
    emission_ll: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
    step: float,
    rates: np.ndarray,
    rate_weights: np.ndarray,
    sig_p: float,
    start_p: float,
    start_sig: float,
    lam: float,
):
    time_count, position_count = emission_ll.shape
    rate_count = len(rates)
    alpha = np.empty((time_count, position_count), np.float32)
    posterior = np.empty((time_count, position_count), np.float64)
    previous = np.empty(position_count, np.float64)
    predictive = np.empty(position_count, np.float64)
    current = np.empty(position_count, np.float64)
    likelihood = np.empty(position_count, np.float64)
    predictive_rate_mean = np.empty(time_count, np.float64)
    filtered_rate_mean = np.empty(time_count, np.float64)
    filtered_rate_std = np.empty(time_count, np.float64)
    filtered_rate_edge_mass = np.empty(time_count, np.float64)
    maximum_transition_row_sum_error = 0.0
    maximum_forward_normalization_error = 0.0
    log_likelihood = 0.0

    initial_total = 0.0
    for position_index in range(position_count):
        delta = (position_index - start_p) * step
        value = np.exp(-0.5 * (delta / start_sig) ** 2)
        previous[position_index] = value
        initial_total += value
    for position_index in range(position_count):
        previous[position_index] /= initial_total

    for time_index in range(time_count):
        offsets, kernel, row_error = mixed_position_kernel(
            rates,
            rate_weights,
            dm[time_index],
            dz[time_index],
            step,
            sig_p,
        )
        maximum_transition_row_sum_error = max(
            maximum_transition_row_sum_error,
            row_error,
        )
        for position_index in range(position_count):
            predictive[position_index] = 0.0
        for kernel_index in range(len(offsets)):
            offset = offsets[kernel_index]
            coefficient = kernel[kernel_index]
            for source_index in range(position_count):
                destination_index = source_index + offset
                if 0 <= destination_index < position_count:
                    predictive[destination_index] += (
                        previous[source_index] * coefficient
                    )

        emission_max = -1.0e300
        for position_index in range(position_count):
            emission_value = lam * float(emission_ll[time_index, position_index])
            emission_max = max(emission_max, emission_value)
        current_total = 0.0
        predictive_total = 0.0
        for position_index in range(position_count):
            likelihood[position_index] = np.exp(
                lam * float(emission_ll[time_index, position_index]) - emission_max
            )
            predictive_total += predictive[position_index]
            current[position_index] = (
                predictive[position_index] * likelihood[position_index]
            )
            current_total += current[position_index]
        if current_total <= 0.0 or not np.isfinite(current_total):
            raise RuntimeError("TVT-only forward message became non-finite")
        log_likelihood += emission_max + np.log(current_total)

        prior_mean = 0.0
        prior_second = 0.0
        prior_edge = 0.0
        for rate_index in range(rate_count):
            prior_mean += rate_weights[rate_index] * rates[rate_index]
            prior_second += rate_weights[rate_index] * rates[rate_index] ** 2
            if rate_index == 0 or rate_index == rate_count - 1:
                prior_edge += rate_weights[rate_index]
        predictive_rate_mean[time_index] = prior_mean
        filtered_rate_mean[time_index] = prior_mean
        filtered_variance = max(
            prior_second - prior_mean**2,
            0.0,
        )
        filtered_rate_std[time_index] = np.sqrt(filtered_variance)
        filtered_rate_edge_mass[time_index] = prior_edge

        normalized_total = 0.0
        for position_index in range(position_count):
            normalized = current[position_index] / current_total
            alpha[time_index, position_index] = np.float32(normalized)
            previous[position_index] = normalized
            normalized_total += normalized
        maximum_forward_normalization_error = max(
            maximum_forward_normalization_error,
            abs(normalized_total - 1.0),
        )

    for position_index in range(position_count):
        posterior[time_count - 1, position_index] = float(
            alpha[time_count - 1, position_index]
        )
    last_total = posterior[time_count - 1].sum()
    posterior[time_count - 1] /= last_total

    beta_next = np.ones(position_count, np.float64)
    beta_current = np.empty(position_count, np.float64)
    for time_index in range(time_count - 1, 0, -1):
        offsets, kernel, _ = mixed_position_kernel(
            rates,
            rate_weights,
            dm[time_index],
            dz[time_index],
            step,
            sig_p,
        )
        emission_max = -1.0e300
        for position_index in range(position_count):
            emission_value = lam * float(emission_ll[time_index, position_index])
            emission_max = max(emission_max, emission_value)
        for position_index in range(position_count):
            likelihood[position_index] = np.exp(
                lam * float(emission_ll[time_index, position_index]) - emission_max
            )
        for source_index in range(position_count):
            total = 0.0
            for kernel_index in range(len(offsets)):
                destination_index = source_index + offsets[kernel_index]
                if 0 <= destination_index < position_count:
                    total += (
                        kernel[kernel_index]
                        * likelihood[destination_index]
                        * beta_next[destination_index]
                    )
            beta_current[source_index] = total
        beta_scale = beta_current.max()
        if beta_scale <= 0.0 or not np.isfinite(beta_scale):
            raise RuntimeError("TVT-only backward message became non-finite")
        for position_index in range(position_count):
            beta_current[position_index] /= beta_scale
        posterior_total = 0.0
        for position_index in range(position_count):
            value = (
                float(alpha[time_index - 1, position_index])
                * beta_current[position_index]
            )
            posterior[time_index - 1, position_index] = value
            posterior_total += value
        for position_index in range(position_count):
            posterior[time_index - 1, position_index] /= posterior_total
            beta_next[position_index] = beta_current[position_index]

    maximum_posterior_normalization_error = 0.0
    for time_index in range(time_count):
        maximum_posterior_normalization_error = max(
            maximum_posterior_normalization_error,
            abs(posterior[time_index].sum() - 1.0),
        )
    return (
        posterior,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        filtered_rate_std,
        filtered_rate_edge_mass,
        maximum_transition_row_sum_error,
        maximum_forward_normalization_error,
        maximum_posterior_normalization_error,
    )


def run_tvt_only_hmm(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    *,
    rates: np.ndarray,
    rate_weights: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    rates = np.asarray(rates, dtype=np.float64)
    rate_weights = np.asarray(rate_weights, dtype=np.float64)
    if rates.shape != rate_weights.shape:
        raise ValueError("rate support and weights must have the same shape")
    if not np.isfinite(rate_weights).all() or np.any(rate_weights < 0.0):
        raise ValueError("rate weights must be finite and nonnegative")
    rate_weights = rate_weights / rate_weights.sum()
    result = _tvt_only_forward_backward(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(hmm["step"]),
        rates,
        rate_weights,
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(hmm["lam"]),
    )
    (
        posterior,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        filtered_rate_std,
        filtered_rate_edge_mass,
        transition_row_sum_error,
        forward_normalization_error,
        posterior_normalization_error,
    ) = result
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = posterior @ grid
    posterior_second = posterior @ (grid * grid)
    posterior_std = np.sqrt(
        np.maximum(posterior_second - posterior_mean * posterior_mean, 0.0)
    )
    prediction_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        posterior_mean=np.asarray(posterior_mean, dtype=np.float32),
        posterior_std=np.asarray(posterior_std, dtype=np.float32),
    )
    diagnostic_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        predictive_rate_mean=np.asarray(predictive_rate_mean, dtype=np.float64),
        filtered_rate_mean=np.asarray(filtered_rate_mean, dtype=np.float64),
        filtered_rate_std=np.asarray(filtered_rate_std, dtype=np.float64),
        filtered_rate_edge_mass=np.asarray(filtered_rate_edge_mass, dtype=np.float64),
    )
    return {
        "posterior_mean": np.asarray(posterior_mean, dtype=np.float64),
        "posterior_std": np.asarray(posterior_std, dtype=np.float64),
        "log_likelihood": float(log_likelihood),
        "predictive_rate_mean": np.asarray(predictive_rate_mean, dtype=np.float64),
        "filtered_rate_mean": np.asarray(filtered_rate_mean, dtype=np.float64),
        "filtered_rate_std": np.asarray(filtered_rate_std, dtype=np.float64),
        "filtered_rate_edge_mass": np.asarray(
            filtered_rate_edge_mass, dtype=np.float64
        ),
        "transition_row_sum_max_error": float(transition_row_sum_error),
        "posterior_normalization_max_error": max(
            float(forward_normalization_error),
            float(posterior_normalization_error),
        ),
        "persistent_state_shape": (len(prepared["eval_index"]), len(grid)),
        "edge_rate_count": len(rates),
        "prediction_sha256": prediction_sha,
        "diagnostic_sha256": diagnostic_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def variant_rate_contract(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == "memoryless_41rate":
        rates = np.asarray(prepared["rates"], dtype=np.float64)
        weights = stationary_rate_weights(
            rates,
            sig_r=float(hmm["sig_r"]),
            mom=float(hmm["mom"]),
        )
        return rates, weights
    if variant == "dz_only_r0":
        return np.asarray([0.0], dtype=np.float64), np.asarray([1.0], dtype=np.float64)
    raise ValueError(f"unknown scientific variant: {variant}")


def run_hmm_variant(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    rates, weights = variant_rate_contract(prepared, hmm, variant)
    result = run_tvt_only_hmm(
        prepared,
        hmm,
        rates=rates,
        rate_weights=weights,
    )
    result["variant"] = variant
    result["rate_support"] = rates
    result["rate_weights"] = weights
    return result


# %% [markdown]
# ## 6. Two-treatment decoding and target-free freeze
#
# Each fixed32 well is prepared once and decoded by both TVT-only treatments.
# Saved exp209 is loaded as a frozen comparison and is never rerun. Role/fold,
# suffix truth, episode ranges, and exp408 causes are not accepted by the
# decoder. All 32 wells and both variant diagnostics are frozen before any of
# those fields are read.


# %%
def synthetic_transition_contract(
    shared_hmm: Mapping[str, Any],
) -> dict[str, Any]:
    rates = np.linspace(
        -float(shared_hmm["rate_span"]),
        float(shared_hmm["rate_span"]),
        int(shared_hmm["n_rates"]),
        dtype=np.float64,
    )
    weights = stationary_rate_weights(
        rates,
        sig_r=float(shared_hmm["sig_r"]),
        mom=float(shared_hmm["mom"]),
    )
    _, memoryless_kernel, memoryless_row_error = position_edge_kernels(
        rates,
        dm=1.0,
        dz=0.17,
        step=float(shared_hmm["step"]),
        sig_p=float(shared_hmm["sig_p"]),
    )
    _, dz_kernel, dz_row_error = position_edge_kernels(
        np.asarray([0.0], dtype=np.float64),
        dm=1.0,
        dz=0.17,
        step=float(shared_hmm["step"]),
        sig_p=float(shared_hmm["sig_p"]),
    )
    return {
        "memoryless_rate_count": len(rates),
        "memoryless_weight_sum": float(weights.sum()),
        "memoryless_weighted_rate_mean": float(weights @ rates),
        "memoryless_stationary_sd": float(shared_hmm["sig_r"])
        / math.sqrt(1.0 - float(shared_hmm["mom"]) ** 2),
        "memoryless_position_kernel_row_sum_max_error": max(
            float(memoryless_row_error),
            float(np.max(np.abs(memoryless_kernel.sum(axis=1) - 1.0))),
        ),
        "dz_rate_count": 1,
        "dz_weight_sum": 1.0,
        "dz_position_kernel_row_sum_max_error": max(
            float(dz_row_error),
            float(np.max(np.abs(dz_kernel.sum(axis=1) - 1.0))),
        ),
        "persistent_state": "tvt_probability_distribution",
        "persistent_rate_state_cells": 0,
        "rate_responsibility_persisted": False,
        "pass": bool(
            len(rates) == 41
            and abs(float(weights.sum()) - 1.0) <= 1.0e-14
            and abs(float(weights @ rates)) <= 1.0e-15
            and memoryless_row_error <= 1.0e-14
            and dz_row_error <= 1.0e-14
        ),
    }


@dataclass
class FrozenWell:
    well: str
    role: str
    fold: int
    eval_id: np.ndarray
    row_idx: np.ndarray
    raw_gr_missing: np.ndarray
    parent_prediction: np.ndarray
    memoryless_prediction: np.ndarray
    dz_only_prediction: np.ndarray
    memoryless_posterior_std: np.ndarray
    dz_only_posterior_std: np.ndarray
    memoryless_filtered_rate_mean: np.ndarray
    memoryless_filtered_rate_std: np.ndarray
    memoryless_filtered_rate_edge_mass: np.ndarray
    dz_only_filtered_rate_mean: np.ndarray
    dz_only_filtered_rate_std: np.ndarray
    dz_only_filtered_rate_edge_mass: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    memoryless_prediction_sha256: str
    dz_only_prediction_sha256: str
    memoryless_diagnostic_sha256: str
    dz_only_diagnostic_sha256: str
    dz_delta_rate_parity_max_abs_ft: float
    maximum_transition_row_sum_error: float
    maximum_posterior_normalization_error: float
    memoryless_log_likelihood: float
    dz_only_log_likelihood: float
    memoryless_elapsed_seconds: float
    dz_only_elapsed_seconds: float
    prefix_rows: int


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    saved_parent: pd.DataFrame,
    shared_hmm: Mapping[str, Any],
    dz_parity_tolerance_ft: float,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, shared_hmm)
    parent = saved_parent.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    eval_id = parent_cache_ids_for_rows(well, row_idx)
    if not np.array_equal(parent["row_idx"].to_numpy(np.int64), row_idx):
        raise ValueError(f"{well}: parent row index does not align with raw suffix")
    if not np.array_equal(parent["id"].astype(str).to_numpy(), eval_id):
        raise ValueError(f"{well}: parent id does not align with raw suffix")
    parent_prediction = parent["parent_prediction"].to_numpy(np.float64)
    memoryless = run_hmm_variant(
        prepared,
        shared_hmm,
        "memoryless_41rate",
    )
    dz_only = run_hmm_variant(prepared, shared_hmm, "dz_only_r0")
    explicit_rates, explicit_weights = variant_rate_contract(
        prepared,
        shared_hmm,
        "dz_only_r0",
    )
    if not np.array_equal(explicit_rates, np.asarray([0.0], dtype=np.float64)):
        raise RuntimeError("dz-only support is not delta-at-zero")
    if not np.array_equal(explicit_weights, np.asarray([1.0], dtype=np.float64)):
        raise RuntimeError("dz-only weights are not a unit delta")
    dz_parity = 0.0
    if not np.array_equal(
        np.asarray(dz_only["rate_support"], dtype=np.float64),
        explicit_rates,
    ):
        raise RuntimeError("dz-only execution support differs from its contract")
    if not np.array_equal(
        np.asarray(dz_only["rate_weights"], dtype=np.float64),
        explicit_weights,
    ):
        raise RuntimeError("dz-only execution weights differ from its contract")
    if dz_parity > float(dz_parity_tolerance_ft):
        raise RuntimeError(f"{well}: dz-only delta-rate parity failed: {dz_parity}")
    ledger.freeze(well)
    return FrozenWell(
        well=str(well),
        role="",
        fold=-1,
        eval_id=eval_id,
        row_idx=row_idx,
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        parent_prediction=parent_prediction,
        memoryless_prediction=np.asarray(
            memoryless["posterior_mean"], dtype=np.float64
        ),
        dz_only_prediction=np.asarray(dz_only["posterior_mean"], dtype=np.float64),
        memoryless_posterior_std=np.asarray(
            memoryless["posterior_std"], dtype=np.float64
        ),
        dz_only_posterior_std=np.asarray(dz_only["posterior_std"], dtype=np.float64),
        memoryless_filtered_rate_mean=np.asarray(
            memoryless["filtered_rate_mean"], dtype=np.float64
        ),
        memoryless_filtered_rate_std=np.asarray(
            memoryless["filtered_rate_std"], dtype=np.float64
        ),
        memoryless_filtered_rate_edge_mass=np.asarray(
            memoryless["filtered_rate_edge_mass"], dtype=np.float64
        ),
        dz_only_filtered_rate_mean=np.asarray(
            dz_only["filtered_rate_mean"], dtype=np.float64
        ),
        dz_only_filtered_rate_std=np.asarray(
            dz_only["filtered_rate_std"], dtype=np.float64
        ),
        dz_only_filtered_rate_edge_mass=np.asarray(
            dz_only["filtered_rate_edge_mass"], dtype=np.float64
        ),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        memoryless_prediction_sha256=str(memoryless["prediction_sha256"]),
        dz_only_prediction_sha256=str(dz_only["prediction_sha256"]),
        memoryless_diagnostic_sha256=str(memoryless["diagnostic_sha256"]),
        dz_only_diagnostic_sha256=str(dz_only["diagnostic_sha256"]),
        dz_delta_rate_parity_max_abs_ft=dz_parity,
        maximum_transition_row_sum_error=max(
            float(memoryless["transition_row_sum_max_error"]),
            float(dz_only["transition_row_sum_max_error"]),
        ),
        maximum_posterior_normalization_error=max(
            float(memoryless["posterior_normalization_max_error"]),
            float(dz_only["posterior_normalization_max_error"]),
        ),
        memoryless_log_likelihood=float(memoryless["log_likelihood"]),
        dz_only_log_likelihood=float(dz_only["log_likelihood"]),
        memoryless_elapsed_seconds=float(memoryless["elapsed_seconds"]),
        dz_only_elapsed_seconds=float(dz_only["elapsed_seconds"]),
        prefix_rows=int(prepared["prefix_rows"]),
    )


def attach_scope_identity(
    frozen: FrozenWell,
    manifest_row: pd.Series,
) -> FrozenWell:
    frozen.role = str(manifest_row["role"])
    frozen.fold = int(manifest_row["fold"])
    return frozen


def prediction_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "id": item.eval_id,
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "parent_prediction": item.parent_prediction,
                    "memoryless_41rate_prediction": item.memoryless_prediction,
                    "dz_only_r0_prediction": item.dz_only_prediction,
                    "memoryless_41rate_posterior_std": (
                        item.memoryless_posterior_std
                    ),
                    "dz_only_r0_posterior_std": item.dz_only_posterior_std,
                }
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def rate_readout_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "suffix_offset": np.arange(len(item.row_idx), dtype=np.int64),
                    "memoryless_edge_rate_prior_mean": (
                        item.memoryless_filtered_rate_mean
                    ),
                    "memoryless_edge_rate_prior_std": (
                        item.memoryless_filtered_rate_std
                    ),
                    "memoryless_edge_rate_prior_edge_mass": (
                        item.memoryless_filtered_rate_edge_mass
                    ),
                    "dz_only_edge_rate_prior_mean": (
                        item.dz_only_filtered_rate_mean
                    ),
                    "dz_only_edge_rate_prior_std": item.dz_only_filtered_rate_std,
                    "dz_only_edge_rate_prior_edge_mass": (
                        item.dz_only_filtered_rate_edge_mass
                    ),
                }
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def combined_well_sha(
    frozen_wells: Sequence[FrozenWell],
    attribute: str,
) -> str:
    rows = [
        {"well": item.well, "sha256": str(getattr(item, attribute))}
        for item in sorted(frozen_wells, key=lambda value: value.well)
    ]
    return hashlib.sha256(stable_json_bytes(rows)).hexdigest()


# %% [markdown]
# ## 7. Truth-late persistent-episode and safety readout
#
# Suffix truth and the exp408 episode ledger are opened only after the
# predictions and rate messages for all 32 wells have content SHAs. The
# smoothed-rate under-response definition is fixed as: true and estimated rate
# have the same nonzero sign, while `abs(estimated) < abs(true)`.


# %%
def load_truth_after_all_freeze(
    frozen: FrozenWell,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{frozen.well}__horizontal_well.csv",
        usecols=["MD", "Z", "TVT", "TVT_input"],
    )
    suffix = frame.loc[frame["TVT_input"].isna()].copy()
    ledger.record_truth_late(len(suffix))
    if not np.array_equal(suffix.index.to_numpy(np.int64), frozen.row_idx):
        raise ValueError(f"{frozen.well}: truth row index changed after freeze")
    reconstructed_ids = parent_cache_ids_for_rows(
        frozen.well,
        suffix.index.to_numpy(np.int64),
    )
    if not np.array_equal(reconstructed_ids, frozen.eval_id):
        raise ValueError(f"{frozen.well}: truth id changed after freeze")
    suffix["id"] = reconstructed_ids
    return suffix.reset_index(names="row_idx")


def physical_true_interval_rate(
    truth: pd.DataFrame,
    *,
    last_known_tvt: float,
    last_known_md: float,
    last_known_z: float,
) -> np.ndarray:
    tvt = truth["TVT"].to_numpy(np.float64)
    md = truth["MD"].to_numpy(np.float64)
    z = truth["Z"].to_numpy(np.float64)
    dtvt = np.diff(np.concatenate([[float(last_known_tvt)], tvt]))
    dz = np.diff(np.concatenate([[float(last_known_z)], z]))
    dmd = np.diff(np.concatenate([[float(last_known_md)], md]))
    rate = np.full(len(truth), np.nan, dtype=np.float64)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    rate[valid] = (dtvt[valid] + dz[valid]) / dmd[valid]
    return rate


def zero_direction_underresponse_stats(
    true_rate: np.ndarray,
    estimated_rate: np.ndarray,
) -> dict[str, Any]:
    true_rate = np.asarray(true_rate, dtype=np.float64)
    estimated_rate = np.asarray(estimated_rate, dtype=np.float64)
    if true_rate.shape != estimated_rate.shape:
        raise ValueError("rate readout shapes differ")
    valid = np.isfinite(true_rate) & np.isfinite(estimated_rate)
    same_nonzero_direction = (
        (np.sign(true_rate) == np.sign(estimated_rate))
        & (np.sign(true_rate) != 0.0)
        & (np.sign(estimated_rate) != 0.0)
    )
    under = valid & same_nonzero_direction & (np.abs(estimated_rate) < np.abs(true_rate))
    error = estimated_rate - true_rate
    total_sse = float(np.sum(error[valid] ** 2))
    under_sse = float(np.sum(error[under] ** 2))
    return {
        "valid_rows": int(valid.sum()),
        "underresponse_rows": int(under.sum()),
        "rate_error_sse": total_sse,
        "underresponse_sse": under_sse,
        "underresponse_sse_share": (under_sse / total_sse if total_sse > 0.0 else math.nan),
    }


def well_truth_late_metrics(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> list[dict[str, Any]]:
    actual = truth["TVT"].to_numpy(np.float64)
    parent_error = frozen.parent_prediction - actual
    true_rate = physical_true_interval_rate(
        truth,
        last_known_tvt=frozen.last_known_tvt,
        last_known_md=frozen.last_known_md,
        last_known_z=frozen.last_known_z,
    )
    parent_rmse = float(np.sqrt(np.mean(parent_error**2)))
    rows: list[dict[str, Any]] = []
    variant_contracts = (
        (
            "memoryless_41rate",
            frozen.memoryless_prediction,
            frozen.memoryless_filtered_rate_mean,
            frozen.memoryless_filtered_rate_edge_mass,
            frozen.memoryless_prediction_sha256,
            frozen.memoryless_diagnostic_sha256,
            frozen.memoryless_elapsed_seconds,
        ),
        (
            "dz_only_r0",
            frozen.dz_only_prediction,
            frozen.dz_only_filtered_rate_mean,
            frozen.dz_only_filtered_rate_edge_mass,
            frozen.dz_only_prediction_sha256,
            frozen.dz_only_diagnostic_sha256,
            frozen.dz_only_elapsed_seconds,
        ),
    )
    for (
        variant,
        prediction,
        rate_mean,
        rate_edge_mass,
        prediction_sha,
        diagnostic_sha,
        elapsed_seconds,
    ) in variant_contracts:
        error = np.asarray(prediction, dtype=np.float64) - actual
        rmse = float(np.sqrt(np.mean(error**2)))
        rate_readout = zero_direction_underresponse_stats(true_rate, rate_mean)
        rows.append(
            {
                "well": frozen.well,
                "role": frozen.role,
                "fold": frozen.fold,
                "variant": variant,
                "rows": len(actual),
                "parent_rmse_ft": parent_rmse,
                "variant_rmse_ft": rmse,
                "rmse_delta_vs_parent_ft": rmse - parent_rmse,
                "improved_vs_parent": rmse < parent_rmse,
                "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
                "rate_error_sse": rate_readout["rate_error_sse"],
                "underresponse_sse_share": rate_readout[
                    "underresponse_sse_share"
                ],
                "filtered_rate_edge_mass_mean": float(np.mean(rate_edge_mass)),
                "dz_delta_rate_parity_max_abs_ft": (
                    frozen.dz_delta_rate_parity_max_abs_ft
                ),
                "maximum_transition_row_sum_error": (
                    frozen.maximum_transition_row_sum_error
                ),
                "maximum_posterior_normalization_error": (
                    frozen.maximum_posterior_normalization_error
                ),
                "prediction_sha256": prediction_sha,
                "diagnostic_sha256": diagnostic_sha,
                "hmm_seconds": elapsed_seconds,
            }
        )
    return rows


def load_persistent_episodes_after_all_freeze(
    config: Mapping[str, Any],
    selected_persistent: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"persistent episode SHA changed: {observed}")
    frame = pd.read_csv(path, dtype={"well": str, "episode_id": str})
    frame = frame.loc[frame["well"].isin(selected_persistent)].copy()
    ledger.record_episode_late(len(frame))
    required = {
        "episode_id",
        "well",
        "start_row_idx",
        "end_row_idx_exclusive",
        "start_suffix_offset",
        "rows",
    }
    if not required.issubset(frame.columns):
        raise ValueError("persistent episode schema changed")
    if frame.empty or frame["well"].nunique() != len(selected_persistent):
        raise ValueError("selected persistent wells are missing episode rows")
    cause_spec = get_nested(config, "data.exp408_episode_causes")
    cause_path = resolve_bootstrap_asset(
        str(cause_spec["filename"]),
        str(cause_spec["local"]),
    )
    cause_sha = sha256_file(cause_path)
    if cause_sha != str(cause_spec["expected_sha256"]):
        raise ValueError(f"exp408 episode-cause SHA changed: {cause_sha}")
    causes = pd.read_csv(
        cause_path,
        usecols=["episode_id", "well", "fold", "cause"],
        dtype={"episode_id": str, "well": str},
    )
    causes = causes.loc[causes["well"].isin(selected_persistent)].copy()
    ledger.record_episode_late(len(causes))
    if causes["episode_id"].duplicated().any():
        raise ValueError("exp408 episode-cause identity is not unique")
    frame = frame.merge(
        causes,
        on=["episode_id", "well"],
        how="left",
        validate="one_to_one",
    )
    if frame["cause"].isna().any():
        raise ValueError("exp408 cause is missing for selected persistent episodes")
    return frame.sort_values(
        ["well", "start_suffix_offset"], kind="mergesort"
    ).reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "cause_path": str(cause_path),
        "cause_sha256": cause_sha,
        "selected_rows": len(frame),
        "selected_wells": frame["well"].nunique(),
    }


def episode_truth_late_readout(
    episodes: pd.DataFrame,
    frozen_by_well: Mapping[str, FrozenWell],
    truth_by_well: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        frozen = frozen_by_well[str(episode.well)]
        if int(episode.fold) != int(frozen.fold):
            raise ValueError(f"{episode.episode_id}: exp408/manifest fold changed")
        truth = truth_by_well[str(episode.well)]
        row_idx = truth["row_idx"].to_numpy(np.int64)
        mask = (row_idx >= int(episode.start_row_idx)) & (
            row_idx < int(episode.end_row_idx_exclusive)
        )
        offsets = np.flatnonzero(mask)
        if len(offsets) != int(episode.rows):
            raise ValueError(f"{episode.episode_id}: episode row coverage changed")
        actual = truth["TVT"].to_numpy(np.float64)[offsets]
        parent_error = frozen.parent_prediction[offsets] - actual
        parent_sse = float(np.sum(parent_error**2))
        for variant, prediction in (
            ("memoryless_41rate", frozen.memoryless_prediction),
            ("dz_only_r0", frozen.dz_only_prediction),
        ):
            variant_error = np.asarray(prediction, dtype=np.float64)[offsets] - actual
            variant_sse = float(np.sum(variant_error**2))
            rows.append(
                {
                    "episode_id": str(episode.episode_id),
                    "well": str(episode.well),
                    "fold": frozen.fold,
                    "cause": str(episode.cause),
                    "variant": variant,
                    "rows": len(offsets),
                    "start_row_idx": int(episode.start_row_idx),
                    "end_row_idx_exclusive": int(
                        episode.end_row_idx_exclusive
                    ),
                    "parent_sse": parent_sse,
                    "variant_sse": variant_sse,
                    "variant_sse_reduction_vs_parent": (
                        1.0 - variant_sse / parent_sse
                        if parent_sse > 0.0
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["variant", "fold", "well", "start_row_idx"], kind="mergesort"
    )


# %% [markdown]
# ## 8. Stage 0 gates, generated artifacts, and metrics
#
# Technical gates are shared. Mechanism gates are evaluated independently for
# the two treatments so one failure does not suppress the preregistered readout
# for the other. A passing treatment only becomes eligible for a separate
# Stage 1 approval discussion. The fixed32 result is not CV or promotion
# evidence, and this notebook contains no parameter or blend rescue path.


# %%
def fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def finite_readout_counts(
    frozen_wells: Sequence[FrozenWell],
) -> tuple[int, int]:
    finite = 0
    total = 0
    attributes = (
        "memoryless_prediction",
        "dz_only_prediction",
        "memoryless_posterior_std",
        "dz_only_posterior_std",
        "memoryless_filtered_rate_mean",
        "memoryless_filtered_rate_std",
        "memoryless_filtered_rate_edge_mass",
        "dz_only_filtered_rate_mean",
        "dz_only_filtered_rate_std",
        "dz_only_filtered_rate_edge_mass",
    )
    for item in frozen_wells:
        for attribute in attributes:
            array = np.asarray(getattr(item, attribute), dtype=np.float64)
            finite += int(np.isfinite(array).sum())
            total += int(array.size)
    return finite, total


def pooled_rmse_from_well_rows(
    frame: pd.DataFrame,
    column: str,
) -> float:
    weights = frame["rows"].to_numpy(np.float64)
    values = frame[column].to_numpy(np.float64)
    return float(np.sqrt(np.average(values**2, weights=weights)))


def evaluate_variant_mechanism_gates(
    *,
    variant: str,
    episode_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    mechanism_config: Mapping[str, Any],
    forward_cause: str,
) -> dict[str, Any]:
    episodes = episode_readout.loc[episode_readout["variant"].eq(variant)]
    wells = well_metrics.loc[well_metrics["variant"].eq(variant)]
    persistent_wells = wells.loc[wells["role"].eq("persistent")]
    control_wells = wells.loc[wells["role"].eq("control")]
    parent_episode_sse = float(episodes["parent_sse"].sum())
    variant_episode_sse = float(episodes["variant_sse"].sum())
    persistent_reduction = (
        1.0 - variant_episode_sse / parent_episode_sse
        if parent_episode_sse > 0.0
        else math.nan
    )
    forward = episodes.loc[episodes["cause"].eq(forward_cause)]
    forward_parent_sse = float(forward["parent_sse"].sum())
    forward_variant_sse = float(forward["variant_sse"].sum())
    forward_reduction = (
        1.0 - forward_variant_sse / forward_parent_sse
        if forward_parent_sse > 0.0
        else math.nan
    )
    improved_wells = int(persistent_wells["improved_vs_parent"].astype(bool).sum())
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_episodes = episodes.loc[episodes["fold"].eq(fold)]
        parent_sse = float(fold_episodes["parent_sse"].sum())
        variant_sse = float(fold_episodes["variant_sse"].sum())
        fold_rows.append(
            {
                "fold": fold,
                "episodes": len(fold_episodes),
                "parent_sse": parent_sse,
                "variant_sse": variant_sse,
                "improved": bool(
                    len(fold_episodes) > 0 and variant_sse < parent_sse
                ),
            }
        )
    improving_folds = int(sum(row["improved"] for row in fold_rows))
    control_parent_rmse = pooled_rmse_from_well_rows(
        control_wells,
        "parent_rmse_ft",
    )
    control_variant_rmse = pooled_rmse_from_well_rows(
        control_wells,
        "variant_rmse_ft",
    )
    control_delta = control_variant_rmse - control_parent_rmse
    control_p95 = float(
        np.quantile(
            control_wells["rmse_delta_vs_parent_ft"].to_numpy(np.float64),
            0.95,
        )
    )
    gates = {
        "forward_cause_episode_sse_reduction": bool(
            math.isfinite(forward_reduction)
            and forward_reduction
            >= float(
                mechanism_config[
                    "forward_cause_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_episode_sse_reduction": bool(
            math.isfinite(persistent_reduction)
            and persistent_reduction
            >= float(
                mechanism_config[
                    "persistent_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_improved_wells": bool(
            improved_wells
            >= int(mechanism_config["persistent_improved_wells_min"])
        ),
        "persistent_improving_folds": bool(
            improving_folds
            >= int(mechanism_config["persistent_improving_folds_min"])
        ),
        "matched_control_pooled_rmse": bool(
            control_delta
            <= float(
                mechanism_config["matched_control_pooled_rmse_delta_max_ft"]
            )
        ),
        "matched_control_by_well_p95": bool(
            control_p95
            <= float(
                mechanism_config[
                    "matched_control_by_well_delta_p95_max_ft"
                ]
            )
        ),
    }
    diagnostics = {
        "variant": variant,
        "parent_persistent_episode_sse": parent_episode_sse,
        "variant_persistent_episode_sse": variant_episode_sse,
        "persistent_episode_sse_reduction_fraction": persistent_reduction,
        "forward_cause": forward_cause,
        "forward_cause_episodes": len(forward),
        "forward_cause_parent_sse": forward_parent_sse,
        "forward_cause_variant_sse": forward_variant_sse,
        "forward_cause_episode_sse_reduction_fraction": forward_reduction,
        "persistent_improved_wells": improved_wells,
        "persistent_sse_by_fold": fold_rows,
        "persistent_improving_folds": improving_folds,
        "control_parent_rmse_ft": control_parent_rmse,
        "control_variant_rmse_ft": control_variant_rmse,
        "control_rmse_delta_ft": control_delta,
        "control_by_well_rmse_delta_p95_ft": control_p95,
    }
    return {
        "gates": gates,
        "all_mechanism_gates_pass": bool(all(gates.values())),
        "diagnostics": diagnostics,
    }


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    transition_contract: Mapping[str, Any],
    prediction_artifact: Mapping[str, Any],
    rate_artifact: Mapping[str, Any],
    episode_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "validation.stage_0.technical")
    mechanism_config = get_nested(config, "validation.stage_0.mechanism")
    expected_fold_counts = {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    maximum_transition_error = max(
        item.maximum_transition_row_sum_error for item in frozen_wells
    )
    maximum_posterior_error = max(
        item.maximum_posterior_normalization_error for item in frozen_wells
    )
    maximum_dz_parity = max(
        item.dz_delta_rate_parity_max_abs_ft for item in frozen_wells
    )
    finite_values, total_values = finite_readout_counts(frozen_wells)
    finite_coverage = fraction(finite_values, total_values)
    treatment_seconds = float(
        sum(
            item.memoryless_elapsed_seconds + item.dz_only_elapsed_seconds
            for item in frozen_wells
        )
    )
    runtime_projection = treatment_seconds * 773.0 / 32.0
    forbidden_reads = int(
        technical_config["role_fold_episode_error_reads_before_freeze"]
    )
    technical = {
        "fixed32_roles_and_unique_wells": bool(
            len(manifest) == 32
            and manifest["well"].nunique() == 32
            and manifest["role"].value_counts().to_dict()
            == {"persistent": 16, "control": 16}
        ),
        "fixed32_fold_counts": bool(
            manifest.groupby("fold").size().to_dict() == expected_fold_counts
        ),
        "transition_contract": bool(transition_contract["pass"]),
        "truth_reads_before_all_freeze": bool(
            ledger.truth_rows_before_all_freeze
            == int(technical_config["truth_reads_before_freeze"])
        ),
        "episode_reads_before_all_freeze": bool(
            ledger.episode_rows_before_all_freeze == forbidden_reads
        ),
        "role_fold_reads_before_all_freeze": bool(
            ledger.role_fold_rows_before_all_freeze == forbidden_reads
        ),
        "transition_row_sum": bool(
            maximum_transition_error
            <= float(technical_config["transition_row_sum_max_error"])
        ),
        "posterior_normalization": bool(
            maximum_posterior_error
            <= float(technical_config["posterior_normalization_max_error"])
        ),
        "dz_delta_rate_parity": bool(
            maximum_dz_parity
            <= float(technical_config["dz_delta_rate_parity_max_abs_ft"])
        ),
        "finite_prediction_and_diagnostic_coverage": bool(
            finite_coverage >= float(technical_config["finite_coverage"])
        ),
        "rate_responsibility_not_persisted": bool(
            transition_contract["persistent_rate_state_cells"] == 0
            and not transition_contract["rate_responsibility_persisted"]
        ),
        "prediction_readback_sha": bool(
            prediction_artifact["logical_sha256"]
            == prediction_artifact["readback_logical_sha256"]
        ),
        "diagnostic_readback_sha": bool(
            rate_artifact["logical_sha256"]
            == rate_artifact["readback_logical_sha256"]
        ),
        "runtime_projection": bool(
            runtime_projection
            <= float(
                technical_config[
                    "full_eligible_variants_runtime_projection_max_seconds"
                ]
            )
        ),
        "peak_rss": bool(
            peak_rss_gb() <= float(technical_config["peak_rss_max_gb"])
        ),
    }
    forward_cause = str(
        get_nested(config, "data.exp408_episode_causes.forward_cause")
    )
    variant_results: dict[str, Any] = {}
    eligible: list[str] = []
    for variant in EXPECTED_VARIANTS:
        result = evaluate_variant_mechanism_gates(
            variant=variant,
            episode_readout=episode_readout,
            well_metrics=well_metrics,
            mechanism_config=mechanism_config,
            forward_cause=forward_cause,
        )
        result["stage0_all_gates_pass"] = bool(
            all(technical.values()) and result["all_mechanism_gates_pass"]
        )
        result["stage1_eligible_for_separate_approval"] = result[
            "stage0_all_gates_pass"
        ]
        if result["stage0_all_gates_pass"]:
            eligible.append(variant)
        variant_results[variant] = result
    diagnostics = {
        "total_wells": len(frozen_wells),
        "total_suffix_rows": total_rows,
        "maximum_transition_row_sum_error": maximum_transition_error,
        "maximum_posterior_normalization_error": maximum_posterior_error,
        "maximum_dz_delta_rate_parity_abs_ft": maximum_dz_parity,
        "finite_coverage": finite_coverage,
        "stage0_elapsed_seconds": float(elapsed_seconds),
        "stage1_all_treatments_runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
        "truth_rows_before_all_freeze": ledger.truth_rows_before_all_freeze,
        "episode_rows_before_all_freeze": ledger.episode_rows_before_all_freeze,
        "role_fold_rows_before_all_freeze": (
            ledger.role_fold_rows_before_all_freeze
        ),
        "fixed32_is_mechanism_only_not_cv_or_promotion": True,
    }
    return {
        "technical": technical,
        "variants": variant_results,
        "diagnostics": diagnostics,
        "stage0_all_variants_pass": bool(
            len(eligible) == len(EXPECTED_VARIANTS)
        ),
        "stage1_eligible_variants_for_separate_approval": eligible,
        "fixed32_is_cv": False,
        "fixed32_is_promotion_evidence": False,
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP435_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp435 Stage 0 must run on Kaggle CPU; local execution is disabled")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha = hashlib.sha256(stable_json_bytes(scientific_contract)).hexdigest()
    set_num_threads(int(get_nested(config, "runtime.numba_threads_per_worker")))
    ledger = LeakageLedger(expected_wells=32)
    manifest, manifest_input = load_fixed32_manifest(config, ledger)
    target_wells = set(manifest["well"].astype(str))
    expected_rows = int(manifest["suffix_rows"].sum())
    parent, parent_input = load_saved_parent_predictions(
        config,
        target_wells,
        expected_rows,
        ledger,
    )
    shared_hmm = get_nested(config, "model.shared_hmm")
    transition_contract = synthetic_transition_contract(shared_hmm)
    if not transition_contract["pass"]:
        raise RuntimeError(f"transition mean contract failed: {transition_contract}")
    raw_dir = train_data_dir(config)
    parent_groups = parent.groupby("well", sort=False).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    dz_parity_tolerance = float(
        get_nested(
            config,
            "validation.stage_0.technical.dz_delta_rate_parity_max_abs_ft",
        )
    )

    for well_index, row in enumerate(manifest.itertuples(index=False), start=1):
        well = str(row.well)
        if well not in parent_groups:
            raise ValueError(f"{well}: saved parent rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            saved_parent=parent.iloc[parent_groups[well]].copy(),
            shared_hmm=shared_hmm,
            dz_parity_tolerance_ft=dz_parity_tolerance,
            ledger=ledger,
        )
        if len(frozen.row_idx) != int(row.suffix_rows):
            raise ValueError(f"{well}: suffix rows changed from fixed manifest")
        if frozen.prefix_rows != int(row.prefix_rows):
            raise ValueError(f"{well}: prefix rows changed from fixed manifest")
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(f"Stage 0 runtime hard guard exceeded: {elapsed}")
        if peak_rss_gb() > hard_rss:
            raise MemoryError(f"Stage 0 RSS hard guard exceeded: {peak_rss_gb()}")
        print(
            json.dumps(
                {
                    "event": "exp435_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "memoryless_hmm_seconds": frozen.memoryless_elapsed_seconds,
                    "dz_only_hmm_seconds": frozen.dz_only_elapsed_seconds,
                    "dz_delta_rate_parity_max_abs_ft": (
                        frozen.dz_delta_rate_parity_max_abs_ft
                    ),
                    "maximum_transition_row_sum_error": (
                        frozen.maximum_transition_row_sum_error
                    ),
                    "maximum_posterior_normalization_error": (
                        frozen.maximum_posterior_normalization_error
                    ),
                    "elapsed_seconds": elapsed,
                    "peak_rss_gb": peak_rss_gb(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all fixed32 wells were frozen")
    scope_manifest = load_fixed32_scope_after_all_freeze(
        config,
        manifest,
        ledger,
    )
    scope_by_well = scope_manifest.set_index("well", drop=False)
    for frozen in frozen_wells:
        attach_scope_identity(frozen, scope_by_well.loc[frozen.well])

    output = artifacts_dir()
    predictions = prediction_frame(frozen_wells)
    rate_readouts = rate_readout_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    rate_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_rate_readouts.csv.gz",
        rate_readouts,
    )
    if (
        prediction_artifact["logical_sha256"] != prediction_artifact["readback_logical_sha256"]
        or rate_artifact["logical_sha256"] != rate_artifact["readback_logical_sha256"]
    ):
        raise RuntimeError("target-free artifact readback SHA mismatch")

    frozen_by_well = {item.well: item for item in frozen_wells}
    truth_by_well: dict[str, pd.DataFrame] = {}
    well_metric_rows: list[dict[str, Any]] = []
    for item in frozen_wells:
        truth = load_truth_after_all_freeze(item, raw_dir, ledger)
        truth_by_well[item.well] = truth
        well_metric_rows.extend(well_truth_late_metrics(item, truth))
    selected_persistent = set(
        scope_manifest.loc[
            scope_manifest["role"].eq("persistent"),
            "well",
        ].astype(str)
    )
    episodes, episode_input = load_persistent_episodes_after_all_freeze(
        config,
        selected_persistent,
        ledger,
    )
    episode_readout = episode_truth_late_readout(
        episodes,
        frozen_by_well,
        truth_by_well,
    )
    well_metrics = pd.DataFrame(well_metric_rows).sort_values(
        ["variant", "fold", "role", "well"], kind="mergesort"
    )
    episode_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_episode_truth_late_readout.csv",
        episode_readout,
    )
    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    elapsed = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        manifest=scope_manifest,
        frozen_wells=frozen_wells,
        transition_contract=transition_contract,
        prediction_artifact=prediction_artifact,
        rate_artifact=rate_artifact,
        episode_readout=episode_readout,
        well_metrics=well_metrics,
        ledger=ledger,
        elapsed_seconds=elapsed,
    )
    input_manifest = {
        "fixed32_manifest": manifest_input,
        "saved_exp209_control": parent_input,
        "persistent_episodes": episode_input,
        "raw_train_dir": str(raw_dir),
        "scientific_contract_sha256": scientific_contract_sha,
        "leakage": {
            "scope_rows": ledger.scope_rows,
            "target_free_rows": ledger.target_free_rows,
            "frozen_wells": len(ledger.frozen_wells),
            "truth_rows_before_all_freeze": ledger.truth_rows_before_all_freeze,
            "episode_rows_before_all_freeze": ledger.episode_rows_before_all_freeze,
            "role_fold_rows_before_all_freeze": (
                ledger.role_fold_rows_before_all_freeze
            ),
            "truth_rows_after_all_freeze": ledger.truth_rows_after_all_freeze,
            "episode_rows_after_all_freeze": ledger.episode_rows_after_all_freeze,
            "role_fold_rows_after_all_freeze": (
                ledger.role_fold_rows_after_all_freeze
            ),
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    eligible_variants = gates[
        "stage1_eligible_variants_for_separate_approval"
    ]
    status = (
        "stage0_mechanism_preflight_pass_some_variants"
        if eligible_variants
        else "stage0_fail_closed_all_variants"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "transition_contract": transition_contract,
        "gates": gates,
        "memoryless_prediction_manifest_sha256": combined_well_sha(
            frozen_wells,
            "memoryless_prediction_sha256",
        ),
        "dz_only_prediction_manifest_sha256": combined_well_sha(
            frozen_wells,
            "dz_only_prediction_sha256",
        ),
        "memoryless_diagnostic_manifest_sha256": combined_well_sha(
            frozen_wells,
            "memoryless_diagnostic_sha256",
        ),
        "dz_only_diagnostic_manifest_sha256": combined_well_sha(
            frozen_wells,
            "dz_only_diagnostic_sha256",
        ),
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
            "cpu_only": True,
            "numba_threads": int(get_nested(config, "runtime.numba_threads_per_worker")),
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "rate_readouts": rate_artifact,
            "episode_truth_late_readout": episode_artifact,
            "well_metrics": well_artifact,
            "input_manifest": input_artifact,
        },
        "stage_1": {
            "implemented": False,
            "execution_authorized": False,
            "eligible_variants_for_separate_approval": eligible_variants,
            "requires_variant_stage0_pass_and_separate_user_approval": True,
        },
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "stage": "stage_0_fixed32",
            "cv": None,
            "lb": None,
            "fixed32_is_mechanism_only": True,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "technical_gates": gates["technical"],
        "variant_results": gates["variants"],
        "stage0_all_variants_pass": gates["stage0_all_variants_pass"],
        "stage1_eligible_variants_for_separate_approval": eligible_variants,
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 9. Configuration preview and guarded execution
#
# The notebook always prints the 2 treatments × 32 wells / zero-parent-rerun /
# zero-model cost contract. Stage 0 is enabled only when both the design and
# execution guards are true. Stage 1, inference, and submission remain
# fail-closed.

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    EXECUTION_COUNTS = validate_execution_contract(
        CONFIG,
        require_run_authorization=False,
    )
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "event": "exp435_stage0_preview",
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "selected_stage": get_nested(CONFIG, "execution.selected_stage"),
                "execution_counts": EXECUTION_COUNTS,
                "implementation_authorized": get_nested(
                    CONFIG,
                    "design.implementation_authorized",
                ),
                "kaggle_stage_0_authorized": get_nested(
                    CONFIG,
                    "design.kaggle_stage_0_authorized",
                ),
                "run_hmm": get_nested(CONFIG, "execution.run_hmm"),
                "stage_1": False,
                "inference": False,
                "submission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if bool(get_nested(CONFIG, "execution.run_hmm", False)):
        SUMMARY = run_stage0(CONFIG)
    else:
        SUMMARY = {
            "experiment": EXPERIMENT_NAME,
            "status": "implementation_complete_stage0_execution_locked",
            "execution_counts": EXECUTION_COUNTS,
            "stage_1": False,
            "inference": False,
            "submission": False,
        }
        print(json.dumps(SUMMARY, sort_keys=True), flush=True)
