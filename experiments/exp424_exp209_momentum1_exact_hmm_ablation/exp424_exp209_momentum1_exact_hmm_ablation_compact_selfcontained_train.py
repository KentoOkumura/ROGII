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
# # exp424 exp209 momentum=1 exact-HMM ablation — Stage 0
#
# This CPU-only notebook implements the preregistered one-factor ablation of
# exp209. The parent pass uses `mom=0.998`; the treatment pass uses `mom=1.0`.
# The rate diffusion, state support, position transition, GR emission, priors,
# forward-backward smoother, and posterior-mean position readout are identical.
# The fixed32 sample is mechanism-only and is not CV or promotion evidence.
# Stage 0 Version 1 completed on 2026-07-28 and failed the preregistered
# mechanism gates. The repository config disables rerunning this notebook;
# Stage 1, inference, and submission remain closed.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 manifest, saved parent, and target-free raw inputs
# 4. Exact exp209 HMM input preparation
# 5. Exact forward-backward kernel with rate-message readout
# 6. Parent/treatment decoding and target-free freeze
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

EXPERIMENT_NAME = "exp424_exp209_momentum1_exact_hmm_ablation"
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

EXPECTED_PARENT_HMM = {
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
}
EXPECTED_TREATMENT_HMM = {**EXPECTED_PARENT_HMM, "mom": 1.0}


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def changed_leaf_paths(
    parent: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> list[str]:
    keys = sorted(set(parent) | set(treatment))
    return [key for key in keys if parent.get(key) != treatment.get(key)]


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> dict[str, int]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp424 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp424 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp424 scientific parent changed")
    if get_nested(config, "lineage.evidence_parent") != EVIDENCE_EXPERIMENT:
        raise ValueError("exp424 evidence parent changed")
    if not bool(get_nested(config, "design.implementation_authorized", False)):
        raise RuntimeError("exp424 implementation is not authorized")
    if not bool(get_nested(config, "design.canonical_notebook_adoption_authorized", False)):
        raise RuntimeError("canonical exp424 notebook adoption is not authorized")
    if bool(get_nested(config, "design.kaggle_stage_1_authorized", True)):
        raise ValueError("Stage 1 must remain disabled during Stage 0")
    if bool(get_nested(config, "design.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "design.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.enable_gpu", True)):
        raise ValueError("exp424 is CPU-only")
    if bool(get_nested(config, "runtime.enable_internet", True)):
        raise ValueError("exp424 must run with internet disabled")
    if get_nested(config, "execution.selected_stage") != "stage_0_fixed32":
        raise ValueError("selected_stage must remain stage_0_fixed32")

    expected = {
        "active_variants": 1,
        "stage_0_baseline_hmm_well_runs": 32,
        "stage_0_treatment_hmm_well_runs": 32,
        "stage_0_total_hmm_well_runs": 64,
        "parent_control_hmm_reruns_stage_0": 32,
        "planned_stage_1_treatment_hmm_well_runs": 773,
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
                "implementation approval does not authorize exp424 Stage 0 execution"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("execution.run_hmm is false")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError("execution.create_prediction is false")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = get_nested(config, "model.parent_hmm")
    treatment = get_nested(config, "model.treatment_hmm")
    if parent != EXPECTED_PARENT_HMM:
        raise ValueError(f"exp209 HMM contract changed: {parent}")
    if treatment != EXPECTED_TREATMENT_HMM:
        raise ValueError(f"momentum=1 treatment contract changed: {treatment}")
    changed = changed_leaf_paths(parent, treatment)
    if changed != ["mom"]:
        raise ValueError(f"treatment must change only mom, observed {changed}")
    if get_nested(config, "model.only_changed_key") != "model.treatment_hmm.mom":
        raise ValueError("only_changed_key contract changed")
    if get_nested(config, "validation.truth_join") != ("after_prediction_and_rate_readout_freeze"):
        raise ValueError("truth-late contract changed")
    if get_nested(config, "validation.stage_0.mechanism.rate_underresponse_readout") != (
        "smoothed_rate_mean"
    ):
        raise ValueError("under-response readout must remain smoothed_rate_mean")
    if int(get_nested(config, "validation.stage_0.technical.rate_edge_state_cells", -1)) != 1:
        raise ValueError("rate-edge contract must remain one state at each edge")
    return {
        "parent_hmm": parent,
        "treatment_hmm": treatment,
        "changed_leaf_paths": changed,
        "transition_mean_move_cells": "-(1-mom)*source_rate*dMD/rate_step",
        "parent_expected_rate": "0.998*source_rate for dMD=1 ft before edge clipping",
        "treatment_expected_rate": "source_rate for dMD=1 ft before edge clipping",
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
    raise FileNotFoundError("exp424 config.yaml was not found")


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
    truth_rows_after_all_freeze: int = 0
    episode_rows_after_all_freeze: int = 0
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
    frame = pd.read_csv(
        path,
        dtype={"well": str, "matched_persistent_well": str},
    )
    expected_columns = {
        "well",
        "role",
        "fold",
        "prefix_rows",
        "suffix_rows",
        "selection_hash",
    }
    if not expected_columns.issubset(frame.columns):
        raise ValueError("fixed32 manifest schema changed")
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 manifest must contain 32 unique wells")
    if frame["role"].value_counts().to_dict() != {"persistent": 16, "control": 16}:
        raise ValueError("fixed32 role counts changed")
    expected_fold_counts = {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}
    if frame.groupby("fold").size().to_dict() != expected_fold_counts:
        raise ValueError("fixed32 fold counts changed")
    if set(frame.loc[frame["role"].eq("persistent"), "fold"].astype(int)) != set(range(5)):
        raise ValueError("persistent fixed32 wells must cover all five folds")
    ledger.record_scope(len(frame))
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "logical_sha256": logical_frame_sha256(frame),
        "mechanism_only_not_cv_or_promotion": True,
    }


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
# ## 5. Exact forward-backward kernel with rate-message readout
#
# For source rate `r`, row spacing `dMD`, and rate step `h`, exp209 uses
# `mean_move_cells = -(1-mom) * r * dMD / h`. The treatment changes only
# `mom=0.998` to `mom=1.0`; therefore the zero-directed mean move becomes zero.
# The variance term and the adjacent three-state support are unchanged.


# %%
@njit(cache=True, nogil=True)
def rate_kernel_probabilities(
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    mom: float,
) -> np.ndarray:
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    sig_rate_step = sig_r * np.sqrt(dm)
    rate_var_cells = (sig_rate_step / rate_step) ** 2
    kernel = np.empty((r_count, 3), np.float64)
    for r_i in range(r_count):
        mean_rate_move = -(1.0 - mom) * rates[r_i] * dm / rate_step
        p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1.0e-12)
        p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1.0e-12)
        total = p_plus + p_minus
        if total > 0.9:
            p_plus *= 0.9 / total
            p_minus *= 0.9 / total
        kernel[r_i, 0] = p_minus
        kernel[r_i, 1] = 1.0 - p_plus - p_minus
        kernel[r_i, 2] = p_plus
    return kernel


def rate_kernel_expected_destination(
    rates: np.ndarray,
    *,
    source_index: int,
    dm: float,
    sig_r: float,
    mom: float,
) -> float:
    rates = np.asarray(rates, dtype=np.float64)
    source_index = int(source_index)
    if not 0 < source_index < len(rates) - 1:
        raise ValueError("expectation helper requires an interior source state")
    kernel = rate_kernel_probabilities(rates, float(dm), float(sig_r), float(mom))
    return float(
        kernel[source_index, 0] * rates[source_index - 1]
        + kernel[source_index, 1] * rates[source_index]
        + kernel[source_index, 2] * rates[source_index + 1]
    )


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_rate_moments(
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
    t_count, p_count = em.shape
    r_count = len(rates)
    neg = np.float32(-1.0e18)

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
    predictive = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)
    predictive_rate_mean = np.empty(t_count, np.float64)
    filtered_rate_mean = np.empty(t_count, np.float64)
    filtered_rate_second = np.empty(t_count, np.float64)
    filtered_rate_edge_mass = np.empty(t_count, np.float64)
    maximum_forward_normalization_error = 0.0

    for t_i in range(t_count):
        kernel = rate_kernel_probabilities(
            rates,
            dm[t_i],
            sig_r,
            mom,
        )
        rate_log_kernel = np.log(kernel)
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
                        total += np.exp(prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best)
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
                    pre_emission = best + np.log(total)
                    predictive[p2, r2] = np.float32(pre_emission)
                    cur[p2, r2] = np.float32(pre_emission + lam * em[t_i, p2])
                else:
                    predictive[p2, r2] = neg
                    cur[p2, r2] = neg

        predictive_best = neg
        filtered_best = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_best = max(predictive_best, predictive[p_i, r_i])
                filtered_best = max(filtered_best, cur[p_i, r_i])
        predictive_total = 0.0
        filtered_total = 0.0
        predictive_r1 = 0.0
        filtered_r1 = 0.0
        filtered_r2 = 0.0
        filtered_edge = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_probability = np.exp(predictive[p_i, r_i] - predictive_best)
                filtered_probability = np.exp(cur[p_i, r_i] - filtered_best)
                predictive_total += predictive_probability
                filtered_total += filtered_probability
                predictive_r1 += predictive_probability * rates[r_i]
                filtered_r1 += filtered_probability * rates[r_i]
                filtered_r2 += filtered_probability * rates[r_i] * rates[r_i]
                if r_i == 0 or r_i == r_count - 1:
                    filtered_edge += filtered_probability
        predictive_rate_mean[t_i] = predictive_r1 / predictive_total
        filtered_rate_mean[t_i] = filtered_r1 / filtered_total
        filtered_rate_second[t_i] = filtered_r2 / filtered_total
        filtered_rate_edge_mass[t_i] = filtered_edge / filtered_total
        predictive_check = 0.0
        filtered_check = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_check += (
                    np.exp(predictive[p_i, r_i] - predictive_best) / predictive_total
                )
                filtered_check += np.exp(cur[p_i, r_i] - filtered_best) / filtered_total
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]
        maximum_forward_normalization_error = max(
            maximum_forward_normalization_error,
            abs(predictive_check - 1.0),
            abs(filtered_check - 1.0),
        )

    best = np.float32(neg)
    for p_i in range(p_count):
        for r_i in range(r_count):
            best = max(best, alpha[t_count - 1, p_i, r_i])
    total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            total += np.exp(alpha[t_count - 1, p_i, r_i] - best)
    log_likelihood = float(best) + np.log(total)

    post_p = np.zeros((t_count, p_count), np.float64)
    beta_next = np.zeros((p_count, r_count), np.float32)
    values = alpha[t_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for p_i in range(p_count):
        acc = 0.0
        for r_i in range(r_count):
            acc += np.exp(values[p_i, r_i] - best)
        post_p[t_count - 1, p_i] = acc
        total += acc
    post_p[t_count - 1] /= total
    for p_i in range(p_count):
        for r_i in range(r_count):
            alpha[t_count - 1, p_i, r_i] = np.float32(np.exp(values[p_i, r_i] - best) / total)

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        kernel = rate_kernel_probabilities(
            rates,
            dm[t_i],
            sig_r,
            mom,
        )
        rate_log_kernel = np.log(kernel)
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
                    if 0 <= p2 < p_count:
                        value = position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2]
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if 0 <= p2 < p_count:
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
                            rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg

        values = alpha[t_i - 1] + beta_cur
        best = np.max(values)
        total = 0.0
        for p_i in range(p_count):
            acc = 0.0
            for r_i in range(r_count):
                acc += np.exp(values[p_i, r_i] - best)
            post_p[t_i - 1, p_i] = acc
            total += acc
        post_p[t_i - 1] /= total
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i - 1, p_i, r_i] = np.float32(np.exp(values[p_i, r_i] - best) / total)
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]

    maximum_posterior_normalization_error = 0.0
    smoothed_rate_mean = np.zeros(t_count, np.float64)
    smoothed_rate_second = np.zeros(t_count, np.float64)
    smoothed_rate_edge_mass = np.zeros(t_count, np.float64)
    for t_i in range(t_count):
        row_total = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                probability = float(alpha[t_i, p_i, r_i])
                row_total += probability
                smoothed_rate_mean[t_i] += probability * rates[r_i]
                smoothed_rate_second[t_i] += probability * rates[r_i] * rates[r_i]
                if r_i == 0 or r_i == r_count - 1:
                    smoothed_rate_edge_mass[t_i] += probability
        maximum_posterior_normalization_error = max(
            maximum_posterior_normalization_error, abs(row_total - 1.0)
        )
        if row_total > 0.0:
            smoothed_rate_mean[t_i] /= row_total
            smoothed_rate_second[t_i] /= row_total
            smoothed_rate_edge_mass[t_i] /= row_total

    filtered_rate_variance = np.maximum(
        filtered_rate_second - filtered_rate_mean * filtered_rate_mean,
        0.0,
    )
    smoothed_rate_variance = np.maximum(
        smoothed_rate_second - smoothed_rate_mean * smoothed_rate_mean,
        0.0,
    )
    return (
        post_p,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        np.sqrt(filtered_rate_variance),
        smoothed_rate_mean,
        np.sqrt(smoothed_rate_variance),
        filtered_rate_edge_mass,
        smoothed_rate_edge_mass,
        maximum_forward_normalization_error,
        maximum_posterior_normalization_error,
    )


def run_hmm_variant(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    result = _hmm2_rate_moments(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
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
        float(hmm["mom"]),
    )
    (
        post_p,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        filtered_rate_std,
        smoothed_rate_mean,
        smoothed_rate_std,
        filtered_rate_edge_mass,
        smoothed_rate_edge_mass,
        forward_normalization_error,
        posterior_normalization_error,
    ) = result
    posterior_mean = np.asarray(post_p, dtype=np.float64) @ np.asarray(
        prepared["grid"], dtype=np.float64
    )
    prediction_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        posterior_mean=np.asarray(posterior_mean, dtype=np.float32),
    )
    rate_readout_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        predictive_rate_mean=np.asarray(predictive_rate_mean, dtype=np.float64),
        filtered_rate_mean=np.asarray(filtered_rate_mean, dtype=np.float64),
        filtered_rate_std=np.asarray(filtered_rate_std, dtype=np.float64),
        smoothed_rate_mean=np.asarray(smoothed_rate_mean, dtype=np.float64),
        smoothed_rate_std=np.asarray(smoothed_rate_std, dtype=np.float64),
        filtered_rate_edge_mass=np.asarray(filtered_rate_edge_mass, dtype=np.float64),
        smoothed_rate_edge_mass=np.asarray(smoothed_rate_edge_mass, dtype=np.float64),
    )
    return {
        "posterior_mean": posterior_mean,
        "log_likelihood": float(log_likelihood),
        "predictive_rate_mean": np.asarray(predictive_rate_mean, dtype=np.float64),
        "filtered_rate_mean": np.asarray(filtered_rate_mean, dtype=np.float64),
        "filtered_rate_std": np.asarray(filtered_rate_std, dtype=np.float64),
        "smoothed_rate_mean": np.asarray(smoothed_rate_mean, dtype=np.float64),
        "smoothed_rate_std": np.asarray(smoothed_rate_std, dtype=np.float64),
        "filtered_rate_edge_mass": np.asarray(filtered_rate_edge_mass, dtype=np.float64),
        "smoothed_rate_edge_mass": np.asarray(smoothed_rate_edge_mass, dtype=np.float64),
        "maximum_normalization_error": max(
            float(forward_normalization_error),
            float(posterior_normalization_error),
        ),
        "prediction_sha256": prediction_sha,
        "rate_readout_sha256": rate_readout_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 6. Parent/treatment decoding and target-free freeze
#
# Each fixed32 well is prepared once, decoded with parent momentum, checked
# against the saved exp209 float32 prediction, and only then decoded with
# treatment momentum. Role/fold, suffix truth, and episode ranges are not
# accepted by this function. All 32 wells are frozen before any truth is read.


# %%
def synthetic_transition_contract(
    parent_hmm: Mapping[str, Any],
    treatment_hmm: Mapping[str, Any],
) -> dict[str, Any]:
    rates = np.linspace(-0.10, 0.10, int(parent_hmm["n_rates"]), dtype=np.float64)
    source_index = 28
    source_rate = float(rates[source_index])
    parent_expected = rate_kernel_expected_destination(
        rates,
        source_index=source_index,
        dm=1.0,
        sig_r=float(parent_hmm["sig_r"]),
        mom=float(parent_hmm["mom"]),
    )
    treatment_expected = rate_kernel_expected_destination(
        rates,
        source_index=source_index,
        dm=1.0,
        sig_r=float(treatment_hmm["sig_r"]),
        mom=float(treatment_hmm["mom"]),
    )
    parent_target = float(parent_hmm["mom"]) * source_rate
    return {
        "source_rate": source_rate,
        "parent_expected_destination": parent_expected,
        "parent_formula_destination": parent_target,
        "treatment_expected_destination": treatment_expected,
        "parent_zero_direction_drift": parent_expected - source_rate,
        "treatment_zero_direction_drift": treatment_expected - source_rate,
        "pass": bool(
            abs(parent_expected - parent_target) <= 1.0e-12
            and abs(treatment_expected - source_rate) <= 1.0e-12
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
    baseline_prediction: np.ndarray
    treatment_prediction: np.ndarray
    baseline_predictive_rate_mean: np.ndarray
    baseline_filtered_rate_mean: np.ndarray
    baseline_filtered_rate_std: np.ndarray
    baseline_smoothed_rate_mean: np.ndarray
    baseline_smoothed_rate_std: np.ndarray
    baseline_filtered_rate_edge_mass: np.ndarray
    baseline_smoothed_rate_edge_mass: np.ndarray
    treatment_predictive_rate_mean: np.ndarray
    treatment_filtered_rate_mean: np.ndarray
    treatment_filtered_rate_std: np.ndarray
    treatment_smoothed_rate_mean: np.ndarray
    treatment_smoothed_rate_std: np.ndarray
    treatment_filtered_rate_edge_mass: np.ndarray
    treatment_smoothed_rate_edge_mass: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    baseline_prediction_sha256: str
    treatment_prediction_sha256: str
    baseline_rate_readout_sha256: str
    treatment_rate_readout_sha256: str
    baseline_saved_parent_max_abs_diff_ft: float
    maximum_normalization_error: float
    baseline_log_likelihood: float
    treatment_log_likelihood: float
    baseline_elapsed_seconds: float
    treatment_elapsed_seconds: float
    prefix_rows: int


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    saved_parent: pd.DataFrame,
    parent_hmm: Mapping[str, Any],
    treatment_hmm: Mapping[str, Any],
    parent_parity_tolerance_ft: float,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, parent_hmm)
    baseline = run_hmm_variant(prepared, parent_hmm)
    parent = saved_parent.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    eval_id = parent_cache_ids_for_rows(well, row_idx)
    if not np.array_equal(parent["row_idx"].to_numpy(np.int64), row_idx):
        raise ValueError(f"{well}: parent row index does not align with raw suffix")
    if not np.array_equal(parent["id"].astype(str).to_numpy(), eval_id):
        raise ValueError(f"{well}: parent id does not align with raw suffix")
    parent_prediction = parent["parent_prediction"].to_numpy(np.float64)
    baseline_prediction = np.asarray(baseline["posterior_mean"], dtype=np.float64)
    parity = saved_float32_parity_max_abs_diff(
        baseline_prediction,
        parent_prediction,
    )
    if parity > float(parent_parity_tolerance_ft):
        raise RuntimeError(f"{well}: baseline exp209 parity failed before treatment: {parity}")
    treatment = run_hmm_variant(prepared, treatment_hmm)
    ledger.freeze(well)
    return FrozenWell(
        well=str(well),
        role="",
        fold=-1,
        eval_id=eval_id,
        row_idx=row_idx,
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        parent_prediction=parent_prediction,
        baseline_prediction=baseline_prediction,
        treatment_prediction=np.asarray(treatment["posterior_mean"], dtype=np.float64),
        baseline_predictive_rate_mean=np.asarray(
            baseline["predictive_rate_mean"], dtype=np.float64
        ),
        baseline_filtered_rate_mean=np.asarray(baseline["filtered_rate_mean"], dtype=np.float64),
        baseline_filtered_rate_std=np.asarray(baseline["filtered_rate_std"], dtype=np.float64),
        baseline_smoothed_rate_mean=np.asarray(baseline["smoothed_rate_mean"], dtype=np.float64),
        baseline_smoothed_rate_std=np.asarray(baseline["smoothed_rate_std"], dtype=np.float64),
        baseline_filtered_rate_edge_mass=np.asarray(
            baseline["filtered_rate_edge_mass"], dtype=np.float64
        ),
        baseline_smoothed_rate_edge_mass=np.asarray(
            baseline["smoothed_rate_edge_mass"], dtype=np.float64
        ),
        treatment_predictive_rate_mean=np.asarray(
            treatment["predictive_rate_mean"], dtype=np.float64
        ),
        treatment_filtered_rate_mean=np.asarray(treatment["filtered_rate_mean"], dtype=np.float64),
        treatment_filtered_rate_std=np.asarray(treatment["filtered_rate_std"], dtype=np.float64),
        treatment_smoothed_rate_mean=np.asarray(treatment["smoothed_rate_mean"], dtype=np.float64),
        treatment_smoothed_rate_std=np.asarray(treatment["smoothed_rate_std"], dtype=np.float64),
        treatment_filtered_rate_edge_mass=np.asarray(
            treatment["filtered_rate_edge_mass"], dtype=np.float64
        ),
        treatment_smoothed_rate_edge_mass=np.asarray(
            treatment["smoothed_rate_edge_mass"], dtype=np.float64
        ),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        baseline_prediction_sha256=str(baseline["prediction_sha256"]),
        treatment_prediction_sha256=str(treatment["prediction_sha256"]),
        baseline_rate_readout_sha256=str(baseline["rate_readout_sha256"]),
        treatment_rate_readout_sha256=str(treatment["rate_readout_sha256"]),
        baseline_saved_parent_max_abs_diff_ft=float(parity),
        maximum_normalization_error=max(
            float(baseline["maximum_normalization_error"]),
            float(treatment["maximum_normalization_error"]),
        ),
        baseline_log_likelihood=float(baseline["log_likelihood"]),
        treatment_log_likelihood=float(treatment["log_likelihood"]),
        baseline_elapsed_seconds=float(baseline["elapsed_seconds"]),
        treatment_elapsed_seconds=float(treatment["elapsed_seconds"]),
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
                    "baseline_prediction": item.baseline_prediction,
                    "treatment_prediction": item.treatment_prediction,
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
                    "baseline_predictive_rate_mean": (item.baseline_predictive_rate_mean),
                    "baseline_filtered_rate_mean": item.baseline_filtered_rate_mean,
                    "baseline_filtered_rate_std": item.baseline_filtered_rate_std,
                    "baseline_smoothed_rate_mean": item.baseline_smoothed_rate_mean,
                    "baseline_smoothed_rate_std": item.baseline_smoothed_rate_std,
                    "baseline_filtered_rate_edge_mass": (item.baseline_filtered_rate_edge_mass),
                    "baseline_smoothed_rate_edge_mass": (item.baseline_smoothed_rate_edge_mass),
                    "treatment_predictive_rate_mean": (item.treatment_predictive_rate_mean),
                    "treatment_filtered_rate_mean": item.treatment_filtered_rate_mean,
                    "treatment_filtered_rate_std": item.treatment_filtered_rate_std,
                    "treatment_smoothed_rate_mean": item.treatment_smoothed_rate_mean,
                    "treatment_smoothed_rate_std": item.treatment_smoothed_rate_std,
                    "treatment_filtered_rate_edge_mass": (item.treatment_filtered_rate_edge_mass),
                    "treatment_smoothed_rate_edge_mass": (item.treatment_smoothed_rate_edge_mass),
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
) -> dict[str, Any]:
    actual = truth["TVT"].to_numpy(np.float64)
    parent_error = frozen.parent_prediction - actual
    baseline_error = frozen.baseline_prediction - actual
    treatment_error = frozen.treatment_prediction - actual
    true_rate = physical_true_interval_rate(
        truth,
        last_known_tvt=frozen.last_known_tvt,
        last_known_md=frozen.last_known_md,
        last_known_z=frozen.last_known_z,
    )
    baseline_rate = zero_direction_underresponse_stats(
        true_rate,
        frozen.baseline_smoothed_rate_mean,
    )
    treatment_rate = zero_direction_underresponse_stats(
        true_rate,
        frozen.treatment_smoothed_rate_mean,
    )
    parent_rmse = float(np.sqrt(np.mean(parent_error**2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
    treatment_rmse = float(np.sqrt(np.mean(treatment_error**2)))
    return {
        "well": frozen.well,
        "role": frozen.role,
        "fold": frozen.fold,
        "rows": len(actual),
        "parent_rmse_ft": parent_rmse,
        "baseline_rmse_ft": baseline_rmse,
        "treatment_rmse_ft": treatment_rmse,
        "rmse_delta_vs_parent_ft": treatment_rmse - parent_rmse,
        "rmse_delta_vs_baseline_ft": treatment_rmse - baseline_rmse,
        "improved_vs_parent": treatment_rmse < parent_rmse,
        "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
        "baseline_underresponse_sse_share": baseline_rate["underresponse_sse_share"],
        "treatment_underresponse_sse_share": treatment_rate["underresponse_sse_share"],
        "baseline_smoothed_rate_edge_mass_mean": float(
            np.mean(frozen.baseline_smoothed_rate_edge_mass)
        ),
        "treatment_smoothed_rate_edge_mass_mean": float(
            np.mean(frozen.treatment_smoothed_rate_edge_mass)
        ),
        "baseline_saved_parent_max_abs_diff_ft": (frozen.baseline_saved_parent_max_abs_diff_ft),
        "maximum_normalization_error": frozen.maximum_normalization_error,
        "baseline_prediction_sha256": frozen.baseline_prediction_sha256,
        "treatment_prediction_sha256": frozen.treatment_prediction_sha256,
        "baseline_rate_readout_sha256": frozen.baseline_rate_readout_sha256,
        "treatment_rate_readout_sha256": frozen.treatment_rate_readout_sha256,
        "baseline_hmm_seconds": frozen.baseline_elapsed_seconds,
        "treatment_hmm_seconds": frozen.treatment_elapsed_seconds,
    }


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
    return frame.sort_values(["well", "start_suffix_offset"], kind="mergesort").reset_index(
        drop=True
    ), {
        "path": str(path),
        "sha256": observed,
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
        baseline_error = frozen.baseline_prediction[offsets] - actual
        treatment_error = frozen.treatment_prediction[offsets] - actual
        true_rate = physical_true_interval_rate(
            truth,
            last_known_tvt=frozen.last_known_tvt,
            last_known_md=frozen.last_known_md,
            last_known_z=frozen.last_known_z,
        )[offsets]
        baseline_rate = zero_direction_underresponse_stats(
            true_rate,
            frozen.baseline_smoothed_rate_mean[offsets],
        )
        treatment_rate = zero_direction_underresponse_stats(
            true_rate,
            frozen.treatment_smoothed_rate_mean[offsets],
        )
        parent_sse = float(np.sum(parent_error**2))
        baseline_sse = float(np.sum(baseline_error**2))
        treatment_sse = float(np.sum(treatment_error**2))
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": str(episode.well),
                "fold": frozen.fold,
                "rows": len(offsets),
                "start_row_idx": int(episode.start_row_idx),
                "end_row_idx_exclusive": int(episode.end_row_idx_exclusive),
                "parent_sse": parent_sse,
                "baseline_sse": baseline_sse,
                "treatment_sse": treatment_sse,
                "treatment_sse_reduction_vs_parent": (
                    1.0 - treatment_sse / parent_sse if parent_sse > 0.0 else math.nan
                ),
                "baseline_rate_valid_rows": baseline_rate["valid_rows"],
                "baseline_underresponse_rows": baseline_rate["underresponse_rows"],
                "baseline_rate_error_sse": baseline_rate["rate_error_sse"],
                "baseline_underresponse_sse": baseline_rate["underresponse_sse"],
                "baseline_underresponse_sse_share": baseline_rate["underresponse_sse_share"],
                "treatment_rate_valid_rows": treatment_rate["valid_rows"],
                "treatment_underresponse_rows": treatment_rate["underresponse_rows"],
                "treatment_rate_error_sse": treatment_rate["rate_error_sse"],
                "treatment_underresponse_sse": treatment_rate["underresponse_sse"],
                "treatment_underresponse_sse_share": treatment_rate["underresponse_sse_share"],
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "well", "start_row_idx"], kind="mergesort")


# %% [markdown]
# ## 8. Stage 0 gates, generated artifacts, and metrics
#
# Every gate is an AND gate. A failure closes the branch; this notebook does not
# search momentum values, alter `sig_r`, relax the fixed32 sample, or blend a
# rescue candidate. Passing Stage 0 only permits a separate Stage 1 approval
# discussion and is not CV evidence.


# %%
def fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def finite_readout_counts(
    frozen_wells: Sequence[FrozenWell],
) -> tuple[int, int, int]:
    finite = 0
    total = 0
    nonfinite_rate = 0
    prediction_attributes = (
        "baseline_prediction",
        "treatment_prediction",
    )
    rate_attributes = (
        "baseline_predictive_rate_mean",
        "baseline_filtered_rate_mean",
        "baseline_filtered_rate_std",
        "baseline_smoothed_rate_mean",
        "baseline_smoothed_rate_std",
        "baseline_filtered_rate_edge_mass",
        "baseline_smoothed_rate_edge_mass",
        "treatment_predictive_rate_mean",
        "treatment_filtered_rate_mean",
        "treatment_filtered_rate_std",
        "treatment_smoothed_rate_mean",
        "treatment_smoothed_rate_std",
        "treatment_filtered_rate_edge_mass",
        "treatment_smoothed_rate_edge_mass",
    )
    for item in frozen_wells:
        for attribute in prediction_attributes:
            array = np.asarray(getattr(item, attribute), dtype=np.float64)
            finite += int(np.isfinite(array).sum())
            total += int(array.size)
        for attribute in rate_attributes:
            array = np.asarray(getattr(item, attribute), dtype=np.float64)
            count = int(np.isfinite(array).sum())
            finite += count
            total += int(array.size)
            nonfinite_rate += int(array.size - count)
    return finite, total, nonfinite_rate


def pooled_rmse_from_well_rows(
    frame: pd.DataFrame,
    column: str,
) -> float:
    weights = frame["rows"].to_numpy(np.float64)
    values = frame[column].to_numpy(np.float64)
    return float(np.sqrt(np.average(values**2, weights=weights)))


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
    maximum_parent_parity = max(item.baseline_saved_parent_max_abs_diff_ft for item in frozen_wells)
    maximum_normalization_error = max(item.maximum_normalization_error for item in frozen_wells)
    finite_values, total_values, nonfinite_rate_moments = finite_readout_counts(frozen_wells)
    finite_coverage = fraction(finite_values, total_values)
    treatment_seconds = float(sum(item.treatment_elapsed_seconds for item in frozen_wells))
    runtime_projection = treatment_seconds * 773.0 / 32.0

    baseline_edge_sum = float(
        sum(np.sum(item.baseline_smoothed_rate_edge_mass) for item in frozen_wells)
    )
    treatment_edge_sum = float(
        sum(np.sum(item.treatment_smoothed_rate_edge_mass) for item in frozen_wells)
    )
    baseline_edge_mean = baseline_edge_sum / total_rows
    treatment_edge_mean = treatment_edge_sum / total_rows
    edge_delta = treatment_edge_mean - baseline_edge_mean

    parent_episode_sse = float(episode_readout["parent_sse"].sum())
    baseline_episode_sse = float(episode_readout["baseline_sse"].sum())
    treatment_episode_sse = float(episode_readout["treatment_sse"].sum())
    persistent_episode_sse_reduction = (
        1.0 - treatment_episode_sse / parent_episode_sse if parent_episode_sse > 0.0 else math.nan
    )
    baseline_rate_error_sse = float(episode_readout["baseline_rate_error_sse"].sum())
    baseline_under_sse = float(episode_readout["baseline_underresponse_sse"].sum())
    treatment_rate_error_sse = float(episode_readout["treatment_rate_error_sse"].sum())
    treatment_under_sse = float(episode_readout["treatment_underresponse_sse"].sum())
    baseline_under_share = fraction(baseline_under_sse, baseline_rate_error_sse)
    treatment_under_share = fraction(
        treatment_under_sse,
        treatment_rate_error_sse,
    )
    under_share_reduction = baseline_under_share - treatment_under_share

    persistent_wells = well_metrics.loc[well_metrics["role"].eq("persistent")]
    control_wells = well_metrics.loc[well_metrics["role"].eq("control")]
    persistent_improved_wells = int(persistent_wells["improved_vs_parent"].astype(bool).sum())
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_episodes = episode_readout.loc[episode_readout["fold"].eq(fold)]
        fold_parent_sse = float(fold_episodes["parent_sse"].sum())
        fold_treatment_sse = float(fold_episodes["treatment_sse"].sum())
        fold_rows.append(
            {
                "fold": fold,
                "episodes": len(fold_episodes),
                "parent_sse": fold_parent_sse,
                "treatment_sse": fold_treatment_sse,
                "improved": bool(len(fold_episodes) > 0 and fold_treatment_sse < fold_parent_sse),
            }
        )
    persistent_sse_improving_folds = int(sum(row["improved"] for row in fold_rows))

    control_parent_rmse = pooled_rmse_from_well_rows(
        control_wells,
        "parent_rmse_ft",
    )
    control_treatment_rmse = pooled_rmse_from_well_rows(
        control_wells,
        "treatment_rmse_ft",
    )
    control_rmse_delta = control_treatment_rmse - control_parent_rmse
    control_delta_p95 = float(
        np.quantile(
            control_wells["rmse_delta_vs_parent_ft"].to_numpy(np.float64),
            0.95,
        )
    )

    technical = {
        "fixed32_roles_and_unique_wells": bool(
            len(manifest) == 32
            and manifest["well"].nunique() == 32
            and manifest["role"].value_counts().to_dict() == {"persistent": 16, "control": 16}
        ),
        "fixed32_fold_counts": bool(
            manifest.groupby("fold").size().to_dict() == expected_fold_counts
        ),
        "transition_mean_contract": bool(transition_contract["pass"]),
        "truth_reads_before_all_freeze": bool(
            ledger.truth_rows_before_all_freeze
            == int(technical_config["truth_reads_before_freeze"])
        ),
        "episode_reads_before_all_freeze": bool(
            ledger.episode_rows_before_all_freeze
            == int(technical_config["episode_reads_before_freeze"])
        ),
        "baseline_saved_exp209_parity": bool(
            maximum_parent_parity <= float(technical_config["untreated_parent_parity_max_abs_ft"])
        ),
        "posterior_normalization": bool(
            maximum_normalization_error <= float(technical_config["normalization_max_abs_error"])
        ),
        "finite_prediction_and_rate_coverage": bool(
            finite_coverage >= float(technical_config["finite_coverage"])
        ),
        "nonfinite_rate_moments": nonfinite_rate_moments == 0,
        "prediction_readback_sha": bool(
            prediction_artifact["logical_sha256"] == prediction_artifact["readback_logical_sha256"]
        ),
        "rate_readout_readback_sha": bool(
            rate_artifact["logical_sha256"] == rate_artifact["readback_logical_sha256"]
        ),
        "runtime_projection": bool(
            runtime_projection <= float(technical_config["full_runtime_projection_max_seconds"])
        ),
        "peak_rss": bool(peak_rss_gb() <= float(technical_config["peak_rss_max_gb"])),
    }
    mechanism = {
        "persistent_episode_sse_reduction": bool(
            math.isfinite(persistent_episode_sse_reduction)
            and persistent_episode_sse_reduction
            >= float(mechanism_config["persistent_episode_sse_reduction_min"])
        ),
        "persistent_improved_wells": bool(
            persistent_improved_wells >= int(mechanism_config["persistent_improved_wells_min"])
        ),
        "underresponse_sse_share_reduction": bool(
            math.isfinite(under_share_reduction)
            and under_share_reduction
            >= float(mechanism_config["underresponse_sse_share_reduction_min_points"])
        ),
        "persistent_sse_improving_folds": bool(
            persistent_sse_improving_folds
            >= int(mechanism_config["persistent_sse_improving_folds_min"])
        ),
        "matched_control_rmse_safety": bool(
            control_rmse_delta <= float(mechanism_config["control_rmse_delta_max_ft"])
        ),
        "matched_control_by_well_p95_safety": bool(
            control_delta_p95 <= float(mechanism_config["control_by_well_delta_p95_max_ft"])
        ),
        "rate_edge_mass_nonworse": bool(
            edge_delta <= float(technical_config["rate_edge_mass_nonworse_tolerance"])
        ),
    }
    diagnostics = {
        "total_wells": len(frozen_wells),
        "total_suffix_rows": total_rows,
        "maximum_baseline_saved_parent_abs_diff_ft": maximum_parent_parity,
        "maximum_normalization_error": maximum_normalization_error,
        "finite_coverage": finite_coverage,
        "nonfinite_rate_moments": nonfinite_rate_moments,
        "parent_episode_sse": parent_episode_sse,
        "baseline_episode_sse": baseline_episode_sse,
        "treatment_episode_sse": treatment_episode_sse,
        "persistent_episode_sse_reduction": persistent_episode_sse_reduction,
        "persistent_improved_wells": persistent_improved_wells,
        "persistent_sse_by_fold": fold_rows,
        "persistent_sse_improving_folds": persistent_sse_improving_folds,
        "baseline_underresponse_sse_share": baseline_under_share,
        "treatment_underresponse_sse_share": treatment_under_share,
        "underresponse_sse_share_reduction_points": under_share_reduction,
        "control_parent_rmse_ft": control_parent_rmse,
        "control_treatment_rmse_ft": control_treatment_rmse,
        "control_rmse_delta_ft": control_rmse_delta,
        "control_by_well_rmse_delta_p95_ft": control_delta_p95,
        "baseline_smoothed_rate_edge_mass_mean": baseline_edge_mean,
        "treatment_smoothed_rate_edge_mass_mean": treatment_edge_mean,
        "smoothed_rate_edge_mass_delta": edge_delta,
        "stage0_elapsed_seconds": float(elapsed_seconds),
        "stage1_treatment_runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
        "fixed32_is_mechanism_only_not_cv_or_promotion": True,
    }
    stage0_pass = bool(all(technical.values()) and all(mechanism.values()))
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "stage0_all_gates_pass": stage0_pass,
        "stage1_eligible_for_separate_approval": stage0_pass,
        "fixed32_is_cv": False,
        "fixed32_is_promotion_evidence": False,
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP424_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp424 Stage 0 must run on Kaggle CPU; local execution is disabled")


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
    parent_hmm = get_nested(config, "model.parent_hmm")
    treatment_hmm = get_nested(config, "model.treatment_hmm")
    transition_contract = synthetic_transition_contract(
        parent_hmm,
        treatment_hmm,
    )
    if not transition_contract["pass"]:
        raise RuntimeError(f"transition mean contract failed: {transition_contract}")
    raw_dir = train_data_dir(config)
    parent_groups = parent.groupby("well", sort=False).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    parity_tolerance = float(
        get_nested(
            config,
            "validation.stage_0.technical.untreated_parent_parity_max_abs_ft",
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
            parent_hmm=parent_hmm,
            treatment_hmm=treatment_hmm,
            parent_parity_tolerance_ft=parity_tolerance,
            ledger=ledger,
        )
        frozen = attach_scope_identity(
            frozen,
            pd.Series({"role": row.role, "fold": row.fold}),
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
                    "event": "exp424_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "baseline_hmm_seconds": frozen.baseline_elapsed_seconds,
                    "treatment_hmm_seconds": frozen.treatment_elapsed_seconds,
                    "baseline_saved_parent_max_abs_diff_ft": (
                        frozen.baseline_saved_parent_max_abs_diff_ft
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
        well_metric_rows.append(well_truth_late_metrics(item, truth))
    selected_persistent = set(manifest.loc[manifest["role"].eq("persistent"), "well"].astype(str))
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
        ["fold", "role", "well"], kind="mergesort"
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
        manifest=manifest,
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
            "truth_rows_after_all_freeze": ledger.truth_rows_after_all_freeze,
            "episode_rows_after_all_freeze": ledger.episode_rows_after_all_freeze,
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    status = (
        "stage0_mechanism_preflight_pass"
        if gates["stage0_all_gates_pass"]
        else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "transition_contract": transition_contract,
        "gates": gates,
        "baseline_prediction_manifest_sha256": combined_well_sha(
            frozen_wells,
            "baseline_prediction_sha256",
        ),
        "treatment_prediction_manifest_sha256": combined_well_sha(
            frozen_wells,
            "treatment_prediction_sha256",
        ),
        "baseline_rate_readout_manifest_sha256": combined_well_sha(
            frozen_wells,
            "baseline_rate_readout_sha256",
        ),
        "treatment_rate_readout_manifest_sha256": combined_well_sha(
            frozen_wells,
            "treatment_rate_readout_sha256",
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
            "requires_stage0_all_gates_and_separate_user_approval": True,
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
        "mechanism_gates": gates["mechanism"],
        "stage0_all_gates_pass": gates["stage0_all_gates_pass"],
        "stage1_eligible_for_separate_approval": gates["stage1_eligible_for_separate_approval"],
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 9. Configuration preview and guarded execution
#
# The notebook always prints the 32 baseline + 32 treatment / zero-model cost
# contract. The repository config records the completed fail-closed Stage 0
# and disables an accidental rerun; Stage 1, inference, and submission remain
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
                "event": "exp424_stage0_preview",
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
    SUMMARY = run_stage0(CONFIG)
