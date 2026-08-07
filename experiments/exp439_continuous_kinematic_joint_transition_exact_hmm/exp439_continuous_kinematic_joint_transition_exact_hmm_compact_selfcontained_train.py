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
# # exp439 continuous-kinematic joint-transition exact HMM — Stage 0
#
# This CPU-only Stage 0 candidate preserves the exp209 persistent
# `(TVT, U-rate)` state and its legal three-neighbour rate transition. The only
# scientific change is the position displacement on each source/destination
# rate edge:
#
# `delta_TVT = 0.5 * (r_source + r_destination) * delta_MD - delta_Z + eta_p`.
#
# A deterministic 5/7/9-cell maximum-entropy projection preserves probability,
# conditional displacement mean, and conditional variance. Forward and
# backward messages consume the same precomputed joint-edge table.
#
# The notebook is fail-closed: implementation approval does not authorize a
# Kaggle run. Stage 1, inference, submission, parent-HMM regeneration, and any
# grid/noise/emission rescue are disabled.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 manifest, saved parent, and target-free raw inputs
# 4. Exact exp209 input preparation
# 5. Moment-preserving correlated joint-edge table
# 6. Exact forward-backward HMM and target-free freeze
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
from numba import njit, set_num_threads

EXPERIMENT_NAME = "exp439_continuous_kinematic_joint_transition_exact_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
EVIDENCE_EXPERIMENT = "exp408_hmm_message_rate_basin_audit"
SCIENTIFIC_VARIANT = "continuous_kinematic_joint_edge"
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
        raise ValueError("wrong exp439 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp439 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp439 scientific parent changed")
    if not bool(get_nested(config, "runtime.implementation_approved", False)):
        raise RuntimeError("exp439 implementation is not approved")
    if bool(get_nested(config, "runtime.stage1_approved", True)):
        raise ValueError("Stage 1 must remain disabled during Stage 0")
    if bool(get_nested(config, "runtime.inference_enabled", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "runtime.submission_enabled", True)):
        raise ValueError("submission must remain disabled")
    if str(get_nested(config, "runtime.accelerator")) != "cpu":
        raise ValueError("exp439 is CPU-only")
    if bool(get_nested(config, "runtime.internet", True)):
        raise ValueError("exp439 internet must remain disabled")

    expected = {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "stage0_hmm_well_runs": 32,
        "stage1_max_hmm_well_runs": 773,
        "parent_control_hmm_reruns": 0,
        "fitted_ml_models": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1)) for key in expected
    }
    if observed != expected:
        raise ValueError(f"Stage 0 execution contract changed: {observed} != {expected}")
    if get_nested(config, "execution.selected_stage") != "stage_0_fixed32":
        raise ValueError("selected_stage must remain stage_0_fixed32")
    if bool(get_nested(config, "validation.parent_rerun", True)):
        raise ValueError("saved exp209 prediction must remain the control")
    if require_run_authorization:
        if not bool(get_nested(config, "runtime.run_approved", False)):
            raise RuntimeError(
                "implementation approval does not authorize Kaggle execution"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("execution.run_hmm is false")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = get_nested(config, "model.fixed_from_exp209")
    expected_fixed = {
        "position_grid_step_ft": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "effective_position_sigma_ft": 0.1225,
        "momentum": 0.998,
        "emission": "gaussian_typewell_gr",
        "emission_lambda": 1.0,
        "start_sigma_ft": 0.75,
        "initial_rate_sigma": 0.01,
        "band_pad_ft": 100.0,
        "rate_center": "zero",
        "rate_boundary_semantics": "preserve_parent_substochastic_outward_mass",
        "position_band_boundary_semantics": "preserve_parent_truncation",
        "output": "smoothed_posterior_mean_and_std",
    }
    if fixed != expected_fixed:
        raise ValueError(f"exp209 HMM contract changed: {fixed} != {expected_fixed}")
    candidate = get_nested(config, "model.candidate_transition")
    expected_candidate = {
        "rate_marginal": "exact_exp209_adjacent_three_state_kernel",
        "conditional_position_mean_formula": (
            "0.5*(r_source+r_destination)*delta_MD-delta_Z"
        ),
        "conditional_position_sigma_formula": (
            "max(sig_p,0.35*position_grid_step)"
        ),
        "eta_p_conditionally_independent_of_rate_edge": True,
        "coupling": "source_and_destination_rate_in_same_joint_edge",
    }
    if candidate != expected_candidate:
        raise ValueError(
            f"continuous-kinematic joint-edge contract changed: "
            f"{candidate} != {expected_candidate}"
        )
    if get_nested(config, "model.state") != ["tvt_position", "u_rate"]:
        raise ValueError("candidate state must remain joint (TVT, U-rate)")
    projection = get_nested(config, "model.lattice_projection")
    if projection["support_cells_order"] != [5, 7, 9]:
        raise ValueError("moment support order must remain 5/7/9")
    if projection["objective"] != "finite_support_maximum_entropy":
        raise ValueError("moment projection objective changed")
    if not bool(projection["fail_if_infeasible"]):
        raise ValueError("infeasible moment projection must fail closed")
    return {
        "fixed_from_exp209": fixed,
        "candidate_transition": candidate,
        "candidate_state": get_nested(config, "model.state"),
        "lattice_projection": projection,
        "forbidden": get_nested(config, "model.forbidden"),
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
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            value = yaml.safe_load(candidate.read_text()) or {}
            if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
                return candidate
    raise FileNotFoundError("exp439 config.yaml was not found")


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
        else:
            normalized[column] = normalized[column].astype(str)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


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


def runtime_versions() -> dict[str, Any]:
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
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


def resolve_unique_file(
    *,
    filename: str,
    candidates: Sequence[str],
    patterns: Sequence[str],
) -> Path:
    root = find_project_root()
    matches: list[Path] = []
    for item in candidates:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            matches.append(candidate)
        elif (candidate / filename).is_file():
            matches.append(candidate / filename)
    if KAGGLE_INPUT_ROOT.is_dir():
        for pattern in patterns:
            matches.extend(KAGGLE_INPUT_ROOT.glob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if not unique:
        raise FileNotFoundError(f"could not resolve {filename}")
    if len(unique) > 1:
        hashes = {sha256_file(path) for path in unique}
        if len(hashes) != 1:
            raise RuntimeError(f"multiple non-identical files found for {filename}")
    return unique[0]


@dataclass
class LeakageLedger:
    frozen_wells: set[str] = field(default_factory=set)
    scope_rows: int = 0
    target_free_rows: int = 0
    truth_rows_before_all_freeze: int = 0
    role_fold_rows_before_all_freeze: int = 0
    episode_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    role_fold_rows_after_all_freeze: int = 0
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

    def record_role_fold_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.role_fold_rows_before_all_freeze += int(rows)
            raise RuntimeError("role/fold was read before all fixed32 predictions were frozen")
        self.role_fold_rows_after_all_freeze += int(rows)

    def record_episode_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.episode_rows_before_all_freeze += int(rows)
            raise RuntimeError("episodes were read before all fixed32 predictions were frozen")
        self.episode_rows_after_all_freeze += int(rows)


# %% [markdown]
# ## 3. Fixed32 manifest, saved parent, and target-free raw inputs
#
# Before all 32 candidate predictions and diagnostic SHAs are frozen, the
# manifest reader opens only `well`, `prefix_rows`, and `suffix_rows`. Role and
# fold are reread through a separately guarded function after freeze.

# %%
def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.is_dir():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT
            / "competitions"
            / "rogii-wellbore-geology-prediction"
            / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    return find_project_root() / str(get_nested(config, "data.train_dir"))


def fixed32_manifest_path(config: Mapping[str, Any]) -> tuple[Path, str]:
    spec = get_nested(config, "data.fixed32_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed: {observed}")
    return path, observed


def load_fixed32_target_free_scope(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, observed = fixed32_manifest_path(config)
    frame = pd.read_csv(
        path,
        usecols=["well", "prefix_rows", "suffix_rows"],
        dtype={"well": str},
    )
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 manifest must contain 32 unique wells")
    ledger.record_scope(len(frame))
    frame = frame.sort_values("well", kind="mergesort").reset_index(drop=True)
    return frame, {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "target_free_logical_sha256": logical_frame_sha256(frame),
    }


def load_fixed32_identity_after_all_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    path, _ = fixed32_manifest_path(config)
    frame = pd.read_csv(path, dtype={"well": str, "matched_persistent_well": str})
    ledger.record_role_fold_late(len(frame))
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 identity changed")
    if frame["role"].value_counts().to_dict() != {"persistent": 16, "control": 16}:
        raise ValueError("fixed32 role counts changed")
    if set(frame["fold"].astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("fixed32 fold coverage changed")
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


def parent_row_indices_from_cache_ids(frame: pd.DataFrame) -> np.ndarray:
    row_indices = np.empty(len(frame), dtype=np.int64)
    for offset, (well, identifier) in enumerate(
        zip(frame["well"].astype(str), frame["id"].astype(str), strict=True)
    ):
        prefix = f"{well}_"
        if not identifier.startswith(prefix):
            raise ValueError(
                f"saved parent id does not start with exact well prefix: {identifier}"
            )
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


def load_saved_parent_predictions(
    config: Mapping[str, Any],
    target_wells: set[str],
    expected_rows: int,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp209_saved_control")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(value) for value in spec["candidates"]],
        patterns=[str(value) for value in spec["patterns"]],
    )
    decompressed = sha256_decompressed_csv(path)
    if decompressed != str(spec["expected_decompressed_sha256"]):
        raise ValueError(f"saved exp209 decompressed SHA changed: {decompressed}")
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
        raise ValueError("saved exp209 control has no fixed32 rows")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.rename(columns={str(spec["prediction_column"]): "parent_prediction"})
    frame["row_idx"] = parent_row_indices_from_cache_ids(frame)
    frame["parent_prediction"] = pd.to_numeric(
        frame["parent_prediction"], errors="raise"
    )
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if len(frame) != expected_rows:
        raise ValueError(f"saved parent rows={len(frame)}/{expected_rows}")
    if frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("saved parent keys are not unique")
    ledger.record_target_free(len(frame))
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": decompressed,
        "rows": len(frame),
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
# ## 4. Exact exp209 input preparation
#
# Prefix calibration, the fixed TVT grid, GR emission, initial rate, and priors
# are copied from exp209. Unknown-suffix truth is absent from this phase.

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

    step = float(hmm["position_grid_step_ft"])
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    grid_min = max(
        float(typewell_tvt.min()) - 40.0,
        last_tvt - float(hmm["band_pad_ft"]),
    )
    grid_max = min(
        float(typewell_tvt.max()) + 40.0,
        last_tvt + float(hmm["band_pad_ft"]),
    )
    tvt_grid = np.arange(
        grid_min,
        grid_max + step,
        step,
        dtype=np.float64,
    )
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
    dz = np.diff(np.concatenate([[last_z], z]))
    gr_grid = np.interp(tvt_grid, typewell_tvt, typewell_gr)
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    emission_ll_exact = -0.5 * np.minimum(zscore**2, 600.0)
    emission_ll = emission_ll_exact.astype(np.float32)
    span = max(float(hmm["rate_span"]), abs(init_rate) + 0.04)
    rates = np.linspace(-span, span, int(hmm["n_rates"]), dtype=np.float64)
    return {
        "emission_ll": emission_ll,
        "emission_ll_exact": emission_ll_exact,
        "dm": dm,
        "dz": dz,
        "z": z,
        "gr": gr,
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "tvt_grid": tvt_grid,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "r0": float(init_rate),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "last_known_z": last_z,
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "cal_a": cal_a,
        "cal_b": cal_b,
        "robust_sigma_unused": robust_sigma,
    }


def input_contract_from_prepared(
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    tvt_grid = np.asarray(prepared["tvt_grid"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    typewell_tvt = np.asarray(prepared["typewell_tvt"], dtype=np.float64)
    typewell_gr = np.asarray(prepared["typewell_gr"], dtype=np.float64)
    gr = np.asarray(prepared["gr"], dtype=np.float64)
    sigma = float(prepared["prefix_sigma"])
    direct_grid = np.interp(tvt_grid, typewell_tvt, typewell_gr)
    direct_zscore = (gr[:, None] - direct_grid[None, :]) / sigma
    direct_emission = -0.5 * np.minimum(direct_zscore**2, 600.0)
    emission_identity = float(
        np.max(
            np.abs(
                direct_emission
                - np.asarray(prepared["emission_ll_exact"], dtype=np.float64)
            )
        )
    )
    return {
        "emission_identity_max_abs": emission_identity,
        "rows": len(prepared["dm"]),
        "position_states": len(tvt_grid),
        "rate_states": len(rates),
        "sha256": array_bundle_sha256(
            eval_index=np.asarray(prepared["eval_index"], dtype=np.int64),
            tvt_grid=tvt_grid,
            dm=np.asarray(prepared["dm"], dtype=np.float64),
            dz=np.asarray(prepared["dz"], dtype=np.float64),
            rates=rates,
            emission_ll=np.asarray(prepared["emission_ll"], dtype=np.float32),
        ),
    }


# %% [markdown]
# ## 5. Moment-preserving correlated joint-edge implementation
#
# The candidate path precomputes one table indexed by
# `(row, source_rate, rate_delta, position_offset)`. Forward and backward
# messages consume those same arrays.

# %%
@njit(cache=True, nogil=True)
def rate_kernel_probabilities(
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    sigma_rate_step = sig_r * np.sqrt(dm)
    rate_variance_cells = (sigma_rate_step / rate_step) ** 2
    kernel = np.empty((rate_count, 3), np.float64)
    for rate_index in range(rate_count):
        mean_rate_move = (
            -(1.0 - momentum) * rates[rate_index] * dm / rate_step
        )
        p_plus = max(
            0.5 * (rate_variance_cells + mean_rate_move),
            1.0e-12,
        )
        p_minus = max(
            0.5 * (rate_variance_cells - mean_rate_move),
            1.0e-12,
        )
        total = p_plus + p_minus
        if total > 0.9:
            p_plus *= 0.9 / total
            p_minus *= 0.9 / total
        kernel[rate_index, 0] = p_minus
        kernel[rate_index, 1] = 1.0 - p_plus - p_minus
        kernel[rate_index, 2] = p_plus
    return kernel


@njit(cache=True, nogil=True)
def parent_position_kernel_probabilities(
    mean_shift: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_position = max(sig_p, 0.35 * step)
    center = int(np.floor(mean_shift / step + 0.5))
    offsets = np.empty(5, np.int64)
    log_weights = np.empty(5, np.float64)
    for kernel_index in range(5):
        offset = center - 2 + kernel_index
        delta = offset * step - mean_shift
        offsets[kernel_index] = offset
        log_weights[kernel_index] = -0.5 * (
            delta / sigma_position
        ) ** 2
    maximum = np.max(log_weights)
    weights = np.exp(log_weights - maximum)
    weights /= np.sum(weights)
    return offsets, weights


@njit(cache=True, nogil=True)
def _moment_projection_for_support(
    mean_shift: float,
    step: float,
    variance: float,
    support_cells: int,
    maximum_iterations: int,
    moment_tolerance: float,
    feasibility_tolerance: float,
    damping_min: float,
) -> tuple[np.ndarray, np.ndarray, int, bool, int]:
    offsets = np.zeros(9, np.int64)
    weights = np.zeros(9, np.float64)
    center = int(np.floor(mean_shift / step + 0.5))
    half = support_cells // 2
    residual = np.empty(support_cells, np.float64)
    scaled = np.empty(support_cells, np.float64)
    for index in range(support_cells):
        offset = center - half + index
        offsets[index] = offset
        residual[index] = offset * step - mean_shift
        scaled[index] = residual[index] / step

    negative_index = -1
    positive_index = -1
    for index in range(support_cells):
        if residual[index] <= 0.0:
            negative_index = index
        if positive_index < 0 and residual[index] >= 0.0:
            positive_index = index
    if negative_index < 0 or positive_index < 0:
        return offsets, weights, support_cells, False, 0

    if negative_index == positive_index:
        minimum_variance = 0.0
    else:
        minimum_variance = (
            -residual[negative_index] * residual[positive_index]
        )
    minimum_residual = residual[0]
    maximum_residual = residual[support_cells - 1]
    maximum_variance = -minimum_residual * maximum_residual
    if (
        variance < minimum_variance - feasibility_tolerance
        or variance > maximum_variance + feasibility_tolerance
    ):
        return offsets, weights, support_cells, False, 0

    if abs(variance - minimum_variance) <= feasibility_tolerance:
        if negative_index == positive_index:
            weights[negative_index] = 1.0
        else:
            denominator = (
                residual[positive_index] - residual[negative_index]
            )
            weights[negative_index] = (
                residual[positive_index] / denominator
            )
            weights[positive_index] = (
                -residual[negative_index] / denominator
            )
        return offsets, weights, support_cells, True, 0
    if abs(variance - maximum_variance) <= feasibility_tolerance:
        denominator = residual[support_cells - 1] - residual[0]
        weights[0] = residual[support_cells - 1] / denominator
        weights[support_cells - 1] = -residual[0] / denominator
        return offsets, weights, support_cells, True, 0

    target_second = variance / (step * step)
    theta_first = 0.0
    theta_second = -0.5 / max(target_second, 1.0e-12)
    work = np.empty(support_cells, np.float64)
    proposal = np.empty(support_cells, np.float64)
    iterations = 0
    converged = False
    for iteration in range(maximum_iterations):
        iterations = iteration + 1
        maximum_log_weight = -1.0e300
        for index in range(support_cells):
            value = (
                theta_first * scaled[index]
                + theta_second * scaled[index] * scaled[index]
            )
            work[index] = value
            maximum_log_weight = max(maximum_log_weight, value)
        total = 0.0
        for index in range(support_cells):
            work[index] = np.exp(work[index] - maximum_log_weight)
            total += work[index]
        mean_first = 0.0
        mean_second = 0.0
        mean_third = 0.0
        mean_fourth = 0.0
        for index in range(support_cells):
            probability = work[index] / total
            work[index] = probability
            value = scaled[index]
            square = value * value
            mean_first += probability * value
            mean_second += probability * square
            mean_third += probability * square * value
            mean_fourth += probability * square * square
        residual_first = mean_first
        residual_second = mean_second - target_second
        physical_error = max(
            abs(residual_first * step),
            abs(residual_second * step * step),
        )
        if physical_error <= moment_tolerance:
            converged = True
            break

        covariance_11 = mean_second - mean_first * mean_first
        covariance_12 = mean_third - mean_first * mean_second
        covariance_22 = mean_fourth - mean_second * mean_second
        determinant = (
            covariance_11 * covariance_22
            - covariance_12 * covariance_12
        )
        if determinant <= 1.0e-24 or not np.isfinite(determinant):
            break
        delta_first = (
            covariance_22 * residual_first
            - covariance_12 * residual_second
        ) / determinant
        delta_second = (
            -covariance_12 * residual_first
            + covariance_11 * residual_second
        ) / determinant

        old_objective = max(
            abs(residual_first),
            abs(residual_second),
        )
        damping = 1.0
        accepted = False
        while damping >= damping_min:
            candidate_first = theta_first - damping * delta_first
            candidate_second = theta_second - damping * delta_second
            candidate_maximum = -1.0e300
            for index in range(support_cells):
                value = (
                    candidate_first * scaled[index]
                    + candidate_second * scaled[index] * scaled[index]
                )
                proposal[index] = value
                candidate_maximum = max(candidate_maximum, value)
            candidate_total = 0.0
            for index in range(support_cells):
                proposal[index] = np.exp(
                    proposal[index] - candidate_maximum
                )
                candidate_total += proposal[index]
            candidate_mean = 0.0
            candidate_second_moment = 0.0
            for index in range(support_cells):
                probability = proposal[index] / candidate_total
                candidate_mean += probability * scaled[index]
                candidate_second_moment += (
                    probability * scaled[index] * scaled[index]
                )
            candidate_objective = max(
                abs(candidate_mean),
                abs(candidate_second_moment - target_second),
            )
            if candidate_objective < old_objective:
                theta_first = candidate_first
                theta_second = candidate_second
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break

    if not converged:
        return offsets, weights, support_cells, False, iterations
    for index in range(support_cells):
        weights[index] = work[index]
    return offsets, weights, support_cells, True, iterations


@njit(cache=True, nogil=True)
def moment_preserving_projection(
    mean_shift: float,
    step: float,
    variance: float,
    maximum_iterations: int,
    moment_tolerance: float,
    feasibility_tolerance: float,
    damping_min: float,
) -> tuple[np.ndarray, np.ndarray, int, bool, int]:
    for support_cells in (5, 7, 9):
        offsets, weights, count, feasible, iterations = (
            _moment_projection_for_support(
                mean_shift,
                step,
                variance,
                support_cells,
                maximum_iterations,
                moment_tolerance,
                feasibility_tolerance,
                damping_min,
            )
        )
        if feasible:
            return offsets, weights, count, True, iterations
    return (
        np.zeros(9, np.int64),
        np.zeros(9, np.float64),
        0,
        False,
        0,
    )


def projection_moments(
    mean_shift: float,
    step: float,
    offsets: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> tuple[float, float, float]:
    locations = np.asarray(offsets[:count], dtype=np.float64) * step
    probabilities = np.asarray(weights[:count], dtype=np.float64)
    total = float(probabilities.sum())
    mean = float(np.sum(probabilities * locations))
    variance = float(np.sum(probabilities * (locations - mean_shift) ** 2))
    return total, mean, variance


@njit(cache=True, nogil=True)
def precompute_joint_edge_table_numba(
    dm: np.ndarray,
    dz: np.ndarray,
    rates: np.ndarray,
    step: float,
    sig_r: float,
    sig_p: float,
    momentum: float,
    maximum_iterations: int,
    moment_tolerance: float,
    feasibility_tolerance: float,
    damping_min: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    time_count = len(dm)
    rate_count = len(rates)
    offsets = np.zeros((time_count, rate_count, 3, 9), np.int64)
    weights = np.zeros((time_count, rate_count, 3, 9), np.float64)
    counts = np.zeros((time_count, rate_count, 3), np.int8)
    rate_probability = np.zeros(
        (time_count, rate_count, 3),
        np.float64,
    )
    feasible = np.zeros((time_count, rate_count, 3), np.uint8)
    iterations = np.zeros((time_count, rate_count, 3), np.int16)
    position_sigma = max(sig_p, 0.35 * step)
    variance = position_sigma * position_sigma
    for time_index in range(time_count):
        rate_kernel = rate_kernel_probabilities(
            rates,
            dm[time_index],
            sig_r,
            momentum,
        )
        for source_rate in range(rate_count):
            for edge_index in range(3):
                destination_rate = source_rate + edge_index - 1
                rate_probability[
                    time_index,
                    source_rate,
                    edge_index,
                ] = rate_kernel[source_rate, edge_index]
                if destination_rate < 0 or destination_rate >= rate_count:
                    continue
                mean_shift = (
                    0.5
                    * (rates[source_rate] + rates[destination_rate])
                    * dm[time_index]
                    - dz[time_index]
                )
                edge_offsets, edge_weights, count, ok, solver_iterations = (
                    moment_preserving_projection(
                        mean_shift,
                        step,
                        variance,
                        maximum_iterations,
                        moment_tolerance,
                        feasibility_tolerance,
                        damping_min,
                    )
                )
                if ok:
                    feasible[time_index, source_rate, edge_index] = 1
                    counts[time_index, source_rate, edge_index] = count
                    iterations[
                        time_index,
                        source_rate,
                        edge_index,
                    ] = solver_iterations
                    for kernel_index in range(count):
                        offsets[
                            time_index,
                            source_rate,
                            edge_index,
                            kernel_index,
                        ] = edge_offsets[kernel_index]
                        weights[
                            time_index,
                            source_rate,
                            edge_index,
                            kernel_index,
                        ] = edge_weights[kernel_index]
    return (
        offsets,
        weights,
        counts,
        rate_probability,
        feasible,
        iterations,
    )


def build_joint_edge_table(
    *,
    dm: np.ndarray,
    dz: np.ndarray,
    rates: np.ndarray,
    step: float,
    sig_r: float,
    sig_p: float,
    momentum: float,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    solver = projection["solver"]
    arrays = precompute_joint_edge_table_numba(
        np.asarray(dm, dtype=np.float64),
        np.asarray(dz, dtype=np.float64),
        np.asarray(rates, dtype=np.float64),
        float(step),
        float(sig_r),
        float(sig_p),
        float(momentum),
        int(solver["maximum_iterations"]),
        float(solver["moment_tolerance"]),
        float(solver["feasibility_tolerance"]),
        float(solver["damping_min"]),
    )
    offsets, weights, counts, rate_probability, feasible, iterations = arrays
    legal = np.zeros_like(feasible, dtype=bool)
    for source_rate in range(len(rates)):
        for edge_index in range(3):
            destination_rate = source_rate + edge_index - 1
            if 0 <= destination_rate < len(rates):
                legal[:, source_rate, edge_index] = True
    invalid = np.argwhere(legal & ~feasible.astype(bool))
    if len(invalid):
        time_index, source_rate, edge_index = (
            int(value) for value in invalid[0]
        )
        destination_rate = source_rate + edge_index - 1
        mean_shift = (
            0.5
            * (rates[source_rate] + rates[destination_rate])
            * dm[time_index]
            - dz[time_index]
        )
        raise RuntimeError(
            "moment projection infeasible; candidate fails closed at "
            f"row={time_index}, source_rate={source_rate}, "
            f"destination_rate={destination_rate}, mean_shift={mean_shift:.17g}"
        )
    return {
        "offsets": offsets,
        "weights": weights,
        "counts": counts,
        "rate_probability": rate_probability,
        "feasible": feasible,
        "iterations": iterations,
        "legal": legal,
        "sha256": array_bundle_sha256(
            offsets=offsets,
            weights=weights,
            counts=counts,
            rate_probability=rate_probability,
            feasible=feasible,
        ),
    }


@njit(cache=True, nogil=True)
def _hmm2_fb_correlated_joint(
    emission_ll: np.ndarray,
    step: float,
    rates: np.ndarray,
    start_p: float,
    start_sig: float,
    r0: float,
    r0_sig: float,
    emission_lambda: float,
    edge_offsets: np.ndarray,
    edge_weights: np.ndarray,
    edge_counts: np.ndarray,
    rate_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    time_count, position_count = emission_ll.shape
    rate_count = len(rates)
    negative = np.float32(-1.0e18)
    alpha = np.full(
        (time_count, position_count, rate_count),
        negative,
        np.float32,
    )
    previous = np.full((position_count, rate_count), negative, np.float32)
    for position_index in range(position_count):
        delta_position = (position_index - start_p) * step
        initial_position_logp = -0.5 * (delta_position / start_sig) ** 2
        if initial_position_logp < -60.0:
            continue
        for rate_index in range(rate_count):
            delta_rate = (rates[rate_index] - r0) / r0_sig
            previous[position_index, rate_index] = np.float32(
                initial_position_logp - 0.5 * delta_rate * delta_rate
            )

    current = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count):
        for destination_position in range(position_count):
            for destination_rate in range(rate_count):
                best = negative
                first_source = max(destination_rate - 1, 0)
                last_source = min(destination_rate + 1, rate_count - 1)
                for source_rate in range(first_source, last_source + 1):
                    edge_index = destination_rate - source_rate + 1
                    count = int(edge_counts[time_index, source_rate, edge_index])
                    rate_weight = rate_probability[
                        time_index,
                        source_rate,
                        edge_index,
                    ]
                    for kernel_index in range(count):
                        position_weight = edge_weights[
                            time_index,
                            source_rate,
                            edge_index,
                            kernel_index,
                        ]
                        if position_weight <= 0.0:
                            continue
                        source_position = (
                            destination_position
                            - edge_offsets[
                                time_index,
                                source_rate,
                                edge_index,
                                kernel_index,
                            ]
                        )
                        if 0 <= source_position < position_count:
                            value = (
                                previous[source_position, source_rate]
                                + np.log(rate_weight)
                                + np.log(position_weight)
                            )
                            best = max(best, value)
                if best > negative / 2:
                    total = 0.0
                    for source_rate in range(first_source, last_source + 1):
                        edge_index = destination_rate - source_rate + 1
                        count = int(
                            edge_counts[
                                time_index,
                                source_rate,
                                edge_index,
                            ]
                        )
                        rate_weight = rate_probability[
                            time_index,
                            source_rate,
                            edge_index,
                        ]
                        for kernel_index in range(count):
                            position_weight = edge_weights[
                                time_index,
                                source_rate,
                                edge_index,
                                kernel_index,
                            ]
                            if position_weight <= 0.0:
                                continue
                            source_position = (
                                destination_position
                                - edge_offsets[
                                    time_index,
                                    source_rate,
                                    edge_index,
                                    kernel_index,
                                ]
                            )
                            if 0 <= source_position < position_count:
                                total += np.exp(
                                    previous[source_position, source_rate]
                                    + np.log(rate_weight)
                                    + np.log(position_weight)
                                    - best
                                )
                    current[destination_position, destination_rate] = (
                        np.float32(
                            best
                            + np.log(total)
                            + emission_lambda
                            * emission_ll[
                                time_index,
                                destination_position,
                            ]
                        )
                    )
                else:
                    current[destination_position, destination_rate] = negative
        alpha[time_index] = current
        previous[:, :] = current

    final_best = np.max(alpha[-1])
    final_total = np.sum(np.exp(alpha[-1] - final_best))
    log_likelihood = float(final_best) + np.log(final_total)
    posterior_position = np.zeros((time_count, position_count), np.float64)
    posterior_rate = np.zeros((time_count, rate_count), np.float64)
    beta_next = np.zeros((position_count, rate_count), np.float32)
    values = alpha[-1] + beta_next
    best = np.max(values)
    total = np.sum(np.exp(values - best))
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            probability = np.exp(
                values[position_index, rate_index] - best
            ) / total
            posterior_position[-1, position_index] += probability
            posterior_rate[-1, rate_index] += probability

    beta_current = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count - 1, 0, -1):
        for source_position in range(position_count):
            for source_rate in range(rate_count):
                best = negative
                first_destination = max(source_rate - 1, 0)
                last_destination = min(source_rate + 1, rate_count - 1)
                for destination_rate in range(
                    first_destination,
                    last_destination + 1,
                ):
                    edge_index = destination_rate - source_rate + 1
                    count = int(edge_counts[time_index, source_rate, edge_index])
                    rate_weight = rate_probability[
                        time_index,
                        source_rate,
                        edge_index,
                    ]
                    for kernel_index in range(count):
                        position_weight = edge_weights[
                            time_index,
                            source_rate,
                            edge_index,
                            kernel_index,
                        ]
                        if position_weight <= 0.0:
                            continue
                        destination_position = (
                            source_position
                            + edge_offsets[
                                time_index,
                                source_rate,
                                edge_index,
                                kernel_index,
                            ]
                        )
                        if 0 <= destination_position < position_count:
                            value = (
                                np.log(rate_weight)
                                + np.log(position_weight)
                                + emission_lambda
                                * emission_ll[
                                    time_index,
                                    destination_position,
                                ]
                                + beta_next[
                                    destination_position,
                                    destination_rate,
                                ]
                            )
                            best = max(best, value)
                if best > negative / 2:
                    total = 0.0
                    for destination_rate in range(
                        first_destination,
                        last_destination + 1,
                    ):
                        edge_index = destination_rate - source_rate + 1
                        count = int(
                            edge_counts[
                                time_index,
                                source_rate,
                                edge_index,
                            ]
                        )
                        rate_weight = rate_probability[
                            time_index,
                            source_rate,
                            edge_index,
                        ]
                        for kernel_index in range(count):
                            position_weight = edge_weights[
                                time_index,
                                source_rate,
                                edge_index,
                                kernel_index,
                            ]
                            if position_weight <= 0.0:
                                continue
                            destination_position = (
                                source_position
                                + edge_offsets[
                                    time_index,
                                    source_rate,
                                    edge_index,
                                    kernel_index,
                                ]
                            )
                            if 0 <= destination_position < position_count:
                                total += np.exp(
                                    np.log(rate_weight)
                                    + np.log(position_weight)
                                    + emission_lambda
                                    * emission_ll[
                                        time_index,
                                        destination_position,
                                    ]
                                    + beta_next[
                                        destination_position,
                                        destination_rate,
                                    ]
                                    - best
                                )
                    beta_current[source_position, source_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    beta_current[source_position, source_rate] = negative
        values = alpha[time_index - 1] + beta_current
        best = np.max(values)
        total = np.sum(np.exp(values - best))
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                probability = np.exp(
                    values[position_index, rate_index] - best
                ) / total
                posterior_position[time_index - 1, position_index] += probability
                posterior_rate[time_index - 1, rate_index] += probability
                beta_next[position_index, rate_index] = beta_current[
                    position_index,
                    rate_index,
                ]

    maximum_normalization_error = 0.0
    for time_index in range(time_count):
        position_total = np.sum(posterior_position[time_index])
        rate_total = np.sum(posterior_rate[time_index])
        posterior_position[time_index] /= position_total
        posterior_rate[time_index] /= rate_total
        maximum_normalization_error = max(
            maximum_normalization_error,
            abs(np.sum(posterior_position[time_index]) - 1.0),
            abs(np.sum(posterior_rate[time_index]) - 1.0),
        )
    return (
        posterior_position,
        posterior_rate,
        log_likelihood,
        maximum_normalization_error,
    )


def joint_edge_moment_audit(
    *,
    dm: np.ndarray,
    dz: np.ndarray,
    rates: np.ndarray,
    step: float,
    sig_r: float,
    sig_p: float,
    momentum: float,
    table: Mapping[str, Any],
    row_idx: np.ndarray,
) -> pd.DataFrame:
    position_sigma = max(sig_p, 0.35 * step)
    target_variance = position_sigma**2
    offsets = np.asarray(table["offsets"])
    weights = np.asarray(table["weights"])
    counts = np.asarray(table["counts"])
    rate_probability = np.asarray(table["rate_probability"])
    rows: list[dict[str, Any]] = []
    for time_index in range(len(dm)):
        maximum_rate_error = 0.0
        maximum_weight_error = 0.0
        maximum_mean_error = 0.0
        maximum_variance_error = 0.0
        maximum_covariance_error = 0.0
        parent_abs_bias_sum = 0.0
        candidate_abs_bias_sum = 0.0
        legal_edges = 0
        support_5 = 0
        support_7 = 0
        support_9 = 0
        for source_rate in range(len(rates)):
            parent_rate_kernel = rate_kernel_probabilities(
                rates,
                float(dm[time_index]),
                sig_r,
                momentum,
            )
            legal_destination: list[int] = []
            legal_probability: list[float] = []
            projected_means: list[float] = []
            target_means: list[float] = []
            for edge_index in range(3):
                destination_rate = source_rate + edge_index - 1
                if not 0 <= destination_rate < len(rates):
                    continue
                legal_edges += 1
                probability = float(
                    rate_probability[
                        time_index,
                        source_rate,
                        edge_index,
                    ]
                )
                maximum_rate_error = max(
                    maximum_rate_error,
                    abs(
                        probability
                        - float(parent_rate_kernel[source_rate, edge_index])
                    ),
                )
                count = int(counts[time_index, source_rate, edge_index])
                support_5 += int(count == 5)
                support_7 += int(count == 7)
                support_9 += int(count == 9)
                mean_shift = (
                    0.5
                    * (rates[source_rate] + rates[destination_rate])
                    * dm[time_index]
                    - dz[time_index]
                )
                total, projected_mean, projected_variance = projection_moments(
                    mean_shift,
                    step,
                    offsets[time_index, source_rate, edge_index],
                    weights[time_index, source_rate, edge_index],
                    count,
                )
                maximum_weight_error = max(
                    maximum_weight_error,
                    abs(total - 1.0),
                )
                maximum_mean_error = max(
                    maximum_mean_error,
                    abs(projected_mean - mean_shift),
                )
                maximum_variance_error = max(
                    maximum_variance_error,
                    abs(projected_variance - target_variance),
                )
                parent_offsets, parent_weights = (
                    parent_position_kernel_probabilities(
                        float(
                            rates[destination_rate] * dm[time_index]
                            - dz[time_index]
                        ),
                        step,
                        sig_p,
                    )
                )
                parent_mean = float(
                    np.sum(
                        parent_offsets.astype(np.float64)
                        * step
                        * parent_weights
                    )
                )
                parent_target = float(
                    rates[destination_rate] * dm[time_index] - dz[time_index]
                )
                parent_abs_bias_sum += abs(parent_mean - parent_target)
                candidate_abs_bias_sum += abs(projected_mean - mean_shift)
                legal_destination.append(destination_rate)
                legal_probability.append(probability)
                projected_means.append(projected_mean)
                target_means.append(mean_shift)

            probability_array = np.asarray(legal_probability, dtype=np.float64)
            probability_array /= probability_array.sum()
            delta_rate = (
                rates[np.asarray(legal_destination, dtype=np.int64)]
                - rates[source_rate]
            )
            projected_array = np.asarray(projected_means, dtype=np.float64)
            target_array = np.asarray(target_means, dtype=np.float64)
            mean_delta_rate = float(np.sum(probability_array * delta_rate))
            projected_displacement = float(
                np.sum(probability_array * projected_array)
            )
            target_displacement = float(
                np.sum(probability_array * target_array)
            )
            projected_covariance = float(
                np.sum(
                    probability_array
                    * (projected_array - projected_displacement)
                    * (delta_rate - mean_delta_rate)
                )
            )
            target_covariance = float(
                np.sum(
                    probability_array
                    * (target_array - target_displacement)
                    * (delta_rate - mean_delta_rate)
                )
            )
            maximum_covariance_error = max(
                maximum_covariance_error,
                abs(projected_covariance - target_covariance),
            )
        rows.append(
            {
                "row_idx": int(row_idx[time_index]),
                "delta_md": float(dm[time_index]),
                "delta_z": float(dz[time_index]),
                "legal_edges": legal_edges,
                "support_5_edges": support_5,
                "support_7_edges": support_7,
                "support_9_edges": support_9,
                "rate_marginal_max_abs_error": maximum_rate_error,
                "legal_edge_weight_sum_max_error": maximum_weight_error,
                "conditional_mean_max_abs_error_ft": maximum_mean_error,
                "conditional_variance_max_abs_error_ft2": (
                    maximum_variance_error
                ),
                "source_row_joint_covariance_max_abs_error": (
                    maximum_covariance_error
                ),
                "exp209_grid_abs_mean_bias_sum_ft": parent_abs_bias_sum,
                "candidate_grid_abs_mean_bias_sum_ft": (
                    candidate_abs_bias_sum
                ),
                "forward_backward_joint_table_identity": True,
                "joint_edge_table_sha256": str(table["sha256"]),
            }
        )
    return pd.DataFrame(rows)


def run_joint_hmm(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    table = build_joint_edge_table(
        dm=np.asarray(prepared["dm"], dtype=np.float64),
        dz=np.asarray(prepared["dz"], dtype=np.float64),
        rates=np.asarray(prepared["rates"], dtype=np.float64),
        step=float(hmm["position_grid_step_ft"]),
        sig_r=float(hmm["sig_r"]),
        sig_p=float(hmm["sig_p"]),
        momentum=float(hmm["momentum"]),
        projection=projection,
    )
    posterior_position, posterior_rate, log_likelihood, normalization_error = (
        _hmm2_fb_correlated_joint(
            np.asarray(prepared["emission_ll"], dtype=np.float32),
            float(hmm["position_grid_step_ft"]),
            np.asarray(prepared["rates"], dtype=np.float64),
            float(prepared["start_p"]),
            float(hmm["start_sigma_ft"]),
            float(prepared["r0"]),
            float(hmm["initial_rate_sigma"]),
            float(hmm["emission_lambda"]),
            np.asarray(table["offsets"], dtype=np.int64),
            np.asarray(table["weights"], dtype=np.float64),
            np.asarray(table["counts"], dtype=np.int8),
            np.asarray(table["rate_probability"], dtype=np.float64),
        )
    )
    tvt_grid = np.asarray(prepared["tvt_grid"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    mean_tvt = posterior_position @ tvt_grid
    variance_tvt = np.sum(
        posterior_position * (tvt_grid[None, :] - mean_tvt[:, None]) ** 2,
        axis=1,
    )
    std_tvt = np.sqrt(np.maximum(variance_tvt, 0.0))
    rate_mean = posterior_rate @ rates
    rate_variance = posterior_rate @ (rates**2) - rate_mean**2
    rate_std = np.sqrt(np.maximum(rate_variance, 0.0))
    rate_edge_mass = posterior_rate[:, 0] + posterior_rate[:, -1]
    audit = joint_edge_moment_audit(
        dm=np.asarray(prepared["dm"], dtype=np.float64),
        dz=np.asarray(prepared["dz"], dtype=np.float64),
        rates=rates,
        step=float(hmm["position_grid_step_ft"]),
        sig_r=float(hmm["sig_r"]),
        sig_p=float(hmm["sig_p"]),
        momentum=float(hmm["momentum"]),
        table=table,
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
    )
    prediction_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        mean_tvt=np.asarray(mean_tvt, dtype=np.float32),
        std_tvt=np.asarray(std_tvt, dtype=np.float32),
    )
    rate_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        rate_mean=np.asarray(rate_mean, dtype=np.float32),
        rate_std=np.asarray(rate_std, dtype=np.float32),
        rate_edge_mass=np.asarray(rate_edge_mass, dtype=np.float32),
    )
    return {
        "posterior_position": posterior_position,
        "posterior_rate": posterior_rate,
        "mean_tvt": mean_tvt,
        "std_tvt": std_tvt,
        "rate_mean": rate_mean,
        "rate_std": rate_std,
        "rate_edge_mass": rate_edge_mass,
        "log_likelihood": float(log_likelihood),
        "posterior_normalization_max_error": float(normalization_error),
        "moment_audit": audit,
        "joint_edge_table_sha256": str(table["sha256"]),
        "prediction_sha256": prediction_sha,
        "rate_readout_sha256": rate_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 6. Numerical contracts and target-free prediction freeze
#
# Contracts are evaluated before any suffix truth, role, fold, persistent
# episode, or error is opened:
#
# - exact exp209 emission/input identity on every fixed32 well;
# - rate marginal and edge moment/covariance parity;
# - an exhaustive small-path reference for the correlated joint HMM;
# - shared forward/backward table identity and posterior normalization.
#
# Candidate prediction, posterior rate readout, joint-edge table, and moment
# audit receive content SHAs before the guarded truth-late phase.

# %%
def dense_joint_transition_matrix(
    *,
    time_index: int,
    position_count: int,
    rate_count: int,
    table: Mapping[str, Any],
) -> np.ndarray:
    state_count = position_count * rate_count
    transition = np.zeros((state_count, state_count), dtype=np.float64)
    for source_position in range(position_count):
        for source_rate in range(rate_count):
            source_state = source_position * rate_count + source_rate
            for edge_index in range(3):
                destination_rate = source_rate + edge_index - 1
                if not 0 <= destination_rate < rate_count:
                    continue
                rate_weight = float(
                    table["rate_probability"][
                        time_index,
                        source_rate,
                        edge_index,
                    ]
                )
                count = int(
                    table["counts"][time_index, source_rate, edge_index]
                )
                for kernel_index in range(count):
                    destination_position = (
                        source_position
                        + int(
                            table["offsets"][
                                time_index,
                                source_rate,
                                edge_index,
                                kernel_index,
                            ]
                        )
                    )
                    if 0 <= destination_position < position_count:
                        destination_state = (
                            destination_position * rate_count
                            + destination_rate
                        )
                        transition[source_state, destination_state] += (
                            rate_weight
                            * float(
                                table["weights"][
                                    time_index,
                                    source_rate,
                                    edge_index,
                                    kernel_index,
                                ]
                            )
                        )
    return transition


def exhaustive_joint_path_reference(
    *,
    emission_ll: np.ndarray,
    step: float,
    rates: np.ndarray,
    start_p: float,
    start_sig: float,
    r0: float,
    r0_sig: float,
    emission_lambda: float,
    table: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    time_count, position_count = emission_ll.shape
    rate_count = len(rates)
    state_count = position_count * rate_count
    if time_count > 3 or state_count > 9:
        raise ValueError("joint exhaustive reference is limited to tiny HMMs")
    initial = np.zeros(state_count, dtype=np.float64)
    for position_index in range(position_count):
        delta_position = (position_index - start_p) * step
        position_logp = -0.5 * (delta_position / start_sig) ** 2
        if position_logp < -60.0:
            continue
        for rate_index, rate in enumerate(rates):
            delta_rate = (rate - r0) / r0_sig
            initial[position_index * rate_count + rate_index] = np.exp(
                position_logp - 0.5 * delta_rate * delta_rate
            )
    transitions = [
        dense_joint_transition_matrix(
            time_index=time_index,
            position_count=position_count,
            rate_count=rate_count,
            table=table,
        )
        for time_index in range(time_count)
    ]
    emission_probability = np.repeat(
        np.exp(emission_lambda * emission_ll),
        rate_count,
        axis=1,
    )
    paths: list[tuple[int, ...]] = []
    path_weights: list[float] = []

    def extend(
        time_index: int,
        previous_state: int,
        path: tuple[int, ...],
        weight: float,
    ) -> None:
        if time_index == time_count:
            paths.append(path)
            path_weights.append(weight)
            return
        for destination_state in range(state_count):
            next_weight = (
                weight
                * transitions[time_index][previous_state, destination_state]
                * emission_probability[time_index, destination_state]
            )
            if next_weight > 0.0:
                extend(
                    time_index + 1,
                    destination_state,
                    (*path, destination_state),
                    next_weight,
                )

    for initial_state in range(state_count):
        if initial[initial_state] > 0.0:
            extend(0, initial_state, (), float(initial[initial_state]))
    weights = np.asarray(path_weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("joint exhaustive reference has zero mass")
    posterior_position = np.zeros(
        (time_count, position_count),
        dtype=np.float64,
    )
    posterior_rate = np.zeros((time_count, rate_count), dtype=np.float64)
    for probability, path in zip(weights / total, paths, strict=True):
        for time_index, state in enumerate(path):
            posterior_position[time_index, state // rate_count] += probability
            posterior_rate[time_index, state % rate_count] += probability
    return posterior_position, posterior_rate, float(np.log(total))


def brute_force_joint_reference_contract(
    hmm: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    emission_ll = np.asarray(
        [
            [-0.20, -0.01, -0.40],
            [-0.35, -0.05, -0.10],
            [-0.60, -0.15, -0.02],
        ],
        dtype=np.float32,
    )
    dm = np.ones(3, dtype=np.float64)
    dz = np.zeros(3, dtype=np.float64)
    rates = np.asarray([-0.70, 0.0, 0.70], dtype=np.float64)
    step = float(hmm["position_grid_step_ft"])
    table = build_joint_edge_table(
        dm=dm,
        dz=dz,
        rates=rates,
        step=step,
        sig_r=float(hmm["sig_r"]),
        sig_p=float(hmm["sig_p"]),
        momentum=float(hmm["momentum"]),
        projection=projection,
    )
    common = {
        "emission_ll": emission_ll,
        "step": step,
        "rates": rates,
        "start_p": 1.0,
        "start_sig": float(hmm["start_sigma_ft"]),
        "r0": 0.0,
        "r0_sig": 0.5,
        "emission_lambda": float(hmm["emission_lambda"]),
    }
    observed_position, observed_rate, observed_loglik, observed_norm = (
        _hmm2_fb_correlated_joint(
            common["emission_ll"],
            common["step"],
            common["rates"],
            common["start_p"],
            common["start_sig"],
            common["r0"],
            common["r0_sig"],
            common["emission_lambda"],
            table["offsets"],
            table["weights"],
            table["counts"],
            table["rate_probability"],
        )
    )
    reference_position, reference_rate, reference_loglik = (
        exhaustive_joint_path_reference(table=table, **common)
    )
    position_diff = float(
        np.max(np.abs(observed_position - reference_position))
    )
    rate_diff = float(np.max(np.abs(observed_rate - reference_rate)))
    prediction_grid = np.arange(3, dtype=np.float64) * step
    prediction_diff = float(
        np.max(
            np.abs(
                observed_position @ prediction_grid
                - reference_position @ prediction_grid
            )
        )
    )
    loglik_diff = abs(float(observed_loglik) - float(reference_loglik))
    maximum = max(
        position_diff,
        rate_diff,
        prediction_diff,
        loglik_diff,
    )
    return {
        "position_posterior_max_abs": position_diff,
        "rate_posterior_max_abs": rate_diff,
        "prediction_max_abs_ft": prediction_diff,
        "log_likelihood_abs": loglik_diff,
        "maximum_abs": maximum,
        "posterior_normalization_max_error": float(observed_norm),
        "joint_edge_table_sha256": table["sha256"],
        "pass": bool(maximum <= 1.0e-6),
    }


@dataclass
class FrozenWell:
    well: str
    eval_id: np.ndarray
    row_idx: np.ndarray
    raw_gr_missing: np.ndarray
    parent_prediction: np.ndarray
    candidate_prediction: np.ndarray
    candidate_posterior_std: np.ndarray
    candidate_rate_mean: np.ndarray
    candidate_rate_std: np.ndarray
    candidate_rate_edge_mass: np.ndarray
    moment_audit: pd.DataFrame
    input_contract: dict[str, Any]
    prediction_sha256: str
    rate_readout_sha256: str
    joint_edge_table_sha256: str
    log_likelihood: float
    posterior_normalization_max_error: float
    elapsed_seconds: float
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    prefix_rows: int
    role: str | None = None
    fold: int | None = None


def freeze_target_free_well(
    *,
    well: str,
    expected_prefix_rows: int,
    expected_suffix_rows: int,
    parent_rows: pd.DataFrame,
    raw_dir: Path,
    hmm: Mapping[str, Any],
    projection: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, hmm)
    if int(prepared["prefix_rows"]) != int(expected_prefix_rows):
        raise ValueError(f"{well}: prefix row count changed")
    if len(prepared["eval_index"]) != int(expected_suffix_rows):
        raise ValueError(f"{well}: suffix row count changed")
    expected_ids = parent_cache_ids_for_rows(
        well,
        np.asarray(prepared["eval_index"], dtype=np.int64),
    )
    aligned_parent = parent_rows.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(
        aligned_parent["id"].astype(str).to_numpy(),
        expected_ids,
    ):
        raise ValueError(f"{well}: saved parent row identity changed")
    input_contract = input_contract_from_prepared(prepared)
    result = run_joint_hmm(prepared, hmm, projection)
    moment_audit = result["moment_audit"].copy()
    moment_audit.insert(0, "well", well)
    frozen = FrozenWell(
        well=well,
        eval_id=expected_ids,
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        parent_prediction=aligned_parent["parent_prediction"].to_numpy(np.float64),
        candidate_prediction=np.asarray(result["mean_tvt"], dtype=np.float64),
        candidate_posterior_std=np.asarray(result["std_tvt"], dtype=np.float64),
        candidate_rate_mean=np.asarray(result["rate_mean"], dtype=np.float64),
        candidate_rate_std=np.asarray(result["rate_std"], dtype=np.float64),
        candidate_rate_edge_mass=np.asarray(
            result["rate_edge_mass"],
            dtype=np.float64,
        ),
        moment_audit=moment_audit,
        input_contract=input_contract,
        prediction_sha256=str(result["prediction_sha256"]),
        rate_readout_sha256=str(result["rate_readout_sha256"]),
        joint_edge_table_sha256=str(result["joint_edge_table_sha256"]),
        log_likelihood=float(result["log_likelihood"]),
        posterior_normalization_max_error=float(
            result["posterior_normalization_max_error"]
        ),
        elapsed_seconds=float(result["elapsed_seconds"]),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        prefix_rows=int(prepared["prefix_rows"]),
    )
    ledger.freeze(well)
    return frozen


def attach_scope_identity(
    frozen_wells: Sequence[FrozenWell],
    identity: pd.DataFrame,
) -> None:
    by_well = identity.set_index("well")
    if set(by_well.index) != {item.well for item in frozen_wells}:
        raise ValueError("fixed32 identity/prediction wells differ")
    for item in frozen_wells:
        row = by_well.loc[item.well]
        item.role = str(row["role"])
        item.fold = int(row["fold"])


def prediction_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "id": item.eval_id,
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "parent_exp209_tvt": item.parent_prediction,
                    "candidate_continuous_kinematic_tvt": (
                        item.candidate_prediction
                    ),
                    "candidate_posterior_std_tvt": (
                        item.candidate_posterior_std
                    ),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"],
        kind="mergesort",
    )


def rate_readout_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "candidate_rate_mean": item.candidate_rate_mean,
                    "candidate_rate_std": item.candidate_rate_std,
                    "candidate_rate_edge_mass": item.candidate_rate_edge_mass,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"],
        kind="mergesort",
    )


def moment_audit_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    return pd.concat(
        [item.moment_audit for item in frozen_wells],
        ignore_index=True,
    ).sort_values(["well", "row_idx"], kind="mergesort")


def input_contract_summary(
    frozen_wells: Sequence[FrozenWell],
) -> dict[str, Any]:
    return {
        "emission_identity_max_abs": max(
            item.input_contract["emission_identity_max_abs"]
            for item in frozen_wells
        ),
        "per_well_sha256": {
            item.well: item.input_contract["sha256"]
            for item in frozen_wells
        },
    }


# %% [markdown]
# ## 7. Truth-late persistent-episode and safety readout
#
# Suffix TVT, role/fold identity, persistent episodes, exp408 cause labels, and
# all error metrics are opened only after all candidate predictions, posterior
# rate readouts, input contracts, joint-edge SHAs, and moment audits are frozen.

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


def well_truth_late_metrics(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    if frozen.role is None or frozen.fold is None:
        raise RuntimeError("role/fold identity was not attached after freeze")
    actual = truth["TVT"].to_numpy(np.float64)
    parent_error = frozen.parent_prediction - actual
    candidate_error = frozen.candidate_prediction - actual
    parent_rmse = float(np.sqrt(np.mean(parent_error**2)))
    candidate_rmse = float(np.sqrt(np.mean(candidate_error**2)))
    return {
        "well": frozen.well,
        "role": frozen.role,
        "fold": frozen.fold,
        "rows": len(actual),
        "parent_rmse_ft": parent_rmse,
        "candidate_rmse_ft": candidate_rmse,
        "rmse_delta_vs_parent_ft": candidate_rmse - parent_rmse,
        "improved_vs_parent": candidate_rmse < parent_rmse,
        "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
        "prediction_sha256": frozen.prediction_sha256,
        "rate_readout_sha256": frozen.rate_readout_sha256,
        "joint_edge_table_sha256": frozen.joint_edge_table_sha256,
        "hmm_seconds": frozen.elapsed_seconds,
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
        ["well", "start_suffix_offset"],
        kind="mergesort",
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
        if frozen.fold is None or int(episode.fold) != int(frozen.fold):
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
        candidate_error = frozen.candidate_prediction[offsets] - actual
        parent_sse = float(np.sum(parent_error**2))
        candidate_sse = float(np.sum(candidate_error**2))
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": str(episode.well),
                "fold": int(frozen.fold),
                "cause": str(episode.cause),
                "rows": len(offsets),
                "start_row_idx": int(episode.start_row_idx),
                "end_row_idx_exclusive": int(episode.end_row_idx_exclusive),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "candidate_sse_reduction_vs_parent": (
                    1.0 - candidate_sse / parent_sse
                    if parent_sse > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fold", "well", "start_row_idx"],
        kind="mergesort",
    )


# %% [markdown]
# ## 8. Stage 0 gates, generated artifacts, and metrics
#
# The fixed32 result is a mechanism preflight, not CV or promotion evidence.
# Every technical and mechanism gate is an AND condition. Any failure closes
# this branch without changing support selection, solver, grid, position/rate
# noise, emission, prior, selector, or blend on the same fixed32 sample.

# %%
def fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def pooled_rmse_from_well_rows(
    frame: pd.DataFrame,
    column: str,
) -> float:
    weights = frame["rows"].to_numpy(np.float64)
    values = frame[column].to_numpy(np.float64)
    return float(np.sqrt(np.average(values**2, weights=weights)))


def finite_readout_coverage(
    frozen_wells: Sequence[FrozenWell],
) -> float:
    finite = 0
    total = 0
    attributes = (
        "candidate_prediction",
        "candidate_posterior_std",
        "candidate_rate_mean",
        "candidate_rate_std",
        "candidate_rate_edge_mass",
    )
    for item in frozen_wells:
        for attribute in attributes:
            values = np.asarray(getattr(item, attribute), dtype=np.float64)
            finite += int(np.isfinite(values).sum())
            total += int(values.size)
    return fraction(finite, total)


def evaluate_mechanism_gates(
    *,
    config: Mapping[str, Any],
    episode_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
) -> dict[str, Any]:
    mechanism = get_nested(config, "gates.stage0_fixed32.mechanism")
    parent_episode_sse = float(episode_readout["parent_sse"].sum())
    candidate_episode_sse = float(episode_readout["candidate_sse"].sum())
    persistent_reduction = (
        1.0 - candidate_episode_sse / parent_episode_sse
        if parent_episode_sse > 0.0
        else math.nan
    )
    forward_cause = str(
        get_nested(config, "data.exp408_episode_causes.forward_cause")
    )
    forward = episode_readout.loc[
        episode_readout["cause"].eq(forward_cause)
    ]
    forward_parent_sse = float(forward["parent_sse"].sum())
    forward_candidate_sse = float(forward["candidate_sse"].sum())
    forward_reduction = (
        1.0 - forward_candidate_sse / forward_parent_sse
        if forward_parent_sse > 0.0
        else math.nan
    )

    persistent_wells = well_metrics.loc[
        well_metrics["role"].eq("persistent")
    ]
    control_wells = well_metrics.loc[well_metrics["role"].eq("control")]
    improved_wells = int(
        persistent_wells["improved_vs_parent"].astype(bool).sum()
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_episodes = episode_readout.loc[
            episode_readout["fold"].eq(fold)
        ]
        parent_sse = float(fold_episodes["parent_sse"].sum())
        candidate_sse = float(fold_episodes["candidate_sse"].sum())
        fold_rows.append(
            {
                "fold": fold,
                "episodes": len(fold_episodes),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "improved": bool(
                    len(fold_episodes) > 0 and candidate_sse < parent_sse
                ),
            }
        )
    improving_folds = int(sum(row["improved"] for row in fold_rows))
    control_parent_rmse = pooled_rmse_from_well_rows(
        control_wells,
        "parent_rmse_ft",
    )
    control_candidate_rmse = pooled_rmse_from_well_rows(
        control_wells,
        "candidate_rmse_ft",
    )
    control_delta = control_candidate_rmse - control_parent_rmse
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
                mechanism[
                    "forward_cause_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_episode_sse_reduction": bool(
            math.isfinite(persistent_reduction)
            and persistent_reduction
            >= float(
                mechanism[
                    "persistent_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_improved_wells": bool(
            improved_wells
            >= int(mechanism["persistent_improved_wells_min"])
        ),
        "persistent_improving_folds": bool(
            improving_folds
            >= int(mechanism["persistent_improving_folds_min"])
        ),
        "matched_control_pooled_rmse": bool(
            control_delta
            <= float(mechanism["matched_control_pooled_rmse_delta_max_ft"])
        ),
        "matched_control_by_well_p95": bool(
            control_p95
            <= float(
                mechanism["matched_control_by_well_delta_p95_max_ft"]
            )
        ),
    }
    return {
        "gates": gates,
        "all_mechanism_gates_pass": bool(all(gates.values())),
        "diagnostics": {
            "parent_persistent_episode_sse": parent_episode_sse,
            "candidate_persistent_episode_sse": candidate_episode_sse,
            "persistent_episode_sse_reduction_fraction": persistent_reduction,
            "forward_cause": forward_cause,
            "forward_cause_episodes": len(forward),
            "forward_cause_parent_sse": forward_parent_sse,
            "forward_cause_candidate_sse": forward_candidate_sse,
            "forward_cause_episode_sse_reduction_fraction": forward_reduction,
            "persistent_improved_wells": improved_wells,
            "persistent_sse_by_fold": fold_rows,
            "persistent_improving_folds": improving_folds,
            "control_parent_rmse_ft": control_parent_rmse,
            "control_candidate_rmse_ft": control_candidate_rmse,
            "control_rmse_delta_ft": control_delta,
            "control_by_well_rmse_delta_p95_ft": control_p95,
        },
    }


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    identity: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    input_contract: Mapping[str, Any],
    brute_force_contract: Mapping[str, Any],
    moment_audit: pd.DataFrame,
    prediction_artifact: Mapping[str, Any],
    rate_artifact: Mapping[str, Any],
    transition_artifact: Mapping[str, Any],
    episode_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "gates.stage0_fixed32.technical")
    maximum_posterior_error = max(
        item.posterior_normalization_max_error for item in frozen_wells
    )
    maximum_rate_error = float(
        moment_audit["rate_marginal_max_abs_error"].max()
    )
    maximum_weight_error = float(
        moment_audit["legal_edge_weight_sum_max_error"].max()
    )
    maximum_mean_error = float(
        moment_audit["conditional_mean_max_abs_error_ft"].max()
    )
    maximum_variance_error = float(
        moment_audit["conditional_variance_max_abs_error_ft2"].max()
    )
    maximum_covariance_error = float(
        moment_audit["source_row_joint_covariance_max_abs_error"].max()
    )
    parent_grid_bias = float(
        moment_audit["exp209_grid_abs_mean_bias_sum_ft"].sum()
    )
    candidate_grid_bias = float(
        moment_audit["candidate_grid_abs_mean_bias_sum_ft"].sum()
    )
    grid_bias_reduction = (
        1.0 - candidate_grid_bias / parent_grid_bias
        if parent_grid_bias > 0.0
        else math.nan
    )
    finite_coverage = finite_readout_coverage(frozen_wells)
    treatment_seconds = float(
        sum(item.elapsed_seconds for item in frozen_wells)
    )
    runtime_projection = treatment_seconds * 773.0 / 32.0
    nonidentity_rows = int(
        sum(
            np.count_nonzero(
                np.asarray(item.candidate_prediction, dtype=np.float32)
                != np.asarray(item.parent_prediction, dtype=np.float32)
            )
            for item in frozen_wells
        )
    )
    technical = {
        "fixed32_roles_and_unique_wells": bool(
            len(identity) == 32
            and identity["well"].nunique() == 32
            and identity["role"].value_counts().to_dict()
            == {"persistent": 16, "control": 16}
        ),
        "fixed32_fold_coverage": bool(
            set(identity["fold"].astype(int)) == {0, 1, 2, 3, 4}
        ),
        "fixed32_rows": bool(
            int(moment_audit.shape[0])
            == int(technical_config["expected_rows"])
        ),
        "emission_identity": bool(
            input_contract["emission_identity_max_abs"] <= 1.0e-12
        ),
        "rate_marginal_parity": bool(
            maximum_rate_error
            <= float(technical_config["rate_marginal_max_abs_error"])
        ),
        "legal_edge_weight_sum": bool(
            maximum_weight_error
            <= float(
                technical_config["legal_edge_weight_sum_max_error"]
            )
        ),
        "conditional_mean": bool(
            maximum_mean_error
            <= float(
                technical_config["conditional_mean_max_abs_error_ft"]
            )
        ),
        "conditional_variance": bool(
            maximum_variance_error
            <= float(
                technical_config[
                    "conditional_variance_max_abs_error_ft2"
                ]
            )
        ),
        "source_row_joint_covariance": bool(
            maximum_covariance_error
            <= float(
                technical_config[
                    "source_row_joint_covariance_max_abs_error"
                ]
            )
        ),
        "forward_backward_joint_table_identity": bool(
            moment_audit[
                "forward_backward_joint_table_identity"
            ].astype(bool).all()
            and bool(
                technical_config[
                    "forward_backward_joint_table_identity_required"
                ]
            )
        ),
        "brute_force_small_reference": bool(
            brute_force_contract["pass"]
            and brute_force_contract["maximum_abs"]
            <= float(
                technical_config[
                    "brute_force_posterior_prediction_max_abs_error"
                ]
            )
        ),
        "posterior_normalization": bool(
            maximum_posterior_error
            <= float(
                technical_config["posterior_normalization_max_error"]
            )
        ),
        "finite_prediction_and_diagnostic_coverage": bool(
            finite_coverage
            >= float(technical_config["finite_coverage_min"])
        ),
        "exp209_grid_mean_bias_reduction": bool(
            math.isfinite(grid_bias_reduction)
            and grid_bias_reduction
            >= float(
                technical_config[
                    "exp209_grid_mean_bias_reduction_min_fraction"
                ]
            )
        ),
        "truth_reads_before_all_freeze": bool(
            ledger.truth_rows_before_all_freeze
            == int(
                technical_config[
                    "truth_role_fold_episode_reads_before_freeze_max"
                ]
            )
        ),
        "role_fold_reads_before_all_freeze": bool(
            ledger.role_fold_rows_before_all_freeze
            == int(
                technical_config[
                    "truth_role_fold_episode_reads_before_freeze_max"
                ]
            )
        ),
        "episode_reads_before_all_freeze": bool(
            ledger.episode_rows_before_all_freeze
            == int(
                technical_config[
                    "truth_role_fold_episode_reads_before_freeze_max"
                ]
            )
        ),
        "prediction_readback_sha": bool(
            prediction_artifact["logical_sha256"]
            == prediction_artifact["readback_logical_sha256"]
        ),
        "rate_readout_readback_sha": bool(
            rate_artifact["logical_sha256"]
            == rate_artifact["readback_logical_sha256"]
        ),
        "moment_audit_readback_sha": bool(
            transition_artifact["logical_sha256"]
            == transition_artifact["readback_logical_sha256"]
        ),
        "runtime_projection": bool(
            runtime_projection
            <= float(
                technical_config["projected_stage1_runtime_seconds_max"]
            )
        ),
        "peak_rss": bool(
            peak_rss_gb()
            <= float(technical_config["peak_rss_gb_max"])
        ),
    }
    mechanism = evaluate_mechanism_gates(
        config=config,
        episode_readout=episode_readout,
        well_metrics=well_metrics,
    )
    all_pass = bool(all(technical.values()) and mechanism["all_mechanism_gates_pass"])
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": {
            "total_wells": len(frozen_wells),
            "total_suffix_rows": int(
                sum(len(item.row_idx) for item in frozen_wells)
            ),
            "maximum_rate_marginal_error": maximum_rate_error,
            "maximum_legal_edge_weight_sum_error": maximum_weight_error,
            "maximum_conditional_mean_error_ft": maximum_mean_error,
            "maximum_conditional_variance_error_ft2": (
                maximum_variance_error
            ),
            "maximum_source_row_joint_covariance_error": (
                maximum_covariance_error
            ),
            "exp209_grid_abs_mean_bias_sum_ft": parent_grid_bias,
            "candidate_grid_abs_mean_bias_sum_ft": candidate_grid_bias,
            "exp209_grid_mean_bias_reduction_fraction": grid_bias_reduction,
            "maximum_posterior_normalization_error": maximum_posterior_error,
            "finite_coverage": finite_coverage,
            "candidate_control_prediction_nonidentity_rows": (
                nonidentity_rows
            ),
            "stage0_elapsed_seconds": float(elapsed_seconds),
            "candidate_hmm_seconds": treatment_seconds,
            "stage1_runtime_projection_seconds": runtime_projection,
            "peak_rss_gb": peak_rss_gb(),
            "truth_rows_before_all_freeze": (
                ledger.truth_rows_before_all_freeze
            ),
            "role_fold_rows_before_all_freeze": (
                ledger.role_fold_rows_before_all_freeze
            ),
            "episode_rows_before_all_freeze": (
                ledger.episode_rows_before_all_freeze
            ),
            "fixed32_is_mechanism_only_not_cv_or_promotion": True,
        },
        "stage0_all_gates_pass": all_pass,
        "stage1_eligible_for_separate_approval": all_pass,
        "fail_action": get_nested(
            config,
            "gates.stage0_fixed32.fail_action",
        ),
        "fixed32_is_cv": False,
        "fixed32_is_promotion_evidence": False,
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP439_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp439 Stage 0 must run on Kaggle CPU; local execution is disabled")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    set_num_threads(1)
    hmm = scientific_contract["fixed_from_exp209"]
    projection = scientific_contract["lattice_projection"]
    brute_force_contract = brute_force_joint_reference_contract(
        hmm,
        projection,
    )

    ledger = LeakageLedger(expected_wells=32)
    scope, manifest_report = load_fixed32_target_free_scope(config, ledger)
    expected_rows = int(scope["suffix_rows"].sum())
    parent, parent_report = load_saved_parent_predictions(
        config,
        set(scope["well"].astype(str)),
        expected_rows,
        ledger,
    )
    raw_dir = train_data_dir(config)

    frozen_wells: list[FrozenWell] = []
    for scope_row in scope.itertuples(index=False):
        well = str(scope_row.well)
        parent_rows = parent.loc[parent["well"].eq(well)].copy()
        frozen = freeze_target_free_well(
            well=well,
            expected_prefix_rows=int(scope_row.prefix_rows),
            expected_suffix_rows=int(scope_row.suffix_rows),
            parent_rows=parent_rows,
            raw_dir=raw_dir,
            hmm=hmm,
            projection=projection,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        print(
            json.dumps(
                {
                    "well": well,
                    "rows": len(frozen.row_idx),
                    "seconds": frozen.elapsed_seconds,
                    "prediction_sha256": frozen.prediction_sha256,
                    "joint_edge_table_sha256": (
                        frozen.joint_edge_table_sha256
                    ),
                },
                sort_keys=True,
            )
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all fixed32 wells were frozen")

    output_dir = artifacts_dir()
    predictions = prediction_frame(frozen_wells)
    rate_readout = rate_readout_frame(frozen_wells)
    moment_audit = moment_audit_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output_dir / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    rate_artifact = write_deterministic_gzip_csv(
        output_dir / f"{EXPERIMENT_NAME}_stage0_rate_readout.csv.gz",
        rate_readout,
    )
    transition_artifact = write_deterministic_gzip_csv(
        output_dir / f"{EXPERIMENT_NAME}_stage0_moment_audit.csv.gz",
        moment_audit,
    )
    input_contract = input_contract_summary(frozen_wells)
    numerical_contract = {
        "input": input_contract,
        "brute_force_joint_reference": brute_force_contract,
    }
    numerical_contract_artifact = write_json(
        output_dir / f"{EXPERIMENT_NAME}_stage0_numerical_contract.json",
        numerical_contract,
    )

    # The guarded truth-late phase begins only after all target-free SHAs exist.
    identity = load_fixed32_identity_after_all_freeze(config, ledger)
    attach_scope_identity(frozen_wells, identity)
    truth_by_well: dict[str, pd.DataFrame] = {}
    well_metric_rows: list[dict[str, Any]] = []
    for item in frozen_wells:
        truth = load_truth_after_all_freeze(item, raw_dir, ledger)
        truth_by_well[item.well] = truth
        well_metric_rows.append(well_truth_late_metrics(item, truth))
    well_metrics = pd.DataFrame(well_metric_rows).sort_values(
        ["role", "fold", "well"],
        kind="mergesort",
    )
    frozen_by_well = {item.well: item for item in frozen_wells}
    persistent_wells = set(
        identity.loc[identity["role"].eq("persistent"), "well"].astype(str)
    )
    episodes, episode_input_report = load_persistent_episodes_after_all_freeze(
        config,
        persistent_wells,
        ledger,
    )
    episode_readout = episode_truth_late_readout(
        episodes,
        frozen_by_well,
        truth_by_well,
    )
    well_artifact = write_csv(
        output_dir / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    episode_artifact = write_csv(
        output_dir / f"{EXPERIMENT_NAME}_stage0_episode_metrics.csv",
        episode_readout,
    )
    elapsed_seconds = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        identity=identity,
        frozen_wells=frozen_wells,
        input_contract=input_contract,
        brute_force_contract=brute_force_contract,
        moment_audit=moment_audit,
        prediction_artifact=prediction_artifact,
        rate_artifact=rate_artifact,
        transition_artifact=transition_artifact,
        episode_readout=episode_readout,
        well_metrics=well_metrics,
        ledger=ledger,
        elapsed_seconds=elapsed_seconds,
    )
    gate_artifact = write_json(
        output_dir / f"{EXPERIMENT_NAME}_stage0_gate_report.json",
        gates,
    )
    rerun_contract = {
        "deterministic_anchor": False,
        "first_run_is_anchor": False,
        "required_before_anchor_reconsideration": [
            "identical_prediction_logical_sha256",
            "identical_moment_audit_logical_sha256",
        ],
        "first_run_prediction_logical_sha256": prediction_artifact[
            "logical_sha256"
        ],
        "first_run_moment_audit_logical_sha256": transition_artifact[
            "logical_sha256"
        ],
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0_pass_pending_separate_stage1_approval"
            if gates["stage0_all_gates_pass"]
            else "stage0_fail_closed"
        ),
        "scientific_variant": SCIENTIFIC_VARIANT,
        "execution_contract": execution_contract,
        "scientific_contract": scientific_contract,
        "runtime": runtime_versions(),
        "numba_threads": 1,
        "state_order": (
            "well,row,tvt_position,u_rate,source_rate,"
            "destination_rate,position_offset"
        ),
        "input_reports": {
            "fixed32": manifest_report,
            "saved_exp209": parent_report,
            "persistent_episodes": episode_input_report,
        },
        "numerical_contract": numerical_contract,
        "gates": gates,
        "reproducibility": rerun_contract,
        "artifacts": {
            "prediction": prediction_artifact,
            "rate_readout": rate_artifact,
            "moment_audit": transition_artifact,
            "numerical_contract": numerical_contract_artifact,
            "well_metrics": well_artifact,
            "episode_metrics": episode_artifact,
            "gate_report": gate_artifact,
        },
        "elapsed_seconds": elapsed_seconds,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    metrics_report = write_json(metrics_path(), summary)
    summary["artifacts"]["metrics"] = metrics_report
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Configuration preview and guarded execution
#
# Importing or opening the notebook never starts Stage 0. A Kaggle run requires
# both `runtime.run_approved=true` and `execution.run_hmm=true`. After Kaggle
# version 1 reproduced the preregistered moment-infeasibility fail-close, the
# committed config keeps both false to prevent a same-experiment rescue rerun.

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    EXECUTION_PREVIEW = validate_execution_contract(
        CONFIG,
        require_run_authorization=False,
    )
    SCIENTIFIC_PREVIEW = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "execution": EXECUTION_PREVIEW,
                "scientific_variant": SCIENTIFIC_VARIANT,
                "parent": PARENT_EXPERIMENT,
                "scientific_difference": (
                    "arrival_rate_position_step_to_trapezoidal_"
                    "source_destination_joint_edge"
                ),
                "run_approved": get_nested(CONFIG, "runtime.run_approved"),
                "run_hmm": get_nested(CONFIG, "execution.run_hmm"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    STAGE0_RESULT = run_stage0(CONFIG)
