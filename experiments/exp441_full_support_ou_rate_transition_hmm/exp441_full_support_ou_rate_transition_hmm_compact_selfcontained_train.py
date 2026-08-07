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
# # exp441 full-support OU rate-transition exact HMM — train
#
# This implementation keeps the complete exp209 exact-HMM contract except for
# one preregistered scientific change: the adjacent three-state Euler rate
# kernel is replaced by the exact OU conditional law integrated over every
# finite rate-grid Voronoi bin. Rate mass outside the finite grid is discarded
# without row renormalization. Position transition, GR emission, prior,
# forward/backward ordering, and posterior-mean readout remain unchanged.
#
# This file is an implementation candidate only. It does not authorize a
# Kaggle package, Stage 0 execution, Stage 1, inference, or submission.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable contracts
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Fixed32 scope, saved parent, and target-free raw inputs
# 4. Exact exp209 input preparation
# 5. Full-support exact OU rate kernel
# 6. Exact forward-backward and brute-force contracts
# 7. Target-free kernel, diagnostic, and prediction freeze
# 8. Truth-late mechanism and safety readout
# 9. Technical and mechanism gates
# 10. Guarded Kaggle CPU orchestration

# %% [markdown]
# ## 1. Imports and immutable contracts

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

EXPERIMENT_NAME = "exp441_full_support_ou_rate_transition_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
NEGATIVE_LOG_SENTINEL = np.float32(-1.0e18)

FORBIDDEN_TARGET_FREE_COLUMNS = frozenset(
    {
        "TVT",
        "tvt_true",
        "target_tvt",
        "error",
        "abs_error",
        "fold",
        "role",
        "hidden_like_role",
        "episode_id",
        "cause",
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
        raise ValueError("wrong exp441 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp441 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp441 scientific parent changed")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp441 implementation is not authorized")
    if bool(get_nested(config, "execution.stage1_run_authorized", True)):
        raise ValueError("Stage 1 must remain disabled")
    if bool(get_nested(config, "execution.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "execution.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu", True)):
        raise ValueError("exp441 is CPU-only")

    expected = {
        "scientific_variants": 1,
        "stage0_candidate_hmm_well_runs": 32,
        "stage1_candidate_hmm_well_runs": 773,
        "parent_control_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1)) for key in expected
    }
    if observed != expected:
        raise ValueError(f"exp441 execution contract changed: {observed} != {expected}")
    if bool(get_nested(config, "data.exp209_saved_control.regenerate", True)):
        raise ValueError("saved exp209 control must not be regenerated")
    if require_run_authorization:
        if get_nested(config, "execution.selected_stage") != "stage0_fixed32":
            raise RuntimeError("exp441 selected_stage must be stage0_fixed32")
        if not bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ):
            raise RuntimeError(
                "exp441 Stage 0 requires canonical train notebook adoption"
            )
        if not bool(get_nested(config, "execution.stage0_run_authorized", False)):
            raise RuntimeError(
                "exp441 implementation approval does not authorize Stage 0 execution"
            )
        if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
            raise RuntimeError(
                "exp441 Stage 0 execution requires separate Kaggle package approval"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("exp441 run_hmm remains fail-closed")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError(
                "exp441 Stage 0 prediction creation remains fail-closed"
            )
        if bool(get_nested(config, "execution.create_submission", True)):
            raise ValueError("exp441 Stage 0 must not create a submission")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(get_nested(config, "model.fixed_from_exp209") or {})
    expected_fixed = {
        "position_grid_step_ft": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "momentum": 0.998,
        "emission_family": "gaussian_typewell_gr",
        "emission_lambda": 1.0,
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "sigma_clip": [10.0, 60.0],
        "start_sigma_ft": 0.75,
        "initial_rate_sigma": 0.01,
        "band_pad_ft": 100.0,
        "rate_center": "zero",
        "rate_boundary_semantics": "preserve_parent_substochastic_outward_mass",
        "position_boundary_semantics": "preserve_parent_truncation",
        "position_mean_formula": "r_destination*delta_MD-delta_Z",
        "output": "smoothed_posterior_mean_and_std",
    }
    if fixed != expected_fixed:
        raise ValueError(f"exp209 HMM contract changed: {fixed} != {expected_fixed}")
    candidate = dict(get_nested(config, "model.candidate_rate_transition") or {})
    expected_candidate = {
        "family": "exact_ornstein_uhlenbeck",
        "kappa_formula": "-log(parent_momentum)",
        "conditional_mean_formula": "exp(-kappa*delta_MD)*r_source",
        "conditional_variance_formula": (
            "sig_r^2*(1-exp(-2*kappa*delta_MD))/(2*kappa)"
        ),
        "zero_kappa_limit_formula": "sig_r^2*delta_MD",
        "discretization": (
            "gaussian_cdf_integral_over_all_rate_bin_voronoi_cells"
        ),
        "outer_tail_policy": (
            "discard_outside_finite_parent_rate_support_without_renormalization"
        ),
        "numerical_dtype": "float64",
    }
    if candidate != expected_candidate:
        raise ValueError(
            f"exp441 candidate contract changed: {candidate} != {expected_candidate}"
        )
    variants = list(get_nested(config, "model.active_scientific_variants") or [])
    if variants != ["full_support_exact_ou"]:
        raise ValueError("exp441 must contain exactly one frozen scientific candidate")
    return {
        "fixed_from_exp209": fixed,
        "candidate_rate_transition": candidate,
        "active_scientific_variants": variants,
        "forbidden": list(get_nested(config, "model.forbidden") or []),
    }


# %% [markdown]
# ## 2. Notebook-safe paths, SHA helpers, and leakage ledger

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    for candidate in (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp441 config.yaml was not found")


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
            normalized[column] = normalized[column].astype(np.int8)
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
    persisted = frame.copy()
    for column in persisted.columns:
        if pd.api.types.is_float_dtype(persisted[column]):
            persisted[column] = persisted[column].astype(np.float64)
        elif pd.api.types.is_integer_dtype(persisted[column]):
            persisted[column] = persisted[column].astype(np.int64)
        elif pd.api.types.is_bool_dtype(persisted[column]):
            persisted[column] = persisted[column].astype(np.int8)
        else:
            persisted[column] = persisted[column].astype(str)
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
        persisted.to_csv(text, index=False, lineterminator="\n")
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
        "logical_sha256": logical_frame_sha256(persisted),
        "readback_logical_sha256": logical_frame_sha256(readback),
        "rows": len(persisted),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0**3)
    return value / (1024.0**2)


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


def resolve_unique_file(
    *,
    filename: str,
    candidates: Sequence[str],
    patterns: Sequence[str],
) -> Path:
    root = find_project_root()
    matches: list[Path] = []
    for raw in candidates:
        candidate = Path(raw)
        for path in (
            candidate,
            root / candidate if not candidate.is_absolute() else candidate,
            PACKAGE_DIR / candidate if not candidate.is_absolute() else candidate,
        ):
            if path.is_file() and path.name == filename:
                matches.append(path)
            elif path.is_dir() and (path / filename).is_file():
                matches.append(path / filename)
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
    expected_wells: int = 32
    frozen_wells: set[str] = field(default_factory=set)
    scope_rows: int = 0
    target_free_rows: int = 0
    role_fold_rows_before_all_freeze: int = 0
    truth_rows_before_all_freeze: int = 0
    episode_rows_before_all_freeze: int = 0
    cause_rows_before_all_freeze: int = 0
    role_fold_rows_after_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    episode_rows_after_all_freeze: int = 0
    cause_rows_after_all_freeze: int = 0
    freeze_records: list[dict[str, str]] = field(default_factory=list)

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    @property
    def forbidden_reads_before_all_freeze(self) -> int:
        return (
            self.role_fold_rows_before_all_freeze
            + self.truth_rows_before_all_freeze
            + self.episode_rows_before_all_freeze
            + self.cause_rows_before_all_freeze
        )

    def record_scope(self, rows: int) -> None:
        self.scope_rows += int(rows)

    def record_target_free(self, rows: int) -> None:
        self.target_free_rows += int(rows)

    def freeze(
        self,
        well: str,
        *,
        transition_kernel_sha256: str,
        prediction_sha256: str,
        diagnostic_sha256: str,
    ) -> None:
        if (
            not transition_kernel_sha256
            or not prediction_sha256
            or not diagnostic_sha256
        ):
            raise ValueError("all target-free SHA values are required before freeze")
        self.frozen_wells.add(str(well))
        self.freeze_records.append(
            {
                "well": str(well),
                "transition_kernel_sha256": transition_kernel_sha256,
                "prediction_sha256": prediction_sha256,
                "diagnostic_sha256": diagnostic_sha256,
            }
        )

    def record_role_fold_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.role_fold_rows_before_all_freeze += int(rows)
            raise RuntimeError("role/fold identity was read before all target-free freeze")
        self.role_fold_rows_after_all_freeze += int(rows)

    def record_truth_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.truth_rows_before_all_freeze += int(rows)
            raise RuntimeError("truth was read before all target-free freeze")
        self.truth_rows_after_all_freeze += int(rows)

    def record_episode_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.episode_rows_before_all_freeze += int(rows)
            raise RuntimeError("episodes were read before all target-free freeze")
        self.episode_rows_after_all_freeze += int(rows)

    def record_cause_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.cause_rows_before_all_freeze += int(rows)
            raise RuntimeError("episode causes were read before all target-free freeze")
        self.cause_rows_after_all_freeze += int(rows)


# %% [markdown]
# ## 3. Fixed32 scope, saved parent, and target-free raw inputs
#
# Before all 32 predictions are frozen the fixed manifest is opened with
# `usecols=["well"]` only. Role, fold, suffix counts, episode membership, and
# cause labels are deliberately read in a second late phase.

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
    expected = str(spec["expected_sha256"])
    if observed != expected:
        raise ValueError(
            f"fixed32 manifest SHA changed: expected={expected}, observed={observed}"
        )
    return path, observed


def load_fixed32_scope(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[list[str], dict[str, Any]]:
    path, observed = fixed32_manifest_path(config)
    frame = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    wells = sorted(frame["well"].astype(str).tolist())
    expected = int(get_nested(config, "data.fixed32_manifest.total_wells"))
    if len(wells) != expected or len(set(wells)) != expected:
        raise ValueError("fixed32 scope must contain 32 unique wells")
    ledger.record_scope(len(wells))
    return wells, {"path": str(path), "sha256": observed, "rows": len(wells)}


def load_fixed32_identity_after_all_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, observed = fixed32_manifest_path(config)
    frame = pd.read_csv(path, dtype={"well": str, "matched_persistent_well": str})
    ledger.record_role_fold_late(len(frame))
    required = {"well", "role", "fold", "prefix_rows", "suffix_rows", "selection_hash"}
    if not required.issubset(frame.columns):
        raise ValueError("fixed32 manifest schema changed")
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 identity must contain 32 unique wells")
    if frame["role"].value_counts().to_dict() != {"persistent": 16, "control": 16}:
        raise ValueError("fixed32 role counts changed")
    if frame.groupby("fold").size().to_dict() != {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}:
        raise ValueError("fixed32 fold counts changed")
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    if int(frame["suffix_rows"].sum()) != expected_rows:
        raise ValueError("fixed32 suffix row count changed")
    frame = frame.sort_values("well", kind="mergesort").reset_index(drop=True)
    return frame, {
        "path": str(path),
        "sha256": observed,
        "logical_sha256": logical_frame_sha256(frame),
        "rows": len(frame),
    }


def parent_row_indices_from_cache_ids(frame: pd.DataFrame) -> np.ndarray:
    row_indices = np.empty(len(frame), dtype=np.int64)
    for offset, (well, identifier) in enumerate(
        zip(frame["well"].astype(str), frame["id"].astype(str), strict=True)
    ):
        prefix = f"{well}_"
        if not identifier.startswith(prefix):
            raise ValueError(f"saved parent id has wrong well prefix: {identifier}")
        suffix = identifier[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"saved parent id has invalid row suffix: {identifier}")
        row_indices[offset] = int(suffix)
    return row_indices


def parent_cache_ids_for_rows(well: str, row_indices: np.ndarray) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    if not str(well) or rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("invalid well or row indices for parent cache ids")
    return np.asarray([f"{well}_{int(row)}" for row in rows], dtype=str)


def load_saved_parent_predictions(
    config: Mapping[str, Any],
    target_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp209_saved_control")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(value) for value in spec["candidates"]],
        patterns=[str(value) for value in spec["patterns"]],
    )
    decompressed = sha256_decompressed_csv(path)
    expected_sha = str(spec["expected_decompressed_sha256"])
    if decompressed != expected_sha:
        raise ValueError(
            f"saved exp209 decompressed SHA changed: {decompressed} != {expected_sha}"
        )
    prediction_column = str(spec["prediction_column"])
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["id", "well", prediction_column],
        dtype={"id": str, "well": str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("saved exp209 control has no fixed32 rows")
    frame = pd.concat(pieces, ignore_index=True).rename(
        columns={prediction_column: "parent_prediction"}
    )
    frame["row_idx"] = parent_row_indices_from_cache_ids(frame)
    frame["parent_prediction"] = pd.to_numeric(
        frame["parent_prediction"], errors="raise"
    )
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
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
    forbidden = FORBIDDEN_TARGET_FREE_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: decoder input contains {sorted(forbidden)}")
    typewell = pd.read_csv(typewell_path).sort_values("TVT").reset_index(drop=True)
    ledger.record_target_free(len(horizontal) + len(typewell))
    return horizontal, typewell


# %% [markdown]
# ## 4. Exact exp209 input preparation

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


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    fixed: Mapping[str, Any],
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

    init_rate, rate_rows, valid_steps = robust_initial_rate(known)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
    sigma_clip = list(fixed["sigma_clip"])
    gr_sigma = float(
        np.clip(np.nanstd(residual), float(sigma_clip[0]), float(sigma_clip[1]))
    )

    step = float(fixed["position_grid_step_ft"])
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    grid_min = max(
        float(typewell_tvt.min()) - 40.0,
        last_tvt - float(fixed["band_pad_ft"]),
    )
    grid_max = min(
        float(typewell_tvt.max()) + 40.0,
        last_tvt + float(fixed["band_pad_ft"]),
    )
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
    span = max(float(fixed["rate_span"]), abs(init_rate) + 0.04)
    rates = np.linspace(-span, span, int(fixed["n_rates"]), dtype=np.float64)
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
    }


# %% [markdown]
# ## 5. Full-support exact OU rate kernel
#
# The parent momentum is interpreted as a unit-MD OU decay. For each observed
# `delta_MD`, the exact Gaussian conditional is integrated over all finite
# rate-bin Voronoi intervals. The two outer intervals end one half-grid step
# beyond the first and last rate centers. Probability beyond those finite
# edges is deliberately discarded, and no transition row is renormalized.

# %%
@njit(cache=True, nogil=True)
def ou_conditional_parameters(
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> tuple[float, float, float]:
    if delta_md < 0.0:
        raise ValueError("delta_MD must be non-negative")
    if sig_r <= 0.0:
        raise ValueError("sig_r must be positive")
    if momentum <= 0.0 or momentum > 1.0:
        raise ValueError("momentum must be in (0, 1]")
    kappa = -math.log(momentum)
    if abs(kappa) <= 1.0e-14:
        decay = 1.0
        variance = sig_r * sig_r * delta_md
    else:
        decay = math.exp(-kappa * delta_md)
        variance = (
            sig_r
            * sig_r
            * (-math.expm1(-2.0 * kappa * delta_md))
            / (2.0 * kappa)
        )
    return kappa, decay, max(variance, 0.0)


@njit(cache=True, nogil=True)
def finite_voronoi_edges(rates: np.ndarray) -> np.ndarray:
    if len(rates) < 2:
        raise ValueError("at least two rate centers are required")
    edges = np.empty(len(rates) + 1, dtype=np.float64)
    for index in range(1, len(rates)):
        edges[index] = 0.5 * (rates[index - 1] + rates[index])
    edges[0] = rates[0] - 0.5 * (rates[1] - rates[0])
    edges[-1] = rates[-1] + 0.5 * (rates[-1] - rates[-2])
    return edges


@njit(cache=True, nogil=True, parallel=True)
def precompute_full_support_ou_log_kernels(
    delta_md: np.ndarray,
    rates: np.ndarray,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    time_count = len(delta_md)
    rate_count = len(rates)
    edges = finite_voronoi_edges(rates)
    output = np.full(
        (time_count, rate_count, rate_count),
        -np.inf,
        dtype=np.float64,
    )
    sqrt_two = math.sqrt(2.0)
    for time_index in prange(time_count):
        _, decay, variance = ou_conditional_parameters(
            float(delta_md[time_index]),
            sig_r,
            momentum,
        )
        sigma = math.sqrt(variance)
        for source_rate in range(rate_count):
            mean = decay * rates[source_rate]
            if sigma <= 0.0:
                for destination_rate in range(rate_count):
                    if (
                        edges[destination_rate] <= mean
                        and mean < edges[destination_rate + 1]
                    ):
                        output[time_index, source_rate, destination_rate] = 0.0
                if mean == edges[-1]:
                    output[time_index, source_rate, rate_count - 1] = 0.0
                continue
            for destination_rate in range(rate_count):
                lower_z = (
                    edges[destination_rate] - mean
                ) / (sigma * sqrt_two)
                upper_z = (
                    edges[destination_rate + 1] - mean
                ) / (sigma * sqrt_two)
                probability = 0.5 * (
                    math.erf(upper_z) - math.erf(lower_z)
                )
                if probability > 0.0:
                    output[
                        time_index, source_rate, destination_rate
                    ] = math.log(probability)
    return output


def full_support_ou_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    logs = precompute_full_support_ou_log_kernels(
        np.asarray([delta_md], dtype=np.float64),
        np.asarray(rates, dtype=np.float64),
        float(sig_r),
        float(momentum),
    )[0]
    return np.exp(logs)


@njit(cache=True, nogil=True)
def parent_position_kernel_probabilities(
    mean_shift: float,
    position_step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    effective_sigma = max(sig_p, 0.35 * position_step)
    center = int(math.floor(mean_shift / position_step + 0.5))
    offsets = np.empty(5, dtype=np.int64)
    log_values = np.empty(5, dtype=np.float64)
    for kernel_index in range(5):
        offset = center - 2 + kernel_index
        offsets[kernel_index] = offset
        delta = offset * position_step - mean_shift
        log_values[kernel_index] = -0.5 * (
            delta / effective_sigma
        ) ** 2
    maximum = np.max(log_values)
    probabilities = np.exp(log_values - maximum)
    probabilities /= np.sum(probabilities)
    return offsets, probabilities


def position_kernel_parity_contract(
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    step = float(fixed["position_grid_step_ft"])
    sig_p = float(fixed["sig_p"])
    maximum_error = 0.0
    for mean_shift in np.linspace(-3.25, 3.25, 67):
        offsets, observed = parent_position_kernel_probabilities(
            float(mean_shift),
            step,
            sig_p,
        )
        center = int(np.floor(float(mean_shift) / step + 0.5))
        expected_offsets = np.arange(center - 2, center + 3, dtype=np.int64)
        sigma = max(sig_p, 0.35 * step)
        expected = np.exp(
            -0.5
            * (
                (expected_offsets.astype(np.float64) * step - mean_shift)
                / sigma
            )
            ** 2
        )
        expected /= expected.sum()
        if not np.array_equal(offsets, expected_offsets):
            maximum_error = math.inf
            break
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(observed - expected))),
        )
    return {
        "maximum_absolute_error": maximum_error,
        "pass": bool(maximum_error <= 1.0e-12),
    }


def ou_kernel_numeric_audit(
    log_kernels: np.ndarray,
    delta_md: np.ndarray,
    rates: np.ndarray,
    sig_r: float,
    momentum: float,
) -> dict[str, float]:
    kernels = np.exp(np.asarray(log_kernels, dtype=np.float64))
    edges = finite_voronoi_edges(np.asarray(rates, dtype=np.float64))
    maximum_mass_error = 0.0
    maximum_mean_error = 0.0
    maximum_variance_error = 0.0
    minimum_row_mass = 1.0
    maximum_row_mass = 0.0
    sqrt_two = math.sqrt(2.0)
    kappa = -math.log(float(momentum))
    for time_index, step_md in enumerate(
        np.asarray(delta_md, dtype=np.float64)
    ):
        _, decay, variance = ou_conditional_parameters(
            float(step_md),
            float(sig_r),
            float(momentum),
        )
        independent_decay = float(momentum) ** float(step_md)
        if abs(kappa) <= 1.0e-14:
            independent_variance = float(sig_r) ** 2 * float(step_md)
        else:
            independent_variance = (
                float(sig_r)
                ** 2
                * (1.0 - independent_decay * independent_decay)
                / (2.0 * kappa)
            )
        maximum_variance_error = max(
            maximum_variance_error,
            abs(variance - independent_variance),
        )
        for source_rate, source_value in enumerate(rates):
            mean = decay * float(source_value)
            independent_mean = independent_decay * float(source_value)
            maximum_mean_error = max(
                maximum_mean_error,
                abs(mean - independent_mean),
            )
            sigma = math.sqrt(variance)
            if sigma <= 0.0:
                expected_mass = float(edges[0] <= mean <= edges[-1])
            else:
                expected_mass = 0.5 * (
                    math.erf((edges[-1] - mean) / (sigma * sqrt_two))
                    - math.erf((edges[0] - mean) / (sigma * sqrt_two))
                )
            observed_mass = float(
                kernels[time_index, source_rate].sum(dtype=np.float64)
            )
            maximum_mass_error = max(
                maximum_mass_error,
                abs(observed_mass - expected_mass),
            )
            minimum_row_mass = min(minimum_row_mass, observed_mass)
            maximum_row_mass = max(maximum_row_mass, observed_mass)
    return {
        "analytic_in_support_mass_max_abs_error": maximum_mass_error,
        "interior_conditional_mean_max_abs_error": maximum_mean_error,
        "interior_conditional_variance_max_abs_error": maximum_variance_error,
        "minimum_transition_row_mass": minimum_row_mass,
        "maximum_transition_row_mass": maximum_row_mass,
    }


# %% [markdown]
# ## 6. Exact forward-backward and brute-force contracts
#
# The forward/backward message arrays remain float32 as in exp209. Exact OU
# transition tables are float64 and are reused unchanged in both directions.
# Predictive and filtered rate means are captured target-free for the later
# preregistered under-response readout.

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_full_support_ou(
    emission,
    delta_md,
    delta_z,
    position_step,
    rates,
    rate_log_kernels,
    sig_p,
    start_position_index,
    start_sigma,
    initial_rate,
    initial_rate_sigma,
    emission_lambda,
):
    time_count, position_count = emission.shape
    rate_count = len(rates)
    neg = np.float32(-1.0e18)
    alpha = np.full((time_count, position_count, rate_count), neg, np.float32)
    previous = np.full((position_count, rate_count), neg, np.float32)
    for position_index in range(position_count):
        delta_position = (
            position_index - start_position_index
        ) * position_step
        position_log = -0.5 * (delta_position / start_sigma) ** 2
        if position_log < -60.0:
            continue
        for rate_index in range(rate_count):
            delta_rate = (
                rates[rate_index] - initial_rate
            ) / initial_rate_sigma
            previous[position_index, rate_index] = np.float32(
                position_log - 0.5 * delta_rate * delta_rate
            )

    rate_updated = np.empty((position_count, rate_count), np.float32)
    predictive = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)
    predictive_rate_mean = np.empty(time_count, np.float64)
    filtered_rate_mean = np.empty(time_count, np.float64)
    maximum_forward_normalization_error = 0.0

    for time_index in range(time_count):
        rate_log_kernel = rate_log_kernels[time_index]
        for position_index in prange(position_count):
            for destination_rate in range(rate_count):
                best = neg
                for source_rate in range(rate_count):
                    transition_log = rate_log_kernel[
                        source_rate, destination_rate
                    ]
                    if not np.isfinite(transition_log):
                        continue
                    value = (
                        previous[position_index, source_rate]
                        + transition_log
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for source_rate in range(rate_count):
                        transition_log = rate_log_kernel[
                            source_rate, destination_rate
                        ]
                        if not np.isfinite(transition_log):
                            continue
                        total += np.exp(
                            previous[position_index, source_rate]
                            + transition_log
                            - best
                        )
                    rate_updated[
                        position_index, destination_rate
                    ] = np.float32(best + np.log(total))
                else:
                    rate_updated[position_index, destination_rate] = neg

        for destination_rate in prange(rate_count):
            mean_shift = (
                rates[destination_rate] * delta_md[time_index]
                - delta_z[time_index]
            )
            offsets, probabilities = parent_position_kernel_probabilities(
                mean_shift,
                position_step,
                sig_p,
            )
            position_log_kernel = np.log(probabilities)
            for destination_position in range(position_count):
                best = neg
                for kernel_index in range(5):
                    source_position = (
                        destination_position - offsets[kernel_index]
                    )
                    if 0 <= source_position < position_count:
                        value = (
                            rate_updated[source_position, destination_rate]
                            + position_log_kernel[kernel_index]
                        )
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        source_position = (
                            destination_position - offsets[kernel_index]
                        )
                        if 0 <= source_position < position_count:
                            total += np.exp(
                                rate_updated[
                                    source_position, destination_rate
                                ]
                                + position_log_kernel[kernel_index]
                                - best
                            )
                    value = best + np.log(total)
                    predictive[
                        destination_position, destination_rate
                    ] = np.float32(value)
                    current[
                        destination_position, destination_rate
                    ] = np.float32(
                        value
                        + emission_lambda
                        * emission[time_index, destination_position]
                    )
                else:
                    predictive[destination_position, destination_rate] = neg
                    current[destination_position, destination_rate] = neg

        predictive_best = neg
        filtered_best = neg
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_best = max(
                    predictive_best,
                    predictive[position_index, rate_index],
                )
                filtered_best = max(
                    filtered_best,
                    current[position_index, rate_index],
                )
        predictive_total = 0.0
        filtered_total = 0.0
        predictive_rate_total = 0.0
        filtered_rate_total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_probability = np.exp(
                    predictive[position_index, rate_index] - predictive_best
                )
                filtered_probability = np.exp(
                    current[position_index, rate_index] - filtered_best
                )
                predictive_total += predictive_probability
                filtered_total += filtered_probability
                predictive_rate_total += (
                    predictive_probability * rates[rate_index]
                )
                filtered_rate_total += (
                    filtered_probability * rates[rate_index]
                )
        predictive_rate_mean[time_index] = (
            predictive_rate_total / predictive_total
        )
        filtered_rate_mean[time_index] = (
            filtered_rate_total / filtered_total
        )
        predictive_check = 0.0
        filtered_check = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_check += (
                    np.exp(
                        predictive[position_index, rate_index]
                        - predictive_best
                    )
                    / predictive_total
                )
                filtered_check += (
                    np.exp(
                        current[position_index, rate_index] - filtered_best
                    )
                    / filtered_total
                )
                alpha[time_index, position_index, rate_index] = current[
                    position_index, rate_index
                ]
                previous[position_index, rate_index] = current[
                    position_index, rate_index
                ]
        maximum_forward_normalization_error = max(
            maximum_forward_normalization_error,
            abs(predictive_check - 1.0),
            abs(filtered_check - 1.0),
        )

    final_best = np.max(alpha[time_count - 1])
    final_total = 0.0
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            final_total += math.exp(
                float(alpha[time_count - 1, position_index, rate_index])
                - float(final_best)
            )
    log_likelihood = float(final_best) + np.log(final_total)

    posterior_position = np.zeros(
        (time_count, position_count), dtype=np.float64
    )
    posterior_rate = np.zeros(
        (time_count, rate_count), dtype=np.float64
    )
    beta_next = np.zeros((position_count, rate_count), np.float32)
    final_values = alpha[time_count - 1] + beta_next
    final_best = np.max(final_values)
    final_total = 0.0
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            final_total += math.exp(
                float(final_values[position_index, rate_index])
                - float(final_best)
            )
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            probability = (
                math.exp(
                    float(final_values[position_index, rate_index])
                    - float(final_best)
                )
                / final_total
            )
            posterior_position[
                time_count - 1, position_index
            ] += probability
            posterior_rate[time_count - 1, rate_index] += probability

    beta_current = np.empty((position_count, rate_count), np.float32)
    beta_position = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count - 1, 0, -1):
        for destination_rate in prange(rate_count):
            mean_shift = (
                rates[destination_rate] * delta_md[time_index]
                - delta_z[time_index]
            )
            offsets, probabilities = parent_position_kernel_probabilities(
                mean_shift,
                position_step,
                sig_p,
            )
            position_log_kernel = np.log(probabilities)
            for source_position in range(position_count):
                best = neg
                for kernel_index in range(5):
                    destination_position = (
                        source_position + offsets[kernel_index]
                    )
                    if 0 <= destination_position < position_count:
                        value = (
                            position_log_kernel[kernel_index]
                            + emission_lambda
                            * emission[time_index, destination_position]
                            + beta_next[
                                destination_position, destination_rate
                            ]
                        )
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        destination_position = (
                            source_position + offsets[kernel_index]
                        )
                        if 0 <= destination_position < position_count:
                            total += np.exp(
                                position_log_kernel[kernel_index]
                                + emission_lambda
                                * emission[
                                    time_index, destination_position
                                ]
                                + beta_next[
                                    destination_position, destination_rate
                                ]
                                - best
                            )
                    beta_position[
                        source_position, destination_rate
                    ] = np.float32(best + np.log(total))
                else:
                    beta_position[source_position, destination_rate] = neg

        rate_log_kernel = rate_log_kernels[time_index]
        for position_index in prange(position_count):
            for source_rate in range(rate_count):
                best = neg
                for destination_rate in range(rate_count):
                    transition_log = rate_log_kernel[
                        source_rate, destination_rate
                    ]
                    if not np.isfinite(transition_log):
                        continue
                    value = (
                        transition_log
                        + beta_position[
                            position_index, destination_rate
                        ]
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for destination_rate in range(rate_count):
                        transition_log = rate_log_kernel[
                            source_rate, destination_rate
                        ]
                        if not np.isfinite(transition_log):
                            continue
                        total += np.exp(
                            transition_log
                            + beta_position[
                                position_index, destination_rate
                            ]
                            - best
                        )
                    beta_current[
                        position_index, source_rate
                    ] = np.float32(best + np.log(total))
                else:
                    beta_current[position_index, source_rate] = neg

        values = alpha[time_index - 1] + beta_current
        best = np.max(values)
        total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                total += math.exp(
                    float(values[position_index, rate_index])
                    - float(best)
                )
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                probability = (
                    math.exp(
                        float(values[position_index, rate_index])
                        - float(best)
                    )
                    / total
                )
                posterior_position[
                    time_index - 1, position_index
                ] += probability
                posterior_rate[time_index - 1, rate_index] += probability
                beta_next[position_index, rate_index] = beta_current[
                    position_index, rate_index
                ]

    maximum_posterior_normalization_error = 0.0
    for time_index in range(time_count):
        maximum_posterior_normalization_error = max(
            maximum_posterior_normalization_error,
            abs(np.sum(posterior_position[time_index]) - 1.0),
            abs(np.sum(posterior_rate[time_index]) - 1.0),
        )
    return (
        posterior_position,
        posterior_rate,
        predictive_rate_mean,
        filtered_rate_mean,
        log_likelihood,
        max(
            maximum_forward_normalization_error,
            maximum_posterior_normalization_error,
        ),
    )


def run_full_support_ou_hmm(
    prepared: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    delta_md = np.asarray(prepared["dm"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    rate_log_kernels = precompute_full_support_ou_log_kernels(
        delta_md,
        rates,
        float(fixed["sig_r"]),
        float(fixed["momentum"]),
    )
    (
        posterior_position,
        posterior_rate,
        predictive_rate_mean,
        filtered_rate_mean,
        log_likelihood,
        maximum_normalization_error,
    ) = _hmm2_full_support_ou(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        delta_md,
        np.asarray(prepared["dz"], dtype=np.float64),
        float(fixed["position_grid_step_ft"]),
        rates,
        rate_log_kernels,
        float(fixed["sig_p"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        float(fixed["emission_lambda"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = posterior_position @ grid
    posterior_variance = (
        posterior_position @ (grid**2) - posterior_mean**2
    )
    posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0))
    posterior_rate_mean = posterior_rate @ rates
    posterior_rate_variance = (
        posterior_rate @ (rates**2) - posterior_rate_mean**2
    )
    posterior_rate_std = np.sqrt(
        np.maximum(posterior_rate_variance, 0.0)
    )
    posterior_rate_edge_mass = (
        posterior_rate[:, 0] + posterior_rate[:, -1]
    )
    kernel_audit = ou_kernel_numeric_audit(
        rate_log_kernels,
        delta_md,
        rates,
        float(fixed["sig_r"]),
        float(fixed["momentum"]),
    )
    transition_kernel_sha256 = array_bundle_sha256(
        delta_md=delta_md,
        rates=rates,
        rate_log_kernels=rate_log_kernels,
    )
    prediction_sha256 = array_bundle_sha256(
        posterior_mean=posterior_mean.astype(np.float32),
        posterior_std=posterior_std.astype(np.float32),
    )
    diagnostic_sha256 = array_bundle_sha256(
        predictive_rate_mean=predictive_rate_mean.astype(np.float32),
        filtered_rate_mean=filtered_rate_mean.astype(np.float32),
        posterior_rate_mean=posterior_rate_mean.astype(np.float32),
        posterior_rate_std=posterior_rate_std.astype(np.float32),
        posterior_rate_edge_mass=posterior_rate_edge_mass.astype(np.float32),
    )
    return {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "predictive_rate_mean": predictive_rate_mean,
        "filtered_rate_mean": filtered_rate_mean,
        "posterior_rate_mean": posterior_rate_mean,
        "posterior_rate_std": posterior_rate_std,
        "posterior_rate_edge_mass": posterior_rate_edge_mass,
        "log_likelihood": float(log_likelihood),
        "maximum_normalization_error": float(
            maximum_normalization_error
        ),
        "kernel_audit": kernel_audit,
        "transition_kernel_sha256": transition_kernel_sha256,
        "prediction_sha256": prediction_sha256,
        "diagnostic_sha256": diagnostic_sha256,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def dense_full_support_reference(
    emission: np.ndarray,
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    position_step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    start_position_index: float,
    start_sigma: float,
    initial_rate: float,
    initial_rate_sigma: float,
    momentum: float,
) -> np.ndarray:
    emission = np.asarray(emission, dtype=np.float64)
    delta_md = np.asarray(delta_md, dtype=np.float64)
    delta_z = np.asarray(delta_z, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    time_count, position_count = emission.shape
    rate_count = len(rates)
    state_count = position_count * rate_count
    rate_kernels = np.exp(
        precompute_full_support_ou_log_kernels(
            delta_md,
            rates,
            sig_r,
            momentum,
        )
    )
    transitions: list[np.ndarray] = []
    for time_index in range(time_count):
        transition = np.zeros(
            (state_count, state_count),
            dtype=np.float64,
        )
        for source_position in range(position_count):
            for source_rate in range(rate_count):
                source_state = (
                    source_position * rate_count + source_rate
                )
                for destination_rate in range(rate_count):
                    rate_probability = rate_kernels[
                        time_index, source_rate, destination_rate
                    ]
                    if rate_probability <= 0.0:
                        continue
                    shift = (
                        rates[destination_rate] * delta_md[time_index]
                        - delta_z[time_index]
                    )
                    offsets, position_probabilities = (
                        parent_position_kernel_probabilities(
                            shift,
                            position_step,
                            sig_p,
                        )
                    )
                    for offset, position_probability in zip(
                        offsets,
                        position_probabilities,
                        strict=True,
                    ):
                        destination_position = (
                            source_position + int(offset)
                        )
                        if 0 <= destination_position < position_count:
                            destination_state = (
                                destination_position * rate_count
                                + destination_rate
                            )
                            transition[
                                source_state, destination_state
                            ] += (
                                rate_probability
                                * float(position_probability)
                            )
        transitions.append(transition)

    initial = np.empty(
        (position_count, rate_count),
        dtype=np.float64,
    )
    for position_index in range(position_count):
        position_log = -0.5 * (
            (
                (position_index - start_position_index) * position_step
            )
            / start_sigma
        ) ** 2
        for rate_index, rate in enumerate(rates):
            initial[position_index, rate_index] = math.exp(
                position_log
                - 0.5
                * ((rate - initial_rate) / initial_rate_sigma) ** 2
            )
    previous = initial.reshape(-1)
    forward = np.empty((time_count, state_count), dtype=np.float64)
    emission_probability = np.exp(emission)
    for time_index in range(time_count):
        current = previous @ transitions[time_index]
        current *= np.repeat(
            emission_probability[time_index],
            rate_count,
        )
        current /= current.sum()
        forward[time_index] = current
        previous = current

    backward = np.ones((time_count, state_count), dtype=np.float64)
    for time_index in range(time_count - 1, 0, -1):
        weighted_next = backward[time_index] * np.repeat(
            emission_probability[time_index],
            rate_count,
        )
        values = transitions[time_index] @ weighted_next
        values /= values.sum()
        backward[time_index - 1] = values
    posterior_position = np.empty(
        (time_count, position_count),
        dtype=np.float64,
    )
    for time_index in range(time_count):
        posterior = forward[time_index] * backward[time_index]
        posterior /= posterior.sum()
        posterior_position[time_index] = posterior.reshape(
            position_count,
            rate_count,
        ).sum(axis=1)
    return posterior_position


def brute_force_posterior_contract(
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    rows = 4
    positions = 7
    rates = np.linspace(-0.04, 0.04, 5, dtype=np.float64)
    grid = 12_000.0 + np.arange(positions, dtype=np.float64) * float(
        fixed["position_grid_step_ft"]
    )
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.18 * math.sin(index)) / 0.45) ** 2
            for index in range(rows)
        ]
    ).astype(np.float32)
    prepared = {
        "emission_ll": emission,
        "dm": np.asarray([1.0, 2.5, 7.0, 13.0], dtype=np.float64),
        "dz": np.asarray([0.1, -0.2, 0.3, 0.0], dtype=np.float64),
        "grid": grid,
        "rates": rates,
        "start_p": 3.0,
        "r0": 0.01,
    }
    observed = run_full_support_ou_hmm(prepared, fixed)
    reference_position = dense_full_support_reference(
        emission,
        prepared["dm"],
        prepared["dz"],
        float(fixed["position_grid_step_ft"]),
        rates,
        float(fixed["sig_r"]),
        float(fixed["sig_p"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        float(fixed["momentum"]),
    )
    reference_mean = reference_position @ grid
    maximum_error = float(
        np.max(
            np.abs(
                np.asarray(observed["posterior_mean"], dtype=np.float64)
                - reference_mean
            )
        )
    )
    return {
        "posterior_prediction_max_abs_error": maximum_error,
        "pass": bool(maximum_error <= 1.0e-6),
    }
# %% [markdown]
# ## 7. Target-free kernel, diagnostic, and prediction freeze

# %%
@dataclass
class FrozenWell:
    well: str
    eval_id: np.ndarray
    row_idx: np.ndarray
    raw_gr_missing: np.ndarray
    parent_prediction: np.ndarray
    candidate_prediction: np.ndarray
    candidate_posterior_std: np.ndarray
    predictive_rate_mean: np.ndarray
    filtered_rate_mean: np.ndarray
    posterior_rate_mean: np.ndarray
    posterior_rate_std: np.ndarray
    posterior_rate_edge_mass: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    prefix_rows: int
    transition_kernel_sha256: str
    prediction_sha256: str
    diagnostic_sha256: str
    kernel_audit: dict[str, float]
    maximum_normalization_error: float
    log_likelihood: float
    elapsed_seconds: float
    role: str = ""
    fold: int = -1


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    saved_parent: pd.DataFrame,
    fixed: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, fixed)
    decoded = run_full_support_ou_hmm(prepared, fixed)
    parent = saved_parent.sort_values(
        "row_idx", kind="mergesort"
    ).reset_index(drop=True)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    eval_id = parent_cache_ids_for_rows(well, row_idx)
    if not np.array_equal(
        parent["row_idx"].to_numpy(np.int64),
        row_idx,
    ):
        raise ValueError(f"{well}: saved parent row index does not align")
    if not np.array_equal(
        parent["id"].astype(str).to_numpy(),
        eval_id,
    ):
        raise ValueError(f"{well}: saved parent id does not align")
    frozen = FrozenWell(
        well=str(well),
        eval_id=eval_id,
        row_idx=row_idx,
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        parent_prediction=parent["parent_prediction"].to_numpy(np.float64),
        candidate_prediction=np.asarray(
            decoded["posterior_mean"], dtype=np.float64
        ),
        candidate_posterior_std=np.asarray(
            decoded["posterior_std"], dtype=np.float64
        ),
        predictive_rate_mean=np.asarray(
            decoded["predictive_rate_mean"], dtype=np.float64
        ),
        filtered_rate_mean=np.asarray(
            decoded["filtered_rate_mean"], dtype=np.float64
        ),
        posterior_rate_mean=np.asarray(
            decoded["posterior_rate_mean"], dtype=np.float64
        ),
        posterior_rate_std=np.asarray(
            decoded["posterior_rate_std"], dtype=np.float64
        ),
        posterior_rate_edge_mass=np.asarray(
            decoded["posterior_rate_edge_mass"], dtype=np.float64
        ),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        prefix_rows=int(prepared["prefix_rows"]),
        transition_kernel_sha256=str(
            decoded["transition_kernel_sha256"]
        ),
        prediction_sha256=str(decoded["prediction_sha256"]),
        diagnostic_sha256=str(decoded["diagnostic_sha256"]),
        kernel_audit=dict(decoded["kernel_audit"]),
        maximum_normalization_error=float(
            decoded["maximum_normalization_error"]
        ),
        log_likelihood=float(decoded["log_likelihood"]),
        elapsed_seconds=float(decoded["elapsed_seconds"]),
    )
    ledger.freeze(
        well,
        transition_kernel_sha256=frozen.transition_kernel_sha256,
        prediction_sha256=frozen.prediction_sha256,
        diagnostic_sha256=frozen.diagnostic_sha256,
    )
    return frozen


def attach_late_identity(
    frozen_wells: Sequence[FrozenWell],
    manifest: pd.DataFrame,
) -> None:
    identity = manifest.set_index("well")
    for item in frozen_wells:
        if item.well not in identity.index:
            raise ValueError(f"{item.well}: missing late fixed32 identity")
        row = identity.loc[item.well]
        item.role = str(row["role"])
        item.fold = int(row["fold"])
        if len(item.row_idx) != int(row["suffix_rows"]):
            raise ValueError(f"{item.well}: suffix rows changed")
        if item.prefix_rows != int(row["prefix_rows"]):
            raise ValueError(f"{item.well}: prefix rows changed")


def prediction_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    pieces = [
        pd.DataFrame(
            {
                "id": item.eval_id,
                "well": item.well,
                "row_idx": item.row_idx,
                "parent_prediction": item.parent_prediction,
                "candidate_prediction": item.candidate_prediction,
                "candidate_posterior_std": item.candidate_posterior_std,
            }
        )
        for item in frozen_wells
    ]
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


def diagnostic_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    pieces = [
        pd.DataFrame(
            {
                "well": item.well,
                "row_idx": item.row_idx,
                "suffix_offset": np.arange(
                    len(item.row_idx), dtype=np.int64
                ),
                "raw_gr_missing": item.raw_gr_missing,
                "predictive_rate_mean": item.predictive_rate_mean,
                "filtered_rate_mean": item.filtered_rate_mean,
                "posterior_rate_mean": item.posterior_rate_mean,
                "posterior_rate_std": item.posterior_rate_std,
                "posterior_rate_edge_mass": item.posterior_rate_edge_mass,
            }
        )
        for item in frozen_wells
    ]
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


def kernel_audit_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    rows = []
    for item in frozen_wells:
        rows.append(
            {
                "well": item.well,
                "transition_kernel_sha256": (
                    item.transition_kernel_sha256
                ),
                **item.kernel_audit,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "well", kind="mergesort"
    ).reset_index(drop=True)


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
# ## 8. Truth-late mechanism and safety readout
#
# Suffix truth, role/fold identity, persistent episode boundaries, exp408 cause
# labels, and the truth-centered exp408 parent row ledger are opened only after
# all 32 kernel, diagnostic, and prediction SHA values have been frozen.

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
    if not np.array_equal(
        suffix.index.to_numpy(np.int64),
        frozen.row_idx,
    ):
        raise ValueError(f"{frozen.well}: truth row index changed after freeze")
    suffix["id"] = parent_cache_ids_for_rows(
        frozen.well,
        suffix.index.to_numpy(np.int64),
    )
    return suffix.reset_index(names="row_idx")


def physical_true_interval_rate(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> np.ndarray:
    tvt = truth["TVT"].to_numpy(np.float64)
    md = truth["MD"].to_numpy(np.float64)
    z = truth["Z"].to_numpy(np.float64)
    delta_tvt = np.diff(
        np.concatenate([[frozen.last_known_tvt], tvt])
    )
    delta_md = np.maximum(
        np.diff(np.concatenate([[frozen.last_known_md], md])),
        1.0,
    )
    delta_z = np.diff(
        np.concatenate([[frozen.last_known_z], z])
    )
    return (delta_tvt + delta_z) / delta_md


def zero_directed_under_response_mask(
    true_rate: np.ndarray,
    decoded_rate: np.ndarray,
    *,
    moving_epsilon: float = 1.0e-12,
) -> np.ndarray:
    true_values = np.asarray(true_rate, dtype=np.float64)
    decoded_values = np.asarray(decoded_rate, dtype=np.float64)
    if true_values.shape != decoded_values.shape:
        raise ValueError("true and decoded rate shapes differ")
    return (
        (np.abs(true_values) > moving_epsilon)
        & (true_values * decoded_values >= 0.0)
        & (np.abs(decoded_values) < np.abs(true_values))
    )


def well_truth_late_metrics(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> dict[str, Any]:
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
        "parent_sse": float(np.sum(parent_error**2)),
        "candidate_sse": float(np.sum(candidate_error**2)),
        "parent_rmse_ft": parent_rmse,
        "candidate_rmse_ft": candidate_rmse,
        "rmse_delta_ft": candidate_rmse - parent_rmse,
        "raw_gr_missing_fraction": float(
            np.mean(frozen.raw_gr_missing)
        ),
        "maximum_normalization_error": (
            frozen.maximum_normalization_error
        ),
        "hmm_elapsed_seconds": frozen.elapsed_seconds,
        "transition_kernel_sha256": frozen.transition_kernel_sha256,
        "prediction_sha256": frozen.prediction_sha256,
        "diagnostic_sha256": frozen.diagnostic_sha256,
    }


def load_persistent_episodes_after_all_freeze(
    config: Mapping[str, Any],
    persistent_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(
        str(spec["filename"]),
        str(spec["local"]),
    )
    observed = sha256_file(path)
    expected = str(spec["expected_sha256"])
    if observed != expected:
        raise ValueError(
            f"persistent episode SHA changed: {observed} != {expected}"
        )
    frame = pd.read_csv(
        path,
        dtype={"well": str, "episode_id": str},
    )
    frame = frame.loc[frame["well"].isin(persistent_wells)].copy()
    ledger.record_episode_late(len(frame))
    required = {
        "episode_id",
        "well",
        "start_row_idx",
        "end_row_idx_exclusive",
        "rows",
    }
    if not required.issubset(frame.columns):
        raise ValueError("persistent episode schema changed")
    if frame.empty or frame["well"].nunique() != len(persistent_wells):
        raise ValueError("selected persistent wells are missing episode rows")
    return frame.sort_values(
        ["well", "start_row_idx"], kind="mergesort"
    ).reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "selected_rows": len(frame),
    }


def load_episode_causes_after_all_freeze(
    config: Mapping[str, Any],
    selected_episode_ids: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp408_episode_causes")
    path = resolve_bootstrap_asset(
        str(spec["filename"]),
        str(spec["local"]),
    )
    observed = sha256_file(path)
    expected = str(spec["expected_sha256"])
    if observed != expected:
        raise ValueError(
            f"exp408 cause SHA changed: {observed} != {expected}"
        )
    frame = pd.read_csv(
        path,
        usecols=["episode_id", "well", "fold", "cause"],
        dtype={"well": str, "episode_id": str},
    )
    frame = frame.loc[
        frame["episode_id"].isin(selected_episode_ids)
    ].copy()
    ledger.record_cause_late(len(frame))
    if frame["episode_id"].duplicated().any():
        raise ValueError("exp408 cause rows are not unique")
    if set(frame["episode_id"]) != selected_episode_ids:
        raise ValueError("exp408 cause coverage changed")
    return frame, {
        "path": str(path),
        "sha256": observed,
        "selected_rows": len(frame),
    }


def load_exp408_parent_row_ledger_after_all_freeze(
    config: Mapping[str, Any],
    persistent_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp408_parent_row_ledger")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(value) for value in spec["candidates"]],
        patterns=[str(value) for value in spec["patterns"]],
    )
    observed = sha256_decompressed_csv(path)
    expected = str(spec["expected_decompressed_sha256"])
    if observed != expected:
        raise ValueError(
            f"exp408 row-ledger SHA changed: {observed} != {expected}"
        )
    usecols = [
        "well",
        "row_idx",
        "tvt_true",
        "true_rate",
        "filtered__rate_mean",
        "mean_error_ft",
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"well": str},
        chunksize=100_000,
    ):
        selected = chunk.loc[
            chunk["well"].isin(persistent_wells)
        ].copy()
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("exp408 row ledger has no fixed32 persistent rows")
    frame = pd.concat(pieces, ignore_index=True)
    ledger.record_truth_late(len(frame))
    if frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("exp408 selected row ledger keys are not unique")
    return frame.sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True), {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": observed,
        "selected_rows": len(frame),
        "selected_wells": frame["well"].nunique(),
    }


def episode_truth_late_readout(
    episodes: pd.DataFrame,
    causes: pd.DataFrame,
    frozen_by_well: Mapping[str, FrozenWell],
    truth_by_well: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    causes_by_episode = causes.set_index("episode_id")
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        well = str(episode.well)
        frozen = frozen_by_well[well]
        truth = truth_by_well[well]
        start = int(episode.start_row_idx)
        end = int(episode.end_row_idx_exclusive)
        mask = (frozen.row_idx >= start) & (frozen.row_idx < end)
        offsets = np.flatnonzero(mask)
        if len(offsets) != int(episode.rows):
            raise ValueError(
                f"{episode.episode_id}: episode row coverage changed"
            )
        actual = truth["TVT"].to_numpy(np.float64)[offsets]
        parent_error = frozen.parent_prediction[offsets] - actual
        candidate_error = frozen.candidate_prediction[offsets] - actual
        cause_row = causes_by_episode.loc[str(episode.episode_id)]
        if int(cause_row["fold"]) != frozen.fold:
            raise ValueError(
                f"{episode.episode_id}: exp408/manifest fold changed"
            )
        parent_sse = float(np.sum(parent_error**2))
        candidate_sse = float(np.sum(candidate_error**2))
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": well,
                "fold": frozen.fold,
                "cause": str(cause_row["cause"]),
                "start_row_idx": start,
                "end_row_idx_exclusive": end,
                "rows": len(offsets),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "sse_reduction_fraction": (
                    1.0 - candidate_sse / parent_sse
                    if parent_sse > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fold", "well", "start_row_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def under_response_truth_late_readout(
    parent_rows: pd.DataFrame,
    frozen_by_well: Mapping[str, FrozenWell],
    truth_by_well: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for well in sorted(parent_rows["well"].astype(str).unique()):
        frozen = frozen_by_well[well]
        truth = truth_by_well[well]
        true_rate = physical_true_interval_rate(frozen, truth)
        candidate = pd.DataFrame(
            {
                "well": well,
                "row_idx": frozen.row_idx,
                "candidate_prediction": frozen.candidate_prediction,
                "candidate_filtered_rate_mean": (
                    frozen.filtered_rate_mean
                ),
                "candidate_true_rate": true_rate,
                "candidate_truth_tvt": truth["TVT"].to_numpy(np.float64),
            }
        )
        selected_parent = parent_rows.loc[
            parent_rows["well"].eq(well)
        ].copy()
        joined = selected_parent.merge(
            candidate,
            on=["well", "row_idx"],
            how="left",
            validate="one_to_one",
        )
        if joined["candidate_prediction"].isna().any():
            raise ValueError(f"{well}: candidate under-response rows missing")
        np.testing.assert_allclose(
            joined["true_rate"].to_numpy(np.float64),
            joined["candidate_true_rate"].to_numpy(np.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            joined["tvt_true"].to_numpy(np.float64),
            joined["candidate_truth_tvt"].to_numpy(np.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
        parent_mask = zero_directed_under_response_mask(
            joined["true_rate"].to_numpy(np.float64),
            joined["filtered__rate_mean"].to_numpy(np.float64),
        )
        candidate_mask = zero_directed_under_response_mask(
            joined["candidate_true_rate"].to_numpy(np.float64),
            joined["candidate_filtered_rate_mean"].to_numpy(np.float64),
        )
        joined["parent_zero_directed_under_response"] = parent_mask
        joined["candidate_zero_directed_under_response"] = candidate_mask
        joined["parent_squared_error"] = (
            joined["mean_error_ft"].to_numpy(np.float64) ** 2
        )
        joined["candidate_squared_error"] = (
            joined["candidate_prediction"].to_numpy(np.float64)
            - joined["candidate_truth_tvt"].to_numpy(np.float64)
        ) ** 2
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 9. Technical and mechanism gates

# %%
def safe_fraction(
    numerator: float | int,
    denominator: float | int,
) -> float:
    return float(numerator / denominator) if denominator else math.nan


def sse_reduction(
    parent_sse: float,
    candidate_sse: float,
) -> float:
    return (
        float((parent_sse - candidate_sse) / parent_sse)
        if parent_sse > 0.0
        else math.nan
    )


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    brute_force_contract: Mapping[str, Any],
    position_parity: Mapping[str, Any],
    prediction_artifact: Mapping[str, Any],
    diagnostic_artifact: Mapping[str, Any],
    kernel_audit_artifact: Mapping[str, Any],
    episode_readout: pd.DataFrame,
    under_response_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(
        config, "gates.stage0_fixed32.technical"
    )
    mechanism_config = get_nested(
        config, "gates.stage0_fixed32.mechanism"
    )
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    arrays = []
    for item in frozen_wells:
        arrays.extend(
            [
                item.candidate_prediction,
                item.candidate_posterior_std,
                item.predictive_rate_mean,
                item.filtered_rate_mean,
                item.posterior_rate_mean,
                item.posterior_rate_std,
                item.posterior_rate_edge_mass,
            ]
        )
    finite_count = sum(
        int(np.isfinite(np.asarray(values)).sum()) for values in arrays
    )
    value_count = sum(int(np.asarray(values).size) for values in arrays)
    finite_coverage = safe_fraction(finite_count, value_count)
    maximum_normalization_error = max(
        item.maximum_normalization_error for item in frozen_wells
    )
    maximum_mass_error = max(
        item.kernel_audit[
            "analytic_in_support_mass_max_abs_error"
        ]
        for item in frozen_wells
    )
    maximum_mean_error = max(
        item.kernel_audit[
            "interior_conditional_mean_max_abs_error"
        ]
        for item in frozen_wells
    )
    maximum_variance_error = max(
        item.kernel_audit[
            "interior_conditional_variance_max_abs_error"
        ]
        for item in frozen_wells
    )
    runtime_projection = float(elapsed_seconds * 773.0 / 32.0)
    persistent_metrics = well_metrics.loc[
        well_metrics["role"].eq("persistent")
    ]
    control_metrics = well_metrics.loc[
        well_metrics["role"].eq("control")
    ]

    technical = {
        "expected_wells": len(frozen_wells)
        == int(technical_config["expected_wells"]),
        "expected_rows": total_rows
        == int(technical_config["expected_rows"]),
        "expected_roles": (
            manifest["role"].value_counts().to_dict()
            == {
                "persistent": int(
                    technical_config["expected_persistent_wells"]
                ),
                "control": int(
                    technical_config["expected_control_wells"]
                ),
            }
        ),
        "expected_folds": manifest["fold"].nunique()
        == int(technical_config["expected_folds"]),
        "finite_coverage": finite_coverage
        >= float(technical_config["finite_coverage_min"]),
        "analytic_in_support_mass": maximum_mass_error
        <= float(
            technical_config[
                "analytic_in_support_mass_max_abs_error"
            ]
        ),
        "interior_conditional_mean": maximum_mean_error
        <= float(
            technical_config[
                "interior_conditional_mean_max_abs_error"
            ]
        ),
        "interior_conditional_variance": maximum_variance_error
        <= float(
            technical_config[
                "interior_conditional_variance_max_abs_error"
            ]
        ),
        "posterior_normalization": maximum_normalization_error
        <= float(
            technical_config["posterior_normalization_max_error"]
        ),
        "brute_force_posterior_prediction": bool(
            brute_force_contract["pass"]
        )
        and float(
            brute_force_contract[
                "posterior_prediction_max_abs_error"
            ]
        )
        <= float(
            technical_config[
                "brute_force_posterior_prediction_max_abs_error"
            ]
        ),
        "parent_position_kernel_parity": bool(position_parity["pass"])
        and float(position_parity["maximum_absolute_error"])
        <= float(
            technical_config[
                "no_intervention_parent_position_kernel_parity_max_abs_error"
            ]
        ),
        "truth_role_fold_episode_reads_before_freeze": (
            ledger.forbidden_reads_before_all_freeze
            <= int(
                technical_config[
                    "truth_role_fold_episode_reads_before_freeze_max"
                ]
            )
        ),
        "prediction_readback_sha": (
            prediction_artifact["logical_sha256"]
            == prediction_artifact["readback_logical_sha256"]
        ),
        "diagnostic_readback_sha": (
            diagnostic_artifact["logical_sha256"]
            == diagnostic_artifact["readback_logical_sha256"]
        ),
        "kernel_audit_readback_sha": (
            kernel_audit_artifact["logical_sha256"]
            == kernel_audit_artifact["readback_logical_sha256"]
        ),
        "runtime_projection": runtime_projection
        <= float(
            technical_config[
                "projected_stage1_runtime_seconds_max"
            ]
        ),
        "peak_rss": peak_rss_gb()
        <= float(technical_config["peak_rss_gb_max"]),
    }

    parent_under_sse = float(
        under_response_readout.loc[
            under_response_readout[
                "parent_zero_directed_under_response"
            ],
            "parent_squared_error",
        ].sum()
    )
    parent_total_sse = float(
        under_response_readout["parent_squared_error"].sum()
    )
    candidate_under_sse = float(
        under_response_readout.loc[
            under_response_readout[
                "candidate_zero_directed_under_response"
            ],
            "candidate_squared_error",
        ].sum()
    )
    candidate_total_sse = float(
        under_response_readout["candidate_squared_error"].sum()
    )
    parent_under_share = safe_fraction(
        parent_under_sse, parent_total_sse
    )
    candidate_under_share = safe_fraction(
        candidate_under_sse, candidate_total_sse
    )
    under_share_reduction = parent_under_share - candidate_under_share

    parent_episode_sse = float(episode_readout["parent_sse"].sum())
    candidate_episode_sse = float(
        episode_readout["candidate_sse"].sum()
    )
    persistent_reduction = sse_reduction(
        parent_episode_sse,
        candidate_episode_sse,
    )
    forward_cause = str(
        get_nested(config, "data.exp408_episode_causes.forward_cause")
    )
    forward = episode_readout.loc[
        episode_readout["cause"].eq(forward_cause)
    ]
    forward_reduction = sse_reduction(
        float(forward["parent_sse"].sum()),
        float(forward["candidate_sse"].sum()),
    )
    persistent_improved_wells = int(
        (persistent_metrics["rmse_delta_ft"] < 0.0).sum()
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        frame = episode_readout.loc[episode_readout["fold"].eq(fold)]
        parent_sse = float(frame["parent_sse"].sum())
        candidate_sse = float(frame["candidate_sse"].sum())
        reduction = sse_reduction(parent_sse, candidate_sse)
        fold_rows.append(
            {
                "fold": fold,
                "episodes": len(frame),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "sse_reduction_fraction": reduction,
                "improving": bool(
                    math.isfinite(reduction) and reduction > 0.0
                ),
            }
        )
    persistent_improving_folds = sum(
        row["improving"] for row in fold_rows
    )
    control_parent_sse = float(control_metrics["parent_sse"].sum())
    control_candidate_sse = float(
        control_metrics["candidate_sse"].sum()
    )
    control_rows = int(control_metrics["rows"].sum())
    control_delta = (
        math.sqrt(control_candidate_sse / control_rows)
        - math.sqrt(control_parent_sse / control_rows)
    )
    control_p95 = float(
        np.quantile(
            control_metrics["rmse_delta_ft"].to_numpy(np.float64),
            0.95,
        )
    )
    mechanism = {
        "zero_directed_under_response_sse_share_reduction": (
            math.isfinite(under_share_reduction)
            and under_share_reduction
            >= float(
                mechanism_config[
                    "zero_directed_under_response_sse_share_reduction_min_absolute"
                ]
            )
        ),
        "forward_cause_episode_sse_reduction": (
            math.isfinite(forward_reduction)
            and forward_reduction
            >= float(
                mechanism_config[
                    "forward_cause_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_episode_sse_reduction": (
            math.isfinite(persistent_reduction)
            and persistent_reduction
            >= float(
                mechanism_config[
                    "persistent_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_improved_wells": persistent_improved_wells
        >= int(mechanism_config["persistent_improved_wells_min"]),
        "persistent_improving_folds": persistent_improving_folds
        >= int(mechanism_config["persistent_improving_folds_min"]),
        "matched_control_pooled_rmse_delta": control_delta
        <= float(
            mechanism_config[
                "matched_control_pooled_rmse_delta_max_ft"
            ]
        ),
        "matched_control_by_well_delta_p95": control_p95
        <= float(
            mechanism_config[
                "matched_control_by_well_delta_p95_max_ft"
            ]
        ),
    }
    diagnostics = {
        "total_rows": total_rows,
        "finite_coverage": finite_coverage,
        "maximum_normalization_error": maximum_normalization_error,
        "analytic_in_support_mass_max_abs_error": maximum_mass_error,
        "interior_conditional_mean_max_abs_error": maximum_mean_error,
        "interior_conditional_variance_max_abs_error": (
            maximum_variance_error
        ),
        "brute_force_posterior_prediction_max_abs_error": (
            brute_force_contract[
                "posterior_prediction_max_abs_error"
            ]
        ),
        "position_kernel_parity_max_abs_error": position_parity[
            "maximum_absolute_error"
        ],
        "parent_zero_directed_under_response_sse_share": (
            parent_under_share
        ),
        "candidate_zero_directed_under_response_sse_share": (
            candidate_under_share
        ),
        "zero_directed_under_response_sse_share_reduction_absolute": (
            under_share_reduction
        ),
        "forward_cause": forward_cause,
        "forward_cause_episodes": len(forward),
        "forward_cause_episode_sse_reduction_fraction": (
            forward_reduction
        ),
        "persistent_episode_sse_reduction_fraction": (
            persistent_reduction
        ),
        "persistent_improved_wells": persistent_improved_wells,
        "persistent_episode_by_fold": fold_rows,
        "persistent_improving_folds": persistent_improving_folds,
        "matched_control_pooled_rmse_delta_ft": control_delta,
        "matched_control_by_well_delta_p95_ft": control_p95,
        "runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
        "fixed32_is_mechanism_only_not_cv_or_promotion_evidence": True,
    }
    all_pass = bool(all(technical.values()) and all(mechanism.values()))
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "stage1_eligible_pending_separate_user_approval": all_pass,
        "fail_action": (
            None
            if all_pass
            else get_nested(config, "gates.stage0_fixed32.fail_action")
        ),
    }
# %% [markdown]
# ## 10. Guarded Kaggle CPU orchestration

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP441_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp441 Stage 0 must run on Kaggle CPU")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha = hashlib.sha256(
        stable_json_bytes(scientific_contract)
    ).hexdigest()
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    fixed = get_nested(config, "model.fixed_from_exp209")
    brute_force_contract = brute_force_posterior_contract(fixed)
    position_parity = position_kernel_parity_contract(fixed)
    if not brute_force_contract["pass"]:
        raise RuntimeError(
            f"brute-force HMM contract failed: {brute_force_contract}"
        )
    if not position_parity["pass"]:
        raise RuntimeError(
            f"parent position-kernel parity failed: {position_parity}"
        )

    ledger = LeakageLedger(expected_wells=32)
    wells, scope_input = load_fixed32_scope(config, ledger)
    parent, parent_input = load_saved_parent_predictions(
        config,
        set(wells),
        ledger,
    )
    raw_dir = train_data_dir(config)
    parent_groups = parent.groupby("well", sort=False).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(
        get_nested(config, "runtime.hard_runtime_limit_seconds")
    )
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    for well_index, well in enumerate(wells, start=1):
        if well not in parent_groups:
            raise ValueError(f"{well}: saved parent rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            saved_parent=parent.iloc[parent_groups[well]].copy(),
            fixed=fixed,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(
                f"Stage 0 runtime hard guard exceeded: {elapsed}"
            )
        if peak_rss_gb() > hard_rss:
            raise MemoryError(
                f"Stage 0 RSS hard guard exceeded: {peak_rss_gb()}"
            )
        print(
            json.dumps(
                {
                    "event": "exp441_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "transition_kernel_sha256": (
                        frozen.transition_kernel_sha256
                    ),
                    "hmm_seconds": frozen.elapsed_seconds,
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
    diagnostics = diagnostic_frame(frozen_wells)
    kernel_audits = kernel_audit_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    diagnostic_artifact = write_deterministic_gzip_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_target_free_rate_diagnostics.csv.gz",
        diagnostics,
    )
    kernel_audit_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_ou_kernel_audit.csv.gz",
        kernel_audits,
    )
    for label, artifact in (
        ("prediction", prediction_artifact),
        ("diagnostic", diagnostic_artifact),
        ("kernel_audit", kernel_audit_artifact),
    ):
        if artifact["logical_sha256"] != artifact[
            "readback_logical_sha256"
        ]:
            raise RuntimeError(f"{label} readback SHA mismatch")
    transition_manifest = {
        "combined_transition_kernel_sha256": combined_well_sha(
            frozen_wells,
            "transition_kernel_sha256",
        ),
        "per_well": [
            {
                "well": item.well,
                "transition_kernel_sha256": (
                    item.transition_kernel_sha256
                ),
            }
            for item in sorted(frozen_wells, key=lambda value: value.well)
        ],
    }
    transition_manifest_artifact = write_json(
        output
        / f"{EXPERIMENT_NAME}_stage0_transition_kernel_manifest.json",
        transition_manifest,
    )

    manifest, manifest_input = load_fixed32_identity_after_all_freeze(
        config,
        ledger,
    )
    attach_late_identity(frozen_wells, manifest)
    frozen_by_well = {item.well: item for item in frozen_wells}
    truth_by_well: dict[str, pd.DataFrame] = {}
    well_rows: list[dict[str, Any]] = []
    for item in frozen_wells:
        truth = load_truth_after_all_freeze(item, raw_dir, ledger)
        truth_by_well[item.well] = truth
        well_rows.append(well_truth_late_metrics(item, truth))
    well_metrics = pd.DataFrame(well_rows).sort_values(
        ["fold", "role", "well"],
        kind="mergesort",
    )
    persistent_wells = set(
        manifest.loc[
            manifest["role"].eq("persistent"),
            "well",
        ].astype(str)
    )
    episodes, episode_input = load_persistent_episodes_after_all_freeze(
        config,
        persistent_wells,
        ledger,
    )
    selected_episode_ids = set(episodes["episode_id"].astype(str))
    causes, cause_input = load_episode_causes_after_all_freeze(
        config,
        selected_episode_ids,
        ledger,
    )
    parent_rows, row_ledger_input = (
        load_exp408_parent_row_ledger_after_all_freeze(
            config,
            persistent_wells,
            ledger,
        )
    )
    episode_readout = episode_truth_late_readout(
        episodes,
        causes,
        frozen_by_well,
        truth_by_well,
    )
    under_response_readout = under_response_truth_late_readout(
        parent_rows,
        frozen_by_well,
        truth_by_well,
    )

    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    episode_artifact = write_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_episode_truth_late_readout.csv",
        episode_readout,
    )
    under_response_artifact = write_deterministic_gzip_csv(
        output
        / (
            f"{EXPERIMENT_NAME}_stage0_zero_directed_"
            "under_response_readout.csv.gz"
        ),
        under_response_readout,
    )
    elapsed = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        manifest=manifest,
        frozen_wells=frozen_wells,
        brute_force_contract=brute_force_contract,
        position_parity=position_parity,
        prediction_artifact=prediction_artifact,
        diagnostic_artifact=diagnostic_artifact,
        kernel_audit_artifact=kernel_audit_artifact,
        episode_readout=episode_readout,
        under_response_readout=under_response_readout,
        well_metrics=well_metrics,
        ledger=ledger,
        elapsed_seconds=elapsed,
    )
    input_manifest = {
        "fixed32_scope": scope_input,
        "fixed32_identity_truth_late": manifest_input,
        "saved_exp209_control": parent_input,
        "persistent_episodes_truth_late": episode_input,
        "exp408_episode_causes_truth_late": cause_input,
        "exp408_parent_row_ledger_truth_late": row_ledger_input,
        "raw_train_dir": str(raw_dir),
        "scientific_contract_sha256": scientific_contract_sha,
        "leakage": {
            "scope_rows": ledger.scope_rows,
            "target_free_rows": ledger.target_free_rows,
            "frozen_wells": len(ledger.frozen_wells),
            "forbidden_reads_before_all_freeze": (
                ledger.forbidden_reads_before_all_freeze
            ),
            "role_fold_rows_after_all_freeze": (
                ledger.role_fold_rows_after_all_freeze
            ),
            "truth_rows_after_all_freeze": (
                ledger.truth_rows_after_all_freeze
            ),
            "episode_rows_after_all_freeze": (
                ledger.episode_rows_after_all_freeze
            ),
            "cause_rows_after_all_freeze": (
                ledger.cause_rows_after_all_freeze
            ),
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    eligible = bool(
        gates["stage1_eligible_pending_separate_user_approval"]
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0_all_gates_pass_pending_separate_stage1_approval"
            if eligible
            else "stage0_fail_closed"
        ),
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "brute_force_contract": brute_force_contract,
        "position_kernel_parity": position_parity,
        "gates": gates,
        "transition_kernel_manifest_sha256": transition_manifest[
            "combined_transition_kernel_sha256"
        ],
        "prediction_manifest_sha256": combined_well_sha(
            frozen_wells,
            "prediction_sha256",
        ),
        "diagnostic_manifest_sha256": combined_well_sha(
            frozen_wells,
            "diagnostic_sha256",
        ),
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
            "cpu_only": True,
            "numba_threads": int(
                get_nested(config, "runtime.numba_num_threads")
            ),
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "target_free_rate_diagnostics": diagnostic_artifact,
            "ou_kernel_audit": kernel_audit_artifact,
            "transition_kernel_manifest": (
                transition_manifest_artifact
            ),
            "well_metrics": well_artifact,
            "episode_truth_late_readout": episode_artifact,
            "zero_directed_under_response_readout": (
                under_response_artifact
            ),
            "input_manifest": input_artifact,
        },
        "stage1": {
            "eligible": eligible,
            "execution_approved": False,
            "requires_separate_user_approval": True,
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
        "status": summary["status"],
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "stage": "stage0_fixed32",
            "cv": None,
            "lb": None,
            "fixed32_is_mechanism_only": True,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "technical_gates": gates["technical"],
        "mechanism_gates": gates["mechanism"],
        "stage1_eligible_pending_separate_user_approval": eligible,
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# Importing this notebook remains side-effect free. Direct execution runs only
# the separately authorized fixed32 Stage 0 and keeps Stage 1/inference/submission
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
                "event": "exp441_stage0_start",
                "experiment": EXPERIMENT_NAME,
                "status": get_nested(CONFIG, "experiment.status"),
                "selected_stage": get_nested(
                    CONFIG,
                    "execution.selected_stage",
                ),
                "execution_counts": EXECUTION_COUNTS,
                "canonical_notebook_adoption_authorized": bool(
                    get_nested(
                        CONFIG,
                        "execution.canonical_notebook_adoption_authorized",
                    )
                ),
                "kaggle_package_authorized": bool(
                    get_nested(CONFIG, "execution.kaggle_package_authorized")
                ),
                "stage0_run_authorized": bool(
                    get_nested(CONFIG, "execution.stage0_run_authorized")
                ),
                "stage1_run_authorized": bool(
                    get_nested(CONFIG, "execution.stage1_run_authorized")
                ),
                "inference": bool(
                    get_nested(CONFIG, "execution.inference_authorized")
                ),
                "submission": bool(
                    get_nested(CONFIG, "execution.submission_authorized")
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    SUMMARY = run_stage0(CONFIG)
