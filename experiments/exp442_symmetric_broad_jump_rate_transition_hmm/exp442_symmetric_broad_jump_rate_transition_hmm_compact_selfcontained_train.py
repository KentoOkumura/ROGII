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
# # exp442 symmetric broad-jump rate-transition exact HMM — train
#
# This implementation keeps the complete exp209 exact-HMM contract except for
# one frozen scientific change: every source-rate row uses
# `0.99 * K_parent + 0.01 * K_broad`. `K_broad` is a target-free, two-sided
# Gaussian with rate sigma 0.02, integrated over every finite rate-bin Voronoi
# cell without boundary renormalization. The branch is analytically
# marginalized; no random jump is sampled.
#
# The current contract authorizes only the fixed32 Kaggle private CPU Stage 0.
# Stage 1, inference, and submission remain separately gated.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable contracts
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Fixed32 scope, saved parent, and target-free raw inputs
# 4. Exact exp209 input preparation
# 5. Local, broad, and mixture rate-kernel helpers
# 6. Symmetric broad-jump exact forward-backward
# 7. Target-free transition diagnostics and prediction freeze
# 8. Truth-late Stage 0 readout
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

EXPERIMENT_NAME = "exp442_symmetric_broad_jump_rate_transition_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
BROAD_VARIANT = "symmetric_broad_jump_w001_s002"
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
        raise ValueError("wrong exp442 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp442 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp442 scientific parent changed")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp442 implementation is not authorized")
    if get_nested(config, "execution.selected_stage") != "stage0_fixed32":
        raise ValueError("exp442 selected stage must remain fixed32 Stage 0")
    if not bool(
        get_nested(config, "execution.canonical_notebook_adoption_authorized", False)
    ):
        raise ValueError("canonical train notebook adoption must be authorized")
    if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
        raise ValueError("Kaggle Stage 0 packaging must be authorized")
    if not bool(get_nested(config, "execution.stage0_run_authorized", False)):
        raise ValueError("fixed32 Stage 0 must be explicitly authorized")
    if bool(get_nested(config, "execution.stage1_run_authorized", True)):
        raise ValueError("Stage 1 must remain disabled")
    if bool(get_nested(config, "execution.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "execution.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu", True)):
        raise ValueError("exp442 is CPU-only")
    if not bool(get_nested(config, "execution.run_hmm", False)):
        raise ValueError("fixed32 HMM execution must be enabled")
    if not bool(get_nested(config, "execution.create_prediction", False)):
        raise ValueError("fixed32 prediction creation must be enabled")
    if bool(get_nested(config, "execution.create_submission", True)):
        raise ValueError("submission creation must remain disabled")

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
        raise ValueError(f"exp442 execution contract changed: {observed} != {expected}")
    if bool(get_nested(config, "data.exp209_saved_control.regenerate", True)):
        raise ValueError("saved exp209 control must not be regenerated")
    if require_run_authorization:
        if not bool(get_nested(config, "execution.stage0_run_authorized", False)):
            raise RuntimeError(
                "exp442 fixed32 Stage 0 execution is not authorized"
            )
        if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
            raise RuntimeError(
                "exp442 Stage 0 execution requires separate Kaggle package approval"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("exp442 run_hmm remains fail-closed")
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
        "output": "smoothed_posterior_mean_and_std",
    }
    observed_fixed = {key: fixed.get(key) for key in expected_fixed}
    if observed_fixed != expected_fixed:
        raise ValueError(
            f"exp209 HMM contract changed: {observed_fixed} != {expected_fixed}"
        )
    candidate = dict(get_nested(config, "model.candidate_rate_transition") or {})
    expected_candidate = {
        "formula": (
            "(1-jump_weight)*parent_tridiagonal_kernel+jump_weight*broad_kernel"
        ),
        "jump_weight": 0.01,
        "broad_sigma_rate": 0.02,
        "broad_center_formula": (
            "r_source-(1-parent_momentum)*r_source*delta_MD"
        ),
        "broad_discretization": (
            "gaussian_cdf_integral_over_all_rate_bin_voronoi_cells"
        ),
        "symmetry": "target_free_two_sided",
        "outer_tail_policy": (
            "discard_outside_finite_parent_rate_support_without_renormalization"
        ),
        "branch_responsibility_audit": True,
        "nonadjacent_edge_definition": (
            "absolute_destination_minus_source_index_gt_1"
        ),
        "future_rate_horizon_rows": 32,
        "future_rate_direction_definition": (
            "median_next32_minus_median_past32_physical_interval_rate"
        ),
        "direction_agreement_weight": (
            "posterior_nonadjacent_broad_edge_mass"
        ),
        "positive_fold_definition": (
            "weighted_direction_agreement_strictly_above_0p50"
        ),
    }
    observed_candidate = {
        key: candidate.get(key) for key in expected_candidate
    }
    if observed_candidate != expected_candidate:
        raise ValueError(
            "exp442 broad-jump contract changed: "
            f"{observed_candidate} != {expected_candidate}"
        )
    variants = list(get_nested(config, "model.active_scientific_variants") or [])
    if variants != [BROAD_VARIANT]:
        raise ValueError("exp442 must contain exactly one frozen scientific candidate")
    return {
        "fixed_from_exp209": observed_fixed,
        "candidate_rate_transition": observed_candidate,
        "active_scientific_variants": variants,
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
    raise FileNotFoundError("exp442 config.yaml was not found")


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
        schedule_sha256: str,
        prediction_sha256: str,
        diagnostic_sha256: str,
    ) -> None:
        if not schedule_sha256 or not prediction_sha256 or not diagnostic_sha256:
            raise ValueError("all target-free SHA values are required before freeze")
        self.frozen_wells.add(str(well))
        self.freeze_records.append(
            {
                "well": str(well),
                "schedule_sha256": schedule_sha256,
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
# ## 5. Local, broad, and mixture rate-kernel helpers
#
# The local branch reproduces the exp209 three-destination Euler kernel,
# including its substochastic outward boundary mass. The broad branch
# integrates a Gaussian over every finite rate-bin Voronoi cell. Probability
# outside the finite parent support is discarded and is never moved to an edge
# bin. The two branches are mixed in probability space.

# %%
@njit(cache=True, nogil=True)
def parent_local_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    rate_variance_cells = (sig_r * math.sqrt(delta_md) / rate_step) ** 2
    kernel = np.zeros((rate_count, rate_count), dtype=np.float64)
    for source in range(rate_count):
        mean_move_cells = (
            -(1.0 - momentum) * rates[source] * delta_md / rate_step
        )
        p_plus = 0.5 * (rate_variance_cells + mean_move_cells)
        p_minus = 0.5 * (rate_variance_cells - mean_move_cells)
        if p_plus < 1.0e-12:
            p_plus = 1.0e-12
        if p_minus < 1.0e-12:
            p_minus = 1.0e-12
        moving_mass = p_plus + p_minus
        if moving_mass > 0.9:
            scale = 0.9 / moving_mass
            p_plus *= scale
            p_minus *= scale
        if source > 0:
            kernel[source, source - 1] = p_minus
        kernel[source, source] = 1.0 - p_plus - p_minus
        if source + 1 < rate_count:
            kernel[source, source + 1] = p_plus
    return kernel


@njit(cache=True, nogil=True)
def standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@njit(cache=True, nogil=True)
def broad_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    momentum: float,
    broad_sigma_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    half_step = 0.5 * rate_step
    support_lower = rates[0] - half_step
    support_upper = rates[-1] + half_step
    kernel = np.empty((rate_count, rate_count), dtype=np.float64)
    in_support_mass = np.empty(rate_count, dtype=np.float64)
    for source in range(rate_count):
        center = (
            rates[source]
            - (1.0 - momentum) * rates[source] * delta_md
        )
        for destination in range(rate_count):
            lower = rates[destination] - half_step
            upper = rates[destination] + half_step
            lower_z = (lower - center) / broad_sigma_rate
            upper_z = (upper - center) / broad_sigma_rate
            kernel[source, destination] = (
                standard_normal_cdf(upper_z)
                - standard_normal_cdf(lower_z)
            )
        in_support_mass[source] = (
            standard_normal_cdf(
                (support_upper - center) / broad_sigma_rate
            )
            - standard_normal_cdf(
                (support_lower - center) / broad_sigma_rate
            )
        )
    return kernel, in_support_mass


@njit(cache=True, nogil=True)
def mixed_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
    jump_weight: float,
    broad_sigma_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = parent_local_rate_kernel(rates, delta_md, sig_r, momentum)
    broad, broad_mass = broad_rate_kernel(
        rates,
        delta_md,
        momentum,
        broad_sigma_rate,
    )
    mixture = (1.0 - jump_weight) * local + jump_weight * broad
    return local, broad, mixture, broad_mass


@njit(cache=True, nogil=True, parallel=True)
def precompute_mixture_rate_log_kernels(
    delta_md: np.ndarray,
    rates: np.ndarray,
    sig_r: float,
    momentum: float,
    jump_weight: float,
    broad_sigma_rate: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    row_count = len(delta_md)
    rate_count = len(rates)
    mixture_logs = np.full(
        (row_count, rate_count, rate_count),
        -1.0e18,
        dtype=np.float64,
    )
    weighted_broad_logs = np.full(
        (row_count, rate_count, rate_count),
        -1.0e18,
        dtype=np.float64,
    )
    broad_mass_errors = np.zeros(row_count, dtype=np.float64)
    mixture_errors = np.zeros(row_count, dtype=np.float64)
    for row in prange(row_count):
        local, broad, mixture, broad_mass = mixed_rate_kernel(
            rates,
            float(delta_md[row]),
            sig_r,
            momentum,
            jump_weight,
            broad_sigma_rate,
        )
        row_mass_error = 0.0
        row_mixture_error = 0.0
        for source in range(rate_count):
            observed_broad_mass = 0.0
            for destination in range(rate_count):
                observed_broad_mass += broad[source, destination]
                reconstructed = (
                    (1.0 - jump_weight)
                    * local[source, destination]
                    + jump_weight
                    * broad[source, destination]
                )
                row_mixture_error = max(
                    row_mixture_error,
                    abs(
                        mixture[source, destination]
                        - reconstructed
                    ),
                )
                if mixture[source, destination] > 0.0:
                    mixture_logs[row, source, destination] = math.log(
                        mixture[source, destination]
                    )
                weighted_broad = (
                    jump_weight * broad[source, destination]
                )
                if weighted_broad > 0.0:
                    weighted_broad_logs[
                        row,
                        source,
                        destination,
                    ] = math.log(weighted_broad)
            row_mass_error = max(
                row_mass_error,
                abs(observed_broad_mass - broad_mass[source]),
            )
        broad_mass_errors[row] = row_mass_error
        mixture_errors[row] = row_mixture_error
    return (
        mixture_logs,
        weighted_broad_logs,
        float(np.max(broad_mass_errors)),
        float(np.max(mixture_errors)),
    )


@njit(cache=True, nogil=True)
def branch_responsibility_from_messages(
    alpha_previous: np.ndarray,
    beta_after_position: np.ndarray,
    local_kernel: np.ndarray,
    broad_kernel: np.ndarray,
    rates: np.ndarray,
    jump_weight: float,
) -> tuple[float, float, float, float]:
    rate_count = len(rates)
    denominator = 0.0
    broad_numerator = 0.0
    nonadjacent_numerator = 0.0
    signed_nonadjacent_numerator = 0.0
    for position in range(alpha_previous.shape[0]):
        for source in range(rate_count):
            alpha_value = alpha_previous[position, source]
            if alpha_value <= 0.0:
                continue
            for destination in range(rate_count):
                continuation = beta_after_position[position, destination]
                if continuation <= 0.0:
                    continue
                local_probability = (
                    (1.0 - jump_weight)
                    * local_kernel[source, destination]
                )
                broad_probability = (
                    jump_weight * broad_kernel[source, destination]
                )
                common = alpha_value * continuation
                denominator += common * (
                    local_probability + broad_probability
                )
                broad_edge = common * broad_probability
                broad_numerator += broad_edge
                if abs(destination - source) > 1:
                    nonadjacent_numerator += broad_edge
                    signed_nonadjacent_numerator += (
                        broad_edge
                        * (rates[destination] - rates[source])
                    )
    if denominator <= 0.0:
        return math.nan, math.nan, math.nan, denominator
    responsibility = broad_numerator / denominator
    nonadjacent_mass = nonadjacent_numerator / denominator
    signed_rate_delta = (
        signed_nonadjacent_numerator / nonadjacent_numerator
        if nonadjacent_numerator > 0.0
        else 0.0
    )
    return (
        responsibility,
        nonadjacent_mass,
        signed_rate_delta,
        denominator,
    )


def reference_parent_local_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64)
    rate_step = float(rates[1] - rates[0])
    variance_cells = (float(sig_r) * math.sqrt(float(delta_md)) / rate_step) ** 2
    output = np.zeros((len(rates), len(rates)), dtype=np.float64)
    for source, source_rate in enumerate(rates):
        move = (
            -(1.0 - float(momentum))
            * float(source_rate)
            * float(delta_md)
            / rate_step
        )
        p_plus = max(0.5 * (variance_cells + move), 1.0e-12)
        p_minus = max(0.5 * (variance_cells - move), 1.0e-12)
        if p_plus + p_minus > 0.9:
            scale = 0.9 / (p_plus + p_minus)
            p_plus *= scale
            p_minus *= scale
        if source:
            output[source, source - 1] = p_minus
        output[source, source] = 1.0 - p_plus - p_minus
        if source + 1 < len(rates):
            output[source, source + 1] = p_plus
    return output


def synthetic_kernel_contract(
    fixed: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    rates = np.linspace(
        -float(fixed["rate_span"]),
        float(fixed["rate_span"]),
        int(fixed["n_rates"]),
        dtype=np.float64,
    )
    local_error = 0.0
    mixture_error = 0.0
    broad_mass_error = 0.0
    symmetry_error = 0.0
    for delta_md in (1.0, 7.5, 25.0):
        local, broad, mixture, broad_mass = mixed_rate_kernel(
            rates,
            delta_md,
            float(fixed["sig_r"]),
            float(fixed["momentum"]),
            float(candidate["jump_weight"]),
            float(candidate["broad_sigma_rate"]),
        )
        reference_local = reference_parent_local_rate_kernel(
            rates,
            delta_md,
            float(fixed["sig_r"]),
            float(fixed["momentum"]),
        )
        local_error = max(
            local_error,
            float(np.max(np.abs(local - reference_local))),
        )
        reconstructed = (
            (1.0 - float(candidate["jump_weight"])) * local
            + float(candidate["jump_weight"]) * broad
        )
        mixture_error = max(
            mixture_error,
            float(np.max(np.abs(mixture - reconstructed))),
        )
        broad_mass_error = max(
            broad_mass_error,
            float(np.max(np.abs(broad.sum(axis=1) - broad_mass))),
        )
        middle = len(rates) // 2
        symmetry_error = max(
            symmetry_error,
            float(
                np.max(
                    np.abs(
                        broad[middle]
                        - broad[middle][::-1]
                    )
                )
            ),
        )
    rng = np.random.default_rng(442)
    alpha = rng.uniform(0.01, 1.0, size=(5, len(rates)))
    alpha /= alpha.sum()
    beta = rng.uniform(0.01, 1.0, size=(5, len(rates)))
    local, broad, _, _ = mixed_rate_kernel(
        rates,
        13.0,
        float(fixed["sig_r"]),
        float(fixed["momentum"]),
        float(candidate["jump_weight"]),
        float(candidate["broad_sigma_rate"]),
    )
    fast = branch_responsibility_from_messages(
        alpha,
        beta,
        local,
        broad,
        rates,
        float(candidate["jump_weight"]),
    )
    total = 0.0
    broad_total = 0.0
    nonadjacent_total = 0.0
    signed_total = 0.0
    for position in range(alpha.shape[0]):
        for source in range(len(rates)):
            for destination in range(len(rates)):
                common = (
                    alpha[position, source]
                    * beta[position, destination]
                )
                local_edge = (
                    (1.0 - float(candidate["jump_weight"]))
                    * local[source, destination]
                )
                broad_edge = (
                    float(candidate["jump_weight"])
                    * broad[source, destination]
                )
                total += common * (local_edge + broad_edge)
                broad_total += common * broad_edge
                if abs(destination - source) > 1:
                    nonadjacent_total += common * broad_edge
                    signed_total += (
                        common
                        * broad_edge
                        * (rates[destination] - rates[source])
                    )
    brute = (
        broad_total / total,
        nonadjacent_total / total,
        signed_total / nonadjacent_total,
    )
    responsibility_error = float(
        np.max(
            np.abs(
                np.asarray(fast[:3], dtype=np.float64)
                - np.asarray(brute, dtype=np.float64)
            )
        )
    )
    return {
        "local_branch_parent_parity_max_abs_error": local_error,
        "mixture_decomposition_max_abs_error": mixture_error,
        "broad_in_support_mass_max_abs_error": broad_mass_error,
        "centered_broad_symmetry_max_abs_error": symmetry_error,
        "brute_force_branch_responsibility_max_abs_error": (
            responsibility_error
        ),
    }


# %% [markdown]
# ## 6. Symmetric broad-jump exact forward-backward
#
# Forward and backward messages are normalized after every row. This is
# algebraically equivalent to log-message scaling and avoids underflow while
# allowing the dense 41-by-41 broad branch to use matrix multiplication. The
# posterior branch responsibility is evaluated from the smoothed transition
# edge measure after the entire target-free pass is complete.

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_symmetric_broad_jump(
    emission_log_likelihood: np.ndarray,
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    position_step: float,
    rates: np.ndarray,
    rate_log_kernels: np.ndarray,
    weighted_broad_log_kernels: np.ndarray,
    sig_p: float,
    start_position: float,
    start_sigma: float,
    initial_rate: float,
    initial_rate_sigma: float,
    emission_lambda: float,
    jump_weight: float,
) -> tuple:
    row_count, position_count = emission_log_likelihood.shape
    rate_count = len(rates)
    negative = np.float32(-1.0e18)
    alpha = np.full(
        (row_count, position_count, rate_count),
        negative,
        dtype=np.float32,
    )
    previous = np.full(
        (position_count, rate_count),
        negative,
        dtype=np.float32,
    )
    for position in range(position_count):
        delta_position = (
            (position - start_position) * position_step
        )
        position_prior = -0.5 * (
            delta_position / start_sigma
        ) ** 2
        if position_prior < -60.0:
            continue
        for rate_index in range(rate_count):
            rate_z = (
                (rates[rate_index] - initial_rate)
                / initial_rate_sigma
            )
            previous[position, rate_index] = np.float32(
                position_prior - 0.5 * rate_z * rate_z
            )
    after_rate = np.empty(
        (position_count, rate_count),
        dtype=np.float32,
    )
    current = np.empty(
        (position_count, rate_count),
        dtype=np.float32,
    )

    for row in range(row_count):
        rate_log_kernel = rate_log_kernels[row]
        for position in prange(position_count):
            for destination in range(rate_count):
                source_start = 0
                source_end = rate_count
                if jump_weight == 0.0:
                    source_start = max(destination - 1, 0)
                    source_end = min(destination + 2, rate_count)
                best = negative
                for source in range(source_start, source_end):
                    value = (
                        previous[position, source]
                        + rate_log_kernel[source, destination]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for source in range(source_start, source_end):
                        total += math.exp(
                            previous[position, source]
                            + rate_log_kernel[source, destination]
                            - best
                        )
                    after_rate[position, destination] = np.float32(
                        best + math.log(total)
                    )
                else:
                    after_rate[position, destination] = negative

        sigma_position = (
            sig_p
            if sig_p > 0.35 * position_step
            else 0.35 * position_step
        )
        for destination_rate in range(rate_count):
            mean_position_move = (
                rates[destination_rate] * delta_md[row]
                - delta_z[row]
            )
            center_offset = int(
                math.floor(mean_position_move / position_step + 0.5)
            )
            position_log_kernel = np.empty(5, dtype=np.float64)
            for offset_index in range(5):
                discrete_offset = center_offset - 2 + offset_index
                residual = (
                    discrete_offset * position_step
                    - mean_position_move
                )
                position_log_kernel[offset_index] = (
                    -0.5 * (residual / sigma_position) ** 2
                )
            kernel_max = position_log_kernel[0]
            for offset_index in range(1, 5):
                kernel_max = max(
                    kernel_max,
                    position_log_kernel[offset_index],
                )
            kernel_sum = 0.0
            for offset_index in range(5):
                kernel_sum += math.exp(
                    position_log_kernel[offset_index]
                    - kernel_max
                )
            log_normalizer = kernel_max + math.log(kernel_sum)
            for offset_index in range(5):
                position_log_kernel[offset_index] -= (
                    log_normalizer
                )
            for destination_position in prange(position_count):
                best = negative
                for offset_index in range(5):
                    source_position = (
                        destination_position
                        - (center_offset - 2 + offset_index)
                    )
                    if (
                        source_position < 0
                        or source_position >= position_count
                    ):
                        continue
                    value = (
                        after_rate[
                            source_position,
                            destination_rate,
                        ]
                        + position_log_kernel[offset_index]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for offset_index in range(5):
                        source_position = (
                            destination_position
                            - (center_offset - 2 + offset_index)
                        )
                        if (
                            source_position < 0
                            or source_position >= position_count
                        ):
                            continue
                        total += math.exp(
                            after_rate[
                                source_position,
                                destination_rate,
                            ]
                            + position_log_kernel[offset_index]
                            - best
                        )
                    current[
                        destination_position,
                        destination_rate,
                    ] = np.float32(
                        best
                        + math.log(total)
                        + emission_lambda
                        * emission_log_likelihood[
                            row,
                            destination_position,
                        ]
                    )
                else:
                    current[
                        destination_position,
                        destination_rate,
                    ] = negative
        for position in range(position_count):
            for rate_index in range(rate_count):
                alpha[row, position, rate_index] = current[
                    position,
                    rate_index,
                ]
                previous[position, rate_index] = current[
                    position,
                    rate_index,
                ]

    best = negative
    for position in range(position_count):
        for rate_index in range(rate_count):
            best = max(
                best,
                alpha[
                    row_count - 1,
                    position,
                    rate_index,
                ],
            )
    likelihood_sum = 0.0
    for position in range(position_count):
        for rate_index in range(rate_count):
            likelihood_sum += math.exp(
                alpha[
                    row_count - 1,
                    position,
                    rate_index,
                ]
                - best
            )
    log_likelihood = float(best) + math.log(likelihood_sum)

    posterior_position = np.zeros(
        (row_count, position_count),
        dtype=np.float64,
    )
    posterior_rate = np.zeros(
        (row_count, rate_count),
        dtype=np.float64,
    )
    beta_next = np.zeros(
        (position_count, rate_count),
        dtype=np.float32,
    )
    best = negative
    for position in range(position_count):
        for rate_index in range(rate_count):
            value = (
                alpha[
                    row_count - 1,
                    position,
                    rate_index,
                ]
                + beta_next[position, rate_index]
            )
            best = max(best, value)
    posterior_total = 0.0
    for position in range(position_count):
        for rate_index in range(rate_count):
            value = math.exp(
                alpha[
                    row_count - 1,
                    position,
                    rate_index,
                ]
                + beta_next[position, rate_index]
                - best
            )
            posterior_position[
                row_count - 1,
                position,
            ] += value
            posterior_rate[
                row_count - 1,
                rate_index,
            ] += value
            posterior_total += value
    posterior_position[row_count - 1] /= posterior_total
    posterior_rate[row_count - 1] /= posterior_total

    beta_current = np.empty(
        (position_count, rate_count),
        dtype=np.float32,
    )
    beta_after_position = np.empty(
        (position_count, rate_count),
        dtype=np.float32,
    )
    branch_responsibility = np.zeros(
        row_count,
        dtype=np.float64,
    )
    nonadjacent_edge_mass = np.zeros(
        row_count,
        dtype=np.float64,
    )
    signed_nonadjacent_rate_delta = np.zeros(
        row_count,
        dtype=np.float64,
    )

    for row in range(row_count - 1, 0, -1):
        rate_log_kernel = rate_log_kernels[row]
        broad_log_kernel = weighted_broad_log_kernels[row]
        sigma_position = (
            sig_p
            if sig_p > 0.35 * position_step
            else 0.35 * position_step
        )
        for destination_rate in range(rate_count):
            mean_position_move = (
                rates[destination_rate] * delta_md[row]
                - delta_z[row]
            )
            center_offset = int(
                math.floor(mean_position_move / position_step + 0.5)
            )
            position_log_kernel = np.empty(5, dtype=np.float64)
            for offset_index in range(5):
                discrete_offset = center_offset - 2 + offset_index
                residual = (
                    discrete_offset * position_step
                    - mean_position_move
                )
                position_log_kernel[offset_index] = (
                    -0.5 * (residual / sigma_position) ** 2
                )
            kernel_max = position_log_kernel[0]
            for offset_index in range(1, 5):
                kernel_max = max(
                    kernel_max,
                    position_log_kernel[offset_index],
                )
            kernel_sum = 0.0
            for offset_index in range(5):
                kernel_sum += math.exp(
                    position_log_kernel[offset_index]
                    - kernel_max
                )
            log_normalizer = kernel_max + math.log(kernel_sum)
            for offset_index in range(5):
                position_log_kernel[offset_index] -= (
                    log_normalizer
                )
            for source_position in prange(position_count):
                best = negative
                for offset_index in range(5):
                    destination_position = (
                        source_position
                        + (center_offset - 2 + offset_index)
                    )
                    if (
                        destination_position < 0
                        or destination_position >= position_count
                    ):
                        continue
                    value = (
                        position_log_kernel[offset_index]
                        + emission_lambda
                        * emission_log_likelihood[
                            row,
                            destination_position,
                        ]
                        + beta_next[
                            destination_position,
                            destination_rate,
                        ]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for offset_index in range(5):
                        destination_position = (
                            source_position
                            + (center_offset - 2 + offset_index)
                        )
                        if (
                            destination_position < 0
                            or destination_position >= position_count
                        ):
                            continue
                        total += math.exp(
                            position_log_kernel[offset_index]
                            + emission_lambda
                            * emission_log_likelihood[
                                row,
                                destination_position,
                            ]
                            + beta_next[
                                destination_position,
                                destination_rate,
                            ]
                            - best
                        )
                    beta_after_position[
                        source_position,
                        destination_rate,
                    ] = np.float32(
                        best + math.log(total)
                    )
                else:
                    beta_after_position[
                        source_position,
                        destination_rate,
                    ] = negative

        edge_best = negative
        for position in range(position_count):
            for source in range(rate_count):
                for destination in range(rate_count):
                    value = (
                        alpha[row - 1, position, source]
                        + rate_log_kernel[source, destination]
                        + beta_after_position[
                            position,
                            destination,
                        ]
                    )
                    if value > edge_best:
                        edge_best = value
        edge_total = 0.0
        broad_total = 0.0
        nonadjacent_total = 0.0
        signed_total = 0.0
        for position in range(position_count):
            for source in range(rate_count):
                for destination in range(rate_count):
                    edge_total += math.exp(
                        alpha[row - 1, position, source]
                        + rate_log_kernel[source, destination]
                        + beta_after_position[
                            position,
                            destination,
                        ]
                        - edge_best
                    )
                    if (
                        broad_log_kernel[source, destination]
                        <= negative / 2
                    ):
                        continue
                    broad_edge = math.exp(
                        alpha[row - 1, position, source]
                        + broad_log_kernel[source, destination]
                        + beta_after_position[
                            position,
                            destination,
                        ]
                        - edge_best
                    )
                    broad_total += broad_edge
                    if abs(destination - source) > 1:
                        nonadjacent_total += broad_edge
                        signed_total += (
                            broad_edge
                            * (
                                rates[destination]
                                - rates[source]
                            )
                        )
        if edge_total > 0.0:
            branch_responsibility[row] = (
                broad_total / edge_total
            )
            nonadjacent_edge_mass[row] = (
                nonadjacent_total / edge_total
            )
        if nonadjacent_total > 0.0:
            signed_nonadjacent_rate_delta[row] = (
                signed_total / nonadjacent_total
            )

        for position in prange(position_count):
            for source in range(rate_count):
                destination_start = 0
                destination_end = rate_count
                if jump_weight == 0.0:
                    destination_start = max(source - 1, 0)
                    destination_end = min(source + 2, rate_count)
                best = negative
                for destination in range(
                    destination_start,
                    destination_end,
                ):
                    value = (
                        rate_log_kernel[source, destination]
                        + beta_after_position[
                            position,
                            destination,
                        ]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for destination in range(
                        destination_start,
                        destination_end,
                    ):
                        total += math.exp(
                            rate_log_kernel[
                                source,
                                destination,
                            ]
                            + beta_after_position[
                                position,
                                destination,
                            ]
                            - best
                        )
                    beta_current[position, source] = np.float32(
                        best + math.log(total)
                    )
                else:
                    beta_current[position, source] = negative

        best = negative
        for position in range(position_count):
            for rate_index in range(rate_count):
                value = (
                    alpha[row - 1, position, rate_index]
                    + beta_current[position, rate_index]
                )
                best = max(best, value)
        posterior_total = 0.0
        for position in range(position_count):
            for rate_index in range(rate_count):
                value = math.exp(
                    alpha[row - 1, position, rate_index]
                    + beta_current[position, rate_index]
                    - best
                )
                posterior_position[row - 1, position] += value
                posterior_rate[row - 1, rate_index] += value
                posterior_total += value
        posterior_position[row - 1] /= posterior_total
        posterior_rate[row - 1] /= posterior_total
        for position in range(position_count):
            for rate_index in range(rate_count):
                beta_next[position, rate_index] = beta_current[
                    position,
                    rate_index,
                ]

    maximum_posterior_normalization_error = 0.0
    for row in range(row_count):
        maximum_posterior_normalization_error = max(
            maximum_posterior_normalization_error,
            abs(posterior_position[row].sum() - 1.0),
            abs(posterior_rate[row].sum() - 1.0),
        )
    return (
        posterior_position,
        posterior_rate,
        log_likelihood,
        branch_responsibility,
        nonadjacent_edge_mass,
        signed_nonadjacent_rate_delta,
        0.0,
        maximum_posterior_normalization_error,
    )


def run_symmetric_broad_jump_hmm(
    prepared: Mapping[str, Any],
    fixed: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    jump_weight_override: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    jump_weight = (
        float(candidate["jump_weight"])
        if jump_weight_override is None
        else float(jump_weight_override)
    )
    delta_md = np.asarray(prepared["dm"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    (
        rate_log_kernels,
        weighted_broad_log_kernels,
        broad_mass_max_abs_error,
        mixture_decomposition_max_abs_error,
    ) = precompute_mixture_rate_log_kernels(
        delta_md,
        rates,
        float(fixed["sig_r"]),
        float(fixed["momentum"]),
        jump_weight,
        float(candidate["broad_sigma_rate"]),
    )
    (
        posterior_position,
        posterior_rate,
        log_likelihood,
        branch_responsibility,
        nonadjacent_edge_mass,
        signed_nonadjacent_rate_delta,
        forward_normalization_error,
        posterior_normalization_error,
    ) = _hmm2_symmetric_broad_jump(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        delta_md,
        np.asarray(prepared["dz"], dtype=np.float64),
        float(fixed["position_grid_step_ft"]),
        rates,
        rate_log_kernels,
        weighted_broad_log_kernels,
        float(fixed["sig_p"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        float(fixed["emission_lambda"]),
        jump_weight,
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = posterior_position @ grid
    posterior_std = np.sqrt(
        np.maximum(
            posterior_position @ (grid**2) - posterior_mean**2,
            0.0,
        )
    )
    posterior_rate_mean = posterior_rate @ rates
    transition_kernel_sha256 = array_bundle_sha256(
        rates=rates,
        delta_md=delta_md,
        rate_log_kernels=rate_log_kernels,
        weighted_broad_log_kernels=weighted_broad_log_kernels,
    )
    responsibility_sha256 = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        branch_responsibility=np.asarray(
            branch_responsibility,
            dtype=np.float64,
        ),
        nonadjacent_edge_mass=np.asarray(
            nonadjacent_edge_mass,
            dtype=np.float64,
        ),
        signed_nonadjacent_rate_delta=np.asarray(
            signed_nonadjacent_rate_delta,
            dtype=np.float64,
        ),
    )
    prediction_sha256 = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        posterior_mean=np.asarray(posterior_mean, dtype=np.float32),
        posterior_std=np.asarray(posterior_std, dtype=np.float32),
    )
    diagnostic_sha256 = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        posterior_rate_mean=np.asarray(
            posterior_rate_mean,
            dtype=np.float32,
        ),
        branch_responsibility=np.asarray(
            branch_responsibility,
            dtype=np.float32,
        ),
        nonadjacent_edge_mass=np.asarray(
            nonadjacent_edge_mass,
            dtype=np.float32,
        ),
        signed_nonadjacent_rate_delta=np.asarray(
            signed_nonadjacent_rate_delta,
            dtype=np.float32,
        ),
    )
    return {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "posterior_rate_mean": posterior_rate_mean,
        "log_likelihood": float(log_likelihood),
        "branch_responsibility": branch_responsibility,
        "nonadjacent_edge_mass": nonadjacent_edge_mass,
        "signed_nonadjacent_rate_delta": (
            signed_nonadjacent_rate_delta
        ),
        "maximum_normalization_error": max(
            float(forward_normalization_error),
            float(posterior_normalization_error),
        ),
        "kernel_audit": {
            "broad_in_support_mass_max_abs_error": float(
                broad_mass_max_abs_error
            ),
            "mixture_decomposition_max_abs_error": float(
                mixture_decomposition_max_abs_error
            ),
        },
        "transition_kernel_sha256": transition_kernel_sha256,
        "responsibility_sha256": responsibility_sha256,
        "prediction_sha256": prediction_sha256,
        "diagnostic_sha256": diagnostic_sha256,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 7. Target-free transition diagnostics and prediction freeze
#
# The fixed32 manifest is read with only its well identity before this phase.
# For every well, transition-kernel, branch-responsibility, prediction, and
# diagnostic SHA values are frozen before role, fold, suffix truth, persistent
# episodes, or exp408 cause labels can be opened.

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
    posterior_rate_mean: np.ndarray
    branch_responsibility: np.ndarray
    nonadjacent_edge_mass: np.ndarray
    signed_nonadjacent_rate_delta: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    prefix_rows: int
    transition_kernel_sha256: str
    responsibility_sha256: str
    prediction_sha256: str
    diagnostic_sha256: str
    broad_in_support_mass_max_abs_error: float
    mixture_decomposition_max_abs_error: float
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
    candidate: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, fixed)
    decoded = run_symmetric_broad_jump_hmm(
        prepared,
        fixed,
        candidate,
    )
    parent = (
        saved_parent.sort_values("row_idx", kind="mergesort")
        .reset_index(drop=True)
    )
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
        raw_gr_missing=np.asarray(
            prepared["raw_gr_missing"],
            dtype=bool,
        ),
        parent_prediction=parent["parent_prediction"].to_numpy(
            np.float64
        ),
        candidate_prediction=np.asarray(
            decoded["posterior_mean"],
            dtype=np.float64,
        ),
        candidate_posterior_std=np.asarray(
            decoded["posterior_std"],
            dtype=np.float64,
        ),
        posterior_rate_mean=np.asarray(
            decoded["posterior_rate_mean"],
            dtype=np.float64,
        ),
        branch_responsibility=np.asarray(
            decoded["branch_responsibility"],
            dtype=np.float64,
        ),
        nonadjacent_edge_mass=np.asarray(
            decoded["nonadjacent_edge_mass"],
            dtype=np.float64,
        ),
        signed_nonadjacent_rate_delta=np.asarray(
            decoded["signed_nonadjacent_rate_delta"],
            dtype=np.float64,
        ),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        prefix_rows=int(prepared["prefix_rows"]),
        transition_kernel_sha256=str(
            decoded["transition_kernel_sha256"]
        ),
        responsibility_sha256=str(
            decoded["responsibility_sha256"]
        ),
        prediction_sha256=str(decoded["prediction_sha256"]),
        diagnostic_sha256=str(decoded["diagnostic_sha256"]),
        broad_in_support_mass_max_abs_error=float(
            decoded["kernel_audit"][
                "broad_in_support_mass_max_abs_error"
            ]
        ),
        mixture_decomposition_max_abs_error=float(
            decoded["kernel_audit"][
                "mixture_decomposition_max_abs_error"
            ]
        ),
        maximum_normalization_error=float(
            decoded["maximum_normalization_error"]
        ),
        log_likelihood=float(decoded["log_likelihood"]),
        elapsed_seconds=float(decoded["elapsed_seconds"]),
    )
    combined_transition_sha = hashlib.sha256(
        stable_json_bytes(
            {
                "transition_kernel_sha256": (
                    frozen.transition_kernel_sha256
                ),
                "responsibility_sha256": frozen.responsibility_sha256,
            }
        )
    ).hexdigest()
    ledger.freeze(
        well,
        schedule_sha256=combined_transition_sha,
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
                "candidate_posterior_std": (
                    item.candidate_posterior_std
                ),
            }
        )
        for item in frozen_wells
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def transition_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    pieces = [
        pd.DataFrame(
            {
                "well": item.well,
                "row_idx": item.row_idx,
                "suffix_offset": np.arange(
                    len(item.row_idx),
                    dtype=np.int64,
                ),
                "branch_responsibility": (
                    item.branch_responsibility
                ),
                "nonadjacent_edge_mass": item.nonadjacent_edge_mass,
                "signed_nonadjacent_rate_delta": (
                    item.signed_nonadjacent_rate_delta
                ),
            }
        )
        for item in frozen_wells
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def diagnostic_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    pieces = [
        pd.DataFrame(
            {
                "well": item.well,
                "row_idx": item.row_idx,
                "suffix_offset": np.arange(
                    len(item.row_idx),
                    dtype=np.int64,
                ),
                "raw_gr_missing": item.raw_gr_missing,
                "posterior_rate_mean": item.posterior_rate_mean,
                "candidate_posterior_std": (
                    item.candidate_posterior_std
                ),
            }
        )
        for item in frozen_wells
    ]
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
        {
            "well": item.well,
            "sha256": str(getattr(item, attribute)),
        }
        for item in sorted(
            frozen_wells,
            key=lambda value: value.well,
        )
    ]
    return hashlib.sha256(stable_json_bytes(rows)).hexdigest()
# %% [markdown]
# ## 8. Truth-late Stage 0 readout
#
# Truth is opened only after all fixed32 transition diagnostics and predictions
# are frozen. Future direction follows the preregistered exp411 definition:
# the median physical interval rate in the next 32 rows minus the median in the
# preceding 32 rows. Agreement is weighted by posterior non-adjacent broad-edge
# mass; this does not feed back into the HMM.

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
    truth: pd.DataFrame,
    *,
    last_known_tvt: float,
    last_known_md: float,
    last_known_z: float,
) -> np.ndarray:
    tvt = truth["TVT"].to_numpy(np.float64)
    md = truth["MD"].to_numpy(np.float64)
    z = truth["Z"].to_numpy(np.float64)
    delta_tvt = np.diff(
        np.concatenate([[float(last_known_tvt)], tvt])
    )
    delta_z = np.diff(
        np.concatenate([[float(last_known_z)], z])
    )
    delta_md = np.diff(
        np.concatenate([[float(last_known_md)], md])
    )
    rate = np.full(len(truth), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(delta_tvt)
        & np.isfinite(delta_z)
        & np.isfinite(delta_md)
        & (delta_md > 0.0)
    )
    rate[valid] = (
        delta_tvt[valid] + delta_z[valid]
    ) / delta_md[valid]
    return rate


def jump_future_direction_readout(
    frozen: FrozenWell,
    truth: pd.DataFrame,
    *,
    horizon_rows: int,
) -> pd.DataFrame:
    true_rate = physical_true_interval_rate(
        truth,
        last_known_tvt=frozen.last_known_tvt,
        last_known_md=frozen.last_known_md,
        last_known_z=frozen.last_known_z,
    )
    rows: list[dict[str, Any]] = []
    for offset in range(1, len(true_rate)):
        past_start = offset - int(horizon_rows) + 1
        future_end = offset + 1 + int(horizon_rows)
        eligible = (
            past_start >= 0
            and future_end <= len(true_rate)
        )
        past_median = math.nan
        future_median = math.nan
        true_change = math.nan
        true_direction = 0
        if eligible:
            past = true_rate[past_start : offset + 1]
            future = true_rate[offset + 1 : future_end]
            eligible = bool(
                np.isfinite(past).all()
                and np.isfinite(future).all()
            )
            if eligible:
                past_median = float(np.median(past))
                future_median = float(np.median(future))
                true_change = future_median - past_median
                true_direction = int(np.sign(true_change))
        jump_delta = float(
            frozen.signed_nonadjacent_rate_delta[offset]
        )
        jump_direction = int(np.sign(jump_delta))
        mass = float(frozen.nonadjacent_edge_mass[offset])
        rows.append(
            {
                "well": frozen.well,
                "role": frozen.role,
                "fold": frozen.fold,
                "row_idx": int(frozen.row_idx[offset]),
                "suffix_offset": int(offset),
                "branch_responsibility": float(
                    frozen.branch_responsibility[offset]
                ),
                "nonadjacent_edge_mass": mass,
                "signed_nonadjacent_rate_delta": jump_delta,
                "jump_direction": jump_direction,
                "eligible_future_direction": bool(eligible),
                "past_true_rate_median": past_median,
                "future_true_rate_median": future_median,
                "future_true_rate_change": true_change,
                "future_true_rate_direction": true_direction,
                "direction_agreement": bool(
                    eligible
                    and jump_direction != 0
                    and true_direction == jump_direction
                ),
                "agreement_weight": (
                    mass if eligible and jump_direction != 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def well_truth_late_metrics(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    actual = truth["TVT"].to_numpy(np.float64)
    parent_error = frozen.parent_prediction - actual
    candidate_error = frozen.candidate_prediction - actual
    return {
        "well": frozen.well,
        "role": frozen.role,
        "fold": frozen.fold,
        "rows": len(actual),
        "parent_sse": float(np.sum(parent_error**2)),
        "candidate_sse": float(np.sum(candidate_error**2)),
        "parent_rmse_ft": float(np.sqrt(np.mean(parent_error**2))),
        "candidate_rmse_ft": float(
            np.sqrt(np.mean(candidate_error**2))
        ),
        "rmse_delta_ft": float(
            np.sqrt(np.mean(candidate_error**2))
            - np.sqrt(np.mean(parent_error**2))
        ),
        "mean_branch_responsibility": float(
            np.mean(frozen.branch_responsibility[1:])
        ),
        "mean_nonadjacent_edge_mass": float(
            np.mean(frozen.nonadjacent_edge_mass[1:])
        ),
        "maximum_nonadjacent_edge_mass": float(
            np.max(frozen.nonadjacent_edge_mass[1:])
        ),
        "raw_gr_missing_fraction": float(
            np.mean(frozen.raw_gr_missing)
        ),
        "maximum_normalization_error": (
            frozen.maximum_normalization_error
        ),
        "hmm_elapsed_seconds": frozen.elapsed_seconds,
        "transition_kernel_sha256": (
            frozen.transition_kernel_sha256
        ),
        "responsibility_sha256": frozen.responsibility_sha256,
        "prediction_sha256": frozen.prediction_sha256,
        "diagnostic_sha256": frozen.diagnostic_sha256,
        "broad_in_support_mass_max_abs_error": (
            frozen.broad_in_support_mass_max_abs_error
        ),
        "mixture_decomposition_max_abs_error": (
            frozen.mixture_decomposition_max_abs_error
        ),
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
    frame = frame.loc[
        frame["well"].isin(persistent_wells)
    ].copy()
    ledger.record_episode_late(len(frame))
    if (
        frame.empty
        or frame["well"].nunique() != len(persistent_wells)
    ):
        raise ValueError(
            "selected persistent wells are missing episode rows"
        )
    return (
        frame.sort_values(
            ["well", "start_row_idx"],
            kind="mergesort",
        ).reset_index(drop=True),
        {
            "path": str(path),
            "sha256": observed,
            "selected_rows": len(frame),
        },
    )


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
        dtype={"well": str, "episode_id": str},
    )
    frame = frame.loc[
        frame["episode_id"].isin(selected_episode_ids)
    ].copy()
    ledger.record_cause_late(len(frame))
    if frame["episode_id"].duplicated().any():
        raise ValueError("exp408 cause rows are not unique")
    return frame, {
        "path": str(path),
        "sha256": observed,
        "selected_rows": len(frame),
    }


def episode_truth_late_readout(
    episodes: pd.DataFrame,
    causes: pd.DataFrame,
    frozen_by_well: Mapping[str, FrozenWell],
    truth_by_well: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    cause_lookup = (
        causes.set_index("episode_id")["cause"]
        .astype(str)
        .to_dict()
        if "cause" in causes.columns
        else {}
    )
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        well = str(episode.well)
        frozen = frozen_by_well[well]
        truth = truth_by_well[well]
        start = int(episode.start_row_idx)
        end = int(episode.end_row_idx_exclusive)
        mask = (
            (frozen.row_idx >= start)
            & (frozen.row_idx < end)
        )
        if not np.any(mask):
            raise ValueError(
                f"{episode.episode_id}: episode has no fixed32 rows"
            )
        actual_lookup = truth.set_index("row_idx")["TVT"]
        actual = actual_lookup.loc[
            frozen.row_idx[mask]
        ].to_numpy(np.float64)
        parent_sse = float(
            np.sum(
                (
                    frozen.parent_prediction[mask]
                    - actual
                )
                ** 2
            )
        )
        candidate_sse = float(
            np.sum(
                (
                    frozen.candidate_prediction[mask]
                    - actual
                )
                ** 2
            )
        )
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": well,
                "fold": frozen.fold,
                "cause": cause_lookup.get(
                    str(episode.episode_id),
                    "unavailable",
                ),
                "start_row_idx": start,
                "end_row_idx_exclusive": end,
                "rows": int(np.count_nonzero(mask)),
                "mean_branch_responsibility": float(
                    np.mean(
                        frozen.branch_responsibility[mask]
                    )
                ),
                "mean_nonadjacent_edge_mass": float(
                    np.mean(
                        frozen.nonadjacent_edge_mass[mask]
                    )
                ),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "sse_reduction_fraction": (
                    (parent_sse - candidate_sse) / parent_sse
                    if parent_sse > 0.0
                    else math.nan
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["well", "start_row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
# %% [markdown]
# ## 9. Technical and mechanism gates

# %%
def safe_fraction(
    numerator: float | int,
    denominator: float | int,
) -> float:
    return (
        float(numerator / denominator)
        if denominator
        else math.nan
    )


def sse_reduction(
    parent_sse: float,
    candidate_sse: float,
) -> float:
    return (
        float((parent_sse - candidate_sse) / parent_sse)
        if parent_sse > 0.0
        else math.nan
    )


def weighted_direction_agreement(
    frame: pd.DataFrame,
) -> float:
    if frame.empty:
        return math.nan
    eligible = frame.loc[
        frame["eligible_future_direction"].astype(bool)
        & frame["jump_direction"].ne(0)
        & frame["future_true_rate_direction"].ne(0)
        & frame["agreement_weight"].gt(0.0)
    ]
    total_weight = float(eligible["agreement_weight"].sum())
    if total_weight <= 0.0:
        return math.nan
    matched_weight = float(
        eligible.loc[
            eligible["direction_agreement"].astype(bool),
            "agreement_weight",
        ].sum()
    )
    return matched_weight / total_weight


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    contract_checks: Mapping[str, Any],
    prediction_artifact: Mapping[str, Any],
    transition_artifact: Mapping[str, Any],
    diagnostic_artifact: Mapping[str, Any],
    direction_readout: pd.DataFrame,
    episode_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(
        config,
        "gates.stage0_fixed32.technical",
    )
    mechanism_config = get_nested(
        config,
        "gates.stage0_fixed32.mechanism",
    )
    total_rows = int(
        sum(len(item.row_idx) for item in frozen_wells)
    )
    transition_rows = max(
        total_rows - len(frozen_wells),
        0,
    )
    finite_rows = int(
        sum(
            np.isfinite(item.candidate_prediction).sum()
            for item in frozen_wells
        )
    )
    finite_coverage = safe_fraction(finite_rows, total_rows)
    maximum_normalization_error = float(
        max(
            item.maximum_normalization_error
            for item in frozen_wells
        )
    )
    actual_broad_mass_error = float(
        max(
            item.broad_in_support_mass_max_abs_error
            for item in frozen_wells
        )
    )
    actual_mixture_decomposition_error = float(
        max(
            item.mixture_decomposition_max_abs_error
            for item in frozen_wells
        )
    )
    all_nonadjacent_mass = np.concatenate(
        [item.nonadjacent_edge_mass[1:] for item in frozen_wells]
    )
    all_responsibility = np.concatenate(
        [item.branch_responsibility[1:] for item in frozen_wells]
    )
    pooled_nonadjacent_edge_mass = float(
        np.sum(all_nonadjacent_mass)
        / transition_rows
    )
    pooled_branch_responsibility = float(
        np.sum(all_responsibility)
        / transition_rows
    )
    runtime_projection = float(
        elapsed_seconds * 773.0 / 32.0
    )
    persistent_metrics = well_metrics.loc[
        well_metrics["role"].eq("persistent")
    ]
    control_metrics = well_metrics.loc[
        well_metrics["role"].eq("control")
    ]
    technical = {
        "expected_wells": (
            len(frozen_wells)
            == int(technical_config["expected_wells"])
        ),
        "expected_rows": (
            total_rows
            == int(technical_config["expected_rows"])
        ),
        "fixed32_roles_and_folds": bool(
            manifest["role"].value_counts().to_dict()
            == {"persistent": 16, "control": 16}
            and manifest["fold"].nunique() == 5
        ),
        "finite_coverage": (
            finite_coverage
            >= float(technical_config["finite_coverage_min"])
        ),
        "local_branch_parent_parity": (
            float(
                contract_checks[
                    "local_branch_parent_parity_max_abs_error"
                ]
            )
            <= float(
                technical_config[
                    "local_branch_parent_parity_max_abs_error"
                ]
            )
        ),
        "mixture_decomposition": (
            max(
                float(
                    contract_checks[
                        "mixture_decomposition_max_abs_error"
                    ]
                ),
                actual_mixture_decomposition_error,
            )
            <= float(
                technical_config[
                    "mixture_decomposition_max_abs_error"
                ]
            )
        ),
        "broad_in_support_mass": (
            max(
                float(
                    contract_checks[
                        "broad_in_support_mass_max_abs_error"
                    ]
                ),
                actual_broad_mass_error,
            )
            <= float(
                technical_config[
                    "broad_in_support_mass_max_abs_error"
                ]
            )
        ),
        "posterior_normalization": (
            maximum_normalization_error
            <= float(
                technical_config[
                    "posterior_normalization_max_error"
                ]
            )
        ),
        "brute_force_branch_responsibility": (
            float(
                contract_checks[
                    "brute_force_branch_responsibility_max_abs_error"
                ]
            )
            <= float(
                technical_config[
                    "brute_force_branch_responsibility_max_abs_error"
                ]
            )
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
        "transition_readback_sha": (
            transition_artifact["logical_sha256"]
            == transition_artifact["readback_logical_sha256"]
        ),
        "diagnostic_readback_sha": (
            diagnostic_artifact["logical_sha256"]
            == diagnostic_artifact["readback_logical_sha256"]
        ),
        "runtime_projection": (
            runtime_projection
            <= float(
                technical_config[
                    "projected_stage1_runtime_seconds_max"
                ]
            )
        ),
        "peak_rss": (
            peak_rss_gb()
            <= float(technical_config["peak_rss_gb_max"])
        ),
    }

    direction_agreement = weighted_direction_agreement(
        direction_readout
    )
    direction_by_fold: list[dict[str, Any]] = []
    for fold in range(5):
        frame = direction_readout.loc[
            direction_readout["fold"].eq(fold)
        ]
        agreement = weighted_direction_agreement(frame)
        direction_by_fold.append(
            {
                "fold": fold,
                "eligible_rows": int(
                    frame[
                        "eligible_future_direction"
                    ].astype(bool).sum()
                ),
                "weighted_direction_agreement": agreement,
                "positive": bool(
                    math.isfinite(agreement)
                    and agreement > 0.50
                ),
            }
        )
    positive_direction_folds = int(
        sum(row["positive"] for row in direction_by_fold)
    )

    forward_episodes = episode_readout.loc[
        episode_readout["cause"].eq(
            "forward_transition_prior_hysteresis"
        )
    ]
    forward_sse_reduction = sse_reduction(
        float(forward_episodes["parent_sse"].sum()),
        float(forward_episodes["candidate_sse"].sum()),
    )
    persistent_sse_reduction = sse_reduction(
        float(episode_readout["parent_sse"].sum()),
        float(episode_readout["candidate_sse"].sum()),
    )
    persistent_improved_wells = int(
        (persistent_metrics["rmse_delta_ft"] < 0.0).sum()
    )
    persistent_fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        frame = episode_readout.loc[
            episode_readout["fold"].eq(fold)
        ]
        reduction = sse_reduction(
            float(frame["parent_sse"].sum()),
            float(frame["candidate_sse"].sum()),
        )
        persistent_fold_rows.append(
            {
                "fold": fold,
                "episodes": len(frame),
                "sse_reduction_fraction": reduction,
                "improving": bool(
                    math.isfinite(reduction)
                    and reduction > 0.0
                ),
            }
        )
    persistent_improving_folds = int(
        sum(
            row["improving"]
            for row in persistent_fold_rows
        )
    )
    control_parent_sse = float(
        control_metrics["parent_sse"].sum()
    )
    control_candidate_sse = float(
        control_metrics["candidate_sse"].sum()
    )
    control_rows = int(control_metrics["rows"].sum())
    control_pooled_delta = (
        math.sqrt(control_candidate_sse / control_rows)
        - math.sqrt(control_parent_sse / control_rows)
    )
    control_by_well_p95 = float(
        np.quantile(
            control_metrics["rmse_delta_ft"].to_numpy(
                np.float64
            ),
            0.95,
        )
    )
    mechanism = {
        "nonadjacent_posterior_edge_mass": (
            pooled_nonadjacent_edge_mass
            >= float(
                mechanism_config[
                    "nonadjacent_posterior_edge_mass_min"
                ]
            )
        ),
        "jump_edge_future_rate_direction_agreement": (
            math.isfinite(direction_agreement)
            and direction_agreement
            >= float(
                mechanism_config[
                    "jump_edge_future_rate_direction_agreement_min"
                ]
            )
        ),
        "jump_edge_direction_positive_folds": (
            positive_direction_folds
            >= int(
                mechanism_config[
                    "jump_edge_direction_positive_folds_min"
                ]
            )
        ),
        "forward_cause_episode_sse_reduction": (
            math.isfinite(forward_sse_reduction)
            and forward_sse_reduction
            >= float(
                mechanism_config[
                    "forward_cause_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_episode_sse_reduction": (
            math.isfinite(persistent_sse_reduction)
            and persistent_sse_reduction
            >= float(
                mechanism_config[
                    "persistent_episode_sse_reduction_min_fraction"
                ]
            )
        ),
        "persistent_improved_wells": (
            persistent_improved_wells
            >= int(
                mechanism_config[
                    "persistent_improved_wells_min"
                ]
            )
        ),
        "persistent_improving_folds": (
            persistent_improving_folds
            >= int(
                mechanism_config[
                    "persistent_improving_folds_min"
                ]
            )
        ),
        "matched_control_pooled_rmse_delta": (
            control_pooled_delta
            <= float(
                mechanism_config[
                    "matched_control_pooled_rmse_delta_max_ft"
                ]
            )
        ),
        "matched_control_by_well_delta_p95": (
            control_by_well_p95
            <= float(
                mechanism_config[
                    "matched_control_by_well_delta_p95_max_ft"
                ]
            )
        ),
    }
    diagnostics = {
        "total_rows": total_rows,
        "transition_rows": transition_rows,
        "finite_coverage": finite_coverage,
        "maximum_normalization_error": (
            maximum_normalization_error
        ),
        "actual_broad_in_support_mass_max_abs_error": (
            actual_broad_mass_error
        ),
        "actual_mixture_decomposition_max_abs_error": (
            actual_mixture_decomposition_error
        ),
        "pooled_branch_responsibility": (
            pooled_branch_responsibility
        ),
        "pooled_nonadjacent_posterior_edge_mass": (
            pooled_nonadjacent_edge_mass
        ),
        "jump_edge_future_rate_direction_agreement": (
            direction_agreement
        ),
        "jump_edge_direction_by_fold": direction_by_fold,
        "jump_edge_direction_positive_folds": (
            positive_direction_folds
        ),
        "forward_cause_episode_sse_reduction_fraction": (
            forward_sse_reduction
        ),
        "persistent_episode_sse_reduction_fraction": (
            persistent_sse_reduction
        ),
        "persistent_improved_wells": persistent_improved_wells,
        "persistent_episode_by_fold": persistent_fold_rows,
        "persistent_improving_folds": (
            persistent_improving_folds
        ),
        "matched_control_pooled_rmse_delta_ft": (
            control_pooled_delta
        ),
        "matched_control_by_well_delta_p95_ft": (
            control_by_well_p95
        ),
        "runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
        "fixed32_is_mechanism_only_not_cv_or_promotion_evidence": (
            True
        ),
    }
    all_pass = bool(
        all(technical.values())
        and all(mechanism.values())
    )
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "stage1_eligible_pending_separate_user_approval": (
            all_pass
        ),
        "fail_action": (
            None
            if all_pass
            else get_nested(
                config,
                "gates.stage0_fixed32.fail_action",
            )
        ),
    }
# %% [markdown]
# ## 10. Guarded Kaggle CPU orchestration

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP442_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp442 Stage 0 must run on Kaggle CPU")


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
    fixed = get_nested(config, "model.fixed_from_exp209")
    candidate = get_nested(
        config,
        "model.candidate_rate_transition",
    )
    contract_checks = synthetic_kernel_contract(
        fixed,
        candidate,
    )
    technical_thresholds = get_nested(
        config,
        "gates.stage0_fixed32.technical",
    )
    for key in (
        "local_branch_parent_parity_max_abs_error",
        "mixture_decomposition_max_abs_error",
        "broad_in_support_mass_max_abs_error",
        "brute_force_branch_responsibility_max_abs_error",
    ):
        if (
            float(contract_checks[key])
            > float(technical_thresholds[key])
        ):
            raise RuntimeError(
                f"exp442 preflight contract failed: {key}="
                f"{contract_checks[key]}"
            )
    set_num_threads(
        int(get_nested(config, "runtime.numba_num_threads"))
    )
    ledger = LeakageLedger(expected_wells=32)
    wells, scope_input = load_fixed32_scope(config, ledger)
    parent, parent_input = load_saved_parent_predictions(
        config,
        set(wells),
        ledger,
    )
    raw_dir = train_data_dir(config)
    parent_groups = parent.groupby(
        "well",
        sort=False,
    ).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(
        get_nested(
            config,
            "runtime.hard_runtime_limit_seconds",
        )
    )
    hard_rss = float(
        get_nested(config, "runtime.peak_rss_limit_gb")
    )
    for well_index, well in enumerate(wells, start=1):
        if well not in parent_groups:
            raise ValueError(
                f"{well}: saved parent rows are missing"
            )
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            saved_parent=parent.iloc[
                parent_groups[well]
            ].copy(),
            fixed=fixed,
            candidate=candidate,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(
                "Stage 0 runtime hard guard exceeded: "
                f"{elapsed}"
            )
        if peak_rss_gb() > hard_rss:
            raise MemoryError(
                "Stage 0 RSS hard guard exceeded: "
                f"{peak_rss_gb()}"
            )
        print(
            json.dumps(
                {
                    "event": "exp442_target_free_well_frozen",
                    "well_index": well_index,
                    "well_count": len(wells),
                    "well": well,
                    "rows": len(frozen.row_idx),
                    "elapsed_seconds": frozen.elapsed_seconds,
                    "transition_kernel_sha256": (
                        frozen.transition_kernel_sha256
                    ),
                    "responsibility_sha256": (
                        frozen.responsibility_sha256
                    ),
                    "prediction_sha256": (
                        frozen.prediction_sha256
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all fixed32 wells were frozen")

    manifest, manifest_input = (
        load_fixed32_identity_after_all_freeze(
            config,
            ledger,
        )
    )
    attach_late_identity(frozen_wells, manifest)
    output = artifacts_dir()
    predictions = prediction_frame(frozen_wells)
    transitions = transition_frame(frozen_wells)
    diagnostics = diagnostic_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    transition_artifact = write_deterministic_gzip_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_transition_diagnostics.csv.gz",
        transitions,
    )
    diagnostic_artifact = write_deterministic_gzip_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_target_free_diagnostics.csv.gz",
        diagnostics,
    )

    truth_by_well: dict[str, pd.DataFrame] = {}
    well_rows: list[dict[str, Any]] = []
    direction_pieces: list[pd.DataFrame] = []
    horizon_rows = int(
        candidate["future_rate_horizon_rows"]
    )
    for item in frozen_wells:
        truth = load_truth_after_all_freeze(
            item,
            raw_dir,
            ledger,
        )
        truth_by_well[item.well] = truth
        well_rows.append(
            well_truth_late_metrics(item, truth)
        )
        direction_pieces.append(
            jump_future_direction_readout(
                item,
                truth,
                horizon_rows=horizon_rows,
            )
        )
    well_metrics = pd.DataFrame(well_rows).sort_values(
        "well",
        kind="mergesort",
    )
    direction_readout = pd.concat(
        direction_pieces,
        ignore_index=True,
    ).sort_values(
        ["well", "row_idx"],
        kind="mergesort",
    )
    persistent_wells = set(
        manifest.loc[
            manifest["role"].eq("persistent"),
            "well",
        ].astype(str)
    )
    episodes, episode_input = (
        load_persistent_episodes_after_all_freeze(
            config,
            persistent_wells,
            ledger,
        )
    )
    causes, cause_input = (
        load_episode_causes_after_all_freeze(
            config,
            set(episodes["episode_id"].astype(str)),
            ledger,
        )
    )
    frozen_by_well = {
        item.well: item for item in frozen_wells
    }
    episode_readout = episode_truth_late_readout(
        episodes,
        causes,
        frozen_by_well,
        truth_by_well,
    )
    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    direction_artifact = write_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_jump_direction_readout.csv",
        direction_readout,
    )
    episode_artifact = write_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0_episode_truth_late_readout.csv",
        episode_readout,
    )
    elapsed_seconds = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        manifest=manifest,
        frozen_wells=frozen_wells,
        contract_checks=contract_checks,
        prediction_artifact=prediction_artifact,
        transition_artifact=transition_artifact,
        diagnostic_artifact=diagnostic_artifact,
        direction_readout=direction_readout,
        episode_readout=episode_readout,
        well_metrics=well_metrics,
        ledger=ledger,
        elapsed_seconds=elapsed_seconds,
    )
    input_manifest = {
        "fixed32_scope": scope_input,
        "fixed32_identity_truth_late": manifest_input,
        "saved_exp209_control": parent_input,
        "persistent_episodes_truth_late": episode_input,
        "exp408_episode_causes_truth_late": cause_input,
        "scientific_contract_sha256": scientific_contract_sha,
        "contract_checks": contract_checks,
        "combined_transition_kernel_sha256": (
            combined_well_sha(
                frozen_wells,
                "transition_kernel_sha256",
            )
        ),
        "combined_responsibility_sha256": (
            combined_well_sha(
                frozen_wells,
                "responsibility_sha256",
            )
        ),
        "combined_prediction_sha256": combined_well_sha(
            frozen_wells,
            "prediction_sha256",
        ),
        "combined_diagnostic_sha256": combined_well_sha(
            frozen_wells,
            "diagnostic_sha256",
        ),
        "leakage_ledger": {
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
        output
        / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    eligible = bool(
        gates[
            "stage1_eligible_pending_separate_user_approval"
        ]
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0_all_gates_passed_pending_separate_stage1_approval"
            if eligible
            else "stage0_fail_closed"
        ),
        "stage": "stage0_fixed32",
        "fixed32_is_mechanism_only_not_cv": True,
        "cv": None,
        "lb": None,
        "execution_contract": execution_contract,
        "scientific_contract": scientific_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "contract_checks": contract_checks,
        "gates": gates,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "projected_stage1_seconds": (
                elapsed_seconds * 773.0 / 32.0
            ),
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
            "numba_threads": int(
                get_nested(
                    config,
                    "runtime.numba_num_threads",
                )
            ),
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "transition_diagnostics": transition_artifact,
            "target_free_diagnostics": diagnostic_artifact,
            "well_metrics": well_artifact,
            "jump_direction_truth_late_readout": (
                direction_artifact
            ),
            "episode_truth_late_readout": episode_artifact,
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
            "strategy": get_nested(
                config,
                "validation.strategy",
            ),
            "stage": "stage0_fixed32",
            "cv": None,
            "lb": None,
            "fixed32_is_mechanism_only": True,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "technical_gates": gates["technical"],
        "mechanism_gates": gates["mechanism"],
        "stage1_eligible_pending_separate_user_approval": (
            eligible
        ),
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(
        json.dumps(
            to_jsonable(summary),
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


# %% [markdown]
# Direct execution is limited by the current config to the authorized fixed32
# Kaggle CPU Stage 0.

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
                "event": "exp442_stage0_execution",
                "experiment": EXPERIMENT_NAME,
                "status": get_nested(
                    CONFIG,
                    "experiment.status",
                ),
                "selected_stage": get_nested(
                    CONFIG,
                    "execution.selected_stage",
                ),
                "execution_counts": EXECUTION_COUNTS,
                "canonical_notebook_adoption_authorized": get_nested(
                    CONFIG,
                    "execution.canonical_notebook_adoption_authorized",
                ),
                "kaggle_package_authorized": get_nested(
                    CONFIG,
                    "execution.kaggle_package_authorized",
                ),
                "stage0_run_authorized": get_nested(
                    CONFIG,
                    "execution.stage0_run_authorized",
                ),
                "stage1_run_authorized": get_nested(
                    CONFIG,
                    "execution.stage1_run_authorized",
                ),
                "inference": get_nested(
                    CONFIG,
                    "execution.inference_authorized",
                ),
                "submission": get_nested(
                    CONFIG,
                    "execution.submission_authorized",
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    SUMMARY = run_stage0(CONFIG)
