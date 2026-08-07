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
# # exp440 ambiguity-gated predictive-prior exact HMM — train
#
# This implementation keeps the complete exp209 exact-HMM contract. During
# the candidate's causal forward pass it applies the ordinary Gaussian GR
# emission provisionally, detects exp236 bimodality in the resulting TVT
# marginal, and neutralizes that emission only on raw-GR-observed ambiguous
# rows. The completed row-wise emission schedule is frozen and reused without
# modification in the backward pass.
#
# The fixed32 result remains failed closed. A later explicit user request
# authorizes only the unchanged candidate's four-shard full-OOF confirmation.
# Inference and submission remain disabled.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable contracts
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Fixed32 scope, saved parent, and target-free raw inputs
# 4. Exact exp209 input preparation
# 5. Fixed exp236 bimodality detector
# 6. Ambiguity-gated exact forward-backward
# 7. Target-free schedule and prediction freeze
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

EXPERIMENT_NAME = "exp440_ambiguity_gated_predictive_prior_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
BIMODALITY_SOURCE = "exp236_exact_hmm_posterior_bimodality_audit"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
NEGATIVE_LOG_SENTINEL = np.float32(-1.0e18)
STAGE1_SHARD_COUNT = 4

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
        raise ValueError("wrong exp440 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp440 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp440 scientific parent changed")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp440 implementation is not authorized")
    if bool(get_nested(config, "execution.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "execution.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu", True)):
        raise ValueError("exp440 is CPU-only")

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
        raise ValueError(f"exp440 execution contract changed: {observed} != {expected}")
    if bool(get_nested(config, "data.exp209_saved_control.regenerate", True)):
        raise ValueError("saved exp209 control must not be regenerated")
    if require_run_authorization:
        stage = str(
            os.environ.get("EXP440_STAGE")
            or get_nested(config, "execution.selected_stage")
        )
        if not bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ):
            raise RuntimeError(
                "exp440 Stage 0 requires canonical train notebook adoption"
            )
        if not bool(get_nested(config, "execution.stage0_run_authorized", False)):
            raise RuntimeError(
                "exp440 implementation approval does not authorize Stage 0 execution"
            )
        if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
            raise RuntimeError(
                "exp440 execution requires separate Kaggle package approval"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("exp440 run_hmm remains fail-closed")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError("exp440 prediction creation remains fail-closed")
        if bool(get_nested(config, "execution.create_submission", True)):
            raise ValueError("exp440 must not create a submission")
        if stage == "stage0_fixed32":
            if (
                get_nested(config, "runtime.kaggle.train_kernel_version")
                is not None
                and not bool(get_nested(config, "execution.rerun_authorized", False))
            ):
                raise RuntimeError(
                    "exp440 Stage 0 failed closed; rerun is not authorized"
                )
            if not bool(get_nested(config, "execution.stage0_run_authorized", False)):
                raise RuntimeError(
                    "exp440 implementation approval does not authorize Stage 0"
                )
        elif stage in {"stage1_full_oof", "stage1_shard", "stage1_merge"}:
            if (
                get_nested(config, "experiment.status")
                == "stage1_full_oof_failed_closed"
                and not bool(
                    get_nested(
                        config,
                        "execution.stage1_rerun_authorized",
                        False,
                    )
                )
            ):
                raise RuntimeError(
                    "exp440 Stage 1 failed closed; rerun is not authorized"
                )
            if not bool(get_nested(config, "execution.stage1_run_authorized", False)):
                raise RuntimeError("exp440 Stage 1 execution is not authorized")
            if not bool(
                get_nested(
                    config,
                    "execution.stage1_prerequisite_override_authorized",
                    False,
                )
            ):
                raise RuntimeError(
                    "exp440 Stage 1 needs an explicit Stage-0 prerequisite override"
                )
            if int(
                get_nested(config, "execution.stage1_well_shard_count", -1)
            ) != STAGE1_SHARD_COUNT:
                raise ValueError("exp440 Stage 1 must use four deterministic shards")
        else:
            raise RuntimeError(f"unsupported exp440 execution stage: {stage}")
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
        "emission_lambda_clear": 1.0,
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "sigma_clip": [10.0, 60.0],
        "start_sigma_ft": 0.75,
        "initial_rate_sigma": 0.01,
        "band_pad_ft": 100.0,
        "rate_center": "zero",
        "output": "smoothed_posterior_mean_and_std",
    }
    if fixed != expected_fixed:
        raise ValueError(f"exp209 HMM contract changed: {fixed} != {expected_fixed}")
    ambiguity = dict(get_nested(config, "model.ambiguity_contract") or {})
    expected_ambiguity = {
        "source": BIMODALITY_SOURCE,
        "min_peak_height": 0.02,
        "min_top2_mass": 0.10,
        "min_top2_to_top1_mass_ratio": 0.25,
        "min_peak_separation_ft": 6.0,
        "min_valley_depth": 0.30,
    }
    if ambiguity != expected_ambiguity:
        raise ValueError(
            f"exp236 bimodality contract changed: {ambiguity} != {expected_ambiguity}"
        )
    candidate = dict(get_nested(config, "model.candidate") or {})
    expected_candidate = {
        "ambiguity_source": "causal_provisional_filtered_tvt_position_marginal",
        "eligibility": "raw_gr_observed_only",
        "ambiguous_emission_lambda": 0.0,
        "clear_emission_lambda": 1.0,
        "candidate_filtered_if_ambiguous": "transitioned_predictive_joint_distribution",
        "hard_previous_tvt_point_freeze": False,
        "schedule_iteration": "single_causal_forward_pass_no_backward_or_truth_feedback",
        "backward_policy": "reuse_frozen_forward_ambiguity_schedule",
    }
    if candidate != expected_candidate:
        raise ValueError(
            f"exp440 candidate contract changed: {candidate} != {expected_candidate}"
        )
    variants = list(get_nested(config, "model.active_scientific_variants") or [])
    if variants != ["ambiguity_gated_predictive_prior_hold"]:
        raise ValueError("exp440 must contain exactly one frozen scientific candidate")
    return {
        "fixed_from_exp209": fixed,
        "ambiguity_contract": ambiguity,
        "candidate": candidate,
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
    raise FileNotFoundError("exp440 config.yaml was not found")


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
    if len(target_wells) == int(get_nested(config, "validation.expected_wells")):
        expected_rows = int(get_nested(config, "validation.expected_rows"))
    elif len(target_wells) == int(
        get_nested(config, "data.fixed32_manifest.total_wells")
    ):
        expected_rows = int(
            get_nested(config, "data.fixed32_manifest.expected_suffix_rows")
        )
    else:
        expected_rows = len(frame)
    if len(frame) != expected_rows:
        raise ValueError(f"saved parent rows={len(frame)}/{expected_rows}")
    if frame["well"].nunique() != len(target_wells):
        raise ValueError("saved parent well coverage mismatch")
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
# ## 5. Fixed exp236 bimodality detector
#
# Boundary maxima, interior tie handling, peak ranking, valley split, mass
# definitions, and all five thresholds reproduce exp236 exactly.

# %%
@njit(cache=True, nogil=True)
def bimodality_diagnostics_1d(
    raw_probabilities,
    grid_step,
    min_peak_height,
    min_top2_mass,
    min_top2_to_top1_mass_ratio,
    min_peak_separation_ft,
    min_valley_depth,
):
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    count = len(probabilities)
    total = 0.0
    for index in range(count):
        total += probabilities[index]
    if count == 0 or not np.isfinite(total) or total <= 0.0:
        return (False, 0, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0)
    probabilities = probabilities / total
    peaks = np.empty(count, dtype=np.int64)
    peak_count = 0
    if count == 1:
        if probabilities[0] >= min_peak_height:
            peaks[0] = 0
            peak_count = 1
    else:
        if (
            probabilities[0] >= probabilities[1]
            and probabilities[0] >= min_peak_height
        ):
            peaks[peak_count] = 0
            peak_count += 1
        for index in range(1, count - 1):
            if (
                probabilities[index] >= min_peak_height
                and probabilities[index] >= probabilities[index - 1]
                and probabilities[index] > probabilities[index + 1]
            ):
                peaks[peak_count] = index
                peak_count += 1
        if (
            probabilities[count - 1] > probabilities[count - 2]
            and probabilities[count - 1] >= min_peak_height
        ):
            peaks[peak_count] = count - 1
            peak_count += 1
    if peak_count < 2:
        top1 = int(peaks[0]) if peak_count == 1 else -1
        return (False, peak_count, top1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0)

    top1 = -1
    top2 = -1
    for offset in range(peak_count):
        index = int(peaks[offset])
        if (
            top1 < 0
            or probabilities[index] > probabilities[top1]
            or (
                probabilities[index] == probabilities[top1]
                and index < top1
            )
        ):
            top2 = top1
            top1 = index
        elif (
            top2 < 0
            or probabilities[index] > probabilities[top2]
            or (
                probabilities[index] == probabilities[top2]
                and index < top2
            )
        ):
            top2 = index

    low = min(top1, top2)
    high = max(top1, top2)
    valley = low
    valley_density = probabilities[low]
    for index in range(low + 1, high + 1):
        if probabilities[index] < valley_density:
            valley = index
            valley_density = probabilities[index]
    lower_mass = 0.0
    for index in range(valley + 1):
        lower_mass += probabilities[index]
    upper_mass = 0.0
    for index in range(valley + 1, count):
        upper_mass += probabilities[index]
    if top1 <= valley:
        top1_mass = lower_mass
        top2_mass = upper_mass
    else:
        top1_mass = upper_mass
        top2_mass = lower_mass
    minimum_peak_density = min(probabilities[top1], probabilities[top2])
    valley_depth = (
        0.0
        if minimum_peak_density <= 0.0
        else 1.0 - valley_density / minimum_peak_density
    )
    separation = abs(top1 - top2) * grid_step
    mass_ratio = 0.0 if top1_mass <= 0.0 else top2_mass / top1_mass
    active = (
        top2_mass >= min_top2_mass
        and mass_ratio >= min_top2_to_top1_mass_ratio
        and separation >= min_peak_separation_ft
        and valley_depth >= min_valley_depth
    )
    return (
        active,
        peak_count,
        top1,
        top2,
        valley,
        top1_mass,
        top2_mass,
        mass_ratio,
        separation,
        valley_depth,
    )


def analyse_bimodality_1d(
    probabilities: np.ndarray,
    grid: np.ndarray,
    ambiguity: Mapping[str, Any],
) -> dict[str, Any]:
    grid = np.asarray(grid, dtype=np.float64)
    if len(grid) < 2:
        raise ValueError("bimodality grid requires at least two positions")
    values = bimodality_diagnostics_1d(
        np.asarray(probabilities, dtype=np.float64),
        float(grid[1] - grid[0]),
        float(ambiguity["min_peak_height"]),
        float(ambiguity["min_top2_mass"]),
        float(ambiguity["min_top2_to_top1_mass_ratio"]),
        float(ambiguity["min_peak_separation_ft"]),
        float(ambiguity["min_valley_depth"]),
    )
    (
        flag,
        peak_count,
        top1,
        top2,
        valley,
        top1_mass,
        top2_mass,
        ratio,
        separation,
        valley_depth,
    ) = values
    return {
        "bimodal_flag": bool(flag),
        "peak_count": int(peak_count),
        "top1_index": int(top1),
        "top2_index": int(top2),
        "valley_index": int(valley),
        "top1_mass": float(top1_mass),
        "top2_mass": float(top2_mass),
        "top2_to_top1_mass_ratio": float(ratio),
        "peak_separation_ft": float(separation),
        "valley_depth": float(valley_depth),
    }


# %% [markdown]
# ## 6. Ambiguity-gated exact forward-backward
#
# `provisional = predictive + parent emission` is always evaluated first.
# The exp236 detector consumes only its TVT marginal. On eligible ambiguous
# rows `cur` is reset to `predictive`; otherwise it remains `provisional`.

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_ambiguity_gated(
    emission,
    raw_gr_observed,
    dm,
    dz,
    position_step,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sigma,
    initial_rate,
    initial_rate_sigma,
    momentum,
    min_peak_height,
    min_top2_mass,
    min_top2_to_top1_mass_ratio,
    min_peak_separation_ft,
    min_valley_depth,
    gate_enabled,
):
    time_count, position_count = emission.shape
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1.0e18)

    alpha = np.full((time_count, position_count, rate_count), neg, np.float32)
    previous = np.full((position_count, rate_count), neg, np.float32)
    for position_index in range(position_count):
        delta_position = (position_index - start_p) * position_step
        initial_position_log = -0.5 * (delta_position / start_sigma) ** 2
        if initial_position_log < -60.0:
            continue
        for rate_index in range(rate_count):
            delta_rate = (
                rates[rate_index] - initial_rate
            ) / initial_rate_sigma
            previous[position_index, rate_index] = np.float32(
                initial_position_log - 0.5 * delta_rate * delta_rate
            )

    rate_updated = np.empty((position_count, rate_count), np.float32)
    predictive = np.empty((position_count, rate_count), np.float32)
    provisional = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)

    predictive_position_mass = np.zeros(
        (time_count, position_count), dtype=np.float32
    )
    provisional_position_mass = np.zeros(
        (time_count, position_count), dtype=np.float32
    )
    candidate_filtered_position_mass = np.zeros(
        (time_count, position_count), dtype=np.float32
    )
    ambiguity_active = np.zeros(time_count, dtype=np.int8)
    raw_bimodal = np.zeros(time_count, dtype=np.int8)
    peak_count = np.zeros(time_count, dtype=np.int16)
    top1_index = np.full(time_count, -1, dtype=np.int32)
    top2_index = np.full(time_count, -1, dtype=np.int32)
    valley_index = np.full(time_count, -1, dtype=np.int32)
    top1_mass = np.zeros(time_count, dtype=np.float32)
    top2_mass = np.zeros(time_count, dtype=np.float32)
    top2_to_top1_mass_ratio = np.zeros(time_count, dtype=np.float32)
    peak_separation_ft = np.zeros(time_count, dtype=np.float32)
    valley_depth = np.zeros(time_count, dtype=np.float32)

    for time_index in range(time_count):
        sigma_rate_step = sig_r * np.sqrt(dm[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((rate_count, 3), dtype=np.float64)
        for source_rate in range(rate_count):
            mean_rate_move = (
                -(1.0 - momentum)
                * rates[source_rate]
                * dm[time_index]
                / rate_step
            )
            probability_plus = max(
                0.5 * (rate_variance_cells + mean_rate_move), 1.0e-12
            )
            probability_minus = max(
                0.5 * (rate_variance_cells - mean_rate_move), 1.0e-12
            )
            move_total = probability_plus + probability_minus
            if move_total > 0.9:
                probability_plus *= 0.9 / move_total
                probability_minus *= 0.9 / move_total
            rate_log_kernel[source_rate, 0] = np.log(probability_minus)
            rate_log_kernel[source_rate, 1] = np.log(
                1.0 - probability_plus - probability_minus
            )
            rate_log_kernel[source_rate, 2] = np.log(probability_plus)

        for position_index in prange(position_count):
            for destination_rate in range(rate_count):
                best = neg
                lower = max(destination_rate - 1, 0)
                upper = min(destination_rate + 1, rate_count - 1)
                for source_rate in range(lower, upper + 1):
                    value = (
                        previous[position_index, source_rate]
                        + rate_log_kernel[
                            source_rate, destination_rate - source_rate + 1
                        ]
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for source_rate in range(lower, upper + 1):
                        total += np.exp(
                            previous[position_index, source_rate]
                            + rate_log_kernel[
                                source_rate, destination_rate - source_rate + 1
                            ]
                            - best
                        )
                    rate_updated[position_index, destination_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    rate_updated[position_index, destination_rate] = neg

        sigma_position = max(sig_p, 0.35 * position_step)
        for destination_rate in prange(rate_count):
            mean_shift = (
                rates[destination_rate] * dm[time_index] - dz[time_index]
            )
            center = int(np.floor(mean_shift / position_step + 0.5))
            position_log_kernel = np.empty(5, dtype=np.float64)
            for kernel_index in range(5):
                delta = (
                    center - 2 + kernel_index
                ) * position_step - mean_shift
                position_log_kernel[kernel_index] = (
                    -0.5 * (delta / sigma_position) ** 2
                )
            kernel_max = np.max(position_log_kernel)
            log_normalizer = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_normalizer
            for destination_position in range(position_count):
                best = neg
                for kernel_index in range(5):
                    source_position = destination_position - (
                        center - 2 + kernel_index
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
                        source_position = destination_position - (
                            center - 2 + kernel_index
                        )
                        if 0 <= source_position < position_count:
                            total += np.exp(
                                rate_updated[source_position, destination_rate]
                                + position_log_kernel[kernel_index]
                                - best
                            )
                    value = best + np.log(total)
                    predictive[destination_position, destination_rate] = np.float32(
                        value
                    )
                    provisional[
                        destination_position, destination_rate
                    ] = np.float32(
                        value + emission[time_index, destination_position]
                    )
                else:
                    predictive[destination_position, destination_rate] = neg
                    provisional[destination_position, destination_rate] = neg

        predictive_best = neg
        provisional_best = neg
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_best = max(
                    predictive_best, predictive[position_index, rate_index]
                )
                provisional_best = max(
                    provisional_best, provisional[position_index, rate_index]
                )
        predictive_total = 0.0
        provisional_total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_total += np.exp(
                    predictive[position_index, rate_index] - predictive_best
                )
                provisional_total += np.exp(
                    provisional[position_index, rate_index] - provisional_best
                )
        for position_index in range(position_count):
            predictive_mass = 0.0
            provisional_mass = 0.0
            for rate_index in range(rate_count):
                predictive_mass += np.exp(
                    predictive[position_index, rate_index] - predictive_best
                ) / predictive_total
                provisional_mass += np.exp(
                    provisional[position_index, rate_index] - provisional_best
                ) / provisional_total
            predictive_position_mass[
                time_index, position_index
            ] = np.float32(predictive_mass)
            provisional_position_mass[
                time_index, position_index
            ] = np.float32(provisional_mass)

        detector = bimodality_diagnostics_1d(
            provisional_position_mass[time_index],
            position_step,
            min_peak_height,
            min_top2_mass,
            min_top2_to_top1_mass_ratio,
            min_peak_separation_ft,
            min_valley_depth,
        )
        is_bimodal = bool(detector[0])
        raw_bimodal[time_index] = np.int8(is_bimodal)
        peak_count[time_index] = np.int16(detector[1])
        top1_index[time_index] = np.int32(detector[2])
        top2_index[time_index] = np.int32(detector[3])
        valley_index[time_index] = np.int32(detector[4])
        top1_mass[time_index] = np.float32(detector[5])
        top2_mass[time_index] = np.float32(detector[6])
        top2_to_top1_mass_ratio[time_index] = np.float32(detector[7])
        peak_separation_ft[time_index] = np.float32(detector[8])
        valley_depth[time_index] = np.float32(detector[9])
        active = bool(gate_enabled and raw_gr_observed[time_index] and is_bimodal)
        ambiguity_active[time_index] = np.int8(active)

        for position_index in range(position_count):
            candidate_mass = (
                predictive_position_mass[time_index, position_index]
                if active
                else provisional_position_mass[time_index, position_index]
            )
            candidate_filtered_position_mass[
                time_index, position_index
            ] = candidate_mass
            for rate_index in range(rate_count):
                current[position_index, rate_index] = (
                    predictive[position_index, rate_index]
                    if active
                    else provisional[position_index, rate_index]
                )
                alpha[time_index, position_index, rate_index] = current[
                    position_index, rate_index
                ]
                previous[position_index, rate_index] = current[
                    position_index, rate_index
                ]

    last_values = alpha[time_count - 1]
    last_best = np.max(last_values)
    last_total = 0.0
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            last_total += np.exp(
                last_values[position_index, rate_index] - last_best
            )
    log_likelihood = float(last_best) + np.log(last_total)

    posterior_position_mass = np.zeros(
        (time_count, position_count), dtype=np.float64
    )
    beta_next = np.zeros((position_count, rate_count), dtype=np.float32)
    values = alpha[time_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for position_index in range(position_count):
        mass = 0.0
        for rate_index in range(rate_count):
            mass += np.exp(values[position_index, rate_index] - best)
        posterior_position_mass[time_count - 1, position_index] = mass
        total += mass
    posterior_position_mass[time_count - 1] /= total

    beta_current = np.empty((position_count, rate_count), dtype=np.float32)
    beta_position = np.empty((position_count, rate_count), dtype=np.float32)
    for time_index in range(time_count - 1, 0, -1):
        sigma_rate_step = sig_r * np.sqrt(dm[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((rate_count, 3), dtype=np.float64)
        for source_rate in range(rate_count):
            mean_rate_move = (
                -(1.0 - momentum)
                * rates[source_rate]
                * dm[time_index]
                / rate_step
            )
            probability_plus = max(
                0.5 * (rate_variance_cells + mean_rate_move), 1.0e-12
            )
            probability_minus = max(
                0.5 * (rate_variance_cells - mean_rate_move), 1.0e-12
            )
            move_total = probability_plus + probability_minus
            if move_total > 0.9:
                probability_plus *= 0.9 / move_total
                probability_minus *= 0.9 / move_total
            rate_log_kernel[source_rate, 0] = np.log(probability_minus)
            rate_log_kernel[source_rate, 1] = np.log(
                1.0 - probability_plus - probability_minus
            )
            rate_log_kernel[source_rate, 2] = np.log(probability_plus)

        row_lambda = 0.0 if ambiguity_active[time_index] else 1.0
        sigma_position = max(sig_p, 0.35 * position_step)
        for destination_rate in prange(rate_count):
            mean_shift = (
                rates[destination_rate] * dm[time_index] - dz[time_index]
            )
            center = int(np.floor(mean_shift / position_step + 0.5))
            position_log_kernel = np.empty(5, dtype=np.float64)
            for kernel_index in range(5):
                delta = (
                    center - 2 + kernel_index
                ) * position_step - mean_shift
                position_log_kernel[kernel_index] = (
                    -0.5 * (delta / sigma_position) ** 2
                )
            kernel_max = np.max(position_log_kernel)
            log_normalizer = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_normalizer
            for source_position in range(position_count):
                best = neg
                for kernel_index in range(5):
                    destination_position = source_position + (
                        center - 2 + kernel_index
                    )
                    if 0 <= destination_position < position_count:
                        value = (
                            position_log_kernel[kernel_index]
                            + row_lambda
                            * emission[time_index, destination_position]
                            + beta_next[destination_position, destination_rate]
                        )
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        destination_position = source_position + (
                            center - 2 + kernel_index
                        )
                        if 0 <= destination_position < position_count:
                            total += np.exp(
                                position_log_kernel[kernel_index]
                                + row_lambda
                                * emission[time_index, destination_position]
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

        for position_index in prange(position_count):
            for source_rate in range(rate_count):
                best = neg
                lower = max(source_rate - 1, 0)
                upper = min(source_rate + 1, rate_count - 1)
                for destination_rate in range(lower, upper + 1):
                    value = (
                        rate_log_kernel[
                            source_rate, destination_rate - source_rate + 1
                        ]
                        + beta_position[position_index, destination_rate]
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for destination_rate in range(lower, upper + 1):
                        total += np.exp(
                            rate_log_kernel[
                                source_rate,
                                destination_rate - source_rate + 1,
                            ]
                            + beta_position[position_index, destination_rate]
                            - best
                        )
                    beta_current[position_index, source_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    beta_current[position_index, source_rate] = neg

        values = alpha[time_index - 1] + beta_current
        best = np.max(values)
        total = 0.0
        for position_index in range(position_count):
            mass = 0.0
            for rate_index in range(rate_count):
                mass += np.exp(values[position_index, rate_index] - best)
            posterior_position_mass[time_index - 1, position_index] = mass
            total += mass
        posterior_position_mass[time_index - 1] /= total
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                beta_next[position_index, rate_index] = beta_current[
                    position_index, rate_index
                ]

    return (
        posterior_position_mass,
        predictive_position_mass,
        provisional_position_mass,
        candidate_filtered_position_mass,
        ambiguity_active,
        raw_bimodal,
        peak_count,
        top1_index,
        top2_index,
        valley_index,
        top1_mass,
        top2_mass,
        top2_to_top1_mass_ratio,
        peak_separation_ft,
        valley_depth,
        log_likelihood,
    )


def run_ambiguity_gated_hmm(
    prepared: Mapping[str, Any],
    fixed: Mapping[str, Any],
    ambiguity: Mapping[str, Any],
    *,
    gate_enabled: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = _hmm2_ambiguity_gated(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        ~np.asarray(prepared["raw_gr_missing"], dtype=bool),
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(fixed["position_grid_step_ft"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(fixed["sig_r"]),
        float(fixed["sig_p"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        float(fixed["momentum"]),
        float(ambiguity["min_peak_height"]),
        float(ambiguity["min_top2_mass"]),
        float(ambiguity["min_top2_to_top1_mass_ratio"]),
        float(ambiguity["min_peak_separation_ft"]),
        float(ambiguity["min_valley_depth"]),
        bool(gate_enabled),
    )
    (
        posterior,
        predictive,
        provisional,
        candidate_filtered,
        active,
        raw_bimodal,
        peak_count,
        top1_index,
        top2_index,
        valley_index,
        top1_mass,
        top2_mass,
        ratio,
        separation,
        valley_depth,
        log_likelihood,
    ) = result
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = np.asarray(posterior, dtype=np.float64) @ grid
    posterior_second = np.asarray(posterior, dtype=np.float64) @ (grid**2)
    posterior_std = np.sqrt(
        np.maximum(posterior_second - posterior_mean**2, 0.0)
    )
    predictive_mean = np.asarray(predictive, dtype=np.float64) @ grid
    provisional_mean = np.asarray(provisional, dtype=np.float64) @ grid
    candidate_filtered_mean = np.asarray(candidate_filtered, dtype=np.float64) @ grid
    normalization_error = max(
        float(
            np.max(
                np.abs(
                    np.sum(np.asarray(values, dtype=np.float64), axis=1) - 1.0
                )
            )
        )
        for values in (posterior, predictive, provisional, candidate_filtered)
    )
    schedule_sha = array_bundle_sha256(
        ambiguity_active=np.asarray(active, dtype=np.int8),
        raw_bimodal=np.asarray(raw_bimodal, dtype=np.int8),
    )
    prediction_sha = array_bundle_sha256(
        posterior_mean=posterior_mean.astype(np.float32),
        posterior_std=posterior_std.astype(np.float32),
    )
    diagnostic_sha = array_bundle_sha256(
        predictive_mean=predictive_mean.astype(np.float32),
        provisional_mean=provisional_mean.astype(np.float32),
        candidate_filtered_mean=candidate_filtered_mean.astype(np.float32),
        peak_count=np.asarray(peak_count, dtype=np.int16),
        top1_index=np.asarray(top1_index, dtype=np.int32),
        top2_index=np.asarray(top2_index, dtype=np.int32),
        valley_index=np.asarray(valley_index, dtype=np.int32),
        top1_mass=np.asarray(top1_mass, dtype=np.float32),
        top2_mass=np.asarray(top2_mass, dtype=np.float32),
        ratio=np.asarray(ratio, dtype=np.float32),
        separation=np.asarray(separation, dtype=np.float32),
        valley_depth=np.asarray(valley_depth, dtype=np.float32),
    )
    return {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "predictive_mean": predictive_mean,
        "provisional_mean": provisional_mean,
        "candidate_filtered_mean": candidate_filtered_mean,
        "ambiguity_active": np.asarray(active, dtype=bool),
        "raw_bimodal": np.asarray(raw_bimodal, dtype=bool),
        "peak_count": np.asarray(peak_count, dtype=np.int16),
        "top1_index": np.asarray(top1_index, dtype=np.int32),
        "top2_index": np.asarray(top2_index, dtype=np.int32),
        "valley_index": np.asarray(valley_index, dtype=np.int32),
        "top1_mass": np.asarray(top1_mass, dtype=np.float32),
        "top2_mass": np.asarray(top2_mass, dtype=np.float32),
        "top2_to_top1_mass_ratio": np.asarray(ratio, dtype=np.float32),
        "peak_separation_ft": np.asarray(separation, dtype=np.float32),
        "valley_depth": np.asarray(valley_depth, dtype=np.float32),
        "maximum_normalization_error": normalization_error,
        "log_likelihood": float(log_likelihood),
        "schedule_sha256": schedule_sha,
        "backward_schedule_sha256": schedule_sha,
        "prediction_sha256": prediction_sha,
        "diagnostic_sha256": diagnostic_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def synthetic_no_ambiguity_parent_parity(
    fixed: Mapping[str, Any],
    ambiguity: Mapping[str, Any],
) -> dict[str, Any]:
    rows = 12
    positions = 31
    grid = 11_900.0 + np.arange(positions, dtype=np.float64) * float(
        fixed["position_grid_step_ft"]
    )
    rates = np.linspace(
        -float(fixed["rate_span"]),
        float(fixed["rate_span"]),
        int(fixed["n_rates"]),
        dtype=np.float64,
    )
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.35 * math.sin(row / 3.0)) / 0.42) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    prepared = {
        "emission_ll": emission,
        "raw_gr_missing": np.zeros(rows, dtype=bool),
        "dm": np.linspace(9.0, 21.0, rows, dtype=np.float64),
        "dz": np.linspace(-0.4, 0.7, rows, dtype=np.float64),
        "grid": grid,
        "rates": rates,
        "start_p": 14.5,
        "r0": 0.0,
    }
    parent = run_ambiguity_gated_hmm(
        prepared, fixed, ambiguity, gate_enabled=False
    )
    impossible = dict(ambiguity)
    impossible["min_peak_height"] = 2.0
    candidate = run_ambiguity_gated_hmm(
        prepared, fixed, impossible, gate_enabled=True
    )
    posterior_diff = float(
        np.max(np.abs(parent["posterior_mean"] - candidate["posterior_mean"]))
    )
    std_diff = float(
        np.max(np.abs(parent["posterior_std"] - candidate["posterior_std"]))
    )
    return {
        "posterior_mean_max_abs_diff_ft": posterior_diff,
        "posterior_std_max_abs_diff_ft": std_diff,
        "active_rows": int(np.count_nonzero(candidate["ambiguity_active"])),
        "parent_prediction_sha256": parent["prediction_sha256"],
        "candidate_prediction_sha256": candidate["prediction_sha256"],
        "pass": bool(
            posterior_diff <= 1.0e-10
            and std_diff <= 1.0e-10
            and not np.any(candidate["ambiguity_active"])
        ),
    }


# %% [markdown]
# ## 7. Target-free schedule and prediction freeze

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
    predictive_mean: np.ndarray
    provisional_mean: np.ndarray
    candidate_filtered_mean: np.ndarray
    ambiguity_active: np.ndarray
    raw_bimodal: np.ndarray
    peak_count: np.ndarray
    top1_index: np.ndarray
    top2_index: np.ndarray
    valley_index: np.ndarray
    top1_mass: np.ndarray
    top2_mass: np.ndarray
    mass_ratio: np.ndarray
    peak_separation_ft: np.ndarray
    valley_depth: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    prefix_rows: int
    schedule_sha256: str
    prediction_sha256: str
    diagnostic_sha256: str
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
    ambiguity: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, fixed)
    decoded = run_ambiguity_gated_hmm(prepared, fixed, ambiguity)
    parent = saved_parent.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    eval_id = parent_cache_ids_for_rows(well, row_idx)
    if not np.array_equal(parent["row_idx"].to_numpy(np.int64), row_idx):
        raise ValueError(f"{well}: saved parent row index does not align")
    if not np.array_equal(parent["id"].astype(str).to_numpy(), eval_id):
        raise ValueError(f"{well}: saved parent id does not align")
    frozen = FrozenWell(
        well=str(well),
        eval_id=eval_id,
        row_idx=row_idx,
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        parent_prediction=parent["parent_prediction"].to_numpy(np.float64),
        candidate_prediction=np.asarray(decoded["posterior_mean"], dtype=np.float64),
        candidate_posterior_std=np.asarray(
            decoded["posterior_std"], dtype=np.float64
        ),
        predictive_mean=np.asarray(decoded["predictive_mean"], dtype=np.float64),
        provisional_mean=np.asarray(decoded["provisional_mean"], dtype=np.float64),
        candidate_filtered_mean=np.asarray(
            decoded["candidate_filtered_mean"], dtype=np.float64
        ),
        ambiguity_active=np.asarray(decoded["ambiguity_active"], dtype=bool),
        raw_bimodal=np.asarray(decoded["raw_bimodal"], dtype=bool),
        peak_count=np.asarray(decoded["peak_count"], dtype=np.int16),
        top1_index=np.asarray(decoded["top1_index"], dtype=np.int32),
        top2_index=np.asarray(decoded["top2_index"], dtype=np.int32),
        valley_index=np.asarray(decoded["valley_index"], dtype=np.int32),
        top1_mass=np.asarray(decoded["top1_mass"], dtype=np.float32),
        top2_mass=np.asarray(decoded["top2_mass"], dtype=np.float32),
        mass_ratio=np.asarray(
            decoded["top2_to_top1_mass_ratio"], dtype=np.float32
        ),
        peak_separation_ft=np.asarray(
            decoded["peak_separation_ft"], dtype=np.float32
        ),
        valley_depth=np.asarray(decoded["valley_depth"], dtype=np.float32),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        prefix_rows=int(prepared["prefix_rows"]),
        schedule_sha256=str(decoded["schedule_sha256"]),
        prediction_sha256=str(decoded["prediction_sha256"]),
        diagnostic_sha256=str(decoded["diagnostic_sha256"]),
        maximum_normalization_error=float(decoded["maximum_normalization_error"]),
        log_likelihood=float(decoded["log_likelihood"]),
        elapsed_seconds=float(decoded["elapsed_seconds"]),
    )
    ledger.freeze(
        well,
        schedule_sha256=frozen.schedule_sha256,
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


def prediction_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
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


def schedule_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces = [
        pd.DataFrame(
            {
                "well": item.well,
                "row_idx": item.row_idx,
                "suffix_offset": np.arange(len(item.row_idx), dtype=np.int64),
                "raw_gr_observed": ~item.raw_gr_missing,
                "raw_bimodal": item.raw_bimodal,
                "ambiguity_active": item.ambiguity_active,
                "emission_lambda": np.where(item.ambiguity_active, 0.0, 1.0),
            }
        )
        for item in frozen_wells
    ]
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


def diagnostic_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces = [
        pd.DataFrame(
            {
                "well": item.well,
                "row_idx": item.row_idx,
                "suffix_offset": np.arange(len(item.row_idx), dtype=np.int64),
                "predictive_mean": item.predictive_mean,
                "provisional_mean": item.provisional_mean,
                "candidate_filtered_mean": item.candidate_filtered_mean,
                "peak_count": item.peak_count,
                "top1_index": item.top1_index,
                "top2_index": item.top2_index,
                "valley_index": item.valley_index,
                "top1_mass": item.top1_mass,
                "top2_mass": item.top2_mass,
                "top2_to_top1_mass_ratio": item.mass_ratio,
                "peak_separation_ft": item.peak_separation_ft,
                "valley_depth": item.valley_depth,
            }
        )
        for item in frozen_wells
    ]
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
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
# ## 8. Truth-late Stage 0 readout

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
    suffix["id"] = parent_cache_ids_for_rows(
        frozen.well, suffix.index.to_numpy(np.int64)
    )
    return suffix.reset_index(names="row_idx")


def well_truth_late_metrics(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    actual = truth["TVT"].to_numpy(np.float64)
    parent_error = frozen.parent_prediction - actual
    candidate_error = frozen.candidate_prediction - actual
    active = frozen.ambiguity_active
    predictive_better = (
        np.abs(frozen.predictive_mean[active] - actual[active])
        < np.abs(frozen.provisional_mean[active] - actual[active])
    )
    parent_active_sse = float(np.sum(parent_error[active] ** 2))
    candidate_active_sse = float(np.sum(candidate_error[active] ** 2))
    return {
        "well": frozen.well,
        "role": frozen.role,
        "fold": frozen.fold,
        "rows": len(actual),
        "ambiguous_rows": int(np.count_nonzero(active)),
        "ambiguous_row_fraction": float(np.mean(active)),
        "predictive_better_ambiguous_rows": int(np.count_nonzero(predictive_better)),
        "predictive_better_rate": (
            float(np.mean(predictive_better)) if len(predictive_better) else math.nan
        ),
        "parent_sse": float(np.sum(parent_error**2)),
        "candidate_sse": float(np.sum(candidate_error**2)),
        "parent_rmse_ft": float(np.sqrt(np.mean(parent_error**2))),
        "candidate_rmse_ft": float(np.sqrt(np.mean(candidate_error**2))),
        "rmse_delta_ft": float(
            np.sqrt(np.mean(candidate_error**2))
            - np.sqrt(np.mean(parent_error**2))
        ),
        "parent_ambiguous_sse": parent_active_sse,
        "candidate_ambiguous_sse": candidate_active_sse,
        "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
        "maximum_normalization_error": frozen.maximum_normalization_error,
        "hmm_elapsed_seconds": frozen.elapsed_seconds,
        "schedule_sha256": frozen.schedule_sha256,
        "prediction_sha256": frozen.prediction_sha256,
        "diagnostic_sha256": frozen.diagnostic_sha256,
    }


def ambiguous_truth_late_readout(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    actual = truth["TVT"].to_numpy(np.float64)
    active_offsets = np.flatnonzero(frozen.ambiguity_active)
    return pd.DataFrame(
        {
            "well": frozen.well,
            "role": frozen.role,
            "fold": frozen.fold,
            "row_idx": frozen.row_idx[active_offsets],
            "suffix_offset": active_offsets,
            "truth_tvt": actual[active_offsets],
            "predictive_mean": frozen.predictive_mean[active_offsets],
            "provisional_mean": frozen.provisional_mean[active_offsets],
            "parent_prediction": frozen.parent_prediction[active_offsets],
            "candidate_prediction": frozen.candidate_prediction[active_offsets],
            "predictive_better": (
                np.abs(frozen.predictive_mean[active_offsets] - actual[active_offsets])
                < np.abs(
                    frozen.provisional_mean[active_offsets] - actual[active_offsets]
                )
            ),
        }
    )


def load_persistent_episodes_after_all_freeze(
    config: Mapping[str, Any],
    persistent_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    expected = str(spec["expected_sha256"])
    if observed != expected:
        raise ValueError(f"persistent episode SHA changed: {observed} != {expected}")
    frame = pd.read_csv(path, dtype={"well": str, "episode_id": str})
    frame = frame.loc[frame["well"].isin(persistent_wells)].copy()
    ledger.record_episode_late(len(frame))
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
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    expected = str(spec["expected_sha256"])
    if observed != expected:
        raise ValueError(f"exp408 cause SHA changed: {observed} != {expected}")
    frame = pd.read_csv(path, dtype={"well": str, "episode_id": str})
    frame = frame.loc[frame["episode_id"].isin(selected_episode_ids)].copy()
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
        causes.set_index("episode_id")["cause"].astype(str).to_dict()
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
        mask = (frozen.row_idx >= start) & (frozen.row_idx < end)
        if not np.any(mask):
            raise ValueError(f"{episode.episode_id}: episode has no fixed32 rows")
        actual_lookup = truth.set_index("row_idx")["TVT"]
        actual = actual_lookup.loc[frozen.row_idx[mask]].to_numpy(np.float64)
        parent_sse = float(
            np.sum((frozen.parent_prediction[mask] - actual) ** 2)
        )
        candidate_sse = float(
            np.sum((frozen.candidate_prediction[mask] - actual) ** 2)
        )
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": well,
                "fold": frozen.fold,
                "cause": cause_lookup.get(str(episode.episode_id), "unavailable"),
                "start_row_idx": start,
                "end_row_idx_exclusive": end,
                "rows": int(np.count_nonzero(mask)),
                "ambiguous_rows": int(np.count_nonzero(frozen.ambiguity_active[mask])),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "sse_reduction_fraction": (
                    (parent_sse - candidate_sse) / parent_sse
                    if parent_sse > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["well", "start_row_idx"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 9. Technical and mechanism gates

# %%
def safe_fraction(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def sse_reduction(parent_sse: float, candidate_sse: float) -> float:
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
    parity: Mapping[str, Any],
    prediction_artifact: Mapping[str, Any],
    schedule_artifact: Mapping[str, Any],
    diagnostic_artifact: Mapping[str, Any],
    ambiguous_readout: pd.DataFrame,
    episode_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "gates.stage0_fixed32.technical")
    mechanism_config = get_nested(config, "gates.stage0_fixed32.mechanism")
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    active_rows = int(
        sum(np.count_nonzero(item.ambiguity_active) for item in frozen_wells)
    )
    active_wells = int(
        sum(bool(np.any(item.ambiguity_active)) for item in frozen_wells)
    )
    active_fraction = safe_fraction(active_rows, total_rows)
    maximum_normalization_error = max(
        item.maximum_normalization_error for item in frozen_wells
    )
    finite_rows = sum(
        int(np.isfinite(item.candidate_prediction).sum()) for item in frozen_wells
    )
    finite_coverage = safe_fraction(finite_rows, total_rows)
    runtime_projection = float(elapsed_seconds * 773.0 / 32.0)
    persistent_metrics = well_metrics.loc[well_metrics["role"].eq("persistent")]
    control_metrics = well_metrics.loc[well_metrics["role"].eq("control")]

    technical = {
        "expected_wells": len(frozen_wells)
        == int(technical_config["expected_wells"]),
        "expected_rows": total_rows == int(technical_config["expected_rows"]),
        "expected_roles": (
            manifest["role"].value_counts().to_dict()
            == {
                "persistent": int(technical_config["expected_persistent_wells"]),
                "control": int(technical_config["expected_control_wells"]),
            }
        ),
        "expected_folds": manifest["fold"].nunique()
        == int(technical_config["expected_folds"]),
        "finite_coverage": finite_coverage
        >= float(technical_config["finite_coverage_min"]),
        "posterior_normalization": maximum_normalization_error
        <= float(technical_config["posterior_normalization_max_error"]),
        "no_ambiguity_parent_parity": bool(parity["pass"])
        and float(parity["posterior_mean_max_abs_diff_ft"])
        <= float(technical_config["no_ambiguity_parent_parity_max_abs_ft"]),
        "ambiguous_activation_fraction": (
            float(technical_config["ambiguous_activation_fraction_min"])
            <= active_fraction
            <= float(technical_config["ambiguous_activation_fraction_max"])
        ),
        "ambiguous_active_wells": active_wells
        >= int(technical_config["ambiguous_active_wells_min"]),
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
        "schedule_readback_sha": (
            schedule_artifact["logical_sha256"]
            == schedule_artifact["readback_logical_sha256"]
        ),
        "diagnostic_readback_sha": (
            diagnostic_artifact["logical_sha256"]
            == diagnostic_artifact["readback_logical_sha256"]
        ),
        "runtime_projection": runtime_projection
        <= float(technical_config["projected_stage1_runtime_seconds_max"]),
        "peak_rss": peak_rss_gb()
        <= float(technical_config["peak_rss_gb_max"]),
    }

    predictive_better_rate = (
        float(ambiguous_readout["predictive_better"].mean())
        if len(ambiguous_readout)
        else math.nan
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        frame = ambiguous_readout.loc[ambiguous_readout["fold"].eq(fold)]
        rate = (
            float(frame["predictive_better"].mean())
            if len(frame)
            else math.nan
        )
        fold_rows.append(
            {
                "fold": fold,
                "ambiguous_rows": len(frame),
                "predictive_better_rate": rate,
                "positive": bool(math.isfinite(rate) and rate > 0.5),
            }
        )
    predictive_positive_folds = sum(row["positive"] for row in fold_rows)

    parent_ambiguous_sse = float(well_metrics["parent_ambiguous_sse"].sum())
    candidate_ambiguous_sse = float(
        well_metrics["candidate_ambiguous_sse"].sum()
    )
    ambiguous_sse_reduction = sse_reduction(
        parent_ambiguous_sse, candidate_ambiguous_sse
    )
    parent_episode_sse = float(episode_readout["parent_sse"].sum())
    candidate_episode_sse = float(episode_readout["candidate_sse"].sum())
    episode_sse_reduction = sse_reduction(
        parent_episode_sse, candidate_episode_sse
    )
    persistent_improved_wells = int(
        (persistent_metrics["rmse_delta_ft"] < 0.0).sum()
    )
    persistent_fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        frame = episode_readout.loc[episode_readout["fold"].eq(fold)]
        parent_sse = float(frame["parent_sse"].sum())
        candidate_sse = float(frame["candidate_sse"].sum())
        reduction = sse_reduction(parent_sse, candidate_sse)
        persistent_fold_rows.append(
            {
                "fold": fold,
                "episodes": len(frame),
                "sse_reduction_fraction": reduction,
                "improving": bool(math.isfinite(reduction) and reduction > 0.0),
            }
        )
    persistent_improving_folds = sum(
        row["improving"] for row in persistent_fold_rows
    )
    control_parent_sse = float(control_metrics["parent_sse"].sum())
    control_candidate_sse = float(control_metrics["candidate_sse"].sum())
    control_rows = int(control_metrics["rows"].sum())
    control_pooled_delta = (
        math.sqrt(control_candidate_sse / control_rows)
        - math.sqrt(control_parent_sse / control_rows)
    )
    control_by_well_p95 = float(
        np.quantile(control_metrics["rmse_delta_ft"].to_numpy(np.float64), 0.95)
    )

    mechanism = {
        "predictive_better_rate_on_ambiguous_rows": (
            math.isfinite(predictive_better_rate)
            and predictive_better_rate
            >= float(
                mechanism_config[
                    "predictive_better_rate_on_ambiguous_rows_min"
                ]
            )
        ),
        "predictive_better_positive_folds": predictive_positive_folds
        >= int(mechanism_config["predictive_better_positive_folds_min"]),
        "ambiguous_row_sse_reduction": (
            math.isfinite(ambiguous_sse_reduction)
            and ambiguous_sse_reduction
            >= float(
                mechanism_config["ambiguous_row_sse_reduction_min_fraction"]
            )
        ),
        "persistent_episode_sse_reduction": (
            math.isfinite(episode_sse_reduction)
            and episode_sse_reduction
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
        "matched_control_pooled_rmse_delta": control_pooled_delta
        <= float(mechanism_config["matched_control_pooled_rmse_delta_max_ft"]),
        "matched_control_by_well_delta_p95": control_by_well_p95
        <= float(
            mechanism_config["matched_control_by_well_delta_p95_max_ft"]
        ),
    }
    diagnostics = {
        "total_rows": total_rows,
        "ambiguous_rows": active_rows,
        "ambiguous_activation_fraction": active_fraction,
        "ambiguous_active_wells": active_wells,
        "maximum_normalization_error": maximum_normalization_error,
        "finite_coverage": finite_coverage,
        "predictive_better_rate_on_ambiguous_rows": predictive_better_rate,
        "predictive_better_by_fold": fold_rows,
        "predictive_better_positive_folds": predictive_positive_folds,
        "ambiguous_row_sse_reduction_fraction": ambiguous_sse_reduction,
        "persistent_episode_sse_reduction_fraction": episode_sse_reduction,
        "persistent_improved_wells": persistent_improved_wells,
        "persistent_episode_by_fold": persistent_fold_rows,
        "persistent_improving_folds": persistent_improving_folds,
        "matched_control_pooled_rmse_delta_ft": control_pooled_delta,
        "matched_control_by_well_delta_p95_ft": control_by_well_p95,
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
    if os.environ.get("EXP440_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp440 Stage 0 must run on Kaggle CPU")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config, require_run_authorization=True
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha = hashlib.sha256(
        stable_json_bytes(scientific_contract)
    ).hexdigest()
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    ledger = LeakageLedger(expected_wells=32)
    wells, scope_input = load_fixed32_scope(config, ledger)
    parent, parent_input = load_saved_parent_predictions(config, set(wells), ledger)
    fixed = get_nested(config, "model.fixed_from_exp209")
    ambiguity = get_nested(config, "model.ambiguity_contract")
    parity = synthetic_no_ambiguity_parent_parity(fixed, ambiguity)
    if not parity["pass"]:
        raise RuntimeError(f"synthetic no-ambiguity parity failed: {parity}")

    raw_dir = train_data_dir(config)
    parent_groups = parent.groupby("well", sort=False).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    for well_index, well in enumerate(wells, start=1):
        if well not in parent_groups:
            raise ValueError(f"{well}: saved parent rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            saved_parent=parent.iloc[parent_groups[well]].copy(),
            fixed=fixed,
            ambiguity=ambiguity,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(f"Stage 0 runtime hard guard exceeded: {elapsed}")
        if peak_rss_gb() > hard_rss:
            raise MemoryError(f"Stage 0 RSS hard guard exceeded: {peak_rss_gb()}")
        print(
            json.dumps(
                {
                    "event": "exp440_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "ambiguous_rows": int(
                        np.count_nonzero(frozen.ambiguity_active)
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
    schedule = schedule_frame(frozen_wells)
    diagnostics = diagnostic_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    schedule_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_ambiguity_schedule.csv.gz",
        schedule,
    )
    diagnostic_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_target_free_diagnostics.csv.gz",
        diagnostics,
    )
    for label, artifact in (
        ("prediction", prediction_artifact),
        ("schedule", schedule_artifact),
        ("diagnostic", diagnostic_artifact),
    ):
        if artifact["logical_sha256"] != artifact["readback_logical_sha256"]:
            raise RuntimeError(f"{label} readback SHA mismatch")

    manifest, manifest_input = load_fixed32_identity_after_all_freeze(
        config, ledger
    )
    attach_late_identity(frozen_wells, manifest)
    frozen_by_well = {item.well: item for item in frozen_wells}
    truth_by_well: dict[str, pd.DataFrame] = {}
    well_rows: list[dict[str, Any]] = []
    ambiguous_pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        truth = load_truth_after_all_freeze(item, raw_dir, ledger)
        truth_by_well[item.well] = truth
        well_rows.append(well_truth_late_metrics(item, truth))
        ambiguous_pieces.append(ambiguous_truth_late_readout(item, truth))
    well_metrics = pd.DataFrame(well_rows).sort_values(
        ["fold", "role", "well"], kind="mergesort"
    )
    ambiguous_readout = pd.concat(ambiguous_pieces, ignore_index=True)
    persistent_wells = set(
        manifest.loc[manifest["role"].eq("persistent"), "well"].astype(str)
    )
    episodes, episode_input = load_persistent_episodes_after_all_freeze(
        config, persistent_wells, ledger
    )
    causes, cause_input = load_episode_causes_after_all_freeze(
        config, set(episodes["episode_id"].astype(str)), ledger
    )
    episode_readout = episode_truth_late_readout(
        episodes, causes, frozen_by_well, truth_by_well
    )

    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv", well_metrics
    )
    ambiguous_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_ambiguous_truth_late_readout.csv",
        ambiguous_readout,
    )
    episode_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_episode_truth_late_readout.csv",
        episode_readout,
    )
    elapsed = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        manifest=manifest,
        frozen_wells=frozen_wells,
        parity=parity,
        prediction_artifact=prediction_artifact,
        schedule_artifact=schedule_artifact,
        diagnostic_artifact=diagnostic_artifact,
        ambiguous_readout=ambiguous_readout,
        episode_readout=episode_readout,
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
            "truth_rows_after_all_freeze": ledger.truth_rows_after_all_freeze,
            "episode_rows_after_all_freeze": ledger.episode_rows_after_all_freeze,
            "cause_rows_after_all_freeze": ledger.cause_rows_after_all_freeze,
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    eligible = bool(gates["stage1_eligible_pending_separate_user_approval"])
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "stage0_all_gates_pass_pending_separate_stage1_approval"
        if eligible
        else "stage0_fail_closed",
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "no_ambiguity_parent_parity": parity,
        "gates": gates,
        "prediction_manifest_sha256": combined_well_sha(
            frozen_wells, "prediction_sha256"
        ),
        "schedule_manifest_sha256": combined_well_sha(
            frozen_wells, "schedule_sha256"
        ),
        "diagnostic_manifest_sha256": combined_well_sha(
            frozen_wells, "diagnostic_sha256"
        ),
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
            "cpu_only": True,
            "numba_threads": int(get_nested(config, "runtime.numba_num_threads")),
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "ambiguity_schedule": schedule_artifact,
            "target_free_diagnostics": diagnostic_artifact,
            "well_metrics": well_artifact,
            "ambiguous_truth_late_readout": ambiguous_artifact,
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
        output / f"{EXPERIMENT_NAME}_stage0_summary.json", summary
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
# ## 11. Full-OOF deterministic sharding and truth-late merge
#
# The user explicitly authorized a full-well confirmation after the fixed32
# gate failed. The scientific candidate is unchanged. Four LPT shards keep
# each Kaggle CPU run below the existing nine-hour hard guard, and each well
# is decoded exactly once. Fold, truth, hidden-like role, and error values are
# attached only after all four target-free outputs have been strictly merged
# and frozen.

# %%
def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def build_stage1_raw_manifest(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.removesuffix("__horizontal_well.csv")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(typewell_path)
        visible = pd.read_csv(horizontal_path, usecols=["TVT_input"])
        rows.append(
            {
                "well": str(well),
                "suffix_rows": int(
                    pd.to_numeric(
                        visible["TVT_input"], errors="coerce"
                    ).isna().sum()
                ),
                "horizontal_raw_sha256": sha256_file(horizontal_path),
                "typewell_raw_sha256": sha256_file(typewell_path),
            }
        )
    manifest = (
        pd.DataFrame(rows)
        .sort_values("well", kind="mergesort")
        .reset_index(drop=True)
    )
    identity_sha = dataframe_content_sha256(
        manifest.rename(columns={"well": "well_id"}),
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    if (
        len(manifest) != int(get_nested(config, "validation.expected_wells"))
        or int(manifest["suffix_rows"].sum())
        != int(get_nested(config, "validation.expected_rows"))
        or manifest["well"].duplicated().any()
        or identity_sha
        != str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    ):
        raise ValueError("exp440 full raw well identity or suffix coverage mismatch")
    manifest.attrs["raw_well_identity_sha256"] = identity_sha
    return manifest


def assign_stage1_lpt_shards(
    manifest: pd.DataFrame,
    shard_count: int = STAGE1_SHARD_COUNT,
) -> pd.DataFrame:
    if (
        shard_count <= 0
        or manifest["well"].astype(str).duplicated().any()
        or not {"well", "suffix_rows"}.issubset(manifest.columns)
    ):
        raise ValueError("invalid exp440 LPT manifest")
    loads = [0] * shard_count
    assignments: dict[str, int] = {}
    ordered = manifest.sort_values(
        ["suffix_rows", "well"],
        ascending=[False, True],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        shard_index = min(
            range(shard_count),
            key=lambda index: (loads[index], index),
        )
        assignments[str(row.well)] = shard_index
        loads[shard_index] += int(row.suffix_rows)
    result = manifest.copy()
    result["shard_index"] = (
        result["well"].astype(str).map(assignments).astype(np.int8)
    )
    result = result.sort_values("well", kind="mergesort").reset_index(drop=True)
    result.attrs["shard_suffix_rows"] = {
        str(index): int(value) for index, value in enumerate(loads)
    }
    return result


def _stage1_artifact_file(root: Path, filename: str) -> Path:
    candidates = [root / filename, root / "artifacts" / filename]
    candidates.extend(sorted(root.glob(f"**/{filename}")))
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} below {root}; found={unique}"
        )
    return unique[0]


def _stage1_output_name(shard_index: int, kind: str, suffix: str) -> str:
    return f"{EXPERIMENT_NAME}_stage1_shard{shard_index}_{kind}.{suffix}"


def run_stage1_shard(
    config: Mapping[str, Any],
    shard_index: int,
) -> dict[str, Any]:
    require_kaggle_runtime()
    os.environ["EXP440_STAGE"] = "stage1_shard"
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha = hashlib.sha256(
        stable_json_bytes(scientific_contract)
    ).hexdigest()
    if shard_index not in range(STAGE1_SHARD_COUNT):
        raise ValueError("exp440 Stage 1 shard index must be in [0, 3]")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    started = time.perf_counter()
    raw_dir = train_data_dir(config)
    full_manifest = assign_stage1_lpt_shards(
        build_stage1_raw_manifest(config, raw_dir)
    )
    selected = full_manifest.loc[
        full_manifest["shard_index"].eq(shard_index)
    ].copy()
    wells = selected["well"].astype(str).tolist()
    if not wells:
        raise ValueError(f"exp440 Stage 1 shard {shard_index} is empty")
    ledger = LeakageLedger(expected_wells=len(wells))
    ledger.record_scope(len(wells))
    parent, parent_input = load_saved_parent_predictions(
        config,
        set(wells),
        ledger,
    )
    if len(parent) != int(selected["suffix_rows"].sum()):
        raise ValueError("saved parent shard row coverage mismatch")
    parity = synthetic_no_ambiguity_parent_parity(
        get_nested(config, "model.fixed_from_exp209"),
        get_nested(config, "model.ambiguity_contract"),
    )
    if not parity["pass"]:
        raise RuntimeError(f"synthetic no-ambiguity parity failed: {parity}")

    parent_groups = parent.groupby("well", sort=False).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    for well_index, well in enumerate(wells, start=1):
        if well not in parent_groups:
            raise ValueError(f"{well}: saved parent rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            saved_parent=parent.iloc[parent_groups[well]].copy(),
            fixed=get_nested(config, "model.fixed_from_exp209"),
            ambiguity=get_nested(config, "model.ambiguity_contract"),
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(
                f"Stage 1 shard runtime hard guard exceeded: {elapsed}"
            )
        if peak_rss_gb() > hard_rss:
            raise MemoryError(
                f"Stage 1 shard RSS hard guard exceeded: {peak_rss_gb()}"
            )
        print(
            json.dumps(
                {
                    "event": "exp440_stage1_shard_progress",
                    "shard_index": shard_index,
                    "well_index": well_index,
                    "well_count": len(wells),
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "ambiguous_rows": int(
                        np.count_nonzero(frozen.ambiguity_active)
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
        raise RuntimeError("not all exp440 Stage 1 shard wells were frozen")

    predictions = prediction_frame(frozen_wells)
    schedule = schedule_frame(frozen_wells)
    diagnostics = diagnostic_frame(frozen_wells)
    audit = pd.DataFrame(
        [
            {
                "well": item.well,
                "status": "ok",
                "rows": len(item.row_idx),
                "prefix_rows": item.prefix_rows,
                "ambiguous_rows": int(
                    np.count_nonzero(item.ambiguity_active)
                ),
                "maximum_normalization_error": (
                    item.maximum_normalization_error
                ),
                "hmm_elapsed_seconds": item.elapsed_seconds,
                "schedule_sha256": item.schedule_sha256,
                "prediction_sha256": item.prediction_sha256,
                "diagnostic_sha256": item.diagnostic_sha256,
            }
            for item in frozen_wells
        ]
    ).sort_values("well", kind="mergesort")
    if (
        len(predictions) != int(selected["suffix_rows"].sum())
        or predictions["well"].nunique() != len(selected)
        or len(audit) != len(selected)
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError(f"exp440 Stage 1 shard {shard_index} coverage mismatch")

    output = artifacts_dir()
    prediction_path = output / _stage1_output_name(
        shard_index, "predictions", "csv.gz"
    )
    schedule_path = output / _stage1_output_name(
        shard_index, "ambiguity_schedule", "csv.gz"
    )
    diagnostic_path = output / _stage1_output_name(
        shard_index, "target_free_diagnostics", "csv.gz"
    )
    audit_path = output / _stage1_output_name(
        shard_index, "well_audit", "csv"
    )
    manifest_path = output / _stage1_output_name(
        shard_index, "well_manifest", "csv"
    )
    contract_path = (
        output
        / f"{EXPERIMENT_NAME}_stage1_shard{shard_index}_scientific_contract.json"
    )
    prediction_artifact = write_deterministic_gzip_csv(
        prediction_path, predictions
    )
    schedule_artifact = write_deterministic_gzip_csv(schedule_path, schedule)
    diagnostic_artifact = write_deterministic_gzip_csv(
        diagnostic_path, diagnostics
    )
    audit_artifact = write_csv(audit_path, audit)
    manifest_artifact = write_csv(manifest_path, selected)
    contract_artifact = write_json(contract_path, scientific_contract)
    for label, artifact in (
        ("prediction", prediction_artifact),
        ("schedule", schedule_artifact),
        ("diagnostic", diagnostic_artifact),
    ):
        if artifact["logical_sha256"] != artifact["readback_logical_sha256"]:
            raise RuntimeError(f"Stage 1 shard {label} readback SHA mismatch")
    elapsed = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage1_candidate_shard",
        "status": "complete",
        "shard_index": shard_index,
        "shard_count": STAGE1_SHARD_COUNT,
        "scientific_contract_sha256": scientific_contract_sha,
        "execution_contract": execution_contract,
        "counts": {
            "wells": len(selected),
            "rows": len(predictions),
            "scientific_variants": 1,
            "candidate_hmm_well_runs": len(selected),
            "parent_control_hmm_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_ml_folds": 0,
            "boosters": 0,
            "fitted_models": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
        "input": {
            "raw_train_dir": str(raw_dir),
            "raw_well_identity_sha256": full_manifest.attrs[
                "raw_well_identity_sha256"
            ],
            "saved_exp209_control": parent_input,
        },
        "frozen": {
            "prediction_artifact": prediction_artifact,
            "schedule_artifact": schedule_artifact,
            "diagnostic_artifact": diagnostic_artifact,
            "prediction_manifest_sha256": combined_well_sha(
                frozen_wells, "prediction_sha256"
            ),
            "schedule_manifest_sha256": combined_well_sha(
                frozen_wells, "schedule_sha256"
            ),
            "diagnostic_manifest_sha256": combined_well_sha(
                frozen_wells, "diagnostic_sha256"
            ),
            "forbidden_reads_before_all_freeze": (
                ledger.forbidden_reads_before_all_freeze
            ),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_rss_gb(),
            "hard_limit_seconds": hard_runtime,
            "peak_rss_limit_gb": hard_rss,
            "versions": runtime_versions(),
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "ambiguity_schedule": schedule_artifact,
            "target_free_diagnostics": diagnostic_artifact,
            "well_audit": audit_artifact,
            "well_manifest": manifest_artifact,
            "scientific_contract": contract_artifact,
        },
        "inference": False,
        "submission": False,
    }
    summary_path = (
        output / f"{EXPERIMENT_NAME}_stage1_shard{shard_index}_summary.json"
    )
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


def resolve_stage1_shard_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [
        Path(str(value))
        for value in (
            get_nested(config, "execution.stage1_merge_shard_dirs") or []
        )
    ]
    if len(roots) != STAGE1_SHARD_COUNT:
        raise ValueError("exp440 Stage 1 merge requires four ordered shard roots")
    return roots


def merge_stage1_target_free_outputs(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    scientific_contract_sha = hashlib.sha256(
        stable_json_bytes(validate_scientific_contract(config))
    ).hexdigest()
    predictions: list[pd.DataFrame] = []
    schedules: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for shard_index, root in enumerate(resolve_stage1_shard_roots(config)):
        summary_path = _stage1_artifact_file(
            root,
            f"{EXPERIMENT_NAME}_stage1_shard{shard_index}_summary.json",
        )
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("stage") != "stage1_candidate_shard"
            or summary.get("status") != "complete"
            or int(summary.get("shard_index", -1)) != shard_index
            or int(summary.get("shard_count", -1)) != STAGE1_SHARD_COUNT
            or summary.get("scientific_contract_sha256")
            != scientific_contract_sha
        ):
            raise ValueError(f"exp440 Stage 1 shard {shard_index} contract mismatch")
        prediction_path = _stage1_artifact_file(
            root,
            _stage1_output_name(shard_index, "predictions", "csv.gz"),
        )
        schedule_path = _stage1_artifact_file(
            root,
            _stage1_output_name(
                shard_index, "ambiguity_schedule", "csv.gz"
            ),
        )
        diagnostic_path = _stage1_artifact_file(
            root,
            _stage1_output_name(
                shard_index, "target_free_diagnostics", "csv.gz"
            ),
        )
        audit_path = _stage1_artifact_file(
            root,
            _stage1_output_name(shard_index, "well_audit", "csv"),
        )
        manifest_path = _stage1_artifact_file(
            root,
            _stage1_output_name(shard_index, "well_manifest", "csv"),
        )
        prediction = pd.read_csv(
            prediction_path,
            dtype={"id": str, "well": str},
            float_precision="round_trip",
        )
        schedule = pd.read_csv(
            schedule_path,
            dtype={"well": str},
            float_precision="round_trip",
        )
        diagnostic = pd.read_csv(
            diagnostic_path,
            dtype={"well": str},
            float_precision="round_trip",
        )
        audit = pd.read_csv(audit_path, dtype={"well": str})
        manifest = pd.read_csv(manifest_path, dtype={"well": str})
        frozen = summary["frozen"]
        if (
            logical_frame_sha256(prediction)
            != frozen["prediction_artifact"]["logical_sha256"]
            or logical_frame_sha256(schedule)
            != frozen["schedule_artifact"]["logical_sha256"]
            or logical_frame_sha256(diagnostic)
            != frozen["diagnostic_artifact"]["logical_sha256"]
        ):
            raise ValueError(f"exp440 Stage 1 shard {shard_index} logical SHA mismatch")
        if (
            not manifest["shard_index"].astype(int).eq(shard_index).all()
            or int(manifest["suffix_rows"].sum()) != len(prediction)
            or audit["well"].nunique() != len(manifest)
        ):
            raise ValueError(
                f"exp440 Stage 1 shard {shard_index} manifest mismatch"
            )
        predictions.append(prediction)
        schedules.append(schedule)
        diagnostics.append(diagnostic)
        audits.append(audit)
        manifests.append(manifest)
        summaries.append(summary)

    prediction = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    schedule = (
        pd.concat(schedules, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    diagnostic = (
        pd.concat(diagnostics, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.concat(audits, ignore_index=True)
        .sort_values("well", kind="mergesort")
        .reset_index(drop=True)
    )
    manifest = (
        pd.concat(manifests, ignore_index=True)
        .sort_values("well", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(prediction) != expected_rows
        or prediction["well"].nunique() != expected_wells
        or prediction["id"].duplicated().any()
        or prediction.duplicated(["well", "row_idx"]).any()
        or len(schedule) != expected_rows
        or schedule.duplicated(["well", "row_idx"]).any()
        or len(diagnostic) != expected_rows
        or diagnostic.duplicated(["well", "row_idx"]).any()
        or len(audit) != expected_wells
        or audit["well"].duplicated().any()
        or not audit["status"].eq("ok").all()
        or len(manifest) != expected_wells
        or manifest["well"].duplicated().any()
        or int(manifest["suffix_rows"].sum()) != expected_rows
        or sum(
            int(summary["counts"]["candidate_hmm_well_runs"])
            for summary in summaries
        )
        != int(get_nested(config, "execution.stage1_candidate_hmm_well_runs"))
    ):
        raise ValueError("exp440 strict Stage 1 shard merge coverage mismatch")
    key_values = prediction[["well", "row_idx"]].to_numpy()
    if (
        not np.array_equal(
            key_values, schedule[["well", "row_idx"]].to_numpy()
        )
        or not np.array_equal(
            key_values, diagnostic[["well", "row_idx"]].to_numpy()
        )
    ):
        raise ValueError("exp440 Stage 1 merged row order mismatch")

    output = artifacts_dir()
    merged_artifacts = (
        (
            prediction,
            output / f"{EXPERIMENT_NAME}_stage1_predictions.csv.gz",
        ),
        (
            schedule,
            output / f"{EXPERIMENT_NAME}_stage1_ambiguity_schedule.csv.gz",
        ),
        (
            diagnostic,
            output
            / f"{EXPERIMENT_NAME}_stage1_target_free_diagnostics.csv.gz",
        ),
    )
    for frame, path in merged_artifacts:
        report = write_deterministic_gzip_csv(path, frame)
        if report["logical_sha256"] != report["readback_logical_sha256"]:
            raise RuntimeError(f"exp440 Stage 1 merged readback SHA failed: {path}")
    write_csv(
        output / f"{EXPERIMENT_NAME}_stage1_well_audit.csv",
        audit,
    )
    write_csv(
        output / f"{EXPERIMENT_NAME}_stage1_well_manifest.csv",
        manifest,
    )
    for row in audit.itertuples(index=False):
        ledger.freeze(
            str(row.well),
            schedule_sha256=str(row.schedule_sha256),
            prediction_sha256=str(row.prediction_sha256),
            diagnostic_sha256=str(row.diagnostic_sha256),
        )
    if not ledger.all_frozen:
        raise RuntimeError("exp440 Stage 1 merge did not freeze all wells")
    return prediction, schedule, diagnostic, audit, manifest, summaries


def load_stage1_truth_fold_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.fold_assignment")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(value) for value in spec["candidates"]],
        patterns=[f"**/{spec['filename']}"],
    )
    observed_sha = sha256_decompressed_csv(path)
    if observed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp440 Stage 1 fold/truth decompressed SHA mismatch")
    columns = [str(value) for value in spec["truth_columns"]]
    if set(columns) != {"well_id", "row_idx", "tvt_true", "fold"}:
        raise ValueError("exp440 Stage 1 late truth/fold allowlist changed")
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    frame["row_idx"] = pd.to_numeric(
        frame["row_idx"], errors="raise"
    ).astype(np.int64)
    frame["fold"] = pd.to_numeric(
        frame["fold"], errors="raise"
    ).astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(
        frame["tvt_true"], errors="raise"
    ).astype(np.float64)
    frame = frame.rename(columns={"well_id": "well"}).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    ledger.record_role_fold_late(len(frame))
    ledger.record_truth_late(len(frame))
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or frame["well"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
        or frame.duplicated(["well", "row_idx"]).any()
        or sorted(frame["fold"].unique().tolist())
        != [int(value) for value in get_nested(config, "validation.expected_folds")]
        or not np.isfinite(frame["tvt_true"].to_numpy(np.float64)).all()
    ):
        raise ValueError("exp440 Stage 1 late truth/fold coverage mismatch")
    return frame, {
        "path": str(path),
        "decompressed_sha256": observed_sha,
        "rows": len(frame),
    }


def load_stage1_hidden_roles_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed_sha = sha256_file(path)
    if observed_sha != str(spec["expected_sha256"]):
        raise ValueError("exp440 Stage 1 hidden-like SHA mismatch")
    role_columns = {
        str(scope): str(column)
        for scope, column in spec["role_columns"].items()
    }
    frame = pd.read_csv(
        path,
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    ).rename(columns={"well_id": "well"})
    ledger.record_role_fold_late(len(frame))
    if (
        len(frame) != int(get_nested(config, "validation.expected_wells"))
        or frame["well"].duplicated().any()
    ):
        raise ValueError("exp440 Stage 1 hidden-like well coverage mismatch")
    for scope, column in role_columns.items():
        observed_counts = {
            str(key): int(value)
            for key, value in frame[column]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }
        expected_counts = {
            str(key): int(value)
            for key, value in spec["expected_role_counts"][scope].items()
        }
        if observed_counts != expected_counts:
            raise ValueError(f"exp440 hidden-like role counts changed: {scope}")
        frame[scope] = frame[column].astype(str).eq("valid")
    return frame[["well", *role_columns]], {
        "path": str(path),
        "sha256": observed_sha,
        "rows": len(frame),
    }


def load_stage1_md_since_after_freeze(
    raw_dir: Path,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for well in manifest["well"].astype(str):
        horizontal = pd.read_csv(
            raw_dir / f"{well}__horizontal_well.csv",
            usecols=["MD", "TVT_input"],
        )
        visible = horizontal["TVT_input"].notna().to_numpy(bool)
        if not visible.any():
            raise ValueError(f"{well}: no known TVT prefix for md_since")
        suffix = ~visible
        last_known_md = float(
            pd.to_numeric(
                horizontal.loc[visible, "MD"], errors="raise"
            ).iloc[-1]
        )
        row_idx = horizontal.index.to_numpy(np.int64)[suffix]
        md = pd.to_numeric(
            horizontal.loc[suffix, "MD"], errors="raise"
        ).to_numpy(np.float64)
        pieces.append(
            pd.DataFrame(
                {
                    "well": well,
                    "row_idx": row_idx,
                    "md_since": md - last_known_md,
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(frame) != int(manifest["suffix_rows"].sum())
        or not np.isfinite(frame["md_since"].to_numpy(np.float64)).all()
    ):
        raise ValueError("exp440 Stage 1 md_since coverage mismatch")
    return frame


def build_stage1_late_readout(
    config: Mapping[str, Any],
    prediction: pd.DataFrame,
    schedule: pd.DataFrame,
    manifest: pd.DataFrame,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth, truth_input = load_stage1_truth_fold_after_freeze(config, ledger)
    hidden, hidden_input = load_stage1_hidden_roles_after_freeze(config, ledger)
    md_since = load_stage1_md_since_after_freeze(
        train_data_dir(config),
        manifest,
    )
    prediction_columns = [
        "well",
        "row_idx",
        "parent_prediction",
        "candidate_prediction",
    ]
    schedule_columns = [
        "well",
        "row_idx",
        "raw_gr_observed",
        "ambiguity_active",
    ]
    frame = prediction[prediction_columns].merge(
        schedule[schedule_columns],
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        truth,
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        md_since,
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        hidden,
        on="well",
        how="left",
        validate="many_to_one",
    ).sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    for column in (
        "raw_gr_observed",
        "ambiguity_active",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ):
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype(bool)
    frame["raw_gr_missing"] = ~frame["raw_gr_observed"]
    frame["well_missing_fraction"] = frame.groupby(
        "well", sort=False
    )["raw_gr_missing"].transform("mean")
    numeric_columns = [
        "parent_prediction",
        "candidate_prediction",
        "tvt_true",
        "md_since",
        "well_missing_fraction",
    ]
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or frame["well"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
        or frame.duplicated(["well", "row_idx"]).any()
        or not np.isfinite(
            frame[numeric_columns].to_numpy(np.float64)
        ).all()
    ):
        raise ValueError("exp440 Stage 1 late readout coverage mismatch")
    return frame, {
        "truth_attachment": "after_four_shard_prediction_schedule_diagnostic_freeze",
        "truth_fold_input": truth_input,
        "hidden_like_input": hidden_input,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "forbidden_reads_before_all_freeze": (
            ledger.forbidden_reads_before_all_freeze
        ),
    }


def _stage1_metric_row(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scope: str,
) -> dict[str, Any]:
    if not bool(mask.any()):
        raise ValueError(f"exp440 Stage 1 scope {scope} has zero rows")
    selected = frame.loc[mask]
    truth = selected["tvt_true"].to_numpy(np.float64)
    parent = selected["parent_prediction"].to_numpy(np.float64)
    candidate = selected["candidate_prediction"].to_numpy(np.float64)
    parent_rmse = float(np.sqrt(np.mean((parent - truth) ** 2)))
    candidate_rmse = float(np.sqrt(np.mean((candidate - truth) ** 2)))
    return {
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well"].nunique()),
        "parent_rmse_ft": parent_rmse,
        "candidate_rmse_ft": candidate_rmse,
        "rmse_delta_candidate_minus_parent_ft": candidate_rmse - parent_rmse,
        "improvement_ft": parent_rmse - candidate_rmse,
    }


def build_stage1_metrics(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool))
    ]
    for fold in [int(value) for value in get_nested(config, "validation.expected_folds")]:
        scopes.append(
            (f"fold_{fold}", frame["fold"].to_numpy(np.int64) == fold)
        )
    scopes.extend(
        [
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", frame["raw_gr_missing"].to_numpy(bool)),
            (
                "high_missing_fraction",
                frame["well_missing_fraction"].to_numpy(np.float64)
                >= float(
                    get_nested(
                        config, "validation.high_missing_fraction_min"
                    )
                ),
            ),
            (
                "md_1000_plus",
                frame["md_since"].to_numpy(np.float64) >= 1000.0,
            ),
            (
                "hidden_like_spatial",
                frame["hidden_like_spatial"].to_numpy(bool),
            ),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    metrics = pd.DataFrame(
        [_stage1_metric_row(frame, mask, scope) for scope, mask in scopes]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, selected in frame.groupby("well", sort=True):
        truth = selected["tvt_true"].to_numpy(np.float64)
        parent = selected["parent_prediction"].to_numpy(np.float64)
        candidate = selected["candidate_prediction"].to_numpy(np.float64)
        parent_rmse = float(np.sqrt(np.mean((parent - truth) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate - truth) ** 2)))
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(selected),
                "fold": int(selected["fold"].iloc[0]),
                "parent_rmse_ft": parent_rmse,
                "candidate_rmse_ft": candidate_rmse,
                "rmse_delta_candidate_minus_parent_ft": (
                    candidate_rmse - parent_rmse
                ),
            }
        )
    return metrics, pd.DataFrame(by_well_rows)


def evaluate_stage1_gates(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    audit: pd.DataFrame,
    summaries: Sequence[Mapping[str, Any]],
    ledger: LeakageLedger,
) -> dict[str, Any]:
    gate_config = get_nested(config, "gates.stage1_full_oof")
    overall = metrics.loc[metrics["scope"].eq("overall")].iloc[0]
    fold_metrics = metrics.loc[
        metrics["scope"].str.startswith("fold_")
    ]
    positive_folds = int((fold_metrics["improvement_ft"] > 0.0).sum())
    active = frame["ambiguity_active"].to_numpy(bool)
    truth = frame["tvt_true"].to_numpy(np.float64)
    parent = frame["parent_prediction"].to_numpy(np.float64)
    candidate = frame["candidate_prediction"].to_numpy(np.float64)
    parent_active_sse = float(np.sum((parent[active] - truth[active]) ** 2))
    candidate_active_sse = float(
        np.sum((candidate[active] - truth[active]) ** 2)
    )
    active_sse_reduction = sse_reduction(
        parent_active_sse, candidate_active_sse
    )
    required_scopes = [
        str(value) for value in gate_config["required_nonworse_scopes"]
    ]
    nonworse = {
        scope: bool(
            float(
                metrics.loc[
                    metrics["scope"].eq(scope),
                    "rmse_delta_candidate_minus_parent_ft",
                ].iloc[0]
            )
            <= 0.0
        )
        for scope in required_scopes
    }
    well_delta = by_well[
        "rmse_delta_candidate_minus_parent_ft"
    ].to_numpy(np.float64)
    by_well_p95 = float(np.quantile(well_delta, 0.95))
    worst_well = float(np.max(well_delta))
    technical = {
        "expected_rows": len(frame)
        == int(get_nested(config, "validation.expected_rows")),
        "expected_wells": frame["well"].nunique()
        == int(get_nested(config, "validation.expected_wells")),
        "expected_folds": sorted(frame["fold"].unique().tolist())
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
        "all_wells_completed_without_fallback": bool(
            audit["status"].eq("ok").all()
        ),
        "finite_prediction_coverage": bool(
            np.isfinite(
                frame[
                    ["parent_prediction", "candidate_prediction"]
                ].to_numpy(np.float64)
            ).all()
        ),
        "posterior_normalization": float(
            audit["maximum_normalization_error"].max()
        )
        <= 1.0e-6,
        "truth_role_fold_reads_before_freeze": (
            ledger.forbidden_reads_before_all_freeze == 0
        ),
        "shard_count": len(summaries) == STAGE1_SHARD_COUNT,
        "candidate_hmm_well_runs": sum(
            int(summary["counts"]["candidate_hmm_well_runs"])
            for summary in summaries
        )
        == int(get_nested(config, "execution.stage1_candidate_hmm_well_runs")),
        "parent_control_hmm_well_runs": sum(
            int(summary["counts"]["parent_control_hmm_well_runs"])
            for summary in summaries
        )
        == 0,
        "execution_zero_counts": all(
            sum(int(summary["counts"][key]) for summary in summaries) == 0
            for key in (
                "lightgbm_configs",
                "trained_ml_folds",
                "boosters",
                "fitted_models",
                "pf_runs",
                "beam_runs",
                "gpu_runs",
            )
        ),
        "runtime_per_shard": all(
            float(summary["runtime"]["elapsed_seconds"])
            <= float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
            for summary in summaries
        ),
        "rss_per_shard": all(
            float(summary["runtime"]["peak_rss_gb"])
            <= float(get_nested(config, "runtime.peak_rss_limit_gb"))
            for summary in summaries
        ),
        "stage0_failure_override_recorded": bool(
            get_nested(
                config,
                "execution.stage1_prerequisite_override_authorized",
            )
        ),
    }
    scientific = {
        "direct_rmse_gain_vs_exp209_ft": float(overall["improvement_ft"]),
        "direct_rmse_gain_vs_exp209_min_ft": float(
            gate_config["direct_rmse_gain_vs_exp209_min_ft"]
        ),
        "positive_fold_count": positive_folds,
        "positive_fold_count_min": int(
            gate_config["positive_fold_count_min"]
        ),
        "ambiguous_rows": int(np.count_nonzero(active)),
        "ambiguous_row_sse_reduction_fraction": active_sse_reduction,
        "ambiguous_row_sse_reduction_min_fraction": float(
            gate_config["ambiguous_row_sse_reduction_min_fraction"]
        ),
        "required_nonworse_scopes": nonworse,
        "by_well_delta_p95_ft": by_well_p95,
        "by_well_delta_p95_max_ft": float(
            gate_config["by_well_delta_p95_max_ft"]
        ),
        "worst_well_regression_ft": worst_well,
        "worst_well_regression_max_ft": float(
            gate_config["worst_well_regression_max_ft"]
        ),
    }
    technical_passed = bool(all(technical.values()))
    scientific_passed = bool(
        scientific["direct_rmse_gain_vs_exp209_ft"]
        >= scientific["direct_rmse_gain_vs_exp209_min_ft"]
        and positive_folds >= scientific["positive_fold_count_min"]
        and math.isfinite(active_sse_reduction)
        and active_sse_reduction
        >= scientific["ambiguous_row_sse_reduction_min_fraction"]
        and all(nonworse.values())
        and by_well_p95 <= scientific["by_well_delta_p95_max_ft"]
        and worst_well <= scientific["worst_well_regression_max_ft"]
    )
    passed = bool(technical_passed and scientific_passed)
    return {
        "passed": passed,
        "technical_passed": technical_passed,
        "scientific_passed": scientific_passed,
        "technical": technical,
        "scientific": scientific,
        "decision": (
            "stage1_full_oof_passed_no_automatic_inference"
            if passed
            else str(gate_config["fail_action"])
        ),
        "stage0_fail_closed_interpretation_retained": True,
    }


def run_stage1_merge(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    os.environ["EXP440_STAGE"] = "stage1_merge"
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha = hashlib.sha256(
        stable_json_bytes(scientific_contract)
    ).hexdigest()
    ledger = LeakageLedger(
        expected_wells=int(get_nested(config, "validation.expected_wells"))
    )
    (
        prediction,
        schedule,
        diagnostic,
        audit,
        manifest,
        summaries,
    ) = merge_stage1_target_free_outputs(config, ledger)
    expected_manifest = assign_stage1_lpt_shards(
        build_stage1_raw_manifest(config, train_data_dir(config))
    )
    manifest_columns = [
        "well",
        "suffix_rows",
        "horizontal_raw_sha256",
        "typewell_raw_sha256",
        "shard_index",
    ]
    if not np.array_equal(
        manifest[manifest_columns].astype(str).to_numpy(),
        expected_manifest[manifest_columns].astype(str).to_numpy(),
    ):
        raise ValueError("exp440 Stage 1 merged manifest changed from LPT input")
    frame, late_attachment = build_stage1_late_readout(
        config,
        prediction,
        schedule,
        manifest,
        ledger,
    )
    metrics, by_well = build_stage1_metrics(config, frame)
    gates = evaluate_stage1_gates(
        config,
        frame,
        metrics,
        by_well,
        audit,
        summaries,
        ledger,
    )
    output = artifacts_dir()
    metric_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage1_scope_metrics.csv",
        metrics,
    )
    by_well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage1_by_well_metrics.csv",
        by_well,
    )
    gate_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage1_gates.json",
        gates,
    )
    contract_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage1_scientific_contract.json",
        scientific_contract,
    )
    overall = metrics.loc[metrics["scope"].eq("overall")].iloc[0]
    status = (
        "stage1_full_oof_passed_no_automatic_inference"
        if gates["passed"]
        else "stage1_full_oof_failed_closed"
    )
    elapsed = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage1_full_oof",
        "status": status,
        "scientific_contract_sha256": scientific_contract_sha,
        "execution_contract": execution_contract,
        "counts": {
            "rows": len(frame),
            "wells": int(frame["well"].nunique()),
            "reporting_folds": int(frame["fold"].nunique()),
            "scientific_variants": 1,
            "candidate_hmm_well_runs": int(audit["well"].nunique()),
            "parent_control_hmm_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_ml_folds": 0,
            "boosters": 0,
            "fitted_models": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
        "cv": {
            "metric": "rmse",
            "candidate_rmse_ft": float(overall["candidate_rmse_ft"]),
            "parent_exp209_rmse_ft": float(overall["parent_rmse_ft"]),
            "improvement_ft": float(overall["improvement_ft"]),
        },
        "gates": gates,
        "late_attachment": late_attachment,
        "target_free_sha": {
            "predictions": logical_frame_sha256(prediction),
            "ambiguity_schedule": logical_frame_sha256(schedule),
            "target_free_diagnostics": logical_frame_sha256(diagnostic),
            "raw_well_identity": expected_manifest.attrs[
                "raw_well_identity_sha256"
            ],
        },
        "runtime": {
            "merge_and_readout_elapsed_seconds": elapsed,
            "merge_peak_rss_gb": peak_rss_gb(),
            "shard_elapsed_seconds": [
                float(summary["runtime"]["elapsed_seconds"])
                for summary in summaries
            ],
            "shard_peak_rss_gb": [
                float(summary["runtime"]["peak_rss_gb"])
                for summary in summaries
            ],
            "versions": runtime_versions(),
        },
        "artifacts": {
            "scope_metrics": metric_artifact,
            "by_well_metrics": by_well_artifact,
            "gates": gate_artifact,
            "scientific_contract": contract_artifact,
        },
        "stage0_fail_closed_interpretation_retained": True,
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage1_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "stage": "stage1_full_oof",
            "metric": "rmse",
            "cv": float(overall["candidate_rmse_ft"]),
            "parent_cv": float(overall["parent_rmse_ft"]),
            "lb": None,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "gates": gates,
        "target_free_sha": summary["target_free_sha"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics_payload)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(
        os.environ.get("EXP440_STAGE")
        or get_nested(config, "execution.selected_stage")
    )
    if stage == "stage0_fixed32":
        return run_stage0(config)
    if stage == "stage1_shard":
        shard_index = int(os.environ["EXP440_SHARD_INDEX"])
        return run_stage1_shard(config, shard_index)
    if stage == "stage1_merge":
        return run_stage1_merge(config)
    if stage == "stage1_full_oof":
        raise RuntimeError(
            "exp440 Stage 1 full OOF must run through four shard wrappers and "
            "the strict merge wrapper"
        )
    raise ValueError(f"unknown exp440 stage: {stage}")


# %% [markdown]
# Importing this notebook remains side-effect free. Direct execution dispatches
# only an explicitly authorized stage. The full OOF requires the four shard
# wrappers plus the strict merge wrapper; inference and submission remain
# fail-closed.

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    EXECUTION_COUNTS = validate_execution_contract(
        CONFIG, require_run_authorization=False
    )
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "event": "exp440_selected_stage_start",
                "experiment": EXPERIMENT_NAME,
                "status": get_nested(CONFIG, "experiment.status"),
                "selected_stage": get_nested(CONFIG, "execution.selected_stage"),
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
    SUMMARY = run_selected_stage(CONFIG)
