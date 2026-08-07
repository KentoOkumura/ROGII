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
# # exp458 acceleration-state exact runtime engine — Stage 0A candidate
#
# This runtime-only experiment keeps the complete exp444 scientific contract.
# It replaces only the log-space execution engine with a float64 scaled
# probability-space factorization, exact-bit delta_MD kernel reuse, and four
# independent well processes.
#
# This source is an implementation candidate only. It does not authorize
# canonical-notebook adoption, a Kaggle package/run, Stage 0B, Stage 1,
# inference, or submission.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable contracts
# 2. Notebook-safe paths, SHA helpers, and leakage guard
# 3. Identity-only fixed4 selection and target-free inputs
# 4. Exact exp209 input preparation
# 5. Frozen acceleration/OU/position kernels and exact-bit cache
# 6. Scaled probability-space factorized forward-backward
# 7. Dense reference and kernel contracts
# 8. Target-free per-well freeze and four-process execution
# 9. Saved-parent parity, repeatability, runtime, and RSS gates
# 10. Guarded Stage 0A orchestration

# %% [markdown]
# ## 1. Imports and immutable contracts

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numba import get_num_threads, njit, set_num_threads

EXPERIMENT_NAME = "exp458_acceleration_state_exact_runtime_engine_audit"
STRUCTURAL_PARENT = "exp444_acceleration_state_exact_hmm"
ROOT_PARENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
EXPECTED_SCIENTIFIC_CONTRACT_SHA256 = (
    "f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972"
)
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

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
        raise ValueError("wrong exp458 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp458 route must remain pf_beam")
    if get_nested(config, "lineage.structural_parent") != STRUCTURAL_PARENT:
        raise ValueError("exp458 structural parent changed")
    if get_nested(config, "lineage.root_parent") != ROOT_PARENT:
        raise ValueError("exp458 root parent changed")
    if not bool(get_nested(config, "design.independent_runtime_hypothesis", False)):
        raise ValueError("exp458 must remain an independent runtime hypothesis")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp458 implementation is not authorized")
    if (
        get_nested(config, "execution.selected_stage") == "implementation_only"
        and bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                True,
            )
        )
    ):
        raise ValueError("canonical notebook adoption remains separately gated")
    for key, label in (
        ("stage0b_run_authorized", "Stage 0B"),
        ("stage1_run_authorized", "Stage 1"),
        ("inference_authorized", "inference"),
        ("submission_authorized", "submission"),
    ):
        if bool(get_nested(config, f"execution.{key}", True)):
            raise ValueError(f"{label} must remain disabled")
    if bool(get_nested(config, "runtime_engine.gpu", True)):
        raise ValueError("exp458 is CPU-only")
    if bool(get_nested(config, "data.exp444_saved_fixed4.regenerate", True)):
        raise ValueError("terminal-closed exp444 baseline must remain load-only")

    expected = {
        "scientific_variants": 1,
        "runtime_engine_candidates": 1,
        "stage0a_repeat_count": 2,
        "stage0a_candidate_hmm_well_runs_per_repeat": 4,
        "stage0a_total_candidate_hmm_well_runs": 8,
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
        raise ValueError(f"exp458 execution contract changed: {observed} != {expected}")

    if require_run_authorization:
        if (
            get_nested(config, "execution.selected_stage")
            != "stage0a_fixed4_runtime_equivalence"
        ):
            raise RuntimeError("exp458 selected_stage must remain Stage 0A")
        if not bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ):
            raise RuntimeError(
                "exp458 Stage 0A requires separate canonical notebook approval"
            )
        if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
            raise RuntimeError("exp458 Stage 0A requires separate package approval")
        if not bool(get_nested(config, "execution.stage0a_run_authorized", False)):
            raise RuntimeError(
                "exp458 implementation approval does not authorize Stage 0A execution"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("exp458 run_hmm remains fail-closed")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError("exp458 prediction creation remains fail-closed")
        if bool(get_nested(config, "execution.create_submission", True)):
            raise ValueError("exp458 Stage 0A must not create a submission")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(get_nested(config, "model.fixed_from_exp441") or {})
    expected_fixed = {
        "position_grid_step_ft": 0.35,
        "rate_grid": "per_well_zero_centered_41_state_parent_grid",
        "n_rates": 41,
        "rate_span": 0.10,
        "rate_span_min": 0.10,
        "rate_transition_family": (
            "exact_ornstein_uhlenbeck_full_support_bin_integral"
        ),
        "sig_r": 0.002,
        "rate_momentum": 0.998,
        "sig_p": 0.02,
        "emission_family": "gaussian_typewell_gr",
        "emission_lambda": 1.0,
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "sigma_clip": [10.0, 60.0],
        "start_sigma_ft": 0.75,
        "initial_rate_sigma": 0.01,
        "band_pad_ft": 100.0,
        "rate_boundary_semantics": "preserve_parent_substochastic_outward_mass",
        "position_boundary_semantics": "preserve_parent_truncation",
        "position_mean_formula": "r_destination*delta_MD-delta_Z",
        "output": "smoothed_posterior_mean_and_std",
    }
    if fixed != expected_fixed:
        raise ValueError(f"exp441 fixed contract changed: {fixed} != {expected_fixed}")

    acceleration = dict(get_nested(config, "model.acceleration_state") or {})
    expected_acceleration = {
        "units": "u_rate_per_md_ft",
        "values": [-0.0005, 0.0, 0.0005],
        "initial_probability": [0.0, 1.0, 0.0],
        "transition": {
            "interior_to_lower": 0.08,
            "interior_stay": 0.84,
            "interior_to_upper": 0.08,
            "boundary_outward_policy": (
                "add_outward_probability_to_boundary_stay"
            ),
            "derivation_note": (
                "nominal_exp209_one_ft_rate_move_probability_per_direction"
            ),
        },
        "destination_rate_mean_formula": (
            "exp(-kappa*delta_MD)*r_source+a_destination*delta_MD"
        ),
        "destination_rate_variance_formula": (
            "sig_r^2*(1-exp(-2*kappa*delta_MD))/(2*kappa)"
        ),
        "position_update_uses": "destination_rate",
    }
    if acceleration != expected_acceleration:
        raise ValueError(
            f"exp458/exp444 acceleration contract changed: "
            f"{acceleration} != {expected_acceleration}"
        )
    variants = list(get_nested(config, "model.active_scientific_variants") or [])
    if variants != ["three_state_persistent_acceleration"]:
        raise ValueError("exp458 must contain the one frozen exp444 candidate")
    contract = {
        "fixed_from_exp441": fixed,
        "acceleration_state": acceleration,
        "active_scientific_variants": variants,
        "forbidden": list(get_nested(config, "model.forbidden") or []),
    }
    observed_sha = hashlib.sha256(stable_json_bytes(contract)).hexdigest()
    configured_sha = str(get_nested(config, "scientific_contract.expected_sha256"))
    if observed_sha != EXPECTED_SCIENTIFIC_CONTRACT_SHA256:
        raise ValueError(
            "embedded exp444 scientific contract SHA changed: "
            f"{observed_sha}"
        )
    if configured_sha != observed_sha:
        raise ValueError(
            f"configured scientific contract SHA changed: {configured_sha}"
        )
    return contract


def runtime_engine_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    engine = get_nested(config, "runtime_engine")
    observed = {
        "candidates": list(engine["candidates"]),
        "dtype": str(engine["dtype"]),
        "forward": list(engine["forward"]),
        "backward": str(engine["backward"]),
        "joint_dense_transition_materialized": bool(
            engine["joint_dense_transition_materialized"]
        ),
        "delta_md_ou_cache": dict(engine["delta_md_ou_cache"]),
        "parallel": dict(engine["parallel"]),
        "accelerator": str(engine["accelerator"]),
        "gpu": bool(engine["gpu"]),
        "use_amp": bool(engine["use_amp"]),
    }
    if observed["candidates"] != ["scaled_probability_factorized_cached_outer4"]:
        raise ValueError("exp458 must contain exactly one frozen runtime engine")
    if observed["dtype"] != "float64":
        raise ValueError("exp458 probability-space state must remain float64")
    expected_forward = [
        "acceleration_3x3_matrix",
        "destination_acceleration_conditioned_rate_41x41_matrix",
        "exact_parent_position_five_offset_stencil",
        "exact_parent_gr_emission",
        "per_row_scale_normalization_and_log_scale_record",
    ]
    if observed["forward"] != expected_forward:
        raise ValueError("exp458 factorized forward operator changed")
    if (
        observed["backward"]
        != "use_saved_forward_scale_with_reverse_factorized_operator"
    ):
        raise ValueError("exp458 reverse factorized operator changed")
    if observed["joint_dense_transition_materialized"]:
        raise ValueError("exp458 must not materialize a dense joint transition")
    cache = observed["delta_md_ou_cache"]
    if cache != {
        "enabled": True,
        "key": "exact_float64_bit_pattern_within_well",
        "rounding_or_quantization": False,
    }:
        raise ValueError("exp458 exact-bit delta_MD cache contract changed")
    parallel = observed["parallel"]
    expected_parallel = {
        "outer_well_workers": 4,
        "process_isolation": True,
        "numba_threads_per_worker": 1,
        "omp_threads_per_worker": 1,
        "mkl_threads_per_worker": 1,
        "openblas_threads_per_worker": 1,
        "shared_reduction_across_wells": False,
        "output_order": "stable_well_id_then_row_idx",
    }
    if parallel != expected_parallel:
        raise ValueError(f"exp458 parallel contract changed: {parallel}")
    if observed["gpu"] or observed["use_amp"] or observed["accelerator"] != "cpu_only":
        raise ValueError("exp458 must remain float64 CPU-only")
    return observed


# %% [markdown]
# ## 2. Notebook-safe paths, SHA helpers, and leakage guard

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
    raise FileNotFoundError("exp458 config.yaml was not found")


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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
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
    return hashlib.sha256(
        normalized.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_deterministic_gzip_csv(
    path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    persisted = frame.copy()
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


@dataclass
class LeakageLedger:
    expected_wells: int = 4
    frozen_wells: set[str] = field(default_factory=set)
    identity_rows_read: int = 0
    target_free_rows_read: int = 0
    forbidden_reads_before_all_freeze: int = 0
    freeze_records: list[dict[str, str]] = field(default_factory=list)

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def record_identity(self, rows: int) -> None:
        self.identity_rows_read += int(rows)

    def record_target_free(self, rows: int) -> None:
        self.target_free_rows_read += int(rows)

    def record_forbidden(self, label: str, rows: int) -> None:
        if not self.all_frozen:
            self.forbidden_reads_before_all_freeze += int(rows)
            raise RuntimeError(f"{label} was read before all target-free freeze")

    def freeze(
        self,
        well: str,
        *,
        joint_transition_sha256: str,
        prediction_sha256: str,
        acceleration_posterior_sha256: str,
        diagnostic_sha256: str,
    ) -> None:
        values = (
            joint_transition_sha256,
            prediction_sha256,
            acceleration_posterior_sha256,
            diagnostic_sha256,
        )
        if not all(values):
            raise ValueError("all target-free SHA values are required")
        self.frozen_wells.add(str(well))
        self.freeze_records.append(
            {
                "well": str(well),
                "joint_transition_sha256": joint_transition_sha256,
                "prediction_sha256": prediction_sha256,
                "acceleration_posterior_sha256": (
                    acceleration_posterior_sha256
                ),
                "diagnostic_sha256": diagnostic_sha256,
            }
        )


# %% [markdown]
# ## 3. Identity-only fixed4 selection and target-free inputs
#
# The fixed32 manifest is opened with `usecols=["well"]` only. The exact exp444
# fixed4 order is SHA256("exp444_runtime_preflight" + well). Role, fold, suffix counts,
# episode labels, truth, and parent error never enter Stage 0A.

# %%
def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    for candidate in (
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "assets" / filename,
        find_project_root() / local_path,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"bootstrap asset not found: {filename}")


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


def stage0a_identity_hash(well: str) -> str:
    return hashlib.sha256(f"exp444_runtime_preflight{well}".encode()).hexdigest()


def select_stage0a_wells(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[list[str], dict[str, Any]]:
    path, observed = fixed32_manifest_path(config)
    frame = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    wells = frame["well"].astype(str).tolist()
    expected_total = int(get_nested(config, "data.fixed32_manifest.total_wells"))
    if len(wells) != expected_total or len(set(wells)) != expected_total:
        raise ValueError("fixed32 identity scope must contain 32 unique wells")
    ledger.record_identity(len(wells))
    ranked = sorted(wells, key=lambda well: (stage0a_identity_hash(well), well))
    count = int(get_nested(config, "data.fixed4_selection.count"))
    selected = ranked[:count]
    if count != ledger.expected_wells or len(selected) != count:
        raise ValueError("Stage 0A must select exactly four wells")
    selection = pd.DataFrame(
        {
            "well": selected,
            "selection_hash": [stage0a_identity_hash(well) for well in selected],
        }
    )
    return selected, {
        "path": str(path),
        "sha256": observed,
        "identity_rows": len(wells),
        "selected_wells": selected,
        "selection_logical_sha256": logical_frame_sha256(selection),
        "columns_read": ["well"],
    }


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


def load_target_free_well(
    well: str,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=lambda column: str(column) != "TVT",
    )
    forbidden = FORBIDDEN_TARGET_FREE_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: target-free input contains {sorted(forbidden)}")
    typewell = (
        pd.read_csv(raw_dir / f"{well}__typewell.csv")
        .sort_values("TVT")
        .reset_index(drop=True)
    )
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
    if not {"MD", "Z", "GR", "TVT_input"}.issubset(horizontal.columns):
        raise ValueError("horizontal input schema changed")
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell input schema changed")
    if "TVT" in horizontal.columns:
        raise ValueError("unknown-suffix TVT reached HMM preparation")

    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(eval_rows) == 0:
        raise ValueError("expected a visible prefix and non-empty suffix")

    initial_rate, rate_rows, valid_steps = robust_initial_rate(known)
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
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[eval_rows.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    span = max(float(fixed["rate_span"]), abs(initial_rate) + 0.04)
    rates = np.linspace(-span, span, int(fixed["n_rates"]), dtype=np.float64)
    return {
        "emission_ll": emission_ll,
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "r0": float(initial_rate),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "last_known_z": float(last["Z"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": initial_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
    }


# %% [markdown]
# ## 5. Acceleration and acceleration-conditioned OU kernels

# %%
def acceleration_transition_matrix(
    acceleration: Mapping[str, Any],
) -> np.ndarray:
    transition = acceleration["transition"]
    lower = float(transition["interior_to_lower"])
    stay = float(transition["interior_stay"])
    upper = float(transition["interior_to_upper"])
    matrix = np.asarray(
        [
            [stay + lower, upper, 0.0],
            [lower, stay, upper],
            [0.0, lower, stay + upper],
        ],
        dtype=np.float64,
    )
    if np.any(matrix < 0.0):
        raise ValueError("acceleration transition contains negative mass")
    return matrix


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


@njit(cache=True, nogil=True)
def precompute_acceleration_ou_probability_kernels(
    delta_md: np.ndarray,
    rates: np.ndarray,
    accelerations: np.ndarray,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    time_count = len(delta_md)
    acceleration_count = len(accelerations)
    rate_count = len(rates)
    edges = finite_voronoi_edges(rates)
    output = np.zeros(
        (time_count, acceleration_count, rate_count, rate_count),
        dtype=np.float64,
    )
    sqrt_two = math.sqrt(2.0)
    for time_index in range(time_count):
        _, decay, variance = ou_conditional_parameters(
            float(delta_md[time_index]),
            sig_r,
            momentum,
        )
        sigma = math.sqrt(variance)
        for acceleration_index in range(acceleration_count):
            acceleration_shift = (
                accelerations[acceleration_index] * delta_md[time_index]
            )
            for source_rate in range(rate_count):
                mean = decay * rates[source_rate] + acceleration_shift
                if sigma <= 0.0:
                    for destination_rate in range(rate_count):
                        if (
                            edges[destination_rate] <= mean
                            and mean < edges[destination_rate + 1]
                        ):
                            output[
                                time_index,
                                acceleration_index,
                                source_rate,
                                destination_rate,
                            ] = 1.0
                    if mean == edges[-1]:
                        output[
                            time_index,
                            acceleration_index,
                            source_rate,
                            rate_count - 1,
                        ] = 1.0
                    continue
                for destination_rate in range(rate_count):
                    lower_z = (
                        (edges[destination_rate] - mean) / (sigma * sqrt_two)
                    )
                    upper_z = (
                        (edges[destination_rate + 1] - mean) / (sigma * sqrt_two)
                    )
                    probability = 0.5 * (
                        math.erf(upper_z) - math.erf(lower_z)
                    )
                    if probability > 0.0:
                        output[
                            time_index,
                            acceleration_index,
                            source_rate,
                            destination_rate,
                        ] = probability
    return output


def exact_float64_bit_pattern(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return contiguous.view(np.uint64)


def build_exact_delta_md_ou_cache(
    delta_md: np.ndarray,
    rates: np.ndarray,
    accelerations: np.ndarray,
    sig_r: float,
    momentum: float,
) -> dict[str, Any]:
    values = np.ascontiguousarray(delta_md, dtype=np.float64)
    bits = exact_float64_bit_pattern(values)
    first_seen: dict[int, int] = {}
    unique_bits: list[int] = []
    row_kernel_index = np.empty(len(bits), dtype=np.int64)
    for row_index, raw_bit in enumerate(bits):
        key = int(raw_bit)
        cache_index = first_seen.get(key)
        if cache_index is None:
            cache_index = len(unique_bits)
            first_seen[key] = cache_index
            unique_bits.append(key)
        row_kernel_index[row_index] = cache_index
    unique_bit_array = np.asarray(unique_bits, dtype=np.uint64)
    unique_values = unique_bit_array.view(np.float64)
    kernels = precompute_acceleration_ou_probability_kernels(
        unique_values,
        np.asarray(rates, dtype=np.float64),
        np.asarray(accelerations, dtype=np.float64),
        float(sig_r),
        float(momentum),
    )
    ledger_sha = array_bundle_sha256(
        row_delta_md_bits=bits,
        row_kernel_index=row_kernel_index,
        unique_delta_md_bits=unique_bit_array,
    )
    return {
        "row_kernel_index": row_kernel_index,
        "unique_bits": unique_bit_array,
        "unique_values": unique_values,
        "kernels": kernels,
        "ledger_sha256": ledger_sha,
        "unique_key_count": int(len(unique_values)),
        "cache_hit_count": int(len(values) - len(unique_values)),
    }


def full_support_ou_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    values = precompute_acceleration_ou_probability_kernels(
        np.asarray([delta_md], dtype=np.float64),
        np.asarray(rates, dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        float(sig_r),
        float(momentum),
    )
    return values[0, 0]


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
        log_values[kernel_index] = -0.5 * (delta / effective_sigma) ** 2
    maximum = np.max(log_values)
    probabilities = np.exp(log_values - maximum)
    probabilities /= np.sum(probabilities)
    return offsets, probabilities


@njit(cache=True, nogil=True)
def precompute_position_kernels(
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    rates: np.ndarray,
    position_step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_count = len(delta_md)
    rate_count = len(rates)
    offsets = np.empty((time_count, rate_count, 5), dtype=np.int64)
    probabilities = np.empty((time_count, rate_count, 5), dtype=np.float64)
    for time_index in range(time_count):
        for destination_rate in range(rate_count):
            mean_shift = (
                rates[destination_rate] * delta_md[time_index]
                - delta_z[time_index]
            )
            local_offsets, local_probabilities = (
                parent_position_kernel_probabilities(
                    mean_shift,
                    position_step,
                    sig_p,
                )
            )
            offsets[time_index, destination_rate] = local_offsets
            probabilities[time_index, destination_rate] = local_probabilities
    return offsets, probabilities


# %% [markdown]
# ## 6. Scaled probability-space factorized forward-backward
#
# The transition is evaluated in the preregistered order:
# acceleration -> rate -> TVT position -> current GR emission. Forward and
# backward passes use the same float64 factorization and saved row scales.
# No state is pruned and no GR evidence is reused.

# %%
@njit(cache=True, nogil=True)
def _hmm3_acceleration_ou_scaled_probability(
    emission,
    position_step,
    rates,
    accelerations,
    acceleration_transition,
    rate_probability_kernels,
    row_kernel_index,
    position_offsets,
    position_probabilities,
    start_position_index,
    start_sigma,
    initial_rate,
    initial_rate_sigma,
    initial_acceleration_probability,
    emission_lambda,
):
    time_count, position_count = emission.shape
    rate_count = len(rates)
    acceleration_count = len(accelerations)
    alpha = np.zeros(
        (time_count, position_count, rate_count, acceleration_count),
        np.float64,
    )
    previous = np.zeros(
        (position_count, rate_count, acceleration_count),
        np.float64,
    )
    initial_total = 0.0
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
            for acceleration_index in range(acceleration_count):
                prior = initial_acceleration_probability[acceleration_index]
                if prior > 0.0:
                    probability = math.exp(
                        position_log
                        - 0.5 * delta_rate * delta_rate
                    ) * prior
                    previous[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = probability
                    initial_total += probability
    if initial_total <= 0.0 or not math.isfinite(initial_total):
        raise ValueError("invalid initial probability mass")
    previous /= initial_total

    acceleration_updated = np.empty(
        (position_count, rate_count, acceleration_count),
        np.float64,
    )
    rate_updated = np.empty_like(acceleration_updated)
    predictive = np.empty_like(acceleration_updated)
    current = np.empty_like(acceleration_updated)
    predictive_rate_mean = np.empty(time_count, np.float64)
    filtered_rate_mean = np.empty(time_count, np.float64)
    predictive_acceleration_mean = np.empty(time_count, np.float64)
    filtered_acceleration_mean = np.empty(time_count, np.float64)
    forward_scale = np.empty(time_count, np.float64)
    emission_offset = np.empty(time_count, np.float64)
    log_likelihood = math.log(initial_total)
    maximum_forward_normalization_error = 0.0

    for time_index in range(time_count):
        for position_index in range(position_count):
            for source_rate in range(rate_count):
                for destination_acceleration in range(acceleration_count):
                    total = 0.0
                    for source_acceleration in range(acceleration_count):
                        total += previous[
                            position_index,
                            source_rate,
                            source_acceleration,
                        ] * acceleration_transition[
                            source_acceleration,
                            destination_acceleration,
                        ]
                    acceleration_updated[
                        position_index,
                        source_rate,
                        destination_acceleration,
                    ] = total

        kernel_index = row_kernel_index[time_index]
        for position_index in range(position_count):
            for destination_rate in range(rate_count):
                for destination_acceleration in range(acceleration_count):
                    total = 0.0
                    for source_rate in range(rate_count):
                        total += acceleration_updated[
                            position_index,
                            source_rate,
                            destination_acceleration,
                        ] * rate_probability_kernels[
                            kernel_index,
                            destination_acceleration,
                            source_rate,
                            destination_rate,
                        ]
                    rate_updated[
                        position_index,
                        destination_rate,
                        destination_acceleration,
                    ] = total

        for destination_rate in range(rate_count):
            for destination_acceleration in range(acceleration_count):
                for destination_position in range(position_count):
                    total = 0.0
                    for kernel_index in range(5):
                        source_position = (
                            destination_position
                            - position_offsets[
                                time_index,
                                destination_rate,
                                kernel_index,
                            ]
                        )
                        if 0 <= source_position < position_count:
                            total += rate_updated[
                                source_position,
                                destination_rate,
                                destination_acceleration,
                            ] * position_probabilities[
                                time_index,
                                destination_rate,
                                kernel_index,
                            ]
                    predictive[
                        destination_position,
                        destination_rate,
                        destination_acceleration,
                    ] = total

        predictive_total = 0.0
        predictive_rate_total = 0.0
        predictive_acceleration_total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    predictive_probability = predictive[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ]
                    predictive_total += predictive_probability
                    predictive_rate_total += (
                        predictive_probability * rates[rate_index]
                    )
                    predictive_acceleration_total += (
                        predictive_probability
                        * accelerations[acceleration_index]
                    )
        if predictive_total <= 0.0 or not math.isfinite(predictive_total):
            raise ValueError("predictive mass is non-positive")
        predictive_rate_mean[time_index] = (
            predictive_rate_total / predictive_total
        )
        predictive_acceleration_mean[time_index] = (
            predictive_acceleration_total / predictive_total
        )

        row_emission_offset = -math.inf
        for position_index in range(position_count):
            row_emission_offset = max(
                row_emission_offset,
                emission_lambda * emission[time_index, position_index],
            )
        emission_offset[time_index] = row_emission_offset
        filtered_total = 0.0
        filtered_rate_total = 0.0
        filtered_acceleration_total = 0.0
        for position_index in range(position_count):
            emission_probability = math.exp(
                emission_lambda * emission[time_index, position_index]
                - row_emission_offset
            )
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    value = predictive[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] * emission_probability
                    current[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = value
                    filtered_total += value
                    filtered_rate_total += value * rates[rate_index]
                    filtered_acceleration_total += (
                        value * accelerations[acceleration_index]
                    )
        if filtered_total <= 0.0 or not math.isfinite(filtered_total):
            raise ValueError("filtered mass is non-positive")
        forward_scale[time_index] = filtered_total
        log_likelihood += math.log(filtered_total) + row_emission_offset
        filtered_check = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    value = (
                        current[
                            position_index,
                            rate_index,
                            acceleration_index,
                        ]
                        / filtered_total
                    )
                    alpha[
                        time_index,
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = value
                    previous[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = value
                    filtered_check += value
        filtered_rate_mean[time_index] = filtered_rate_total / filtered_total
        filtered_acceleration_mean[time_index] = (
            filtered_acceleration_total / filtered_total
        )
        maximum_forward_normalization_error = max(
            maximum_forward_normalization_error,
            abs(filtered_check - 1.0),
        )

    posterior_position = np.zeros(
        (time_count, position_count),
        dtype=np.float64,
    )
    posterior_rate = np.zeros((time_count, rate_count), dtype=np.float64)
    posterior_acceleration = np.zeros(
        (time_count, acceleration_count),
        dtype=np.float64,
    )
    beta_next = np.ones(
        (position_count, rate_count, acceleration_count),
        np.float64,
    )

    for time_index in range(time_count - 1, -1, -1):
        total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    total += alpha[
                        time_index,
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] * beta_next[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ]
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("posterior mass is non-positive")
        maximum_forward_normalization_error = max(
            maximum_forward_normalization_error,
            abs(total - 1.0),
        )
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    probability = (
                        alpha[
                            time_index,
                            position_index,
                            rate_index,
                            acceleration_index,
                        ]
                        * beta_next[
                            position_index,
                            rate_index,
                            acceleration_index,
                        ]
                        / total
                    )
                    posterior_position[time_index, position_index] += probability
                    posterior_rate[time_index, rate_index] += probability
                    posterior_acceleration[
                        time_index,
                        acceleration_index,
                    ] += probability
        if time_index == 0:
            continue

        beta_position = np.empty_like(beta_next)
        for destination_rate in range(rate_count):
            for destination_acceleration in range(acceleration_count):
                for source_position in range(position_count):
                    subtotal = 0.0
                    for kernel_index in range(5):
                        destination_position = (
                            source_position
                            + position_offsets[
                                time_index,
                                destination_rate,
                                kernel_index,
                            ]
                        )
                        if 0 <= destination_position < position_count:
                            subtotal += position_probabilities[
                                time_index,
                                destination_rate,
                                kernel_index,
                            ] * math.exp(
                                emission_lambda
                                * emission[
                                    time_index,
                                    destination_position,
                                ]
                                - emission_offset[time_index]
                            ) * beta_next[
                                destination_position,
                                destination_rate,
                                destination_acceleration,
                            ]
                    beta_position[
                        source_position,
                        destination_rate,
                        destination_acceleration,
                    ] = subtotal

        beta_rate = np.empty_like(beta_next)
        kernel_index = row_kernel_index[time_index]
        for position_index in range(position_count):
            for source_rate in range(rate_count):
                for destination_acceleration in range(acceleration_count):
                    subtotal = 0.0
                    for destination_rate in range(rate_count):
                        subtotal += rate_probability_kernels[
                            kernel_index,
                            destination_acceleration,
                            source_rate,
                            destination_rate,
                        ] * beta_position[
                            position_index,
                            destination_rate,
                            destination_acceleration,
                        ]
                    beta_rate[
                        position_index,
                        source_rate,
                        destination_acceleration,
                    ] = subtotal

        beta_current = np.empty_like(beta_next)
        for position_index in range(position_count):
            for source_rate in range(rate_count):
                for source_acceleration in range(acceleration_count):
                    subtotal = 0.0
                    for destination_acceleration in range(acceleration_count):
                        subtotal += acceleration_transition[
                            source_acceleration,
                            destination_acceleration,
                        ] * beta_rate[
                            position_index,
                            source_rate,
                            destination_acceleration,
                        ]
                    beta_current[
                        position_index,
                        source_rate,
                        source_acceleration,
                    ] = subtotal / forward_scale[time_index]
        beta_next = beta_current

    maximum_posterior_normalization_error = 0.0
    for time_index in range(time_count):
        maximum_posterior_normalization_error = max(
            maximum_posterior_normalization_error,
            abs(np.sum(posterior_position[time_index]) - 1.0),
            abs(np.sum(posterior_rate[time_index]) - 1.0),
            abs(np.sum(posterior_acceleration[time_index]) - 1.0),
        )
    return (
        posterior_position,
        posterior_rate,
        posterior_acceleration,
        predictive_rate_mean,
        filtered_rate_mean,
        predictive_acceleration_mean,
        filtered_acceleration_mean,
        float(log_likelihood),
        max(
            maximum_forward_normalization_error,
            maximum_posterior_normalization_error,
        ),
        forward_scale,
        emission_offset,
    )


def run_acceleration_state_hmm(
    prepared: Mapping[str, Any],
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    delta_md = np.asarray(prepared["dm"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    accelerations = np.asarray(acceleration["values"], dtype=np.float64)
    acceleration_transition = acceleration_transition_matrix(acceleration)
    rate_cache = build_exact_delta_md_ou_cache(
        delta_md,
        rates,
        accelerations,
        float(fixed["sig_r"]),
        float(fixed["rate_momentum"]),
    )
    position_offsets, position_probabilities = precompute_position_kernels(
        delta_md,
        np.asarray(prepared["dz"], dtype=np.float64),
        rates,
        float(fixed["position_grid_step_ft"]),
        float(fixed["sig_p"]),
    )
    (
        posterior_position,
        posterior_rate,
        posterior_acceleration,
        predictive_rate_mean,
        filtered_rate_mean,
        predictive_acceleration_mean,
        filtered_acceleration_mean,
        log_likelihood,
        maximum_normalization_error,
        forward_scale,
        emission_offset,
    ) = _hmm3_acceleration_ou_scaled_probability(
        np.asarray(prepared["emission_ll"], dtype=np.float64),
        float(fixed["position_grid_step_ft"]),
        rates,
        accelerations,
        acceleration_transition,
        np.asarray(rate_cache["kernels"], dtype=np.float64),
        np.asarray(rate_cache["row_kernel_index"], dtype=np.int64),
        position_offsets,
        position_probabilities,
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        np.asarray(acceleration["initial_probability"], dtype=np.float64),
        float(fixed["emission_lambda"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = np.sum(posterior_position * grid[None, :], axis=1)
    posterior_variance = (
        np.sum(posterior_position * (grid[None, :] ** 2), axis=1)
        - posterior_mean**2
    )
    posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0))
    posterior_rate_mean = np.sum(posterior_rate * rates[None, :], axis=1)
    posterior_rate_variance = (
        np.sum(posterior_rate * (rates[None, :] ** 2), axis=1)
        - posterior_rate_mean**2
    )
    posterior_rate_std = np.sqrt(
        np.maximum(posterior_rate_variance, 0.0)
    )
    posterior_acceleration_mean = np.sum(
        posterior_acceleration * accelerations[None, :],
        axis=1,
    )
    posterior_acceleration_nonzero_mass = (
        posterior_acceleration[:, 0] + posterior_acceleration[:, 2]
    )
    joint_transition_sha256 = array_bundle_sha256(
        delta_md=delta_md,
        rates=rates,
        accelerations=accelerations,
        acceleration_transition=acceleration_transition,
        rate_probability_kernels=np.asarray(rate_cache["kernels"]),
        row_kernel_index=np.asarray(rate_cache["row_kernel_index"]),
        position_offsets=position_offsets,
        position_probabilities=position_probabilities,
    )
    prediction_sha256 = array_bundle_sha256(
        posterior_mean=posterior_mean.astype(np.float32),
        posterior_std=posterior_std.astype(np.float32),
    )
    acceleration_posterior_sha256 = array_bundle_sha256(
        posterior_acceleration=posterior_acceleration.astype(np.float32),
        posterior_acceleration_mean=(
            posterior_acceleration_mean.astype(np.float32)
        ),
    )
    diagnostic_sha256 = array_bundle_sha256(
        predictive_rate_mean=predictive_rate_mean.astype(np.float32),
        filtered_rate_mean=filtered_rate_mean.astype(np.float32),
        posterior_rate_mean=posterior_rate_mean.astype(np.float32),
        posterior_rate_std=posterior_rate_std.astype(np.float32),
        predictive_acceleration_mean=(
            predictive_acceleration_mean.astype(np.float32)
        ),
        filtered_acceleration_mean=(
            filtered_acceleration_mean.astype(np.float32)
        ),
        posterior_acceleration_nonzero_mass=(
            posterior_acceleration_nonzero_mass.astype(np.float32)
        ),
    )
    return {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "posterior_rate_mean": posterior_rate_mean,
        "posterior_rate_std": posterior_rate_std,
        "posterior_acceleration": posterior_acceleration,
        "posterior_acceleration_mean": posterior_acceleration_mean,
        "posterior_acceleration_nonzero_mass": (
            posterior_acceleration_nonzero_mass
        ),
        "predictive_rate_mean": predictive_rate_mean,
        "filtered_rate_mean": filtered_rate_mean,
        "predictive_acceleration_mean": predictive_acceleration_mean,
        "filtered_acceleration_mean": filtered_acceleration_mean,
        "log_likelihood": float(log_likelihood),
        "maximum_normalization_error": float(maximum_normalization_error),
        "joint_transition_sha256": joint_transition_sha256,
        "delta_md_key_ledger_sha256": str(rate_cache["ledger_sha256"]),
        "ou_cache_unique_keys": int(rate_cache["unique_key_count"]),
        "ou_cache_hits": int(rate_cache["cache_hit_count"]),
        "position_kernel_sha256": array_bundle_sha256(
            offsets=position_offsets,
            probabilities=position_probabilities,
        ),
        "forward_scale_sha256": array_bundle_sha256(
            scale=forward_scale,
            emission_offset=emission_offset,
        ),
        "prediction_sha256": prediction_sha256,
        "acceleration_posterior_sha256": acceleration_posterior_sha256,
        "diagnostic_sha256": diagnostic_sha256,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 7. Dense brute-force and zero-acceleration contracts

# %%
def zero_acceleration_kernel_parity_contract(
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    rates = np.linspace(-0.10, 0.10, 9, dtype=np.float64)
    delta_md = np.asarray([1.0, 7.5, 25.0], dtype=np.float64)
    observed = precompute_acceleration_ou_probability_kernels(
        delta_md,
        rates,
        np.asarray([0.0], dtype=np.float64),
        float(fixed["sig_r"]),
        float(fixed["rate_momentum"]),
    )[:, 0]
    maximum_error = 0.0
    for row, step_md in enumerate(delta_md):
        expected = full_support_ou_rate_kernel(
            rates,
            float(step_md),
            float(fixed["sig_r"]),
            float(fixed["rate_momentum"]),
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(observed[row] - expected))),
        )
    return {
        "zero_acceleration_rate_kernel_parity_vs_exp441_max_abs_error": (
            maximum_error
        ),
        "pass": bool(maximum_error <= 1.0e-12),
    }


def acceleration_transition_contract(
    acceleration: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = acceleration_transition_matrix(acceleration)
    initial = np.asarray(
        acceleration["initial_probability"],
        dtype=np.float64,
    )
    row_sum_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    expected = np.asarray(
        [
            [0.92, 0.08, 0.0],
            [0.08, 0.84, 0.08],
            [0.0, 0.08, 0.92],
        ],
        dtype=np.float64,
    )
    matrix_error = float(np.max(np.abs(matrix - expected)))
    initial_error = float(np.max(np.abs(initial - np.asarray([0.0, 1.0, 0.0]))))
    maximum_error = max(row_sum_error, matrix_error, initial_error)
    return {
        "acceleration_row_sum_max_error": row_sum_error,
        "acceleration_matrix_max_abs_error": matrix_error,
        "initial_acceleration_prior_max_abs_error": initial_error,
        "pass": bool(maximum_error <= 1.0e-12),
    }


def dense_acceleration_reference(
    prepared: Mapping[str, Any],
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    emission = np.asarray(prepared["emission_ll"], dtype=np.float64)
    delta_md = np.asarray(prepared["dm"], dtype=np.float64)
    delta_z = np.asarray(prepared["dz"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    accelerations = np.asarray(acceleration["values"], dtype=np.float64)
    acceleration_transition = acceleration_transition_matrix(acceleration)
    rate_kernels = precompute_acceleration_ou_probability_kernels(
        delta_md,
        rates,
        accelerations,
        float(fixed["sig_r"]),
        float(fixed["rate_momentum"]),
    )
    time_count, position_count = emission.shape
    rate_count = len(rates)
    acceleration_count = len(accelerations)
    state_count = position_count * rate_count * acceleration_count
    transitions: list[np.ndarray] = []
    for time_index in range(time_count):
        matrix = np.zeros((state_count, state_count), dtype=np.float64)
        for source_position in range(position_count):
            for source_rate in range(rate_count):
                for source_acceleration in range(acceleration_count):
                    source_state = (
                        (
                            source_position * rate_count
                            + source_rate
                        )
                        * acceleration_count
                        + source_acceleration
                    )
                    for destination_acceleration in range(acceleration_count):
                        acceleration_probability = acceleration_transition[
                            source_acceleration,
                            destination_acceleration,
                        ]
                        for destination_rate in range(rate_count):
                            rate_probability = rate_kernels[
                                time_index,
                                destination_acceleration,
                                source_rate,
                                destination_rate,
                            ]
                            if acceleration_probability * rate_probability <= 0.0:
                                continue
                            shift = (
                                rates[destination_rate] * delta_md[time_index]
                                - delta_z[time_index]
                            )
                            offsets, position_probabilities = (
                                parent_position_kernel_probabilities(
                                    shift,
                                    float(fixed["position_grid_step_ft"]),
                                    float(fixed["sig_p"]),
                                )
                            )
                            for kernel_index in range(5):
                                destination_position = (
                                    source_position + offsets[kernel_index]
                                )
                                if 0 <= destination_position < position_count:
                                    destination_state = (
                                        (
                                            destination_position * rate_count
                                            + destination_rate
                                        )
                                        * acceleration_count
                                        + destination_acceleration
                                    )
                                    matrix[
                                        source_state,
                                        destination_state,
                                    ] += (
                                        acceleration_probability
                                        * rate_probability
                                        * position_probabilities[kernel_index]
                                    )
        transitions.append(matrix)

    initial = np.zeros(
        (position_count, rate_count, acceleration_count),
        dtype=np.float64,
    )
    initial_acceleration = np.asarray(
        acceleration["initial_probability"],
        dtype=np.float64,
    )
    for position_index in range(position_count):
        position_log = -0.5 * (
            (
                (
                    position_index - float(prepared["start_p"])
                )
                * float(fixed["position_grid_step_ft"])
            )
            / float(fixed["start_sigma_ft"])
        ) ** 2
        for rate_index, rate in enumerate(rates):
            rate_log = -0.5 * (
                (
                    rate - float(prepared["r0"])
                )
                / float(fixed["initial_rate_sigma"])
            ) ** 2
            initial[position_index, rate_index] = (
                math.exp(position_log + rate_log) * initial_acceleration
            )
    previous = initial.reshape(-1)
    forward = np.empty((time_count, state_count), dtype=np.float64)
    emission_probability = np.exp(emission)
    for time_index in range(time_count):
        current = previous @ transitions[time_index]
        current *= np.repeat(
            emission_probability[time_index],
            rate_count * acceleration_count,
        )
        current /= current.sum()
        forward[time_index] = current
        previous = current

    backward = np.ones((time_count, state_count), dtype=np.float64)
    for time_index in range(time_count - 1, 0, -1):
        weighted_next = backward[time_index] * np.repeat(
            emission_probability[time_index],
            rate_count * acceleration_count,
        )
        values = transitions[time_index] @ weighted_next
        values /= values.sum()
        backward[time_index - 1] = values

    posterior_position = np.empty(
        (time_count, position_count),
        dtype=np.float64,
    )
    posterior_acceleration = np.empty(
        (time_count, acceleration_count),
        dtype=np.float64,
    )
    for time_index in range(time_count):
        posterior = forward[time_index] * backward[time_index]
        posterior /= posterior.sum()
        cube = posterior.reshape(
            position_count,
            rate_count,
            acceleration_count,
        )
        posterior_position[time_index] = cube.sum(axis=(1, 2))
        posterior_acceleration[time_index] = cube.sum(axis=(0, 1))
    return posterior_position, posterior_acceleration


def brute_force_posterior_contract(
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> dict[str, Any]:
    rows = 3
    positions = 5
    rates = np.linspace(-0.03, 0.03, 3, dtype=np.float64)
    grid = (
        12_000.0
        + np.arange(positions, dtype=np.float64)
        * float(fixed["position_grid_step_ft"])
    )
    x = np.linspace(-1.0, 1.0, positions)
    prepared = {
        "emission_ll": np.vstack(
            [
                -0.5 * ((x - 0.2 * math.sin(index)) / 0.45) ** 2
                for index in range(rows)
            ]
        ).astype(np.float32),
        "dm": np.asarray([1.0, 5.0, 11.0], dtype=np.float64),
        "dz": np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        "grid": grid,
        "rates": rates,
        "start_p": 2.0,
        "r0": 0.0,
    }
    observed = run_acceleration_state_hmm(prepared, fixed, acceleration)
    reference_position, reference_acceleration = dense_acceleration_reference(
        prepared,
        fixed,
        acceleration,
    )
    observed_mean = np.asarray(observed["posterior_mean"], dtype=np.float64)
    reference_mean = reference_position @ grid
    position_error = float(np.max(np.abs(observed_mean - reference_mean)))
    acceleration_error = float(
        np.max(
            np.abs(
                np.asarray(
                    observed["posterior_acceleration"],
                    dtype=np.float64,
                )
                - reference_acceleration
            )
        )
    )
    maximum_error = max(position_error, acceleration_error)
    return {
        "posterior_prediction_max_abs_error": position_error,
        "posterior_acceleration_max_abs_error": acceleration_error,
        "maximum_abs_error": maximum_error,
        "pass": bool(maximum_error <= 1.0e-6),
    }


# %% [markdown]
# ## 8. Target-free prediction/posterior freeze

# %%
@dataclass
class FrozenWell:
    well: str
    eval_id: np.ndarray
    row_idx: np.ndarray
    raw_gr_missing: np.ndarray
    candidate_prediction: np.ndarray
    candidate_posterior_std: np.ndarray
    posterior_rate_mean: np.ndarray
    posterior_rate_std: np.ndarray
    posterior_acceleration: np.ndarray
    posterior_acceleration_mean: np.ndarray
    posterior_acceleration_nonzero_mass: np.ndarray
    predictive_rate_mean: np.ndarray
    filtered_rate_mean: np.ndarray
    joint_transition_sha256: str
    delta_md_key_ledger_sha256: str
    position_kernel_sha256: str
    forward_scale_sha256: str
    input_sha256: str
    prepared_input_sha256: str
    prediction_sha256: str
    acceleration_posterior_sha256: str
    diagnostic_sha256: str
    maximum_normalization_error: float
    log_likelihood: float
    elapsed_seconds: float
    peak_rss_gb: float
    worker_pid: int
    numba_threads: int
    thread_environment: dict[str, str]
    ou_cache_unique_keys: int
    ou_cache_hits: int


THREAD_ENVIRONMENT_KEYS = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_single_thread_worker_guard() -> dict[str, str]:
    for key in THREAD_ENVIRONMENT_KEYS:
        os.environ[key] = "1"
    set_num_threads(1)
    return {key: os.environ[key] for key in THREAD_ENVIRONMENT_KEYS}


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    thread_environment = apply_single_thread_worker_guard()
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    input_sha256 = array_bundle_sha256(
        horizontal_file_sha256=np.frombuffer(
            bytes.fromhex(sha256_file(horizontal_path)),
            dtype=np.uint8,
        ),
        typewell_file_sha256=np.frombuffer(
            bytes.fromhex(sha256_file(typewell_path)),
            dtype=np.uint8,
        ),
    )
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, fixed)
    prepared_input_sha256 = array_bundle_sha256(
        emission_ll=np.asarray(prepared["emission_ll"]),
        delta_md=np.asarray(prepared["dm"]),
        delta_z=np.asarray(prepared["dz"]),
        grid=np.asarray(prepared["grid"]),
        rates=np.asarray(prepared["rates"]),
        eval_index=np.asarray(prepared["eval_index"]),
    )
    decoded = run_acceleration_state_hmm(prepared, fixed, acceleration)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    frozen = FrozenWell(
        well=str(well),
        eval_id=np.asarray([f"{well}_{int(row)}" for row in row_idx]),
        row_idx=row_idx,
        raw_gr_missing=np.asarray(
            prepared["raw_gr_missing"],
            dtype=bool,
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
        posterior_rate_std=np.asarray(
            decoded["posterior_rate_std"],
            dtype=np.float64,
        ),
        posterior_acceleration=np.asarray(
            decoded["posterior_acceleration"],
            dtype=np.float64,
        ),
        posterior_acceleration_mean=np.asarray(
            decoded["posterior_acceleration_mean"],
            dtype=np.float64,
        ),
        posterior_acceleration_nonzero_mass=np.asarray(
            decoded["posterior_acceleration_nonzero_mass"],
            dtype=np.float64,
        ),
        predictive_rate_mean=np.asarray(
            decoded["predictive_rate_mean"],
            dtype=np.float64,
        ),
        filtered_rate_mean=np.asarray(
            decoded["filtered_rate_mean"],
            dtype=np.float64,
        ),
        joint_transition_sha256=str(decoded["joint_transition_sha256"]),
        delta_md_key_ledger_sha256=str(
            decoded["delta_md_key_ledger_sha256"]
        ),
        position_kernel_sha256=str(decoded["position_kernel_sha256"]),
        forward_scale_sha256=str(decoded["forward_scale_sha256"]),
        input_sha256=input_sha256,
        prepared_input_sha256=prepared_input_sha256,
        prediction_sha256=str(decoded["prediction_sha256"]),
        acceleration_posterior_sha256=str(
            decoded["acceleration_posterior_sha256"]
        ),
        diagnostic_sha256=str(decoded["diagnostic_sha256"]),
        maximum_normalization_error=float(
            decoded["maximum_normalization_error"]
        ),
        log_likelihood=float(decoded["log_likelihood"]),
        elapsed_seconds=float(decoded["elapsed_seconds"]),
        peak_rss_gb=peak_rss_gb(),
        worker_pid=os.getpid(),
        numba_threads=int(get_num_threads()),
        thread_environment=thread_environment,
        ou_cache_unique_keys=int(decoded["ou_cache_unique_keys"]),
        ou_cache_hits=int(decoded["ou_cache_hits"]),
    )
    arrays = (
        frozen.candidate_prediction,
        frozen.candidate_posterior_std,
        frozen.posterior_rate_mean,
        frozen.posterior_rate_std,
        frozen.posterior_acceleration,
        frozen.posterior_acceleration_mean,
        frozen.posterior_acceleration_nonzero_mass,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{well}: non-finite target-free decoder output")
    ledger.freeze(
        well,
        joint_transition_sha256=frozen.joint_transition_sha256,
        prediction_sha256=frozen.prediction_sha256,
        acceleration_posterior_sha256=(
            frozen.acceleration_posterior_sha256
        ),
        diagnostic_sha256=frozen.diagnostic_sha256,
    )
    return frozen


def prediction_frame(frozen_wells: list[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "id": item.eval_id,
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "hmm_mean_tvt": item.candidate_prediction,
                    "hmm_std_tvt": item.candidate_posterior_std,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"],
        kind="mergesort",
    )


def acceleration_frame(frozen_wells: list[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "acceleration_negative_mass": (
                        item.posterior_acceleration[:, 0]
                    ),
                    "acceleration_zero_mass": (
                        item.posterior_acceleration[:, 1]
                    ),
                    "acceleration_positive_mass": (
                        item.posterior_acceleration[:, 2]
                    ),
                    "posterior_acceleration_mean": (
                        item.posterior_acceleration_mean
                    ),
                    "posterior_acceleration_nonzero_mass": (
                        item.posterior_acceleration_nonzero_mass
                    ),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"],
        kind="mergesort",
    )


def diagnostic_frame(frozen_wells: list[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "raw_gr_missing": item.raw_gr_missing,
                    "predictive_rate_mean": item.predictive_rate_mean,
                    "filtered_rate_mean": item.filtered_rate_mean,
                    "posterior_rate_mean": item.posterior_rate_mean,
                    "posterior_rate_std": item.posterior_rate_std,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"],
        kind="mergesort",
    )


def combined_well_sha(
    frozen_wells: list[FrozenWell],
    attribute: str,
) -> str:
    payload = [
        {"well": item.well, attribute: str(getattr(item, attribute))}
        for item in sorted(frozen_wells, key=lambda value: value.well)
    ]
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


# %% [markdown]
# ## 9. Saved-parent parity, repeatability, runtime, and RSS gates
#
# Saved exp444 output is read only after both candidate repeats are frozen.
# Runtime uses four isolated well processes; completion order is discarded.

# %%
def frame_decompressed_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def repeat_output_sha256(
    frozen_wells: list[FrozenWell],
) -> dict[str, str]:
    return {
        "prediction_decompressed_sha256": frame_decompressed_sha256(
            prediction_frame(frozen_wells)
        ),
        "posterior_bundle_sha256": frame_decompressed_sha256(
            acceleration_frame(frozen_wells)
        ),
        "diagnostic_sha256": frame_decompressed_sha256(
            diagnostic_frame(frozen_wells)
        ),
    }


def _read_linux_rss_gb(pid: int) -> float:
    status = Path(f"/proc/{int(pid)}/status")
    if not status.is_file():
        return 0.0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / (1024.0**2)
    return 0.0


def process_tree_rss_gb(executor: ProcessPoolExecutor) -> float:
    pids = [os.getpid()]
    processes = getattr(executor, "_processes", {})
    pids.extend(
        int(process.pid)
        for process in processes.values()
        if process.pid is not None
    )
    return float(sum(_read_linux_rss_gb(pid) for pid in sorted(set(pids))))


def _decode_worker(
    well: str,
    raw_dir: str,
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> tuple[FrozenWell, dict[str, Any]]:
    ledger = LeakageLedger(expected_wells=1)
    frozen = freeze_target_free_well(
        well=str(well),
        raw_dir=Path(raw_dir),
        fixed=fixed,
        acceleration=acceleration,
        ledger=ledger,
    )
    return frozen, {
        "target_free_rows_read": int(ledger.target_free_rows_read),
        "forbidden_reads_before_all_freeze": int(
            ledger.forbidden_reads_before_all_freeze
        ),
        "all_frozen": bool(ledger.all_frozen),
    }


def create_outer_executor(worker_count: int) -> ProcessPoolExecutor:
    if platform.system() != "Linux":
        raise RuntimeError("exp458 four-process engine requires Kaggle Linux")
    if int(worker_count) != 4:
        raise ValueError("exp458 outer worker count must remain four")
    apply_single_thread_worker_guard()
    return ProcessPoolExecutor(
        max_workers=int(worker_count),
        mp_context=mp.get_context("fork"),
    )


def run_parallel_repeat(
    *,
    repeat_index: int,
    executor: ProcessPoolExecutor,
    wells: list[str],
    raw_dir: Path,
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> tuple[list[FrozenWell], dict[str, Any], LeakageLedger]:
    if len(wells) != 4 or len(set(wells)) != 4:
        raise ValueError("exp458 Stage 0A repeat requires four unique wells")
    started = time.perf_counter()
    futures = {
        executor.submit(
            _decode_worker,
            str(well),
            str(raw_dir),
            dict(fixed),
            dict(acceleration),
        ): str(well)
        for well in wells
    }
    pending = set(futures)
    peak_tree_rss = process_tree_rss_gb(executor)
    completed: dict[str, tuple[FrozenWell, dict[str, Any]]] = {}
    while pending:
        done, pending = wait(
            pending,
            timeout=0.10,
            return_when=FIRST_COMPLETED,
        )
        peak_tree_rss = max(
            peak_tree_rss,
            process_tree_rss_gb(executor),
        )
        for future in done:
            well = futures[future]
            completed[well] = future.result()
    decode_wall_seconds = float(time.perf_counter() - started)
    peak_tree_rss = max(peak_tree_rss, process_tree_rss_gb(executor))

    ledger = LeakageLedger(expected_wells=4)
    frozen_wells: list[FrozenWell] = []
    worker_audits = []
    for well in sorted(wells):
        frozen, worker_audit = completed[well]
        if not bool(worker_audit["all_frozen"]):
            raise RuntimeError(f"{well}: worker did not freeze its output")
        if int(worker_audit["forbidden_reads_before_all_freeze"]) != 0:
            raise RuntimeError(f"{well}: worker leakage guard failed")
        ledger.record_target_free(int(worker_audit["target_free_rows_read"]))
        ledger.freeze(
            well,
            joint_transition_sha256=frozen.joint_transition_sha256,
            prediction_sha256=frozen.prediction_sha256,
            acceleration_posterior_sha256=(
                frozen.acceleration_posterior_sha256
            ),
            diagnostic_sha256=frozen.diagnostic_sha256,
        )
        frozen_wells.append(frozen)
        worker_audits.append(
            {
                "well": well,
                "pid": int(frozen.worker_pid),
                "numba_threads": int(frozen.numba_threads),
                "thread_environment": dict(frozen.thread_environment),
                "peak_rss_gb": float(frozen.peak_rss_gb),
                "hmm_seconds": float(frozen.elapsed_seconds),
                "input_sha256": frozen.input_sha256,
                "prepared_input_sha256": frozen.prepared_input_sha256,
                "delta_md_key_ledger_sha256": (
                    frozen.delta_md_key_ledger_sha256
                ),
                "ou_cache_unique_keys": int(frozen.ou_cache_unique_keys),
                "ou_cache_hits": int(frozen.ou_cache_hits),
                "forbidden_reads_before_all_freeze": int(
                    worker_audit["forbidden_reads_before_all_freeze"]
                ),
            }
        )
    if not ledger.all_frozen:
        raise RuntimeError("exp458 repeat did not freeze all four wells")
    runtime = {
        "repeat_index": int(repeat_index),
        "decode_wall_seconds": decode_wall_seconds,
        "sum_well_hmm_seconds": float(
            sum(item.elapsed_seconds for item in frozen_wells)
        ),
        "process_tree_peak_rss_gb": peak_tree_rss,
        "effective_outer_worker_pids": sorted(
            {int(item.worker_pid) for item in frozen_wells}
        ),
        "effective_outer_workers": int(
            len({int(item.worker_pid) for item in frozen_wells})
        ),
        "worker_audits": worker_audits,
        "output_sha256": repeat_output_sha256(frozen_wells),
    }
    return frozen_wells, runtime, ledger


def _resolve_parent_artifact(
    config: Mapping[str, Any],
    filename_key: str,
) -> Path:
    spec = get_nested(config, "data.exp444_saved_fixed4")
    filename = str(spec[filename_key])
    roots: list[Path] = []
    for raw in spec["candidates"]:
        candidate = Path(str(raw))
        roots.append(candidate)
        if not candidate.is_absolute():
            roots.append(find_project_root() / candidate)
    for root in roots:
        direct = root / filename
        if direct.is_file():
            return direct
        if root.is_dir():
            matches = sorted(root.glob(f"**/{filename}"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"saved exp444 artifact not found: {filename}")


def resolve_parent_fixed4_artifacts(
    config: Mapping[str, Any],
) -> dict[str, Path]:
    return {
        "prediction": _resolve_parent_artifact(
            config,
            "prediction_filename",
        ),
        "acceleration": _resolve_parent_artifact(
            config,
            "acceleration_posterior_filename",
        ),
        "diagnostic": _resolve_parent_artifact(
            config,
            "diagnostic_filename",
        ),
    }


def load_parent_fixed4_after_candidate_freeze(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    all_candidate_wells_frozen: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not all_candidate_wells_frozen:
        raise RuntimeError("saved exp444 output cannot be read before candidate freeze")
    spec = get_nested(config, "data.exp444_saved_fixed4")
    expected_sha = {
        "prediction": str(spec["expected_prediction_decompressed_sha256"]),
        "acceleration": str(
            spec["expected_acceleration_posterior_decompressed_sha256"]
        ),
        "diagnostic": str(spec["expected_diagnostic_decompressed_sha256"]),
    }
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for label in ("prediction", "acceleration", "diagnostic"):
        path = Path(paths[label])
        observed = sha256_decompressed_csv(path)
        if observed != expected_sha[label]:
            raise ValueError(
                f"saved exp444 {label} SHA changed: "
                f"expected={expected_sha[label]}, observed={observed}"
            )
        frame = pd.read_csv(path, float_precision="round_trip")
        frames[label] = frame.sort_values(
            ["well", "row_idx"],
            kind="mergesort",
        ).reset_index(drop=True)
        evidence[label] = {
            "path": str(path),
            "decompressed_sha256": observed,
            "rows": int(len(frame)),
        }
    return frames, evidence


def _aligned_max_abs_error(
    candidate: pd.DataFrame,
    parent: pd.DataFrame,
    columns: list[str],
) -> float:
    keys = ["well", "row_idx"]
    left = candidate[keys + columns].copy()
    right = parent[keys + columns].copy()
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("parity frame contains duplicate well/row keys")
    merged = left.merge(
        right,
        on=keys,
        how="outer",
        suffixes=("_candidate", "_parent"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("candidate and saved exp444 keys differ")
    maximum = 0.0
    for column in columns:
        delta = np.abs(
            merged[f"{column}_candidate"].to_numpy(np.float64)
            - merged[f"{column}_parent"].to_numpy(np.float64)
        )
        if not np.isfinite(delta).all():
            raise ValueError(f"non-finite parity delta for {column}")
        maximum = max(maximum, float(delta.max(initial=0.0)))
    return maximum


def evaluate_saved_parent_parity(
    frozen_wells: list[FrozenWell],
    parent_frames: Mapping[str, pd.DataFrame],
) -> dict[str, float]:
    candidate_prediction = prediction_frame(frozen_wells)
    candidate_acceleration = acceleration_frame(frozen_wells)
    candidate_diagnostic = diagnostic_frame(frozen_wells)
    return {
        "parent_prediction_mean_max_abs_error_ft": _aligned_max_abs_error(
            candidate_prediction,
            parent_frames["prediction"],
            ["hmm_mean_tvt"],
        ),
        "parent_prediction_std_max_abs_error_ft": _aligned_max_abs_error(
            candidate_prediction,
            parent_frames["prediction"],
            ["hmm_std_tvt"],
        ),
        "parent_acceleration_posterior_max_abs_error": _aligned_max_abs_error(
            candidate_acceleration,
            parent_frames["acceleration"],
            [
                "acceleration_negative_mass",
                "acceleration_zero_mass",
                "acceleration_positive_mass",
                "posterior_acceleration_mean",
                "posterior_acceleration_nonzero_mass",
            ],
        ),
        "parent_rate_diagnostic_max_abs_error": _aligned_max_abs_error(
            candidate_diagnostic,
            parent_frames["diagnostic"],
            [
                "predictive_rate_mean",
                "filtered_rate_mean",
                "posterior_rate_mean",
                "posterior_rate_std",
            ],
        ),
    }


def exact_cache_kernel_contract(
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> dict[str, Any]:
    left = np.float64(10.0)
    right = np.nextafter(left, np.float64(np.inf))
    delta_md = np.asarray([left, right, left, 1.0], dtype=np.float64)
    rates = np.linspace(-0.10, 0.10, 9, dtype=np.float64)
    accelerations = np.asarray(acceleration["values"], dtype=np.float64)
    cache = build_exact_delta_md_ou_cache(
        delta_md,
        rates,
        accelerations,
        float(fixed["sig_r"]),
        float(fixed["rate_momentum"]),
    )
    expanded = np.asarray(cache["kernels"])[
        np.asarray(cache["row_kernel_index"])
    ]
    direct = precompute_acceleration_ou_probability_kernels(
        delta_md,
        rates,
        accelerations,
        float(fixed["sig_r"]),
        float(fixed["rate_momentum"]),
    )
    maximum_error = float(np.max(np.abs(expanded - direct)))
    distinct_adjacent = bool(
        cache["row_kernel_index"][0] != cache["row_kernel_index"][1]
    )
    duplicate_reused = bool(
        cache["row_kernel_index"][0] == cache["row_kernel_index"][2]
    )
    return {
        "maximum_abs_error": maximum_error,
        "unique_key_count": int(cache["unique_key_count"]),
        "cache_hit_count": int(cache["cache_hit_count"]),
        "nextafter_bit_pattern_is_distinct": distinct_adjacent,
        "duplicate_bit_pattern_is_reused": duplicate_reused,
        "pass": bool(
            maximum_error <= 1.0e-12
            and distinct_adjacent
            and duplicate_reused
            and int(cache["unique_key_count"]) == 3
        ),
    }


def position_kernel_contract(fixed: Mapping[str, Any]) -> dict[str, Any]:
    delta_md = np.asarray([1.0, 10.0], dtype=np.float64)
    delta_z = np.asarray([0.2, -0.3], dtype=np.float64)
    rates = np.linspace(-0.05, 0.05, 7, dtype=np.float64)
    offsets, probabilities = precompute_position_kernels(
        delta_md,
        delta_z,
        rates,
        float(fixed["position_grid_step_ft"]),
        float(fixed["sig_p"]),
    )
    maximum_error = 0.0
    for row in range(len(delta_md)):
        for rate_index, rate in enumerate(rates):
            expected_offsets, expected_probabilities = (
                parent_position_kernel_probabilities(
                    rate * delta_md[row] - delta_z[row],
                    float(fixed["position_grid_step_ft"]),
                    float(fixed["sig_p"]),
                )
            )
            maximum_error = max(
                maximum_error,
                float(
                    np.max(
                        np.abs(
                            probabilities[row, rate_index]
                            - expected_probabilities
                        )
                    )
                ),
            )
            if not np.array_equal(
                offsets[row, rate_index],
                expected_offsets,
            ):
                maximum_error = math.inf
    return {
        "position_kernel_parent_max_abs_error": maximum_error,
        "pass": bool(maximum_error <= 1.0e-12),
    }


def evaluate_exp458_stage0a_gates(
    *,
    config: Mapping[str, Any],
    repeats: list[list[FrozenWell]],
    repeat_runtime: list[Mapping[str, Any]],
    parent_parity: list[Mapping[str, float]],
    numerical_contracts: Mapping[str, Mapping[str, Any]],
    scientific_contract_sha256: str,
) -> dict[str, Any]:
    gate = get_nested(config, "gates.stage0a_fixed4_runtime_equivalence")
    numerical = gate["numerical"]
    identity = gate["identity"]
    runtime_reference = gate["runtime_reference"]
    repeat_sha = [repeat_output_sha256(items) for items in repeats]
    slower_seconds = max(
        float(item["decode_wall_seconds"]) for item in repeat_runtime
    )
    parent_seconds = float(
        runtime_reference["exp444_fixed4_candidate_hmm_seconds"]
    )
    speedup = parent_seconds / slower_seconds
    fixed32_projection = (
        slower_seconds * int(runtime_reference["fixed32_batches"])
    )
    full_projection = (
        slower_seconds * int(runtime_reference["full_773_well_batches"])
    )
    peak_rss = max(
        float(item["process_tree_peak_rss_gb"]) for item in repeat_runtime
    )
    maximum_normalization_error = max(
        item.maximum_normalization_error
        for repeat in repeats
        for item in repeat
    )
    total_rows = [sum(len(item.row_idx) for item in repeat) for repeat in repeats]
    finite_rows = [
        sum(np.isfinite(item.candidate_prediction).sum() for item in repeat)
        for repeat in repeats
    ]
    finite_coverage = min(
        finite / rows if rows else 0.0
        for finite, rows in zip(finite_rows, total_rows, strict=True)
    )
    maximum_parent = {
        key: max(float(item[key]) for item in parent_parity)
        for key in parent_parity[0]
    }
    threads_one = all(
        int(worker["numba_threads"]) == 1
        and all(
            str(worker["thread_environment"][key]) == "1"
            for key in THREAD_ENVIRONMENT_KEYS
        )
        for runtime in repeat_runtime
        for worker in runtime["worker_audits"]
    )
    technical = {
        "scientific_contract_sha_exact": (
            bool(identity["scientific_contract_sha_exact"])
            and scientific_contract_sha256
            == EXPECTED_SCIENTIFIC_CONTRACT_SHA256
        ),
        "expected_wells": all(
            len(repeat) == int(identity["expected_wells"])
            for repeat in repeats
        ),
        "expected_rows": all(
            rows == int(identity["expected_rows"]) for rows in total_rows
        ),
        "expected_acceleration_states": (
            len(get_nested(config, "model.acceleration_state.values"))
            == int(identity["expected_acceleration_states"])
        ),
        "expected_rate_states": (
            int(get_nested(config, "model.fixed_from_exp441.n_rates"))
            == int(identity["expected_rate_states"])
        ),
        "parent_prediction_mean": (
            maximum_parent["parent_prediction_mean_max_abs_error_ft"]
            <= float(numerical["parent_prediction_mean_max_abs_error_ft"])
        ),
        "parent_prediction_std": (
            maximum_parent["parent_prediction_std_max_abs_error_ft"]
            <= float(numerical["parent_prediction_std_max_abs_error_ft"])
        ),
        "parent_acceleration_posterior": (
            maximum_parent["parent_acceleration_posterior_max_abs_error"]
            <= float(numerical["parent_acceleration_posterior_max_abs_error"])
        ),
        "parent_rate_diagnostic": (
            maximum_parent["parent_rate_diagnostic_max_abs_error"]
            <= float(numerical["parent_rate_diagnostic_max_abs_error"])
        ),
        "acceleration_transition": (
            float(
                numerical_contracts["acceleration"][
                    "acceleration_row_sum_max_error"
                ]
            )
            <= float(numerical["acceleration_transition_row_sum_max_error"])
        ),
        "ou_rate_kernel_parent": (
            float(numerical_contracts["ou_cache"]["maximum_abs_error"])
            <= float(numerical["ou_rate_kernel_parent_max_abs_error"])
            and bool(numerical_contracts["ou_cache"]["pass"])
        ),
        "position_kernel_parent": (
            float(
                numerical_contracts["position"][
                    "position_kernel_parent_max_abs_error"
                ]
            )
            <= float(numerical["position_kernel_parent_max_abs_error"])
            and bool(numerical_contracts["position"]["pass"])
        ),
        "small_dense_prediction": (
            float(
                numerical_contracts["dense"][
                    "posterior_prediction_max_abs_error"
                ]
            )
            <= float(numerical["small_dense_prediction_max_abs_error_ft"])
        ),
        "small_dense_posterior": (
            float(
                numerical_contracts["dense"][
                    "posterior_acceleration_max_abs_error"
                ]
            )
            <= float(numerical["small_dense_posterior_max_abs_error"])
            and bool(numerical_contracts["dense"]["pass"])
        ),
        "posterior_normalization": (
            maximum_normalization_error
            <= float(numerical["posterior_normalization_max_error"])
        ),
        "finite_coverage": (
            finite_coverage >= float(numerical["finite_coverage_min"])
        ),
        "repeat_prediction_sha": (
            repeat_sha[0]["prediction_decompressed_sha256"]
            == repeat_sha[1]["prediction_decompressed_sha256"]
        ),
        "repeat_posterior_sha": (
            repeat_sha[0]["posterior_bundle_sha256"]
            == repeat_sha[1]["posterior_bundle_sha256"]
        ),
        "repeat_diagnostic_sha": (
            repeat_sha[0]["diagnostic_sha256"]
            == repeat_sha[1]["diagnostic_sha256"]
        ),
        "minimum_speedup": (
            speedup
            >= float(runtime_reference["minimum_speedup_using_slower_repeat"])
        ),
        "projected_fixed32_runtime": (
            fixed32_projection
            <= float(runtime_reference["projected_fixed32_seconds_max"])
        ),
        "projected_full_runtime": (
            full_projection
            <= float(runtime_reference["projected_full_seconds_max"])
        ),
        "peak_rss": (
            peak_rss <= float(runtime_reference["peak_rss_gb_max"])
        ),
        "effective_outer_workers": all(
            int(item["effective_outer_workers"])
            == int(
                get_nested(
                    config,
                    "runtime_engine.parallel.outer_well_workers",
                )
            )
            for item in repeat_runtime
        ),
        "worker_threads_one": threads_one,
        "truth_role_fold_episode_cause_reads_before_freeze": (
            max(
                int(worker["forbidden_reads_before_all_freeze"])
                for runtime in repeat_runtime
                for worker in runtime["worker_audits"]
            )
            <= int(
                gate["leakage"][
                    "truth_role_fold_episode_cause_reads_before_freeze_max"
                ]
            )
        ),
    }
    diagnostics = {
        "repeat_output_sha256": repeat_sha,
        "repeat_runtime": list(repeat_runtime),
        "parent_parity": list(parent_parity),
        "maximum_parent_parity_error": maximum_parent,
        "slower_repeat_decode_wall_seconds": slower_seconds,
        "exp444_fixed4_reference_seconds": parent_seconds,
        "speedup_vs_exp444": speedup,
        "projected_fixed32_seconds": fixed32_projection,
        "projected_full_seconds": full_projection,
        "process_tree_peak_rss_gb": peak_rss,
        "maximum_normalization_error": maximum_normalization_error,
        "finite_coverage": finite_coverage,
        "rows_per_repeat": total_rows,
        "numerical_contracts": dict(numerical_contracts),
    }
    all_pass = bool(all(technical.values()))
    return {
        "technical": technical,
        "diagnostics": diagnostics,
        "all_pass": all_pass,
        "stage0b_eligible_pending_separate_user_approval": all_pass,
        "fail_action": None if all_pass else str(gate["fail_action"]),
    }


# %% [markdown]
# ## 10. Guarded Stage 0A orchestration

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP458_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp458 Stage 0A must run on Kaggle private CPU")


def run_stage0a(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific = validate_scientific_contract(config)
    engine_contract = runtime_engine_contract(config)
    scientific_sha = hashlib.sha256(
        stable_json_bytes(scientific)
    ).hexdigest()
    engine_contract_sha = hashlib.sha256(
        stable_json_bytes(engine_contract)
    ).hexdigest()
    fixed = get_nested(config, "model.fixed_from_exp441")
    acceleration = get_nested(config, "model.acceleration_state")
    numerical_contracts = {
        "acceleration": acceleration_transition_contract(acceleration),
        "zero_acceleration": zero_acceleration_kernel_parity_contract(fixed),
        "ou_cache": exact_cache_kernel_contract(fixed, acceleration),
        "position": position_kernel_contract(fixed),
        "dense": brute_force_posterior_contract(fixed, acceleration),
    }
    if not all(
        bool(contract["pass"]) for contract in numerical_contracts.values()
    ):
        raise RuntimeError(
            f"exp458 numerical preflight failed: {numerical_contracts}"
        )

    selection_ledger = LeakageLedger(expected_wells=4)
    wells, selection_input = select_stage0a_wells(config, selection_ledger)
    raw_dir = train_data_dir(config)
    parent_paths = resolve_parent_fixed4_artifacts(config)
    repeat_count = int(
        get_nested(config, "execution.stage0a_repeat_count")
    )
    worker_count = int(
        get_nested(config, "runtime_engine.parallel.outer_well_workers")
    )
    repeats: list[list[FrozenWell]] = []
    repeat_runtime: list[Mapping[str, Any]] = []
    repeat_ledgers: list[LeakageLedger] = []
    with create_outer_executor(worker_count) as executor:
        for repeat_index in range(1, repeat_count + 1):
            frozen, runtime, ledger = run_parallel_repeat(
                repeat_index=repeat_index,
                executor=executor,
                wells=wells,
                raw_dir=raw_dir,
                fixed=fixed,
                acceleration=acceleration,
            )
            repeats.append(frozen)
            repeat_runtime.append(runtime)
            repeat_ledgers.append(ledger)
            print(
                json.dumps(
                    {
                        "event": "exp458_stage0a_repeat_complete",
                        "repeat": repeat_index,
                        "decode_wall_seconds": runtime[
                            "decode_wall_seconds"
                        ],
                        "process_tree_peak_rss_gb": runtime[
                            "process_tree_peak_rss_gb"
                        ],
                        "effective_outer_worker_pids": runtime[
                            "effective_outer_worker_pids"
                        ],
                        "output_sha256": runtime["output_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(repeats) != 2 or not all(
        ledger.all_frozen for ledger in repeat_ledgers
    ):
        raise RuntimeError("exp458 requires two fully frozen candidate repeats")
    parent_frames, parent_evidence = (
        load_parent_fixed4_after_candidate_freeze(
            config,
            parent_paths,
            all_candidate_wells_frozen=True,
        )
    )
    parent_parity = [
        evaluate_saved_parent_parity(repeat, parent_frames)
        for repeat in repeats
    ]
    gates = evaluate_exp458_stage0a_gates(
        config=config,
        repeats=repeats,
        repeat_runtime=repeat_runtime,
        parent_parity=parent_parity,
        numerical_contracts=numerical_contracts,
        scientific_contract_sha256=scientific_sha,
    )

    output = artifacts_dir()
    repeat_artifacts: list[dict[str, Any]] = []
    for repeat_index, frozen in enumerate(repeats, start=1):
        repeat_artifacts.append(
            {
                "prediction": write_deterministic_gzip_csv(
                    output
                    / f"{EXPERIMENT_NAME}_stage0a_repeat{repeat_index}_predictions.csv.gz",
                    prediction_frame(frozen),
                ),
                "acceleration_posterior": write_deterministic_gzip_csv(
                    output
                    / f"{EXPERIMENT_NAME}_stage0a_repeat{repeat_index}_acceleration_posterior.csv.gz",
                    acceleration_frame(frozen),
                ),
                "target_free_diagnostics": write_deterministic_gzip_csv(
                    output
                    / f"{EXPERIMENT_NAME}_stage0a_repeat{repeat_index}_target_free_diagnostics.csv.gz",
                    diagnostic_frame(frozen),
                ),
            }
        )
    for repeat_index, artifacts in enumerate(repeat_artifacts, start=1):
        for label, artifact in artifacts.items():
            if artifact["logical_sha256"] != artifact["readback_logical_sha256"]:
                raise RuntimeError(
                    f"repeat {repeat_index} {label} readback SHA mismatch"
                )

    runtime_manifest = {
        "experiment": EXPERIMENT_NAME,
        "scientific_contract_sha256": scientific_sha,
        "runtime_engine_contract_sha256": engine_contract_sha,
        "source_contract_sha256": array_bundle_sha256(
            scientific_contract=np.frombuffer(
                bytes.fromhex(scientific_sha),
                dtype=np.uint8,
            ),
            runtime_engine_contract=np.frombuffer(
                bytes.fromhex(engine_contract_sha),
                dtype=np.uint8,
            ),
        ),
        "selection": selection_input,
        "parent_fixed4": parent_evidence,
        "repeats": list(repeat_runtime),
        "combined_input_sha256": [
            combined_well_sha(repeat, "input_sha256")
            for repeat in repeats
        ],
        "combined_prepared_input_sha256": [
            combined_well_sha(repeat, "prepared_input_sha256")
            for repeat in repeats
        ],
        "combined_delta_md_key_ledger_sha256": [
            combined_well_sha(repeat, "delta_md_key_ledger_sha256")
            for repeat in repeats
        ],
        "combined_position_kernel_sha256": [
            combined_well_sha(repeat, "position_kernel_sha256")
            for repeat in repeats
        ],
        "combined_forward_scale_sha256": [
            combined_well_sha(repeat, "forward_scale_sha256")
            for repeat in repeats
        ],
        "model_sha256": None,
        "submission_sha256": None,
        "kernel_version": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "versions": runtime_versions(),
    }
    runtime_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0a_runtime_manifest.json",
        runtime_manifest,
    )
    eligible = bool(
        gates["stage0b_eligible_pending_separate_user_approval"]
    )
    elapsed = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0a_all_gates_pass_pending_separate_stage0b_approval"
            if eligible
            else "stage0a_fail_closed"
        ),
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_sha,
        "runtime_engine_contract_sha256": engine_contract_sha,
        "gates": gates,
        "runtime": {
            "elapsed_seconds": elapsed,
            "process_tree_peak_rss_gb": gates["diagnostics"][
                "process_tree_peak_rss_gb"
            ],
            "outer_workers": worker_count,
            "inner_threads": 1,
            "cpu_only": True,
            "repeat_runtime": repeat_runtime,
        },
        "artifacts": {
            "repeat_outputs": repeat_artifacts,
            "runtime_manifest": runtime_artifact,
        },
        "parent_control_rerun": False,
        "stage0b": {
            "eligible": eligible,
            "execution_approved": False,
            "requires_separate_user_approval": True,
        },
        "stage1": False,
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0a_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": summary["status"],
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "stage": "stage0a_fixed4_runtime_equivalence",
            "cv": None,
            "lb": None,
            "fixed4_is_target_free_runtime_audit": True,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_sha,
        "runtime_engine_contract_sha256": engine_contract_sha,
        "technical_gates": gates["technical"],
        "stage0b_eligible_pending_separate_user_approval": eligible,
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %%
if __name__ == "__main__":
    CONFIG = load_config()
    EXECUTION_COUNTS = validate_execution_contract(
        CONFIG,
        require_run_authorization=False,
    )
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    RUNTIME_ENGINE_CONTRACT = runtime_engine_contract(CONFIG)
    print(
        json.dumps(
            {
                "event": "exp458_implementation_contract",
                "experiment": EXPERIMENT_NAME,
                "status": get_nested(CONFIG, "experiment.status"),
                "selected_stage": get_nested(
                    CONFIG,
                    "execution.selected_stage",
                ),
                "execution_counts": EXECUTION_COUNTS,
                "scientific_contract_sha256": hashlib.sha256(
                    stable_json_bytes(SCIENTIFIC_CONTRACT)
                ).hexdigest(),
                "runtime_engine_contract_sha256": hashlib.sha256(
                    stable_json_bytes(RUNTIME_ENGINE_CONTRACT)
                ).hexdigest(),
                "stage0a_run_authorized": bool(
                    get_nested(
                        CONFIG,
                        "execution.stage0a_run_authorized",
                    )
                ),
                "stage0b_run_authorized": bool(
                    get_nested(
                        CONFIG,
                        "execution.stage0b_run_authorized",
                    )
                ),
                "inference_authorized": bool(
                    get_nested(
                        CONFIG,
                        "execution.inference_authorized",
                    )
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if bool(get_nested(CONFIG, "execution.stage0a_run_authorized", False)):
        SUMMARY = run_stage0a(CONFIG)
