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
# # exp444 acceleration-state exact HMM — Stage 0A train-side preflight
#
# This independent high-risk hypothesis keeps the exp441 full-support OU rate
# law and every exp209 position/emission/prior/readout contract, then adds one
# frozen three-state acceleration variable. The acceleration state is updated
# first, its destination value shifts the OU destination-rate mean, the
# destination rate advances TVT, and current-row GR is used exactly once.
#
# This source implements only the target-free fixed4 exactness/runtime
# preflight. It does not authorize canonical-notebook adoption, a Kaggle run,
# Stage 0B, Stage 1, inference, or submission.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable contracts
# 2. Notebook-safe paths, SHA helpers, and leakage guard
# 3. Identity-only fixed4 selection and target-free inputs
# 4. Exact exp209 input preparation
# 5. Acceleration and acceleration-conditioned OU kernels
# 6. Factorized exact forward-backward
# 7. Dense brute-force and zero-acceleration contracts
# 8. Target-free prediction/posterior freeze
# 9. Stage 0A technical gates and generated artifacts
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numba import njit, prange, set_num_threads

EXPERIMENT_NAME = "exp444_acceleration_state_exact_hmm"
STRUCTURAL_PARENT = "exp441_full_support_ou_rate_transition_hmm"
ROOT_PARENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
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
        raise ValueError("wrong exp444 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp444 route must remain pf_beam")
    if get_nested(config, "lineage.structural_parent") != STRUCTURAL_PARENT:
        raise ValueError("exp444 structural parent changed")
    if get_nested(config, "lineage.root_parent") != ROOT_PARENT:
        raise ValueError("exp444 root parent changed")
    if not bool(get_nested(config, "design.independent_execution_hypothesis", False)):
        raise ValueError("exp444 must remain an independent execution hypothesis")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp444 implementation is not authorized")
    for key, label in (
        ("stage0b_run_authorized", "Stage 0B"),
        ("stage1_run_authorized", "Stage 1"),
        ("inference_authorized", "inference"),
        ("submission_authorized", "submission"),
    ):
        if bool(get_nested(config, f"execution.{key}", True)):
            raise ValueError(f"{label} must remain disabled")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu", True)):
        raise ValueError("exp444 is CPU-only")
    if bool(get_nested(config, "data.exp441_saved_control.regenerate", True)):
        raise ValueError("terminal-closed exp441 control must not be regenerated")
    if bool(get_nested(config, "data.exp209_saved_root_reference.regenerate", True)):
        raise ValueError("saved exp209 root reference must not be regenerated")

    expected = {
        "scientific_variants": 1,
        "stage0a_candidate_hmm_well_runs": 4,
        "stage0b_total_candidate_hmm_well_runs": 32,
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
        raise ValueError(f"exp444 execution contract changed: {observed} != {expected}")

    if require_run_authorization:
        if (
            get_nested(config, "execution.selected_stage")
            != "stage0a_fixed4_runtime_contract"
        ):
            raise RuntimeError("exp444 selected_stage must remain Stage 0A")
        if not bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ):
            raise RuntimeError(
                "exp444 Stage 0A requires separate canonical notebook approval"
            )
        if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
            raise RuntimeError("exp444 Stage 0A requires separate package approval")
        if not bool(get_nested(config, "execution.stage0a_run_authorized", False)):
            raise RuntimeError(
                "exp444 implementation approval does not authorize Stage 0A execution"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("exp444 run_hmm remains fail-closed")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError("exp444 prediction creation remains fail-closed")
        if bool(get_nested(config, "execution.create_submission", True)):
            raise ValueError("exp444 Stage 0A must not create a submission")
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
            f"exp444 acceleration contract changed: "
            f"{acceleration} != {expected_acceleration}"
        )
    variants = list(get_nested(config, "model.active_scientific_variants") or [])
    if variants != ["three_state_persistent_acceleration"]:
        raise ValueError("exp444 must contain one frozen acceleration candidate")
    return {
        "fixed_from_exp441": fixed,
        "acceleration_state": acceleration,
        "active_scientific_variants": variants,
        "forbidden": list(get_nested(config, "model.forbidden") or []),
    }


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
    raise FileNotFoundError("exp444 config.yaml was not found")


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
# The fixed32 manifest is opened with `usecols=["well"]` only. The fixed4 order
# is SHA256("exp444_runtime_preflight" + well). Role, fold, suffix counts,
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
    count = int(get_nested(config, "data.stage0a_selection.count"))
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


@njit(cache=True, nogil=True, parallel=True)
def precompute_acceleration_ou_log_kernels(
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
    output = np.full(
        (time_count, acceleration_count, rate_count, rate_count),
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
                            ] = 0.0
                    if mean == edges[-1]:
                        output[
                            time_index,
                            acceleration_index,
                            source_rate,
                            rate_count - 1,
                        ] = 0.0
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
                        ] = math.log(probability)
    return output


def full_support_ou_rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    logs = precompute_acceleration_ou_log_kernels(
        np.asarray([delta_md], dtype=np.float64),
        np.asarray(rates, dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        float(sig_r),
        float(momentum),
    )
    return np.exp(logs[0, 0])


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


# %% [markdown]
# ## 6. Factorized exact forward-backward
#
# The transition is evaluated in the preregistered order:
# acceleration -> rate -> TVT position -> current GR emission. Forward and
# backward passes use the same factorization; no GR evidence is reused.

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm3_acceleration_ou(
    emission,
    delta_md,
    delta_z,
    position_step,
    rates,
    accelerations,
    acceleration_log_transition,
    rate_log_kernels,
    sig_p,
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
    neg = np.float32(-1.0e18)
    alpha = np.full(
        (time_count, position_count, rate_count, acceleration_count),
        neg,
        np.float32,
    )
    previous = np.full(
        (position_count, rate_count, acceleration_count),
        neg,
        np.float32,
    )
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
                    previous[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = np.float32(
                        position_log
                        - 0.5 * delta_rate * delta_rate
                        + math.log(prior)
                    )

    acceleration_updated = np.empty(
        (position_count, rate_count, acceleration_count),
        np.float32,
    )
    rate_updated = np.empty_like(acceleration_updated)
    predictive = np.empty_like(acceleration_updated)
    current = np.empty_like(acceleration_updated)
    predictive_rate_mean = np.empty(time_count, np.float64)
    filtered_rate_mean = np.empty(time_count, np.float64)
    predictive_acceleration_mean = np.empty(time_count, np.float64)
    filtered_acceleration_mean = np.empty(time_count, np.float64)
    maximum_forward_normalization_error = 0.0

    for time_index in range(time_count):
        for position_index in prange(position_count):
            for source_rate in range(rate_count):
                for destination_acceleration in range(acceleration_count):
                    best = neg
                    for source_acceleration in range(acceleration_count):
                        transition_log = acceleration_log_transition[
                            source_acceleration,
                            destination_acceleration,
                        ]
                        if not np.isfinite(transition_log):
                            continue
                        value = (
                            previous[
                                position_index,
                                source_rate,
                                source_acceleration,
                            ]
                            + transition_log
                        )
                        if value > best:
                            best = value
                    if best > neg / 2:
                        total = 0.0
                        for source_acceleration in range(acceleration_count):
                            transition_log = acceleration_log_transition[
                                source_acceleration,
                                destination_acceleration,
                            ]
                            if np.isfinite(transition_log):
                                total += math.exp(
                                    previous[
                                        position_index,
                                        source_rate,
                                        source_acceleration,
                                    ]
                                    + transition_log
                                    - best
                                )
                        acceleration_updated[
                            position_index,
                            source_rate,
                            destination_acceleration,
                        ] = np.float32(best + math.log(total))
                    else:
                        acceleration_updated[
                            position_index,
                            source_rate,
                            destination_acceleration,
                        ] = neg

        for position_index in prange(position_count):
            for destination_rate in range(rate_count):
                for destination_acceleration in range(acceleration_count):
                    best = neg
                    for source_rate in range(rate_count):
                        transition_log = rate_log_kernels[
                            time_index,
                            destination_acceleration,
                            source_rate,
                            destination_rate,
                        ]
                        if not np.isfinite(transition_log):
                            continue
                        value = (
                            acceleration_updated[
                                position_index,
                                source_rate,
                                destination_acceleration,
                            ]
                            + transition_log
                        )
                        if value > best:
                            best = value
                    if best > neg / 2:
                        total = 0.0
                        for source_rate in range(rate_count):
                            transition_log = rate_log_kernels[
                                time_index,
                                destination_acceleration,
                                source_rate,
                                destination_rate,
                            ]
                            if np.isfinite(transition_log):
                                total += math.exp(
                                    acceleration_updated[
                                        position_index,
                                        source_rate,
                                        destination_acceleration,
                                    ]
                                    + transition_log
                                    - best
                                )
                        rate_updated[
                            position_index,
                            destination_rate,
                            destination_acceleration,
                        ] = np.float32(best + math.log(total))
                    else:
                        rate_updated[
                            position_index,
                            destination_rate,
                            destination_acceleration,
                        ] = neg

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
            for destination_acceleration in range(acceleration_count):
                for destination_position in range(position_count):
                    best = neg
                    for kernel_index in range(5):
                        source_position = (
                            destination_position - offsets[kernel_index]
                        )
                        if 0 <= source_position < position_count:
                            value = (
                                rate_updated[
                                    source_position,
                                    destination_rate,
                                    destination_acceleration,
                                ]
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
                                total += math.exp(
                                    rate_updated[
                                        source_position,
                                        destination_rate,
                                        destination_acceleration,
                                    ]
                                    + position_log_kernel[kernel_index]
                                    - best
                                )
                        value = best + math.log(total)
                        predictive[
                            destination_position,
                            destination_rate,
                            destination_acceleration,
                        ] = np.float32(value)
                        current[
                            destination_position,
                            destination_rate,
                            destination_acceleration,
                        ] = np.float32(
                            value
                            + emission_lambda
                            * emission[time_index, destination_position]
                        )
                    else:
                        predictive[
                            destination_position,
                            destination_rate,
                            destination_acceleration,
                        ] = neg
                        current[
                            destination_position,
                            destination_rate,
                            destination_acceleration,
                        ] = neg

        predictive_best = np.max(predictive)
        filtered_best = np.max(current)
        predictive_total = 0.0
        filtered_total = 0.0
        predictive_rate_total = 0.0
        filtered_rate_total = 0.0
        predictive_acceleration_total = 0.0
        filtered_acceleration_total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    predictive_probability = math.exp(
                        float(
                            predictive[
                                position_index,
                                rate_index,
                                acceleration_index,
                            ]
                        )
                        - float(predictive_best)
                    )
                    filtered_probability = math.exp(
                        float(
                            current[
                                position_index,
                                rate_index,
                                acceleration_index,
                            ]
                        )
                        - float(filtered_best)
                    )
                    predictive_total += predictive_probability
                    filtered_total += filtered_probability
                    predictive_rate_total += (
                        predictive_probability * rates[rate_index]
                    )
                    filtered_rate_total += (
                        filtered_probability * rates[rate_index]
                    )
                    predictive_acceleration_total += (
                        predictive_probability
                        * accelerations[acceleration_index]
                    )
                    filtered_acceleration_total += (
                        filtered_probability
                        * accelerations[acceleration_index]
                    )
                    alpha[
                        time_index,
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = current[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ]
                    previous[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ] = current[
                        position_index,
                        rate_index,
                        acceleration_index,
                    ]
        predictive_rate_mean[time_index] = (
            predictive_rate_total / predictive_total
        )
        filtered_rate_mean[time_index] = (
            filtered_rate_total / filtered_total
        )
        predictive_acceleration_mean[time_index] = (
            predictive_acceleration_total / predictive_total
        )
        filtered_acceleration_mean[time_index] = (
            filtered_acceleration_total / filtered_total
        )
        predictive_check = 0.0
        filtered_check = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    predictive_check += math.exp(
                        float(
                            predictive[
                                position_index,
                                rate_index,
                                acceleration_index,
                            ]
                        )
                        - float(predictive_best)
                    ) / predictive_total
                    filtered_check += math.exp(
                        float(
                            current[
                                position_index,
                                rate_index,
                                acceleration_index,
                            ]
                        )
                        - float(filtered_best)
                    ) / filtered_total
        maximum_forward_normalization_error = max(
            maximum_forward_normalization_error,
            abs(predictive_check - 1.0),
            abs(filtered_check - 1.0),
        )

    final_best = np.max(alpha[time_count - 1])
    final_total = 0.0
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            for acceleration_index in range(acceleration_count):
                final_total += math.exp(
                    float(
                        alpha[
                            time_count - 1,
                            position_index,
                            rate_index,
                            acceleration_index,
                        ]
                    )
                    - float(final_best)
                )
    log_likelihood = float(final_best) + math.log(final_total)

    posterior_position = np.zeros(
        (time_count, position_count),
        dtype=np.float64,
    )
    posterior_rate = np.zeros((time_count, rate_count), dtype=np.float64)
    posterior_acceleration = np.zeros(
        (time_count, acceleration_count),
        dtype=np.float64,
    )
    beta_next = np.zeros(
        (position_count, rate_count, acceleration_count),
        np.float32,
    )

    for time_index in range(time_count - 1, -1, -1):
        values = alpha[time_index] + beta_next
        best = np.max(values)
        total = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    total += math.exp(
                        float(
                            values[
                                position_index,
                                rate_index,
                                acceleration_index,
                            ]
                        )
                        - float(best)
                    )
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                for acceleration_index in range(acceleration_count):
                    probability = (
                        math.exp(
                            float(
                                values[
                                    position_index,
                                    rate_index,
                                    acceleration_index,
                                ]
                            )
                            - float(best)
                        )
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
            for destination_acceleration in range(acceleration_count):
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
                                * emission[
                                    time_index,
                                    destination_position,
                                ]
                                + beta_next[
                                    destination_position,
                                    destination_rate,
                                    destination_acceleration,
                                ]
                            )
                            if value > best:
                                best = value
                    if best > neg / 2:
                        subtotal = 0.0
                        for kernel_index in range(5):
                            destination_position = (
                                source_position + offsets[kernel_index]
                            )
                            if 0 <= destination_position < position_count:
                                subtotal += math.exp(
                                    position_log_kernel[kernel_index]
                                    + emission_lambda
                                    * emission[
                                        time_index,
                                        destination_position,
                                    ]
                                    + beta_next[
                                        destination_position,
                                        destination_rate,
                                        destination_acceleration,
                                    ]
                                    - best
                                )
                        beta_position[
                            source_position,
                            destination_rate,
                            destination_acceleration,
                        ] = np.float32(best + math.log(subtotal))
                    else:
                        beta_position[
                            source_position,
                            destination_rate,
                            destination_acceleration,
                        ] = neg

        beta_rate = np.empty_like(beta_next)
        for position_index in prange(position_count):
            for source_rate in range(rate_count):
                for destination_acceleration in range(acceleration_count):
                    best = neg
                    for destination_rate in range(rate_count):
                        transition_log = rate_log_kernels[
                            time_index,
                            destination_acceleration,
                            source_rate,
                            destination_rate,
                        ]
                        if not np.isfinite(transition_log):
                            continue
                        value = (
                            transition_log
                            + beta_position[
                                position_index,
                                destination_rate,
                                destination_acceleration,
                            ]
                        )
                        if value > best:
                            best = value
                    if best > neg / 2:
                        subtotal = 0.0
                        for destination_rate in range(rate_count):
                            transition_log = rate_log_kernels[
                                time_index,
                                destination_acceleration,
                                source_rate,
                                destination_rate,
                            ]
                            if np.isfinite(transition_log):
                                subtotal += math.exp(
                                    transition_log
                                    + beta_position[
                                        position_index,
                                        destination_rate,
                                        destination_acceleration,
                                    ]
                                    - best
                                )
                        beta_rate[
                            position_index,
                            source_rate,
                            destination_acceleration,
                        ] = np.float32(best + math.log(subtotal))
                    else:
                        beta_rate[
                            position_index,
                            source_rate,
                            destination_acceleration,
                        ] = neg

        beta_current = np.empty_like(beta_next)
        for position_index in prange(position_count):
            for source_rate in range(rate_count):
                for source_acceleration in range(acceleration_count):
                    best = neg
                    for destination_acceleration in range(acceleration_count):
                        transition_log = acceleration_log_transition[
                            source_acceleration,
                            destination_acceleration,
                        ]
                        if not np.isfinite(transition_log):
                            continue
                        value = (
                            transition_log
                            + beta_rate[
                                position_index,
                                source_rate,
                                destination_acceleration,
                            ]
                        )
                        if value > best:
                            best = value
                    if best > neg / 2:
                        subtotal = 0.0
                        for destination_acceleration in range(
                            acceleration_count
                        ):
                            transition_log = acceleration_log_transition[
                                source_acceleration,
                                destination_acceleration,
                            ]
                            if np.isfinite(transition_log):
                                subtotal += math.exp(
                                    transition_log
                                    + beta_rate[
                                        position_index,
                                        source_rate,
                                        destination_acceleration,
                                    ]
                                    - best
                                )
                        beta_current[
                            position_index,
                            source_rate,
                            source_acceleration,
                        ] = np.float32(best + math.log(subtotal))
                    else:
                        beta_current[
                            position_index,
                            source_rate,
                            source_acceleration,
                        ] = neg
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
        log_likelihood,
        max(
            maximum_forward_normalization_error,
            maximum_posterior_normalization_error,
        ),
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
    acceleration_log_transition = np.full_like(
        acceleration_transition,
        -np.inf,
    )
    positive = acceleration_transition > 0.0
    acceleration_log_transition[positive] = np.log(
        acceleration_transition[positive]
    )
    rate_log_kernels = precompute_acceleration_ou_log_kernels(
        delta_md,
        rates,
        accelerations,
        float(fixed["sig_r"]),
        float(fixed["rate_momentum"]),
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
    ) = _hmm3_acceleration_ou(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        delta_md,
        np.asarray(prepared["dz"], dtype=np.float64),
        float(fixed["position_grid_step_ft"]),
        rates,
        accelerations,
        acceleration_log_transition,
        rate_log_kernels,
        float(fixed["sig_p"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        np.asarray(acceleration["initial_probability"], dtype=np.float64),
        float(fixed["emission_lambda"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = posterior_position @ grid
    posterior_variance = posterior_position @ (grid**2) - posterior_mean**2
    posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0))
    posterior_rate_mean = posterior_rate @ rates
    posterior_rate_variance = (
        posterior_rate @ (rates**2) - posterior_rate_mean**2
    )
    posterior_rate_std = np.sqrt(
        np.maximum(posterior_rate_variance, 0.0)
    )
    posterior_acceleration_mean = posterior_acceleration @ accelerations
    posterior_acceleration_nonzero_mass = (
        posterior_acceleration[:, 0] + posterior_acceleration[:, 2]
    )
    joint_transition_sha256 = array_bundle_sha256(
        delta_md=delta_md,
        rates=rates,
        accelerations=accelerations,
        acceleration_transition=acceleration_transition,
        rate_log_kernels=rate_log_kernels,
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
    observed = precompute_acceleration_ou_log_kernels(
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
            float(np.max(np.abs(np.exp(observed[row]) - expected))),
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
    rate_kernels = np.exp(
        precompute_acceleration_ou_log_kernels(
            delta_md,
            rates,
            accelerations,
            float(fixed["sig_r"]),
            float(fixed["rate_momentum"]),
        )
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
    prediction_sha256: str
    acceleration_posterior_sha256: str
    diagnostic_sha256: str
    maximum_normalization_error: float
    log_likelihood: float
    elapsed_seconds: float


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    fixed: Mapping[str, Any],
    acceleration: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, fixed)
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
# ## 9. Stage 0A technical gates and generated artifacts

# %%
def evaluate_stage0a_gates(
    *,
    config: Mapping[str, Any],
    frozen_wells: list[FrozenWell],
    acceleration_contract: Mapping[str, Any],
    zero_acceleration_contract: Mapping[str, Any],
    brute_force_contract: Mapping[str, Any],
    ledger: LeakageLedger,
    stage0a_elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(
        config,
        "gates.stage0a_fixed4_runtime.technical",
    )
    hmm_seconds = float(sum(item.elapsed_seconds for item in frozen_wells))
    fixed4_count = int(get_nested(config, "execution.stage0a_candidate_hmm_well_runs"))
    fixed32_count = int(
        get_nested(config, "execution.stage0b_total_candidate_hmm_well_runs")
    )
    full_count = int(get_nested(config, "execution.stage1_candidate_hmm_well_runs"))
    fixed32_projection = (
        hmm_seconds * fixed32_count / fixed4_count
    )
    full_projection = hmm_seconds * full_count / fixed4_count
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    finite_rows = int(
        sum(np.isfinite(item.candidate_prediction).sum() for item in frozen_wells)
    )
    finite_coverage = finite_rows / total_rows if total_rows else 0.0
    maximum_normalization_error = max(
        item.maximum_normalization_error for item in frozen_wells
    )
    rss = peak_rss_gb()
    technical = {
        "expected_wells": (
            len(frozen_wells) == int(technical_config["expected_wells"])
        ),
        "expected_acceleration_states": (
            len(
                get_nested(
                    config,
                    "model.acceleration_state.values",
                )
            )
            == int(technical_config["expected_acceleration_states"])
        ),
        "finite_coverage": (
            finite_coverage >= float(technical_config["finite_coverage_min"])
        ),
        "acceleration_row_sum": (
            float(acceleration_contract["acceleration_row_sum_max_error"])
            <= float(technical_config["acceleration_row_sum_max_error"])
        ),
        "zero_acceleration_rate_kernel_parity_vs_exp441": (
            float(
                zero_acceleration_contract[
                    "zero_acceleration_rate_kernel_parity_vs_exp441_max_abs_error"
                ]
            )
            <= float(
                technical_config[
                    "zero_acceleration_rate_kernel_parity_vs_exp441_max_abs_error"
                ]
            )
        ),
        "posterior_normalization": (
            maximum_normalization_error
            <= float(technical_config["posterior_normalization_max_error"])
        ),
        "brute_force_posterior_prediction": (
            float(
                brute_force_contract[
                    "posterior_prediction_max_abs_error"
                ]
            )
            <= float(
                technical_config[
                    "brute_force_posterior_prediction_max_abs_error"
                ]
            )
            and bool(brute_force_contract["pass"])
        ),
        "truth_role_fold_episode_reads_before_freeze": (
            ledger.forbidden_reads_before_all_freeze
            <= int(
                technical_config[
                    "truth_role_fold_episode_reads_before_freeze_max"
                ]
            )
        ),
        "projected_stage0b_runtime": (
            fixed32_projection
            <= float(
                technical_config[
                    "projected_stage0b_runtime_seconds_max"
                ]
            )
        ),
        "projected_stage1_runtime": (
            full_projection
            <= float(
                technical_config[
                    "projected_stage1_runtime_seconds_max"
                ]
            )
        ),
        "peak_rss": rss <= float(technical_config["peak_rss_gb_max"]),
    }
    diagnostics = {
        "stage0a_wells": len(frozen_wells),
        "stage0a_rows": total_rows,
        "finite_coverage": finite_coverage,
        "candidate_hmm_seconds": hmm_seconds,
        "stage0a_elapsed_seconds": stage0a_elapsed_seconds,
        "projected_stage0b_runtime_seconds": fixed32_projection,
        "projected_stage1_runtime_seconds": full_projection,
        "peak_rss_gb": rss,
        "maximum_normalization_error": maximum_normalization_error,
        **dict(acceleration_contract),
        **dict(zero_acceleration_contract),
        **dict(brute_force_contract),
        "forbidden_reads_before_all_freeze": (
            ledger.forbidden_reads_before_all_freeze
        ),
    }
    all_pass = bool(all(technical.values()))
    return {
        "technical": technical,
        "diagnostics": diagnostics,
        "stage0b_eligible_pending_separate_user_approval": all_pass,
        "fail_action": (
            None
            if all_pass
            else get_nested(
                config,
                "gates.stage0a_fixed4_runtime.fail_action",
            )
        ),
    }


# %% [markdown]
# ## 10. Guarded Kaggle CPU orchestration

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP444_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp444 Stage 0A must run on Kaggle CPU")


def run_stage0a(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha256 = hashlib.sha256(
        stable_json_bytes(scientific_contract)
    ).hexdigest()
    fixed = get_nested(config, "model.fixed_from_exp441")
    acceleration = get_nested(config, "model.acceleration_state")
    acceleration_contract = acceleration_transition_contract(acceleration)
    zero_acceleration_contract = (
        zero_acceleration_kernel_parity_contract(fixed)
    )
    brute_force_contract = brute_force_posterior_contract(
        fixed,
        acceleration,
    )
    for label, contract in (
        ("acceleration", acceleration_contract),
        ("zero-acceleration", zero_acceleration_contract),
        ("brute-force", brute_force_contract),
    ):
        if not bool(contract["pass"]):
            raise RuntimeError(f"{label} numerical contract failed: {contract}")

    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    ledger = LeakageLedger(
        expected_wells=int(
            get_nested(config, "execution.stage0a_candidate_hmm_well_runs")
        )
    )
    wells, selection_input = select_stage0a_wells(config, ledger)
    raw_dir = train_data_dir(config)
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    frozen_wells: list[FrozenWell] = []
    for well_index, well in enumerate(wells, start=1):
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            fixed=fixed,
            acceleration=acceleration,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(f"Stage 0A hard runtime exceeded: {elapsed}")
        if peak_rss_gb() > hard_rss:
            raise MemoryError(f"Stage 0A RSS exceeded: {peak_rss_gb()}")
        print(
            json.dumps(
                {
                    "event": "exp444_stage0a_progress",
                    "well_index": well_index,
                    "well_count": len(wells),
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "hmm_seconds": frozen.elapsed_seconds,
                    "elapsed_seconds": elapsed,
                    "peak_rss_gb": peak_rss_gb(),
                    "joint_transition_sha256": (
                        frozen.joint_transition_sha256
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all Stage 0A wells were frozen")

    output = artifacts_dir()
    predictions = prediction_frame(frozen_wells)
    acceleration_posteriors = acceleration_frame(frozen_wells)
    diagnostics = diagnostic_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0a_predictions.csv.gz",
        predictions,
    )
    acceleration_artifact = write_deterministic_gzip_csv(
        output
        / f"{EXPERIMENT_NAME}_stage0a_acceleration_posterior.csv.gz",
        acceleration_posteriors,
    )
    diagnostic_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0a_target_free_diagnostics.csv.gz",
        diagnostics,
    )
    for label, artifact in (
        ("prediction", prediction_artifact),
        ("acceleration", acceleration_artifact),
        ("diagnostic", diagnostic_artifact),
    ):
        if artifact["logical_sha256"] != artifact["readback_logical_sha256"]:
            raise RuntimeError(f"{label} readback SHA mismatch")

    elapsed = float(time.perf_counter() - started)
    gates = evaluate_stage0a_gates(
        config=config,
        frozen_wells=frozen_wells,
        acceleration_contract=acceleration_contract,
        zero_acceleration_contract=zero_acceleration_contract,
        brute_force_contract=brute_force_contract,
        ledger=ledger,
        stage0a_elapsed_seconds=elapsed,
    )
    input_manifest = {
        "fixed4_identity_selection": selection_input,
        "raw_train_dir": str(raw_dir),
        "scientific_contract_sha256": scientific_contract_sha256,
        "frozen_exp441_fixed32_prediction_decompressed_sha256": get_nested(
            config,
            "data.exp441_saved_control.expected_fixed32_decompressed_sha256",
        ),
        "frozen_exp441_prediction_manifest_sha256": get_nested(
            config,
            "data.exp441_saved_control.expected_fixed32_prediction_manifest_sha256",
        ),
        "leakage": {
            "identity_rows_read": ledger.identity_rows_read,
            "target_free_rows_read": ledger.target_free_rows_read,
            "frozen_wells": len(ledger.frozen_wells),
            "forbidden_reads_before_all_freeze": (
                ledger.forbidden_reads_before_all_freeze
            ),
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0a_input_manifest.json",
        input_manifest,
    )
    transition_manifest = {
        "combined_joint_transition_sha256": combined_well_sha(
            frozen_wells,
            "joint_transition_sha256",
        ),
        "combined_prediction_sha256": combined_well_sha(
            frozen_wells,
            "prediction_sha256",
        ),
        "combined_acceleration_posterior_sha256": combined_well_sha(
            frozen_wells,
            "acceleration_posterior_sha256",
        ),
        "combined_diagnostic_sha256": combined_well_sha(
            frozen_wells,
            "diagnostic_sha256",
        ),
        "per_well": ledger.freeze_records,
    }
    transition_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0a_sha_manifest.json",
        transition_manifest,
    )
    eligible = bool(
        gates["stage0b_eligible_pending_separate_user_approval"]
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0a_all_gates_pass_pending_separate_stage0b_approval"
            if eligible
            else "stage0a_fail_closed"
        ),
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha256,
        "gates": gates,
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
            "acceleration_posterior": acceleration_artifact,
            "target_free_diagnostics": diagnostic_artifact,
            "input_manifest": input_artifact,
            "sha_manifest": transition_artifact,
        },
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
            "stage": "stage0a_fixed4_runtime_contract",
            "cv": None,
            "lb": None,
            "fixed4_is_target_free_preflight": True,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha256,
        "technical_gates": gates["technical"],
        "stage0b_eligible_pending_separate_user_approval": eligible,
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# Import is side-effect free. Direct execution prints the frozen implementation
# contract. It runs Stage 0A only after the later config explicitly records
# canonical-notebook, package, and Stage 0A approvals.

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
                "event": "exp444_implementation_contract",
                "experiment": EXPERIMENT_NAME,
                "status": get_nested(CONFIG, "experiment.status"),
                "selected_stage": get_nested(
                    CONFIG,
                    "execution.selected_stage",
                ),
                "execution_counts": EXECUTION_COUNTS,
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
