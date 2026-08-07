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
# # exp445 TVT-to-U coordinate parity exact HMM
#
# This CPU-only technical audit re-expresses exp209's fixed TVT lattice
# `P_j` as the row-shifted U coordinate `U_t,j = P_j + Z_t`. It makes no
# scientific model change. Parent and candidate emissions, initial priors,
# physical position kernels, forward/backward messages, posteriors, and
# readouts are assembled independently and compared without reading suffix
# truth, fold, role, episode, or error columns.
#
# The canonical notebook, package, and one fixed32 Stage 0 execution were
# authorized on 2026-07-30. Inference and submission remain fail-closed.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Fixed32 manifest and target-free raw inputs
# 4. Parent TVT and candidate row-shifted U input assembly
# 5. Independent physical position kernels and exact HMM
# 6. Synthetic coordinate and brute-force parity
# 7. Fixed32 paired parent/candidate parity freeze
# 8. Technical gates, generated artifacts, and metrics
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
import platform
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from numba import njit, set_num_threads

EXPERIMENT_NAME = "exp445_tvt_to_u_coordinate_parity_exact_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
COMPARISON_REFERENCE = "exp438_u_state_fixed_lattice_exact_hmm"
SCIENTIFIC_VARIANT = "row_shifted_u_coordinate_relabel"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

FORBIDDEN_HORIZONTAL_COLUMNS = frozenset(
    {
        "TVT",
        "tvt_true",
        "error",
        "abs_error",
        "fold",
        "role",
        "hidden_like_role",
        "episode_id",
        "start_row_idx",
        "end_row_idx_exclusive",
    }
)


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


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> dict[str, int]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp445 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp445 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp445 parent changed")
    if (
        get_nested(config, "lineage.comparison_reference")
        != COMPARISON_REFERENCE
    ):
        raise ValueError("exp445 comparison reference changed")
    if not bool(get_nested(config, "design.implementation_authorized", False)):
        raise RuntimeError("exp445 implementation is not authorized")
    if not bool(
        get_nested(
            config,
            "design.canonical_notebook_adoption_authorized",
            False,
        )
    ):
        raise ValueError("canonical notebook adoption is not authorized")
    if not bool(get_nested(config, "design.kaggle_package_authorized", False)):
        raise ValueError("Kaggle packaging is not authorized")
    if not bool(get_nested(config, "design.kaggle_run_authorized", False)):
        raise ValueError("Kaggle Stage 0 is not authorized")
    if bool(get_nested(config, "design.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "design.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if str(get_nested(config, "runtime.accelerator")) != "cpu":
        raise ValueError("exp445 is CPU-only")
    if bool(get_nested(config, "runtime.internet", True)):
        raise ValueError("exp445 internet must remain disabled")

    expected = {
        "coordinate_candidates": 1,
        "manifest_wells": 32,
        "candidate_hmm_well_runs": 32,
        "paired_parent_hmm_well_runs": 32,
        "total_hmm_well_runs": 64,
        "reporting_folds": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1))
        for key in expected
    }
    if observed != expected:
        raise ValueError(
            f"exp445 execution contract changed: {observed} != {expected}"
        )
    if get_nested(config, "execution.selected_stage") != "stage_0_fixed32":
        raise ValueError("selected_stage must remain stage_0_fixed32")
    if bool(get_nested(config, "execution.create_submission", True)):
        raise ValueError("submission creation must remain disabled")
    if require_run_authorization:
        if not bool(get_nested(config, "runtime.run_approved", False)):
            raise RuntimeError(
                "implementation approval does not authorize Kaggle execution"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("execution.run_hmm is false")
    return observed


def validate_scientific_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = get_nested(config, "model.fixed_from_exp209")
    expected_fixed = {
        "position_grid_step_ft": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "effective_position_sigma_ft": 0.1225,
        "position_kernel_cells": 5,
        "momentum": 0.998,
        "emission": "gaussian_typewell_gr",
        "emission_lambda": 1.0,
        "start_sigma_ft": 0.75,
        "initial_rate_sigma": 0.01,
        "band_pad_ft": 100.0,
        "rate_center": "zero",
        "output": "smoothed_posterior_mean_and_std",
    }
    if fixed != expected_fixed:
        raise ValueError(
            f"exp209 HMM contract changed: {fixed} != {expected_fixed}"
        )
    coordinate = get_nested(config, "model.coordinate")
    expected_coordinate = {
        "formula": "U=TVT+Z",
        "parent_tvt_grid_formula": "P_j",
        "candidate_u_grid_formula": "U_t_j=P_j+Z_t",
        "candidate_tvt_view_formula": "U_t_j-Z_t=P_j",
        "candidate_grid_is_fixed_absolute_u": False,
        "candidate_grid_translation_source": "known_row_Z_only",
        "cell_index_identity": "parent_j_equals_candidate_j",
        "continuous_equivalence": "exact",
        "discrete_equivalence": "exact",
        "scientific_difference": "none_coordinate_relabel_only",
    }
    if coordinate != expected_coordinate:
        raise ValueError(
            f"row-shifted U coordinate contract changed: {coordinate}"
        )
    transition = get_nested(config, "model.transition")
    if (
        transition["parent_index_mean_formula"]
        != "r_current*delta_MD-delta_Z"
        or transition["candidate_index_mean_formula"]
        != "r_current*delta_MD-delta_Z"
        or transition["candidate_physical_u_edge_formula"]
        != "(P_k-P_j)+delta_Z"
    ):
        raise ValueError("TVT/U transition identity changed")
    if bool(get_nested(config, "model.coordinate.candidate_grid_is_fixed_absolute_u")):
        raise ValueError("exp438 fixed absolute-U lattice is forbidden")
    return {
        "fixed_from_exp209": fixed,
        "coordinate": coordinate,
        "transition": transition,
        "emission": get_nested(config, "model.emission"),
        "readout": get_nested(config, "model.readout"),
        "forbidden": get_nested(config, "model.forbidden"),
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
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("exp445 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = config_path() if path is None else path
    value = yaml.safe_load(selected.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{selected} must contain a YAML mapping")
    if get_nested(value, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError(f"{selected} is not the exp445 config")
    return value


def artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix == ".gz":
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(stable_json_bytes(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            to_jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return {"path": str(path), "sha256": sha256_file(path)}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    path.write_bytes(text)
    readback = pd.read_csv(path)
    expected_sha = hashlib.sha256(text).hexdigest()
    observed_sha = sha256_file(path)
    return {
        "path": str(path),
        "sha256": observed_sha,
        "logical_sha256": logical_frame_sha256(frame),
        "readback_logical_sha256": logical_frame_sha256(readback),
        "readback_match": (
            expected_sha == observed_sha
            and list(readback.columns) == list(frame.columns)
            and len(readback) == len(frame)
        ),
    }


def write_deterministic_gzip_csv(
    path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(text)
    readback = pd.read_csv(path)
    logical = logical_frame_sha256(frame)
    readback_logical = logical_frame_sha256(readback)
    expected_decompressed = hashlib.sha256(text).hexdigest()
    observed_decompressed = sha256_decompressed_csv(path)
    return {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": observed_decompressed,
        "logical_sha256": logical,
        "readback_logical_sha256": readback_logical,
        "readback_match": (
            expected_decompressed == observed_decompressed
            and list(readback.columns) == list(frame.columns)
            and len(readback) == len(frame)
        ),
        "rows": len(frame),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if platform.system() != "Darwin" else 1024.0**3
    return value / divisor


def runtime_versions() -> dict[str, Any]:
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
    }


def resolve_asset(filename: str, local_path: str) -> Path:
    candidates = (
        Path(local_path),
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "assets" / filename,
        find_project_root() / local_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if KAGGLE_INPUT_ROOT.is_dir():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            hashes = {sha256_file(path) for path in matches}
            if len(hashes) == 1:
                return matches[0]
            raise RuntimeError(f"ambiguous Kaggle assets for {filename}")
    raise FileNotFoundError(f"asset not found: {filename}")


@dataclass
class LeakageLedger:
    scope_rows: int = 0
    target_free_rows: int = 0
    suffix_truth_reads: int = 0
    fold_reads: int = 0
    role_reads: int = 0
    episode_reads: int = 0
    error_reads: int = 0
    frozen_wells: list[str] = field(default_factory=list)

    def record_scope(self, rows: int) -> None:
        self.scope_rows += int(rows)

    def record_target_free(self, rows: int) -> None:
        self.target_free_rows += int(rows)

    def freeze(self, well: str) -> None:
        self.frozen_wells.append(str(well))

    def report(self) -> dict[str, Any]:
        return {
            "scope_rows": self.scope_rows,
            "target_free_rows": self.target_free_rows,
            "suffix_truth_reads": self.suffix_truth_reads,
            "fold_reads": self.fold_reads,
            "role_reads": self.role_reads,
            "episode_reads": self.episode_reads,
            "error_reads": self.error_reads,
            "frozen_wells": list(self.frozen_wells),
            "frozen_well_count": len(self.frozen_wells),
        }


# %% [markdown]
# ## 3. Fixed32 manifest and target-free raw inputs

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


def fixed32_manifest_path(
    config: Mapping[str, Any],
) -> tuple[Path, str]:
    spec = get_nested(config, "data.fixed32_manifest")
    filename = str(spec.get("filename", Path(str(spec["local"])).name))
    path = resolve_asset(filename, str(spec["local"]))
    observed = sha256_file(path)
    expected = str(spec["expected_sha256"])
    if observed != expected:
        raise ValueError(
            f"fixed32 manifest SHA changed: {observed} != {expected}"
        )
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
    frame = frame.sort_values("well", kind="mergesort").reset_index(drop=True)
    ledger.record_scope(len(frame))
    return frame, {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "target_free_logical_sha256": logical_frame_sha256(frame),
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
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    forbidden = FORBIDDEN_HORIZONTAL_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: forbidden columns reached HMM: {forbidden}")
    typewell = pd.read_csv(
        typewell_path,
        usecols=["TVT", "GR"],
    ).sort_values("TVT", kind="mergesort")
    ledger.record_target_free(len(horizontal) + len(typewell))
    return horizontal, typewell.reset_index(drop=True)


# %% [markdown]
# ## 4. Parent TVT and candidate row-shifted U input assembly
#
# Raw observed series are frozen once. From that immutable source, the parent
# TVT path and candidate U path independently construct state values, Type Well
# emission arrays, initial position priors, and index-space transition means.
# The candidate never constructs a fixed absolute-U lattice.

# %%
def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int]:
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = (
        np.isfinite(dtvt)
        & np.isfinite(dz)
        & np.isfinite(dmd)
        & (dmd > 0.0)
    )
    valid_steps = int(valid.sum())
    if valid_steps < int(min_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    if not np.isfinite(rate):
        rate = float(fallback_rate)
    return rate, int(len(tail)), valid_steps


def prefix_sigma_and_rate(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[float, float, int, int]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = (
        known["GR"].fillna(0.0).to_numpy(np.float64) - typewell_at_known
    )
    sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
    rate, effective_rows, valid_steps = robust_initial_rate(known)
    return sigma, rate, effective_rows, valid_steps


def prepare_observed_base(
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
        raise ValueError("suffix truth reached observed input preparation")

    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(eval_rows) == 0:
        raise ValueError("expected a visible prefix and non-empty suffix")
    sigma, init_rate, rate_rows, valid_steps = prefix_sigma_and_rate(
        horizontal,
        typewell_tvt,
        typewell_gr,
    )
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    step = float(hmm["position_grid_step_ft"])
    grid_min = max(
        float(typewell_tvt.min()) - 40.0,
        last_tvt - float(hmm["band_pad_ft"]),
    )
    grid_max = min(
        float(typewell_tvt.max()) + 40.0,
        last_tvt + float(hmm["band_pad_ft"]),
    )
    parent_tvt_grid = np.arange(
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
    dm = np.maximum(
        np.diff(np.concatenate([[float(last["MD"])], md])),
        1.0,
    )
    dz = np.diff(np.concatenate([[last_z], z]))
    span = max(float(hmm["rate_span"]), abs(init_rate) + 0.04)
    rates = np.linspace(
        -span,
        span,
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "parent_tvt_grid": parent_tvt_grid,
        "grid_min": grid_min,
        "step": step,
        "md": md,
        "z": z,
        "gr": gr,
        "dm": dm,
        "dz": dz,
        "rates": rates,
        "last_known_tvt": last_tvt,
        "last_known_z": last_z,
        "last_known_md": float(last["MD"]),
        "start_p": float((last_tvt - grid_min) / step),
        "r0": float(init_rate),
        "prefix_sigma": sigma,
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "prefix_rows": int(len(known)),
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
    }


def gaussian_emission(
    gr: np.ndarray,
    state_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if state_tvt.ndim == 1:
        state_tvt = np.broadcast_to(
            state_tvt[None, :],
            (len(gr), len(state_tvt)),
        )
    emission = np.empty(state_tvt.shape, dtype=np.float64)
    for row in range(len(gr)):
        expected_gr = np.interp(
            state_tvt[row],
            typewell_tvt,
            typewell_gr,
        )
        zscore = (gr[row] - expected_gr) / sigma
        emission[row] = -0.5 * np.minimum(zscore**2, 600.0)
    return emission


def initial_position_log_prior(
    state_values: np.ndarray,
    initial_value: float,
    sigma: float,
) -> np.ndarray:
    values = np.asarray(state_values, dtype=np.float64)
    log_prior = -0.5 * ((values - initial_value) / sigma) ** 2
    log_prior[log_prior < -60.0] = -np.inf
    return log_prior


def exp209_initial_position_log_prior(
    position_count: int,
    start_index: float,
    step: float,
    sigma: float,
) -> np.ndarray:
    indices = np.arange(position_count, dtype=np.float64)
    delta = (indices - float(start_index)) * float(step)
    log_prior = -0.5 * (delta / float(sigma)) ** 2
    log_prior[log_prior < -60.0] = -np.inf
    return log_prior


def assemble_parent_tvt_inputs(
    base: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    tvt_grid = np.array(base["parent_tvt_grid"], dtype=np.float64, copy=True)
    emission_exact = gaussian_emission(
        np.array(base["gr"], dtype=np.float64, copy=True),
        tvt_grid,
        np.array(base["typewell_tvt"], dtype=np.float64, copy=True),
        np.array(base["typewell_gr"], dtype=np.float64, copy=True),
        float(base["prefix_sigma"]),
    )
    transition_mean = (
        np.array(base["dm"], dtype=np.float64, copy=True)[:, None]
        * np.array(base["rates"], dtype=np.float64, copy=True)[None, :]
        - np.array(base["dz"], dtype=np.float64, copy=True)[:, None]
    )
    prior = exp209_initial_position_log_prior(
        len(tvt_grid),
        float(base["start_p"]),
        float(base["step"]),
        float(hmm["start_sigma_ft"]),
    )
    return {
        "state_tvt_grid": tvt_grid,
        "emission_ll_exact": emission_exact,
        "emission_ll": emission_exact.astype(np.float32),
        "initial_position_log_prior": prior,
        "transition_index_mean": transition_mean,
        "dm": np.array(base["dm"], dtype=np.float64, copy=True),
        "dz": np.array(base["dz"], dtype=np.float64, copy=True),
        "rates": np.array(base["rates"], dtype=np.float64, copy=True),
        "r0": float(base["r0"]),
    }


def assemble_candidate_u_inputs(
    base: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    parent_grid_copy = np.array(
        base["parent_tvt_grid"],
        dtype=np.longdouble,
        copy=True,
    )
    z_copy = np.array(base["z"], dtype=np.longdouble, copy=True)
    row_u_grid = (
        parent_grid_copy[None, :]
        + z_copy[:, None]
    )
    row_tvt_view = row_u_grid - z_copy[:, None]
    emission_exact = gaussian_emission(
        np.array(base["gr"], dtype=np.float64, copy=True),
        np.asarray(row_tvt_view, dtype=np.float64),
        np.array(base["typewell_tvt"], dtype=np.float64, copy=True),
        np.array(base["typewell_gr"], dtype=np.float64, copy=True),
        float(base["prefix_sigma"]),
    )
    initial_u_grid = (
        parent_grid_copy + np.longdouble(base["last_known_z"])
    )
    last_known_u = (
        np.longdouble(base["last_known_tvt"])
        + np.longdouble(base["last_known_z"])
    )
    initial_u_origin = (
        np.longdouble(base["grid_min"])
        + np.longdouble(base["last_known_z"])
    )
    candidate_start_p = float(
        (last_known_u - initial_u_origin)
        / np.longdouble(base["step"])
    )
    prior = exp209_initial_position_log_prior(
        len(initial_u_grid),
        candidate_start_p,
        float(base["step"]),
        float(hmm["start_sigma_ft"]),
    )
    dm_copy = np.array(base["dm"], dtype=np.float64, copy=True)
    dz_copy = np.array(base["dz"], dtype=np.float64, copy=True)
    rates_copy = np.array(base["rates"], dtype=np.float64, copy=True)
    physical_u_displacement = dm_copy[:, None] * rates_copy[None, :]
    index_mean_from_moving_grid = physical_u_displacement - dz_copy[:, None]
    return {
        "row_u_grid": row_u_grid,
        "row_tvt_view": row_tvt_view,
        "initial_u_grid": initial_u_grid,
        "last_known_u": last_known_u,
        "candidate_start_p": candidate_start_p,
        "emission_ll_exact": emission_exact,
        "emission_ll": emission_exact.astype(np.float32),
        "initial_position_log_prior": prior,
        "physical_u_displacement": physical_u_displacement,
        "transition_index_mean": index_mean_from_moving_grid,
        "dm": dm_copy,
        "dz": dz_copy,
        "rates": rates_copy,
        "r0": float(base["r0"]),
    }


def prepare_paired_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    base = prepare_observed_base(horizontal, typewell, hmm)
    parent = assemble_parent_tvt_inputs(base, hmm)
    candidate = assemble_candidate_u_inputs(base, hmm)
    return {"base": base, "parent": parent, "candidate": candidate}


def coordinate_contract_from_prepared(
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    base = prepared["base"]
    parent = prepared["parent"]
    candidate = prepared["candidate"]
    parent_grid = np.asarray(parent["state_tvt_grid"], dtype=np.float64)
    row_u = np.asarray(candidate["row_u_grid"], dtype=np.longdouble)
    z = np.asarray(base["z"], dtype=np.longdouble)
    parent_grid_extended = np.asarray(parent_grid, dtype=np.longdouble)
    coordinate_diff = float(
        np.max(
            np.abs(
                row_u
                - z[:, None]
                - parent_grid_extended[None, :]
            )
        )
    )
    transition_diff = float(
        np.max(
            np.abs(
                np.asarray(
                    parent["transition_index_mean"],
                    dtype=np.float64,
                )
                - np.asarray(
                    candidate["transition_index_mean"],
                    dtype=np.float64,
                )
            )
        )
    )
    emission_diff = float(
        np.max(
            np.abs(
                np.asarray(parent["emission_ll_exact"], dtype=np.float64)
                - np.asarray(
                    candidate["emission_ll_exact"],
                    dtype=np.float64,
                )
            )
        )
    )
    parent_log_prior = np.asarray(
        parent["initial_position_log_prior"],
        dtype=np.float64,
    )
    candidate_log_prior = np.asarray(
        candidate["initial_position_log_prior"],
        dtype=np.float64,
    )
    parent_prior = np.exp(parent_log_prior)
    candidate_prior = np.exp(candidate_log_prior)
    parent_prior /= parent_prior.sum()
    candidate_prior /= candidate_prior.sum()
    prior_diff = float(np.max(np.abs(parent_prior - candidate_prior)))
    finite_prior = np.isfinite(parent_log_prior) & np.isfinite(
        candidate_log_prior
    )
    prior_log_diff = float(
        np.max(
            np.abs(
                parent_log_prior[finite_prior]
                - candidate_log_prior[finite_prior]
            )
        )
    )
    return {
        "coordinate_tvt_equals_u_minus_z_max_abs_ft": coordinate_diff,
        "transition_index_mean_max_abs_ft": transition_diff,
        "emission_max_abs": emission_diff,
        "initial_prior_max_abs": prior_diff,
        "initial_log_prior_max_abs": prior_log_diff,
        "rows": len(z),
        "position_states": len(parent_grid),
        "rate_states": len(parent["rates"]),
        "parent_array_sha256": array_bundle_sha256(
            state_tvt_grid=parent_grid,
            emission_ll=np.asarray(parent["emission_ll"], dtype=np.float32),
            initial_position_log_prior=np.asarray(
                parent["initial_position_log_prior"],
                dtype=np.float64,
            ),
            transition_index_mean=np.asarray(
                parent["transition_index_mean"],
                dtype=np.float64,
            ),
        ),
        "candidate_array_sha256": array_bundle_sha256(
            row_u_grid=row_u,
            row_tvt_view=np.asarray(
                candidate["row_tvt_view"],
                dtype=np.float64,
            ),
            emission_ll=np.asarray(
                candidate["emission_ll"],
                dtype=np.float32,
            ),
            initial_position_log_prior=np.asarray(
                candidate["initial_position_log_prior"],
                dtype=np.float64,
            ),
            transition_index_mean=np.asarray(
                candidate["transition_index_mean"],
                dtype=np.float64,
            ),
        ),
    }


# %% [markdown]
# ## 5. Independent physical position kernels and exact HMM
#
# The parent kernel measures `offset*h - (r*dMD-dZ)`. The candidate measures
# the moving-U physical edge directly:
# `offset*h + dZ - r*dMD`. These are computed by different functions. The
# forward/backward kernel selects the appropriate construction and is executed
# once per parent and once per candidate for every fixed32 well. Parent
# position-posterior marginalization follows exp209's exact accumulation order.
# Coordinate expectations explicitly normalize that posterior; exp209's raw
# matrix-product mean/std are retained as a report-only numerical ledger.

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
    rate: float,
    dm: float,
    dz: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_shift = rate * dm - dz
    sigma_position = max(sig_p, 0.35 * step)
    center = int(np.floor(mean_shift / step + 0.5))
    offsets = np.empty(5, np.int64)
    log_weights = np.empty(5, np.float64)
    for kernel_index in range(5):
        offset = center - 2 + kernel_index
        residual_tvt = offset * step - mean_shift
        offsets[kernel_index] = offset
        log_weights[kernel_index] = (
            -0.5 * (residual_tvt / sigma_position) ** 2
        )
    maximum = np.max(log_weights)
    weights = np.exp(log_weights - maximum)
    weights /= np.sum(weights)
    return offsets, weights


@njit(cache=True, nogil=True)
def candidate_u_position_kernel_probabilities(
    rate: float,
    dm: float,
    dz: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    index_mean = rate * dm - dz
    sigma_position = max(sig_p, 0.35 * step)
    center = int(np.floor(index_mean / step + 0.5))
    offsets = np.empty(5, np.int64)
    log_weights = np.empty(5, np.float64)
    for kernel_index in range(5):
        offset = center - 2 + kernel_index
        physical_u_edge = offset * step + dz
        residual_u = physical_u_edge - rate * dm
        offsets[kernel_index] = offset
        log_weights[kernel_index] = (
            -0.5 * (residual_u / sigma_position) ** 2
        )
    maximum = np.max(log_weights)
    weights = np.exp(log_weights - maximum)
    weights /= np.sum(weights)
    return offsets, weights


@njit(cache=True, nogil=True)
def _position_kernel_for_coordinate(
    coordinate_mode: int,
    rate: float,
    dm: float,
    dz: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate_mode == 0:
        return parent_position_kernel_probabilities(
            rate,
            dm,
            dz,
            step,
            sig_p,
        )
    return candidate_u_position_kernel_probabilities(
        rate,
        dm,
        dz,
        step,
        sig_p,
    )


@njit(cache=True, nogil=True)
def _hmm2_fb_coordinate(
    coordinate_mode: int,
    emission_ll: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
    step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    initial_position_log_prior: np.ndarray,
    r0: float,
    r0_sig: float,
    emission_lambda: float,
    momentum: float,
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
        initial_position_logp = initial_position_log_prior[position_index]
        if not np.isfinite(initial_position_logp):
            continue
        for rate_index in range(rate_count):
            delta_rate = (rates[rate_index] - r0) / r0_sig
            previous[position_index, rate_index] = np.float32(
                initial_position_logp - 0.5 * delta_rate * delta_rate
            )

    rate_updated = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count):
        rate_kernel = rate_kernel_probabilities(
            rates,
            dm[time_index],
            sig_r,
            momentum,
        )
        rate_log_kernel = np.log(rate_kernel)
        for position_index in range(position_count):
            for destination_rate in range(rate_count):
                best = negative
                first_source = max(destination_rate - 1, 0)
                last_source = min(destination_rate + 1, rate_count - 1)
                for source_rate in range(first_source, last_source + 1):
                    value = (
                        previous[position_index, source_rate]
                        + rate_log_kernel[
                            source_rate,
                            destination_rate - source_rate + 1,
                        ]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for source_rate in range(first_source, last_source + 1):
                        total += np.exp(
                            previous[position_index, source_rate]
                            + rate_log_kernel[
                                source_rate,
                                destination_rate - source_rate + 1,
                            ]
                            - best
                        )
                    rate_updated[position_index, destination_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    rate_updated[position_index, destination_rate] = negative

        for destination_rate in range(rate_count):
            offsets, position_weights = _position_kernel_for_coordinate(
                coordinate_mode,
                rates[destination_rate],
                dm[time_index],
                dz[time_index],
                step,
                sig_p,
            )
            position_log_weights = np.log(position_weights)
            for destination_position in range(position_count):
                best = negative
                for kernel_index in range(5):
                    source_position = (
                        destination_position - offsets[kernel_index]
                    )
                    if 0 <= source_position < position_count:
                        value = (
                            rate_updated[source_position, destination_rate]
                            + position_log_weights[kernel_index]
                        )
                        if value > best:
                            best = value
                if best > negative / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        source_position = (
                            destination_position - offsets[kernel_index]
                        )
                        if 0 <= source_position < position_count:
                            total += np.exp(
                                rate_updated[source_position, destination_rate]
                                + position_log_weights[kernel_index]
                                - best
                            )
                    current[
                        destination_position,
                        destination_rate,
                    ] = np.float32(
                        best
                        + np.log(total)
                        + emission_lambda
                        * emission_ll[time_index, destination_position]
                    )
                else:
                    current[
                        destination_position,
                        destination_rate,
                    ] = negative
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                alpha[time_index, position_index, rate_index] = current[
                    position_index,
                    rate_index,
                ]
                previous[position_index, rate_index] = current[
                    position_index,
                    rate_index,
                ]

    best = negative
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            best = max(best, alpha[-1, position_index, rate_index])
    total = 0.0
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            total += np.exp(alpha[-1, position_index, rate_index] - best)
    log_likelihood = float(best) + np.log(total)

    posterior_position = np.zeros((time_count, position_count), np.float64)
    posterior_rate = np.zeros((time_count, rate_count), np.float64)
    beta_next = np.zeros((position_count, rate_count), np.float32)
    values = alpha[-1] + beta_next
    best = negative
    for position_index in range(position_count):
        for rate_index in range(rate_count):
            if values[position_index, rate_index] > best:
                best = values[position_index, rate_index]
    total = 0.0
    for position_index in range(position_count):
        position_mass = 0.0
        for rate_index in range(rate_count):
            position_mass += np.exp(
                values[position_index, rate_index] - best
            )
        posterior_position[-1, position_index] = position_mass
        total += position_mass
    for position_index in range(position_count):
        posterior_position[-1, position_index] /= total
    for rate_index in range(rate_count):
        rate_mass = 0.0
        for position_index in range(position_count):
            rate_mass += np.exp(
                values[position_index, rate_index] - best
            )
        posterior_rate[-1, rate_index] = rate_mass / total

    beta_current = np.empty((position_count, rate_count), np.float32)
    beta_position = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count - 1, 0, -1):
        rate_kernel = rate_kernel_probabilities(
            rates,
            dm[time_index],
            sig_r,
            momentum,
        )
        rate_log_kernel = np.log(rate_kernel)
        for destination_rate in range(rate_count):
            offsets, position_weights = _position_kernel_for_coordinate(
                coordinate_mode,
                rates[destination_rate],
                dm[time_index],
                dz[time_index],
                step,
                sig_p,
            )
            position_log_weights = np.log(position_weights)
            for source_position in range(position_count):
                best = negative
                for kernel_index in range(5):
                    destination_position = (
                        source_position + offsets[kernel_index]
                    )
                    if 0 <= destination_position < position_count:
                        value = (
                            position_log_weights[kernel_index]
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
                        if value > best:
                            best = value
                if best > negative / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        destination_position = (
                            source_position + offsets[kernel_index]
                        )
                        if 0 <= destination_position < position_count:
                            total += np.exp(
                                position_log_weights[kernel_index]
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
                    beta_position[
                        source_position,
                        destination_rate,
                    ] = np.float32(best + np.log(total))
                else:
                    beta_position[
                        source_position,
                        destination_rate,
                    ] = negative

        for source_position in range(position_count):
            for source_rate in range(rate_count):
                best = negative
                first_destination = max(source_rate - 1, 0)
                last_destination = min(source_rate + 1, rate_count - 1)
                for destination_rate in range(
                    first_destination,
                    last_destination + 1,
                ):
                    value = (
                        rate_log_kernel[
                            source_rate,
                            destination_rate - source_rate + 1,
                        ]
                        + beta_position[source_position, destination_rate]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for destination_rate in range(
                        first_destination,
                        last_destination + 1,
                    ):
                        total += np.exp(
                            rate_log_kernel[
                                source_rate,
                                destination_rate - source_rate + 1,
                            ]
                            + beta_position[
                                source_position,
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
        best = negative
        for rate_index in range(rate_count):
            for position_index in range(position_count):
                if values[position_index, rate_index] > best:
                    best = values[position_index, rate_index]
        total = 0.0
        for position_index in range(position_count):
            position_mass = 0.0
            for rate_index in range(rate_count):
                position_mass += np.exp(
                    values[position_index, rate_index] - best
                )
            posterior_position[
                time_index - 1,
                position_index,
            ] = position_mass
            total += position_mass
        for position_index in range(position_count):
            posterior_position[
                time_index - 1,
                position_index,
            ] /= total
        for rate_index in range(rate_count):
            rate_mass = 0.0
            for position_index in range(position_count):
                rate_mass += np.exp(
                    values[position_index, rate_index] - best
                )
            posterior_rate[time_index - 1, rate_index] = rate_mass / total
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                beta_next[position_index, rate_index] = beta_current[
                    position_index,
                    rate_index,
                ]

    maximum_normalization_error = 0.0
    for time_index in range(time_count):
        position_total = np.sum(posterior_position[time_index])
        rate_total = np.sum(posterior_rate[time_index])
        for rate_index in range(rate_count):
            posterior_rate[time_index, rate_index] /= rate_total
        rate_total = np.sum(posterior_rate[time_index])
        maximum_normalization_error = max(
            maximum_normalization_error,
            abs(position_total - 1.0),
            abs(rate_total - 1.0),
        )
    return (
        posterior_position,
        posterior_rate,
        log_likelihood,
        maximum_normalization_error,
    )


def transition_kernel_contract(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    base = prepared["base"]
    rates = np.asarray(base["rates"], dtype=np.float64)
    dm = np.asarray(base["dm"], dtype=np.float64)
    dz = np.asarray(base["dz"], dtype=np.float64)
    step = float(hmm["position_grid_step_ft"])
    sig_p = float(hmm["sig_p"])
    parent_offsets = np.empty((len(dm), len(rates), 5), dtype=np.int64)
    candidate_offsets = np.empty_like(parent_offsets)
    parent_weights = np.empty((len(dm), len(rates), 5), dtype=np.float64)
    candidate_weights = np.empty_like(parent_weights)
    parent_rate_weights = np.empty(
        (len(dm), len(rates), 3),
        dtype=np.float64,
    )
    candidate_rate_weights = np.empty_like(parent_rate_weights)
    maximum_rate_row_sum_error = 0.0
    for row in range(len(dm)):
        parent_rate_kernel = rate_kernel_probabilities(
            rates,
            float(dm[row]),
            float(hmm["sig_r"]),
            float(hmm["momentum"]),
        )
        candidate_rate_kernel = rate_kernel_probabilities(
            np.array(rates, dtype=np.float64, copy=True),
            float(dm[row]),
            float(hmm["sig_r"]),
            float(hmm["momentum"]),
        )
        parent_rate_weights[row] = parent_rate_kernel
        candidate_rate_weights[row] = candidate_rate_kernel
        maximum_rate_row_sum_error = max(
            maximum_rate_row_sum_error,
            float(
                np.max(
                    np.abs(parent_rate_kernel.sum(axis=1) - 1.0)
                )
            ),
            float(
                np.max(
                    np.abs(candidate_rate_kernel.sum(axis=1) - 1.0)
                )
            ),
        )
        for rate_index, rate in enumerate(rates):
            po, pw = parent_position_kernel_probabilities(
                float(rate),
                float(dm[row]),
                float(dz[row]),
                step,
                sig_p,
            )
            co, cw = candidate_u_position_kernel_probabilities(
                float(rate),
                float(dm[row]),
                float(dz[row]),
                step,
                sig_p,
            )
            parent_offsets[row, rate_index] = po
            candidate_offsets[row, rate_index] = co
            parent_weights[row, rate_index] = pw
            candidate_weights[row, rate_index] = cw
    offset_identity = bool(np.array_equal(parent_offsets, candidate_offsets))
    weight_diff = float(np.max(np.abs(parent_weights - candidate_weights)))
    physical_residual_diff = 0.0
    for row in range(len(dm)):
        for rate_index, rate in enumerate(rates):
            parent_residual = (
                parent_offsets[row, rate_index].astype(np.float64) * step
                - (rate * dm[row] - dz[row])
            )
            candidate_residual = (
                candidate_offsets[row, rate_index].astype(np.float64) * step
                + dz[row]
                - rate * dm[row]
            )
            physical_residual_diff = max(
                physical_residual_diff,
                float(np.max(np.abs(parent_residual - candidate_residual))),
            )
    return {
        "offset_identity": offset_identity,
        "rate_kernel_max_abs": float(
            np.max(
                np.abs(parent_rate_weights - candidate_rate_weights)
            )
        ),
        "position_kernel_max_abs": weight_diff,
        "physical_edge_residual_identity_max_abs_ft": physical_residual_diff,
        "rate_kernel_row_sum_max_error": maximum_rate_row_sum_error,
        "position_kernel_row_sum_max_error": max(
            float(np.max(np.abs(parent_weights.sum(axis=2) - 1.0))),
            float(np.max(np.abs(candidate_weights.sum(axis=2) - 1.0))),
        ),
        "parent_transition_sha256": array_bundle_sha256(
            rate_weights=parent_rate_weights,
            offsets=parent_offsets,
            weights=parent_weights,
        ),
        "candidate_transition_sha256": array_bundle_sha256(
            rate_weights=candidate_rate_weights,
            offsets=candidate_offsets,
            weights=candidate_weights,
        ),
    }


def run_paired_hmms(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    parent = prepared["parent"]
    candidate = prepared["candidate"]
    base = prepared["base"]
    common = {
        "step": float(hmm["position_grid_step_ft"]),
        "sig_r": float(hmm["sig_r"]),
        "sig_p": float(hmm["sig_p"]),
        "r0_sig": float(hmm["initial_rate_sigma"]),
        "emission_lambda": float(hmm["emission_lambda"]),
        "momentum": float(hmm["momentum"]),
    }
    parent_started = time.perf_counter()
    (
        parent_position,
        parent_rate,
        parent_loglik,
        parent_norm,
    ) = _hmm2_fb_coordinate(
        0,
        np.asarray(parent["emission_ll"], dtype=np.float32),
        np.asarray(parent["dm"], dtype=np.float64),
        np.asarray(parent["dz"], dtype=np.float64),
        common["step"],
        np.asarray(parent["rates"], dtype=np.float64),
        common["sig_r"],
        common["sig_p"],
        np.asarray(parent["initial_position_log_prior"], dtype=np.float64),
        float(parent["r0"]),
        common["r0_sig"],
        common["emission_lambda"],
        common["momentum"],
    )
    parent_seconds = time.perf_counter() - parent_started

    candidate_started = time.perf_counter()
    (
        candidate_position,
        candidate_rate,
        candidate_loglik,
        candidate_norm,
    ) = _hmm2_fb_coordinate(
        1,
        np.asarray(candidate["emission_ll"], dtype=np.float32),
        np.asarray(candidate["dm"], dtype=np.float64),
        np.asarray(candidate["dz"], dtype=np.float64),
        common["step"],
        np.asarray(candidate["rates"], dtype=np.float64),
        common["sig_r"],
        common["sig_p"],
        np.asarray(
            candidate["initial_position_log_prior"],
            dtype=np.float64,
        ),
        float(candidate["r0"]),
        common["r0_sig"],
        common["emission_lambda"],
        common["momentum"],
    )
    candidate_seconds = time.perf_counter() - candidate_started

    parent_grid = np.asarray(parent["state_tvt_grid"], dtype=np.float64)
    row_u_grid = np.asarray(candidate["row_u_grid"], dtype=np.longdouble)
    z_extended = np.asarray(base["z"], dtype=np.longdouble)
    z = np.asarray(base["z"], dtype=np.float64)
    rates = np.asarray(base["rates"], dtype=np.float64)
    parent_position_readout = (
        parent_position / parent_position.sum(axis=1, keepdims=True)
    )
    candidate_position_readout = (
        candidate_position / candidate_position.sum(axis=1, keepdims=True)
    )
    parent_origin = float(parent_grid[0])
    parent_offsets = parent_grid - parent_origin
    parent_exp209_raw_mean_tvt = parent_position @ parent_grid
    parent_exp209_raw_variance = (
        parent_position @ (parent_grid**2)
        - parent_exp209_raw_mean_tvt**2
    )
    parent_exp209_raw_std_tvt = np.sqrt(
        np.maximum(parent_exp209_raw_variance, 0.0)
    )
    parent_mean_tvt = (
        parent_origin + parent_position_readout @ parent_offsets
    )
    parent_variance = np.sum(
        parent_position_readout
        * (parent_grid[None, :] - parent_mean_tvt[:, None]) ** 2,
        axis=1,
    )
    parent_std_tvt = np.sqrt(np.maximum(parent_variance, 0.0))
    candidate_position_extended = candidate_position_readout.astype(
        np.longdouble,
        copy=False,
    )
    candidate_u_origin = row_u_grid[:, 0]
    candidate_mean_u_extended = np.sum(
        candidate_position_extended
        * (row_u_grid - candidate_u_origin[:, None]),
        axis=1,
    ) + candidate_u_origin
    candidate_mean_tvt_extended = candidate_mean_u_extended - z_extended
    candidate_variance = np.sum(
        candidate_position_extended
        * (row_u_grid - candidate_mean_u_extended[:, None]) ** 2,
        axis=1,
    )
    candidate_mean_u = np.asarray(
        candidate_mean_u_extended,
        dtype=np.float64,
    )
    candidate_mean_tvt = np.asarray(
        candidate_mean_tvt_extended,
        dtype=np.float64,
    )
    candidate_std_tvt = np.asarray(
        np.sqrt(np.maximum(candidate_variance, np.longdouble(0.0))),
        dtype=np.float64,
    )
    parent_rate_mean = parent_rate @ rates
    candidate_rate_mean = candidate_rate @ rates
    transition = transition_kernel_contract(prepared, hmm)
    finite_values = np.concatenate(
        [
            parent_mean_tvt,
            parent_std_tvt,
            parent_exp209_raw_mean_tvt,
            parent_exp209_raw_std_tvt,
            candidate_mean_u,
            candidate_mean_tvt,
            candidate_std_tvt,
            parent_rate_mean,
            candidate_rate_mean,
        ]
    )
    return {
        "parent_posterior_position": parent_position,
        "candidate_posterior_position": candidate_position,
        "parent_posterior_rate": parent_rate,
        "candidate_posterior_rate": candidate_rate,
        "parent_mean_tvt": parent_mean_tvt,
        "parent_std_tvt": parent_std_tvt,
        "parent_exp209_raw_mean_tvt": parent_exp209_raw_mean_tvt,
        "parent_exp209_raw_std_tvt": parent_exp209_raw_std_tvt,
        "candidate_mean_u": candidate_mean_u,
        "candidate_mean_tvt": candidate_mean_tvt,
        "candidate_std_tvt": candidate_std_tvt,
        "parent_rate_mean": parent_rate_mean,
        "candidate_rate_mean": candidate_rate_mean,
        "parent_log_likelihood": float(parent_loglik),
        "candidate_log_likelihood": float(candidate_loglik),
        "parent_normalization_max_error": float(parent_norm),
        "candidate_normalization_max_error": float(candidate_norm),
        "position_posterior_max_abs": float(
            np.max(np.abs(parent_position - candidate_position))
        ),
        "rate_posterior_max_abs": float(
            np.max(np.abs(parent_rate - candidate_rate))
        ),
        "log_likelihood_abs": abs(
            float(parent_loglik) - float(candidate_loglik)
        ),
        "tvt_mean_max_abs_ft": float(
            np.max(np.abs(parent_mean_tvt - candidate_mean_tvt))
        ),
        "tvt_std_max_abs_ft": float(
            np.max(np.abs(parent_std_tvt - candidate_std_tvt))
        ),
        "parent_exp209_raw_vs_normalized_mean_max_abs_ft": float(
            np.max(
                np.abs(
                    parent_exp209_raw_mean_tvt - parent_mean_tvt
                )
            )
        ),
        "parent_exp209_raw_vs_normalized_std_max_abs_ft": float(
            np.max(
                np.abs(
                    parent_exp209_raw_std_tvt - parent_std_tvt
                )
            )
        ),
        "candidate_u_minus_z_readout_max_abs_ft": float(
            np.max(
                np.abs(
                    candidate_mean_u_extended
                    - z_extended
                    - (
                        np.longdouble(parent_origin)
                        + candidate_position_extended
                        @ parent_offsets.astype(np.longdouble)
                    )
                )
            )
        ),
        "finite_coverage": float(np.mean(np.isfinite(finite_values))),
        "parent_seconds": float(parent_seconds),
        "candidate_seconds": float(candidate_seconds),
        "transition_contract": transition,
        "parent_posterior_sha256": array_bundle_sha256(
            position=parent_position,
            rate=parent_rate,
        ),
        "candidate_posterior_sha256": array_bundle_sha256(
            position=candidate_position,
            rate=candidate_rate,
        ),
        "parent_prediction_sha256": array_bundle_sha256(
            row_idx=np.asarray(base["eval_index"], dtype=np.int64),
            exp209_raw_mean_tvt=np.asarray(
                parent_exp209_raw_mean_tvt,
                dtype=np.float32,
            ),
            exp209_raw_std_tvt=np.asarray(
                parent_exp209_raw_std_tvt,
                dtype=np.float32,
            ),
            normalized_mean_tvt=np.asarray(
                parent_mean_tvt,
                dtype=np.float32,
            ),
            normalized_std_tvt=np.asarray(
                parent_std_tvt,
                dtype=np.float32,
            ),
        ),
        "candidate_prediction_sha256": array_bundle_sha256(
            row_idx=np.asarray(base["eval_index"], dtype=np.int64),
            mean_u=np.asarray(candidate_mean_u, dtype=np.float32),
            mean_tvt=np.asarray(candidate_mean_tvt, dtype=np.float32),
            std_tvt=np.asarray(candidate_std_tvt, dtype=np.float32),
        ),
    }


# %% [markdown]
# ## 6. Synthetic coordinate and brute-force parity

# %%
def synthetic_inputs(
    *,
    variable_z: bool,
    rows: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows = 8
    total = prefix_rows + rows
    md = np.arange(total, dtype=np.float64) * 10.0
    if variable_z:
        z = (
            8_000.0
            + 0.31 * np.sin(np.arange(total, dtype=np.float64) / 2.5)
            + 0.07 * np.arange(total, dtype=np.float64)
        )
    else:
        z = np.full(total, 8_000.0, dtype=np.float64)
    visible_tvt = 12_000.0 + 0.02 * md - (z - z[0])
    tvt_input = visible_tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": 65.0 + 4.0 * np.sin(np.arange(total) / 3.0),
            "TVT_input": tvt_input,
        }
    )
    typewell_tvt = np.linspace(11_850.0, 12_150.0, 401)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 65.0
            + 8.0 * np.sin((typewell_tvt - 12_000.0) / 18.0),
        }
    )
    return horizontal, typewell


def _dense_transition_matrix(
    coordinate_mode: int,
    *,
    dm: float,
    dz: float,
    step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    momentum: float,
    position_count: int,
) -> np.ndarray:
    rate_count = len(rates)
    state_count = position_count * rate_count
    matrix = np.zeros((state_count, state_count), dtype=np.float64)
    rate_kernel = rate_kernel_probabilities(rates, dm, sig_r, momentum)
    for source_position in range(position_count):
        for source_rate in range(rate_count):
            source_state = source_position * rate_count + source_rate
            for destination_rate in range(
                max(0, source_rate - 1),
                min(rate_count, source_rate + 2),
            ):
                offsets, weights = _position_kernel_for_coordinate(
                    coordinate_mode,
                    float(rates[destination_rate]),
                    dm,
                    dz,
                    step,
                    sig_p,
                )
                rate_probability = rate_kernel[
                    source_rate,
                    destination_rate - source_rate + 1,
                ]
                for offset, weight in zip(offsets, weights, strict=True):
                    destination_position = source_position + int(offset)
                    if 0 <= destination_position < position_count:
                        destination_state = (
                            destination_position * rate_count
                            + destination_rate
                        )
                        matrix[source_state, destination_state] += (
                            rate_probability * float(weight)
                        )
    return matrix


def exhaustive_small_path_reference(
    coordinate_mode: int,
    emission_ll: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
    step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    initial_position_log_prior: np.ndarray,
    r0: float,
    r0_sig: float,
    emission_lambda: float,
    momentum: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    emission_ll = np.asarray(emission_ll, dtype=np.float64)
    time_count, position_count = emission_ll.shape
    rate_count = len(rates)
    state_count = position_count * rate_count
    if time_count > 3 or state_count > 9:
        raise ValueError("exhaustive reference is limited to tiny HMMs")
    initial = np.zeros(state_count, dtype=np.float64)
    for position_index in range(position_count):
        position_logp = initial_position_log_prior[position_index]
        if not np.isfinite(position_logp):
            continue
        for rate_index, rate in enumerate(rates):
            delta_rate = (rate - r0) / r0_sig
            initial[position_index * rate_count + rate_index] = np.exp(
                position_logp - 0.5 * delta_rate * delta_rate
            )
    transitions = [
        _dense_transition_matrix(
            coordinate_mode,
            dm=float(dm[row]),
            dz=float(dz[row]),
            step=step,
            rates=rates,
            sig_r=sig_r,
            sig_p=sig_p,
            momentum=momentum,
            position_count=position_count,
        )
        for row in range(time_count)
    ]
    emission_probability = np.empty((time_count, state_count), np.float64)
    for row in range(time_count):
        for position_index in range(position_count):
            value = np.exp(emission_lambda * emission_ll[row, position_index])
            start = position_index * rate_count
            emission_probability[row, start : start + rate_count] = value

    path_states: list[tuple[int, ...]] = []
    path_weights: list[float] = []

    def extend_path(
        time_index: int,
        previous_state: int,
        path: tuple[int, ...],
        weight: float,
    ) -> None:
        if time_index == time_count:
            path_states.append(path)
            path_weights.append(weight)
            return
        transition = transitions[time_index]
        for destination_state in range(state_count):
            next_weight = (
                weight
                * transition[previous_state, destination_state]
                * emission_probability[time_index, destination_state]
            )
            if next_weight > 0.0:
                extend_path(
                    time_index + 1,
                    destination_state,
                    (*path, destination_state),
                    next_weight,
                )

    for initial_state in range(state_count):
        if initial[initial_state] > 0.0:
            extend_path(0, initial_state, (), float(initial[initial_state]))
    weights = np.asarray(path_weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("exhaustive reference has zero or non-finite mass")
    normalized = weights / total
    posterior_position = np.zeros((time_count, position_count), np.float64)
    posterior_rate = np.zeros((time_count, rate_count), np.float64)
    for probability, path in zip(normalized, path_states, strict=True):
        for time_index, state in enumerate(path):
            position_index = state // rate_count
            rate_index = state % rate_count
            posterior_position[time_index, position_index] += probability
            posterior_rate[time_index, rate_index] += probability
    return posterior_position, posterior_rate, float(np.log(total))


def brute_force_small_reference_contract(
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    emission_parent = np.asarray(
        [
            [-0.20, -0.01, -0.40],
            [-0.35, -0.05, -0.10],
            [-0.60, -0.15, -0.02],
        ],
        dtype=np.float32,
    )
    emission_candidate = emission_parent.copy()
    dm = np.asarray([1.0, 1.25, 0.75], dtype=np.float64)
    dz = np.asarray([0.08, -0.04, 0.11], dtype=np.float64)
    rates = np.asarray([-0.02, 0.0, 0.02], dtype=np.float64)
    prior_parent = initial_position_log_prior(
        np.asarray([0.0, 0.35, 0.70]),
        0.385,
        float(hmm["start_sigma_ft"]),
    )
    prior_candidate = initial_position_log_prior(
        np.asarray([8_000.0, 8_000.35, 8_000.70]),
        8_000.385,
        float(hmm["start_sigma_ft"]),
    )
    common = {
        "step": float(hmm["position_grid_step_ft"]),
        "sig_r": float(hmm["sig_r"]),
        "sig_p": float(hmm["sig_p"]),
        "r0": 0.0,
        "r0_sig": float(hmm["initial_rate_sigma"]),
        "emission_lambda": float(hmm["emission_lambda"]),
        "momentum": float(hmm["momentum"]),
    }
    observed: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    references: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name, mode, emission, prior in (
        ("parent", 0, emission_parent, prior_parent),
        ("candidate", 1, emission_candidate, prior_candidate),
    ):
        observed[name] = _hmm2_fb_coordinate(
            mode,
            emission,
            dm,
            dz,
            common["step"],
            rates,
            common["sig_r"],
            common["sig_p"],
            prior,
            common["r0"],
            common["r0_sig"],
            common["emission_lambda"],
            common["momentum"],
        )
        references[name] = exhaustive_small_path_reference(
            mode,
            emission,
            dm,
            dz,
            common["step"],
            rates,
            common["sig_r"],
            common["sig_p"],
            prior,
            common["r0"],
            common["r0_sig"],
            common["emission_lambda"],
            common["momentum"],
        )
    diffs: dict[str, float] = {}
    for name in ("parent", "candidate"):
        diffs[f"{name}_position_posterior_max_abs"] = float(
            np.max(np.abs(observed[name][0] - references[name][0]))
        )
        diffs[f"{name}_rate_posterior_max_abs"] = float(
            np.max(np.abs(observed[name][1] - references[name][1]))
        )
        diffs[f"{name}_log_likelihood_abs"] = abs(
            float(observed[name][2]) - float(references[name][2])
        )
    diffs["paired_position_posterior_max_abs"] = float(
        np.max(np.abs(observed["parent"][0] - observed["candidate"][0]))
    )
    diffs["paired_rate_posterior_max_abs"] = float(
        np.max(np.abs(observed["parent"][1] - observed["candidate"][1]))
    )
    diffs["paired_log_likelihood_abs"] = abs(
        float(observed["parent"][2]) - float(observed["candidate"][2])
    )
    maximum = max(diffs.values())
    return {
        **diffs,
        "maximum_abs": maximum,
        "parent_normalization_max_error": float(observed["parent"][3]),
        "candidate_normalization_max_error": float(observed["candidate"][3]),
        "pass": bool(maximum <= 1.0e-6),
    }


def synthetic_coordinate_contract(
    hmm: Mapping[str, Any],
    *,
    variable_z: bool,
) -> dict[str, Any]:
    horizontal, typewell = synthetic_inputs(variable_z=variable_z)
    prepared = prepare_paired_inputs(horizontal, typewell, hmm)
    coordinate = coordinate_contract_from_prepared(prepared)
    transition = transition_kernel_contract(prepared, hmm)
    paired = run_paired_hmms(prepared, hmm)
    return {
        "coordinate": coordinate,
        "transition": transition,
        "paired": {
            key: paired[key]
            for key in (
                "position_posterior_max_abs",
                "rate_posterior_max_abs",
                "log_likelihood_abs",
                "tvt_mean_max_abs_ft",
                "tvt_std_max_abs_ft",
                "candidate_u_minus_z_readout_max_abs_ft",
                "finite_coverage",
            )
        },
    }


def run_synthetic_contracts(
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "variable_z": synthetic_coordinate_contract(
            hmm,
            variable_z=True,
        ),
        "constant_z": synthetic_coordinate_contract(
            hmm,
            variable_z=False,
        ),
        "brute_force": brute_force_small_reference_contract(hmm),
    }


# %% [markdown]
# ## 7. Fixed32 paired parent/candidate parity freeze

# %%
@dataclass
class FrozenWellParity:
    well: str
    prediction_rows: pd.DataFrame
    parity_row: dict[str, Any]
    transition_row: dict[str, Any]


def cache_ids_for_rows(
    well: str,
    row_indices: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    if not well or rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("invalid well or row indices")
    return np.asarray([f"{well}_{int(row)}" for row in rows], dtype=str)


def freeze_target_free_well(
    *,
    well: str,
    expected_prefix_rows: int,
    expected_suffix_rows: int,
    raw_dir: Path,
    hmm: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWellParity:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_paired_inputs(horizontal, typewell, hmm)
    base = prepared["base"]
    if int(base["prefix_rows"]) != int(expected_prefix_rows):
        raise ValueError(f"{well}: prefix row count changed")
    if len(base["eval_index"]) != int(expected_suffix_rows):
        raise ValueError(f"{well}: suffix row count changed")
    coordinate = coordinate_contract_from_prepared(prepared)
    paired = run_paired_hmms(prepared, hmm)
    row_idx = np.asarray(base["eval_index"], dtype=np.int64)
    prediction = pd.DataFrame(
        {
            "id": cache_ids_for_rows(well, row_idx),
            "well": well,
            "row_idx": row_idx,
            "parent_tvt_mean": paired["parent_mean_tvt"],
            "parent_tvt_std": paired["parent_std_tvt"],
            "parent_exp209_raw_tvt_mean": paired[
                "parent_exp209_raw_mean_tvt"
            ],
            "parent_exp209_raw_tvt_std": paired[
                "parent_exp209_raw_std_tvt"
            ],
            "candidate_u_mean": paired["candidate_mean_u"],
            "candidate_tvt_mean": paired["candidate_mean_tvt"],
            "candidate_tvt_std": paired["candidate_std_tvt"],
            "parent_rate_mean": paired["parent_rate_mean"],
            "candidate_rate_mean": paired["candidate_rate_mean"],
        }
    )
    transition = paired["transition_contract"]
    parity_row = {
        "well": well,
        "rows": len(prediction),
        "position_states": coordinate["position_states"],
        "rate_states": coordinate["rate_states"],
        "coordinate_tvt_equals_u_minus_z_max_abs_ft": coordinate[
            "coordinate_tvt_equals_u_minus_z_max_abs_ft"
        ],
        "emission_max_abs": coordinate["emission_max_abs"],
        "initial_prior_max_abs": coordinate["initial_prior_max_abs"],
        "transition_index_mean_max_abs_ft": coordinate[
            "transition_index_mean_max_abs_ft"
        ],
        "physical_edge_residual_identity_max_abs_ft": transition[
            "physical_edge_residual_identity_max_abs_ft"
        ],
        "position_kernel_max_abs": transition[
            "position_kernel_max_abs"
        ],
        "position_posterior_max_abs": paired[
            "position_posterior_max_abs"
        ],
        "rate_posterior_max_abs": paired["rate_posterior_max_abs"],
        "log_likelihood_abs": paired["log_likelihood_abs"],
        "tvt_mean_max_abs_ft": paired["tvt_mean_max_abs_ft"],
        "tvt_std_max_abs_ft": paired["tvt_std_max_abs_ft"],
        "parent_exp209_raw_vs_normalized_mean_max_abs_ft": paired[
            "parent_exp209_raw_vs_normalized_mean_max_abs_ft"
        ],
        "parent_exp209_raw_vs_normalized_std_max_abs_ft": paired[
            "parent_exp209_raw_vs_normalized_std_max_abs_ft"
        ],
        "candidate_u_minus_z_readout_max_abs_ft": paired[
            "candidate_u_minus_z_readout_max_abs_ft"
        ],
        "finite_coverage": paired["finite_coverage"],
        "parent_normalization_max_error": paired[
            "parent_normalization_max_error"
        ],
        "candidate_normalization_max_error": paired[
            "candidate_normalization_max_error"
        ],
        "parent_log_likelihood": paired["parent_log_likelihood"],
        "candidate_log_likelihood": paired["candidate_log_likelihood"],
        "parent_hmm_seconds": paired["parent_seconds"],
        "candidate_hmm_seconds": paired["candidate_seconds"],
        "parent_coordinate_array_sha256": coordinate[
            "parent_array_sha256"
        ],
        "candidate_coordinate_array_sha256": coordinate[
            "candidate_array_sha256"
        ],
        "parent_posterior_sha256": paired["parent_posterior_sha256"],
        "candidate_posterior_sha256": paired[
            "candidate_posterior_sha256"
        ],
        "parent_prediction_sha256": paired["parent_prediction_sha256"],
        "candidate_prediction_sha256": paired[
            "candidate_prediction_sha256"
        ],
    }
    transition_row = {
        "well": well,
        "rows": len(prediction),
        "offset_identity": transition["offset_identity"],
        "rate_kernel_max_abs": transition["rate_kernel_max_abs"],
        "rate_kernel_row_sum_max_error": transition[
            "rate_kernel_row_sum_max_error"
        ],
        "position_kernel_row_sum_max_error": transition[
            "position_kernel_row_sum_max_error"
        ],
        "position_kernel_max_abs": transition[
            "position_kernel_max_abs"
        ],
        "physical_edge_residual_identity_max_abs_ft": transition[
            "physical_edge_residual_identity_max_abs_ft"
        ],
        "parent_transition_sha256": transition[
            "parent_transition_sha256"
        ],
        "candidate_transition_sha256": transition[
            "candidate_transition_sha256"
        ],
        "parent_emission_sha256": array_bundle_sha256(
            emission=np.asarray(
                prepared["parent"]["emission_ll"],
                dtype=np.float32,
            )
        ),
        "candidate_emission_sha256": array_bundle_sha256(
            emission=np.asarray(
                prepared["candidate"]["emission_ll"],
                dtype=np.float32,
            )
        ),
    }
    ledger.freeze(well)
    return FrozenWellParity(
        well=well,
        prediction_rows=prediction,
        parity_row=parity_row,
        transition_row=transition_row,
    )


def predictions_frame(
    frozen_wells: Sequence[FrozenWellParity],
) -> pd.DataFrame:
    return pd.concat(
        [item.prediction_rows for item in frozen_wells],
        ignore_index=True,
    ).sort_values(["well", "row_idx"], kind="mergesort")


def parity_ledger_frame(
    frozen_wells: Sequence[FrozenWellParity],
) -> pd.DataFrame:
    return pd.DataFrame(
        [item.parity_row for item in frozen_wells]
    ).sort_values("well", kind="mergesort")


def transition_emission_ledger_frame(
    frozen_wells: Sequence[FrozenWellParity],
) -> pd.DataFrame:
    return pd.DataFrame(
        [item.transition_row for item in frozen_wells]
    ).sort_values("well", kind="mergesort")


# %% [markdown]
# ## 8. Technical gates, generated artifacts, and metrics

# %%
def _maximum_nested(
    payloads: Sequence[Mapping[str, Any]],
    *keys: str,
) -> float:
    values: list[float] = []
    for payload in payloads:
        current: Any = payload
        for key in keys:
            current = current[key]
        values.append(float(current))
    return max(values)


def evaluate_technical_gates(
    *,
    config: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    parity_ledger: pd.DataFrame,
    transition_ledger: pd.DataFrame,
    leakage: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gates = get_nested(config, "validation.gates")
    coordinate_max = max(
        float(parity_ledger["coordinate_tvt_equals_u_minus_z_max_abs_ft"].max()),
        _maximum_nested(
            [synthetic["variable_z"], synthetic["constant_z"]],
            "coordinate",
            "coordinate_tvt_equals_u_minus_z_max_abs_ft",
        ),
    )
    physical_edge_max = max(
        float(
            parity_ledger[
                "physical_edge_residual_identity_max_abs_ft"
            ].max()
        ),
        _maximum_nested(
            [synthetic["variable_z"], synthetic["constant_z"]],
            "transition",
            "physical_edge_residual_identity_max_abs_ft",
        ),
    )
    emission_max = max(
        float(parity_ledger["emission_max_abs"].max()),
        _maximum_nested(
            [synthetic["variable_z"], synthetic["constant_z"]],
            "coordinate",
            "emission_max_abs",
        ),
    )
    prior_max = max(
        float(parity_ledger["initial_prior_max_abs"].max()),
        _maximum_nested(
            [synthetic["variable_z"], synthetic["constant_z"]],
            "coordinate",
            "initial_prior_max_abs",
        ),
    )
    position_kernel_max = max(
        float(transition_ledger["position_kernel_max_abs"].max()),
        _maximum_nested(
            [synthetic["variable_z"], synthetic["constant_z"]],
            "transition",
            "position_kernel_max_abs",
        ),
    )
    rate_kernel_max = max(
        float(transition_ledger["rate_kernel_max_abs"].max()),
        _maximum_nested(
            [synthetic["variable_z"], synthetic["constant_z"]],
            "transition",
            "rate_kernel_max_abs",
        ),
    )
    brute_force_max = float(synthetic["brute_force"]["maximum_abs"])
    finite_coverage = float(parity_ledger["finite_coverage"].min())
    readback_match = all(
        bool(value.get("readback_match", False))
        for value in artifacts.values()
        if "readback_match" in value
    )
    forbidden_reads = sum(
        int(leakage[key])
        for key in (
            "suffix_truth_reads",
            "fold_reads",
            "role_reads",
            "episode_reads",
            "error_reads",
        )
    )
    observed = {
        "coordinate_tvt_equals_u_minus_z_max_abs_ft": coordinate_max,
        "physical_edge_residual_identity_max_abs_ft": physical_edge_max,
        "emission_max_abs": emission_max,
        "initial_prior_max_abs": prior_max,
        "rate_kernel_max_abs": rate_kernel_max,
        "position_kernel_max_abs": position_kernel_max,
        "brute_force_loglik_posterior_max_abs": brute_force_max,
        "real_log_likelihood_max_abs": float(
            parity_ledger["log_likelihood_abs"].max()
        ),
        "smoothed_position_posterior_max_abs": float(
            parity_ledger["position_posterior_max_abs"].max()
        ),
        "smoothed_rate_posterior_max_abs": float(
            parity_ledger["rate_posterior_max_abs"].max()
        ),
        "tvt_mean_std_max_abs_ft": max(
            float(parity_ledger["tvt_mean_max_abs_ft"].max()),
            float(parity_ledger["tvt_std_max_abs_ft"].max()),
        ),
        "candidate_u_minus_z_readout_max_abs_ft": float(
            parity_ledger[
                "candidate_u_minus_z_readout_max_abs_ft"
            ].max()
        ),
        "finite_coverage_min": finite_coverage,
        "readback_sha_match_required": readback_match,
        "forbidden_truth_fold_role_episode_error_reads": forbidden_reads,
        "manifest_wells": len(parity_ledger),
        "paired_hmm_well_runs": 2 * len(parity_ledger),
    }
    checks = {
        "coordinate": coordinate_max
        <= float(gates["coordinate_tvt_equals_u_minus_z_max_abs_ft"]),
        "physical_edge": physical_edge_max
        <= float(gates["physical_edge_residual_identity_max_abs_ft"]),
        "emission": emission_max <= float(gates["emission_max_abs"]),
        "initial_prior": prior_max
        <= float(gates["initial_prior_max_abs"]),
        "rate_kernel": observed["rate_kernel_max_abs"]
        <= float(gates["rate_kernel_max_abs"]),
        "position_kernel": position_kernel_max
        <= float(gates["position_kernel_max_abs"]),
        "brute_force": brute_force_max
        <= float(gates["brute_force_loglik_posterior_max_abs"]),
        "real_log_likelihood": observed["real_log_likelihood_max_abs"]
        <= float(gates["real_log_likelihood_max_abs"]),
        "smoothed_position_posterior": observed[
            "smoothed_position_posterior_max_abs"
        ]
        <= float(gates["smoothed_position_posterior_max_abs"]),
        "smoothed_rate_posterior": observed[
            "smoothed_rate_posterior_max_abs"
        ]
        <= float(gates["smoothed_rate_posterior_max_abs"]),
        "tvt_mean_std": observed["tvt_mean_std_max_abs_ft"]
        <= float(gates["tvt_mean_std_max_abs_ft"]),
        "candidate_u_minus_z_readout": observed[
            "candidate_u_minus_z_readout_max_abs_ft"
        ]
        <= float(gates["candidate_u_minus_z_readout_max_abs_ft"]),
        "finite_coverage": finite_coverage
        >= float(gates["finite_coverage_min"]),
        "readback_sha": readback_match
        is bool(gates["readback_sha_match_required"]),
        "truth_free": forbidden_reads == 0,
        "run_count": len(parity_ledger) == 32
        and 2 * len(parity_ledger) == 64,
    }
    return {
        "observed": observed,
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "decision": (
            "coordinate_parity_verified"
            if all(checks.values())
            else "technical_parity_failed"
        ),
    }


def require_kaggle_runtime() -> None:
    if not (
        KAGGLE_INPUT_ROOT.is_dir()
        and KAGGLE_WORKING_ROOT.is_dir()
    ):
        raise RuntimeError("exp445 Stage 0 is Kaggle-runtime only")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config, require_run_authorization=True)
    scientific = validate_scientific_contract(config)
    require_kaggle_runtime()
    set_num_threads(1)
    started = time.perf_counter()
    ledger = LeakageLedger()
    manifest, manifest_report = load_fixed32_target_free_scope(
        config,
        ledger,
    )
    hmm = scientific["fixed_from_exp209"]
    synthetic = run_synthetic_contracts(hmm)
    raw_dir = train_data_dir(config)
    frozen: list[FrozenWellParity] = []
    for row in manifest.itertuples(index=False):
        frozen.append(
            freeze_target_free_well(
                well=str(row.well),
                expected_prefix_rows=int(row.prefix_rows),
                expected_suffix_rows=int(row.suffix_rows),
                raw_dir=raw_dir,
                hmm=hmm,
                ledger=ledger,
            )
        )
        print(
            json.dumps(
                {
                    "event": "exp445_well_frozen",
                    "well": str(row.well),
                    "completed_wells": len(frozen),
                    "parent_candidate_hmm_runs": 2 * len(frozen),
                },
                sort_keys=True,
            )
        )

    prediction = predictions_frame(frozen)
    parity = parity_ledger_frame(frozen)
    transition = transition_emission_ledger_frame(frozen)
    output = artifacts_dir()
    artifact_reports = {
        "target_free_manifest": write_csv(
            output / "exp445_fixed32_target_free_manifest.csv",
            manifest,
        ),
        "predictions": write_deterministic_gzip_csv(
            output / "exp445_coordinate_parity_predictions.csv.gz",
            prediction,
        ),
        "posterior_parity_ledger": write_deterministic_gzip_csv(
            output / "exp445_posterior_parity_ledger.csv.gz",
            parity,
        ),
        "transition_emission_ledger": write_deterministic_gzip_csv(
            output / "exp445_transition_emission_ledger.csv.gz",
            transition,
        ),
    }
    leakage_report = ledger.report()
    gates = evaluate_technical_gates(
        config=config,
        synthetic=synthetic,
        parity_ledger=parity,
        transition_ledger=transition,
        leakage=leakage_report,
        artifacts=artifact_reports,
    )
    report = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": gates["decision"],
        "evaluation_role": "technical_coordinate_parity_only",
        "scientific_variant": SCIENTIFIC_VARIANT,
        "parent": PARENT_EXPERIMENT,
        "comparison_reference": COMPARISON_REFERENCE,
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "manifest": manifest_report,
        "execution": {
            "coordinate_candidates": 1,
            "manifest_wells": len(frozen),
            "candidate_hmm_well_runs": len(frozen),
            "paired_parent_hmm_well_runs": len(frozen),
            "total_hmm_well_runs": 2 * len(frozen),
            "reporting_folds": 0,
            "lightgbm_configs": 0,
            "trained_ml_folds": 0,
            "boosters": 0,
            "fitted_models": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
        "synthetic": synthetic,
        "technical_gates": gates,
        "leakage_ledger": leakage_report,
        "artifacts": artifact_reports,
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
            "num_workers": 1,
            "numba_threads": 1,
        },
        "reproducibility": {
            "deterministic_anchor": False,
            "first_run_only": True,
            "independent_rerun_required": True,
        },
        "interpretation": (
            "PASS verifies only a coordinate relabel. It does not improve "
            "exp209 or rehabilitate exp438."
        ),
    }
    report_artifact = write_json(
        output / "exp445_stage0_report.json",
        report,
    )
    report["artifacts"]["stage0_report"] = report_artifact
    write_json(metrics_path(), report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    if not gates["all_pass"]:
        raise RuntimeError(
            "exp445 technical parity failed; parameter rescue is forbidden"
        )
    return report


# %% [markdown]
# ## 9. Configuration preview and guarded execution

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    EXECUTION = validate_execution_contract(
        CONFIG,
        require_run_authorization=False,
    )
    SCIENTIFIC = validate_scientific_contract(CONFIG)
    PREVIEW = {
        "event": "exp445_implementation_preview",
        "experiment": EXPERIMENT_NAME,
        "status": get_nested(CONFIG, "experiment.status"),
        "route": get_nested(CONFIG, "experiment.route"),
        "parent": get_nested(CONFIG, "lineage.parent"),
        "comparison_reference": get_nested(
            CONFIG,
            "lineage.comparison_reference",
        ),
        "execution": EXECUTION,
        "coordinate": SCIENTIFIC["coordinate"],
        "message": (
            "Canonical fixed32 Stage 0 is authorized. Inference and "
            "submission remain disabled."
        ),
    }
    print(json.dumps(PREVIEW, indent=2, sort_keys=True))
    if bool(get_nested(CONFIG, "execution.run_hmm", False)):
        run_stage0(CONFIG)
