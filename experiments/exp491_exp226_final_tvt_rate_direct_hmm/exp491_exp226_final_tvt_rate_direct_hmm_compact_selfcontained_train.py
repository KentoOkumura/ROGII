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
# # exp491 exp226 final-TVT-rate direct HMM — Stage 0
#
# This CPU-only notebook changes one scientific variable relative to exp437:
# the TVT-only HMM transition center is the exact adjacent difference of the
# saved, group-safe exp226 final `tvt_pred`, rather than the geometry-only
# `tvt_geop`. There is no persistent rate, offset, or branch state. The fixed32
# run is a mechanism preflight, not CV or promotion evidence.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 identity and strict exp226 final-prediction input
# 4. Final-TVT schedule and unchanged exp437 HMM inputs
# 5. TVT-only direct-transition forward-backward
# 6. Candidate decoding and target-free freeze
# 7. Truth-late fixed32 and persistent-episode readout
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

try:
    import numba as numba_module
    from numba import njit, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:
    numba_module = None
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    def set_num_threads(thread_count: int) -> None:
        del thread_count

EXPERIMENT_NAME = "exp491_exp226_final_tvt_rate_direct_hmm"
PARENT_EXPERIMENT = "exp437_neighbor_geometry_tvt_only_transition_hmm"
PREDICTION_SOURCE = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
)
VARIANT = "exp226_final_rate_direct_transition"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SAFE_EXP226_COLUMNS = (
    "well_id",
    "row_idx",
    "suffix_offset",
    "fold",
    "tvt_pred",
)
FORBIDDEN_EXP226_COLUMNS = frozenset(
    {"TVT", "tvt_true", "tvt_geop", "gr_delta", "error", "abs_error"}
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
        raise ValueError("wrong exp491 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp491 route must remain pf_beam")
    if not bool(get_nested(config, "design.implementation_authorized", False)):
        raise ValueError("exp491 implementation is not authorized")
    if not bool(get_nested(config, "implementation.enabled", False)):
        raise ValueError("exp491 implementation.enabled must be true")
    if get_nested(config, "execution.selected_stage") != "stage_0_fixed32":
        raise ValueError("only Stage 0 fixed32 is implemented")
    variants = tuple(get_nested(config, "model.active_scientific_variants", ()))
    if variants != (VARIANT,):
        raise ValueError("the scientific candidate must remain one fixed variant")
    expected = {
        "current_scientific_variants": 1,
        "current_hmm_well_runs": 32,
        "stage_0_planned_scientific_variants": 1,
        "stage_0_planned_candidate_hmm_well_runs": 32,
        "stage_1_max_candidate_hmm_well_runs": 773,
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
        raise ValueError(f"execution count contract changed: {observed}")
    if bool(get_nested(config, "design.kaggle_stage_1_authorized", False)):
        raise ValueError("Stage 1 is not implemented or authorized")
    if bool(get_nested(config, "design.inference_authorized", False)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "design.submission_authorized", False)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "execution.create_submission", False)):
        raise ValueError("submission creation must remain disabled")
    if require_run_authorization:
        if not bool(get_nested(config, "design.kaggle_stage_0_authorized", False)):
            raise RuntimeError("design.kaggle_stage_0_authorized is false")
        if not bool(get_nested(config, "runtime.run_approved", False)):
            raise RuntimeError("runtime.run_approved is false")
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError("execution.run_hmm is false")
        if not bool(get_nested(config, "execution.create_prediction", False)):
            raise RuntimeError("execution.create_prediction is false")
    return observed


def exp437_hmm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(get_nested(config, "model.fixed_from_exp437") or {})
    expected = {
        "state_coordinate": "absolute_tvt_grid",
        "step_ft": 0.35,
        "sig_p": 0.02,
        "start_sig_ft": 0.75,
        "band_pad_ft": 100.0,
        "emission": "gaussian_typewell_gr",
        "emission_lambda": 1.0,
        "sigma_mode": "known_prefix_population_std",
        "sigma_clip": [10.0, 60.0],
        "missing_gr_policy": "interpolate_both_directions_then_typewell_mean",
        "position_kernel_cells": 5,
        "forward_backward": True,
        "output": "smoothed_posterior_mean",
    }
    if fixed != expected:
        raise ValueError(f"fixed exp437 HMM contract changed: {fixed}")
    return {
        "step": float(fixed["step_ft"]),
        "sig_p": float(fixed["sig_p"]),
        "start_sig": float(fixed["start_sig_ft"]),
        "band_pad": float(fixed["band_pad_ft"]),
        "lam": float(fixed["emission_lambda"]),
    }


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    transition = dict(get_nested(config, "model.transition") or {})
    required = {
        "source_column": "tvt_pred",
        "source_semantics": "saved_group_safe_exp226_final_prediction",
        "tvt_rate_formula": "delta_exp226_final_tvt_pred_div_delta_md",
        "u_rate_audit_formula": (
            "delta_exp226_final_tvt_pred_plus_delta_z_div_delta_md"
        ),
        "hmm_transition_center_formula": "delta_exp226_final_tvt_pred",
        "schedule_updated_by_hmm_emission": False,
        "rate_smoothing": "none",
        "rate_clipping": "none",
        "segment_aggregation": "none",
        "persistent_rate_momentum": "none",
        "position_kernel_cells": 5,
    }
    for key, expected in required.items():
        if transition.get(key) != expected:
            raise ValueError(f"transition contract changed: {key}")
    if bool(get_nested(config, "model.rate_state_present", True)):
        raise ValueError("persistent rate state is forbidden")
    if bool(get_nested(config, "model.residual_offset_state_present", True)):
        raise ValueError("residual offset state is forbidden")
    if bool(get_nested(config, "model.branch_state_present", True)):
        raise ValueError("branch state is forbidden")
    if get_nested(config, "validation.truth_join") != (
        "after_schedule_prediction_diagnostic_and_content_sha_freeze"
    ):
        raise ValueError("truth-late contract changed")
    allowlist = tuple(
        get_nested(config, "data.exp226_final_oof.pre_freeze_allowlist", ())
    )
    if allowlist != SAFE_EXP226_COLUMNS:
        raise ValueError("exp226 read-time allowlist changed")
    forbidden = frozenset(
        get_nested(config, "data.exp226_final_oof.pre_freeze_forbidden", ())
    )
    if forbidden != FORBIDDEN_EXP226_COLUMNS:
        raise ValueError("exp226 forbidden-column contract changed")
    hmm = exp437_hmm_config(config)
    return {
        "variant": VARIANT,
        "parent": PARENT_EXPERIMENT,
        "prediction_source": PREDICTION_SOURCE,
        "persistent_state": "tvt_probability_distribution_only",
        "transition_center": "adjacent_difference_of_group_safe_exp226_final_tvt_pred",
        "first_transition": "tvt_pred_0_minus_last_known_tvt_input",
        "rate_identity": "u_rate_times_delta_md_minus_delta_z_equals_delta_tvt_pred",
        "fixed_exp437_hmm": hmm,
        "exp226_allowlist": list(allowlist),
        "exp226_forbidden": sorted(forbidden),
        "rate_state_present": False,
        "residual_offset_state_present": False,
        "branch_state_present": False,
        "parent_control_hmm_reruns": 0,
        "truth_join": get_nested(config, "validation.truth_join"),
        "raw_suffix_gr_evidence_reused": True,
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
    raise FileNotFoundError("exp491 config.yaml was not found")


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


def write_deterministic_gzip_csv(
    path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=1,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text:
                frame.to_csv(text, index=False, lineterminator="\n")
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
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": (
            str(numba_module.__version__)
            if numba_module is not None
            else "unavailable"
        ),
    }


def candidate_paths(spec: Mapping[str, Any]) -> list[Path]:
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
        for pattern in spec.get("patterns", ()):
            matches.extend(KAGGLE_INPUT_ROOT.glob(str(pattern)))
    return sorted({path.resolve() for path in matches if path.is_file()})


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
    expected_wells: int = 32
    frozen_wells: set[str] = field(default_factory=set)
    exp226_rows_read_with_allowlist: int = 0
    forbidden_exp226_columns_read_before_freeze: int = 0
    truth_rows_before_all_freeze: int = 0
    role_rows_before_all_freeze: int = 0
    episode_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    role_rows_after_all_freeze: int = 0
    episode_rows_after_all_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def freeze(self, well: str) -> None:
        self.frozen_wells.add(str(well))

    def _record(self, kind: str, rows: int) -> None:
        before = f"{kind}_rows_before_all_freeze"
        after = f"{kind}_rows_after_all_freeze"
        if not self.all_frozen:
            setattr(self, before, int(getattr(self, before)) + int(rows))
            raise RuntimeError(f"{kind} was read before all candidates froze")
        setattr(self, after, int(getattr(self, after)) + int(rows))

    def record_truth(self, rows: int) -> None:
        self._record("truth", rows)

    def record_roles(self, rows: int) -> None:
        self._record("role", rows)

    def record_episodes(self, rows: int) -> None:
        self._record("episode", rows)


# %% [markdown]
# ## 3. Fixed32 identity and strict exp226 final-prediction input
#
# Before candidate freeze the fixed32 manifest exposes only well identity and
# prefix/suffix row counts. The exp226 gzip is verified by decompressed content
# SHA and read with the five-column allowlist at `read_csv` time. Truth, role,
# episode boundaries, `tvt_geop`, and error columns never enter the decoder.

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


def load_fixed32_identity(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.fixed32_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed: {observed}")
    header = pd.read_csv(path, nrows=0)
    required = {"well", "role", "fold", "prefix_rows", "suffix_rows"}
    if not required.issubset(header.columns):
        raise ValueError("fixed32 manifest schema changed")
    frame = pd.read_csv(
        path,
        usecols=["well", "prefix_rows", "suffix_rows"],
        dtype={"well": str},
    )
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 manifest must contain 32 unique wells")
    if int(frame["suffix_rows"].sum()) != 156_088:
        raise ValueError("fixed32 suffix row count changed")
    frame = frame.sort_values("well", kind="mergesort").reset_index(drop=True)
    return frame, {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "logical_sha256": logical_frame_sha256(frame),
        "mechanism_only_not_cv_or_promotion": True,
    }


def load_scope_after_freeze(
    config: Mapping[str, Any],
    identity: pd.DataFrame,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("scope identity is unavailable before candidate freeze")
    spec = get_nested(config, "data.fixed32_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    if sha256_file(path) != str(spec["expected_sha256"]):
        raise ValueError("fixed32 manifest SHA changed after freeze")
    frame = pd.read_csv(
        path,
        usecols=["well", "role", "fold", "prefix_rows", "suffix_rows"],
        dtype={"well": str},
    )
    ledger.record_roles(len(frame))
    if frame["role"].value_counts().to_dict() != {
        "persistent": 16,
        "control": 16,
    }:
        raise ValueError("fixed32 role counts changed")
    if frame.groupby("fold").size().to_dict() != {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}:
        raise ValueError("fixed32 fold counts changed")
    if set(frame["well"]) != set(identity["well"]):
        raise ValueError("fixed32 well identity changed")
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


def load_exp226_final_oof(
    config: Mapping[str, Any],
    target_wells: set[str],
    expected_rows: int,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_final_oof")
    paths = candidate_paths(spec)
    expected_sha = str(spec["expected_decompressed_sha256"])
    matching = [path for path in paths if sha256_decompressed_csv(path) == expected_sha]
    if not matching:
        raise FileNotFoundError("SHA-matching exp226 final OOF was not found")
    path = matching[0]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=SAFE_EXP226_COLUMNS,
        dtype={"well_id": str},
        chunksize=200_000,
    ):
        if FORBIDDEN_EXP226_COLUMNS.intersection(chunk.columns):
            ledger.forbidden_exp226_columns_read_before_freeze += len(chunk)
            raise ValueError("forbidden exp226 columns crossed the read-time allowlist")
        selected = chunk.loc[chunk["well_id"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("exp226 final OOF contains none of the fixed32 wells")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if len(frame) != expected_rows:
        raise ValueError(f"exp226 fixed32 rows={len(frame)}/{expected_rows}")
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 final OOF keys are not unique")
    if not np.isfinite(frame["tvt_pred"].to_numpy(np.float64)).all():
        raise ValueError("exp226 final tvt_pred must be finite")
    ledger.exp226_rows_read_with_allowlist += len(frame)
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": expected_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "allowlist": list(SAFE_EXP226_COLUMNS),
        "allowlist_schema_sha256": hashlib.sha256(
            stable_json_bytes(list(SAFE_EXP226_COLUMNS))
        ).hexdigest(),
        "matching_candidates": len(matching),
    }


def load_target_free_well(
    well: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=lambda column: str(column) != "TVT",
    )
    if "TVT" in horizontal.columns:
        raise ValueError(f"{well}: unknown-suffix TVT reached the decoder")
    typewell = pd.read_csv(raw_dir / f"{well}__typewell.csv")
    return horizontal, typewell.sort_values("TVT", kind="mergesort").reset_index(
        drop=True
    )


# %% [markdown]
# ## 4. Final-TVT schedule and unchanged exp437 HMM inputs
#
# The exp437 grid, start prior, Gaussian typewell-GR emission, position noise,
# and five-cell kernel are fixed. The only changed scientific array is
# `transition_delta = diff([last_known_TVT_input, exp226_final_tvt_pred])`.

# %%
def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[float, float]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_gr = known["GR"].fillna(0).to_numpy(np.float64)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
    return sigma, float(known_tvt[-1])


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    if not {"MD", "Z", "GR", "TVT_input"}.issubset(horizontal.columns):
        raise ValueError("horizontal input schema changed")
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell input schema changed")
    if "TVT" in horizontal.columns:
        raise ValueError("unknown-suffix TVT reached HMM preparation")
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    suffix = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or suffix.empty:
        raise ValueError("expected a visible prefix and a non-empty suffix")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    gr_sigma, last_tvt = prefix_stats(horizontal, typewell_tvt, typewell_gr)
    step = float(hmm["step"])
    grid_min = max(typewell_tvt.min() - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(typewell_tvt.max() + 40.0, last_tvt + float(hmm["band_pad"]))
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = np.interp(grid, typewell_tvt, typewell_gr)
    raw_gr = suffix["GR"].to_numpy(np.float64)
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[suffix.index]
    )
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    last = known.iloc[-1]
    return {
        "emission_ll": emission_ll,
        "grid": grid,
        "start_p": float((last_tvt - grid_min) / step),
        "eval_index": suffix.index.to_numpy(np.int64),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "last_known_z": float(last["Z"]),
        "suffix_md": suffix["MD"].to_numpy(np.float64),
        "suffix_z": suffix["Z"].to_numpy(np.float64),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
    }


def build_final_tvt_schedule(
    source: pd.DataFrame,
    *,
    expected_row_idx: np.ndarray,
    suffix_md: np.ndarray,
    suffix_z: np.ndarray,
    last_known_tvt: float,
    last_known_md: float,
    last_known_z: float,
) -> dict[str, Any]:
    required = set(SAFE_EXP226_COLUMNS)
    if not required.issubset(source.columns):
        raise ValueError("exp226 allowlist schema is incomplete")
    if FORBIDDEN_EXP226_COLUMNS.intersection(source.columns):
        raise ValueError("forbidden columns reached schedule construction")
    ordered = source.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = ordered["row_idx"].to_numpy(np.int64)
    suffix_offset = ordered["suffix_offset"].to_numpy(np.int64)
    path = ordered["tvt_pred"].to_numpy(np.float64)
    md = np.asarray(suffix_md, dtype=np.float64)
    z = np.asarray(suffix_z, dtype=np.float64)
    if not np.array_equal(row_idx, np.asarray(expected_row_idx, dtype=np.int64)):
        raise ValueError("exp226 row_idx does not align with raw suffix")
    if not np.array_equal(suffix_offset, np.arange(len(path), dtype=np.int64)):
        raise ValueError("exp226 suffix_offset is not contiguous")
    if len(row_idx) > 1 and not np.all(np.diff(row_idx) == 1):
        raise ValueError("exp226 row_idx is not contiguous")
    if path.shape != md.shape or path.shape != z.shape:
        raise ValueError("schedule/raw coordinate shape mismatch")
    transition_delta = np.diff(
        np.concatenate([[float(last_known_tvt)], path])
    )
    explicit = np.empty_like(path)
    explicit[0] = path[0] - float(last_known_tvt)
    explicit[1:] = path[1:] - path[:-1]
    delta_md = np.diff(np.concatenate([[float(last_known_md)], md]))
    delta_z = np.diff(np.concatenate([[float(last_known_z)], z]))
    if not np.isfinite(
        np.concatenate([path, transition_delta, delta_md, delta_z])
    ).all():
        raise ValueError("schedule inputs must be finite")
    if not np.all(delta_md > 0.0):
        raise ValueError("suffix delta_MD must be strictly positive")
    tvt_rate = transition_delta / delta_md
    u_rate = (transition_delta + delta_z) / delta_md
    identity_error = u_rate * delta_md - delta_z - transition_delta
    return {
        "row_idx": row_idx,
        "tvt_pred": path,
        "transition_delta": transition_delta,
        "delta_md": delta_md,
        "delta_z": delta_z,
        "tvt_rate": tvt_rate,
        "u_rate": u_rate,
        "first_difference_parity_max_abs_ft": float(
            np.max(np.abs(transition_delta - explicit))
        ),
        "rate_increment_identity_max_abs_ft": float(
            np.max(np.abs(identity_error))
        ),
        "logical_sha256": array_bundle_sha256(
            row_idx=row_idx,
            tvt_pred=path,
            transition_delta=transition_delta,
            delta_md=delta_md,
            delta_z=delta_z,
            tvt_rate=tvt_rate,
            u_rate=u_rate,
        ),
    }


# %% [markdown]
# ## 5. TVT-only direct-transition forward-backward
#
# The persistent state is one probability vector over the fixed absolute-TVT
# grid. Each row uses one five-cell position kernel centered at the frozen
# exp226-final increment. There is no rate support, momentum, or branch state.

# %%
@njit(cache=True, nogil=True)
def direct_position_kernel(
    expected_delta: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    center = int(np.floor(expected_delta / step + 0.5))
    offsets = np.empty(5, np.int64)
    probabilities = np.empty(5, np.float64)
    sigma_position = max(sig_p, 0.35 * step)
    total = 0.0
    for kernel_index in range(5):
        offset = center - 2 + kernel_index
        residual = offset * step - expected_delta
        probability = np.exp(-0.5 * (residual / sigma_position) ** 2)
        offsets[kernel_index] = offset
        probabilities[kernel_index] = probability
        total += probability
    for kernel_index in range(5):
        probabilities[kernel_index] /= total
    return offsets, probabilities, abs(probabilities.sum() - 1.0)


@njit(cache=True, nogil=True)
def _direct_transition_forward_backward(
    emission_ll: np.ndarray,
    transition_delta: np.ndarray,
    step: float,
    sig_p: float,
    start_p: float,
    start_sig: float,
    lam: float,
):
    time_count, position_count = emission_ll.shape
    alpha = np.empty((time_count, position_count), np.float32)
    posterior = np.empty((time_count, position_count), np.float64)
    previous = np.empty(position_count, np.float64)
    predictive = np.empty(position_count, np.float64)
    current = np.empty(position_count, np.float64)
    likelihood = np.empty(position_count, np.float64)
    maximum_transition_error = 0.0
    maximum_forward_error = 0.0
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
        offsets, kernel, row_error = direct_position_kernel(
            transition_delta[time_index],
            step,
            sig_p,
        )
        maximum_transition_error = max(maximum_transition_error, row_error)
        for position_index in range(position_count):
            predictive[position_index] = 0.0
        for kernel_index in range(5):
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
            emission_max = max(
                emission_max,
                lam * float(emission_ll[time_index, position_index]),
            )
        current_total = 0.0
        for position_index in range(position_count):
            likelihood[position_index] = np.exp(
                lam * float(emission_ll[time_index, position_index]) - emission_max
            )
            current[position_index] = (
                predictive[position_index] * likelihood[position_index]
            )
            current_total += current[position_index]
        if current_total <= 0.0 or not np.isfinite(current_total):
            raise RuntimeError("TVT-only forward message became non-finite")
        log_likelihood += emission_max + np.log(current_total)
        normalized_total = 0.0
        for position_index in range(position_count):
            normalized = current[position_index] / current_total
            alpha[time_index, position_index] = np.float32(normalized)
            previous[position_index] = normalized
            normalized_total += normalized
        maximum_forward_error = max(
            maximum_forward_error,
            abs(normalized_total - 1.0),
        )

    for position_index in range(position_count):
        posterior[time_count - 1, position_index] = float(
            alpha[time_count - 1, position_index]
        )
    posterior[time_count - 1] /= posterior[time_count - 1].sum()
    beta_next = np.ones(position_count, np.float64)
    beta_current = np.empty(position_count, np.float64)
    for time_index in range(time_count - 1, 0, -1):
        offsets, kernel, _ = direct_position_kernel(
            transition_delta[time_index],
            step,
            sig_p,
        )
        emission_max = -1.0e300
        for position_index in range(position_count):
            emission_max = max(
                emission_max,
                lam * float(emission_ll[time_index, position_index]),
            )
        for position_index in range(position_count):
            likelihood[position_index] = np.exp(
                lam * float(emission_ll[time_index, position_index]) - emission_max
            )
        for source_index in range(position_count):
            total = 0.0
            for kernel_index in range(5):
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

    maximum_posterior_error = 0.0
    for time_index in range(time_count):
        maximum_posterior_error = max(
            maximum_posterior_error,
            abs(posterior[time_index].sum() - 1.0),
        )
    return (
        posterior,
        log_likelihood,
        maximum_transition_error,
        maximum_forward_error,
        maximum_posterior_error,
    )


def run_direct_transition_hmm(
    prepared: Mapping[str, Any],
    transition_delta: np.ndarray,
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    transition_delta = np.asarray(transition_delta, dtype=np.float64)
    if transition_delta.shape != (len(prepared["eval_index"]),):
        raise ValueError("transition schedule length does not match suffix rows")
    if not np.isfinite(transition_delta).all():
        raise ValueError("transition schedule must be finite")
    (
        posterior,
        log_likelihood,
        transition_error,
        forward_error,
        posterior_error,
    ) = _direct_transition_forward_backward(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        transition_delta,
        float(hmm["step"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(hmm["lam"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    posterior_mean = posterior @ grid
    posterior_second = posterior @ (grid * grid)
    posterior_std = np.sqrt(
        np.maximum(posterior_second - posterior_mean * posterior_mean, 0.0)
    )
    return {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "log_likelihood": float(log_likelihood),
        "transition_row_sum_max_error": float(transition_error),
        "posterior_normalization_max_error": max(
            float(forward_error),
            float(posterior_error),
        ),
        "prediction_sha256": array_bundle_sha256(
            row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
            posterior_mean=np.asarray(posterior_mean, dtype=np.float32),
            posterior_std=np.asarray(posterior_std, dtype=np.float32),
        ),
        "diagnostic_sha256": array_bundle_sha256(
            row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
            transition_delta=transition_delta,
            posterior_std=np.asarray(posterior_std, dtype=np.float32),
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 6. Candidate decoding and target-free freeze
#
# Each fixed32 well is decoded exactly once. The saved exp226 final path is the
# comparison control and the transition schedule source; it is never blended
# with the HMM output. Schedule, prediction, and diagnostic SHAs freeze before
# role, fold outcome, episode boundaries, suffix truth, or errors are attached.

# %%
@dataclass
class FrozenWell:
    well: str
    row_idx: np.ndarray
    suffix_offset: np.ndarray
    source_fold: np.ndarray
    exp226_prediction: np.ndarray
    transition_delta: np.ndarray
    delta_md: np.ndarray
    delta_z: np.ndarray
    tvt_rate: np.ndarray
    u_rate: np.ndarray
    candidate_prediction: np.ndarray
    posterior_std: np.ndarray
    raw_gr_missing: np.ndarray
    prefix_rows: int
    schedule_sha256: str
    prediction_sha256: str
    diagnostic_sha256: str
    first_difference_parity_max_abs_ft: float
    rate_increment_identity_max_abs_ft: float
    transition_row_sum_max_error: float
    posterior_normalization_max_error: float
    log_likelihood: float
    hmm_seconds: float
    role: str = ""
    fold: int = -1


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    exp226: pd.DataFrame,
    hmm: Mapping[str, Any],
    first_difference_tolerance_ft: float,
    rate_identity_tolerance_ft: float,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir)
    prepared = prepare_hmm_inputs(horizontal, typewell, hmm)
    schedule = build_final_tvt_schedule(
        exp226,
        expected_row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        suffix_md=np.asarray(prepared["suffix_md"], dtype=np.float64),
        suffix_z=np.asarray(prepared["suffix_z"], dtype=np.float64),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
    )
    if (
        schedule["first_difference_parity_max_abs_ft"]
        > first_difference_tolerance_ft
    ):
        raise RuntimeError(f"{well}: first-difference parity failed")
    if (
        schedule["rate_increment_identity_max_abs_ft"]
        > rate_identity_tolerance_ft
    ):
        raise RuntimeError(f"{well}: TVT/U-rate identity failed")
    result = run_direct_transition_hmm(
        prepared,
        schedule["transition_delta"],
        hmm,
    )
    ledger.freeze(well)
    ordered = exp226.sort_values("row_idx", kind="mergesort")
    return FrozenWell(
        well=str(well),
        row_idx=np.asarray(schedule["row_idx"], dtype=np.int64),
        suffix_offset=ordered["suffix_offset"].to_numpy(np.int64),
        source_fold=ordered["fold"].to_numpy(np.int64),
        exp226_prediction=np.asarray(schedule["tvt_pred"], dtype=np.float64),
        transition_delta=np.asarray(
            schedule["transition_delta"], dtype=np.float64
        ),
        delta_md=np.asarray(schedule["delta_md"], dtype=np.float64),
        delta_z=np.asarray(schedule["delta_z"], dtype=np.float64),
        tvt_rate=np.asarray(schedule["tvt_rate"], dtype=np.float64),
        u_rate=np.asarray(schedule["u_rate"], dtype=np.float64),
        candidate_prediction=np.asarray(result["posterior_mean"], dtype=np.float64),
        posterior_std=np.asarray(result["posterior_std"], dtype=np.float64),
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        prefix_rows=int(prepared["prefix_rows"]),
        schedule_sha256=str(schedule["logical_sha256"]),
        prediction_sha256=str(result["prediction_sha256"]),
        diagnostic_sha256=str(result["diagnostic_sha256"]),
        first_difference_parity_max_abs_ft=float(
            schedule["first_difference_parity_max_abs_ft"]
        ),
        rate_increment_identity_max_abs_ft=float(
            schedule["rate_increment_identity_max_abs_ft"]
        ),
        transition_row_sum_max_error=float(
            result["transition_row_sum_max_error"]
        ),
        posterior_normalization_max_error=float(
            result["posterior_normalization_max_error"]
        ),
        log_likelihood=float(result["log_likelihood"]),
        hmm_seconds=float(result["elapsed_seconds"]),
    )


def attach_scope_identity(frozen: FrozenWell, manifest_row: pd.Series) -> None:
    frozen.role = str(manifest_row["role"])
    frozen.fold = int(manifest_row["fold"])
    if not np.all(frozen.source_fold == frozen.fold):
        raise ValueError(f"{frozen.well}: exp226 source fold and manifest fold differ")


def prediction_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "suffix_offset": item.suffix_offset,
                    "exp226_final_tvt_pred": item.exp226_prediction,
                    "exp226_final_transition_delta": item.transition_delta,
                    "delta_md": item.delta_md,
                    "delta_z": item.delta_z,
                    "exp226_implied_tvt_rate": item.tvt_rate,
                    "exp226_implied_u_rate": item.u_rate,
                    "candidate_prediction": item.candidate_prediction,
                    "candidate_posterior_std": item.posterior_std,
                    "raw_gr_missing": item.raw_gr_missing,
                }
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def schedule_manifest_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "well": item.well,
                "rows": len(item.row_idx),
                "schedule_sha256": item.schedule_sha256,
                "prediction_sha256": item.prediction_sha256,
                "diagnostic_sha256": item.diagnostic_sha256,
                "first_difference_parity_max_abs_ft": (
                    item.first_difference_parity_max_abs_ft
                ),
                "rate_increment_identity_max_abs_ft": (
                    item.rate_increment_identity_max_abs_ft
                ),
                "positive_delta_md_fraction": float(np.mean(item.delta_md > 0.0)),
                "transition_row_sum_max_error": (
                    item.transition_row_sum_max_error
                ),
                "posterior_normalization_max_error": (
                    item.posterior_normalization_max_error
                ),
                "hmm_seconds": item.hmm_seconds,
            }
            for item in sorted(frozen_wells, key=lambda value: value.well)
        ]
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
# ## 7. Truth-late fixed32 and persistent-episode readout

# %%
def load_truth_after_freeze(
    frozen: FrozenWell,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> np.ndarray:
    frame = pd.read_csv(
        raw_dir / f"{frozen.well}__horizontal_well.csv",
        usecols=["TVT", "TVT_input"],
    )
    suffix = frame.loc[frame["TVT_input"].isna()]
    ledger.record_truth(len(suffix))
    if not np.array_equal(suffix.index.to_numpy(np.int64), frozen.row_idx):
        raise ValueError(f"{frozen.well}: truth row index changed after freeze")
    actual = suffix["TVT"].to_numpy(np.float64)
    if not np.isfinite(actual).all():
        raise ValueError(f"{frozen.well}: suffix truth must be finite")
    return actual


def truth_late_frame(
    frozen_wells: Sequence[FrozenWell],
    raw_dir: Path,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for frozen in frozen_wells:
        actual = load_truth_after_freeze(frozen, raw_dir, ledger)
        pieces.append(
            pd.DataFrame(
                {
                    "well": frozen.well,
                    "row_idx": frozen.row_idx,
                    "suffix_offset": frozen.suffix_offset,
                    "role": frozen.role,
                    "fold": frozen.fold,
                    "tvt_true": actual,
                    "exp226_final_tvt_pred": frozen.exp226_prediction,
                    "candidate_prediction": frozen.candidate_prediction,
                    "raw_gr_missing": frozen.raw_gr_missing,
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True)
    frame["exp226_error"] = (
        frame["exp226_final_tvt_pred"] - frame["tvt_true"]
    )
    frame["candidate_error"] = frame["candidate_prediction"] - frame["tvt_true"]
    return frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )


def load_persistent_episodes_after_freeze(
    config: Mapping[str, Any],
    persistent_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.all_frozen:
        raise RuntimeError("persistent episodes require complete freeze")
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"persistent episode SHA changed: {observed}")
    columns = [
        "episode_id",
        "well",
        "start_row_idx",
        "end_row_idx_exclusive",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"well": str})
    frame = frame.loc[frame["well"].isin(persistent_wells)].copy()
    ledger.record_episodes(len(frame))
    if frame.empty or set(frame["well"]) != persistent_wells:
        raise ValueError("fixed persistent wells are missing episode boundaries")
    return frame.sort_values(
        ["well", "start_row_idx"], kind="mergesort"
    ).reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "selected_rows": len(frame),
        "selected_wells": int(frame["well"].nunique()),
        "loaded_after_freeze": True,
    }


def rmse_from_error(error: np.ndarray | pd.Series) -> float:
    values = np.asarray(error, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("RMSE requires non-empty finite errors")
    return float(np.sqrt(np.mean(np.square(values))))


def build_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, part in readout.groupby("well", sort=True):
        exp226_rmse = rmse_from_error(part["exp226_error"])
        candidate_rmse = rmse_from_error(part["candidate_error"])
        rows.append(
            {
                "well": str(well),
                "role": str(part["role"].iloc[0]),
                "fold": int(part["fold"].iloc[0]),
                "rows": len(part),
                "exp226_sse": float(np.square(part["exp226_error"]).sum()),
                "candidate_sse": float(np.square(part["candidate_error"]).sum()),
                "exp226_rmse_ft": exp226_rmse,
                "candidate_rmse_ft": candidate_rmse,
                "candidate_delta_vs_exp226_ft": candidate_rmse - exp226_rmse,
                "candidate_improved_vs_exp226": candidate_rmse < exp226_rmse,
                "raw_gr_missing_fraction": float(part["raw_gr_missing"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_episode_metrics(
    episodes: pd.DataFrame,
    readout: pd.DataFrame,
) -> pd.DataFrame:
    grouped = {
        str(well): part.sort_values("row_idx", kind="mergesort")
        for well, part in readout.groupby("well", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        well = str(episode.well)
        start = int(episode.start_row_idx)
        end = int(episode.end_row_idx_exclusive)
        part = grouped[well]
        window = part.loc[part["row_idx"].ge(start) & part["row_idx"].lt(end)]
        if window.empty:
            raise ValueError(f"{episode.episode_id}: fixed episode window is empty")
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": well,
                "start_row_idx": start,
                "end_row_idx_exclusive": end,
                "rows": len(window),
                "exp226_sse": float(np.square(window["exp226_error"]).sum()),
                "candidate_sse": float(np.square(window["candidate_error"]).sum()),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. Stage 0 gates, generated artifacts, and metrics

# %%
def pooled_rmse(frame: pd.DataFrame, sse_column: str) -> float:
    return float(math.sqrt(frame[sse_column].sum() / frame["rows"].sum()))


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    scope_manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    prediction_artifact: Mapping[str, Any],
    schedule_artifact: Mapping[str, Any],
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "gates.stage_0_technical")
    mechanism_config = get_nested(config, "gates.stage_0_mechanism")
    predictions = prediction_frame(frozen_wells)
    total_rows = len(predictions)
    max_transition_error = max(
        item.transition_row_sum_max_error for item in frozen_wells
    )
    max_posterior_error = max(
        item.posterior_normalization_max_error for item in frozen_wells
    )
    max_difference_parity = max(
        item.first_difference_parity_max_abs_ft for item in frozen_wells
    )
    max_rate_identity = max(
        item.rate_increment_identity_max_abs_ft for item in frozen_wells
    )
    finite_columns = [
        "exp226_final_tvt_pred",
        "exp226_final_transition_delta",
        "delta_md",
        "delta_z",
        "exp226_implied_tvt_rate",
        "exp226_implied_u_rate",
        "candidate_prediction",
        "candidate_posterior_std",
    ]
    finite_coverage = float(
        np.isfinite(predictions[finite_columns].to_numpy(np.float64)).mean()
    )
    positive_delta_md_coverage = float(
        np.mean(predictions["delta_md"].to_numpy(np.float64) > 0.0)
    )
    candidate_seconds = float(sum(item.hmm_seconds for item in frozen_wells))
    projected_full_runtime = candidate_seconds * 773.0 / 32.0
    source_fold_match = float(
        np.mean(
            np.concatenate(
                [
                    item.source_fold == item.fold
                    for item in sorted(frozen_wells, key=lambda value: value.well)
                ]
            )
        )
    )
    pre_freeze_truth_role_episode = (
        ledger.truth_rows_before_all_freeze
        + ledger.role_rows_before_all_freeze
        + ledger.episode_rows_before_all_freeze
    )
    technical = {
        "fixed32_roles_and_unique_wells": bool(
            len(scope_manifest) == int(technical_config["expected_wells"])
            and scope_manifest["well"].nunique()
            == int(technical_config["expected_wells"])
            and scope_manifest["role"].value_counts().to_dict()
            == {"persistent": 16, "control": 16}
        ),
        "fixed32_rows": total_rows == int(technical_config["expected_rows"]),
        "fixed32_folds": scope_manifest["fold"].nunique()
        == int(technical_config["expected_folds"]),
        "source_manifest_fold_match": source_fold_match
        >= float(technical_config["source_manifest_fold_match_min"]),
        "duplicate_rows": int(predictions.duplicated(["well", "row_idx"]).sum())
        <= int(technical_config["duplicate_rows_max"]),
        "missing_rows": total_rows == int(technical_config["expected_rows"]),
        "finite_source_schedule_prediction_coverage": finite_coverage
        >= float(
            technical_config["finite_source_schedule_prediction_coverage_min"]
        ),
        "positive_delta_md_coverage": positive_delta_md_coverage
        >= float(technical_config["positive_delta_md_coverage_min"]),
        "forbidden_exp226_columns_before_freeze": (
            ledger.forbidden_exp226_columns_read_before_freeze
            <= int(technical_config["forbidden_column_reads_before_freeze_max"])
        ),
        "truth_role_episode_reads_before_freeze": pre_freeze_truth_role_episode
        <= int(technical_config["truth_role_episode_reads_before_freeze_max"]),
        "first_difference_parity": max_difference_parity
        <= float(technical_config["first_difference_parity_max_abs_ft"]),
        "rate_increment_identity": max_rate_identity
        <= float(technical_config["rate_increment_identity_max_abs_ft"]),
        "transition_row_sum": max_transition_error
        <= float(technical_config["transition_row_sum_max_error"]),
        "posterior_normalization": max_posterior_error
        <= float(technical_config["posterior_normalization_max_error"]),
        "prediction_readback_logical_sha": (
            prediction_artifact["logical_sha256"]
            == prediction_artifact["readback_logical_sha256"]
        ),
        "schedule_manifest_sha_recorded": bool(
            schedule_artifact["logical_sha256"]
        ),
        "projected_full_runtime": projected_full_runtime
        <= float(technical_config["projected_full_runtime_seconds_max"]),
        "peak_rss": peak_rss_gb()
        <= float(technical_config["peak_rss_gb_max"]),
    }

    all_candidate = pooled_rmse(well_metrics, "candidate_sse")
    all_exp226 = pooled_rmse(well_metrics, "exp226_sse")
    matched = well_metrics.loc[well_metrics["role"].eq("control")]
    persistent = well_metrics.loc[well_metrics["role"].eq("persistent")]
    matched_candidate = pooled_rmse(matched, "candidate_sse")
    matched_exp226 = pooled_rmse(matched, "exp226_sse")
    persistent_candidate = pooled_rmse(persistent, "candidate_sse")
    persistent_exp226 = pooled_rmse(persistent, "exp226_sse")
    fold_rows: list[dict[str, Any]] = []
    for fold, part in well_metrics.groupby("fold", sort=True):
        candidate_rmse = pooled_rmse(part, "candidate_sse")
        exp226_rmse = pooled_rmse(part, "exp226_sse")
        fold_rows.append(
            {
                "fold": int(fold),
                "candidate_rmse_ft": candidate_rmse,
                "exp226_final_rmse_ft": exp226_rmse,
                "delta_ft": candidate_rmse - exp226_rmse,
                "improved": candidate_rmse < exp226_rmse,
            }
        )
    improving_folds = sum(row["improved"] for row in fold_rows)
    paired_delta = well_metrics[
        "candidate_delta_vs_exp226_ft"
    ].to_numpy(np.float64)
    paired_p95 = float(np.quantile(paired_delta, 0.95))
    paired_worst = float(np.max(paired_delta))
    episode_exp226_sse = float(episode_metrics["exp226_sse"].sum())
    episode_candidate_sse = float(episode_metrics["candidate_sse"].sum())
    episode_sse_reduction = (
        1.0 - episode_candidate_sse / episode_exp226_sse
        if episode_exp226_sse > 0.0
        else math.nan
    )
    mechanism = {
        "gain_vs_exp226_final_all32": all_exp226 - all_candidate
        >= float(mechanism_config["gain_vs_exp226_final_all32_min_ft"]),
        "matched_control_delta_vs_exp226_final": (
            matched_candidate - matched_exp226
            <= float(
                mechanism_config[
                    "matched_control_delta_vs_exp226_final_max_ft"
                ]
            )
        ),
        "persistent_gain_vs_exp226_final": (
            persistent_exp226 - persistent_candidate
            >= float(
                mechanism_config["persistent_gain_vs_exp226_final_min_ft"]
            )
        ),
        "improving_folds_vs_exp226_final": improving_folds
        >= int(mechanism_config["improving_folds_vs_exp226_final_min"]),
        "persistent_episode_sse_reduction": (
            math.isfinite(episode_sse_reduction)
            and episode_sse_reduction
            >= float(
                mechanism_config["persistent_episode_sse_reduction_min"]
            )
        ),
        "paired_by_well_delta_p95": paired_p95
        <= float(mechanism_config["paired_by_well_delta_p95_max_ft"]),
        "worst_well_delta": paired_worst
        <= float(mechanism_config["worst_well_delta_max_ft"]),
    }
    observed_rows = readout.loc[~readout["raw_gr_missing"]]
    missing_rows = readout.loc[readout["raw_gr_missing"]]
    raw_gr_scopes = {
        "observed": {
            "rows": len(observed_rows),
            "exp226_rmse_ft": rmse_from_error(observed_rows["exp226_error"]),
            "candidate_rmse_ft": rmse_from_error(
                observed_rows["candidate_error"]
            ),
        },
        "missing": {
            "rows": len(missing_rows),
            "exp226_rmse_ft": rmse_from_error(missing_rows["exp226_error"]),
            "candidate_rmse_ft": rmse_from_error(missing_rows["candidate_error"]),
        },
    }
    for scope in raw_gr_scopes.values():
        scope["delta_ft"] = (
            scope["candidate_rmse_ft"] - scope["exp226_rmse_ft"]
        )
    all_gates_pass = bool(all(technical.values()) and all(mechanism.values()))
    diagnostics = {
        "candidate_all32_rmse_ft": all_candidate,
        "exp226_final_all32_rmse_ft": all_exp226,
        "gain_vs_exp226_final_all32_ft": all_exp226 - all_candidate,
        "candidate_matched_control_rmse_ft": matched_candidate,
        "exp226_final_matched_control_rmse_ft": matched_exp226,
        "candidate_persistent_rmse_ft": persistent_candidate,
        "exp226_final_persistent_rmse_ft": persistent_exp226,
        "improving_folds_vs_exp226_final": improving_folds,
        "fold_metrics": fold_rows,
        "persistent_episode_sse_reduction_fraction": episode_sse_reduction,
        "paired_by_well_delta_p95_ft": paired_p95,
        "worst_well_delta_ft": paired_worst,
        "raw_gr_scopes_report_only_stage0": raw_gr_scopes,
        "maximum_transition_row_sum_error": max_transition_error,
        "maximum_posterior_normalization_error": max_posterior_error,
        "maximum_first_difference_parity_abs_ft": max_difference_parity,
        "maximum_rate_increment_identity_abs_ft": max_rate_identity,
        "finite_coverage": finite_coverage,
        "positive_delta_md_coverage": positive_delta_md_coverage,
        "source_manifest_fold_match": source_fold_match,
        "candidate_hmm_seconds": candidate_seconds,
        "stage0_elapsed_seconds": float(elapsed_seconds),
        "projected_full_runtime_seconds": projected_full_runtime,
        "peak_rss_gb": peak_rss_gb(),
        "fixed32_is_mechanism_only_not_cv_or_promotion": True,
        "raw_suffix_gr_evidence_reused": True,
    }
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "stage0_all_gates_pass": all_gates_pass,
        "stage1_eligible_for_separate_approval": all_gates_pass,
        "fixed32_is_cv": False,
        "fixed32_is_promotion_evidence": False,
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP491_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp491 Stage 0 must run on Kaggle CPU")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the exp491 exact HMM")
    started = time.perf_counter()
    counts = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    scientific_contract_sha = hashlib.sha256(
        stable_json_bytes(scientific_contract)
    ).hexdigest()
    set_num_threads(int(get_nested(config, "runtime.numba_threads_per_worker")))
    ledger = LeakageLedger(expected_wells=32)
    identity, identity_input = load_fixed32_identity(config)
    target_wells = set(identity["well"].astype(str))
    expected_rows = int(identity["suffix_rows"].sum())
    exp226, exp226_input = load_exp226_final_oof(
        config,
        target_wells,
        expected_rows,
        ledger,
    )
    raw_dir = train_data_dir(config)
    exp226_groups = exp226.groupby("well_id", sort=False).indices
    hmm = exp437_hmm_config(config)
    difference_tolerance = float(
        get_nested(
            config,
            "gates.stage_0_technical.first_difference_parity_max_abs_ft",
        )
    )
    identity_tolerance = float(
        get_nested(
            config,
            "gates.stage_0_technical.rate_increment_identity_max_abs_ft",
        )
    )
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    frozen_wells: list[FrozenWell] = []
    for well_index, row in enumerate(identity.itertuples(index=False), start=1):
        well = str(row.well)
        if well not in exp226_groups:
            raise ValueError(f"{well}: exp226 source rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            exp226=exp226.iloc[exp226_groups[well]].copy(),
            hmm=hmm,
            first_difference_tolerance_ft=difference_tolerance,
            rate_identity_tolerance_ft=identity_tolerance,
            ledger=ledger,
        )
        if len(frozen.row_idx) != int(row.suffix_rows):
            raise ValueError(f"{well}: suffix row count changed")
        if frozen.prefix_rows != int(row.prefix_rows):
            raise ValueError(f"{well}: prefix row count changed")
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(f"Stage 0 runtime hard guard exceeded: {elapsed}")
        if peak_rss_gb() > hard_rss:
            raise MemoryError(f"Stage 0 RSS hard guard exceeded: {peak_rss_gb()}")
        print(
            json.dumps(
                {
                    "event": "exp491_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "candidate_hmm_seconds": frozen.hmm_seconds,
                    "schedule_sha256": frozen.schedule_sha256,
                    "prediction_sha256": frozen.prediction_sha256,
                    "elapsed_seconds": elapsed,
                    "peak_rss_gb": peak_rss_gb(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all fixed32 candidates were frozen")

    output = artifacts_dir()
    predictions = prediction_frame(frozen_wells)
    schedule_manifest = schedule_manifest_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    schedule_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_schedule_manifest.csv",
        schedule_manifest,
    )
    if (
        prediction_artifact["logical_sha256"]
        != prediction_artifact["readback_logical_sha256"]
    ):
        raise RuntimeError("target-free prediction readback SHA mismatch")

    scope_manifest = load_scope_after_freeze(config, identity, ledger)
    scope_by_well = scope_manifest.set_index("well", drop=False)
    for frozen in frozen_wells:
        attach_scope_identity(frozen, scope_by_well.loc[frozen.well])
    persistent_wells = set(
        scope_manifest.loc[
            scope_manifest["role"].eq("persistent"), "well"
        ].astype(str)
    )
    episodes, episode_input = load_persistent_episodes_after_freeze(
        config,
        persistent_wells,
        ledger,
    )
    readout = truth_late_frame(frozen_wells, raw_dir, ledger)
    well_metrics = build_well_metrics(readout)
    episode_metrics = build_episode_metrics(episodes, readout)

    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    episode_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_episode_metrics.csv",
        episode_metrics,
    )
    elapsed = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        scope_manifest=scope_manifest,
        frozen_wells=frozen_wells,
        readout=readout,
        well_metrics=well_metrics,
        episode_metrics=episode_metrics,
        prediction_artifact=prediction_artifact,
        schedule_artifact=schedule_artifact,
        ledger=ledger,
        elapsed_seconds=elapsed,
    )
    input_manifest = {
        "fixed32_manifest": identity_input,
        "exp226_final_oof": exp226_input,
        "persistent_episodes_post_freeze": episode_input,
        "raw_train_dir": str(raw_dir),
        "scientific_contract_sha256": scientific_contract_sha,
        "leakage": {
            "exp226_rows_read_with_allowlist": (
                ledger.exp226_rows_read_with_allowlist
            ),
            "forbidden_exp226_columns_read_before_freeze": (
                ledger.forbidden_exp226_columns_read_before_freeze
            ),
            "frozen_wells": len(ledger.frozen_wells),
            "truth_rows_before_all_freeze": ledger.truth_rows_before_all_freeze,
            "role_rows_before_all_freeze": ledger.role_rows_before_all_freeze,
            "episode_rows_before_all_freeze": (
                ledger.episode_rows_before_all_freeze
            ),
            "truth_rows_after_all_freeze": ledger.truth_rows_after_all_freeze,
            "role_rows_after_all_freeze": ledger.role_rows_after_all_freeze,
            "episode_rows_after_all_freeze": (
                ledger.episode_rows_after_all_freeze
            ),
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    status = (
        "stage0_mechanism_preflight_pass_eligible_for_separate_stage1_approval"
        if gates["stage0_all_gates_pass"]
        else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "execution_contract": counts,
        "scientific_contract_sha256": scientific_contract_sha,
        "gates": gates,
        "schedule_manifest_sha256": combined_well_sha(
            frozen_wells,
            "schedule_sha256",
        ),
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
                get_nested(config, "runtime.numba_threads_per_worker")
            ),
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "schedule_manifest": schedule_artifact,
            "well_metrics": well_artifact,
            "episode_metrics": episode_artifact,
            "input_manifest": input_artifact,
        },
        "stage_1": {
            "implemented": False,
            "execution_authorized": False,
            "eligible_for_separate_approval": gates[
                "stage1_eligible_for_separate_approval"
            ],
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
        "execution_contract": counts,
        "scientific_contract_sha256": scientific_contract_sha,
        "technical_gates": gates["technical"],
        "mechanism_gates": gates["mechanism"],
        "stage0_all_gates_pass": gates["stage0_all_gates_pass"],
        "stage1_eligible_for_separate_approval": gates[
            "stage1_eligible_for_separate_approval"
        ],
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 9. Configuration preview and guarded execution
#
# The notebook always prints the one-candidate / 32-well / zero-control-rerun /
# zero-model contract. Stage 0 execution stays locked until a separate user
# approval changes every run guard. Stage 1, inference, PF, and submission are
# not implemented.

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
                "event": "exp491_stage0_preview",
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": PARENT_EXPERIMENT,
                "prediction_source": PREDICTION_SOURCE,
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
                "pf": False,
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
            "pf": False,
            "submission": False,
        }
        print(json.dumps(SUMMARY, sort_keys=True), flush=True)
