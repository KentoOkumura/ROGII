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
# # exp437 neighbor-geometry TVT-only transition HMM — Stage 0
#
# This CPU-only notebook keeps the exp435 TVT-only probability state and
# replaces only its `-delta_Z` transition center with adjacent differences of
# the fold-safe exp226 `tvt_geop` path. The fixed32 run is a mechanism
# preflight, not CV or promotion evidence. Parent/control HMMs are never rerun.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 manifest and saved target-free inputs
# 4. Geometry schedule and exact exp435 HMM inputs
# 5. TVT-only direct-transition forward-backward
# 6. Candidate decoding and target-free freeze
# 7. Truth-late mechanism readout
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

EXPERIMENT_NAME = "exp437_neighbor_geometry_tvt_only_transition_hmm"
PARENT_EXPERIMENT = "exp435_tvt_memoryless_u_rate_dzonly_hmm"
GEOMETRY_PARENT = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
)
VARIANT = "neighbor_geometry_direct_transition"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
GEOMETRY_ALLOWLIST = (
    "well_id",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "fold",
)
GEOMETRY_FORBIDDEN = frozenset(
    {"TVT", "tvt_true", "tvt_pred", "gr_delta", "error", "abs_error"}
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
        raise ValueError("wrong exp437 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp437 route must remain pf_beam")
    if not bool(get_nested(config, "design.implementation_authorized", False)):
        raise ValueError("exp437 implementation is not authorized")
    if not bool(get_nested(config, "implementation.enabled", False)):
        raise ValueError("exp437 implementation.enabled must be true")
    if get_nested(config, "execution.selected_stage") != "stage_0_fixed32":
        raise ValueError("only Stage 0 fixed32 is implemented")
    variants = tuple(get_nested(config, "model.active_scientific_variants", ()))
    if variants != (VARIANT,):
        raise ValueError("the scientific candidate must remain a single fixed variant")
    expected = {
        "scientific_variants": 1,
        "stage_0_candidate_hmm_well_runs": 32,
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


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_exp435_hmm")
    expected_hmm = {
        "step": 0.35,
        "sig_p": 0.02,
        "start_sig": 0.75,
        "band_pad": 100.0,
        "emission": "gauss",
        "lam": 1.0,
        "sigma_mode": "std",
        "position_kernel_cells": 5,
        "typewell_gr_emission": True,
        "forward_backward": True,
    }
    if hmm != expected_hmm:
        raise ValueError(f"fixed exp435 HMM contract changed: {hmm}")
    transition = get_nested(config, "model.transition")
    if transition["geometry_schedule_updated_by_emission"]:
        raise ValueError("geometry schedule cannot be updated by the emission")
    if any(
        bool(transition[key])
        for key in (
            "geometry_schedule_uses_exp226_gr_delta",
            "geometry_schedule_uses_exp226_u_projection",
            "geometry_schedule_uses_exp226_final_prediction",
        )
    ):
        raise ValueError("forbidden exp226 signals reached the transition")
    if bool(get_nested(config, "model.rate_state_present", True)):
        raise ValueError("persistent rate state is forbidden")
    if bool(get_nested(config, "model.branch_state_present", True)):
        raise ValueError("branch state is forbidden")
    if get_nested(config, "validation.truth_join") != (
        "after_schedule_prediction_and_diagnostic_sha_freeze"
    ):
        raise ValueError("truth-late contract changed")
    allowlist = tuple(
        get_nested(config, "data.exp226_geometry_oof.pre_freeze_allowlist", ())
    )
    if allowlist != GEOMETRY_ALLOWLIST:
        raise ValueError("exp226 read-time allowlist changed")
    return {
        "variant": VARIANT,
        "persistent_state": "tvt_probability_distribution_only",
        "transition_center": "adjacent_difference_of_fold_safe_exp226_tvt_geop",
        "fixed_exp435_hmm": hmm,
        "geometry_allowlist": list(allowlist),
        "rate_state_present": False,
        "branch_state_present": False,
        "parent_control_hmm_reruns": 0,
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
    raise FileNotFoundError("exp437 config.yaml was not found")


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
    geometry_rows_read_with_allowlist: int = 0
    forbidden_geometry_columns_read_before_freeze: int = 0
    truth_rows_before_all_freeze: int = 0
    role_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    role_rows_after_all_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def freeze(self, well: str) -> None:
        self.frozen_wells.add(str(well))

    def record_truth(self, rows: int) -> None:
        if not self.all_frozen:
            self.truth_rows_before_all_freeze += int(rows)
            raise RuntimeError("truth was read before all candidate artifacts froze")
        self.truth_rows_after_all_freeze += int(rows)

    def record_roles(self, rows: int) -> None:
        if not self.all_frozen:
            self.role_rows_before_all_freeze += int(rows)
            raise RuntimeError("manifest roles were read before all candidates froze")
        self.role_rows_after_all_freeze += int(rows)


# %% [markdown]
# ## 3. Fixed32 manifest and saved target-free inputs
#
# Before candidate freeze the manifest exposes only well identity and prefix /
# suffix row counts. exp226 is read with the five-column allowlist at
# `read_csv` time. The suffix truth and fixed32 role are unavailable to the
# decoder. The saved exp435 file is a comparison only and no parent HMM runs.

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


def load_fixed32_manifest(
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
    execution_manifest: pd.DataFrame,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("scope identity is unavailable before candidate freeze")
    spec = get_nested(config, "data.fixed32_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    if sha256_file(path) != str(spec["expected_sha256"]):
        raise ValueError("fixed32 manifest SHA changed after freeze")
    frame = pd.read_csv(path, dtype={"well": str})
    ledger.record_roles(len(frame))
    if frame["role"].value_counts().to_dict() != {
        "persistent": 16,
        "control": 16,
    }:
        raise ValueError("fixed32 role counts changed")
    if frame.groupby("fold").size().to_dict() != {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}:
        raise ValueError("fixed32 fold counts changed")
    if set(frame["well"]) != set(execution_manifest["well"]):
        raise ValueError("fixed32 well identity changed")
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


def load_geometry_oof(
    config: Mapping[str, Any],
    target_wells: set[str],
    expected_rows: int,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_geometry_oof")
    paths = candidate_paths(spec)
    expected_sha = str(spec["expected_decompressed_sha256"])
    matching = [path for path in paths if sha256_decompressed_csv(path) == expected_sha]
    if not matching:
        raise FileNotFoundError("SHA-matching exp226 geometry OOF was not found")
    path = matching[0]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=GEOMETRY_ALLOWLIST,
        dtype={"well_id": str},
        chunksize=200_000,
    ):
        if GEOMETRY_FORBIDDEN.intersection(chunk.columns):
            ledger.forbidden_geometry_columns_read_before_freeze += len(chunk)
            raise ValueError("forbidden exp226 columns crossed the read-time allowlist")
        selected = chunk.loc[chunk["well_id"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("exp226 geometry contains none of the fixed32 wells")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if len(frame) != expected_rows:
        raise ValueError(f"exp226 fixed32 rows={len(frame)}/{expected_rows}")
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 geometry keys are not unique")
    if not np.isfinite(frame["tvt_geop"].to_numpy(np.float64)).all():
        raise ValueError("exp226 tvt_geop must be finite")
    ledger.geometry_rows_read_with_allowlist += len(frame)
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": expected_sha,
        "rows": len(frame),
        "wells": frame["well_id"].nunique(),
        "allowlist": list(GEOMETRY_ALLOWLIST),
        "allowlist_schema_sha256": hashlib.sha256(
            stable_json_bytes(list(GEOMETRY_ALLOWLIST))
        ).hexdigest(),
        "matching_candidates": len(matching),
    }


def load_exp435_saved_control(
    config: Mapping[str, Any],
    target_wells: set[str],
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp435_stage0_predictions")
    paths = candidate_paths(spec)
    expected_logical = str(spec["expected_logical_sha256"])
    matching: list[tuple[Path, pd.DataFrame]] = []
    for path in paths:
        frame = pd.read_csv(path, float_precision="round_trip", dtype={"well": str})
        if logical_frame_sha256(frame) == expected_logical:
            matching.append((path, frame))
    if not matching:
        raise FileNotFoundError("logical-SHA-matching exp435 Stage 0 control was not found")
    path, full = matching[0]
    required = {"well", "row_idx", "dz_only_r0_prediction"}
    if not required.issubset(full.columns):
        raise ValueError("exp435 saved prediction schema changed")
    frame = full.loc[
        full["well"].isin(target_wells),
        ["well", "row_idx", "dz_only_r0_prediction"],
    ].copy()
    frame = frame.rename(columns={"dz_only_r0_prediction": "exp435_dz_only"})
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if len(frame) != expected_rows or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("exp435 fixed32 control keys changed")
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "logical_sha256": expected_logical,
        "scientific_contract_sha256": str(spec["scientific_contract_sha256"]),
        "rows": len(frame),
        "matching_candidates": len(matching),
        "regenerated": False,
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
# ## 4. Geometry schedule and exact exp435 HMM inputs
#
# The exp435 grid, start prior, Gaussian GR emission, process noise, and
# five-cell kernel are preserved. The only new array is `transition_delta`,
# built from `tvt_geop` and the exact last finite `TVT_input`.

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
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or eval_rows.empty:
        raise ValueError("expected a visible prefix and a non-empty suffix")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    gr_sigma, last_tvt = prefix_stats(horizontal, typewell_tvt, typewell_gr)
    step = float(hmm["step"])
    grid_min = max(typewell_tvt.min() - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(typewell_tvt.max() + 40.0, last_tvt + float(hmm["band_pad"]))
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = np.interp(grid, typewell_tvt, typewell_gr)
    raw_gr = eval_rows["GR"].to_numpy(np.float64)
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[eval_rows.index]
    )
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    last = known.iloc[-1]
    return {
        "emission_ll": emission_ll,
        "grid": grid,
        "start_p": float((last_tvt - grid_min) / step),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "last_known_z": float(last["Z"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
    }


def build_geometry_schedule(
    geometry: pd.DataFrame,
    *,
    expected_row_idx: np.ndarray,
    last_known_tvt: float,
) -> dict[str, Any]:
    required = set(GEOMETRY_ALLOWLIST)
    if not required.issubset(geometry.columns):
        raise ValueError("geometry allowlist schema is incomplete")
    if GEOMETRY_FORBIDDEN.intersection(geometry.columns):
        raise ValueError("forbidden columns reached geometry schedule construction")
    ordered = geometry.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = ordered["row_idx"].to_numpy(np.int64)
    suffix_offset = ordered["suffix_offset"].to_numpy(np.int64)
    path = ordered["tvt_geop"].to_numpy(np.float64)
    if not np.array_equal(row_idx, np.asarray(expected_row_idx, dtype=np.int64)):
        raise ValueError("geometry row_idx does not align with raw suffix")
    if not np.array_equal(suffix_offset, np.arange(len(path), dtype=np.int64)):
        raise ValueError("geometry suffix_offset is not contiguous")
    if len(row_idx) > 1 and not np.all(np.diff(row_idx) == 1):
        raise ValueError("geometry row_idx is not contiguous")
    if not np.isfinite(path).all() or not math.isfinite(float(last_known_tvt)):
        raise ValueError("geometry schedule inputs must be finite")
    transition_delta = np.diff(
        np.concatenate([[float(last_known_tvt)], path])
    )
    explicit = np.empty_like(path)
    explicit[0] = path[0] - float(last_known_tvt)
    explicit[1:] = path[1:] - path[:-1]
    parity = float(np.max(np.abs(transition_delta - explicit)))
    return {
        "row_idx": row_idx,
        "tvt_geop": path,
        "transition_delta": transition_delta,
        "first_difference_parity_max_abs_ft": parity,
        "logical_sha256": array_bundle_sha256(
            row_idx=row_idx,
            tvt_geop=path,
            transition_delta=transition_delta,
        ),
    }


# %% [markdown]
# ## 5. TVT-only direct-transition forward-backward
#
# The persistent state is one probability vector over the fixed TVT grid.
# Each row uses one five-cell position kernel centered at the frozen geometry
# increment. There is no rate support, rate responsibility, or branch state.

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
    result = _direct_transition_forward_backward(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        transition_delta,
        float(hmm["step"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(hmm["lam"]),
    )
    (
        posterior,
        log_likelihood,
        transition_error,
        forward_error,
        posterior_error,
    ) = result
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
        "persistent_state_shape": (len(transition_delta), len(grid)),
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
# Each fixed32 well is decoded exactly once. The absolute exp226 path and saved
# exp435 dz-only prediction are comparisons; they are not blended into the
# candidate. All schedule, prediction, and diagnostic SHAs freeze before role,
# fold, suffix truth, or error is attached.

# %%
@dataclass
class FrozenWell:
    well: str
    row_idx: np.ndarray
    suffix_offset: np.ndarray
    source_fold: np.ndarray
    geometry_prediction: np.ndarray
    geometry_delta: np.ndarray
    exp435_dz_only: np.ndarray
    candidate_prediction: np.ndarray
    posterior_std: np.ndarray
    raw_gr_missing: np.ndarray
    last_known_tvt: float
    prefix_rows: int
    schedule_sha256: str
    prediction_sha256: str
    diagnostic_sha256: str
    first_difference_parity_max_abs_ft: float
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
    geometry: pd.DataFrame,
    saved_exp435: pd.DataFrame,
    hmm: Mapping[str, Any],
    parity_tolerance_ft: float,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir)
    prepared = prepare_hmm_inputs(horizontal, typewell, hmm)
    schedule = build_geometry_schedule(
        geometry,
        expected_row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        last_known_tvt=float(prepared["last_known_tvt"]),
    )
    if schedule["first_difference_parity_max_abs_ft"] > parity_tolerance_ft:
        raise RuntimeError(f"{well}: geometry first-difference parity failed")
    control = saved_exp435.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(
        control["row_idx"].to_numpy(np.int64),
        schedule["row_idx"],
    ):
        raise ValueError(f"{well}: exp435 control row index does not align")
    result = run_direct_transition_hmm(
        prepared,
        schedule["transition_delta"],
        hmm,
    )
    ledger.freeze(well)
    ordered_geometry = geometry.sort_values("row_idx", kind="mergesort")
    return FrozenWell(
        well=str(well),
        row_idx=np.asarray(schedule["row_idx"], dtype=np.int64),
        suffix_offset=ordered_geometry["suffix_offset"].to_numpy(np.int64),
        source_fold=ordered_geometry["fold"].to_numpy(np.int64),
        geometry_prediction=np.asarray(schedule["tvt_geop"], dtype=np.float64),
        geometry_delta=np.asarray(schedule["transition_delta"], dtype=np.float64),
        exp435_dz_only=control["exp435_dz_only"].to_numpy(np.float64),
        candidate_prediction=np.asarray(result["posterior_mean"], dtype=np.float64),
        posterior_std=np.asarray(result["posterior_std"], dtype=np.float64),
        raw_gr_missing=np.asarray(prepared["raw_gr_missing"], dtype=bool),
        last_known_tvt=float(prepared["last_known_tvt"]),
        prefix_rows=int(prepared["prefix_rows"]),
        schedule_sha256=str(schedule["logical_sha256"]),
        prediction_sha256=str(result["prediction_sha256"]),
        diagnostic_sha256=str(result["diagnostic_sha256"]),
        first_difference_parity_max_abs_ft=float(
            schedule["first_difference_parity_max_abs_ft"]
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
                    "tvt_geop": item.geometry_prediction,
                    "geometry_transition_delta": item.geometry_delta,
                    "exp435_dz_only_prediction": item.exp435_dz_only,
                    "candidate_prediction": item.candidate_prediction,
                    "candidate_posterior_std": item.posterior_std,
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
# ## 7. Truth-late mechanism readout

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


def well_truth_late_metrics(
    frozen: FrozenWell,
    actual: np.ndarray,
) -> dict[str, Any]:
    candidate_error = frozen.candidate_prediction - actual
    geometry_error = frozen.geometry_prediction - actual
    dz_error = frozen.exp435_dz_only - actual
    rows = len(actual)
    candidate_sse = float(np.sum(candidate_error**2))
    geometry_sse = float(np.sum(geometry_error**2))
    dz_sse = float(np.sum(dz_error**2))
    candidate_rmse = math.sqrt(candidate_sse / rows)
    geometry_rmse = math.sqrt(geometry_sse / rows)
    dz_rmse = math.sqrt(dz_sse / rows)
    return {
        "well": frozen.well,
        "role": frozen.role,
        "fold": frozen.fold,
        "rows": rows,
        "candidate_sse": candidate_sse,
        "exp226_geometry_sse": geometry_sse,
        "exp435_dz_only_sse": dz_sse,
        "candidate_rmse_ft": candidate_rmse,
        "exp226_geometry_rmse_ft": geometry_rmse,
        "exp435_dz_only_rmse_ft": dz_rmse,
        "candidate_delta_vs_exp226_geometry_ft": candidate_rmse - geometry_rmse,
        "candidate_delta_vs_exp435_dz_only_ft": candidate_rmse - dz_rmse,
        "candidate_improved_vs_exp226_geometry": candidate_rmse < geometry_rmse,
        "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
        "prediction_sha256": frozen.prediction_sha256,
        "diagnostic_sha256": frozen.diagnostic_sha256,
        "schedule_sha256": frozen.schedule_sha256,
        "hmm_seconds": frozen.hmm_seconds,
    }


def pooled_rmse(
    frame: pd.DataFrame,
    sse_column: str,
) -> float:
    return float(math.sqrt(frame[sse_column].sum() / frame["rows"].sum()))


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    scope_manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    well_metrics: pd.DataFrame,
    prediction_artifact: Mapping[str, Any],
    schedule_artifact: Mapping[str, Any],
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "gates.stage_0_technical")
    mechanism_config = get_nested(config, "gates.stage_0_mechanism")
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    max_transition_error = max(
        item.transition_row_sum_max_error for item in frozen_wells
    )
    max_posterior_error = max(
        item.posterior_normalization_max_error for item in frozen_wells
    )
    max_schedule_parity = max(
        item.first_difference_parity_max_abs_ft for item in frozen_wells
    )
    finite_values = sum(
        int(
            np.isfinite(item.candidate_prediction).sum()
            + np.isfinite(item.posterior_std).sum()
            + np.isfinite(item.geometry_delta).sum()
        )
        for item in frozen_wells
    )
    total_values = total_rows * 3
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
        "duplicate_rows": not prediction_frame(frozen_wells).duplicated(
            ["well", "row_idx"]
        ).any(),
        "missing_rows": len(prediction_frame(frozen_wells)) == total_rows,
        "finite_schedule_prediction_coverage": (
            finite_values / total_values
            >= float(
                technical_config["finite_schedule_prediction_coverage_min"]
            )
        ),
        "forbidden_geometry_columns_before_freeze": (
            ledger.forbidden_geometry_columns_read_before_freeze
            <= int(technical_config["forbidden_column_reads_before_freeze_max"])
        ),
        "truth_reads_before_freeze": ledger.truth_rows_before_all_freeze
        <= int(technical_config["truth_role_episode_reads_before_freeze_max"]),
        "role_reads_before_freeze": ledger.role_rows_before_all_freeze
        <= int(technical_config["truth_role_episode_reads_before_freeze_max"]),
        "schedule_first_difference_parity": max_schedule_parity
        <= float(
            technical_config["schedule_first_difference_parity_max_abs_ft"]
        ),
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
    all_geometry = pooled_rmse(well_metrics, "exp226_geometry_sse")
    matched = well_metrics.loc[well_metrics["role"].eq("control")]
    persistent = well_metrics.loc[well_metrics["role"].eq("persistent")]
    matched_candidate = pooled_rmse(matched, "candidate_sse")
    matched_geometry = pooled_rmse(matched, "exp226_geometry_sse")
    matched_dz = pooled_rmse(matched, "exp435_dz_only_sse")
    persistent_candidate = pooled_rmse(persistent, "candidate_sse")
    persistent_geometry = pooled_rmse(persistent, "exp226_geometry_sse")
    fold_rows: list[dict[str, Any]] = []
    for fold, part in well_metrics.groupby("fold", sort=True):
        candidate_rmse = pooled_rmse(part, "candidate_sse")
        geometry_rmse = pooled_rmse(part, "exp226_geometry_sse")
        fold_rows.append(
            {
                "fold": int(fold),
                "candidate_rmse_ft": candidate_rmse,
                "exp226_geometry_rmse_ft": geometry_rmse,
                "delta_ft": candidate_rmse - geometry_rmse,
                "improved": candidate_rmse < geometry_rmse,
            }
        )
    improving_folds = sum(row["improved"] for row in fold_rows)
    paired_delta = well_metrics[
        "candidate_delta_vs_exp226_geometry_ft"
    ].to_numpy(np.float64)
    paired_p95 = float(np.quantile(paired_delta, 0.95))
    paired_worst = float(np.max(paired_delta))
    mechanism = {
        "gain_vs_exp226_geometry_all32": all_geometry - all_candidate
        >= float(
            mechanism_config["gain_vs_exp226_geometry_all32_min_ft"]
        ),
        "matched_control_delta_vs_exp226_geometry": (
            matched_candidate - matched_geometry
            <= float(
                mechanism_config[
                    "matched_control_delta_vs_exp226_geometry_max_ft"
                ]
            )
        ),
        "persistent_gain_vs_exp226_geometry": (
            persistent_geometry - persistent_candidate
            >= float(
                mechanism_config[
                    "persistent_gain_vs_exp226_geometry_min_ft"
                ]
            )
        ),
        "matched_control_gain_vs_exp435_dz_only": (
            matched_dz - matched_candidate
            >= float(
                mechanism_config[
                    "matched_control_gain_vs_exp435_dz_only_min_ft"
                ]
            )
        ),
        "improving_folds_vs_exp226_geometry": improving_folds
        >= int(mechanism_config["improving_folds_vs_exp226_geometry_min"]),
        "paired_by_well_delta_p95": paired_p95
        <= float(mechanism_config["paired_by_well_delta_p95_max_ft"]),
        "worst_well_delta": paired_worst
        <= float(mechanism_config["worst_well_delta_max_ft"]),
    }
    all_gates_pass = bool(all(technical.values()) and all(mechanism.values()))
    diagnostics = {
        "candidate_all32_rmse_ft": all_candidate,
        "exp226_geometry_all32_rmse_ft": all_geometry,
        "gain_vs_exp226_geometry_all32_ft": all_geometry - all_candidate,
        "candidate_matched_control_rmse_ft": matched_candidate,
        "exp226_geometry_matched_control_rmse_ft": matched_geometry,
        "exp435_dz_only_matched_control_rmse_ft": matched_dz,
        "candidate_persistent_rmse_ft": persistent_candidate,
        "exp226_geometry_persistent_rmse_ft": persistent_geometry,
        "improving_folds_vs_exp226_geometry": improving_folds,
        "fold_metrics": fold_rows,
        "paired_by_well_delta_p95_ft": paired_p95,
        "worst_well_delta_ft": paired_worst,
        "maximum_transition_row_sum_error": max_transition_error,
        "maximum_posterior_normalization_error": max_posterior_error,
        "maximum_schedule_first_difference_parity_abs_ft": max_schedule_parity,
        "source_manifest_fold_match": source_fold_match,
        "candidate_hmm_seconds": candidate_seconds,
        "stage0_elapsed_seconds": float(elapsed_seconds),
        "projected_full_runtime_seconds": projected_full_runtime,
        "peak_rss_gb": peak_rss_gb(),
        "fixed32_is_mechanism_only_not_cv_or_promotion": True,
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


# %% [markdown]
# ## 8. Stage 0 gates, generated artifacts, and metrics

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP437_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp437 Stage 0 must run on Kaggle CPU")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
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
    manifest, manifest_input = load_fixed32_manifest(config)
    target_wells = set(manifest["well"].astype(str))
    expected_rows = int(manifest["suffix_rows"].sum())
    geometry, geometry_input = load_geometry_oof(
        config,
        target_wells,
        expected_rows,
        ledger,
    )
    exp435_control, exp435_input = load_exp435_saved_control(
        config,
        target_wells,
        expected_rows,
    )
    raw_dir = train_data_dir(config)
    geometry_groups = geometry.groupby("well_id", sort=False).indices
    control_groups = exp435_control.groupby("well", sort=False).indices
    hmm = get_nested(config, "model.fixed_exp435_hmm")
    parity_tolerance = float(
        get_nested(
            config,
            "gates.stage_0_technical.schedule_first_difference_parity_max_abs_ft",
        )
    )
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    frozen_wells: list[FrozenWell] = []
    for well_index, row in enumerate(manifest.itertuples(index=False), start=1):
        well = str(row.well)
        if well not in geometry_groups or well not in control_groups:
            raise ValueError(f"{well}: saved input rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            geometry=geometry.iloc[geometry_groups[well]].copy(),
            saved_exp435=exp435_control.iloc[control_groups[well]].copy(),
            hmm=hmm,
            parity_tolerance_ft=parity_tolerance,
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
                    "event": "exp437_stage0_progress",
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

    scope_manifest = load_scope_after_freeze(config, manifest, ledger)
    scope_by_well = scope_manifest.set_index("well", drop=False)
    for frozen in frozen_wells:
        attach_scope_identity(frozen, scope_by_well.loc[frozen.well])

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

    well_rows: list[dict[str, Any]] = []
    for frozen in frozen_wells:
        actual = load_truth_after_freeze(frozen, raw_dir, ledger)
        well_rows.append(well_truth_late_metrics(frozen, actual))
    well_metrics = pd.DataFrame(well_rows).sort_values(
        ["fold", "role", "well"],
        kind="mergesort",
    )
    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    elapsed = float(time.perf_counter() - started)
    gates = evaluate_stage0_gates(
        config=config,
        scope_manifest=scope_manifest,
        frozen_wells=frozen_wells,
        well_metrics=well_metrics,
        prediction_artifact=prediction_artifact,
        schedule_artifact=schedule_artifact,
        ledger=ledger,
        elapsed_seconds=elapsed,
    )
    input_manifest = {
        "fixed32_manifest": manifest_input,
        "exp226_geometry_oof": geometry_input,
        "exp435_saved_control": exp435_input,
        "raw_train_dir": str(raw_dir),
        "scientific_contract_sha256": scientific_contract_sha,
        "leakage": {
            "geometry_rows_read_with_allowlist": (
                ledger.geometry_rows_read_with_allowlist
            ),
            "forbidden_geometry_columns_read_before_freeze": (
                ledger.forbidden_geometry_columns_read_before_freeze
            ),
            "frozen_wells": len(ledger.frozen_wells),
            "truth_rows_before_all_freeze": ledger.truth_rows_before_all_freeze,
            "role_rows_before_all_freeze": ledger.role_rows_before_all_freeze,
            "truth_rows_after_all_freeze": ledger.truth_rows_after_all_freeze,
            "role_rows_after_all_freeze": ledger.role_rows_after_all_freeze,
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
# The notebook always prints the one-candidate / 32-well / zero-parent-rerun /
# zero-model contract. Stage 0 remains execution-locked until a separate user
# approval changes all three run guards. Stage 1, inference, and submission are
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
                "event": "exp437_stage0_preview",
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
