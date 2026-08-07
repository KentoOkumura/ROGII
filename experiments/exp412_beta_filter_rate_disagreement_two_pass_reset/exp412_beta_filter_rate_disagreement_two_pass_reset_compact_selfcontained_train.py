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
# # exp412 beta-filter rate disagreement two-pass reset — Stage 0
#
# This CPU-only notebook runs an unchanged exp209 first pass, freezes rows where
# the smoothed-versus-filtered rate disagreement is strong and persistent, and
# runs a second pass that transfers 10% of stay mass to the indicated adjacent
# rate state only on those rows. Stage 0 evaluates the preregistered 8 backward,
# 8 forward, and 16 matched-control wells. Stage 0 execution is explicitly
# authorized; Stage 1, inference, and submission remain separately disabled.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 manifest, saved parent, and target-free raw inputs
# 4. Exact exp209 HMM input preparation
# 5. Frozen-schedule two-pass forward/backward kernel
# 6. Parent parity, trigger freeze, and target-free prediction freeze
# 7. Truth-late direction, cause, and safety readout
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from numba import njit, prange, set_num_threads

EXPERIMENT_NAME = "exp412_beta_filter_rate_disagreement_two_pass_reset"
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
        raise ValueError("wrong exp412 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp412 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp412 scientific parent changed")
    if get_nested(config, "lineage.evidence_parent") != EVIDENCE_EXPERIMENT:
        raise ValueError("exp412 evidence parent changed")
    if not bool(get_nested(config, "design.implementation_enabled", False)):
        raise RuntimeError("exp412 implementation is not enabled")
    if bool(get_nested(config, "design.stage_1_execution_approved", True)):
        raise ValueError("Stage 1 must remain disabled during Stage 0")
    if bool(get_nested(config, "design.inference_enabled", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "design.submission_enabled", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.enable_gpu", True)):
        raise ValueError("exp412 is CPU-only")

    expected = {
        "active_treatment_variants": 1,
        "stage_0_baseline_hmm_well_runs": 32,
        "stage_0_treatment_hmm_well_runs": 32,
        "stage_0_total_hmm_well_runs": 64,
        "parent_control_hmm_reruns": 32,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1)) for key in expected
    }
    if observed != expected:
        raise ValueError(f"Stage 0 execution contract changed: {observed} != {expected}")
    if not bool(
        get_nested(config, "execution.parent_control_regeneration_stage_0", False)
    ):
        raise ValueError("exp412 Stage 0 must regenerate the exp209 parent control")
    if get_nested(config, "execution.run_stage") != "stage_0_fixed32":
        raise ValueError("run_stage must remain stage_0_fixed32")
    if require_run_authorization:
        if not bool(get_nested(config, "design.stage_0_execution_approved", False)):
            raise RuntimeError("Stage 0 execution is not approved")
        if not bool(get_nested(config, "execution.kaggle_execution_authorized", False)):
            raise RuntimeError(
                "implementation approval does not authorize Kaggle execution"
            )
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = get_nested(config, "model.parent_hmm")
    expected_parent = {
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
    if parent != expected_parent:
        raise ValueError(f"exp209 HMM contract changed: {parent} != {expected_parent}")
    trigger = get_nested(config, "model.trigger")
    expected_trigger = {
        "statistic": "standardized_smoothed_minus_filtered_rate",
        "denominator_floor": 0.005,
        "absolute_z_threshold": 2.0,
        "rolling_window_rows": 16,
        "qualifying_rows_min": 8,
        "same_sign_fraction_min": 0.75,
        "tie_policy": "inactive",
        "freeze_before_truth": True,
        "recompute_from_treatment": False,
    }
    if trigger != expected_trigger:
        raise ValueError(f"trigger contract changed: {trigger} != {expected_trigger}")
    treatment = get_nested(config, "model.treatment")
    expected_treatment = {
        "stay_mass_transfer_fraction": 0.10,
        "target": "signed_adjacent_rate_state",
        "add_rate_states": False,
        "edge_policy": "no_op_for_outward_source_state",
        "freeze_schedule_before_backward": True,
        "first_affected_transition": "transition_entering_active_row",
    }
    if treatment != expected_treatment:
        raise ValueError(
            f"directional transition contract changed: {treatment} != {expected_treatment}"
        )
    return {
        "parent_hmm": parent,
        "trigger": trigger,
        "treatment": treatment,
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
    raise FileNotFoundError("exp412 config.yaml was not found")


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
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.stage_0_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed: {observed}")
    frame = pd.read_csv(
        path,
        dtype={"well": str, "matched_cause_well": str},
        keep_default_na=False,
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
    if frame["role"].value_counts().to_dict() != {
        "backward_cause": 8,
        "forward_cause": 8,
        "control": 16,
    }:
        raise ValueError("fixed32 role counts changed")
    if set(frame["fold"].astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("fixed32 must cover all five folds")
    for role in ("backward_cause", "forward_cause"):
        if set(frame.loc[frame["role"].eq(role), "fold"].astype(int)) != set(range(5)):
            raise ValueError(f"{role} must cover all five folds")
    ledger.record_scope(len(frame))
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "logical_sha256": logical_frame_sha256(frame),
    }


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


def saved_float32_parity_max_abs_diff(
    observed: np.ndarray,
    saved: np.ndarray,
) -> float:
    """Compare predictions in the float32 representation persisted by exp209."""
    observed32 = np.asarray(observed, dtype=np.float32)
    saved32 = np.asarray(saved, dtype=np.float32)
    if observed32.shape != saved32.shape:
        raise ValueError("saved exp209 parity shapes differ")
    if not (np.isfinite(observed32).all() and np.isfinite(saved32).all()):
        raise ValueError("saved exp209 parity inputs must be finite")
    return float(
        np.max(
            np.abs(
                observed32.astype(np.float64)
                - saved32.astype(np.float64)
            )
        )
    )


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
# ## 4. Exact exp209 HMM input preparation

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
# ## 5. Frozen-schedule two-pass forward/backward kernel
#
# `frozen_direction[t]` belongs to the transition entering suffix row `t`.
# The array is created only from the unchanged first-pass filtered and smoothed
# rate moments. The second pass consumes that same array in both directions and
# never recomputes it from treatment messages.

# %%
@njit(cache=True, nogil=True)
def rate_kernel_probabilities(
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    mom: float,
    direction: int,
    stay_transfer: float,
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
        p_stay = 1.0 - p_plus - p_minus
        if direction > 0 and r_i < r_count - 1:
            moved = stay_transfer * p_stay
            p_plus += moved
            p_stay -= moved
        elif direction < 0 and r_i > 0:
            moved = stay_transfer * p_stay
            p_minus += moved
            p_stay -= moved
        kernel[r_i, 0] = p_minus
        kernel[r_i, 1] = p_stay
        kernel[r_i, 2] = p_plus
    return kernel


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_frozen_rate_schedule(
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
    frozen_direction,
    stay_transfer,
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
    filtered_rate_second_moment = np.empty(t_count, np.float64)
    maximum_forward_normalization_error = 0.0

    for t_i in range(t_count):
        kernel = rate_kernel_probabilities(
            rates,
            dm[t_i],
            sig_r,
            mom,
            int(frozen_direction[t_i]),
            stay_transfer,
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
                        total += np.exp(
                            prev[p_i, r_i]
                            + rate_log_kernel[r_i, r2 - r_i + 1]
                            - best
                        )
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
            log_norm = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
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
                            total += np.exp(
                                tmp[p1, r2] + position_log_kernel[k_i] - best
                            )
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
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_probability = np.exp(
                    predictive[p_i, r_i] - predictive_best
                )
                filtered_probability = np.exp(cur[p_i, r_i] - filtered_best)
                predictive_total += predictive_probability
                filtered_total += filtered_probability
                predictive_r1 += predictive_probability * rates[r_i]
                filtered_r1 += filtered_probability * rates[r_i]
                filtered_r2 += filtered_probability * rates[r_i] * rates[r_i]
        predictive_rate_mean[t_i] = predictive_r1 / predictive_total
        filtered_rate_mean[t_i] = filtered_r1 / filtered_total
        filtered_rate_second_moment[t_i] = filtered_r2 / filtered_total
        predictive_check = 0.0
        filtered_check = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_check += (
                    np.exp(predictive[p_i, r_i] - predictive_best)
                    / predictive_total
                )
                filtered_check += (
                    np.exp(cur[p_i, r_i] - filtered_best) / filtered_total
                )
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
            alpha[t_count - 1, p_i, r_i] = np.float32(
                np.exp(values[p_i, r_i] - best) / total
            )

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        kernel = rate_kernel_probabilities(
            rates,
            dm[t_i],
            sig_r,
            mom,
            int(frozen_direction[t_i]),
            stay_transfer,
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
            log_norm = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        value = (
                            position_log_kernel[k_i]
                            + lam * em[t_i, p2]
                            + beta_next[p2, r2]
                        )
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
                    value = (
                        rate_log_kernel[r_i, r2 - r_i + 1]
                        + beta_tmp[p_i, r2]
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1]
                            + beta_tmp[p_i, r2]
                            - best
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
                alpha[t_i - 1, p_i, r_i] = np.float32(
                    np.exp(values[p_i, r_i] - best) / total
                )
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]

    maximum_posterior_normalization_error = 0.0
    smoothed_rate_mean = np.zeros(t_count, np.float64)
    for t_i in range(t_count):
        row_total = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                probability = float(alpha[t_i, p_i, r_i])
                row_total += probability
                smoothed_rate_mean[t_i] += probability * rates[r_i]
        maximum_posterior_normalization_error = max(
            maximum_posterior_normalization_error, abs(row_total - 1.0)
        )
        if row_total > 0.0:
            smoothed_rate_mean[t_i] /= row_total
    filtered_rate_variance = np.maximum(
        filtered_rate_second_moment - filtered_rate_mean * filtered_rate_mean,
        0.0,
    )
    return (
        post_p,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        np.sqrt(filtered_rate_variance),
        smoothed_rate_mean,
        maximum_forward_normalization_error,
        maximum_posterior_normalization_error,
    )


def run_hmm_pass(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    frozen_direction: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    direction = np.asarray(frozen_direction, dtype=np.int8)
    if len(direction) != len(prepared["eval_index"]):
        raise ValueError("frozen direction length does not match suffix rows")
    if not set(np.unique(direction)).issubset({-1, 0, 1}):
        raise ValueError("frozen direction must contain only -1, 0, or 1")
    result = _hmm2_frozen_rate_schedule(
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
        direction,
        float(treatment["stay_mass_transfer_fraction"]),
    )
    (
        post_p,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        filtered_rate_std,
        smoothed_rate_mean,
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
    message_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        predictive_rate_mean=np.asarray(predictive_rate_mean, dtype=np.float64),
        filtered_rate_mean=np.asarray(filtered_rate_mean, dtype=np.float64),
        filtered_rate_std=np.asarray(filtered_rate_std, dtype=np.float64),
        smoothed_rate_mean=np.asarray(smoothed_rate_mean, dtype=np.float64),
    )
    return {
        "posterior_mean": posterior_mean,
        "log_likelihood": float(log_likelihood),
        "predictive_rate_mean": predictive_rate_mean,
        "filtered_rate_mean": filtered_rate_mean,
        "filtered_rate_std": filtered_rate_std,
        "smoothed_rate_mean": smoothed_rate_mean,
        "frozen_direction": direction,
        "maximum_normalization_error": max(
            float(forward_normalization_error),
            float(posterior_normalization_error),
        ),
        "message_sha256": message_sha,
        "prediction_sha256": prediction_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def beta_filter_activation_schedule(
    filtered_rate_mean: np.ndarray,
    filtered_rate_std: np.ndarray,
    smoothed_rate_mean: np.ndarray,
    trigger: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Freeze the preregistered rolling beta-filter disagreement schedule."""
    filtered = np.asarray(filtered_rate_mean, dtype=np.float64)
    filtered_std = np.asarray(filtered_rate_std, dtype=np.float64)
    smoothed = np.asarray(smoothed_rate_mean, dtype=np.float64)
    if not (len(filtered) == len(filtered_std) == len(smoothed)):
        raise ValueError("first-pass rate moment lengths differ")
    denominator = np.maximum(
        filtered_std,
        float(trigger["denominator_floor"]),
    )
    z_beta = (smoothed - filtered) / denominator
    threshold = float(trigger["absolute_z_threshold"])
    window = int(trigger["rolling_window_rows"])
    required = int(trigger["qualifying_rows_min"])
    same_sign_min = float(trigger["same_sign_fraction_min"])
    active_direction = np.zeros(len(z_beta), dtype=np.int8)
    qualifying_count = np.zeros(len(z_beta), dtype=np.int16)
    majority_fraction = np.zeros(len(z_beta), dtype=np.float64)
    for row in range(len(z_beta)):
        start = max(0, row - window + 1)
        values = z_beta[start : row + 1]
        qualifying = values[np.abs(values) >= threshold]
        qualifying_count[row] = len(qualifying)
        if len(qualifying) < required:
            continue
        positive = int(np.count_nonzero(qualifying > 0.0))
        negative = int(np.count_nonzero(qualifying < 0.0))
        fraction = max(positive, negative) / len(qualifying)
        majority_fraction[row] = fraction
        if fraction < same_sign_min or positive == negative:
            continue
        active_direction[row] = np.int8(1 if positive > negative else -1)
    return {
        "z_beta": z_beta,
        "qualifying_count": qualifying_count,
        "majority_fraction": majority_fraction,
        "active_direction": active_direction,
    }


# %% [markdown]
# ## 6. Parent parity, trigger freeze, and target-free prediction freeze

# %%
def synthetic_zero_schedule_parity(
    hmm: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    rows = 12
    positions = 19
    grid = np.arange(positions, dtype=np.float64) * float(hmm["step"]) + 11_900.0
    rates = np.linspace(-0.10, 0.10, int(hmm["n_rates"]), dtype=np.float64)
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.35 * math.sin(row / 3.0)) / 0.42) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    prepared = {
        "emission_ll": emission,
        "dm": np.linspace(9.0, 21.0, rows, dtype=np.float64),
        "dz": np.linspace(-0.4, 0.7, rows, dtype=np.float64),
        "grid": grid,
        "rates": rates,
        "start_p": 8.5,
        "r0": 0.0,
        "eval_index": np.arange(rows, dtype=np.int64),
    }
    zero_schedule = np.zeros(rows, dtype=np.int8)
    parent = run_hmm_pass(
        prepared,
        hmm,
        {**treatment, "stay_mass_transfer_fraction": 0.0},
        frozen_direction=zero_schedule,
    )
    no_active = run_hmm_pass(
        prepared,
        hmm,
        treatment,
        frozen_direction=zero_schedule,
    )
    posterior_diff = float(
        np.max(np.abs(parent["posterior_mean"] - no_active["posterior_mean"]))
    )
    log_likelihood_diff = abs(
        float(parent["log_likelihood"]) - float(no_active["log_likelihood"])
    )
    return {
        "posterior_mean_max_abs_diff_ft": posterior_diff,
        "log_likelihood_abs_diff": log_likelihood_diff,
        "active_rows": 0,
        "parent_prediction_sha256": parent["prediction_sha256"],
        "zero_schedule_prediction_sha256": no_active["prediction_sha256"],
        "pass": bool(
            posterior_diff <= 1.0e-10
            and log_likelihood_diff <= 1.0e-10
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
    predictive_rate_mean: np.ndarray
    filtered_rate_mean: np.ndarray
    filtered_rate_std: np.ndarray
    smoothed_rate_mean: np.ndarray
    z_beta: np.ndarray
    qualifying_count: np.ndarray
    majority_fraction: np.ndarray
    active_direction: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    schedule_sha256: str
    baseline_message_sha256: str
    baseline_prediction_sha256: str
    treatment_prediction_sha256: str
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
    hmm: Mapping[str, Any],
    trigger: Mapping[str, Any],
    treatment: Mapping[str, Any],
    parent_parity_tolerance_ft: float,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, hmm)
    zero_schedule = np.zeros(len(prepared["eval_index"]), dtype=np.int8)
    baseline = run_hmm_pass(
        prepared,
        hmm,
        treatment,
        frozen_direction=zero_schedule,
    )
    parent = saved_parent.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    eval_id = parent_cache_ids_for_rows(well, row_idx)
    if not np.array_equal(parent["row_idx"].to_numpy(np.int64), row_idx):
        raise ValueError(f"{well}: parent row index does not align with raw suffix")
    if not np.array_equal(parent["id"].astype(str).to_numpy(), eval_id):
        raise ValueError(f"{well}: parent id does not align with raw suffix")
    parent_prediction = parent["parent_prediction"].to_numpy(np.float64)
    baseline_prediction = np.asarray(baseline["posterior_mean"], dtype=np.float64)
    baseline_saved_parent_max_abs_diff = saved_float32_parity_max_abs_diff(
        baseline_prediction,
        parent_prediction,
    )
    if baseline_saved_parent_max_abs_diff > float(parent_parity_tolerance_ft):
        raise RuntimeError(
            f"{well}: baseline exp209 parity failed before treatment: "
            f"{baseline_saved_parent_max_abs_diff}"
        )
    schedule = beta_filter_activation_schedule(
        baseline["filtered_rate_mean"],
        baseline["filtered_rate_std"],
        baseline["smoothed_rate_mean"],
        trigger,
    )
    active_direction = np.asarray(schedule["active_direction"], dtype=np.int8)
    schedule_sha = array_bundle_sha256(
        row_idx=row_idx,
        filtered_rate_mean=np.asarray(
            baseline["filtered_rate_mean"], dtype=np.float64
        ),
        filtered_rate_std=np.asarray(
            baseline["filtered_rate_std"], dtype=np.float64
        ),
        smoothed_rate_mean=np.asarray(
            baseline["smoothed_rate_mean"], dtype=np.float64
        ),
        z_beta=np.asarray(schedule["z_beta"], dtype=np.float64),
        qualifying_count=np.asarray(schedule["qualifying_count"], dtype=np.int16),
        majority_fraction=np.asarray(schedule["majority_fraction"], dtype=np.float64),
        active_direction=active_direction,
    )
    treatment_result = run_hmm_pass(
        prepared,
        hmm,
        treatment,
        frozen_direction=active_direction,
    )
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
        treatment_prediction=np.asarray(
            treatment_result["posterior_mean"], dtype=np.float64
        ),
        predictive_rate_mean=np.asarray(
            baseline["predictive_rate_mean"], dtype=np.float64
        ),
        filtered_rate_mean=np.asarray(
            baseline["filtered_rate_mean"], dtype=np.float64
        ),
        filtered_rate_std=np.asarray(
            baseline["filtered_rate_std"], dtype=np.float64
        ),
        smoothed_rate_mean=np.asarray(
            baseline["smoothed_rate_mean"], dtype=np.float64
        ),
        z_beta=np.asarray(schedule["z_beta"], dtype=np.float64),
        qualifying_count=np.asarray(schedule["qualifying_count"], dtype=np.int16),
        majority_fraction=np.asarray(
            schedule["majority_fraction"], dtype=np.float64
        ),
        active_direction=active_direction,
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        schedule_sha256=schedule_sha,
        baseline_message_sha256=str(baseline["message_sha256"]),
        baseline_prediction_sha256=str(baseline["prediction_sha256"]),
        treatment_prediction_sha256=str(treatment_result["prediction_sha256"]),
        baseline_saved_parent_max_abs_diff_ft=baseline_saved_parent_max_abs_diff,
        maximum_normalization_error=max(
            float(baseline["maximum_normalization_error"]),
            float(treatment_result["maximum_normalization_error"]),
        ),
        baseline_log_likelihood=float(baseline["log_likelihood"]),
        treatment_log_likelihood=float(treatment_result["log_likelihood"]),
        baseline_elapsed_seconds=float(baseline["elapsed_seconds"]),
        treatment_elapsed_seconds=float(treatment_result["elapsed_seconds"]),
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


def schedule_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "suffix_offset": np.arange(len(item.row_idx), dtype=np.int64),
                    "predictive_rate_mean": item.predictive_rate_mean,
                    "filtered_rate_mean": item.filtered_rate_mean,
                    "filtered_rate_std": item.filtered_rate_std,
                    "smoothed_rate_mean": item.smoothed_rate_mean,
                    "z_beta": item.z_beta,
                    "qualifying_count": item.qualifying_count,
                    "majority_fraction": item.majority_fraction,
                    "active_direction": item.active_direction,
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
# ## 7. Truth-late direction, cause, and safety readout
#
# All 32 first-pass messages, schedules, and treatment predictions are frozen
# before suffix truth or exp408 cause intervals are opened.

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


def active_direction_truth_readout(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    true_rate = physical_true_interval_rate(
        truth,
        last_known_tvt=frozen.last_known_tvt,
        last_known_md=frozen.last_known_md,
        last_known_z=frozen.last_known_z,
    )
    rows: list[dict[str, Any]] = []
    active_offsets = np.flatnonzero(frozen.active_direction != 0)
    for offset in active_offsets:
        correction = float(true_rate[offset] - frozen.filtered_rate_mean[offset])
        eligible = bool(np.isfinite(correction) and correction != 0.0)
        true_direction = int(np.sign(correction)) if eligible else 0
        direction = int(frozen.active_direction[offset])
        rows.append(
            {
                "well": frozen.well,
                "role": frozen.role,
                "fold": frozen.fold,
                "row_idx": int(frozen.row_idx[offset]),
                "suffix_offset": int(offset),
                "active_direction": direction,
                "filtered_rate_mean": float(frozen.filtered_rate_mean[offset]),
                "smoothed_rate_mean": float(frozen.smoothed_rate_mean[offset]),
                "filtered_rate_std": float(frozen.filtered_rate_std[offset]),
                "z_beta": float(frozen.z_beta[offset]),
                "true_interval_rate": float(true_rate[offset]),
                "true_minus_filtered_rate": correction,
                "eligible_correction_direction": eligible,
                "true_correction_direction": true_direction,
                "direction_agreement": bool(eligible and true_direction == direction),
            }
        )
    columns = [
        "well",
        "role",
        "fold",
        "row_idx",
        "suffix_offset",
        "active_direction",
        "filtered_rate_mean",
        "smoothed_rate_mean",
        "filtered_rate_std",
        "z_beta",
        "true_interval_rate",
        "true_minus_filtered_rate",
        "eligible_correction_direction",
        "true_correction_direction",
        "direction_agreement",
    ]
    return pd.DataFrame(rows, columns=columns)


def load_cause_episodes_after_all_freeze(
    config: Mapping[str, Any],
    selected_cause_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp408_episode_summary")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"exp408 episode summary SHA changed: {observed}")
    frame = pd.read_csv(
        path,
        usecols=[
            "episode_id",
            "well",
            "fold",
            "start_row_idx",
            "end_row_idx_exclusive",
            "rows",
            "episode_sse",
            "cause",
        ],
        dtype={"well": str, "episode_id": str},
    )
    frame = frame.loc[frame["well"].isin(selected_cause_wells)].copy()
    ledger.record_episode_late(len(frame))
    if frame.empty or frame["well"].nunique() != len(selected_cause_wells):
        raise ValueError("selected cause wells are missing exp408 episode rows")
    return frame.sort_values(
        ["well", "start_row_idx"], kind="mergesort"
    ).reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "selected_rows": len(frame),
    }


def cause_episode_readout(
    episodes: pd.DataFrame,
    frozen_by_well: Mapping[str, FrozenWell],
    truth_by_well: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_cause = {
        "backward_cause": "backward_smoothing_reversal",
        "forward_cause": "forward_transition_prior_hysteresis",
    }
    for episode in episodes.itertuples(index=False):
        frozen = frozen_by_well[str(episode.well)]
        if frozen.role not in expected_cause:
            continue
        if str(episode.cause) != expected_cause[frozen.role]:
            continue
        truth = truth_by_well[frozen.well]
        mask = (
            truth["row_idx"].to_numpy(np.int64) >= int(episode.start_row_idx)
        ) & (
            truth["row_idx"].to_numpy(np.int64)
            < int(episode.end_row_idx_exclusive)
        )
        offsets = np.flatnonzero(mask)
        if not len(offsets):
            raise ValueError(f"{episode.episode_id}: cause interval has no suffix rows")
        actual = truth.loc[mask, "TVT"].to_numpy(np.float64)
        baseline_error = frozen.baseline_prediction[offsets] - actual
        treatment_error = frozen.treatment_prediction[offsets] - actual
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": str(episode.well),
                "fold": frozen.fold,
                "role": frozen.role,
                "cause": str(episode.cause),
                "start_row_idx": int(episode.start_row_idx),
                "end_row_idx_exclusive": int(episode.end_row_idx_exclusive),
                "rows": len(offsets),
                "baseline_sse": float(np.sum(baseline_error**2)),
                "treatment_sse": float(np.sum(treatment_error**2)),
                "active_rows": int(
                    np.count_nonzero(frozen.active_direction[offsets])
                ),
                "active_row_fraction": float(
                    np.mean(frozen.active_direction[offsets] != 0)
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no selected exclusive cause episodes survived late join")
    return result.sort_values(["role", "well", "start_row_idx"], kind="mergesort")


def well_truth_late_metrics(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    actual = truth["TVT"].to_numpy(np.float64)
    parent_error = frozen.parent_prediction - actual
    baseline_error = frozen.baseline_prediction - actual
    treatment_error = frozen.treatment_prediction - actual
    return {
        "well": frozen.well,
        "role": frozen.role,
        "fold": frozen.fold,
        "rows": len(actual),
        "parent_rmse_ft": float(np.sqrt(np.mean(parent_error**2))),
        "baseline_rmse_ft": float(np.sqrt(np.mean(baseline_error**2))),
        "treatment_rmse_ft": float(np.sqrt(np.mean(treatment_error**2))),
        "rmse_delta_ft": float(
            np.sqrt(np.mean(treatment_error**2))
            - np.sqrt(np.mean(baseline_error**2))
        ),
        "active_rows": int(np.count_nonzero(frozen.active_direction)),
        "active_row_fraction": float(np.mean(frozen.active_direction != 0)),
        "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
        "baseline_prediction_sha256": frozen.baseline_prediction_sha256,
        "treatment_prediction_sha256": frozen.treatment_prediction_sha256,
        "baseline_message_sha256": frozen.baseline_message_sha256,
        "schedule_sha256": frozen.schedule_sha256,
        "baseline_saved_parent_max_abs_diff_ft": (
            frozen.baseline_saved_parent_max_abs_diff_ft
        ),
        "maximum_normalization_error": frozen.maximum_normalization_error,
        "baseline_hmm_elapsed_seconds": frozen.baseline_elapsed_seconds,
        "treatment_hmm_elapsed_seconds": frozen.treatment_elapsed_seconds,
    }


# %% [markdown]
# ## 8. Stage 0 gates, generated artifacts, and metrics

# %%
def fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    parity: Mapping[str, Any],
    schedule_artifact: Mapping[str, Any],
    direction_readout: pd.DataFrame,
    cause_readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "validation.stage_0.technical")
    mechanism_config = get_nested(config, "validation.stage_0.mechanism")
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    active_rows = int(
        sum(np.count_nonzero(item.active_direction) for item in frozen_wells)
    )
    active_row_fraction = fraction(active_rows, total_rows)
    active_wells = int(sum(np.any(item.active_direction) for item in frozen_wells))
    controls = [item for item in frozen_wells if item.role == "control"]
    backward = [item for item in frozen_wells if item.role == "backward_cause"]
    control_rows = int(sum(len(item.row_idx) for item in controls))
    control_active_rows = int(
        sum(np.count_nonzero(item.active_direction) for item in controls)
    )
    control_active_row_fraction = fraction(control_active_rows, control_rows)
    backward_rows = int(sum(len(item.row_idx) for item in backward))
    backward_active_rows = int(
        sum(np.count_nonzero(item.active_direction) for item in backward)
    )
    backward_active_row_coverage = fraction(
        backward_active_rows,
        backward_rows,
    )
    baseline_saved_parent_max_abs_diff = max(
        item.baseline_saved_parent_max_abs_diff_ft for item in frozen_wells
    )
    maximum_normalization_error = max(
        item.maximum_normalization_error for item in frozen_wells
    )
    finite_predictions = sum(
        int(np.isfinite(item.baseline_prediction).sum())
        + int(np.isfinite(item.treatment_prediction).sum())
        for item in frozen_wells
    )
    finite_coverage = fraction(finite_predictions, 2 * total_rows)
    runtime_projection = float(elapsed_seconds * 1546.0 / 64.0)

    expected_roles = {
        "backward_cause": 8,
        "forward_cause": 8,
        "control": 16,
    }
    cause_fold_coverage = all(
        set(manifest.loc[manifest["role"].eq(role), "fold"].astype(int))
        == set(range(5))
        for role in ("backward_cause", "forward_cause")
    )
    technical = {
        "fixed32_roles_and_unique_wells": bool(
            len(manifest) == 32
            and manifest["well"].nunique() == 32
            and manifest["role"].value_counts().to_dict() == expected_roles
        ),
        "cause_roles_cover_all_folds": bool(cause_fold_coverage),
        "truth_reads_before_all_freeze": ledger.truth_rows_before_all_freeze == 0,
        "cause_reads_before_all_freeze": ledger.episode_rows_before_all_freeze == 0,
        "zero_schedule_self_parity": bool(parity["pass"]),
        "baseline_saved_exp209_parity": bool(
            baseline_saved_parent_max_abs_diff
            <= float(technical_config["baseline_saved_prediction_tolerance_ft"])
        ),
        "baseline_and_treatment_normalization": bool(
            maximum_normalization_error
            <= float(technical_config["normalization_max_abs_error"])
        ),
        "finite_prediction_coverage": bool(
            finite_coverage >= float(technical_config["finite_coverage"])
        ),
        "activation_schedule_readback_sha": bool(
            schedule_artifact["logical_sha256"]
            == schedule_artifact["readback_logical_sha256"]
        ),
        "active_row_fraction": bool(
            float(technical_config["active_row_fraction_min"])
            <= active_row_fraction
            <= float(technical_config["active_row_fraction_max"])
        ),
        "active_wells": bool(
            active_wells >= int(technical_config["active_wells_min"])
        ),
        "runtime_projection": bool(
            runtime_projection
            <= float(technical_config["full_runtime_projection_max_seconds"])
        ),
        "peak_rss": bool(
            peak_rss_gb() <= float(technical_config["peak_rss_max_gb"])
        ),
    }

    eligible = direction_readout.loc[
        direction_readout["eligible_correction_direction"].astype(bool)
    ]
    direction_agreement = (
        float(eligible["direction_agreement"].mean()) if len(eligible) else math.nan
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_frame = eligible.loc[eligible["fold"].eq(fold)]
        agreement = (
            float(fold_frame["direction_agreement"].mean())
            if len(fold_frame)
            else math.nan
        )
        fold_rows.append(
            {
                "fold": fold,
                "eligible_active_rows": len(fold_frame),
                "direction_agreement": agreement,
                "strict_pass": bool(
                    math.isfinite(agreement)
                    and agreement
                    > float(
                        mechanism_config[
                            "per_fold_direction_agreement_strictly_above"
                        ]
                    )
                ),
            }
        )
    passing_folds = int(sum(row["strict_pass"] for row in fold_rows))

    cause_totals = (
        cause_readout.groupby("role", sort=True)[
            ["rows", "baseline_sse", "treatment_sse", "active_rows"]
        ]
        .sum()
        .to_dict(orient="index")
    )
    backward_totals = cause_totals.get("backward_cause", {})
    forward_totals = cause_totals.get("forward_cause", {})
    backward_sse_reduction = (
        1.0
        - float(backward_totals["treatment_sse"])
        / float(backward_totals["baseline_sse"])
        if float(backward_totals.get("baseline_sse", 0.0)) > 0.0
        else math.nan
    )
    forward_sse_regression = (
        float(forward_totals["treatment_sse"])
        / float(forward_totals["baseline_sse"])
        - 1.0
        if float(forward_totals.get("baseline_sse", 0.0)) > 0.0
        else math.nan
    )
    control_metrics = well_metrics.loc[well_metrics["role"].eq("control")]
    control_weight = control_metrics["rows"].to_numpy(np.float64)
    control_baseline_rmse = float(
        np.sqrt(
            np.average(
                control_metrics["baseline_rmse_ft"].to_numpy(np.float64) ** 2,
                weights=control_weight,
            )
        )
    )
    control_treatment_rmse = float(
        np.sqrt(
            np.average(
                control_metrics["treatment_rmse_ft"].to_numpy(np.float64) ** 2,
                weights=control_weight,
            )
        )
    )
    control_rmse_delta = control_treatment_rmse - control_baseline_rmse
    mechanism = {
        "beta_direction_correction_agreement": bool(
            math.isfinite(direction_agreement)
            and direction_agreement
            >= float(mechanism_config["beta_direction_correction_agreement_min"])
        ),
        "passing_direction_folds": bool(
            passing_folds >= int(mechanism_config["passing_folds_min"])
        ),
        "backward_active_row_coverage": bool(
            backward_active_row_coverage
            >= float(mechanism_config["backward_active_row_coverage_min"])
        ),
        "backward_cause_sse_reduction": bool(
            math.isfinite(backward_sse_reduction)
            and backward_sse_reduction
            >= float(mechanism_config["backward_sse_reduction_min"])
        ),
        "forward_cause_sse_safety": bool(
            math.isfinite(forward_sse_regression)
            and forward_sse_regression
            <= float(mechanism_config["forward_sse_regression_max"])
        ),
        "matched_control_safety": bool(
            control_rmse_delta
            <= float(mechanism_config["control_rmse_delta_max_ft"])
            and control_active_row_fraction
            <= float(mechanism_config["control_active_row_fraction_max"])
        ),
    }
    diagnostics = {
        "total_rows": total_rows,
        "active_rows": active_rows,
        "active_row_fraction": active_row_fraction,
        "active_wells": active_wells,
        "baseline_saved_parent_max_abs_diff_ft": (
            baseline_saved_parent_max_abs_diff
        ),
        "maximum_normalization_error": maximum_normalization_error,
        "finite_coverage": finite_coverage,
        "eligible_active_direction_rows": len(eligible),
        "beta_direction_correction_agreement": direction_agreement,
        "direction_agreement_by_fold": fold_rows,
        "passing_direction_folds": passing_folds,
        "backward_active_row_coverage": backward_active_row_coverage,
        "backward_cause_sse_reduction": backward_sse_reduction,
        "forward_cause_sse_regression": forward_sse_regression,
        "control_baseline_rmse_ft": control_baseline_rmse,
        "control_treatment_rmse_ft": control_treatment_rmse,
        "control_rmse_delta_ft": control_rmse_delta,
        "control_active_row_fraction": control_active_row_fraction,
        "runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
    }
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "promotion_eligible": bool(all(technical.values()) and all(mechanism.values())),
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP412_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp412 Stage 0 must run on Kaggle CPU; local execution is disabled"
    )


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
    hmm = get_nested(config, "model.parent_hmm")
    trigger = get_nested(config, "model.trigger")
    treatment = get_nested(config, "model.treatment")
    parity = synthetic_zero_schedule_parity(hmm, treatment)
    if not parity["pass"]:
        raise RuntimeError(f"synthetic zero-schedule parity failed: {parity}")
    raw_dir = train_data_dir(config)
    parent_groups = parent.groupby("well", sort=False).indices
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))

    for well_index, row in enumerate(manifest.itertuples(index=False), start=1):
        well = str(row.well)
        if well not in parent_groups:
            raise ValueError(f"{well}: saved parent rows are missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            saved_parent=parent.iloc[parent_groups[well]].copy(),
            hmm=hmm,
            trigger=trigger,
            treatment=treatment,
            parent_parity_tolerance_ft=float(
                get_nested(
                    config,
                    "validation.stage_0.technical."
                    "baseline_saved_prediction_tolerance_ft",
                )
            ),
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
                    "event": "exp412_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "active_rows": int(np.count_nonzero(frozen.active_direction)),
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
    schedule = schedule_frame(frozen_wells)
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz",
        predictions,
    )
    schedule_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_activation_schedule.csv.gz",
        schedule,
    )
    if (
        schedule_artifact["logical_sha256"]
        != schedule_artifact["readback_logical_sha256"]
    ):
        raise RuntimeError("activation schedule readback SHA mismatch")

    frozen_by_well = {item.well: item for item in frozen_wells}
    truth_by_well: dict[str, pd.DataFrame] = {}
    direction_pieces: list[pd.DataFrame] = []
    well_metric_rows: list[dict[str, Any]] = []
    for item in frozen_wells:
        truth = load_truth_after_all_freeze(item, raw_dir, ledger)
        truth_by_well[item.well] = truth
        direction_pieces.append(active_direction_truth_readout(item, truth))
        well_metric_rows.append(well_truth_late_metrics(item, truth))
    direction_readout = pd.concat(direction_pieces, ignore_index=True)
    if direction_readout.empty:
        direction_readout = active_direction_truth_readout(
            frozen_wells[0], truth_by_well[frozen_wells[0].well]
        ).iloc[0:0]
    cause_wells = set(
        manifest.loc[
            manifest["role"].isin(["backward_cause", "forward_cause"]),
            "well",
        ].astype(str)
    )
    episodes, episode_input = load_cause_episodes_after_all_freeze(
        config,
        cause_wells,
        ledger,
    )
    cause_readout = cause_episode_readout(
        episodes,
        frozen_by_well,
        truth_by_well,
    )
    well_metrics = pd.DataFrame(well_metric_rows).sort_values(
        ["fold", "role", "well"], kind="mergesort"
    )

    direction_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_direction_truth_late_readout.csv",
        direction_readout,
    )
    cause_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_cause_episode_readout.csv",
        cause_readout,
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
        parity=parity,
        schedule_artifact=schedule_artifact,
        direction_readout=direction_readout,
        cause_readout=cause_readout,
        well_metrics=well_metrics,
        ledger=ledger,
        elapsed_seconds=elapsed,
    )
    input_manifest = {
        "fixed32_manifest": manifest_input,
        "saved_exp209_control": parent_input,
        "exp408_cause_episodes": episode_input,
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
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0_promotion_eligible"
            if gates["promotion_eligible"]
            else "stage0_fail_closed"
        ),
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "zero_schedule_self_parity": parity,
        "gates": gates,
        "baseline_message_manifest_sha256": combined_well_sha(
            frozen_wells, "baseline_message_sha256"
        ),
        "baseline_prediction_manifest_sha256": combined_well_sha(
            frozen_wells, "baseline_prediction_sha256"
        ),
        "treatment_prediction_manifest_sha256": combined_well_sha(
            frozen_wells, "treatment_prediction_sha256"
        ),
        "schedule_manifest_sha256": combined_well_sha(
            frozen_wells, "schedule_sha256"
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
            "activation_schedule": schedule_artifact,
            "direction_truth_late_readout": direction_artifact,
            "cause_episode_readout": cause_artifact,
            "well_metrics": well_artifact,
            "input_manifest": input_artifact,
        },
        "stage_1": {
            "implemented": False,
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
            "stage": "stage_0_fixed32",
            "cv": None,
            "lb": None,
            "rmse_is_diagnostic_not_promotion_gate": True,
        },
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "technical_gates": gates["technical"],
        "mechanism_gates": gates["mechanism"],
        "promotion_eligible": gates["promotion_eligible"],
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 9. Configuration preview and guarded execution
#
# The notebook prints the 32-baseline + 32-treatment / zero-model cost contract
# before executing the explicitly authorized Stage 0 run on Kaggle CPU.

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
                "event": "exp412_stage0_preview",
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "run_stage": get_nested(CONFIG, "execution.run_stage"),
                "execution_counts": EXECUTION_COUNTS,
                "stage_0_execution_approved": get_nested(
                    CONFIG, "design.stage_0_execution_approved"
                ),
                "kaggle_execution_authorized": get_nested(
                    CONFIG, "execution.kaggle_execution_authorized"
                ),
                "stage_1": False,
                "inference": False,
                "submission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    SUMMARY = run_stage0(CONFIG)
