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
# # exp425 symmetric datum reanchor exact HMM — Stage 0
#
# This CPU-only notebook runs an unchanged exp209 first pass and freezes the
# first persistent smoothed-versus-filtered rate-disagreement event. At that
# transition it creates negative, parent, and positive absolute-datum branches
# with fixed 0.10/0.80/0.10 prior mass. The three conditional exact-HMM passes
# are combined by their full-sequence evidence, which is algebraically the same
# exact sum-product marginal as an explicit branch-state dimension. This source
# is implemented but Kaggle execution, Stage 1, inference, and submission remain
# separately disabled.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 manifest, saved parent, and target-free raw inputs
# 4. Exact exp209 HMM input preparation
# 5. Exact position-shift HMM and symmetric branch marginalization
# 6. Parent parity, event freeze, and target-free prediction freeze
# 7. Truth-late datum-direction, cause, and safety readout
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
from numba import njit, prange, set_num_threads

EXPERIMENT_NAME = "exp425_symmetric_datum_reanchor_exact_hmm"
PARENT_EXPERIMENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
EVIDENCE_EXPERIMENTS = (
    "exp408_hmm_message_rate_basin_audit",
    "exp412_beta_filter_rate_disagreement_two_pass_reset",
)
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
        raise ValueError("wrong exp425 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp425 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp425 scientific parent changed")
    if tuple(get_nested(config, "lineage.evidence_parents", ())) != EVIDENCE_EXPERIMENTS:
        raise ValueError("exp425 evidence parents changed")
    if not bool(get_nested(config, "design.implementation_authorized", False)):
        raise RuntimeError("exp425 implementation is not authorized")
    if not bool(
        get_nested(config, "design.canonical_notebook_adoption_authorized", False)
    ):
        raise RuntimeError("exp425 canonical notebook adoption is not authorized")
    if bool(get_nested(config, "design.kaggle_stage_1_authorized", True)):
        raise ValueError("Stage 1 must remain disabled during Stage 0")
    if bool(get_nested(config, "design.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "design.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if bool(get_nested(config, "runtime.enable_gpu", True)):
        raise ValueError("exp425 is CPU-only")

    expected = {
        "active_scientific_variants": 1,
        "planned_stage_0_baseline_hmm_well_runs": 32,
        "planned_stage_0_treatment_hmm_logical_well_runs": 32,
        "planned_stage_0_total_logical_hmm_well_runs": 64,
        "planned_stage_0_treatment_branch_states": 3,
        "parent_control_hmm_reruns_stage_0": 32,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
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
    if require_run_authorization:
        if not bool(get_nested(config, "design.kaggle_stage_0_authorized", False)):
            raise RuntimeError(
                "implementation approval does not authorize Kaggle Stage 0"
            )
        if not bool(get_nested(config, "execution.run_hmm", False)):
            raise RuntimeError(
                "execution.run_hmm is false; Kaggle Stage 0 remains fail-closed"
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
        "event_definition": "first_false_to_true_persistent_activation",
        "maximum_events_per_well": 1,
        "rate_sign_selects_datum_direction": False,
        "tie_policy": "inactive",
        "freeze_before_truth": True,
        "recompute_from_treatment": False,
    }
    if trigger != expected_trigger:
        raise ValueError(f"trigger contract changed: {trigger} != {expected_trigger}")
    branch = get_nested(config, "model.datum_branch")
    expected_branch = {
        "explicit_state_values": ["negative", "parent", "positive"],
        "branch_count": 3,
        "prior_mass": {"negative": 0.10, "parent": 0.80, "positive": 0.10},
        "shift_source": "first_pass_filtered_position_std_at_event",
        "shift_scale": 1.0,
        "shift_floor_ft": 0.35,
        "shift_formula": "max(filtered_position_std_ft, 0.35)",
        "negative_shift_sign": -1,
        "positive_shift_sign": 1,
        "event_transition_only": True,
        "persist_branch_identity_after_event": True,
        "allow_branch_switch_after_event": False,
        "allow_additional_events": False,
        "selection": "exact_sum_product_soft_marginalization",
        "hard_map_or_viterbi_selection": False,
    }
    if branch != expected_branch:
        raise ValueError(
            f"symmetric datum branch contract changed: {branch} != {expected_branch}"
        )
    return {
        "parent_hmm": parent,
        "trigger": trigger,
        "datum_branch": branch,
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
    raise FileNotFoundError("exp425 config.yaml was not found")


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
    metadata_spec = get_nested(config, "data.stage_0_manifest_metadata")
    metadata_path = resolve_bootstrap_asset(
        str(metadata_spec["filename"]),
        str(metadata_spec["local"]),
    )
    metadata_sha = sha256_file(metadata_path)
    if metadata_sha != str(metadata_spec["expected_sha256"]):
        raise ValueError(f"fixed32 metadata SHA changed: {metadata_sha}")
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("manifest", {}).get("rows", -1)) != 32:
        raise ValueError("fixed32 metadata row count changed")
    ledger.record_scope(len(frame))
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "logical_sha256": logical_frame_sha256(frame),
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha,
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
# ## 5. Exact position-shift HMM and symmetric branch marginalization
#
# `position_shift_ft[t]` belongs to the transition entering suffix row `t`.
# It is zero everywhere for the unchanged parent. A conditional reanchor branch
# has exactly one nonzero transition. The rate kernel, position noise, emission,
# support, and posterior-mean readout remain identical to exp209.

# %%
@njit(cache=True, nogil=True)
def rate_kernel_probabilities(
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    mom: float,
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
        kernel[r_i, 0] = p_minus
        kernel[r_i, 1] = 1.0 - p_plus - p_minus
        kernel[r_i, 2] = p_plus
    return kernel


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_position_shift_schedule(
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
    position_shift_ft,
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
    filtered_position_mean = np.empty(t_count, np.float64)
    filtered_position_second_moment = np.empty(t_count, np.float64)
    maximum_forward_normalization_error = 0.0

    for t_i in range(t_count):
        kernel = rate_kernel_probabilities(
            rates,
            dm[t_i],
            sig_r,
            mom,
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
            mu = rates[r2] * dm[t_i] - dz[t_i] + position_shift_ft[t_i]
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
        filtered_p1 = 0.0
        filtered_p2 = 0.0
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
                position_ft = p_i * sp
                filtered_p1 += filtered_probability * position_ft
                filtered_p2 += filtered_probability * position_ft * position_ft
        predictive_rate_mean[t_i] = predictive_r1 / predictive_total
        filtered_rate_mean[t_i] = filtered_r1 / filtered_total
        filtered_rate_second_moment[t_i] = filtered_r2 / filtered_total
        filtered_position_mean[t_i] = filtered_p1 / filtered_total
        filtered_position_second_moment[t_i] = filtered_p2 / filtered_total
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
        )
        rate_log_kernel = np.log(kernel)
        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i] + position_shift_ft[t_i]
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
    filtered_position_variance = np.maximum(
        filtered_position_second_moment
        - filtered_position_mean * filtered_position_mean,
        0.0,
    )
    return (
        post_p,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        np.sqrt(filtered_rate_variance),
        np.sqrt(filtered_position_variance),
        smoothed_rate_mean,
        maximum_forward_normalization_error,
        maximum_posterior_normalization_error,
    )


def run_hmm_pass(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    *,
    position_shift_ft: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    shift = np.asarray(position_shift_ft, dtype=np.float64)
    if len(shift) != len(prepared["eval_index"]):
        raise ValueError("position-shift schedule length does not match suffix rows")
    if not np.isfinite(shift).all():
        raise ValueError("position-shift schedule must be finite")
    result = _hmm2_position_shift_schedule(
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
        shift,
    )
    (
        post_p,
        log_likelihood,
        predictive_rate_mean,
        filtered_rate_mean,
        filtered_rate_std,
        filtered_position_std,
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
        filtered_position_std=np.asarray(filtered_position_std, dtype=np.float64),
        smoothed_rate_mean=np.asarray(smoothed_rate_mean, dtype=np.float64),
        position_shift_ft=shift,
    )
    return {
        "posterior_mean": posterior_mean,
        "log_likelihood": float(log_likelihood),
        "predictive_rate_mean": predictive_rate_mean,
        "filtered_rate_mean": filtered_rate_mean,
        "filtered_rate_std": filtered_rate_std,
        "filtered_position_std": filtered_position_std,
        "smoothed_rate_mean": smoothed_rate_mean,
        "position_shift_ft": shift,
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


def first_persistent_activation_event(active_direction: np.ndarray) -> int:
    """Return the first inactive-to-active row, ignoring the rate-gap sign."""
    active = np.asarray(active_direction, dtype=np.int8)
    if not set(np.unique(active)).issubset({-1, 0, 1}):
        raise ValueError("activation direction must contain only -1, 0, or 1")
    active_flag = active != 0
    entered = active_flag & ~np.r_[False, active_flag[:-1]]
    indices = np.flatnonzero(entered)
    return int(indices[0]) if len(indices) else -1


def normalized_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("three finite branch log weights are required")
    maximum = float(np.max(values))
    denominator = float(np.sum(np.exp(values - maximum)))
    log_evidence = maximum + math.log(denominator)
    weights = np.exp(values - log_evidence)
    return weights, log_evidence


def run_symmetric_datum_treatment(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
    branch: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    event_index: int,
    datum_shift_ft: float,
) -> dict[str, Any]:
    """Exact three-state datum marginal via conditional-HMM factorization.

    Conditional on branch identity, every transition is the parent exp209
    transition except the position kernel entering ``event_index``. Summing
    the three conditional smoothers with posterior model probabilities is
    algebraically identical to exact sum-product on an explicit persistent
    branch-state dimension.
    """
    started = time.perf_counter()
    rows = len(prepared["eval_index"])
    event_index = int(event_index)
    if event_index < -1 or event_index >= rows:
        raise ValueError("event index lies outside the suffix")
    shift = float(datum_shift_ft)
    if not math.isfinite(shift) or shift < float(branch["shift_floor_ft"]):
        raise ValueError("datum shift violates the fixed finite floor")

    prior_map = branch["prior_mass"]
    priors = np.asarray(
        [
            float(prior_map["negative"]),
            float(prior_map["parent"]),
            float(prior_map["positive"]),
        ],
        dtype=np.float64,
    )
    if not np.all(priors > 0.0) or abs(float(priors.sum()) - 1.0) > 1.0e-12:
        raise ValueError("branch priors must be positive and sum to one")

    if event_index < 0:
        branch_mass = np.zeros((rows, 3), dtype=np.float64)
        branch_mass[:, 1] = 1.0
        prediction = np.asarray(baseline["posterior_mean"], dtype=np.float64).copy()
        return {
            "posterior_mean": prediction,
            "log_likelihood": float(baseline["log_likelihood"]),
            "event_index": -1,
            "datum_shift_ft": 0.0,
            "branch_posterior_final": np.asarray([0.0, 1.0, 0.0]),
            "branch_posterior_mass": branch_mass,
            "conditional_log_likelihood": np.full(
                3, float(baseline["log_likelihood"]), dtype=np.float64
            ),
            "conditional_prediction": np.vstack([prediction, prediction, prediction]),
            "maximum_normalization_error": float(
                baseline["maximum_normalization_error"]
            ),
            "prediction_sha256": str(baseline["prediction_sha256"]),
            "branch_posterior_sha256": array_bundle_sha256(
                row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
                branch_posterior_mass=branch_mass,
            ),
            "shift_schedule_sha256": array_bundle_sha256(
                row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
                event_index=np.asarray([-1], dtype=np.int64),
                datum_shift_ft=np.asarray([0.0], dtype=np.float64),
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        }

    schedules = []
    for sign in (-1.0, 1.0):
        schedule = np.zeros(rows, dtype=np.float64)
        schedule[event_index] = sign * shift
        schedules.append(schedule)
    negative = run_hmm_pass(
        prepared,
        hmm,
        position_shift_ft=schedules[0],
    )
    positive = run_hmm_pass(
        prepared,
        hmm,
        position_shift_ft=schedules[1],
    )
    conditionals = (negative, baseline, positive)
    log_likelihoods = np.asarray(
        [float(item["log_likelihood"]) for item in conditionals],
        dtype=np.float64,
    )
    weights, log_evidence = normalized_log_weights(
        np.log(priors) + log_likelihoods
    )
    conditional_prediction = np.vstack(
        [
            np.asarray(item["posterior_mean"], dtype=np.float64)
            for item in conditionals
        ]
    )
    prediction = weights @ conditional_prediction
    branch_mass = np.zeros((rows, 3), dtype=np.float64)
    branch_mass[:event_index, 1] = 1.0
    branch_mass[event_index:] = weights
    prediction_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        posterior_mean=np.asarray(prediction, dtype=np.float32),
    )
    branch_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        branch_posterior_mass=branch_mass,
    )
    shift_sha = array_bundle_sha256(
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
        event_index=np.asarray([event_index], dtype=np.int64),
        datum_shift_ft=np.asarray([shift], dtype=np.float64),
        negative_schedule=schedules[0],
        positive_schedule=schedules[1],
    )
    maximum_normalization_error = max(
        abs(float(weights.sum()) - 1.0),
        *(float(item["maximum_normalization_error"]) for item in conditionals),
    )
    return {
        "posterior_mean": prediction,
        "log_likelihood": log_evidence,
        "event_index": event_index,
        "datum_shift_ft": shift,
        "branch_posterior_final": weights,
        "branch_posterior_mass": branch_mass,
        "conditional_log_likelihood": log_likelihoods,
        "conditional_prediction": conditional_prediction,
        "maximum_normalization_error": maximum_normalization_error,
        "prediction_sha256": prediction_sha,
        "branch_posterior_sha256": branch_sha,
        "shift_schedule_sha256": shift_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 6. Parent parity, event freeze, and target-free prediction freeze

# %%
def synthetic_parent_only_branch_parity(
    hmm: Mapping[str, Any],
    branch: Mapping[str, Any],
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
    zero_schedule = np.zeros(rows, dtype=np.float64)
    parent = run_hmm_pass(
        prepared,
        hmm,
        position_shift_ft=zero_schedule,
    )
    no_event = run_symmetric_datum_treatment(
        prepared,
        hmm,
        branch,
        baseline=parent,
        event_index=-1,
        datum_shift_ft=float(branch["shift_floor_ft"]),
    )
    posterior_diff = float(
        np.max(np.abs(parent["posterior_mean"] - no_event["posterior_mean"]))
    )
    log_likelihood_diff = abs(
        float(parent["log_likelihood"]) - float(no_event["log_likelihood"])
    )
    return {
        "posterior_mean_max_abs_diff_ft": posterior_diff,
        "log_likelihood_abs_diff": log_likelihood_diff,
        "event_index": -1,
        "parent_prediction_sha256": parent["prediction_sha256"],
        "parent_only_prediction_sha256": no_event["prediction_sha256"],
        "branch_posterior_final": no_event["branch_posterior_final"],
        "pass": bool(
            posterior_diff <= 1.0e-10
            and log_likelihood_diff <= 1.0e-10
            and np.array_equal(
                no_event["branch_posterior_final"],
                np.asarray([0.0, 1.0, 0.0]),
            )
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
    filtered_position_std: np.ndarray
    smoothed_rate_mean: np.ndarray
    z_beta: np.ndarray
    qualifying_count: np.ndarray
    majority_fraction: np.ndarray
    active_direction: np.ndarray
    event_index: int
    datum_shift_ft: float
    branch_posterior_mass: np.ndarray
    last_known_tvt: float
    last_known_md: float
    last_known_z: float
    schedule_sha256: str
    shift_schedule_sha256: str
    branch_posterior_sha256: str
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
    branch: Mapping[str, Any],
    parent_parity_tolerance_ft: float,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_hmm_inputs(horizontal, typewell, hmm)
    zero_schedule = np.zeros(len(prepared["eval_index"]), dtype=np.float64)
    baseline = run_hmm_pass(
        prepared,
        hmm,
        position_shift_ft=zero_schedule,
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
    event_index = first_persistent_activation_event(active_direction)
    datum_shift_ft = float(branch["shift_floor_ft"])
    if event_index >= 0:
        datum_shift_ft = max(
            float(baseline["filtered_position_std"][event_index])
            * float(branch["shift_scale"]),
            float(branch["shift_floor_ft"]),
        )
    schedule_sha = array_bundle_sha256(
        row_idx=row_idx,
        filtered_rate_mean=np.asarray(
            baseline["filtered_rate_mean"], dtype=np.float64
        ),
        filtered_rate_std=np.asarray(
            baseline["filtered_rate_std"], dtype=np.float64
        ),
        filtered_position_std=np.asarray(
            baseline["filtered_position_std"], dtype=np.float64
        ),
        smoothed_rate_mean=np.asarray(
            baseline["smoothed_rate_mean"], dtype=np.float64
        ),
        z_beta=np.asarray(schedule["z_beta"], dtype=np.float64),
        qualifying_count=np.asarray(schedule["qualifying_count"], dtype=np.int16),
        majority_fraction=np.asarray(schedule["majority_fraction"], dtype=np.float64),
        active_direction=active_direction,
        event_index=np.asarray([event_index], dtype=np.int64),
        datum_shift_ft=np.asarray(
            [datum_shift_ft if event_index >= 0 else 0.0], dtype=np.float64
        ),
    )
    treatment_result = run_symmetric_datum_treatment(
        prepared,
        hmm,
        branch,
        baseline=baseline,
        event_index=event_index,
        datum_shift_ft=datum_shift_ft,
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
        filtered_position_std=np.asarray(
            baseline["filtered_position_std"], dtype=np.float64
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
        event_index=event_index,
        datum_shift_ft=(
            float(treatment_result["datum_shift_ft"]) if event_index >= 0 else 0.0
        ),
        branch_posterior_mass=np.asarray(
            treatment_result["branch_posterior_mass"], dtype=np.float64
        ),
        last_known_tvt=float(prepared["last_known_tvt"]),
        last_known_md=float(prepared["last_known_md"]),
        last_known_z=float(prepared["last_known_z"]),
        schedule_sha256=schedule_sha,
        shift_schedule_sha256=str(treatment_result["shift_schedule_sha256"]),
        branch_posterior_sha256=str(
            treatment_result["branch_posterior_sha256"]
        ),
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
                    "filtered_position_std_ft": item.filtered_position_std,
                    "smoothed_rate_mean": item.smoothed_rate_mean,
                    "z_beta": item.z_beta,
                    "qualifying_count": item.qualifying_count,
                    "majority_fraction": item.majority_fraction,
                    "active_direction": item.active_direction,
                    "event_index": item.event_index,
                    "event_row": (
                        np.arange(len(item.row_idx), dtype=np.int64)
                        == item.event_index
                    ),
                    "datum_shift_ft": item.datum_shift_ft,
                    "negative_branch_mass": item.branch_posterior_mass[:, 0],
                    "parent_branch_mass": item.branch_posterior_mass[:, 1],
                    "positive_branch_mass": item.branch_posterior_mass[:, 2],
                    "reanchor_branch_mass": (
                        item.branch_posterior_mass[:, 0]
                        + item.branch_posterior_mass[:, 2]
                    ),
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
# ## 7. Truth-late datum-direction, cause, and safety readout
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


def soft_datum_direction_truth_readout(
    frozen: FrozenWell,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_offsets = (
        np.arange(frozen.event_index, len(frozen.row_idx), dtype=np.int64)
        if frozen.event_index >= 0
        else np.asarray([], dtype=np.int64)
    )
    actual = truth["TVT"].to_numpy(np.float64)
    for offset in active_offsets:
        branch_delta = float(
            frozen.branch_posterior_mass[offset, 2]
            - frozen.branch_posterior_mass[offset, 0]
        )
        truth_delta = float(actual[offset] - frozen.baseline_prediction[offset])
        eligible = bool(
            np.isfinite(branch_delta)
            and np.isfinite(truth_delta)
            and branch_delta != 0.0
            and truth_delta != 0.0
        )
        soft_direction = int(np.sign(branch_delta)) if eligible else 0
        true_direction = int(np.sign(truth_delta)) if eligible else 0
        rows.append(
            {
                "well": frozen.well,
                "role": frozen.role,
                "fold": frozen.fold,
                "row_idx": int(frozen.row_idx[offset]),
                "suffix_offset": int(offset),
                "event_index": frozen.event_index,
                "event_row_idx": int(frozen.row_idx[frozen.event_index]),
                "datum_shift_ft": frozen.datum_shift_ft,
                "negative_branch_mass": float(
                    frozen.branch_posterior_mass[offset, 0]
                ),
                "parent_branch_mass": float(
                    frozen.branch_posterior_mass[offset, 1]
                ),
                "positive_branch_mass": float(
                    frozen.branch_posterior_mass[offset, 2]
                ),
                "reanchor_branch_mass": float(
                    frozen.branch_posterior_mass[offset, 0]
                    + frozen.branch_posterior_mass[offset, 2]
                ),
                "positive_minus_negative_mass": branch_delta,
                "parent_prediction": float(frozen.baseline_prediction[offset]),
                "treatment_prediction": float(frozen.treatment_prediction[offset]),
                "truth_tvt": float(actual[offset]),
                "truth_minus_parent_prediction": truth_delta,
                "eligible_soft_datum_direction": eligible,
                "soft_datum_direction": soft_direction,
                "true_datum_direction": true_direction,
                "direction_agreement": bool(
                    eligible and true_direction == soft_direction
                ),
            }
        )
    columns = [
        "well",
        "role",
        "fold",
        "row_idx",
        "suffix_offset",
        "event_index",
        "event_row_idx",
        "datum_shift_ft",
        "negative_branch_mass",
        "parent_branch_mass",
        "positive_branch_mass",
        "reanchor_branch_mass",
        "positive_minus_negative_mass",
        "parent_prediction",
        "treatment_prediction",
        "truth_tvt",
        "truth_minus_parent_prediction",
        "eligible_soft_datum_direction",
        "soft_datum_direction",
        "true_datum_direction",
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
                "post_event_rows": int(
                    np.count_nonzero(offsets >= frozen.event_index)
                    if frozen.event_index >= 0
                    else 0
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
        "event_present": frozen.event_index >= 0,
        "event_index": frozen.event_index,
        "datum_shift_ft": frozen.datum_shift_ft,
        "post_event_rows": (
            len(frozen.row_idx) - frozen.event_index
            if frozen.event_index >= 0
            else 0
        ),
        "reanchor_branch_mass_mean": float(
            np.mean(
                frozen.branch_posterior_mass[:, 0]
                + frozen.branch_posterior_mass[:, 2]
            )
        ),
        "raw_gr_missing_fraction": float(np.mean(frozen.raw_gr_missing)),
        "baseline_prediction_sha256": frozen.baseline_prediction_sha256,
        "treatment_prediction_sha256": frozen.treatment_prediction_sha256,
        "baseline_message_sha256": frozen.baseline_message_sha256,
        "schedule_sha256": frozen.schedule_sha256,
        "shift_schedule_sha256": frozen.shift_schedule_sha256,
        "branch_posterior_sha256": frozen.branch_posterior_sha256,
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
    event_wells = [item for item in frozen_wells if item.event_index >= 0]
    event_well_count = len(event_wells)
    event_folds = {item.fold for item in event_wells}
    controls = [item for item in frozen_wells if item.role == "control"]
    baseline_saved_parent_max_abs_diff = max(
        item.baseline_saved_parent_max_abs_diff_ft for item in frozen_wells
    )
    maximum_normalization_error = max(
        item.maximum_normalization_error for item in frozen_wells
    )
    finite_values = 0
    total_values = 0
    for item in frozen_wells:
        for values in (
            item.baseline_prediction,
            item.treatment_prediction,
            item.filtered_position_std,
            item.branch_posterior_mass,
        ):
            array = np.asarray(values, dtype=np.float64)
            finite_values += int(np.isfinite(array).sum())
            total_values += int(array.size)
    finite_coverage = fraction(finite_values, total_values)
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
        "parent_only_branch_self_parity": bool(parity["pass"]),
        "baseline_saved_exp209_parity": bool(
            baseline_saved_parent_max_abs_diff
            <= float(technical_config["untreated_parent_parity_max_abs_ft"])
        ),
        "baseline_and_treatment_normalization": bool(
            maximum_normalization_error
            <= float(technical_config["normalization_max_abs_error"])
        ),
        "finite_prediction_coverage": bool(
            finite_coverage >= float(technical_config["finite_coverage"])
        ),
        "event_schedule_readback_sha": bool(
            schedule_artifact["logical_sha256"]
            == schedule_artifact["readback_logical_sha256"]
        ),
        "event_wells": bool(
            event_well_count >= int(technical_config["event_wells_min"])
        ),
        "event_wells_cover_all_folds": bool(
            (not bool(technical_config["require_event_well_all_folds"]))
            or event_folds == set(range(5))
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
        direction_readout["eligible_soft_datum_direction"].astype(bool)
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
                "eligible_post_event_rows": len(fold_frame),
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
            ["rows", "baseline_sse", "treatment_sse", "post_event_rows"]
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
    control_event_values = [
        item.branch_posterior_mass[item.event_index :, 0]
        + item.branch_posterior_mass[item.event_index :, 2]
        for item in controls
        if item.event_index >= 0
    ]
    control_reanchor_mass = (
        float(np.mean(np.concatenate(control_event_values)))
        if control_event_values
        else 0.0
    )
    active_reanchor_values = np.concatenate(
        [
            item.branch_posterior_mass[item.event_index :, 0]
            + item.branch_posterior_mass[item.event_index :, 2]
            for item in event_wells
        ]
    )
    active_reanchor_mass = (
        float(np.mean(active_reanchor_values))
        if len(active_reanchor_values)
        else math.nan
    )
    mechanism = {
        "soft_datum_direction_agreement": bool(
            math.isfinite(direction_agreement)
            and direction_agreement
            >= float(mechanism_config["soft_datum_direction_agreement_min"])
        ),
        "passing_direction_folds": bool(
            passing_folds >= int(mechanism_config["passing_folds_min"])
        ),
        "backward_cause_sse_reduction": bool(
            math.isfinite(backward_sse_reduction)
            and backward_sse_reduction
            >= float(mechanism_config["backward_cause_sse_reduction_min"])
        ),
        "forward_cause_sse_safety": bool(
            math.isfinite(forward_sse_regression)
            and forward_sse_regression
            <= float(mechanism_config["forward_cause_sse_regression_max"])
        ),
        "matched_control_rmse_safety": bool(
            control_rmse_delta
            <= float(mechanism_config["control_rmse_delta_max_ft"])
        ),
        "matched_control_branch_mass_safety": bool(
            control_reanchor_mass
            <= float(
                mechanism_config["control_reanchor_posterior_mass_mean_max"]
            )
        ),
        "active_reanchor_mass": bool(
            math.isfinite(active_reanchor_mass)
            and active_reanchor_mass
            >= float(mechanism_config["active_reanchor_posterior_mass_mean_min"])
        ),
    }
    diagnostics = {
        "total_rows": total_rows,
        "event_wells": event_well_count,
        "event_folds": sorted(event_folds),
        "baseline_saved_parent_max_abs_diff_ft": (
            baseline_saved_parent_max_abs_diff
        ),
        "maximum_normalization_error": maximum_normalization_error,
        "finite_coverage": finite_coverage,
        "eligible_soft_datum_direction_rows": len(eligible),
        "soft_datum_direction_agreement": direction_agreement,
        "direction_agreement_by_fold": fold_rows,
        "passing_direction_folds": passing_folds,
        "backward_cause_sse_reduction": backward_sse_reduction,
        "forward_cause_sse_regression": forward_sse_regression,
        "control_baseline_rmse_ft": control_baseline_rmse,
        "control_treatment_rmse_ft": control_treatment_rmse,
        "control_rmse_delta_ft": control_rmse_delta,
        "control_reanchor_posterior_mass_mean": control_reanchor_mass,
        "active_reanchor_posterior_mass_mean": active_reanchor_mass,
        "runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
    }
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "stage_1_eligible": bool(all(technical.values()) and all(mechanism.values())),
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP425_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp425 Stage 0 must run on Kaggle CPU; local execution is disabled"
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
    branch = get_nested(config, "model.datum_branch")
    parity = synthetic_parent_only_branch_parity(hmm, branch)
    if not parity["pass"]:
        raise RuntimeError(f"synthetic parent-only branch parity failed: {parity}")
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
            branch=branch,
            parent_parity_tolerance_ft=float(
                get_nested(
                    config,
                    "validation.stage_0.technical.untreated_parent_parity_max_abs_ft",
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
                    "event": "exp425_stage0_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "event_index": frozen.event_index,
                    "datum_shift_ft": frozen.datum_shift_ft,
                    "branch_posterior_final": (
                        frozen.branch_posterior_mass[-1].tolist()
                    ),
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
        output / f"{EXPERIMENT_NAME}_stage0_event_branch_schedule.csv.gz",
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
        direction_pieces.append(soft_datum_direction_truth_readout(item, truth))
        well_metric_rows.append(well_truth_late_metrics(item, truth))
    direction_readout = pd.concat(direction_pieces, ignore_index=True)
    if direction_readout.empty:
        direction_readout = soft_datum_direction_truth_readout(
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
            "stage0_mechanism_pass_stage1_approval_required"
            if gates["stage_1_eligible"]
            else "stage0_fail_closed"
        ),
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "parent_only_branch_self_parity": parity,
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
        "shift_schedule_manifest_sha256": combined_well_sha(
            frozen_wells, "shift_schedule_sha256"
        ),
        "branch_posterior_manifest_sha256": combined_well_sha(
            frozen_wells, "branch_posterior_sha256"
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
            "event_branch_schedule": schedule_artifact,
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
        "stage_1_eligible": gates["stage_1_eligible"],
        "result": gates["diagnostics"],
        "artifacts": summary["artifacts"],
    }
    metrics_artifact = write_json(metrics_path(), metrics)
    summary["artifacts"]["metrics"] = metrics_artifact
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
                "event": "exp425_stage0_preview",
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "selected_stage": get_nested(CONFIG, "execution.selected_stage"),
                "execution_counts": EXECUTION_COUNTS,
                "scientific_contract": SCIENTIFIC_CONTRACT,
                "stage_0_execution_approved": get_nested(
                    CONFIG, "design.kaggle_stage_0_authorized"
                ),
                "run_hmm": get_nested(
                    CONFIG, "execution.run_hmm"
                ),
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
        print(
            "Stage 0 implementation is ready, but Kaggle execution remains "
            "disabled pending separate user approval.",
            flush=True,
        )
