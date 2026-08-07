# %% [markdown]
# # exp488 isolated GR shock prior hold — train
#
# This Jupytext source is the canonical self-contained support-only Stage A1
# implementation. It preserves the exp482 trigger and row-local intervention while
# removing the zero-shock-well control requirement requested by the user.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contracts
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Raw-only isolated-shock census and support32 manifest
# 4. Saved exp209 control and target-free raw inputs
# 5. Exact exp209 input preparation
# 6. Unchanged exp209 forward-backward message replay
# 7. Leave-one-current-observation-out trigger and prediction freeze
# 8. Truth-late Stage A1 readout
# 9. Technical and scientific gates
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

EXPERIMENT_NAME = "exp488_isolated_gr_shock_prior_hold_support_only"
LINEAGE_PARENT = "exp482_isolated_gr_shock_prior_hold"
SCIENTIFIC_PARENT = "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
FORBIDDEN_TARGET_FREE_COLUMNS = {
    "TVT",
    "tvt_true",
    "target",
    "error",
    "absolute_error",
    "fold",
    "hidden_like_spatial",
    "hidden_like_typewell_purged",
    "persistent_episode",
    "exp440_active",
}


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
        raise ValueError("wrong exp488 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp488 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != LINEAGE_PARENT:
        raise ValueError("exp488 lineage parent changed")
    if not bool(get_nested(config, "design.design_frozen", False)):
        raise ValueError("exp488 scientific design must remain frozen")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp488 implementation is not authorized")
    if bool(get_nested(config, "execution.inference_authorized", True)):
        raise ValueError("exp488 inference must remain disabled")
    if bool(get_nested(config, "execution.submission_authorized", True)):
        raise ValueError("exp488 submission must remain disabled")
    if bool(get_nested(config, "runtime.enable_gpu", True)):
        raise ValueError("exp488 is CPU-only")
    if bool(get_nested(config, "data.exp209_saved_control.regenerate", True)):
        raise ValueError("saved exp209 control must not be regenerated")

    expected = {
        "scientific_variants": 1,
        "stage0_raw_census_wells": 773,
        "stage0_parent_message_hmm_replays": 32,
        "stage1_parent_message_hmm_replays": 773,
        "candidate_state_modifying_hmm_runs": 0,
        "saved_parent_prediction_reruns": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = {key: int(get_nested(config, f"execution.{key}", -1)) for key in expected}
    if observed != expected:
        raise ValueError(f"exp488 execution contract changed: {observed} != {expected}")

    if require_run_authorization:
        if str(get_nested(config, "execution.selected_stage")) != "stage0_support32":
            raise RuntimeError("exp488 candidate only implements stage0_support32")
        if not bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ):
            raise RuntimeError("exp488 execution requires separate canonical notebook adoption")
        if not bool(get_nested(config, "execution.kaggle_package_authorized", False)):
            raise RuntimeError("exp488 execution requires separate Kaggle package approval")
        if not bool(get_nested(config, "execution.stage0_run_authorized", False)):
            raise RuntimeError("exp488 implementation approval does not authorize Stage A0/A1")
        if not bool(get_nested(config, "execution.run_parent_message_hmm", False)):
            raise RuntimeError("exp488 parent message replay remains fail-closed")
        if not bool(get_nested(config, "execution.create_candidate_prediction", False)):
            raise RuntimeError("exp488 candidate prediction remains fail-closed")
        if bool(get_nested(config, "execution.create_submission", True)):
            raise ValueError("exp488 must not create a submission")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    design = dict(get_nested(config, "design") or {})
    if (
        not bool(design.get("independent_from_exp440"))
        or bool(design.get("exp440_schedule_or_truth_used_for_trigger"))
        or int(design.get("scientific_variants", -1)) != 1
        or bool(design.get("same_oof_rescue_allowed"))
        or not bool(design.get("support_only_validation"))
        or bool(design.get("zero_shock_control_required"))
    ):
        raise ValueError("exp488 independent one-candidate design contract changed")
    parent = dict(get_nested(config, "model.parent") or {})
    if parent != {
        "experiment": SCIENTIFIC_PARENT,
        "state_transition_emission_prior_and_readout_changed": False,
    }:
        raise ValueError("exp488 parent-state preservation contract changed")
    fixed = dict(get_nested(config, "model.fixed_from_exp209") or {})
    expected_fixed = {
        "position_grid_step_ft": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "rate_process_sigma": 0.002,
        "position_process_sigma": 0.02,
        "momentum": 0.998,
        "emission_lambda": 1.0,
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "sigma_clip": [10.0, 60.0],
        "start_sigma_ft": 0.75,
        "initial_rate_sigma": 0.01,
        "band_pad_ft": 100.0,
        "rate_center": "zero",
        "state_space": "tvt_position_and_u_rate",
        "transition": "unchanged",
        "prior": "unchanged",
        "emission": "gaussian_typewell_gr",
        "gr_preprocessing_and_missing_policy": "unchanged",
        "sigma_and_clip": "unchanged",
        "forward_backward_reduction_order": "unchanged",
    }
    if fixed != expected_fixed:
        raise ValueError(f"exp209 HMM contract changed: {fixed} != {expected_fixed}")

    raw = dict(get_nested(config, "model.raw_shock") or {})
    expected_raw = {
        "source": "raw_gr_only",
        "raw_observed_required": True,
        "window_radius_rows": 5,
        "current_row_excluded": True,
        "minimum_finite_neighbors_each_side": 3,
        "center": "neighbor_median",
        "scale_formula": "max(1.4826*neighbor_mad,1.0)",
        "robust_z_min": 4.5,
        "left_right_median_difference_scale_max": 2.0,
        "suffix_boundary_exclusion_rows": 5,
        "isolation_radius_rows": 2,
        "collision_policy": "suppress_entire_cluster",
    }
    if raw != expected_raw:
        raise ValueError(f"exp488 raw-shock contract changed: {raw} != {expected_raw}")

    agreement = dict(get_nested(config, "model.message_agreement") or {})
    conflict = dict(get_nested(config, "model.current_emission_conflict") or {})
    trigger = dict(get_nested(config, "model.trigger") or {})
    intervention = dict(get_nested(config, "model.intervention") or {})
    if agreement != {
        "predictive_mean_vs_loo_mean_max_ft": 1.05,
        "predictive_or_loo_std_max_ft": 6.0,
    }:
        raise ValueError("exp488 message-agreement contract changed")
    if conflict != {
        "predictive_to_provisional_mean_shift_min_ft": 1.05,
        "saved_parent_to_loo_output_difference_min_ft": 0.35,
    }:
        raise ValueError("exp488 current-emission conflict contract changed")
    if trigger != {
        "logic": "raw_shock_and_message_agreement_and_current_emission_conflict",
        "uses_true_tvt_error_fold_role_episode_or_exp440_flag": False,
    }:
        raise ValueError("exp488 three-way AND trigger contract changed")
    if intervention != {
        "active_output": "leave_one_current_observation_out_posterior_mean",
        "inactive_output": "saved_exp209_smoothed_posterior_mean",
        "loo_formula": "normalize(predictive_joint*future_beta)",
        "modifies_parent_filtered_state": False,
        "modifies_backward_message": False,
        "modifies_later_predictions": False,
    }:
        raise ValueError("exp488 row-local intervention contract changed")
    return {
        "design": {
            "independent_from_exp440": True,
            "scientific_variants": 1,
            "same_oof_rescue_allowed": False,
        },
        "parent": parent,
        "fixed_from_exp209": fixed,
        "raw_shock": raw,
        "message_agreement": agreement,
        "current_emission_conflict": conflict,
        "trigger": trigger,
        "intervention": intervention,
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
    raise FileNotFoundError("exp488 config.yaml was not found")


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
    if isinstance(value, np.bool_):
        return bool(value)
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
            normalized[column] = normalized[column].fillna("").astype(str)
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
            persisted[column] = persisted[column].fillna("").astype(str)
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
    census_frozen: bool = False
    manifest_frozen: bool = False
    frozen_wells: set[str] = field(default_factory=set)
    forbidden_rows_before_all_freeze: int = 0
    late_truth_rows: int = 0
    late_fold_rows: int = 0
    freeze_records: list[dict[str, str]] = field(default_factory=list)

    @property
    def all_frozen(self) -> bool:
        return (
            self.census_frozen
            and self.manifest_frozen
            and len(self.frozen_wells) == self.expected_wells
        )

    def freeze_census(self, census_sha256: str, shock_rows_sha256: str) -> None:
        if not census_sha256 or not shock_rows_sha256:
            raise ValueError("raw census SHA values are required")
        self.census_frozen = True

    def freeze_manifest(self, manifest_sha256: str) -> None:
        if not self.census_frozen:
            raise RuntimeError("support32 manifest cannot precede raw census freeze")
        if not manifest_sha256:
            raise ValueError("support32 manifest SHA is required")
        self.manifest_frozen = True

    def freeze_well(
        self,
        well: str,
        *,
        message_sha256: str,
        trigger_sha256: str,
        prediction_sha256: str,
    ) -> None:
        if not self.manifest_frozen:
            raise RuntimeError("message replay cannot precede support32 manifest freeze")
        if not message_sha256 or not trigger_sha256 or not prediction_sha256:
            raise ValueError("message, trigger, and prediction SHA values are required")
        self.frozen_wells.add(str(well))
        self.freeze_records.append(
            {
                "well": str(well),
                "message_sha256": message_sha256,
                "trigger_sha256": trigger_sha256,
                "prediction_sha256": prediction_sha256,
            }
        )

    def record_truth_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.forbidden_rows_before_all_freeze += int(rows)
            raise RuntimeError("truth was read before all target-free freeze")
        self.late_truth_rows += int(rows)

    def record_fold_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.forbidden_rows_before_all_freeze += int(rows)
            raise RuntimeError("fold was read before all target-free freeze")
        self.late_fold_rows += int(rows)


# %% [markdown]
# ## 3. Raw-only isolated-shock census and support32 manifest
#
# The census reads only `GR` and `TVT_input`; the latter is used solely to
# identify the unknown suffix. It never opens the true `TVT` column.


# %%
def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.is_dir():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    return find_project_root() / str(get_nested(config, "data.train_dir"))


def discover_train_wells(raw_dir: Path, expected_wells: int | None = None) -> list[str]:
    suffix = "__horizontal_well.csv"
    wells = sorted(
        path.name[: -len(suffix)]
        for path in raw_dir.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    )
    if len(wells) != len(set(wells)):
        raise ValueError("duplicate horizontal-well identities")
    if expected_wells is not None and len(wells) != int(expected_wells):
        raise ValueError(f"train wells={len(wells)}/{int(expected_wells)}")
    return wells


def isolated_raw_shock_diagnostics(
    raw_gr: np.ndarray,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    values = np.asarray(raw_gr, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("raw GR must be one-dimensional")
    count = len(values)
    radius = int(spec["window_radius_rows"])
    minimum_side = int(spec["minimum_finite_neighbors_each_side"])
    boundary = int(spec["suffix_boundary_exclusion_rows"])
    isolation_radius = int(spec["isolation_radius_rows"])
    center = np.full(count, np.nan, dtype=np.float64)
    scale = np.full(count, np.nan, dtype=np.float64)
    robust_z = np.full(count, np.nan, dtype=np.float64)
    left_median = np.full(count, np.nan, dtype=np.float64)
    right_median = np.full(count, np.nan, dtype=np.float64)
    left_count = np.zeros(count, dtype=np.int16)
    right_count = np.zeros(count, dtype=np.int16)
    precluster = np.zeros(count, dtype=bool)

    for index in range(boundary, max(boundary, count - boundary)):
        current = values[index]
        if not np.isfinite(current):
            continue
        left = values[index - radius : index]
        right = values[index + 1 : index + radius + 1]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        left_count[index] = len(left)
        right_count[index] = len(right)
        if len(left) < minimum_side or len(right) < minimum_side:
            continue
        neighbors = np.concatenate([left, right])
        local_center = float(np.median(neighbors))
        local_mad = float(np.median(np.abs(neighbors - local_center)))
        local_scale = max(1.4826 * local_mad, 1.0)
        local_left = float(np.median(left))
        local_right = float(np.median(right))
        local_z = abs(current - local_center) / local_scale
        center[index] = local_center
        scale[index] = local_scale
        robust_z[index] = local_z
        left_median[index] = local_left
        right_median[index] = local_right
        precluster[index] = bool(
            local_z >= float(spec["robust_z_min"])
            and abs(local_left - local_right)
            <= float(spec["left_right_median_difference_scale_max"]) * local_scale
        )

    isolated = precluster.copy()
    candidate_indices = np.flatnonzero(precluster)
    for index in candidate_indices:
        nearby = candidate_indices[
            (candidate_indices >= index - isolation_radius)
            & (candidate_indices <= index + isolation_radius)
        ]
        if len(nearby) > 1:
            isolated[index] = False

    return pd.DataFrame(
        {
            "suffix_offset": np.arange(count, dtype=np.int64),
            "raw_gr_observed": np.isfinite(values),
            "neighbor_center": center,
            "neighbor_scale": scale,
            "robust_z": robust_z,
            "left_median": left_median,
            "right_median": right_median,
            "finite_left_neighbors": left_count,
            "finite_right_neighbors": right_count,
            "raw_shock_precluster": precluster,
            "isolated_raw_shock": isolated,
        }
    )


def load_raw_suffix_for_census(
    well: str,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    path = raw_dir / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(path, usecols=["GR", "TVT_input"])
    suffix = horizontal["TVT_input"].isna().to_numpy(bool)
    if not suffix.any():
        raise ValueError(f"{well}: empty unknown suffix")
    row_idx = horizontal.index.to_numpy(np.int64)[suffix]
    raw_gr = pd.to_numeric(horizontal.loc[suffix, "GR"], errors="coerce").to_numpy(np.float64)
    return row_idx, raw_gr, sha256_file(path)


def build_raw_shock_census(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wells = discover_train_wells(raw_dir, int(get_nested(config, "data.expected_train_wells")))
    spec = get_nested(config, "model.raw_shock")
    census_rows: list[dict[str, Any]] = []
    shock_pieces: list[pd.DataFrame] = []
    for well in wells:
        row_idx, raw_gr, raw_sha = load_raw_suffix_for_census(well, raw_dir)
        diagnostics = isolated_raw_shock_diagnostics(raw_gr, spec)
        diagnostics.insert(0, "row_idx", row_idx)
        diagnostics.insert(0, "well", well)
        selected = diagnostics.loc[
            diagnostics["raw_shock_precluster"],
            [
                "well",
                "row_idx",
                "suffix_offset",
                "robust_z",
                "neighbor_center",
                "neighbor_scale",
                "left_median",
                "right_median",
                "isolated_raw_shock",
            ],
        ]
        if not selected.empty:
            shock_pieces.append(selected)
        census_rows.append(
            {
                "well": well,
                "suffix_rows": len(row_idx),
                "raw_missing_fraction": float(np.mean(~np.isfinite(raw_gr))),
                "raw_shock_precluster_count": int(diagnostics["raw_shock_precluster"].sum()),
                "shock_count": int(diagnostics["isolated_raw_shock"].sum()),
                "horizontal_raw_sha256": raw_sha,
            }
        )
    census = pd.DataFrame(census_rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    shock_rows = (
        pd.concat(shock_pieces, ignore_index=True)
        if shock_pieces
        else pd.DataFrame(
            columns=[
                "well",
                "row_idx",
                "suffix_offset",
                "robust_z",
                "neighbor_center",
                "neighbor_scale",
                "left_median",
                "right_median",
                "isolated_raw_shock",
            ]
        )
    )
    shock_rows = shock_rows.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if (
        len(census) != len(wells)
        or census["well"].nunique() != len(wells)
        or int(census["suffix_rows"].sum()) != int(get_nested(config, "data.expected_suffix_rows"))
    ):
        raise ValueError("raw-only census coverage changed")
    return census, shock_rows


def raw_census_eligibility(
    config: Mapping[str, Any],
    census: pd.DataFrame,
) -> dict[str, Any]:
    spec = get_nested(config, "data.raw_shock_census")
    isolated_rows = int(census["shock_count"].sum())
    support_wells = int((census["shock_count"] > 0).sum())
    zero_shock_wells = int((census["shock_count"] == 0).sum())
    checks = {
        "minimum_isolated_raw_shock_rows": isolated_rows >= int(spec["minimum_rows"]),
        "minimum_support_wells": support_wells >= int(spec["minimum_support_wells"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "isolated_raw_shock_rows": isolated_rows,
        "support_wells": support_wells,
        "zero_shock_control_wells": zero_shock_wells,
    }


def build_support32_manifest(
    config: Mapping[str, Any],
    census: pd.DataFrame,
) -> pd.DataFrame:
    eligibility = raw_census_eligibility(config, census)
    if not eligibility["passed"]:
        raise RuntimeError("raw census eligibility failed; support32 is not permitted")
    spec = get_nested(config, "data.stage0_manifest")
    support_count = int(spec["support_wells"])
    support = (
        census.loc[census["shock_count"] > 0]
        .sort_values(
            ["shock_count", "suffix_rows", "well"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(support_count)
        .copy()
    )
    support["selection_role"] = "support"
    support["matched_support_well"] = support["well"].astype(str)
    support["match_distance"] = 0.0
    manifest = (
        support[
            [
                "well",
                "selection_role",
                "matched_support_well",
                "match_distance",
                "shock_count",
                "raw_shock_precluster_count",
                "suffix_rows",
                "raw_missing_fraction",
                "horizontal_raw_sha256",
            ]
        ]
        .sort_values(["selection_role", "well"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(manifest) != int(spec["total_wells"])
        or manifest["well"].nunique() != int(spec["total_wells"])
        or manifest["selection_role"].value_counts().to_dict() != {"support": support_count}
        or not manifest["shock_count"].gt(0).all()
    ):
        raise ValueError("support32 manifest contract changed")
    return manifest


# %% [markdown]
# ## 4. Saved exp209 control and target-free raw inputs


# %%
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


def load_saved_parent_predictions(
    config: Mapping[str, Any],
    target_wells: set[str],
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
        raise ValueError(f"saved exp209 decompressed SHA changed: {decompressed} != {expected_sha}")
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
        raise ValueError("saved exp209 control has no support32 rows")
    frame = pd.concat(pieces, ignore_index=True).rename(
        columns={prediction_column: "parent_prediction"}
    )
    frame["row_idx"] = parent_row_indices_from_cache_ids(frame)
    frame["parent_prediction"] = pd.to_numeric(frame["parent_prediction"], errors="raise")
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if (
        frame["well"].nunique() != len(target_wells)
        or frame.duplicated(["well", "row_idx"]).any()
        or not np.isfinite(frame["parent_prediction"].to_numpy(np.float64)).all()
    ):
        raise ValueError("saved exp209 support32 coverage mismatch")
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": decompressed,
        "rows": len(frame),
    }


def load_target_free_well(
    well: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=lambda column: str(column) != "TVT",
    )
    forbidden = FORBIDDEN_TARGET_FREE_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: target-free input contains {sorted(forbidden)}")
    typewell = pd.read_csv(typewell_path).sort_values("TVT").reset_index(drop=True)
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError(f"{well}: typewell schema changed")
    return (
        horizontal,
        typewell,
        {
            "horizontal_raw_sha256": sha256_file(horizontal_path),
            "typewell_raw_sha256": sha256_file(typewell_path),
        },
    )


# %% [markdown]
# ## 5. Exact exp209 input preparation


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
        raise ValueError("unknown-suffix true TVT reached HMM preparation")

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
    gr_sigma = float(np.clip(np.nanstd(residual), float(sigma_clip[0]), float(sigma_clip[1])))

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
        "raw_gr": raw_gr,
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
# ## 6. Unchanged exp209 forward-backward message replay
#
# The forward state is the parent exp209 state. During the backward pass,
# `loo_t` is read out from `alpha_t - emission_t + beta_t`. This removes only
# the current observation and never feeds the row-local result into a later row.


# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_parent_and_loo_position_marginals(
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
):
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)

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
    predictive_pos = np.zeros((t_count, p_count), np.float64)
    provisional_pos = np.zeros((t_count, p_count), np.float64)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)

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
                        total += np.exp(prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best)
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
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
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
                            total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    pre_emission_value = best + np.log(total)
                    predictive[p2, r2] = np.float32(pre_emission_value)
                    cur[p2, r2] = np.float32(pre_emission_value + lam * em[t_i, p2])
                else:
                    predictive[p2, r2] = neg
                    cur[p2, r2] = neg

        predictive_best = np.max(predictive)
        provisional_best = np.max(cur)
        predictive_total = 0.0
        provisional_total = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_total += np.exp(predictive[p_i, r_i] - predictive_best)
                provisional_total += np.exp(cur[p_i, r_i] - provisional_best)
        for p_i in range(p_count):
            predictive_mass = 0.0
            provisional_mass = 0.0
            for r_i in range(r_count):
                predictive_mass += np.exp(predictive[p_i, r_i] - predictive_best)
                provisional_mass += np.exp(cur[p_i, r_i] - provisional_best)
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]
            predictive_pos[t_i, p_i] = predictive_mass / predictive_total
            provisional_pos[t_i, p_i] = provisional_mass / provisional_total

    parent_pos = np.zeros((t_count, p_count), np.float64)
    loo_pos = np.zeros((t_count, p_count), np.float64)
    beta_next = np.zeros((p_count, r_count), np.float32)

    parent_values = alpha[t_count - 1] + beta_next
    parent_best = np.max(parent_values)
    loo_best = neg
    for p_i in range(p_count):
        for r_i in range(r_count):
            value = alpha[t_count - 1, p_i, r_i] - lam * em[t_count - 1, p_i]
            if value > loo_best:
                loo_best = value
    parent_total = 0.0
    loo_total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            parent_total += np.exp(parent_values[p_i, r_i] - parent_best)
            loo_total += np.exp(
                alpha[t_count - 1, p_i, r_i] - lam * em[t_count - 1, p_i] - loo_best
            )
    for p_i in range(p_count):
        parent_mass = 0.0
        loo_mass = 0.0
        for r_i in range(r_count):
            parent_mass += np.exp(parent_values[p_i, r_i] - parent_best)
            loo_mass += np.exp(alpha[t_count - 1, p_i, r_i] - lam * em[t_count - 1, p_i] - loo_best)
        parent_pos[t_count - 1, p_i] = parent_mass / parent_total
        loo_pos[t_count - 1, p_i] = loo_mass / loo_total

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)

        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        value = position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2]
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
                    value = rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg

        row = t_i - 1
        parent_best = neg
        loo_best = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                parent_value = alpha[row, p_i, r_i] + beta_cur[p_i, r_i]
                loo_value = alpha[row, p_i, r_i] - lam * em[row, p_i] + beta_cur[p_i, r_i]
                if parent_value > parent_best:
                    parent_best = parent_value
                if loo_value > loo_best:
                    loo_best = loo_value
        parent_total = 0.0
        loo_total = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                parent_total += np.exp(alpha[row, p_i, r_i] + beta_cur[p_i, r_i] - parent_best)
                loo_total += np.exp(
                    alpha[row, p_i, r_i] - lam * em[row, p_i] + beta_cur[p_i, r_i] - loo_best
                )
        for p_i in range(p_count):
            parent_mass = 0.0
            loo_mass = 0.0
            for r_i in range(r_count):
                parent_mass += np.exp(alpha[row, p_i, r_i] + beta_cur[p_i, r_i] - parent_best)
                loo_mass += np.exp(
                    alpha[row, p_i, r_i] - lam * em[row, p_i] + beta_cur[p_i, r_i] - loo_best
                )
            parent_pos[row, p_i] = parent_mass / parent_total
            loo_pos[row, p_i] = loo_mass / loo_total
        for p_i in range(p_count):
            for r_i in range(r_count):
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]

    return parent_pos, predictive_pos, provisional_pos, loo_pos


def position_distribution_summary(
    distribution: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    probabilities = np.asarray(distribution, dtype=np.float64)
    positions = np.asarray(grid, dtype=np.float64)
    normalization = np.sum(probabilities, axis=1, dtype=np.float64)
    if np.any(normalization <= 0.0):
        raise RuntimeError("position distribution lost all mass")
    probabilities = probabilities / normalization[:, None]
    mean = probabilities @ positions
    second = probabilities @ (positions**2)
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    error = float(np.max(np.abs(normalization - 1.0)))
    return mean, std, error


def run_parent_message_replay(
    prepared: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    distributions = _hmm2_parent_and_loo_position_marginals(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(fixed["position_grid_step_ft"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(fixed["rate_process_sigma"]),
        float(fixed["position_process_sigma"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        float(fixed["emission_lambda"]),
        float(fixed["momentum"]),
    )
    names = ("parent", "predictive", "provisional", "loo")
    summaries: dict[str, dict[str, np.ndarray]] = {}
    normalization: dict[str, float] = {}
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    for name, distribution in zip(names, distributions, strict=True):
        mean, std, error = position_distribution_summary(distribution, grid)
        summaries[name] = {"mean": mean, "std": std}
        normalization[name] = error
    message_sha = array_bundle_sha256(
        **{
            f"{name}_position_marginal": np.asarray(distribution, dtype=np.float32)
            for name, distribution in zip(names, distributions, strict=True)
        }
    )
    return {
        "summaries": summaries,
        "normalization": normalization,
        "message_sha256": message_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 7. Leave-one-current-observation-out trigger and prediction freeze


# %%
def build_isolated_shock_trigger(
    config: Mapping[str, Any],
    raw_diagnostics: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, np.ndarray]],
    saved_parent_prediction: np.ndarray,
) -> pd.DataFrame:
    parent = np.asarray(saved_parent_prediction, dtype=np.float64)
    predictive_mean = np.asarray(summaries["predictive"]["mean"], dtype=np.float64)
    predictive_std = np.asarray(summaries["predictive"]["std"], dtype=np.float64)
    provisional_mean = np.asarray(summaries["provisional"]["mean"], dtype=np.float64)
    provisional_std = np.asarray(summaries["provisional"]["std"], dtype=np.float64)
    replay_parent_mean = np.asarray(summaries["parent"]["mean"], dtype=np.float64)
    replay_parent_std = np.asarray(summaries["parent"]["std"], dtype=np.float64)
    loo_mean = np.asarray(summaries["loo"]["mean"], dtype=np.float64)
    loo_std = np.asarray(summaries["loo"]["std"], dtype=np.float64)
    expected_rows = len(raw_diagnostics)
    arrays = (
        parent,
        predictive_mean,
        predictive_std,
        provisional_mean,
        provisional_std,
        replay_parent_mean,
        replay_parent_std,
        loo_mean,
        loo_std,
    )
    if any(len(values) != expected_rows for values in arrays):
        raise ValueError("exp488 trigger input length mismatch")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("exp488 trigger received non-finite message summaries")

    agreement = get_nested(config, "model.message_agreement")
    conflict = get_nested(config, "model.current_emission_conflict")
    predictive_loo_difference = np.abs(loo_mean - predictive_mean)
    predictive_provisional_shift = np.abs(provisional_mean - predictive_mean)
    parent_loo_difference = np.abs(parent - loo_mean)
    uncertainty_max = np.maximum(predictive_std, loo_std)
    message_agreement = (
        predictive_loo_difference <= float(agreement["predictive_mean_vs_loo_mean_max_ft"])
    ) & (uncertainty_max <= float(agreement["predictive_or_loo_std_max_ft"]))
    emission_conflict = (
        predictive_provisional_shift
        >= float(conflict["predictive_to_provisional_mean_shift_min_ft"])
    ) & (parent_loo_difference >= float(conflict["saved_parent_to_loo_output_difference_min_ft"]))
    isolated = raw_diagnostics["isolated_raw_shock"].to_numpy(bool)
    active = isolated & message_agreement & emission_conflict
    candidate = np.where(active, loo_mean, parent)
    return pd.DataFrame(
        {
            "suffix_offset": raw_diagnostics["suffix_offset"].to_numpy(np.int64),
            "raw_gr_observed": raw_diagnostics["raw_gr_observed"].to_numpy(bool),
            "raw_shock_precluster": raw_diagnostics["raw_shock_precluster"].to_numpy(bool),
            "isolated_raw_shock": isolated,
            "message_agreement": message_agreement,
            "current_emission_conflict": emission_conflict,
            "trigger_active": active,
            "robust_z": raw_diagnostics["robust_z"].to_numpy(np.float64),
            "predictive_mean": predictive_mean,
            "predictive_std": predictive_std,
            "provisional_mean": provisional_mean,
            "provisional_std": provisional_std,
            "loo_mean": loo_mean,
            "loo_std": loo_std,
            "replay_parent_mean": replay_parent_mean,
            "replay_parent_std": replay_parent_std,
            "saved_parent_prediction": parent,
            "predictive_loo_difference_ft": predictive_loo_difference,
            "predictive_provisional_shift_ft": predictive_provisional_shift,
            "saved_parent_loo_difference_ft": parent_loo_difference,
            "candidate_prediction": candidate,
        }
    )


def expected_census_shock_rows(
    well: str,
    shock_rows: pd.DataFrame,
) -> pd.DataFrame:
    selected = shock_rows.loc[shock_rows["well"].astype(str).eq(str(well))].copy()
    return selected.sort_values("row_idx", kind="mergesort").reset_index(drop=True)


def validate_census_replay(
    well: str,
    row_idx: np.ndarray,
    diagnostics: pd.DataFrame,
    frozen_shock_rows: pd.DataFrame,
) -> None:
    replay = diagnostics.copy()
    replay.insert(0, "row_idx", np.asarray(row_idx, dtype=np.int64))
    replay.insert(0, "well", str(well))
    replay = replay.loc[
        replay["raw_shock_precluster"],
        [
            "well",
            "row_idx",
            "suffix_offset",
            "robust_z",
            "neighbor_center",
            "neighbor_scale",
            "left_median",
            "right_median",
            "isolated_raw_shock",
        ],
    ].reset_index(drop=True)
    expected = expected_census_shock_rows(well, frozen_shock_rows)
    if logical_frame_sha256(replay) != logical_frame_sha256(expected):
        raise ValueError(f"{well}: raw shock replay changed after census freeze")


@dataclass(frozen=True)
class FrozenWell:
    well: str
    selection_role: str
    prediction: pd.DataFrame
    trigger: pd.DataFrame
    diagnostic: pd.DataFrame
    audit: dict[str, Any]


def freeze_target_free_well(
    config: Mapping[str, Any],
    well: str,
    selection_role: str,
    raw_dir: Path,
    parent_well: pd.DataFrame,
    frozen_shock_rows: pd.DataFrame,
    expected_horizontal_sha256: str,
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell, raw_identity = load_target_free_well(well, raw_dir)
    if raw_identity["horizontal_raw_sha256"] != expected_horizontal_sha256:
        raise ValueError(f"{well}: horizontal raw SHA changed after census")
    fixed = get_nested(config, "model.fixed_from_exp209")
    prepared = prepare_hmm_inputs(horizontal, typewell, fixed)
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    parent_well = parent_well.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        parent_well["row_idx"].to_numpy(np.int64),
        row_idx,
    ):
        raise ValueError(f"{well}: saved parent row identity mismatch")
    raw_diagnostics = isolated_raw_shock_diagnostics(
        np.asarray(prepared["raw_gr"], dtype=np.float64),
        get_nested(config, "model.raw_shock"),
    )
    validate_census_replay(well, row_idx, raw_diagnostics, frozen_shock_rows)
    replay = run_parent_message_replay(prepared, fixed)
    message_sha = hashlib.sha256(
        stable_json_bytes(
            {
                "kernel_message_sha256": replay["message_sha256"],
                "horizontal_raw_sha256": raw_identity["horizontal_raw_sha256"],
                "typewell_raw_sha256": raw_identity["typewell_raw_sha256"],
            }
        )
    ).hexdigest()
    trigger = build_isolated_shock_trigger(
        config,
        raw_diagnostics,
        replay["summaries"],
        parent_well["parent_prediction"].to_numpy(np.float64),
    )
    trigger.insert(0, "row_idx", row_idx)
    trigger.insert(0, "well", well)
    trigger_sha = logical_frame_sha256(trigger)

    prediction = trigger[
        [
            "well",
            "row_idx",
            "saved_parent_prediction",
            "candidate_prediction",
            "trigger_active",
        ]
    ].rename(columns={"saved_parent_prediction": "parent_prediction"})
    prediction_sha = logical_frame_sha256(prediction)
    diagnostic = trigger[
        [
            "well",
            "row_idx",
            "raw_gr_observed",
            "raw_shock_precluster",
            "isolated_raw_shock",
            "message_agreement",
            "current_emission_conflict",
            "trigger_active",
            "robust_z",
            "predictive_mean",
            "predictive_std",
            "provisional_mean",
            "provisional_std",
            "loo_mean",
            "loo_std",
            "replay_parent_mean",
            "replay_parent_std",
            "predictive_loo_difference_ft",
            "predictive_provisional_shift_ft",
            "saved_parent_loo_difference_ft",
        ]
    ]
    ledger.freeze_well(
        well,
        message_sha256=message_sha,
        trigger_sha256=trigger_sha,
        prediction_sha256=prediction_sha,
    )
    parity = np.abs(
        trigger["replay_parent_mean"].to_numpy(np.float64)
        - trigger["saved_parent_prediction"].to_numpy(np.float64)
    )
    audit = {
        "well": well,
        "selection_role": selection_role,
        "rows": len(trigger),
        "raw_shock_rows": int(trigger["isolated_raw_shock"].sum()),
        "trigger_rows": int(trigger["trigger_active"].sum()),
        "parent_replay_prediction_max_abs_error_ft": float(np.max(parity)),
        "maximum_normalization_error": float(max(replay["normalization"].values())),
        "finite_coverage": float(
            np.isfinite(
                trigger[
                    [
                        "predictive_mean",
                        "predictive_std",
                        "provisional_mean",
                        "provisional_std",
                        "loo_mean",
                        "loo_std",
                        "replay_parent_mean",
                        "candidate_prediction",
                    ]
                ].to_numpy(np.float64)
            ).mean()
        ),
        "elapsed_seconds": float(replay["elapsed_seconds"]),
        "message_sha256": message_sha,
        "trigger_sha256": trigger_sha,
        "prediction_sha256": prediction_sha,
        **raw_identity,
    }
    return FrozenWell(
        well=well,
        selection_role=selection_role,
        prediction=prediction,
        trigger=trigger,
        diagnostic=diagnostic,
        audit=audit,
    )


def concatenate_frozen(
    frozen: Sequence[FrozenWell],
    attribute: str,
) -> pd.DataFrame:
    pieces = [getattr(item, attribute) for item in frozen]
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


# %% [markdown]
# ## 8. Truth-late Stage A1 readout


# %%
def load_truth_after_all_freeze(
    raw_dir: Path,
    target_wells: Sequence[str],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for well in sorted(str(value) for value in target_wells):
        horizontal = pd.read_csv(
            raw_dir / f"{well}__horizontal_well.csv",
            usecols=["TVT", "TVT_input"],
        )
        suffix = horizontal["TVT_input"].isna().to_numpy(bool)
        frame = pd.DataFrame(
            {
                "well": well,
                "row_idx": horizontal.index.to_numpy(np.int64)[suffix],
                "tvt_true": pd.to_numeric(horizontal.loc[suffix, "TVT"], errors="raise").to_numpy(
                    np.float64
                ),
            }
        )
        pieces.append(frame)
    truth = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    ledger.record_truth_late(len(truth))
    if not np.isfinite(truth["tvt_true"].to_numpy(np.float64)).all():
        raise ValueError("truth-late suffix contains non-finite values")
    return truth


def load_folds_after_all_freeze(
    config: Mapping[str, Any],
    target_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.fold_source")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(value) for value in spec["candidates"]],
        patterns=[str(value) for value in spec["patterns"]],
    )
    decompressed = sha256_decompressed_csv(path)
    expected = str(spec["expected_decompressed_sha256"])
    if decompressed != expected:
        raise ValueError(f"fold source decompressed SHA changed: {decompressed}")
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["well_id", "row_idx", "fold"],
        dtype={"well_id": str},
        chunksize=250_000,
    ):
        selected = chunk.loc[chunk["well_id"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True).rename(columns={"well_id": "well"})
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    ledger.record_fold_late(len(frame))
    if frame["well"].nunique() != len(target_wells) or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("truth-late fold identity coverage mismatch")
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": decompressed,
        "rows": len(frame),
    }


def build_truth_late_readout(
    config: Mapping[str, Any],
    raw_dir: Path,
    prediction: pd.DataFrame,
    diagnostic: pd.DataFrame,
    manifest: pd.DataFrame,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_wells = set(manifest["well"].astype(str))
    truth = load_truth_after_all_freeze(raw_dir, sorted(target_wells), ledger)
    folds, fold_input = load_folds_after_all_freeze(config, target_wells, ledger)
    frame = prediction.merge(
        diagnostic[
            [
                "well",
                "row_idx",
                "raw_gr_observed",
                "isolated_raw_shock",
                "message_agreement",
                "current_emission_conflict",
            ]
        ],
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        manifest[["well", "selection_role"]],
        on="well",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        truth,
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    frame = (
        frame.merge(
            folds,
            on=["well", "row_idx"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(frame) != len(prediction)
        or frame["well"].nunique() != len(target_wells)
        or frame.duplicated(["well", "row_idx"]).any()
        or sorted(frame["fold"].unique().tolist()) != [0, 1, 2, 3, 4]
        or not np.isfinite(
            frame[["parent_prediction", "candidate_prediction", "tvt_true"]].to_numpy(np.float64)
        ).all()
    ):
        raise ValueError("exp488 truth-late readout coverage mismatch")
    return frame, {
        "truth_attachment": ("after_raw_census_manifest_message_trigger_prediction_sha_freeze"),
        "fold_input": fold_input,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "forbidden_reads_before_all_freeze": (ledger.forbidden_rows_before_all_freeze),
    }


def rmse(values: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(values, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def metric_row(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    truth = frame["tvt_true"].to_numpy(np.float64)
    parent = frame["parent_prediction"].to_numpy(np.float64)
    candidate = frame["candidate_prediction"].to_numpy(np.float64)
    parent_rmse = rmse(parent, truth)
    candidate_rmse = rmse(candidate, truth)
    return {
        "scope": scope,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "parent_rmse_ft": parent_rmse,
        "candidate_rmse_ft": candidate_rmse,
        "rmse_delta_candidate_minus_parent_ft": candidate_rmse - parent_rmse,
        "improvement_ft": parent_rmse - candidate_rmse,
    }


def build_stage0_metrics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = frame.loc[frame["selection_role"].eq("support")]
    rows = [metric_row(support, "support32")]
    for fold in range(5):
        rows.append(metric_row(support.loc[support["fold"].eq(fold)], f"fold_{fold}"))
    by_well_rows: list[dict[str, Any]] = []
    for well, selected in support.groupby("well", sort=True):
        parent_rmse = rmse(
            selected["parent_prediction"].to_numpy(np.float64),
            selected["tvt_true"].to_numpy(np.float64),
        )
        candidate_rmse = rmse(
            selected["candidate_prediction"].to_numpy(np.float64),
            selected["tvt_true"].to_numpy(np.float64),
        )
        by_well_rows.append(
            {
                "well": str(well),
                "fold": int(selected["fold"].iloc[0]),
                "rows": len(selected),
                "parent_rmse_ft": parent_rmse,
                "candidate_rmse_ft": candidate_rmse,
                "rmse_delta_candidate_minus_parent_ft": (candidate_rmse - parent_rmse),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(by_well_rows)


# %% [markdown]
# ## 9. Technical and scientific gates


# %%
def sse_reduction(parent_sse: float, candidate_sse: float) -> float:
    if not math.isfinite(parent_sse) or parent_sse <= 0.0:
        return float("nan")
    return float((parent_sse - candidate_sse) / parent_sse)


def evaluate_stage0_gates(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    audit: pd.DataFrame,
    ledger: LeakageLedger,
    *,
    elapsed_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "validation.stage0.technical")
    scientific_config = get_nested(config, "validation.stage0.scientific")
    active = frame["trigger_active"].to_numpy(bool)
    active_rows = int(np.count_nonzero(active))
    active_wells = int(frame.loc[active, "well"].nunique())
    active_folds = int(frame.loc[active, "fold"].nunique())
    trigger_fraction = float(active_rows / len(frame))
    projected_runtime = float(
        audit["elapsed_seconds"].sum()
        / int(get_nested(config, "validation.stage0.unchanged_parent_message_hmm_replays"))
        * int(get_nested(config, "validation.stage1.unchanged_parent_message_hmm_replays"))
    )
    technical = {
        "expected_wells": frame["well"].nunique() == int(technical_config["expected_wells"]),
        "expected_support_wells": int(manifest["selection_role"].eq("support").sum())
        == int(technical_config["expected_support_wells"]),
        "support_only_manifest": manifest["selection_role"].eq("support").all(),
        "well_overlap_zero": manifest["well"].nunique() == len(manifest),
        "parent_replay_prediction_max_abs_error_ft": float(
            audit["parent_replay_prediction_max_abs_error_ft"].max()
        )
        <= float(technical_config["parent_replay_prediction_max_abs_error_ft"]),
        "posterior_normalization_max_error": float(audit["maximum_normalization_error"].max())
        <= float(technical_config["posterior_normalization_max_error"]),
        "finite_coverage_min": float(audit["finite_coverage"].min())
        >= float(technical_config["finite_coverage_min"]),
        "final_trigger_rows_min": active_rows >= int(technical_config["final_trigger_rows_min"]),
        "final_trigger_wells_min": active_wells >= int(technical_config["final_trigger_wells_min"]),
        "final_trigger_folds_required": active_folds
        == int(technical_config["final_trigger_folds_required"]),
        "final_trigger_fraction_max": trigger_fraction
        <= float(technical_config["final_trigger_fraction_max"]),
        "forbidden_reads_before_freeze": ledger.forbidden_rows_before_all_freeze
        <= int(technical_config["forbidden_reads_before_freeze"]),
        "full_runtime_projection_seconds_max": projected_runtime
        <= float(technical_config["full_runtime_projection_seconds_max"]),
        "peak_rss_gb_max": rss_gb <= float(technical_config["peak_rss_gb_max"]),
    }

    truth = frame["tvt_true"].to_numpy(np.float64)
    parent = frame["parent_prediction"].to_numpy(np.float64)
    candidate = frame["candidate_prediction"].to_numpy(np.float64)
    parent_active_error = np.abs(parent[active] - truth[active])
    candidate_active_error = np.abs(candidate[active] - truth[active])
    better_fraction = (
        float(np.mean(candidate_active_error < parent_active_error))
        if active_rows
        else float("nan")
    )
    active_sse_reduction = sse_reduction(
        float(np.sum((parent[active] - truth[active]) ** 2)),
        float(np.sum((candidate[active] - truth[active]) ** 2)),
    )
    fold_metrics = metrics.loc[metrics["scope"].str.startswith("fold_")]
    improving_folds = int((fold_metrics["improvement_ft"] > 0.0).sum())
    support_metric = metrics.loc[metrics["scope"].eq("support32")].iloc[0]
    well_delta = by_well["rmse_delta_candidate_minus_parent_ft"].to_numpy(np.float64)
    by_well_p95 = float(np.quantile(well_delta, 0.95))
    worst_well = float(np.max(well_delta))
    scientific = {
        "candidate_better_fraction_on_trigger_rows": better_fraction,
        "candidate_better_fraction_on_trigger_rows_min": float(
            scientific_config["candidate_better_fraction_on_trigger_rows_min"]
        ),
        "trigger_row_sse_reduction_fraction": active_sse_reduction,
        "trigger_row_sse_reduction_fraction_min": float(
            scientific_config["trigger_row_sse_reduction_fraction_min"]
        ),
        "improving_folds": improving_folds,
        "improving_folds_min": int(scientific_config["improving_folds_min"]),
        "support_pooled_rmse_improvement_ft": float(support_metric["improvement_ft"]),
        "support_pooled_rmse_improvement_ft_min": float(
            scientific_config["support_pooled_rmse_improvement_ft_min"]
        ),
        "by_well_delta_p95_ft": by_well_p95,
        "by_well_delta_p95_ft_max": float(scientific_config["by_well_delta_p95_ft_max"]),
        "worst_well_regression_ft": worst_well,
        "worst_well_regression_ft_max": float(scientific_config["worst_well_regression_ft_max"]),
    }
    technical_passed = bool(all(technical.values()))
    scientific_passed = bool(
        math.isfinite(better_fraction)
        and better_fraction >= scientific["candidate_better_fraction_on_trigger_rows_min"]
        and math.isfinite(active_sse_reduction)
        and active_sse_reduction >= scientific["trigger_row_sse_reduction_fraction_min"]
        and improving_folds >= scientific["improving_folds_min"]
        and scientific["support_pooled_rmse_improvement_ft"]
        >= scientific["support_pooled_rmse_improvement_ft_min"]
        and by_well_p95 <= scientific["by_well_delta_p95_ft_max"]
        and worst_well <= scientific["worst_well_regression_ft_max"]
    )
    passed = bool(technical_passed and scientific_passed)
    return {
        "passed": passed,
        "technical_passed": technical_passed,
        "scientific_passed": scientific_passed,
        "technical": technical,
        "scientific": scientific,
        "observed": {
            "trigger_rows": active_rows,
            "trigger_wells": active_wells,
            "trigger_folds": active_folds,
            "trigger_fraction": trigger_fraction,
            "projected_full_runtime_seconds": projected_runtime,
            "stage0_elapsed_seconds": float(elapsed_seconds),
            "peak_rss_gb": float(rss_gb),
        },
        "decision": (
            "stage0_passed_requires_separate_stage1_approval"
            if passed
            else str(get_nested(config, "validation.stage0.fail_action"))
        ),
    }


# %% [markdown]
# ## 10. Guarded Kaggle CPU orchestration


# %%
def require_kaggle_runtime() -> None:
    if not KAGGLE_INPUT_ROOT.is_dir() or not KAGGLE_WORKING_ROOT.is_dir():
        raise RuntimeError(
            "exp488 authoritative execution is Kaggle-only; local execution is disabled"
        )


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_execution_contract(config, require_run_authorization=True)
    scientific_contract = validate_scientific_contract(config)
    set_num_threads(int(get_nested(config, "runtime.numba_threads")))
    output = artifacts_dir()
    raw_dir = train_data_dir(config)

    census, shock_rows = build_raw_shock_census(config, raw_dir)
    census_artifact = write_csv(output / f"{EXPERIMENT_NAME}_raw_shock_census.csv", census)
    shock_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_raw_shock_rows.csv.gz", shock_rows
    )
    eligibility = raw_census_eligibility(config, census)
    ledger = LeakageLedger(
        expected_wells=int(get_nested(config, "data.stage0_manifest.total_wells"))
    )
    ledger.freeze_census(
        str(census_artifact["logical_sha256"]),
        str(shock_artifact["logical_sha256"]),
    )
    if not eligibility["passed"]:
        status = "stage_a0_eligibility_failed_closed"
        summary = {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": status,
            "stage": "stage_a0_raw_only_census",
            "execution_contract": execution_contract,
            "scientific_contract": scientific_contract,
            "eligibility": eligibility,
            "counts": {
                "raw_census_wells": len(census),
                "parent_message_hmm_replays": 0,
                "candidate_state_modifying_hmm_runs": 0,
                "saved_parent_prediction_reruns": 0,
                "lightgbm_configs": 0,
                "boosters": 0,
                "pf_runs": 0,
                "beam_runs": 0,
                "gpu_runs": 0,
            },
            "artifacts": {
                "raw_census": census_artifact,
                "raw_shock_rows": shock_artifact,
            },
            "inference": False,
            "submission": False,
        }
        write_json(output / f"{EXPERIMENT_NAME}_stage0_summary.json", summary)
        write_json(
            metrics_path(),
            {
                "experiment": EXPERIMENT_NAME,
                "route": "pf_beam",
                "status": status,
                "validation": {"stage": "stage_a0", "cv": None, "metric": "rmse"},
                "eligibility": eligibility,
            },
        )
        print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
        return summary

    manifest = build_support32_manifest(config, census)
    manifest_artifact = write_csv(output / f"{EXPERIMENT_NAME}_support32_manifest.csv", manifest)
    ledger.freeze_manifest(str(manifest_artifact["logical_sha256"]))
    target_wells = set(manifest["well"].astype(str))
    parent, parent_input = load_saved_parent_predictions(config, target_wells)
    expected_rows = int(manifest["suffix_rows"].sum())
    if len(parent) != expected_rows:
        raise ValueError(f"saved parent rows={len(parent)}/{expected_rows}")

    frozen: list[FrozenWell] = []
    for row in manifest.sort_values("well", kind="mergesort").itertuples(index=False):
        parent_well = parent.loc[parent["well"].astype(str).eq(str(row.well))]
        frozen.append(
            freeze_target_free_well(
                config,
                str(row.well),
                str(row.selection_role),
                raw_dir,
                parent_well,
                shock_rows,
                str(row.horizontal_raw_sha256),
                ledger,
            )
        )
    if not ledger.all_frozen:
        raise RuntimeError("exp488 truth-late phase reached before all support32 freeze")

    prediction = concatenate_frozen(frozen, "prediction")
    trigger = concatenate_frozen(frozen, "trigger")
    diagnostic = concatenate_frozen(frozen, "diagnostic")
    audit = (
        pd.DataFrame([item.audit for item in frozen])
        .sort_values("well", kind="mergesort")
        .reset_index(drop=True)
    )
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_predictions.csv.gz", prediction
    )
    trigger_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_trigger_schedule.csv.gz", trigger
    )
    diagnostic_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_message_diagnostics.csv.gz",
        diagnostic,
    )
    audit_artifact = write_csv(output / f"{EXPERIMENT_NAME}_stage0_well_audit.csv", audit)
    frame, late_attachment = build_truth_late_readout(
        config,
        raw_dir,
        prediction,
        diagnostic,
        manifest,
        ledger,
    )
    metrics, by_well = build_stage0_metrics(frame)
    elapsed = float(time.perf_counter() - started)
    rss = peak_rss_gb()
    gates = evaluate_stage0_gates(
        config,
        frame,
        manifest,
        metrics,
        by_well,
        audit,
        ledger,
        elapsed_seconds=elapsed,
        rss_gb=rss,
    )
    metrics_artifact = write_csv(output / f"{EXPERIMENT_NAME}_stage0_scope_metrics.csv", metrics)
    by_well_artifact = write_csv(output / f"{EXPERIMENT_NAME}_stage0_by_well_metrics.csv", by_well)
    gates_artifact = write_json(output / f"{EXPERIMENT_NAME}_stage0_gates.json", gates)
    status = (
        "stage0_passed_requires_separate_stage1_approval"
        if gates["passed"]
        else "stage0_failed_closed"
    )
    support_metric = metrics.loc[metrics["scope"].eq("support32")].iloc[0]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage_a0_a1_support32",
        "execution_contract": execution_contract,
        "scientific_contract_sha256": hashlib.sha256(
            stable_json_bytes(scientific_contract)
        ).hexdigest(),
        "eligibility": eligibility,
        "counts": {
            "raw_census_wells": len(census),
            "support32_wells": len(manifest),
            "support_wells": int(manifest["selection_role"].eq("support").sum()),
            "control_wells": int(manifest["selection_role"].eq("control").sum()),
            "rows": len(frame),
            "trigger_rows": int(frame["trigger_active"].sum()),
            "scientific_variants": 1,
            "parent_message_hmm_replays": len(frozen),
            "candidate_state_modifying_hmm_runs": 0,
            "saved_parent_prediction_reruns": 0,
            "lightgbm_configs": 0,
            "trained_ml_folds": 0,
            "boosters": 0,
            "fitted_models": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
        "mechanism_readout": {
            "metric": "rmse",
            "support_candidate_rmse_ft": float(support_metric["candidate_rmse_ft"]),
            "support_parent_rmse_ft": float(support_metric["parent_rmse_ft"]),
            "support_improvement_ft": float(support_metric["improvement_ft"]),
            "support32_is_not_cv_or_promotion_evidence": True,
        },
        "gates": gates,
        "late_attachment": late_attachment,
        "target_free_sha": {
            "raw_census": census_artifact["logical_sha256"],
            "raw_shock_rows": shock_artifact["logical_sha256"],
            "manifest": manifest_artifact["logical_sha256"],
            "message_bundle": logical_frame_sha256(audit[["well", "message_sha256"]]),
            "message_diagnostics": diagnostic_artifact["logical_sha256"],
            "trigger": trigger_artifact["logical_sha256"],
            "prediction": prediction_artifact["logical_sha256"],
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": rss,
            "versions": runtime_versions(),
        },
        "artifacts": {
            "raw_census": census_artifact,
            "raw_shock_rows": shock_artifact,
            "manifest": manifest_artifact,
            "parent_input": parent_input,
            "prediction": prediction_artifact,
            "trigger": trigger_artifact,
            "diagnostic": diagnostic_artifact,
            "well_audit": audit_artifact,
            "scope_metrics": metrics_artifact,
            "by_well_metrics": by_well_artifact,
            "gates": gates_artifact,
        },
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(output / f"{EXPERIMENT_NAME}_stage0_summary.json", summary)
    summary["artifacts"]["summary"] = summary_artifact
    write_json(
        metrics_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": status,
            "validation": {
                "strategy": get_nested(config, "validation.strategy"),
                "stage": "stage_a0_a1_support32",
                "metric": "rmse",
                "cv": None,
                "mechanism_candidate_rmse": float(support_metric["candidate_rmse_ft"]),
                "mechanism_parent_rmse": float(support_metric["parent_rmse_ft"]),
                "public_lb": None,
                "private_lb": None,
            },
            "eligibility": eligibility,
            "gates": gates,
            "target_free_sha": summary["target_free_sha"],
            "artifacts": summary["artifacts"],
            "deterministic_anchor": False,
        },
    )
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    runtime_config = load_config()
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(runtime_config, "experiment.route"),
                "status": get_nested(runtime_config, "experiment.status"),
                "parent": get_nested(runtime_config, "lineage.parent"),
                "scientific_variants": get_nested(runtime_config, "execution.scientific_variants"),
                "stage0_parent_message_hmm_replays": get_nested(
                    runtime_config, "execution.stage0_parent_message_hmm_replays"
                ),
                "candidate_state_modifying_hmm_runs": 0,
                "saved_parent_prediction_reruns": 0,
                "lightgbm_configs": 0,
                "boosters": 0,
                "pf_runs": 0,
                "beam_runs": 0,
                "gpu_runs": 0,
            },
            sort_keys=True,
        )
    )
    run_stage0(runtime_config)
