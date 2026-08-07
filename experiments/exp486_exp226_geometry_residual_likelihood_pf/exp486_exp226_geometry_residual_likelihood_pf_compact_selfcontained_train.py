# %% [markdown]
# # exp486 exp226 geometry/residual likelihood-PF — train
#
# This train-side implementation evaluates two preregistered PF variants.
# Variant A adds an absolute exp226 geometry unary to the unchanged exp404
# position/rate filter. Variant B expresses TVT as `tvt_geop + offset` and
# filters a slow `(offset, offset_rate)` state. Stage 0 is the completed
# fixed32 preflight. User-approved Stage 1 runs both variants on all 773 wells
# under a recorded runtime exception. Predictions and target-free mechanism
# ledgers freeze before truth, saved controls, roles, or folds are read.

# %% [markdown]
# ## Contents
# 1. Imports and notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Fixed32 scope, exp226 allowlist, and leakage ledger
# 5. Exp404 likelihood-PF input preparation
# 6. Absolute-unary and residual-offset PF kernels
# 7. Seed aggregation, parity, and state-transition contracts
# 8. Target-free two-variant generation and freeze
# 9. Truth-late fixed32 readout and fail-closed gates
# 10. Generated artifacts and guarded Stage 0 orchestration
# 11. All-well Stage 1 truth-late CV and independent promotion gates
# 12. Setup and configuration preview

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


EXPERIMENT_NAME = "exp486_exp226_geometry_residual_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
ABSOLUTE_VARIANT = "absolute_geometry_unary_sigma20_lambda050"
RESIDUAL_VARIANT = "slow_residual_offset_state"
ABSOLUTE_PREDICTION = "likpf_scale5_absolute_geometry_unary"
RESIDUAL_PREDICTION = "likpf_scale5_slow_residual_offset"
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
ACTIVE_VARIANTS = (ABSOLUTE_VARIANT, RESIDUAL_VARIANT)
PREDICTION_COLUMNS = (ABSOLUTE_PREDICTION, RESIDUAL_PREDICTION)
GEOMETRY_ALLOWLIST = ("well_id", "row_idx", "suffix_offset", "tvt_geop")
ABSOLUTE_LEDGER_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "geometry_residual_mean",
    "geometry_residual_std",
    "geometry_log_factor_mean",
    "effective_sample_size",
    "resampled_seed_fraction",
)
RESIDUAL_LEDGER_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "geometry_delta",
    "filtered_offset_mean",
    "filtered_offset_std",
    "filtered_offset_rate_mean",
    "filtered_offset_rate_std",
    "particle_drift_minus_geometry_delta",
    "typewell_support_fraction",
    "effective_sample_size",
    "resampled_seed_fraction",
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP486_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Notebook-safe configuration, path, and SHA helpers


# %%
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
        item = float(value)
        return item if math.isfinite(item) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def candidate_package_dirs() -> list[Path]:
    root = project_root()
    candidates = [
        Path.cwd(),
        root / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    return candidates


def load_experiment_config(package_dir: Path | None = None) -> dict[str, Any]:
    candidates = [package_dir] if package_dir is not None else candidate_package_dirs()
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "config.yaml"
        checked.append(str(path))
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp486 config not found; checked={checked}")


def artifact_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
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
    return project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_csv_payload(path: str | Path) -> str:
    selected = Path(path)
    return sha256_decompressed_csv(selected) if selected.suffix == ".gz" else sha256_path(selected)


def dataframe_content_sha(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    chosen = list(columns)
    digest = hashlib.sha256()
    chunksize = 200_000
    for start in range(0, len(frame), chunksize):
        payload = frame.loc[
            frame.index[start : start + chunksize],
            chosen,
        ].to_csv(
            index=False,
            header=start == 0,
            lineterminator="\n",
        )
        digest.update(payload.encode("utf-8"))
    if len(frame) == 0:
        digest.update(
            pd.DataFrame(columns=chosen).to_csv(index=False, lineterminator="\n").encode("utf-8")
        )
    return digest.hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return mapping_sha256(schema)


def typed_dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(str(column).encode())
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


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "raw_sha256": sha256_path(path)}


def write_deterministic_gzip_csv(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            frame.to_csv(zipped, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": frame.columns.astype(str).tolist(),
        "schema_sha256": dataframe_schema_sha(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
    }


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if platform.system() != "Darwin" else 1024.0**3
    return value / divisor


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "numba_available": str(NUMBA_AVAILABLE),
    }
    if NUMBA_AVAILABLE:
        import numba

        versions["numba"] = numba.__version__
    return versions


def resolve_existing(
    filename: str,
    candidates: Iterable[str],
    patterns: Iterable[str] = (),
) -> Path:
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        direct = candidate if candidate.name == filename else candidate / filename
        checked.append(str(direct))
        if direct.exists():
            return direct
        if candidate.exists() and candidate.is_dir():
            for pattern in patterns:
                matches = sorted(candidate.glob(str(pattern)))
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    raise ValueError(f"ambiguous {filename}: {matches}")
    root = project_root()
    local_matches = sorted(root.glob(f"**/{filename}"))
    if len(local_matches) == 1:
        return local_matches[0]
    checked.extend(str(path) for path in local_matches)
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = [
        Path.cwd() / "assets" / filename,
        project_root() / local_path,
        KAGGLE_WORKING_ROOT / "assets" / filename,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    matches = [candidate for candidate in candidates if candidate.exists()]
    if not matches:
        raise FileNotFoundError(f"bootstrap asset not found: {filename}")
    return matches[0]


# %% [markdown]
# ## 3. Frozen scientific and execution contracts


# %%
def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, int]:
    counts = {
        "scientific_variants": 2,
        "stage_0_candidate_pf_well_runs": 64,
        "stage_0_seed_well_trajectories": 8192,
        "stage_0_particle_starts": 4096000,
        "stage_1_candidate_pf_well_runs": 1546,
        "stage_1_seed_well_trajectories": 197888,
        "stage_1_particle_starts": 98944000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    for key, expected in counts.items():
        observed = int(get_nested(config, f"execution.{key}", -1))
        if observed != expected:
            raise ValueError(
                f"exp486 execution count changed: {key}={observed}, expected={expected}"
            )
    run_stage0 = bool(get_nested(config, "execution.run_stage_0", False))
    run_stage1 = bool(get_nested(config, "execution.run_stage_1", False))
    if run_stage0 and run_stage1:
        raise ValueError("exp486 permits exactly one active execution stage")
    if run_stage1:
        if not bool(get_nested(config, "stage_0_result.runtime_exception.approved")):
            raise RuntimeError("exp486 Stage 1 requires the recorded runtime exception")
        if not bool(
            get_nested(
                config,
                "stage_0_result.support_bound_numerical_exception.accepted",
            )
        ):
            raise RuntimeError(
                "exp486 Stage 1 requires the recorded support numerical interpretation"
            )
        if bool(
            get_nested(
                config,
                "execution.stage_1_resume_from_frozen_v2",
                False,
            )
        ):
            if (
                int(
                    get_nested(
                        config,
                        "execution.stage_1_resume_current_kernel_pf_well_runs",
                        -1,
                    )
                )
                != 0
            ):
                raise ValueError("exp486 Stage 1 resume must rerun zero PF wells")
            resume = dict(get_nested(config, "data.stage1_frozen_resume") or {})
            if (
                int(resume.get("source_kernel_version", -1)) != 2
                or str(resume.get("scientific_contract_sha256", ""))
                != "62dcb499c0c9c9320091fa28663771493847dd6f46f03737015d1373dddc5f8e"
            ):
                raise ValueError("exp486 Stage 1 frozen resume contract changed")
    if bool(get_nested(config, "execution.run_inference", False)) or bool(
        get_nested(config, "execution.create_submission", False)
    ):
        raise ValueError("exp486 inference/submission must remain disabled")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise RuntimeError("exp486 Kaggle push is not approved")
        if run_stage0 and not bool(
            get_nested(config, "execution.stage_0_execution_approved", False)
        ):
            raise RuntimeError("exp486 Stage 0 Kaggle execution is not approved")
        if run_stage1 and not bool(
            get_nested(config, "execution.stage_1_execution_approved", False)
        ):
            raise RuntimeError("exp486 Stage 1 Kaggle execution is not approved")
        completed = bool(get_nested(config, "execution.stage_1_completed", False))
        if not (run_stage0 or run_stage1 or completed):
            raise RuntimeError("exp486 has no approved execution stage selected")
    return counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(get_nested(config, "model.fixed_from_exp404_for_both") or {})
    absolute = dict(get_nested(config, "model.absolute_geometry_unary") or {})
    residual = dict(get_nested(config, "model.residual_offset_state") or {})
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "implementation_reference": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "active_variants": list(ACTIVE_VARIANTS),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "selection_policy": "independent_report_only_no_same_oof_winner",
        "geometry_allowlist": list(GEOMETRY_ALLOWLIST),
        "absolute_geometry_unary": {
            "state": "exp404_position_and_rate_unchanged",
            "sigma_ft": float(absolute["sigma_ft"]),
            "lambda": float(absolute["lambda"]),
            "clip_z2": 600.0,
            "added_to": "exp404_capped_gaussian_gr_log_likelihood",
        },
        "residual_offset_state": {
            "state": ["offset_from_tvt_geop", "offset_rate"],
            "output_tvt": "tvt_geop_plus_offset",
            "initial_offset_center": ("last_known_tvt_minus_tvt_geop_at_first_score_row"),
            "initial_offset_spread_ft": float(residual["initial_offset_spread_ft"]),
            "initial_offset_rate_center": float(residual["initial_offset_rate_center"]),
            "initial_offset_rate_spread": float(residual["initial_offset_rate_spread"]),
            "offset_rate_momentum": 0.998,
            "offset_rate_noise": 0.002,
            "offset_position_noise": 0.005,
        },
        "pf": {
            "particles": int(fixed["particles"]),
            "seeds": int(fixed["seeds"]),
            "temperature": float(fixed["primary_seed_weighting_temperature"]),
            "gr_scale_multiplier": float(fixed["gr_scale_multiplier"]),
            "momentum": float(fixed["momentum"]),
            "rate_noise": float(fixed["rate_noise"]),
            "position_noise": float(fixed["position_noise"]),
            "rough_position": float(fixed["rough_position"]),
            "rough_rate": float(fixed["rough_rate"]),
            "resample_fraction": float(fixed["resample_threshold_fraction"]),
        },
        "rng": {
            "seed_base": 'sha256_first16("likpf::train::<well_id>")',
            "variant_name_in_seed": False,
            "common_random_number_labels": True,
        },
        "saved_control_rerun": False,
        "execution_counts": validate_execution_contract(config),
    }
    payload["scientific_contract_sha256"] = mapping_sha256(payload)
    return payload


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "implementation.enabled": True,
        "implementation.implementation_approval_received": True,
        "implementation.canonical_train_notebook_adopted": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.fixed32_is_cv": False,
        "model.active_variants": list(ACTIVE_VARIANTS),
        "model.absolute_geometry_unary.sigma_ft": 20.0,
        "model.absolute_geometry_unary.lambda": 0.5,
        "model.residual_offset_state.initial_offset_spread_ft": 4.5,
        "model.residual_offset_state.initial_offset_rate_center": 0.0,
        "model.residual_offset_state.initial_offset_rate_spread": 0.01,
        "model.fixed_from_exp404_for_both.particles": 500,
        "model.fixed_from_exp404_for_both.seeds": 128,
        "model.fixed_from_exp404_for_both.primary_seed_weighting_temperature": 5.0,
        "model.fixed_from_exp404_for_both.gr_scale_multiplier": 1.0,
        "model.fixed_from_exp404_for_both.momentum": 0.998,
        "model.fixed_from_exp404_for_both.rate_noise": 0.002,
        "model.fixed_from_exp404_for_both.position_noise": 0.005,
        "model.fixed_from_exp404_for_both.rough_position": 0.1,
        "model.fixed_from_exp404_for_both.rough_rate": 0.001,
        "model.fixed_from_exp404_for_both.resample_threshold_fraction": 0.5,
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    status = str(get_nested(config, "experiment.status"))
    if status not in {
        "stage0_authorized_pending_push",
        "stage0_fail_closed",
        "stage1_approved_pending_kaggle",
        "stage1_resume_approved_pending_kaggle",
        "stage1_completed_with_eligible_variants",
        "stage1_all_variants_gate_failed_terminal_close",
    }:
        raise ValueError(f"exp486 scientific contract mismatch: experiment.status={status!r}")
    run_stage1 = bool(get_nested(config, "execution.run_stage_1", False))
    selected = get_nested(config, "execution.selected_stage")
    completed_statuses = {
        "stage1_completed_with_eligible_variants",
        "stage1_all_variants_gate_failed_terminal_close",
    }
    if status in completed_statuses:
        if run_stage1 or selected not in {None, "", "null"}:
            raise ValueError("exp486 completed Stage 1 must have no active execution stage")
        if not bool(get_nested(config, "execution.stage_1_completed", False)):
            raise ValueError("exp486 completed status requires stage_1_completed=true")
    elif status.startswith("stage1_") and not run_stage1:
        raise ValueError("exp486 pending Stage 1 status requires run_stage_1=true")
    for key, required in expected.items():
        observed = get_nested(config, key)
        if observed != required:
            raise ValueError(
                f"exp486 scientific contract mismatch: {key}={observed!r}, expected={required!r}"
            )
    safe_columns = list(
        get_nested(config, "data.exp226_oof_geometry.prediction_time_usecols") or []
    )
    forbidden = set(get_nested(config, "data.exp226_oof_geometry.forbidden_pf_columns") or [])
    if safe_columns != list(GEOMETRY_ALLOWLIST):
        raise ValueError("exp486 exp226 prediction-time allowlist changed")
    if forbidden != {"tvt_pred", "gr_delta", "tvt_true", "error", "abs_error"}:
        raise ValueError("exp486 exp226 forbidden-column contract changed")
    validate_execution_contract(config, require_run_approval=require_run_approval)
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Fixed32 scope, exp226 allowlist, and leakage ledger


# %%
@dataclass
class LeakageLedger:
    expected_variant_wells: int
    frozen_variant_wells: set[str] = field(default_factory=set)
    geometry_safe_rows_before_freeze: int = 0
    forbidden_geometry_columns_read_before_freeze: int = 0
    truth_rows_before_all_freeze: int = 0
    control_rows_before_all_freeze: int = 0
    role_fold_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    control_rows_after_all_freeze: int = 0
    role_fold_rows_after_all_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return (
            self.expected_variant_wells > 0
            and len(self.frozen_variant_wells) == self.expected_variant_wells
        )

    def record_geometry_safe(self, rows: int) -> None:
        self.geometry_safe_rows_before_freeze += int(rows)

    def freeze(self, variant: str, well: str) -> None:
        self.frozen_variant_wells.add(f"{variant}::{well}")

    def _record_late(self, label: str, rows: int) -> None:
        before_name = f"{label}_before_all_freeze"
        after_name = f"{label}_after_all_freeze"
        if not self.all_frozen:
            setattr(self, before_name, int(getattr(self, before_name)) + int(rows))
            raise RuntimeError(f"{label} was read before both exp486 variants froze")
        setattr(self, after_name, int(getattr(self, after_name)) + int(rows))

    def record_truth(self, rows: int) -> None:
        self._record_late("truth_rows", rows)

    def record_control(self, rows: int) -> None:
        self._record_late("control_rows", rows)

    def record_role_fold(self, rows: int) -> None:
        self._record_late("role_fold_rows", rows)

    def report(self) -> dict[str, Any]:
        return {
            "expected_variant_wells": self.expected_variant_wells,
            "frozen_variant_wells": len(self.frozen_variant_wells),
            "all_frozen": self.all_frozen,
            "before_freeze": {
                "geometry_safe_rows": self.geometry_safe_rows_before_freeze,
                "forbidden_geometry_columns": (self.forbidden_geometry_columns_read_before_freeze),
                "truth_rows": self.truth_rows_before_all_freeze,
                "control_rows": self.control_rows_before_all_freeze,
                "role_fold_rows": self.role_fold_rows_before_all_freeze,
            },
            "after_freeze": {
                "truth_rows": self.truth_rows_after_all_freeze,
                "control_rows": self.control_rows_after_all_freeze,
                "role_fold_rows": self.role_fold_rows_after_all_freeze,
            },
        }


def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("exp486 fixed32 manifest SHA mismatch")
    return path


def load_fixed32_scope(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    path = fixed32_manifest_path(config)
    scope = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    expected = int(get_nested(config, "stages.stage_0.wells"))
    if len(scope) != expected or scope["well"].nunique() != expected:
        raise ValueError("exp486 fixed32 scope identity changed")
    wells = scope["well"].astype(str).tolist()
    return wells, {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "wells": len(wells),
        "columns_read_before_freeze": ["well"],
        "well_order_sha256": mapping_sha256(wells),
    }


def load_fixed32_roles_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(fixed32_manifest_path(config), dtype={"well": str})
    ledger.record_role_fold(len(frame))
    required = {"well", "role", "fold"}
    if not required.issubset(frame.columns):
        raise ValueError("exp486 fixed32 role/fold columns are missing")
    if (
        len(frame) != 32
        or frame["well"].nunique() != 32
        or frame["role"].value_counts().to_dict() != {"control": 16, "persistent": 16}
        or set(frame["fold"].astype(int)) != set(range(5))
    ):
        raise ValueError("exp486 fixed32 role/fold balance changed")
    return frame


def geometry_input_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.exp226_oof_geometry") or {})
    path = resolve_existing(
        str(spec["filename"]),
        spec.get("candidates", []),
        spec.get("patterns", []),
    )
    expected = str(spec["expected_decompressed_sha256"])
    if sha256_decompressed_csv(path) != expected:
        raise ValueError("exp486 exp226 geometry decompressed SHA mismatch")
    return path


def load_fold_safe_geometry(
    path: Path,
    config: Mapping[str, Any],
    *,
    wells: set[str] | None = None,
    ledger: LeakageLedger | None = None,
) -> pd.DataFrame:
    safe_columns = list(
        get_nested(config, "data.exp226_oof_geometry.prediction_time_usecols") or []
    )
    if safe_columns != list(GEOMETRY_ALLOWLIST):
        raise ValueError("exp486 geometry allowlist changed")
    geometry = pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"well_id": str},
        compression="infer",
    )
    geometry = geometry.loc[:, safe_columns]
    geometry["row_idx"] = pd.to_numeric(geometry["row_idx"], errors="raise").astype(np.int64)
    geometry["suffix_offset"] = pd.to_numeric(geometry["suffix_offset"], errors="raise").astype(
        np.int64
    )
    geometry["tvt_geop"] = pd.to_numeric(geometry["tvt_geop"], errors="raise").astype(np.float64)
    if wells is not None:
        geometry = geometry.loc[geometry["well_id"].isin(wells)].copy()
    if (
        geometry.duplicated(["well_id", "row_idx"]).any()
        or not np.isfinite(geometry["tvt_geop"]).all()
    ):
        raise ValueError("exp486 geometry rows are duplicated or non-finite")
    if ledger is not None:
        ledger.record_geometry_safe(len(geometry))
    return geometry


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    for column in ("MD", "Z", "GR", "TVT_input"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["MD"].notna().all() or not frame["Z"].notna().all():
        raise ValueError(f"{well}: MD/Z contains missing values")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    if len(frame) < 2 or not np.isfinite(frame["TVT"]).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    mean_gr = float(frame["GR"].mean())
    if not math.isfinite(mean_gr):
        raise ValueError(f"{well}: Type Well GR support is invalid")
    frame["GR"] = frame["GR"].fillna(mean_gr)
    return frame


# %% [markdown]
# ## 5. Exp404 likelihood-PF input preparation
#
# Prefix-derived initial state, missing-GR filling, Type Well grid, and the
# clipped x1.0 Gaussian GR scale reproduce exp404. The exp226 file contributes
# only the four prediction-time allowlist columns.


# %%
def uniform_typewell_grid(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step: float,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(typewell_tvt))
    maximum = float(np.max(typewell_tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    grid_gr = np.interp(grid_tvt, typewell_tvt, typewell_gr).astype(np.float64)
    return grid_gr, minimum, float(step)


def exp072_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires a known prefix")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    residual = known_gr - np.interp(known_tvt, typewell_tvt, typewell_gr)
    raw_scale = float(np.nanstd(residual))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    base_scale = float(np.clip(raw_scale, 10.0, 60.0))
    return {
        "raw_scale": raw_scale,
        "base_scale": base_scale,
        "candidate_scale": base_scale,
        "multiplier": 1.0,
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
    }


def exp072_initial_rate(horizontal: pd.DataFrame, *, tail_rows: int = 30) -> float:
    tail = horizontal.loc[horizontal["TVT_input"].notna()].tail(tail_rows)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    valid = delta_md > 0.0
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    grid_step: float = 0.2,
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires a known prefix and unknown suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    scale_audit = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr.mean()))
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_z = evaluation["Z"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    if not (
        np.isfinite(eval_md).all() and np.isfinite(eval_z).all() and np.isfinite(eval_gr).all()
    ):
        raise ValueError("likelihood-PF evaluation inputs are not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": eval_z,
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - float(last_known["MD"]),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "last_known_tvt": float(last_known["TVT_input"]),
        "last_known_position": float(last_known["TVT_input"] + last_known["Z"]),
        "initial_rate": exp072_initial_rate(horizontal),
        "scale_audit": scale_audit,
    }


def align_geometry_to_prepared(
    well: str,
    geometry_rows: pd.DataFrame,
    prepared: Mapping[str, Any],
) -> np.ndarray:
    ordered = geometry_rows.sort_values("suffix_offset", kind="mergesort")
    expected_rows = np.asarray(prepared["eval_indices"], dtype=np.int64)
    observed_rows = ordered["row_idx"].to_numpy(np.int64)
    observed_offsets = ordered["suffix_offset"].to_numpy(np.int64)
    if (
        len(ordered) != len(expected_rows)
        or not np.array_equal(observed_rows, expected_rows)
        or not np.array_equal(
            observed_offsets,
            np.arange(len(expected_rows), dtype=np.int64),
        )
        or ordered["well_id"].astype(str).nunique() != 1
        or str(ordered["well_id"].iloc[0]) != str(well)
    ):
        raise ValueError(f"{well}: exp226 geometry identity does not match raw suffix")
    values = ordered["tvt_geop"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{well}: exp226 geometry contains non-finite values")
    return values


# %% [markdown]
# ## 6. Absolute-unary and residual-offset PF kernels
#
# Both variants preserve the exp404 base RNG consumption order: two draws per
# particle at initialization, two at propagation, one systematic-resampling
# uniform, and two roughening draws per resampled particle. Variant names do
# not enter the common stable seed label.


# %%
@njit(cache=True, nogil=True)
def _interp1(
    grid: np.ndarray,
    value: float,
    minimum: float,
    step: float,
) -> float:
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    final = len(grid) - 1
    if index >= final:
        return grid[final]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _pf_absolute_geometry_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    geometry_tvt_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_position: float,
    initial_rate: float,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
    initial_rate_spread: float,
    geometry_sigma: float,
    geometry_lambda: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    geometry_residual_mean = np.empty((seeds, rows))
    geometry_residual_std = np.empty((seeds, rows))
    geometry_log_factor_mean = np.empty((seeds, rows))
    effective_sample_size = np.empty((seeds, rows))
    resampled = np.zeros((seeds, rows), np.int8)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = last_position + initial_spread * np.random.randn()
            rate[particle] = initial_rate + initial_rate_spread * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = momentum * rate[particle] + rate_noise * np.random.randn()
                position[particle] += rate[particle] * delta_md + position_noise * np.random.randn()
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                    position_clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                    position_clip_counts[seed_index] += 1
                position[particle] = tvt_value + z_v[row]
            average_likelihood = 0.0
            for particle in range(particles):
                tvt_value = position[particle] - z_v[row]
                expected_gr = _interp1(
                    grid_gr,
                    tvt_value,
                    grid_minimum,
                    grid_step,
                )
                gr_zscore = (gr_v[row] - expected_gr) / gr_scale
                gr_squared = gr_zscore * gr_zscore
                if gr_squared > 600.0:
                    gr_squared = 600.0
                geometry_zscore = (tvt_value - geometry_tvt_v[row]) / geometry_sigma
                geometry_squared = geometry_zscore * geometry_zscore
                if geometry_squared > 600.0:
                    geometry_squared = 600.0
                log_factor = geometry_lambda * (-0.5 * geometry_squared)
                likelihood = np.exp(-0.5 * gr_squared + log_factor)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)
            weight_sum = 0.0
            for particle in range(particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(particles):
                    weights[particle] = 1.0 / particles
            residual_mean = 0.0
            log_factor_mean = 0.0
            inverse_ess = 0.0
            for particle in range(particles):
                tvt_value = position[particle] - z_v[row]
                residual = tvt_value - geometry_tvt_v[row]
                geometry_zscore = residual / geometry_sigma
                geometry_squared = geometry_zscore * geometry_zscore
                if geometry_squared > 600.0:
                    geometry_squared = 600.0
                residual_mean += weights[particle] * residual
                log_factor_mean += weights[particle] * geometry_lambda * (-0.5 * geometry_squared)
                inverse_ess += weights[particle] * weights[particle]
            residual_variance = 0.0
            for particle in range(particles):
                residual = position[particle] - z_v[row] - geometry_tvt_v[row]
                residual_variance += weights[particle] * (residual - residual_mean) ** 2
            ess = 1.0 / inverse_ess
            geometry_residual_mean[seed_index, row] = residual_mean
            geometry_residual_std[seed_index, row] = np.sqrt(max(residual_variance, 0.0))
            geometry_log_factor_mean[seed_index, row] = log_factor_mean
            effective_sample_size[seed_index, row] = ess
            if ess < minimum_ess[seed_index]:
                minimum_ess[seed_index] = ess
            if ess < resample_fraction * particles:
                cumulative = np.empty(particles)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_position[particle] = position[cursor] + rough_position * np.random.randn()
                    new_rate[particle] = rate[cursor] + rough_rate * np.random.randn()
                position = new_position
                rate = new_rate
                weights[:] = 1.0 / particles
                resampling_counts[seed_index] += 1
                resampled[seed_index, row] = 1
            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (position[particle] - z_v[row])
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        geometry_residual_mean,
        geometry_residual_std,
        geometry_log_factor_mean,
        effective_sample_size,
        resampled,
    )


@njit(cache=True, nogil=True)
def _pf_residual_offset_allseeds(
    md_v: np.ndarray,
    gr_v: np.ndarray,
    geometry_tvt_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    initial_offset_center: float,
    initial_offset_rate_center: float,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
    initial_rate_spread: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    offset_mean = np.empty((seeds, rows))
    offset_std = np.empty((seeds, rows))
    offset_rate_mean = np.empty((seeds, rows))
    offset_rate_std = np.empty((seeds, rows))
    support_fraction = np.empty((seeds, rows))
    effective_sample_size = np.empty((seeds, rows))
    resampled = np.zeros((seeds, rows), np.int8)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        offset = np.empty(particles)
        offset_rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            offset[particle] = initial_offset_center + initial_spread * np.random.randn()
            offset_rate[particle] = (
                initial_offset_rate_center + initial_rate_spread * np.random.randn()
            )
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                offset_rate[particle] = (
                    momentum * offset_rate[particle] + rate_noise * np.random.randn()
                )
                offset[particle] += (
                    offset_rate[particle] * delta_md + position_noise * np.random.randn()
                )
                tvt_value = geometry_tvt_v[row] + offset[particle]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                    position_clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                    position_clip_counts[seed_index] += 1
                offset[particle] = tvt_value - geometry_tvt_v[row]
            average_likelihood = 0.0
            for particle in range(particles):
                tvt_value = geometry_tvt_v[row] + offset[particle]
                expected_gr = _interp1(
                    grid_gr,
                    tvt_value,
                    grid_minimum,
                    grid_step,
                )
                zscore = (gr_v[row] - expected_gr) / gr_scale
                squared = zscore * zscore
                if squared > 600.0:
                    squared = 600.0
                likelihood = np.exp(-0.5 * squared)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)
            weight_sum = 0.0
            for particle in range(particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(particles):
                    weights[particle] = 1.0 / particles
            mean_offset = 0.0
            mean_rate = 0.0
            inverse_ess = 0.0
            in_support = 0.0
            for particle in range(particles):
                mean_offset += weights[particle] * offset[particle]
                mean_rate += weights[particle] * offset_rate[particle]
                inverse_ess += weights[particle] * weights[particle]
                tvt_value = geometry_tvt_v[row] + offset[particle]
                if grid_minimum <= tvt_value <= grid_maximum:
                    in_support += weights[particle]
            variance_offset = 0.0
            variance_rate = 0.0
            for particle in range(particles):
                variance_offset += weights[particle] * (offset[particle] - mean_offset) ** 2
                variance_rate += weights[particle] * (offset_rate[particle] - mean_rate) ** 2
            ess = 1.0 / inverse_ess
            offset_mean[seed_index, row] = mean_offset
            offset_std[seed_index, row] = np.sqrt(max(variance_offset, 0.0))
            offset_rate_mean[seed_index, row] = mean_rate
            offset_rate_std[seed_index, row] = np.sqrt(max(variance_rate, 0.0))
            support_fraction[seed_index, row] = in_support
            effective_sample_size[seed_index, row] = ess
            if ess < minimum_ess[seed_index]:
                minimum_ess[seed_index] = ess
            if ess < resample_fraction * particles:
                cumulative = np.empty(particles)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_offset = np.empty(particles)
                new_offset_rate = np.empty(particles)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_offset[particle] = offset[cursor] + rough_position * np.random.randn()
                    new_offset_rate[particle] = offset_rate[cursor] + rough_rate * np.random.randn()
                offset = new_offset
                offset_rate = new_offset_rate
                weights[:] = 1.0 / particles
                resampling_counts[seed_index] += 1
                resampled[seed_index, row] = 1
            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (geometry_tvt_v[row] + offset[particle])
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        offset_mean,
        offset_std,
        offset_rate_mean,
        offset_rate_std,
        support_fraction,
        effective_sample_size,
        resampled,
    )


# %% [markdown]
# ## 7. Seed aggregation, parity, and state-transition contracts


# %%
def aggregate_temperature(
    values: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / temperature)
    weights /= np.sum(weights)
    return (weights[:, None] * values).sum(axis=0), weights


def evidence_weighted_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights[:, None] * values).sum(axis=0)


def run_absolute_geometry_pf(
    prepared: Mapping[str, Any],
    geometry_tvt: np.ndarray,
    *,
    particles: int,
    seeds: int,
    seed_base: int,
    temperature: float,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
    initial_rate_spread: float,
    geometry_sigma: float,
    geometry_lambda: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    output = _pf_absolute_geometry_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(geometry_tvt, dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        int(particles),
        int(seeds),
        int(seed_base),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_spread),
        float(initial_rate_spread),
        float(geometry_sigma),
        float(geometry_lambda),
    )
    prediction, seed_weights = aggregate_temperature(
        output[0],
        output[1],
        temperature=temperature,
    )
    ledger = pd.DataFrame(
        {
            "suffix_offset": np.arange(len(prediction), dtype=np.int64),
            "tvt_geop": geometry_tvt,
            "geometry_residual_mean": evidence_weighted_rows(output[5], seed_weights),
            "geometry_residual_std": evidence_weighted_rows(output[6], seed_weights),
            "geometry_log_factor_mean": evidence_weighted_rows(output[7], seed_weights),
            "effective_sample_size": evidence_weighted_rows(output[8], seed_weights),
            "resampled_seed_fraction": output[9].mean(axis=0),
        }
    )
    diagnostics = {
        "seed_log_likelihood_minimum": float(np.min(output[1])),
        "seed_log_likelihood_maximum": float(np.max(output[1])),
        "seed_weight_maximum": float(np.max(seed_weights)),
        "resampling_count": int(np.sum(output[2])),
        "minimum_effective_sample_size": float(np.min(output[3])),
        "position_clip_count": int(np.sum(output[4])),
        "mean_geometry_log_factor": float(ledger["geometry_log_factor_mean"].mean()),
    }
    return prediction.astype(np.float32), ledger, diagnostics


def run_residual_offset_pf(
    prepared: Mapping[str, Any],
    geometry_tvt: np.ndarray,
    *,
    particles: int,
    seeds: int,
    seed_base: int,
    temperature: float,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
    initial_rate_center: float,
    initial_rate_spread: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    initial_offset_center = float(prepared["last_known_tvt"]) - float(geometry_tvt[0])
    output = _pf_residual_offset_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(geometry_tvt, dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        initial_offset_center,
        float(initial_rate_center),
        int(particles),
        int(seeds),
        int(seed_base),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_spread),
        float(initial_rate_spread),
    )
    prediction, seed_weights = aggregate_temperature(
        output[0],
        output[1],
        temperature=temperature,
    )
    geometry_delta = np.empty(len(geometry_tvt), dtype=np.float64)
    geometry_delta[0] = 0.0
    geometry_delta[1:] = np.diff(geometry_tvt)
    particle_delta = np.empty(len(prediction), dtype=np.float64)
    particle_delta[0] = float(prediction[0] - prepared["last_known_tvt"])
    particle_delta[1:] = np.diff(prediction)
    ledger = pd.DataFrame(
        {
            "suffix_offset": np.arange(len(prediction), dtype=np.int64),
            "tvt_geop": geometry_tvt,
            "geometry_delta": geometry_delta,
            "filtered_offset_mean": evidence_weighted_rows(output[5], seed_weights),
            "filtered_offset_std": evidence_weighted_rows(output[6], seed_weights),
            "filtered_offset_rate_mean": evidence_weighted_rows(output[7], seed_weights),
            "filtered_offset_rate_std": evidence_weighted_rows(output[8], seed_weights),
            "particle_drift_minus_geometry_delta": (particle_delta - geometry_delta),
            "typewell_support_fraction": evidence_weighted_rows(output[9], seed_weights),
            "effective_sample_size": evidence_weighted_rows(output[10], seed_weights),
            "resampled_seed_fraction": output[11].mean(axis=0),
        }
    )
    diagnostics = {
        "initial_offset_center": initial_offset_center,
        "initial_offset_rate_center": float(initial_rate_center),
        "seed_log_likelihood_minimum": float(np.min(output[1])),
        "seed_log_likelihood_maximum": float(np.max(output[1])),
        "seed_weight_maximum": float(np.max(seed_weights)),
        "resampling_count": int(np.sum(output[2])),
        "minimum_effective_sample_size": float(np.min(output[3])),
        "position_clip_count": int(np.sum(output[4])),
        "minimum_typewell_support_fraction": float(ledger["typewell_support_fraction"].min()),
    }
    return prediction.astype(np.float32), ledger, diagnostics


def residual_transition_contract() -> dict[str, Any]:
    previous_offset = 2.5
    previous_rate = -0.04
    delta_md = 3.0
    rate_draw = 0.25
    position_draw = -0.5
    observed_rate = 0.998 * previous_rate + 0.002 * rate_draw
    observed_offset = previous_offset + observed_rate * delta_md + 0.005 * position_draw
    expected_rate = -0.03942
    expected_offset = 2.37924
    return {
        "observed_offset_rate": observed_rate,
        "expected_offset_rate": expected_rate,
        "observed_offset": observed_offset,
        "expected_offset": expected_offset,
        "update_order": "offset_rate_then_offset_then_gr_weight",
        "pass": bool(
            abs(observed_rate - expected_rate) <= 1.0e-15
            and abs(observed_offset - expected_offset) <= 1.0e-15
        ),
    }


def warm_up_pf_kernels() -> None:
    md = np.asarray([1.0, 2.0], dtype=np.float64)
    z = np.asarray([0.0, 0.1], dtype=np.float64)
    gr = np.asarray([50.0, 51.0], dtype=np.float64)
    geometry = np.asarray([100.0, 100.2], dtype=np.float64)
    grid = np.linspace(40.0, 70.0, 101)
    _pf_absolute_geometry_allseeds(
        md,
        z,
        gr,
        geometry,
        grid,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        8,
        2,
        123,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        20.0,
        0.5,
    )
    _pf_residual_offset_allseeds(
        md,
        gr,
        geometry,
        grid,
        90.0,
        0.2,
        20.0,
        0.0,
        0.0,
        8,
        2,
        123,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
    )


# %% [markdown]
# ## 8. Target-free two-variant generation and freeze


# %%
@dataclass
class FrozenWell:
    well_id: str
    prediction: pd.DataFrame
    absolute_ledger: pd.DataFrame
    residual_ledger: pd.DataFrame
    audit: dict[str, Any]


def decode_target_free_well(
    well: str,
    raw_dir: Path,
    geometry_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> FrozenWell:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    fixed = dict(get_nested(config, "model.fixed_from_exp404_for_both") or {})
    absolute = dict(get_nested(config, "model.absolute_geometry_unary") or {})
    residual = dict(get_nested(config, "model.residual_offset_state") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(fixed["typewell_grid_step_ft"]),
    )
    geometry_tvt = align_geometry_to_prepared(well, geometry_rows, prepared)
    seed_base = stable_seed("likpf", "train", well)
    absolute_prediction, absolute_ledger, absolute_diagnostics = run_absolute_geometry_pf(
        prepared,
        geometry_tvt,
        particles=int(fixed["particles"]),
        seeds=int(fixed["seeds"]),
        seed_base=seed_base,
        temperature=float(fixed["primary_seed_weighting_temperature"]),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(fixed["rough_position"]),
        rough_rate=float(fixed["rough_rate"]),
        resample_fraction=float(fixed["resample_threshold_fraction"]),
        initial_spread=float(fixed["initial_state_spread_ft"]),
        initial_rate_spread=float(fixed["initial_rate_spread"]),
        geometry_sigma=float(absolute["sigma_ft"]),
        geometry_lambda=float(absolute["lambda"]),
    )
    residual_prediction, residual_ledger, residual_diagnostics = run_residual_offset_pf(
        prepared,
        geometry_tvt,
        particles=int(fixed["particles"]),
        seeds=int(fixed["seeds"]),
        seed_base=seed_base,
        temperature=float(fixed["primary_seed_weighting_temperature"]),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(fixed["rough_position"]),
        rough_rate=float(fixed["rough_rate"]),
        resample_fraction=float(fixed["resample_threshold_fraction"]),
        initial_spread=float(residual["initial_offset_spread_ft"]),
        initial_rate_center=float(residual["initial_offset_rate_center"]),
        initial_rate_spread=float(residual["initial_offset_rate_spread"]),
    )
    eval_indices = np.asarray(prepared["eval_indices"], dtype=np.int64)
    raw_observed = np.asarray(prepared["raw_gr_observed"], dtype=bool)
    identifiers = [f"{well}_{int(row)}" for row in eval_indices]
    prediction = pd.DataFrame(
        {
            "id": identifiers,
            "well_id": str(well),
            "row_idx": eval_indices,
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64((~raw_observed).mean()),
            ABSOLUTE_PREDICTION: absolute_prediction,
            RESIDUAL_PREDICTION: residual_prediction,
        }
    )
    if not np.isfinite(prediction[list(PREDICTION_COLUMNS)]).all().all():
        raise ValueError(f"{well}: exp486 prediction contains non-finite values")
    for mechanism in (absolute_ledger, residual_ledger):
        mechanism.insert(0, "row_idx", eval_indices)
        mechanism.insert(0, "well_id", str(well))
        mechanism.insert(0, "id", identifiers)
    if list(absolute_ledger.columns) != list(ABSOLUTE_LEDGER_COLUMNS):
        raise ValueError("exp486 absolute mechanism schema changed")
    if list(residual_ledger.columns) != list(RESIDUAL_LEDGER_COLUMNS):
        raise ValueError("exp486 residual mechanism schema changed")
    seeds = int(fixed["seeds"])
    particles = int(fixed["particles"])
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "eval_rows": int(len(prediction)),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "geometry_rows": int(len(geometry_tvt)),
        "geometry_minimum": float(np.min(geometry_tvt)),
        "geometry_maximum": float(np.max(geometry_tvt)),
        "seed_base_absolute": int(seed_base),
        "seed_base_residual": int(seed_base),
        "variant_names_excluded_from_seed": True,
        "scientific_variants": 2,
        "candidate_pf_well_runs": 2,
        "seed_well_trajectories": 2 * seeds,
        "particle_starts": 2 * seeds * particles,
        "seeds_per_variant": seeds,
        "particles_per_seed": particles,
        "absolute_prediction_logical_sha256": dataframe_content_sha(
            prediction,
            ["id", "well_id", "row_idx", ABSOLUTE_PREDICTION],
        ),
        "residual_prediction_logical_sha256": dataframe_content_sha(
            prediction,
            ["id", "well_id", "row_idx", RESIDUAL_PREDICTION],
        ),
        "absolute_ledger_logical_sha256": dataframe_content_sha(
            absolute_ledger,
            ABSOLUTE_LEDGER_COLUMNS,
        ),
        "residual_ledger_logical_sha256": dataframe_content_sha(
            residual_ledger,
            RESIDUAL_LEDGER_COLUMNS,
        ),
        **{f"absolute_{key}": value for key, value in absolute_diagnostics.items()},
        **{f"residual_{key}": value for key, value in residual_diagnostics.items()},
        "wall_seconds": time.time() - started,
    }
    return FrozenWell(
        well_id=str(well),
        prediction=prediction,
        absolute_ledger=absolute_ledger,
        residual_ledger=residual_ledger,
        audit=audit,
    )


def freeze_target_free_outputs(
    frozen_wells: Sequence[FrozenWell],
    output: Path,
    *,
    ledger: LeakageLedger,
    stage: str = "stage0",
    expected_rows: int | None = None,
    expected_wells: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in {"stage0", "stage1"}:
        raise ValueError(f"exp486 unsupported freeze stage: {stage}")
    predictions = (
        pd.concat([item.prediction for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    absolute = (
        pd.concat(
            [item.absolute_ledger for item in frozen_wells],
            ignore_index=True,
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    residual = (
        pd.concat(
            [item.residual_ledger for item in frozen_wells],
            ignore_index=True,
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([item.audit for item in frozen_wells])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        predictions["id"].duplicated().any()
        or absolute["id"].duplicated().any()
        or residual["id"].duplicated().any()
        or len(predictions) != len(absolute)
        or len(predictions) != len(residual)
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp486 target-free output coverage mismatch")
    if expected_rows is not None and len(predictions) != int(expected_rows):
        raise ValueError(f"exp486 {stage} prediction row count changed")
    if expected_wells is not None and predictions["well_id"].nunique() != int(expected_wells):
        raise ValueError(f"exp486 {stage} prediction well count changed")
    prediction_path = output / f"{OUTPUT_PREFIX}_{stage}_predictions.csv.gz"
    absolute_path = output / f"{OUTPUT_PREFIX}_{stage}_absolute_ledger.csv.gz"
    residual_path = output / f"{OUTPUT_PREFIX}_{stage}_residual_ledger.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_{stage}_well_audit.csv"
    prediction_artifact = write_deterministic_gzip_csv(
        predictions,
        prediction_path,
    )
    absolute_artifact = write_deterministic_gzip_csv(absolute, absolute_path)
    residual_artifact = write_deterministic_gzip_csv(residual, residual_path)
    audit.to_csv(audit_path, index=False)
    for item in frozen_wells:
        ledger.freeze(ABSOLUTE_VARIANT, item.well_id)
        ledger.freeze(RESIDUAL_VARIANT, item.well_id)
    if not ledger.all_frozen:
        raise RuntimeError(f"exp486 did not freeze both variants for all {stage} wells")
    prediction_logical_columns = [
        "id",
        "well_id",
        "row_idx",
        *PREDICTION_COLUMNS,
    ]
    frozen = {
        "stage": stage,
        "frozen_before_truth_attachment": True,
        "rows": int(len(predictions)),
        "wells": int(predictions["well_id"].nunique()),
        "scientific_variants": 2,
        "prediction_logical_columns": prediction_logical_columns,
        "prediction_logical_sha256": dataframe_content_sha(
            predictions,
            prediction_logical_columns,
        ),
        "absolute_ledger_logical_sha256": dataframe_content_sha(
            absolute,
            ABSOLUTE_LEDGER_COLUMNS,
        ),
        "residual_ledger_logical_sha256": dataframe_content_sha(
            residual,
            RESIDUAL_LEDGER_COLUMNS,
        ),
        "prediction_artifact": prediction_artifact,
        "absolute_ledger_artifact": absolute_artifact,
        "residual_ledger_artifact": residual_artifact,
        "well_audit": {
            "path": str(audit_path),
            "raw_sha256": sha256_path(audit_path),
        },
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    readback = {
        "prediction_raw_sha256": sha256_path(prediction_path),
        "prediction_decompressed_sha256": sha256_decompressed_csv(prediction_path),
        "absolute_raw_sha256": sha256_path(absolute_path),
        "absolute_decompressed_sha256": sha256_decompressed_csv(absolute_path),
        "residual_raw_sha256": sha256_path(residual_path),
        "residual_decompressed_sha256": sha256_decompressed_csv(residual_path),
    }
    frozen["sha_readback"] = {
        **readback,
        "pass": bool(
            readback["prediction_raw_sha256"] == prediction_artifact["raw_sha256"]
            and readback["prediction_decompressed_sha256"]
            == prediction_artifact["decompressed_sha256"]
            and readback["absolute_raw_sha256"] == absolute_artifact["raw_sha256"]
            and readback["absolute_decompressed_sha256"] == absolute_artifact["decompressed_sha256"]
            and readback["residual_raw_sha256"] == residual_artifact["raw_sha256"]
            and readback["residual_decompressed_sha256"] == residual_artifact["decompressed_sha256"]
        ),
    }
    return predictions, absolute, residual, audit, frozen


# %% [markdown]
# ## 9. Truth-late fixed32 readout and fail-closed gates
#
# Fixed32 is a mechanism sample, not CV. Truth-late RMSE is descriptive only
# and does not choose a winner or activate either variant. Stage 1 scientific
# promotion gates remain specified in config but are not implemented here.


# %%
def _require_frozen(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("exp486 late readout requires both frozen predictions")
    for key in (
        "prediction_logical_sha256",
        "absolute_ledger_logical_sha256",
        "residual_ledger_logical_sha256",
    ):
        if len(str(frozen.get(key) or "")) != 64:
            raise RuntimeError(f"exp486 frozen output is missing {key}")


def load_suffix_truth(
    well: str,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["TVT_input", "TVT"],
    )
    horizontal["TVT_input"] = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    horizontal["TVT"] = pd.to_numeric(horizontal["TVT"], errors="coerce")
    eval_indices = np.flatnonzero(horizontal["TVT_input"].isna()).astype(np.int64)
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "true_tvt": horizontal.loc[eval_indices, "TVT"].to_numpy(np.float64),
        }
    )
    if not np.isfinite(frame["true_tvt"]).all():
        raise ValueError(f"{well}: suffix truth is not finite")
    ledger.record_truth(len(frame))
    return frame


def saved_control_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.saved_control") or {})
    path = resolve_existing(
        str(spec["filename"]),
        spec.get("candidates", []),
        spec.get("patterns", []),
    )
    expected_raw = str(spec.get("expected_raw_sha256") or "")
    expected_decompressed = str(spec.get("expected_decompressed_sha256") or "")
    if expected_raw and sha256_path(path) != expected_raw:
        raise ValueError("exp486 saved exp404 control raw SHA mismatch")
    if expected_decompressed and sha256_decompressed_csv(path) != expected_decompressed:
        raise ValueError("exp486 saved exp404 control decompressed SHA mismatch")
    return path


def load_saved_control_after_freeze(
    config: Mapping[str, Any],
    identifiers: set[str],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    spec = dict(get_nested(config, "data.saved_control") or {})
    source_column = str(spec["prediction_column"])
    frame = pd.read_csv(
        saved_control_path(config),
        usecols=["id", source_column],
        dtype={"id": str},
        compression="gzip",
    )
    frame = frame.loc[frame["id"].isin(identifiers)].copy()
    ledger.record_control(len(frame))
    if len(frame) != len(identifiers) or frame["id"].nunique() != len(identifiers):
        raise ValueError("exp486 saved control coverage mismatch")
    return frame.rename(columns={source_column: PRIMARY_CONTROL})


def attach_truth_late_readout(
    predictions: pd.DataFrame,
    absolute: pd.DataFrame,
    residual: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_frozen(frozen)
    logical_columns = ["id", "well_id", "row_idx", *PREDICTION_COLUMNS]
    if dataframe_content_sha(
        predictions,
        logical_columns,
    ) != str(frozen["prediction_logical_sha256"]):
        raise ValueError("exp486 predictions changed after freeze")
    if dataframe_content_sha(
        absolute,
        ABSOLUTE_LEDGER_COLUMNS,
    ) != str(frozen["absolute_ledger_logical_sha256"]):
        raise ValueError("exp486 absolute ledger changed after freeze")
    if dataframe_content_sha(
        residual,
        RESIDUAL_LEDGER_COLUMNS,
    ) != str(frozen["residual_ledger_logical_sha256"]):
        raise ValueError("exp486 residual ledger changed after freeze")
    roles = load_fixed32_roles_after_freeze(config, ledger)
    truth = pd.concat(
        [
            load_suffix_truth(str(well), raw_dir, ledger)
            for well in predictions["well_id"].drop_duplicates().tolist()
        ],
        ignore_index=True,
    )
    control = load_saved_control_after_freeze(
        config,
        set(predictions["id"].astype(str)),
        ledger,
    )
    frame = predictions.merge(
        truth,
        on=["id", "well_id", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(control, on="id", how="inner", validate="one_to_one")
    frame = frame.merge(
        roles[["well", "role", "fold"]].rename(columns={"well": "well_id"}),
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if (
        len(frame) != len(predictions)
        or frame["role"].isna().any()
        or not np.isfinite(frame[[*PREDICTION_COLUMNS, PRIMARY_CONTROL, "true_tvt"]]).all().all()
    ):
        raise ValueError("exp486 truth-late coverage mismatch")
    rows: list[dict[str, Any]] = []
    for name, column in (
        (ABSOLUTE_VARIANT, ABSOLUTE_PREDICTION),
        (RESIDUAL_VARIANT, RESIDUAL_PREDICTION),
        ("saved_exp404_control", PRIMARY_CONTROL),
        ("exp226_tvt_geop_reference", "tvt_geop"),
    ):
        source = (
            frame.merge(
                absolute[["id", "tvt_geop"]],
                on="id",
                how="inner",
                validate="one_to_one",
            )
            if column == "tvt_geop"
            else frame
        )
        squared = (source[column] - source["true_tvt"]) ** 2
        rows.append(
            {
                "variant": name,
                "rows": int(len(source)),
                "rmse": float(np.sqrt(squared.mean())),
                "descriptive_fixed32_not_cv": True,
            }
        )
    summary = pd.DataFrame(rows)
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        control_rmse = float(np.sqrt(np.mean((group[PRIMARY_CONTROL] - group["true_tvt"]) ** 2)))
        for name, column in (
            (ABSOLUTE_VARIANT, ABSOLUTE_PREDICTION),
            (RESIDUAL_VARIANT, RESIDUAL_PREDICTION),
        ):
            candidate_rmse = float(np.sqrt(np.mean((group[column] - group["true_tvt"]) ** 2)))
            by_well_rows.append(
                {
                    "well_id": str(well),
                    "variant": name,
                    "role": str(group["role"].iloc[0]),
                    "fold": int(group["fold"].iloc[0]),
                    "rows": int(len(group)),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "delta_rmse": candidate_rmse - control_rmse,
                }
            )
    return frame, summary, pd.DataFrame(by_well_rows)


def evaluate_stage0_gates(
    predictions: pd.DataFrame,
    absolute: pd.DataFrame,
    residual: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    elapsed_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical = dict(get_nested(config, "guards.technical_stage_0") or {})
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    projected_full_seconds = elapsed_seconds / 64.0 * 1546.0 if elapsed_seconds > 0.0 else 0.0
    before = ledger.report()["before_freeze"]
    technical_checks = {
        "geometry_allowlist_exact": list(
            get_nested(
                config,
                "data.exp226_oof_geometry.prediction_time_usecols",
            )
            or []
        )
        == list(GEOMETRY_ALLOWLIST),
        "forbidden_geometry_reads_zero": int(before["forbidden_geometry_columns"]) == 0,
        "geometry_row_coverage": bool(
            len(predictions) == expected_rows
            and len(absolute) == expected_rows
            and len(residual) == expected_rows
        ),
        "variant_formula_and_state_contracts": bool(residual_transition_contract()["pass"]),
        "finite_prediction_coverage": bool(
            np.isfinite(predictions[list(PREDICTION_COLUMNS)]).all().all()
        ),
        "finite_mechanism_ledgers": bool(
            np.isfinite(absolute.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(residual.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "common_stable_seed_identity": bool(
            audit["seed_base_absolute"].eq(audit["seed_base_residual"]).all()
            and audit["variant_names_excluded_from_seed"].all()
        ),
        "execution_count_match": bool(
            int(audit["candidate_pf_well_runs"].sum()) == 64
            and int(audit["seed_well_trajectories"].sum()) == 8192
            and int(audit["particle_starts"].sum()) == 4096000
        ),
        "truth_control_role_fold_reads_before_freeze_zero": bool(
            int(before["truth_rows"]) == 0
            and int(before["control_rows"]) == 0
            and int(before["role_fold_rows"]) == 0
        ),
        "runtime_projection": bool(
            projected_full_seconds <= float(technical["maximum_seconds_full_projection"])
        ),
        "peak_rss": bool(rss_gb <= float(technical["maximum_peak_rss_gb"])),
    }
    mechanism_checks = {
        "absolute_geometry_factor_active": bool(
            (absolute["geometry_log_factor_mean"] <= 0.0).all()
            and (absolute["geometry_log_factor_mean"] < 0.0).any()
        ),
        "absolute_geometry_residual_nonnegative_std": bool(
            (absolute["geometry_residual_std"] >= 0.0).all()
        ),
        "residual_offset_state_non_degenerate": bool(
            (residual["filtered_offset_std"] > 0.0).any()
            and (residual["filtered_offset_rate_std"] > 0.0).any()
        ),
        "residual_support_fraction_bounded": bool(
            residual["typewell_support_fraction"].between(0.0, 1.0).all()
        ),
        "ess_positive_both_variants": bool(
            (absolute["effective_sample_size"] > 0.0).all()
            and (residual["effective_sample_size"] > 0.0).all()
        ),
    }
    technical_all_pass = bool(all(technical_checks.values()))
    mechanism_all_pass = bool(all(mechanism_checks.values()))
    return {
        "stage": "stage0_fixed32_technical_mechanism_preflight_not_cv",
        "technical_checks": technical_checks,
        "mechanism_checks": mechanism_checks,
        "technical_all_pass": technical_all_pass,
        "mechanism_all_pass": mechanism_all_pass,
        "all_pass": bool(technical_all_pass and mechanism_all_pass),
        "stage1_eligible_pending_separate_user_approval": bool(
            technical_all_pass and mechanism_all_pass
        ),
        "measurements": {
            "elapsed_seconds": elapsed_seconds,
            "projected_full_seconds": projected_full_seconds,
            "peak_rss_gb": rss_gb,
            "mean_absolute_geometry_log_factor": float(absolute["geometry_log_factor_mean"].mean()),
            "mean_residual_offset_std": float(residual["filtered_offset_std"].mean()),
            "minimum_residual_support_fraction": float(residual["typewell_support_fraction"].min()),
        },
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 10. Generated artifacts and guarded Stage 0 orchestration
#
# The implementation is complete, but config leaves the execution stage
# unselected. A later approval must enable the canonical Notebook/package and
# Stage 0 flags before this orchestration can run on Kaggle CPU.


# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL") == "1":
        return
    raise RuntimeError("exp486 train stages must run first on Kaggle CPU")


def build_input_manifest(
    config: Mapping[str, Any],
    raw_dir: Path,
    wells: Sequence[str],
    scope_report: Mapping[str, Any],
    geometry_path: Path,
    geometry: pd.DataFrame,
) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    for well in wells:
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        raw_rows.append(
            {
                "well_id": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    raw_frame = pd.DataFrame(raw_rows).sort_values(
        "well_id",
        kind="mergesort",
    )
    return {
        "split": "train",
        "fixed32": dict(scope_report),
        "raw_dir": str(raw_dir),
        "wells": len(wells),
        "raw_well_content_sha256": dataframe_content_sha(
            raw_frame,
            ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
        ),
        "exp226_geometry": {
            "path": str(geometry_path),
            "raw_sha256": sha256_path(geometry_path),
            "decompressed_sha256": sha256_decompressed_csv(geometry_path),
            "logical_sha256_fixed32": dataframe_content_sha(
                geometry.sort_values(
                    ["well_id", "row_idx"],
                    kind="mergesort",
                ),
                GEOMETRY_ALLOWLIST,
            ),
            "columns_read": list(GEOMETRY_ALLOWLIST),
            "rows": len(geometry),
            "wells": int(geometry["well_id"].nunique()),
        },
        "saved_control": {
            "expected_raw_sha256": get_nested(
                config,
                "data.saved_control.expected_raw_sha256",
            ),
            "expected_decompressed_sha256": get_nested(
                config,
                "data.saved_control.expected_decompressed_sha256",
            ),
            "rerun": False,
            "parsed_before_freeze": False,
        },
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    require_kaggle_runtime()
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    wells, scope_report = load_fixed32_scope(config)
    ledger = LeakageLedger(expected_variant_wells=2 * len(wells))
    geometry_path = geometry_input_path(config)
    geometry = load_fold_safe_geometry(
        geometry_path,
        config,
        wells=set(wells),
        ledger=ledger,
    )
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    if len(geometry) != expected_rows or geometry["well_id"].nunique() != len(wells):
        raise ValueError("exp486 fixed32 geometry coverage mismatch")
    scientific_contract = build_scientific_contract(config)
    scientific_path = output / f"{OUTPUT_PREFIX}_scientific_contract.json"
    scientific_artifact = write_json(scientific_path, scientific_contract)
    input_report = build_input_manifest(
        config,
        raw_dir,
        wells,
        scope_report,
        geometry_path,
        geometry,
    )
    input_path = output / f"{OUTPUT_PREFIX}_stage0_input_manifest.json"
    input_artifact = write_json(input_path, input_report)

    warm_up_pf_kernels()
    geometry_groups = {
        str(well): group.copy() for well, group in geometry.groupby("well_id", sort=False)
    }
    workers = int(get_nested(config, "runtime.num_workers", 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frozen_wells = list(
            executor.map(
                lambda well: decode_target_free_well(
                    str(well),
                    raw_dir,
                    geometry_groups[str(well)],
                    config,
                ),
                wells,
            )
        )
    predictions, absolute, residual, audit, frozen = freeze_target_free_outputs(
        frozen_wells,
        output,
        ledger=ledger,
    )
    if len(predictions) != expected_rows:
        raise ValueError("exp486 fixed32 prediction row count changed")
    candidate_elapsed = float(audit["wall_seconds"].sum())
    prefreeze_elapsed = time.time() - started
    runtime_ledger = {
        "stage": "stage0_target_free_two_variant_freeze",
        "candidate_wells": len(wells),
        "candidate_rows": len(predictions),
        "scientific_variants": 2,
        "candidate_pf_well_runs": int(audit["candidate_pf_well_runs"].sum()),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "saved_control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "summed_variant_well_seconds": candidate_elapsed,
        "prefreeze_wall_seconds": prefreeze_elapsed,
        "projected_full_seconds": candidate_elapsed / 64.0 * 1546.0,
        "peak_rss_gb": peak_rss_gb(),
        "versions": runtime_versions(),
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    runtime_path = output / f"{OUTPUT_PREFIX}_stage0_runtime_ledger.json"
    runtime_artifact = write_json(runtime_path, runtime_ledger)
    frozen.update(
        {
            "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
            "scientific_contract_file_sha256": scientific_artifact["raw_sha256"],
            "input_manifest_sha256": input_artifact["raw_sha256"],
            "runtime_ledger_sha256": runtime_artifact["raw_sha256"],
        }
    )
    freeze_path = output / f"{OUTPUT_PREFIX}_stage0_freeze_manifest.json"
    freeze_artifact = write_json(freeze_path, frozen)

    frame, fixed32_summary, by_well = attach_truth_late_readout(
        predictions,
        absolute,
        residual,
        frozen,
        config=config,
        raw_dir=raw_dir,
        ledger=ledger,
    )
    gates = evaluate_stage0_gates(
        predictions,
        absolute,
        residual,
        audit,
        config=config,
        ledger=ledger,
        elapsed_seconds=candidate_elapsed,
        rss_gb=peak_rss_gb(),
    )
    truth_path = output / f"{OUTPUT_PREFIX}_stage0_truth_late_rows.csv.gz"
    summary_table_path = output / f"{OUTPUT_PREFIX}_stage0_fixed32_descriptive_metrics.csv"
    by_well_path = output / f"{OUTPUT_PREFIX}_stage0_by_well.csv"
    gate_path = output / f"{OUTPUT_PREFIX}_stage0_gate_report.json"
    truth_artifact = write_deterministic_gzip_csv(frame, truth_path)
    fixed32_summary.to_csv(summary_table_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    gate_artifact = write_json(gate_path, gates)
    status = (
        "stage0_all_pass_pending_stage1_approval" if gates["all_pass"] else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage0_fixed32_technical_mechanism_preflight_not_cv",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "counts": {
            "wells": len(wells),
            "rows": len(predictions),
            "scientific_variants": 2,
            "candidate_pf_well_runs": int(audit["candidate_pf_well_runs"].sum()),
            "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
            "particle_starts": int(audit["particle_starts"].sum()),
            "saved_control_pf_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "fixed32_descriptive_metrics_not_cv": fixed32_summary.to_dict(orient="records"),
        "frozen_outputs": frozen,
        "gates": gates,
        "runtime": runtime_ledger,
        "artifacts": {
            "scientific_contract": scientific_artifact,
            "input_manifest": input_artifact,
            "runtime_ledger": runtime_artifact,
            "freeze_manifest": freeze_artifact,
            "truth_late_rows": truth_artifact,
            "descriptive_metrics": {
                "path": str(summary_table_path),
                "raw_sha256": sha256_path(summary_table_path),
            },
            "by_well": {
                "path": str(by_well_path),
                "raw_sha256": sha256_path(by_well_path),
            },
            "gate_report": gate_artifact,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": (
            "request_separate_stage1_approval_without_variant_selection"
            if gates["all_pass"]
            else "close_failed_contract_without_parameter_or_gate_rescue"
        ),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_stage0_summary.json"
    summary_artifact = write_json(summary_path, summary)
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. All-well Stage 1 truth-late CV and independent promotion gates
#
# Stage 1 runs both unchanged PF variants for all 773 train wells. The
# exp226 geometry allowlist is the only saved input parsed before both
# predictions and mechanism ledgers freeze. Truth, saved exp404/exp209
# controls, reporting folds, and hidden-like roles are attached afterwards.
# The original Stage 0 failures remain recorded; Stage 1 uses the explicit
# runtime exception and a 1e-12 numerical readback tolerance for normalized
# residual support only.


# %%
def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    actual = typed_dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != expected_wells or actual != expected_sha:
        raise ValueError("exp486 raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": actual,
        "well_ids": frame["well_id"].astype(str).tolist(),
        "rows": rows,
    }


def stage1_saved_input_paths(config: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "saved_control": str(saved_control_path(config)),
        "exp226_oof_geometry": str(geometry_input_path(config)),
    }
    for key in ("exp209_hmm_control", "hidden_like_assignment"):
        spec = dict(get_nested(config, f"data.{key}") or {})
        path = resolve_existing(
            str(spec["filename"]),
            [str(value) for value in spec.get("candidates", [])],
            [str(value) for value in spec.get("patterns", [])],
        )
        paths[key] = str(path)
    return paths


def _align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    aligned_source = source.copy()
    aligned_source["id"] = aligned_source["id"].astype(str)
    if aligned_source["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = aligned_source.set_index("id").reindex(frame["id"].astype(str))
    if aligned[list(columns)].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[str(column)] = aligned[str(column)].to_numpy()
    return result


def attach_truth_late_stage1(
    predictions: pd.DataFrame,
    absolute: pd.DataFrame,
    residual: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    saved_paths: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_frozen(frozen)
    logical_columns = ["id", "well_id", "row_idx", *PREDICTION_COLUMNS]
    prediction_sha = dataframe_content_sha(predictions, logical_columns)
    if bool(frozen.get("resumed_from_kaggle_version")):
        absolute_sha = sha256_csv_payload(Path(frozen["resume_file_paths"]["absolute_ledger"]))
        residual_sha = sha256_csv_payload(Path(frozen["resume_file_paths"]["residual_ledger"]))
    else:
        absolute_sha = dataframe_content_sha(
            absolute,
            ABSOLUTE_LEDGER_COLUMNS,
        )
        residual_sha = dataframe_content_sha(
            residual,
            RESIDUAL_LEDGER_COLUMNS,
        )
    if prediction_sha != str(frozen["prediction_logical_sha256"]):
        raise RuntimeError("exp486 Stage 1 predictions changed after freeze")
    if absolute_sha != str(frozen["absolute_ledger_logical_sha256"]):
        raise RuntimeError("exp486 Stage 1 absolute ledger changed after freeze")
    if residual_sha != str(frozen["residual_ledger_logical_sha256"]):
        raise RuntimeError("exp486 Stage 1 residual ledger changed after freeze")

    wells = sorted(predictions["well_id"].astype(str).unique().tolist())
    truth = pd.concat(
        [load_suffix_truth(well, raw_dir, ledger) for well in wells],
        ignore_index=True,
    )
    frame = predictions.merge(
        truth,
        on=["id", "well_id", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    geometry_reference = absolute[["id", "tvt_geop"]].copy()
    frame = _align_on_id(
        frame,
        geometry_reference,
        ["tvt_geop"],
        label="frozen exp226 geometry reference",
    )

    control = load_saved_control_after_freeze(
        config,
        set(predictions["id"].astype(str)),
        ledger,
    )
    frame = _align_on_id(
        frame,
        control[["id", PRIMARY_CONTROL]],
        [PRIMARY_CONTROL],
        label="saved exp404 scale-5 control",
    )

    hmm_spec = dict(get_nested(config, "data.exp209_hmm_control") or {})
    hmm_path = Path(saved_paths["exp209_hmm_control"])
    if sha256_decompressed_csv(hmm_path) != str(hmm_spec["expected_decompressed_sha256"]):
        raise ValueError("exp486 saved exp209 HMM decompressed SHA mismatch")
    hmm_source_column = str(hmm_spec["prediction_column"])
    hmm = pd.read_csv(
        hmm_path,
        usecols=["id", hmm_source_column],
        dtype={"id": str},
        compression="infer",
    )
    ledger.record_control(len(hmm))
    hmm[hmm_source_column] = pd.to_numeric(hmm[hmm_source_column], errors="raise")
    hmm = hmm.rename(columns={hmm_source_column: "saved_exp209_hmm"})
    frame = _align_on_id(
        frame,
        hmm[["id", "saved_exp209_hmm"]],
        ["saved_exp209_hmm"],
        label="saved exp209 HMM",
    )

    fold_path = Path(saved_paths["exp226_oof_geometry"])
    fold_spec = dict(get_nested(config, "data.exp226_oof_geometry") or {})
    if sha256_decompressed_csv(fold_path) != str(fold_spec["expected_decompressed_sha256"]):
        raise ValueError("exp486 reporting-fold decompressed SHA mismatch")
    fold_columns = ["well_id", "row_idx", "suffix_offset", "fold"]
    fold = pd.read_csv(
        fold_path,
        usecols=fold_columns,
        dtype={"well_id": str},
        compression="infer",
    )
    ledger.record_role_fold(len(fold))
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp486 reporting-fold identity is duplicated")
    fold = fold.rename(columns={"suffix_offset": "reporting_suffix_offset"})
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if frame[["fold", "reporting_suffix_offset"]].isna().any().any():
        raise ValueError("exp486 reporting-fold attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["reporting_suffix_offset"].to_numpy(np.int64),
    ):
        raise ValueError("exp486 reporting-fold suffix identity mismatch")
    frame = frame.drop(columns=["reporting_suffix_offset"])

    hidden_spec = dict(get_nested(config, "data.hidden_like_assignment") or {})
    hidden_path = Path(saved_paths["hidden_like_assignment"])
    if sha256_path(hidden_path) != str(hidden_spec["expected_sha256"]):
        raise ValueError("exp486 hidden-like assignment raw SHA mismatch")
    role_columns = {
        str(scope): str(column) for scope, column in dict(hidden_spec["role_columns"]).items()
    }
    hidden = pd.read_csv(
        hidden_path,
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    ledger.record_role_fold(len(hidden))
    if hidden["well_id"].duplicated().any():
        raise ValueError("exp486 hidden-like assignment has duplicate wells")
    for scope, column in role_columns.items():
        actual = {
            str(key): int(value)
            for key, value in hidden[column]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .items()
        }
        expected = {
            str(key): int(value)
            for key, value in dict(hidden_spec["expected_role_counts"][scope]).items()
        }
        if actual != expected:
            raise ValueError(f"exp486 hidden-like role counts changed for {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("exp486 hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[role_columns["hidden_like_spatial"]].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[role_columns["hidden_like_typewell_purged"]].eq(
        "valid"
    )

    for prediction_column in PREDICTION_COLUMNS:
        frame[f"{prediction_column}__hmm_50_50"] = 0.5 * (
            frame[prediction_column].to_numpy(np.float64)
            + frame["saved_exp209_hmm"].to_numpy(np.float64)
        )
    frame["saved_control_hmm_50_50"] = 0.5 * (
        frame[PRIMARY_CONTROL].to_numpy(np.float64) + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    finite_columns = [
        "true_tvt",
        "tvt_geop",
        PRIMARY_CONTROL,
        *PREDICTION_COLUMNS,
        "saved_exp209_hmm",
        *(f"{column}__hmm_50_50" for column in PREDICTION_COLUMNS),
        "saved_control_hmm_50_50",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("exp486 Stage 1 late readout contains non-finite values")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if len(frame) != expected_rows or frame["well_id"].nunique() != expected_wells:
        raise ValueError("exp486 Stage 1 late readout coverage changed")
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("exp486 reporting-fold set mismatch")
    return frame, {
        "truth_attached_after_both_variant_freezes": True,
        "prediction_content_sha256_reverified": prediction_sha,
        "absolute_ledger_sha256_reverified": absolute_sha,
        "residual_ledger_sha256_reverified": residual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "saved_input_paths": dict(saved_paths),
        "truth_access_ledger": ledger.report(),
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def variant_prediction_pairs() -> tuple[tuple[str, str], ...]:
    return (
        (ABSOLUTE_VARIANT, ABSOLUTE_PREDICTION),
        (RESIDUAL_VARIANT, RESIDUAL_PREDICTION),
    )


def stage1_metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    variant: str,
    candidate_column: str,
    control_column: str,
    comparison: str,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"exp486 Stage 1 metric scope is empty: {variant}/{scope}")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[candidate_column].to_numpy(np.float64)
    control = selected[control_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "variant": variant,
        "candidate": candidate_column,
        "control": control_column,
        "comparison": comparison,
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "control_mae": float(np.mean(np.abs(control - truth))),
    }


def stage1_metric_scopes(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool)),
    ]
    for fold in sorted(frame["fold"].astype(int).unique().tolist()):
        scopes.append((f"fold_{fold}", frame["fold"].eq(fold).to_numpy()))
    scopes.extend(
        [
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
            (
                "missing_fraction_high",
                frame["well_missing_fraction"].ge(0.30).to_numpy(),
            ),
            ("md_since_1000_plus", frame["md_since"].ge(1000.0).to_numpy()),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    return scopes


def build_stage1_metric_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = stage1_metric_scopes(frame)
    primary_rows: list[dict[str, Any]] = []
    blend_rows: list[dict[str, Any]] = []
    for variant, prediction_column in variant_prediction_pairs():
        for scope, mask in scopes:
            primary_rows.append(
                stage1_metric_record(
                    frame,
                    mask,
                    variant=variant,
                    candidate_column=prediction_column,
                    control_column=PRIMARY_CONTROL,
                    comparison="variant_vs_saved_exp404_scale5_x1p0",
                    scope=scope,
                )
            )
            blend_rows.append(
                stage1_metric_record(
                    frame,
                    mask,
                    variant=variant,
                    candidate_column=f"{prediction_column}__hmm_50_50",
                    control_column="saved_control_hmm_50_50",
                    comparison="fixed_exp209_hmm_pf_50_50",
                    scope=scope,
                )
            )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        control = group[PRIMARY_CONTROL].to_numpy(np.float64)
        control_rmse = rmse(truth, control)
        for variant, prediction_column in variant_prediction_pairs():
            candidate = group[prediction_column].to_numpy(np.float64)
            candidate_rmse = rmse(truth, candidate)
            by_well_rows.append(
                {
                    "well_id": str(well),
                    "variant": variant,
                    "rows": len(group),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "improvement_ft": control_rmse - candidate_rmse,
                    "delta_rmse_candidate_minus_control": (candidate_rmse - control_rmse),
                    "well_missing_fraction": float(group["well_missing_fraction"].iloc[0]),
                }
            )
    reference = pd.DataFrame(
        [
            stage1_metric_record(
                frame,
                mask,
                variant="exp226_tvt_geop_reference",
                candidate_column="tvt_geop",
                control_column=PRIMARY_CONTROL,
                comparison="report_only_exp226_geometry_vs_saved_exp404",
                scope=scope,
            )
            for scope, mask in scopes
        ]
    )
    return (
        pd.DataFrame(primary_rows),
        pd.DataFrame(by_well_rows),
        pd.DataFrame(blend_rows),
        reference,
    )


def _stage1_scope_row(
    metrics: pd.DataFrame,
    variant: str,
    scope: str,
) -> pd.Series:
    selected = metrics.loc[metrics["variant"].eq(variant) & metrics["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"exp486 expected one Stage 1 row for {variant}/{scope}")
    return selected.iloc[0]


def residual_support_within_tolerance(
    residual: pd.DataFrame,
    tolerance: float,
) -> bool:
    support = residual["typewell_support_fraction"].to_numpy(np.float64)
    return bool(
        np.isfinite(support).all()
        and (support >= -float(tolerance)).all()
        and (support <= 1.0 + float(tolerance)).all()
    )


def evaluate_stage1_gate(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    absolute: pd.DataFrame,
    residual: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    primary_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    blend_metrics: pd.DataFrame,
    ledger_at_freeze: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    runtime_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical_stage_0") or {})
    scientific_config = dict(get_nested(config, "guards.scientific_each_variant") or {})
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    before = dict(ledger_at_freeze["before_freeze"])
    runtime_exception = bool(get_nested(config, "stage_0_result.runtime_exception.approved"))
    support_exception = bool(
        get_nested(
            config,
            "stage_0_result.support_bound_numerical_exception.accepted",
        )
    )
    support_tolerance = float(
        get_nested(
            config,
            "stage_0_result.support_bound_numerical_exception.stage_1_readback_tolerance",
        )
    )
    execution_counts = {
        "scientific_variants": 2,
        "candidate_pf_well_runs": int(audit["candidate_pf_well_runs"].sum()),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        "scientific_variants": 2,
        "candidate_pf_well_runs": 1546,
        "seed_well_trajectories": 197888,
        "particle_starts": 98944000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    control_differences = [
        abs(
            float(_stage1_scope_row(primary_metrics, variant, "overall")["control_rmse"])
            - float(get_nested(config, "validation.primary_control_rmse_ft"))
        )
        for variant in ACTIVE_VARIANTS
    ]
    blend_control_differences = [
        abs(
            float(_stage1_scope_row(blend_metrics, variant, "overall")["control_rmse"])
            - float(
                get_nested(
                    config,
                    "validation.fixed_hmm_pf_50_50_control_rmse_ft",
                )
            )
        )
        for variant in ACTIVE_VARIANTS
    ]
    runtime_limit = float(get_nested(config, "runtime.maximum_seconds"))
    original_runtime_passed = bool(runtime_seconds <= runtime_limit)
    technical_checks = {
        "stage0_original_failures_preserved": bool(
            not bool(get_nested(config, "stage_0_result.all_pass"))
            and not bool(
                get_nested(
                    config,
                    "stage_0_result.support_bound_numerical_exception.original_check_passed",
                )
            )
        ),
        "stage0_runtime_exception_approved": runtime_exception,
        "stage0_support_numerical_interpretation_accepted": support_exception,
        "raw_input_identity": bool(
            raw_manifest["content_sha256"]
            == str(get_nested(config, "data.expected_raw_well_identity_sha256"))
        ),
        "prediction_rows": len(frame) == expected_rows,
        "prediction_wells": int(frame["well_id"].nunique()) == expected_wells,
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist()) == expected_folds,
        "all_wells_completed": bool(
            len(audit) == expected_wells and audit["status"].eq("ok").all()
        ),
        "geometry_row_coverage": bool(
            len(absolute) == expected_rows and len(residual) == expected_rows
        ),
        "finite_prediction_coverage": bool(
            np.isfinite(frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all()
        ),
        "finite_mechanism_ledgers": bool(
            np.isfinite(absolute.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(residual.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "common_stable_seed_identity": bool(
            audit["seed_base_absolute"].eq(audit["seed_base_residual"]).all()
            and audit["variant_names_excluded_from_seed"].all()
        ),
        "residual_support_within_numerical_tolerance": bool(
            support_exception and residual_support_within_tolerance(residual, support_tolerance)
        ),
        "truth_control_role_fold_reads_before_freeze_zero": bool(
            int(before["truth_rows"]) == 0
            and int(before["control_rows"]) == 0
            and int(before["role_fold_rows"]) == 0
            and int(before["forbidden_geometry_columns"]) == 0
        ),
        "execution_count_match": execution_counts == expected_counts,
        "artifact_sha_readback": bool(get_nested(frozen, "sha_readback.pass")),
        "saved_control_rmse_parity": bool(max(control_differences) <= 0.00001),
        "fixed_hmm_pf_50_50_control_parity": bool(max(blend_control_differences) <= 0.00001),
        "runtime_accepted": bool(original_runtime_passed or runtime_exception),
        "peak_rss": bool(rss_gb <= float(get_nested(config, "runtime.maximum_peak_rss_gb"))),
    }
    technical = {
        "checks": technical_checks,
        "passed": bool(all(technical_checks.values())),
        "execution_counts": execution_counts,
        "saved_control_rmse_absolute_difference_max": max(control_differences),
        "fixed_hmm_pf_50_50_control_rmse_absolute_difference_max": max(blend_control_differences),
        "runtime_seconds": runtime_seconds,
        "original_runtime_limit_seconds": runtime_limit,
        "original_runtime_gate_passed": original_runtime_passed,
        "runtime_user_exception_approved": runtime_exception,
        "runtime_user_exception_applied": bool(not original_runtime_passed and runtime_exception),
        "support_numerical_tolerance": support_tolerance,
        "support_minimum": float(residual["typewell_support_fraction"].min()),
        "support_maximum": float(residual["typewell_support_fraction"].max()),
        "peak_rss_gb": rss_gb,
        "truth_access_ledger_at_freeze": dict(ledger_at_freeze),
    }

    scope_rules = {
        "raw_gr_observed": ("minimum_gain", "minimum_raw_gr_observed_gain_ft"),
        "raw_gr_missing": (
            "maximum_regression",
            "maximum_raw_gr_missing_regression_ft",
        ),
        "missing_fraction_high": (
            "maximum_regression",
            "maximum_high_missing_well_regression_ft",
        ),
        "md_since_1000_plus": (
            "maximum_regression",
            "maximum_long_tail_1000_plus_regression_ft",
        ),
        "hidden_like_spatial": (
            "maximum_regression",
            "maximum_hidden_like_spatial_regression_ft",
        ),
        "hidden_like_typewell_purged": (
            "maximum_regression",
            "maximum_hidden_like_typewell_purged_regression_ft",
        ),
    }
    variant_gates: dict[str, Any] = {}
    eligible_variants: list[str] = []
    for variant in ACTIVE_VARIANTS:
        overall = _stage1_scope_row(primary_metrics, variant, "overall")
        blend_overall = _stage1_scope_row(blend_metrics, variant, "overall")
        fold_rows = primary_metrics.loc[
            primary_metrics["variant"].eq(variant)
            & primary_metrics["scope"].str.startswith("fold_")
        ]
        improved_folds = int((fold_rows["improvement_ft"] > 0.0).sum())
        scope_checks: dict[str, Any] = {}
        for scope, (kind, key) in scope_rules.items():
            row = _stage1_scope_row(primary_metrics, variant, scope)
            threshold = float(scientific_config[key])
            improvement = float(row["improvement_ft"])
            delta = float(row["delta_rmse_candidate_minus_control"])
            passed = improvement >= threshold if kind == "minimum_gain" else delta <= threshold
            scope_checks[scope] = {
                "candidate_rmse": float(row["candidate_rmse"]),
                "control_rmse": float(row["control_rmse"]),
                "improvement_ft": improvement,
                "delta_rmse_candidate_minus_control": delta,
                "rule": kind,
                "threshold_ft": threshold,
                "passed": bool(passed),
            }
        variant_by_well = by_well_metrics.loc[by_well_metrics["variant"].eq(variant)]
        by_well_delta = variant_by_well["delta_rmse_candidate_minus_control"]
        by_well_p95 = float(by_well_delta.quantile(0.95))
        worst_index = by_well_delta.idxmax()
        worst_well = float(by_well_delta.loc[worst_index])
        worst_well_id = str(variant_by_well.loc[worst_index, "well_id"])
        primary_gate = {
            "candidate_rmse": float(overall["candidate_rmse"]),
            "control_rmse": float(overall["control_rmse"]),
            "improvement_ft": float(overall["improvement_ft"]),
            "minimum_improvement_ft": float(
                scientific_config["minimum_pooled_rmse_gain_vs_control_ft"]
            ),
            "improved_folds": improved_folds,
            "minimum_improved_folds": int(scientific_config["minimum_improved_folds"]),
            "scope_checks": scope_checks,
            "by_well_delta_p95_ft": by_well_p95,
            "maximum_by_well_delta_p95_ft": float(
                scientific_config["maximum_by_well_delta_p95_ft"]
            ),
            "worst_well_id": worst_well_id,
            "worst_well_regression_ft": worst_well,
            "maximum_worst_well_regression_ft": float(
                scientific_config["maximum_worst_well_regression_ft"]
            ),
        }
        primary_gate["passed"] = bool(
            primary_gate["improvement_ft"] >= primary_gate["minimum_improvement_ft"]
            and improved_folds >= primary_gate["minimum_improved_folds"]
            and all(item["passed"] for item in scope_checks.values())
            and by_well_p95 <= primary_gate["maximum_by_well_delta_p95_ft"]
            and worst_well <= primary_gate["maximum_worst_well_regression_ft"]
        )
        blend_guard = {
            "candidate_rmse": float(blend_overall["candidate_rmse"]),
            "control_rmse": float(blend_overall["control_rmse"]),
            "delta_rmse_candidate_minus_control": float(
                blend_overall["delta_rmse_candidate_minus_control"]
            ),
            "maximum_regression_ft": float(
                scientific_config["maximum_fixed_hmm_pf_50_50_regression_ft"]
            ),
        }
        blend_guard["passed"] = bool(
            blend_guard["delta_rmse_candidate_minus_control"]
            <= blend_guard["maximum_regression_ft"]
        )
        variant_passed = bool(
            technical["passed"] and primary_gate["passed"] and blend_guard["passed"]
        )
        variant_gates[variant] = {
            "passed": variant_passed,
            "primary_scientific_gate": primary_gate,
            "fixed_exp209_hmm_pf_50_50_guard": blend_guard,
        }
        if variant_passed:
            eligible_variants.append(variant)
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage1_all_well_train_side_cv",
        "passed": bool(eligible_variants),
        "technical_gate": technical,
        "variant_gates": variant_gates,
        "eligible_variants": eligible_variants,
        "selection_policy": "independent_report_only_no_same_oof_winner",
        "decision": (
            "eligible_variants_require_separate_inference_design_no_selection"
            if eligible_variants
            else "terminal_close_without_geometry_pf_rescue"
        ),
        "failure_action": (
            "close_without_sigma_lambda_offset_noise_particle_seed_temperature_"
            "gate_blend_selector_or_same_oof_rescue"
        ),
    }


def load_stage1_frozen_resume(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resume = dict(get_nested(config, "data.stage1_frozen_resume") or {})
    file_specs = {str(key): dict(value) for key, value in dict(resume["files"]).items()}
    candidates = [str(value) for value in resume.get("candidates", [])]
    paths: dict[str, Path] = {}
    for key, spec in file_specs.items():
        filename = str(spec["filename"])
        try:
            paths[key] = resolve_existing(filename, candidates)
        except FileNotFoundError:
            if not filename.endswith(".csv.gz"):
                raise
            paths[key] = resolve_existing(filename.removesuffix(".gz"), candidates)
    for key, path in paths.items():
        expected_raw = str(file_specs[key]["raw_sha256"])
        stored_as_gzip = path.suffix == ".gz"
        if stored_as_gzip and sha256_path(path) != expected_raw:
            raise ValueError(f"exp486 Stage 1 frozen resume raw SHA mismatch: {key}")
        expected_decompressed = file_specs[key].get("decompressed_sha256")
        if expected_decompressed is not None and sha256_csv_payload(path) != str(
            expected_decompressed
        ):
            raise ValueError(f"exp486 Stage 1 frozen resume decompressed SHA mismatch: {key}")

    frozen = json.loads(paths["freeze_manifest"].read_text())
    if (
        str(frozen["scientific_contract_sha256"]) != str(resume["scientific_contract_sha256"])
        or not bool(frozen["frozen_before_truth_attachment"])
        or not bool(get_nested(frozen, "sha_readback.pass"))
    ):
        raise ValueError("exp486 Stage 1 frozen resume manifest contract mismatch")

    predictions = pd.read_csv(
        paths["predictions"],
        dtype={"id": str, "well_id": str},
        compression="infer",
    )
    absolute = pd.read_csv(
        paths["absolute_ledger"],
        dtype={"id": str, "well_id": str},
        compression="infer",
    )
    residual = pd.read_csv(
        paths["residual_ledger"],
        dtype={"id": str, "well_id": str},
        compression="infer",
    )
    audit = pd.read_csv(paths["well_audit"], dtype={"well_id": str})
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(predictions) != expected_rows
        or len(absolute) != expected_rows
        or len(residual) != expected_rows
        or predictions["well_id"].nunique() != expected_wells
        or len(audit) != expected_wells
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp486 Stage 1 frozen resume coverage mismatch")
    if list(absolute.columns) != list(ABSOLUTE_LEDGER_COLUMNS):
        raise ValueError("exp486 Stage 1 frozen absolute schema changed")
    if list(residual.columns) != list(RESIDUAL_LEDGER_COLUMNS):
        raise ValueError("exp486 Stage 1 frozen residual schema changed")

    prediction_logical_columns = [
        "id",
        "well_id",
        "row_idx",
        *PREDICTION_COLUMNS,
    ]
    logical_checks = {
        "predictions": dataframe_content_sha(
            predictions,
            prediction_logical_columns,
        ),
        "absolute_ledger": sha256_csv_payload(paths["absolute_ledger"]),
        "residual_ledger": sha256_csv_payload(paths["residual_ledger"]),
    }
    for key, actual in logical_checks.items():
        if actual != str(file_specs[key]["logical_sha256"]):
            raise ValueError(f"exp486 Stage 1 frozen resume logical SHA mismatch: {key}")
    if (
        logical_checks["predictions"] != str(frozen["prediction_logical_sha256"])
        or logical_checks["absolute_ledger"] != str(frozen["absolute_ledger_logical_sha256"])
        or logical_checks["residual_ledger"] != str(frozen["residual_ledger_logical_sha256"])
    ):
        raise ValueError("exp486 Stage 1 frozen resume manifest logical SHA mismatch")

    ledger.record_geometry_safe(expected_rows)
    for well in audit["well_id"].astype(str):
        ledger.freeze(ABSOLUTE_VARIANT, well)
        ledger.freeze(RESIDUAL_VARIANT, well)
    if not ledger.all_frozen:
        raise RuntimeError("exp486 Stage 1 frozen resume did not restore all freezes")

    frozen["resumed_from_kaggle_version"] = int(resume["source_kernel_version"])
    frozen["resume_dataset_source"] = str(resume["dataset_source"])
    frozen["resume_file_paths"] = {key: str(path) for key, path in paths.items()}
    frozen["prediction_artifact"]["path"] = str(paths["predictions"])
    frozen["absolute_ledger_artifact"]["path"] = str(paths["absolute_ledger"])
    frozen["residual_ledger_artifact"]["path"] = str(paths["residual_ledger"])
    frozen["well_audit"]["path"] = str(paths["well_audit"])
    frozen["truth_access_ledger_at_resume"] = ledger.report()
    return predictions, absolute, residual, audit, frozen


def run_stage1(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    if not bool(get_nested(config, "execution.run_stage_1")):
        raise RuntimeError("exp486 Stage 1 is not selected")

    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest = validate_raw_well_identity(config, raw_dir)
    wells = list(raw_manifest["well_ids"])
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(wells) != expected_wells:
        raise ValueError("exp486 Stage 1 well count changed")

    ledger = LeakageLedger(expected_variant_wells=2 * len(wells))
    saved_paths = stage1_saved_input_paths(config)
    resume_from_frozen = bool(get_nested(config, "execution.stage_1_resume_from_frozen_v2", False))
    scientific_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_scientific_contract.json",
        scientific_contract,
    )
    if resume_from_frozen:
        (
            predictions,
            absolute,
            residual,
            audit,
            frozen,
        ) = load_stage1_frozen_resume(config, ledger)
        runtime_to_freeze_current_version = time.time() - started
        geometry_report = {
            "path": saved_paths["exp226_oof_geometry"],
            "columns_parsed_in_current_version_before_resume_freeze": [],
            "source_version_allowlist_verified_by_frozen_manifest": list(GEOMETRY_ALLOWLIST),
            "rows": expected_rows,
            "wells": expected_wells,
        }
    else:
        geometry_path = Path(saved_paths["exp226_oof_geometry"])
        geometry = load_fold_safe_geometry(
            geometry_path,
            config,
            wells=set(wells),
            ledger=ledger,
        )
        if len(geometry) != expected_rows or geometry["well_id"].nunique() != expected_wells:
            raise ValueError("exp486 Stage 1 exp226 geometry coverage mismatch")
        geometry_report = {
            "path": str(geometry_path),
            "decompressed_sha256": sha256_decompressed_csv(geometry_path),
            "columns_parsed_before_freeze": list(GEOMETRY_ALLOWLIST),
            "rows": len(geometry),
            "wells": int(geometry["well_id"].nunique()),
        }
        warm_up_pf_kernels()
        geometry_groups = {
            str(well): group.copy() for well, group in geometry.groupby("well_id", sort=False)
        }
        workers = int(get_nested(config, "runtime.num_workers", 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            frozen_wells = list(
                executor.map(
                    lambda well: decode_target_free_well(
                        str(well),
                        raw_dir,
                        geometry_groups[str(well)],
                        config,
                    ),
                    wells,
                )
            )
        predictions, absolute, residual, audit, frozen = freeze_target_free_outputs(
            frozen_wells,
            output,
            ledger=ledger,
            stage="stage1",
            expected_rows=expected_rows,
            expected_wells=expected_wells,
        )
        runtime_to_freeze_current_version = time.time() - started

    ledger_at_freeze = ledger.report()
    input_report = {
        "split": "train",
        "raw": raw_manifest,
        "exp226_geometry_allowlist": geometry_report,
        "resume_from_frozen_target_free_version_2": resume_from_frozen,
        "resume_dataset": (
            get_nested(config, "data.stage1_frozen_resume.dataset_source")
            if resume_from_frozen
            else None
        ),
        "late_saved_inputs": {
            key: {
                "path": value,
                "content_values_parsed_before_freeze": False,
            }
            for key, value in saved_paths.items()
            if key != "exp226_oof_geometry"
        },
        "stage0_original_result": {
            "status": get_nested(config, "stage_0_result.status"),
            "all_pass": get_nested(config, "stage_0_result.all_pass"),
            "runtime_exception": get_nested(
                config,
                "stage_0_result.runtime_exception",
            ),
            "support_bound_numerical_exception": get_nested(
                config,
                "stage_0_result.support_bound_numerical_exception",
            ),
        },
    }
    input_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_input_manifest.json",
        input_report,
    )

    frozen.update(
        {
            "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
            "scientific_contract_file_sha256": scientific_artifact["raw_sha256"],
            "resume_input_manifest_sha256": input_artifact["raw_sha256"],
            "current_version_resume_from_frozen": resume_from_frozen,
        }
    )
    freeze_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_freeze_manifest.json",
        frozen,
    )

    frame, late_report = attach_truth_late_stage1(
        predictions,
        absolute,
        residual,
        frozen,
        raw_dir=raw_dir,
        config=config,
        ledger=ledger,
        saved_paths=saved_paths,
    )
    (
        primary_metrics,
        by_well_metrics,
        blend_metrics,
        reference_metrics,
    ) = build_stage1_metric_outputs(frame)
    paths = {
        "truth_late_rows": (output / f"{OUTPUT_PREFIX}_stage1_truth_late_rows.csv.gz"),
        "primary_metrics": (output / f"{OUTPUT_PREFIX}_stage1_primary_metrics.csv"),
        "by_well_metrics": (output / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv"),
        "blend_metrics": (output / f"{OUTPUT_PREFIX}_stage1_fixed_hmm_pf_50_50_metrics.csv"),
        "reference_metrics": (output / f"{OUTPUT_PREFIX}_stage1_exp226_reference_metrics.csv"),
        "promotion_gate": (output / f"{OUTPUT_PREFIX}_stage1_promotion_gate.json"),
        "runtime_ledger": (output / f"{OUTPUT_PREFIX}_stage1_runtime_ledger.json"),
    }
    truth_artifact = write_deterministic_gzip_csv(
        frame,
        paths["truth_late_rows"],
    )
    primary_metrics.to_csv(paths["primary_metrics"], index=False)
    by_well_metrics.to_csv(paths["by_well_metrics"], index=False)
    blend_metrics.to_csv(paths["blend_metrics"], index=False)
    reference_metrics.to_csv(paths["reference_metrics"], index=False)

    current_version_runtime_seconds = time.time() - started
    source_version_elapsed_seconds = (
        float(
            get_nested(
                config,
                "data.stage1_frozen_resume.source_kernel_elapsed_to_error_seconds",
            )
        )
        if resume_from_frozen
        else 0.0
    )
    runtime_seconds = source_version_elapsed_seconds + current_version_runtime_seconds
    rss_gb = peak_rss_gb()
    gate = evaluate_stage1_gate(
        config,
        frame,
        absolute,
        residual,
        audit,
        frozen,
        primary_metrics,
        by_well_metrics,
        blend_metrics,
        ledger_at_freeze,
        raw_manifest,
        runtime_seconds,
        rss_gb,
    )
    gate_artifact = write_json(paths["promotion_gate"], gate)
    runtime_ledger = {
        "stage": "stage1_all_well_two_variant_train_side_cv",
        "resume_from_frozen_target_free_version_2": resume_from_frozen,
        "source_version_elapsed_to_error_seconds": (source_version_elapsed_seconds),
        "current_version_seconds_to_restored_freeze": (runtime_to_freeze_current_version),
        "current_version_seconds_through_large_artifacts": (current_version_runtime_seconds),
        "combined_stage1_elapsed_seconds": runtime_seconds,
        "summed_variant_well_seconds": float(audit["wall_seconds"].sum()),
        "peak_rss_gb": rss_gb,
        "runtime_versions": runtime_versions(),
        "kaggle_kernel_version": None,
        "kernel_version_recording": "record_from_kaggle_api_after_run",
        "user_runtime_exception_approved": True,
        "original_runtime_limit_seconds": float(get_nested(config, "runtime.maximum_seconds")),
    }
    runtime_artifact = write_json(paths["runtime_ledger"], runtime_ledger)
    csv_artifacts = {
        name: {
            "path": str(paths[name]),
            "raw_sha256": sha256_path(paths[name]),
        }
        for name in (
            "primary_metrics",
            "by_well_metrics",
            "blend_metrics",
            "reference_metrics",
        )
    }
    artifacts = {
        "scientific_contract": scientific_artifact,
        "input_manifest": input_artifact,
        "prediction": frozen["prediction_artifact"],
        "absolute_ledger": frozen["absolute_ledger_artifact"],
        "residual_ledger": frozen["residual_ledger_artifact"],
        "well_audit": frozen["well_audit"],
        "freeze_manifest": freeze_artifact,
        "truth_late_rows": truth_artifact,
        **csv_artifacts,
        "promotion_gate": gate_artifact,
        "runtime_ledger": runtime_artifact,
    }
    variant_cv = {
        variant: float(_stage1_scope_row(primary_metrics, variant, "overall")["candidate_rmse"])
        for variant in ACTIVE_VARIANTS
    }
    variant_improvement = {
        variant: float(_stage1_scope_row(primary_metrics, variant, "overall")["improvement_ft"])
        for variant in ACTIVE_VARIANTS
    }
    saved_control_rmse = float(
        _stage1_scope_row(
            primary_metrics,
            ACTIVE_VARIANTS[0],
            "overall",
        )["control_rmse"]
    )
    status = (
        "stage1_completed_with_eligible_variants"
        if gate["eligible_variants"]
        else "stage1_all_variants_gate_failed_terminal_close"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage1_all_well_train_side_cv",
        "cv": None,
        "variant_cv": variant_cv,
        "saved_control_rmse": saved_control_rmse,
        "variant_improvement_ft": variant_improvement,
        "eligible_variants": list(gate["eligible_variants"]),
        "selection_policy": "independent_report_only_no_same_oof_winner",
        "public_lb": None,
        "private_lb": None,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "counts": {
            "scientific_variants": 2,
            "candidate_pf_well_runs": int(audit["candidate_pf_well_runs"].sum()),
            "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
            "particle_starts": int(audit["particle_starts"].sum()),
            "saved_control_pf_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "absolute_ledger_sha256": frozen["absolute_ledger_logical_sha256"],
        "residual_ledger_sha256": frozen["residual_ledger_logical_sha256"],
        "late_readout": late_report,
        "promotion_gate": gate,
        "truth_access_ledger": ledger.report(),
        "runtime": runtime_ledger,
        "artifacts": artifacts,
        "deterministic_anchor": False,
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": gate["decision"],
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(
        metrics_output_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": status,
            "metric": "rmse",
            "cv": None,
            "variant_cv": variant_cv,
            "saved_control_rmse": saved_control_rmse,
            "variant_improvement_ft": variant_improvement,
            "eligible_variants": list(gate["eligible_variants"]),
            "public_lb": None,
            "private_lb": None,
            "stage1": True,
            "promotion_gate": gate,
            "prediction_sha256": frozen["prediction_logical_sha256"],
            "notes": (
                "All-well train-side Stage 1 under the user-approved runtime "
                "and numerical-support interpretation exceptions. The two "
                "variants are evaluated independently with no same-OOF winner."
            ),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def selected_stage(config: Mapping[str, Any]) -> str | None:
    value = get_nested(config, "execution.selected_stage")
    if value in (None, "", "null"):
        return None
    if str(value) not in {"stage_0", "stage_1"}:
        raise ValueError(f"unsupported exp486 execution stage: {value}")
    return str(value)


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    if stage == "stage_0":
        return run_stage0(config)
    return run_stage1(config)


# %% [markdown]
# ## 12. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(
    CONFIG,
    require_run_approval=False,
)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "active_variants": get_nested(CONFIG, "model.active_variants"),
            "selected_stage": selected_stage(CONFIG),
            "stage0_candidate_pf_well_runs": get_nested(
                CONFIG,
                "execution.stage_0_candidate_pf_well_runs",
            ),
            "stage1_candidate_pf_well_runs": get_nested(
                CONFIG,
                "execution.stage_1_candidate_pf_well_runs",
            ),
            "seeds_per_variant": get_nested(
                CONFIG,
                "model.fixed_from_exp404_for_both.seeds",
            ),
            "particles_per_seed": get_nested(
                CONFIG,
                "model.fixed_from_exp404_for_both.particles",
            ),
            "control_pf_reruns": get_nested(
                CONFIG,
                "execution.control_pf_well_runs",
            ),
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
            "inference_enabled": get_nested(
                CONFIG,
                "implementation.inference_enabled",
            ),
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT["scientific_contract_sha256"],
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    STAGE_RESULT = run_selected_stage(CONFIG)
    if STAGE_RESULT is None:
        print("exp486 has no active execution stage. Inference and submission remain disabled.")
