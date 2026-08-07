# %% [markdown]
# # exp487 time-varying GR affine likelihood-PF — train
#
# This compact self-contained candidate combines the frozen exp345 causal EKF,
# exp350 fixed-interval extended RTS smoother, and exp404 likelihood-PF.  Both
# affine schedules are generated and frozen before either candidate is scored.
# The same raw GR is intentionally used by the affine update and the PF
# likelihood; this preregistered double-use risk is reported, not tuned away.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contracts
# 2. Notebook-safe configuration, paths, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Scope, saved-input, and leakage-boundary helpers
# 5. Robust prefix affine and outer-fold process-noise helpers
# 6. Exp404 likelihood-PF input preparation
# 7. Causal EKF and bidirectional extended-RTS schedules
# 8. Dynamic-affine likelihood-PF kernel and seed aggregation
# 9. Target-free two-variant generation and content freeze
# 10. Truth-late readout and Stage 0 technical gates
# 11. All-well Stage 1 independent scientific gates
# 12. Guarded orchestration, configuration preview, and generated artifacts

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


EXPERIMENT_NAME = "exp487_time_varying_gr_affine_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
CAUSAL_VARIANT = "causal_ekf_affine_emission"
RTS_VARIANT = "bidirectional_rts_affine_emission"
CAUSAL_PREDICTION = "likpf_scale5_causal_affine"
RTS_PREDICTION = "likpf_scale5_bidirectional_rts_affine"
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
ACTIVE_VARIANTS = (CAUSAL_VARIANT, RTS_VARIANT)
PREDICTION_COLUMNS = (CAUSAL_PREDICTION, RTS_PREDICTION)
SCHEDULE_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "affine_scale_a",
    "affine_intercept_b",
    "raw_gr_update",
    "predictive_nll_identity",
    "predictive_nll_affine",
    "observation_variance",
    "predicted_intercept_b",
    "predicted_log_scale_a",
    "predicted_p00",
    "predicted_p01",
    "predicted_p11",
    "filtered_intercept_b",
    "filtered_log_scale_a",
    "filtered_p00",
    "filtered_p01",
    "filtered_p11",
    "schedule_kind",
)
RTS_EXTRA_COLUMNS = (
    "smoothed_intercept_b",
    "smoothed_log_scale_a",
    "smoothed_p00",
    "smoothed_p01",
    "smoothed_p11",
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP487_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Notebook-safe configuration, paths, and SHA helpers


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
    raise FileNotFoundError(f"exp487 config not found; checked={checked}")


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
            KAGGLE_INPUT_ROOT
            / "competitions"
            / "rogii-wellbore-geology-prediction"
            / "train",
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


def dataframe_content_sha(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    return mapping_sha256([(str(column), str(frame[column].dtype)) for column in frame.columns])


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
    local_matches = sorted(project_root().glob(f"**/{filename}"))
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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"bootstrap asset not found: {filename}")


def parse_row_index(identifier: str) -> int:
    return int(str(identifier).rsplit("_", 1)[1])


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
                f"exp487 execution count changed: {key}={observed}, expected={expected}"
            )
    run_stage0 = bool(get_nested(config, "execution.run_stage_0", False))
    run_stage1 = bool(get_nested(config, "execution.run_stage_1", False))
    if run_stage0 and run_stage1:
        raise ValueError("exp487 permits exactly one active execution stage")
    if run_stage1 and not (
        bool(get_nested(config, "execution.stage_0_completed", False))
        and bool(get_nested(config, "execution.stage_0_all_gates_pass", False))
        and bool(
            get_nested(
                config,
                "execution.stage_1_eligible_pending_separate_user_approval",
                False,
            )
        )
    ):
        raise RuntimeError("exp487 Stage 1 requires a recorded all-PASS Stage 0")
    if bool(get_nested(config, "execution.run_inference", False)) or bool(
        get_nested(config, "execution.create_submission", False)
    ):
        raise ValueError("exp487 inference/submission must remain disabled")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise RuntimeError("exp487 Kaggle push is not approved")
        if run_stage0 and not bool(
            get_nested(config, "execution.stage_0_execution_approved", False)
        ):
            raise RuntimeError("exp487 Stage 0 Kaggle execution is not approved")
        if run_stage1 and not bool(
            get_nested(config, "execution.stage_1_execution_approved", False)
        ):
            raise RuntimeError("exp487 Stage 1 Kaggle execution is not approved")
        if not (run_stage0 or run_stage1):
            raise RuntimeError("exp487 has no approved execution stage selected")
    return counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    affine = dict(get_nested(config, "model.affine_state_common") or {})
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    rts = dict(get_nested(config, "model.bidirectional_rts") or {})
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "implementation_reference": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "causal_reference": "exp345_exp209_time_varying_gr_affine_calibration_hmm",
        "rts_reference": "exp350_exp345_bidirectional_gr_affine_smoother",
        "active_variants": list(ACTIVE_VARIANTS),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "selection_policy": "independent_report_only_no_same_oof_winner",
        "base_path": {
            "source": "saved_exp209_posterior_mean_and_std",
            "decompressed_sha256": str(
                get_nested(
                    config,
                    "data.saved_exp209_base_path.expected_decompressed_sha256",
                )
            ),
            "hmm_rerun": False,
        },
        "affine_state": {
            "state": ["intercept_b", "log_scale_a"],
            "minimum_prefix_pairs": int(affine["minimum_prefix_pairs"]),
            "minimum_typewell_gr_std": float(affine["minimum_typewell_gr_std"]),
            "slope_bounds": [float(value) for value in affine["slope_bounds"]],
            "maximum_prefix_rmse": float(affine["maximum_prefix_rmse"]),
            "trim_quantile": float(affine["trim_quantile"]),
            "robust_iterations": int(affine["robust_iterations"]),
            "process_noise_floor": float(affine["process_noise_numerical_floor"]),
            "schedule_timing": "current_row_posterior_after_one_finite_raw_gr_update",
            "fallback": "identity_a1_b0",
        },
        "rts": {
            "transition": "identity_2x2",
            "gain": "filtered_covariance_times_pinv_next_predicted_covariance",
            "rcond": float(rts["pseudoinverse_rcond"]),
            "terminal": "smoothed_terminal_equals_filtered_terminal",
            "covariance_floor": float(rts["covariance_numerical_floor"]),
        },
        "pf": {
            "particles": int(fixed["particles"]),
            "seeds": int(fixed["seeds"]),
            "temperature": float(fixed["primary_seed_weighting_temperature"]),
            "gr_scale_multiplier": 1.0,
            "sigma_rescaled_by_affine_a": False,
            "emission_clip_z2": float(
                get_nested(config, "model.pf_emission.emission_clip_z2")
            ),
            "base_scale_clip": [float(value) for value in fixed["base_scale_clip"]],
            "initial_position_spread_ft": float(
                fixed["initial_position_spread_ft"]
            ),
            "initial_rate_spread": float(fixed["initial_rate_spread"]),
            "momentum": float(fixed["momentum"]),
            "rate_noise": float(fixed["rate_noise"]),
            "position_noise": float(fixed["position_noise"]),
            "rough_position": float(fixed["rough_position"]),
            "rough_rate": float(fixed["rough_rate"]),
            "resample_fraction": float(fixed["resample_threshold_fraction"]),
            "typewell_grid_step_ft": float(fixed["typewell_grid_step_ft"]),
            "typewell_tvt_pad_ft": float(fixed["typewell_tvt_pad_ft"]),
            "missing_gr_policy": str(fixed["missing_gr_policy"]),
            "output_dtype": str(fixed["output_dtype"]),
        },
        "raw_gr_double_use": "affine_schedule_update_and_particle_likelihood",
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
        "model.affine_state_common.minimum_prefix_pairs": 40,
        "model.affine_state_common.minimum_typewell_gr_std": 5.0,
        "model.affine_state_common.slope_bounds": [0.25, 4.0],
        "model.affine_state_common.maximum_prefix_rmse": 60.0,
        "model.affine_state_common.trim_quantile": 0.9,
        "model.affine_state_common.robust_iterations": 2,
        "model.causal_ekf.schedule_state_timing": (
            "current_row_posterior_after_one_finite_raw_gr_update"
        ),
        "model.bidirectional_rts.pseudoinverse_rcond": 1.0e-12,
        "model.fixed_from_exp404.particles": 500,
        "model.fixed_from_exp404.seeds": 128,
        "model.fixed_from_exp404.primary_seed_weighting_temperature": 5.0,
        "model.fixed_from_exp404.base_scale_clip": [10.0, 60.0],
        "model.fixed_from_exp404.initial_position_spread_ft": 4.5,
        "model.fixed_from_exp404.initial_rate_spread": 0.01,
        "model.fixed_from_exp404.momentum": 0.998,
        "model.fixed_from_exp404.rate_noise": 0.002,
        "model.fixed_from_exp404.position_noise": 0.005,
        "model.fixed_from_exp404.rough_position": 0.1,
        "model.fixed_from_exp404.rough_rate": 0.001,
        "model.fixed_from_exp404.resample_threshold_fraction": 0.5,
        "model.fixed_from_exp404.typewell_grid_step_ft": 0.2,
        "model.fixed_from_exp404.typewell_tvt_pad_ft": 100.0,
        "model.fixed_from_exp404.missing_gr_policy": (
            "linear_interpolate_both_directions_then_typewell_mean"
        ),
        "model.fixed_from_exp404.output_dtype": "float32",
        "model.pf_emission.emission_clip_z2": 600.0,
        "data.saved_control.logical_columns": [
            "id",
            "well_id",
            "row_idx",
            "likpf_scale_5_x1p0",
            "likpf_scale_5_x1p3",
            "likpf_mean_x1p0",
            "likpf_mean_x1p3",
        ],
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    status = str(get_nested(config, "experiment.status"))
    if status not in {
        "implementation_complete_pending_stage0_approval",
        "stage0_authorized_pending_push",
        "stage0_completed_all_pass_pending_stage1_approval",
        "stage0_fail_closed",
        "stage1_authorized_pending_push",
        "stage1_completed_with_eligible_variants",
        "stage1_all_variants_gate_failed_terminal_close",
    }:
        raise ValueError(f"exp487 scientific contract mismatch: status={status!r}")
    for key, required in expected.items():
        observed = get_nested(config, key)
        if observed != required:
            raise ValueError(
                f"exp487 scientific contract mismatch: {key}={observed!r}, "
                f"expected={required!r}"
            )
    forbidden = set(get_nested(config, "guards.forbidden") or [])
    required_forbidden = {
        "affine_parameter_process_noise_or_slope_bound_search",
        "causal_rts_schedule_blend",
        "static_dynamic_schedule_selection",
        "sigma_temperature_particle_or_seed_change",
        "well_or_row_affine_gate",
        "same_oof_variant_winner_selection",
        "blend_or_selector_rescue",
        "same_oof_rescue",
    }
    if forbidden != required_forbidden:
        raise ValueError("exp487 forbidden-rescue contract changed")
    validate_execution_contract(config, require_run_approval=require_run_approval)
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Scope, saved-input, and leakage-boundary helpers


# %%
@dataclass
class LeakageLedger:
    expected_variant_wells: int
    base_rows_before_freeze: int = 0
    process_fold_rows_before_freeze: int = 0
    truth_rows_before_freeze: int = 0
    error_rows_before_freeze: int = 0
    outcome_fold_rows_before_freeze: int = 0
    hidden_role_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    control_rows_after_freeze: int = 0
    outcome_fold_rows_after_freeze: int = 0
    hidden_role_rows_after_freeze: int = 0
    frozen_keys: set[tuple[str, str]] = field(default_factory=set)

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_keys) == self.expected_variant_wells

    def freeze(self, variant: str, well: str) -> None:
        key = (str(variant), str(well))
        if key in self.frozen_keys:
            raise RuntimeError(f"duplicate exp487 freeze key: {key}")
        self.frozen_keys.add(key)

    def record_base(self, rows: int) -> None:
        if self.all_frozen:
            raise RuntimeError("exp487 base path unexpectedly loaded after freeze")
        self.base_rows_before_freeze += int(rows)

    def record_process_fold(self, rows: int) -> None:
        if self.all_frozen:
            raise RuntimeError("exp487 process fold unexpectedly loaded after freeze")
        self.process_fold_rows_before_freeze += int(rows)

    def record_truth(self, rows: int) -> None:
        if self.all_frozen:
            self.truth_rows_after_freeze += int(rows)
        else:
            self.truth_rows_before_freeze += int(rows)

    def record_control(self, rows: int) -> None:
        if not self.all_frozen:
            raise RuntimeError("exp487 control read occurred before both variants froze")
        self.control_rows_after_freeze += int(rows)

    def record_outcome_fold(self, rows: int) -> None:
        if self.all_frozen:
            self.outcome_fold_rows_after_freeze += int(rows)
        else:
            self.outcome_fold_rows_before_freeze += int(rows)

    def record_hidden_roles(self, rows: int) -> None:
        if self.all_frozen:
            self.hidden_role_rows_after_freeze += int(rows)
        else:
            self.hidden_role_rows_before_freeze += int(rows)

    def report(self) -> dict[str, Any]:
        return {
            "expected_variant_wells": int(self.expected_variant_wells),
            "frozen_variant_wells": int(len(self.frozen_keys)),
            "all_frozen": bool(self.all_frozen),
            "allowed_before_freeze": {
                "base_rows": int(self.base_rows_before_freeze),
                "process_fold_rows": int(self.process_fold_rows_before_freeze),
            },
            "forbidden_before_freeze": {
                "truth_rows": int(self.truth_rows_before_freeze),
                "error_rows": int(self.error_rows_before_freeze),
                "outcome_fold_rows": int(self.outcome_fold_rows_before_freeze),
                "hidden_role_rows": int(self.hidden_role_rows_before_freeze),
            },
            "after_freeze": {
                "truth_rows": int(self.truth_rows_after_freeze),
                "control_rows": int(self.control_rows_after_freeze),
                "outcome_fold_rows": int(self.outcome_fold_rows_after_freeze),
                "hidden_role_rows": int(self.hidden_role_rows_after_freeze),
            },
        }


def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("exp487 fixed32 manifest SHA mismatch")
    return path


def load_fixed32_scope(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    path = fixed32_manifest_path(config)
    frame = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    wells = frame["well"].astype(str).tolist()
    if len(wells) != 32 or len(set(wells)) != 32:
        raise ValueError("exp487 fixed32 manifest must contain 32 unique wells")
    return wells, {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "columns_read_before_freeze": ["well"],
        "wells": 32,
    }


def load_fixed32_roles_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("exp487 fixed32 role/fold read requires both variants frozen")
    frame = pd.read_csv(
        fixed32_manifest_path(config),
        usecols=["well", "role", "fold"],
        dtype={"well": str},
    )
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int64)
    ledger.record_outcome_fold(len(frame))
    return frame


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    if "TVT" in frame.columns:
        raise RuntimeError("unknown-suffix truth entered the exp487 target-free frame")
    return frame.reset_index(drop=True)


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.sort_values("TVT", kind="mergesort").reset_index(drop=True)
    frame["GR"] = frame["GR"].ffill().bfill()
    frame = frame.loc[np.isfinite(frame["TVT"]) & np.isfinite(frame["GR"])].copy()
    if len(frame) < 2:
        raise ValueError(f"well={well} has invalid Type Well TVT/GR")
    return frame


def saved_exp209_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.saved_exp209_base_path") or {})
    path = resolve_existing(
        str(spec["filename"]),
        spec.get("candidates", []),
        spec.get("patterns", []),
    )
    if sha256_decompressed_csv(path) != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp487 saved exp209 decompressed SHA mismatch")
    return path


def load_saved_exp209_base(
    config: Mapping[str, Any],
    ledger: LeakageLedger | None = None,
) -> pd.DataFrame:
    spec = dict(get_nested(config, "data.saved_exp209_base_path") or {})
    mean_column = str(spec["mean_column"])
    std_column = str(spec["std_column"])
    frame = pd.read_csv(
        saved_exp209_path(config),
        usecols=["id", "well", mean_column, std_column],
        dtype={"id": str, "well": str},
    ).rename(
        columns={
            "well": "well_id",
            mean_column: "base_mean",
            std_column: "base_std",
        }
    )
    frame["row_idx"] = frame["id"].map(parse_row_index).astype(np.int64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp487 saved exp209 base path has duplicate row identities")
    if not np.isfinite(frame[["base_mean", "base_std"]].to_numpy(np.float64)).all():
        raise ValueError("exp487 saved exp209 base path contains non-finite values")
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or frame["well_id"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
    ):
        raise ValueError("exp487 saved exp209 base path coverage mismatch")
    if ledger is not None:
        ledger.record_base(len(frame))
    return frame


def fold_assignment_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fold_assignment") or {})
    path = resolve_existing(
        str(spec["filename"]),
        spec.get("candidates", []),
        spec.get("patterns", []),
    )
    if sha256_decompressed_csv(path) != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp487 fold assignment decompressed SHA mismatch")
    return path


def load_process_fold_map(
    config: Mapping[str, Any],
    ledger: LeakageLedger | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    spec = dict(get_nested(config, "data.fold_assignment") or {})
    usecols = [str(value) for value in spec["process_noise_usecols"]]
    if usecols != ["well_id", "fold"]:
        raise ValueError("exp487 process-noise fold allowlist changed")
    frame = pd.read_csv(
        fold_assignment_path(config),
        usecols=usecols,
        dtype={"well_id": str},
    )
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int64)
    mapping = frame.groupby("well_id", sort=True)["fold"].first()
    if not frame.groupby("well_id", sort=True)["fold"].nunique().eq(1).all():
        raise ValueError("exp487 each well must have exactly one outer fold")
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or len(mapping) != int(get_nested(config, "validation.expected_wells"))
        or sorted(frame["fold"].unique().tolist()) != expected_folds
    ):
        raise ValueError("exp487 fold assignment row/well/fold coverage mismatch")
    if ledger is not None:
        ledger.record_process_fold(len(mapping))
    return mapping.astype(int).to_dict(), {
        "path": str(fold_assignment_path(config)),
        "decompressed_sha256": sha256_decompressed_csv(fold_assignment_path(config)),
        "columns_read_before_freeze": usecols,
        "rows_read_before_freeze": int(len(frame)),
        "wells": int(len(mapping)),
    }


# %% [markdown]
# ## 5. Robust prefix affine and outer-fold process-noise helpers


# %%
def robust_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    spec = dict(get_nested(config, "model.affine_state_common") or {})
    minimum = int(spec["minimum_prefix_pairs"])
    minimum_std = float(spec["minimum_typewell_gr_std"])
    maximum_rmse = float(spec["maximum_prefix_rmse"])
    trim_quantile = float(spec["trim_quantile"])
    iterations = int(spec["robust_iterations"])
    slope_low, slope_high = (float(value) for value in spec["slope_bounds"])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    pair_count = len(x)
    x_std = float(np.std(x, ddof=0)) if pair_count else float("nan")
    fallback_reason: str | None = None
    if pair_count < minimum:
        fallback_reason = "insufficient_pairs"
    elif not math.isfinite(x_std) or x_std < minimum_std:
        fallback_reason = "insufficient_typewell_gr_std"
    keep = np.ones(pair_count, dtype=bool)
    beta = np.array([0.0, 1.0], dtype=np.float64)
    fit_rmse = float("nan")
    all_pair_rmse = float("nan")
    if fallback_reason is None:
        for _ in range(iterations):
            design = np.column_stack([np.ones(int(keep.sum())), x[keep]])
            beta = np.linalg.lstsq(design, y[keep], rcond=None)[0]
            slope = float(np.clip(float(beta[1]), slope_low, slope_high))
            intercept = float(np.mean(y[keep] - slope * x[keep]))
            beta = np.array([intercept, slope], dtype=np.float64)
            residual_abs = np.abs(y - (intercept + slope * x))
            threshold = float(np.quantile(residual_abs, trim_quantile))
            next_keep = residual_abs <= threshold
            if int(next_keep.sum()) < minimum:
                break
            keep = next_keep
        residual_all = y - (beta[0] + beta[1] * x)
        fit_rmse = float(np.sqrt(np.mean(residual_all[keep] ** 2)))
        all_pair_rmse = float(np.sqrt(np.mean(residual_all**2)))
        if not math.isfinite(fit_rmse) or fit_rmse > maximum_rmse:
            fallback_reason = "prefix_rmse_above_limit"
    if fallback_reason is not None:
        return {
            "valid": False,
            "fallback_reason": fallback_reason,
            "pair_count": pair_count,
            "kept_pairs": int(keep.sum()) if pair_count else 0,
            "typewell_gr_std": x_std,
            "intercept_b": 0.0,
            "scale_a": 1.0,
            "log_scale_a": 0.0,
            "fit_rmse": fit_rmse,
            "all_pair_rmse": all_pair_rmse,
            "covariance": np.zeros((2, 2), dtype=np.float64),
        }
    design = np.column_stack([np.ones(int(keep.sum())), x[keep]])
    residual = y[keep] - design @ beta
    residual_variance = float(np.sum(residual**2) / max(1, int(keep.sum()) - 2))
    covariance_beta = residual_variance * np.linalg.pinv(design.T @ design)
    jacobian = np.array(
        [[1.0, 0.0], [0.0, 1.0 / float(beta[1])]],
        dtype=np.float64,
    )
    covariance = jacobian @ covariance_beta @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    return {
        "valid": True,
        "fallback_reason": None,
        "pair_count": pair_count,
        "kept_pairs": int(keep.sum()),
        "typewell_gr_std": x_std,
        "intercept_b": float(beta[0]),
        "scale_a": float(beta[1]),
        "log_scale_a": float(math.log(beta[1])),
        "fit_rmse": fit_rmse,
        "all_pair_rmse": all_pair_rmse,
        "covariance": covariance,
    }


def visible_prefix_pairs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(tvt) & np.isfinite(raw_gr)
    rows = np.flatnonzero(finite)
    x = np.interp(
        tvt[finite],
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    return x, raw_gr[finite], rows


def prefix_process_noise_raw(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    minimum = int(get_nested(config, "model.affine_state_common.minimum_prefix_pairs"))
    x, y, source_rows = visible_prefix_pairs(horizontal, typewell)
    if len(x) < 2 * minimum:
        return {
            "raw_q_intercept": float("nan"),
            "raw_q_log_scale": float("nan"),
            "process_increments": 0,
            "state_fits": 0,
        }
    endpoints = list(range(minimum, len(x) + 1, minimum))
    if endpoints[-1] != len(x):
        endpoints.append(len(x))
    states: list[np.ndarray] = []
    rows: list[int] = []
    for endpoint in endpoints:
        fit = robust_affine_fit(x[:endpoint], y[:endpoint], config)
        if fit["valid"]:
            states.append(
                np.array([fit["intercept_b"], fit["log_scale_a"]], dtype=np.float64)
            )
            rows.append(int(source_rows[endpoint - 1]))
    if len(states) < 2:
        return {
            "raw_q_intercept": float("nan"),
            "raw_q_log_scale": float("nan"),
            "process_increments": 0,
            "state_fits": len(states),
        }
    state_array = np.vstack(states)
    row_delta = np.maximum(np.diff(np.asarray(rows, dtype=np.float64)), 1.0)
    increments = np.diff(state_array, axis=0) ** 2 / row_delta[:, None]
    return {
        "raw_q_intercept": float(np.median(increments[:, 0])),
        "raw_q_log_scale": float(np.median(increments[:, 1])),
        "process_increments": int(len(increments)),
        "state_fits": int(len(states)),
    }


def build_outer_fold_process_noise(
    raw_dir: Path,
    fold_map: Mapping[str, int],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = sorted((str(well), int(fold)) for well, fold in fold_map.items())
    for index, (well, fold) in enumerate(ordered, start=1):
        print(f"[process-noise {index}/{len(ordered)}] well={well}", flush=True)
        horizontal = load_horizontal_without_truth(well, raw_dir)
        typewell = load_typewell(well, raw_dir)
        raw = prefix_process_noise_raw(horizontal, typewell, config)
        rows.append({"well_id": well, "fold": fold, **raw})
    audit = pd.DataFrame(rows)
    pseudocount = 100.0
    floor = float(get_nested(config, "model.affine_state_common.process_noise_numerical_floor"))
    for fold in sorted(audit["fold"].unique()):
        outer = audit.loc[audit["fold"] != fold]
        for raw_column, output_column in (
            ("raw_q_intercept", "q_intercept"),
            ("raw_q_log_scale", "q_log_scale"),
        ):
            finite_outer = pd.to_numeric(outer[raw_column], errors="coerce")
            finite_outer = finite_outer[np.isfinite(finite_outer)]
            if finite_outer.empty:
                raise ValueError(f"fold={fold} has no finite outer-train {raw_column}")
            global_median = max(float(np.median(finite_outer)), floor)
            mask = audit["fold"] == fold
            for row_index in audit.index[mask]:
                raw_value = float(audit.at[row_index, raw_column])
                support = int(audit.at[row_index, "process_increments"])
                alpha = (
                    float(support / (support + pseudocount))
                    if math.isfinite(raw_value)
                    else 0.0
                )
                shrunk = (
                    alpha * max(raw_value, floor) + (1.0 - alpha) * global_median
                )
                audit.at[row_index, f"outer_train_median_{output_column}"] = global_median
                audit.at[row_index, f"shrinkage_alpha_{output_column}"] = alpha
                audit.at[row_index, output_column] = max(float(shrunk), floor)
    required = [
        "fold",
        "process_increments",
        "state_fits",
        "outer_train_median_q_intercept",
        "shrinkage_alpha_q_intercept",
        "q_intercept",
        "outer_train_median_q_log_scale",
        "shrinkage_alpha_q_log_scale",
        "q_log_scale",
    ]
    if not np.isfinite(audit[required].to_numpy(np.float64)).all():
        raise ValueError("exp487 process-noise audit is non-finite after shrinkage")
    return audit.sort_values(["fold", "well_id"], kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 6. Exp404 likelihood-PF input preparation


# %%
def uniform_typewell_grid(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step: float = 0.2,
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
    *,
    clip: tuple[float, float] = (10.0, 60.0),
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires at least one known-prefix row")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    raw_scale = float(np.nanstd(residual))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    base_scale = float(np.clip(raw_scale, clip[0], clip[1]))
    return {
        "raw_scale": raw_scale,
        "base_scale": base_scale,
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "base_clip_min": float(clip[0]),
        "base_clip_max": float(clip[1]),
    }


def exp072_initial_rate(horizontal: pd.DataFrame, *, tail_rows: int = 30) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    tail = known.tail(tail_rows)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    valid = delta_md > 0
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    multiplier: float = 1.0,
    grid_step: float = 0.2,
    base_scale_clip: tuple[float, float] = (10.0, 60.0),
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires non-empty known prefix and unknown suffix")
    known_indices = np.flatnonzero(known_mask)
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    if int(known_indices[-1]) >= int(eval_indices[0]):
        raise ValueError("exp487 requires one contiguous known prefix and unknown suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    last_known_tvt = float(last_known["TVT_input"])
    last_known_md = float(last_known["MD"])
    last_position = last_known_tvt + float(last_known["Z"])
    scale_audit = exp072_base_gr_scale(
        horizontal,
        typewell_tvt,
        typewell_gr,
        clip=base_scale_clip,
    )
    candidate_scale = float(scale_audit["base_scale"]) * float(multiplier)
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    typewell_mean = float(typewell_gr.mean())
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy(np.float64)
    )
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_z = evaluation["Z"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    if not np.isfinite(eval_gr).all():
        raise ValueError("evaluation GR interpolation is not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": eval_z,
        "eval_gr": eval_gr,
        "raw_gr_eval": pd.to_numeric(
            evaluation["GR"],
            errors="coerce",
        ).to_numpy(np.float64),
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - last_known_md,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_position,
        "initial_rate": exp072_initial_rate(horizontal),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "prefix_scale": {
            **scale_audit,
            "sigma_gr": candidate_scale,
        },
        "scale_audit": {
            **scale_audit,
            "candidate_scale": candidate_scale,
            "multiplier": float(multiplier),
            "post_multiplier_clip_applied": False,
            "post_multiplier_clip_count": 0,
        },
    }


# %% [markdown]
# ## 7. Causal EKF and bidirectional extended-RTS schedules


# %%
def typewell_value_and_gradient(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_tvt, inverse = np.unique(
        np.asarray(typewell_tvt, dtype=np.float64),
        return_inverse=True,
    )
    if len(unique_tvt) < 2:
        raise ValueError("Type Well gradient needs at least two unique TVT values")
    gr_sum = np.bincount(inverse, weights=np.asarray(typewell_gr, dtype=np.float64))
    gr_count = np.bincount(inverse)
    unique_gr = gr_sum / np.maximum(gr_count, 1)
    edge_order = 2 if len(unique_tvt) >= 3 else 1
    gradient = np.gradient(unique_gr, unique_tvt, edge_order=edge_order)
    return (
        np.interp(query, unique_tvt, unique_gr),
        np.interp(query, unique_tvt, gradient),
    )


def project_covariance(
    covariance: np.ndarray,
    *,
    floor: float,
) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum_before = float(np.min(eigenvalues))
    if minimum_before < 0.0:
        symmetric = (
            eigenvectors
            @ np.diag(np.maximum(eigenvalues, float(floor)))
            @ eigenvectors.T
        )
        symmetric = 0.5 * (symmetric + symmetric.T)
    return symmetric, minimum_before


def forward_affine_schedule(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prepared: Mapping[str, Any],
    base_mean: np.ndarray,
    base_std: np.ndarray,
    process_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    x_prefix, y_prefix, _ = visible_prefix_pairs(horizontal, typewell)
    fit = robust_affine_fit(x_prefix, y_prefix, config)
    eval_index = np.asarray(prepared["eval_indices"], dtype=np.int64)
    raw_gr = np.asarray(prepared["raw_gr_eval"], dtype=np.float64)
    base_mean = np.asarray(base_mean, dtype=np.float64)
    base_std = np.asarray(base_std, dtype=np.float64)
    if not (len(eval_index) == len(raw_gr) == len(base_mean) == len(base_std)):
        raise ValueError("base path and affine-filter suffix lengths differ")
    suffix_x, suffix_gradient = typewell_value_and_gradient(
        np.asarray(prepared["typewell_tvt"]),
        np.asarray(prepared["typewell_gr"]),
        base_mean,
    )
    sigma_gr = float(prepared["prefix_scale"]["sigma_gr"])
    observation_variance = sigma_gr**2 + (suffix_gradient * base_std) ** 2
    if (
        not np.isfinite(observation_variance).all()
        or np.any(observation_variance <= 0.0)
    ):
        raise ValueError("affine EKF observation variance is invalid")
    length = len(eval_index)
    floor = float(
        get_nested(config, "model.affine_state_common.process_noise_numerical_floor")
    )
    slope_low, slope_high = (
        float(value)
        for value in get_nested(config, "model.affine_state_common.slope_bounds")
    )
    log_low, log_high = math.log(slope_low), math.log(slope_high)
    process_covariance = np.diag(
        [float(process_row["q_intercept"]), float(process_row["q_log_scale"])]
    )
    updated = np.isfinite(raw_gr)
    identity_nll = np.full(length, np.nan, dtype=np.float64)
    forward_nll = np.full(length, np.nan, dtype=np.float64)
    predicted_state = np.empty((length, 2), dtype=np.float64)
    filtered_state = np.empty((length, 2), dtype=np.float64)
    predicted_covariance = np.empty((length, 2, 2), dtype=np.float64)
    filtered_covariance = np.empty((length, 2, 2), dtype=np.float64)
    if not fit["valid"]:
        state = np.array([0.0, 0.0], dtype=np.float64)
        covariance = np.eye(2, dtype=np.float64) * floor
        for index in range(length):
            predicted_state[index] = state
            predicted_covariance[index] = covariance
            filtered_state[index] = state
            filtered_covariance[index] = covariance
            if updated[index]:
                residual = float(raw_gr[index] - suffix_x[index])
                identity_nll[index] = 0.5 * (
                    math.log(2.0 * math.pi * observation_variance[index])
                    + residual**2 / observation_variance[index]
                )
                forward_nll[index] = identity_nll[index]
        scale = np.ones(length, dtype=np.float64)
        boundary_jump = 0.0
        initial_state = state.copy()
        initial_covariance = covariance.copy()
    else:
        state = np.array(
            [fit["intercept_b"], fit["log_scale_a"]],
            dtype=np.float64,
        )
        covariance = np.asarray(fit["covariance"], dtype=np.float64)
        covariance = covariance + np.eye(2, dtype=np.float64) * floor
        initial_state = state.copy()
        initial_covariance = covariance.copy()
        identity_matrix = np.eye(2, dtype=np.float64)
        boundary_jump = 0.0
        for index in range(length):
            predicted = state.copy()
            predicted_p = covariance + process_covariance
            predicted_state[index] = predicted
            predicted_covariance[index] = predicted_p
            predicted_scale = float(math.exp(predicted[1]))
            predicted_mean = predicted[0] + predicted_scale * suffix_x[index]
            observation_gradient = np.array(
                [1.0, predicted_scale * suffix_x[index]],
                dtype=np.float64,
            )
            innovation_variance = float(
                observation_gradient @ predicted_p @ observation_gradient
                + observation_variance[index]
            )
            if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
                raise ValueError("exp487 EKF innovation variance is invalid")
            if updated[index]:
                affine_residual = float(raw_gr[index] - predicted_mean)
                identity_residual = float(raw_gr[index] - suffix_x[index])
                forward_nll[index] = 0.5 * (
                    math.log(2.0 * math.pi * innovation_variance)
                    + affine_residual**2 / innovation_variance
                )
                identity_nll[index] = 0.5 * (
                    math.log(2.0 * math.pi * observation_variance[index])
                    + identity_residual**2 / observation_variance[index]
                )
                gain = predicted_p @ observation_gradient / innovation_variance
                state = predicted + gain * affine_residual
                state[1] = float(np.clip(state[1], log_low, log_high))
                kh = np.outer(gain, observation_gradient)
                covariance = (
                    (identity_matrix - kh)
                    @ predicted_p
                    @ (identity_matrix - kh).T
                    + np.outer(gain, gain) * observation_variance[index]
                )
                covariance = 0.5 * (covariance + covariance.T)
            else:
                state = predicted
                covariance = predicted_p
            if index == 0:
                delta = state - initial_state
                boundary_jump = float(
                    math.sqrt(
                        max(
                            0.0,
                            delta @ np.linalg.pinv(predicted_p) @ delta,
                        )
                    )
                )
            filtered_state[index] = state
            filtered_covariance[index] = covariance
        scale = np.exp(filtered_state[:, 1])
    minimum_covariance_eigenvalue = float(
        min(
            np.min(np.linalg.eigvalsh(predicted_covariance)),
            np.min(np.linalg.eigvalsh(filtered_covariance)),
        )
    )
    frame = pd.DataFrame(
        {
            "row_idx": eval_index,
            "affine_scale_a": scale,
            "affine_intercept_b": filtered_state[:, 0],
            "raw_gr_update": updated,
            "predictive_nll_identity": identity_nll,
            "predictive_nll_affine": forward_nll,
            "observation_variance": observation_variance,
            "predicted_intercept_b": predicted_state[:, 0],
            "predicted_log_scale_a": predicted_state[:, 1],
            "predicted_p00": predicted_covariance[:, 0, 0],
            "predicted_p01": predicted_covariance[:, 0, 1],
            "predicted_p11": predicted_covariance[:, 1, 1],
            "filtered_intercept_b": filtered_state[:, 0],
            "filtered_log_scale_a": filtered_state[:, 1],
            "filtered_p00": filtered_covariance[:, 0, 0],
            "filtered_p01": filtered_covariance[:, 0, 1],
            "filtered_p11": filtered_covariance[:, 1, 1],
            "schedule_kind": CAUSAL_VARIANT,
        }
    )
    audit = {
        **{key: value for key, value in fit.items() if key != "covariance"},
        "fallback": not bool(fit["valid"]),
        "q_intercept": float(process_row["q_intercept"]),
        "q_log_scale": float(process_row["q_log_scale"]),
        "forward_boundary_jump_sigma": boundary_jump,
        "finite_updates": int(updated.sum()),
        "missing_updates_skipped": int((~updated).sum()),
        "minimum_covariance_eigenvalue_before_floor": minimum_covariance_eigenvalue,
        "output_scale_clip_fraction": float(
            np.mean((scale <= slope_low) | (scale >= slope_high))
        ),
    }
    context = {
        "initial_state": initial_state,
        "initial_covariance": initial_covariance,
        "predicted_state": predicted_state,
        "predicted_covariance": predicted_covariance,
        "filtered_state": filtered_state,
        "filtered_covariance": filtered_covariance,
        "suffix_x": suffix_x,
        "raw_gr": raw_gr,
    }
    return frame, audit, context


def forward_schedule_parity(
    causal: pd.DataFrame,
    rts_forward: pd.DataFrame,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    columns = [
        "row_idx",
        "raw_gr_update",
        "predictive_nll_identity",
        "predictive_nll_affine",
        "observation_variance",
        "predicted_intercept_b",
        "predicted_log_scale_a",
        "predicted_p00",
        "predicted_p01",
        "predicted_p11",
        "filtered_intercept_b",
        "filtered_log_scale_a",
        "filtered_p00",
        "filtered_p01",
        "filtered_p11",
    ]
    maximum = 0.0
    for column in columns:
        left = causal[column].to_numpy()
        right = rts_forward[column].to_numpy()
        if left.dtype == bool or right.dtype == bool or column == "row_idx":
            if not np.array_equal(left, right):
                return {
                    "passed": False,
                    "maximum_absolute_difference": float("inf"),
                    "failed_column": column,
                }
            continue
        left = left.astype(np.float64)
        right = right.astype(np.float64)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            return {
                "passed": False,
                "maximum_absolute_difference": float("inf"),
                "failed_column": column,
            }
        finite = np.isfinite(left) & np.isfinite(right)
        difference = (
            float(np.max(np.abs(left[finite] - right[finite])))
            if finite.any()
            else 0.0
        )
        maximum = max(maximum, difference)
    return {
        "passed": bool(maximum <= tolerance),
        "maximum_absolute_difference": maximum,
        "failed_column": None,
    }


def bidirectional_rts_schedule(
    forward: pd.DataFrame,
    forward_audit: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rts = dict(get_nested(config, "model.bidirectional_rts") or {})
    floor = float(rts["covariance_numerical_floor"])
    tolerance = float(rts["covariance_negative_eigen_tolerance"])
    rcond = float(rts["pseudoinverse_rcond"])
    slope_low, slope_high = (
        float(value)
        for value in get_nested(config, "model.affine_state_common.slope_bounds")
    )
    predicted_state = np.asarray(context["predicted_state"], dtype=np.float64)
    predicted_covariance = np.asarray(
        context["predicted_covariance"],
        dtype=np.float64,
    )
    filtered_state = np.asarray(context["filtered_state"], dtype=np.float64)
    filtered_covariance = np.asarray(
        context["filtered_covariance"],
        dtype=np.float64,
    )
    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    minimum_before_floor = float("inf")
    maximum_contraction_eigenvalue = -float("inf")
    fallback = bool(forward_audit["fallback"])
    if not fallback:
        for index in range(len(forward) - 2, -1, -1):
            gain = filtered_covariance[index] @ np.linalg.pinv(
                predicted_covariance[index + 1],
                rcond=rcond,
            )
            smoothed_state[index] = filtered_state[index] + gain @ (
                smoothed_state[index + 1] - predicted_state[index + 1]
            )
            candidate_covariance = filtered_covariance[index] + gain @ (
                smoothed_covariance[index + 1] - predicted_covariance[index + 1]
            ) @ gain.T
            smoothed_covariance[index], minimum = project_covariance(
                candidate_covariance,
                floor=floor,
            )
            minimum_before_floor = min(minimum_before_floor, minimum)
            contraction = np.linalg.eigvalsh(
                smoothed_covariance[index] - filtered_covariance[index]
            )
            maximum_contraction_eigenvalue = max(
                maximum_contraction_eigenvalue,
                float(np.max(contraction)),
            )
    for index in range(len(forward)):
        minimum_before_floor = min(
            minimum_before_floor,
            float(np.min(np.linalg.eigvalsh(smoothed_covariance[index]))),
        )
        maximum_contraction_eigenvalue = max(
            maximum_contraction_eigenvalue,
            float(
                np.max(
                    np.linalg.eigvalsh(
                        smoothed_covariance[index] - filtered_covariance[index]
                    )
                )
            ),
        )
    raw_scale = np.exp(smoothed_state[:, 1])
    scale = np.clip(raw_scale, slope_low, slope_high)
    output = forward.copy()
    output["affine_scale_a"] = scale
    output["affine_intercept_b"] = smoothed_state[:, 0]
    output["smoothed_intercept_b"] = smoothed_state[:, 0]
    output["smoothed_log_scale_a"] = smoothed_state[:, 1]
    output["smoothed_p00"] = smoothed_covariance[:, 0, 0]
    output["smoothed_p01"] = smoothed_covariance[:, 0, 1]
    output["smoothed_p11"] = smoothed_covariance[:, 1, 1]
    output["schedule_kind"] = RTS_VARIANT
    terminal_state_error = float(
        np.max(np.abs(smoothed_state[-1] - filtered_state[-1]))
    )
    terminal_covariance_error = float(
        np.max(np.abs(smoothed_covariance[-1] - filtered_covariance[-1]))
    )
    numeric = output.select_dtypes(include=[np.number]).to_numpy(np.float64)
    finite_or_nan = np.isfinite(numeric) | np.isnan(numeric)
    audit = {
        "smoother_fallback_identity": fallback,
        "terminal_state_max_abs_error": terminal_state_error,
        "terminal_covariance_max_abs_error": terminal_covariance_error,
        "covariance_minimum_eigenvalue_before_floor": minimum_before_floor,
        "covariance_contraction_max_positive_eigenvalue": (
            maximum_contraction_eigenvalue
        ),
        "covariance_tolerance": tolerance,
        "output_scale_clip_fraction": float(np.mean(raw_scale != scale)),
        "finite_or_expected_nan": bool(finite_or_nan.all()),
    }
    return output, audit


# %% [markdown]
# ## 8. Dynamic-affine likelihood-PF kernel and seed aggregation


# %%
@njit(cache=True)
def _interp1(grid: np.ndarray, value: float, minimum: float, step: float) -> float:
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    final = len(grid) - 1
    if index >= final:
        return grid[final]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _pf_affine_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    scale_v: np.ndarray,
    intercept_v: np.ndarray,
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
    typewell_tvt_pad: float,
    emission_clip_z2: float,
) -> tuple[
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
                position[particle] += (
                    rate[particle] * delta_md + position_noise * np.random.randn()
                )
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - typewell_tvt_pad:
                    tvt_value = grid_minimum - typewell_tvt_pad
                    position_clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + typewell_tvt_pad:
                    tvt_value = grid_maximum + typewell_tvt_pad
                    position_clip_counts[seed_index] += 1
                position[particle] = tvt_value + z_v[row]
            average_likelihood = 0.0
            for particle in range(particles):
                typewell_gr = _interp1(
                    grid_gr,
                    position[particle] - z_v[row],
                    grid_minimum,
                    grid_step,
                )
                expected_gr = scale_v[row] * typewell_gr + intercept_v[row]
                zscore = (gr_v[row] - expected_gr) / gr_scale
                squared = zscore * zscore
                if squared > emission_clip_z2:
                    squared = emission_clip_z2
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
            inverse_ess = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
            ess = 1.0 / inverse_ess
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
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = rate[cursor] + rough_rate * np.random.randn()
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles
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
        effective_sample_size,
        resampled,
    )


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


def run_affine_likelihood_pf(
    prepared: Mapping[str, Any],
    schedule: pd.DataFrame,
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
    typewell_tvt_pad: float,
    emission_clip_z2: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if not np.array_equal(
        schedule["row_idx"].to_numpy(np.int64),
        np.asarray(prepared["eval_indices"], dtype=np.int64),
    ):
        raise ValueError("exp487 affine schedule row identity does not match PF suffix")
    output = _pf_affine_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        schedule["affine_scale_a"].to_numpy(np.float64),
        schedule["affine_intercept_b"].to_numpy(np.float64),
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
        float(typewell_tvt_pad),
        float(emission_clip_z2),
    )
    prediction, seed_weights = aggregate_temperature(
        output[0],
        output[1],
        temperature=temperature,
    )
    row_diagnostics = pd.DataFrame(
        {
            "row_idx": schedule["row_idx"].to_numpy(np.int64),
            "effective_sample_size": (
                seed_weights[:, None] * output[5]
            ).sum(axis=0),
            "resampled_seed_fraction": output[6].mean(axis=0),
        }
    )
    diagnostics = {
        "seed_log_likelihood_minimum": float(np.min(output[1])),
        "seed_log_likelihood_maximum": float(np.max(output[1])),
        "seed_weight_maximum": float(np.max(seed_weights)),
        "resampling_count": int(np.sum(output[2])),
        "minimum_effective_sample_size": float(np.min(output[3])),
        "position_clip_count": int(np.sum(output[4])),
    }
    return prediction.astype(np.float32), row_diagnostics, diagnostics


def dynamic_affine_emission_contract() -> dict[str, Any]:
    typewell_gr = 50.0
    scale = 1.2
    intercept = -4.0
    observed_gr = 61.0
    sigma = 20.0
    expected_center = 56.0
    expected_log_likelihood = -0.5 * ((observed_gr - expected_center) / sigma) ** 2
    observed_center = scale * typewell_gr + intercept
    observed_log_likelihood = -0.5 * min(
        ((observed_gr - observed_center) / sigma) ** 2,
        600.0,
    )
    return {
        "observed_center": observed_center,
        "expected_center": expected_center,
        "observed_log_likelihood": observed_log_likelihood,
        "expected_log_likelihood": expected_log_likelihood,
        "sigma_rescaled_by_a": False,
        "pass": bool(
            observed_center == expected_center
            and observed_log_likelihood == expected_log_likelihood
        ),
    }


def warm_up_pf_kernel() -> None:
    md = np.asarray([1.0, 2.0], dtype=np.float64)
    z = np.asarray([0.0, 0.1], dtype=np.float64)
    gr = np.asarray([50.0, 51.0], dtype=np.float64)
    schedule_a = np.asarray([1.0, 1.0], dtype=np.float64)
    schedule_b = np.asarray([0.0, 0.0], dtype=np.float64)
    grid = np.linspace(40.0, 70.0, 101)
    _pf_affine_allseeds(
        md,
        z,
        gr,
        schedule_a,
        schedule_b,
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
        100.0,
        600.0,
    )


# %% [markdown]
# ## 9. Target-free two-variant generation and content freeze


# %%
@dataclass(frozen=True)
class FrozenWell:
    well_id: str
    prediction: pd.DataFrame
    causal_schedule: pd.DataFrame
    rts_schedule: pd.DataFrame
    causal_pf_ledger: pd.DataFrame
    rts_pf_ledger: pd.DataFrame
    audit: dict[str, Any]


def align_saved_base_to_prepared(
    well: str,
    saved_base: pd.DataFrame,
    prepared: Mapping[str, Any],
) -> pd.DataFrame:
    expected_rows = np.asarray(prepared["eval_indices"], dtype=np.int64)
    frame = (
        saved_base.loc[saved_base["well_id"].eq(str(well))]
        .sort_values("row_idx", kind="mergesort")
        .reset_index(drop=True)
    )
    if not np.array_equal(frame["row_idx"].to_numpy(np.int64), expected_rows):
        raise ValueError(f"{well}: saved exp209 row identity does not match raw suffix")
    return frame


def decorate_schedule(
    schedule: pd.DataFrame,
    *,
    well: str,
    kind: str,
) -> pd.DataFrame:
    output = schedule.copy()
    output["schedule_kind"] = str(kind)
    output.insert(0, "suffix_offset", np.arange(len(output), dtype=np.int64))
    output.insert(0, "well_id", str(well))
    output.insert(
        0,
        "id",
        [f"{well}_{int(row)}" for row in output["row_idx"].to_numpy(np.int64)],
    )
    expected = (
        [*SCHEDULE_COLUMNS, *RTS_EXTRA_COLUMNS]
        if kind == RTS_VARIANT
        else list(SCHEDULE_COLUMNS)
    )
    if set(output.columns.astype(str)) != set(expected):
        raise ValueError(f"exp487 {kind} schedule columns changed before ordering")
    return output.loc[:, expected].copy()


def decode_target_free_well(
    well: str,
    raw_dir: Path,
    saved_base: pd.DataFrame,
    process_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> FrozenWell:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        multiplier=1.0,
        grid_step=float(get_nested(config, "model.fixed_from_exp404.typewell_grid_step_ft")),
        base_scale_clip=tuple(
            float(value)
            for value in get_nested(config, "model.fixed_from_exp404.base_scale_clip")
        ),
    )
    base = align_saved_base_to_prepared(well, saved_base, prepared)
    causal_raw, causal_audit, forward_context = forward_affine_schedule(
        horizontal,
        typewell,
        prepared,
        base["base_mean"].to_numpy(np.float64),
        base["base_std"].to_numpy(np.float64),
        process_row,
        config,
    )
    rts_raw, rts_audit = bidirectional_rts_schedule(
        causal_raw,
        causal_audit,
        forward_context,
        config,
    )
    parity = forward_schedule_parity(causal_raw, rts_raw)
    if not parity["passed"]:
        raise RuntimeError(f"{well}: RTS forward record parity failed")
    causal_schedule = decorate_schedule(
        causal_raw,
        well=well,
        kind=CAUSAL_VARIANT,
    )
    rts_schedule = decorate_schedule(
        rts_raw,
        well=well,
        kind=RTS_VARIANT,
    )
    if causal_schedule.columns.astype(str).tolist() != list(SCHEDULE_COLUMNS):
        raise ValueError("exp487 causal schedule schema changed")
    if rts_schedule.columns.astype(str).tolist() != [
        *SCHEDULE_COLUMNS,
        *RTS_EXTRA_COLUMNS,
    ]:
        raise ValueError("exp487 RTS schedule schema changed")
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    seed_base = stable_seed("likpf", "train", str(well))
    common = {
        "particles": int(fixed["particles"]),
        "seeds": int(fixed["seeds"]),
        "seed_base": int(seed_base),
        "temperature": float(fixed["primary_seed_weighting_temperature"]),
        "momentum": float(fixed["momentum"]),
        "rate_noise": float(fixed["rate_noise"]),
        "position_noise": float(fixed["position_noise"]),
        "rough_position": float(fixed["rough_position"]),
        "rough_rate": float(fixed["rough_rate"]),
        "resample_fraction": float(fixed["resample_threshold_fraction"]),
        "initial_spread": float(fixed["initial_position_spread_ft"]),
        "initial_rate_spread": float(fixed["initial_rate_spread"]),
        "typewell_tvt_pad": float(fixed["typewell_tvt_pad_ft"]),
        "emission_clip_z2": float(
            get_nested(config, "model.pf_emission.emission_clip_z2")
        ),
    }
    causal_prediction, causal_pf, causal_diagnostics = run_affine_likelihood_pf(
        prepared,
        causal_raw,
        **common,
    )
    rts_prediction, rts_pf, rts_diagnostics = run_affine_likelihood_pf(
        prepared,
        rts_raw,
        **common,
    )
    eval_indices = np.asarray(prepared["eval_indices"], dtype=np.int64)
    identifiers = [f"{well}_{int(row)}" for row in eval_indices]
    raw_observed = np.asarray(prepared["raw_gr_observed"], dtype=bool)
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
            "base_exp209_mean": base["base_mean"].to_numpy(np.float64),
            "base_exp209_std": base["base_std"].to_numpy(np.float64),
            CAUSAL_PREDICTION: causal_prediction,
            RTS_PREDICTION: rts_prediction,
        }
    )
    if not np.isfinite(
        prediction[
            ["base_exp209_mean", "base_exp209_std", *PREDICTION_COLUMNS]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError(f"{well}: exp487 target-free prediction is non-finite")
    for pf_frame in (causal_pf, rts_pf):
        pf_frame.insert(0, "well_id", str(well))
        pf_frame.insert(
            0,
            "id",
            [f"{well}_{int(row)}" for row in pf_frame["row_idx"].to_numpy(np.int64)],
        )
    seeds = int(fixed["seeds"])
    particles = int(fixed["particles"])
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "eval_rows": int(len(prediction)),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "sigma_gr": float(prepared["scale_audit"]["candidate_scale"]),
        "sigma_rescaled_by_affine_a": False,
        "seed_base_causal": int(seed_base),
        "seed_base_rts": int(seed_base),
        "variant_names_excluded_from_seed": True,
        "scientific_variants": 2,
        "candidate_pf_well_runs": 2,
        "seed_well_trajectories": 2 * seeds,
        "particle_starts": 2 * seeds * particles,
        "seeds_per_variant": seeds,
        "particles_per_seed": particles,
        "causal_prediction_logical_sha256": dataframe_content_sha(
            prediction,
            ["id", "well_id", "row_idx", CAUSAL_PREDICTION],
        ),
        "rts_prediction_logical_sha256": dataframe_content_sha(
            prediction,
            ["id", "well_id", "row_idx", RTS_PREDICTION],
        ),
        "causal_schedule_logical_sha256": dataframe_content_sha(
            causal_schedule,
            causal_schedule.columns,
        ),
        "rts_schedule_logical_sha256": dataframe_content_sha(
            rts_schedule,
            rts_schedule.columns,
        ),
        "forward_parity_pass": bool(parity["passed"]),
        "forward_parity_maximum_absolute_difference": float(
            parity["maximum_absolute_difference"]
        ),
        **{f"causal_{key}": value for key, value in causal_audit.items()},
        **{f"rts_{key}": value for key, value in rts_audit.items()},
        **{f"causal_pf_{key}": value for key, value in causal_diagnostics.items()},
        **{f"rts_pf_{key}": value for key, value in rts_diagnostics.items()},
        "wall_seconds": time.time() - started,
    }
    return FrozenWell(
        well_id=str(well),
        prediction=prediction,
        causal_schedule=causal_schedule,
        rts_schedule=rts_schedule,
        causal_pf_ledger=causal_pf,
        rts_pf_ledger=rts_pf,
        audit=audit,
    )


def freeze_target_free_outputs(
    frozen_wells: Sequence[FrozenWell],
    output: Path,
    *,
    ledger: LeakageLedger,
    stage: str,
    expected_rows: int,
    expected_wells: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    predictions = (
        pd.concat([item.prediction for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    causal_schedule = (
        pd.concat(
            [item.causal_schedule for item in frozen_wells],
            ignore_index=True,
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    rts_schedule = (
        pd.concat([item.rts_schedule for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    causal_pf = (
        pd.concat(
            [item.causal_pf_ledger for item in frozen_wells],
            ignore_index=True,
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    rts_pf = (
        pd.concat([item.rts_pf_ledger for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([item.audit for item in frozen_wells])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    frames = (predictions, causal_schedule, rts_schedule, causal_pf, rts_pf)
    if any(len(frame) != expected_rows for frame in frames):
        raise ValueError(f"exp487 {stage} target-free row coverage mismatch")
    if predictions["well_id"].nunique() != expected_wells:
        raise ValueError(f"exp487 {stage} target-free well coverage mismatch")
    if any(frame["id"].duplicated().any() for frame in frames):
        raise ValueError(f"exp487 {stage} target-free identity is duplicated")
    if not audit["status"].eq("ok").all():
        raise ValueError(f"exp487 {stage} well audit contains a failed well")
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_frames = {
        "predictions": predictions,
        "causal_schedule": causal_schedule,
        "rts_schedule": rts_schedule,
        "causal_pf_ledger": causal_pf,
        "rts_pf_ledger": rts_pf,
    }
    for name, frame in artifact_frames.items():
        path = output / f"{OUTPUT_PREFIX}_{stage}_{name}.csv.gz"
        artifacts[name] = write_deterministic_gzip_csv(frame, path)
    audit_path = output / f"{OUTPUT_PREFIX}_{stage}_well_audit.csv"
    audit.to_csv(audit_path, index=False)
    for item in frozen_wells:
        ledger.freeze(CAUSAL_VARIANT, item.well_id)
        ledger.freeze(RTS_VARIANT, item.well_id)
    if not ledger.all_frozen:
        raise RuntimeError(f"exp487 did not freeze both variants for all {stage} wells")
    logical_columns = [
        "id",
        "well_id",
        "row_idx",
        "base_exp209_mean",
        *PREDICTION_COLUMNS,
    ]
    frozen = {
        "stage": stage,
        "frozen_before_truth_attachment": True,
        "rows": int(len(predictions)),
        "wells": int(predictions["well_id"].nunique()),
        "scientific_variants": 2,
        "prediction_logical_columns": logical_columns,
        "prediction_logical_sha256": dataframe_content_sha(
            predictions,
            logical_columns,
        ),
        "causal_schedule_logical_sha256": dataframe_content_sha(
            causal_schedule,
            causal_schedule.columns,
        ),
        "rts_schedule_logical_sha256": dataframe_content_sha(
            rts_schedule,
            rts_schedule.columns,
        ),
        "causal_pf_ledger_logical_sha256": dataframe_content_sha(
            causal_pf,
            causal_pf.columns,
        ),
        "rts_pf_ledger_logical_sha256": dataframe_content_sha(
            rts_pf,
            rts_pf.columns,
        ),
        "artifacts": artifacts,
        "well_audit": {
            "path": str(audit_path),
            "raw_sha256": sha256_path(audit_path),
        },
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    frozen["freeze_manifest_sha256"] = mapping_sha256(frozen)
    return (
        predictions,
        causal_schedule,
        rts_schedule,
        causal_pf,
        rts_pf,
        audit,
        frozen,
    )


# %% [markdown]
# ## 10. Truth-late readout and Stage 0 technical gates


# %%
def _require_frozen(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("exp487 late readout requires both variants frozen")
    for key in (
        "prediction_logical_sha256",
        "causal_schedule_logical_sha256",
        "rts_schedule_logical_sha256",
        "freeze_manifest_sha256",
    ):
        if len(str(frozen.get(key) or "")) != 64:
            raise RuntimeError(f"exp487 frozen output is missing {key}")


def verify_frozen_content(
    predictions: pd.DataFrame,
    causal_schedule: pd.DataFrame,
    rts_schedule: pd.DataFrame,
    frozen: Mapping[str, Any],
) -> None:
    _require_frozen(frozen)
    if dataframe_content_sha(
        predictions,
        frozen["prediction_logical_columns"],
    ) != str(frozen["prediction_logical_sha256"]):
        raise ValueError("exp487 predictions changed after freeze")
    if dataframe_content_sha(
        causal_schedule,
        causal_schedule.columns,
    ) != str(frozen["causal_schedule_logical_sha256"]):
        raise ValueError("exp487 causal schedule changed after freeze")
    if dataframe_content_sha(
        rts_schedule,
        rts_schedule.columns,
    ) != str(frozen["rts_schedule_logical_sha256"]):
        raise ValueError("exp487 RTS schedule changed after freeze")


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
    if sha256_path(path) != str(spec["expected_raw_sha256"]):
        raise ValueError("exp487 saved control raw SHA mismatch")
    if sha256_decompressed_csv(path) != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp487 saved control decompressed SHA mismatch")
    return path


def load_saved_control_after_freeze(
    config: Mapping[str, Any],
    identifiers: set[str],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    spec = dict(get_nested(config, "data.saved_control") or {})
    source_column = str(spec["prediction_column"])
    logical_columns = [str(column) for column in spec["logical_columns"]]
    if logical_columns != [
        "id",
        "well_id",
        "row_idx",
        "likpf_scale_5_x1p0",
        "likpf_scale_5_x1p3",
        "likpf_mean_x1p0",
        "likpf_mean_x1p3",
    ]:
        raise ValueError("exp487 saved control logical-column contract changed")
    full_frame = pd.read_csv(
        saved_control_path(config),
        usecols=logical_columns,
        dtype={"id": str},
        compression="gzip",
    )
    expected_logical = str(spec.get("expected_logical_sha256") or "")
    if expected_logical and len(expected_logical) != 64:
        raise ValueError("exp487 saved control provenance logical SHA is invalid")
    if (
        full_frame["id"].duplicated().any()
        or not np.isfinite(
            full_frame[
                [column for column in logical_columns if column not in {"id", "well_id"}]
            ].to_numpy(np.float64)
        ).all()
    ):
        raise ValueError("exp487 saved control artifact content is invalid")
    frame = full_frame.loc[
        full_frame["id"].isin(identifiers),
        ["id", source_column],
    ].copy()
    ledger.record_control(len(frame))
    if len(frame) != len(identifiers) or frame["id"].nunique() != len(identifiers):
        raise ValueError("exp487 saved control coverage mismatch")
    return frame.rename(columns={source_column: PRIMARY_CONTROL})


def load_hidden_roles_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("exp487 hidden-like roles require frozen predictions")
    spec = dict(get_nested(config, "data.hidden_like_assignment") or {})
    path = resolve_existing(
        str(spec["filename"]),
        spec.get("candidates", []),
        spec.get("patterns", []),
    )
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("exp487 hidden-like assignment SHA mismatch")
    roles = dict(spec["role_columns"])
    frame = pd.read_csv(
        path,
        usecols=["well_id", *roles.values()],
        dtype={"well_id": str},
    )
    output = pd.DataFrame({"well_id": frame["well_id"].astype(str)})
    if output["well_id"].duplicated().any():
        raise ValueError("exp487 hidden-like assignment has duplicate wells")
    for output_column, source_column in roles.items():
        output[output_column] = (
            frame[source_column].astype(str).str.lower().eq("valid")
        )
        expected_counts = dict(spec["expected_role_counts"][output_column])
        observed_counts = (
            frame[source_column].astype(str).str.lower().value_counts().to_dict()
        )
        if observed_counts != {
            str(key).lower(): int(value) for key, value in expected_counts.items()
        }:
            raise ValueError(
                f"exp487 hidden-like role counts changed: {output_column}"
            )
    ledger.record_hidden_roles(len(output))
    return output


def attach_truth_late(
    predictions: pd.DataFrame,
    causal_schedule: pd.DataFrame,
    rts_schedule: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    stage: str,
    config: Mapping[str, Any],
    raw_dir: Path,
    fold_map: Mapping[str, int],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    verify_frozen_content(predictions, causal_schedule, rts_schedule, frozen)
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
    ).merge(control, on="id", how="inner", validate="one_to_one")
    if stage == "stage0":
        roles = load_fixed32_roles_after_freeze(config, ledger).rename(
            columns={"well": "well_id"}
        )
        frame = frame.merge(
            roles[["well_id", "role", "fold"]],
            on="well_id",
            how="left",
            validate="many_to_one",
        )
    else:
        folds = pd.DataFrame(
            {
                "well_id": list(fold_map),
                "fold": [int(fold_map[well]) for well in fold_map],
            }
        )
        ledger.record_outcome_fold(len(folds))
        frame = frame.merge(
            folds,
            on="well_id",
            how="left",
            validate="many_to_one",
        )
        frame = frame.merge(
            load_hidden_roles_after_freeze(config, ledger),
            on="well_id",
            how="left",
            validate="many_to_one",
        )
    required = [*PREDICTION_COLUMNS, PRIMARY_CONTROL, "true_tvt", "fold"]
    if (
        len(frame) != len(predictions)
        or not np.isfinite(frame[required].to_numpy(np.float64)).all()
        or (
            stage == "stage1"
            and frame[
                ["hidden_like_spatial", "hidden_like_typewell_purged"]
            ].isna().any().any()
        )
    ):
        raise ValueError("exp487 truth-late coverage mismatch")
    return frame


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(truth, dtype=np.float64)
                    - np.asarray(prediction, dtype=np.float64)
                )
                ** 2
            )
        )
    )


def fixed32_descriptive_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    truth = frame["true_tvt"].to_numpy(np.float64)
    for name, column in (
        (CAUSAL_VARIANT, CAUSAL_PREDICTION),
        (RTS_VARIANT, RTS_PREDICTION),
        ("saved_exp404_control", PRIMARY_CONTROL),
    ):
        rows.append(
            {
                "variant": name,
                "rows": int(len(frame)),
                "wells": int(frame["well_id"].nunique()),
                "rmse": rmse(truth, frame[column].to_numpy(np.float64)),
                "descriptive_fixed32_not_cv": True,
            }
        )
    return pd.DataFrame(rows)


def evaluate_stage0_gates(
    predictions: pd.DataFrame,
    causal_schedule: pd.DataFrame,
    rts_schedule: pd.DataFrame,
    causal_pf: pd.DataFrame,
    rts_pf: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    elapsed_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical = dict(get_nested(config, "guards.technical") or {})
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    projected_full_seconds = elapsed_seconds / 64.0 * 1546.0
    forbidden_reads = ledger.report()["forbidden_before_freeze"]
    numeric_causal = causal_schedule.select_dtypes(include=[np.number]).to_numpy(
        np.float64
    )
    numeric_rts = rts_schedule.select_dtypes(include=[np.number]).to_numpy(np.float64)
    allowed_nan_columns = {
        "predictive_nll_identity",
        "predictive_nll_affine",
    }
    causal_required = [
        column
        for column in causal_schedule.select_dtypes(include=[np.number]).columns
        if column not in allowed_nan_columns
    ]
    rts_required = [
        column
        for column in rts_schedule.select_dtypes(include=[np.number]).columns
        if column not in allowed_nan_columns
    ]
    del numeric_causal, numeric_rts
    checks = {
        "exp209_base_path_sha_match": (
            sha256_decompressed_csv(saved_exp209_path(config))
            == str(
                get_nested(
                    config,
                    "data.saved_exp209_base_path.expected_decompressed_sha256",
                )
            )
        ),
        "causal_schedule_contract": bool(
            dynamic_affine_emission_contract()["pass"]
            and audit["causal_missing_updates_skipped"].ge(0).all()
            and audit["causal_finite_updates"].ge(0).all()
        ),
        "rts_forward_parity": bool(
            audit["forward_parity_pass"].all()
            and audit["forward_parity_maximum_absolute_difference"].le(1.0e-10).all()
        ),
        "rts_terminal_contract": bool(
            audit["rts_terminal_state_max_abs_error"].le(1.0e-12).all()
            and audit["rts_terminal_covariance_max_abs_error"].le(1.0e-12).all()
        ),
        "covariance_finite_and_psd": bool(
            np.isfinite(causal_schedule[causal_required].to_numpy(np.float64)).all()
            and np.isfinite(rts_schedule[rts_required].to_numpy(np.float64)).all()
            and audit["causal_minimum_covariance_eigenvalue_before_floor"].ge(
                -float(
                    get_nested(
                        config,
                        "model.bidirectional_rts.covariance_negative_eigen_tolerance",
                    )
                )
            ).all()
            and audit["rts_covariance_minimum_eigenvalue_before_floor"].ge(
                -float(
                    get_nested(
                        config,
                        "model.bidirectional_rts.covariance_negative_eigen_tolerance",
                    )
                )
            ).all()
            and audit[
                "rts_covariance_contraction_max_positive_eigenvalue"
            ].le(
                float(
                    get_nested(
                        config,
                        "model.bidirectional_rts.covariance_negative_eigen_tolerance",
                    )
                )
            ).all()
        ),
        "output_scale_clip_fraction": bool(
            audit["causal_output_scale_clip_fraction"].le(
                float(technical["maximum_output_scale_clip_fraction"])
            ).all()
            and audit["rts_output_scale_clip_fraction"].le(
                float(technical["maximum_output_scale_clip_fraction"])
            ).all()
        ),
        "finite_prediction_coverage": bool(
            len(predictions) == expected_rows
            and np.isfinite(
                predictions[list(PREDICTION_COLUMNS)].to_numpy(np.float64)
            ).all()
        ),
        "finite_pf_ledgers": bool(
            np.isfinite(causal_pf.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(rts_pf.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "boundary_jump_reported": bool(
            np.isfinite(audit["causal_forward_boundary_jump_sigma"]).all()
        ),
        "stable_seed_identity": bool(
            audit["seed_base_causal"].eq(audit["seed_base_rts"]).all()
            and audit["variant_names_excluded_from_seed"].all()
        ),
        "execution_count_match": bool(
            int(audit["candidate_pf_well_runs"].sum()) == 64
            and int(audit["seed_well_trajectories"].sum()) == 8192
            and int(audit["particle_starts"].sum()) == 4096000
        ),
        "truth_error_outcome_fold_hidden_reads_before_freeze_zero": bool(
            all(int(value) == 0 for value in forbidden_reads.values())
        ),
        "freeze_sha_contract": bool(
            all(
                len(str(frozen[key])) == 64
                for key in (
                    "prediction_logical_sha256",
                    "causal_schedule_logical_sha256",
                    "rts_schedule_logical_sha256",
                    "freeze_manifest_sha256",
                )
            )
        ),
        "runtime_projection": bool(
            projected_full_seconds <= float(technical["maximum_seconds_full_projection"])
        ),
        "peak_rss": bool(rss_gb <= float(technical["maximum_peak_rss_gb"])),
    }
    return {
        "stage": "stage0_fixed32_technical_schedule_preflight_not_cv",
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "stage1_eligible_pending_separate_user_approval": bool(all(checks.values())),
        "measurements": {
            "elapsed_seconds": elapsed_seconds,
            "projected_full_seconds": projected_full_seconds,
            "peak_rss_gb": rss_gb,
            "fallback_wells": int(audit["causal_fallback"].sum()),
            "causal_scale_clip_fraction_maximum": float(
                audit["causal_output_scale_clip_fraction"].max()
            ),
            "rts_scale_clip_fraction_maximum": float(
                audit["rts_output_scale_clip_fraction"].max()
            ),
            "causal_boundary_jump_sigma_p95": float(
                audit["causal_forward_boundary_jump_sigma"].quantile(0.95)
            ),
        },
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 11. All-well Stage 1 independent scientific gates


# %%
def stage1_metric_scopes(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    return [
        ("overall", np.ones(len(frame), dtype=bool)),
        *[
            (
                f"fold_{fold}",
                frame["fold"].to_numpy(np.int64) == int(fold),
            )
            for fold in sorted(frame["fold"].unique())
        ],
        ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
        ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
        (
            "missing_fraction_high",
            frame["well_missing_fraction"].to_numpy(np.float64) >= 0.5,
        ),
        (
            "md_since_1000_plus",
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


def build_stage1_metric_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    truth_all = frame["true_tvt"].to_numpy(np.float64)
    for variant, prediction_column in (
        (CAUSAL_VARIANT, CAUSAL_PREDICTION),
        (RTS_VARIANT, RTS_PREDICTION),
    ):
        for scope, mask in stage1_metric_scopes(frame):
            count = int(mask.sum())
            if count == 0:
                rows.append(
                    {
                        "variant": variant,
                        "scope": scope,
                        "rows": 0,
                        "wells": 0,
                        "candidate_rmse": np.nan,
                        "control_rmse": np.nan,
                        "improvement_ft": np.nan,
                    }
                )
                continue
            truth = truth_all[mask]
            candidate_rmse = rmse(
                truth,
                frame.loc[mask, prediction_column].to_numpy(np.float64),
            )
            control_rmse = rmse(
                truth,
                frame.loc[mask, PRIMARY_CONTROL].to_numpy(np.float64),
            )
            rows.append(
                {
                    "variant": variant,
                    "scope": scope,
                    "rows": count,
                    "wells": int(frame.loc[mask, "well_id"].nunique()),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "improvement_ft": control_rmse - candidate_rmse,
                }
            )
    primary = pd.DataFrame(rows)
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        control_rmse = rmse(
            truth,
            group[PRIMARY_CONTROL].to_numpy(np.float64),
        )
        for variant, prediction_column in (
            (CAUSAL_VARIANT, CAUSAL_PREDICTION),
            (RTS_VARIANT, RTS_PREDICTION),
        ):
            candidate_rmse = rmse(
                truth,
                group[prediction_column].to_numpy(np.float64),
            )
            by_well_rows.append(
                {
                    "well_id": str(well),
                    "variant": variant,
                    "fold": int(group["fold"].iloc[0]),
                    "rows": int(len(group)),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "delta_rmse": candidate_rmse - control_rmse,
                }
            )
    blend_rows: list[dict[str, Any]] = []
    control_blend = 0.5 * (
        frame["base_exp209_mean"].to_numpy(np.float64)
        + frame[PRIMARY_CONTROL].to_numpy(np.float64)
    )
    control_blend_rmse = rmse(truth_all, control_blend)
    for variant, prediction_column in (
        (CAUSAL_VARIANT, CAUSAL_PREDICTION),
        (RTS_VARIANT, RTS_PREDICTION),
    ):
        candidate_blend = 0.5 * (
            frame["base_exp209_mean"].to_numpy(np.float64)
            + frame[prediction_column].to_numpy(np.float64)
        )
        candidate_blend_rmse = rmse(truth_all, candidate_blend)
        blend_rows.append(
            {
                "variant": variant,
                "rows": int(len(frame)),
                "candidate_hmm_pf_50_50_rmse": candidate_blend_rmse,
                "control_hmm_pf_50_50_rmse": control_blend_rmse,
                "improvement_ft": control_blend_rmse - candidate_blend_rmse,
            }
        )
    return primary, pd.DataFrame(by_well_rows), pd.DataFrame(blend_rows)


def _scope_row(
    primary: pd.DataFrame,
    variant: str,
    scope: str,
) -> pd.Series:
    rows = primary.loc[
        primary["variant"].eq(variant) & primary["scope"].eq(scope)
    ]
    if len(rows) != 1:
        raise ValueError(f"exp487 missing metric row variant={variant}, scope={scope}")
    return rows.iloc[0]


def evaluate_stage1_gate(
    primary: pd.DataFrame,
    by_well: pd.DataFrame,
    blend: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    elapsed_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    guard = dict(get_nested(config, "guards.scientific_each_variant") or {})
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    gates: dict[str, Any] = {}
    for variant in ACTIVE_VARIANTS:
        overall = _scope_row(primary, variant, "overall")
        observed = _scope_row(primary, variant, "raw_gr_observed")
        missing = _scope_row(primary, variant, "raw_gr_missing")
        high_missing = _scope_row(primary, variant, "missing_fraction_high")
        long_tail = _scope_row(primary, variant, "md_since_1000_plus")
        spatial = _scope_row(primary, variant, "hidden_like_spatial")
        purged = _scope_row(primary, variant, "hidden_like_typewell_purged")
        fold_rows = primary.loc[
            primary["variant"].eq(variant)
            & primary["scope"].isin([f"fold_{fold}" for fold in expected_folds])
        ]
        well_rows = by_well.loc[by_well["variant"].eq(variant)]
        blend_row = blend.loc[blend["variant"].eq(variant)].iloc[0]
        checks = {
            "pooled_gain": float(overall["improvement_ft"])
            >= float(guard["minimum_pooled_rmse_gain_vs_control_ft"]),
            "improved_folds": int((fold_rows["improvement_ft"] > 0.0).sum())
            >= int(guard["minimum_improved_folds"]),
            "raw_gr_observed_gain": float(observed["improvement_ft"])
            >= float(guard["minimum_raw_gr_observed_gain_ft"]),
            "raw_gr_missing_non_regression": float(missing["improvement_ft"])
            >= -float(guard["maximum_raw_gr_missing_regression_ft"]),
            "high_missing_non_regression": float(high_missing["improvement_ft"])
            >= -float(guard["maximum_high_missing_well_regression_ft"]),
            "long_tail_non_regression": float(long_tail["improvement_ft"])
            >= -float(guard["maximum_long_tail_1000_plus_regression_ft"]),
            "hidden_like_spatial_non_regression": float(spatial["improvement_ft"])
            >= -float(guard["maximum_hidden_like_spatial_regression_ft"]),
            "hidden_like_typewell_purged_non_regression": float(
                purged["improvement_ft"]
            )
            >= -float(
                guard["maximum_hidden_like_typewell_purged_regression_ft"]
            ),
            "by_well_delta_p95": float(well_rows["delta_rmse"].quantile(0.95))
            <= float(guard["maximum_by_well_delta_p95_ft"]),
            "worst_well": float(well_rows["delta_rmse"].max())
            <= float(guard["maximum_worst_well_regression_ft"]),
            "fixed_hmm_pf_50_50_non_regression": float(blend_row["improvement_ft"])
            >= -float(guard["maximum_fixed_hmm_pf_50_50_regression_ft"]),
        }
        gates[variant] = {
            "checks": checks,
            "passed": bool(all(checks.values())),
            "measurements": {
                "pooled_gain_ft": float(overall["improvement_ft"]),
                "improved_folds": int((fold_rows["improvement_ft"] > 0.0).sum()),
                "raw_gr_observed_gain_ft": float(observed["improvement_ft"]),
                "raw_gr_missing_gain_ft": float(missing["improvement_ft"]),
                "missing_fraction_high_gain_ft": float(
                    high_missing["improvement_ft"]
                ),
                "md_since_1000_plus_gain_ft": float(long_tail["improvement_ft"]),
                "hidden_like_spatial_gain_ft": float(spatial["improvement_ft"]),
                "hidden_like_typewell_purged_gain_ft": float(
                    purged["improvement_ft"]
                ),
                "by_well_delta_p95_ft": float(
                    well_rows["delta_rmse"].quantile(0.95)
                ),
                "worst_well_delta_ft": float(well_rows["delta_rmse"].max()),
                "fixed_hmm_pf_50_50_gain_ft": float(blend_row["improvement_ft"]),
            },
        }
    execution_match = bool(
        int(audit["candidate_pf_well_runs"].sum()) == 1546
        and int(audit["seed_well_trajectories"].sum()) == 197888
        and int(audit["particle_starts"].sum()) == 98944000
    )
    control_rmse = float(
        _scope_row(primary, CAUSAL_VARIANT, "overall")["control_rmse"]
    )
    blend_control_rmse = float(blend["control_hmm_pf_50_50_rmse"].iloc[0])
    baseline_tolerance = 1.0e-6
    technical_checks = {
        "execution_count_match": execution_match,
        "saved_exp404_control_rmse_parity": bool(
            abs(
                control_rmse
                - float(get_nested(config, "validation.primary_control_rmse_ft"))
            )
            <= baseline_tolerance
        ),
        "fixed_hmm_pf_control_rmse_parity": bool(
            abs(
                blend_control_rmse
                - float(
                    get_nested(
                        config,
                        "validation.fixed_hmm_pf_50_50_control_rmse_ft",
                    )
                )
            )
            <= baseline_tolerance
        ),
        "finite_metrics": bool(
            np.isfinite(
                primary[
                    ["candidate_rmse", "control_rmse", "improvement_ft"]
                ].to_numpy(np.float64)
            ).all()
        ),
        "truth_late_contract": bool(
            all(
                int(value) == 0
                for value in ledger.report()["forbidden_before_freeze"].values()
            )
        ),
        "runtime": bool(
            elapsed_seconds <= float(get_nested(config, "runtime.maximum_seconds"))
        ),
        "peak_rss": bool(
            rss_gb <= float(get_nested(config, "runtime.maximum_peak_rss_gb"))
        ),
    }
    technical_all_pass = bool(all(technical_checks.values()))
    for variant in ACTIVE_VARIANTS:
        gates[variant]["scientific_passed"] = bool(gates[variant]["passed"])
        gates[variant]["passed"] = bool(
            gates[variant]["scientific_passed"] and technical_all_pass
        )
    return {
        "stage": "stage1_all_well_independent_dynamic_affine_pf_cv",
        "variant_gates": gates,
        "eligible_variants": [
            variant for variant in ACTIVE_VARIANTS if gates[variant]["passed"]
        ],
        "selection_within_exp487": "forbidden",
        "technical": {
            **technical_checks,
            "all_pass": technical_all_pass,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_gb": rss_gb,
        },
        "any_variant_passed": bool(
            any(gates[variant]["passed"] for variant in ACTIVE_VARIANTS)
        ),
        "all_variants_passed": bool(
            all(gates[variant]["passed"] for variant in ACTIVE_VARIANTS)
        ),
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 12. Guarded orchestration, configuration preview, and generated artifacts


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
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    actual = typed_dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if (
        len(frame) != int(get_nested(config, "validation.expected_wells"))
        or actual != expected
    ):
        raise ValueError("exp487 current raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": int(len(frame)),
        "content_sha256": actual,
        "well_ids": frame["well_id"].astype(str).tolist(),
    }


def require_kaggle_runtime() -> None:
    if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
        raise RuntimeError("exp487 approved execution must run in a Kaggle Notebook")


def selected_stage(config: Mapping[str, Any]) -> str | None:
    run_stage0 = bool(get_nested(config, "execution.run_stage_0", False))
    run_stage1 = bool(get_nested(config, "execution.run_stage_1", False))
    selected = get_nested(config, "execution.selected_stage")
    if run_stage0:
        if selected not in {"stage_0", "stage0"}:
            raise ValueError("exp487 selected_stage must be stage_0")
        return "stage0"
    if run_stage1:
        if selected not in {"stage_1", "stage1"}:
            raise ValueError("exp487 selected_stage must be stage_1")
        return "stage1"
    if selected is not None:
        raise ValueError("exp487 selected_stage must be null when no run flag is active")
    return None


def build_input_manifest(
    raw_report: Mapping[str, Any],
    base_path: Path,
    fold_path: Path,
    fixed32_report: Mapping[str, Any] | None,
    scientific_contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    control_spec = dict(get_nested(config, "data.saved_control") or {})
    control_path = saved_control_path(config)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "raw": dict(raw_report),
        "saved_exp209": {
            "path": str(base_path),
            "raw_sha256": sha256_path(base_path),
            "decompressed_sha256": sha256_decompressed_csv(base_path),
        },
        "fold_assignment": {
            "path": str(fold_path),
            "raw_sha256": sha256_path(fold_path),
            "decompressed_sha256": sha256_decompressed_csv(fold_path),
        },
        "fixed32": dict(fixed32_report) if fixed32_report is not None else None,
        "saved_exp404_control": {
            "path": str(control_path),
            "raw_sha256": sha256_path(control_path),
            "decompressed_sha256": sha256_decompressed_csv(control_path),
            "source_prediction_logical_sha256": str(
                control_spec["expected_logical_sha256"]
            ),
            "source_logical_sha_policy": (
                "record_frozen_pre_serialization_provenance; raw and "
                "decompressed artifact SHA are the executable input guards"
            ),
        },
        "scientific_contract_sha256": str(
            scientific_contract["scientific_contract_sha256"]
        ),
        "runtime_versions": runtime_versions(),
    }


def generate_target_free_scope(
    wells: Sequence[str],
    *,
    raw_dir: Path,
    saved_base: pd.DataFrame,
    process_noise: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[FrozenWell]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("exp487 PF execution requires numba")
    warm_up_pf_kernel()
    process_lookup = process_noise.set_index("well_id")
    workers = int(get_nested(config, "runtime.num_workers", 1))

    def build_one(item: tuple[int, str]) -> FrozenWell:
        index, well = item
        print(f"[{index}/{len(wells)}] exp487 well={well}", flush=True)
        result = decode_target_free_well(
            str(well),
            raw_dir,
            saved_base,
            process_lookup.loc[str(well)].to_dict(),
            config,
        )
        print(json.dumps(to_jsonable(result.audit), sort_keys=True), flush=True)
        return result

    enumerated = list(enumerate([str(well) for well in wells], start=1))
    if workers <= 1:
        return [build_one(item) for item in enumerated]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(build_one, enumerated))


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    require_kaggle_runtime()
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_report = validate_raw_well_identity(config, raw_dir)
    if stage == "stage0":
        wells, fixed32_report = load_fixed32_scope(config)
        expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
        expected_wells = 32
    else:
        wells = list(raw_report["well_ids"])
        fixed32_report = None
        expected_rows = int(get_nested(config, "validation.expected_rows"))
        expected_wells = int(get_nested(config, "validation.expected_wells"))
    ledger = LeakageLedger(expected_variant_wells=2 * expected_wells)
    base_path = saved_exp209_path(config)
    saved_base = load_saved_exp209_base(config, ledger)
    fold_map, fold_report = load_process_fold_map(config, ledger)
    if sorted(fold_map) != sorted(raw_report["well_ids"]):
        raise ValueError("exp487 fold assignment and raw well identities differ")
    process_noise = build_outer_fold_process_noise(raw_dir, fold_map, config)
    process_path = output / f"{OUTPUT_PREFIX}_{stage}_process_noise.csv"
    process_noise.to_csv(process_path, index=False)
    process_report = {
        "path": str(process_path),
        "raw_sha256": sha256_path(process_path),
        "logical_sha256": dataframe_content_sha(
            process_noise,
            process_noise.columns,
        ),
        "rows": int(len(process_noise)),
    }
    generated = generate_target_free_scope(
        wells,
        raw_dir=raw_dir,
        saved_base=saved_base,
        process_noise=process_noise,
        config=config,
    )
    (
        predictions,
        causal_schedule,
        rts_schedule,
        causal_pf,
        rts_pf,
        audit,
        frozen,
    ) = freeze_target_free_outputs(
        generated,
        output,
        ledger=ledger,
        stage=stage,
        expected_rows=expected_rows,
        expected_wells=expected_wells,
    )
    frame = attach_truth_late(
        predictions,
        causal_schedule,
        rts_schedule,
        frozen,
        stage=stage,
        config=config,
        raw_dir=raw_dir,
        fold_map=fold_map,
        ledger=ledger,
    )
    elapsed_seconds = time.time() - started
    rss_gb = peak_rss_gb()
    paths: dict[str, Path] = {
        "input_manifest": output / f"{OUTPUT_PREFIX}_{stage}_input_manifest.json",
        "freeze_manifest": output / f"{OUTPUT_PREFIX}_{stage}_freeze_manifest.json",
        "process_noise_manifest": output
        / f"{OUTPUT_PREFIX}_{stage}_process_noise_manifest.json",
        "truth_late_rows": output / f"{OUTPUT_PREFIX}_{stage}_truth_late_rows.csv.gz",
        "gate": output / f"{OUTPUT_PREFIX}_{stage}_gate.json",
    }
    input_manifest = build_input_manifest(
        raw_report,
        base_path,
        fold_assignment_path(config),
        fixed32_report,
        scientific_contract,
        config,
    )
    input_manifest["process_fold_read"] = fold_report
    write_json(paths["input_manifest"], input_manifest)
    write_json(paths["freeze_manifest"], frozen)
    write_json(paths["process_noise_manifest"], process_report)
    write_deterministic_gzip_csv(frame, paths["truth_late_rows"])
    if stage == "stage0":
        summary = fixed32_descriptive_summary(frame)
        summary_path = output / f"{OUTPUT_PREFIX}_{stage}_descriptive_metrics.csv"
        summary.to_csv(summary_path, index=False)
        gate = evaluate_stage0_gates(
            predictions,
            causal_schedule,
            rts_schedule,
            causal_pf,
            rts_pf,
            audit,
            frozen,
            config=config,
            ledger=ledger,
            elapsed_seconds=elapsed_seconds,
            rss_gb=rss_gb,
        )
        metric_payload: dict[str, Any] = {
            "fixed32_descriptive": summary.to_dict(orient="records"),
            "stage0_gate": gate,
        }
    else:
        primary, by_well, blend = build_stage1_metric_outputs(frame)
        primary_path = output / f"{OUTPUT_PREFIX}_{stage}_paired_metrics.csv"
        by_well_path = output / f"{OUTPUT_PREFIX}_{stage}_by_well.csv"
        blend_path = output / f"{OUTPUT_PREFIX}_{stage}_fixed_blend_metrics.csv"
        primary.to_csv(primary_path, index=False)
        by_well.to_csv(by_well_path, index=False)
        blend.to_csv(blend_path, index=False)
        gate = evaluate_stage1_gate(
            primary,
            by_well,
            blend,
            audit,
            config=config,
            ledger=ledger,
            elapsed_seconds=elapsed_seconds,
            rss_gb=rss_gb,
        )
        metric_payload = {
            "stage1_primary": primary.to_dict(orient="records"),
            "stage1_blend": blend.to_dict(orient="records"),
            "stage1_gate": gate,
        }
    write_json(paths["gate"], gate)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0_completed_all_pass_pending_stage1_approval"
            if stage == "stage0" and gate["all_pass"]
            else "stage0_fail_closed"
            if stage == "stage0"
            else "stage1_completed_with_eligible_variants"
            if gate["any_variant_passed"]
            else "stage1_all_variants_gate_failed_terminal_close"
        ),
        "stage": stage,
        "metric": "rmse",
        "public_lb": None,
        "private_lb": None,
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "input_manifest_sha256": sha256_path(paths["input_manifest"]),
        "freeze_manifest_sha256": str(frozen["freeze_manifest_sha256"]),
        "process_noise_logical_sha256": str(process_report["logical_sha256"]),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gb": rss_gb,
        **metric_payload,
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))
    return metrics


# %% [markdown]
# ## 13. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
SELECTED_STAGE = selected_stage(CONFIG)
PREVIEW = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(CONFIG, "experiment.route"),
    "status": get_nested(CONFIG, "experiment.status"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "active_variants": list(ACTIVE_VARIANTS),
    "selected_stage": SELECTED_STAGE,
    "implementation_approved": get_nested(
        CONFIG,
        "implementation.implementation_approval_received",
    ),
    "canonical_train_notebook_adopted": get_nested(
        CONFIG,
        "implementation.canonical_train_notebook_adopted",
    ),
    "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
    "stage_0_execution_approved": get_nested(
        CONFIG,
        "execution.stage_0_execution_approved",
    ),
    "stage_1_execution_approved": get_nested(
        CONFIG,
        "execution.stage_1_execution_approved",
    ),
    "execution_counts": SCIENTIFIC_CONTRACT["execution_counts"],
    "base_path_sha256": get_nested(
        CONFIG,
        "data.saved_exp209_base_path.expected_decompressed_sha256",
    ),
    "raw_gr_double_use_risk": SCIENTIFIC_CONTRACT["raw_gr_double_use"],
    "inference_enabled": get_nested(CONFIG, "implementation.inference_enabled"),
}
print(json.dumps(to_jsonable(PREVIEW), indent=2, sort_keys=True))


# %% [markdown]
# ## 14. Run only an explicitly approved Kaggle CPU stage


# %%
STAGE_RESULT: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    STAGE_RESULT = run_selected_stage(CONFIG)
    if STAGE_RESULT is None:
        print("exp487 implementation preview only; no Kaggle execution stage is enabled")
