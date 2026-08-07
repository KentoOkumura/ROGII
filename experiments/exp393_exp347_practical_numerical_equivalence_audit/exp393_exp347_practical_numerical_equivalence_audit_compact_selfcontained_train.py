# %% [markdown]
# # exp393 exp347 practical numerical equivalence audit
#
# This notebook leaves exp347 and the exp393 Stage 0 FAIL immutable. Stage 0
# remains the fixed practical-equivalence audit. Under the explicit 2026-07-25
# user override, Stage A trains exactly one fold-0 seed-42 neural unary using
# the unchanged exp347 four-window objective, freezes outer-valid predictions
# before truth readout, and applies the unchanged exp347 science gate. No
# parent/control retraining, Stage B, inference, or submission is authorized.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Stage 0/Stage A scientific, execution-cost, and leakage guards
# 4. Complete-well folds, mask-first loading, and frozen window manifests
# 5. Robust GR preprocessing and prefix-context helpers
# 6. Prefix-conditioned multi-scale neural emission
# 7. Scalar and batched exp209 exact forward-backward helpers
# 8. Four-window structured training and outer-train early stopping
# 9. Practical equivalence comparison, diagnostics, and gates
# 10. Freeze-first outer-valid decoding, readout, and Stage A gates
# 11. Stage A orchestration and generated artifacts
# 12. Setup, override, and contract preview
# 13. Run only the explicitly authorized Kaggle GPU stage

# %% [markdown]
# ## 1. Imports and fixed experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


EXPERIMENT_NAME = "exp393_exp347_practical_numerical_equivalence_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
HORIZONTAL_INPUT_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_INPUT_COLUMNS = ("TVT", "GR")
FORBIDDEN_MODEL_COLUMNS = {
    "TVT",
    "formation",
    "target",
    "error",
    "abs_error",
    "oracle",
    "candidate",
}
CONTROL_NAMES = ("real_gr", "circular_shuffle", "geometry_only")
STAGE_A_FOLD = 0


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP393_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers

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
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


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


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp347 config not found in {[str(path) for path in candidates]}")


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    root = project_root()
    candidates = (
        configured,
        root / configured,
        Path.cwd() / configured,
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
    )
    for candidate in candidates:
        if candidate.exists() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None):
                return candidate.resolve()
    raise FileNotFoundError("raw train directory with paired well CSV files was not found")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_stable_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_content_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def require_kaggle_gpu(config: Mapping[str, Any]) -> Any:
    if not KAGGLE_WORKING_ROOT.exists():
        raise RuntimeError("GPU execution is Kaggle Notebook only; local execution is disabled")
    selected_stage = validate_selected_stage(config)
    if selected_stage not in {"stage0_practical_audit", "stage_a_fold0"}:
        raise RuntimeError("exp393 has no approved Kaggle GPU stage selected")
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError("exp393 requires a Kaggle CUDA runtime")
    capability = torch.cuda.get_device_capability(0)
    minimum_major = int(get_nested(config, "model.training.minimum_cuda_capability_major", 7))
    if capability[0] < minimum_major:
        raise RuntimeError(
            f"CUDA capability {capability} is below required major {minimum_major}"
        )
    return torch.device("cuda")


def resolve_existing_path(candidates: Sequence[str | Path], filename: str | None = None) -> Path:
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate
        if candidate.is_dir() and filename and (candidate / filename).is_file():
            return candidate / filename
    if KAGGLE_INPUT_ROOT.exists() and filename:
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"required artifact {filename!r} not found; checked={checked[:20]}")


# %% [markdown]
# ## 3. Scientific, execution-cost, and leakage contract guards

# %%
def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "experiment.route": "ensemble",
        "lineage.parent": "exp347_prefix_gr_unary_batched_window_exact_ssm",
        "validation.strategy": (
            "fixed16_exp347_window_practical_numerical_equivalence_audit"
        ),
        "validation.n_folds": 5,
        "validation.sample_unit": "complete_well",
        "validation.reference_mode": "scalar_fp32_reference",
        "validation.candidate_mode": "batched_fp32_batch4_production",
        "model.sample_unit": "well_window",
        "model.output": "tvt_posterior_mean",
        "model.inference_neighbor_well_data": False,
        "model.candidate_bank": "none",
        "model.test_time_gradient_updates": False,
        "model.unary_model.temporary_models": 1,
        "model.unary_model.persisted_models": 0,
        "model.unary_model.seed": 42,
        "model.unary_model.eval_mode": True,
        "model.unary_model.dropout_enabled_during_comparison": False,
        "model.unary_model.generated_once_then_frozen": True,
        "model.architecture.embedding_dim": 64,
        "model.architecture.dilations": [1, 2, 4, 8, 16],
        "model.architecture.prefix_context_dim": 32,
        "model.state_space.step": 0.35,
        "model.state_space.n_rates": 41,
        "model.state_space.rate_span": 0.10,
        "model.state_space.sig_r": 0.002,
        "model.state_space.sig_p": 0.02,
        "model.state_space.start_sig": 0.75,
        "model.state_space.r0_sig": 0.01,
        "model.state_space.band_pad": 100.0,
        "model.state_space.mom": 0.998,
        "model.state_space.rate_center": "zero",
        "model.state_space.solver": "batched_exact_log_space_forward_backward",
        "model.state_space.primary_readout": "posterior_mean_tvt",
        "model.state_space.training_use": (
            "batched_exact_log_space_forward_backward_inside_fixed_windows"
        ),
        "model.state_space.evaluation_use": (
            "batched_exact_log_space_forward_backward_full_official_suffix"
        ),
        "model.training.objective.name": (
            "gaussian_soft_label_structured_nll_plus_local_ce"
        ),
        "model.training.objective.structured_label_nll_weight": 1.0,
        "model.training.objective.label_observation_distribution": "gaussian",
        "model.training.objective.label_observation_sigma_ft": 0.35,
        "model.training.objective.local_true_state_ce_weight": 0.25,
        "model.training.objective.exact_dp_sweeps_per_window": 4,
        "model.training.windows.length_rows": 256,
        "model.training.windows.scheduled_slots_per_well_per_epoch": 3,
        "model.training.windows.maximum_active_windows_per_well_per_epoch": 3,
        "model.training.windows.manifest_epochs": 8,
        "model.training.windows.maximum_fit_wells_fold0": 556,
        "model.training.windows.maximum_windows_per_epoch": 1668,
        "model.training.windows.maximum_scored_positions_per_epoch": 427008,
        "model.training.windows.selection_uses_truth_or_error": False,
        "model.training.windows.official_prefix_only_in_encoder_context": True,
        "model.training.early_stopping_source": (
            "stable_outer_train_holdout_window_objective_only"
        ),
        "model.training.max_epochs": 8,
        "model.training.batch_size_well_windows": 4,
        "model.training.gradient_accumulation_windows": 1,
        "model.training.batching.windows_per_batch": 4,
        "model.training.batching.gradient_accumulation_steps": 1,
        "model.training.batching.exp332_effective_batch_windows": 4,
        "model.training.batching.batch_formation": "consecutive_frozen_schedule_chunks",
        "model.training.batching.final_incomplete_batch": "inactive_masked_dummy_windows",
        "model.training.batching.loss_reduction": (
            "mean_of_four_per_window_valid_row_normalized_losses"
        ),
        "model.training.amp": True,
        "model.training.dataloader_shuffle": True,
        "model.training.dataloader_workers": 0,
        "data.hidden_like_assignment.valid_role_columns": {
            "hidden_like_spatial": "verification_like_spatial_role",
            "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
        },
        "validation.stage0.fixed_window_count": 16,
        "validation.stage0.fp64_diagnostic_window_count": 4,
        "validation.stage0.temporary_neural_model_count": 1,
        "validation.stage0.unary_generation_count": 1,
        "validation.stage0.freeze_unary_before_mode_comparison": True,
        "validation.stage0.posterior_cell_1e6_gate_policy": (
            "diagnostic_only_do_not_reclassify_exp347"
        ),
        "validation.stage0.failure_action": (
            "close_without_threshold_dtype_batch_padding_or_kernel_rescue"
        ),
    }
    changed = [
        f"{key}={get_nested(config, key)!r} expected {value!r}"
        for key, value in expected.items()
        if get_nested(config, key) != value
    ]
    if changed:
        raise ValueError("exp393 locked scientific contract changed: " + "; ".join(changed))
    inputs = tuple(get_nested(config, "data.horizontal_file_input_columns", []))
    if inputs != HORIZONTAL_INPUT_COLUMNS:
        raise ValueError("horizontal input allowlist must be exactly MD/X/Y/Z/GR/TVT_input")
    forbidden_actions = set(get_nested(config, "model.forbidden", []))
    required_forbidden = {
        "reopen_or_reclassify_exp347",
        "use_posterior_cell_1e6_as_exp393_promotion_gate",
        "tune_practical_gate_after_output",
        "threshold_or_metric_grid",
        "change_batch_size4_production_path",
        "change_padding_transition_grid_objective_boundary_architecture_or_optimizer",
        "compile_or_fused_kernel_rescue",
        "parent_or_control_retraining",
        "stage_a_without_explicit_user_override_after_stage0_fail",
        "inference_or_submission_before_later_promotion_and_approval",
    }
    if not required_forbidden.issubset(forbidden_actions):
        raise ValueError("practical-audit no-rescue prohibitions are incomplete")
    gates = get_nested(config, "validation.stage0.required_checks", {}) or {}
    expected_gates = {
        "posterior_mean_tvt_rmse_ft_max": 0.001,
        "posterior_mean_tvt_p99_abs_ft_max": 0.005,
        "posterior_mean_tvt_max_abs_ft_max": 0.02,
        "marginal_map_state_agreement_min": 0.9999,
        "loss_or_partition_max_abs_error_max": 1e-6,
        "gradient_or_adamw_update_max_abs_error_max": 1e-5,
        "posterior_row_sum_max_abs_error_max": 1e-5,
        "invalid_posterior_or_gradient_max_abs": 0.0,
        "finite_rate_min": 1.0,
        "outer_valid_truth_access_count_max": 0,
        "stage_a_model_count_max": 0,
        "peak_gpu_memory_gb_max": 14.0,
        "audit_runtime_hours_max": 1.0,
    }
    if gates != expected_gates:
        raise ValueError(f"exp393 preregistered practical gates changed: {gates!r}")
    controls = tuple(get_nested(config, "model.controls", []))
    required_controls = {
        "real_gr",
        "stable_within_well_circular_shuffled_typewell_gr_same_trained_model",
        "zero_gr_unary_geometry_only_same_trained_model",
    }
    if len(controls) != 3 or not required_controls.issubset(controls):
        raise ValueError("the three same-model Stage A controls must remain fixed")
    stage_a_gate = get_nested(config, "validation.stage_a_pass", {}) or {}
    expected_stage_a_gate = {
        "minimum_target_in_grid_rate": 0.995,
        "maximum_prefix_clamp_abs_error_ft": 1e-6,
        "minimum_real_nll_gain_vs_circular_shuffle_nats_per_token": 0.05,
        "minimum_real_within10_mass_gain_vs_circular_shuffle": 0.03,
        "minimum_real_rmse_gain_vs_geometry_only_ft": 0.25,
        "minimum_real_rmse_gain_vs_exp209_ft": 0.25,
        "require_well_rmse_p95_non_regression_vs_exp209": True,
        "maximum_worst_well_regression_vs_exp209_ft": 10.0,
        "maximum_fold_runtime_hours": 8.5,
        "maximum_peak_gpu_memory_gb": 14.0,
    }
    if stage_a_gate != expected_stage_a_gate:
        raise ValueError(f"exp393 Stage A science gate changed: {stage_a_gate!r}")
    return {
        "locked_fields": len(expected),
        "preregistered_stage0_gates": len(gates),
        "stage_a_controls": len(controls),
        "stage_a_gates": len(stage_a_gate),
    }


def validate_execution_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = get_nested(config, "execution_contract", {}) or {}
    contract = {
        "active_audits": int(stage.get("stage0_active_audits", -1)),
        "fixed_windows": int(stage.get("stage0_fixed_windows", -1)),
        "temporary_neural_models": int(stage.get("stage0_temporary_neural_models", -1)),
        "persisted_models": int(stage.get("stage0_persisted_models", -1)),
        "trained_folds": int(stage.get("stage0_trained_folds", -1)),
        "lightgbm_configs": int(stage.get("stage0_lightgbm_configs", -1)),
        "boosters": int(stage.get("stage0_boosters", -1)),
        "pf_beam_runs": int(stage.get("stage0_pf_beam_runs", -1)),
        "parent_or_control_retraining": int(
            stage.get("stage0_parent_or_control_retraining", -1)
        ),
    }
    expected = {
        "active_audits": 1,
        "fixed_windows": 16,
        "temporary_neural_models": 1,
        "persisted_models": 0,
        "trained_folds": 0,
        "lightgbm_configs": 0,
        "boosters": 0,
        "pf_beam_runs": 0,
        "parent_or_control_retraining": 0,
    }
    if contract != expected:
        raise ValueError(f"Stage 0 execution contract changed: {contract!r}")
    if not bool(get_nested(config, "execution.implementation_approved", False)):
        raise ValueError("exp393 implementation approval is not recorded")
    if bool(get_nested(config, "execution.inference_approved", False)):
        raise ValueError("Stage 0 implementation must not enable inference")
    if bool(get_nested(config, "execution.submission_approved", False)):
        raise ValueError("Stage 0 implementation must not enable submission")
    if bool(get_nested(config, "execution.control_or_parent_retraining", True)):
        raise ValueError("exp393 must not retrain any parent or control")
    return contract


def validate_stage_a_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = get_nested(config, "execution.stage_a_plan", {}) or {}
    contract = {
        "active_variants": int(stage.get("active_variants", -1)),
        "fold_indices": list(stage.get("fold_indices", [])),
        "active_architectures": int(stage.get("active_architectures", -1)),
        "seeds": list(stage.get("seeds", [])),
        "neural_model_count": int(stage.get("neural_model_count", -1)),
        "persisted_model_count": int(stage.get("persisted_model_count", -1)),
        "lightgbm_config_count": int(stage.get("lightgbm_config_count", -1)),
        "total_boosters": int(stage.get("total_boosters", -1)),
        "control_model_training": int(stage.get("control_model_training", -1)),
        "pf_beam_well_runs": int(get_nested(config, "execution.current_pf_beam_well_runs", -1)),
        "parent_control_retraining": bool(
            get_nested(config, "execution.control_or_parent_retraining", True)
        ),
    }
    expected = {
        "active_variants": 1,
        "fold_indices": [0],
        "active_architectures": 1,
        "seeds": [42],
        "neural_model_count": 1,
        "persisted_model_count": 1,
        "lightgbm_config_count": 0,
        "total_boosters": 0,
        "control_model_training": 0,
        "pf_beam_well_runs": 0,
        "parent_control_retraining": False,
    }
    if contract != expected:
        raise ValueError(f"Stage A cost contract changed: {contract!r}")
    if not bool(get_nested(config, "execution.implementation_approved", False)):
        raise ValueError("Stage A implementation approval is not recorded")
    if bool(get_nested(config, "execution.inference_approved", False)):
        raise ValueError("Stage A implementation must not enable inference")
    if bool(get_nested(config, "execution.submission_approved", False)):
        raise ValueError("Stage A implementation must not enable submission")
    if not bool(get_nested(config, "execution.run_stage_a", False)):
        raise ValueError("Stage A run flag is not enabled")
    return contract


def validate_selected_stage(config: Mapping[str, Any]) -> str:
    selected = str(get_nested(config, "execution.selected_stage", "implementation_only"))
    allowed = {"implementation_only", "stage0_practical_audit", "stage_a_fold0"}
    if selected not in allowed:
        raise ValueError(f"unknown exp393 selected stage: {selected}")
    if selected == "stage0_practical_audit":
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise ValueError("Stage 0 Kaggle push requires separate user approval")
        if not bool(get_nested(config, "execution.run_stage0", False)):
            raise ValueError("Stage 0 run flag requires separate user approval")
    if selected == "stage_a_fold0":
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise ValueError("Stage A Kaggle push requires explicit user approval")
        if not bool(get_nested(config, "execution.stage_a_gpu_approved", False)):
            raise ValueError("Stage A GPU training requires explicit user approval")
        if not bool(get_nested(config, "execution.run_stage_a", False)):
            raise ValueError("Stage A run flag requires explicit user approval")
        if bool(get_nested(config, "execution.run_stage0", False)):
            raise ValueError("Stage 0 must not be rerun with the Stage A override")
        override = get_nested(config, "execution.stage_a_user_override", {}) or {}
        expected_failed_checks = {
            "posterior_mean_tvt_rmse",
            "posterior_mean_tvt_max_abs",
            "posterior_row_sum",
        }
        if not bool(override.get("approved", False)):
            raise ValueError("Stage A requires the explicit post-FAIL user override")
        if str(override.get("approval_source")) != (
            "user_message_adopt_and_proceed_stage_a_2026_07_25"
        ):
            raise ValueError("Stage A override approval source is not fixed")
        if set(override.get("acknowledged_failed_checks", [])) != expected_failed_checks:
            raise ValueError("Stage A override must acknowledge all three failed checks")
        if not bool(override.get("preserve_stage0_failure", False)):
            raise ValueError("Stage A override must preserve the Stage 0 FAIL")
        if bool(get_nested(config, "execution.stage0_gate.passed", True)):
            raise ValueError("Stage 0 FAIL must not be reclassified by the override")
        expected_report_sha = (
            "14f646a9d835bf0d724dc1efcd59c9dbaa7fdaa28a56417819a45b85877794db"
        )
        if get_nested(config, "execution.stage0_gate.report_sha256") != expected_report_sha:
            raise ValueError("Stage A override requires the frozen exp393 Stage 0 report")
    elif bool(get_nested(config, "execution.run_stage_a", False)):
        raise ValueError("Stage A run flag requires selected_stage=stage_a_fold0")
    return selected


def assert_mask_first_schema(columns: Sequence[str], *, allow_truth: bool = False) -> None:
    names = set(columns)
    if not set(HORIZONTAL_INPUT_COLUMNS).issubset(names):
        missing = sorted(set(HORIZONTAL_INPUT_COLUMNS) - names)
        raise ValueError(f"horizontal input is missing required columns: {missing}")
    selected = set(HORIZONTAL_INPUT_COLUMNS)
    if not allow_truth and selected & FORBIDDEN_MODEL_COLUMNS:
        raise ValueError("forbidden truth/candidate column entered the model allowlist")


# %% [markdown]
# ## 4. Complete-well folds, mask-first loading, and frozen window manifests

# %%
@dataclass(frozen=True)
class WellInput:
    well: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_input: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    horizontal_path: Path
    typewell_path: Path


@dataclass(frozen=True)
class WellTruth:
    well: str
    tvt: np.ndarray


@dataclass(frozen=True)
class WindowKey:
    well: str
    epoch: int
    slot: int
    start_row: int
    stop_row: int
    scored_rows: int
    boundary_source: str
    boundary_tvt: float
    boundary_rate: float
    boundary_rate_index: int


@dataclass(frozen=True)
class StateSpec:
    grid: np.ndarray
    rates: np.ndarray
    suffix_index: np.ndarray
    dm: np.ndarray
    dz: np.ndarray
    start_p: float
    init_rate: float
    prefix_end: int
    last_known_tvt: float


@dataclass(frozen=True)
class BatchedStateSpec:
    specs: tuple[StateSpec | None, ...]
    active_mask: np.ndarray
    row_mask: np.ndarray
    position_mask: np.ndarray
    rate_mask: np.ndarray
    grids: np.ndarray
    rates: np.ndarray
    suffix_index: np.ndarray
    dm: np.ndarray
    dz: np.ndarray
    start_p: np.ndarray
    init_rate: np.ndarray


def build_batched_state_spec(
    specs: Sequence[StateSpec | None], *, required_batch_size: int | None = None
) -> BatchedStateSpec:
    values = list(specs)
    if required_batch_size is not None:
        if len(values) > required_batch_size:
            raise ValueError("state batch is larger than the fixed batch size")
        values.extend([None] * (required_batch_size - len(values)))
    if not values or not any(spec is not None for spec in values):
        raise ValueError("state batch must contain at least one active window")
    active_specs = [spec for spec in values if spec is not None]
    row_count = max(len(spec.suffix_index) for spec in active_specs)
    position_count = max(len(spec.grid) for spec in active_specs)
    rate_count = max(len(spec.rates) for spec in active_specs)
    if rate_count < 2:
        raise ValueError("batched exact DP requires at least two rate states")
    batch_size = len(values)
    active_mask = np.zeros(batch_size, dtype=bool)
    row_mask = np.zeros((batch_size, row_count), dtype=bool)
    position_mask = np.zeros((batch_size, position_count), dtype=bool)
    rate_mask = np.zeros((batch_size, rate_count), dtype=bool)
    grids = np.zeros((batch_size, position_count), dtype=np.float64)
    rates = np.zeros((batch_size, rate_count), dtype=np.float64)
    suffix_index = np.full((batch_size, row_count), -1, dtype=np.int64)
    dm = np.ones((batch_size, row_count), dtype=np.float64)
    dz = np.zeros((batch_size, row_count), dtype=np.float64)
    start_p = np.zeros(batch_size, dtype=np.float64)
    init_rate = np.zeros(batch_size, dtype=np.float64)
    for batch_index, spec in enumerate(values):
        if spec is None:
            continue
        rows = len(spec.suffix_index)
        positions = len(spec.grid)
        rate_states = len(spec.rates)
        if rows < 1 or positions < 2 or rate_states < 2:
            raise ValueError("active batched state has an empty state dimension")
        active_mask[batch_index] = True
        row_mask[batch_index, :rows] = True
        position_mask[batch_index, :positions] = True
        rate_mask[batch_index, :rate_states] = True
        grids[batch_index, :positions] = spec.grid
        rates[batch_index, :rate_states] = spec.rates
        suffix_index[batch_index, :rows] = spec.suffix_index
        dm[batch_index, :rows] = spec.dm
        dz[batch_index, :rows] = spec.dz
        start_p[batch_index] = spec.start_p
        init_rate[batch_index] = spec.init_rate
    return BatchedStateSpec(
        specs=tuple(values),
        active_mask=active_mask,
        row_mask=row_mask,
        position_mask=position_mask,
        rate_mask=rate_mask,
        grids=grids,
        rates=rates,
        suffix_index=suffix_index,
        dm=dm,
        dz=dz,
        start_p=start_p,
        init_rate=init_rate,
    )


def batch_padding_manifest(batch: BatchedStateSpec) -> pd.DataFrame:
    rows = []
    padded_rows, padded_positions = batch.row_mask.shape[1], batch.position_mask.shape[1]
    padded_rates = batch.rate_mask.shape[1]
    for batch_index, _spec in enumerate(batch.specs):
        row_count = int(batch.row_mask[batch_index].sum())
        position_count = int(batch.position_mask[batch_index].sum())
        rate_count = int(batch.rate_mask[batch_index].sum())
        rows.append(
            {
                "batch_index": batch_index,
                "active": bool(batch.active_mask[batch_index]),
                "row_count": row_count,
                "position_count": position_count,
                "rate_count": rate_count,
                "padded_rows": padded_rows - row_count,
                "padded_positions": padded_positions - position_count,
                "padded_rates": padded_rates - rate_count,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PreparedView:
    well: str
    view_name: str
    tvt_input: np.ndarray
    horizontal_channels: np.ndarray
    typewell_channels: np.ndarray
    prefix_pairs: np.ndarray
    prefix_huber_summary: np.ndarray
    prefix_pair_count: int
    state: StateSpec


def list_paired_wells(train_dir: Path) -> list[str]:
    wells = []
    for path in sorted(train_dir.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        if (train_dir / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def load_well_input(well: str, train_dir: Path) -> WellInput:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    header = pd.read_csv(horizontal_path, nrows=0).columns.tolist()
    assert_mask_first_schema(header)
    horizontal = pd.read_csv(horizontal_path, usecols=list(HORIZONTAL_INPUT_COLUMNS))
    if list(horizontal.columns) != list(HORIZONTAL_INPUT_COLUMNS):
        horizontal = horizontal.loc[:, list(HORIZONTAL_INPUT_COLUMNS)]
    typewell = pd.read_csv(typewell_path, usecols=list(TYPEWELL_INPUT_COLUMNS))
    typewell = typewell.sort_values("TVT", kind="mergesort").reset_index(drop=True)
    typewell_tvt = numeric_array(typewell, "TVT")
    keep = np.isfinite(typewell_tvt)
    typewell_tvt = typewell_tvt[keep]
    typewell_gr = numeric_array(typewell, "GR")[keep]
    if len(typewell_tvt) < 32 or np.any(np.diff(typewell_tvt) < 0):
        raise ValueError(f"{well}: invalid Type Well TVT axis")
    return WellInput(
        well=well,
        md=numeric_array(horizontal, "MD"),
        x=numeric_array(horizontal, "X"),
        y=numeric_array(horizontal, "Y"),
        z=numeric_array(horizontal, "Z"),
        gr=numeric_array(horizontal, "GR"),
        tvt_input=numeric_array(horizontal, "TVT_input"),
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
    )


def load_well_truth(well: str, train_dir: Path) -> WellTruth:
    path = train_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["TVT"])
    return WellTruth(well=well, tvt=numeric_array(frame, "TVT"))


def build_fold_map(wells: Sequence[str], n_folds: int = 5) -> pd.DataFrame:
    ordered = sorted(str(well) for well in wells)
    if len(ordered) < n_folds:
        raise ValueError("not enough complete wells for GroupKFold")
    groups = np.asarray(ordered)
    dummy_x = np.zeros((len(ordered), 1), dtype=np.float32)
    dummy_y = np.zeros(len(ordered), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (_, valid_idx) in enumerate(splitter.split(dummy_x, dummy_y, groups=groups)):
        for index in valid_idx:
            rows.append({"well": ordered[int(index)], "fold": int(fold)})
    result = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    if len(result) != len(ordered) or result["well"].nunique() != len(ordered):
        raise ValueError("fold map coverage is not one row per well")
    return result


def build_input_manifest(fold_map: pd.DataFrame, train_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in fold_map.sort_values("well", kind="mergesort").itertuples(index=False):
        well = str(row.well)
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        typewell_path = train_dir / f"{well}__typewell.csv"
        horizontal_header = pd.read_csv(horizontal_path, nrows=0).columns.tolist()
        typewell_header = pd.read_csv(typewell_path, nrows=0).columns.tolist()
        assert_mask_first_schema(horizontal_header)
        if not set(TYPEWELL_INPUT_COLUMNS).issubset(typewell_header):
            raise ValueError(f"{well}: Type Well input schema is incomplete")
        rows.append(
            {
                "well": well,
                "fold": int(row.fold),
                "stage_a_role": str(row.stage_a_role),
                "horizontal_source_count": 1,
                "horizontal_file": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_file": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
                "horizontal_model_columns": ",".join(HORIZONTAL_INPUT_COLUMNS),
                "typewell_model_columns": ",".join(TYPEWELL_INPUT_COLUMNS),
                "outer_valid_truth_columns_loaded_before_freeze": 0,
            }
        )
    return pd.DataFrame(rows)


def split_stage_a_wells(fold_map: pd.DataFrame) -> tuple[list[str], list[str]]:
    valid = sorted(fold_map.loc[fold_map["fold"] == STAGE_A_FOLD, "well"].astype(str))
    train = sorted(fold_map.loc[fold_map["fold"] != STAGE_A_FOLD, "well"].astype(str))
    if set(train) & set(valid):
        raise ValueError("outer train/valid well overlap")
    return train, valid


def split_early_stop_wells(
    outer_train_wells: Sequence[str], config: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    fraction = float(get_nested(config, "model.training.early_stopping_holdout_fraction", 0.10))
    minimum = int(get_nested(config, "model.training.early_stopping_minimum_wells", 16))
    count = max(minimum, int(round(len(outer_train_wells) * fraction)))
    count = min(count, max(1, len(outer_train_wells) - 1))
    ordered = sorted(
        outer_train_wells,
        key=lambda well: stable_uint64(EXPERIMENT_NAME, "early-stop", STAGE_A_FOLD, well),
    )
    holdout = sorted(ordered[:count])
    fit = sorted(ordered[count:])
    if not fit or set(fit) & set(holdout):
        raise ValueError("invalid stable outer-train early-stop split")
    return fit, holdout


def prefix_end_index(tvt_input: np.ndarray) -> int:
    finite = np.flatnonzero(np.isfinite(tvt_input))
    if len(finite) < 32:
        raise ValueError("known TVT_input prefix has fewer than 32 finite rows")
    end = int(finite[-1])
    if not np.all(np.isfinite(tvt_input[: end + 1])):
        raise ValueError("TVT_input must be a contiguous visible prefix")
    if np.isfinite(tvt_input[end + 1 :]).any():
        raise ValueError("TVT_input has a finite value after the hidden suffix starts")
    if end >= len(tvt_input) - 1:
        raise ValueError("well has no hidden suffix")
    return end


def select_window_slots(
    well: str,
    suffix_start: int,
    row_count: int,
    epoch: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    window = get_nested(config, "model.training.windows", {}) or {}
    length = int(window["length_rows"])
    slots = int(window["scheduled_slots_per_well_per_epoch"])
    if length != 256 or slots != 3:
        raise ValueError("exp347 window contract must remain 256 rows x 3 slots")
    if not 0 <= suffix_start < row_count:
        raise ValueError(f"{well}: invalid official hidden suffix start")
    selected: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    first_stop = min(row_count, suffix_start + length)
    selected.append((suffix_start, first_stop))
    rows.append(
        {
            "well": str(well),
            "epoch": int(epoch),
            "slot": 0,
            "active": True,
            "start_row": int(suffix_start),
            "stop_row": int(first_stop),
            "scored_rows": int(first_stop - suffix_start),
            "selection_hash": stable_uint64(
                EXPERIMENT_NAME, "window", STAGE_A_FOLD, well, epoch, 0, suffix_start
            ),
        }
    )
    full_starts = range(suffix_start, max(suffix_start, row_count - length + 1))
    for slot in range(1, slots):
        nonoverlapping = [
            int(start)
            for start in full_starts
            if all(start + length <= left or start >= right for left, right in selected)
        ]
        remaining_slots = slots - slot - 1

        def remaining_capacity(candidate: int) -> int:
            intervals = sorted([*selected, (candidate, candidate + length)])
            cursor = suffix_start
            capacity = 0
            for left, right in intervals:
                capacity += max(0, left - cursor) // length
                cursor = max(cursor, right)
            capacity += max(0, row_count - cursor) // length
            return int(capacity)

        eligible = [
            start
            for start in nonoverlapping
            if remaining_capacity(start) >= remaining_slots
        ]
        if eligible:
            start = min(
                eligible,
                key=lambda value: stable_uint64(
                    EXPERIMENT_NAME,
                    "window",
                    STAGE_A_FOLD,
                    well,
                    epoch,
                    slot,
                    value,
                ),
            )
            stop = start + length
            selected.append((start, stop))
            active = True
            score = length
            selection_hash: int | None = stable_uint64(
                EXPERIMENT_NAME, "window", STAGE_A_FOLD, well, epoch, slot, start
            )
        else:
            start = -1
            stop = -1
            active = False
            score = 0
            selection_hash = None
        rows.append(
            {
                "well": str(well),
                "epoch": int(epoch),
                "slot": int(slot),
                "active": bool(active),
                "start_row": int(start),
                "stop_row": int(stop),
                "scored_rows": int(score),
                "selection_hash": selection_hash,
            }
        )
    return rows


def build_window_schedule_manifest(
    wells: Sequence[str],
    train_dir: Path,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    epochs = int(get_nested(config, "model.training.windows.manifest_epochs", 8))
    if epochs != int(get_nested(config, "model.training.max_epochs", 8)):
        raise ValueError("window manifest must cover every fixed training epoch")
    rows: list[dict[str, Any]] = []
    for well in sorted(str(value) for value in wells):
        item = load_well_input(well, train_dir)
        suffix_start = prefix_end_index(item.tvt_input) + 1
        for epoch in range(epochs):
            for row in select_window_slots(well, suffix_start, len(item.md), epoch, config):
                row.update(
                    {
                        "role": str(roles[well]),
                        "fold": STAGE_A_FOLD,
                        "official_prefix_end": int(suffix_start - 1),
                        "official_suffix_start": int(suffix_start),
                        "suffix_rows": int(len(item.md) - suffix_start),
                        "row_count": int(len(item.md)),
                    }
                )
                rows.append(row)
    manifest = pd.DataFrame(rows).sort_values(
        ["role", "well", "epoch", "slot"], kind="mergesort"
    ).reset_index(drop=True)
    active = manifest.loc[manifest["active"]]
    maximum = int(
        get_nested(config, "model.training.windows.maximum_active_windows_per_well_per_epoch", 3)
    )
    counts = active.groupby(["well", "epoch"]).size()
    if len(counts) and int(counts.max()) > maximum:
        raise ValueError("window manifest exceeds the active-window limit")
    for _, group in active.groupby(["well", "epoch"], sort=False):
        intervals = sorted(
            (int(row.start_row), int(row.stop_row))
            for row in group.itertuples(index=False)
        )
        if any(
            right_start < left_stop
            for (_, left_stop), (right_start, _) in zip(
                intervals, intervals[1:], strict=False
            )
        ):
            raise ValueError("window manifest contains overlapping active windows")
    return manifest


# %% [markdown]
# ## 5. Robust GR preprocessing and prefix-context helpers

# %%
def finite_median_mad(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    return center, max(scale, 1.0)


def robust_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(values)
    center, scale = finite_median_mad(values)
    normalized = (values - center) / scale
    normalized[missing] = 0.0
    return normalized.astype(np.float32), missing.astype(np.float32), center, scale


def safe_derivative(values: np.ndarray, axis: np.ndarray, clip_abs: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float32)
    delta_axis = np.gradient(axis)
    valid = np.isfinite(values) & np.isfinite(axis)
    filled = pd.Series(values).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    derivative = np.gradient(filled) / np.where(np.abs(delta_axis) > 1e-9, delta_axis, 1.0)
    derivative[~valid] = 0.0
    return np.clip(derivative, -clip_abs, clip_abs).astype(np.float32)


def interpolate_no_extrapolation(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    finite = np.isfinite(xp) & np.isfinite(fp)
    xp = xp[finite]
    fp = fp[finite]
    if len(xp) < 2:
        return np.zeros(len(x), dtype=np.float64), np.zeros(len(x), dtype=bool)
    unique_x, inverse = np.unique(xp, return_inverse=True)
    if len(unique_x) != len(xp):
        sums = np.bincount(inverse, weights=fp)
        counts = np.bincount(inverse)
        xp, fp = unique_x, sums / np.maximum(counts, 1)
    inside = np.isfinite(x) & (x >= xp[0]) & (x <= xp[-1])
    out = np.zeros(len(x), dtype=np.float64)
    out[inside] = np.interp(np.asarray(x)[inside], xp, fp)
    return out, inside


def huber_affine_summary(
    type_gr: np.ndarray,
    horizontal_gr: np.ndarray,
    *,
    minimum_pairs: int = 32,
    max_iterations: int = 20,
) -> tuple[np.ndarray, int]:
    x = np.asarray(type_gr, dtype=np.float64)
    y = np.asarray(horizontal_gr, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    count = int(valid.sum())
    if count < minimum_pairs or float(np.std(x[valid])) < 1e-6:
        return np.zeros(5, dtype=np.float32), count
    x = x[valid]
    y = y[valid]
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(max_iterations):
        residual = y - design @ coef
        _, scale = finite_median_mad(residual)
        cutoff = 1.345 * scale
        weight = np.ones(len(residual), dtype=np.float64)
        large = np.abs(residual) > cutoff
        weight[large] = cutoff / np.maximum(np.abs(residual[large]), 1e-12)
        weighted = design * np.sqrt(weight)[:, None]
        target = y * np.sqrt(weight)
        updated, *_ = np.linalg.lstsq(weighted, target, rcond=None)
        if np.max(np.abs(updated - coef)) <= 1e-8 * max(1.0, float(np.max(np.abs(coef)))):
            coef = updated
            break
        coef = updated
    residual = y - design @ coef
    _, residual_scale = finite_median_mad(residual)
    y_scale = max(finite_median_mad(y)[1], 1.0)
    summary = np.asarray(
        [
            float(coef[0]),
            float(coef[1] / y_scale),
            float(residual_scale / y_scale),
            float(count / max(len(type_gr), 1)),
            1.0,
        ],
        dtype=np.float32,
    )
    return summary, count


def estimate_initial_rate(item: WellInput, tvt_input: np.ndarray, tail_n: int = 30) -> float:
    end = prefix_end_index(tvt_input)
    start = max(0, end - tail_n + 1)
    tvt = tvt_input[start : end + 1]
    z = item.z[start : end + 1]
    md = item.md[start : end + 1]
    dmd = np.diff(md)
    rate = (np.diff(tvt) + np.diff(z)) / np.where(np.abs(dmd) > 1e-9, dmd, np.nan)
    finite = np.isfinite(rate) & (dmd > 0)
    return float(np.median(rate[finite])) if int(finite.sum()) >= 3 else 0.0


def build_state_spec(
    item: WellInput, tvt_input: np.ndarray, config: Mapping[str, Any]
) -> StateSpec:
    state = get_nested(config, "model.state_space", {}) or {}
    step = float(state["step"])
    band_pad = float(state["band_pad"])
    prefix_end = prefix_end_index(tvt_input)
    last_tvt = float(tvt_input[prefix_end])
    grid_min = max(float(np.nanmin(item.typewell_tvt)) - 40.0, last_tvt - band_pad)
    grid_max = min(float(np.nanmax(item.typewell_tvt)) + 40.0, last_tvt + band_pad)
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    if len(grid) < 8:
        raise ValueError(f"{item.well}: state grid is too short")
    init_rate = estimate_initial_rate(item, tvt_input)
    rate_span = float(state["rate_span"])
    span = max(rate_span, abs(init_rate) + 0.04)
    rates = np.linspace(-span, span, int(state["n_rates"]), dtype=np.float64)
    suffix_index = np.arange(prefix_end + 1, len(item.md), dtype=np.int64)
    md = item.md[suffix_index]
    z = item.z[suffix_index]
    dm = np.maximum(np.diff(np.concatenate([[item.md[prefix_end]], md])), 1.0)
    dz = np.diff(np.concatenate([[item.z[prefix_end]], z]))
    if not np.isfinite(dm).all() or not np.isfinite(dz).all():
        raise ValueError(f"{item.well}: non-finite transition trajectory")
    return StateSpec(
        grid=grid,
        rates=rates,
        suffix_index=suffix_index,
        dm=dm.astype(np.float64),
        dz=dz.astype(np.float64),
        start_p=float((last_tvt - grid_min) / step),
        init_rate=init_rate,
        prefix_end=prefix_end,
        last_known_tvt=last_tvt,
    )


def stable_circular_shuffle(values: np.ndarray, well: str, seed: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values).copy()
    if len(values) < 2:
        raise ValueError("circular-shuffle control requires at least two Type Well rows")
    roll = 1 + stable_uint64(EXPERIMENT_NAME, "typewell-shuffle", well, seed) % (len(values) - 1)
    return np.roll(values, int(roll)), int(roll)


def prepare_view(
    item: WellInput,
    tvt_input: np.ndarray,
    config: Mapping[str, Any],
    *,
    view_name: str,
    typewell_control: str = "real",
) -> PreparedView:
    architecture = get_nested(config, "model.architecture", {}) or {}
    clip_abs = float(architecture.get("derivative_clip_abs", 8.0))
    state = build_state_spec(item, tvt_input, config)
    horizontal_z, horizontal_missing, _, _ = robust_normalize(item.gr)
    horizontal_derivative = safe_derivative(horizontal_z, item.md, clip_abs)
    horizontal_channels = np.stack(
        [horizontal_z, horizontal_missing, horizontal_derivative], axis=0
    ).astype(np.float32)

    typewell_raw = item.typewell_gr.copy()
    if typewell_control == "shuffle":
        seed = int(get_nested(config, "reproducibility.seed", 42))
        typewell_raw, _ = stable_circular_shuffle(typewell_raw, item.well, seed)
    elif typewell_control != "real":
        raise ValueError(f"unknown Type Well control: {typewell_control}")
    typewell_z, typewell_missing_raw, _, _ = robust_normalize(typewell_raw)
    typewell_grid, inside = interpolate_no_extrapolation(
        state.grid, item.typewell_tvt, typewell_z
    )
    missing_grid_value, missing_inside = interpolate_no_extrapolation(
        state.grid, item.typewell_tvt, typewell_missing_raw
    )
    typewell_missing = (~inside) | (~missing_inside) | (missing_grid_value > 0.0)
    typewell_grid[typewell_missing] = 0.0
    typewell_derivative = safe_derivative(typewell_grid, state.grid, clip_abs)
    typewell_derivative[typewell_missing] = 0.0
    typewell_channels = np.stack(
        [typewell_grid, typewell_missing.astype(np.float32), typewell_derivative], axis=0
    ).astype(np.float32)

    visible = np.flatnonzero(np.isfinite(tvt_input))
    matched_raw, matched_inside = interpolate_no_extrapolation(
        tvt_input[visible], item.typewell_tvt, typewell_raw
    )
    horizontal_visible = item.gr[visible]
    pair_valid = matched_inside & np.isfinite(horizontal_visible) & np.isfinite(matched_raw)
    horizontal_pair_z, _, h_center, h_scale = robust_normalize(horizontal_visible)
    type_pair_z = ((matched_raw - h_center) / h_scale).astype(np.float32)
    pair_features = np.stack(
        [
            horizontal_pair_z,
            type_pair_z,
            horizontal_pair_z - type_pair_z,
            pair_valid.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    pair_features = pair_features[pair_valid]
    huber_summary, pair_count = huber_affine_summary(
        matched_raw, horizontal_visible, minimum_pairs=32
    )
    if pair_count < 32:
        pair_features = np.empty((0, 4), dtype=np.float32)
        huber_summary = np.zeros(5, dtype=np.float32)
    return PreparedView(
        well=item.well,
        view_name=view_name,
        tvt_input=np.asarray(tvt_input, dtype=np.float64),
        horizontal_channels=horizontal_channels,
        typewell_channels=typewell_channels,
        prefix_pairs=pair_features,
        prefix_huber_summary=huber_summary,
        prefix_pair_count=pair_count,
        state=state,
    )


def nearest_grid_indices(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(grid, values, side="left")
    left = np.clip(positions - 1, 0, len(grid) - 1)
    right = np.clip(positions, 0, len(grid) - 1)
    choose_right = np.abs(grid[right] - values) < np.abs(grid[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def build_teacher_boundary_manifest(
    schedule: pd.DataFrame,
    train_dir: Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    active = schedule.loc[schedule["active"]].copy()
    rows: list[dict[str, Any]] = []
    for well, group in active.groupby("well", sort=True):
        item = load_well_input(str(well), train_dir)
        full_state = build_state_spec(item, item.tvt_input, config)
        well_truth: WellTruth | None = None
        for row in group.sort_values(["epoch", "slot"], kind="mergesort").itertuples(
            index=False
        ):
            start = int(row.start_row)
            if start == int(row.official_suffix_start):
                boundary_source = "official_prefix"
                boundary_row = int(full_state.prefix_end)
                boundary_tvt = float(full_state.last_known_tvt)
                boundary_rate = float(full_state.init_rate)
            else:
                if start < 2:
                    raise ValueError(f"{well}: interior teacher boundary needs two prior rows")
                if well_truth is None:
                    well_truth = load_well_truth(str(well), train_dir)
                boundary_source = "interior_teacher_loss_only"
                boundary_row = start - 1
                previous_row = start - 2
                boundary_tvt = float(well_truth.tvt[boundary_row])
                dm = max(float(item.md[boundary_row] - item.md[previous_row]), 1.0)
                dz = float(item.z[boundary_row] - item.z[previous_row])
                boundary_rate = float(
                    (well_truth.tvt[boundary_row] - well_truth.tvt[previous_row] + dz) / dm
                )
            if not math.isfinite(boundary_tvt) or not math.isfinite(boundary_rate):
                raise ValueError(f"{well}: non-finite teacher boundary")
            rate_index = int(np.argmin(np.abs(full_state.rates - boundary_rate)))
            prior_rate = (
                float(boundary_rate)
                if boundary_source == "official_prefix"
                else float(full_state.rates[rate_index])
            )
            rows.append(
                {
                    "well": str(well),
                    "epoch": int(row.epoch),
                    "slot": int(row.slot),
                    "start_row": start,
                    "stop_row": int(row.stop_row),
                    "scored_rows": int(row.scored_rows),
                    "boundary_source": boundary_source,
                    "boundary_row": boundary_row,
                    "boundary_tvt": boundary_tvt,
                    "boundary_rate_raw": boundary_rate,
                    "boundary_rate": prior_rate,
                    "boundary_rate_index": rate_index,
                    "encoder_tvt_input_source": "official_prefix_only",
                }
            )
    boundary = pd.DataFrame(rows).sort_values(
        ["well", "epoch", "slot"], kind="mergesort"
    ).reset_index(drop=True)
    expected = active[["well", "epoch", "slot"]].sort_values(
        ["well", "epoch", "slot"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(boundary[["well", "epoch", "slot"]], expected)
    return boundary


def window_keys_from_manifests(
    schedule: pd.DataFrame,
    boundary: pd.DataFrame,
    *,
    role: str,
    epoch: int,
) -> list[WindowKey]:
    selected = schedule.loc[
        schedule["active"]
        & (schedule["role"] == role)
        & (schedule["epoch"] == int(epoch))
    ].merge(
        boundary,
        on=["well", "epoch", "slot", "start_row", "stop_row", "scored_rows"],
        how="left",
        validate="one_to_one",
    )
    if selected["boundary_source"].isna().any():
        raise ValueError("active window is missing its frozen teacher boundary")
    return [
        WindowKey(
            well=str(row.well),
            epoch=int(row.epoch),
            slot=int(row.slot),
            start_row=int(row.start_row),
            stop_row=int(row.stop_row),
            scored_rows=int(row.scored_rows),
            boundary_source=str(row.boundary_source),
            boundary_tvt=float(row.boundary_tvt),
            boundary_rate=float(row.boundary_rate),
            boundary_rate_index=int(row.boundary_rate_index),
        )
        for row in selected.itertuples(index=False)
    ]


def build_window_state_spec(
    item: WellInput,
    full_state: StateSpec,
    key: WindowKey,
    config: Mapping[str, Any],
) -> StateSpec:
    absolute = np.arange(key.start_row, key.stop_row, dtype=np.int64)
    if len(absolute) != key.scored_rows or not len(absolute):
        raise ValueError(f"{item.well}: invalid active window extent")
    offset = key.start_row - int(full_state.suffix_index[0])
    stop = offset + key.scored_rows
    if offset < 0 or stop > len(full_state.suffix_index):
        raise ValueError(f"{item.well}: window is outside the official hidden suffix")
    if not np.array_equal(full_state.suffix_index[offset:stop], absolute):
        raise ValueError(f"{item.well}: window row identity mismatch")
    step = float(get_nested(config, "model.state_space.step"))
    start_p = float((key.boundary_tvt - full_state.grid[0]) / step)
    return StateSpec(
        grid=full_state.grid,
        rates=full_state.rates,
        suffix_index=absolute,
        dm=full_state.dm[offset:stop].copy(),
        dz=full_state.dz[offset:stop].copy(),
        start_p=start_p,
        init_rate=float(key.boundary_rate),
        prefix_end=key.start_row - 1,
        last_known_tvt=float(key.boundary_tvt),
    )


# %% [markdown]
# ## 6. Prefix-conditioned multi-scale neural emission

# %%
if TORCH_AVAILABLE:

    class FiLMResidualBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            context_dim: int,
            dilation: int,
            kernel_size: int,
            groups: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = dilation * (kernel_size // 2)
            self.conv1 = nn.Conv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation
            )
            self.conv2 = nn.Conv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation
            )
            self.norm1 = nn.GroupNorm(groups, channels)
            self.norm2 = nn.GroupNorm(groups, channels)
            self.film1 = nn.Linear(context_dim, 2 * channels)
            self.film2 = nn.Linear(context_dim, 2 * channels)
            self.dropout = nn.Dropout(dropout)
            nn.init.zeros_(self.film1.weight)
            nn.init.zeros_(self.film1.bias)
            nn.init.zeros_(self.film2.weight)
            nn.init.zeros_(self.film2.bias)

        @staticmethod
        def apply_film(value: Any, parameters: Any) -> Any:
            gamma, beta = parameters.chunk(2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            return value * (1.0 + gamma) + beta

        def forward(self, value: Any, context: Any) -> Any:
            residual = value
            value = self.apply_film(self.norm1(self.conv1(value)), self.film1(context))
            value = F.gelu(value)
            value = self.dropout(value)
            value = self.apply_film(self.norm2(self.conv2(value)), self.film2(context))
            return F.gelu(residual + self.dropout(value))


    class MultiScaleEncoder(nn.Module):
        def __init__(self, input_dim: int, config: Mapping[str, Any]) -> None:
            super().__init__()
            architecture = get_nested(config, "model.architecture", {}) or {}
            channels = int(architecture["embedding_dim"])
            context_dim = int(architecture["prefix_context_dim"])
            kernel = int(architecture["convolution_kernel_size"])
            groups = int(architecture["group_norm_groups"])
            dropout = float(architecture.get("dropout", 0.10))
            self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    FiLMResidualBlock(
                        channels,
                        context_dim,
                        int(dilation),
                        kernel,
                        groups,
                        dropout,
                    )
                    for dilation in architecture["dilations"]
                ]
            )

        def forward(self, value: Any, context: Any) -> Any:
            value = self.input_projection(value)
            for block in self.blocks:
                value = block(value, context)
            return value.transpose(1, 2)


    class PrefixContextEncoder(nn.Module):
        def __init__(self, config: Mapping[str, Any]) -> None:
            super().__init__()
            architecture = get_nested(config, "model.architecture", {}) or {}
            pair_dim = int(architecture["prefix_pair_embedding_dim"])
            context_dim = int(architecture["prefix_context_dim"])
            self.pair_projection = nn.Sequential(
                nn.Linear(4, pair_dim), nn.GELU(), nn.Linear(pair_dim, pair_dim), nn.GELU()
            )
            self.attention = nn.Linear(pair_dim, 1)
            self.context_projection = nn.Sequential(
                nn.Linear(pair_dim + 5, 64), nn.GELU(), nn.Linear(64, context_dim)
            )
            self.context_dim = context_dim

        def forward(self, pairs: Any, huber_summary: Any) -> Any:
            if pairs.shape[0] < 32:
                return torch.zeros(
                    (1, self.context_dim), dtype=huber_summary.dtype, device=huber_summary.device
                )
            embedded = self.pair_projection(pairs)
            weight = torch.softmax(self.attention(embedded).squeeze(-1), dim=0)
            pooled = torch.sum(weight[:, None] * embedded, dim=0, keepdim=True)
            return self.context_projection(torch.cat([pooled, huber_summary[None, :]], dim=1))


    class PrefixConditionedUnary(nn.Module):
        def __init__(self, config: Mapping[str, Any]) -> None:
            super().__init__()
            architecture = get_nested(config, "model.architecture", {}) or {}
            embedding = int(architecture["embedding_dim"])
            context_dim = int(architecture["prefix_context_dim"])
            self.context_encoder = PrefixContextEncoder(config)
            self.horizontal_encoder = MultiScaleEncoder(3, config)
            self.typewell_encoder = MultiScaleEncoder(3, config)
            self.horizontal_projection = nn.Linear(embedding, embedding, bias=False)
            self.typewell_projection = nn.Linear(embedding, embedding, bias=False)
            self.temperature_head = nn.Linear(context_dim, 1)
            initial = float(architecture["temperature_initial"])
            nn.init.zeros_(self.temperature_head.weight)
            nn.init.constant_(self.temperature_head.bias, math.log(initial))
            self.temperature_min = float(architecture["temperature_min"])
            self.temperature_max = float(architecture["temperature_max"])
            self.row_chunk_size = int(architecture["unary_row_chunk_size"])

        def forward(
            self,
            horizontal_channels: Any,
            typewell_channels: Any,
            prefix_pairs: Any,
            huber_summary: Any,
            row_indices: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            context = self.context_encoder(prefix_pairs, huber_summary)
            horizontal = self.horizontal_encoder(horizontal_channels[None, ...], context)[0]
            typewell = self.typewell_encoder(typewell_channels[None, ...], context)[0]
            horizontal = F.normalize(self.horizontal_projection(horizontal), dim=-1, eps=1e-6)
            typewell = F.normalize(self.typewell_projection(typewell), dim=-1, eps=1e-6)
            if row_indices is not None:
                horizontal = horizontal[row_indices]
            temperature = torch.exp(self.temperature_head(context)).clamp(
                self.temperature_min, self.temperature_max
            )
            chunks = []
            for start in range(0, horizontal.shape[0], self.row_chunk_size):
                stop = min(horizontal.shape[0], start + self.row_chunk_size)
                chunks.append(horizontal[start:stop] @ typewell.T / temperature)
            return torch.cat(chunks, dim=0), context, temperature.squeeze()


else:

    class PrefixConditionedUnary:  # type: ignore[no-redef]
        def __init__(self, _: Mapping[str, Any]) -> None:
            raise RuntimeError("PyTorch is required to instantiate the exp347 neural unary")


def prepared_to_torch(view: PreparedView, device: Any) -> tuple[Any, Any, Any, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")
    horizontal = torch.as_tensor(view.horizontal_channels, dtype=torch.float32, device=device)
    typewell = torch.as_tensor(view.typewell_channels, dtype=torch.float32, device=device)
    pairs = torch.as_tensor(view.prefix_pairs, dtype=torch.float32, device=device)
    summary = torch.as_tensor(view.prefix_huber_summary, dtype=torch.float32, device=device)
    return horizontal, typewell, pairs, summary


def model_unary(
    model: Any,
    view: PreparedView,
    device: Any,
    row_indices: np.ndarray | None = None,
) -> tuple[Any, float]:
    horizontal, typewell, pairs, summary = prepared_to_torch(view, device)
    selected = view.state.suffix_index if row_indices is None else np.asarray(row_indices)
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
    unary_selected, _, temperature = model(
        horizontal, typewell, pairs, summary, selected_tensor
    )
    return unary_selected.float(), float(temperature.detach().cpu())


# %% [markdown]
# ## 7. Scalar and batched exp209 exact forward-backward helpers

# %%
if TORCH_AVAILABLE:
    NEGATIVE_LOG_ZERO = -1.0e18


    def initial_log_prior(spec: StateSpec, config: Mapping[str, Any], device: Any) -> Any:
        state = get_nested(config, "model.state_space", {}) or {}
        position = torch.arange(len(spec.grid), dtype=torch.float32, device=device)
        rates = torch.as_tensor(spec.rates, dtype=torch.float32, device=device)
        position_log = -0.5 * (
            (position - float(spec.start_p)) * float(state["step"]) / float(state["start_sig"])
        ) ** 2
        position_log = torch.where(
            position_log >= -60.0,
            position_log,
            torch.full_like(position_log, NEGATIVE_LOG_ZERO),
        )
        rate_log = -0.5 * ((rates - float(spec.init_rate)) / float(state["r0_sig"])) ** 2
        return position_log[:, None] + rate_log[None, :]


    def rate_transition_tables(
        dm: float, rates: Any, config: Mapping[str, Any]
    ) -> tuple[Any, Any, Any, Any, Any]:
        state = get_nested(config, "model.state_space", {}) or {}
        rate_step = rates[1] - rates[0]
        sigma_step = float(state["sig_r"]) * math.sqrt(float(dm))
        variance_cells = (sigma_step / rate_step) ** 2
        mean_move = -(1.0 - float(state["mom"])) * rates * float(dm) / rate_step
        p_plus = torch.clamp(0.5 * (variance_cells + mean_move), min=1e-12)
        p_minus = torch.clamp(0.5 * (variance_cells - mean_move), min=1e-12)
        total = p_plus + p_minus
        factor = torch.where(total > 0.9, 0.9 / total, torch.ones_like(total))
        p_plus = p_plus * factor
        p_minus = p_minus * factor
        kernel = torch.stack(
            [torch.log(p_minus), torch.log1p(-p_plus - p_minus), torch.log(p_plus)], dim=1
        )
        rate_count = len(rates)
        destination = torch.arange(rate_count, device=rates.device)[:, None]
        source_offsets = torch.tensor([-1, 0, 1], device=rates.device)[None, :]
        source = destination + source_offsets
        source_valid = (source >= 0) & (source < rate_count)
        source_clamped = source.clamp(0, rate_count - 1)
        delta_column = (-source_offsets + 1).expand_as(source)
        forward_log = kernel[source_clamped, delta_column]
        forward_log = torch.where(
            source_valid, forward_log, torch.full_like(forward_log, NEGATIVE_LOG_ZERO)
        )
        source_rate = torch.arange(rate_count, device=rates.device)[:, None]
        delta = torch.tensor([-1, 0, 1], device=rates.device)[None, :]
        backward_destination = source_rate + delta
        backward_valid = (backward_destination >= 0) & (backward_destination < rate_count)
        backward_destination = backward_destination.clamp(0, rate_count - 1)
        backward_log = torch.where(
            backward_valid,
            kernel[:, :],
            torch.full_like(kernel, NEGATIVE_LOG_ZERO),
        )
        return source_clamped, forward_log, backward_destination, backward_log, kernel


    def position_transition_tables(
        dm: float, dz: float, rates: Any, position_count: int, config: Mapping[str, Any]
    ) -> tuple[Any, Any, Any, Any, Any]:
        state = get_nested(config, "model.state_space", {}) or {}
        step = float(state["step"])
        sigma_position = max(float(state["sig_p"]), 0.35 * step)
        mu = rates * float(dm) - float(dz)
        base = torch.floor(mu / step + 0.5).to(torch.long)
        offsets = torch.arange(-2, 3, device=rates.device, dtype=torch.long)
        shifts = base[:, None] + offsets[None, :]
        delta = shifts.to(torch.float32) * step - mu[:, None]
        position_log = -0.5 * (delta / sigma_position) ** 2
        position_log = position_log - torch.logsumexp(position_log, dim=1, keepdim=True)
        p2 = torch.arange(position_count, device=rates.device)[:, None, None]
        predecessor = p2 - shifts[None, :, :]
        predecessor_valid = (predecessor >= 0) & (predecessor < position_count)
        predecessor = predecessor.clamp(0, position_count - 1)
        p1 = torch.arange(position_count, device=rates.device)[:, None, None]
        successor = p1 + shifts[None, :, :]
        successor_valid = (successor >= 0) & (successor < position_count)
        successor = successor.clamp(0, position_count - 1)
        return predecessor, predecessor_valid, successor, successor_valid, position_log


    def forward_transition(
        previous: Any,
        emission: Any,
        dm: float,
        dz: float,
        rates: Any,
        config: Mapping[str, Any],
    ) -> Any:
        source, rate_log, _, _, _ = rate_transition_tables(dm, rates, config)
        rate_values = previous[:, source] + rate_log[None, :, :]
        after_rate = torch.logsumexp(rate_values, dim=2)
        predecessor, valid, _, _, position_log = position_transition_tables(
            dm, dz, rates, previous.shape[0], config
        )
        gathered = torch.gather(
            after_rate.T.unsqueeze(-1).expand(-1, -1, 5),
            1,
            predecessor.permute(1, 0, 2),
        )
        gathered = torch.where(
            valid.permute(1, 0, 2),
            gathered,
            torch.full_like(gathered, NEGATIVE_LOG_ZERO),
        )
        after_position = torch.logsumexp(gathered + position_log[:, None, :], dim=2).T
        return after_position + emission[:, None]


    def backward_transition(
        beta_next: Any,
        emission_next: Any,
        dm: float,
        dz: float,
        rates: Any,
        config: Mapping[str, Any],
    ) -> Any:
        _, _, successor, valid, position_log = position_transition_tables(
            dm, dz, rates, beta_next.shape[0], config
        )
        future = beta_next + emission_next[:, None]
        gathered = torch.gather(
            future.T.unsqueeze(-1).expand(-1, -1, 5),
            1,
            successor.permute(1, 0, 2),
        )
        gathered = torch.where(
            valid.permute(1, 0, 2),
            gathered,
            torch.full_like(gathered, NEGATIVE_LOG_ZERO),
        )
        after_position = torch.logsumexp(gathered + position_log[:, None, :], dim=2).T
        _, _, destination, rate_log, _ = rate_transition_tables(dm, rates, config)
        rate_values = after_position[:, destination] + rate_log[None, :, :]
        return torch.logsumexp(rate_values, dim=2)


    def exact_forward_backward(
        unary: Any, spec: StateSpec, config: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        rates = torch.as_tensor(spec.rates, dtype=torch.float32, device=unary.device)
        previous = initial_log_prior(spec, config, unary.device)
        alpha_rows = []
        for index in range(len(spec.suffix_index)):
            previous = forward_transition(
                previous,
                unary[index],
                float(spec.dm[index]),
                float(spec.dz[index]),
                rates,
                config,
            )
            alpha_rows.append(previous)
        alpha = torch.stack(alpha_rows, dim=0)
        log_partition = torch.logsumexp(alpha[-1].reshape(-1), dim=0)
        posterior_rows: list[Any] = [None] * len(spec.suffix_index)
        beta = torch.zeros_like(alpha[-1])
        joint = alpha[-1] + beta
        posterior_rows[-1] = torch.exp(
            torch.logsumexp(joint, dim=1) - torch.logsumexp(joint.reshape(-1), dim=0)
        )
        for index in range(len(spec.suffix_index) - 1, 0, -1):
            beta = backward_transition(
                beta,
                unary[index],
                float(spec.dm[index]),
                float(spec.dz[index]),
                rates,
                config,
            )
            joint = alpha[index - 1] + beta
            posterior_rows[index - 1] = torch.exp(
                torch.logsumexp(joint, dim=1) - torch.logsumexp(joint.reshape(-1), dim=0)
            )
        posterior = torch.stack(posterior_rows, dim=0)
        return posterior, log_partition


    def gaussian_label_log_emission(
        target: Any, spec: StateSpec, config: Mapping[str, Any]
    ) -> Any:
        objective = get_nested(config, "model.training.objective", {}) or {}
        sigma = float(objective["label_observation_sigma_ft"])
        if sigma != 0.35:
            raise ValueError("exp347 label_observation_sigma_ft must remain 0.35")
        grid = torch.as_tensor(spec.grid, dtype=target.dtype, device=target.device)
        residual = (grid[None, :] - target[:, None]) / sigma
        return -0.5 * residual.square()


    def soft_label_structured_terms(
        unary: Any, target: Any, spec: StateSpec, config: Mapping[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        label_log_emission = gaussian_label_log_emission(target, spec, config)
        posterior, log_partition = exact_forward_backward(unary, spec, config)
        conditioned_posterior, conditioned_log_partition = exact_forward_backward(
            unary + label_log_emission, spec, config
        )
        return (
            posterior,
            conditioned_posterior,
            log_partition,
            conditioned_log_partition,
        )


    class SoftLabelStructuredNLL(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, unary: Any, target: Any, spec: StateSpec, config: Any) -> Any:
            with torch.no_grad():
                (
                    posterior,
                    conditioned_posterior,
                    log_partition,
                    conditioned_log_partition,
                ) = soft_label_structured_terms(unary, target, spec, config)
                token_count = max(1, len(spec.suffix_index))
                value = (log_partition - conditioned_log_partition) / token_count
            ctx.save_for_backward(posterior, conditioned_posterior)
            ctx.token_count = token_count
            return value

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, None, None, None]:
            posterior, conditioned_posterior = ctx.saved_tensors
            grad_unary = posterior - conditioned_posterior
            grad_unary = grad_unary * gradient / ctx.token_count
            return grad_unary, None, None, None


    def pad_unary_batch(
        unaries: Sequence[Any | None], batch: BatchedStateSpec
    ) -> Any:
        if len(unaries) > len(batch.specs):
            raise ValueError("unary count exceeds padded state batch size")
        values = list(unaries) + [None] * (len(batch.specs) - len(unaries))
        reference = next((value for value in values if value is not None), None)
        if reference is None:
            raise ValueError("unary batch must contain at least one active tensor")
        padded_rows = batch.row_mask.shape[1]
        padded_positions = batch.position_mask.shape[1]
        padded = []
        for index, (value, spec) in enumerate(zip(values, batch.specs, strict=True)):
            if spec is None:
                padded.append(
                    torch.full(
                        (padded_rows, padded_positions),
                        NEGATIVE_LOG_ZERO,
                        dtype=reference.dtype,
                        device=reference.device,
                    )
                )
                continue
            if value is None or tuple(value.shape) != (
                len(spec.suffix_index),
                len(spec.grid),
            ):
                raise ValueError(f"unary/state shape mismatch at batch index {index}")
            padded.append(
                F.pad(
                    value,
                    (
                        0,
                        padded_positions - value.shape[1],
                        0,
                        padded_rows - value.shape[0],
                    ),
                    value=NEGATIVE_LOG_ZERO,
                )
            )
        result = torch.stack(padded, dim=0)
        row_mask = torch.as_tensor(batch.row_mask, dtype=torch.bool, device=result.device)
        position_mask = torch.as_tensor(
            batch.position_mask, dtype=torch.bool, device=result.device
        )
        valid = row_mask[:, :, None] & position_mask[:, None, :]
        return torch.where(valid, result, torch.full_like(result, NEGATIVE_LOG_ZERO))


    def pad_target_batch(
        targets: Sequence[np.ndarray | Any | None],
        batch: BatchedStateSpec,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        values = list(targets) + [None] * (len(batch.specs) - len(targets))
        if len(values) != len(batch.specs):
            raise ValueError("target count exceeds padded state batch size")
        output = torch.zeros(batch.row_mask.shape, dtype=dtype, device=device)
        for index, (value, spec) in enumerate(zip(values, batch.specs, strict=True)):
            if spec is None:
                continue
            if value is None:
                raise ValueError(f"active target missing at batch index {index}")
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
            if tensor.ndim != 1 or len(tensor) != len(spec.suffix_index):
                raise ValueError(f"target/state shape mismatch at batch index {index}")
            output[index, : len(tensor)] = tensor
        return output


    def batched_initial_log_prior(
        batch: BatchedStateSpec, config: Mapping[str, Any], device: Any
    ) -> Any:
        state = get_nested(config, "model.state_space", {}) or {}
        position_count = batch.position_mask.shape[1]
        position = torch.arange(position_count, dtype=torch.float32, device=device)
        start_p = torch.as_tensor(batch.start_p, dtype=torch.float32, device=device)
        rates = torch.as_tensor(batch.rates, dtype=torch.float32, device=device)
        init_rate = torch.as_tensor(batch.init_rate, dtype=torch.float32, device=device)
        position_log = -0.5 * (
            (position[None, :] - start_p[:, None])
            * float(state["step"])
            / float(state["start_sig"])
        ) ** 2
        position_log = torch.where(
            position_log >= -60.0,
            position_log,
            torch.full_like(position_log, NEGATIVE_LOG_ZERO),
        )
        rate_log = -0.5 * (
            (rates - init_rate[:, None]) / float(state["r0_sig"])
        ) ** 2
        result = position_log[:, :, None] + rate_log[:, None, :]
        position_mask = torch.as_tensor(
            batch.position_mask, dtype=torch.bool, device=device
        )
        rate_mask = torch.as_tensor(batch.rate_mask, dtype=torch.bool, device=device)
        valid = position_mask[:, :, None] & rate_mask[:, None, :]
        return torch.where(valid, result, torch.full_like(result, NEGATIVE_LOG_ZERO))


    def batched_rate_transition_tables(
        dm: Any, rates: Any, rate_mask: Any, config: Mapping[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        state = get_nested(config, "model.state_space", {}) or {}
        batch_size, rate_count = rates.shape
        active = rate_mask.sum(dim=1) >= 2
        rate_step = torch.where(active, rates[:, 1] - rates[:, 0], torch.ones_like(dm))
        sigma_step = float(state["sig_r"]) * torch.sqrt(dm)
        variance_cells = (sigma_step / rate_step) ** 2
        mean_move = (
            -(1.0 - float(state["mom"]))
            * rates
            * dm[:, None]
            / rate_step[:, None]
        )
        p_plus = torch.clamp(0.5 * (variance_cells[:, None] + mean_move), min=1e-12)
        p_minus = torch.clamp(0.5 * (variance_cells[:, None] - mean_move), min=1e-12)
        total = p_plus + p_minus
        factor = torch.where(total > 0.9, 0.9 / total, torch.ones_like(total))
        p_plus = p_plus * factor
        p_minus = p_minus * factor
        kernel = torch.stack(
            [torch.log(p_minus), torch.log1p(-p_plus - p_minus), torch.log(p_plus)],
            dim=2,
        )
        offsets = torch.tensor([-1, 0, 1], device=rates.device)
        destination = torch.arange(rate_count, device=rates.device)[:, None]
        source = destination + offsets[None, :]
        source_valid = (source >= 0) & (source < rate_count)
        source_clamped = source.clamp(0, rate_count - 1)
        delta_column = (-offsets + 1)[None, :].expand(rate_count, -1)
        forward_log = kernel[:, source_clamped, delta_column]
        gathered_source_mask = torch.gather(
            rate_mask,
            1,
            source_clamped.reshape(1, -1).expand(batch_size, -1),
        ).reshape(batch_size, rate_count, 3)
        forward_valid = (
            source_valid[None, :, :]
            & rate_mask[:, :, None]
            & gathered_source_mask
        )
        forward_log = torch.where(
            forward_valid,
            forward_log,
            torch.full_like(forward_log, NEGATIVE_LOG_ZERO),
        )
        source_rate = torch.arange(rate_count, device=rates.device)[:, None]
        backward_destination = source_rate + offsets[None, :]
        backward_valid = (backward_destination >= 0) & (
            backward_destination < rate_count
        )
        backward_destination = backward_destination.clamp(0, rate_count - 1)
        gathered_destination_mask = torch.gather(
            rate_mask,
            1,
            backward_destination.reshape(1, -1).expand(batch_size, -1),
        ).reshape(batch_size, rate_count, 3)
        backward_valid_batched = (
            backward_valid[None, :, :]
            & rate_mask[:, :, None]
            & gathered_destination_mask
        )
        backward_log = torch.where(
            backward_valid_batched,
            kernel,
            torch.full_like(kernel, NEGATIVE_LOG_ZERO),
        )
        return source_clamped, forward_log, backward_destination, backward_log


    def batched_position_transition_tables(
        dm: Any,
        dz: Any,
        rates: Any,
        position_mask: Any,
        rate_mask: Any,
        config: Mapping[str, Any],
    ) -> tuple[Any, Any, Any, Any, Any]:
        state = get_nested(config, "model.state_space", {}) or {}
        step = float(state["step"])
        sigma_position = max(float(state["sig_p"]), 0.35 * step)
        batch_size, rate_count = rates.shape
        position_count = position_mask.shape[1]
        mu = rates * dm[:, None] - dz[:, None]
        base = torch.floor(mu / step + 0.5).to(torch.long)
        offsets = torch.arange(-2, 3, device=rates.device, dtype=torch.long)
        shifts = base[:, :, None] + offsets[None, None, :]
        delta = shifts.to(torch.float32) * step - mu[:, :, None]
        position_log = -0.5 * (delta / sigma_position) ** 2
        position_log = position_log - torch.logsumexp(
            position_log, dim=2, keepdim=True
        )
        p2 = torch.arange(position_count, device=rates.device)[None, None, :, None]
        predecessor = p2 - shifts[:, :, None, :]
        predecessor_valid = (predecessor >= 0) & (predecessor < position_count)
        predecessor = predecessor.clamp(0, position_count - 1)
        predecessor_position_mask = torch.gather(
            position_mask,
            1,
            predecessor.reshape(batch_size, -1),
        ).reshape(batch_size, rate_count, position_count, 5)
        destination_position_mask = position_mask[:, None, :, None]
        predecessor_valid = (
            predecessor_valid
            & predecessor_position_mask
            & destination_position_mask
            & rate_mask[:, :, None, None]
        )
        p1 = torch.arange(position_count, device=rates.device)[None, None, :, None]
        successor = p1 + shifts[:, :, None, :]
        successor_valid = (successor >= 0) & (successor < position_count)
        successor = successor.clamp(0, position_count - 1)
        successor_position_mask = torch.gather(
            position_mask,
            1,
            successor.reshape(batch_size, -1),
        ).reshape(batch_size, rate_count, position_count, 5)
        source_position_mask = position_mask[:, None, :, None]
        successor_valid = (
            successor_valid
            & successor_position_mask
            & source_position_mask
            & rate_mask[:, :, None, None]
        )
        position_log = torch.where(
            rate_mask[:, :, None],
            position_log,
            torch.full_like(position_log, NEGATIVE_LOG_ZERO),
        )
        return (
            predecessor,
            predecessor_valid,
            successor,
            successor_valid,
            position_log,
        )


    def batched_forward_transition(
        previous: Any,
        emission: Any,
        dm: Any,
        dz: Any,
        rates: Any,
        position_mask: Any,
        rate_mask: Any,
        config: Mapping[str, Any],
    ) -> Any:
        batch_size, position_count, rate_count = previous.shape
        source, rate_log, _, _ = batched_rate_transition_tables(
            dm, rates, rate_mask, config
        )
        source_index = source[None, None, :, :].expand(
            batch_size, position_count, -1, -1
        )
        rate_values = torch.gather(
            previous[:, :, :, None].expand(-1, -1, -1, 3), 2, source_index
        ) + rate_log[:, None, :, :]
        after_rate = torch.logsumexp(rate_values, dim=3)
        predecessor, valid, _, _, position_log = batched_position_transition_tables(
            dm, dz, rates, position_mask, rate_mask, config
        )
        gathered = torch.gather(
            after_rate.permute(0, 2, 1)[:, :, :, None].expand(-1, -1, -1, 5),
            2,
            predecessor,
        )
        gathered = torch.where(
            valid, gathered, torch.full_like(gathered, NEGATIVE_LOG_ZERO)
        )
        after_position = torch.logsumexp(
            gathered + position_log[:, :, None, :], dim=3
        ).permute(0, 2, 1)
        result = after_position + emission[:, :, None]
        valid_state = position_mask[:, :, None] & rate_mask[:, None, :]
        return torch.where(
            valid_state, result, torch.full_like(result, NEGATIVE_LOG_ZERO)
        )


    def batched_backward_transition(
        beta_next: Any,
        emission_next: Any,
        dm: Any,
        dz: Any,
        rates: Any,
        position_mask: Any,
        rate_mask: Any,
        config: Mapping[str, Any],
    ) -> Any:
        batch_size, position_count, rate_count = beta_next.shape
        _, _, successor, valid, position_log = batched_position_transition_tables(
            dm, dz, rates, position_mask, rate_mask, config
        )
        future = beta_next + emission_next[:, :, None]
        gathered = torch.gather(
            future.permute(0, 2, 1)[:, :, :, None].expand(-1, -1, -1, 5),
            2,
            successor,
        )
        gathered = torch.where(
            valid, gathered, torch.full_like(gathered, NEGATIVE_LOG_ZERO)
        )
        after_position = torch.logsumexp(
            gathered + position_log[:, :, None, :], dim=3
        ).permute(0, 2, 1)
        _, _, destination, rate_log = batched_rate_transition_tables(
            dm, rates, rate_mask, config
        )
        destination_index = destination[None, None, :, :].expand(
            batch_size, position_count, -1, -1
        )
        rate_values = torch.gather(
            after_position[:, :, :, None].expand(-1, -1, -1, 3),
            2,
            destination_index,
        ) + rate_log[:, None, :, :]
        result = torch.logsumexp(rate_values, dim=3)
        valid_state = position_mask[:, :, None] & rate_mask[:, None, :]
        return torch.where(
            valid_state, result, torch.full_like(result, NEGATIVE_LOG_ZERO)
        )


    def batched_exact_forward_backward(
        unary: Any, batch: BatchedStateSpec, config: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        expected = (
            len(batch.specs),
            batch.row_mask.shape[1],
            batch.position_mask.shape[1],
        )
        if tuple(unary.shape) != expected:
            raise ValueError(f"batched unary shape {tuple(unary.shape)} != {expected}")
        device = unary.device
        rates = torch.as_tensor(batch.rates, dtype=torch.float32, device=device)
        dm = torch.as_tensor(batch.dm, dtype=torch.float32, device=device)
        dz = torch.as_tensor(batch.dz, dtype=torch.float32, device=device)
        row_mask = torch.as_tensor(batch.row_mask, dtype=torch.bool, device=device)
        position_mask = torch.as_tensor(
            batch.position_mask, dtype=torch.bool, device=device
        )
        rate_mask = torch.as_tensor(batch.rate_mask, dtype=torch.bool, device=device)
        previous = batched_initial_log_prior(batch, config, device)
        alpha_rows = []
        for index in range(unary.shape[1]):
            updated = batched_forward_transition(
                previous,
                unary[:, index],
                dm[:, index],
                dz[:, index],
                rates,
                position_mask,
                rate_mask,
                config,
            )
            previous = torch.where(row_mask[:, index, None, None], updated, previous)
            alpha_rows.append(previous)
        alpha = torch.stack(alpha_rows, dim=1)
        log_partition = torch.logsumexp(alpha[:, -1].flatten(1), dim=1)
        posterior = torch.zeros_like(unary)
        beta = torch.zeros_like(alpha[:, -1])
        for index in range(unary.shape[1] - 1, -1, -1):
            joint = alpha[:, index] + beta
            log_position = torch.logsumexp(joint, dim=2)
            normalizer = torch.logsumexp(joint.flatten(1), dim=1)
            row_posterior = torch.exp(log_position - normalizer[:, None])
            valid_position = row_mask[:, index, None] & position_mask
            posterior[:, index] = torch.where(
                valid_position, row_posterior, torch.zeros_like(row_posterior)
            )
            if index > 0:
                updated = batched_backward_transition(
                    beta,
                    unary[:, index],
                    dm[:, index],
                    dz[:, index],
                    rates,
                    position_mask,
                    rate_mask,
                    config,
                )
                beta = torch.where(row_mask[:, index, None, None], updated, beta)
        return posterior, log_partition


    def batched_gaussian_label_log_emission(
        target: Any, batch: BatchedStateSpec, config: Mapping[str, Any]
    ) -> Any:
        sigma = float(
            get_nested(config, "model.training.objective.label_observation_sigma_ft")
        )
        if sigma != 0.35:
            raise ValueError("exp347 label_observation_sigma_ft must remain 0.35")
        grid = torch.as_tensor(batch.grids, dtype=target.dtype, device=target.device)
        residual = (grid[:, None, :] - target[:, :, None]) / sigma
        result = -0.5 * residual.square()
        row_mask = torch.as_tensor(batch.row_mask, dtype=torch.bool, device=target.device)
        position_mask = torch.as_tensor(
            batch.position_mask, dtype=torch.bool, device=target.device
        )
        valid = row_mask[:, :, None] & position_mask[:, None, :]
        return torch.where(valid, result, torch.full_like(result, NEGATIVE_LOG_ZERO))


    def batched_soft_label_structured_terms(
        unary: Any, target: Any, batch: BatchedStateSpec, config: Mapping[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        label_log_emission = batched_gaussian_label_log_emission(target, batch, config)
        posterior, log_partition = batched_exact_forward_backward(unary, batch, config)
        conditioned_posterior, conditioned_log_partition = batched_exact_forward_backward(
            unary + label_log_emission, batch, config
        )
        return posterior, conditioned_posterior, log_partition, conditioned_log_partition


    class BatchedSoftLabelStructuredNLL(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            unary: Any,
            target: Any,
            batch: BatchedStateSpec,
            config: Any,
        ) -> Any:
            with torch.no_grad():
                posterior, conditioned, log_partition, conditioned_partition = (
                    batched_soft_label_structured_terms(unary, target, batch, config)
                )
                token_count = torch.as_tensor(
                    batch.row_mask.sum(axis=1),
                    dtype=unary.dtype,
                    device=unary.device,
                )
                active = torch.as_tensor(
                    batch.active_mask, dtype=torch.bool, device=unary.device
                )
                safe_count = torch.clamp(token_count, min=1.0)
                value = (log_partition - conditioned_partition) / safe_count
                value = torch.where(active, value, torch.zeros_like(value))
            ctx.save_for_backward(posterior, conditioned, safe_count, active)
            return value

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, None, None, None]:
            posterior, conditioned, token_count, active = ctx.saved_tensors
            grad_unary = posterior - conditioned
            grad_unary = grad_unary * gradient[:, None, None] / token_count[:, None, None]
            grad_unary = torch.where(
                active[:, None, None], grad_unary, torch.zeros_like(grad_unary)
            )
            return grad_unary, None, None, None


    def exact_viterbi(unary: Any, spec: StateSpec, config: Mapping[str, Any]) -> Any:
        rates = torch.as_tensor(spec.rates, dtype=torch.float32, device=unary.device)
        previous = initial_log_prior(spec, config, unary.device)
        pointers: list[Any] = []
        position_count, rate_count = previous.shape
        rate_grid = torch.arange(rate_count, device=unary.device)[None, :].expand(
            position_count, -1
        )
        for index in range(len(spec.suffix_index)):
            source, rate_log, _, _, _ = rate_transition_tables(
                float(spec.dm[index]), rates, config
            )
            rate_values = previous[:, source] + rate_log[None, :, :]
            after_rate, rate_choice = torch.max(rate_values, dim=2)
            rate_predecessor = source[None, :, :].expand(position_count, -1, -1).gather(
                2, rate_choice[..., None]
            )[..., 0]
            predecessor, valid, _, _, position_log = position_transition_tables(
                float(spec.dm[index]),
                float(spec.dz[index]),
                rates,
                position_count,
                config,
            )
            gathered = torch.gather(
                after_rate.T.unsqueeze(-1).expand(-1, -1, 5),
                1,
                predecessor.permute(1, 0, 2),
            )
            gathered = torch.where(
                valid.permute(1, 0, 2),
                gathered,
                torch.full_like(gathered, NEGATIVE_LOG_ZERO),
            )
            values, position_choice = torch.max(
                gathered + position_log[:, None, :], dim=2
            )
            predecessor_p = predecessor.permute(1, 0, 2).gather(
                2, position_choice[..., None]
            )[..., 0].T
            predecessor_r = rate_predecessor[predecessor_p, rate_grid]
            pointer = predecessor_p * rate_count + predecessor_r
            pointers.append(pointer.to(torch.int32).cpu())
            previous = values.T + unary[index, :, None]
        state_index = int(torch.argmax(previous).detach().cpu())
        path = np.empty(len(spec.suffix_index), dtype=np.int64)
        for index in range(len(spec.suffix_index) - 1, -1, -1):
            path[index] = state_index // rate_count
            if index > 0:
                p = state_index // rate_count
                r = state_index % rate_count
                state_index = int(pointers[index][p, r])
        return torch.as_tensor(path, dtype=torch.long, device=unary.device)


@dataclass(frozen=True)
class DecodeResult:
    posterior: np.ndarray
    prediction: np.ndarray
    marginal_map: np.ndarray
    viterbi: np.ndarray
    posterior_std: np.ndarray
    entropy: np.ndarray
    edge_mass: np.ndarray
    log_partition: float


def decode_unary(
    unary: Any,
    view: PreparedView,
    config: Mapping[str, Any],
    *,
    compute_viterbi: bool,
) -> DecodeResult:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for frozen exact exp347 decoding")
    with torch.no_grad():
        posterior_t, log_partition = exact_forward_backward(unary.float(), view.state, config)
        grid = torch.as_tensor(view.state.grid, dtype=torch.float32, device=unary.device)
        mean = posterior_t @ grid
        variance = posterior_t @ (grid**2) - mean**2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        entropy = -torch.sum(
            posterior_t * torch.log(torch.clamp(posterior_t, min=1e-12)), dim=1
        )
        edge_width = min(3, posterior_t.shape[1] // 2)
        edge_mass = posterior_t[:, :edge_width].sum(dim=1) + posterior_t[:, -edge_width:].sum(
            dim=1
        )
        map_index = torch.argmax(posterior_t, dim=1)
        viterbi_index = (
            exact_viterbi(unary.float(), view.state, config)
            if compute_viterbi
            else map_index
        )
    return DecodeResult(
        posterior=posterior_t.cpu().numpy().astype(np.float32),
        prediction=mean.cpu().numpy().astype(np.float32),
        marginal_map=grid[map_index].cpu().numpy().astype(np.float32),
        viterbi=grid[viterbi_index].cpu().numpy().astype(np.float32),
        posterior_std=std.cpu().numpy().astype(np.float32),
        entropy=entropy.cpu().numpy().astype(np.float32),
        edge_mass=edge_mass.cpu().numpy().astype(np.float32),
        log_partition=float(log_partition.cpu()),
    )


def decode_unary_batch(
    unaries: Sequence[Any],
    views: Sequence[PreparedView],
    config: Mapping[str, Any],
    *,
    compute_viterbi: bool,
    required_batch_size: int = 4,
) -> list[DecodeResult]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for batched exact exp347 decoding")
    if len(unaries) != len(views) or not views:
        raise ValueError("batched decode requires one unary per non-empty view")
    batch = build_batched_state_spec(
        [view.state for view in views], required_batch_size=required_batch_size
    )
    padded_unary = pad_unary_batch(unaries, batch)
    with torch.no_grad():
        posterior_batch, partitions = batched_exact_forward_backward(
            padded_unary.float(), batch, config
        )
    results: list[DecodeResult] = []
    for index, (unary, view) in enumerate(zip(unaries, views, strict=True)):
        rows = len(view.state.suffix_index)
        positions = len(view.state.grid)
        posterior_t = posterior_batch[index, :rows, :positions]
        grid = torch.as_tensor(
            view.state.grid, dtype=torch.float32, device=posterior_t.device
        )
        mean = posterior_t @ grid
        variance = posterior_t @ (grid**2) - mean**2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        entropy = -torch.sum(
            posterior_t * torch.log(torch.clamp(posterior_t, min=1e-12)), dim=1
        )
        edge_width = min(3, posterior_t.shape[1] // 2)
        edge_mass = posterior_t[:, :edge_width].sum(
            dim=1
        ) + posterior_t[:, -edge_width:].sum(dim=1)
        map_index = torch.argmax(posterior_t, dim=1)
        viterbi_index = (
            exact_viterbi(unary.float(), view.state, config)
            if compute_viterbi
            else map_index
        )
        results.append(
            DecodeResult(
                posterior=posterior_t.cpu().numpy().astype(np.float32),
                prediction=mean.cpu().numpy().astype(np.float32),
                marginal_map=grid[map_index].cpu().numpy().astype(np.float32),
                viterbi=grid[viterbi_index].cpu().numpy().astype(np.float32),
                posterior_std=std.cpu().numpy().astype(np.float32),
                entropy=entropy.cpu().numpy().astype(np.float32),
                edge_mass=edge_mass.cpu().numpy().astype(np.float32),
                log_partition=float(partitions[index].cpu()),
            )
        )
    return results


# %% [markdown]
# ## 8. Four-window structured training and outer-train early stopping

# %%
def stable_window_order(keys: Sequence[WindowKey], seed: int, epoch: int) -> list[WindowKey]:
    return sorted(
        keys,
        key=lambda key: stable_uint64(
            EXPERIMENT_NAME, "epoch-order", seed, epoch, key.well, key.slot, key.start_row
        ),
    )


def prepare_training_window(
    key: WindowKey,
    train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[WellInput, PreparedView, WellTruth, StateSpec]:
    item = load_well_input(key.well, train_dir)
    view = prepare_view(item, item.tvt_input, config, view_name="official_prefix_encoder")
    if not np.array_equal(view.tvt_input, item.tvt_input, equal_nan=True):
        raise ValueError("teacher boundary entered the encoder TVT_input")
    window_state = build_window_state_spec(item, view.state, key, config)
    truth = load_well_truth(key.well, train_dir)
    return item, view, truth, window_state


def window_training_loss(
    model: Any,
    view: PreparedView,
    truth: WellTruth,
    window_state: StateSpec,
    config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, float]]:
    unary, temperature = model_unary(model, view, device, window_state.suffix_index)
    target = truth.tvt[window_state.suffix_index]
    if not np.isfinite(target).all():
        raise ValueError(f"{view.well}: window truth contains non-finite values")
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    truth_index = torch.as_tensor(
        nearest_grid_indices(window_state.grid, target), dtype=torch.long, device=device
    )
    structured = SoftLabelStructuredNLL.apply(
        unary, target_tensor, window_state, config
    )
    local = F.cross_entropy(unary, truth_index)
    structured_weight = float(
        get_nested(config, "model.training.objective.structured_label_nll_weight", 1.0)
    )
    local_weight = float(
        get_nested(config, "model.training.objective.local_true_state_ce_weight", 0.25)
    )
    if structured_weight != 1.0 or local_weight != 0.25:
        raise ValueError("exp347 structured/local objective weights changed")
    loss = structured_weight * structured + local_weight * local
    coverage = float(
        np.mean((target >= window_state.grid[0]) & (target <= window_state.grid[-1]))
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "structured_label_nll": float(structured.detach().cpu()),
        "local_ce": float(local.detach().cpu()),
        "temperature": temperature,
        "target_in_grid_rate": coverage,
        "tokens": float(len(target)),
        "exact_dp_sweeps": 4.0,
    }


def fixed_window_batches(
    keys: Sequence[WindowKey], batch_size: int = 4
) -> list[list[WindowKey]]:
    if batch_size != 4:
        raise ValueError("exp347 fixes four consecutive windows per batch")
    return [list(keys[start : start + batch_size]) for start in range(0, len(keys), batch_size)]


def prepare_training_batch(
    keys: Sequence[WindowKey], train_dir: Path, config: Mapping[str, Any]
) -> tuple[list[PreparedView], list[WellTruth], list[StateSpec]]:
    views: list[PreparedView] = []
    truths: list[WellTruth] = []
    states: list[StateSpec] = []
    for key in keys:
        _, view, truth, state = prepare_training_window(key, train_dir, config)
        views.append(view)
        truths.append(truth)
        states.append(state)
    return views, truths, states


def batched_window_training_loss(
    model: Any,
    views: Sequence[PreparedView],
    truths: Sequence[WellTruth],
    states: Sequence[StateSpec],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, list[dict[str, float]], BatchedStateSpec]:
    if not (len(views) == len(truths) == len(states)) or not views:
        raise ValueError("batched training inputs must be non-empty and aligned")
    batch_size = int(get_nested(config, "model.training.batching.windows_per_batch", 4))
    if batch_size != 4:
        raise ValueError("exp347 windows_per_batch must remain four")
    unaries: list[Any] = []
    temperatures: list[float] = []
    target_values: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    coverages: list[float] = []
    for view, truth, state in zip(views, truths, states, strict=True):
        unary, temperature = model_unary(model, view, device, state.suffix_index)
        target = truth.tvt[state.suffix_index]
        if not np.isfinite(target).all():
            raise ValueError(f"{view.well}: window truth contains non-finite values")
        unaries.append(unary)
        temperatures.append(temperature)
        target_values.append(np.asarray(target, dtype=np.float32))
        target_indices.append(nearest_grid_indices(state.grid, target))
        coverages.append(
            float(np.mean((target >= state.grid[0]) & (target <= state.grid[-1])))
        )
    batch = build_batched_state_spec(states, required_batch_size=batch_size)
    unary_batch = pad_unary_batch(unaries, batch)
    target_batch = pad_target_batch(
        target_values, batch, dtype=torch.float32, device=device
    )
    structured = BatchedSoftLabelStructuredNLL.apply(
        unary_batch, target_batch, batch, config
    )
    padded_indices = torch.full(
        batch.row_mask.shape, -100, dtype=torch.long, device=device
    )
    for index, values in enumerate(target_indices):
        padded_indices[index, : len(values)] = torch.as_tensor(
            values, dtype=torch.long, device=device
        )
    local_rows = F.cross_entropy(
        unary_batch.reshape(-1, unary_batch.shape[-1]),
        padded_indices.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(padded_indices.shape)
    token_count = torch.as_tensor(
        batch.row_mask.sum(axis=1), dtype=unary_batch.dtype, device=device
    )
    local = local_rows.sum(dim=1) / torch.clamp(token_count, min=1.0)
    active = torch.as_tensor(batch.active_mask, dtype=torch.bool, device=device)
    structured_weight = float(
        get_nested(config, "model.training.objective.structured_label_nll_weight", 1.0)
    )
    local_weight = float(
        get_nested(config, "model.training.objective.local_true_state_ce_weight", 0.25)
    )
    if structured_weight != 1.0 or local_weight != 0.25:
        raise ValueError("exp347 structured/local objective weights changed")
    per_window = structured_weight * structured + local_weight * local
    loss = per_window[active].mean()
    diagnostics = []
    for index, state in enumerate(states):
        diagnostics.append(
            {
                "loss": float(per_window[index].detach().cpu()),
                "structured_label_nll": float(structured[index].detach().cpu()),
                "local_ce": float(local[index].detach().cpu()),
                "temperature": temperatures[index],
                "target_in_grid_rate": coverages[index],
                "tokens": float(len(state.suffix_index)),
                "exact_dp_sweeps": 4.0,
                "batch_active_windows": float(len(states)),
                "batch_padded_windows": float(batch_size - len(states)),
            }
        )
    return loss, diagnostics, batch


def evaluate_early_stop_window_objective(
    model: Any,
    keys: Sequence[WindowKey],
    train_dir: Path,
    config: Mapping[str, Any],
    device: Any,
) -> float:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for key_batch in fixed_window_batches(keys):
            views, truths, states = prepare_training_batch(key_batch, train_dir, config)
            _, diagnostics, _ = batched_window_training_loss(
                model, views, truths, states, config, device
            )
            values.extend(item["loss"] for item in diagnostics)
    return float(np.mean(values)) if values else float("inf")


def train_fold0_model(
    config: Mapping[str, Any],
    train_dir: Path,
    schedule: pd.DataFrame,
    boundary: pd.DataFrame,
    device: Any,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    seed = int(get_nested(config, "reproducibility.seed", 42))
    set_reproducibility(seed)
    model = PrefixConditionedUnary(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_nested(config, "model.training.learning_rate", 3e-4)),
        weight_decay=float(get_nested(config, "model.training.weight_decay", 1e-4)),
    )
    amp_enabled = bool(get_nested(config, "model.training.amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(get_nested(config, "model.training.gradient_accumulation_windows", 1))
    if accumulation != 1:
        raise ValueError("exp347 fixes gradient accumulation to one batched update")
    max_epochs = int(get_nested(config, "model.training.max_epochs", 8))
    clip_norm = float(get_nested(config, "model.training.gradient_clip_norm", 1.0))
    patience = int(get_nested(config, "model.training.early_stopping_patience_epochs", 2))
    min_delta = float(
        get_nested(config, "model.training.early_stopping_min_delta_nll_per_token", 0.001)
    )
    maximum_windows = int(
        get_nested(config, "model.training.windows.maximum_windows_per_epoch", 1668)
    )
    maximum_positions = int(
        get_nested(config, "model.training.windows.maximum_scored_positions_per_epoch", 427008)
    )
    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_objective = float("inf")
    epochs_without_improvement = 0
    started = time.time()
    for epoch in range(max_epochs):
        fit_keys = window_keys_from_manifests(schedule, boundary, role="fit", epoch=epoch)
        early_keys = window_keys_from_manifests(
            schedule, boundary, role="early_stop", epoch=epoch
        )
        if (
            len(fit_keys) > maximum_windows
            or sum(key.scored_rows for key in fit_keys) > maximum_positions
        ):
            raise ValueError("fit window workload exceeds the fixed exp347 ceiling")
        ordered = stable_window_order(fit_keys, seed, epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_rows: list[dict[str, float]] = []
        for key_batch in fixed_window_batches(ordered):
            views, truths, states = prepare_training_batch(key_batch, train_dir, config)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, diagnostics, _ = batched_window_training_loss(
                    model, views, truths, states, config, device
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            epoch_rows.extend(diagnostics)
        early_objective = evaluate_early_stop_window_objective(
            model, early_keys, train_dir, config, device
        )
        row = {
            "epoch": epoch + 1,
            "train_windows": len(epoch_rows),
            "train_scored_positions": int(sum(item["tokens"] for item in epoch_rows)),
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])),
            "train_structured_label_nll": float(
                np.mean([item["structured_label_nll"] for item in epoch_rows])
            ),
            "train_local_ce": float(np.mean([item["local_ce"] for item in epoch_rows])),
            "train_temperature": float(np.mean([item["temperature"] for item in epoch_rows])),
            "train_target_in_grid_rate": float(
                np.mean([item["target_in_grid_rate"] for item in epoch_rows])
            ),
            "early_stop_window_objective": early_objective,
            "elapsed_seconds": time.time() - started,
        }
        history_rows.append(row)
        print(json.dumps(to_jsonable(row), sort_keys=True), flush=True)
        if early_objective < best_objective - min_delta:
            best_objective = early_objective
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite outer-train window-objective checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device).eval()
    meta = {
        "best_early_stop_window_objective": best_objective,
        "selected_epoch": int(
            min(history_rows, key=lambda row: row["early_stop_window_objective"])["epoch"]
        ),
        "completed_epochs": len(history_rows),
        "train_seconds": time.time() - started,
    }
    return model, pd.DataFrame(history_rows), meta


# %% [markdown]
# ## 9. Practical equivalence comparison, diagnostics, and gates

# %%
def _cuda_sync(device: Any) -> None:
    if TORCH_AVAILABLE and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _resolve_evidence_entry(
    config: Mapping[str, Any],
    dotted_key: str,
) -> tuple[Path, str]:
    entry = get_nested(config, dotted_key, {}) or {}
    filename = str(entry.get("filename", ""))
    expected_sha = str(entry.get("sha256", ""))
    candidates = list(entry.get("candidates", []))
    if not filename or len(expected_sha) != 64 or not candidates:
        raise ValueError(f"{dotted_key} evidence contract is incomplete")
    path = resolve_existing_path(candidates, filename)
    actual_sha = sha256_path(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{dotted_key} SHA mismatch: actual={actual_sha} expected={expected_sha}"
        )
    return path, actual_sha


def load_parent_fixed16_contract(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    evidence: dict[str, Any] = {}
    resolved: dict[str, Path] = {}
    for name in (
        "parent_source",
        "parent_config",
        "parent_stage0_report",
        "parent_window_manifest",
        "parent_boundary_manifest",
    ):
        path, actual_sha = _resolve_evidence_entry(config, f"data.{name}")
        resolved[name] = path
        evidence[name] = {"path": str(path), "sha256": actual_sha}

    parent_report = json.loads(resolved["parent_stage0_report"].read_text())
    if parent_report.get("experiment") != "exp347_prefix_gr_unary_batched_window_exact_ssm":
        raise ValueError("parent Stage 0 report is not exp347")
    if parent_report.get("fixed_window_count") != 16:
        raise ValueError("parent Stage 0 report does not contain fixed16")
    if parent_report.get("gate", {}).get("passed") is not False:
        raise ValueError("exp347 terminal FAIL evidence changed")
    parent_error = float(
        parent_report.get("scalar_batch_parity", {}).get(
            "posterior_max_abs_error", float("nan")
        )
    )
    if not math.isclose(parent_error, 1.4662742614746094e-05, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("exp347 posterior-cell failure evidence changed")

    windows = pd.read_csv(resolved["parent_window_manifest"])
    boundaries = pd.read_csv(resolved["parent_boundary_manifest"])
    if len(windows) != 16 or windows["well"].nunique() != 16:
        raise ValueError("parent window manifest must contain 16 unique wells")
    if windows["benchmark_order"].tolist() != list(range(16)):
        raise ValueError("parent fixed16 benchmark order changed")
    if not windows["active"].astype(bool).all() or set(windows["role"]) != {"fit"}:
        raise ValueError("parent fixed16 windows must all be active fit windows")
    if len(boundaries) != 16 or boundaries["well"].nunique() != 16:
        raise ValueError("parent boundary manifest must contain 16 unique wells")
    if set(boundaries["encoder_tvt_input_source"]) != {"official_prefix_only"}:
        raise ValueError("parent boundary encoder contract changed")
    if set(boundaries["boundary_source"]) != {"official_prefix"}:
        raise ValueError("exp393 fixed16 must not need interior truth for boundaries")
    return windows, boundaries, evidence


def prepare_audit_windows(
    keys: Sequence[WindowKey],
    train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[list[WellInput], list[PreparedView], list[StateSpec], dict[str, Any]]:
    items: list[WellInput] = []
    views: list[PreparedView] = []
    states: list[StateSpec] = []
    input_rows: list[dict[str, Any]] = []
    wells = list_paired_wells(train_dir)
    fold_map = build_fold_map(wells, int(get_nested(config, "validation.n_folds", 5)))
    outer_train, outer_valid = split_stage_a_wells(fold_map)
    selected_wells = {key.well for key in keys}
    if not selected_wells.issubset(set(outer_train)):
        overlap = sorted(selected_wells & set(outer_valid))
        raise ValueError(f"parent fixed16 overlaps outer-valid wells: {overlap}")
    for key in keys:
        item = load_well_input(key.well, train_dir)
        view = prepare_view(item, item.tvt_input, config, view_name="official")
        state = build_window_state_spec(item, view.state, key, config)
        if state.suffix_index[0] != key.start_row or len(state.suffix_index) != key.scored_rows:
            raise ValueError(f"{key.well}: fixed-window row identity changed")
        items.append(item)
        views.append(view)
        states.append(state)
        input_rows.append(
            {
                "well": key.well,
                "horizontal_path": str(item.horizontal_path),
                "horizontal_sha256": sha256_path(item.horizontal_path),
                "typewell_path": str(item.typewell_path),
                "typewell_sha256": sha256_path(item.typewell_path),
                "row_count": int(len(item.md)),
                "window_start_row": int(key.start_row),
                "window_stop_row": int(key.stop_row),
                "model_input_columns": list(HORIZONTAL_INPUT_COLUMNS),
                "outer_valid_truth_access_before_unary_freeze": 0,
            }
        )
    manifest = {
        "selected_well_count": len(selected_wells),
        "fixed_window_count": len(keys),
        "outer_train_well_count": len(outer_train),
        "outer_valid_well_count": len(outer_valid),
        "outer_valid_truth_access_count": 0,
        "rows": input_rows,
    }
    return items, views, states, manifest


def freeze_unaries_once(
    model: Any,
    views: Sequence[PreparedView],
    states: Sequence[StateSpec],
    device: Any,
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    if len(views) != 16 or len(states) != 16:
        raise ValueError("unary freeze requires exactly the parent fixed16 windows")
    model.eval()
    frozen: list[Any] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for window_order, (view, state) in enumerate(
            zip(views, states, strict=True)
        ):
            _cuda_sync(device)
            started = time.perf_counter()
            unary, temperature = model_unary(
                model, view, device, state.suffix_index
            )
            _cuda_sync(device)
            elapsed = time.perf_counter() - started
            value = unary.detach().clone().contiguous()
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{view.well}: frozen unary must be finite float32")
            frozen.append(value)
            rows.append(
                {
                    "phase": "frozen_unary_generation",
                    "mode": "temporary_seed42_model_eval",
                    "batch_id": window_order,
                    "window_order": window_order,
                    "well": view.well,
                    "rows": int(value.shape[0]),
                    "positions": int(value.shape[1]),
                    "seconds": elapsed,
                    "temperature": temperature,
                    "unary_sha256": array_content_sha256(
                        value.detach().cpu().numpy()
                    ),
                }
            )
    manifest = {
        "logical_unary_generation_count": 1,
        "temporary_neural_model_count": 1,
        "persisted_model_count": 0,
        "model_eval": not model.training,
        "dropout_enabled_during_comparison": False,
        "dtype": "float32",
        "window_count": len(frozen),
        "per_window": [
            {
                "window_order": row["window_order"],
                "well": row["well"],
                "shape": [row["rows"], row["positions"]],
                "temperature": row["temperature"],
                "unary_sha256": row["unary_sha256"],
            }
            for row in rows
        ],
        "combined_unary_sha256": array_content_sha256(
            *[value.detach().cpu().numpy() for value in frozen]
        ),
        "truth_loaded_before_freeze": False,
        "outer_valid_truth_access_count_before_freeze": 0,
    }
    return frozen, manifest, rows


def generalized_initial_log_prior(
    spec: StateSpec,
    config: Mapping[str, Any],
    device: Any,
    dtype: Any,
) -> Any:
    state = get_nested(config, "model.state_space", {}) or {}
    position = torch.arange(len(spec.grid), dtype=dtype, device=device)
    rates = torch.as_tensor(spec.rates, dtype=dtype, device=device)
    position_log = -0.5 * (
        (position - float(spec.start_p))
        * float(state["step"])
        / float(state["start_sig"])
    ) ** 2
    position_log = torch.where(
        position_log >= -60.0,
        position_log,
        torch.full_like(position_log, NEGATIVE_LOG_ZERO),
    )
    rate_log = -0.5 * (
        (rates - float(spec.init_rate)) / float(state["r0_sig"])
    ) ** 2
    return position_log[:, None] + rate_log[None, :]


def generalized_rate_transition_tables(
    dm: float,
    rates: Any,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any]:
    state = get_nested(config, "model.state_space", {}) or {}
    rate_step = rates[1] - rates[0]
    sigma_step = float(state["sig_r"]) * math.sqrt(float(dm))
    variance_cells = (sigma_step / rate_step) ** 2
    mean_move = (
        -(1.0 - float(state["mom"])) * rates * float(dm) / rate_step
    )
    p_plus = torch.clamp(0.5 * (variance_cells + mean_move), min=1e-12)
    p_minus = torch.clamp(0.5 * (variance_cells - mean_move), min=1e-12)
    total = p_plus + p_minus
    factor = torch.where(total > 0.9, 0.9 / total, torch.ones_like(total))
    p_plus = p_plus * factor
    p_minus = p_minus * factor
    kernel = torch.stack(
        [torch.log(p_minus), torch.log1p(-p_plus - p_minus), torch.log(p_plus)],
        dim=1,
    )
    rate_count = len(rates)
    destination = torch.arange(rate_count, device=rates.device)[:, None]
    offsets = torch.tensor([-1, 0, 1], device=rates.device)[None, :]
    source = destination + offsets
    source_valid = (source >= 0) & (source < rate_count)
    source_clamped = source.clamp(0, rate_count - 1)
    delta_column = (-offsets + 1).expand_as(source)
    forward_log = kernel[source_clamped, delta_column]
    forward_log = torch.where(
        source_valid,
        forward_log,
        torch.full_like(forward_log, NEGATIVE_LOG_ZERO),
    )
    source_rate = torch.arange(rate_count, device=rates.device)[:, None]
    delta = torch.tensor([-1, 0, 1], device=rates.device)[None, :]
    backward_destination = source_rate + delta
    backward_valid = (
        (backward_destination >= 0) & (backward_destination < rate_count)
    )
    backward_destination = backward_destination.clamp(0, rate_count - 1)
    backward_log = torch.where(
        backward_valid,
        kernel,
        torch.full_like(kernel, NEGATIVE_LOG_ZERO),
    )
    return source_clamped, forward_log, backward_destination, backward_log


def generalized_position_transition_tables(
    dm: float,
    dz: float,
    rates: Any,
    position_count: int,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    state = get_nested(config, "model.state_space", {}) or {}
    step = float(state["step"])
    sigma_position = max(float(state["sig_p"]), 0.35 * step)
    mu = rates * float(dm) - float(dz)
    base = torch.floor(mu / step + 0.5).to(torch.long)
    offsets = torch.arange(-2, 3, device=rates.device, dtype=torch.long)
    shifts = base[:, None] + offsets[None, :]
    delta = shifts.to(rates.dtype) * step - mu[:, None]
    position_log = -0.5 * (delta / sigma_position) ** 2
    position_log = position_log - torch.logsumexp(
        position_log, dim=1, keepdim=True
    )
    position = torch.arange(position_count, device=rates.device)[:, None, None]
    predecessor = position - shifts[None, :, :]
    predecessor_valid = (
        (predecessor >= 0) & (predecessor < position_count)
    )
    predecessor = predecessor.clamp(0, position_count - 1)
    successor = position + shifts[None, :, :]
    successor_valid = (successor >= 0) & (successor < position_count)
    successor = successor.clamp(0, position_count - 1)
    return (
        predecessor,
        predecessor_valid,
        successor,
        successor_valid,
        position_log,
    )


def generalized_forward_transition(
    previous: Any,
    emission: Any,
    dm: float,
    dz: float,
    rates: Any,
    config: Mapping[str, Any],
) -> Any:
    source, rate_log, _, _ = generalized_rate_transition_tables(
        dm, rates, config
    )
    rate_values = previous[:, source] + rate_log[None, :, :]
    after_rate = torch.logsumexp(rate_values, dim=2)
    predecessor, valid, _, _, position_log = (
        generalized_position_transition_tables(
            dm, dz, rates, previous.shape[0], config
        )
    )
    gathered = torch.gather(
        after_rate.T.unsqueeze(-1).expand(-1, -1, 5),
        1,
        predecessor.permute(1, 0, 2),
    )
    gathered = torch.where(
        valid.permute(1, 0, 2),
        gathered,
        torch.full_like(gathered, NEGATIVE_LOG_ZERO),
    )
    after_position = torch.logsumexp(
        gathered + position_log[:, None, :], dim=2
    ).T
    return after_position + emission[:, None]


def generalized_backward_transition(
    beta_next: Any,
    emission_next: Any,
    dm: float,
    dz: float,
    rates: Any,
    config: Mapping[str, Any],
) -> Any:
    _, _, successor, valid, position_log = (
        generalized_position_transition_tables(
            dm, dz, rates, beta_next.shape[0], config
        )
    )
    future = beta_next + emission_next[:, None]
    gathered = torch.gather(
        future.T.unsqueeze(-1).expand(-1, -1, 5),
        1,
        successor.permute(1, 0, 2),
    )
    gathered = torch.where(
        valid.permute(1, 0, 2),
        gathered,
        torch.full_like(gathered, NEGATIVE_LOG_ZERO),
    )
    after_position = torch.logsumexp(
        gathered + position_log[:, None, :], dim=2
    ).T
    _, _, destination, rate_log = generalized_rate_transition_tables(
        dm, rates, config
    )
    rate_values = after_position[:, destination] + rate_log[None, :, :]
    return torch.logsumexp(rate_values, dim=2)


def scalar_exact_forward_backward_dtype(
    unary: Any,
    spec: StateSpec,
    config: Mapping[str, Any],
    dtype: Any,
) -> tuple[Any, Any]:
    unary_value = unary.to(dtype=dtype)
    rates = torch.as_tensor(spec.rates, dtype=dtype, device=unary.device)
    previous = generalized_initial_log_prior(
        spec, config, unary.device, dtype
    )
    alpha_rows = []
    for index in range(len(spec.suffix_index)):
        previous = generalized_forward_transition(
            previous,
            unary_value[index],
            float(spec.dm[index]),
            float(spec.dz[index]),
            rates,
            config,
        )
        alpha_rows.append(previous)
    alpha = torch.stack(alpha_rows, dim=0)
    log_partition = torch.logsumexp(alpha[-1].reshape(-1), dim=0)
    posterior_rows: list[Any] = [None] * len(spec.suffix_index)
    beta = torch.zeros_like(alpha[-1])
    joint = alpha[-1] + beta
    posterior_rows[-1] = torch.exp(
        torch.logsumexp(joint, dim=1)
        - torch.logsumexp(joint.reshape(-1), dim=0)
    )
    for index in range(len(spec.suffix_index) - 1, 0, -1):
        beta = generalized_backward_transition(
            beta,
            unary_value[index],
            float(spec.dm[index]),
            float(spec.dz[index]),
            rates,
            config,
        )
        joint = alpha[index - 1] + beta
        posterior_rows[index - 1] = torch.exp(
            torch.logsumexp(joint, dim=1)
            - torch.logsumexp(joint.reshape(-1), dim=0)
        )
    return torch.stack(posterior_rows, dim=0), log_partition


def decode_frozen_unaries(
    unaries: Sequence[Any],
    states: Sequence[StateSpec],
    config: Mapping[str, Any],
    device: Any,
    mode: str,
) -> dict[str, Any]:
    allowed = {
        "scalar_fp32_reference",
        "batched_fp32_batch1",
        "batched_fp32_batch4_production",
        "scalar_fp64_first4_diagnostic",
    }
    if mode not in allowed:
        raise ValueError(f"unknown comparison mode {mode}")
    limit = (
        int(get_nested(config, "validation.stage0.fp64_diagnostic_window_count", 4))
        if mode == "scalar_fp64_first4_diagnostic"
        else len(unaries)
    )
    posteriors: list[np.ndarray] = []
    partitions: list[float] = []
    measurements: list[dict[str, Any]] = []
    padding_frames: list[pd.DataFrame] = []
    invalid_posterior_max = 0.0
    with torch.no_grad():
        if mode in {"scalar_fp32_reference", "scalar_fp64_first4_diagnostic"}:
            dtype = (
                torch.float64
                if mode == "scalar_fp64_first4_diagnostic"
                else torch.float32
            )
            for index, (unary, state) in enumerate(
                zip(unaries[:limit], states[:limit], strict=True)
            ):
                _cuda_sync(device)
                started = time.perf_counter()
                if dtype == torch.float32:
                    posterior, partition = exact_forward_backward(
                        unary, state, config
                    )
                else:
                    posterior, partition = scalar_exact_forward_backward_dtype(
                        unary, state, config, dtype
                    )
                _cuda_sync(device)
                elapsed = time.perf_counter() - started
                posteriors.append(posterior.detach().cpu().numpy())
                partitions.append(float(partition.detach().cpu()))
                measurements.append(
                    {
                        "phase": "posterior_comparison",
                        "mode": mode,
                        "batch_id": index,
                        "window_order": index,
                        "well": None,
                        "rows": len(state.suffix_index),
                        "positions": len(state.grid),
                        "seconds": elapsed,
                    }
                )
        else:
            batch_size = (
                1 if mode == "batched_fp32_batch1" else 4
            )
            for batch_id, start in enumerate(range(0, limit, batch_size)):
                stop = min(limit, start + batch_size)
                batch_states = list(states[start:stop])
                batch_unaries = list(unaries[start:stop])
                batch = build_batched_state_spec(
                    batch_states, required_batch_size=batch_size
                )
                padded_unary = pad_unary_batch(batch_unaries, batch)
                _cuda_sync(device)
                started = time.perf_counter()
                posterior_batch, partition_batch = (
                    batched_exact_forward_backward(
                        padded_unary, batch, config
                    )
                )
                _cuda_sync(device)
                elapsed = time.perf_counter() - started
                row_mask = torch.as_tensor(
                    batch.row_mask,
                    dtype=torch.bool,
                    device=posterior_batch.device,
                )
                position_mask = torch.as_tensor(
                    batch.position_mask,
                    dtype=torch.bool,
                    device=posterior_batch.device,
                )
                invalid = posterior_batch.masked_select(
                    ~(row_mask[:, :, None] & position_mask[:, None, :])
                )
                if invalid.numel():
                    invalid_posterior_max = max(
                        invalid_posterior_max,
                        float(torch.max(torch.abs(invalid)).detach().cpu()),
                    )
                padding = batch_padding_manifest(batch)
                padding["mode"] = mode
                padding["batch_id"] = batch_id
                padding_frames.append(padding)
                for local_index, state in enumerate(batch_states):
                    rows = len(state.suffix_index)
                    positions = len(state.grid)
                    posteriors.append(
                        posterior_batch[
                            local_index, :rows, :positions
                        ].detach().cpu().numpy()
                    )
                    partitions.append(
                        float(partition_batch[local_index].detach().cpu())
                    )
                measurements.append(
                    {
                        "phase": "posterior_comparison",
                        "mode": mode,
                        "batch_id": batch_id,
                        "window_order": start,
                        "well": None,
                        "rows": int(sum(len(state.suffix_index) for state in batch_states)),
                        "positions": int(max(len(state.grid) for state in batch_states)),
                        "seconds": elapsed,
                    }
                )
    if len(posteriors) != limit:
        raise ValueError(f"{mode}: incomplete posterior outputs")
    means = [
        posterior.astype(np.float64) @ state.grid.astype(np.float64)
        for posterior, state in zip(
            posteriors, states[:limit], strict=True
        )
    ]
    maps = [np.argmax(posterior, axis=1) for posterior in posteriors]
    finite = all(
        np.isfinite(value).all()
        for value in [*posteriors, *means, np.asarray(partitions)]
    )
    return {
        "mode": mode,
        "posteriors": posteriors,
        "partitions": np.asarray(partitions, dtype=np.float64),
        "means": means,
        "maps": maps,
        "measurements": measurements,
        "padding": (
            pd.concat(padding_frames, ignore_index=True)
            if padding_frames
            else pd.DataFrame()
        ),
        "invalid_posterior_max_abs": invalid_posterior_max,
        "finite": bool(finite),
        "posterior_sha256": array_content_sha256(*posteriors),
        "readout_sha256": array_content_sha256(
            *means, *[value.astype(np.int64) for value in maps]
        ),
    }


def load_audit_truths(
    keys: Sequence[WindowKey],
    train_dir: Path,
    states: Sequence[StateSpec],
) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for key, state in zip(keys, states, strict=True):
        truth = load_well_truth(key.well, train_dir)
        selected = np.asarray(truth.tvt[state.suffix_index], dtype=np.float32)
        if not np.isfinite(selected).all():
            raise ValueError(f"{key.well}: audit objective truth is non-finite")
        values.append(selected)
    return values


def run_frozen_unary_objective_parity(
    unaries: Sequence[Any],
    targets: Sequence[np.ndarray],
    states: Sequence[StateSpec],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(unaries) != 16 or len(targets) != 16 or len(states) != 16:
        raise ValueError("objective parity requires all fixed16 windows")
    loss_error = 0.0
    partition_error = 0.0
    gradient_error = 0.0
    update_error = 0.0
    invalid_gradient_max = 0.0
    finite = True
    measurements: list[dict[str, Any]] = []
    learning_rate = float(
        get_nested(config, "model.training.learning_rate", 3e-4)
    )
    weight_decay = float(
        get_nested(config, "model.training.weight_decay", 1e-4)
    )
    clip_norm = float(
        get_nested(config, "model.training.gradient_clip_norm", 1.0)
    )
    for batch_id, start in enumerate(range(0, 16, 4)):
        stop = start + 4
        batch_states = list(states[start:stop])
        batch_targets = list(targets[start:stop])
        scalar_parameters = [
            nn.Parameter(value.detach().clone())
            for value in unaries[start:stop]
        ]
        batch = build_batched_state_spec(
            batch_states, required_batch_size=4
        )
        batched_parameter = nn.Parameter(
            pad_unary_batch(unaries[start:stop], batch).detach().clone()
        )
        scalar_optimizer = torch.optim.AdamW(
            scalar_parameters, lr=learning_rate, weight_decay=weight_decay
        )
        batched_optimizer = torch.optim.AdamW(
            [batched_parameter], lr=learning_rate, weight_decay=weight_decay
        )
        scalar_optimizer.zero_grad(set_to_none=True)
        batched_optimizer.zero_grad(set_to_none=True)
        _cuda_sync(device)
        started = time.perf_counter()
        scalar_losses = []
        target_indices = []
        scalar_partitions = []
        scalar_conditioned_partitions = []
        for parameter, target, state in zip(
            scalar_parameters, batch_targets, batch_states, strict=True
        ):
            target_tensor = torch.as_tensor(
                target, dtype=torch.float32, device=device
            )
            truth_index = nearest_grid_indices(state.grid, target)
            index_tensor = torch.as_tensor(
                truth_index, dtype=torch.long, device=device
            )
            terms = soft_label_structured_terms(
                parameter.detach(), target_tensor, state, config
            )
            scalar_partitions.append(terms[2])
            scalar_conditioned_partitions.append(terms[3])
            structured = SoftLabelStructuredNLL.apply(
                parameter, target_tensor, state, config
            )
            local = F.cross_entropy(parameter, index_tensor)
            scalar_losses.append(structured + 0.25 * local)
            target_indices.append(truth_index)
        scalar_loss = torch.stack(scalar_losses).mean()
        target_batch = pad_target_batch(
            batch_targets,
            batch,
            dtype=torch.float32,
            device=device,
        )
        batched_terms = batched_soft_label_structured_terms(
            batched_parameter.detach(), target_batch, batch, config
        )
        batched_structured = BatchedSoftLabelStructuredNLL.apply(
            batched_parameter, target_batch, batch, config
        )
        padded_indices = torch.full(
            batch.row_mask.shape, -100, dtype=torch.long, device=device
        )
        for index, values in enumerate(target_indices):
            padded_indices[index, : len(values)] = torch.as_tensor(
                values, dtype=torch.long, device=device
            )
        local_rows = F.cross_entropy(
            batched_parameter.reshape(-1, batched_parameter.shape[-1]),
            padded_indices.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(padded_indices.shape)
        token_count = torch.as_tensor(
            batch.row_mask.sum(axis=1),
            dtype=batched_parameter.dtype,
            device=device,
        )
        batched_local = local_rows.sum(dim=1) / torch.clamp(
            token_count, min=1.0
        )
        active = torch.as_tensor(
            batch.active_mask, dtype=torch.bool, device=device
        )
        batched_loss = (
            batched_structured + 0.25 * batched_local
        )[active].mean()
        loss_error = max(
            loss_error,
            float(torch.abs(scalar_loss - batched_loss).detach().cpu()),
        )
        for index in range(4):
            partition_error = max(
                partition_error,
                float(
                    torch.abs(
                        scalar_partitions[index]
                        - batched_terms[2][index]
                    ).detach().cpu()
                ),
                float(
                    torch.abs(
                        scalar_conditioned_partitions[index]
                        - batched_terms[3][index]
                    ).detach().cpu()
                ),
            )
        scalar_loss.backward()
        batched_loss.backward()
        if batched_parameter.grad is None:
            raise ValueError("batched unary gradient is missing")
        for index, (parameter, state) in enumerate(
            zip(scalar_parameters, batch_states, strict=True)
        ):
            if parameter.grad is None:
                raise ValueError("scalar unary gradient is missing")
            batch_gradient = batched_parameter.grad[
                index,
                : len(state.suffix_index),
                : len(state.grid),
            ]
            gradient_error = max(
                gradient_error,
                float(
                    torch.max(torch.abs(parameter.grad - batch_gradient))
                    .detach()
                    .cpu()
                ),
            )
            finite = finite and bool(
                torch.isfinite(parameter.grad).all()
                and torch.isfinite(batch_gradient).all()
            )
        row_mask = torch.as_tensor(
            batch.row_mask, dtype=torch.bool, device=device
        )
        position_mask = torch.as_tensor(
            batch.position_mask, dtype=torch.bool, device=device
        )
        invalid = batched_parameter.grad.masked_select(
            ~(row_mask[:, :, None] & position_mask[:, None, :])
        )
        if invalid.numel():
            invalid_gradient_max = max(
                invalid_gradient_max,
                float(torch.max(torch.abs(invalid)).detach().cpu()),
            )
        torch.nn.utils.clip_grad_norm_(scalar_parameters, clip_norm)
        torch.nn.utils.clip_grad_norm_([batched_parameter], clip_norm)
        scalar_optimizer.step()
        batched_optimizer.step()
        for index, (parameter, state) in enumerate(
            zip(scalar_parameters, batch_states, strict=True)
        ):
            batch_value = batched_parameter[
                index,
                : len(state.suffix_index),
                : len(state.grid),
            ]
            update_error = max(
                update_error,
                float(
                    torch.max(torch.abs(parameter - batch_value))
                    .detach()
                    .cpu()
                ),
            )
        _cuda_sync(device)
        measurements.append(
            {
                "phase": "loss_gradient_adamw_parity",
                "mode": "scalar_fp32_vs_batched_fp32_batch4",
                "batch_id": batch_id,
                "window_order": start,
                "well": None,
                "rows": int(sum(len(state.suffix_index) for state in batch_states)),
                "positions": int(max(len(state.grid) for state in batch_states)),
                "seconds": time.perf_counter() - started,
            }
        )
    report = {
        "window_count": 16,
        "batch_count": 4,
        "loss_max_abs_error": loss_error,
        "partition_max_abs_error": partition_error,
        "gradient_max_abs_error": gradient_error,
        "optimizer_step_max_abs_error": update_error,
        "invalid_gradient_max_abs": invalid_gradient_max,
        "finite": bool(
            finite
            and all(
                math.isfinite(value)
                for value in (
                    loss_error,
                    partition_error,
                    gradient_error,
                    update_error,
                    invalid_gradient_max,
                )
            )
        ),
    }
    return report, measurements


def build_practical_readout(
    keys: Sequence[WindowKey],
    states: Sequence[StateSpec],
    scalar: Mapping[str, Any],
    batch1: Mapping[str, Any],
    batch4: Mapping[str, Any],
    fp64: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    posterior_cell_values: list[np.ndarray] = []
    for index, (key, state) in enumerate(zip(keys, states, strict=True)):
        reference = np.asarray(scalar["posteriors"][index], dtype=np.float64)
        candidate1 = np.asarray(batch1["posteriors"][index], dtype=np.float64)
        candidate4 = np.asarray(batch4["posteriors"][index], dtype=np.float64)
        scalar_mean = np.asarray(scalar["means"][index], dtype=np.float64)
        batch1_mean = np.asarray(batch1["means"][index], dtype=np.float64)
        batch4_mean = np.asarray(batch4["means"][index], dtype=np.float64)
        scalar_map = np.asarray(scalar["maps"][index], dtype=np.int64)
        batch1_map = np.asarray(batch1["maps"][index], dtype=np.int64)
        batch4_map = np.asarray(batch4["maps"][index], dtype=np.int64)
        cell_abs = np.abs(reference - candidate4)
        posterior_cell_values.append(cell_abs.reshape(-1))
        total_variation = 0.5 * cell_abs.sum(axis=1)
        top2_scalar = np.partition(reference, -2, axis=1)[:, -2:]
        top2_batch4 = np.partition(candidate4, -2, axis=1)[:, -2:]
        frame = pd.DataFrame(
            {
                "window_order": index,
                "well": key.well,
                "row_index": state.suffix_index,
                "scalar_fp32_mean_tvt": scalar_mean,
                "batched_fp32_batch1_mean_tvt": batch1_mean,
                "batched_fp32_batch4_mean_tvt": batch4_mean,
                "batch1_minus_scalar_tvt_ft": batch1_mean - scalar_mean,
                "batch4_minus_scalar_tvt_ft": batch4_mean - scalar_mean,
                "scalar_fp32_map_state": scalar_map,
                "batched_fp32_batch1_map_state": batch1_map,
                "batched_fp32_batch4_map_state": batch4_map,
                "batch1_map_state_equal": batch1_map == scalar_map,
                "batch4_map_state_equal": batch4_map == scalar_map,
                "batch4_map_state_distance": np.abs(batch4_map - scalar_map),
                "batch4_map_tvt_distance_ft": np.abs(
                    state.grid[batch4_map] - state.grid[scalar_map]
                ),
                "posterior_cell_max_abs_error": cell_abs.max(axis=1),
                "posterior_cell_mean_abs_error": cell_abs.mean(axis=1),
                "row_total_variation": total_variation,
                "scalar_top2_margin": top2_scalar[:, 1] - top2_scalar[:, 0],
                "batch4_top2_margin": top2_batch4[:, 1] - top2_batch4[:, 0],
                "scalar_row_sum_abs_error": np.abs(reference.sum(axis=1) - 1.0),
                "batch1_row_sum_abs_error": np.abs(candidate1.sum(axis=1) - 1.0),
                "batch4_row_sum_abs_error": np.abs(candidate4.sum(axis=1) - 1.0),
            }
        )
        if index < len(fp64["posteriors"]):
            fp64_posterior = np.asarray(
                fp64["posteriors"][index], dtype=np.float64
            )
            fp64_mean = np.asarray(fp64["means"][index], dtype=np.float64)
            fp64_map = np.asarray(fp64["maps"][index], dtype=np.int64)
            frame["scalar_fp64_mean_tvt"] = fp64_mean
            frame["scalar_fp32_minus_fp64_tvt_ft"] = scalar_mean - fp64_mean
            frame["batch4_fp32_minus_fp64_tvt_ft"] = batch4_mean - fp64_mean
            frame["scalar_fp64_map_state"] = fp64_map
            frame["scalar_fp32_vs_fp64_total_variation"] = (
                0.5 * np.abs(reference - fp64_posterior).sum(axis=1)
            )
            frame["batch4_fp32_vs_fp64_total_variation"] = (
                0.5 * np.abs(candidate4 - fp64_posterior).sum(axis=1)
            )
            frame["scalar_fp64_row_sum_abs_error"] = np.abs(
                fp64_posterior.sum(axis=1) - 1.0
            )
        else:
            frame["scalar_fp64_mean_tvt"] = np.nan
            frame["scalar_fp32_minus_fp64_tvt_ft"] = np.nan
            frame["batch4_fp32_minus_fp64_tvt_ft"] = np.nan
            frame["scalar_fp64_map_state"] = np.nan
            frame["scalar_fp32_vs_fp64_total_variation"] = np.nan
            frame["batch4_fp32_vs_fp64_total_variation"] = np.nan
            frame["scalar_fp64_row_sum_abs_error"] = np.nan
        frames.append(frame)
    readout = pd.concat(frames, ignore_index=True)
    absolute_tvt = np.abs(
        readout["batch4_minus_scalar_tvt_ft"].to_numpy(dtype=np.float64)
    )
    posterior_cell = np.concatenate(posterior_cell_values)
    valid_fp64 = readout["scalar_fp64_mean_tvt"].notna()
    diagnostics = {
        "posterior_mean_tvt": {
            "rmse_ft": float(
                np.sqrt(
                    np.mean(
                        readout["batch4_minus_scalar_tvt_ft"].to_numpy(
                            dtype=np.float64
                        )
                        ** 2
                    )
                )
            ),
            "mean_abs_ft": float(np.mean(absolute_tvt)),
            "p95_abs_ft": float(np.quantile(absolute_tvt, 0.95)),
            "p99_abs_ft": float(np.quantile(absolute_tvt, 0.99)),
            "max_abs_ft": float(np.max(absolute_tvt)),
        },
        "marginal_map": {
            "agreement_rate": float(
                readout["batch4_map_state_equal"].mean()
            ),
            "disagreement_rows": int(
                (~readout["batch4_map_state_equal"]).sum()
            ),
            "max_state_distance": int(
                readout["batch4_map_state_distance"].max()
            ),
            "max_tvt_distance_ft": float(
                readout["batch4_map_tvt_distance_ft"].max()
            ),
        },
        "posterior_cell_diagnostic_only": {
            "mean_abs_error": float(np.mean(posterior_cell)),
            "p99_abs_error": float(np.quantile(posterior_cell, 0.99)),
            "max_abs_error": float(np.max(posterior_cell)),
            "legacy_exp347_1e6_check": bool(
                float(np.max(posterior_cell)) <= 1e-6
            ),
            "promotion_gate": False,
        },
        "rowwise_total_variation_diagnostic_only": {
            "mean": float(readout["row_total_variation"].mean()),
            "p95": float(readout["row_total_variation"].quantile(0.95)),
            "p99": float(readout["row_total_variation"].quantile(0.99)),
            "max": float(readout["row_total_variation"].max()),
        },
        "batch1_diagnostic_only": {
            "posterior_mean_tvt_rmse_ft": float(
                np.sqrt(
                    np.mean(
                        readout["batch1_minus_scalar_tvt_ft"].to_numpy(
                            dtype=np.float64
                        )
                        ** 2
                    )
                )
            ),
            "posterior_mean_tvt_max_abs_ft": float(
                np.abs(readout["batch1_minus_scalar_tvt_ft"]).max()
            ),
            "map_agreement_rate": float(
                readout["batch1_map_state_equal"].mean()
            ),
        },
        "fp64_diagnostic_only": {
            "rows": int(valid_fp64.sum()),
            "scalar_fp32_vs_fp64_tvt_rmse_ft": float(
                np.sqrt(
                    np.mean(
                        readout.loc[
                            valid_fp64, "scalar_fp32_minus_fp64_tvt_ft"
                        ].to_numpy(dtype=np.float64)
                        ** 2
                    )
                )
            ),
            "batch4_fp32_vs_fp64_tvt_rmse_ft": float(
                np.sqrt(
                    np.mean(
                        readout.loc[
                            valid_fp64, "batch4_fp32_minus_fp64_tvt_ft"
                        ].to_numpy(dtype=np.float64)
                        ** 2
                    )
                )
            ),
        },
    }
    return readout, diagnostics


def evaluate_practical_gate(
    diagnostics: Mapping[str, Any],
    objective: Mapping[str, Any],
    modes: Sequence[Mapping[str, Any]],
    *,
    outer_valid_truth_access_count: int,
    stage_a_model_count: int,
    peak_gpu_memory_gb: float,
    audit_runtime_hours: float,
    config: Mapping[str, Any],
    runtime_measurements: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    limits = get_nested(
        config, "validation.stage0.required_checks", {}
    ) or {}
    tvt = diagnostics["posterior_mean_tvt"]
    marginal_map = diagnostics["marginal_map"]
    row_sum_max = 0.0
    finite = bool(objective["finite"])
    invalid_posterior = 0.0
    for mode in modes:
        finite = finite and bool(mode["finite"])
        invalid_posterior = max(
            invalid_posterior,
            float(mode["invalid_posterior_max_abs"]),
        )
        for posterior in mode["posteriors"]:
            row_sum_max = max(
                row_sum_max,
                float(
                    np.max(
                        np.abs(
                            np.asarray(posterior, dtype=np.float64).sum(axis=1)
                            - 1.0
                        )
                    )
                ),
            )
        finite = finite and all(
            math.isfinite(float(row["seconds"]))
            for row in mode["measurements"]
        )
    finite = finite and all(
        math.isfinite(float(row["seconds"]))
        and float(row["seconds"]) >= 0.0
        for row in runtime_measurements
    )
    loss_or_partition = max(
        float(objective["loss_max_abs_error"]),
        float(objective["partition_max_abs_error"]),
    )
    gradient_or_update = max(
        float(objective["gradient_max_abs_error"]),
        float(objective["optimizer_step_max_abs_error"]),
    )
    invalid_value = max(
        invalid_posterior,
        float(objective["invalid_gradient_max_abs"]),
    )
    checks = {
        "posterior_mean_tvt_rmse": (
            float(tvt["rmse_ft"])
            <= float(limits["posterior_mean_tvt_rmse_ft_max"])
        ),
        "posterior_mean_tvt_p99_abs": (
            float(tvt["p99_abs_ft"])
            <= float(limits["posterior_mean_tvt_p99_abs_ft_max"])
        ),
        "posterior_mean_tvt_max_abs": (
            float(tvt["max_abs_ft"])
            <= float(limits["posterior_mean_tvt_max_abs_ft_max"])
        ),
        "marginal_map_state_agreement": (
            float(marginal_map["agreement_rate"])
            >= float(limits["marginal_map_state_agreement_min"])
        ),
        "loss_and_partition": (
            loss_or_partition
            <= float(limits["loss_or_partition_max_abs_error_max"])
        ),
        "gradient_and_adamw_update": (
            gradient_or_update
            <= float(limits["gradient_or_adamw_update_max_abs_error_max"])
        ),
        "posterior_row_sum": (
            row_sum_max
            <= float(limits["posterior_row_sum_max_abs_error_max"])
        ),
        "invalid_posterior_and_gradient": (
            invalid_value
            == float(limits["invalid_posterior_or_gradient_max_abs"])
        ),
        "finite": finite and float(limits["finite_rate_min"]) == 1.0,
        "outer_valid_truth_access_zero": (
            outer_valid_truth_access_count
            <= int(limits["outer_valid_truth_access_count_max"])
        ),
        "stage_a_model_zero": (
            stage_a_model_count <= int(limits["stage_a_model_count_max"])
        ),
        "peak_gpu_memory": (
            peak_gpu_memory_gb
            <= float(limits["peak_gpu_memory_gb_max"])
        ),
        "audit_runtime": (
            audit_runtime_hours
            <= float(limits["audit_runtime_hours_max"])
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": (
            "practical_equivalence_pass_exp347_remains_failed"
            if all(checks.values())
            else "fail_close_without_threshold_dtype_batch_padding_or_kernel_rescue"
        ),
        "loss_or_partition_max_abs_error": loss_or_partition,
        "gradient_or_adamw_update_max_abs_error": gradient_or_update,
        "posterior_row_sum_max_abs_error": row_sum_max,
        "invalid_posterior_or_gradient_max_abs": invalid_value,
        "finite_rate": 1.0 if finite else 0.0,
        "outer_valid_truth_access_count": outer_valid_truth_access_count,
        "stage_a_model_count": stage_a_model_count,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "audit_runtime_hours": audit_runtime_hours,
        "legacy_posterior_cell_1e6_is_gate": False,
    }


def run_stage0_practical_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(config)
    execution_contract = validate_execution_contract(config)
    if validate_selected_stage(config) != "stage0_practical_audit":
        raise ValueError(
            "run_stage0_practical_audit requires selected_stage=stage0_practical_audit"
        )
    device = require_kaggle_gpu(config)
    seed = int(get_nested(config, "reproducibility.seed", 42))
    set_reproducibility(seed)
    torch.cuda.reset_peak_memory_stats(device)
    audit_started = time.perf_counter()
    artifacts = artifact_dir()
    train_dir = resolve_train_dir(config)

    windows, boundaries, parent_evidence = load_parent_fixed16_contract(config)
    windows = windows.sort_values("benchmark_order", kind="mergesort").reset_index(
        drop=True
    )
    keys = window_keys_from_manifests(
        windows, boundaries, role="fit", epoch=0
    )
    if [key.well for key in keys] != windows["well"].astype(str).tolist():
        raise ValueError("fixed16 window order changed while joining boundaries")
    combined_manifest = windows.merge(
        boundaries,
        on=["well", "epoch", "slot", "start_row", "stop_row", "scored_rows"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_boundary"),
    )
    fixed_manifest_path = artifacts / f"{OUTPUT_PREFIX}_fixed16_manifest.csv"
    combined_manifest.to_csv(fixed_manifest_path, index=False)

    _, views, states, input_manifest = prepare_audit_windows(
        keys, train_dir, config
    )
    input_manifest["parent_evidence"] = parent_evidence
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.json"
    write_json(input_manifest_path, input_manifest)

    model = PrefixConditionedUnary(config).to(device)
    frozen_unaries, unary_manifest, unary_measurements = freeze_unaries_once(
        model, views, states, device
    )
    unary_manifest_path = artifacts / f"{OUTPUT_PREFIX}_frozen_unary_manifest.json"
    write_json(unary_manifest_path, unary_manifest)

    scalar = decode_frozen_unaries(
        frozen_unaries,
        states,
        config,
        device,
        "scalar_fp32_reference",
    )
    batch1 = decode_frozen_unaries(
        frozen_unaries,
        states,
        config,
        device,
        "batched_fp32_batch1",
    )
    batch4 = decode_frozen_unaries(
        frozen_unaries,
        states,
        config,
        device,
        "batched_fp32_batch4_production",
    )
    fp64 = decode_frozen_unaries(
        frozen_unaries,
        states,
        config,
        device,
        "scalar_fp64_first4_diagnostic",
    )

    # Truth is loaded only after the input contract and every unary tensor are frozen.
    targets = load_audit_truths(keys, train_dir, states)
    objective, objective_measurements = run_frozen_unary_objective_parity(
        frozen_unaries, targets, states, config, device
    )
    readout, diagnostics = build_practical_readout(
        keys, states, scalar, batch1, batch4, fp64
    )
    row_path = artifacts / f"{OUTPUT_PREFIX}_row_comparison.csv"
    disagreement_path = artifacts / f"{OUTPUT_PREFIX}_map_disagreement.csv"
    readout.to_csv(row_path, index=False)
    readout.loc[~readout["batch4_map_state_equal"]].to_csv(
        disagreement_path, index=False
    )

    measurement_rows = [
        *unary_measurements,
        *scalar["measurements"],
        *batch1["measurements"],
        *batch4["measurements"],
        *fp64["measurements"],
        *objective_measurements,
    ]
    measurements = pd.DataFrame(measurement_rows)
    measurement_path = artifacts / f"{OUTPUT_PREFIX}_runtime_measurements.csv"
    measurements.to_csv(measurement_path, index=False)
    padding_frames = [
        value["padding"]
        for value in (batch1, batch4)
        if not value["padding"].empty
    ]
    padding = pd.concat(padding_frames, ignore_index=True)
    padding_path = artifacts / f"{OUTPUT_PREFIX}_padding_manifest.csv"
    padding.to_csv(padding_path, index=False)

    _cuda_sync(device)
    runtime_hours = (time.perf_counter() - audit_started) / 3600.0
    peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    gate = evaluate_practical_gate(
        diagnostics,
        objective,
        (scalar, batch1, batch4, fp64),
        outer_valid_truth_access_count=0,
        stage_a_model_count=0,
        peak_gpu_memory_gb=peak_memory_gb,
        audit_runtime_hours=runtime_hours,
        config=config,
        runtime_measurements=measurement_rows,
    )
    mode_manifests = {
        value["mode"]: {
            "posterior_sha256": value["posterior_sha256"],
            "readout_sha256": value["readout_sha256"],
            "partition_sha256": array_content_sha256(value["partitions"]),
            "finite": value["finite"],
            "invalid_posterior_max_abs": value[
                "invalid_posterior_max_abs"
            ],
        }
        for value in (scalar, batch1, batch4, fp64)
    }
    output_paths = {
        "input_manifest": input_manifest_path,
        "fixed16_manifest": fixed_manifest_path,
        "frozen_unary_manifest": unary_manifest_path,
        "row_comparison": row_path,
        "map_disagreement": disagreement_path,
        "runtime_measurements": measurement_path,
        "padding_manifest": padding_path,
    }
    report = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_practical_numerical_equivalence_audit",
        "status": "passed" if gate["passed"] else "failed_closed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scientific_contract": scientific_contract,
        "execution_contract": execution_contract,
        "fixed_window_count": 16,
        "fp64_diagnostic_window_count": 4,
        "temporary_neural_model_count": 1,
        "persisted_model_count": 0,
        "trained_fold_count": 0,
        "parent_or_control_retraining": 0,
        "truth_loaded_before_unary_freeze": False,
        "outer_valid_truth_access_count": 0,
        "diagnostics": diagnostics,
        "objective_parity": objective,
        "gate": gate,
        "mode_manifests": mode_manifests,
        "parent_evidence": parent_evidence,
        "artifact_sha256": {
            name: sha256_path(path) for name, path in output_paths.items()
        },
        "exp347_status_changed": False,
        "stage_a_authorized": False,
        "inference_authorized": False,
        "submission_authorized": False,
    }
    report_path = artifacts / f"{OUTPUT_PREFIX}_stage0_report.json"
    write_json(report_path, report)
    report_sha = sha256_path(report_path)
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": report["status"],
        "route": get_nested(config, "experiment.route"),
        "stage": report["stage"],
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "stage0_gate_passed": gate["passed"],
        "stage0_report_sha256": report_sha,
        "posterior_mean_tvt": diagnostics["posterior_mean_tvt"],
        "marginal_map": diagnostics["marginal_map"],
        "objective_parity": objective,
        "gate": gate,
        "deterministic_anchor": False,
        "artifacts_generated": True,
    }
    write_json(metrics_output_path(), metrics_payload)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": report["status"],
        "stage0_gate_passed": gate["passed"],
        "decision": gate["decision"],
        "posterior_mean_tvt": diagnostics["posterior_mean_tvt"],
        "marginal_map": diagnostics["marginal_map"],
        "legacy_posterior_cell_diagnostic": diagnostics[
            "posterior_cell_diagnostic_only"
        ],
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "exp347_remains_terminal_failed": True,
        "stage_a_authorized": False,
    }
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Freeze-first outer-valid decoding, readout, and Stage A gates

# %%
def load_exp209_baseline(
    config: Mapping[str, Any], valid_wells: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp209_baseline_cache", {}) or {}
    path = resolve_existing_path(spec.get("candidates", []), str(spec.get("filename")))
    actual = sha256_gzip_decompressed(path)
    expected = str(spec.get("expected_decompressed_sha256"))
    if actual != expected:
        raise ValueError(f"exp209 baseline decompressed SHA mismatch: {actual} != {expected}")
    frame = pd.read_csv(path, usecols=["id", "well", str(spec["prediction_column"])])
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame = frame.loc[frame["well"].isin(set(valid_wells))].copy()
    frame = frame.rename(columns={str(spec["prediction_column"]): "exp209_prediction"})
    if frame["id"].duplicated().any():
        raise ValueError("exp209 baseline has duplicate ids")
    return frame, {
        "path": str(path),
        "decompressed_content_sha256": actual,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "truth_columns_loaded": 0,
    }


def save_model_checkpoint(
    model: Any,
    config: Mapping[str, Any],
    artifacts: Path,
    training_meta: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    model_path = artifacts / f"{OUTPUT_PREFIX}_fold0_model.pt"
    torch.save(
        {
            "experiment": EXPERIMENT_NAME,
            "fold": STAGE_A_FOLD,
            "seed": int(get_nested(config, "reproducibility.seed", 42)),
            "model_state_dict": model.state_dict(),
            "model_config": get_nested(config, "model"),
            "training_meta": dict(training_meta),
        },
        model_path,
    )
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "fold": STAGE_A_FOLD,
        "architecture_count": 1,
        "seed_count": 1,
        "neural_model_count": 1,
        "lightgbm_config_count": 0,
        "booster_count": 0,
        "pf_beam_well_runs": 0,
        "parent_control_retraining": False,
        "model_file": model_path.name,
        "model_sha256": sha256_path(model_path),
        "training_meta": dict(training_meta),
    }
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_model_manifest.json"
    write_json(manifest_path, manifest)
    return model_path, manifest_path, manifest


def freeze_outer_valid_predictions(
    model: Any,
    valid_wells: Sequence[str],
    train_dir: Path,
    config: Mapping[str, Any],
    device: Any,
    artifacts: Path,
    model_manifest: Mapping[str, Any],
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, dict[str, Any], Path]:
    baseline, baseline_manifest = load_exp209_baseline(config, valid_wells)
    temp_root = Path("/tmp") / f"{OUTPUT_PREFIX}_posterior_freeze"
    temp_root.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    content_rows: list[dict[str, Any]] = []
    model.eval()
    length_rows = []
    for well in sorted(valid_wells):
        tvt_input = pd.read_csv(
            train_dir / f"{well}__horizontal_well.csv", usecols=["TVT_input"]
        )["TVT_input"].to_numpy(np.float64)
        length_rows.append((len(tvt_input) - prefix_end_index(tvt_input) - 1, well))
    ordered_wells = [well for _, well in sorted(length_rows)]
    for batch_start in range(0, len(ordered_wells), 4):
        well_batch = ordered_wells[batch_start : batch_start + 4]
        print(
            f"[valid batch {batch_start // 4 + 1}] freeze wells={well_batch}",
            flush=True,
        )
        items = [load_well_input(well, train_dir) for well in well_batch]
        views = [
            prepare_view(item, item.tvt_input, config, view_name="official")
            for item in items
        ]
        shuffled_views = [
            prepare_view(
                item,
                item.tvt_input,
                config,
                view_name="official_circular_shuffle",
                typewell_control="shuffle",
            )
            for item in items
        ]
        with torch.no_grad():
            real_outputs = [model_unary(model, view, device) for view in views]
            shuffled_outputs = [
                model_unary(model, view, device) for view in shuffled_views
            ]
            real_unaries = [value[0] for value in real_outputs]
            shuffled_unaries = [value[0] for value in shuffled_outputs]
            geometry_unaries = [torch.zeros_like(value) for value in real_unaries]
            real_results = decode_unary_batch(
                real_unaries, views, config, compute_viterbi=True
            )
            shuffled_results = decode_unary_batch(
                shuffled_unaries, shuffled_views, config, compute_viterbi=False
            )
            geometry_results = decode_unary_batch(
                geometry_unaries, views, config, compute_viterbi=False
            )
        for batch_offset, values in enumerate(
            zip(
                well_batch,
                items,
                views,
                real_outputs,
                shuffled_outputs,
                real_unaries,
                shuffled_unaries,
                geometry_unaries,
                real_results,
                shuffled_results,
                geometry_results,
                strict=True,
            )
        ):
            (
                well,
                item,
                view,
                real_output,
                shuffled_output,
                real_unary,
                shuffled_unary,
                geometry_unary,
                real,
                shuffled,
                geometry,
            ) = values
            suffix = view.state.suffix_index
            row_id = np.asarray([f"{well}_{int(index)}" for index in suffix])
            frame = pd.DataFrame(
                {
                    "id": row_id,
                    "well": well,
                    "row_index": suffix,
                    "md_since": item.md[suffix] - item.md[view.state.prefix_end],
                    "real_prediction": real.prediction,
                    "shuffle_prediction": shuffled.prediction,
                    "geometry_prediction": geometry.prediction,
                    "marginal_map_prediction": real.marginal_map,
                    "viterbi_prediction": real.viterbi,
                    "posterior_std": real.posterior_std,
                    "posterior_entropy": real.entropy,
                    "grid_edge_mass": real.edge_mass,
                    "grid_min": float(view.state.grid[0]),
                    "grid_max": float(view.state.grid[-1]),
                    "prefix_end": int(view.state.prefix_end),
                    "last_known_tvt": float(view.state.last_known_tvt),
                    "real_temperature": real_output[1],
                    "shuffle_temperature": shuffled_output[1],
                }
            )
            prediction_frames.append(frame)
            posterior_path = temp_root / f"{well}.npz"
            np.savez_compressed(
                posterior_path,
                grid=view.state.grid.astype(np.float32),
                suffix_index=suffix.astype(np.int32),
                real_posterior=real.posterior,
                shuffle_posterior=shuffled.posterior,
            )
            full_prediction = item.tvt_input.copy()
            full_prediction[suffix] = real.prediction
            prefix_clamp_error = float(
                np.max(
                    np.abs(
                        full_prediction[: view.state.prefix_end + 1]
                        - item.tvt_input[: view.state.prefix_end + 1]
                    )
                )
            )
            content_rows.append(
                {
                    "well": well,
                    "decode_batch_id": batch_start // 4,
                    "decode_batch_offset": batch_offset,
                    "rows": len(suffix),
                    "grid_rows": len(view.state.grid),
                    "real_unary_sha256": array_content_sha256(real_unary.cpu().numpy()),
                    "shuffle_unary_sha256": array_content_sha256(
                        shuffled_unary.cpu().numpy()
                    ),
                    "geometry_unary_sha256": array_content_sha256(
                        geometry_unary.cpu().numpy()
                    ),
                    "real_posterior_sha256": array_content_sha256(real.posterior),
                    "shuffle_posterior_sha256": array_content_sha256(
                        shuffled.posterior
                    ),
                    "row_identity_sha256": array_content_sha256(
                        suffix.astype(np.int64)
                    ),
                    "prefix_clamp_max_abs_error_ft": prefix_clamp_error,
                    "posterior_temp_path": str(posterior_path),
                    "truth_loaded_before_freeze": False,
                }
            )
    frozen = pd.concat(prediction_frames, ignore_index=True)
    frozen = frozen.merge(baseline, on=["id", "well"], how="left", validate="one_to_one")
    if frozen["exp209_prediction"].isna().any():
        raise ValueError("exp209 baseline does not cover every fold-0 validation row")
    frozen = frozen.sort_values(["well", "row_index"], kind="mergesort").reset_index(drop=True)
    frozen_path = artifacts / f"{OUTPUT_PREFIX}_frozen_predictions.csv.gz"
    write_stable_gzip_csv(frozen, frozen_path)
    content_manifest = pd.DataFrame(content_rows).sort_values("well", kind="mergesort")
    content_path = artifacts / f"{OUTPUT_PREFIX}_emission_posterior_manifest.csv"
    content_manifest.drop(columns=["posterior_temp_path"]).to_csv(content_path, index=False)
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "outer_valid_before_truth_read",
        "model_sha256": model_manifest["model_sha256"],
        "frozen_prediction_gzip_sha256": sha256_path(frozen_path),
        "frozen_prediction_decompressed_sha256": sha256_gzip_decompressed(frozen_path),
        "emission_posterior_manifest_sha256": sha256_path(content_path),
        "input_manifest_sha256": sha256_path(input_path),
        "exp209_baseline": baseline_manifest,
        "rows": len(frozen),
        "wells": int(frozen["well"].nunique()),
        "outer_valid_truth_access_count_before_freeze": 0,
        "truth_loaded": False,
        "horizontal_source_count_per_well": 1,
        "forbidden_neighbor_sources": 0,
    }
    freeze_path = artifacts / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    write_json(freeze_path, freeze_manifest)
    return frozen, content_manifest, frozen_path, freeze_path, freeze_manifest, input_path


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def load_hidden_like_roles(config: Mapping[str, Any]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment", {}) or {}
    filename = str(spec.get("filename", ""))
    if not filename:
        raise ValueError("hidden-like assignment filename is required")
    path = resolve_existing_path(list(spec.get("candidates", [])), filename=filename)
    frame = pd.read_csv(path, dtype={"well_id": str})
    if "well_id" not in frame.columns:
        raise ValueError("hidden-like assignment is missing well_id")
    groups: dict[str, set[str]] = {}
    for name, role_column in (spec.get("valid_role_columns", {}) or {}).items():
        if role_column not in frame.columns:
            raise ValueError(f"hidden-like assignment is missing {role_column}")
        groups[str(name)] = set(
            frame.loc[frame[role_column].astype(str) == "valid", "well_id"].astype(str)
        )
    manifest = {
        "path": str(path),
        "sha256": sha256_path(path),
        "rows": len(frame),
        "groups": {name: len(wells) for name, wells in groups.items()},
        "loaded_after_prediction_freeze": True,
    }
    return groups, manifest


def subgroup_metric_table(
    readout: pd.DataFrame,
    hidden_like_groups: Mapping[str, set[str]],
) -> pd.DataFrame:
    masks: dict[str, np.ndarray] = {
        "distance_1000_plus": readout["md_since"].to_numpy(np.float64) >= 1000.0,
    }
    wells = readout["well"].astype(str).to_numpy()
    for name, valid_wells in hidden_like_groups.items():
        masks[str(name)] = np.isin(wells, sorted(valid_wells))
    candidates = {
        "real_gr": "real_prediction",
        "circular_shuffle": "shuffle_prediction",
        "geometry_only": "geometry_prediction",
        "exp209": "exp209_prediction",
    }
    rows: list[dict[str, Any]] = []
    truth = readout["truth_tvt"].to_numpy(np.float64)
    for subgroup, mask in masks.items():
        for candidate, column in candidates.items():
            rows.append(
                {
                    "subgroup": subgroup,
                    "candidate": candidate,
                    "rows": int(mask.sum()),
                    "wells": int(readout.loc[mask, "well"].nunique()),
                    "rmse": rmse(truth[mask], readout.loc[mask, column])
                    if mask.any()
                    else None,
                }
            )
    return pd.DataFrame(rows)


def post_freeze_readout(
    frozen: pd.DataFrame,
    content_manifest: pd.DataFrame,
    valid_wells: Sequence[str],
    train_dir: Path,
    config: Mapping[str, Any],
    runtime_seconds: float,
    peak_gpu_memory_gb: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    truth_frames: list[pd.DataFrame] = []
    posterior_metrics: list[pd.DataFrame] = []
    content_by_well = content_manifest.set_index("well")
    for well in sorted(valid_wells):
        truth = load_well_truth(well, train_dir)
        rows = frozen.loc[frozen["well"] == well, "row_index"].to_numpy(np.int64)
        truth_frames.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in rows],
                    "truth_tvt": truth.tvt[rows],
                }
            )
        )
        posterior_path = Path(str(content_by_well.loc[well, "posterior_temp_path"]))
        bundle = np.load(posterior_path)
        grid = bundle["grid"].astype(np.float64)
        real = bundle["real_posterior"].astype(np.float64)
        shuffled = bundle["shuffle_posterior"].astype(np.float64)
        target = truth.tvt[rows]
        index = nearest_grid_indices(grid, target)
        row_number = np.arange(len(rows))
        in_grid = (target >= grid[0]) & (target <= grid[-1])
        real_probability = np.clip(real[row_number, index], 1e-12, 1.0)
        shuffle_probability = np.clip(shuffled[row_number, index], 1e-12, 1.0)
        within10 = np.abs(grid[None, :] - target[:, None]) <= 10.0
        within20 = np.abs(grid[None, :] - target[:, None]) <= 20.0
        posterior_metrics.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in rows],
                    "target_in_grid": in_grid,
                    "real_true_state_nll": -np.log(real_probability),
                    "shuffle_true_state_nll": -np.log(shuffle_probability),
                    "real_within10_mass": np.sum(real * within10, axis=1),
                    "shuffle_within10_mass": np.sum(shuffled * within10, axis=1),
                    "real_within20_mass": np.sum(real * within20, axis=1),
                    "shuffle_within20_mass": np.sum(shuffled * within20, axis=1),
                }
            )
        )
    truth_frame = pd.concat(truth_frames, ignore_index=True)
    posterior_frame = pd.concat(posterior_metrics, ignore_index=True)
    readout = frozen.merge(truth_frame, on="id", validate="one_to_one")
    readout = readout.merge(posterior_frame, on="id", validate="one_to_one")
    hidden_like_groups, hidden_like_manifest = load_hidden_like_roles(config)
    subgroup_metrics = subgroup_metric_table(readout, hidden_like_groups)
    numeric_predictions = readout[
        ["real_prediction", "shuffle_prediction", "geometry_prediction", "exp209_prediction"]
    ].to_numpy(np.float64)
    finite_coverage = float(np.isfinite(numeric_predictions).all(axis=1).mean())
    by_well_rows = []
    for well, group in readout.groupby("well", sort=True):
        truth = group["truth_tvt"].to_numpy(np.float64)
        by_well_rows.append(
            {
                "well": well,
                "rows": len(group),
                "real_rmse": rmse(truth, group["real_prediction"]),
                "shuffle_rmse": rmse(truth, group["shuffle_prediction"]),
                "geometry_rmse": rmse(truth, group["geometry_prediction"]),
                "exp209_rmse": rmse(truth, group["exp209_prediction"]),
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    by_well["real_minus_exp209_rmse"] = by_well["real_rmse"] - by_well["exp209_rmse"]
    truth = readout["truth_tvt"].to_numpy(np.float64)
    metrics = {
        "rows": len(readout),
        "wells": int(readout["well"].nunique()),
        "finite_prediction_coverage": finite_coverage,
        "target_in_grid_rate": float(readout["target_in_grid"].mean()),
        "prefix_clamp_max_abs_error_ft": float(
            content_manifest["prefix_clamp_max_abs_error_ft"].max()
        ),
        "real_rmse": rmse(truth, readout["real_prediction"]),
        "shuffle_rmse": rmse(truth, readout["shuffle_prediction"]),
        "geometry_rmse": rmse(truth, readout["geometry_prediction"]),
        "exp209_rmse": rmse(truth, readout["exp209_prediction"]),
        "real_true_state_nll": float(readout["real_true_state_nll"].mean()),
        "shuffle_true_state_nll": float(readout["shuffle_true_state_nll"].mean()),
        "real_within10_mass": float(readout["real_within10_mass"].mean()),
        "shuffle_within10_mass": float(readout["shuffle_within10_mass"].mean()),
        "real_within20_mass": float(readout["real_within20_mass"].mean()),
        "shuffle_within20_mass": float(readout["shuffle_within20_mass"].mean()),
        "real_well_rmse_p95": float(by_well["real_rmse"].quantile(0.95)),
        "exp209_well_rmse_p95": float(by_well["exp209_rmse"].quantile(0.95)),
        "maximum_well_regression_vs_exp209_ft": float(
            by_well["real_minus_exp209_rmse"].max()
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_hours": runtime_seconds / 3600.0,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "subgroup_rmse": {
            f"{row.subgroup}__{row.candidate}": row.rmse
            for row in subgroup_metrics.itertuples(index=False)
        },
        "hidden_like_assignment": hidden_like_manifest,
    }
    stage = get_nested(config, "validation.stage_a_pass", {}) or {}
    checks = {
        "finite_prediction": finite_coverage >= 1.0,
        "target_in_grid": metrics["target_in_grid_rate"]
        >= float(stage["minimum_target_in_grid_rate"]),
        "prefix_clamp": metrics["prefix_clamp_max_abs_error_ft"]
        <= float(stage["maximum_prefix_clamp_abs_error_ft"]),
        "real_nll_vs_shuffle": metrics["shuffle_true_state_nll"]
        - metrics["real_true_state_nll"]
        >= float(stage["minimum_real_nll_gain_vs_circular_shuffle_nats_per_token"]),
        "real_within10_vs_shuffle": metrics["real_within10_mass"]
        - metrics["shuffle_within10_mass"]
        >= float(stage["minimum_real_within10_mass_gain_vs_circular_shuffle"]),
        "real_rmse_vs_geometry": metrics["geometry_rmse"] - metrics["real_rmse"]
        >= float(stage["minimum_real_rmse_gain_vs_geometry_only_ft"]),
        "real_rmse_vs_exp209": metrics["exp209_rmse"] - metrics["real_rmse"]
        >= float(stage["minimum_real_rmse_gain_vs_exp209_ft"]),
        "well_p95_non_regression": metrics["real_well_rmse_p95"]
        <= metrics["exp209_well_rmse_p95"],
        "worst_well_regression": metrics["maximum_well_regression_vs_exp209_ft"]
        <= float(stage["maximum_worst_well_regression_vs_exp209_ft"]),
        "peak_gpu_memory": peak_gpu_memory_gb
        <= float(stage["maximum_peak_gpu_memory_gb"]),
        "fold_runtime": runtime_seconds / 3600.0
        <= float(stage["maximum_fold_runtime_hours"]),
    }
    guard = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "decision": "permit_separate_stage_b_approval"
        if all(checks.values())
        else "close_stage_b_without_exp347_rescue_grid",
    }
    return readout, by_well, subgroup_metrics, metrics, guard, hidden_like_manifest


# %% [markdown]
# ## 11. Stage A orchestration and generated artifacts

# %%
def run_stage_a(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    cost = validate_stage_a_cost_contract(config)
    if validate_selected_stage(config) != "stage_a_fold0":
        raise ValueError("run_stage_a requires selected_stage=stage_a_fold0")
    device = require_kaggle_gpu(config)
    seed = int(get_nested(config, "reproducibility.seed", 42))
    set_reproducibility(seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    artifacts = artifact_dir()
    train_dir = resolve_train_dir(config)
    wells = list_paired_wells(train_dir)
    fold_map = build_fold_map(wells, int(get_nested(config, "validation.n_folds", 5)))
    outer_train, outer_valid = split_stage_a_wells(fold_map)
    fit_wells, early_stop_wells = split_early_stop_wells(outer_train, config)
    fold_map["stage_a_role"] = np.where(
        fold_map["well"].isin(outer_valid),
        "outer_valid",
        np.where(
            fold_map["well"].isin(early_stop_wells),
            "outer_train_early_stop",
            "outer_train_fit",
        ),
    )
    fold_path = artifacts / f"{OUTPUT_PREFIX}_fold_map.csv"
    fold_map.to_csv(fold_path, index=False)
    input_manifest = build_input_manifest(fold_map, train_dir)
    input_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_path, index=False)
    maximum_fit_wells = int(
        get_nested(config, "model.training.windows.maximum_fit_wells_fold0", 556)
    )
    if len(fit_wells) > maximum_fit_wells:
        raise ValueError("fold-0 fit well count exceeds the fixed exp347 contract")
    roles = {
        well: "fit" if well in set(fit_wells) else "early_stop"
        for well in outer_train
    }
    schedule = build_window_schedule_manifest(outer_train, train_dir, roles, config)
    schedule_path = artifacts / f"{OUTPUT_PREFIX}_window_schedule_manifest.csv"
    schedule.to_csv(schedule_path, index=False)
    schedule_sha = sha256_path(schedule_path)
    boundary = build_teacher_boundary_manifest(schedule, train_dir, config)
    boundary_path = artifacts / f"{OUTPUT_PREFIX}_teacher_boundary_manifest.csv"
    boundary.to_csv(boundary_path, index=False)
    boundary_sha = sha256_path(boundary_path)
    model, history, training_meta = train_fold0_model(
        config, train_dir, schedule, boundary, device
    )
    history_path = artifacts / f"{OUTPUT_PREFIX}_training_history.csv"
    history.to_csv(history_path, index=False)
    model_path, model_manifest_path, model_manifest = save_model_checkpoint(
        model, config, artifacts, training_meta
    )
    frozen, content_manifest, frozen_path, freeze_path, freeze_manifest, input_path = (
        freeze_outer_valid_predictions(
            model,
            outer_valid,
            train_dir,
            config,
            device,
            artifacts,
            model_manifest,
            input_path,
        )
    )
    runtime_seconds = time.time() - started
    peak_gpu_memory_gb = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    (
        readout,
        by_well,
        subgroup_metrics,
        stage_metrics,
        guard,
        hidden_like_manifest,
    ) = post_freeze_readout(
        frozen,
        content_manifest,
        outer_valid,
        train_dir,
        config,
        runtime_seconds,
        peak_gpu_memory_gb,
    )
    readout_path = artifacts / f"{OUTPUT_PREFIX}_validation_readout.csv.gz"
    write_stable_gzip_csv(readout, readout_path)
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv"
    by_well.to_csv(by_well_path, index=False)
    subgroup_path = artifacts / f"{OUTPUT_PREFIX}_subgroup_metrics.csv"
    subgroup_metrics.to_csv(subgroup_path, index=False)
    stage_metrics_path = artifacts / f"{OUTPUT_PREFIX}_stage_a_metrics.json"
    write_json(stage_metrics_path, {"metrics": stage_metrics, "guard": guard})
    status = (
        "stage_a_passed_waiting_stage_b_approval"
        if guard["passed"]
        else "stage_a_failed_branch_closed"
    )
    output_paths = {
        "fold_map": fold_path,
        "window_schedule_manifest": schedule_path,
        "teacher_boundary_manifest": boundary_path,
        "training_history": history_path,
        "model": model_path,
        "model_manifest": model_manifest_path,
        "frozen_predictions": frozen_path,
        "freeze_manifest": freeze_path,
        "input_manifest": input_path,
        "validation_readout": readout_path,
        "by_well_metrics": by_well_path,
        "subgroup_metrics": subgroup_path,
        "stage_a_metrics": stage_metrics_path,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "stage": "A_fold0",
        "cost_contract": cost,
        "fold_counts": {
            "outer_train_fit": len(fit_wells),
            "outer_train_early_stop": len(early_stop_wells),
            "outer_valid": len(outer_valid),
        },
        "training": training_meta,
        "window_contract": {
            "schedule_frozen_before_teacher_truth_load": True,
            "schedule_sha256": schedule_sha,
            "teacher_boundary_sha256": boundary_sha,
            "fit_max_active_windows_per_epoch": int(
                schedule.loc[schedule["active"] & (schedule["role"] == "fit")]
                .groupby("epoch")
                .size()
                .max()
            ),
            "fit_max_scored_positions_per_epoch": int(
                schedule.loc[schedule["active"] & (schedule["role"] == "fit")]
                .groupby("epoch")["scored_rows"]
                .sum()
                .max()
            ),
            "teacher_boundary_encoder_access_count": 0,
        },
        "metrics": stage_metrics,
        "guard": guard,
        "truth_freeze": {
            **freeze_manifest,
            "truth_loaded_after_global_freeze": True,
        },
        "hidden_like_assignment": hidden_like_manifest,
        "outputs": {key: path.name for key, path in output_paths.items()},
        "file_sha256": {key: sha256_path(path) for key, path in output_paths.items()},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": guard["decision"],
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "ensemble",
        "metric": get_nested(config, "validation.metric"),
        "cv": stage_metrics["real_rmse"],
        "public_lb": None,
        "private_lb": None,
        "stage_a": {"metrics": stage_metrics, "guard": guard},
        "execution": cost,
        "inference": False,
        "submission": False,
    }
    write_json(metrics_output_path(), metrics_payload)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 12. Setup, override, and contract preview

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    SELECTED_STAGE = validate_selected_stage(CONFIG)
    if SELECTED_STAGE == "stage0_practical_audit":
        COST_CONTRACT = validate_execution_contract(CONFIG)
    elif SELECTED_STAGE == "stage_a_fold0":
        COST_CONTRACT = validate_stage_a_cost_contract(CONFIG)
    else:
        raise RuntimeError("exp393 requires an explicitly authorized Kaggle stage")
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "implementation_approved": get_nested(
                    CONFIG, "execution.implementation_approved"
                ),
                "kaggle_push_approved": get_nested(
                    CONFIG, "execution.kaggle_push_approved"
                ),
                "selected_stage": SELECTED_STAGE,
                "scientific_contract": SCIENTIFIC_CONTRACT,
                "execution_contract": COST_CONTRACT,
                "exp347_status_changed": False,
                "stage0_gate_passed": get_nested(
                    CONFIG, "execution.stage0_gate.passed"
                ),
                "stage0_failure_user_override": get_nested(
                    CONFIG, "execution.stage_a_user_override.approved"
                ),
                "stage_a_authorized": get_nested(
                    CONFIG, "execution.stage_a_gpu_approved"
                ),
                "inference_authorized": False,
                "submission_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 13. Run only the explicitly authorized Kaggle GPU stage

# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    if SELECTED_STAGE == "stage0_practical_audit":
        STAGE_SUMMARY = run_stage0_practical_audit(CONFIG)
    elif SELECTED_STAGE == "stage_a_fold0":
        STAGE_SUMMARY = run_stage_a(CONFIG)
    else:
        raise RuntimeError("exp393 selected stage is not executable")
