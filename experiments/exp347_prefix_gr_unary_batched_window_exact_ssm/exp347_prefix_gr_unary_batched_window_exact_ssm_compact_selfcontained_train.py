# %% [markdown]
# # exp347 prefix GR unary batched-window exact SSM train
#
# Stage 0 benchmarks a fixed 16-window sample before any full fold is allowed.
# Stage A, when separately approved after a passing benchmark, trains exactly
# one fold-0 neural GR unary with the exp295 structured objective restricted to
# deterministic non-overlapping 256-row windows.  Interior truth is confined to
# loss-only teacher boundaries and never enters the encoder.
# Each validation well is decoded from its own horizontal GR, its paired Type
# Well GR, and its known TVT prefix.  The exp209 state grid and transition are
# fixed.  Validation truth is first read only after the model, unaries,
# posterior marginals, controls, row identities, and SHA manifests are frozen.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Scientific, execution-cost, and leakage contract guards
# 4. Complete-well folds, mask-first loading, and frozen window manifests
# 5. Robust GR preprocessing and prefix-context helpers
# 6. Prefix-conditioned multi-scale neural emission
# 7. Scalar and batched exp209 exact forward-backward helpers
# 8. Four-window structured training and outer-train early stopping
# 9. Scalar/batch parity and fixed 16-window T4 microbenchmark
# 10. Freeze-first outer-valid decoding, readout, and Stage A gates
# 11. Stage A orchestration and generated artifacts
# 12. Setup and contract preview
# 13. Run only the separately authorized Kaggle GPU stage

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


EXPERIMENT_NAME = "exp347_prefix_gr_unary_batched_window_exact_ssm"
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


EXECUTE_NOTEBOOK = os.environ.get("EXP347_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
        raise RuntimeError("exp347 Kaggle GPU push has not been separately approved")
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError("exp347 requires a Kaggle CUDA runtime")
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
        "validation.n_folds": 5,
        "validation.sample_unit": "complete_well",
        "model.sample_unit": "well_window",
        "model.output": "tvt_posterior_mean",
        "model.inference_neighbor_well_data": False,
        "model.candidate_bank": "none",
        "model.test_time_gradient_updates": False,
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
        "model.training.dataloader_workers": 0,
        "data.hidden_like_assignment.valid_role_columns": {
            "hidden_like_spatial": "verification_like_spatial_role",
            "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
        },
        "validation.stage0.fixed_window_count": 16,
        "validation.stage0.scalar_parity_window_count": 4,
        "validation.stage0.conservative_throughput_quantile": 0.10,
    }
    changed = [
        f"{key}={get_nested(config, key)!r} expected {value!r}"
        for key, value in expected.items()
        if get_nested(config, key) != value
    ]
    if changed:
        raise ValueError("exp347 locked scientific contract changed: " + "; ".join(changed))
    inputs = tuple(get_nested(config, "data.horizontal_file_input_columns", []))
    if inputs != HORIZONTAL_INPUT_COLUMNS:
        raise ValueError("horizontal input allowlist must be exactly MD/X/Y/Z/GR/TVT_input")
    forbidden_actions = set(get_nested(config, "model.forbidden", []))
    required_forbidden = {
        "reopen_or_modify_exp332",
        "batch_size_or_padding_bucket_grid",
        "change_window_length_count_or_schedule",
        "change_boundary_objective_loss_weight_or_sigma",
        "change_architecture_band_temperature_view_or_epoch",
        "parent_or_control_retraining",
        "inference_or_submission_before_promotion_and_approval",
    }
    if not required_forbidden.issubset(forbidden_actions):
        raise ValueError(
            "window, leakage, retraining, and inference prohibitions are incomplete"
        )
    controls = tuple(get_nested(config, "model.controls", []))
    if len(controls) != 3 or not {
        "real_gr",
        "stable_within_well_circular_shuffled_typewell_gr_same_trained_model",
        "zero_gr_unary_geometry_only_same_trained_model",
    }.issubset(controls):
        raise ValueError("the three same-model GR attribution controls must remain fixed")
    return {"locked_fields": len(expected), "controls": len(controls)}


def validate_stage_a_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = get_nested(config, "execution.stage_a_plan", {}) or {}
    contract = {
        "fold_indices": list(stage.get("fold_indices", [])),
        "active_architectures": int(stage.get("active_architectures", -1)),
        "seeds": list(stage.get("seeds", [])),
        "neural_model_count": int(stage.get("neural_model_count", -1)),
        "lightgbm_config_count": int(stage.get("lightgbm_config_count", -1)),
        "total_boosters": int(stage.get("total_boosters", -1)),
        "control_model_training": int(stage.get("control_model_training", -1)),
        "pf_beam_well_runs": int(get_nested(config, "execution.current_pf_beam_well_runs", -1)),
        "parent_control_retraining": bool(
            get_nested(config, "execution.control_or_parent_retraining", True)
        ),
    }
    expected = {
        "fold_indices": [0],
        "active_architectures": 1,
        "seeds": [42],
        "neural_model_count": 1,
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
    return contract


def validate_selected_stage(config: Mapping[str, Any]) -> str:
    selected = str(get_nested(config, "execution.selected_stage", "implementation_only"))
    allowed = {"implementation_only", "stage0_microbenchmark", "stage_a_fold0"}
    if selected not in allowed:
        raise ValueError(f"unknown exp347 selected stage: {selected}")
    if selected == "stage0_microbenchmark":
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise ValueError("Stage 0 Kaggle push requires separate user approval")
    if selected == "stage_a_fold0":
        if not bool(get_nested(config, "execution.stage_a_gpu_approved", False)):
            raise ValueError("Stage A GPU training requires separate user approval")
        if not bool(get_nested(config, "execution.stage0_gate.passed", False)):
            raise ValueError("Stage A is blocked until the recorded Stage 0 gate passes")
        if not get_nested(config, "execution.stage0_gate.report_sha256"):
            raise ValueError("Stage A requires the frozen Stage 0 report SHA")
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
# ## 9. Scalar/batch parity and fixed 16-window T4 microbenchmark

# %%
def select_fixed_benchmark_windows(
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    benchmark = get_nested(config, "validation.stage0", {}) or {}
    fixed_count = int(benchmark.get("fixed_window_count", 16))
    quartile_count = int(benchmark.get("suffix_length_quartiles", 4))
    per_quartile = int(benchmark.get("windows_per_quartile", 4))
    if fixed_count != 16 or quartile_count != 4 or per_quartile != 4:
        raise ValueError("exp347 benchmark must remain 4 suffix quartiles x 4 windows")
    selected = schedule.loc[
        schedule["active"] & (schedule["role"] == "fit") & (schedule["epoch"] == 0)
    ].copy()
    selected = selected.sort_values(["well", "slot"], kind="mergesort")
    selected = selected.drop_duplicates("well", keep="first")
    selected = selected.sort_values(
        ["suffix_rows", "well", "slot"], kind="mergesort"
    ).reset_index(drop=True)
    if len(selected) < fixed_count:
        raise ValueError("not enough active fit windows for the fixed benchmark")
    selected["suffix_length_quartile"] = pd.qcut(
        np.arange(len(selected)), quartile_count, labels=False
    ).astype(int)
    chosen = []
    for quartile, group in selected.groupby("suffix_length_quartile", sort=True):
        ranked = group.assign(
            benchmark_hash=group.apply(
                lambda row, fixed_quartile=int(quartile): stable_uint64(
                    EXPERIMENT_NAME,
                    "stage0-benchmark",
                    fixed_quartile,
                    str(row["well"]),
                    int(row["slot"]),
                    int(row["start_row"]),
                ),
                axis=1,
            )
        ).sort_values(["benchmark_hash", "well", "slot"], kind="mergesort")
        if len(ranked) < per_quartile:
            raise ValueError(f"suffix quartile {quartile} has fewer than four windows")
        chosen.append(ranked.head(per_quartile))
    result = pd.concat(chosen, ignore_index=True)
    result = result.sort_values(
        ["suffix_length_quartile", "benchmark_hash"], kind="mergesort"
    ).reset_index(drop=True)
    result["benchmark_order"] = np.arange(len(result), dtype=np.int64)
    if len(result) != fixed_count or result["well"].nunique() != fixed_count:
        raise ValueError("fixed benchmark selection contract failed")
    return result


def _cuda_sync(device: Any) -> None:
    if TORCH_AVAILABLE and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def run_scalar_batch_parity(
    model: Any,
    keys: Sequence[WindowKey],
    train_dir: Path,
    config: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = int(get_nested(config, "validation.stage0.scalar_parity_window_count", 4))
    if required != 4 or len(keys) < required:
        raise ValueError("Stage 0 parity requires the first four fixed windows")
    parity_keys = list(keys[:required])
    views, truths, states = prepare_training_batch(parity_keys, train_dir, config)
    model.eval()
    with torch.no_grad():
        scalar_unaries = [
            model_unary(model, view, device, state.suffix_index)[0]
            for view, state in zip(views, states, strict=True)
        ]
    target_values = [
        np.asarray(truth.tvt[state.suffix_index], dtype=np.float32)
        for truth, state in zip(truths, states, strict=True)
    ]
    target_indices = [
        nearest_grid_indices(state.grid, target)
        for state, target in zip(states, target_values, strict=True)
    ]
    padded_state = build_batched_state_spec(states, required_batch_size=required)
    scalar_parameters = [
        nn.Parameter(unary.detach().clone()) for unary in scalar_unaries
    ]
    batched_parameter = nn.Parameter(
        pad_unary_batch(scalar_unaries, padded_state).detach().clone()
    )
    learning_rate = float(get_nested(config, "model.training.learning_rate", 3e-4))
    weight_decay = float(get_nested(config, "model.training.weight_decay", 1e-4))
    clip_norm = float(get_nested(config, "model.training.gradient_clip_norm", 1.0))
    scalar_optimizer = torch.optim.AdamW(
        scalar_parameters, lr=learning_rate, weight_decay=weight_decay
    )
    batched_optimizer = torch.optim.AdamW(
        [batched_parameter], lr=learning_rate, weight_decay=weight_decay
    )
    scalar_optimizer.zero_grad(set_to_none=True)
    batched_optimizer.zero_grad(set_to_none=True)
    scalar_losses = []
    for unary, target, truth_index, state in zip(
        scalar_parameters, target_values, target_indices, states, strict=True
    ):
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        index_tensor = torch.as_tensor(truth_index, dtype=torch.long, device=device)
        structured = SoftLabelStructuredNLL.apply(
            unary, target_tensor, state, config
        )
        local = F.cross_entropy(unary, index_tensor)
        scalar_losses.append(structured + 0.25 * local)
    scalar_loss = torch.stack(scalar_losses).mean()
    target_batch = pad_target_batch(
        target_values, padded_state, dtype=torch.float32, device=device
    )
    batched_structured = BatchedSoftLabelStructuredNLL.apply(
        batched_parameter, target_batch, padded_state, config
    )
    padded_indices = torch.full(
        padded_state.row_mask.shape, -100, dtype=torch.long, device=device
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
        padded_state.row_mask.sum(axis=1),
        dtype=batched_parameter.dtype,
        device=device,
    )
    batched_local = local_rows.sum(dim=1) / torch.clamp(token_count, min=1.0)
    active = torch.as_tensor(padded_state.active_mask, dtype=torch.bool, device=device)
    batched_loss = (batched_structured + 0.25 * batched_local)[active].mean()
    loss_error = float(torch.abs(scalar_loss.detach() - batched_loss.detach()).cpu())

    scalar_loss.backward()
    batched_loss.backward()
    gradient_error = 0.0
    finite_gradients = True
    for index, (scalar_parameter, state) in enumerate(
        zip(scalar_parameters, states, strict=True)
    ):
        scalar_gradient = scalar_parameter.grad
        if scalar_gradient is None or batched_parameter.grad is None:
            raise ValueError(f"missing unary gradient at parity index {index}")
        batch_gradient = batched_parameter.grad[
            index, : len(state.suffix_index), : len(state.grid)
        ]
        finite_gradients = finite_gradients and bool(
            torch.isfinite(scalar_gradient).all() and torch.isfinite(batch_gradient).all()
        )
        gradient_error = max(
            gradient_error,
            float(torch.max(torch.abs(scalar_gradient - batch_gradient)).detach().cpu()),
        )
    parity_row_mask = torch.as_tensor(
        padded_state.row_mask, dtype=torch.bool, device=device
    )
    parity_position_mask = torch.as_tensor(
        padded_state.position_mask, dtype=torch.bool, device=device
    )
    invalid_gradient = batched_parameter.grad.masked_select(
        ~(parity_row_mask[:, :, None] & parity_position_mask[:, None, :])
    )
    invalid_gradient_max = (
        float(torch.max(torch.abs(invalid_gradient)).detach().cpu())
        if invalid_gradient.numel()
        else 0.0
    )
    finite_gradients = finite_gradients and bool(
        torch.isfinite(batched_parameter.grad).all()
    )
    torch.nn.utils.clip_grad_norm_(scalar_parameters, clip_norm)
    torch.nn.utils.clip_grad_norm_([batched_parameter], clip_norm)
    scalar_optimizer.step()
    batched_optimizer.step()
    update_error = 0.0
    for index, (scalar_parameter, state) in enumerate(
        zip(scalar_parameters, states, strict=True)
    ):
        batch_slice = batched_parameter[
            index, : len(state.suffix_index), : len(state.grid)
        ]
        update_error = max(
            update_error,
            float(torch.max(torch.abs(scalar_parameter - batch_slice)).detach().cpu()),
        )

    with torch.no_grad():
        unary_batch = pad_unary_batch(scalar_unaries, padded_state)
        batch_posterior, batch_partition = batched_exact_forward_backward(
            unary_batch, padded_state, config
        )
        posterior_error = 0.0
        partition_error = 0.0
        finite_posterior = True
        for index, (unary, state) in enumerate(
            zip(scalar_unaries, states, strict=True)
        ):
            scalar_posterior, scalar_partition = exact_forward_backward(
                unary, state, config
            )
            sliced = batch_posterior[
                index, : len(state.suffix_index), : len(state.grid)
            ]
            posterior_error = max(
                posterior_error,
                float(torch.max(torch.abs(scalar_posterior - sliced)).cpu()),
            )
            partition_error = max(
                partition_error,
                float(torch.abs(scalar_partition - batch_partition[index]).cpu()),
            )
            finite_posterior = finite_posterior and bool(
                torch.isfinite(sliced).all() and torch.isfinite(batch_partition[index])
            )
        row_mask = torch.as_tensor(
            padded_state.row_mask, dtype=torch.bool, device=batch_posterior.device
        )
        position_mask = torch.as_tensor(
            padded_state.position_mask,
            dtype=torch.bool,
            device=batch_posterior.device,
        )
        invalid_posterior = batch_posterior.masked_select(
            ~(row_mask[:, :, None] & position_mask[:, None, :])
        )
        invalid_posterior_max = (
            float(torch.max(torch.abs(invalid_posterior)).cpu())
            if invalid_posterior.numel()
            else 0.0
        )

    stage0 = get_nested(config, "validation.stage0", {}) or {}
    loss_posterior_limit = float(stage0["maximum_loss_or_posterior_abs_error"])
    gradient_update_limit = float(stage0["maximum_gradient_or_update_abs_error"])
    checks = {
        "loss": loss_error <= loss_posterior_limit,
        "partition": partition_error <= loss_posterior_limit,
        "posterior": posterior_error <= loss_posterior_limit,
        "gradient": gradient_error <= gradient_update_limit,
        "optimizer_step": update_error <= gradient_update_limit,
        "padding_mask": invalid_posterior_max == 0.0 and invalid_gradient_max == 0.0,
        "finite": finite_gradients and finite_posterior,
    }
    report = {
        "window_count": required,
        "loss_max_abs_error": loss_error,
        "partition_max_abs_error": partition_error,
        "posterior_max_abs_error": posterior_error,
        "gradient_max_abs_error": gradient_error,
        "optimizer_step_max_abs_error": update_error,
        "invalid_posterior_max_abs": invalid_posterior_max,
        "invalid_gradient_max_abs": invalid_gradient_max,
        "finite_rate": 1.0 if finite_gradients and finite_posterior else 0.0,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    return report, batch_padding_manifest(padded_state)


def _positive_rate_quantile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0.0)]
    if not len(array):
        raise ValueError("benchmark produced no finite positive throughput")
    return float(np.quantile(array, quantile))


def project_stage_a_runtime(
    train_measurements: pd.DataFrame,
    window_forward_measurements: pd.DataFrame,
    unary_measurements: pd.DataFrame,
    decode_measurements: pd.DataFrame,
    *,
    fit_window_state_cells_per_epoch: int,
    early_stop_window_state_cells_per_epoch: int,
    valid_unary_positions: int,
    valid_decode_cells: int,
    max_epochs: int,
    conservative_quantile: float = 0.10,
) -> dict[str, Any]:
    if not 0.0 < conservative_quantile < 0.50:
        raise ValueError("conservative throughput quantile must be between zero and 0.5")
    train_rates = train_measurements["state_cells"] / train_measurements["seconds"]
    window_forward_rates = (
        window_forward_measurements["state_cells"]
        / window_forward_measurements["seconds"]
    )
    unary_rates = (
        unary_measurements["unary_positions"] / unary_measurements["seconds"]
    )
    decode_rates = decode_measurements["state_cells"] / decode_measurements["seconds"]
    p50_train = _positive_rate_quantile(train_rates, 0.50)
    conservative_train = _positive_rate_quantile(train_rates, conservative_quantile)
    p50_window_forward = _positive_rate_quantile(window_forward_rates, 0.50)
    conservative_window_forward = _positive_rate_quantile(
        window_forward_rates, conservative_quantile
    )
    p50_unary = _positive_rate_quantile(unary_rates, 0.50)
    conservative_unary = _positive_rate_quantile(unary_rates, conservative_quantile)
    p50_decode = _positive_rate_quantile(decode_rates, 0.50)
    conservative_decode = _positive_rate_quantile(decode_rates, conservative_quantile)

    def seconds(
        train_rate: float,
        window_forward_rate: float,
        unary_rate: float,
        decode_rate: float,
    ) -> float:
        epoch_seconds = (
            fit_window_state_cells_per_epoch / train_rate
            + early_stop_window_state_cells_per_epoch / window_forward_rate
        )
        frozen_eval_seconds = (
            valid_unary_positions / unary_rate + valid_decode_cells / decode_rate
        )
        return float(max_epochs * epoch_seconds + frozen_eval_seconds)

    p50_seconds = seconds(p50_train, p50_window_forward, p50_unary, p50_decode)
    upper_seconds = seconds(
        conservative_train,
        conservative_window_forward,
        conservative_unary,
        conservative_decode,
    )
    return {
        "rates_per_second": {
            "training_window_state_cells_p50": p50_train,
            "training_window_state_cells_conservative": conservative_train,
            "early_window_state_cells_p50": p50_window_forward,
            "early_window_state_cells_conservative": conservative_window_forward,
            "full_unary_positions_p50": p50_unary,
            "full_unary_positions_conservative": conservative_unary,
            "exact_state_cells_p50": p50_decode,
            "exact_state_cells_conservative": conservative_decode,
            "conservative_throughput_quantile": conservative_quantile,
        },
        "workload": {
            "fit_window_state_cells_per_epoch": fit_window_state_cells_per_epoch,
            "early_stop_window_state_cells_per_epoch": early_stop_window_state_cells_per_epoch,
            "valid_unary_positions_real_plus_shuffle": valid_unary_positions,
            "valid_decode_cells_three_controls": valid_decode_cells,
            "maximum_epochs": max_epochs,
        },
        "projected_fold_runtime_seconds_p50": p50_seconds,
        "projected_fold_runtime_hours_p50": p50_seconds / 3600.0,
        "projected_fold_runtime_seconds_conservative": upper_seconds,
        "projected_fold_runtime_hours_conservative": upper_seconds / 3600.0,
    }


def _representative_grid_size_by_quartile(
    benchmark_windows: pd.DataFrame,
    measurement_frame: pd.DataFrame,
) -> dict[int, float]:
    merged = benchmark_windows.merge(
        measurement_frame[["well", "slot", "start_row", "grid_rows"]].drop_duplicates(),
        on=["well", "slot", "start_row"],
        validate="one_to_one",
    )
    return {
        int(quartile): float(group["grid_rows"].median())
        for quartile, group in merged.groupby("suffix_length_quartile", sort=True)
    }


def _assign_quartiles_like_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(
        ["suffix_rows", "well", "slot"], kind="mergesort"
    ).reset_index(drop=True)
    result["suffix_length_quartile"] = pd.qcut(
        np.arange(len(result)), 4, labels=False
    ).astype(int)
    return result


def run_stage0_microbenchmark(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    cost = validate_stage_a_cost_contract(config)
    if validate_selected_stage(config) != "stage0_microbenchmark":
        raise ValueError("run_stage0_microbenchmark requires selected_stage=stage0_microbenchmark")
    device = require_kaggle_gpu(config)
    seed = int(get_nested(config, "reproducibility.seed", 42))
    set_reproducibility(seed)
    torch.cuda.reset_peak_memory_stats(device)
    artifacts = artifact_dir()
    train_dir = resolve_train_dir(config)
    wells = list_paired_wells(train_dir)
    fold_map = build_fold_map(wells, int(get_nested(config, "validation.n_folds", 5)))
    outer_train, outer_valid = split_stage_a_wells(fold_map)
    fit_wells, early_stop_wells = split_early_stop_wells(outer_train, config)
    roles = {
        well: "fit" if well in set(fit_wells) else "early_stop"
        for well in outer_train
    }
    schedule = build_window_schedule_manifest(outer_train, train_dir, roles, config)
    benchmark_windows = select_fixed_benchmark_windows(schedule, config)
    benchmark_boundary = build_teacher_boundary_manifest(
        benchmark_windows, train_dir, config
    )
    benchmark_path = artifacts / f"{OUTPUT_PREFIX}_stage0_fixed16_window_manifest.csv"
    boundary_path = artifacts / f"{OUTPUT_PREFIX}_stage0_fixed16_boundary_manifest.csv"
    benchmark_windows.to_csv(benchmark_path, index=False)
    benchmark_boundary.to_csv(boundary_path, index=False)
    benchmark_keys = window_keys_from_manifests(
        benchmark_windows, benchmark_boundary, role="fit", epoch=0
    )

    model = PrefixConditionedUnary(config).to(device)
    parity_report, parity_padding = run_scalar_batch_parity(
        model, benchmark_keys, train_dir, config, device
    )
    parity_path = artifacts / f"{OUTPUT_PREFIX}_stage0_scalar_batch_parity.json"
    write_json(parity_path, parity_report)
    padding_frames = [parity_padding.assign(phase="scalar_batch_parity", batch_id=0)]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_nested(config, "model.training.learning_rate", 3e-4)),
        weight_decay=float(get_nested(config, "model.training.weight_decay", 1e-4)),
    )
    amp_enabled = bool(get_nested(config, "model.training.amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(get_nested(config, "model.training.gradient_accumulation_windows", 1))
    if accumulation != 1:
        raise ValueError("Stage 0 requires one optimizer step per four-window batch")
    clip_norm = float(get_nested(config, "model.training.gradient_clip_norm", 1.0))
    optimizer.zero_grad(set_to_none=True)
    train_rows: list[dict[str, Any]] = []
    model.train()
    for batch_id, key_batch in enumerate(fixed_window_batches(benchmark_keys)):
        _cuda_sync(device)
        end_to_end_started = time.perf_counter()
        views, truths, states = prepare_training_batch(key_batch, train_dir, config)
        _cuda_sync(device)
        gpu_started = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss, diagnostics, padded_state = batched_window_training_loss(
                model, views, truths, states, config, device
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _cuda_sync(device)
        gpu_elapsed = time.perf_counter() - gpu_started
        elapsed = time.perf_counter() - end_to_end_started
        padding = batch_padding_manifest(padded_state)
        padding["phase"] = "window_structured_forward_backward_optimizer"
        padding["batch_id"] = batch_id
        padding["well"] = [
            key_batch[index].well if index < len(key_batch) else None
            for index in range(len(padding))
        ]
        padding_frames.append(padding)
        train_rows.append(
            {
                "phase": "batched_window_structured_forward_backward_optimizer",
                "batch_id": batch_id,
                "well": ",".join(key.well for key in key_batch),
                "active_windows": len(key_batch),
                "scored_rows": sum(key.scored_rows for key in key_batch),
                "grid_rows": max(len(state.grid) for state in states),
                "rate_states": max(len(state.rates) for state in states),
                "state_cells": sum(
                    len(state.suffix_index) * len(state.grid) * len(state.rates)
                    for state in states
                ),
                "seconds": elapsed,
                "gpu_seconds": gpu_elapsed,
                "loss": float(np.mean([item["loss"] for item in diagnostics])),
                "structured_label_nll": float(
                    np.mean([item["structured_label_nll"] for item in diagnostics])
                ),
                "local_ce": float(np.mean([item["local_ce"] for item in diagnostics])),
                "target_in_grid_rate": float(
                    np.mean([item["target_in_grid_rate"] for item in diagnostics])
                ),
            }
        )

    window_forward_rows: list[dict[str, Any]] = []
    unary_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_id, key_batch in enumerate(fixed_window_batches(benchmark_keys)):
            _cuda_sync(device)
            end_to_end_started = time.perf_counter()
            views, truths, states = prepare_training_batch(key_batch, train_dir, config)
            _cuda_sync(device)
            gpu_started = time.perf_counter()
            _, diagnostics, _ = batched_window_training_loss(
                model, views, truths, states, config, device
            )
            _cuda_sync(device)
            gpu_elapsed = time.perf_counter() - gpu_started
            elapsed = time.perf_counter() - end_to_end_started
            window_forward_rows.append(
                {
                    "phase": "batched_window_structured_objective_forward_only",
                    "batch_id": batch_id,
                    "well": ",".join(key.well for key in key_batch),
                    "active_windows": len(key_batch),
                    "scored_rows": sum(key.scored_rows for key in key_batch),
                    "grid_rows": max(len(state.grid) for state in states),
                    "rate_states": max(len(state.rates) for state in states),
                    "state_cells": sum(
                        len(state.suffix_index) * len(state.grid) * len(state.rates)
                        for state in states
                    ),
                    "seconds": elapsed,
                    "gpu_seconds": gpu_elapsed,
                    "structured_label_nll": float(
                        np.mean([item["structured_label_nll"] for item in diagnostics])
                    ),
                    "local_ce": float(
                        np.mean([item["local_ce"] for item in diagnostics])
                    ),
                }
            )

        prepared_full = []
        for key in benchmark_keys:
            item = load_well_input(key.well, train_dir)
            view = prepare_view(item, item.tvt_input, config, view_name="official")
            shuffled_view = prepare_view(
                item,
                item.tvt_input,
                config,
                view_name="official_circular_shuffle",
                typewell_control="shuffle",
            )
            prepared_full.append((key, item, view, shuffled_view))
        prepared_full.sort(
            key=lambda value: (len(value[2].state.suffix_index), value[0].well)
        )
        for batch_id in range(0, len(prepared_full), 4):
            entries = prepared_full[batch_id : batch_id + 4]
            real_views = [entry[2] for entry in entries]
            shuffled_views = [entry[3] for entry in entries]
            real_unaries = []
            shuffled_unaries = []
            for key, _, view, shuffled_view in entries:
                _cuda_sync(device)
                started = time.perf_counter()
                real_unary, _ = model_unary(model, view, device)
                _cuda_sync(device)
                unary_rows.append(
                    {
                        "phase": "full_well_unary_forward",
                        "control": "real_gr",
                        "well": key.well,
                        "unary_positions": len(view.state.suffix_index)
                        * len(view.state.grid),
                        "seconds": time.perf_counter() - started,
                    }
                )
                real_unaries.append(real_unary)
                _cuda_sync(device)
                started = time.perf_counter()
                shuffled_unary, _ = model_unary(model, shuffled_view, device)
                _cuda_sync(device)
                unary_rows.append(
                    {
                        "phase": "full_well_unary_forward",
                        "control": "circular_shuffle",
                        "well": key.well,
                        "unary_positions": len(shuffled_view.state.suffix_index)
                        * len(shuffled_view.state.grid),
                        "seconds": time.perf_counter() - started,
                    }
                )
                shuffled_unaries.append(shuffled_unary)
            controls = {
                "real_gr": (real_unaries, real_views),
                "circular_shuffle": (shuffled_unaries, shuffled_views),
                "geometry_only": (
                    [torch.zeros_like(value) for value in real_unaries],
                    real_views,
                ),
            }
            for control, (unaries, control_views) in controls.items():
                decode_state = build_batched_state_spec(
                    [view.state for view in control_views], required_batch_size=4
                )
                decode_padding = batch_padding_manifest(decode_state)
                decode_padding["phase"] = f"full_well_decode_{control}"
                decode_padding["batch_id"] = batch_id // 4
                decode_padding["well"] = [
                    entries[index][0].well if index < len(entries) else None
                    for index in range(len(decode_padding))
                ]
                padding_frames.append(decode_padding)
                _cuda_sync(device)
                started = time.perf_counter()
                decoded_batch = decode_unary_batch(
                    unaries, control_views, config, compute_viterbi=False
                )
                _cuda_sync(device)
                elapsed = time.perf_counter() - started
                if not all(np.isfinite(value.prediction).all() for value in decoded_batch):
                    raise ValueError(f"non-finite {control} batched benchmark prediction")
                decode_rows.append(
                    {
                        "phase": "batched_frozen_exact_ssm_decode",
                        "control": control,
                        "batch_id": batch_id // 4,
                        "well": ",".join(entry[0].well for entry in entries),
                        "suffix_rows": sum(
                            len(view.state.suffix_index) for view in control_views
                        ),
                        "grid_rows": max(len(view.state.grid) for view in control_views),
                        "rate_states": max(len(view.state.rates) for view in control_views),
                        "state_cells": sum(
                            len(view.state.suffix_index)
                            * len(view.state.grid)
                            * len(view.state.rates)
                            for view in control_views
                        ),
                        "seconds": elapsed,
                    }
                )

    padding_manifest = pd.concat(padding_frames, ignore_index=True)
    padding_path = artifacts / f"{OUTPUT_PREFIX}_stage0_batch_padding_manifest.csv"
    padding_manifest.to_csv(padding_path, index=False)

    train_measurements = pd.DataFrame(train_rows)
    window_forward_measurements = pd.DataFrame(window_forward_rows)
    unary_measurements = pd.DataFrame(unary_rows)
    decode_measurements = pd.DataFrame(decode_rows)
    measurements = pd.concat(
        [
            train_measurements,
            window_forward_measurements,
            unary_measurements,
            decode_measurements,
        ],
        ignore_index=True,
        sort=False,
    )
    measurement_path = artifacts / f"{OUTPUT_PREFIX}_stage0_measurements.csv"
    measurements.to_csv(measurement_path, index=False)

    median_grid = float(train_measurements["grid_rows"].median())
    rate_states = int(get_nested(config, "model.state_space.n_rates", 41))
    active_schedule = schedule.loc[schedule["active"]].copy()
    active_schedule["estimated_state_cells"] = (
        active_schedule["scored_rows"] * median_grid * rate_states
    )
    fit_cells = int(
        active_schedule.loc[active_schedule["role"] == "fit"]
        .groupby("epoch")["estimated_state_cells"]
        .sum()
        .max()
    )
    early_cells = int(
        active_schedule.loc[active_schedule["role"] == "early_stop"]
        .groupby("epoch")["estimated_state_cells"]
        .sum()
        .max()
    )
    valid_positions_single = 0
    valid_decode_cells_single = 0
    for well in sorted(outer_valid):
        item = load_well_input(well, train_dir)
        view = prepare_view(item, item.tvt_input, config, view_name="official")
        positions = len(view.state.suffix_index) * len(view.state.grid)
        valid_positions_single += positions
        valid_decode_cells_single += positions * len(view.state.rates)
    gate_config = get_nested(config, "validation.stage0", {}) or {}
    projection = project_stage_a_runtime(
        train_measurements,
        window_forward_measurements,
        unary_measurements,
        decode_measurements,
        fit_window_state_cells_per_epoch=fit_cells,
        early_stop_window_state_cells_per_epoch=early_cells,
        valid_unary_positions=2 * valid_positions_single,
        valid_decode_cells=3 * valid_decode_cells_single,
        max_epochs=int(get_nested(config, "model.training.max_epochs", 8)),
        conservative_quantile=float(
            gate_config.get("conservative_throughput_quantile", 0.10)
        ),
    )
    peak_gpu_memory_gb = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    runtime_pass = projection["projected_fold_runtime_hours_conservative"] <= float(
        gate_config["maximum_projected_fold_runtime_hours"]
    )
    memory_pass = peak_gpu_memory_gb <= float(gate_config["maximum_peak_gpu_memory_gb"])
    exp332_hours = float(
        get_nested(
            config,
            "validation.baselines.exp332_runtime.projected_fold_runtime_hours_conservative",
        )
    )
    speedup = exp332_hours / max(
        float(projection["projected_fold_runtime_hours_conservative"]), 1e-12
    )
    speedup_pass = speedup >= float(gate_config["minimum_speedup_vs_exp332"])
    technical_pass = bool(parity_report["passed"])
    measurement_finite = bool(
        np.isfinite(measurements["seconds"]).all()
        and (measurements["seconds"] > 0.0).all()
    )
    finite_pass = technical_pass and measurement_finite
    passed = bool(
        technical_pass and finite_pass and runtime_pass and memory_pass and speedup_pass
    )
    report = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_fixed16_t4_microbenchmark",
        "status": "passed_waiting_stage_a_approval" if passed else "failed_branch_closed",
        "fixed_window_count": len(benchmark_windows),
        "selection_manifest_sha256": sha256_path(benchmark_path),
        "teacher_boundary_manifest_sha256": sha256_path(boundary_path),
        "batch_padding_manifest_sha256": sha256_path(padding_path),
        "scalar_batch_parity_sha256": sha256_path(parity_path),
        "measurement_sha256": sha256_path(measurement_path),
        "scalar_batch_parity": parity_report,
        "projection": projection,
        "speedup_vs_exp332": speedup,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "gate": {
            "passed": passed,
            "technical_parity_pass": technical_pass,
            "finite_pass": finite_pass,
            "runtime_pass": runtime_pass,
            "memory_pass": memory_pass,
            "speedup_pass": speedup_pass,
            "maximum_projected_fold_runtime_hours": float(
                gate_config["maximum_projected_fold_runtime_hours"]
            ),
            "maximum_peak_gpu_memory_gb": float(
                gate_config["maximum_peak_gpu_memory_gb"]
            ),
            "minimum_speedup_vs_exp332": float(
                gate_config["minimum_speedup_vs_exp332"]
            ),
            "decision": "request_separate_stage_a_approval"
            if passed
            else str(gate_config["failure_action"]),
        },
        "cost_contract": cost,
        "outer_valid_truth_access_count": 0,
        "trained_stage_a_model_count": 0,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    report_path = artifacts / f"{OUTPUT_PREFIX}_stage0_benchmark_report.json"
    write_json(report_path, report)
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_passed_waiting_stage_a_approval"
        if passed
        else "stage0_failed_branch_closed",
        "route": "ensemble",
        "metric": get_nested(config, "validation.metric"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "stage0": report,
        "inference": False,
        "submission": False,
    }
    write_json(metrics_output_path(), metrics_payload)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return report


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
# ## 12. Setup and contract preview

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    COST_CONTRACT = validate_stage_a_cost_contract(CONFIG)
    SELECTED_STAGE = validate_selected_stage(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "selected_stage": SELECTED_STAGE,
                "scientific_contract": SCIENTIFIC_CONTRACT,
                "stage_a_cost": COST_CONTRACT,
                "stage_b_authorized": False,
                "inference_authorized": False,
                "submission_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 13. Run only the separately authorized Kaggle GPU stage

# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    if SELECTED_STAGE == "stage0_microbenchmark":
        STAGE_SUMMARY = run_stage0_microbenchmark(CONFIG)
    elif SELECTED_STAGE == "stage_a_fold0":
        STAGE_SUMMARY = run_stage_a(CONFIG)
    else:
        raise RuntimeError(
            "exp347 is implementation-only; select an explicitly approved Kaggle stage"
        )
