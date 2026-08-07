# %% [markdown]
# # exp295 prefix-anchored whole-well GR alignment SSM train
#
# Stage A trains exactly one fold-0 complete-well neural GR emission model.
# Each validation well is decoded from its own horizontal GR, its paired Type
# Well GR, and its known TVT prefix.  The exp209 state grid and transition are
# fixed.  Validation truth is first read only after the model, unaries,
# posterior marginals, controls, row identities, and SHA manifests are frozen.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Scientific, execution-cost, and leakage contract guards
# 4. Complete-well folds, mask-first loading, and pseudo-cut manifests
# 5. Robust GR preprocessing and prefix-context helpers
# 6. Prefix-conditioned multi-scale neural emission
# 7. Fixed exp209 exact state-space forward-backward and Viterbi helpers
# 8. Fold-0 training and outer-train early stopping
# 9. Freeze-first outer-valid decoding, readout, and Stage A gates
# 10. Stage A orchestration and generated artifacts
# 11. Setup and contract preview
# 12. Run the separately authorized Kaggle GPU stage

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


EXPERIMENT_NAME = "exp295_prefix_anchored_wholewell_gr_alignment_ssm"
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


EXECUTE_NOTEBOOK = os.environ.get("EXP295_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp295 config not found in {[str(path) for path in candidates]}")


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
        raise RuntimeError("Stage A execution is Kaggle Notebook only; local execution is disabled")
    if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
        raise RuntimeError("Stage A Kaggle GPU push has not been separately approved")
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError("Stage A requires a Kaggle CUDA runtime")
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
        "model.sample_unit": "complete_well",
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
        "model.state_space.solver": "exact_log_space_forward_backward",
        "model.state_space.primary_readout": "posterior_mean_tvt",
        "model.training.objective.structured_label_nll_weight": 1.0,
        "model.training.objective.label_observation_distribution": "gaussian",
        "model.training.objective.label_observation_sigma_ft": 0.35,
        "model.training.objective.local_true_state_ce_weight": 0.25,
        "model.training.max_epochs": 8,
        "model.training.batch_size_wells": 1,
        "model.training.gradient_accumulation_wells": 4,
        "model.training.dataloader_workers": 0,
    }
    changed = [
        f"{key}={get_nested(config, key)!r} expected {value!r}"
        for key, value in expected.items()
        if get_nested(config, key) != value
    ]
    if changed:
        raise ValueError("exp295 locked scientific contract changed: " + "; ".join(changed))
    inputs = tuple(get_nested(config, "data.horizontal_file_input_columns", []))
    if inputs != HORIZONTAL_INPUT_COLUMNS:
        raise ValueError("horizontal input allowlist must be exactly MD/X/Y/Z/GR/TVT_input")
    forbidden_actions = set(get_nested(config, "model.forbidden_actions", []))
    required_forbidden = {
        "neighbor_well_path_or_tvt_input",
        "same_typewell_horizontal_donor",
        "spatial_neighbor_feature_or_prior",
        "candidate_bank_or_selector",
        "test_time_backpropagation",
        "blend_with_existing_ml_pf_or_beam",
    }
    if not required_forbidden.issubset(forbidden_actions):
        raise ValueError("neighbor/candidate/adaptation/blend prohibitions are incomplete")
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


def assert_mask_first_schema(columns: Sequence[str], *, allow_truth: bool = False) -> None:
    names = set(columns)
    if not set(HORIZONTAL_INPUT_COLUMNS).issubset(names):
        missing = sorted(set(HORIZONTAL_INPUT_COLUMNS) - names)
        raise ValueError(f"horizontal input is missing required columns: {missing}")
    selected = set(HORIZONTAL_INPUT_COLUMNS)
    if not allow_truth and selected & FORBIDDEN_MODEL_COLUMNS:
        raise ValueError("forbidden truth/candidate column entered the model allowlist")


# %% [markdown]
# ## 4. Complete-well folds, mask-first loading, and pseudo-cut manifests

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
class ViewKey:
    well: str
    offset_rows: int
    view_name: str


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


def make_view_tvt_input(base: np.ndarray, offset_rows: int) -> tuple[np.ndarray, int]:
    official_end = prefix_end_index(base)
    cut_end = official_end - int(offset_rows)
    if cut_end < 31:
        raise ValueError("pseudo-cut would leave fewer than 32 known-prefix rows")
    view = np.asarray(base, dtype=np.float64).copy()
    view[cut_end + 1 :] = np.nan
    if np.isfinite(view[cut_end + 1 :]).any():
        raise ValueError("pseudo-cut failed to hide the full suffix")
    return view, cut_end


def build_pseudo_cut_manifest(
    wells: Sequence[str], train_dir: Path, config: Mapping[str, Any]
) -> pd.DataFrame:
    offsets = [int(value) for value in get_nested(
        config, "validation.outer_train_pseudo_cut_offsets_rows", [0, 256, 512]
    )]
    rows: list[dict[str, Any]] = []
    for well in sorted(wells):
        item = load_well_input(well, train_dir)
        official_end = prefix_end_index(item.tvt_input)
        for offset in offsets:
            cut_end = official_end - offset
            eligible = cut_end >= 31
            rows.append(
                {
                    "well": well,
                    "view_name": "official" if offset == 0 else f"pseudo_minus_{offset}",
                    "offset_rows": offset,
                    "official_prefix_end": official_end,
                    "cut_end": cut_end if eligible else None,
                    "known_rows": cut_end + 1 if eligible else 0,
                    "hidden_rows": len(item.md) - cut_end - 1 if eligible else 0,
                    "eligible": bool(eligible),
                    "stable_seed": stable_uint64(EXPERIMENT_NAME, STAGE_A_FOLD, well, offset),
                }
            )
    return pd.DataFrame(rows)


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
        ) -> tuple[Any, Any, Any]:
            context = self.context_encoder(prefix_pairs, huber_summary)
            horizontal = self.horizontal_encoder(horizontal_channels[None, ...], context)[0]
            typewell = self.typewell_encoder(typewell_channels[None, ...], context)[0]
            horizontal = F.normalize(self.horizontal_projection(horizontal), dim=-1, eps=1e-6)
            typewell = F.normalize(self.typewell_projection(typewell), dim=-1, eps=1e-6)
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
            raise RuntimeError("PyTorch is required to instantiate the exp295 neural emission")


def prepared_to_torch(view: PreparedView, device: Any) -> tuple[Any, Any, Any, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")
    horizontal = torch.as_tensor(view.horizontal_channels, dtype=torch.float32, device=device)
    typewell = torch.as_tensor(view.typewell_channels, dtype=torch.float32, device=device)
    pairs = torch.as_tensor(view.prefix_pairs, dtype=torch.float32, device=device)
    summary = torch.as_tensor(view.prefix_huber_summary, dtype=torch.float32, device=device)
    return horizontal, typewell, pairs, summary


def model_unary(model: Any, view: PreparedView, device: Any) -> tuple[Any, float]:
    horizontal, typewell, pairs, summary = prepared_to_torch(view, device)
    unary_full, _, temperature = model(horizontal, typewell, pairs, summary)
    unary_suffix = unary_full[
        torch.as_tensor(view.state.suffix_index, dtype=torch.long, device=device)
    ]
    return unary_suffix.float(), float(temperature.detach().cpu())


# %% [markdown]
# ## 7. Fixed exp209 exact state-space forward-backward and Viterbi helpers

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
        if sigma <= 0.0:
            raise ValueError("label_observation_sigma_ft must be positive")
        grid = torch.as_tensor(
            spec.grid, dtype=target.dtype, device=target.device
        )
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
                ) = soft_label_structured_terms(
                    unary, target, spec, config
                )
                token_count = max(1, len(spec.suffix_index))
                value = (log_partition - conditioned_log_partition) / token_count
            ctx.save_for_backward(posterior, conditioned_posterior)
            ctx.token_count = max(1, len(spec.suffix_index))
            return value

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, None, None, None]:
            posterior, conditioned_posterior = ctx.saved_tensors
            grad_unary = posterior - conditioned_posterior
            grad_unary = grad_unary * gradient / ctx.token_count
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
        raise RuntimeError("PyTorch is required for exact exp295 decoding")
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


# %% [markdown]
# ## 8. Fold-0 training and outer-train early stopping

# %%
def view_keys_from_manifest(manifest: pd.DataFrame, wells: Sequence[str]) -> list[ViewKey]:
    allowed = set(wells)
    selected = manifest.loc[manifest["eligible"] & manifest["well"].isin(allowed)]
    return [
        ViewKey(str(row.well), int(row.offset_rows), str(row.view_name))
        for row in selected.itertuples(index=False)
    ]


def stable_epoch_order(keys: Sequence[ViewKey], seed: int, epoch: int) -> list[ViewKey]:
    return sorted(
        keys,
        key=lambda key: stable_uint64(
            EXPERIMENT_NAME, "epoch-order", seed, epoch, key.well, key.offset_rows
        ),
    )


def prepare_training_view(
    key: ViewKey,
    train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[PreparedView, WellTruth]:
    item = load_well_input(key.well, train_dir)
    truth = load_well_truth(key.well, train_dir)
    tvt_input, _ = make_view_tvt_input(item.tvt_input, key.offset_rows)
    view = prepare_view(item, tvt_input, config, view_name=key.view_name)
    return view, truth


def training_loss(
    model: Any,
    view: PreparedView,
    truth: WellTruth,
    config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, float]]:
    unary, temperature = model_unary(model, view, device)
    target = truth.tvt[view.state.suffix_index]
    if not np.isfinite(target).all():
        raise ValueError(f"{view.well}: training truth contains non-finite suffix values")
    truth_index_np = nearest_grid_indices(view.state.grid, target)
    truth_index = torch.as_tensor(truth_index_np, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    structured = SoftLabelStructuredNLL.apply(unary, target_tensor, view.state, config)
    local = F.cross_entropy(unary, truth_index)
    structured_weight = float(
        get_nested(config, "model.training.objective.structured_label_nll_weight", 1.0)
    )
    local_weight = float(
        get_nested(config, "model.training.objective.local_true_state_ce_weight", 0.25)
    )
    loss = structured_weight * structured + local_weight * local
    coverage = float(
        np.mean((target >= view.state.grid[0]) & (target <= view.state.grid[-1]))
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "structured_label_nll": float(structured.detach().cpu()),
        "local_ce": float(local.detach().cpu()),
        "temperature": temperature,
        "target_in_grid_rate": coverage,
        "tokens": float(len(target)),
    }


def evaluate_early_stop_loss(
    model: Any,
    keys: Sequence[ViewKey],
    train_dir: Path,
    config: Mapping[str, Any],
    device: Any,
) -> float:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for key in keys:
            view, truth = prepare_training_view(key, train_dir, config)
            unary, _ = model_unary(model, view, device)
            target = truth.tvt[view.state.suffix_index]
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
            _, _, log_partition, conditioned_log_partition = soft_label_structured_terms(
                unary, target_tensor, view.state, config
            )
            token_count = max(1, len(view.state.suffix_index))
            structured_nll = (log_partition - conditioned_log_partition) / token_count
            values.append(float(structured_nll.cpu()))
    return float(np.mean(values)) if values else float("inf")


def train_fold0_model(
    config: Mapping[str, Any],
    train_dir: Path,
    fit_keys: Sequence[ViewKey],
    early_stop_keys: Sequence[ViewKey],
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
    accumulation = int(get_nested(config, "model.training.gradient_accumulation_wells", 4))
    max_epochs = int(get_nested(config, "model.training.max_epochs", 8))
    clip_norm = float(get_nested(config, "model.training.gradient_clip_norm", 1.0))
    patience = int(get_nested(config, "model.training.early_stopping_patience_epochs", 2))
    min_delta = float(
        get_nested(config, "model.training.early_stopping_min_delta_nll_per_token", 0.001)
    )
    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_nll = float("inf")
    epochs_without_improvement = 0
    started = time.time()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ordered = stable_epoch_order(fit_keys, seed, epoch)
        epoch_rows: list[dict[str, float]] = []
        for index, key in enumerate(ordered):
            view, truth = prepare_training_view(key, train_dir, config)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, diagnostics = training_loss(model, view, truth, config, device)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(ordered):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_rows.append(diagnostics)
        early_nll = evaluate_early_stop_loss(
            model, early_stop_keys, train_dir, config, device
        )
        row = {
            "epoch": epoch + 1,
            "train_views": len(epoch_rows),
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])),
            "train_structured_label_nll": float(
                np.mean([item["structured_label_nll"] for item in epoch_rows])
            ),
            "train_local_ce": float(np.mean([item["local_ce"] for item in epoch_rows])),
            "train_temperature": float(
                np.mean([item["temperature"] for item in epoch_rows])
            ),
            "train_target_in_grid_rate": float(
                np.mean([item["target_in_grid_rate"] for item in epoch_rows])
            ),
            "early_stop_true_state_nll": early_nll,
            "elapsed_seconds": time.time() - started,
        }
        history_rows.append(row)
        print(json.dumps(to_jsonable(row), sort_keys=True), flush=True)
        if early_nll < best_nll - min_delta:
            best_nll = early_nll
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite outer-train early-stop checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device).eval()
    meta = {
        "best_early_stop_true_state_nll": best_nll,
        "selected_epoch": int(
            min(history_rows, key=lambda row: row["early_stop_true_state_nll"])["epoch"]
        ),
        "completed_epochs": len(history_rows),
        "train_seconds": time.time() - started,
    }
    return model, pd.DataFrame(history_rows), meta


# %% [markdown]
# ## 9. Freeze-first outer-valid decoding, readout, and Stage A gates

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
    for number, well in enumerate(sorted(valid_wells), start=1):
        print(f"[valid {number}/{len(valid_wells)}] freeze well={well}", flush=True)
        item = load_well_input(well, train_dir)
        view = prepare_view(item, item.tvt_input, config, view_name="official")
        shuffled_view = prepare_view(
            item,
            item.tvt_input,
            config,
            view_name="official_circular_shuffle",
            typewell_control="shuffle",
        )
        with torch.no_grad():
            real_unary, real_temperature = model_unary(model, view, device)
            shuffled_unary, shuffled_temperature = model_unary(model, shuffled_view, device)
            geometry_unary = torch.zeros_like(real_unary)
            real = decode_unary(real_unary, view, config, compute_viterbi=True)
            shuffled = decode_unary(
                shuffled_unary, shuffled_view, config, compute_viterbi=False
            )
            geometry = decode_unary(
                geometry_unary, view, config, compute_viterbi=False
            )
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
                "real_temperature": real_temperature,
                "shuffle_temperature": shuffled_temperature,
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
                "rows": len(suffix),
                "grid_rows": len(view.state.grid),
                "real_unary_sha256": array_content_sha256(real_unary.cpu().numpy()),
                "shuffle_unary_sha256": array_content_sha256(shuffled_unary.cpu().numpy()),
                "geometry_unary_sha256": array_content_sha256(geometry_unary.cpu().numpy()),
                "real_posterior_sha256": array_content_sha256(real.posterior),
                "shuffle_posterior_sha256": array_content_sha256(shuffled.posterior),
                "row_identity_sha256": array_content_sha256(suffix.astype(np.int64)),
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


def post_freeze_readout(
    frozen: pd.DataFrame,
    content_manifest: pd.DataFrame,
    valid_wells: Sequence[str],
    train_dir: Path,
    config: Mapping[str, Any],
    runtime_seconds: float,
    peak_gpu_memory_gb: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
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
        else "close_stage_b_without_exp295_rescue_grid",
    }
    return readout, by_well, metrics, guard


# %% [markdown]
# ## 10. Stage A orchestration and generated artifacts

# %%
def run_stage_a(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    cost = validate_stage_a_cost_contract(config)
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
    pseudo_manifest = build_pseudo_cut_manifest(outer_train, train_dir, config)
    pseudo_path = artifacts / f"{OUTPUT_PREFIX}_pseudo_cut_manifest.csv"
    pseudo_manifest.to_csv(pseudo_path, index=False)
    fit_keys = view_keys_from_manifest(pseudo_manifest, fit_wells)
    early_keys = [
        key
        for key in view_keys_from_manifest(pseudo_manifest, early_stop_wells)
        if key.offset_rows == 0
    ]
    model, history, training_meta = train_fold0_model(
        config, train_dir, fit_keys, early_keys, device
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
    readout, by_well, stage_metrics, guard = post_freeze_readout(
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
    stage_metrics_path = artifacts / f"{OUTPUT_PREFIX}_stage_a_metrics.json"
    write_json(stage_metrics_path, {"metrics": stage_metrics, "guard": guard})
    status = (
        "stage_a_passed_waiting_stage_b_approval"
        if guard["passed"]
        else "stage_a_failed_branch_closed"
    )
    output_paths = {
        "fold_map": fold_path,
        "pseudo_cut_manifest": pseudo_path,
        "training_history": history_path,
        "model": model_path,
        "model_manifest": model_manifest_path,
        "frozen_predictions": frozen_path,
        "freeze_manifest": freeze_path,
        "input_manifest": input_path,
        "validation_readout": readout_path,
        "by_well_metrics": by_well_path,
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
        "metrics": stage_metrics,
        "guard": guard,
        "truth_freeze": {
            **freeze_manifest,
            "truth_loaded_after_global_freeze": True,
        },
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
# ## 11. Setup and contract preview

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    COST_CONTRACT = validate_stage_a_cost_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
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
# ## 12. Run the separately authorized Kaggle GPU stage

# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    STAGE_A_SUMMARY = run_stage_a(CONFIG)
