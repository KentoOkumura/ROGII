# %% [markdown]
# # exp420 exp226/HMM-guided defensive-mixture likelihood-PF — train
#
# This train-side candidate starts from exp419. Half of the particles retain
# the original exp072 transition proposal. Inactive rows allocate the remaining
# half to the fold-safe exp226 geometry rate. For 32 transitions after a frozen
# untreated-HMM innovation trigger, half of the guidance mass is moved to a
# one-rate-cell directional family. Exact p0/q correction preserves the
# original target posterior. HMM absolute predictions, backward messages,
# exp226 final predictions, truth, roles, and cause labels are forbidden until
# schedule, candidate, and target-free diagnostics have been frozen.

# %% [markdown]
# ## Contents
# 1. Imports and fixed notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen HMM schedule and proposal scientific contract
# 4. Truth-free raw input checks and deterministic LPT sharding
# 5. Exact untreated-HMM forward innovation schedule
# 6. Exact exp072 PF input preparation and exp226 geometry rate
# 7. Scheduled defensive-mixture proposal and importance correction
# 8. Stage-0/full shard candidate generation and freeze
# 9. Strict merge and optional rerun probes
# 10. Late truth, saved controls, roles, episodes, and scopes
# 11. Stage-0/full metrics and fail-closed gates
# 12. Generated artifacts and stage orchestration
# 13. Setup and configuration preview

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
from dataclasses import dataclass
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


EXPERIMENT_NAME = "exp420_exp226_hmm_guided_defensive_mixture_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "exp226_hmm_guided_defensive_mixture_scale5"
PREDICTION_COLUMNS = (PRIMARY_CANDIDATE,)
SHARD_COUNT = 4
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP420_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


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
    raise FileNotFoundError(f"exp420 config not found; checked={checked}")


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
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    csv_path = Path(path)
    return {
        "path": str(csv_path),
        "bytes": csv_path.stat().st_size,
        "raw_sha256": sha256_path(csv_path),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(csv_path, nrows=0, compression="gzip").columns.astype(str).tolist(),
    }


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
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
    return mapping_sha256({str(column): str(dtype) for column, dtype in frame.dtypes.items()})


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        paths = (
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            Path.cwd() / candidate
            if candidate.name == filename
            else Path.cwd() / candidate / filename,
        )
        for path in paths:
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "numba_available": NUMBA_AVAILABLE,
    }
    if NUMBA_AVAILABLE:
        import numba

        versions["numba"] = numba.__version__
    return versions


def maximum_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


# %% [markdown]
# ## 3. Frozen HMM schedule and proposal scientific contract


# %%
def proposal_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    proposal = dict(get_nested(config, "model.rate_proposal") or {})
    target = dict(proposal.get("target_component") or {})
    inactive = dict(proposal.get("inactive") or {})
    active = dict(proposal.get("active") or {})
    target_weight = float(target["weight"])
    inactive_geometry_weights = [
        float(value) for value in inactive["geometry_component_weights"]
    ]
    active_geometry_weights = [
        float(value) for value in active["geometry_component_weights"]
    ]
    active_hmm_weights = [float(value) for value in active["hmm_component_weights"]]
    sigma_multipliers = [float(value) for value in inactive["sigma_multipliers"]]
    active_sigma_multipliers = [
        float(value) for value in active["sigma_multipliers"]
    ]
    inactive_weight_sum = target_weight + sum(inactive_geometry_weights)
    active_weight_sum = (
        target_weight + sum(active_geometry_weights) + sum(active_hmm_weights)
    )
    contract = {
        "family": str(proposal["family"]),
        "target_weight": target_weight,
        "target_sigma_multiplier": float(target["sigma_multiplier"]),
        "inactive_geometry_weights": inactive_geometry_weights,
        "active_geometry_weights": active_geometry_weights,
        "active_hmm_weights": active_hmm_weights,
        "sigma_multipliers": sigma_multipliers,
        "hmm_center": str(active["hmm_center"]),
        "importance_ratio": str(proposal["importance_ratio"]),
        "importance_clipping": bool(proposal["importance_clipping"]),
        "maximum_importance_ratio_by_construction": float(
            proposal["maximum_importance_ratio_by_construction"]
        ),
        "inactive_weight_sum": inactive_weight_sum,
        "active_weight_sum": active_weight_sum,
        "geometry_input_columns": ["well_id", "row_idx", "suffix_offset", "tvt_geop"],
        "schedule_input_columns": [
            "well_id",
            "row_idx",
            "suffix_offset",
            "active",
            "direction",
        ],
        "geometry_surface": "tvt_geop_plus_z",
        "target_posterior_changed": False,
    }
    tolerance = float(
        get_nested(config, "guards.stage_0_technical.weight_sum_absolute_tolerance")
    )
    if (
        abs(inactive_weight_sum - 1.0) > tolerance
        or abs(active_weight_sum - 1.0) > tolerance
    ):
        raise ValueError(
            "exp420 inactive/active proposal weights do not sum to one: "
            f"{inactive_weight_sum}/{active_weight_sum}"
        )
    if (
        target_weight != 0.5
        or inactive_geometry_weights != [1.0 / 6.0] * 3
        or active_geometry_weights != [1.0 / 12.0] * 3
        or active_hmm_weights != [1.0 / 12.0] * 3
        or sigma_multipliers != [1.0, 4.0, 16.0]
        or active_sigma_multipliers != sigma_multipliers
        or float(target["sigma_multiplier"]) != 1.0
        or bool(proposal["importance_clipping"])
        or str(proposal["importance_ratio"])
        != "target_rate_density_divided_by_mixture_rate_density"
        or str(active["hmm_center"])
        != "target_transition_rate_mean_plus_direction_times_0p005"
    ):
        raise ValueError("exp420 fixed scheduled proposal contract changed")
    contract["proposal_contract_sha256"] = mapping_sha256(contract)
    return contract


def hmm_schedule_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    signal = dict(get_nested(config, "model.hmm_signal") or {})
    cusum = dict(signal.get("cusum") or {})
    contract = {
        "kernel": str(signal["kernel"]),
        "uses_backward_pass": bool(signal["uses_backward_pass"]),
        "uses_hmm_prediction": bool(signal["uses_hmm_prediction"]),
        "uses_absolute_hmm_state": bool(signal["uses_absolute_hmm_state"]),
        "step": float(signal["step"]),
        "n_rates": int(signal["n_rates"]),
        "rate_span": float(signal["rate_span"]),
        "rate_step": float(signal["rate_step"]),
        "sig_r": float(signal["sig_r"]),
        "sig_p": float(signal["sig_p"]),
        "momentum": float(signal["momentum"]),
        "emission": str(signal["emission"]),
        "statistic": str(signal["statistic"]),
        "drift_allowance_rate_cells": float(cusum["drift_allowance_rate_cells"]),
        "positive_threshold_rate_cells": float(
            cusum["positive_threshold_rate_cells"]
        ),
        "negative_threshold_rate_cells": float(
            cusum["negative_threshold_rate_cells"]
        ),
        "activation_transitions": int(cusum["activation_transitions"]),
        "refractory_rows": int(cusum["refractory_rows"]),
        "reset_on_trigger": bool(cusum["reset_on_trigger"]),
        "allow_overlapping_trigger": bool(cusum["allow_overlapping_trigger"]),
        "allow_direction_flip_while_active": bool(
            cusum["allow_direction_flip_while_active"]
        ),
        "first_affected_transition": str(cusum["first_affected_transition"]),
        "freeze_before_pf": bool(cusum["freeze_before_pf"]),
    }
    expected = {
        "kernel": "untreated_exp209_forward_filter",
        "uses_backward_pass": False,
        "uses_hmm_prediction": False,
        "uses_absolute_hmm_state": False,
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "rate_step": 0.005,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "momentum": 0.998,
        "emission": "gaussian",
        "statistic": (
            "filtered_rate_mean_minus_predictive_rate_mean_divided_by_rate_step"
        ),
        "drift_allowance_rate_cells": 0.01,
        "positive_threshold_rate_cells": 1.0,
        "negative_threshold_rate_cells": 1.0,
        "activation_transitions": 32,
        "refractory_rows": 128,
        "reset_on_trigger": True,
        "allow_overlapping_trigger": False,
        "allow_direction_flip_while_active": False,
        "first_affected_transition": "row_after_trigger",
        "freeze_before_pf": True,
    }
    if contract != expected:
        raise ValueError("exp420 untreated-HMM schedule contract changed")
    contract["schedule_contract_sha256"] = mapping_sha256(contract)
    return contract


def pf_fixed_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    pf = dict(get_nested(config, "model.pf") or {})
    transition = dict(get_nested(config, "model.target_transition") or {})
    return {
        "particles": int(pf["particles"]),
        "seeds": int(pf["seeds"]),
        "typewell_grid_step_ft": float(pf["typewell_grid_step_ft"]),
        "initial_position_spread_ft": float(pf["initial_position_spread_ft"]),
        "initial_rate_spread": float(pf["initial_rate_spread"]),
        "momentum": float(transition["momentum"]),
        "rate_noise": float(transition["rate_noise"]),
        "position_noise": float(transition["position_noise"]),
        "minimum_md_delta": float(transition["minimum_md_delta"]),
        "resample_threshold_fraction": float(pf["resample_threshold_fraction"]),
        "resampling": str(pf["resampling"]),
        "rough_position": float(pf["rough_position"]),
        "rough_rate": float(pf["rough_rate"]),
        "emission": str(pf["emission"]),
        "emission_clip_z2": float(pf["emission_clip_z2"]),
        "gr_sigma_multiplier": float(pf["gr_sigma_multiplier"]),
        "gr_sigma_clip": [float(value) for value in pf["gr_sigma_clip"]],
        "typewell_tvt_pad_ft": float(pf["typewell_tvt_pad_ft"]),
        "missing_gr_policy": str(pf["missing_gr_policy"]),
        "seed_aggregation_temperature": float(get_nested(config, "model.aggregation.temperature")),
    }


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "geometry_parent": get_nested(config, "lineage.geometry_parent"),
        "hmm_kernel_parent": get_nested(config, "lineage.hmm_kernel_parent"),
        "hmm_schedule_reference": get_nested(
            config, "lineage.hmm_schedule_reference"
        ),
        "mechanism_evidence": list(get_nested(config, "lineage.mechanism_evidence") or []),
        "truth_attached": False,
        "primary_control": str(get_nested(config, "validation.primary_control")),
        "primary_candidate": PRIMARY_CANDIDATE,
        "standalone_reference": str(get_nested(config, "validation.standalone_reference")),
        "control_pf": "saved_exp404_scale5_x1p0_load_only_zero_reruns",
        "fixed_pf_parameters": pf_fixed_parameters(config),
        "hmm_schedule": hmm_schedule_contract(config),
        "proposal": proposal_contract(config),
        "execution_counts": {
            "scientific_variants": get_nested(config, "execution.scientific_variants"),
            "stage_0": dict(get_nested(config, "execution.stage_0") or {}),
            "full": dict(get_nested(config, "execution.full") or {}),
            "reporting_folds": get_nested(config, "execution.reporting_folds"),
            "lightgbm_configs": get_nested(config, "execution.lightgbm_configs"),
            "trained_folds": get_nested(config, "execution.trained_folds"),
            "boosters": get_nested(config, "execution.boosters"),
            "models": get_nested(config, "execution.models"),
            "beam_well_runs": get_nested(config, "execution.beam_well_runs"),
            "gpu_runs": get_nested(config, "execution.gpu_runs"),
        },
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "proposal_allowlist": [
            "MD",
            "Z",
            "GR",
            "TVT_input",
            "tvt_geop",
            "active",
            "direction",
        ],
        "forbidden": list(get_nested(config, "guards.forbidden") or []),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp419_exp226_guided_defensive_mixture_pf",
        "lineage.scientific_pf_parent": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "lineage.geometry_parent": (
            "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
        ),
        "lineage.hmm_kernel_parent": (
            "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
        ),
        "lineage.hmm_schedule_reference": (
            "exp411_predictive_filtered_rate_innovation_destick"
        ),
        "implementation.enabled": True,
        "implementation.scope": "train_side_stage0_and_full_candidate_implementation_only",
        "model.active_variants": ["exp226_hmm_guided_defensive_mixture_scale5"],
        "model.pf.particles": 500,
        "model.pf.seeds": 128,
        "model.pf.initial_position_spread_ft": 4.5,
        "model.pf.initial_rate_spread": 0.01,
        "model.pf.typewell_grid_step_ft": 0.2,
        "model.target_transition.momentum": 0.998,
        "model.target_transition.rate_noise": 0.002,
        "model.target_transition.position_noise": 0.005,
        "model.target_transition.minimum_md_delta": 1.0,
        "model.pf.rough_position": 0.1,
        "model.pf.rough_rate": 0.001,
        "model.pf.resample_threshold_fraction": 0.5,
        "model.pf.resampling": "systematic",
        "model.pf.emission_clip_z2": 600.0,
        "model.pf.gr_sigma_multiplier": 1.0,
        "model.pf.gr_sigma_clip": [10.0, 60.0],
        "model.pf.typewell_tvt_pad_ft": 100.0,
        "model.aggregation.temperature": 5.0,
        "execution.scientific_variants": 1,
        "execution.stage_0.hmm_signal_well_runs": 44,
        "execution.stage_0.candidate_pf_well_runs": 44,
        "execution.stage_0.parent_hmm_control_reruns": 0,
        "execution.stage_0.parent_pf_control_reruns": 0,
        "execution.stage_0.exp226_reruns": 0,
        "execution.stage_0.seed_well_trajectories": 5632,
        "execution.stage_0.particle_starts": 2816000,
        "execution.full.hmm_signal_well_runs": 773,
        "execution.full.candidate_pf_well_runs": 773,
        "execution.full.parent_hmm_control_reruns": 0,
        "execution.full.parent_pf_control_reruns": 0,
        "execution.full.exp226_reruns": 0,
        "execution.full.seed_well_trajectories": 98944,
        "execution.full.particle_starts": 49472000,
        "execution.full.well_shard_count": 4,
        "execution.reporting_folds": 5,
        "execution.lightgbm_configs": 0,
        "execution.trained_folds": 0,
        "execution.boosters": 0,
        "execution.models": 0,
        "execution.beam_well_runs": 0,
        "execution.gpu_runs": 0,
        "runtime.num_workers": 1,
        "runtime.numba_num_threads": 1,
        "runtime.device": "cpu",
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "execution.inference_approved": False,
        "execution.submission_approved": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp420 fixed contract mismatch: {key} must be {value!r}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp420 implementation approval must be recorded")
    hmm_schedule_contract(config)
    proposal_contract(config)
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_package_approved"))
        and bool(get_nested(config, "execution.kaggle_push_approved"))
        and (
            bool(get_nested(config, "execution.stage_0_run_approved"))
            or bool(get_nested(config, "execution.full_run_approved"))
        )
    ):
        raise RuntimeError("exp420 Kaggle package/push/train run is not approved")
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Truth-free raw input checks and deterministic LPT sharding


# %%
def build_raw_well_manifest(config: Mapping[str, Any], raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        visible = pd.read_csv(horizontal_path, usecols=["TVT_input"])
        suffix_rows = int(pd.to_numeric(visible["TVT_input"], errors="coerce").isna().sum())
        rows.append(
            {
                "well_id": str(well),
                "suffix_rows": suffix_rows,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    identity_sha = dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if (
        len(frame) != expected_wells
        or int(frame["suffix_rows"].sum()) != expected_rows
        or frame["well_id"].duplicated().any()
        or identity_sha != expected_sha
    ):
        raise ValueError("exp420 raw train well identity or suffix coverage mismatch")
    frame.attrs["raw_identity_sha256"] = identity_sha
    return frame


def assign_lpt_shards(manifest: pd.DataFrame, shard_count: int = SHARD_COUNT) -> pd.DataFrame:
    required = {"well_id", "suffix_rows"}
    if not required.issubset(manifest.columns):
        raise ValueError("LPT manifest is missing well_id or suffix_rows")
    if manifest["well_id"].astype(str).duplicated().any() or shard_count <= 0:
        raise ValueError("LPT manifest must have unique wells and a positive shard count")
    loads = [0] * shard_count
    assignments: dict[str, int] = {}
    ordered = manifest.assign(well_id=manifest["well_id"].astype(str)).sort_values(
        ["suffix_rows", "well_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[str(row.well_id)] = shard
        loads[shard] += int(row.suffix_rows)
    result = manifest.copy()
    result["shard_index"] = result["well_id"].astype(str).map(assignments).astype(np.int8)
    result = result.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    result.attrs["shard_suffix_rows"] = {str(index): int(load) for index, load in enumerate(loads)}
    return result


def input_spec(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.{key}") or {}
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a mapping")
    return value


def load_stage0_fixed44_manifest(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fixed32_spec = input_spec(config, "stage_0_fixed32")
    sentinel_spec = input_spec(config, "stage_0_pf_sentinel")
    fixed32_path = resolve_existing(
        str(fixed32_spec["filename"]),
        fixed32_spec.get("candidates", []),
    )
    sentinel_path = resolve_existing(
        str(sentinel_spec["filename"]),
        sentinel_spec.get("candidates", []),
    )
    observed_fixed32_sha = sha256_path(fixed32_path)
    observed_sentinel_sha = sha256_path(sentinel_path)
    if observed_fixed32_sha != str(fixed32_spec["expected_sha256"]):
        raise ValueError("exp420 fixed32 manifest SHA mismatch")
    if observed_sentinel_sha != str(sentinel_spec["expected_sha256"]):
        raise ValueError("exp420 PF sentinel manifest SHA mismatch")
    fixed32 = pd.read_csv(
        fixed32_path,
        usecols=["well"],
        dtype={"well": str},
    )
    sentinel = pd.read_csv(
        sentinel_path,
        usecols=["well"],
        dtype={"well": str},
    )
    if (
        len(fixed32) != int(fixed32_spec["expected_wells"])
        or fixed32["well"].nunique() != len(fixed32)
        or len(sentinel) != int(sentinel_spec["expected_wells"])
        or sentinel["well"].nunique() != len(sentinel)
    ):
        raise ValueError("exp420 fixed32/sentinel well counts changed")
    overlap = set(fixed32["well"]) & set(sentinel["well"])
    if overlap:
        raise ValueError(f"exp420 fixed44 manifests overlap: {sorted(overlap)}")
    manifest = pd.concat(
        [
            fixed32.assign(stage0_source="hmm_fixed32"),
            sentinel.assign(stage0_source="pf_sentinel12"),
        ],
        ignore_index=True,
    ).rename(columns={"well": "well_id"})
    manifest = manifest.sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    if (
        len(manifest) != int(
            get_nested(config, "validation.stage_0.expected_unique_wells")
        )
        or manifest["well_id"].nunique() != len(manifest)
    ):
        raise ValueError("exp420 fixed44 union contract changed")
    return manifest, {
        "fixed32_path": str(fixed32_path),
        "fixed32_sha256": observed_fixed32_sha,
        "sentinel_path": str(sentinel_path),
        "sentinel_sha256": observed_sentinel_sha,
        "fixed32_wells": len(fixed32),
        "sentinel_wells": len(sentinel),
        "overlap": 0,
        "union_wells": len(manifest),
        "union_logical_sha256": dataframe_content_sha(
            manifest,
            ["well_id", "stage0_source"],
        ),
        "prefreeze_columns_parsed": ["well"],
    }


def preflight_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and hash proposal/late-readout inputs without parsing late values."""

    keys = (
        "exp226_fold_safe_geometry",
        "exp404_frozen_predictions",
        "exp072_saved_likpf",
        "exp209_saved_hmm",
        "hidden_like_assignment",
        "exp410_target_wells",
        "exp408_persistent_episodes",
        "exp410_persistent_episodes",
    )
    specs = {key: input_spec(config, key) for key in keys}
    paths = {
        key: resolve_existing(str(spec["filename"]), spec.get("candidates", []))
        for key, spec in specs.items()
    }
    reports: dict[str, Any] = {}
    for key in (
        "exp226_fold_safe_geometry",
        "exp404_frozen_predictions",
        "exp072_saved_likpf",
        "exp209_saved_hmm",
    ):
        report = inspect_gzip_csv(paths[key])
        if report["decompressed_sha256"] != str(specs[key]["expected_decompressed_sha256"]):
            raise ValueError(f"{key} decompressed SHA mismatch")
        expected_raw = specs[key].get("expected_raw_sha256")
        if expected_raw and report["raw_sha256"] != str(expected_raw):
            raise ValueError(f"{key} raw gzip SHA mismatch")
        reports[key] = report
    for key in (
        "hidden_like_assignment",
        "exp410_target_wells",
        "exp408_persistent_episodes",
        "exp410_persistent_episodes",
    ):
        raw_sha = sha256_path(paths[key])
        if raw_sha != str(specs[key]["expected_sha256"]):
            raise ValueError(f"{key} raw SHA mismatch")
        reports[key] = {
            "path": str(paths[key]),
            "bytes": paths[key].stat().st_size,
            "raw_sha256": raw_sha,
            "columns": pd.read_csv(paths[key], nrows=0).columns.astype(str).tolist(),
        }
    ledger_specs = list(get_nested(config, "data.exp410_baseline_row_ledgers.shards") or [])
    if len(ledger_specs) != SHARD_COUNT:
        raise ValueError("exp420 requires four frozen exp410 row-ledger shards")
    ledger_paths: list[Path] = []
    ledger_reports: list[dict[str, Any]] = []
    for shard_index, raw_spec in enumerate(ledger_specs):
        spec = dict(raw_spec)
        path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
        report = inspect_gzip_csv(path)
        if report["raw_sha256"] != str(spec["expected_raw_sha256"]):
            raise ValueError(f"exp410 row ledger shard {shard_index} raw SHA mismatch")
        if report["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
            raise ValueError(f"exp410 row ledger shard {shard_index} decompressed SHA mismatch")
        ledger_paths.append(path)
        ledger_reports.append(report)
    required_columns = {
        "exp226_fold_safe_geometry": {
            "well_id",
            "row_idx",
            "suffix_offset",
            "tvt_geop",
            "tvt_pred",
            "gr_delta",
            "tvt_true",
            "error",
            "abs_error",
            "fold",
        },
        "exp404_frozen_predictions": {
            "id",
            "well_id",
            "row_idx",
            "suffix_offset",
            str(specs["exp404_frozen_predictions"]["control_column"]),
        },
        "exp072_saved_likpf": {
            "id",
            str(specs["exp072_saved_likpf"]["residual_column"]),
            str(specs["exp072_saved_likpf"]["anchor_column"]),
        },
        "exp209_saved_hmm": {
            "id",
            str(specs["exp209_saved_hmm"]["prediction_column"]),
        },
        "hidden_like_assignment": {
            "well_id",
            *[str(value) for value in specs["hidden_like_assignment"]["role_columns"].values()],
        },
        "exp410_target_wells": {"well", "episodes", "episode_rows", "suffix_rows"},
        "exp408_persistent_episodes": {
            "episode_id",
            "well",
            "start_row_idx",
            "end_row_idx_exclusive",
            "rows",
        },
        "exp410_persistent_episodes": {
            "episode_id",
            "well",
            "start_row_idx",
            "end_row_idx_exclusive",
            "rows",
        },
    }
    for key, required in required_columns.items():
        missing = sorted(required - set(reports[key]["columns"]))
        if missing:
            raise ValueError(f"{key} missing required columns: {missing}")
    ledger_required = {
        "well",
        "row_idx",
        "predictive_truth_support_fraction",
    }
    for shard_index, report in enumerate(ledger_reports):
        missing = sorted(ledger_required - set(report["columns"]))
        if missing:
            raise ValueError(f"exp410 row ledger shard {shard_index} missing columns: {missing}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if int(reports["exp226_fold_safe_geometry"]["data_rows"]) != expected_rows:
        raise ValueError("exp226 geometry row count mismatch")
    if int(reports["exp404_frozen_predictions"]["data_rows"]) != expected_rows:
        raise ValueError("exp404 frozen prediction row count mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "exp410_row_ledger_paths": [str(value) for value in ledger_paths],
        "reports": reports,
        "exp410_row_ledger_reports": ledger_reports,
        "proposal_columns_parsed_before_freeze": [
            "well_id",
            "row_idx",
            "suffix_offset",
            "tvt_geop",
        ],
        "truth_or_reporting_values_parsed_before_freeze": {
            "unknown_suffix_tvt_rows": 0,
            "control_prediction_rows": 0,
            "exp226_final_rows": 0,
            "fold_rows": 0,
            "hidden_like_role_rows": 0,
            "exp410_scope_rows": 0,
            "exp408_scope_rows": 0,
            "physical_anchor_rows": 0,
        },
    }


def load_fold_safe_geometry(
    path: str | Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Read the exact proposal allowlist; forbidden exp226 columns stay unread."""

    spec = input_spec(config, "exp226_fold_safe_geometry")
    safe_columns = [str(value) for value in spec["proposal_columns"]]
    expected = ["well_id", "row_idx", "suffix_offset", "tvt_geop"]
    if safe_columns != expected:
        raise ValueError(f"exp420 geometry proposal allowlist changed: {safe_columns}")
    geometry = pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"well_id": str},
        compression="gzip",
    )
    if list(geometry.columns) != safe_columns:
        geometry = geometry[safe_columns]
    geometry["row_idx"] = pd.to_numeric(geometry["row_idx"], errors="raise").astype(np.int64)
    geometry["suffix_offset"] = pd.to_numeric(geometry["suffix_offset"], errors="raise").astype(
        np.int64
    )
    geometry["tvt_geop"] = pd.to_numeric(geometry["tvt_geop"], errors="raise").astype(np.float64)
    if (
        geometry.duplicated(["well_id", "row_idx"]).any()
        or not np.isfinite(geometry["tvt_geop"].to_numpy(np.float64)).all()
    ):
        raise ValueError("exp226 geometry proposal rows are duplicated or non-finite")
    return geometry


# %% [markdown]
# ## 5. Exact untreated-HMM forward innovation schedule
#
# The forward filter is exp209/exp411-compatible, but no directional treatment
# is applied to its transition and no backward pass or HMM prediction is
# computed. A trigger observed after filtering row `t` first activates the PF
# proposal for transition `t + 1`. Only `active` and `direction` cross the
# schedule freeze boundary.


# %%
def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> float:
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    if int(valid.sum()) < int(min_valid_steps):
        return float(fallback_rate)
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    return rate if math.isfinite(rate) else float(fallback_rate)


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    signal: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("unknown-suffix TVT reached exp420 HMM schedule preparation")
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    evaluation = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or evaluation.empty:
        raise ValueError("exp420 HMM requires a visible prefix and non-empty suffix")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known["GR"].fillna(0.0).to_numpy(np.float64) - typewell_at_known
    gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
    step = float(signal["step"])
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    band_pad = float(signal["band_pad"])
    grid_minimum = max(float(typewell_tvt.min()) - 40.0, last_tvt - band_pad)
    grid_maximum = min(float(typewell_tvt.max()) + 40.0, last_tvt + band_pad)
    grid = np.arange(grid_minimum, grid_maximum + step, step, dtype=np.float64)
    grid_gr = np.interp(grid, typewell_tvt, typewell_gr)
    md = evaluation["MD"].to_numpy(np.float64)
    z = evaluation["Z"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    gr = interpolated_gr[evaluation.index.to_numpy(np.int64)]
    delta_md = np.maximum(
        np.diff(np.concatenate([[float(last["MD"])], md])),
        1.0,
    )
    delta_z = np.diff(np.concatenate([[float(last["Z"])], z]))
    zscore = (gr[:, None] - grid_gr[None, :]) / gr_sigma
    emission_log_likelihood = (-0.5 * np.minimum(zscore**2, 600.0)).astype(
        np.float32
    )
    initial_rate = robust_initial_rate(known)
    span = max(float(signal["rate_span"]), abs(initial_rate) + 0.04)
    rates = np.linspace(
        -span,
        span,
        int(signal["n_rates"]),
        dtype=np.float64,
    )
    return {
        "emission_log_likelihood": emission_log_likelihood,
        "delta_md": delta_md,
        "delta_z": delta_z,
        "rates": rates,
        "position_step": step,
        "start_position_index": float((last_tvt - grid_minimum) / step),
        "initial_rate": initial_rate,
        "eval_indices": evaluation.index.to_numpy(np.int64),
        "raw_gr_missing": evaluation["GR"].isna().to_numpy(bool),
        "gr_sigma": gr_sigma,
    }


@njit(cache=True, nogil=True)
def untreated_rate_kernel_probabilities(
    rates: np.ndarray,
    delta_md: float,
    sigma_rate: float,
    momentum: float,
) -> np.ndarray:
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    rate_variance_cells = (sigma_rate * np.sqrt(delta_md) / rate_step) ** 2
    kernel = np.empty((rate_count, 3), np.float64)
    for rate_index in range(rate_count):
        mean_move = (
            -(1.0 - momentum)
            * rates[rate_index]
            * delta_md
            / rate_step
        )
        probability_plus = max(
            0.5 * (rate_variance_cells + mean_move),
            1.0e-12,
        )
        probability_minus = max(
            0.5 * (rate_variance_cells - mean_move),
            1.0e-12,
        )
        moving = probability_plus + probability_minus
        if moving > 0.9:
            probability_plus *= 0.9 / moving
            probability_minus *= 0.9 / moving
        kernel[rate_index, 0] = probability_minus
        kernel[rate_index, 1] = 1.0 - probability_plus - probability_minus
        kernel[rate_index, 2] = probability_plus
    return kernel


@njit(cache=True, nogil=True)
def _untreated_hmm_forward_schedule(
    emission_log_likelihood: np.ndarray,
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    position_step: float,
    rates: np.ndarray,
    sigma_rate: float,
    sigma_position: float,
    start_position_index: float,
    start_position_sigma: float,
    initial_rate: float,
    initial_rate_sigma: float,
    emission_power: float,
    momentum: float,
    innovation_scale: float,
    drift_allowance: float,
    positive_threshold: float,
    negative_threshold: float,
    tie_tolerance: float,
    activation_transitions: int,
    refractory_rows: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    time_count, position_count = emission_log_likelihood.shape
    rate_count = len(rates)
    negative_infinity = np.float32(-1.0e18)
    previous = np.full((position_count, rate_count), negative_infinity, np.float32)
    for position_index in range(position_count):
        position_delta = (position_index - start_position_index) * position_step
        initial_position_log_probability = -0.5 * (
            position_delta / start_position_sigma
        ) ** 2
        if initial_position_log_probability < -60.0:
            continue
        for rate_index in range(rate_count):
            rate_delta = (
                rates[rate_index] - initial_rate
            ) / initial_rate_sigma
            previous[position_index, rate_index] = np.float32(
                initial_position_log_probability - 0.5 * rate_delta * rate_delta
            )

    rate_propagated = np.empty((position_count, rate_count), np.float32)
    predictive = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)
    predictive_rate_mean = np.empty(time_count, np.float64)
    filtered_rate_mean = np.empty(time_count, np.float64)
    innovation = np.empty(time_count, np.float64)
    positive_cusum = np.empty(time_count, np.float64)
    negative_cusum = np.empty(time_count, np.float64)
    trigger_direction = np.zeros(time_count, np.int8)
    active_direction = np.zeros(time_count, np.int8)
    maximum_normalization_error = 0.0
    log_likelihood = 0.0
    positive_accumulator = 0.0
    negative_accumulator = 0.0
    activation_remaining = 0
    refractory_remaining = 0
    scheduled_direction = 0

    for time_index in range(time_count):
        row_started_active = activation_remaining > 0
        row_started_refractory = refractory_remaining > 0
        if row_started_active:
            active_direction[time_index] = np.int8(scheduled_direction)
            activation_remaining -= 1
        kernel = untreated_rate_kernel_probabilities(
            rates,
            delta_md[time_index],
            sigma_rate,
            momentum,
        )
        log_kernel = np.log(kernel)
        for position_index in range(position_count):
            for next_rate_index in range(rate_count):
                best = negative_infinity
                first_rate_index = max(next_rate_index - 1, 0)
                last_rate_index = min(next_rate_index + 1, rate_count - 1)
                for rate_index in range(first_rate_index, last_rate_index + 1):
                    value = (
                        previous[position_index, rate_index]
                        + log_kernel[
                            rate_index,
                            next_rate_index - rate_index + 1,
                        ]
                    )
                    if value > best:
                        best = value
                if best > negative_infinity / 2:
                    total = 0.0
                    for rate_index in range(
                        first_rate_index,
                        last_rate_index + 1,
                    ):
                        total += np.exp(
                            previous[position_index, rate_index]
                            + log_kernel[
                                rate_index,
                                next_rate_index - rate_index + 1,
                            ]
                            - best
                        )
                    rate_propagated[position_index, next_rate_index] = np.float32(
                        best + np.log(total)
                    )
                else:
                    rate_propagated[position_index, next_rate_index] = (
                        negative_infinity
                    )

        effective_sigma_position = max(sigma_position, 0.35 * position_step)
        for next_rate_index in range(rate_count):
            position_mean = (
                rates[next_rate_index] * delta_md[time_index]
                - delta_z[time_index]
            )
            center_offset = int(
                np.floor(position_mean / position_step + 0.5)
            )
            position_log_kernel = np.empty(5)
            for kernel_index in range(5):
                delta = (
                    (center_offset - 2 + kernel_index) * position_step
                    - position_mean
                )
                position_log_kernel[kernel_index] = -0.5 * (
                    delta / effective_sigma_position
                ) ** 2
            kernel_maximum = np.max(position_log_kernel)
            position_log_kernel -= kernel_maximum + np.log(
                np.sum(np.exp(position_log_kernel - kernel_maximum))
            )
            for next_position_index in range(position_count):
                best = negative_infinity
                for kernel_index in range(5):
                    position_index = next_position_index - (
                        center_offset - 2 + kernel_index
                    )
                    if 0 <= position_index < position_count:
                        value = (
                            rate_propagated[position_index, next_rate_index]
                            + position_log_kernel[kernel_index]
                        )
                        if value > best:
                            best = value
                if best > negative_infinity / 2:
                    total = 0.0
                    for kernel_index in range(5):
                        position_index = next_position_index - (
                            center_offset - 2 + kernel_index
                        )
                        if 0 <= position_index < position_count:
                            total += np.exp(
                                rate_propagated[
                                    position_index,
                                    next_rate_index,
                                ]
                                + position_log_kernel[kernel_index]
                                - best
                            )
                    predictive[next_position_index, next_rate_index] = (
                        np.float32(best + np.log(total))
                    )
                    current[next_position_index, next_rate_index] = np.float32(
                        best
                        + np.log(total)
                        + emission_power
                        * emission_log_likelihood[
                            time_index,
                            next_position_index,
                        ]
                    )
                else:
                    predictive[next_position_index, next_rate_index] = (
                        negative_infinity
                    )
                    current[next_position_index, next_rate_index] = (
                        negative_infinity
                    )

        predictive_best = np.max(predictive)
        filtered_best = np.max(current)
        predictive_total = 0.0
        filtered_total = 0.0
        predictive_first_moment = 0.0
        filtered_first_moment = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_probability = np.exp(
                    predictive[position_index, rate_index] - predictive_best
                )
                filtered_probability = np.exp(
                    current[position_index, rate_index] - filtered_best
                )
                predictive_total += predictive_probability
                filtered_total += filtered_probability
                predictive_first_moment += (
                    predictive_probability * rates[rate_index]
                )
                filtered_first_moment += (
                    filtered_probability * rates[rate_index]
                )
        predictive_rate_mean[time_index] = (
            predictive_first_moment / predictive_total
        )
        filtered_rate_mean[time_index] = filtered_first_moment / filtered_total
        predictive_check = 0.0
        filtered_check = 0.0
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                predictive_check += (
                    np.exp(
                        predictive[position_index, rate_index]
                        - predictive_best
                    )
                    / predictive_total
                )
                filtered_check += (
                    np.exp(
                        current[position_index, rate_index] - filtered_best
                    )
                    / filtered_total
                )
                previous[position_index, rate_index] = current[
                    position_index,
                    rate_index,
                ]
        maximum_normalization_error = max(
            maximum_normalization_error,
            abs(predictive_check - 1.0),
            abs(filtered_check - 1.0),
        )
        log_likelihood = float(filtered_best) + np.log(filtered_total)

        innovation_value = (
            filtered_rate_mean[time_index]
            - predictive_rate_mean[time_index]
        ) / innovation_scale
        innovation[time_index] = innovation_value
        positive_accumulator = max(
            0.0,
            positive_accumulator + innovation_value - drift_allowance,
        )
        negative_accumulator = max(
            0.0,
            negative_accumulator - innovation_value - drift_allowance,
        )
        if row_started_active:
            if activation_remaining == 0:
                refractory_remaining = refractory_rows
        elif row_started_refractory:
            refractory_remaining -= 1
        else:
            positive_hit = positive_accumulator >= positive_threshold
            negative_hit = negative_accumulator >= negative_threshold
            direction = 0
            if positive_hit and negative_hit:
                difference = positive_accumulator - negative_accumulator
                if difference > tie_tolerance:
                    direction = 1
                elif difference < -tie_tolerance:
                    direction = -1
            elif positive_hit:
                direction = 1
            elif negative_hit:
                direction = -1
            if direction != 0:
                trigger_direction[time_index] = np.int8(direction)
                scheduled_direction = direction
                activation_remaining = activation_transitions
                positive_accumulator = 0.0
                negative_accumulator = 0.0
        positive_cusum[time_index] = positive_accumulator
        negative_cusum[time_index] = negative_accumulator
    return (
        predictive_rate_mean,
        filtered_rate_mean,
        innovation,
        positive_cusum,
        negative_cusum,
        trigger_direction,
        active_direction,
        maximum_normalization_error,
        log_likelihood,
    )


def build_untreated_hmm_schedule(
    well: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = hmm_schedule_contract(config)
    signal = dict(get_nested(config, "model.hmm_signal") or {})
    prepared = prepare_hmm_inputs(horizontal, typewell, signal)
    outputs = _untreated_hmm_forward_schedule(
        prepared["emission_log_likelihood"],
        prepared["delta_md"],
        prepared["delta_z"],
        float(prepared["position_step"]),
        prepared["rates"],
        float(signal["sig_r"]),
        float(signal["sig_p"]),
        float(prepared["start_position_index"]),
        float(signal["start_sig"]),
        float(prepared["initial_rate"]),
        float(signal["r0_sig"]),
        float(signal["lam"]),
        float(signal["momentum"]),
        float(signal["rate_step"]),
        float(contract["drift_allowance_rate_cells"]),
        float(contract["positive_threshold_rate_cells"]),
        float(contract["negative_threshold_rate_cells"]),
        float(signal["tie_tolerance"]),
        int(contract["activation_transitions"]),
        int(contract["refractory_rows"]),
    )
    (
        predictive_rate_mean,
        filtered_rate_mean,
        innovation,
        positive_cusum,
        negative_cusum,
        trigger_direction,
        active_direction,
        maximum_normalization_error,
        log_likelihood,
    ) = outputs
    eval_indices = prepared["eval_indices"]
    schedule = pd.DataFrame(
        {
            "well_id": str(well),
            "row_idx": eval_indices,
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "predictive_rate_mean": predictive_rate_mean,
            "filtered_rate_mean": filtered_rate_mean,
            "innovation_rate_cells": innovation,
            "positive_cusum": positive_cusum,
            "negative_cusum": negative_cusum,
            "trigger_direction": trigger_direction,
            "active": active_direction != 0,
            "direction": active_direction,
        }
    )
    logical_columns = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "active",
        "direction",
    ]
    diagnostics = {
        "schedule_contract_sha256": contract["schedule_contract_sha256"],
        "schedule_rows": len(schedule),
        "schedule_logical_sha256": dataframe_content_sha(
            schedule,
            logical_columns,
        ),
        "trigger_count": int((trigger_direction != 0).sum()),
        "active_rows": int((active_direction != 0).sum()),
        "active_fraction": float((active_direction != 0).mean()),
        "active_direction_positive_rows": int((active_direction > 0).sum()),
        "active_direction_negative_rows": int((active_direction < 0).sum()),
        "maximum_forward_normalization_error": float(
            maximum_normalization_error
        ),
        "untreated_hmm_forward_log_likelihood": float(log_likelihood),
        "hmm_grid_positions": int(
            prepared["emission_log_likelihood"].shape[1]
        ),
        "hmm_rate_states": len(prepared["rates"]),
        "hmm_rate_step_actual": float(
            prepared["rates"][1] - prepared["rates"][0]
        ),
        "hmm_prefix_gr_sigma": float(prepared["gr_sigma"]),
    }
    if (
        not np.isfinite(
            schedule[
                [
                    "predictive_rate_mean",
                    "filtered_rate_mean",
                    "innovation_rate_cells",
                ]
            ].to_numpy(np.float64)
        ).all()
        or not schedule["direction"].isin([-1, 0, 1]).all()
        or not schedule.loc[~schedule["active"], "direction"].eq(0).all()
    ):
        raise RuntimeError(f"{well}: invalid untreated-HMM schedule")
    return schedule, diagnostics


# %% [markdown]
# ## 6. Exact exp072 PF input preparation and fold-safe geometry rate


# %%
@dataclass
class TruthAccessLedger:
    prediction_frozen: bool = False
    unknown_suffix_tvt_rows_before_freeze: int = 0
    control_prediction_rows_before_freeze: int = 0
    exp226_final_rows_before_freeze: int = 0
    fold_rows_before_freeze: int = 0
    hidden_like_role_rows_before_freeze: int = 0
    exp410_scope_rows_before_freeze: int = 0
    exp408_scope_rows_before_freeze: int = 0
    physical_anchor_rows_before_freeze: int = 0
    unknown_suffix_tvt_rows_after_freeze: int = 0
    control_prediction_rows_after_freeze: int = 0
    exp226_final_rows_after_freeze: int = 0
    fold_rows_after_freeze: int = 0
    hidden_like_role_rows_after_freeze: int = 0
    exp410_scope_rows_after_freeze: int = 0
    exp408_scope_rows_after_freeze: int = 0
    physical_anchor_rows_after_freeze: int = 0

    def require_frozen(self) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("late reporting input requires a frozen prediction")

    def mark_frozen(self) -> None:
        if any(self.report()["before_freeze"].values()):
            raise RuntimeError("truth/reporting values were accessed before prediction freeze")
        self.prediction_frozen = True

    def report(self) -> dict[str, Any]:
        return {
            "prediction_frozen": self.prediction_frozen,
            "before_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_before_freeze,
                "control_prediction_rows": self.control_prediction_rows_before_freeze,
                "exp226_final_rows": self.exp226_final_rows_before_freeze,
                "fold_rows": self.fold_rows_before_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_before_freeze,
                "exp410_scope_rows": self.exp410_scope_rows_before_freeze,
                "exp408_scope_rows": self.exp408_scope_rows_before_freeze,
                "physical_anchor_rows": self.physical_anchor_rows_before_freeze,
            },
            "after_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_after_freeze,
                "control_prediction_rows": self.control_prediction_rows_after_freeze,
                "exp226_final_rows": self.exp226_final_rows_after_freeze,
                "fold_rows": self.fold_rows_after_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_after_freeze,
                "exp410_scope_rows": self.exp410_scope_rows_after_freeze,
                "exp408_scope_rows": self.exp408_scope_rows_after_freeze,
                "physical_anchor_rows": self.physical_anchor_rows_after_freeze,
            },
        }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    frame = frame[["MD", "Z", "GR", "TVT_input"]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "Z"]].isna().any().any():
        raise ValueError(f"{well}: MD/Z must be finite")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    if len(frame) < 2 or not np.isfinite(frame["TVT"].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    typewell_mean = float(frame["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(typewell_mean)
    return frame


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
    return {
        "raw_scale": raw_scale,
        "base_scale": float(np.clip(raw_scale, clip[0], clip[1])),
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
    }


def exp072_initial_rate(horizontal: pd.DataFrame, *, tail_rows: int = 30) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()].tail(tail_rows)
    delta_tvt = np.diff(known["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(known["Z"].to_numpy(np.float64))
    delta_md = np.diff(known["MD"].to_numpy(np.float64))
    valid = delta_md > 0
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    grid_step: float,
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires non-empty known prefix and unknown suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    last_known_tvt = float(last_known["TVT_input"])
    last_known_md = float(last_known["MD"])
    scale_audit = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr.mean()))
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    if not np.isfinite(eval_gr).all():
        raise ValueError("evaluation GR interpolation is not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": evaluation["Z"].to_numpy(np.float64),
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - last_known_md,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_known_tvt + float(last_known["Z"]),
        "initial_rate": exp072_initial_rate(horizontal),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "scale_audit": scale_audit,
    }


# %% [markdown]
# ## 7. Scheduled defensive-mixture proposal and exact importance correction


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


@njit(cache=True)
def _normal_logpdf(value: float, mean: float, sigma: float) -> float:
    zscore = (value - mean) / sigma
    return -0.5 * zscore * zscore - np.log(sigma) - 0.9189385332046727


@njit(cache=True)
def _logsumexp7(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
) -> float:
    maximum = max(a, b, c, d, e, f, g)
    return maximum + np.log(
        np.exp(a - maximum)
        + np.exp(b - maximum)
        + np.exp(c - maximum)
        + np.exp(d - maximum)
        + np.exp(e - maximum)
        + np.exp(f - maximum)
        + np.exp(g - maximum)
    )


@njit(cache=True)
def scheduled_mixture_importance_ratio(
    sampled_rate: float,
    target_mean: float,
    geometry_rate: float,
    schedule_direction: int,
    hmm_rate_step: float,
    rate_sigma: float,
    target_weight: float,
    geometry_weights: np.ndarray,
    hmm_weights: np.ndarray,
    geometry_sigma_multipliers: np.ndarray,
) -> float:
    """Return p0/q without clipping, evaluated in log space."""

    log_p0 = _normal_logpdf(sampled_rate, target_mean, rate_sigma)
    if target_weight >= 1.0:
        return 1.0
    inactive = schedule_direction == 0 or hmm_weights.sum() <= 0.0
    if inactive:
        first = (
            np.log(target_weight)
            + log_p0
        )
        second = np.log(geometry_weights[0]) + _normal_logpdf(
            sampled_rate,
            geometry_rate,
            rate_sigma * geometry_sigma_multipliers[0],
        )
        third = np.log(geometry_weights[1]) + _normal_logpdf(
            sampled_rate,
            geometry_rate,
            rate_sigma * geometry_sigma_multipliers[1],
        )
        fourth = np.log(geometry_weights[2]) + _normal_logpdf(
            sampled_rate,
            geometry_rate,
            rate_sigma * geometry_sigma_multipliers[2],
        )
        maximum = max(first, second, third, fourth)
        log_q = maximum + np.log(
            np.exp(first - maximum)
            + np.exp(second - maximum)
            + np.exp(third - maximum)
            + np.exp(fourth - maximum)
        )
        return np.exp(log_p0 - log_q)
    hmm_center = target_mean + schedule_direction * hmm_rate_step
    log_q = _logsumexp7(
        np.log(target_weight) + log_p0,
        np.log(geometry_weights[0])
        + _normal_logpdf(
            sampled_rate,
            geometry_rate,
            rate_sigma * geometry_sigma_multipliers[0],
        ),
        np.log(geometry_weights[1])
        + _normal_logpdf(
            sampled_rate,
            geometry_rate,
            rate_sigma * geometry_sigma_multipliers[1],
        ),
        np.log(geometry_weights[2])
        + _normal_logpdf(
            sampled_rate,
            geometry_rate,
            rate_sigma * geometry_sigma_multipliers[2],
        ),
        np.log(hmm_weights[0])
        + _normal_logpdf(
            sampled_rate,
            hmm_center,
            rate_sigma * geometry_sigma_multipliers[0],
        ),
        np.log(hmm_weights[1])
        + _normal_logpdf(
            sampled_rate,
            hmm_center,
            rate_sigma * geometry_sigma_multipliers[1],
        ),
        np.log(hmm_weights[2])
        + _normal_logpdf(
            sampled_rate,
            hmm_center,
            rate_sigma * geometry_sigma_multipliers[2],
        ),
    )
    return np.exp(log_p0 - log_q)


@njit(cache=True, nogil=True)
def _pf_guided_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    geometry_rate_v: np.ndarray,
    schedule_direction_v: np.ndarray,
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
    target_weight: float,
    inactive_geometry_weights: np.ndarray,
    active_geometry_weights: np.ndarray,
    active_hmm_weights: np.ndarray,
    geometry_sigma_multipliers: np.ndarray,
    hmm_rate_step: float,
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
    """Exp404 target model with only the importance-corrected proposal changed."""

    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    importance_minimum = np.full(seeds, 1.0e300)
    importance_maximum = np.zeros(seeds)
    importance_sum = np.zeros(seeds)
    component_counts = np.zeros((seeds, 7), np.int64)
    predictive_support_min = np.empty((seeds, rows), np.float32)
    predictive_support_max = np.empty((seeds, rows), np.float32)
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
            schedule_direction = int(schedule_direction_v[row])
            hmm_active = (
                schedule_direction != 0
                and active_hmm_weights.sum() > 0.0
            )
            geometry_weights = (
                active_geometry_weights
                if hmm_active
                else inactive_geometry_weights
            )
            hmm_weights = active_hmm_weights if hmm_active else np.zeros(3)
            for particle in range(particles):
                target_mean = momentum * rate[particle]
                if target_weight >= 1.0:
                    # The parity mode consumes the exact exp404 RNG sequence.
                    sampled_rate = target_mean + rate_noise * np.random.randn()
                    importance = 1.0
                    component_counts[seed_index, 0] += 1
                else:
                    component_draw = np.random.uniform()
                    gaussian_draw = np.random.randn()
                    hmm_center = (
                        target_mean + schedule_direction * hmm_rate_step
                    )
                    if component_draw < target_weight:
                        sampled_rate = target_mean + rate_noise * gaussian_draw
                        component_counts[seed_index, 0] += 1
                    elif component_draw < target_weight + geometry_weights[0]:
                        sampled_rate = (
                            geometry_rate_v[row]
                            + rate_noise * geometry_sigma_multipliers[0] * gaussian_draw
                        )
                        component_counts[seed_index, 1] += 1
                    elif component_draw < target_weight + geometry_weights[0] + geometry_weights[1]:
                        sampled_rate = (
                            geometry_rate_v[row]
                            + rate_noise * geometry_sigma_multipliers[1] * gaussian_draw
                        )
                        component_counts[seed_index, 2] += 1
                    elif (
                        component_draw
                        < target_weight
                        + geometry_weights[0]
                        + geometry_weights[1]
                        + geometry_weights[2]
                    ):
                        sampled_rate = (
                            geometry_rate_v[row]
                            + rate_noise * geometry_sigma_multipliers[2] * gaussian_draw
                        )
                        component_counts[seed_index, 3] += 1
                    elif (
                        component_draw
                        < target_weight
                        + geometry_weights[0]
                        + geometry_weights[1]
                        + geometry_weights[2]
                        + hmm_weights[0]
                    ):
                        sampled_rate = (
                            hmm_center
                            + rate_noise
                            * geometry_sigma_multipliers[0]
                            * gaussian_draw
                        )
                        component_counts[seed_index, 4] += 1
                    elif (
                        component_draw
                        < target_weight
                        + geometry_weights[0]
                        + geometry_weights[1]
                        + geometry_weights[2]
                        + hmm_weights[0]
                        + hmm_weights[1]
                    ):
                        sampled_rate = (
                            hmm_center
                            + rate_noise
                            * geometry_sigma_multipliers[1]
                            * gaussian_draw
                        )
                        component_counts[seed_index, 5] += 1
                    else:
                        sampled_rate = (
                            hmm_center
                            + rate_noise
                            * geometry_sigma_multipliers[2]
                            * gaussian_draw
                        )
                        component_counts[seed_index, 6] += 1
                    importance = scheduled_mixture_importance_ratio(
                        sampled_rate,
                        target_mean,
                        geometry_rate_v[row],
                        schedule_direction,
                        hmm_rate_step,
                        rate_noise,
                        target_weight,
                        geometry_weights,
                        hmm_weights,
                        geometry_sigma_multipliers,
                    )
                rate[particle] = sampled_rate
                position[particle] += rate[particle] * delta_md + position_noise * np.random.randn()
                weights[particle] *= importance
                importance_sum[seed_index] += importance
                if importance < importance_minimum[seed_index]:
                    importance_minimum[seed_index] = importance
                if importance > importance_maximum[seed_index]:
                    importance_maximum[seed_index] = importance
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                    position_clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                    position_clip_counts[seed_index] += 1
                position[particle] = tvt_value + z_v[row]
            minimum_support = 1.0e300
            maximum_support = -1.0e300
            for particle in range(particles):
                tvt_value = position[particle] - z_v[row]
                if tvt_value < minimum_support:
                    minimum_support = tvt_value
                if tvt_value > maximum_support:
                    maximum_support = tvt_value
            predictive_support_min[seed_index, row] = minimum_support
            predictive_support_max[seed_index, row] = maximum_support
            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(
                    grid_gr,
                    position[particle] - z_v[row],
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
            inverse_ess = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
            effective_sample_size = 1.0 / inverse_ess
            if effective_sample_size < minimum_ess[seed_index]:
                minimum_ess[seed_index] = effective_sample_size
            if effective_sample_size < resample_fraction * particles:
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
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles
                resampling_counts[seed_index] += 1
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
        importance_minimum,
        importance_maximum,
        importance_sum,
        component_counts,
        predictive_support_min,
        predictive_support_max,
    )


def aggregate_seed_predictions(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / temperature)
    weights /= float(weights.sum())
    return (weights[:, None] * predictions).sum(axis=0), weights


def geometry_surface_rate(
    prepared: Mapping[str, Any],
    geometry_rows: pd.DataFrame,
) -> np.ndarray:
    expected_rows = prepared["eval_indices"].astype(np.int64)
    ordered = geometry_rows.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        ordered["row_idx"].to_numpy(np.int64), expected_rows
    ) or not np.array_equal(
        ordered["suffix_offset"].to_numpy(np.int64),
        np.arange(len(expected_rows), dtype=np.int64),
    ):
        raise ValueError("exp226 geometry identity does not match the raw suffix")
    surface = ordered["tvt_geop"].to_numpy(np.float64) + prepared["eval_z"].astype(np.float64)
    rate = np.empty(len(surface), dtype=np.float64)
    previous_surface = float(prepared["last_known_position"])
    previous_md = float(prepared["eval_md"][0] - 1.0)
    for row in range(len(surface)):
        delta_md = max(float(prepared["eval_md"][row] - previous_md), 1.0)
        rate[row] = (float(surface[row]) - previous_surface) / delta_md
        previous_surface = float(surface[row])
        previous_md = float(prepared["eval_md"][row])
    if not np.isfinite(rate).all():
        raise ValueError("exp226 geometry surface rate is not finite")
    return rate


def run_guided_likelihood_pf(
    prepared: Mapping[str, Any],
    geometry_rate: np.ndarray,
    schedule_direction: np.ndarray,
    *,
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
    target_weight: float,
    inactive_geometry_weights: Sequence[float],
    active_geometry_weights: Sequence[float],
    active_hmm_weights: Sequence[float],
    geometry_sigma_multipliers: Sequence[float],
    hmm_rate_step: float,
    temperature: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    started = time.time()
    outputs = _pf_guided_allseeds(
        prepared["eval_md"],
        prepared["eval_z"],
        prepared["eval_gr"],
        geometry_rate,
        schedule_direction.astype(np.int8),
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["base_scale"]),
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
        float(target_weight),
        np.asarray(inactive_geometry_weights, dtype=np.float64),
        np.asarray(active_geometry_weights, dtype=np.float64),
        np.asarray(active_hmm_weights, dtype=np.float64),
        np.asarray(geometry_sigma_multipliers, dtype=np.float64),
        float(hmm_rate_step),
    )
    (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        clip_counts,
        importance_minimum,
        importance_maximum,
        importance_sum,
        component_counts,
        support_minimum,
        support_maximum,
    ) = outputs
    candidate, seed_weights = aggregate_seed_predictions(
        predictions,
        log_likelihoods,
        temperature=float(temperature),
    )
    importance_count = float(particles * len(candidate))
    total_components = component_counts.sum(axis=0).astype(np.float64)
    total_components /= float(total_components.sum())
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "seed_loglik_mean_per_row": float(log_likelihoods.mean()) / len(candidate),
        "seed_loglik_best_per_row": float(log_likelihoods.max()) / len(candidate),
        "seed_loglik_spread": float(log_likelihoods.std()),
        "resampling_count_total": int(resampling_counts.sum()),
        "resampling_count_min": int(resampling_counts.min()),
        "resampling_count_max": int(resampling_counts.max()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "position_clip_count_total": int(clip_counts.sum()),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
        "importance_ratio_minimum": float(importance_minimum.min()),
        "importance_ratio_maximum": float(importance_maximum.max()),
        "importance_ratio_mean": float(np.mean(importance_sum / importance_count)),
        "importance_ratio_finite": bool(
            np.isfinite(importance_minimum).all()
            and np.isfinite(importance_maximum).all()
            and np.isfinite(importance_sum).all()
        ),
        "component_fraction_target": float(total_components[0]),
        "component_fraction_geometry_1x": float(total_components[1]),
        "component_fraction_geometry_4x": float(total_components[2]),
        "component_fraction_geometry_16x": float(total_components[3]),
        "component_fraction_hmm_1x": float(total_components[4]),
        "component_fraction_hmm_4x": float(total_components[5]),
        "component_fraction_hmm_16x": float(total_components[6]),
        "schedule_active_rows": int((schedule_direction != 0).sum()),
        "schedule_active_fraction": float((schedule_direction != 0).mean()),
        "seed_weight_minimum": float(seed_weights.min()),
        "seed_weight_maximum": float(seed_weights.max()),
        "seed_weight_sum": float(seed_weights.sum()),
        "seed_aggregation_temperature": float(temperature),
    }
    if (
        not diagnostics["importance_ratio_finite"]
        or diagnostics["importance_ratio_maximum"] > 2.000000000001
    ):
        raise RuntimeError("exp420 importance-ratio contract failed")
    return (
        candidate,
        diagnostics,
        support_minimum.T.copy(),
        support_maximum.T.copy(),
    )


# %% [markdown]
# ## 8. Stage-0/full shard candidate generation and freeze


# %%
def warm_up_pf_kernel() -> None:
    _pf_guided_allseeds(
        np.linspace(1.0, 8.0, 8),
        np.zeros(8),
        np.full(8, 50.0),
        np.zeros(8),
        np.zeros(8, dtype=np.int8),
        np.linspace(45.0, 55.0, 100),
        0.0,
        0.2,
        20.0,
        50.0,
        0.0,
        8,
        2,
        1,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        0.5,
        np.asarray([1.0 / 6.0] * 3, dtype=np.float64),
        np.asarray([1.0 / 12.0] * 3, dtype=np.float64),
        np.asarray([1.0 / 12.0] * 3, dtype=np.float64),
        np.asarray([1.0, 4.0, 16.0], dtype=np.float64),
        0.005,
    )


def synthetic_kernel_parity_report() -> dict[str, Any]:
    rows = 8
    common = (
        np.linspace(1.0, 8.0, rows),
        np.linspace(0.0, 0.7, rows),
        np.linspace(48.0, 54.0, rows),
        np.linspace(-0.01, 0.02, rows),
    )
    schedule = np.asarray([0, 0, 1, 1, -1, -1, 0, 0], dtype=np.int8)
    tail = (
        np.linspace(45.0, 55.0, 100),
        0.0,
        0.2,
        20.0,
        50.0,
        0.0,
        24,
        4,
        1729,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
    )
    multipliers = np.asarray([1.0, 4.0, 16.0], dtype=np.float64)
    inactive_weights = np.asarray([1.0 / 6.0] * 3, dtype=np.float64)
    active_weights = np.asarray([1.0 / 12.0] * 3, dtype=np.float64)
    hmm_weights = np.asarray([1.0 / 12.0] * 3, dtype=np.float64)
    zero_weights = np.zeros(3, dtype=np.float64)
    all_zero_a = _pf_guided_allseeds(
        *common,
        np.zeros(rows, dtype=np.int8),
        *tail,
        1.0,
        inactive_weights,
        active_weights,
        hmm_weights,
        multipliers,
        0.005,
    )
    all_zero_b = _pf_guided_allseeds(
        *common,
        schedule,
        *tail,
        1.0,
        inactive_weights,
        active_weights,
        hmm_weights,
        multipliers,
        0.005,
    )
    exp419_reference = _pf_guided_allseeds(
        *common,
        np.zeros(rows, dtype=np.int8),
        *tail,
        0.5,
        inactive_weights,
        inactive_weights,
        zero_weights,
        multipliers,
        0.005,
    )
    hmm_zero = _pf_guided_allseeds(
        *common,
        schedule,
        *tail,
        0.5,
        inactive_weights,
        inactive_weights,
        zero_weights,
        multipliers,
        0.005,
    )
    all_guidance_zero_equal = all(
        np.array_equal(all_zero_a[index], all_zero_b[index])
        for index in range(len(all_zero_a))
    )
    hmm_weight_zero_equal = all(
        np.array_equal(exp419_reference[index], hmm_zero[index])
        for index in range(len(exp419_reference))
    )
    return {
        "synthetic_rows": rows,
        "particles": 24,
        "seeds": 4,
        "all_guidance_zero_exp404_rng_parity": all_guidance_zero_equal,
        "hmm_weight_zero_exp419_rng_parity": hmm_weight_zero_equal,
        "all_guidance_zero_prediction_max_abs_ft": float(
            np.max(np.abs(all_zero_a[0] - all_zero_b[0]))
        ),
        "hmm_weight_zero_prediction_max_abs_ft": float(
            np.max(np.abs(exp419_reference[0] - hmm_zero[0]))
        ),
    }


def decode_well(
    well: str,
    raw_dir: Path,
    geometry_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    pf = dict(get_nested(config, "model.pf") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(pf["typewell_grid_step_ft"]),
    )
    schedule, schedule_diagnostics = build_untreated_hmm_schedule(
        well,
        horizontal,
        typewell,
        config,
    )
    if not np.array_equal(
        schedule["row_idx"].to_numpy(np.int64),
        prepared["eval_indices"].astype(np.int64),
    ):
        raise ValueError(f"{well}: HMM schedule/PF suffix identity mismatch")
    geometry_rate = geometry_surface_rate(prepared, geometry_rows)
    seed_base = stable_seed("likpf", "train", well)
    fixed = pf_fixed_parameters(config)
    proposal = proposal_contract(config)
    candidate_values, diagnostics, support_minimum, support_maximum = run_guided_likelihood_pf(
        prepared,
        geometry_rate,
        schedule["direction"].to_numpy(np.int8),
        particles=int(pf["particles"]),
        seeds=int(pf["seeds"]),
        seed_base=seed_base,
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(pf["rough_position"]),
        rough_rate=float(pf["rough_rate"]),
        resample_fraction=float(pf["resample_threshold_fraction"]),
        initial_spread=float(pf["initial_position_spread_ft"]),
        initial_rate_spread=float(pf["initial_rate_spread"]),
        target_weight=float(proposal["target_weight"]),
        inactive_geometry_weights=proposal["inactive_geometry_weights"],
        active_geometry_weights=proposal["active_geometry_weights"],
        active_hmm_weights=proposal["active_hmm_weights"],
        geometry_sigma_multipliers=proposal["sigma_multipliers"],
        hmm_rate_step=float(get_nested(config, "model.hmm_signal.rate_step")),
        temperature=float(get_nested(config, "model.aggregation.temperature")),
    )
    eval_indices = prepared["eval_indices"]
    raw_observed = prepared["raw_gr_observed"]
    missing_fraction = float((~raw_observed).mean())
    candidate = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices.astype(np.int64),
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": prepared["md_since"].astype(np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64(missing_fraction),
            "geometry_surface_rate": geometry_rate.astype(np.float32),
            "hmm_predictive_rate_mean": schedule[
                "predictive_rate_mean"
            ].to_numpy(np.float32),
            "hmm_filtered_rate_mean": schedule[
                "filtered_rate_mean"
            ].to_numpy(np.float32),
            "hmm_innovation_rate_cells": schedule[
                "innovation_rate_cells"
            ].to_numpy(np.float32),
            "hmm_positive_cusum": schedule["positive_cusum"].to_numpy(
                np.float32
            ),
            "hmm_negative_cusum": schedule["negative_cusum"].to_numpy(
                np.float32
            ),
            "hmm_trigger_direction": schedule[
                "trigger_direction"
            ].to_numpy(np.int8),
            "hmm_active": schedule["active"].to_numpy(bool),
            "hmm_direction": schedule["direction"].to_numpy(np.int8),
            PRIMARY_CANDIDATE: candidate_values.astype(np.float32),
        }
    )
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "prefix_gr_missing_rows": int(prepared["scale_audit"]["known_gr_missing_rows"]),
        "eval_rows": len(candidate),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "eval_raw_gr_missing_fraction": missing_fraction,
        "last_known_tvt": float(prepared["last_known_tvt"]),
        "last_known_position": float(prepared["last_known_position"]),
        "initial_rate": float(prepared["initial_rate"]),
        "gr_scale_raw": float(prepared["scale_audit"]["raw_scale"]),
        "gr_scale_clipped": float(prepared["scale_audit"]["base_scale"]),
        "seed_base": int(seed_base),
        "seed_first": int(seed_base),
        "seed_last": int(seed_base + int(pf["seeds"]) - 1),
        "seeds": int(pf["seeds"]),
        "particles": int(pf["particles"]),
        "rough_position": float(pf["rough_position"]),
        "rough_rate": float(pf["rough_rate"]),
        "geometry_rate_minimum": float(geometry_rate.min()),
        "geometry_rate_maximum": float(geometry_rate.max()),
        "proposal_contract_sha256": proposal["proposal_contract_sha256"],
        **schedule_diagnostics,
        "seed_well_trajectories": int(pf["seeds"]),
        "particle_starts": int(pf["seeds"]) * int(pf["particles"]),
        **diagnostics,
        "wall_seconds": time.time() - started,
    }
    if not np.isfinite(candidate[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: candidate prediction contains non-finite values")
    return candidate, audit, support_minimum, support_maximum


def freeze_prediction_frame(
    candidate: pd.DataFrame,
    output_path: Path,
    *,
    ledger: TruthAccessLedger | None = None,
) -> dict[str, Any]:
    logical_columns = [
        "id",
        "well_id",
        "row_idx",
        "hmm_active",
        "hmm_direction",
        *PREDICTION_COLUMNS,
    ]
    schedule_columns = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "hmm_active",
        "hmm_direction",
    ]
    if (
        candidate["id"].astype(str).duplicated().any()
        or candidate.duplicated(["well_id", "row_idx"]).any()
    ):
        raise ValueError("candidate row identity is duplicated")
    if not np.isfinite(candidate[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError("candidate prediction contains non-finite values")
    write_deterministic_gzip_csv(candidate, output_path)
    gzip_report = inspect_gzip_csv(output_path)
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].astype(str).nunique()),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "logical_columns": logical_columns,
        "logical_content_sha256": dataframe_content_sha(candidate, logical_columns),
        "schedule_logical_columns": schedule_columns,
        "schedule_logical_sha256": dataframe_content_sha(
            candidate,
            schedule_columns,
        ),
        "schema_sha256": dataframe_schema_sha(candidate),
        "raw_gzip_sha256": gzip_report["raw_sha256"],
        "decompressed_sha256": gzip_report["decompressed_sha256"],
    }
    if ledger is not None:
        ledger.mark_frozen()
        frozen["truth_access_ledger_at_freeze"] = ledger.report()
    return frozen


def _require_frozen_prediction(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("late attachment requires a frozen prediction")
    if len(str(frozen.get("logical_content_sha256") or "")) != 64:
        raise RuntimeError("frozen prediction logical content SHA is missing")


def run_shard(
    config: Mapping[str, Any],
    shard_index: int,
    *,
    scope: str = "full",
    require_run_approval: bool = True,
) -> dict[str, Any]:
    contract = validate_scientific_contract(
        config,
        require_run_approval=require_run_approval,
    )
    if scope not in {"stage_0", "full"}:
        raise ValueError("exp420 candidate scope must be stage_0 or full")
    if scope == "stage_0" and shard_index != 0:
        raise ValueError("exp420 Stage 0 is a single fixed44 shard")
    if scope == "full" and shard_index not in range(SHARD_COUNT):
        raise ValueError(f"shard_index must be in [0, {SHARD_COUNT - 1}]")
    if require_run_approval and scope == "stage_0" and not bool(
        get_nested(config, "execution.stage_0_run_approved")
    ):
        raise RuntimeError("exp420 Stage 0 run is not approved")
    if require_run_approval and scope == "full" and not bool(
        get_nested(config, "execution.full_run_approved")
    ):
        raise RuntimeError("exp420 full run is not approved")
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError("exp420 PF shards must run first on Kaggle CPU")
    started = time.time()
    raw_dir = train_data_dir(config)
    raw_manifest = assign_lpt_shards(build_raw_well_manifest(config, raw_dir))
    stage0_manifest_report: dict[str, Any] | None = None
    if scope == "stage_0":
        fixed44, stage0_manifest_report = load_stage0_fixed44_manifest(config)
        selected = fixed44.merge(
            raw_manifest[["well_id", "suffix_rows"]],
            on="well_id",
            how="left",
            validate="one_to_one",
        )
        selected["shard_index"] = np.int8(0)
        if selected["suffix_rows"].isna().any():
            raise ValueError("exp420 fixed44 contains a well absent from raw train")
    else:
        selected = raw_manifest.loc[
            raw_manifest["shard_index"].eq(shard_index)
        ].copy()
    if selected.empty:
        raise ValueError(f"shard {shard_index} has no wells")
    preflight = preflight_inputs(config)
    geometry = load_fold_safe_geometry(
        preflight["paths"]["exp226_fold_safe_geometry"],
        config,
    )
    selected_wells = selected["well_id"].astype(str).tolist()
    geometry = geometry.loc[geometry["well_id"].isin(selected_wells)].copy()
    if geometry["well_id"].nunique() != len(selected) or len(geometry) != int(
        selected["suffix_rows"].sum()
    ):
        raise ValueError(
            f"{scope} shard {shard_index} exp226 geometry coverage mismatch"
        )
    warm_up_pf_kernel()
    results = [
        decode_well(
            str(well),
            raw_dir,
            geometry.loc[geometry["well_id"].eq(str(well))].copy(),
            config,
        )
        for well in selected_wells
    ]
    candidate = (
        pd.concat([result[0] for result in results], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([result[1] for result in results])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    support_minimum = np.concatenate([result[2] for result in results], axis=0)
    support_maximum = np.concatenate([result[3] for result in results], axis=0)
    if (
        len(candidate) != int(selected["suffix_rows"].sum())
        or candidate["well_id"].nunique() != len(selected)
        or len(audit) != len(selected)
        or not audit["status"].eq("ok").all()
        or support_minimum.shape != (len(candidate), int(get_nested(config, "model.pf.seeds")))
        or support_maximum.shape != support_minimum.shape
        or not np.isfinite(support_minimum).all()
        or not np.isfinite(support_maximum).all()
        or not np.less_equal(support_minimum, support_maximum).all()
    ):
        raise ValueError(f"{scope} shard {shard_index} coverage mismatch")
    output = artifact_dir()
    artifact_tag = "stage0" if scope == "stage_0" else f"shard{shard_index}"
    prediction_path = output / (
        f"{OUTPUT_PREFIX}_{artifact_tag}_candidate_predictions.csv.gz"
    )
    audit_path = output / f"{OUTPUT_PREFIX}_{artifact_tag}_well_audit.csv"
    manifest_path = output / f"{OUTPUT_PREFIX}_{artifact_tag}_well_manifest.csv"
    support_minimum_path = (
        output
        / f"{OUTPUT_PREFIX}_{artifact_tag}_predictive_support_min_float32.npy"
    )
    support_maximum_path = (
        output
        / f"{OUTPUT_PREFIX}_{artifact_tag}_predictive_support_max_float32.npy"
    )
    contract_path = output / f"{OUTPUT_PREFIX}_scientific_contract.json"
    frozen = freeze_prediction_frame(candidate, prediction_path)
    np.save(support_minimum_path, support_minimum.astype(np.float32, copy=False))
    np.save(support_maximum_path, support_maximum.astype(np.float32, copy=False))
    frozen["proposal_diagnostics_logical_content_sha256"] = dataframe_content_sha(
        candidate,
        [
            "id",
            "well_id",
            "row_idx",
            "geometry_surface_rate",
            "hmm_active",
            "hmm_direction",
        ],
    )
    frozen["predictive_support"] = {
        "row_identity_logical_content_sha256": dataframe_content_sha(
            candidate,
            ["id", "well_id", "row_idx"],
        ),
        "minimum_raw_sha256": sha256_path(support_minimum_path),
        "maximum_raw_sha256": sha256_path(support_maximum_path),
        "shape": list(support_minimum.shape),
        "dtype": str(support_minimum.dtype),
        "truth_free": True,
        "semantic": ("per-row per-seed pre-GR predictive particle TVT support extrema"),
    }
    audit.to_csv(audit_path, index=False)
    selected.to_csv(manifest_path, index=False)
    write_json(contract_path, contract)
    elapsed = time.time() - started
    pf = dict(get_nested(config, "model.pf") or {})
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_candidate" if scope == "stage_0" else "candidate_shard",
        "scope": scope,
        "status": "complete",
        "route": "pf_beam",
        "shard_index": shard_index,
        "shard_count": 1 if scope == "stage_0" else SHARD_COUNT,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "counts": {
            "wells": int(len(selected)),
            "rows": int(len(candidate)),
            "scientific_variants": 1,
            "candidate_pf_well_runs": int(len(selected)),
            "seed_well_trajectories": int(len(selected) * int(pf["seeds"])),
            "particle_starts": int(len(selected) * int(pf["seeds"]) * int(pf["particles"])),
            "parent_pf_control_reruns": 0,
            "parent_hmm_control_reruns": 0,
            "exp226_reruns": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_signal_well_runs": int(len(selected)),
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "frozen_prediction": frozen,
        "proposal_input": {
            "safe_columns": preflight["proposal_columns_parsed_before_freeze"],
            "geometry_logical_content_sha256": dataframe_content_sha(
                geometry,
                ["well_id", "row_idx", "suffix_offset", "tvt_geop"],
            ),
            "forbidden_exp226_columns_parsed": [],
            "stage0_manifest": stage0_manifest_report,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": maximum_rss_gb(),
            "versions": runtime_versions(),
        },
        "artifacts": {
            "prediction": {
                "path": str(prediction_path),
                **inspect_gzip_csv(prediction_path),
                "logical_content_sha256": frozen["logical_content_sha256"],
            },
            "well_audit": {
                "path": str(audit_path),
                "raw_sha256": sha256_path(audit_path),
            },
            "well_manifest": {
                "path": str(manifest_path),
                "raw_sha256": sha256_path(manifest_path),
            },
            "predictive_support_minimum": {
                "path": str(support_minimum_path),
                "raw_sha256": sha256_path(support_minimum_path),
                "shape": list(support_minimum.shape),
                "dtype": str(support_minimum.dtype),
            },
            "predictive_support_maximum": {
                "path": str(support_maximum_path),
                "raw_sha256": sha256_path(support_maximum_path),
                "shape": list(support_maximum.shape),
                "dtype": str(support_maximum.dtype),
            },
            "scientific_contract": {
                "path": str(contract_path),
                "raw_sha256": sha256_path(contract_path),
            },
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_{artifact_tag}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Strict shard merge and optional rerun probe


# %%
def _artifact_file(root: Path, filename: str) -> Path:
    direct = root / filename
    nested = root / "artifacts" / filename
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    matches = sorted(root.glob(f"**/{filename}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"expected one {filename} below {root}; found={matches}")


def merge_shard_outputs(
    shard_roots: Sequence[Path],
    output: Path,
    config: Mapping[str, Any],
    *,
    ledger: TruthAccessLedger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Path],
    list[dict[str, Any]],
]:
    if len(shard_roots) != SHARD_COUNT:
        raise ValueError(f"exp420 merge requires exactly {SHARD_COUNT} shard roots")
    contract = validate_scientific_contract(config)
    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    support_shards: list[dict[str, Any]] = []
    for shard_index, root in enumerate(shard_roots):
        summary_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_summary.json",
        )
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("stage") != "candidate_shard"
            or int(summary.get("shard_index", -1)) != shard_index
            or str(summary.get("scientific_contract_sha256"))
            != str(contract["scientific_contract_sha256"])
        ):
            raise ValueError(f"shard {shard_index} summary contract mismatch")
        prediction_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_candidate_predictions.csv.gz",
        )
        audit_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_well_audit.csv",
        )
        manifest_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_well_manifest.csv",
        )
        support_minimum_path = _artifact_file(
            root,
            (f"{OUTPUT_PREFIX}_shard{shard_index}_predictive_support_min_float32.npy"),
        )
        support_maximum_path = _artifact_file(
            root,
            (f"{OUTPUT_PREFIX}_shard{shard_index}_predictive_support_max_float32.npy"),
        )
        prediction = pd.read_csv(
            prediction_path,
            dtype={
                "id": str,
                "well_id": str,
                "row_idx": np.int64,
                "suffix_offset": np.int64,
                "hmm_active": bool,
                "hmm_direction": np.int8,
                PRIMARY_CANDIDATE: np.float32,
            },
        )
        audit = pd.read_csv(audit_path, dtype={"well_id": str})
        manifest = pd.read_csv(manifest_path, dtype={"well_id": str})
        support_minimum = np.load(support_minimum_path, mmap_mode="r")
        support_maximum = np.load(support_maximum_path, mmap_mode="r")
        expected_support = summary["frozen_prediction"]["predictive_support"]
        if (
            dataframe_content_sha(
                prediction,
                summary["frozen_prediction"]["logical_columns"],
            )
            != summary["frozen_prediction"]["logical_content_sha256"]
        ):
            raise ValueError(f"shard {shard_index} logical prediction SHA mismatch")
        if (
            list(support_minimum.shape) != list(expected_support["shape"])
            or support_maximum.shape != support_minimum.shape
            or support_minimum.shape != (len(prediction), int(get_nested(config, "model.pf.seeds")))
            or sha256_path(support_minimum_path) != str(expected_support["minimum_raw_sha256"])
            or sha256_path(support_maximum_path) != str(expected_support["maximum_raw_sha256"])
            or dataframe_content_sha(prediction, ["id", "well_id", "row_idx"])
            != str(expected_support["row_identity_logical_content_sha256"])
        ):
            raise ValueError(f"shard {shard_index} predictive-support contract mismatch")
        if not manifest["shard_index"].astype(int).eq(shard_index).all():
            raise ValueError(f"shard {shard_index} manifest assignment mismatch")
        prediction_parts.append(prediction)
        audit_parts.append(audit)
        manifest_parts.append(manifest)
        summaries.append(summary)
        support_shards.append(
            {
                "shard_index": shard_index,
                "prediction_path": prediction_path,
                "minimum_path": support_minimum_path,
                "maximum_path": support_maximum_path,
                "shape": list(support_minimum.shape),
                "minimum_raw_sha256": expected_support["minimum_raw_sha256"],
                "maximum_raw_sha256": expected_support["maximum_raw_sha256"],
            }
        )
    candidate = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.concat(audit_parts, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    manifest = (
        pd.concat(manifest_parts, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(candidate) != expected_rows
        or candidate["well_id"].nunique() != expected_wells
        or candidate["id"].duplicated().any()
        or candidate.duplicated(["well_id", "row_idx"]).any()
        or len(audit) != expected_wells
        or audit["well_id"].duplicated().any()
        or not audit["status"].eq("ok").all()
        or len(manifest) != expected_wells
        or manifest["well_id"].duplicated().any()
        or int(manifest["suffix_rows"].sum()) != expected_rows
    ):
        raise ValueError("strict exp420 shard merge coverage mismatch")
    expected_counts = {
        "candidate_pf_well_runs": int(
            get_nested(config, "execution.full.candidate_pf_well_runs")
        ),
        "hmm_signal_well_runs": int(
            get_nested(config, "execution.full.hmm_signal_well_runs")
        ),
        "seed_well_trajectories": int(
            get_nested(config, "execution.full.seed_well_trajectories")
        ),
        "particle_starts": int(
            get_nested(config, "execution.full.particle_starts")
        ),
    }
    actual_counts = {
        "candidate_pf_well_runs": int(
            sum(item["counts"]["candidate_pf_well_runs"] for item in summaries)
        ),
        "hmm_signal_well_runs": int(
            sum(item["counts"]["hmm_signal_well_runs"] for item in summaries)
        ),
        "seed_well_trajectories": int(
            sum(item["counts"]["seed_well_trajectories"] for item in summaries)
        ),
        "particle_starts": int(sum(item["counts"]["particle_starts"] for item in summaries)),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"exp420 execution count mismatch: {actual_counts} != {expected_counts}")
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / f"{OUTPUT_PREFIX}_merged_candidate_predictions.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_merged_well_audit.csv"
    manifest_path = output / f"{OUTPUT_PREFIX}_merged_well_manifest.csv"
    frozen = freeze_prediction_frame(candidate, prediction_path, ledger=ledger)
    audit.to_csv(audit_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    frozen["execution_counts"] = actual_counts
    frozen["shard_logical_content_sha256"] = [
        item["frozen_prediction"]["logical_content_sha256"] for item in summaries
    ]
    frozen["predictive_support_shards"] = [
        {
            key: to_jsonable(value)
            for key, value in item.items()
            if key not in {"prediction_path", "minimum_path", "maximum_path"}
        }
        for item in support_shards
    ]
    return (
        candidate,
        audit,
        frozen,
        {
            "merged_prediction": prediction_path,
            "merged_well_audit": audit_path,
            "merged_well_manifest": manifest_path,
        },
        support_shards,
    )


def probe_rerun_report(
    merged_candidate: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    probe_well: str,
    geometry_rows: pd.DataFrame,
) -> dict[str, Any]:
    expected = merged_candidate.loc[
        merged_candidate["well_id"].astype(str).eq(str(probe_well))
    ].sort_values("row_idx", kind="mergesort")
    observed, audit, support_minimum, support_maximum = decode_well(
        str(probe_well),
        raw_dir,
        geometry_rows,
        config,
    )
    observed = observed.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(
        expected["row_idx"].to_numpy(np.int64),
        observed["row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("probe rerun row identity mismatch")
    expected_values = expected[PRIMARY_CANDIDATE].to_numpy(np.float32)
    observed_values = observed[PRIMARY_CANDIDATE].to_numpy(np.float32)
    byte_identical = bool(np.array_equal(expected_values, observed_values))
    schedule_columns = ["hmm_active", "hmm_direction"]
    schedule_byte_identical = bool(
        all(
            np.array_equal(
                expected[column].to_numpy(),
                observed[column].to_numpy(),
            )
            for column in schedule_columns
        )
    )
    normalized_expected = expected.copy()
    normalized_expected[PRIMARY_CANDIDATE] = expected_values
    horizontal = load_horizontal_without_truth(str(probe_well), raw_dir)
    typewell = load_typewell(str(probe_well), raw_dir)
    pf = dict(get_nested(config, "model.pf") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(pf["typewell_grid_step_ft"]),
    )
    geometry_rate = geometry_surface_rate(prepared, geometry_rows)
    fixed = pf_fixed_parameters(config)
    baseline_values, _, _, _ = run_guided_likelihood_pf(
        prepared,
        geometry_rate,
        np.zeros(len(geometry_rate), dtype=np.int8),
        particles=int(pf["particles"]),
        seeds=int(pf["seeds"]),
        seed_base=stable_seed("likpf", "train", probe_well),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(pf["rough_position"]),
        rough_rate=float(pf["rough_rate"]),
        resample_fraction=float(pf["resample_threshold_fraction"]),
        initial_spread=float(pf["initial_position_spread_ft"]),
        initial_rate_spread=float(pf["initial_rate_spread"]),
        target_weight=1.0,
        inactive_geometry_weights=[1.0 / 6.0] * 3,
        active_geometry_weights=[1.0 / 12.0] * 3,
        active_hmm_weights=[1.0 / 12.0] * 3,
        geometry_sigma_multipliers=[1.0, 4.0, 16.0],
        hmm_rate_step=0.005,
        temperature=float(get_nested(config, "model.aggregation.temperature")),
    )
    preflight = preflight_inputs(config)
    control_column = str(get_nested(config, "data.exp404_frozen_predictions.control_column"))
    saved = pd.read_csv(
        preflight["paths"]["exp404_frozen_predictions"],
        usecols=["id", control_column],
        dtype={"id": str},
        compression="gzip",
    )
    saved = saved.loc[saved["id"].isin(observed["id"].astype(str))].set_index("id")
    saved_values = saved.reindex(observed["id"].astype(str))[control_column].to_numpy(np.float32)
    baseline_float32 = baseline_values.astype(np.float32)
    geometry_weight_zero_parity = float(
        np.max(np.abs(baseline_float32.astype(np.float64) - saved_values.astype(np.float64)))
    )
    schedule_direction = observed["hmm_direction"].to_numpy(np.int8)
    exp419_reference, _, _, _ = run_guided_likelihood_pf(
        prepared,
        geometry_rate,
        np.zeros(len(geometry_rate), dtype=np.int8),
        particles=int(pf["particles"]),
        seeds=int(pf["seeds"]),
        seed_base=stable_seed("likpf", "train", probe_well),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(pf["rough_position"]),
        rough_rate=float(pf["rough_rate"]),
        resample_fraction=float(pf["resample_threshold_fraction"]),
        initial_spread=float(pf["initial_position_spread_ft"]),
        initial_rate_spread=float(pf["initial_rate_spread"]),
        target_weight=0.5,
        inactive_geometry_weights=[1.0 / 6.0] * 3,
        active_geometry_weights=[1.0 / 6.0] * 3,
        active_hmm_weights=[0.0, 0.0, 0.0],
        geometry_sigma_multipliers=[1.0, 4.0, 16.0],
        hmm_rate_step=0.005,
        temperature=float(get_nested(config, "model.aggregation.temperature")),
    )
    hmm_zero_values, _, _, _ = run_guided_likelihood_pf(
        prepared,
        geometry_rate,
        schedule_direction,
        particles=int(pf["particles"]),
        seeds=int(pf["seeds"]),
        seed_base=stable_seed("likpf", "train", probe_well),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(pf["rough_position"]),
        rough_rate=float(pf["rough_rate"]),
        resample_fraction=float(pf["resample_threshold_fraction"]),
        initial_spread=float(pf["initial_position_spread_ft"]),
        initial_rate_spread=float(pf["initial_rate_spread"]),
        target_weight=0.5,
        inactive_geometry_weights=[1.0 / 6.0] * 3,
        active_geometry_weights=[1.0 / 6.0] * 3,
        active_hmm_weights=[0.0, 0.0, 0.0],
        geometry_sigma_multipliers=[1.0, 4.0, 16.0],
        hmm_rate_step=0.005,
        temperature=float(get_nested(config, "model.aggregation.temperature")),
    )
    hmm_zero_exp419_parity = float(
        np.max(
            np.abs(
                exp419_reference.astype(np.float32).astype(np.float64)
                - hmm_zero_values.astype(np.float32).astype(np.float64)
            )
        )
    )
    return {
        "probe_well": str(probe_well),
        "rows": len(observed),
        "byte_identical_float32": byte_identical,
        "schedule_byte_identical": schedule_byte_identical,
        "maximum_absolute_difference_ft": float(
            np.max(np.abs(expected_values.astype(np.float64) - observed_values.astype(np.float64)))
        ),
        "expected_logical_content_sha256": dataframe_content_sha(
            normalized_expected,
            ["id", "well_id", "row_idx", *PREDICTION_COLUMNS],
        ),
        "observed_logical_content_sha256": dataframe_content_sha(
            observed,
            ["id", "well_id", "row_idx", *PREDICTION_COLUMNS],
        ),
        "audit": audit,
        "all_guidance_zero_exp404_parity_max_abs_ft": (
            geometry_weight_zero_parity
        ),
        "all_guidance_zero_exp404_parity_atol_ft": float(
            get_nested(
                config,
                "guards.stage_0_technical."
                "require_all_guidance_zero_exp404_parity_atol_ft_after_float32",
            )
        ),
        "hmm_weight_zero_exp419_parity_max_abs_ft": (
            hmm_zero_exp419_parity
        ),
        "predictive_support_minimum_sha256": hashlib.sha256(
            np.ascontiguousarray(support_minimum).tobytes()
        ).hexdigest(),
        "predictive_support_maximum_sha256": hashlib.sha256(
            np.ascontiguousarray(support_maximum).tobytes()
        ).hexdigest(),
    }


# %% [markdown]
# ## 10. Late truth, saved-control, fold, hidden-like, and episode attachment


# %%
def load_unknown_suffix_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["TVT_input", "TVT"],
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    true_tvt = pd.to_numeric(horizontal["TVT"], errors="coerce")
    eval_indices = np.flatnonzero(tvt_input.isna().to_numpy()).astype(np.int64)
    values = true_tvt.iloc[eval_indices].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{well}: unknown-suffix TVT contains non-finite values")
    return pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "true_tvt": values,
        }
    )


def align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    lookup_source = source.copy()
    lookup_source["id"] = lookup_source["id"].astype(str)
    if lookup_source["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = lookup_source.set_index("id").reindex(frame["id"].astype(str))
    if aligned[list(columns)].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def attach_candidate_predictive_support(
    frame: pd.DataFrame,
    support_shards: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Attach exact per-row fraction of seeds whose predictive support contains truth."""

    truth_lookup = frame.set_index(frame["id"].astype(str))["true_tvt"]
    parts: list[pd.DataFrame] = []
    for shard in support_shards:
        identity = pd.read_csv(
            Path(str(shard["prediction_path"])),
            usecols=["id"],
            dtype={"id": str},
        )
        truth = truth_lookup.reindex(identity["id"].astype(str)).to_numpy(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError("predictive-support shard truth alignment is incomplete")
        minimum = np.load(Path(str(shard["minimum_path"])), mmap_mode="r")
        maximum = np.load(Path(str(shard["maximum_path"])), mmap_mode="r")
        if minimum.shape != maximum.shape or minimum.shape[0] != len(identity):
            raise ValueError("predictive-support shard shape changed after freeze")
        fraction = np.empty(len(identity), dtype=np.float32)
        chunk_rows = 20_000
        for start in range(0, len(identity), chunk_rows):
            stop = min(start + chunk_rows, len(identity))
            target = truth[start:stop, None]
            fraction[start:stop] = np.mean(
                (minimum[start:stop] <= target) & (target <= maximum[start:stop]),
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
        parts.append(
            pd.DataFrame(
                {
                    "id": identity["id"].astype(str),
                    "candidate_predictive_truth_support_fraction": fraction,
                }
            )
        )
    support = pd.concat(parts, ignore_index=True)
    if len(support) != len(frame) or support["id"].duplicated().any():
        raise ValueError("candidate predictive-support identity coverage mismatch")
    return align_on_id(
        frame,
        support,
        ["candidate_predictive_truth_support_fraction"],
        label="candidate predictive support",
    )


def expand_fixed_episode_rows(episodes: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for episode in episodes.itertuples(index=False):
        row_idx = np.arange(
            int(episode.start_row_idx),
            int(episode.end_row_idx_exclusive),
            dtype=np.int64,
        )
        parts.append(
            pd.DataFrame(
                {
                    "well_id": str(episode.well),
                    "row_idx": row_idx,
                    "episode_id": str(episode.episode_id),
                }
            )
        )
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("fixed exp410 episodes overlap")
    return result


def load_late_readout_frame(
    candidate: pd.DataFrame,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any],
    support_shards: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    _require_frozen_prediction(frozen)
    ledger.require_frozen()
    reverified = dataframe_content_sha(candidate, list(frozen["logical_columns"]))
    if reverified != str(frozen["logical_content_sha256"]):
        raise ValueError("in-memory candidate changed after prediction freeze")
    wells = sorted(candidate["well_id"].astype(str).unique().tolist())
    truth = (
        pd.concat([load_unknown_suffix_truth(well, raw_dir) for well in wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    ledger.unknown_suffix_tvt_rows_after_freeze += len(truth)
    frame = align_on_id(candidate, truth, ["true_tvt"], label="raw suffix truth")
    frame = attach_candidate_predictive_support(frame, support_shards)

    control_spec = input_spec(config, "exp404_frozen_predictions")
    control_column = str(control_spec["control_column"])
    exp404 = pd.read_csv(
        preflight["paths"]["exp404_frozen_predictions"],
        usecols=["id", "well_id", "row_idx", "suffix_offset", control_column],
        dtype={"id": str},
        compression="gzip",
    )
    exp404[control_column] = pd.to_numeric(exp404[control_column], errors="raise").astype(
        np.float64
    )
    frame = align_on_id(
        frame,
        exp404[["id", control_column]],
        [control_column],
        label="saved exp404 scale5 control",
    )
    ledger.control_prediction_rows_after_freeze += len(frame)

    exp226 = pd.read_csv(
        preflight["paths"]["exp226_fold_safe_geometry"],
        usecols=["well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"],
        dtype={"well_id": str},
        compression="gzip",
    )
    for column in ("row_idx", "suffix_offset", "fold"):
        exp226[column] = pd.to_numeric(exp226[column], errors="raise").astype(np.int64)
    exp226["exp226_final_oof"] = pd.to_numeric(exp226.pop("tvt_pred"), errors="raise").astype(
        np.float64
    )
    if exp226.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 final/fold identity is duplicated")
    ledger.fold_rows_after_freeze += len(exp226)
    ledger.exp226_final_rows_after_freeze += len(exp226)
    frame = frame.merge(
        exp226,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_exp226"),
        sort=False,
    )
    if frame[["fold", "suffix_offset_exp226"]].isna().any().any():
        raise ValueError("reporting fold attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["suffix_offset_exp226"].to_numpy(np.int64),
    ):
        raise ValueError("exp226 suffix offset identity mismatch")
    frame = frame.drop(columns=["suffix_offset_exp226"])

    exp072_spec = input_spec(config, "exp072_saved_likpf")
    exp072_residual_column = str(exp072_spec["residual_column"])
    exp072_anchor_column = str(exp072_spec["anchor_column"])
    if str(exp072_spec["transform"]) != "anchor_plus":
        raise ValueError("exp072 physical-anchor input must use anchor_plus")
    exp072 = pd.read_csv(
        preflight["paths"]["exp072_saved_likpf"],
        usecols=["id", exp072_residual_column, exp072_anchor_column],
        dtype={"id": str},
        compression="gzip",
    )
    exp072["exp072_likpf_mean"] = (
        pd.to_numeric(exp072[exp072_anchor_column], errors="raise").to_numpy(np.float32)
        + pd.to_numeric(exp072[exp072_residual_column], errors="raise").to_numpy(
            np.float32
        )
    ).astype(np.float32)
    exp072 = exp072[["id", "exp072_likpf_mean"]]
    exp209_spec = input_spec(config, "exp209_saved_hmm")
    exp209_column = str(exp209_spec["prediction_column"])
    exp209 = pd.read_csv(
        preflight["paths"]["exp209_saved_hmm"],
        usecols=["id", exp209_column],
        dtype={"id": str},
        compression="gzip",
    ).rename(columns={exp209_column: "exp209_exact_hmm"})
    frame = align_on_id(
        frame,
        exp072,
        ["exp072_likpf_mean"],
        label="saved exp072 likpf_mean physical-anchor input",
    )
    frame = align_on_id(
        frame,
        exp209,
        ["exp209_exact_hmm"],
        label="saved exp209 exact-HMM physical-anchor input",
    )
    frame["exp263_fixed_physical_blend"] = (
        np.float32(0.5) * frame["exp226_final_oof"].to_numpy(np.float32)
        + np.float32(0.25)
        * frame["exp072_likpf_mean"].to_numpy(np.float32)
        + np.float32(0.25)
        * frame["exp209_exact_hmm"].to_numpy(np.float32)
    ).astype(np.float32)
    ledger.physical_anchor_rows_after_freeze += len(frame)

    hidden_spec = input_spec(config, "hidden_like_assignment")
    role_columns = {
        str(scope): str(column) for scope, column in hidden_spec["role_columns"].items()
    }
    hidden = pd.read_csv(
        preflight["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    if hidden["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment has duplicate wells")
    expected_role_counts = hidden_spec.get("expected_role_counts") or {}
    for scope, column in role_columns.items():
        actual = {
            str(key): int(value)
            for key, value in hidden[column].astype(str).value_counts().sort_index().items()
        }
        expected = {
            str(key): int(value) for key, value in (expected_role_counts.get(scope) or {}).items()
        }
        if actual != expected:
            raise ValueError(f"hidden-like role counts mismatch for {scope}")
    ledger.hidden_like_role_rows_after_freeze += len(hidden)
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[role_columns["hidden_like_spatial"]].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[role_columns["hidden_like_typewell_purged"]].eq(
        "valid"
    )

    target_wells = pd.read_csv(
        preflight["paths"]["exp410_target_wells"],
        dtype={"well": str},
    )
    episodes = pd.read_csv(
        preflight["paths"]["exp410_persistent_episodes"],
        dtype={"episode_id": str, "well": str},
    )
    selected_episodes = episodes.loc[episodes["well"].isin(target_wells["well"].astype(str))].copy()
    expected_wells = int(get_nested(config, "data.exp410_target_wells.expected_wells"))
    expected_episodes = int(get_nested(config, "data.exp410_persistent_episodes.expected_episodes"))
    expected_episode_rows = int(
        get_nested(config, "data.exp410_persistent_episodes.expected_episode_rows")
    )
    if (
        target_wells["well"].nunique() != expected_wells
        or selected_episodes["episode_id"].nunique() != expected_episodes
        or int(selected_episodes["rows"].sum()) != expected_episode_rows
    ):
        raise ValueError("exp410 target-well/episode identity changed")
    episode_rows = expand_fixed_episode_rows(selected_episodes)
    baseline_parts = [
        pd.read_csv(
            path,
            usecols=["well", "row_idx", "predictive_truth_support_fraction"],
            dtype={"well": str},
            compression="gzip",
        )
        for path in preflight["exp410_row_ledger_paths"]
    ]
    baseline = pd.concat(baseline_parts, ignore_index=True).rename(
        columns={
            "well": "well_id",
            "predictive_truth_support_fraction": (
                "exp410_baseline_predictive_truth_support_fraction"
            ),
        }
    )
    baseline["row_idx"] = pd.to_numeric(baseline["row_idx"], errors="raise").astype(np.int64)
    fixed_support = episode_rows.merge(
        baseline,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if (
        len(fixed_support) != expected_episode_rows
        or fixed_support["exp410_baseline_predictive_truth_support_fraction"].isna().any()
    ):
        raise ValueError("exp410 baseline support coverage changed")
    frame = frame.merge(
        fixed_support,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    frame["exp410_target_well"] = frame["well_id"].isin(target_wells["well"].astype(str))
    frame["exp410_fixed_episode"] = frame["episode_id"].notna()
    ledger.exp410_scope_rows_after_freeze += len(fixed_support)

    exp408_spec = input_spec(config, "exp408_persistent_episodes")
    exp408_episodes = pd.read_csv(
        preflight["paths"]["exp408_persistent_episodes"],
        dtype={"episode_id": str, "well": str},
    )
    if (
        exp408_episodes["episode_id"].nunique()
        != int(exp408_spec["expected_episodes"])
        or int(exp408_episodes["rows"].sum())
        != int(exp408_spec["expected_episode_rows"])
    ):
        raise ValueError("exp408 persistent episode identity changed")
    exp408_rows = expand_fixed_episode_rows(exp408_episodes).rename(
        columns={"episode_id": "exp408_episode_id"}
    )
    frame = frame.merge(
        exp408_rows,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    frame["exp408_fixed_episode"] = frame["exp408_episode_id"].notna()
    ledger.exp408_scope_rows_after_freeze += len(exp408_rows)
    if not np.isfinite(
        frame[
            [
                "true_tvt",
                control_column,
                "exp226_final_oof",
                "exp263_fixed_physical_blend",
                "candidate_predictive_truth_support_fraction",
                *PREDICTION_COLUMNS,
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("late readout contains non-finite values")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("reporting fold set mismatch")
    return (
        frame,
        {
            "exp410": selected_episodes,
            "exp408": exp408_episodes,
        },
        {
            "truth_attached_after_prediction_freeze": True,
            "candidate_content_sha256_reverified": reverified,
            "rows": len(frame),
            "wells": int(frame["well_id"].nunique()),
            "folds": expected_folds,
            "persistent_episode_count": len(selected_episodes),
            "persistent_episode_rows": int(selected_episodes["rows"].sum()),
            "exp408_persistent_episode_count": len(exp408_episodes),
            "exp408_persistent_episode_rows": int(exp408_episodes["rows"].sum()),
            "candidate_predictive_support_attached_after_freeze": True,
            "exp410_baseline_support_attached_after_freeze": True,
            "truth_access_ledger": ledger.report(),
        },
    )


# %% [markdown]
# ## 11. Metrics and fail-closed scientific gate


# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"metric scope {scope} is empty")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[PRIMARY_CANDIDATE].to_numpy(np.float64)
    control = selected["likpf_scale_5_x1p0"].to_numpy(np.float64)
    exp226 = selected["exp226_final_oof"].to_numpy(np.float64)
    physical = selected["exp263_fixed_physical_blend"].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    exp226_rmse = rmse(truth, exp226)
    physical_rmse = rmse(truth, physical)
    return {
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate": PRIMARY_CANDIDATE,
        "candidate_rmse": candidate_rmse,
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "candidate_bias": float(np.mean(candidate - truth)),
        "candidate_within_10ft": float(np.mean(np.abs(candidate - truth) <= 10.0)),
        "control": "likpf_scale_5_x1p0",
        "control_rmse": control_rmse,
        "control_mae": float(np.mean(np.abs(control - truth))),
        "control_bias": float(np.mean(control - truth)),
        "control_within_10ft": float(np.mean(np.abs(control - truth) <= 10.0)),
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
        "standalone_reference": "exp226_final_oof",
        "exp226_final_rmse": exp226_rmse,
        "improvement_vs_exp226_ft": exp226_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_exp226": candidate_rmse - exp226_rmse,
        "physical_anchor_reference": "exp263_fixed_physical_blend",
        "physical_anchor_rmse": physical_rmse,
        "improvement_vs_physical_anchor_ft": physical_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_physical_anchor": (
            candidate_rmse - physical_rmse
        ),
    }


def metric_scopes(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
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


def build_metric_outputs(
    frame: pd.DataFrame,
    episode_sets: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    overall = pd.DataFrame(
        [metric_record(frame, mask, scope=scope) for scope, mask in metric_scopes(frame)]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        candidate = group[PRIMARY_CANDIDATE].to_numpy(np.float64)
        control = group["likpf_scale_5_x1p0"].to_numpy(np.float64)
        exp226 = group["exp226_final_oof"].to_numpy(np.float64)
        physical = group["exp263_fixed_physical_blend"].to_numpy(np.float64)
        candidate_rmse = rmse(truth, candidate)
        control_rmse = rmse(truth, control)
        exp226_rmse = rmse(truth, exp226)
        physical_rmse = rmse(truth, physical)
        by_well_rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "candidate_rmse": candidate_rmse,
                "control_rmse": control_rmse,
                "improvement_ft": control_rmse - candidate_rmse,
                "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
                "exp226_final_rmse": exp226_rmse,
                "improvement_vs_exp226_ft": exp226_rmse - candidate_rmse,
                "delta_rmse_candidate_minus_exp226": candidate_rmse - exp226_rmse,
                "physical_anchor_rmse": physical_rmse,
                "improvement_vs_physical_anchor_ft": (
                    physical_rmse - candidate_rmse
                ),
                "delta_rmse_candidate_minus_physical_anchor": (
                    candidate_rmse - physical_rmse
                ),
            }
        )
    episode_outputs: dict[str, pd.DataFrame] = {}
    for scope, selected_episodes in episode_sets.items():
        episode_column = "episode_id" if scope == "exp410" else "exp408_episode_id"
        episode_rows: list[dict[str, Any]] = []
        episode_contract = selected_episodes.set_index("episode_id")
        fixed_rows = frame.loc[frame[episode_column].notna()]
        observed_episode_ids = set(fixed_rows[episode_column].astype(str))
        if observed_episode_ids != set(episode_contract.index.astype(str)):
            raise ValueError(f"{scope} persistent episode identity coverage mismatch")
        for episode_id, selected in fixed_rows.groupby(episode_column, sort=True):
            episode = episode_contract.loc[str(episode_id)]
            if len(selected) != int(episode["rows"]):
                raise ValueError(
                    f"{scope} {episode_id}: persistent episode coverage mismatch"
                )
            truth = selected["true_tvt"].to_numpy(np.float64)
            candidate = selected[PRIMARY_CANDIDATE].to_numpy(np.float64)
            control = selected["likpf_scale_5_x1p0"].to_numpy(np.float64)
            candidate_sse = float(np.square(candidate - truth).sum())
            control_sse = float(np.square(control - truth).sum())
            episode_rows.append(
                {
                    "episode_scope": scope,
                    "episode_id": str(episode_id),
                    "well_id": str(episode["well"]),
                    "rows": len(selected),
                    "candidate_sse": candidate_sse,
                    "control_sse": control_sse,
                    "candidate_rmse": math.sqrt(candidate_sse / len(selected)),
                    "control_rmse": math.sqrt(control_sse / len(selected)),
                    "sse_reduction_fraction": (
                        1.0 - candidate_sse / control_sse
                    ),
                    "improved": candidate_sse < control_sse,
                }
            )
        episode_outputs[scope] = pd.DataFrame(episode_rows)
    return overall, pd.DataFrame(by_well_rows), episode_outputs


def scope_row(metrics: pd.DataFrame, scope: str) -> pd.Series:
    selected = metrics.loc[metrics["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one metric row for scope={scope}")
    return selected.iloc[0]


def evaluate_full_gate(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    episode_metrics: Mapping[str, pd.DataFrame],
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    ledger: TruthAccessLedger,
    shard_summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    probe_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    technical_config = dict(
        get_nested(config, "guards.stage_0_technical") or {}
    )
    mechanism_config = dict(
        get_nested(config, "guards.full_mechanism") or {}
    )
    adoption_config = dict(
        get_nested(config, "guards.standalone_adoption") or {}
    )
    physical_config = dict(get_nested(config, "guards.physical_anchor") or {})
    overall = scope_row(metrics, "overall")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    control_parity_difference = abs(
        float(overall["control_rmse"])
        - float(get_nested(config, "validation.saved_control_rmse_ft"))
    )
    exp226_parity_difference = abs(
        float(overall["exp226_final_rmse"])
        - float(get_nested(config, "validation.saved_exp226_final_rmse_ft"))
    )
    physical_parity_difference = abs(
        float(overall["physical_anchor_rmse"])
        - float(
            get_nested(
                config,
                "validation.saved_exp263_physical_blend_rmse_ft",
            )
        )
    )
    actual_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": len(audit),
        "hmm_signal_well_runs": len(audit),
        "parent_hmm_control_reruns": 0,
        "parent_pf_control_reruns": 0,
        "exp226_reruns": 0,
        "seed_well_trajectories": int(
            audit["seed_well_trajectories"].sum()
        ),
        "particle_starts": int(audit["particle_starts"].sum()),
        "reporting_folds": int(frame["fold"].nunique()),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        "scientific_variants": int(
            get_nested(config, "execution.scientific_variants")
        ),
        "candidate_pf_well_runs": int(
            get_nested(config, "execution.full.candidate_pf_well_runs")
        ),
        "hmm_signal_well_runs": int(
            get_nested(config, "execution.full.hmm_signal_well_runs")
        ),
        "parent_hmm_control_reruns": int(
            get_nested(config, "execution.full.parent_hmm_control_reruns")
        ),
        "parent_pf_control_reruns": int(
            get_nested(config, "execution.full.parent_pf_control_reruns")
        ),
        "exp226_reruns": int(
            get_nested(config, "execution.full.exp226_reruns")
        ),
        "seed_well_trajectories": int(
            get_nested(config, "execution.full.seed_well_trajectories")
        ),
        "particle_starts": int(
            get_nested(config, "execution.full.particle_starts")
        ),
        "reporting_folds": int(get_nested(config, "execution.reporting_folds")),
        "lightgbm_configs": int(
            get_nested(config, "execution.lightgbm_configs")
        ),
        "trained_folds": int(get_nested(config, "execution.trained_folds")),
        "boosters": int(get_nested(config, "execution.boosters")),
        "models": int(get_nested(config, "execution.models")),
        "beam_well_runs": int(get_nested(config, "execution.beam_well_runs")),
        "gpu_runs": int(get_nested(config, "execution.gpu_runs")),
    }
    runtime_limit = float(get_nested(config, "runtime.hard_seconds_per_shard"))
    shard_runtime_seconds = [
        float(summary["runtime"]["elapsed_seconds"])
        for summary in shard_summaries
    ]
    before_freeze = ledger.report()["before_freeze"]
    proposal = proposal_contract(config)
    importance_minimum = float(audit["importance_ratio_minimum"].min())
    importance_maximum = float(audit["importance_ratio_maximum"].max())
    proposal_allowlists = {
        tuple(summary["proposal_input"]["safe_columns"])
        for summary in shard_summaries
    }
    forbidden_exp226_columns = sum(
        len(summary["proposal_input"]["forbidden_exp226_columns_parsed"])
        for summary in shard_summaries
    )
    probe_byte_identical = bool(
        probe_report is not None
        and probe_report.get("byte_identical_float32", False)
    )
    probe_schedule_parity = bool(
        probe_report is not None
        and probe_report.get("schedule_byte_identical", False)
    )
    probe_exp404_parity = (
        math.inf
        if probe_report is None
        else float(
            probe_report.get(
                "all_guidance_zero_exp404_parity_max_abs_ft",
                math.inf,
            )
        )
    )
    probe_exp419_parity = (
        math.inf
        if probe_report is None
        else float(
            probe_report.get(
                "hmm_weight_zero_exp419_parity_max_abs_ft",
                math.inf,
            )
        )
    )
    technical = {
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "reporting_folds": sorted(frame["fold"].astype(int).unique()),
        "audit_wells": len(audit),
        "all_wells_completed_without_fallback": bool(
            audit["status"].eq("ok").all()
        ),
        "finite_candidate_coverage": float(
            np.isfinite(
                frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)
            ).mean()
        ),
        "saved_control_rmse_parity_absolute_difference_ft": (
            control_parity_difference
        ),
        "saved_exp226_final_rmse_parity_absolute_difference_ft": (
            exp226_parity_difference
        ),
        "saved_physical_anchor_rmse_parity_absolute_difference_ft": (
            physical_parity_difference
        ),
        "truth_or_reporting_values_parsed_before_freeze": before_freeze,
        "execution_counts": actual_counts,
        "execution_count_match": actual_counts == expected_counts,
        "proposal_contract": proposal,
        "proposal_allowlists": [
            list(value) for value in sorted(proposal_allowlists)
        ],
        "forbidden_exp226_columns_parsed": forbidden_exp226_columns,
        "importance_ratio_minimum": importance_minimum,
        "importance_ratio_maximum": importance_maximum,
        "prediction_logical_content_sha256": frozen[
            "logical_content_sha256"
        ],
        "schedule_logical_content_sha256": frozen[
            "schedule_logical_sha256"
        ],
        "shard_count": len(shard_summaries),
        "shard_runtime_seconds": shard_runtime_seconds,
        "runtime_limit_seconds_per_shard": runtime_limit,
        "probe_rerun_available": probe_report is not None,
        "probe_rerun_byte_identical_float32": probe_byte_identical,
        "schedule_rerun_logical_parity": probe_schedule_parity,
        "all_guidance_zero_exp404_parity_max_abs_ft": probe_exp404_parity,
        "hmm_weight_zero_exp419_parity_max_abs_ft": probe_exp419_parity,
    }
    technical["passed"] = bool(
        len(frame) == expected_rows
        and frame["well_id"].nunique() == expected_wells
        and technical["reporting_folds"] == expected_folds
        and len(audit) == expected_wells
        and technical["all_wells_completed_without_fallback"]
        and technical["finite_candidate_coverage"]
        == float(technical_config["require_finite_candidate_coverage"])
        and control_parity_difference <= 1.0e-5
        and exp226_parity_difference <= 1.0e-5
        and physical_parity_difference <= 1.0e-5
        and all(int(value) == 0 for value in before_freeze.values())
        and actual_counts == expected_counts
        and proposal["inactive_weight_sum"]
        == float(technical_config["require_inactive_and_active_weight_sum"])
        and proposal["active_weight_sum"]
        == float(technical_config["require_inactive_and_active_weight_sum"])
        and importance_minimum >= 0.0
        and importance_maximum
        <= float(technical_config["maximum_importance_ratio"])
        and proposal_allowlists
        == {("well_id", "row_idx", "suffix_offset", "tvt_geop")}
        and forbidden_exp226_columns == 0
        and len(shard_summaries) == SHARD_COUNT
        and all(
            seconds <= runtime_limit for seconds in shard_runtime_seconds
        )
        and probe_byte_identical
        and probe_schedule_parity
        and probe_exp404_parity
        <= float(
            technical_config[
                "require_all_guidance_zero_exp404_parity_atol_ft_after_float32"
            ]
        )
        and probe_exp419_parity
        <= float(
            technical_config[
                "require_hmm_weight_zero_exp419_proposal_parity_atol_ft_after_float32"
            ]
        )
    )

    fold_rows = metrics.loc[metrics["scope"].str.startswith("fold_")]
    improved_folds = int((fold_rows["improvement_ft"] > 0.0).sum())
    improved_folds_vs_exp226 = int(
        (fold_rows["improvement_vs_exp226_ft"] > 0.0).sum()
    )
    improved_folds_vs_physical = int(
        (fold_rows["improvement_vs_physical_anchor_ft"] > 0.0).sum()
    )
    observed = scope_row(metrics, "raw_gr_observed")
    non_regression_limits = {
        "raw_gr_missing": float(
            mechanism_config["maximum_raw_gr_missing_regression_ft"]
        ),
        "missing_fraction_high": float(
            mechanism_config["maximum_high_missing_regression_ft"]
        ),
        "md_since_1000_plus": float(
            mechanism_config["maximum_long_tail_1000_plus_regression_ft"]
        ),
        "hidden_like_spatial": float(
            mechanism_config["maximum_hidden_like_spatial_regression_ft"]
        ),
        "hidden_like_typewell_purged": float(
            mechanism_config[
                "maximum_hidden_like_typewell_purged_regression_ft"
            ]
        ),
    }
    non_regression_scopes = {
        scope: float(
            scope_row(metrics, scope)[
                "delta_rmse_candidate_minus_control"
            ]
        )
        <= limit
        for scope, limit in non_regression_limits.items()
    }
    by_well_delta = by_well["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well = float(by_well_delta.max())
    exp410_metrics = episode_metrics["exp410"]
    exp408_metrics = episode_metrics["exp408"]
    exp410_candidate_sse = float(exp410_metrics["candidate_sse"].sum())
    exp410_control_sse = float(exp410_metrics["control_sse"].sum())
    exp410_sse_reduction = (
        1.0 - exp410_candidate_sse / exp410_control_sse
    )
    exp408_candidate_sse = float(exp408_metrics["candidate_sse"].sum())
    exp408_control_sse = float(exp408_metrics["control_sse"].sum())
    exp408_sse_reduction = (
        1.0 - exp408_candidate_sse / exp408_control_sse
    )
    fixed_episode = frame.loc[frame["exp410_fixed_episode"].to_numpy(bool)]
    baseline_outside_rate = float(
        (
            fixed_episode[
                "exp410_baseline_predictive_truth_support_fraction"
            ]
            < 0.5
        ).mean()
    )
    candidate_outside_rate = float(
        (
            fixed_episode[
                "candidate_predictive_truth_support_fraction"
            ]
            < 0.5
        ).mean()
    )
    support_outside_reduction_pp = 100.0 * (
        baseline_outside_rate - candidate_outside_rate
    )
    mechanism = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            mechanism_config["minimum_direct_rmse_gain_vs_scale5_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(
            mechanism_config["minimum_improved_folds_vs_scale5"]
        ),
        "raw_gr_observed_improvement_ft": float(observed["improvement_ft"]),
        "minimum_raw_gr_observed_improvement_ft": float(
            mechanism_config["minimum_raw_gr_observed_gain_ft"]
        ),
        "non_regression_scopes": non_regression_scopes,
        "by_well_rmse_delta_p95": by_well_p95,
        "maximum_by_well_rmse_delta_p95": float(
            mechanism_config["maximum_by_well_delta_rmse_p95_ft"]
        ),
        "worst_well_rmse_regression": worst_well,
        "maximum_worst_well_rmse_regression": float(
            mechanism_config["maximum_worst_well_regression_ft"]
        ),
        "exp410_episode_sse_reduction_fraction": exp410_sse_reduction,
        "minimum_exp410_episode_sse_reduction_fraction": float(
            mechanism_config[
                "minimum_exp410_episode_sse_reduction_fraction"
            ]
        ),
        "exp408_episode_sse_reduction_fraction": exp408_sse_reduction,
        "minimum_exp408_episode_sse_reduction_fraction": float(
            mechanism_config[
                "minimum_exp408_hmm_episode_sse_reduction_fraction"
            ]
        ),
        "exp410_baseline_majority_seed_support_outside_rate": (
            baseline_outside_rate
        ),
        "candidate_majority_seed_support_outside_rate": (
            candidate_outside_rate
        ),
        "support_outside_rate_reduction_percentage_points": (
            support_outside_reduction_pp
        ),
        "minimum_support_outside_rate_reduction_percentage_points": float(
            mechanism_config[
                "minimum_exp410_support_outside_reduction_percentage_points"
            ]
        ),
    }
    mechanism["passed"] = bool(
        mechanism["improvement_ft"] >= mechanism["minimum_improvement_ft"]
        and improved_folds >= mechanism["minimum_improved_folds"]
        and mechanism["raw_gr_observed_improvement_ft"]
        >= mechanism["minimum_raw_gr_observed_improvement_ft"]
        and all(non_regression_scopes.values())
        and by_well_p95 <= mechanism["maximum_by_well_rmse_delta_p95"]
        and worst_well <= mechanism["maximum_worst_well_rmse_regression"]
        and exp410_sse_reduction
        >= mechanism["minimum_exp410_episode_sse_reduction_fraction"]
        and exp408_sse_reduction
        >= mechanism["minimum_exp408_episode_sse_reduction_fraction"]
        and support_outside_reduction_pp
        >= mechanism[
            "minimum_support_outside_rate_reduction_percentage_points"
        ]
    )
    standalone = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "exp226_final_rmse": float(overall["exp226_final_rmse"]),
        "improvement_vs_exp226_ft": float(
            overall["improvement_vs_exp226_ft"]
        ),
        "minimum_improvement_vs_exp226_ft": float(
            adoption_config["minimum_gain_vs_exp226_final_ft"]
        ),
        "improved_folds_vs_exp226": improved_folds_vs_exp226,
        "minimum_improved_folds_vs_exp226": int(
            adoption_config["minimum_improved_folds_vs_exp226_final"]
        ),
    }
    standalone["passed"] = bool(
        mechanism["passed"]
        and standalone["improvement_vs_exp226_ft"]
        >= standalone["minimum_improvement_vs_exp226_ft"]
        and improved_folds_vs_exp226
        >= standalone["minimum_improved_folds_vs_exp226"]
    )
    physical = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "physical_anchor_rmse": float(overall["physical_anchor_rmse"]),
        "improvement_vs_physical_anchor_ft": float(
            overall["improvement_vs_physical_anchor_ft"]
        ),
        "minimum_improvement_vs_physical_anchor_ft": float(
            physical_config["minimum_gain_vs_exp263_physical_blend_ft"]
        ),
        "improved_folds_vs_physical_anchor": improved_folds_vs_physical,
        "minimum_improved_folds_vs_physical_anchor": int(
            physical_config[
                "minimum_improved_folds_vs_exp263_physical_blend"
            ]
        ),
    }
    physical["passed"] = bool(
        standalone["passed"]
        and physical["improvement_vs_physical_anchor_ft"]
        >= physical["minimum_improvement_vs_physical_anchor_ft"]
        and improved_folds_vs_physical
        >= physical["minimum_improved_folds_vs_physical_anchor"]
    )
    mechanism_positive = bool(technical["passed"] and mechanism["passed"])
    passed = bool(
        mechanism_positive and standalone["passed"] and physical["passed"]
    )
    deterministic_anchor_eligible = bool(
        passed and probe_byte_identical and probe_schedule_parity
    )
    if physical["passed"]:
        decision = str(
            get_nested(config, "guards.decision.physical_anchor_pass_action")
        )
    elif standalone["passed"]:
        decision = str(
            get_nested(config, "guards.decision.standalone_pass_action")
        )
    elif mechanism_positive:
        decision = str(
            get_nested(config, "guards.decision.mechanism_only_action")
        )
    else:
        decision = str(get_nested(config, "guards.decision.full_fail_action"))
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "mechanism_positive": mechanism_positive,
        "decision": decision,
        "technical_gate": technical,
        "mechanism_gate": mechanism,
        "standalone_adoption_gate": standalone,
        "physical_anchor_gate": physical,
        "deterministic_anchor_eligible": deterministic_anchor_eligible,
        "deterministic_anchor_blocker": (
            None
            if deterministic_anchor_eligible
            else "scientific_gate_or_probe_parity_failed"
        ),
        "failure_action": str(
            get_nested(config, "guards.decision.full_fail_action")
        ),
    }


# %% [markdown]
# ## 12. Generated artifacts and stage orchestration


# %%
def artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }
    if path.suffix == ".gz":
        report["decompressed_sha256"] = inspect_gzip_csv(path)["decompressed_sha256"]
    return report


def true_suffix_physical_rate(
    well: str,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "TVT", "TVT_input"],
    )
    suffix = horizontal["TVT_input"].isna().to_numpy(bool)
    eval_indices = np.flatnonzero(suffix).astype(np.int64)
    if len(eval_indices) == 0 or int(eval_indices[0]) == 0:
        raise ValueError(f"{well}: invalid suffix boundary for rate readout")
    previous_indices = np.concatenate(
        [[int(eval_indices[0]) - 1], eval_indices[:-1]]
    )
    surface = (
        horizontal["TVT"].to_numpy(np.float64)
        + horizontal["Z"].to_numpy(np.float64)
    )
    delta_surface = surface[eval_indices] - surface[previous_indices]
    delta_md = np.maximum(
        horizontal["MD"].to_numpy(np.float64)[eval_indices]
        - horizontal["MD"].to_numpy(np.float64)[previous_indices],
        1.0,
    )
    rate = delta_surface / delta_md
    if not np.isfinite(rate).all():
        raise ValueError(f"{well}: non-finite truth rate")
    return eval_indices, rate


def stage0_trigger_direction_readout(
    frame: pd.DataFrame,
    raw_dir: Path,
    *,
    horizon_rows: int = 32,
) -> pd.DataFrame:
    columns = [
        "well_id",
        "fold",
        "row_idx",
        "suffix_offset",
        "trigger_direction",
        "eligible",
        "future_rate_change",
        "true_direction",
        "agrees",
    ]
    rows: list[dict[str, Any]] = []
    fixed32 = frame.loc[
        frame["stage0_source"].eq("hmm_fixed32")
        & frame["hmm_trigger_direction"].ne(0)
    ]
    for well, selected in fixed32.groupby("well_id", sort=True):
        row_indices, true_rate = true_suffix_physical_rate(
            str(well),
            raw_dir,
        )
        offset_by_row = {
            int(row): offset for offset, row in enumerate(row_indices)
        }
        for trigger in selected.itertuples(index=False):
            offset = offset_by_row[int(trigger.row_idx)]
            past_start = offset - horizon_rows + 1
            future_end = offset + 1 + horizon_rows
            eligible = past_start >= 0 and future_end <= len(true_rate)
            true_direction = 0
            change = math.nan
            if eligible:
                past = true_rate[past_start : offset + 1]
                future = true_rate[offset + 1 : future_end]
                eligible = bool(
                    np.isfinite(past).all() and np.isfinite(future).all()
                )
                if eligible:
                    change = float(np.median(future) - np.median(past))
                    true_direction = int(np.sign(change))
            rows.append(
                {
                    "well_id": str(well),
                    "fold": int(trigger.fold),
                    "row_idx": int(trigger.row_idx),
                    "suffix_offset": int(trigger.suffix_offset),
                    "trigger_direction": int(trigger.hmm_trigger_direction),
                    "eligible": eligible,
                    "future_rate_change": change,
                    "true_direction": true_direction,
                    "agrees": bool(
                        eligible
                        and true_direction != 0
                        and true_direction
                        == int(trigger.hmm_trigger_direction)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def evaluate_stage0_gate(
    candidate: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    support_shards: Sequence[Mapping[str, Any]],
    shard_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _require_frozen_prediction(frozen)
    preflight = preflight_inputs(config)
    raw_dir = train_data_dir(config)
    truth = pd.concat(
        [
            load_unknown_suffix_truth(well, raw_dir)
            for well in sorted(candidate["well_id"].astype(str).unique())
        ],
        ignore_index=True,
    )
    frame = align_on_id(
        candidate,
        truth,
        ["true_tvt"],
        label="Stage 0 late suffix truth",
    )
    frame = attach_candidate_predictive_support(frame, support_shards)
    control_spec = input_spec(config, "exp404_frozen_predictions")
    control_column = str(control_spec["control_column"])
    control = pd.read_csv(
        preflight["paths"]["exp404_frozen_predictions"],
        usecols=["id", control_column],
        dtype={"id": str},
        compression="gzip",
    )
    frame = align_on_id(
        frame,
        control,
        [control_column],
        label="Stage 0 saved exp404 control",
    )
    exp226 = pd.read_csv(
        preflight["paths"]["exp226_fold_safe_geometry"],
        usecols=["well_id", "row_idx", "fold"],
        dtype={"well_id": str},
        compression="gzip",
    )
    frame = frame.merge(
        exp226,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if frame["fold"].isna().any():
        raise ValueError("Stage 0 fold attachment is incomplete")
    fixed44, fixed44_report = load_stage0_fixed44_manifest(config)
    fixed32_path = Path(str(fixed44_report["fixed32_path"]))
    sentinel_path = Path(str(fixed44_report["sentinel_path"]))
    fixed32 = pd.read_csv(
        fixed32_path,
        dtype={"well": str},
    ).rename(columns={"well": "well_id"})
    sentinel = pd.read_csv(
        sentinel_path,
        dtype={"well": str},
    ).rename(columns={"well": "well_id"})
    roles = pd.concat(
        [
            fixed32[["well_id", "role"]].assign(
                stage0_source="hmm_fixed32"
            ),
            sentinel[["well_id", "representative_cause"]]
            .assign(stage0_source="pf_sentinel12", role="pf_sentinel")
            .rename(columns={"representative_cause": "sentinel_cause"}),
        ],
        ignore_index=True,
        sort=False,
    )
    frame = frame.merge(
        fixed44,
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        roles,
        on=["well_id", "stage0_source"],
        how="left",
        validate="many_to_one",
    )
    if frame["role"].isna().any():
        raise ValueError("Stage 0 late role attachment is incomplete")
    trigger_readout = stage0_trigger_direction_readout(frame, raw_dir)
    eligible_triggers = trigger_readout.loc[trigger_readout["eligible"]]
    direction_agreement = (
        float(eligible_triggers["agrees"].mean())
        if not eligible_triggers.empty
        else 0.0
    )
    per_fold_direction = {
        int(fold): float(
            eligible_triggers.loc[
                eligible_triggers["fold"].eq(fold),
                "agrees",
            ].mean()
        )
        if eligible_triggers["fold"].eq(fold).any()
        else 0.0
        for fold in range(5)
    }
    passing_direction_folds = sum(
        value
        > float(
            get_nested(
                config,
                "guards.stage_0_mechanism."
                "per_fold_direction_agreement_strictly_above",
            )
        )
        for value in per_fold_direction.values()
    )

    exp408_episodes = pd.read_csv(
        preflight["paths"]["exp408_persistent_episodes"],
        dtype={"episode_id": str, "well": str},
    )
    persistent_wells = set(
        fixed32.loc[fixed32["role"].eq("persistent"), "well_id"]
    )
    persistent_episodes = exp408_episodes.loc[
        exp408_episodes["well"].isin(persistent_wells)
    ].copy()
    trigger_rows_by_well = {
        str(well): set(group["row_idx"].astype(int))
        for well, group in frame.loc[
            frame["hmm_trigger_direction"].ne(0)
        ].groupby("well_id")
    }
    lead_rows = int(
        get_nested(config, "guards.stage_0_mechanism.pre_onset_lead_rows_min")
    )
    onset_rows: list[dict[str, Any]] = []
    for episode in persistent_episodes.itertuples(index=False):
        start = int(episode.start_row_idx)
        first_row = int(
            frame.loc[frame["well_id"].eq(str(episode.well)), "row_idx"].min()
        )
        eligible = start - first_row >= lead_rows
        triggers = trigger_rows_by_well.get(str(episode.well), set())
        covered = bool(
            eligible
            and any(start - lead_rows <= row < start for row in triggers)
        )
        onset_rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well_id": str(episode.well),
                "eligible": eligible,
                "covered": covered,
            }
        )
    onset_readout = pd.DataFrame(
        onset_rows,
        columns=["episode_id", "well_id", "eligible", "covered"],
    )
    eligible_onsets = onset_readout.loc[onset_readout["eligible"]]
    onset_coverage = (
        float(eligible_onsets["covered"].mean())
        if not eligible_onsets.empty
        else 0.0
    )

    active_by_well = (
        frame.groupby(["well_id", "role"], sort=True)["hmm_active"]
        .agg(["mean", "any"])
        .reset_index()
    )
    persistent_active_well_fraction = float(
        active_by_well.loc[
            active_by_well["role"].eq("persistent"),
            "any",
        ].mean()
    )
    control_active_well_fraction = float(
        active_by_well.loc[
            active_by_well["role"].eq("control"),
            "any",
        ].mean()
    )
    control_active_row_fraction = float(
        frame.loc[frame["role"].eq("control"), "hmm_active"].mean()
    )

    sentinel_wells = set(sentinel["well_id"])
    exp410_episodes = pd.read_csv(
        preflight["paths"]["exp410_persistent_episodes"],
        dtype={"episode_id": str, "well": str},
    )
    sentinel_episodes = exp410_episodes.loc[
        exp410_episodes["well"].isin(sentinel_wells)
    ].copy()
    sentinel_episode_rows = expand_fixed_episode_rows(
        sentinel_episodes
    )
    sentinel_frame = frame.merge(
        sentinel_episode_rows,
        on=["well_id", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    baseline = pd.concat(
        [
            pd.read_csv(
                path,
                usecols=[
                    "well",
                    "row_idx",
                    "predictive_truth_support_fraction",
                ],
                dtype={"well": str},
                compression="gzip",
            )
            for path in preflight["exp410_row_ledger_paths"]
        ],
        ignore_index=True,
    ).rename(
        columns={
            "well": "well_id",
            "predictive_truth_support_fraction": (
                "baseline_support_fraction"
            ),
        }
    )
    sentinel_frame = sentinel_frame.merge(
        baseline,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if sentinel_frame["baseline_support_fraction"].isna().any():
        raise ValueError("Stage 0 sentinel baseline support is incomplete")
    candidate_error = (
        sentinel_frame[PRIMARY_CANDIDATE].to_numpy(np.float64)
        - sentinel_frame["true_tvt"].to_numpy(np.float64)
    )
    control_error = (
        sentinel_frame[control_column].to_numpy(np.float64)
        - sentinel_frame["true_tvt"].to_numpy(np.float64)
    )
    sentinel_sse_reduction = 1.0 - float(np.square(candidate_error).sum()) / float(
        np.square(control_error).sum()
    )
    baseline_outside_rate = float(
        (sentinel_frame["baseline_support_fraction"] < 0.5).mean()
    )
    candidate_outside_rate = float(
        (
            sentinel_frame[
                "candidate_predictive_truth_support_fraction"
            ]
            < 0.5
        ).mean()
    )
    support_reduction_pp = 100.0 * (
        baseline_outside_rate - candidate_outside_rate
    )
    well_regressions: list[float] = []
    for _, selected in frame.loc[
        frame["well_id"].isin(sentinel_wells)
    ].groupby("well_id"):
        truth_values = selected["true_tvt"].to_numpy(np.float64)
        well_regressions.append(
            rmse(
                truth_values,
                selected[PRIMARY_CANDIDATE].to_numpy(np.float64),
            )
            - rmse(
                truth_values,
                selected[control_column].to_numpy(np.float64),
            )
        )
    worst_well_regression = max(well_regressions)

    parity = synthetic_kernel_parity_report()
    technical_config = dict(
        get_nested(config, "guards.stage_0_technical") or {}
    )
    mechanism_config = dict(
        get_nested(config, "guards.stage_0_mechanism") or {}
    )
    proposal = proposal_contract(config)
    active_fraction = float(frame["hmm_active"].mean())
    persistent_active_wells = int(
        active_by_well.loc[
            active_by_well["role"].eq("persistent"),
            "any",
        ].sum()
    )
    expected_counts = dict(get_nested(config, "execution.stage_0") or {})
    actual_counts = {
        "hmm_signal_well_runs": len(audit),
        "candidate_pf_well_runs": len(audit),
        "parent_hmm_control_reruns": 0,
        "parent_pf_control_reruns": 0,
        "exp226_reruns": 0,
        "seeds_per_well": int(get_nested(config, "model.pf.seeds")),
        "seed_well_trajectories": int(
            audit["seed_well_trajectories"].sum()
        ),
        "particles_per_seed": int(get_nested(config, "model.pf.particles")),
        "particle_starts": int(audit["particle_starts"].sum()),
    }
    technical = {
        "fixed44_manifest": fixed44_report,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique()),
        "finite_candidate_coverage": float(
            np.isfinite(frame[PRIMARY_CANDIDATE]).mean()
        ),
        "active_row_fraction": active_fraction,
        "persistent_active_wells": persistent_active_wells,
        "importance_ratio_maximum": float(
            audit["importance_ratio_maximum"].max()
        ),
        "inactive_weight_sum": proposal["inactive_weight_sum"],
        "active_weight_sum": proposal["active_weight_sum"],
        "execution_counts": actual_counts,
        "execution_count_match": actual_counts == expected_counts,
        "synthetic_kernel_parity": parity,
        "schedule_logical_sha256": frozen["schedule_logical_sha256"],
        "prediction_logical_sha256": frozen["logical_content_sha256"],
        "runtime_seconds": float(shard_summary["runtime"]["elapsed_seconds"]),
        "peak_rss_gb": float(shard_summary["runtime"]["peak_rss_gb"]),
        "truth_role_cause_reads_before_freeze": 0,
    }
    technical["passed"] = bool(
        technical["wells"]
        == int(technical_config["require_unique_wells"])
        and technical["folds"]
        == [
            int(value)
            for value in technical_config["require_expected_folds"]
        ]
        and technical["finite_candidate_coverage"]
        == float(technical_config["require_finite_candidate_coverage"])
        and technical_config["active_row_fraction_min"]
        <= active_fraction
        <= technical_config["active_row_fraction_max"]
        and persistent_active_wells
        >= int(technical_config["persistent_active_wells_min"])
        and technical["importance_ratio_maximum"]
        <= float(technical_config["maximum_importance_ratio"])
        and proposal["inactive_weight_sum"]
        == float(technical_config["require_inactive_and_active_weight_sum"])
        and proposal["active_weight_sum"]
        == float(technical_config["require_inactive_and_active_weight_sum"])
        and actual_counts == expected_counts
        and parity["all_guidance_zero_exp404_rng_parity"]
        and parity["hmm_weight_zero_exp419_rng_parity"]
        and technical["runtime_seconds"]
        <= float(get_nested(config, "runtime.hard_seconds_per_shard"))
        and technical["peak_rss_gb"]
        <= float(get_nested(config, "runtime.maximum_peak_rss_gb"))
    )
    mechanism = {
        "future_rate_direction_agreement": direction_agreement,
        "passing_direction_folds": passing_direction_folds,
        "per_fold_direction_agreement": per_fold_direction,
        "eligible_direction_triggers": len(eligible_triggers),
        "pre_onset_trigger_coverage": onset_coverage,
        "lead_time_eligible_episodes": len(eligible_onsets),
        "control_active_row_fraction": control_active_row_fraction,
        "persistent_active_well_fraction": (
            persistent_active_well_fraction
        ),
        "control_active_well_fraction": control_active_well_fraction,
        "persistent_minus_control_active_well_fraction": (
            persistent_active_well_fraction - control_active_well_fraction
        ),
        "pf_sentinel_episode_sse_reduction": sentinel_sse_reduction,
        "pf_sentinel_support_outside_reduction_percentage_points": (
            support_reduction_pp
        ),
        "pf_sentinel_worst_well_regression_ft": worst_well_regression,
    }
    mechanism["passed"] = bool(
        direction_agreement
        >= float(mechanism_config["future_rate_direction_agreement_min"])
        and passing_direction_folds
        >= int(mechanism_config["passing_folds_min"])
        and onset_coverage
        >= float(mechanism_config["pre_onset_trigger_coverage_min"])
        and len(eligible_onsets)
        >= int(mechanism_config["lead_time_eligible_episodes_min"])
        and control_active_row_fraction
        <= float(mechanism_config["control_active_row_fraction_max"])
        and mechanism[
            "persistent_minus_control_active_well_fraction"
        ]
        >= float(
            mechanism_config[
                "persistent_minus_control_active_well_fraction_min"
            ]
        )
        and sentinel_sse_reduction
        >= float(mechanism_config["pf_sentinel_episode_sse_reduction_min"])
        and support_reduction_pp
        >= float(
            mechanism_config[
                "pf_sentinel_support_outside_reduction_percentage_points_min"
            ]
        )
        and worst_well_regression
        <= float(
            mechanism_config[
                "pf_sentinel_worst_well_regression_max_ft"
            ]
        )
    )
    passed = bool(technical["passed"] and mechanism["passed"])
    gate = {
        "experiment": EXPERIMENT_NAME,
        "scope": "stage_0_fixed44",
        "passed": passed,
        "decision": (
            "eligible_for_separate_full_oof_approval"
            if passed
            else str(
                get_nested(
                    config,
                    "guards.decision.stage_0_fail_action",
                )
            )
        ),
        "technical_gate": technical,
        "mechanism_gate": mechanism,
        "pooled_rmse_used_for_promotion": False,
        "same_oof_rescue_allowed": False,
    }
    return gate, {
        "late_frame": frame,
        "trigger_direction_readout": trigger_readout,
        "episode_onset_readout": onset_readout,
        "sentinel_episode_frame": sentinel_frame,
    }


def load_optional_probe_report(config: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = get_nested(config, "reproducibility.probe_report")
    if not isinstance(spec, Mapping) or not spec.get("filename"):
        return None
    path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    report = json.loads(path.read_text())
    expected_sha = spec.get("expected_sha256")
    if expected_sha and sha256_path(path) != str(expected_sha):
        raise ValueError("probe report SHA mismatch")
    return report


def resolve_shard_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [Path(str(value)) for value in get_nested(config, "execution.merge_shard_dirs") or []]
    if len(roots) != SHARD_COUNT:
        raise ValueError("execution.merge_shard_dirs must contain four ordered shard roots")
    return roots


def run_stage0_stage(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    shard_summary = run_shard(
        config,
        0,
        scope="stage_0",
        require_run_approval=require_run_approval,
    )
    output = artifact_dir()
    prediction_path = (
        output / f"{OUTPUT_PREFIX}_stage0_candidate_predictions.csv.gz"
    )
    audit_path = output / f"{OUTPUT_PREFIX}_stage0_well_audit.csv"
    support_minimum_path = (
        output
        / f"{OUTPUT_PREFIX}_stage0_predictive_support_min_float32.npy"
    )
    support_maximum_path = (
        output
        / f"{OUTPUT_PREFIX}_stage0_predictive_support_max_float32.npy"
    )
    candidate = pd.read_csv(
        prediction_path,
        dtype={"id": str, "well_id": str},
        compression="gzip",
    )
    audit = pd.read_csv(audit_path, dtype={"well_id": str})
    frozen = dict(shard_summary["frozen_prediction"])
    support_shards = [
        {
            "prediction_path": str(prediction_path),
            "minimum_path": str(support_minimum_path),
            "maximum_path": str(support_maximum_path),
        }
    ]
    gate, readouts = evaluate_stage0_gate(
        candidate,
        audit,
        frozen,
        support_shards,
        shard_summary,
        config,
    )
    paths: dict[str, Path] = {
        "scientific_gate": output / f"{OUTPUT_PREFIX}_stage0_gate.json",
        "late_frame": output / f"{OUTPUT_PREFIX}_stage0_late_frame.csv.gz",
        "trigger_direction_readout": (
            output / f"{OUTPUT_PREFIX}_stage0_trigger_direction_readout.csv"
        ),
        "episode_onset_readout": (
            output / f"{OUTPUT_PREFIX}_stage0_episode_onset_readout.csv"
        ),
        "sentinel_episode_frame": (
            output / f"{OUTPUT_PREFIX}_stage0_sentinel_episode_frame.csv.gz"
        ),
    }
    write_json(paths["scientific_gate"], gate)
    write_deterministic_gzip_csv(readouts["late_frame"], paths["late_frame"])
    readouts["trigger_direction_readout"].to_csv(
        paths["trigger_direction_readout"],
        index=False,
    )
    readouts["episode_onset_readout"].to_csv(
        paths["episode_onset_readout"],
        index=False,
    )
    write_deterministic_gzip_csv(
        readouts["sentinel_episode_frame"],
        paths["sentinel_episode_frame"],
    )
    artifact_manifest = pd.DataFrame(
        [
            {"name": name, **artifact_report(path)}
            for name, path in paths.items()
        ]
    ).sort_values("name", kind="mergesort")
    artifact_manifest_path = (
        output / f"{OUTPUT_PREFIX}_stage0_artifact_manifest.csv"
    )
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    status = (
        "stage0_passed_pending_separate_full_oof_approval"
        if gate["passed"]
        else "stage0_failed_closed_without_same_oof_rescue"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "scope": "stage_0_fixed44",
        "status": status,
        "route": "pf_beam",
        "scientific_variants": 1,
        "hmm_signal_well_runs": 44,
        "candidate_pf_well_runs": 44,
        "parent_hmm_control_reruns": 0,
        "parent_pf_control_reruns": 0,
        "exp226_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "gpu_runs": 0,
        "frozen_prediction": frozen,
        "gate": gate,
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_stage0_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def run_probe_stage(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=require_run_approval)
    if require_run_approval and not bool(get_nested(config, "execution.probe_run_approved")):
        raise RuntimeError("exp420 probe rerun is not approved")
    spec = dict(get_nested(config, "reproducibility.probe_source") or {})
    merged_path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    merged = pd.read_csv(merged_path, dtype={"id": str, "well_id": str})
    probe_well = str(get_nested(config, "reproducibility.probe_well"))
    preflight = preflight_inputs(config)
    geometry = load_fold_safe_geometry(
        preflight["paths"]["exp226_fold_safe_geometry"],
        config,
    )
    geometry = geometry.loc[geometry["well_id"].eq(probe_well)].copy()
    report = probe_rerun_report(
        merged,
        train_data_dir(config),
        config,
        probe_well,
        geometry,
    )
    output_path = artifact_dir() / f"{OUTPUT_PREFIX}_probe_rerun_report.json"
    write_json(output_path, report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return report


def run_merge_stage(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    contract = validate_scientific_contract(
        config,
        require_run_approval=require_run_approval,
    )
    started = time.time()
    preflight = preflight_inputs(config)
    ledger = TruthAccessLedger()
    output = artifact_dir()
    shard_roots = resolve_shard_roots(config)
    candidate, audit, frozen, merged_paths, support_shards = merge_shard_outputs(
        shard_roots,
        output,
        config,
        ledger=ledger,
    )
    shard_summaries = [
        json.loads(_artifact_file(root, f"{OUTPUT_PREFIX}_shard{index}_summary.json").read_text())
        for index, root in enumerate(shard_roots)
    ]
    raw_dir = train_data_dir(config)
    expected_manifest = assign_lpt_shards(build_raw_well_manifest(config, raw_dir))
    observed_manifest = pd.read_csv(
        merged_paths["merged_well_manifest"],
        dtype={"well_id": str},
    ).sort_values("well_id", kind="mergesort")
    manifest_columns = ["well_id", "suffix_rows", "shard_index"]
    expected_values = expected_manifest[manifest_columns].astype(
        {"well_id": str, "suffix_rows": np.int64, "shard_index": np.int64}
    )
    observed_values = observed_manifest[manifest_columns].astype(
        {"well_id": str, "suffix_rows": np.int64, "shard_index": np.int64}
    )
    if not np.array_equal(
        expected_values.to_numpy(),
        observed_values.to_numpy(),
    ):
        raise ValueError("merged shard manifest does not match deterministic raw-data LPT")
    frame, episode_sets, late_attachment = load_late_readout_frame(
        candidate,
        frozen,
        preflight,
        support_shards,
        raw_dir,
        config,
        ledger,
    )
    metrics, by_well, episode_metrics = build_metric_outputs(
        frame,
        episode_sets,
    )
    probe_report = load_optional_probe_report(config)
    gate = evaluate_full_gate(
        frame,
        metrics,
        by_well,
        episode_metrics,
        audit,
        frozen,
        ledger,
        shard_summaries,
        config,
        probe_report=probe_report,
    )
    paths = {
        **merged_paths,
        "overall_fold_scope_metrics": output / f"{OUTPUT_PREFIX}_overall_fold_scope_metrics.csv",
        "by_well_metrics": output / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "exp410_episode_metrics": (
            output / f"{OUTPUT_PREFIX}_exp410_episode_metrics.csv"
        ),
        "exp408_episode_metrics": (
            output / f"{OUTPUT_PREFIX}_exp408_episode_metrics.csv"
        ),
        "scientific_gate": output / f"{OUTPUT_PREFIX}_scientific_gate.json",
        "scientific_contract": output / f"{OUTPUT_PREFIX}_scientific_contract.json",
    }
    metrics.to_csv(paths["overall_fold_scope_metrics"], index=False)
    by_well.to_csv(paths["by_well_metrics"], index=False)
    episode_metrics["exp410"].to_csv(
        paths["exp410_episode_metrics"],
        index=False,
    )
    episode_metrics["exp408"].to_csv(
        paths["exp408_episode_metrics"],
        index=False,
    )
    write_json(paths["scientific_gate"], gate)
    write_json(paths["scientific_contract"], contract)
    artifact_manifest = pd.DataFrame(
        [{"name": name, **artifact_report(path)} for name, path in paths.items()]
    ).sort_values("name", kind="mergesort")
    artifact_manifest_path = output / f"{OUTPUT_PREFIX}_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    if gate["passed"]:
        status = (
            "train_side_hmm_guided_defensive_mixture_all_gates_passed_"
            "no_automatic_downstream"
        )
    elif gate["mechanism_positive"]:
        status = (
            "train_side_hmm_guided_defensive_mixture_mechanism_positive_"
            "no_inference_promotion"
        )
    else:
        status = "train_side_hmm_guided_defensive_mixture_gate_failed_closed"
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "runtime_seconds_merge_and_readout": time.time() - started,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "scientific_variants": 1,
        "candidate_pf_well_runs": int(audit["well_id"].nunique()),
        "hmm_signal_well_runs": int(audit["well_id"].nunique()),
        "parent_hmm_control_reruns": 0,
        "parent_pf_control_reruns": 0,
        "exp226_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "frozen_prediction": frozen,
        "truth_attachment": late_attachment,
        "gate": gate,
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
        },
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    overall = scope_row(metrics, "overall")
    metrics_json = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": float(overall["candidate_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": overall.to_dict(),
        "gate": gate,
        "prediction_sha256": frozen["logical_content_sha256"],
        "hmm_schedule_sha256": frozen["schedule_logical_sha256"],
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Train-side candidate only. No saved exp404/exp209 control rerun, "
            "exp226 rerun, model, raw-test prediction, inference, or submission "
            "is produced."
        ),
    }
    write_json(metrics_output_path(), metrics_json)
    print(metrics.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def selected_stage(config: Mapping[str, Any]) -> str | None:
    value = os.environ.get("EXP420_STAGE") or get_nested(config, "execution.selected_stage")
    if value in (None, "", "preview"):
        return None
    return str(value)


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    if stage == "stage0":
        return run_stage0_stage(config)
    if stage in {"shard", "full_shard"}:
        raw_index = os.environ.get("EXP420_SHARD_INDEX")
        shard_index = (
            int(raw_index)
            if raw_index is not None
            else int(get_nested(config, "execution.selected_shard_index"))
        )
        return run_shard(config, shard_index, scope="full")
    if stage == "probe":
        return run_probe_stage(config)
    if stage == "merge":
        return run_merge_stage(config)
    raise ValueError(f"unknown exp420 execution stage: {stage}")


# %% [markdown]
# ## 13. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    PREVIEW = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(CONFIG, "experiment.route"),
        "parent": get_nested(CONFIG, "lineage.parent"),
        "geometry_parent": get_nested(CONFIG, "lineage.geometry_parent"),
        "hmm_kernel_parent": get_nested(CONFIG, "lineage.hmm_kernel_parent"),
        "hmm_schedule_reference": get_nested(
            CONFIG,
            "lineage.hmm_schedule_reference",
        ),
        "primary_candidate": PRIMARY_CANDIDATE,
        "hmm_schedule_contract": hmm_schedule_contract(CONFIG),
        "proposal_contract": proposal_contract(CONFIG),
        "scientific_variants": get_nested(CONFIG, "execution.scientific_variants"),
        "stage_0_execution": get_nested(CONFIG, "execution.stage_0"),
        "full_execution": get_nested(CONFIG, "execution.full"),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "canonical_notebook_adoption_approved": get_nested(
            CONFIG,
            "execution.canonical_notebook_adoption_approved",
        ),
        "kaggle_package_approved": get_nested(CONFIG, "execution.kaggle_package_approved"),
        "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
        "stage_0_run_approved": get_nested(
            CONFIG,
            "execution.stage_0_run_approved",
        ),
        "full_run_approved": get_nested(CONFIG, "execution.full_run_approved"),
        "selected_stage": selected_stage(CONFIG),
    }
    print(json.dumps(to_jsonable(PREVIEW), indent=2, sort_keys=True))
    SUMMARY = run_selected_stage(CONFIG)
