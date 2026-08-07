# %% [markdown]
# # exp500 exp490 mean-reversion residual likelihood-PF — train
#
# This Jupytext source implements the Stage 0 audit and the explicitly
# user-overridden Stage 1 full OOF. It transfers exp490's fixed K16-segment half-life
# shrink onto the exp486 residual-offset likelihood-PF. Candidate prediction,
# K16/rho, particle diagnostics, seed evidence, and content SHAs freeze before
# truth, saved controls, roles, folds, or persistent episodes are read.
# The Stage 0 safety failures remain negative evidence. Stage 1 uses the exact
# same single variant in four target-free CPU shards plus a strict truth-late
# merge. Inference and submission remain disabled.

# %% [markdown]
# ## Contents
# 1. Imports and notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Fixed44 identity scope, exp226 allowlist, and leakage ledger
# 5. Exp404 likelihood-PF input preparation
# 6. K16 half-life and mean-reverting residual PF kernel
# 7. Seed aggregation, parity, and state-transition contracts
# 8. Target-free single-variant generation and freeze
# 9. Truth-late fixed44 readout and fail-closed gates
# 10. Generated artifacts and guarded Stage 0 orchestration
# 11. Stage 1 target-free shards, strict merge, and full OOF gates
# 12. Setup, execution selection, and configuration preview

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


EXPERIMENT_NAME = "exp500_exp490_mean_reversion_residual_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
ACTIVE_VARIANT = "k16_half_life_mean_reverting_residual_likpf"
PREDICTION_COLUMN = "likpf_scale5_k16_mean_reverting_residual_offset"
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
PARENT_RESIDUAL = "likpf_scale5_slow_residual_offset"
ACTIVE_VARIANTS = (ACTIVE_VARIANT,)
PREDICTION_COLUMNS = (PREDICTION_COLUMN,)
SHARD_COUNT = 4
GEOMETRY_ALLOWLIST = ("well_id", "row_idx", "suffix_offset", "tvt_geop")
RESIDUAL_LEDGER_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "dmd",
    "k16_segment_id",
    "k16_segment_span",
    "rho",
    "geometry_delta",
    "filtered_offset_mean",
    "filtered_offset_std",
    "filtered_offset_rate_mean",
    "filtered_offset_rate_std",
    "particle_drift_minus_geometry_delta",
    "typewell_support_fraction",
    "particle_weight_sum",
    "offset_edge_mass",
    "effective_sample_size",
    "resampled_seed_fraction",
)
SEED_EVIDENCE_COLUMNS = (
    "well_id",
    "seed_index",
    "seed_value",
    "total_log_evidence",
    "temperature_weight",
    "resampling_count",
    "minimum_effective_sample_size",
    "position_clip_count",
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP500_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp500 config not found; checked={checked}")


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


def is_gzip_payload(path: str | Path) -> bool:
    with Path(path).open("rb") as file_pointer:
        return file_pointer.read(2) == b"\x1f\x8b"


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    selected = Path(path)
    opener = gzip.open if is_gzip_payload(selected) else Path.open
    with opener(selected, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_csv_payload(path: str | Path) -> str:
    selected = Path(path)
    return sha256_decompressed_csv(selected) if is_gzip_payload(selected) else sha256_path(selected)


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
    expected = {
        "scientific_variants": 1,
        "stage_0.candidate_pf_well_runs": 44,
        "stage_0.seed_well_trajectories": 5632,
        "stage_0.particle_starts": 2816000,
        "stage_0.parent_pf_control_reruns": 0,
        "stage_0.hmm_well_runs": 0,
        "stage_0.beam_well_runs": 0,
        "stage_1.candidate_pf_well_runs": 773,
        "stage_1.seed_well_trajectories": 98944,
        "stage_1.particle_starts": 49472000,
        "stage_1.parent_pf_control_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "gpu_runs": 0,
    }
    observed_counts: dict[str, int] = {}
    for key, required in expected.items():
        observed = int(get_nested(config, f"execution.{key}", -1))
        if observed != required:
            raise ValueError(
                f"exp500 execution count changed: {key}={observed}, expected={required}"
            )
        observed_counts[key] = observed
    if not bool(get_nested(config, "execution.stage_1_implementation_approved", False)):
        raise ValueError("exp500 Stage 1 implementation approval is required")
    if not bool(get_nested(config, "execution.stage_1_stage0_gate_override_approved", False)):
        raise ValueError("exp500 Stage 1 requires the explicit Stage 0 gate override")
    if bool(get_nested(config, "implementation.stage_0_all_pass", True)):
        raise ValueError("exp500 Stage 0 fail must remain preserved")
    if bool(get_nested(config, "execution.inference_approved", False)) or bool(
        get_nested(config, "execution.submission_approved", False)
    ):
        raise ValueError("exp500 inference/submission must remain disabled")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise RuntimeError("exp500 Kaggle push is not approved")
        if not bool(get_nested(config, "execution.stage_1_run_approved", False)):
            raise RuntimeError("exp500 Stage 1 run is not approved")
    return observed_counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    transition = dict(get_nested(config, "model.transition") or {})
    initialization = dict(get_nested(config, "model.initialization") or {})
    particle_filter = dict(get_nested(config, "model.particle_filter") or {})
    mean_reversion = dict(get_nested(config, "model.mean_reversion") or {})
    segment = dict(get_nested(config, "model.k16_segment") or {})
    emission = dict(get_nested(config, "model.emission") or {})
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "scientific_parent": "exp486_exp226_geometry_residual_likelihood_pf",
        "parent_variant": "slow_residual_offset_state",
        "mechanism_parent": "exp490_geometry_centered_mean_reverting_offset_hmm",
        "active_variants": list(ACTIVE_VARIANTS),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "geometry_allowlist": list(GEOMETRY_ALLOWLIST),
        "state": ["residual_offset_from_exp226_geometry", "residual_offset_rate"],
        "output": "tvt_geop_plus_residual_offset",
        "k16": {
            "segments": int(segment["count"]),
            "boundary_owner": str(segment["boundary_transition_owner"]),
            "span_definition": str(segment["span_definition"]),
            "half_life_segments": float(mean_reversion["half_life_segments"]),
            "rho_formula": str(mean_reversion["rho_formula"]),
            "segment_cumulative_rho_target": float(
                mean_reversion["segment_cumulative_rho_target"]
            ),
        },
        "transition": {
            "rate_center": "momentum * rho_t * previous_rate",
            "offset_center": "rho_t * previous_offset + current_rate * dmd_t",
            "momentum": float(transition["momentum"]),
            "rate_noise": float(transition["rate_noise"]),
            "position_noise": float(transition["position_noise"]),
            "noise_after_center": True,
        },
        "initialization": initialization,
        "pf": particle_filter,
        "emission": emission,
        "rng": {
            "seed_formula": str(get_nested(config, "reproducibility.seed_formula")),
            "variant_name_in_seed": False,
            "global_rng_outside_numba_kernel": False,
            "well_and_seed_stream_independent_of_worker_order": True,
        },
        "saved_controls_rerun": False,
        "stage_0_is_cv": False,
        "stage_1_implemented": True,
        "stage_1_user_override": {
            "approved": True,
            "scope": "unchanged_single_variant_full_oof_only",
            "stage_0_fail_preserved": True,
        },
        "inference_enabled": False,
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
        "experiment.status": "stage1_fail_closed_under_override",
        "lineage.parent": "exp486_exp226_geometry_residual_likelihood_pf",
        "lineage.parent_variant": "slow_residual_offset_state",
        "lineage.mechanism_parent": "exp490_geometry_centered_mean_reverting_offset_hmm",
        "implementation.enabled": True,
        "implementation.scope": "stage0_and_stage1_single_variant",
        "implementation.implementation_approval_received": True,
        "implementation.canonical_notebook_adopted": True,
        "implementation.kaggle_package_created": True,
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": True,
        "implementation.stage_1_override_received": True,
        "implementation.stage_1_override_preserves_stage_0_fail": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.fixed44_is_cv": False,
        "model.active_variants": list(ACTIVE_VARIANTS),
        "model.k16_segment.count": 16,
        "model.mean_reversion.half_life_segments": 1.0,
        "model.transition.momentum": 0.998,
        "model.transition.rate_noise": 0.002,
        "model.transition.position_noise": 0.005,
        "model.initialization.offset_spread_ft": 4.5,
        "model.initialization.rate_center": 0.0,
        "model.initialization.rate_spread": 0.01,
        "model.particle_filter.particles": 500,
        "model.particle_filter.seeds": 128,
        "model.particle_filter.rough_position": 0.1,
        "model.particle_filter.rough_rate": 0.001,
        "model.particle_filter.resample_threshold_fraction": 0.5,
        "model.particle_filter.primary_seed_weighting_temperature": 5.0,
        "model.emission.gr_scale_multiplier": 1.0,
        "execution.run_stage_0": False,
        "execution.canonical_notebook_adoption_approved": True,
        "execution.kaggle_package_approved": True,
        "execution.kaggle_push_approved": True,
        "execution.stage_0_run_approved": True,
        "execution.stage_1_implementation_approved": True,
        "execution.stage_1_run_approved": True,
        "execution.stage_1_stage0_gate_override_approved": True,
        "execution.inference_approved": False,
        "execution.submission_approved": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, required in expected.items():
        observed = get_nested(config, key)
        if observed != required:
            raise ValueError(
                f"exp500 scientific contract mismatch: {key}={observed!r}, "
                f"expected={required!r}"
            )
    safe_columns = list(
        get_nested(config, "data.exp226_oof_geometry.prediction_time_usecols") or []
    )
    forbidden = set(
        get_nested(config, "data.exp226_oof_geometry.forbidden_columns") or []
    )
    if safe_columns != list(GEOMETRY_ALLOWLIST):
        raise ValueError("exp500 exp226 prediction-time allowlist changed")
    if list(
        get_nested(config, "data.exp226_oof_geometry.postfreeze_columns") or []
    ) != ["fold", "tvt_pred"]:
        raise ValueError("exp500 exp226 truth-late readout columns changed")
    if forbidden != {"tvt_pred", "gr_delta", "tvt_true", "error", "abs_error"}:
        raise ValueError("exp500 exp226 forbidden-column contract changed")
    validate_execution_contract(config, require_run_approval=require_run_approval)
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Fixed44 identity scope, exp226 allowlist, and leakage ledger


# %%
@dataclass
class LeakageLedger:
    expected_variant_wells: int
    frozen_variant_wells: set[str] = field(default_factory=set)
    geometry_safe_rows_before_freeze: int = 0
    forbidden_geometry_columns_read_before_freeze: int = 0
    truth_rows_before_all_freeze: int = 0
    control_rows_before_all_freeze: int = 0
    role_fold_episode_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    control_rows_after_all_freeze: int = 0
    role_fold_episode_rows_after_all_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_variant_wells) == self.expected_variant_wells > 0

    def record_geometry_safe(self, rows: int) -> None:
        self.geometry_safe_rows_before_freeze += int(rows)

    def freeze(self, variant: str, well: str) -> None:
        self.frozen_variant_wells.add(f"{variant}::{well}")

    def _record_late(self, label: str, rows: int) -> None:
        before_name = f"{label}_before_all_freeze"
        after_name = f"{label}_after_all_freeze"
        if not self.all_frozen:
            setattr(self, before_name, int(getattr(self, before_name)) + int(rows))
            raise RuntimeError(f"{label} was read before the exp500 candidate froze")
        setattr(self, after_name, int(getattr(self, after_name)) + int(rows))

    def record_truth(self, rows: int) -> None:
        self._record_late("truth_rows", rows)

    def record_control(self, rows: int) -> None:
        self._record_late("control_rows", rows)

    def record_role_fold_episode(self, rows: int) -> None:
        self._record_late("role_fold_episode_rows", rows)

    def report(self) -> dict[str, Any]:
        return {
            "expected_variant_wells": self.expected_variant_wells,
            "frozen_variant_wells": len(self.frozen_variant_wells),
            "all_frozen": self.all_frozen,
            "before_freeze": {
                "geometry_safe_rows": self.geometry_safe_rows_before_freeze,
                "forbidden_geometry_columns": self.forbidden_geometry_columns_read_before_freeze,
                "truth_rows": self.truth_rows_before_all_freeze,
                "control_rows": self.control_rows_before_all_freeze,
                "role_fold_episode_rows": self.role_fold_episode_rows_before_all_freeze,
            },
            "after_freeze": {
                "truth_rows": self.truth_rows_after_all_freeze,
                "control_rows": self.control_rows_after_all_freeze,
                "role_fold_episode_rows": self.role_fold_episode_rows_after_all_freeze,
            },
        }


def asset_path(config: Mapping[str, Any], dotted_key: str) -> Path:
    spec = dict(get_nested(config, dotted_key) or {})
    local = str(spec.get("local") or "")
    if local:
        path = resolve_bootstrap_asset(str(spec["filename"]), local)
    else:
        path = resolve_existing(
            str(spec["filename"]), spec.get("candidates", []), spec.get("patterns", [])
        )
    expected = str(spec.get("expected_sha256") or "")
    if expected and sha256_path(path) != expected:
        raise ValueError(f"exp500 {dotted_key} SHA mismatch")
    return path


def load_fixed44_identity(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    fixed_path = asset_path(config, "data.stage_0_fixed32")
    sentinel_path = asset_path(config, "data.stage_0_pf_sentinel")
    fixed = pd.read_csv(fixed_path, usecols=["well"], dtype={"well": str})
    sentinel = pd.read_csv(sentinel_path, usecols=["well"], dtype={"well": str})
    fixed_wells = set(fixed["well"])
    sentinel_wells = set(sentinel["well"])
    overlap = fixed_wells & sentinel_wells
    wells = sorted(fixed_wells | sentinel_wells)
    if len(fixed_wells) != 32 or len(sentinel_wells) != 12 or overlap or len(wells) != 44:
        raise ValueError("exp500 fixed32/sentinel identity contract changed")
    return wells, {
        "fixed32_path": str(fixed_path),
        "fixed32_sha256": sha256_path(fixed_path),
        "sentinel_path": str(sentinel_path),
        "sentinel_sha256": sha256_path(sentinel_path),
        "fixed32_wells": 32,
        "sentinel_wells": 12,
        "overlap_wells": 0,
        "union_wells": 44,
        "columns_read_before_freeze": ["well"],
        "stable_well_order_sha256": mapping_sha256(wells),
    }


def load_fixed44_readout_after_freeze(
    config: Mapping[str, Any], ledger: LeakageLedger
) -> pd.DataFrame:
    fixed = pd.read_csv(asset_path(config, "data.stage_0_fixed32"), dtype={"well": str})
    sentinel = pd.read_csv(
        asset_path(config, "data.stage_0_pf_sentinel"), dtype={"well": str}
    )
    ledger.record_role_fold_episode(len(fixed) + len(sentinel))
    if (
        fixed["well"].nunique() != 32
        or fixed["role"].value_counts().to_dict() != {"control": 16, "persistent": 16}
        or set(fixed["fold"].astype(int)) != set(range(5))
        or sentinel["well"].nunique() != 12
    ):
        raise ValueError("exp500 fixed44 late role contract changed")
    fixed_roles = fixed[["well", "role", "fold"]].copy()
    fixed_roles["sentinel_cause"] = ""
    sentinel_roles = sentinel[["well", "representative_cause"]].rename(
        columns={"representative_cause": "sentinel_cause"}
    )
    sentinel_roles["role"] = "pf_sentinel"
    sentinel_roles["fold"] = -1
    return pd.concat(
        [fixed_roles, sentinel_roles[["well", "role", "fold", "sentinel_cause"]]],
        ignore_index=True,
    ).sort_values("well", kind="mergesort").reset_index(drop=True)


def geometry_input_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.exp226_oof_geometry") or {})
    path = resolve_existing(
        str(spec["filename"]), spec.get("candidates", []), spec.get("patterns", [])
    )
    if sha256_decompressed_csv(path) != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp500 exp226 geometry decompressed SHA mismatch")
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
        raise ValueError("exp500 geometry allowlist changed")
    geometry = pd.read_csv(
        path, usecols=safe_columns, dtype={"well_id": str}, compression="infer"
    ).loc[:, safe_columns]
    geometry["row_idx"] = pd.to_numeric(geometry["row_idx"], errors="raise").astype(np.int64)
    geometry["suffix_offset"] = pd.to_numeric(
        geometry["suffix_offset"], errors="raise"
    ).astype(np.int64)
    geometry["tvt_geop"] = pd.to_numeric(
        geometry["tvt_geop"], errors="raise"
    ).astype(np.float64)
    if wells is not None:
        geometry = geometry.loc[geometry["well_id"].isin(wells)].copy()
    if geometry.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(
        geometry["tvt_geop"]
    ).all():
        raise ValueError("exp500 geometry rows are duplicated or non-finite")
    if ledger is not None:
        ledger.record_geometry_safe(len(geometry))
    return geometry


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv", usecols=["MD", "Z", "GR", "TVT_input"]
    )
    for column in ("MD", "Z", "GR", "TVT_input"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["MD"].notna().all() or not frame["Z"].notna().all():
        raise ValueError(f"{well}: MD/Z contains missing values")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(
        drop=True
    )
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
# Prefix scale, missing-GR interpolation, and Type Well gridding remain fixed
# from exp486/exp404. No unknown-suffix truth is loaded in this section.


# %%
def uniform_typewell_grid(
    typewell_tvt: np.ndarray, typewell_gr: np.ndarray, *, step: float
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(typewell_tvt))
    maximum = float(np.max(typewell_tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    return (
        np.interp(grid_tvt, typewell_tvt, typewell_gr).astype(np.float64),
        minimum,
        float(step),
    )


def exp072_base_gr_scale(
    horizontal: pd.DataFrame, typewell_tvt: np.ndarray, typewell_gr: np.ndarray
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires a known prefix")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    raw_scale = float(
        np.nanstd(known_gr - np.interp(known_tvt, typewell_tvt, typewell_gr))
    )
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


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame, typewell: pd.DataFrame, *, grid_step: float = 0.2
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
        typewell_tvt, typewell_gr, step=grid_step
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
    if not np.isfinite(eval_md).all() or not np.isfinite(eval_gr).all():
        raise ValueError("likelihood-PF evaluation inputs are not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - float(last_known["MD"]),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "last_known_tvt": float(last_known["TVT_input"]),
        "last_known_md": float(last_known["MD"]),
        "scale_audit": exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr),
    }


def align_geometry_to_prepared(
    well: str, geometry_rows: pd.DataFrame, prepared: Mapping[str, Any]
) -> np.ndarray:
    ordered = geometry_rows.sort_values("suffix_offset", kind="mergesort")
    expected_rows = np.asarray(prepared["eval_indices"], dtype=np.int64)
    if (
        len(ordered) != len(expected_rows)
        or not np.array_equal(ordered["row_idx"].to_numpy(np.int64), expected_rows)
        or not np.array_equal(
            ordered["suffix_offset"].to_numpy(np.int64),
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
# ## 6. K16 half-life and mean-reverting residual PF kernel
#
# `rho_t` is the only scientific change from exp486's residual-state PF. The
# draw and resampling order remains initialization, rate noise, offset noise,
# GR weighting, systematic resampling, then roughening.


# %%
def k16_segment_half_life(
    unknown_md: np.ndarray, *, last_known_md: float, segment_count: int = 16
) -> dict[str, np.ndarray]:
    md = np.asarray(unknown_md, dtype=np.float64)
    if md.ndim != 1 or len(md) < int(segment_count):
        raise ValueError("K16 requires at least one destination row per segment")
    if not np.isfinite(md).all() or not np.isfinite(last_known_md):
        raise ValueError("K16 MD values must be finite")
    dmd = np.diff(np.concatenate([[float(last_known_md)], md]))
    if not np.all(dmd > 0.0):
        raise ValueError("every transition-entering dMD must be strictly positive")
    edges = np.linspace(0.0, float(len(md)), int(segment_count) + 1)
    destination_rows = np.arange(1, len(md) + 1, dtype=np.float64)
    segment_id = np.clip(
        np.searchsorted(edges[1:], destination_rows, side="left"),
        0,
        int(segment_count) - 1,
    ).astype(np.int16)
    segment_span = np.bincount(
        segment_id.astype(np.int64), weights=dmd, minlength=int(segment_count)
    ).astype(np.float64)
    segment_rows = np.bincount(
        segment_id.astype(np.int64), minlength=int(segment_count)
    ).astype(np.int64)
    if np.any(segment_rows <= 0) or np.any(segment_span <= 0.0):
        raise ValueError("every K16 destination segment must be non-empty and positive")
    rho = np.power(2.0, -dmd / segment_span[segment_id])
    if not np.isfinite(rho).all() or np.any(rho <= 0.0) or np.any(rho > 1.0):
        raise ValueError("rho must be finite and in (0, 1]")
    cumulative = np.asarray(
        [np.prod(rho[segment_id == segment]) for segment in range(int(segment_count))],
        dtype=np.float64,
    )
    return {
        "dmd": dmd,
        "segment_id": segment_id,
        "segment_span": segment_span,
        "segment_rows": segment_rows,
        "rho": rho,
        "segment_cumulative_rho": cumulative,
        "edges": edges,
    }


@njit(cache=True, nogil=True)
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
def _pf_residual_offset_allseeds(
    dmd_v: np.ndarray,
    rho_v: np.ndarray,
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
) -> tuple[np.ndarray, ...]:
    rows = len(dmd_v)
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
    particle_weight_sum = np.empty((seeds, rows))
    offset_edge_mass = np.empty((seeds, rows))
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
        for row in range(rows):
            delta_md = dmd_v[row]
            rho = rho_v[row]
            for particle in range(particles):
                offset_rate[particle] = (
                    momentum * rho * offset_rate[particle] + rate_noise * np.random.randn()
                )
                offset[particle] = (
                    rho * offset[particle]
                    + offset_rate[particle] * delta_md
                    + position_noise * np.random.randn()
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
                expected_gr = _interp1(grid_gr, tvt_value, grid_minimum, grid_step)
                zscore = (gr_v[row] - expected_gr) / gr_scale
                squared = min(zscore * zscore, 600.0)
                likelihood = max(np.exp(-0.5 * squared), 1e-300)
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            log_likelihood += np.log(max(average_likelihood, 1e-300))
            weight_sum = np.sum(weights)
            if weight_sum > 0.0:
                weights /= weight_sum
            else:
                weights[:] = 1.0 / particles
            mean_offset = np.sum(weights * offset)
            mean_rate = np.sum(weights * offset_rate)
            inverse_ess = np.sum(weights * weights)
            in_support = 0.0
            edge_mass = 0.0
            for particle in range(particles):
                tvt_value = geometry_tvt_v[row] + offset[particle]
                if grid_minimum <= tvt_value <= grid_maximum:
                    in_support += weights[particle]
                if tvt_value <= grid_minimum - 100.0 or tvt_value >= grid_maximum + 100.0:
                    edge_mass += weights[particle]
            variance_offset = np.sum(weights * (offset - mean_offset) ** 2)
            variance_rate = np.sum(weights * (offset_rate - mean_rate) ** 2)
            ess = 1.0 / inverse_ess
            offset_mean[seed_index, row] = mean_offset
            offset_std[seed_index, row] = np.sqrt(max(variance_offset, 0.0))
            offset_rate_mean[seed_index, row] = mean_rate
            offset_rate_std[seed_index, row] = np.sqrt(max(variance_rate, 0.0))
            support_fraction[seed_index, row] = in_support
            particle_weight_sum[seed_index, row] = np.sum(weights)
            offset_edge_mass[seed_index, row] = edge_mass
            effective_sample_size[seed_index, row] = ess
            minimum_ess[seed_index] = min(minimum_ess[seed_index], ess)
            if ess < resample_fraction * particles:
                cumulative = np.cumsum(weights)
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_offset = np.empty(particles)
                new_offset_rate = np.empty(particles)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_offset[particle] = offset[cursor] + rough_position * np.random.randn()
                    new_offset_rate[particle] = (
                        offset_rate[cursor] + rough_rate * np.random.randn()
                    )
                offset = new_offset
                offset_rate = new_offset_rate
                weights[:] = 1.0 / particles
                resampling_counts[seed_index] += 1
                resampled[seed_index, row] = 1
            predictions[seed_index, row] = np.sum(
                weights * (geometry_tvt_v[row] + offset)
            )
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
        particle_weight_sum,
        offset_edge_mass,
        effective_sample_size,
        resampled,
    )


# %% [markdown]
# ## 7. Seed aggregation, parity, and state-transition contracts


# %%
def aggregate_temperature(
    values: np.ndarray, log_likelihoods: np.ndarray, *, temperature: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / temperature)
    weights /= np.sum(weights)
    return (weights[:, None] * values).sum(axis=0), weights


def evidence_weighted_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights[:, None] * values).sum(axis=0)


def run_residual_offset_pf(
    prepared: Mapping[str, Any],
    geometry_tvt: np.ndarray,
    k16: Mapping[str, np.ndarray],
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
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    initial_offset_center = float(prepared["last_known_tvt"]) - float(geometry_tvt[0])
    output = _pf_residual_offset_allseeds(
        np.asarray(k16["dmd"], dtype=np.float64),
        np.asarray(k16["rho"], dtype=np.float64),
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
        output[0], output[1], temperature=temperature
    )
    geometry_delta = np.r_[0.0, np.diff(geometry_tvt)]
    particle_delta = np.r_[
        float(prediction[0] - prepared["last_known_tvt"]), np.diff(prediction)
    ]
    segment_id = np.asarray(k16["segment_id"], dtype=np.int64)
    ledger = pd.DataFrame(
        {
            "suffix_offset": np.arange(len(prediction), dtype=np.int64),
            "tvt_geop": geometry_tvt,
            "dmd": np.asarray(k16["dmd"], dtype=np.float64),
            "k16_segment_id": segment_id.astype(np.int16),
            "k16_segment_span": np.asarray(k16["segment_span"])[segment_id],
            "rho": np.asarray(k16["rho"], dtype=np.float64),
            "geometry_delta": geometry_delta,
            "filtered_offset_mean": evidence_weighted_rows(output[5], seed_weights),
            "filtered_offset_std": evidence_weighted_rows(output[6], seed_weights),
            "filtered_offset_rate_mean": evidence_weighted_rows(output[7], seed_weights),
            "filtered_offset_rate_std": evidence_weighted_rows(output[8], seed_weights),
            "particle_drift_minus_geometry_delta": particle_delta - geometry_delta,
            "typewell_support_fraction": evidence_weighted_rows(output[9], seed_weights),
            "particle_weight_sum": evidence_weighted_rows(output[10], seed_weights),
            "offset_edge_mass": evidence_weighted_rows(output[11], seed_weights),
            "effective_sample_size": evidence_weighted_rows(output[12], seed_weights),
            "resampled_seed_fraction": output[13].mean(axis=0),
        }
    )
    seed_evidence = pd.DataFrame(
        {
            "seed_index": np.arange(seeds, dtype=np.int64),
            "seed_value": seed_base + np.arange(seeds, dtype=np.int64),
            "total_log_evidence": output[1],
            "temperature_weight": seed_weights,
            "resampling_count": output[2],
            "minimum_effective_sample_size": output[3],
            "position_clip_count": output[4],
        }
    )
    diagnostics = {
        "initial_offset_center": initial_offset_center,
        "seed_log_likelihood_minimum": float(np.min(output[1])),
        "seed_log_likelihood_maximum": float(np.max(output[1])),
        "seed_weight_maximum": float(np.max(seed_weights)),
        "resampling_count": int(np.sum(output[2])),
        "minimum_effective_sample_size": float(np.min(output[3])),
        "position_clip_count": int(np.sum(output[4])),
        "minimum_typewell_support_fraction": float(ledger["typewell_support_fraction"].min()),
        "maximum_particle_weight_sum_error": float(
            np.max(np.abs(ledger["particle_weight_sum"] - 1.0))
        ),
        "maximum_offset_edge_mass": float(ledger["offset_edge_mass"].max()),
    }
    return prediction.astype(np.float32), ledger, seed_evidence, diagnostics


def zero_state_geometry_identity(dmd: np.ndarray, rho: np.ndarray) -> dict[str, Any]:
    zeros = np.zeros_like(np.asarray(dmd, dtype=np.float64))
    rate_center = 0.998 * np.asarray(rho, dtype=np.float64) * zeros
    offset_center = np.asarray(rho, dtype=np.float64) * zeros + rate_center * dmd
    maximum_abs_offset = float(np.max(np.abs(offset_center)))
    return {"pass": maximum_abs_offset == 0.0, "maximum_abs_offset_ft": maximum_abs_offset}


def rho_one_exp486_transition_parity() -> dict[str, Any]:
    previous_offset = np.asarray([-2.5, 0.0, 4.25], dtype=np.float64)
    previous_rate = np.asarray([-0.04, 0.0, 0.08], dtype=np.float64)
    dmd = np.asarray([1.0, 3.0, 2.0], dtype=np.float64)
    rate_draw = np.asarray([0.25, -1.0, 0.5], dtype=np.float64)
    position_draw = np.asarray([-0.5, 0.75, -0.25], dtype=np.float64)
    parent_rate = 0.998 * previous_rate + 0.002 * rate_draw
    parent_offset = previous_offset + parent_rate * dmd + 0.005 * position_draw
    candidate_rate = 0.998 * 1.0 * previous_rate + 0.002 * rate_draw
    candidate_offset = 1.0 * previous_offset + candidate_rate * dmd + 0.005 * position_draw
    rate_equal = np.array_equal(parent_rate.astype(np.float32), candidate_rate.astype(np.float32))
    offset_equal = np.array_equal(
        parent_offset.astype(np.float32), candidate_offset.astype(np.float32)
    )
    return {
        "pass": bool(rate_equal and offset_equal),
        "rate_float32_bitwise_equal": bool(rate_equal),
        "offset_float32_bitwise_equal": bool(offset_equal),
    }


def warm_up_pf_kernel() -> None:
    _pf_residual_offset_allseeds(
        np.ones(2),
        np.asarray([0.5, 0.5]),
        np.asarray([50.0, 51.0]),
        np.asarray([100.0, 100.2]),
        np.linspace(40.0, 70.0, 101),
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
# ## 8. Target-free single-variant generation and freeze


# %%
@dataclass
class FrozenWell:
    well_id: str
    prediction: pd.DataFrame
    residual_ledger: pd.DataFrame
    seed_evidence: pd.DataFrame
    segment_contract: pd.DataFrame
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
    particle_filter = dict(get_nested(config, "model.particle_filter") or {})
    transition = dict(get_nested(config, "model.transition") or {})
    initialization = dict(get_nested(config, "model.initialization") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(particle_filter["typewell_grid_step_ft"]),
    )
    geometry_tvt = align_geometry_to_prepared(well, geometry_rows, prepared)
    k16 = k16_segment_half_life(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        last_known_md=float(prepared["last_known_md"]),
        segment_count=int(get_nested(config, "model.k16_segment.count")),
    )
    seed_base = stable_seed("likpf", "train", well)
    prediction_values, residual, evidence, diagnostics = run_residual_offset_pf(
        prepared,
        geometry_tvt,
        k16,
        particles=int(particle_filter["particles"]),
        seeds=int(particle_filter["seeds"]),
        seed_base=seed_base,
        temperature=float(particle_filter["primary_seed_weighting_temperature"]),
        momentum=float(transition["momentum"]),
        rate_noise=float(transition["rate_noise"]),
        position_noise=float(transition["position_noise"]),
        rough_position=float(particle_filter["rough_position"]),
        rough_rate=float(particle_filter["rough_rate"]),
        resample_fraction=float(particle_filter["resample_threshold_fraction"]),
        initial_spread=float(initialization["offset_spread_ft"]),
        initial_rate_center=float(initialization["rate_center"]),
        initial_rate_spread=float(initialization["rate_spread"]),
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
            PREDICTION_COLUMN: prediction_values,
        }
    )
    residual.insert(0, "row_idx", eval_indices)
    residual.insert(0, "well_id", str(well))
    residual.insert(0, "id", identifiers)
    evidence.insert(0, "well_id", str(well))
    if list(residual.columns) != list(RESIDUAL_LEDGER_COLUMNS):
        raise ValueError("exp500 residual mechanism schema changed")
    if list(evidence.columns) != list(SEED_EVIDENCE_COLUMNS):
        raise ValueError("exp500 seed-evidence schema changed")
    if not np.isfinite(prediction[PREDICTION_COLUMN]).all():
        raise ValueError(f"{well}: exp500 prediction contains non-finite values")
    segment_contract = pd.DataFrame(
        {
            "well_id": str(well),
            "k16_segment_id": np.arange(16, dtype=np.int16),
            "rows": np.asarray(k16["segment_rows"], dtype=np.int64),
            "dmd_span": np.asarray(k16["segment_span"], dtype=np.float64),
            "rho_product": np.asarray(k16["segment_cumulative_rho"], dtype=np.float64),
        }
    )
    segment_contract["rho_product_abs_error_vs_half"] = (
        segment_contract["rho_product"] - 0.5
    ).abs()
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "eval_rows": len(prediction),
        "seed_base": int(seed_base),
        "variant_name_excluded_from_seed": True,
        "scientific_variants": 1,
        "candidate_pf_well_runs": 1,
        "seed_well_trajectories": int(particle_filter["seeds"]),
        "particle_starts": int(particle_filter["seeds"]) * int(particle_filter["particles"]),
        "zero_state_geometry_identity": zero_state_geometry_identity(
            np.asarray(k16["dmd"]), np.asarray(k16["rho"])
        )["pass"],
        "prediction_logical_sha256": dataframe_content_sha(
            prediction, ["id", "well_id", "row_idx", PREDICTION_COLUMN]
        ),
        "residual_ledger_logical_sha256": dataframe_content_sha(
            residual, RESIDUAL_LEDGER_COLUMNS
        ),
        "seed_evidence_logical_sha256": dataframe_content_sha(
            evidence, SEED_EVIDENCE_COLUMNS
        ),
        "segment_contract_logical_sha256": dataframe_content_sha(
            segment_contract, list(segment_contract.columns)
        ),
        **diagnostics,
        "wall_seconds": time.time() - started,
        "seconds_per_suffix_row": (time.time() - started) / len(prediction),
    }
    return FrozenWell(
        well_id=well,
        prediction=prediction,
        residual_ledger=residual,
        seed_evidence=evidence,
        segment_contract=segment_contract,
        audit=audit,
    )


def freeze_target_free_outputs(
    frozen_wells: Sequence[FrozenWell],
    output: Path,
    *,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    artifact_tag: str = "stage0",
    expected_rows: int | None = None,
    expected_wells: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions = pd.concat(
        [item.prediction for item in frozen_wells], ignore_index=True
    ).sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    residual = pd.concat(
        [item.residual_ledger for item in frozen_wells], ignore_index=True
    ).sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    evidence = pd.concat(
        [item.seed_evidence for item in frozen_wells], ignore_index=True
    ).sort_values(["well_id", "seed_index"], kind="mergesort").reset_index(drop=True)
    segments = pd.concat(
        [item.segment_contract for item in frozen_wells], ignore_index=True
    ).sort_values(["well_id", "k16_segment_id"], kind="mergesort").reset_index(drop=True)
    audit = pd.DataFrame([item.audit for item in frozen_wells]).sort_values(
        "well_id", kind="mergesort"
    ).reset_index(drop=True)
    if expected_rows is None:
        expected_rows = int(get_nested(config, "data.stage_0_expected_suffix_rows", 224400))
    if expected_wells is None:
        expected_wells = 44
    expected_evidence_rows = expected_wells * int(
        get_nested(config, "model.particle_filter.seeds")
    )
    expected_segment_rows = expected_wells * int(
        get_nested(config, "model.k16_segment.count")
    )
    if (
        predictions["id"].duplicated().any()
        or residual["id"].duplicated().any()
        or len(predictions) != expected_rows
        or len(residual) != expected_rows
        or predictions["well_id"].nunique() != expected_wells
        or len(evidence) != expected_evidence_rows
        or len(segments) != expected_segment_rows
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp500 target-free output coverage mismatch")
    prediction_artifact = write_deterministic_gzip_csv(
        predictions, output / f"{OUTPUT_PREFIX}_{artifact_tag}_predictions.csv.gz"
    )
    residual_artifact = write_deterministic_gzip_csv(
        residual, output / f"{OUTPUT_PREFIX}_{artifact_tag}_residual_ledger.csv.gz"
    )
    evidence_artifact = write_deterministic_gzip_csv(
        evidence, output / f"{OUTPUT_PREFIX}_{artifact_tag}_seed_evidence.csv.gz"
    )
    segment_artifact = write_deterministic_gzip_csv(
        segments, output / f"{OUTPUT_PREFIX}_{artifact_tag}_k16_rho_contract.csv.gz"
    )
    audit_path = output / f"{OUTPUT_PREFIX}_{artifact_tag}_well_audit.csv"
    audit.to_csv(audit_path, index=False)
    for item in frozen_wells:
        ledger.freeze(ACTIVE_VARIANT, item.well_id)
    if not ledger.all_frozen:
        raise RuntimeError(f"exp500 did not freeze every {artifact_tag} candidate")
    frozen = {
        "stage": f"{artifact_tag}_target_free_freeze",
        "frozen_before_truth_attachment": True,
        "rows": len(predictions),
        "wells": int(predictions["well_id"].nunique()),
        "scientific_variants": 1,
        "prediction_logical_sha256": dataframe_content_sha(
            predictions, ["id", "well_id", "row_idx", PREDICTION_COLUMN]
        ),
        "residual_ledger_logical_sha256": dataframe_content_sha(
            residual, RESIDUAL_LEDGER_COLUMNS
        ),
        "seed_evidence_logical_sha256": dataframe_content_sha(
            evidence, SEED_EVIDENCE_COLUMNS
        ),
        "k16_rho_contract_logical_sha256": dataframe_content_sha(
            segments, list(segments.columns)
        ),
        "prediction_artifact": prediction_artifact,
        "residual_ledger_artifact": residual_artifact,
        "seed_evidence_artifact": evidence_artifact,
        "k16_rho_contract_artifact": segment_artifact,
        "well_audit": {"path": str(audit_path), "raw_sha256": sha256_path(audit_path)},
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    return predictions, residual, evidence, segments, audit, frozen


# %% [markdown]
# ## 9. Truth-late fixed44 readout and fail-closed gates
#
# Everything below requires the complete target-free freeze. Fixed32 roles,
# sentinel causes, truth, saved exp404/exp486 predictions, folds, and fixed
# episode boundaries are evaluation-only fields.


# %%
def _require_frozen(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("exp500 late readout requires a complete freeze")
    for key in (
        "prediction_logical_sha256",
        "residual_ledger_logical_sha256",
        "seed_evidence_logical_sha256",
        "k16_rho_contract_logical_sha256",
    ):
        if len(str(frozen.get(key) or "")) != 64:
            raise RuntimeError(f"exp500 frozen output is missing {key}")


def load_suffix_truth(well: str, raw_dir: Path, ledger: LeakageLedger) -> pd.DataFrame:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv", usecols=["TVT_input", "TVT"]
    )
    eval_indices = np.flatnonzero(horizontal["TVT_input"].isna()).astype(np.int64)
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": well,
            "row_idx": eval_indices,
            "true_tvt": pd.to_numeric(
                horizontal.loc[eval_indices, "TVT"], errors="raise"
            ).to_numpy(np.float64),
        }
    )
    if not np.isfinite(frame["true_tvt"]).all():
        raise ValueError(f"{well}: suffix truth is not finite")
    ledger.record_truth(len(frame))
    return frame


def saved_prediction_path(config: Mapping[str, Any], dotted_key: str) -> Path:
    spec = dict(get_nested(config, dotted_key) or {})
    path = resolve_existing(
        str(spec["filename"]), spec.get("candidates", []), spec.get("patterns", [])
    )
    expected_raw = str(spec.get("expected_raw_sha256") or "")
    expected_decompressed = str(spec.get("expected_decompressed_sha256") or "")
    if expected_raw and sha256_path(path) != expected_raw:
        raise ValueError(f"exp500 {dotted_key} raw SHA mismatch")
    if expected_decompressed and sha256_decompressed_csv(path) != expected_decompressed:
        raise ValueError(f"exp500 {dotted_key} decompressed SHA mismatch")
    return path


def load_saved_prediction_after_freeze(
    config: Mapping[str, Any],
    dotted_key: str,
    identifiers: set[str],
    ledger: LeakageLedger,
    output_column: str,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("saved prediction requires complete exp500 freeze")
    spec = dict(get_nested(config, dotted_key) or {})
    source_column = str(spec["prediction_column"])
    pieces: list[pd.DataFrame] = []
    path = saved_prediction_path(config, dotted_key)
    for chunk in pd.read_csv(
        path,
        usecols=["id", source_column],
        dtype={"id": str},
        compression="gzip" if is_gzip_payload(path) else None,
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["id"].isin(identifiers)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True)
    ledger.record_control(len(frame))
    if len(frame) != len(identifiers) or frame["id"].nunique() != len(identifiers):
        raise ValueError(f"exp500 {dotted_key} coverage mismatch")
    return frame.rename(columns={source_column: output_column})


def load_episode_boundaries_after_freeze(
    config: Mapping[str, Any],
    dotted_key: str,
    wells: set[str],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("episode boundaries require complete exp500 freeze")
    path = asset_path(config, dotted_key)
    frame = pd.read_csv(
        path,
        usecols=["episode_id", "well", "start_row_idx", "end_row_idx_exclusive"],
        dtype={"well": str},
    )
    frame = frame.loc[frame["well"].isin(wells)].copy()
    ledger.record_role_fold_episode(len(frame))
    if frame.empty or set(frame["well"]) != wells:
        raise ValueError(f"exp500 {dotted_key} does not cover the fixed wells")
    return frame.sort_values(["well", "start_row_idx"], kind="mergesort").reset_index(
        drop=True
    )


def attach_truth_late_readout(
    predictions: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_frozen(frozen)
    if dataframe_content_sha(
        predictions, ["id", "well_id", "row_idx", PREDICTION_COLUMN]
    ) != str(frozen["prediction_logical_sha256"]):
        raise ValueError("exp500 predictions changed after freeze")
    roles = load_fixed44_readout_after_freeze(config, ledger).rename(
        columns={"well": "well_id"}
    )
    wells = predictions["well_id"].drop_duplicates().tolist()
    truth = pd.concat(
        [load_suffix_truth(str(well), raw_dir, ledger) for well in wells],
        ignore_index=True,
    )
    identifiers = set(predictions["id"].astype(str))
    exp404 = load_saved_prediction_after_freeze(
        config,
        "data.saved_exp404_control",
        identifiers,
        ledger,
        PRIMARY_CONTROL,
    )
    exp486 = load_saved_prediction_after_freeze(
        config,
        "data.saved_exp486_parent",
        identifiers,
        ledger,
        PARENT_RESIDUAL,
    )
    frame = predictions.merge(
        truth, on=["id", "well_id", "row_idx"], how="inner", validate="one_to_one"
    ).merge(exp404, on="id", how="inner", validate="one_to_one").merge(
        exp486, on="id", how="inner", validate="one_to_one"
    ).merge(roles, on="well_id", how="left", validate="many_to_one")
    if len(frame) != len(predictions) or frame["role"].isna().any():
        raise ValueError("exp500 truth-late readout coverage mismatch")
    frame["candidate_error"] = frame[PREDICTION_COLUMN] - frame["true_tvt"]
    frame["exp404_error"] = frame[PRIMARY_CONTROL] - frame["true_tvt"]
    frame["exp486_error"] = frame[PARENT_RESIDUAL] - frame["true_tvt"]
    if not np.isfinite(
        frame[["candidate_error", "exp404_error", "exp486_error"]].to_numpy(np.float64)
    ).all():
        raise ValueError("exp500 truth-late errors are non-finite")
    by_well_rows: list[dict[str, Any]] = []
    for well, part in frame.groupby("well_id", sort=True):
        row = {
            "well_id": str(well),
            "role": str(part["role"].iloc[0]),
            "fold": int(part["fold"].iloc[0]),
            "rows": len(part),
            "candidate_rmse": rmse(part["candidate_error"]),
            "exp404_rmse": rmse(part["exp404_error"]),
            "exp486_rmse": rmse(part["exp486_error"]),
        }
        row["candidate_minus_exp404_rmse"] = row["candidate_rmse"] - row["exp404_rmse"]
        row["candidate_minus_exp486_rmse"] = row["candidate_rmse"] - row["exp486_rmse"]
        by_well_rows.append(row)
    return frame.sort_values(["well_id", "row_idx"], kind="mergesort"), pd.DataFrame(
        by_well_rows
    )


def rmse(error: np.ndarray | pd.Series) -> float:
    values = np.asarray(error, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("RMSE requires non-empty finite errors")
    return float(np.sqrt(np.mean(values * values)))


def build_episode_metrics(
    episodes: pd.DataFrame,
    readout: pd.DataFrame,
    *,
    baseline_error_column: str,
    recovery_horizons: Sequence[int] = (256, 512),
    recovery_threshold_ft: float = 5.0,
) -> pd.DataFrame:
    grouped = {
        str(well): part.sort_values("row_idx", kind="mergesort")
        for well, part in readout.groupby("well_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        part = grouped[str(episode.well)]
        start = int(episode.start_row_idx)
        end = int(episode.end_row_idx_exclusive)
        window = part.loc[part["row_idx"].ge(start) & part["row_idx"].lt(end)]
        if window.empty:
            raise ValueError(f"{episode.episode_id}: fixed episode window is empty")
        row: dict[str, Any] = {
            "episode_id": str(episode.episode_id),
            "well": str(episode.well),
            "rows": len(window),
            "baseline_sse": float(np.square(window[baseline_error_column]).sum()),
            "candidate_sse": float(np.square(window["candidate_error"]).sum()),
        }
        for horizon in recovery_horizons:
            recovery = part.loc[part["row_idx"].ge(end) & part["row_idx"].lt(end + horizon)]
            row[f"baseline_recovered_{horizon}"] = bool(
                (recovery[baseline_error_column].abs() <= recovery_threshold_ft).any()
            )
            row[f"candidate_recovered_{horizon}"] = bool(
                (recovery["candidate_error"].abs() <= recovery_threshold_ft).any()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def contiguous_episode_count(
    error: np.ndarray, *, threshold_ft: float = 10.0, minimum_rows: int = 128
) -> int:
    mask = np.abs(np.asarray(error, dtype=np.float64)) >= float(threshold_ft)
    count = 0
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        count += int(end - start >= int(minimum_rows))
        start = end
    return count


def shard_index(well: str, count: int = 4) -> int:
    digest = hashlib.sha256(f"exp500::full_pf_shard::{well}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % int(count)


def full_shard_suffix_rows(raw_dir: Path) -> dict[int, int]:
    rows = {index: 0 for index in range(4)}
    files = sorted(raw_dir.glob("*__horizontal_well.csv"))
    if len(files) != 773:
        raise ValueError("exp500 full runtime projection requires 773 raw wells")
    for path in files:
        well = path.name.removesuffix("__horizontal_well.csv")
        tvt_input = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"]
        rows[shard_index(well)] += int(tvt_input.isna().sum())
    if sum(rows.values()) != 3_783_989:
        raise ValueError("exp500 full runtime projection row count changed")
    return rows


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    scope_report: Mapping[str, Any],
    predictions: pd.DataFrame,
    residual: pd.DataFrame,
    segments: pd.DataFrame,
    audit: pd.DataFrame,
    readout: pd.DataFrame,
    by_well: pd.DataFrame,
    exp408_parent_metrics: pd.DataFrame,
    exp408_control_metrics: pd.DataFrame,
    exp410_metrics: pd.DataFrame,
    ledger: LeakageLedger,
    raw_dir: Path,
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical_stage_0") or {})
    mechanism_config = dict(get_nested(config, "guards.mechanism_stage_0") or {})
    before = dict(ledger.report()["before_freeze"])
    maximum_half_life_error = float(segments["rho_product_abs_error_vs_half"].max())
    seconds_per_row_p95 = float(np.quantile(audit["seconds_per_suffix_row"], 0.95))
    shard_rows = full_shard_suffix_rows(raw_dir)
    shard_projection = {
        str(index): seconds_per_row_p95 * rows for index, rows in shard_rows.items()
    }
    maximum_shard_projection = max(shard_projection.values())
    technical = {
        "fixed32_and_sentinel_sha_and_union": bool(
            scope_report["fixed32_wells"] == 32
            and scope_report["sentinel_wells"] == 12
            and scope_report["overlap_wells"] == 0
            and scope_report["union_wells"] == 44
        ),
        "geometry_allowlist_and_row_coverage": bool(
            len(predictions) == 224400
            and predictions["well_id"].nunique() == 44
            and list(get_nested(config, "data.exp226_oof_geometry.prediction_time_usecols"))
            == list(GEOMETRY_ALLOWLIST)
        ),
        "k16_segment_coverage": bool(
            len(segments) == 44 * 16
            and (segments.groupby("well_id").size() == 16).all()
        ),
        "positive_dmd_and_segment_span": bool(
            (residual["dmd"] > 0.0).all() and (segments["dmd_span"] > 0.0).all()
        ),
        "rho_finite_and_bounded": bool(
            np.isfinite(residual["rho"]).all()
            and (residual["rho"] > 0.0).all()
            and (residual["rho"] <= 1.0).all()
        ),
        "segment_cumulative_half_life": bool(
            maximum_half_life_error
            <= float(technical_config["segment_cumulative_rho_atol"])
        ),
        "zero_state_geometry_identity": bool(audit["zero_state_geometry_identity"].all()),
        "rho_one_exp486_float32_parity": bool(rho_one_exp486_transition_parity()["pass"]),
        "finite_prediction_weight_ess_and_ledger": bool(
            np.isfinite(predictions[PREDICTION_COLUMN]).all()
            and np.isfinite(residual.select_dtypes(include=[np.number]).to_numpy()).all()
            and (residual["effective_sample_size"] > 0.0).all()
            and float(np.max(np.abs(residual["particle_weight_sum"] - 1.0))) <= 1.0e-12
        ),
        "stable_seed_and_execution_count": bool(
            audit["seed_base"].nunique() == 44
            and audit["variant_name_excluded_from_seed"].all()
            and int(audit["candidate_pf_well_runs"].sum()) == 44
            and int(audit["seed_well_trajectories"].sum()) == 5632
            and int(audit["particle_starts"].sum()) == 2816000
        ),
        "truth_control_role_fold_episode_reads_before_freeze_zero": bool(
            int(before["truth_rows"]) == 0
            and int(before["control_rows"]) == 0
            and int(before["role_fold_episode_rows"]) == 0
            and int(before["forbidden_geometry_columns"]) == 0
        ),
        "runtime_projection": bool(
            maximum_shard_projection
            <= float(technical_config["maximum_projected_full_shard_seconds"])
        ),
        "peak_rss": bool(peak_rss_gb() <= float(technical_config["maximum_peak_rss_gib"])),
    }

    persistent_rows = readout.loc[readout["role"].eq("persistent")]
    control_rows = readout.loc[readout["role"].eq("control")]
    sentinel_rows = readout.loc[readout["role"].eq("pf_sentinel")]
    persistent_wells = by_well.loc[by_well["role"].eq("persistent")]
    controls = by_well.loc[by_well["role"].eq("control")]
    sentinels = by_well.loc[by_well["role"].eq("pf_sentinel")]

    exp408_baseline_sse = float(exp408_parent_metrics["baseline_sse"].sum())
    exp408_candidate_sse = float(exp408_parent_metrics["candidate_sse"].sum())
    exp408_reduction = 1.0 - exp408_candidate_sse / exp408_baseline_sse
    exp410_baseline_sse = float(exp410_metrics["baseline_sse"].sum())
    exp410_candidate_sse = float(exp410_metrics["candidate_sse"].sum())
    exp410_reduction = 1.0 - exp410_candidate_sse / exp410_baseline_sse
    persistent_improved_wells = int(
        (persistent_wells["candidate_minus_exp486_rmse"] < 0.0).sum()
    )
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(5):
        part = persistent_rows.loc[persistent_rows["fold"].eq(fold)]
        candidate_rmse = rmse(part["candidate_error"])
        baseline_rmse = rmse(part["exp486_error"])
        fold_metrics.append(
            {
                "fold": fold,
                "candidate_rmse": candidate_rmse,
                "exp486_rmse": baseline_rmse,
                "improved": candidate_rmse < baseline_rmse,
            }
        )
    persistent_improved_folds = sum(row["improved"] for row in fold_metrics)
    control_regression = rmse(control_rows["candidate_error"]) - rmse(
        control_rows["exp404_error"]
    )
    control_p95 = float(np.quantile(controls["candidate_minus_exp404_rmse"], 0.95))
    exp408_baseline_count = sum(
        contiguous_episode_count(part["exp404_error"].to_numpy(np.float64))
        for _, part in persistent_rows.groupby("well_id", sort=True)
    )
    exp408_candidate_count = sum(
        contiguous_episode_count(part["candidate_error"].to_numpy(np.float64))
        for _, part in persistent_rows.groupby("well_id", sort=True)
    )
    recovery: dict[str, Any] = {}
    recovery_pass = True
    for horizon in (256, 512):
        baseline_rate = float(
            exp408_control_metrics[f"baseline_recovered_{horizon}"].mean()
        )
        candidate_rate = float(
            exp408_control_metrics[f"candidate_recovered_{horizon}"].mean()
        )
        delta = candidate_rate - baseline_rate
        recovery[str(horizon)] = {
            "baseline_rate": baseline_rate,
            "candidate_rate": candidate_rate,
            "delta": delta,
        }
        recovery_pass &= delta >= float(
            mechanism_config[f"recovery_rate_{horizon}_delta_vs_exp404_min"]
        )
    collapse_by_well = residual.groupby("well_id", sort=True).agg(
        finite_ess=("effective_sample_size", lambda value: bool(np.isfinite(value).all())),
        maximum_edge_mass=("offset_edge_mass", "max"),
        all_rows_edge_collapsed=(
            "offset_edge_mass",
            lambda value: bool(np.all(np.asarray(value) >= 1.0 - 1.0e-12)),
        ),
        minimum_support=("typewell_support_fraction", "min"),
    )
    no_collapse = bool(
        collapse_by_well["finite_ess"].all()
        and np.isfinite(collapse_by_well[["maximum_edge_mass", "minimum_support"]]).all().all()
        and not collapse_by_well["all_rows_edge_collapsed"].any()
    )
    mechanism = {
        "persistent_episode_sse_reduction_vs_exp486": bool(
            exp408_reduction
            >= float(mechanism_config["persistent_episode_sse_reduction_vs_exp486_residual_min"])
        ),
        "persistent_improved_wells_vs_exp486": bool(
            persistent_improved_wells
            >= int(mechanism_config["persistent_improved_wells_vs_exp486_residual_min"])
        ),
        "persistent_improved_folds_vs_exp486": bool(
            persistent_improved_folds
            >= int(mechanism_config["persistent_improved_folds_vs_exp486_residual_min"])
        ),
        "matched_control_pooled_vs_exp404": bool(
            control_regression
            <= float(mechanism_config["matched_control_pooled_regression_vs_exp404_max_ft"])
        ),
        "matched_control_by_well_p95_vs_exp404": bool(
            control_p95
            <= float(mechanism_config["matched_control_by_well_p95_vs_exp404_max_ft"])
        ),
        "exp408_episode_count_vs_exp404": bool(
            exp408_candidate_count - exp408_baseline_count
            <= int(mechanism_config["exp408_episode_count_delta_vs_exp404_max"])
        ),
        "exp408_recovery_256_512_vs_exp404": bool(recovery_pass),
        "exp410_pf_episode_sse_reduction_vs_exp404": bool(
            exp410_reduction
            >= float(mechanism_config["exp410_pf_episode_sse_reduction_vs_exp404_min"])
        ),
        "pf_sentinel_worst_well_vs_exp404": bool(
            float(sentinels["candidate_minus_exp404_rmse"].max())
            <= float(mechanism_config["pf_sentinel_worst_well_regression_vs_exp404_max_ft"])
        ),
        "no_nonfinite_or_all_row_particle_collapse": no_collapse,
    }
    return {
        "stage": "stage0_fixed44_mechanism_preflight_not_cv",
        "technical_checks": technical,
        "mechanism_checks": mechanism,
        "technical_all_pass": bool(all(technical.values())),
        "mechanism_all_pass": bool(all(mechanism.values())),
        "all_pass": bool(all(technical.values()) and all(mechanism.values())),
        "diagnostics": {
            "rows": len(readout),
            "wells": 44,
            "persistent_wells": len(persistent_wells),
            "matched_control_wells": len(controls),
            "pf_sentinel_wells": len(sentinels),
            "maximum_segment_half_life_abs_error": maximum_half_life_error,
            "seconds_per_suffix_row_p95": seconds_per_row_p95,
            "full_shard_suffix_rows": shard_rows,
            "full_shard_projected_seconds": shard_projection,
            "maximum_full_shard_projected_seconds": maximum_shard_projection,
            "peak_rss_gib": peak_rss_gb(),
            "persistent_episode_sse_reduction_fraction": exp408_reduction,
            "persistent_improved_wells": persistent_improved_wells,
            "persistent_improved_folds": persistent_improved_folds,
            "persistent_fold_metrics": fold_metrics,
            "matched_control_regression_ft": control_regression,
            "matched_control_by_well_p95_ft": control_p95,
            "exp408_episode_count_delta": exp408_candidate_count - exp408_baseline_count,
            "exp408_recovery": recovery,
            "exp410_pf_episode_sse_reduction_fraction": exp410_reduction,
            "pf_sentinel_worst_well_regression_ft": float(
                sentinels["candidate_minus_exp404_rmse"].max()
            ),
            "persistent_pooled_candidate_rmse": rmse(persistent_rows["candidate_error"]),
            "persistent_pooled_exp486_rmse": rmse(persistent_rows["exp486_error"]),
            "sentinel_pooled_candidate_rmse": rmse(sentinel_rows["candidate_error"]),
            "stage_0_is_cv": False,
        },
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 10. Generated artifacts and guarded Stage 0 orchestration
#
# The code surface is implemented, but execution flags remain false. A later
# approval must separately adopt the canonical Notebook, package it, and run
# this Stage 0 path on Kaggle private CPU.


# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXP500_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp500 Stage 0 must run first on Kaggle private CPU")


def build_input_manifest(
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
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    raw_frame = pd.DataFrame(raw_rows).sort_values("well_id", kind="mergesort")
    return {
        "split": "train",
        "scope": dict(scope_report),
        "raw_dir": str(raw_dir),
        "wells": len(wells),
        "raw_well_content_sha256": dataframe_content_sha(
            raw_frame, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
        ),
        "exp226_geometry": {
            "path": str(geometry_path),
            "raw_sha256": sha256_path(geometry_path),
            "decompressed_sha256": sha256_decompressed_csv(geometry_path),
            "logical_sha256_fixed44": dataframe_content_sha(
                geometry.sort_values(["well_id", "row_idx"], kind="mergesort"),
                GEOMETRY_ALLOWLIST,
            ),
            "columns_read": list(GEOMETRY_ALLOWLIST),
            "rows": len(geometry),
            "wells": int(geometry["well_id"].nunique()),
        },
        "saved_controls": {
            "exp404_rerun": False,
            "exp486_rerun": False,
            "parsed_before_freeze": False,
        },
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    require_kaggle_runtime()
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exp500 Stage 0")
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    wells, scope_report = load_fixed44_identity(config)
    ledger = LeakageLedger(expected_variant_wells=len(wells))
    geometry_path = geometry_input_path(config)
    geometry = load_fold_safe_geometry(
        geometry_path, config, wells=set(wells), ledger=ledger
    )
    if len(geometry) != int(get_nested(config, "data.stage_0_expected_suffix_rows")):
        raise ValueError("exp500 fixed44 geometry coverage changed")
    scientific_contract = build_scientific_contract(config)
    scientific_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_scientific_contract.json", scientific_contract
    )
    input_manifest = build_input_manifest(
        raw_dir, wells, scope_report, geometry_path, geometry
    )
    input_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_input_manifest.json", input_manifest
    )
    warm_up_pf_kernel()
    geometry_groups = {
        str(well): group.copy() for well, group in geometry.groupby("well_id", sort=False)
    }
    workers = int(get_nested(config, "runtime.num_workers", 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frozen_wells = list(
            executor.map(
                lambda well: decode_target_free_well(
                    str(well), raw_dir, geometry_groups[str(well)], config
                ),
                wells,
            )
        )
    predictions, residual, evidence, segments, audit, frozen = freeze_target_free_outputs(
        frozen_wells, output, config=config, ledger=ledger
    )
    prefreeze_seconds = time.time() - started
    runtime_ledger = {
        "stage": "stage0_fixed44_target_free_freeze_not_cv",
        "candidate_wells": 44,
        "candidate_rows": len(predictions),
        "scientific_variants": 1,
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
        "summed_candidate_well_seconds": float(audit["wall_seconds"].sum()),
        "prefreeze_wall_seconds": prefreeze_seconds,
        "peak_rss_gib": peak_rss_gb(),
        "versions": runtime_versions(),
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    runtime_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_runtime_ledger.json", runtime_ledger
    )
    frozen.update(
        {
            "scientific_contract_sha256": scientific_contract[
                "scientific_contract_sha256"
            ],
            "scientific_contract_file_sha256": scientific_artifact["raw_sha256"],
            "input_manifest_sha256": input_artifact["raw_sha256"],
            "runtime_ledger_sha256": runtime_artifact["raw_sha256"],
        }
    )
    freeze_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_freeze_manifest.json", frozen
    )

    readout, by_well = attach_truth_late_readout(
        predictions,
        frozen,
        config=config,
        raw_dir=raw_dir,
        ledger=ledger,
    )
    persistent_wells = set(
        by_well.loc[by_well["role"].eq("persistent"), "well_id"].astype(str)
    )
    sentinel_wells = set(
        by_well.loc[by_well["role"].eq("pf_sentinel"), "well_id"].astype(str)
    )
    exp408_episodes = load_episode_boundaries_after_freeze(
        config, "data.exp408_persistent_episodes", persistent_wells, ledger
    )
    exp410_episodes = load_episode_boundaries_after_freeze(
        config, "data.exp410_persistent_episodes", sentinel_wells, ledger
    )
    exp408_parent_metrics = build_episode_metrics(
        exp408_episodes, readout, baseline_error_column="exp486_error"
    )
    exp408_control_metrics = build_episode_metrics(
        exp408_episodes, readout, baseline_error_column="exp404_error"
    )
    exp410_metrics = build_episode_metrics(
        exp410_episodes, readout, baseline_error_column="exp404_error"
    )
    gates = evaluate_stage0_gates(
        config=config,
        scope_report=scope_report,
        predictions=predictions,
        residual=residual,
        segments=segments,
        audit=audit,
        readout=readout,
        by_well=by_well,
        exp408_parent_metrics=exp408_parent_metrics,
        exp408_control_metrics=exp408_control_metrics,
        exp410_metrics=exp410_metrics,
        ledger=ledger,
        raw_dir=raw_dir,
    )
    truth_artifact = write_deterministic_gzip_csv(
        readout, output / f"{OUTPUT_PREFIX}_stage0_truth_late_rows.csv.gz"
    )
    by_well_path = output / f"{OUTPUT_PREFIX}_stage0_by_well.csv"
    exp408_parent_path = output / f"{OUTPUT_PREFIX}_stage0_exp408_vs_exp486_episode_metrics.csv"
    exp408_control_path = output / f"{OUTPUT_PREFIX}_stage0_exp408_vs_exp404_episode_metrics.csv"
    exp410_path = output / f"{OUTPUT_PREFIX}_stage0_exp410_episode_metrics.csv"
    by_well.to_csv(by_well_path, index=False)
    exp408_parent_metrics.to_csv(exp408_parent_path, index=False)
    exp408_control_metrics.to_csv(exp408_control_path, index=False)
    exp410_metrics.to_csv(exp410_path, index=False)
    gate_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_gate_report.json", gates
    )
    status = (
        "stage0_all_pass_pending_separate_stage1_implementation_approval"
        if gates["all_pass"]
        else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage0_fixed44_mechanism_preflight_not_cv",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "counts": {
            "scientific_variants": 1,
            "candidate_pf_well_runs": 44,
            "seed_well_trajectories": 5632,
            "particle_starts": 2816000,
            "control_pf_reruns": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "gpu_runs": 0,
        },
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "frozen_outputs": frozen,
        "gates": gates,
        "runtime": runtime_ledger,
        "artifacts": {
            "scientific_contract": scientific_artifact,
            "input_manifest": input_artifact,
            "runtime_ledger": runtime_artifact,
            "freeze_manifest": freeze_artifact,
            "truth_late_rows": truth_artifact,
            "by_well": {"path": str(by_well_path), "raw_sha256": sha256_path(by_well_path)},
            "exp408_vs_exp486_episode_metrics": {
                "path": str(exp408_parent_path),
                "raw_sha256": sha256_path(exp408_parent_path),
            },
            "exp408_vs_exp404_episode_metrics": {
                "path": str(exp408_control_path),
                "raw_sha256": sha256_path(exp408_control_path),
            },
            "exp410_episode_metrics": {
                "path": str(exp410_path),
                "raw_sha256": sha256_path(exp410_path),
            },
            "gate_report": gate_artifact,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": (
            "request_separate_stage1_implementation_and_run_approval"
            if gates["all_pass"]
            else "terminal_close_without_same_fixed44_rescue"
        ),
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_summary.json", summary
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. Stage 1 target-free shards, strict merge, and full OOF gates
#
# The user-approved override runs the unchanged candidate over all 773 train
# wells. Four CPU shards may parse only deployable raw inputs and the exp226
# geometry allowlist. The merge revalidates every artifact and freezes their
# exact union before truth, saved predictions, folds, roles, or episodes are
# parsed. Stage 0 remains failed evidence regardless of this Stage 1 result.


# %%
def validate_raw_well_identity(
    config: Mapping[str, Any], raw_dir: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizontal in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal.name.removesuffix("__horizontal_well.csv")
        typewell = raw_dir / f"{well}__typewell.csv"
        if not typewell.exists():
            raise FileNotFoundError(typewell)
        rows.append(
            {
                "well_id": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    content_sha = typed_dataframe_content_sha(
        frame, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    )
    if (
        len(frame) != int(get_nested(config, "validation.expected_wells"))
        or content_sha
        != str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    ):
        raise ValueError("exp500 raw train well identity changed")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": content_sha,
        "well_ids": frame["well_id"].astype(str).tolist(),
        "rows": rows,
    }


def build_stage1_well_manifest(
    config: Mapping[str, Any], raw_dir: Path, wells: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in sorted(str(value) for value in wells):
        tvt_input = pd.read_csv(
            raw_dir / f"{well}__horizontal_well.csv", usecols=["TVT_input"]
        )["TVT_input"]
        suffix_rows = int(tvt_input.isna().sum())
        rows.append(
            {
                "well_id": well,
                "rows": len(tvt_input),
                "prefix_rows": len(tvt_input) - suffix_rows,
                "suffix_rows": suffix_rows,
                "shard_index": shard_index(well, SHARD_COUNT),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    observed_wells = [
        int(manifest["shard_index"].eq(index).sum()) for index in range(SHARD_COUNT)
    ]
    observed_rows = [
        int(manifest.loc[manifest["shard_index"].eq(index), "suffix_rows"].sum())
        for index in range(SHARD_COUNT)
    ]
    if (
        len(manifest) != int(get_nested(config, "validation.expected_wells"))
        or int(manifest["suffix_rows"].sum())
        != int(get_nested(config, "validation.expected_rows"))
        or observed_wells
        != [int(value) for value in get_nested(config, "execution.stage_1.expected_shard_wells")]
        or observed_rows
        != [
            int(value)
            for value in get_nested(config, "execution.stage_1.expected_shard_suffix_rows")
        ]
    ):
        raise ValueError("exp500 deterministic Stage 1 shard census changed")
    return manifest


def selected_stage1(config: Mapping[str, Any]) -> str | None:
    shard = bool(get_nested(config, "execution.run_stage_1_shard", False))
    merge = bool(get_nested(config, "execution.run_stage_1_merge", False))
    if shard and merge:
        raise ValueError("exp500 permits exactly one Stage 1 execution stage")
    selected = get_nested(config, "execution.selected_stage_1_shard_index")
    if shard:
        if selected is None or int(selected) not in range(SHARD_COUNT):
            raise ValueError("selected_stage_1_shard_index must be in [0, 3]")
        return "stage1_shard"
    if merge:
        if selected not in (None, "", "null"):
            raise ValueError("Stage 1 merge must not select a shard index")
        roots = list(get_nested(config, "execution.stage_1_merge_shard_dirs") or [])
        if len(roots) != SHARD_COUNT:
            raise ValueError("Stage 1 merge requires four ordered shard roots")
        return "stage1_merge"
    if selected not in (None, "", "null"):
        raise ValueError("a shard index is set while Stage 1 execution is disarmed")
    return None


def _artifact_file(root: Path, filename: str) -> Path:
    filenames = [filename]
    if filename.endswith(".csv.gz"):
        filenames.append(filename.removesuffix(".gz"))
    matches: list[Path] = []
    for candidate_name in filenames:
        direct = root / candidate_name
        if direct.exists():
            matches.append(direct)
        matches.extend(sorted(root.glob(f"**/{candidate_name}")))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one of {filenames} under {root}; found={unique}"
        )
    return unique[0]


def _verify_artifact_report(path: Path, report: Mapping[str, Any], label: str) -> None:
    raw_matches = sha256_path(path) == str(report.get("raw_sha256"))
    expected_decompressed = report.get("decompressed_sha256")
    if expected_decompressed is None:
        if not raw_matches:
            raise ValueError(f"{label} raw SHA mismatch")
        return
    if sha256_decompressed_csv(path) != str(expected_decompressed):
        raise ValueError(f"{label} decompressed SHA mismatch")
    if is_gzip_payload(path) and not raw_matches:
        raise ValueError(f"{label} raw gzip SHA mismatch")


def run_stage1_shard(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(
        config, require_run_approval=True
    )
    require_kaggle_runtime()
    if selected_stage1(config) != "stage1_shard":
        raise RuntimeError("exp500 Stage 1 shard is not selected")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exp500 Stage 1")
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_identity = validate_raw_well_identity(config, raw_dir)
    manifest = build_stage1_well_manifest(config, raw_dir, raw_identity["well_ids"])
    index = int(get_nested(config, "execution.selected_stage_1_shard_index"))
    selected = manifest.loc[manifest["shard_index"].eq(index)].copy()
    wells = selected["well_id"].astype(str).tolist()
    expected_rows = int(selected["suffix_rows"].sum())
    tag = f"stage1_shard{index}"
    manifest_path = output / f"{OUTPUT_PREFIX}_{tag}_well_manifest.csv"
    selected.to_csv(manifest_path, index=False)
    ledger = LeakageLedger(expected_variant_wells=len(wells))
    geometry_path = geometry_input_path(config)
    geometry = load_fold_safe_geometry(
        geometry_path, config, wells=set(wells), ledger=ledger
    )
    if (
        len(geometry) != expected_rows
        or int(geometry["well_id"].nunique()) != len(wells)
    ):
        raise ValueError(f"exp500 Stage 1 shard {index} geometry coverage changed")
    warm_up_pf_kernel()
    geometry_groups = {
        str(well): group.copy()
        for well, group in geometry.groupby("well_id", sort=False)
    }
    workers = int(get_nested(config, "runtime.num_workers", 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frozen_wells = list(
            executor.map(
                lambda well: decode_target_free_well(
                    str(well), raw_dir, geometry_groups[str(well)], config
                ),
                wells,
            )
        )
    predictions, residual, evidence, segments, audit, frozen = (
        freeze_target_free_outputs(
            frozen_wells,
            output,
            config=config,
            ledger=ledger,
            artifact_tag=tag,
            expected_rows=expected_rows,
            expected_wells=len(wells),
        )
    )
    elapsed = time.time() - started
    expected_well_count = int(
        get_nested(config, "execution.stage_1.expected_shard_wells")[index]
    )
    expected_seed_runs = expected_well_count * int(
        get_nested(config, "model.particle_filter.seeds")
    )
    expected_particle_starts = expected_seed_runs * int(
        get_nested(config, "model.particle_filter.particles")
    )
    technical_checks = {
        "coverage": bool(len(predictions) == expected_rows and len(wells) == expected_well_count),
        "residual_coverage": len(residual) == expected_rows,
        "seed_evidence_coverage": len(evidence) == expected_seed_runs,
        "k16_coverage": len(segments) == expected_well_count * 16,
        "audit_all_ok": bool(len(audit) == expected_well_count and audit["status"].eq("ok").all()),
        "execution_counts": bool(
            int(audit["candidate_pf_well_runs"].sum()) == expected_well_count
            and int(audit["seed_well_trajectories"].sum()) == expected_seed_runs
            and int(audit["particle_starts"].sum()) == expected_particle_starts
        ),
        "no_late_reads_before_freeze": bool(
            ledger.all_frozen
            and ledger.truth_rows_before_all_freeze == 0
            and ledger.control_rows_before_all_freeze == 0
            and ledger.role_fold_episode_rows_before_all_freeze == 0
            and ledger.forbidden_geometry_columns_read_before_freeze == 0
        ),
        "finite_and_normalized": bool(
            np.isfinite(predictions[PREDICTION_COLUMN]).all()
            and np.isfinite(residual.select_dtypes(include=[np.number]).to_numpy()).all()
            and float(np.max(np.abs(residual["particle_weight_sum"] - 1.0))) <= 1.0e-12
        ),
        "runtime": elapsed <= float(get_nested(config, "runtime.maximum_projected_full_shard_seconds")),
        "peak_rss": peak_rss_gb() <= float(get_nested(config, "runtime.maximum_peak_rss_gib")),
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "stage1_target_free_shard_complete",
        "stage": "stage1_target_free_shard",
        "shard_index": index,
        "shard_count": SHARD_COUNT,
        "rows": len(predictions),
        "wells": len(wells),
        "counts": {
            "candidate_pf_well_runs": int(audit["candidate_pf_well_runs"].sum()),
            "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
            "particle_starts": int(audit["particle_starts"].sum()),
            "control_pf_reruns": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "gpu_runs": 0,
        },
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "raw_identity_sha256": raw_identity["content_sha256"],
        "frozen_outputs": frozen,
        "technical_checks": technical_checks,
        "technical_all_pass": bool(all(technical_checks.values())),
        "truth_access_ledger": ledger.report(),
        "runtime": {
            "wall_seconds": elapsed,
            "summed_candidate_well_seconds": float(audit["wall_seconds"].sum()),
            "peak_rss_gib": peak_rss_gb(),
            "versions": runtime_versions(),
        },
        "artifacts": {
            "predictions": frozen["prediction_artifact"],
            "residual_ledger": frozen["residual_ledger_artifact"],
            "seed_evidence": frozen["seed_evidence_artifact"],
            "k16_rho_contract": frozen["k16_rho_contract_artifact"],
            "well_audit": frozen["well_audit"],
            "well_manifest": {
                "path": str(manifest_path),
                "raw_sha256": sha256_path(manifest_path),
            },
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_{tag}_summary.json"
    summary_artifact = write_json(summary_path, summary)
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def merge_stage1_shards(
    config: Mapping[str, Any],
    raw_manifest: pd.DataFrame,
    ledger: LeakageLedger,
    output: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    list[dict[str, Any]],
]:
    roots = [
        Path(str(value))
        for value in get_nested(config, "execution.stage_1_merge_shard_dirs")
    ]
    prediction_parts: list[pd.DataFrame] = []
    residual_parts: list[pd.DataFrame] = []
    evidence_parts: list[pd.DataFrame] = []
    segment_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    contract_sha = build_scientific_contract(config)["scientific_contract_sha256"]
    for index, root in enumerate(roots):
        tag = f"stage1_shard{index}"
        summary_path = _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_summary.json")
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("experiment") != EXPERIMENT_NAME
            or summary.get("stage") != "stage1_target_free_shard"
            or int(summary.get("shard_index", -1)) != index
            or int(summary.get("shard_count", -1)) != SHARD_COUNT
            or str(summary.get("scientific_contract_sha256")) != contract_sha
            or not bool(summary.get("technical_all_pass"))
        ):
            raise ValueError(f"exp500 Stage 1 shard {index} summary contract failed")
        paths = {
            "predictions": _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_predictions.csv.gz"),
            "residual_ledger": _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_residual_ledger.csv.gz"),
            "seed_evidence": _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_seed_evidence.csv.gz"),
            "k16_rho_contract": _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_k16_rho_contract.csv.gz"),
            "well_audit": _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_well_audit.csv"),
            "well_manifest": _artifact_file(root, f"{OUTPUT_PREFIX}_{tag}_well_manifest.csv"),
        }
        for name, path in paths.items():
            _verify_artifact_report(path, summary["artifacts"][name], f"shard {index} {name}")
        prediction = pd.read_csv(paths["predictions"], dtype={"id": str, "well_id": str})
        residual = pd.read_csv(paths["residual_ledger"], dtype={"id": str, "well_id": str})
        evidence = pd.read_csv(paths["seed_evidence"], dtype={"well_id": str})
        segments = pd.read_csv(paths["k16_rho_contract"], dtype={"well_id": str})
        audit = pd.read_csv(paths["well_audit"], dtype={"well_id": str})
        manifest = pd.read_csv(paths["well_manifest"], dtype={"well_id": str})
        frozen = summary["frozen_outputs"]
        logical_checks = {
            "prediction_logical_sha256": dataframe_content_sha(
                prediction, ["id", "well_id", "row_idx", PREDICTION_COLUMN]
            ),
            # These ledgers freeze their complete CSV payload. Re-hash the
            # exact saved bytes after decompression instead of formatting a
            # pandas float round-trip, which is content-equivalent but is not
            # guaranteed to reproduce the original decimal spelling.
            "residual_ledger_logical_sha256": sha256_csv_payload(
                paths["residual_ledger"]
            ),
            "seed_evidence_logical_sha256": sha256_csv_payload(
                paths["seed_evidence"]
            ),
            "k16_rho_contract_logical_sha256": sha256_csv_payload(
                paths["k16_rho_contract"]
            ),
        }
        for key, actual in logical_checks.items():
            if actual != str(frozen[key]):
                raise ValueError(f"shard {index} {key} mismatch after readback")
        if not manifest["shard_index"].astype(int).eq(index).all():
            raise ValueError(f"shard {index} manifest assignment changed")
        prediction_parts.append(prediction)
        residual_parts.append(residual)
        evidence_parts.append(evidence)
        segment_parts.append(segments)
        audit_parts.append(audit)
        manifest_parts.append(manifest)
        summaries.append(summary)

    def stable_concat(parts: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True).sort_values(
            list(columns), kind="mergesort"
        ).reset_index(drop=True)

    predictions = stable_concat(prediction_parts, ["well_id", "row_idx"])
    residual = stable_concat(residual_parts, ["well_id", "row_idx"])
    evidence = stable_concat(evidence_parts, ["well_id", "seed_index"])
    segments = stable_concat(segment_parts, ["well_id", "k16_segment_id"])
    audit = stable_concat(audit_parts, ["well_id"])
    manifest = stable_concat(manifest_parts, ["well_id"])
    expected_manifest = raw_manifest.sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    manifest_columns = ["well_id", "rows", "prefix_rows", "suffix_rows", "shard_index"]
    for column in manifest_columns[1:]:
        manifest[column] = pd.to_numeric(manifest[column], errors="raise").astype(np.int64)
        expected_manifest[column] = pd.to_numeric(
            expected_manifest[column], errors="raise"
        ).astype(np.int64)
    if not manifest[manifest_columns].equals(expected_manifest[manifest_columns]):
        raise ValueError("exp500 merged manifest differs from deterministic raw manifest")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(predictions) != expected_rows
        or len(residual) != expected_rows
        or len(evidence) != expected_wells * 128
        or len(segments) != expected_wells * 16
        or len(audit) != expected_wells
        or predictions["id"].duplicated().any()
        or residual["id"].duplicated().any()
        or evidence.duplicated(["well_id", "seed_index"]).any()
        or segments.duplicated(["well_id", "k16_segment_id"]).any()
        or audit["well_id"].duplicated().any()
    ):
        raise ValueError("exp500 merged Stage 1 coverage mismatch")
    tag = "stage1_merged"
    artifacts = {
        "predictions": write_deterministic_gzip_csv(
            predictions, output / f"{OUTPUT_PREFIX}_{tag}_predictions.csv.gz"
        ),
        "residual_ledger": write_deterministic_gzip_csv(
            residual, output / f"{OUTPUT_PREFIX}_{tag}_residual_ledger.csv.gz"
        ),
        "seed_evidence": write_deterministic_gzip_csv(
            evidence, output / f"{OUTPUT_PREFIX}_{tag}_seed_evidence.csv.gz"
        ),
        "k16_rho_contract": write_deterministic_gzip_csv(
            segments, output / f"{OUTPUT_PREFIX}_{tag}_k16_rho_contract.csv.gz"
        ),
    }
    audit_path = output / f"{OUTPUT_PREFIX}_{tag}_well_audit.csv"
    manifest_path = output / f"{OUTPUT_PREFIX}_{tag}_well_manifest.csv"
    audit.to_csv(audit_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    artifacts["well_audit"] = {"path": str(audit_path), "raw_sha256": sha256_path(audit_path)}
    artifacts["well_manifest"] = {
        "path": str(manifest_path),
        "raw_sha256": sha256_path(manifest_path),
    }
    for well in manifest["well_id"].astype(str):
        ledger.freeze(ACTIVE_VARIANT, well)
    if not ledger.all_frozen:
        raise RuntimeError("exp500 Stage 1 merged union did not freeze")
    frozen = {
        "stage": "stage1_merged_target_free_freeze",
        "frozen_before_truth_attachment": True,
        "rows": len(predictions),
        "wells": expected_wells,
        "prediction_logical_sha256": dataframe_content_sha(
            predictions, ["id", "well_id", "row_idx", PREDICTION_COLUMN]
        ),
        "residual_ledger_logical_sha256": dataframe_content_sha(
            residual, RESIDUAL_LEDGER_COLUMNS
        ),
        "seed_evidence_logical_sha256": dataframe_content_sha(
            evidence, SEED_EVIDENCE_COLUMNS
        ),
        "k16_rho_contract_logical_sha256": dataframe_content_sha(
            segments, list(segments.columns)
        ),
        "artifacts": artifacts,
        "shard_prediction_logical_sha256": [
            value["frozen_outputs"]["prediction_logical_sha256"] for value in summaries
        ],
        "truth_access_ledger_at_freeze": ledger.report(),
        "all_shard_artifact_sha_reverified": True,
    }
    return predictions, residual, evidence, segments, audit, frozen, summaries


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


def load_full_episode_boundaries_after_freeze(
    config: Mapping[str, Any], dotted_key: str, ledger: LeakageLedger
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("full episode boundaries require complete Stage 1 freeze")
    spec = dict(get_nested(config, dotted_key) or {})
    path = asset_path(config, dotted_key)
    frame = pd.read_csv(
        path,
        usecols=["episode_id", "well", "start_row_idx", "end_row_idx_exclusive"],
        dtype={"well": str},
    )
    ledger.record_role_fold_episode(len(frame))
    episode_rows = int((frame["end_row_idx_exclusive"] - frame["start_row_idx"]).sum())
    if len(frame) != int(spec["expected_episodes"]) or episode_rows != int(
        spec["expected_episode_rows"]
    ):
        raise ValueError(f"exp500 {dotted_key} episode census changed")
    return frame.sort_values(["well", "start_row_idx"], kind="mergesort").reset_index(
        drop=True
    )


def attach_truth_late_stage1(
    predictions: pd.DataFrame,
    residual: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    raw_dir: Path,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    _require_frozen(frozen)
    if dataframe_content_sha(
        predictions, ["id", "well_id", "row_idx", PREDICTION_COLUMN]
    ) != str(frozen["prediction_logical_sha256"]):
        raise RuntimeError("exp500 Stage 1 predictions changed after union freeze")
    if dataframe_content_sha(residual, RESIDUAL_LEDGER_COLUMNS) != str(
        frozen["residual_ledger_logical_sha256"]
    ):
        raise RuntimeError("exp500 Stage 1 residual ledger changed after union freeze")
    wells = sorted(predictions["well_id"].astype(str).unique().tolist())
    truth = pd.concat(
        [load_suffix_truth(well, raw_dir, ledger) for well in wells], ignore_index=True
    )
    frame = predictions.merge(
        truth, on=["id", "well_id", "row_idx"], how="inner", validate="one_to_one"
    )
    frame = _align_on_id(
        frame,
        residual[["id", "tvt_geop"]],
        ["tvt_geop"],
        label="frozen exp226 geometry",
    )
    identifiers = set(frame["id"].astype(str))
    for dotted_key, output_column in (
        ("data.saved_exp404_control", PRIMARY_CONTROL),
        ("data.saved_exp486_parent", PARENT_RESIDUAL),
    ):
        saved = load_saved_prediction_after_freeze(
            config, dotted_key, identifiers, ledger, output_column
        )
        frame = _align_on_id(frame, saved, [output_column], label=dotted_key)

    hmm_spec = dict(get_nested(config, "data.saved_exp209_hmm") or {})
    hmm_path = resolve_existing(
        str(hmm_spec["filename"]), hmm_spec.get("candidates", []), hmm_spec.get("patterns", [])
    )
    if sha256_decompressed_csv(hmm_path) != str(hmm_spec["expected_decompressed_sha256"]):
        raise ValueError("exp500 saved exp209 HMM decompressed SHA mismatch")
    hmm_source = str(hmm_spec["prediction_column"])
    hmm = pd.read_csv(
        hmm_path, usecols=["id", hmm_source], dtype={"id": str}, compression="infer"
    )
    ledger.record_control(len(hmm))
    hmm = hmm.rename(columns={hmm_source: "saved_exp209_hmm"})
    frame = _align_on_id(frame, hmm, ["saved_exp209_hmm"], label="saved exp209 HMM")

    geometry_path = geometry_input_path(config)
    fold = pd.read_csv(
        geometry_path,
        usecols=["well_id", "row_idx", "suffix_offset", "fold", "tvt_pred"],
        dtype={"well_id": str},
        compression="infer",
    )
    ledger.record_role_fold_episode(len(fold))
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    fold["tvt_pred"] = pd.to_numeric(fold["tvt_pred"], errors="raise").astype(
        np.float64
    )
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp500 reporting folds contain duplicate identities")
    fold = fold.rename(
        columns={
            "suffix_offset": "reporting_suffix_offset",
            "tvt_pred": "exp226_final_tvt_pred",
        }
    )
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if frame[["fold", "reporting_suffix_offset"]].isna().any().any() or not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["reporting_suffix_offset"].to_numpy(np.int64),
    ):
        raise ValueError("exp500 reporting fold alignment failed")
    frame = frame.drop(columns=["reporting_suffix_offset"])

    hidden_spec = dict(get_nested(config, "data.hidden_like_assignment") or {})
    hidden_path = resolve_existing(
        str(hidden_spec["filename"]),
        hidden_spec.get("candidates", []),
        hidden_spec.get("patterns", []),
    )
    if sha256_path(hidden_path) != str(hidden_spec["expected_sha256"]):
        raise ValueError("exp500 hidden-like role SHA mismatch")
    role_columns = {
        str(scope): str(column)
        for scope, column in dict(hidden_spec["role_columns"]).items()
    }
    hidden = pd.read_csv(
        hidden_path, usecols=["well_id", *role_columns.values()], dtype={"well_id": str}
    )
    ledger.record_role_fold_episode(len(hidden))
    if hidden["well_id"].duplicated().any():
        raise ValueError("exp500 hidden-like roles contain duplicate wells")
    for scope, column in role_columns.items():
        actual = {
            str(key): int(value)
            for key, value in hidden[column].astype(str).value_counts().sort_index().items()
        }
        expected = {
            str(key): int(value)
            for key, value in dict(hidden_spec["expected_role_counts"][scope]).items()
        }
        if actual != expected:
            raise ValueError(f"exp500 hidden-like role counts changed for {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("exp500 hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[role_columns["hidden_like_spatial"]].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[
        role_columns["hidden_like_typewell_purged"]
    ].eq("valid")
    frame["candidate_hmm_50_50"] = 0.5 * (
        frame[PREDICTION_COLUMN].to_numpy(np.float64)
        + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    frame["exp404_hmm_50_50"] = 0.5 * (
        frame[PRIMARY_CONTROL].to_numpy(np.float64)
        + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    for name, prediction_column in (
        ("candidate", PREDICTION_COLUMN),
        ("exp404", PRIMARY_CONTROL),
        ("exp486", PARENT_RESIDUAL),
        ("exp226", "exp226_final_tvt_pred"),
    ):
        frame[f"{name}_error"] = frame[prediction_column] - frame["true_tvt"]
    finite = [
        "true_tvt",
        "tvt_geop",
        "exp226_final_tvt_pred",
        PREDICTION_COLUMN,
        PRIMARY_CONTROL,
        PARENT_RESIDUAL,
        "saved_exp209_hmm",
        "candidate_hmm_50_50",
        "exp404_hmm_50_50",
    ]
    if not np.isfinite(frame[finite].to_numpy(np.float64)).all():
        raise ValueError("exp500 Stage 1 late readout contains non-finite values")
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or int(frame["well_id"].nunique())
        != int(get_nested(config, "validation.expected_wells"))
        or sorted(frame["fold"].astype(int).unique().tolist())
        != [int(value) for value in get_nested(config, "validation.expected_folds")]
    ):
        raise ValueError("exp500 Stage 1 late readout coverage changed")
    return frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )


def stage1_metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    scope: str,
    candidate_column: str = PREDICTION_COLUMN,
    control_column: str = PRIMARY_CONTROL,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"exp500 Stage 1 metric scope is empty: {scope}")
    candidate_error = selected[candidate_column] - selected["true_tvt"]
    control_error = selected[control_column] - selected["true_tvt"]
    candidate_rmse = rmse(candidate_error)
    control_rmse = rmse(control_error)
    return {
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate": candidate_column,
        "control": control_column,
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
    }


def build_stage1_metrics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    scopes: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(frame), dtype=bool))]
    for fold in sorted(frame["fold"].astype(int).unique().tolist()):
        scopes.append((f"fold_{fold}", frame["fold"].eq(fold).to_numpy()))
    scopes.extend(
        [
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
            ("missing_fraction_high", frame["well_missing_fraction"].ge(0.30).to_numpy()),
            ("md_since_1000_plus", frame["md_since"].ge(1000.0).to_numpy()),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    primary = pd.DataFrame(
        [stage1_metric_record(frame, mask, scope=scope) for scope, mask in scopes]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        candidate_rmse = rmse(group["candidate_error"])
        control_rmse = rmse(group["exp404_error"])
        by_well_rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "candidate_rmse": candidate_rmse,
                "exp404_rmse": control_rmse,
                "delta_rmse_candidate_minus_exp404": candidate_rmse - control_rmse,
                "well_missing_fraction": float(group["well_missing_fraction"].iloc[0]),
            }
        )
    reference = {
        "candidate_rmse": rmse(frame["candidate_error"]),
        "exp404_rmse": rmse(frame["exp404_error"]),
        "exp486_rmse": rmse(frame["exp486_error"]),
        "exp226_rmse": rmse(frame["exp226_error"]),
        "candidate_hmm_50_50_rmse": rmse(
            frame["candidate_hmm_50_50"] - frame["true_tvt"]
        ),
        "exp404_hmm_50_50_rmse": rmse(
            frame["exp404_hmm_50_50"] - frame["true_tvt"]
        ),
    }
    return primary, pd.DataFrame(by_well_rows), reference


def evaluate_stage1_gates(
    *,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    residual: pd.DataFrame,
    evidence: pd.DataFrame,
    segments: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    shard_summaries: Sequence[Mapping[str, Any]],
    primary_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    reference: Mapping[str, float],
    exp408_metrics: pd.DataFrame,
    exp410_metrics: pd.DataFrame,
    ledger_at_freeze: Mapping[str, Any],
    raw_identity: Mapping[str, Any],
) -> dict[str, Any]:
    guards = dict(get_nested(config, "guards.scientific_stage_1") or {})
    before = dict(ledger_at_freeze["before_freeze"])
    overall = primary_metrics.loc[primary_metrics["scope"].eq("overall")].iloc[0]
    folds = primary_metrics.loc[primary_metrics["scope"].str.startswith("fold_")]
    improved_folds = int((folds["improvement_ft"] > 0.0).sum())
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    counts = {
        "candidate_pf_well_runs": int(audit["candidate_pf_well_runs"].sum()),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_reruns": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        "candidate_pf_well_runs": 773,
        "seed_well_trajectories": 98944,
        "particle_starts": 49472000,
        "control_pf_reruns": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu_runs": 0,
    }
    temperature_sums = evidence.groupby("well_id")["temperature_weight"].sum()
    technical_checks = {
        "stage0_fail_preserved_with_explicit_override": bool(
            not bool(get_nested(config, "implementation.stage_0_all_pass"))
            and bool(get_nested(config, "execution.stage_1_stage0_gate_override_approved"))
        ),
        "raw_input_identity": raw_identity["content_sha256"]
        == str(get_nested(config, "data.expected_raw_well_identity_sha256")),
        "row_and_well_coverage": bool(
            len(frame) == expected_rows and frame["well_id"].nunique() == expected_wells
        ),
        "reporting_fold_coverage": sorted(frame["fold"].astype(int).unique().tolist())
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
        "all_wells_completed": bool(
            len(audit) == expected_wells and audit["status"].eq("ok").all()
        ),
        "finite_coverage": bool(
            np.isfinite(frame[[PREDICTION_COLUMN, "true_tvt"]].to_numpy(np.float64)).all()
            and np.isfinite(residual.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "weight_normalization": bool(
            float(np.max(np.abs(residual["particle_weight_sum"] - 1.0))) <= 1.0e-12
            and float(np.max(np.abs(temperature_sums - 1.0))) <= 1.0e-12
        ),
        "k16_segment_half_life": bool(
            len(segments) == expected_wells * 16
            and float(segments["rho_product_abs_error_vs_half"].max())
            <= float(get_nested(config, "guards.technical_stage_0.segment_cumulative_rho_atol"))
        ),
        "truth_control_role_fold_episode_reads_before_freeze_zero": bool(
            int(before["truth_rows"]) == 0
            and int(before["control_rows"]) == 0
            and int(before["role_fold_episode_rows"]) == 0
            and int(before["forbidden_geometry_columns"]) == 0
        ),
        "execution_count_match": counts == expected_counts,
        "all_shard_artifact_sha_reverified": bool(
            frozen["all_shard_artifact_sha_reverified"]
        ),
        "all_shard_technical_pass": all(
            bool(summary["technical_all_pass"]) for summary in shard_summaries
        ),
        "saved_exp404_rmse_parity": abs(
            float(reference["exp404_rmse"])
            - float(get_nested(config, "validation.primary_control_rmse_ft"))
        )
        <= 0.00001,
        "saved_exp486_rmse_parity": abs(
            float(reference["exp486_rmse"])
            - float(get_nested(config, "validation.parent_residual_pf_rmse_ft"))
        )
        <= 0.00001,
        "exp226_reference_rmse_parity": abs(
            float(reference["exp226_rmse"])
            - float(get_nested(config, "validation.exp226_final_reference_rmse_ft"))
        )
        <= 0.00001,
        "fixed_exp209_exp404_50_50_parity": abs(
            float(reference["exp404_hmm_50_50_rmse"])
            - float(get_nested(config, "validation.fixed_hmm_pf_50_50_reference_rmse_ft"))
        )
        <= 0.00001,
        "maximum_shard_runtime": max(
            float(summary["runtime"]["wall_seconds"]) for summary in shard_summaries
        )
        <= float(get_nested(config, "runtime.maximum_projected_full_shard_seconds")),
        "maximum_peak_rss": max(
            float(summary["runtime"]["peak_rss_gib"]) for summary in shard_summaries
        )
        <= float(get_nested(config, "runtime.maximum_peak_rss_gib")),
    }
    scope_rules = {
        "raw_gr_observed": ("gain", "minimum_raw_gr_observed_gain_vs_exp404_ft"),
        "raw_gr_missing": ("regression", "maximum_raw_gr_missing_regression_vs_exp404_ft"),
        "missing_fraction_high": ("regression", "maximum_high_missing_regression_vs_exp404_ft"),
        "md_since_1000_plus": ("regression", "maximum_long_tail_1000_plus_regression_vs_exp404_ft"),
        "hidden_like_spatial": ("regression", "maximum_hidden_like_spatial_regression_vs_exp404_ft"),
        "hidden_like_typewell_purged": (
            "regression",
            "maximum_hidden_like_typewell_purged_regression_vs_exp404_ft",
        ),
    }
    scope_checks: dict[str, Any] = {}
    for scope, (kind, key) in scope_rules.items():
        row = primary_metrics.loc[primary_metrics["scope"].eq(scope)].iloc[0]
        threshold = float(guards[key])
        value = float(row["improvement_ft"] if kind == "gain" else row["delta_rmse_candidate_minus_control"])
        scope_checks[scope] = {
            "candidate_rmse": float(row["candidate_rmse"]),
            "exp404_rmse": float(row["control_rmse"]),
            "improvement_ft": float(row["improvement_ft"]),
            "delta_rmse_candidate_minus_exp404": float(
                row["delta_rmse_candidate_minus_control"]
            ),
            "rule": kind,
            "threshold_ft": threshold,
            "passed": bool(value >= threshold if kind == "gain" else value <= threshold),
        }
    by_well_delta = by_well["delta_rmse_candidate_minus_exp404"]
    p95 = float(by_well_delta.quantile(0.95))
    worst_index = by_well_delta.idxmax()
    worst = float(by_well_delta.loc[worst_index])

    def episode_report(metrics: pd.DataFrame, *, include_recovery: bool) -> dict[str, Any]:
        baseline_sse = float(metrics["baseline_sse"].sum())
        candidate_sse = float(metrics["candidate_sse"].sum())
        result: dict[str, Any] = {
            "baseline_sse": baseline_sse,
            "candidate_sse": candidate_sse,
            "sse_reduction_fraction": 1.0 - candidate_sse / baseline_sse,
        }
        if include_recovery:
            result["recovery"] = {}
            for horizon in (256, 512):
                baseline_rate = float(metrics[f"baseline_recovered_{horizon}"].mean())
                candidate_rate = float(metrics[f"candidate_recovered_{horizon}"].mean())
                result["recovery"][str(horizon)] = {
                    "baseline_rate": baseline_rate,
                    "candidate_rate": candidate_rate,
                    "delta": candidate_rate - baseline_rate,
                }
        return result

    exp408_report = episode_report(exp408_metrics, include_recovery=True)
    exp410_report = episode_report(exp410_metrics, include_recovery=False)
    exp408_wells = set(exp408_metrics["well"].astype(str))
    count_rows = frame.loc[frame["well_id"].isin(exp408_wells)]
    baseline_count = sum(
        contiguous_episode_count(part["exp404_error"].to_numpy(np.float64))
        for _, part in count_rows.groupby("well_id", sort=True)
    )
    candidate_count = sum(
        contiguous_episode_count(part["candidate_error"].to_numpy(np.float64))
        for _, part in count_rows.groupby("well_id", sort=True)
    )
    scientific_checks = {
        "pooled_gain_vs_exp404": float(reference["exp404_rmse"] - reference["candidate_rmse"])
        >= float(guards["minimum_rmse_gain_vs_exp404_ft"]),
        "pooled_gain_vs_exp486": float(reference["exp486_rmse"] - reference["candidate_rmse"])
        >= float(guards["minimum_rmse_gain_vs_exp486_residual_ft"]),
        "improved_folds_vs_exp404": improved_folds
        >= int(guards["minimum_improved_folds_vs_exp404"]),
        "scope_checks": all(value["passed"] for value in scope_checks.values()),
        "by_well_p95": p95 <= float(guards["maximum_by_well_delta_rmse_p95_vs_exp404_ft"]),
        "worst_well": worst <= float(guards["maximum_worst_well_regression_vs_exp404_ft"]),
        "exp408_episode_sse": exp408_report["sse_reduction_fraction"]
        >= float(guards["minimum_exp408_episode_sse_reduction_vs_exp404"]),
        "exp408_episode_count": candidate_count - baseline_count
        <= int(guards["maximum_exp408_episode_count_delta_vs_exp404"]),
        "exp408_recovery_256": exp408_report["recovery"]["256"]["delta"]
        >= float(guards["minimum_recovery_rate_256_delta_vs_exp404"]),
        "exp408_recovery_512": exp408_report["recovery"]["512"]["delta"]
        >= float(guards["minimum_recovery_rate_512_delta_vs_exp404"]),
        "exp410_episode_sse": exp410_report["sse_reduction_fraction"]
        >= float(guards["minimum_exp410_episode_sse_reduction_vs_exp404"]),
        "fixed_exp209_50_50_rmse": float(reference["candidate_hmm_50_50_rmse"])
        <= float(guards["maximum_fixed_exp209_50_50_rmse_ft"]),
        "gain_vs_exp226": float(reference["exp226_rmse"] - reference["candidate_rmse"])
        >= float(guards["minimum_gain_vs_exp226_final_ft"]),
        "absolute_candidate_rmse": float(reference["candidate_rmse"])
        <= float(guards["maximum_candidate_rmse_ft"]),
    }
    technical_all_pass = bool(all(technical_checks.values()))
    scientific_all_pass = bool(all(scientific_checks.values()))
    return {
        "stage": "stage1_full_oof_under_explicit_stage0_override",
        "stage0_failure_preserved": True,
        "technical_checks": technical_checks,
        "technical_all_pass": technical_all_pass,
        "scientific_checks": scientific_checks,
        "scientific_all_pass": scientific_all_pass,
        "all_pass": bool(technical_all_pass and scientific_all_pass),
        "counts": counts,
        "reference_rmse": dict(reference),
        "improved_folds_vs_exp404": improved_folds,
        "scope_checks": scope_checks,
        "by_well_delta_p95_ft": p95,
        "worst_well_id": str(by_well.loc[worst_index, "well_id"]),
        "worst_well_regression_ft": worst,
        "exp408": {
            **exp408_report,
            "baseline_episode_count": baseline_count,
            "candidate_episode_count": candidate_count,
            "episode_count_delta": candidate_count - baseline_count,
        },
        "exp410": exp410_report,
        "truth_access_ledger_at_freeze": dict(ledger_at_freeze),
        "decision": (
            "eligible_for_same_exp_inference_design_pending_separate_approval"
            if technical_all_pass and scientific_all_pass
            else "terminal_close_without_same_oof_rescue"
        ),
    }


def run_stage1_merge(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(
        config, require_run_approval=True
    )
    require_kaggle_runtime()
    if selected_stage1(config) != "stage1_merge":
        raise RuntimeError("exp500 Stage 1 merge is not selected")
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_identity = validate_raw_well_identity(config, raw_dir)
    raw_manifest = build_stage1_well_manifest(config, raw_dir, raw_identity["well_ids"])
    ledger = LeakageLedger(expected_variant_wells=int(get_nested(config, "validation.expected_wells")))
    predictions, residual, evidence, segments, audit, frozen, shard_summaries = (
        merge_stage1_shards(config, raw_manifest, ledger, output)
    )
    ledger_at_freeze = ledger.report()
    freeze_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_merged_freeze_manifest.json", frozen
    )
    frame = attach_truth_late_stage1(
        predictions,
        residual,
        frozen,
        config=config,
        raw_dir=raw_dir,
        ledger=ledger,
    )
    primary_metrics, by_well, reference = build_stage1_metrics(frame)
    exp408_episodes = load_full_episode_boundaries_after_freeze(
        config, "data.exp408_persistent_episodes", ledger
    )
    exp410_episodes = load_full_episode_boundaries_after_freeze(
        config, "data.exp410_persistent_episodes", ledger
    )
    exp408_metrics = build_episode_metrics(
        exp408_episodes, frame, baseline_error_column="exp404_error"
    )
    exp410_metrics = build_episode_metrics(
        exp410_episodes, frame, baseline_error_column="exp404_error"
    )
    gate = evaluate_stage1_gates(
        config=config,
        frame=frame,
        residual=residual,
        evidence=evidence,
        segments=segments,
        audit=audit,
        frozen=frozen,
        shard_summaries=shard_summaries,
        primary_metrics=primary_metrics,
        by_well=by_well,
        reference=reference,
        exp408_metrics=exp408_metrics,
        exp410_metrics=exp410_metrics,
        ledger_at_freeze=ledger_at_freeze,
        raw_identity=raw_identity,
    )
    paths = {
        "primary_metrics": output / f"{OUTPUT_PREFIX}_stage1_primary_metrics.csv",
        "by_well": output / f"{OUTPUT_PREFIX}_stage1_by_well.csv",
        "exp408_episode_metrics": output / f"{OUTPUT_PREFIX}_stage1_exp408_episode_metrics.csv",
        "exp410_episode_metrics": output / f"{OUTPUT_PREFIX}_stage1_exp410_episode_metrics.csv",
        "gate": output / f"{OUTPUT_PREFIX}_stage1_gate_report.json",
    }
    primary_metrics.to_csv(paths["primary_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    exp408_metrics.to_csv(paths["exp408_episode_metrics"], index=False)
    exp410_metrics.to_csv(paths["exp410_episode_metrics"], index=False)
    gate_artifact = write_json(paths["gate"], gate)
    status = (
        "stage1_all_pass_under_override_pending_separate_inference_approval"
        if gate["all_pass"]
        else "stage1_fail_closed_under_override"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage1_full_oof_under_explicit_stage0_override",
        "stage0_status_preserved": "stage0_fail_closed",
        "metric": "rmse",
        "cv": float(reference["candidate_rmse"]),
        "reference_rmse": dict(reference),
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "counts": gate["counts"],
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "promotion_gate": gate,
        "truth_access_ledger": ledger.report(),
        "runtime": {
            "merge_and_evaluation_wall_seconds": time.time() - started,
            "shard_wall_seconds": [
                float(value["runtime"]["wall_seconds"]) for value in shard_summaries
            ],
            "peak_rss_gib": peak_rss_gb(),
            "versions": runtime_versions(),
        },
        "artifacts": {
            **frozen["artifacts"],
            "freeze_manifest": freeze_artifact,
            "primary_metrics": {
                "path": str(paths["primary_metrics"]),
                "raw_sha256": sha256_path(paths["primary_metrics"]),
            },
            "by_well": {"path": str(paths["by_well"]), "raw_sha256": sha256_path(paths["by_well"])},
            "exp408_episode_metrics": {
                "path": str(paths["exp408_episode_metrics"]),
                "raw_sha256": sha256_path(paths["exp408_episode_metrics"]),
            },
            "exp410_episode_metrics": {
                "path": str(paths["exp410_episode_metrics"]),
                "raw_sha256": sha256_path(paths["exp410_episode_metrics"]),
            },
            "gate": gate_artifact,
        },
        "public_lb": None,
        "private_lb": None,
        "deterministic_anchor": False,
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": gate["decision"],
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_summary.json", summary
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    if bool(get_nested(config, "execution.run_stage_0", False)):
        if selected_stage1(config) is not None:
            raise ValueError("exp500 Stage 0 and Stage 1 cannot run together")
        return run_stage0(config)
    stage = selected_stage1(config)
    if stage == "stage1_shard":
        return run_stage1_shard(config)
    if stage == "stage1_merge":
        return run_stage1_merge(config)
    return None


# %% [markdown]
# ## 12. Setup, execution selection, and configuration preview


# %%
CONFIG = load_experiment_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG, require_run_approval=False)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "active_variants": get_nested(CONFIG, "model.active_variants"),
            "implementation_scope": get_nested(CONFIG, "implementation.scope"),
            "stage0_candidate_pf_well_runs": get_nested(
                CONFIG, "execution.stage_0.candidate_pf_well_runs"
            ),
            "stage0_seed_well_trajectories": get_nested(
                CONFIG, "execution.stage_0.seed_well_trajectories"
            ),
            "stage0_particle_starts": get_nested(
                CONFIG, "execution.stage_0.particle_starts"
            ),
            "control_pf_reruns": get_nested(
                CONFIG, "execution.stage_0.parent_pf_control_reruns"
            ),
            "canonical_notebook_adoption_approved": get_nested(
                CONFIG, "execution.canonical_notebook_adoption_approved"
            ),
            "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
            "stage0_run_approved": get_nested(CONFIG, "execution.stage_0_run_approved"),
            "stage1_implemented": get_nested(CONFIG, "implementation.stage_1_implemented"),
            "selected_stage1": selected_stage1(CONFIG),
            "stage1_candidate_pf_well_runs": get_nested(
                CONFIG, "execution.stage_1.candidate_pf_well_runs"
            ),
            "stage1_seed_well_trajectories": get_nested(
                CONFIG, "execution.stage_1.seed_well_trajectories"
            ),
            "stage1_particle_starts": get_nested(
                CONFIG, "execution.stage_1.particle_starts"
            ),
            "stage0_fail_preserved": not bool(
                get_nested(CONFIG, "implementation.stage_0_all_pass")
            ),
            "inference_enabled": get_nested(CONFIG, "implementation.inference_enabled"),
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                "scientific_contract_sha256"
            ],
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    STAGE_RESULT = run_selected_stage(CONFIG)
    if STAGE_RESULT is None:
        print("exp500 has no armed execution stage.")
else:
    STAGE_RESULT = None
    print(
        "exp500 import-only preview; Stage 0 remains failed, while Stage 1 "
        "execution is selected only in generated shard/merge packages. "
        "Inference and submission remain disabled."
    )
