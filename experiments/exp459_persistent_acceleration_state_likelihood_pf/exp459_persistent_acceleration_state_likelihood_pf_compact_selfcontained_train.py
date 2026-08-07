# %% [markdown]
# # exp459 persistent acceleration state likelihood-PF — train
#
# This train-side experiment extends the frozen exp404/exp417 likelihood-PF
# state from `(TVT, U-rate)` to `(TVT, U-rate, persistent U-acceleration)`.
# Stage 0 is a fixed32 technical/mechanism preflight, not CV. Candidate
# predictions and target-free acceleration diagnostics are frozen before
# suffix truth, saved controls, roles, folds, or persistent episodes are read.

# %% [markdown]
# ## Contents
# 1. Imports and notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Fixed32 scope, target-free input, and leakage ledger
# 5. Exp404 likelihood-PF input preparation
# 6. Persistent acceleration transition and likelihood-PF kernel
# 7. Zero-acceleration exp404 parity and kinematic contracts
# 8. Target-free candidate generation and freeze
# 9. Truth-late mechanism readout and fail-closed gates
# 10. Generated artifacts and Stage 0 orchestration
# 11. Setup and configuration preview

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


EXPERIMENT_NAME = "exp459_persistent_acceleration_state_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "likpf_scale5_persistent_acceleration3"
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
ACCELERATION_LOGICAL_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "acceleration_probability_negative",
    "acceleration_probability_zero",
    "acceleration_probability_positive",
    "filtered_acceleration_mean",
    "nonzero_acceleration_mass",
    "particle_count_negative_all_seeds",
    "particle_count_zero_all_seeds",
    "particle_count_positive_all_seeds",
    "filtered_rate_mean",
    "filtered_rate_std",
    "effective_sample_size",
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP459_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp459 config not found; checked={checked}")


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


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return mapping_sha256(schema)


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
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
def acceleration_transition_matrix(acceleration: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(acceleration["values_rate_per_md_ft"], dtype=np.float64)
    matrix = np.asarray(acceleration["transition_matrix"], dtype=np.float64)
    if values.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("exp459 requires exactly three acceleration states")
    if not np.array_equal(values, np.asarray([-0.0005, 0.0, 0.0005])):
        raise ValueError("exp459 acceleration values changed")
    expected = np.asarray(
        [[0.92, 0.08, 0.0], [0.08, 0.84, 0.08], [0.0, 0.08, 0.92]],
        dtype=np.float64,
    )
    if not np.array_equal(matrix, expected):
        raise ValueError("exp459 boundary-folded acceleration transition changed")
    if np.any(matrix < 0.0) or float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) > 1e-12:
        raise ValueError("exp459 acceleration transition is not stochastic")
    return matrix


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool,
) -> dict[str, int]:
    expected = {
        "active_variants": 1,
        "stage_0_candidate_pf_well_runs": 32,
        "stage_0_seed_well_trajectories": 4096,
        "stage_0_particle_starts": 2048000,
        "stage_0_zero_acceleration_sentinel_wells": 4,
        "stage_1_candidate_pf_well_runs": 773,
        "stage_1_seed_well_trajectories": 98944,
        "stage_1_particle_starts": 49472000,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "control_pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1)) for key in expected
    }
    if observed != expected:
        raise ValueError(f"exp459 execution count contract changed: {observed}")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise RuntimeError("exp459 Kaggle Stage 0 run is not approved")
        if not bool(get_nested(config, "execution.run_stage_0", False)):
            raise RuntimeError("exp459 Stage 0 execution is disabled")
    return observed


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    acceleration = dict(get_nested(config, "model.acceleration") or {})
    matrix = acceleration_transition_matrix(acceleration)
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "implementation_reference": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "candidate": PRIMARY_CANDIDATE,
        "state": ["tvt_position", "u_rate", "u_acceleration"],
        "u_definition": "tvt_plus_z",
        "acceleration_values": acceleration["values_rate_per_md_ft"],
        "acceleration_transition": matrix.tolist(),
        "initial_acceleration_probability": [0.0, 1.0, 0.0],
        "update_order": [
            "sample_acceleration",
            "update_u_rate",
            "update_tvt_via_u_position_minus_z",
            "apply_current_gr_weight",
        ],
        "transition": {
            "delta_md": "max(current_md_minus_previous_md,1.0)",
            "rate": (
                "0.998*previous_rate+current_acceleration*delta_md"
                "+0.002*base_normal_rate"
            ),
            "tvt": (
                "previous_tvt+current_rate*delta_md-delta_z"
                "+0.005*base_normal_position"
            ),
        },
        "pf": {
            "particles": int(fixed["particles"]),
            "seeds": int(fixed["seeds"]),
            "temperature": float(fixed["primary_seed_weighting_temperature"]),
            "momentum": float(fixed["momentum"]),
            "rate_noise": float(fixed["rate_noise"]),
            "position_noise": float(fixed["position_noise"]),
            "rough_position": float(fixed["rough_position"]),
            "rough_rate": float(fixed["rough_rate"]),
            "resample_threshold_fraction": float(
                fixed["resample_threshold_fraction"]
            ),
            "gr_scale_multiplier": float(fixed["gr_scale_multiplier"]),
            "gr_emission": str(fixed["gr_emission"]),
        },
        "rng": {
            "base": "stable_seed(likpf,train,well)+seed_index",
            "acceleration": (
                "stable_seed(exp459,acceleration,split,well,seed_index)"
                " with independent Park-Miller stream"
            ),
            "acceleration_advances_base_stream": False,
        },
        "truth_attachment": str(get_nested(config, "validation.truth_attachment")),
        "saved_control_rerun": False,
        "scientific_variants": 1,
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
        "experiment.status": "stage0_fail_closed",
        "lineage.parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "implementation.enabled": True,
        "implementation.implementation_approval_received": True,
        "implementation.canonical_train_notebook_adopted": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.fixed32_is_cv": False,
        "model.active_variants": ["persistent_acceleration3"],
        "model.fixed_from_exp404.particles": 500,
        "model.fixed_from_exp404.seeds": 128,
        "model.fixed_from_exp404.primary_seed_weighting_temperature": 5.0,
        "model.fixed_from_exp404.gr_scale_multiplier": 1.0,
        "model.fixed_from_exp404.momentum": 0.998,
        "model.fixed_from_exp404.rate_noise": 0.002,
        "model.fixed_from_exp404.position_noise": 0.005,
        "model.fixed_from_exp404.rough_position": 0.1,
        "model.fixed_from_exp404.rough_rate": 0.001,
        "model.fixed_from_exp404.resample_threshold_fraction": 0.5,
        "model.fixed_from_exp404.typewell_grid_step_ft": 0.2,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    for key, required in expected.items():
        observed = get_nested(config, key)
        if observed != required:
            raise ValueError(
                f"exp459 scientific contract mismatch: {key}={observed!r}, "
                f"expected={required!r}"
            )
    validate_execution_contract(config, require_run_approval=require_run_approval)
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Fixed32 scope, target-free input, and leakage ledger


# %%
@dataclass
class LeakageLedger:
    frozen_wells: set[str] = field(default_factory=set)
    expected_wells: int = 0
    truth_rows_before_all_freeze: int = 0
    control_rows_before_all_freeze: int = 0
    role_fold_rows_before_all_freeze: int = 0
    episode_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    control_rows_after_all_freeze: int = 0
    role_fold_rows_after_all_freeze: int = 0
    episode_rows_after_all_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return self.expected_wells > 0 and len(self.frozen_wells) == self.expected_wells

    def freeze(self, well: str) -> None:
        self.frozen_wells.add(str(well))

    def _record(self, label: str, rows: int) -> None:
        before_name = f"{label}_before_all_freeze"
        after_name = f"{label}_after_all_freeze"
        if not self.all_frozen:
            setattr(self, before_name, int(getattr(self, before_name)) + int(rows))
            raise RuntimeError(f"{label} was read before all fixed32 artifacts froze")
        setattr(self, after_name, int(getattr(self, after_name)) + int(rows))

    def record_truth(self, rows: int) -> None:
        self._record("truth_rows", rows)

    def record_control(self, rows: int) -> None:
        self._record("control_rows", rows)

    def record_role_fold(self, rows: int) -> None:
        self._record("role_fold_rows", rows)

    def record_episode(self, rows: int) -> None:
        self._record("episode_rows", rows)

    def report(self) -> dict[str, Any]:
        return {
            "expected_wells": self.expected_wells,
            "frozen_wells": len(self.frozen_wells),
            "all_frozen": self.all_frozen,
            "before_freeze": {
                "truth_rows": self.truth_rows_before_all_freeze,
                "control_rows": self.control_rows_before_all_freeze,
                "role_fold_rows": self.role_fold_rows_before_all_freeze,
                "episode_rows": self.episode_rows_before_all_freeze,
            },
            "after_freeze": {
                "truth_rows": self.truth_rows_after_all_freeze,
                "control_rows": self.control_rows_after_all_freeze,
                "role_fold_rows": self.role_fold_rows_after_all_freeze,
                "episode_rows": self.episode_rows_after_all_freeze,
            },
        }


def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_path(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(
            f"exp459 fixed32 manifest SHA mismatch: expected={spec['expected_sha256']}, "
            f"observed={observed}"
        )
    return path


def load_fixed32_scope(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    path = fixed32_manifest_path(config)
    scope = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    expected = int(get_nested(config, "stages.stage_0.candidate_pf_well_runs"))
    if len(scope) != expected or scope["well"].nunique() != expected:
        raise ValueError("exp459 fixed32 scope identity changed")
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
    path = fixed32_manifest_path(config)
    frame = pd.read_csv(path, dtype={"well": str})
    ledger.record_role_fold(len(frame))
    required = {"well", "role", "fold", "matched_persistent_well"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fixed32 role manifest missing columns: {missing}")
    if (
        len(frame) != 32
        or frame["well"].nunique() != 32
        or frame["role"].value_counts().to_dict()
        != {"control": 16, "persistent": 16}
        or set(frame["fold"].astype(int)) != set(range(5))
    ):
        raise ValueError("exp459 fixed32 role/fold balance changed")
    return frame


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    for column in ("MD", "Z", "GR", "TVT_input"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["MD"].notna().all() or not frame["Z"].notna().all():
        raise ValueError(f"{well}: MD/Z contains missing values")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = (
        frame.dropna(subset=["TVT"])
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(frame) < 2 or not np.isfinite(
        frame["TVT"].to_numpy(np.float64)
    ).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    typewell_mean = float(frame["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(typewell_mean)
    return frame


# %% [markdown]
# ## 5. Exp404 likelihood-PF input preparation
#
# The target-free inputs, initial position/rate, Type Well grid, GR
# interpolation, and prefix GR scale reproduce the exp404 x1.0 surface.


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
    grid_gr = np.interp(
        grid_tvt,
        typewell_tvt,
        typewell_gr,
    ).astype(np.float64)
    return grid_gr, minimum, float(step)


def exp072_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires at least one known-prefix row")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = (
        horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    )
    typewell_at_known = np.interp(
        known_tvt,
        typewell_tvt,
        typewell_gr,
    )
    residual = known_gr - typewell_at_known
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
        "known_gr_missing_rows": int(
            horizontal.loc[known, "GR"].isna().sum()
        ),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "base_clip_min": 10.0,
        "base_clip_max": 60.0,
        "post_multiplier_clip_applied": False,
        "post_multiplier_clip_count": 0,
    }


def exp072_initial_rate(horizontal: pd.DataFrame, *, tail_rows: int = 30) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    tail = known.tail(tail_rows)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    valid = delta_md > 0.0
    if int(valid.sum()) < 3:
        return 0.0
    return float(
        np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid])
    )


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
    last_known_tvt = float(last_known["TVT_input"])
    last_known_z = float(last_known["Z"])
    last_known_md = float(last_known["MD"])
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    scale_audit = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    typewell_mean = float(typewell_gr.mean())
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_z = evaluation["Z"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    raw_gr_observed = evaluation["GR"].notna().to_numpy(bool)
    if not (
        np.isfinite(eval_md).all()
        and np.isfinite(eval_z).all()
        and np.isfinite(eval_gr).all()
    ):
        raise ValueError("likelihood-PF evaluation inputs are not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": eval_z,
        "eval_gr": eval_gr,
        "raw_gr_observed": raw_gr_observed,
        "md_since": eval_md - last_known_md,
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_known_tvt + last_known_z,
        "initial_rate": exp072_initial_rate(horizontal),
        "scale_audit": scale_audit,
    }


# %% [markdown]
# ## 6. Persistent acceleration transition and likelihood-PF kernel
#
# Acceleration uses a separate per-well/per-seed Park-Miller stream. All
# particle initialization, rate/position noise, resampling uniforms, and
# roughening remain on the exact exp404 NumPy stream. Resampling copies the
# selected acceleration state and adds no acceleration roughening.


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
def _park_miller_uniform(state: int) -> tuple[int, float]:
    next_state = (state * 48_271) % 2_147_483_647
    if next_state <= 0:
        next_state = 1
    return next_state, next_state / 2_147_483_647.0


@njit(cache=True, nogil=True)
def _draw_acceleration_state(
    source_state: int,
    uniform: float,
    transition: np.ndarray,
) -> int:
    first = transition[source_state, 0]
    second = first + transition[source_state, 1]
    if uniform < first:
        return 0
    if uniform < second:
        return 1
    return 2


@njit(cache=True, nogil=True)
def _pf_parent_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact exp404 x1.0 kernel, including base RNG consumption order."""

    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = (
                last_position + initial_spread * np.random.randn()
            )
            rate[particle] = (
                initial_rate + initial_rate_spread * np.random.randn()
            )
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = (
                    momentum * rate[particle] + rate_noise * np.random.randn()
                )
                position[particle] += (
                    rate[particle] * delta_md
                    + position_noise * np.random.randn()
                )
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
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = (
                        rate[cursor] + rough_rate * np.random.randn()
                    )
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles
                resampling_counts[seed_index] += 1
            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (
                    position[particle] - z_v[row]
                )
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
    )


@njit(cache=True, nogil=True)
def _pf_persistent_acceleration_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_position: float,
    initial_rate: float,
    particles: int,
    seeds: int,
    seed_base: int,
    acceleration_seeds: np.ndarray,
    acceleration_values: np.ndarray,
    acceleration_transition: np.ndarray,
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
]:
    """Exp404 kernel with one separate-RNG persistent acceleration coordinate."""

    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    acceleration_mass = np.zeros((seeds, rows, 3))
    filtered_rate_mean = np.zeros((seeds, rows))
    filtered_rate_std = np.zeros((seeds, rows))
    effective_sample_size_by_row = np.zeros((seeds, rows))
    state_counts = np.zeros((seeds, rows, 3), np.int64)
    acceleration_enabled = bool(
        acceleration_values[0] != 0.0
        or acceleration_values[1] != 0.0
        or acceleration_values[2] != 0.0
    )
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        acceleration_rng_state = int(acceleration_seeds[seed_index])
        position = np.empty(particles)
        rate = np.empty(particles)
        acceleration_state = np.ones(particles, np.int8)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = (
                last_position + initial_spread * np.random.randn()
            )
            rate[particle] = (
                initial_rate + initial_rate_spread * np.random.randn()
            )
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                acceleration_rng_state, draw = _park_miller_uniform(
                    acceleration_rng_state
                )
                next_state = _draw_acceleration_state(
                    int(acceleration_state[particle]),
                    draw,
                    acceleration_transition,
                )
                acceleration_state[particle] = next_state
                if acceleration_enabled:
                    rate[particle] = (
                        momentum * rate[particle]
                        + acceleration_values[next_state] * delta_md
                        + rate_noise * np.random.randn()
                    )
                else:
                    # Exact parent expression for the four parity sentinels.
                    rate[particle] = (
                        momentum * rate[particle]
                        + rate_noise * np.random.randn()
                    )
                position[particle] += (
                    rate[particle] * delta_md
                    + position_noise * np.random.randn()
                )
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
            rate_mean = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
                state = int(acceleration_state[particle])
                acceleration_mass[seed_index, row, state] += weights[particle]
                rate_mean += weights[particle] * rate[particle]
            rate_variance = 0.0
            for particle in range(particles):
                difference = rate[particle] - rate_mean
                rate_variance += weights[particle] * difference * difference
            filtered_rate_mean[seed_index, row] = rate_mean
            filtered_rate_std[seed_index, row] = np.sqrt(
                max(rate_variance, 0.0)
            )
            effective_sample_size = 1.0 / inverse_ess
            effective_sample_size_by_row[seed_index, row] = (
                effective_sample_size
            )
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
                new_acceleration_state = np.empty(particles, np.int8)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = (
                        rate[cursor] + rough_rate * np.random.randn()
                    )
                    new_acceleration_state[particle] = acceleration_state[cursor]
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    acceleration_state[particle] = new_acceleration_state[particle]
                    weights[particle] = 1.0 / particles
                resampling_counts[seed_index] += 1
            estimate = 0.0
            for particle in range(particles):
                state_counts[
                    seed_index,
                    row,
                    int(acceleration_state[particle]),
                ] += 1
                estimate += weights[particle] * (
                    position[particle] - z_v[row]
                )
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        acceleration_mass,
        filtered_rate_mean,
        filtered_rate_std,
        effective_sample_size_by_row,
        state_counts,
    )


def acceleration_seed_vector(
    split: str,
    well: str,
    seeds: int,
) -> np.ndarray:
    return np.asarray(
        [
            stable_seed("exp459", "acceleration", split, well, seed_index)
            for seed_index in range(seeds)
        ],
        dtype=np.int64,
    )


def aggregate_seed_predictions(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / float(temperature))
    weights /= float(weights.sum())
    return (weights[:, None] * predictions).sum(axis=0), weights


def run_persistent_acceleration_pf(
    prepared: Mapping[str, Any],
    *,
    well: str,
    split: str,
    particles: int,
    seeds: int,
    seed_base: int,
    acceleration_values: Sequence[float],
    acceleration_transition: np.ndarray,
    temperature: float,
    momentum: float = 0.998,
    rate_noise: float = 0.002,
    position_noise: float = 0.005,
    rough_position: float = 0.1,
    rough_rate: float = 0.001,
    resample_fraction: float = 0.5,
    initial_spread: float = 4.5,
    initial_rate_spread: float = 0.01,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    started = time.time()
    acceleration_seeds = acceleration_seed_vector(split, well, seeds)
    (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        acceleration_mass,
        filtered_rate_mean,
        filtered_rate_std,
        effective_sample_size_by_row,
        state_counts,
    ) = _pf_persistent_acceleration_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        int(particles),
        int(seeds),
        int(seed_base),
        acceleration_seeds,
        np.asarray(acceleration_values, dtype=np.float64),
        np.asarray(acceleration_transition, dtype=np.float64),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_spread),
        float(initial_rate_spread),
    )
    candidate, seed_weights = aggregate_seed_predictions(
        predictions,
        log_likelihoods,
        temperature=temperature,
    )
    row_mass = np.einsum(
        "s,srk->rk",
        seed_weights,
        acceleration_mass,
        optimize=True,
    )
    row_rate_mean = np.einsum(
        "s,sr->r",
        seed_weights,
        filtered_rate_mean,
        optimize=True,
    )
    row_second_moment = np.einsum(
        "s,sr->r",
        seed_weights,
        filtered_rate_std**2 + filtered_rate_mean**2,
        optimize=True,
    )
    row_rate_std = np.sqrt(
        np.maximum(row_second_moment - row_rate_mean**2, 0.0)
    )
    row_effective_sample_size = np.einsum(
        "s,sr->r",
        seed_weights,
        effective_sample_size_by_row,
        optimize=True,
    )
    count_sum = state_counts.sum(axis=0)
    values = np.asarray(acceleration_values, dtype=np.float64)
    ledger = pd.DataFrame(
        {
            "well_id": str(well),
            "row_idx": np.asarray(prepared["eval_indices"], dtype=np.int64),
            "suffix_offset": np.arange(len(candidate), dtype=np.int64),
            "acceleration_probability_negative": row_mass[:, 0],
            "acceleration_probability_zero": row_mass[:, 1],
            "acceleration_probability_positive": row_mass[:, 2],
            "filtered_acceleration_mean": row_mass @ values,
            "nonzero_acceleration_mass": row_mass[:, 0] + row_mass[:, 2],
            "particle_count_negative_all_seeds": count_sum[:, 0],
            "particle_count_zero_all_seeds": count_sum[:, 1],
            "particle_count_positive_all_seeds": count_sum[:, 2],
            "filtered_rate_mean": row_rate_mean,
            "filtered_rate_std": row_rate_std,
            "effective_sample_size": row_effective_sample_size,
        }
    )
    first_extinction: dict[str, int | None] = {}
    for index, name in enumerate(("negative", "zero", "positive")):
        extinct = np.flatnonzero(count_sum[:, index] == 0)
        first_extinction[name] = (
            int(prepared["eval_indices"][extinct[0]]) if len(extinct) else None
        )
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
        "position_clip_count_total": int(position_clip_counts.sum()),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
        "seed_weight_minimum": float(seed_weights.min()),
        "seed_weight_maximum": float(seed_weights.max()),
        "seed_weight_sum": float(seed_weights.sum()),
        "seed_aggregation_temperature": float(temperature),
        "mean_nonzero_acceleration_mass": float(
            ledger["nonzero_acceleration_mass"].mean()
        ),
        "first_acceleration_state_extinction_row": first_extinction,
        "base_seed": int(seed_base),
        "acceleration_seed_first": int(acceleration_seeds[0]),
        "acceleration_seed_last": int(acceleration_seeds[-1]),
        "acceleration_seed_vector_sha256": array_bundle_sha256(
            acceleration_seeds=acceleration_seeds
        ),
        "base_and_acceleration_seed_vectors_distinct": bool(
            not np.array_equal(
                acceleration_seeds,
                seed_base + np.arange(seeds, dtype=np.int64),
            )
        ),
    }
    return candidate, ledger, diagnostics


# %% [markdown]
# ## 7. Zero-acceleration exp404 parity and kinematic contracts


# %%
def synthetic_update_order_contract() -> dict[str, Any]:
    previous_tvt = 12_000.0
    previous_z = 1_500.0
    current_z = 1_501.25
    previous_rate = 0.03
    acceleration = 0.0005
    delta_md = 10.0
    rate_noise_draw = -0.25
    position_noise_draw = 0.75
    rate = (
        0.998 * previous_rate
        + acceleration * delta_md
        + 0.002 * rate_noise_draw
    )
    direct_tvt = (
        previous_tvt
        + rate * delta_md
        - (current_z - previous_z)
        + 0.005 * position_noise_draw
    )
    previous_u = previous_tvt + previous_z
    updated_u = previous_u + rate * delta_md + 0.005 * position_noise_draw
    via_u_tvt = updated_u - current_z
    return {
        "update_order": "acceleration_then_rate_then_tvt_then_gr_weight",
        "minus_delta_z_identity_max_abs_error": float(
            abs(direct_tvt - via_u_tvt)
        ),
        "pass": bool(direct_tvt == via_u_tvt),
    }


def acceleration_transition_contract(
    acceleration: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = acceleration_transition_matrix(acceleration)
    initial = np.asarray(
        get_nested(
            {"acceleration": acceleration},
            "acceleration.initial_probability",
            [0.0, 1.0, 0.0],
        ),
        dtype=np.float64,
    )
    if not np.array_equal(initial, np.asarray([0.0, 1.0, 0.0])):
        raise ValueError("exp459 initial acceleration prior changed")
    return {
        "acceleration_values": list(acceleration["values_rate_per_md_ft"]),
        "transition_matrix": matrix.tolist(),
        "transition_row_sum_max_error": float(
            np.max(np.abs(matrix.sum(axis=1) - 1.0))
        ),
        "boundary_negative_outward_mass_folded_to_stay": bool(
            matrix[0, 0] == 0.92 and matrix[0, 2] == 0.0
        ),
        "boundary_positive_outward_mass_folded_to_stay": bool(
            matrix[2, 2] == 0.92 and matrix[2, 0] == 0.0
        ),
        "initial_probability": initial.tolist(),
        "pass": True,
    }


def zero_acceleration_exp404_parity_contract(
    prepared: Mapping[str, Any],
    *,
    well: str,
    particles: int,
    seeds: int,
    seed_base: int,
    acceleration_transition: np.ndarray,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
    initial_rate_spread: float,
) -> dict[str, Any]:
    parent = _pf_parent_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
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
    )
    zero = _pf_persistent_acceleration_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        int(particles),
        int(seeds),
        int(seed_base),
        acceleration_seed_vector("train", well, seeds),
        np.zeros(3, dtype=np.float64),
        np.asarray(acceleration_transition, dtype=np.float64),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_spread),
        float(initial_rate_spread),
    )
    names = (
        "prediction",
        "log_likelihood",
        "resampling_count",
        "minimum_ess",
        "position_clip_count",
    )
    equality = {
        name: bool(np.array_equal(parent[index], zero[index]))
        for index, name in enumerate(names)
    }
    maximum_error = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(parent[index], dtype=np.float64)
                    - np.asarray(zero[index], dtype=np.float64)
                )
            )
        )
        for index, name in enumerate(names)
    }
    return {
        "well_id": str(well),
        "particles": int(particles),
        "seeds": int(seeds),
        "base_seed": int(seed_base),
        "array_equal": equality,
        "maximum_abs_error": maximum_error,
        "parent_bundle_sha256": array_bundle_sha256(
            prediction=parent[0],
            log_likelihood=parent[1],
            resampling_count=parent[2],
            minimum_ess=parent[3],
            position_clip_count=parent[4],
        ),
        "zero_acceleration_bundle_sha256": array_bundle_sha256(
            prediction=zero[0],
            log_likelihood=zero[1],
            resampling_count=zero[2],
            minimum_ess=zero[3],
            position_clip_count=zero[4],
        ),
        "pass": bool(all(equality.values())),
    }


def select_zero_acceleration_sentinel_wells(
    wells: Sequence[str],
    *,
    count: int = 4,
) -> list[str]:
    ordered = sorted(
        (str(well) for well in wells),
        key=lambda well: (
            hashlib.sha256(
                f"exp459::zero-sentinel::{well}".encode("utf-8")
            ).hexdigest(),
            well,
        ),
    )
    if len(ordered) < count:
        raise ValueError("not enough wells for exp459 zero-acceleration sentinel")
    return ordered[:count]


def warm_up_pf_kernel() -> None:
    md = np.arange(1.0, 4.0, dtype=np.float64)
    z = np.zeros(3, dtype=np.float64)
    gr = np.full(3, 50.0, dtype=np.float64)
    grid = np.linspace(40.0, 60.0, 101, dtype=np.float64)
    _pf_persistent_acceleration_allseeds(
        md,
        z,
        gr,
        grid,
        0.0,
        0.2,
        20.0,
        50.0,
        0.0,
        8,
        2,
        123,
        acceleration_seed_vector("train", "warmup", 2),
        np.asarray([-0.0005, 0.0, 0.0005], dtype=np.float64),
        np.asarray(
            [[0.92, 0.08, 0.0], [0.08, 0.84, 0.08], [0.0, 0.08, 0.92]],
            dtype=np.float64,
        ),
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
# ## 8. Target-free candidate generation and freeze


# %%
@dataclass
class FrozenWell:
    well_id: str
    prediction: pd.DataFrame
    acceleration_ledger: pd.DataFrame
    audit: dict[str, Any]
    prediction_logical_sha256: str
    acceleration_logical_sha256: str


def decode_target_free_well(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> FrozenWell:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    acceleration = dict(get_nested(config, "model.acceleration") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(fixed["typewell_grid_step_ft"]),
    )
    seed_base = stable_seed("likpf", "train", well)
    candidate, acceleration_ledger, diagnostics = run_persistent_acceleration_pf(
        prepared,
        well=well,
        split="train",
        particles=int(fixed["particles"]),
        seeds=int(fixed["seeds"]),
        seed_base=seed_base,
        acceleration_values=acceleration["values_rate_per_md_ft"],
        acceleration_transition=acceleration_transition_matrix(acceleration),
        temperature=float(fixed["primary_seed_weighting_temperature"]),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(fixed["rough_position"]),
        rough_rate=float(fixed["rough_rate"]),
        resample_fraction=float(fixed["resample_threshold_fraction"]),
        initial_spread=float(fixed["initial_state_spread_ft"]),
        initial_rate_spread=float(fixed["initial_rate_spread"]),
    )
    eval_indices = np.asarray(prepared["eval_indices"], dtype=np.int64)
    raw_observed = np.asarray(prepared["raw_gr_observed"], dtype=bool)
    prediction = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64((~raw_observed).mean()),
            PRIMARY_CANDIDATE: np.asarray(candidate, dtype=np.float32),
        }
    )
    if not np.isfinite(prediction[PRIMARY_CANDIDATE]).all():
        raise ValueError(f"{well}: exp459 prediction contains non-finite values")
    acceleration_ledger.insert(
        0,
        "id",
        prediction["id"].astype(str).to_numpy(),
    )
    prediction_sha = dataframe_content_sha(
        prediction,
        ["id", "well_id", "row_idx", PRIMARY_CANDIDATE],
    )
    acceleration_sha = dataframe_content_sha(
        acceleration_ledger,
        ACCELERATION_LOGICAL_COLUMNS,
    )
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "prefix_gr_missing_rows": int(
            prepared["scale_audit"]["known_gr_missing_rows"]
        ),
        "eval_rows": int(len(prediction)),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "eval_raw_gr_missing_fraction": float((~raw_observed).mean()),
        "last_known_tvt": float(prepared["last_known_tvt"]),
        "last_known_position": float(prepared["last_known_position"]),
        "initial_rate": float(prepared["initial_rate"]),
        "gr_scale_raw": float(prepared["scale_audit"]["raw_scale"]),
        "gr_scale_clipped": float(prepared["scale_audit"]["candidate_scale"]),
        "seed_base": int(seed_base),
        "seed_first": int(seed_base),
        "seed_last": int(seed_base + int(fixed["seeds"]) - 1),
        "seeds": int(fixed["seeds"]),
        "particles": int(fixed["particles"]),
        "seed_well_trajectories": int(fixed["seeds"]),
        "particle_starts": int(fixed["seeds"]) * int(fixed["particles"]),
        "prediction_logical_sha256": prediction_sha,
        "acceleration_ledger_logical_sha256": acceleration_sha,
        **diagnostics,
        "wall_seconds": time.time() - started,
    }
    return FrozenWell(
        well_id=str(well),
        prediction=prediction,
        acceleration_ledger=acceleration_ledger,
        audit=audit,
        prediction_logical_sha256=prediction_sha,
        acceleration_logical_sha256=acceleration_sha,
    )


def freeze_target_free_outputs(
    frozen_wells: Sequence[FrozenWell],
    output: Path,
    *,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions = (
        pd.concat([item.prediction for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    acceleration = (
        pd.concat(
            [item.acceleration_ledger for item in frozen_wells],
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
        or acceleration["id"].duplicated().any()
        or len(predictions) != len(acceleration)
        or predictions["well_id"].nunique() != ledger.expected_wells
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp459 target-free output coverage mismatch")
    prediction_path = output / f"{OUTPUT_PREFIX}_stage0_predictions.csv.gz"
    acceleration_path = (
        output / f"{OUTPUT_PREFIX}_stage0_acceleration_ledger.csv.gz"
    )
    audit_path = output / f"{OUTPUT_PREFIX}_stage0_well_audit.csv"
    prediction_artifact = write_deterministic_gzip_csv(
        predictions,
        prediction_path,
    )
    acceleration_artifact = write_deterministic_gzip_csv(
        acceleration,
        acceleration_path,
    )
    audit.to_csv(audit_path, index=False)
    for item in frozen_wells:
        ledger.freeze(item.well_id)
    if not ledger.all_frozen:
        raise RuntimeError("exp459 did not freeze all fixed32 wells")
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": int(len(predictions)),
        "wells": int(predictions["well_id"].nunique()),
        "prediction_logical_columns": [
            "id",
            "well_id",
            "row_idx",
            PRIMARY_CANDIDATE,
        ],
        "prediction_logical_sha256": dataframe_content_sha(
            predictions,
            ["id", "well_id", "row_idx", PRIMARY_CANDIDATE],
        ),
        "acceleration_logical_columns": list(ACCELERATION_LOGICAL_COLUMNS),
        "acceleration_logical_sha256": dataframe_content_sha(
            acceleration,
            ACCELERATION_LOGICAL_COLUMNS,
        ),
        "prediction_artifact": prediction_artifact,
        "acceleration_artifact": acceleration_artifact,
        "well_audit": {
            "path": str(audit_path),
            "raw_sha256": sha256_path(audit_path),
        },
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    return predictions, acceleration, audit, frozen


# %% [markdown]
# ## 9. Truth-late mechanism readout and fail-closed gates
#
# Direction agreement uses the sign of the evidence-weighted filtered
# acceleration mean at row `t` and the sign of the one-step-future true
# U-rate curvature `(rate[t+1]-rate[t]) / delta_MD[t+1]`. Exact zeros and the
# final suffix row are excluded. Persistent-episode SSE uses the frozen
# exp408 `[start_row_idx, end_row_idx_exclusive)` intervals.


# %%
def _require_frozen(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("exp459 late readout requires frozen predictions")
    for key in (
        "prediction_logical_sha256",
        "acceleration_logical_sha256",
    ):
        if len(str(frozen.get(key) or "")) != 64:
            raise RuntimeError(f"exp459 frozen output is missing {key}")


def load_suffix_truth(
    well: str,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(
        path,
        usecols=["MD", "Z", "TVT_input", "TVT"],
    )
    for column in ("MD", "Z", "TVT_input", "TVT"):
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    eval_mask = horizontal["TVT_input"].isna().to_numpy()
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    truth = horizontal["TVT"].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    md = horizontal["MD"].to_numpy(np.float64)
    u = truth + z
    rate = np.full(len(horizontal), np.nan, dtype=np.float64)
    for index in range(1, len(horizontal)):
        delta_md = max(float(md[index] - md[index - 1]), 1.0)
        rate[index] = (u[index] - u[index - 1]) / delta_md
    future_curvature = np.full(len(horizontal), np.nan, dtype=np.float64)
    for index in range(len(horizontal) - 1):
        delta_md = max(float(md[index + 1] - md[index]), 1.0)
        future_curvature[index] = (rate[index + 1] - rate[index]) / delta_md
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "true_tvt": truth[eval_indices],
            "true_u_rate": rate[eval_indices],
            "future_true_u_rate_curvature": future_curvature[eval_indices],
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
        raise ValueError("exp459 saved exp404 control raw SHA mismatch")
    if expected_decompressed and sha256_decompressed_csv(path) != expected_decompressed:
        raise ValueError("exp459 saved exp404 control decompressed SHA mismatch")
    return path


def load_saved_control_after_freeze(
    config: Mapping[str, Any],
    ids: set[str],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    spec = dict(get_nested(config, "data.saved_control") or {})
    control_column = str(spec["prediction_column"])
    frame = pd.read_csv(
        saved_control_path(config),
        usecols=["id", control_column],
        dtype={"id": str},
        compression="gzip",
    )
    frame = frame.loc[frame["id"].isin(ids)].copy()
    ledger.record_control(len(frame))
    if len(frame) != len(ids) or frame["id"].nunique() != len(ids):
        raise ValueError("exp459 saved exp404 control coverage mismatch")
    return frame.rename(columns={control_column: PRIMARY_CONTROL})


def persistent_episode_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.persistent_episodes") or {})
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("exp459 persistent episode SHA mismatch")
    return path


def load_persistent_episodes_after_freeze(
    config: Mapping[str, Any],
    wells: set[str],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(
        persistent_episode_path(config),
        usecols=[
            "episode_id",
            "well",
            "start_row_idx",
            "end_row_idx_exclusive",
        ],
        dtype={"episode_id": str, "well": str},
    )
    frame = frame.loc[frame["well"].isin(wells)].copy()
    ledger.record_episode(len(frame))
    for column in ("start_row_idx", "end_row_idx_exclusive"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
            np.int64
        )
    if frame.empty or frame["well"].nunique() != len(wells):
        raise ValueError("exp459 persistent episode coverage mismatch")
    return frame


def attach_truth_late_readout(
    predictions: pd.DataFrame,
    acceleration: pd.DataFrame,
    frozen: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_frozen(frozen)
    if dataframe_content_sha(
        predictions,
        ["id", "well_id", "row_idx", PRIMARY_CANDIDATE],
    ) != str(frozen["prediction_logical_sha256"]):
        raise ValueError("exp459 candidate changed after prediction freeze")
    if dataframe_content_sha(
        acceleration,
        ACCELERATION_LOGICAL_COLUMNS,
    ) != str(frozen["acceleration_logical_sha256"]):
        raise ValueError("exp459 acceleration ledger changed after freeze")
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
        acceleration,
        on=["id", "well_id", "row_idx", "suffix_offset"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
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
        or not np.isfinite(
            frame[[PRIMARY_CANDIDATE, PRIMARY_CONTROL, "true_tvt"]]
        ).all().all()
    ):
        raise ValueError("exp459 truth-late joined frame coverage mismatch")
    frame["candidate_squared_error"] = (
        frame[PRIMARY_CANDIDATE] - frame["true_tvt"]
    ) ** 2
    frame["control_squared_error"] = (
        frame[PRIMARY_CONTROL] - frame["true_tvt"]
    ) ** 2
    frame["candidate_abs_error"] = (
        frame[PRIMARY_CANDIDATE] - frame["true_tvt"]
    ).abs()
    frame["control_abs_error"] = (
        frame[PRIMARY_CONTROL] - frame["true_tvt"]
    ).abs()
    direction_mask = (
        np.isfinite(frame["future_true_u_rate_curvature"])
        & frame["future_true_u_rate_curvature"].ne(0.0)
        & frame["filtered_acceleration_mean"].ne(0.0)
    )
    frame["direction_evaluable"] = direction_mask
    frame["acceleration_future_curvature_sign_agreement"] = False
    frame.loc[
        direction_mask,
        "acceleration_future_curvature_sign_agreement",
    ] = (
        np.sign(
            frame.loc[direction_mask, "filtered_acceleration_mean"].to_numpy()
        )
        == np.sign(
            frame.loc[
                direction_mask,
                "future_true_u_rate_curvature",
            ].to_numpy()
        )
    )
    persistent_wells = set(
        roles.loc[roles["role"].eq("persistent"), "well"].astype(str)
    )
    episodes = load_persistent_episodes_after_freeze(
        config,
        persistent_wells,
        ledger,
    )
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        selected = frame.loc[
            frame["well_id"].eq(str(episode.well))
            & frame["row_idx"].ge(int(episode.start_row_idx))
            & frame["row_idx"].lt(int(episode.end_row_idx_exclusive))
        ]
        if selected.empty:
            continue
        candidate_sse = float(selected["candidate_squared_error"].sum())
        control_sse = float(selected["control_squared_error"].sum())
        episode_rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well_id": str(episode.well),
                "rows": int(len(selected)),
                "candidate_sse": candidate_sse,
                "control_sse": control_sse,
                "sse_reduction_fraction": (
                    (control_sse - candidate_sse) / control_sse
                    if control_sse > 0.0
                    else np.nan
                ),
            }
        )
    episode_metrics = pd.DataFrame(episode_rows)
    if episode_metrics.empty or episode_metrics["well_id"].nunique() != 16:
        raise ValueError("exp459 persistent episode truth-late readout incomplete")
    well_metrics = (
        frame.groupby(["well_id", "role", "fold"], as_index=False)
        .agg(
            rows=("id", "size"),
            candidate_sse=("candidate_squared_error", "sum"),
            control_sse=("control_squared_error", "sum"),
            mean_nonzero_acceleration_mass=("nonzero_acceleration_mass", "mean"),
            direction_rows=("direction_evaluable", "sum"),
            direction_agreement=(
                "acceleration_future_curvature_sign_agreement",
                "mean",
            ),
        )
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    well_metrics["candidate_rmse"] = np.sqrt(
        well_metrics["candidate_sse"] / well_metrics["rows"]
    )
    well_metrics["control_rmse"] = np.sqrt(
        well_metrics["control_sse"] / well_metrics["rows"]
    )
    well_metrics["delta_rmse"] = (
        well_metrics["candidate_rmse"] - well_metrics["control_rmse"]
    )
    return frame, well_metrics, episode_metrics


def _pooled_rmse(frame: pd.DataFrame, squared_error: str) -> float:
    return float(np.sqrt(frame[squared_error].sum() / len(frame)))


def evaluate_stage0_gates(
    frame: pd.DataFrame,
    well_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    audit: pd.DataFrame,
    parity: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    elapsed_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical_stage_0") or {})
    mechanism_config = dict(get_nested(config, "guards.mechanism_stage_0") or {})
    before = ledger.report()["before_freeze"]
    projected_full_seconds = elapsed_seconds / max(len(audit), 1) * 773.0
    seed_identity = bool(
        all(
            int(row.seed_base)
            == stable_seed("likpf", "train", str(row.well_id))
            for row in audit.itertuples(index=False)
        )
        and audit["base_and_acceleration_seed_vectors_distinct"].all()
    )
    transition_contract = acceleration_transition_contract(
        dict(get_nested(config, "model.acceleration") or {})
    )
    update_contract = synthetic_update_order_contract()
    technical_checks = {
        "state_transition_contract": bool(transition_contract["pass"]),
        "transition_row_sum": (
            float(transition_contract["transition_row_sum_max_error"])
            <= float(
                technical_config["maximum_acceleration_transition_row_sum_error"]
            )
        ),
        "update_order_and_minus_delta_z_identity": bool(update_contract["pass"]),
        "zero_acceleration_exp404_bitwise_parity": bool(
            len(parity) == 4 and all(bool(item["pass"]) for item in parity)
        ),
        "base_acceleration_rng_stream_separation": seed_identity,
        "finite_prediction_coverage": bool(
            np.isfinite(frame[PRIMARY_CANDIDATE]).mean()
            >= float(technical_config["require_finite_prediction_coverage"])
        ),
        "execution_count": bool(
            len(audit)
            == int(get_nested(config, "stages.stage_0.candidate_pf_well_runs"))
            and int(audit["seed_well_trajectories"].sum()) == 4096
            and int(audit["particle_starts"].sum()) == 2048000
        ),
        "truth_control_role_episode_reads_before_freeze_zero": bool(
            all(int(value) == 0 for value in before.values())
        ),
        "runtime_projection": bool(
            projected_full_seconds
            <= float(technical_config["maximum_seconds_full_projection"])
        ),
        "peak_rss": bool(
            rss_gb <= float(technical_config["maximum_peak_rss_gb"])
        ),
    }
    direction = frame.loc[frame["direction_evaluable"]].copy()
    overall_direction = float(
        direction["acceleration_future_curvature_sign_agreement"].mean()
    )
    direction_by_fold = (
        direction.groupby("fold")[
            "acceleration_future_curvature_sign_agreement"
        ]
        .mean()
        .reindex(range(5))
    )
    direction_positive_folds = int(
        (
            direction_by_fold
            >= float(
                mechanism_config[
                    "minimum_acceleration_future_rate_curvature_sign_agreement"
                ]
            )
        ).sum()
    )
    episode_candidate_sse = float(episode_metrics["candidate_sse"].sum())
    episode_control_sse = float(episode_metrics["control_sse"].sum())
    episode_reduction = (
        (episode_control_sse - episode_candidate_sse) / episode_control_sse
        if episode_control_sse > 0.0
        else float("nan")
    )
    persistent_well = (
        episode_metrics.groupby("well_id", as_index=False)[
            ["candidate_sse", "control_sse"]
        ]
        .sum()
        .merge(
            well_metrics[["well_id", "fold"]],
            on="well_id",
            how="left",
            validate="one_to_one",
        )
    )
    persistent_well["improved"] = (
        persistent_well["candidate_sse"] < persistent_well["control_sse"]
    )
    improved_persistent_wells = int(persistent_well["improved"].sum())
    persistent_fold = (
        persistent_well.groupby("fold")[["candidate_sse", "control_sse"]]
        .sum()
        .reindex(range(5))
    )
    improved_persistent_folds = int(
        (
            persistent_fold["candidate_sse"]
            < persistent_fold["control_sse"]
        ).sum()
    )
    controls = frame.loc[frame["role"].eq("control")].copy()
    control_candidate_rmse = _pooled_rmse(controls, "candidate_squared_error")
    control_saved_rmse = _pooled_rmse(controls, "control_squared_error")
    control_pooled_delta = control_candidate_rmse - control_saved_rmse
    control_delta_p95 = float(
        well_metrics.loc[
            well_metrics["role"].eq("control"),
            "delta_rmse",
        ].quantile(0.95)
    )
    mean_nonzero_mass = float(frame["nonzero_acceleration_mass"].mean())
    mechanism_checks = {
        "nonzero_acceleration_mass": bool(
            float(mechanism_config["minimum_mean_nonzero_acceleration_mass"])
            <= mean_nonzero_mass
            <= float(mechanism_config["maximum_mean_nonzero_acceleration_mass"])
        ),
        "future_rate_curvature_direction": bool(
            overall_direction
            >= float(
                mechanism_config[
                    "minimum_acceleration_future_rate_curvature_sign_agreement"
                ]
            )
        ),
        "future_rate_curvature_direction_folds": bool(
            direction_positive_folds
            >= int(
                mechanism_config["minimum_positive_reporting_folds_for_direction"]
            )
        ),
        "persistent_episode_sse": bool(
            episode_reduction
            >= float(
                mechanism_config[
                    "minimum_persistent_episode_sse_reduction_fraction"
                ]
            )
        ),
        "persistent_improved_wells": bool(
            improved_persistent_wells
            >= int(mechanism_config["minimum_improved_persistent_wells"])
        ),
        "persistent_improved_folds": bool(
            improved_persistent_folds
            >= int(
                mechanism_config["minimum_improved_persistent_reporting_folds"]
            )
        ),
        "matched_control_pooled_safety": bool(
            control_pooled_delta
            <= float(
                mechanism_config[
                    "maximum_matched_control_pooled_rmse_regression_ft"
                ]
            )
        ),
        "matched_control_by_well_p95_safety": bool(
            control_delta_p95
            <= float(
                mechanism_config[
                    "maximum_matched_control_by_well_delta_p95_ft"
                ]
            )
        ),
    }
    technical_all_pass = bool(all(technical_checks.values()))
    mechanism_all_pass = bool(all(mechanism_checks.values()))
    return {
        "stage": "stage0_fixed32_mechanism_preflight_not_cv",
        "technical_checks": technical_checks,
        "mechanism_checks": mechanism_checks,
        "technical_all_pass": technical_all_pass,
        "mechanism_all_pass": mechanism_all_pass,
        "all_pass": bool(technical_all_pass and mechanism_all_pass),
        "stage1_eligible_pending_separate_user_approval": bool(
            technical_all_pass and mechanism_all_pass
        ),
        "measurements": {
            "mean_nonzero_acceleration_mass": mean_nonzero_mass,
            "direction_agreement": overall_direction,
            "direction_agreement_by_fold": {
                str(index): to_jsonable(value)
                for index, value in direction_by_fold.items()
            },
            "direction_positive_folds": direction_positive_folds,
            "persistent_episode_candidate_sse": episode_candidate_sse,
            "persistent_episode_control_sse": episode_control_sse,
            "persistent_episode_sse_reduction_fraction": episode_reduction,
            "improved_persistent_wells": improved_persistent_wells,
            "improved_persistent_folds": improved_persistent_folds,
            "matched_control_candidate_rmse": control_candidate_rmse,
            "matched_control_saved_rmse": control_saved_rmse,
            "matched_control_pooled_delta_rmse": control_pooled_delta,
            "matched_control_by_well_delta_p95": control_delta_p95,
            "elapsed_seconds": elapsed_seconds,
            "projected_full_seconds": projected_full_seconds,
            "peak_rss_gb": rss_gb,
        },
        "transition_contract": transition_contract,
        "update_order_contract": update_contract,
        "zero_acceleration_parity": list(parity),
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 10. Generated artifacts and Stage 0 orchestration
#
# Standard Stage 0 outputs:
#
# - fixed32 candidate predictions;
# - target-free acceleration ledger;
# - well/runtime audit;
# - four-well zero-acceleration parity report;
# - scientific/input/freeze SHA manifests;
# - truth-late row, well, and persistent-episode readouts;
# - fail-closed technical/mechanism gate report and `metrics.json`.


# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL") == "1":
        return
    raise RuntimeError("exp459 Stage 0 must run first on Kaggle CPU")


def input_manifest(
    config: Mapping[str, Any],
    raw_dir: Path,
    wells: Sequence[str],
    scope_report: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for well in wells:
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        rows.append(
            {
                "well_id": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        "well_id",
        kind="mergesort",
    )
    return {
        "split": "train",
        "fixed32": dict(scope_report),
        "raw_dir": str(raw_dir),
        "wells": int(len(frame)),
        "raw_well_content_sha256": dataframe_content_sha(
            frame,
            ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
        ),
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
        "persistent_episodes": {
            "expected_raw_sha256": get_nested(
                config,
                "data.persistent_episodes.expected_sha256",
            ),
            "parsed_before_freeze": False,
        },
    }


def run_zero_acceleration_sentinels(
    wells: Sequence[str],
    raw_dir: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = select_zero_acceleration_sentinel_wells(wells)
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    acceleration = dict(get_nested(config, "model.acceleration") or {})
    transition = acceleration_transition_matrix(acceleration)
    reports: list[dict[str, Any]] = []
    for well in selected:
        horizontal = load_horizontal_without_truth(well, raw_dir)
        typewell = load_typewell(well, raw_dir)
        prepared = prepare_likelihood_pf_inputs(
            horizontal,
            typewell,
            grid_step=float(fixed["typewell_grid_step_ft"]),
        )
        reports.append(
            zero_acceleration_exp404_parity_contract(
                prepared,
                well=well,
                particles=int(fixed["particles"]),
                seeds=int(fixed["seeds"]),
                seed_base=stable_seed("likpf", "train", well),
                acceleration_transition=transition,
                momentum=float(fixed["momentum"]),
                rate_noise=float(fixed["rate_noise"]),
                position_noise=float(fixed["position_noise"]),
                rough_position=float(fixed["rough_position"]),
                rough_rate=float(fixed["rough_rate"]),
                resample_fraction=float(fixed["resample_threshold_fraction"]),
                initial_spread=float(fixed["initial_state_spread_ft"]),
                initial_rate_spread=float(fixed["initial_rate_spread"]),
            )
        )
    return reports


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    require_kaggle_runtime()
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    wells, scope_report = load_fixed32_scope(config)
    ledger = LeakageLedger(expected_wells=len(wells))
    scientific_contract = build_scientific_contract(config)
    scientific_contract_path = (
        output / f"{OUTPUT_PREFIX}_scientific_contract.json"
    )
    scientific_contract_artifact = write_json(
        scientific_contract_path,
        scientific_contract,
    )
    input_report = input_manifest(config, raw_dir, wells, scope_report)
    input_manifest_path = output / f"{OUTPUT_PREFIX}_stage0_input_manifest.json"
    input_manifest_artifact = write_json(input_manifest_path, input_report)

    warm_up_pf_kernel()
    frozen_wells = [
        decode_target_free_well(str(well), raw_dir, config) for well in wells
    ]
    parity = run_zero_acceleration_sentinels(wells, raw_dir, config)
    parity_path = (
        output / f"{OUTPUT_PREFIX}_stage0_zero_acceleration_parity.json"
    )
    parity_artifact = write_json(
        parity_path,
        {
            "wells": len(parity),
            "all_pass": bool(all(item["pass"] for item in parity)),
            "reports": parity,
        },
    )
    predictions, acceleration, audit, frozen = freeze_target_free_outputs(
        frozen_wells,
        output,
        ledger=ledger,
    )
    expected_stage0_rows = int(
        get_nested(config, "data.fixed32_manifest.expected_suffix_rows")
    )
    if len(predictions) != expected_stage0_rows:
        raise ValueError(
            "exp459 fixed32 suffix row count changed: "
            f"expected={expected_stage0_rows}, observed={len(predictions)}"
        )
    prefreeze_elapsed = time.time() - started
    candidate_elapsed = float(audit["wall_seconds"].sum())
    runtime_ledger = {
        "stage": "stage0_target_free_freeze",
        "candidate_wells": len(wells),
        "candidate_rows": len(predictions),
        "candidate_pf_well_runs": len(wells),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "zero_acceleration_sentinel_wells": len(parity),
        "saved_control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "candidate_elapsed_seconds": candidate_elapsed,
        "prefreeze_elapsed_seconds": prefreeze_elapsed,
        "projected_full_seconds": candidate_elapsed / len(wells) * 773.0,
        "peak_rss_gb": peak_rss_gb(),
        "versions": runtime_versions(),
        "truth_access_ledger": ledger.report(),
    }
    runtime_path = output / f"{OUTPUT_PREFIX}_stage0_runtime_ledger.json"
    runtime_artifact = write_json(runtime_path, runtime_ledger)
    frozen.update(
        {
            "scientific_contract_sha256": scientific_contract[
                "scientific_contract_sha256"
            ],
            "scientific_contract_file_sha256": scientific_contract_artifact[
                "raw_sha256"
            ],
            "input_manifest_sha256": input_manifest_artifact["raw_sha256"],
            "zero_acceleration_parity_sha256": parity_artifact["raw_sha256"],
            "runtime_ledger_sha256": runtime_artifact["raw_sha256"],
        }
    )
    freeze_manifest_path = (
        output / f"{OUTPUT_PREFIX}_stage0_freeze_manifest.json"
    )
    freeze_manifest_artifact = write_json(freeze_manifest_path, frozen)

    frame, well_metrics, episode_metrics = attach_truth_late_readout(
        predictions,
        acceleration,
        frozen,
        config=config,
        raw_dir=raw_dir,
        ledger=ledger,
    )
    elapsed = time.time() - started
    rss_gb = peak_rss_gb()
    gates = evaluate_stage0_gates(
        frame,
        well_metrics,
        episode_metrics,
        audit,
        parity,
        config=config,
        ledger=ledger,
        elapsed_seconds=candidate_elapsed,
        rss_gb=rss_gb,
    )
    truth_path = output / f"{OUTPUT_PREFIX}_stage0_truth_late_rows.csv.gz"
    well_path = output / f"{OUTPUT_PREFIX}_stage0_by_well.csv"
    episode_path = (
        output / f"{OUTPUT_PREFIX}_stage0_persistent_episode_metrics.csv"
    )
    gate_path = output / f"{OUTPUT_PREFIX}_stage0_gate_report.json"
    truth_artifact = write_deterministic_gzip_csv(frame, truth_path)
    well_metrics.to_csv(well_path, index=False)
    episode_metrics.to_csv(episode_path, index=False)
    gate_artifact = write_json(gate_path, gates)
    status = (
        "stage0_all_pass_pending_stage1_approval"
        if gates["all_pass"]
        else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage0_fixed32_mechanism_preflight_not_cv",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "counts": {
            "wells": len(wells),
            "rows": len(predictions),
            "scientific_variants": 1,
            "candidate_pf_well_runs": len(wells),
            "seed_well_trajectories": int(
                audit["seed_well_trajectories"].sum()
            ),
            "particle_starts": int(audit["particle_starts"].sum()),
            "zero_acceleration_sentinel_wells": len(parity),
            "saved_control_pf_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "frozen_outputs": frozen,
        "gates": gates,
        "runtime": {
            "candidate_seconds": candidate_elapsed,
            "prefreeze_seconds": prefreeze_elapsed,
            "total_seconds": elapsed,
            "peak_rss_gb": rss_gb,
            "versions": runtime_versions(),
        },
        "artifacts": {
            "scientific_contract": scientific_contract_artifact,
            "input_manifest": input_manifest_artifact,
            "zero_acceleration_parity": parity_artifact,
            "runtime_ledger": runtime_artifact,
            "freeze_manifest": freeze_manifest_artifact,
            "truth_late_rows": truth_artifact,
            "by_well": {
                "path": str(well_path),
                "raw_sha256": sha256_path(well_path),
            },
            "persistent_episode_metrics": {
                "path": str(episode_path),
                "raw_sha256": sha256_path(episode_path),
            },
            "gate_report": gate_artifact,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": (
            "request_separate_stage1_approval"
            if gates["all_pass"]
            else "close_branch_without_parameter_or_gate_rescue"
        ),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_stage0_summary.json"
    summary_artifact = write_json(summary_path, summary)
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. Setup and configuration preview


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
            "stage0_candidate_pf_well_runs": get_nested(
                CONFIG,
                "execution.stage_0_candidate_pf_well_runs",
            ),
            "seeds": get_nested(
                CONFIG,
                "model.fixed_from_exp404.seeds",
            ),
            "particles": get_nested(
                CONFIG,
                "model.fixed_from_exp404.particles",
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
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                "scientific_contract_sha256"
            ],
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_stage_0", False)):
        STAGE0_RESULT = run_stage0(CONFIG)
    else:
        print(
            "exp459 implementation is ready; Stage 0 execution remains "
            "disabled pending separate approval."
        )
