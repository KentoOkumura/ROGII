# %% [markdown]
# # exp450 dZ/dMD-conditioned TVT-rate likelihood-PF — Stage 0 train
#
# This compact self-contained notebook candidate implements the frozen exp450
# Stage 0 contract. Stage 0A checks that `(TVT, q)` with `beta=-1,
# intercept=0` preserves the saved-parent `(U, r_U)` temperature-5 output
# within `1e-6 ft`. Internal particle/resampling differences caused by
# floating-point operation order remain diagnostics. Only after the output
# parity gate passes, Stage 0B fits one visible-prefix affine center per
# fixed32 well and generates the scientific candidate. Suffix truth, fold,
# role, and episode fields stay closed until the candidate, prefix-fit ledger,
# diagnostics, and saved-control identity are frozen.

# %% [markdown]
# ## Contents
# 1. Imports and immutable identifiers
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Truth-free assets, raw inputs, and leakage ledger
# 5. Visible-prefix affine fit and target-free backtest
# 6. Exact exp404 input preparation
# 7. Parent/exact-transform paired parity kernel
# 8. Learned residual-AR likelihood-PF kernel
# 9. Stage 0A parity and Stage 0B prediction freeze
# 10. Truth-late mechanism readout and fail-closed gates
# 11. Generated artifacts and guarded orchestration
# 12. Setup and configuration preview
# 13. Run the selected Kaggle CPU stage

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
from joblib import Parallel, delayed

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


EXPERIMENT_NAME = "exp450_dzdmd_conditioned_tvt_rate_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
PRIMARY_CANDIDATE = "likpf_scale5_dzdmd_conditioned_tvt_rate"
EXACT_SENTINEL = "exact_beta_minus1_intercept0"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP450_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def write_deterministic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
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


def load_experiment_config(
    package_dir: Path | None = None,
) -> dict[str, Any]:
    candidates = [package_dir] if package_dir is not None else candidate_package_dirs()
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "config.yaml"
        checked.append(str(path))
        if not path.exists():
            continue
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp450 config not found; checked={checked}")


def artifacts_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_path() -> Path:
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
            if next(candidate.glob("*__horizontal_well.csv"), None):
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None):
                return candidate
    configured = Path(str(get_nested(config, "data.train_dir")))
    if not configured.is_absolute():
        configured = project_root() / configured
    if next(configured.glob("*__horizontal_well.csv"), None) is None:
        raise FileNotFoundError(f"raw train wells not found under {configured}")
    return configured


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame.loc[:, list(columns)] if columns else frame
    payload = selected.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def dataframe_typed_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected_columns = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in selected_columns:
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
    return mapping_sha256(
        [{"column": str(column), "dtype": str(frame[column].dtype)} for column in frame.columns]
    )


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def maximum_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        value *= 1024.0
    return value / (1024.0**3)


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba_available": NUMBA_AVAILABLE,
    }


def resolve_asset(
    filename: str,
    *,
    local: str | None = None,
    candidates: Iterable[str] = (),
) -> Path:
    paths: list[Path] = []
    if local:
        local_path = Path(local)
        paths.append(local_path if local_path.is_absolute() else project_root() / local_path)
    for candidate in candidates:
        base = Path(str(candidate))
        paths.extend([base, base / filename])
    paths.extend(
        path
        for root in candidate_package_dirs()
        for path in (root / "assets" / filename, root / filename)
    )
    if KAGGLE_INPUT_ROOT.exists():
        paths.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    matches = [path for path in unique if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"{filename} not found; checked={[str(path) for path in unique]}")
    return matches[0]


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent_endpoint": get_nested(config, "lineage.parent"),
        "implementation_reference": get_nested(config, "lineage.exact_pf_implementation_reference"),
        "primary_control": PRIMARY_CONTROL,
        "primary_candidate": PRIMARY_CANDIDATE,
        "active_scientific_variants": get_nested(config, "model.active_scientific_variants"),
        "prefix_affine_fit": get_nested(config, "model.prefix_affine_fit"),
        "transition": get_nested(config, "model.transition"),
        "initialization": get_nested(config, "model.initialization"),
        "fixed_pf": get_nested(config, "model.fixed_from_exp404"),
        "stages": get_nested(config, "stages"),
        "guards": get_nested(config, "guards"),
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
        "reproducibility": get_nested(config, "reproducibility"),
        "control_rerun": False,
        "model_count": 0,
        "booster_count": 0,
        "gpu_count": 0,
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
        "lineage.parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "lineage.exact_pf_implementation_reference": (
            "exp404_scale5_sigma_gr_likelihood_pf_ablation"
        ),
        "implementation.enabled": True,
        "implementation.implementation_approval_received": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "model.active_scientific_variants": ["learned_prefix_affine_residual_ar"],
        "model.prefix_affine_fit.minimum_valid_steps": 10,
        "model.prefix_affine_fit.fallback_beta": -1.0,
        "model.prefix_affine_fit.fallback_intercept": 0.0,
        "model.prefix_affine_fit.coefficient_clip": None,
        "model.prefix_affine_fit.coefficient_shrinkage": None,
        "model.prefix_affine_fit.regularization": None,
        "model.initialization.q_spread": 0.01,
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
        "stages.stage_0a.total_pf_well_runs": 24,
        "stages.stage_0a.seed_well_trajectories": 3072,
        "stages.stage_0a.particle_starts": 1536000,
        "stages.stage_0b.scientific_variants": 1,
        "stages.stage_0b.candidate_pf_well_runs": 32,
        "stages.stage_0b.seed_well_trajectories": 4096,
        "stages.stage_0b.particle_starts": 2048000,
        "stages.stage_0b.control_pf_well_runs": 0,
        "stages.stage_1.candidate_pf_well_runs": 773,
        "stages.stage_1.seed_well_trajectories": 98944,
        "stages.stage_1.particle_starts": 49472000,
        "guards.technical_stage_0a.parity_primary": "temperature5_aggregate_output",
        "guards.technical_stage_0a.maximum_temperature5_prediction_abs_diff": 1.0e-6,
        "guards.technical_stage_0a.internal_seed_weight_log_state_resampling_checks_are_diagnostic_only": (
            True
        ),
        "guards.technical_stage_0a.require_rate_position_clip_resampling_roughening_parity": (
            False
        ),
        "runtime.device": "cpu",
        "runtime.use_gpu": False,
        "execution.stage1_approved": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp450 fixed contract mismatch: {key} must be {value!r}")
    forbidden = {
        "beta_or_intercept_clip_shrink_regularization_or_grid",
        "prefix_window_minimum_support_or_holdout_search",
        "pf_z_rate_likelihood_or_smoothed_gr_mixture",
        "exact_beta_minus_one_as_scientific_candidate",
        "momentum_noise_roughening_particle_seed_temperature_or_gr_scale_change",
        "well_or_row_gate",
        "blend_or_selector_rescue",
        "same_fixed32_or_same_oof_rescue",
    }
    if set(get_nested(config, "guards.forbidden", [])) != forbidden:
        raise ValueError("exp450 forbidden rescue list changed")
    selected_stage = get_nested(config, "execution.selected_stage")
    if selected_stage not in (None, "stage_0ab"):
        raise ValueError("exp450 selected_stage must be null or 'stage_0ab'")
    if selected_stage == "stage_0ab":
        approvals = (
            get_nested(config, "implementation.train_execution_enabled"),
            get_nested(config, "execution.kaggle_package_approved"),
            get_nested(config, "execution.kaggle_push_approved"),
            get_nested(config, "execution.train_run_approved"),
            get_nested(config, "execution.run_stage_0a"),
            get_nested(config, "execution.run_stage_0b"),
        )
        if not all(bool(value) for value in approvals):
            raise RuntimeError("exp450 Kaggle Stage 0A/0B run is not approved")
    if require_run_approval:
        if selected_stage != "stage_0ab":
            raise RuntimeError("exp450 has no approved Stage 0A/0B selection")
    return scientific_contract(config)


# %% [markdown]
# ## 4. Truth-free assets, raw inputs, and leakage ledger


# %%
@dataclass
class LeakageLedger:
    prefix_fit_frozen: bool = False
    candidate_frozen: bool = False
    saved_control_frozen: bool = False
    suffix_truth_rows_before_freeze: int = 0
    fold_role_rows_before_freeze: int = 0
    episode_rows_before_freeze: int = 0
    suffix_truth_rows_after_freeze: int = 0
    fold_role_rows_after_freeze: int = 0
    episode_rows_after_freeze: int = 0

    def mark_prefix_fit_frozen(self) -> None:
        if self.suffix_truth_rows_before_freeze:
            raise RuntimeError("suffix truth was accessed before prefix freeze")
        self.prefix_fit_frozen = True

    def mark_candidate_frozen(self) -> None:
        if not self.prefix_fit_frozen:
            raise RuntimeError("candidate cannot freeze before prefix fit")
        if any(
            (
                self.suffix_truth_rows_before_freeze,
                self.fold_role_rows_before_freeze,
                self.episode_rows_before_freeze,
            )
        ):
            raise RuntimeError("late reporting fields were accessed before freeze")
        self.candidate_frozen = True

    def mark_saved_control_frozen(self) -> None:
        if not self.candidate_frozen:
            raise RuntimeError("saved control cannot freeze before candidate")
        self.saved_control_frozen = True

    def require_all_frozen(self) -> None:
        if not (self.prefix_fit_frozen and self.candidate_frozen and self.saved_control_frozen):
            raise RuntimeError("truth-late input requires all prediction freezes")

    def report(self) -> dict[str, Any]:
        return {
            "prefix_fit_frozen": self.prefix_fit_frozen,
            "candidate_frozen": self.candidate_frozen,
            "saved_control_frozen": self.saved_control_frozen,
            "before_freeze": {
                "suffix_truth_rows": self.suffix_truth_rows_before_freeze,
                "fold_role_rows": self.fold_role_rows_before_freeze,
                "episode_rows": self.episode_rows_before_freeze,
            },
            "after_freeze": {
                "suffix_truth_rows": self.suffix_truth_rows_after_freeze,
                "fold_role_rows": self.fold_role_rows_after_freeze,
                "episode_rows": self.episode_rows_after_freeze,
            },
        }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "Z"]].isna().any().any():
        raise ValueError(f"{well}: MD/Z must be finite")
    known = frame["TVT_input"].notna().to_numpy()
    if not known.any() or known.all():
        raise ValueError(f"{well}: visible prefix and unknown suffix are required")
    first_unknown = int(np.flatnonzero(~known)[0])
    if known[first_unknown:].any():
        raise ValueError(f"{well}: TVT_input must be one contiguous prefix")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"{well}: invalid Type Well support")
    mean_gr = float(frame["GR"].mean())
    if not math.isfinite(mean_gr):
        raise ValueError(f"{well}: invalid Type Well GR")
    frame["GR"] = frame["GR"].fillna(mean_gr)
    return frame


def load_scope_wells_truth_free(
    config: Mapping[str, Any],
    key: str,
) -> tuple[list[str], dict[str, Any]]:
    spec = get_nested(config, f"data.{key}")
    path = resolve_asset(
        str(spec["filename"]),
        local=spec.get("local"),
        candidates=spec.get("candidates", []),
    )
    observed = sha256_path(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"{key} SHA changed: {observed}")
    well_column = str(spec.get("well_column", "well"))
    frame = pd.read_csv(path, usecols=[well_column], dtype={well_column: str})
    wells = sorted(frame[well_column].astype(str).unique())
    if len(wells) != int(spec["expected_wells"]):
        raise ValueError(f"{key} well count changed")
    return wells, {
        "path": str(path),
        "raw_sha256": observed,
        "wells": len(wells),
        "columns_parsed": [well_column],
        "truth_fold_role_episode_fields_parsed": 0,
    }


def selected_raw_input_manifest(
    wells: Sequence[str],
    raw_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in sorted(str(value) for value in wells):
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        if not horizontal.exists() or not typewell.exists():
            raise FileNotFoundError(f"{well}: raw input pair is incomplete")
        rows.append(
            {
                "well": well,
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 5. Visible-prefix affine fit and target-free backtest


# %%
@dataclass(frozen=True)
class PrefixAffineFit:
    valid_step_count: int
    fallback_used: bool
    beta: float
    intercept: float
    previous_g: float
    fitted_residual_sse: float
    g_min: float
    g_max: float
    g_mean: float
    g_std: float
    q_min: float
    q_max: float
    q_mean: float
    q_std: float


def visible_prefix_step_arrays(
    horizontal: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    md = known["MD"].to_numpy(np.float64)
    z = known["Z"].to_numpy(np.float64)
    tvt = known["TVT_input"].to_numpy(np.float64)
    delta_md = np.diff(md)
    delta_z = np.diff(z)
    delta_tvt = np.diff(tvt)
    valid = (delta_md > 0.0) & np.isfinite(delta_md) & np.isfinite(delta_z) & np.isfinite(delta_tvt)
    return (
        (delta_z[valid] / delta_md[valid]).astype(np.float64),
        (delta_tvt[valid] / delta_md[valid]).astype(np.float64),
    )


def fit_affine_arrays(
    g: np.ndarray,
    q: np.ndarray,
    *,
    minimum_steps: int = 10,
) -> tuple[float, float, bool]:
    g_values = np.asarray(g, dtype=np.float64)
    q_values = np.asarray(q, dtype=np.float64)
    valid = np.isfinite(g_values) & np.isfinite(q_values)
    if int(valid.sum()) < minimum_steps:
        return -1.0, 0.0, True
    design = np.column_stack([g_values[valid], np.ones(int(valid.sum()), dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, q_values[valid], rcond=None)[0]
    beta, intercept = float(coefficients[0]), float(coefficients[1])
    if not math.isfinite(beta) or not math.isfinite(intercept):
        return -1.0, 0.0, True
    return beta, intercept, False


def fit_prefix_affine(
    horizontal: pd.DataFrame,
    *,
    minimum_steps: int = 10,
) -> PrefixAffineFit:
    g, q = visible_prefix_step_arrays(horizontal)
    beta, intercept, fallback = fit_affine_arrays(
        g,
        q,
        minimum_steps=minimum_steps,
    )
    fitted = beta * g + intercept
    residual_sse = float(np.sum((q - fitted) ** 2))
    previous_g = float(g[-1]) if len(g) else 0.0

    def summary(values: np.ndarray) -> tuple[float, float, float, float]:
        if not len(values):
            return math.nan, math.nan, math.nan, math.nan
        return (
            float(np.min(values)),
            float(np.max(values)),
            float(np.mean(values)),
            float(np.std(values)),
        )

    g_min, g_max, g_mean, g_std = summary(g)
    q_min, q_max, q_mean, q_std = summary(q)
    return PrefixAffineFit(
        valid_step_count=len(g),
        fallback_used=fallback,
        beta=beta,
        intercept=intercept,
        previous_g=previous_g,
        fitted_residual_sse=residual_sse,
        g_min=g_min,
        g_max=g_max,
        g_mean=g_mean,
        g_std=g_std,
        q_min=q_min,
        q_max=q_max,
        q_mean=q_mean,
        q_std=q_std,
    )


def prefix_tail_backtest(
    horizontal: pd.DataFrame,
    *,
    holdout_steps: int = 20,
    minimum_fit_steps: int = 10,
) -> dict[str, Any]:
    g, q = visible_prefix_step_arrays(horizontal)
    eligible = len(g) >= minimum_fit_steps + holdout_steps
    if not eligible:
        return {
            "eligible": False,
            "fit_steps": max(len(g) - holdout_steps, 0),
            "holdout_steps": min(len(g), holdout_steps),
            "candidate_sse": math.nan,
            "exact_sse": math.nan,
        }
    split = len(g) - holdout_steps
    beta, intercept, fallback = fit_affine_arrays(
        g[:split],
        q[:split],
        minimum_steps=minimum_fit_steps,
    )
    if fallback:
        raise RuntimeError("eligible prefix backtest unexpectedly fell back")
    candidate_error = q[split:] - (beta * g[split:] + intercept)
    exact_error = q[split:] + g[split:]
    return {
        "eligible": True,
        "fit_steps": split,
        "holdout_steps": holdout_steps,
        "candidate_beta": beta,
        "candidate_intercept": intercept,
        "candidate_sse": float(np.sum(candidate_error**2)),
        "exact_sse": float(np.sum(exact_error**2)),
    }


def exp404_initial_u_rate(
    horizontal: pd.DataFrame,
    *,
    tail_rows: int = 30,
) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()].tail(tail_rows)
    delta_tvt = np.diff(known["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(known["Z"].to_numpy(np.float64))
    delta_md = np.diff(known["MD"].to_numpy(np.float64))
    valid = (delta_md > 0.0) & np.isfinite(delta_md) & np.isfinite(delta_tvt) & np.isfinite(delta_z)
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))


# %% [markdown]
# ## 6. Exact exp404 input preparation


# %%
def uniform_typewell_grid(
    tvt: np.ndarray,
    gr: np.ndarray,
    *,
    step: float,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(tvt))
    maximum = float(np.max(tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    return (
        np.interp(grid_tvt, tvt, gr).astype(np.float64),
        minimum,
        float(step),
    )


def exp404_gr_scale(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    reference = np.interp(
        known_tvt,
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    residual = known_gr - reference
    raw = float(np.nanstd(residual))
    if not math.isfinite(raw):
        raise ValueError("known-prefix GR residual scale is not finite")
    return {
        "raw_scale": raw,
        "base_scale": float(np.clip(raw, 10.0, 60.0)),
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
    }


def prepare_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prefix_fit: PrefixAffineFit,
    *,
    grid_step: float = 0.2,
) -> dict[str, Any]:
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_indices = np.flatnonzero(~known_mask).astype(np.int64)
    known = horizontal.loc[known_mask]
    evaluation = horizontal.iloc[eval_indices]
    last = known.iloc[-1]
    last_md = float(last["MD"])
    last_z = float(last["Z"])
    last_tvt = float(last["TVT_input"])
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_z = evaluation["Z"].to_numpy(np.float64)
    previous_md = np.concatenate([[last_md], eval_md[:-1]])
    previous_z = np.concatenate([[last_z], eval_z[:-1]])
    delta_md = np.maximum(eval_md - previous_md, 1.0)
    g = (eval_z - previous_z) / delta_md
    mu = prefix_fit.beta * g + prefix_fit.intercept
    previous_mu = prefix_fit.beta * prefix_fit.previous_g + prefix_fit.intercept
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    typewell_mean = float(np.mean(typewell_gr))
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy(np.float64)
    )
    eval_gr = interpolated_gr[eval_indices]
    if not np.isfinite(eval_gr).all():
        raise ValueError("evaluation GR interpolation is not finite")
    initial_u_rate = exp404_initial_u_rate(horizontal)
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": eval_z,
        "eval_delta_md": delta_md,
        "eval_g": g,
        "eval_mu": mu,
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "last_known_md": last_md,
        "last_known_z": last_z,
        "last_known_tvt": last_tvt,
        "last_known_u": last_tvt + last_z,
        "initial_u_rate": initial_u_rate,
        "initial_q_rate": initial_u_rate - prefix_fit.previous_g,
        "previous_g": prefix_fit.previous_g,
        "previous_mu": previous_mu,
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "gr_scale": exp404_gr_scale(horizontal, typewell),
    }


# %% [markdown]
# ## 7. Parent/exact-transform paired parity kernel


# %%
@njit(cache=True)
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
def _paired_parent_exact_allseeds(
    delta_md_v: np.ndarray,
    z_v: np.ndarray,
    g_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_u: float,
    last_tvt: float,
    initial_u_rate: float,
    previous_g: float,
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    int,
    int,
]:
    rows = len(delta_md_v)
    parent_predictions = np.empty((seeds, rows))
    exact_predictions = np.empty((seeds, rows))
    parent_loglik = np.empty(seeds)
    exact_loglik = np.empty(seeds)
    maximum_weight_diff = 0.0
    maximum_position_coordinate_diff = 0.0
    maximum_rate_coordinate_diff = 0.0
    resampling_decision_mismatches = 0
    clip_decision_mismatches = 0
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        parent_u = np.empty(particles)
        exact_tvt = np.empty(particles)
        parent_rate = np.empty(particles)
        exact_q = np.empty(particles)
        parent_weights = np.ones(particles) / particles
        exact_weights = np.ones(particles) / particles
        for particle in range(particles):
            position_draw = initial_spread * np.random.randn()
            rate_draw = 0.01 * np.random.randn()
            parent_u[particle] = last_u + position_draw
            exact_tvt[particle] = last_tvt + position_draw
            parent_rate[particle] = initial_u_rate + rate_draw
            exact_q[particle] = initial_u_rate - previous_g + rate_draw
        parent_ll = 0.0
        exact_ll = 0.0
        previous_step_g = previous_g
        for row in range(rows):
            delta_md = delta_md_v[row]
            current_g = g_v[row]
            for particle in range(particles):
                rate_draw = rate_noise * np.random.randn()
                position_draw = position_noise * np.random.randn()
                parent_rate[particle] = momentum * parent_rate[particle] + rate_draw
                exact_q[particle] = (
                    -current_g + momentum * (exact_q[particle] + previous_step_g) + rate_draw
                )
                parent_u[particle] += parent_rate[particle] * delta_md + position_draw
                exact_tvt[particle] += exact_q[particle] * delta_md + position_draw
                parent_tvt = parent_u[particle] - z_v[row]
                parent_low = parent_tvt < grid_minimum - 100.0
                parent_high = parent_tvt > grid_maximum + 100.0
                exact_low = exact_tvt[particle] < grid_minimum - 100.0
                exact_high = exact_tvt[particle] > grid_maximum + 100.0
                if parent_low != exact_low or parent_high != exact_high:
                    clip_decision_mismatches += 1
                if parent_low:
                    parent_tvt = grid_minimum - 100.0
                if parent_high:
                    parent_tvt = grid_maximum + 100.0
                if exact_low:
                    exact_tvt[particle] = grid_minimum - 100.0
                if exact_high:
                    exact_tvt[particle] = grid_maximum + 100.0
                parent_u[particle] = parent_tvt + z_v[row]
                position_diff = abs(parent_tvt - exact_tvt[particle])
                rate_diff = abs(parent_rate[particle] - (exact_q[particle] + current_g))
                if position_diff > maximum_position_coordinate_diff:
                    maximum_position_coordinate_diff = position_diff
                if rate_diff > maximum_rate_coordinate_diff:
                    maximum_rate_coordinate_diff = rate_diff
            parent_average = 0.0
            exact_average = 0.0
            for particle in range(particles):
                parent_tvt = parent_u[particle] - z_v[row]
                parent_expected = _interp1(
                    grid_gr,
                    parent_tvt,
                    grid_minimum,
                    grid_step,
                )
                exact_expected = _interp1(
                    grid_gr,
                    exact_tvt[particle],
                    grid_minimum,
                    grid_step,
                )
                parent_zscore = (gr_v[row] - parent_expected) / gr_scale
                exact_zscore = (gr_v[row] - exact_expected) / gr_scale
                parent_squared = parent_zscore * parent_zscore
                exact_squared = exact_zscore * exact_zscore
                if parent_squared > 600.0:
                    parent_squared = 600.0
                if exact_squared > 600.0:
                    exact_squared = 600.0
                parent_likelihood = max(np.exp(-0.5 * parent_squared), 1e-300)
                exact_likelihood = max(np.exp(-0.5 * exact_squared), 1e-300)
                parent_average += parent_weights[particle] * parent_likelihood
                exact_average += exact_weights[particle] * exact_likelihood
                parent_weights[particle] *= parent_likelihood
                exact_weights[particle] *= exact_likelihood
            parent_average = max(parent_average, 1e-300)
            exact_average = max(exact_average, 1e-300)
            parent_ll += np.log(parent_average)
            exact_ll += np.log(exact_average)
            parent_sum = np.sum(parent_weights)
            exact_sum = np.sum(exact_weights)
            if parent_sum > 0.0:
                parent_weights /= parent_sum
            else:
                parent_weights[:] = 1.0 / particles
            if exact_sum > 0.0:
                exact_weights /= exact_sum
            else:
                exact_weights[:] = 1.0 / particles
            for particle in range(particles):
                weight_diff = abs(parent_weights[particle] - exact_weights[particle])
                if weight_diff > maximum_weight_diff:
                    maximum_weight_diff = weight_diff
            parent_ess = 1.0 / np.sum(parent_weights * parent_weights)
            exact_ess = 1.0 / np.sum(exact_weights * exact_weights)
            parent_resample = parent_ess < resample_fraction * particles
            exact_resample = exact_ess < resample_fraction * particles
            if parent_resample != exact_resample:
                resampling_decision_mismatches += 1
            if parent_resample or exact_resample:
                parent_cumulative = np.cumsum(parent_weights)
                exact_cumulative = np.cumsum(exact_weights)
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_parent_u = np.empty(particles)
                new_exact_tvt = np.empty(particles)
                new_parent_rate = np.empty(particles)
                new_exact_q = np.empty(particles)
                parent_cursor = 0
                exact_cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while (
                        parent_cursor < particles - 1 and parent_cumulative[parent_cursor] < uniform
                    ):
                        parent_cursor += 1
                    while exact_cursor < particles - 1 and exact_cumulative[exact_cursor] < uniform:
                        exact_cursor += 1
                    position_draw = rough_position * np.random.randn()
                    rate_draw = rough_rate * np.random.randn()
                    new_parent_u[particle] = parent_u[parent_cursor] + position_draw
                    new_exact_tvt[particle] = exact_tvt[exact_cursor] + position_draw
                    new_parent_rate[particle] = parent_rate[parent_cursor] + rate_draw
                    new_exact_q[particle] = exact_q[exact_cursor] + rate_draw
                parent_u = new_parent_u
                exact_tvt = new_exact_tvt
                parent_rate = new_parent_rate
                exact_q = new_exact_q
                parent_weights[:] = 1.0 / particles
                exact_weights[:] = 1.0 / particles
                for particle in range(particles):
                    position_diff = abs((parent_u[particle] - z_v[row]) - exact_tvt[particle])
                    rate_diff = abs(parent_rate[particle] - (exact_q[particle] + current_g))
                    if position_diff > maximum_position_coordinate_diff:
                        maximum_position_coordinate_diff = position_diff
                    if rate_diff > maximum_rate_coordinate_diff:
                        maximum_rate_coordinate_diff = rate_diff
            parent_estimate = 0.0
            exact_estimate = 0.0
            for particle in range(particles):
                parent_estimate += parent_weights[particle] * (parent_u[particle] - z_v[row])
                exact_estimate += exact_weights[particle] * exact_tvt[particle]
            parent_predictions[seed_index, row] = parent_estimate
            exact_predictions[seed_index, row] = exact_estimate
            previous_step_g = current_g
        parent_loglik[seed_index] = parent_ll
        exact_loglik[seed_index] = exact_ll
    return (
        parent_predictions,
        exact_predictions,
        parent_loglik,
        exact_loglik,
        maximum_weight_diff,
        maximum_position_coordinate_diff,
        maximum_rate_coordinate_diff,
        resampling_decision_mismatches,
        clip_decision_mismatches,
    )


def aggregate_temperature(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float = 5.0,
) -> np.ndarray:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / temperature)
    weights /= np.sum(weights)
    return (weights[:, None] * predictions).sum(axis=0)


def paired_coordinate_parity(
    prepared: Mapping[str, Any],
    *,
    particles: int,
    seeds: int,
    seed_base: int,
    temperature: float = 5.0,
    momentum: float = 0.998,
    rate_noise: float = 0.002,
    position_noise: float = 0.005,
    rough_position: float = 0.1,
    rough_rate: float = 0.001,
    resample_fraction: float = 0.5,
    initial_spread: float = 4.5,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    (
        parent_predictions,
        exact_predictions,
        parent_loglik,
        exact_loglik,
        weight_diff,
        position_diff,
        rate_diff,
        resample_mismatches,
        clip_mismatches,
    ) = _paired_parent_exact_allseeds(
        prepared["eval_delta_md"],
        prepared["eval_z"],
        prepared["eval_g"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["gr_scale"]["base_scale"]),
        float(prepared["last_known_u"]),
        float(prepared["last_known_tvt"]),
        float(prepared["initial_u_rate"]),
        float(prepared["previous_g"]),
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
    )
    parent_aggregate = aggregate_temperature(
        parent_predictions, parent_loglik, temperature=temperature
    )
    exact_aggregate = aggregate_temperature(
        exact_predictions, exact_loglik, temperature=temperature
    )
    report = {
        "maximum_seed_prediction_abs_diff": float(
            np.max(np.abs(parent_predictions - exact_predictions))
        ),
        "maximum_particle_weight_abs_diff": float(weight_diff),
        "maximum_log_likelihood_abs_diff": float(np.max(np.abs(parent_loglik - exact_loglik))),
        "maximum_temperature5_prediction_abs_diff": float(
            np.max(np.abs(parent_aggregate - exact_aggregate))
        ),
        "maximum_position_coordinate_abs_diff": float(position_diff),
        "maximum_rate_coordinate_abs_diff": float(rate_diff),
        "resampling_decision_mismatches": int(resample_mismatches),
        "clip_decision_mismatches": int(clip_mismatches),
        "finite": bool(
            np.isfinite(parent_predictions).all()
            and np.isfinite(exact_predictions).all()
            and np.isfinite(parent_loglik).all()
            and np.isfinite(exact_loglik).all()
        ),
    }
    return report, parent_aggregate, exact_aggregate


# %% [markdown]
# ## 8. Learned residual-AR likelihood-PF kernel


# %%
@njit(cache=True, nogil=True)
def _pf_residual_ar_allseeds(
    delta_md_v: np.ndarray,
    mu_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_tvt: float,
    initial_q: float,
    initial_rate_spread: float,
    previous_mu: float,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(delta_md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        tvt = np.empty(particles)
        q = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            tvt[particle] = last_tvt + initial_spread * np.random.randn()
            q[particle] = initial_q + initial_rate_spread * np.random.randn()
        log_likelihood = 0.0
        previous_step_mu = previous_mu
        for row in range(rows):
            for particle in range(particles):
                q[particle] = (
                    mu_v[row]
                    + momentum * (q[particle] - previous_step_mu)
                    + rate_noise * np.random.randn()
                )
                tvt[particle] += q[particle] * delta_md_v[row] + position_noise * np.random.randn()
                if tvt[particle] < grid_minimum - 100.0:
                    tvt[particle] = grid_minimum - 100.0
                    position_clip_counts[seed_index] += 1
                if tvt[particle] > grid_maximum + 100.0:
                    tvt[particle] = grid_maximum + 100.0
                    position_clip_counts[seed_index] += 1
            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(
                    grid_gr,
                    tvt[particle],
                    grid_minimum,
                    grid_step,
                )
                zscore = (gr_v[row] - expected_gr) / gr_scale
                squared = min(zscore * zscore, 600.0)
                likelihood = max(np.exp(-0.5 * squared), 1e-300)
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            average_likelihood = max(average_likelihood, 1e-300)
            log_likelihood += np.log(average_likelihood)
            weight_sum = np.sum(weights)
            if weight_sum > 0.0:
                weights /= weight_sum
            else:
                weights[:] = 1.0 / particles
            ess = 1.0 / np.sum(weights * weights)
            minimum_ess[seed_index] = min(minimum_ess[seed_index], ess)
            if ess < resample_fraction * particles:
                cumulative = np.cumsum(weights)
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_tvt = np.empty(particles)
                new_q = np.empty(particles)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_tvt[particle] = tvt[cursor] + rough_position * np.random.randn()
                    new_q[particle] = q[cursor] + rough_rate * np.random.randn()
                tvt = new_tvt
                q = new_q
                weights[:] = 1.0 / particles
                resampling_counts[seed_index] += 1
            predictions[seed_index, row] = np.sum(weights * tvt)
            previous_step_mu = mu_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
    )


def run_residual_ar_pf(
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed_base: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    fixed = get_nested(config, "model.fixed_from_exp404")
    started = time.time()
    (
        seed_predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        clip_counts,
    ) = _pf_residual_ar_allseeds(
        prepared["eval_delta_md"],
        prepared["eval_mu"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["gr_scale"]["base_scale"]),
        float(prepared["last_known_tvt"]),
        float(prepared["initial_q_rate"]),
        float(get_nested(config, "model.initialization.q_spread")),
        float(prepared["previous_mu"]),
        int(fixed["particles"]),
        int(fixed["seeds"]),
        int(seed_base),
        float(fixed["momentum"]),
        float(fixed["rate_noise"]),
        float(fixed["position_noise"]),
        float(fixed["rough_position"]),
        float(fixed["rough_rate"]),
        float(fixed["resample_threshold_fraction"]),
        float(get_nested(config, "model.initialization.position_spread_ft")),
    )
    prediction = aggregate_temperature(
        seed_predictions,
        log_likelihoods,
        temperature=float(fixed["primary_seed_weighting_temperature"]),
    )
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "seed_loglik_mean_per_row": float(np.mean(log_likelihoods)) / len(prediction),
        "seed_loglik_best_per_row": float(np.max(log_likelihoods)) / len(prediction),
        "seed_loglik_spread": float(np.std(log_likelihoods)),
        "resampling_count_total": int(np.sum(resampling_counts)),
        "resampling_count_min": int(np.min(resampling_counts)),
        "resampling_count_max": int(np.max(resampling_counts)),
        "minimum_ess_min": float(np.min(minimum_ess)),
        "minimum_ess_mean": float(np.mean(minimum_ess)),
        "position_clip_count_total": int(np.sum(clip_counts)),
        "seed_prediction_std_mean": float(np.mean(np.std(seed_predictions, axis=0))),
    }
    return prediction, diagnostics, seed_predictions, log_likelihoods


def finite_interval_rate(
    prediction: np.ndarray,
    delta_md: np.ndarray,
    *,
    last_tvt: float,
) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    return np.diff(np.concatenate([[last_tvt], values])) / np.asarray(delta_md, dtype=np.float64)


# %% [markdown]
# ## 9. Stage 0A parity and Stage 0B prediction freeze


# %%
def pf_arguments(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = get_nested(config, "model.fixed_from_exp404")
    return {
        "particles": int(fixed["particles"]),
        "seeds": int(fixed["seeds"]),
        "temperature": float(fixed["primary_seed_weighting_temperature"]),
        "momentum": float(fixed["momentum"]),
        "rate_noise": float(fixed["rate_noise"]),
        "position_noise": float(fixed["position_noise"]),
        "rough_position": float(fixed["rough_position"]),
        "rough_rate": float(fixed["rough_rate"]),
        "resample_fraction": float(fixed["resample_threshold_fraction"]),
        "initial_spread": float(get_nested(config, "model.initialization.position_spread_ft")),
    }


def decode_parity_well(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    exact_fit = PrefixAffineFit(
        valid_step_count=0,
        fallback_used=True,
        beta=-1.0,
        intercept=0.0,
        previous_g=(fit_prefix_affine(horizontal).previous_g),
        fitted_residual_sse=math.nan,
        g_min=math.nan,
        g_max=math.nan,
        g_mean=math.nan,
        g_std=math.nan,
        q_min=math.nan,
        q_max=math.nan,
        q_mean=math.nan,
        q_std=math.nan,
    )
    prepared = prepare_pf_inputs(horizontal, typewell, exact_fit)
    seed_base = stable_seed("likpf", "train", well)
    report, parent, exact = paired_coordinate_parity(
        prepared,
        seed_base=seed_base,
        **pf_arguments(config),
    )
    report.update(
        {
            "well": well,
            "seed_base": seed_base,
            "rows": len(parent),
            "parent_pf_well_runs": 1,
            "exact_transform_pf_well_runs": 1,
            "seed_well_trajectories": 2 * int(get_nested(config, "model.fixed_from_exp404.seeds")),
            "particle_starts": 2
            * int(get_nested(config, "model.fixed_from_exp404.seeds"))
            * int(get_nested(config, "model.fixed_from_exp404.particles")),
        }
    )
    frame = pd.DataFrame(
        {
            "well": well,
            "row_idx": prepared["eval_indices"],
            "parent_temperature5": parent,
            "exact_temperature5": exact,
        }
    )
    return report, frame


def evaluate_stage0a(
    reports: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guard = get_nested(config, "guards.technical_stage_0a")
    internal_threshold = float(guard["maximum_paired_seed_prediction_abs_diff"])
    checks = {
        "well_count": reports["well"].nunique()
        == int(get_nested(config, "data.stage_0a_sentinel12.expected_wells")),
        "parent_pf_well_runs": int(reports["parent_pf_well_runs"].sum())
        == int(get_nested(config, "stages.stage_0a.parent_pf_well_runs")),
        "exact_transform_pf_well_runs": int(reports["exact_transform_pf_well_runs"].sum())
        == int(get_nested(config, "stages.stage_0a.exact_transform_pf_well_runs")),
        "seed_well_trajectories": int(reports["seed_well_trajectories"].sum())
        == int(get_nested(config, "stages.stage_0a.seed_well_trajectories")),
        "particle_starts": int(reports["particle_starts"].sum())
        == int(get_nested(config, "stages.stage_0a.particle_starts")),
        "finite": bool(reports["finite"].all()),
        "temperature5_prediction_parity": float(
            reports["maximum_temperature5_prediction_abs_diff"].max()
        )
        <= float(guard["maximum_temperature5_prediction_abs_diff"]),
        "clip_decision_parity": int(reports["clip_decision_mismatches"].sum()) == 0,
        "prediction_readback_finite": bool(
            np.isfinite(
                predictions[["parent_temperature5", "exact_temperature5"]].to_numpy(np.float64)
            ).all()
        ),
    }
    diagnostic_checks = {
        "seed_prediction_parity": float(reports["maximum_seed_prediction_abs_diff"].max())
        <= internal_threshold,
        "particle_weight_parity": float(reports["maximum_particle_weight_abs_diff"].max())
        <= float(guard["maximum_paired_particle_weight_abs_diff"]),
        "position_coordinate_parity": float(reports["maximum_position_coordinate_abs_diff"].max())
        <= internal_threshold,
        "rate_coordinate_parity": float(reports["maximum_rate_coordinate_abs_diff"].max())
        <= internal_threshold,
        "log_likelihood_parity": float(reports["maximum_log_likelihood_abs_diff"].max())
        <= float(guard["maximum_paired_log_likelihood_abs_diff"]),
        "resampling_decision_parity": int(reports["resampling_decision_mismatches"].sum()) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostic_checks_not_used_for_gate": diagnostic_checks,
        "parity_primary": str(guard["parity_primary"]),
        "temperature5_tolerance_ft": float(guard["maximum_temperature5_prediction_abs_diff"]),
        "maximums": {
            column: float(reports[column].max())
            for column in (
                "maximum_seed_prediction_abs_diff",
                "maximum_particle_weight_abs_diff",
                "maximum_log_likelihood_abs_diff",
                "maximum_temperature5_prediction_abs_diff",
                "maximum_position_coordinate_abs_diff",
                "maximum_rate_coordinate_abs_diff",
            )
        },
    }


def decode_candidate_well(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    minimum_steps = int(get_nested(config, "model.prefix_affine_fit.minimum_valid_steps"))
    prefix_fit = fit_prefix_affine(
        horizontal,
        minimum_steps=minimum_steps,
    )
    backtest = prefix_tail_backtest(
        horizontal,
        holdout_steps=int(
            get_nested(
                config,
                "mechanism_readout.prefix_tail_backtest.holdout_valid_steps",
            )
        ),
        minimum_fit_steps=int(
            get_nested(
                config,
                "mechanism_readout.prefix_tail_backtest.minimum_fit_valid_steps",
            )
        ),
    )
    prepared = prepare_pf_inputs(horizontal, typewell, prefix_fit)
    seed_base = stable_seed("likpf", "train", well)
    prediction, diagnostics, _, _ = run_residual_ar_pf(
        prepared,
        config,
        seed_base=seed_base,
    )
    rate = finite_interval_rate(
        prediction,
        prepared["eval_delta_md"],
        last_tvt=prepared["last_known_tvt"],
    )
    prediction_frame = pd.DataFrame(
        {
            "well": well,
            "row_idx": prepared["eval_indices"],
            PRIMARY_CANDIDATE: prediction.astype(np.float64),
            "candidate_tvt_rate": rate,
            "delta_md": prepared["eval_delta_md"],
            "known_z_rate": prepared["eval_g"],
            "raw_gr_observed": prepared["raw_gr_observed"],
            "last_known_tvt": prepared["last_known_tvt"],
            "last_known_md": prepared["last_known_md"],
            "last_known_z": prepared["last_known_z"],
        }
    )
    prefix_row = {
        "well": well,
        **{field: getattr(prefix_fit, field) for field in PrefixAffineFit.__dataclass_fields__},
        **{f"backtest_{key}": value for key, value in backtest.items()},
    }
    audit_row = {
        "well": well,
        "seed_base": seed_base,
        "rows": len(prediction),
        "status": "ok",
        "pf_well_runs": 1,
        "seeds": int(get_nested(config, "model.fixed_from_exp404.seeds")),
        "seed_well_trajectories": int(get_nested(config, "model.fixed_from_exp404.seeds")),
        "particles": int(get_nested(config, "model.fixed_from_exp404.particles")),
        "particle_starts": int(get_nested(config, "model.fixed_from_exp404.seeds"))
        * int(get_nested(config, "model.fixed_from_exp404.particles")),
        "elapsed_seconds": time.time() - started,
        "prediction_content_sha256": dataframe_content_sha(
            prediction_frame,
            ["well", "row_idx", PRIMARY_CANDIDATE],
        ),
        **diagnostics,
    }
    return prediction_frame, prefix_row, audit_row


def freeze_stage0b_candidate(
    predictions: pd.DataFrame,
    prefix_ledger: pd.DataFrame,
    audit: pd.DataFrame,
    artifacts: Path,
    ledger: LeakageLedger,
) -> tuple[dict[str, Any], dict[str, Path]]:
    predictions = predictions.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    prefix_ledger = prefix_ledger.sort_values("well", kind="mergesort").reset_index(drop=True)
    audit = audit.sort_values("well", kind="mergesort").reset_index(drop=True)
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_stage0b_candidate_predictions.csv.gz"
    prefix_path = artifacts / f"{OUTPUT_PREFIX}_prefix_fit_ledger.csv.gz"
    audit_path = artifacts / f"{OUTPUT_PREFIX}_stage0b_pf_audit.csv"
    write_deterministic_gzip_csv(predictions, prediction_path)
    write_deterministic_gzip_csv(prefix_ledger, prefix_path)
    audit.to_csv(audit_path, index=False)
    ledger.mark_prefix_fit_frozen()
    frozen = {
        "rows": len(predictions),
        "wells": int(predictions["well"].nunique()),
        "prediction_logical_sha256": dataframe_content_sha(
            predictions,
            ["well", "row_idx", PRIMARY_CANDIDATE],
        ),
        "prediction_schema_sha256": dataframe_schema_sha(predictions),
        "prediction_raw_gzip_sha256": sha256_path(prediction_path),
        "prediction_decompressed_sha256": sha256_decompressed(prediction_path),
        "prefix_fit_logical_sha256": dataframe_content_sha(prefix_ledger),
        "prefix_fit_raw_gzip_sha256": sha256_path(prefix_path),
        "prefix_fit_decompressed_sha256": sha256_decompressed(prefix_path),
        "diagnostic_logical_sha256": dataframe_content_sha(audit),
        "diagnostic_raw_sha256": sha256_path(audit_path),
        "forbidden_late_columns_present": bool(
            {
                "TVT",
                "true_tvt",
                "error",
                "fold",
                "role",
                "episode_id",
            }
            & set(predictions.columns)
        ),
    }
    if frozen["forbidden_late_columns_present"]:
        raise RuntimeError("candidate freeze contains a late-readout column")
    ledger.mark_candidate_frozen()
    return frozen, {
        "candidate_predictions": prediction_path,
        "prefix_fit_ledger": prefix_path,
        "pf_audit": audit_path,
    }


def load_and_freeze_saved_control(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.saved_control")
    path = resolve_asset(
        str(spec["filename"]),
        candidates=spec.get("candidates", []),
    )
    raw_sha = sha256_path(path)
    decompressed_sha = sha256_decompressed(path)
    if raw_sha != str(spec["expected_raw_sha256"]):
        raise ValueError("saved exp404 control raw SHA changed")
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("saved exp404 control decompressed SHA changed")
    usecols = ["well_id", "row_idx", str(spec["prediction_column"])]
    control = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"well_id": str},
        compression="gzip",
    ).rename(
        columns={
            "well_id": "well",
            str(spec["prediction_column"]): PRIMARY_CONTROL,
        }
    )
    control["row_idx"] = pd.to_numeric(control["row_idx"], errors="raise").astype(np.int64)
    selected = control.merge(
        predictions[["well", "row_idx"]],
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != len(predictions):
        raise ValueError("saved exp404 control does not cover fixed32 rows")
    logical_sha = dataframe_content_sha(
        selected.sort_values(["well", "row_idx"], kind="mergesort"),
        ["well", "row_idx", PRIMARY_CONTROL],
    )
    expected_logical = str(spec["expected_logical_sha256"])
    source_logical = str(spec.get("expected_source_logical_sha256"))
    if source_logical and source_logical != "None":
        source_frame = pd.read_csv(
            path,
            compression="gzip",
            dtype={"id": str, "well_id": str},
        )
        source_columns = [
            "id",
            "well_id",
            "row_idx",
            "likpf_scale_5_x1p0",
            "likpf_scale_5_x1p3",
            "likpf_mean_x1p0",
            "likpf_mean_x1p3",
        ]
        if not set(source_columns).issubset(source_frame.columns):
            raise ValueError("saved exp404 source logical schema changed")
        source_frame["id"] = source_frame["id"].astype(object)
        source_frame["well_id"] = source_frame["well_id"].astype(object)
        source_frame["row_idx"] = pd.to_numeric(
            source_frame["row_idx"],
            errors="raise",
        ).astype(np.int64)
        for column in source_columns[3:]:
            source_frame[column] = pd.to_numeric(
                source_frame[column],
                errors="raise",
            ).astype(np.float64)
        observed_source_logical = dataframe_typed_content_sha(source_frame, source_columns)
        if observed_source_logical != source_logical:
            raise ValueError(
                "saved exp404 source logical SHA changed: "
                f"observed={observed_source_logical} expected={source_logical}"
            )
    if len(expected_logical) == 64 and logical_sha != expected_logical:
        raise ValueError("selected saved control logical SHA changed")
    ledger.mark_saved_control_frozen()
    return selected, {
        "path": str(path),
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "source_logical_sha256": source_logical,
        "selected_logical_sha256": logical_sha,
        "selected_rows": len(selected),
        "selected_wells": int(selected["well"].nunique()),
    }


# %% [markdown]
# ## 10. Truth-late mechanism readout and fail-closed gates


# %%
def load_fixed32_manifest_late(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    ledger.require_all_frozen()
    spec = get_nested(config, "data.stage_0b_fixed32")
    path = resolve_asset(
        str(spec["filename"]),
        local=spec.get("local"),
        candidates=spec.get("candidates", []),
    )
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("fixed32 manifest SHA changed")
    frame = pd.read_csv(path, dtype={"well": str})
    ledger.fold_role_rows_after_freeze += len(frame)
    if frame["well"].nunique() != int(spec["expected_wells"]):
        raise ValueError("fixed32 well count changed")
    if frame["role"].value_counts().to_dict() != {
        "persistent": 16,
        "control": 16,
    }:
        raise ValueError("fixed32 role counts changed")
    observed_folds = {
        str(int(key)): int(value)
        for key, value in frame["fold"].value_counts().sort_index().items()
    }
    if observed_folds != {
        str(key): int(value) for key, value in spec["expected_fold_counts"].items()
    }:
        raise ValueError("fixed32 fold counts changed")
    return frame


def load_truth_late(
    well: str,
    raw_dir: Path,
    row_indices: np.ndarray,
    ledger: LeakageLedger,
) -> np.ndarray:
    ledger.require_all_frozen()
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["TVT", "TVT_input"],
    )
    suffix_indices = np.flatnonzero(
        pd.to_numeric(frame["TVT_input"], errors="coerce").isna().to_numpy()
    )
    if not np.array_equal(
        suffix_indices.astype(np.int64),
        np.asarray(row_indices, dtype=np.int64),
    ):
        raise ValueError(f"{well}: suffix identity changed after freeze")
    truth = pd.to_numeric(frame.iloc[suffix_indices]["TVT"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(truth).all():
        raise ValueError(f"{well}: suffix truth is non-finite")
    ledger.suffix_truth_rows_after_freeze += len(truth)
    return truth


def attach_truth_late(
    predictions: pd.DataFrame,
    control: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> pd.DataFrame:
    frame = predictions.merge(
        control,
        on=["well", "row_idx"],
        how="left",
        validate="one_to_one",
    ).merge(
        manifest[["well", "role", "fold", "matched_persistent_well"]],
        on="well",
        how="left",
        validate="many_to_one",
    )
    if frame[[PRIMARY_CONTROL, "role", "fold"]].isna().any().any():
        raise ValueError("late control/manifest attachment is incomplete")
    pieces: list[pd.DataFrame] = []
    for well, part in frame.groupby("well", sort=True):
        selected = part.sort_values("row_idx", kind="mergesort").copy()
        selected["true_tvt"] = load_truth_late(
            str(well),
            raw_dir,
            selected["row_idx"].to_numpy(np.int64),
            ledger,
        )
        pieces.append(selected)
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def zero_directed_under_response_mask(
    true_rate: np.ndarray,
    decoded_rate: np.ndarray,
    *,
    moving_epsilon: float = 1e-12,
) -> np.ndarray:
    true_values = np.asarray(true_rate, dtype=np.float64)
    decoded_values = np.asarray(decoded_rate, dtype=np.float64)
    if true_values.shape != decoded_values.shape:
        raise ValueError("true and decoded rate shapes differ")
    return (
        (np.abs(true_values) > moving_epsilon)
        & (true_values * decoded_values >= 0.0)
        & (np.abs(decoded_values) < np.abs(true_values))
    )


def add_truth_late_rates(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, part in frame.groupby("well", sort=True):
        selected = part.sort_values("row_idx", kind="mergesort").copy()
        delta_md = selected["delta_md"].to_numpy(np.float64)
        last_tvt = float(selected["last_known_tvt"].iloc[0])
        selected["true_tvt_rate"] = finite_interval_rate(
            selected["true_tvt"].to_numpy(np.float64),
            delta_md,
            last_tvt=last_tvt,
        )
        selected["control_tvt_rate"] = finite_interval_rate(
            selected[PRIMARY_CONTROL].to_numpy(np.float64),
            delta_md,
            last_tvt=last_tvt,
        )
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def load_episode_readout_late(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    ledger.require_all_frozen()
    episode_spec = get_nested(config, "data.persistent_episodes")
    episode_path = resolve_asset(
        str(episode_spec["filename"]),
        local=episode_spec.get("local"),
        candidates=episode_spec.get("candidates", []),
    )
    if sha256_path(episode_path) != str(episode_spec["expected_sha256"]):
        raise ValueError("persistent episode SHA changed")
    persistent_wells = set(frame.loc[frame["role"].eq("persistent"), "well"].astype(str))
    episodes = pd.read_csv(
        episode_path,
        dtype={"well": str, "episode_id": str},
    )
    episodes = episodes.loc[episodes["well"].isin(persistent_wells)].copy()
    cause_spec = get_nested(config, "data.exp408_episode_causes")
    cause_path = resolve_asset(
        str(cause_spec["filename"]),
        local=cause_spec.get("local"),
        candidates=cause_spec.get("candidates", []),
    )
    if sha256_path(cause_path) != str(cause_spec["expected_sha256"]):
        raise ValueError("exp408 episode cause SHA changed")
    causes = pd.read_csv(
        cause_path,
        usecols=["episode_id", "well", "fold", "cause"],
        dtype={"well": str, "episode_id": str},
    )
    episodes = episodes.merge(
        causes,
        on=["episode_id", "well"],
        how="left",
        validate="one_to_one",
    )
    ledger.episode_rows_after_freeze += len(episodes)
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        selected = frame.loc[
            frame["well"].eq(str(episode.well))
            & frame["row_idx"].between(
                int(episode.start_row_idx),
                int(episode.end_row_idx_exclusive) - 1,
            )
        ]
        if len(selected) != int(episode.rows):
            raise ValueError(f"{episode.episode_id}: episode row coverage changed")
        truth = selected["true_tvt"].to_numpy(np.float64)
        parent_sse = float(np.sum((selected[PRIMARY_CONTROL].to_numpy(np.float64) - truth) ** 2))
        candidate_sse = float(
            np.sum((selected[PRIMARY_CANDIDATE].to_numpy(np.float64) - truth) ** 2)
        )
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well": str(episode.well),
                "fold": int(episode.fold),
                "cause": str(episode.cause),
                "rows": len(selected),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["fold", "well", "episode_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64))
                ** 2
            )
        )
    )


def sse_reduction(parent_sse: float, candidate_sse: float) -> float:
    return float((parent_sse - candidate_sse) / parent_sse) if parent_sse > 0.0 else math.nan


def evaluate_stage0b(
    frame: pd.DataFrame,
    prefix_ledger: pd.DataFrame,
    audit: pd.DataFrame,
    episodes: pd.DataFrame,
    runtime_seconds: float,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    guard = get_nested(config, "guards.mechanism_stage_0b")
    frame = add_truth_late_rates(frame)
    frame["control_squared_error"] = (frame[PRIMARY_CONTROL] - frame["true_tvt"]) ** 2
    frame["candidate_squared_error"] = (frame[PRIMARY_CANDIDATE] - frame["true_tvt"]) ** 2
    persistent = frame.loc[frame["role"].eq("persistent")].copy()
    persistent["control_under_response"] = zero_directed_under_response_mask(
        persistent["true_tvt_rate"].to_numpy(np.float64),
        persistent["control_tvt_rate"].to_numpy(np.float64),
    )
    persistent["candidate_under_response"] = zero_directed_under_response_mask(
        persistent["true_tvt_rate"].to_numpy(np.float64),
        persistent["candidate_tvt_rate"].to_numpy(np.float64),
    )
    control_share = float(
        persistent.loc[persistent["control_under_response"], "control_squared_error"].sum()
        / persistent["control_squared_error"].sum()
    )
    candidate_share = float(
        persistent.loc[
            persistent["candidate_under_response"],
            "candidate_squared_error",
        ].sum()
        / persistent["candidate_squared_error"].sum()
    )
    under_response_reduction = control_share - candidate_share
    forward = episodes.loc[
        episodes["cause"].eq(str(get_nested(config, "data.exp408_episode_causes.forward_cause")))
    ]
    forward_reduction = sse_reduction(
        float(forward["parent_sse"].sum()),
        float(forward["candidate_sse"].sum()),
    )
    persistent_episode_reduction = sse_reduction(
        float(episodes["parent_sse"].sum()),
        float(episodes["candidate_sse"].sum()),
    )
    well_rows: list[dict[str, Any]] = []
    for (well, role, fold), part in frame.groupby(["well", "role", "fold"], sort=True):
        truth = part["true_tvt"].to_numpy(np.float64)
        parent_rmse = rmse(truth, part[PRIMARY_CONTROL].to_numpy(np.float64))
        candidate_rmse = rmse(truth, part[PRIMARY_CANDIDATE].to_numpy(np.float64))
        well_rows.append(
            {
                "well": str(well),
                "role": str(role),
                "fold": int(fold),
                "rows": len(part),
                "parent_rmse": parent_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - parent_rmse,
            }
        )
    well_metrics = pd.DataFrame(well_rows)
    persistent_wells = well_metrics.loc[well_metrics["role"].eq("persistent")]
    control_wells = well_metrics.loc[well_metrics["role"].eq("control")]
    persistent_improved_wells = int(persistent_wells["delta_rmse"].le(0.0).sum())
    persistent_fold_rows: list[dict[str, Any]] = []
    for fold, part in persistent.groupby("fold", sort=True):
        truth = part["true_tvt"].to_numpy(np.float64)
        parent = rmse(truth, part[PRIMARY_CONTROL].to_numpy(np.float64))
        candidate = rmse(truth, part[PRIMARY_CANDIDATE].to_numpy(np.float64))
        persistent_fold_rows.append(
            {
                "fold": int(fold),
                "parent_rmse": parent,
                "candidate_rmse": candidate,
                "delta_rmse": candidate - parent,
            }
        )
    fold_metrics = pd.DataFrame(persistent_fold_rows)
    persistent_improving_folds = int(fold_metrics["delta_rmse"].le(0.0).sum())
    controls = frame.loc[frame["role"].eq("control")]
    control_truth = controls["true_tvt"].to_numpy(np.float64)
    control_pooled_delta = rmse(
        control_truth, controls[PRIMARY_CANDIDATE].to_numpy(np.float64)
    ) - rmse(control_truth, controls[PRIMARY_CONTROL].to_numpy(np.float64))
    control_p95 = float(np.quantile(control_wells["delta_rmse"], 0.95))
    eligible = prefix_ledger.loc[prefix_ledger["backtest_eligible"].astype(bool)].merge(
        well_metrics[["well", "fold"]].drop_duplicates(),
        on="well",
        how="left",
        validate="one_to_one",
    )
    candidate_backtest_sse = float(eligible["backtest_candidate_sse"].sum())
    exact_backtest_sse = float(eligible["backtest_exact_sse"].sum())
    pooled_backtest_ratio = (
        candidate_backtest_sse / exact_backtest_sse if exact_backtest_sse > 0.0 else math.inf
    )
    nonworse_backtest_folds = 0
    for _, part in eligible.groupby("fold"):
        if float(part["backtest_candidate_sse"].sum()) <= float(part["backtest_exact_sse"].sum()):
            nonworse_backtest_folds += 1
    projection = runtime_seconds / 32.0 * 773.0
    diagnostics = {
        "eligible_prefix_backtest_wells": len(eligible),
        "prefix_backtest_pooled_sse_ratio_vs_exact": pooled_backtest_ratio,
        "prefix_backtest_nonworse_folds": nonworse_backtest_folds,
        "parent_zero_directed_under_response_sse_share": control_share,
        "candidate_zero_directed_under_response_sse_share": candidate_share,
        "zero_directed_under_response_sse_share_reduction_absolute": (under_response_reduction),
        "forward_cause_episodes": len(forward),
        "forward_cause_episode_sse_reduction_fraction": forward_reduction,
        "persistent_episode_sse_reduction_fraction": (persistent_episode_reduction),
        "persistent_improved_wells": persistent_improved_wells,
        "persistent_improving_folds": persistent_improving_folds,
        "matched_control_pooled_rmse_delta_ft": control_pooled_delta,
        "matched_control_by_well_delta_p95_ft": control_p95,
        "full_runtime_projection_seconds": projection,
        "peak_rss_gb": maximum_rss_gb(),
    }
    checks = {
        "prefix_ols_and_fallback_contract": bool(
            np.isfinite(prefix_ledger[["beta", "intercept"]].to_numpy(np.float64)).all()
        ),
        "all_coefficients_and_predictions_finite": bool(
            np.isfinite(
                frame[[PRIMARY_CONTROL, PRIMARY_CANDIDATE, "candidate_tvt_rate"]].to_numpy(
                    np.float64
                )
            ).all()
        ),
        "fixed32_roles_and_fold_counts": (
            frame["well"].nunique() == 32
            and frame.loc[frame["role"].eq("persistent"), "well"].nunique() == 16
            and frame.loc[frame["role"].eq("control"), "well"].nunique() == 16
            and frame["fold"].nunique() == 5
        ),
        "execution_counts": (
            int(audit["pf_well_runs"].sum()) == 32
            and int(audit["seed_well_trajectories"].sum()) == 4096
            and int(audit["particle_starts"].sum()) == 2048000
        ),
        "truth_late": not any(ledger.report()["before_freeze"].values()),
        "prefix_backtest_pooled": pooled_backtest_ratio
        <= float(guard["maximum_prefix_backtest_pooled_sse_ratio_vs_exact"]),
        "prefix_backtest_folds": nonworse_backtest_folds
        >= int(guard["minimum_prefix_backtest_nonworse_folds"]),
        "zero_directed_under_response": under_response_reduction
        >= float(guard["minimum_zero_directed_under_response_sse_share_reduction_absolute"]),
        "forward_cause_episode": forward_reduction
        >= float(guard["minimum_forward_cause_episode_sse_reduction_fraction"]),
        "persistent_episode": persistent_episode_reduction
        >= float(guard["minimum_persistent_episode_sse_reduction_fraction"]),
        "persistent_improved_wells": persistent_improved_wells
        >= int(guard["minimum_persistent_improved_wells"]),
        "persistent_improving_folds": persistent_improving_folds
        >= int(guard["minimum_persistent_improving_folds"]),
        "matched_control_pooled": control_pooled_delta
        <= float(guard["maximum_matched_control_pooled_rmse_regression_ft"]),
        "matched_control_tail": control_p95
        <= float(guard["maximum_matched_control_by_well_delta_p95_ft"]),
        "runtime_projection": projection <= float(guard["maximum_seconds_full_projection"]),
        "peak_rss": maximum_rss_gb() <= float(guard["maximum_peak_rss_gb"]),
    }
    return (
        {"passed": all(checks.values()), "checks": checks, **diagnostics},
        well_metrics,
        fold_metrics,
    )


# %% [markdown]
# ## 11. Generated artifacts and guarded orchestration


# %%
def warm_up_kernels() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(1.0, 18.0),
            "Z": np.linspace(0.0, 1.6, 17),
            "GR": np.linspace(45.0, 55.0, 17),
            "TVT_input": [
                *np.linspace(100.0, 103.0, 12),
                *([math.nan] * 5),
            ],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 140.0, 301),
            "GR": np.linspace(40.0, 60.0, 301),
        }
    )
    fit = fit_prefix_affine(horizontal)
    prepared = prepare_pf_inputs(horizontal, typewell, fit)
    paired_coordinate_parity(
        prepared,
        particles=8,
        seeds=2,
        seed_base=1,
    )
    _pf_residual_ar_allseeds(
        prepared["eval_delta_md"],
        prepared["eval_mu"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["gr_scale"]["base_scale"]),
        float(prepared["last_known_tvt"]),
        float(prepared["initial_q_rate"]),
        0.01,
        float(prepared["previous_mu"]),
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
    )


def run_stage0ab(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp450 first execution must be on Kaggle; local execution needs "
            "explicit smoke approval"
        )
    contract = validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifacts_dir()
    raw_dir = train_data_dir(config)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, contract)
    warm_up_kernels()

    sentinel_wells, sentinel_input = load_scope_wells_truth_free(config, "stage_0a_sentinel12")
    sentinel_raw_manifest = selected_raw_input_manifest(sentinel_wells, raw_dir)
    sentinel_raw_manifest_path = artifacts / f"{OUTPUT_PREFIX}_stage0a_raw_input_manifest.csv"
    sentinel_raw_manifest.to_csv(sentinel_raw_manifest_path, index=False)
    sentinel_input["raw_input_manifest_sha256"] = sha256_path(sentinel_raw_manifest_path)
    sentinel_input["raw_input_logical_sha256"] = dataframe_content_sha(sentinel_raw_manifest)
    parity_results = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(delayed(decode_parity_well)(well, raw_dir, config) for well in sentinel_wells)
    parity_reports = pd.DataFrame([item[0] for item in parity_results])
    parity_predictions = pd.concat([item[1] for item in parity_results], ignore_index=True)
    stage0a_gate = evaluate_stage0a(parity_reports, parity_predictions, config)
    parity_report_path = artifacts / f"{OUTPUT_PREFIX}_stage0a_parity_report.csv"
    parity_prediction_path = artifacts / f"{OUTPUT_PREFIX}_stage0a_paired_predictions.csv.gz"
    stage0a_gate_path = artifacts / f"{OUTPUT_PREFIX}_stage0a_gate.json"
    write_deterministic_csv(parity_reports, parity_report_path)
    write_deterministic_gzip_csv(parity_predictions, parity_prediction_path)
    stage0a_gate["readback"] = {
        "report_sha256": sha256_path(parity_report_path),
        "prediction_raw_sha256": sha256_path(parity_prediction_path),
        "prediction_decompressed_sha256": sha256_decompressed(parity_prediction_path),
    }
    report_readback = pd.read_csv(
        parity_report_path,
        dtype={"well": str},
        float_precision="round_trip",
    )
    prediction_readback = pd.read_csv(
        parity_prediction_path,
        compression="gzip",
        dtype={"well": str},
        float_precision="round_trip",
    )
    stage0a_gate["checks"]["artifact_readback"] = dataframe_content_sha(
        report_readback
    ) == dataframe_content_sha(parity_reports) and dataframe_content_sha(
        prediction_readback
    ) == dataframe_content_sha(parity_predictions)
    stage0a_gate["passed"] = all(stage0a_gate["checks"].values())
    write_json(stage0a_gate_path, stage0a_gate)
    if not stage0a_gate["passed"]:
        summary = {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": "stage0a_exact_coordinate_parity_failed_closed",
            "stage0a": stage0a_gate,
            "stage0b_started": False,
            "sentinel_input": sentinel_input,
            "scientific_contract_sha256": contract["scientific_contract_sha256"],
            "models": 0,
            "boosters": 0,
            "gpu": 0,
        }
        write_json(artifacts / f"{OUTPUT_PREFIX}_summary.json", summary)
        write_json(metrics_path(), summary)
        return summary

    fixed32_wells, fixed32_input = load_scope_wells_truth_free(config, "stage_0b_fixed32")
    fixed32_raw_manifest = selected_raw_input_manifest(fixed32_wells, raw_dir)
    fixed32_raw_manifest_path = artifacts / f"{OUTPUT_PREFIX}_stage0b_raw_input_manifest.csv"
    fixed32_raw_manifest.to_csv(fixed32_raw_manifest_path, index=False)
    fixed32_input["raw_input_manifest_sha256"] = sha256_path(fixed32_raw_manifest_path)
    fixed32_input["raw_input_logical_sha256"] = dataframe_content_sha(fixed32_raw_manifest)
    stage0b_started = time.time()
    candidate_results = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(delayed(decode_candidate_well)(well, raw_dir, config) for well in fixed32_wells)
    predictions = pd.concat([item[0] for item in candidate_results], ignore_index=True)
    prefix_ledger = pd.DataFrame([item[1] for item in candidate_results])
    audit = pd.DataFrame([item[2] for item in candidate_results])
    ledger = LeakageLedger()
    frozen, frozen_paths = freeze_stage0b_candidate(
        predictions,
        prefix_ledger,
        audit,
        artifacts,
        ledger,
    )
    control, control_freeze = load_and_freeze_saved_control(
        predictions,
        config,
        ledger,
    )
    manifest = load_fixed32_manifest_late(config, ledger)
    attached = attach_truth_late(predictions, control, manifest, raw_dir, ledger)
    episodes = load_episode_readout_late(attached, config, ledger)
    stage0b_runtime = time.time() - stage0b_started
    stage0b_gate, well_metrics, fold_metrics = evaluate_stage0b(
        attached,
        prefix_ledger,
        audit,
        episodes,
        stage0b_runtime,
        config,
        ledger,
    )
    output_frames = {
        "truth_late_readout": (
            attached,
            artifacts / f"{OUTPUT_PREFIX}_stage0b_truth_late_readout.csv.gz",
        ),
        "episode_readout": (
            episodes,
            artifacts / f"{OUTPUT_PREFIX}_stage0b_episode_readout.csv",
        ),
        "well_metrics": (
            well_metrics,
            artifacts / f"{OUTPUT_PREFIX}_stage0b_well_metrics.csv",
        ),
        "fold_metrics": (
            fold_metrics,
            artifacts / f"{OUTPUT_PREFIX}_stage0b_fold_metrics.csv",
        ),
    }
    for _, (frame, path) in output_frames.items():
        if path.suffix == ".gz":
            write_deterministic_gzip_csv(frame, path)
        else:
            frame.to_csv(path, index=False)
    stage0b_gate_path = artifacts / f"{OUTPUT_PREFIX}_stage0b_gate.json"
    write_json(stage0b_gate_path, stage0b_gate)
    status = (
        "stage0ab_passed_stage1_not_authorized"
        if stage0b_gate["passed"]
        else "stage0b_mechanism_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "runtime_seconds": time.time() - started,
        "stage0a": stage0a_gate,
        "stage0b": stage0b_gate,
        "stage0a_input": sentinel_input,
        "stage0b_input": fixed32_input,
        "stage0b_frozen_candidate": frozen,
        "saved_control_freeze": control_freeze,
        "leakage_ledger": ledger.report(),
        "stage1_started": False,
        "stage1_requires_separate_user_approval": True,
        "scientific_variants": 1,
        "stage0a_pf_well_runs": int(
            parity_reports["parent_pf_well_runs"].sum()
            + parity_reports["exact_transform_pf_well_runs"].sum()
        ),
        "stage0b_candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "saved_control_pf_well_reruns": 0,
        "models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_versions": runtime_versions(),
        "kaggle_kernel_version": None,
        "model_sha256": None,
        "submission_sha256": None,
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "stage0a": stage0a_gate,
        "stage0b": stage0b_gate,
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "prefix_fit_ledger_sha256": frozen["prefix_fit_logical_sha256"],
        "diagnostic_sha256": frozen["diagnostic_logical_sha256"],
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Stage 0A/0B mechanism preflight only; fixed32 is not CV or "
            "promotion evidence. Stage 1, inference, and submission are not run."
        ),
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def selected_stage(config: Mapping[str, Any]) -> str | None:
    value = get_nested(config, "execution.selected_stage")
    if value is None:
        return None
    if str(value) != "stage_0ab":
        raise ValueError(f"unsupported exp450 execution stage: {value}")
    return str(value)


def run_selected_stage(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    return run_stage0ab(config)


# %% [markdown]
# ## 12. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "primary_control": PRIMARY_CONTROL,
                "primary_candidate": PRIMARY_CANDIDATE,
                "selected_stage": selected_stage(CONFIG),
                "stage0a_total_pf_well_runs": get_nested(
                    CONFIG, "stages.stage_0a.total_pf_well_runs"
                ),
                "stage0b_candidate_pf_well_runs": get_nested(
                    CONFIG, "stages.stage_0b.candidate_pf_well_runs"
                ),
                "stage0b_control_reruns": 0,
                "scientific_variants": 1,
                "models": 0,
                "boosters": 0,
                "gpu": 0,
                "stage1_approved": get_nested(CONFIG, "execution.stage1_approved"),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 13. Run the selected Kaggle CPU stage


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_selected_stage(CONFIG)
    if SUMMARY is None:
        print(
            "No execution stage selected. Implementation is ready, but "
            "Kaggle package/push/run remain disabled."
        )
