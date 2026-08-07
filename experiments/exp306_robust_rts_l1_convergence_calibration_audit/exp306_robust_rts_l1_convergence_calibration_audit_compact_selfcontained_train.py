# %% [markdown]
# # exp306 robust RTS / L1 convergence calibration audit train
#
# Target-free Stage 0 technical audit for the two exp304 solvers that did not
# converge within their original iteration budgets. No truth, separability
# score, prediction, or submission path exists in this notebook.

# %% [markdown]
# ## Contents
# 1. Imports and deterministic CPU runtime
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen technical contract and execution guards
# 4. Target-free raw input and fixed Stage 0 sample
# 5. Exp304-compatible common GR preparation
# 6. Robust RTS and second-order L1 solver kernels
# 7. Branch execution and technical gates
# 8. Eight-well deterministic parity audit
# 9. Stage 0 orchestration and generated artifacts
# 10. Setup and configuration preview
# 11. Run the separately approved Kaggle CPU Stage 0

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

EXPERIMENT_NAME = "exp306_robust_rts_l1_convergence_calibration_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
BRANCH_RTS_A = "rts_iter32_tol1e6"
BRANCH_RTS_B = "rts_iter32_tol1e4"
BRANCH_L1 = "l1_iter2000_rho1_tol1e4"
SERIES_KINDS = ("horizontal", "typewell")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP306_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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
    raise FileNotFoundError(f"exp306 config not found in {[str(path) for path in candidates]}")


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
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


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


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


# %% [markdown]
# ## 3. Frozen technical contract and execution guards


# %%
def _require_equal(config: Mapping[str, Any], dotted_key: str, expected: Any) -> None:
    actual = get_nested(config, dotted_key)
    if actual != expected:
        raise ValueError(
            f"exp306 contract mismatch: {dotted_key}={actual!r}, expected {expected!r}"
        )


def validate_technical_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> None:
    fixed_values = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp304_gr_denoiser_emission_separability_readout",
        "implementation.enabled": True,
        "implementation.scope": "stage0_only",
        "implementation.canonical_notebook_adopted": True,
        "implementation.full_audit_implemented": False,
        "implementation.scientific_score_implemented": False,
        "validation.strategy": "two_stage_target_free_solver_convergence_calibration_audit",
        "validation.metric": "solver_technical_coverage",
        "validation.n_folds": 0,
        "validation.score_rows": "no_truth_no_scientific_score",
        "validation.expected_wells": 773,
        "validation.expected_series_per_full_branch": 1546,
        "data.horizontal_allowed_columns": ["MD", "GR", "TVT_input"],
        "data.typewell_required_columns": ["TVT", "GR"],
        "model.lightgbm_config_count": 0,
        "model.fold_training_count": 0,
        "model.booster_count": 0,
        "model.parent_control_retraining": False,
        "model.robust_rts_fixed.measurement_df": 4.0,
        "model.robust_rts_fixed.finite_scale_floor": 1.0e-6,
        "model.robust_rts_stage0.candidate_a.name": BRANCH_RTS_A,
        "model.robust_rts_stage0.candidate_a.maximum_iterations": 32,
        "model.robust_rts_stage0.candidate_a.relative_mean_tolerance": 1.0e-6,
        "model.robust_rts_stage0.conditional_candidate_b.name": BRANCH_RTS_B,
        (
            "model.robust_rts_stage0.conditional_candidate_b."
            "enabled_only_if_candidate_a_has_any_stage0_failure"
        ): True,
        "model.robust_rts_stage0.conditional_candidate_b.maximum_iterations": 32,
        "model.robust_rts_stage0.conditional_candidate_b.relative_mean_tolerance": 1.0e-4,
        "model.robust_rts_stage0.additional_grid": False,
        "model.l1_trend.rho": 1.0,
        "model.l1_trend.maximum_iterations": 2000,
        "model.l1_trend.absolute_tolerance": 1.0e-4,
        "model.l1_trend.relative_tolerance": 1.0e-4,
        "model.l1_trend.adaptive_rho": False,
        "model.l1_trend.additional_grid": False,
        "audit.stage0.sample_wells": 64,
        "audit.stage0.sample_salt": "exp306-stage0-v1",
        "audit.stage0.series_kinds": ["horizontal", "typewell"],
        "audit.stage0.expected_series_per_candidate": 128,
        "audit.stage0.deterministic_parity_wells": 8,
        "audit.stage0.runtime_projection_limit_seconds": 30600,
        "runtime.num_workers": 1,
        "runtime.blas_threads": 1,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "execution.implementation_approved": True,
        "inference.enabled": False,
        "inference.create_submission": False,
        "inference.mode": "disabled_train_side_solver_audit_only",
    }
    for dotted_key, expected in fixed_values.items():
        _require_equal(config, dotted_key, expected)

    forbidden_flags = (
        "execution.run_full_rts",
        "execution.run_full_l1",
        "execution.run_scientific_score",
        "execution.run_inference",
        "execution.create_submission",
    )
    enabled_forbidden = [key for key in forbidden_flags if bool(get_nested(config, key))]
    if enabled_forbidden:
        raise RuntimeError(
            "exp306 Stage 0 implementation keeps full/scientific/inference paths fail-closed: "
            f"{enabled_forbidden}"
        )
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp306 Kaggle package/push/run is not approved")
        if not bool(get_nested(config, "execution.run_stage0")):
            raise RuntimeError("exp306 Stage 0 run flag is not enabled")


def build_technical_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_technical_contract(config)
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "truth_or_scientific_score_loaded": False,
        "raw_well_identity_expected_sha256": get_nested(
            config, "data.expected_raw_well_identity_sha256"
        ),
        "exp304_scientific_contract_anchor_sha256": get_nested(
            config, "data.expected_exp304_scientific_contract_sha256"
        ),
        "stage0_sample": {
            "salt": get_nested(config, "audit.stage0.sample_salt"),
            "wells": get_nested(config, "audit.stage0.sample_wells"),
            "series_kinds": list(SERIES_KINDS),
            "parity_wells": get_nested(config, "audit.stage0.deterministic_parity_wells"),
        },
        "branches": {
            BRANCH_RTS_A: {
                "maximum_iterations": 32,
                "relative_mean_tolerance": 1.0e-6,
            },
            BRANCH_RTS_B: {
                "conditional_on_any_rts_a_technical_failure": True,
                "maximum_iterations": 32,
                "relative_mean_tolerance": 1.0e-4,
            },
            BRANCH_L1: {
                "maximum_iterations": 2000,
                "rho": 1.0,
                "absolute_tolerance": 1.0e-4,
                "relative_tolerance": 1.0e-4,
            },
        },
        "forbidden": [
            "horizontal_TVT",
            "truth",
            "error",
            "formation",
            "MRR",
            "top3",
            "RMSE",
            "prediction",
            "submission",
            "solver_grid",
        ],
        "runtime_contract": {
            "num_workers": 1,
            "blas_threads": 1,
            "gpu": False,
            "internet": False,
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Target-free raw input and fixed Stage 0 sample


# %%
def validate_horizontal_target_free_frame(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    allowed = [str(value) for value in get_nested(config, "data.horizontal_allowed_columns")]
    forbidden = {
        str(value).casefold()
        for value in get_nested(config, "data.horizontal_forbidden_columns")
    }
    forbidden.update(
        {"truth", "tvt_true", "target", "mrr", "top3", "rmse", "score", "prediction"}
    )
    actual_casefold = {str(column).casefold() for column in frame.columns}
    leaked = sorted(actual_casefold & forbidden)
    if leaked:
        raise ValueError(f"target-free horizontal frame exposes forbidden columns: {leaked}")
    if list(frame.columns) != allowed:
        raise ValueError(
            f"target-free horizontal frame must expose exactly {allowed}, got {list(frame.columns)}"
        )


def validate_typewell_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    required = [str(value) for value in get_nested(config, "data.typewell_required_columns")]
    if list(frame.columns) != required:
        raise ValueError(
            f"typewell frame must expose exactly {required}, got {list(frame.columns)}"
        )


def load_horizontal_target_free(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    allowed = [str(value) for value in get_nested(config, "data.horizontal_allowed_columns")]
    frame = pd.read_csv(path, usecols=allowed)
    frame = frame[allowed]
    validate_horizontal_target_free_frame(frame, config)
    return frame


def load_typewell_target_free(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    required = [str(value) for value in get_nested(config, "data.typewell_required_columns")]
    frame = pd.read_csv(path, usecols=required)
    frame = frame[required]
    validate_typewell_frame(frame, config)
    return frame


def enumerate_paired_wells(raw_dir: Path) -> list[str]:
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv")
        for path in raw_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv")
        for path in raw_dir.glob("*__typewell.csv")
    }
    if horizontal != typewell:
        raise ValueError(
            "horizontal/typewell identities differ: "
            f"missing_typewell={sorted(horizontal - typewell)[:5]}, "
            f"missing_horizontal={sorted(typewell - horizontal)[:5]}"
        )
    return sorted(horizontal)


def stable_stage0_sample(
    well_ids: Iterable[str],
    *,
    salt: str,
    sample_wells: int,
) -> pd.DataFrame:
    unique = sorted({str(well_id) for well_id in well_ids})
    if len(unique) < sample_wells:
        raise ValueError(f"cannot sample {sample_wells} wells from {len(unique)}")
    rows = []
    for well_id in unique:
        sample_sha = hashlib.sha256(f"{salt}|{well_id}".encode()).hexdigest()
        rows.append({"well_id": well_id, "sample_sha256": sample_sha})
    sample = (
        pd.DataFrame(rows)
        .sort_values(["sample_sha256", "well_id"], kind="mergesort")
        .head(sample_wells)
        .reset_index(drop=True)
    )
    sample.insert(0, "sample_rank", np.arange(1, len(sample) + 1, dtype=np.int64))
    return sample


def raw_well_identity_manifest(raw_dir: Path, well_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for well_id in sorted(str(value) for value in well_ids):
        horizontal_path = raw_dir / f"{well_id}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well_id}__typewell.csv"
        if not horizontal_path.exists() or not typewell_path.exists():
            raise FileNotFoundError(f"missing paired files for {well_id}")
        rows.append(
            {
                "well_id": well_id,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    return pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 5. Exp304-compatible common GR preparation


# %%
def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_horizontal_target_free_frame(horizontal_without_truth, config)
    validate_typewell_frame(typewell, config)

    horizontal = horizontal_without_truth.copy()
    horizontal["MD"] = pd.to_numeric(horizontal["MD"], errors="raise")
    horizontal["TVT_input"] = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    horizontal_gr_observed = pd.to_numeric(horizontal["GR"], errors="coerce")
    horizontal_original_missing = horizontal_gr_observed.isna().to_numpy(bool)
    horizontal_md = horizontal["MD"].to_numpy(np.float64)
    if len(horizontal) < 2 or not np.isfinite(horizontal_md).all():
        raise ValueError("horizontal MD must contain at least two finite rows")
    if bool((np.diff(horizontal_md) < 0.0).any()):
        raise ValueError("horizontal rows must already be in non-decreasing MD order")
    visible_tvt_input = horizontal["TVT_input"].dropna().to_numpy(np.float64)
    if not np.isfinite(visible_tvt_input).all():
        raise ValueError("visible TVT_input rows must be finite")

    tw = typewell.copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    typewell_original_missing = tw["GR"].isna().to_numpy(bool)
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy(np.float64)).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    if bool((np.diff(typewell_tvt) < 0.0).any()):
        raise ValueError("typewell TVT must be non-decreasing after stable sort")

    gr_fill = float(np.nanmean(typewell_gr))
    horizontal_gr = (
        horizontal_gr_observed.interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    if not np.isfinite(horizontal_gr).all():
        raise ValueError("common horizontal GR interpolation must be finite")
    return {
        "horizontal_coordinate": horizontal_md,
        "horizontal_gr": horizontal_gr,
        "horizontal_original_missing": horizontal_original_missing,
        "typewell_coordinate": typewell_tvt,
        "typewell_gr": typewell_gr,
        "typewell_original_missing": typewell_original_missing,
    }


def load_prepared_stage0(
    raw_dir: Path,
    sample: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prepared_by_well: dict[str, dict[str, Any]] = {}
    for well_id in sample["well_id"].astype(str):
        horizontal = load_horizontal_target_free(
            raw_dir / f"{well_id}__horizontal_well.csv", config
        )
        typewell = load_typewell_target_free(
            raw_dir / f"{well_id}__typewell.csv", config
        )
        prepared_by_well[well_id] = prepare_gr_inputs(horizontal, typewell, config)
    return prepared_by_well


def build_input_frame(prepared_by_well: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well_id in sorted(prepared_by_well):
        prepared = prepared_by_well[well_id]
        for series_kind in SERIES_KINDS:
            coordinate = np.asarray(prepared[f"{series_kind}_coordinate"], dtype=np.float64)
            values = np.asarray(prepared[f"{series_kind}_gr"], dtype=np.float64)
            parts.append(
                pd.DataFrame(
                    {
                        "well_id": well_id,
                        "series_kind": series_kind,
                        "position": np.arange(len(values), dtype=np.int64),
                        "coordinate": coordinate,
                        "input_gr": values,
                        "original_missing": np.asarray(
                            prepared[f"{series_kind}_original_missing"], dtype=bool
                        ),
                    }
                )
            )
    if not parts:
        raise ValueError("Stage 0 has no prepared input series")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 6. Robust RTS and second-order L1 solver kernels


# %%
def robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0
    center = float(np.median(finite))
    return float(np.median(np.abs(finite - center)) / 0.67448975)


def normalized_coordinate(coordinate: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("denoiser coordinate must have at least two finite values")
    positive = np.diff(values)
    positive = positive[positive > 0.0]
    if not len(positive):
        raise ValueError("denoiser coordinate needs a positive spacing")
    spacing = float(np.median(positive))
    return (values - values[0]) / spacing, spacing


def second_divided_differences(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    u = np.asarray(coordinate, dtype=np.float64)
    left = np.diff(u[:-1])
    right = np.diff(u[1:])
    valid = (left > 0.0) & (right > 0.0) & ((u[2:] - u[:-2]) > 0.0)
    if not bool(valid.any()):
        return np.empty(0, dtype=np.float64)
    first_left = np.diff(y[:-1])[valid] / left[valid]
    first_right = np.diff(y[1:])[valid] / right[valid]
    span = (u[2:] - u[:-2])[valid]
    return 2.0 * (first_right - first_left) / span


def _student_t_rts_pass(
    observations: np.ndarray,
    coordinate: np.ndarray,
    *,
    measurement_variance: float,
    acceleration_variance: float,
    weights: np.ndarray,
    initial_slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observations, dtype=np.float64)
    u = np.asarray(coordinate, dtype=np.float64)
    n = len(y)
    filtered_mean = np.empty((n, 2), dtype=np.float64)
    predicted_mean = np.empty((n, 2), dtype=np.float64)
    filtered_cov = np.empty((n, 2, 2), dtype=np.float64)
    predicted_cov = np.empty((n, 2, 2), dtype=np.float64)
    state = np.array([y[0], initial_slope], dtype=np.float64)
    covariance = np.diag(
        [measurement_variance, measurement_variance + acceleration_variance]
    ).astype(np.float64)
    identity = np.eye(2, dtype=np.float64)

    for index in range(n):
        if index:
            dt = float(u[index] - u[index - 1])
            if dt < 0.0:
                raise ValueError("RTS coordinate must be non-decreasing")
            transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
            process = acceleration_variance * np.array(
                [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
                dtype=np.float64,
            )
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        predicted_mean[index] = state
        predicted_cov[index] = covariance
        effective_r = measurement_variance / float(weights[index])
        innovation_variance = float(covariance[0, 0] + effective_r)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise FloatingPointError("RTS innovation variance is not positive finite")
        gain = covariance[:, 0] / innovation_variance
        state = state + gain * (y[index] - state[0])
        covariance = (identity - np.outer(gain, np.array([1.0, 0.0]))) @ covariance
        covariance = 0.5 * (covariance + covariance.T)
        filtered_mean[index] = state
        filtered_cov[index] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_cov = filtered_cov.copy()
    for index in range(n - 2, -1, -1):
        dt = float(u[index + 1] - u[index])
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        smoother_gain = np.linalg.solve(
            predicted_cov[index + 1], transition @ filtered_cov[index]
        ).T
        smoothed_mean[index] = filtered_mean[index] + smoother_gain @ (
            smoothed_mean[index + 1] - predicted_mean[index + 1]
        )
        smoothed_cov[index] = filtered_cov[index] + smoother_gain @ (
            smoothed_cov[index + 1] - predicted_cov[index + 1]
        ) @ smoother_gain.T
        smoothed_cov[index] = 0.5 * (smoothed_cov[index] + smoothed_cov[index].T)
    return smoothed_mean, smoothed_cov


def robust_rts_smooth(
    values: np.ndarray,
    coordinate: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("robust RTS requires a finite one-dimensional series")
    u, coordinate_spacing = normalized_coordinate(coordinate)
    floor = float(spec["finite_scale_floor"])
    measurement_std = max(robust_scale(np.diff(y)) / math.sqrt(2.0), floor)
    acceleration_std = max(robust_scale(second_divided_differences(y, u)), floor)
    positive = np.diff(u) > 0.0
    slopes = np.diff(y)[positive] / np.diff(u)[positive]
    initial_slope = float(np.median(slopes)) if len(slopes) else 0.0
    measurement_variance = measurement_std**2
    acceleration_variance = acceleration_std**2
    degrees_of_freedom = float(spec["measurement_df"])
    maximum_iterations = int(spec["maximum_iterations"])
    tolerance = float(spec["relative_mean_tolerance"])
    weights = np.ones(len(y), dtype=np.float64)
    previous_level: np.ndarray | None = None
    relative_change = math.inf
    converged = False
    smoothed_mean = np.column_stack([y, np.full(len(y), initial_slope)])
    smoothed_cov = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], len(y), axis=0)

    iteration = 0
    for _iteration in range(1, maximum_iterations + 1):
        iteration = _iteration
        smoothed_mean, smoothed_cov = _student_t_rts_pass(
            y,
            u,
            measurement_variance=measurement_variance,
            acceleration_variance=acceleration_variance,
            weights=weights,
            initial_slope=initial_slope,
        )
        level = smoothed_mean[:, 0]
        if previous_level is not None:
            relative_change = float(
                np.linalg.norm(level - previous_level)
                / max(np.linalg.norm(previous_level), 1.0e-12)
            )
            if relative_change <= tolerance:
                converged = True
                break
        normalized_residual_sq = ((y - level) / measurement_std) ** 2
        weights = (degrees_of_freedom + 1.0) / (
            degrees_of_freedom + normalized_residual_sq
        )
        if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
            raise FloatingPointError("RTS Student-t weights are not positive finite")
        previous_level = level.copy()

    level = smoothed_mean[:, 0]
    posterior_variance = np.maximum(smoothed_cov[:, 0, 0], 0.0)
    finite = bool(np.isfinite(level).all() and np.isfinite(posterior_variance).all())
    return level, posterior_variance, {
        "converged": converged,
        "iterations": iteration,
        "relative_mean_change": relative_change,
        "measurement_std": measurement_std,
        "measurement_variance": measurement_variance,
        "acceleration_std": acceleration_std,
        "acceleration_variance": acceleration_variance,
        "measurement_df": degrees_of_freedom,
        "coordinate_spacing": coordinate_spacing,
        "initial_slope": initial_slope,
        "minimum_weight": float(np.min(weights)),
        "finite_output": finite,
    }


def second_difference(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x[:-2] - 2.0 * x[1:-1] + x[2:]


def second_difference_transpose(values: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.zeros(size, dtype=np.float64)
    output[:-2] += source
    output[1:-1] -= 2.0 * source
    output[2:] += source
    return output


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.maximum(np.abs(array) - float(threshold), 0.0)


def l1_trend_smooth(
    values: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("L1 trend filtering requires a finite series with at least two rows")
    noise_std = robust_scale(np.diff(y)) / math.sqrt(2.0)
    regularization = float(noise_std * math.sqrt(2.0 * math.log(len(y))))
    if len(y) < 3 or regularization == 0.0:
        return y.copy(), {
            "converged": True,
            "iterations": 0,
            "lambda": regularization,
            "rho": float(spec["rho"]),
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "finite_output": True,
        }
    try:
        from scipy.linalg import cho_solve_banded, cholesky_banded
    except ImportError as error:
        raise RuntimeError(
            "SciPy is unavailable; l1_trend must technical-fail without fallback"
        ) from error

    rho = float(spec["rho"])
    maximum_iterations = int(spec["maximum_iterations"])
    absolute_tolerance = float(spec["absolute_tolerance"])
    relative_tolerance = float(spec["relative_tolerance"])
    n = len(y)
    m = n - 2
    diagonal = np.ones(n, dtype=np.float64)
    diagonal[:-2] += rho
    diagonal[1:-1] += 4.0 * rho
    diagonal[2:] += rho
    first_upper = np.zeros(n - 1, dtype=np.float64)
    first_upper[:-1] -= 2.0 * rho
    first_upper[1:] -= 2.0 * rho
    second_upper = np.full(n - 2, rho, dtype=np.float64)
    banded = np.zeros((3, n), dtype=np.float64)
    banded[2] = diagonal
    banded[1, 1:] = first_upper
    banded[0, 2:] = second_upper
    factor = cholesky_banded(banded, lower=False, check_finite=False)

    x = y.copy()
    z = second_difference(x)
    dual = np.zeros(m, dtype=np.float64)
    converged = False
    primal_norm = math.inf
    dual_norm = math.inf
    iteration = 0
    for _iteration in range(1, maximum_iterations + 1):
        iteration = _iteration
        right_hand_side = y + rho * second_difference_transpose(z - dual, n)
        x = cho_solve_banded((factor, False), right_hand_side, check_finite=False)
        d2x = second_difference(x)
        previous_z = z.copy()
        z = soft_threshold(d2x + dual, regularization / rho)
        dual = dual + d2x - z
        primal_norm = float(np.linalg.norm(d2x - z))
        dual_norm = float(
            np.linalg.norm(rho * second_difference_transpose(z - previous_z, n))
        )
        primal_tolerance = math.sqrt(m) * absolute_tolerance + relative_tolerance * max(
            np.linalg.norm(d2x), np.linalg.norm(z)
        )
        dual_tolerance = math.sqrt(n) * absolute_tolerance + relative_tolerance * float(
            np.linalg.norm(rho * second_difference_transpose(dual, n))
        )
        if primal_norm <= primal_tolerance and dual_norm <= dual_tolerance:
            converged = True
            break
    return np.asarray(x, dtype=np.float64), {
        "converged": converged,
        "iterations": iteration,
        "lambda": regularization,
        "rho": rho,
        "primal_residual": primal_norm,
        "dual_residual": dual_norm,
        "finite_output": bool(np.isfinite(x).all()),
    }


def branch_spec(config: Mapping[str, Any], branch: str) -> dict[str, Any]:
    if branch in {BRANCH_RTS_A, BRANCH_RTS_B}:
        fixed = dict(get_nested(config, "model.robust_rts_fixed") or {})
        key = (
            "model.robust_rts_stage0.candidate_a"
            if branch == BRANCH_RTS_A
            else "model.robust_rts_stage0.conditional_candidate_b"
        )
        candidate = dict(get_nested(config, key) or {})
        fixed.update(
            {
                "maximum_iterations": candidate["maximum_iterations"],
                "relative_mean_tolerance": candidate["relative_mean_tolerance"],
            }
        )
        return fixed
    if branch == BRANCH_L1:
        return dict(get_nested(config, "model.l1_trend") or {})
    raise ValueError(f"unknown exp306 solver branch {branch}")


# %% [markdown]
# ## 7. Branch execution and technical gates


# %%
def solve_branch_series(
    branch: str,
    values: np.ndarray,
    coordinate: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = branch_spec(config, branch)
    if branch in {BRANCH_RTS_A, BRANCH_RTS_B}:
        return robust_rts_smooth(values, coordinate, spec)
    if branch == BRANCH_L1:
        output, status = l1_trend_smooth(values, spec)
        return output, np.full(len(output), np.nan, dtype=np.float64), status
    raise ValueError(f"unknown exp306 solver branch {branch}")


def run_branch(
    prepared_by_well: Mapping[str, Mapping[str, Any]],
    branch: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    output_parts: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for well_id in sorted(prepared_by_well):
        prepared = prepared_by_well[well_id]
        for series_kind in SERIES_KINDS:
            values = np.asarray(prepared[f"{series_kind}_gr"], dtype=np.float64)
            coordinate = np.asarray(
                prepared[f"{series_kind}_coordinate"], dtype=np.float64
            )
            try:
                output, variance, details = solve_branch_series(
                    branch, values, coordinate, config
                )
                length_match = len(output) == len(values)
                variance_valid = branch == BRANCH_L1 or (
                    len(variance) == len(values) and np.isfinite(variance).all()
                )
                finite_output = bool(
                    length_match and np.isfinite(output).all() and variance_valid
                )
                converged = bool(details.get("converged", False))
                technical_pass = bool(
                    np.isfinite(values).all()
                    and length_match
                    and finite_output
                    and converged
                )
                error = ""
            except Exception as exception:
                output = np.full(len(values), np.nan, dtype=np.float64)
                variance = np.full(len(values), np.nan, dtype=np.float64)
                details = {"converged": False, "iterations": 0, "finite_output": False}
                length_match = True
                finite_output = False
                converged = False
                technical_pass = False
                error = f"{type(exception).__name__}: {exception}"[:500]

            output_parts.append(
                pd.DataFrame(
                    {
                        "branch": branch,
                        "well_id": well_id,
                        "series_kind": series_kind,
                        "position": np.arange(len(values), dtype=np.int64),
                        "coordinate": coordinate,
                        "input_gr": values,
                        "output_gr": np.asarray(output, dtype=np.float64),
                        "posterior_variance": np.asarray(variance, dtype=np.float64),
                    }
                )
            )
            status_rows.append(
                {
                    "branch": branch,
                    "well_id": well_id,
                    "series_kind": series_kind,
                    "rows": len(values),
                    "finite_input": bool(np.isfinite(values).all()),
                    "length_match": bool(length_match),
                    "order_match": True,
                    "finite_output": bool(finite_output),
                    "converged": bool(converged),
                    "silent_fallback": False,
                    "technical_pass": bool(technical_pass),
                    "iterations": int(details.get("iterations", 0)),
                    "relative_mean_change": details.get("relative_mean_change", np.nan),
                    "measurement_std": details.get("measurement_std", np.nan),
                    "acceleration_std": details.get("acceleration_std", np.nan),
                    "minimum_weight": details.get("minimum_weight", np.nan),
                    "lambda": details.get("lambda", np.nan),
                    "rho": details.get("rho", np.nan),
                    "primal_residual": details.get("primal_residual", np.nan),
                    "dual_residual": details.get("dual_residual", np.nan),
                    "error": error,
                }
            )
    elapsed = float(time.perf_counter() - started)
    output_frame = pd.concat(output_parts, ignore_index=True).sort_values(
        ["branch", "well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)
    status_frame = pd.DataFrame(status_rows).sort_values(
        ["branch", "well_id", "series_kind"], kind="mergesort"
    ).reset_index(drop=True)
    return output_frame, status_frame, elapsed


def branch_has_all_technical_passes(
    status: pd.DataFrame,
    expected_series: int,
) -> bool:
    return bool(
        len(status) == expected_series
        and not status.duplicated(["well_id", "series_kind"]).any()
        and status["technical_pass"].astype(bool).all()
    )


def run_stage0_core(
    prepared_by_well: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, float]]:
    expected_series = len(prepared_by_well) * len(SERIES_KINDS)
    results = {
        BRANCH_RTS_A: run_branch(prepared_by_well, BRANCH_RTS_A, config),
        BRANCH_L1: run_branch(prepared_by_well, BRANCH_L1, config),
    }
    rts_a_status = results[BRANCH_RTS_A][1]
    if not branch_has_all_technical_passes(rts_a_status, expected_series):
        results[BRANCH_RTS_B] = run_branch(prepared_by_well, BRANCH_RTS_B, config)
    return results


def output_matches_input(input_frame: pd.DataFrame, output_frame: pd.DataFrame) -> bool:
    keys = ["well_id", "series_kind", "position"]
    expected = input_frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    actual = output_frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(expected) != len(actual) or actual.duplicated(keys).any():
        return False
    if not expected[keys].equals(actual[keys]):
        return False
    return bool(
        np.array_equal(
            expected["coordinate"].to_numpy(np.float64),
            actual["coordinate"].to_numpy(np.float64),
        )
        and np.array_equal(
            expected["input_gr"].to_numpy(np.float64),
            actual["input_gr"].to_numpy(np.float64),
        )
    )


def evaluate_branch_gate(
    branch: str,
    input_frame: pd.DataFrame,
    output_frame: pd.DataFrame,
    status: pd.DataFrame,
    *,
    stage0_elapsed_seconds: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_wells = int(get_nested(config, "audit.stage0.sample_wells"))
    expected_series = int(get_nested(config, "audit.stage0.expected_series_per_candidate"))
    full_wells = int(get_nested(config, "validation.expected_wells"))
    runtime_limit = float(get_nested(config, "audit.stage0.runtime_projection_limit_seconds"))
    actual_wells = int(status["well_id"].nunique()) if len(status) else 0
    projected_runtime = (
        float(stage0_elapsed_seconds) / actual_wells * full_wells
        if actual_wells
        else math.inf
    )
    criteria = {
        "expected_well_coverage": actual_wells == expected_wells,
        "expected_status_rows": len(status) == expected_series,
        "status_identity": not status.duplicated(["branch", "well_id", "series_kind"]).any(),
        "finite_input": bool(len(status) and status["finite_input"].astype(bool).all()),
        "finite_output": bool(
            len(status)
            and status["finite_output"].astype(bool).all()
            and np.isfinite(output_frame["output_gr"].to_numpy(np.float64)).all()
            and (
                branch == BRANCH_L1
                or np.isfinite(
                    output_frame["posterior_variance"].to_numpy(np.float64)
                ).all()
            )
        ),
        "length_order_identity": bool(
            len(status)
            and status[["length_match", "order_match"]].astype(bool).all().all()
            and output_matches_input(input_frame, output_frame)
        ),
        "all_converged": bool(len(status) and status["converged"].astype(bool).all()),
        "silent_fallback_zero": bool(
            len(status) and not status["silent_fallback"].astype(bool).any()
        ),
        "all_technical_pass": bool(
            len(status) and status["technical_pass"].astype(bool).all()
        ),
        "runtime_projection_within_limit": projected_runtime <= runtime_limit,
    }
    technical_keys = [
        key for key in criteria if key != "runtime_projection_within_limit"
    ]
    technical_passed = all(criteria[key] for key in technical_keys)
    return {
        "branch": branch,
        "stage0_elapsed_seconds": float(stage0_elapsed_seconds),
        "projected_full_runtime_seconds": float(projected_runtime),
        "runtime_limit_seconds": runtime_limit,
        "criteria": criteria,
        "technical_passed": technical_passed,
        "provisional_full_eligible": technical_passed
        and criteria["runtime_projection_within_limit"],
        "status_content_sha256": dataframe_content_sha(status),
        "output_content_sha256": dataframe_content_sha(output_frame),
    }


# %% [markdown]
# ## 8. Eight-well deterministic parity audit


# %%
def parity_content_hashes(
    output_frame: pd.DataFrame,
    status_frame: pd.DataFrame,
) -> dict[str, str]:
    output_sorted = output_frame.sort_values(
        ["branch", "well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)
    status_sorted = status_frame.sort_values(
        ["branch", "well_id", "series_kind"], kind="mergesort"
    ).reset_index(drop=True)
    iteration_columns = ["branch", "well_id", "series_kind", "iterations"]
    return {
        "output_content_sha256": dataframe_content_sha(output_sorted),
        "status_content_sha256": dataframe_content_sha(status_sorted),
        "iteration_content_sha256": dataframe_content_sha(
            status_sorted[iteration_columns]
        ),
    }


def run_parity_audit(
    branch: str,
    prepared_by_well: Mapping[str, Mapping[str, Any]],
    main_output: pd.DataFrame,
    main_status: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    parity_count = int(get_nested(config, "audit.stage0.deterministic_parity_wells"))
    parity_wells = list(prepared_by_well)[:parity_count]
    parity_inputs = {well_id: prepared_by_well[well_id] for well_id in parity_wells}
    rerun_output, rerun_status, rerun_elapsed = run_branch(parity_inputs, branch, config)
    main_output_subset = main_output.loc[main_output["well_id"].isin(parity_wells)].copy()
    main_status_subset = main_status.loc[main_status["well_id"].isin(parity_wells)].copy()
    main_hashes = parity_content_hashes(main_output_subset, main_status_subset)
    rerun_hashes = parity_content_hashes(rerun_output, rerun_status)
    exact_identity = main_hashes == rerun_hashes
    return {
        "branch": branch,
        "wells": parity_wells,
        "series_runs": len(parity_wells) * len(SERIES_KINDS),
        "rerun_elapsed_seconds": rerun_elapsed,
        "main": main_hashes,
        "rerun": rerun_hashes,
        "exact_identity": exact_identity,
    }


# %% [markdown]
# ## 9. Stage 0 orchestration and generated artifacts


# %%
def run_stage0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp306 Stage 0 must run on Kaggle; EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for an explicitly approved local smoke run"
        )
    validate_technical_contract(config, require_run_approval=True)
    outputs = artifact_dir()
    raw_dir = train_data_dir(config)
    all_wells = enumerate_paired_wells(raw_dir)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(all_wells) != expected_wells:
        raise ValueError(f"raw train has {len(all_wells)} wells, expected {expected_wells}")

    contract = build_technical_contract(config)
    contract_path = outputs / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, contract)

    raw_identity = raw_well_identity_manifest(raw_dir, all_wells)
    raw_identity_sha = dataframe_content_sha(
        raw_identity,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_raw_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if raw_identity_sha != expected_raw_sha:
        raise ValueError(
            f"raw well identity SHA mismatch: {raw_identity_sha} != {expected_raw_sha}"
        )

    sample = stable_stage0_sample(
        all_wells,
        salt=str(get_nested(config, "audit.stage0.sample_salt")),
        sample_wells=int(get_nested(config, "audit.stage0.sample_wells")),
    )
    sample = sample.merge(raw_identity, on="well_id", how="left", validate="one_to_one")
    sample_path = outputs / f"{OUTPUT_PREFIX}_stage0_sample_manifest.csv"
    sample.to_csv(sample_path, index=False)
    sample_manifest = {
        "path": str(sample_path),
        "rows": len(sample),
        "raw_sha256": sha256_path(sample_path),
        "content_sha256": dataframe_content_sha(sample),
    }

    preparation_started = time.perf_counter()
    prepared = load_prepared_stage0(raw_dir, sample, config)
    input_frame = build_input_frame(prepared)
    preparation_elapsed = float(time.perf_counter() - preparation_started)
    input_artifact = write_csv_gzip(
        input_frame,
        outputs / f"{OUTPUT_PREFIX}_stage0_input.csv.gz",
    )

    branch_results = run_stage0_core(prepared, config)
    branch_gates: dict[str, dict[str, Any]] = {}
    for branch, (branch_output, branch_status, solver_elapsed) in branch_results.items():
        branch_gates[branch] = evaluate_branch_gate(
            branch,
            input_frame,
            branch_output,
            branch_status,
            stage0_elapsed_seconds=preparation_elapsed + solver_elapsed,
            config=config,
        )

    rts_a_passed = bool(branch_gates[BRANCH_RTS_A]["technical_passed"])
    selected_rts_branch = BRANCH_RTS_A if rts_a_passed else BRANCH_RTS_B
    if not rts_a_passed and BRANCH_RTS_B not in branch_results:
        raise RuntimeError("RTS A failed but conditional RTS B did not run")

    parity_branches = [
        branch
        for branch in (selected_rts_branch, BRANCH_L1)
        if bool(branch_gates[branch]["provisional_full_eligible"])
    ]
    parity_rows = []
    for branch in parity_branches:
        branch_output, branch_status, _ = branch_results[branch]
        parity_rows.append(
            run_parity_audit(
                branch,
                prepared,
                branch_output,
                branch_status,
                config,
            )
        )
    parity_by_branch = {str(row["branch"]): row for row in parity_rows}
    for branch, gate in branch_gates.items():
        parity = parity_by_branch.get(branch)
        gate["parity_required"] = branch in parity_branches
        gate["parity_exact_identity"] = (
            bool(parity["exact_identity"]) if parity is not None else None
        )
        gate["full_eligible"] = bool(
            gate["provisional_full_eligible"]
            and parity is not None
            and parity["exact_identity"]
            and branch in {selected_rts_branch, BRANCH_L1}
        )

    parity_manifest = {
        "experiment": EXPERIMENT_NAME,
        "parity_wells": list(prepared)[
            : int(get_nested(config, "audit.stage0.deterministic_parity_wells"))
        ],
        "branches": parity_rows,
        "all_required_exact": all(bool(row["exact_identity"]) for row in parity_rows),
    }
    parity_path = outputs / f"{OUTPUT_PREFIX}_parity_manifest.json"
    write_json(parity_path, parity_manifest)

    combined_output = pd.concat(
        [value[0] for value in branch_results.values()], ignore_index=True
    ).sort_values(
        ["branch", "well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)
    combined_status = pd.concat(
        [value[1] for value in branch_results.values()], ignore_index=True
    ).sort_values(
        ["branch", "well_id", "series_kind"], kind="mergesort"
    ).reset_index(drop=True)
    output_artifact = write_csv_gzip(
        combined_output,
        outputs / f"{OUTPUT_PREFIX}_stage0_output.csv.gz",
    )
    status_artifact = write_csv_gzip(
        combined_status,
        outputs / f"{OUTPUT_PREFIX}_stage0_solver_status.csv.gz",
    )

    gate_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0",
        "truth_or_scientific_score_loaded": False,
        "raw_well_identity_content_sha256": raw_identity_sha,
        "sample_manifest": sample_manifest,
        "input_artifact": input_artifact,
        "output_artifact": output_artifact,
        "solver_status_artifact": status_artifact,
        "runtime_environment": runtime_manifest(),
        "selected_rts_branch": selected_rts_branch,
        "rts_b_executed": BRANCH_RTS_B in branch_results,
        "preparation_elapsed_seconds": preparation_elapsed,
        "branches": branch_gates,
        "full_eligible_branches": sorted(
            branch for branch, gate in branch_gates.items() if gate["full_eligible"]
        ),
        "full_audit_executed": False,
    }
    gate_path = outputs / f"{OUTPUT_PREFIX}_stage0_gate.json"
    write_json(gate_path, gate_manifest)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed_awaiting_separate_full_audit_decision",
        "route": "pf_beam",
        "stage0_sample_wells": len(sample),
        "stage0_series_per_branch": len(sample) * len(SERIES_KINDS),
        "executed_branches": sorted(branch_results),
        "selected_rts_branch": selected_rts_branch,
        "full_eligible_branches": gate_manifest["full_eligible_branches"],
        "scientific_score": None,
        "cv": None,
        "prediction": None,
        "submission": None,
        "full_audit_executed": False,
        "artifacts": {
            "contract": str(contract_path),
            "sample": sample_manifest,
            "input": input_artifact,
            "output": output_artifact,
            "solver_status": status_artifact,
            "parity": str(parity_path),
            "gate": str(gate_path),
        },
    }
    summary_path = outputs / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_technical_contract(CONFIG)
    CONTRACT_PREVIEW = build_technical_contract(CONFIG)
    print(json.dumps(to_jsonable(CONTRACT_PREVIEW), indent=2, sort_keys=True))


# %% [markdown]
# ## 11. Run the separately approved Kaggle CPU Stage 0


# %%
if EXECUTE_NOTEBOOK:
    if not bool(get_nested(CONFIG, "execution.run_stage0")):
        raise RuntimeError(
            "exp306 Stage 0 is implemented, but Kaggle package/push/run is not approved; "
            "full audit, scientific scoring, inference, and submission remain disabled"
        )
    STAGE0_SUMMARY = run_stage0_experiment(CONFIG)
