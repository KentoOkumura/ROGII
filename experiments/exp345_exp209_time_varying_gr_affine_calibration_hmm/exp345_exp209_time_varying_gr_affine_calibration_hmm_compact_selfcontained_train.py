# %% [markdown]
# # exp345 exp209 time-varying GR affine calibration HMM train
#
# Train-side staged audit of one deterministic current-well causal affine
# schedule. The exp209 observation scale and exact-HMM state grammar are fixed.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Robust prefix affine and outer-fold process-noise helpers
# 5. Exact exp209 forward-backward kernel
# 6. Prefix masking, causal affine schedule, and exact-HMM decoding
# 7. Target-free staged generation and freeze
# 8. Late truth attachment, diagnostics, and promotion gates
# 9. Experiment orchestration and generated artifacts
# 10. Setup and configuration preview
# 11. Run the approved Kaggle CPU stage

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def prange(*args: Any) -> range:
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None

    def get_num_threads() -> int | None:
        return None

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(func: Any) -> Any:
            return func

        return decorator


EXPERIMENT_NAME = "exp345_exp209_time_varying_gr_affine_calibration_hmm"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT = "one_pass_causal_affine_schedule_on_exp209"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP345_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
    raise FileNotFoundError(f"exp345 config not found in {[str(path) for path in candidates]}")


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
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": digest.hexdigest(),
        "content_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
    }


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
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


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate_paths(spec: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in spec.get("candidates", [])]


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        variants = (
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            Path.cwd() / candidate
            if candidate.name == filename
            else Path.cwd() / candidate / filename,
        )
        for path in variants:
            checked.append(str(path))
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file() and path.stat().st_size > 0:
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
        versions["numba_num_threads"] = get_num_threads()
    return versions


def stable_sha_well_order(wells: Iterable[str]) -> list[str]:
    return sorted(
        (str(well) for well in wells),
        key=lambda well: (hashlib.sha256(well.encode()).hexdigest(), well),
    )


def parse_row_index(identifier: str) -> int:
    try:
        return int(str(identifier).rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid exp209 row id: {identifier}") from exc


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return inspect_gzip_csv(path)


# %% [markdown]
# ## 3. Frozen scientific contract and input preflight


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> None:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "implementation.enabled": True,
        "model.active_variants": [VARIANT],
        "model.base_path.calibration_iterations": 1,
        "model.affine_state.state": ["intercept_b", "log_scale_a"],
        "model.affine_state.transition": "local_level_random_walk",
        "model.affine_state.filter": "deterministic_causal_extended_kalman_filter_no_smoother",
        "model.affine_state.minimum_prefix_pairs": 40,
        "model.affine_state.minimum_typewell_gr_std": 5.0,
        "model.affine_state.maximum_prefix_rmse": 60.0,
        "model.affine_state.prefix_rmse_scope": ("retained_pairs_after_fixed_trim_iterations"),
        "model.affine_state.trim_quantile": 0.90,
        "model.affine_state.robust_iterations": 2,
        "model.affine_state.process_noise_state_sequence": (
            "cumulative_robust_fit_every_40_finite_pairs_plus_final"
        ),
        "model.affine_state.process_noise_increment_normalization": (
            "squared_state_increment_per_source_row"
        ),
        "model.affine_state.process_noise_shrinkage_space": "linear_variance",
        "model.affine_state.initial_covariance": (
            "trimmed_ols_covariance_transformed_to_intercept_and_log_scale"
        ),
        "model.affine_state.covariance_update": "joseph_form",
        "model.affine_state.missing_raw_gr_update_policy": "skip_update_propagate_state",
        "model.affine_state.schedule_state_timing": (
            "current_row_posterior_after_one_finite_raw_gr_update"
        ),
        "model.affine_state.gr_nll_policy": "one_step_predictive_before_current_row_update",
        "runtime.num_workers": 2,
        "runtime.numba_num_threads": 2,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp345 fixed contract mismatch: {key} must be {value!r}")
    if [float(value) for value in get_nested(config, "model.affine_state.slope_bounds")] != [
        0.25,
        4.0,
    ]:
        raise ValueError("exp345 fixes affine slope bounds to [0.25, 4.0]")
    hmm = get_nested(config, "model.fixed_exp209_hmm") or {}
    fixed_numeric = {
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "lam": 1.0,
        "start_sig": 0.75,
        "r0_sig": 0.01,
        "band_pad": 100.0,
        "momentum": 0.998,
        "emission_clip_z2": 600.0,
        "affine_parent_a": 1.0,
        "affine_parent_b": 0.0,
        "effective_position_sigma_floor": 0.1225,
    }
    for key, value in fixed_numeric.items():
        if float(hmm.get(key, -1.0)) != value:
            raise ValueError(f"exp345 fixes model.fixed_exp209_hmm.{key}={value}")
    fixed_text = {
        "emission": "gaussian",
        "sigma_mode": "std",
        "sigma_known_gr_policy": "fill_missing_with_zero_before_population_std",
        "missing_gr_weight": "none_exact_weight_1",
        "rate_center": "zero",
        "evaluation_gr_policy": "interpolate_both_directions_then_typewell_mean",
        "typewell_gr_policy": "sort_tvt_ffill_bfill_then_linear_interp",
        "output": "posterior_mean",
    }
    for key, value in fixed_text.items():
        if hmm.get(key) != value:
            raise ValueError(f"exp345 fixes model.fixed_exp209_hmm.{key}={value!r}")
    if [float(value) for value in hmm.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("exp345 fixes exp209 sigma clip to [10, 60]")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise RuntimeError("exp345 implementation approval must be recorded")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp345 Kaggle package/push/run is not approved")
        flags = [
            bool(get_nested(config, "execution.run_microbenchmark")),
            bool(get_nested(config, "execution.run_stage_0")),
            bool(get_nested(config, "execution.run_stage_1")),
        ]
        if sum(flags) != 1:
            raise RuntimeError("exactly one exp345 execution stage must be enabled")
        if flags[1]:
            evidence = str(get_nested(config, "execution.runtime_gate_evidence_sha256") or "")
            if len(evidence) != 64:
                raise RuntimeError("Stage 0 requires frozen runtime-gate evidence SHA")
        if flags[2]:
            evidence = str(get_nested(config, "execution.stage_0_gate_evidence_sha256") or "")
            if len(evidence) != 64:
                raise RuntimeError("Stage 1 requires frozen Stage 0 gate evidence SHA")
            if get_nested(config, "execution.stage_0_gate_passed") is not True:
                raise RuntimeError("Stage 1 requires Stage 0 gate PASS")


def selected_stage(config: Mapping[str, Any]) -> str:
    flags = {
        "stage_0_microbenchmark": bool(get_nested(config, "execution.run_microbenchmark")),
        "stage_0_full": bool(get_nested(config, "execution.run_stage_0")),
        "stage_1_full_suffix": bool(get_nested(config, "execution.run_stage_1")),
    }
    active = [name for name, enabled in flags.items() if enabled]
    if len(active) != 1:
        raise RuntimeError(f"one run stage is required, got {active}")
    return active[0]


def build_scientific_contract(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "stage": stage,
        "truth_attached": False,
        "variant": VARIANT,
        "base_path": get_nested(config, "model.base_path"),
        "affine_state": get_nested(config, "model.affine_state"),
        "fixed_exp209_hmm": get_nested(config, "model.fixed_exp209_hmm"),
        "prefix_mask": get_nested(config, "validation.prefix_mask_backtest"),
        "promotion_gates": get_nested(config, "promotion_gates"),
        "execution_contract": get_nested(config, "execution_contract"),
        "forbidden": get_nested(config, "model.forbidden"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_raw_well_identity(config: Mapping[str, Any], raw_dir: Path) -> dict[str, Any]:
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
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    actual = dataframe_content_sha(
        frame, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    )
    expected = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != int(get_nested(config, "validation.expected_wells")) or actual != expected:
        raise ValueError("current raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": actual,
        "well_ids": frame["well_id"].tolist(),
    }


def preflight_controls_and_assignments(config: Mapping[str, Any]) -> dict[str, Any]:
    saved = get_nested(config, "data.saved_exp209") or {}
    hmm_path = resolve_existing(str(saved["hmm_cache_filename"]), _candidate_paths(saved))
    likpf_path = resolve_existing(str(saved["likpf_cache_filename"]), _candidate_paths(saved))
    hmm_report = inspect_gzip_csv(hmm_path)
    likpf_report = inspect_gzip_csv(likpf_path)
    if hmm_report["decompressed_sha256"] != str(saved["expected_hmm_decompressed_sha256"]):
        raise ValueError("saved exp209 HMM decompressed SHA mismatch")
    if likpf_report["decompressed_sha256"] != str(saved["expected_likpf_decompressed_sha256"]):
        raise ValueError("saved exp209 LikPF decompressed SHA mismatch")
    fold = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(str(fold["filename"]), _candidate_paths(fold))
    fold_report = inspect_gzip_csv(fold_path)
    if fold_report["decompressed_sha256"] != str(fold["expected_decompressed_sha256"]):
        raise ValueError("fold assignment decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold["safe_columns"]]
    safe = pd.read_csv(fold_path, usecols=safe_columns, dtype={"well_id": str})
    for column in ("row_idx", "suffix_offset", "fold"):
        safe[column] = pd.to_numeric(safe[column], errors="raise").astype(np.int64)
    safe = safe.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if safe.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("fold assignment identity is duplicated")
    fold_counts = safe.groupby("well_id", sort=True)["fold"].nunique()
    if not fold_counts.eq(1).all():
        raise ValueError("each well must have one outer fold")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if (
        len(safe) != int(get_nested(config, "validation.expected_rows"))
        or safe["well_id"].nunique() != int(get_nested(config, "validation.expected_wells"))
        or sorted(safe["fold"].unique().tolist()) != expected_folds
    ):
        raise ValueError("fold assignment row/well/fold coverage mismatch")
    fold_map = safe.groupby("well_id", sort=True)["fold"].first().astype(int)
    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    hidden_path = resolve_existing(str(hidden["filename"]), _candidate_paths(hidden))
    hidden_sha = sha256_path(hidden_path)
    if hidden_sha != str(hidden["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    role_columns = [str(value) for value in hidden["role_columns"].values()]
    hidden_frame = pd.read_csv(
        hidden_path, usecols=["well_id", *role_columns], dtype={"well_id": str}
    )
    if hidden_frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment has duplicate wells")
    if sorted(hidden_frame["well_id"]) != sorted(fold_map.index.astype(str)):
        raise ValueError("hidden-like and fold assignment well identities differ")
    return {
        "paths": {
            "saved_hmm": str(hmm_path),
            "saved_likpf": str(likpf_path),
            "fold_assignment": str(fold_path),
            "hidden_like_assignment": str(hidden_path),
        },
        "saved_hmm": hmm_report,
        "saved_likpf": likpf_report,
        "fold_assignment": {**fold_report, "well_ids": sorted(fold_map.index.astype(str).tolist())},
        "hidden_like_assignment": {
            "path": str(hidden_path),
            "raw_sha256": hidden_sha,
            "wells": len(hidden_frame),
        },
    }


# %% [markdown]
# ## 4. Robust prefix affine and outer-fold process-noise helpers


# %%
def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv", usecols=["MD", "Z", "GR", "TVT_input"]
    )
    if "TVT" in frame.columns:
        raise RuntimeError("unknown-suffix truth entered the target-free horizontal frame")
    return frame.reset_index(drop=True)


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.sort_values("TVT", kind="mergesort").reset_index(drop=True)
    frame["GR"] = frame["GR"].ffill().bfill()
    frame = frame.loc[np.isfinite(frame["TVT"]) & np.isfinite(frame["GR"])].copy()
    if len(frame) < 2 or np.any(np.diff(frame["TVT"].to_numpy(np.float64)) < 0):
        raise ValueError(f"well={well} has invalid Type Well TVT/GR")
    return frame


def robust_affine_fit(x: np.ndarray, y: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    spec = get_nested(config, "model.affine_state") or {}
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
    if fallback_reason is None:
        for _ in range(iterations):
            design = np.column_stack([np.ones(int(keep.sum())), x[keep]])
            beta = np.linalg.lstsq(design, y[keep], rcond=None)[0]
            raw_slope = float(beta[1])
            slope = float(np.clip(raw_slope, slope_low, slope_high))
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
    else:
        fit_rmse = float("nan")
        all_pair_rmse = float("nan")
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
    dof = max(1, int(keep.sum()) - 2)
    residual_variance = float(np.sum(residual**2) / dof)
    covariance_beta = residual_variance * np.linalg.pinv(design.T @ design)
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0 / float(beta[1])]], dtype=np.float64)
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
    horizontal: pd.DataFrame, typewell: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    finite = np.isfinite(tvt) & np.isfinite(raw_gr)
    rows = np.flatnonzero(finite)
    x = np.interp(tvt[finite], typewell_tvt, typewell_gr)
    return x, raw_gr[finite], rows


def prefix_process_noise_raw(
    horizontal: pd.DataFrame, typewell: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    minimum = int(get_nested(config, "model.affine_state.minimum_prefix_pairs"))
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
            states.append(np.array([fit["intercept_b"], fit["log_scale_a"]], dtype=np.float64))
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
    stage: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = sorted((str(well), int(fold)) for well, fold in fold_map.items())
    for index, (well, fold) in enumerate(ordered, start=1):
        print(f"[process-noise {index}/{len(ordered)}] well={well}", flush=True)
        horizontal = load_horizontal_without_truth(well, raw_dir)
        typewell = load_typewell(well, raw_dir)
        outer_train_raw = prefix_process_noise_raw(horizontal, typewell, config)
        self_horizontal = (
            stage0_masked_horizontal(horizontal, config)[0]
            if stage.startswith("stage_0")
            else horizontal
        )
        heldout_visible_raw = prefix_process_noise_raw(self_horizontal, typewell, config)
        rows.append(
            {
                "well_id": well,
                "fold": fold,
                **heldout_visible_raw,
                **{f"outer_train_source_{key}": value for key, value in outer_train_raw.items()},
            }
        )
    audit = pd.DataFrame(rows)
    pseudocount = 100.0
    floor = float(get_nested(config, "model.affine_state.process_noise_numerical_floor"))
    for fold in sorted(audit["fold"].unique()):
        outer = audit.loc[audit["fold"] != fold]
        for raw_column, output_column in (
            ("raw_q_intercept", "q_intercept"),
            ("raw_q_log_scale", "q_log_scale"),
        ):
            outer_source_column = f"outer_train_source_{raw_column}"
            finite_outer = pd.to_numeric(outer[outer_source_column], errors="coerce")
            finite_outer = finite_outer[np.isfinite(finite_outer)]
            if finite_outer.empty:
                raise ValueError(f"fold={fold} has no finite outer-train {raw_column}")
            global_median = max(float(np.median(finite_outer)), floor)
            mask = audit["fold"] == fold
            for row_index in audit.index[mask]:
                raw_value = float(audit.at[row_index, raw_column])
                support = int(audit.at[row_index, "process_increments"])
                alpha = (
                    float(support / (support + pseudocount)) if math.isfinite(raw_value) else 0.0
                )
                shrunk = alpha * max(raw_value, floor) + (1.0 - alpha) * global_median
                audit.at[row_index, f"outer_train_median_{output_column}"] = global_median
                audit.at[row_index, f"shrinkage_alpha_{output_column}"] = alpha
                audit.at[row_index, output_column] = max(float(shrunk), floor)
    finite_columns = [
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
    numeric = audit[finite_columns].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("process-noise audit contains non-finite values after fold shrinkage")
    return audit.sort_values(["fold", "well_id"], kind="mergesort").reset_index(drop=True)


def exp209_prefix_scale(horizontal: pd.DataFrame, typewell: pd.DataFrame) -> dict[str, Any]:
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    known = np.isfinite(tvt)
    typewell_at_known = np.interp(
        tvt[known],
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    zero_fill_residual = (
        np.where(np.isfinite(raw_gr[known]), raw_gr[known], 0.0) - typewell_at_known
    )
    raw_sigma = float(np.std(zero_fill_residual, ddof=0))
    sigma = float(np.clip(raw_sigma, 10.0, 60.0))
    return {
        "known_prefix_rows": int(known.sum()),
        "known_missing_gr_rows": int((known & ~np.isfinite(raw_gr)).sum()),
        "zero_fill_std_raw": raw_sigma,
        "sigma_gr": sigma,
    }


def robust_initial_rate(horizontal: pd.DataFrame, tail_n: int = 30) -> float:
    known = horizontal.loc[pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna()].tail(
        tail_n
    )
    tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(known["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(known["MD"], errors="coerce").to_numpy(np.float64)
    dmd = np.diff(md)
    valid = np.isfinite(dmd) & (dmd > 0.0)
    if len(dmd) and int(valid.sum()) >= 3:
        return float(np.median((np.diff(tvt)[valid] + np.diff(z)[valid]) / dmd[valid]))
    return 0.0


# %% [markdown]
# ## 5. Exact exp209 forward-backward kernel


# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb(
    em,
    dm,
    dz,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    """Amerhu exact forward-backward over joint state (TVT position, dip-rate)."""
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    alpha = np.full((t_count, p_count, r_count), neg, np.float32)

    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)

    tmp = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = 0.5 * (rate_var_cells + mean_rate_move)
            p_minus = 0.5 * (rate_var_cells - mean_rate_move)
            if p_plus < 1e-12:
                p_plus = 1e-12
            if p_minus < 1e-12:
                p_minus = 1e-12
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)

        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = r2 - 1 if r2 - 1 >= 0 else 0
                k1 = r2 + 1 if r2 + 1 <= r_count - 1 else r_count - 1
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best)
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = sig_p if sig_p > 0.35 * sp else 0.35 * sp
        for r2 in range(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = position_log_kernel[0]
            for k_i in range(1, 5):
                if position_log_kernel[k_i] > kernel_max:
                    kernel_max = position_log_kernel[k_i]
            kernel_sum = 0.0
            for k_i in range(5):
                kernel_sum += np.exp(position_log_kernel[k_i] - kernel_max)
            log_norm = kernel_max + np.log(kernel_sum)
            for k_i in range(5):
                position_log_kernel[k_i] -= log_norm
            for p2 in prange(p_count):
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if p1 < 0 or p1 >= p_count:
                        continue
                    value = tmp[p1, r2] + position_log_kernel[k_i]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if p1 < 0 or p1 >= p_count:
                            continue
                        total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    cur[p2, r2] = np.float32(best + np.log(total) + lam * em[t_i, p2])
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]

    best = np.float32(neg)
    for p_i in range(p_count):
        for r_i in range(r_count):
            if alpha[t_count - 1, p_i, r_i] > best:
                best = alpha[t_count - 1, p_i, r_i]
    total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            total += np.exp(alpha[t_count - 1, p_i, r_i] - best)
    loglik = float(best) + np.log(total)

    post_p = np.zeros((t_count, p_count))
    beta_next = np.zeros((p_count, r_count), np.float32)

    best = neg
    for p_i in range(p_count):
        for r_i in range(r_count):
            value = alpha[t_count - 1, p_i, r_i] + beta_next[p_i, r_i]
            if value > best:
                best = value
    total = 0.0
    for p_i in range(p_count):
        acc = 0.0
        for r_i in range(r_count):
            acc += np.exp(alpha[t_count - 1, p_i, r_i] + beta_next[p_i, r_i] - best)
        post_p[t_count - 1, p_i] = acc
        total += acc
    for p_i in range(p_count):
        post_p[t_count - 1, p_i] /= total

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = 0.5 * (rate_var_cells + mean_rate_move)
            p_minus = 0.5 * (rate_var_cells - mean_rate_move)
            if p_plus < 1e-12:
                p_plus = 1e-12
            if p_minus < 1e-12:
                p_minus = 1e-12
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        sigma_position = sig_p if sig_p > 0.35 * sp else 0.35 * sp
        for r2 in range(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = position_log_kernel[0]
            for k_i in range(1, 5):
                if position_log_kernel[k_i] > kernel_max:
                    kernel_max = position_log_kernel[k_i]
            kernel_sum = 0.0
            for k_i in range(5):
                kernel_sum += np.exp(position_log_kernel[k_i] - kernel_max)
            log_norm = kernel_max + np.log(kernel_sum)
            for k_i in range(5):
                position_log_kernel[k_i] -= log_norm
            for p1 in prange(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if p2 < 0 or p2 >= p_count:
                        continue
                    value = position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if p2 < 0 or p2 >= p_count:
                            continue
                        total += np.exp(
                            position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2] - best
                        )
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg

        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = r_i - 1 if r_i - 1 >= 0 else 0
                k1 = r_i + 1 if r_i + 1 <= r_count - 1 else r_count - 1
                for r2 in range(k0, k1 + 1):
                    value = rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg

        best = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                value = alpha[t_i - 1, p_i, r_i] + beta_cur[p_i, r_i]
                if value > best:
                    best = value
        total = 0.0
        for p_i in range(p_count):
            acc = 0.0
            for r_i in range(r_count):
                acc += np.exp(alpha[t_i - 1, p_i, r_i] + beta_cur[p_i, r_i] - best)
            post_p[t_i - 1, p_i] = acc
            total += acc
        for p_i in range(p_count):
            post_p[t_i - 1, p_i] /= total
        for p_i in range(p_count):
            for r_i in range(r_count):
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    return post_p, loglik


# %% [markdown]
# ## 6. Prefix masking, causal affine schedule, and exact-HMM decoding


# %%
def stage0_masked_horizontal(
    horizontal: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizon = int(get_nested(config, "validation.prefix_mask_backtest.horizon_rows"))
    minimum_visible = int(
        get_nested(config, "validation.prefix_mask_backtest.minimum_visible_prefix_rows")
    )
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_index = np.flatnonzero(np.isfinite(tvt))
    if len(known_index) <= minimum_visible:
        raise ValueError("well has insufficient visible prefix for Stage 0 mask")
    mask_count = min(horizon, len(known_index) - minimum_visible)
    score_index = known_index[-mask_count:]
    stop = int(known_index[-1]) + 1
    masked = horizontal.iloc[:stop].copy().reset_index(drop=True)
    masked.loc[score_index, "TVT_input"] = np.nan
    remaining = int(pd.to_numeric(masked["TVT_input"], errors="coerce").notna().sum())
    if remaining < minimum_visible:
        raise ValueError("Stage 0 mask violated minimum visible-prefix rows")
    return masked, {
        "score_start_row": int(score_index[0]),
        "score_stop_row_exclusive": int(score_index[-1]) + 1,
        "score_rows": int(mask_count),
        "visible_prefix_rows": remaining,
        "original_known_prefix_rows": int(len(known_index)),
        "truncated_at_original_prefix_boundary": True,
    }


def prepare_hmm_inputs(
    horizontal: pd.DataFrame, typewell: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_exp209_hmm") or {}
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    if np.isinf(tvt_input).any():
        raise ValueError("TVT_input may contain finite values or NaN only")
    known_mask = np.isfinite(tvt_input)
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("each exp345 decoding frame needs a known prefix and an unknown suffix")
    known_index = np.flatnonzero(known_mask)
    eval_index = np.flatnonzero(eval_mask)
    if int(known_index[-1]) >= int(eval_index[0]):
        raise ValueError("exp345 requires one contiguous known prefix followed by one suffix")
    last_index = int(known_index[-1])
    last_tvt = float(tvt_input[last_index])
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    step = float(hmm["step"])
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    state_gr = np.interp(grid, typewell_tvt, typewell_gr)
    raw_gr_series = pd.to_numeric(horizontal["GR"], errors="coerce")
    raw_gr = raw_gr_series.to_numpy(np.float64)
    observed_gr = (
        raw_gr_series.interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[eval_index]
    )
    md_all = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="raise").to_numpy(np.float64)
    md = md_all[eval_index]
    z = z_all[eval_index]
    dm = np.maximum(np.diff(np.concatenate([[md_all[last_index]], md])), 1.0)
    dz = np.diff(np.concatenate([[z_all[last_index]], z]))
    init_rate = robust_initial_rate(horizontal)
    span = max(float(hmm["rate_span"]), abs(init_rate) + 0.04)
    rates = np.linspace(-span, span, int(hmm["n_rates"]), dtype=np.float64)
    return {
        "eval_index": eval_index,
        "last_index": last_index,
        "last_tvt": last_tvt,
        "last_md": float(md_all[last_index]),
        "grid": grid,
        "state_gr": state_gr,
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "raw_gr_eval": raw_gr[eval_index],
        "observed_gr": observed_gr,
        "dm": dm,
        "dz": dz,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "init_rate": init_rate,
        "prefix_scale": exp209_prefix_scale(horizontal, typewell),
        "md_since": md - float(md_all[last_index]),
    }


def run_exact_hmm(
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    scale_schedule: np.ndarray,
    intercept_schedule: np.ndarray,
) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_exp209_hmm") or {}
    sigma_gr = float(prepared["prefix_scale"]["sigma_gr"])
    if not 10.0 <= sigma_gr <= 60.0:
        raise ValueError("exp345 must preserve the exp209 sigma_GR clip")
    scale = np.asarray(scale_schedule, dtype=np.float64)
    intercept = np.asarray(intercept_schedule, dtype=np.float64)
    if len(scale) != len(prepared["eval_index"]) or len(intercept) != len(scale):
        raise ValueError("affine schedule length does not match HMM suffix")
    state_gr = scale[:, None] * np.asarray(prepared["state_gr"])[None, :] + intercept[:, None]
    zscore = (np.asarray(prepared["observed_gr"])[:, None] - state_gr) / sigma_gr
    emission = (-0.5 * np.minimum(zscore**2, float(hmm["emission_clip_z2"]))).astype(np.float32)
    post_p, loglik = _hmm2_fb(
        emission,
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(hmm["step"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["init_rate"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["momentum"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    mean = post_p @ grid
    variance = post_p @ (grid**2) - mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    return {
        "mean": mean,
        "std": std,
        "loglik": float(loglik),
        "posterior_row_sum_max_abs_error": float(np.max(np.abs(post_p.sum(axis=1) - 1.0))),
    }


def typewell_value_and_gradient(
    typewell_tvt: np.ndarray, typewell_gr: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique_tvt, inverse = np.unique(np.asarray(typewell_tvt, dtype=np.float64), return_inverse=True)
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


def causal_affine_schedule(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prepared: Mapping[str, Any],
    base_mean: np.ndarray,
    base_std: np.ndarray,
    process_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_prefix, y_prefix, _ = visible_prefix_pairs(horizontal, typewell)
    fit = robust_affine_fit(x_prefix, y_prefix, config)
    eval_index = np.asarray(prepared["eval_index"], dtype=np.int64)
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
    variance = sigma_gr**2 + (suffix_gradient * base_std) ** 2
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise ValueError("affine EKF observation variance is invalid")
    length = len(eval_index)
    if not fit["valid"]:
        frame = pd.DataFrame(
            {
                "row_idx": eval_index,
                "affine_scale_a": np.ones(length, dtype=np.float64),
                "affine_intercept_b": np.zeros(length, dtype=np.float64),
                "raw_gr_update": np.isfinite(raw_gr),
                "predictive_nll_identity": 0.5
                * (np.log(2.0 * np.pi * variance) + (raw_gr - suffix_x) ** 2 / variance),
                "predictive_nll_affine": 0.5
                * (np.log(2.0 * np.pi * variance) + (raw_gr - suffix_x) ** 2 / variance),
            }
        )
        finite_raw = np.isfinite(raw_gr)
        for column in ("predictive_nll_identity", "predictive_nll_affine"):
            frame.loc[~finite_raw, column] = np.nan
        return frame, {
            **{key: value for key, value in fit.items() if key != "covariance"},
            "fallback": True,
            "q_intercept": float(process_row["q_intercept"]),
            "q_log_scale": float(process_row["q_log_scale"]),
            "boundary_jump_sigma": 0.0,
            "finite_updates": int(finite_raw.sum()),
        }
    state = np.array([fit["intercept_b"], fit["log_scale_a"]], dtype=np.float64)
    floor = float(get_nested(config, "model.affine_state.process_noise_numerical_floor"))
    covariance = np.asarray(fit["covariance"], dtype=np.float64)
    covariance = covariance + np.eye(2, dtype=np.float64) * floor
    process_covariance = np.diag(
        [float(process_row["q_intercept"]), float(process_row["q_log_scale"])]
    )
    slope_low, slope_high = (
        float(value) for value in get_nested(config, "model.affine_state.slope_bounds")
    )
    scale_schedule = np.empty(length, dtype=np.float64)
    intercept_schedule = np.empty(length, dtype=np.float64)
    updated = np.isfinite(raw_gr)
    identity_nll = np.full(length, np.nan, dtype=np.float64)
    affine_nll = np.full(length, np.nan, dtype=np.float64)
    boundary_jump = 0.0
    identity = np.eye(2, dtype=np.float64)
    initial_state = state.copy()
    for index in range(length):
        predicted_state = state.copy()
        predicted_covariance = covariance + process_covariance
        scale_pred = float(math.exp(predicted_state[1]))
        predicted_y = float(predicted_state[0] + scale_pred * suffix_x[index])
        h = np.array([1.0, scale_pred * suffix_x[index]], dtype=np.float64)
        innovation_variance = float(h @ predicted_covariance @ h + variance[index])
        if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise ValueError("affine EKF innovation variance is invalid")
        if updated[index]:
            residual = float(raw_gr[index] - predicted_y)
            identity_residual = float(raw_gr[index] - suffix_x[index])
            affine_nll[index] = 0.5 * (
                math.log(2.0 * math.pi * innovation_variance) + residual**2 / innovation_variance
            )
            identity_nll[index] = 0.5 * (
                math.log(2.0 * math.pi * variance[index]) + identity_residual**2 / variance[index]
            )
            gain = predicted_covariance @ h / innovation_variance
            state = predicted_state + gain * residual
            state[1] = float(np.clip(state[1], math.log(slope_low), math.log(slope_high)))
            kh = np.outer(gain, h)
            covariance = (identity - kh) @ predicted_covariance @ (identity - kh).T + np.outer(
                gain, gain
            ) * variance[index]
            covariance = 0.5 * (covariance + covariance.T)
        else:
            state = predicted_state
            covariance = predicted_covariance
        if index == 0:
            delta = state - initial_state
            boundary_jump = float(
                math.sqrt(
                    max(
                        0.0,
                        delta @ np.linalg.pinv(predicted_covariance) @ delta,
                    )
                )
            )
        intercept_schedule[index] = state[0]
        scale_schedule[index] = math.exp(state[1])
    frame = pd.DataFrame(
        {
            "row_idx": eval_index,
            "affine_scale_a": scale_schedule,
            "affine_intercept_b": intercept_schedule,
            "raw_gr_update": updated,
            "predictive_nll_identity": identity_nll,
            "predictive_nll_affine": affine_nll,
        }
    )
    return frame, {
        **{key: value for key, value in fit.items() if key != "covariance"},
        "fallback": False,
        "q_intercept": float(process_row["q_intercept"]),
        "q_log_scale": float(process_row["q_log_scale"]),
        "boundary_jump_sigma": boundary_jump,
        "finite_updates": int(updated.sum()),
    }


def load_saved_exp209_base(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    saved = get_nested(config, "data.saved_exp209") or {}
    columns = [
        "id",
        "well",
        str(saved["hmm_prediction_column"]),
        str(saved["hmm_std_column"]),
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"id": str, "well": str})
    frame = frame.rename(
        columns={
            "well": "well_id",
            str(saved["hmm_prediction_column"]): "base_mean",
            str(saved["hmm_std_column"]): "base_std",
        }
    )
    frame["row_idx"] = frame["id"].map(parse_row_index).astype(np.int64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("saved exp209 base path has duplicate row identities")
    numeric = frame[["base_mean", "base_std"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("saved exp209 base path contains non-finite values")
    return frame


def decode_well(
    well: str,
    raw_dir: Path,
    process_row: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
    saved_base: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    horizontal = load_horizontal_without_truth(well, raw_dir)
    mask_manifest: dict[str, Any] = {}
    if stage.startswith("stage_0"):
        horizontal, mask_manifest = stage0_masked_horizontal(horizontal, config)
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_hmm_inputs(horizontal, typewell, config)
    eval_index = np.asarray(prepared["eval_index"], dtype=np.int64)
    runtime_rows: list[dict[str, Any]] = []
    if stage.startswith("stage_0"):
        started = time.time()
        parent = run_exact_hmm(
            prepared,
            config,
            scale_schedule=np.ones(len(eval_index), dtype=np.float64),
            intercept_schedule=np.zeros(len(eval_index), dtype=np.float64),
        )
        runtime_rows.append(
            {
                "well_id": well,
                "run": "masked_exp209_parent",
                "rows": len(eval_index),
                "elapsed_seconds": time.time() - started,
                "loglik": parent["loglik"],
                "posterior_row_sum_max_abs_error": parent["posterior_row_sum_max_abs_error"],
            }
        )
        base_mean = np.asarray(parent["mean"], dtype=np.float64)
        base_std = np.asarray(parent["std"], dtype=np.float64)
    else:
        if saved_base is None:
            raise ValueError("Stage 1 requires the saved exp209 base path")
        base_well = saved_base.loc[saved_base["well_id"] == well].sort_values(
            "row_idx", kind="mergesort"
        )
        if not np.array_equal(base_well["row_idx"].to_numpy(np.int64), eval_index):
            raise ValueError(f"well={well} saved exp209 base rows do not match raw suffix")
        base_mean = base_well["base_mean"].to_numpy(np.float64)
        base_std = base_well["base_std"].to_numpy(np.float64)
        parent = {"mean": base_mean, "std": base_std, "loglik": float("nan")}
    schedule, affine_audit = causal_affine_schedule(
        horizontal,
        typewell,
        prepared,
        base_mean,
        base_std,
        process_row,
        config,
    )
    if not np.array_equal(schedule["row_idx"].to_numpy(np.int64), eval_index):
        raise RuntimeError("affine schedule row identity changed before HMM")
    started = time.time()
    candidate = run_exact_hmm(
        prepared,
        config,
        scale_schedule=schedule["affine_scale_a"].to_numpy(np.float64),
        intercept_schedule=schedule["affine_intercept_b"].to_numpy(np.float64),
    )
    runtime_rows.append(
        {
            "well_id": well,
            "run": VARIANT,
            "rows": len(eval_index),
            "elapsed_seconds": time.time() - started,
            "loglik": candidate["loglik"],
            "posterior_row_sum_max_abs_error": candidate["posterior_row_sum_max_abs_error"],
        }
    )
    prediction = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_index],
            "well_id": well,
            "row_idx": eval_index,
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "parent_hmm_tvt": np.asarray(parent["mean"], dtype=np.float64),
            "parent_hmm_std": np.asarray(parent["std"], dtype=np.float64),
            f"{VARIANT}_hmm_tvt": np.asarray(candidate["mean"], dtype=np.float64),
            f"{VARIANT}_hmm_std": np.asarray(candidate["std"], dtype=np.float64),
        }
    )
    schedule.insert(0, "well_id", well)
    audit = {
        "well_id": well,
        "stage": stage,
        **mask_manifest,
        **{f"prefix_scale_{key}": value for key, value in prepared["prefix_scale"].items()},
        **affine_audit,
        "parent_loglik": parent["loglik"],
        "candidate_loglik": candidate["loglik"],
        "gr_predictive_nll_identity_mean": float(schedule["predictive_nll_identity"].mean()),
        "gr_predictive_nll_affine_mean": float(schedule["predictive_nll_affine"].mean()),
    }
    numeric = prediction.drop(columns=["id", "well_id"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"well={well} produced non-finite predictions")
    return prediction, schedule, audit, pd.DataFrame(runtime_rows), mask_manifest


# %% [markdown]
# ## 7. Target-free staged generation and freeze


# %%
def generate_and_freeze(
    raw_dir: Path,
    artifacts: Path,
    config: Mapping[str, Any],
    stage: str,
    wells: list[str],
    process_noise: pd.DataFrame,
    saved_base: pd.DataFrame | None,
) -> tuple[dict[str, Any], dict[str, Path], pd.DataFrame]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exact exp209 HMM")
    requested_threads = int(get_nested(config, "runtime.numba_num_threads"))
    set_num_threads(requested_threads)
    process_lookup = process_noise.set_index("well_id")
    outer_workers = int(get_nested(config, "runtime.num_workers"))

    def build_one(
        index: int, well: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
        print(f"[{index}/{len(wells)}] exp345 stage={stage} well={well}", flush=True)
        result = decode_well(
            well,
            raw_dir,
            process_lookup.loc[well].to_dict(),
            config,
            stage,
            saved_base,
        )
        print(json.dumps(to_jsonable(result[2]), sort_keys=True), flush=True)
        return result

    if outer_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(build_one)(index, well) for index, well in enumerate(wells, start=1)
        )
    else:
        results = [build_one(index, well) for index, well in enumerate(wells, start=1)]
    prediction = pd.concat([item[0] for item in results], ignore_index=True)
    schedule = pd.concat([item[1] for item in results], ignore_index=True)
    audit = pd.DataFrame([item[2] for item in results])
    runtime = pd.concat([item[3] for item in results], ignore_index=True)
    mask_manifest = pd.DataFrame(
        [{"well_id": wells[index], **item[4]} for index, item in enumerate(results)]
    )
    prediction = prediction.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    schedule = schedule.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if not prediction[["well_id", "row_idx"]].equals(schedule[["well_id", "row_idx"]]):
        raise RuntimeError("prediction and affine schedule identities differ before freeze")
    stage_prefix = f"{OUTPUT_PREFIX}_{stage}"
    paths = {
        "prediction": artifacts / f"{stage_prefix}_predictions.csv.gz",
        "affine_schedule": artifacts / f"{stage_prefix}_affine_schedule.csv.gz",
        "affine_audit": artifacts / f"{stage_prefix}_affine_audit.csv",
        "process_noise": artifacts / f"{stage_prefix}_process_noise.csv",
        "runtime": artifacts / f"{stage_prefix}_runtime.csv",
        "mask_manifest": artifacts / f"{stage_prefix}_mask_manifest.csv",
    }
    reports = {
        "prediction": write_gzip_csv(prediction, paths["prediction"]),
        "affine_schedule": write_gzip_csv(schedule, paths["affine_schedule"]),
    }
    audit.to_csv(paths["affine_audit"], index=False)
    process_noise.to_csv(paths["process_noise"], index=False)
    runtime.to_csv(paths["runtime"], index=False)
    mask_manifest.to_csv(paths["mask_manifest"], index=False)
    for name in ("affine_audit", "process_noise", "runtime", "mask_manifest"):
        reports[name] = {
            "path": str(paths[name]),
            "bytes": paths[name].stat().st_size,
            "raw_sha256": sha256_path(paths[name]),
        }
    freeze = {
        "stage": stage,
        "truth_attached": False,
        "rows": len(prediction),
        "wells": int(prediction["well_id"].nunique()),
        "prediction": reports["prediction"],
        "affine_schedule": reports["affine_schedule"],
        "affine_audit": reports["affine_audit"],
        "process_noise": reports["process_noise"],
        "runtime": reports["runtime"],
        "mask_manifest": reports["mask_manifest"],
        "identity_content_sha256": dataframe_content_sha(prediction, ["well_id", "row_idx"]),
    }
    freeze["freeze_manifest_sha256"] = mapping_sha256(freeze)
    return freeze, paths, runtime


# %% [markdown]
# ## 8. Late truth attachment, diagnostics, and promotion gates


# %%
def role_mask(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.lower()
    return normalized.str.contains("verification", regex=False).to_numpy(bool)


def load_late_readout(
    prediction_path: Path,
    raw_dir: Path,
    controls: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction = pd.read_csv(prediction_path, dtype={"id": str, "well_id": str})
    if stage.startswith("stage_0"):
        truth_rows: list[pd.DataFrame] = []
        for well, group in prediction.groupby("well_id", sort=True):
            source = pd.read_csv(
                raw_dir / f"{well}__horizontal_well.csv",
                usecols=["TVT_input"],
            )
            row_index = group["row_idx"].to_numpy(np.int64)
            truth = pd.to_numeric(source.loc[row_index, "TVT_input"], errors="raise").to_numpy(
                np.float64
            )
            truth_rows.append(
                pd.DataFrame(
                    {
                        "well_id": well,
                        "row_idx": row_index,
                        "true_tvt": truth,
                    }
                )
            )
        truth_frame = pd.concat(truth_rows, ignore_index=True)
    else:
        fold_spec = get_nested(config, "data.fold_assignment") or {}
        truth_columns = [str(value) for value in fold_spec["truth_columns"]]
        fold_path = Path(str(controls["paths"]["fold_assignment"]))
        truth_frame = pd.read_csv(fold_path, usecols=truth_columns, dtype={"well_id": str})
        truth_frame["row_idx"] = pd.to_numeric(truth_frame["row_idx"], errors="raise").astype(
            np.int64
        )
        truth_frame = truth_frame.rename(columns={"tvt_true": "true_tvt"})
    frame = prediction.merge(
        truth_frame,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if frame["true_tvt"].isna().any():
        raise ValueError("late truth attachment left missing values")
    fold_spec = get_nested(config, "data.fold_assignment") or {}
    safe = pd.read_csv(
        Path(str(controls["paths"]["fold_assignment"])),
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
    )
    fold_map = safe.groupby("well_id", sort=True)["fold"].first().astype(int)
    frame["fold"] = frame["well_id"].map(fold_map)
    hidden_spec = get_nested(config, "data.hidden_like_assignment") or {}
    role_columns = [str(value) for value in hidden_spec["role_columns"].values()]
    hidden = pd.read_csv(
        Path(str(controls["paths"]["hidden_like_assignment"])),
        usecols=["well_id", *role_columns],
        dtype={"well_id": str},
    ).set_index("well_id")
    for output_name, source_name in hidden_spec["role_columns"].items():
        role_lookup = hidden[source_name]
        mapped = frame["well_id"].map(role_lookup)
        if mapped.isna().any():
            raise ValueError(f"hidden-like role attachment failed for {output_name}")
        frame[str(output_name)] = role_mask(mapped)
    if stage == "stage_1_full_suffix":
        saved = get_nested(config, "data.saved_exp209") or {}
        likpf = pd.read_csv(
            Path(str(controls["paths"]["saved_likpf"])),
            usecols=[
                "id",
                "well",
                str(saved["likpf_anchor_column"]),
                str(saved["likpf_delta_column"]),
            ],
            dtype={"id": str, "well": str},
        )
        likpf["row_idx"] = likpf["id"].map(parse_row_index).astype(np.int64)
        likpf["likpf_mean_tvt"] = pd.to_numeric(
            likpf[str(saved["likpf_anchor_column"])], errors="raise"
        ) + pd.to_numeric(likpf[str(saved["likpf_delta_column"])], errors="raise")
        likpf = likpf.rename(columns={"well": "well_id"})
        frame = frame.merge(
            likpf[["well_id", "row_idx", "likpf_mean_tvt"]],
            on=["well_id", "row_idx"],
            how="left",
            validate="one_to_one",
        )
        if frame["likpf_mean_tvt"].isna().any():
            raise ValueError("saved LikPF late attachment left missing values")
        frame["parent_likpf_50_50"] = 0.5 * (frame["parent_hmm_tvt"] + frame["likpf_mean_tvt"])
        frame[f"{VARIANT}_likpf_50_50"] = 0.5 * (
            frame[f"{VARIANT}_hmm_tvt"] + frame["likpf_mean_tvt"]
        )
    late = {
        "attached_after_prediction_freeze": True,
        "truth_source": "raw_visible_TVT_input"
        if stage.startswith("stage_0")
        else str(controls["paths"]["fold_assignment"]),
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "identity_content_sha256": dataframe_content_sha(frame, ["well_id", "row_idx"]),
    }
    return frame, late


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def paired_metric(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scope: str,
    *,
    comparison: str,
    parent_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    subset = frame.loc[mask]
    truth = subset["true_tvt"].to_numpy(np.float64)
    parent = subset[parent_column].to_numpy(np.float64)
    candidate = subset[candidate_column].to_numpy(np.float64)
    parent_rmse = rmse(truth, parent)
    candidate_rmse = rmse(truth, candidate)
    return {
        "comparison": comparison,
        "scope": scope,
        "rows": len(subset),
        "wells": int(subset["well_id"].nunique()),
        "parent_rmse": parent_rmse,
        "candidate_rmse": candidate_rmse,
        "improvement_ft": parent_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_parent": candidate_rmse - parent_rmse,
    }


def build_metrics(frame: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(frame), dtype=bool))]
    for fold in sorted(frame["fold"].astype(int).unique()):
        scopes.append((f"fold_{fold}", frame["fold"].to_numpy(np.int64) == fold))
    for column in ("hidden_like_spatial", "hidden_like_typewell_purged"):
        mask = frame[column].to_numpy(bool)
        if mask.any():
            scopes.append((column, mask))
    if stage == "stage_1_full_suffix":
        distance = frame["md_since"].to_numpy(np.float64)
        mask = distance >= 1000.0
        if mask.any():
            scopes.append(("md_since_1000_plus", mask))
    comparisons = [
        (
            "direct",
            "parent_hmm_tvt",
            f"{VARIANT}_hmm_tvt",
        )
    ]
    if stage == "stage_1_full_suffix":
        comparisons.append(
            (
                "fixed_likpf_50_50_diagnostic",
                "parent_likpf_50_50",
                f"{VARIANT}_likpf_50_50",
            )
        )
    paired = pd.DataFrame(
        [
            paired_metric(
                frame,
                mask,
                scope,
                comparison=comparison,
                parent_column=parent_column,
                candidate_column=candidate_column,
            )
            for comparison, parent_column, candidate_column in comparisons
            for scope, mask in scopes
            if mask.any()
        ]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        parent_rmse = rmse(truth, group["parent_hmm_tvt"].to_numpy(np.float64))
        candidate_rmse = rmse(truth, group[f"{VARIANT}_hmm_tvt"].to_numpy(np.float64))
        by_well_rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "parent_rmse": parent_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse_candidate_minus_parent": candidate_rmse - parent_rmse,
            }
        )
    return paired, pd.DataFrame(by_well_rows)


def evaluate_gate(
    paired: pd.DataFrame,
    by_well: pd.DataFrame,
    frame: pd.DataFrame,
    affine_audit: pd.DataFrame,
    runtime: pd.DataFrame,
    runtime_seconds: float,
    stage: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    maximum_hours = float(get_nested(config, "model.runtime_gate.maximum_projected_hours"))
    selected_wells = int(affine_audit["well_id"].nunique())
    if stage == "stage_0_microbenchmark":
        outer_workers = int(get_nested(config, "runtime.num_workers"))
        measured_hmm_parallel_seconds = float(runtime["elapsed_seconds"].sum()) / outer_workers
        projected_seconds = (
            measured_hmm_parallel_seconds
            * int(get_nested(config, "validation.expected_wells"))
            / selected_wells
        )
    else:
        measured_hmm_parallel_seconds = float(runtime["elapsed_seconds"].sum()) / int(
            get_nested(config, "runtime.num_workers")
        )
        projected_seconds = runtime_seconds
    runtime_gate = {
        "actual_seconds": runtime_seconds,
        "measured_hmm_parallel_seconds": measured_hmm_parallel_seconds,
        "projected_full_seconds": projected_seconds,
        "maximum_seconds": maximum_hours * 3600.0,
        "passed": bool(projected_seconds <= maximum_hours * 3600.0),
    }
    expected_wells = (
        int(get_nested(config, "model.runtime_gate.stable_sha256_microbenchmark_wells"))
        if stage == "stage_0_microbenchmark"
        else int(get_nested(config, "validation.expected_wells"))
    )
    expected_hmm_runs = expected_wells * (2 if stage.startswith("stage_0") else 1)
    expected_rows = (
        expected_wells * int(get_nested(config, "validation.prefix_mask_backtest.horizon_rows"))
        if stage.startswith("stage_0")
        else int(get_nested(config, "validation.expected_rows"))
    )
    expected_run_counts = (
        {"masked_exp209_parent": expected_wells, VARIANT: expected_wells}
        if stage.startswith("stage_0")
        else {VARIANT: expected_wells}
    )
    technical = {
        "finite_predictions": True,
        "prediction_rows": len(frame),
        "expected_prediction_rows": expected_rows,
        "prediction_wells": int(frame["well_id"].nunique()),
        "expected_prediction_wells": expected_wells,
        "affine_audit_wells": int(affine_audit["well_id"].nunique()),
        "hmm_runs": len(runtime),
        "expected_hmm_runs": expected_hmm_runs,
        "hmm_run_counts": runtime["run"].value_counts().sort_index().to_dict(),
        "expected_hmm_run_counts": expected_run_counts,
        "posterior_normalization_max_abs_error": float(
            runtime["posterior_row_sum_max_abs_error"].max()
        ),
        "fallback_fraction": float(affine_audit["fallback"].astype(bool).mean()),
        "maximum_fallback_fraction": float(
            get_nested(config, "promotion_gates.maximum_fallback_fraction")
        ),
        "runtime_gate": runtime_gate,
    }
    baseline_parity: dict[str, Any] = {}
    if stage == "stage_1_full_suffix":
        tolerance = float(get_nested(config, "promotion_gates.baseline_metric_absolute_tolerance"))
        direct_overall = paired.loc[
            (paired["comparison"] == "direct") & (paired["scope"] == "overall")
        ].iloc[0]
        blend_overall = paired.loc[
            (paired["comparison"] == "fixed_likpf_50_50_diagnostic")
            & (paired["scope"] == "overall")
        ].iloc[0]
        actual = {
            "exp209_raw_hmm": float(direct_overall["parent_rmse"]),
            "exp209_likpf": rmse(
                frame["true_tvt"].to_numpy(np.float64),
                frame["likpf_mean_tvt"].to_numpy(np.float64),
            ),
            "exp209_hmm_likpf_50_50": float(blend_overall["parent_rmse"]),
        }
        expected = {
            "exp209_raw_hmm": float(get_nested(config, "references.exp209_raw_hmm_rmse")),
            "exp209_likpf": float(get_nested(config, "references.exp209_likpf_rmse")),
            "exp209_hmm_likpf_50_50": float(
                get_nested(config, "references.exp209_hmm_likpf_50_50_rmse")
            ),
        }
        baseline_parity = {
            name: {
                "actual": actual[name],
                "expected": expected[name],
                "absolute_difference": abs(actual[name] - expected[name]),
                "tolerance": tolerance,
                "passed": bool(abs(actual[name] - expected[name]) <= tolerance),
            }
            for name in actual
        }
        technical["baseline_metric_parity"] = baseline_parity
    technical["passed"] = bool(
        technical["prediction_rows"] == technical["expected_prediction_rows"]
        and technical["prediction_wells"] == technical["expected_prediction_wells"]
        and technical["affine_audit_wells"] == technical["expected_prediction_wells"]
        and technical["hmm_runs"] == technical["expected_hmm_runs"]
        and technical["hmm_run_counts"] == technical["expected_hmm_run_counts"]
        and technical["posterior_normalization_max_abs_error"] <= 1.0e-6
        and (
            stage == "stage_0_microbenchmark"
            or technical["fallback_fraction"] <= technical["maximum_fallback_fraction"]
        )
        and runtime_gate["passed"]
        and all(bool(record["passed"]) for record in baseline_parity.values())
    )
    if stage == "stage_0_microbenchmark":
        return {
            "stage": stage,
            "passed": technical["passed"],
            "decision": "runtime_gate_passed_wait_for_stage_0_approval"
            if technical["passed"]
            else "runtime_gate_failed_close_without_full_stage_0",
            "technical_gate": technical,
            "scientific_gate": None,
        }
    direct = paired.loc[paired["comparison"] == "direct"]
    overall = direct.loc[direct["scope"] == "overall"].iloc[0]
    folds = direct.loc[direct["scope"].str.startswith("fold_")]
    folds_improved = int((folds["improvement_ft"] > 0.0).sum())
    hidden_checks = {
        scope: bool(
            direct.loc[direct["scope"] == scope, "delta_rmse_candidate_minus_parent"].iloc[0] <= 0.0
        )
        for scope in ("hidden_like_spatial", "hidden_like_typewell_purged")
        if (direct["scope"] == scope).any()
    }
    p95_delta = float(
        by_well["candidate_rmse"].quantile(0.95) - by_well["parent_rmse"].quantile(0.95)
    )
    worst_delta = float(by_well["delta_rmse_candidate_minus_parent"].max())
    nll_weights = pd.to_numeric(
        affine_audit.get("score_rows", pd.Series(np.ones(len(affine_audit)))),
        errors="coerce",
    ).to_numpy(np.float64)
    identity_values = pd.to_numeric(
        affine_audit["gr_predictive_nll_identity_mean"], errors="coerce"
    ).to_numpy(np.float64)
    affine_values = pd.to_numeric(
        affine_audit["gr_predictive_nll_affine_mean"], errors="coerce"
    ).to_numpy(np.float64)
    valid_identity = np.isfinite(identity_values) & np.isfinite(nll_weights) & (nll_weights > 0.0)
    valid_affine = np.isfinite(affine_values) & np.isfinite(nll_weights) & (nll_weights > 0.0)
    if not valid_identity.any() or not valid_affine.any():
        raise ValueError("GR predictive NLL has no finite weighted wells")
    gr_nll_identity = float(
        np.average(identity_values[valid_identity], weights=nll_weights[valid_identity])
    )
    gr_nll_affine = float(
        np.average(affine_values[valid_affine], weights=nll_weights[valid_affine])
    )
    boundary_p95 = float(affine_audit["boundary_jump_sigma"].quantile(0.95))
    science = {
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            get_nested(config, "promotion_gates.minimum_rmse_gain_vs_exp209_raw_hmm_ft")
        ),
        "folds_improved": folds_improved,
        "minimum_folds_improved": int(get_nested(config, "promotion_gates.minimum_improved_folds")),
        "gr_predictive_nll_identity": gr_nll_identity,
        "gr_predictive_nll_affine": gr_nll_affine,
        "gr_nll_improved": bool(gr_nll_affine < gr_nll_identity),
        "boundary_jump_p95_sigma": boundary_p95,
        "boundary_jump_max_sigma": 3.0,
        "hidden_like_checks": hidden_checks,
        "by_well_p95_delta": p95_delta,
        "worst_well_delta": worst_delta,
        "maximum_worst_well_delta": float(
            get_nested(config, "promotion_gates.maximum_worst_well_regression_ft")
        ),
    }
    stage1_extra = True
    if stage == "stage_1_full_suffix":
        thousand = direct.loc[direct["scope"] == "md_since_1000_plus"]
        stage1_extra = bool(
            not thousand.empty
            and float(thousand["delta_rmse_candidate_minus_parent"].iloc[0]) <= 0.0
            and p95_delta <= 0.0
        )
        science["md_since_1000_plus_non_regression"] = bool(
            not thousand.empty
            and float(thousand["delta_rmse_candidate_minus_parent"].iloc[0]) <= 0.0
        )
        science["by_well_p95_non_regression"] = bool(p95_delta <= 0.0)
    science["passed"] = bool(
        science["improvement_ft"] >= science["minimum_improvement_ft"]
        and folds_improved >= science["minimum_folds_improved"]
        and science["gr_nll_improved"]
        and boundary_p95 <= science["boundary_jump_max_sigma"]
        and len(hidden_checks) == 2
        and all(hidden_checks.values())
        and worst_delta <= science["maximum_worst_well_delta"]
        and stage1_extra
    )
    passed = bool(technical["passed"] and science["passed"])
    return {
        "stage": stage,
        "passed": passed,
        "decision": "stage_passed_wait_for_separate_approval"
        if passed
        else "stage_failed_close_without_rescue",
        "technical_gate": technical,
        "scientific_gate": science,
    }


# %% [markdown]
# ## 9. Experiment orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp345 must run first on Kaggle; local execution requires explicit smoke approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    stage = selected_stage(config)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    controls = preflight_controls_and_assignments(config)
    safe = pd.read_csv(
        Path(str(controls["paths"]["fold_assignment"])),
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
    )
    fold_map = safe.groupby("well_id", sort=True)["fold"].first().astype(int).to_dict()
    if sorted(fold_map) != sorted(raw_preflight["well_ids"]):
        raise ValueError("raw and fold-assignment well identities differ")
    process_noise = build_outer_fold_process_noise(raw_dir, fold_map, config, stage)
    all_wells = sorted(fold_map, key=lambda well: (fold_map[well], well))
    if stage == "stage_0_microbenchmark":
        count = int(get_nested(config, "model.runtime_gate.stable_sha256_microbenchmark_wells"))
        wells = stable_sha_well_order(all_wells)[:count]
    else:
        wells = all_wells
    saved_base = (
        load_saved_exp209_base(Path(str(controls["paths"]["saved_hmm"])), config)
        if stage == "stage_1_full_suffix"
        else None
    )
    scientific_contract = build_scientific_contract(config, stage)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_{stage}_scientific_contract.json"
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_{stage}_input_manifest.json"
    write_json(contract_path, scientific_contract)
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": stage,
        "truth_attached": False,
        "raw_train": {key: value for key, value in raw_preflight.items() if key != "well_ids"},
        "controls": controls,
        "selected_wells": wells,
        "selected_well_order_sha256": mapping_sha256({"wells": wells}),
    }
    write_json(manifest_path, input_manifest)
    freeze, paths, runtime = generate_and_freeze(
        raw_dir,
        artifacts,
        config,
        stage,
        wells,
        process_noise,
        saved_base,
    )
    prediction_frozen_at_seconds = time.time() - started
    frame, late_attachment = load_late_readout(
        paths["prediction"],
        raw_dir,
        controls,
        config,
        stage,
    )
    paired, by_well = build_metrics(frame, stage)
    affine_audit = pd.read_csv(paths["affine_audit"], dtype={"well_id": str})
    runtime_seconds = time.time() - started
    gate = evaluate_gate(
        paired,
        by_well,
        frame,
        affine_audit,
        runtime,
        runtime_seconds,
        stage,
        config,
    )
    output_paths = {
        "paired_metrics": artifacts / f"{OUTPUT_PREFIX}_{stage}_paired_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_{stage}_by_well_metrics.csv",
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_{stage}_promotion_gate.json",
    }
    paired.to_csv(output_paths["paired_metrics"], index=False)
    by_well.to_csv(output_paths["by_well_metrics"], index=False)
    write_json(output_paths["promotion_gate"], gate)
    status = (
        f"{stage}_passed_wait_for_separate_approval" if gate["passed"] else f"{stage}_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": stage,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "runtime_seconds": runtime_seconds,
        "prediction_frozen_at_seconds": prediction_frozen_at_seconds,
        "scientific_variants": 1,
        "hmm_well_runs": len(runtime),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": int((runtime["run"] == "masked_exp209_parent").sum()),
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "input_manifest_sha256": sha256_path(manifest_path),
        "freeze": freeze,
        "truth_attachment": late_attachment,
        "promotion_gate": gate,
        "runtime_versions": runtime_versions(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": None,
        "submission_sha256": None,
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_{stage}_summary.json"
    write_json(summary_path, summary)
    overall = (
        paired.loc[(paired["comparison"] == "direct") & (paired["scope"] == "overall")]
        .iloc[0]
        .to_dict()
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": stage,
        "cv": float(overall["candidate_rmse"])
        if gate["passed"] and stage != "stage_0_microbenchmark"
        else None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": overall,
        "promotion_gate": gate,
        "prediction_sha256": freeze["prediction"],
        "affine_schedule_sha256": freeze["affine_schedule"],
        "model_sha256": None,
        "submission_sha256": None,
        "notes": "Train-side staged audit only; no inference or submission is generated.",
    }
    write_json(metrics_output_path(), metrics)
    print(paired.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup and configuration preview

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "variant": VARIANT,
                "implementation_complete": get_nested(CONFIG, "implementation.enabled"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "active_stage": get_nested(CONFIG, "execution.active_stage"),
                "run_flags": {
                    "microbenchmark": get_nested(CONFIG, "execution.run_microbenchmark"),
                    "stage_0": get_nested(CONFIG, "execution.run_stage_0"),
                    "stage_1": get_nested(CONFIG, "execution.run_stage_1"),
                },
                "hmm_fixed": get_nested(CONFIG, "model.fixed_exp209_hmm"),
                "affine_state": get_nested(CONFIG, "model.affine_state"),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 11. Run the approved Kaggle CPU stage

# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_full_experiment(CONFIG)
