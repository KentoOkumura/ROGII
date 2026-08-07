# %% [markdown]
# # exp350 exp345 bidirectional GR affine smoother train
#
# Train-side Stage 0 audit of one deterministic full-well bidirectional affine
# smoother. Saved exp345 controls and the exp209 exact-HMM grammar are fixed.

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


EXPERIMENT_NAME = "exp350_exp345_bidirectional_gr_affine_smoother"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT = "one_pass_bidirectional_rts_affine_schedule_on_exp209"
CAUSAL_VARIANT = "one_pass_causal_affine_schedule_on_exp209"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP350_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp350 config not found in {[str(path) for path in candidates]}")


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
        "lineage.parent": "exp345_exp209_time_varying_gr_affine_calibration_hmm",
        "implementation.enabled": True,
        "model.active_variants": [VARIANT],
        "model.base_path.calibration_iterations": 1,
        "model.base_path.parent_control_rerun": False,
        "model.base_path.causal_control_rerun": False,
        "model.forward_affine_state.state": ["intercept_b", "log_scale_a"],
        "model.forward_affine_state.transition": "local_level_random_walk",
        "model.forward_affine_state.minimum_prefix_pairs": 40,
        "model.forward_affine_state.minimum_typewell_gr_std": 5.0,
        "model.forward_affine_state.maximum_prefix_rmse": 60.0,
        "model.forward_affine_state.trim_quantile": 0.90,
        "model.forward_affine_state.robust_iterations": 2,
        "model.forward_affine_state.covariance_update": "joseph_form",
        "model.forward_affine_state.missing_raw_gr_update_policy": (
            "skip_update_propagate_state"
        ),
        "model.bidirectional_smoother.algorithm": (
            "deterministic_fixed_interval_extended_rts"
        ),
        "model.bidirectional_smoother.transition_matrix": "identity_2x2",
        "model.bidirectional_smoother.pseudoinverse_rcond": 1.0e-12,
        "model.bidirectional_smoother.backward_order": (
            "last_score_row_to_first_score_row"
        ),
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
            raise ValueError(f"exp350 fixed contract mismatch: {key} must be {value!r}")
    bounds = [
        float(value)
        for value in get_nested(config, "model.forward_affine_state.slope_bounds")
    ]
    if bounds != [0.25, 4.0]:
        raise ValueError("exp350 fixes affine slope bounds to [0.25, 4.0]")
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
        "effective_position_sigma_floor": 0.1225,
    }
    for key, value in fixed_numeric.items():
        if float(hmm.get(key, -1.0)) != value:
            raise ValueError(f"exp350 fixes model.fixed_exp209_hmm.{key}={value}")
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
            raise ValueError(f"exp350 fixes model.fixed_exp209_hmm.{key}={value!r}")
    if [float(value) for value in hmm.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("exp350 fixes exp209 sigma clip to [10, 60]")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise RuntimeError("exp350 implementation approval must be recorded")
    if require_run_approval:
        for key in (
            "execution.canonical_notebook_adoption_approved",
            "execution.kaggle_package_approved",
            "execution.kaggle_push_approved",
            "execution.run_stage_0",
        ):
            if not bool(get_nested(config, key)):
                raise RuntimeError(f"exp350 run approval is missing: {key}")
        if bool(get_nested(config, "execution.run_stage_1")):
            raise RuntimeError("Stage 1 is not approved for this run")


def selected_stage(config: Mapping[str, Any]) -> str:
    if not bool(get_nested(config, "execution.run_stage_0")):
        raise RuntimeError("exp350 Stage 0 must be enabled")
    if bool(get_nested(config, "execution.run_stage_1")):
        raise RuntimeError("exp350 Stage 1 must remain disabled")
    return "stage_0_full"


def build_scientific_contract(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "stage": stage,
        "truth_attached": False,
        "variant": VARIANT,
        "base_path": get_nested(config, "model.base_path"),
        "forward_affine_state": get_nested(config, "model.forward_affine_state"),
        "bidirectional_smoother": get_nested(config, "model.bidirectional_smoother"),
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
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    actual = dataframe_content_sha(
        frame, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    )
    expected = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("current raw train well count mismatch")
    if actual != expected:
        raise ValueError("current raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": actual,
        "well_ids": frame["well_id"].tolist(),
    }


def inspect_plain_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }


def preflight_controls_and_assignments(config: Mapping[str, Any]) -> dict[str, Any]:
    saved = get_nested(config, "data.saved_exp345") or {}
    candidates = _candidate_paths(saved)
    file_specs = {
        "promotion_gate": (
            "promotion_gate_filename",
            "expected_promotion_gate_raw_sha256",
            False,
        ),
        "prediction": (
            "prediction_filename",
            "expected_prediction_raw_sha256",
            True,
        ),
        "affine_schedule": (
            "affine_schedule_filename",
            "expected_affine_schedule_raw_sha256",
            True,
        ),
        "process_noise": (
            "process_noise_filename",
            "expected_process_noise_raw_sha256",
            False,
        ),
        "mask_manifest": (
            "mask_manifest_filename",
            "expected_mask_manifest_raw_sha256",
            False,
        ),
        "paired_metrics": (
            "paired_metrics_filename",
            "expected_paired_metrics_raw_sha256",
            False,
        ),
    }
    paths: dict[str, str] = {}
    reports: dict[str, Any] = {}
    for name, (filename_key, raw_sha_key, compressed) in file_specs.items():
        path = resolve_existing(str(saved[filename_key]), candidates)
        report = inspect_gzip_csv(path) if compressed else inspect_plain_file(path)
        if report["raw_sha256"] != str(saved[raw_sha_key]):
            raise ValueError(f"saved exp345 {name} raw SHA mismatch")
        paths[f"saved_{name}"] = str(path)
        reports[name] = report
    for name, expected_key in (
        ("prediction", "expected_prediction_decompressed_sha256"),
        ("affine_schedule", "expected_affine_schedule_decompressed_sha256"),
    ):
        if reports[name]["decompressed_sha256"] != str(saved[expected_key]):
            raise ValueError(f"saved exp345 {name} decompressed SHA mismatch")
    gate = json.loads(Path(paths["saved_promotion_gate"]).read_text())
    if gate.get("stage") != "stage_0_full" or gate.get("decision") != (
        "stage_failed_close_without_rescue"
    ):
        raise ValueError("saved exp345 Stage 0 gate identity mismatch")
    fold = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(str(fold["filename"]), _candidate_paths(fold))
    fold_report = inspect_gzip_csv(fold_path)
    if fold_report["decompressed_sha256"] != str(fold["expected_decompressed_sha256"]):
        raise ValueError("fold assignment decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold["safe_columns"]]
    safe = pd.read_csv(fold_path, usecols=safe_columns, dtype={"well_id": str})
    for column in ("row_idx", "suffix_offset", "fold"):
        safe[column] = pd.to_numeric(safe[column], errors="raise").astype(np.int64)
    safe = safe.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if safe.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("fold assignment identity is duplicated")
    fold_counts = safe.groupby("well_id", sort=True)["fold"].nunique()
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    if (
        len(safe) != int(get_nested(config, "validation.expected_raw_rows"))
        or safe["well_id"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
        or sorted(safe["fold"].unique().tolist()) != expected_folds
        or not fold_counts.eq(1).all()
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
    paths["fold_assignment"] = str(fold_path)
    paths["hidden_like_assignment"] = str(hidden_path)
    return {
        "paths": paths,
        "saved_exp345": reports,
        "saved_exp345_gate": gate,
        "fold_assignment": {
            **fold_report,
            "well_ids": sorted(fold_map.index.astype(str).tolist()),
        },
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
    spec = get_nested(config, "model.forward_affine_state") or {}
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
    minimum = int(get_nested(config, "model.forward_affine_state.minimum_prefix_pairs"))
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
    floor = float(get_nested(config, "model.forward_affine_state.process_noise_numerical_floor"))
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
# ## 6. Prefix masking, forward parity, RTS smoothing, and exact-HMM decoding


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
    observation_variance = sigma_gr**2 + (suffix_gradient * base_std) ** 2
    if (
        not np.isfinite(observation_variance).all()
        or np.any(observation_variance <= 0.0)
    ):
        raise ValueError("affine EKF observation variance is invalid")
    length = len(eval_index)
    floor = float(
        get_nested(config, "model.forward_affine_state.process_noise_numerical_floor")
    )
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
        frame = pd.DataFrame(
            {
                "row_idx": eval_index,
                "affine_scale_a": np.ones(length, dtype=np.float64),
                "affine_intercept_b": np.zeros(length, dtype=np.float64),
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
            }
        )
        audit = {
            **{key: value for key, value in fit.items() if key != "covariance"},
            "fallback": True,
            "q_intercept": float(process_row["q_intercept"]),
            "q_log_scale": float(process_row["q_log_scale"]),
            "forward_boundary_jump_sigma": 0.0,
            "finite_updates": int(updated.sum()),
        }
        context = {
            "initial_state": state.copy(),
            "initial_covariance": covariance.copy(),
            "suffix_x": suffix_x,
            "raw_gr": raw_gr,
        }
        return frame, audit, context
    state = np.array([fit["intercept_b"], fit["log_scale_a"]], dtype=np.float64)
    covariance = np.asarray(fit["covariance"], dtype=np.float64)
    covariance = covariance + np.eye(2, dtype=np.float64) * floor
    slope_low, slope_high = (
        float(value)
        for value in get_nested(config, "model.forward_affine_state.slope_bounds")
    )
    boundary_jump = 0.0
    identity = np.eye(2, dtype=np.float64)
    initial_state = state.copy()
    initial_covariance = covariance.copy()
    for index in range(length):
        state_pred = state.copy()
        covariance_pred = covariance + process_covariance
        predicted_state[index] = state_pred
        predicted_covariance[index] = covariance_pred
        scale_pred = float(math.exp(state_pred[1]))
        predicted_y = float(state_pred[0] + scale_pred * suffix_x[index])
        h = np.array([1.0, scale_pred * suffix_x[index]], dtype=np.float64)
        innovation_variance = float(
            h @ covariance_pred @ h + observation_variance[index]
        )
        if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise ValueError("affine EKF innovation variance is invalid")
        if updated[index]:
            residual = float(raw_gr[index] - predicted_y)
            identity_residual = float(raw_gr[index] - suffix_x[index])
            forward_nll[index] = 0.5 * (
                math.log(2.0 * math.pi * innovation_variance)
                + residual**2 / innovation_variance
            )
            identity_nll[index] = 0.5 * (
                math.log(2.0 * math.pi * observation_variance[index])
                + identity_residual**2 / observation_variance[index]
            )
            gain = covariance_pred @ h / innovation_variance
            state = state_pred + gain * residual
            state[1] = float(
                np.clip(state[1], math.log(slope_low), math.log(slope_high))
            )
            kh = np.outer(gain, h)
            covariance = (
                (identity - kh)
                @ covariance_pred
                @ (identity - kh).T
                + np.outer(gain, gain) * observation_variance[index]
            )
            covariance = 0.5 * (covariance + covariance.T)
        else:
            state = state_pred
            covariance = covariance_pred
        if index == 0:
            delta = state - initial_state
            boundary_jump = float(
                math.sqrt(
                    max(
                        0.0,
                        delta @ np.linalg.pinv(covariance_pred) @ delta,
                    )
                )
            )
        filtered_state[index] = state
        filtered_covariance[index] = covariance
    frame = pd.DataFrame(
        {
            "row_idx": eval_index,
            "affine_scale_a": np.exp(filtered_state[:, 1]),
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
        }
    )
    audit = {
        **{key: value for key, value in fit.items() if key != "covariance"},
        "fallback": False,
        "q_intercept": float(process_row["q_intercept"]),
        "q_log_scale": float(process_row["q_log_scale"]),
        "forward_boundary_jump_sigma": boundary_jump,
        "finite_updates": int(updated.sum()),
    }
    context = {
        "initial_state": initial_state,
        "initial_covariance": initial_covariance,
        "suffix_x": suffix_x,
        "raw_gr": raw_gr,
    }
    return frame, audit, context


def forward_schedule_parity(
    regenerated: pd.DataFrame,
    saved: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    if not np.array_equal(
        regenerated["row_idx"].to_numpy(np.int64),
        saved["row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("saved and regenerated exp345 schedule identities differ")
    maximum = 0.0
    per_column: dict[str, float] = {}
    for column in (
        "affine_scale_a",
        "affine_intercept_b",
        "predictive_nll_identity",
        "predictive_nll_affine",
    ):
        left = pd.to_numeric(regenerated[column], errors="coerce").to_numpy(np.float64)
        right = pd.to_numeric(saved[column], errors="coerce").to_numpy(np.float64)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            raise ValueError(f"forward parity NaN pattern mismatch: {column}")
        finite = np.isfinite(left) & np.isfinite(right)
        delta = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        per_column[column] = delta
        maximum = max(maximum, delta)
    update_match = bool(
        np.array_equal(
            regenerated["raw_gr_update"].astype(bool).to_numpy(),
            saved["raw_gr_update"].astype(bool).to_numpy(),
        )
    )
    passed = bool(maximum <= tolerance and update_match)
    report = {
        "maximum_absolute_difference": maximum,
        "per_column_maximum_absolute_difference": per_column,
        "raw_gr_update_exact_match": update_match,
        "tolerance": tolerance,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"exp345 forward schedule parity failed: {report}")
    return report


def project_covariance(
    covariance: np.ndarray, floor: float
) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum = float(eigenvalues.min())
    if minimum < 0.0:
        symmetric = (
            eigenvectors
            @ np.diag(np.maximum(eigenvalues, floor))
            @ eigenvectors.T
        )
        symmetric = 0.5 * (symmetric + symmetric.T)
    return symmetric, minimum


def bidirectional_rts_schedule(
    forward: pd.DataFrame,
    forward_audit: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    length = len(forward)
    filtered_state = forward[
        ["filtered_intercept_b", "filtered_log_scale_a"]
    ].to_numpy(np.float64)
    predicted_state = forward[
        ["predicted_intercept_b", "predicted_log_scale_a"]
    ].to_numpy(np.float64)
    filtered_covariance = np.zeros((length, 2, 2), dtype=np.float64)
    predicted_covariance = np.zeros((length, 2, 2), dtype=np.float64)
    for target, prefix in (
        (filtered_covariance, "filtered"),
        (predicted_covariance, "predicted"),
    ):
        target[:, 0, 0] = forward[f"{prefix}_p00"].to_numpy(np.float64)
        target[:, 0, 1] = forward[f"{prefix}_p01"].to_numpy(np.float64)
        target[:, 1, 0] = target[:, 0, 1]
        target[:, 1, 1] = forward[f"{prefix}_p11"].to_numpy(np.float64)
    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    smoother = get_nested(config, "model.bidirectional_smoother") or {}
    rcond = float(smoother["pseudoinverse_rcond"])
    floor = float(smoother["covariance_numerical_floor"])
    negative_tolerance = float(smoother["covariance_negative_eigen_tolerance"])
    minimum_eigenvalue = float("inf")
    contraction_maximum = float("-inf")
    fallback = bool(forward_audit["fallback"])
    if not fallback:
        for index in range(length - 2, -1, -1):
            gain = filtered_covariance[index] @ np.linalg.pinv(
                predicted_covariance[index + 1], rcond=rcond
            )
            smoothed_state[index] = (
                filtered_state[index]
                + gain @ (smoothed_state[index + 1] - predicted_state[index + 1])
            )
            raw_covariance = (
                filtered_covariance[index]
                + gain
                @ (smoothed_covariance[index + 1] - predicted_covariance[index + 1])
                @ gain.T
            )
            smoothed_covariance[index], raw_minimum = project_covariance(
                raw_covariance, floor
            )
            minimum_eigenvalue = min(minimum_eigenvalue, raw_minimum)
            contraction = np.linalg.eigvalsh(
                smoothed_covariance[index] - filtered_covariance[index]
            )
            contraction_maximum = max(
                contraction_maximum, float(contraction.max())
            )
    for index in range(length):
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            float(np.linalg.eigvalsh(smoothed_covariance[index]).min()),
        )
        contraction_maximum = max(
            contraction_maximum,
            float(
                np.linalg.eigvalsh(
                    smoothed_covariance[index] - filtered_covariance[index]
                ).max()
            ),
        )
    slope_low, slope_high = (
        float(value)
        for value in get_nested(config, "model.forward_affine_state.slope_bounds")
    )
    scale_unclipped = np.exp(smoothed_state[:, 1])
    scale = np.clip(scale_unclipped, slope_low, slope_high)
    clip_mask = scale != scale_unclipped
    raw_gr = np.asarray(context["raw_gr"], dtype=np.float64)
    suffix_x = np.asarray(context["suffix_x"], dtype=np.float64)
    observation_variance = forward["observation_variance"].to_numpy(np.float64)
    reconstruction_nll = np.full(length, np.nan, dtype=np.float64)
    finite = np.isfinite(raw_gr)
    residual = raw_gr[finite] - (
        scale[finite] * suffix_x[finite] + smoothed_state[finite, 0]
    )
    reconstruction_nll[finite] = 0.5 * (
        np.log(2.0 * np.pi * observation_variance[finite])
        + residual**2 / observation_variance[finite]
    )
    initial_state = np.asarray(context["initial_state"], dtype=np.float64)
    first_delta = smoothed_state[0] - initial_state
    boundary_jump = float(
        math.sqrt(
            max(
                0.0,
                first_delta
                @ np.linalg.pinv(predicted_covariance[0], rcond=rcond)
                @ first_delta,
            )
        )
    )
    terminal_state_error = float(
        np.max(np.abs(smoothed_state[-1] - filtered_state[-1]))
    )
    terminal_covariance_error = float(
        np.max(np.abs(smoothed_covariance[-1] - filtered_covariance[-1]))
    )
    frame = pd.DataFrame(
        {
            "row_idx": forward["row_idx"].to_numpy(np.int64),
            "forward_affine_scale_a": forward["affine_scale_a"].to_numpy(
                np.float64
            ),
            "forward_affine_intercept_b": forward[
                "affine_intercept_b"
            ].to_numpy(np.float64),
            "affine_scale_a": scale,
            "affine_intercept_b": smoothed_state[:, 0],
            "smoothed_log_scale_a": smoothed_state[:, 1],
            "smoothed_p00": smoothed_covariance[:, 0, 0],
            "smoothed_p01": smoothed_covariance[:, 0, 1],
            "smoothed_p11": smoothed_covariance[:, 1, 1],
            "raw_gr_update": forward["raw_gr_update"].astype(bool).to_numpy(),
            "predictive_nll_identity": forward[
                "predictive_nll_identity"
            ].to_numpy(np.float64),
            "predictive_nll_forward": forward[
                "predictive_nll_affine"
            ].to_numpy(np.float64),
            "gr_reconstruction_nll_smoother": reconstruction_nll,
        }
    )
    audit = {
        "smoother_fallback_identity": fallback,
        "terminal_state_max_abs_error": terminal_state_error,
        "terminal_covariance_max_abs_error": terminal_covariance_error,
        "covariance_minimum_eigenvalue_before_floor": minimum_eigenvalue,
        "covariance_negative_eigen_tolerance": negative_tolerance,
        "covariance_contraction_max_positive_eigenvalue": contraction_maximum,
        "output_scale_clip_rows": int(clip_mask.sum()),
        "output_scale_clip_fraction": float(clip_mask.mean()),
        "boundary_jump_sigma": boundary_jump,
        "gr_reconstruction_nll_smoother_mean": float(
            np.nanmean(reconstruction_nll)
        ),
    }
    return frame, audit


def load_saved_exp345_frames(
    controls: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    paths = controls["paths"]
    prediction = pd.read_csv(
        Path(str(paths["saved_prediction"])),
        dtype={"id": str, "well_id": str},
    )
    schedule = pd.read_csv(
        Path(str(paths["saved_affine_schedule"])), dtype={"well_id": str}
    )
    process_noise = pd.read_csv(
        Path(str(paths["saved_process_noise"])), dtype={"well_id": str}
    )
    mask_manifest = pd.read_csv(
        Path(str(paths["saved_mask_manifest"])), dtype={"well_id": str}
    )
    for frame in (prediction, schedule):
        frame["row_idx"] = pd.to_numeric(
            frame["row_idx"], errors="raise"
        ).astype(np.int64)
        frame.sort_values(
            ["well_id", "row_idx"], kind="mergesort", inplace=True
        )
        frame.reset_index(drop=True, inplace=True)
        if frame.duplicated(["well_id", "row_idx"]).any():
            raise ValueError("saved exp345 control contains duplicate row identities")
    if not prediction[["well_id", "row_idx"]].equals(
        schedule[["well_id", "row_idx"]]
    ):
        raise ValueError("saved exp345 prediction and schedule identities differ")
    expected_rows = int(get_nested(config, "validation.expected_stage_0_score_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or len(schedule) != expected_rows
        or process_noise["well_id"].nunique() != expected_wells
        or mask_manifest["well_id"].nunique() != expected_wells
    ):
        raise ValueError("saved exp345 Stage 0 control coverage mismatch")
    if process_noise["well_id"].duplicated().any():
        raise ValueError("saved exp345 process-noise table has duplicate wells")
    if mask_manifest["well_id"].duplicated().any():
        raise ValueError("saved exp345 mask manifest has duplicate wells")
    required_prediction = [
        "parent_hmm_tvt",
        "parent_hmm_std",
        f"{CAUSAL_VARIANT}_hmm_tvt",
        f"{CAUSAL_VARIANT}_hmm_std",
    ]
    if not np.isfinite(
        prediction[required_prediction].to_numpy(np.float64)
    ).all():
        raise ValueError("saved exp345 predictions contain non-finite values")
    return {
        "prediction": prediction,
        "schedule": schedule,
        "process_noise": process_noise,
        "mask_manifest": mask_manifest,
    }


def decode_well(
    well: str,
    raw_dir: Path,
    process_row: Mapping[str, Any],
    saved_prediction: pd.DataFrame,
    saved_schedule: pd.DataFrame,
    saved_mask: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    horizontal = load_horizontal_without_truth(well, raw_dir)
    horizontal, mask_manifest = stage0_masked_horizontal(horizontal, config)
    for key in (
        "score_start_row",
        "score_stop_row_exclusive",
        "score_rows",
        "visible_prefix_rows",
        "original_known_prefix_rows",
    ):
        if int(mask_manifest[key]) != int(saved_mask[key]):
            raise ValueError(f"well={well} saved exp345 mask mismatch: {key}")
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_hmm_inputs(horizontal, typewell, config)
    eval_index = np.asarray(prepared["eval_index"], dtype=np.int64)
    saved_prediction = saved_prediction.sort_values("row_idx", kind="mergesort")
    saved_schedule = saved_schedule.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(
        saved_prediction["row_idx"].to_numpy(np.int64), eval_index
    ):
        raise ValueError(f"well={well} saved exp345 prediction rows differ")
    if not np.array_equal(
        saved_schedule["row_idx"].to_numpy(np.int64), eval_index
    ):
        raise ValueError(f"well={well} saved exp345 schedule rows differ")
    base_mean = saved_prediction["parent_hmm_tvt"].to_numpy(np.float64)
    base_std = saved_prediction["parent_hmm_std"].to_numpy(np.float64)
    forward_started = time.time()
    forward, forward_audit, context = forward_affine_schedule(
        horizontal,
        typewell,
        prepared,
        base_mean,
        base_std,
        process_row,
        config,
    )
    parity = forward_schedule_parity(
        forward,
        saved_schedule,
        float(
            get_nested(
                config, "promotion_gates.technical.forward_schedule_max_abs_tolerance"
            )
        ),
    )
    forward_seconds = time.time() - forward_started
    smoother_started = time.time()
    smoothed, smoother_audit = bidirectional_rts_schedule(
        forward, forward_audit, context, config
    )
    smoother_seconds = time.time() - smoother_started
    hmm_started = time.time()
    candidate = run_exact_hmm(
        prepared,
        config,
        scale_schedule=smoothed["affine_scale_a"].to_numpy(np.float64),
        intercept_schedule=smoothed["affine_intercept_b"].to_numpy(np.float64),
    )
    hmm_seconds = time.time() - hmm_started
    prediction = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_index],
            "well_id": well,
            "row_idx": eval_index,
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "parent_hmm_tvt": base_mean,
            "parent_hmm_std": base_std,
            "causal_hmm_tvt": saved_prediction[
                f"{CAUSAL_VARIANT}_hmm_tvt"
            ].to_numpy(np.float64),
            "causal_hmm_std": saved_prediction[
                f"{CAUSAL_VARIANT}_hmm_std"
            ].to_numpy(np.float64),
            f"{VARIANT}_hmm_tvt": np.asarray(candidate["mean"], dtype=np.float64),
            f"{VARIANT}_hmm_std": np.asarray(candidate["std"], dtype=np.float64),
        }
    )
    forward.insert(0, "well_id", well)
    smoothed.insert(0, "well_id", well)
    runtime = pd.DataFrame(
        [
            {
                "well_id": well,
                "run": VARIANT,
                "rows": len(eval_index),
                "forward_seconds": forward_seconds,
                "smoother_seconds": smoother_seconds,
                "elapsed_seconds": hmm_seconds,
                "total_well_seconds": (
                    forward_seconds + smoother_seconds + hmm_seconds
                ),
                "loglik": candidate["loglik"],
                "posterior_row_sum_max_abs_error": candidate[
                    "posterior_row_sum_max_abs_error"
                ],
            }
        ]
    )
    audit = {
        "well_id": well,
        "stage": stage,
        **mask_manifest,
        **{
            f"prefix_scale_{key}": value
            for key, value in prepared["prefix_scale"].items()
        },
        **forward_audit,
        **{f"forward_parity_{key}": value for key, value in parity.items()},
        **smoother_audit,
        "candidate_loglik": candidate["loglik"],
    }
    numeric = prediction.drop(columns=["id", "well_id"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"well={well} produced non-finite predictions")
    return prediction, forward, smoothed, audit, runtime, mask_manifest


# %% [markdown]
# ## 7. Target-free staged generation and freeze


# %%
def generate_and_freeze(
    raw_dir: Path,
    artifacts: Path,
    config: Mapping[str, Any],
    stage: str,
    wells: list[str],
    saved_frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, Path], pd.DataFrame]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exact exp209 HMM")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    process_noise = saved_frames["process_noise"].copy()
    process_lookup = process_noise.set_index("well_id")
    prediction_lookup = {
        str(well): group.copy()
        for well, group in saved_frames["prediction"].groupby(
            "well_id", sort=True
        )
    }
    schedule_lookup = {
        str(well): group.copy()
        for well, group in saved_frames["schedule"].groupby(
            "well_id", sort=True
        )
    }
    mask_lookup = saved_frames["mask_manifest"].set_index("well_id")
    outer_workers = int(get_nested(config, "runtime.num_workers"))

    def build_one(
        index: int, well: str
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
        pd.DataFrame,
        dict[str, Any],
    ]:
        print(f"[{index}/{len(wells)}] exp350 stage={stage} well={well}", flush=True)
        result = decode_well(
            well,
            raw_dir,
            process_lookup.loc[well].to_dict(),
            prediction_lookup[well],
            schedule_lookup[well],
            mask_lookup.loc[well].to_dict(),
            config,
            stage,
        )
        print(json.dumps(to_jsonable(result[3]), sort_keys=True), flush=True)
        return result

    if outer_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(build_one)(index, well)
            for index, well in enumerate(wells, start=1)
        )
    else:
        results = [
            build_one(index, well)
            for index, well in enumerate(wells, start=1)
        ]
    prediction = pd.concat([item[0] for item in results], ignore_index=True)
    forward_schedule = pd.concat(
        [item[1] for item in results], ignore_index=True
    )
    smoothed_schedule = pd.concat(
        [item[2] for item in results], ignore_index=True
    )
    audit = pd.DataFrame([item[3] for item in results])
    runtime = pd.concat([item[4] for item in results], ignore_index=True)
    mask_manifest = pd.DataFrame(
        [
            {"well_id": wells[index], **item[5]}
            for index, item in enumerate(results)
        ]
    )
    for frame in (prediction, forward_schedule, smoothed_schedule):
        frame.sort_values(
            ["well_id", "row_idx"], kind="mergesort", inplace=True
        )
        frame.reset_index(drop=True, inplace=True)
    identity = prediction[["well_id", "row_idx"]]
    if not identity.equals(forward_schedule[["well_id", "row_idx"]]):
        raise RuntimeError("prediction and forward schedule identities differ")
    if not identity.equals(smoothed_schedule[["well_id", "row_idx"]]):
        raise RuntimeError("prediction and smoothed schedule identities differ")
    stage_prefix = f"{OUTPUT_PREFIX}_{stage}"
    paths = {
        "prediction": artifacts / f"{stage_prefix}_predictions.csv.gz",
        "forward_schedule": (
            artifacts / f"{stage_prefix}_forward_affine_schedule.csv.gz"
        ),
        "smoothed_schedule": (
            artifacts / f"{stage_prefix}_smoothed_affine_schedule.csv.gz"
        ),
        "numerical_audit": artifacts / f"{stage_prefix}_numerical_audit.csv",
        "runtime": artifacts / f"{stage_prefix}_runtime.csv",
        "mask_manifest": artifacts / f"{stage_prefix}_mask_manifest.csv",
        "used_process_noise": (
            artifacts / f"{stage_prefix}_used_process_noise.csv"
        ),
    }
    reports = {
        "prediction": write_gzip_csv(prediction, paths["prediction"]),
        "forward_schedule": write_gzip_csv(
            forward_schedule, paths["forward_schedule"]
        ),
        "smoothed_schedule": write_gzip_csv(
            smoothed_schedule, paths["smoothed_schedule"]
        ),
    }
    audit.to_csv(paths["numerical_audit"], index=False)
    runtime.to_csv(paths["runtime"], index=False)
    mask_manifest.to_csv(paths["mask_manifest"], index=False)
    process_noise.to_csv(paths["used_process_noise"], index=False)
    for name in (
        "numerical_audit",
        "runtime",
        "mask_manifest",
        "used_process_noise",
    ):
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
        "forward_schedule": reports["forward_schedule"],
        "smoothed_schedule": reports["smoothed_schedule"],
        "numerical_audit": reports["numerical_audit"],
        "runtime": reports["runtime"],
        "mask_manifest": reports["mask_manifest"],
        "used_process_noise": reports["used_process_noise"],
        "identity_content_sha256": dataframe_content_sha(
            prediction, ["well_id", "row_idx"]
        ),
    }
    freeze["freeze_manifest_sha256"] = mapping_sha256(freeze)
    return freeze, paths, runtime


# %% [markdown]
# ## 8. Late truth attachment, diagnostics, and promotion gates


# %%
def role_mask(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.eq("valid").to_numpy(bool)


def load_late_readout(
    prediction_path: Path,
    raw_dir: Path,
    controls: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction = pd.read_csv(
        prediction_path, dtype={"id": str, "well_id": str}
    )
    truth_rows: list[pd.DataFrame] = []
    for well, group in prediction.groupby("well_id", sort=True):
        source = pd.read_csv(
            raw_dir / f"{well}__horizontal_well.csv",
            usecols=["TVT_input"],
        )
        row_index = group["row_idx"].to_numpy(np.int64)
        truth = pd.to_numeric(
            source.loc[row_index, "TVT_input"], errors="raise"
        ).to_numpy(np.float64)
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
    frame = prediction.merge(
        truth_frame,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if frame["true_tvt"].isna().any():
        raise ValueError("late truth attachment left missing values")
    safe = pd.read_csv(
        Path(str(controls["paths"]["fold_assignment"])),
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
    )
    fold_map = safe.groupby("well_id", sort=True)["fold"].first().astype(int)
    frame["fold"] = frame["well_id"].map(fold_map)
    if frame["fold"].isna().any():
        raise ValueError("late fold attachment left missing values")
    hidden_spec = get_nested(config, "data.hidden_like_assignment") or {}
    role_columns = [str(value) for value in hidden_spec["role_columns"].values()]
    hidden = pd.read_csv(
        Path(str(controls["paths"]["hidden_like_assignment"])),
        usecols=["well_id", *role_columns],
        dtype={"well_id": str},
    ).set_index("well_id")
    hidden_counts: dict[str, Any] = {}
    for output_name, source_name in hidden_spec["role_columns"].items():
        mapped = frame["well_id"].map(hidden[source_name])
        if mapped.isna().any():
            raise ValueError(
                f"hidden-like role attachment failed for {output_name}"
            )
        frame[str(output_name)] = role_mask(mapped)
        hidden_counts[str(output_name)] = {
            "rows": int(frame[str(output_name)].sum()),
            "wells": int(
                frame.loc[frame[str(output_name)], "well_id"].nunique()
            ),
        }
    late = {
        "attached_after_prediction_freeze": True,
        "truth_source": "raw_visible_TVT_input",
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "hidden_like_scopes": hidden_counts,
        "identity_content_sha256": dataframe_content_sha(
            frame, ["well_id", "row_idx"]
        ),
    }
    return frame, late


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(prediction) - np.asarray(truth)) ** 2
            )
        )
    )


def paired_metric(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scope: str,
    *,
    comparison: str,
    baseline_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    subset = frame.loc[mask]
    truth = subset["true_tvt"].to_numpy(np.float64)
    baseline = subset[baseline_column].to_numpy(np.float64)
    candidate = subset[candidate_column].to_numpy(np.float64)
    baseline_rmse = rmse(truth, baseline)
    candidate_rmse = rmse(truth, candidate)
    return {
        "comparison": comparison,
        "scope": scope,
        "rows": len(subset),
        "wells": int(subset["well_id"].nunique()),
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "improvement_ft": baseline_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_baseline": (
            candidate_rmse - baseline_rmse
        ),
    }


def build_metrics(
    frame: pd.DataFrame, stage: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool))
    ]
    for fold in sorted(frame["fold"].astype(int).unique()):
        scopes.append(
            (
                f"fold_{fold}",
                frame["fold"].to_numpy(np.int64) == fold,
            )
        )
    for column in (
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ):
        mask = frame[column].to_numpy(bool)
        if mask.any():
            scopes.append((column, mask))
    comparisons = [
        ("vs_masked_exp209_parent", "parent_hmm_tvt"),
        ("vs_exp345_causal", "causal_hmm_tvt"),
    ]
    paired = pd.DataFrame(
        [
            paired_metric(
                frame,
                mask,
                scope,
                comparison=comparison,
                baseline_column=baseline_column,
                candidate_column=f"{VARIANT}_hmm_tvt",
            )
            for comparison, baseline_column in comparisons
            for scope, mask in scopes
            if mask.any()
        ]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        parent_rmse = rmse(
            truth, group["parent_hmm_tvt"].to_numpy(np.float64)
        )
        causal_rmse = rmse(
            truth, group["causal_hmm_tvt"].to_numpy(np.float64)
        )
        candidate_rmse = rmse(
            truth, group[f"{VARIANT}_hmm_tvt"].to_numpy(np.float64)
        )
        by_well_rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "parent_rmse": parent_rmse,
                "causal_rmse": causal_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_candidate_minus_parent": (
                    candidate_rmse - parent_rmse
                ),
                "delta_candidate_minus_causal": (
                    candidate_rmse - causal_rmse
                ),
            }
        )
    return paired, pd.DataFrame(by_well_rows)


def evaluate_gate(
    paired: pd.DataFrame,
    by_well: pd.DataFrame,
    frame: pd.DataFrame,
    numerical_audit: pd.DataFrame,
    runtime: pd.DataFrame,
    runtime_seconds: float,
    stage: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical_spec = get_nested(config, "promotion_gates.technical") or {}
    science_spec = get_nested(config, "promotion_gates.scientific") or {}
    expected_wells = int(technical_spec["expected_prediction_wells"])
    expected_rows = int(technical_spec["expected_prediction_rows"])
    expected_hmm_runs = int(technical_spec["expected_new_hmm_runs"])
    maximum_seconds = float(technical_spec["maximum_runtime_hours"]) * 3600.0
    parity_maximum = float(
        numerical_audit[
            "forward_parity_maximum_absolute_difference"
        ].max()
    )
    terminal_state_maximum = float(
        numerical_audit["terminal_state_max_abs_error"].max()
    )
    terminal_covariance_maximum = float(
        numerical_audit["terminal_covariance_max_abs_error"].max()
    )
    minimum_eigenvalue = float(
        numerical_audit[
            "covariance_minimum_eigenvalue_before_floor"
        ].min()
    )
    contraction_maximum = float(
        numerical_audit[
            "covariance_contraction_max_positive_eigenvalue"
        ].max()
    )
    clip_fraction = float(
        numerical_audit["output_scale_clip_rows"].sum()
        / numerical_audit["score_rows"].sum()
    )
    control_overall = {
        comparison: paired.loc[
            (paired["comparison"] == comparison)
            & (paired["scope"] == "overall")
        ].iloc[0]
        for comparison in (
            "vs_masked_exp209_parent",
            "vs_exp345_causal",
        )
    }
    control_tolerance = float(
        technical_spec["saved_control_metric_absolute_tolerance"]
    )
    control_parity = {
        "masked_exp209_parent": {
            "actual": float(
                control_overall["vs_masked_exp209_parent"][
                    "baseline_rmse"
                ]
            ),
            "expected": float(
                get_nested(config, "references.exp345_stage_0_parent_rmse")
            ),
        },
        "exp345_causal": {
            "actual": float(
                control_overall["vs_exp345_causal"]["baseline_rmse"]
            ),
            "expected": float(
                get_nested(config, "references.exp345_stage_0_causal_rmse")
            ),
        },
    }
    for record in control_parity.values():
        record["absolute_difference"] = abs(
            record["actual"] - record["expected"]
        )
        record["tolerance"] = control_tolerance
        record["passed"] = bool(
            record["absolute_difference"] <= control_tolerance
        )
    hmm_run_counts = (
        runtime["run"].value_counts().sort_index().astype(int).to_dict()
    )
    finite_predictions = bool(
        np.isfinite(
            frame[
                [
                    "parent_hmm_tvt",
                    "causal_hmm_tvt",
                    f"{VARIANT}_hmm_tvt",
                ]
            ].to_numpy(np.float64)
        ).all()
    )
    technical = {
        "all_saved_exp345_sha_verified": True,
        "finite_predictions": finite_predictions,
        "prediction_rows": len(frame),
        "expected_prediction_rows": expected_rows,
        "prediction_wells": int(frame["well_id"].nunique()),
        "expected_prediction_wells": expected_wells,
        "hmm_runs": len(runtime),
        "expected_hmm_runs": expected_hmm_runs,
        "hmm_run_counts": hmm_run_counts,
        "expected_hmm_run_counts": {VARIANT: expected_hmm_runs},
        "forward_schedule_parity_max_abs": parity_maximum,
        "forward_schedule_parity_tolerance": float(
            technical_spec["forward_schedule_max_abs_tolerance"]
        ),
        "saved_control_metric_parity": control_parity,
        "terminal_state_max_abs_error": terminal_state_maximum,
        "terminal_covariance_max_abs_error": terminal_covariance_maximum,
        "terminal_tolerance": float(
            technical_spec["terminal_state_max_abs_tolerance"]
        ),
        "covariance_minimum_eigenvalue_before_floor": (
            minimum_eigenvalue
        ),
        "covariance_minimum_eigenvalue_threshold": float(
            technical_spec["covariance_minimum_eigenvalue"]
        ),
        "covariance_contraction_max_positive_eigenvalue": (
            contraction_maximum
        ),
        "covariance_contraction_threshold": float(
            technical_spec[
                "covariance_contraction_max_positive_eigenvalue"
            ]
        ),
        "output_scale_clip_fraction": clip_fraction,
        "maximum_output_scale_clip_fraction": float(
            technical_spec["maximum_output_scale_clip_fraction"]
        ),
        "posterior_normalization_max_abs_error": float(
            runtime["posterior_row_sum_max_abs_error"].max()
        ),
        "runtime_seconds": runtime_seconds,
        "maximum_runtime_seconds": maximum_seconds,
    }
    technical["passed"] = bool(
        technical["finite_predictions"]
        and technical["prediction_rows"]
        == technical["expected_prediction_rows"]
        and technical["prediction_wells"]
        == technical["expected_prediction_wells"]
        and technical["hmm_runs"] == technical["expected_hmm_runs"]
        and technical["hmm_run_counts"]
        == technical["expected_hmm_run_counts"]
        and technical["forward_schedule_parity_max_abs"]
        <= technical["forward_schedule_parity_tolerance"]
        and all(
            bool(record["passed"])
            for record in control_parity.values()
        )
        and technical["terminal_state_max_abs_error"]
        <= technical["terminal_tolerance"]
        and technical["terminal_covariance_max_abs_error"]
        <= technical["terminal_tolerance"]
        and technical["covariance_minimum_eigenvalue_before_floor"]
        >= technical["covariance_minimum_eigenvalue_threshold"]
        and technical[
            "covariance_contraction_max_positive_eigenvalue"
        ]
        <= technical["covariance_contraction_threshold"]
        and technical["output_scale_clip_fraction"]
        <= technical["maximum_output_scale_clip_fraction"]
        and technical["posterior_normalization_max_abs_error"] <= 1.0e-6
        and technical["runtime_seconds"]
        <= technical["maximum_runtime_seconds"]
    )
    science: dict[str, Any] = {}
    comparison_specs = {
        "vs_masked_exp209_parent": {
            "minimum_improvement": float(
                science_spec[
                    "minimum_improvement_vs_masked_exp209_parent_ft"
                ]
            ),
            "minimum_folds": int(
                science_spec[
                    "minimum_improved_folds_vs_masked_exp209_parent"
                ]
            ),
        },
        "vs_exp345_causal": {
            "minimum_improvement": float(
                science_spec[
                    "minimum_improvement_vs_exp345_causal_ft"
                ]
            ),
            "minimum_folds": int(
                science_spec["minimum_improved_folds_vs_exp345_causal"]
            ),
        },
    }
    comparisons_pass = True
    for comparison, thresholds in comparison_specs.items():
        subset = paired.loc[paired["comparison"] == comparison]
        overall = subset.loc[subset["scope"] == "overall"].iloc[0]
        folds = subset.loc[subset["scope"].str.startswith("fold_")]
        folds_improved = int((folds["improvement_ft"] > 0.0).sum())
        hidden = {
            scope: bool(
                subset.loc[
                    subset["scope"] == scope,
                    "delta_rmse_candidate_minus_baseline",
                ].iloc[0]
                <= 0.0
            )
            for scope in (
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            )
            if (subset["scope"] == scope).any()
        }
        record_passed = bool(
            float(overall["improvement_ft"])
            >= thresholds["minimum_improvement"]
            and folds_improved >= thresholds["minimum_folds"]
            and len(hidden)
            == int(science_spec["require_hidden_like_scope_count"])
            and all(hidden.values())
        )
        comparisons_pass = comparisons_pass and record_passed
        science[comparison] = {
            "baseline_rmse": float(overall["baseline_rmse"]),
            "candidate_rmse": float(overall["candidate_rmse"]),
            "improvement_ft": float(overall["improvement_ft"]),
            "minimum_improvement_ft": thresholds[
                "minimum_improvement"
            ],
            "folds_improved": folds_improved,
            "minimum_folds_improved": thresholds["minimum_folds"],
            "hidden_like_non_regression": hidden,
            "passed": record_passed,
        }
    parent_delta = by_well[
        "delta_candidate_minus_parent"
    ].to_numpy(np.float64)
    median_delta = float(np.median(parent_delta))
    p95_delta = float(np.quantile(parent_delta, 0.95))
    worst_delta = float(np.max(parent_delta))
    boundary_p95 = float(
        numerical_audit["boundary_jump_sigma"].quantile(0.95)
    )
    nll_weights = numerical_audit["score_rows"].to_numpy(np.float64)
    smoother_nll = float(
        np.average(
            numerical_audit[
                "gr_reconstruction_nll_smoother_mean"
            ].to_numpy(np.float64),
            weights=nll_weights,
        )
    )
    tail_passed = bool(
        median_delta
        <= float(
            science_spec[
                "require_by_well_median_delta_vs_parent_max"
            ]
        )
        and p95_delta
        <= float(
            science_spec[
                "require_by_well_p95_delta_vs_parent_max"
            ]
        )
        and worst_delta
        <= float(
            science_spec[
                "maximum_worst_well_regression_vs_parent_ft"
            ]
        )
        and boundary_p95
        <= float(science_spec["maximum_boundary_jump_p95_sigma"])
    )
    science["tail_and_boundary"] = {
        "by_well_median_delta_vs_parent": median_delta,
        "maximum_median_delta": float(
            science_spec[
                "require_by_well_median_delta_vs_parent_max"
            ]
        ),
        "by_well_p95_delta_vs_parent": p95_delta,
        "maximum_p95_delta": float(
            science_spec["require_by_well_p95_delta_vs_parent_max"]
        ),
        "worst_well_delta_vs_parent": worst_delta,
        "maximum_worst_well_delta": float(
            science_spec[
                "maximum_worst_well_regression_vs_parent_ft"
            ]
        ),
        "boundary_jump_p95_sigma": boundary_p95,
        "maximum_boundary_jump_p95_sigma": float(
            science_spec["maximum_boundary_jump_p95_sigma"]
        ),
        "passed": tail_passed,
    }
    science["diagnostic_only"] = {
        "gr_reconstruction_nll_smoother": smoother_nll,
        "role": str(science_spec["gr_reconstruction_nll_role"]),
    }
    science["passed"] = bool(comparisons_pass and tail_passed)
    passed = bool(technical["passed"] and science["passed"])
    return {
        "stage": stage,
        "passed": passed,
        "decision": str(
            get_nested(
                config,
                "promotion_gates.decision_if_pass"
                if passed
                else "promotion_gates.decision_if_fail",
            )
        ),
        "technical_gate": technical,
        "scientific_gate": science,
    }


# %% [markdown]
# ## 9. Experiment orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not KAGGLE_WORKING_ROOT.exists()
        and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1"
    ):
        raise RuntimeError(
            "exp350 must run first on Kaggle; local execution requires explicit "
            "smoke approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    stage = selected_stage(config)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    controls = preflight_controls_and_assignments(config)
    saved_frames = load_saved_exp345_frames(controls, config)
    safe = pd.read_csv(
        Path(str(controls["paths"]["fold_assignment"])),
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
    )
    fold_map = (
        safe.groupby("well_id", sort=True)["fold"].first().astype(int).to_dict()
    )
    if sorted(fold_map) != sorted(raw_preflight["well_ids"]):
        raise ValueError("raw and fold-assignment well identities differ")
    wells = sorted(fold_map, key=lambda well: (fold_map[well], well))
    if sorted(wells) != sorted(
        saved_frames["prediction"]["well_id"].astype(str).unique()
    ):
        raise ValueError("saved exp345 controls and raw wells differ")
    scientific_contract = build_scientific_contract(config, stage)
    contract_path = artifacts / (
        f"{OUTPUT_PREFIX}_{stage}_scientific_contract.json"
    )
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_{stage}_input_manifest.json"
    write_json(contract_path, scientific_contract)
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": stage,
        "truth_attached": False,
        "raw_train": {
            key: value
            for key, value in raw_preflight.items()
            if key != "well_ids"
        },
        "controls": controls,
        "selected_wells": wells,
        "selected_well_order_sha256": mapping_sha256({"wells": wells}),
        "execution_counts": {
            "scientific_variants": 1,
            "forward_filter_wells": len(wells),
            "bidirectional_smoother_wells": len(wells),
            "new_hmm_well_runs": len(wells),
            "parent_control_hmm_reruns": 0,
            "causal_control_hmm_reruns": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "gpu_runs": 0,
        },
    }
    write_json(manifest_path, input_manifest)
    freeze, paths, runtime = generate_and_freeze(
        raw_dir,
        artifacts,
        config,
        stage,
        wells,
        saved_frames,
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
    numerical_audit = pd.read_csv(
        paths["numerical_audit"], dtype={"well_id": str}
    )
    runtime_seconds = time.time() - started
    gate = evaluate_gate(
        paired,
        by_well,
        frame,
        numerical_audit,
        runtime,
        runtime_seconds,
        stage,
        config,
    )
    output_paths = {
        "paired_metrics": (
            artifacts / f"{OUTPUT_PREFIX}_{stage}_paired_metrics.csv"
        ),
        "by_well_metrics": (
            artifacts / f"{OUTPUT_PREFIX}_{stage}_by_well_metrics.csv"
        ),
        "promotion_gate": (
            artifacts / f"{OUTPUT_PREFIX}_{stage}_promotion_gate.json"
        ),
    }
    paired.to_csv(output_paths["paired_metrics"], index=False)
    by_well.to_csv(output_paths["by_well_metrics"], index=False)
    write_json(output_paths["promotion_gate"], gate)
    status = (
        "stage_0_passed_wait_for_separate_stage_1_approval"
        if gate["passed"]
        else "stage_0_failed_closed"
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
        "forward_filter_wells": len(wells),
        "bidirectional_smoother_wells": len(wells),
        "hmm_well_runs": len(runtime),
        "parent_control_reruns": 0,
        "causal_control_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
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
            "kernel_version_recording": (
                "record_from_kaggle_api_after_run"
            ),
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_{stage}_summary.json"
    write_json(summary_path, summary)
    overall = paired.loc[
        (paired["comparison"] == "vs_masked_exp209_parent")
        & (paired["scope"] == "overall")
    ].iloc[0].to_dict()
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": stage,
        "cv": (
            float(overall["candidate_rmse"])
            if gate["technical_gate"]["passed"]
            else None
        ),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall_vs_masked_exp209_parent": overall,
        "overall_vs_exp345_causal": paired.loc[
            (paired["comparison"] == "vs_exp345_causal")
            & (paired["scope"] == "overall")
        ].iloc[0].to_dict(),
        "promotion_gate": gate,
        "prediction_sha256": freeze["prediction"],
        "forward_schedule_sha256": freeze["forward_schedule"],
        "smoothed_schedule_sha256": freeze["smoothed_schedule"],
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Train-side Stage 0 full-well bidirectional smoother audit only; "
            "no Stage 1, inference, or submission is generated."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(paired.to_string(index=False), flush=True)
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True), flush=True)
    print(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True),
        flush=True,
    )
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
                "implementation_complete": get_nested(
                    CONFIG, "implementation.enabled"
                ),
                "kaggle_push_approved": get_nested(
                    CONFIG, "execution.kaggle_push_approved"
                ),
                "active_stage": get_nested(CONFIG, "execution.active_stage"),
                "run_flags": {
                    "stage_0": get_nested(
                        CONFIG, "execution.run_stage_0"
                    ),
                    "stage_1": get_nested(
                        CONFIG, "execution.run_stage_1"
                    ),
                },
                "execution_counts": get_nested(
                    CONFIG,
                    "execution_contract.stage_0_if_implemented_and_separately_approved",
                ),
                "hmm_fixed": get_nested(
                    CONFIG, "model.fixed_exp209_hmm"
                ),
                "forward_affine_state": get_nested(
                    CONFIG, "model.forward_affine_state"
                ),
                "bidirectional_smoother": get_nested(
                    CONFIG, "model.bidirectional_smoother"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


# %% [markdown]
# ## 11. Run the approved Kaggle CPU Stage 0

# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_full_experiment(CONFIG)
