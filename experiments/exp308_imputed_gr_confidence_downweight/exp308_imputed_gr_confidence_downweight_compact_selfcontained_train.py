# %% [markdown]
# # exp308 imputed GR confidence downweight train
#
# Train-side exact-HMM audit of one preregistered missing-distance confidence.
# The promoted exp307 finite-MAD prediction/scale and saved exp072 likPF control
# are read-only; no control, PF, Beam, model, inference, or submission is rerun.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Missing-distance confidence and parent-identical HMM inputs
# 5. Weighted Gaussian emission and exact exp209 forward-backward kernel
# 6. Target-free weight audit, decoding, and prediction freeze
# 7. Late truth/parent/control attachment and paired metrics
# 8. Promotion gate, diagnostics, and generated artifacts
# 9. Setup and configuration preview
# 10. Run the Kaggle CPU audit

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable
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


EXPERIMENT_NAME = "exp308_imputed_gr_confidence_downweight"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT_ORDER = ("missing_distance_half8_floor025",)
PARENT_VARIANT = "finite_mad_primary"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP308_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
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
    raise FileNotFoundError(f"exp308 config not found in {[str(path) for path in candidates]}")


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


def train_data_dir(config: dict[str, Any]) -> Path:
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


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


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
        versions["numba_num_threads"] = get_num_threads()
    return versions


# %% [markdown]
# ## 3. Frozen scientific contract and input preflight


# %%
def validate_scientific_contract(
    config: dict[str, Any], *, require_run_approval: bool = False
) -> None:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp307_finite_only_robust_sigma_gr",
        "model.active_variants": list(VARIANT_ORDER),
        "model.promotion_candidate": "missing_distance_half8_floor025",
        "model.confidence.observed_weight": 1.0,
        "model.confidence.half_life_rows": 8.0,
        "model.confidence.minimum_weight": 0.25,
        "model.confidence.no_finite_gr_fallback": 0.25,
        "model.confidence.apply_to": "gaussian_log_emission",
        "model.confidence.additional_grid": False,
        "model.fixed_parent.sigma_column": "finite_mad",
        "model.fixed_parent.prediction_column": "finite_mad_primary_hmm_tvt",
        "model.execution_counts.variants": 1,
        "model.execution_counts.hmm_well_runs": 773,
        "model.execution_counts.lightgbm_configs": 0,
        "model.execution_counts.trained_folds": 0,
        "model.execution_counts.pf_well_runs": 0,
        "model.execution_counts.beam_well_runs": 0,
        "model.execution_counts.boosters": 0,
        "model.execution_counts.parent_control_retraining": False,
        "runtime.num_workers": 2,
        "runtime.numba_num_threads": 2,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
        "execution.create_submission": False,
        "execution.run_inference": False,
        "gate.require_1000_plus_non_regression": True,
        "gate.require_hidden_like_spatial_non_regression": True,
        "gate.require_hidden_like_typewell_purged_non_regression": True,
        "gate.require_fixed_likpf_blend_non_regression": True,
        "gate.require_gap_bucket_readout": True,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp308 fixed contract mismatch: {key} must be {value!r}")
    hmm = get_nested(config, "model.fixed_hmm") or {}
    fixed_hmm = {
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
    }
    for key, value in fixed_hmm.items():
        if float(hmm.get(key, -1.0)) != value:
            raise ValueError(f"exp308 fixes model.fixed_hmm.{key}={value}")
    if hmm.get("emission") != "gaussian" or hmm.get("rate_center") != "zero":
        raise ValueError("exp308 fixes Gaussian emission and zero-centered rate grid")
    fixed_policies = {
        "evaluation_gr_policy": "interpolate_both_directions_then_typewell_mean",
        "typewell_gr_policy": "sort_tvt_ffill_bfill_then_linear_interp",
        "output": "posterior_mean",
    }
    for key, value in fixed_policies.items():
        if hmm.get(key) != value:
            raise ValueError(f"exp308 fixes model.fixed_hmm.{key}={value}")
    confidence = get_nested(config, "model.confidence") or {}
    if confidence.get("distance_unit") != "within_well_row_distance_to_nearest_raw_finite_gr":
        raise ValueError("exp308 fixes confidence distance to within-well raw-row distance")
    if confidence.get("missing_formula") != "max_floor_0.25_times_power2_negative_distance_div8":
        raise ValueError("exp308 fixes the preregistered missing confidence formula")
    if confidence.get("raw_finite_rows_exact_one") is not True:
        raise ValueError("exp308 requires exact weight 1 on raw-finite rows")
    parent = get_nested(config, "data.parent_exp307") or {}
    required_parent_values = {
        "required_status": "train_side_finite_mad_gate_passed_no_automatic_downstream",
        "required_variant": PARENT_VARIANT,
        "kernel_source": "kentookumura/exp307-finite-only-robust-sigma-gr-train",
    }
    for key, value in required_parent_values.items():
        if parent.get(key) != value:
            raise ValueError(f"exp308 parent dependency fixes data.parent_exp307.{key}={value!r}")
    for key in (
        "expected_scientific_contract_sha256",
        "expected_input_control_manifest_sha256",
        "expected_prediction_decompressed_sha256",
        "expected_scale_audit_decompressed_sha256",
        "expected_promotion_gate_sha256",
    ):
        value = str(parent.get(key, ""))
        if value != "PENDING_EXP307_PASS" and len(value) != 64:
            raise ValueError(f"exp308 parent dependency must be pending or a frozen SHA: {key}")
    reference_parent_prediction_sha = str(
        get_nested(config, "references.parent_exp307_prediction_decompressed_sha256") or ""
    )
    if (
        reference_parent_prediction_sha != "PENDING_EXP307_PASS"
        and len(reference_parent_prediction_sha) != 64
    ):
        raise ValueError("exp308 parent prediction reference must be pending or a frozen SHA")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp308 implementation approval must be recorded")
    if require_run_approval:
        parent_shas = [
            str(parent.get(key, ""))
            for key in (
                "expected_scientific_contract_sha256",
                "expected_input_control_manifest_sha256",
                "expected_prediction_decompressed_sha256",
                "expected_scale_audit_decompressed_sha256",
                "expected_promotion_gate_sha256",
            )
        ]
        parent_metrics = [
            get_nested(config, "references.parent_exp307_rmse"),
            get_nested(config, "references.parent_exp307_likpf_50_50_rmse"),
        ]
        dependency_frozen = bool(get_nested(config, "execution.parent_dependency_frozen"))
        dependency_frozen = dependency_frozen and all(len(value) == 64 for value in parent_shas)
        dependency_frozen = dependency_frozen and (
            parent_shas[2] == reference_parent_prediction_sha
        )
        dependency_frozen = dependency_frozen and all(
            value is not None and math.isfinite(float(value)) for value in parent_metrics
        )
        if not dependency_frozen:
            raise RuntimeError("exp308 parent exp307 dependency status/SHA/metrics are not frozen")
        if not (
            bool(get_nested(config, "execution.kaggle_push_approved"))
            and bool(get_nested(config, "execution.run_weight_audit"))
            and bool(get_nested(config, "execution.run_hmm"))
        ):
            raise RuntimeError("exp308 Kaggle package/push/run is not approved")


def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "truth_attached": False,
        "variants": list(VARIANT_ORDER),
        "promotion_candidate": VARIANT_ORDER[0],
        "confidence": get_nested(config, "model.confidence"),
        "fixed_parent": get_nested(config, "model.fixed_parent"),
        "fixed_hmm": get_nested(config, "model.fixed_hmm"),
        "gate": get_nested(config, "gate"),
        "parent_dependency": {
            key: value
            for key, value in (get_nested(config, "data.parent_exp307") or {}).items()
            if key.startswith("expected_") or key in {"required_status", "required_variant"}
        },
        "execution_counts": {
            "active_variants": 1,
            "hmm_well_runs": 773,
            "models": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "boosters": 0,
            "control_reruns": 0,
        },
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "forbidden": [
            "half_life_floor_weight_or_run_length_grid",
            "zero_weight_or_hard_missing_mask",
            "evaluation_gr_change",
            "typewell_change",
            "sigma_reestimation_or_grid",
            "hmm_grid_transition_prior_or_output_change",
            "likpf_pf_or_beam_rerun",
            "inference",
            "submission",
        ],
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def _candidate_paths(spec: dict[str, Any]) -> list[str]:
    return [str(value) for value in spec.get("candidates", [])]


def validate_raw_well_identity(config: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
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
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != int(get_nested(config, "validation.expected_wells")) or actual != expected:
        raise ValueError("current raw train well-file identity mismatch")
    return {"path": str(raw_dir), "wells": len(frame), "content_sha256": actual}


def preflight_parent_exp307(config: dict[str, Any]) -> dict[str, Any]:
    spec = get_nested(config, "data.parent_exp307") or {}
    candidates = _candidate_paths(spec)
    filenames = {
        "prediction": spec["prediction_filename"],
        "scale_audit": spec["scale_audit_filename"],
        "summary": spec["summary_filename"],
        "promotion_gate": spec["promotion_gate_filename"],
        "scientific_contract": spec["scientific_contract_filename"],
        "input_control_manifest": spec["input_control_manifest_filename"],
    }
    paths = {
        name: resolve_existing(str(filename), candidates)
        for name, filename in filenames.items()
    }
    prediction_report = inspect_gzip_csv(paths["prediction"])
    scale_report = inspect_gzip_csv(paths["scale_audit"])
    summary = read_json(paths["summary"])
    promotion = read_json(paths["promotion_gate"])
    parent_contract = read_json(paths["scientific_contract"])
    parent_manifest = read_json(paths["input_control_manifest"])
    expected = {
        "scientific_contract_sha256": str(spec["expected_scientific_contract_sha256"]),
        "input_control_manifest_sha256": str(spec["expected_input_control_manifest_sha256"]),
        "prediction_decompressed_sha256": str(spec["expected_prediction_decompressed_sha256"]),
        "scale_audit_decompressed_sha256": str(spec["expected_scale_audit_decompressed_sha256"]),
        "promotion_gate_sha256": str(spec["expected_promotion_gate_sha256"]),
    }
    parent_direct_gate = promotion.get("primary_direct_gate") or {}
    parent_blend_gate = promotion.get("fixed_likpf_50_50_guard") or {}
    parent_rmse = parent_direct_gate.get("candidate_rmse")
    parent_blend_rmse = parent_blend_gate.get("candidate_rmse")
    baseline_tolerance = float(get_nested(config, "gate.baseline_metric_absolute_tolerance"))
    expected_parent_rmse = get_nested(config, "references.parent_exp307_rmse")
    expected_parent_blend_rmse = get_nested(
        config, "references.parent_exp307_likpf_50_50_rmse"
    )
    checks = {
        "summary_status": summary.get("status") == spec["required_status"],
        "summary_gate_passed": bool((summary.get("promotion_gate") or {}).get("passed")),
        "promotion_gate_passed": bool(promotion.get("passed")),
        "scientific_contract": parent_contract.get("scientific_contract_sha256")
        == expected["scientific_contract_sha256"],
        "summary_scientific_contract": summary.get("scientific_contract_sha256")
        == expected["scientific_contract_sha256"],
        "input_control_manifest": sha256_path(paths["input_control_manifest"])
        == expected["input_control_manifest_sha256"],
        "summary_input_control_manifest": summary.get("input_control_manifest_sha256")
        == expected["input_control_manifest_sha256"],
        "prediction": prediction_report["decompressed_sha256"]
        == expected["prediction_decompressed_sha256"],
        "prediction_reference": expected["prediction_decompressed_sha256"]
        == get_nested(config, "references.parent_exp307_prediction_decompressed_sha256"),
        "scale_audit": scale_report["decompressed_sha256"]
        == expected["scale_audit_decompressed_sha256"],
        "promotion_gate_sha": sha256_path(paths["promotion_gate"])
        == expected["promotion_gate_sha256"],
        "parent_direct_metric": parent_rmse is not None
        and expected_parent_rmse is not None
        and abs(float(parent_rmse) - float(expected_parent_rmse)) <= baseline_tolerance,
        "parent_blend_metric": parent_blend_rmse is not None
        and expected_parent_blend_rmse is not None
        and abs(float(parent_blend_rmse) - float(expected_parent_blend_rmse))
        <= baseline_tolerance,
        "raw_identity": (parent_manifest.get("raw_train") or {}).get("content_sha256")
        == get_nested(config, "data.expected_raw_well_identity_sha256"),
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise RuntimeError(f"exp308 parent exp307 dependency failed closed: {failed}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    prediction = pd.read_csv(
        paths["prediction"],
        usecols=["id", "well_id", "row_idx", str(spec["prediction_column"])],
        dtype={"id": str, "well_id": str},
    )
    scale = pd.read_csv(
        paths["scale_audit"],
        usecols=["well_id", str(spec["sigma_column"])],
        dtype={"well_id": str},
    )
    sigma = pd.to_numeric(scale[str(spec["sigma_column"])], errors="raise").to_numpy(np.float64)
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or prediction["id"].duplicated().any()
        or len(scale) != expected_wells
        or scale["well_id"].duplicated().any()
        or not np.isfinite(sigma).all()
        or bool(((sigma < 10.0) | (sigma > 60.0)).any())
    ):
        raise ValueError("exp307 parent prediction/scale coverage contract mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "checks": checks,
        "prediction": prediction_report,
        "scale_audit": scale_report,
        "parent_status": summary.get("status"),
        "parent_rmse": float(parent_rmse),
        "parent_likpf_50_50_rmse": float(parent_blend_rmse),
        "scientific_contract_sha256": parent_contract.get("scientific_contract_sha256"),
        "promotion_gate_sha256": sha256_path(paths["promotion_gate"]),
        "sigma_by_well": dict(zip(scale["well_id"].astype(str), sigma, strict=True)),
    }


def preflight_controls_and_assignments(config: dict[str, Any]) -> dict[str, Any]:
    parent = preflight_parent_exp307(config)
    control = get_nested(config, "data.saved_controls") or {}
    paths = {
        "saved_exp072": resolve_existing(
            str(control["exp072_cache_filename"]), _candidate_paths(control)
        ),
    }
    exp072_report = inspect_gzip_csv(paths["saved_exp072"])
    if exp072_report["decompressed_sha256"] != str(control["expected_decompressed_sha256"]):
        raise ValueError("saved exp072 control decompressed SHA mismatch")
    fold = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(str(fold["filename"]), _candidate_paths(fold))
    fold_report = inspect_gzip_csv(fold_path)
    if fold_report["decompressed_sha256"] != str(fold["expected_decompressed_sha256"]):
        raise ValueError("fold/truth assignment decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold["safe_columns"]]
    safe = pd.read_csv(fold_path, usecols=safe_columns, dtype={"well_id": str})
    for column in ("row_idx", "suffix_offset", "fold"):
        safe[column] = pd.to_numeric(safe[column], errors="raise").astype(np.int64)
    safe = safe.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if safe.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("fold assignment identity is duplicated")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if (
        len(safe) != int(get_nested(config, "validation.expected_rows"))
        or safe["well_id"].nunique() != int(get_nested(config, "validation.expected_wells"))
        or sorted(safe["fold"].unique().tolist()) != expected_folds
    ):
        raise ValueError("fold assignment row/well/fold coverage mismatch")
    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    hidden_path = resolve_existing(str(hidden["filename"]), _candidate_paths(hidden))
    hidden_sha = sha256_path(hidden_path)
    if hidden_sha != str(hidden["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    hidden_frame = pd.read_csv(hidden_path, dtype={"well_id": str})
    role_columns = [str(value) for value in hidden["role_columns"].values()]
    if not {"well_id", *role_columns}.issubset(hidden_frame.columns):
        raise ValueError("hidden-like assignment columns are incomplete")
    if hidden_frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment has duplicate wells")
    safe_wells = sorted(safe["well_id"].astype(str).unique().tolist())
    hidden_wells = sorted(hidden_frame["well_id"].astype(str).unique().tolist())
    if hidden_wells != safe_wells:
        raise ValueError("hidden-like assignment well identity mismatch")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if int(exp072_report["data_rows"]) != expected_rows:
        raise ValueError("saved control row coverage mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()}
        | {
            "parent_prediction": parent["paths"]["prediction"],
            "parent_scale_audit": parent["paths"]["scale_audit"],
            "fold_assignment": str(fold_path),
            "hidden_like_assignment": str(hidden_path),
        },
        "parent_exp307": parent,
        "saved_exp072": exp072_report,
        "fold_assignment": {
            **fold_report,
            "well_ids": safe_wells,
        },
        "hidden_like_assignment": {
            "path": str(hidden_path),
            "raw_sha256": hidden_sha,
            "wells": int(hidden_frame["well_id"].nunique()),
        },
    }


# %% [markdown]
# ## 4. Missing-distance confidence and parent-identical HMM inputs


# %%
def contiguous_missing_run_lengths(missing: np.ndarray) -> np.ndarray:
    values = np.asarray(missing, dtype=bool)
    lengths = np.zeros(len(values), dtype=np.int32)
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(values) - 1):
            stop = index if not value else index + 1
            lengths[start:stop] = stop - start
            start = None
    return lengths


def build_missing_distance_confidence(
    raw_gr: np.ndarray,
    *,
    half_life_rows: float = 8.0,
    minimum_weight: float = 0.25,
) -> dict[str, Any]:
    values = np.asarray(raw_gr, dtype=np.float64)
    finite = np.isfinite(values)
    missing = ~finite
    if half_life_rows != 8.0 or minimum_weight != 0.25:
        raise ValueError("exp308 forbids a half-life/floor grid")
    distance = np.zeros(len(values), dtype=np.int32)
    no_finite_fallback = not bool(finite.any())
    if no_finite_fallback:
        distance.fill(-1)
    else:
        indices = np.arange(len(values), dtype=np.int64)
        sentinel = 2 * len(values) + 1
        left = np.maximum.accumulate(np.where(finite, indices, -sentinel))
        right = np.minimum.accumulate(np.where(finite, indices, sentinel)[::-1])[::-1]
        distance = np.minimum(indices - left, right - indices).astype(np.int32)
        distance[finite] = 0
    weight = np.ones(len(values), dtype=np.float64)
    if no_finite_fallback:
        weight.fill(minimum_weight)
    else:
        weight[missing] = np.maximum(
            minimum_weight,
            np.exp2(-distance[missing].astype(np.float64) / half_life_rows),
        )
    run_length = contiguous_missing_run_lengths(missing)
    gap_bucket = np.full(len(values), "observed", dtype=object)
    gap_bucket[missing & (run_length <= 3)] = "gap_1_3"
    gap_bucket[missing & (run_length >= 4) & (run_length <= 15)] = "gap_4_15"
    gap_bucket[missing & (run_length >= 16)] = "gap_16_plus"
    if finite.any() and not np.array_equal(weight[finite], np.ones(int(finite.sum()))):
        raise AssertionError("raw-finite GR rows must have exact confidence 1")
    if missing.any() and not no_finite_fallback:
        if not bool(((weight[missing] >= 0.25) & (weight[missing] < 1.0)).all()):
            raise AssertionError("raw-missing GR confidence must be in [0.25, 1)")
    return {
        "raw_gr_finite": finite,
        "raw_gr_missing": missing,
        "nearest_finite_row_distance": distance,
        "confidence_weight": weight,
        "missing_run_length": run_length,
        "gap_bucket": gap_bucket,
        "no_finite_gr_fallback": no_finite_fallback,
    }


def parent_interpolated_gr(raw_gr: np.ndarray, typewell_gr: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(raw_gr, dtype=np.float64))
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(np.asarray(typewell_gr, dtype=np.float64))))
        .to_numpy(np.float64)
    )


def robust_initial_rate(horizontal: pd.DataFrame, tail_n: int = 30) -> float:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = horizontal.loc[np.isfinite(tvt_input)].tail(tail_n)
    dtvt = np.diff(pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(np.float64))
    dz = np.diff(pd.to_numeric(known["Z"], errors="coerce").to_numpy(np.float64))
    dmd = np.diff(pd.to_numeric(known["MD"], errors="coerce").to_numpy(np.float64))
    valid = (dmd > 0.0) & np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd)
    return float(np.median((dtvt[valid] + dz[valid]) / dmd[valid])) if valid.sum() >= 3 else 0.0


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    if "TVT" in frame.columns:
        raise RuntimeError("unknown-suffix truth entered the target-free horizontal frame")
    return frame.reset_index(drop=True)


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    return frame.sort_values("TVT", kind="mergesort").reset_index(drop=True)


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_hmm") or {}
    confidence_config = get_nested(config, "model.confidence") or {}
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    if np.isinf(tvt_input).any():
        raise ValueError("TVT_input may contain finite known values or NaN only")
    known_mask = np.isfinite(tvt_input)
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("each exp308 well must contain known prefix and unknown suffix rows")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    valid_typewell = np.isfinite(typewell_tvt) & np.isfinite(typewell_gr)
    typewell_tvt = typewell_tvt[valid_typewell]
    typewell_gr = typewell_gr[valid_typewell]
    if len(typewell_tvt) < 2 or np.any(np.diff(typewell_tvt) < 0):
        raise ValueError("typewell TVT/GR contract is invalid")
    known_index = np.flatnonzero(known_mask)
    eval_index = np.flatnonzero(eval_mask)
    last_index = int(known_index[-1])
    last_tvt = float(tvt_input[last_index])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    step = float(hmm["step"])
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    state_gr = np.interp(grid, typewell_tvt, typewell_gr)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    interpolated_gr = parent_interpolated_gr(horizontal_gr, typewell_gr)
    confidence = build_missing_distance_confidence(
        horizontal_gr,
        half_life_rows=float(confidence_config["half_life_rows"]),
        minimum_weight=float(confidence_config["minimum_weight"]),
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
        "grid": grid,
        "state_gr": state_gr,
        "observed_gr": interpolated_gr[eval_index],
        "raw_gr": horizontal_gr[eval_index],
        "raw_gr_finite": confidence["raw_gr_finite"][eval_index],
        "raw_gr_missing": confidence["raw_gr_missing"][eval_index],
        "nearest_finite_row_distance": confidence["nearest_finite_row_distance"][eval_index],
        "confidence_weight": confidence["confidence_weight"][eval_index],
        "missing_run_length": confidence["missing_run_length"][eval_index],
        "gap_bucket": confidence["gap_bucket"][eval_index],
        "no_finite_gr_fallback": confidence["no_finite_gr_fallback"],
        "dm": dm,
        "dz": dz,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "init_rate": init_rate,
    }


# %% [markdown]
# ## 5. Weighted Gaussian emission and exact exp209 forward-backward kernel


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


def build_weighted_gaussian_emission(
    observed_gr: np.ndarray,
    state_gr: np.ndarray,
    sigma_gr: float,
    confidence_weight: np.ndarray,
    emission_clip_z2: float,
) -> np.ndarray:
    observed = np.asarray(observed_gr, dtype=np.float64)
    states = np.asarray(state_gr, dtype=np.float64)
    weight = np.asarray(confidence_weight, dtype=np.float64)
    if observed.ndim != 1 or states.ndim != 1 or weight.shape != observed.shape:
        raise ValueError("exp308 weighted emission shape mismatch")
    if not np.isfinite(observed).all() or not np.isfinite(states).all():
        raise ValueError("exp308 weighted emission requires finite interpolated GR")
    if not np.isfinite(weight).all() or bool(((weight < 0.25) | (weight > 1.0)).any()):
        raise ValueError("exp308 confidence weight must stay in [0.25, 1]")
    zscore = (observed[:, None] - states[None, :]) / float(sigma_gr)
    base = -0.5 * np.minimum(zscore**2, float(emission_clip_z2))
    return (weight[:, None] * base).astype(np.float32)


def run_exact_hmm_variant(
    prepared: dict[str, Any],
    sigma_gr: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_hmm") or {}
    if not math.isfinite(float(sigma_gr)) or not 10.0 <= float(sigma_gr) <= 60.0:
        raise ValueError("exp308 sigma_gr must satisfy the frozen [10, 60] contract")
    emission = build_weighted_gaussian_emission(
        prepared["observed_gr"],
        prepared["state_gr"],
        sigma_gr,
        prepared["confidence_weight"],
        float(hmm["emission_clip_z2"]),
    )
    post_p, loglik = _hmm2_fb(
        emission,
        prepared["dm"].astype(np.float64),
        prepared["dz"].astype(np.float64),
        float(hmm["step"]),
        prepared["rates"].astype(np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["init_rate"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["momentum"]),
    )
    grid = prepared["grid"].astype(np.float64)
    mean = post_p @ grid
    variance = post_p @ (grid**2) - mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    return {
        "mean": mean,
        "std": std,
        "loglik": float(loglik),
        "posterior_row_sum_max_abs_error": float(np.max(np.abs(post_p.sum(axis=1) - 1.0))),
    }


# %% [markdown]
# ## 6. Target-free weight audit, decoding, and prediction freeze


# %%
def decode_well(
    well: str,
    raw_dir: Path,
    sigma_gr: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_hmm_inputs(horizontal, typewell, config)
    eval_index = prepared["eval_index"].astype(np.int64)
    predictions = pd.DataFrame(
        {
            "id": [f"{well}_{int(row_idx)}" for row_idx in eval_index],
            "well_id": well,
            "row_idx": eval_index,
        }
    )
    variant = VARIANT_ORDER[0]
    started = time.time()
    result = run_exact_hmm_variant(prepared, sigma_gr, config)
    predictions[f"{variant}_hmm_tvt"] = result["mean"]
    predictions[f"{variant}_hmm_std"] = result["std"]
    predictions[f"{variant}_hmm_loglik"] = float(result["loglik"])
    runtime_rows = [
        {
            "well_id": well,
            "variant_index": 0,
            "variant": variant,
            "rows": len(eval_index),
            "sigma_gr": float(sigma_gr),
            "raw_missing_rows": int(np.asarray(prepared["raw_gr_missing"]).sum()),
            "no_finite_gr_fallback": bool(prepared["no_finite_gr_fallback"]),
            "loglik": float(result["loglik"]),
            "posterior_row_sum_max_abs_error": float(result["posterior_row_sum_max_abs_error"]),
            "elapsed_seconds": time.time() - started,
        }
    ]
    numeric = predictions.drop(columns=["id", "well_id"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"well={well} produced non-finite prediction diagnostics")
    weight_audit = pd.DataFrame(
        {
            "id": predictions["id"],
            "well_id": well,
            "row_idx": eval_index,
            "raw_gr": prepared["raw_gr"],
            "interpolated_gr": prepared["observed_gr"],
            "raw_gr_finite": prepared["raw_gr_finite"],
            "raw_gr_missing": prepared["raw_gr_missing"],
            "nearest_finite_row_distance": prepared["nearest_finite_row_distance"],
            "confidence_weight": prepared["confidence_weight"],
            "missing_run_length": prepared["missing_run_length"],
            "gap_bucket": prepared["gap_bucket"],
            "no_finite_gr_fallback": bool(prepared["no_finite_gr_fallback"]),
            "parent_sigma_gr": float(sigma_gr),
        }
    )
    return predictions, weight_audit, pd.DataFrame(runtime_rows)


def generate_and_freeze_predictions(
    raw_dir: Path,
    artifacts: Path,
    config: dict[str, Any],
    wells: list[str],
    sigma_by_well: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exp308 exact-HMM audit")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    outer_workers = int(get_nested(config, "runtime.num_workers"))

    def build_one(index: int, well: str):
        print(f"[{index}/{len(wells)}] exp308 well={well}", flush=True)
        if well not in sigma_by_well:
            raise KeyError(f"missing frozen exp307 finite-MAD sigma for well={well}")
        return decode_well(well, raw_dir, float(sigma_by_well[well]), config)

    started = time.time()
    if outer_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(build_one)(index, well) for index, well in enumerate(wells, start=1)
        )
    else:
        results = [build_one(index, well) for index, well in enumerate(wells, start=1)]
    prediction = pd.concat([row[0] for row in results], ignore_index=True)
    weight_audit = pd.concat([row[1] for row in results], ignore_index=True)
    runtime = pd.concat([row[2] for row in results], ignore_index=True)
    prediction = prediction.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    weight_audit = weight_audit.sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    runtime = runtime.sort_values(["well_id", "variant_index"], kind="mergesort").reset_index(
        drop=True
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or prediction["id"].duplicated().any()
        or len(weight_audit) != expected_rows
        or weight_audit["id"].duplicated().any()
        or len(runtime) != int(get_nested(config, "model.execution_counts.hmm_well_runs"))
    ):
        raise ValueError("exp308 prediction/weight/runtime coverage mismatch")
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_predictions.csv.gz"
    weight_path = artifacts / f"{OUTPUT_PREFIX}_weight_audit.csv.gz"
    runtime_path = artifacts / f"{OUTPUT_PREFIX}_by_well_variant_runtime.csv"
    prediction.to_csv(prediction_path, index=False, compression="gzip")
    weight_audit.to_csv(weight_path, index=False, compression="gzip")
    runtime.to_csv(runtime_path, index=False)
    prediction_report = inspect_gzip_csv(prediction_path)
    weight_report = inspect_gzip_csv(weight_path)
    prediction_report["frozen_before_truth_attachment"] = True
    weight_report["frozen_before_truth_attachment"] = True
    reports = {
        "prediction": prediction_report,
        "weight_audit": weight_report,
        "runtime": {
            "path": str(runtime_path),
            "bytes": runtime_path.stat().st_size,
            "raw_sha256": sha256_path(runtime_path),
            "rows": len(runtime),
        },
        "elapsed_seconds": time.time() - started,
        "numba_num_threads": get_num_threads(),
        "outer_workers": outer_workers,
    }
    return (
        reports,
        {"prediction": prediction_path, "weight_audit": weight_path, "runtime": runtime_path},
        runtime,
    )


# %% [markdown]
# ## 7. Late truth/parent/control attachment and paired metrics


# %%
def _require_frozen_prediction(report: dict[str, Any]) -> None:
    if (
        not bool(report.get("frozen_before_truth_attachment"))
        or len(str(report.get("decompressed_sha256", ""))) != 64
        or report.get("content_sha256") != report.get("decompressed_sha256")
    ):
        raise RuntimeError("late truth attachment requires a frozen prediction content SHA")


def _assert_same_order(label: str, expected: pd.Series, actual: pd.Series) -> None:
    expected_values = expected.astype(str).to_numpy()
    actual_values = actual.astype(str).to_numpy()
    if len(expected_values) != len(actual_values) or not np.array_equal(
        expected_values, actual_values
    ):
        raise ValueError(f"{label} row identity/order mismatch")


def load_late_readout_frame(
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    output_paths: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_frozen_prediction(frozen["prediction"])
    _require_frozen_prediction(frozen["weight_audit"])
    prediction = (
        pd.read_csv(
            output_paths["prediction"],
            dtype={"id": str, "well_id": str},
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    weight_audit = (
        pd.read_csv(
            output_paths["weight_audit"],
            dtype={"id": str, "well_id": str},
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    control = get_nested(config, "data.saved_controls") or {}
    parent_spec = get_nested(config, "data.parent_exp307") or {}
    exp072 = pd.read_csv(
        preflight["controls"]["paths"]["saved_exp072"],
        usecols=["id", "well", "md_since", str(control["likpf_prediction_column"])],
        dtype={"id": str, "well": str},
    )
    parent_prediction = pd.read_csv(
        preflight["controls"]["paths"]["parent_prediction"],
        usecols=["id", "well_id", "row_idx", str(parent_spec["prediction_column"])],
        dtype={"id": str, "well_id": str},
    )
    exp072["row_idx"] = pd.to_numeric(
        exp072["id"].astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
    ).astype(np.int64)
    exp072.sort_values(["well", "row_idx"], kind="mergesort", inplace=True)
    exp072.reset_index(drop=True, inplace=True)
    parent_prediction["row_idx"] = pd.to_numeric(
        parent_prediction["row_idx"], errors="raise"
    ).astype(np.int64)
    parent_prediction.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    parent_prediction.reset_index(drop=True, inplace=True)
    _assert_same_order("prediction vs weight audit", prediction["id"], weight_audit["id"])
    _assert_same_order("prediction vs exp072", prediction["id"], exp072["id"])
    _assert_same_order("prediction vs parent exp307", prediction["id"], parent_prediction["id"])
    fold = get_nested(config, "data.fold_assignment") or {}
    truth_columns = [str(value) for value in fold["truth_columns"]]
    truth = pd.read_csv(
        preflight["controls"]["paths"]["fold_assignment"],
        usecols=[*truth_columns, "fold"],
        dtype={"well_id": str},
    )
    truth["row_idx"] = pd.to_numeric(truth["row_idx"], errors="raise").astype(np.int64)
    truth["id"] = truth["well_id"].astype(str) + "_" + truth["row_idx"].astype(str)
    truth.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    truth.reset_index(drop=True, inplace=True)
    _assert_same_order("prediction vs late truth", prediction["id"], truth["id"])
    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    role_columns = [str(value) for value in hidden["role_columns"].values()]
    hidden_frame = pd.read_csv(
        preflight["controls"]["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns],
        dtype={"well_id": str},
    ).set_index("well_id")
    frame = pd.DataFrame(
        {
            "id": prediction["id"].astype(str),
            "well_id": prediction["well_id"].astype(str),
            "row_idx": prediction["row_idx"].to_numpy(np.int64),
            "fold": pd.to_numeric(truth["fold"], errors="raise").to_numpy(np.int64),
            "true_tvt": pd.to_numeric(truth["tvt_true"], errors="raise").to_numpy(np.float64),
            "md_since": pd.to_numeric(exp072["md_since"], errors="raise").to_numpy(np.float64),
            "parent_hmm_tvt": pd.to_numeric(
                parent_prediction[str(parent_spec["prediction_column"])], errors="raise"
            ).to_numpy(np.float64),
            "likpf_mean": pd.to_numeric(
                exp072[str(control["likpf_prediction_column"])], errors="raise"
            ).to_numpy(np.float64),
            "raw_gr_missing": weight_audit["raw_gr_missing"].astype(bool).to_numpy(),
            "raw_gr_finite": weight_audit["raw_gr_finite"].astype(bool).to_numpy(),
            "nearest_finite_row_distance": pd.to_numeric(
                weight_audit["nearest_finite_row_distance"], errors="raise"
            ).to_numpy(np.int64),
            "confidence_weight": pd.to_numeric(
                weight_audit["confidence_weight"], errors="raise"
            ).to_numpy(np.float64),
            "missing_run_length": pd.to_numeric(
                weight_audit["missing_run_length"], errors="raise"
            ).to_numpy(np.int64),
            "gap_bucket": weight_audit["gap_bucket"].astype(str).to_numpy(),
        }
    )
    for variant in VARIANT_ORDER:
        frame[f"{variant}_hmm_tvt"] = pd.to_numeric(
            prediction[f"{variant}_hmm_tvt"], errors="raise"
        ).to_numpy(np.float64)
        frame[f"{variant}_likpf_50_50"] = (
            0.5 * frame[f"{variant}_hmm_tvt"] + 0.5 * frame["likpf_mean"]
        )
    frame["parent_hmm_likpf_50_50"] = (
        0.5 * frame["parent_hmm_tvt"] + 0.5 * frame["likpf_mean"]
    )
    for scope, role_column in hidden["role_columns"].items():
        frame[str(scope)] = (
            frame["well_id"].map(hidden_frame[role_column].astype(str)).eq("valid").to_numpy()
        )
    numeric = [
        "true_tvt",
        "md_since",
        "parent_hmm_tvt",
        "likpf_mean",
        "parent_hmm_likpf_50_50",
        "confidence_weight",
        *[f"{variant}_hmm_tvt" for variant in VARIANT_ORDER],
        *[f"{variant}_likpf_50_50" for variant in VARIANT_ORDER],
    ]
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise ValueError("late readout contains non-finite values")
    return frame, {
        "truth_attachment_stage": "after_weight_and_prediction_gzip_content_sha_freeze",
        "prediction_content_sha256": frozen["prediction"]["content_sha256"],
        "weight_content_sha256": frozen["weight_audit"]["content_sha256"],
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "id_mismatches": 0,
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def paired_metric_row(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    variant: str,
    comparison: str,
    candidate_column: str,
    control_column: str,
    scope: str,
) -> dict[str, Any]:
    if not bool(mask.any()):
        raise ValueError(f"paired scope {scope} selected zero rows")
    truth = frame.loc[mask, "true_tvt"].to_numpy(np.float64)
    candidate = frame.loc[mask, candidate_column].to_numpy(np.float64)
    control = frame.loc[mask, control_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "variant": variant,
        "comparison": comparison,
        "scope": scope,
        "rows": int(mask.sum()),
        "wells": int(frame.loc[mask, "well_id"].nunique()),
        "candidate_column": candidate_column,
        "control_column": control_column,
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
    }


def build_paired_metrics(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(frame), dtype=bool))]
    for fold in [int(value) for value in get_nested(config, "validation.expected_folds")]:
        scopes.append((f"fold_{fold}", frame["fold"].to_numpy(np.int64) == fold))
    scopes.extend(
        [
            ("md_since_1000_plus", frame["md_since"].to_numpy(np.float64) >= 1000.0),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
            ("gr_observed", frame["raw_gr_finite"].to_numpy(bool)),
            ("gr_missing", frame["raw_gr_missing"].to_numpy(bool)),
        ]
    )
    for bucket in ("gap_1_3", "gap_4_15", "gap_16_plus"):
        mask = frame["gap_bucket"].astype(str).eq(bucket).to_numpy()
        if mask.any():
            scopes.append((bucket, mask))
    rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        comparisons = {
            "direct": (f"{variant}_hmm_tvt", "parent_hmm_tvt"),
            "fixed_likpf_50_50": (
                f"{variant}_likpf_50_50",
                "parent_hmm_likpf_50_50",
            ),
        }
        for comparison, (candidate_column, control_column) in comparisons.items():
            for scope, mask in scopes:
                rows.append(
                    paired_metric_row(
                        frame,
                        mask,
                        variant=variant,
                        comparison=comparison,
                        candidate_column=candidate_column,
                        control_column=control_column,
                        scope=scope,
                    )
                )
            distance_frame = frame.loc[
                frame["raw_gr_missing"],
                [
                    "well_id",
                    "nearest_finite_row_distance",
                    "true_tvt",
                    candidate_column,
                    control_column,
                ],
            ].copy()
            distance_frame["candidate_squared_error"] = (
                distance_frame[candidate_column] - distance_frame["true_tvt"]
            ) ** 2
            distance_frame["control_squared_error"] = (
                distance_frame[control_column] - distance_frame["true_tvt"]
            ) ** 2
            distance_summary = distance_frame.groupby(
                "nearest_finite_row_distance", sort=True
            ).agg(
                rows=("well_id", "size"),
                wells=("well_id", "nunique"),
                candidate_squared_error=("candidate_squared_error", "sum"),
                control_squared_error=("control_squared_error", "sum"),
            )
            for distance, record in distance_summary.iterrows():
                distance_value = int(distance)
                candidate_rmse = math.sqrt(
                    float(record["candidate_squared_error"]) / int(record["rows"])
                )
                control_rmse = math.sqrt(
                    float(record["control_squared_error"]) / int(record["rows"])
                )
                scope = (
                    "distance_no_finite_fallback"
                    if distance_value < 0
                    else f"distance_{distance_value}"
                )
                rows.append(
                    {
                        "variant": variant,
                        "comparison": comparison,
                        "scope": scope,
                        "rows": int(record["rows"]),
                        "wells": int(record["wells"]),
                        "candidate_column": candidate_column,
                        "control_column": control_column,
                        "candidate_rmse": candidate_rmse,
                        "control_rmse": control_rmse,
                        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
                        "improvement_ft": control_rmse - candidate_rmse,
                    }
                )
            for well, group in frame.groupby("well_id", sort=True):
                truth = group["true_tvt"].to_numpy(np.float64)
                candidate_rmse = rmse(truth, group[candidate_column].to_numpy(np.float64))
                control_rmse = rmse(truth, group[control_column].to_numpy(np.float64))
                by_well_rows.append(
                    {
                        "variant": variant,
                        "comparison": comparison,
                        "well_id": str(well),
                        "rows": len(group),
                        "candidate_rmse": candidate_rmse,
                        "control_rmse": control_rmse,
                        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(by_well_rows)


# %% [markdown]
# ## 8. Promotion gate and generated artifacts


# %%
def evaluate_promotion_gate(
    paired_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    frame: pd.DataFrame,
    runtime: pd.DataFrame,
    weight_audit: pd.DataFrame,
    preflight: dict[str, Any],
    runtime_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "gate") or {}
    tolerance = float(gate["non_regression_tolerance_ft"])
    baseline_tolerance = float(gate["baseline_metric_absolute_tolerance"])
    primary = VARIANT_ORDER[0]
    direct = paired_metrics.loc[
        (paired_metrics["variant"] == primary) & (paired_metrics["comparison"] == "direct")
    ]
    blend = paired_metrics.loc[
        (paired_metrics["variant"] == primary)
        & (paired_metrics["comparison"] == "fixed_likpf_50_50")
    ]
    direct_overall = direct.loc[direct["scope"] == "overall"].iloc[0]
    blend_overall = blend.loc[blend["scope"] == "overall"].iloc[0]
    likpf_actual = rmse(
        frame["true_tvt"].to_numpy(np.float64),
        frame["likpf_mean"].to_numpy(np.float64),
    )
    baseline_parity = {
        "parent_exp307_hmm": {
            "actual": float(direct_overall["control_rmse"]),
            "expected": float(get_nested(config, "references.parent_exp307_rmse")),
        },
        "likpf": {
            "actual": likpf_actual,
            "expected": float(get_nested(config, "references.likpf_rmse")),
        },
        "parent_exp307_likpf_50_50": {
            "actual": float(blend_overall["control_rmse"]),
            "expected": float(get_nested(config, "references.parent_exp307_likpf_50_50_rmse")),
        },
    }
    for record in baseline_parity.values():
        record["absolute_difference"] = abs(record["actual"] - record["expected"])
        record["passed"] = bool(record["absolute_difference"] <= baseline_tolerance)
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    finite_columns = [
        *[f"{variant}_hmm_tvt" for variant in VARIANT_ORDER],
        *[f"{variant}_likpf_50_50" for variant in VARIANT_ORDER],
    ]
    observed_weight = weight_audit.loc[
        weight_audit["raw_gr_finite"].astype(bool), "confidence_weight"
    ].to_numpy(np.float64)
    missing_weight = weight_audit.loc[
        weight_audit["raw_gr_missing"].astype(bool), "confidence_weight"
    ].to_numpy(np.float64)
    missing_distance = weight_audit.loc[
        weight_audit["raw_gr_missing"].astype(bool), "nearest_finite_row_distance"
    ].to_numpy(np.int64)
    expected_missing_weight = np.where(
        missing_distance < 0,
        0.25,
        np.maximum(0.25, np.exp2(-missing_distance.astype(np.float64) / 8.0)),
    )
    gap_scopes = set(
        paired_metrics.loc[
            paired_metrics["comparison"] == "direct", "scope"
        ].astype(str)
    )
    technical = {
        "input_preflight_passed": True,
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "weight_audit_rows": len(weight_audit),
        "weight_audit_wells": int(weight_audit["well_id"].nunique()),
        "finite_coverage": float(np.isfinite(frame[finite_columns].to_numpy(np.float64)).mean()),
        "id_mismatches": 0,
        "hmm_well_runs": len(runtime),
        "expected_hmm_well_runs": int(get_nested(config, "model.execution_counts.hmm_well_runs")),
        "variant_order": runtime["variant"].drop_duplicates().tolist(),
        "variant_run_counts": runtime["variant"].value_counts().sort_index().to_dict(),
        "posterior_normalization_max_abs_error": float(
            runtime["posterior_row_sum_max_abs_error"].max()
        ),
        "observed_weight_exact_one": bool(
            len(observed_weight) > 0
            and np.array_equal(observed_weight, np.ones(len(observed_weight)))
        ),
        "missing_weight_formula_exact": bool(
            len(missing_weight) > 0 and np.array_equal(missing_weight, expected_missing_weight)
        ),
        "missing_weight_min": float(missing_weight.min()) if len(missing_weight) else None,
        "missing_weight_max": float(missing_weight.max()) if len(missing_weight) else None,
        "gap_bucket_readout_complete": {"gap_1_3", "gap_4_15", "gap_16_plus"}.issubset(
            gap_scopes
        ),
        "parent_dependency_checks": preflight["controls"]["parent_exp307"]["checks"],
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.kaggle.runtime_limit_seconds")),
        "baseline_metric_parity": baseline_parity,
        "raw_identity_sha256": preflight["raw_train"]["content_sha256"],
    }
    technical["passed"] = bool(
        technical["prediction_rows"] == expected_rows
        and technical["prediction_wells"] == expected_wells
        and technical["weight_audit_rows"] == expected_rows
        and technical["weight_audit_wells"] == expected_wells
        and technical["finite_coverage"] == 1.0
        and technical["id_mismatches"] == 0
        and technical["hmm_well_runs"] == technical["expected_hmm_well_runs"]
        and technical["variant_order"] == list(VARIANT_ORDER)
        and technical["variant_run_counts"]
        == {variant: expected_wells for variant in sorted(VARIANT_ORDER)}
        and technical["posterior_normalization_max_abs_error"] <= 1.0e-6
        and technical["observed_weight_exact_one"]
        and technical["missing_weight_formula_exact"]
        and technical["gap_bucket_readout_complete"]
        and all(technical["parent_dependency_checks"].values())
        and runtime_seconds <= technical["runtime_limit_seconds"]
        and all(bool(record["passed"]) for record in baseline_parity.values())
    )
    folds = direct.loc[direct["scope"].str.startswith("fold_")]
    folds_improved = int((folds["delta_rmse_candidate_minus_control"] < -tolerance).sum())
    required_scope_names = (
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    )
    scope_checks = {
        scope: bool(
            direct.loc[
                direct["scope"] == scope,
                "delta_rmse_candidate_minus_control",
            ].iloc[0]
            <= tolerance
        )
        for scope in required_scope_names
    }
    primary_by_well = by_well.loc[
        (by_well["variant"] == primary) & (by_well["comparison"] == "direct")
    ]
    candidate_p95 = float(primary_by_well["candidate_rmse"].quantile(0.95))
    control_p95 = float(primary_by_well["control_rmse"].quantile(0.95))
    p95_delta = candidate_p95 - control_p95
    worst_delta = float(primary_by_well["delta_rmse_candidate_minus_control"].max())
    primary_direct = {
        "candidate_rmse": float(direct_overall["candidate_rmse"]),
        "control_rmse": float(direct_overall["control_rmse"]),
        "improvement_ft": float(direct_overall["improvement_ft"]),
        "minimum_improvement_ft": float(gate["min_rmse_improvement_vs_parent"]),
        "folds_improved": folds_improved,
        "minimum_folds_improved": int(gate["min_improved_folds"]),
        "required_scope_checks": scope_checks,
        "by_well_candidate_rmse_p95": candidate_p95,
        "by_well_control_rmse_p95": control_p95,
        "by_well_rmse_p95_delta": p95_delta,
        "worst_well_rmse_delta": worst_delta,
        "worst_well_rmse_delta_max": float(gate["maximum_worst_well_regression"]),
    }
    primary_direct["passed"] = bool(
        primary_direct["improvement_ft"] >= primary_direct["minimum_improvement_ft"]
        and folds_improved >= primary_direct["minimum_folds_improved"]
        and all(scope_checks.values())
        and p95_delta <= tolerance
        and worst_delta <= primary_direct["worst_well_rmse_delta_max"]
    )
    blend_guard = {
        "candidate_rmse": float(blend_overall["candidate_rmse"]),
        "control_rmse": float(blend_overall["control_rmse"]),
        "delta_rmse_candidate_minus_control": float(
            blend_overall["delta_rmse_candidate_minus_control"]
        ),
    }
    blend_guard["passed"] = bool(blend_guard["delta_rmse_candidate_minus_control"] <= tolerance)
    passed = bool(technical["passed"] and primary_direct["passed"] and blend_guard["passed"])
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            "missing_distance_confidence_passed_train_side_only_no_automatic_downstream"
            if passed
            else "missing_distance_confidence_failed_close_without_rescue"
        ),
        "technical_gate": technical,
        "primary_direct_gate": primary_direct,
        "fixed_likpf_50_50_guard": blend_guard,
        "failure_action": "close_without_half_life_floor_grid_pf_port_inference_or_submission",
    }


def output_file_reports(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "raw_sha256": sha256_path(path),
        }
        for name, path in paths.items()
    }


def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp308 must run first on Kaggle; local execution requires explicit smoke approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    control_preflight = preflight_controls_and_assignments(config)
    wells = control_preflight["fold_assignment"].pop("well_ids")
    sigma_by_well = control_preflight["parent_exp307"].pop("sigma_by_well")
    preflight = {
        "experiment": EXPERIMENT_NAME,
        "raw_train": raw_preflight,
        "controls": control_preflight,
        "truth_attached": False,
    }
    scientific_contract = build_scientific_contract(config)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_control_manifest.json"
    write_json(contract_path, scientific_contract)
    write_json(manifest_path, preflight)

    frozen, output_paths, runtime = generate_and_freeze_predictions(
        raw_dir, artifacts, config, wells, sigma_by_well
    )
    prediction_frozen_at_seconds = time.time() - started
    weight_audit = pd.read_csv(output_paths["weight_audit"], dtype={"well_id": str})
    # Unknown-suffix truth and row-level saved controls are first parsed here.
    frame, late_attachment = load_late_readout_frame(preflight, frozen, output_paths, config)
    paired_metrics, by_well_metrics = build_paired_metrics(frame, config)
    runtime_seconds = time.time() - started
    promotion_gate = evaluate_promotion_gate(
        paired_metrics,
        by_well_metrics,
        frame,
        runtime,
        weight_audit,
        preflight,
        runtime_seconds,
        config,
    )
    output_metric_paths = {
        "overall_fold_scope_metrics": artifacts / f"{OUTPUT_PREFIX}_overall_fold_scope_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
    }
    paired_metrics.to_csv(output_metric_paths["overall_fold_scope_metrics"], index=False)
    by_well_metrics.to_csv(output_metric_paths["by_well_metrics"], index=False)
    write_json(output_metric_paths["promotion_gate"], promotion_gate)
    status = (
        "train_side_missing_distance_confidence_gate_passed_no_automatic_downstream"
        if promotion_gate["passed"]
        else "train_side_missing_distance_confidence_gate_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "runtime_seconds": runtime_seconds,
        "prediction_frozen_at_seconds": prediction_frozen_at_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "active_scientific_variants": 1,
        "hmm_well_runs": len(runtime),
        "models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "boosters": 0,
        "control_reruns": 0,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "input_control_manifest_sha256": sha256_path(manifest_path),
        "frozen_outputs": frozen,
        "truth_attachment": late_attachment,
        "promotion_gate": promotion_gate,
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": None,
        "submission_sha256": None,
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    summary["generated_files"] = output_file_reports(
        {
            **output_metric_paths,
            "scientific_contract": contract_path,
            "input_control_manifest": manifest_path,
            "prediction": Path(output_paths["prediction"]),
            "weight_audit": Path(output_paths["weight_audit"]),
            "by_well_variant_runtime": Path(output_paths["runtime"]),
        }
    )
    write_json(summary_path, summary)
    primary_overall = paired_metrics.loc[
        (paired_metrics["variant"] == VARIANT_ORDER[0]) & (paired_metrics["scope"] == "overall")
    ]
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": (
            float(
                primary_overall.loc[
                    primary_overall["comparison"] == "direct", "candidate_rmse"
                ].iloc[0]
            )
            if promotion_gate["passed"]
            else None
        ),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": primary_overall.to_dict(orient="records"),
        "promotion_gate": promotion_gate,
        "prediction_sha256": frozen["prediction"],
        "weight_audit_sha256": frozen["weight_audit"],
        "model_sha256": None,
        "submission_sha256": None,
        "notes": "Train-side only; no raw-test prediction, inference, or submission is produced.",
    }
    write_json(metrics_output_path(), metrics)
    print(primary_overall.to_string(index=False))
    print(json.dumps(to_jsonable(promotion_gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and configuration preview


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
                "variants": list(VARIANT_ORDER),
                "hmm_well_runs": get_nested(CONFIG, "model.execution_counts.hmm_well_runs"),
                "control_reruns": 0,
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 10. Run the Kaggle CPU audit


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_full_experiment(CONFIG)
