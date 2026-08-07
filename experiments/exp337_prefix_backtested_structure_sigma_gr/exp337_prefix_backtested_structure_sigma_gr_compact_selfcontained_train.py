# %% [markdown]
# # exp337 prefix-backtested structure sigma GR train
#
# Stage 0 is a deterministic, target-free rolling-origin audit. It compares
# three fixed GR emission scales on future blocks inside the visible
# `TVT_input` prefix. It does not run an HMM, read unknown-suffix truth, train a
# model, or create predictions/submissions.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and dependency preflight
# 4. Prefix residual and structural-scale helpers
# 5. Rolling-origin Stage 0 audit
# 6. Stage 0 promotion gate
# 7. Metrics and generated artifacts
# 8. Setup and configuration preview
# 9. Run the approved Stage 0 audit only

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp337_prefix_backtested_structure_sigma_gr"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
ORIGIN_ORDER = (0.60, 0.80)
SCALE_POLICY_ORDER = ("finite_only", "exp209_zero_fill", "structure_added")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP337_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
    raise FileNotFoundError(f"exp337 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir") or "data/raw/train"))
    root = project_root()
    candidates = (
        configured,
        root / configured,
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
    )
    for path in candidates:
        if path.exists() and any(path.glob("*__horizontal_well.csv")):
            return path
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob("**/train/*__horizontal_well.csv"))
        if matches:
            return matches[0].parent
    checked = [str(path) for path in candidates]
    raise FileNotFoundError(f"raw train directory not found; checked={checked}")


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


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


# %% [markdown]
# ## 3. Frozen scientific contract and dependency preflight


# %%
def validate_scientific_contract(
    config: dict[str, Any], *, require_run_approval: bool = False
) -> None:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "implementation.enabled": True,
        "implementation.scope": "stage0_only",
        "implementation.canonical_notebook_adopted": True,
        "model.residual_contract.pair_policy": (
            "finite_horizontal_gr_and_finite_typewell_interp_only"
        ),
        "model.residual_contract.formula": "horizontal_gr_minus_typewell_gr_at_tvt_input",
        "model.residual_contract.affine_a": 1.0,
        "model.residual_contract.affine_b": 0.0,
        "model.residual_contract.order": "raw_known_prefix_row_order",
        "model.scale_estimator.internal_fit_fraction": 0.60,
        "model.scale_estimator.internal_calibration_fraction": 0.40,
        "model.scale_estimator.minimum_total_finite_pairs": 50,
        "model.scale_estimator.minimum_early_finite_pairs": 20,
        "model.scale_estimator.minimum_late_finite_pairs": 20,
        "model.scale_estimator.finite_sigma": (
            "population_standard_deviation_early_finite_pairs"
        ),
        "model.scale_estimator.finite_only_comparator_sigma": (
            "population_standard_deviation_all_available_finite_pairs"
        ),
        "model.scale_estimator.structure_variance": (
            "max_0_mean_squared_late_residual_minus_finite_variance"
        ),
        "model.scale_estimator.effective_variance": (
            "early_finite_variance_plus_structure_variance"
        ),
        "model.scale_estimator.fallback": "exp209_zero_fill_sigma_on_same_available_prefix",
        "model.stage_0_prefix_backtest.enabled_after_implementation": True,
        "model.stage_0_prefix_backtest.rolling_origins": [0.60, 0.80],
        "model.stage_0_prefix_backtest.forward_evaluation_fraction": 0.20,
        "model.stage_0_prefix_backtest.scale_policies": list(SCALE_POLICY_ORDER),
        "model.stage_0_prefix_backtest.gaussian_nll_without_constant": (
            "log_sigma_plus_half_squared_residual_over_variance"
        ),
        "model.stage_1_exact_hmm.enabled_after_implementation": False,
        "execution_contract.stage_0.diagnostic_variants": 1,
        "execution_contract.stage_0.hmm_well_runs": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.pf_well_runs": 0,
        "execution_contract.stage_0.beam_well_runs": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
        "execution.implementation_approval_scope": "user_message_implement_exp337",
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp337 fixed contract mismatch: {key} must be {value!r}")
    expected_stage0_gate = {
        "minimum_evaluable_well_fraction_each_origin": 0.90,
        "maximum_fallback_well_fraction_each_origin": 0.10,
        "structure_beats_finite_pooled_each_origin": True,
        "minimum_folds_structure_beats_finite_each_origin": 4,
        "minimum_nll_gain_vs_zero_fill_per_finite_residual_each_origin": 0.005,
        "minimum_folds_structure_beats_zero_fill_each_origin": 4,
        "minimum_final_median_tau_structure_gr_units": 5.0,
        "maximum_final_lower_clip_well_fraction": 0.10,
    }
    actual_stage0_gate = get_nested(
        config, "model.stage_0_prefix_backtest.pass_requires_all"
    )
    if actual_stage0_gate != expected_stage0_gate:
        raise ValueError("exp337 fixed Stage 0 gate contract changed")
    if [float(value) for value in get_nested(config, "model.scale_estimator.clip")] != [10.0, 60.0]:
        raise ValueError("exp337 fixes the GR scale clip to [10, 60]")
    configured_origins = get_nested(config, "model.stage_0_prefix_backtest.rolling_origins")
    if tuple(float(value) for value in configured_origins) != ORIGIN_ORDER:
        raise ValueError("exp337 fixes rolling origins to (0.60, 0.80)")
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
            raise ValueError(f"exp337 fixes model.fixed_hmm.{key}={value}")
    fixed_policies = {
        "emission": "gaussian",
        "rate_center": "zero",
        "evaluation_gr_policy": "interpolate_both_directions_then_typewell_mean",
        "typewell_gr_policy": "sort_tvt_ffill_bfill_then_linear_interp",
        "output": "posterior_mean",
    }
    for key, value in fixed_policies.items():
        if hmm.get(key) != value:
            raise ValueError(f"exp337 fixes model.fixed_hmm.{key}={value}")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise RuntimeError("exp337 Stage 0 Kaggle package/push/run is not approved")


def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "active_stage": "stage0_known_prefix_rolling_origin_only",
        "unknown_suffix_truth_read": False,
        "residual_contract": get_nested(config, "model.residual_contract"),
        "scale_estimator": get_nested(config, "model.scale_estimator"),
        "stage0": get_nested(config, "model.stage_0_prefix_backtest"),
        "stage1_enabled": False,
        "fixed_hmm_reference_only_not_executed": get_nested(config, "model.fixed_hmm"),
        "execution_counts": get_nested(config, "execution_contract.stage_0"),
        "parent_control_retraining": False,
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "forbidden": get_nested(config, "model.forbidden"),
        "seed_policy": get_nested(config, "reproducibility.seed_policy"),
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
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "well_ids": frame["well_id"].tolist(),
        "content_sha256": actual,
    }


def load_fold_map(
    path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_folds: list[int],
) -> dict[str, int]:
    parts: list[pd.DataFrame] = []
    row_count = 0
    chunks = pd.read_csv(
        path,
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
        chunksize=250_000,
    )
    for chunk in chunks:
        chunk["fold"] = pd.to_numeric(chunk["fold"], errors="raise").astype(np.int64)
        row_count += len(chunk)
        if (chunk.groupby("well_id", sort=False)["fold"].nunique() > 1).any():
            raise ValueError("fold assignment changes inside a well")
        parts.append(chunk.drop_duplicates(["well_id", "fold"]))
    pairs = pd.concat(parts, ignore_index=True).drop_duplicates(["well_id", "fold"])
    if (pairs.groupby("well_id", sort=False)["fold"].nunique() > 1).any():
        raise ValueError("fold assignment changes across chunks inside a well")
    fold_frame = pairs.drop_duplicates("well_id").sort_values("well_id", kind="mergesort")
    if (
        row_count != expected_rows
        or len(fold_frame) != expected_wells
        or sorted(fold_frame["fold"].unique().tolist()) != expected_folds
    ):
        raise ValueError("fold assignment row/well/fold coverage mismatch")
    return dict(zip(fold_frame["well_id"], fold_frame["fold"].astype(int), strict=True))


def preflight_dependencies(
    config: dict[str, Any], raw_dir: Path
) -> tuple[dict[str, Any], dict[str, int]]:
    raw_report = validate_raw_well_identity(config, raw_dir)
    control = get_nested(config, "data.saved_controls") or {}
    fold = get_nested(config, "data.fold_assignment") or {}
    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    negative = get_nested(config, "data.negative_evidence") or {}
    paths = {
        "saved_hmm": resolve_existing(
            str(control["hmm_cache_filename"]), _candidate_paths(control)
        ),
        "saved_exp072": resolve_existing(
            str(control["exp072_cache_filename"]), _candidate_paths(control)
        ),
        "fold_assignment": resolve_existing(str(fold["filename"]), _candidate_paths(fold)),
        "hidden_like_assignment": resolve_existing(
            str(hidden["filename"]), _candidate_paths(hidden)
        ),
        "exp307_scale_audit": resolve_existing(
            str(negative["scale_audit_filename"]), _candidate_paths(negative)
        ),
        "exp307_summary": resolve_existing(
            str(negative["summary_filename"]), _candidate_paths(negative)
        ),
    }
    reports = {
        "saved_hmm": inspect_gzip_csv(paths["saved_hmm"]),
        "saved_exp072": inspect_gzip_csv(paths["saved_exp072"]),
        "fold_assignment": inspect_gzip_csv(paths["fold_assignment"]),
        "exp307_scale_audit": inspect_gzip_csv(paths["exp307_scale_audit"]),
    }
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if reports["saved_hmm"]["decompressed_sha256"] != str(
        control["expected_hmm_prediction_decompressed_sha256"]
    ):
        raise ValueError("saved exp209 HMM decompressed SHA mismatch")
    if reports["saved_exp072"]["decompressed_sha256"] != str(
        control["expected_exp072_cache_decompressed_sha256"]
    ):
        raise ValueError("saved exp072 cache decompressed SHA mismatch")
    if any(reports[key]["data_rows"] != expected_rows for key in ("saved_hmm", "saved_exp072")):
        raise ValueError("saved exp209 control row coverage mismatch")
    hmm_columns = set(pd.read_csv(paths["saved_hmm"], nrows=0).columns.astype(str))
    exp072_columns = set(pd.read_csv(paths["saved_exp072"], nrows=0).columns.astype(str))
    required_hmm = {"id", "well", str(control["raw_hmm_prediction_column"])}
    required_exp072 = {
        "id",
        "well",
        "md_since",
        str(control["likpf_anchor_column"]),
        str(control["likpf_delta_column"]),
    }
    if not required_hmm.issubset(hmm_columns) or not required_exp072.issubset(exp072_columns):
        raise ValueError("saved exp209 control column contract mismatch")
    if reports["fold_assignment"]["decompressed_sha256"] != str(
        fold["expected_decompressed_sha256"]
    ):
        raise ValueError("fold assignment decompressed SHA mismatch")
    fold_map = load_fold_map(
        paths["fold_assignment"],
        expected_rows=expected_rows,
        expected_wells=expected_wells,
        expected_folds=[int(value) for value in get_nested(config, "validation.expected_folds")],
    )
    hidden_sha = sha256_path(paths["hidden_like_assignment"])
    if hidden_sha != str(hidden["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    hidden_frame = pd.read_csv(paths["hidden_like_assignment"], dtype={"well_id": str})
    role_columns = {str(value) for value in hidden["role_columns"].values()}
    if not {"well_id", *role_columns}.issubset(hidden_frame.columns):
        raise ValueError("hidden-like assignment column contract mismatch")
    hidden_wells = sorted(hidden_frame["well_id"])
    if hidden_frame["well_id"].duplicated().any() or hidden_wells != sorted(fold_map):
        raise ValueError("hidden-like assignment well identity mismatch")
    if reports["exp307_scale_audit"]["decompressed_sha256"] != str(
        negative["expected_scale_decompressed_sha256"]
    ):
        raise ValueError("exp307 scale audit decompressed SHA mismatch")
    scale_columns = set(pd.read_csv(paths["exp307_scale_audit"], nrows=0).columns.astype(str))
    if reports["exp307_scale_audit"]["data_rows"] != expected_wells or not {
        "well_id",
        "current_zero_fill_std",
        "finite_std",
        "finite_mad",
    }.issubset(scale_columns):
        raise ValueError("exp307 scale audit coverage/column contract mismatch")
    summary_sha = sha256_path(paths["exp307_summary"])
    summary = json.loads(paths["exp307_summary"].read_text())
    if summary_sha != str(negative["expected_summary_sha256"]):
        raise ValueError("exp307 summary SHA mismatch")
    if (
        summary.get("status") != negative["expected_status"]
        or bool(summary.get("promotion_gate", {}).get("passed"))
        != bool(negative["expected_gate_passed"])
    ):
        raise ValueError("exp307 negative-result contract mismatch")
    raw_wells = sorted(str(value) for value in raw_report.pop("well_ids"))
    if raw_wells != sorted(fold_map):
        raise ValueError("raw/fold well identity mismatch")
    report = {
        "raw_train": raw_report,
        "paths": {key: str(value) for key, value in paths.items()},
        "saved_hmm": {**reports["saved_hmm"], "columns": sorted(hmm_columns)},
        "saved_exp072": {**reports["saved_exp072"], "columns": sorted(exp072_columns)},
        "fold_assignment": {**reports["fold_assignment"], "wells": len(fold_map)},
        "hidden_like_assignment": {
            "path": str(paths["hidden_like_assignment"]),
            "raw_sha256": hidden_sha,
            "wells": int(hidden_frame["well_id"].nunique()),
        },
        "exp307_negative_evidence": {
            "scale_audit": reports["exp307_scale_audit"],
            "summary_path": str(paths["exp307_summary"]),
            "summary_raw_sha256": summary_sha,
            "status": summary["status"],
            "gate_passed": bool(summary["promotion_gate"]["passed"]),
        },
        "unknown_suffix_truth_read": False,
    }
    report["dependency_contract_sha256"] = mapping_sha256(report)
    return report, fold_map


# %% [markdown]
# ## 4. Prefix residual and structural-scale helpers


# %%
def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["GR", "TVT_input"],
    )
    if list(frame.columns) != ["GR", "TVT_input"]:
        frame = frame[["GR", "TVT_input"]]
    return frame.reset_index(drop=True)


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.sort_values("TVT", kind="mergesort").reset_index(drop=True)
    frame["GR"] = frame["GR"].ffill().bfill()
    valid = np.isfinite(frame["TVT"].to_numpy(np.float64)) & np.isfinite(
        frame["GR"].to_numpy(np.float64)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    if len(frame) < 2 or np.any(np.diff(frame["TVT"].to_numpy(np.float64)) < 0.0):
        raise ValueError("typewell TVT/GR contract is invalid")
    return frame


def build_prefix_residual_frame(
    horizontal: pd.DataFrame, typewell: pd.DataFrame
) -> pd.DataFrame:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    if np.isinf(tvt_input).any():
        raise ValueError("TVT_input may contain finite values or NaN only")
    known_index = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_index) == 0 or not np.array_equal(known_index, np.arange(len(known_index))):
        raise ValueError("exp337 requires one contiguous known TVT_input prefix")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    typewell_at_known = np.interp(tvt_input[known_index], typewell_tvt, typewell_gr)
    observed = horizontal_gr[known_index]
    residual = observed - typewell_at_known
    finite_pair = np.isfinite(observed) & np.isfinite(typewell_at_known) & np.isfinite(residual)
    return pd.DataFrame(
        {
            "row_idx": known_index.astype(np.int64),
            "tvt_input": tvt_input[known_index],
            "horizontal_gr": observed,
            "typewell_gr_at_tvt_input": typewell_at_known,
            "residual": np.where(finite_pair, residual, np.nan),
            "finite_pair": finite_pair,
        }
    )


def _clip_scale(raw_scale: float, clip: tuple[float, float]) -> tuple[float, bool, bool]:
    if not math.isfinite(raw_scale):
        raise ValueError("scale must be finite before clipping")
    return (
        float(np.clip(raw_scale, clip[0], clip[1])),
        bool(raw_scale < clip[0]),
        bool(raw_scale > clip[1]),
    )


def compute_available_prefix_scales(
    residual_frame: pd.DataFrame,
    available_rows: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    scale = get_nested(config, "model.scale_estimator") or {}
    fit_fraction = float(scale["internal_fit_fraction"])
    minimum_total = int(scale["minimum_total_finite_pairs"])
    minimum_early = int(scale["minimum_early_finite_pairs"])
    minimum_late = int(scale["minimum_late_finite_pairs"])
    clip = tuple(float(value) for value in scale["clip"])
    available = residual_frame.loc[residual_frame["row_idx"] < int(available_rows)].copy()
    if len(available) != int(available_rows):
        raise ValueError("available prefix boundary is not row-contiguous")
    finite_residual = available.loc[available["finite_pair"], "residual"].to_numpy(np.float64)
    pair_count = int(len(finite_residual))
    split_index = int(math.floor(pair_count * fit_fraction))
    early = finite_residual[:split_index]
    late = finite_residual[split_index:]
    observed = available["horizontal_gr"].to_numpy(np.float64)
    typewell_at = available["typewell_gr_at_tvt_input"].to_numpy(np.float64)
    zero_fill_residual = np.where(np.isfinite(observed), observed, 0.0) - typewell_at
    zero_fill_raw = float(np.std(zero_fill_residual, ddof=0))
    zero_fill_sigma, zero_clip_low, zero_clip_high = _clip_scale(zero_fill_raw, clip)
    finite_raw = float(np.std(finite_residual, ddof=0)) if pair_count else float("nan")
    if math.isfinite(finite_raw):
        finite_sigma, finite_clip_low, finite_clip_high = _clip_scale(finite_raw, clip)
    else:
        finite_sigma, finite_clip_low, finite_clip_high = (
            zero_fill_sigma,
            zero_clip_low,
            zero_clip_high,
        )
    sigma_early_raw = float(np.std(early, ddof=0)) if len(early) else float("nan")
    late_mse = float(np.mean(np.square(late))) if len(late) else float("nan")
    valid_structure = (
        pair_count >= minimum_total
        and len(early) >= minimum_early
        and len(late) >= minimum_late
        and math.isfinite(sigma_early_raw)
        and math.isfinite(late_mse)
    )
    if valid_structure:
        tau_variance = max(0.0, late_mse - sigma_early_raw**2)
        tau_structure = math.sqrt(tau_variance)
        structure_raw = math.sqrt(sigma_early_raw**2 + tau_variance)
        fallback_reason = "none"
    else:
        tau_variance = float("nan")
        tau_structure = float("nan")
        structure_raw = zero_fill_raw
        reasons = []
        if pair_count < minimum_total:
            reasons.append("minimum_total_finite_pairs")
        if len(early) < minimum_early:
            reasons.append("minimum_early_finite_pairs")
        if len(late) < minimum_late:
            reasons.append("minimum_late_finite_pairs")
        if not math.isfinite(sigma_early_raw) or not math.isfinite(late_mse):
            reasons.append("nonfinite_structure_moment")
        fallback_reason = "+".join(reasons) or "nonfinite_structure_moment"
    structure_sigma, structure_clip_low, structure_clip_high = _clip_scale(structure_raw, clip)
    return {
        "available_prefix_rows": int(available_rows),
        "finite_pair_count": pair_count,
        "missing_gr_count": int((~np.isfinite(observed)).sum()),
        "early_finite_pair_count": int(len(early)),
        "late_finite_pair_count": int(len(late)),
        "finite_only_sigma_raw": finite_raw,
        "finite_only_sigma": finite_sigma,
        "finite_only_clip_low": finite_clip_low,
        "finite_only_clip_high": finite_clip_high,
        "exp209_zero_fill_sigma_raw": zero_fill_raw,
        "exp209_zero_fill_sigma": zero_fill_sigma,
        "exp209_zero_fill_clip_low": zero_clip_low,
        "exp209_zero_fill_clip_high": zero_clip_high,
        "sigma_finite_early_raw": sigma_early_raw,
        "late_zero_center_mse": late_mse,
        "tau_structure_variance": tau_variance,
        "tau_structure": tau_structure,
        "structure_sigma_raw": structure_raw,
        "structure_sigma": structure_sigma,
        "structure_fallback": not valid_structure,
        "structure_fallback_reason": fallback_reason,
        "structure_clip_low": structure_clip_low,
        "structure_clip_high": structure_clip_high,
        "affine_a": 1.0,
        "affine_b": 0.0,
    }


def gaussian_nll_without_constant(residual: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive")
    return np.log(float(sigma)) + 0.5 * np.square(values / float(sigma))


# %% [markdown]
# ## 5. Rolling-origin Stage 0 audit


# %%
def compute_well_stage0(
    well: str,
    fold: int,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    residual_frame = build_prefix_residual_frame(horizontal, typewell)
    known_rows = int(len(residual_frame))
    forward_fraction = float(
        get_nested(config, "model.stage_0_prefix_backtest.forward_evaluation_fraction")
    )
    audit_rows: list[dict[str, Any]] = []
    for origin in ORIGIN_ORDER:
        origin_stop = int(math.floor(known_rows * origin))
        evaluation_stop = int(math.floor(known_rows * min(1.0, origin + forward_fraction)))
        scales = compute_available_prefix_scales(residual_frame, origin_stop, config)
        evaluation = residual_frame.loc[
            (residual_frame["row_idx"] >= origin_stop)
            & (residual_frame["row_idx"] < evaluation_stop)
            & residual_frame["finite_pair"]
        ]
        evaluation_residual = evaluation["residual"].to_numpy(np.float64)
        row: dict[str, Any] = {
            "well_id": str(well),
            "fold": int(fold),
            "origin": float(origin),
            "known_prefix_rows": known_rows,
            "origin_stop_row": origin_stop,
            "evaluation_stop_row": evaluation_stop,
            "evaluation_block_rows": evaluation_stop - origin_stop,
            "evaluation_finite_pair_count": int(len(evaluation_residual)),
            "evaluable": bool(len(evaluation_residual) > 0),
            **scales,
        }
        policy_sigma = {
            "finite_only": float(scales["finite_only_sigma"]),
            "exp209_zero_fill": float(scales["exp209_zero_fill_sigma"]),
            "structure_added": float(scales["structure_sigma"]),
        }
        for policy in SCALE_POLICY_ORDER:
            if len(evaluation_residual):
                nll = gaussian_nll_without_constant(evaluation_residual, policy_sigma[policy])
                row[f"{policy}_nll_sum"] = float(nll.sum())
                row[f"{policy}_nll_mean"] = float(nll.mean())
            else:
                row[f"{policy}_nll_sum"] = float("nan")
                row[f"{policy}_nll_mean"] = float("nan")
        audit_rows.append(row)
    final_scales = compute_available_prefix_scales(residual_frame, known_rows, config)
    final_row = {
        "well_id": str(well),
        "fold": int(fold),
        "known_prefix_rows": known_rows,
        **final_scales,
        "frozen_before_unknown_suffix_truth_attachment": True,
    }
    return audit_rows, final_row


def run_stage0_prefix_audit(
    wells: list[str],
    fold_map: dict[str, int],
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rolling_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for index, well in enumerate(sorted(wells), 1):
        horizontal = load_horizontal_without_truth(well, raw_dir)
        typewell = load_typewell(well, raw_dir)
        well_rolling, well_final = compute_well_stage0(
            well,
            fold_map[well],
            horizontal,
            typewell,
            config,
        )
        rolling_rows.extend(well_rolling)
        final_rows.append(well_final)
        if index == 1 or index % 25 == 0 or index == len(wells):
            print(f"[exp337 Stage 0] {index}/{len(wells)} wells", flush=True)
    rolling = pd.DataFrame(rolling_rows).sort_values(
        ["origin", "fold", "well_id"], kind="mergesort"
    ).reset_index(drop=True)
    final = pd.DataFrame(final_rows).sort_values(["fold", "well_id"], kind="mergesort").reset_index(
        drop=True
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(rolling) != expected_wells * len(ORIGIN_ORDER) or len(final) != expected_wells:
        raise ValueError("exp337 Stage 0 output coverage mismatch")
    if (
        rolling["well_id"].nunique() != expected_wells
        or final["well_id"].nunique() != expected_wells
    ):
        raise ValueError("exp337 Stage 0 well coverage mismatch")
    return rolling, final


# %% [markdown]
# ## 6. Stage 0 promotion gate


# %%
def _pooled_nll(frame: pd.DataFrame, policy: str) -> float:
    count = int(frame["evaluation_finite_pair_count"].sum())
    if count <= 0:
        return float("nan")
    return float(frame[f"{policy}_nll_sum"].sum() / count)


def evaluate_stage0_gate(
    rolling: pd.DataFrame,
    final: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate_config = get_nested(config, "model.stage_0_prefix_backtest.pass_requires_all") or {}
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    origin_reports: dict[str, Any] = {}
    origin_passes: list[bool] = []
    for origin in ORIGIN_ORDER:
        subset = rolling.loc[np.isclose(rolling["origin"], origin)].copy()
        evaluable = subset.loc[subset["evaluable"]].copy()
        pooled = {policy: _pooled_nll(evaluable, policy) for policy in SCALE_POLICY_ORDER}
        fold_rows = []
        for fold in expected_folds:
            fold_frame = evaluable.loc[evaluable["fold"] == fold]
            fold_nll = {policy: _pooled_nll(fold_frame, policy) for policy in SCALE_POLICY_ORDER}
            fold_rows.append(
                {
                    "fold": fold,
                    **fold_nll,
                    "structure_beats_finite": bool(
                        fold_nll["structure_added"] < fold_nll["finite_only"]
                    ),
                    "structure_beats_zero_fill": bool(
                        fold_nll["structure_added"] < fold_nll["exp209_zero_fill"]
                    ),
                }
            )
        folds_structure_beats_finite = sum(row["structure_beats_finite"] for row in fold_rows)
        folds_structure_beats_zero = sum(row["structure_beats_zero_fill"] for row in fold_rows)
        evaluable_fraction = float(len(evaluable) / expected_wells)
        fallback_fraction = float(subset["structure_fallback"].mean())
        gain_vs_zero = float(pooled["exp209_zero_fill"] - pooled["structure_added"])
        checks = {
            "evaluable_well_fraction": evaluable_fraction
            >= float(gate_config["minimum_evaluable_well_fraction_each_origin"]),
            "fallback_well_fraction": fallback_fraction
            <= float(gate_config["maximum_fallback_well_fraction_each_origin"]),
            "structure_beats_finite_pooled": pooled["structure_added"] < pooled["finite_only"],
            "structure_beats_finite_folds": folds_structure_beats_finite
            >= int(gate_config["minimum_folds_structure_beats_finite_each_origin"]),
            "structure_gain_vs_zero_fill": gain_vs_zero
            >= float(gate_config["minimum_nll_gain_vs_zero_fill_per_finite_residual_each_origin"]),
            "structure_beats_zero_fill_folds": folds_structure_beats_zero
            >= int(gate_config["minimum_folds_structure_beats_zero_fill_each_origin"]),
        }
        origin_pass = all(checks.values())
        origin_passes.append(origin_pass)
        origin_reports[f"{origin:.2f}"] = {
            "origin": origin,
            "evaluable_wells": int(len(evaluable)),
            "evaluable_well_fraction": evaluable_fraction,
            "fallback_wells": int(subset["structure_fallback"].sum()),
            "fallback_well_fraction": fallback_fraction,
            "evaluation_finite_pairs": int(evaluable["evaluation_finite_pair_count"].sum()),
            "pooled_nll_per_finite_residual": pooled,
            "structure_nll_gain_vs_zero_fill_per_finite_residual": gain_vs_zero,
            "folds_structure_beats_finite": int(folds_structure_beats_finite),
            "folds_structure_beats_zero_fill": int(folds_structure_beats_zero),
            "fold_metrics": fold_rows,
            "checks": checks,
            "passed": origin_pass,
        }
    valid_tau = pd.to_numeric(
        final.loc[~final["structure_fallback"], "tau_structure"], errors="coerce"
    ).dropna()
    median_tau = float(valid_tau.median()) if len(valid_tau) else float("nan")
    lower_clip_fraction = float(final["structure_clip_low"].mean())
    final_checks = {
        "full_prefix_well_coverage": len(final) == expected_wells
        and final["well_id"].nunique() == expected_wells,
        "median_tau_structure": math.isfinite(median_tau)
        and median_tau >= float(gate_config["minimum_final_median_tau_structure_gr_units"]),
        "lower_clip_fraction": lower_clip_fraction
        <= float(gate_config["maximum_final_lower_clip_well_fraction"]),
    }
    passed = all(origin_passes) and all(final_checks.values())
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_known_prefix_rolling_origin",
        "unknown_suffix_truth_read": False,
        "hmm_well_runs": 0,
        "models": 0,
        "boosters": 0,
        "origin_reports": origin_reports,
        "final_full_prefix": {
            "wells": int(len(final)),
            "fallback_wells": int(final["structure_fallback"].sum()),
            "median_tau_structure": median_tau,
            "lower_clip_wells": int(final["structure_clip_low"].sum()),
            "lower_clip_fraction": lower_clip_fraction,
            "checks": final_checks,
        },
        "passed": passed,
        "decision": (
            "stage0_pass_waiting_separate_stage1_implementation_and_run_approval"
            if passed
            else "stage0_fail_close_without_hmm_rescue_inference_or_submission"
        ),
        "stage1_enabled": False,
    }


# %% [markdown]
# ## 7. Metrics and generated artifacts


# %%
def run_stage0_experiment(config: dict[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started = datetime.now(UTC)
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    scientific_contract = build_scientific_contract(config)
    dependency_contract, fold_map = preflight_dependencies(config, raw_dir)
    scientific_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    dependency_path = artifacts / f"{OUTPUT_PREFIX}_input_dependency_contract.json"
    write_json(scientific_path, scientific_contract)
    write_json(dependency_path, dependency_contract)
    rolling, final = run_stage0_prefix_audit(sorted(fold_map), fold_map, raw_dir, config)
    rolling_path = artifacts / f"{OUTPUT_PREFIX}_rolling_origin_scale_nll_audit.csv.gz"
    final_path = artifacts / f"{OUTPUT_PREFIX}_full_prefix_scale_audit.csv.gz"
    rolling.to_csv(rolling_path, index=False, compression="gzip")
    final.to_csv(final_path, index=False, compression="gzip")
    rolling_report = inspect_gzip_csv(rolling_path)
    final_report = inspect_gzip_csv(final_path)
    rolling_report["frozen_before_unknown_suffix_truth_attachment"] = True
    final_report["frozen_before_unknown_suffix_truth_attachment"] = True
    gate = evaluate_stage0_gate(rolling, final, config)
    gate["rolling_origin_content_sha256"] = rolling_report["content_sha256"]
    gate["full_prefix_scale_content_sha256"] = final_report["content_sha256"]
    gate_path = artifacts / f"{OUTPUT_PREFIX}_stage0_gate_summary.json"
    write_json(gate_path, gate)
    finished = datetime.now(UTC)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "stage0_passed_waiting_separate_stage1_approval"
            if gate["passed"]
            else "stage0_gate_failed_branch_closed"
        ),
        "generated_at_utc": finished.isoformat(),
        "runtime_seconds": (finished - started).total_seconds(),
        "runtime_versions": runtime_versions(),
        "rows": int(len(rolling)),
        "wells": int(final["well_id"].nunique()),
        "origins": list(ORIGIN_ORDER),
        "stage0_diagnostics": 1,
        "hmm_well_runs": 0,
        "models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "boosters": 0,
        "control_reruns": 0,
        "unknown_suffix_truth_read": False,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "dependency_contract_sha256": dependency_contract["dependency_contract_sha256"],
        "rolling_origin_audit_sha256": rolling_report,
        "full_prefix_scale_audit_sha256": final_report,
        "stage0_gate": gate,
        "stage1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
        "model_sha256": None,
        "prediction_sha256": None,
        "submission_sha256": None,
        "generated_files": {
            "scientific_contract": {
                "path": str(scientific_path),
                "raw_sha256": sha256_path(scientific_path),
            },
            "input_dependency_contract": {
                "path": str(dependency_path),
                "raw_sha256": sha256_path(dependency_path),
            },
            "rolling_origin_audit": rolling_report,
            "full_prefix_scale_audit": final_report,
            "stage0_gate": {
                "path": str(gate_path),
                "raw_sha256": sha256_path(gate_path),
            },
        },
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
        },
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 8. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "active_stage": get_nested(CONFIG, "execution.active_stage"),
                "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
                "rolling_origins": get_nested(
                    CONFIG, "model.stage_0_prefix_backtest.rolling_origins"
                ),
                "scale_policies": get_nested(
                    CONFIG, "model.stage_0_prefix_backtest.scale_policies"
                ),
                "hmm_well_runs": get_nested(
                    CONFIG, "execution_contract.stage_0.hmm_well_runs"
                ),
                "boosters": get_nested(CONFIG, "execution_contract.stage_0.boosters"),
                "stage1_enabled": get_nested(
                    CONFIG, "model.stage_1_exact_hmm.enabled_after_implementation"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 9. Run the approved Stage 0 audit only
#
# The execution guard requires separate Kaggle package/push/run approval in
# `config.yaml`. Even a passing Stage 0 writes no HMM prediction and does not
# enable Stage 1.


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_stage0_experiment(CONFIG)
