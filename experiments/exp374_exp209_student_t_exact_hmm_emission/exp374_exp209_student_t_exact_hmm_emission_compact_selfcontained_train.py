# %% [markdown]
# # exp374 exp209 Student-t exact-HMM emission train
#
# Train-side exact-HMM audit of one preregistered fixed-df Student-t emission.
# The exp209 absolute-TVT grid, rate states, transitions, priors, GR scale,
# missing-GR handling, Type Well interpolation, and posterior-mean output are
# unchanged. Saved exp209 HMM and exp072 LikPF predictions are read-only
# controls; no control, PF, Beam, model, inference, or submission is regenerated.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Exp209 observation model and Student-t emission preparation
# 5. Exact exp209 forward-backward kernel
# 6. Target-free decoding and prediction freeze
# 7. Late truth/control attachment and paired metrics
# 8. Promotion gate and generated artifacts
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


EXPERIMENT_NAME = "exp374_exp209_student_t_exact_hmm_emission"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT_ORDER = ("student_t_df4_on_exp209_absolute_tvt",)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP374_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp374 config not found in {[str(path) for path in candidates]}")


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


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    return mapping_sha256(schema)


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
        "lineage.parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "implementation.enabled": True,
        "implementation.scope": "train_side_candidate_audit_only",
        "model.active_variants": list(VARIANT_ORDER),
        "model.promotion_candidate": VARIANT_ORDER[0],
        "model.emission.control_family": "gaussian",
        "model.emission.candidate_family": "student_t",
        "model.emission.degrees_of_freedom": 4.0,
        "model.emission.additional_clip": "none",
        "model.emission.temperature": 1.0,
        "model.emission.likelihood_weight": 1.0,
        "execution_contract.scientific_variants": 1,
        "execution_contract.hmm_well_runs": 773,
        "execution_contract.model_configs": 0,
        "execution_contract.lightgbm_configs": 0,
        "execution_contract.trained_folds": 0,
        "execution_contract.pf_well_runs": 0,
        "execution_contract.beam_well_runs": 0,
        "execution_contract.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "runtime.num_workers": 1,
        "runtime.numba_num_threads": 2,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
        "execution.create_submission": False,
        "promotion_gates.require_raw_missing_row_non_regression": True,
        "promotion_gates.require_high_missing_fraction_well_non_regression": True,
        "promotion_gates.require_1000_plus_non_regression": True,
        "promotion_gates.require_hidden_like_spatial_non_regression": True,
        "promotion_gates.require_hidden_like_typewell_purged_non_regression": True,
        "promotion_gates.require_fixed_likpf_50_50_non_regression": True,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp374 fixed contract mismatch: {key} must be {value!r}")
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
        "degrees_of_freedom": 4.0,
    }
    for key, value in fixed_hmm.items():
        if float(hmm.get(key, -1.0)) != value:
            raise ValueError(f"exp374 fixes model.fixed_hmm.{key}={value}")
    if (
        hmm.get("emission") != "student_t_negative_half_df_plus_1_log1p_z2_over_df"
        or hmm.get("rate_center") != "zero"
    ):
        raise ValueError("exp374 fixes df=4 Student-t emission and a zero-centered rate grid")
    fixed_policies = {
        "coordinate": "absolute_tvt",
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "missing_gr_policy": "linear_interpolate_both_directions_then_typewell_mean",
        "evaluation_gr_policy": "interpolate_both_directions_then_typewell_mean",
        "typewell_gr_policy": "sort_tvt_ffill_bfill_then_linear_interp",
        "output": "posterior_mean",
    }
    for key, value in fixed_policies.items():
        if hmm.get(key) != value:
            raise ValueError(f"exp374 fixes model.fixed_hmm.{key}={value}")
    if [float(value) for value in hmm.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("exp374 fixes the exp209 GR sigma clip to [10, 60]")
    forbidden = set(str(value) for value in get_nested(config, "guards.forbidden") or [])
    required_forbidden = {
        "exp342_shift_rank_stage_0",
        "exp226_tvt_geop_or_residual_offset_coordinate",
        "gaussian_parent_hmm_rerun",
        "df_or_scale_search",
        "likelihood_temperature_or_clip_search",
        "sigma_or_missing_weight_change",
        "transition_rate_grid_or_prior_change",
        "blend_weight_search",
        "same_oof_rescue",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp374 no-rescue and single-change exclusions are incomplete")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp374 implementation approval must be recorded")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.train_run_approved"))
        and bool(get_nested(config, "execution.run_hmm"))
    ):
        raise RuntimeError("exp374 Kaggle package/push/run is not approved")


def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "truth_attached": False,
        "variants": list(VARIANT_ORDER),
        "promotion_candidate": VARIANT_ORDER[0],
        "emission": get_nested(config, "model.emission"),
        "fixed_hmm": get_nested(config, "model.fixed_hmm"),
        "promotion_gates": get_nested(config, "promotion_gates"),
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
            "exp342_shift_rank_or_residual_offset_coordinate",
            "df_scale_temperature_clip_mixture_or_huber_search",
            "sigma_missing_weight_or_gr_preprocessing_change",
            "hmm_grid_transition_prior_or_output_change",
            "blend_weight_search",
            "likpf_pf_or_beam_rerun",
            "same_oof_rescue",
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


def preflight_controls_and_assignments(config: dict[str, Any]) -> dict[str, Any]:
    control = get_nested(config, "data.saved_controls") or {}
    paths = {
        "saved_hmm": resolve_existing(
            str(control["hmm_cache_filename"]), _candidate_paths(control)
        ),
        "saved_exp072": resolve_existing(
            str(control["exp072_cache_filename"]), _candidate_paths(control)
        ),
    }
    hmm_report = inspect_gzip_csv(paths["saved_hmm"])
    exp072_report = inspect_gzip_csv(paths["saved_exp072"])
    hmm_columns = pd.read_csv(paths["saved_hmm"], nrows=0).columns.astype(str).tolist()
    exp072_columns = pd.read_csv(paths["saved_exp072"], nrows=0).columns.astype(str).tolist()
    required_hmm_columns = {
        "id",
        "well",
        str(control["raw_hmm_prediction_column"]),
    }
    required_exp072_columns = {
        "id",
        "well",
        "md_since",
        str(control["likpf_anchor_column"]),
        str(control["likpf_delta_column"]),
    }
    missing_hmm = sorted(required_hmm_columns - set(hmm_columns))
    missing_exp072 = sorted(required_exp072_columns - set(exp072_columns))
    if missing_hmm:
        raise ValueError(f"saved exp209 HMM missing required columns: {missing_hmm}")
    if missing_exp072:
        raise ValueError(f"saved exp209 exp072-cache missing required columns: {missing_exp072}")
    hmm_report["columns"] = hmm_columns
    exp072_report["columns"] = exp072_columns
    if hmm_report["decompressed_sha256"] != str(
        control["expected_hmm_prediction_decompressed_sha256"]
    ):
        raise ValueError("saved exp209 HMM decompressed SHA mismatch")
    if exp072_report["decompressed_sha256"] != str(
        control["expected_exp072_cache_decompressed_sha256"]
    ):
        raise ValueError("saved exp209 exp072-cache decompressed SHA mismatch")
    fold = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(str(fold["filename"]), _candidate_paths(fold))
    fold_report = inspect_gzip_csv(fold_path)
    if fold_report["decompressed_sha256"] != str(fold["expected_decompressed_sha256"]):
        raise ValueError("fold/truth assignment decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold["safe_columns"]]
    forbidden_decoder_columns = {
        str(value) for value in fold.get("forbidden_decoder_columns", [])
    }
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp374 reporting decoder allowlist must contain identity/fold only")
    if set(safe_columns) & forbidden_decoder_columns:
        raise ValueError("exp374 reporting decoder allowlist contains forbidden columns")
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
    expected_role_counts = hidden.get("expected_role_counts") or {}
    actual_role_counts: dict[str, dict[str, int]] = {}
    for scope, column in hidden["role_columns"].items():
        counts = {
            str(key): int(value)
            for key, value in hidden_frame[str(column)]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .items()
        }
        expected_counts = {
            str(key): int(value)
            for key, value in (expected_role_counts.get(str(scope)) or {}).items()
        }
        if counts != expected_counts:
            raise ValueError(f"hidden-like role counts mismatch for {scope}")
        actual_role_counts[str(scope)] = counts
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if (
        int(hmm_report["data_rows"]) != expected_rows
        or int(exp072_report["data_rows"]) != expected_rows
    ):
        raise ValueError("saved control row coverage mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()}
        | {
            "fold_assignment": str(fold_path),
            "hidden_like_assignment": str(hidden_path),
        },
        "saved_hmm": hmm_report,
        "saved_exp072": exp072_report,
        "fold_assignment": {
            **fold_report,
            "well_ids": safe_wells,
        },
        "hidden_like_assignment": {
            "path": str(hidden_path),
            "raw_sha256": hidden_sha,
            "wells": int(hidden_frame["well_id"].nunique()),
            "role_counts": actual_role_counts,
        },
    }


# %% [markdown]
# ## 4. Exp209 observation model and Student-t emission preparation


# %%
def compute_exp209_zero_fill_sigma_audit(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    clip: tuple[float, float] = (10.0, 60.0),
) -> dict[str, Any]:
    """Reproduce exp209 known-prefix zero-fill population standard deviation."""
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    known_mask = np.isfinite(tvt_input)
    known_index = np.flatnonzero(known_mask)
    if len(known_index) == 0 or not np.array_equal(known_index, np.arange(len(known_index))):
        raise ValueError("exp374 requires one contiguous known TVT_input prefix")
    typewell_at_known = np.interp(
        tvt_input[known_mask],
        np.asarray(typewell_tvt, dtype=np.float64),
        np.asarray(typewell_gr, dtype=np.float64),
    )
    known_gr = horizontal_gr[known_mask]
    zero_fill_residual = np.where(np.isfinite(known_gr), known_gr, 0.0) - typewell_at_known
    zero_fill_raw = float(np.nanstd(zero_fill_residual, ddof=0))
    if not math.isfinite(zero_fill_raw):
        raise ValueError("exp209 zero-fill prefix sigma is non-finite")
    exp209_sigma = float(np.clip(zero_fill_raw, clip[0], clip[1]))
    known_count = int(known_mask.sum())
    finite_count = int(np.isfinite(known_gr).sum())
    missing_known_gr_count = known_count - finite_count
    return {
        "known_prefix_rows": known_count,
        "finite_known_gr_rows": finite_count,
        "missing_known_gr_count": missing_known_gr_count,
        "missing_known_gr_fraction": (
            float(missing_known_gr_count / known_count) if known_count else 0.0
        ),
        "exp209_zero_fill_sigma_raw": zero_fill_raw,
        "exp209_zero_fill_sigma": exp209_sigma,
        "exp209_zero_fill_clip_low": bool(zero_fill_raw < clip[0]),
        "exp209_zero_fill_clip_high": bool(zero_fill_raw > clip[1]),
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "missing_gr_fill_value": 0.0,
        "affine_a": 1.0,
        "affine_b": 0.0,
        "observation_weight": 1.0,
    }


def student_t_log_likelihood(
    zscore: np.ndarray,
    degrees_of_freedom: float = 4.0,
) -> np.ndarray:
    """State-comparison log likelihood with the state-independent constant omitted."""
    df = float(degrees_of_freedom)
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("Student-t degrees_of_freedom must be positive and finite")
    values = np.asarray(zscore, dtype=np.float64)
    return -0.5 * (df + 1.0) * np.log1p(np.square(values) / df)


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
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    if np.isinf(tvt_input).any():
        raise ValueError("TVT_input may contain finite known values or NaN only")
    known_mask = np.isfinite(tvt_input)
    eval_mask = ~known_mask
    known_index = np.flatnonzero(known_mask)
    if (
        len(known_index) == 0
        or not eval_mask.any()
        or not np.array_equal(known_index, np.arange(len(known_index)))
    ):
        raise ValueError("each exp374 well must contain one contiguous known prefix and suffix")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    valid_typewell = np.isfinite(typewell_tvt) & np.isfinite(typewell_gr)
    typewell_tvt = typewell_tvt[valid_typewell]
    typewell_gr = typewell_gr[valid_typewell]
    if len(typewell_tvt) < 2 or np.any(np.diff(typewell_tvt) < 0):
        raise ValueError("typewell TVT/GR contract is invalid")
    observation_audit = compute_exp209_zero_fill_sigma_audit(
        horizontal,
        typewell_tvt,
        typewell_gr,
        clip=tuple(float(value) for value in hmm["sigma_clip"]),
    )
    eval_index = np.flatnonzero(eval_mask)
    last_index = int(known_index[-1])
    last_tvt = float(tvt_input[last_index])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    step = float(hmm["step"])
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    state_gr = np.interp(grid, typewell_tvt, typewell_gr)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce")
    raw_gr = horizontal_gr.to_numpy(np.float64)
    raw_gr_observed = np.isfinite(raw_gr[eval_index])
    observed_gr = (
        horizontal_gr.interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[eval_index]
    )
    sigma_gr = float(observation_audit["exp209_zero_fill_sigma"])
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
        "observed_gr": observed_gr,
        "raw_gr_observed": raw_gr_observed,
        "sigma_gr": sigma_gr,
        "dm": dm,
        "dz": dz,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "init_rate": init_rate,
        "observation_audit": observation_audit,
        "evaluation_observed_rows": int(raw_gr_observed.sum()),
        "evaluation_missing_rows": int((~raw_gr_observed).sum()),
        "evaluation_missing_fraction": float((~raw_gr_observed).mean()),
    }


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


def run_exact_hmm_variant(
    prepared: dict[str, Any],
    sigma_gr: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    hmm = get_nested(config, "model.fixed_hmm") or {}
    if not math.isfinite(float(sigma_gr)) or not 10.0 <= float(sigma_gr) <= 60.0:
        raise ValueError("exp374 exp209 sigma_gr must satisfy the frozen [10, 60] contract")
    df = float(hmm["degrees_of_freedom"])
    zscore = (
        prepared["observed_gr"][:, None] - prepared["state_gr"][None, :]
    ) / float(sigma_gr)
    emission = student_t_log_likelihood(zscore, df).astype(np.float32)
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
        "emission_finite_coverage": float(np.isfinite(emission).mean()),
        "degrees_of_freedom": df,
    }


# %% [markdown]
# ## 6. Target-free decoding and prediction freeze


# %%
def decode_well(
    well: str,
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
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
    observation_audit = prepared["observation_audit"]
    schedule = pd.DataFrame(
        {
            "id": predictions["id"],
            "well_id": well,
            "row_idx": eval_index,
            "raw_gr_observed": prepared["raw_gr_observed"].astype(np.int8),
            "exp209_zero_fill_sigma": float(observation_audit["exp209_zero_fill_sigma"]),
            "student_t_degrees_of_freedom": float(
                get_nested(config, "model.emission.degrees_of_freedom")
            ),
            "evaluation_missing_fraction": float(prepared["evaluation_missing_fraction"]),
        }
    )
    runtime_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(VARIANT_ORDER):
        started = time.time()
        result = run_exact_hmm_variant(
            prepared,
            float(prepared["sigma_gr"]),
            config,
        )
        predictions[f"{variant}_hmm_tvt"] = result["mean"]
        predictions[f"{variant}_hmm_std"] = result["std"]
        predictions[f"{variant}_hmm_loglik"] = float(result["loglik"])
        runtime_rows.append(
            {
                "well_id": well,
                "variant_index": variant_index,
                "variant": variant,
                "rows": len(eval_index),
                "exp209_zero_fill_sigma": float(observation_audit["exp209_zero_fill_sigma"]),
                "student_t_degrees_of_freedom": float(result["degrees_of_freedom"]),
                "raw_observed_rows": int(prepared["evaluation_observed_rows"]),
                "raw_missing_rows": int(prepared["evaluation_missing_rows"]),
                "loglik": float(result["loglik"]),
                "posterior_row_sum_max_abs_error": float(result["posterior_row_sum_max_abs_error"]),
                "emission_finite_coverage": float(result["emission_finite_coverage"]),
                "elapsed_seconds": time.time() - started,
            }
        )
    numeric = predictions.drop(columns=["id", "well_id"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"well={well} produced non-finite prediction diagnostics")
    observation_row = {
        "well_id": well,
        **observation_audit,
        "evaluation_observed_rows": int(prepared["evaluation_observed_rows"]),
        "evaluation_missing_rows": int(prepared["evaluation_missing_rows"]),
        "evaluation_missing_fraction": float(prepared["evaluation_missing_fraction"]),
    }
    return predictions, schedule, observation_row, pd.DataFrame(runtime_rows)


def generate_and_freeze_predictions(
    raw_dir: Path,
    artifacts: Path,
    config: dict[str, Any],
    wells: list[str],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exp374 exact-HMM audit")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    outer_workers = int(get_nested(config, "runtime.num_workers"))

    def build_one(index: int, well: str):
        print(f"[{index}/{len(wells)}] exp374 well={well}", flush=True)
        return decode_well(well, raw_dir, config)

    started = time.time()
    if outer_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(build_one)(index, well) for index, well in enumerate(wells, start=1)
        )
    else:
        results = [build_one(index, well) for index, well in enumerate(wells, start=1)]
    prediction = pd.concat([row[0] for row in results], ignore_index=True)
    schedule = pd.concat([row[1] for row in results], ignore_index=True)
    observation_audit = pd.DataFrame([row[2] for row in results])
    runtime = pd.concat([row[3] for row in results], ignore_index=True)
    prediction = prediction.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    schedule = schedule.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    observation_audit = observation_audit.sort_values(
        "well_id", kind="mergesort"
    ).reset_index(drop=True)
    runtime = runtime.sort_values(["well_id", "variant_index"], kind="mergesort").reset_index(
        drop=True
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(prediction) != expected_rows
        or len(schedule) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or schedule["well_id"].nunique() != expected_wells
        or prediction["id"].duplicated().any()
        or schedule["id"].duplicated().any()
        or len(observation_audit) != expected_wells
        or len(runtime) != int(get_nested(config, "execution_contract.hmm_well_runs"))
    ):
        raise ValueError("exp374 prediction/observation/runtime coverage mismatch")
    _assert_same_order("prediction vs raw-GR audit", prediction["id"], schedule["id"])
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_predictions.csv.gz"
    schedule_path = artifacts / f"{OUTPUT_PREFIX}_raw_gr_emission_contract.csv.gz"
    observation_path = artifacts / f"{OUTPUT_PREFIX}_observation_audit.csv.gz"
    runtime_path = artifacts / f"{OUTPUT_PREFIX}_by_well_variant_runtime.csv"
    prediction.to_csv(prediction_path, index=False, compression="gzip")
    schedule.to_csv(schedule_path, index=False, compression="gzip")
    observation_audit.to_csv(observation_path, index=False, compression="gzip")
    runtime.to_csv(runtime_path, index=False)
    prediction_report = inspect_gzip_csv(prediction_path)
    schedule_report = inspect_gzip_csv(schedule_path)
    observation_report = inspect_gzip_csv(observation_path)
    prediction_report["schema_sha256"] = dataframe_schema_sha(prediction)
    schedule_report["schema_sha256"] = dataframe_schema_sha(schedule)
    observation_report["schema_sha256"] = dataframe_schema_sha(observation_audit)
    prediction_report["frozen_before_truth_attachment"] = True
    schedule_report["frozen_before_truth_attachment"] = True
    observation_report["frozen_before_truth_attachment"] = True
    reports = {
        "prediction": prediction_report,
        "raw_gr_emission_contract": schedule_report,
        "observation_audit": observation_report,
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
        {
            "prediction": prediction_path,
            "raw_gr_emission_contract": schedule_path,
            "observation_audit": observation_path,
            "runtime": runtime_path,
        },
        runtime,
    )


# %% [markdown]
# ## 7. Late truth/control attachment and paired metrics


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


def materialize_saved_likpf_tvt(
    frame: pd.DataFrame,
    control: dict[str, Any],
) -> np.ndarray:
    anchor_column = str(control["likpf_anchor_column"])
    delta_column = str(control["likpf_delta_column"])
    missing = sorted({anchor_column, delta_column} - set(frame.columns))
    if missing:
        raise ValueError(f"saved exp072 LikPF materialization columns missing: {missing}")
    anchor = pd.to_numeric(frame[anchor_column], errors="raise").to_numpy(np.float64)
    delta = pd.to_numeric(frame[delta_column], errors="raise").to_numpy(np.float64)
    values = anchor + delta
    if not np.isfinite(values).all():
        raise ValueError("saved exp072 LikPF materialization produced non-finite values")
    return values


def load_late_readout_frame(
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    output_paths: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_frozen_prediction(frozen["prediction"])
    _require_frozen_prediction(frozen["raw_gr_emission_contract"])
    prediction = (
        pd.read_csv(
            output_paths["prediction"],
            dtype={"id": str, "well_id": str},
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    schedule = (
        pd.read_csv(
            output_paths["raw_gr_emission_contract"],
            dtype={"id": str, "well_id": str},
        )
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    _assert_same_order("prediction vs raw-GR audit", prediction["id"], schedule["id"])
    control = get_nested(config, "data.saved_controls") or {}
    likpf_columns = [
        str(control["likpf_anchor_column"]),
        str(control["likpf_delta_column"]),
    ]
    exp072 = pd.read_csv(
        preflight["controls"]["paths"]["saved_exp072"],
        usecols=["id", "well", "md_since", *likpf_columns],
        dtype={"id": str, "well": str},
    )
    saved_hmm = pd.read_csv(
        preflight["controls"]["paths"]["saved_hmm"],
        usecols=["id", "well", str(control["raw_hmm_prediction_column"])],
        dtype={"id": str, "well": str},
    )
    for control_frame in (exp072, saved_hmm):
        control_frame["row_idx"] = pd.to_numeric(
            control_frame["id"].astype(str).str.rsplit("_", n=1).str[-1],
            errors="raise",
        ).astype(np.int64)
        control_frame.sort_values(["well", "row_idx"], kind="mergesort", inplace=True)
        control_frame.reset_index(drop=True, inplace=True)
    _assert_same_order("prediction vs exp072", prediction["id"], exp072["id"])
    _assert_same_order("prediction vs saved HMM", prediction["id"], saved_hmm["id"])
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
            "raw_hmm_tvt": pd.to_numeric(
                saved_hmm[str(control["raw_hmm_prediction_column"])], errors="raise"
            ).to_numpy(np.float64),
            "likpf_mean": materialize_saved_likpf_tvt(exp072, control),
            "raw_gr_observed": pd.to_numeric(schedule["raw_gr_observed"], errors="raise")
            .to_numpy(np.int8)
            .astype(bool),
            "evaluation_missing_fraction": pd.to_numeric(
                schedule["evaluation_missing_fraction"], errors="raise"
            ).to_numpy(np.float64),
        }
    )
    for variant in VARIANT_ORDER:
        frame[f"{variant}_hmm_tvt"] = pd.to_numeric(
            prediction[f"{variant}_hmm_tvt"], errors="raise"
        ).to_numpy(np.float64)
        frame[f"{variant}_likpf_50_50"] = (
            0.5 * frame[f"{variant}_hmm_tvt"] + 0.5 * frame["likpf_mean"]
        )
    frame["raw_hmm_likpf_50_50"] = 0.5 * frame["raw_hmm_tvt"] + 0.5 * frame["likpf_mean"]
    for scope, role_column in hidden["role_columns"].items():
        frame[str(scope)] = (
            frame["well_id"].map(hidden_frame[role_column].astype(str)).eq("valid").to_numpy()
        )
    numeric = [
        "true_tvt",
        "md_since",
        "evaluation_missing_fraction",
        "raw_hmm_tvt",
        "likpf_mean",
        "raw_hmm_likpf_50_50",
        *[f"{variant}_hmm_tvt" for variant in VARIANT_ORDER],
        *[f"{variant}_likpf_50_50" for variant in VARIANT_ORDER],
    ]
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise ValueError("late readout contains non-finite values")
    return frame, {
        "truth_attachment_stage": (
            "after_raw_gr_emission_contract_and_prediction_gzip_content_sha_freeze"
        ),
        "prediction_content_sha256": frozen["prediction"]["content_sha256"],
        "raw_gr_emission_contract_content_sha256": frozen["raw_gr_emission_contract"][
            "content_sha256"
        ],
        "observation_audit_content_sha256": frozen["observation_audit"]["content_sha256"],
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
    raw_observed = frame["raw_gr_observed"].to_numpy(bool)
    md_since = frame["md_since"].to_numpy(np.float64)
    missing_fraction = frame["evaluation_missing_fraction"].to_numpy(np.float64)
    buckets = get_nested(config, "validation.missing_fraction_buckets") or {}
    scopes.extend(
        [
            ("raw_gr_observed", raw_observed),
            ("raw_gr_missing", ~raw_observed),
            (
                "missing_fraction_low",
                (missing_fraction >= float(buckets["low"][0]))
                & (missing_fraction < float(buckets["low"][1])),
            ),
            (
                "missing_fraction_mid",
                (missing_fraction >= float(buckets["mid"][0]))
                & (missing_fraction < float(buckets["mid"][1])),
            ),
            (
                "missing_fraction_high",
                (missing_fraction >= float(buckets["high"][0]))
                & (missing_fraction <= float(buckets["high"][1])),
            ),
            ("md_since_0_250", (md_since >= 0.0) & (md_since < 250.0)),
            ("md_since_250_1000", (md_since >= 250.0) & (md_since < 1000.0)),
            ("md_since_1000_plus", md_since >= 1000.0),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        comparisons = {
            "direct": (f"{variant}_hmm_tvt", "raw_hmm_tvt"),
            "fixed_likpf_50_50": (
                f"{variant}_likpf_50_50",
                "raw_hmm_likpf_50_50",
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
    observation_audit: pd.DataFrame,
    preflight: dict[str, Any],
    runtime_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "promotion_gates") or {}
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
        "exact_hmm": {
            "actual": float(direct_overall["control_rmse"]),
            "expected": float(get_nested(config, "references.exp209_raw_hmm_rmse")),
        },
        "likpf": {
            "actual": likpf_actual,
            "expected": float(get_nested(config, "references.exp209_likpf_rmse")),
        },
        "exact_hmm_likpf_50_50": {
            "actual": float(blend_overall["control_rmse"]),
            "expected": float(get_nested(config, "references.exp209_hmm_likpf_50_50_rmse")),
        },
    }
    for record in baseline_parity.values():
        record["absolute_difference"] = abs(record["actual"] - record["expected"])
        record["passed"] = bool(record["absolute_difference"] <= baseline_tolerance)

    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_runs = int(get_nested(config, "execution_contract.hmm_well_runs"))
    finite_columns = [
        f"{primary}_hmm_tvt",
        f"{primary}_likpf_50_50",
    ]
    observed_rows = int(runtime["raw_observed_rows"].sum())
    missing_rows = int(runtime["raw_missing_rows"].sum())
    sigma_values = observation_audit["exp209_zero_fill_sigma"].to_numpy(np.float64)
    emission_finite_coverage = float(runtime["emission_finite_coverage"].min())
    df_values = runtime["student_t_degrees_of_freedom"].to_numpy(np.float64)
    technical = {
        "input_preflight_passed": True,
        "truth_rows_accessed_before_prediction_freeze": 0,
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "observation_audit_wells": len(observation_audit),
        "finite_coverage": float(np.isfinite(frame[finite_columns].to_numpy(np.float64)).mean()),
        "id_mismatches": 0,
        "hmm_well_runs": len(runtime),
        "expected_hmm_well_runs": expected_runs,
        "variant_order": runtime["variant"].drop_duplicates().tolist(),
        "variant_run_counts": runtime["variant"].value_counts().sort_index().to_dict(),
        "posterior_normalization_max_abs_error": float(
            runtime["posterior_row_sum_max_abs_error"].max()
        ),
        "student_t_emission_finite_coverage": emission_finite_coverage,
        "student_t_df_contract_passed": bool(
            np.isfinite(df_values).all() and np.all(df_values == 4.0)
        ),
        "exp209_sigma_finite_coverage": float(np.isfinite(sigma_values).mean()),
        "exp209_sigma_clip_contract_passed": bool(
            np.isfinite(sigma_values).all()
            and (sigma_values >= 10.0).all()
            and (sigma_values <= 60.0).all()
        ),
        "raw_observed_rows": observed_rows,
        "raw_missing_rows": missing_rows,
        "raw_mask_partition_passed": bool(
            observed_rows > 0 and missing_rows > 0 and observed_rows + missing_rows == expected_rows
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.kaggle.runtime_limit_seconds")),
        "baseline_metric_parity": baseline_parity,
        "raw_identity_sha256": preflight["raw_train"]["content_sha256"],
    }
    technical["passed"] = bool(
        technical["prediction_rows"] == expected_rows
        and technical["prediction_wells"] == expected_wells
        and technical["observation_audit_wells"] == expected_wells
        and technical["finite_coverage"] == 1.0
        and technical["id_mismatches"] == 0
        and technical["hmm_well_runs"] == expected_runs
        and technical["variant_order"] == list(VARIANT_ORDER)
        and technical["variant_run_counts"] == {primary: expected_wells}
        and technical["posterior_normalization_max_abs_error"] <= 1.0e-6
        and technical["student_t_emission_finite_coverage"] == 1.0
        and technical["student_t_df_contract_passed"]
        and technical["exp209_sigma_finite_coverage"] == 1.0
        and technical["exp209_sigma_clip_contract_passed"]
        and technical["raw_mask_partition_passed"]
        and runtime_seconds <= technical["runtime_limit_seconds"]
        and all(bool(record["passed"]) for record in baseline_parity.values())
    )

    def scope_record(scope: str) -> pd.Series:
        selected = direct.loc[direct["scope"] == scope]
        if len(selected) != 1:
            raise ValueError(f"exp374 expected exactly one direct metric row for scope={scope}")
        return selected.iloc[0]

    folds = direct.loc[direct["scope"].str.startswith("fold_")]
    folds_improved = int((folds["delta_rmse_candidate_minus_control"] < -tolerance).sum())
    non_regression_scopes = {
        "raw_gr_missing": bool(
            scope_record("raw_gr_missing")["delta_rmse_candidate_minus_control"] <= tolerance
        ),
        "missing_fraction_high": bool(
            scope_record("missing_fraction_high")["delta_rmse_candidate_minus_control"] <= tolerance
        ),
        "md_since_1000_plus": bool(
            scope_record("md_since_1000_plus")["delta_rmse_candidate_minus_control"] <= tolerance
        ),
        "hidden_like_spatial": bool(
            scope_record("hidden_like_spatial")["delta_rmse_candidate_minus_control"] <= tolerance
        ),
        "hidden_like_typewell_purged": bool(
            scope_record("hidden_like_typewell_purged")["delta_rmse_candidate_minus_control"]
            <= tolerance
        ),
    }
    observed = scope_record("raw_gr_observed")
    primary_by_well = by_well.loc[
        (by_well["variant"] == primary) & (by_well["comparison"] == "direct")
    ]
    delta_values = primary_by_well["delta_rmse_candidate_minus_control"]
    p95_delta = float(delta_values.quantile(0.95))
    worst_delta = float(primary_by_well["delta_rmse_candidate_minus_control"].max())
    primary_direct = {
        "candidate_rmse": float(direct_overall["candidate_rmse"]),
        "control_rmse": float(direct_overall["control_rmse"]),
        "improvement_ft": float(direct_overall["improvement_ft"]),
        "minimum_improvement_ft": float(gate["minimum_rmse_gain_vs_exp209_raw_hmm_ft"]),
        "folds_improved": folds_improved,
        "minimum_folds_improved": int(gate["minimum_improved_folds"]),
        "raw_observed_improvement_ft": float(observed["improvement_ft"]),
        "minimum_raw_observed_improvement_ft": float(gate["minimum_raw_observed_row_rmse_gain_ft"]),
        "required_non_regression_scopes": non_regression_scopes,
        "by_well_rmse_delta_p95": p95_delta,
        "by_well_rmse_delta_p95_max": float(gate["require_by_well_p95_delta_max"]),
        "worst_well_rmse_delta": worst_delta,
        "worst_well_rmse_delta_max": float(gate["maximum_worst_well_regression_ft"]),
    }
    primary_direct["passed"] = bool(
        primary_direct["improvement_ft"] >= primary_direct["minimum_improvement_ft"]
        and folds_improved >= primary_direct["minimum_folds_improved"]
        and primary_direct["raw_observed_improvement_ft"]
        >= primary_direct["minimum_raw_observed_improvement_ft"]
        and all(non_regression_scopes.values())
        and p95_delta <= primary_direct["by_well_rmse_delta_p95_max"]
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
            "student_t_exp209_passed_train_side_only_no_automatic_downstream"
            if passed
            else "student_t_exp209_failed_close_without_rescue"
        ),
        "technical_gate": technical,
        "primary_direct_gate": primary_direct,
        "fixed_likpf_50_50_guard": blend_guard,
        "failure_action": (
            "close_without_df_scale_temperature_clip_mixture_huber_sigma_missing_weight_"
            "transition_grid_blend_rescue_inference_or_submission"
        ),
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
            "Full exp374 must run first on Kaggle; local execution requires explicit smoke approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    control_preflight = preflight_controls_and_assignments(config)
    wells = control_preflight["fold_assignment"].pop("well_ids")
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
        raw_dir, artifacts, config, wells
    )
    prediction_frozen_at_seconds = time.time() - started
    observation_audit = pd.read_csv(
        output_paths["observation_audit"], dtype={"well_id": str}
    )
    # Unknown-suffix truth and row-level saved controls are first parsed here.
    frame, late_attachment = load_late_readout_frame(preflight, frozen, output_paths, config)
    paired_metrics, by_well_metrics = build_paired_metrics(frame, config)
    runtime_seconds = time.time() - started
    promotion_gate = evaluate_promotion_gate(
        paired_metrics,
        by_well_metrics,
        frame,
        runtime,
        observation_audit,
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
        "train_side_student_t_exp209_gate_passed_no_automatic_downstream"
        if promotion_gate["passed"]
        else "train_side_student_t_exp209_gate_failed_closed"
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
            "raw_gr_emission_contract": Path(output_paths["raw_gr_emission_contract"]),
            "observation_audit": Path(output_paths["observation_audit"]),
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
        "prediction_sha256": frozen["prediction"]["content_sha256"],
        "raw_gr_emission_contract_sha256": frozen["raw_gr_emission_contract"][
            "content_sha256"
        ],
        "observation_audit_sha256": frozen["observation_audit"]["content_sha256"],
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
                "hmm_well_runs": get_nested(CONFIG, "execution_contract.hmm_well_runs"),
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
