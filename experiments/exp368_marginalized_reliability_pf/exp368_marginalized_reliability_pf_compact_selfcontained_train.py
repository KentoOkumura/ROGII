# %% [markdown]
# # exp368 marginalized reliability PF — Stage 0 train-side readout
#
# This notebook implements only the design-frozen, zero-PF Stage 0 diagnostic.
# It first evaluates predictive NLL on a known-prefix 128/64 history/held-out
# split. It then tests whether the same exact `normal / weak` Gaussian forward
# recursion, scored along the saved exp072 `likpf_mean` path, identifies bad
# suffix blocks before unknown-suffix truth is attached. Stage 1 PF replay,
# inference, blending, and submission remain unimplemented.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Input preflight and target-free exp072 path preparation
# 5. Exact reliability recursion, known-prefix NLL, and suffix block freeze
# 6. Late truth, fold, and hidden-like attachment
# 7. AUC, quartile, fold, and promotion-gate readout
# 8. Metrics, diagnostics, and generated artifacts
# 9. Setup and configuration preview
# 10. Run the approved Kaggle CPU Stage 0

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

import numpy as np
import pandas as pd
import yaml

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover
    def get_ipython() -> None:
        return None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp368_marginalized_reliability_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FREE_FORBIDDEN = {
    "TVT",
    "target",
    "tvt_true",
    "true_tvt",
    "error",
    "abs_error",
    "block_rmse",
    "bad10",
}

PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP368_IMPORT_ONLY", "0") == "1"
EXECUTE_NOTEBOOK = get_ipython() is not None and not IMPORT_ONLY


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return PACKAGE_DIR


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"Could not locate exp368 config; checked={candidates}")


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


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame, columns: Iterable[str] | None = None
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


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.17g",
    )
    report = inspect_gzip_csv(path)
    report["rows"] = len(frame)
    report["schema_sha256"] = dataframe_schema_sha(frame)
    report["dataframe_content_sha256"] = dataframe_content_sha(frame)
    return report


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        paths = (
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            PACKAGE_DIR / candidate
            if candidate.name == filename
            else PACKAGE_DIR / candidate / filename,
        )
        for path in paths:
            checked.append(str(path))
            if path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


def assert_target_free(frame: pd.DataFrame, *, stage: str) -> None:
    leaked = sorted(TARGET_FREE_FORBIDDEN.intersection(frame.columns))
    if leaked:
        raise ValueError(f"{stage} contains forbidden pre-freeze columns: {leaked}")


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_kaggle_approval: bool = False
) -> None:
    checks = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp072_exp063_full_replay_feature_cache",
        "implementation.enabled": True,
        "implementation.scope": "stage0_train_side_reliability_readout",
        "implementation.stage_1_implemented": False,
        "validation.n_folds": 5,
        "validation.expected_folds": [0, 1, 2, 3, 4],
        "validation.score_rows": "unknown_suffix_only",
        "validation.truth_attachment": "after_prefix_and_suffix_feature_content_sha_freeze",
        "validation.stage_0.known_prefix.history_rows": 128,
        "validation.stage_0.known_prefix.heldout_rows": 64,
        "validation.stage_0.known_prefix.selection": (
            "final_contiguous_192_rows_with_finite_tvt_input"
        ),
        "validation.stage_0.known_prefix.gr_preprocessing": (
            "exp072_linear_interpolate_both_directions_then_typewell_mean"
        ),
        "validation.stage_0.known_prefix.sigma_fit_rows": (
            "exp072_full_known_prefix_with_missing_gr_filled_zero"
        ),
        "validation.stage_0.suffix.block_rows": 512,
        "validation.stage_0.suffix.stride_rows": 256,
        "validation.stage_0.suffix.tail_policy": "keep_short_tail_from_stride_starts",
        "validation.stage_0.suffix.bad_block_label_after_freeze": (
            "exp072_likpf_block_rmse_greater_than_or_equal_10ft"
        ),
        "validation.stage_0.suffix.posterior_score": (
            "row_weighted_mean_forward_filtered_weak_probability"
        ),
        "validation.stage_0.suffix.fold_pass_definition": (
            "real_bad10_auc_strictly_greater_than_0p50"
        ),
        "validation.stage_0.suffix.negative_control.kind": (
            "within_well_nonzero_circular_shift_of_block_weak_score"
        ),
        "model.gr_reliability.states": ["normal", "weak"],
        "model.gr_reliability.initial_probability": [0.8, 0.2],
        "model.gr_reliability.normal_sigma_multiplier": 1.0,
        "model.gr_reliability.weak_sigma_multiplier": 4.0,
        "model.gr_reliability.update": (
            "exact_two_state_normalized_gaussian_forward_per_path"
        ),
        "model.path_emission.kind": "normalized_gaussian_logpdf_with_squared_z_clip",
        "model.path_emission.suffix_sigma_source": (
            "exp072_full_known_prefix_residual_std"
        ),
        "execution.implementation_approved": True,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "execution_contract.stage_0.diagnostic_variants": 1,
        "execution_contract.stage_0.reporting_folds": 5,
        "execution_contract.stage_0.pf_seed_well_runs": 0,
        "execution_contract.stage_0.pf_control_replays": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "runtime.use_gpu": False,
        "runtime.num_workers": 1,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in checks.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(
                f"exp368 frozen contract changed: {key}={actual!r}, expected {expected!r}"
            )

    transition = np.asarray(
        get_nested(config, "model.gr_reliability.transition_matrix"), dtype=np.float64
    )
    expected_transition = np.asarray(
        [[511.0 / 512.0, 1.0 / 512.0], [1.0 / 128.0, 127.0 / 128.0]],
        dtype=np.float64,
    )
    if transition.shape != (2, 2) or not np.array_equal(transition, expected_transition):
        raise ValueError("exp368 q transition matrix must remain exactly design-frozen")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=0.0, rtol=0.0):
        raise ValueError("exp368 q transition rows must sum exactly to one")

    gates = get_nested(config, "validation.stage_0.all_required") or {}
    expected_gates = {
        "minimum_bad_block_auc": 0.60,
        "minimum_auc_gain_over_circular": 0.02,
        "minimum_known_prefix_predictive_nll_gain_fraction": 0.01,
        "minimum_passing_folds": 4,
        "minimum_hidden_like_auc": 0.55,
        "weak_posterior_mean_range": [0.02, 0.50],
    }
    if gates != expected_gates:
        raise ValueError("exp368 Stage 0 gate values must remain design-frozen")

    forbidden = set(get_nested(config, "model.forbidden") or [])
    required_forbidden = {
        "sampled_reliability_particles",
        "uniform_outlier_component",
        "adaptive_temperature_or_sigma",
        "transition_or_sigma_multiplier_grid",
        "rate_prediction_from_prefix_or_geometry",
        "particle_or_seed_increase",
        "blend_or_selector",
        "parent_pf_control_replay",
    }
    if forbidden != required_forbidden:
        raise ValueError("exp368 forbidden-operation contract changed")

    if require_kaggle_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise PermissionError("exp368 Kaggle Stage 0 run is not approved")
        if not bool(get_nested(config, "execution.run_stage_0")):
            raise PermissionError("exp368 execution.run_stage_0 must be true for a Kaggle run")


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "stage": "stage_0_only",
        "truth_attached": False,
        "path_emission": get_nested(config, "model.path_emission"),
        "gr_reliability": get_nested(config, "model.gr_reliability"),
        "stage_0_contract": get_nested(config, "validation.stage_0"),
        "execution_contract": get_nested(config, "execution_contract"),
        "forbidden": get_nested(config, "model.forbidden"),
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "rng": "none",
    }
    contract["content_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Input preflight and target-free exp072 path preparation


# %%
def _candidate_paths(spec: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in spec.get("candidates", [])]


def list_raw_wells(raw_dir: Path) -> list[str]:
    horizontal = {
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.replace("__typewell.csv", "") for path in raw_dir.glob("*__typewell.csv")
    }
    wells = sorted(horizontal.intersection(typewell))
    if horizontal != typewell:
        raise ValueError("raw horizontal/typewell well identity mismatch")
    return wells


def raw_well_identity_manifest(raw_dir: Path, wells: list[str]) -> pd.DataFrame:
    rows = []
    for well in wells:
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    return pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)


def load_saved_exp072_path(
    path: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    saved = get_nested(config, "data.saved_exp072_path") or {}
    columns = [
        str(saved["id_column"]),
        str(saved["well_column"]),
        str(saved["base_column"]),
        str(saved["delta_column"]),
    ]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={str(saved["id_column"]): str, str(saved["well_column"]): str},
    )
    frame.rename(
        columns={
            str(saved["id_column"]): "id",
            str(saved["well_column"]): "well_id",
        },
        inplace=True,
    )
    frame["row_idx"] = pd.to_numeric(
        frame["id"].astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
    ).astype(np.int64)
    base = pd.to_numeric(frame.pop(str(saved["base_column"])), errors="raise")
    delta = pd.to_numeric(frame.pop(str(saved["delta_column"])), errors="raise")
    frame["path_tvt"] = (base + delta).astype(np.float64)
    frame.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    assert_target_free(frame, stage="saved exp072 path")
    if frame["id"].duplicated().any():
        raise ValueError("saved exp072 path has duplicate row identities")
    if not np.isfinite(frame[["path_tvt"]].to_numpy()).all():
        raise ValueError("saved exp072 path contains non-finite values")
    report = {
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "safe_columns_read": columns,
        "forbidden_columns_read": [],
        "schema_sha256": dataframe_schema_sha(frame),
        "content_sha256": dataframe_content_sha(frame),
    }
    return frame, report


def preflight_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_dir = train_data_dir(config)
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    wells = list_raw_wells(raw_dir)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(wells) != expected_wells:
        raise ValueError(f"raw wells={len(wells)}, expected={expected_wells}")

    saved = get_nested(config, "data.saved_exp072_path") or {}
    saved_path = resolve_existing(str(saved["filename"]), _candidate_paths(saved))
    saved_file_report = {
        "path": str(saved_path),
        "bytes": saved_path.stat().st_size,
        "raw_sha256": sha256_path(saved_path),
    }
    if saved_file_report["raw_sha256"] != str(saved["expected_raw_gzip_sha256"]):
        raise ValueError("saved exp072 path raw gzip SHA mismatch")

    fold = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(str(fold["filename"]), _candidate_paths(fold))
    fold_report = inspect_gzip_csv(fold_path)
    if fold_report["decompressed_sha256"] != str(fold["expected_decompressed_sha256"]):
        raise ValueError("fold/truth assignment decompressed SHA mismatch")

    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    hidden_path = resolve_existing(str(hidden["filename"]), _candidate_paths(hidden))
    hidden_raw_sha = sha256_path(hidden_path)
    if hidden_raw_sha != str(hidden["expected_sha256"]):
        raise ValueError("hidden-like assignment raw SHA mismatch")

    expected_rows = int(get_nested(config, "validation.expected_rows"))
    path_frame, path_report = load_saved_exp072_path(saved_path, config)
    path_wells = sorted(path_frame["well_id"].unique().tolist())
    if path_wells != wells:
        raise ValueError("saved exp072 path/raw well identity mismatch")
    if len(path_frame) != expected_rows:
        raise ValueError("saved exp072 path parsed row count mismatch")
    saved_file_report["data_rows"] = len(path_frame)
    saved_file_report["safe_projection_content_sha256"] = path_report["content_sha256"]
    return {
        "paths": {
            "raw_dir": raw_dir,
            "saved_exp072_path": saved_path,
            "fold_assignment": fold_path,
            "hidden_like_assignment": hidden_path,
        },
        "wells": wells,
        "saved_path_frame": path_frame,
        "input_reports": {
            "saved_exp072_path_file": saved_file_report,
            "saved_exp072_path_safe_frame": path_report,
            "fold_assignment_file": fold_report,
            "hidden_like_assignment_file": {
                "path": str(hidden_path),
                "raw_sha256": hidden_raw_sha,
            },
        },
    }


def prepare_typewell(typewell_path: Path) -> tuple[np.ndarray, np.ndarray]:
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell["TVT"] = pd.to_numeric(typewell["TVT"], errors="coerce")
    typewell["GR"] = pd.to_numeric(typewell["GR"], errors="coerce")
    typewell = (
        typewell.dropna(subset=["TVT", "GR"])
        .sort_values("TVT", kind="mergesort")
        .drop_duplicates("TVT", keep="last")
        .reset_index(drop=True)
    )
    if len(typewell) < 2:
        raise ValueError(f"{typewell_path}: fewer than two finite Type Well rows")
    tvt = typewell["TVT"].to_numpy(np.float64)
    gr = typewell["GR"].to_numpy(np.float64)
    if np.any(np.diff(tvt) <= 0.0):
        raise ValueError(f"{typewell_path}: Type Well TVT must be strictly increasing")
    return tvt, gr


def clipped_gr_sigma(
    residual: np.ndarray,
    config: Mapping[str, Any],
) -> float:
    values = np.asarray(residual, dtype=np.float64)
    values = values[np.isfinite(values)]
    emission = get_nested(config, "model.path_emission") or {}
    if len(values) < 20:
        return float(emission["gr_sigma_default"])
    return float(
        np.clip(
            np.std(values),
            float(emission["gr_sigma_min"]),
            float(emission["gr_sigma_max"]),
        )
    )


def load_target_free_gr_path_rows(
    well_id: str,
    path_rows: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    horizontal_path = raw_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well_id}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
    tw_tvt, tw_gr = prepare_typewell(typewell_path)

    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce")
    typewell_mean = float(np.mean(tw_gr))
    processed_gr = raw_gr.interpolate(limit_direction="both").fillna(typewell_mean)
    prefix_tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    prefix_mask = prefix_tvt.notna()
    prefix_residual = (
        raw_gr.loc[prefix_mask].fillna(0.0).to_numpy(np.float64)
        - np.interp(prefix_tvt.loc[prefix_mask].to_numpy(np.float64), tw_tvt, tw_gr)
    )
    sigma = clipped_gr_sigma(prefix_residual, config)
    row_idx = path_rows["row_idx"].to_numpy(np.int64)
    if len(row_idx) == 0 or row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise ValueError(f"{well_id}: saved path row index is outside raw horizontal rows")
    if len(np.unique(row_idx)) != len(row_idx) or np.any(np.diff(row_idx) <= 0):
        raise ValueError(f"{well_id}: saved path rows must be strictly increasing")

    path_tvt = path_rows["path_tvt"].to_numpy(np.float64)
    expected_gr = np.interp(path_tvt, tw_tvt, tw_gr)
    observed_gr = processed_gr.to_numpy(np.float64)[row_idx]
    output = path_rows[["id", "well_id", "row_idx"]].copy()
    output["suffix_offset"] = np.arange(len(output), dtype=np.int64)
    output["raw_gr_observed"] = np.isfinite(raw_gr.to_numpy(np.float64)[row_idx]).astype(
        np.int8
    )
    output["gr_residual"] = observed_gr - expected_gr
    output["gr_sigma"] = sigma
    assert_target_free(output, stage=f"{well_id} target-free GR path rows")
    if not np.isfinite(output[["gr_residual", "gr_sigma"]].to_numpy()).all():
        raise ValueError(f"{well_id}: non-finite target-free GR path values")
    return output


# %% [markdown]
# ## 5. Exact reliability recursion, known-prefix NLL, and suffix block freeze


# %%
def gaussian_log_density(
    residual: np.ndarray,
    sigma: float,
    sigma_multiplier: float,
    squared_z_clip: float,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    scale = float(sigma) * float(sigma_multiplier)
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Gaussian scale must be finite and positive")
    z2 = np.minimum((residual / scale) ** 2, float(squared_z_clip))
    return -0.5 * math.log(2.0 * math.pi) - math.log(scale) - 0.5 * z2


def sticky_gaussian_forward_filter(
    residual: np.ndarray,
    sigma: float,
    transition: np.ndarray,
    initial_probability: np.ndarray,
    sigma_multipliers: np.ndarray,
    squared_z_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float64)
    transition = np.asarray(transition, dtype=np.float64)
    posterior = np.asarray(initial_probability, dtype=np.float64).copy()
    multipliers = np.asarray(sigma_multipliers, dtype=np.float64)
    if residual.ndim != 1 or len(residual) == 0 or not np.isfinite(residual).all():
        raise ValueError("sticky forward filter requires finite row residuals")
    if transition.shape != (2, 2) or posterior.shape != (2,) or multipliers.shape != (2,):
        raise ValueError("sticky forward filter expects exactly two reliability states")
    if not np.isclose(posterior.sum(), 1.0) or not np.allclose(
        transition.sum(axis=1), 1.0
    ):
        raise ValueError("invalid sticky reliability probabilities")

    state_log_density = np.column_stack(
        [
            gaussian_log_density(residual, sigma, multiplier, squared_z_clip)
            for multiplier in multipliers
        ]
    )
    weak = np.empty(len(residual), dtype=np.float64)
    predictive_log_density = np.empty(len(residual), dtype=np.float64)
    for row_index in range(len(residual)):
        if row_index > 0:
            posterior = posterior @ transition
        log_weight = np.log(np.maximum(posterior, np.finfo(np.float64).tiny))
        log_weight += state_log_density[row_index]
        maximum = float(np.max(log_weight))
        weight = np.exp(log_weight - maximum)
        evidence = float(weight.sum())
        predictive_log_density[row_index] = maximum + math.log(evidence)
        posterior = weight / evidence
        weak[row_index] = posterior[1]
    if (
        not np.isfinite(weak).all()
        or not np.isfinite(predictive_log_density).all()
        or np.any((weak < 0.0) | (weak > 1.0))
    ):
        raise RuntimeError("sticky weak posterior is invalid")
    return weak, predictive_log_density, posterior


def final_contiguous_true_indices(mask: np.ndarray, required_rows: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    true_indices = np.flatnonzero(values)
    if len(true_indices) == 0:
        return np.empty(0, dtype=np.int64)
    stop = int(true_indices[-1]) + 1
    start = stop
    while start > 0 and values[start - 1]:
        start -= 1
    if stop - start < int(required_rows):
        return np.empty(0, dtype=np.int64)
    return np.arange(stop - int(required_rows), stop, dtype=np.int64)


def build_known_prefix_nll_rows_for_well(
    well_id: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    horizontal = pd.read_csv(
        raw_dir / f"{well_id}__horizontal_well.csv",
        usecols=["GR", "TVT_input"],
    )
    tw_tvt, tw_gr = prepare_typewell(raw_dir / f"{well_id}__typewell.csv")
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    prefix = get_nested(config, "validation.stage_0.known_prefix") or {}
    history_rows = int(prefix["history_rows"])
    heldout_rows = int(prefix["heldout_rows"])
    selected = final_contiguous_true_indices(
        np.isfinite(tvt_input), history_rows + heldout_rows
    )
    if len(selected) != history_rows + heldout_rows:
        raise ValueError(
            f"{well_id}: final contiguous known TVT_input run is shorter than "
            f"{history_rows + heldout_rows} rows"
        )
    processed_gr = (
        pd.Series(raw_gr)
        .interpolate(limit_direction="both")
        .fillna(float(np.mean(tw_gr)))
        .to_numpy(np.float64)
    )
    full_known = np.isfinite(tvt_input)
    sigma_residual = np.nan_to_num(raw_gr[full_known], nan=0.0) - np.interp(
        tvt_input[full_known], tw_tvt, tw_gr
    )
    sigma = clipped_gr_sigma(sigma_residual, config)
    residual = processed_gr[selected] - np.interp(
        tvt_input[selected], tw_tvt, tw_gr
    )
    q = get_nested(config, "model.gr_reliability") or {}
    emission = get_nested(config, "model.path_emission") or {}
    sigma_multipliers = np.asarray(
        [q["normal_sigma_multiplier"], q["weak_sigma_multiplier"]],
        dtype=np.float64,
    )
    weak, marginal_log_density, _ = sticky_gaussian_forward_filter(
        residual,
        sigma,
        np.asarray(q["transition_matrix"], dtype=np.float64),
        np.asarray(q["initial_probability"], dtype=np.float64),
        sigma_multipliers,
        float(emission["squared_z_clip"]),
    )
    base_log_density = gaussian_log_density(
        residual,
        sigma,
        float(q["normal_sigma_multiplier"]),
        float(emission["squared_z_clip"]),
    )
    heldout_slice = slice(history_rows, history_rows + heldout_rows)
    output = pd.DataFrame(
        {
            "well_id": well_id,
            "row_idx": selected[heldout_slice],
            "heldout_offset": np.arange(heldout_rows, dtype=np.int64),
            "gr_residual": residual[heldout_slice],
            "gr_sigma": sigma,
            "raw_gr_observed": np.isfinite(raw_gr[selected[heldout_slice]]).astype(
                np.int8
            ),
            "base_log_density": base_log_density[heldout_slice],
            "marginal_log_density": marginal_log_density[heldout_slice],
            "weak_posterior": weak[heldout_slice],
        }
    )
    output["base_nll"] = -output["base_log_density"]
    output["marginal_nll"] = -output["marginal_log_density"]
    output["nll_gain"] = output["base_nll"] - output["marginal_nll"]
    assert_target_free(output, stage=f"{well_id} target-free known-prefix NLL")
    if not np.isfinite(
        output[
            [
                "gr_residual",
                "gr_sigma",
                "base_nll",
                "marginal_nll",
                "nll_gain",
                "weak_posterior",
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError(f"{well_id}: non-finite known-prefix NLL readout")
    return output


def stable_circular_offset(well_id: str, block_count: int, key_prefix: str) -> int:
    if block_count <= 1:
        return 0
    digest = hashlib.sha256(f"{key_prefix}|{well_id}".encode()).hexdigest()
    return 1 + int(digest[:16], 16) % (block_count - 1)


def build_well_block_features(
    well_rows: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows = int(get_nested(config, "validation.stage_0.suffix.block_rows"))
    stride_rows = int(get_nested(config, "validation.stage_0.suffix.stride_rows"))
    q = get_nested(config, "model.gr_reliability") or {}
    emission = get_nested(config, "model.path_emission") or {}
    transition = np.asarray(q["transition_matrix"], dtype=np.float64)
    initial = np.asarray(q["initial_probability"], dtype=np.float64)
    sigma_multipliers = np.asarray(
        [q["normal_sigma_multiplier"], q["weak_sigma_multiplier"]],
        dtype=np.float64,
    )
    rows = well_rows.sort_values("suffix_offset", kind="mergesort").reset_index(drop=True)
    expected_offsets = np.arange(len(rows), dtype=np.int64)
    if not np.array_equal(rows["suffix_offset"].to_numpy(np.int64), expected_offsets):
        raise ValueError("suffix offsets must be contiguous and zero-based")

    ledger_records = []
    posterior_records = []
    block_id = 0
    for start in range(0, len(rows), stride_rows):
        stop = min(start + block_rows, len(rows))
        part = rows.iloc[start:stop]
        sigma_values = part["gr_sigma"].to_numpy(np.float64)
        if not np.allclose(sigma_values, sigma_values[0], atol=0.0, rtol=0.0):
            raise ValueError("suffix block sigma must be constant within a well")
        weak, marginal_log_density, _ = sticky_gaussian_forward_filter(
            part["gr_residual"].to_numpy(np.float64),
            float(sigma_values[0]),
            transition,
            initial,
            sigma_multipliers,
            float(emission["squared_z_clip"]),
        )
        base_log_density = gaussian_log_density(
            part["gr_residual"].to_numpy(np.float64),
            float(sigma_values[0]),
            float(q["normal_sigma_multiplier"]),
            float(emission["squared_z_clip"]),
        )
        ledger_records.append(
            {
                "well_id": str(part["well_id"].iloc[0]),
                "block_id": block_id,
                "start_suffix_offset": start,
                "stop_suffix_offset_exclusive": stop,
                "start_row_idx": int(part["row_idx"].iloc[0]),
                "end_row_idx": int(part["row_idx"].iloc[-1]),
                "block_row_count": len(part),
                "raw_gr_observed_rows": int(part["raw_gr_observed"].sum()),
                "raw_gr_observed_fraction": float(part["raw_gr_observed"].mean()),
                "gr_sigma": float(sigma_values[0]),
                "mean_base_log_density": float(base_log_density.mean()),
                "mean_marginal_log_density": float(marginal_log_density.mean()),
            }
        )
        posterior_records.append(
            {
                "well_id": str(part["well_id"].iloc[0]),
                "block_id": block_id,
                "weak_posterior_sum": float(weak.sum()),
                "weak_posterior_mean": float(weak.mean()),
                "weak_posterior_last": float(weak[-1]),
                "weak_posterior_min": float(weak.min()),
                "weak_posterior_max": float(weak.max()),
            }
        )
        block_id += 1
    return pd.DataFrame(ledger_records), pd.DataFrame(posterior_records)


def freeze_target_free_blocks(
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    path_frame = preflight["saved_path_frame"]
    raw_dir = preflight["paths"]["raw_dir"]
    ledgers = []
    posteriors = []
    prefix_nll_rows = []
    for index, well_id in enumerate(preflight["wells"], start=1):
        path_rows = path_frame.loc[path_frame["well_id"] == well_id].copy()
        prefix_nll_rows.append(
            build_known_prefix_nll_rows_for_well(str(well_id), raw_dir, config)
        )
        target_free_rows = load_target_free_gr_path_rows(
            str(well_id), path_rows, raw_dir, config
        )
        ledger, posterior = build_well_block_features(target_free_rows, config)
        ledgers.append(ledger)
        posteriors.append(posterior)
        if index == 1 or index % 25 == 0 or index == len(preflight["wells"]):
            print(
                f"[{index}/{len(preflight['wells'])}] target-free weak posterior "
                f"well={well_id} blocks={len(ledger)}",
                flush=True,
            )

    ledger = (
        pd.concat(ledgers, ignore_index=True)
        .sort_values(["well_id", "block_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    posterior = (
        pd.concat(posteriors, ignore_index=True)
        .sort_values(["well_id", "block_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    prefix_nll = (
        pd.concat(prefix_nll_rows, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    assert_target_free(prefix_nll, stage="target-free known-prefix NLL")
    assert_target_free(ledger, stage="target-free block ledger")
    assert_target_free(posterior, stage="target-free weak posterior")
    if len(ledger) != len(posterior):
        raise ValueError("block ledger/posterior row count mismatch")
    if not ledger[["well_id", "block_id"]].equals(posterior[["well_id", "block_id"]]):
        raise ValueError("block ledger/posterior identity mismatch")

    negative = get_nested(config, "validation.stage_0.suffix.negative_control") or {}
    posterior["circular_weak_score"] = np.nan
    posterior["circular_offset_blocks"] = 0
    for well_id, indices in posterior.groupby("well_id", sort=True).groups.items():
        ordered_indices = posterior.loc[indices].sort_values(
            "block_id", kind="mergesort"
        ).index
        values = posterior.loc[ordered_indices, "weak_posterior_mean"].to_numpy(
            np.float64
        )
        offset = stable_circular_offset(
            str(well_id), len(values), str(negative["key_prefix"])
        )
        posterior.loc[ordered_indices, "circular_weak_score"] = np.roll(values, offset)
        posterior.loc[ordered_indices, "circular_offset_blocks"] = offset
        if not np.array_equal(
            np.sort(values),
            np.sort(
                posterior.loc[ordered_indices, "circular_weak_score"].to_numpy(
                    np.float64
                )
            ),
        ):
            raise RuntimeError("circular control failed to preserve weak-score values")
        if len(values) > 1 and offset == 0:
            raise RuntimeError("multi-block circular control offset must be nonzero")

    quantile_low, quantile_high = [
        float(value)
        for value in get_nested(config, "validation.stage_0.suffix.quartile_edges")
    ]
    q1_boundary = float(posterior["weak_posterior_mean"].quantile(quantile_low))
    q4_boundary = float(posterior["weak_posterior_mean"].quantile(quantile_high))
    posterior["weak_quartile"] = 0
    if q1_boundary < q4_boundary:
        posterior.loc[posterior["weak_posterior_mean"] <= q1_boundary, "weak_quartile"] = 1
        posterior.loc[posterior["weak_posterior_mean"] >= q4_boundary, "weak_quartile"] = 4
    posterior["weak_quartile"] = posterior["weak_quartile"].astype(np.int8)

    ledger_path = artifacts / f"{OUTPUT_PREFIX}_target_free_block_ledger.csv.gz"
    posterior_path = (
        artifacts / f"{OUTPUT_PREFIX}_target_free_weak_posterior_blocks.csv.gz"
    )
    prefix_nll_path = (
        artifacts / f"{OUTPUT_PREFIX}_target_free_known_prefix_nll.csv.gz"
    )
    reports = {
        "known_prefix_nll": write_deterministic_gzip_csv(prefix_nll, prefix_nll_path),
        "block_ledger": write_deterministic_gzip_csv(ledger, ledger_path),
        "weak_posterior_blocks": write_deterministic_gzip_csv(
            posterior, posterior_path
        ),
    }
    for report in reports.values():
        report["frozen_before_truth_attachment"] = True
    freeze = {
        "truth_attached": False,
        "rows": len(posterior),
        "wells": int(posterior["well_id"].nunique()),
        "known_prefix_rows": len(prefix_nll),
        "known_prefix_wells": int(prefix_nll["well_id"].nunique()),
        "q1_boundary": q1_boundary,
        "q4_boundary": q4_boundary,
        "strict_quartile_boundaries": bool(q1_boundary < q4_boundary),
        "block_ledger_content_sha256": reports["block_ledger"]["content_sha256"],
        "weak_posterior_content_sha256": reports["weak_posterior_blocks"][
            "content_sha256"
        ],
        "known_prefix_nll_content_sha256": reports["known_prefix_nll"][
            "content_sha256"
        ],
        "forbidden_columns_present": [],
        "truth_columns_read_before_freeze": 0,
    }
    return posterior.merge(
        ledger, on=["well_id", "block_id"], how="inner", validate="one_to_one"
    ), prefix_nll, {"reports": reports, "freeze": freeze}, {
        "known_prefix_nll": prefix_nll_path,
        "block_ledger": ledger_path,
        "weak_posterior_blocks": posterior_path,
    }


# %% [markdown]
# ## 6. Late truth, fold, and hidden-like attachment


# %%
def require_frozen_blocks(frozen: Mapping[str, Any]) -> None:
    freeze = frozen["freeze"]
    reports = frozen["reports"]
    if freeze["truth_attached"] or freeze["truth_columns_read_before_freeze"] != 0:
        raise RuntimeError("block feature freeze boundary was violated")
    for key in ("known_prefix_nll", "block_ledger", "weak_posterior_blocks"):
        report = reports[key]
        if (
            not report.get("frozen_before_truth_attachment")
            or len(str(report.get("content_sha256", ""))) != 64
        ):
            raise RuntimeError(f"{key} is not content-SHA frozen")


def load_late_row_readout(
    preflight: Mapping[str, Any],
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen_blocks(frozen)
    fold = get_nested(config, "data.fold_assignment") or {}
    truth_columns = [str(value) for value in fold["truth_columns"]]
    truth = pd.read_csv(
        preflight["paths"]["fold_assignment"],
        usecols=[*truth_columns, "fold"],
        dtype={"well_id": str},
    )
    truth["row_idx"] = pd.to_numeric(truth["row_idx"], errors="raise").astype(np.int64)
    truth["id"] = truth["well_id"].astype(str) + "_" + truth["row_idx"].astype(str)
    truth["true_tvt"] = pd.to_numeric(truth.pop("tvt_true"), errors="raise").astype(
        np.float64
    )
    truth["fold"] = pd.to_numeric(truth["fold"], errors="raise").astype(np.int8)
    truth.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    truth.reset_index(drop=True, inplace=True)

    path = preflight["saved_path_frame"][
        ["id", "well_id", "row_idx", "path_tvt"]
    ].copy()
    path.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    path.reset_index(drop=True, inplace=True)
    if len(path) != len(truth) or not np.array_equal(
        path["id"].astype(str).to_numpy(), truth["id"].astype(str).to_numpy()
    ):
        raise ValueError("saved exp072 path/late truth row identity mismatch")
    rows = path.merge(
        truth[["id", "well_id", "row_idx", "fold", "true_tvt"]],
        on=["id", "well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if rows[["fold", "true_tvt"]].isna().any().any():
        raise ValueError("late truth/fold coverage mismatch")

    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    role_columns = [str(value) for value in hidden["role_columns"].values()]
    roles = pd.read_csv(
        preflight["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns],
        dtype={"well_id": str},
    )
    if roles["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    roles = roles.set_index("well_id")
    for scope, role_column in hidden["role_columns"].items():
        rows[str(scope)] = rows["well_id"].map(roles[str(role_column)]).eq("valid")
    numeric = rows[["path_tvt", "true_tvt"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("late row readout contains non-finite prediction/truth")
    return rows, {
        "truth_attachment_stage": (
            "after_prefix_and_suffix_feature_content_sha_freeze"
        ),
        "rows": len(rows),
        "wells": int(rows["well_id"].nunique()),
        "block_ledger_content_sha256": frozen["freeze"][
            "block_ledger_content_sha256"
        ],
        "weak_posterior_content_sha256": frozen["freeze"][
            "weak_posterior_content_sha256"
        ],
        "known_prefix_nll_content_sha256": frozen["freeze"][
            "known_prefix_nll_content_sha256"
        ],
        "identity_mismatches": 0,
    }


def attach_block_truth(
    target_free_blocks: pd.DataFrame,
    row_readout: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    bad_threshold = float(
        get_nested(config, "validation.stage_0.suffix.bad_block_rmse_ft")
    )
    by_well = {
        str(well): part.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        for well, part in row_readout.groupby("well_id", sort=True)
    }
    records = []
    for block in target_free_blocks.itertuples(index=False):
        rows = by_well[str(block.well_id)]
        start = int(block.start_suffix_offset)
        stop = int(block.stop_suffix_offset_exclusive)
        selected = rows.iloc[start:stop]
        if len(selected) != int(block.block_row_count):
            raise ValueError("late block slice row count mismatch")
        if selected["fold"].nunique() != 1:
            raise ValueError("a well block crosses fold assignments")
        squared_error = (
            selected["path_tvt"].to_numpy(np.float64)
            - selected["true_tvt"].to_numpy(np.float64)
        ) ** 2
        records.append(
            {
                "well_id": str(block.well_id),
                "block_id": int(block.block_id),
                "fold": int(selected["fold"].iloc[0]),
                "block_rmse": float(np.sqrt(np.mean(squared_error))),
                "bad10": bool(np.sqrt(np.mean(squared_error)) >= bad_threshold),
                "hidden_like_spatial": bool(selected["hidden_like_spatial"].iloc[0]),
                "hidden_like_typewell_purged": bool(
                    selected["hidden_like_typewell_purged"].iloc[0]
                ),
            }
        )
    truth = pd.DataFrame(records)
    output = target_free_blocks.merge(
        truth, on=["well_id", "block_id"], how="left", validate="one_to_one"
    )
    if output[["fold", "block_rmse", "bad10"]].isna().any().any():
        raise ValueError("late block truth did not cover all target-free blocks")
    return output.sort_values(["well_id", "block_id"], kind="mergesort").reset_index(
        drop=True
    )


# %% [markdown]
# ## 7. AUC, quartile, fold, and promotion-gate readout


# %%
def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("AUC labels/scores contract is invalid")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(np.float64)
    positive_rank_sum = float(ranks[labels].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def metric_row(frame: pd.DataFrame, mask: np.ndarray, scope: str) -> dict[str, Any]:
    part = frame.loc[mask]
    real_auc = roc_auc_binary(
        part["bad10"].to_numpy(bool),
        part["weak_posterior_mean"].to_numpy(np.float64),
    )
    circular_auc = roc_auc_binary(
        part["bad10"].to_numpy(bool),
        part["circular_weak_score"].to_numpy(np.float64),
    )
    q1 = part.loc[part["weak_quartile"] == 1, "block_rmse"]
    q4 = part.loc[part["weak_quartile"] == 4, "block_rmse"]
    q4_minus_q1 = (
        float(q4.mean() - q1.mean()) if len(q1) > 0 and len(q4) > 0 else None
    )
    weak_mass = float(
        part["weak_posterior_sum"].sum() / part["block_row_count"].sum()
    )
    return {
        "scope": scope,
        "blocks": len(part),
        "wells": int(part["well_id"].nunique()),
        "bad_blocks": int(part["bad10"].sum()),
        "good_blocks": int((~part["bad10"]).sum()),
        "bad_block_rate": float(part["bad10"].mean()),
        "real_bad10_auc": real_auc,
        "circular_bad10_auc": circular_auc,
        "real_minus_circular_auc": (
            float(real_auc - circular_auc)
            if real_auc is not None and circular_auc is not None
            else None
        ),
        "q1_blocks": len(q1),
        "q4_blocks": len(q4),
        "q1_mean_block_rmse": float(q1.mean()) if len(q1) else None,
        "q4_mean_block_rmse": float(q4.mean()) if len(q4) else None,
        "q4_minus_q1_mean_block_rmse": q4_minus_q1,
        "row_weighted_weak_mass": weak_mass,
    }


def build_scope_metrics(
    block_readout: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(block_readout), dtype=bool))
    ]
    for fold in get_nested(config, "validation.expected_folds"):
        scopes.append(
            (
                f"fold_{int(fold)}",
                block_readout["fold"].to_numpy(np.int64) == int(fold),
            )
        )
    scopes.extend(
        [
            (
                "hidden_like_spatial",
                block_readout["hidden_like_spatial"].to_numpy(bool),
            ),
            (
                "hidden_like_typewell_purged",
                block_readout["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    rows = []
    for scope, mask in scopes:
        if not bool(mask.any()):
            raise ValueError(f"scope {scope} contains no blocks")
        rows.append(metric_row(block_readout, mask, scope))
    return pd.DataFrame(rows)


def evaluate_stage_0_gate(
    block_readout: pd.DataFrame,
    prefix_nll: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "validation.stage_0.all_required") or {}
    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"].iloc[0]
    fold_rows = scope_metrics.loc[scope_metrics["scope"].str.startswith("fold_")]
    hidden_rows = scope_metrics.loc[
        scope_metrics["scope"].isin(
            ["hidden_like_spatial", "hidden_like_typewell_purged"]
        )
    ]
    passing_folds = int((fold_rows["real_bad10_auc"].fillna(-np.inf) > 0.50).sum())
    weak_low, weak_high = [float(value) for value in gates["weak_posterior_mean_range"]]

    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    observed_folds = sorted(block_readout["fold"].astype(int).unique().tolist())
    multi_block = block_readout.groupby("well_id", sort=True).size() > 1
    multi_block_wells = set(multi_block[multi_block].index.astype(str))
    multi_offsets = block_readout.loc[
        block_readout["well_id"].isin(multi_block_wells), "circular_offset_blocks"
    ]
    expected_prefix_rows = expected_wells * int(
        get_nested(config, "validation.stage_0.known_prefix.heldout_rows")
    )
    prefix_counts = prefix_nll.groupby("well_id", sort=True).size()

    technical = {
        "expected_rows": expected_rows,
        "saved_path_rows": len(preflight["saved_path_frame"]),
        "expected_wells": expected_wells,
        "readout_wells": int(block_readout["well_id"].nunique()),
        "expected_known_prefix_rows": expected_prefix_rows,
        "known_prefix_rows": len(prefix_nll),
        "known_prefix_wells": int(prefix_nll["well_id"].nunique()),
        "known_prefix_exact_heldout_rows_per_well": bool(
            (prefix_counts == expected_prefix_rows // expected_wells).all()
        ),
        "expected_folds": expected_folds,
        "observed_folds": observed_folds,
        "block_count": len(block_readout),
        "all_scores_finite": bool(
            np.isfinite(
                block_readout[
                    [
                        "weak_posterior_mean",
                        "circular_weak_score",
                        "block_rmse",
                    ]
                ].to_numpy(np.float64)
            ).all()
        ),
        "weak_scores_in_unit_interval": bool(
            block_readout["weak_posterior_mean"].between(0.0, 1.0).all()
        ),
        "strict_quartile_boundaries": bool(
            frozen["freeze"]["strict_quartile_boundaries"]
        ),
        "q1_blocks": int((block_readout["weak_quartile"] == 1).sum()),
        "q4_blocks": int((block_readout["weak_quartile"] == 4).sum()),
        "multi_block_circular_offsets_nonzero": bool((multi_offsets > 0).all()),
        "truth_columns_read_before_freeze": int(
            frozen["freeze"]["truth_columns_read_before_freeze"]
        ),
        "known_prefix_values_finite": bool(
            np.isfinite(
                prefix_nll[
                    ["base_nll", "marginal_nll", "nll_gain", "weak_posterior"]
                ].to_numpy(np.float64)
            ).all()
        ),
        "pf_seed_well_runs": 0,
        "pf_control_replays": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
    }
    technical_pass = bool(
        technical["saved_path_rows"] == expected_rows
        and technical["readout_wells"] == expected_wells
        and technical["known_prefix_rows"] == expected_prefix_rows
        and technical["known_prefix_wells"] == expected_wells
        and technical["known_prefix_exact_heldout_rows_per_well"]
        and technical["observed_folds"] == expected_folds
        and technical["block_count"] > 0
        and technical["all_scores_finite"]
        and technical["weak_scores_in_unit_interval"]
        and technical["strict_quartile_boundaries"]
        and technical["q1_blocks"] > 0
        and technical["q4_blocks"] > 0
        and technical["multi_block_circular_offsets_nonzero"]
        and technical["truth_columns_read_before_freeze"] == 0
        and technical["known_prefix_values_finite"]
    )

    base_nll = float(prefix_nll["base_nll"].sum())
    marginalized_nll = float(prefix_nll["marginal_nll"].sum())
    predictive_nll_gain_fraction = (
        float((base_nll - marginalized_nll) / base_nll)
        if base_nll > 0.0
        else float("-inf")
    )
    real_auc = overall["real_bad10_auc"]
    auc_gain = overall["real_minus_circular_auc"]
    quartile_gap = overall["q4_minus_q1_mean_block_rmse"]
    weak_mass = float(overall["row_weighted_weak_mass"])
    hidden_auc_values = hidden_rows["real_bad10_auc"]
    scientific = {
        "known_prefix_base_predictive_nll": base_nll,
        "known_prefix_marginalized_predictive_nll": marginalized_nll,
        "known_prefix_predictive_nll_gain_fraction": predictive_nll_gain_fraction,
        "minimum_known_prefix_predictive_nll_gain_fraction": float(
            gates["minimum_known_prefix_predictive_nll_gain_fraction"]
        ),
        "pooled_bad10_auc": real_auc,
        "minimum_bad_block_auc": float(gates["minimum_bad_block_auc"]),
        "real_minus_circular_auc": auc_gain,
        "minimum_auc_gain_over_circular": float(
            gates["minimum_auc_gain_over_circular"]
        ),
        "q4_minus_q1_mean_block_rmse_ft": quartile_gap,
        "passing_folds_auc_gt_0p50": passing_folds,
        "minimum_passing_folds": int(gates["minimum_passing_folds"]),
        "hidden_like_auc": {
            str(row.scope): row.real_bad10_auc
            for row in hidden_rows.itertuples(index=False)
        },
        "minimum_hidden_like_auc_each": float(gates["minimum_hidden_like_auc"]),
        "row_weighted_weak_mass": weak_mass,
        "weak_posterior_mean_range": [weak_low, weak_high],
    }
    scientific_pass = bool(
        predictive_nll_gain_fraction
        >= float(gates["minimum_known_prefix_predictive_nll_gain_fraction"])
        and real_auc is not None
        and float(real_auc) >= float(gates["minimum_bad_block_auc"])
        and auc_gain is not None
        and float(auc_gain) >= float(gates["minimum_auc_gain_over_circular"])
        and passing_folds >= int(gates["minimum_passing_folds"])
        and hidden_auc_values.notna().all()
        and bool(
            (
                hidden_auc_values.astype(float)
                >= float(gates["minimum_hidden_like_auc"])
            ).all()
        )
        and weak_low <= weak_mass <= weak_high
    )
    passed = bool(technical_pass and scientific_pass)
    return {
        "stage": "stage_0",
        "technical": technical,
        "technical_pass": technical_pass,
        "scientific": scientific,
        "scientific_pass": scientific_pass,
        "passed": passed,
        "stage_1_eligible": passed,
        "decision": (
            "stage_0_pass_wait_for_separate_stage_1_approval"
            if passed
            else "stage_0_failed_close_without_rescue"
        ),
    }


# %% [markdown]
# ## 8. Metrics, diagnostics, and generated artifacts


# %%
def output_file_reports(paths: Mapping[str, Path]) -> dict[str, Any]:
    reports = {}
    for name, path in paths.items():
        if path.suffix == ".gz":
            reports[name] = inspect_gzip_csv(path)
        else:
            reports[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "raw_sha256": sha256_path(path),
            }
    return reports


def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.time()
    validate_scientific_contract(config, require_kaggle_approval=True)
    artifacts = artifact_dir()
    preflight = preflight_inputs(config)

    raw_manifest = raw_well_identity_manifest(
        preflight["paths"]["raw_dir"], preflight["wells"]
    )
    expected_identity_sha = str(
        get_nested(config, "data.expected_raw_well_identity_sha256")
    )
    raw_identity_sha = dataframe_content_sha(
        raw_manifest,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    if raw_identity_sha != expected_identity_sha:
        raise ValueError("raw well identity manifest SHA mismatch")

    scientific_contract = build_scientific_contract(config)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, scientific_contract)
    raw_manifest_path = artifacts / f"{OUTPUT_PREFIX}_raw_well_manifest.csv"
    raw_manifest.to_csv(raw_manifest_path, index=False)

    target_free_blocks, prefix_nll, frozen, frozen_paths = freeze_target_free_blocks(
        preflight, config, artifacts
    )
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "raw_well_identity_content_sha256": raw_identity_sha,
        "input_reports": preflight["input_reports"],
        "scientific_contract_content_sha256": scientific_contract["content_sha256"],
        "target_free_freeze": frozen["freeze"],
    }
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.json"
    write_json(input_manifest_path, input_manifest)

    # The first parsing of unknown-suffix truth and hidden-like roles happens here.
    row_readout, late_attachment = load_late_row_readout(
        preflight, frozen, config
    )
    block_readout = attach_block_truth(target_free_blocks, row_readout, config)
    scope_metrics = build_scope_metrics(block_readout, config)
    gate = evaluate_stage_0_gate(
        block_readout, prefix_nll, scope_metrics, frozen, preflight, config
    )

    block_readout_path = artifacts / f"{OUTPUT_PREFIX}_late_truth_block_readout.csv.gz"
    scope_metrics_path = artifacts / f"{OUTPUT_PREFIX}_scope_metrics.csv"
    gate_path = artifacts / f"{OUTPUT_PREFIX}_stage0_gate.json"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    block_report = write_deterministic_gzip_csv(block_readout, block_readout_path)
    scope_metrics.to_csv(scope_metrics_path, index=False)
    write_json(gate_path, gate)

    output_paths = {
        **frozen_paths,
        "scientific_contract": contract_path,
        "raw_well_manifest": raw_manifest_path,
        "input_manifest": input_manifest_path,
        "late_truth_block_readout": block_readout_path,
        "scope_metrics": scope_metrics_path,
        "stage0_gate": gate_path,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0_pass_wait_for_separate_stage_1_approval"
            if gate["passed"]
            else "stage_0_failed_close_without_rescue"
        ),
        "route": "pf_beam",
        "stage": "stage_0",
        "rows": len(row_readout),
        "wells": int(row_readout["well_id"].nunique()),
        "blocks": len(block_readout),
        "known_prefix_heldout_rows": len(prefix_nll),
        "execution_contract": get_nested(config, "execution_contract"),
        "scientific_contract": scientific_contract,
        "input_manifest": input_manifest,
        "late_truth_attachment": late_attachment,
        "block_readout_report": block_report,
        "scope_metrics": scope_metrics.to_dict(orient="records"),
        "gate": gate,
        "runtime": runtime_versions(),
        "elapsed_seconds": float(time.time() - started),
        "outputs": {name: path.name for name, path in output_paths.items()},
    }
    summary["output_reports"] = output_file_reports(output_paths)
    write_json(summary_path, summary)
    output_paths["summary"] = summary_path

    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"]
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": summary["status"],
        "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "bad10_auc",
        "stage_0": {
            "overall": overall.iloc[0].to_dict(),
            "known_prefix": {
                "rows": len(prefix_nll),
                "wells": int(prefix_nll["well_id"].nunique()),
                "base_predictive_nll": float(prefix_nll["base_nll"].sum()),
                "marginalized_predictive_nll": float(
                    prefix_nll["marginal_nll"].sum()
                ),
                "predictive_nll_gain_fraction": gate["scientific"][
                    "known_prefix_predictive_nll_gain_fraction"
                ],
            },
            "gate": gate,
            "blocks": len(block_readout),
            "wells": int(block_readout["well_id"].nunique()),
        },
        "reproducibility": {
            "scientific_contract_content_sha256": scientific_contract[
                "content_sha256"
            ],
            "block_ledger_content_sha256": frozen["freeze"][
                "block_ledger_content_sha256"
            ],
            "weak_posterior_content_sha256": frozen["freeze"][
                "weak_posterior_content_sha256"
            ],
            "known_prefix_nll_content_sha256": frozen["freeze"][
                "known_prefix_nll_content_sha256"
            ],
            "late_truth_block_readout_content_sha256": block_report[
                "content_sha256"
            ],
        },
        "notes": (
            "Stage 0 only; no PF replay, model training, inference, blend, "
            "or submission is produced."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(overall.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG, require_kaggle_approval=False)

if EXECUTE_NOTEBOOK:
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "status": get_nested(CONFIG, "experiment.status"),
            "active_stage": get_nested(CONFIG, "execution.active_stage"),
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "kaggle_push_approved": get_nested(
                CONFIG, "execution.kaggle_push_approved"
            ),
            "known_prefix_history_rows": get_nested(
                CONFIG, "validation.stage_0.known_prefix.history_rows"
            ),
            "known_prefix_heldout_rows": get_nested(
                CONFIG, "validation.stage_0.known_prefix.heldout_rows"
            ),
            "block_rows": get_nested(
                CONFIG, "validation.stage_0.suffix.block_rows"
            ),
            "stride_rows": get_nested(
                CONFIG, "validation.stage_0.suffix.stride_rows"
            ),
            "stage_0_execution_contract": get_nested(
                CONFIG, "execution_contract.stage_0"
            ),
            "stage_1_implemented": False,
            "stage_1_enabled": False,
            "inference_enabled": False,
            "submission_enabled": False,
        }
    )


# %% [markdown]
# ## 10. Run the approved Kaggle CPU Stage 0


# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_stage_0")):
        SUMMARY = run_full_experiment(CONFIG)
    else:
        print(
            "exp368 Stage 0 implementation is ready, but execution.run_stage_0=false. "
            "No Kaggle diagnostic, PF replay, inference, or submission was run."
        )
