# %% [markdown]
# # exp355 exp226 dip-rate prior on exp209 — Stage 0 train-side readout
#
# This compact self-contained notebook candidate tests whether the fold-safe
# exp226 K16 geometry field identifies *changes* in dip rate before any HMM is
# run.  It never trains a model, reruns the exp209 control, or creates a
# submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Input and dependency preflight
# 5. K16 geometry rate-schedule construction
# 6. Truth late-join and Stage 0 readout helpers
# 7. Stage 0 gate
# 8. Setup, orchestration, metrics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_NAME = "exp355_exp226_dip_rate_prior_on_exp209"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SAFE_GEOMETRY_COLUMNS = ["well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"]
SCHEDULE_CONTENT_COLUMNS = [
    "well_id",
    "row_idx",
    "suffix_offset",
    "fold",
    "segment_id",
    "md",
    "z",
    "delta_md",
    "md_since",
    "tvt_geop",
    "parent_initial_rate",
    "geometry_segment_rate",
    "geometry_delta_rate",
    "mu_rate",
    "baseline_path_tvt",
    "candidate_path_tvt",
    "geometry_fallback",
    "anchor_u",
]


# %% [markdown]
# ## 2. Notebook-safe runtime, configuration, and SHA helpers

# %%
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


def resolve_config_path() -> Path:
    root = project_root()
    candidates = [
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"exp355 config.yaml was not found: {candidates}")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def read_config() -> dict[str, Any]:
    config = read_yaml(resolve_config_path())
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("resolved config does not belong to exp355")
    return config


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
        return to_jsonable(value.tolist())
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(source, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "raw_sha256": sha256_path(source),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
    }


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in selected:
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


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "raw_sha256": sha256_path(path)}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "content_sha256": dataframe_content_sha256(frame),
        "schema_sha256": dataframe_schema_sha256(frame),
    }


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    report = inspect_gzip_csv(path)
    report["rows"] = len(frame)
    report["logical_content_sha256"] = dataframe_content_sha256(frame)
    report["schema_sha256"] = dataframe_schema_sha256(frame)
    return report


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw_candidate in candidates:
        candidate = Path(str(raw_candidate))
        if candidate.is_file():
            checked.append(str(candidate))
            return candidate
        possible = [
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            PACKAGE_DIR / candidate
            if candidate.name == filename
            else PACKAGE_DIR / candidate / filename,
        ]
        for path in possible:
            checked.append(str(path))
            if path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.is_dir():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.is_dir():
        fixed = [
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT
            / "competitions"
            / "rogii-wellbore-geology-prediction"
            / "train",
        ]
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    return configured if configured.is_absolute() else project_root() / configured


def output_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.is_dir()
        else project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": str(getattr(yaml, "__version__", "unknown")),
    }


# %% [markdown]
# ## 3. Frozen scientific and execution contract

# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "experiment.status": "stage1_completed_scientific_gate_failed_closed",
        "lineage.parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "implementation.enabled": True,
        "implementation.scope": "stage_0_completed_stage_1_exact_hmm",
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": True,
        "implementation.canonical_notebook_adopted": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.truth_attachment": (
            "after_geometry_ledger_schedule_prediction_and_content_sha_freeze"
        ),
        "model.stage_0.exp226_component": "geometry_only_pre_gr_pre_u_projection",
        "model.stage_0.k_segments": 16,
        "model.stage_0.geometry_rate": "delta_tvt_geop_plus_delta_z_div_delta_md",
        "model.stage_0.target_anchor_rate": "exp209_known_prefix_initial_rate",
        "model.stage_0.formula": (
            "parent_initial_rate_plus_geometry_rate_minus_first_segment_geometry_rate"
        ),
        "model.stage_0.fallback": "constant_parent_initial_rate_when_geometry_invalid",
        "model.stage_0.segment_reducer": "median_finite_positive_delta_md_steps",
        "model.stage_0.fold_gate_scope": "segment_and_cumulative_path_both",
        "model.stage_0.compare_against": "exp209_constant_rate_prior",
        "model.stage_1.enabled_condition": (
            "explicit_user_override_after_stage0_average_improvement"
        ),
        "model.stage_1.active_variants": [
            "exp226_geometry_rate_prior_mean_residual_hmm"
        ],
        "model.stage_1.hmm_well_runs": 773,
        "model.fixed_parent.observation_model": "exact_exp209",
        "model.fixed_parent.sigma_gr": "exact_exp209",
        "model.fixed_parent.sig_r": 0.002,
        "model.fixed_parent.sig_p": 0.02,
        "model.fixed_parent.effective_position_sigma_floor": 0.1225,
        "model.fixed_parent.step": 0.35,
        "model.fixed_parent.n_rates": 41,
        "model.fixed_parent.rate_span": 0.10,
        "model.fixed_parent.momentum": 0.998,
        "model.fixed_parent.prior": "exact_exp209",
        "model.fixed_parent.output": "posterior_mean",
        "data.exp226_oof.allowed_columns": SAFE_GEOMETRY_COLUMNS,
        "execution_contract.stage_0.diagnostic_variants": 1,
        "execution_contract.stage_0.reporting_folds": 5,
        "execution_contract.stage_0.hmm_well_runs": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.stage_1_if_pass.scientific_variants": 1,
        "execution_contract.stage_1_if_pass.hmm_well_runs": 773,
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "inference.enabled": False,
        "inference.create_submission": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
    }
    for key, expected_value in expected.items():
        actual = get_nested(config, key)
        if actual != expected_value:
            raise ValueError(
                f"exp355 fixed contract mismatch: {key} must be {expected_value!r}, "
                f"got {actual!r}"
            )

    expected_folds = list(range(5))
    if list(get_nested(config, "validation.expected_folds", [])) != expected_folds:
        raise ValueError("exp355 fixes reporting folds to [0, 1, 2, 3, 4]")

    gates = get_nested(config, "model.stage_0.pass_requires_all", {})
    fixed_gates = {
        "minimum_segment_rate_change_rmse_gain_fraction": 0.05,
        "minimum_cumulative_path_rmse_gain_ft": 0.05,
        "minimum_improved_folds": 4,
        "require_1000_plus_non_regression": True,
        "require_hidden_like_spatial_non_regression": True,
        "require_hidden_like_typewell_purged_non_regression": True,
        "maximum_worst_well_regression_ft": 0.25,
    }
    if gates != fixed_gates:
        raise ValueError("exp355 Stage 0 gate contract changed")

    forbidden = set(str(value) for value in get_nested(config, "model.forbidden", []))
    required_forbidden = {
        "exp307_or_exp308_or_exp309_or_exp338_inputs",
        "exp226_final_tvt_prediction",
        "exp226_gr_correction",
        "exp226_u_projection",
        "absolute_tvt_unary",
        "fixed_exp226_path_shape",
        "prediction_blend",
        "parameter_grid",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp355 forbidden-input contract is incomplete")

    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise RuntimeError(
            "exp355 Kaggle package/push/run is not approved; "
            "Stage 0 remains fail-closed"
        )

    return {
        "stage_0_variants": 1,
        "reporting_folds": 5,
        "stage_0_hmm_well_runs": 0,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "conditional_stage_1_hmm_well_runs": 773,
    }


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "stage": "stage_0_zero_hmm_identifiability_readout",
        "truth_attached": False,
        "rate_schedule": get_nested(config, "model.stage_0"),
        "fixed_parent": get_nested(config, "model.fixed_parent"),
        "execution_contract": get_nested(config, "execution_contract"),
        "forbidden": get_nested(config, "model.forbidden"),
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Input and dependency preflight

# %%
def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    manifest = (
        pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    )
    actual_sha = dataframe_content_sha256(
        manifest,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(manifest) != expected_wells:
        raise ValueError(f"raw well count mismatch: {len(manifest)} != {expected_wells}")
    if actual_sha != expected_sha:
        raise ValueError("current raw train well-file identity mismatch")
    return manifest, {
        "path": str(raw_dir),
        "wells": len(manifest),
        "content_sha256": actual_sha,
    }


def load_exp226_geometry(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof", {})
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    inspection = inspect_gzip_csv(path)
    if inspection["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")

    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(SAFE_GEOMETRY_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"exp226 OOF is missing safe geometry columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=SAFE_GEOMETRY_COLUMNS,
        dtype={
            "well_id": "string",
            "row_idx": "int32",
            "suffix_offset": "int32",
            "fold": "int8",
            "tvt_geop": "float64",
        },
    )
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 geometry identity contains duplicate well/row pairs")
    if not np.isfinite(frame["tvt_geop"].to_numpy(np.float64)).all():
        raise ValueError("exp226 tvt_geop must be finite")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = set(int(value) for value in get_nested(config, "validation.expected_folds"))
    if len(frame) != expected_rows or frame["well_id"].nunique() != expected_wells:
        raise ValueError("exp226 geometry row/well contract mismatch")
    if set(int(value) for value in frame["fold"].unique()) != expected_folds:
        raise ValueError("exp226 geometry fold contract mismatch")
    if not frame.groupby("well_id", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("each exp226 validation well must belong to exactly one fold")
    inspection.update(
        {
            "rows_loaded": len(frame),
            "wells_loaded": frame["well_id"].nunique(),
            "safe_columns_loaded": list(frame.columns),
            "source_columns": list(header.columns),
            "forbidden_columns_loaded": [],
            "safe_content_sha256": dataframe_content_sha256(frame),
            "safe_schema_sha256": dataframe_schema_sha256(frame),
        }
    )
    return frame, inspection


def validate_exp209_control_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = get_nested(config, "data.exp209_control", {})
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    inspection = inspect_gzip_csv(path)
    expected_sha = str(spec["expected_hmm_cache_decompressed_sha256"])
    if inspection["decompressed_sha256"] != expected_sha:
        raise ValueError("exp209 trusted control cache decompressed SHA mismatch")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if inspection["data_rows"] != expected_rows:
        raise ValueError("exp209 trusted control cache row count mismatch")
    inspection["usage"] = "dependency_sha_guard_only_no_control_rerun"
    return inspection


def load_hidden_like_assignment(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like", {})
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("exp115 hidden-like assignment SHA mismatch")
    role_columns = dict(spec["valid_role_columns"])
    selected = ["well_id", *role_columns.values()]
    frame = pd.read_csv(path, usecols=selected, dtype={"well_id": "string"})
    frame["well_id"] = frame["well_id"].astype(str)
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment must contain one row per well")
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_wells:
        raise ValueError("hidden-like assignment well count mismatch")
    allowed = {
        "verification_like_spatial": {"train", "valid"},
        "verification_like_typewell_purged": {
            "train",
            "valid",
            "purged_train_excluded",
        },
    }
    counts: dict[str, dict[str, int]] = {}
    for scope, column in role_columns.items():
        values = set(frame[column].astype(str).unique())
        if not values.issubset(allowed[scope]):
            raise ValueError(f"unexpected role values for {scope}: {sorted(values)}")
        counts[scope] = {
            str(key): int(value)
            for key, value in frame[column].value_counts().sort_index().items()
        }
    return frame, {
        "path": str(path),
        "raw_sha256": actual_sha,
        "wells": len(frame),
        "role_counts": counts,
    }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    header = pd.read_csv(path, nrows=0)
    required = ["MD", "Z", "TVT_input"]
    missing = sorted(set(required) - set(header.columns))
    if missing:
        raise ValueError(f"{well} horizontal file misses {missing}")
    frame = pd.read_csv(path, usecols=required)
    if "TVT" in frame.columns:
        raise RuntimeError("unknown-suffix truth crossed the pre-freeze boundary")
    return frame


# %% [markdown]
# ## 5. K16 geometry rate-schedule construction

# %%
def exp209_initial_rate(horizontal: pd.DataFrame, tail_n: int = 30) -> dict[str, Any]:
    known = horizontal.loc[horizontal["TVT_input"].notna(), ["MD", "Z", "TVT_input"]]
    tail = known.tail(tail_n)
    dmd = np.diff(tail["MD"].to_numpy(np.float64))
    du = np.diff(
        tail["TVT_input"].to_numpy(np.float64) + tail["Z"].to_numpy(np.float64)
    )
    valid = np.isfinite(dmd) & np.isfinite(du) & (dmd > 0.0)
    rate = float(np.median(du[valid] / dmd[valid])) if valid.sum() >= 3 else 0.0
    return {
        "initial_rate": rate,
        "known_rows": len(known),
        "tail_rows": len(tail),
        "valid_steps": int(valid.sum()),
        "fallback": bool(valid.sum() < 3),
    }


def k16_segment_ids(n_rows: int, k_segments: int = 16) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("K16 segmentation requires at least one row")
    if k_segments <= 0:
        raise ValueError("k_segments must be positive")
    edges = np.linspace(0.0, float(n_rows), k_segments + 1)
    step_idx = np.arange(1.0, n_rows + 1.0)
    return np.clip(
        np.searchsorted(edges[1:], step_idx, side="left"),
        0,
        k_segments - 1,
    ).astype(np.int16)


def segment_step_rates(
    md: np.ndarray,
    u: np.ndarray,
    segment_ids: np.ndarray,
    k_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    md = np.asarray(md, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    segment_ids = np.asarray(segment_ids, dtype=np.int16)
    rates = np.full(k_segments, np.nan, dtype=np.float64)
    counts = np.zeros(k_segments, dtype=np.int32)
    if len(md) < 2:
        return rates, counts
    dmd = np.diff(md)
    du = np.diff(u)
    valid = np.isfinite(dmd) & np.isfinite(du) & (dmd > 0.0)
    step_rate = np.full(len(dmd), np.nan, dtype=np.float64)
    step_rate[valid] = du[valid] / dmd[valid]
    destination_segment = segment_ids[1:]
    for segment_id in range(k_segments):
        selected = step_rate[
            valid & (destination_segment == segment_id) & np.isfinite(step_rate)
        ]
        counts[segment_id] = len(selected)
        if len(selected):
            rates[segment_id] = float(np.median(selected))
    return rates, counts


def _validate_well_alignment(
    well: str,
    geometry: pd.DataFrame,
    horizontal: pd.DataFrame,
) -> np.ndarray:
    row_idx = geometry["row_idx"].to_numpy(np.int64)
    suffix_offset = geometry["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(geometry), dtype=np.int64)):
        raise ValueError(f"{well} suffix_offset is not stable contiguous row order")
    unknown_idx = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    if not np.array_equal(row_idx, unknown_idx):
        raise ValueError(f"{well} exp226 rows do not match the raw unknown suffix")
    if len(row_idx) == 0 or row_idx[0] == 0:
        raise ValueError(f"{well} has no known-prefix anchor")
    if not horizontal.loc[: row_idx[0] - 1, "TVT_input"].notna().all():
        raise ValueError(f"{well} known prefix is not contiguous")
    return row_idx


def build_well_rate_schedule(
    well: str,
    geometry: pd.DataFrame,
    horizontal: pd.DataFrame,
    *,
    k_segments: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    geometry = geometry.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = _validate_well_alignment(well, geometry, horizontal)
    segment_ids = k16_segment_ids(len(geometry), k_segments)
    md = horizontal.loc[row_idx, "MD"].to_numpy(np.float64)
    z = horizontal.loc[row_idx, "Z"].to_numpy(np.float64)
    geop = geometry["tvt_geop"].to_numpy(np.float64)
    if not (
        np.isfinite(md).all() and np.isfinite(z).all() and np.isfinite(geop).all()
    ):
        raise ValueError(f"{well} geometry inputs must be finite")

    rate_audit = exp209_initial_rate(horizontal)
    parent_rate = float(rate_audit["initial_rate"])
    geometry_rate, valid_steps = segment_step_rates(
        md,
        geop + z,
        segment_ids,
        k_segments,
    )
    first_geometry_rate = float(geometry_rate[0])
    first_valid = math.isfinite(first_geometry_rate)
    mu_by_segment = np.full(k_segments, parent_rate, dtype=np.float64)
    delta_by_segment = np.zeros(k_segments, dtype=np.float64)
    fallback_by_segment = np.ones(k_segments, dtype=bool)
    if first_valid:
        valid_segments = np.isfinite(geometry_rate)
        delta_by_segment[valid_segments] = (
            geometry_rate[valid_segments] - first_geometry_rate
        )
        mu_by_segment[valid_segments] = parent_rate + delta_by_segment[valid_segments]
        fallback_by_segment[valid_segments] = False

    anchor_idx = int(row_idx[0] - 1)
    anchor_md = float(horizontal.loc[anchor_idx, "MD"])
    anchor_z = float(horizontal.loc[anchor_idx, "Z"])
    anchor_tvt = float(horizontal.loc[anchor_idx, "TVT_input"])
    if not all(math.isfinite(value) for value in [anchor_md, anchor_z, anchor_tvt]):
        raise ValueError(f"{well} prefix anchor must be finite")
    delta_md = np.diff(np.r_[anchor_md, md])
    if not np.isfinite(delta_md).all() or np.any(delta_md <= 0.0):
        raise ValueError(f"{well} MD must increase strictly across the suffix")
    anchor_u = anchor_tvt + anchor_z
    row_mu = mu_by_segment[segment_ids]
    baseline_u = anchor_u + np.cumsum(parent_rate * delta_md)
    candidate_u = anchor_u + np.cumsum(row_mu * delta_md)

    schedule = pd.DataFrame(
        {
            "well_id": well,
            "row_idx": row_idx.astype(np.int32),
            "suffix_offset": geometry["suffix_offset"].to_numpy(np.int32),
            "fold": geometry["fold"].to_numpy(np.int8),
            "segment_id": segment_ids,
            "md": md,
            "z": z,
            "delta_md": delta_md,
            "md_since": md - anchor_md,
            "tvt_geop": geop,
            "parent_initial_rate": parent_rate,
            "geometry_segment_rate": geometry_rate[segment_ids],
            "geometry_delta_rate": delta_by_segment[segment_ids],
            "mu_rate": row_mu,
            "baseline_path_tvt": baseline_u - z,
            "candidate_path_tvt": candidate_u - z,
            "geometry_fallback": fallback_by_segment[segment_ids],
            "anchor_u": anchor_u,
        }
    )
    segment_ledger = pd.DataFrame(
        {
            "well_id": well,
            "fold": int(geometry["fold"].iloc[0]),
            "segment_id": np.arange(k_segments, dtype=np.int16),
            "row_count": np.bincount(segment_ids, minlength=k_segments).astype(np.int32),
            "valid_geometry_steps": valid_steps,
            "parent_initial_rate": parent_rate,
            "first_segment_geometry_rate": first_geometry_rate,
            "geometry_segment_rate": geometry_rate,
            "geometry_delta_rate": delta_by_segment,
            "mu_rate": mu_by_segment,
            "geometry_fallback": fallback_by_segment,
        }
    )
    fallback = {
        "well_id": well,
        "fold": int(geometry["fold"].iloc[0]),
        "rows": len(schedule),
        "parent_initial_rate": parent_rate,
        "parent_initial_rate_fallback": bool(rate_audit["fallback"]),
        "parent_initial_rate_valid_steps": int(rate_audit["valid_steps"]),
        "first_geometry_segment_valid": first_valid,
        "fallback_segments": int(fallback_by_segment.sum()),
        "fallback_rows": int(fallback_by_segment[segment_ids].sum()),
    }
    return schedule, segment_ledger, fallback


@dataclass(frozen=True)
class FrozenSchedule:
    schedule: pd.DataFrame
    segment_ledger: pd.DataFrame
    fallback_summary: pd.DataFrame
    schedule_content_sha256: str
    segment_ledger_content_sha256: str


def freeze_geometry_rate_schedule(
    geometry: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> FrozenSchedule:
    k_segments = int(get_nested(config, "model.stage_0.k_segments"))
    schedules: list[pd.DataFrame] = []
    ledgers: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    for well, well_geometry in geometry.groupby("well_id", sort=True, observed=True):
        safe_horizontal = load_horizontal_without_truth(str(well), raw_dir)
        schedule, ledger, fallback = build_well_rate_schedule(
            str(well),
            well_geometry,
            safe_horizontal,
            k_segments=k_segments,
        )
        schedules.append(schedule)
        ledgers.append(ledger)
        fallback_rows.append(fallback)
    schedule = pd.concat(schedules, ignore_index=True)
    schedule = schedule.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    segment_ledger = pd.concat(ledgers, ignore_index=True)
    segment_ledger = segment_ledger.sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    fallback_summary = pd.DataFrame(fallback_rows).sort_values(
        "well_id", kind="mergesort"
    ).reset_index(drop=True)

    if any(
        forbidden in schedule.columns
        for forbidden in ["tvt_true", "TVT", "error", "abs_error"]
    ):
        raise RuntimeError("truth entered the frozen geometry schedule")
    if len(schedule) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("frozen schedule row count mismatch")
    schedule_sha = dataframe_content_sha256(schedule, SCHEDULE_CONTENT_COLUMNS)
    ledger_sha = dataframe_content_sha256(segment_ledger)
    return FrozenSchedule(
        schedule=schedule,
        segment_ledger=segment_ledger,
        fallback_summary=fallback_summary,
        schedule_content_sha256=schedule_sha,
        segment_ledger_content_sha256=ledger_sha,
    )


# %% [markdown]
# ## 6. Truth late-join and Stage 0 readout helpers

# %%
def attach_truth_after_freeze(
    frozen: FrozenSchedule,
    raw_dir: Path,
) -> pd.DataFrame:
    actual_sha = dataframe_content_sha256(frozen.schedule, SCHEDULE_CONTENT_COLUMNS)
    if actual_sha != frozen.schedule_content_sha256:
        raise RuntimeError("truth attachment requires an unchanged frozen schedule")
    frame = frozen.schedule.copy()
    truth = np.full(len(frame), np.nan, dtype=np.float64)
    for _, positions in frame.groupby("well_id", sort=True).indices.items():
        integer_positions = np.asarray(positions, dtype=np.int64)
        well = str(frame.loc[integer_positions[0], "well_id"])
        path = raw_dir / f"{well}__horizontal_well.csv"
        horizontal_truth = pd.read_csv(path, usecols=["TVT"])
        row_idx = frame.loc[integer_positions, "row_idx"].to_numpy(np.int64)
        truth[integer_positions] = horizontal_truth.loc[row_idx, "TVT"].to_numpy(
            np.float64
        )
    if not np.isfinite(truth).all():
        raise ValueError("unknown-suffix TVT truth must be finite after late join")
    frame["tvt_true_readout_only"] = truth
    return frame


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(actual_array[valid] - predicted_array[valid]))))


def build_segment_rate_readout(
    readout: pd.DataFrame,
    segment_ledger: pd.DataFrame,
    k_segments: int = 16,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ledger_lookup = segment_ledger.set_index(["well_id", "segment_id"])
    for well, well_frame in readout.groupby("well_id", sort=True, observed=True):
        well_frame = well_frame.sort_values("row_idx", kind="mergesort")
        md = well_frame["md"].to_numpy(np.float64)
        u_true = (
            well_frame["tvt_true_readout_only"].to_numpy(np.float64)
            + well_frame["z"].to_numpy(np.float64)
        )
        segment_ids = well_frame["segment_id"].to_numpy(np.int16)
        actual_rate, actual_steps = segment_step_rates(
            md,
            u_true,
            segment_ids,
            k_segments,
        )
        first_actual = float(actual_rate[0])
        if not math.isfinite(first_actual):
            continue
        fold = int(well_frame["fold"].iloc[0])
        for segment_id in range(1, k_segments):
            if not math.isfinite(float(actual_rate[segment_id])):
                continue
            ledger = ledger_lookup.loc[(str(well), segment_id)]
            predicted_delta = float(ledger["mu_rate"] - ledger["parent_initial_rate"])
            actual_delta = float(actual_rate[segment_id] - first_actual)
            rows.append(
                {
                    "well_id": str(well),
                    "fold": fold,
                    "segment_id": segment_id,
                    "actual_valid_steps": int(actual_steps[segment_id]),
                    "actual_rate_change": actual_delta,
                    "baseline_rate_change": 0.0,
                    "candidate_rate_change": predicted_delta,
                    "geometry_fallback": bool(ledger["geometry_fallback"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("segment rate readout is empty")
    return result.sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)


def _metric_record(
    frame: pd.DataFrame,
    *,
    scope: str,
    scope_value: str,
    actual_column: str,
    baseline_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    baseline_rmse = rmse(frame[actual_column], frame[baseline_column])
    candidate_rmse = rmse(frame[actual_column], frame[candidate_column])
    return {
        "scope": scope,
        "scope_value": scope_value,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_candidate_minus_baseline": candidate_rmse - baseline_rmse,
        "gain_baseline_minus_candidate": baseline_rmse - candidate_rmse,
        "gain_fraction": (
            (baseline_rmse - candidate_rmse) / baseline_rmse
            if baseline_rmse > 0.0
            else float("nan")
        ),
    }


def build_segment_metrics(segment_readout: pd.DataFrame) -> pd.DataFrame:
    records = [
        _metric_record(
            segment_readout,
            scope="overall",
            scope_value="all",
            actual_column="actual_rate_change",
            baseline_column="baseline_rate_change",
            candidate_column="candidate_rate_change",
        )
    ]
    for fold, fold_frame in segment_readout.groupby("fold", sort=True, observed=True):
        records.append(
            _metric_record(
                fold_frame,
                scope="fold",
                scope_value=str(int(fold)),
                actual_column="actual_rate_change",
                baseline_column="baseline_rate_change",
                candidate_column="candidate_rate_change",
            )
        )
    return pd.DataFrame(records)


def build_path_metrics(
    readout: pd.DataFrame,
    hidden_assignment: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_columns = dict(
        get_nested(config, "data.hidden_like.valid_role_columns", {})
    )
    enriched = readout.merge(
        hidden_assignment[["well_id", *role_columns.values()]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if enriched[list(role_columns.values())].isna().any().any():
        raise ValueError("hidden-like roles do not cover every exp355 well")

    kwargs = {
        "actual_column": "tvt_true_readout_only",
        "baseline_column": "baseline_path_tvt",
        "candidate_column": "candidate_path_tvt",
    }
    records = [
        _metric_record(enriched, scope="overall", scope_value="all", **kwargs)
    ]
    for fold, frame in enriched.groupby("fold", sort=True, observed=True):
        records.append(
            _metric_record(
                frame,
                scope="fold",
                scope_value=str(int(fold)),
                **kwargs,
            )
        )
    distance_1000 = enriched.loc[enriched["md_since"] >= 1000.0]
    records.append(
        _metric_record(
            distance_1000,
            scope="distance",
            scope_value="1000_plus",
            **kwargs,
        )
    )
    for scope, column in role_columns.items():
        frame = enriched.loc[enriched[column].astype(str) == "valid"]
        records.append(
            _metric_record(
                frame,
                scope="hidden_like",
                scope_value=scope,
                **kwargs,
            )
        )

    by_well_records: list[dict[str, Any]] = []
    for well, frame in enriched.groupby("well_id", sort=True, observed=True):
        metric = _metric_record(
            frame,
            scope="well",
            scope_value=str(well),
            **kwargs,
        )
        metric["well_id"] = str(well)
        metric["fold"] = int(frame["fold"].iloc[0])
        by_well_records.append(metric)
    return pd.DataFrame(records), pd.DataFrame(by_well_records)


# %% [markdown]
# ## 7. Stage 0 gate

# %%
def evaluate_stage_0_gate(
    segment_metrics: pd.DataFrame,
    path_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = dict(get_nested(config, "model.stage_0.pass_requires_all", {}))
    segment_overall = segment_metrics.loc[segment_metrics["scope"] == "overall"].iloc[0]
    path_overall = path_metrics.loc[path_metrics["scope"] == "overall"].iloc[0]
    segment_folds = segment_metrics.loc[segment_metrics["scope"] == "fold"]
    path_folds = path_metrics.loc[path_metrics["scope"] == "fold"]
    distance_1000 = path_metrics.loc[
        (path_metrics["scope"] == "distance")
        & (path_metrics["scope_value"] == "1000_plus")
    ].iloc[0]
    hidden_spatial = path_metrics.loc[
        (path_metrics["scope"] == "hidden_like")
        & (path_metrics["scope_value"] == "verification_like_spatial")
    ].iloc[0]
    hidden_purged = path_metrics.loc[
        (path_metrics["scope"] == "hidden_like")
        & (path_metrics["scope_value"] == "verification_like_typewell_purged")
    ].iloc[0]

    segment_improved_folds = int(
        (segment_folds["delta_candidate_minus_baseline"] < 0.0).sum()
    )
    path_improved_folds = int(
        (path_folds["delta_candidate_minus_baseline"] < 0.0).sum()
    )
    worst_well = by_well_metrics.sort_values(
        "delta_candidate_minus_baseline",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    checks = {
        "segment_rate_change_gain_fraction": bool(
            float(segment_overall["gain_fraction"])
            >= float(gate["minimum_segment_rate_change_rmse_gain_fraction"])
        ),
        "cumulative_path_gain_ft": bool(
            float(path_overall["gain_baseline_minus_candidate"])
            >= float(gate["minimum_cumulative_path_rmse_gain_ft"])
        ),
        "segment_folds_improved": bool(
            segment_improved_folds >= int(gate["minimum_improved_folds"])
        ),
        "path_folds_improved": bool(
            path_improved_folds >= int(gate["minimum_improved_folds"])
        ),
        "distance_1000_plus_non_regression": bool(
            float(distance_1000["delta_candidate_minus_baseline"]) <= 0.0
        ),
        "hidden_like_spatial_non_regression": bool(
            float(hidden_spatial["delta_candidate_minus_baseline"]) <= 0.0
        ),
        "hidden_like_typewell_purged_non_regression": bool(
            float(hidden_purged["delta_candidate_minus_baseline"]) <= 0.0
        ),
        "worst_well_regression_guard": bool(
            float(worst_well["delta_candidate_minus_baseline"])
            <= float(gate["maximum_worst_well_regression_ft"])
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": gate,
        "segment_rate_change": {
            "baseline_rmse": float(segment_overall["baseline_rmse"]),
            "candidate_rmse": float(segment_overall["candidate_rmse"]),
            "gain_fraction": float(segment_overall["gain_fraction"]),
            "improved_folds": segment_improved_folds,
        },
        "cumulative_path": {
            "baseline_rmse": float(path_overall["baseline_rmse"]),
            "candidate_rmse": float(path_overall["candidate_rmse"]),
            "gain_ft": float(path_overall["gain_baseline_minus_candidate"]),
            "improved_folds": path_improved_folds,
            "distance_1000_plus_delta": float(
                distance_1000["delta_candidate_minus_baseline"]
            ),
            "hidden_like_spatial_delta": float(
                hidden_spatial["delta_candidate_minus_baseline"]
            ),
            "hidden_like_typewell_purged_delta": float(
                hidden_purged["delta_candidate_minus_baseline"]
            ),
            "worst_well_delta": float(worst_well["delta_candidate_minus_baseline"]),
            "worst_well_id": str(worst_well["well_id"]),
        },
        "decision": (
            "stage_0_pass_stage_1_still_requires_separate_user_approval"
            if all(checks.values())
            else "stage_0_failed_close_without_parameter_rescue"
        ),
    }


# %% [markdown]
# ## 8. Setup, orchestration, metrics, and generated artifacts

# %%
def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    contract_counts = validate_scientific_contract(config, require_run_approval=True)
    artifacts = output_dir()
    raw_dir = train_data_dir(config)
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    scientific_contract = build_scientific_contract(config)
    contract_report = write_json(
        artifacts / f"{EXPERIMENT_NAME}_scientific_contract.json",
        scientific_contract,
    )
    raw_manifest, raw_report = validate_raw_well_identity(config, raw_dir)
    geometry, geometry_report = load_exp226_geometry(config)
    exp209_report = validate_exp209_control_dependency(config)
    raw_identity_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_raw_well_identity.csv",
        raw_manifest,
    )

    frozen = freeze_geometry_rate_schedule(geometry, raw_dir, config)
    schedule_report = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_rate_prior_schedule.csv.gz",
        frozen.schedule,
    )
    ledger_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_geometry_segment_ledger.csv",
        frozen.segment_ledger,
    )
    fallback_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_fallback_summary.csv",
        frozen.fallback_summary,
    )
    freeze_manifest = {
        "truth_attached": False,
        "schedule_content_sha256": frozen.schedule_content_sha256,
        "segment_ledger_content_sha256": frozen.segment_ledger_content_sha256,
        "schedule_artifact": schedule_report,
        "segment_ledger_artifact": ledger_report,
        "fallback_artifact": fallback_report,
        "rows": len(frozen.schedule),
        "wells": frozen.schedule["well_id"].nunique(),
        "segments": len(frozen.segment_ledger),
        "fallback_wells": int(
            (~frozen.fallback_summary["first_geometry_segment_valid"]).sum()
        ),
        "fallback_segments": int(
            frozen.segment_ledger["geometry_fallback"].astype(bool).sum()
        ),
    }
    freeze_manifest["freeze_manifest_sha256"] = mapping_sha256(freeze_manifest)
    freeze_report = write_json(
        artifacts / f"{EXPERIMENT_NAME}_freeze_manifest.json",
        freeze_manifest,
    )

    hidden_assignment, hidden_report = load_hidden_like_assignment(config)
    input_manifest = {
        "truth_attached": False,
        "schedule_frozen_before_hidden_like_load": True,
        "frozen_schedule_content_sha256": frozen.schedule_content_sha256,
        "raw_train": raw_report,
        "exp226_geometry": geometry_report,
        "exp209_control": exp209_report,
        "hidden_like_assignment": hidden_report,
    }
    input_manifest["input_manifest_sha256"] = mapping_sha256(input_manifest)
    input_report = write_json(
        artifacts / f"{EXPERIMENT_NAME}_input_manifest.json",
        input_manifest,
    )

    readout = attach_truth_after_freeze(frozen, raw_dir)
    segment_readout = build_segment_rate_readout(
        readout,
        frozen.segment_ledger,
        int(get_nested(config, "model.stage_0.k_segments")),
    )
    segment_metrics = build_segment_metrics(segment_readout)
    path_metrics, by_well_metrics = build_path_metrics(
        readout,
        hidden_assignment,
        config,
    )
    gate = evaluate_stage_0_gate(
        segment_metrics,
        path_metrics,
        by_well_metrics,
        config,
    )

    segment_readout_report = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_segment_rate_readout.csv.gz",
        segment_readout,
    )
    path_readout_report = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_path_readout.csv.gz",
        readout,
    )
    segment_metrics_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_segment_rate_metrics.csv",
        segment_metrics,
    )
    path_metrics_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_path_metrics.csv",
        path_metrics,
    )
    by_well_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_by_well_metrics.csv",
        by_well_metrics,
    )
    gate_report = write_json(
        artifacts / f"{EXPERIMENT_NAME}_stage0_gate.json",
        gate,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed_pass" if gate["passed"] else "stage0_completed_fail",
        "route": "pf_beam",
        "stage": "stage_0_zero_hmm_identifiability_readout",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": elapsed,
        "rows": len(readout),
        "wells": readout["well_id"].nunique(),
        "reporting_folds": sorted(int(value) for value in readout["fold"].unique()),
        "truth_attached_after_freeze": True,
        "gate": gate,
        "execution_counts": contract_counts,
        "runtime": runtime_versions(),
        "sha256": {
            "scientific_contract": scientific_contract[
                "scientific_contract_sha256"
            ],
            "input_manifest": input_manifest["input_manifest_sha256"],
            "frozen_schedule": frozen.schedule_content_sha256,
            "geometry_segment_ledger": frozen.segment_ledger_content_sha256,
            "segment_rate_readout": segment_readout_report[
                "logical_content_sha256"
            ],
            "path_readout": path_readout_report["logical_content_sha256"],
        },
        "generated_artifacts": {
            "scientific_contract": contract_report,
            "input_manifest": input_report,
            "raw_well_identity": raw_identity_report,
            "freeze_manifest": freeze_report,
            "schedule": schedule_report,
            "geometry_segment_ledger": ledger_report,
            "fallback_summary": fallback_report,
            "segment_rate_readout": segment_readout_report,
            "path_readout": path_readout_report,
            "segment_rate_metrics": segment_metrics_report,
            "path_metrics": path_metrics_report,
            "by_well_metrics": by_well_report,
            "stage0_gate": gate_report,
            "summary": {
                "path": str(artifacts / f"{EXPERIMENT_NAME}_summary.json")
            },
        },
        "stage_1_automatically_enabled": False,
        "inference_enabled": False,
        "submission_created": False,
    }
    summary_path = artifacts / f"{EXPERIMENT_NAME}_summary.json"
    write_json(summary_path, summary)
    write_json(metrics_output_path(), summary)
    return summary


# %%
CONFIG = read_config()
CONTRACT_COUNTS = validate_scientific_contract(CONFIG)
print("Experiment:", get_nested(CONFIG, "experiment.name"))
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Stage 0 execution contract:", CONTRACT_COUNTS)
print("Kaggle push approved:", get_nested(CONFIG, "execution.kaggle_push_approved"))
print("Run Stage 0:", get_nested(CONFIG, "execution.run_stage_0"))

if os.environ.get("EXP355_IMPORT_ONLY") != "1":
    STAGE_0_SUMMARY = run_stage_0_experiment(CONFIG)
    print(json.dumps(to_jsonable(STAGE_0_SUMMARY["gate"]), indent=2, sort_keys=True))
