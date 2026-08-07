# %% [markdown]
# # exp358 exp209 missing-distance emission downweight train
#
# Stage 0 is a deterministic, zero-HMM technical audit. It freezes the raw-GR
# missing mask, distance to the nearest raw finite GR row, the one fixed
# confidence weight, and the exp209-interpolated GR surface without reading
# unknown-suffix TVT truth. Stage 1 remains unimplemented and requires a
# separate user approval even when every Stage 0 gate passes.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and raw-input preflight
# 4. Missing-distance and exp209 interpolation helpers
# 5. Target-free per-well weight-surface generation
# 6. Stage 0 technical gate and generated artifacts
# 7. Setup and configuration preview
# 8. Run the approved Stage 0 audit only

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp358_exp209_missing_distance_emission_downweight"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SURFACE_ID_COLUMNS = ["well_id", "row_idx", "suffix_offset"]
SURFACE_FREEZE_COLUMNS = [
    *SURFACE_ID_COLUMNS,
    "raw_gr_finite",
    "raw_gr_missing",
    "nearest_finite_row_distance",
    "confidence_weight",
    "interpolated_gr",
    "missing_run_length",
    "gap_bucket",
    "all_missing_well_fallback",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP358_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")


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
    raise FileNotFoundError(f"exp358 config not found in {[str(path) for path in candidates]}")


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


def train_data_dir(config: Mapping[str, Any]) -> Path:
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
    raise FileNotFoundError(
        "raw train directory not found; checked=" + str([str(path) for path in candidates])
    )


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
    payload = json.dumps(to_jsonable(dict(value)), sort_keys=True, separators=(",", ":"))
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
    schema = {
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }
    return mapping_sha256(schema)


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    return inspect_gzip_csv(path)


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


# %% [markdown]
# ## 3. Frozen scientific contract and raw-input preflight


# %%
def stage0_execution_counts() -> dict[str, Any]:
    return {
        "stage_0_technical_audits": 1,
        "reporting_folds": 0,
        "stage_0_hmm_well_runs": 0,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "conditional_stage_1_hmm_well_runs": 773,
    }


def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> dict[str, Any]:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "implementation.enabled": True,
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": False,
        "implementation.canonical_notebook_adopted": True,
        "model.missing_weight.raw_finite_observed": 1.0,
        "model.missing_weight.missing_formula": (
            "max_floor_two_to_negative_distance_over_half_life"
        ),
        "model.missing_weight.distance_unit": "rows_to_nearest_raw_finite_gr",
        "model.missing_weight.half_life_rows": 8,
        "model.missing_weight.floor": 0.25,
        "model.missing_weight.all_missing_well_fallback": 0.25,
        "model.missing_weight.application": "multiply_gaussian_log_emission_exactly_once",
        "model.stage_0.audit": "full_missing_weight_surface_contract",
        "model.stage_0.pass_requires_all.expected_wells": 773,
        "model.stage_0.pass_requires_all.expected_rows": 3783989,
        "model.stage_0.pass_requires_all.require_all_finite": True,
        "model.stage_0.pass_requires_all.require_observed_weight_exact_one": True,
        "model.stage_0.pass_requires_all.minimum_missing_weight": 0.25,
        "model.stage_0.pass_requires_all.require_missing_weight_strictly_below_one": True,
        "model.stage_0.pass_requires_all.require_missing_rows": True,
        "model.stage_0.pass_requires_all.require_missing_weight_above_floor": True,
        "model.stage_0.pass_requires_all.require_multiple_missing_weight_values": True,
        "model.stage_0.pass_requires_all.require_all_missing_well_fallback_exact_floor": True,
        "model.stage_1.enabled_condition": ("stage_0_all_gates_pass_and_separate_user_approval"),
        "model.stage_1.hmm_well_runs": 773,
        "execution_contract.stage_0.technical_audits": 1,
        "execution_contract.stage_0.reporting_folds": 0,
        "execution_contract.stage_0.hmm_well_runs": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.stage_1_if_pass.scientific_variants": 1,
        "execution_contract.stage_1_if_pass.reporting_folds": 5,
        "execution_contract.stage_1_if_pass.hmm_well_runs": 773,
        "execution_contract.stage_1_if_pass.model_configs": 0,
        "execution_contract.stage_1_if_pass.trained_folds": 0,
        "execution_contract.stage_1_if_pass.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
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
            raise ValueError(f"exp358 fixed contract mismatch: {key} must be {value!r}")

    fixed_parent = get_nested(config, "model.fixed_parent") or {}
    expected_parent = {
        "interpolation": "exact_exp209",
        "observation_sigma": "exact_exp209_zero_filled_residual",
        "typewell": "exact_exp209",
        "sig_r": 0.002,
        "sig_p": 0.02,
        "effective_position_sigma_floor": 0.1225,
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "momentum": 0.998,
        "prior": "exact_exp209",
        "output": "posterior_mean",
    }
    for key, value in expected_parent.items():
        actual = fixed_parent.get(key)
        if isinstance(value, float):
            if float(actual) != value:
                raise ValueError(f"exp358 fixes model.fixed_parent.{key}={value}")
        elif actual != value:
            raise ValueError(f"exp358 fixes model.fixed_parent.{key}={value!r}")

    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise RuntimeError("exp358 Stage 0 Kaggle package/push/run is not approved")

    return stage0_execution_counts()


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    counts = validate_scientific_contract(config)
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage_0_target_free_weight_surface_technical_audit",
        "truth_attached": False,
        "parent": get_nested(config, "lineage.parent"),
        "missing_weight": get_nested(config, "model.missing_weight"),
        "fixed_parent": get_nested(config, "model.fixed_parent"),
        "stage_0_gate": get_nested(config, "model.stage_0.pass_requires_all"),
        "execution_counts": counts,
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "forbidden": get_nested(config, "model.forbidden"),
        "stage_1": {
            "implemented": False,
            "execution_approved": False,
            "separate_user_approval_required": True,
            "reserved_hmm_well_runs": 773,
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_raw_well_identity(
    config: Mapping[str, Any], raw_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    content_sha = dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != expected_wells:
        raise ValueError(f"raw well count mismatch: expected={expected_wells}, actual={len(frame)}")
    if content_sha != expected_sha:
        raise ValueError(
            f"raw well identity SHA mismatch: expected={expected_sha}, actual={content_sha}"
        )
    report = {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": content_sha,
        "truth_columns_read": [],
    }
    return frame, report


# %% [markdown]
# ## 4. Missing-distance and exp209 interpolation helpers


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
        raise ValueError("exp358 forbids a half-life/floor grid")

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

    if finite.any() and not np.array_equal(
        weight[finite], np.ones(int(finite.sum()), dtype=np.float64)
    ):
        raise AssertionError("raw-finite GR rows must have exact confidence 1")
    if missing.any() and not no_finite_fallback:
        missing_weight = weight[missing]
        if not bool(((missing_weight >= minimum_weight) & (missing_weight < 1.0)).all()):
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
    fallback = float(np.nanmean(np.asarray(typewell_gr, dtype=np.float64)))
    if not math.isfinite(fallback):
        raise ValueError("exp358 requires a finite Type Well GR fallback mean")
    interpolated = (
        pd.Series(np.asarray(raw_gr, dtype=np.float64))
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(fallback)
        .to_numpy(np.float64)
    )
    if not np.isfinite(interpolated).all():
        raise ValueError("exp209-compatible interpolated GR must be finite")
    return interpolated


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["GR", "TVT_input"])
    forbidden = {"TVT", "error", "abs_error", "formation"}
    if forbidden.intersection(frame.columns):
        raise RuntimeError("truth/error columns entered the target-free horizontal frame")
    return frame.reset_index(drop=True)


def load_typewell_gr(well: str, raw_dir: Path) -> np.ndarray:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.sort_values("TVT", kind="mergesort")
    values = frame["GR"].ffill().bfill().to_numpy(np.float64)
    if not np.isfinite(values).any():
        raise ValueError(f"Type Well GR has no finite value for well={well}")
    return values


# %% [markdown]
# ## 5. Target-free per-well weight-surface generation


# %%
def build_well_weight_surface(
    well: str,
    horizontal: pd.DataFrame,
    typewell_gr: np.ndarray,
    *,
    half_life_rows: float = 8.0,
    minimum_weight: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    if np.isinf(tvt_input).any():
        raise ValueError("TVT_input may contain finite known values or NaN only")

    known_mask = np.isfinite(tvt_input)
    score_mask = ~known_mask
    known_index = np.flatnonzero(known_mask)
    score_index = np.flatnonzero(score_mask)
    if len(known_index) == 0 or len(score_index) == 0:
        raise ValueError(f"well={well} must contain a known prefix and unknown suffix")
    if int(known_index[-1]) >= int(score_index[0]) or not bool(
        score_mask[int(score_index[0]) :].all()
    ):
        raise ValueError(f"well={well} TVT_input unknown rows must form one suffix")

    confidence = build_missing_distance_confidence(
        raw_gr,
        half_life_rows=half_life_rows,
        minimum_weight=minimum_weight,
    )
    interpolated_gr = parent_interpolated_gr(raw_gr, typewell_gr)
    suffix_offset = np.arange(len(score_index), dtype=np.int32)
    fallback = bool(confidence["no_finite_gr_fallback"])
    surface = pd.DataFrame(
        {
            "well_id": well,
            "row_idx": score_index.astype(np.int32),
            "suffix_offset": suffix_offset,
            "raw_gr_finite": confidence["raw_gr_finite"][score_index].astype(bool),
            "raw_gr_missing": confidence["raw_gr_missing"][score_index].astype(bool),
            "nearest_finite_row_distance": confidence["nearest_finite_row_distance"][
                score_index
            ].astype(np.int32),
            "confidence_weight": confidence["confidence_weight"][score_index].astype(np.float64),
            "interpolated_gr": interpolated_gr[score_index].astype(np.float64),
            "missing_run_length": confidence["missing_run_length"][score_index].astype(np.int32),
            "gap_bucket": confidence["gap_bucket"][score_index].astype(str),
            "all_missing_well_fallback": fallback,
        }
    )
    summary = {
        "well_id": well,
        "total_rows": len(horizontal),
        "known_prefix_rows": len(known_index),
        "score_rows": len(surface),
        "raw_finite_score_rows": int(surface["raw_gr_finite"].sum()),
        "raw_missing_score_rows": int(surface["raw_gr_missing"].sum()),
        "missing_weight_min": (
            float(surface.loc[surface["raw_gr_missing"], "confidence_weight"].min())
            if bool(surface["raw_gr_missing"].any())
            else None
        ),
        "missing_weight_max": (
            float(surface.loc[surface["raw_gr_missing"], "confidence_weight"].max())
            if bool(surface["raw_gr_missing"].any())
            else None
        ),
        "all_missing_well_fallback": fallback,
        "truth_columns_read": 0,
    }
    return surface, summary


def generate_and_freeze_weight_surface(
    raw_identity: pd.DataFrame,
    raw_dir: Path,
    artifacts: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    weight_config = get_nested(config, "model.missing_weight") or {}
    surfaces: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for well in raw_identity["well_id"].astype(str):
        horizontal = load_horizontal_without_truth(well, raw_dir)
        typewell_gr = load_typewell_gr(well, raw_dir)
        surface, summary = build_well_weight_surface(
            well,
            horizontal,
            typewell_gr,
            half_life_rows=float(weight_config["half_life_rows"]),
            minimum_weight=float(weight_config["floor"]),
        )
        surfaces.append(surface)
        summaries.append(summary)

    surface = (
        pd.concat(surfaces, ignore_index=True)
        .sort_values(SURFACE_ID_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )
    per_well = (
        pd.DataFrame(summaries).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    )
    if surface.duplicated(SURFACE_ID_COLUMNS).any():
        raise ValueError("duplicate exp358 weight-surface row identity")

    paths = {
        "weight_surface": artifacts / f"{OUTPUT_PREFIX}_weight_surface.csv.gz",
        "per_well_summary": artifacts / f"{OUTPUT_PREFIX}_per_well_summary.csv",
        "freeze_manifest": artifacts / f"{OUTPUT_PREFIX}_freeze_manifest.json",
    }
    gzip_report = write_gzip_csv(paths["weight_surface"], surface)
    per_well.to_csv(paths["per_well_summary"], index=False)
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "truth_attached": False,
        "truth_columns_read": [],
        "rows": len(surface),
        "wells": int(surface["well_id"].nunique()),
        "surface": gzip_report,
        "surface_schema_sha256": dataframe_schema_sha(surface),
        "surface_logical_sha256": dataframe_content_sha(surface, SURFACE_FREEZE_COLUMNS),
        "raw_missing_mask_logical_sha256": dataframe_content_sha(
            surface, [*SURFACE_ID_COLUMNS, "raw_gr_missing"]
        ),
        "nearest_finite_distance_logical_sha256": dataframe_content_sha(
            surface, [*SURFACE_ID_COLUMNS, "nearest_finite_row_distance"]
        ),
        "confidence_weight_logical_sha256": dataframe_content_sha(
            surface, [*SURFACE_ID_COLUMNS, "confidence_weight"]
        ),
        "interpolated_gr_logical_sha256": dataframe_content_sha(
            surface, [*SURFACE_ID_COLUMNS, "interpolated_gr"]
        ),
        "per_well_summary_logical_sha256": dataframe_content_sha(per_well),
        "frozen_before_truth_attachment": True,
        "stage_1_prediction_created": False,
    }
    freeze_manifest["freeze_manifest_logical_sha256"] = mapping_sha256(freeze_manifest)
    write_json(paths["freeze_manifest"], freeze_manifest)
    return surface, per_well, freeze_manifest, paths


# %% [markdown]
# ## 6. Stage 0 technical gate and generated artifacts


# %%
def evaluate_stage0_gate(
    surface: pd.DataFrame,
    per_well: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    runtime_seconds: float,
    raw_preflight_passed: bool = True,
) -> dict[str, Any]:
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    floor = float(get_nested(config, "model.missing_weight.floor"))
    half_life = float(get_nested(config, "model.missing_weight.half_life_rows"))
    observed = surface.loc[surface["raw_gr_finite"].astype(bool), "confidence_weight"].to_numpy(
        np.float64
    )
    missing_frame = surface.loc[surface["raw_gr_missing"].astype(bool)]
    missing_weight = missing_frame["confidence_weight"].to_numpy(np.float64)
    missing_distance = missing_frame["nearest_finite_row_distance"].to_numpy(np.int64)
    expected_missing_weight = np.where(
        missing_distance < 0,
        floor,
        np.maximum(floor, np.exp2(-missing_distance.astype(np.float64) / half_life)),
    )
    all_missing = surface["all_missing_well_fallback"].astype(bool).to_numpy()
    all_missing_weight = surface.loc[all_missing, "confidence_weight"].to_numpy(np.float64)
    finite_matrix = surface[
        ["nearest_finite_row_distance", "confidence_weight", "interpolated_gr"]
    ].to_numpy(np.float64)
    unique_missing_weights = np.unique(missing_weight)

    checks = {
        "raw_input_preflight_passed": bool(raw_preflight_passed),
        "expected_rows": len(surface) == expected_rows,
        "expected_wells": int(surface["well_id"].nunique()) == expected_wells,
        "per_well_summary_wells": len(per_well) == expected_wells,
        "unique_row_identity": not bool(surface.duplicated(SURFACE_ID_COLUMNS).any()),
        "raw_mask_partition": bool(
            np.logical_xor(
                surface["raw_gr_finite"].to_numpy(bool),
                surface["raw_gr_missing"].to_numpy(bool),
            ).all()
        ),
        "all_surface_values_finite": bool(np.isfinite(finite_matrix).all()),
        "interpolated_gr_finite": bool(
            np.isfinite(surface["interpolated_gr"].to_numpy(np.float64)).all()
        ),
        "observed_rows_present": len(observed) > 0,
        "observed_weight_exact_one": bool(
            len(observed) > 0 and np.array_equal(observed, np.ones(len(observed), dtype=np.float64))
        ),
        "missing_rows_present": len(missing_weight) > 0,
        "missing_weight_formula_exact": bool(
            len(missing_weight) > 0 and np.array_equal(missing_weight, expected_missing_weight)
        ),
        "missing_weight_in_closed_open_range": bool(
            len(missing_weight) > 0 and ((missing_weight >= floor) & (missing_weight < 1.0)).all()
        ),
        "missing_weight_has_above_floor_values": bool(
            len(missing_weight) > 0 and (missing_weight > floor).any()
        ),
        "missing_weight_has_multiple_values": len(unique_missing_weights) >= 2,
        "all_missing_well_fallback_exact_floor": bool(
            len(all_missing_weight) == 0
            or np.array_equal(
                all_missing_weight,
                np.full(len(all_missing_weight), floor, dtype=np.float64),
            )
        ),
        "frozen_before_truth_attachment": bool(
            freeze_manifest.get("frozen_before_truth_attachment")
        )
        and not bool(freeze_manifest.get("truth_attached")),
        "truth_columns_read_zero": freeze_manifest.get("truth_columns_read") == [],
        "hmm_well_runs_zero": True,
        "model_configs_zero": True,
        "trained_folds_zero": True,
        "boosters_zero": True,
        "parent_control_reruns_zero": True,
    }
    passed = bool(all(checks.values()))
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "passed": passed,
        "decision": (
            "stage_0_technical_pass_awaiting_separate_stage_1_approval"
            if passed
            else "stage_0_technical_fail_close_without_rescue"
        ),
        "checks": checks,
        "rows": len(surface),
        "wells": int(surface["well_id"].nunique()),
        "observed_rows": len(observed),
        "missing_rows": len(missing_weight),
        "missing_fraction": (float(len(missing_weight) / len(surface)) if len(surface) else None),
        "missing_weight_min": (float(missing_weight.min()) if len(missing_weight) else None),
        "missing_weight_max": (float(missing_weight.max()) if len(missing_weight) else None),
        "missing_weight_unique_count": len(unique_missing_weights),
        "missing_weight_above_floor_rows": int((missing_weight > floor).sum()),
        "all_missing_wells": int(per_well["all_missing_well_fallback"].astype(bool).sum()),
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.kaggle.runtime_limit_seconds")),
        "execution_counts": stage0_execution_counts(),
        "stage_1_technical_eligibility": passed,
        "stage_1_implemented": False,
        "stage_1_execution_approved": False,
        "stage_1_requires_separate_user_approval": True,
        "failure_action": ("close_without_half_life_floor_grid_hard_mask_sigma_change_or_hmm_run"),
    }


def file_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp358 Stage 0 must run first on Kaggle; "
            "local execution requires explicit smoke approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_identity, raw_report = validate_raw_well_identity(config, raw_dir)

    scientific_contract = build_scientific_contract(config)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    raw_identity_path = artifacts / f"{OUTPUT_PREFIX}_raw_well_identity.csv"
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.json"
    raw_identity.to_csv(raw_identity_path, index=False)
    write_json(contract_path, scientific_contract)
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "truth_attached": False,
        "raw_train": raw_report,
        "raw_identity_logical_sha256": dataframe_content_sha(raw_identity),
        "parent_control_usage": "not_loaded_stage_0_zero_hmm",
        "fold_assignment_usage": "not_loaded_stage_0_zero_reporting_folds",
        "hidden_like_usage": "not_loaded_stage_0_zero_reporting_folds",
    }
    input_manifest["input_manifest_logical_sha256"] = mapping_sha256(input_manifest)
    write_json(input_manifest_path, input_manifest)

    surface, per_well, freeze_manifest, output_paths = generate_and_freeze_weight_surface(
        raw_identity,
        raw_dir,
        artifacts,
        config,
    )
    runtime_seconds = time.time() - started
    gate = evaluate_stage0_gate(
        surface,
        per_well,
        freeze_manifest,
        config,
        runtime_seconds=runtime_seconds,
    )
    gate_path = artifacts / f"{OUTPUT_PREFIX}_stage0_gate.json"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_stage0_summary.json"
    write_json(gate_path, gate)
    status = (
        "stage_0_technical_pass_awaiting_separate_stage_1_approval"
        if gate["passed"]
        else "stage_0_technical_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": "stage_0",
        "runtime_seconds": runtime_seconds,
        "rows": len(surface),
        "wells": int(surface["well_id"].nunique()),
        "technical_audits": 1,
        "reporting_folds": 0,
        "hmm_well_runs": 0,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "freeze_manifest": freeze_manifest,
        "stage0_gate": gate,
        "stage1": {
            "implemented": False,
            "execution_approved": False,
            "separate_user_approval_required": True,
        },
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "prediction_sha256": None,
        "model_sha256": None,
        "submission_sha256": None,
    }
    generated_paths = {
        "scientific_contract": contract_path,
        "raw_well_identity": raw_identity_path,
        "input_manifest": input_manifest_path,
        **output_paths,
        "stage0_gate": gate_path,
    }
    summary["generated_files"] = {name: file_report(path) for name, path in generated_paths.items()}
    write_json(summary_path, summary)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "technical_contract",
        "stage0_gate": gate,
        "weight_surface_sha256": freeze_manifest["surface"]["decompressed_sha256"],
        "weight_surface_logical_sha256": freeze_manifest["surface_logical_sha256"],
        "prediction_sha256": None,
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Stage 0 target-free weight-surface audit only; no HMM, "
            "truth readout, inference, or submission."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 7. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    COUNTS = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "active_stage": get_nested(CONFIG, "execution.active_stage"),
                "stage_0_counts": COUNTS,
                "weight": get_nested(CONFIG, "model.missing_weight"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
                "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 8. Run the approved Stage 0 audit only


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_stage0(CONFIG)
