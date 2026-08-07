# %% [markdown]
# # exp355 exp226 dip-rate prior on exp209 — Stage 1 exact HMM

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Raw input and immutable dependency preflight
# 5. Frozen exp226 K16 geometry-rate schedule reconstruction
# 6. Exp209 observation preparation
# 7. Exact residual-rate HMM kernel and decoding
# 8. Prediction freeze and late truth/control attachment
# 9. Metrics, scientific gate, and generated artifacts
# 10. Experiment orchestration
# 11. Setup and configuration preview
# 12. Kaggle CPU execution

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import numba
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - validation-only fallback
    numba = None
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args: Any, **kwargs: Any):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator

    def set_num_threads(value: int) -> None:
        del value


EXPERIMENT_NAME = "exp355_exp226_dip_rate_prior_on_exp209"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT = "exp226_geometry_rate_prior_mean_residual_hmm"
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
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXECUTE_NOTEBOOK = os.environ.get("EXP355_IMPORT_ONLY") != "1"


# %% [markdown]
# ## 2. Notebook-safe runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file() and (candidate / "AGENTS.md").is_file():
            return candidate
    return start


def resolve_config_path() -> Path:
    root = project_root()
    candidates = [
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp355 config.yaml was not found")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def read_config() -> dict[str, Any]:
    return read_yaml(resolve_config_path())


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with gzip.open(resolved, "rt") as stream:
        data_rows = max(sum(1 for _ in stream) - 1, 0)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "data_rows": data_rows,
        "raw_sha256": sha256_path(resolved),
        "decompressed_sha256": sha256_gzip_decompressed(resolved),
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
        digest.update(str(column).encode())
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
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


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
    checked: list[Path] = []
    for raw_candidate in candidates:
        candidate = Path(raw_candidate)
        path = candidate if candidate.name == filename else candidate / filename
        checked.append(path)
        if path.is_file():
            return path
    root = project_root()
    for path in (
        root / filename,
        root / "artifacts" / filename,
        root / "experiments" / filename,
        Path("/tmp") / filename,
    ):
        checked.append(path)
        if path.is_file():
            return path
    if KAGGLE_INPUT_ROOT.is_dir():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{filename} was not found; checked: {[str(path) for path in checked]}")


def train_data_dir(config: Mapping[str, Any]) -> Path:
    local = project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))
    if local.is_dir() and any(local.glob("*__horizontal_well.csv")):
        return local
    if KAGGLE_INPUT_ROOT.is_dir():
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    return local


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        value = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        value = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    value.mkdir(parents=True, exist_ok=True)
    return value


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": getattr(numba, "__version__", None),
        "numba_available": NUMBA_AVAILABLE,
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
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": True,
        "implementation.canonical_notebook_adopted": True,
        "validation.truth_attachment": (
            "after_geometry_ledger_schedule_prediction_and_content_sha_freeze"
        ),
        "model.stage_0.k_segments": 16,
        "model.stage_0.segment_reducer": "median_finite_positive_delta_md_steps",
        "model.stage_1.user_override.approved": True,
        "model.stage_1.user_override.overridden_failed_gate": (
            "stage0_worst_well_regression_guard"
        ),
        "model.stage_1.state_coordinate": "residual_rate_about_time_varying_mu",
        "model.stage_1.prior_mean_schedule": "stage0_frozen_mu_rate_t",
        "model.stage_1.active_variants": [VARIANT],
        "model.stage_1.hmm_well_runs": 773,
        "model.stage_1.hmm.step": 0.35,
        "model.stage_1.hmm.n_rates": 41,
        "model.stage_1.hmm.residual_rate_span": 0.10,
        "model.stage_1.hmm.sig_r": 0.002,
        "model.stage_1.hmm.sig_p": 0.02,
        "model.stage_1.hmm.emission": "gauss",
        "model.stage_1.hmm.sigma_mode": "std",
        "model.stage_1.hmm.start_sig": 0.75,
        "model.stage_1.hmm.r0_sig": 0.01,
        "model.stage_1.hmm.band_pad": 100.0,
        "model.stage_1.hmm.momentum": 0.998,
        "model.stage_1.hmm.output": "posterior_mean",
        "model.fixed_parent.observation_model": "exact_exp209",
        "model.fixed_parent.sig_r": 0.002,
        "model.fixed_parent.sig_p": 0.02,
        "model.fixed_parent.momentum": 0.998,
        "data.exp226_oof.allowed_columns": SAFE_GEOMETRY_COLUMNS,
        "execution_contract.stage_1_if_pass.scientific_variants": 1,
        "execution_contract.stage_1_if_pass.hmm_well_runs": 773,
        "execution_contract.stage_1_if_pass.model_configs": 0,
        "execution_contract.stage_1_if_pass.trained_folds": 0,
        "execution_contract.stage_1_if_pass.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.run_stage_0": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for dotted_key, expected_value in expected.items():
        actual = get_nested(config, dotted_key)
        if actual != expected_value:
            raise ValueError(
                f"frozen contract drifted for {dotted_key}: "
                f"{actual!r} != {expected_value!r}"
            )

    expected_counts = {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu": False,
        "parent_control_retraining": False,
    }
    frozen_values = {
        "model.stage_1.expected_stage0_schedule_logical_sha256": (
            "53f9d42bcca0f5596568971b5da6c440114922d0a25b5622592e1b7b50774c85"
        ),
        "model.stage_1.expected_stage0_geometry_ledger_logical_sha256": (
            "b527d3401e2d730ec883681051c476c929a428e7fc28ed88fff3091045915a39"
        ),
    }
    for dotted_key, expected in frozen_values.items():
        actual = get_nested(config, dotted_key)
        if actual != expected:
            raise ValueError(
                f"frozen contract drifted for {dotted_key}: {actual!r} != {expected!r}"
            )

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
    if not required_forbidden.issubset(set(get_nested(config, "model.forbidden", []))):
        raise ValueError("exp355 forbidden-input contract drifted")

    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp355 Kaggle package/push/run is not approved")
        if not bool(get_nested(config, "execution.run_stage_1")):
            raise RuntimeError("exp355 run_stage_1 must be explicitly enabled")
        if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
            raise RuntimeError("exp355 must run on Kaggle CPU")
    return expected_counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "variant": VARIANT,
        "stage": "stage_1_exact_hmm_user_override_after_stage0_worst_well_fail",
        "change": "constant_rate_mean_to_stage0_frozen_exp226_geometry_schedule",
        "truth_attached": False,
        "stage_0": get_nested(config, "model.stage_0"),
        "stage_1": get_nested(config, "model.stage_1"),
        "fixed_parent": get_nested(config, "model.fixed_parent"),
        "execution_contract": get_nested(config, "execution_contract"),
        "forbidden": get_nested(config, "model.forbidden"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Raw input, parent-control, and fold preflight


# %%
def list_well_ids(raw_dir: Path) -> list[str]:
    wells: list[str] = []
    for path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = path.name.replace("__horizontal_well.csv", "")
        if not (raw_dir / f"{well}__typewell.csv").is_file():
            raise FileNotFoundError(raw_dir / f"{well}__typewell.csv")
        wells.append(well)
    return wells


def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for well in list_well_ids(raw_dir):
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    content_sha = dataframe_content_sha256(
        manifest,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(manifest) != expected_wells:
        raise ValueError(f"raw well count mismatch: {len(manifest)} != {expected_wells}")
    if content_sha != expected_sha:
        raise ValueError("raw train well-file identity mismatch")
    return manifest, {
        "path": str(raw_dir),
        "wells": len(manifest),
        "content_sha256": content_sha,
    }


def load_exp226_geometry(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.exp226_oof", {}))
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    report = inspect_gzip_csv(path)
    if report["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
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
    if (
        len(frame) != int(get_nested(config, "validation.expected_rows"))
        or frame["well_id"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
    ):
        raise ValueError("exp226 geometry row/well contract mismatch")
    if not frame.groupby("well_id", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("each exp226 validation well must belong to exactly one fold")
    report.update(
        {
            "rows_loaded": len(frame),
            "wells_loaded": int(frame["well_id"].nunique()),
            "safe_columns_loaded": list(frame.columns),
            "source_columns": list(header.columns),
            "forbidden_columns_loaded": [],
            "safe_content_sha256": dataframe_content_sha256(frame),
            "safe_schema_sha256": dataframe_schema_sha256(frame),
        }
    )
    return frame, report


def validate_parent_control_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(get_nested(config, "data.exp209_control", {}))
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    report = inspect_gzip_csv(path)
    if report["decompressed_sha256"] != str(
        spec["expected_hmm_cache_decompressed_sha256"]
    ):
        raise ValueError("exp209 trusted control cache decompressed SHA mismatch")
    if report["data_rows"] != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp209 trusted control cache row count mismatch")
    columns = set(pd.read_csv(path, nrows=0).columns.astype(str))
    required = {"id", "well", str(spec["prediction_column"])}
    if not required.issubset(columns):
        raise ValueError(f"exp209 control columns are incomplete: {sorted(required - columns)}")
    return {
        **report,
        "columns": sorted(columns),
        "usage": "late_prediction_and_metric_baseline_only_no_control_rerun",
    }


def validate_enriched_control_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(get_nested(config, "data.exp209_enriched_control", {}))
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    report = inspect_gzip_csv(path)
    if report["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp209 enriched LikPF control decompressed SHA mismatch")
    if report["data_rows"] != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp209 enriched LikPF control row count mismatch")
    columns = set(pd.read_csv(path, nrows=0).columns.astype(str))
    required = {
        "id",
        "well",
        str(spec["hmm_prediction_column"]),
        str(spec["hmm_minus_likpf_column"]),
    }
    if not required.issubset(columns):
        raise ValueError(
            f"exp209 enriched columns are incomplete: {sorted(required - columns)}"
        )
    return {
        **report,
        "columns": sorted(columns),
        "usage": "late_fixed_likpf_50_50_diagnostic_only",
    }


def preflight_hidden_like_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(get_nested(config, "data.hidden_like", {}))
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("exp115 hidden-like assignment SHA mismatch")
    columns = set(pd.read_csv(path, nrows=0).columns.astype(str))
    required = {"well_id", *dict(spec["valid_role_columns"]).values()}
    if not required.issubset(columns):
        raise ValueError("hidden-like assignment schema mismatch")
    return {
        "path": str(path),
        "raw_sha256": actual_sha,
        "roles_loaded": False,
        "usage": "roles_are_loaded_only_after_prediction_freeze",
    }


def load_hidden_like_assignment(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.hidden_like", {}))
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("exp115 hidden-like assignment SHA mismatch")
    role_columns = dict(spec["valid_role_columns"])
    frame = pd.read_csv(
        path,
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": "string"},
    )
    frame["well_id"] = frame["well_id"].astype(str)
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment must contain one row per well")
    if len(frame) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("hidden-like assignment well count mismatch")
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "wells": len(frame),
        "loaded_after_prediction_freeze": True,
    }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    if "TVT" in frame.columns:
        raise RuntimeError("unknown-suffix truth crossed the pre-freeze boundary")
    return frame.reset_index(drop=True)


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.sort_values("TVT", kind="mergesort").reset_index(drop=True)
    frame["GR"] = frame["GR"].ffill().bfill()
    finite = np.isfinite(frame[["TVT", "GR"]].to_numpy(np.float64)).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"{well} typewell has insufficient finite TVT/GR rows")
    return frame


# %% [markdown]
# ## 5. Frozen exp226 K16 geometry-rate schedule reconstruction


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
    if n_rows <= 0 or k_segments <= 0:
        raise ValueError("K16 segmentation requires positive row and segment counts")
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
    if not (np.isfinite(md).all() and np.isfinite(z).all() and np.isfinite(geop).all()):
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
        delta_by_segment[valid_segments] = geometry_rate[valid_segments] - first_geometry_rate
        mu_by_segment[valid_segments] = parent_rate + delta_by_segment[valid_segments]
        fallback_by_segment[valid_segments] = False

    anchor_idx = int(row_idx[0] - 1)
    anchor_md = float(horizontal.loc[anchor_idx, "MD"])
    anchor_z = float(horizontal.loc[anchor_idx, "Z"])
    anchor_tvt = float(horizontal.loc[anchor_idx, "TVT_input"])
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
    ledger = pd.DataFrame(
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
    return schedule, ledger, fallback


@dataclass(frozen=True)
class FrozenPriorSchedule:
    rowwise_schedule: pd.DataFrame
    segment_ledger: pd.DataFrame
    fallback_summary: pd.DataFrame
    rowwise_schedule_content_sha256: str
    segment_ledger_content_sha256: str


def build_and_freeze_prior_schedule(
    geometry: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> FrozenPriorSchedule:
    k_segments = int(get_nested(config, "model.stage_0.k_segments"))
    schedules: list[pd.DataFrame] = []
    ledgers: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    for well, well_geometry in geometry.groupby("well_id", sort=True, observed=True):
        schedule, ledger, fallback = build_well_rate_schedule(
            str(well),
            well_geometry,
            load_horizontal_without_truth(str(well), raw_dir),
            k_segments=k_segments,
        )
        schedules.append(schedule)
        ledgers.append(ledger)
        fallback_rows.append(fallback)
    schedule = (
        pd.concat(schedules, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    segment_ledger = (
        pd.concat(ledgers, ignore_index=True)
        .sort_values(["well_id", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    fallback_summary = (
        pd.DataFrame(fallback_rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if any(
        forbidden in schedule.columns
        for forbidden in ["tvt_true", "TVT", "error", "abs_error"]
    ):
        raise RuntimeError("truth entered the frozen geometry schedule")
    if len(schedule) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("frozen schedule row count mismatch")
    schedule_sha = dataframe_content_sha256(schedule, SCHEDULE_CONTENT_COLUMNS)
    ledger_sha = dataframe_content_sha256(segment_ledger)
    if schedule_sha != str(
        get_nested(config, "model.stage_1.expected_stage0_schedule_logical_sha256")
    ):
        raise RuntimeError("reconstructed Stage 0 schedule SHA mismatch")
    if ledger_sha != str(
        get_nested(config, "model.stage_1.expected_stage0_geometry_ledger_logical_sha256")
    ):
        raise RuntimeError("reconstructed Stage 0 geometry ledger SHA mismatch")
    schedule.insert(
        0,
        "id",
        schedule["well_id"].astype(str) + "_" + schedule["row_idx"].astype(str),
    )
    schedule["prefix_rate"] = schedule["parent_initial_rate"].to_numpy(np.float64)
    return FrozenPriorSchedule(
        rowwise_schedule=schedule,
        segment_ledger=segment_ledger,
        fallback_summary=fallback_summary,
        rowwise_schedule_content_sha256=schedule_sha,
        segment_ledger_content_sha256=ledger_sha,
    )


# %% [markdown]
# ## 6. Exp209 observation preparation


# %%
def exp209_prefix_scale(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> dict[str, Any]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(
        known_tvt,
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    residual = known_gr - typewell_at_known
    raw_sigma = float(np.nanstd(residual, ddof=0))
    sigma = float(np.clip(raw_sigma, 10.0, 60.0))
    return {
        "sigma_gr_raw": raw_sigma,
        "sigma_gr": sigma,
        "known_prefix_rows": int(len(known)),
        "missing_known_gr_rows": int(pd.to_numeric(known["GR"], errors="coerce").isna().sum()),
        "sigma_mode": "std",
        "missing_known_gr_fill": 0.0,
        "affine_a": 1.0,
        "affine_b": 0.0,
    }


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    hmm = dict(get_nested(config, "model.stage_1.hmm", {}))
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_index = np.flatnonzero(np.isfinite(tvt_input))
    eval_index = np.flatnonzero(~np.isfinite(tvt_input))
    if len(known_index) == 0 or len(eval_index) == 0:
        raise ValueError("HMM input requires a visible prefix and unknown suffix")
    if not np.array_equal(known_index, np.arange(len(known_index))):
        raise ValueError("HMM input requires one contiguous visible prefix")
    if not np.array_equal(eval_index, np.arange(eval_index[0], len(horizontal))):
        raise ValueError("HMM input requires one contiguous unknown suffix")
    schedule = schedule.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(schedule["row_idx"].to_numpy(np.int64), eval_index):
        raise ValueError("prior schedule row identities do not match the HMM suffix")

    last_index = int(known_index[-1])
    last_tvt = float(tvt_input[last_index])
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    step = float(hmm["step"])
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    state_gr = np.interp(grid, typewell_tvt, typewell_gr)
    observed_gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[eval_index]
    )
    md_all = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="raise").to_numpy(np.float64)
    md = md_all[eval_index]
    z = z_all[eval_index]
    dm = np.maximum(np.diff(np.concatenate([[md_all[last_index]], md])), 1.0)
    dz = np.diff(np.concatenate([[z_all[last_index]], z]))
    prefix = exp209_initial_rate(horizontal)
    prior_mu = schedule["mu_rate"].to_numpy(np.float64)
    residual_rates = np.linspace(
        -float(hmm["residual_rate_span"]),
        float(hmm["residual_rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    initial_residual_rate = float(prefix["initial_rate"] - prior_mu[0])
    if abs(initial_residual_rate) > float(hmm["residual_rate_span"]) + 1.0e-12:
        raise ValueError("initial residual rate lies outside the frozen residual grid")
    scale = exp209_prefix_scale(horizontal, typewell)
    return {
        "eval_index": eval_index,
        "grid": grid,
        "state_gr": state_gr,
        "observed_gr": observed_gr,
        "dm": dm,
        "dz": dz,
        "effective_dz": dz - prior_mu * dm,
        "rates": residual_rates,
        "start_p": float((last_tvt - grid_min) / step),
        "prefix_rate": float(prefix["initial_rate"]),
        "prior_mu": prior_mu,
        "initial_residual_rate": initial_residual_rate,
        "prefix_scale": scale,
    }


# %% [markdown]
# ## 7. Exact residual-rate HMM kernel and decoding


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
    """Exp209 exact forward-backward; dz may absorb a frozen rate-mean schedule."""
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


def run_exact_hmm(
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    hmm = dict(get_nested(config, "model.stage_1.hmm", {}))
    sigma_gr = float(prepared["prefix_scale"]["sigma_gr"])
    zscore = (
        np.asarray(prepared["observed_gr"], dtype=np.float64)[:, None]
        - np.asarray(prepared["state_gr"], dtype=np.float64)[None, :]
    ) / sigma_gr
    emission = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    post_p, loglik = _hmm2_fb(
        emission,
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["effective_dz"], dtype=np.float64),
        float(hmm["step"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["initial_residual_rate"]),
        float(hmm["r0_sig"]),
        1.0,
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


def decode_well(
    well: str,
    raw_dir: Path,
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_hmm_inputs(horizontal, typewell, schedule, config)
    result = run_exact_hmm(prepared, config)
    ordered = schedule.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    prediction = ordered[
        [
            "id",
            "well_id",
            "row_idx",
            "suffix_offset",
            "fold",
            "segment_id",
            "md_since",
            "mu_rate",
            "prefix_rate",
        ]
    ].copy()
    prediction["candidate_tvt"] = np.asarray(result["mean"], dtype=np.float64)
    prediction["candidate_std"] = np.asarray(result["std"], dtype=np.float64)
    prediction["hmm_loglik"] = float(result["loglik"])
    if not np.isfinite(
        prediction[["candidate_tvt", "candidate_std", "mu_rate", "prefix_rate"]].to_numpy(
            np.float64
        )
    ).all():
        raise RuntimeError(f"{well} prediction contains non-finite values")
    return prediction, {
        "well_id": str(well),
        "fold": int(ordered["fold"].iloc[0]),
        "rows": len(prediction),
        "elapsed_seconds": time.perf_counter() - started,
        "prefix_rate": float(prepared["prefix_rate"]),
        "initial_residual_rate": float(prepared["initial_residual_rate"]),
        "prior_mu_min": float(np.min(prepared["prior_mu"])),
        "prior_mu_max": float(np.max(prepared["prior_mu"])),
        "sigma_gr": float(prepared["prefix_scale"]["sigma_gr"]),
        "loglik": float(result["loglik"]),
        "posterior_row_sum_max_abs_error": float(result["posterior_row_sum_max_abs_error"]),
    }


# %% [markdown]
# ## 8. Prediction freeze and late truth/control attachment


# %%
@dataclass(frozen=True)
class FrozenPrediction:
    frame: pd.DataFrame
    runtime: pd.DataFrame
    prediction_content_sha256: str
    runtime_content_sha256: str
    truth_access_before_freeze: int = 0


def generate_and_freeze_predictions(
    raw_dir: Path,
    frozen_schedule: FrozenPriorSchedule,
    config: Mapping[str, Any],
) -> FrozenPrediction:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exp355 exact HMM")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    schedule = frozen_schedule.rowwise_schedule
    wells = sorted(schedule["well_id"].astype(str).unique())

    def decode_one(index: int, well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(f"[HMM {index}/{len(wells)}] well={well}", flush=True)
        return decode_well(
            well,
            raw_dir,
            schedule.loc[schedule["well_id"] == well],
            config,
        )

    num_workers = int(get_nested(config, "runtime.num_workers", 1))
    if num_workers > 1:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=num_workers, prefer="threads")(
            delayed(decode_one)(index, well) for index, well in enumerate(wells, start=1)
        )
    else:
        outputs = [decode_one(index, well) for index, well in enumerate(wells, start=1)]
    prediction = (
        pd.concat([item[0] for item in outputs], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    runtime = (
        pd.DataFrame([item[1] for item in outputs])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if prediction.duplicated(["well_id", "row_idx"]).any():
        raise RuntimeError("candidate prediction contains duplicate row identities")
    return FrozenPrediction(
        frame=prediction,
        runtime=runtime,
        prediction_content_sha256=dataframe_content_sha256(prediction),
        runtime_content_sha256=dataframe_content_sha256(runtime),
    )


def _require_prediction_frozen(frozen: FrozenPrediction | None) -> FrozenPrediction:
    if frozen is None or len(frozen.prediction_content_sha256) != 64:
        raise RuntimeError("truth/control attachment requires a frozen prediction SHA")
    if frozen.truth_access_before_freeze != 0:
        raise RuntimeError("target truth was accessed before prediction freeze")
    return frozen


def attach_truth_and_controls_after_freeze(
    frozen: FrozenPrediction,
    raw_dir: Path,
    parent_report: Mapping[str, Any],
    enriched_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen = _require_prediction_frozen(frozen)
    truth_frames: list[pd.DataFrame] = []
    for well in sorted(frozen.frame["well_id"].astype(str).unique()):
        path = raw_dir / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        eval_index = np.flatnonzero(tvt_input.isna().to_numpy())
        truth = pd.to_numeric(horizontal.iloc[eval_index]["TVT"], errors="raise")
        truth_frames.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in eval_index],
                    "well_id": str(well),
                    "row_idx": eval_index.astype(np.int32),
                    "true_tvt": truth.to_numpy(np.float64),
                }
            )
        )
    truth = pd.concat(truth_frames, ignore_index=True)
    parent_spec = dict(get_nested(config, "data.exp209_control", {}))
    parent_path = Path(str(parent_report["path"]))
    parent = pd.read_csv(
        parent_path,
        usecols=["id", "well", str(parent_spec["prediction_column"])],
        dtype={"id": str, "well": str},
    ).rename(
        columns={
            "well": "well_id",
            str(parent_spec["prediction_column"]): "parent_tvt",
        }
    )
    enriched_spec = dict(get_nested(config, "data.exp209_enriched_control", {}))
    enriched = pd.read_csv(
        Path(str(enriched_report["path"])),
        usecols=[
            "id",
            "well",
            str(enriched_spec["hmm_prediction_column"]),
            str(enriched_spec["hmm_minus_likpf_column"]),
        ],
        dtype={"id": str, "well": str},
    ).rename(
        columns={
            "well": "well_id",
            str(enriched_spec["hmm_prediction_column"]): "enriched_parent_tvt",
            str(enriched_spec["hmm_minus_likpf_column"]): "hmm_minus_likpf",
        }
    )
    enriched["likpf_tvt"] = (
        pd.to_numeric(enriched["enriched_parent_tvt"], errors="raise")
        - pd.to_numeric(enriched["hmm_minus_likpf"], errors="raise")
    )
    hidden, hidden_report = load_hidden_like_assignment(config)
    roles = dict(get_nested(config, "data.hidden_like.valid_role_columns"))
    hidden = hidden.rename(
        columns={
            roles["verification_like_spatial"]: "hidden_like_spatial_role",
            roles[
                "verification_like_typewell_purged"
            ]: "hidden_like_typewell_purged_role",
        }
    )
    frame = (
        frozen.frame.merge(
            truth,
            on=["id", "well_id", "row_idx"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            parent,
            on=["id", "well_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            enriched[["id", "well_id", "enriched_parent_tvt", "likpf_tvt"]],
            on=["id", "well_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            hidden,
            on="well_id",
            how="left",
            validate="many_to_one",
        )
    )
    frame["hidden_like_spatial"] = frame["hidden_like_spatial_role"].astype(str) == "valid"
    frame["hidden_like_typewell_purged"] = (
        frame["hidden_like_typewell_purged_role"].astype(str) == "valid"
    )
    frame["parent_likpf_50_50"] = 0.5 * (
        frame["parent_tvt"] + frame["likpf_tvt"]
    )
    frame["candidate_likpf_50_50"] = 0.5 * (
        frame["candidate_tvt"] + frame["likpf_tvt"]
    )
    parent_cross_source_max_abs = float(
        np.max(
            np.abs(
                frame["parent_tvt"].to_numpy(np.float64)
                - frame["enriched_parent_tvt"].to_numpy(np.float64)
            )
        )
    )
    finite = frame[
        [
            "candidate_tvt",
            "parent_tvt",
            "likpf_tvt",
            "parent_likpf_50_50",
            "candidate_likpf_50_50",
            "true_tvt",
        ]
    ].to_numpy(np.float64)
    if len(frame) != len(frozen.frame) or not np.isfinite(finite).all():
        raise RuntimeError("late truth/control attachment failed identity or finite checks")
    if parent_cross_source_max_abs > 1.0e-6:
        raise RuntimeError("saved exp209 HMM predictions disagree across trusted caches")
    return frame, {
        "prediction_content_sha256_before_truth": frozen.prediction_content_sha256,
        "truth_access_before_freeze": frozen.truth_access_before_freeze,
        "truth_rows_attached": len(truth),
        "parent_rows_attached": len(parent),
        "enriched_rows_attached": len(enriched),
        "parent_cross_source_max_abs": parent_cross_source_max_abs,
        "hidden_like": hidden_report,
    }


# %% [markdown]
# ## 9. Metrics, scientific gate, and generated artifacts


# %%
def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(actual_array - predicted_array))))


def paired_metric_row(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    scope: str,
    comparison: str,
    parent_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"metric scope is empty: {scope}")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = rmse(truth, selected[candidate_column].to_numpy(np.float64))
    parent = rmse(truth, selected[parent_column].to_numpy(np.float64))
    return {
        "comparison": comparison,
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate,
        "parent_rmse": parent,
        "delta_rmse_candidate_minus_parent": candidate - parent,
        "improvement_ft": parent - candidate,
    }


def build_scope_metrics(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("pooled", np.ones(len(frame), dtype=bool)),
    ]
    for fold in [int(value) for value in get_nested(config, "validation.expected_folds")]:
        scopes.append((f"fold_{fold}", frame["fold"].to_numpy(np.int64) == fold))
    distance = frame["md_since"].to_numpy(np.float64)
    distance_scopes = [
        ("distance_000_050", (distance >= 0.0) & (distance < 50.0)),
        ("distance_050_100", (distance >= 50.0) & (distance < 100.0)),
        ("distance_100_250", (distance >= 100.0) & (distance < 250.0)),
        ("distance_250_500", (distance >= 250.0) & (distance < 500.0)),
        ("distance_500_1000", (distance >= 500.0) & (distance < 1000.0)),
        ("distance_1000_plus", distance >= 1000.0),
    ]
    scopes.extend((name, mask) for name, mask in distance_scopes if bool(mask.any()))
    scopes.extend(
        [
            (
                "hidden_like_spatial",
                frame["hidden_like_spatial"].to_numpy(bool),
            ),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    comparisons = [
        ("direct", "parent_tvt", "candidate_tvt"),
        (
            "fixed_likpf_50_50",
            "parent_likpf_50_50",
            "candidate_likpf_50_50",
        ),
    ]
    metrics = pd.DataFrame(
        [
            paired_metric_row(
                frame,
                mask,
                scope=scope,
                comparison=comparison,
                parent_column=parent_column,
                candidate_column=candidate_column,
            )
            for comparison, parent_column, candidate_column in comparisons
            for scope, mask in scopes
        ]
    )
    fold_metrics = metrics.loc[
        metrics["scope"].isin(["pooled", *[f"fold_{value}" for value in range(5)]])
    ].reset_index(drop=True)
    distance_metrics = metrics.loc[metrics["scope"].str.startswith("distance_")].reset_index(
        drop=True
    )
    hidden_metrics = metrics.loc[metrics["scope"].str.startswith("hidden_like_")].reset_index(
        drop=True
    )
    return fold_metrics, distance_metrics, hidden_metrics


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        candidate = rmse(truth, group["candidate_tvt"].to_numpy(np.float64))
        parent = rmse(truth, group["parent_tvt"].to_numpy(np.float64))
        rows.append(
            {
                "well_id": str(well),
                "fold": int(group["fold"].iloc[0]),
                "rows": len(group),
                "candidate_rmse": candidate,
                "parent_rmse": parent,
                "delta_rmse_candidate_minus_parent": candidate - parent,
            }
        )
    return pd.DataFrame(rows)


def build_geometry_fallback_metrics(
    segment_ledger: pd.DataFrame,
    fallback_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("pooled", segment_ledger, fallback_summary)
    ]
    for fold, group in segment_ledger.groupby("fold", sort=True):
        scopes.append(
            (
                f"fold_{int(fold)}",
                group,
                fallback_summary.loc[
                    fallback_summary["fold"].astype(int) == int(fold)
                ],
            )
        )
    for scope, segments, wells in scopes:
        rows.append(
            {
                "scope": scope,
                "segments": len(segments),
                "wells": int(segments["well_id"].nunique()),
                "fallback_segments": int(
                    segments["geometry_fallback"].astype(bool).sum()
                ),
                "fallback_wells": int(
                    (~wells["first_geometry_segment_valid"].astype(bool)).sum()
                ),
                "prefix_rate_fallback_wells": int(
                    wells["parent_initial_rate_fallback"].astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_gate(
    frame: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    distance_metrics: pd.DataFrame,
    hidden_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    frozen_schedule: FrozenPriorSchedule,
    frozen_prediction: FrozenPrediction,
    parent_report: Mapping[str, Any],
    enriched_report: Mapping[str, Any],
    runtime_seconds: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = dict(get_nested(config, "model.stage_1.pass_requires_all", {}))
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    direct = fold_metrics.loc[fold_metrics["comparison"] == "direct"]
    blend = fold_metrics.loc[fold_metrics["comparison"] == "fixed_likpf_50_50"]
    pooled = direct.loc[direct["scope"] == "pooled"].iloc[0]
    pooled_blend = blend.loc[blend["scope"] == "pooled"].iloc[0]
    folds = direct.loc[direct["scope"].str.startswith("fold_")]
    improved_folds = int((folds["improvement_ft"] > 0.0).sum())
    distance_1000 = distance_metrics.loc[
        (distance_metrics["comparison"] == "direct")
        & (distance_metrics["scope"] == "distance_1000_plus")
    ].iloc[0]
    spatial = hidden_metrics.loc[
        (hidden_metrics["comparison"] == "direct")
        & (hidden_metrics["scope"] == "hidden_like_spatial")
    ].iloc[0]
    typewell = hidden_metrics.loc[
        (hidden_metrics["comparison"] == "direct")
        & (hidden_metrics["scope"] == "hidden_like_typewell_purged")
    ].iloc[0]
    parent_p95 = float(by_well["parent_rmse"].quantile(0.95))
    candidate_p95 = float(by_well["candidate_rmse"].quantile(0.95))
    p95_delta = candidate_p95 - parent_p95
    worst = by_well.loc[by_well["delta_rmse_candidate_minus_parent"].idxmax()]

    expected_parent = float(get_nested(config, "data.exp209_control.direct_rmse_ft"))
    expected_blend = float(get_nested(config, "data.exp209_control.fixed_likpf_50_50_rmse"))
    tolerance = float(get_nested(config, "model.stage_1.baseline_metric_absolute_tolerance"))
    technical = {
        "rows": len(frame),
        "expected_rows": expected_rows,
        "wells": int(frame["well_id"].nunique()),
        "expected_wells": expected_wells,
        "finite_prediction_coverage": float(
            np.isfinite(frame["candidate_tvt"].to_numpy(np.float64)).mean()
        ),
        "duplicate_rows": int(frame.duplicated(["well_id", "row_idx"]).sum()),
        "truth_access_before_freeze": frozen_prediction.truth_access_before_freeze,
        "hmm_well_runs": len(frozen_prediction.runtime),
        "expected_hmm_well_runs": int(
            get_nested(config, "execution_contract.stage_1_if_pass.hmm_well_runs")
        ),
        "posterior_normalization_max_abs_error": float(
            frozen_prediction.runtime["posterior_row_sum_max_abs_error"].max()
        ),
        "parent_control_decompressed_sha256": parent_report["decompressed_sha256"],
        "expected_parent_control_decompressed_sha256": get_nested(
            config, "data.exp209_control.expected_hmm_cache_decompressed_sha256"
        ),
        "enriched_control_decompressed_sha256": enriched_report["decompressed_sha256"],
        "expected_enriched_control_decompressed_sha256": get_nested(
            config, "data.exp209_enriched_control.expected_decompressed_sha256"
        ),
        "stage0_schedule_logical_sha256": (
            frozen_schedule.rowwise_schedule_content_sha256
        ),
        "expected_stage0_schedule_logical_sha256": get_nested(
            config, "model.stage_1.expected_stage0_schedule_logical_sha256"
        ),
        "stage0_geometry_ledger_logical_sha256": (
            frozen_schedule.segment_ledger_content_sha256
        ),
        "expected_stage0_geometry_ledger_logical_sha256": get_nested(
            config, "model.stage_1.expected_stage0_geometry_ledger_logical_sha256"
        ),
        "parent_rmse": float(pooled["parent_rmse"]),
        "expected_parent_rmse": expected_parent,
        "parent_rmse_absolute_difference": abs(
            float(pooled["parent_rmse"]) - expected_parent
        ),
        "parent_blend_rmse": float(pooled_blend["parent_rmse"]),
        "expected_parent_blend_rmse": expected_blend,
        "parent_blend_rmse_absolute_difference": abs(
            float(pooled_blend["parent_rmse"]) - expected_blend
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(
            get_nested(config, "runtime.kaggle.runtime_limit_seconds")
        ),
        "prediction_content_sha256": frozen_prediction.prediction_content_sha256,
    }
    technical["passed"] = bool(
        technical["rows"] == expected_rows
        and technical["wells"] == expected_wells
        and technical["finite_prediction_coverage"] == 1.0
        and technical["duplicate_rows"] == 0
        and technical["truth_access_before_freeze"] == 0
        and technical["hmm_well_runs"] == technical["expected_hmm_well_runs"]
        and technical["posterior_normalization_max_abs_error"] <= 1.0e-6
        and technical["parent_control_decompressed_sha256"]
        == technical["expected_parent_control_decompressed_sha256"]
        and technical["enriched_control_decompressed_sha256"]
        == technical["expected_enriched_control_decompressed_sha256"]
        and technical["stage0_schedule_logical_sha256"]
        == technical["expected_stage0_schedule_logical_sha256"]
        and technical["stage0_geometry_ledger_logical_sha256"]
        == technical["expected_stage0_geometry_ledger_logical_sha256"]
        and technical["parent_rmse_absolute_difference"] <= tolerance
        and technical["parent_blend_rmse_absolute_difference"] <= tolerance
        and runtime_seconds <= technical["runtime_limit_seconds"]
    )

    scientific = {
        "candidate_rmse": float(pooled["candidate_rmse"]),
        "parent_rmse": float(pooled["parent_rmse"]),
        "improvement_ft": float(pooled["improvement_ft"]),
        "minimum_improvement_ft": float(
            criteria["minimum_rmse_gain_vs_exp209_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(criteria["minimum_improved_folds"]),
        "distance_1000_plus_delta": float(
            distance_1000["delta_rmse_candidate_minus_parent"]
        ),
        "hidden_like_spatial_delta": float(
            spatial["delta_rmse_candidate_minus_parent"]
        ),
        "hidden_like_typewell_purged_delta": float(
            typewell["delta_rmse_candidate_minus_parent"]
        ),
        "by_well_p95_delta": p95_delta,
        "worst_well_id": str(worst["well_id"]),
        "worst_well_delta": float(worst["delta_rmse_candidate_minus_parent"]),
        "maximum_worst_well_regression_ft": float(
            criteria["maximum_worst_well_regression_ft"]
        ),
        "fixed_likpf_50_50_parent_rmse": float(pooled_blend["parent_rmse"]),
        "fixed_likpf_50_50_candidate_rmse": float(pooled_blend["candidate_rmse"]),
        "fixed_likpf_50_50_delta": float(
            pooled_blend["delta_rmse_candidate_minus_parent"]
        ),
    }
    checks = {
        "minimum_direct_rmse_gain": bool(
            scientific["improvement_ft"] >= scientific["minimum_improvement_ft"]
        ),
        "minimum_improved_folds": bool(
            scientific["improved_folds"] >= scientific["minimum_improved_folds"]
        ),
        "distance_1000_plus_non_regression": bool(
            scientific["distance_1000_plus_delta"] <= 0.0
        ),
        "hidden_like_spatial_non_regression": bool(
            scientific["hidden_like_spatial_delta"] <= 0.0
        ),
        "hidden_like_typewell_purged_non_regression": bool(
            scientific["hidden_like_typewell_purged_delta"] <= 0.0
        ),
        "by_well_p95_non_regression": bool(
            scientific["by_well_p95_delta"] <= 0.0
        ),
        "worst_well_regression_guard": bool(
            scientific["worst_well_delta"]
            <= scientific["maximum_worst_well_regression_ft"]
        ),
        "fixed_likpf_50_50_non_regression": bool(
            scientific["fixed_likpf_50_50_delta"] <= 0.0
        ),
    }
    scientific["checks"] = checks
    scientific["passed"] = bool(all(checks.values()))
    passed = bool(technical["passed"] and scientific["passed"])
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            "stage_1_exact_hmm_passed_train_side_no_automatic_inference"
            if passed
            else "stage_1_exact_hmm_failed_no_automatic_inference"
        ),
        "user_override": get_nested(config, "model.stage_1.user_override"),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "failure_action": (
            "report_result_without_parameter_blend_selector_inference_or_submission_rescue"
        ),
    }


# %% [markdown]
# ## 10. Experiment orchestration


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.is_dir() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp355 Stage 1 must run first on Kaggle; "
            "local execution requires explicit approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.perf_counter()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest, raw_report = validate_raw_well_identity(config, raw_dir)
    geometry, geometry_report = load_exp226_geometry(config)
    parent_report = validate_parent_control_dependency(config)
    enriched_report = validate_enriched_control_dependency(config)
    hidden_preflight = preflight_hidden_like_dependency(config)

    scientific_contract = build_scientific_contract(config)
    contract_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_stage1_scientific_contract.json",
        scientific_contract,
    )
    raw_manifest_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_raw_well_identity.csv",
        raw_manifest,
    )
    input_manifest = {
        "truth_attached": False,
        "raw_train": raw_report,
        "exp226_safe_geometry": geometry_report,
        "exp209_parent_control": parent_report,
        "exp209_enriched_control": enriched_report,
        "hidden_like_dependency": hidden_preflight,
        "safe_geometry_columns_loaded": SAFE_GEOMETRY_COLUMNS,
        "forbidden_exp226_columns_loaded": [],
    }
    input_manifest["input_manifest_sha256"] = mapping_sha256(input_manifest)
    input_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_stage1_input_manifest.json",
        input_manifest,
    )

    frozen_schedule = build_and_freeze_prior_schedule(geometry, raw_dir, config)
    schedule_report = write_gzip_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_rate_prior_schedule.csv.gz",
        frozen_schedule.rowwise_schedule,
    )
    ledger_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_geometry_segment_ledger.csv",
        frozen_schedule.segment_ledger,
    )
    fallback_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_fallback_summary.csv",
        frozen_schedule.fallback_summary,
    )
    freeze_manifest = {
        "truth_attached": False,
        "truth_access_before_prediction_freeze": 0,
        "schedule_logical_sha256": (
            frozen_schedule.rowwise_schedule_content_sha256
        ),
        "expected_schedule_logical_sha256": get_nested(
            config, "model.stage_1.expected_stage0_schedule_logical_sha256"
        ),
        "geometry_ledger_logical_sha256": (
            frozen_schedule.segment_ledger_content_sha256
        ),
        "expected_geometry_ledger_logical_sha256": get_nested(
            config, "model.stage_1.expected_stage0_geometry_ledger_logical_sha256"
        ),
        "schedule_artifact": schedule_report,
        "geometry_ledger_artifact": ledger_report,
        "fallback_artifact": fallback_report,
        "rows": len(frozen_schedule.rowwise_schedule),
        "wells": int(frozen_schedule.rowwise_schedule["well_id"].nunique()),
    }
    freeze_manifest["freeze_manifest_sha256"] = mapping_sha256(freeze_manifest)
    freeze_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_stage1_schedule_freeze_manifest.json",
        freeze_manifest,
    )

    frozen_prediction = generate_and_freeze_predictions(
        raw_dir,
        frozen_schedule,
        config,
    )
    prediction_report = write_gzip_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_oof_predictions.csv.gz",
        frozen_prediction.frame,
    )
    runtime_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_by_well_runtime.csv",
        frozen_prediction.runtime,
    )
    prediction_freeze_seconds = time.perf_counter() - started

    frame, late_attachment = attach_truth_and_controls_after_freeze(
        frozen_prediction,
        raw_dir,
        parent_report,
        enriched_report,
        config,
    )
    fold_metrics, distance_metrics, hidden_metrics = build_scope_metrics(frame, config)
    by_well = build_by_well_metrics(frame)
    fallback_metrics = build_geometry_fallback_metrics(
        frozen_schedule.segment_ledger,
        frozen_schedule.fallback_summary,
    )
    runtime_seconds = time.perf_counter() - started
    gate = evaluate_gate(
        frame,
        fold_metrics,
        distance_metrics,
        hidden_metrics,
        by_well,
        frozen_schedule,
        frozen_prediction,
        parent_report,
        enriched_report,
        runtime_seconds,
        config,
    )

    fold_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_fold_metrics.csv",
        fold_metrics,
    )
    distance_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_distance_bucket_metrics.csv",
        distance_metrics,
    )
    hidden_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_hidden_like_metrics.csv",
        hidden_metrics,
    )
    by_well_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv",
        by_well,
    )
    fallback_metrics_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_geometry_fallback_metrics.csv",
        fallback_metrics,
    )
    gate_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_stage1_promotion_gate.json",
        gate,
    )
    sha_manifest = pd.DataFrame(
        [
            {
                "name": name,
                "path": report["path"],
                "raw_sha256": report.get("raw_sha256"),
                "logical_content_sha256": report.get(
                    "logical_content_sha256", report.get("content_sha256")
                ),
                "decompressed_sha256": report.get("decompressed_sha256"),
            }
            for name, report in {
                "scientific_contract": contract_report,
                "input_manifest": input_report,
                "raw_well_identity": raw_manifest_report,
                "schedule": schedule_report,
                "geometry_ledger": ledger_report,
                "fallback_summary": fallback_report,
                "schedule_freeze_manifest": freeze_report,
                "oof_predictions": prediction_report,
                "by_well_runtime": runtime_report,
                "fold_metrics": fold_report,
                "distance_metrics": distance_report,
                "hidden_like_metrics": hidden_report,
                "by_well_metrics": by_well_report,
                "geometry_fallback_metrics": fallback_metrics_report,
                "promotion_gate": gate_report,
            }.items()
        ]
    )
    sha_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_stage1_sha_manifest.csv",
        sha_manifest,
    )
    direct_pooled = fold_metrics.loc[
        (fold_metrics["comparison"] == "direct")
        & (fold_metrics["scope"] == "pooled")
    ].iloc[0]
    blend_pooled = fold_metrics.loc[
        (fold_metrics["comparison"] == "fixed_likpf_50_50")
        & (fold_metrics["scope"] == "pooled")
    ].iloc[0]
    status = (
        "stage1_train_side_gate_passed_no_automatic_inference"
        if gate["passed"]
        else "stage1_train_side_gate_failed_no_automatic_inference"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": "stage_1_exact_hmm_user_override",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "prediction_frozen_at_seconds": prediction_freeze_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "execution_counts": validate_scientific_contract(config),
        "truth_attachment": late_attachment,
        "promotion_gate": gate,
        "direct_overall": direct_pooled.to_dict(),
        "fixed_likpf_50_50_overall": blend_pooled.to_dict(),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "schedule_logical_sha256": (
            frozen_schedule.rowwise_schedule_content_sha256
        ),
        "prediction_logical_sha256": (
            frozen_prediction.prediction_content_sha256
        ),
        "runtime_versions": runtime_versions(),
        "models": 0,
        "boosters": 0,
        "control_reruns": 0,
        "inference_enabled": False,
        "submission_created": False,
        "sha_manifest": sha_report,
    }
    summary_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_stage1_summary.json",
        summary,
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": "stage_1_exact_hmm_user_override",
        "cv": float(direct_pooled["candidate_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "direct_overall": direct_pooled.to_dict(),
        "fixed_likpf_50_50_overall": blend_pooled.to_dict(),
        "promotion_gate": gate,
        "schedule_sha256": frozen_schedule.rowwise_schedule_content_sha256,
        "prediction_sha256": frozen_prediction.prediction_content_sha256,
        "model_sha256": None,
        "submission_sha256": None,
        "summary_path": summary_report["path"],
        "notes": (
            "User-overridden Stage 1 train-side exact HMM only; no parameter "
            "rescue, raw-test inference, model, selector, or submission."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(fold_metrics.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. Setup and configuration preview

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = read_config()
    CONTRACT_COUNTS = validate_scientific_contract(CONFIG)
    print("Experiment:", get_nested(CONFIG, "experiment.name"))
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Status:", get_nested(CONFIG, "experiment.status"))
    print("Parent:", get_nested(CONFIG, "lineage.parent"))
    print("Variant:", get_nested(CONFIG, "model.stage_1.active_variants"))
    print("Execution contract:", json.dumps(CONTRACT_COUNTS, sort_keys=True))
    print("Kaggle push approved:", get_nested(CONFIG, "execution.kaggle_push_approved"))
    print("Run Stage 1:", get_nested(CONFIG, "execution.run_stage_1"))
    print("Inference enabled:", get_nested(CONFIG, "inference.enabled"))


# %% [markdown]
# ## 12. Kaggle CPU execution

# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_stage_1")):
        EXP355_SUMMARY = run_full_experiment(CONFIG)
        print(json.dumps(to_jsonable(EXP355_SUMMARY), indent=2, sort_keys=True))
    else:
        print(
            "Implementation is ready. Kaggle Stage 1 execution remains disabled "
            "until a separate run approval updates config.yaml."
        )
