# %% [markdown]
# # exp362 segment local donor slope exact HMM train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Raw input, parent-control, and fold preflight
# 5. Fold-safe K16 donor-segment construction
# 6. Target-segment local-gradient prior construction
# 7. Exp209 observation preparation
# 8. Exact residual-rate HMM kernel and decoding
# 9. Prediction freeze and late truth/control attachment
# 10. Metrics, scientific gate, and generated artifacts
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


EXPERIMENT_NAME = "exp362_segment_local_donor_slope_exact_hmm"
OUTPUT_PREFIX = EXPERIMENT_NAME
VARIANT = "k16_segment_local_gradient_residual_hmm"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXECUTE_NOTEBOOK = os.environ.get("EXP362_IMPORT_ONLY") != "1"


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
    raise FileNotFoundError("exp362 config.yaml was not found")


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
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name mismatch")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp362 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != (
        "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    ):
        raise ValueError("exp362 must remain directly parented to exp209")
    if get_nested(config, "model.active_variants") != [VARIANT]:
        raise ValueError("exp362 allows exactly one frozen scientific variant")

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
    actual_counts = dict(get_nested(config, "execution_contract", {}))
    if actual_counts != expected_counts:
        raise ValueError(
            f"execution contract drifted: actual={actual_counts}, expected={expected_counts}"
        )

    frozen_values = {
        "model.donor_segments.k_segments": 16,
        "model.local_gradient.n_neighbors": 50,
        "model.local_gradient.bandwidth_ft": 500.0,
        "model.local_gradient.ridge_relative_trace": 0.001,
        "model.local_gradient.minimum_valid_donor_wells": 10,
        "model.local_gradient.minimum_effective_donors": 10.0,
        "model.local_gradient.maximum_nearest_distance_ft": 1500.0,
        "model.local_gradient.minimum_directional_information": 0.30,
        "model.local_gradient.minimum_target_horizontal_speed_norm": 0.30,
        "model.local_gradient.maximum_abs_delta_from_prefix_rate": 0.10,
        "model.hmm.step": 0.35,
        "model.hmm.n_rates": 41,
        "model.hmm.residual_rate_span": 0.10,
        "model.hmm.sig_r": 0.002,
        "model.hmm.sig_p": 0.02,
        "model.hmm.start_sig": 0.75,
        "model.hmm.r0_sig": 0.01,
        "model.hmm.band_pad": 100.0,
        "model.hmm.momentum": 0.998,
    }
    for dotted_key, expected in frozen_values.items():
        actual = get_nested(config, dotted_key)
        if actual != expected:
            raise ValueError(
                f"frozen contract drifted for {dotted_key}: {actual!r} != {expected!r}"
            )

    forbidden_inputs = {
        "exp226_oof",
        "tvt_geop",
        "tvt_pred",
        "gr_delta",
        "adaptive_kappa",
        "near_strike_ancc",
        "exp226_u_projection",
    }
    actual_forbidden = set(get_nested(config, "data.forbidden_inputs", []))
    if forbidden_inputs != actual_forbidden:
        raise ValueError("exp226 forbidden-input contract drifted")
    if not bool(get_nested(config, "implementation.enabled")):
        raise RuntimeError("exp362 implementation must be enabled")
    if bool(get_nested(config, "inference.enabled")):
        raise RuntimeError("exp362 inference must remain disabled")
    if bool(get_nested(config, "execution.run_inference")):
        raise RuntimeError("exp362 inference execution is forbidden")
    if bool(get_nested(config, "execution.create_submission")):
        raise RuntimeError("exp362 submission creation is forbidden")

    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp362 Kaggle package/push/run is not approved")
        if not bool(get_nested(config, "execution.run_hmm")):
            raise RuntimeError("exp362 run_hmm must be explicitly enabled")
        if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
            raise RuntimeError("exp362 must run on Kaggle CPU")
    return expected_counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "variant": VARIANT,
        "change": "constant_rate_mean_to_fold_safe_k16_segment_local_gradient_schedule",
        "truth_attached": False,
        "model": get_nested(config, "model"),
        "success_criteria": get_nested(config, "success_criteria"),
        "execution_contract": get_nested(config, "execution_contract"),
        "forbidden_inputs": get_nested(config, "data.forbidden_inputs"),
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


def stable_fold_assignment(
    wells: Sequence[str],
    *,
    seed: int = 42,
    n_folds: int = 5,
) -> pd.DataFrame:
    rows = [
        {
            "well_id": str(well),
            "fold_order_sha256": hashlib.sha256(f"{seed}|{well}".encode()).hexdigest(),
        }
        for well in sorted(set(str(value) for value in wells))
    ]
    frame = pd.DataFrame(rows).sort_values(["fold_order_sha256", "well_id"], kind="mergesort")
    frame["fold"] = (np.arange(len(frame), dtype=np.int64) % int(n_folds)).astype(np.int8)
    return frame.sort_values("well_id", kind="mergesort").reset_index(drop=True)


def validate_parent_control_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(get_nested(config, "data.exp209_control", {}))
    filename = str(spec["filename"])
    path = resolve_existing(filename, [str(value) for value in spec.get("candidates", [])])
    report = inspect_gzip_csv(path)
    expected_sha = str(spec["expected_hmm_cache_decompressed_sha256"])
    if report["decompressed_sha256"] != expected_sha:
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


def load_hidden_like_assignment(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.hidden_like", {}))
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("exp115 hidden-like assignment SHA mismatch")
    role_columns = dict(spec["valid_role_columns"])
    selected = ["well_id", *role_columns.values()]
    frame = pd.read_csv(path, usecols=selected, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment must contain one row per well")
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_wells:
        raise ValueError("hidden-like assignment well count mismatch")
    allowed = {
        "hidden_like_spatial": {"train", "valid"},
        "hidden_like_typewell_purged": {
            "train",
            "valid",
            "purged_train_excluded",
        },
    }
    counts: dict[str, dict[str, int]] = {}
    for scope, column in role_columns.items():
        values = set(frame[column].astype(str).unique())
        if not values.issubset(allowed[scope]):
            raise ValueError(f"unexpected hidden-like roles for {scope}: {sorted(values)}")
        counts[scope] = {
            str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()
        }
    return frame, {
        "path": str(path),
        "raw_sha256": actual_sha,
        "wells": len(frame),
        "role_counts": counts,
        "loaded_after_prediction_freeze": True,
    }


def preflight_hidden_like_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check immutable file identity/schema without parsing evaluation roles early."""
    spec = dict(get_nested(config, "data.hidden_like", {}))
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("exp115 hidden-like assignment SHA mismatch")
    header = set(pd.read_csv(path, nrows=0).columns.astype(str))
    required = {"well_id", *dict(spec["valid_role_columns"]).values()}
    if not required.issubset(header):
        raise ValueError(
            f"hidden-like dependency columns are incomplete: {sorted(required - header)}"
        )
    return {
        "path": str(path),
        "raw_sha256": actual_sha,
        "columns": sorted(header),
        "roles_parsed_before_prediction_freeze": False,
    }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    required = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    frame = pd.read_csv(path, usecols=required)
    if "TVT" in frame.columns:
        raise RuntimeError("unknown-suffix truth crossed the target pre-freeze boundary")
    return frame.reset_index(drop=True)


def load_donor_horizontal_with_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT"])
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
# ## 5. Fold-safe K16 donor-segment construction


# %%
def md_segment_ids(md: Sequence[float], k_segments: int = 16) -> np.ndarray:
    values = np.asarray(md, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("segment MD values must be non-empty and finite")
    md_min = float(values.min())
    md_max = float(values.max())
    if not md_max > md_min:
        return np.zeros(len(values), dtype=np.int16)
    edges = np.linspace(md_min, md_max, int(k_segments) + 1)
    segment = np.searchsorted(edges[1:], values, side="left")
    return np.clip(segment, 0, int(k_segments) - 1).astype(np.int16)


def _linear_fit(md: np.ndarray, values: np.ndarray) -> tuple[float, float] | None:
    valid = np.isfinite(md) & np.isfinite(values)
    x = md[valid]
    y = values[valid]
    if len(x) < 2 or not float(x.max()) > float(x.min()):
        return None
    center = float(x.mean())
    centered = x - center
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        return None
    slope = float(centered @ (y - float(y.mean())) / denominator)
    intercept = float(y.mean() - slope * center)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        return None
    return intercept, slope


def build_donor_segments_for_well(
    well: str,
    horizontal: pd.DataFrame,
    *,
    k_segments: int = 16,
) -> pd.DataFrame:
    required = {"MD", "X", "Y", "Z", "TVT"}
    if not required.issubset(horizontal.columns):
        raise ValueError(f"{well} donor frame misses {sorted(required - set(horizontal.columns))}")
    numeric = horizontal[list(required)].apply(pd.to_numeric, errors="coerce")
    finite_md = np.isfinite(numeric["MD"].to_numpy(np.float64))
    numeric = numeric.loc[finite_md].copy()
    if len(numeric) < 2:
        return pd.DataFrame()
    numeric["segment_id"] = md_segment_ids(numeric["MD"], k_segments)
    numeric["U"] = numeric["TVT"] + numeric["Z"]
    md_min = float(numeric["MD"].min())
    md_max = float(numeric["MD"].max())
    edges = np.linspace(md_min, md_max, int(k_segments) + 1)
    rows: list[dict[str, Any]] = []
    for segment_id in range(int(k_segments)):
        group = numeric.loc[numeric["segment_id"] == segment_id]
        if len(group) < 2:
            continue
        md = group["MD"].to_numpy(np.float64)
        fit_x = _linear_fit(md, group["X"].to_numpy(np.float64))
        fit_y = _linear_fit(md, group["Y"].to_numpy(np.float64))
        fit_u = _linear_fit(md, group["U"].to_numpy(np.float64))
        if fit_x is None or fit_y is None or fit_u is None:
            continue
        segment_mid_md = 0.5 * (edges[segment_id] + edges[segment_id + 1])
        center_x = fit_x[0] + fit_x[1] * segment_mid_md
        center_y = fit_y[0] + fit_y[1] * segment_mid_md
        values = np.array([center_x, center_y, fit_u[1], fit_x[1], fit_y[1]])
        if not np.isfinite(values).all():
            continue
        rows.append(
            {
                "donor_well_id": str(well),
                "segment_id": int(segment_id),
                "segment_mid_md": segment_mid_md,
                "observed_md_min": float(md.min()),
                "observed_md_max": float(md.max()),
                "finite_rows": int(len(group)),
                "center_x": center_x,
                "center_y": center_y,
                "u_rate": fit_u[1],
                "heading_x": fit_x[1],
                "heading_y": fit_y[1],
                "heading_norm": float(math.hypot(fit_x[1], fit_y[1])),
            }
        )
    return pd.DataFrame(rows)


def build_fold_safe_donor_ledger(
    raw_dir: Path,
    fold_assignment: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    k_segments = int(get_nested(config, "model.donor_segments.k_segments"))
    well_to_fold = fold_assignment.set_index("well_id")["fold"].astype(int).to_dict()
    base_frames: list[pd.DataFrame] = []
    wells = sorted(well_to_fold)
    for index, well in enumerate(wells, start=1):
        print(f"[donor {index}/{len(wells)}] well={well}", flush=True)
        frame = build_donor_segments_for_well(
            well,
            load_donor_horizontal_with_truth(well, raw_dir),
            k_segments=k_segments,
        )
        if frame.empty:
            raise ValueError(f"{well} produced no valid donor segments")
        frame["well_fold"] = int(well_to_fold[well])
        base_frames.append(frame)
    base = pd.concat(base_frames, ignore_index=True)
    fold_frames: list[pd.DataFrame] = []
    for outer_fold in sorted(int(value) for value in fold_assignment["fold"].unique()):
        eligible = base.loc[base["well_fold"] != outer_fold].copy()
        eligible.insert(0, "outer_fold", np.int8(outer_fold))
        fold_frames.append(eligible)
    ledger = pd.concat(fold_frames, ignore_index=True)
    ledger = ledger.sort_values(
        ["outer_fold", "donor_well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    if (ledger["outer_fold"].astype(int) == ledger["well_fold"].astype(int)).any():
        raise RuntimeError("outer-valid well entered its own fold donor ledger")
    expected_folds = set(int(value) for value in get_nested(config, "validation.expected_folds"))
    if set(ledger["outer_fold"].astype(int).unique()) != expected_folds:
        raise ValueError("donor ledger outer-fold coverage mismatch")
    return ledger


# %% [markdown]
# ## 6. Target-segment local-gradient prior construction


# %%
def exp209_initial_rate(horizontal: pd.DataFrame, tail_n: int = 30) -> dict[str, Any]:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_index = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_index) == 0 or not np.array_equal(
        known_index, np.arange(len(known_index), dtype=np.int64)
    ):
        raise ValueError("target must contain one contiguous visible TVT_input prefix")
    known = horizontal.iloc[known_index[-int(tail_n) :]]
    md = pd.to_numeric(known["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(known["Z"], errors="coerce").to_numpy(np.float64)
    tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(np.float64)
    dmd = np.diff(md)
    rate = (np.diff(tvt) + np.diff(z)) / dmd
    valid = np.isfinite(rate) & np.isfinite(dmd) & (dmd > 0.0)
    value = float(np.median(rate[valid])) if int(valid.sum()) >= 3 else 0.0
    return {
        "prefix_rate": value,
        "known_prefix_rows": int(len(known_index)),
        "valid_tail_rate_steps": int(valid.sum()),
        "fallback": bool(valid.sum() < 3),
        "last_known_index": int(known_index[-1]),
    }


def build_target_segments(
    well: str,
    horizontal: pd.DataFrame,
    fold: int,
    *,
    k_segments: int = 16,
) -> tuple[pd.DataFrame, np.ndarray]:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    eval_index = np.flatnonzero(~np.isfinite(tvt_input))
    if len(eval_index) == 0:
        raise ValueError(f"{well} has no unknown suffix")
    if not np.array_equal(eval_index, np.arange(eval_index[0], len(horizontal))):
        raise ValueError(f"{well} unknown TVT_input rows are not one suffix")
    suffix = horizontal.iloc[eval_index].copy()
    suffix_md = pd.to_numeric(suffix["MD"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(suffix_md).all() or not float(suffix_md.max()) > float(suffix_md.min()):
        raise ValueError(f"{well} suffix MD is invalid")
    segment_id = md_segment_ids(suffix_md, k_segments)
    edges = np.linspace(float(suffix_md.min()), float(suffix_md.max()), k_segments + 1)
    rows: list[dict[str, Any]] = []
    for segment in range(k_segments):
        mask = segment_id == segment
        group = suffix.iloc[np.flatnonzero(mask)]
        md_mid = 0.5 * (edges[segment] + edges[segment + 1])
        fit_x = _linear_fit(
            pd.to_numeric(group["MD"], errors="coerce").to_numpy(np.float64),
            pd.to_numeric(group["X"], errors="coerce").to_numpy(np.float64),
        )
        fit_y = _linear_fit(
            pd.to_numeric(group["MD"], errors="coerce").to_numpy(np.float64),
            pd.to_numeric(group["Y"], errors="coerce").to_numpy(np.float64),
        )
        valid = fit_x is not None and fit_y is not None
        rows.append(
            {
                "well_id": str(well),
                "fold": int(fold),
                "segment_id": int(segment),
                "segment_mid_md": md_mid,
                "suffix_rows": int(mask.sum()),
                "center_x": (fit_x[0] + fit_x[1] * md_mid) if fit_x else np.nan,
                "center_y": (fit_y[0] + fit_y[1] * md_mid) if fit_y else np.nan,
                "heading_x": fit_x[1] if fit_x else np.nan,
                "heading_y": fit_y[1] if fit_y else np.nan,
                "target_geometry_valid": bool(valid),
            }
        )
    return pd.DataFrame(rows), eval_index


def estimate_local_gradient_prior(
    target_segment: Mapping[str, Any],
    donors: pd.DataFrame,
    prefix_rate: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    local = dict(get_nested(config, "model.local_gradient", {}))
    record: dict[str, Any] = {
        "mu_rate": float(prefix_rate),
        "fallback": True,
        "fallback_reason": "target_geometry_invalid",
        "candidate_donor_wells": 0,
        "selected_donor_wells": 0,
        "effective_donors": 0.0,
        "nearest_distance_ft": np.nan,
        "directional_information": 0.0,
        "ridge_lambda": np.nan,
        "gradient_x": np.nan,
        "gradient_y": np.nan,
    }
    values = np.array(
        [
            target_segment["center_x"],
            target_segment["center_y"],
            target_segment["heading_x"],
            target_segment["heading_y"],
        ],
        dtype=np.float64,
    )
    if not bool(target_segment["target_geometry_valid"]) or not np.isfinite(values).all():
        return record
    target_xy = values[:2]
    target_heading = values[2:]
    target_norm = float(np.linalg.norm(target_heading))
    if target_norm < float(local["minimum_target_horizontal_speed_norm"]):
        record["fallback_reason"] = "target_heading_norm"
        return record

    work = donors.copy()
    work["distance_ft"] = np.hypot(
        work["center_x"].to_numpy(np.float64) - target_xy[0],
        work["center_y"].to_numpy(np.float64) - target_xy[1],
    )
    work = work.sort_values(
        ["donor_well_id", "distance_ft", "segment_id"], kind="mergesort"
    ).drop_duplicates("donor_well_id", keep="first")
    record["candidate_donor_wells"] = int(len(work))
    work = work.sort_values(["distance_ft", "donor_well_id", "segment_id"], kind="mergesort").head(
        int(local["n_neighbors"])
    )
    finite_columns = [
        "distance_ft",
        "u_rate",
        "heading_x",
        "heading_y",
    ]
    finite = np.isfinite(work[finite_columns].to_numpy(np.float64)).all(axis=1)
    work = work.loc[finite].reset_index(drop=True)
    record["selected_donor_wells"] = int(len(work))
    if len(work) < int(local["minimum_valid_donor_wells"]):
        record["fallback_reason"] = "donor_count"
        return record
    distance = work["distance_ft"].to_numpy(np.float64)
    record["nearest_distance_ft"] = float(distance.min())
    if record["nearest_distance_ft"] > float(local["maximum_nearest_distance_ft"]):
        record["fallback_reason"] = "nearest_distance"
        return record
    bandwidth = float(local["bandwidth_ft"])
    weights = np.exp(-0.5 * np.square(distance / bandwidth))
    weight_sum = float(weights.sum())
    effective = float(weight_sum**2 / float(weights @ weights))
    record["effective_donors"] = effective
    if effective < float(local["minimum_effective_donors"]):
        record["fallback_reason"] = "effective_donors"
        return record

    heading = work[["heading_x", "heading_y"]].to_numpy(np.float64)
    rate = work["u_rate"].to_numpy(np.float64)
    information = heading.T @ (weights[:, None] * heading)
    directional = float(
        target_heading @ information @ target_heading / (target_norm**2 * weight_sum)
    )
    record["directional_information"] = directional
    if directional < float(local["minimum_directional_information"]):
        record["fallback_reason"] = "directional_information"
        return record
    ridge = float(local["ridge_relative_trace"]) * max(
        float(np.trace(information) / 2.0),
        1.0e-12,
    )
    record["ridge_lambda"] = ridge
    gradient = np.linalg.solve(
        information + ridge * np.eye(2, dtype=np.float64),
        heading.T @ (weights * rate),
    )
    mu_rate = float(target_heading @ gradient)
    record["gradient_x"] = float(gradient[0])
    record["gradient_y"] = float(gradient[1])
    if not math.isfinite(mu_rate):
        record["fallback_reason"] = "non_finite_mu"
        return record
    if abs(mu_rate - prefix_rate) > float(local["maximum_abs_delta_from_prefix_rate"]):
        record["fallback_reason"] = "mu_prefix_delta"
        return record
    record.update(
        {
            "mu_rate": mu_rate,
            "fallback": False,
            "fallback_reason": "none",
        }
    )
    return record


@dataclass(frozen=True)
class FrozenPriorSchedule:
    donor_ledger: pd.DataFrame
    target_segments: pd.DataFrame
    rowwise_schedule: pd.DataFrame
    fold_assignment: pd.DataFrame
    donor_ledger_content_sha256: str
    target_segments_content_sha256: str
    rowwise_schedule_content_sha256: str


def build_and_freeze_prior_schedule(
    raw_dir: Path,
    fold_assignment: pd.DataFrame,
    donor_ledger: pd.DataFrame,
    config: Mapping[str, Any],
) -> FrozenPriorSchedule:
    fold_lookup = fold_assignment.set_index("well_id")["fold"].astype(int).to_dict()
    k_segments = int(get_nested(config, "model.target_segments.k_segments"))
    target_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    wells = sorted(fold_lookup)
    for index, well in enumerate(wells, start=1):
        print(f"[schedule {index}/{len(wells)}] well={well}", flush=True)
        fold = int(fold_lookup[well])
        horizontal = load_horizontal_without_truth(well, raw_dir)
        prefix = exp209_initial_rate(
            horizontal,
            int(get_nested(config, "model.prefix_rate.tail_steps")),
        )
        target, eval_index = build_target_segments(
            well,
            horizontal,
            fold,
            k_segments=k_segments,
        )
        donors = donor_ledger.loc[donor_ledger["outer_fold"].astype(int) == fold]
        if well in set(donors["donor_well_id"].astype(str)):
            raise RuntimeError(f"{well} leaked into its outer-valid donor set")
        prior_rows: list[dict[str, Any]] = []
        for target_row in target.to_dict(orient="records"):
            estimate = estimate_local_gradient_prior(
                target_row,
                donors,
                float(prefix["prefix_rate"]),
                config,
            )
            prior_rows.append({**target_row, **estimate, **prefix})
        target_prior = pd.DataFrame(prior_rows).sort_values("segment_id", kind="mergesort")
        suffix_md = pd.to_numeric(horizontal.iloc[eval_index]["MD"], errors="raise").to_numpy(
            np.float64
        )
        row_mu = np.interp(
            suffix_md,
            target_prior["segment_mid_md"].to_numpy(np.float64),
            target_prior["mu_rate"].to_numpy(np.float64),
        )
        if not np.isfinite(row_mu).all():
            raise RuntimeError(f"{well} rowwise prior contains non-finite values")
        segment_id = md_segment_ids(suffix_md, k_segments)
        last_md = float(pd.to_numeric(horizontal.iloc[int(prefix["last_known_index"])]["MD"]))
        schedule = pd.DataFrame(
            {
                "id": [f"{well}_{int(row)}" for row in eval_index],
                "well_id": str(well),
                "row_idx": eval_index.astype(np.int32),
                "suffix_offset": np.arange(len(eval_index), dtype=np.int32),
                "fold": np.int8(fold),
                "segment_id": segment_id,
                "md": suffix_md,
                "md_since": suffix_md - last_md,
                "mu_rate": row_mu,
                "prefix_rate": float(prefix["prefix_rate"]),
            }
        )
        target_frames.append(target_prior)
        schedule_frames.append(schedule)
    target_segments = (
        pd.concat(target_frames, ignore_index=True)
        .sort_values(["well_id", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    rowwise = (
        pd.concat(schedule_frames, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    if rowwise.duplicated(["well_id", "row_idx"]).any():
        raise RuntimeError("rowwise prior schedule contains duplicate identities")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(rowwise) != expected_rows:
        raise ValueError(f"rowwise prior schedule row mismatch: {len(rowwise)} != {expected_rows}")
    if rowwise["well_id"].nunique() != expected_wells:
        raise ValueError("rowwise prior schedule well coverage mismatch")
    if target_segments["well_id"].nunique() != expected_wells:
        raise ValueError("target-segment prior well coverage mismatch")
    if len(target_segments) != expected_wells * k_segments:
        raise ValueError("target-segment prior must contain exactly K anchors per well")
    if any(
        forbidden in rowwise.columns
        for forbidden in ("TVT", "tvt_true", "tvt_pred", "tvt_geop", "gr_delta")
    ):
        raise RuntimeError("forbidden truth or exp226 column entered the schedule")
    return FrozenPriorSchedule(
        donor_ledger=donor_ledger,
        target_segments=target_segments,
        rowwise_schedule=rowwise,
        fold_assignment=fold_assignment,
        donor_ledger_content_sha256=dataframe_content_sha256(donor_ledger),
        target_segments_content_sha256=dataframe_content_sha256(target_segments),
        rowwise_schedule_content_sha256=dataframe_content_sha256(rowwise),
    )


# %% [markdown]
# ## 7. Exp209 observation preparation


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
    hmm = dict(get_nested(config, "model.hmm", {}))
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
    prefix = exp209_initial_rate(
        horizontal,
        int(get_nested(config, "model.prefix_rate.tail_steps")),
    )
    prior_mu = schedule["mu_rate"].to_numpy(np.float64)
    residual_rates = np.linspace(
        -float(hmm["residual_rate_span"]),
        float(hmm["residual_rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    initial_residual_rate = float(prefix["prefix_rate"] - prior_mu[0])
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
        "prefix_rate": float(prefix["prefix_rate"]),
        "prior_mu": prior_mu,
        "initial_residual_rate": initial_residual_rate,
        "prefix_scale": scale,
    }


# %% [markdown]
# ## 8. Exact residual-rate HMM kernel and decoding


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
    hmm = dict(get_nested(config, "model.hmm", {}))
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
# ## 9. Prediction freeze and late truth/control attachment


# %%
@dataclass(frozen=True)
class FrozenPrediction:
    frame: pd.DataFrame
    runtime: pd.DataFrame
    prediction_content_sha256: str
    runtime_content_sha256: str
    truth_access_before_freeze: int = 0
    exp226_artifacts_resolved: int = 0


def generate_and_freeze_predictions(
    raw_dir: Path,
    frozen_schedule: FrozenPriorSchedule,
    config: Mapping[str, Any],
) -> FrozenPrediction:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exp362 exact HMM")
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
    if frozen.exp226_artifacts_resolved != 0:
        raise RuntimeError("an exp226 artifact was resolved")
    return frozen


def attach_truth_and_controls_after_freeze(
    frozen: FrozenPrediction,
    raw_dir: Path,
    parent_report: Mapping[str, Any],
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
    hidden, hidden_report = load_hidden_like_assignment(config)
    roles = dict(get_nested(config, "data.hidden_like.valid_role_columns"))
    hidden = hidden.rename(
        columns={
            roles["hidden_like_spatial"]: "hidden_like_spatial_role",
            roles["hidden_like_typewell_purged"]: "hidden_like_typewell_purged_role",
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
    finite = frame[["candidate_tvt", "parent_tvt", "true_tvt"]].to_numpy(np.float64)
    if len(frame) != len(frozen.frame) or not np.isfinite(finite).all():
        raise RuntimeError("late truth/control attachment failed identity or finite checks")
    return frame, {
        "prediction_content_sha256_before_truth": frozen.prediction_content_sha256,
        "truth_access_before_freeze": frozen.truth_access_before_freeze,
        "exp226_artifacts_resolved": frozen.exp226_artifacts_resolved,
        "truth_rows_attached": len(truth),
        "parent_rows_attached": len(parent),
        "hidden_like": hidden_report,
    }


# %% [markdown]
# ## 10. Metrics, scientific gate, and generated artifacts


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
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"metric scope is empty: {scope}")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = rmse(truth, selected["candidate_tvt"].to_numpy(np.float64))
    parent = rmse(truth, selected["parent_tvt"].to_numpy(np.float64))
    return {
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
    metrics = pd.DataFrame([paired_metric_row(frame, mask, scope=scope) for scope, mask in scopes])
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


def build_support_fallback_metrics(target_segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", target_segments)]
    scopes.extend(
        (f"fold_{fold}", group) for fold, group in target_segments.groupby("fold", sort=True)
    )
    for scope, group in scopes:
        reason_counts = group["fallback_reason"].astype(str).value_counts().sort_index().to_dict()
        rows.append(
            {
                "scope": scope,
                "segments": len(group),
                "wells": int(group["well_id"].nunique()),
                "fallback_segments": int(group["fallback"].astype(bool).sum()),
                "fallback_fraction": float(group["fallback"].astype(bool).mean()),
                "effective_donors_mean": float(group["effective_donors"].mean()),
                "nearest_distance_p95": float(
                    pd.to_numeric(group["nearest_distance_ft"], errors="coerce").quantile(0.95)
                ),
                "directional_information_mean": float(group["directional_information"].mean()),
                "fallback_reason_counts_json": json.dumps(reason_counts, sort_keys=True),
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
    runtime_seconds: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = dict(get_nested(config, "success_criteria.scientific_all_required"))
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    pooled = fold_metrics.loc[fold_metrics["scope"] == "pooled"].iloc[0]
    expected_parent_rmse = float(get_nested(config, "data.exp209_control.direct_rmse_ft"))
    parent_parity_abs_diff = abs(float(pooled["parent_rmse"]) - expected_parent_rmse)
    folds = fold_metrics.loc[fold_metrics["scope"].str.startswith("fold_")]
    improved_folds = int((folds["improvement_ft"] > 0.0).sum())

    distance_1000 = distance_metrics.loc[distance_metrics["scope"] == "distance_1000_plus"].iloc[0]
    spatial = hidden_metrics.loc[hidden_metrics["scope"] == "hidden_like_spatial"].iloc[0]
    typewell = hidden_metrics.loc[hidden_metrics["scope"] == "hidden_like_typewell_purged"].iloc[0]
    candidate_p95 = float(by_well["candidate_rmse"].quantile(0.95))
    parent_p95 = float(by_well["parent_rmse"].quantile(0.95))
    p95_delta = candidate_p95 - parent_p95
    worst_index = by_well["delta_rmse_candidate_minus_parent"].idxmax()
    worst = by_well.loc[worst_index]

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
        "exp226_artifacts_resolved": frozen_prediction.exp226_artifacts_resolved,
        "fold_donor_exclusion_passed": bool(
            (
                frozen_schedule.donor_ledger["outer_fold"].astype(int)
                != frozen_schedule.donor_ledger["well_fold"].astype(int)
            ).all()
        ),
        "hmm_well_runs": len(frozen_prediction.runtime),
        "expected_hmm_well_runs": int(get_nested(config, "execution_contract.hmm_well_runs")),
        "posterior_normalization_max_abs_error": float(
            frozen_prediction.runtime["posterior_row_sum_max_abs_error"].max()
        ),
        "parent_control_decompressed_sha256": parent_report["decompressed_sha256"],
        "expected_parent_control_decompressed_sha256": get_nested(
            config, "data.exp209_control.expected_hmm_cache_decompressed_sha256"
        ),
        "parent_rmse": float(pooled["parent_rmse"]),
        "expected_parent_rmse": expected_parent_rmse,
        "parent_rmse_absolute_difference": parent_parity_abs_diff,
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.kaggle.runtime_limit_seconds")),
        "schedule_content_sha256": frozen_schedule.rowwise_schedule_content_sha256,
        "prediction_content_sha256": frozen_prediction.prediction_content_sha256,
    }
    technical["passed"] = bool(
        technical["rows"] == expected_rows
        and technical["wells"] == expected_wells
        and technical["finite_prediction_coverage"] == 1.0
        and technical["duplicate_rows"] == 0
        and technical["truth_access_before_freeze"] == 0
        and technical["exp226_artifacts_resolved"] == 0
        and technical["fold_donor_exclusion_passed"]
        and technical["hmm_well_runs"] == technical["expected_hmm_well_runs"]
        and technical["posterior_normalization_max_abs_error"] <= 1.0e-6
        and technical["parent_control_decompressed_sha256"]
        == technical["expected_parent_control_decompressed_sha256"]
        and parent_parity_abs_diff <= 1.0e-5
        and runtime_seconds <= technical["runtime_limit_seconds"]
    )

    scientific = {
        "candidate_rmse": float(pooled["candidate_rmse"]),
        "parent_rmse": float(pooled["parent_rmse"]),
        "improvement_ft": float(pooled["improvement_ft"]),
        "minimum_improvement_ft": float(criteria["minimum_direct_rmse_gain_vs_exp209_ft"]),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(criteria["minimum_improved_folds"]),
        "distance_1000_plus_delta": float(distance_1000["delta_rmse_candidate_minus_parent"]),
        "maximum_1000_plus_regression_ft": float(criteria["maximum_1000_plus_regression_ft"]),
        "hidden_like_spatial_delta": float(spatial["delta_rmse_candidate_minus_parent"]),
        "maximum_hidden_like_spatial_regression_ft": float(
            criteria["maximum_hidden_like_spatial_regression_ft"]
        ),
        "hidden_like_typewell_purged_delta": float(typewell["delta_rmse_candidate_minus_parent"]),
        "maximum_hidden_like_typewell_purged_regression_ft": float(
            criteria["maximum_hidden_like_typewell_purged_regression_ft"]
        ),
        "by_well_p95_delta": p95_delta,
        "maximum_by_well_p95_regression_ft": float(criteria["maximum_by_well_p95_regression_ft"]),
        "worst_well_id": str(worst["well_id"]),
        "worst_well_delta": float(worst["delta_rmse_candidate_minus_parent"]),
        "maximum_worst_well_regression_ft": float(criteria["maximum_worst_well_regression_ft"]),
    }
    scientific["passed"] = bool(
        scientific["improvement_ft"] >= scientific["minimum_improvement_ft"]
        and scientific["improved_folds"] >= scientific["minimum_improved_folds"]
        and scientific["distance_1000_plus_delta"] <= scientific["maximum_1000_plus_regression_ft"]
        and scientific["hidden_like_spatial_delta"]
        <= scientific["maximum_hidden_like_spatial_regression_ft"]
        and scientific["hidden_like_typewell_purged_delta"]
        <= scientific["maximum_hidden_like_typewell_purged_regression_ft"]
        and scientific["by_well_p95_delta"] <= scientific["maximum_by_well_p95_regression_ft"]
        and scientific["worst_well_delta"] <= scientific["maximum_worst_well_regression_ft"]
    )
    passed = bool(technical["passed"] and scientific["passed"])
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            "segment_local_donor_slope_exact_hmm_passed_train_side_only"
            if passed
            else "segment_local_donor_slope_exact_hmm_failed_close_without_rescue"
        ),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "failure_action": (
            "close_without_k_neighbor_bandwidth_ridge_fallback_hmm_parameter_"
            "blend_selector_inference_or_submission_rescue"
        ),
    }


def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.is_dir() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp362 must run first on Kaggle; local execution requires explicit approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.perf_counter()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest, raw_report = validate_raw_well_identity(config, raw_dir)
    fold_assignment = stable_fold_assignment(
        raw_manifest["well_id"].astype(str).tolist(),
        seed=int(get_nested(config, "validation.seed")),
        n_folds=int(get_nested(config, "validation.n_folds")),
    )
    parent_report = validate_parent_control_dependency(config)
    hidden_like_preflight = preflight_hidden_like_dependency(config)
    scientific_contract = build_scientific_contract(config)
    contract_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json",
        scientific_contract,
    )
    raw_input_manifest = {
        "truth_attached": False,
        "raw_train": raw_report,
        "parent_control": parent_report,
        "hidden_like_dependency": hidden_like_preflight,
        "forbidden_exp226_artifacts_resolved": 0,
    }
    raw_input_manifest["raw_input_manifest_sha256"] = mapping_sha256(raw_input_manifest)
    input_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_raw_input_manifest.json",
        raw_input_manifest,
    )
    fold_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_fold_assignment.csv",
        fold_assignment,
    )

    donor_ledger = build_fold_safe_donor_ledger(raw_dir, fold_assignment, config)
    frozen_schedule = build_and_freeze_prior_schedule(
        raw_dir,
        fold_assignment,
        donor_ledger,
        config,
    )
    donor_report = write_gzip_csv(
        artifacts / f"{OUTPUT_PREFIX}_donor_segment_ledger.csv.gz",
        frozen_schedule.donor_ledger,
    )
    target_report = write_gzip_csv(
        artifacts / f"{OUTPUT_PREFIX}_target_segment_prior.csv.gz",
        frozen_schedule.target_segments,
    )
    schedule_report = write_gzip_csv(
        artifacts / f"{OUTPUT_PREFIX}_rowwise_prior_schedule.csv.gz",
        frozen_schedule.rowwise_schedule,
    )
    freeze_manifest = {
        "truth_attached": False,
        "truth_access_before_freeze": 0,
        "exp226_artifacts_resolved": 0,
        "raw_identity_sha256": raw_report["content_sha256"],
        "fold_assignment_sha256": fold_report["content_sha256"],
        "donor_segment_ledger_logical_sha256": (frozen_schedule.donor_ledger_content_sha256),
        "donor_segment_ledger_decompressed_sha256": donor_report["decompressed_sha256"],
        "target_segment_prior_logical_sha256": (frozen_schedule.target_segments_content_sha256),
        "target_segment_prior_decompressed_sha256": target_report["decompressed_sha256"],
        "rowwise_prior_schedule_logical_sha256": (frozen_schedule.rowwise_schedule_content_sha256),
        "rowwise_prior_schedule_decompressed_sha256": schedule_report["decompressed_sha256"],
    }
    freeze_manifest["freeze_manifest_sha256"] = mapping_sha256(freeze_manifest)
    freeze_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_freeze_manifest.json",
        freeze_manifest,
    )

    frozen_prediction = generate_and_freeze_predictions(
        raw_dir,
        frozen_schedule,
        config,
    )
    prediction_report = write_gzip_csv(
        artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz",
        frozen_prediction.frame,
    )
    runtime_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_by_well_runtime.csv",
        frozen_prediction.runtime,
    )
    prediction_freeze_seconds = time.perf_counter() - started

    frame, late_attachment = attach_truth_and_controls_after_freeze(
        frozen_prediction,
        raw_dir,
        parent_report,
        config,
    )
    fold_metrics, distance_metrics, hidden_metrics = build_scope_metrics(frame, config)
    by_well = build_by_well_metrics(frame)
    support = build_support_fallback_metrics(frozen_schedule.target_segments)
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
        runtime_seconds,
        config,
    )

    fold_metrics_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        fold_metrics,
    )
    distance_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv",
        distance_metrics,
    )
    hidden_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv",
        hidden_metrics,
    )
    by_well_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        by_well,
    )
    support_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_support_fallback_metrics.csv",
        support,
    )
    gate_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
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
                "raw_input_manifest": input_report,
                "fold_assignment": fold_report,
                "donor_segment_ledger": donor_report,
                "target_segment_prior": target_report,
                "rowwise_prior_schedule": schedule_report,
                "freeze_manifest": freeze_report,
                "oof_predictions": prediction_report,
                "runtime": runtime_report,
                "fold_metrics": fold_metrics_report,
                "distance_metrics": distance_report,
                "hidden_like_metrics": hidden_report,
                "by_well_metrics": by_well_report,
                "support_fallback_metrics": support_report,
                "promotion_gate": gate_report,
            }.items()
        ]
    )
    sha_report = write_csv(
        artifacts / f"{OUTPUT_PREFIX}_sha_manifest.csv",
        sha_manifest,
    )
    status = (
        "train_side_gate_passed_no_automatic_downstream"
        if gate["passed"]
        else "train_side_gate_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "prediction_frozen_at_seconds": prediction_freeze_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "execution_counts": validate_scientific_contract(config),
        "truth_attachment": late_attachment,
        "promotion_gate": gate,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "raw_input_manifest_sha256": raw_input_manifest["raw_input_manifest_sha256"],
        "schedule_logical_sha256": (frozen_schedule.rowwise_schedule_content_sha256),
        "prediction_logical_sha256": (frozen_prediction.prediction_content_sha256),
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_id": get_nested(config, "execution.kaggle_kernel_id"),
            "kernel_version": None,
            "kernel_version_recording": "record_after_kaggle_run",
        },
        "models": 0,
        "boosters": 0,
        "control_reruns": 0,
        "inference_enabled": False,
        "submission_created": False,
        "model_sha256": None,
        "submission_sha256": None,
        "sha_manifest": sha_report,
    }
    summary_report = write_json(
        artifacts / f"{OUTPUT_PREFIX}_summary.json",
        summary,
    )
    pooled = fold_metrics.loc[fold_metrics["scope"] == "pooled"].iloc[0]
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": float(pooled["candidate_rmse"]) if gate["passed"] else None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": pooled.to_dict(),
        "promotion_gate": gate,
        "schedule_sha256": frozen_schedule.rowwise_schedule_content_sha256,
        "prediction_sha256": frozen_prediction.prediction_content_sha256,
        "model_sha256": None,
        "submission_sha256": None,
        "summary_path": summary_report["path"],
        "notes": (
            "Train-side exact-HMM only; no raw-test inference, model, blend, "
            "selector, or submission is produced."
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
    print("Variant:", get_nested(CONFIG, "model.active_variants"))
    print("Execution contract:", json.dumps(CONTRACT_COUNTS, sort_keys=True))
    print("Kaggle push approved:", get_nested(CONFIG, "execution.kaggle_push_approved"))
    print("Run HMM:", get_nested(CONFIG, "execution.run_hmm"))
    print("Inference enabled:", get_nested(CONFIG, "inference.enabled"))


# %% [markdown]
# ## 12. Kaggle CPU execution

# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_hmm")):
        EXP362_SUMMARY = run_full_experiment(CONFIG)
        print(json.dumps(to_jsonable(EXP362_SUMMARY), indent=2, sort_keys=True))
    else:
        print(
            "Implementation is ready. Kaggle HMM execution remains disabled "
            "until a separate run approval updates config.yaml."
        )
