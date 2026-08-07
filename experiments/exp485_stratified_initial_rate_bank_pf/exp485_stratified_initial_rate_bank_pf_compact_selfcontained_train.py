# %% [markdown]
# # exp485 stratified initial-rate bank likelihood-PF — train
#
# This train-side experiment changes one exp404/exp417 likelihood-PF factor:
# the single tail-30 initial U-rate center becomes one equal-strata mixture over
# tail-30/32/64/128/256 centers. Stage 0 is a fixed32 target-free technical and
# mechanism preflight, not CV. Separately approved Stage 1 is the all-773-well
# truth-late train-side CV under a recorded user-approved runtime exception.

# %% [markdown]
# ## Contents
# 1. Imports and notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Fixed32 scope and target-free raw inputs
# 5. Initial-rate bank and exp404 likelihood-PF inputs
# 6. Stratified initial-rate likelihood-PF kernel
# 7. Synthetic allocation, fallback, and exp404 parity contracts
# 8. Target-free candidate generation and freeze
# 9. Fail-closed Stage 0 gates
# 10. Generated artifacts and Stage 0 orchestration
# 11. All-well Stage 1 truth-late CV and promotion gate
# 12. Setup and configuration preview

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


EXPERIMENT_NAME = "exp485_stratified_initial_rate_bank_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
PRIMARY_CANDIDATE = "likpf_scale5_equal_strata_initial_rate_bank5"
PREDICTION_COLUMNS = (PRIMARY_CANDIDATE,)
COMPONENT_NAMES = ("tail30", "w32", "w64", "w128", "w256")
RATE_WINDOWS = (30, 32, 64, 128, 256)
CHECKPOINT_LABELS = ("row_0", "row_32", "row_128", "row_512", "final")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP485_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Notebook-safe configuration, path, and SHA helpers


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


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def candidate_package_dirs() -> list[Path]:
    root = project_root()
    candidates = [
        Path.cwd(),
        root / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    return candidates


def load_experiment_config(package_dir: Path | None = None) -> dict[str, Any]:
    candidates = [package_dir] if package_dir is not None else candidate_package_dirs()
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "config.yaml"
        checked.append(str(path))
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp485 config not found; checked={checked}")


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


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_csv_payload(path: str | Path) -> str:
    selected = Path(path)
    return (
        sha256_decompressed_csv(selected)
        if selected.suffix == ".gz"
        else sha256_path(selected)
    )


def read_selected_csv_with_content_sha(
    path: str | Path,
    columns: Sequence[str],
    *,
    numeric_columns: Sequence[str] = (),
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered_columns = [str(column) for column in columns]
    numeric = {str(column) for column in numeric_columns}
    digest = hashlib.sha256()
    chunks: list[pd.DataFrame] = []
    rows = 0
    write_header = True
    for chunk in pd.read_csv(
        path,
        usecols=ordered_columns,
        dtype={"id": str},
        chunksize=chunksize,
    ):
        chunk = chunk.loc[:, ordered_columns]
        for column in numeric:
            chunk[column] = pd.to_numeric(chunk[column], errors="raise")
        payload = chunk.to_csv(
            index=False,
            header=write_header,
            lineterminator="\n",
            float_format="%.17g",
        ).encode("utf-8")
        digest.update(payload)
        write_header = False
        rows += len(chunk)
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"selected CSV input is empty: {path}")
    frame = pd.concat(chunks, ignore_index=True)
    return frame, {
        "path": str(path),
        "columns": ordered_columns,
        "rows": rows,
        "selected_columns_sha256": digest.hexdigest(),
    }


def dataframe_content_sha(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return mapping_sha256(schema)


def typed_dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
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


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "raw_sha256": sha256_path(path)}


def write_deterministic_gzip_csv(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            frame.to_csv(zipped, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": frame.columns.astype(str).tolist(),
        "schema_sha256": dataframe_schema_sha(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
    }


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if platform.system() != "Darwin" else 1024.0**3
    return value / divisor


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "numba_available": str(NUMBA_AVAILABLE),
    }
    if NUMBA_AVAILABLE:
        import numba

        versions["numba"] = numba.__version__
    return versions


def resolve_existing(
    filename: str,
    candidates: Iterable[str],
    patterns: Iterable[str] = (),
) -> Path:
    checked: list[str] = []
    for raw in candidates:
        root = Path(raw)
        options = [root] if root.name == filename else [root / filename]
        for option in options:
            checked.append(str(option))
            if option.exists():
                return option
        if root.exists():
            for pattern in patterns:
                for option in sorted(root.glob(pattern)):
                    checked.append(str(option))
                    if option.is_file():
                        return option
    if KAGGLE_INPUT_ROOT.exists():
        for pattern in patterns:
            for option in sorted(KAGGLE_INPUT_ROOT.glob(pattern)):
                checked.append(str(option))
                if option.is_file():
                    return option
    raise FileNotFoundError(f"{filename} was not found; checked={checked[:12]}")


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = [
        Path.cwd() / "assets" / filename,
        project_root() / local_path,
        KAGGLE_WORKING_ROOT / "assets" / filename,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    matches = [candidate for candidate in candidates if candidate.exists()]
    if not matches:
        raise FileNotFoundError(f"bootstrap asset not found: {filename}")
    return matches[0]


# %% [markdown]
# ## 3. Frozen scientific and execution contracts


# %%
def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, int]:
    counts = {
        "active_variants": int(get_nested(config, "execution.active_variants")),
        "stage_0_candidate_pf_well_runs": int(
            get_nested(config, "execution.stage_0_candidate_pf_well_runs")
        ),
        "stage_0_seed_well_trajectories": int(
            get_nested(config, "execution.stage_0_seed_well_trajectories")
        ),
        "stage_0_particle_starts": int(
            get_nested(config, "execution.stage_0_particle_starts")
        ),
        "stage_1_candidate_pf_well_runs": int(
            get_nested(config, "execution.stage_1_candidate_pf_well_runs")
        ),
        "stage_1_seed_well_trajectories": int(
            get_nested(config, "execution.stage_1_seed_well_trajectories")
        ),
        "stage_1_particle_starts": int(
            get_nested(config, "execution.stage_1_particle_starts")
        ),
        "control_pf_well_runs": int(
            get_nested(config, "execution.control_pf_well_runs")
        ),
        "lightgbm_configs": int(get_nested(config, "execution.lightgbm_configs")),
        "trained_folds": int(get_nested(config, "execution.trained_folds")),
        "boosters": int(get_nested(config, "execution.boosters")),
        "hmm_well_runs": int(get_nested(config, "execution.hmm_well_runs")),
        "beam_well_runs": int(get_nested(config, "execution.beam_well_runs")),
        "gpu_runs": int(get_nested(config, "execution.gpu_runs")),
    }
    expected = {
        "active_variants": 1,
        "stage_0_candidate_pf_well_runs": 32,
        "stage_0_seed_well_trajectories": 4096,
        "stage_0_particle_starts": 2_048_000,
        "stage_1_candidate_pf_well_runs": 773,
        "stage_1_seed_well_trajectories": 98_944,
        "stage_1_particle_starts": 49_472_000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    if counts != expected:
        raise ValueError(f"exp485 execution count contract changed: {counts}")
    run_stage0 = bool(get_nested(config, "execution.run_stage_0"))
    run_stage1 = bool(get_nested(config, "execution.run_stage_1"))
    if run_stage0 and run_stage1:
        raise ValueError("exp485 permits exactly one active execution stage")
    if run_stage1:
        if not bool(
            get_nested(config, "stage_0_result.all_non_runtime_gates_passed")
        ):
            raise RuntimeError("exp485 Stage 1 requires all non-runtime Stage 0 gates")
        if not bool(get_nested(config, "stage_0_result.runtime_exception.approved")):
            raise RuntimeError("exp485 Stage 1 requires the recorded runtime exception")
        if bool(get_nested(config, "data.stage1_resume.enabled")) and not bool(
            get_nested(config, "execution.stage_1_resume_from_version_2_approved")
        ):
            raise RuntimeError("exp485 Stage 1 resume is not approved")
    if require_run_approval:
        if not bool(get_nested(config, "implementation.enabled")):
            raise RuntimeError("exp485 implementation is disabled")
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp485 Kaggle push is not approved")
        if run_stage0 and not bool(
            get_nested(config, "execution.stage_0_execution_approved")
        ):
            raise RuntimeError("exp485 Stage 0 is not approved")
        if run_stage1 and not bool(
            get_nested(config, "execution.stage_1_execution_approved")
        ):
            raise RuntimeError("exp485 Stage 1 is not approved")
        if not (run_stage0 or run_stage1):
            raise RuntimeError("exp485 has no approved execution stage selected")
    return counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    changed = dict(get_nested(config, "model.changed_factor") or {})
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    return {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "implementation_reference": get_nested(
            config, "lineage.exact_pf_implementation_reference"
        ),
        "candidate": get_nested(config, "validation.primary_candidate"),
        "changed_factor": changed,
        "fixed_from_exp404": fixed,
        "stage0": get_nested(config, "stages.stage_0"),
        "stage1": get_nested(config, "stages.stage_1"),
        "saved_control": {
            "source": get_nested(config, "data.saved_control.source"),
            "rerun": False,
        },
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
        "forbidden": get_nested(config, "guards.forbidden"),
    }


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool,
) -> dict[str, Any]:
    changed = dict(get_nested(config, "model.changed_factor") or {})
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    expected = {
        "windows_rows": list(RATE_WINDOWS),
        "component_order": list(COMPONENT_NAMES),
        "minimum_valid_steps": 3,
        "fallback_rate": 0.0,
        "particles_per_component": 100,
        "component_assignment": "particle_index_modulo_5",
        "within_component_rate_spread": 0.01,
        "duplicate_center_policy": "retain_all_equal_strata_without_deduplication",
        "component_label_use": "diagnostic_only",
    }
    for key, value in expected.items():
        if changed.get(key) != value:
            raise ValueError(f"exp485 changed-factor contract mismatch: {key}")
    required_fixed = {
        "particles": 500,
        "seeds": 128,
        "primary_seed_weighting_temperature": 5.0,
        "momentum": 0.998,
        "rate_noise": 0.002,
        "position_noise": 0.005,
        "rough_position": 0.1,
        "rough_rate": 0.001,
        "resample_threshold_fraction": 0.5,
        "initial_position_spread_ft": 4.5,
        "typewell_grid_step_ft": 0.2,
    }
    for key, value in required_fixed.items():
        if fixed.get(key) != value:
            raise ValueError(f"exp485 exp404 fixed contract mismatch: {key}")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp485 route must remain pf_beam")
    if get_nested(config, "model.active_variants") != ["equal_strata_rate_bank5"]:
        raise ValueError("exp485 must contain exactly one scientific variant")
    if int(get_nested(config, "stages.stage_0.candidate_pf_well_runs")) != 32:
        raise ValueError("exp485 Stage 0 must contain exactly 32 PF well-runs")
    validate_execution_contract(
        config,
        require_run_approval=require_run_approval,
    )
    contract = build_scientific_contract(config)
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Fixed32 scope and target-free raw inputs
#
# Before the prediction/rate-bank/component freeze, only the `well` column of
# the SHA-fixed manifest and the raw `MD/Z/GR/TVT_input` plus Type Well are read.
# Suffix `TVT`, error, fold, hidden-like role, and saved control are never read
# by Stage 0.


# %%
def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_path(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(
            "exp485 fixed32 manifest SHA mismatch: "
            f"expected={spec['expected_sha256']}, observed={observed}"
        )
    return path


def load_fixed32_scope(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    path = fixed32_manifest_path(config)
    scope = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    expected = int(get_nested(config, "stages.stage_0.candidate_pf_well_runs"))
    if len(scope) != expected or scope["well"].nunique() != expected:
        raise ValueError("exp485 fixed32 scope identity changed")
    wells = scope["well"].astype(str).tolist()
    return wells, {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "wells": len(wells),
        "columns_read_before_freeze": ["well"],
        "well_order_sha256": mapping_sha256(wells),
    }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    for column in ("MD", "Z", "GR", "TVT_input"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["MD"].notna().all() or not frame["Z"].notna().all():
        raise ValueError(f"{well}: MD/Z contains missing values")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = (
        frame.dropna(subset=["TVT"])
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(frame) < 2 or not np.isfinite(frame["TVT"].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    typewell_mean = float(frame["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(typewell_mean)
    return frame


# %% [markdown]
# ## 5. Initial-rate bank and exp404 likelihood-PF inputs


# %%
def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int,
    *,
    minimum_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int, bool]:
    if int(window_rows) <= 1:
        raise ValueError("initial-rate window must be greater than one")
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    delta_tvt = np.diff(tvt)
    delta_z = np.diff(z)
    delta_md = np.diff(md)
    valid = (
        np.isfinite(delta_tvt)
        & np.isfinite(delta_z)
        & np.isfinite(delta_md)
        & (delta_md > 0.0)
    )
    valid_steps = int(valid.sum())
    if valid_steps < int(minimum_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps, True
    rate = float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))
    if not math.isfinite(rate):
        return float(fallback_rate), int(len(tail)), valid_steps, True
    return rate, int(len(tail)), valid_steps, False


def initial_rate_bank(
    horizontal: pd.DataFrame,
    *,
    windows: Sequence[int] = RATE_WINDOWS,
    minimum_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    known = horizontal.loc[horizontal["TVT_input"].notna()].copy()
    if known.empty:
        raise ValueError("initial-rate bank requires a non-empty known prefix")
    rows: list[dict[str, Any]] = []
    rates: list[float] = []
    for component_index, (component, window) in enumerate(
        zip(COMPONENT_NAMES, windows, strict=True)
    ):
        rate, effective_rows, valid_steps, used_fallback = robust_initial_rate(
            known,
            int(window),
            minimum_valid_steps=int(minimum_valid_steps),
            fallback_rate=float(fallback_rate),
        )
        rates.append(rate)
        rows.append(
            {
                "component_index": component_index,
                "component": component,
                "window_rows": int(window),
                "center_value": rate,
                "effective_rows": effective_rows,
                "valid_steps": valid_steps,
                "used_fallback": used_fallback,
            }
        )
    values = np.asarray(rates, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("initial-rate bank contains non-finite centers")
    diagnostics = pd.DataFrame(rows)
    diagnostics["unique_center_count"] = int(np.unique(values).size)
    diagnostics["center_range"] = float(np.max(values) - np.min(values))
    diagnostics["fallback_count"] = int(diagnostics["used_fallback"].sum())
    return values, diagnostics


def uniform_typewell_grid(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step: float,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(typewell_tvt))
    maximum = float(np.max(typewell_tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    grid_gr = np.interp(grid_tvt, typewell_tvt, typewell_gr).astype(np.float64)
    return grid_gr, minimum, float(step)


def exp404_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires at least one known-prefix row")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    raw_scale = float(np.nanstd(residual))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    base_scale = float(np.clip(raw_scale, 10.0, 60.0))
    return {
        "raw_scale": raw_scale,
        "base_scale": base_scale,
        "candidate_scale": base_scale,
        "multiplier": 1.0,
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
    }


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    windows: Sequence[int] = RATE_WINDOWS,
    minimum_valid_steps: int = 3,
    fallback_rate: float = 0.0,
    grid_step: float = 0.2,
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires a known prefix and unknown suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=float(grid_step),
    )
    scale_audit = exp404_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    rate_centers, rate_diagnostics = initial_rate_bank(
        horizontal,
        windows=windows,
        minimum_valid_steps=minimum_valid_steps,
        fallback_rate=fallback_rate,
    )
    typewell_mean = float(typewell_gr.mean())
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_z = evaluation["Z"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    raw_gr_observed = evaluation["GR"].notna().to_numpy(bool)
    if not (
        np.isfinite(eval_md).all()
        and np.isfinite(eval_z).all()
        and np.isfinite(eval_gr).all()
    ):
        raise ValueError("likelihood-PF evaluation inputs are not finite")
    last_known_tvt = float(last_known["TVT_input"])
    last_known_md = float(last_known["MD"])
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": eval_z,
        "eval_gr": eval_gr,
        "raw_gr_observed": raw_gr_observed,
        "md_since": eval_md - last_known_md,
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_known_tvt + float(last_known["Z"]),
        "initial_rate_centers": rate_centers,
        "rate_diagnostics": rate_diagnostics,
        "scale_audit": scale_audit,
    }


# %% [markdown]
# ## 6. Stratified initial-rate likelihood-PF kernel
#
# Particle `i` always starts in component `i % 5`. Component labels are copied
# by resampling and never alter likelihood, dynamics, weights, roughening, or
# output. Filtered component mass is captured before resampling; surviving
# counts are captured after any resampling at the same row.


# %%
def stratified_component_ids(
    particles: int,
    *,
    components: int = 5,
) -> np.ndarray:
    if particles <= 0 or particles % components != 0:
        raise ValueError("particles must be positive and divisible by components")
    return np.arange(particles, dtype=np.int64) % int(components)


def checkpoint_indices(rows: int) -> np.ndarray:
    if rows <= 0:
        raise ValueError("checkpoint rows must be positive")
    return np.asarray(
        [0, min(32, rows - 1), min(128, rows - 1), min(512, rows - 1), rows - 1],
        dtype=np.int64,
    )


@njit(cache=True)
def _interp1(grid: np.ndarray, value: float, minimum: float, step: float) -> float:
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    final = len(grid) - 1
    if index >= final:
        return grid[final]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _pf_stratified_rate_bank_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_position: float,
    initial_rate_centers: np.ndarray,
    checkpoints: np.ndarray,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_position_spread: float,
    initial_rate_spread: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = len(md_v)
    components = len(initial_rate_centers)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    initial_counts = np.zeros((seeds, components), np.int64)
    filtered_mass = np.zeros((seeds, len(checkpoints), components))
    surviving_counts = np.zeros((seeds, len(checkpoints), components), np.int64)
    first_extinction_rows = np.full((seeds, components), -1, np.int64)
    first_resample_rows = np.full(seeds, -1, np.int64)
    first_resample_ess = np.full(seeds, np.nan)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step

    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        component = np.empty(particles, np.int64)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            label = particle % components
            component[particle] = label
            initial_counts[seed_index, label] += 1
            position[particle] = (
                last_position + initial_position_spread * np.random.randn()
            )
            rate[particle] = (
                initial_rate_centers[label]
                + initial_rate_spread * np.random.randn()
            )

        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = (
                    momentum * rate[particle] + rate_noise * np.random.randn()
                )
                position[particle] += (
                    rate[particle] * delta_md + position_noise * np.random.randn()
                )
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                    position_clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                    position_clip_counts[seed_index] += 1
                position[particle] = tvt_value + z_v[row]

            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(
                    grid_gr,
                    position[particle] - z_v[row],
                    grid_minimum,
                    grid_step,
                )
                zscore = (gr_v[row] - expected_gr) / gr_scale
                squared = zscore * zscore
                if squared > 600.0:
                    squared = 600.0
                likelihood = np.exp(-0.5 * squared)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)

            weight_sum = 0.0
            for particle in range(particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(particles):
                    weights[particle] = 1.0 / particles

            inverse_ess = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
            effective_sample_size = 1.0 / inverse_ess
            if effective_sample_size < minimum_ess[seed_index]:
                minimum_ess[seed_index] = effective_sample_size

            for checkpoint_index in range(len(checkpoints)):
                if row == checkpoints[checkpoint_index]:
                    for particle in range(particles):
                        filtered_mass[
                            seed_index,
                            checkpoint_index,
                            component[particle],
                        ] += weights[particle]

            if effective_sample_size < resample_fraction * particles:
                if first_resample_rows[seed_index] < 0:
                    first_resample_rows[seed_index] = row
                    first_resample_ess[seed_index] = effective_sample_size
                cumulative = np.empty(particles)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                new_component = np.empty(particles, np.int64)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = rate[cursor] + rough_rate * np.random.randn()
                    new_component[particle] = component[cursor]
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    component[particle] = new_component[particle]
                    weights[particle] = 1.0 / particles
                resampling_counts[seed_index] += 1

            current_counts = np.zeros(components, np.int64)
            for particle in range(particles):
                current_counts[component[particle]] += 1
            for label in range(components):
                if (
                    current_counts[label] == 0
                    and first_extinction_rows[seed_index, label] < 0
                ):
                    first_extinction_rows[seed_index, label] = row
            for checkpoint_index in range(len(checkpoints)):
                if row == checkpoints[checkpoint_index]:
                    for label in range(components):
                        surviving_counts[
                            seed_index,
                            checkpoint_index,
                            label,
                        ] = current_counts[label]

            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (position[particle] - z_v[row])
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood

    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        initial_counts,
        filtered_mass,
        surviving_counts,
        first_extinction_rows,
        first_resample_rows,
        first_resample_ess,
    )


def aggregate_seed_predictions(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / float(temperature))
    weights /= weights.sum()
    return (weights[:, None] * predictions).sum(axis=0), weights


def component_ledger_frame(
    *,
    well: str,
    checkpoints: np.ndarray,
    seed_weights: np.ndarray,
    filtered_mass: np.ndarray,
    surviving_counts: np.ndarray,
    first_extinction_rows: np.ndarray,
    first_resample_rows: np.ndarray,
    first_resample_ess: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    particles = int(surviving_counts.shape[2] * 100)
    for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
        for component_index, component in enumerate(COMPONENT_NAMES):
            extinction = first_extinction_rows[:, component_index]
            rows.append(
                {
                    "well_id": str(well),
                    "checkpoint": checkpoint_label,
                    "checkpoint_row": int(checkpoints[checkpoint_index]),
                    "component_index": component_index,
                    "component": component,
                    "filtered_posterior_mass_mean": float(
                        filtered_mass[:, checkpoint_index, component_index].mean()
                    ),
                    "filtered_posterior_mass_temperature5": float(
                        np.dot(
                            seed_weights,
                            filtered_mass[:, checkpoint_index, component_index],
                        )
                    ),
                    "surviving_particle_count_mean": float(
                        surviving_counts[:, checkpoint_index, component_index].mean()
                    ),
                    "surviving_particle_fraction_mean": float(
                        surviving_counts[:, checkpoint_index, component_index].mean()
                        / particles
                    ),
                    "extinct_seed_fraction": float(np.mean(extinction >= 0)),
                    "first_extinction_row_mean_when_extinct": (
                        float(extinction[extinction >= 0].mean())
                        if np.any(extinction >= 0)
                        else np.nan
                    ),
                    "first_resample_row_mean_when_present": (
                        float(first_resample_rows[first_resample_rows >= 0].mean())
                        if np.any(first_resample_rows >= 0)
                        else np.nan
                    ),
                    "ess_before_first_resample_mean": (
                        float(np.nanmean(first_resample_ess))
                        if np.isfinite(first_resample_ess).any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_stratified_rate_bank_pf(
    prepared: Mapping[str, Any],
    *,
    well: str,
    particles: int,
    seeds: int,
    seed_base: int,
    temperature: float,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_position_spread: float,
    initial_rate_spread: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    started = time.time()
    checkpoints = checkpoint_indices(len(prepared["eval_md"]))
    (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        initial_counts,
        filtered_mass,
        surviving_counts,
        first_extinction_rows,
        first_resample_rows,
        first_resample_ess,
    ) = _pf_stratified_rate_bank_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        float(prepared["last_known_position"]),
        np.asarray(prepared["initial_rate_centers"], dtype=np.float64),
        checkpoints,
        int(particles),
        int(seeds),
        int(seed_base),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_position_spread),
        float(initial_rate_spread),
    )
    candidate, seed_weights = aggregate_seed_predictions(
        predictions,
        log_likelihoods,
        temperature=float(temperature),
    )
    ledger = component_ledger_frame(
        well=well,
        checkpoints=checkpoints,
        seed_weights=seed_weights,
        filtered_mass=filtered_mass,
        surviving_counts=surviving_counts,
        first_extinction_rows=first_extinction_rows,
        first_resample_rows=first_resample_rows,
        first_resample_ess=first_resample_ess,
    )
    expected_counts = np.full((seeds, 5), particles // 5, dtype=np.int64)
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "initial_component_counts_contract": bool(
            np.array_equal(initial_counts, expected_counts)
        ),
        "initial_component_count_min": int(initial_counts.min()),
        "initial_component_count_max": int(initial_counts.max()),
        "seed_loglik_mean_per_row": float(log_likelihoods.mean())
        / len(prepared["eval_md"]),
        "seed_loglik_best_per_row": float(log_likelihoods.max())
        / len(prepared["eval_md"]),
        "seed_loglik_spread": float(log_likelihoods.std()),
        "temperature5_effective_seed_count": float(
            1.0 / np.sum(seed_weights * seed_weights)
        ),
        "resampling_count_total": int(resampling_counts.sum()),
        "resampling_count_min": int(resampling_counts.min()),
        "resampling_count_max": int(resampling_counts.max()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "first_resample_seed_fraction": float(np.mean(first_resample_rows >= 0)),
        "ess_before_first_resample_mean": (
            float(np.nanmean(first_resample_ess))
            if np.isfinite(first_resample_ess).any()
            else np.nan
        ),
        "position_clip_count_total": int(position_clip_counts.sum()),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
    }
    return candidate, ledger, diagnostics


# %% [markdown]
# ## 7. Synthetic allocation, fallback, and exp404 parity contracts


# %%
@njit(cache=True, nogil=True)
def _pf_parent_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_position: float,
    initial_rate: float,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_position_spread: float,
    initial_rate_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = (
                last_position + initial_position_spread * np.random.randn()
            )
            rate[particle] = initial_rate + initial_rate_spread * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = (
                    momentum * rate[particle] + rate_noise * np.random.randn()
                )
                position[particle] += (
                    rate[particle] * delta_md + position_noise * np.random.randn()
                )
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                position[particle] = tvt_value + z_v[row]
            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(
                    grid_gr,
                    position[particle] - z_v[row],
                    grid_minimum,
                    grid_step,
                )
                zscore = (gr_v[row] - expected_gr) / gr_scale
                squared = zscore * zscore
                if squared > 600.0:
                    squared = 600.0
                likelihood = np.exp(-0.5 * squared)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)
            weight_sum = 0.0
            for particle in range(particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(particles):
                    weights[particle] = 1.0 / particles
            inverse_ess = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
            effective_sample_size = 1.0 / inverse_ess
            if effective_sample_size < resample_fraction * particles:
                cumulative = np.empty(particles)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = rate[cursor] + rough_rate * np.random.randn()
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles
            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (position[particle] - z_v[row])
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return predictions, log_likelihoods


def synthetic_contracts() -> dict[str, Any]:
    components = stratified_component_ids(500)
    counts = np.bincount(components, minlength=5)
    synthetic = pd.DataFrame(
        {
            "MD": np.arange(8, dtype=np.float64),
            "Z": np.arange(8, dtype=np.float64) * 0.25,
            "TVT_input": np.arange(8, dtype=np.float64) * 0.75,
        }
    )
    rate, _, valid_steps, fallback = robust_initial_rate(synthetic, 8)
    insufficient = synthetic.iloc[:3].copy()
    fallback_rate, _, insufficient_steps, used_fallback = robust_initial_rate(
        insufficient,
        8,
    )
    return {
        "interleave_first15": components[:15].tolist(),
        "component_counts": counts.tolist(),
        "allocation_pass": bool(np.array_equal(counts, np.full(5, 100))),
        "known_formula_expected": 1.0,
        "known_formula_observed": rate,
        "known_formula_valid_steps": valid_steps,
        "known_formula_pass": bool(rate == 1.0 and not fallback),
        "fallback_rate": fallback_rate,
        "fallback_valid_steps": insufficient_steps,
        "fallback_pass": bool(fallback_rate == 0.0 and used_fallback),
    }


def duplicated_center_exp404_parity_contract() -> dict[str, Any]:
    rows = 10
    particles = 50
    seeds = 2
    md = np.arange(1, rows + 1, dtype=np.float64)
    z = np.linspace(0.0, 2.0, rows)
    gr = np.linspace(50.0, 80.0, rows)
    grid_gr = np.linspace(40.0, 90.0, 601)
    center = 0.125
    args = (
        md,
        z,
        gr,
        grid_gr,
        -20.0,
        0.2,
        20.0,
        10.0,
    )
    common = (
        particles,
        seeds,
        123456,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
    )
    parent_prediction, parent_loglik = _pf_parent_allseeds(
        *args,
        center,
        *common,
    )
    candidate = _pf_stratified_rate_bank_allseeds(
        *args,
        np.full(5, center, dtype=np.float64),
        checkpoint_indices(rows),
        *common,
    )
    candidate_prediction = candidate[0]
    candidate_loglik = candidate[1]
    prediction_max_abs = float(
        np.max(np.abs(parent_prediction - candidate_prediction))
    )
    loglik_max_abs = float(np.max(np.abs(parent_loglik - candidate_loglik)))
    return {
        "particles": particles,
        "seeds": seeds,
        "prediction_bitwise_equal": bool(
            np.array_equal(parent_prediction, candidate_prediction)
        ),
        "loglik_bitwise_equal": bool(
            np.array_equal(parent_loglik, candidate_loglik)
        ),
        "prediction_max_abs_difference": prediction_max_abs,
        "loglik_max_abs_difference": loglik_max_abs,
        "pass": bool(prediction_max_abs == 0.0 and loglik_max_abs == 0.0),
    }


# %% [markdown]
# ## 8. Target-free candidate generation and freeze


# %%
@dataclass
class LeakageLedger:
    expected_wells: int
    frozen_wells: set[str] = field(default_factory=set)
    truth_rows_before_freeze: int = 0
    control_rows_before_freeze: int = 0
    fold_rows_before_freeze: int = 0
    hidden_like_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    control_rows_after_freeze: int = 0
    fold_rows_after_freeze: int = 0
    hidden_like_rows_after_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def freeze(self, well: str) -> None:
        if well in self.frozen_wells:
            raise RuntimeError(f"{well}: duplicate prediction freeze")
        self.frozen_wells.add(str(well))

    def _record(self, kind: str, rows: int) -> None:
        before_name = f"{kind}_rows_before_freeze"
        after_name = f"{kind}_rows_after_freeze"
        if not self.all_frozen:
            setattr(self, before_name, getattr(self, before_name) + int(rows))
            raise RuntimeError(f"{kind} input was read before all predictions froze")
        setattr(self, after_name, getattr(self, after_name) + int(rows))

    def record_truth(self, rows: int) -> None:
        self._record("truth", rows)

    def record_control(self, rows: int) -> None:
        self._record("control", rows)

    def record_fold(self, rows: int) -> None:
        self._record("fold", rows)

    def record_hidden_like(self, rows: int) -> None:
        self._record("hidden_like", rows)

    def report(self) -> dict[str, Any]:
        return {
            "expected_wells": self.expected_wells,
            "frozen_wells": len(self.frozen_wells),
            "all_frozen": self.all_frozen,
            "before_freeze": {
                "truth_rows": self.truth_rows_before_freeze,
                "control_rows": self.control_rows_before_freeze,
                "fold_rows": self.fold_rows_before_freeze,
                "hidden_like_rows": self.hidden_like_rows_before_freeze,
            },
            "after_freeze": {
                "truth_rows": self.truth_rows_after_freeze,
                "control_rows": self.control_rows_after_freeze,
                "fold_rows": self.fold_rows_after_freeze,
                "hidden_like_rows": self.hidden_like_rows_after_freeze,
            },
        }


@dataclass
class FrozenWell:
    well_id: str
    prediction: pd.DataFrame
    rate_bank: pd.DataFrame
    component_ledger: pd.DataFrame
    audit: dict[str, Any]


def decode_target_free_well(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> FrozenWell:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    changed = dict(get_nested(config, "model.changed_factor") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        windows=changed["windows_rows"],
        minimum_valid_steps=int(changed["minimum_valid_steps"]),
        fallback_rate=float(changed["fallback_rate"]),
        grid_step=float(fixed["typewell_grid_step_ft"]),
    )
    seed_base = stable_seed("likpf", "train", well)
    candidate, component_ledger, diagnostics = run_stratified_rate_bank_pf(
        prepared,
        well=well,
        particles=int(fixed["particles"]),
        seeds=int(fixed["seeds"]),
        seed_base=seed_base,
        temperature=float(fixed["primary_seed_weighting_temperature"]),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(fixed["rough_position"]),
        rough_rate=float(fixed["rough_rate"]),
        resample_fraction=float(fixed["resample_threshold_fraction"]),
        initial_position_spread=float(fixed["initial_position_spread_ft"]),
        initial_rate_spread=float(changed["within_component_rate_spread"]),
    )
    eval_indices = np.asarray(prepared["eval_indices"], dtype=np.int64)
    raw_observed = np.asarray(prepared["raw_gr_observed"], dtype=bool)
    prediction = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64((~raw_observed).mean()),
            PRIMARY_CANDIDATE: np.asarray(candidate, dtype=np.float32),
        }
    )
    if not np.isfinite(prediction[PRIMARY_CANDIDATE]).all():
        raise ValueError(f"{well}: exp485 prediction contains non-finite values")
    rate_bank = prepared["rate_diagnostics"].copy()
    rate_bank.insert(0, "well_id", str(well))
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "prefix_gr_missing_rows": int(
            prepared["scale_audit"]["known_gr_missing_rows"]
        ),
        "eval_rows": int(len(prediction)),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "last_known_tvt": float(prepared["last_known_tvt"]),
        "rate_unique_center_count": int(rate_bank["unique_center_count"].iloc[0]),
        "rate_center_range": float(rate_bank["center_range"].iloc[0]),
        "rate_fallback_count": int(rate_bank["fallback_count"].iloc[0]),
        "seed_base": int(seed_base),
        "seed_first": int(seed_base),
        "seed_last": int(seed_base + int(fixed["seeds"]) - 1),
        "seeds": int(fixed["seeds"]),
        "particles": int(fixed["particles"]),
        "pf_well_runs": 1,
        "seed_well_trajectories": int(fixed["seeds"]),
        "particle_starts": int(fixed["seeds"]) * int(fixed["particles"]),
        "prediction_logical_sha256": dataframe_content_sha(
            prediction,
            ["id", "well_id", "row_idx", PRIMARY_CANDIDATE],
        ),
        "rate_bank_logical_sha256": dataframe_content_sha(
            rate_bank,
            [
                "well_id",
                "component_index",
                "window_rows",
                "center_value",
                "valid_steps",
                "used_fallback",
            ],
        ),
        "component_ledger_logical_sha256": dataframe_content_sha(
            component_ledger,
            [
                "well_id",
                "checkpoint",
                "checkpoint_row",
                "component_index",
                "filtered_posterior_mass_mean",
                "surviving_particle_count_mean",
            ],
        ),
        **diagnostics,
        "wall_seconds": time.time() - started,
    }
    return FrozenWell(
        well_id=str(well),
        prediction=prediction,
        rate_bank=rate_bank,
        component_ledger=component_ledger,
        audit=audit,
    )


def freeze_target_free_outputs(
    frozen_wells: Sequence[FrozenWell],
    output: Path,
    *,
    stage: str = "stage0",
    expected_rows: int | None = None,
    expected_wells: int | None = None,
    ledger: LeakageLedger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions = (
        pd.concat([item.prediction for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    rate_bank = (
        pd.concat([item.rate_bank for item in frozen_wells], ignore_index=True)
        .sort_values(["well_id", "component_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    components = (
        pd.concat([item.component_ledger for item in frozen_wells], ignore_index=True)
        .sort_values(
            ["well_id", "checkpoint_row", "checkpoint", "component_index"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([item.audit for item in frozen_wells])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    wells = len(frozen_wells)
    if expected_rows is None:
        expected_rows = len(predictions)
    if expected_wells is None:
        expected_wells = wells
    if (
        len(predictions) != expected_rows
        or predictions["id"].duplicated().any()
        or predictions["well_id"].nunique() != expected_wells
        or len(rate_bank) != expected_wells * 5
        or len(components) != expected_wells * 25
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp485 target-free output coverage mismatch")
    prediction_artifact = write_deterministic_gzip_csv(
        predictions,
        output / f"{OUTPUT_PREFIX}_{stage}_predictions.csv.gz",
    )
    rate_bank_artifact = write_deterministic_gzip_csv(
        rate_bank,
        output / f"{OUTPUT_PREFIX}_{stage}_rate_bank.csv.gz",
    )
    component_artifact = write_deterministic_gzip_csv(
        components,
        output / f"{OUTPUT_PREFIX}_{stage}_component_ancestry.csv.gz",
    )
    audit_path = output / f"{OUTPUT_PREFIX}_{stage}_well_audit.csv"
    audit.to_csv(audit_path, index=False)
    audit_raw_sha = sha256_path(audit_path)
    logical_columns = ["id", "well_id", "row_idx", PRIMARY_CANDIDATE]
    prediction_logical_sha = dataframe_content_sha(predictions, logical_columns)
    readback = pd.read_csv(
        prediction_artifact["path"],
        compression="gzip",
        dtype={"id": str, "well_id": str},
    )
    readback_logical_sha = dataframe_content_sha(readback, logical_columns)
    readback_pass = bool(
        readback_logical_sha == prediction_logical_sha
        and sha256_decompressed_csv(prediction_artifact["path"])
        == prediction_artifact["decompressed_sha256"]
        and sha256_path(audit_path) == audit_raw_sha
    )
    if not readback_pass:
        raise RuntimeError(f"exp485 {stage} frozen artifact SHA readback failed")
    if ledger is not None:
        for item in sorted(frozen_wells, key=lambda value: value.well_id):
            ledger.freeze(item.well_id)
    frozen = {
        "frozen_before_truth_attachment": True,
        "truth_attached_in_stage0": False,
        "truth_error_fold_hidden_reads_before_freeze": 0,
        "truth_error_fold_hidden_reads_after_freeze": 0,
        "rows": int(len(predictions)),
        "wells": int(predictions["well_id"].nunique()),
        "logical_columns": logical_columns,
        "prediction_logical_sha256": prediction_logical_sha,
        "prediction_schema_sha256": dataframe_schema_sha(predictions),
        "rate_bank_logical_sha256": dataframe_content_sha(
            rate_bank,
            [
                "well_id",
                "component_index",
                "window_rows",
                "center_value",
                "valid_steps",
                "used_fallback",
            ],
        ),
        "component_ancestry_logical_sha256": dataframe_content_sha(
            components,
            [
                "well_id",
                "checkpoint",
                "checkpoint_row",
                "component_index",
                "filtered_posterior_mass_mean",
                "surviving_particle_count_mean",
            ],
        ),
        "prediction_artifact": prediction_artifact,
        "rate_bank_artifact": rate_bank_artifact,
        "component_ancestry_artifact": component_artifact,
        "well_audit": {
            "path": str(audit_path),
            "raw_sha256": audit_raw_sha,
        },
        "well_audit_artifact": {
            "path": str(audit_path),
            "raw_sha256": audit_raw_sha,
        },
        "sha_readback": {
            "prediction_logical_sha256": readback_logical_sha,
            "well_audit_raw_sha256": audit_raw_sha,
            "pass": readback_pass,
        },
        "truth_access_ledger_at_freeze": (
            ledger.report() if ledger is not None else None
        ),
    }
    return predictions, rate_bank, components, audit, frozen


# %% [markdown]
# ## 9. Fail-closed Stage 0 gates


# %%
def evaluate_stage0_gates(
    predictions: pd.DataFrame,
    rate_bank: pd.DataFrame,
    components: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    parity: Mapping[str, Any],
    candidate_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical = dict(get_nested(config, "guards.technical") or {})
    projected_full_seconds = candidate_seconds / max(len(audit), 1) * 773.0
    posterior_sums = components.groupby(
        ["well_id", "checkpoint"],
        sort=False,
    )["filtered_posterior_mass_mean"].sum()
    count_sums = components.groupby(
        ["well_id", "checkpoint"],
        sort=False,
    )["surviving_particle_count_mean"].sum()
    expected_seed_bases = np.asarray(
        [stable_seed("likpf", "train", well) for well in audit["well_id"]],
        dtype=np.int64,
    )
    checks = {
        "initial_component_counts_100_each": bool(
            audit["initial_component_counts_contract"].all()
            and audit["initial_component_count_min"].eq(100).all()
            and audit["initial_component_count_max"].eq(100).all()
        ),
        "particle_component_interleave_contract": bool(
            synthetic["allocation_pass"]
            and synthetic["interleave_first15"] == [0, 1, 2, 3, 4] * 3
        ),
        "rate_formula_contract": bool(synthetic["known_formula_pass"]),
        "fallback_contract": bool(synthetic["fallback_pass"]),
        "duplicate_center_policy_preserved": bool(len(rate_bank) == len(audit) * 5),
        "mechanism_not_globally_degenerate": bool(
            not audit["rate_unique_center_count"].eq(1).all()
        ),
        "finite_prediction_coverage": bool(
            len(predictions) > 0
            and np.isfinite(predictions[PRIMARY_CANDIDATE]).all()
        ),
        "posterior_mass_normalized": bool(
            np.allclose(posterior_sums.to_numpy(np.float64), 1.0, atol=1e-10)
        ),
        "surviving_particle_count_conserved": bool(
            np.allclose(count_sums.to_numpy(np.float64), 500.0, atol=1e-10)
        ),
        "stable_seed_identity": bool(
            np.array_equal(audit["seed_base"].to_numpy(np.int64), expected_seed_bases)
        ),
        "execution_count_match": bool(
            len(audit) == 32
            and int(audit["seed_well_trajectories"].sum()) == 4096
            and int(audit["particle_starts"].sum()) == 2_048_000
        ),
        "truth_late_zero_read_contract": True,
        "duplicated_center_exp404_bitwise_parity": bool(parity["pass"]),
        "runtime_projection_within_limit": bool(
            projected_full_seconds
            <= float(technical["maximum_seconds_full_projection"])
        ),
        "peak_rss_within_limit": bool(
            rss_gb <= float(technical["maximum_peak_rss_gb"])
        ),
    }
    return {
        "stage": "stage0_fixed32_target_free_technical_mechanism_not_cv",
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "candidate_seconds": candidate_seconds,
        "projected_full_seconds": projected_full_seconds,
        "peak_rss_gb": rss_gb,
        "synthetic_contracts": dict(synthetic),
        "duplicated_center_exp404_parity": dict(parity),
        "mechanism": {
            "wells_with_one_unique_center": int(
                audit["rate_unique_center_count"].eq(1).sum()
            ),
            "wells_with_multiple_unique_centers": int(
                audit["rate_unique_center_count"].gt(1).sum()
            ),
            "fallback_centers": int(rate_bank["used_fallback"].sum()),
            "component_extinct_seed_fraction_max": float(
                components["extinct_seed_fraction"].max()
            ),
        },
    }


# %% [markdown]
# ## 10. Generated artifacts and Stage 0 orchestration
#
# Standard outputs are target-free predictions, the five-center rate bank,
# component ancestry at five checkpoints, a per-well/runtime audit, scientific
# and input contracts, a freeze manifest, a fail-closed gate report, and
# `metrics.json`. No Stage 0 output contains suffix truth or saved-control data.


# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL") == "1":
        return
    raise RuntimeError("exp485 train stages must run on Kaggle CPU")


def input_manifest(
    raw_dir: Path,
    wells: Sequence[str],
    scope_report: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for well in wells:
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        rows.append(
            {
                "well_id": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort")
    return {
        "split": "train",
        "fixed32": dict(scope_report),
        "raw_dir": str(raw_dir),
        "wells": int(len(frame)),
        "raw_well_content_sha256": dataframe_content_sha(
            frame,
            ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
        ),
        "columns_read_from_horizontal": ["MD", "Z", "GR", "TVT_input"],
        "suffix_truth_read": False,
        "saved_control_read": False,
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    require_kaggle_runtime()
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    wells, scope_report = load_fixed32_scope(config)
    scientific_contract_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_scientific_contract.json",
        scientific_contract,
    )
    input_report = input_manifest(raw_dir, wells, scope_report)
    input_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_input_manifest.json",
        input_report,
    )
    synthetic = synthetic_contracts()
    parity = duplicated_center_exp404_parity_contract()
    frozen_wells = [
        decode_target_free_well(str(well), raw_dir, config) for well in wells
    ]
    predictions, rate_bank, components, audit, frozen = (
        freeze_target_free_outputs(frozen_wells, output)
    )
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    if len(predictions) != expected_rows:
        raise ValueError(
            "exp485 fixed32 suffix row count changed: "
            f"expected={expected_rows}, observed={len(predictions)}"
        )
    candidate_seconds = float(audit["wall_seconds"].sum())
    rss_gb = peak_rss_gb()
    runtime_ledger = {
        "candidate_wells": len(wells),
        "candidate_rows": len(predictions),
        "candidate_pf_well_runs": len(wells),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "candidate_seconds": candidate_seconds,
        "projected_full_seconds": candidate_seconds / len(wells) * 773.0,
        "peak_rss_gb": rss_gb,
        "versions": runtime_versions(),
        "truth_error_fold_hidden_reads_before_freeze": 0,
        "truth_error_fold_hidden_reads_after_freeze": 0,
    }
    runtime_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_runtime_ledger.json",
        runtime_ledger,
    )
    frozen.update(
        {
            "scientific_contract_sha256": scientific_contract[
                "scientific_contract_sha256"
            ],
            "scientific_contract_file_sha256": scientific_contract_artifact[
                "raw_sha256"
            ],
            "input_manifest_sha256": input_artifact["raw_sha256"],
            "runtime_ledger_sha256": runtime_artifact["raw_sha256"],
        }
    )
    freeze_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_freeze_manifest.json",
        frozen,
    )
    gates = evaluate_stage0_gates(
        predictions,
        rate_bank,
        components,
        audit,
        config=config,
        synthetic=synthetic,
        parity=parity,
        candidate_seconds=candidate_seconds,
        rss_gb=rss_gb,
    )
    gate_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_gate_report.json",
        gates,
    )
    status = (
        "stage0_all_pass_pending_stage1_approval"
        if gates["all_pass"]
        else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage0_fixed32_target_free_technical_mechanism_not_cv",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "counts": {
            "wells": len(wells),
            "rows": len(predictions),
            "scientific_variants": 1,
            "candidate_pf_well_runs": len(wells),
            "seed_well_trajectories": int(
                audit["seed_well_trajectories"].sum()
            ),
            "particle_starts": int(audit["particle_starts"].sum()),
            "control_pf_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "frozen_outputs": frozen,
        "gates": gates,
        "runtime": {
            **runtime_ledger,
            "total_seconds": time.time() - started,
        },
        "artifacts": {
            "scientific_contract": scientific_contract_artifact,
            "input_manifest": input_artifact,
            "runtime_ledger": runtime_artifact,
            "freeze_manifest": freeze_artifact,
            "gate_report": gate_artifact,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": (
            "request_separate_stage1_approval"
            if gates["all_pass"]
            else "close_branch_without_parameter_or_gate_rescue"
        ),
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. All-well Stage 1 truth-late CV and promotion gate
#
# Stage 1 runs the unchanged five-center PF once for all 773 train wells.
# Candidate predictions, rate banks, component ledgers, and their hashes freeze
# before suffix truth, saved controls, reporting folds, or hidden-like roles
# are parsed. The original Stage 0 runtime failure remains in the audit trail;
# the user-approved exception permits this run without reclassifying that gate.


# %%
def require_frozen(frozen: Mapping[str, Any], ledger: LeakageLedger) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")) or not ledger.all_frozen:
        raise RuntimeError("exp485 truth-late readout requires all predictions frozen")
    if len(str(frozen.get("prediction_logical_sha256", ""))) != 64:
        raise RuntimeError("exp485 frozen prediction logical SHA is missing")
    if not bool(get_nested(frozen, "sha_readback.pass", False)):
        raise RuntimeError("exp485 frozen prediction SHA readback did not pass")


def load_suffix_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["TVT_input", "TVT"],
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce")
    indices = np.flatnonzero(tvt_input.isna().to_numpy()).astype(np.int64)
    values = truth.iloc[indices].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{well}: suffix truth is non-finite")
    return pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in indices],
            "true_tvt": values,
        }
    )


def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = (
        pd.DataFrame(rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    actual = typed_dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != expected_wells or actual != expected_sha:
        raise ValueError("exp485 raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": actual,
        "well_ids": frame["well_id"].astype(str).tolist(),
        "rows": rows,
    }


def stage1_saved_input_paths(config: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key in (
        "saved_control",
        "exp209_hmm_control",
        "fold_assignment",
        "hidden_like_assignment",
    ):
        spec = dict(get_nested(config, f"data.{key}") or {})
        path = resolve_existing(
            str(spec["filename"]),
            [str(value) for value in spec.get("candidates", [])],
            [str(value) for value in spec.get("patterns", [])],
        )
        paths[key] = str(path)
    return paths


def stage1_resume_input_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    resume = dict(get_nested(config, "data.stage1_resume") or {})
    candidates = [str(value) for value in resume.get("candidates", [])]
    paths: dict[str, Path] = {}
    for key in (
        "prediction",
        "rate_bank",
        "component_ancestry",
        "well_audit",
        "source_manifest",
    ):
        spec = dict(resume.get(key) or {})
        filename = str(spec["filename"])
        try:
            paths[key] = resolve_existing(
                filename,
                candidates,
                [f"**/{filename}"],
            )
        except FileNotFoundError:
            if not filename.endswith(".csv.gz"):
                raise
            unpacked_filename = filename.removesuffix(".gz")
            paths[key] = resolve_existing(
                unpacked_filename,
                candidates,
                [f"**/{unpacked_filename}"],
            )
    return paths


def load_stage1_resume(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    scientific_contract: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    resume = dict(get_nested(config, "data.stage1_resume") or {})
    if not bool(resume.get("enabled")):
        raise RuntimeError("exp485 Stage 1 resume is disabled")
    if not bool(
        get_nested(config, "execution.stage_1_resume_from_version_2_approved")
    ):
        raise RuntimeError("exp485 Stage 1 resume is not approved")
    paths = stage1_resume_input_paths(config)
    source_manifest_spec = dict(resume["source_manifest"])
    if sha256_path(paths["source_manifest"]) != str(
        source_manifest_spec["raw_sha256"]
    ):
        raise ValueError("exp485 Stage 1 resume source manifest SHA mismatch")
    source_manifest = json.loads(paths["source_manifest"].read_text())
    expected_contract_sha = str(scientific_contract["scientific_contract_sha256"])
    if str(source_manifest.get("scientific_contract_sha256")) != expected_contract_sha:
        raise ValueError("exp485 Stage 1 resume scientific contract mismatch")
    expected_rows = int(resume["expected_rows"])
    expected_wells = int(resume["expected_wells"])
    manifest_checks = {
        "experiment": source_manifest.get("experiment") == EXPERIMENT_NAME,
        "source_kernel_version": int(source_manifest.get("source_kernel_version", -1))
        == int(resume["source_kernel_version"]),
        "rows": int(source_manifest.get("rows", -1)) == expected_rows,
        "wells": int(source_manifest.get("wells", -1)) == expected_wells,
        "candidate_pf_well_runs": int(
            source_manifest.get("candidate_pf_well_runs", -1)
        )
        == 773,
        "seed_well_trajectories": int(
            source_manifest.get("seed_well_trajectories", -1)
        )
        == 98_944,
        "particle_starts": int(source_manifest.get("particle_starts", -1))
        == 49_472_000,
        "reads_before_freeze_zero": int(
            source_manifest.get(
                "truth_control_fold_hidden_reads_before_freeze",
                -1,
            )
        )
        == 0,
    }
    if not all(manifest_checks.values()):
        raise ValueError(
            f"exp485 Stage 1 resume source manifest contract failed: {manifest_checks}"
        )
    artifact_integrity: dict[str, Any] = {}
    for key in ("prediction", "rate_bank", "component_ancestry"):
        spec = dict(resume[key])
        raw_sha = sha256_path(paths[key])
        decompressed_sha = sha256_csv_payload(paths[key])
        stored_as_gzip = paths[key].suffix == ".gz"
        artifact_integrity[key] = {
            "path": str(paths[key]),
            "raw_sha256": raw_sha,
            "expected_raw_sha256": str(spec["raw_sha256"]),
            "decompressed_sha256": decompressed_sha,
            "expected_decompressed_sha256": str(spec["decompressed_sha256"]),
            "stored_as_gzip": stored_as_gzip,
            "raw_sha256_check": (
                "required" if stored_as_gzip else "not_applicable_unpacked_by_kaggle"
            ),
            "passed": bool(
                decompressed_sha == str(spec["decompressed_sha256"])
                and (
                    not stored_as_gzip
                    or raw_sha == str(spec["raw_sha256"])
                )
            ),
        }
    audit_spec = dict(resume["well_audit"])
    audit_raw_sha = sha256_path(paths["well_audit"])
    artifact_integrity["well_audit"] = {
        "path": str(paths["well_audit"]),
        "raw_sha256": audit_raw_sha,
        "expected_raw_sha256": str(audit_spec["raw_sha256"]),
        "passed": audit_raw_sha == str(audit_spec["raw_sha256"]),
    }
    if not all(item["passed"] for item in artifact_integrity.values()):
        raise ValueError("exp485 Stage 1 resume artifact SHA mismatch")

    candidate = pd.read_csv(
        paths["prediction"],
        dtype={"id": str, "well_id": str},
    )
    rate_bank = pd.read_csv(paths["rate_bank"], dtype={"well_id": str})
    components = pd.read_csv(
        paths["component_ancestry"],
        dtype={"well_id": str},
    )
    audit = pd.read_csv(paths["well_audit"], dtype={"well_id": str})
    logical_columns = ["id", "well_id", "row_idx", PRIMARY_CANDIDATE]
    prediction_logical_sha = dataframe_content_sha(candidate, logical_columns)
    expected_logical_sha = str(resume["prediction"]["logical_sha256"])
    coverage_checks = {
        "prediction_rows": len(candidate) == expected_rows,
        "prediction_unique_ids": int(candidate["id"].nunique()) == expected_rows,
        "prediction_wells": int(candidate["well_id"].nunique()) == expected_wells,
        "prediction_finite": bool(
            np.isfinite(candidate[PRIMARY_CANDIDATE].to_numpy(np.float64)).all()
        ),
        "prediction_logical_sha": prediction_logical_sha == expected_logical_sha,
        "rate_bank_rows": len(rate_bank) == expected_wells * 5,
        "component_ancestry_rows": len(components) == expected_wells * 25,
        "audit_rows": len(audit) == expected_wells,
        "audit_status": bool(audit["status"].eq("ok").all()),
        "audit_pf_well_runs": int(audit["pf_well_runs"].sum()) == 773,
        "audit_seed_well_trajectories": int(
            audit["seed_well_trajectories"].sum()
        )
        == 98_944,
        "audit_particle_starts": int(audit["particle_starts"].sum())
        == 49_472_000,
    }
    if not all(coverage_checks.values()):
        raise ValueError(
            f"exp485 Stage 1 resume coverage contract failed: {coverage_checks}"
        )
    for well in sorted(candidate["well_id"].astype(str).unique().tolist()):
        ledger.freeze(well)
    prediction_spec = dict(resume["prediction"])
    rate_spec = dict(resume["rate_bank"])
    component_spec = dict(resume["component_ancestry"])
    frozen = {
        "frozen_before_truth_attachment": True,
        "truth_attached_in_stage0": False,
        "truth_error_fold_hidden_reads_before_freeze": 0,
        "truth_error_fold_hidden_reads_after_freeze": 0,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "logical_columns": logical_columns,
        "prediction_logical_sha256": prediction_logical_sha,
        "prediction_schema_sha256": dataframe_schema_sha(candidate),
        "rate_bank_logical_sha256": dataframe_content_sha(
            rate_bank,
            [
                "well_id",
                "component_index",
                "window_rows",
                "center_value",
                "valid_steps",
                "used_fallback",
            ],
        ),
        "component_ancestry_logical_sha256": dataframe_content_sha(
            components,
            [
                "well_id",
                "checkpoint",
                "checkpoint_row",
                "component_index",
                "filtered_posterior_mass_mean",
                "surviving_particle_count_mean",
            ],
        ),
        "prediction_artifact": {
            "path": str(paths["prediction"]),
            "rows": len(candidate),
            "raw_sha256": str(prediction_spec["raw_sha256"]),
            "decompressed_sha256": str(prediction_spec["decompressed_sha256"]),
            "source_dataset": str(resume["dataset_source"]),
        },
        "rate_bank_artifact": {
            "path": str(paths["rate_bank"]),
            "rows": len(rate_bank),
            "raw_sha256": str(rate_spec["raw_sha256"]),
            "decompressed_sha256": str(rate_spec["decompressed_sha256"]),
            "source_dataset": str(resume["dataset_source"]),
        },
        "component_ancestry_artifact": {
            "path": str(paths["component_ancestry"]),
            "rows": len(components),
            "raw_sha256": str(component_spec["raw_sha256"]),
            "decompressed_sha256": str(component_spec["decompressed_sha256"]),
            "source_dataset": str(resume["dataset_source"]),
        },
        "well_audit": {
            "path": str(paths["well_audit"]),
            "raw_sha256": audit_raw_sha,
            "source_dataset": str(resume["dataset_source"]),
        },
        "well_audit_artifact": {
            "path": str(paths["well_audit"]),
            "raw_sha256": audit_raw_sha,
            "source_dataset": str(resume["dataset_source"]),
        },
        "sha_readback": {
            "prediction_logical_sha256": prediction_logical_sha,
            "expected_prediction_logical_sha256": expected_logical_sha,
            "artifact_integrity": artifact_integrity,
            "pass": True,
        },
        "truth_access_ledger_at_freeze": ledger.report(),
        "resumed_from_sha_pinned_target_free_artifacts": True,
    }
    resume_report = {
        "enabled": True,
        "source_kernel_id": str(resume["source_kernel_id"]),
        "source_kernel_version": int(resume["source_kernel_version"]),
        "source_kernel_status": str(resume["source_kernel_status"]),
        "source_dataset": str(resume["dataset_source"]),
        "source_runtime_seconds_to_error": float(
            resume["source_runtime_seconds_to_error"]
        ),
        "candidate_pf_well_runs_in_resume_kernel": 0,
        "source_manifest": {
            "path": str(paths["source_manifest"]),
            "raw_sha256": sha256_path(paths["source_manifest"]),
            "checks": manifest_checks,
        },
        "artifact_integrity": artifact_integrity,
        "coverage_checks": coverage_checks,
        "truth_access_ledger_at_refreeze": ledger.report(),
    }
    return candidate, rate_bank, components, audit, frozen, resume_report


def _align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    aligned_source = source.copy()
    aligned_source["id"] = aligned_source["id"].astype(str)
    if aligned_source["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = aligned_source.set_index("id").reindex(frame["id"].astype(str))
    if aligned[list(columns)].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[str(column)] = aligned[str(column)].to_numpy()
    return result


def attach_truth_late_stage1(
    candidate: pd.DataFrame,
    frozen: Mapping[str, Any],
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    saved_paths: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen(frozen, ledger)
    logical_sha = dataframe_content_sha(candidate, frozen["logical_columns"])
    if logical_sha != str(frozen["prediction_logical_sha256"]):
        raise RuntimeError("exp485 Stage 1 candidate changed after prediction freeze")
    wells = sorted(candidate["well_id"].astype(str).unique().tolist())
    truth_parts = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(delayed(load_suffix_truth)(well, raw_dir) for well in wells)
    truth = pd.concat(truth_parts, ignore_index=True)
    ledger.record_truth(len(truth))
    frame = _align_on_id(candidate, truth, ["true_tvt"], label="raw suffix truth")

    control_spec = dict(get_nested(config, "data.saved_control") or {})
    control_path = Path(saved_paths["saved_control"])
    if sha256_path(control_path) != str(control_spec["expected_raw_sha256"]):
        raise ValueError("exp485 saved exp404 control raw SHA mismatch")
    if sha256_decompressed_csv(control_path) != str(
        control_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp485 saved exp404 control decompressed SHA mismatch")
    control_source_column = str(control_spec["prediction_column"])
    control = pd.read_csv(
        control_path,
        compression="gzip",
        usecols=["id", control_source_column],
        dtype={"id": str},
    )
    ledger.record_control(len(control))
    control[control_source_column] = pd.to_numeric(
        control[control_source_column],
        errors="raise",
    )
    control = control.rename(columns={control_source_column: PRIMARY_CONTROL})
    frame = _align_on_id(
        frame,
        control[["id", PRIMARY_CONTROL]],
        [PRIMARY_CONTROL],
        label="saved exp404 scale-5 control",
    )

    hmm_spec = dict(get_nested(config, "data.exp209_hmm_control") or {})
    hmm_path = Path(saved_paths["exp209_hmm_control"])
    hmm_source_column = str(hmm_spec["prediction_column"])
    hmm_decompressed_sha = sha256_decompressed_csv(hmm_path)
    hmm, hmm_selected_integrity = read_selected_csv_with_content_sha(
        hmm_path,
        [str(value) for value in hmm_spec["expected_selected_columns"]],
        numeric_columns=[hmm_source_column],
    )
    hmm_selected_integrity.update(
        {
            "expected_rows": int(hmm_spec["expected_selected_columns_rows"]),
            "expected_selected_columns_sha256": str(
                hmm_spec["expected_selected_columns_sha256"]
            ),
            "decompressed_sha256": hmm_decompressed_sha,
            "expected_decompressed_sha256": str(
                hmm_spec["expected_decompressed_sha256"]
            ),
            "decompressed_sha256_match": bool(
                hmm_decompressed_sha
                == str(hmm_spec["expected_decompressed_sha256"])
            ),
        }
    )
    hmm_selected_integrity["passed"] = bool(
        int(hmm_selected_integrity["rows"])
        == int(hmm_selected_integrity["expected_rows"])
        and str(hmm_selected_integrity["selected_columns_sha256"])
        == str(hmm_selected_integrity["expected_selected_columns_sha256"])
    )
    if not bool(hmm_selected_integrity["passed"]):
        raise ValueError("exp485 saved exp209 HMM selected-column SHA mismatch")
    ledger.record_control(len(hmm))
    hmm = hmm.rename(columns={hmm_source_column: "saved_exp209_hmm"})
    frame = _align_on_id(
        frame,
        hmm[["id", "saved_exp209_hmm"]],
        ["saved_exp209_hmm"],
        label="saved exp209 HMM",
    )

    fold_spec = dict(get_nested(config, "data.fold_assignment") or {})
    fold_path = Path(saved_paths["fold_assignment"])
    if sha256_decompressed_csv(fold_path) != str(
        fold_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp485 reporting-fold decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    forbidden = {
        str(value) for value in fold_spec.get("forbidden_decoder_columns", [])
    }
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp485 fold allowlist must contain identity/fold columns")
    if set(safe_columns) & forbidden:
        raise ValueError("exp485 fold allowlist contains forbidden decoder columns")
    fold = pd.read_csv(fold_path, usecols=safe_columns, dtype={"well_id": str})
    ledger.record_fold(len(fold))
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp485 reporting-fold identity is duplicated")
    fold = fold.rename(columns={"suffix_offset": "reporting_suffix_offset"})
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if frame[["fold", "reporting_suffix_offset"]].isna().any().any():
        raise ValueError("exp485 reporting-fold attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["reporting_suffix_offset"].to_numpy(np.int64),
    ):
        raise ValueError("exp485 reporting-fold suffix identity mismatch")
    frame = frame.drop(columns=["reporting_suffix_offset"])

    hidden_spec = dict(get_nested(config, "data.hidden_like_assignment") or {})
    hidden_path = Path(saved_paths["hidden_like_assignment"])
    if sha256_path(hidden_path) != str(hidden_spec["expected_sha256"]):
        raise ValueError("exp485 hidden-like assignment raw SHA mismatch")
    role_columns = {
        str(scope): str(column)
        for scope, column in dict(hidden_spec["role_columns"]).items()
    }
    hidden = pd.read_csv(
        hidden_path,
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    ledger.record_hidden_like(len(hidden))
    if hidden["well_id"].duplicated().any():
        raise ValueError("exp485 hidden-like assignment has duplicate wells")
    for scope, column in role_columns.items():
        actual = {
            str(key): int(value)
            for key, value in hidden[column]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .items()
        }
        expected = {
            str(key): int(value)
            for key, value in dict(hidden_spec["expected_role_counts"][scope]).items()
        }
        if actual != expected:
            raise ValueError(f"exp485 hidden-like role counts changed for {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("exp485 hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[
        role_columns["hidden_like_spatial"]
    ].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[
        role_columns["hidden_like_typewell_purged"]
    ].eq("valid")
    frame["candidate_hmm_50_50"] = 0.5 * (
        frame[PRIMARY_CANDIDATE].to_numpy(np.float64)
        + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    frame["control_hmm_50_50"] = 0.5 * (
        frame[PRIMARY_CONTROL].to_numpy(np.float64)
        + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    finite_columns = [
        "true_tvt",
        PRIMARY_CONTROL,
        PRIMARY_CANDIDATE,
        "saved_exp209_hmm",
        "candidate_hmm_50_50",
        "control_hmm_50_50",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("exp485 Stage 1 late readout contains non-finite values")
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("exp485 reporting-fold set mismatch")
    return frame, {
        "truth_attached_after_prediction_freeze": True,
        "candidate_content_sha256_reverified": logical_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "saved_input_paths": dict(saved_paths),
        "saved_input_integrity": {
            "exp209_hmm_selected_columns": hmm_selected_integrity,
        },
        "truth_access_ledger": ledger.report(),
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def stage1_metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    candidate_column: str,
    control_column: str,
    comparison: str,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"exp485 Stage 1 metric scope is empty: {scope}")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[candidate_column].to_numpy(np.float64)
    control = selected[control_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "candidate": candidate_column,
        "control": control_column,
        "comparison": comparison,
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "control_mae": float(np.mean(np.abs(control - truth))),
    }


def stage1_metric_scopes(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool)),
    ]
    for fold in sorted(frame["fold"].astype(int).unique().tolist()):
        scopes.append((f"fold_{fold}", frame["fold"].eq(fold).to_numpy()))
    scopes.extend(
        [
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
            (
                "missing_fraction_high",
                frame["well_missing_fraction"].ge(0.30).to_numpy(),
            ),
            ("md_since_1000_plus", frame["md_since"].ge(1000.0).to_numpy()),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    return scopes


def build_stage1_metric_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = stage1_metric_scopes(frame)
    primary = pd.DataFrame(
        [
            stage1_metric_record(
                frame,
                mask,
                candidate_column=PRIMARY_CANDIDATE,
                control_column=PRIMARY_CONTROL,
                comparison="rate_bank5_vs_saved_exp404_scale5_x1p0",
                scope=scope,
            )
            for scope, mask in scopes
        ]
    )
    blend = pd.DataFrame(
        [
            stage1_metric_record(
                frame,
                mask,
                candidate_column="candidate_hmm_50_50",
                control_column="control_hmm_50_50",
                comparison="fixed_exp209_hmm_pf_50_50",
                scope=scope,
            )
            for scope, mask in scopes
        ]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        candidate = group[PRIMARY_CANDIDATE].to_numpy(np.float64)
        control = group[PRIMARY_CONTROL].to_numpy(np.float64)
        candidate_rmse = rmse(truth, candidate)
        control_rmse = rmse(truth, control)
        by_well_rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "candidate_rmse": candidate_rmse,
                "control_rmse": control_rmse,
                "improvement_ft": control_rmse - candidate_rmse,
                "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
                "well_missing_fraction": float(
                    group["well_missing_fraction"].iloc[0]
                ),
            }
        )
    return primary, pd.DataFrame(by_well_rows), blend


def _stage1_scope_row(metrics: pd.DataFrame, scope: str) -> pd.Series:
    selected = metrics.loc[metrics["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"exp485 expected one Stage 1 metric row for {scope}")
    return selected.iloc[0]


def evaluate_stage1_gate(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    rate_bank: pd.DataFrame,
    components: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    primary_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    blend_metrics: pd.DataFrame,
    ledger_at_freeze: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    runtime_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical") or {})
    scientific_config = dict(get_nested(config, "guards.scientific") or {})
    overall = _stage1_scope_row(primary_metrics, "overall")
    blend_overall = _stage1_scope_row(blend_metrics, "overall")
    fold_rows = primary_metrics.loc[primary_metrics["scope"].str.startswith("fold_")]
    improved_folds = int((fold_rows["improvement_ft"] > 0.0).sum())
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    before = dict(ledger_at_freeze["before_freeze"])
    control_difference = abs(
        float(overall["control_rmse"])
        - float(get_nested(config, "validation.primary_control_rmse_ft"))
    )
    blend_control_difference = abs(
        float(blend_overall["control_rmse"])
        - float(get_nested(config, "validation.fixed_hmm_pf_50_50_control_rmse_ft"))
    )
    execution_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": 773,
        "seed_well_trajectories": 98_944,
        "particle_starts": 49_472_000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    original_runtime_passed = bool(
        runtime_seconds <= float(get_nested(config, "runtime.maximum_seconds"))
    )
    runtime_exception_approved = bool(
        get_nested(config, "stage_0_result.runtime_exception.approved")
    )
    technical_checks = {
        "stage0_all_non_runtime_gates_passed": bool(
            get_nested(config, "stage_0_result.all_non_runtime_gates_passed")
        ),
        "stage0_runtime_exception_approved": runtime_exception_approved,
        "raw_input_identity": bool(
            raw_manifest["content_sha256"]
            == str(get_nested(config, "data.expected_raw_well_identity_sha256"))
        ),
        "prediction_rows": len(frame) == expected_rows,
        "prediction_wells": int(frame["well_id"].nunique()) == expected_wells,
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist())
        == expected_folds,
        "all_wells_completed": bool(
            len(audit) == expected_wells and audit["status"].eq("ok").all()
        ),
        "initial_component_counts_100_each": bool(
            audit["initial_component_counts_contract"].all()
        ),
        "mechanism_not_globally_degenerate": bool(
            not audit["rate_unique_center_count"].eq(1).all()
        ),
        "duplicate_center_policy_preserved": len(rate_bank) == expected_wells * 5,
        "component_ledger_coverage": len(components) == expected_wells * 25,
        "finite_prediction_coverage": bool(
            np.isfinite(frame[PRIMARY_CANDIDATE].to_numpy(np.float64)).all()
        ),
        "truth_error_fold_hidden_reads_before_freeze_zero": bool(
            all(int(value) == 0 for value in before.values())
        ),
        "execution_count_match": execution_counts == expected_counts,
        "artifact_sha_readback": bool(get_nested(frozen, "sha_readback.pass")),
        "saved_control_rmse_parity": bool(
            control_difference
            <= float(technical_config["require_saved_control_rmse_parity_atol_ft"])
        ),
        "fixed_hmm_pf_50_50_control_parity": bool(
            blend_control_difference
            <= float(
                technical_config["require_fixed_hmm_pf_50_50_parity_atol_ft"]
            )
        ),
        "runtime_accepted": bool(
            original_runtime_passed or runtime_exception_approved
        ),
        "peak_rss": bool(
            rss_gb <= float(get_nested(config, "runtime.maximum_peak_rss_gb"))
        ),
    }
    technical = {
        "checks": technical_checks,
        "passed": bool(all(technical_checks.values())),
        "execution_counts": execution_counts,
        "saved_control_rmse_absolute_difference": control_difference,
        "fixed_hmm_pf_50_50_control_rmse_absolute_difference": (
            blend_control_difference
        ),
        "runtime_seconds": runtime_seconds,
        "original_runtime_limit_seconds": float(
            get_nested(config, "runtime.maximum_seconds")
        ),
        "original_runtime_gate_passed": original_runtime_passed,
        "runtime_user_exception_approved": runtime_exception_approved,
        "runtime_user_exception_applied": bool(
            not original_runtime_passed and runtime_exception_approved
        ),
        "peak_rss_gb": rss_gb,
        "truth_access_ledger_at_freeze": dict(ledger_at_freeze),
    }
    scope_rules = {
        "raw_gr_observed": ("minimum_gain", "minimum_raw_gr_observed_gain_ft"),
        "raw_gr_missing": (
            "maximum_regression",
            "maximum_raw_gr_missing_regression_ft",
        ),
        "missing_fraction_high": (
            "maximum_regression",
            "maximum_high_missing_well_regression_ft",
        ),
        "md_since_1000_plus": (
            "maximum_regression",
            "maximum_long_tail_1000_plus_regression_ft",
        ),
        "hidden_like_spatial": (
            "maximum_regression",
            "maximum_hidden_like_spatial_regression_ft",
        ),
        "hidden_like_typewell_purged": (
            "maximum_regression",
            "maximum_hidden_like_typewell_purged_regression_ft",
        ),
    }
    scope_checks: dict[str, Any] = {}
    for scope, (kind, key) in scope_rules.items():
        row = _stage1_scope_row(primary_metrics, scope)
        threshold = float(scientific_config[key])
        improvement = float(row["improvement_ft"])
        delta = float(row["delta_rmse_candidate_minus_control"])
        passed = improvement >= threshold if kind == "minimum_gain" else delta <= threshold
        scope_checks[scope] = {
            "candidate_rmse": float(row["candidate_rmse"]),
            "control_rmse": float(row["control_rmse"]),
            "improvement_ft": improvement,
            "delta_rmse_candidate_minus_control": delta,
            "rule": kind,
            "threshold_ft": threshold,
            "passed": bool(passed),
        }
    by_well_delta = by_well_metrics["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well = float(by_well_delta.max())
    primary_gate = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            scientific_config["minimum_pooled_rmse_gain_vs_control_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(scientific_config["minimum_improved_folds"]),
        "scope_checks": scope_checks,
        "by_well_delta_p95_ft": by_well_p95,
        "maximum_by_well_delta_p95_ft": float(
            scientific_config["maximum_by_well_delta_p95_ft"]
        ),
        "worst_well_regression_ft": worst_well,
        "maximum_worst_well_regression_ft": float(
            scientific_config["maximum_worst_well_regression_ft"]
        ),
    }
    primary_gate["passed"] = bool(
        primary_gate["improvement_ft"] >= primary_gate["minimum_improvement_ft"]
        and improved_folds >= primary_gate["minimum_improved_folds"]
        and all(item["passed"] for item in scope_checks.values())
        and by_well_p95 <= primary_gate["maximum_by_well_delta_p95_ft"]
        and worst_well <= primary_gate["maximum_worst_well_regression_ft"]
    )
    blend_guard = {
        "candidate_rmse": float(blend_overall["candidate_rmse"]),
        "control_rmse": float(blend_overall["control_rmse"]),
        "delta_rmse_candidate_minus_control": float(
            blend_overall["delta_rmse_candidate_minus_control"]
        ),
        "maximum_regression_ft": float(
            scientific_config["maximum_fixed_hmm_pf_50_50_regression_ft"]
        ),
    }
    blend_guard["passed"] = bool(
        blend_guard["delta_rmse_candidate_minus_control"]
        <= blend_guard["maximum_regression_ft"]
    )
    passed = bool(technical["passed"] and primary_gate["passed"] and blend_guard["passed"])
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage1_all_well_train_side_cv",
        "passed": passed,
        "decision": (
            "eligible_for_separate_raw_test_inference_design"
            if passed
            else "terminal_close_without_rate_bank_or_pf_rescue"
        ),
        "technical_gate": technical,
        "primary_scientific_gate": primary_gate,
        "fixed_exp209_hmm_pf_50_50_guard": blend_guard,
        "failure_action": (
            "close_without_window_allocation_spread_particle_seed_temperature_"
            "gate_blend_selector_or_same_oof_rescue"
        ),
    }


def run_stage1(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    if not bool(get_nested(config, "execution.run_stage_1")):
        raise RuntimeError("exp485 Stage 1 is not selected")
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest = validate_raw_well_identity(config, raw_dir)
    wells = list(raw_manifest["well_ids"])
    saved_paths = stage1_saved_input_paths(config)
    synthetic = synthetic_contracts()
    parity = duplicated_center_exp404_parity_contract()
    if not bool(parity["pass"]):
        raise RuntimeError("exp485 Stage 1 exp404 duplicate-center parity failed")
    contract_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_scientific_contract.json",
        scientific_contract,
    )
    input_report = {
        "raw": raw_manifest,
        "synthetic_contracts": synthetic,
        "duplicated_center_exp404_parity": parity,
        "runtime_exception": get_nested(config, "stage_0_result.runtime_exception"),
        "stage1_resume": get_nested(config, "data.stage1_resume"),
        "saved_inputs": {
            key: {
                "path": value,
                "content_values_parsed_before_freeze": False,
            }
            for key, value in saved_paths.items()
        },
    }
    input_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_input_manifest.json",
        input_report,
    )
    ledger = LeakageLedger(expected_wells=len(wells))
    started = time.time()
    resume_enabled = bool(get_nested(config, "data.stage1_resume.enabled"))
    if resume_enabled:
        (
            candidate,
            rate_bank,
            components,
            audit,
            frozen,
            resume_report,
        ) = load_stage1_resume(config, ledger, scientific_contract)
        source_runtime_seconds = float(
            get_nested(config, "data.stage1_resume.source_runtime_seconds_to_error")
        )
    else:
        results = Parallel(
            n_jobs=int(get_nested(config, "runtime.num_workers")),
            prefer="threads",
        )(delayed(decode_target_free_well)(well, raw_dir, config) for well in wells)
        candidate, rate_bank, components, audit, frozen = freeze_target_free_outputs(
            results,
            output,
            stage="stage1",
            expected_rows=int(get_nested(config, "validation.expected_rows")),
            expected_wells=int(get_nested(config, "validation.expected_wells")),
            ledger=ledger,
        )
        resume_report = {
            "enabled": False,
            "candidate_pf_well_runs_in_resume_kernel": 773,
        }
        source_runtime_seconds = 0.0
    current_runtime_to_freeze = time.time() - started
    runtime_to_freeze = source_runtime_seconds + current_runtime_to_freeze
    ledger_at_freeze = ledger.report()
    frame, late_report = attach_truth_late_stage1(
        candidate,
        frozen,
        raw_dir,
        config,
        ledger,
        saved_paths,
    )
    primary_metrics, by_well_metrics, blend_metrics = build_stage1_metric_outputs(frame)
    current_runtime_seconds = time.time() - started
    runtime_seconds = source_runtime_seconds + current_runtime_seconds
    rss_gb = peak_rss_gb()
    gate = evaluate_stage1_gate(
        config,
        frame,
        rate_bank,
        components,
        audit,
        frozen,
        primary_metrics,
        by_well_metrics,
        blend_metrics,
        ledger_at_freeze,
        raw_manifest,
        runtime_seconds,
        rss_gb,
    )
    paths = {
        "truth_late_rows": output / f"{OUTPUT_PREFIX}_stage1_truth_late_rows.csv.gz",
        "primary_metrics": output / f"{OUTPUT_PREFIX}_stage1_primary_metrics.csv",
        "by_well_metrics": output / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv",
        "blend_metrics": (
            output / f"{OUTPUT_PREFIX}_stage1_fixed_hmm_pf_50_50_metrics.csv"
        ),
        "promotion_gate": output / f"{OUTPUT_PREFIX}_stage1_promotion_gate.json",
        "runtime_ledger": output / f"{OUTPUT_PREFIX}_stage1_runtime_ledger.json",
    }
    truth_artifact = write_deterministic_gzip_csv(frame, paths["truth_late_rows"])
    primary_metrics.to_csv(paths["primary_metrics"], index=False)
    by_well_metrics.to_csv(paths["by_well_metrics"], index=False)
    blend_metrics.to_csv(paths["blend_metrics"], index=False)
    gate_artifact = write_json(paths["promotion_gate"], gate)
    runtime_artifact = write_json(
        paths["runtime_ledger"],
        {
            "runtime_seconds_to_prediction_freeze": runtime_to_freeze,
            "runtime_seconds_current_kernel_to_prediction_refreeze": (
                current_runtime_to_freeze
            ),
            "runtime_seconds_current_kernel_total": current_runtime_seconds,
            "runtime_seconds_source_kernel_to_error": source_runtime_seconds,
            "runtime_seconds_aggregate_stage1": runtime_seconds,
            "peak_rss_gb": rss_gb,
            "runtime_versions": runtime_versions(),
            "kaggle_kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
        },
    )
    artifact_manifest = {
        "scientific_contract": contract_artifact,
        "input_manifest": input_artifact,
        "prediction": frozen["prediction_artifact"],
        "rate_bank": frozen["rate_bank_artifact"],
        "component_ancestry": frozen["component_ancestry_artifact"],
        "well_audit": frozen["well_audit_artifact"],
        "truth_late_rows": truth_artifact,
        "primary_metrics": {
            "path": str(paths["primary_metrics"]),
            "raw_sha256": sha256_path(paths["primary_metrics"]),
        },
        "by_well_metrics": {
            "path": str(paths["by_well_metrics"]),
            "raw_sha256": sha256_path(paths["by_well_metrics"]),
        },
        "blend_metrics": {
            "path": str(paths["blend_metrics"]),
            "raw_sha256": sha256_path(paths["blend_metrics"]),
        },
        "promotion_gate": gate_artifact,
        "runtime_ledger": runtime_artifact,
    }
    status = (
        "stage1_all_gates_passed"
        if gate["passed"]
        else "stage1_gate_failed_terminal_close"
    )
    overall = _stage1_scope_row(primary_metrics, "overall")
    blend_overall = _stage1_scope_row(blend_metrics, "overall")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage1_all_well_train_side_cv",
        "cv": float(overall["candidate_rmse"]),
        "public_lb": None,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "candidate_pf_well_runs_current_kernel": int(
            resume_report["candidate_pf_well_runs_in_resume_kernel"]
        ),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu_runs": 0,
        "candidate_rmse": float(overall["candidate_rmse"]),
        "saved_control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "improved_folds": int(
            (
                primary_metrics.loc[
                    primary_metrics["scope"].str.startswith("fold_"),
                    "improvement_ft",
                ]
                > 0.0
            ).sum()
        ),
        "fixed_hmm_pf_50_50_candidate_rmse": float(
            blend_overall["candidate_rmse"]
        ),
        "fixed_hmm_pf_50_50_control_rmse": float(blend_overall["control_rmse"]),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "rate_bank_sha256": frozen["rate_bank_logical_sha256"],
        "component_ancestry_sha256": frozen[
            "component_ancestry_logical_sha256"
        ],
        "late_readout": late_report,
        "stage1_resume": resume_report,
        "promotion_gate": gate,
        "truth_access_ledger": ledger.report(),
        "artifacts": artifact_manifest,
        "deterministic_anchor": False,
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(
        metrics_output_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": status,
            "cv": float(overall["candidate_rmse"]),
            "public_lb": None,
            "private_lb": None,
            "metric": "rmse",
            "stage1": True,
            "promotion_gate": gate,
            "prediction_sha256": frozen["prediction_logical_sha256"],
            "notes": (
                "All-well train-side Stage 1 under a user-approved runtime "
                "exception. No raw-test inference or submission."
            ),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 12. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(
    CONFIG,
    require_run_approval=False,
)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "active_variants": get_nested(CONFIG, "model.active_variants"),
            "rate_windows": get_nested(
                CONFIG,
                "model.changed_factor.windows_rows",
            ),
            "particles_per_component": get_nested(
                CONFIG,
                "model.changed_factor.particles_per_component",
            ),
            "stage0_candidate_pf_well_runs": get_nested(
                CONFIG,
                "stages.stage_0.candidate_pf_well_runs",
            ),
            "seeds": get_nested(CONFIG, "model.fixed_from_exp404.seeds"),
            "particles": get_nested(CONFIG, "model.fixed_from_exp404.particles"),
            "control_pf_reruns": get_nested(
                CONFIG,
                "execution.control_pf_well_runs",
            ),
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
            "stage_1_approved": get_nested(
                CONFIG,
                "execution.stage_1_execution_approved",
            ),
            "runtime_exception_approved": get_nested(
                CONFIG,
                "stage_0_result.runtime_exception.approved",
            ),
            "inference_enabled": get_nested(
                CONFIG,
                "implementation.inference_enabled",
            ),
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                "scientific_contract_sha256"
            ],
        },
        indent=2,
        sort_keys=True,
    )
)

if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_stage_0", False)):
        STAGE0_RESULT = run_stage0(CONFIG)
    elif bool(get_nested(CONFIG, "execution.run_stage_1", False)):
        STAGE1_RESULT = run_stage1(CONFIG)
    else:
        print(
            "exp485 has no selected execution stage; Kaggle execution remains "
            "disabled pending approval."
        )
