# %% [markdown]
# # exp416 roughening ×10 likelihood-PF full-OOF ablation — train
#
# This train-side candidate audits one preregistered change to the exp072
# likelihood-weighted particle filter: post-resampling position roughening is
# changed from 0.10 to 1.00 ft and rate roughening from 0.001 to 0.010. The
# saved exp072 arithmetic seed mean remains the control and is never rerun.
# Candidate generation is split into four deterministic CPU shards. Suffix
# truth and reporting roles are read only after all candidate predictions have
# been merged and frozen with a logical content SHA.

# %% [markdown]
# ## Contents
# 1. Imports and fixed notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen one-factor scientific contract
# 4. Truth-free raw input checks and deterministic LPT sharding
# 5. Exact exp072 input preparation
# 6. Exact exp072 likelihood-PF kernel and roughening-only intervention
# 7. Shard candidate generation and prediction freeze
# 8. Strict shard merge and optional rerun probe
# 9. Late truth, saved-control, fold, hidden-like, and episode attachment
# 10. Metrics and fail-closed scientific gate
# 11. Generated artifacts and stage orchestration
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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


EXPERIMENT_NAME = "exp416_roughening_x10_likpf_full_oof_ablation"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "likpf_roughening_x10_mean"
PREDICTION_COLUMNS = (PRIMARY_CANDIDATE,)
SHARD_COUNT = 4
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP416_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
    raise FileNotFoundError(f"exp416 config not found; checked={checked}")


def resolve_package_file(filename: str) -> Path:
    checked: list[str] = []
    for package_dir in candidate_package_dirs():
        path = package_dir / filename
        checked.append(str(path))
        if path.exists() and path.is_file():
            return path
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"exp416 package file {filename} not found; checked={checked}")


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
    csv_path = Path(path)
    return {
        "path": str(csv_path),
        "bytes": csv_path.stat().st_size,
        "raw_sha256": sha256_path(csv_path),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist(),
    }


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
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
    return versions


def maximum_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


# %% [markdown]
# ## 3. Frozen one-factor scientific contract


# %%
def pf_parameter_sets(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pf = dict(get_nested(config, "model.pf") or {})
    shared = {
        "particles": int(pf["particles"]),
        "seeds": int(pf["seeds"]),
        "seed_indices": [int(value) for value in pf["seed_indices"]],
        "typewell_grid_step_ft": float(pf["typewell_grid_step_ft"]),
        "initial_position_spread_ft": float(pf["initial_position_spread_ft"]),
        "initial_rate_spread": float(pf["initial_rate_spread"]),
        "momentum": float(pf["momentum"]),
        "rate_noise": float(pf["rate_noise"]),
        "position_noise": float(pf["position_noise"]),
        "resample_threshold_fraction": float(pf["resample_threshold_fraction"]),
        "resampling": str(pf["resampling"]),
        "emission": str(pf["emission"]),
        "emission_clip_z2": float(pf["emission_clip_z2"]),
        "gr_sigma_clip": [float(value) for value in pf["gr_sigma_clip"]],
        "typewell_tvt_pad_ft": float(pf["typewell_tvt_pad_ft"]),
        "missing_gr_policy": str(pf["missing_gr_policy"]),
        "output_dtype": str(pf["output_dtype"]),
        "seed_aggregation": str(get_nested(config, "model.replay.seed_aggregation")),
    }
    control = {
        **shared,
        "rough_position": float(pf["control_rough_position"]),
        "rough_rate": float(pf["control_rough_rate"]),
    }
    candidate = {
        **shared,
        "rough_position": float(pf["candidate_rough_position"]),
        "rough_rate": float(pf["candidate_rough_rate"]),
    }
    return control, candidate


def roughening_only_parameter_diff(config: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    control, candidate = pf_parameter_sets(config)
    diff = {
        key: (control[key], candidate[key])
        for key in sorted(control)
        if control[key] != candidate[key]
    }
    if set(diff) != {"rough_position", "rough_rate"}:
        raise ValueError(f"exp416 allows only two roughening parameter changes; got {diff}")
    multiplier = float(get_nested(config, "model.pf.roughening_multiplier"))
    tolerance = float(
        get_nested(config, "guards.technical.roughening_ratio_absolute_tolerance")
    )
    for key, (base, treatment) in diff.items():
        if abs(float(treatment) / float(base) - multiplier) > tolerance:
            raise ValueError(f"exp416 {key} is not exactly the fixed roughening multiplier")
    return diff


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    control, candidate = pf_parameter_sets(config)
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "evidence": list(get_nested(config, "lineage.evidence") or []),
        "truth_attached": False,
        "primary_control": str(get_nested(config, "validation.primary_control")),
        "primary_candidate": PRIMARY_CANDIDATE,
        "control_pf": "saved_exp072_load_only_zero_reruns",
        "control_parameters": control,
        "candidate_parameters": candidate,
        "parameter_diff": roughening_only_parameter_diff(config),
        "execution_counts": {
            key: get_nested(config, f"execution.{key}")
            for key in (
                "scientific_variants",
                "candidate_pf_well_runs",
                "parent_pf_control_reruns",
                "seeds_per_well",
                "seed_well_trajectories",
                "particles_per_seed",
                "particle_starts",
                "reporting_folds",
                "well_shard_count",
                "lightgbm_configs",
                "trained_folds",
                "boosters",
                "hmm_well_runs",
                "beam_well_runs",
                "gpu_runs",
            )
        },
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "forbidden": list(get_nested(config, "guards.forbidden") or []),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp072_exp063_full_replay_feature_cache",
        "implementation.enabled": True,
        "implementation.scope": "train_side_candidate_audit_only",
        "model.active_variants": ["roughening_x10"],
        "model.pf.particles": 500,
        "model.pf.seeds": 128,
        "model.pf.seed_indices": [0, 127],
        "model.pf.typewell_grid_step_ft": 0.2,
        "model.pf.initial_position_spread_ft": 4.5,
        "model.pf.initial_rate_spread": 0.01,
        "model.pf.momentum": 0.998,
        "model.pf.rate_noise": 0.002,
        "model.pf.position_noise": 0.005,
        "model.pf.control_rough_position": 0.1,
        "model.pf.control_rough_rate": 0.001,
        "model.pf.candidate_rough_position": 1.0,
        "model.pf.candidate_rough_rate": 0.01,
        "model.pf.roughening_multiplier": 10.0,
        "model.pf.resample_threshold_fraction": 0.5,
        "model.pf.resampling": "systematic",
        "model.pf.emission": "capped_gaussian",
        "model.pf.emission_clip_z2": 600.0,
        "model.pf.gr_sigma_clip": [10.0, 60.0],
        "model.pf.typewell_tvt_pad_ft": 100.0,
        "model.pf.missing_gr_policy": "linear_interpolate_both_directions_then_typewell_mean",
        "model.pf.output_dtype": "float32",
        "model.replay.seed_aggregation": "arithmetic_mean",
        "validation.expected_rows": 3783989,
        "validation.expected_wells": 773,
        "validation.expected_folds": [0, 1, 2, 3, 4],
        "validation.primary_control": "exp072_likpf_mean",
        "validation.primary_candidate": PRIMARY_CANDIDATE,
        "execution.scientific_variants": 1,
        "execution.candidate_pf_well_runs": 773,
        "execution.parent_pf_control_reruns": 0,
        "execution.seed_well_trajectories": 98944,
        "execution.particle_starts": 49472000,
        "execution.well_shard_count": 4,
        "execution.shard_policy": "deterministic_lpt_suffix_rows",
        "execution.lightgbm_configs": 0,
        "execution.trained_folds": 0,
        "execution.boosters": 0,
        "execution.hmm_well_runs": 0,
        "execution.beam_well_runs": 0,
        "execution.gpu_runs": 0,
        "runtime.num_workers": 1,
        "runtime.numba_num_threads": 1,
        "runtime.device": "cpu",
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "data.exp209_reconstructed_control.reconstruction_columns": [
            "hmm_mean_tvt",
            "hmm_minus_likpf_mean",
        ],
        "data.exp209_reconstructed_control.reconstruction": (
            "float32(hmm_mean_tvt - hmm_minus_likpf_mean)"
        ),
        "inference.enabled": False,
        "execution.inference_approved": False,
        "execution.submission_approved": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp416 fixed contract mismatch: {key} must be {value!r}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp416 implementation approval must be recorded")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_package_approved"))
        and bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.train_run_approved"))
    ):
        raise RuntimeError("exp416 Kaggle package/push/train run is not approved")
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Truth-free raw input checks and deterministic LPT sharding


# %%
def build_raw_well_manifest(config: Mapping[str, Any], raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        visible = pd.read_csv(horizontal_path, usecols=["TVT_input"])
        suffix_rows = int(pd.to_numeric(visible["TVT_input"], errors="coerce").isna().sum())
        rows.append(
            {
                "well_id": str(well),
                "suffix_rows": suffix_rows,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    identity_sha = dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if (
        len(frame) != expected_wells
        or int(frame["suffix_rows"].sum()) != expected_rows
        or frame["well_id"].duplicated().any()
        or identity_sha != expected_sha
    ):
        raise ValueError("exp416 raw train well identity or suffix coverage mismatch")
    frame.attrs["raw_identity_sha256"] = identity_sha
    return frame


def assign_lpt_shards(manifest: pd.DataFrame, shard_count: int = SHARD_COUNT) -> pd.DataFrame:
    required = {"well_id", "suffix_rows"}
    if not required.issubset(manifest.columns):
        raise ValueError("LPT manifest is missing well_id or suffix_rows")
    if manifest["well_id"].astype(str).duplicated().any() or shard_count <= 0:
        raise ValueError("LPT manifest must have unique wells and a positive shard count")
    loads = [0] * shard_count
    assignments: dict[str, int] = {}
    ordered = manifest.assign(well_id=manifest["well_id"].astype(str)).sort_values(
        ["suffix_rows", "well_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[str(row.well_id)] = shard
        loads[shard] += int(row.suffix_rows)
    result = manifest.copy()
    result["shard_index"] = result["well_id"].astype(str).map(assignments).astype(np.int8)
    result = result.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    result.attrs["shard_suffix_rows"] = {str(index): int(load) for index, load in enumerate(loads)}
    return result


def input_spec(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.{key}") or {}
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a mapping")
    return value


def exp209_reconstruction_columns(config: Mapping[str, Any]) -> list[str]:
    spec = input_spec(config, "exp209_reconstructed_control")
    columns = [str(value) for value in spec.get("reconstruction_columns", [])]
    expected = ["hmm_mean_tvt", "hmm_minus_likpf_mean"]
    if columns != expected:
        raise ValueError(f"exp209 reconstruction columns must be {expected!r}")
    formula = str(spec.get("reconstruction", ""))
    expected_formula = "float32(hmm_mean_tvt - hmm_minus_likpf_mean)"
    if formula != expected_formula:
        raise ValueError(f"exp209 reconstruction must be {expected_formula!r}")
    return columns


def reconstruct_exp209_likpf(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> np.ndarray:
    mean_column, delta_column = exp209_reconstruction_columns(config)
    mean = pd.to_numeric(frame[mean_column], errors="raise").to_numpy(np.float32)
    delta = pd.to_numeric(frame[delta_column], errors="raise").to_numpy(np.float32)
    reconstructed = (mean - delta).astype(np.float32, copy=False)
    if not np.isfinite(reconstructed).all():
        raise ValueError("exp209 reconstructed LiKPF contains non-finite values")
    return reconstructed


def preflight_late_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "exp072_control",
        "exp209_reconstructed_control",
        "fold_assignment",
        "hidden_like_assignment",
        "persistent_episodes",
        "counterfactual_sentinel_wells",
    )
    specs = {key: input_spec(config, key) for key in keys}
    paths = {
        key: resolve_existing(str(spec["filename"]), spec.get("candidates", []))
        for key, spec in specs.items()
    }
    reports: dict[str, Any] = {}
    for key in ("exp072_control", "exp209_reconstructed_control", "fold_assignment"):
        report = inspect_gzip_csv(paths[key])
        expected_decompressed = str(specs[key]["expected_decompressed_sha256"])
        if report["decompressed_sha256"] != expected_decompressed:
            raise ValueError(f"{key} decompressed SHA mismatch")
        expected_raw = specs[key].get("expected_raw_gzip_sha256")
        if expected_raw and report["raw_sha256"] != str(expected_raw):
            raise ValueError(f"{key} raw gzip SHA mismatch")
        reports[key] = report
    for key in ("hidden_like_assignment", "persistent_episodes", "counterfactual_sentinel_wells"):
        raw_sha = sha256_path(paths[key])
        if raw_sha != str(specs[key]["expected_sha256"]):
            raise ValueError(f"{key} raw SHA mismatch")
        reports[key] = {
            "path": str(paths[key]),
            "bytes": paths[key].stat().st_size,
            "raw_sha256": raw_sha,
            "columns": pd.read_csv(paths[key], nrows=0).columns.astype(str).tolist(),
        }
    required_columns = {
        "exp072_control": {"id", "last_known_tvt", "likpf_mean_d"},
        "exp209_reconstructed_control": {
            "id",
            *exp209_reconstruction_columns(config),
        },
        "fold_assignment": set(specs["fold_assignment"]["safe_columns"]),
        "hidden_like_assignment": {
            "well_id",
            *[str(value) for value in specs["hidden_like_assignment"]["role_columns"].values()],
        },
        "persistent_episodes": {
            "episode_id",
            "well",
            "start_row_idx",
            "end_row_idx_exclusive",
            "rows",
        },
        "counterfactual_sentinel_wells": {"well", "episodes", "episode_rows"},
    }
    for key, required in required_columns.items():
        missing = sorted(required - set(reports[key]["columns"]))
        if missing:
            raise ValueError(f"{key} missing required columns: {missing}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    for key in ("exp072_control", "exp209_reconstructed_control", "fold_assignment"):
        if int(reports[key]["data_rows"]) != expected_rows:
            raise ValueError(f"{key} row count mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "reports": reports,
        "truth_or_reporting_values_parsed_before_freeze": {
            "unknown_suffix_tvt_rows": 0,
            "control_prediction_rows": 0,
            "fold_rows": 0,
            "hidden_like_role_rows": 0,
            "persistent_episode_rows": 0,
        },
    }


# %% [markdown]
# ## 5. Exact exp072 input preparation


# %%
@dataclass
class TruthAccessLedger:
    prediction_frozen: bool = False
    unknown_suffix_tvt_rows_before_freeze: int = 0
    control_prediction_rows_before_freeze: int = 0
    fold_rows_before_freeze: int = 0
    hidden_like_role_rows_before_freeze: int = 0
    persistent_episode_rows_before_freeze: int = 0
    unknown_suffix_tvt_rows_after_freeze: int = 0
    control_prediction_rows_after_freeze: int = 0
    fold_rows_after_freeze: int = 0
    hidden_like_role_rows_after_freeze: int = 0
    persistent_episode_rows_after_freeze: int = 0

    def require_frozen(self) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("late reporting input requires a frozen prediction")

    def mark_frozen(self) -> None:
        if any(self.report()["before_freeze"].values()):
            raise RuntimeError("truth/reporting values were accessed before prediction freeze")
        self.prediction_frozen = True

    def report(self) -> dict[str, Any]:
        return {
            "prediction_frozen": self.prediction_frozen,
            "before_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_before_freeze,
                "control_prediction_rows": self.control_prediction_rows_before_freeze,
                "fold_rows": self.fold_rows_before_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_before_freeze,
                "persistent_episode_rows": self.persistent_episode_rows_before_freeze,
            },
            "after_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_after_freeze,
                "control_prediction_rows": self.control_prediction_rows_after_freeze,
                "fold_rows": self.fold_rows_after_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_after_freeze,
                "persistent_episode_rows": self.persistent_episode_rows_after_freeze,
            },
        }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    frame = frame[["MD", "Z", "GR", "TVT_input"]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "Z"]].isna().any().any():
        raise ValueError(f"{well}: MD/Z must be finite")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    if len(frame) < 2 or not np.isfinite(frame["TVT"].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    typewell_mean = float(frame["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(typewell_mean)
    return frame


def uniform_typewell_grid(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step: float = 0.2,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(typewell_tvt))
    maximum = float(np.max(typewell_tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    grid_gr = np.interp(grid_tvt, typewell_tvt, typewell_gr).astype(np.float64)
    return grid_gr, minimum, float(step)


def exp072_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    clip: tuple[float, float] = (10.0, 60.0),
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
    return {
        "raw_scale": raw_scale,
        "base_scale": float(np.clip(raw_scale, clip[0], clip[1])),
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
    }


def exp072_initial_rate(horizontal: pd.DataFrame, *, tail_rows: int = 30) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()].tail(tail_rows)
    delta_tvt = np.diff(known["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(known["Z"].to_numpy(np.float64))
    delta_md = np.diff(known["MD"].to_numpy(np.float64))
    valid = delta_md > 0
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    grid_step: float,
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires non-empty known prefix and unknown suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    last_known_tvt = float(last_known["TVT_input"])
    last_known_md = float(last_known["MD"])
    scale_audit = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr.mean()))
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    if not np.isfinite(eval_gr).all():
        raise ValueError("evaluation GR interpolation is not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": evaluation["Z"].to_numpy(np.float64),
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - last_known_md,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_known_tvt + float(last_known["Z"]),
        "initial_rate": exp072_initial_rate(horizontal),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "scale_audit": scale_audit,
    }


# %% [markdown]
# ## 6. Exact exp072 likelihood-PF kernel and roughening-only intervention


# %%
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
def _pf_lik_allseeds(
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
    initial_spread: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact exp072 state/RNG order plus passive counters."""
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = last_position + initial_spread * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = momentum * rate[particle] + rate_noise * np.random.randn()
                position[particle] += rate[particle] * delta_md + position_noise * np.random.randn()
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
                    new_position[particle] = position[cursor] + rough_position * np.random.randn()
                    new_rate[particle] = rate[cursor] + rough_rate * np.random.randn()
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles
                resampling_counts[seed_index] += 1
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
    )


def run_likelihood_pf(
    prepared: Mapping[str, Any],
    *,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.time()
    predictions, log_likelihoods, resampling_counts, minimum_ess, clip_counts = (
        _pf_lik_allseeds(
            prepared["eval_md"],
            prepared["eval_z"],
            prepared["eval_gr"],
            prepared["grid_gr"],
            float(prepared["grid_minimum"]),
            float(prepared["grid_step"]),
            float(prepared["scale_audit"]["base_scale"]),
            float(prepared["last_known_position"]),
            float(prepared["initial_rate"]),
            int(particles),
            int(seeds),
            int(seed_base),
            float(momentum),
            float(rate_noise),
            float(position_noise),
            float(rough_position),
            float(rough_rate),
            float(resample_fraction),
            float(initial_spread),
        )
    )
    candidate = predictions.mean(axis=0)
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "seed_loglik_mean_per_row": float(log_likelihoods.mean()) / len(candidate),
        "seed_loglik_best_per_row": float(log_likelihoods.max()) / len(candidate),
        "seed_loglik_spread": float(log_likelihoods.std()),
        "resampling_count_total": int(resampling_counts.sum()),
        "resampling_count_min": int(resampling_counts.min()),
        "resampling_count_max": int(resampling_counts.max()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "position_clip_count_total": int(clip_counts.sum()),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
    }
    return candidate, diagnostics


# %% [markdown]
# ## 7. Shard candidate generation and prediction freeze


# %%
def warm_up_pf_kernel() -> None:
    _pf_lik_allseeds(
        np.linspace(1.0, 8.0, 8),
        np.zeros(8),
        np.full(8, 50.0),
        np.linspace(45.0, 55.0, 100),
        0.0,
        0.2,
        20.0,
        50.0,
        0.0,
        8,
        2,
        1,
        0.998,
        0.002,
        0.005,
        1.0,
        0.010,
        0.5,
        4.5,
    )


def decode_well(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    pf = dict(get_nested(config, "model.pf") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(pf["typewell_grid_step_ft"]),
    )
    seed_base = stable_seed("likpf", "train", well)
    candidate_values, diagnostics = run_likelihood_pf(
        prepared,
        particles=int(pf["particles"]),
        seeds=int(pf["seeds"]),
        seed_base=seed_base,
        momentum=float(pf["momentum"]),
        rate_noise=float(pf["rate_noise"]),
        position_noise=float(pf["position_noise"]),
        rough_position=float(pf["candidate_rough_position"]),
        rough_rate=float(pf["candidate_rough_rate"]),
        resample_fraction=float(pf["resample_threshold_fraction"]),
        initial_spread=float(pf["initial_position_spread_ft"]),
    )
    eval_indices = prepared["eval_indices"]
    raw_observed = prepared["raw_gr_observed"]
    candidate = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices.astype(np.int64),
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": prepared["md_since"].astype(np.float64),
            "raw_gr_observed": raw_observed,
            PRIMARY_CANDIDATE: candidate_values.astype(np.float32),
        }
    )
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(prepared["scale_audit"]["known_rows"]),
        "prefix_gr_missing_rows": int(prepared["scale_audit"]["known_gr_missing_rows"]),
        "eval_rows": len(candidate),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "last_known_tvt": float(prepared["last_known_tvt"]),
        "last_known_position": float(prepared["last_known_position"]),
        "initial_rate": float(prepared["initial_rate"]),
        "gr_scale_raw": float(prepared["scale_audit"]["raw_scale"]),
        "gr_scale_clipped": float(prepared["scale_audit"]["base_scale"]),
        "seed_base": int(seed_base),
        "seed_first": int(seed_base),
        "seed_last": int(seed_base + int(pf["seeds"]) - 1),
        "seeds": int(pf["seeds"]),
        "particles": int(pf["particles"]),
        "rough_position": float(pf["candidate_rough_position"]),
        "rough_rate": float(pf["candidate_rough_rate"]),
        "seed_well_trajectories": int(pf["seeds"]),
        "particle_starts": int(pf["seeds"]) * int(pf["particles"]),
        **diagnostics,
        "wall_seconds": time.time() - started,
    }
    if not np.isfinite(candidate[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: candidate prediction contains non-finite values")
    return candidate, audit


def freeze_prediction_frame(
    candidate: pd.DataFrame,
    output_path: Path,
    *,
    ledger: TruthAccessLedger | None = None,
) -> dict[str, Any]:
    logical_columns = ["id", "well_id", "row_idx", *PREDICTION_COLUMNS]
    if candidate["id"].astype(str).duplicated().any() or candidate.duplicated(
        ["well_id", "row_idx"]
    ).any():
        raise ValueError("candidate row identity is duplicated")
    if not np.isfinite(candidate[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError("candidate prediction contains non-finite values")
    write_deterministic_gzip_csv(candidate, output_path)
    gzip_report = inspect_gzip_csv(output_path)
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].astype(str).nunique()),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "logical_columns": logical_columns,
        "logical_content_sha256": dataframe_content_sha(candidate, logical_columns),
        "schema_sha256": dataframe_schema_sha(candidate),
        "raw_gzip_sha256": gzip_report["raw_sha256"],
        "decompressed_sha256": gzip_report["decompressed_sha256"],
    }
    if ledger is not None:
        ledger.mark_frozen()
        frozen["truth_access_ledger_at_freeze"] = ledger.report()
    return frozen


def _require_frozen_prediction(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("late attachment requires a frozen prediction")
    if len(str(frozen.get("logical_content_sha256") or "")) != 64:
        raise RuntimeError("frozen prediction logical content SHA is missing")


def run_shard(
    config: Mapping[str, Any],
    shard_index: int,
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    contract = validate_scientific_contract(
        config,
        require_run_approval=require_run_approval,
    )
    if shard_index not in range(SHARD_COUNT):
        raise ValueError(f"shard_index must be in [0, {SHARD_COUNT - 1}]")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("exp416 full PF execution requires numba")
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError("exp416 PF shards must run first on Kaggle CPU")
    import numba

    numba.set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    started = time.time()
    raw_dir = train_data_dir(config)
    manifest = assign_lpt_shards(build_raw_well_manifest(config, raw_dir))
    selected = manifest.loc[manifest["shard_index"].eq(shard_index)].copy()
    if selected.empty:
        raise ValueError(f"shard {shard_index} has no wells")
    warm_up_pf_kernel()
    results = [
        decode_well(str(well), raw_dir, config)
        for well in selected["well_id"].astype(str).tolist()
    ]
    candidate = (
        pd.concat([result[0] for result in results], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([result[1] for result in results])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(candidate) != int(selected["suffix_rows"].sum())
        or candidate["well_id"].nunique() != len(selected)
        or len(audit) != len(selected)
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError(f"shard {shard_index} coverage mismatch")
    output = artifact_dir()
    prediction_path = output / f"{OUTPUT_PREFIX}_shard{shard_index}_candidate_predictions.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_shard{shard_index}_well_audit.csv"
    manifest_path = output / f"{OUTPUT_PREFIX}_shard{shard_index}_well_manifest.csv"
    contract_path = output / f"{OUTPUT_PREFIX}_scientific_contract.json"
    frozen = freeze_prediction_frame(candidate, prediction_path)
    audit.to_csv(audit_path, index=False)
    selected.to_csv(manifest_path, index=False)
    write_json(contract_path, contract)
    elapsed = time.time() - started
    pf = dict(get_nested(config, "model.pf") or {})
    source_path = resolve_package_file(SOURCE_FILENAME)
    config_path = resolve_package_file("config.yaml")
    raw_identity_sha = dataframe_content_sha(
        manifest,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "candidate_shard",
        "status": "complete",
        "route": "pf_beam",
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "counts": {
            "wells": int(len(selected)),
            "rows": int(len(candidate)),
            "scientific_variants": 1,
            "candidate_pf_well_runs": int(len(selected)),
            "seed_well_trajectories": int(len(selected) * int(pf["seeds"])),
            "particle_starts": int(
                len(selected) * int(pf["seeds"]) * int(pf["particles"])
            ),
            "parent_pf_control_reruns": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "frozen_prediction": frozen,
        "input": {
            "raw_train_dir": str(raw_dir),
            "raw_well_identity_sha256": raw_identity_sha,
            "raw_well_files_sha_recorded_in_manifest": True,
        },
        "implementation": {
            "source_path": str(source_path),
            "source_sha256": sha256_path(source_path),
            "config_path": str(config_path),
            "config_sha256": sha256_path(config_path),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": maximum_rss_gb(),
            "versions": runtime_versions(),
        },
        "artifacts": {
            "prediction": {
                "path": str(prediction_path),
                **inspect_gzip_csv(prediction_path),
                "logical_content_sha256": frozen["logical_content_sha256"],
            },
            "well_audit": {
                "path": str(audit_path),
                "raw_sha256": sha256_path(audit_path),
            },
            "well_manifest": {
                "path": str(manifest_path),
                "raw_sha256": sha256_path(manifest_path),
            },
            "scientific_contract": {
                "path": str(contract_path),
                "raw_sha256": sha256_path(contract_path),
            },
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_shard{shard_index}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 8. Strict shard merge and optional rerun probe


# %%
def _artifact_file(root: Path, filename: str) -> Path:
    direct = root / filename
    nested = root / "artifacts" / filename
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    matches = sorted(root.glob(f"**/{filename}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"expected one {filename} below {root}; found={matches}")


def merge_shard_outputs(
    shard_roots: Sequence[Path],
    output: Path,
    config: Mapping[str, Any],
    *,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    if len(shard_roots) != SHARD_COUNT:
        raise ValueError(f"exp416 merge requires exactly {SHARD_COUNT} shard roots")
    contract = validate_scientific_contract(config)
    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for shard_index, root in enumerate(shard_roots):
        summary_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_summary.json",
        )
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("stage") != "candidate_shard"
            or int(summary.get("shard_index", -1)) != shard_index
            or str(summary.get("scientific_contract_sha256"))
            != str(contract["scientific_contract_sha256"])
        ):
            raise ValueError(f"shard {shard_index} summary contract mismatch")
        prediction_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_candidate_predictions.csv.gz",
        )
        audit_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_well_audit.csv",
        )
        manifest_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_well_manifest.csv",
        )
        prediction = pd.read_csv(
            prediction_path,
            dtype={
                "id": str,
                "well_id": str,
                "row_idx": np.int64,
                "suffix_offset": np.int64,
                PRIMARY_CANDIDATE: np.float32,
            },
        )
        audit = pd.read_csv(audit_path, dtype={"well_id": str})
        manifest = pd.read_csv(manifest_path, dtype={"well_id": str})
        if (
            dataframe_content_sha(
                prediction,
                ["id", "well_id", "row_idx", *PREDICTION_COLUMNS],
            )
            != summary["frozen_prediction"]["logical_content_sha256"]
        ):
            raise ValueError(f"shard {shard_index} logical prediction SHA mismatch")
        if not manifest["shard_index"].astype(int).eq(shard_index).all():
            raise ValueError(f"shard {shard_index} manifest assignment mismatch")
        prediction_parts.append(prediction)
        audit_parts.append(audit)
        manifest_parts.append(manifest)
        summaries.append(summary)
    candidate = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.concat(audit_parts, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    manifest = (
        pd.concat(manifest_parts, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(candidate) != expected_rows
        or candidate["well_id"].nunique() != expected_wells
        or candidate["id"].duplicated().any()
        or candidate.duplicated(["well_id", "row_idx"]).any()
        or len(audit) != expected_wells
        or audit["well_id"].duplicated().any()
        or not audit["status"].eq("ok").all()
        or len(manifest) != expected_wells
        or manifest["well_id"].duplicated().any()
        or int(manifest["suffix_rows"].sum()) != expected_rows
    ):
        raise ValueError("strict exp416 shard merge coverage mismatch")
    expected_counts = {
        "candidate_pf_well_runs": int(get_nested(config, "execution.candidate_pf_well_runs")),
        "seed_well_trajectories": int(get_nested(config, "execution.seed_well_trajectories")),
        "particle_starts": int(get_nested(config, "execution.particle_starts")),
    }
    actual_counts = {
        "candidate_pf_well_runs": int(sum(item["counts"]["candidate_pf_well_runs"] for item in summaries)),
        "seed_well_trajectories": int(
            sum(item["counts"]["seed_well_trajectories"] for item in summaries)
        ),
        "particle_starts": int(sum(item["counts"]["particle_starts"] for item in summaries)),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"exp416 execution count mismatch: {actual_counts} != {expected_counts}")
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / f"{OUTPUT_PREFIX}_merged_candidate_predictions.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_merged_well_audit.csv"
    manifest_path = output / f"{OUTPUT_PREFIX}_merged_well_manifest.csv"
    frozen = freeze_prediction_frame(candidate, prediction_path, ledger=ledger)
    audit.to_csv(audit_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    frozen["execution_counts"] = actual_counts
    frozen["shard_logical_content_sha256"] = [
        item["frozen_prediction"]["logical_content_sha256"] for item in summaries
    ]
    return (
        candidate,
        audit,
        frozen,
        {
            "merged_prediction": prediction_path,
            "merged_well_audit": audit_path,
            "merged_well_manifest": manifest_path,
        },
    )


def probe_rerun_report(
    merged_candidate: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    probe_well: str,
) -> dict[str, Any]:
    expected = merged_candidate.loc[
        merged_candidate["well_id"].astype(str).eq(str(probe_well))
    ].sort_values("row_idx", kind="mergesort")
    observed, audit = decode_well(str(probe_well), raw_dir, config)
    observed = observed.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(
        expected["row_idx"].to_numpy(np.int64),
        observed["row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("probe rerun row identity mismatch")
    expected_values = expected[PRIMARY_CANDIDATE].to_numpy(np.float32)
    observed_values = observed[PRIMARY_CANDIDATE].to_numpy(np.float32)
    byte_identical = bool(np.array_equal(expected_values, observed_values))
    normalized_expected = expected.copy()
    normalized_expected[PRIMARY_CANDIDATE] = expected_values
    return {
        "probe_well": str(probe_well),
        "rows": len(observed),
        "byte_identical_float32": byte_identical,
        "maximum_absolute_difference_ft": float(
            np.max(np.abs(expected_values.astype(np.float64) - observed_values.astype(np.float64)))
        ),
        "expected_logical_content_sha256": dataframe_content_sha(
            normalized_expected,
            ["id", "well_id", "row_idx", *PREDICTION_COLUMNS],
        ),
        "observed_logical_content_sha256": dataframe_content_sha(
            observed,
            ["id", "well_id", "row_idx", *PREDICTION_COLUMNS],
        ),
        "audit": audit,
    }


# %% [markdown]
# ## 9. Late truth, saved-control, fold, hidden-like, and episode attachment


# %%
def load_unknown_suffix_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["TVT_input", "TVT"],
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    true_tvt = pd.to_numeric(horizontal["TVT"], errors="coerce")
    eval_indices = np.flatnonzero(tvt_input.isna().to_numpy()).astype(np.int64)
    values = true_tvt.iloc[eval_indices].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{well}: unknown-suffix TVT contains non-finite values")
    return pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "true_tvt": values,
        }
    )


def align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    lookup_source = source.copy()
    lookup_source["id"] = lookup_source["id"].astype(str)
    if lookup_source["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = lookup_source.set_index("id").reindex(frame["id"].astype(str))
    if aligned[list(columns)].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def load_late_readout_frame(
    candidate: pd.DataFrame,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any],
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _require_frozen_prediction(frozen)
    ledger.require_frozen()
    reverified = dataframe_content_sha(candidate, list(frozen["logical_columns"]))
    if reverified != str(frozen["logical_content_sha256"]):
        raise ValueError("in-memory candidate changed after prediction freeze")
    wells = sorted(candidate["well_id"].astype(str).unique().tolist())
    truth = (
        pd.concat([load_unknown_suffix_truth(well, raw_dir) for well in wells], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    ledger.unknown_suffix_tvt_rows_after_freeze += len(truth)
    frame = align_on_id(candidate, truth, ["true_tvt"], label="raw suffix truth")

    exp072 = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=["id", "last_known_tvt", "likpf_mean_d"],
        dtype={"id": str},
    )
    exp072["saved_exp072_likpf_mean"] = (
        pd.to_numeric(exp072["last_known_tvt"], errors="raise")
        + pd.to_numeric(exp072["likpf_mean_d"], errors="raise")
    )
    frame = align_on_id(
        frame,
        exp072[["id", "saved_exp072_likpf_mean"]],
        ["saved_exp072_likpf_mean"],
        label="saved exp072 control",
    )

    exp209_columns = exp209_reconstruction_columns(config)
    exp209 = pd.read_csv(
        preflight["paths"]["exp209_reconstructed_control"],
        usecols=["id", *exp209_columns],
        dtype={"id": str},
    )
    exp209["exp209_reconstructed_likpf_mean"] = reconstruct_exp209_likpf(
        exp209,
        config,
    )
    exp209 = exp209[["id", "exp209_reconstructed_likpf_mean"]]
    frame = align_on_id(
        frame,
        exp209,
        ["exp209_reconstructed_likpf_mean"],
        label="exp209 reconstructed exp072 control",
    )
    ledger.control_prediction_rows_after_freeze += len(frame) * 2

    fold_spec = input_spec(config, "fold_assignment")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp416 fold allowlist must contain identity/fold columns only")
    fold = pd.read_csv(
        preflight["paths"]["fold_assignment"],
        usecols=safe_columns,
        dtype={"well_id": str},
    )
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("reporting fold identity is duplicated")
    ledger.fold_rows_after_freeze += len(fold)
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_exp226"),
        sort=False,
    )
    if frame[["fold", "suffix_offset_exp226"]].isna().any().any():
        raise ValueError("reporting fold attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["suffix_offset_exp226"].to_numpy(np.int64),
    ):
        raise ValueError("exp226 suffix offset identity mismatch")
    frame = frame.drop(columns=["suffix_offset_exp226"])

    hidden_spec = input_spec(config, "hidden_like_assignment")
    role_columns = {
        str(scope): str(column) for scope, column in hidden_spec["role_columns"].items()
    }
    hidden = pd.read_csv(
        preflight["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    if hidden["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment has duplicate wells")
    expected_role_counts = hidden_spec.get("expected_role_counts") or {}
    for scope, column in role_columns.items():
        actual = {
            str(key): int(value)
            for key, value in hidden[column].astype(str).value_counts().sort_index().items()
        }
        expected = {
            str(key): int(value)
            for key, value in (expected_role_counts.get(scope) or {}).items()
        }
        if actual != expected:
            raise ValueError(f"hidden-like role counts mismatch for {scope}")
    ledger.hidden_like_role_rows_after_freeze += len(hidden)
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[role_columns["hidden_like_spatial"]].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[
        role_columns["hidden_like_typewell_purged"]
    ].eq("valid")

    sentinels = pd.read_csv(
        preflight["paths"]["counterfactual_sentinel_wells"],
        dtype={"well": str},
    )
    episodes = pd.read_csv(
        preflight["paths"]["persistent_episodes"],
        dtype={"episode_id": str, "well": str},
    )
    selected_episodes = episodes.loc[
        episodes["well"].isin(sentinels["well"].astype(str))
    ].copy()
    expected_wells = int(get_nested(config, "data.counterfactual_sentinel_wells.expected_wells"))
    expected_episodes = int(get_nested(config, "data.persistent_episodes.expected_selected_episodes"))
    expected_episode_rows = int(
        get_nested(config, "data.persistent_episodes.expected_selected_episode_rows")
    )
    if (
        sentinels["well"].nunique() != expected_wells
        or selected_episodes["episode_id"].nunique() != expected_episodes
        or int(selected_episodes["rows"].sum()) != expected_episode_rows
    ):
        raise ValueError("exp410 sentinel episode identity changed")
    ledger.persistent_episode_rows_after_freeze += len(selected_episodes)
    if not np.isfinite(
        frame[
            [
                "true_tvt",
                "saved_exp072_likpf_mean",
                "exp209_reconstructed_likpf_mean",
                *PREDICTION_COLUMNS,
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("late readout contains non-finite values")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("reporting fold set mismatch")
    return frame, selected_episodes, {
        "truth_attached_after_prediction_freeze": True,
        "candidate_content_sha256_reverified": reverified,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "persistent_episode_count": len(selected_episodes),
        "persistent_episode_rows": int(selected_episodes["rows"].sum()),
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 10. Metrics and fail-closed scientific gate


# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"metric scope {scope} is empty")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[PRIMARY_CANDIDATE].to_numpy(np.float64)
    control = selected["saved_exp072_likpf_mean"].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate": PRIMARY_CANDIDATE,
        "candidate_rmse": candidate_rmse,
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "candidate_bias": float(np.mean(candidate - truth)),
        "candidate_within_10ft": float(np.mean(np.abs(candidate - truth) <= 10.0)),
        "control": "saved_exp072_likpf_mean",
        "control_rmse": control_rmse,
        "control_mae": float(np.mean(np.abs(control - truth))),
        "control_bias": float(np.mean(control - truth)),
        "control_within_10ft": float(np.mean(np.abs(control - truth) <= 10.0)),
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
    }


def metric_scopes(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool)),
    ]
    for fold in sorted(frame["fold"].astype(int).unique().tolist()):
        scopes.append((f"fold_{fold}", frame["fold"].eq(fold).to_numpy()))
    scopes.extend(
        [
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
            ("md_since_1000_plus", frame["md_since"].ge(1000.0).to_numpy()),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    return scopes


def build_metric_outputs(
    frame: pd.DataFrame,
    selected_episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            metric_record(frame, mask, scope=scope)
            for scope, mask in metric_scopes(frame)
        ]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        candidate = group[PRIMARY_CANDIDATE].to_numpy(np.float64)
        control = group["saved_exp072_likpf_mean"].to_numpy(np.float64)
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
            }
        )
    episode_rows: list[dict[str, Any]] = []
    for episode in selected_episodes.itertuples(index=False):
        selected = frame.loc[
            frame["well_id"].astype(str).eq(str(episode.well))
            & frame["row_idx"].ge(int(episode.start_row_idx))
            & frame["row_idx"].lt(int(episode.end_row_idx_exclusive))
        ]
        if len(selected) != int(episode.rows):
            raise ValueError(f"{episode.episode_id}: persistent episode coverage mismatch")
        truth = selected["true_tvt"].to_numpy(np.float64)
        candidate = selected[PRIMARY_CANDIDATE].to_numpy(np.float64)
        control = selected["saved_exp072_likpf_mean"].to_numpy(np.float64)
        candidate_sse = float(np.square(candidate - truth).sum())
        control_sse = float(np.square(control - truth).sum())
        episode_rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well_id": str(episode.well),
                "rows": len(selected),
                "candidate_sse": candidate_sse,
                "control_sse": control_sse,
                "candidate_rmse": math.sqrt(candidate_sse / len(selected)),
                "control_rmse": math.sqrt(control_sse / len(selected)),
                "sse_reduction_fraction": 1.0 - candidate_sse / control_sse,
                "improved": candidate_sse < control_sse,
            }
        )
    return overall, pd.DataFrame(by_well_rows), pd.DataFrame(episode_rows)


def scope_row(metrics: pd.DataFrame, scope: str) -> pd.Series:
    selected = metrics.loc[metrics["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one metric row for scope={scope}")
    return selected.iloc[0]


def evaluate_gate(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    ledger: TruthAccessLedger,
    shard_summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    probe_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical") or {})
    scientific_config = dict(get_nested(config, "guards.scientific") or {})
    overall = scope_row(metrics, "overall")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    control_expected = float(get_nested(config, "validation.saved_control_rmse_ft"))
    control_parity_difference = abs(float(overall["control_rmse"]) - control_expected)
    exp209_control_parity_max_abs = float(
        np.max(
            np.abs(
                frame["saved_exp072_likpf_mean"].to_numpy(np.float64)
                - frame["exp209_reconstructed_likpf_mean"].to_numpy(np.float64)
            )
        )
    )
    actual_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": len(audit),
        "parent_pf_control_reruns": 0,
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "reporting_folds": int(frame["fold"].nunique()),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        key: int(get_nested(config, f"execution.{key}"))
        for key in actual_counts
    }
    runtime_limit = float(get_nested(config, "runtime.hard_seconds_per_shard"))
    shard_runtime_seconds = [
        float(summary["runtime"]["elapsed_seconds"]) for summary in shard_summaries
    ]
    before_freeze = ledger.report()["before_freeze"]
    parameter_diff = roughening_only_parameter_diff(config)
    probe_byte_identical = (
        bool(probe_report.get("byte_identical_float32")) if probe_report is not None else False
    )
    shard_peak_rss_gb = [
        float(summary["runtime"]["peak_rss_gb"]) for summary in shard_summaries
    ]
    peak_rss_limit = float(get_nested(config, "runtime.maximum_peak_rss_gb"))
    technical = {
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "audit_wells": len(audit),
        "all_wells_completed_without_fallback": bool(audit["status"].eq("ok").all()),
        "finite_candidate_coverage": float(
            np.isfinite(frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).mean()
        ),
        "saved_control_rmse_parity_absolute_difference_ft": control_parity_difference,
        "exp209_reconstructed_control_row_parity_max_abs_ft": exp209_control_parity_max_abs,
        "truth_or_reporting_values_parsed_before_freeze": before_freeze,
        "execution_counts": actual_counts,
        "execution_count_match": actual_counts == expected_counts,
        "roughening_only_parameter_diff": parameter_diff,
        "scientific_contract_sha256": build_scientific_contract(config)[
            "scientific_contract_sha256"
        ],
        "prediction_logical_content_sha256": frozen["logical_content_sha256"],
        "shard_count": len(shard_summaries),
        "shard_runtime_seconds": shard_runtime_seconds,
        "runtime_limit_seconds_per_shard": runtime_limit,
        "shard_peak_rss_gb": shard_peak_rss_gb,
        "peak_rss_limit_gb_per_shard": peak_rss_limit,
        "probe_rerun_available": probe_report is not None,
        "probe_rerun_byte_identical_float32": probe_byte_identical,
    }
    technical["passed"] = bool(
        technical["prediction_rows"] == expected_rows
        and technical["prediction_wells"] == expected_wells
        and technical["reporting_folds"] == expected_folds
        and technical["audit_wells"] == expected_wells
        and technical["all_wells_completed_without_fallback"]
        and technical["finite_candidate_coverage"]
        == float(technical_config["require_finite_candidate_coverage"])
        and control_parity_difference
        <= float(technical_config["require_saved_control_rmse_parity_atol_ft"])
        and exp209_control_parity_max_abs
        <= float(technical_config["require_exp209_control_row_parity_atol_ft"])
        and all(int(value) == 0 for value in before_freeze.values())
        and actual_counts == expected_counts
        and set(parameter_diff) == {"rough_position", "rough_rate"}
        and len(shard_summaries) == SHARD_COUNT
        and all(value <= runtime_limit for value in shard_runtime_seconds)
        and all(value <= peak_rss_limit for value in shard_peak_rss_gb)
    )

    fold_rows = metrics.loc[metrics["scope"].str.startswith("fold_")]
    improved_folds = int((fold_rows["improvement_ft"] > 0.0).sum())
    observed = scope_row(metrics, "raw_gr_observed")
    non_regression_limits = {
        "raw_gr_missing": float(scientific_config["maximum_raw_gr_missing_regression_ft"]),
        "md_since_1000_plus": float(
            scientific_config["maximum_long_tail_1000_plus_regression_ft"]
        ),
        "hidden_like_spatial": float(
            scientific_config["maximum_hidden_like_spatial_regression_ft"]
        ),
        "hidden_like_typewell_purged": float(
            scientific_config["maximum_hidden_like_typewell_purged_regression_ft"]
        ),
    }
    non_regression_scopes = {
        scope: float(scope_row(metrics, scope)["delta_rmse_candidate_minus_control"]) <= limit
        for scope, limit in non_regression_limits.items()
    }
    by_well_delta = by_well["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well = float(by_well_delta.max())
    candidate_episode_sse = float(episode_metrics["candidate_sse"].sum())
    control_episode_sse = float(episode_metrics["control_sse"].sum())
    episode_sse_reduction = 1.0 - candidate_episode_sse / control_episode_sse
    scientific = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            scientific_config["minimum_direct_rmse_gain_vs_exp072_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(scientific_config["minimum_improved_folds"]),
        "raw_gr_observed_improvement_ft": float(observed["improvement_ft"]),
        "minimum_raw_gr_observed_improvement_ft": float(
            scientific_config["minimum_raw_gr_observed_gain_ft"]
        ),
        "non_regression_scopes": non_regression_scopes,
        "by_well_rmse_delta_p95": by_well_p95,
        "maximum_by_well_rmse_delta_p95": float(
            scientific_config["maximum_by_well_delta_p95_ft"]
        ),
        "worst_well_rmse_regression": worst_well,
        "maximum_worst_well_rmse_regression": float(
            scientific_config["maximum_worst_well_regression_ft"]
        ),
        "persistent_episode_count": len(episode_metrics),
        "persistent_episode_rows": int(episode_metrics["rows"].sum()),
        "persistent_episode_candidate_sse": candidate_episode_sse,
        "persistent_episode_control_sse": control_episode_sse,
        "persistent_episode_sse_reduction_fraction": episode_sse_reduction,
        "minimum_persistent_episode_sse_reduction_fraction": float(
            scientific_config["minimum_persistent_offset_sse_reduction_fraction"]
        ),
    }
    scientific["passed"] = bool(
        scientific["improvement_ft"] >= scientific["minimum_improvement_ft"]
        and improved_folds >= scientific["minimum_improved_folds"]
        and scientific["raw_gr_observed_improvement_ft"]
        >= scientific["minimum_raw_gr_observed_improvement_ft"]
        and all(non_regression_scopes.values())
        and by_well_p95 <= scientific["maximum_by_well_rmse_delta_p95"]
        and worst_well <= scientific["maximum_worst_well_rmse_regression"]
        and episode_sse_reduction
        >= scientific["minimum_persistent_episode_sse_reduction_fraction"]
    )
    passed = bool(technical["passed"] and scientific["passed"])
    deterministic_anchor_eligible = bool(passed and probe_byte_identical)
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            str(get_nested(config, "guards.decision.pass_action"))
            if passed
            else str(get_nested(config, "guards.decision.fail_action"))
        ),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "deterministic_anchor_eligible": deterministic_anchor_eligible,
        "deterministic_anchor_blocker": (
            None
            if deterministic_anchor_eligible
            else (
                "scientific_or_technical_gate_failed"
                if not passed
                else "fixed_probe_rerun_parity_not_recorded"
            )
        ),
        "failure_action": (
            "close_without_roughening_multiplier_position_rate_process_noise_ess_"
            "gr_sigma_seed_particle_well_row_gate_or_same_oof_rescue"
        ),
    }


# %% [markdown]
# ## 11. Generated artifacts and stage orchestration


# %%
def artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }
    if path.suffix == ".gz":
        report["decompressed_sha256"] = inspect_gzip_csv(path)["decompressed_sha256"]
    return report


def load_optional_probe_report(config: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = get_nested(config, "reproducibility.probe_report")
    if not isinstance(spec, Mapping) or not spec.get("filename"):
        return None
    path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    report = json.loads(path.read_text())
    expected_sha = spec.get("expected_sha256")
    if expected_sha and sha256_path(path) != str(expected_sha):
        raise ValueError("probe report SHA mismatch")
    return report


def resolve_shard_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [Path(str(value)) for value in get_nested(config, "execution.merge_shard_dirs") or []]
    if len(roots) != SHARD_COUNT:
        raise ValueError("execution.merge_shard_dirs must contain four ordered shard roots")
    return roots


def run_probe_stage(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=require_run_approval)
    if require_run_approval and not bool(get_nested(config, "execution.probe_run_approved")):
        raise RuntimeError("exp416 probe rerun is not approved")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("exp416 PF probe execution requires numba")
    spec = dict(get_nested(config, "reproducibility.probe_source") or {})
    merged_path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    merged = pd.read_csv(merged_path, dtype={"id": str, "well_id": str})
    probe_well = str(get_nested(config, "reproducibility.probe_well"))
    report = probe_rerun_report(merged, train_data_dir(config), config, probe_well)
    output_path = artifact_dir() / f"{OUTPUT_PREFIX}_probe_rerun_report.json"
    write_json(output_path, report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return report


def run_merge_stage(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    contract = validate_scientific_contract(
        config,
        require_run_approval=require_run_approval,
    )
    started = time.time()
    preflight = preflight_late_inputs(config)
    ledger = TruthAccessLedger()
    output = artifact_dir()
    shard_roots = resolve_shard_roots(config)
    candidate, audit, frozen, merged_paths = merge_shard_outputs(
        shard_roots,
        output,
        config,
        ledger=ledger,
    )
    shard_summaries = [
        json.loads(
            _artifact_file(root, f"{OUTPUT_PREFIX}_shard{index}_summary.json").read_text()
        )
        for index, root in enumerate(shard_roots)
    ]
    raw_dir = train_data_dir(config)
    expected_manifest = assign_lpt_shards(build_raw_well_manifest(config, raw_dir))
    observed_manifest = pd.read_csv(
        merged_paths["merged_well_manifest"],
        dtype={"well_id": str},
    ).sort_values("well_id", kind="mergesort")
    manifest_columns = ["well_id", "suffix_rows", "shard_index"]
    expected_values = expected_manifest[manifest_columns].astype(
        {"well_id": str, "suffix_rows": np.int64, "shard_index": np.int64}
    )
    observed_values = observed_manifest[manifest_columns].astype(
        {"well_id": str, "suffix_rows": np.int64, "shard_index": np.int64}
    )
    if not np.array_equal(
        expected_values.to_numpy(),
        observed_values.to_numpy(),
    ):
        raise ValueError("merged shard manifest does not match deterministic raw-data LPT")
    frame, selected_episodes, late_attachment = load_late_readout_frame(
        candidate,
        frozen,
        preflight,
        raw_dir,
        config,
        ledger,
    )
    metrics, by_well, episode_metrics = build_metric_outputs(frame, selected_episodes)
    probe_report = load_optional_probe_report(config)
    gate = evaluate_gate(
        frame,
        metrics,
        by_well,
        episode_metrics,
        audit,
        frozen,
        ledger,
        shard_summaries,
        config,
        probe_report=probe_report,
    )
    paths = {
        **merged_paths,
        "overall_fold_scope_metrics": output / f"{OUTPUT_PREFIX}_overall_fold_scope_metrics.csv",
        "by_well_metrics": output / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "persistent_episode_metrics": output / f"{OUTPUT_PREFIX}_persistent_episode_metrics.csv",
        "scientific_gate": output / f"{OUTPUT_PREFIX}_scientific_gate.json",
        "scientific_contract": output / f"{OUTPUT_PREFIX}_scientific_contract.json",
    }
    metrics.to_csv(paths["overall_fold_scope_metrics"], index=False)
    by_well.to_csv(paths["by_well_metrics"], index=False)
    episode_metrics.to_csv(paths["persistent_episode_metrics"], index=False)
    write_json(paths["scientific_gate"], gate)
    write_json(paths["scientific_contract"], contract)
    artifact_manifest = pd.DataFrame(
        [{"name": name, **artifact_report(path)} for name, path in paths.items()]
    ).sort_values("name", kind="mergesort")
    artifact_manifest_path = output / f"{OUTPUT_PREFIX}_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    status = (
        "train_side_roughening_x10_full_oof_gate_passed_no_automatic_downstream"
        if gate["passed"]
        else "train_side_roughening_x10_full_oof_gate_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "runtime_seconds_merge_and_readout": time.time() - started,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "scientific_variants": 1,
        "candidate_pf_well_runs": int(audit["well_id"].nunique()),
        "parent_pf_control_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "frozen_prediction": frozen,
        "truth_attachment": late_attachment,
        "gate": gate,
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
        },
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    overall = scope_row(metrics, "overall")
    metrics_json = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": float(overall["candidate_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": overall.to_dict(),
        "gate": gate,
        "prediction_sha256": frozen["logical_content_sha256"],
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Train-side candidate only. No saved exp072 control PF rerun, model, "
            "raw-test prediction, inference, or submission is produced."
        ),
    }
    write_json(metrics_output_path(), metrics_json)
    print(metrics.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def selected_stage(config: Mapping[str, Any]) -> str | None:
    value = os.environ.get("EXP416_STAGE") or get_nested(config, "execution.selected_stage")
    if value in (None, "", "preview"):
        return None
    return str(value)


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    if stage == "shard":
        raw_index = os.environ.get("EXP416_SHARD_INDEX")
        shard_index = (
            int(raw_index)
            if raw_index is not None
            else int(get_nested(config, "execution.selected_shard_index"))
        )
        return run_shard(config, shard_index)
    if stage == "probe":
        return run_probe_stage(config)
    if stage == "merge":
        return run_merge_stage(config)
    raise ValueError(f"unknown exp416 execution stage: {stage}")


# %% [markdown]
# ## 12. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    PREVIEW = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(CONFIG, "experiment.route"),
        "parent": get_nested(CONFIG, "lineage.parent"),
        "primary_candidate": PRIMARY_CANDIDATE,
        "parameter_diff": roughening_only_parameter_diff(CONFIG),
        "scientific_variants": get_nested(CONFIG, "execution.scientific_variants"),
        "candidate_pf_well_runs": get_nested(CONFIG, "execution.candidate_pf_well_runs"),
        "parent_pf_control_reruns": get_nested(
            CONFIG,
            "execution.parent_pf_control_reruns",
        ),
        "seed_well_trajectories": get_nested(
            CONFIG,
            "execution.seed_well_trajectories",
        ),
        "particle_starts": get_nested(CONFIG, "execution.particle_starts"),
        "well_shard_count": get_nested(CONFIG, "execution.well_shard_count"),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "canonical_notebook_adoption_approved": get_nested(
            CONFIG,
            "execution.canonical_notebook_adoption_approved",
        ),
        "kaggle_package_approved": get_nested(CONFIG, "execution.kaggle_package_approved"),
        "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
        "train_run_approved": get_nested(CONFIG, "execution.train_run_approved"),
        "selected_stage": selected_stage(CONFIG),
    }
    print(json.dumps(to_jsonable(PREVIEW), indent=2, sort_keys=True))
    SUMMARY = run_selected_stage(CONFIG)
