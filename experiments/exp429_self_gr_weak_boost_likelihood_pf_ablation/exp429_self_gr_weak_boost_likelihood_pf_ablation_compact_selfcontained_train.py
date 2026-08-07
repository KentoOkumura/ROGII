# %% [markdown]
# # exp429 self-GR weak-boost likelihood-PF ablation train
#
# Train-side audit of one preregistered change to the deterministic exp072
# likelihood-weighted particle filter. The fixed exp223 visible-prefix self-GR
# boost-only surface is added directly to each particle observation
# log-likelihood. Particle dynamics, x1.0 GR scale, seed policy, resampling,
# and fixed temperature-5 aggregation remain unchanged. Unknown-suffix TVT and
# reporting roles are read only after predictions and surface identity freeze.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Truth-free PF inputs and self-GR surface
# 5. Exact exp072 likelihood-PF kernel with self-GR boost
# 6. Technical preflight, shard generation, and prediction freeze
# 7. Strict shard merge and late reporting attachment
# 8. Paired metrics and fail-closed gates
# 9. Generated artifacts and stage orchestration
# 10. Setup and configuration preview
# 11. Run the selected Kaggle CPU stage

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


EXPERIMENT_NAME = "exp429_self_gr_weak_boost_likelihood_pf_ablation"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "likpf_scale5_selfgr_boost_only_a070_c100"
SECONDARY_CANDIDATE = "likpf_mean_selfgr_boost_only_a070_c100"
PRIMARY_CONTROL = "saved_exp404_scale5_x1p0"
SECONDARY_CONTROL = "saved_exp072_likpf_mean"
PREFLIGHT_ALPHA0_CONTROL = "saved_exp404_likpf_mean_x1p0"
PREDICTION_COLUMNS = (PRIMARY_CANDIDATE, SECONDARY_CANDIDATE)
SHARD_COUNT = 4
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP429_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def restore_frozen_float32_column(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite values")
    restored = numeric.astype(np.float32)
    if not np.isfinite(restored).all():
        raise ValueError(f"{label} cannot be represented as finite float32")
    return pd.Series(restored, index=values.index, name=values.name)


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
    raise FileNotFoundError(f"exp429 config not found; checked={checked}")


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
    csv_path = Path(path)
    return {
        "path": str(csv_path),
        "bytes": csv_path.stat().st_size,
        "raw_sha256": sha256_path(csv_path),
        "decompressed_sha256": digest.hexdigest(),
        "content_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(csv_path, nrows=0, compression="gzip")
        .columns.astype(str)
        .tolist(),
    }


def dataframe_content_sha(
    frame: pd.DataFrame, columns: list[str] | tuple[str, ...] | None = None
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


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


# %% [markdown]
# ## 3. Frozen scientific contract and input preflight


# %%
def validate_scientific_contract(
    config: dict[str, Any], *, require_run_approval: bool = False
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "implementation.enabled": True,
        "model.active_scientific_variants": [PRIMARY_CANDIDATE],
        "model.self_gr_surface.source_contract": "exp223_exact",
        "model.self_gr_surface.mode": "boost_only",
        "model.self_gr_surface.alpha": 0.07,
        "model.self_gr_surface.clip": 1.0,
        "model.self_gr_surface.window_radius_rows": 12,
        "model.self_gr_surface.descriptor_offsets": [-12, -8, -4, 0, 4, 8, 12],
        "model.self_gr_surface.top_k": 5,
        "model.self_gr_surface.prefix_anchor_stride": 3,
        "model.self_gr_surface.max_prefix_anchors": 128,
        "model.self_gr_surface.keep_last_prefix_anchors": 32,
        "model.self_gr_surface.min_prefix_anchors": 12,
        "model.self_gr_surface.max_window_missing_rate": 0.35,
        "model.self_gr_surface.gaussian_sigma_tvt_ft": 12.0,
        "model.self_gr_surface.descriptor_distance_temperature": 1.5,
        "model.self_gr_surface.typewell_agreement_sigma_tvt_ft": 18.0,
        "model.self_gr_surface.surface_quadratic_clip": 60.0,
        "model.self_gr_surface.grid_step_ft": 0.2,
        "model.self_gr_surface.grid_pad_ft": 100.0,
        "model.typewell_gr_emission.multiplier": 1.0,
        "model.typewell_gr_emission.post_multiplier_clip": None,
        "model.typewell_gr_emission.z2_clip": 600.0,
        "model.pf.particles": 500,
        "model.pf.seeds": 128,
        "model.pf.primary_seed_weighting_temperature": 5.0,
        "model.pf.secondary_seed_aggregation": "arithmetic_mean",
        "model.pf.other_seed_weighting_scales_enabled": False,
        "model.pf.initial_position_spread_ft": 4.5,
        "model.pf.initial_rate_spread": 0.01,
        "model.pf.momentum": 0.998,
        "model.pf.rate_noise": 0.002,
        "model.pf.position_noise": 0.005,
        "model.pf.rough_position": 0.1,
        "model.pf.rough_rate": 0.001,
        "model.pf.resample_threshold_fraction": 0.5,
        "model.pf.typewell_tvt_pad_ft": 100.0,
        "model.execution_count.scientific_variants": 1,
        "model.execution_count.full_candidate_pf_well_runs": 773,
        "model.execution_count.full_seed_well_trajectories": 98944,
        "model.execution_count.full_particle_starts": 49472000,
        "model.execution_count.full_shards": 4,
        "model.execution_count.parent_full_control_reruns": 0,
        "model.execution_count.hmm_well_runs": 0,
        "model.execution_count.beam_well_runs": 0,
        "model.execution_count.boosters": 0,
        "model.execution_count.models": 0,
        "model.execution_count.gpu_runs": 0,
        "data.exp404_scale5_control.arithmetic_prediction_column": (
            "likpf_mean_x1p0"
        ),
        "data.exp404_scale5_control.arithmetic_prediction_dtype": "float32",
        "guards.technical.require_preflight_alpha0_comparator": (
            "saved_exp404_likpf_mean_x1p0_bit_exact"
        ),
        "guards.technical.require_preflight_alpha0_bit_exact": True,
        "runtime.num_workers": 2,
        "runtime.numba_num_threads": 1,
        "runtime.full_shard_count": 4,
        "runtime.device": "cpu",
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp429 fixed contract mismatch: {key} must be {value!r}")
    if not bool(get_nested(config, "implementation.implementation_approval_received")):
        raise ValueError("exp429 implementation approval must be recorded")
    if require_run_approval:
        stage = selected_stage(config)
        approvals = {
            "preflight": "execution.preflight_run_approved",
            "full_shard": "execution.full_run_approved",
            "merge": "execution.full_run_approved",
        }
        if stage is None:
            raise RuntimeError("exp429 has no approved execution stage selected")
        if not bool(get_nested(config, "execution.kaggle_package_approved")):
            raise RuntimeError("exp429 Kaggle package is not approved")
        if not bool(get_nested(config, approvals[stage])):
            raise RuntimeError(f"exp429 {stage} run is not approved")
    contract = build_scientific_contract(config)
    return contract


def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "truth_attached": False,
        "primary_candidate": PRIMARY_CANDIDATE,
        "secondary_candidate": SECONDARY_CANDIDATE,
        "primary_control": get_nested(config, "validation.primary_control"),
        "secondary_control": get_nested(config, "validation.secondary_control"),
        "typewell_gr_emission": get_nested(config, "model.typewell_gr_emission"),
        "self_gr_surface": get_nested(config, "model.self_gr_surface"),
        "pf": get_nested(config, "model.pf"),
        "execution_counts": get_nested(config, "model.execution_count"),
        "truth_freeze_policy": get_nested(config, "validation.leakage_policy"),
        "controls": {
            "exp072_pf": "saved_load_only",
            "exp404_scale5": "saved_load_only_primary",
            "exp209_hmm": "saved_load_only_fixed_50_50_guard",
            "pf_control_reruns": 0,
            "hmm_reruns": 0,
            "beam_reruns": 0,
        },
        "forbidden": get_nested(config, "execution.forbidden_same_experiment_rescues"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


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
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_wells or actual != expected:
        raise ValueError("current raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": actual,
        "well_ids": frame["well_id"].astype(str).tolist(),
    }


def _input_spec(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.{key}") or {}
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a mapping")
    return value


def preflight_saved_inputs(config: dict[str, Any]) -> dict[str, Any]:
    specs = {
        "exp072_control": _input_spec(config, "exp072_control"),
        "exp404_scale5_control": _input_spec(config, "exp404_scale5_control"),
        "exp209_hmm_control": _input_spec(config, "exp209_hmm_control"),
        "fold_assignment": _input_spec(config, "exp226_reporting"),
        "hidden_like_assignment": _input_spec(config, "exp115_hidden_like"),
    }
    paths = {
        name: resolve_existing(str(spec["filename"]), spec.get("candidates", []))
        for name, spec in specs.items()
    }
    reports: dict[str, Any] = {}
    for name in (
        "exp072_control",
        "exp404_scale5_control",
        "exp209_hmm_control",
        "fold_assignment",
    ):
        report = inspect_gzip_csv(paths[name])
        expected = str(specs[name]["expected_decompressed_sha256"])
        if report["decompressed_sha256"] != expected:
            raise ValueError(f"{name} decompressed SHA mismatch")
        reports[name] = report
    exp072_expected_raw = str(specs["exp072_control"]["expected_raw_gzip_sha256"])
    if reports["exp072_control"]["raw_sha256"] != exp072_expected_raw:
        raise ValueError("exp072 control raw gzip SHA mismatch")
    if reports["exp404_scale5_control"]["raw_sha256"] != str(
        specs["exp404_scale5_control"]["expected_raw_sha256"]
    ):
        raise ValueError("exp404 scale-5 control raw SHA mismatch")
    hidden_sha = sha256_path(paths["hidden_like_assignment"])
    if hidden_sha != str(specs["hidden_like_assignment"]["expected_sha256"]):
        raise ValueError("hidden-like assignment raw SHA mismatch")
    reports["hidden_like_assignment"] = {
        "path": str(paths["hidden_like_assignment"]),
        "bytes": paths["hidden_like_assignment"].stat().st_size,
        "raw_sha256": hidden_sha,
        "columns": pd.read_csv(paths["hidden_like_assignment"], nrows=0)
        .columns.astype(str)
        .tolist(),
    }
    required_columns = {
        "exp072_control": {"id", "well", "last_known_tvt", "likpf_mean_d"},
        "exp404_scale5_control": {
            "id",
            str(specs["exp404_scale5_control"]["prediction_column"]),
            str(specs["exp404_scale5_control"]["arithmetic_prediction_column"]),
        },
        "exp209_hmm_control": {
            "id",
            str(specs["exp209_hmm_control"]["prediction_column"]),
        },
        "fold_assignment": set(specs["fold_assignment"]["safe_columns"]),
        "hidden_like_assignment": {
            "well_id",
            *[str(value) for value in specs["hidden_like_assignment"]["role_columns"].values()],
        },
    }
    for name, required in required_columns.items():
        missing = sorted(required - set(reports[name]["columns"]))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    for name in (
        "exp072_control",
        "exp404_scale5_control",
        "exp209_hmm_control",
        "fold_assignment",
    ):
        if int(reports[name]["data_rows"]) != expected_rows:
            raise ValueError(f"{name} row count mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "reports": reports,
        "truth_or_reporting_values_parsed_before_freeze": {
            "unknown_suffix_tvt_rows": 0,
            "error_rows": 0,
            "fold_rows": 0,
            "hidden_like_role_rows": 0,
        },
    }


# %% [markdown]
# ## 4. Truth-free PF inputs and self-GR surface


# %%
@dataclass
class TruthAccessLedger:
    prediction_frozen: bool = False
    unknown_suffix_tvt_rows_before_freeze: int = 0
    error_rows_before_freeze: int = 0
    fold_rows_before_freeze: int = 0
    hidden_like_role_rows_before_freeze: int = 0
    unknown_suffix_tvt_rows_after_freeze: int = 0
    fold_rows_after_freeze: int = 0
    hidden_like_role_rows_after_freeze: int = 0

    def require_frozen(self) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("late reporting input requires a frozen prediction")

    def mark_frozen(self) -> None:
        if any(
            value
            for value in (
                self.unknown_suffix_tvt_rows_before_freeze,
                self.error_rows_before_freeze,
                self.fold_rows_before_freeze,
                self.hidden_like_role_rows_before_freeze,
            )
        ):
            raise RuntimeError("truth/reporting values were accessed before prediction freeze")
        self.prediction_frozen = True

    def report(self) -> dict[str, Any]:
        return {
            "prediction_frozen": self.prediction_frozen,
            "before_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_before_freeze,
                "error_rows": self.error_rows_before_freeze,
                "fold_rows": self.fold_rows_before_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_before_freeze,
            },
            "after_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_after_freeze,
                "fold_rows": self.fold_rows_after_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_after_freeze,
            },
        }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    if list(frame.columns) != ["MD", "Z", "GR", "TVT_input"]:
        frame = frame[["MD", "Z", "GR", "TVT_input"]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "Z"]].isna().any().any():
        raise ValueError(f"{well}: MD/Z must be finite")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
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
    base_scale = float(np.clip(raw_scale, clip[0], clip[1]))
    return {
        "raw_scale": raw_scale,
        "base_scale": base_scale,
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "base_clip_min": float(clip[0]),
        "base_clip_max": float(clip[1]),
    }


def exp072_initial_rate(horizontal: pd.DataFrame, *, tail_rows: int = 30) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    tail = known.tail(tail_rows)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    valid = delta_md > 0
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid]))


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    multiplier: float,
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
    last_position = last_known_tvt + float(last_known["Z"])
    scale_audit = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    candidate_scale = float(scale_audit["base_scale"]) * float(multiplier)
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
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
    if not np.isfinite(eval_gr).all():
        raise ValueError("evaluation GR interpolation is not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": eval_z,
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - last_known_md,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_position,
        "initial_rate": exp072_initial_rate(horizontal),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "scale_audit": {
            **scale_audit,
            "candidate_scale": candidate_scale,
            "multiplier": float(multiplier),
            "post_multiplier_clip_applied": False,
            "post_multiplier_clip_count": 0,
        },
    }


def _safe_interp_gr(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    finite = series.dropna()
    fill_value = float(np.nanmedian(finite.to_numpy(np.float64))) if len(finite) else 0.0
    return series.interpolate(limit_direction="both").fillna(fill_value).to_numpy(np.float64)


def build_gr_window_descriptors(
    horizontal: pd.DataFrame,
    *,
    radius: int,
    offsets: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact exp223 descriptor formula, using only observable GR."""
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(gr_raw).astype(np.float64)
    gr = _safe_interp_gr(gr_raw)
    series = pd.Series(gr)
    window = 2 * int(radius) + 1
    minimum_periods = max(3, int(radius) // 2)
    roll_mean = series.rolling(
        window=window, center=True, min_periods=minimum_periods
    ).mean()
    roll_std = series.rolling(
        window=window, center=True, min_periods=minimum_periods
    ).std(ddof=0)
    mean = (
        roll_mean.interpolate(limit_direction="both")
        .fillna(float(np.mean(gr)))
        .to_numpy(np.float64)
    )
    fallback_std = float(np.std(gr)) if float(np.std(gr)) > 1e-6 else 1.0
    std = (
        roll_std.interpolate(limit_direction="both")
        .fillna(fallback_std)
        .to_numpy(np.float64)
    )
    std = np.clip(std, 1.0, None)
    missing_rate = 1.0 - (
        pd.Series(finite)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )
    descriptors: list[np.ndarray] = []
    for offset in offsets:
        shifted = (
            pd.Series(gr)
            .shift(-int(offset))
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
        )
        descriptors.append(
            ((shifted.to_numpy(np.float64) - mean) / std).astype(np.float64)
        )
    global_std = float(np.std(gr)) if float(np.std(gr)) > 1e-6 else 1.0
    descriptors.append(((mean - float(np.mean(gr))) / global_std).astype(np.float64))
    descriptors.append(np.log1p(std).astype(np.float64))
    if radius > 0:
        left = (
            pd.Series(gr)
            .shift(radius)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(np.float64)
        )
        right = (
            pd.Series(gr)
            .shift(-radius)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(np.float64)
        )
        descriptors.append(((right - left) / (2.0 * radius * std)).astype(np.float64))
    matrix = np.vstack(descriptors).T
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix.astype(np.float32), np.clip(missing_rate, 0.0, 1.0).astype(np.float32)


def select_prefix_anchor_indices(
    known_indices: np.ndarray,
    *,
    radius: int,
    stride: int,
    max_anchors: int,
    keep_last: int,
) -> np.ndarray:
    if len(known_indices) == 0:
        return np.array([], dtype=np.int64)
    last_known = int(np.max(known_indices))
    usable = known_indices[known_indices <= last_known - int(radius)]
    if len(usable) == 0:
        usable = known_indices
    selected = usable[:: max(1, int(stride))]
    if keep_last > 0:
        selected = np.unique(
            np.concatenate([selected, usable[-int(keep_last) :]])
        ).astype(np.int64)
    if max_anchors > 0 and len(selected) > max_anchors:
        take = np.linspace(0, len(selected) - 1, int(max_anchors)).round().astype(np.int64)
        selected = selected[take]
    return selected.astype(np.int64)


def build_self_gr_likelihood_surface(
    horizontal: pd.DataFrame,
    eval_index: np.ndarray,
    grid: np.ndarray,
    typewell_peak_tvt: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the fixed exp223 centered same-well GR likelihood surface."""
    radius = int(config["window_radius_rows"])
    offsets = [int(value) for value in config["descriptor_offsets"]]
    top_k = int(config["top_k"])
    stride = int(config["prefix_anchor_stride"])
    max_anchors = int(config["max_prefix_anchors"])
    keep_last = int(config["keep_last_prefix_anchors"])
    min_anchors = int(config["min_prefix_anchors"])
    max_missing_rate = float(config["max_window_missing_rate"])
    sigma_tvt = float(config["gaussian_sigma_tvt_ft"])
    distance_temperature = float(config["descriptor_distance_temperature"])
    agreement_sigma = float(config["typewell_agreement_sigma_tvt_ft"])
    surface_clip = float(config["surface_quadratic_clip"])
    chunk_size = int(config["surface_chunk_size"])
    n_eval = len(eval_index)
    n_grid = len(grid)
    zero_surface = np.zeros((n_eval, n_grid), dtype=np.float32)
    zero_vector = np.zeros(n_eval, dtype=np.float32)
    empty = {
        "centered_logl": zero_surface,
        "quality": zero_vector,
        "peak_tvt": np.full(n_eval, np.nan, dtype=np.float64),
        "peak_gap": zero_vector,
        "typewell_agreement": zero_vector,
        "valid": zero_vector,
        "prefix_anchor_count": 0,
    }
    if n_eval == 0 or n_grid == 0:
        return empty
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_indices = np.flatnonzero(np.isfinite(tvt_input))
    anchor_indices = select_prefix_anchor_indices(
        known_indices,
        radius=radius,
        stride=stride,
        max_anchors=max_anchors,
        keep_last=keep_last,
    )
    if len(anchor_indices) < min_anchors:
        empty["prefix_anchor_count"] = int(len(anchor_indices))
        return empty
    descriptors, missing_rate = build_gr_window_descriptors(
        horizontal, radius=radius, offsets=offsets
    )
    anchor_indices = anchor_indices[
        missing_rate[anchor_indices] <= max_missing_rate
    ]
    if len(anchor_indices) < min_anchors:
        empty["prefix_anchor_count"] = int(len(anchor_indices))
        return empty
    anchor_desc = descriptors[anchor_indices].astype(np.float32)
    anchor_tvt = tvt_input[anchor_indices].astype(np.float64)
    eval_desc = descriptors[eval_index].astype(np.float32)
    eval_missing = missing_rate[eval_index].astype(np.float32)
    prefix_coverage_quality = float(
        np.clip(len(anchor_indices) / max(float(min_anchors), 1.0), 0.0, 1.0)
    )
    centered = np.zeros((n_eval, n_grid), dtype=np.float32)
    quality = np.zeros(n_eval, dtype=np.float32)
    peak_tvt = np.full(n_eval, np.nan, dtype=np.float64)
    peak_gap = np.zeros(n_eval, dtype=np.float32)
    agreement = np.zeros(n_eval, dtype=np.float32)
    valid = np.zeros(n_eval, dtype=np.float32)
    k_eff = min(top_k, len(anchor_indices))
    eps = 1e-6
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        desc = eval_desc[start:end]
        diff = desc[:, None, :] - anchor_desc[None, :, :]
        cost = np.mean(diff * diff, axis=2)
        if k_eff < cost.shape[1]:
            top_idx_unsorted = np.argpartition(
                cost, kth=k_eff - 1, axis=1
            )[:, :k_eff]
        else:
            top_idx_unsorted = np.tile(
                np.arange(cost.shape[1]), (cost.shape[0], 1)
            )
        top_cost_unsorted = np.take_along_axis(cost, top_idx_unsorted, axis=1)
        order = np.argsort(top_cost_unsorted, axis=1)
        top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
        top_cost = np.take_along_axis(top_cost_unsorted, order, axis=1)
        centers = anchor_tvt[top_idx]
        rel_cost = top_cost - top_cost[:, :1]
        weights = np.exp(-rel_cost / (2.0 * distance_temperature**2))
        weights /= np.clip(weights.sum(axis=1, keepdims=True), eps, None)
        zscore = (grid[None, None, :] - centers[:, :, None]) / sigma_tvt
        component_ll = (
            np.log(np.clip(weights, eps, None))[:, :, None]
            - 0.5 * np.minimum(zscore * zscore, surface_clip)
        )
        best = np.max(component_ll, axis=1)
        log_likelihood = best + np.log(
            np.clip(
                np.exp(component_ll - best[:, None, :]).sum(axis=1),
                eps,
                None,
            )
        )
        centered_chunk = log_likelihood - np.mean(
            log_likelihood, axis=1, keepdims=True
        )
        centered_chunk /= np.clip(
            np.std(centered_chunk, axis=1, keepdims=True), 0.25, None
        )
        centered[start:end] = centered_chunk.astype(np.float32)
        cost_q75 = np.quantile(cost, 0.75, axis=1)
        sharpness = np.clip(
            (cost_q75 - top_cost[:, 0]) / np.clip(cost_q75, eps, None),
            0.0,
            1.0,
        )
        gap = (
            top_cost[:, 1] - top_cost[:, 0]
            if k_eff >= 2
            else np.zeros(end - start, dtype=np.float32)
        )
        gap_quality = np.clip(gap / max(distance_temperature, eps), 0.0, 1.0)
        peak = centers[:, 0]
        agree = np.exp(
            -0.5 * ((peak - typewell_peak_tvt[start:end]) / agreement_sigma) ** 2
        )
        miss_quality = np.clip(1.0 - eval_missing[start:end], 0.0, 1.0)
        row_quality = (
            prefix_coverage_quality
            * miss_quality
            * (0.25 + 0.75 * sharpness)
            * (0.25 + 0.75 * gap_quality)
            * (0.15 + 0.85 * agree)
        )
        row_valid = (
            np.isfinite(peak)
            & np.isfinite(row_quality)
            & (eval_missing[start:end] <= max_missing_rate)
        )
        quality[start:end] = np.where(
            row_valid, np.clip(row_quality, 0.0, 1.0), 0.0
        ).astype(np.float32)
        peak_tvt[start:end] = peak
        peak_gap[start:end] = gap.astype(np.float32)
        agreement[start:end] = agree.astype(np.float32)
        valid[start:end] = row_valid.astype(np.float32)
    return {
        "centered_logl": centered,
        "quality": quality,
        "peak_tvt": peak_tvt,
        "peak_gap": peak_gap,
        "typewell_agreement": agreement,
        "valid": valid,
        "prefix_anchor_count": int(len(anchor_indices)),
    }


def self_gr_surface_content_sha(
    grid: np.ndarray,
    boost: np.ndarray,
    quality: np.ndarray,
    valid: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("grid", grid),
        ("boost", boost),
        ("quality", quality),
        ("valid", valid),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_self_gr_surface(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the padded, pre-clipped boost surface used by PF particles."""
    surface_config = get_nested(dict(config), "model.self_gr_surface") or {}
    step = float(surface_config["grid_step_ft"])
    pad = float(surface_config["grid_pad_ft"])
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    grid = np.arange(
        float(typewell_tvt.min()) - pad,
        float(typewell_tvt.max()) + pad + step,
        step,
        dtype=np.float64,
    )
    expected_gr = np.interp(grid, typewell_tvt, typewell_gr)
    eval_gr = np.asarray(prepared["eval_gr"], dtype=np.float64)
    peak_index = np.argmin(np.abs(eval_gr[:, None] - expected_gr[None, :]), axis=1)
    typewell_peak_tvt = grid[peak_index]
    surface = build_self_gr_likelihood_surface(
        horizontal,
        np.asarray(prepared["eval_indices"], dtype=np.int64),
        grid,
        typewell_peak_tvt,
        surface_config,
    )
    clip_value = float(surface_config["clip"])
    boost = np.clip(
        np.asarray(surface["centered_logl"], dtype=np.float32),
        0.0,
        clip_value,
    ).astype(np.float32)
    quality = np.asarray(surface["quality"], dtype=np.float32)
    valid = np.asarray(surface["valid"], dtype=np.float32)
    if boost.shape != (len(eval_gr), len(grid)):
        raise ValueError("self-GR surface shape mismatch")
    if not np.isfinite(boost).all() or not np.isfinite(quality).all():
        raise ValueError("self-GR surface contains non-finite values")
    return {
        **surface,
        "boost": boost,
        "quality": quality,
        "valid": valid,
        "grid": grid,
        "grid_minimum": float(grid[0]),
        "grid_step": step,
        "alpha": float(surface_config["alpha"]),
        "content_sha256": self_gr_surface_content_sha(grid, boost, quality, valid),
    }


# %% [markdown]
# ## 5. Exact exp072 likelihood-PF kernel with self-GR boost


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
def _pf_selfgr_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    self_boost: np.ndarray,
    self_quality: np.ndarray,
    self_grid_minimum: float,
    self_grid_step: float,
    self_alpha: float,
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Exact exp072 kernel plus one fixed self-GR observation-loglik boost."""
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    positive_boost_counts = np.zeros(seeds, np.int64)
    weighted_boost_sums = np.zeros(seeds)
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
                boost = _interp1(
                    self_boost[row],
                    position[particle] - z_v[row],
                    self_grid_minimum,
                    self_grid_step,
                )
                weighted_boost = self_alpha * self_quality[row] * boost
                if weighted_boost > 0.0:
                    positive_boost_counts[seed_index] += 1
                    weighted_boost_sums[seed_index] += weighted_boost
                likelihood = np.exp(-0.5 * squared + weighted_boost)
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
        positive_boost_counts,
        weighted_boost_sums,
    )


def aggregate_seed_predictions(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    scales: Iterable[float],
) -> dict[str, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    outputs: dict[str, np.ndarray] = {}
    for scale in scales:
        weights = np.exp(centered / float(scale))
        weights /= weights.sum()
        outputs[f"pf_scale_{float(scale):g}"] = (weights[:, None] * predictions).sum(axis=0)
    outputs["pf_mean"] = predictions.mean(axis=0)
    return outputs


def run_likelihood_pf(
    prepared: dict[str, Any],
    surface: Mapping[str, Any],
    *,
    particles: int,
    seeds: int,
    scales: Iterable[float],
    seed_base: int,
    momentum: float = 0.998,
    rate_noise: float = 0.002,
    position_noise: float = 0.005,
    rough_position: float = 0.1,
    rough_rate: float = 0.001,
    resample_fraction: float = 0.5,
    initial_spread: float = 4.5,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    started = time.time()
    (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
        positive_boost_counts,
        weighted_boost_sums,
    ) = _pf_selfgr_allseeds(
        prepared["eval_md"],
        prepared["eval_z"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        np.asarray(surface["boost"], dtype=np.float32),
        np.asarray(surface["quality"], dtype=np.float32),
        float(surface["grid_minimum"]),
        float(surface["grid_step"]),
        float(surface["alpha"]),
        float(prepared["scale_audit"]["candidate_scale"]),
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
    outputs = aggregate_seed_predictions(predictions, log_likelihoods, scales)
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "seed_loglik_mean_per_row": float(log_likelihoods.mean()) / len(prepared["eval_md"]),
        "seed_loglik_best_per_row": float(log_likelihoods.max()) / len(prepared["eval_md"]),
        "seed_loglik_spread": float(log_likelihoods.std()),
        "resampling_count_total": int(resampling_counts.sum()),
        "resampling_count_min": int(resampling_counts.min()),
        "resampling_count_max": int(resampling_counts.max()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "position_clip_count_total": int(position_clip_counts.sum()),
        "positive_boost_application_count": int(positive_boost_counts.sum()),
        "weighted_boost_sum": float(weighted_boost_sums.sum()),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
    }
    return outputs, diagnostics, predictions, log_likelihoods


# %% [markdown]
# ## 6. Technical preflight, shard generation, and prediction freeze


# %%
def warm_up_pf_kernel() -> None:
    md = np.linspace(1.0, 8.0, 8)
    z = np.zeros(8)
    observed_gr = np.full(8, 50.0)
    grid_gr = np.linspace(45.0, 55.0, 100)
    _pf_selfgr_allseeds(
        md,
        z,
        observed_gr,
        grid_gr,
        0.0,
        0.2,
        np.zeros((8, 100), dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        0.0,
        0.2,
        0.0,
        20.0,
        50.0,
        0.0,
        8,
        2,
        1,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )


def decode_well(
    well: str,
    raw_dir: Path,
    config: dict[str, Any],
    *,
    alpha_override: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        multiplier=float(get_nested(config, "model.typewell_gr_emission.multiplier")),
        grid_step=float(get_nested(config, "model.pf.typewell_grid_step_ft")),
    )
    surface = prepare_self_gr_surface(horizontal, typewell, prepared, config)
    if alpha_override is not None:
        surface["alpha"] = float(alpha_override)
    pf_config = get_nested(config, "model.pf") or {}
    seed_base = stable_seed("likpf", "train", well)
    outputs, diagnostics, _, _ = run_likelihood_pf(
        prepared,
        surface,
        particles=int(pf_config["particles"]),
        seeds=int(pf_config["seeds"]),
        scales=[float(pf_config["primary_seed_weighting_temperature"])],
        seed_base=seed_base,
        momentum=float(pf_config["momentum"]),
        rate_noise=float(pf_config["rate_noise"]),
        position_noise=float(pf_config["position_noise"]),
        rough_position=float(pf_config["rough_position"]),
        rough_rate=float(pf_config["rough_rate"]),
        resample_fraction=float(pf_config["resample_threshold_fraction"]),
        initial_spread=float(pf_config["initial_position_spread_ft"]),
    )
    eval_indices = prepared["eval_indices"]
    raw_observed = prepared["raw_gr_observed"]
    missing_fraction = float(1.0 - raw_observed.mean())
    candidate = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices.astype(np.int64),
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": prepared["md_since"].astype(np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64(missing_fraction),
            "self_gr_quality": np.asarray(surface["quality"], dtype=np.float32),
            "self_gr_valid": np.asarray(surface["valid"], dtype=np.float32),
            "self_gr_peak_tvt": np.asarray(surface["peak_tvt"], dtype=np.float64),
            "self_gr_peak_gap": np.asarray(surface["peak_gap"], dtype=np.float32),
            "self_gr_typewell_agreement": np.asarray(
                surface["typewell_agreement"], dtype=np.float32
            ),
            PRIMARY_CANDIDATE: outputs["pf_scale_5"].astype(np.float32),
            SECONDARY_CANDIDATE: outputs["pf_mean"].astype(np.float32),
        }
    )
    scale_audit = prepared["scale_audit"]
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(scale_audit["known_rows"]),
        "prefix_gr_missing_rows": int(scale_audit["known_gr_missing_rows"]),
        "eval_rows": len(candidate),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "eval_raw_gr_missing_fraction": missing_fraction,
        "last_known_tvt": float(prepared["last_known_tvt"]),
        "last_known_position": float(prepared["last_known_position"]),
        "initial_rate": float(prepared["initial_rate"]),
        "gs_raw": float(scale_audit["raw_scale"]),
        "gs_base": float(scale_audit["base_scale"]),
        "gs_candidate": float(scale_audit["candidate_scale"]),
        "multiplier": float(scale_audit["multiplier"]),
        "post_multiplier_clip_applied": bool(scale_audit["post_multiplier_clip_applied"]),
        "post_multiplier_clip_count": int(scale_audit["post_multiplier_clip_count"]),
        "seed_base": int(seed_base),
        "seed_first": int(seed_base),
        "seed_last": int(seed_base + int(pf_config["seeds"]) - 1),
        "seeds": int(pf_config["seeds"]),
        "particles": int(pf_config["particles"]),
        "seed_well_trajectories": int(pf_config["seeds"]),
        "particle_starts": int(pf_config["seeds"]) * int(pf_config["particles"]),
        "self_gr_mode": str(get_nested(config, "model.self_gr_surface.mode")),
        "self_gr_alpha": float(surface["alpha"]),
        "self_gr_clip": float(get_nested(config, "model.self_gr_surface.clip")),
        "self_gr_prefix_anchor_count": int(surface["prefix_anchor_count"]),
        "self_gr_valid_rows": int(np.asarray(surface["valid"]).sum()),
        "self_gr_quality_positive_rows": int(
            (np.asarray(surface["quality"]) > 0.0).sum()
        ),
        "self_gr_quality_mean": float(np.asarray(surface["quality"]).mean()),
        "self_gr_boost_positive_cells": int(
            (np.asarray(surface["boost"]) > 0.0).sum()
        ),
        "self_gr_surface_rows": int(np.asarray(surface["boost"]).shape[0]),
        "self_gr_surface_states": int(np.asarray(surface["boost"]).shape[1]),
        "self_gr_surface_logical_sha256": str(surface["content_sha256"]),
        **diagnostics,
        "wall_seconds": time.time() - started,
    }
    finite = np.isfinite(candidate[list(PREDICTION_COLUMNS)].to_numpy(np.float64))
    if not finite.all():
        raise ValueError(f"{well}: candidate prediction contains non-finite values")
    return candidate, audit


def generate_and_freeze_predictions(
    raw_dir: Path,
    artifacts: Path,
    config: dict[str, Any],
    wells: list[str],
    ledger: TruthAccessLedger,
    *,
    artifact_tag: str,
    expected_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    if ledger.prediction_frozen:
        raise RuntimeError("prediction ledger is already frozen")
    expected_wells = len(wells)
    if wells != sorted(set(wells)) or not wells:
        raise ValueError("exp429 requires sorted unique well IDs")
    warm_up_pf_kernel()
    workers = int(get_nested(config, "runtime.num_workers"))
    results = Parallel(n_jobs=workers, prefer="threads")(
        delayed(decode_well)(well, raw_dir, config) for well in wells
    )
    candidate = pd.concat([result[0] for result in results], ignore_index=True)
    audit = pd.DataFrame([result[1] for result in results])
    candidate = candidate.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    audit = audit.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if candidate.duplicated(["well_id", "row_idx"]).any() or candidate["id"].duplicated().any():
        raise ValueError("candidate row identity is duplicated")
    if (
        (expected_rows is not None and len(candidate) != expected_rows)
        or candidate["well_id"].nunique() != expected_wells
        or len(audit) != expected_wells
        or audit["well_id"].nunique() != expected_wells
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("candidate generation coverage mismatch")
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_{artifact_tag}_candidate_predictions.csv.gz"
    audit_path = artifacts / f"{OUTPUT_PREFIX}_{artifact_tag}_well_surface_pf_audit.csv"
    surface_manifest_path = artifacts / f"{OUTPUT_PREFIX}_{artifact_tag}_surface_manifest.csv"
    write_deterministic_gzip_csv(candidate, prediction_path)
    audit.to_csv(audit_path, index=False)
    surface_manifest = audit[
        [
            "well_id",
            "self_gr_surface_rows",
            "self_gr_surface_states",
            "self_gr_prefix_anchor_count",
            "self_gr_valid_rows",
            "self_gr_surface_logical_sha256",
        ]
    ].copy()
    surface_manifest.to_csv(surface_manifest_path, index=False)
    logical_columns = [
        "id",
        "well_id",
        "row_idx",
        *PREDICTION_COLUMNS,
    ]
    surface_content_sha = dataframe_content_sha(
        surface_manifest,
        [
            "well_id",
            "self_gr_surface_rows",
            "self_gr_surface_states",
            "self_gr_prefix_anchor_count",
            "self_gr_valid_rows",
            "self_gr_surface_logical_sha256",
        ],
    )
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "logical_columns": logical_columns,
        "logical_content_sha256": dataframe_content_sha(candidate, logical_columns),
        "schema_sha256": dataframe_schema_sha(candidate),
        "raw_gzip_sha256": sha256_path(prediction_path),
        "well_surface_pf_audit_sha256": sha256_path(audit_path),
        "surface_manifest_sha256": sha256_path(surface_manifest_path),
        "surface_logical_content_sha256": surface_content_sha,
        "truth_or_reporting_values_parsed_before_freeze": {
            "unknown_suffix_tvt_rows": ledger.unknown_suffix_tvt_rows_before_freeze,
            "error_rows": ledger.error_rows_before_freeze,
            "fold_rows": ledger.fold_rows_before_freeze,
            "hidden_like_role_rows": ledger.hidden_like_role_rows_before_freeze,
        },
    }
    ledger.mark_frozen()
    return (
        candidate,
        audit,
        frozen,
        {
            "prediction": prediction_path,
            "well_surface_pf_audit": audit_path,
            "surface_manifest": surface_manifest_path,
        },
    )


def _require_frozen_prediction(frozen: dict[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("late attachment requires a frozen prediction")
    value = str(frozen.get("logical_content_sha256") or "")
    if len(value) != 64:
        raise RuntimeError("frozen prediction logical content SHA is missing")


def build_raw_well_manifest(config: Mapping[str, Any], raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = path.name.replace("__horizontal_well.csv", "")
        tvt_input = pd.to_numeric(
            pd.read_csv(path, usecols=["TVT_input"])["TVT_input"],
            errors="coerce",
        )
        rows.append(
            {
                "well_id": well,
                "rows": int(len(tvt_input)),
                "prefix_rows": int(tvt_input.notna().sum()),
                "suffix_rows": int(tvt_input.isna().sum()),
            }
        )
    manifest = (
        pd.DataFrame(rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(manifest) != int(get_nested(dict(config), "validation.expected_wells"))
        or int(manifest["suffix_rows"].sum())
        != int(get_nested(dict(config), "validation.expected_rows"))
    ):
        raise ValueError("raw well manifest coverage mismatch")
    return manifest


def assign_lpt_shards(
    manifest: pd.DataFrame,
    shard_count: int = SHARD_COUNT,
) -> pd.DataFrame:
    """Deterministic longest-processing-time assignment by suffix row count."""
    ordered = manifest.sort_values(
        ["suffix_rows", "well_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    loads = [0] * int(shard_count)
    assignments: dict[str, int] = {}
    for row in ordered.itertuples(index=False):
        shard = min(range(int(shard_count)), key=lambda value: (loads[value], value))
        assignments[str(row.well_id)] = shard
        loads[shard] += int(row.suffix_rows)
    result = manifest.copy()
    result["shard_index"] = (
        result["well_id"].map(assignments).astype(np.int8)
    )
    return result


def load_preflight_well_asset(config: Mapping[str, Any]) -> pd.DataFrame:
    spec = _input_spec(dict(config), "preflight_wells")
    path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("fixed preflight-well asset SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "sha256_order",
        "prefix_anchor_count",
        "eligible_rows",
    }
    if set(frame.columns) != required or len(frame) != 4:
        raise ValueError("fixed preflight-well asset schema/count mismatch")
    if frame["well_id"].duplicated().any() or not frame["eligible_rows"].gt(0).all():
        raise ValueError("fixed preflight-well asset eligibility mismatch")
    expected = sorted(
        frame["well_id"].astype(str),
        key=lambda well: hashlib.sha256(well.encode()).hexdigest(),
    )
    if frame.sort_values("sha256_order")["well_id"].astype(str).tolist() != expected:
        raise ValueError("fixed preflight-well asset SHA ordering mismatch")
    return frame.sort_values("sha256_order", kind="mergesort").reset_index(drop=True)


def select_preflight_wells(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> pd.DataFrame:
    """Target-free fixed-four selection used once to build the pinned asset."""
    eligible: list[dict[str, Any]] = []
    paths = sorted(
        raw_dir.glob("*__horizontal_well.csv"),
        key=lambda path: hashlib.sha256(
            path.name.replace("__horizontal_well.csv", "").encode()
        ).hexdigest(),
    )
    for path in paths:
        well = path.name.replace("__horizontal_well.csv", "")
        horizontal = load_horizontal_without_truth(well, raw_dir)
        surface_config = get_nested(dict(config), "model.self_gr_surface") or {}
        descriptors, missing_rate = build_gr_window_descriptors(
            horizontal,
            radius=int(surface_config["window_radius_rows"]),
            offsets=[int(value) for value in surface_config["descriptor_offsets"]],
        )
        del descriptors
        tvt_input = pd.to_numeric(
            horizontal["TVT_input"], errors="coerce"
        ).to_numpy(np.float64)
        known_indices = np.flatnonzero(np.isfinite(tvt_input))
        anchor_indices = select_prefix_anchor_indices(
            known_indices,
            radius=int(surface_config["window_radius_rows"]),
            stride=int(surface_config["prefix_anchor_stride"]),
            max_anchors=int(surface_config["max_prefix_anchors"]),
            keep_last=int(surface_config["keep_last_prefix_anchors"]),
        )
        anchor_indices = anchor_indices[
            missing_rate[anchor_indices]
            <= float(surface_config["max_window_missing_rate"])
        ]
        eval_indices = np.flatnonzero(~np.isfinite(tvt_input))
        eligible_rows = (
            int(
                (
                    missing_rate[eval_indices]
                    <= float(surface_config["max_window_missing_rate"])
                ).sum()
            )
            if len(anchor_indices) >= int(surface_config["min_prefix_anchors"])
            else 0
        )
        if eligible_rows > 0:
            eligible.append(
                {
                    "well_id": well,
                    "sha": hashlib.sha256(well.encode()).hexdigest(),
                    "prefix_anchor_count": int(len(anchor_indices)),
                    "eligible_rows": eligible_rows,
                }
            )
            if len(eligible) == 4:
                break
    selected = sorted(eligible, key=lambda row: (row["sha"], row["well_id"]))
    if len(selected) != 4:
        raise ValueError("fewer than four target-free self-GR eligible wells")
    return pd.DataFrame(
        [
            {
                "well_id": row["well_id"],
                "sha256_order": index,
                "prefix_anchor_count": row["prefix_anchor_count"],
                "eligible_rows": row["eligible_rows"],
            }
            for index, row in enumerate(selected)
        ]
    )


def run_preflight_stage(config: dict[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    raw_dir = train_data_dir(config)
    asset = load_preflight_well_asset(config)
    preflight = preflight_saved_inputs(config)
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for well in asset["well_id"].astype(str):
        for label, alpha in (("alpha0_parity", 0.0), ("alpha07_candidate", 0.07)):
            prediction, audit = decode_well(
                well,
                raw_dir,
                config,
                alpha_override=alpha,
            )
            prediction["preflight_variant"] = label
            audit["preflight_variant"] = label
            rows.append(prediction)
            audits.append(audit)
    frame = pd.concat(rows, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    alpha0 = frame.loc[
        frame["preflight_variant"].eq("alpha0_parity"),
        ["id", SECONDARY_CANDIDATE],
    ]
    exp404_spec = _input_spec(config, "exp404_scale5_control")
    exp404_arithmetic_column = str(exp404_spec["arithmetic_prediction_column"])
    exp404 = pd.read_csv(
        preflight["paths"]["exp404_scale5_control"],
        usecols=["id", exp404_arithmetic_column],
        dtype={"id": str},
        compression="gzip",
    )
    exp404[exp404_arithmetic_column] = restore_frozen_float32_column(
        exp404[exp404_arithmetic_column],
        label="saved exp404 x1.0 arithmetic",
    )
    exp404 = exp404.rename(
        columns={exp404_arithmetic_column: PREFLIGHT_ALPHA0_CONTROL}
    )
    aligned = _align_on_id(
        alpha0,
        exp404[["id", PREFLIGHT_ALPHA0_CONTROL]],
        [PREFLIGHT_ALPHA0_CONTROL],
        label="saved exp404 x1.0 arithmetic alpha0 parity",
    )
    alpha0_values = aligned[SECONDARY_CANDIDATE].to_numpy(np.float32)
    comparator_values = aligned[PREFLIGHT_ALPHA0_CONTROL].to_numpy(np.float32)
    parity = np.abs(
        alpha0_values.astype(np.float64) - comparator_values.astype(np.float64)
    )
    bit_equal = alpha0_values.view(np.uint32) == comparator_values.view(np.uint32)
    maximum = float(parity.max())
    technical = {
        "variants": int(frame["preflight_variant"].nunique()),
        "wells": int(frame["well_id"].nunique()),
        "pf_well_runs": len(audit_frame),
        "seed_well_trajectories": int(audit_frame["seed_well_trajectories"].sum()),
        "particle_starts": int(audit_frame["particle_starts"].sum()),
        "alpha0_comparator": str(
            get_nested(
                config,
                "guards.technical.require_preflight_alpha0_comparator",
            )
        ),
        "alpha0_comparator_rows": len(aligned),
        "alpha0_comparator_dtype": "float32",
        "alpha0_bit_equal_rows": int(bit_equal.sum()),
        "alpha0_arithmetic_max_abs_parity_ft": maximum,
        "candidate_positive_quality_rows": int(
            audit_frame.loc[
                audit_frame["preflight_variant"].eq("alpha07_candidate"),
                "self_gr_quality_positive_rows",
            ].sum()
        ),
        "candidate_positive_boost_applications": int(
            audit_frame.loc[
                audit_frame["preflight_variant"].eq("alpha07_candidate"),
                "positive_boost_application_count",
            ].sum()
        ),
    }
    expected = get_nested(config, "model.execution_count") or {}
    technical["passed"] = bool(
        technical["variants"] == int(expected["technical_preflight_variants"])
        and technical["wells"] == int(expected["technical_preflight_wells"])
        and technical["pf_well_runs"] == int(expected["technical_preflight_pf_well_runs"])
        and technical["seed_well_trajectories"]
        == int(expected["technical_preflight_seed_well_trajectories"])
        and technical["particle_starts"]
        == int(expected["technical_preflight_particle_starts"])
        and maximum
        <= float(
            get_nested(
                config,
                "guards.technical.require_preflight_alpha0_arithmetic_max_abs_parity_ft",
            )
        )
        and bool(
            get_nested(
                config,
                "guards.technical.require_preflight_alpha0_bit_exact",
            )
        )
        and bool(bit_equal.all())
        and technical["candidate_positive_quality_rows"] > 0
        and technical["candidate_positive_boost_applications"] > 0
    )
    artifacts = artifact_dir()
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_preflight_predictions.csv.gz"
    audit_path = artifacts / f"{OUTPUT_PREFIX}_preflight_audit.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_preflight_summary.json"
    write_deterministic_gzip_csv(frame, prediction_path)
    audit_frame.to_csv(audit_path, index=False)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "preflight",
        "scientific_result": False,
        "technical_gate": technical,
        "prediction_sha256": dataframe_content_sha(
            frame,
            ["id", "preflight_variant", *PREDICTION_COLUMNS],
        ),
        "surface_manifest_sha256": dataframe_content_sha(
            audit_frame,
            [
                "well_id",
                "preflight_variant",
                "self_gr_surface_logical_sha256",
            ],
        ),
    }
    write_json(summary_path, summary)
    if not technical["passed"]:
        raise RuntimeError("exp429 technical preflight failed closed")
    return summary


# %% [markdown]
# ## 7. Strict shard merge and late reporting attachment


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


def _align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if source["id"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    lookup = source.assign(id=source["id"].astype(str)).set_index("id")
    aligned = lookup.reindex(frame["id"].astype(str))
    if aligned[columns].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def load_late_readout_frame(
    candidate: pd.DataFrame,
    frozen: dict[str, Any],
    preflight: dict[str, Any],
    raw_dir: Path,
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_frozen_prediction(frozen)
    ledger.require_frozen()
    expected_sha = dataframe_content_sha(
        candidate,
        list(frozen["logical_columns"]),
    )
    if expected_sha != str(frozen["logical_content_sha256"]):
        raise ValueError("in-memory candidate changed after prediction freeze")
    wells = sorted(candidate["well_id"].astype(str).unique().tolist())
    truth_parts = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(delayed(load_unknown_suffix_truth)(well, raw_dir) for well in wells)
    truth = (
        pd.concat(truth_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    ledger.unknown_suffix_tvt_rows_after_freeze += len(truth)
    frame = _align_on_id(candidate, truth, ["true_tvt"], label="raw suffix truth")

    exp072_usecols = ["id", "last_known_tvt", "likpf_mean_d"]
    exp072 = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=exp072_usecols,
        dtype={"id": str},
    )
    for column in exp072_usecols:
        if column != "id":
            exp072[column] = pd.to_numeric(exp072[column], errors="raise")
    exp072["saved_exp072_likpf_mean"] = exp072["last_known_tvt"] + exp072["likpf_mean_d"]
    frame = _align_on_id(
        frame,
        exp072[["id", SECONDARY_CONTROL]],
        [SECONDARY_CONTROL],
        label="saved exp072 control",
    )

    exp404_spec = _input_spec(config, "exp404_scale5_control")
    exp404_column = str(exp404_spec["prediction_column"])
    exp404 = pd.read_csv(
        preflight["paths"]["exp404_scale5_control"],
        usecols=["id", exp404_column],
        dtype={"id": str},
        compression="gzip",
    )
    exp404[exp404_column] = pd.to_numeric(exp404[exp404_column], errors="raise")
    exp404 = exp404.rename(columns={exp404_column: PRIMARY_CONTROL})
    frame = _align_on_id(
        frame,
        exp404[["id", PRIMARY_CONTROL]],
        [PRIMARY_CONTROL],
        label="saved exp404 scale-5 x1.0 control",
    )

    hmm_spec = _input_spec(config, "exp209_hmm_control")
    hmm_column = str(hmm_spec["prediction_column"])
    hmm = pd.read_csv(
        preflight["paths"]["exp209_hmm_control"],
        usecols=["id", hmm_column],
        dtype={"id": str},
    )
    hmm[hmm_column] = pd.to_numeric(hmm[hmm_column], errors="raise")
    hmm = hmm.rename(columns={hmm_column: "saved_exp209_hmm"})
    frame = _align_on_id(
        frame,
        hmm,
        ["saved_exp209_hmm"],
        label="saved exp209 HMM",
    )

    fold_spec = _input_spec(config, "exp226_reporting")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    forbidden_columns = {str(value) for value in fold_spec.get("forbidden_decoder_columns", [])}
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp429 fold allowlist must contain identity/fold columns only")
    if set(safe_columns) & forbidden_columns:
        raise ValueError("exp429 fold allowlist contains forbidden decoder columns")
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

    hidden_spec = _input_spec(config, "exp115_hidden_like")
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
    ledger.hidden_like_role_rows_after_freeze += len(hidden)
    expected_role_counts = hidden_spec.get("expected_role_counts") or {}
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
            str(key): int(value) for key, value in (expected_role_counts.get(scope) or {}).items()
        }
        if actual != expected:
            raise ValueError(f"hidden-like role counts mismatch for {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[role_columns["hidden_like_spatial"]].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[role_columns["hidden_like_typewell_purged"]].eq(
        "valid"
    )

    frame["candidate_hmm_50_50"] = 0.5 * (frame[PRIMARY_CANDIDATE] + frame["saved_exp209_hmm"])
    frame["parent_hmm_50_50"] = 0.5 * (frame[PRIMARY_CONTROL] + frame["saved_exp209_hmm"])
    if not np.isfinite(
        frame[
            [
                "true_tvt",
                SECONDARY_CONTROL,
                PRIMARY_CONTROL,
                "saved_exp209_hmm",
                "candidate_hmm_50_50",
                "parent_hmm_50_50",
                *PREDICTION_COLUMNS,
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("late readout contains non-finite values")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("reporting fold set mismatch")
    return frame, {
        "truth_attached_after_prediction_freeze": True,
        "candidate_content_sha256_reverified": expected_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "primary_control": PRIMARY_CONTROL,
        "secondary_control": SECONDARY_CONTROL,
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 8. Paired metrics and fail-closed gates


# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    candidate_column: str,
    control_column: str | None,
    comparison: str,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"metric scope {scope} is empty")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[candidate_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    candidate_mae = float(np.mean(np.abs(candidate - truth)))
    candidate_bias = float(np.mean(candidate - truth))
    candidate_within10 = float(np.mean(np.abs(candidate - truth) <= 10.0))
    record: dict[str, Any] = {
        "candidate": candidate_column,
        "comparison": comparison,
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate_rmse,
        "candidate_mae": candidate_mae,
        "candidate_bias": candidate_bias,
        "candidate_within_10ft": candidate_within10,
        "control": control_column,
        "control_available": control_column is not None,
    }
    if control_column is not None:
        control = selected[control_column].to_numpy(np.float64)
        control_rmse = rmse(truth, control)
        record.update(
            {
                "control_rmse": control_rmse,
                "control_mae": float(np.mean(np.abs(control - truth))),
                "control_bias": float(np.mean(control - truth)),
                "control_within_10ft": float(np.mean(np.abs(control - truth) <= 10.0)),
                "improvement_ft": control_rmse - candidate_rmse,
                "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
            }
        )
    else:
        record.update(
            {
                "control_rmse": None,
                "control_mae": None,
                "control_bias": None,
                "control_within_10ft": None,
                "improvement_ft": None,
                "delta_rmse_candidate_minus_control": None,
            }
        )
    return record


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
            (
                "missing_fraction_high",
                frame["well_missing_fraction"].ge(0.30).to_numpy(),
            ),
            ("md_since_1000_plus", frame["md_since"].ge(1000.0).to_numpy()),
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
    return scopes


def build_metric_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = metric_scopes(frame)
    primary_rows = [
        metric_record(
            frame,
            mask,
            candidate_column=PRIMARY_CANDIDATE,
            control_column=PRIMARY_CONTROL,
            comparison="fixed_scale5_selfgr_vs_saved_exp404_scale5_x1p0",
            scope=scope,
        )
        for scope, mask in scopes
    ]
    secondary_rows = [
        metric_record(
            frame,
            mask,
            candidate_column=SECONDARY_CANDIDATE,
            control_column=SECONDARY_CONTROL,
            comparison="fixed_arithmetic_selfgr_vs_saved_exp072_arithmetic",
            scope=scope,
        )
        for scope, mask in scopes
    ]
    blend_rows = [
        metric_record(
            frame,
            mask,
            candidate_column="candidate_hmm_50_50",
            control_column="parent_hmm_50_50",
            comparison="fixed_exp209_hmm_scale5_likpf_50_50",
            scope=scope,
        )
        for scope, mask in scopes
    ]
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
                "well_missing_fraction": float(group["well_missing_fraction"].iloc[0]),
            }
        )
    return (
        pd.DataFrame(primary_rows),
        pd.DataFrame(by_well_rows),
        pd.DataFrame(secondary_rows),
        pd.DataFrame(blend_rows),
    )


def _scope_row(frame: pd.DataFrame, scope: str) -> pd.Series:
    selected = frame.loc[frame["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one metric row for scope={scope}")
    return selected.iloc[0]


def evaluate_promotion_gate(
    frame: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    secondary_metrics: pd.DataFrame,
    blend_metrics: pd.DataFrame,
    audit: pd.DataFrame,
    preflight: dict[str, Any],
    raw_preflight: dict[str, Any],
    frozen: dict[str, Any],
    ledger: TruthAccessLedger,
    shard_summaries: Sequence[Mapping[str, Any]],
    preflight_summary: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    technical_config = get_nested(config, "guards.technical") or {}
    scientific_config = get_nested(config, "guards.scientific") or {}
    overall = _scope_row(primary_metrics, "overall")
    secondary_overall = _scope_row(secondary_metrics, "overall")
    blend_overall = _scope_row(blend_metrics, "overall")
    parity_tolerance = float(
        technical_config["require_saved_control_rmse_parity_atol_ft"]
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_counts = get_nested(config, "model.execution_count") or {}
    actual_counts = {
        "scientific_variants": 1,
        "full_candidate_pf_well_runs": len(audit),
        "full_seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "full_particle_starts": int(audit["particle_starts"].sum()),
        "full_shards": len(shard_summaries),
        "parent_full_control_reruns": 0,
        "reporting_folds": int(frame["fold"].nunique()),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    count_keys = list(actual_counts)
    execution_count_match = all(
        int(actual_counts[key]) == int(expected_counts[key]) for key in count_keys
    )
    before_freeze = ledger.report()["before_freeze"]
    finite_coverage = float(
        np.isfinite(frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).mean()
    )
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    baseline_parity = {
        PRIMARY_CONTROL: {
            "actual_rmse": float(overall["control_rmse"]),
            "expected_rmse": float(get_nested(config, "validation.primary_control_rmse_ft")),
        },
        SECONDARY_CONTROL: {
            "actual_rmse": float(secondary_overall["control_rmse"]),
            "expected_rmse": float(get_nested(config, "validation.secondary_control_rmse_ft")),
        },
        "fixed_exp209_hmm_scale5_likpf_50_50": {
            "actual_rmse": float(blend_overall["control_rmse"]),
            "expected_rmse": float(
                get_nested(config, "validation.fixed_hmm_likpf_blend_control_rmse_ft")
            ),
        },
    }
    for value in baseline_parity.values():
        value["absolute_difference"] = abs(
            value["actual_rmse"] - value["expected_rmse"]
        )
        value["passed"] = bool(value["absolute_difference"] <= parity_tolerance)
    alpha = audit["self_gr_alpha"].to_numpy(np.float64)
    clip_value = audit["self_gr_clip"].to_numpy(np.float64)
    mode = audit["self_gr_mode"].astype(str)
    sha_columns = [
        "logical_content_sha256",
        "schema_sha256",
        "surface_logical_content_sha256",
        "surface_manifest_sha256",
    ]
    sha_complete = all(len(str(frozen.get(key) or "")) == 64 for key in sha_columns)
    runtime_limit = float(get_nested(config, "runtime.maximum_seconds_per_shard"))
    shard_runtime = [
        float(summary["runtime_seconds"]) for summary in shard_summaries
    ]
    technical = {
        "all_input_sha_matches": True,
        "raw_identity_sha256": raw_preflight["content_sha256"],
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "audit_wells": len(audit),
        "all_wells_completed_without_fallback": bool(audit["status"].eq("ok").all()),
        "finite_candidate_coverage": finite_coverage,
        "self_gr_alpha_max_abs_error": float(np.abs(alpha - 0.07).max()),
        "self_gr_clip_max_abs_error": float(np.abs(clip_value - 1.0).max()),
        "self_gr_mode_exact": bool(mode.eq("boost_only").all()),
        "positive_self_gr_valid_rows": int(audit["self_gr_valid_rows"].sum()),
        "positive_self_gr_boost_applications": int(
            audit["positive_boost_application_count"].sum()
        ),
        "truth_or_reporting_values_parsed_before_freeze": before_freeze,
        "execution_counts": actual_counts,
        "execution_count_match": execution_count_match,
        "baseline_metric_parity": baseline_parity,
        "preflight_passed": bool(
            preflight_summary.get("technical_gate", {}).get("passed")
        ),
        "preflight_alpha0_arithmetic_max_abs_parity_ft": float(
            preflight_summary.get("technical_gate", {}).get(
                "alpha0_arithmetic_max_abs_parity_ft", float("inf")
            )
        ),
        "shard_runtime_seconds": shard_runtime,
        "runtime_limit_seconds_per_shard": runtime_limit,
        "prediction_logical_content_sha256": frozen["logical_content_sha256"],
        "surface_logical_content_sha256": frozen["surface_logical_content_sha256"],
        "schema_sha256": frozen["schema_sha256"],
        "manifest_sha256": frozen["surface_manifest_sha256"],
        "required_sha_complete": sha_complete,
    }
    technical["passed"] = bool(
        technical["prediction_rows"] == expected_rows
        and technical["prediction_wells"] == expected_wells
        and technical["reporting_folds"] == expected_folds
        and technical["audit_wells"] == expected_wells
        and technical["all_wells_completed_without_fallback"]
        and finite_coverage == float(
            technical_config["require_finite_candidate_coverage"]
        )
        and technical["self_gr_alpha_max_abs_error"] <= 1.0e-12
        and technical["self_gr_clip_max_abs_error"] <= 1.0e-12
        and technical["self_gr_mode_exact"]
        and technical["positive_self_gr_valid_rows"] > 0
        and technical["positive_self_gr_boost_applications"] > 0
        and all(int(value) == 0 for value in before_freeze.values())
        and execution_count_match
        and all(bool(value["passed"]) for value in baseline_parity.values())
        and technical["preflight_passed"]
        and technical["preflight_alpha0_arithmetic_max_abs_parity_ft"]
        <= float(
            technical_config[
                "require_preflight_alpha0_arithmetic_max_abs_parity_ft"
            ]
        )
        and len(shard_runtime) == SHARD_COUNT
        and all(value <= runtime_limit for value in shard_runtime)
        and sha_complete
    )
    fold_rows = primary_metrics.loc[
        primary_metrics["scope"].str.startswith("fold_")
    ]
    improved_folds = int(
        (fold_rows["delta_rmse_candidate_minus_control"] <= 0.0).sum()
    )
    non_regression_scopes = {}
    scope_gate_keys = {
        "raw_gr_observed": "maximum_raw_gr_observed_regression_ft",
        "raw_gr_missing": "maximum_raw_gr_missing_regression_ft",
        "missing_fraction_high": "maximum_high_missing_well_regression_ft",
        "md_since_1000_plus": "maximum_long_tail_1000_plus_regression_ft",
        "hidden_like_spatial": "maximum_hidden_like_spatial_regression_ft",
        "hidden_like_typewell_purged": (
            "maximum_hidden_like_typewell_purged_regression_ft"
        ),
    }
    for scope, key in scope_gate_keys.items():
        delta = float(
            _scope_row(primary_metrics, scope)[
                "delta_rmse_candidate_minus_control"
            ]
        )
        maximum = float(scientific_config[key])
        non_regression_scopes[scope] = {
            "delta_rmse_candidate_minus_control": delta,
            "maximum_regression_ft": maximum,
            "passed": delta <= maximum,
        }
    by_well_delta = by_well_metrics["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well = float(by_well_delta.max())
    primary_gate = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            scientific_config["minimum_primary_scale5_rmse_gain_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(
            scientific_config["minimum_primary_improved_folds"]
        ),
        "arithmetic_delta_rmse_candidate_minus_control": float(
            secondary_overall["delta_rmse_candidate_minus_control"]
        ),
        "maximum_arithmetic_regression_ft": float(
            scientific_config["maximum_arithmetic_mean_regression_ft"]
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
    }
    primary_gate["passed"] = bool(
        primary_gate["improvement_ft"] >= primary_gate["minimum_improvement_ft"]
        and improved_folds >= primary_gate["minimum_improved_folds"]
        and primary_gate["arithmetic_delta_rmse_candidate_minus_control"]
        <= primary_gate["maximum_arithmetic_regression_ft"]
        and all(value["passed"] for value in non_regression_scopes.values())
        and by_well_p95 <= primary_gate["maximum_by_well_rmse_delta_p95"]
        and worst_well <= primary_gate["maximum_worst_well_rmse_regression"]
    )
    blend_guard = {
        "candidate_rmse": float(blend_overall["candidate_rmse"]),
        "control_rmse": float(blend_overall["control_rmse"]),
        "delta_rmse_candidate_minus_control": float(
            blend_overall["delta_rmse_candidate_minus_control"]
        ),
        "maximum_regression_ft": float(
            scientific_config["maximum_fixed_hmm_likpf_blend_regression_ft"]
        ),
    }
    blend_guard["passed"] = bool(
        blend_guard["delta_rmse_candidate_minus_control"]
        <= blend_guard["maximum_regression_ft"]
    )
    passed = bool(
        technical["passed"] and primary_gate["passed"] and blend_guard["passed"]
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            "eligible_for_separate_raw_test_regeneration_design_in_same_experiment"
            if passed
            else "terminal_close_without_self_gr_or_pf_rescue_grid"
        ),
        "technical_gate": technical,
        "primary_scientific_gate": primary_gate,
        "fixed_exp209_hmm_likpf_50_50_guard": blend_guard,
        "secondary_policy": (
            "arithmetic mean is a fixed non-regression guard and cannot replace "
            "the preregistered temperature-5 primary"
        ),
        "failure_action": (
            "close_without_alpha_clip_window_topk_temperature_gr_sigma_particle_"
            "seed_transition_resampling_blend_selector_or_same_oof_rescue"
        ),
    }


# %% [markdown]
# ## 9. Generated artifacts and stage orchestration


# %%
def input_manifest_frame(raw_preflight: dict[str, Any], preflight: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "name": "raw_train_well_identity",
            "path": raw_preflight["path"],
            "bytes": None,
            "raw_sha256": None,
            "decompressed_sha256": None,
            "logical_content_sha256": raw_preflight["content_sha256"],
            "data_rows": None,
            "columns": None,
        }
    ]
    for name, report in preflight["reports"].items():
        if name == "exp072_secondary_scale_control":
            continue
        rows.append(
            {
                "name": name,
                "path": report.get("path"),
                "bytes": report.get("bytes"),
                "raw_sha256": report.get("raw_sha256"),
                "decompressed_sha256": report.get("decompressed_sha256"),
                "logical_content_sha256": report.get("content_sha256"),
                "data_rows": report.get("data_rows"),
                "columns": json.dumps(report.get("columns"), separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }
    if path.suffix == ".gz":
        report["decompressed_sha256"] = inspect_gzip_csv(path)["decompressed_sha256"]
    return report


def build_artifact_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        report = artifact_report(path)
        rows.append({"name": name, **report})
    return pd.DataFrame(rows).sort_values("name", kind="mergesort").reset_index(drop=True)


def _artifact_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.glob(f"**/{filename}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} under {root}; found={matches}"
        )
    return matches[0]


def selected_stage(config: Mapping[str, Any]) -> str | None:
    flags = {
        "preflight": bool(get_nested(dict(config), "execution.run_preflight")),
        "full_shard": bool(get_nested(dict(config), "execution.run_full")),
        "merge": bool(get_nested(dict(config), "execution.run_merge")),
    }
    active = [stage for stage, enabled in flags.items() if enabled]
    if len(active) > 1:
        raise ValueError("exp429 permits exactly one execution stage at a time")
    return active[0] if active else None


def run_full_shard_stage(config: dict[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    load_preflight_summary(config)
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get(
        "EXPERIMENT_ALLOW_LOCAL"
    ) != "1":
        raise RuntimeError(
            "exp429 full PF shards must run first on Kaggle CPU"
        )
    started = time.time()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    manifest = assign_lpt_shards(build_raw_well_manifest(config, raw_dir))
    shard_index = int(get_nested(config, "execution.selected_shard_index"))
    if shard_index not in range(SHARD_COUNT):
        raise ValueError("selected_shard_index must be in [0, 3]")
    selected = manifest.loc[
        manifest["shard_index"].eq(shard_index)
    ].sort_values("well_id", kind="mergesort")
    wells = selected["well_id"].astype(str).tolist()
    expected_rows = int(selected["suffix_rows"].sum())
    artifacts = artifact_dir()
    manifest_path = (
        artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}_well_manifest.csv"
    )
    selected.to_csv(manifest_path, index=False)
    ledger = TruthAccessLedger()
    candidate, audit, frozen, frozen_paths = generate_and_freeze_predictions(
        raw_dir,
        artifacts,
        config,
        wells,
        ledger,
        artifact_tag=f"shard{shard_index}",
        expected_rows=expected_rows,
    )
    runtime_seconds = time.time() - started
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "full_shard",
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "runtime_seconds": runtime_seconds,
        "rows": len(candidate),
        "wells": len(wells),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "positive_self_gr_valid_rows": int(audit["self_gr_valid_rows"].sum()),
        "positive_self_gr_boost_applications": int(
            audit["positive_boost_application_count"].sum()
        ),
        "well_manifest_sha256": sha256_path(manifest_path),
        "frozen_prediction": frozen,
        "generated_artifacts": {
            name: artifact_report(path)
            for name, path in {
                **frozen_paths,
                "well_manifest": manifest_path,
            }.items()
        },
        "truth_access_ledger": ledger.report(),
        "runtime_versions": runtime_versions(),
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_shard{shard_index}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def load_preflight_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = _input_spec(dict(config), "preflight_result")
    path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    expected_sha = str(spec.get("expected_sha256") or "")
    if len(expected_sha) != 64 or sha256_path(path) != expected_sha:
        raise ValueError("approved preflight summary SHA is missing or mismatched")
    summary = json.loads(path.read_text())
    if (
        summary.get("experiment") != EXPERIMENT_NAME
        or summary.get("stage") != "preflight"
        or not bool(summary.get("technical_gate", {}).get("passed"))
    ):
        raise ValueError("approved preflight summary did not pass")
    return summary


def resolve_shard_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [
        Path(str(value))
        for value in (
            get_nested(dict(config), "execution.merge_shard_dirs") or []
        )
    ]
    if len(roots) != SHARD_COUNT:
        raise ValueError("execution.merge_shard_dirs must list four ordered roots")
    return roots


def read_shard_manifest(path: Path) -> pd.DataFrame:
    """Restore the compact shard-index dtype used by deterministic LPT."""
    return pd.read_csv(
        path,
        dtype={"well_id": str, "shard_index": np.int8},
    )


def merge_and_freeze_shards(
    config: dict[str, Any],
    roots: Sequence[Path],
    artifacts: Path,
    raw_manifest: pd.DataFrame,
    ledger: TruthAccessLedger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Path],
    list[dict[str, Any]],
]:
    candidates: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    surfaces: list[pd.DataFrame] = []
    shard_manifests: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    prediction_dtypes: dict[str, Any] = {
        "id": str,
        "well_id": str,
        "row_idx": np.int64,
        "suffix_offset": np.int64,
        "raw_gr_observed": bool,
        "self_gr_quality": np.float32,
        "self_gr_valid": np.float32,
        "self_gr_peak_tvt": np.float64,
        "self_gr_peak_gap": np.float32,
        "self_gr_typewell_agreement": np.float32,
        PRIMARY_CANDIDATE: np.float32,
        SECONDARY_CANDIDATE: np.float32,
    }
    for shard_index, root in enumerate(roots):
        summary = json.loads(
            _artifact_file(
                root,
                f"{OUTPUT_PREFIX}_shard{shard_index}_summary.json",
            ).read_text()
        )
        if (
            summary.get("stage") != "full_shard"
            or int(summary.get("shard_index", -1)) != shard_index
            or int(summary.get("shard_count", -1)) != SHARD_COUNT
        ):
            raise ValueError(f"shard {shard_index} summary contract mismatch")
        prediction_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_candidate_predictions.csv.gz",
        )
        audit_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_well_surface_pf_audit.csv",
        )
        surface_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_surface_manifest.csv",
        )
        manifest_path = _artifact_file(
            root,
            f"{OUTPUT_PREFIX}_shard{shard_index}_well_manifest.csv",
        )
        prediction = pd.read_csv(
            prediction_path,
            dtype=prediction_dtypes,
        )
        audit = pd.read_csv(audit_path, dtype={"well_id": str})
        surface = pd.read_csv(surface_path, dtype={"well_id": str})
        manifest = read_shard_manifest(manifest_path)
        frozen = summary["frozen_prediction"]
        logical_sha = dataframe_content_sha(
            prediction, list(frozen["logical_columns"])
        )
        if logical_sha != str(frozen["logical_content_sha256"]):
            raise ValueError(f"shard {shard_index} logical prediction SHA mismatch")
        if dataframe_content_sha(
            surface,
            [
                "well_id",
                "self_gr_surface_rows",
                "self_gr_surface_states",
                "self_gr_prefix_anchor_count",
                "self_gr_valid_rows",
                "self_gr_surface_logical_sha256",
            ],
        ) != str(frozen["surface_logical_content_sha256"]):
            raise ValueError(f"shard {shard_index} surface SHA mismatch")
        if not manifest["shard_index"].eq(shard_index).all():
            raise ValueError(f"shard {shard_index} manifest assignment mismatch")
        candidates.append(prediction)
        audits.append(audit)
        surfaces.append(surface)
        shard_manifests.append(manifest)
        summaries.append(summary)
    candidate = (
        pd.concat(candidates, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.concat(audits, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    surface = (
        pd.concat(surfaces, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    merged_manifest = (
        pd.concat(shard_manifests, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_manifest = raw_manifest.sort_values(
        "well_id", kind="mergesort"
    ).reset_index(drop=True)
    columns = ["well_id", "rows", "prefix_rows", "suffix_rows", "shard_index"]
    if not merged_manifest[columns].equals(expected_manifest[columns]):
        raise ValueError("merged shard manifest differs from deterministic raw LPT")
    if (
        len(candidate) != int(get_nested(config, "validation.expected_rows"))
        or candidate["well_id"].nunique()
        != int(get_nested(config, "validation.expected_wells"))
        or candidate["id"].duplicated().any()
        or audit["well_id"].duplicated().any()
        or surface["well_id"].duplicated().any()
    ):
        raise ValueError("merged shard coverage mismatch")
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_merged_candidate_predictions.csv.gz"
    audit_path = artifacts / f"{OUTPUT_PREFIX}_merged_well_surface_pf_audit.csv"
    surface_path = artifacts / f"{OUTPUT_PREFIX}_merged_surface_manifest.csv"
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_merged_well_manifest.csv"
    write_deterministic_gzip_csv(candidate, prediction_path)
    audit.to_csv(audit_path, index=False)
    surface.to_csv(surface_path, index=False)
    merged_manifest.to_csv(manifest_path, index=False)
    logical_columns = ["id", "well_id", "row_idx", *PREDICTION_COLUMNS]
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "logical_columns": logical_columns,
        "logical_content_sha256": dataframe_content_sha(
            candidate, logical_columns
        ),
        "schema_sha256": dataframe_schema_sha(candidate),
        "raw_gzip_sha256": sha256_path(prediction_path),
        "surface_logical_content_sha256": dataframe_content_sha(
            surface,
            [
                "well_id",
                "self_gr_surface_rows",
                "self_gr_surface_states",
                "self_gr_prefix_anchor_count",
                "self_gr_valid_rows",
                "self_gr_surface_logical_sha256",
            ],
        ),
        "surface_manifest_sha256": sha256_path(surface_path),
        "well_surface_pf_audit_sha256": sha256_path(audit_path),
        "merged_well_manifest_sha256": sha256_path(manifest_path),
        "shard_logical_content_sha256": [
            summary["frozen_prediction"]["logical_content_sha256"]
            for summary in summaries
        ],
        "truth_or_reporting_values_parsed_before_freeze": ledger.report()[
            "before_freeze"
        ],
    }
    ledger.mark_frozen()
    return (
        candidate,
        audit,
        frozen,
        {
            "prediction": prediction_path,
            "well_surface_pf_audit": audit_path,
            "surface_manifest": surface_path,
            "well_manifest": manifest_path,
        },
        summaries,
    )


def run_merge_stage(config: dict[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(
        config, require_run_approval=True
    )
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    raw_manifest = assign_lpt_shards(build_raw_well_manifest(config, raw_dir))
    preflight = preflight_saved_inputs(config)
    preflight_summary = load_preflight_summary(config)
    ledger = TruthAccessLedger()
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    write_json(contract_path, scientific_contract)
    input_manifest_frame(raw_preflight, preflight).to_csv(
        input_manifest_path, index=False
    )
    (
        candidate,
        audit,
        frozen,
        frozen_paths,
        shard_summaries,
    ) = merge_and_freeze_shards(
        config,
        resolve_shard_roots(config),
        artifacts,
        raw_manifest,
        ledger,
    )
    prediction_frozen_at_seconds = time.time() - started
    frame, late_attachment = load_late_readout_frame(
        candidate,
        frozen,
        preflight,
        raw_dir,
        config,
        ledger,
    )
    (
        primary_metrics,
        by_well_metrics,
        secondary_metrics,
        blend_metrics,
    ) = build_metric_outputs(frame)
    promotion_gate = evaluate_promotion_gate(
        frame,
        primary_metrics,
        by_well_metrics,
        secondary_metrics,
        blend_metrics,
        audit,
        preflight,
        raw_preflight,
        frozen,
        ledger,
        shard_summaries,
        preflight_summary,
        config,
    )
    metric_paths = {
        "primary_metrics": artifacts / f"{OUTPUT_PREFIX}_primary_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "secondary_metrics": artifacts / f"{OUTPUT_PREFIX}_secondary_metrics.csv",
        "fixed_hmm_likpf_blend_metrics": (
            artifacts / f"{OUTPUT_PREFIX}_fixed_hmm_likpf_blend_metrics.csv"
        ),
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
    }
    primary_metrics.to_csv(metric_paths["primary_metrics"], index=False)
    by_well_metrics.to_csv(metric_paths["by_well_metrics"], index=False)
    secondary_metrics.to_csv(metric_paths["secondary_metrics"], index=False)
    blend_metrics.to_csv(
        metric_paths["fixed_hmm_likpf_blend_metrics"], index=False
    )
    write_json(metric_paths["promotion_gate"], promotion_gate)
    artifact_manifest = build_artifact_manifest(
        {
            **frozen_paths,
            **metric_paths,
            "scientific_contract": contract_path,
            "input_manifest": input_manifest_path,
        }
    )
    artifact_manifest_path = artifacts / f"{OUTPUT_PREFIX}_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    status = (
        "completed_train_side_gate_passed_no_automatic_downstream"
        if promotion_gate["passed"]
        else "completed_train_side_gate_failed_terminal_close"
    )
    overall = _scope_row(primary_metrics, "overall")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "runtime_seconds": time.time() - started,
        "prediction_frozen_at_seconds": prediction_frozen_at_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "active_scientific_variants": 1,
        "candidate_pf_well_runs": len(audit),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "models": 0,
        "boosters": 0,
        "parent_pf_control_reruns": 0,
        "hmm_reruns": 0,
        "beam_reruns": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": (
            scientific_contract["scientific_contract_sha256"]
        ),
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
        "frozen_prediction": frozen,
        "truth_attachment": late_attachment,
        "promotion_gate": promotion_gate,
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    write_json(
        metrics_output_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "status": status,
            "route": "pf_beam",
            "cv": float(overall["candidate_rmse"]),
            "public_lb": None,
            "private_lb": None,
            "metric": "rmse",
            "overall": overall.to_dict(),
            "promotion_gate": promotion_gate,
            "prediction_sha256": frozen["logical_content_sha256"],
            "surface_sha256": frozen["surface_logical_content_sha256"],
            "artifact_manifest_sha256": sha256_path(artifact_manifest_path),
            "model_sha256": None,
            "submission_sha256": None,
        },
    )
    print(primary_metrics.to_string(index=False))
    print(secondary_metrics.to_string(index=False))
    print(json.dumps(to_jsonable(promotion_gate), indent=2, sort_keys=True))
    return summary


def run_selected_stage(config: dict[str, Any]) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    if stage == "preflight":
        return run_preflight_stage(config)
    if stage == "full_shard":
        return run_full_shard_stage(config)
    if stage == "merge":
        return run_merge_stage(config)
    raise AssertionError(stage)


# %% [markdown]
# ## 10. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "primary_candidate": PRIMARY_CANDIDATE,
                "secondary_candidate": SECONDARY_CANDIDATE,
                "primary_control": PRIMARY_CONTROL,
                "secondary_control": SECONDARY_CONTROL,
                "selected_stage": selected_stage(CONFIG),
                "scientific_variants": get_nested(
                    CONFIG, "model.execution_count.scientific_variants"
                ),
                "full_candidate_pf_well_runs": get_nested(
                    CONFIG, "model.execution_count.full_candidate_pf_well_runs"
                ),
                "full_seed_well_trajectories": get_nested(
                    CONFIG, "model.execution_count.full_seed_well_trajectories"
                ),
                "full_particle_starts": get_nested(
                    CONFIG, "model.execution_count.full_particle_starts"
                ),
                "parent_full_control_reruns": 0,
                "lightgbm_configs": 0,
                "trained_folds": 0,
                "boosters": 0,
                "models": 0,
                "hmm_well_runs": 0,
                "beam_well_runs": 0,
                "gpu_runs": 0,
                "kaggle_package_approved": get_nested(
                    CONFIG, "execution.kaggle_package_approved"
                ),
                "preflight_run_approved": get_nested(
                    CONFIG, "execution.preflight_run_approved"
                ),
                "full_run_approved": get_nested(
                    CONFIG, "execution.full_run_approved"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 11. Run the selected Kaggle CPU stage


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_selected_stage(CONFIG)
