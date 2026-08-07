# %% [markdown]
# # exp400 all-well GR likelihood scale ×1.3 PF train
#
# Train-side audit of one preregistered change to the deterministic exp072
# likelihood-weighted particle filter. The already-clipped well-level GR
# residual scale is multiplied by exactly 1.3 for every train well. Particle
# dynamics, seed policy, resampling, interpolation, seed aggregation, and all
# saved controls remain fixed. Unknown-suffix TVT and reporting roles are not
# read until candidate predictions and their logical content SHA are frozen.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Truth-free exp072 input preparation
# 5. Exact exp072 likelihood-PF kernel with diagnostics
# 6. Candidate generation and prediction freeze
# 7. Late truth, control, fold, and hidden-like attachment
# 8. Paired metrics and promotion gates
# 9. Generated artifacts and execution orchestration
# 10. Setup and configuration preview
# 11. Run the Kaggle CPU audit

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


EXPERIMENT_NAME = "exp400_all_well_1p3_sigma_gr_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "likpf_mean_x1p3"
SCALE_CANDIDATES = (
    "likpf_scale_3_x1p3",
    "likpf_scale_5_x1p3",
    "likpf_scale_8_x1p3",
    "likpf_scale_12_x1p3",
)
PREDICTION_COLUMNS = (PRIMARY_CANDIDATE, *SCALE_CANDIDATES)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP400_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp400 config not found; checked={checked}")


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
        "columns": pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist(),
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
        "lineage.parent": "exp072_exp063_full_replay_feature_cache",
        "implementation.enabled": True,
        "implementation.scope": "train_side_candidate_audit_only",
        "model.active_variants": ["all_well_gs_x1p3"],
        "model.promotion_candidate": PRIMARY_CANDIDATE,
        "model.gr_scale.multiplier": 1.3,
        "model.gr_scale.application_scope": "all_wells",
        "model.gr_scale.application_order": "clip_base_then_multiply_exactly_once",
        "model.gr_scale.post_multiplier_clip": None,
        "model.pf.particles": 500,
        "model.pf.seeds": 128,
        "model.pf.seed_weighting_scales": [3.0, 5.0, 8.0, 12.0],
        "model.pf.initial_position_spread_ft": 4.5,
        "model.pf.initial_rate_spread": 0.01,
        "model.pf.momentum": 0.998,
        "model.pf.rate_noise": 0.002,
        "model.pf.position_noise": 0.005,
        "model.pf.rough_position": 0.1,
        "model.pf.rough_rate": 0.001,
        "model.pf.resample_threshold_fraction": 0.5,
        "model.pf.emission_clip_z2": 600.0,
        "model.pf.typewell_tvt_pad_ft": 100.0,
        "model.execution_count.scientific_variants": 1,
        "model.execution_count.candidate_pf_well_runs": 773,
        "model.execution_count.seed_well_trajectories": 98944,
        "model.execution_count.particle_starts": 49472000,
        "model.execution_count.parent_pf_control_reruns": 0,
        "model.execution_count.hmm_well_runs": 0,
        "model.execution_count.beam_well_runs": 0,
        "model.execution_count.boosters": 0,
        "runtime.num_workers": 8,
        "runtime.device": "cpu",
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "execution.submit_to_kaggle": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp400 fixed contract mismatch: {key} must be {value!r}")
    if [float(value) for value in get_nested(config, "model.gr_scale.base_clip")] != [
        10.0,
        60.0,
    ]:
        raise ValueError("exp400 fixes the base GR scale clip to [10, 60]")
    if [float(value) for value in get_nested(config, "model.gr_scale.effective_range")] != [
        13.0,
        78.0,
    ]:
        raise ValueError("exp400 fixes the effective GR scale range to [13, 78]")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp400 implementation approval must be recorded")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.train_run_approved"))
        and bool(get_nested(config, "execution.run_train"))
    ):
        raise RuntimeError("exp400 Kaggle package/push/train run is not approved")
    contract = build_scientific_contract(config)
    return contract


def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "truth_attached": False,
        "primary_candidate": PRIMARY_CANDIDATE,
        "secondary_candidates": list(SCALE_CANDIDATES),
        "gr_scale": get_nested(config, "model.gr_scale"),
        "pf": get_nested(config, "model.pf"),
        "execution_counts": get_nested(config, "model.execution_count"),
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "controls": {
            "exp072_pf": "saved_load_only",
            "exp209_hmm": "saved_load_only_fixed_50_50_guard",
            "pf_control_reruns": 0,
            "hmm_reruns": 0,
            "beam_reruns": 0,
        },
        "forbidden": get_nested(config, "guards.forbidden"),
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
        "exp209_hmm_control": _input_spec(config, "exp209_hmm_control"),
        "fold_assignment": _input_spec(config, "fold_assignment"),
        "hidden_like_assignment": _input_spec(config, "hidden_like_assignment"),
    }
    paths = {
        name: resolve_existing(str(spec["filename"]), spec.get("candidates", []))
        for name, spec in specs.items()
    }
    reports: dict[str, Any] = {}
    for name in ("exp072_control", "exp209_hmm_control", "fold_assignment"):
        report = inspect_gzip_csv(paths[name])
        expected = str(specs[name]["expected_decompressed_sha256"])
        if report["decompressed_sha256"] != expected:
            raise ValueError(f"{name} decompressed SHA mismatch")
        reports[name] = report
    exp072_expected_raw = str(specs["exp072_control"]["expected_raw_gzip_sha256"])
    if reports["exp072_control"]["raw_sha256"] != exp072_expected_raw:
        raise ValueError("exp072 control raw gzip SHA mismatch")
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
    for name in ("exp072_control", "exp209_hmm_control", "fold_assignment"):
        if int(reports[name]["data_rows"]) != expected_rows:
            raise ValueError(f"{name} row count mismatch")
    exp072_columns = set(reports["exp072_control"]["columns"])
    saved_scale_columns = {f"likpf_scale_{scale}_d" for scale in (3, 5, 8, 12)}
    reports["exp072_secondary_scale_control"] = {
        "expected_columns": sorted(saved_scale_columns),
        "available_columns": sorted(saved_scale_columns & exp072_columns),
        "all_available": saved_scale_columns.issubset(exp072_columns),
        "policy": (
            "paired_saved_x1p0_scale_diagnostics"
            if saved_scale_columns.issubset(exp072_columns)
            else "candidate_only_nonselective_diagnostics_no_parent_pf_rerun"
        ),
    }
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
# ## 4. Truth-free exp072 input preparation


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


# %% [markdown]
# ## 5. Exact exp072 likelihood-PF kernel with diagnostics


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
    """Exact exp072 kernel plus passive resampling/ESS/clip counters."""
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
    ) = _pf_lik_allseeds(
        prepared["eval_md"],
        prepared["eval_z"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
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
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
    }
    return outputs, diagnostics, predictions, log_likelihoods


# %% [markdown]
# ## 6. Candidate generation and prediction freeze


# %%
def warm_up_pf_kernel() -> None:
    md = np.linspace(1.0, 8.0, 8)
    z = np.zeros(8)
    observed_gr = np.full(8, 50.0)
    grid_gr = np.linspace(45.0, 55.0, 100)
    _pf_lik_allseeds(
        md,
        z,
        observed_gr,
        grid_gr,
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
        0.1,
        0.001,
        0.5,
        4.5,
    )


def decode_well(
    well: str,
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    multiplier = float(get_nested(config, "model.gr_scale.multiplier"))
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        multiplier=multiplier,
        grid_step=float(get_nested(config, "model.pf.typewell_grid_step_ft")),
    )
    pf_config = get_nested(config, "model.pf") or {}
    seed_base = stable_seed("likpf", "train", well)
    outputs, diagnostics, _, _ = run_likelihood_pf(
        prepared,
        particles=int(pf_config["particles"]),
        seeds=int(pf_config["seeds"]),
        scales=[float(value) for value in pf_config["seed_weighting_scales"]],
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
            PRIMARY_CANDIDATE: outputs["pf_mean"].astype(np.float32),
            "likpf_scale_3_x1p3": outputs["pf_scale_3"].astype(np.float32),
            "likpf_scale_5_x1p3": outputs["pf_scale_5"].astype(np.float32),
            "likpf_scale_8_x1p3": outputs["pf_scale_8"].astype(np.float32),
            "likpf_scale_12_x1p3": outputs["pf_scale_12"].astype(np.float32),
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    if ledger.prediction_frozen:
        raise RuntimeError("prediction ledger is already frozen")
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if wells != sorted(wells) or len(wells) != expected_wells:
        raise ValueError("exp400 requires all sorted train well IDs exactly once")
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
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if (
        len(candidate) != expected_rows
        or candidate["well_id"].nunique() != expected_wells
        or len(audit) != expected_wells
        or audit["well_id"].nunique() != expected_wells
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("candidate generation coverage mismatch")
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_candidate_predictions.csv.gz"
    audit_path = artifacts / f"{OUTPUT_PREFIX}_well_gr_scale_audit.csv"
    write_deterministic_gzip_csv(candidate, prediction_path)
    audit.to_csv(audit_path, index=False)
    logical_columns = [
        "id",
        "well_id",
        "row_idx",
        *PREDICTION_COLUMNS,
    ]
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "logical_columns": logical_columns,
        "logical_content_sha256": dataframe_content_sha(candidate, logical_columns),
        "schema_sha256": dataframe_schema_sha(candidate),
        "raw_gzip_sha256": sha256_path(prediction_path),
        "well_gr_scale_audit_sha256": sha256_path(audit_path),
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
            "well_gr_scale_audit": audit_path,
        },
    )


def _require_frozen_prediction(frozen: dict[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("late attachment requires a frozen prediction")
    value = str(frozen.get("logical_content_sha256") or "")
    if len(value) != 64:
        raise RuntimeError("frozen prediction logical content SHA is missing")


# %% [markdown]
# ## 7. Late truth, control, fold, and hidden-like attachment


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

    exp072_scale_columns = [f"likpf_scale_{scale}_d" for scale in (3, 5, 8, 12)]
    available_scale_columns = [
        column
        for column in exp072_scale_columns
        if column in preflight["reports"]["exp072_control"]["columns"]
    ]
    exp072_usecols = [
        "id",
        "last_known_tvt",
        "likpf_mean_d",
        *available_scale_columns,
    ]
    exp072 = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=exp072_usecols,
        dtype={"id": str},
    )
    for column in exp072_usecols:
        if column != "id":
            exp072[column] = pd.to_numeric(exp072[column], errors="raise")
    exp072["saved_exp072_likpf_mean"] = exp072["last_known_tvt"] + exp072["likpf_mean_d"]
    control_columns = ["saved_exp072_likpf_mean"]
    for scale in (3, 5, 8, 12):
        source = f"likpf_scale_{scale}_d"
        if source in exp072:
            target = f"saved_exp072_likpf_scale_{scale}"
            exp072[target] = exp072["last_known_tvt"] + exp072[source]
            control_columns.append(target)
    frame = _align_on_id(
        frame,
        exp072[["id", *control_columns]],
        control_columns,
        label="saved exp072 control",
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

    fold_spec = _input_spec(config, "fold_assignment")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    forbidden_columns = {str(value) for value in fold_spec.get("forbidden_decoder_columns", [])}
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp400 fold allowlist must contain identity/fold columns only")
    if set(safe_columns) & forbidden_columns:
        raise ValueError("exp400 fold allowlist contains forbidden decoder columns")
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

    hidden_spec = _input_spec(config, "hidden_like_assignment")
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
    frame["parent_hmm_50_50"] = 0.5 * (frame["saved_exp072_likpf_mean"] + frame["saved_exp209_hmm"])
    if not np.isfinite(
        frame[
            [
                "true_tvt",
                "saved_exp072_likpf_mean",
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
        "saved_exp072_scale_controls_available": available_scale_columns,
        "saved_exp072_scale_control_policy": preflight["reports"]["exp072_secondary_scale_control"][
            "policy"
        ],
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 8. Paired metrics and promotion gates


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
            control_column="saved_exp072_likpf_mean",
            comparison="direct_saved_exp072_likpf_mean",
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
            comparison="fixed_exp209_hmm_likpf_50_50",
            scope=scope,
        )
        for scope, mask in scopes
    ]
    secondary_rows: list[dict[str, Any]] = []
    overall_mask = np.ones(len(frame), dtype=bool)
    for scale in (3, 5, 8, 12):
        candidate_column = f"likpf_scale_{scale}_x1p3"
        saved_column = f"saved_exp072_likpf_scale_{scale}"
        control_column = saved_column if saved_column in frame.columns else None
        secondary_rows.append(
            metric_record(
                frame,
                overall_mask,
                candidate_column=candidate_column,
                control_column=control_column,
                comparison=(
                    "paired_saved_exp072_scale"
                    if control_column is not None
                    else "candidate_only_saved_x1p0_scale_unavailable"
                ),
                scope="overall",
            )
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
    blend_metrics: pd.DataFrame,
    audit: pd.DataFrame,
    preflight: dict[str, Any],
    raw_preflight: dict[str, Any],
    frozen: dict[str, Any],
    ledger: TruthAccessLedger,
    runtime_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    technical_config = get_nested(config, "guards.technical") or {}
    scientific_config = get_nested(config, "guards.scientific") or {}
    overall = _scope_row(primary_metrics, "overall")
    blend_overall = _scope_row(blend_metrics, "overall")
    control_rmse_expected = float(get_nested(config, "validation.control_rmse_ft"))
    blend_rmse_expected = float(
        get_nested(config, "validation.fixed_hmm_likpf_blend_control_rmse_ft")
    )
    parity_tolerance = float(technical_config["require_saved_control_rmse_parity_atol_ft"])
    scale_tolerance = float(technical_config["multiplier_absolute_tolerance"])
    multiplier_error = np.abs(audit["multiplier"].to_numpy(np.float64) - 1.3)
    scale_error = np.abs(
        audit["gs_candidate"].to_numpy(np.float64) - 1.3 * audit["gs_base"].to_numpy(np.float64)
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_counts = get_nested(config, "model.execution_count") or {}
    actual_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": len(audit),
        "seeds_per_well": int(audit["seeds"].iloc[0]),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particles_per_seed": int(audit["particles"].iloc[0]),
        "particle_starts": int(audit["particle_starts"].sum()),
        "prediction_readouts": len(PREDICTION_COLUMNS),
        "reporting_folds": int(frame["fold"].nunique()),
        "parent_pf_control_reruns": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    count_keys = [
        "scientific_variants",
        "candidate_pf_well_runs",
        "seeds_per_well",
        "seed_well_trajectories",
        "particles_per_seed",
        "particle_starts",
        "prediction_readouts",
        "reporting_folds",
        "parent_pf_control_reruns",
        "hmm_well_runs",
        "beam_well_runs",
        "lightgbm_configs",
        "trained_folds",
        "boosters",
    ]
    execution_count_match = all(
        int(actual_counts[key]) == int(expected_counts[key]) for key in count_keys
    )
    before_freeze = ledger.report()["before_freeze"]
    finite_coverage = float(
        np.isfinite(frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).mean()
    )
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    baseline_parity = {
        "saved_exp072_likpf_mean": {
            "actual_rmse": float(overall["control_rmse"]),
            "expected_rmse": control_rmse_expected,
            "absolute_difference": abs(float(overall["control_rmse"]) - control_rmse_expected),
        },
        "saved_exp209_hmm_exp072_likpf_50_50": {
            "actual_rmse": float(blend_overall["control_rmse"]),
            "expected_rmse": blend_rmse_expected,
            "absolute_difference": abs(float(blend_overall["control_rmse"]) - blend_rmse_expected),
        },
    }
    for value in baseline_parity.values():
        value["passed"] = bool(value["absolute_difference"] <= parity_tolerance)
    technical = {
        "all_input_sha_matches": True,
        "raw_identity_sha256": raw_preflight["content_sha256"],
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "audit_wells": len(audit),
        "all_wells_completed_without_fallback": bool(audit["status"].eq("ok").all()),
        "finite_candidate_coverage": finite_coverage,
        "multiplier_max_abs_error": float(multiplier_error.max()),
        "effective_scale_max_abs_error": float(scale_error.max()),
        "base_scale_min": float(audit["gs_base"].min()),
        "base_scale_max": float(audit["gs_base"].max()),
        "candidate_scale_min": float(audit["gs_candidate"].min()),
        "candidate_scale_max": float(audit["gs_candidate"].max()),
        "post_multiplier_clip_count": int(audit["post_multiplier_clip_count"].sum()),
        "truth_or_reporting_values_parsed_before_freeze": before_freeze,
        "execution_counts": actual_counts,
        "execution_count_match": execution_count_match,
        "baseline_metric_parity": baseline_parity,
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.maximum_seconds")),
        "prediction_logical_content_sha256": frozen["logical_content_sha256"],
        "saved_exp072_scale_control_policy": preflight["reports"]["exp072_secondary_scale_control"][
            "policy"
        ],
    }
    technical["passed"] = bool(
        technical["prediction_rows"] == expected_rows
        and technical["prediction_wells"] == expected_wells
        and technical["reporting_folds"] == expected_folds
        and technical["audit_wells"] == expected_wells
        and technical["all_wells_completed_without_fallback"]
        and finite_coverage == float(technical_config["require_finite_candidate_coverage"])
        and technical["multiplier_max_abs_error"] <= scale_tolerance
        and technical["effective_scale_max_abs_error"] <= scale_tolerance
        and technical["base_scale_min"] >= 10.0
        and technical["base_scale_max"] <= 60.0
        and technical["candidate_scale_min"] >= 13.0
        and technical["candidate_scale_max"] <= 78.0
        and technical["post_multiplier_clip_count"] == 0
        and all(int(value) == 0 for value in before_freeze.values())
        and execution_count_match
        and all(bool(value["passed"]) for value in baseline_parity.values())
        and runtime_seconds <= technical["runtime_limit_seconds"]
    )

    fold_rows = primary_metrics.loc[primary_metrics["scope"].str.startswith("fold_")]
    folds_non_regressed = int((fold_rows["delta_rmse_candidate_minus_control"] <= 0.0).sum())
    observed = _scope_row(primary_metrics, "raw_gr_observed")
    non_regression_scopes = {
        "raw_gr_missing": float(
            _scope_row(primary_metrics, "raw_gr_missing")["delta_rmse_candidate_minus_control"]
        )
        <= float(scientific_config["maximum_raw_gr_missing_regression_ft"]),
        "missing_fraction_high": float(
            _scope_row(primary_metrics, "missing_fraction_high")[
                "delta_rmse_candidate_minus_control"
            ]
        )
        <= float(scientific_config["maximum_high_missing_well_regression_ft"]),
        "md_since_1000_plus": float(
            _scope_row(primary_metrics, "md_since_1000_plus")["delta_rmse_candidate_minus_control"]
        )
        <= float(scientific_config["maximum_long_tail_1000_plus_regression_ft"]),
        "hidden_like_spatial": float(
            _scope_row(primary_metrics, "hidden_like_spatial")["delta_rmse_candidate_minus_control"]
        )
        <= float(scientific_config["maximum_hidden_like_spatial_regression_ft"]),
        "hidden_like_typewell_purged": float(
            _scope_row(primary_metrics, "hidden_like_typewell_purged")[
                "delta_rmse_candidate_minus_control"
            ]
        )
        <= float(scientific_config["maximum_hidden_like_typewell_purged_regression_ft"]),
    }
    by_well_delta = by_well_metrics["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well = float(by_well_delta.max())
    primary_gate = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            scientific_config["minimum_direct_rmse_gain_vs_exp072_likpf_mean_ft"]
        ),
        "folds_non_regressed": folds_non_regressed,
        "minimum_folds_non_regressed": int(scientific_config["minimum_improved_folds"]),
        "raw_gr_observed_improvement_ft": float(observed["improvement_ft"]),
        "minimum_raw_gr_observed_improvement_ft": float(
            scientific_config["minimum_raw_gr_observed_gain_ft"]
        ),
        "non_regression_scopes": non_regression_scopes,
        "by_well_rmse_delta_p95": by_well_p95,
        "maximum_by_well_rmse_delta_p95": float(scientific_config["maximum_by_well_delta_p95_ft"]),
        "worst_well_rmse_regression": worst_well,
        "maximum_worst_well_rmse_regression": float(
            scientific_config["maximum_worst_well_regression_ft"]
        ),
    }
    primary_gate["passed"] = bool(
        primary_gate["improvement_ft"] >= primary_gate["minimum_improvement_ft"]
        and folds_non_regressed >= primary_gate["minimum_folds_non_regressed"]
        and primary_gate["raw_gr_observed_improvement_ft"]
        >= primary_gate["minimum_raw_gr_observed_improvement_ft"]
        and all(non_regression_scopes.values())
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
        blend_guard["delta_rmse_candidate_minus_control"] <= blend_guard["maximum_regression_ft"]
    )
    passed = bool(technical["passed"] and primary_gate["passed"] and blend_guard["passed"])
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            "eligible_for_separate_fail_closed_inference_implementation_design"
            if passed
            else "all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue"
        ),
        "technical_gate": technical,
        "primary_scientific_gate": primary_gate,
        "fixed_exp209_hmm_likpf_50_50_guard": blend_guard,
        "secondary_scale_policy": (
            "candidate scale 3/5/8/12 metrics are nonselective diagnostics and cannot "
            "replace the arithmetic pf_mean primary"
        ),
        "failure_action": (
            "close_without_multiplier_clip_particle_seed_scale_initial_spread_"
            "resampling_blend_selector_or_same_oof_rescue"
        ),
    }


# %% [markdown]
# ## 9. Generated artifacts and execution orchestration


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


def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp400 must run first on Kaggle; local execution requires explicit smoke approval"
        )
    scientific_contract = validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    preflight = preflight_saved_inputs(config)
    ledger = TruthAccessLedger()
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    write_json(contract_path, scientific_contract)
    input_manifest_frame(raw_preflight, preflight).to_csv(input_manifest_path, index=False)
    candidate, audit, frozen, frozen_paths = generate_and_freeze_predictions(
        raw_dir,
        artifacts,
        config,
        list(raw_preflight["well_ids"]),
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
    primary_metrics, by_well_metrics, secondary_metrics, blend_metrics = build_metric_outputs(frame)
    runtime_seconds = time.time() - started
    promotion_gate = evaluate_promotion_gate(
        frame,
        primary_metrics,
        by_well_metrics,
        blend_metrics,
        audit,
        preflight,
        raw_preflight,
        frozen,
        ledger,
        runtime_seconds,
        config,
    )
    metric_paths = {
        "overall_fold_scope_metrics": artifacts / f"{OUTPUT_PREFIX}_overall_fold_scope_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "secondary_scale_metrics": artifacts / f"{OUTPUT_PREFIX}_secondary_scale_metrics.csv",
        "fixed_hmm_likpf_blend_metrics": artifacts
        / f"{OUTPUT_PREFIX}_fixed_hmm_likpf_blend_metrics.csv",
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
    }
    primary_metrics.to_csv(metric_paths["overall_fold_scope_metrics"], index=False)
    by_well_metrics.to_csv(metric_paths["by_well_metrics"], index=False)
    secondary_metrics.to_csv(metric_paths["secondary_scale_metrics"], index=False)
    blend_metrics.to_csv(metric_paths["fixed_hmm_likpf_blend_metrics"], index=False)
    write_json(metric_paths["promotion_gate"], promotion_gate)
    manifest_sources = {
        **frozen_paths,
        **metric_paths,
        "scientific_contract": contract_path,
        "input_manifest": input_manifest_path,
    }
    artifact_manifest = build_artifact_manifest(manifest_sources)
    artifact_manifest_path = artifacts / f"{OUTPUT_PREFIX}_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    artifact_manifest_sha = sha256_path(artifact_manifest_path)
    status = (
        "train_side_all_well_likelihood_pf_gs_x1p3_gate_passed_no_automatic_downstream"
        if promotion_gate["passed"]
        else "train_side_all_well_likelihood_pf_gs_x1p3_gate_failed_closed"
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
        "candidate_pf_well_runs": len(audit),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_pf_control_reruns": 0,
        "hmm_reruns": 0,
        "beam_reruns": 0,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "artifact_manifest_sha256": artifact_manifest_sha,
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
    overall = _scope_row(primary_metrics, "overall")
    metrics = {
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
        "artifact_manifest_sha256": artifact_manifest_sha,
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Train-side candidate only. No parent PF, HMM, Beam, model, raw-test "
            "prediction, inference, or submission is produced."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(primary_metrics.to_string(index=False))
    print(secondary_metrics.to_string(index=False))
    print(json.dumps(to_jsonable(promotion_gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


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
                "secondary_candidates": list(SCALE_CANDIDATES),
                "candidate_pf_well_runs": get_nested(
                    CONFIG, "model.execution_count.candidate_pf_well_runs"
                ),
                "seed_well_trajectories": get_nested(
                    CONFIG, "model.execution_count.seed_well_trajectories"
                ),
                "particle_starts": get_nested(CONFIG, "model.execution_count.particle_starts"),
                "parent_pf_control_reruns": 0,
                "hmm_reruns": 0,
                "beam_reruns": 0,
                "boosters": 0,
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "train_run_approved": get_nested(CONFIG, "execution.train_run_approved"),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 11. Run the Kaggle CPU audit


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_full_experiment(CONFIG)
