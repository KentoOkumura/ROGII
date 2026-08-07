# %% [markdown]
# # exp430 Huber seed-evidence reaggregation train
#
# Replays the exact exp404 x1.0 likelihood-PF once, freezes the float64
# per-seed trajectory bank, and only then scores the same trajectories with
# fixed Gaussian and Huber evidence. Unknown-suffix TVT, reporting folds, and
# hidden-like roles remain inaccessible until all four full shards are frozen.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Truth-free exp404-compatible PF input preparation
# 5. Exact exp404 likelihood-PF trajectory kernel
# 6. Float64 trajectory-bank generation and freeze
# 7. Gaussian and Huber evidence readouts
# 8. Technical preflight and full-shard execution
# 9. Truth-late merge, metrics, and promotion gates
# 10. Generated artifacts and stage orchestration
# 11. Setup and configuration preview
# 12. Run the explicitly approved Kaggle CPU stage

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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


EXPERIMENT_NAME = "exp430_huber_seed_evidence_reaggregation"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CONTROL = "gaussian_matched"
PRIMARY_CANDIDATE = "huber_delta_1p345"
ARITHMETIC_CONTROL = "arithmetic_mean"
PARENT_REPLAY = "parent_gaussian_marginal_replay"
SAVED_PARENT = "saved_exp404_temperature5"
PREDICTION_COLUMNS = (
    PRIMARY_CONTROL,
    PRIMARY_CANDIDATE,
    ARITHMETIC_CONTROL,
    PARENT_REPLAY,
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP430_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n"
    )


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
    raise FileNotFoundError(f"exp430 config not found; checked={checked}")


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
        "content_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(csv_path, nrows=0, compression="gzip")
        .columns.astype(str)
        .tolist(),
    }


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    digest.update(canonical_json(chosen).encode())
    for column in chosen:
        series = frame[column]
        digest.update(column.encode())
        digest.update(str(series.dtype).encode())
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            digest.update(np.ascontiguousarray(series.to_numpy()).tobytes())
        else:
            for value in series.astype(str):
                encoded = value.encode()
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
    return digest.hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    return mapping_sha256([(str(column), str(frame[column].dtype)) for column in frame])


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    checked: list[str] = []
    roots = [Path.cwd(), project_root(), *candidate_package_dirs()]
    for value in candidates:
        candidate = Path(str(value))
        options = [candidate / filename] if candidate.suffix == "" else [candidate]
        if not candidate.is_absolute():
            options.extend(root / candidate / filename for root in roots)
            options.extend(root / candidate for root in roots if candidate.name == filename)
        for option in options:
            checked.append(str(option))
            if option.is_file():
                return option
    if KAGGLE_INPUT_ROOT.exists():
        for option in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if option.is_file():
                return option
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "platform": platform.platform(),
        "processor": platform.processor(),
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


def _input_spec(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.{key}")
    if not isinstance(value, Mapping):
        raise ValueError(f"data.{key} must be a mapping")
    return dict(value)


def _stage_artifact_file(root: Path, filename: str) -> Path:
    direct = root / filename
    nested = root / "artifacts" / filename
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"{filename} not found below {root}")


# %% [markdown]
# ## 3. Frozen scientific contract and input preflight


# %%
def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "primary_control": PRIMARY_CONTROL,
        "primary_candidate": PRIMARY_CANDIDATE,
        "pf": get_nested(config, "model.pf"),
        "evidence": get_nested(config, "model.evidence"),
        "execution_counts": get_nested(config, "model.execution_count"),
        "validation": {
            "expected_rows": get_nested(config, "validation.expected_rows"),
            "expected_wells": get_nested(config, "validation.expected_wells"),
            "expected_folds": get_nested(config, "validation.expected_folds"),
            "truth_attachment": get_nested(config, "validation.truth_attachment"),
            "scope_contract": get_nested(config, "validation.scope_contract"),
        },
        "forbidden": get_nested(config, "guards.forbidden"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def selected_stage(config: Mapping[str, Any]) -> str | None:
    value = get_nested(config, "execution.selected_stage")
    if value in (None, "", "none"):
        return None
    stage = str(value)
    if stage not in {"preflight", "full_shard", "merge"}:
        raise ValueError(f"unsupported exp430 stage: {stage}")
    return stage


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    fixed = {
        "experiment.route": "pf_beam",
        "lineage.parent": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "model.typewell_gr_emission.multiplier": 1.0,
        "model.typewell_gr_emission.z2_clip": 600.0,
        "model.pf.particles": 500,
        "model.pf.seeds": 128,
        "model.pf.trajectory_storage_dtype": "float64",
        "model.pf.filtering_particle_likelihood": "exp072_gaussian_unchanged",
        "model.evidence.temperature": 5.0,
        "model.evidence.huber_delta_1p345.delta": 1.345,
        "model.evidence.huber_delta_1p345.extra_clip": None,
        "model.evidence.require_shared_trajectory_sha": True,
        "model.execution_count.technical_preflight_pf_well_runs": 4,
        "model.execution_count.technical_preflight_seed_well_trajectories": 512,
        "model.execution_count.technical_preflight_particle_starts": 256000,
        "model.execution_count.full_pf_well_runs": 773,
        "model.execution_count.full_seed_well_trajectories": 98944,
        "model.execution_count.full_particle_starts": 49472000,
        "model.execution_count.full_shards": 4,
        "model.execution_count.parent_independent_full_reruns": 0,
        "model.execution_count.lightgbm_configs": 0,
        "model.execution_count.trained_folds": 0,
        "model.execution_count.boosters": 0,
        "model.execution_count.models": 0,
        "model.execution_count.gpu_runs": 0,
        "guards.technical.parent_parity_storage_dtype": "float32",
        "guards.technical.parent_parity_csv_reload_normalization": True,
        "runtime.device": "cpu",
        "runtime.use_amp": False,
    }
    mismatches = {
        key: {"actual": get_nested(config, key), "expected": expected}
        for key, expected in fixed.items()
        if get_nested(config, key) != expected
    }
    active = get_nested(config, "model.active_scientific_variants")
    if active != [PRIMARY_CANDIDATE]:
        mismatches["model.active_scientific_variants"] = {
            "actual": active,
            "expected": [PRIMARY_CANDIDATE],
        }
    if mismatches:
        raise ValueError(f"exp430 scientific contract mismatch: {mismatches}")
    if require_run_approval:
        stage = selected_stage(config)
        if stage is None:
            raise RuntimeError("exp430 stage is not selected")
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp430 Kaggle push is not approved")
        if not bool(get_nested(config, "execution.stage_run_approved")):
            raise RuntimeError("exp430 selected stage run is not approved")
        if stage in {"full_shard", "merge"} and not bool(
            get_nested(config, "execution.technical_preflight_approved")
        ):
            raise RuntimeError("exp430 technical preflight is not approved")
        if stage == "full_shard":
            shard = get_nested(config, "execution.shard_index")
            if not isinstance(shard, int) or not 0 <= shard < 4:
                raise RuntimeError("full_shard requires shard_index in 0..3")
        if stage == "merge":
            shard_spec = _input_spec(config, "full_shards")
            candidates = list(shard_spec.get("candidates", []))
            summary_shas = [
                str(value)
                for value in shard_spec.get("expected_summary_sha256", [])
            ]
            if len(candidates) != 4 or len(summary_shas) != 4 or any(
                len(value) != 64 for value in summary_shas
            ):
                raise RuntimeError(
                    "merge requires four pinned shard roots and four summary SHAs"
                )
    return build_scientific_contract(config)


def preflight_saved_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for key in (
        "exp072_control",
        "exp404_saved_prediction",
        "fold_assignment",
        "hidden_like_assignment",
    ):
        spec = _input_spec(config, key)
        path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
        paths[key] = str(path)
        if str(spec["filename"]).endswith((".gz", ".gz.bin")):
            report = inspect_gzip_csv(path)
            expected = str(spec["expected_decompressed_sha256"])
            if report["decompressed_sha256"] != expected:
                raise ValueError(f"{key} decompressed SHA mismatch")
            if key == "exp404_saved_prediction" and report["raw_sha256"] != str(
                spec["expected_raw_sha256"]
            ):
                raise ValueError("exp404_saved_prediction raw SHA mismatch")
        else:
            report = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "raw_sha256": sha256_path(path),
            }
            if report["raw_sha256"] != str(spec["expected_sha256"]):
                raise ValueError(f"{key} SHA mismatch")
        reports[key] = report
    return {
        "paths": paths,
        "reports": reports,
        "all_input_sha_matches": True,
    }


def load_preflight_wells(config: Mapping[str, Any]) -> pd.DataFrame:
    spec = _input_spec(config, "preflight_wells")
    path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    if sha256_path(path) != str(spec["expected_sha256"]):
        raise ValueError("fixed preflight-well asset SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    if list(frame.columns) != ["well_id", "sha256_order"] or len(frame) != 4:
        raise ValueError("fixed preflight-well asset schema/count mismatch")
    expected = sorted(
        frame["well_id"].astype(str),
        key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
    )
    observed = (
        frame.sort_values("sha256_order", kind="mergesort")["well_id"]
        .astype(str)
        .tolist()
    )
    if observed != expected or frame["well_id"].duplicated().any():
        raise ValueError("fixed preflight-well order/identity mismatch")
    return frame.sort_values("sha256_order", kind="mergesort").reset_index(drop=True)


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

    def mark_frozen(self) -> None:
        if any(self.report()["before_freeze"].values()):
            raise RuntimeError("truth/reporting values were accessed before freeze")
        self.prediction_frozen = True

    def require_frozen(self) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("truth-late input requires frozen predictions")

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


# %% [markdown]
# ## 4. Truth-free exp404-compatible PF input preparation


# %%
def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    frame = frame[["MD", "Z", "GR", "TVT_input"]]
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "Z"]].isna().any().any():
        raise ValueError(f"{well}: MD/Z must be finite")
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
    if len(frame) < 2:
        raise ValueError(f"{well}: invalid Type Well support")
    mean_gr = float(frame["GR"].mean())
    if not math.isfinite(mean_gr):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(mean_gr)
    return frame


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


def exp072_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires a known prefix")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    residual = known_gr - np.interp(known_tvt, typewell_tvt, typewell_gr)
    raw_scale = float(np.nanstd(residual))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    return {
        "raw_scale": raw_scale,
        "base_scale": float(np.clip(raw_scale, 10.0, 60.0)),
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
    config: Mapping[str, Any],
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires non-empty prefix and suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    grid_gr, grid_minimum, grid_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=float(get_nested(config, "model.pf.typewell_grid_step_ft")),
    )
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr.mean()))
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    scale = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
    return {
        "eval_indices": eval_indices,
        "eval_md": evaluation["MD"].to_numpy(np.float64),
        "eval_z": evaluation["Z"].to_numpy(np.float64),
        "eval_gr": interpolated_gr[eval_indices],
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": evaluation["MD"].to_numpy(np.float64)
        - float(last_known["MD"]),
        "last_known_tvt": float(last_known["TVT_input"]),
        "last_known_position": float(last_known["TVT_input"]) + float(last_known["Z"]),
        "initial_rate": exp072_initial_rate(horizontal),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": grid_step,
        "gr_scale": float(scale["base_scale"]),
        "scale_audit": scale,
    }


# %% [markdown]
# ## 5. Exact exp404 likelihood-PF trajectory kernel


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
    """Exact exp404 kernel; all scientific re-scoring happens after this call."""
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
        0.1,
        0.001,
        0.5,
        4.5,
    )


def decode_well_trajectory(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_likelihood_pf_inputs(horizontal, typewell, config)
    pf = get_nested(config, "model.pf") or {}
    seed_base = stable_seed("likpf", "train", well)
    (
        predictions,
        parent_log_likelihoods,
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
        float(prepared["gr_scale"]),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        int(pf["particles"]),
        int(pf["seeds"]),
        seed_base,
        float(pf["momentum"]),
        float(pf["rate_noise"]),
        float(pf["position_noise"]),
        float(pf["rough_position"]),
        float(pf["rough_rate"]),
        float(pf["resample_threshold_fraction"]),
        float(pf["initial_position_spread_ft"]),
    )
    if predictions.dtype != np.float64:
        raise ValueError("trajectory bank must remain float64 before freeze")
    eval_indices = prepared["eval_indices"]
    observed = prepared["raw_gr_observed"]
    identity = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices.astype(np.int64),
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": prepared["md_since"].astype(np.float64),
            "raw_gr_observed": observed,
            "well_missing_fraction": np.float64(1.0 - observed.mean()),
        }
    )
    audit = {
        "well_id": str(well),
        "status": "ok",
        "suffix_rows": len(identity),
        "seed_base": seed_base,
        "seed_first": seed_base,
        "seed_last": seed_base + int(pf["seeds"]) - 1,
        "seeds": int(pf["seeds"]),
        "particles": int(pf["particles"]),
        "pf_well_runs": 1,
        "seed_well_trajectories": int(pf["seeds"]),
        "particle_starts": int(pf["seeds"]) * int(pf["particles"]),
        "gr_scale": float(prepared["gr_scale"]),
        "resampling_count_total": int(resampling_counts.sum()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "position_clip_count_total": int(position_clip_counts.sum()),
        "trajectory_std_mean": float(predictions.std(axis=0).mean()),
        "wall_seconds": time.time() - started,
    }
    return identity, predictions, parent_log_likelihoods, audit


# %% [markdown]
# ## 6. Float64 trajectory-bank generation and freeze


# %%
def build_raw_well_manifest(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> pd.DataFrame:
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
                "rows": len(tvt_input),
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
        len(manifest) != int(get_nested(config, "validation.expected_wells"))
        or int(manifest["suffix_rows"].sum())
        != int(get_nested(config, "validation.expected_rows"))
    ):
        raise ValueError("raw train well manifest coverage mismatch")
    return manifest


def assign_lpt_shards(
    manifest: pd.DataFrame,
    shard_count: int,
) -> pd.DataFrame:
    ordered = manifest.sort_values(
        ["suffix_rows", "well_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    loads = [0] * shard_count
    assignments: dict[str, int] = {}
    for row in ordered.itertuples(index=False):
        shard = min(range(shard_count), key=lambda value: (loads[value], value))
        assignments[str(row.well_id)] = shard
        loads[shard] += int(row.suffix_rows)
    result = manifest.copy()
    result["shard_index"] = result["well_id"].map(assignments).astype(np.int8)
    return result


def trajectory_bank_logical_sha(
    bank_path: Path,
    trajectory_index: pd.DataFrame,
) -> str:
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    if bank.dtype != np.float64 or bank.ndim != 2:
        raise ValueError("trajectory bank must be a two-dimensional float64 array")
    digest = hashlib.sha256()
    digest.update(b"exp430_float64_trajectory_bank_v1")
    digest.update(canonical_json(list(bank.shape)).encode())
    digest.update(
        dataframe_content_sha(
            trajectory_index,
            ["well_id", "start", "stop", "suffix_rows", "seed_base"],
        ).encode()
    )
    column_chunk = max(1, (64 * 1024 * 1024) // (bank.shape[0] * bank.dtype.itemsize))
    for start in range(0, bank.shape[1], column_chunk):
        stop = min(bank.shape[1], start + column_chunk)
        digest.update(np.ascontiguousarray(bank[:, start:stop]).tobytes())
    return digest.hexdigest()


def _bounded_decode(
    wells: list[str],
    raw_dir: Path,
    config: Mapping[str, Any],
    workers: int,
) -> Iterable[
    tuple[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]]
]:
    iterator = iter(wells)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Any, str] = {}
        for _ in range(workers):
            try:
                well = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(decode_well_trajectory, well, raw_dir, config)] = well
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                well = pending.pop(future)
                yield well, future.result()
                try:
                    next_well = next(iterator)
                except StopIteration:
                    continue
                pending[
                    executor.submit(
                        decode_well_trajectory,
                        next_well,
                        raw_dir,
                        config,
                    )
                ] = next_well


def generate_trajectory_bank(
    raw_dir: Path,
    artifacts: Path,
    config: Mapping[str, Any],
    selected_manifest: pd.DataFrame,
    *,
    tag: str,
) -> dict[str, Any]:
    if selected_manifest.empty:
        raise ValueError("trajectory stage requires at least one well")
    selected = (
        selected_manifest.sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    selected["start"] = selected["suffix_rows"].cumsum().shift(fill_value=0).astype(np.int64)
    selected["stop"] = (selected["start"] + selected["suffix_rows"]).astype(np.int64)
    total_rows = int(selected["suffix_rows"].sum())
    seeds = int(get_nested(config, "model.pf.seeds"))
    bank_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_trajectory_bank.npy"
    index_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_trajectory_index.csv"
    identity_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_row_identity.csv.gz"
    parent_score_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_parent_seed_scores.csv.gz"
    pf_audit_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_pf_audit.csv"
    trajectory_manifest_path = (
        artifacts / f"{OUTPUT_PREFIX}_{tag}_trajectory_manifest.json"
    )
    bank = np.lib.format.open_memmap(
        bank_path,
        mode="w+",
        dtype=np.float64,
        shape=(seeds, total_rows),
    )
    index_lookup = selected.set_index("well_id")
    identity_frames: list[pd.DataFrame] = []
    parent_score_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    execution_order = (
        selected.sort_values(
            ["suffix_rows", "well_id"],
            ascending=[False, True],
            kind="mergesort",
        )["well_id"]
        .astype(str)
        .tolist()
    )
    warm_up_pf_kernel()
    workers = int(get_nested(config, "runtime.num_workers"))
    for well, (identity, trajectories, parent_scores, audit) in _bounded_decode(
        execution_order,
        raw_dir,
        config,
        workers,
    ):
        row = index_lookup.loc[well]
        start = int(row["start"])
        stop = int(row["stop"])
        if trajectories.shape != (seeds, stop - start):
            raise ValueError(f"{well}: trajectory shape mismatch")
        if identity["well_id"].nunique() != 1 or str(identity["well_id"].iloc[0]) != well:
            raise ValueError(f"{well}: row identity mismatch")
        bank[:, start:stop] = trajectories
        identity_frames.append(identity)
        parent_score_frames.append(
            pd.DataFrame(
                {
                    "well_id": well,
                    "seed_index": np.arange(seeds, dtype=np.int16),
                    "seed_value": int(audit["seed_base"])
                    + np.arange(seeds, dtype=np.int64),
                    "parent_gaussian_marginal_score": parent_scores.astype(np.float64),
                }
            )
        )
        audit_rows.append(audit)
    bank.flush()
    del bank
    identity = (
        pd.concat(identity_frames, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    parent_scores = (
        pd.concat(parent_score_frames, ignore_index=True)
        .sort_values(["well_id", "seed_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    pf_audit = (
        pd.DataFrame(audit_rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    selected["seed_base"] = selected["well_id"].map(
        lambda well: stable_seed("likpf", "train", str(well))
    )
    if (
        len(identity) != total_rows
        or identity["well_id"].nunique() != len(selected)
        or identity.duplicated(["well_id", "row_idx"]).any()
        or identity["id"].duplicated().any()
        or len(parent_scores) != len(selected) * seeds
        or len(pf_audit) != len(selected)
        or not pf_audit["status"].eq("ok").all()
    ):
        raise ValueError("trajectory-bank coverage or identity mismatch")
    selected[
        [
            "well_id",
            "rows",
            "prefix_rows",
            "suffix_rows",
            "start",
            "stop",
            "seed_base",
        ]
    ].to_csv(index_path, index=False)
    write_deterministic_gzip_csv(identity, identity_path)
    write_deterministic_gzip_csv(parent_scores, parent_score_path)
    pf_audit.to_csv(pf_audit_path, index=False)
    index_frame = pd.read_csv(index_path, dtype={"well_id": str})
    logical_sha = trajectory_bank_logical_sha(bank_path, index_frame)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "tag": tag,
        "frozen_before_evidence_readout": True,
        "dtype": "float64",
        "shape": [seeds, total_rows],
        "rows": total_rows,
        "wells": len(selected),
        "seeds": seeds,
        "pf_well_runs": int(pf_audit["pf_well_runs"].sum()),
        "seed_well_trajectories": int(pf_audit["seed_well_trajectories"].sum()),
        "particle_starts": int(pf_audit["particle_starts"].sum()),
        "trajectory_bank_raw_sha256": sha256_path(bank_path),
        "trajectory_bank_logical_sha256": logical_sha,
        "trajectory_index_sha256": sha256_path(index_path),
        "row_identity_raw_sha256": sha256_path(identity_path),
        "row_identity_decompressed_sha256": inspect_gzip_csv(identity_path)[
            "decompressed_sha256"
        ],
        "parent_seed_scores_decompressed_sha256": inspect_gzip_csv(parent_score_path)[
            "decompressed_sha256"
        ],
        "pf_audit_sha256": sha256_path(pf_audit_path),
        "seed_manifest_sha256": dataframe_content_sha(
            parent_scores,
            ["well_id", "seed_index", "seed_value"],
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(trajectory_manifest_path, manifest)
    return {
        "bank_path": bank_path,
        "index_path": index_path,
        "identity_path": identity_path,
        "parent_score_path": parent_score_path,
        "pf_audit_path": pf_audit_path,
        "trajectory_manifest_path": trajectory_manifest_path,
        "manifest": manifest,
    }


# %% [markdown]
# ## 7. Gaussian and Huber evidence readouts


# %%
def _interp_grid_many(
    grid: np.ndarray,
    values: np.ndarray,
    minimum: float,
    step: float,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    coordinate = (flat - minimum) / step
    index = np.trunc(coordinate).astype(np.int64)
    output = np.empty_like(flat)
    low = index < 0
    high = index >= len(grid) - 1
    middle = ~(low | high)
    output[low] = grid[0]
    output[high] = grid[-1]
    middle_index = index[middle]
    fraction = coordinate[middle] - middle_index
    output[middle] = (
        grid[middle_index] * (1.0 - fraction)
        + grid[middle_index + 1] * fraction
    )
    return output.reshape(values.shape)


def trajectory_residual_scores(
    trajectories: np.ndarray,
    prepared: Mapping[str, Any],
    *,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    expected_gr = _interp_grid_many(
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        np.asarray(trajectories, dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
    )
    zscore = (
        np.asarray(prepared["eval_gr"], dtype=np.float64)[None, :] - expected_gr
    ) / float(prepared["gr_scale"])
    squared = zscore * zscore
    gaussian = (-0.5 * np.minimum(squared, 600.0)).sum(axis=1, dtype=np.float64)
    absolute = np.abs(zscore)
    huber_loss = np.where(
        absolute <= delta,
        0.5 * squared,
        delta * absolute - 0.5 * delta * delta,
    )
    huber = (-huber_loss).sum(axis=1, dtype=np.float64)
    if not np.isfinite(gaussian).all() or not np.isfinite(huber).all():
        raise ValueError("trajectory evidence contains non-finite values")
    return gaussian, huber


def centered_softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    weights = np.exp((values - float(values.max())) / temperature)
    weights /= weights.sum(dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("seed weights contain non-finite values")
    return weights


def aggregate_frozen_trajectories(
    trajectories: np.ndarray,
    gaussian_scores: np.ndarray,
    huber_scores: np.ndarray,
    parent_scores: np.ndarray,
    *,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    weights = {
        PRIMARY_CONTROL: centered_softmax_weights(gaussian_scores, temperature),
        PRIMARY_CANDIDATE: centered_softmax_weights(huber_scores, temperature),
        PARENT_REPLAY: centered_softmax_weights(parent_scores, temperature),
    }
    predictions = {
        name: (value[:, None] * trajectories).sum(axis=0, dtype=np.float64)
        for name, value in weights.items()
    }
    predictions[ARITHMETIC_CONTROL] = trajectories.mean(axis=0, dtype=np.float64)
    return predictions, weights


def _trajectory_well_sha(
    well: str,
    row_idx: np.ndarray,
    trajectories: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(well).encode())
    digest.update(np.ascontiguousarray(row_idx, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(trajectories, dtype=np.float64).tobytes())
    return digest.hexdigest()


def score_one_frozen_well(
    well: str,
    bank: np.ndarray,
    index_row: Mapping[str, Any],
    identity: pd.DataFrame,
    parent_scores: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    bank_logical_sha: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start = int(index_row["start"])
    stop = int(index_row["stop"])
    trajectories = np.asarray(bank[:, start:stop], dtype=np.float64)
    rows = identity.iloc[start:stop].copy()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    expected_eval_indices = np.flatnonzero(
        horizontal["TVT_input"].isna().to_numpy()
    )
    if (
        not rows["well_id"].eq(well).all()
        or len(rows) != trajectories.shape[1]
        or not np.array_equal(
            rows["row_idx"].to_numpy(np.int64),
            expected_eval_indices,
        )
    ):
        raise ValueError(f"{well}: frozen trajectory row identity mismatch")
    typewell = load_typewell(well, raw_dir)
    prepared = prepare_likelihood_pf_inputs(horizontal, typewell, config)
    gaussian_scores, huber_scores = trajectory_residual_scores(
        trajectories,
        prepared,
        delta=float(get_nested(config, "model.evidence.huber_delta_1p345.delta")),
    )
    selected_parent = (
        parent_scores.loc[parent_scores["well_id"].eq(well)]
        .sort_values("seed_index", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(selected_parent) != trajectories.shape[0]:
        raise ValueError(f"{well}: parent seed-score coverage mismatch")
    parent_score_array = selected_parent[
        "parent_gaussian_marginal_score"
    ].to_numpy(np.float64)
    predictions, weights = aggregate_frozen_trajectories(
        trajectories,
        gaussian_scores,
        huber_scores,
        parent_score_array,
        temperature=float(get_nested(config, "model.evidence.temperature")),
    )
    well_sha = _trajectory_well_sha(
        well,
        rows["row_idx"].to_numpy(np.int64),
        trajectories,
    )
    for column, values in predictions.items():
        rows[column] = values.astype(np.float32)
    evidence = selected_parent.copy()
    evidence["trajectory_bank_logical_sha256"] = bank_logical_sha
    evidence["trajectory_well_logical_sha256"] = well_sha
    evidence["gaussian_score"] = gaussian_scores
    evidence["huber_score"] = huber_scores
    evidence["gaussian_weight"] = weights[PRIMARY_CONTROL]
    evidence["huber_weight"] = weights[PRIMARY_CANDIDATE]
    evidence["parent_gaussian_marginal_weight"] = weights[PARENT_REPLAY]
    control = predictions[ARITHMETIC_CONTROL]
    second_difference = np.diff(control, n=2)
    roughness = (
        float(np.sqrt(np.mean(second_difference * second_difference)))
        if len(second_difference)
        else 0.0
    )
    audit = {
        "well_id": well,
        "trajectory_bank_logical_sha256": bank_logical_sha,
        "trajectory_well_logical_sha256": well_sha,
        "suffix_rows": len(rows),
        "gaussian_weight_sum": float(weights[PRIMARY_CONTROL].sum()),
        "huber_weight_sum": float(weights[PRIMARY_CANDIDATE].sum()),
        "parent_gaussian_marginal_weight_sum": float(weights[PARENT_REPLAY].sum()),
        "gaussian_weight_ess": float(
            1.0 / np.sum(weights[PRIMARY_CONTROL] ** 2)
        ),
        "huber_weight_ess": float(
            1.0 / np.sum(weights[PRIMARY_CANDIDATE] ** 2)
        ),
        "parent_gaussian_marginal_weight_ess": float(
            1.0 / np.sum(weights[PARENT_REPLAY] ** 2)
        ),
        "gaussian_best_seed_index": int(np.argmax(gaussian_scores)),
        "huber_best_seed_index": int(np.argmax(huber_scores)),
        "control_roughness_rms_second_difference": roughness,
    }
    rows["control_roughness_rms_second_difference"] = roughness
    return rows, evidence, audit


def score_frozen_bank(
    bank_bundle: Mapping[str, Any],
    raw_dir: Path,
    artifacts: Path,
    config: Mapping[str, Any],
    *,
    tag: str,
    workers: int,
    write_outputs: bool = True,
) -> dict[str, Any]:
    manifest = dict(bank_bundle["manifest"])
    if not bool(manifest["frozen_before_evidence_readout"]):
        raise RuntimeError("evidence readout requires a frozen trajectory bank")
    bank_path = Path(bank_bundle["bank_path"])
    index = pd.read_csv(bank_bundle["index_path"], dtype={"well_id": str})
    identity = pd.read_csv(
        bank_bundle["identity_path"],
        dtype={"id": str, "well_id": str},
        compression="gzip",
    )
    identity = identity.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    parent_scores = pd.read_csv(
        bank_bundle["parent_score_path"],
        dtype={"well_id": str},
        compression="gzip",
    )
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    if (
        bank.dtype != np.float64
        or list(bank.shape) != list(manifest["shape"])
        or trajectory_bank_logical_sha(bank_path, index)
        != str(manifest["trajectory_bank_logical_sha256"])
    ):
        raise ValueError("frozen trajectory bank identity changed before scoring")
    index_rows = {
        str(row.well_id): row._asdict()
        for row in index.itertuples(index=False)
    }

    def run(well: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        return score_one_frozen_well(
            well,
            bank,
            index_rows[well],
            identity,
            parent_scores,
            raw_dir,
            config,
            str(manifest["trajectory_bank_logical_sha256"]),
        )

    wells = sorted(index_rows)
    if workers == 1:
        results = [run(well) for well in wells]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run, wells))
    predictions = (
        pd.concat([value[0] for value in results], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    evidence = (
        pd.concat([value[1] for value in results], ignore_index=True)
        .sort_values(["well_id", "seed_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([value[2] for value in results])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    prediction_columns = ["id", "well_id", "row_idx", *PREDICTION_COLUMNS]
    evidence_columns = [
        "well_id",
        "seed_index",
        "trajectory_bank_logical_sha256",
        "trajectory_well_logical_sha256",
        "gaussian_score",
        "huber_score",
        "gaussian_weight",
        "huber_weight",
        "parent_gaussian_marginal_score",
        "parent_gaussian_marginal_weight",
    ]
    summary = {
        "tag": tag,
        "workers": workers,
        "rows": len(predictions),
        "wells": int(predictions["well_id"].nunique()),
        "trajectory_bank_logical_sha256": manifest[
            "trajectory_bank_logical_sha256"
        ],
        "prediction_logical_sha256": dataframe_content_sha(
            predictions,
            prediction_columns,
        ),
        "evidence_logical_sha256": dataframe_content_sha(
            evidence,
            evidence_columns,
        ),
        "maximum_weight_sum_absolute_error": float(
            np.max(
                np.abs(
                    audit[
                        [
                            "gaussian_weight_sum",
                            "huber_weight_sum",
                            "parent_gaussian_marginal_weight_sum",
                        ]
                    ].to_numpy(np.float64)
                    - 1.0
                )
            )
        ),
    }
    paths: dict[str, Path] = {}
    if write_outputs:
        prediction_path = (
            artifacts / f"{OUTPUT_PREFIX}_{tag}_aggregated_predictions.csv.gz"
        )
        evidence_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_seed_evidence.csv.gz"
        audit_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_well_evidence_audit.csv"
        write_deterministic_gzip_csv(predictions, prediction_path)
        write_deterministic_gzip_csv(evidence, evidence_path)
        audit.to_csv(audit_path, index=False)
        paths = {
            "aggregated_predictions": prediction_path,
            "seed_evidence": evidence_path,
            "well_evidence_audit": audit_path,
        }
        summary["prediction_raw_sha256"] = sha256_path(prediction_path)
        summary["prediction_decompressed_sha256"] = inspect_gzip_csv(
            prediction_path
        )["decompressed_sha256"]
        summary["evidence_raw_sha256"] = sha256_path(evidence_path)
        summary["evidence_decompressed_sha256"] = inspect_gzip_csv(evidence_path)[
            "decompressed_sha256"
        ]
    return {
        "predictions": predictions,
        "evidence": evidence,
        "audit": audit,
        "summary": summary,
        "paths": paths,
    }


# %% [markdown]
# ## 8. Technical preflight and full-shard execution


# %%
def _align_reference(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if reference["id"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = reference.assign(id=reference["id"].astype(str)).set_index("id").reindex(
        frame["id"].astype(str)
    )
    if aligned[columns].isna().any().any():
        raise ValueError(f"{label} is missing aligned rows")
    output = frame.copy()
    for column in columns:
        output[column] = aligned[column].to_numpy()
    return output


def float32_storage_values(values: pd.Series | np.ndarray) -> np.ndarray:
    """Recover the binary float32 values represented by a saved parent CSV."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def input_manifest_frame(preflight: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, report in preflight["reports"].items():
        rows.append(
            {
                "name": name,
                "path": report.get("path"),
                "bytes": report.get("bytes"),
                "raw_sha256": report.get("raw_sha256"),
                "decompressed_sha256": report.get("decompressed_sha256"),
                "data_rows": report.get("data_rows"),
                "columns": json.dumps(
                    report.get("columns"),
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def parent_replay_parity(
    predictions: pd.DataFrame,
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    parent_spec = _input_spec(config, "exp404_saved_prediction")
    parent_prediction_column = str(parent_spec["prediction_column"])
    parent_arithmetic_column = str(parent_spec["arithmetic_column"])
    parent = pd.read_csv(
        preflight["paths"]["exp404_saved_prediction"],
        usecols=["id", parent_prediction_column, parent_arithmetic_column],
        dtype={"id": str},
        compression="gzip",
    ).rename(
        columns={
            parent_prediction_column: SAVED_PARENT,
            parent_arithmetic_column: "saved_exp404_arithmetic_mean",
        }
    )
    frame = _align_reference(
        predictions,
        parent,
        [SAVED_PARENT, "saved_exp404_arithmetic_mean"],
        label="saved exp404 temperature-5 prediction",
    )
    parent_replay_values = frame[PARENT_REPLAY].to_numpy(np.float64)
    saved_parent_values = frame[SAVED_PARENT].to_numpy(np.float64)
    parent_difference_before_storage_normalization = np.abs(
        parent_replay_values - saved_parent_values
    )
    parent_difference = np.abs(
        float32_storage_values(parent_replay_values)
        - float32_storage_values(saved_parent_values)
    )
    exp072_spec = _input_spec(config, "exp072_control")
    anchor_column = str(exp072_spec["anchor_column"])
    delta_column = str(exp072_spec["delta_column"])
    exp072 = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=["id", anchor_column, delta_column],
        dtype={"id": str},
    )
    exp072["saved_exp072_arithmetic_mean"] = (
        pd.to_numeric(exp072[anchor_column], errors="raise")
        + pd.to_numeric(exp072[delta_column], errors="raise")
    )
    frame = _align_reference(
        frame,
        exp072[["id", "saved_exp072_arithmetic_mean"]],
        ["saved_exp072_arithmetic_mean"],
        label="saved exp072 arithmetic mean",
    )
    arithmetic_replay_values = frame[ARITHMETIC_CONTROL].to_numpy(np.float64)
    saved_arithmetic_values = frame[
        "saved_exp404_arithmetic_mean"
    ].to_numpy(np.float64)
    arithmetic_difference_before_storage_normalization = np.abs(
        arithmetic_replay_values - saved_arithmetic_values
    )
    arithmetic_difference = np.abs(
        float32_storage_values(arithmetic_replay_values)
        - float32_storage_values(saved_arithmetic_values)
    )
    exp072_representation_difference = np.abs(
        frame["saved_exp404_arithmetic_mean"].to_numpy(np.float64)
        - frame["saved_exp072_arithmetic_mean"].to_numpy(np.float64)
    )
    return {
        "parent_marginal_replay_max_abs_ft": float(parent_difference.max()),
        "parent_marginal_replay_mean_abs_ft": float(parent_difference.mean()),
        "parent_marginal_replay_pre_normalization_max_abs_ft": float(
            parent_difference_before_storage_normalization.max()
        ),
        "arithmetic_mean_replay_max_abs_ft": float(arithmetic_difference.max()),
        "arithmetic_mean_replay_mean_abs_ft": float(arithmetic_difference.mean()),
        "arithmetic_mean_replay_pre_normalization_max_abs_ft": float(
            arithmetic_difference_before_storage_normalization.max()
        ),
        "saved_exp404_vs_exp072_arithmetic_representation_max_abs_ft": float(
            exp072_representation_difference.max()
        ),
        "saved_exp404_vs_exp072_arithmetic_representation_mean_abs_ft": float(
            exp072_representation_difference.mean()
        ),
    }


def run_preflight_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest = build_raw_well_manifest(config, raw_dir)
    fixed = load_preflight_wells(config)
    expected_sha_first = (
        raw_manifest.assign(
            sha=lambda frame: frame["well_id"].map(
                lambda value: hashlib.sha256(str(value).encode()).hexdigest()
            )
        )
        .sort_values(["sha", "well_id"], kind="mergesort")
        .head(4)["well_id"]
        .astype(str)
        .tolist()
    )
    if fixed["well_id"].astype(str).tolist() != expected_sha_first:
        raise ValueError("preflight well asset is not the fixed SHA-first raw-train set")
    preflight = preflight_saved_inputs(config)
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_preflight_input_manifest.csv"
    raw_manifest_path = artifacts / f"{OUTPUT_PREFIX}_raw_well_manifest.csv"
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, contract)
    input_manifest_frame(preflight).to_csv(input_manifest_path, index=False)
    raw_manifest.to_csv(raw_manifest_path, index=False)
    selected = raw_manifest.loc[
        raw_manifest["well_id"].isin(fixed["well_id"])
    ].copy()
    bank_bundle = generate_trajectory_bank(
        raw_dir,
        artifacts,
        config,
        selected,
        tag="preflight",
    )
    parallel = score_frozen_bank(
        bank_bundle,
        raw_dir,
        artifacts,
        config,
        tag="preflight",
        workers=int(get_nested(config, "runtime.num_workers")),
        write_outputs=True,
    )
    serial = score_frozen_bank(
        bank_bundle,
        raw_dir,
        artifacts,
        config,
        tag="preflight_serial_parity",
        workers=1,
        write_outputs=False,
    )
    parity = parent_replay_parity(
        parallel["predictions"],
        preflight,
        config,
    )
    technical_config = get_nested(config, "guards.technical") or {}
    expected = get_nested(config, "model.execution_count") or {}
    manifest = bank_bundle["manifest"]
    common_sha_values = parallel["evidence"][
        "trajectory_bank_logical_sha256"
    ].astype(str).unique()
    checks = {
        "fixed_well_count": int(manifest["wells"])
        == int(expected["technical_preflight_wells"]),
        "pf_well_runs": int(manifest["pf_well_runs"])
        == int(expected["technical_preflight_pf_well_runs"]),
        "seed_well_trajectories": int(manifest["seed_well_trajectories"])
        == int(expected["technical_preflight_seed_well_trajectories"]),
        "particle_starts": int(manifest["particle_starts"])
        == int(expected["technical_preflight_particle_starts"]),
        "float64_trajectory_bank": str(manifest["dtype"]) == "float64",
        "shared_trajectory_bank_sha": bool(
            len(common_sha_values) == 1
            and common_sha_values[0]
            == str(manifest["trajectory_bank_logical_sha256"])
        ),
        "partition_prediction_sha_parity": parallel["summary"][
            "prediction_logical_sha256"
        ]
        == serial["summary"]["prediction_logical_sha256"],
        "partition_evidence_sha_parity": parallel["summary"][
            "evidence_logical_sha256"
        ]
        == serial["summary"]["evidence_logical_sha256"],
        "weight_sum": float(
            parallel["summary"]["maximum_weight_sum_absolute_error"]
        )
        <= float(technical_config["require_weight_sum_atol"]),
        "parent_kernel_prediction_parity": float(
            parity["parent_marginal_replay_max_abs_ft"]
        )
        <= float(
            technical_config["require_parent_kernel_prediction_parity_atol_ft"]
        ),
        "arithmetic_mean_parity": float(
            parity["arithmetic_mean_replay_max_abs_ft"]
        )
        <= float(
            technical_config[
                "require_arithmetic_mean_parity_vs_saved_exp404_atol_ft"
            ]
        ),
        "truth_late": True,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "preflight",
        "status": "technical_preflight_passed" if all(checks.values()) else "failed",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "fixed_wells": fixed["well_id"].astype(str).tolist(),
        "execution_counts": {
            "pf_well_runs": int(manifest["pf_well_runs"]),
            "seed_well_trajectories": int(manifest["seed_well_trajectories"]),
            "particle_starts": int(manifest["particle_starts"]),
            "scientific_variants": 1,
            "readouts_from_same_bank": 2,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "models": 0,
            "gpu_runs": 0,
        },
        "trajectory_manifest": manifest,
        "parallel_scoring": parallel["summary"],
        "serial_scoring": serial["summary"],
        "saved_parent_parity": parity,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "raw_well_manifest_sha256": sha256_path(raw_manifest_path),
        "runtime_seconds": time.time() - started,
        "runtime_versions": runtime_versions(),
        "truth_access": {
            "suffix_tvt_rows": 0,
            "fold_rows": 0,
            "hidden_like_role_rows": 0,
            "error_rows": 0,
        },
        "promotion_evidence": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_preflight_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise RuntimeError("exp430 technical preflight failed; full replay is blocked")
    return summary


def load_approved_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = _input_spec(config, "preflight_result")
    expected = str(spec["expected_sha256"])
    if len(expected) != 64:
        raise RuntimeError("approved preflight summary SHA is not pinned")
    path = resolve_existing(str(spec["filename"]), spec.get("candidates", []))
    if sha256_path(path) != expected:
        raise ValueError("approved preflight summary raw SHA mismatch")
    payload = json.loads(path.read_text())
    if (
        not bool(payload.get("passed"))
        or payload.get("stage") != "preflight"
        or payload.get("experiment") != EXPERIMENT_NAME
    ):
        raise ValueError("approved preflight summary did not pass")
    return payload


def run_full_shard_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_scientific_contract(config, require_run_approval=True)
    preflight = load_approved_preflight(config)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest = build_raw_well_manifest(config, raw_dir)
    shard_count = int(get_nested(config, "model.execution_count.full_shards"))
    assigned = assign_lpt_shards(raw_manifest, shard_count)
    shard_index = int(get_nested(config, "execution.shard_index"))
    selected = assigned.loc[assigned["shard_index"].eq(shard_index)].copy()
    if selected.empty:
        raise ValueError(f"full shard {shard_index} has no wells")
    tag = f"full_shard{shard_index}"
    shard_manifest_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_raw_well_manifest.csv"
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, contract)
    selected.to_csv(shard_manifest_path, index=False)
    bank_bundle = generate_trajectory_bank(
        raw_dir,
        artifacts,
        config,
        selected,
        tag=tag,
    )
    scored = score_frozen_bank(
        bank_bundle,
        raw_dir,
        artifacts,
        config,
        tag=tag,
        workers=int(get_nested(config, "runtime.num_workers")),
        write_outputs=True,
    )
    trajectory_manifest = bank_bundle["manifest"]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "full_shard",
        "status": "full_shard_frozen_truth_unread",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "rows": int(trajectory_manifest["rows"]),
        "wells": int(trajectory_manifest["wells"]),
        "pf_well_runs": int(trajectory_manifest["pf_well_runs"]),
        "seed_well_trajectories": int(
            trajectory_manifest["seed_well_trajectories"]
        ),
        "particle_starts": int(trajectory_manifest["particle_starts"]),
        "scientific_variants": 1,
        "readouts_from_same_bank": 2,
        "parent_independent_full_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "gpu_runs": 0,
        "trajectory_manifest": trajectory_manifest,
        "scoring_summary": scored["summary"],
        "raw_well_manifest_sha256": sha256_path(shard_manifest_path),
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "approved_preflight_summary_sha256": str(
            _input_spec(config, "preflight_result")["expected_sha256"]
        ),
        "approved_preflight_trajectory_sha256": preflight[
            "trajectory_manifest"
        ]["trajectory_bank_logical_sha256"],
        "truth_access": {
            "suffix_tvt_rows": 0,
            "fold_rows": 0,
            "hidden_like_role_rows": 0,
            "error_rows": 0,
        },
        "runtime_seconds": time.time() - started,
        "runtime_versions": runtime_versions(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_{tag}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Truth-late merge, metrics, and promotion gates


# %%
def resolve_full_shard_roots(config: Mapping[str, Any]) -> list[Path]:
    spec = _input_spec(config, "full_shards")
    candidates = [Path(str(value)) for value in spec.get("candidates", [])]
    expected_count = int(spec["expected_count"])
    resolved: list[Path] = []
    for candidate in candidates:
        options = [candidate]
        if not candidate.is_absolute():
            options.extend([Path.cwd() / candidate, project_root() / candidate])
        for option in options:
            if option.is_dir():
                resolved.append(option)
                break
    if len(resolved) != expected_count:
        raise RuntimeError(
            f"merge requires {expected_count} full-shard roots; resolved={resolved}"
        )
    return resolved


def load_and_freeze_full_shards(
    config: Mapping[str, Any],
    artifacts: Path,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    if ledger.prediction_frozen:
        raise RuntimeError("merge ledger is already frozen")
    roots = resolve_full_shard_roots(config)
    expected_summary_sha = [
        str(value)
        for value in _input_spec(config, "full_shards").get(
            "expected_summary_sha256",
            [],
        )
    ]
    if expected_summary_sha and len(expected_summary_sha) != 4:
        raise ValueError("full-shard expected summary SHA list must have four entries")
    prediction_frames: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    shard_rows: list[dict[str, Any]] = []
    trajectory_shas: dict[int, str] = {}
    summary_shas: dict[int, str] = {}
    for root in roots:
        found = False
        for shard_index in range(4):
            tag = f"full_shard{shard_index}"
            summary_name = f"{OUTPUT_PREFIX}_{tag}_summary.json"
            try:
                summary_path = _stage_artifact_file(root, summary_name)
            except FileNotFoundError:
                continue
            if found:
                raise ValueError(f"{root} contains multiple exp430 shard summaries")
            found = True
            if shard_index in summary_shas:
                raise ValueError(f"duplicate full shard index: {shard_index}")
            summary_sha = sha256_path(summary_path)
            if expected_summary_sha and summary_sha != expected_summary_sha[shard_index]:
                raise ValueError(f"full shard {shard_index} summary SHA mismatch")
            summary = json.loads(summary_path.read_text())
            if (
                summary.get("experiment") != EXPERIMENT_NAME
                or summary.get("stage") != "full_shard"
                or int(summary.get("shard_index", -1)) != shard_index
                or summary.get("scientific_contract_sha256")
                != build_scientific_contract(config)["scientific_contract_sha256"]
                or summary.get("approved_preflight_summary_sha256")
                != str(_input_spec(config, "preflight_result")["expected_sha256"])
                or any(int(value) != 0 for value in summary["truth_access"].values())
            ):
                raise ValueError(f"full shard {shard_index} summary contract mismatch")
            prediction_path = _stage_artifact_file(
                root,
                f"{OUTPUT_PREFIX}_{tag}_aggregated_predictions.csv.gz",
            )
            evidence_path = _stage_artifact_file(
                root,
                f"{OUTPUT_PREFIX}_{tag}_seed_evidence.csv.gz",
            )
            audit_path = _stage_artifact_file(
                root,
                f"{OUTPUT_PREFIX}_{tag}_well_evidence_audit.csv",
            )
            trajectory_manifest_path = _stage_artifact_file(
                root,
                f"{OUTPUT_PREFIX}_{tag}_trajectory_manifest.json",
            )
            bank_path = _stage_artifact_file(
                root,
                f"{OUTPUT_PREFIX}_{tag}_trajectory_bank.npy",
            )
            trajectory_manifest = json.loads(trajectory_manifest_path.read_text())
            if (
                sha256_path(bank_path)
                != trajectory_manifest["trajectory_bank_raw_sha256"]
                or trajectory_manifest["trajectory_bank_logical_sha256"]
                != summary["trajectory_manifest"][
                    "trajectory_bank_logical_sha256"
                ]
            ):
                raise ValueError(
                    f"full shard {shard_index} trajectory bank identity mismatch"
                )
            prediction_report = inspect_gzip_csv(prediction_path)
            evidence_report = inspect_gzip_csv(evidence_path)
            if (
                prediction_report["decompressed_sha256"]
                != summary["scoring_summary"]["prediction_decompressed_sha256"]
                or evidence_report["decompressed_sha256"]
                != summary["scoring_summary"]["evidence_decompressed_sha256"]
            ):
                raise ValueError(f"full shard {shard_index} scored output SHA mismatch")
            prediction = pd.read_csv(
                prediction_path,
                dtype={"id": str, "well_id": str},
                compression="gzip",
            )
            evidence = pd.read_csv(
                evidence_path,
                dtype={"well_id": str},
                compression="gzip",
            )
            audit = pd.read_csv(audit_path, dtype={"well_id": str})
            prediction_frames.append(prediction)
            evidence_frames.append(evidence)
            audit_frames.append(audit)
            summary_shas[shard_index] = summary_sha
            trajectory_shas[shard_index] = str(
                trajectory_manifest["trajectory_bank_logical_sha256"]
            )
            shard_rows.append(
                {
                    "shard_index": shard_index,
                    "root": str(root),
                    "summary_sha256": summary_sha,
                    "trajectory_bank_raw_sha256": trajectory_manifest[
                        "trajectory_bank_raw_sha256"
                    ],
                    "trajectory_bank_logical_sha256": trajectory_manifest[
                        "trajectory_bank_logical_sha256"
                    ],
                    "prediction_decompressed_sha256": prediction_report[
                        "decompressed_sha256"
                    ],
                    "evidence_decompressed_sha256": evidence_report[
                        "decompressed_sha256"
                    ],
                    "rows": len(prediction),
                    "wells": int(prediction["well_id"].nunique()),
                    "pf_well_runs": int(summary["pf_well_runs"]),
                    "seed_well_trajectories": int(
                        summary["seed_well_trajectories"]
                    ),
                    "particle_starts": int(summary["particle_starts"]),
                }
            )
        if not found:
            raise ValueError(f"{root} contains no exp430 full-shard summary")
    if sorted(summary_shas) != [0, 1, 2, 3]:
        raise ValueError("full-shard set is incomplete")
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    evidence = (
        pd.concat(evidence_frames, ignore_index=True)
        .sort_values(["well_id", "seed_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.concat(audit_frames, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    seeds = int(get_nested(config, "model.pf.seeds"))
    if (
        len(predictions) != expected_rows
        or predictions["well_id"].nunique() != expected_wells
        or predictions.duplicated(["well_id", "row_idx"]).any()
        or predictions["id"].duplicated().any()
        or len(evidence) != expected_wells * seeds
        or evidence.duplicated(["well_id", "seed_index"]).any()
        or len(audit) != expected_wells
        or audit["well_id"].duplicated().any()
    ):
        raise ValueError("combined full-shard identity/coverage mismatch")
    evidence_shared = (
        evidence.groupby("well_id", sort=True)[
            ["trajectory_bank_logical_sha256", "trajectory_well_logical_sha256"]
        ]
        .nunique()
        .eq(1)
        .all()
        .all()
    )
    if not evidence_shared:
        raise ValueError("Gaussian and Huber do not share one trajectory identity per well")
    roughness_quantile = float(
        get_nested(config, "validation.scope_contract.roughness_split_quantile")
    )
    well_roughness = (
        predictions[["well_id", "control_roughness_rms_second_difference"]]
        .drop_duplicates("well_id")
        .sort_values("well_id", kind="mergesort")
    )
    roughness_threshold = float(
        well_roughness["control_roughness_rms_second_difference"].quantile(
            roughness_quantile
        )
    )
    predictions["roughness_high"] = predictions[
        "control_roughness_rms_second_difference"
    ].ge(roughness_threshold)
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_combined_predictions.csv.gz"
    evidence_path = artifacts / f"{OUTPUT_PREFIX}_combined_seed_evidence.csv.gz"
    audit_path = artifacts / f"{OUTPUT_PREFIX}_combined_well_evidence_audit.csv"
    shard_manifest_path = artifacts / f"{OUTPUT_PREFIX}_combined_shard_manifest.csv"
    identity_path = artifacts / f"{OUTPUT_PREFIX}_combined_prediction_identity.json"
    write_deterministic_gzip_csv(predictions, prediction_path)
    write_deterministic_gzip_csv(evidence, evidence_path)
    audit.to_csv(audit_path, index=False)
    shard_manifest = pd.DataFrame(shard_rows).sort_values(
        "shard_index",
        kind="mergesort",
    )
    shard_manifest.to_csv(shard_manifest_path, index=False)
    global_trajectory_sha = mapping_sha256(
        {
            "policy": "ordered_four_float64_trajectory_shards",
            "shards": trajectory_shas,
        }
    )
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(predictions),
        "wells": int(predictions["well_id"].nunique()),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "prediction_logical_sha256": dataframe_content_sha(
            predictions,
            [
                "id",
                "well_id",
                "row_idx",
                *PREDICTION_COLUMNS,
                "control_roughness_rms_second_difference",
                "roughness_high",
            ],
        ),
        "prediction_schema_sha256": dataframe_schema_sha(predictions),
        "prediction_decompressed_sha256": inspect_gzip_csv(prediction_path)[
            "decompressed_sha256"
        ],
        "evidence_logical_sha256": dataframe_content_sha(
            evidence,
            [
                "well_id",
                "seed_index",
                "trajectory_bank_logical_sha256",
                "trajectory_well_logical_sha256",
                "gaussian_score",
                "huber_score",
                "gaussian_weight",
                "huber_weight",
            ],
        ),
        "evidence_decompressed_sha256": inspect_gzip_csv(evidence_path)[
            "decompressed_sha256"
        ],
        "global_trajectory_manifest_sha256": global_trajectory_sha,
        "roughness_threshold": roughness_threshold,
        "roughness_definition": get_nested(
            config,
            "validation.scope_contract.roughness_definition",
        ),
        "shard_summary_sha256": summary_shas,
        "shard_trajectory_logical_sha256": trajectory_shas,
        "shared_trajectory_identity_per_well": bool(evidence_shared),
        "actual_execution_counts": {
            "pf_well_runs": int(
                sum(int(row["pf_well_runs"]) for row in shard_rows)
            ),
            "seed_well_trajectories": int(
                sum(int(row["seed_well_trajectories"]) for row in shard_rows)
            ),
            "particle_starts": int(
                sum(int(row["particle_starts"]) for row in shard_rows)
            ),
            "shards": len(shard_rows),
        },
        "truth_or_reporting_values_parsed_before_freeze": ledger.report()[
            "before_freeze"
        ],
    }
    write_json(identity_path, frozen)
    ledger.mark_frozen()
    return (
        predictions,
        evidence,
        audit,
        frozen,
        {
            "combined_predictions": prediction_path,
            "combined_seed_evidence": evidence_path,
            "combined_well_evidence_audit": audit_path,
            "combined_shard_manifest": shard_manifest_path,
            "combined_prediction_identity": identity_path,
        },
    )


def load_truth_late_frame(
    prediction: pd.DataFrame,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger.require_frozen()
    prediction_sha = dataframe_content_sha(
        prediction,
        [
            "id",
            "well_id",
            "row_idx",
            *PREDICTION_COLUMNS,
            "control_roughness_rms_second_difference",
            "roughness_high",
        ],
    )
    if prediction_sha != str(frozen["prediction_logical_sha256"]):
        raise ValueError("combined prediction changed after freeze")
    fold_spec = _input_spec(config, "fold_assignment")
    safe = [str(value) for value in fold_spec["safe_columns"]]
    truth = [str(value) for value in fold_spec["truth_columns"]]
    if set(safe) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("fold safe-column allowlist mismatch")
    if set(safe) & set(str(value) for value in fold_spec["forbidden_decoder_columns"]):
        raise ValueError("fold safe columns contain forbidden decoder columns")
    fold = pd.read_csv(
        preflight["paths"]["fold_assignment"],
        usecols=list(dict.fromkeys([*safe, *truth])),
        dtype={"well_id": str},
        compression="gzip",
    )
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    fold["tvt_true"] = pd.to_numeric(
        fold["tvt_true"],
        errors="raise",
    ).astype(np.float64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("truth-late fold identity is duplicated")
    ledger.fold_rows_after_freeze += len(fold)
    ledger.unknown_suffix_tvt_rows_after_freeze += len(fold)
    frame = prediction.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_exp226"),
        sort=False,
    )
    if frame[["fold", "tvt_true", "suffix_offset_exp226"]].isna().any().any():
        raise ValueError("truth-late fold/TVT attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["suffix_offset_exp226"].to_numpy(np.int64),
    ):
        raise ValueError("truth-late suffix offset identity mismatch")
    frame = frame.drop(columns=["suffix_offset_exp226"]).rename(
        columns={"tvt_true": "true_tvt"}
    )
    parent_spec = _input_spec(config, "exp404_saved_prediction")
    parent = pd.read_csv(
        preflight["paths"]["exp404_saved_prediction"],
        usecols=["id", str(parent_spec["prediction_column"])],
        dtype={"id": str},
        compression="gzip",
    ).rename(columns={str(parent_spec["prediction_column"]): SAVED_PARENT})
    frame = _align_reference(
        frame,
        parent,
        [SAVED_PARENT],
        label="saved exp404 temperature-5 prediction",
    )
    exp072_spec = _input_spec(config, "exp072_control")
    anchor = str(exp072_spec["anchor_column"])
    delta = str(exp072_spec["delta_column"])
    exp072 = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=["id", anchor, delta],
        dtype={"id": str},
    )
    exp072["saved_exp072_arithmetic_mean"] = (
        pd.to_numeric(exp072[anchor], errors="raise")
        + pd.to_numeric(exp072[delta], errors="raise")
    )
    frame = _align_reference(
        frame,
        exp072[["id", "saved_exp072_arithmetic_mean"]],
        ["saved_exp072_arithmetic_mean"],
        label="saved exp072 arithmetic mean",
    )
    hidden_spec = _input_spec(config, "hidden_like_assignment")
    role_columns = {
        str(scope): str(column)
        for scope, column in hidden_spec["role_columns"].items()
    }
    hidden = pd.read_csv(
        preflight["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    if hidden["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    ledger.hidden_like_role_rows_after_freeze += len(hidden)
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
            for key, value in hidden_spec["expected_role_counts"][scope].items()
        }
        if actual != expected:
            raise ValueError(f"hidden-like role count mismatch: {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[
        role_columns["hidden_like_spatial"]
    ].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[
        role_columns["hidden_like_typewell_purged"]
    ].eq("valid")
    roughness_threshold = float(frozen["roughness_threshold"])
    observed_roughness = frame[
        "control_roughness_rms_second_difference"
    ].ge(roughness_threshold)
    if not np.array_equal(
        observed_roughness.to_numpy(bool),
        frame["roughness_high"].to_numpy(bool),
    ):
        raise ValueError("target-free roughness scope changed after prediction freeze")
    finite_columns = [
        "true_tvt",
        *PREDICTION_COLUMNS,
        SAVED_PARENT,
        "saved_exp072_arithmetic_mean",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("truth-late metric frame contains non-finite values")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    observed_folds = sorted(frame["fold"].astype(int).unique().tolist())
    if observed_folds != expected_folds:
        raise ValueError("truth-late reporting fold set mismatch")
    return frame, {
        "prediction_sha256_reverified": prediction_sha,
        "truth_attached_after_freeze": True,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": observed_folds,
        "roughness_threshold": roughness_threshold,
        "roughness_definition": frozen["roughness_definition"],
        "truth_access_ledger": ledger.report(),
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_values = np.asarray(truth, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction_values - truth_values) ** 2)))


def metric_scopes(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[tuple[str, np.ndarray]]:
    shallow = float(
        get_nested(config, "validation.scope_contract.shallow_md_since_lt_ft")
    )
    deep = float(get_nested(config, "validation.scope_contract.deep_md_since_ge_ft"))
    high_missing = float(
        get_nested(config, "validation.scope_contract.high_missing_fraction_ge")
    )
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool))
    ]
    for fold in sorted(frame["fold"].astype(int).unique().tolist()):
        scopes.append((f"fold_{fold}", frame["fold"].eq(fold).to_numpy()))
    scopes.extend(
        [
            ("shallow", frame["md_since"].lt(shallow).to_numpy()),
            ("deep", frame["md_since"].ge(deep).to_numpy()),
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
            (
                "missing_fraction_high",
                frame["well_missing_fraction"].ge(high_missing).to_numpy(),
            ),
            ("roughness_low", ~frame["roughness_high"].to_numpy(bool)),
            ("roughness_high", frame["roughness_high"].to_numpy(bool)),
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
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = {
        "huber_vs_gaussian_matched": (PRIMARY_CANDIDATE, PRIMARY_CONTROL),
        "huber_vs_saved_exp404_temperature5": (PRIMARY_CANDIDATE, SAVED_PARENT),
        "huber_vs_arithmetic_mean": (PRIMARY_CANDIDATE, ARITHMETIC_CONTROL),
    }
    paired_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    for comparison, (candidate_column, control_column) in comparisons.items():
        for scope, mask in metric_scopes(frame, config):
            if not bool(mask.any()):
                raise ValueError(f"metric scope is empty: {scope}")
            selected = frame.loc[mask]
            truth = selected["true_tvt"].to_numpy(np.float64)
            candidate = selected[candidate_column].to_numpy(np.float64)
            control = selected[control_column].to_numpy(np.float64)
            candidate_rmse = rmse(truth, candidate)
            control_rmse = rmse(truth, control)
            paired_rows.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "rows": len(selected),
                    "wells": int(selected["well_id"].nunique()),
                    "candidate_column": candidate_column,
                    "control_column": control_column,
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "improvement_ft": control_rmse - candidate_rmse,
                    "delta_rmse_candidate_minus_control": candidate_rmse
                    - control_rmse,
                    "candidate_mae": float(np.mean(np.abs(candidate - truth))),
                    "control_mae": float(np.mean(np.abs(control - truth))),
                    "candidate_bias": float(np.mean(candidate - truth)),
                    "control_bias": float(np.mean(control - truth)),
                }
            )
        for well, group in frame.groupby("well_id", sort=True):
            truth = group["true_tvt"].to_numpy(np.float64)
            candidate = group[candidate_column].to_numpy(np.float64)
            control = group[control_column].to_numpy(np.float64)
            candidate_rmse = rmse(truth, candidate)
            control_rmse = rmse(truth, control)
            candidate_mse = float(np.mean((candidate - truth) ** 2))
            control_mse = float(np.mean((control - truth) ** 2))
            by_well_rows.append(
                {
                    "comparison": comparison,
                    "well_id": str(well),
                    "rows": len(group),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "delta_rmse_candidate_minus_control": candidate_rmse
                    - control_rmse,
                    "candidate_mse": candidate_mse,
                    "control_mse": control_mse,
                    "delta_squared_error_candidate_minus_control": candidate_mse
                    - control_mse,
                }
            )
    return pd.DataFrame(paired_rows), pd.DataFrame(by_well_rows)


def _metric_row(
    metrics: pd.DataFrame,
    comparison: str,
    scope: str,
) -> pd.Series:
    selected = metrics.loc[
        metrics["comparison"].eq(comparison) & metrics["scope"].eq(scope)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row: {comparison}/{scope}")
    return selected.iloc[0]


def evaluate_promotion_gate(
    frame: pd.DataFrame,
    paired_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    frozen: Mapping[str, Any],
    audit: pd.DataFrame,
    late_attachment: Mapping[str, Any],
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "guards.technical") or {}
    scientific_config = get_nested(config, "guards.scientific") or {}
    parent_parity = parent_replay_parity(frame, preflight, config)
    finite_coverage = float(
        np.isfinite(
            frame[[*PREDICTION_COLUMNS, SAVED_PARENT]].to_numpy(np.float64)
        ).mean()
    )
    before_freeze = late_attachment["truth_access_ledger"]["before_freeze"]
    expected = get_nested(config, "model.execution_count") or {}
    actual_counts = frozen["actual_execution_counts"]
    technical_checks = {
        "all_input_sha_matches": bool(preflight["all_input_sha_matches"]),
        "rows": len(frame) == int(get_nested(config, "validation.expected_rows")),
        "wells": int(frame["well_id"].nunique())
        == int(get_nested(config, "validation.expected_wells")),
        "folds": sorted(frame["fold"].astype(int).unique().tolist())
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
        "finite_prediction_coverage": finite_coverage
        == float(technical_config["require_finite_prediction_coverage"]),
        "shared_trajectory_identity": bool(
            frozen["shared_trajectory_identity_per_well"]
            and audit[
                ["trajectory_bank_logical_sha256", "trajectory_well_logical_sha256"]
            ]
            .notna()
            .all()
            .all()
        ),
        "parent_kernel_prediction_parity": float(
            parent_parity["parent_marginal_replay_max_abs_ft"]
        )
        <= float(
            technical_config["require_parent_kernel_prediction_parity_atol_ft"]
        ),
        "arithmetic_mean_parity": float(
            parent_parity["arithmetic_mean_replay_max_abs_ft"]
        )
        <= float(
            technical_config[
                "require_arithmetic_mean_parity_vs_saved_exp404_atol_ft"
            ]
        ),
        "weight_sum": float(
            np.max(
                np.abs(
                    audit[
                        [
                            "gaussian_weight_sum",
                            "huber_weight_sum",
                            "parent_gaussian_marginal_weight_sum",
                        ]
                    ].to_numpy(np.float64)
                    - 1.0
                )
            )
        )
        <= float(technical_config["require_weight_sum_atol"]),
        "truth_late": all(int(value) == 0 for value in before_freeze.values()),
        "full_execution_counts": (
            int(actual_counts["pf_well_runs"]) == int(expected["full_pf_well_runs"])
            and int(actual_counts["seed_well_trajectories"])
            == int(expected["full_seed_well_trajectories"])
            and int(actual_counts["particle_starts"])
            == int(expected["full_particle_starts"])
            and int(actual_counts["shards"]) == int(expected["full_shards"])
            and int(expected["parent_independent_full_reruns"]) == 0
        ),
    }
    technical = {
        "checks": technical_checks,
        "passed": bool(all(technical_checks.values())),
        "finite_prediction_coverage": finite_coverage,
        "parent_replay_parity": parent_parity,
        "truth_access_before_freeze": before_freeze,
        "actual_execution_counts": actual_counts,
        "global_trajectory_manifest_sha256": frozen[
            "global_trajectory_manifest_sha256"
        ],
        "prediction_logical_sha256": frozen["prediction_logical_sha256"],
        "evidence_logical_sha256": frozen["evidence_logical_sha256"],
        "runtime_seconds": runtime_seconds,
    }
    fixed_scopes = [
        "shallow",
        "deep",
        "raw_gr_observed",
        "raw_gr_missing",
        "missing_fraction_high",
        "roughness_low",
        "roughness_high",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    scientific_references = [
        "huber_vs_gaussian_matched",
        "huber_vs_saved_exp404_temperature5",
    ]
    reference_reports: dict[str, Any] = {}
    reference_passes: list[bool] = []
    for comparison in scientific_references:
        overall = _metric_row(paired_metrics, comparison, "overall")
        fold_rows = paired_metrics.loc[
            paired_metrics["comparison"].eq(comparison)
            & paired_metrics["scope"].str.startswith("fold_")
        ]
        nonworse_folds = int(
            (fold_rows["delta_rmse_candidate_minus_control"] <= 0.0).sum()
        )
        scope_deltas = {
            scope: float(
                _metric_row(
                    paired_metrics,
                    comparison,
                    scope,
                )["delta_rmse_candidate_minus_control"]
            )
            for scope in fixed_scopes
        }
        well = by_well_metrics.loc[
            by_well_metrics["comparison"].eq(comparison)
        ].copy()
        squared_delta_p95 = float(
            well["delta_squared_error_candidate_minus_control"].quantile(0.95)
        )
        worst_rmse_delta = float(
            well["delta_rmse_candidate_minus_control"].max()
        )
        checks = {
            "overall_gain": float(overall["improvement_ft"])
            >= float(scientific_config["minimum_overall_rmse_gain_ft"]),
            "nonworse_folds": nonworse_folds
            >= int(scientific_config["minimum_nonworse_reporting_folds"]),
            "all_fixed_scopes_nonworse": all(
                delta <= float(scientific_config["maximum_scope_regression_ft"])
                for delta in scope_deltas.values()
            ),
            "paired_well_squared_error_delta_p95": squared_delta_p95
            <= float(
                scientific_config[
                    "maximum_paired_well_squared_error_delta_p95"
                ]
            ),
            "worst_paired_well_rmse_delta": worst_rmse_delta
            <= float(
                scientific_config["maximum_worst_paired_well_rmse_delta_ft"]
            ),
        }
        passed = bool(all(checks.values()))
        reference_passes.append(passed)
        reference_reports[comparison] = {
            "checks": checks,
            "passed": passed,
            "candidate_rmse": float(overall["candidate_rmse"]),
            "control_rmse": float(overall["control_rmse"]),
            "overall_gain_ft": float(overall["improvement_ft"]),
            "nonworse_folds": nonworse_folds,
            "fixed_scope_delta_rmse_ft": scope_deltas,
            "paired_well_squared_error_delta_p95": squared_delta_p95,
            "worst_paired_well_rmse_delta_ft": worst_rmse_delta,
        }
    arithmetic = _metric_row(
        paired_metrics,
        "huber_vs_arithmetic_mean",
        "overall",
    )
    arithmetic_check = float(arithmetic["improvement_ft"]) > 0.0
    scientific = {
        "references": reference_reports,
        "better_than_arithmetic_mean": {
            "passed": arithmetic_check,
            "candidate_rmse": float(arithmetic["candidate_rmse"]),
            "arithmetic_rmse": float(arithmetic["control_rmse"]),
            "gain_ft": float(arithmetic["improvement_ft"]),
        },
        "passed": bool(all(reference_passes) and arithmetic_check),
    }
    passed = bool(technical["passed"] and scientific["passed"])
    decision = get_nested(config, "guards.decision") or {}
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": str(
            decision["pass_action"] if passed else decision["fail_action"]
        ),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "primary_policy": (
            "fixed_huber_delta_1p345_temperature5_vs_gaussian_matched_and_"
            "saved_exp404_on_one_frozen_float64_trajectory_bank"
        ),
        "failure_action": (
            "close_without_delta_temperature_clip_scale_particle_seed_filtering_"
            "likelihood_affine_ar1_selfgr_reinjection_or_same_oof_rescue"
        ),
    }


# %% [markdown]
# ## 10. Generated artifacts and stage orchestration


# %%
def artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }
    if path.suffix == ".gz":
        report["decompressed_sha256"] = inspect_gzip_csv(path)[
            "decompressed_sha256"
        ]
    return report


def build_artifact_manifest(paths: Mapping[str, Path]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [
                {"name": name, **artifact_report(path)}
                for name, path in paths.items()
            ]
        )
        .sort_values("name", kind="mergesort")
        .reset_index(drop=True)
    )


def run_merge_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_scientific_contract(config, require_run_approval=True)
    approved_preflight = load_approved_preflight(config)
    started = time.time()
    artifacts = artifact_dir()
    preflight = preflight_saved_inputs(config)
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, contract)
    input_manifest_frame(preflight).to_csv(input_manifest_path, index=False)
    ledger = TruthAccessLedger()
    predictions, evidence, audit, frozen, frozen_paths = (
        load_and_freeze_full_shards(
            config,
            artifacts,
            ledger,
        )
    )
    frame, late_attachment = load_truth_late_frame(
        predictions,
        frozen,
        preflight,
        config,
        ledger,
    )
    paired_metrics, by_well_metrics = build_metric_outputs(frame, config)
    runtime_seconds = time.time() - started
    promotion_gate = evaluate_promotion_gate(
        frame,
        paired_metrics,
        by_well_metrics,
        frozen,
        audit,
        late_attachment,
        preflight,
        config,
        runtime_seconds,
    )
    metric_paths = {
        "overall_fold_scope_metrics": (
            artifacts / f"{OUTPUT_PREFIX}_overall_fold_scope_metrics.csv"
        ),
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
    }
    paired_metrics.to_csv(metric_paths["overall_fold_scope_metrics"], index=False)
    by_well_metrics.to_csv(metric_paths["by_well_metrics"], index=False)
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
    overall = _metric_row(
        paired_metrics,
        "huber_vs_gaussian_matched",
        "overall",
    )
    status = (
        "train_side_huber_seed_evidence_gate_passed_no_automatic_inference"
        if promotion_gate["passed"]
        else "train_side_huber_seed_evidence_gate_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "stage": "merge",
        "route": "pf_beam",
        "runtime_seconds": runtime_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "active_scientific_variants": 1,
        "pf_well_runs": int(
            frozen["actual_execution_counts"]["pf_well_runs"]
        ),
        "seed_well_trajectories": int(
            frozen["actual_execution_counts"]["seed_well_trajectories"]
        ),
        "particle_starts": int(
            frozen["actual_execution_counts"]["particle_starts"]
        ),
        "readouts_from_same_trajectory_bank": 2,
        "parent_independent_full_reruns": 0,
        "models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "approved_preflight_summary_sha256": str(
            _input_spec(config, "preflight_result")["expected_sha256"]
        ),
        "approved_preflight_passed": bool(approved_preflight["passed"]),
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "artifact_manifest_sha256": artifact_manifest_sha,
        "frozen_identity": frozen,
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
        "trajectory_sha256": frozen["global_trajectory_manifest_sha256"],
        "evidence_sha256": frozen["evidence_logical_sha256"],
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "artifact_manifest_sha256": artifact_manifest_sha,
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Train-side fixed Huber seed-evidence readout only. No inference, "
            "raw-test prediction, model, or submission is produced."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(paired_metrics.to_string(index=False))
    print(json.dumps(to_jsonable(promotion_gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def run_selected_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    stage = selected_stage(config)
    if stage is None:
        return None
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get(
        "EXPERIMENT_ALLOW_LOCAL"
    ) != "1":
        raise RuntimeError(
            "exp430 stages must run first on Kaggle; local execution requires "
            "explicit smoke approval"
        )
    if stage == "preflight":
        return run_preflight_stage(config)
    if stage == "full_shard":
        return run_full_shard_stage(config)
    if stage == "merge":
        return run_merge_stage(config)
    raise AssertionError(f"unreachable exp430 stage: {stage}")


# %% [markdown]
# ## 11. Setup and configuration preview


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
                "status": get_nested(CONFIG, "experiment.status"),
                "selected_stage": selected_stage(CONFIG),
                "shard_index": get_nested(CONFIG, "execution.shard_index"),
                "primary_control": PRIMARY_CONTROL,
                "primary_candidate": PRIMARY_CANDIDATE,
                "particles": get_nested(CONFIG, "model.pf.particles"),
                "seeds": get_nested(CONFIG, "model.pf.seeds"),
                "temperature": get_nested(CONFIG, "model.evidence.temperature"),
                "huber_delta": get_nested(
                    CONFIG,
                    "model.evidence.huber_delta_1p345.delta",
                ),
                "full_pf_well_runs": get_nested(
                    CONFIG,
                    "model.execution_count.full_pf_well_runs",
                ),
                "full_seed_well_trajectories": get_nested(
                    CONFIG,
                    "model.execution_count.full_seed_well_trajectories",
                ),
                "full_particle_starts": get_nested(
                    CONFIG,
                    "model.execution_count.full_particle_starts",
                ),
                "lightgbm_configs": 0,
                "trained_folds": 0,
                "boosters": 0,
                "models": 0,
                "gpu_runs": 0,
                "kaggle_push_approved": get_nested(
                    CONFIG,
                    "execution.kaggle_push_approved",
                ),
                "stage_run_approved": get_nested(
                    CONFIG,
                    "execution.stage_run_approved",
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 12. Run the explicitly approved Kaggle CPU stage


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_selected_stage(CONFIG)
    if SUMMARY is None:
        print(
            "exp430 implementation is ready, but no Kaggle stage is selected or "
            "approved. Set execution.selected_stage only after explicit approval."
        )
