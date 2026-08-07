# %% [markdown]
# # exp483 fixed-Huber GR filtering likelihood-PF — staged train
#
# This compact self-contained candidate changes one factor in the frozen
# exp404/exp417 likelihood-PF: each particle's Gaussian GR filtering score is
# replaced by the exp389 fixed Huber score with delta=1.345. Particle dynamics,
# the x1.0 prefix-derived GR scale, 500 particles, 128 stable seeds,
# resampling/roughening, and the temperature-5 primary seed aggregation remain
# fixed. Stage 0 is a fixed32 technical preflight; separately approved Stage 1
# is the all-773-well train-side CV and frozen promotion gate.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Fixed32 scope and truth-access ledger
# 5. Exp404 likelihood-PF input preparation
# 6. Gaussian reference and fixed-Huber filtering kernels
# 7. Formula, no-op PF, and stable-seed contracts
# 8. Target-free candidate generation and prediction freeze
# 9. Truth-late fixed32 diagnostics and technical gates
# 10. Generated artifacts and Stage 0 orchestration
# 11. All-well Stage 1 truth-late CV and promotion gate
# 12. Setup, configuration preview, and selected execution

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


EXPERIMENT_NAME = "exp483_huber_gr_filtering_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
PRIMARY_CANDIDATE = "likpf_scale5_huber_delta1p345"
ARITHMETIC_PARITY = "likpf_mean_huber_delta1p345"
PREDICTION_COLUMNS = (PRIMARY_CANDIDATE, ARITHMETIC_PARITY)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
HUBER_DELTA = 1.345


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP483_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    candidates = ([package_dir / "config.yaml"] if package_dir is not None else []) + [
        path / "config.yaml" for path in candidate_package_dirs()
    ]
    for path in candidates:
        if path.exists():
            config = read_yaml(path)
            if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
                return config
    raise FileNotFoundError(f"could not resolve {EXPERIMENT_NAME}/config.yaml")


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
    local = project_root() / str(get_nested(config, "data.train_dir"))
    if local.exists():
        return local
    if KAGGLE_INPUT_ROOT.exists():
        candidates = sorted(KAGGLE_INPUT_ROOT.glob("**/train"))
        for candidate in candidates:
            if any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    raise FileNotFoundError("raw train directory was not found")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in PREDICTION_COLUMNS:
        if column in selected:
            selected[column] = selected[column].astype(np.float32)
    payload = selected.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.9g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return mapping_sha256(schema)


def typed_dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    """Match the frozen all-well raw-input identity contract used by exp404."""
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
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
    }


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if platform.system() != "Darwin" else 1024.0**2
    return value / divisor / 1024.0


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba_available": str(NUMBA_AVAILABLE),
    }


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
    local = project_root() / local_path
    candidates = [
        Path.cwd() / "assets" / filename,
        KAGGLE_WORKING_ROOT / "assets" / filename,
        local,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(filename)


# %% [markdown]
# ## 3. Frozen scientific and execution contracts


# %%
def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, int]:
    counts = {
        "stage_0_candidate_pf_well_runs": int(
            get_nested(config, "execution.stage_0_candidate_pf_well_runs")
        ),
        "stage_0_seed_well_trajectories": int(
            get_nested(config, "execution.stage_0_seed_well_trajectories")
        ),
        "stage_0_particle_starts": int(get_nested(config, "execution.stage_0_particle_starts")),
        "stage_1_candidate_pf_well_runs": int(
            get_nested(config, "execution.stage_1_candidate_pf_well_runs")
        ),
        "stage_1_seed_well_trajectories": int(
            get_nested(config, "execution.stage_1_seed_well_trajectories")
        ),
        "stage_1_particle_starts": int(get_nested(config, "execution.stage_1_particle_starts")),
        "control_pf_well_runs": int(get_nested(config, "execution.control_pf_well_runs")),
        "lightgbm_configs": int(get_nested(config, "execution.lightgbm_configs")),
        "boosters": int(get_nested(config, "execution.boosters")),
        "hmm_well_runs": int(get_nested(config, "execution.hmm_well_runs")),
        "beam_well_runs": int(get_nested(config, "execution.beam_well_runs")),
        "gpu_runs": int(get_nested(config, "execution.gpu_runs")),
    }
    expected = {
        "stage_0_candidate_pf_well_runs": 32,
        "stage_0_seed_well_trajectories": 4096,
        "stage_0_particle_starts": 2048000,
        "stage_1_candidate_pf_well_runs": 773,
        "stage_1_seed_well_trajectories": 98944,
        "stage_1_particle_starts": 49472000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    if counts != expected:
        raise ValueError(f"exp483 execution count contract changed: {counts}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp483 implementation approval is not recorded")
    run_stage0 = bool(get_nested(config, "execution.run_stage_0"))
    run_stage1 = bool(get_nested(config, "execution.run_stage_1"))
    if run_stage0 and run_stage1:
        raise ValueError("exp483 permits exactly one active execution stage")
    if run_stage1 and not bool(
        get_nested(config, "stage_0_result.all_technical_gates_passed")
    ):
        raise RuntimeError("exp483 Stage 1 requires a recorded Stage 0 technical PASS")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp483 Kaggle push is not approved")
        if run_stage0 and not bool(
            get_nested(config, "execution.stage_0_execution_approved")
        ):
            raise RuntimeError("exp483 Stage 0 Kaggle execution is not approved")
        if run_stage1 and not bool(
            get_nested(config, "execution.stage_1_execution_approved")
        ):
            raise RuntimeError("exp483 Stage 1 Kaggle execution is not approved")
        if not (run_stage0 or run_stage1):
            raise RuntimeError("exp483 has no approved execution stage selected")
    return counts


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "scientific_parent": get_nested(config, "lineage.parent"),
        "implementation_reference": get_nested(config, "lineage.exact_pf_implementation_reference"),
        "changed_factor": {
            "family": "huber",
            "delta": HUBER_DELTA,
            "z": "(gr_observed-typewell_gr_at_particle_tvt)/exp404_sigma_gr",
            "rho": "0.5*z^2 if abs(z)<=delta else delta*abs(z)-0.5*delta^2",
            "log_likelihood": "-rho",
            "normalization_constant": "omitted_state_independent",
            "additional_clip": None,
            "application_scope": "every_finite_gr_particle_update",
        },
        "fixed_pf": dict(get_nested(config, "model.fixed_from_exp404") or {}),
        "primary_readout": {
            "column": PRIMARY_CANDIDATE,
            "temperature": 5.0,
        },
        "secondary_readout": {
            "column": ARITHMETIC_PARITY,
            "promotion_eligible": False,
        },
        "saved_control": {
            "source": get_nested(config, "data.saved_control.source"),
            "rerun": False,
        },
        "stage_0": dict(get_nested(config, "stages.stage_0") or {}),
        "stage_1": dict(get_nested(config, "stages.stage_1") or {}),
        "seed_policy": get_nested(config, "reproducibility.seed_policy"),
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
        "forbidden": list(get_nested(config, "guards.forbidden") or []),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "model.active_variants": ["huber_delta1p345"],
        "model.changed_factor.family": "huber",
        "model.changed_factor.delta": 1.345,
        "model.changed_factor.additional_clip": "none",
        "model.fixed_from_exp404.particles": 500,
        "model.fixed_from_exp404.seeds": 128,
        "model.fixed_from_exp404.primary_seed_weighting_temperature": 5.0,
        "model.fixed_from_exp404.gr_scale_multiplier": 1.0,
        "model.fixed_from_exp404.initial_position_spread_ft": 4.5,
        "model.fixed_from_exp404.initial_rate_spread": 0.01,
        "model.fixed_from_exp404.momentum": 0.998,
        "model.fixed_from_exp404.rate_noise": 0.002,
        "model.fixed_from_exp404.position_noise": 0.005,
        "model.fixed_from_exp404.rough_position": 0.1,
        "model.fixed_from_exp404.rough_rate": 0.001,
        "model.fixed_from_exp404.resample_threshold_fraction": 0.5,
        "model.fixed_from_exp404.typewell_grid_step_ft": 0.2,
        "model.fixed_from_exp404.typewell_tvt_pad_ft": 100.0,
        "data.saved_control.rerun": False,
        "validation.fixed32_is_cv": False,
        "execution.active_variants": 1,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.use_gpu": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp483 fixed contract mismatch: {key} must be {value!r}")
    scale_clip = get_nested(config, "model.fixed_from_exp404.base_scale_clip")
    if [float(value) for value in scale_clip] != [
        10.0,
        60.0,
    ]:
        raise ValueError("exp483 fixes the exp404 GR scale clip to [10, 60]")
    validate_execution_contract(config)
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Fixed32 scope and truth-access ledger


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
            raise RuntimeError(f"{kind} input was read before all fixed32 artifacts froze")
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


def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_path(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(
            f"exp483 fixed32 manifest SHA mismatch: expected={spec['expected_sha256']}, "
            f"observed={observed}"
        )
    return path


def load_fixed32_scope(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    path = fixed32_manifest_path(config)
    frame = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    wells = sorted(frame["well"].astype(str).tolist())
    if len(wells) != 32 or len(set(wells)) != 32:
        raise ValueError("exp483 fixed32 scope identity changed")
    return wells, {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "wells": len(wells),
        "well_identity_sha256": mapping_sha256(wells),
        "columns_read_before_freeze": ["well"],
    }


def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
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
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    frame = frame.reset_index(drop=True)
    if len(frame) < 2 or not np.isfinite(frame["TVT"].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    typewell_mean = float(frame["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(typewell_mean)
    return frame


# %% [markdown]
# ## 5. Exp404 likelihood-PF input preparation


# %%
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
    raw_scale = float(np.nanstd(known_gr - typewell_at_known))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    return {
        "raw_scale": raw_scale,
        "base_scale": float(np.clip(raw_scale, clip[0], clip[1])),
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
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
    grid_step: float = 0.2,
) -> dict[str, Any]:
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires non-empty known prefix and unknown suffix")
    last_known_index = int(np.flatnonzero(known_mask)[-1])
    last_known_tvt = float(horizontal["TVT_input"].iloc[last_known_index])
    last_known_md = float(horizontal["MD"].iloc[last_known_index])
    last_position = last_known_tvt + float(horizontal["Z"].iloc[last_known_index])
    evaluation = horizontal.loc[eval_mask, ["MD", "Z", "GR"]]
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    scale_audit = exp072_base_gr_scale(horizontal, typewell_tvt, typewell_gr)
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
            "candidate_scale": float(scale_audit["base_scale"]),
            "multiplier": 1.0,
            "post_multiplier_clip_applied": False,
            "post_multiplier_clip_count": 0,
        },
    }


# %% [markdown]
# ## 6. Gaussian reference and fixed-Huber filtering kernels


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


@njit(cache=True)
def _huber_loss_scalar(zscore: float, delta: float) -> float:
    absolute = abs(zscore)
    if absolute <= delta:
        return 0.5 * zscore * zscore
    return delta * absolute - 0.5 * delta * delta


def huber_loss(zscore: np.ndarray | float, delta: float = HUBER_DELTA) -> np.ndarray:
    values = np.asarray(zscore, dtype=np.float64)
    absolute = np.abs(values)
    return np.where(
        absolute <= float(delta),
        0.5 * values * values,
        float(delta) * absolute - 0.5 * float(delta) ** 2,
    )


@njit(cache=True, nogil=True)
def _pf_gaussian_allseeds(
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
    """Exact exp404 capped-Gaussian reference, used only by synthetic parity."""
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
            delta_md = max(md_v[row] - previous_md, 1.0)
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
                squared = min(zscore * zscore, 600.0)
                likelihood = max(np.exp(-0.5 * squared), 1e-300)
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            average_likelihood = max(average_likelihood, 1e-300)
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
            minimum_ess[seed_index] = min(minimum_ess[seed_index], effective_sample_size)
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


@njit(cache=True, nogil=True)
def _pf_huber_allseeds(
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
    delta: float,
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
]:
    """Exp404 PF with only the per-particle GR score replaced by fixed Huber."""
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    huber_linear_counts = np.zeros(seeds, np.int64)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        row_log_weight = np.empty(particles)
        for particle in range(particles):
            position[particle] = last_position + initial_spread * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = max(md_v[row] - previous_md, 1.0)
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
            maximum_log_weight = -np.inf
            for particle in range(particles):
                expected_gr = _interp1(
                    grid_gr,
                    position[particle] - z_v[row],
                    grid_minimum,
                    grid_step,
                )
                zscore = (gr_v[row] - expected_gr) / gr_scale
                if abs(zscore) > delta:
                    huber_linear_counts[seed_index] += 1
                score = -_huber_loss_scalar(zscore, delta)
                if weights[particle] > 0.0:
                    log_weight = np.log(weights[particle]) + score
                else:
                    log_weight = -np.inf
                row_log_weight[particle] = log_weight
                maximum_log_weight = max(maximum_log_weight, log_weight)
            scaled_weight_sum = 0.0
            for particle in range(particles):
                scaled_weight = np.exp(row_log_weight[particle] - maximum_log_weight)
                scaled_weight_sum += scaled_weight
                weights[particle] = scaled_weight
            log_likelihood += maximum_log_weight + np.log(scaled_weight_sum)
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
            minimum_ess[seed_index] = min(minimum_ess[seed_index], effective_sample_size)
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
        huber_linear_counts,
    )


def aggregate_seed_predictions(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    temperature: float,
) -> dict[str, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / float(temperature))
    weights /= weights.sum()
    return {
        PRIMARY_CANDIDATE: (weights[:, None] * predictions).sum(axis=0),
        ARITHMETIC_PARITY: predictions.mean(axis=0),
    }


def run_huber_pf(
    prepared: Mapping[str, Any],
    *,
    particles: int,
    seeds: int,
    seed_base: int,
    temperature: float,
    delta: float = HUBER_DELTA,
    momentum: float = 0.998,
    rate_noise: float = 0.002,
    position_noise: float = 0.005,
    rough_position: float = 0.1,
    rough_rate: float = 0.001,
    resample_fraction: float = 0.5,
    initial_spread: float = 4.5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.time()
    output = _pf_huber_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        int(particles),
        int(seeds),
        int(seed_base),
        float(delta),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_spread),
    )
    predictions, log_likelihoods, resampling, minimum_ess, clips, linear = output
    readouts = aggregate_seed_predictions(predictions, log_likelihoods, temperature)
    particle_updates = int(particles) * len(prepared["eval_md"]) * int(seeds)
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "seed_loglik_mean_per_row": float(log_likelihoods.mean()) / len(prepared["eval_md"]),
        "seed_loglik_best_per_row": float(log_likelihoods.max()) / len(prepared["eval_md"]),
        "seed_loglik_spread": float(log_likelihoods.std()),
        "resampling_count_total": int(resampling.sum()),
        "resampling_count_min": int(resampling.min()),
        "resampling_count_max": int(resampling.max()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "position_clip_count_total": int(clips.sum()),
        "huber_linear_particle_updates": int(linear.sum()),
        "huber_linear_particle_update_fraction": float(linear.sum()) / float(particle_updates),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
    }
    return readouts, diagnostics


# %% [markdown]
# ## 7. Formula, no-op PF, and stable-seed contracts


# %%
def formula_unit_contract(delta: float = HUBER_DELTA) -> dict[str, Any]:
    zscore = np.asarray(
        [-1000.0, -2.0, -delta, -0.5, 0.0, 0.5, delta, 2.0, 1000.0],
        dtype=np.float64,
    )
    observed = huber_loss(zscore, delta)
    expected = np.asarray(
        [
            delta * 1000.0 - 0.5 * delta**2,
            delta * 2.0 - 0.5 * delta**2,
            0.5 * delta**2,
            0.125,
            0.0,
            0.125,
            0.5 * delta**2,
            delta * 2.0 - 0.5 * delta**2,
            delta * 1000.0 - 0.5 * delta**2,
        ],
        dtype=np.float64,
    )
    inside = np.abs(zscore) <= delta
    gaussian_inside = 0.5 * zscore[inside] ** 2
    return {
        "delta": float(delta),
        "maximum_formula_abs_error": float(np.max(np.abs(observed - expected))),
        "maximum_inside_gaussian_abs_error": float(
            np.max(np.abs(observed[inside] - gaussian_inside))
        ),
        "large_z_loss": float(observed[-1]),
        "large_z_expected_unclipped_loss": float(expected[-1]),
        "additional_clip_detected": bool(observed[-1] != expected[-1]),
        "pass": bool(
            np.array_equal(observed, expected)
            and np.array_equal(observed[inside], gaussian_inside)
            and observed[-1] > 1000.0
        ),
    }


def no_op_toy_pf_contract() -> dict[str, Any]:
    md = np.arange(1.0, 9.0, dtype=np.float64)
    z = np.zeros_like(md)
    observed_gr = np.full_like(md, 50.0)
    grid_gr = np.full(201, 50.0, dtype=np.float64)
    common = (
        md,
        z,
        observed_gr,
        grid_gr,
        80.0,
        0.2,
        20.0,
        100.0,
        0.01,
        32,
        4,
        12345,
    )
    tail = (0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5)
    gaussian = _pf_gaussian_allseeds(*common, *tail)
    huber = _pf_huber_allseeds(*common, HUBER_DELTA, *tail)
    parity = [np.array_equal(gaussian[index], huber[index]) for index in range(5)]
    return {
        "constant_gr_zero_residual": True,
        "prediction_bitwise_equal": parity[0],
        "seed_log_likelihood_bitwise_equal": parity[1],
        "resampling_ledger_bitwise_equal": parity[2],
        "minimum_ess_bitwise_equal": parity[3],
        "position_clip_ledger_bitwise_equal": parity[4],
        "huber_linear_particle_updates": int(huber[5].sum()),
        "pass": bool(all(parity) and int(huber[5].sum()) == 0),
    }


def stable_seed_contract(wells: Sequence[str], seeds: int) -> dict[str, Any]:
    rows = []
    for well in wells:
        base = stable_seed("likpf", "train", well)
        repeated = stable_seed("likpf", "train", well)
        rows.append(
            {
                "well": str(well),
                "seed_base": base,
                "seed_last": base + int(seeds) - 1,
                "repeat_equal": base == repeated,
                "variant_excluded": "huber" not in f"likpf::train::{well}",
            }
        )
    return {
        "rows": rows,
        "unique_seed_bases": len({row["seed_base"] for row in rows}),
        "all_repeat_equal": all(row["repeat_equal"] for row in rows),
        "variant_name_excluded": all(row["variant_excluded"] for row in rows),
        "pass": bool(
            len({row["seed_base"] for row in rows}) == len(wells)
            and all(row["repeat_equal"] for row in rows)
            and all(row["variant_excluded"] for row in rows)
        ),
    }


def warm_up_pf_kernel() -> None:
    no_op_toy_pf_contract()


# %% [markdown]
# ## 8. Target-free candidate generation and prediction freeze


# %%
@dataclass
class FrozenWell:
    well: str
    prediction: pd.DataFrame
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
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(fixed["typewell_grid_step_ft"]),
    )
    seed_base = stable_seed("likpf", "train", well)
    readouts, diagnostics = run_huber_pf(
        prepared,
        particles=int(fixed["particles"]),
        seeds=int(fixed["seeds"]),
        seed_base=seed_base,
        temperature=float(fixed["primary_seed_weighting_temperature"]),
        delta=float(get_nested(config, "model.changed_factor.delta")),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(fixed["rough_position"]),
        rough_rate=float(fixed["rough_rate"]),
        resample_fraction=float(fixed["resample_threshold_fraction"]),
        initial_spread=float(fixed["initial_position_spread_ft"]),
    )
    eval_indices = np.asarray(prepared["eval_indices"], dtype=np.int64)
    raw_observed = np.asarray(prepared["raw_gr_observed"], dtype=bool)
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64(1.0 - raw_observed.mean()),
            PRIMARY_CANDIDATE: readouts[PRIMARY_CANDIDATE].astype(np.float32),
            ARITHMETIC_PARITY: readouts[ARITHMETIC_PARITY].astype(np.float32),
        }
    )
    if not np.isfinite(frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: non-finite exp483 prediction")
    particles = int(fixed["particles"])
    seeds = int(fixed["seeds"])
    audit = {
        "well_id": str(well),
        "status": "ok",
        "rows": len(frame),
        "raw_gr_observed_rows": int(raw_observed.sum()),
        "raw_gr_missing_rows": int((~raw_observed).sum()),
        "gr_scale_raw": float(prepared["scale_audit"]["raw_scale"]),
        "gr_scale_x1p0": float(prepared["scale_audit"]["candidate_scale"]),
        "seed_base": seed_base,
        "seed_last": seed_base + seeds - 1,
        "pf_well_runs": 1,
        "seeds": seeds,
        "seed_well_trajectories": seeds,
        "particles": particles,
        "particle_starts": particles * seeds,
        "runtime_seconds": time.time() - started,
        **diagnostics,
    }
    return FrozenWell(well=str(well), prediction=frame, audit=audit)


def freeze_target_free_outputs(
    results: Sequence[FrozenWell],
    output: Path,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
    *,
    stage: str = "stage0",
    expected_rows: int | None = None,
    expected_wells: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = sorted(results, key=lambda item: item.well)
    candidate = pd.concat(
        [item.prediction for item in ordered],
        ignore_index=True,
    ).sort_values(["well_id", "row_idx"], kind="mergesort")
    candidate = candidate.reset_index(drop=True)
    audit = pd.DataFrame([item.audit for item in ordered]).sort_values("well_id", kind="mergesort")
    audit = audit.reset_index(drop=True)
    if expected_rows is None:
        expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    if (
        len(candidate) != expected_rows
        or candidate["well_id"].nunique() != expected_wells
        or candidate["id"].duplicated().any()
        or candidate.duplicated(["well_id", "row_idx"]).any()
    ):
        raise ValueError(f"exp483 {stage} prediction identity or coverage changed")
    prediction_path = output / f"{OUTPUT_PREFIX}_{stage}_predictions.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_{stage}_well_audit.csv"
    logical_columns = ["id", "well_id", "row_idx", *PREDICTION_COLUMNS]
    logical_sha = dataframe_content_sha(candidate, logical_columns)
    prediction_artifact = write_deterministic_gzip_csv(candidate, prediction_path)
    audit.to_csv(audit_path, index=False)
    audit_artifact = {
        "path": str(audit_path),
        "bytes": audit_path.stat().st_size,
        "raw_sha256": sha256_path(audit_path),
    }
    readback = pd.read_csv(prediction_path, compression="gzip", dtype={"id": str, "well_id": str})
    readback_logical_sha = dataframe_content_sha(readback, logical_columns)
    readback_audit_sha = sha256_path(audit_path)
    readback_pass = bool(
        logical_sha == readback_logical_sha
        and readback_audit_sha == audit_artifact["raw_sha256"]
        and sha256_decompressed_csv(prediction_path) == prediction_artifact["decompressed_sha256"]
    )
    if not readback_pass:
        raise RuntimeError("exp483 frozen artifact SHA readback failed")
    for item in ordered:
        ledger.freeze(item.well)
    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "logical_columns": logical_columns,
        "prediction_logical_sha256": logical_sha,
        "prediction_schema_sha256": dataframe_schema_sha(candidate),
        "prediction_artifact": prediction_artifact,
        "well_audit_artifact": audit_artifact,
        "sha_readback": {
            "prediction_logical_sha256": readback_logical_sha,
            "well_audit_raw_sha256": readback_audit_sha,
            "pass": readback_pass,
        },
        "truth_access_ledger_at_freeze": ledger.report(),
    }
    return candidate, audit, frozen


# %% [markdown]
# ## 9. Truth-late fixed32 diagnostics and technical gates
#
# The fixed32 truth/control join is report-only and is not CV or a promotion
# decision. It occurs only after candidate predictions, the well audit, and
# their content hashes have been frozen and read back.


# %%
def require_frozen(frozen: Mapping[str, Any], ledger: LeakageLedger) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")) or not ledger.all_frozen:
        raise RuntimeError("truth-late readout requires all fixed32 artifacts to be frozen")
    if len(str(frozen.get("prediction_logical_sha256", ""))) != 64:
        raise RuntimeError("frozen prediction logical SHA is missing")


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


def saved_control_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.saved_control") or {})
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
        [str(value) for value in spec.get("patterns", [])],
    )
    observed_raw = sha256_path(path)
    observed_decompressed = sha256_decompressed_csv(path)
    if observed_raw != str(spec["expected_raw_sha256"]):
        raise ValueError("exp483 saved exp404 control raw SHA mismatch")
    if observed_decompressed != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp483 saved exp404 control decompressed SHA mismatch")
    return path


def attach_truth_late_fixed32(
    candidate: pd.DataFrame,
    frozen: Mapping[str, Any],
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    require_frozen(frozen, ledger)
    logical_sha = dataframe_content_sha(candidate, frozen["logical_columns"])
    if logical_sha != str(frozen["prediction_logical_sha256"]):
        raise RuntimeError("exp483 candidate changed after prediction freeze")
    wells = sorted(candidate["well_id"].astype(str).unique().tolist())
    truth = pd.concat(
        [load_suffix_truth(well, raw_dir) for well in wells],
        ignore_index=True,
    )
    ledger.record_truth(len(truth))
    control_spec = dict(get_nested(config, "data.saved_control") or {})
    control_column = str(control_spec["prediction_column"])
    control = pd.read_csv(
        saved_control_path(config),
        compression="gzip",
        usecols=["id", control_column],
        dtype={"id": str},
    )
    ledger.record_control(len(control))
    selected_ids = set(candidate["id"].astype(str))
    control = control.loc[control["id"].astype(str).isin(selected_ids)].copy()
    frame = candidate.merge(truth, on="id", how="left", validate="one_to_one")
    frame = frame.merge(control, on="id", how="left", validate="one_to_one")
    if frame[["true_tvt", control_column]].isna().any().any():
        raise ValueError("exp483 truth-late fixed32 join is incomplete")
    frame["candidate_squared_error"] = (
        frame[PRIMARY_CANDIDATE].to_numpy(np.float64) - frame["true_tvt"].to_numpy(np.float64)
    ) ** 2
    frame["control_squared_error"] = (
        frame[control_column].to_numpy(np.float64) - frame["true_tvt"].to_numpy(np.float64)
    ) ** 2
    by_well = (
        frame.groupby("well_id", sort=True)
        .agg(
            rows=("id", "size"),
            candidate_mse=("candidate_squared_error", "mean"),
            control_mse=("control_squared_error", "mean"),
            raw_gr_observed_fraction=("raw_gr_observed", "mean"),
        )
        .reset_index()
    )
    by_well["candidate_rmse"] = np.sqrt(by_well.pop("candidate_mse"))
    by_well["control_rmse"] = np.sqrt(by_well.pop("control_mse"))
    by_well["delta_rmse_candidate_minus_control"] = (
        by_well["candidate_rmse"] - by_well["control_rmse"]
    )
    report = {
        "policy": "fixed32_report_only_not_cv_not_promotion",
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "candidate_rmse": float(np.sqrt(frame["candidate_squared_error"].mean())),
        "saved_control_rmse": float(np.sqrt(frame["control_squared_error"].mean())),
        "delta_rmse_candidate_minus_control": float(
            np.sqrt(frame["candidate_squared_error"].mean())
            - np.sqrt(frame["control_squared_error"].mean())
        ),
        "improved_wells": int((by_well["delta_rmse_candidate_minus_control"] < 0.0).sum()),
        "truth_access_ledger": ledger.report(),
    }
    return frame, by_well, report


def evaluate_stage0_technical_gates(
    config: Mapping[str, Any],
    formula: Mapping[str, Any],
    no_op: Mapping[str, Any],
    seed_report: Mapping[str, Any],
    candidate: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    ledger_at_freeze: Mapping[str, Any],
    runtime_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    guards = dict(get_nested(config, "guards.technical") or {})
    projected_seconds = runtime_seconds * 773.0 / 32.0
    before = dict(ledger_at_freeze["before_freeze"])
    checks = {
        "formula_unit_contract": bool(formula["pass"]),
        "huber_equals_gaussian_inside_delta": bool(
            float(formula["maximum_inside_gaussian_abs_error"]) == 0.0
        ),
        "no_op_toy_pf_bitwise_parity": bool(no_op["pass"]),
        "finite_prediction_coverage": bool(
            np.isfinite(candidate[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).mean()
            >= float(guards["require_finite_prediction_coverage"])
        ),
        "stable_seed_identity": bool(seed_report["pass"]),
        "execution_count_match": bool(
            len(audit) == 32
            and int(audit["pf_well_runs"].sum()) == 32
            and int(audit["seed_well_trajectories"].sum()) == 4096
            and int(audit["particle_starts"].sum()) == 2048000
        ),
        "truth_error_fold_hidden_reads_before_freeze_zero": bool(
            sum(int(value) for value in before.values())
            <= int(guards["maximum_truth_error_fold_hidden_reads_before_freeze"])
        ),
        "artifact_sha_readback": bool(frozen["sha_readback"]["pass"]),
        "full_runtime_projection": bool(
            projected_seconds <= float(guards["maximum_seconds_full_projection"])
        ),
        "peak_rss": bool(rss_gb <= float(guards["maximum_peak_rss_gb"])),
    }
    return {
        "stage": "stage0_fixed32_technical_preflight_not_cv",
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "runtime_seconds_fixed32": runtime_seconds,
        "runtime_seconds_full_773_projection": projected_seconds,
        "peak_rss_gb": rss_gb,
        "promotion_decision": "not_evaluated_stage0_is_not_cv",
    }


# %% [markdown]
# ## 10. Generated artifacts and Stage 0 orchestration


# %%
def require_kaggle_runtime() -> None:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXP483_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp483 train stages must run on Kaggle; local execution requires "
            "explicit smoke approval"
        )


def input_manifest(
    config: Mapping[str, Any],
    raw_dir: Path,
    wells: Sequence[str],
    scope_report: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for well in wells:
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        if not horizontal.exists() or not typewell.exists():
            raise FileNotFoundError(f"{well}: raw input pair is missing")
        rows.append(
            {
                "well": str(well),
                "horizontal_raw_sha256": sha256_path(horizontal),
                "typewell_raw_sha256": sha256_path(typewell),
            }
        )
    return {
        "experiment": EXPERIMENT_NAME,
        "fixed32": dict(scope_report),
        "raw_dir": str(raw_dir),
        "raw_well_identity_sha256": mapping_sha256(rows),
        "raw_wells": rows,
        "saved_control_expected_raw_sha256": get_nested(
            config, "data.saved_control.expected_raw_sha256"
        ),
        "saved_control_expected_decompressed_sha256": get_nested(
            config, "data.saved_control.expected_decompressed_sha256"
        ),
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    scientific_contract = validate_scientific_contract(config)
    validate_execution_contract(config, require_run_approval=True)
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    wells, scope_report = load_fixed32_scope(config)
    ledger = LeakageLedger(expected_wells=len(wells))
    formula = formula_unit_contract()
    no_op = no_op_toy_pf_contract()
    seed_report = stable_seed_contract(
        wells,
        int(get_nested(config, "model.fixed_from_exp404.seeds")),
    )
    if not (
        bool(formula["pass"])
        and bool(no_op["pass"])
        and bool(seed_report["pass"])
    ):
        raise RuntimeError("exp483 formula/no-op/seed contract failed")
    contract_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_scientific_contract.json",
        scientific_contract,
    )
    inputs = input_manifest(config, raw_dir, wells, scope_report)
    input_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_input_manifest.json",
        inputs,
    )
    warm_up_pf_kernel()
    results = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(delayed(decode_target_free_well)(well, raw_dir, config) for well in wells)
    candidate, audit, frozen = freeze_target_free_outputs(
        results,
        output,
        config,
        ledger,
    )
    runtime_seconds = time.time() - started
    rss_gb = peak_rss_gb()
    runtime_report = {
        "runtime_seconds_to_prediction_freeze": runtime_seconds,
        "peak_rss_gb": rss_gb,
        "runtime_versions": runtime_versions(),
        "kaggle_kernel_version": None,
        "kernel_version_recording": "record_from_kaggle_api_after_run",
    }
    runtime_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_runtime_ledger.json",
        runtime_report,
    )
    freeze_manifest = {
        **frozen,
        "scientific_contract_file_sha256": contract_artifact["raw_sha256"],
        "input_manifest_file_sha256": input_artifact["raw_sha256"],
        "runtime_ledger_file_sha256": runtime_artifact["raw_sha256"],
        "formula_contract": formula,
        "no_op_toy_pf_contract": no_op,
        "stable_seed_contract": seed_report,
    }
    freeze_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_freeze_manifest.json",
        freeze_manifest,
    )
    ledger_at_freeze = ledger.report()
    frame, by_well, report_only = attach_truth_late_fixed32(
        candidate,
        frozen,
        raw_dir,
        config,
        ledger,
    )
    gates = evaluate_stage0_technical_gates(
        config,
        formula,
        no_op,
        seed_report,
        candidate,
        audit,
        frozen,
        ledger_at_freeze,
        runtime_seconds,
        rss_gb,
    )
    truth_path = output / f"{OUTPUT_PREFIX}_stage0_truth_late_rows.csv.gz"
    by_well_path = output / f"{OUTPUT_PREFIX}_stage0_truth_late_by_well.csv"
    gate_path = output / f"{OUTPUT_PREFIX}_stage0_technical_gate.json"
    report_path = output / f"{OUTPUT_PREFIX}_stage0_report_only_metrics.json"
    truth_artifact = write_deterministic_gzip_csv(frame, truth_path)
    by_well.to_csv(by_well_path, index=False)
    gate_artifact = write_json(gate_path, gates)
    report_artifact = write_json(report_path, report_only)
    status = (
        "stage0_technical_pass_no_automatic_stage1"
        if gates["all_pass"]
        else "stage0_technical_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage0_fixed32_technical_preflight_not_cv",
        "cv": None,
        "public_lb": None,
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "seed_well_trajectories": int(audit["seed_well_trajectories"].sum()),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "lightgbm_configs": 0,
        "boosters": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "technical_gate": gates,
        "fixed32_report_only": report_only,
        "truth_access_ledger": ledger.report(),
        "artifacts": {
            "scientific_contract": contract_artifact,
            "input_manifest": input_artifact,
            "runtime_ledger": runtime_artifact,
            "freeze_manifest": freeze_artifact,
            "prediction": frozen["prediction_artifact"],
            "well_audit": frozen["well_audit_artifact"],
            "truth_late_rows": truth_artifact,
            "truth_late_by_well": {
                "path": str(by_well_path),
                "raw_sha256": sha256_path(by_well_path),
            },
            "technical_gate": gate_artifact,
            "report_only_metrics": report_artifact,
        },
        "deterministic_anchor": False,
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "stage0_is_cv": False,
        "technical_gate": gates,
        "fixed32_report_only": report_only,
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "notes": (
            "Fixed32 technical preflight only. No route anchor, Stage 1, raw-test "
            "prediction, inference, submission, or deterministic-anchor claim."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. All-well Stage 1 truth-late CV and promotion gate
#
# Stage 1 preserves the exact Stage 0 PF treatment and runs it once for all
# 773 train wells. Candidate prediction and content hashes freeze before
# suffix truth, saved controls, reporting folds, or hidden-like roles are
# parsed. Saved exp404 PF and exp209 HMM predictions are read only after the
# freeze; neither control is rerun.


# %%
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
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    actual = typed_dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != expected_wells or actual != expected_sha:
        raise ValueError("exp483 raw train well-file identity mismatch")
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
        raise RuntimeError("exp483 Stage 1 candidate changed after prediction freeze")
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
        raise ValueError("exp483 saved exp404 control raw SHA mismatch")
    if sha256_decompressed_csv(control_path) != str(
        control_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp483 saved exp404 control decompressed SHA mismatch")
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
    if sha256_decompressed_csv(hmm_path) != str(
        hmm_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp483 saved exp209 HMM decompressed SHA mismatch")
    hmm_source_column = str(hmm_spec["prediction_column"])
    hmm = pd.read_csv(
        hmm_path,
        usecols=["id", hmm_source_column],
        dtype={"id": str},
    )
    ledger.record_control(len(hmm))
    hmm[hmm_source_column] = pd.to_numeric(hmm[hmm_source_column], errors="raise")
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
        raise ValueError("exp483 reporting-fold decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    forbidden = {str(value) for value in fold_spec.get("forbidden_decoder_columns", [])}
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp483 fold allowlist must contain identity/fold columns only")
    if set(safe_columns) & forbidden:
        raise ValueError("exp483 fold allowlist contains forbidden decoder columns")
    fold = pd.read_csv(fold_path, usecols=safe_columns, dtype={"well_id": str})
    ledger.record_fold(len(fold))
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp483 reporting-fold identity is duplicated")
    fold = fold.rename(columns={"suffix_offset": "reporting_suffix_offset"})
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if frame[["fold", "reporting_suffix_offset"]].isna().any().any():
        raise ValueError("exp483 reporting-fold attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["reporting_suffix_offset"].to_numpy(np.int64),
    ):
        raise ValueError("exp483 reporting-fold suffix identity mismatch")
    frame = frame.drop(columns=["reporting_suffix_offset"])

    hidden_spec = dict(get_nested(config, "data.hidden_like_assignment") or {})
    hidden_path = Path(saved_paths["hidden_like_assignment"])
    if sha256_path(hidden_path) != str(hidden_spec["expected_sha256"]):
        raise ValueError("exp483 hidden-like assignment raw SHA mismatch")
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
        raise ValueError("exp483 hidden-like assignment has duplicate wells")
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
            raise ValueError(f"exp483 hidden-like role counts changed for {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("exp483 hidden-like role attachment is incomplete")
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
        "saved_exp209_hmm",
        "candidate_hmm_50_50",
        "control_hmm_50_50",
        *PREDICTION_COLUMNS,
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("exp483 Stage 1 late readout contains non-finite values")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("exp483 reporting-fold set mismatch")
    return frame, {
        "truth_attached_after_prediction_freeze": True,
        "candidate_content_sha256_reverified": logical_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "saved_input_paths": dict(saved_paths),
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
        raise ValueError(f"exp483 Stage 1 metric scope is empty: {scope}")
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
                comparison="fixed_huber_filtering_vs_saved_exp404_scale5_x1p0",
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
                "well_missing_fraction": float(group["well_missing_fraction"].iloc[0]),
            }
        )
    return primary, pd.DataFrame(by_well_rows), blend


def _stage1_scope_row(metrics: pd.DataFrame, scope: str) -> pd.Series:
    selected = metrics.loc[metrics["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"exp483 expected one Stage 1 metric row for {scope}")
    return selected.iloc[0]


def evaluate_stage1_gate(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
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
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
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
        "seed_well_trajectories": 98944,
        "particle_starts": 49472000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    technical_checks = {
        "stage0_all_technical_gates_passed": bool(
            get_nested(config, "stage_0_result.all_technical_gates_passed")
        ),
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
        "finite_prediction_coverage": bool(
            np.isfinite(frame[list(PREDICTION_COLUMNS)].to_numpy(np.float64)).all()
        ),
        "truth_error_fold_hidden_reads_before_freeze_zero": bool(
            all(int(value) == 0 for value in before.values())
        ),
        "execution_count_match": execution_counts == expected_counts,
        "artifact_sha_readback": bool(frozen["sha_readback"]["pass"]),
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
        "runtime": bool(runtime_seconds <= float(get_nested(config, "runtime.maximum_seconds"))),
        "peak_rss": bool(rss_gb <= float(get_nested(config, "runtime.maximum_peak_rss_gb"))),
    }
    technical = {
        "checks": technical_checks,
        "passed": bool(all(technical_checks.values())),
        "execution_counts": execution_counts,
        "saved_control_rmse_absolute_difference": control_difference,
        "fixed_hmm_pf_50_50_control_rmse_absolute_difference": blend_control_difference,
        "runtime_seconds": runtime_seconds,
        "peak_rss_gb": rss_gb,
        "truth_access_ledger_at_freeze": dict(ledger_at_freeze),
    }
    scope_rules = {
        "raw_gr_observed": ("minimum_gain", "minimum_raw_gr_observed_gain_ft"),
        "raw_gr_missing": ("maximum_regression", "maximum_raw_gr_missing_regression_ft"),
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
            else "terminal_close_without_huber_or_pf_rescue"
        ),
        "technical_gate": technical,
        "primary_scientific_gate": primary_gate,
        "fixed_exp209_hmm_pf_50_50_guard": blend_guard,
        "failure_action": (
            "close_without_delta_scale_temperature_clip_mixture_particle_seed_"
            "transition_resampling_well_gate_blend_selector_or_same_oof_rescue"
        ),
    }


def run_stage1(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    scientific_contract = validate_scientific_contract(config)
    validate_execution_contract(config, require_run_approval=True)
    if not bool(get_nested(config, "execution.run_stage_1")):
        raise RuntimeError("exp483 Stage 1 is not selected")
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_manifest = validate_raw_well_identity(config, raw_dir)
    wells = list(raw_manifest["well_ids"])
    saved_paths = stage1_saved_input_paths(config)
    ledger = LeakageLedger(expected_wells=len(wells))
    formula = formula_unit_contract()
    no_op = no_op_toy_pf_contract()
    seed_report = stable_seed_contract(
        wells,
        int(get_nested(config, "model.fixed_from_exp404.seeds")),
    )
    if not (
        bool(formula["pass"])
        and bool(no_op["pass"])
        and bool(seed_report["pass"])
    ):
        raise RuntimeError("exp483 Stage 1 formula/no-op/seed contract failed")
    contract_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_scientific_contract.json",
        scientific_contract,
    )
    input_report = {
        "raw": raw_manifest,
        "formula_contract": formula,
        "no_op_toy_pf_contract": no_op,
        "stable_seed_contract": seed_report,
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
    warm_up_pf_kernel()
    results = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(delayed(decode_target_free_well)(well, raw_dir, config) for well in wells)
    candidate, audit, frozen = freeze_target_free_outputs(
        results,
        output,
        config,
        ledger,
        stage="stage1",
        expected_rows=int(get_nested(config, "validation.expected_rows")),
        expected_wells=int(get_nested(config, "validation.expected_wells")),
    )
    runtime_to_freeze = time.time() - started
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
    runtime_seconds = time.time() - started
    rss_gb = peak_rss_gb()
    gate = evaluate_stage1_gate(
        config,
        frame,
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
        "blend_metrics": output / f"{OUTPUT_PREFIX}_stage1_fixed_hmm_pf_50_50_metrics.csv",
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
            "runtime_seconds_total": runtime_seconds,
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
        "fixed_hmm_pf_50_50_candidate_rmse": float(blend_overall["candidate_rmse"]),
        "fixed_hmm_pf_50_50_control_rmse": float(blend_overall["control_rmse"]),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "prediction_sha256": frozen["prediction_logical_sha256"],
        "late_readout": late_report,
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
                "All-well train-side Stage 1. No raw-test inference, submission, "
                "or deterministic-anchor claim."
            ),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 12. Setup, configuration preview, and selected execution


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "scientific_parent": get_nested(CONFIG, "lineage.parent"),
                "implementation_reference": get_nested(
                    CONFIG, "lineage.exact_pf_implementation_reference"
                ),
                "changed_factor": SCIENTIFIC_CONTRACT["changed_factor"],
                "fixed_pf": SCIENTIFIC_CONTRACT["fixed_pf"],
                "stage_0": get_nested(CONFIG, "stages.stage_0"),
                "stage_1_approved": get_nested(
                    CONFIG, "execution.stage_1_execution_approved"
                ),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
                "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
                "control_pf_well_runs": 0,
                "hmm_well_runs": 0,
                "beam_well_runs": 0,
                "boosters": 0,
                "gpu_runs": 0,
                "scientific_contract_sha256": SCIENTIFIC_CONTRACT["scientific_contract_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


# %%
if EXECUTE_NOTEBOOK and bool(get_nested(CONFIG, "execution.run_stage_0")):
    SUMMARY = run_stage0(CONFIG)


# %%
if EXECUTE_NOTEBOOK and bool(get_nested(CONFIG, "execution.run_stage_1")):
    SUMMARY = run_stage1(CONFIG)
