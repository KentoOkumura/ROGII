# %% [markdown]
# # exp495 uncertainty-weighted exp226 rate observation HMM — Stage 0B
#
# Stage 0A failed its preregistered transfer-strength gate. The user explicitly
# authorized the unchanged fixed32 Stage 0B as an override. This notebook keeps
# the frozen prefix estimator and exact exp209 HMM fixed, runs one candidate on
# 32 wells, and reads roles, episodes, and suffix truth only after candidate
# predictions freeze. Stage 1, inference, and submission remain fail-closed.

# %% [markdown]
# ## Contents
# 1. Imports and immutable column contracts
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen Stage 0B override contract
# 4. Exp226 fold-safe geometry replay helpers
# 5. Known-prefix uncertainty and suffix schedule freeze helpers
# 6. Fixed32 identity, exact rate-observation HMM, and prediction freeze
# 7. Role/episode/truth late-join and Stage 0B gates
# 8. Kaggle CPU orchestration and generated artifacts

# %% [markdown]
# ## 1. Imports and immutable column contracts

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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

try:
    import numba as numba_module
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - Kaggle image and project venv include numba.
    numba_module = None
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(function: Any) -> Any:
            return function

        return decorator

    prange = range  # type: ignore[assignment]

    def set_num_threads(_: int) -> None:
        return None


EXPERIMENT_NAME = "exp495_uncertainty_weighted_exp226_rate_observation_hmm"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FILE_COLUMNS = ("X", "Y", "Z", "MD", "TVT_input")
TARGET_SAFE_COLUMNS = TARGET_FILE_COLUMNS
TARGET_FORBIDDEN_COLUMNS = {
    "TVT",
    "GR",
    "ANCC",
    "tvt_true",
    "tvt_pred",
    "gr_delta",
    "u_projection",
    "target",
    "error",
    "abs_error",
}
SAFE_OOF_COLUMNS = ["well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"]
PREFIX_TRANSITION_COLUMNS = [
    "well_id",
    "fold",
    "destination_row_idx",
    "transition_rank",
    "delta_md",
    "observed_u_rate",
    "geometry_u_rate",
    "rate_residual",
    "formula_parity_abs",
    "valid",
]
UNCERTAINTY_COLUMNS = [
    "well_id",
    "fold",
    "official_last_known_row",
    "replay_cut_row",
    "selected_transition_count",
    "valid_transition_count",
    "residual_median",
    "residual_mad",
    "sigma_226",
    "observation_enabled",
    "fallback_reason",
    "formula_parity_max_abs",
]
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
    "mu_226",
    "sigma_226",
    "observation_enabled",
    "geometry_fallback",
    "anchor_u",
    "formula_parity_max_abs",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP495_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Notebook-safe runtime, configuration, path, and SHA helpers


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


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            config = read_yaml(path)
            if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
                return config
    raise FileNotFoundError(f"exp495 config not found: {[str(path) for path in candidates]}")


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


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
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
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
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
        "logical_sha256": dataframe_content_sha256(frame),
        "schema_sha256": dataframe_schema_sha256(frame),
    }


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    report = inspect_gzip_csv(path)
    report.update(
        {
            "rows": len(frame),
            "logical_sha256": dataframe_content_sha256(frame),
            "schema_sha256": dataframe_schema_sha256(frame),
        }
    )
    return report


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw_candidate in candidates:
        candidate = Path(str(raw_candidate))
        possible = (
            candidate,
            candidate / filename,
            root / candidate,
            root / candidate / filename,
            PACKAGE_DIR / candidate,
            PACKAGE_DIR / candidate / filename,
        )
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
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    return configured if configured.is_absolute() else project_root() / configured


def artifact_dir() -> Path:
    output = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.is_dir()
        else project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


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
        "numba": str(getattr(numba_module, "__version__", "unavailable")),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024**3) if platform.system() == "Darwin" else value / (1024**2)


def list_horizontal_wells(data_dir: Path) -> list[str]:
    return [
        path.name.removesuffix("__horizontal_well.csv")
        for path in sorted(data_dir.glob("*__horizontal_well.csv"))
    ]


def validate_target_safe_frame(frame: pd.DataFrame) -> None:
    leaked = sorted(TARGET_FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"target-safe frame contains forbidden columns: {leaked}")
    missing = sorted(set(TARGET_SAFE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"target-safe frame is missing {missing}")


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(TARGET_FILE_COLUMNS))
    frame = frame.loc[:, list(TARGET_FILE_COLUMNS)]
    validate_target_safe_frame(frame)
    return frame


# %% [markdown]
# ## 3. Frozen Stage 0B override contract


# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool = False,
) -> dict[str, int]:
    fixed: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "authorization.implementation_approved": True,
        "authorization.canonical_notebook_adoption_approved": True,
        "authorization.kaggle_package_approved": True,
        "authorization.stage_0b_implementation_approved": True,
        "authorization.stage_0b_run_approved": True,
        "authorization.stage_0a_fail_closed_override_approved": True,
        "authorization.stage_1_implementation_approved": False,
        "implementation.enabled": True,
        "implementation.stage_0a_implemented": True,
        "implementation.stage_0b_implemented": True,
        "implementation.stage_1_implemented": False,
        "validation.expected_full_rows": 3783989,
        "validation.expected_full_wells": 773,
        "validation.stage_0a.diagnostic_variants": 1,
        "validation.stage_0a.hmm_well_runs": 0,
        "model.active_scientific_variant_count": 1,
        "model.parent_hmm_fixed.n_rates": 41,
        "model.parent_hmm_fixed.rate_span": 0.10,
        "model.parent_hmm_fixed.sig_r": 0.002,
        "model.parent_hmm_fixed.momentum": 0.998,
        "model.parent_hmm_fixed.sig_p": 0.02,
        "model.parent_hmm_fixed.step_ft": 0.35,
        "model.geometry_rate_center.source_column": "tvt_geop",
        "model.geometry_rate_center.coordinate": "u_equals_tvt_plus_z",
        "model.geometry_rate_center.use_exp226_final_prediction": False,
        "model.geometry_rate_center.use_exp226_gr_correction": False,
        "model.geometry_rate_center.use_exp226_u_projection": False,
        "model.uncertainty.tail_transitions": 128,
        "model.uncertainty.minimum_valid_transitions": 32,
        "model.uncertainty.floor": 0.002,
        "model.uncertainty.ceiling": None,
        "model.rate_observation.additional_lambda": "none",
        "model.rate_observation.temperature": 1.0,
        "model.rate_observation.clipping": "none",
        "model.rate_observation.activation_gate": "none",
        "prefix_uncertainty.maximum_transitions": 128,
        "prefix_uncertainty.minimum_valid_transitions": 32,
        "prefix_uncertainty.mad_multiplier": 1.4826,
        "prefix_uncertainty.floor": 0.002,
        "prefix_uncertainty.bias_correction": "none",
        "geometry_replay.use_exp226_gr_correction": False,
        "geometry_replay.use_exp226_u_projection": False,
        "execution.run_stage_0a": False,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "execution.parent_control_hmm_reruns": 0,
        "execution.fitted_ml_models": 0,
        "execution.lightgbm_configs": 0,
        "execution.trained_ml_folds": 0,
        "execution.boosters": 0,
        "execution.pf_runs": 0,
        "execution.beam_runs": 0,
        "execution.gpu_runs": 0,
        "runtime.enable_gpu": False,
        "runtime.enable_internet": False,
        "runtime.inference_enabled": False,
        "runtime.submission_enabled": False,
        "gates.stage_0a_technical.expected_wells": 773,
        "gates.stage_0a_technical.expected_suffix_rows": 3783989,
        "gates.stage_0a_technical.expected_folds": 5,
        "gates.stage_0a_technical.formula_parity_max_abs": 1.0e-10,
        "gates.stage_0a_mechanism.sigma_vs_suffix_abs_geometry_rate_error_spearman_min": 0.20,
        "gates.stage_0a_mechanism.positive_spearman_folds_min": 4,
        "gates.stage_0a_mechanism.low_sigma_half_rate_rmse_gain_vs_high_sigma_half_min_fraction": 0.10,
        "gates.stage_0a_mechanism.low_sigma_half_exp355_schedule_gain_vs_exp209_constant_min_fraction": 0.05,
        "gates.stage_0a_mechanism.improving_schedule_folds_min": 4,
        "gates.stage_0a_mechanism.prefix_fallback_well_fraction_max": 0.05,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(
                f"exp495 fixed contract mismatch: {key}={actual!r}, expected {expected!r}"
            )
    allowed_statuses = {
        "stage_0b_override_approved_pending_kaggle_cpu_run",
        "stage_0b_completed_fail_closed",
    }
    if str(get_nested(config, "experiment.status")) not in allowed_statuses:
        raise ValueError("exp495 Stage 0B status is outside the frozen pre/post-run states")
    run_stage_0b = bool(get_nested(config, "execution.run_stage_0b"))
    if run_stage_0b != (
        str(get_nested(config, "experiment.status"))
        == "stage_0b_override_approved_pending_kaggle_cpu_run"
    ):
        raise ValueError("exp495 Stage 0B run flag and pre/post-run status differ")
    if list(get_nested(config, "validation.expected_folds")) != [0, 1, 2, 3, 4]:
        raise ValueError("exp495 fixes fold identity to [0, 1, 2, 3, 4]")
    if list(get_nested(config, "data.exp226_geometry.allowed_geometry_fields")) != SAFE_OOF_COLUMNS:
        raise ValueError("exp495 strict exp226 allowlist changed")
    forbidden = set(get_nested(config, "data.exp226_geometry.forbidden_fields", []))
    if not {"tvt_pred", "TVT", "tvt_true", "gr_delta", "error", "abs_error"}.issubset(forbidden):
        raise ValueError("exp495 exp226 forbidden-column contract is incomplete")
    if require_run_authorization and not (
        bool(get_nested(config, "authorization.canonical_notebook_adoption_approved"))
        and bool(get_nested(config, "authorization.kaggle_package_approved"))
        and bool(get_nested(config, "authorization.stage_0b_implementation_approved"))
        and bool(get_nested(config, "authorization.stage_0b_run_approved"))
        and bool(get_nested(config, "authorization.stage_0a_fail_closed_override_approved"))
        and run_stage_0b
    ):
        raise RuntimeError(
            "exp495 Stage 0B requires the explicit Stage 0A fail-closed override, "
            "canonical-notebook, Kaggle-package, implementation, and run authorization"
        )
    return {
        "scientific_variants": 1,
        "hmm_well_runs": 32,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "gpu_runs": 0,
    }


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage_0a_reliability_identifiability_zero_hmm",
        "truth_attached": False,
        "lineage": get_nested(config, "lineage"),
        "geometry_replay": get_nested(config, "geometry_replay"),
        "prefix_uncertainty": get_nested(config, "prefix_uncertainty"),
        "geometry_rate_center": get_nested(config, "model.geometry_rate_center"),
        "rate_observation": get_nested(config, "model.rate_observation"),
        "technical_gate": get_nested(config, "gates.stage_0a_technical"),
        "mechanism_gate": get_nested(config, "gates.stage_0a_mechanism"),
        "forbidden_changes": get_nested(config, "forbidden_changes"),
        "execution_counts": validate_scientific_contract(config),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def build_stage0b_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage_0b_fixed32_user_override",
        "stage_0a_result_preserved": "FAIL",
        "stage_0a_fail_closed_override": True,
        "fixed32": get_nested(config, "validation.stage_0b"),
        "parent_hmm_fixed": get_nested(config, "model.parent_hmm_fixed"),
        "geometry_rate_center": get_nested(config, "model.geometry_rate_center"),
        "uncertainty": get_nested(config, "model.uncertainty"),
        "rate_observation": get_nested(config, "model.rate_observation"),
        "technical_gate": get_nested(config, "gates.stage_0b_technical"),
        "mechanism_gate": get_nested(config, "gates.stage_0b_mechanism"),
        "execution": {
            "scientific_variants": 1,
            "candidate_hmm_well_runs": 32,
            "parent_control_hmm_reruns": 0,
            "fitted_ml_models": 0,
            "boosters": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
        "stage_1_automatic": False,
        "inference": False,
        "submission": False,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Exp226 fold-safe geometry replay helpers


# %%
@dataclass(frozen=True)
class K16Params:
    theta0: float = 118.4
    k_segments: int = 16
    local_linear_k: int = 50
    local_linear_bandwidth: float = 500.0
    local_linear_ridge: float = 1.0
    smooth_rho: float = 10.0
    gate: float = 0.35
    field_min_proj: float = 0.3
    kbins: tuple[float, ...] = (0.0, 750.0, 1500.0, 2500.0, 4000.0, 1.0e18)
    rot_max_deg: float = 60.0
    ancc_theta_bandwidth: float = 1500.0

    @property
    def n_bins(self) -> int:
        return len(self.kbins) - 1

    @property
    def kappa_dim(self) -> int:
        return 2 * self.n_bins + 2


@dataclass
class GeometryWell:
    wid: str
    wi: int
    s: int
    n: int
    ndz: np.ndarray
    anchor: float
    ti: np.ndarray
    segid: np.ndarray
    mid: np.ndarray
    proj: np.ndarray
    az: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    md: np.ndarray
    anc: np.ndarray | None = None
    c_raw: np.ndarray | None = None
    c_sm: np.ndarray | None = None


@dataclass
class FieldPack:
    f_raw: np.ndarray
    f_sm: np.ndarray
    surface_points: np.ndarray
    global_theta: float


def params_from_config(config: Mapping[str, Any]) -> K16Params:
    params = get_nested(config, "geometry_replay.params", {})
    return K16Params(
        theta0=float(params["theta0"]),
        k_segments=int(params["k_segments"]),
        local_linear_k=int(params["local_linear_k"]),
        local_linear_bandwidth=float(params["local_linear_bandwidth"]),
        local_linear_ridge=float(params["local_linear_ridge"]),
        smooth_rho=float(params["smooth_rho"]),
        gate=float(params["gate"]),
        field_min_proj=float(params["field_min_proj"]),
        kbins=tuple(float(value) for value in params["kbins"]),
        rot_max_deg=float(params["rot_max_deg"]),
        ancc_theta_bandwidth=float(params["ancc_theta_bandwidth"]),
    )


def last_contiguous_known_index(values: np.ndarray) -> int:
    finite = np.isfinite(np.asarray(values, dtype=np.float64))
    if not len(finite) or not finite[0]:
        raise ValueError("well has no contiguous TVT_input prefix from row zero")
    missing = np.flatnonzero(~finite)
    end = len(finite) - 1 if not len(missing) else int(missing[0] - 1)
    if finite[end + 1 :].any():
        raise ValueError("TVT_input has finite rows after the contiguous prefix")
    return int(end)


def segment_geometry(
    x: np.ndarray,
    y: np.ndarray,
    s: int,
    n: int,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, n, params.k_segments + 1)
    step_idx = np.arange(1, n + 1.0)
    segid = np.clip(np.searchsorted(edges[1:], step_idx, side="left"), 0, params.k_segments - 1)
    mid = np.empty((params.k_segments, 2), dtype=np.float64)
    proj = np.empty(params.k_segments, dtype=np.float64)
    az = np.empty(params.k_segments, dtype=np.float64)
    theta = np.radians(params.theta0)
    last_idx = len(x) - 1
    for segment in range(params.k_segments):
        first = min(s + 1 + int(edges[segment]), last_idx)
        final_raw = s + 1 + max(int(edges[segment + 1]) - 1, int(edges[segment]))
        final = min(max(final_raw, first), last_idx)
        az[segment] = np.arctan2(y[final] - y[first], x[final] - x[first])
        mid[segment] = ((x[first] + x[final]) / 2.0, (y[first] + y[final]) / 2.0)
        proj[segment] = np.cos(az[segment] - theta)
    return segid.astype(np.int64), mid, proj, az


def fit_coeffs(r0: np.ndarray, u: np.ndarray, n: int, params: K16Params, rho: float) -> np.ndarray:
    positions = np.arange(1, n + 1.0)
    edges = np.linspace(0, n, params.k_segments + 1)
    phi = np.column_stack(
        [
            np.clip(positions - edges[index], 0, edges[index + 1] - edges[index])
            for index in range(params.k_segments)
        ]
    )
    matrix = phi.T @ phi
    if rho > 0:
        difference = np.diff(np.eye(params.k_segments), axis=0)
        scale = float(np.mean(np.diag(matrix))) if matrix.size else 1.0
        matrix = matrix + rho * max(scale, 1.0e-9) * difference.T @ difference
    rhs = phi.T @ (r0 - u)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix + np.eye(params.k_segments) * 1.0e-9, rhs, rcond=None)[0]


def load_source_geometry_well(path: Path, params: K16Params, wi: int) -> GeometryWell:
    columns = ["X", "Y", "Z", "MD", "TVT", "TVT_input", "ANCC"]
    frame = pd.read_csv(path, usecols=columns)
    x = frame["X"].to_numpy(np.float64)
    y = frame["Y"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    md = frame["MD"].to_numpy(np.float64)
    tvt = frame["TVT"].to_numpy(np.float64)
    ti = frame["TVT_input"].to_numpy(np.float64)
    s = last_contiguous_known_index(ti)
    ndz = -np.diff(z)[s:]
    n = len(ndz)
    if n <= 0:
        raise ValueError(f"source well {path.name} has no original suffix")
    r0 = tvt[s + 1 :] - tvt[s]
    u = np.cumsum(ndz)
    segid, mid, proj, az = segment_geometry(x, y, s, n, params)
    return GeometryWell(
        wid=path.name.split("__")[0],
        wi=int(wi),
        s=s,
        n=n,
        ndz=ndz,
        anchor=float(tvt[s]),
        ti=ti,
        segid=segid,
        mid=mid,
        proj=proj,
        az=az,
        x=x,
        y=y,
        z=z,
        md=md,
        anc=frame["ANCC"].to_numpy(np.float64),
        c_raw=fit_coeffs(r0, u, n, params, rho=0.0),
        c_sm=fit_coeffs(r0, u, n, params, rho=params.smooth_rho),
    )


def build_target_geometry_well(
    well: str,
    masked_frame: pd.DataFrame,
    *,
    cut: int,
    params: K16Params,
) -> GeometryWell:
    validate_target_safe_frame(masked_frame)
    if masked_frame.loc[cut + 1 :, "TVT_input"].notna().any():
        raise ValueError("target geometry reader received unmasked post-cut TVT_input")
    x = pd.to_numeric(masked_frame["X"], errors="raise").to_numpy(np.float64)
    y = pd.to_numeric(masked_frame["Y"], errors="raise").to_numpy(np.float64)
    z = pd.to_numeric(masked_frame["Z"], errors="raise").to_numpy(np.float64)
    md = pd.to_numeric(masked_frame["MD"], errors="raise").to_numpy(np.float64)
    ti = pd.to_numeric(masked_frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    n = len(masked_frame) - cut - 1
    ndz = -np.diff(z)[cut:]
    if len(ndz) != n or n <= 0:
        raise ValueError("pseudo geometry must extend from cut through the well end")
    segid, mid, proj, az = segment_geometry(x, y, cut, n, params)
    return GeometryWell(
        wid=str(well),
        wi=-1,
        s=int(cut),
        n=n,
        ndz=ndz,
        anchor=float(ti[cut]),
        ti=ti,
        segid=segid,
        mid=mid,
        proj=proj,
        az=az,
        x=x,
        y=y,
        z=z,
        md=md,
    )


def build_fields(wells: Sequence[GeometryWell], params: K16Params) -> FieldPack:
    def pack(key: str) -> np.ndarray:
        rows: list[tuple[float, float, float, float]] = []
        for well in wells:
            coeffs = getattr(well, key)
            if coeffs is None:
                continue
            for segment in range(params.k_segments):
                if abs(well.proj[segment]) > params.field_min_proj:
                    rows.append(
                        (
                            well.mid[segment, 0],
                            well.mid[segment, 1],
                            coeffs[segment] / well.proj[segment],
                            float(well.wi),
                        )
                    )
        if not rows:
            raise ValueError("empty exp226 donor field")
        return np.asarray(rows, dtype=np.float64)

    surface_parts: list[np.ndarray] = []
    for well in wells:
        if well.anc is None:
            continue
        step = max(len(well.x) // 120, 1)
        anc = well.anc[::step]
        surface_parts.append(
            np.column_stack(
                [
                    well.x[::step],
                    well.y[::step],
                    anc,
                    np.full(len(anc), well.wi, dtype=np.float64),
                ]
            )
        )
    if not surface_parts:
        raise ValueError("empty ANCC donor surface")
    surface = np.vstack(surface_parts)
    surface = surface[np.isfinite(surface[:, 2])]
    centered = np.column_stack(
        [
            np.ones(len(surface)),
            surface[:, 0] - surface[:, 0].mean(),
            surface[:, 1] - surface[:, 1].mean(),
        ]
    )
    beta = np.linalg.lstsq(centered, surface[:, 2], rcond=None)[0]
    return FieldPack(
        f_raw=pack("c_raw"),
        f_sm=pack("c_sm"),
        surface_points=surface,
        global_theta=float(np.arctan2(beta[2], beta[1])),
    )


def _safe_nearest_indices(dist2: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    if len(candidates) == 0:
        return candidates
    count = min(max(int(k), 1), len(candidates))
    return candidates[np.argpartition(dist2[candidates], count - 1)[:count]]


def local_linear(
    field: np.ndarray,
    own_wi: int,
    mid: np.ndarray,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray]:
    keep = field[:, 3] != own_wi
    fx, fy, values = field[keep, 0], field[keep, 1], field[keep, 2]
    drift = np.empty(len(mid), dtype=np.float64)
    distance = np.empty(len(mid), dtype=np.float64)
    for index, point in enumerate(mid):
        dist2 = (fx - point[0]) ** 2 + (fy - point[1]) ** 2
        selected = _safe_nearest_indices(dist2, np.arange(len(dist2)), params.local_linear_k)
        weights = np.exp(
            np.maximum(-dist2[selected] / (2.0 * params.local_linear_bandwidth**2), -700)
        )
        dx = (fx[selected] - point[0]) / 1000.0
        dy = (fy[selected] - point[1]) / 1000.0
        design = np.column_stack([np.ones(len(selected)), dx, dy])
        ridge = params.local_linear_ridge * np.sum(weights) * np.diag([0.0, 1.0, 1.0])
        matrix = (design * weights[:, None]).T @ design + ridge
        rhs = (design * weights[:, None]).T @ values[selected]
        try:
            drift[index] = np.linalg.solve(matrix, rhs)[0]
        except np.linalg.LinAlgError:
            drift[index] = np.linalg.lstsq(matrix + np.eye(3) * 1.0e-9, rhs, rcond=None)[0][0]
        distance[index] = float(
            np.sqrt(np.median(np.sort(dist2[selected])[: min(15, len(selected))]))
        )
    return drift, distance


def kernel_mean(field: np.ndarray, own_wi: int, mid: np.ndarray) -> np.ndarray:
    keep = field[:, 3] != own_wi
    fx, fy, values = field[keep, 0], field[keep, 1], field[keep, 2]
    output = np.empty(len(mid), dtype=np.float64)
    for index, point in enumerate(mid):
        dist2 = (fx - point[0]) ** 2 + (fy - point[1]) ** 2
        selected = _safe_nearest_indices(dist2, np.arange(len(dist2)), 15)
        weights = np.exp(np.maximum(-dist2[selected] / (2.0 * 500.0**2), -700))
        output[index] = float(np.sum(weights * values[selected]) / np.sum(weights))
    return output


def theta_loc_at(
    surface: np.ndarray,
    mids: np.ndarray,
    own_wi: int,
    global_theta: float,
    params: K16Params,
) -> np.ndarray:
    output = np.empty(len(mids), dtype=np.float64)
    bandwidth = params.ancc_theta_bandwidth
    for index, point in enumerate(mids):
        dist2 = (surface[:, 0] - point[0]) ** 2 + (surface[:, 1] - point[1]) ** 2
        mask = (dist2 < (4 * bandwidth) ** 2) & (surface[:, 3] != own_wi)
        if int(mask.sum()) < 30:
            output[index] = global_theta
            continue
        weights = np.exp(-dist2[mask] / (2 * bandwidth**2))
        x = surface[mask, 0] - point[0]
        y = surface[mask, 1] - point[1]
        z = surface[mask, 2]
        matrix = np.array(
            [
                [np.sum(weights), np.sum(weights * x), np.sum(weights * y)],
                [np.sum(weights * x), np.sum(weights * x * x), np.sum(weights * x * y)],
                [np.sum(weights * y), np.sum(weights * x * y), np.sum(weights * y * y)],
            ]
        )
        rhs = np.array([np.sum(weights * z), np.sum(weights * x * z), np.sum(weights * y * z)])
        try:
            beta = np.linalg.solve(matrix, rhs)
            output[index] = np.arctan2(beta[2], beta[1])
        except np.linalg.LinAlgError:
            output[index] = global_theta
    return output


def committee_inputs(
    well: GeometryWell, fields: FieldPack, params: K16Params
) -> tuple[np.ndarray, np.ndarray] | None:
    if not (np.abs(well.proj) < params.gate).any():
        return None
    theta = theta_loc_at(fields.surface_points, well.mid, well.wi, fields.global_theta, params)
    rotation = np.degrees(
        np.abs(
            np.arctan2(
                np.sin(theta - np.radians(params.theta0)),
                np.cos(theta - np.radians(params.theta0)),
            )
        )
    )
    drift = kernel_mean(fields.f_raw, well.wi, well.mid)
    local = drift * np.cos(well.az - theta)
    mask = (np.abs(well.proj[well.segid]) < params.gate) & (rotation < params.rot_max_deg)[
        well.segid
    ]
    return local, mask


def build_columns(
    well: GeometryWell,
    raw_field: np.ndarray,
    smooth_field: np.ndarray,
    donor_distance: np.ndarray,
    params: K16Params,
    substitute: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    gated = np.abs(well.proj[well.segid]) < params.gate
    raw_step = np.where(gated, 0.0, well.ndz + (raw_field * well.proj)[well.segid])
    smooth_step = np.where(gated, 0.0, well.ndz + (smooth_field * well.proj)[well.segid])
    bucket = np.digitize(donor_distance, params.kbins[1:-1])[well.segid]
    position = (well.segid + 0.5) / params.k_segments
    columns = [
        np.cumsum(np.where(bucket == index, raw_step, 0.0)) for index in range(params.n_bins)
    ]
    columns += [
        np.cumsum(np.where(bucket == index, smooth_step, 0.0)) for index in range(params.n_bins)
    ]
    columns.append(np.cumsum(0.5 * (raw_step + smooth_step) * np.sqrt(position)))
    if substitute is None:
        columns.append(np.zeros(well.n, dtype=np.float64))
    else:
        columns.append(
            np.cumsum(np.where(substitute[1], well.ndz + substitute[0][well.segid], 0.0))
        )
    return np.column_stack(columns)


def replay_exp226_geometry(
    target: GeometryWell,
    fields: FieldPack,
    kappa: np.ndarray,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray]:
    raw_field, donor_distance = local_linear(fields.f_raw, target.wi, target.mid, params)
    smooth_field, _ = local_linear(fields.f_sm, target.wi, target.mid, params)
    substitute = committee_inputs(target, fields, params)
    design = build_columns(target, raw_field, smooth_field, donor_distance, params, substitute)
    if design.shape != (target.n, params.kappa_dim) or len(kappa) != params.kappa_dim:
        raise ValueError("exp226 geometry replay design/kappa shape mismatch")
    path = target.anchor + design @ np.asarray(kappa, dtype=np.float64)
    row_distance = donor_distance[target.segid]
    if not np.isfinite(path).all() or not np.isfinite(row_distance).all():
        raise ValueError("exp226 pseudo geometry or donor distance is not finite")
    return path, row_distance


def load_exp226_fold_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, int], dict[int, np.ndarray], list[dict[str, Any]], Path]:
    oof_spec = get_nested(config, "data.exp226_geometry")
    oof_path = resolve_existing(
        str(oof_spec["filename"]), [str(value) for value in oof_spec["candidates"]]
    )
    actual_decompressed = sha256_gzip_decompressed(oof_path)
    if actual_decompressed != str(oof_spec["expected_oof_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    safe_columns = [str(value) for value in oof_spec["allowed_geometry_fields"]]
    if "tvt_true" in safe_columns or "tvt_pred" in safe_columns or "gr_delta" in safe_columns:
        raise ValueError("exp226 target-free OOF allowlist contains forbidden target columns")
    fold_rows = pd.read_csv(oof_path, usecols=["well_id", "fold"], dtype={"well_id": str})
    if len(fold_rows) != int(get_nested(config, "validation.expected_full_rows")):
        raise ValueError("exp226 OOF row count mismatch")
    per_well = fold_rows.drop_duplicates().sort_values("well_id", kind="mergesort")
    if per_well["well_id"].duplicated().any():
        raise ValueError("exp226 OOF maps one well to multiple folds")
    fold_by_well = {str(row.well_id): int(row.fold) for row in per_well.itertuples(index=False)}

    kappa_spec = get_nested(config, "data.exp226_kappa_by_fold")
    kappa_path = resolve_existing(
        str(kappa_spec["filename"]), [str(value) for value in kappa_spec["candidates"]]
    )
    if sha256_path(kappa_path) != str(kappa_spec["expected_sha256"]):
        raise ValueError("exp226 kappa-by-fold SHA mismatch")
    kappa_frame = pd.read_csv(kappa_path)
    kappa_by_fold: dict[int, np.ndarray] = {}
    expected_terms = [
        *[f"raw_bin_{index}" for index in range(5)],
        *[f"smooth_bin_{index}" for index in range(5)],
        "sqrt_position",
        "near_strike_committee",
    ]
    for fold_label, part in kappa_frame.groupby("fold", sort=True):
        fold = int(str(fold_label).replace("fold", ""))
        if len(part) != int(kappa_spec["expected_terms"]):
            raise ValueError(f"exp226 fold {fold} kappa term count mismatch")
        if part["term"].astype(str).tolist() != expected_terms:
            raise ValueError(f"exp226 fold {fold} kappa term order mismatch")
        kappa_by_fold[fold] = part["value"].to_numpy(np.float64)
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(kappa_by_fold) != expected_folds:
        raise ValueError("exp226 kappa fold set mismatch")
    manifests = [
        {
            "name": "exp226_oof_fold_identity",
            "path": str(oof_path),
            "raw_sha256": sha256_path(oof_path),
            "decompressed_sha256": actual_decompressed,
            "rows": len(fold_rows),
            "wells": len(fold_by_well),
        },
        {
            "name": "exp226_kappa_by_fold",
            "path": str(kappa_path),
            "raw_sha256": sha256_path(kappa_path),
            "decompressed_sha256": "",
            "rows": len(kappa_frame),
            "wells": 0,
        },
    ]
    return fold_by_well, kappa_by_fold, manifests, oof_path


# %% [markdown]
# ## 5. Known-prefix uncertainty and suffix schedule freeze helpers


# %%
class IneligiblePrefixError(ValueError):
    """A fixed-contract prefix fallback, not a pipeline defect."""


def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.removesuffix("__horizontal_well.csv")
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
    manifest = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    actual_sha = dataframe_content_sha256(
        manifest, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    )
    if len(manifest) != int(get_nested(config, "validation.expected_full_wells")):
        raise ValueError("raw train well count mismatch")
    if actual_sha != str(get_nested(config, "data.expected_raw_well_identity_sha256")):
        raise ValueError("raw train well identity SHA mismatch")
    return manifest, {
        "path": str(raw_dir),
        "wells": len(manifest),
        "logical_sha256": actual_sha,
    }


def load_exp226_geometry(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    spec = get_nested(config, "data.exp226_geometry")
    path = resolve_existing(str(spec["filename"]), [str(item) for item in spec["candidates"]])
    inspection = inspect_gzip_csv(path)
    if inspection["decompressed_sha256"] != str(spec["expected_oof_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    header = pd.read_csv(path, nrows=0)
    allowlist = [str(item) for item in spec["allowed_geometry_fields"]]
    if allowlist != SAFE_OOF_COLUMNS:
        raise ValueError("exp226 OOF allowlist differs from the frozen contract")
    missing = sorted(set(allowlist) - set(header.columns))
    if missing:
        raise ValueError(f"exp226 OOF misses allowlisted fields: {missing}")
    frame = pd.read_csv(
        path,
        usecols=allowlist,
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
        raise ValueError("exp226 OOF has duplicate well/row identity")
    if len(frame) != int(get_nested(config, "validation.expected_full_rows")):
        raise ValueError("exp226 OOF row count mismatch")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_full_wells")):
        raise ValueError("exp226 OOF well count mismatch")
    if sorted(int(value) for value in frame["fold"].unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("exp226 OOF fold set mismatch")
    if not frame.groupby("well_id", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("exp226 OOF maps a well to multiple folds")
    if not np.isfinite(frame["tvt_geop"].to_numpy(np.float64)).all():
        raise ValueError("exp226 OOF tvt_geop contains non-finite values")
    inspection.update(
        {
            "rows_loaded": len(frame),
            "wells_loaded": int(frame["well_id"].nunique()),
            "selected_columns": allowlist,
            "forbidden_columns_loaded": [],
            "safe_logical_sha256": dataframe_content_sha256(frame),
            "safe_schema_sha256": dataframe_schema_sha256(frame),
        }
    )
    return frame, inspection, path


def validate_exp209_dependency(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = get_nested(config, "data.exp209_parent")
    path = resolve_existing(str(spec["filename"]), [str(item) for item in spec["candidates"]])
    report = inspect_gzip_csv(path)
    if report["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp209 parent cache decompressed SHA mismatch")
    if report["data_rows"] != int(get_nested(config, "validation.expected_full_rows")):
        raise ValueError("exp209 parent cache row count mismatch")
    report["usage"] = "dependency_sha_only_parent_hmm_not_read_or_rerun"
    return report


def valid_prefix_transition_destinations(
    frame: pd.DataFrame,
    *,
    maximum_transitions: int,
) -> tuple[int, np.ndarray]:
    validate_target_safe_frame(frame)
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float64)
    last_known = last_contiguous_known_index(tvt_input)
    destination = np.arange(1, last_known + 1, dtype=np.int64)
    dmd = np.diff(md[: last_known + 1])
    observed_u = tvt_input[: last_known + 1] + z[: last_known + 1]
    du = np.diff(observed_u)
    valid = np.isfinite(dmd) & np.isfinite(du) & (dmd > 0.0)
    selected = destination[valid][-int(maximum_transitions) :]
    return last_known, selected


def build_prefix_mask(
    well: str,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    maximum = int(get_nested(config, "prefix_uncertainty.maximum_transitions"))
    last_known, destinations = valid_prefix_transition_destinations(
        frame, maximum_transitions=maximum
    )
    if len(destinations) == 0:
        raise IneligiblePrefixError("no_finite_positive_delta_md_prefix_transition")
    cut = int(destinations.min() - 1)
    if cut < 0 or not math.isfinite(float(frame.loc[cut, "TVT_input"])):
        raise IneligiblePrefixError("missing_replay_anchor")
    masked = frame.copy()
    masked.loc[cut + 1 :, "TVT_input"] = np.nan
    if masked.loc[cut + 1 :, "TVT_input"].notna().any():
        raise RuntimeError("prefix mask retained target TVT_input after replay cut")
    manifest = {
        "well_id": str(well),
        "official_last_known_row": int(last_known),
        "replay_cut_row": cut,
        "selected_transition_count": int(len(destinations)),
        "full_replay_rows": int(len(frame) - cut - 1),
        "post_cut_tvt_input_finite_rows_after_mask": 0,
        "suffix_truth_reads_before_freeze": 0,
        "target_well_in_donor_field": False,
    }
    return masked, destinations, manifest


def coordinate_step_rates(
    tvt: Sequence[float],
    z: Sequence[float],
    md: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    tvt_values = np.asarray(tvt, dtype=np.float64)
    z_values = np.asarray(z, dtype=np.float64)
    md_values = np.asarray(md, dtype=np.float64)
    if not (len(tvt_values) == len(z_values) == len(md_values)):
        raise ValueError("TVT/Z/MD length mismatch")
    dmd = np.diff(md_values)
    direct = (np.diff(tvt_values) + np.diff(z_values)) / dmd
    coordinate = np.diff(tvt_values + z_values) / dmd
    valid = np.isfinite(dmd) & np.isfinite(direct) & np.isfinite(coordinate) & (dmd > 0.0)
    parity = float(np.max(np.abs(direct[valid] - coordinate[valid]))) if valid.any() else 0.0
    output = np.full(len(dmd), np.nan, dtype=np.float64)
    output[valid] = coordinate[valid]
    return output, valid, parity


def robust_prefix_uncertainty(
    residuals: Sequence[float],
    *,
    minimum_valid: int,
    multiplier: float,
    floor: float,
) -> dict[str, Any]:
    values = np.asarray(residuals, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) < minimum_valid:
        return {
            "valid_transition_count": int(len(finite)),
            "residual_median": float("nan"),
            "residual_mad": float("nan"),
            "sigma_226": float(floor),
            "observation_enabled": False,
            "fallback_reason": "insufficient_valid_prefix_transitions",
        }
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    sigma = max(float(floor), float(multiplier) * mad)
    return {
        "valid_transition_count": int(len(finite)),
        "residual_median": center,
        "residual_mad": mad,
        "sigma_226": sigma,
        "observation_enabled": True,
        "fallback_reason": "none",
    }


def build_prefix_transition_rows(
    well: str,
    fold: int,
    safe_frame: pd.DataFrame,
    target: GeometryWell,
    path: np.ndarray,
    destinations: np.ndarray,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cut = int(manifest["replay_cut_row"])
    last_known = int(manifest["official_last_known_row"])
    if target.n != int(manifest["full_replay_rows"]) or len(path) != target.n:
        raise ValueError("prefix geometry replay did not extend through the well end")
    replay_rows = np.arange(cut + 1, len(safe_frame), dtype=np.int64)
    if len(replay_rows) != len(path):
        raise ValueError("prefix replay row/path identity mismatch")
    prefix_rows = np.arange(cut, last_known + 1, dtype=np.int64)
    predicted_tvt = np.r_[float(safe_frame.loc[cut, "TVT_input"]), path[: last_known - cut]]
    observed_tvt = safe_frame.loc[prefix_rows, "TVT_input"].to_numpy(np.float64)
    z = safe_frame.loc[prefix_rows, "Z"].to_numpy(np.float64)
    md = safe_frame.loc[prefix_rows, "MD"].to_numpy(np.float64)
    geometry_rate, geometry_valid, geometry_parity = coordinate_step_rates(predicted_tvt, z, md)
    observed_rate, observed_valid, observed_parity = coordinate_step_rates(observed_tvt, z, md)
    relative_destinations = destinations - (cut + 1)
    selected_geometry = geometry_rate[relative_destinations]
    selected_observed = observed_rate[relative_destinations]
    dmd = np.diff(md)[relative_destinations]
    valid = (
        geometry_valid[relative_destinations]
        & observed_valid[relative_destinations]
        & np.isfinite(selected_geometry)
        & np.isfinite(selected_observed)
        & (dmd > 0.0)
    )
    transitions = pd.DataFrame(
        {
            "well_id": str(well),
            "fold": int(fold),
            "destination_row_idx": destinations.astype(np.int32),
            "transition_rank": np.arange(1, len(destinations) + 1, dtype=np.int16),
            "delta_md": dmd,
            "observed_u_rate": selected_observed,
            "geometry_u_rate": selected_geometry,
            "rate_residual": selected_geometry - selected_observed,
            "formula_parity_abs": np.maximum(geometry_parity, observed_parity),
            "valid": valid,
        }
    )
    selected_residuals = transitions.loc[transitions["valid"], "rate_residual"].to_numpy(np.float64)
    uncertainty = robust_prefix_uncertainty(
        selected_residuals,
        minimum_valid=int(get_nested(config, "prefix_uncertainty.minimum_valid_transitions")),
        multiplier=float(get_nested(config, "prefix_uncertainty.mad_multiplier")),
        floor=float(get_nested(config, "prefix_uncertainty.floor")),
    )
    summary = {
        "well_id": str(well),
        "fold": int(fold),
        "official_last_known_row": last_known,
        "replay_cut_row": cut,
        "selected_transition_count": int(len(destinations)),
        **uncertainty,
        "formula_parity_max_abs": max(geometry_parity, observed_parity),
    }
    return transitions, summary


def exp209_initial_rate(horizontal: pd.DataFrame, tail_n: int = 30) -> dict[str, Any]:
    known = horizontal.loc[horizontal["TVT_input"].notna(), ["MD", "Z", "TVT_input"]]
    tail = known.tail(tail_n)
    rate, valid, parity = coordinate_step_rates(tail["TVT_input"], tail["Z"], tail["MD"])
    finite = rate[valid]
    return {
        "initial_rate": float(np.median(finite)) if len(finite) >= 3 else 0.0,
        "valid_steps": int(len(finite)),
        "fallback": bool(len(finite) < 3),
        "formula_parity_max_abs": parity,
    }


def k16_segment_ids(n_rows: int, k_segments: int = 16) -> np.ndarray:
    if n_rows <= 0 or k_segments <= 0:
        raise ValueError("K16 segmentation requires positive rows and segment count")
    edges = np.linspace(0.0, float(n_rows), k_segments + 1)
    step_idx = np.arange(1.0, n_rows + 1.0)
    return np.clip(np.searchsorted(edges[1:], step_idx, side="left"), 0, k_segments - 1).astype(
        np.int16
    )


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
    for segment in range(k_segments):
        selected = step_rate[valid & (destination_segment == segment) & np.isfinite(step_rate)]
        counts[segment] = len(selected)
        if len(selected):
            rates[segment] = float(np.median(selected))
    return rates, counts


def validate_suffix_alignment(
    well: str,
    geometry: pd.DataFrame,
    horizontal: pd.DataFrame,
) -> np.ndarray:
    row_idx = geometry["row_idx"].to_numpy(np.int64)
    suffix_offset = geometry["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(geometry), dtype=np.int64)):
        raise ValueError(f"{well} suffix_offset is not contiguous")
    unknown_idx = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    if not np.array_equal(row_idx, unknown_idx):
        raise ValueError(f"{well} exp226 OOF rows differ from raw unknown suffix")
    if len(row_idx) == 0 or row_idx[0] == 0:
        raise ValueError(f"{well} has no known-prefix anchor")
    return row_idx


def build_well_suffix_schedule(
    well: str,
    geometry: pd.DataFrame,
    horizontal: pd.DataFrame,
    uncertainty: Mapping[str, Any],
    *,
    k_segments: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    geometry = geometry.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = validate_suffix_alignment(well, geometry, horizontal)
    segment_ids = k16_segment_ids(len(geometry), k_segments)
    md = horizontal.loc[row_idx, "MD"].to_numpy(np.float64)
    z = horizontal.loc[row_idx, "Z"].to_numpy(np.float64)
    geop = geometry["tvt_geop"].to_numpy(np.float64)
    if not np.isfinite(np.column_stack([md, z, geop])).all():
        raise ValueError(f"{well} suffix geometry inputs are not finite")
    parent = exp209_initial_rate(horizontal)
    parent_rate = float(parent["initial_rate"])
    geometry_rate, valid_steps = segment_step_rates(md, geop + z, segment_ids, k_segments)
    direct_rate, direct_valid, formula_parity = coordinate_step_rates(geop, z, md)
    helper_rate = np.diff(geop + z) / np.diff(md)
    parity_mask = direct_valid & np.isfinite(helper_rate)
    if parity_mask.any():
        formula_parity = max(
            formula_parity,
            float(np.max(np.abs(direct_rate[parity_mask] - helper_rate[parity_mask]))),
        )
    first_geometry_rate = float(geometry_rate[0])
    first_valid = math.isfinite(first_geometry_rate)
    mu = np.full(k_segments, parent_rate, dtype=np.float64)
    delta = np.zeros(k_segments, dtype=np.float64)
    fallback = np.ones(k_segments, dtype=bool)
    if first_valid:
        valid_segments = np.isfinite(geometry_rate)
        delta[valid_segments] = geometry_rate[valid_segments] - first_geometry_rate
        mu[valid_segments] = parent_rate + delta[valid_segments]
        fallback[valid_segments] = False
    anchor_idx = int(row_idx[0] - 1)
    anchor_md = float(horizontal.loc[anchor_idx, "MD"])
    anchor_z = float(horizontal.loc[anchor_idx, "Z"])
    anchor_tvt = float(horizontal.loc[anchor_idx, "TVT_input"])
    delta_md = np.diff(np.r_[anchor_md, md])
    if not np.isfinite(delta_md).all() or np.any(delta_md <= 0.0):
        raise ValueError(f"{well} suffix MD is not strictly increasing")
    sigma = float(uncertainty["sigma_226"])
    enabled = bool(uncertainty["observation_enabled"])
    schedule = pd.DataFrame(
        {
            "well_id": str(well),
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
            "geometry_delta_rate": delta[segment_ids],
            "mu_226": mu[segment_ids],
            "sigma_226": sigma,
            "observation_enabled": enabled,
            "geometry_fallback": fallback[segment_ids],
            "anchor_u": anchor_tvt + anchor_z,
            "formula_parity_max_abs": formula_parity,
        }
    )
    ledger = pd.DataFrame(
        {
            "well_id": str(well),
            "fold": int(geometry["fold"].iloc[0]),
            "segment_id": np.arange(k_segments, dtype=np.int16),
            "row_count": np.bincount(segment_ids, minlength=k_segments).astype(np.int32),
            "valid_geometry_steps": valid_steps,
            "parent_initial_rate": parent_rate,
            "first_segment_geometry_rate": first_geometry_rate,
            "geometry_segment_rate": geometry_rate,
            "geometry_delta_rate": delta,
            "mu_226": mu,
            "sigma_226": sigma,
            "observation_enabled": enabled,
            "geometry_fallback": fallback,
            "formula_parity_max_abs": formula_parity,
        }
    )
    fallback_summary = {
        "well_id": str(well),
        "fold": int(geometry["fold"].iloc[0]),
        "prefix_observation_enabled": enabled,
        "prefix_fallback_reason": str(uncertainty["fallback_reason"]),
        "geometry_first_segment_valid": first_valid,
        "geometry_fallback_segments": int(fallback.sum()),
        "parent_initial_rate_fallback": bool(parent["fallback"]),
    }
    return schedule, ledger, fallback_summary


@dataclass(frozen=True)
class FrozenStage0A:
    prefix_transitions: pd.DataFrame
    uncertainty: pd.DataFrame
    suffix_schedule: pd.DataFrame
    segment_ledger: pd.DataFrame
    fallback_summary: pd.DataFrame
    prefix_transitions_sha256: str
    uncertainty_sha256: str
    suffix_schedule_sha256: str
    segment_ledger_sha256: str


def freeze_stage_0a_features(
    geometry: pd.DataFrame,
    prefix_transitions: pd.DataFrame,
    uncertainty: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    *,
    expected_wells: int | None = None,
    expected_rows: int | None = None,
) -> FrozenStage0A:
    uncertainty = uncertainty.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if uncertainty["well_id"].duplicated().any():
        raise ValueError("uncertainty must contain exactly one row per well")
    expected_wells = (
        int(get_nested(config, "validation.expected_full_wells"))
        if expected_wells is None
        else int(expected_wells)
    )
    if len(uncertainty) != expected_wells:
        raise ValueError("uncertainty well coverage mismatch")
    if not np.isfinite(uncertainty["sigma_226"].to_numpy(np.float64)).all():
        raise ValueError("sigma_226 finite coverage is below one")
    uncertainty_lookup = uncertainty.set_index("well_id")
    schedules: list[pd.DataFrame] = []
    ledgers: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    k_segments = int(get_nested(config, "geometry_replay.params.k_segments"))
    for well, well_geometry in geometry.groupby("well_id", sort=True, observed=True):
        horizontal = load_target_safe_horizontal(raw_dir / f"{well}__horizontal_well.csv")
        schedule, ledger, fallback = build_well_suffix_schedule(
            str(well),
            well_geometry,
            horizontal,
            uncertainty_lookup.loc[str(well)],
            k_segments=k_segments,
        )
        schedules.append(schedule)
        ledgers.append(ledger)
        fallback_rows.append(fallback)
    suffix_schedule = (
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
        pd.DataFrame(fallback_rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    )
    forbidden = {"TVT", "tvt_true", "error", "abs_error", "persistent_role"}
    for name, frame in {
        "prefix_transitions": prefix_transitions,
        "uncertainty": uncertainty,
        "suffix_schedule": suffix_schedule,
    }.items():
        leaked = sorted(forbidden.intersection(frame.columns))
        if leaked:
            raise RuntimeError(f"truth entered frozen {name}: {leaked}")
    expected_rows = (
        int(get_nested(config, "validation.expected_full_rows"))
        if expected_rows is None
        else int(expected_rows)
    )
    if len(suffix_schedule) != expected_rows:
        raise ValueError("frozen suffix schedule row count mismatch")
    return FrozenStage0A(
        prefix_transitions=prefix_transitions,
        uncertainty=uncertainty,
        suffix_schedule=suffix_schedule,
        segment_ledger=segment_ledger,
        fallback_summary=fallback_summary,
        prefix_transitions_sha256=dataframe_content_sha256(
            prefix_transitions, PREFIX_TRANSITION_COLUMNS
        ),
        uncertainty_sha256=dataframe_content_sha256(uncertainty, UNCERTAINTY_COLUMNS),
        suffix_schedule_sha256=dataframe_content_sha256(suffix_schedule, SCHEDULE_CONTENT_COLUMNS),
        segment_ledger_sha256=dataframe_content_sha256(segment_ledger),
    )


# %% [markdown]
# ## 6. Truth late-join and reliability readout


# %%
def require_unchanged_freeze(frozen: FrozenStage0A) -> None:
    checks = {
        "prefix_transitions": (
            dataframe_content_sha256(frozen.prefix_transitions, PREFIX_TRANSITION_COLUMNS),
            frozen.prefix_transitions_sha256,
        ),
        "uncertainty": (
            dataframe_content_sha256(frozen.uncertainty, UNCERTAINTY_COLUMNS),
            frozen.uncertainty_sha256,
        ),
        "suffix_schedule": (
            dataframe_content_sha256(frozen.suffix_schedule, SCHEDULE_CONTENT_COLUMNS),
            frozen.suffix_schedule_sha256,
        ),
        "segment_ledger": (
            dataframe_content_sha256(frozen.segment_ledger),
            frozen.segment_ledger_sha256,
        ),
    }
    changed = [name for name, (actual, expected) in checks.items() if actual != expected]
    if changed:
        raise RuntimeError(f"truth late-join rejected changed frozen inputs: {changed}")


def attach_suffix_truth_after_freeze(
    frozen: FrozenStage0A,
    raw_dir: Path,
) -> pd.DataFrame:
    require_unchanged_freeze(frozen)
    frame = frozen.suffix_schedule.copy()
    truth = np.full(len(frame), np.nan, dtype=np.float64)
    for well, positions in frame.groupby("well_id", sort=True).indices.items():
        integer_positions = np.asarray(positions, dtype=np.int64)
        horizontal_truth = pd.read_csv(raw_dir / f"{well}__horizontal_well.csv", usecols=["TVT"])
        row_idx = frame.loc[integer_positions, "row_idx"].to_numpy(np.int64)
        truth[integer_positions] = horizontal_truth.loc[row_idx, "TVT"].to_numpy(np.float64)
    if not np.isfinite(truth).all():
        raise ValueError("suffix TVT truth finite coverage is below one")
    frame["tvt_true_readout_only"] = truth
    return frame


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(left[valid] - right[valid]))))


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = pd.Series(np.asarray(left, dtype=np.float64))
    right_values = pd.Series(np.asarray(right, dtype=np.float64))
    valid = left_values.notna() & right_values.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    left_rank = left_values.loc[valid].rank(method="average")
    right_rank = right_values.loc[valid].rank(method="average")
    if float(left_rank.std(ddof=0)) == 0.0 or float(right_rank.std(ddof=0)) == 0.0:
        return float("nan")
    return float(left_rank.corr(right_rank, method="pearson"))


def build_suffix_rate_readout(
    readout: pd.DataFrame,
    segment_ledger: pd.DataFrame,
    *,
    k_segments: int,
) -> pd.DataFrame:
    lookup = segment_ledger.set_index(["well_id", "segment_id"])
    rows: list[dict[str, Any]] = []
    for well, well_frame in readout.groupby("well_id", sort=True, observed=True):
        well_frame = well_frame.sort_values("row_idx", kind="mergesort")
        md = well_frame["md"].to_numpy(np.float64)
        true_u = well_frame["tvt_true_readout_only"].to_numpy(np.float64) + well_frame[
            "z"
        ].to_numpy(np.float64)
        segment_ids = well_frame["segment_id"].to_numpy(np.int16)
        actual_rate, actual_steps = segment_step_rates(md, true_u, segment_ids, k_segments)
        first_actual = float(actual_rate[0])
        if not math.isfinite(first_actual):
            continue
        for segment in range(1, k_segments):
            if not math.isfinite(float(actual_rate[segment])):
                continue
            ledger = lookup.loc[(str(well), segment)]
            geometry_rate = float(ledger["geometry_segment_rate"])
            if not math.isfinite(geometry_rate):
                continue
            rows.append(
                {
                    "well_id": str(well),
                    "fold": int(well_frame["fold"].iloc[0]),
                    "segment_id": segment,
                    "actual_valid_steps": int(actual_steps[segment]),
                    "sigma_226": float(ledger["sigma_226"]),
                    "observation_enabled": bool(ledger["observation_enabled"]),
                    "actual_u_rate": float(actual_rate[segment]),
                    "geometry_u_rate": geometry_rate,
                    "abs_geometry_rate_error": abs(geometry_rate - float(actual_rate[segment])),
                    "actual_relative_rate": float(actual_rate[segment] - first_actual),
                    "baseline_relative_rate": 0.0,
                    "exp355_relative_rate": float(ledger["geometry_delta_rate"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("suffix rate readout is empty")
    return result.sort_values(["well_id", "segment_id"], kind="mergesort").reset_index(drop=True)


def build_well_reliability_readout(
    segment_readout: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, part in segment_readout.groupby("well_id", sort=True, observed=True):
        rows.append(
            {
                "well_id": str(well),
                "fold": int(part["fold"].iloc[0]),
                "suffix_geometry_rate_rmse": rmse(part["actual_u_rate"], part["geometry_u_rate"]),
                "suffix_geometry_rate_mae": float(part["abs_geometry_rate_error"].mean()),
                "suffix_segments": len(part),
            }
        )
    result = uncertainty.merge(
        pd.DataFrame(rows), on=["well_id", "fold"], how="left", validate="one_to_one"
    )
    enabled = (
        result["observation_enabled"].astype(bool) & result["suffix_geometry_rate_rmse"].notna()
    )
    ranked = result.loc[enabled, ["well_id", "sigma_226"]].sort_values(
        ["sigma_226", "well_id"], kind="mergesort"
    )
    half = len(ranked) // 2
    low_wells = set(ranked.iloc[:half]["well_id"].astype(str))
    high_wells = set(ranked.iloc[half:]["well_id"].astype(str))
    result["sigma_half"] = "fallback"
    result.loc[result["well_id"].isin(low_wells), "sigma_half"] = "low"
    result.loc[result["well_id"].isin(high_wells), "sigma_half"] = "high"
    return result.sort_values("well_id", kind="mergesort").reset_index(drop=True)


def rate_metric(
    frame: pd.DataFrame,
    actual: str,
    predicted: str,
) -> float:
    return rmse(frame[actual], frame[predicted])


def schedule_gain_record(frame: pd.DataFrame, *, scope: str, value: str) -> dict[str, Any]:
    baseline = rate_metric(frame, "actual_relative_rate", "baseline_relative_rate")
    candidate = rate_metric(frame, "actual_relative_rate", "exp355_relative_rate")
    return {
        "scope": scope,
        "scope_value": value,
        "segments": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "baseline_relative_rate_rmse": baseline,
        "exp355_relative_rate_rmse": candidate,
        "gain_fraction": (
            (baseline - candidate) / baseline
            if math.isfinite(baseline) and baseline > 0
            else float("nan")
        ),
        "improved": bool(candidate < baseline),
    }


def build_mechanism_metrics(
    segment_readout: pd.DataFrame,
    well_readout: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    enabled = well_readout.loc[
        well_readout["observation_enabled"].astype(bool)
        & well_readout["suffix_geometry_rate_mae"].notna()
    ]
    pooled_spearman = spearman_correlation(
        enabled["sigma_226"], enabled["suffix_geometry_rate_mae"]
    )
    fold_rows: list[dict[str, Any]] = []
    for fold, part in enabled.groupby("fold", sort=True, observed=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": len(part),
                "sigma_vs_suffix_error_spearman": spearman_correlation(
                    part["sigma_226"], part["suffix_geometry_rate_mae"]
                ),
            }
        )
    fold_metrics = pd.DataFrame(
        fold_rows,
        columns=["fold", "wells", "sigma_vs_suffix_error_spearman"],
    )
    low_wells = set(well_readout.loc[well_readout["sigma_half"] == "low", "well_id"].astype(str))
    high_wells = set(well_readout.loc[well_readout["sigma_half"] == "high", "well_id"].astype(str))
    low_segments = segment_readout.loc[segment_readout["well_id"].isin(low_wells)]
    high_segments = segment_readout.loc[segment_readout["well_id"].isin(high_wells)]
    low_rmse = rate_metric(low_segments, "actual_u_rate", "geometry_u_rate")
    high_rmse = rate_metric(high_segments, "actual_u_rate", "geometry_u_rate")
    half_gain = (
        (high_rmse - low_rmse) / high_rmse
        if math.isfinite(high_rmse) and high_rmse > 0
        else float("nan")
    )
    schedule_rows = [schedule_gain_record(low_segments, scope="overall", value="low_sigma")]
    for fold, part in low_segments.groupby("fold", sort=True, observed=True):
        schedule_rows.append(schedule_gain_record(part, scope="fold", value=str(int(fold))))
    schedule_metrics = pd.DataFrame(schedule_rows)
    fold_metrics = fold_metrics.merge(
        schedule_metrics.loc[
            schedule_metrics["scope"] == "fold",
            [
                "scope_value",
                "baseline_relative_rate_rmse",
                "exp355_relative_rate_rmse",
                "gain_fraction",
                "improved",
            ],
        ]
        .assign(fold=lambda frame: frame["scope_value"].astype(int))
        .drop(columns="scope_value"),
        on="fold",
        how="left",
        validate="one_to_one",
    )
    overall_schedule = schedule_metrics.loc[schedule_metrics["scope"] == "overall"].iloc[0]
    return {
        "sigma_vs_suffix_error_spearman": pooled_spearman,
        "positive_spearman_folds": int(
            (fold_metrics["sigma_vs_suffix_error_spearman"] > 0.0).sum()
        ),
        "low_sigma_geometry_rate_rmse": low_rmse,
        "high_sigma_geometry_rate_rmse": high_rmse,
        "low_vs_high_rate_rmse_gain_fraction": half_gain,
        "low_sigma_schedule_gain_fraction": float(overall_schedule["gain_fraction"]),
        "improving_schedule_folds": int(fold_metrics["improved"].fillna(False).sum()),
        "enabled_wells": int(len(enabled)),
        "low_sigma_wells": len(low_wells),
        "high_sigma_wells": len(high_wells),
        "fallback_wells": int((~well_readout["observation_enabled"].astype(bool)).sum()),
        "fallback_well_fraction": float((~well_readout["observation_enabled"].astype(bool)).mean()),
    }, fold_metrics


# %% [markdown]
# ## 7. Stage 0A technical and mechanism gates


# %%
def evaluate_stage_0a_gate(
    geometry: pd.DataFrame,
    frozen: FrozenStage0A,
    mechanism: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical_thresholds = dict(get_nested(config, "gates.stage_0a_technical"))
    mechanism_thresholds = dict(get_nested(config, "gates.stage_0a_mechanism"))
    duplicate_rows = int(frozen.suffix_schedule.duplicated(["well_id", "row_idx"]).sum())
    geometry_keys = geometry[["well_id", "row_idx"]]
    schedule_keys = frozen.suffix_schedule[["well_id", "row_idx"]]
    missing_rows = len(
        geometry_keys.merge(
            schedule_keys, on=["well_id", "row_idx"], how="outer", indicator=True
        ).query("_merge != 'both'")
    )
    finite_coverage = float(
        np.isfinite(frozen.suffix_schedule[["mu_226", "sigma_226"]].to_numpy(np.float64))
        .all(axis=1)
        .mean()
    )
    formula_parity = float(
        max(
            frozen.uncertainty["formula_parity_max_abs"].max(),
            frozen.suffix_schedule["formula_parity_max_abs"].max(),
        )
    )
    technical_checks = {
        "well_count": int(frozen.uncertainty["well_id"].nunique())
        == int(technical_thresholds["expected_wells"]),
        "suffix_row_count": len(frozen.suffix_schedule)
        == int(technical_thresholds["expected_suffix_rows"]),
        "fold_count": int(frozen.suffix_schedule["fold"].nunique())
        == int(technical_thresholds["expected_folds"]),
        "duplicate_rows": duplicate_rows <= int(technical_thresholds["duplicate_rows_max"]),
        "missing_rows": missing_rows <= int(technical_thresholds["missing_rows_max"]),
        "finite_schedule_uncertainty_coverage": finite_coverage
        >= float(technical_thresholds["finite_schedule_uncertainty_coverage_min"]),
        "truth_reads_before_freeze": 0
        <= int(technical_thresholds["truth_reads_before_freeze_max"]),
        "forbidden_exp226_columns_before_freeze": 0
        <= int(technical_thresholds["forbidden_exp226_columns_before_freeze_max"]),
        "rate_formula_parity": formula_parity
        <= float(technical_thresholds["formula_parity_max_abs"]),
    }
    mechanism_checks = {
        "sigma_vs_suffix_error_spearman": float(mechanism["sigma_vs_suffix_error_spearman"])
        >= float(mechanism_thresholds["sigma_vs_suffix_abs_geometry_rate_error_spearman_min"]),
        "positive_spearman_folds": int(mechanism["positive_spearman_folds"])
        >= int(mechanism_thresholds["positive_spearman_folds_min"]),
        "low_vs_high_rate_rmse": float(mechanism["low_vs_high_rate_rmse_gain_fraction"])
        >= float(
            mechanism_thresholds["low_sigma_half_rate_rmse_gain_vs_high_sigma_half_min_fraction"]
        ),
        "low_sigma_schedule_gain": float(mechanism["low_sigma_schedule_gain_fraction"])
        >= float(
            mechanism_thresholds[
                "low_sigma_half_exp355_schedule_gain_vs_exp209_constant_min_fraction"
            ]
        ),
        "improving_schedule_folds": int(mechanism["improving_schedule_folds"])
        >= int(mechanism_thresholds["improving_schedule_folds_min"]),
        "prefix_fallback_fraction": float(mechanism["fallback_well_fraction"])
        <= float(mechanism_thresholds["prefix_fallback_well_fraction_max"]),
    }
    passed = bool(all(technical_checks.values()) and all(mechanism_checks.values()))
    return {
        "passed": passed,
        "technical_passed": bool(all(technical_checks.values())),
        "mechanism_passed": bool(all(mechanism_checks.values())),
        "technical_checks": technical_checks,
        "mechanism_checks": mechanism_checks,
        "technical_values": {
            "wells": int(frozen.uncertainty["well_id"].nunique()),
            "suffix_rows": len(frozen.suffix_schedule),
            "folds": int(frozen.suffix_schedule["fold"].nunique()),
            "duplicate_rows": duplicate_rows,
            "missing_rows": missing_rows,
            "finite_schedule_uncertainty_coverage": finite_coverage,
            "truth_reads_before_freeze": 0,
            "forbidden_exp226_columns_before_freeze": 0,
            "formula_parity_max_abs": formula_parity,
        },
        "mechanism_values": dict(mechanism),
        "thresholds": {
            "technical": technical_thresholds,
            "mechanism": mechanism_thresholds,
        },
        "decision": (
            "stage_0a_pass_stage_0b_implementation_still_requires_separate_approval"
            if passed
            else str(mechanism_thresholds["fail_action"])
        ),
        "automatic_stage_0b": False,
    }


# %% [markdown]
# ## 6. Fixed32 identity, exact rate-observation HMM, and prediction freeze


# %%
@dataclass
class Stage0BLeakageLedger:
    expected_wells: int = 32
    frozen_wells: set[str] = field(default_factory=set)
    truth_rows_before_freeze: int = 0
    role_rows_before_freeze: int = 0
    episode_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    role_rows_after_freeze: int = 0
    episode_rows_after_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def freeze(self, well: str) -> None:
        self.frozen_wells.add(str(well))

    def record_late(self, kind: str, rows: int) -> None:
        before = f"{kind}_rows_before_freeze"
        after = f"{kind}_rows_after_freeze"
        if not self.all_frozen:
            setattr(self, before, int(getattr(self, before)) + int(rows))
            raise RuntimeError(f"Stage 0B {kind} was read before all predictions froze")
        setattr(self, after, int(getattr(self, after)) + int(rows))


def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.fixed32_manifest")
    return resolve_existing(
        str(spec["filename"]),
        [str(spec["local"]), *[str(value) for value in spec["candidates"]]],
    )


def load_fixed32_identity(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.fixed32_manifest")
    path = fixed32_manifest_path(config)
    observed = sha256_path(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed: {observed}")
    frame = pd.read_csv(
        path,
        usecols=["well", "prefix_rows", "suffix_rows"],
        dtype={"well": str, "prefix_rows": "int32", "suffix_rows": "int32"},
    )
    frame = frame.sort_values("well", kind="mergesort").reset_index(drop=True)
    expected_rows = int(get_nested(config, "validation.stage_0b.expected_suffix_rows"))
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 identity must contain 32 unique wells")
    if int(frame["suffix_rows"].sum()) != expected_rows:
        raise ValueError("fixed32 suffix row total changed")
    return frame, {
        "path": str(path),
        "raw_sha256": observed,
        "rows": len(frame),
        "suffix_rows": int(frame["suffix_rows"].sum()),
        "identity_logical_sha256": dataframe_content_sha256(frame),
        "role_fold_columns_read_before_prediction_freeze": False,
    }


def load_fixed32_scope_after_freeze(
    config: Mapping[str, Any],
    identity: pd.DataFrame,
    ledger: Stage0BLeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("fixed32 roles require complete prediction freeze")
    path = fixed32_manifest_path(config)
    frame = pd.read_csv(path, dtype={"well": str, "matched_persistent_well": str})
    ledger.record_late("role", len(frame))
    required = {"well", "role", "fold", "matched_persistent_well", "prefix_rows", "suffix_rows"}
    if not required.issubset(frame.columns):
        raise ValueError("fixed32 scope schema changed")
    frame = frame.sort_values("well", kind="mergesort").reset_index(drop=True)
    identity_matches = (
        frame["well"].astype(str).tolist() == identity["well"].astype(str).tolist()
        and np.array_equal(
            frame["prefix_rows"].to_numpy(np.int64),
            identity["prefix_rows"].to_numpy(np.int64),
        )
        and np.array_equal(
            frame["suffix_rows"].to_numpy(np.int64),
            identity["suffix_rows"].to_numpy(np.int64),
        )
    )
    if not identity_matches:
        raise ValueError("fixed32 post-freeze identity differs from pre-freeze identity")
    if frame["role"].value_counts().to_dict() != {"persistent": 16, "control": 16}:
        raise ValueError("fixed32 role counts changed")
    if frame.groupby("fold").size().to_dict() != {0: 8, 1: 6, 2: 6, 3: 6, 4: 6}:
        raise ValueError("fixed32 fold counts changed")
    return frame


def build_fixed32_stage0_features(
    config: Mapping[str, Any],
    raw_dir: Path,
    geometry: pd.DataFrame,
    fold_by_well: Mapping[str, int],
    kappa_by_fold: Mapping[int, np.ndarray],
    identity: pd.DataFrame,
) -> tuple[FrozenStage0A, pd.DataFrame, pd.DataFrame]:
    selected = set(identity["well"].astype(str))
    raw_wells = list_horizontal_wells(raw_dir)
    if not selected.issubset(raw_wells):
        raise ValueError("fixed32 includes wells missing from raw train")
    params = params_from_config(config)
    prefix_parts: list[pd.DataFrame] = []
    uncertainty_rows: list[dict[str, Any]] = []
    prefix_manifest_rows: list[dict[str, Any]] = []
    fold_manifest_rows: list[dict[str, Any]] = []
    for fold in [0, 1, 2, 3, 4]:
        source_ids = sorted(well for well in raw_wells if int(fold_by_well[well]) != fold)
        target_ids = sorted(
            well for well in selected if int(fold_by_well[well]) == fold
        )
        source_wells = [
            load_source_geometry_well(raw_dir / f"{well}__horizontal_well.csv", params, wi=index)
            for index, well in enumerate(source_ids)
        ]
        fields = build_fields(source_wells, params)
        fold_manifest_rows.append(
            {
                "fold": fold,
                "source_wells": len(source_ids),
                "target_wells": len(target_ids),
                "donor_target_overlap": len(set(source_ids).intersection(target_ids)),
            }
        )
        print(f"exp495 Stage 0B feature fold={fold} sources={len(source_ids)} targets={len(target_ids)}")
        for well in target_ids:
            safe_frame = load_target_safe_horizontal(raw_dir / f"{well}__horizontal_well.csv")
            try:
                masked, destinations, manifest = build_prefix_mask(well, safe_frame, config)
                target = build_target_geometry_well(
                    well, masked, cut=int(manifest["replay_cut_row"]), params=params
                )
                path, _ = replay_exp226_geometry(target, fields, kappa_by_fold[fold], params)
                transitions, uncertainty = build_prefix_transition_rows(
                    well,
                    fold,
                    safe_frame,
                    target,
                    path,
                    destinations,
                    manifest,
                    config,
                )
                prefix_parts.append(transitions)
                uncertainty_rows.append(uncertainty)
                manifest.update(
                    {
                        "fold": fold,
                        "geometry_replay_rows": len(path),
                        "geometry_finite_coverage": float(np.isfinite(path).mean()),
                        "valid_transition_count": int(uncertainty["valid_transition_count"]),
                        "observation_enabled": bool(uncertainty["observation_enabled"]),
                        "fallback_reason": str(uncertainty["fallback_reason"]),
                    }
                )
                prefix_manifest_rows.append(manifest)
            except IneligiblePrefixError as exc:
                last_known = last_contiguous_known_index(
                    safe_frame["TVT_input"].to_numpy(np.float64)
                )
                uncertainty_rows.append(
                    {
                        "well_id": well,
                        "fold": fold,
                        "official_last_known_row": last_known,
                        "replay_cut_row": -1,
                        "selected_transition_count": 0,
                        "valid_transition_count": 0,
                        "residual_median": float("nan"),
                        "residual_mad": float("nan"),
                        "sigma_226": float(get_nested(config, "prefix_uncertainty.fallback_sigma")),
                        "observation_enabled": False,
                        "fallback_reason": str(exc),
                        "formula_parity_max_abs": 0.0,
                    }
                )
                prefix_manifest_rows.append(
                    {
                        "well_id": well,
                        "fold": fold,
                        "official_last_known_row": last_known,
                        "replay_cut_row": -1,
                        "selected_transition_count": 0,
                        "suffix_truth_reads_before_freeze": 0,
                        "target_well_in_donor_field": False,
                        "geometry_replay_rows": 0,
                        "geometry_finite_coverage": 1.0,
                        "valid_transition_count": 0,
                        "observation_enabled": False,
                        "fallback_reason": str(exc),
                    }
                )
        del fields, source_wells
    prefix_transitions = (
        pd.concat(prefix_parts, ignore_index=True)
        .sort_values(["well_id", "destination_row_idx"], kind="mergesort")
        .reset_index(drop=True)
        if prefix_parts
        else pd.DataFrame(columns=PREFIX_TRANSITION_COLUMNS)
    )
    uncertainty = pd.DataFrame(uncertainty_rows)
    selected_geometry = geometry.loc[geometry["well_id"].isin(selected)].copy()
    frozen = freeze_stage_0a_features(
        selected_geometry,
        prefix_transitions,
        uncertainty,
        raw_dir,
        config,
        expected_wells=int(get_nested(config, "validation.stage_0b.expected_wells")),
        expected_rows=int(get_nested(config, "validation.stage_0b.expected_suffix_rows")),
    )
    fold_manifest = pd.DataFrame(fold_manifest_rows)
    prefix_manifest = (
        pd.DataFrame(prefix_manifest_rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return frozen, fold_manifest, prefix_manifest


def load_hmm_safe_well(well: str, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    if "TVT" in horizontal.columns:
        raise ValueError("suffix TVT entered HMM input")
    typewell = pd.read_csv(
        raw_dir / f"{well}__typewell.csv", usecols=["TVT", "GR"]
    ).sort_values("TVT", kind="mergesort")
    return horizontal, typewell.reset_index(drop=True)


def prepare_rate_observation_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_index = np.flatnonzero(np.isfinite(tvt_input))
    eval_index = np.flatnonzero(~np.isfinite(tvt_input))
    if not np.array_equal(known_index, np.arange(len(known_index))):
        raise ValueError("HMM requires one contiguous known prefix")
    if not np.array_equal(eval_index, np.arange(eval_index[0], len(horizontal))):
        raise ValueError("HMM requires one contiguous unknown suffix")
    ordered = schedule.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(ordered["row_idx"].to_numpy(np.int64), eval_index):
        raise ValueError("rate observation schedule and HMM suffix rows differ")
    hmm = get_nested(config, "model.parent_hmm_fixed")
    last_index = int(known_index[-1])
    last_tvt = float(tvt_input[last_index])
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .to_numpy(np.float64)
    )
    step = float(hmm["step_ft"])
    band_pad = float(hmm["band_pad_ft"])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - band_pad)
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + band_pad)
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    state_gr = np.interp(grid, typewell_tvt, typewell_gr)
    known = horizontal.iloc[known_index]
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    known_tvt = tvt_input[known_index]
    prefix_residual = known_gr - np.interp(known_tvt, typewell_tvt, typewell_gr)
    sigma_gr = float(np.clip(np.nanstd(prefix_residual, ddof=0), 10.0, 60.0))
    observed_all = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)
    )
    observed_gr = observed_all[eval_index]
    zscore = (observed_gr[:, None] - state_gr[None, :]) / sigma_gr
    emission = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    md_all = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="raise").to_numpy(np.float64)
    dm = np.maximum(
        np.diff(np.concatenate([[md_all[last_index]], md_all[eval_index]])), 1.0
    )
    dz = np.diff(np.concatenate([[z_all[last_index]], z_all[eval_index]]))
    initial = exp209_initial_rate(horizontal)
    rates = np.linspace(
        -float(hmm["rate_span"]),
        float(hmm["rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    sigma_values = ordered["sigma_226"].to_numpy(np.float64)
    enabled_values = ordered["observation_enabled"].astype(bool).to_numpy()
    if not np.allclose(sigma_values, sigma_values[0], rtol=0.0, atol=0.0):
        raise ValueError("Stage 0B fixes one constant sigma_226 per well")
    if not np.all(enabled_values == enabled_values[0]):
        raise ValueError("Stage 0B observation fallback must be constant per well")
    return {
        "eval_index": eval_index,
        "grid": grid,
        "rates": rates,
        "emission": emission,
        "dm": dm,
        "dz": dz,
        "mu_226": ordered["mu_226"].to_numpy(np.float64),
        "sigma_226": float(sigma_values[0]),
        "observation_enabled": bool(enabled_values[0]),
        "start_p": float((last_tvt - grid_min) / step),
        "initial_rate": float(initial["initial_rate"]),
        "sigma_gr": sigma_gr,
        "raw_gr_missing": ~np.isfinite(
            pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)[eval_index]
        ),
    }


@njit(cache=True, nogil=True)
def rate_observation_log_kernel(
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    momentum: float,
    mu_226: float,
    sigma_226: float,
    observation_enabled: bool,
) -> tuple[np.ndarray, float]:
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    sig_rate_step = sig_r * np.sqrt(dm)
    rate_var_cells = (sig_rate_step / rate_step) ** 2
    output = np.empty((r_count, 3), np.float64)
    maximum_error = 0.0
    uniform = (not observation_enabled) or (not np.isfinite(sigma_226))
    for r_i in range(r_count):
        mean_rate_move = -(1.0 - momentum) * rates[r_i] * dm / rate_step
        p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1.0e-12)
        p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1.0e-12)
        side_total = p_plus + p_minus
        if side_total > 0.9:
            p_plus *= 0.9 / side_total
            p_minus *= 0.9 / side_total
        base = np.empty(3, np.float64)
        base[0] = p_minus
        base[1] = 1.0 - p_plus - p_minus
        base[2] = p_plus
        if uniform:
            row_sum = 0.0
            for k_i in range(3):
                output[r_i, k_i] = np.log(base[k_i])
                row_sum += base[k_i]
            maximum_error = max(maximum_error, abs(row_sum - 1.0))
            continue
        log_weight = np.empty(3, np.float64)
        best = -1.0e300
        for k_i in range(3):
            destination = r_i + k_i - 1
            if destination < 0:
                destination = 0
            elif destination >= r_count:
                destination = r_count - 1
            zscore = (rates[destination] - mu_226) / sigma_226
            value = np.log(base[k_i]) - 0.5 * zscore * zscore
            log_weight[k_i] = value
            best = max(best, value)
        total = 0.0
        for k_i in range(3):
            total += np.exp(log_weight[k_i] - best)
        log_norm = best + np.log(total)
        row_sum = 0.0
        for k_i in range(3):
            output[r_i, k_i] = log_weight[k_i] - log_norm
            row_sum += np.exp(output[r_i, k_i])
        maximum_error = max(maximum_error, abs(row_sum - 1.0))
    return output, maximum_error


@njit(cache=True, nogil=True, parallel=True)
def _rate_observation_hmm_fb(
    emission,
    dm,
    dz,
    step,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    initial_rate,
    initial_rate_sig,
    emission_lambda,
    momentum,
    mu_226,
    sigma_226,
    observation_enabled,
):
    t_count, p_count = emission.shape
    r_count = len(rates)
    neg = np.float32(-1.0e18)
    alpha = np.full((t_count, p_count, r_count), neg, np.float32)
    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * step
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - initial_rate) / initial_rate_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)
    tmp = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)
    maximum_transition_error = 0.0
    for t_i in range(t_count):
        rate_log_kernel, row_error = rate_observation_log_kernel(
            rates,
            dm[t_i],
            sig_r,
            momentum,
            mu_226[t_i],
            sigma_226,
            observation_enabled,
        )
        maximum_transition_error = max(maximum_transition_error, row_error)
        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = max(r2 - 1, 0)
                k1 = min(r2 + 1, r_count - 1)
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    best = max(best, value)
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(
                            prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best
                        )
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg
        sigma_position = max(sig_p, 0.35 * step)
        for r2 in prange(r_count):
            position_mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(position_mu / step + 0.5))
            position_log_kernel = np.empty(5, np.float64)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * step - position_mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p2 in range(p_count):
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if 0 <= p1 < p_count:
                        best = max(best, tmp[p1, r2] + position_log_kernel[k_i])
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if 0 <= p1 < p_count:
                            total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    cur[p2, r2] = np.float32(
                        best + np.log(total) + emission_lambda * emission[t_i, p2]
                    )
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]
    best = np.max(alpha[t_count - 1])
    total = np.sum(np.exp(alpha[t_count - 1] - best))
    log_likelihood = float(best) + np.log(total)
    post_p = np.zeros((t_count, p_count), np.float64)
    beta_next = np.zeros((p_count, r_count), np.float32)
    values = alpha[t_count - 1] + beta_next
    best = np.max(values)
    total = np.sum(np.exp(values - best))
    for p_i in range(p_count):
        for r_i in range(r_count):
            post_p[t_count - 1, p_i] += np.exp(values[p_i, r_i] - best) / total
    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        rate_log_kernel, row_error = rate_observation_log_kernel(
            rates,
            dm[t_i],
            sig_r,
            momentum,
            mu_226[t_i],
            sigma_226,
            observation_enabled,
        )
        maximum_transition_error = max(maximum_transition_error, row_error)
        sigma_position = max(sig_p, 0.35 * step)
        for r2 in prange(r_count):
            position_mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(position_mu / step + 0.5))
            position_log_kernel = np.empty(5, np.float64)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * step - position_mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        best = max(
                            best,
                            position_log_kernel[k_i]
                            + emission_lambda * emission[t_i, p2]
                            + beta_next[p2, r2],
                        )
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if 0 <= p2 < p_count:
                            total += np.exp(
                                position_log_kernel[k_i]
                                + emission_lambda * emission[t_i, p2]
                                + beta_next[p2, r2]
                                - best
                            )
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg
        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = max(r_i - 1, 0)
                k1 = min(r_i + 1, r_count - 1)
                for r2 in range(k0, k1 + 1):
                    best = max(
                        best, rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    )
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1]
                            + beta_tmp[p_i, r2]
                            - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg
        values = alpha[t_i - 1] + beta_cur
        best = np.max(values)
        total = np.sum(np.exp(values - best))
        for p_i in range(p_count):
            for r_i in range(r_count):
                post_p[t_i - 1, p_i] += np.exp(values[p_i, r_i] - best) / total
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    maximum_posterior_error = 0.0
    for t_i in range(t_count):
        maximum_posterior_error = max(
            maximum_posterior_error, abs(np.sum(post_p[t_i]) - 1.0)
        )
    return (
        post_p,
        log_likelihood,
        maximum_transition_error,
        maximum_posterior_error,
    )


def run_rate_observation_hmm(
    prepared: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    hmm = get_nested(config, "model.parent_hmm_fixed")
    started = time.perf_counter()
    post_p, log_likelihood, transition_error, posterior_error = _rate_observation_hmm_fb(
        np.asarray(prepared["emission"], dtype=np.float32),
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(hmm["step_ft"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig_ft"]),
        float(prepared["initial_rate"]),
        float(hmm["initial_rate_sig"]),
        float(hmm["emission_lambda"]),
        float(hmm["momentum"]),
        np.asarray(prepared["mu_226"], dtype=np.float64),
        float(prepared["sigma_226"]),
        bool(prepared["observation_enabled"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    mean = np.asarray(post_p, dtype=np.float64) @ grid
    variance = np.asarray(post_p, dtype=np.float64) @ (grid**2) - mean**2
    return {
        "mean": mean,
        "std": np.sqrt(np.maximum(variance, 0.0)),
        "log_likelihood": float(log_likelihood),
        "transition_row_sum_max_error": float(transition_error),
        "posterior_normalization_max_error": float(posterior_error),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def uniform_factor_parent_parity(config: Mapping[str, Any]) -> dict[str, Any]:
    grid = np.linspace(-3.5, 3.5, 21, dtype=np.float64)
    rates = np.linspace(-0.10, 0.10, 41, dtype=np.float64)
    prepared = {
        "emission": np.zeros((7, len(grid)), dtype=np.float32),
        "dm": np.full(7, 1.0, dtype=np.float64),
        "dz": np.zeros(7, dtype=np.float64),
        "grid": grid,
        "rates": rates,
        "start_p": 10.0,
        "initial_rate": 0.0,
        "mu_226": np.linspace(-0.02, 0.02, 7, dtype=np.float64),
        "sigma_226": float("inf"),
        "observation_enabled": False,
    }
    parent = run_rate_observation_hmm(prepared, config)
    prepared["observation_enabled"] = True
    uniform = run_rate_observation_hmm(prepared, config)
    maximum = float(np.max(np.abs(parent["mean"] - uniform["mean"])))
    threshold = float(
        get_nested(config, "gates.stage_0b_technical.parent_uniform_factor_parity_max_abs_ft")
    )
    return {
        "maximum_posterior_mean_abs_diff_ft": maximum,
        "threshold_ft": threshold,
        "passed": maximum <= threshold,
    }


def decode_fixed32_well(
    well: str,
    raw_dir: Path,
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizontal, typewell = load_hmm_safe_well(well, raw_dir)
    prepared = prepare_rate_observation_hmm_inputs(horizontal, typewell, schedule, config)
    result = run_rate_observation_hmm(prepared, config)
    ordered = schedule.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    prediction = ordered[
        [
            "well_id",
            "row_idx",
            "suffix_offset",
            "fold",
            "segment_id",
            "md_since",
            "mu_226",
            "sigma_226",
            "observation_enabled",
        ]
    ].copy()
    prediction["candidate_tvt"] = np.asarray(result["mean"], dtype=np.float64)
    prediction["candidate_std"] = np.asarray(result["std"], dtype=np.float64)
    prediction["raw_gr_missing"] = np.asarray(prepared["raw_gr_missing"], dtype=bool)
    if not np.isfinite(prediction[["candidate_tvt", "candidate_std"]].to_numpy()).all():
        raise RuntimeError(f"{well}: Stage 0B prediction is non-finite")
    runtime = {
        "well_id": well,
        "fold": int(ordered["fold"].iloc[0]),
        "rows": len(prediction),
        "sigma_226": float(prepared["sigma_226"]),
        "observation_enabled": bool(prepared["observation_enabled"]),
        "initial_rate": float(prepared["initial_rate"]),
        "sigma_gr": float(prepared["sigma_gr"]),
        **result,
    }
    runtime.pop("mean")
    runtime.pop("std")
    return prediction, runtime


def load_saved_exp209_fixed32(
    config: Mapping[str, Any], target_wells: set[str], expected_rows: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp209_parent")
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    inspection = inspect_gzip_csv(path)
    if inspection["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("saved exp209 decompressed SHA changed")
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["id", "well", "hmm_mean_tvt"],
        dtype={"id": str, "well": str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True)
    frame["row_idx"] = [
        int(identifier.removeprefix(f"{well}_"))
        for identifier, well in zip(frame["id"], frame["well"], strict=True)
    ]
    frame = frame.rename(columns={"well": "well_id", "hmm_mean_tvt": "exp209_tvt"})
    frame = frame[["well_id", "row_idx", "exp209_tvt"]].sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    )
    if len(frame) != expected_rows or frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("saved exp209 fixed32 identity changed")
    inspection["selected_rows"] = len(frame)
    return frame.reset_index(drop=True), inspection


def load_saved_exp355_fixed32(
    config: Mapping[str, Any], target_wells: set[str], expected_rows: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp355_reference")
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    metrics_path = path.parent.parent / str(spec["metrics_filename"])
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    source_metrics = json.loads(metrics_path.read_text())
    observed_logical = str(
        get_nested(source_metrics, str(spec["metrics_prediction_sha_key"]), "")
    )
    if observed_logical != str(spec["expected_prediction_logical_sha256"]):
        raise ValueError("saved exp355 prediction logical SHA contract changed")
    pieces: list[pd.DataFrame] = []
    columns = ["well_id", "row_idx", str(spec["prediction_column"])]
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"well_id": str, "row_idx": "int32"},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well_id"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True).rename(
        columns={str(spec["prediction_column"]): "exp355_tvt"}
    )
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if len(frame) != expected_rows or frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("saved exp355 fixed32 identity changed")
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "source_metrics_path": str(metrics_path),
        "source_prediction_logical_sha256": observed_logical,
        "selected_rows": len(frame),
        "selected_logical_sha256": dataframe_content_sha256(frame),
    }


# %% [markdown]
# ## 7. Role/episode/truth late-join and Stage 0B gates


# %%
def load_stage0b_truth_after_freeze(
    prediction: pd.DataFrame,
    raw_dir: Path,
    ledger: Stage0BLeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("suffix truth requires complete prediction freeze")
    pieces: list[pd.DataFrame] = []
    for well, part in prediction.groupby("well_id", sort=True):
        truth = pd.read_csv(raw_dir / f"{well}__horizontal_well.csv", usecols=["TVT"])
        row_idx = part["row_idx"].to_numpy(np.int64)
        values = truth.loc[row_idx, "TVT"].to_numpy(np.float64)
        ledger.record_late("truth", len(values))
        piece = part.copy()
        piece["tvt_true"] = values
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


def load_persistent_episodes_stage0b(
    config: Mapping[str, Any],
    persistent_wells: set[str],
    ledger: Stage0BLeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.all_frozen:
        raise RuntimeError("persistent episodes require complete prediction freeze")
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_existing(
        str(spec["filename"]),
        [str(spec["local"]), *[str(value) for value in spec["candidates"]]],
    )
    observed = sha256_path(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError("persistent episode SHA changed")
    frame = pd.read_csv(
        path,
        usecols=["episode_id", "well", "start_row_idx", "end_row_idx_exclusive"],
        dtype={"episode_id": str, "well": str},
    )
    frame = frame.loc[frame["well"].isin(persistent_wells)].copy()
    ledger.record_late("episode", len(frame))
    if frame.empty or set(frame["well"]) != persistent_wells:
        raise ValueError("fixed32 persistent wells are missing episode boundaries")
    return frame.sort_values(["well", "start_row_idx"], kind="mergesort"), {
        "path": str(path),
        "raw_sha256": observed,
        "selected_rows": len(frame),
        "selected_wells": int(frame["well"].nunique()),
    }


def build_stage0b_readouts(
    prediction_with_truth: pd.DataFrame,
    scope: pd.DataFrame,
    exp209: pd.DataFrame,
    exp355: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope_columns = scope[["well", "role", "fold"]].rename(columns={"well": "well_id"})
    frame = prediction_with_truth.merge(
        scope_columns, on="well_id", how="left", validate="many_to_one", suffixes=("", "_scope")
    )
    if not frame["fold"].eq(frame["fold_scope"]).all():
        raise ValueError("exp226 fold and fixed32 scope fold differ")
    frame = frame.drop(columns=["fold_scope"])
    frame = frame.merge(exp209, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    frame = frame.merge(exp355, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    numeric = ["candidate_tvt", "exp209_tvt", "exp355_tvt", "tvt_true"]
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise ValueError("Stage 0B readout contains missing/non-finite predictions or truth")
    for name in ["candidate", "exp209", "exp355"]:
        frame[f"{name}_error"] = frame[f"{name}_tvt"] - frame["tvt_true"]
    well_rows: list[dict[str, Any]] = []
    for well, part in frame.groupby("well_id", sort=True):
        row: dict[str, Any] = {
            "well_id": str(well),
            "role": str(part["role"].iloc[0]),
            "fold": int(part["fold"].iloc[0]),
            "rows": len(part),
        }
        for name in ["candidate", "exp209", "exp355"]:
            error = part[f"{name}_error"].to_numpy(np.float64)
            row[f"{name}_sse"] = float(np.square(error).sum())
            row[f"{name}_rmse_ft"] = float(np.sqrt(np.mean(np.square(error))))
        row["candidate_delta_vs_exp355_ft"] = (
            row["candidate_rmse_ft"] - row["exp355_rmse_ft"]
        )
        row["candidate_delta_vs_exp209_ft"] = (
            row["candidate_rmse_ft"] - row["exp209_rmse_ft"]
        )
        well_rows.append(row)
    well_metrics = pd.DataFrame(well_rows)
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        part = frame.loc[
            frame["well_id"].eq(str(episode.well))
            & frame["row_idx"].ge(int(episode.start_row_idx))
            & frame["row_idx"].lt(int(episode.end_row_idx_exclusive))
        ]
        if part.empty:
            raise ValueError(f"{episode.episode_id}: fixed episode window is empty")
        episode_rows.append(
            {
                "episode_id": str(episode.episode_id),
                "well_id": str(episode.well),
                "rows": len(part),
                "candidate_sse": float(np.square(part["candidate_error"]).sum()),
                "exp355_sse": float(np.square(part["exp355_error"]).sum()),
            }
        )
    return frame, well_metrics, pd.DataFrame(episode_rows)


def pooled_rmse_from_well_metrics(frame: pd.DataFrame, name: str) -> float:
    return float(np.sqrt(frame[f"{name}_sse"].sum() / frame["rows"].sum()))


def evaluate_stage0b_gate(
    config: Mapping[str, Any],
    scope: pd.DataFrame,
    prediction: pd.DataFrame,
    well_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    runtime: pd.DataFrame,
    parity: Mapping[str, Any],
    ledger: Stage0BLeakageLedger,
) -> dict[str, Any]:
    technical_config = get_nested(config, "gates.stage_0b_technical")
    mechanism_config = get_nested(config, "gates.stage_0b_mechanism")
    maximum_transition = float(runtime["transition_row_sum_max_error"].max())
    maximum_posterior = float(runtime["posterior_normalization_max_error"].max())
    finite_coverage = float(
        np.isfinite(prediction[["candidate_tvt", "candidate_std"]].to_numpy(np.float64)).mean()
    )
    hmm_seconds = float(runtime["elapsed_seconds"].sum())
    projected_runtime = hmm_seconds * 773.0 / 32.0
    technical = {
        "fixed32_wells": len(scope) == int(technical_config["expected_wells"]),
        "fixed32_rows": len(prediction) == int(technical_config["expected_rows"]),
        "fixed32_folds": scope["fold"].nunique() == int(technical_config["expected_folds"]),
        "parent_uniform_factor_parity": bool(parity["passed"]),
        "finite_prediction_coverage": finite_coverage
        >= float(technical_config["finite_prediction_coverage_min"]),
        "transition_row_sum": maximum_transition
        <= float(technical_config["transition_row_sum_max_error"]),
        "posterior_normalization": maximum_posterior
        <= float(technical_config["posterior_normalization_max_error"]),
        "projected_full_runtime": projected_runtime
        <= float(technical_config["projected_full_runtime_seconds_max"]),
        "peak_rss": peak_rss_gb() <= float(technical_config["peak_rss_gb_max"]),
        "truth_role_episode_reads_before_freeze": (
            ledger.truth_rows_before_freeze
            + ledger.role_rows_before_freeze
            + ledger.episode_rows_before_freeze
        )
        == 0,
    }
    persistent = well_metrics.loc[well_metrics["role"].eq("persistent")]
    controls = well_metrics.loc[well_metrics["role"].eq("control")]
    candidate_all = pooled_rmse_from_well_metrics(well_metrics, "candidate")
    exp355_all = pooled_rmse_from_well_metrics(well_metrics, "exp355")
    candidate_persistent = pooled_rmse_from_well_metrics(persistent, "candidate")
    exp355_persistent = pooled_rmse_from_well_metrics(persistent, "exp355")
    candidate_control = pooled_rmse_from_well_metrics(controls, "candidate")
    exp209_control = pooled_rmse_from_well_metrics(controls, "exp209")
    fold_rows: list[dict[str, Any]] = []
    for fold, part in well_metrics.groupby("fold", sort=True):
        candidate_fold = pooled_rmse_from_well_metrics(part, "candidate")
        exp355_fold = pooled_rmse_from_well_metrics(part, "exp355")
        fold_rows.append(
            {
                "fold": int(fold),
                "candidate_rmse_ft": candidate_fold,
                "exp355_rmse_ft": exp355_fold,
                "delta_ft": candidate_fold - exp355_fold,
                "improved": candidate_fold < exp355_fold,
            }
        )
    improving_folds = int(sum(bool(row["improved"]) for row in fold_rows))
    paired_delta = well_metrics["candidate_delta_vs_exp355_ft"].to_numpy(np.float64)
    paired_p95 = float(np.quantile(paired_delta, 0.95))
    worst_well = float(np.max(paired_delta))
    episode_exp355_sse = float(episode_metrics["exp355_sse"].sum())
    episode_candidate_sse = float(episode_metrics["candidate_sse"].sum())
    episode_reduction = (
        1.0 - episode_candidate_sse / episode_exp355_sse
        if episode_exp355_sse > 0.0
        else math.nan
    )
    mechanism = {
        "all32_gain_vs_saved_exp355": exp355_all - candidate_all
        >= float(mechanism_config["all32_gain_vs_saved_exp355_min_ft"]),
        "persistent_gain_vs_saved_exp355": exp355_persistent - candidate_persistent
        >= float(mechanism_config["persistent_gain_vs_saved_exp355_min_ft"]),
        "matched_control_delta_vs_saved_exp209": candidate_control - exp209_control
        <= float(mechanism_config["matched_control_delta_vs_saved_exp209_max_ft"]),
        "improving_folds_vs_saved_exp355": improving_folds
        >= int(mechanism_config["improving_folds_vs_saved_exp355_min"]),
        "persistent_episode_sse_reduction_vs_exp355": math.isfinite(episode_reduction)
        and episode_reduction
        >= float(mechanism_config["persistent_episode_sse_reduction_vs_exp355_min"]),
        "paired_by_well_delta_p95_vs_exp355": paired_p95
        <= float(mechanism_config["paired_by_well_delta_p95_vs_exp355_max_ft"]),
        "worst_well_delta_vs_exp355": worst_well
        <= float(mechanism_config["worst_well_delta_vs_exp355_max_ft"]),
    }
    passed = bool(all(technical.values()) and all(mechanism.values()))
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": {
            "candidate_all32_rmse_ft": candidate_all,
            "saved_exp355_all32_rmse_ft": exp355_all,
            "all32_gain_vs_saved_exp355_ft": exp355_all - candidate_all,
            "candidate_persistent_rmse_ft": candidate_persistent,
            "saved_exp355_persistent_rmse_ft": exp355_persistent,
            "persistent_gain_vs_saved_exp355_ft": exp355_persistent - candidate_persistent,
            "candidate_matched_control_rmse_ft": candidate_control,
            "saved_exp209_matched_control_rmse_ft": exp209_control,
            "matched_control_delta_vs_saved_exp209_ft": candidate_control - exp209_control,
            "improving_folds_vs_saved_exp355": improving_folds,
            "fold_metrics": fold_rows,
            "persistent_episode_sse_reduction_vs_exp355_fraction": episode_reduction,
            "paired_by_well_delta_p95_vs_exp355_ft": paired_p95,
            "worst_well_delta_vs_exp355_ft": worst_well,
            "maximum_transition_row_sum_error": maximum_transition,
            "maximum_posterior_normalization_error": maximum_posterior,
            "uniform_factor_parent_parity_max_abs_ft": parity[
                "maximum_posterior_mean_abs_diff_ft"
            ],
            "finite_prediction_coverage": finite_coverage,
            "candidate_hmm_seconds": hmm_seconds,
            "projected_full_runtime_seconds": projected_runtime,
            "peak_rss_gb": peak_rss_gb(),
            "fixed32_is_mechanism_only_not_cv_or_promotion": True,
        },
        "passed": passed,
        "decision": (
            "stage_0b_pass_waiting_for_separate_stage_1_approval"
            if passed
            else str(mechanism_config["fail_action"])
        ),
        "automatic_stage_1": False,
    }


def run_stage_0b(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.is_dir() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError("exp495 Stage 0B must run on Kaggle")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exp495 Stage 0B exact HMM")
    counts = validate_scientific_contract(config, require_run_authorization=True)
    set_num_threads(int(get_nested(config, "runtime.numba_threads_per_worker", 4)))
    started = time.perf_counter()
    raw_dir = train_data_dir(config)
    artifacts = artifact_dir()
    scientific_contract = build_stage0b_scientific_contract(config)
    scientific_contract_artifact = write_json(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_scientific_contract.json",
        scientific_contract,
    )
    identity, identity_input = load_fixed32_identity(config)
    target_wells = set(identity["well"].astype(str))
    expected_rows = int(identity["suffix_rows"].sum())
    geometry, geometry_input, oof_path = load_exp226_geometry(config)
    fold_by_well, kappa_by_fold, fold_inputs, fold_oof_path = load_exp226_fold_contract(config)
    if oof_path != fold_oof_path:
        raise RuntimeError("exp226 geometry and fold contract resolved different inputs")
    frozen_features, fold_manifest, prefix_manifest = build_fixed32_stage0_features(
        config, raw_dir, geometry, fold_by_well, kappa_by_fold, identity
    )
    feature_artifacts = {
        "prefix_uncertainty": write_csv(
            artifacts / f"{EXPERIMENT_NAME}_stage0b_prefix_uncertainty.csv",
            frozen_features.uncertainty,
        ),
        "rate_observation_schedule": write_gzip_csv(
            artifacts / f"{EXPERIMENT_NAME}_stage0b_rate_observation_schedule.csv.gz",
            frozen_features.suffix_schedule,
        ),
        "fold_donor_manifest": write_csv(
            artifacts / f"{EXPERIMENT_NAME}_stage0b_fold_donor_manifest.csv", fold_manifest
        ),
        "prefix_replay_manifest": write_csv(
            artifacts / f"{EXPERIMENT_NAME}_stage0b_prefix_replay_manifest.csv", prefix_manifest
        ),
    }
    parity = uniform_factor_parent_parity(config)
    ledger = Stage0BLeakageLedger(expected_wells=32)
    predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    schedule_groups = frozen_features.suffix_schedule.groupby("well_id", sort=False).indices
    hard_runtime = float(get_nested(config, "runtime.hard_runtime_limit_seconds"))
    hard_rss = float(get_nested(config, "runtime.peak_rss_limit_gb"))
    for index, row in enumerate(identity.itertuples(index=False), start=1):
        well = str(row.well)
        positions = np.asarray(schedule_groups[well], dtype=np.int64)
        prediction, runtime_row = decode_fixed32_well(
            well,
            raw_dir,
            frozen_features.suffix_schedule.iloc[positions].copy(),
            config,
        )
        if len(prediction) != int(row.suffix_rows):
            raise ValueError(f"{well}: fixed32 suffix rows changed")
        predictions.append(prediction)
        runtime_rows.append(runtime_row)
        ledger.freeze(well)
        if time.perf_counter() - started > hard_runtime:
            raise RuntimeError("Stage 0B runtime hard guard exceeded")
        if peak_rss_gb() > hard_rss:
            raise MemoryError("Stage 0B RSS hard guard exceeded")
        print(
            json.dumps(
                {
                    "event": "exp495_stage0b_progress",
                    "well_index": index,
                    "well_count": 32,
                    "well": well,
                    "rows": len(prediction),
                    "hmm_seconds": runtime_row["elapsed_seconds"],
                    "peak_rss_gb": peak_rss_gb(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all Stage 0B predictions froze")
    prediction = pd.concat(predictions, ignore_index=True).sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    runtime_frame = pd.DataFrame(runtime_rows).sort_values("well_id", kind="mergesort")
    prediction_artifact = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_predictions.csv.gz", prediction
    )
    runtime_artifact = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_hmm_runtime.csv", runtime_frame
    )
    scope = load_fixed32_scope_after_freeze(config, identity, ledger)
    exp209, exp209_input = load_saved_exp209_fixed32(config, target_wells, expected_rows)
    exp355, exp355_input = load_saved_exp355_fixed32(config, target_wells, expected_rows)
    persistent_wells = set(scope.loc[scope["role"].eq("persistent"), "well"].astype(str))
    episodes, episode_input = load_persistent_episodes_stage0b(
        config, persistent_wells, ledger
    )
    truth = load_stage0b_truth_after_freeze(prediction, raw_dir, ledger)
    readout, well_metrics, episode_metrics = build_stage0b_readouts(
        truth, scope, exp209, exp355, episodes
    )
    readout_artifact = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_truth_late_readout.csv.gz", readout
    )
    well_artifact = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_well_metrics.csv", well_metrics
    )
    episode_artifact = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_episode_metrics.csv", episode_metrics
    )
    gate = evaluate_stage0b_gate(
        config, scope, prediction, well_metrics, episode_metrics, runtime_frame, parity, ledger
    )
    gate_artifact = write_json(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_gate.json", gate
    )
    input_manifest = {
        "fixed32_identity": identity_input,
        "exp226_geometry": geometry_input,
        "exp226_fold_contract": fold_inputs,
        "saved_exp209": exp209_input,
        "saved_exp355": exp355_input,
        "persistent_episodes_after_freeze": episode_input,
        "leakage": {
            "frozen_wells": len(ledger.frozen_wells),
            "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
            "role_rows_before_freeze": ledger.role_rows_before_freeze,
            "episode_rows_before_freeze": ledger.episode_rows_before_freeze,
            "truth_rows_after_freeze": ledger.truth_rows_after_freeze,
            "role_rows_after_freeze": ledger.role_rows_after_freeze,
            "episode_rows_after_freeze": ledger.episode_rows_after_freeze,
        },
    }
    input_manifest["input_manifest_sha256"] = mapping_sha256(input_manifest)
    input_artifact = write_json(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_input_manifest.json", input_manifest
    )
    elapsed = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage_0b_fixed32_user_override",
        "status": (
            "stage_0b_pass_waiting_for_separate_stage_1_approval"
            if gate["passed"]
            else "stage_0b_fail_closed"
        ),
        "stage_0a_result_preserved": "FAIL",
        "stage_0a_fail_closed_override": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(prediction),
        "wells": int(prediction["well_id"].nunique()),
        "runtime_seconds": elapsed,
        "execution_counts": counts,
        "gate": gate,
        "uniform_factor_parent_parity": parity,
        "runtime": {"versions": runtime_versions(), "peak_rss_gb": peak_rss_gb()},
        "sha256": {
            "scientific_contract": scientific_contract["scientific_contract_sha256"],
            "input_manifest": input_manifest["input_manifest_sha256"],
            "prefix_uncertainty": frozen_features.uncertainty_sha256,
            "rate_observation_schedule": frozen_features.suffix_schedule_sha256,
            "prediction": prediction_artifact["logical_sha256"],
            "truth_late_readout": readout_artifact["logical_sha256"],
        },
        "generated_artifacts": {
            "scientific_contract": scientific_contract_artifact,
            **feature_artifacts,
            "predictions": prediction_artifact,
            "hmm_runtime": runtime_artifact,
            "truth_late_readout": readout_artifact,
            "well_metrics": well_artifact,
            "episode_metrics": episode_artifact,
            "stage0b_gate": gate_artifact,
            "input_manifest": input_artifact,
        },
        "stage_1_implemented": False,
        "stage_1_automatically_enabled": False,
        "inference_enabled": False,
        "submission_created": False,
    }
    summary_artifact = write_json(
        artifacts / f"{EXPERIMENT_NAME}_stage0b_summary.json", summary
    )
    summary["generated_artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    return summary


# %% [markdown]
# ## 8. Kaggle CPU orchestration and generated artifacts


# %%
def run_stage_0a(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.is_dir() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp495 Stage 0A must run on Kaggle; local execution requires explicit "
            "EXPERIMENT_ALLOW_LOCAL=1 approval"
        )
    if not bool(get_nested(config, "authorization.stage_0a_run_approved")):
        raise RuntimeError("exp495 Stage 0A run is not approved")
    if not bool(get_nested(config, "execution.run_stage_0a")):
        raise RuntimeError("exp495 execution.run_stage_0a remains false")
    counts = validate_scientific_contract(config, require_run_authorization=True)
    started = time.perf_counter()
    raw_dir = train_data_dir(config)
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    artifacts = artifact_dir()
    scientific_contract = build_scientific_contract(config)
    contract_report = write_json(
        artifacts / f"{EXPERIMENT_NAME}_scientific_contract.json", scientific_contract
    )
    geometry, geometry_report, oof_path = load_exp226_geometry(config)
    fold_by_well, kappa_by_fold, fold_input_reports, fold_oof_path = load_exp226_fold_contract(
        config
    )
    if oof_path != fold_oof_path:
        raise RuntimeError("exp226 geometry and fold manifest resolved different OOF files")
    raw_manifest, raw_report = validate_raw_well_identity(config, raw_dir)
    raw_wells = list_horizontal_wells(raw_dir)
    if set(raw_wells) != set(fold_by_well):
        raise ValueError("raw train and exp226 fold-manifest well sets differ")
    exp209_report = validate_exp209_dependency(config)
    params = params_from_config(config)
    prefix_parts: list[pd.DataFrame] = []
    uncertainty_rows: list[dict[str, Any]] = []
    prefix_manifest_rows: list[dict[str, Any]] = []
    fold_manifest_rows: list[dict[str, Any]] = []
    for fold in [0, 1, 2, 3, 4]:
        source_ids = sorted(well for well in raw_wells if fold_by_well[well] != fold)
        target_ids = sorted(well for well in raw_wells if fold_by_well[well] == fold)
        overlap = set(source_ids).intersection(target_ids)
        if overlap:
            raise ValueError(f"fold {fold} donor/target overlap")
        source_wells = [
            load_source_geometry_well(raw_dir / f"{well}__horizontal_well.csv", params, wi=index)
            for index, well in enumerate(source_ids)
        ]
        fields = build_fields(source_wells, params)
        fold_manifest_rows.append(
            {
                "fold": fold,
                "source_wells": len(source_ids),
                "target_wells": len(target_ids),
                "donor_target_overlap": 0,
                "raw_field_rows": len(fields.f_raw),
                "smooth_field_rows": len(fields.f_sm),
                "surface_rows": len(fields.surface_points),
            }
        )
        print(f"exp495 Stage 0A fold={fold} sources={len(source_ids)} targets={len(target_ids)}")
        for index, well in enumerate(target_ids, start=1):
            safe_frame = load_target_safe_horizontal(raw_dir / f"{well}__horizontal_well.csv")
            try:
                masked, destinations, manifest = build_prefix_mask(well, safe_frame, config)
                target = build_target_geometry_well(
                    well,
                    masked,
                    cut=int(manifest["replay_cut_row"]),
                    params=params,
                )
                path, _ = replay_exp226_geometry(target, fields, kappa_by_fold[fold], params)
                transitions, uncertainty = build_prefix_transition_rows(
                    well,
                    fold,
                    safe_frame,
                    target,
                    path,
                    destinations,
                    manifest,
                    config,
                )
                prefix_parts.append(transitions)
                uncertainty_rows.append(uncertainty)
                manifest.update(
                    {
                        "fold": fold,
                        "geometry_replay_rows": len(path),
                        "geometry_finite_coverage": float(np.isfinite(path).mean()),
                        "valid_transition_count": int(uncertainty["valid_transition_count"]),
                        "observation_enabled": bool(uncertainty["observation_enabled"]),
                        "fallback_reason": str(uncertainty["fallback_reason"]),
                    }
                )
                prefix_manifest_rows.append(manifest)
            except IneligiblePrefixError as exc:
                last_known = last_contiguous_known_index(
                    safe_frame["TVT_input"].to_numpy(np.float64)
                )
                uncertainty_rows.append(
                    {
                        "well_id": str(well),
                        "fold": fold,
                        "official_last_known_row": last_known,
                        "replay_cut_row": -1,
                        "selected_transition_count": 0,
                        "valid_transition_count": 0,
                        "residual_median": float("nan"),
                        "residual_mad": float("nan"),
                        "sigma_226": float(get_nested(config, "prefix_uncertainty.fallback_sigma")),
                        "observation_enabled": False,
                        "fallback_reason": str(exc),
                        "formula_parity_max_abs": 0.0,
                    }
                )
                prefix_manifest_rows.append(
                    {
                        "well_id": str(well),
                        "fold": fold,
                        "official_last_known_row": last_known,
                        "replay_cut_row": -1,
                        "selected_transition_count": 0,
                        "full_replay_rows": 0,
                        "post_cut_tvt_input_finite_rows_after_mask": 0,
                        "suffix_truth_reads_before_freeze": 0,
                        "target_well_in_donor_field": False,
                        "geometry_replay_rows": 0,
                        "geometry_finite_coverage": 1.0,
                        "valid_transition_count": 0,
                        "observation_enabled": False,
                        "fallback_reason": str(exc),
                    }
                )
            if index % 25 == 0 or index == len(target_ids):
                print(f"exp495 Stage 0A fold={fold} processed={index}/{len(target_ids)}")
        del fields, source_wells
    prefix_transitions = (
        pd.concat(prefix_parts, ignore_index=True)
        .sort_values(["well_id", "destination_row_idx"], kind="mergesort")
        .reset_index(drop=True)
        if prefix_parts
        else pd.DataFrame(columns=[*PREFIX_TRANSITION_COLUMNS])
    )
    uncertainty = pd.DataFrame(uncertainty_rows)
    prefix_manifest = (
        pd.DataFrame(prefix_manifest_rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    frozen = freeze_stage_0a_features(geometry, prefix_transitions, uncertainty, raw_dir, config)
    raw_manifest_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_raw_well_identity.csv", raw_manifest
    )
    fold_manifest = pd.DataFrame(fold_manifest_rows)
    fold_manifest_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_fold_donor_manifest.csv", fold_manifest
    )
    prefix_manifest_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_prefix_replay_manifest.csv", prefix_manifest
    )
    prefix_report = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_prefix_rate_transitions.csv.gz",
        frozen.prefix_transitions,
    )
    uncertainty_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_prefix_uncertainty.csv", frozen.uncertainty
    )
    schedule_report = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_rate_observation_schedule.csv.gz",
        frozen.suffix_schedule,
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
        "suffix_truth_reads_before_freeze": 0,
        "forbidden_exp226_columns_loaded": 0,
        "prefix_transitions_sha256": frozen.prefix_transitions_sha256,
        "uncertainty_sha256": frozen.uncertainty_sha256,
        "suffix_schedule_sha256": frozen.suffix_schedule_sha256,
        "segment_ledger_sha256": frozen.segment_ledger_sha256,
        "rows": len(frozen.suffix_schedule),
        "wells": len(frozen.uncertainty),
        "observation_enabled_wells": int(
            frozen.uncertainty["observation_enabled"].astype(bool).sum()
        ),
        "fallback_wells": int((~frozen.uncertainty["observation_enabled"].astype(bool)).sum()),
    }
    freeze_manifest["freeze_manifest_sha256"] = mapping_sha256(freeze_manifest)
    freeze_report = write_json(
        artifacts / f"{EXPERIMENT_NAME}_freeze_manifest.json", freeze_manifest
    )
    input_manifest = {
        "truth_attached": False,
        "raw_train": raw_report,
        "exp226_geometry": geometry_report,
        "exp226_fold_contract": fold_input_reports,
        "exp209_parent": exp209_report,
        "raw_well_identity_artifact": raw_manifest_report,
        "fold_donor_manifest_artifact": fold_manifest_report,
    }
    input_manifest["input_manifest_sha256"] = mapping_sha256(input_manifest)
    input_report = write_json(artifacts / f"{EXPERIMENT_NAME}_input_manifest.json", input_manifest)
    # The first unknown-suffix TVT read occurs only after every target-free
    # feature table above has been persisted and assigned a logical SHA.
    readout = attach_suffix_truth_after_freeze(frozen, raw_dir)
    segment_readout = build_suffix_rate_readout(
        readout,
        frozen.segment_ledger,
        k_segments=int(get_nested(config, "geometry_replay.params.k_segments")),
    )
    well_readout = build_well_reliability_readout(segment_readout, frozen.uncertainty)
    mechanism, fold_metrics = build_mechanism_metrics(segment_readout, well_readout)
    gate = evaluate_stage_0a_gate(geometry, frozen, mechanism, config)
    segment_report = write_gzip_csv(
        artifacts / f"{EXPERIMENT_NAME}_suffix_rate_readout.csv.gz",
        segment_readout,
    )
    well_report = write_csv(artifacts / f"{EXPERIMENT_NAME}_well_reliability.csv", well_readout)
    fold_report = write_csv(
        artifacts / f"{EXPERIMENT_NAME}_fold_reliability_metrics.csv", fold_metrics
    )
    gate_report = write_json(artifacts / f"{EXPERIMENT_NAME}_stage0a_gate.json", gate)
    elapsed = time.perf_counter() - started
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0a_completed_pass_waiting_for_stage_0b_implementation_approval"
            if gate["passed"]
            else "stage_0a_completed_fail_closed_before_hmm"
        ),
        "route": "pf_beam",
        "stage": "stage_0a_reliability_identifiability_zero_hmm",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": elapsed,
        "rows": len(readout),
        "wells": int(readout["well_id"].nunique()),
        "folds": sorted(int(value) for value in readout["fold"].unique()),
        "truth_attached_after_freeze": True,
        "gate": gate,
        "execution_counts": counts,
        "runtime": runtime_versions(),
        "sha256": {
            "scientific_contract": scientific_contract["scientific_contract_sha256"],
            "input_manifest": input_manifest["input_manifest_sha256"],
            "freeze_manifest": freeze_manifest["freeze_manifest_sha256"],
            "prefix_transitions": frozen.prefix_transitions_sha256,
            "uncertainty": frozen.uncertainty_sha256,
            "suffix_schedule": frozen.suffix_schedule_sha256,
            "segment_ledger": frozen.segment_ledger_sha256,
            "suffix_rate_readout": segment_report["logical_sha256"],
            "well_reliability": well_report["logical_sha256"],
        },
        "generated_artifacts": {
            "scientific_contract": contract_report,
            "input_manifest": input_report,
            "raw_well_identity": raw_manifest_report,
            "fold_donor_manifest": fold_manifest_report,
            "prefix_replay_manifest": prefix_manifest_report,
            "prefix_transitions": prefix_report,
            "prefix_uncertainty": uncertainty_report,
            "rate_observation_schedule": schedule_report,
            "geometry_segment_ledger": ledger_report,
            "fallback_summary": fallback_report,
            "freeze_manifest": freeze_report,
            "suffix_rate_readout": segment_report,
            "well_reliability": well_report,
            "fold_reliability_metrics": fold_report,
            "stage0a_gate": gate_report,
        },
        "stage_0b_implemented": False,
        "stage_0b_automatically_enabled": False,
        "stage_1_implemented": False,
        "inference_enabled": False,
        "submission_created": False,
    }
    summary_report = write_json(artifacts / f"{EXPERIMENT_NAME}_summary.json", summary)
    summary["generated_artifacts"]["summary"] = summary_report
    write_json(metrics_output_path(), summary)
    return summary


# %%
CONFIG = load_experiment_config()
CONTRACT_COUNTS = validate_scientific_contract(CONFIG)
print("Experiment:", get_nested(CONFIG, "experiment.name"))
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Stage 0B implementation counts:", CONTRACT_COUNTS)
print(
    "Canonical notebook adoption approved:",
    get_nested(CONFIG, "authorization.canonical_notebook_adoption_approved"),
)
print("Kaggle package approved:", get_nested(CONFIG, "authorization.kaggle_package_approved"))
print("Stage 0A run approved:", get_nested(CONFIG, "authorization.stage_0a_run_approved"))
print("Run Stage 0A:", get_nested(CONFIG, "execution.run_stage_0a"))
print("Stage 0B override approved:", get_nested(CONFIG, "authorization.stage_0a_fail_closed_override_approved"))
print("Stage 0B run approved:", get_nested(CONFIG, "authorization.stage_0b_run_approved"))
print("Run Stage 0B:", get_nested(CONFIG, "execution.run_stage_0b"))

if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "authorization.stage_0b_run_approved")) and bool(
        get_nested(CONFIG, "execution.run_stage_0b")
    ):
        STAGE_0B_SUMMARY = run_stage_0b(CONFIG)
        print(json.dumps(to_jsonable(STAGE_0B_SUMMARY["gate"]), indent=2, sort_keys=True))
    elif bool(get_nested(CONFIG, "authorization.stage_0a_run_approved")) and bool(
        get_nested(CONFIG, "execution.run_stage_0a")
    ):
        STAGE_0A_SUMMARY = run_stage_0a(CONFIG)
        print(json.dumps(to_jsonable(STAGE_0A_SUMMARY["gate"]), indent=2, sort_keys=True))
    else:
        print(
            "Execution is fail-closed; no Stage 0B HMM, model, GPU, inference, "
            "or submission was run."
        )
