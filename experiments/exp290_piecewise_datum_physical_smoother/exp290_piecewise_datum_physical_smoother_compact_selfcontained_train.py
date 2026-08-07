# %% [markdown]
# # exp290 piecewise datum physical smoother — Stage 0
#
# This zero-booster PF/Beam-route audit replays the fold-safe exp226 geometry at
# three fixed known-prefix pseudo-cuts, fits only target-safe GR calibration,
# and emits one bounded semi-Markov posterior mean.  Every pseudo-tail
# prediction is hashed before held-known `TVT_input` is attached for scoring.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Fold-safe exp226 geometry replay helpers
# 4. Pseudo-cut, Type Well, and hierarchical-prior helpers
# 5. Exact bounded semi-Markov forward-backward solver
# 6. Prediction freeze, truth attachment, metrics, and guards
# 7. Full Kaggle CPU orchestration and generated artifacts
# 8. Setup and contract preview
# 9. Run the fixed Stage 0 audit

# %% [markdown]
# ## 1. Imports and fixed experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

EXPERIMENT_NAME = "exp290_piecewise_datum_physical_smoother"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FILE_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TARGET_SAFE_COLUMNS = ("id", "row_idx", *TARGET_FILE_COLUMNS)
TARGET_FORBIDDEN_COLUMNS = {
    "TVT",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "target",
    "target_error",
    "error",
    "abs_error",
    "oracle_rank",
    "exp281_prediction",
}
PREDICTION_FORBIDDEN_COLUMNS = {
    "held_tvt",
    "base_error",
    "model_error",
    "nll_excess",
    "correction_sign_match",
}
FROZEN_PREDICTION_COLUMNS = (
    "well_id",
    "fold",
    "horizon_rows",
    "cut_row",
    "row_idx",
    "MD",
    "base_geometry",
    "posterior_mean_delta",
    "prediction",
    "posterior_entropy",
    "reset_probability",
    "reliability",
    "event_gate_threshold",
    "truth_attached",
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP290_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
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
    raise FileNotFoundError(f"exp290 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
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
    return project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame if columns is None else frame[list(columns)]
    return hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False).encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(data)
    return {
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(frame),
        "columns": len(frame.columns),
    }


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    summed = np.sum(np.exp(array - finite_maximum), axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        output = finite_maximum + np.log(summed)
    output = np.where(np.isfinite(maximum), output, maximum)
    if axis is None:
        return np.asarray(output.squeeze())
    return np.squeeze(output, axis=axis)


def robust_scale(values: Sequence[float], floor: float = 1.0e-6) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float(floor)
    mad = float(np.median(np.abs(array - np.median(array))))
    return float(max(1.4826 * mad, floor))


def rmse(truth: Sequence[float], prediction: Sequence[float]) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(left[valid] - right[valid]))))


def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    expected = {
        "experiment.route": "pf_beam",
        "physics.state_space.minimum_ft": -15.0,
        "physics.state_space.maximum_ft": 15.0,
        "physics.state_space.step_ft": 0.5,
        "physics.state_space.state_count": 61,
        "physics.state_space.checkpoint_stride_rows": 64,
        "physics.state_space.minimum_segment_duration_rows": 256,
        "physics.state_space.duration_phase_count_per_offset": 5,
        "physics.state_space.expanded_state_count": 305,
        "stages.stage0.active_contracts": 1,
        "stages.stage0.ml_configs": 0,
        "stages.stage0.trained_folds": 0,
        "stages.stage0.boosters": 0,
        "execution.active_variants": 1,
        "execution.lightgbm_config_count": 0,
        "execution.trained_fold_count": 0,
        "execution.total_boosters": 0,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"fixed scientific contract mismatch for {key}")
    phase = str(get_nested(config, "experiment.phase"))
    if phase not in {
        "stage0_kaggle_execution_approved",
        "stage0_scientific_guard_failed",
    }:
        raise ValueError(f"unsupported Stage 0 lifecycle phase: {phase}")
    if bool(get_nested(config, "physics.state_space.cumulative_random_walk")):
        raise ValueError("exp290 forbids a cumulative random walk")
    if bool(get_nested(config, "physics.solver.viterbi_output_allowed")):
        raise ValueError("exp290 forbids Viterbi output")
    if bool(get_nested(config, "physics.solver.candidate_bank_allowed")):
        raise ValueError("exp290 forbids candidate banks")
    if bool(get_nested(config, "execution.control_or_parent_retraining")):
        raise ValueError("exp290 forbids parent/control retraining")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("Stage 0 implementation approval is not recorded")
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise ValueError("Stage 0 Kaggle push approval is not recorded")
    if not str(get_nested(config, "execution.kaggle_push_approval_source")).strip():
        raise ValueError("Stage 0 Kaggle push approval source is not recorded")
    if bool(get_nested(config, "inference.enabled")) or bool(
        get_nested(config, "inference.create_submission")
    ):
        raise ValueError("inference and submission must remain disabled")


def validate_target_safe_frame(frame: pd.DataFrame) -> None:
    leaked = sorted(TARGET_FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"target-safe frame contains forbidden columns: {leaked}")
    if tuple(frame.columns) != TARGET_SAFE_COLUMNS:
        raise ValueError(
            f"target-safe columns mismatch: expected={TARGET_SAFE_COLUMNS}, "
            f"actual={tuple(frame.columns)}"
        )
    if frame["row_idx"].duplicated().any():
        raise ValueError("target-safe row_idx is not unique")
    ordering = np.lexsort(
        (
            frame["row_idx"].to_numpy(np.int64),
            pd.to_numeric(frame["MD"], errors="raise").to_numpy(np.float64),
        )
    )
    if not np.array_equal(ordering, np.arange(len(frame), dtype=np.int64)):
        raise ValueError("horizontal rows are not MD ascending with original-row tie-break")


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    allowed = set(TARGET_FILE_COLUMNS) | {"id"}
    frame = pd.read_csv(path, usecols=lambda column: column in allowed)
    missing = sorted(set(TARGET_FILE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"target horizontal {path.name} missing {missing}")
    well = path.name.split("__")[0]
    if "id" not in frame.columns:
        frame.insert(0, "id", [f"{well}:{index}" for index in range(len(frame))])
    frame.insert(1, "row_idx", np.arange(len(frame), dtype=np.int64))
    frame = frame[list(TARGET_SAFE_COLUMNS)]
    validate_target_safe_frame(frame)
    return frame


def last_contiguous_known_index(values: Sequence[float]) -> int:
    finite = np.isfinite(np.asarray(values, dtype=np.float64))
    if not len(finite) or not finite[0]:
        raise ValueError("well has no contiguous TVT_input prefix from row zero")
    missing = np.flatnonzero(~finite)
    end = len(finite) - 1 if not len(missing) else int(missing[0] - 1)
    if finite[end + 1 :].any():
        raise ValueError("TVT_input has finite rows after the contiguous prefix")
    return int(end)


# %% [markdown]
# ## 3. Fold-safe exp226 geometry replay helpers

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


def geometry_params_from_config(config: Mapping[str, Any]) -> K16Params:
    params = get_nested(config, "physics.geometry_replay.params", {})
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


def segment_geometry(
    x: np.ndarray,
    y: np.ndarray,
    s: int,
    n: int,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, n, params.k_segments + 1)
    step_index = np.arange(1, n + 1.0)
    segment_id = np.clip(
        np.searchsorted(edges[1:], step_index, side="left"),
        0,
        params.k_segments - 1,
    )
    midpoint = np.empty((params.k_segments, 2), dtype=np.float64)
    projection = np.empty(params.k_segments, dtype=np.float64)
    azimuth = np.empty(params.k_segments, dtype=np.float64)
    theta = np.radians(params.theta0)
    final_row = len(x) - 1
    for segment in range(params.k_segments):
        first = min(s + 1 + int(edges[segment]), final_row)
        last_raw = s + 1 + max(int(edges[segment + 1]) - 1, int(edges[segment]))
        last = min(max(last_raw, first), final_row)
        azimuth[segment] = np.arctan2(y[last] - y[first], x[last] - x[first])
        midpoint[segment] = ((x[first] + x[last]) / 2.0, (y[first] + y[last]) / 2.0)
        projection[segment] = np.cos(azimuth[segment] - theta)
    return segment_id.astype(np.int64), midpoint, projection, azimuth


def fit_coefficients(
    residual: np.ndarray,
    vertical: np.ndarray,
    rows: int,
    params: K16Params,
    rho: float,
) -> np.ndarray:
    positions = np.arange(1, rows + 1.0)
    edges = np.linspace(0, rows, params.k_segments + 1)
    basis = np.column_stack(
        [
            np.clip(positions - edges[index], 0, edges[index + 1] - edges[index])
            for index in range(params.k_segments)
        ]
    )
    matrix = basis.T @ basis
    if rho > 0:
        difference = np.diff(np.eye(params.k_segments), axis=0)
        scale = float(np.mean(np.diag(matrix))) if matrix.size else 1.0
        matrix = matrix + rho * max(scale, 1.0e-9) * difference.T @ difference
    rhs = basis.T @ (residual - vertical)
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
    tvt_input = frame["TVT_input"].to_numpy(np.float64)
    anchor_index = last_contiguous_known_index(tvt_input)
    ndz = -np.diff(z)[anchor_index:]
    rows = len(ndz)
    if rows <= 0:
        raise ValueError(f"source well {path.name} has no original suffix")
    residual = tvt[anchor_index + 1 :] - tvt[anchor_index]
    vertical = np.cumsum(ndz)
    segment_id, midpoint, projection, azimuth = segment_geometry(
        x, y, anchor_index, rows, params
    )
    return GeometryWell(
        wid=path.name.split("__")[0],
        wi=int(wi),
        s=anchor_index,
        n=rows,
        ndz=ndz,
        anchor=float(tvt[anchor_index]),
        ti=tvt_input,
        segid=segment_id,
        mid=midpoint,
        proj=projection,
        az=azimuth,
        x=x,
        y=y,
        z=z,
        md=md,
        anc=frame["ANCC"].to_numpy(np.float64),
        c_raw=fit_coefficients(residual, vertical, rows, params, rho=0.0),
        c_sm=fit_coefficients(residual, vertical, rows, params, rho=params.smooth_rho),
    )


def build_target_geometry_well(
    well: str,
    masked_frame: pd.DataFrame,
    *,
    cut: int,
    params: K16Params,
    wi: int = -1,
) -> GeometryWell:
    validate_target_safe_frame(masked_frame)
    if masked_frame.loc[cut + 1 :, "TVT_input"].notna().any():
        raise ValueError("target geometry received unmasked post-cut TVT_input")
    x = pd.to_numeric(masked_frame["X"], errors="raise").to_numpy(np.float64)
    y = pd.to_numeric(masked_frame["Y"], errors="raise").to_numpy(np.float64)
    z = pd.to_numeric(masked_frame["Z"], errors="raise").to_numpy(np.float64)
    md = pd.to_numeric(masked_frame["MD"], errors="raise").to_numpy(np.float64)
    tvt_input = pd.to_numeric(masked_frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    rows = len(masked_frame) - cut - 1
    ndz = -np.diff(z)[cut:]
    if len(ndz) != rows or rows <= 0 or not np.isfinite(tvt_input[cut]):
        raise ValueError("pseudo geometry anchor/horizon is invalid")
    segment_id, midpoint, projection, azimuth = segment_geometry(x, y, cut, rows, params)
    return GeometryWell(
        wid=str(well),
        wi=int(wi),
        s=int(cut),
        n=rows,
        ndz=ndz,
        anchor=float(tvt_input[cut]),
        ti=tvt_input,
        segid=segment_id,
        mid=midpoint,
        proj=projection,
        az=azimuth,
        x=x,
        y=y,
        z=z,
        md=md,
    )


def build_fields(wells: Sequence[GeometryWell], params: K16Params) -> FieldPack:
    def pack(key: str) -> np.ndarray:
        rows: list[tuple[float, float, float, float]] = []
        for well in wells:
            coefficients = getattr(well, key)
            if coefficients is None:
                continue
            for segment in range(params.k_segments):
                if abs(well.proj[segment]) > params.field_min_proj:
                    rows.append(
                        (
                            well.mid[segment, 0],
                            well.mid[segment, 1],
                            coefficients[segment] / well.proj[segment],
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
        ancc = well.anc[::step]
        surface_parts.append(
            np.column_stack(
                [
                    well.x[::step],
                    well.y[::step],
                    ancc,
                    np.full(len(ancc), well.wi, dtype=np.float64),
                ]
            )
        )
    if not surface_parts:
        raise ValueError("empty ANCC donor surface")
    surface = np.vstack(surface_parts)
    surface = surface[np.isfinite(surface[:, 2])]
    if len(surface) < 3:
        raise ValueError("insufficient finite ANCC donor surface")
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


def safe_nearest_indices(dist2: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    if not len(candidates):
        return candidates
    count = min(max(int(k), 1), len(candidates))
    partial = candidates[np.argpartition(dist2[candidates], count - 1)[:count]]
    order = np.lexsort((partial, dist2[partial]))
    return partial[order]


def local_linear(
    field: np.ndarray,
    own_wi: int,
    midpoint: np.ndarray,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray]:
    keep = field[:, 3] != own_wi
    x, y, values = field[keep, 0], field[keep, 1], field[keep, 2]
    if not len(values):
        raise ValueError("exp226 local-linear donor field is empty after self exclusion")
    drift = np.empty(len(midpoint), dtype=np.float64)
    distance = np.empty(len(midpoint), dtype=np.float64)
    for index, point in enumerate(midpoint):
        dist2 = np.square(x - point[0]) + np.square(y - point[1])
        selected = safe_nearest_indices(dist2, np.arange(len(dist2)), params.local_linear_k)
        weights = np.exp(
            np.maximum(-dist2[selected] / (2.0 * params.local_linear_bandwidth**2), -700)
        )
        dx = (x[selected] - point[0]) / 1000.0
        dy = (y[selected] - point[1]) / 1000.0
        design = np.column_stack([np.ones(len(selected)), dx, dy])
        ridge = params.local_linear_ridge * np.sum(weights) * np.diag([0.0, 1.0, 1.0])
        matrix = (design * weights[:, None]).T @ design + ridge
        rhs = (design * weights[:, None]).T @ values[selected]
        try:
            drift[index] = np.linalg.solve(matrix, rhs)[0]
        except np.linalg.LinAlgError:
            drift[index] = np.linalg.lstsq(
                matrix + np.eye(3) * 1.0e-9, rhs, rcond=None
            )[0][0]
        distance[index] = float(
            np.sqrt(np.median(np.sort(dist2[selected])[: min(15, len(selected))]))
        )
    return drift, distance


def kernel_mean(field: np.ndarray, own_wi: int, midpoint: np.ndarray) -> np.ndarray:
    keep = field[:, 3] != own_wi
    x, y, values = field[keep, 0], field[keep, 1], field[keep, 2]
    if not len(values):
        raise ValueError("exp226 kernel donor field is empty after self exclusion")
    output = np.empty(len(midpoint), dtype=np.float64)
    for index, point in enumerate(midpoint):
        dist2 = np.square(x - point[0]) + np.square(y - point[1])
        selected = safe_nearest_indices(dist2, np.arange(len(dist2)), 15)
        weights = np.exp(np.maximum(-dist2[selected] / (2.0 * 500.0**2), -700))
        output[index] = float(np.sum(weights * values[selected]) / np.sum(weights))
    return output


def theta_loc_at(
    surface: np.ndarray,
    midpoint: np.ndarray,
    own_wi: int,
    global_theta: float,
    params: K16Params,
) -> np.ndarray:
    output = np.empty(len(midpoint), dtype=np.float64)
    bandwidth = params.ancc_theta_bandwidth
    for index, point in enumerate(midpoint):
        dist2 = np.square(surface[:, 0] - point[0]) + np.square(surface[:, 1] - point[1])
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
    well: GeometryWell,
    fields: FieldPack,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not (np.abs(well.proj) < params.gate).any():
        return None
    theta = theta_loc_at(
        fields.surface_points, well.mid, well.wi, fields.global_theta, params
    )
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
    mask = (np.abs(well.proj[well.segid]) < params.gate) & (
        rotation < params.rot_max_deg
    )[well.segid]
    return local, mask


def build_geometry_columns(
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
        np.cumsum(np.where(bucket == index, raw_step, 0.0))
        for index in range(params.n_bins)
    ]
    columns += [
        np.cumsum(np.where(bucket == index, smooth_step, 0.0))
        for index in range(params.n_bins)
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
    design = build_geometry_columns(
        target, raw_field, smooth_field, donor_distance, params, substitute
    )
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
    oof_spec = get_nested(config, "data.exp226_oof")
    oof_path = resolve_existing(
        str(oof_spec["filename"]), [str(value) for value in oof_spec["candidates"]]
    )
    decompressed_sha = sha256_gzip_decompressed(oof_path)
    if decompressed_sha != str(oof_spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    fold_rows = pd.read_csv(oof_path, usecols=["well_id", "fold"], dtype={"well_id": str})
    if len(fold_rows) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row count mismatch")
    per_well = fold_rows.drop_duplicates().sort_values("well_id", kind="mergesort")
    if per_well["well_id"].duplicated().any():
        raise ValueError("exp226 OOF maps one well to multiple folds")
    fold_by_well = {str(row.well_id): int(row.fold) for row in per_well.itertuples(index=False)}
    if len(fold_by_well) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 OOF well count mismatch")

    kappa_spec = get_nested(config, "data.exp226_kappa_by_fold")
    kappa_path = resolve_existing(
        str(kappa_spec["filename"]), [str(value) for value in kappa_spec["candidates"]]
    )
    if sha256_path(kappa_path) != str(kappa_spec["expected_sha256"]):
        raise ValueError("exp226 kappa-by-fold SHA mismatch")
    kappa_frame = pd.read_csv(kappa_path)
    expected_terms = [
        *[f"raw_bin_{index}" for index in range(5)],
        *[f"smooth_bin_{index}" for index in range(5)],
        "sqrt_position",
        "near_strike_committee",
    ]
    kappa_by_fold: dict[int, np.ndarray] = {}
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
            "decompressed_sha256": decompressed_sha,
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
# ## 4. Pseudo-cut, Type Well, and hierarchical-prior helpers

# %%
class IneligiblePseudoCutError(ValueError):
    """A fixed-contract pseudo-cut exclusion, not a tunable fallback."""


def build_fixed_pseudocut(
    well: str,
    frame: pd.DataFrame,
    horizon_rows: int,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_target_safe_frame(frame)
    horizons = [int(value) for value in get_nested(config, "validation.pseudo_cut_horizons_rows")]
    if int(horizon_rows) not in horizons:
        raise ValueError("pseudo-cut horizon is outside the fixed contract")
    window_rows = int(get_nested(config, "validation.pseudo_cut_validation_window_rows"))
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    official_last_known = last_contiguous_known_index(tvt_input)
    cut = official_last_known - int(horizon_rows)
    if cut < 15:
        raise IneligiblePseudoCutError(
            f"well={well} horizon={horizon_rows} leaves fewer than 16 visible rows"
        )
    score_start = cut + 1
    score_stop = score_start + window_rows
    if score_stop - 1 > official_last_known:
        raise IneligiblePseudoCutError("fixed pseudo-tail window exceeds known prefix")
    held = frame.loc[
        score_start : score_stop - 1,
        ["id", "row_idx", "MD", "TVT_input"],
    ].copy()
    if len(held) != window_rows or held["TVT_input"].isna().any():
        raise IneligiblePseudoCutError("pseudo-tail truth is not a complete fixed window")
    held.insert(0, "well_id", str(well))
    held.insert(1, "horizon_rows", int(horizon_rows))
    held.insert(2, "cut_row", int(cut))
    held = held.rename(columns={"TVT_input": "held_tvt"})

    masked = frame.copy()
    masked.loc[cut + 1 :, "TVT_input"] = np.nan
    post_cut_finite = int(masked.loc[cut + 1 :, "TVT_input"].notna().sum())
    if post_cut_finite:
        raise ValueError("pseudo-cut mask did not remove every post-cut TVT_input")
    manifest = {
        "well_id": str(well),
        "horizon_rows": int(horizon_rows),
        "official_last_known_row": int(official_last_known),
        "cut_row": int(cut),
        "visible_rows": int(cut + 1),
        "validation_start_row": int(score_start),
        "validation_end_row": int(score_stop - 1),
        "validation_rows": int(window_rows),
        "post_cut_tvt_input_finite_rows_after_mask": post_cut_finite,
        "held_tvt_access_before_prediction_freeze": 0,
        "horizontal_truth_or_formation_columns_materialized": 0,
    }
    return masked, held.reset_index(drop=True), manifest


def load_typewell_template(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].interpolate(limit_direction="both")
    frame = frame.dropna(subset=["GR"])
    frame = frame.groupby("TVT", as_index=False, sort=True)["GR"].median()
    if len(frame) < 2 or not np.isfinite(frame[["TVT", "GR"]].to_numpy(np.float64)).all():
        raise ValueError(f"typewell {path.name} lacks a finite two-point TVT/GR template")
    return frame


def huber_affine_fit(
    template_gr: Sequence[float],
    observed_gr: Sequence[float],
    iterations: int = 12,
) -> tuple[float, float, float]:
    x = np.asarray(template_gr, dtype=np.float64)
    y = np.asarray(observed_gr, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 16:
        raise ValueError("known prefix has fewer than 16 finite GR calibration rows")
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    weights = np.ones(len(x), dtype=np.float64)
    for _ in range(int(iterations)):
        residual = y - design @ coefficients
        scale = robust_scale(residual, floor=1.0)
        normalized = np.abs(residual) / (1.345 * scale)
        weights = np.ones(len(x), dtype=np.float64)
        large = normalized > 1.0
        weights[large] = 1.0 / normalized[large]
        matrix = (design * weights[:, None]).T @ design
        rhs = (design * weights[:, None]).T @ y
        coefficients = np.linalg.lstsq(
            matrix + np.eye(2) * 1.0e-10, rhs, rcond=None
        )[0]
    residual = y - design @ coefficients
    sigma = robust_scale(residual, floor=1.0)
    return float(coefficients[0]), float(coefficients[1]), float(sigma)


def prepare_gr_calibration(
    masked_frame: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_target_safe_frame(masked_frame)
    known = masked_frame["TVT_input"].notna().to_numpy()
    known_tvt = pd.to_numeric(
        masked_frame.loc[known, "TVT_input"], errors="raise"
    ).to_numpy(np.float64)
    known_gr = pd.to_numeric(masked_frame.loc[known, "GR"], errors="coerce").to_numpy(
        np.float64
    )
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    template_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    affine_scale, affine_offset, sigma_raw = huber_affine_fit(template_known, known_gr)
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "physics.observations.gr.scale_clip")
    ]
    sigma = float(np.clip(sigma_raw, sigma_low, sigma_high))
    all_gr = pd.to_numeric(masked_frame["GR"], errors="coerce").to_numpy(np.float64)
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "affine_scale": affine_scale,
        "affine_offset": affine_offset,
        "gr_sigma_raw": float(sigma_raw),
        "gr_sigma": sigma,
        "all_gr": all_gr,
        "known_rows": int(known.sum()),
        "known_gr_finite_rows": int(np.isfinite(known_gr).sum()),
    }


def typewell_group_id(path: Path) -> str:
    return f"exact_{sha256_path(path)[:16]}"


def stable_spatial_neighbors(
    calibration: pd.DataFrame,
    query_x: float,
    query_y: float,
    k: int,
    *,
    exclude_well: str | None = None,
) -> pd.DataFrame:
    required = {"well_id", "x_center", "y_center"}
    if not required.issubset(calibration.columns):
        raise ValueError("spatial calibration table is missing required columns")
    candidates = calibration.copy()
    if exclude_well is not None:
        candidates = candidates.loc[candidates["well_id"].astype(str) != str(exclude_well)]
    if candidates.empty:
        raise ValueError("spatial hyperprior has no outer-train candidates")
    x_values = candidates["x_center"].to_numpy(np.float64)
    y_values = candidates["y_center"].to_numpy(np.float64)
    x_scale = robust_scale(x_values, floor=1.0)
    y_scale = robust_scale(y_values, floor=1.0)
    distance = np.sqrt(
        np.square((x_values - float(query_x)) / x_scale)
        + np.square((y_values - float(query_y)) / y_scale)
    )
    candidates = candidates.assign(distance_standardized=distance)
    candidates = candidates.sort_values(
        ["distance_standardized", "well_id"], kind="mergesort"
    ).head(min(int(k), len(candidates)))
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "neighbor_rank", np.arange(1, len(candidates) + 1, dtype=np.int64))
    return candidates


HYPERPARAMETER_LOG_COLUMNS = (
    "log_prefix_noise",
    "log_jump_scale",
    "log_reset_hazard",
    "log_gr_noise",
)


def build_hyperposterior(
    calibration: pd.DataFrame,
    *,
    well: str,
    fold: int,
    typewell_group: str,
    x_center: float,
    y_center: float,
    current_gr_sigma: float,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if calibration.empty or not set(HYPERPARAMETER_LOG_COLUMNS).issubset(calibration.columns):
        raise ValueError("outer-train hyperprior calibration is empty or incomplete")
    pooled_location = calibration[list(HYPERPARAMETER_LOG_COLUMNS)].median()
    pooled_variance = calibration[list(HYPERPARAMETER_LOG_COLUMNS)].var(ddof=1).fillna(0.0)
    pooled_variance = pooled_variance.clip(lower=0.05**2)
    group = calibration.loc[calibration["typewell_group"].eq(str(typewell_group))]
    minimum_group = int(
        get_nested(config, "physics.hierarchy.typewell_prior.minimum_group_wells")
    )
    group_count = len(group)
    if group_count:
        group_location_raw = group[list(HYPERPARAMETER_LOG_COLUMNS)].median()
        group_weight = min(float(group_count) / float(minimum_group), 1.0)
        group_location = (
            group_weight * group_location_raw + (1.0 - group_weight) * pooled_location
        )
    else:
        group_weight = 0.0
        group_location = pooled_location.copy()

    neighbors = stable_spatial_neighbors(
        calibration,
        float(x_center),
        float(y_center),
        int(get_nested(config, "physics.hierarchy.spatial_prior.neighbors")),
        exclude_well=well,
    )
    local_variance = neighbors[list(HYPERPARAMETER_LOG_COLUMNS)].var(ddof=1).fillna(0.0)
    local_variance = local_variance.clip(lower=0.05**2)
    variance_ratio = (local_variance / pooled_variance).clip(lower=0.5, upper=2.0)
    posterior_variance = pooled_variance * variance_ratio

    gr_column = "log_gr_noise"
    current_log_gr = math.log(max(float(current_gr_sigma), 1.0e-6))
    current_variance = 0.25**2
    gr_prior_precision = 1.0 / float(posterior_variance[gr_column])
    current_precision = 1.0 / current_variance
    group_location[gr_column] = (
        float(group_location[gr_column]) * gr_prior_precision
        + current_log_gr * current_precision
    ) / (gr_prior_precision + current_precision)

    base_hazard = float(get_nested(config, "physics.transition.base_reset_hazard_per_row"))
    maximum_hazard = float(
        get_nested(config, "physics.transition.maximum_event_hazard_per_row")
    )
    values = {
        "prefix_noise": float(np.exp(group_location["log_prefix_noise"])),
        "jump_scale": float(np.exp(group_location["log_jump_scale"])),
        "reset_hazard_per_row": float(
            np.clip(np.exp(group_location["log_reset_hazard"]), base_hazard, maximum_hazard)
        ),
        "gr_noise": float(np.exp(group_location["log_gr_noise"])),
    }
    values.update(
        {
            "well_id": str(well),
            "fold": int(fold),
            "typewell_group": str(typewell_group),
            "typewell_group_outer_train_wells": int(group_count),
            "typewell_group_weight": float(group_weight),
            "spatial_neighbor_count": int(len(neighbors)),
            "datum_location_ft": 0.0,
            "datum_mean_from_typewell_or_neighbor": 0.0,
            "jump_sign_from_typewell_or_neighbor": 0.0,
            "spatial_role": "variance_only",
            "pooled_prior_content_sha": dataframe_content_sha(
                calibration.sort_values("well_id", kind="mergesort")
            ),
        }
    )
    neighbor_output = neighbors[
        ["neighbor_rank", "well_id", "distance_standardized"]
    ].copy()
    neighbor_output.insert(0, "query_well_id", str(well))
    neighbor_output.insert(1, "fold", int(fold))
    return values, neighbor_output


def reliability_from_history(nll_excess_history: Sequence[float]) -> float:
    history = np.asarray(nll_excess_history, dtype=np.float64)
    history = history[np.isfinite(history)]
    if not len(history):
        return 1.0
    return float(np.clip(np.exp(-np.median(history)), 0.10, 1.00))


# %% [markdown]
# ## 5. Exact bounded semi-Markov forward-backward solver

# %%
def state_grid_from_config(config: Mapping[str, Any]) -> np.ndarray:
    minimum = float(get_nested(config, "physics.state_space.minimum_ft"))
    maximum = float(get_nested(config, "physics.state_space.maximum_ft"))
    step = float(get_nested(config, "physics.state_space.step_ft"))
    grid = np.arange(minimum, maximum + step * 0.5, step, dtype=np.float64)
    if len(grid) != int(get_nested(config, "physics.state_space.state_count")):
        raise ValueError("state grid count does not match the fixed contract")
    if grid[0] != minimum or grid[-1] != maximum:
        raise ValueError("state grid bounds do not match the fixed contract")
    return grid


def student_t_log_density(
    residual: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    sigma = max(float(scale), 1.0e-9)
    df = float(degrees_of_freedom)
    normalized = np.asarray(residual, dtype=np.float64) / sigma
    return -math.log(sigma) - 0.5 * (df + 1.0) * np.log1p(np.square(normalized) / df)


def student_t_nll(
    residual: Sequence[float],
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    return -student_t_log_density(
        np.asarray(residual, dtype=np.float64), scale, degrees_of_freedom
    )


def build_gr_block_log_likelihood(
    base_geometry: Sequence[float],
    observed_gr: Sequence[float],
    calibration: Mapping[str, Any],
    hyperposterior: Mapping[str, Any],
    reliability: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    base = np.asarray(base_geometry, dtype=np.float64)
    observed = np.asarray(observed_gr, dtype=np.float64)
    if len(base) != len(observed) or not len(base) or not np.isfinite(base).all():
        raise ValueError("GR likelihood base/observation alignment is invalid")
    states = state_grid_from_config(config)
    candidate_tvt = base[:, None] + states[None, :]
    typewell_tvt = np.asarray(calibration["typewell_tvt"], dtype=np.float64)
    typewell_gr = np.asarray(calibration["typewell_gr"], dtype=np.float64)
    expected = np.empty_like(candidate_tvt)
    for state_index in range(len(states)):
        expected[:, state_index] = np.interp(
            candidate_tvt[:, state_index], typewell_tvt, typewell_gr
        )
    expected = (
        float(calibration["affine_scale"]) * expected
        + float(calibration["affine_offset"])
    )
    df = float(get_nested(config, "physics.observations.gr.degrees_of_freedom"))
    sigma = float(hyperposterior["gr_noise"])
    row_log_likelihood = np.zeros_like(candidate_tvt)
    finite = np.isfinite(observed)
    if finite.any():
        residual = observed[finite, None] - expected[finite]
        row_log_likelihood[finite] = student_t_log_density(residual, sigma, df)
    stride = int(get_nested(config, "physics.state_space.checkpoint_stride_rows"))
    blocks = [
        np.arange(start, min(start + stride, len(base)), dtype=np.int64)
        for start in range(0, len(base), stride)
    ]
    block_values: list[np.ndarray] = []
    observed_fraction: list[float] = []
    for block in blocks:
        block_finite = finite[block]
        observed_fraction.append(float(block_finite.mean()))
        if block_finite.any():
            selected = row_log_likelihood[block[block_finite]]
            values = selected.mean(axis=0)
            values = values - float(logsumexp(values))
        else:
            values = np.zeros(len(states), dtype=np.float64)
        block_values.append(float(reliability) * values)
    output = np.vstack(block_values)
    if not np.isfinite(output).all():
        raise ValueError("GR block likelihood is not finite")
    return output, blocks, np.asarray(observed_fraction, dtype=np.float64)


def robust_rank_unit(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    if len(array) == 1:
        return np.ones(1, dtype=np.float64)
    return ranks / float(len(array) - 1)


def build_target_free_event_evidence(
    block_log_likelihood: np.ndarray,
    blocks: Sequence[np.ndarray],
    base_geometry: Sequence[float],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    likelihood = np.asarray(block_log_likelihood, dtype=np.float64)
    states = state_grid_from_config(config)
    stride = int(get_nested(config, "physics.state_space.checkpoint_stride_rows"))
    windows = [
        int(value) for value in get_nested(config, "physics.transition.event_gate.windows_rows")
    ]
    required_consistency = int(
        get_nested(config, "physics.transition.event_gate.minimum_consistent_shift_windows")
    )
    signs: list[int] = []
    entropy_drop: list[float] = []
    sign_consistency: list[float] = []
    for checkpoint in range(len(likelihood)):
        checkpoint_signs: list[int] = []
        checkpoint_entropy: list[float] = []
        for window in windows:
            block_count = max(int(math.ceil(window / stride)), 1)
            start = max(0, checkpoint - block_count + 1)
            score = likelihood[start : checkpoint + 1].mean(axis=0)
            probability = np.exp(score - float(logsumexp(score)))
            selected = float(states[int(np.argmax(score))])
            checkpoint_signs.append(int(np.sign(selected)))
            entropy = -float(np.sum(probability * np.log(np.clip(probability, 1.0e-300, 1.0))))
            checkpoint_entropy.append(1.0 - entropy / math.log(len(states)))
        nonzero = [value for value in checkpoint_signs if value != 0]
        positive = nonzero.count(1)
        negative = nonzero.count(-1)
        consistent = max(positive, negative) >= required_consistency
        modal_sign = 1 if positive >= negative and positive else (-1 if negative else 0)
        signs.append(modal_sign if consistent else 0)
        entropy_drop.append(float(np.mean(checkpoint_entropy)))
        sign_consistency.append(1.0 if consistent else 0.0)

    base = np.asarray(base_geometry, dtype=np.float64)
    curvature = np.abs(np.gradient(np.gradient(base))) if len(base) >= 3 else np.zeros(len(base))
    block_curvature = np.asarray(
        [float(np.median(curvature[np.asarray(block, dtype=np.int64)])) for block in blocks],
        dtype=np.float64,
    )
    entropy_rank = robust_rank_unit(entropy_drop)
    curvature_rank = robust_rank_unit(block_curvature)
    evidence = (
        np.asarray(sign_consistency, dtype=np.float64) + entropy_rank + curvature_rank
    ) / 3.0
    return pd.DataFrame(
        {
            "checkpoint": np.arange(len(likelihood), dtype=np.int64),
            "shift_sign": signs,
            "shift_sign_consistent": sign_consistency,
            "posterior_entropy_drop_rank": entropy_rank,
            "geometry_curvature_rank": curvature_rank,
            "event_evidence": evidence,
            "truth_columns_used": 0,
        }
    )


def checkpoint_hazard(
    event_evidence: float,
    event_threshold: float,
    base_hazard_per_row: float,
    config: Mapping[str, Any],
) -> float:
    stride = int(get_nested(config, "physics.state_space.checkpoint_stride_rows"))
    maximum_per_row = float(
        get_nested(config, "physics.transition.maximum_event_hazard_per_row")
    )
    base_per_row = float(
        np.clip(
            base_hazard_per_row,
            float(get_nested(config, "physics.transition.base_reset_hazard_per_row")),
            maximum_per_row,
        )
    )
    if float(event_evidence) < float(event_threshold):
        per_row = base_per_row
    else:
        denominator = max(1.0 - float(event_threshold), 1.0e-12)
        strength = np.clip(
            (float(event_evidence) - float(event_threshold)) / denominator, 0.0, 1.0
        )
        per_row = base_per_row + strength * (maximum_per_row - base_per_row)
    return float(1.0 - (1.0 - per_row) ** stride)


def build_transition_log_matrix(
    states: Sequence[float],
    phase_count: int,
    hazard: float,
    jump_scale: float,
) -> np.ndarray:
    grid = np.asarray(states, dtype=np.float64)
    phases = int(phase_count)
    expanded = len(grid) * phases
    probability = np.zeros((expanded, expanded), dtype=np.float64)
    for state_index in range(len(grid)):
        for phase in range(phases - 1):
            source = state_index * phases + phase
            destination = state_index * phases + phase + 1
            probability[source, destination] = 1.0
        source = state_index * phases + phases - 1
        probability[source, source] = 1.0 - float(hazard)
        weights = np.exp(-np.abs(grid - grid[state_index]) / max(float(jump_scale), 1.0e-9))
        weights[state_index] = 0.0
        weights /= weights.sum()
        for destination_state, weight in enumerate(weights):
            if weight:
                destination = destination_state * phases
                probability[source, destination] = float(hazard) * float(weight)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-12):
        raise ValueError("semi-Markov transition rows are not normalized")
    output = np.full_like(probability, -np.inf)
    positive = probability > 0
    output[positive] = np.log(probability[positive])
    return output


def exact_semi_markov_forward_backward(
    block_log_likelihood: np.ndarray,
    hazards: Sequence[float],
    *,
    datum_location: float,
    prior_scale: float,
    jump_scale: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    emission_by_state = np.asarray(block_log_likelihood, dtype=np.float64)
    states = state_grid_from_config(config)
    phase_count = int(get_nested(config, "physics.state_space.duration_phase_count_per_offset"))
    expanded_count = len(states) * phase_count
    if emission_by_state.ndim != 2 or emission_by_state.shape[1] != len(states):
        raise ValueError("semi-Markov emission shape mismatch")
    if len(hazards) != max(len(emission_by_state) - 1, 0):
        raise ValueError("semi-Markov checkpoint hazard length mismatch")
    emission = np.repeat(emission_by_state, phase_count, axis=1)
    initial = np.full(expanded_count, -np.inf, dtype=np.float64)
    prior = student_t_log_density(
        states - float(datum_location),
        float(prior_scale),
        float(get_nested(config, "physics.observations.prefix.degrees_of_freedom")),
    )
    prior -= float(logsumexp(prior))
    initial[np.arange(len(states), dtype=np.int64) * phase_count] = prior

    transitions = [
        build_transition_log_matrix(states, phase_count, float(hazard), float(jump_scale))
        for hazard in hazards
    ]
    alpha = np.full((len(emission), expanded_count), -np.inf, dtype=np.float64)
    alpha[0] = initial + emission[0]
    alpha[0] -= float(logsumexp(alpha[0]))
    for checkpoint in range(1, len(emission)):
        alpha[checkpoint] = emission[checkpoint] + logsumexp(
            alpha[checkpoint - 1][:, None] + transitions[checkpoint - 1], axis=0
        )
        alpha[checkpoint] -= float(logsumexp(alpha[checkpoint]))

    beta = np.zeros_like(alpha)
    for checkpoint in range(len(emission) - 2, -1, -1):
        beta[checkpoint] = logsumexp(
            transitions[checkpoint]
            + emission[checkpoint + 1][None, :]
            + beta[checkpoint + 1][None, :],
            axis=1,
        )
        beta[checkpoint] -= float(logsumexp(beta[checkpoint]))
    posterior_expanded = alpha + beta
    posterior_expanded -= logsumexp(posterior_expanded, axis=1)[:, None]
    posterior_expanded = np.exp(posterior_expanded)
    posterior_state = posterior_expanded.reshape(
        len(emission), len(states), phase_count
    ).sum(axis=2)
    posterior_state /= posterior_state.sum(axis=1, keepdims=True)
    posterior_mean = posterior_state @ states
    entropy = -np.sum(
        posterior_state * np.log(np.clip(posterior_state, 1.0e-300, 1.0)), axis=1
    )

    reset_probability = np.zeros(len(emission), dtype=np.float64)
    eligible_phase = phase_count - 1
    for checkpoint in range(1, len(emission)):
        transition = transitions[checkpoint - 1]
        pair_log = (
            alpha[checkpoint - 1][:, None]
            + transition
            + emission[checkpoint][None, :]
            + beta[checkpoint][None, :]
        )
        pair_log -= float(logsumexp(pair_log))
        mask = np.zeros_like(pair_log, dtype=bool)
        for source_state in range(len(states)):
            source = source_state * phase_count + eligible_phase
            destinations = np.arange(len(states), dtype=np.int64) * phase_count
            destinations = destinations[destinations != source_state * phase_count]
            mask[source, destinations] = True
        reset_probability[checkpoint] = float(np.exp(pair_log[mask]).sum())
    if not (
        np.isfinite(posterior_mean).all()
        and np.isfinite(entropy).all()
        and np.isfinite(reset_probability).all()
    ):
        raise ValueError("semi-Markov posterior contains non-finite values")
    if (posterior_mean < states[0] - 1.0e-12).any() or (
        posterior_mean > states[-1] + 1.0e-12
    ).any():
        raise ValueError("posterior mean violates the absolute datum bound")
    return {
        "posterior_mean_checkpoint": posterior_mean,
        "posterior_entropy_checkpoint": entropy,
        "reset_probability_checkpoint": reset_probability,
        "posterior_state": posterior_state,
        "underflow_count": int(np.count_nonzero(posterior_state == 0.0)),
    }


def interpolate_checkpoint_summary(
    checkpoint_values: Sequence[float],
    blocks: Sequence[np.ndarray],
    row_count: int,
) -> np.ndarray:
    values = np.asarray(checkpoint_values, dtype=np.float64)
    if len(values) != len(blocks):
        raise ValueError("checkpoint summary/block count mismatch")
    centers = np.asarray([float(np.mean(block)) for block in blocks], dtype=np.float64)
    rows = np.arange(int(row_count), dtype=np.float64)
    return np.interp(rows, centers, values, left=values[0], right=values[-1])


def solve_stage0_window(
    base_geometry: Sequence[float],
    observed_gr: Sequence[float],
    calibration: Mapping[str, Any],
    hyperposterior: Mapping[str, Any],
    reliability: float,
    event_gate_threshold: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    block_likelihood, blocks, observed_fraction = build_gr_block_log_likelihood(
        base_geometry,
        observed_gr,
        calibration,
        hyperposterior,
        reliability,
        config,
    )
    event = build_target_free_event_evidence(
        block_likelihood, blocks, base_geometry, config
    )
    event["gate_open"] = event["event_evidence"] >= float(event_gate_threshold)
    hazards = [
        checkpoint_hazard(
            float(event.iloc[index]["event_evidence"]),
            float(event_gate_threshold),
            float(hyperposterior["reset_hazard_per_row"]),
            config,
        )
        for index in range(1, len(event))
    ]
    step = float(get_nested(config, "physics.state_space.step_ft"))
    prior_scale = max(float(hyperposterior["prefix_noise"]) * float(reliability), step)
    posterior = exact_semi_markov_forward_backward(
        block_likelihood,
        hazards,
        datum_location=0.0,
        prior_scale=prior_scale,
        jump_scale=float(hyperposterior["jump_scale"]),
        config=config,
    )
    row_count = len(np.asarray(base_geometry))
    mean_delta = interpolate_checkpoint_summary(
        posterior["posterior_mean_checkpoint"], blocks, row_count
    )
    entropy = interpolate_checkpoint_summary(
        posterior["posterior_entropy_checkpoint"], blocks, row_count
    )
    reset_probability = interpolate_checkpoint_summary(
        posterior["reset_probability_checkpoint"], blocks, row_count
    )
    return {
        "posterior_mean_delta": mean_delta,
        "posterior_entropy": entropy,
        "reset_probability": reset_probability,
        "event_evidence": event,
        "block_observed_gr_fraction": observed_fraction,
        "underflow_count": int(posterior["underflow_count"]),
        "prior_scale": float(prior_scale),
        "hazards": hazards,
    }


# %% [markdown]
# ## 6. Prediction freeze, truth attachment, metrics, and guards

# %%
def build_stage0_prediction_frame(
    *,
    well: str,
    fold: int,
    horizon_rows: int,
    cut_row: int,
    masked_frame: pd.DataFrame,
    base_geometry: Sequence[float],
    solver: Mapping[str, Any],
    reliability: float,
    event_gate_threshold: float,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    window_rows = int(get_nested(config, "validation.pseudo_cut_validation_window_rows"))
    base = np.asarray(base_geometry, dtype=np.float64)[:window_rows]
    delta = np.asarray(solver["posterior_mean_delta"], dtype=np.float64)
    entropy = np.asarray(solver["posterior_entropy"], dtype=np.float64)
    reset = np.asarray(solver["reset_probability"], dtype=np.float64)
    if not (len(base) == len(delta) == len(entropy) == len(reset) == window_rows):
        raise ValueError("Stage 0 prediction arrays do not match the fixed window")
    rows = np.arange(cut_row + 1, cut_row + 1 + window_rows, dtype=np.int64)
    output = pd.DataFrame(
        {
            "well_id": str(well),
            "fold": int(fold),
            "horizon_rows": int(horizon_rows),
            "cut_row": int(cut_row),
            "row_idx": rows,
            "MD": masked_frame.iloc[rows]["MD"].to_numpy(np.float64),
            "base_geometry": base,
            "posterior_mean_delta": delta,
            "prediction": base + delta,
            "posterior_entropy": entropy,
            "reset_probability": reset,
            "reliability": float(reliability),
            "event_gate_threshold": float(event_gate_threshold),
            "truth_attached": False,
        }
    )
    if tuple(output.columns) != FROZEN_PREDICTION_COLUMNS:
        raise ValueError("Stage 0 frozen prediction schema mismatch")
    values = output[
        [
            "base_geometry",
            "posterior_mean_delta",
            "prediction",
            "posterior_entropy",
            "reset_probability",
            "reliability",
        ]
    ].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Stage 0 prediction contains non-finite values")
    minimum = float(get_nested(config, "physics.state_space.minimum_ft"))
    maximum = float(get_nested(config, "physics.state_space.maximum_ft"))
    if (output["posterior_mean_delta"] < minimum - 1.0e-12).any() or (
        output["posterior_mean_delta"] > maximum + 1.0e-12
    ).any():
        raise ValueError("Stage 0 prediction violates the absolute correction bound")
    return output


def freeze_stage0_prediction(prediction: pd.DataFrame) -> str:
    leaked = sorted(PREDICTION_FORBIDDEN_COLUMNS.intersection(prediction.columns))
    if leaked:
        raise ValueError(f"pre-freeze prediction contains truth-derived columns: {leaked}")
    if tuple(prediction.columns) != FROZEN_PREDICTION_COLUMNS:
        raise ValueError("pre-freeze prediction schema mismatch")
    if not prediction["truth_attached"].eq(False).all():  # noqa: E712
        raise ValueError("truth was attached before prediction freeze")
    return dataframe_content_sha(prediction, FROZEN_PREDICTION_COLUMNS)


def attach_pseudotail_truth(
    prediction: pd.DataFrame,
    held_truth: pd.DataFrame,
    *,
    frozen_prediction_sha: str,
    nll_scale: float,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, float]:
    if not frozen_prediction_sha:
        raise ValueError("truth attachment requires a frozen prediction content SHA")
    actual_sha = dataframe_content_sha(prediction, FROZEN_PREDICTION_COLUMNS)
    if actual_sha != str(frozen_prediction_sha):
        raise ValueError("frozen prediction content SHA does not match current prediction")
    required = {"well_id", "horizon_rows", "cut_row", "row_idx", "MD", "held_tvt"}
    if not required.issubset(held_truth.columns):
        raise ValueError("held pseudo-tail truth schema is incomplete")
    merged = prediction.merge(
        held_truth[list(required)],
        on=["well_id", "horizon_rows", "cut_row", "row_idx", "MD"],
        how="left",
        validate="one_to_one",
    )
    if merged["held_tvt"].isna().any():
        raise ValueError("held pseudo-tail truth does not cover every frozen prediction row")
    merged["truth_attached"] = True
    merged["base_error"] = merged["held_tvt"] - merged["base_geometry"]
    merged["model_error"] = merged["held_tvt"] - merged["prediction"]
    merged["correction_sign_match"] = (
        np.sign(merged["posterior_mean_delta"]) == np.sign(merged["base_error"])
    )
    df = float(get_nested(config, "physics.observations.prefix.degrees_of_freedom"))
    base_nll = student_t_nll(merged["base_error"], nll_scale, df)
    model_nll = student_t_nll(merged["model_error"], nll_scale, df)
    nll_excess = float(np.mean(model_nll - base_nll))
    merged["nll_excess"] = nll_excess
    merged["frozen_prediction_sha256"] = str(frozen_prediction_sha)
    return merged, nll_excess


def metric_row(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    base_rmse = rmse(frame["held_tvt"], frame["base_geometry"])
    model_rmse = rmse(frame["held_tvt"], frame["prediction"])
    eligible = frame["base_error"].abs() >= 5.0
    sign_accuracy = (
        float(frame.loc[eligible, "correction_sign_match"].mean())
        if eligible.any()
        else float("nan")
    )
    return {
        "scope": str(scope),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "base_rmse": float(base_rmse),
        "model_rmse": float(model_rmse),
        "rmse_improvement_ft": float(base_rmse - model_rmse),
        "correction_sign_rows_abs_base_error_ge5": int(eligible.sum()),
        "correction_sign_accuracy_abs_base_error_ge5": sign_accuracy,
        "correction_abs_mean": float(frame["posterior_mean_delta"].abs().mean()),
        "correction_abs_max": float(frame["posterior_mean_delta"].abs().max()),
        "posterior_entropy_mean": float(frame["posterior_entropy"].mean()),
        "reset_probability_mean": float(frame["reset_probability"].mean()),
    }


def build_stage0_metrics(
    scored_predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if scored_predictions.empty or not scored_predictions["truth_attached"].eq(True).all():  # noqa: E712
        raise ValueError("Stage 0 metrics require scored post-freeze predictions")
    overall = metric_row(scored_predictions, "overall")
    fold_rows = [
        metric_row(part, f"fold_{int(fold)}")
        for fold, part in scored_predictions.groupby("fold", sort=True)
    ]
    fold_metrics = pd.DataFrame(fold_rows)
    by_well_rows: list[dict[str, Any]] = []
    for (fold, well), part in scored_predictions.groupby(["fold", "well_id"], sort=True):
        base_value = rmse(part["held_tvt"], part["base_geometry"])
        model_value = rmse(part["held_tvt"], part["prediction"])
        by_well_rows.append(
            {
                "fold": int(fold),
                "well_id": str(well),
                "rows": int(len(part)),
                "base_rmse": base_value,
                "model_rmse": model_value,
                "rmse_improvement_ft": base_value - model_value,
                "mean_nll_excess": float(part["nll_excess"].mean()),
                "final_reliability": float(
                    part.sort_values("horizon_rows", ascending=False)["reliability"].iloc[-1]
                ),
            }
        )
    by_well = pd.DataFrame(by_well_rows).sort_values(["fold", "well_id"], kind="mergesort")
    base_p95 = float(by_well["base_rmse"].quantile(0.95))
    model_p95 = float(by_well["model_rmse"].quantile(0.95))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_windows = len(get_nested(config, "validation.pseudo_cut_horizons_rows"))
    expected_window_rows = int(get_nested(config, "validation.pseudo_cut_validation_window_rows"))
    expected_rows = expected_wells * expected_windows * expected_window_rows
    technical = {
        "expected_rows": expected_rows,
        "actual_rows": int(len(scored_predictions)),
        "expected_wells": expected_wells,
        "actual_wells": int(scored_predictions["well_id"].nunique()),
        "expected_windows_per_well": expected_windows,
        "actual_min_windows_per_well": int(
            scored_predictions.groupby("well_id")["horizon_rows"].nunique().min()
        ),
        "folds": sorted(int(value) for value in scored_predictions["fold"].unique()),
        "finite_prediction_coverage": float(
            np.isfinite(scored_predictions["prediction"].to_numpy(np.float64)).mean()
        ),
        "correction_bound_violations": int(
            (
                scored_predictions["posterior_mean_delta"].abs()
                > float(get_nested(config, "physics.state_space.maximum_ft")) + 1.0e-12
            ).sum()
        ),
        "unique_frozen_window_hashes": int(
            scored_predictions["frozen_prediction_sha256"].nunique()
        ),
    }
    guards = get_nested(config, "stages.stage0.guards")
    fold_improvements = int((fold_metrics["rmse_improvement_ft"] > 0.0).sum())
    checks = {
        "technical_row_coverage": technical["actual_rows"] == technical["expected_rows"],
        "technical_well_coverage": technical["actual_wells"] == technical["expected_wells"],
        "technical_window_coverage": technical["actual_min_windows_per_well"]
        == expected_windows,
        "technical_fold_coverage": technical["folds"]
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
        "technical_finite_prediction": technical["finite_prediction_coverage"] == 1.0,
        "technical_correction_bound": technical["correction_bound_violations"] == 0,
        "technical_truth_after_freeze": technical["unique_frozen_window_hashes"]
        == expected_wells * expected_windows,
        "pooled_rmse_improvement": float(overall["rmse_improvement_ft"])
        >= float(guards["pooled_rmse_improvement_ft_min"]),
        "correction_sign_accuracy": float(
            overall["correction_sign_accuracy_abs_base_error_ge5"]
        )
        >= float(guards["correction_sign_accuracy_abs_base_error_ge5_min"]),
        "fold_improvement_count": fold_improvements
        >= int(guards["fold_improvement_count_min"]),
        "well_rmse_p95_nonworse": model_p95 <= base_p95 + 1.0e-12,
    }
    summary = {
        "overall": overall,
        "technical": technical,
        "well_rmse_p95": {"base": base_p95, "model": model_p95},
        "fold_improvement_count": fold_improvements,
        "guard_checks": checks,
        "technical_guard_passed": bool(
            all(value for key, value in checks.items() if key.startswith("technical_"))
        ),
        "scientific_guard_passed": bool(all(checks.values())),
        "failure_policy": get_nested(config, "stages.stage0.failure_policy"),
    }
    return fold_metrics, by_well, summary


# %% [markdown]
# ## 7. Full Kaggle CPU orchestration and generated artifacts

# %%
def list_horizontal_paths(raw_dir: Path) -> dict[str, Path]:
    paths = {
        path.name.split("__")[0]: path
        for path in sorted(raw_dir.glob("*__horizontal_well.csv"))
    }
    if not paths:
        raise FileNotFoundError(f"no horizontal wells found under {raw_dir}")
    return paths


def calibration_geometry_frame(
    well: str,
    horizon: int,
    cut: int,
    base_geometry: Sequence[float],
) -> pd.DataFrame:
    base = np.asarray(base_geometry, dtype=np.float64)
    return pd.DataFrame(
        {
            "well_id": str(well),
            "horizon_rows": int(horizon),
            "cut_row": int(cut),
            "row_offset": np.arange(1, len(base) + 1, dtype=np.int64),
            "base_geometry": base,
            "truth_attached": False,
        }
    )


def build_outer_train_hyperprior_calibration(
    *,
    fold: int,
    source_wells: Sequence[GeometryWell],
    horizontal_paths: Mapping[str, Path],
    fields: FieldPack,
    kappa: np.ndarray,
    params: K16Params,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    window_rows = int(get_nested(config, "validation.pseudo_cut_validation_window_rows"))
    horizons = [int(value) for value in get_nested(config, "validation.pseudo_cut_horizons_rows")]
    step = float(get_nested(config, "physics.state_space.step_ft"))
    base_hazard = float(get_nested(config, "physics.transition.base_reset_hazard_per_row"))
    maximum_hazard = float(
        get_nested(config, "physics.transition.maximum_event_hazard_per_row")
    )
    fixed_jump_scale = float(get_nested(config, "physics.transition.reset_jump_scale_ft"))
    rows: list[dict[str, Any]] = []
    event_rows: list[pd.DataFrame] = []
    input_manifest: list[dict[str, Any]] = []
    for source in sorted(source_wells, key=lambda item: item.wid):
        horizontal_path = horizontal_paths[source.wid]
        typewell_path = horizontal_path.with_name(f"{source.wid}__typewell.csv")
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        safe = load_target_safe_horizontal(horizontal_path)
        typewell = load_typewell_template(typewell_path)
        residual_parts: list[np.ndarray] = []
        window_medians: list[float] = []
        gr_sigmas: list[float] = []
        base_hashes: list[str] = []
        for horizon in horizons:
            masked, held, manifest = build_fixed_pseudocut(
                source.wid, safe, horizon, config
            )
            target = build_target_geometry_well(
                source.wid,
                masked,
                cut=int(manifest["cut_row"]),
                params=params,
                wi=source.wi,
            )
            full_geometry, _ = replay_exp226_geometry(target, fields, kappa, params)
            base = full_geometry[:window_rows]
            frozen_base = calibration_geometry_frame(
                source.wid, horizon, int(manifest["cut_row"]), base
            )
            base_hash = dataframe_content_sha(frozen_base)
            if not base_hash:
                raise ValueError("outer-train calibration base geometry did not freeze")
            base_hashes.append(base_hash)
            truth = held["held_tvt"].to_numpy(np.float64)
            residual = truth - base
            residual_parts.append(residual)
            window_medians.append(float(np.median(residual)))
            gr_calibration = prepare_gr_calibration(masked, typewell, config)
            gr_sigmas.append(float(gr_calibration["gr_sigma"]))
            temporary_hyperposterior = {
                "gr_noise": float(gr_calibration["gr_sigma"]),
            }
            score_rows = np.arange(
                int(manifest["cut_row"]) + 1,
                int(manifest["cut_row"]) + 1 + window_rows,
                dtype=np.int64,
            )
            block_likelihood, blocks, _ = build_gr_block_log_likelihood(
                base,
                masked.iloc[score_rows]["GR"].to_numpy(np.float64),
                gr_calibration,
                temporary_hyperposterior,
                1.0,
                config,
            )
            event = build_target_free_event_evidence(
                block_likelihood, blocks, base, config
            )
            event.insert(0, "well_id", source.wid)
            event.insert(1, "fold", int(fold))
            event.insert(2, "horizon_rows", int(horizon))
            event_rows.append(event)
        all_residual = np.concatenate(residual_parts)
        differences = np.diff(np.asarray(window_medians, dtype=np.float64))
        prefix_noise = robust_scale(all_residual, floor=step)
        jump_scale = (
            float(max(np.median(np.abs(differences)), step))
            if len(differences)
            else fixed_jump_scale
        )
        reset_events = int(np.count_nonzero(np.abs(differences) >= fixed_jump_scale))
        empirical_hazard = reset_events / float(max(len(horizons) * window_rows, 1))
        reset_hazard = float(np.clip(empirical_hazard, base_hazard, maximum_hazard))
        x_center = float(np.median(safe["X"].to_numpy(np.float64)))
        y_center = float(np.median(safe["Y"].to_numpy(np.float64)))
        group = typewell_group_id(typewell_path)
        row = {
            "fold": int(fold),
            "well_id": source.wid,
            "typewell_group": group,
            "x_center": x_center,
            "y_center": y_center,
            "prefix_noise": prefix_noise,
            "jump_scale": jump_scale,
            "reset_hazard": reset_hazard,
            "gr_noise": float(np.median(gr_sigmas)),
            "log_prefix_noise": math.log(prefix_noise),
            "log_jump_scale": math.log(jump_scale),
            "log_reset_hazard": math.log(reset_hazard),
            "log_gr_noise": math.log(float(np.median(gr_sigmas))),
            "datum_location_ft": 0.0,
            "datum_mean_from_typewell_or_neighbor": 0.0,
            "jump_sign_from_typewell_or_neighbor": 0.0,
            "calibration_windows": len(horizons),
            "calibration_rows": len(all_residual),
            "base_geometry_freeze_sha256": mapping_sha256(
                {str(horizon): sha for horizon, sha in zip(horizons, base_hashes, strict=True)}
            ),
        }
        rows.append(row)
        input_manifest.extend(
            [
                {
                    "fold": int(fold),
                    "well_id": source.wid,
                    "role": "outer_train_geometry_and_hyperprior_source_horizontal",
                    "path": str(horizontal_path),
                    "raw_sha256": sha256_path(horizontal_path),
                },
                {
                    "fold": int(fold),
                    "well_id": source.wid,
                    "role": "outer_train_typewell_scale_prior",
                    "path": str(typewell_path),
                    "raw_sha256": sha256_path(typewell_path),
                },
            ]
        )
    calibration = pd.DataFrame(rows).sort_values("well_id", kind="mergesort")
    event_frame = pd.concat(event_rows, ignore_index=True).sort_values(
        ["well_id", "horizon_rows", "checkpoint"], kind="mergesort"
    )
    numeric = calibration.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("outer-train hyperprior calibration contains non-finite values")
    return calibration, event_frame, input_manifest


def state_space_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    states = state_grid_from_config(config)
    phase_count = int(get_nested(config, "physics.state_space.duration_phase_count_per_offset"))
    base_hazard = checkpoint_hazard(
        0.0,
        1.0,
        float(get_nested(config, "physics.transition.base_reset_hazard_per_row")),
        config,
    )
    transition = build_transition_log_matrix(
        states,
        phase_count,
        base_hazard,
        float(get_nested(config, "physics.transition.reset_jump_scale_ft")),
    )
    finite_transition = np.where(np.isfinite(transition), transition, -1.0e300)
    return {
        "states": states.tolist(),
        "state_count": len(states),
        "duration_phase_count": phase_count,
        "expanded_state_count": len(states) * phase_count,
        "checkpoint_stride_rows": int(
            get_nested(config, "physics.state_space.checkpoint_stride_rows")
        ),
        "minimum_segment_duration_rows": int(
            get_nested(config, "physics.state_space.minimum_segment_duration_rows")
        ),
        "base_checkpoint_hazard": base_hazard,
        "transition_log_matrix_sha256": hashlib.sha256(
            finite_transition.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "solver": "exact_log_space_semi_markov_forward_backward",
        "dtype": "float64",
        "logsumexp_implementation": "embedded_numpy_stable_max_shift",
        "numpy_version": np.__version__,
        "candidate_bank": False,
        "viterbi_output": False,
        "cumulative_random_walk": False,
    }


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    started = time.perf_counter()
    raw_dir = train_data_dir(config)
    horizontal_paths = list_horizontal_paths(raw_dir)
    fold_by_well, kappa_by_fold, exp226_manifest, _ = load_exp226_fold_contract(config)
    if set(horizontal_paths) != set(fold_by_well):
        missing_raw = sorted(set(fold_by_well) - set(horizontal_paths))
        missing_fold = sorted(set(horizontal_paths) - set(fold_by_well))
        raise ValueError(
            f"raw/fold well identity mismatch: missing_raw={missing_raw[:5]} "
            f"missing_fold={missing_fold[:5]}"
        )
    params = geometry_params_from_config(config)
    folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    horizons = [int(value) for value in get_nested(config, "validation.pseudo_cut_horizons_rows")]
    window_rows = int(get_nested(config, "validation.pseudo_cut_validation_window_rows"))

    prediction_parts: list[pd.DataFrame] = []
    pseudocut_rows: list[dict[str, Any]] = []
    hyperposterior_rows: list[dict[str, Any]] = []
    neighbor_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    calibration_event_parts: list[pd.DataFrame] = []
    input_manifest: list[dict[str, Any]] = list(exp226_manifest)
    fold_gate_thresholds: dict[int, float] = {}

    for fold in folds:
        source_ids = sorted(well for well, assigned in fold_by_well.items() if assigned != fold)
        valid_ids = sorted(well for well, assigned in fold_by_well.items() if assigned == fold)
        source_wells = [
            load_source_geometry_well(horizontal_paths[well], params, wi=index)
            for index, well in enumerate(source_ids)
        ]
        wi_by_well = {source.wid: source.wi for source in source_wells}
        fields = build_fields(source_wells, params)
        calibration, calibration_event, calibration_inputs = (
            build_outer_train_hyperprior_calibration(
                fold=fold,
                source_wells=source_wells,
                horizontal_paths=horizontal_paths,
                fields=fields,
                kappa=kappa_by_fold[fold],
                params=params,
                config=config,
            )
        )
        calibration_parts.append(calibration)
        calibration_event_parts.append(calibration_event)
        input_manifest.extend(calibration_inputs)
        threshold_quantile = float(
            get_nested(config, "physics.transition.event_gate.outer_train_quantile")
        )
        event_threshold = float(calibration_event["event_evidence"].quantile(threshold_quantile))
        if not np.isfinite(event_threshold):
            raise ValueError("outer-train event gate threshold is not finite")
        fold_gate_thresholds[fold] = event_threshold

        for well in valid_ids:
            horizontal_path = horizontal_paths[well]
            typewell_path = horizontal_path.with_name(f"{well}__typewell.csv")
            if not typewell_path.exists():
                raise FileNotFoundError(typewell_path)
            safe = load_target_safe_horizontal(horizontal_path)
            typewell = load_typewell_template(typewell_path)
            group = typewell_group_id(typewell_path)
            x_center = float(np.median(safe["X"].to_numpy(np.float64)))
            y_center = float(np.median(safe["Y"].to_numpy(np.float64)))
            input_manifest.extend(
                [
                    {
                        "fold": fold,
                        "well_id": well,
                        "role": "outer_valid_target_safe_horizontal",
                        "path": str(horizontal_path),
                        "raw_sha256": sha256_path(horizontal_path),
                    },
                    {
                        "fold": fold,
                        "well_id": well,
                        "role": "outer_valid_typewell_likelihood",
                        "path": str(typewell_path),
                        "raw_sha256": sha256_path(typewell_path),
                    },
                ]
            )
            nll_history: list[float] = []
            for horizon in horizons:
                masked, held, pseudocut_manifest = build_fixed_pseudocut(
                    well, safe, horizon, config
                )
                pseudocut_manifest["fold"] = fold
                pseudocut_rows.append(pseudocut_manifest)
                target = build_target_geometry_well(
                    well,
                    masked,
                    cut=int(pseudocut_manifest["cut_row"]),
                    params=params,
                    wi=wi_by_well.get(well, -1),
                )
                full_geometry, donor_distance = replay_exp226_geometry(
                    target, fields, kappa_by_fold[fold], params
                )
                base = full_geometry[:window_rows]
                gr_calibration = prepare_gr_calibration(masked, typewell, config)
                hyperposterior, neighbors = build_hyperposterior(
                    calibration,
                    well=well,
                    fold=fold,
                    typewell_group=group,
                    x_center=x_center,
                    y_center=y_center,
                    current_gr_sigma=float(gr_calibration["gr_sigma"]),
                    config=config,
                )
                hyperposterior.update(
                    {
                        "horizon_rows": horizon,
                        "event_gate_threshold": event_threshold,
                        "gr_affine_scale": float(gr_calibration["affine_scale"]),
                        "gr_affine_offset": float(gr_calibration["affine_offset"]),
                        "gr_sigma_raw": float(gr_calibration["gr_sigma_raw"]),
                    }
                )
                hyperposterior_rows.append(hyperposterior)
                neighbors.insert(2, "horizon_rows", int(horizon))
                neighbor_parts.append(neighbors)
                reliability = reliability_from_history(nll_history)
                score_rows = np.arange(
                    int(pseudocut_manifest["cut_row"]) + 1,
                    int(pseudocut_manifest["cut_row"]) + 1 + window_rows,
                    dtype=np.int64,
                )
                solver = solve_stage0_window(
                    base,
                    masked.iloc[score_rows]["GR"].to_numpy(np.float64),
                    gr_calibration,
                    hyperposterior,
                    reliability,
                    event_threshold,
                    config,
                )
                event = solver["event_evidence"].copy()
                event.insert(0, "well_id", well)
                event.insert(1, "fold", fold)
                event.insert(2, "horizon_rows", horizon)
                event["reliability"] = reliability
                event_parts.append(event)
                prediction = build_stage0_prediction_frame(
                    well=well,
                    fold=fold,
                    horizon_rows=horizon,
                    cut_row=int(pseudocut_manifest["cut_row"]),
                    masked_frame=masked,
                    base_geometry=base,
                    solver=solver,
                    reliability=reliability,
                    event_gate_threshold=event_threshold,
                    config=config,
                )
                frozen_sha = freeze_stage0_prediction(prediction)
                scored, nll_excess = attach_pseudotail_truth(
                    prediction,
                    held,
                    frozen_prediction_sha=frozen_sha,
                    nll_scale=float(hyperposterior["prefix_noise"]),
                    config=config,
                )
                scored["donor_distance"] = donor_distance[:window_rows]
                scored["prior_scale"] = float(solver["prior_scale"])
                scored["solver_underflow_count"] = int(solver["underflow_count"])
                prediction_parts.append(scored)
                nll_history.append(nll_excess)

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "horizon_rows", "row_idx"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    pseudocuts = pd.DataFrame(pseudocut_rows).sort_values(
        ["fold", "well_id", "horizon_rows"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    hyperpriors = pd.DataFrame(hyperposterior_rows).sort_values(
        ["fold", "well_id", "horizon_rows"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    neighbors = pd.concat(neighbor_parts, ignore_index=True).sort_values(
        ["fold", "query_well_id", "horizon_rows", "neighbor_rank"],
        kind="mergesort",
    )
    events = pd.concat(event_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "horizon_rows", "checkpoint"], kind="mergesort"
    )
    calibration = pd.concat(calibration_parts, ignore_index=True).sort_values(
        ["fold", "well_id"], kind="mergesort"
    )
    calibration_events = pd.concat(calibration_event_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "horizon_rows", "checkpoint"], kind="mergesort"
    )
    input_frame = pd.DataFrame(input_manifest).sort_values(
        ["fold", "well_id", "role"], na_position="first", kind="mergesort"
    )
    fold_metrics, by_well, metric_summary = build_stage0_metrics(predictions, config)

    artifacts = artifact_dir()
    config_path = project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml"
    if not config_path.exists():
        config_path = Path.cwd() / "config.yaml"
    state_manifest = state_space_manifest(config)
    state_manifest_path = artifacts / f"{OUTPUT_PREFIX}_state_space_manifest.json"
    write_json(state_manifest_path, state_manifest)
    state_manifest_sha = sha256_path(state_manifest_path)
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage0_known_prefix_pseudotail_identifiability_audit",
        "active_contracts": 1,
        "ml_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "control_retraining": 0,
        "inference": False,
        "submission": False,
        "horizons_rows": horizons,
        "validation_window_rows": window_rows,
        "fold_gate_thresholds": fold_gate_thresholds,
        "config_sha256": sha256_path(config_path),
        "state_space_manifest_sha256": state_manifest_sha,
        "prediction_truth_policy": "freeze_content_sha_before_held_tvt_join",
        "datum_mean_from_typewell_or_neighbor": False,
        "jump_sign_from_typewell_or_neighbor": False,
        "stage1_implemented": False,
    }
    contract_path = artifacts / f"{OUTPUT_PREFIX}_contract.json"
    write_json(contract_path, contract)

    input_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    pseudocut_path = artifacts / f"{OUTPUT_PREFIX}_pseudocut_manifest.csv"
    calibration_path = artifacts / f"{OUTPUT_PREFIX}_outer_train_calibration.csv"
    calibration_event_path = artifacts / f"{OUTPUT_PREFIX}_outer_train_event_evidence.csv"
    hyperprior_path = artifacts / f"{OUTPUT_PREFIX}_hyperprior_manifest.csv"
    neighbor_path = artifacts / f"{OUTPUT_PREFIX}_spatial_neighbor_manifest.csv"
    event_path = artifacts / f"{OUTPUT_PREFIX}_event_evidence.csv"
    prediction_path = artifacts / f"{OUTPUT_PREFIX}_stage0_predictions.csv.gz"
    fold_metric_path = artifacts / f"{OUTPUT_PREFIX}_stage0_fold_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_stage0_by_well_metrics.csv"
    stage0_metric_path = artifacts / f"{OUTPUT_PREFIX}_stage0_metrics.json"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_stage0_summary.json"
    input_frame.to_csv(input_path, index=False)
    pseudocuts.to_csv(pseudocut_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    calibration_events.to_csv(calibration_event_path, index=False)
    hyperpriors.to_csv(hyperprior_path, index=False)
    neighbors.to_csv(neighbor_path, index=False)
    events.to_csv(event_path, index=False)
    prediction_artifact = write_csv_gzip(predictions, prediction_path)
    fold_metrics.to_csv(fold_metric_path, index=False)
    by_well.to_csv(by_well_path, index=False)

    elapsed = time.perf_counter() - started
    runtime_limit_hours = float(get_nested(config, "runtime.kaggle_time_limit_hours"))
    required_margin_hours = float(get_nested(config, "runtime.required_runtime_margin_hours"))
    runtime_margin_hours = runtime_limit_hours - elapsed / 3600.0
    metric_summary["runtime"] = {
        "elapsed_seconds": elapsed,
        "time_limit_hours": runtime_limit_hours,
        "required_margin_hours": required_margin_hours,
        "actual_margin_hours": runtime_margin_hours,
    }
    metric_summary["guard_checks"]["technical_runtime_margin"] = (
        runtime_margin_hours >= required_margin_hours
    )
    metric_summary["technical_guard_passed"] = bool(
        all(
            value
            for key, value in metric_summary["guard_checks"].items()
            if key.startswith("technical_")
        )
    )
    metric_summary["scientific_guard_passed"] = bool(
        all(metric_summary["guard_checks"].values())
    )
    write_json(stage0_metric_path, metric_summary)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage0_completed"
            if metric_summary["technical_guard_passed"]
            else "stage0_failed"
        ),
        "stage": "known_prefix_pseudotail_identifiability_audit",
        "runtime_seconds": elapsed,
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "metric_summary": metric_summary,
        "fold_gate_thresholds": fold_gate_thresholds,
        "reproducibility": {
            "config_sha256": sha256_path(config_path),
            "fold_map_sha256": mapping_sha256(fold_by_well),
            "input_manifest_content_sha256": dataframe_content_sha(input_frame),
            "base_geometry_and_prediction_decompressed_sha256": prediction_artifact[
                "decompressed_sha256"
            ],
            "typewell_mapping_sha256": dataframe_content_sha(
                hyperpriors[["fold", "well_id", "horizon_rows", "typewell_group"]]
            ),
            "spatial_neighbor_content_sha256": dataframe_content_sha(neighbors),
            "pseudocut_content_sha256": dataframe_content_sha(pseudocuts),
            "outer_train_calibration_content_sha256": dataframe_content_sha(calibration),
            "hyperprior_content_sha256": dataframe_content_sha(hyperpriors),
            "state_space_manifest_sha256": state_manifest_sha,
            "posterior_summary_content_sha256": dataframe_content_sha(events),
            "prediction_raw_gzip_sha256": prediction_artifact["raw_sha256"],
            "prediction_decompressed_sha256": prediction_artifact["decompressed_sha256"],
            "deterministic_anchor": False,
            "kaggle_kernel_version": None,
        },
        "artifacts": {
            "contract": str(contract_path),
            "input_manifest": str(input_path),
            "pseudocut_manifest": str(pseudocut_path),
            "outer_train_calibration": str(calibration_path),
            "outer_train_event_evidence": str(calibration_event_path),
            "hyperprior_manifest": str(hyperprior_path),
            "spatial_neighbor_manifest": str(neighbor_path),
            "event_evidence": str(event_path),
            "stage0_predictions": prediction_artifact,
            "fold_metrics": str(fold_metric_path),
            "by_well_metrics": str(by_well_path),
            "stage0_metrics": str(stage0_metric_path),
            "state_space_manifest": str(state_manifest_path),
        },
        "next_action": (
            "request_separate_stage1_approval"
            if metric_summary["scientific_guard_passed"]
            else "close_branch_without_parameter_rescue"
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(summary_path, summary)
    write_json(metrics_output_path(), summary)
    return summary


# %% [markdown]
# ## 8. Setup and contract preview

# %%
config = load_experiment_config()
validate_scientific_contract(config)
contract_preview = {
    "experiment": get_nested(config, "experiment.name"),
    "route": get_nested(config, "experiment.route"),
    "parent": get_nested(config, "lineage.parent"),
    "stage": get_nested(config, "execution.stage"),
    "active_variants": get_nested(config, "execution.active_variants"),
    "lightgbm_configs": get_nested(config, "execution.lightgbm_config_count"),
    "trained_folds": get_nested(config, "execution.trained_fold_count"),
    "boosters": get_nested(config, "execution.total_boosters"),
    "control_retraining": get_nested(config, "execution.control_or_parent_retraining"),
    "pseudo_cut_horizons_rows": get_nested(config, "validation.pseudo_cut_horizons_rows"),
    "pseudo_tail_window_rows": get_nested(
        config, "validation.pseudo_cut_validation_window_rows"
    ),
    "state_count": get_nested(config, "physics.state_space.state_count"),
    "expanded_state_count": get_nested(config, "physics.state_space.expanded_state_count"),
    "inference_enabled": get_nested(config, "inference.enabled"),
    "kaggle_push_approved": get_nested(config, "execution.kaggle_push_approved"),
}
if EXECUTE_NOTEBOOK:
    print(json.dumps(to_jsonable(contract_preview), indent=2, sort_keys=True))


# %% [markdown]
# ## 9. Run the fixed Stage 0 audit
#
# The first full execution is intentionally Kaggle CPU only.  Stage 1, raw-test
# inference, submission, parameter rescue, selector, blend, and posthoc well
# correction remain outside this notebook and require separate approval.

# %%
if EXECUTE_NOTEBOOK:
    stage0_result = run_stage0(config)
    print(json.dumps(to_jsonable(stage0_result), indent=2, sort_keys=True))
