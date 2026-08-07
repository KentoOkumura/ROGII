# %% [markdown]
# # exp291 prefix-masked Type Well GR multimode safe beam
#
# This zero-booster PF/Beam diagnostic masks the last 640 rows of each known
# `TVT_input` prefix, retains every eligible local maximum in a fixed Type Well
# GR shift bank together with the fold-safe exp226 geometry base, and permits a
# mode commit only after persistent checkpoint evidence. Held-out truth is read
# only after all target-free tables have been persisted and content-hashed.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Exp226 fold-safe geometry replay helpers
# 4. Pseudo-mask, Type Well emission, and local-mode bank helpers
# 5. Safe-anchored branch path and future evidence helpers
# 6. Persistent checkpoint policy and truth-freeze helpers
# 7. Post-freeze metrics and scientific guards
# 8. Full Kaggle CPU orchestration and generated artifacts
# 9. Setup and contract preview
# 10. Run controlled backtest and report generated artifacts

# %% [markdown]
# ## 1. Imports and fixed experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp291_prefix_masked_typewell_gr_multimode_safe_beam"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
HORIZONTAL_INPUT_COLUMNS = ("X", "Y", "Z", "MD", "GR", "TVT_input")
TARGET_SAFE_COLUMNS = ("id", *HORIZONTAL_INPUT_COLUMNS)
TARGET_FORBIDDEN_COLUMNS = {
    "TVT",
    "tvt_true",
    "target",
    "error",
    "abs_error",
    "truth_best",
}
CHECKPOINTS = (128, 256, 512)
POLICIES = (
    "safe_base_only",
    "safe_plus_top1_typewell_mode",
    "safe_plus_all_typewell_modes",
    "safe_plus_matched_count_shuffled_modes",
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP291_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
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
        return [to_jsonable(item) for item in value.tolist()]
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
    raise FileNotFoundError(f"exp291 config not found in {[str(path) for path in candidates]}")


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


def dataframe_content_sha(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
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


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def list_horizontal_wells(data_dir: Path) -> list[str]:
    return [
        path.name.split("__")[0] for path in sorted(Path(data_dir).glob("*__horizontal_well.csv"))
    ]


def validate_target_safe_frame(frame: pd.DataFrame) -> None:
    leaked = sorted(TARGET_FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"target-safe frame contains forbidden truth columns: {leaked}")
    missing = sorted(set(TARGET_SAFE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"target-safe frame is missing {missing}")


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(HORIZONTAL_INPUT_COLUMNS))
    frame = frame.loc[:, list(HORIZONTAL_INPUT_COLUMNS)]
    well = path.name.removesuffix("__horizontal_well.csv")
    frame.insert(0, "id", [f"{well}:{row_idx}" for row_idx in range(len(frame))])
    validate_target_safe_frame(frame)
    return frame


def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp291 fixes route=pf_beam")
    fixed = {
        "pseudo_mask.masked_rows": 640,
        "pseudo_mask.minimum_visible_rows_before_cut": 512,
        "pseudo_mask.visible_shift_score_rows": 128,
        "pseudo_mask.primary_horizon_rows": 256,
        "evidence.primary_horizon_rows": 256,
        "model.active_contract_count": 1,
        "model.active_variant_count": 1,
        "model.fixed_policy_count": 4,
        "model.lightgbm_config_count": 0,
        "model.trained_fold_count": 0,
        "model.booster_count": 0,
        "model.hmm_regeneration_count": 0,
        "model.pf_regeneration_count": 0,
        "execution.total_boosters": 0,
        "execution.hmm_well_runs": 0,
        "execution.pf_well_runs": 0,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"exp291 requires {key}={expected}, got {actual}")
    shifts = [float(value) for value in get_nested(config, "candidate_bank.shift_bank_ft")]
    if shifts != [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
        raise ValueError("exp291 fixes the approved exp280 13-value shift bank")
    if (
        tuple(int(value) for value in get_nested(config, "evidence.checkpoints_rows"))
        != CHECKPOINTS
    ):
        raise ValueError("exp291 fixes checkpoints at 128/256/512 rows")
    if tuple(get_nested(config, "policies.fixed")) != POLICIES:
        raise ValueError("exp291 fixes exactly four policies")
    if (
        get_nested(config, "candidate_bank.reference_path_on_visible_score_rows")
        != "observed_visible_tvt_input"
    ):
        raise ValueError("visible local-mode scoring must use target-free observed TVT_input")
    if get_nested(config, "candidate_bank.local_maximum_fallback") != "safe_only":
        raise ValueError("exp291 forbids forced alternative fallback")
    if not bool(get_nested(config, "candidate_bank.always_keep_safe_base")):
        raise ValueError("exp291 requires an unpruned safe base")
    forbidden = set(get_nested(config, "candidate_bank.forbidden_sources", []))
    required_forbidden = {"same_well_self_gr", "ncc", "self_gr_donor", "self_gr_shuffle"}
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp291 must explicitly forbid every same-well self-GR source")
    if bool(get_nested(config, "execution.gpu")):
        raise ValueError("exp291 is CPU-only")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("exp291 forbids inference and submission")
    if bool(get_nested(config, "model.parent_control_retraining")):
        raise ValueError("exp291 forbids parent/control retraining")


# %% [markdown]
# ## 3. Exp226 fold-safe geometry replay helpers


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
    columns = ["X", "Y", "Z", "TVT", "TVT_input", "ANCC"]
    frame = pd.read_csv(path, usecols=columns)
    x = frame["X"].to_numpy(np.float64)
    y = frame["Y"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
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
        anc=frame["ANCC"].to_numpy(np.float64),
        c_raw=fit_coeffs(r0, u, n, params, rho=0.0),
        c_sm=fit_coeffs(r0, u, n, params, rho=params.smooth_rho),
    )


def build_target_geometry_well(
    well: str,
    masked_frame: pd.DataFrame,
    *,
    cut: int,
    original_last_known: int,
    params: K16Params,
) -> GeometryWell:
    validate_target_safe_frame(masked_frame)
    if masked_frame.loc[cut + 1 : original_last_known, "TVT_input"].notna().any():
        raise ValueError("target geometry reader received unmasked post-cut TVT_input")
    x = pd.to_numeric(masked_frame["X"], errors="raise").to_numpy(np.float64)
    y = pd.to_numeric(masked_frame["Y"], errors="raise").to_numpy(np.float64)
    z = pd.to_numeric(masked_frame["Z"], errors="raise").to_numpy(np.float64)
    ti = pd.to_numeric(masked_frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    n = int(original_last_known - cut)
    ndz = -np.diff(z)[cut:original_last_known]
    if len(ndz) != n or n <= 0:
        raise ValueError("pseudo-suffix geometry length mismatch")
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
        candidates = np.arange(len(dist2))
        selected = _safe_nearest_indices(dist2, candidates, params.local_linear_k)
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
) -> np.ndarray:
    raw_field, donor_distance = local_linear(fields.f_raw, target.wi, target.mid, params)
    smooth_field, _ = local_linear(fields.f_sm, target.wi, target.mid, params)
    substitute = committee_inputs(target, fields, params)
    design = build_columns(target, raw_field, smooth_field, donor_distance, params, substitute)
    if design.shape != (target.n, params.kappa_dim) or len(kappa) != params.kappa_dim:
        raise ValueError("exp226 geometry replay design/kappa shape mismatch")
    path = target.anchor + design @ np.asarray(kappa, dtype=np.float64)
    if not np.isfinite(path).all():
        raise ValueError("exp226 pseudo-suffix geometry is not finite")
    return path


def load_exp226_fold_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, int], dict[int, np.ndarray], list[dict[str, Any]]]:
    oof_spec = get_nested(config, "data.exp226_oof")
    oof_path = resolve_existing(
        str(oof_spec["filename"]), [str(value) for value in oof_spec["candidates"]]
    )
    actual_decompressed = sha256_gzip_decompressed(oof_path)
    if actual_decompressed != str(oof_spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    fold_rows = pd.read_csv(oof_path, usecols=["well_id", "fold"], dtype={"well_id": str})
    if len(fold_rows) != int(get_nested(config, "validation.expected_rows")):
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
            "rows": len(kappa_frame),
            "wells": 0,
        },
    ]
    return fold_by_well, kappa_by_fold, manifests


# %% [markdown]
# ## 4. Pseudo-mask, Type Well emission, and local-mode bank helpers


# %%
class IneligibleWellError(ValueError):
    """A fixed-contract eligibility exclusion, not a pipeline failure."""


def build_pseudo_mask(
    well: str, frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_target_safe_frame(frame)
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    try:
        last_known = last_contiguous_known_index(tvt_input)
    except ValueError as exc:
        raise IneligibleWellError(str(exc)) from exc
    masked_rows = int(get_nested(config, "pseudo_mask.masked_rows"))
    minimum_visible = int(get_nested(config, "pseudo_mask.minimum_visible_rows_before_cut"))
    cut = int(last_known - masked_rows)
    if cut + 1 < minimum_visible:
        raise IneligibleWellError("insufficient_visible_rows_before_cut")
    if last_known - cut != masked_rows:
        raise IneligibleWellError("masked_suffix_length_mismatch")
    required = frame.iloc[cut + 1 : last_known + 1]
    if len(required) != masked_rows or required["TVT_input"].isna().any():
        raise IneligibleWellError("masked_suffix_is_not_contiguous_finite")
    numeric_required = frame.loc[cut:last_known, ["X", "Y", "Z", "MD"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric_required.to_numpy(np.float64)).all():
        raise IneligibleWellError("masked_suffix_geometry_is_not_finite")
    masked = frame.copy(deep=True)
    masked.loc[cut + 1 : last_known, "TVT_input"] = np.nan
    if masked.loc[cut + 1 :, "TVT_input"].notna().any():
        raise ValueError("post-cut TVT_input remained visible after masking")
    ids = frame.loc[cut + 1 : last_known, "id"].astype(str).tolist()
    manifest = {
        "well_id": str(well),
        "horizontal_rows": len(frame),
        "original_last_known_row": last_known,
        "cut_row": cut,
        "visible_rows": cut + 1,
        "masked_rows": masked_rows,
        "masked_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "post_cut_tvt_input_finite_rows_after_mask": int(
            masked.loc[cut + 1 :, "TVT_input"].notna().sum()
        ),
        "target_truth_columns_exposed": 0,
    }
    return masked, manifest


def prepare_typewell_emission(
    masked_frame: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    cut: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_target_safe_frame(masked_frame)
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell requires TVT and GR")
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw.to_numpy(np.float64)).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    visible_tvt = pd.to_numeric(masked_frame.loc[:cut, "TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    visible_gr = pd.to_numeric(masked_frame.loc[:cut, "GR"], errors="coerce")
    gr_fill = float(np.nanmean(typewell_gr))
    visible_gr = visible_gr.interpolate(limit_direction="both").fillna(gr_fill)
    finite = np.isfinite(visible_tvt)
    if int(finite.sum()) < 4:
        raise ValueError("visible prefix is too short for emission calibration")
    expected = np.interp(visible_tvt[finite], typewell_tvt, typewell_gr)
    residual = visible_gr.to_numpy(np.float64)[finite] - expected
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "evidence.emission.sigma_clip")
    ]
    sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("visible-prefix emission sigma is invalid")
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "gr_sigma": sigma,
        "gr_fill": gr_fill,
        "visible_residual_mean": float(np.mean(residual)),
        "visible_residual_std_unclipped": float(np.std(residual)),
    }


def row_log_likelihood(
    observed_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    emission: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    observed = np.asarray(observed_gr, dtype=np.float64)
    candidate = np.asarray(candidate_tvt, dtype=np.float64)
    if observed.shape != candidate.shape:
        raise ValueError("GR and candidate TVT shapes differ")
    expected = np.interp(
        candidate,
        np.asarray(emission["typewell_tvt"], dtype=np.float64),
        np.asarray(emission["typewell_gr"], dtype=np.float64),
    )
    zscore = (observed - expected) / float(emission["gr_sigma"])
    clip = float(get_nested(config, "evidence.emission.log_likelihood_clip"))
    score = -0.5 * np.minimum(zscore**2, clip)
    if not np.isfinite(score).all():
        raise ValueError("typewell evidence contains non-finite row scores")
    return score


def eligible_local_maximum_slots(
    scores: Sequence[float],
    shifts: Sequence[float],
    minimum_abs_shift: float,
) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    bank = np.asarray(shifts, dtype=np.float64)
    if values.shape != bank.shape or not np.isfinite(values).all():
        raise ValueError("local-mode selection requires one finite score per shift")
    selected: list[int] = []
    for index in np.flatnonzero(np.abs(bank) >= float(minimum_abs_shift)):
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index + 1 < len(values) else -np.inf
        if values[index] >= left and values[index] >= right:
            selected.append(int(index))
    return selected


def score_visible_shift_bank(
    well: str,
    masked_frame: pd.DataFrame,
    emission: Mapping[str, Any],
    *,
    cut: int,
    fold: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    validate_target_safe_frame(masked_frame)
    score_rows = int(get_nested(config, "pseudo_mask.visible_shift_score_rows"))
    start = cut - score_rows + 1
    if start < 0:
        raise IneligibleWellError("insufficient_visible_rows_for_shift_score")
    reference = pd.to_numeric(masked_frame.loc[start:cut, "TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    if len(reference) != score_rows or not np.isfinite(reference).all():
        raise IneligibleWellError("visible_shift_reference_is_not_finite")
    observed = pd.to_numeric(masked_frame.loc[start:cut, "GR"], errors="coerce")
    observed = observed.interpolate(limit_direction="both").fillna(float(emission["gr_fill"]))
    observed_values = observed.to_numpy(np.float64)
    shifts = np.asarray(get_nested(config, "candidate_bank.shift_bank_ft"), dtype=np.float64)
    likelihood = np.asarray(
        [
            row_log_likelihood(observed_values, reference + shift, emission, config).mean()
            for shift in shifts
        ],
        dtype=np.float64,
    )
    minimum_abs = float(get_nested(config, "candidate_bank.minimum_alternative_abs_shift_ft"))
    local_slots = eligible_local_maximum_slots(likelihood, shifts, minimum_abs)
    local_by_score = sorted(local_slots, key=lambda slot: (-likelihood[slot], slot))
    rank_by_slot = {slot: rank for rank, slot in enumerate(local_by_score, start=1)}
    output = pd.DataFrame(
        {
            "well_id": str(well),
            "fold": int(fold),
            "cut_row": int(cut),
            "score_start_row": int(start),
            "score_end_row": int(cut),
            "shift_slot": np.arange(len(shifts), dtype=np.int16),
            "shift_ft": shifts,
            "visible_likelihood_mean": likelihood,
            "eligible_alternative": np.abs(shifts) >= minimum_abs,
            "eligible_local_maximum": False,
            "real_mode_rank": pd.array([pd.NA] * len(shifts), dtype="Int16"),
            "is_top1_real_mode": False,
            "truth_attached": False,
        }
    )
    for slot in local_slots:
        output.loc[slot, "eligible_local_maximum"] = True
        output.loc[slot, "real_mode_rank"] = rank_by_slot[slot]
    if local_by_score:
        output.loc[local_by_score[0], "is_top1_real_mode"] = True
    return output


# %% [markdown]
# ## 5. Safe-anchored candidate bank and matched-count control


# %%
def build_candidate_bank(
    well: str,
    shift_scores: pd.DataFrame,
    *,
    fold: int,
    cut: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    ordered = shift_scores.sort_values("shift_slot", kind="mergesort").reset_index(drop=True)
    if ordered["well_id"].nunique() != 1 or str(ordered["well_id"].iloc[0]) != str(well):
        raise ValueError("shift score identity mismatch")
    real = ordered.loc[ordered["eligible_local_maximum"]].copy()
    real = real.sort_values(["real_mode_rank", "shift_slot"], kind="mergesort")
    eligible = ordered.loc[ordered["eligible_alternative"]].copy()
    real_count = len(real)
    if real_count > len(eligible):
        raise ValueError("local-mode count exceeds fixed eligible shift bank")
    seed = int(get_nested(config, "policies.matched_shuffle.seed"))
    rng = np.random.default_rng(
        stable_seed(EXPERIMENT_NAME, well, cut, "matched_count_shuffled_modes", seed)
    )
    if real_count:
        shuffled_slots = sorted(
            int(value)
            for value in rng.choice(
                eligible["shift_slot"].to_numpy(np.int64),
                size=real_count,
                replace=False,
            )
        )
        shuffled = ordered.loc[ordered["shift_slot"].isin(shuffled_slots)].copy()
    else:
        shuffled = ordered.iloc[0:0].copy()

    zero = ordered.loc[ordered["shift_ft"].eq(0.0)]
    if len(zero) != 1:
        raise ValueError("fixed shift bank requires exactly one zero-shift slot")
    safe = zero.iloc[0]
    rows: list[dict[str, Any]] = [
        {
            "well_id": str(well),
            "fold": int(fold),
            "cut_row": int(cut),
            "candidate_id": "safe_base",
            "branch_id": "safe_base",
            "branch_type": "safe_base",
            "branch_order": 0,
            "control": "core",
            "source": "exp226_group_safe_geometry",
            "shift_slot": int(safe["shift_slot"]),
            "shift_ft": 0.0,
            "visible_likelihood_mean": float(safe["visible_likelihood_mean"]),
            "visible_rank": 0,
            "is_top1_real_mode": False,
            "truth_attached": False,
        }
    ]
    for item in real.itertuples(index=False):
        slot = int(item.shift_slot)
        rows.append(
            {
                "well_id": str(well),
                "fold": int(fold),
                "cut_row": int(cut),
                "candidate_id": f"real_mode_slot_{slot:02d}",
                "branch_id": f"real_mode_slot_{slot:02d}",
                "branch_type": "typewell_gr_local_mode",
                "branch_order": 1 + slot,
                "control": "real",
                "source": "typewell_gr_visible_local_maximum",
                "shift_slot": slot,
                "shift_ft": float(item.shift_ft),
                "visible_likelihood_mean": float(item.visible_likelihood_mean),
                "visible_rank": int(item.real_mode_rank),
                "is_top1_real_mode": bool(item.is_top1_real_mode),
                "truth_attached": False,
            }
        )
    for rank, item in enumerate(
        shuffled.sort_values("shift_slot", kind="mergesort").itertuples(index=False), start=1
    ):
        slot = int(item.shift_slot)
        rows.append(
            {
                "well_id": str(well),
                "fold": int(fold),
                "cut_row": int(cut),
                "candidate_id": f"shuffled_mode_slot_{slot:02d}",
                "branch_id": f"shuffled_mode_slot_{slot:02d}",
                "branch_type": "matched_count_shuffled_mode",
                "branch_order": 100 + slot,
                "control": "shuffled",
                "source": "score_blind_matched_count_shift_bank",
                "shift_slot": slot,
                "shift_ft": float(item.shift_ft),
                "visible_likelihood_mean": float(item.visible_likelihood_mean),
                "visible_rank": rank,
                "is_top1_real_mode": False,
                "truth_attached": False,
            }
        )
    output = pd.DataFrame(rows).sort_values("branch_order", kind="mergesort").reset_index(drop=True)
    if int((output["control"] == "real").sum()) != real_count:
        raise ValueError("not every eligible Type Well local mode was retained")
    if int((output["control"] == "shuffled").sum()) != real_count:
        raise ValueError("matched-count control did not preserve the real candidate count")
    if int((output["branch_id"] == "safe_base").sum()) != 1:
        raise ValueError("safe base candidate was not retained exactly once")
    if output["branch_id"].duplicated().any():
        raise ValueError("candidate branch identity is not unique")
    forbidden_tokens = ("selfgr", "self_gr", "ncc", "donor")
    if output["source"].astype(str).str.lower().apply(
        lambda value: any(token in value for token in forbidden_tokens)
    ).any():
        raise ValueError("same-well self-GR source entered the exp291 candidate bank")
    return output


# %% [markdown]
# ## 6. Persistent checkpoint policy and truth-freeze helpers


# %%
def build_branch_paths(
    well: str,
    *,
    fold: int,
    cut: int,
    safe_geometry: np.ndarray,
    candidate_bank: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    masked_rows = int(get_nested(config, "pseudo_mask.masked_rows"))
    if len(safe_geometry) != masked_rows or not np.isfinite(safe_geometry).all():
        raise ValueError("safe geometry must cover exactly the 640-row pseudo suffix")
    path_rows = max(CHECKPOINTS)
    row_idx = np.arange(cut + 1, cut + 1 + path_rows, dtype=np.int64)
    safe_path = np.asarray(safe_geometry[:path_rows], dtype=np.float64)
    if len(safe_path) != path_rows:
        raise ValueError("safe geometry does not cover every fixed checkpoint")
    parts: list[pd.DataFrame] = []
    for candidate in candidate_bank.sort_values("branch_order", kind="mergesort").itertuples(
        index=False
    ):
        path = safe_path + float(candidate.shift_ft)
        if not np.isfinite(path).all():
            raise ValueError("candidate branch path is non-finite")
        parts.append(
            pd.DataFrame(
                {
                    "well_id": str(well),
                    "fold": int(fold),
                    "cut_row": int(cut),
                    "future_offset": np.arange(1, path_rows + 1, dtype=np.int16),
                    "row_idx": row_idx,
                    "branch_tvt": path,
                    "branch_id": str(candidate.branch_id),
                    "branch_type": str(candidate.branch_type),
                    "branch_order": int(candidate.branch_order),
                    "control": str(candidate.control),
                    "source": str(candidate.source),
                    "shift_slot": int(candidate.shift_slot),
                    "shift_ft": float(candidate.shift_ft),
                    "visible_rank": int(candidate.visible_rank),
                    "is_top1_real_mode": bool(candidate.is_top1_real_mode),
                    "truth_attached": False,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True).sort_values(
        ["branch_order", "future_offset"], kind="mergesort"
    )
    if int(output["branch_id"].nunique()) != len(candidate_bank):
        raise ValueError("branch path count differs from the frozen candidate bank")
    if not (output.loc[output["branch_id"] == "safe_base", "shift_ft"] == 0.0).all():
        raise ValueError("safe branch was shifted")
    return output.reset_index(drop=True)


def score_branch_evidence(
    branch_paths: pd.DataFrame,
    masked_frame: pd.DataFrame,
    emission: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    validate_target_safe_frame(masked_frame)
    observed_series = pd.to_numeric(masked_frame["GR"], errors="coerce")
    typewell_min = float(np.min(emission["typewell_tvt"]))
    typewell_max = float(np.max(emission["typewell_tvt"]))
    extension = float(get_nested(config, "evidence.geometry_veto.typewell_extension_ft"))
    maximum_shift = float(get_nested(config, "evidence.geometry_veto.maximum_anchor_shift_ft"))
    rows: list[dict[str, Any]] = []
    for branch_id, path in branch_paths.groupby("branch_id", sort=False):
        ordered = path.sort_values("future_offset", kind="mergesort")
        row_idx = ordered["row_idx"].to_numpy(np.int64)
        candidate = ordered["branch_tvt"].to_numpy(np.float64)
        metadata = ordered.iloc[0]
        anchor_ok = abs(float(metadata["shift_ft"])) <= maximum_shift
        for horizon in CHECKPOINTS:
            candidate_horizon = candidate[:horizon]
            observed = (
                observed_series.iloc[row_idx[:horizon]]
                .reset_index(drop=True)
                .interpolate(limit_direction="both")
                .fillna(float(emission["gr_fill"]))
                .to_numpy(np.float64)
            )
            likelihood = row_log_likelihood(observed, candidate_horizon, emission, config)
            finite_path_and_steps = bool(
                np.isfinite(candidate_horizon).all()
                and np.isfinite(np.diff(candidate_horizon)).all()
            )
            support_ok = bool(
                (
                    (candidate_horizon >= typewell_min - extension)
                    & (candidate_horizon <= typewell_max + extension)
                ).all()
            )
            veto = not (finite_path_and_steps and anchor_ok and support_ok)
            if str(branch_id) == "safe_base":
                veto = False
            rows.append(
                {
                    "well_id": str(metadata["well_id"]),
                    "fold": int(metadata["fold"]),
                    "cut_row": int(metadata["cut_row"]),
                    "branch_id": str(branch_id),
                    "branch_type": str(metadata["branch_type"]),
                    "branch_order": int(metadata["branch_order"]),
                    "control": str(metadata["control"]),
                    "source": str(metadata["source"]),
                    "shift_slot": int(metadata["shift_slot"]),
                    "shift_ft": float(metadata["shift_ft"]),
                    "visible_rank": int(metadata["visible_rank"]),
                    "is_top1_real_mode": bool(metadata["is_top1_real_mode"]),
                    "horizon_rows": int(horizon),
                    "likelihood_mean": float(np.mean(likelihood)),
                    "likelihood_sum": float(np.sum(likelihood)),
                    "finite_path_and_steps": finite_path_and_steps,
                    "anchor_within_veto": anchor_ok,
                    "typewell_support_with_extension": support_ok,
                    "geometry_veto": bool(veto),
                    "truth_attached": False,
                }
            )
    output = pd.DataFrame(rows).sort_values(["horizon_rows", "branch_order"], kind="mergesort")
    if not np.isfinite(output[["likelihood_mean", "likelihood_sum"]].to_numpy()).all():
        raise ValueError("future evidence contains non-finite scores")
    return output.reset_index(drop=True)


def policy_branch_ids(policy: str, evidence: pd.DataFrame) -> list[str]:
    safe = ["safe_base"]
    real = (
        evidence.loc[evidence["control"] == "real"]
        .sort_values("branch_order", kind="mergesort")["branch_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    top1 = (
        evidence.loc[
            (evidence["control"] == "real") & evidence["is_top1_real_mode"].astype(bool)
        ]
        .sort_values("branch_order", kind="mergesort")["branch_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    shuffled = (
        evidence.loc[evidence["control"] == "shuffled"]
        .sort_values("branch_order", kind="mergesort")["branch_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    mapping = {
        "safe_base_only": safe,
        "safe_plus_top1_typewell_mode": [*safe, *top1],
        "safe_plus_all_typewell_modes": [*safe, *real],
        "safe_plus_matched_count_shuffled_modes": [*safe, *shuffled],
    }
    if policy not in mapping:
        raise ValueError(f"unknown fixed policy {policy}")
    return mapping[policy]


def select_persistent_checkpoint_policies(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, well_evidence in evidence.groupby("well_id", sort=True):
        metadata = well_evidence.iloc[0]
        available_horizons = sorted(int(value) for value in well_evidence["horizon_rows"].unique())
        if available_horizons != list(CHECKPOINTS):
            raise ValueError(f"{well} evidence does not cover every fixed checkpoint")
        for policy in POLICIES:
            ids = policy_branch_ids(policy, well_evidence)
            for horizon in CHECKPOINTS:
                checkpoints = [value for value in CHECKPOINTS if value <= horizon]
                current = well_evidence.loc[
                    (well_evidence["horizon_rows"] == horizon)
                    & (well_evidence["branch_id"].isin(ids))
                ].copy()
                safe_current = current.loc[current["branch_id"] == "safe_base"]
                if len(safe_current) != 1:
                    raise ValueError("safe evidence must exist exactly once per checkpoint")
                passing: list[str] = []
                for branch_id in [value for value in ids if value != "safe_base"]:
                    persistent = True
                    for checkpoint in checkpoints:
                        branch_row = well_evidence.loc[
                            (well_evidence["horizon_rows"] == checkpoint)
                            & (well_evidence["branch_id"] == branch_id)
                        ]
                        safe_row = well_evidence.loc[
                            (well_evidence["horizon_rows"] == checkpoint)
                            & (well_evidence["branch_id"] == "safe_base")
                        ]
                        if len(branch_row) != 1 or len(safe_row) != 1:
                            persistent = False
                            break
                        persistent &= not bool(branch_row.iloc[0]["geometry_veto"])
                        persistent &= float(branch_row.iloc[0]["likelihood_mean"]) > float(
                            safe_row.iloc[0]["likelihood_mean"]
                        )
                    if persistent:
                        passing.append(branch_id)
                if passing:
                    chosen_pool = current.loc[current["branch_id"].isin(passing)].sort_values(
                        ["likelihood_mean", "branch_order"],
                        ascending=[False, True],
                        kind="mergesort",
                    )
                    chosen = chosen_pool.iloc[0]
                else:
                    chosen = safe_current.iloc[0]
                rows.append(
                    {
                        "well_id": str(well),
                        "fold": int(metadata["fold"]),
                        "cut_row": int(metadata["cut_row"]),
                        "horizon_rows": int(horizon),
                        "policy": policy,
                        "selected_branch_id": str(chosen["branch_id"]),
                        "selected_branch_type": str(chosen["branch_type"]),
                        "selected_control": str(chosen["control"]),
                        "selected_likelihood_mean": float(chosen["likelihood_mean"]),
                        "candidate_count_including_safe": len(ids),
                        "passing_alternative_count": len(passing),
                        "required_checkpoint_count": len(checkpoints),
                        "selected_is_persistent_commit": str(chosen["branch_id"]) != "safe_base",
                        "truth_attached": False,
                    }
                )
    output = pd.DataFrame(rows).sort_values(
        ["well_id", "horizon_rows", "policy"], kind="mergesort"
    )
    if output.duplicated(["well_id", "policy", "horizon_rows"]).any():
        raise ValueError("policy selection identity is not unique")
    return output.reset_index(drop=True)


def assert_target_free_tables(
    mask_manifest: pd.DataFrame,
    shift_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    branch_paths: pd.DataFrame,
    evidence: pd.DataFrame,
    policy: pd.DataFrame,
) -> dict[str, str]:
    frames = {
        "shift_scores": shift_scores,
        "candidates": candidates,
        "branch_paths": branch_paths,
        "evidence": evidence,
        "policy": policy,
    }
    for name, frame in frames.items():
        leaked = sorted(TARGET_FORBIDDEN_COLUMNS.intersection(frame.columns))
        if leaked:
            raise ValueError(f"target-free {name} table contains truth columns {leaked}")
        if "truth_attached" in frame and bool(frame["truth_attached"].astype(bool).any()):
            raise ValueError(f"target-free {name} table already has truth attached")
    if int(mask_manifest["post_cut_tvt_input_finite_rows_after_mask"].sum()) != 0:
        raise ValueError("mask manifest reports post-cut TVT_input exposure")
    if branch_paths.duplicated(["well_id", "branch_id", "row_idx"]).any():
        raise ValueError("branch path identity is not unique")
    if evidence.duplicated(["well_id", "branch_id", "horizon_rows"]).any():
        raise ValueError("evidence identity is not unique")
    if policy.duplicated(["well_id", "policy", "horizon_rows"]).any():
        raise ValueError("policy identity is not unique")
    candidate_summary = candidates.groupby("well_id").agg(
        safe_count=("control", lambda values: int((values == "core").sum())),
        real_count=("control", lambda values: int((values == "real").sum())),
        shuffled_count=("control", lambda values: int((values == "shuffled").sum())),
        candidate_count=("branch_id", "nunique"),
    )
    local_count = (
        shift_scores.loc[shift_scores["eligible_local_maximum"]]
        .groupby("well_id")
        .size()
        .reindex(candidate_summary.index, fill_value=0)
    )
    branch_count = branch_paths.groupby("well_id")["branch_id"].nunique()
    if not candidate_summary["safe_count"].eq(1).all():
        raise ValueError("safe base is not retained exactly once for every event")
    if not candidate_summary["real_count"].eq(local_count).all():
        raise ValueError("not every fixed-bank local mode was retained")
    if not candidate_summary["shuffled_count"].eq(candidate_summary["real_count"]).all():
        raise ValueError("matched-count control count differs from the real mode count")
    if not branch_count.eq(candidate_summary["candidate_count"]).all():
        raise ValueError("candidate and branch counts differ")
    forbidden_tokens = ("selfgr", "self_gr", "ncc", "donor")
    for name in ("candidates", "branch_paths", "evidence"):
        if frames[name]["source"].astype(str).str.lower().apply(
            lambda value: any(token in value for token in forbidden_tokens)
        ).any():
            raise ValueError(f"same-well self-GR source entered target-free {name}")
    return {
        "mask_manifest": dataframe_content_sha(mask_manifest),
        "shift_scores": dataframe_content_sha(shift_scores),
        "candidates": dataframe_content_sha(candidates),
        "branch_paths": dataframe_content_sha(branch_paths),
        "evidence": dataframe_content_sha(evidence),
        "policy": dataframe_content_sha(policy),
    }


# %% [markdown]
# ## 7. Post-freeze metrics and scientific guards


# %%
def require_frozen_hashes(frozen_hashes: Mapping[str, str]) -> None:
    expected = {
        "mask_manifest",
        "shift_scores",
        "candidates",
        "branch_paths",
        "evidence",
        "policy",
    }
    missing = sorted(key for key in expected if not frozen_hashes.get(key))
    if missing:
        raise ValueError(f"truth attachment requires frozen content SHA for {missing}")


def load_truth_for_branch_rows(
    raw_dir: Path,
    branch_paths: pd.DataFrame,
    *,
    frozen_hashes: Mapping[str, str],
) -> pd.DataFrame:
    require_frozen_hashes(frozen_hashes)
    rows: list[pd.DataFrame] = []
    keys = branch_paths[["well_id", "row_idx"]].drop_duplicates()
    for well, part in keys.groupby("well_id", sort=True):
        path = raw_dir / f"{well}__horizontal_well.csv"
        truth = pd.read_csv(path, usecols=["TVT"])
        positions = part["row_idx"].to_numpy(np.int64)
        values = pd.to_numeric(truth.iloc[positions]["TVT"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"held-out truth is non-finite for well {well}")
        rows.append(pd.DataFrame({"well_id": str(well), "row_idx": positions, "tvt_true": values}))
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("truth identity is not unique")
    return output


def load_hidden_like(
    config: Mapping[str, Any], *, frozen_hashes: Mapping[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen_hashes(frozen_hashes)
    spec = get_nested(config, "data.hidden_like")
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"name": "hidden_like", "enabled": False}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns):
        raise ValueError("hidden-like assignments are missing fixed role columns")
    return frame[list(required)].copy(), {
        "name": "exp115_hidden_like_assignments",
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


def rmse_from_squared_error(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array))) if len(array) else float("nan")


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positive = int(y.sum())
    negative = int((~y).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(np.float64)
    return float((ranks[y].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def balanced_binary_accuracy(labels: Sequence[bool], predictions: Sequence[bool]) -> float:
    truth = np.asarray(labels, dtype=bool)
    chosen = np.asarray(predictions, dtype=bool)
    positive = truth
    negative = ~truth
    if not positive.any() or not negative.any():
        return float("nan")
    sensitivity = float((chosen[positive] == truth[positive]).mean())
    specificity = float((chosen[negative] == truth[negative]).mean())
    return 0.5 * (sensitivity + specificity)


def build_post_freeze_readout(
    branch_paths: pd.DataFrame,
    evidence: pd.DataFrame,
    policy: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_like: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scored_paths = branch_paths.merge(
        truth, on=["well_id", "row_idx"], how="left", validate="many_to_one"
    )
    if scored_paths["tvt_true"].isna().any():
        raise ValueError("truth attachment did not cover all frozen branch rows")
    scored_paths["squared_error"] = (scored_paths["branch_tvt"] - scored_paths["tvt_true"]) ** 2

    branch_metric_rows: list[dict[str, Any]] = []
    for (well, branch_id), part in scored_paths.groupby(["well_id", "branch_id"], sort=True):
        metadata = part.iloc[0]
        for horizon in CHECKPOINTS:
            chosen = part.loc[part["future_offset"] <= horizon]
            branch_metric_rows.append(
                {
                    "well_id": str(well),
                    "fold": int(metadata["fold"]),
                    "branch_id": str(branch_id),
                    "branch_type": str(metadata["branch_type"]),
                    "branch_order": int(metadata["branch_order"]),
                    "control": str(metadata["control"]),
                    "source": str(metadata["source"]),
                    "shift_ft": float(metadata["shift_ft"]),
                    "horizon_rows": int(horizon),
                    "branch_rmse": rmse_from_squared_error(chosen["squared_error"]),
                }
            )
    branch_metrics = pd.DataFrame(branch_metric_rows)

    selected = policy.merge(
        branch_metrics[["well_id", "branch_id", "horizon_rows", "branch_rmse"]],
        left_on=["well_id", "selected_branch_id", "horizon_rows"],
        right_on=["well_id", "branch_id", "horizon_rows"],
        how="left",
        validate="many_to_one",
    ).drop(columns="branch_id")
    if selected["branch_rmse"].isna().any():
        raise ValueError("policy readout is missing selected branch RMSE")

    pooled_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.Series]] = [
        ("overall", pd.Series(True, index=policy.index, dtype=bool))
    ]
    scopes.extend(
        (f"fold_{fold}", policy["fold"].eq(fold)) for fold in sorted(policy["fold"].unique())
    )
    for scope, scope_mask in scopes:
        scoped_policy = policy.loc[scope_mask]
        for (policy_name, horizon), selections in scoped_policy.groupby(
            ["policy", "horizon_rows"], sort=True
        ):
            chosen_parts: list[pd.DataFrame] = []
            for row in selections.itertuples(index=False):
                chosen_parts.append(
                    scored_paths.loc[
                        (scored_paths["well_id"] == row.well_id)
                        & (scored_paths["branch_id"] == row.selected_branch_id)
                        & (scored_paths["future_offset"] <= int(horizon))
                    ]
                )
            chosen = pd.concat(chosen_parts, ignore_index=True)
            pooled_rows.append(
                {
                    "scope": scope,
                    "policy": str(policy_name),
                    "horizon_rows": int(horizon),
                    "wells": int(selections["well_id"].nunique()),
                    "rows": len(chosen),
                    "rmse": rmse_from_squared_error(chosen["squared_error"]),
                }
            )
    pooled = pd.DataFrame(pooled_rows)

    h128 = evidence.loc[
        evidence["horizon_rows"] == 128,
        ["well_id", "branch_id", "likelihood_mean", "geometry_veto"],
    ].rename(
        columns={
            "likelihood_mean": "likelihood_h128",
            "geometry_veto": "veto_h128",
        }
    )
    h256 = evidence.loc[
        evidence["horizon_rows"] == 256,
        ["well_id", "fold", "branch_id", "control", "likelihood_mean", "geometry_veto"],
    ].rename(
        columns={
            "likelihood_mean": "likelihood_h256",
            "geometry_veto": "veto_h256",
        }
    )
    safe_evidence = (
        h128.loc[h128["branch_id"] == "safe_base", ["well_id", "likelihood_h128"]]
        .merge(
            h256.loc[
                h256["branch_id"] == "safe_base",
                ["well_id", "likelihood_h256"],
            ],
            on="well_id",
            validate="one_to_one",
        )
        .rename(
            columns={
                "likelihood_h128": "safe_likelihood_h128",
                "likelihood_h256": "safe_likelihood_h256",
            }
        )
    )
    primary_branches = branch_metrics.loc[branch_metrics["horizon_rows"] == 256]
    safe_rmse = primary_branches.loc[
        primary_branches["branch_id"] == "safe_base",
        ["well_id", "branch_rmse"],
    ].rename(columns={"branch_rmse": "safe_rmse"})
    alternatives = h256.loc[h256["control"] == "real"].merge(
        h128,
        on=["well_id", "branch_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_h128_meta"),
    )
    alternative_rmse = primary_branches.loc[
        primary_branches["control"] == "real",
        ["well_id", "branch_id", "branch_rmse"],
    ].rename(columns={"branch_rmse": "alternative_rmse"})
    pairs = (
        alternatives.merge(alternative_rmse, on=["well_id", "branch_id"], validate="one_to_one")
        .merge(safe_evidence, on="well_id", validate="many_to_one")
        .merge(safe_rmse, on="well_id", validate="many_to_one")
    )
    pairs["alternative_is_better"] = pairs["alternative_rmse"] < pairs["safe_rmse"]
    pairs["evidence_margin_h256"] = (
        pairs["likelihood_h256"] - pairs["safe_likelihood_h256"]
    )
    pairs["persistent_alternative_choice"] = (
        ~pairs["veto_h128"].astype(bool)
        & ~pairs["veto_h256"].astype(bool)
        & (pairs["likelihood_h128"] > pairs["safe_likelihood_h128"])
        & (pairs["likelihood_h256"] > pairs["safe_likelihood_h256"])
    )
    pairs["choice_correct"] = (
        pairs["persistent_alternative_choice"] == pairs["alternative_is_better"]
    )
    pair_rows: list[dict[str, Any]] = []
    pair_scopes: list[tuple[str, pd.DataFrame]] = [("overall", pairs)]
    pair_scopes.extend((f"fold_{fold}", part) for fold, part in pairs.groupby("fold", sort=True))
    for scope, part in pair_scopes:
        pair_rows.append(
            {
                "scope": scope,
                "pairs": len(part),
                "events": int(part["well_id"].nunique()),
                "alternative_better_rate": float(part["alternative_is_better"].mean())
                if len(part)
                else float("nan"),
                "evidence_auc": binary_auc(
                    part["alternative_is_better"], part["evidence_margin_h256"]
                ),
                "balanced_choice_accuracy": balanced_binary_accuracy(
                    part["alternative_is_better"], part["persistent_alternative_choice"]
                ),
                "plain_choice_accuracy": float(part["choice_correct"].mean())
                if len(part)
                else float("nan"),
            }
        )
    pair_metrics = pd.DataFrame(pair_rows)

    safe = primary_branches.loc[
        primary_branches["branch_id"] == "safe_base", ["well_id", "fold", "branch_rmse"]
    ].rename(columns={"branch_rmse": "safe_rmse"})
    real_min = (
        primary_branches.loc[primary_branches["control"] == "real"]
        .groupby("well_id")["branch_rmse"]
        .min()
        .rename("best_real_alternative_rmse")
    )
    safety = safe.merge(real_min, on="well_id", how="left", validate="one_to_one")
    all_selected = selected.loc[
        (selected["policy"] == "safe_plus_all_typewell_modes")
        & (selected["horizon_rows"] == 256),
        ["well_id", "selected_branch_id", "branch_rmse"],
    ]
    safety = safety.merge(all_selected, on="well_id", validate="one_to_one")
    safety["safe_unique_best"] = safety["best_real_alternative_rmse"].isna() | (
        safety["safe_rmse"] < safety["best_real_alternative_rmse"]
    )
    safety["false_switch"] = safety["safe_unique_best"] & (
        safety["selected_branch_id"] != "safe_base"
    )
    safety["selected_delta_rmse_vs_safe"] = safety["branch_rmse"] - safety["safe_rmse"]

    if not hidden_like.empty:
        selected = selected.merge(hidden_like, on="well_id", how="left", validate="many_to_one")
        safety = safety.merge(hidden_like, on="well_id", how="left", validate="one_to_one")
        pairs = pairs.merge(hidden_like, on="well_id", how="left", validate="many_to_one")
    return {
        "scored_paths": scored_paths,
        "branch_metrics": branch_metrics,
        "selected": selected,
        "pooled_metrics": pooled,
        "pairwise": pairs,
        "pairwise_metrics": pair_metrics,
        "safety": safety,
    }


def metric_value(pooled: pd.DataFrame, scope: str, policy: str, horizon: int) -> float:
    row = pooled.loc[
        (pooled["scope"] == scope)
        & (pooled["policy"] == policy)
        & (pooled["horizon_rows"] == horizon)
    ]
    if len(row) != 1:
        raise ValueError(f"missing pooled metric {scope}/{policy}/H{horizon}")
    return float(row.iloc[0]["rmse"])


def evaluate_scientific_guard(
    mask_manifest: pd.DataFrame,
    shift_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    branch_paths: pd.DataFrame,
    evidence: pd.DataFrame,
    readout: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.guards")
    pooled = readout["pooled_metrics"]
    pair_metrics = readout["pairwise_metrics"]
    safety = readout["safety"]
    folds = sorted(int(value) for value in mask_manifest["fold"].unique())
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    branch_finite = float(np.isfinite(branch_paths["branch_tvt"]).mean())
    evidence_finite = float(
        np.isfinite(evidence[["likelihood_mean", "likelihood_sum"]].to_numpy()).mean()
    )
    eligible = int(mask_manifest["well_id"].nunique())

    candidate_summary = candidates.groupby("well_id").agg(
        safe_count=("control", lambda values: int((values == "core").sum())),
        real_count=("control", lambda values: int((values == "real").sum())),
        shuffled_count=("control", lambda values: int((values == "shuffled").sum())),
        candidate_count=("branch_id", "nunique"),
    )
    local_count = (
        shift_scores.loc[shift_scores["eligible_local_maximum"]]
        .groupby("well_id")
        .size()
        .reindex(candidate_summary.index, fill_value=0)
    )
    branch_count = branch_paths.groupby("well_id")["branch_id"].nunique()
    safe_coverage = float(candidate_summary["safe_count"].eq(1).mean())
    all_mode_coverage = float(candidate_summary["real_count"].eq(local_count).mean())
    branch_candidate_count_coverage = float(
        branch_count.eq(1 + candidate_summary["real_count"] + candidate_summary["shuffled_count"])
        .mean()
    )
    real_candidate_count_exact_coverage = float(
        mask_manifest.set_index("well_id")["real_candidate_count_including_safe"]
        .eq(1 + local_count)
        .mean()
    )
    source_text = candidates["source"].astype(str).str.lower()
    self_gr_count = int(
        source_text.apply(
            lambda value: any(token in value for token in ("selfgr", "self_gr", "ncc", "donor"))
        ).sum()
    )
    technical = {
        "minimum_eligible_wells": eligible >= int(guards["minimum_eligible_wells"]),
        "all_expected_folds": folds == expected_folds,
        "mask_identity_coverage": float(
            (mask_manifest["post_cut_tvt_input_finite_rows_after_mask"] == 0).mean()
        )
        == float(guards["required_mask_identity_coverage"]),
        "safe_candidate_coverage": safe_coverage
        == float(guards["required_safe_candidate_coverage"]),
        "all_local_modes_retained_coverage": all_mode_coverage
        == float(guards["required_all_local_modes_retained_coverage"]),
        "real_candidate_count_exact_coverage": real_candidate_count_exact_coverage == 1.0,
        "branch_candidate_count_coverage": branch_candidate_count_coverage == 1.0,
        "self_gr_candidate_count": self_gr_count
        == int(guards["required_self_gr_candidate_count"]),
        "branch_finite_coverage": branch_finite == float(guards["required_branch_finite_coverage"]),
        "evidence_finite_coverage": evidence_finite
        == float(guards["required_evidence_finite_coverage"]),
        "heldout_post_cut_truth_access_before_freeze": 0
        == int(guards["required_post_cut_truth_access_before_freeze"]),
    }

    pair_overall_rows = pair_metrics.loc[pair_metrics["scope"] == "overall"]
    pair_folds = pair_metrics.loc[pair_metrics["scope"].str.startswith("fold_")]
    if len(pair_overall_rows) != 1:
        raise ValueError("pairwise readout is missing the overall row")
    pair_overall = pair_overall_rows.iloc[0]
    pair_guard = {
        "auc_each_fold": bool(
            len(pair_folds) == len(expected_folds)
            and pair_folds["evidence_auc"].notna().all()
            and (
                pair_folds["evidence_auc"]
                >= float(guards["minimum_safe_alternative_auc_each_fold"])
            ).all()
        ),
        "balanced_choice_accuracy_pooled": np.isfinite(
            float(pair_overall["balanced_choice_accuracy"])
        )
        and float(pair_overall["balanced_choice_accuracy"])
        >= float(guards["minimum_balanced_choice_accuracy_pooled"]),
        "balanced_choice_accuracy_each_fold": bool(
            len(pair_folds) == len(expected_folds)
            and pair_folds["balanced_choice_accuracy"].notna().all()
            and (
                pair_folds["balanced_choice_accuracy"]
                > float(guards["minimum_balanced_choice_accuracy_each_fold_exclusive"])
            ).all()
        ),
    }

    safe_policy = "safe_base_only"
    top1_policy = "safe_plus_top1_typewell_mode"
    all_policy = "safe_plus_all_typewell_modes"
    shuffled_policy = "safe_plus_matched_count_shuffled_modes"
    safe_h256 = metric_value(pooled, "overall", safe_policy, 256)
    top1_h256 = metric_value(pooled, "overall", top1_policy, 256)
    all_h256 = metric_value(pooled, "overall", all_policy, 256)
    shuffled_h256 = metric_value(pooled, "overall", shuffled_policy, 256)
    gain_safe_h256 = safe_h256 - all_h256
    gain_top1_h256 = top1_h256 - all_h256
    gain_safe_h512 = metric_value(pooled, "overall", safe_policy, 512) - metric_value(
        pooled, "overall", all_policy, 512
    )
    fold_safe_improved = 0
    fold_top1_improved = 0
    fold_real_nonregressing = 0
    for fold in expected_folds:
        scope = f"fold_{fold}"
        fold_all = metric_value(pooled, scope, all_policy, 256)
        if fold_all < metric_value(pooled, scope, safe_policy, 256):
            fold_safe_improved += 1
        if fold_all < metric_value(pooled, scope, top1_policy, 256):
            fold_top1_improved += 1
        if fold_all <= metric_value(pooled, scope, shuffled_policy, 256):
            fold_real_nonregressing += 1
    safe_unique = safety.loc[safety["safe_unique_best"]]
    false_switch_rate = (
        float(safe_unique["false_switch"].mean()) if len(safe_unique) else float("nan")
    )
    performance_guard = {
        "gain_vs_safe_h256": gain_safe_h256
        >= float(guards["minimum_all_modes_gain_vs_safe_h256_ft"]),
        "gain_vs_safe_required_folds": fold_safe_improved
        >= int(guards["minimum_all_modes_gain_vs_safe_folds"]),
        "gain_vs_top1_h256": gain_top1_h256
        >= float(guards["minimum_all_modes_gain_vs_top1_h256_ft"]),
        "gain_vs_top1_required_folds": fold_top1_improved
        >= int(guards["minimum_all_modes_gain_vs_top1_folds"]),
        "h512_gain_not_below_h256": gain_safe_h512 >= gain_safe_h256,
    }
    safety_guard = {
        "safe_unique_best_false_switch": np.isfinite(false_switch_rate)
        and false_switch_rate <= float(guards["maximum_safe_unique_best_false_switch_rate"]),
        "real_better_than_matched_shuffle_pooled": all_h256 < shuffled_h256,
        "real_nonregressing_all_folds": fold_real_nonregressing
        >= int(guards["minimum_real_vs_matched_shuffle_nonregressing_folds"]),
    }
    checks = [
        *technical.values(),
        *pair_guard.values(),
        *performance_guard.values(),
        *safety_guard.values(),
    ]
    return {
        "passed": bool(all(checks)),
        "technical": technical,
        "pairwise": pair_guard,
        "performance": performance_guard,
        "safety": safety_guard,
        "readout": {
            "eligible_wells": eligible,
            "folds": folds,
            "branch_finite_coverage": branch_finite,
            "evidence_finite_coverage": evidence_finite,
            "safe_candidate_coverage": safe_coverage,
            "all_local_modes_retained_coverage": all_mode_coverage,
            "real_candidate_count_exact_coverage": real_candidate_count_exact_coverage,
            "branch_candidate_count_coverage": branch_candidate_count_coverage,
            "self_gr_candidate_count": self_gr_count,
            "pairwise_evidence_auc_pooled": float(pair_overall["evidence_auc"]),
            "balanced_choice_accuracy_pooled": float(
                pair_overall["balanced_choice_accuracy"]
            ),
            "safe_h256_rmse": safe_h256,
            "top1_h256_rmse": top1_h256,
            "all_modes_h256_rmse": all_h256,
            "matched_shuffle_h256_rmse": shuffled_h256,
            "gain_vs_safe_h256_ft": gain_safe_h256,
            "gain_vs_top1_h256_ft": gain_top1_h256,
            "gain_vs_safe_h512_ft": gain_safe_h512,
            "false_switch_rate": false_switch_rate,
            "fold_safe_improved": fold_safe_improved,
            "fold_top1_improved": fold_top1_improved,
            "fold_real_nonregressing": fold_real_nonregressing,
        },
    }


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp291 backtest must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is "
            "reserved for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp291 Kaggle CPU execution is not approved")
    validate_scientific_contract(config)
    started = time.time()
    raw_dir = train_data_dir(config)
    raw_wells = list_horizontal_wells(raw_dir)
    fold_by_well, kappa_by_fold, input_manifests = load_exp226_fold_contract(config)
    if len(raw_wells) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("raw horizontal well count does not match the fixed contract")
    if set(raw_wells) != set(fold_by_well):
        raise ValueError("raw horizontal and exp226 OOF well sets differ")

    params = params_from_config(config)
    mask_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    shift_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    branch_parts: list[pd.DataFrame] = []
    evidence_parts: list[pd.DataFrame] = []
    policy_parts: list[pd.DataFrame] = []
    raw_manifest_rows: list[dict[str, Any]] = []
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    progress_every = int(get_nested(config, "execution.progress_every_wells", 10))

    for fold in expected_folds:
        source_ids = sorted(well for well in raw_wells if fold_by_well[well] != fold)
        target_ids = sorted(well for well in raw_wells if fold_by_well[well] == fold)
        source_wells = [
            load_source_geometry_well(raw_dir / f"{well}__horizontal_well.csv", params, wi=index)
            for index, well in enumerate(source_ids)
        ]
        fields = build_fields(source_wells, params)
        kappa = kappa_by_fold[fold]
        print(f"exp291 fold={fold} source_wells={len(source_wells)} target_wells={len(target_ids)}")
        for index, well in enumerate(target_ids, start=1):
            horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
            typewell_path = raw_dir / f"{well}__typewell.csv"
            try:
                safe_frame = load_target_safe_horizontal(horizontal_path)
                masked, manifest = build_pseudo_mask(well, safe_frame, config)
                cut = int(manifest["cut_row"])
                typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
                emission = prepare_typewell_emission(masked, typewell, cut=cut, config=config)
                target = build_target_geometry_well(
                    well,
                    masked,
                    cut=cut,
                    original_last_known=int(manifest["original_last_known_row"]),
                    params=params,
                )
                safe_geometry = replay_exp226_geometry(target, fields, kappa, params)
                shift_scores = score_visible_shift_bank(
                    well,
                    masked,
                    emission,
                    cut=cut,
                    fold=fold,
                    config=config,
                )
                candidates = build_candidate_bank(
                    well,
                    shift_scores,
                    fold=fold,
                    cut=cut,
                    config=config,
                )
                branches = build_branch_paths(
                    well,
                    fold=fold,
                    cut=cut,
                    safe_geometry=safe_geometry,
                    candidate_bank=candidates,
                    config=config,
                )
                evidence = score_branch_evidence(branches, masked, emission, config)
                policies = select_persistent_checkpoint_policies(evidence)
                real_count = int((candidates["control"] == "real").sum())
                shuffled_count = int((candidates["control"] == "shuffled").sum())
                manifest.update(
                    {
                        "fold": fold,
                        "geometry_rows": len(safe_geometry),
                        "geometry_finite_coverage": float(np.isfinite(safe_geometry).mean()),
                        "eligible_local_mode_count": int(
                            shift_scores["eligible_local_maximum"].sum()
                        ),
                        "retained_real_mode_count": real_count,
                        "retained_shuffled_mode_count": shuffled_count,
                        "safe_candidate_count": int(
                            (candidates["branch_id"] == "safe_base").sum()
                        ),
                        "real_candidate_count_including_safe": 1 + real_count,
                        "candidate_count": int(candidates["branch_id"].nunique()),
                        "branch_count": int(branches["branch_id"].nunique()),
                        "evidence_rows": len(evidence),
                        "policy_rows": len(policies),
                        "gr_sigma": float(emission["gr_sigma"]),
                        "self_gr_candidate_count": 0,
                        "target_post_cut_truth_access_before_freeze": 0,
                    }
                )
                mask_rows.append(manifest)
                shift_parts.append(shift_scores)
                candidate_parts.append(candidates)
                branch_parts.append(branches)
                evidence_parts.append(evidence)
                policy_parts.append(policies)
                raw_manifest_rows.append(
                    {
                        "well_id": well,
                        "fold": fold,
                        "horizontal_path": str(horizontal_path),
                        "horizontal_raw_sha256": sha256_path(horizontal_path),
                        "typewell_path": str(typewell_path),
                        "typewell_raw_sha256": sha256_path(typewell_path),
                    }
                )
            except IneligibleWellError as exc:
                exclusion_rows.append({"well_id": well, "fold": fold, "reason": str(exc)})
            if index % progress_every == 0 or index == len(target_ids):
                print(
                    f"exp291 fold={fold} processed={index}/{len(target_ids)} "
                    f"eligible_total={len(mask_rows)}"
                )
        del fields, source_wells

    if not mask_rows:
        raise ValueError("exp291 generated zero eligible controlled backtests")
    mask_manifest = pd.DataFrame(mask_rows).sort_values("well_id", kind="mergesort")
    exclusions = pd.DataFrame(exclusion_rows, columns=["well_id", "fold", "reason"])
    shift_scores = pd.concat(shift_parts, ignore_index=True).sort_values(
        ["well_id", "shift_slot"], kind="mergesort"
    )
    candidates = pd.concat(candidate_parts, ignore_index=True).sort_values(
        ["well_id", "branch_order"], kind="mergesort"
    )
    branch_paths = pd.concat(branch_parts, ignore_index=True).sort_values(
        ["well_id", "branch_order", "future_offset"], kind="mergesort"
    )
    evidence = pd.concat(evidence_parts, ignore_index=True).sort_values(
        ["well_id", "horizon_rows", "branch_order"], kind="mergesort"
    )
    policy = pd.concat(policy_parts, ignore_index=True).sort_values(
        ["well_id", "horizon_rows", "policy"], kind="mergesort"
    )
    frozen_hashes = assert_target_free_tables(
        mask_manifest,
        shift_scores,
        candidates,
        branch_paths,
        evidence,
        policy,
    )

    artifacts = artifact_dir()
    mask_path = artifacts / f"{OUTPUT_PREFIX}_mask_manifest.csv"
    mask_manifest.to_csv(mask_path, index=False)
    exclusion_path = artifacts / f"{OUTPUT_PREFIX}_ineligible_wells.csv"
    exclusions.to_csv(exclusion_path, index=False)
    target_free_artifacts = {
        "shift_scores": write_csv_gzip(
            shift_scores,
            artifacts / f"{OUTPUT_PREFIX}_target_free_shift_scores.csv.gz",
        ),
        "candidate_bank": write_csv_gzip(
            candidates,
            artifacts / f"{OUTPUT_PREFIX}_target_free_candidate_bank.csv.gz",
        ),
        "branch_paths": write_csv_gzip(
            branch_paths,
            artifacts / f"{OUTPUT_PREFIX}_target_free_branch_paths.csv.gz",
        ),
        "evidence": write_csv_gzip(
            evidence,
            artifacts / f"{OUTPUT_PREFIX}_target_free_evidence.csv.gz",
        ),
        "policy": write_csv_gzip(
            policy,
            artifacts / f"{OUTPUT_PREFIX}_target_free_policy_selection.csv.gz",
        ),
    }
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "truth_attached": False,
        "heldout_post_cut_truth_access_before_freeze_count": 0,
        "source_truth_policy": "other_fold_wells_only_for_exp226_geometry_field",
        "mask": get_nested(config, "pseudo_mask"),
        "candidate_bank": get_nested(config, "candidate_bank"),
        "evidence": get_nested(config, "evidence"),
        "policies": get_nested(config, "policies"),
        "frozen_content_sha256": frozen_hashes,
        "schemas": {
            "mask_manifest": {column: str(dtype) for column, dtype in mask_manifest.dtypes.items()},
            "shift_scores": {column: str(dtype) for column, dtype in shift_scores.dtypes.items()},
            "candidate_bank": {column: str(dtype) for column, dtype in candidates.dtypes.items()},
            "branch_paths": {column: str(dtype) for column, dtype in branch_paths.dtypes.items()},
            "evidence": {column: str(dtype) for column, dtype in evidence.dtypes.items()},
            "policy": {column: str(dtype) for column, dtype in policy.dtypes.items()},
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_contract.json"
    write_json(contract_path, contract)

    # Held-out target TVT is first read below, after every target-free table has
    # been persisted and its logical content SHA has been fixed.
    truth = load_truth_for_branch_rows(raw_dir, branch_paths, frozen_hashes=frozen_hashes)
    hidden_like, hidden_manifest = load_hidden_like(config, frozen_hashes=frozen_hashes)
    readout = build_post_freeze_readout(branch_paths, evidence, policy, truth, hidden_like)
    guard = evaluate_scientific_guard(
        mask_manifest,
        shift_scores,
        candidates,
        branch_paths,
        evidence,
        readout,
        config,
    )

    pooled_path = artifacts / f"{OUTPUT_PREFIX}_overall_metrics.csv"
    readout["pooled_metrics"].loc[
        readout["pooled_metrics"]["scope"] == "overall"
    ].to_csv(pooled_path, index=False)
    fold_path = artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv"
    readout["pooled_metrics"].loc[
        readout["pooled_metrics"]["scope"].str.startswith("fold_")
    ].to_csv(fold_path, index=False)
    pair_path = artifacts / f"{OUTPUT_PREFIX}_pairwise_metrics.csv"
    readout["pairwise_metrics"].to_csv(pair_path, index=False)

    selected_h256 = (
        readout["selected"]
        .loc[
            readout["selected"]["horizon_rows"] == 256,
            ["well_id", "policy", "selected_branch_id", "branch_rmse"],
        ]
        .pivot(index="well_id", columns="policy")
    )
    selected_h256.columns = [f"{left}_{right}" for left, right in selected_h256.columns]
    selected_h256 = selected_h256.reset_index()
    by_well = readout["safety"].merge(
        selected_h256, on="well_id", how="left", validate="one_to_one"
    )
    by_well = by_well.merge(
        mask_manifest[
            [
                "well_id",
                "eligible_local_mode_count",
                "retained_real_mode_count",
                "retained_shuffled_mode_count",
                "visible_rows",
            ]
        ],
        on="well_id",
        validate="one_to_one",
    )
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv"
    by_well.to_csv(by_well_path, index=False)

    raw_manifest = pd.DataFrame(raw_manifest_rows).sort_values("well_id", kind="mergesort")
    input_manifests.extend(
        [
            hidden_manifest,
            {
                "name": "eligible_raw_horizontal_and_typewell_files",
                "path": str(raw_dir),
                "raw_sha256": dataframe_content_sha(
                    raw_manifest,
                    ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
                ),
                "rows": len(raw_manifest),
                "wells": int(raw_manifest["well_id"].nunique()),
            },
        ]
    )
    input_manifest = pd.DataFrame(input_manifests)
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)

    status = (
        "train_side_backtest_completed_guard_passed"
        if guard["passed"]
        else "train_side_backtest_completed_guard_failed_branch_closed"
    )
    output_paths = {
        "contract": contract_path,
        "mask_manifest": mask_path,
        "ineligible_wells": exclusion_path,
        "overall_metrics": pooled_path,
        "fold_metrics": fold_path,
        "pairwise_metrics": pair_path,
        "by_well_metrics": by_well_path,
        "input_manifest": input_manifest_path,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "eligible_wells": int(mask_manifest["well_id"].nunique()),
        "ineligible_wells": len(exclusions),
        "folds": sorted(int(value) for value in mask_manifest["fold"].unique()),
        "active_contracts": 1,
        "fixed_policies": 4,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "guard": guard,
        "truth_attachment": {
            "stage": "after_all_target_free_tables_persisted_and_content_hashed",
            "heldout_post_cut_truth_access_before_freeze_count": 0,
            "frozen_content_sha256": frozen_hashes,
        },
        "target_free_artifacts": target_free_artifacts,
        "file_sha256": {name: sha256_path(path) for name, path in output_paths.items()},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": "permit_separate_decoder_design_review"
        if guard["passed"]
        else "close_without_parameter_rescue",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "diagnostic": {
            "eligible_wells": summary["eligible_wells"],
            "guard": guard,
            "frozen_content_sha256": frozen_hashes,
        },
        "active_contracts": 1,
        "fixed_policies": 4,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_variants": 0,
        "pf_variants": 0,
        "inference": False,
        "submission": False,
        "notes": "Fixed prefix-mask backtest only; no decoder, prediction, or submission.",
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and contract preview

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent_contract": get_nested(CONFIG, "lineage.parent"),
                "mask_rows": get_nested(CONFIG, "pseudo_mask.masked_rows"),
                "visible_shift_score_rows": get_nested(
                    CONFIG, "pseudo_mask.visible_shift_score_rows"
                ),
                "shift_bank": get_nested(CONFIG, "candidate_bank.shift_bank_ft"),
                "minimum_alternative_abs_shift_ft": get_nested(
                    CONFIG, "candidate_bank.minimum_alternative_abs_shift_ft"
                ),
                "checkpoints": get_nested(CONFIG, "evidence.checkpoints_rows"),
                "active_contracts": get_nested(CONFIG, "execution.active_contract_count"),
                "fixed_policies": get_nested(CONFIG, "execution.fixed_policy_count"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "hmm_well_runs": get_nested(CONFIG, "execution.hmm_well_runs"),
                "pf_well_runs": get_nested(CONFIG, "execution.pf_well_runs"),
                "parent_control_retraining": get_nested(
                    CONFIG, "execution.control_or_parent_retraining"
                ),
                "gpu": get_nested(CONFIG, "execution.gpu"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 10. Run controlled backtest and report generated artifacts

# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP291_SUMMARY = run_full_experiment(CONFIG)
