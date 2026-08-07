# %% [markdown]
# # exp285 exp226 prefix-masked offset predictability readout
#
# This zero-booster PF/Beam diagnostic masks the final 640 rows of each
# eligible known `TVT_input` prefix, replays the fold-safe exp226 geometry-only
# path from the shortened prefix through the well end, freezes prefix-observable
# offset summaries, and only then attaches official evaluation-suffix truth.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Exp226 fold-safe geometry replay helpers
# 4. Fixed pseudo-mask, replay-path, and prefix-summary helpers
# 5. Post-freeze official target and hidden-like helpers
# 6. Correlation, permutation, scope, and scientific-guard helpers
# 7. Full Kaggle CPU orchestration and generated artifacts
# 8. Setup and contract preview
# 9. Run fixed readout and report generated artifacts

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

EXPERIMENT_NAME = "exp285_exp226_prefix_masked_offset_predictability_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FILE_COLUMNS = ("X", "Y", "Z", "MD", "TVT_input")
TARGET_SAFE_COLUMNS = ("id", *TARGET_FILE_COLUMNS)
TARGET_FORBIDDEN_COLUMNS = {
    "TVT",
    "GR",
    "ANCC",
    "tvt_true",
    "tvt_pred",
    "gr_delta",
    "target",
    "error",
    "abs_error",
}
SUMMARY_PAIRS = (
    ("offset_median", "pseudo_offset_median", "official_offset_median"),
    ("offset_slope", "pseudo_offset_slope", "official_offset_slope"),
    (
        "block_drift_rate",
        "pseudo_block_drift_rate",
        "official_block_drift_rate",
    ),
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP285_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp285 config not found in {[str(path) for path in candidates]}")


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
        raise ValueError(f"target-safe frame contains forbidden columns: {leaked}")
    missing = sorted(set(TARGET_SAFE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"target-safe frame is missing {missing}")


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(TARGET_FILE_COLUMNS))
    frame = frame.loc[:, list(TARGET_FILE_COLUMNS)]
    well = path.name.removesuffix("__horizontal_well.csv")
    frame.insert(0, "id", [f"{well}:{row_idx}" for row_idx in range(len(frame))])
    validate_target_safe_frame(frame)
    return frame


def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp285 fixes route=pf_beam")
    fixed = {
        "pseudo_mask.masked_rows": 640,
        "pseudo_mask.minimum_visible_rows_before_cut": 512,
        "pseudo_mask.cuts_per_well": 1,
        "prefix_summary.total_rows": 640,
        "prefix_summary.block_count": 5,
        "prefix_summary.block_rows": 128,
        "negative_control.permutations": 256,
        "model.active_variant_count": 1,
        "model.lightgbm_config_count": 0,
        "model.trained_fold_count": 0,
        "model.booster_count": 0,
        "model.hmm_regeneration_count": 0,
        "model.pf_regeneration_count": 0,
        "execution.total_boosters": 0,
        "execution.hmm_well_runs": 0,
        "execution.pf_well_runs": 0,
        "validation.expected_rows": 3783989,
        "validation.expected_wells": 773,
        "validation.eligibility.minimum_eligible_wells": 750,
        "validation.guards.minimum_primary_spearman_pooled": 0.30,
        "validation.guards.minimum_primary_positive_folds": 5,
        "validation.guards.minimum_primary_spearman_0p20_folds": 4,
        "validation.guards.minimum_primary_sign_balanced_accuracy_pooled": 0.60,
        "validation.guards.maximum_primary_permutation_pvalue": 0.01,
        "validation.guards.minimum_supporting_pair_spearman_pooled": 0.20,
        "validation.guards.minimum_supporting_pair_positive_folds": 4,
        "validation.guards.minimum_passing_supporting_pair_count": 1,
        "geometry_replay.params.theta0": 118.4,
        "geometry_replay.params.k_segments": 16,
        "geometry_replay.params.local_linear_k": 50,
        "geometry_replay.params.local_linear_bandwidth": 500.0,
        "geometry_replay.params.local_linear_ridge": 1.0,
        "geometry_replay.params.smooth_rho": 10.0,
        "geometry_replay.params.gate": 0.35,
        "geometry_replay.params.field_min_proj": 0.3,
        "geometry_replay.params.rot_max_deg": 60.0,
        "geometry_replay.params.ancc_theta_bandwidth": 1500.0,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"exp285 requires {key}={expected}, got {actual}")
    horizons = [
        int(value)
        for value in get_nested(config, "official_target_summary.fixed_horizon_rows")
    ]
    if horizons != [
        256,
        512,
        640,
    ]:
        raise ValueError("exp285 fixes official diagnostic horizons at 256/512/640")
    if [int(value) for value in get_nested(config, "validation.expected_folds")] != [
        0,
        1,
        2,
        3,
        4,
    ]:
        raise ValueError("exp285 fixes the saved exp226 five-fold identity")
    if [float(value) for value in get_nested(config, "geometry_replay.params.kbins")] != [
        0.0,
        750.0,
        1500.0,
        2500.0,
        4000.0,
        1.0e18,
    ]:
        raise ValueError("exp285 fixes the exp226 donor-distance bins")
    if [
        float(value) for value in get_nested(config, "geometry_replay.params.kappa_regimes")
    ] != [0.0, 1000.0, 1500.0, 2000.0]:
        raise ValueError("exp285 fixes the exp226 saved-kappa regime contract")
    if get_nested(config, "pseudo_mask.replay_horizon") != "all_rows_after_cut_through_well_end":
        raise ValueError("exp285 pseudo geometry must extend through the well end")
    if get_nested(config, "prefix_summary.clip") != "none" or get_nested(
        config, "prefix_summary.winsorize"
    ) != "none":
        raise ValueError("exp285 forbids prefix-summary clipping and winsorization")
    required_scopes = [
        "near_0_250_ft",
        "long_tail_1000_plus_ft",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    if list(
        get_nested(config, "validation.guards.require_positive_primary_spearman_scopes")
    ) != required_scopes:
        raise ValueError("exp285 fixes the near/long-tail/hidden-like scope guard")
    if bool(get_nested(config, "geometry_replay.use_exp226_gr_correction")):
        raise ValueError("exp285 forbids exp226 GR correction")
    if bool(get_nested(config, "geometry_replay.use_exp226_u_projection")):
        raise ValueError("exp285 forbids exp226 U projection")
    if bool(get_nested(config, "execution.gpu")):
        raise ValueError("exp285 is CPU-only")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("exp285 forbids inference and submission")
    if bool(get_nested(config, "model.parent_control_retraining")):
        raise ValueError("exp285 forbids parent/control retraining")


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
    oof_spec = get_nested(config, "data.exp226_oof")
    oof_path = resolve_existing(
        str(oof_spec["filename"]), [str(value) for value in oof_spec["candidates"]]
    )
    actual_decompressed = sha256_gzip_decompressed(oof_path)
    if actual_decompressed != str(oof_spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    safe_columns = [str(value) for value in oof_spec["target_free_columns"]]
    if "tvt_true" in safe_columns or "tvt_pred" in safe_columns or "gr_delta" in safe_columns:
        raise ValueError("exp226 target-free OOF allowlist contains forbidden target columns")
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
            "decompressed_sha256": "",
            "rows": len(kappa_frame),
            "wells": 0,
        },
    ]
    return fold_by_well, kappa_by_fold, manifests, oof_path


# %% [markdown]
# ## 4. Fixed pseudo-mask, replay-path, and prefix-summary helpers


# %%
class IneligibleWellError(ValueError):
    """A fixed-contract eligibility exclusion, not a pipeline failure."""


def build_pseudo_mask(
    well: str,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_target_safe_frame(frame)
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    official_last_known = last_contiguous_known_index(tvt_input)
    mask_rows = int(get_nested(config, "pseudo_mask.masked_rows"))
    minimum_visible = int(get_nested(config, "pseudo_mask.minimum_visible_rows_before_cut"))
    cut = official_last_known - mask_rows
    visible_rows = cut + 1
    if visible_rows < minimum_visible:
        raise IneligibleWellError(
            f"visible_rows={visible_rows} below fixed minimum {minimum_visible}"
        )
    heldout = frame.loc[cut + 1 : official_last_known, ["id", "MD", "TVT_input"]].copy()
    if len(heldout) != mask_rows or heldout["TVT_input"].isna().any():
        raise IneligibleWellError("final known block is not exactly 640 contiguous finite rows")
    heldout.insert(0, "well_id", str(well))
    heldout.insert(1, "row_idx", np.arange(cut + 1, official_last_known + 1, dtype=np.int64))
    heldout.insert(2, "masked_offset", np.arange(1, mask_rows + 1, dtype=np.int64))
    heldout = heldout.rename(columns={"TVT_input": "masked_tvt_input"})

    masked = frame.copy()
    masked.loc[cut + 1 :, "TVT_input"] = np.nan
    post_cut_finite = int(masked.loc[cut + 1 :, "TVT_input"].notna().sum())
    if post_cut_finite != 0:
        raise ValueError("pseudo mask failed to remove all post-cut TVT_input")
    manifest = {
        "well_id": str(well),
        "official_last_known_row": int(official_last_known),
        "cut_row": int(cut),
        "visible_rows": int(visible_rows),
        "masked_rows": int(mask_rows),
        "well_rows": int(len(frame)),
        "full_replay_rows": int(len(frame) - cut - 1),
        "post_cut_tvt_input_finite_rows_after_mask": post_cut_finite,
        "masked_tvt_input_access_by_geometry_generator": 0,
        "official_suffix_truth_access_before_prefix_summary_freeze": 0,
    }
    return masked, heldout, manifest


def build_pseudo_path_rows(
    well: str,
    fold: int,
    target: GeometryWell,
    path: np.ndarray,
    row_donor_distance: np.ndarray,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    score_rows = int(get_nested(config, "prefix_summary.total_rows"))
    if target.n != int(manifest["full_replay_rows"]):
        raise ValueError("target replay does not extend through the well end")
    if len(path) != target.n or len(row_donor_distance) != target.n:
        raise ValueError("full replay output length mismatch")
    if target.n < score_rows:
        raise ValueError("full replay is shorter than the fixed prefix score zone")
    cut = int(manifest["cut_row"])
    rows = np.arange(cut + 1, cut + score_rows + 1, dtype=np.int64)
    output = pd.DataFrame(
        {
            "well_id": str(well),
            "fold": int(fold),
            "row_idx": rows,
            "pseudo_cut_row": cut,
            "official_last_known_row": int(manifest["official_last_known_row"]),
            "masked_offset": np.arange(1, score_rows + 1, dtype=np.int64),
            "MD": target.md[rows],
            "pseudo_tvt_geop": path[:score_rows],
            "donor_distance": row_donor_distance[:score_rows],
            "truth_attached": False,
        }
    )
    if output[["MD", "pseudo_tvt_geop", "donor_distance"]].isna().any().any():
        raise ValueError("target-free pseudo geometry rows are not finite")
    return output


def assert_target_free_pseudo_paths(
    pseudo_paths: pd.DataFrame,
    mask_manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, str]:
    forbidden = {"masked_tvt_input", "residual", "tvt_true", "official_offset_median"}
    leaked = sorted(forbidden.intersection(pseudo_paths.columns))
    if leaked:
        raise ValueError(f"target-free pseudo paths contain forbidden columns: {leaked}")
    if pseudo_paths.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("pseudo path row identity is not unique")
    expected_rows = int(get_nested(config, "prefix_summary.total_rows"))
    sizes = pseudo_paths.groupby("well_id", sort=False).size()
    if not sizes.eq(expected_rows).all():
        raise ValueError("every eligible well must have exactly 640 frozen pseudo rows")
    finite = np.isfinite(
        pseudo_paths[["MD", "pseudo_tvt_geop", "donor_distance"]].to_numpy(np.float64)
    )
    if not bool(finite.all()):
        raise ValueError("pseudo path finite coverage is below one")
    if not pseudo_paths["truth_attached"].eq(False).all():  # noqa: E712
        raise ValueError("truth was attached before pseudo path freeze")
    if not mask_manifest["post_cut_tvt_input_finite_rows_after_mask"].eq(0).all():
        raise ValueError("mask identity coverage is below one")
    return {
        "mask_manifest": dataframe_content_sha(mask_manifest),
        "pseudo_geop": dataframe_content_sha(pseudo_paths),
    }


def require_frozen_hashes(frozen_hashes: Mapping[str, str], required: Sequence[str]) -> None:
    missing = sorted(key for key in required if not frozen_hashes.get(key))
    if missing:
        raise ValueError(f"post-freeze access requires frozen content SHA for {missing}")


def summarize_residual_blocks(
    md: Sequence[float],
    residual: Sequence[float],
    blocks: Sequence[np.ndarray],
) -> dict[str, float]:
    md_array = np.asarray(md, dtype=np.float64)
    residual_array = np.asarray(residual, dtype=np.float64)
    if len(md_array) != len(residual_array) or not len(md_array):
        raise ValueError("summary MD/residual length mismatch")
    if not np.isfinite(md_array).all() or not np.isfinite(residual_array).all():
        raise ValueError("summary requires finite MD and residual coverage one")
    if len(blocks) != 5 or any(len(block) == 0 for block in blocks):
        raise ValueError("summary requires exactly five non-empty blocks")
    block_medians = np.asarray(
        [np.median(residual_array[np.asarray(block, dtype=np.int64)]) for block in blocks],
        dtype=np.float64,
    )
    block_centers = np.asarray(
        [np.median(md_array[np.asarray(block, dtype=np.int64)]) for block in blocks],
        dtype=np.float64,
    )
    centered_md = block_centers - block_centers.mean()
    denominator = float(np.sum(centered_md**2))
    center_span = float(block_centers[-1] - block_centers[0])
    if (
        not np.isfinite(denominator)
        or denominator <= 0
        or not np.isfinite(center_span)
        or center_span <= 0
    ):
        raise ValueError("summary block-center MD denominator is non-positive")
    centered_residual = block_medians - block_medians.mean()
    slope = float(np.sum(centered_md * centered_residual) / denominator)
    raw_drift = float(block_medians[-1] - block_medians[0])
    return {
        "offset_median": float(np.median(residual_array)),
        "offset_slope": slope,
        "block_drift_rate": raw_drift / center_span,
        "first_block_median": float(block_medians[0]),
        "last_block_median": float(block_medians[-1]),
        "raw_block_drift": raw_drift,
        "block_center_md_span": center_span,
        "residual_finite_fraction": float(np.isfinite(residual_array).mean()),
    }


def build_prefix_offset_summary(
    pseudo_paths: pd.DataFrame,
    heldout_prefix: pd.DataFrame,
    *,
    frozen_hashes: Mapping[str, str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    require_frozen_hashes(frozen_hashes, ("pseudo_geop",))
    merged = pseudo_paths.merge(
        heldout_prefix,
        on=["well_id", "row_idx", "masked_offset", "MD"],
        how="left",
        validate="one_to_one",
    )
    if merged["masked_tvt_input"].isna().any():
        raise ValueError("masked-known TVT_input did not cover the frozen pseudo path")
    merged["residual"] = merged["masked_tvt_input"] - merged["pseudo_tvt_geop"]
    total_rows = int(get_nested(config, "prefix_summary.total_rows"))
    block_rows = int(get_nested(config, "prefix_summary.block_rows"))
    summary_rows: list[dict[str, Any]] = []
    for well, part in merged.groupby("well_id", sort=True):
        part = part.sort_values("masked_offset", kind="mergesort")
        if len(part) != total_rows:
            raise ValueError(f"prefix summary {well} does not contain {total_rows} rows")
        blocks = [
            np.arange(start, start + block_rows, dtype=np.int64)
            for start in range(0, total_rows, block_rows)
        ]
        values = summarize_residual_blocks(part["MD"], part["residual"], blocks)
        first = part.iloc[0]
        summary_rows.append(
            {
                "well_id": str(well),
                "fold": int(first["fold"]),
                "pseudo_cut_row": int(first["pseudo_cut_row"]),
                "official_last_known_row": int(first["official_last_known_row"]),
                "prefix_rows": len(part),
                **{f"pseudo_{key}": value for key, value in values.items()},
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("well_id", kind="mergesort")
    numeric = summary.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("prefix summary finite coverage is below one")
    if summary["well_id"].duplicated().any():
        raise ValueError("prefix summary must have one row per well")
    return summary


def freeze_prefix_summary(
    prefix_summary: pd.DataFrame,
    frozen_hashes: Mapping[str, str],
) -> dict[str, str]:
    require_frozen_hashes(frozen_hashes, ("mask_manifest", "pseudo_geop"))
    output = dict(frozen_hashes)
    output["prefix_summary"] = dataframe_content_sha(prefix_summary)
    return output


# %% [markdown]
# ## 5. Post-freeze official target and hidden-like helpers


# %%
def load_official_target_rows(
    oof_path: Path,
    prefix_summary: pd.DataFrame,
    *,
    frozen_hashes: Mapping[str, str],
) -> pd.DataFrame:
    require_frozen_hashes(frozen_hashes, ("pseudo_geop", "prefix_summary"))
    columns = ["well_id", "row_idx", "suffix_offset", "fold", "tvt_geop", "tvt_true"]
    frame = pd.read_csv(oof_path, usecols=columns, dtype={"well_id": str})
    eligible = set(prefix_summary["well_id"].astype(str))
    frame = frame.loc[frame["well_id"].isin(eligible)].copy()
    if not set(frame["well_id"].astype(str)) == eligible:
        raise ValueError("official target rows do not cover every eligible well")
    values = frame[["tvt_geop", "tvt_true"]].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("official target rows are not finite")
    frame["official_residual"] = frame["tvt_true"] - frame["tvt_geop"]
    return frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


def build_official_target_summary(
    official_rows: pd.DataFrame,
    raw_dir: Path,
    prefix_summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    prefix_index = prefix_summary.set_index("well_id")
    horizons = [
        int(value)
        for value in get_nested(config, "official_target_summary.fixed_horizon_rows")
    ]
    summary_rows: list[dict[str, Any]] = []
    for well, part in official_rows.groupby("well_id", sort=True):
        part = part.sort_values("row_idx", kind="mergesort").copy()
        md_frame = pd.read_csv(raw_dir / f"{well}__horizontal_well.csv", usecols=["MD"])
        row_idx = part["row_idx"].to_numpy(np.int64)
        official_last = int(prefix_index.loc[str(well), "official_last_known_row"])
        expected_fold = int(prefix_index.loc[str(well), "fold"])
        if int(part["fold"].iloc[0]) != expected_fold or part["fold"].nunique() != 1:
            raise ValueError(f"official target fold identity mismatch for {well}")
        if row_idx[0] != official_last + 1 or not np.array_equal(
            row_idx, np.arange(official_last + 1, official_last + 1 + len(part))
        ):
            raise ValueError(f"official target row identity is not contiguous for {well}")
        if not np.array_equal(
            part["suffix_offset"].to_numpy(np.int64), np.arange(len(part), dtype=np.int64)
        ):
            raise ValueError(f"official target suffix_offset identity mismatch for {well}")
        md = pd.to_numeric(md_frame.iloc[row_idx]["MD"], errors="raise").to_numpy(np.float64)
        anchor_md = float(md_frame.iloc[official_last]["MD"])
        part["MD"] = md
        part["md_since_official_cut"] = md - anchor_md
        blocks = [
            np.asarray(block, dtype=np.int64)
            for block in np.array_split(np.arange(len(part)), 5)
        ]
        values = summarize_residual_blocks(part["MD"], part["official_residual"], blocks)
        output: dict[str, Any] = {
            "well_id": str(well),
            "fold": int(part["fold"].iloc[0]),
            "official_rows": len(part),
            **{f"official_{key}": value for key, value in values.items()},
        }
        residual = part["official_residual"].to_numpy(np.float64)
        for horizon in horizons:
            output[f"official_h{horizon}_offset_median"] = float(
                np.median(residual[: min(horizon, len(residual))])
            )
        near = part.loc[
            (part["md_since_official_cut"] >= 0.0)
            & (part["md_since_official_cut"] < 250.0),
            "official_residual",
        ]
        long_tail = part.loc[
            part["md_since_official_cut"] >= 1000.0, "official_residual"
        ]
        output["official_near_0_250_offset_median"] = (
            float(np.median(near)) if len(near) else float("nan")
        )
        output["official_long_tail_1000_plus_offset_median"] = (
            float(np.median(long_tail)) if len(long_tail) else float("nan")
        )
        summary_rows.append(output)
    output = pd.DataFrame(summary_rows).sort_values("well_id", kind="mergesort")
    if output["well_id"].duplicated().any():
        raise ValueError("official target summary must have one row per well")
    required = [
        "official_offset_median",
        "official_offset_slope",
        "official_block_drift_rate",
    ]
    if not np.isfinite(output[required].to_numpy(np.float64)).all():
        raise ValueError("official primary/supporting summaries are not finite")
    return output


def load_hidden_like(
    config: Mapping[str, Any],
    *,
    frozen_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen_hashes(frozen_hashes, ("prefix_summary",))
    spec = get_nested(config, "data.hidden_like")
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"name": "hidden_like", "enabled": False}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    role_columns = [str(value) for value in spec["role_columns"].values()]
    required = {"well_id", *role_columns}
    if not required.issubset(frame.columns):
        raise ValueError("hidden-like assignments are missing fixed role columns")
    return frame[["well_id", *role_columns]].copy(), {
        "name": "exp115_hidden_like_assignments",
        "path": str(path),
        "raw_sha256": actual_sha,
        "decompressed_sha256": "",
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


# %% [markdown]
# ## 6. Correlation, permutation, scope, and scientific-guard helpers


# %%
def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = pd.Series(np.asarray(left, dtype=np.float64))
    y = pd.Series(np.asarray(right, dtype=np.float64))
    finite = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    return pearson_correlation(
        x.loc[finite].rank(method="average").to_numpy(np.float64),
        y.loc[finite].rank(method="average").to_numpy(np.float64),
    )


def sign_metrics(predictor: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    predicted_positive = x[finite] >= 0.0
    target_positive = y[finite] >= 0.0
    if not len(target_positive):
        return float("nan"), float("nan")
    accuracy = float(np.mean(predicted_positive == target_positive))
    recalls: list[float] = []
    for label in (False, True):
        selected = target_positive == label
        if not selected.any():
            return accuracy, float("nan")
        recalls.append(float(np.mean(predicted_positive[selected] == label)))
    return accuracy, float(np.mean(recalls))


def correlation_metric_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    family: str,
    predictor: str,
    target: str,
) -> dict[str, Any]:
    selected = frame[[predictor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    accuracy, balanced = sign_metrics(selected[predictor], selected[target])
    return {
        "scope": scope,
        "family": family,
        "predictor": predictor,
        "target": target,
        "wells": len(selected),
        "pearson": pearson_correlation(selected[predictor], selected[target]),
        "spearman": spearman_correlation(selected[predictor], selected[target]),
        "sign_accuracy": accuracy,
        "sign_balanced_accuracy": balanced,
    }


def build_predictability_readout(
    prefix_summary: pd.DataFrame,
    official_summary: pd.DataFrame,
    hidden_like: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_well = prefix_summary.merge(
        official_summary, on=["well_id", "fold"], validate="one_to_one"
    )
    if not hidden_like.empty:
        by_well = by_well.merge(hidden_like, on="well_id", how="left", validate="one_to_one")
    metric_rows: list[dict[str, Any]] = []
    for family, predictor, target in SUMMARY_PAIRS:
        metric_rows.append(
            correlation_metric_row(
                by_well,
                scope="overall",
                family=family,
                predictor=predictor,
                target=target,
            )
        )
        for fold, part in by_well.groupby("fold", sort=True):
            metric_rows.append(
                correlation_metric_row(
                    part,
                    scope=f"fold_{int(fold)}",
                    family=family,
                    predictor=predictor,
                    target=target,
                )
            )
    for horizon in (256, 512, 640):
        metric_rows.append(
            correlation_metric_row(
                by_well,
                scope=f"h{horizon}",
                family="offset_median",
                predictor="pseudo_offset_median",
                target=f"official_h{horizon}_offset_median",
            )
        )
    for scope, target in (
        ("near_0_250_ft", "official_near_0_250_offset_median"),
        ("long_tail_1000_plus_ft", "official_long_tail_1000_plus_offset_median"),
    ):
        metric_rows.append(
            correlation_metric_row(
                by_well,
                scope=scope,
                family="offset_median",
                predictor="pseudo_offset_median",
                target=target,
            )
        )
    role_map = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    for scope, role_column in role_map.items():
        if role_column not in by_well:
            part = by_well.iloc[0:0]
        else:
            part = by_well.loc[by_well[role_column].astype(str).eq("valid")]
        metric_rows.append(
            correlation_metric_row(
                part,
                scope=scope,
                family="offset_median",
                predictor="pseudo_offset_median",
                target="official_offset_median",
            )
        )
    return by_well.sort_values("well_id", kind="mergesort"), pd.DataFrame(metric_rows)


def build_permutation_metrics(
    by_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = by_well.sort_values(["fold", "well_id"], kind="mergesort").reset_index(drop=True)
    observed = spearman_correlation(
        ordered["pseudo_offset_median"], ordered["official_offset_median"]
    )
    count = int(get_nested(config, "negative_control.permutations"))
    seed = int(get_nested(config, "negative_control.seed"))
    rows: list[dict[str, Any]] = []
    for permutation in range(count):
        shuffled = ordered["pseudo_offset_median"].to_numpy(np.float64).copy()
        for fold, indices in ordered.groupby("fold", sort=True).groups.items():
            positions = np.asarray(sorted(indices), dtype=np.int64)
            rng = np.random.default_rng(
                stable_seed(EXPERIMENT_NAME, seed, int(fold), permutation, "prefix_summary_shuffle")
            )
            shuffled[positions] = shuffled[positions][rng.permutation(len(positions))]
        value = spearman_correlation(shuffled, ordered["official_offset_median"])
        rows.append({"permutation": permutation, "pooled_spearman": value})
    frame = pd.DataFrame(rows)
    exceedances = int(frame["pooled_spearman"].ge(observed).sum())
    summary = {
        "observed_spearman": observed,
        "null_p95": float(frame["pooled_spearman"].quantile(0.95)),
        "exceedances": float(exceedances),
        "permutations": float(count),
        "pvalue": float((exceedances + 1) / (count + 1)),
    }
    return frame, summary


def metric_lookup(metrics: pd.DataFrame, scope: str, family: str) -> pd.Series:
    row = metrics.loc[(metrics["scope"] == scope) & (metrics["family"] == family)]
    if len(row) != 1:
        raise ValueError(f"missing unique metric row for {scope}/{family}")
    return row.iloc[0]


def evaluate_scientific_guard(
    mask_manifest: pd.DataFrame,
    pseudo_paths: pd.DataFrame,
    prefix_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    permutation_summary: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.guards")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    folds = sorted(int(value) for value in prefix_summary["fold"].unique())
    primary = metric_lookup(metrics, "overall", "offset_median")
    fold_primary = metrics.loc[
        metrics["scope"].str.startswith("fold_") & metrics["family"].eq("offset_median")
    ]
    supporting: dict[str, Any] = {}
    supporting_pass_count = 0
    for family in ("offset_slope", "block_drift_rate"):
        pooled = metric_lookup(metrics, "overall", family)
        fold_rows = metrics.loc[
            metrics["scope"].str.startswith("fold_") & metrics["family"].eq(family)
        ]
        positive_folds = int(fold_rows["spearman"].gt(0.0).sum())
        passed = bool(
            np.isfinite(float(pooled["spearman"]))
            and float(pooled["spearman"])
            >= float(guards["minimum_supporting_pair_spearman_pooled"])
            and positive_folds >= int(guards["minimum_supporting_pair_positive_folds"])
        )
        supporting[family] = {
            "passed": passed,
            "pooled_spearman": float(pooled["spearman"]),
            "positive_folds": positive_folds,
        }
        supporting_pass_count += int(passed)
    scope_values: dict[str, float] = {}
    for scope in [str(value) for value in guards["require_positive_primary_spearman_scopes"]]:
        scope_values[scope] = float(metric_lookup(metrics, scope, "offset_median")["spearman"])

    technical = {
        "minimum_eligible_wells": int(prefix_summary["well_id"].nunique())
        >= int(guards["required_eligible_wells"]),
        "all_expected_folds": folds == expected_folds,
        "target_well_donor_exclusion_coverage": float(
            mask_manifest["target_well_in_donor_field"].eq(False).mean()  # noqa: E712
        )
        == float(guards["required_target_well_donor_exclusion_coverage"]),
        "pseudo_mask_identity_coverage": float(
            mask_manifest["post_cut_tvt_input_finite_rows_after_mask"].eq(0).mean()
        )
        == float(guards["required_pseudo_mask_identity_coverage"]),
        "pseudo_path_finite_coverage": float(
            np.isfinite(
                pseudo_paths[["MD", "pseudo_tvt_geop", "donor_distance"]].to_numpy(np.float64)
            ).mean()
        )
        == float(guards["required_pseudo_path_finite_coverage"]),
        "pseudo_summary_finite_coverage": float(
            np.isfinite(
                prefix_summary[
                    ["pseudo_offset_median", "pseudo_offset_slope", "pseudo_block_drift_rate"]
                ].to_numpy(np.float64)
            ).mean()
        )
        == float(guards["required_pseudo_summary_finite_coverage"]),
        "masked_tvt_input_access_before_path_freeze": 0
        == int(guards["required_masked_tvt_input_access_before_path_freeze"]),
        "official_truth_access_before_summary_freeze": 0
        == int(guards["required_official_truth_access_before_summary_freeze"]),
    }
    primary_guard = {
        "pooled_spearman": np.isfinite(float(primary["spearman"]))
        and float(primary["spearman"]) >= float(guards["minimum_primary_spearman_pooled"]),
        "positive_all_folds": int(fold_primary["spearman"].gt(0.0).sum())
        >= int(guards["minimum_primary_positive_folds"]),
        "spearman_0p20_required_folds": int(fold_primary["spearman"].ge(0.20).sum())
        >= int(guards["minimum_primary_spearman_0p20_folds"]),
        "sign_balanced_accuracy": np.isfinite(float(primary["sign_balanced_accuracy"]))
        and float(primary["sign_balanced_accuracy"])
        >= float(guards["minimum_primary_sign_balanced_accuracy_pooled"]),
        "permutation_pvalue": float(permutation_summary["pvalue"])
        <= float(guards["maximum_primary_permutation_pvalue"]),
    }
    supporting_guard = {
        "minimum_passing_family_count": supporting_pass_count
        >= int(guards["minimum_passing_supporting_pair_count"]),
        "families": supporting,
    }
    scope_guard = {
        "all_required_scopes_positive": bool(
            all(np.isfinite(value) and value > 0.0 for value in scope_values.values())
        ),
        "spearman": scope_values,
    }
    scalar_checks = [
        *technical.values(),
        *primary_guard.values(),
        supporting_guard["minimum_passing_family_count"],
        scope_guard["all_required_scopes_positive"],
    ]
    return {
        "passed": bool(all(bool(value) for value in scalar_checks)),
        "technical": technical,
        "primary": primary_guard,
        "supporting": supporting_guard,
        "scope": scope_guard,
        "readout": {
            "eligible_wells": int(prefix_summary["well_id"].nunique()),
            "folds": folds,
            "primary_spearman": float(primary["spearman"]),
            "primary_sign_balanced_accuracy": float(primary["sign_balanced_accuracy"]),
            "primary_positive_folds": int(fold_primary["spearman"].gt(0.0).sum()),
            "primary_spearman_0p20_folds": int(fold_primary["spearman"].ge(0.20).sum()),
            "permutation_pvalue": float(permutation_summary["pvalue"]),
            "permutation_null_p95": float(permutation_summary["null_p95"]),
        },
    }


# %% [markdown]
# ## 7. Full Kaggle CPU orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp285 readout must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is "
            "reserved for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp285 Kaggle CPU execution is not approved")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise RuntimeError("exp285 implementation has not been approved")
    validate_scientific_contract(config)
    started = time.time()
    raw_dir = train_data_dir(config)
    raw_wells = list_horizontal_wells(raw_dir)
    fold_by_well, kappa_by_fold, input_manifests, oof_path = load_exp226_fold_contract(config)
    if len(raw_wells) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("raw horizontal well count does not match the fixed contract")
    if set(raw_wells) != set(fold_by_well):
        raise ValueError("raw horizontal and exp226 OOF well sets differ")

    params = params_from_config(config)
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    mask_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    pseudo_parts: list[pd.DataFrame] = []
    heldout_parts: list[pd.DataFrame] = []
    raw_manifest_rows = [
        {
            "well_id": well,
            "fold": int(fold_by_well[well]),
            "horizontal_path": str(raw_dir / f"{well}__horizontal_well.csv"),
            "horizontal_raw_sha256": sha256_path(
                raw_dir / f"{well}__horizontal_well.csv"
            ),
        }
        for well in raw_wells
    ]
    fold_manifest_rows: list[dict[str, Any]] = []

    for fold in expected_folds:
        source_ids = sorted(well for well in raw_wells if fold_by_well[well] != fold)
        target_ids = sorted(well for well in raw_wells if fold_by_well[well] == fold)
        overlap = set(source_ids).intersection(target_ids)
        if overlap:
            raise ValueError(f"fold {fold} donor/target overlap: {sorted(overlap)[:3]}")
        source_wells = [
            load_source_geometry_well(raw_dir / f"{well}__horizontal_well.csv", params, wi=index)
            for index, well in enumerate(source_ids)
        ]
        fields = build_fields(source_wells, params)
        kappa = kappa_by_fold[fold]
        fold_manifest_rows.append(
            {
                "fold": fold,
                "source_wells": len(source_ids),
                "target_wells": len(target_ids),
                "donor_target_overlap": len(overlap),
                "raw_field_rows": len(fields.f_raw),
                "smooth_field_rows": len(fields.f_sm),
                "surface_rows": len(fields.surface_points),
            }
        )
        print(f"exp285 fold={fold} source_wells={len(source_ids)} target_wells={len(target_ids)}")
        for index, well in enumerate(target_ids, start=1):
            horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
            try:
                safe_frame = load_target_safe_horizontal(horizontal_path)
                masked, heldout, manifest = build_pseudo_mask(well, safe_frame, config)
                target = build_target_geometry_well(
                    well,
                    masked,
                    cut=int(manifest["cut_row"]),
                    params=params,
                )
                path, donor_distance = replay_exp226_geometry(target, fields, kappa, params)
                pseudo = build_pseudo_path_rows(
                    well,
                    fold,
                    target,
                    path,
                    donor_distance,
                    manifest,
                    config,
                )
                manifest.update(
                    {
                        "fold": fold,
                        "geometry_rows": len(path),
                        "score_rows": len(pseudo),
                        "geometry_finite_coverage": float(np.isfinite(path).mean()),
                        "target_well_in_donor_field": False,
                        "source_well_count": len(source_ids),
                    }
                )
                mask_rows.append(manifest)
                pseudo_parts.append(pseudo)
                heldout_parts.append(heldout)
            except IneligibleWellError as exc:
                exclusion_rows.append({"well_id": well, "fold": fold, "reason": str(exc)})
            if index % 25 == 0 or index == len(target_ids):
                print(
                    f"exp285 fold={fold} processed={index}/{len(target_ids)} "
                    f"eligible_total={len(mask_rows)}"
                )
        del fields, source_wells

    if not mask_rows:
        raise ValueError("exp285 generated zero eligible prefix-masked replays")
    mask_manifest = pd.DataFrame(mask_rows).sort_values("well_id", kind="mergesort")
    exclusions = pd.DataFrame(exclusion_rows)
    pseudo_paths = pd.concat(pseudo_parts, ignore_index=True).sort_values(
        ["well_id", "masked_offset"], kind="mergesort"
    )
    heldout_prefix = pd.concat(heldout_parts, ignore_index=True).sort_values(
        ["well_id", "masked_offset"], kind="mergesort"
    )
    frozen_hashes = assert_target_free_pseudo_paths(pseudo_paths, mask_manifest, config)

    artifacts = artifact_dir()
    mask_path = artifacts / f"{OUTPUT_PREFIX}_mask_manifest.csv"
    mask_manifest.to_csv(mask_path, index=False)
    exclusion_path = artifacts / f"{OUTPUT_PREFIX}_ineligible_wells.csv"
    exclusions.to_csv(exclusion_path, index=False)
    pseudo_artifact = write_csv_gzip(
        pseudo_paths,
        artifacts / f"{OUTPUT_PREFIX}_target_free_pseudo_geop.csv.gz",
    )

    prefix_summary = build_prefix_offset_summary(
        pseudo_paths,
        heldout_prefix,
        frozen_hashes=frozen_hashes,
        config=config,
    )
    frozen_hashes = freeze_prefix_summary(prefix_summary, frozen_hashes)
    prefix_path = artifacts / f"{OUTPUT_PREFIX}_prefix_offset_summary.csv"
    prefix_summary.to_csv(prefix_path, index=False)

    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "official_truth_attached": False,
        "masked_tvt_input_access_by_geometry_generator": 0,
        "official_suffix_truth_access_before_prefix_summary_freeze": 0,
        "source_truth_policy": "other_fold_wells_only_for_exp226_geometry_field",
        "pseudo_mask": get_nested(config, "pseudo_mask"),
        "geometry_replay": get_nested(config, "geometry_replay"),
        "prefix_summary": get_nested(config, "prefix_summary"),
        "negative_control": get_nested(config, "negative_control"),
        "frozen_content_sha256": frozen_hashes,
        "schemas": {
            "mask_manifest": {column: str(dtype) for column, dtype in mask_manifest.dtypes.items()},
            "pseudo_geop": {column: str(dtype) for column, dtype in pseudo_paths.dtypes.items()},
            "prefix_summary": {
                column: str(dtype) for column, dtype in prefix_summary.dtypes.items()
            },
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_contract.json"
    write_json(contract_path, contract)

    # Official true TVT is first read below, after pseudo geometry and prefix
    # summaries have both been persisted and assigned logical content hashes.
    official_rows = load_official_target_rows(
        oof_path,
        prefix_summary,
        frozen_hashes=frozen_hashes,
    )
    official_summary = build_official_target_summary(
        official_rows,
        raw_dir,
        prefix_summary,
        config,
    )
    official_path = artifacts / f"{OUTPUT_PREFIX}_official_target_summary.csv"
    official_summary.to_csv(official_path, index=False)
    hidden_like, hidden_manifest = load_hidden_like(config, frozen_hashes=frozen_hashes)
    by_well, metrics = build_predictability_readout(
        prefix_summary,
        official_summary,
        hidden_like,
    )
    permutation_metrics, permutation_summary = build_permutation_metrics(by_well, config)
    guard = evaluate_scientific_guard(
        mask_manifest,
        pseudo_paths,
        prefix_summary,
        metrics,
        permutation_summary,
        config,
    )

    overall_path = artifacts / f"{OUTPUT_PREFIX}_overall_metrics.csv"
    metrics.loc[metrics["scope"].eq("overall")].to_csv(overall_path, index=False)
    fold_path = artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv"
    metrics.loc[metrics["scope"].str.startswith("fold_")].to_csv(fold_path, index=False)
    scope_path = artifacts / f"{OUTPUT_PREFIX}_scope_metrics.csv"
    metrics.loc[
        ~metrics["scope"].eq("overall") & ~metrics["scope"].str.startswith("fold_")
    ].to_csv(scope_path, index=False)
    permutation_path = artifacts / f"{OUTPUT_PREFIX}_permutation_metrics.csv"
    permutation_metrics.to_csv(permutation_path, index=False)
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv"
    by_well.to_csv(by_well_path, index=False)

    raw_manifest = pd.DataFrame(raw_manifest_rows).sort_values("well_id", kind="mergesort")
    fold_manifest = pd.DataFrame(fold_manifest_rows).sort_values("fold", kind="mergesort")
    input_manifests.extend(
        [
            hidden_manifest,
            {
                "name": "eligible_raw_horizontal_files",
                "path": str(raw_dir),
                "raw_sha256": dataframe_content_sha(
                    raw_manifest, ["well_id", "horizontal_raw_sha256"]
                ),
                "decompressed_sha256": "",
                "rows": len(raw_manifest),
                "wells": int(raw_manifest["well_id"].nunique()),
            },
            {
                "name": "fold_safe_donor_field_contract",
                "path": str(raw_dir),
                "raw_sha256": dataframe_content_sha(fold_manifest),
                "decompressed_sha256": "",
                "rows": len(fold_manifest),
                "wells": 0,
            },
        ]
    )
    input_manifest = pd.DataFrame(input_manifests)
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)

    status = (
        "train_side_readout_completed_guard_passed"
        if guard["passed"]
        else "train_side_readout_completed_guard_failed"
    )
    output_paths = {
        "contract": contract_path,
        "mask_manifest": mask_path,
        "ineligible_wells": exclusion_path,
        "prefix_summary": prefix_path,
        "official_target_summary": official_path,
        "overall_metrics": overall_path,
        "fold_metrics": fold_path,
        "scope_metrics": scope_path,
        "permutation_metrics": permutation_path,
        "by_well_metrics": by_well_path,
        "input_manifest": input_manifest_path,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "eligible_wells": int(prefix_summary["well_id"].nunique()),
        "ineligible_wells": len(exclusions),
        "folds": sorted(int(value) for value in prefix_summary["fold"].unique()),
        "active_readout_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "guard": guard,
        "permutation": permutation_summary,
        "truth_attachment": {
            "stage": "after_pseudo_geometry_and_prefix_summary_persisted_and_hashed",
            "official_suffix_truth_access_before_prefix_summary_freeze": 0,
            "frozen_content_sha256": frozen_hashes,
        },
        "target_free_pseudo_artifact": pseudo_artifact,
        "file_sha256": {name: sha256_path(path) for name, path in output_paths.items()},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": "permit_separate_prefix_calibrated_candidate_design"
        if guard["passed"]
        else "close_without_parameter_rescue",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics_json = {
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
            "permutation": permutation_summary,
            "frozen_content_sha256": frozen_hashes,
        },
        "active_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_variants": 0,
        "pf_variants": 0,
        "inference": False,
        "submission": False,
        "notes": "Prefix-masked predictability readout only; no correction or submission.",
    }
    write_json(metrics_output_path(), metrics_json)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 8. Setup and contract preview


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
                "parent": get_nested(CONFIG, "lineage.parent"),
                "mask_rows": get_nested(CONFIG, "pseudo_mask.masked_rows"),
                "visible_minimum": get_nested(
                    CONFIG, "pseudo_mask.minimum_visible_rows_before_cut"
                ),
                "replay_horizon": get_nested(CONFIG, "pseudo_mask.replay_horizon"),
                "block_count": get_nested(CONFIG, "prefix_summary.block_count"),
                "block_rows": get_nested(CONFIG, "prefix_summary.block_rows"),
                "permutations": get_nested(CONFIG, "negative_control.permutations"),
                "active_variants": get_nested(CONFIG, "execution.active_readout_variants"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "hmm_well_runs": get_nested(CONFIG, "execution.hmm_well_runs"),
                "pf_well_runs": get_nested(CONFIG, "execution.pf_well_runs"),
                "parent_control_retraining": get_nested(
                    CONFIG, "execution.control_or_parent_retraining"
                ),
                "implementation_approved": get_nested(
                    CONFIG, "execution.implementation_approved"
                ),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 9. Run fixed readout and report generated artifacts


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP285_SUMMARY = run_full_experiment(CONFIG)
