# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp376 formation-conditioned exp226 K16 donor kernel
#
# This CPU-only train-side audit keeps exp226's true-TVT K16 donor slopes,
# XY-nearest 50 support, adaptive kappa, ANCC near-strike path, GR correction,
# and U projection fixed. The sole scientific change is a bounded soft factor
# on the existing XY local-linear weights, computed from fold-local formation
# relative coordinates. Formation support, the OOF path, the exp293 deployable12
# bank, and block assignments are frozen before evaluation truth is attached.
# No selector, model, inference path, or submission is created.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Fold-safe formation and exp226 deterministic geometry helpers
# 4. Exp293 fixed candidate-bank and block helpers
# 5. Target-free OOF generation and freeze boundary
# 6. Post-freeze truth loader and direct readout
# 7. Add-one candidate novelty readout
# 8. PASS/FAIL decision and generated artifacts
# 9. Setup, contract preview, and execution

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp376_exp226_formation_conditioned_k16_donor_kernel"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
VALUE_KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
VALUE_READ_COLUMNS = VALUE_KEY_COLUMNS + [
    "last_known_tvt",
    "candidate_tvt",
    "candidate_available",
    "candidate_finite",
]
EXPECTED_CANDIDATE_ORDER = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
FORMULA_CANDIDATES = EXPECTED_CANDIDATE_ORDER[6:]
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
EXPECTED_VARIANTS = {"formation_conditioned_k16": 16}
FORBIDDEN_CANDIDATE_COLUMNS = {
    "TVT",
    "target",
    "true_tvt",
    "tvt_true",
    "error",
    "abs_error",
    "oracle",
    "oracle_label",
    "oracle_candidate",
}
DIRECT_BUCKETS = {
    "near_0_250": (0.0, 250.0),
    "mid_250_1000": (250.0, 1000.0),
    "1000_plus": (1000.0, math.inf),
}

# %% [markdown]
# ## 2. Runtime, path, SHA, and serialization helpers

# %%
def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP376_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    root = project_root()
    nested = root / "experiments" / EXPERIMENT_NAME
    return nested if nested.exists() else Path.cwd()


def find_config_path() -> Path:
    for candidate in (
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    matches = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp376 config.yaml was not found unambiguously")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def runtime_artifacts_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_work_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / ".exp376_work"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / ".exp376_work"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    return (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "metrics.json"
    )


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    string_columns = [
        column
        for column, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.StringDtype)
    ]
    object_columns = [
        column for column, dtype in frame.dtypes.items() if dtype == object
    ]
    if not string_columns and not object_columns:
        return frame
    normalized = frame.copy()
    for column in string_columns:
        normalized[column] = normalized[column].astype(object)
    for column in object_columns:
        normalized[column] = normalized[column].map(_normalize_object_for_hash)
    return normalized


def _normalize_object_for_hash(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        value = sorted(value, key=str)
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(
            to_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return value


def frame_content_sha256(
    frame: pd.DataFrame, columns: Iterable[str] | None = None
) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = _normalize_frame_for_hash(selected)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    row_hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(
        row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes()
    )
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def write_gzip_frame(path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    return sha256_file(path), sha256_decompressed_gzip(path)


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    root = project_root()
    for raw_pattern in patterns:
        raw = str(raw_pattern)
        path = Path(raw)
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            found.append(direct)
            continue
        for match in glob.glob(raw, recursive=True):
            candidate = Path(match)
            if candidate.exists():
                found.append(candidate)
        if not path.is_absolute():
            for match in glob.glob(str(root / raw), recursive=True):
                candidate = Path(match)
                if candidate.exists():
                    found.append(candidate)
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(str(path.resolve()), path)
    return list(unique.values())


def resolve_file(
    patterns: Sequence[str],
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    if expected_sha256:
        matching = [path for path in candidates if sha256_file(path) == expected_sha256]
        if matching:
            return sorted(matching, key=lambda item: len(str(item)))[0]
        if candidates:
            evidence = {str(path): sha256_file(path) for path in candidates}
            raise ValueError(f"{label} SHA mismatch: {evidence}")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"{label} not found from patterns: {patterns}")
    raise ValueError(f"{label} resolved to multiple files: {candidates}")


def resolve_gzip_by_decompressed_sha(
    patterns: Sequence[str], *, label: str, expected_sha256: str
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    matches = [
        path for path in candidates if sha256_decompressed_gzip(path) == expected_sha256
    ]
    if matches:
        return sorted(matches, key=lambda item: len(str(item)))[0]
    evidence = {
        str(path): sha256_decompressed_gzip(path) for path in candidates
    }
    if evidence:
        raise ValueError(f"{label} decompressed SHA mismatch: {evidence}")
    raise FileNotFoundError(f"{label} not found from patterns: {patterns}")


def reject_forbidden_candidate_columns(columns: Iterable[str]) -> None:
    normalized = {str(column) for column in columns}
    forbidden = normalized & FORBIDDEN_CANDIDATE_COLUMNS
    token_forbidden = {
        column
        for column in normalized
        if any(
            token in column.lower()
            for token in ("true_tvt", "tvt_true", "abs_error", "oracle_label")
        )
    }
    if forbidden or token_forbidden:
        raise ValueError(
            "candidate partition exposes forbidden truth/readout columns: "
            f"{sorted(forbidden | token_forbidden)}"
        )


# %% [markdown]
# ## 3. Fold-safe formation and exp226 deterministic geometry helpers
#
# The following numerical helpers are the train-relevant subset of exp226 plus
# the well-median, k=10, self-excluding FormationPlaneKNN semantics used by
# exp287. Raw horizontal truth and all six formation columns are opened only
# for outer-train donor wells. Outer-valid objects retain no truth, formation,
# residual, or fitted segment coefficient arrays.

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
    kbins: tuple[float, ...] = (0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18)
    kappa_regimes: tuple[float, ...] = (0.0, 1000.0, 1500.0, 2000.0)
    rot_max_deg: float = 60.0
    ancc_theta_bandwidth: float = 1500.0
    gr_grid_step: float = 0.5
    gr_window: int = 500
    gr_stride: int = 125
    gr_tau: float = 2.0
    gr_w_mse: float = 0.5
    gr_w_lvl: float = 0.1
    gr_sh_a: float = 1.1
    gr_sh_b: float = 0.12
    gr_sh_lo: float = 0.3
    gr_cap: float = 4.0
    gr_s0: float = 0.10
    gr_extent: float = 30.0
    enable_gr_correction: bool = True
    enable_u_projection: bool = True
    u_projection_deg: int = 4
    u_projection_beta: float = 0.75
    u_projection_iters: int = 4

    @property
    def n_bins(self) -> int:
        return len(self.kbins) - 1

    @property
    def kappa_dim(self) -> int:
        return 2 * self.n_bins + 2


@dataclass
class WellData:
    wid: str
    wi: int
    s: int
    n: int
    ndz: np.ndarray
    anchor: float
    ti: np.ndarray
    segid: np.ndarray
    mid: np.ndarray
    z_mid: np.ndarray
    proj: np.ndarray
    az: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray | None
    typewell_path: Path
    tvt: np.ndarray | None = None
    r0: np.ndarray | None = None
    anc: np.ndarray | None = None
    c_raw: np.ndarray | None = None
    c_sm: np.ndarray | None = None
    formation_signature: np.ndarray | None = None
    formation_surface_distance: np.ndarray | None = None

    @property
    def z_eval(self) -> np.ndarray:
        return self.z[self.s + 1 :]

    @property
    def suffix_row_idx(self) -> np.ndarray:
        return np.arange(self.s + 1, self.s + 1 + self.n, dtype=int)


@dataclass
class FieldPack:
    f_raw: np.ndarray
    f_sm: np.ndarray
    surface_points: np.ndarray
    global_theta: float
    formation_center: np.ndarray
    formation_scale: np.ndarray


@dataclass
class SupportDiagnostics:
    formation_factor_min: np.ndarray
    formation_factor_max: np.ndarray
    formation_factor_mean: np.ndarray
    formation_distance_squared_mean: np.ndarray
    parent_effective_sample_size: np.ndarray
    new_effective_sample_size: np.ndarray
    new_to_parent_ess_ratio: np.ndarray
    nearest_donor_distance: np.ndarray
    fallback: np.ndarray


@dataclass
class PredictionResult:
    pred: np.ndarray
    geop: np.ndarray
    delta: np.ndarray
    donor_distance: np.ndarray
    design: np.ndarray
    raw_drift: np.ndarray
    smooth_drift: np.ndarray
    support: SupportDiagnostics
    gate_count: int
    summary: dict[str, Any]


def last_known_index(tvt_input: np.ndarray) -> int:
    finite = np.where(np.isfinite(tvt_input))[0]
    if len(finite) == 0:
        raise ValueError("well has no finite TVT_input anchor")
    return int(finite.max())


@dataclass
class FormationPlaneKNN:
    wells: np.ndarray
    xy: np.ndarray
    formation_medians: np.ndarray
    k: int = 10

    def __post_init__(self) -> None:
        self.wells = np.asarray(self.wells, dtype=object)
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.formation_medians = np.asarray(
            self.formation_medians, dtype=np.float64
        )
        if len(self.wells) < self.k + 1:
            raise ValueError("formation plane reference requires at least k+1 wells")
        if self.xy.shape != (len(self.wells), 2):
            raise ValueError("formation plane XY shape mismatch")
        if self.formation_medians.shape != (
            len(self.wells),
            len(FORMATION_COLUMNS),
        ):
            raise ValueError("formation plane median shape mismatch")
        if not (
            np.isfinite(self.xy).all()
            and np.isfinite(self.formation_medians).all()
        ):
            raise ValueError("formation plane reference contains nonfinite values")
        self.scale = np.where(
            self.xy.std(axis=0) < 1.0e-3, 1.0, self.xy.std(axis=0)
        )

    def impute(
        self, xy_query: np.ndarray, *, target_well: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.atleast_2d(np.asarray(xy_query, dtype=np.float64))
        output = np.empty((len(query), len(FORMATION_COLUMNS)), dtype=np.float64)
        minimum_distance = np.empty(len(query), dtype=np.float64)
        own = (
            np.flatnonzero(self.wells.astype(str) == str(target_well))
            if target_well is not None
            else np.empty(0, dtype=np.int64)
        )
        own_index = int(own[0]) if len(own) else None
        scaled_reference = self.xy / self.scale
        for row_index, point in enumerate(query):
            distance = np.sqrt(
                np.square(scaled_reference - point / self.scale).sum(axis=1)
            )
            if own_index is not None:
                distance[own_index] = np.inf
            stable_index = np.arange(len(distance), dtype=np.int64)
            order = np.lexsort((stable_index, self.wells.astype(str), distance))
            selected = order[np.isfinite(distance[order])][: self.k]
            if len(selected) != self.k:
                raise ValueError("formation query lacks k non-self references")
            selected_distance = distance[selected]
            weight = 1.0 / (selected_distance + 1.0e-3)
            xn = self.xy[selected, 0]
            yn = self.xy[selected, 1]
            design = np.column_stack([xn, yn, np.ones(len(selected))])
            normal = (design * weight[:, None]).T @ design
            normal.flat[::4] += 1.0e-9
            rhs = (design * weight[:, None]).T @ self.formation_medians[selected]
            try:
                coefficient = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                coefficient = np.linalg.pinv(normal) @ rhs
            output[row_index] = np.r_[point, 1.0] @ coefficient
            minimum_distance[row_index] = selected_distance.min()
        return output, minimum_distance


def build_formation_plane(
    path_by_well: Mapping[str, Path],
    source_well_ids: Sequence[str],
    *,
    plane_k: int,
) -> tuple[FormationPlaneKNN, dict[str, Any]]:
    reference_wells: list[str] = []
    reference_xy: list[np.ndarray] = []
    formation_medians: list[np.ndarray] = []
    unavailable: list[str] = []
    for well_id in sorted(map(str, source_well_ids)):
        frame = pd.read_csv(
            path_by_well[well_id],
            usecols=["X", "Y", *FORMATION_COLUMNS],
        ).dropna(subset=["X", "Y", *FORMATION_COLUMNS])
        if frame.empty:
            unavailable.append(well_id)
            continue
        xy = frame[["X", "Y"]].median().to_numpy(dtype=np.float64)
        formation = frame[list(FORMATION_COLUMNS)].median().to_numpy(
            dtype=np.float64
        )
        if not (np.isfinite(xy).all() and np.isfinite(formation).all()):
            unavailable.append(well_id)
            continue
        reference_wells.append(well_id)
        reference_xy.append(xy)
        formation_medians.append(formation)
    plane = FormationPlaneKNN(
        wells=np.asarray(reference_wells, dtype=object),
        xy=np.asarray(reference_xy, dtype=np.float64),
        formation_medians=np.asarray(formation_medians, dtype=np.float64),
        k=plane_k,
    )
    evidence = {
        "requested_source_wells": len(set(map(str, source_well_ids))),
        "reference_wells": len(reference_wells),
        "unavailable_reference_wells": unavailable,
        "reference_well_sha256": json_sha256(reference_wells),
        "reference_content_sha256": frame_content_sha256(
            pd.DataFrame(
                {
                    "well": reference_wells,
                    "x": np.asarray(reference_xy)[:, 0],
                    "y": np.asarray(reference_xy)[:, 1],
                    **{
                        formation: np.asarray(formation_medians)[:, index]
                        for index, formation in enumerate(FORMATION_COLUMNS)
                    },
                }
            )
        ),
    }
    return plane, evidence


def formation_signature(
    well: WellData, plane: FormationPlaneKNN
) -> WellData:
    surfaces, distance = plane.impute(well.mid, target_well=well.wid)
    relative_depth = well.z_mid[:, None] - surfaces
    adjacent_thickness = np.diff(surfaces, axis=1)
    signature = np.column_stack([relative_depth, adjacent_thickness])
    if signature.shape != (len(well.mid), 11):
        raise ValueError("formation signature must be K16 x 11")
    return replace(
        well,
        formation_signature=signature,
        formation_surface_distance=distance,
    )


def segment_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    s: int,
    n: int,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, n, params.k_segments + 1)
    step_idx = np.arange(1, n + 1.0)
    segid = np.clip(
        np.searchsorted(edges[1:], step_idx, side="left"),
        0,
        params.k_segments - 1,
    )
    mid = np.empty((params.k_segments, 2), dtype=float)
    z_mid = np.empty(params.k_segments, dtype=float)
    proj = np.empty(params.k_segments, dtype=float)
    az = np.empty(params.k_segments, dtype=float)
    theta = np.radians(params.theta0)
    last_idx = len(x) - 1
    for j in range(params.k_segments):
        f0 = min(s + 1 + int(edges[j]), last_idx)
        f1_raw = s + 1 + max(int(edges[j + 1]) - 1, int(edges[j]))
        f1 = min(max(f1_raw, f0), last_idx)
        az[j] = np.arctan2(y[f1] - y[f0], x[f1] - x[f0])
        mid[j] = ((x[f0] + x[f1]) / 2.0, (y[f0] + y[f1]) / 2.0)
        z_mid[j] = (z[f0] + z[f1]) / 2.0
        proj[j] = np.cos(az[j] - theta)
    return segid.astype(int), mid, z_mid, proj, az


def fit_coeffs(r0: np.ndarray, u: np.ndarray, n: int, params: K16Params, rho: float) -> np.ndarray:
    t = np.arange(1, n + 1.0)
    edges = np.linspace(0, n, params.k_segments + 1)
    phi = np.column_stack(
        [np.clip(t - edges[j], 0, edges[j + 1] - edges[j]) for j in range(params.k_segments)]
    )
    a_mat = phi.T @ phi
    if rho > 0:
        dm = np.diff(np.eye(params.k_segments), axis=0)
        scale = float(np.mean(np.diag(a_mat))) if a_mat.size else 1.0
        a_mat = a_mat + rho * max(scale, 1e-9) * dm.T @ dm
    rhs = phi.T @ (r0 - u)
    try:
        return np.linalg.solve(a_mat, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a_mat + np.eye(params.k_segments) * 1e-9, rhs, rcond=None)[0]


def build_fields(wells: list[WellData], params: K16Params) -> FieldPack:
    def pack(key: str) -> np.ndarray:
        rows: list[tuple[float, ...]] = []
        for well in wells:
            coeffs = getattr(well, key)
            if coeffs is None or well.formation_signature is None:
                continue
            for j in range(params.k_segments):
                if abs(well.proj[j]) > params.field_min_proj:
                    rows.append(
                        (
                            well.mid[j, 0],
                            well.mid[j, 1],
                            coeffs[j] / well.proj[j],
                            well.wi,
                            j,
                            *well.formation_signature[j],
                        )
                    )
        if not rows:
            raise ValueError("empty donor field; no segment passes projection guard")
        return np.asarray(rows, dtype=float)

    surface_parts = []
    for well in wells:
        if well.anc is None:
            continue
        step = max(len(well.x) // 120, 1)
        anc = well.anc[::step]
        part = np.column_stack(
            [
                well.x[::step],
                well.y[::step],
                anc,
                np.full(len(anc), well.wi, dtype=float),
            ]
        )
        surface_parts.append(part)
    if not surface_parts:
        raise ValueError("empty ANCC surface sample set")
    surface_points = np.vstack(surface_parts)
    surface_points = surface_points[np.isfinite(surface_points[:, 2])]
    if len(surface_points) < 3:
        raise ValueError("not enough finite ANCC surface samples")
    xg = np.column_stack(
        [
            np.ones(len(surface_points)),
            surface_points[:, 0] - surface_points[:, 0].mean(),
            surface_points[:, 1] - surface_points[:, 1].mean(),
        ]
    )
    beta = np.linalg.lstsq(xg, surface_points[:, 2], rcond=None)[0]
    global_theta = float(np.arctan2(beta[2], beta[1]))
    f_raw = pack("c_raw")
    f_sm = pack("c_sm")
    donor_signature = f_raw[:, 5:]
    formation_center = np.nanmedian(donor_signature, axis=0)
    formation_mad = np.nanmedian(
        np.abs(donor_signature - formation_center), axis=0
    )
    formation_scale = np.maximum(1.4826 * formation_mad, 1.0)
    if not (
        np.isfinite(formation_center).all()
        and np.isfinite(formation_scale).all()
    ):
        raise ValueError("formation robust center/scale is nonfinite")
    return FieldPack(
        f_raw=f_raw,
        f_sm=f_sm,
        surface_points=surface_points,
        global_theta=global_theta,
        formation_center=formation_center,
        formation_scale=formation_scale,
    )


def _safe_nearest_indices(
    d2: np.ndarray,
    cand: np.ndarray,
    k: int,
    donor_well: np.ndarray | None = None,
    donor_segment: np.ndarray | None = None,
) -> np.ndarray:
    if len(cand) == 0:
        return cand
    kk = min(max(int(k), 1), len(cand))
    well_key = (
        np.asarray(donor_well)[cand]
        if donor_well is not None
        else np.zeros(len(cand), dtype=np.int64)
    )
    segment_key = (
        np.asarray(donor_segment)[cand]
        if donor_segment is not None
        else np.arange(len(cand), dtype=np.int64)
    )
    order = np.lexsort((segment_key, well_key, d2[cand]))
    return cand[order[:kk]]


def _empty_support(rows: int) -> SupportDiagnostics:
    return SupportDiagnostics(
        formation_factor_min=np.ones(rows, dtype=float),
        formation_factor_max=np.ones(rows, dtype=float),
        formation_factor_mean=np.ones(rows, dtype=float),
        formation_distance_squared_mean=np.zeros(rows, dtype=float),
        parent_effective_sample_size=np.zeros(rows, dtype=float),
        new_effective_sample_size=np.zeros(rows, dtype=float),
        new_to_parent_ess_ratio=np.ones(rows, dtype=float),
        nearest_donor_distance=np.full(rows, np.inf, dtype=float),
        fallback=np.ones(rows, dtype=bool),
    )


def local_linear(
    field: np.ndarray,
    own_wi: int,
    mid: np.ndarray,
    query_signature: np.ndarray | None,
    formation_center: np.ndarray,
    formation_scale: np.ndarray,
    params: K16Params,
    min_dist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, SupportDiagnostics]:
    keep = field[:, 3] != own_wi
    fx, fy, fd = field[keep, 0], field[keep, 1], field[keep, 2]
    donor_well = field[keep, 3].astype(np.int64)
    donor_segment = field[keep, 4].astype(np.int64)
    donor_signature = field[keep, 5:]
    dh = np.empty(len(mid), dtype=float)
    dd = np.empty(len(mid), dtype=float)
    support = _empty_support(len(mid))
    if len(fd) == 0:
        dh.fill(0.0)
        dd.fill(float("inf"))
        return dh, dd, support
    for j in range(len(mid)):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        cand = np.where(d2 >= min_dist * min_dist)[0] if min_dist else np.arange(len(d2))
        sel = _safe_nearest_indices(
            d2,
            cand,
            params.local_linear_k,
            donor_well,
            donor_segment,
        )
        if len(sel) == 0:
            dh[j] = float(np.median(fd))
            dd[j] = float("inf")
            continue
        w_xy = np.exp(
            np.maximum(
                -d2[sel] / (2.0 * params.local_linear_bandwidth**2),
                -700,
            )
        )
        fallback = (
            query_signature is None
            or not np.isfinite(query_signature[j]).all()
            or not np.isfinite(donor_signature[sel]).all()
        )
        if fallback:
            formation_distance_squared = np.zeros(len(sel), dtype=float)
            formation_factor = np.ones(len(sel), dtype=float)
        else:
            query_standardized = (
                query_signature[j] - formation_center
            ) / formation_scale
            donor_standardized = (
                donor_signature[sel] - formation_center
            ) / formation_scale
            standardized_delta = np.clip(
                donor_standardized - query_standardized,
                -3.0,
                3.0,
            )
            formation_distance_squared = np.mean(
                np.square(standardized_delta), axis=1
            )
            formation_factor = 0.5 + 0.5 * np.exp(
                -0.5 * formation_distance_squared
            )
        w = w_xy * formation_factor
        parent_ess = float(np.square(w_xy.sum()) / np.square(w_xy).sum())
        new_ess = float(np.square(w.sum()) / np.square(w).sum())
        dx = (fx[sel] - mid[j, 0]) / 1000.0
        dy = (fy[sel] - mid[j, 1]) / 1000.0
        xd = np.column_stack([np.ones(len(sel)), dx, dy])
        ridge = params.local_linear_ridge * np.sum(w) * np.diag([0.0, 1.0, 1.0])
        amat = (xd * w[:, None]).T @ xd + ridge
        rhs = (xd * w[:, None]).T @ fd[sel]
        try:
            dh[j] = np.linalg.solve(amat, rhs)[0]
        except np.linalg.LinAlgError:
            dh[j] = np.linalg.lstsq(amat + np.eye(3) * 1e-9, rhs, rcond=None)[0][0]
        dd[j] = float(np.sqrt(np.median(np.sort(d2[sel])[: min(15, len(sel))])))
        support.formation_factor_min[j] = float(formation_factor.min())
        support.formation_factor_max[j] = float(formation_factor.max())
        support.formation_factor_mean[j] = float(formation_factor.mean())
        support.formation_distance_squared_mean[j] = float(
            formation_distance_squared.mean()
        )
        support.parent_effective_sample_size[j] = parent_ess
        support.new_effective_sample_size[j] = new_ess
        support.new_to_parent_ess_ratio[j] = new_ess / max(parent_ess, 1.0e-12)
        support.nearest_donor_distance[j] = float(np.sqrt(d2[sel].min()))
        support.fallback[j] = fallback
    return dh, dd, support


def kernel_mean(
    field: np.ndarray,
    own_wi: int,
    mid: np.ndarray,
    min_dist: float = 0.0,
) -> np.ndarray:
    keep = field[:, 3] != own_wi
    fx, fy, fd = field[keep, 0], field[keep, 1], field[keep, 2]
    dh = np.empty(len(mid), dtype=float)
    if len(fd) == 0:
        dh.fill(0.0)
        return dh
    for j in range(len(mid)):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        cand = np.where(d2 >= min_dist * min_dist)[0] if min_dist else np.arange(len(d2))
        sel = _safe_nearest_indices(
            d2,
            cand,
            15,
            field[keep, 3].astype(np.int64),
            field[keep, 4].astype(np.int64),
        )
        if len(sel) == 0:
            dh[j] = float(np.median(fd))
            continue
        w = np.exp(np.maximum(-d2[sel] / (2.0 * 500.0**2), -700))
        dh[j] = float(np.sum(w * fd[sel]) / np.sum(w))
    return dh


def theta_loc_at(
    surface_points: np.ndarray,
    mids: np.ndarray,
    own_wi: int,
    global_theta: float,
    params: K16Params,
) -> np.ndarray:
    out = np.empty(len(mids), dtype=float)
    h = params.ancc_theta_bandwidth
    for q, mid in enumerate(mids):
        d2 = (surface_points[:, 0] - mid[0]) ** 2 + (surface_points[:, 1] - mid[1]) ** 2
        mask = (d2 < (4 * h) ** 2) & (surface_points[:, 3] != own_wi)
        if int(mask.sum()) < 30:
            out[q] = global_theta
            continue
        w = np.exp(-d2[mask] / (2 * h * h))
        x = surface_points[mask, 0] - mid[0]
        y = surface_points[mask, 1] - mid[1]
        z = surface_points[mask, 2]
        amat = np.array(
            [
                [np.sum(w), np.sum(w * x), np.sum(w * y)],
                [np.sum(w * x), np.sum(w * x * x), np.sum(w * x * y)],
                [np.sum(w * y), np.sum(w * x * y), np.sum(w * y * y)],
            ]
        )
        rhs = np.array([np.sum(w * z), np.sum(w * x * z), np.sum(w * y * z)])
        try:
            beta = np.linalg.solve(amat, rhs)
            out[q] = np.arctan2(beta[2], beta[1])
        except np.linalg.LinAlgError:
            out[q] = global_theta
    return out


def committee_inputs(
    well: WellData,
    fields: FieldPack,
    params: K16Params,
    min_dist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not (np.abs(well.proj) < params.gate).any():
        return None
    theta = theta_loc_at(
        fields.surface_points,
        well.mid,
        well.wi,
        fields.global_theta,
        params,
    )
    rot = np.degrees(
        np.abs(
            np.arctan2(
                np.sin(theta - np.radians(params.theta0)),
                np.cos(theta - np.radians(params.theta0)),
            )
        )
    )
    dk = kernel_mean(fields.f_raw, well.wi, well.mid, min_dist=min_dist)
    ch_l = dk * np.cos(well.az - theta)
    sub_mask = (np.abs(well.proj[well.segid]) < params.gate) & (rot < params.rot_max_deg)[
        well.segid
    ]
    return ch_l, sub_mask


def build_columns(
    ndz: np.ndarray,
    segid: np.ndarray,
    proj: np.ndarray,
    ch_raw: np.ndarray,
    ch_sm: np.ndarray,
    donor_dist: np.ndarray,
    params: K16Params,
    sub: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    gated = np.abs(proj[segid]) < params.gate
    raw_step = np.where(gated, 0.0, ndz + ch_raw[segid])
    sm_step = np.where(gated, 0.0, ndz + ch_sm[segid])
    bucket = np.digitize(donor_dist, params.kbins[1:-1])[segid]
    pos = (segid + 0.5) / params.k_segments
    cols = [np.cumsum(np.where(bucket == b, raw_step, 0.0)) for b in range(params.n_bins)]
    cols += [np.cumsum(np.where(bucket == b, sm_step, 0.0)) for b in range(params.n_bins)]
    cols.append(np.cumsum(0.5 * (raw_step + sm_step) * np.sqrt(pos)))
    if sub is None:
        cols.append(np.zeros(len(ndz), dtype=float))
    else:
        cols.append(np.cumsum(np.where(sub[1], ndz + sub[0][segid], 0.0)))
    return np.column_stack(cols)


def well_design(
    well: WellData,
    fields: FieldPack,
    params: K16Params,
    min_dist: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    SupportDiagnostics,
]:
    raw_field, donor_dist, support = local_linear(
        fields.f_raw,
        well.wi,
        well.mid,
        well.formation_signature,
        fields.formation_center,
        fields.formation_scale,
        params,
        min_dist,
    )
    smooth_field, _, _ = local_linear(
        fields.f_sm,
        well.wi,
        well.mid,
        well.formation_signature,
        fields.formation_center,
        fields.formation_scale,
        params,
        min_dist,
    )
    sub = committee_inputs(well, fields, params, min_dist=min_dist)
    design = build_columns(
        well.ndz,
        well.segid,
        well.proj,
        raw_field * well.proj,
        smooth_field * well.proj,
        donor_dist,
        params,
        sub,
    )
    return design, raw_field, smooth_field, donor_dist, support


def fit_kappa(
    wells: list[WellData],
    fields: FieldPack,
    params: K16Params,
) -> np.ndarray:
    amat = np.zeros((params.kappa_dim, params.kappa_dim), dtype=float)
    rhs = np.zeros(params.kappa_dim, dtype=float)
    for regime in params.kappa_regimes:
        for well in wells:
            if well.r0 is None:
                continue
            design, _, _, _, _ = well_design(
                well, fields, params, min_dist=regime
            )
            amat += design.T @ design
            rhs += design.T @ well.r0
    return np.linalg.lstsq(amat, rhs, rcond=None)[0]


def affine_cal(gr_known: np.ndarray, tw_at_known: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(gr_known) & np.isfinite(tw_at_known)
    if int(mask.sum()) < 30:
        return 1.0, 0.0
    x = gr_known[mask]
    y = tw_at_known[mask]
    for _ in range(2):
        coef = np.polyfit(x, y, 1)
        resid = y - np.polyval(coef, x)
        keep = np.abs(resid) < 2.5 * (np.std(resid) + 1e-9)
        if int(keep.sum()) < 20:
            break
        x, y = x[keep], y[keep]
    a, b = np.polyfit(x, y, 1)
    if not (0.2 < a < 5.0):
        return 1.0, float(np.median(y - x))
    return float(a), float(b)


def emissions(
    grid: np.ndarray,
    typewell_gr: np.ndarray,
    lateral_gr: np.ndarray,
    relpath: np.ndarray,
    n: int,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    centers = list(range(params.gr_window // 2, n - params.gr_window // 2, params.gr_stride))
    if not centers:
        return None
    nz = len(grid)
    corr = np.full((len(centers), nz), np.nan)
    mse = np.full((len(centers), nz), np.nan)
    lvl = np.full((len(centers), nz), np.nan)
    for k, center in enumerate(centers):
        sl = slice(center - params.gr_window // 2, center + params.gr_window // 2)
        gw = lateral_gr[sl]
        finite = np.isfinite(gw)
        if float(finite.mean()) < 0.5:
            continue
        pw = relpath[sl] - relpath[sl].mean()
        if pw.max() - pw.min() < 4:
            continue
        rel_bins = np.arange(pw.min(), pw.max() + params.gr_grid_step, params.gr_grid_step)
        rel_idx = np.clip(((pw - pw.min()) / params.gr_grid_step).astype(int), 0, len(rel_bins) - 1)
        profile = np.full(len(rel_bins), np.nan)
        for bidx in range(len(rel_bins)):
            values = gw[(rel_idx == bidx) & finite]
            if len(values) >= 3:
                profile[bidx] = np.mean(values)
        ok = np.isfinite(profile)
        maxoff = nz - len(rel_bins)
        if int(ok.sum()) < 7 or maxoff < 6:
            continue
        pv = profile[ok]
        pz = (pv - pv.mean()) / (pv.std() + 1e-9)
        base = np.arange(len(rel_bins))
        for off in range(maxoff):
            tv = typewell_gr[base + off][ok]
            if np.isnan(tv).any():
                continue
            zc = int(round(off - pw.min() / params.gr_grid_step))
            if 0 <= zc < nz:
                corr[k, zc] = np.mean(pz * (tv - tv.mean()) / (tv.std() + 1e-9))
                mse[k, zc] = np.mean((pv - tv) ** 2)
                lvl[k, zc] = (pv.mean() - tv.mean()) ** 2
    return np.asarray(centers), corr, mse, lvl


def decode(
    grid: np.ndarray,
    centers: np.ndarray,
    loglike: np.ndarray,
    geop: np.ndarray,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray]:
    nz = len(grid)
    nk = len(centers)
    gm = np.array(
        [
            np.nanmean(geop[max(0, c - params.gr_window // 2) : c + params.gr_window // 2])
            for c in centers
        ]
    )

    def prop(prob: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        shifted = np.interp(grid, grid + mu, prob, left=0, right=0)
        radius = max(int(4 * sigma / params.gr_grid_step), 1)
        offsets = np.arange(-radius, radius + 1) * params.gr_grid_step
        kern = np.exp(-0.5 * (offsets / sigma) ** 2)
        out = np.convolve(shifted, kern / kern.sum(), "same")
        total = out.sum()
        return out / total if total > 0 else np.ones(nz) / nz

    alpha = np.zeros((nk, nz), dtype=float)
    p0 = np.exp(
        -0.5
        * ((grid - gm[0]) / max(params.gr_s0 * np.sqrt(max(centers[0], 1)), params.gr_grid_step))
        ** 2
    )
    p0 /= p0.sum()
    a0 = p0 * np.exp(loglike[0] - loglike[0].max())
    alpha[0] = a0 / max(a0.sum(), 1e-300)
    for k in range(1, nk):
        sigma = max(
            params.gr_s0 * np.sqrt(centers[k] - centers[k - 1]),
            params.gr_grid_step / 2.0,
        )
        prior = prop(alpha[k - 1], gm[k] - gm[k - 1], sigma)
        ak = prior * np.exp(loglike[k] - loglike[k].max())
        alpha[k] = ak / max(ak.sum(), 1e-300)
    beta = np.zeros((nk, nz), dtype=float)
    beta[-1] = np.ones(nz) / nz
    for k in range(nk - 2, -1, -1):
        sigma = max(
            params.gr_s0 * np.sqrt(centers[k + 1] - centers[k]),
            params.gr_grid_step / 2.0,
        )
        bk = beta[k + 1] * np.exp(loglike[k + 1] - loglike[k + 1].max())
        beta[k] = prop(bk[::-1], -(gm[k + 1] - gm[k]), sigma)[::-1]
        beta[k] /= max(beta[k].sum(), 1e-300)
    posterior = alpha * beta
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)
    mu = posterior @ grid
    sd = np.sqrt(np.maximum((posterior @ grid**2) - mu**2, 0.0))
    return mu - gm, sd


def gr_correction(
    typewell_frame: pd.DataFrame,
    tvt_known: np.ndarray,
    gr_known: np.ndarray,
    gr_lateral: np.ndarray,
    geop: np.ndarray,
    relpath: np.ndarray,
    n: int,
    params: K16Params,
) -> np.ndarray:
    try:
        tws = typewell_frame.sort_values("TVT")
        tt = tws["TVT"].to_numpy(float)
        tg = tws["GR"].to_numpy(float)
        if len(tt) < 10 or not np.isfinite(tg).any():
            return np.zeros(n, dtype=float)
        lo = np.nanmin(geop) - params.gr_extent
        hi = np.nanmax(geop) + params.gr_extent
        grid = np.arange(lo, hi + params.gr_grid_step, params.gr_grid_step)
        tw = np.interp(grid, tt, tg, left=np.nan, right=np.nan)
        tw_known = np.interp(np.asarray(tvt_known, float), tt, tg, left=np.nan, right=np.nan)
        a, b = affine_cal(np.asarray(gr_known, float), tw_known)
        lateral = pd.Series(np.asarray(gr_lateral, float)).interpolate(limit=10).to_numpy()
        lateral = lateral * a + b
        known = np.asarray(gr_known, float) * a + b
        mask = np.isfinite(known) & np.isfinite(tw_known)
        sigma = float(np.clip(np.std(known[mask] - tw_known[mask]), 5.0, 60.0))
        if int(mask.sum()) <= 30:
            sigma = 30.0
        emission = emissions(grid, tw, lateral, relpath, n, params)
        if emission is None:
            return np.zeros(n, dtype=float)
        centers, corr, mse, lvl = emission
        loglike = np.where(
            np.isfinite(corr), params.gr_tau * np.arctanh(np.clip(corr, -0.95, 0.95)), 0.0
        )
        loglike = loglike + np.where(np.isfinite(mse), -params.gr_w_mse * mse / (2 * sigma**2), 0.0)
        loglike = loglike + np.where(
            np.isfinite(lvl),
            -params.gr_w_lvl * lvl / (2 * sigma**2 / 8.0),
            0.0,
        )
        delta, sd = decode(grid, centers, loglike, geop, params)
        delta = delta * np.clip(params.gr_sh_a - params.gr_sh_b * sd, params.gr_sh_lo, 1.0)
        x = np.r_[0.0, centers, n]
        y = np.r_[0.0, delta, delta[-1]]
        out = np.interp(np.arange(1, n + 1.0), x, y)
        return np.clip(out, -params.gr_cap, params.gr_cap)
    except Exception:
        return np.zeros(n, dtype=float)


def project_u(pred: np.ndarray, z_eval: np.ndarray, params: K16Params) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if len(pred) <= params.u_projection_deg + 2:
        return pred
    u = pred + np.asarray(z_eval, dtype=float)
    x = np.linspace(-1, 1, len(u))
    weight = np.ones(len(u), dtype=float)
    coef = np.polyfit(x, u, params.u_projection_deg, w=weight)
    for _ in range(params.u_projection_iters):
        coef = np.polyfit(x, u, params.u_projection_deg, w=weight)
        resid = u - np.polyval(coef, x)
        scale = np.median(np.abs(resid)) * 1.4826 + 1e-9
        weight = 1.0 / (1.0 + (resid / (2.5 * scale)) ** 2)
    projected = np.polyval(coef, x) - z_eval
    return (1.0 - params.u_projection_beta) * pred + params.u_projection_beta * projected


def predict_well(
    well: WellData,
    fields: FieldPack,
    kappa: np.ndarray,
    params: K16Params,
) -> PredictionResult:
    design, raw_field, smooth_field, donor_dist, support = well_design(
        well, fields, params
    )
    geop = well.anchor + design @ kappa
    raw_step = (raw_field * well.proj)[well.segid]
    relpath = np.cumsum(well.ndz + raw_step)
    delta = np.zeros(well.n, dtype=float)
    if params.enable_gr_correction and well.gr is not None and well.typewell_path.exists():
        typewell_frame = pd.read_csv(well.typewell_path)
        delta = gr_correction(
            typewell_frame,
            well.ti[: well.s + 1],
            well.gr[: well.s + 1],
            well.gr[well.s + 1 :],
            geop,
            relpath,
            well.n,
            params,
        )
    pred = geop + delta
    if params.enable_u_projection:
        pred = project_u(pred, well.z_eval, params)
    summary = {
        "well_id": well.wid,
        "unknown_rows": int(well.n),
        "last_known_row": int(well.s),
        "anchor": float(well.anchor),
        "gate_segments": int((np.abs(well.proj) < params.gate).sum()),
        "donor_dist_min": float(np.nanmin(donor_dist)),
        "donor_dist_max": float(np.nanmax(donor_dist)),
        "delta_abs_median": float(np.median(np.abs(delta))),
        "delta_abs_max": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "end_minus_anchor": float((design @ kappa)[-1]) if len(design) else 0.0,
    }
    return PredictionResult(
        pred=pred,
        geop=geop,
        delta=delta,
        donor_distance=donor_dist,
        design=design,
        raw_drift=raw_field,
        smooth_drift=smooth_field,
        support=support,
        gate_count=summary["gate_segments"],
        summary=summary,
    )


def kappa_terms(kappa: np.ndarray, params: K16Params, fold: str = "full") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(kappa):
        if idx < params.n_bins:
            term = f"raw_bin_{idx}"
        elif idx < 2 * params.n_bins:
            term = f"smooth_bin_{idx - params.n_bins}"
        elif idx == 2 * params.n_bins:
            term = "sqrt_position"
        else:
            term = "near_strike_committee"
        rows.append({"fold": fold, "term": term, "value": float(value)})
    return rows


def assign_group_folds(wells: list[WellData], n_folds: int, seed: int) -> dict[str, int]:
    n_folds = max(1, min(int(n_folds), len(wells)))

    def key(well: WellData) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{well.wid}".encode()).hexdigest()
        return digest, well.wid

    ordered = sorted(wells, key=key)
    return {well.wid: idx % n_folds for idx, well in enumerate(ordered)}


def params_for_variant(config: Mapping[str, Any], variant: str) -> K16Params:
    variants = {
        str(item["name"]): int(item["k_segments"])
        for item in get_nested(config, "model.params.variants")
    }
    if variants != EXPECTED_VARIANTS:
        raise ValueError(f"scientific variants changed: {variants}")
    fixed = dict(get_nested(config, "model.params.fixed_from_exp226"))
    fixed["k_segments"] = variants[variant]
    tuple_keys = {"kbins", "kappa_regimes"}
    for key in tuple_keys:
        fixed[key] = tuple(float(value) for value in fixed[key])
    params = K16Params(**fixed)
    expected = K16Params()
    if params != expected:
        raise ValueError(
            f"{variant} changes a fixed exp226 parameter: "
            f"{params} != {expected}"
        )
    donor_weight = get_nested(config, "model.donor_weight")
    expected_weight = {
        "formula": (
            "w_new = w_xy * (0.5 + 0.5 * exp(-0.5 * d_form_squared))"
        ),
        "nonfinite_fallback": "parent_xy_weight",
    }
    for key, value in expected_weight.items():
        if donor_weight.get(key) != value:
            raise ValueError(f"formation donor-weight contract changed: {key}")
    return params


def resolve_raw_train_dir(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[Path, list[Path]]:
    patterns = list(get_nested(config, "data.raw_train_dir_patterns"))
    horizontal_glob = str(get_nested(config, "data.raw_horizontal_glob"))
    candidates = [path for path in expand_existing_paths(patterns) if path.is_dir()]
    evidence: list[tuple[Path, list[Path], set[str]]] = []
    for directory in candidates:
        files = sorted(directory.glob(horizontal_glob))
        wells = {
            path.name.split("__horizontal_well.csv", 1)[0] for path in files
        }
        evidence.append((directory, files, wells))
        if wells == expected_wells and len(files) == len(expected_wells):
            return directory, files
    detail = {
        str(directory): {"files": len(files), "wells": len(wells)}
        for directory, files, wells in evidence
    }
    raise FileNotFoundError(
        "raw train directory with exact candidate-well inventory was not found: "
        f"{detail}"
    )


def build_raw_input_manifest(raw_dir: Path, horizontal_files: Sequence[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for horizontal_path in horizontal_files:
        well = horizontal_path.name.split("__horizontal_well.csv", 1)[0]
        typewell_path = horizontal_path.with_name(f"{well}__typewell.csv")
        for role, path in (
            ("horizontal", horizontal_path),
            ("typewell", typewell_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"raw {role} file missing: {path}")
            records.append(
                {
                    "well": well,
                    "role": role,
                    "relative_path": str(path.relative_to(raw_dir)),
                    "bytes": path.stat().st_size,
                    "file_sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["well", "role"], kind="stable"
    ).reset_index(drop=True)


def load_target_free_wells(
    horizontal_files: Sequence[Path], params: K16Params
) -> list[WellData]:
    wells: list[WellData] = []
    columns = ["X", "Y", "Z", "TVT_input", "GR"]
    for wi, path in enumerate(horizontal_files):
        frame = pd.read_csv(path, usecols=columns)
        well = path.name.split("__horizontal_well.csv", 1)[0]
        x = frame["X"].to_numpy(dtype=float)
        y = frame["Y"].to_numpy(dtype=float)
        z = frame["Z"].to_numpy(dtype=float)
        visible = frame["TVT_input"].to_numpy(dtype=float)
        s = last_known_index(visible)
        ndz = -np.diff(z)[s:]
        n = len(ndz)
        if n <= 0:
            raise ValueError(f"{well} has no prediction suffix")
        segid, mid, z_mid, proj, az = segment_geometry(
            x, y, z, s, n, params
        )
        wells.append(
            WellData(
                wid=well,
                wi=wi,
                s=s,
                n=n,
                ndz=ndz,
                anchor=float(visible[s]),
                ti=visible,
                segid=segid,
                mid=mid,
                z_mid=z_mid,
                proj=proj,
                az=az,
                x=x,
                y=y,
                z=z,
                gr=frame["GR"].to_numpy(dtype=float),
                typewell_path=path.with_name(f"{well}__typewell.csv"),
                tvt=None,
                r0=None,
                anc=None,
                c_raw=None,
                c_sm=None,
            )
        )
    return wells


def attach_donor_fit_truth(
    well: WellData, horizontal_path: Path, params: K16Params
) -> WellData:
    frame = pd.read_csv(horizontal_path, usecols=["TVT", "ANCC"])
    tvt = pd.to_numeric(frame["TVT"], errors="raise").to_numpy(dtype=float)
    if len(tvt) != len(well.z) or not np.isfinite(tvt).all():
        raise ValueError(f"invalid donor-fit TVT: {horizontal_path}")
    if not math.isclose(
        float(tvt[well.s]), float(well.anchor), abs_tol=1e-8, rel_tol=0.0
    ):
        raise ValueError(f"donor known-anchor parity failed: {well.wid}")
    r0 = tvt[well.s + 1 :] - tvt[well.s]
    u = np.cumsum(well.ndz)
    return replace(
        well,
        tvt=tvt,
        r0=r0,
        anc=pd.to_numeric(frame["ANCC"], errors="coerce").to_numpy(dtype=float),
        c_raw=fit_coeffs(r0, u, well.n, params, rho=0.0),
        c_sm=fit_coeffs(r0, u, well.n, params, rho=params.smooth_rho),
    )


@dataclass
class VariantPrediction:
    variant: str
    k_segments: int
    values: np.ndarray
    frame: pd.DataFrame
    kappa_by_fold: pd.DataFrame
    prediction_content_sha256: str
    donor_fit_truth_wells: int
    fold_run_count: int
    formation_support: pd.DataFrame
    reference_manifest: pd.DataFrame


def assert_outer_fold_separation(
    source_wells: Sequence[WellData],
    valid_wells: Sequence[WellData],
    fold: int,
) -> None:
    source_ids = {well.wid for well in source_wells}
    valid_ids = {well.wid for well in valid_wells}
    overlap = source_ids & valid_ids
    if overlap:
        raise ValueError(f"outer fold {fold} source/valid overlap: {sorted(overlap)[:5]}")
    for well in valid_wells:
        if any(
            value is not None
            for value in (
                well.tvt,
                well.r0,
                well.anc,
                well.c_raw,
                well.c_sm,
            )
        ):
            raise ValueError(
                f"outer-valid truth/formation state is attached: {well.wid}"
            )
    for well in source_wells:
        if any(
            value is None for value in (well.tvt, well.r0, well.c_raw, well.c_sm)
        ):
            raise ValueError(f"donor-fit truth state is missing: {well.wid}")


def build_variant_oof(
    variant: str,
    config: Mapping[str, Any],
    bank: CandidateBank,
    horizontal_files: Sequence[Path],
    expected_fold_by_well: Mapping[str, int],
) -> VariantPrediction:
    params = params_for_variant(config, variant)
    base_wells = load_target_free_wells(horizontal_files, params)
    path_by_well = {
        path.name.split("__horizontal_well.csv", 1)[0]: path
        for path in horizontal_files
    }
    donor_wells = {
        well.wid: attach_donor_fit_truth(well, path_by_well[well.wid], params)
        for well in base_wells
    }
    n_folds = int(get_nested(config, "validation.n_folds"))
    seed = int(get_nested(config, "validation.seed"))
    fold_by_well = assign_group_folds(base_wells, n_folds, seed)
    canonical_fold = {
        str(well): int(fold)
        for well, fold in expected_fold_by_well.items()
    }
    if fold_by_well != canonical_fold:
        mismatches = {
            well: (fold_by_well.get(well), canonical_fold.get(well))
            for well in sorted(set(fold_by_well) | set(canonical_fold))
            if fold_by_well.get(well) != canonical_fold.get(well)
        }
        raise ValueError(f"exp226/exp263 fold identity mismatch: {list(mismatches.items())[:5]}")

    prediction_parts: list[pd.DataFrame] = []
    kappa_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    plane_k = int(get_nested(config, "model.formation_imputer.plane_k"))
    for fold in range(n_folds):
        source_without_signature = [
            donor_wells[well.wid]
            for well in base_wells
            if fold_by_well[well.wid] != fold
        ]
        valid_without_signature = [
            well for well in base_wells if fold_by_well[well.wid] == fold
        ]
        assert_outer_fold_separation(
            source_without_signature, valid_without_signature, fold
        )
        source_ids = [well.wid for well in source_without_signature]
        valid_ids = [well.wid for well in valid_without_signature]
        plane, reference_evidence = build_formation_plane(
            path_by_well,
            source_ids,
            plane_k=plane_k,
        )
        source_wells = [
            formation_signature(well, plane)
            for well in source_without_signature
        ]
        valid_wells = [
            formation_signature(well, plane)
            for well in valid_without_signature
        ]
        source_set = set(source_ids)
        valid_set = set(valid_ids)
        reference_set = set(map(str, plane.wells))
        reference_rows.append(
            {
                "outer_fold": fold,
                "source_wells": len(source_set),
                "valid_wells": len(valid_set),
                "formation_reference_wells": len(reference_set),
                "source_valid_overlap": len(source_set & valid_set),
                "valid_reference_overlap": len(valid_set & reference_set),
                "validation_reference_count": len(valid_set & reference_set),
                "validation_truth_read_count": 0,
                "validation_formation_read_count": 0,
                "donor_signature_self_exclusion": True,
                "target_formation_columns_read": False,
                "source_well_sha256": json_sha256(sorted(source_set)),
                "valid_well_sha256": json_sha256(sorted(valid_set)),
                **reference_evidence,
            }
        )
        print(
            f"{variant} fold {fold + 1}/{n_folds}: "
            f"source={len(source_wells)} valid={len(valid_wells)}"
        )
        fields = build_fields(source_wells, params)
        kappa = fit_kappa(source_wells, fields, params)
        kappa_rows.extend(kappa_terms(kappa, params, fold=f"fold{fold}"))
        for well in valid_wells:
            result = predict_well(well, fields, kappa, params)
            row_idx = well.suffix_row_idx.astype(np.int32)
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "id": [
                            f"{well.wid}_{int(index)}" for index in row_idx
                        ],
                        "well": well.wid,
                        "well_row_idx": row_idx,
                        "outer_fold": fold,
                        "variant": variant,
                        "candidate_tvt": result.pred.astype(np.float64),
                    }
                )
            )
            for segment_id in range(params.k_segments):
                support_rows.append(
                    {
                        "outer_fold": fold,
                        "well": well.wid,
                        "segment_id": segment_id,
                        "mid_x": float(well.mid[segment_id, 0]),
                        "mid_y": float(well.mid[segment_id, 1]),
                        "mid_z": float(well.z_mid[segment_id]),
                        "signature_finite": bool(
                            np.isfinite(
                                well.formation_signature[segment_id]
                            ).all()
                        ),
                        "formation_surface_distance": float(
                            well.formation_surface_distance[segment_id]
                        ),
                        "formation_factor_min": float(
                            result.support.formation_factor_min[segment_id]
                        ),
                        "formation_factor_max": float(
                            result.support.formation_factor_max[segment_id]
                        ),
                        "formation_factor_mean": float(
                            result.support.formation_factor_mean[segment_id]
                        ),
                        "formation_distance_squared_mean": float(
                            result.support.formation_distance_squared_mean[
                                segment_id
                            ]
                        ),
                        "parent_effective_sample_size": float(
                            result.support.parent_effective_sample_size[
                                segment_id
                            ]
                        ),
                        "new_effective_sample_size": float(
                            result.support.new_effective_sample_size[segment_id]
                        ),
                        "new_to_parent_ess_ratio": float(
                            result.support.new_to_parent_ess_ratio[segment_id]
                        ),
                        "nearest_donor_distance": float(
                            result.support.nearest_donor_distance[segment_id]
                        ),
                        "fallback": bool(
                            result.support.fallback[segment_id]
                        ),
                        "fallback_reason": (
                            "nonfinite_signature"
                            if result.support.fallback[segment_id]
                            else ""
                        ),
                        "formation_center_sha256": json_sha256(
                            fields.formation_center
                        ),
                        "formation_scale_sha256": json_sha256(
                            fields.formation_scale
                        ),
                    }
                )
    generated = pd.concat(prediction_parts, ignore_index=True)
    if generated["id"].duplicated().any():
        raise ValueError(f"{variant} generated duplicate IDs")
    indexer = pd.Index(generated["id"].astype(str)).get_indexer(
        bank.keys["id"].astype(str)
    )
    if np.any(indexer < 0) or len(generated) != len(bank.keys):
        raise ValueError(f"{variant} identity coverage mismatch")
    aligned = generated.iloc[indexer].reset_index(drop=True)
    for column in ("id", "well", "well_row_idx"):
        if not np.array_equal(
            aligned[column].to_numpy(), bank.keys[column].to_numpy()
        ):
            raise ValueError(f"{variant} aligned key mismatch: {column}")
    expected_row_folds = aligned["well"].map(canonical_fold).to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(
        aligned["outer_fold"].to_numpy(dtype=np.int64), expected_row_folds
    ):
        raise ValueError(f"{variant} exp226 fold identity mismatch")
    values = aligned["candidate_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{variant} contains nonfinite predictions")
    content_sha = frame_content_sha256(
        aligned,
        ["id", "well", "well_row_idx", "outer_fold", "variant", "candidate_tvt"],
    )
    return VariantPrediction(
        variant=variant,
        k_segments=params.k_segments,
        values=values,
        frame=aligned,
        kappa_by_fold=pd.DataFrame(kappa_rows),
        prediction_content_sha256=content_sha,
        donor_fit_truth_wells=len(donor_wells),
        fold_run_count=n_folds,
        formation_support=pd.DataFrame(support_rows).sort_values(
            ["outer_fold", "well", "segment_id"], kind="stable"
        ).reset_index(drop=True),
        reference_manifest=pd.DataFrame(reference_rows).sort_values(
            "outer_fold", kind="stable"
        ).reset_index(drop=True),
    )


# %% [markdown]
# ## 4. Exp293 fixed candidate-bank and block helpers
#
# The bank is reconstructed from the exp263 manifest with the exact float32
# formula order used by exp293. Its content SHA and the decompressed block
# assignment SHA must equal the completed exp293 run before the new K16 path can be
# evaluated as add-one candidates.

# %%
@dataclass
class CandidateBank:
    keys: pd.DataFrame
    candidate_ids: tuple[str, ...]
    values: np.memmap
    values_path: Path
    primitive_ids: tuple[str, ...]
    manifest: dict[str, Any]
    manifest_path: Path
    key_content_sha256: str
    candidate_content_sha256: str
    coverage_by_candidate: dict[str, float]
    sample_parity: pd.DataFrame
    input_evidence: list[dict[str, Any]]


def _artifact_path_from_manifest(
    manifest_path: Path, item: Mapping[str, Any]
) -> Path:
    raw = str(item["path"])
    marker = "/artifacts/"
    if marker in raw:
        relative = raw.split(marker, 1)[1]
        candidate = manifest_path.parent / relative
        if candidate.exists():
            return candidate
    direct = Path(raw)
    if direct.exists():
        return direct
    suffix = Path(raw).parts[-4:]
    for root in manifest_path.parents:
        candidate = root.joinpath(*suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest partition is missing: {raw}")


def _read_manifest_partitions(
    manifest_path: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for item in items:
        path = _artifact_path_from_manifest(manifest_path, item)
        actual_file_sha = sha256_file(path)
        expected_file_sha = str(item.get("file_sha256", ""))
        if expected_file_sha and actual_file_sha != expected_file_sha:
            raise ValueError(f"{label} partition file SHA mismatch: {path}")
        full = pd.read_parquet(path)
        reject_forbidden_candidate_columns(full.columns)
        expected_rows = int(item.get("rows", len(full)))
        if len(full) != expected_rows:
            raise ValueError(f"{label} partition row mismatch: {path}")
        expected_schema = str(item.get("schema_sha256", ""))
        actual_schema = frame_schema_sha256(full)
        if expected_schema and actual_schema != expected_schema:
            raise ValueError(f"{label} partition schema SHA mismatch: {path}")
        expected_content = str(item.get("content_sha256", ""))
        actual_content = frame_content_sha256(full)
        if expected_content and actual_content != expected_content:
            raise ValueError(f"{label} partition content SHA mismatch: {path}")
        missing = set(columns) - set(full.columns)
        if missing:
            raise ValueError(f"{label} partition columns missing: {sorted(missing)}")
        frames.append(full[list(columns)].copy())
        evidence.append(
            {
                "phase": "target_free",
                "source": label,
                "path": str(path),
                "rows": len(full),
                "file_sha256": actual_file_sha,
                "decompressed_content_sha256": None,
                "logical_content_sha256": actual_content,
                "schema_sha256": actual_schema,
            }
        )
    if not frames:
        raise ValueError(f"no partitions declared for {label}")
    return pd.concat(frames, ignore_index=True), evidence


def _assert_same_keys(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} key row count mismatch")
    for column in VALUE_KEY_COLUMNS:
        left = reference[column].to_numpy()
        right = candidate[column].to_numpy()
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if column == "md_since"
            else np.array_equal(left, right)
        )
        if not equal:
            raise ValueError(f"{label} key mismatch in {column}")


def candidate_bank_content_sha256(
    bank: CandidateBank, chunk_rows: int = 100_000
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(bank.candidate_ids), separators=(",", ":")).encode())
    digest.update(bank.key_content_sha256.encode())
    for position, candidate_id in enumerate(bank.candidate_ids):
        digest.update(candidate_id.encode())
        for start in range(0, len(bank.keys), chunk_rows):
            end = min(start + chunk_rows, len(bank.keys))
            values = np.asarray(bank.values[start:end, position], dtype="<f4")
            digest.update(values.tobytes())
    return digest.hexdigest()


def _materialize_formulas(
    values: np.memmap,
    column_by_candidate: Mapping[str, int],
    config: Mapping[str, Any],
) -> None:
    pairs = get_nested(config, "candidate_bank.pairs")
    for candidate_id, weights in pairs.items():
        parents = list(weights)
        if len(parents) != 2 or any(
            not math.isclose(float(weights[parent]), 0.5) for parent in parents
        ):
            raise ValueError(f"{candidate_id} differs from fixed 50/50 contract")
        left = values[:, column_by_candidate[parents[0]]]
        right = values[:, column_by_candidate[parents[1]]]
        values[:, column_by_candidate[str(candidate_id)]] = (
            np.float32(0.5) * (left + right)
        ).astype(np.float32)

    fixed = get_nested(config, "candidate_bank.fixed_formula")
    if list(fixed) != ["exp226_w500_50_50"]:
        raise ValueError("fixed formula identity differs from exp293 contract")
    weights = fixed["exp226_w500_50_50"]
    expected = {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25}
    if weights != expected:
        raise ValueError("exp226_w500_50_50 weights differ from fixed contract")
    output = (
        np.float32(0.5) * values[:, column_by_candidate["exp226_k16"]]
        + np.float32(0.25) * values[:, column_by_candidate["likpf_mean"]]
        + np.float32(0.25) * values[:, column_by_candidate["exact_hmm"]]
    ).astype(np.float32)
    values[:, column_by_candidate["exp226_w500_50_50"]] = output
    values.flush()


def _build_sample_parity(
    bank: CandidateBank, parity_path: Path, tolerance: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample = pd.read_parquet(parity_path)
    if "id" not in sample:
        raise ValueError("exp263 small parity sample lacks id")
    missing = set(FORMULA_CANDIDATES) - set(sample.columns)
    if missing:
        raise ValueError(f"exp263 small parity formulas missing: {sorted(missing)}")
    indexer = pd.Index(bank.keys["id"].astype(str)).get_indexer(
        sample["id"].astype(str)
    )
    if np.any(indexer < 0):
        raise ValueError("exp263 small parity IDs are absent from candidate bank")
    position = {name: idx for idx, name in enumerate(bank.candidate_ids)}
    records: list[dict[str, Any]] = []
    for candidate_id in FORMULA_CANDIDATES:
        actual = np.asarray(bank.values[indexer, position[candidate_id]], dtype=np.float64)
        expected = pd.to_numeric(sample[candidate_id], errors="raise").to_numpy(
            dtype=np.float64
        )
        max_abs = float(np.max(np.abs(actual - expected), initial=0.0))
        records.append(
            {
                "check_type": "exp263_small_parity_max_abs_ft",
                "candidate_id": candidate_id,
                "actual": max_abs,
                "expected": 0.0,
                "absolute_difference": max_abs,
                "tolerance": tolerance,
                "passed": bool(max_abs <= tolerance),
            }
        )
    evidence = {
        "phase": "target_free",
        "source": "exp263_small_parity_sample",
        "path": str(parity_path),
        "rows": len(sample),
        "file_sha256": sha256_file(parity_path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(sample),
        "schema_sha256": frame_schema_sha256(sample),
    }
    return pd.DataFrame(records), evidence


def build_candidate_bank(
    config: Mapping[str, Any], work_dir: Path
) -> CandidateBank:
    manifest_cfg = get_nested(config, "data.exp263_manifest")
    manifest_path = resolve_file(
        manifest_cfg["patterns"],
        label="exp263 cache manifest",
        expected_sha256=str(manifest_cfg["expected_file_sha256"]),
    )
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = int(get_nested(config, "validation.n_folds"))
    if int(manifest.get("rows", -1)) != expected_rows:
        raise ValueError("exp263 manifest row contract mismatch")
    if int(manifest.get("wells", -1)) != expected_wells:
        raise ValueError("exp263 manifest well contract mismatch")
    if int(manifest.get("folds", -1)) != expected_folds:
        raise ValueError("exp263 manifest fold contract mismatch")
    if manifest.get("canonical_id_sha256") != manifest_cfg[
        "expected_canonical_id_sha256"
    ]:
        raise ValueError("exp263 canonical ID SHA mismatch")

    candidate_ids = tuple(get_nested(config, "candidate_bank.order"))
    if candidate_ids != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("candidate order differs from exp293 fixed contract")
    if len(candidate_ids) != int(get_nested(config, "candidate_bank.expected_count")):
        raise ValueError("candidate count differs from exp293 fixed contract")
    primitive_ids = tuple(get_nested(config, "candidate_bank.primitives"))
    if primitive_ids != EXPECTED_CANDIDATE_ORDER[:6]:
        raise ValueError("primitive order differs from exp293 fixed contract")

    values_path = work_dir / f"{OUTPUT_PREFIX}_candidate_bank.f32"
    values = np.memmap(
        values_path,
        mode="w+",
        dtype="float32",
        shape=(expected_rows, len(candidate_ids)),
    )
    values[:] = np.nan
    column_by_candidate = {name: idx for idx, name in enumerate(candidate_ids)}
    input_evidence: list[dict[str, Any]] = [
        {
            "phase": "target_free",
            "source": "exp263_manifest",
            "path": str(manifest_path),
            "rows": expected_rows,
            "file_sha256": sha256_file(manifest_path),
            "decompressed_content_sha256": None,
            "logical_content_sha256": manifest.get("canonical_id_sha256"),
            "schema_sha256": manifest.get("generation_config_sha256"),
        }
    ]
    reference_keys: pd.DataFrame | None = None
    coverage_by_candidate: dict[str, float] = {}
    for candidate_id in primitive_ids:
        items = manifest["candidate_value_partitions"].get(candidate_id)
        if not items or len(items) != expected_folds:
            raise ValueError(f"{candidate_id} must have five value partitions")
        frame, evidence = _read_manifest_partitions(
            manifest_path,
            items,
            columns=VALUE_READ_COLUMNS,
            label=f"exp263_value::{candidate_id}",
        )
        input_evidence.extend(evidence)
        if reference_keys is None:
            reference_keys = frame[VALUE_KEY_COLUMNS].copy()
            reference_keys["id"] = reference_keys["id"].astype(str)
            reference_keys["well"] = reference_keys["well"].astype(str)
        else:
            _assert_same_keys(reference_keys, frame, candidate_id)
        available = frame["candidate_available"].astype(bool).to_numpy()
        finite_flag = frame["candidate_finite"].astype(bool).to_numpy()
        candidate_values = pd.to_numeric(
            frame["candidate_tvt"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        valid = available & finite_flag & np.isfinite(candidate_values)
        candidate_values[~valid] = np.nan
        values[:, column_by_candidate[candidate_id]] = candidate_values
        coverage_by_candidate[candidate_id] = float(valid.mean())

    if reference_keys is None:
        raise AssertionError("primitive candidate loading produced no keys")
    if len(reference_keys) != expected_rows:
        raise ValueError("candidate bank total row mismatch")
    if reference_keys["well"].nunique() != expected_wells:
        raise ValueError("candidate bank total well mismatch")
    if reference_keys["id"].duplicated().any():
        raise ValueError("candidate bank IDs must be unique")
    if set(reference_keys["outer_fold"].unique()) != set(range(expected_folds)):
        raise ValueError("candidate bank outer-fold coverage mismatch")

    _materialize_formulas(values, column_by_candidate, config)
    for candidate_id in candidate_ids[6:]:
        finite = np.isfinite(values[:, column_by_candidate[candidate_id]])
        coverage_by_candidate[candidate_id] = float(finite.mean())
    if any(not math.isclose(value, 1.0) for value in coverage_by_candidate.values()):
        raise ValueError(f"candidate finite coverage is not 1.0: {coverage_by_candidate}")

    key_hash = frame_content_sha256(reference_keys[VALUE_KEY_COLUMNS])
    bank = CandidateBank(
        keys=reference_keys.reset_index(drop=True),
        candidate_ids=candidate_ids,
        values=values,
        values_path=values_path,
        primitive_ids=primitive_ids,
        manifest=manifest,
        manifest_path=manifest_path,
        key_content_sha256=key_hash,
        candidate_content_sha256="",
        coverage_by_candidate=coverage_by_candidate,
        sample_parity=pd.DataFrame(),
        input_evidence=input_evidence,
    )
    bank.candidate_content_sha256 = candidate_bank_content_sha256(
        bank, int(get_nested(config, "audit.work_chunk_rows"))
    )
    parity_path = manifest_path.parent / str(
        manifest_cfg["small_parity_filename"]
    )
    if not parity_path.exists():
        raise FileNotFoundError(f"exp263 small parity sample missing: {parity_path}")
    sample_parity, parity_evidence = _build_sample_parity(
        bank,
        parity_path,
        float(get_nested(config, "candidate_bank.formula_parity_max_abs_ft")),
    )
    if not bool(sample_parity["passed"].all()):
        raise ValueError("exp263 formula sample parity failed")
    bank.sample_parity = sample_parity
    bank.input_evidence.append(parity_evidence)
    return bank


@dataclass
class GroupLayout:
    name: str
    codes: np.ndarray
    n_groups: int
    group_rows: np.ndarray
    group_well: np.ndarray
    group_fold: np.ndarray


@dataclass
class BlockAssignments:
    frame: pd.DataFrame
    well_names: np.ndarray
    well_codes: np.ndarray
    well_fold: np.ndarray
    layouts: dict[str, GroupLayout]


def assignments_with_evaluation_folds(
    assignments: BlockAssignments, row_folds: np.ndarray
) -> BlockAssignments:
    """Keep frozen exp293 block IDs but stratify readouts by exp226 folds."""
    folds = np.asarray(row_folds, dtype=np.int8)
    if len(folds) != len(assignments.well_codes):
        raise ValueError("evaluation fold row count mismatch")
    if set(np.unique(folds).astype(int)) != set(range(5)):
        raise ValueError("evaluation fold inventory mismatch")

    n_wells = len(assignments.well_names)
    first_well = np.full(n_wells, len(folds), dtype=np.int64)
    np.minimum.at(
        first_well,
        assignments.well_codes,
        np.arange(len(folds), dtype=np.int64),
    )
    well_fold = folds[first_well]
    if not np.array_equal(well_fold[assignments.well_codes], folds):
        raise ValueError("one well spans multiple evaluation folds")

    layouts: dict[str, GroupLayout] = {}
    for name, layout in assignments.layouts.items():
        first_group = np.full(layout.n_groups, len(folds), dtype=np.int64)
        np.minimum.at(
            first_group,
            layout.codes,
            np.arange(len(folds), dtype=np.int64),
        )
        layouts[name] = replace(
            layout,
            group_fold=folds[first_group].astype(np.int8),
        )
    return BlockAssignments(
        frame=assignments.frame,
        well_names=assignments.well_names,
        well_codes=assignments.well_codes,
        well_fold=well_fold,
        layouts=layouts,
    )


def _layout_from_codes(
    name: str,
    codes: np.ndarray,
    well_codes: np.ndarray,
    row_folds: np.ndarray,
) -> GroupLayout:
    if len(codes) == 0 or np.any(codes < 0):
        raise ValueError(f"invalid group codes for {name}")
    n_groups = int(codes.max()) + 1
    rows = np.bincount(codes, minlength=n_groups).astype(np.int64)
    first = np.full(n_groups, len(codes), dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(codes), dtype=np.int64))
    if np.any(first == len(codes)):
        raise ValueError(f"group code gap for {name}")
    group_well = well_codes[first].astype(np.int32)
    group_fold = row_folds[first].astype(np.int8)
    return GroupLayout(
        name=name,
        codes=codes.astype(np.int32, copy=False),
        n_groups=n_groups,
        group_rows=rows,
        group_well=group_well,
        group_fold=group_fold,
    )


def build_block_assignments(
    keys: pd.DataFrame, horizons: Sequence[int]
) -> BlockAssignments:
    wells = keys["well"].astype(str).to_numpy()
    row_folds = pd.to_numeric(keys["outer_fold"], errors="raise").to_numpy(
        dtype=np.int8
    )
    if len(wells) == 0:
        raise ValueError("candidate keys are empty")
    segment_start_mask = np.r_[True, wells[1:] != wells[:-1]]
    starts = np.flatnonzero(segment_start_mask)
    ends = np.r_[starts[1:], len(wells)]
    segment_wells = wells[starts]
    if pd.Index(segment_wells).duplicated().any():
        raise ValueError("well rows are not contiguous in candidate bank")
    lengths = (ends - starts).astype(np.int64)
    well_codes = np.repeat(
        np.arange(len(starts), dtype=np.int32), lengths
    )
    within_well = np.arange(len(wells), dtype=np.int64) - np.repeat(starts, lengths)
    well_fold = row_folds[starts]
    for start, end, fold in zip(starts, ends, well_fold, strict=True):
        if not np.all(row_folds[start:end] == fold):
            raise ValueError("one well spans multiple outer folds")
    layouts: dict[str, GroupLayout] = {}
    for horizon in horizons:
        if int(horizon) <= 0:
            raise ValueError("block horizon must be positive")
        blocks_per_well = (lengths + int(horizon) - 1) // int(horizon)
        offsets = np.r_[0, np.cumsum(blocks_per_well[:-1])].astype(np.int64)
        codes = offsets[well_codes] + within_well // int(horizon)
        name = f"h{int(horizon)}"
        layouts[name] = _layout_from_codes(
            name, codes.astype(np.int32), well_codes, row_folds
        )
    layouts["whole_well"] = _layout_from_codes(
        "whole_well", well_codes.copy(), well_codes, row_folds
    )
    assignment = keys[VALUE_KEY_COLUMNS].copy()
    assignment["well_code"] = well_codes
    for name, layout in layouts.items():
        assignment[f"{name}_group"] = layout.codes
    return BlockAssignments(
        frame=assignment,
        well_names=np.asarray(segment_wells, dtype=object),
        well_codes=well_codes,
        well_fold=well_fold,
        layouts=layouts,
    )


def load_hidden_like_sets(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    hidden_cfg = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        hidden_cfg["patterns"],
        label="hidden-like assignment",
        expected_sha256=str(hidden_cfg["expected_file_sha256"]),
    )
    frame = pd.read_csv(path)
    well_column = str(hidden_cfg["well_column"])
    if well_column not in frame:
        raise ValueError(f"hidden-like well column missing: {well_column}")
    sets: dict[str, set[str]] = {}
    for scope, role_column in hidden_cfg["role_columns"].items():
        if role_column not in frame:
            raise ValueError(f"hidden-like role column missing: {role_column}")
        selected = set(
            frame.loc[frame[role_column].eq("valid"), well_column].astype(str)
        )
        unknown = selected - expected_wells
        if unknown:
            raise ValueError(f"hidden-like scope has unknown wells: {sorted(unknown)[:5]}")
        sets[str(scope)] = selected
    evidence = {
        "phase": "target_free",
        "source": "hidden_like_assignment",
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": None,
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }
    return sets, evidence


# %% [markdown]
# ## 5. Target-free OOF generation and freeze boundary
#
# The saved K16 control is opened with an explicit target-free allowlist. The
# formation-conditioned K16 prediction, exp293 bank identity, block layout,
# raw-input manifest, formation support ledger, reference manifest,
# and foldwise kappa values are written and hashed before evaluation truth is
# loaded by the next section.

# %%
@dataclass
class SavedControl:
    path: Path
    values: np.ndarray
    frame: pd.DataFrame
    file_sha256: str
    decompressed_sha256: str
    parity_max_abs_ft: float
    folds: np.ndarray
    fold_by_well: dict[str, int]
    bank_fold_match_fraction: float


def extract_saved_control_folds(
    aligned: pd.DataFrame,
    bank: CandidateBank,
    expected_folds: int,
) -> tuple[np.ndarray, dict[str, int], float]:
    folds = pd.to_numeric(aligned["fold"], errors="raise").to_numpy(
        dtype=np.int8
    )
    if set(np.unique(folds).astype(int)) != set(range(expected_folds)):
        raise ValueError("saved control fold inventory mismatch")
    fold_frame = pd.DataFrame(
        {
            "well": aligned["well_id"].astype(str),
            "fold": folds,
        }
    )
    if bool(fold_frame.groupby("well", sort=False)["fold"].nunique().gt(1).any()):
        raise ValueError("saved control well spans multiple folds")
    fold_by_well = (
        fold_frame.drop_duplicates("well").set_index("well")["fold"].astype(int).to_dict()
    )
    if set(fold_by_well) != set(bank.keys["well"].astype(str)):
        raise ValueError("saved control fold well coverage mismatch")
    bank_folds = bank.keys["outer_fold"].to_numpy(dtype=np.int8)
    match_fraction = float(np.mean(folds == bank_folds))
    return folds, fold_by_well, match_fraction


def load_saved_control(
    config: Mapping[str, Any], bank: CandidateBank
) -> SavedControl:
    control_cfg = get_nested(config, "data.exp226_oof")
    path = resolve_gzip_by_decompressed_sha(
        control_cfg["patterns"],
        label="saved exp226 K16 OOF",
        expected_sha256=str(control_cfg["expected_decompressed_sha256"]),
    )
    allowlist = list(control_cfg["pre_freeze_columns"])
    frame = pd.read_csv(path, usecols=allowlist)
    required = {"well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"}
    if set(frame.columns) != required:
        raise ValueError(f"saved control allowlist mismatch: {list(frame.columns)}")
    frame["id"] = (
        frame["well_id"].astype(str)
        + "_"
        + frame["row_idx"].astype(int).astype(str)
    )
    if frame["id"].duplicated().any():
        raise ValueError("saved control contains duplicate IDs")
    indexer = pd.Index(frame["id"].astype(str)).get_indexer(
        bank.keys["id"].astype(str)
    )
    if np.any(indexer < 0) or len(frame) != len(bank.keys):
        raise ValueError("saved control identity coverage mismatch")
    aligned = frame.iloc[indexer].reset_index(drop=True)
    if not np.array_equal(
        aligned["well_id"].astype(str).to_numpy(),
        bank.keys["well"].astype(str).to_numpy(),
    ):
        raise ValueError("saved control well identity mismatch")
    if not np.array_equal(
        aligned["row_idx"].to_numpy(dtype=np.int64),
        bank.keys["well_row_idx"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("saved control row identity mismatch")
    folds, fold_by_well, bank_fold_match_fraction = extract_saved_control_folds(
        aligned,
        bank,
        int(get_nested(config, "validation.n_folds")),
    )
    values = pd.to_numeric(
        aligned["tvt_pred"], errors="raise"
    ).to_numpy(dtype=np.float64)
    bank_control = np.asarray(
        bank.values[:, bank.candidate_ids.index("exp226_k16")],
        dtype=np.float64,
    )
    parity = float(np.max(np.abs(values - bank_control), initial=0.0))
    return SavedControl(
        path=path,
        values=values,
        frame=aligned,
        file_sha256=sha256_file(path),
        decompressed_sha256=sha256_decompressed_gzip(path),
        parity_max_abs_ft=parity,
        folds=folds,
        fold_by_well=fold_by_well,
        bank_fold_match_fraction=bank_fold_match_fraction,
    )


@dataclass(frozen=True)
class FreezeEvidence:
    contract_path: Path
    contract_file_sha256: str
    manifest_path: Path
    manifest_file_sha256: str
    block_path: Path
    block_file_sha256: str
    block_decompressed_sha256: str
    input_manifest_path: Path
    raw_input_manifest_path: Path
    formation_schema_path: Path
    prediction_paths: dict[str, Path]
    kappa_paths: dict[str, Path]
    support_paths: dict[str, Path]
    reference_paths: dict[str, Path]
    prediction_file_sha256: dict[str, str]
    prediction_decompressed_sha256: dict[str, str]
    prediction_content_sha256: dict[str, str]
    support_content_sha256: dict[str, str]
    reference_content_sha256: dict[str, str]
    candidate_bank_content_sha256: str
    raw_input_content_sha256: str
    evaluation_truth_access_count_before_freeze: int


def build_contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "variants": get_nested(config, "model.params.variants"),
        "fixed_from_exp226": get_nested(config, "model.params.fixed_from_exp226"),
        "candidate_order": list(get_nested(config, "candidate_novelty_audit.candidate_order")),
        "block_horizons_rows": get_nested(
            config, "candidate_novelty_audit.block_horizons_rows"
        ),
        "success_criteria": get_nested(config, "success_criteria"),
        "execution": get_nested(config, "execution"),
        "forbidden_actions": get_nested(config, "forbidden_actions"),
    }


def freeze_target_free_bundle(
    bank: CandidateBank,
    assignments: BlockAssignments,
    control: SavedControl,
    variants: Mapping[str, VariantPrediction],
    raw_manifest: pd.DataFrame,
    hidden_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> FreezeEvidence:
    contract_path = artifacts_dir / f"{OUTPUT_PREFIX}_contract.json"
    write_json(contract_path, build_contract_payload(config))
    formation_schema_path = (
        artifacts_dir / f"{OUTPUT_PREFIX}_formation_signature_schema.json"
    )
    write_json(
        formation_schema_path,
        {
            "formation_columns": list(FORMATION_COLUMNS),
            "relative_depth": [
                f"z_minus_{column}" for column in FORMATION_COLUMNS
            ],
            "adjacent_thickness": [
                f"{right}_minus_{left}"
                for left, right in zip(
                    FORMATION_COLUMNS[:-1],
                    FORMATION_COLUMNS[1:],
                    strict=True,
                )
            ],
            "dimensions": 11,
            "robust_center": "outer_train_donor_segment_median",
            "robust_scale": "max(1.4826*MAD,1.0_ft)",
            "standardized_delta_clip": [-3.0, 3.0],
            "distance": "mean_squared_standardized_delta",
            "formation_factor": (
                "0.5 + 0.5 * exp(-0.5 * d_form_squared)"
            ),
            "nonfinite_fallback": "query_segment_parent_xy_weight",
        },
    )

    raw_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_raw_input_manifest.csv"
    _write_frame(raw_manifest_path, raw_manifest)
    raw_input_content_sha = frame_content_sha256(raw_manifest)

    target_free_input_manifest = pd.DataFrame(
        [
            *bank.input_evidence,
            {
                "phase": "target_free_saved_control",
                "source": "exp226_saved_k16_oof_allowlist",
                "path": str(control.path),
                "rows": len(control.frame),
                "file_sha256": control.file_sha256,
                "decompressed_content_sha256": control.decompressed_sha256,
                "logical_content_sha256": frame_content_sha256(
                    control.frame,
                    ["well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"],
                ),
                "schema_sha256": frame_schema_sha256(
                    control.frame[
                        ["well_id", "row_idx", "suffix_offset", "tvt_pred", "fold"]
                    ]
                ),
            },
            dict(hidden_evidence),
            {
                "phase": "target_free_generation_input",
                "source": "raw_horizontal_and_typewell_manifest",
                "path": str(raw_manifest_path),
                "rows": len(raw_manifest),
                "file_sha256": sha256_file(raw_manifest_path),
                "decompressed_content_sha256": None,
                "logical_content_sha256": raw_input_content_sha,
                "schema_sha256": frame_schema_sha256(raw_manifest),
            },
        ]
    )
    input_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv"
    _write_frame(input_manifest_path, target_free_input_manifest)

    block_path = artifacts_dir / f"{OUTPUT_PREFIX}_block_assignment.csv.gz"
    block_file_sha, block_decompressed_sha = write_gzip_frame(
        block_path, assignments.frame
    )

    prediction_paths: dict[str, Path] = {}
    prediction_file_sha: dict[str, str] = {}
    prediction_decompressed_sha: dict[str, str] = {}
    prediction_content_sha: dict[str, str] = {}
    kappa_paths: dict[str, Path] = {}
    support_paths: dict[str, Path] = {}
    reference_paths: dict[str, Path] = {}
    support_content_sha: dict[str, str] = {}
    reference_content_sha: dict[str, str] = {}
    kappa_evidence: dict[str, Any] = {}
    for variant, bundle in variants.items():
        prediction_path = (
            artifacts_dir / f"{OUTPUT_PREFIX}_{variant}_oof_predictions.csv.gz"
        )
        file_sha, decompressed_sha = write_gzip_frame(
            prediction_path, bundle.frame
        )
        kappa_path = (
            artifacts_dir / f"{OUTPUT_PREFIX}_{variant}_kappa_by_fold.csv"
        )
        _write_frame(kappa_path, bundle.kappa_by_fold)
        support_path = (
            artifacts_dir
            / f"{OUTPUT_PREFIX}_formation_signature_support_ledger.csv.gz"
        )
        write_gzip_frame(support_path, bundle.formation_support)
        reference_path = (
            artifacts_dir / f"{OUTPUT_PREFIX}_fold_and_reference_manifest.csv"
        )
        _write_frame(reference_path, bundle.reference_manifest)
        prediction_paths[variant] = prediction_path
        kappa_paths[variant] = kappa_path
        support_paths[variant] = support_path
        reference_paths[variant] = reference_path
        prediction_file_sha[variant] = file_sha
        prediction_decompressed_sha[variant] = decompressed_sha
        prediction_content_sha[variant] = bundle.prediction_content_sha256
        support_content_sha[variant] = frame_content_sha256(
            bundle.formation_support
        )
        reference_content_sha[variant] = frame_content_sha256(
            bundle.reference_manifest
        )
        kappa_evidence[variant] = {
            "path": str(kappa_path),
            "file_sha256": sha256_file(kappa_path),
            "logical_content_sha256": frame_content_sha256(bundle.kappa_by_fold),
        }

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "status": "target_free_predictions_bank_and_blocks_frozen",
        "frozen_at": datetime.now(UTC).isoformat(),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "candidate_bank_folds": sorted(
            int(value) for value in bank.keys["outer_fold"].unique()
        ),
        "evaluation_folds": sorted(int(value) for value in np.unique(control.folds)),
        "variants": {
            variant: {
                "k_segments": bundle.k_segments,
                "rows": len(bundle.frame),
                "finite_coverage": float(np.isfinite(bundle.values).mean()),
                "prediction_path": str(prediction_paths[variant]),
                "prediction_file_sha256": prediction_file_sha[variant],
                "prediction_decompressed_sha256": prediction_decompressed_sha[variant],
                "prediction_content_sha256": prediction_content_sha[variant],
                "donor_fit_truth_wells": bundle.donor_fit_truth_wells,
                "fold_run_count": bundle.fold_run_count,
                "outer_valid_truth_state_count": 0,
                "formation_support_path": str(support_paths[variant]),
                "formation_support_content_sha256": support_content_sha[variant],
                "reference_manifest_path": str(reference_paths[variant]),
                "reference_manifest_content_sha256": reference_content_sha[
                    variant
                ],
            }
            for variant, bundle in variants.items()
        },
        "saved_control": {
            "path": str(control.path),
            "file_sha256": control.file_sha256,
            "decompressed_content_sha256": control.decompressed_sha256,
            "parity_max_abs_ft_vs_exp293_bank": control.parity_max_abs_ft,
            "bank_fold_match_fraction": control.bank_fold_match_fraction,
            "fold_role": "exp226_evaluation_fold",
            "regenerated": False,
        },
        "exp293_bank": {
            "candidate_count": len(bank.candidate_ids),
            "candidate_order": list(bank.candidate_ids),
            "candidate_content_sha256": bank.candidate_content_sha256,
            "expected_candidate_content_sha256": get_nested(
                config, "data.exp293_bank.expected_candidate_content_sha256"
            ),
        },
        "block_assignment": {
            "path": str(block_path),
            "file_sha256": block_file_sha,
            "decompressed_content_sha256": block_decompressed_sha,
            "logical_content_sha256": frame_content_sha256(assignments.frame),
            "expected_decompressed_sha256": get_nested(
                config, "data.exp293_bank.expected_block_assignment_decompressed_sha256"
            ),
            "outer_fold_role": "exp263_bank_provenance_and_exp293_sha_only",
        },
        "raw_input_manifest": {
            "path": str(raw_manifest_path),
            "rows": len(raw_manifest),
            "file_sha256": sha256_file(raw_manifest_path),
            "logical_content_sha256": raw_input_content_sha,
        },
        "target_free_input_manifest": {
            "path": str(input_manifest_path),
            "rows": len(target_free_input_manifest),
            "file_sha256": sha256_file(input_manifest_path),
            "logical_content_sha256": frame_content_sha256(
                target_free_input_manifest
            ),
        },
        "hidden_like_assignment": dict(hidden_evidence),
        "kappa_by_fold": kappa_evidence,
        "evaluation_truth_access_count_before_freeze": 0,
        "evaluation_truth_columns_loaded_before_freeze": [],
        "donor_fit_truth_policy": (
            "TVT is attached only to outer-train donor objects; outer-valid "
            "objects retain no target/residual/coefficient arrays."
        ),
        "contract_file_sha256": sha256_file(contract_path),
        "formation_signature_schema": {
            "path": str(formation_schema_path),
            "file_sha256": sha256_file(formation_schema_path),
        },
        "config_file_sha256": sha256_file(find_config_path()),
        "frozen": True,
    }
    manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    write_json(manifest_path, manifest)
    return FreezeEvidence(
        contract_path=contract_path,
        contract_file_sha256=sha256_file(contract_path),
        manifest_path=manifest_path,
        manifest_file_sha256=sha256_file(manifest_path),
        block_path=block_path,
        block_file_sha256=block_file_sha,
        block_decompressed_sha256=block_decompressed_sha,
        input_manifest_path=input_manifest_path,
        raw_input_manifest_path=raw_manifest_path,
        formation_schema_path=formation_schema_path,
        prediction_paths=prediction_paths,
        kappa_paths=kappa_paths,
        support_paths=support_paths,
        reference_paths=reference_paths,
        prediction_file_sha256=prediction_file_sha,
        prediction_decompressed_sha256=prediction_decompressed_sha,
        prediction_content_sha256=prediction_content_sha,
        support_content_sha256=support_content_sha,
        reference_content_sha256=reference_content_sha,
        candidate_bank_content_sha256=bank.candidate_content_sha256,
        raw_input_content_sha256=raw_input_content_sha,
        evaluation_truth_access_count_before_freeze=0,
    )


def verify_freeze_before_truth(
    bank: CandidateBank,
    variants: Mapping[str, VariantPrediction],
    freeze: FreezeEvidence,
    config: Mapping[str, Any],
) -> None:
    if freeze.evaluation_truth_access_count_before_freeze != 0:
        raise ValueError("evaluation truth was accessed before freeze")
    if sha256_file(freeze.contract_path) != freeze.contract_file_sha256:
        raise ValueError("frozen contract changed before truth load")
    if sha256_file(freeze.manifest_path) != freeze.manifest_file_sha256:
        raise ValueError("freeze manifest changed before truth load")
    if sha256_file(freeze.block_path) != freeze.block_file_sha256:
        raise ValueError("block assignment changed before truth load")
    if (
        sha256_decompressed_gzip(freeze.block_path)
        != freeze.block_decompressed_sha256
    ):
        raise ValueError("block assignment decompressed content changed")
    if freeze.block_decompressed_sha256 != str(
        get_nested(
            config,
            "data.exp293_bank.expected_block_assignment_decompressed_sha256",
        )
    ):
        raise ValueError("block assignment does not match completed exp293")
    current_bank_sha = candidate_bank_content_sha256(
        bank, int(get_nested(config, "candidate_novelty_audit.work_chunk_rows"))
    )
    expected_bank_sha = str(
        get_nested(config, "data.exp293_bank.expected_candidate_content_sha256")
    )
    if current_bank_sha != freeze.candidate_bank_content_sha256:
        raise ValueError("candidate bank changed after freeze")
    if current_bank_sha != expected_bank_sha:
        raise ValueError("candidate bank does not match completed exp293")
    for variant, bundle in variants.items():
        if (
            frame_content_sha256(
                bundle.frame,
                [
                    "id",
                    "well",
                    "well_row_idx",
                    "outer_fold",
                    "variant",
                    "candidate_tvt",
                ],
            )
            != freeze.prediction_content_sha256[variant]
        ):
            raise ValueError(f"{variant} in-memory prediction changed after freeze")
        if (
            sha256_file(freeze.prediction_paths[variant])
            != freeze.prediction_file_sha256[variant]
        ):
            raise ValueError(f"{variant} prediction file changed after freeze")
        if (
            sha256_decompressed_gzip(freeze.prediction_paths[variant])
            != freeze.prediction_decompressed_sha256[variant]
        ):
            raise ValueError(f"{variant} decompressed prediction changed after freeze")
        if (
            frame_content_sha256(bundle.formation_support)
            != freeze.support_content_sha256[variant]
        ):
            raise ValueError(f"{variant} formation support changed after freeze")
        if (
            frame_content_sha256(bundle.reference_manifest)
            != freeze.reference_content_sha256[variant]
        ):
            raise ValueError(f"{variant} formation reference changed after freeze")


# %% [markdown]
# ## 6. Post-freeze truth loader and direct readout

# %%
def load_truth_after_freeze(
    bank: CandidateBank,
    variants: Mapping[str, VariantPrediction],
    freeze: FreezeEvidence,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    verify_freeze_before_truth(bank, variants, freeze, config)
    expected_wells = set(bank.keys["well"].astype(str))
    raw_dir, files = resolve_raw_train_dir(config, expected_wells)
    truth_column = str(get_nested(config, "data.raw_columns.truth"))
    visible_column = str(get_nested(config, "data.raw_columns.visible_input"))
    truth_frames: list[pd.DataFrame] = []
    input_evidence: list[dict[str, Any]] = []
    for path in files:
        well = path.name.split("__horizontal_well.csv", 1)[0]
        frame = pd.read_csv(path, usecols=[truth_column, visible_column])
        visible = pd.to_numeric(frame[visible_column], errors="coerce")
        suffix_mask = visible.isna().to_numpy()
        row_idx = np.flatnonzero(suffix_mask).astype(np.int32)
        truth_values = pd.to_numeric(
            frame.loc[suffix_mask, truth_column], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(truth_values).all():
            raise ValueError(f"raw truth contains nonfinite suffix TVT: {path}")
        truth_frames.append(
            pd.DataFrame(
                {
                    "id": [
                        f"{well}_{int(index)}" for index in row_idx
                    ],
                    "well": well,
                    "well_row_idx": row_idx,
                    "true_tvt": truth_values,
                }
            )
        )
        input_evidence.append(
            {
                "phase": "post_freeze_evaluation_truth",
                "source": "raw_train_horizontal",
                "path": str(path),
                "rows": len(frame),
                "suffix_rows": len(row_idx),
                "file_sha256": sha256_file(path),
                "raw_train_dir": str(raw_dir),
            }
        )
    truth_frame = pd.concat(truth_frames, ignore_index=True)
    if truth_frame["id"].duplicated().any() or len(truth_frame) != len(bank.keys):
        raise ValueError("post-freeze truth identity/count contract failed")
    indexer = pd.Index(truth_frame["id"].astype(str)).get_indexer(
        bank.keys["id"].astype(str)
    )
    if np.any(indexer < 0):
        raise ValueError("candidate IDs are missing from post-freeze truth")
    aligned = truth_frame.iloc[indexer].reset_index(drop=True)
    if not np.array_equal(
        aligned["id"].astype(str).to_numpy(),
        bank.keys["id"].astype(str).to_numpy(),
    ):
        raise ValueError("post-freeze truth alignment failed")
    truth = aligned["true_tvt"].to_numpy(dtype=np.float64)
    return truth, input_evidence, frame_content_sha256(
        aligned[["id", "true_tvt"]]
    )


def rmse_from_arrays(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def direct_metric_record(
    variant: str,
    scope: str,
    fold: int | None,
    mask: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    rows = int(mask.sum())
    if rows == 0:
        return {
            "variant": variant,
            "scope": scope,
            "fold": fold,
            "rows": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "bias": math.nan,
            "within10": math.nan,
        }
    error = prediction[mask] - truth[mask]
    return {
        "variant": variant,
        "scope": scope,
        "fold": fold,
        "rows": rows,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def build_direct_readouts(
    bank: CandidateBank,
    control: SavedControl,
    variants: Mapping[str, VariantPrediction],
    truth: np.ndarray,
    hidden_sets: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = {
        "exp226_k16": control.values,
        **{name: bundle.values for name, bundle in variants.items()},
    }
    folds = control.folds
    wells = bank.keys["well"].astype(str).to_numpy()
    md_since = bank.keys["md_since"].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    all_rows = np.ones(len(bank.keys), dtype=bool)
    for name, prediction in predictions.items():
        records.append(
            direct_metric_record(
                name, "pooled", None, all_rows, truth, prediction
            )
        )
        for fold in range(5):
            mask = folds == fold
            records.append(
                direct_metric_record(
                    name, "fold", fold, mask, truth, prediction
                )
            )
        for scope, (lo, hi) in DIRECT_BUCKETS.items():
            mask = (md_since >= lo) & (md_since < hi)
            records.append(
                direct_metric_record(
                    name, scope, None, mask, truth, prediction
                )
            )
        for scope, selected_wells in hidden_sets.items():
            mask = np.isin(wells, np.asarray(sorted(selected_wells), dtype=object))
            records.append(
                direct_metric_record(
                    name, scope, None, mask, truth, prediction
                )
            )

    well_codes, well_names = pd.factorize(wells, sort=False)
    if not np.array_equal(well_names.astype(str), pd.unique(wells)):
        raise ValueError("well factorization order changed")
    by_well_records: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        error_squared = np.square(prediction - truth)
        sse = np.bincount(well_codes, weights=error_squared)
        rows = np.bincount(well_codes)
        well_fold = control.fold_by_well
        for position, well in enumerate(well_names.astype(str)):
            by_well_records.append(
                {
                    "variant": name,
                    "well": well,
                    "outer_fold": int(well_fold[well]),
                    "rows": int(rows[position]),
                    "rmse": float(np.sqrt(sse[position] / rows[position])),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(by_well_records)


def one_direct_metric(
    metrics: pd.DataFrame,
    variant: str,
    scope: str,
    fold: int | None = None,
) -> float:
    selected = metrics[
        metrics["variant"].eq(variant) & metrics["scope"].eq(scope)
    ]
    if fold is None:
        selected = selected[selected["fold"].isna()]
    else:
        selected = selected[selected["fold"].eq(fold)]
    if len(selected) != 1:
        raise ValueError(
            f"direct metric not unique: {variant}/{scope}/{fold}"
        )
    return float(selected.iloc[0]["rmse"])


def evaluate_direct_guards(
    direct_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = get_nested(
        config, "success_criteria.direct_pass_for_at_least_one_variant"
    )
    control = "exp226_k16"
    output: dict[str, Any] = {}
    for variant in EXPECTED_VARIANTS:
        fold_deltas = [
            one_direct_metric(direct_metrics, variant, "fold", fold)
            - one_direct_metric(direct_metrics, control, "fold", fold)
            for fold in range(5)
        ]
        scope_deltas = {
            scope: one_direct_metric(direct_metrics, variant, scope)
            - one_direct_metric(direct_metrics, control, scope)
            for scope in (
                "near_0_250",
                "mid_250_1000",
                "1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            )
        }
        variant_well = by_well[by_well["variant"].eq(variant)][
            ["well", "rmse"]
        ].rename(columns={"rmse": "variant_rmse"})
        control_well = by_well[by_well["variant"].eq(control)][
            ["well", "rmse"]
        ].rename(columns={"rmse": "control_rmse"})
        paired_well = control_well.merge(
            variant_well, on="well", how="inner", validate="one_to_one"
        )
        if len(paired_well) != len(control_well):
            raise ValueError("paired by-well direct metrics lost well coverage")
        paired_delta = (
            paired_well["variant_rmse"] - paired_well["control_rmse"]
        )
        p95_delta = float(paired_delta.quantile(0.95))
        worst_delta = float(paired_delta.max())
        pooled_gain = one_direct_metric(
            direct_metrics, control, "pooled"
        ) - one_direct_metric(direct_metrics, variant, "pooled")
        checks = {
            "pooled_rmse_gain": pooled_gain
            >= float(criteria["pooled_rmse_gain_minimum_ft"]),
            "nonworse_folds": int(np.sum(np.asarray(fold_deltas) <= 0.0))
            >= int(criteria["nonworse_folds_minimum"]),
            "all_required_scopes": all(
                delta <= float(criteria["maximum_scope_delta_rmse_ft"])
                for delta in scope_deltas.values()
            ),
            "by_well_p95": p95_delta
            <= float(criteria["maximum_by_well_p95_delta_ft"]),
            "worst_well": worst_delta
            <= float(criteria["maximum_worst_well_delta_ft"]),
        }
        output[variant] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "pooled_rmse": one_direct_metric(
                direct_metrics, variant, "pooled"
            ),
            "pooled_delta_ft": one_direct_metric(
                direct_metrics, variant, "pooled"
            )
            - one_direct_metric(direct_metrics, control, "pooled"),
            "pooled_gain_ft": pooled_gain,
            "fold_deltas_ft": fold_deltas,
            "nonworse_folds": int(np.sum(np.asarray(fold_deltas) <= 0.0)),
            "scope_deltas_ft": scope_deltas,
            "by_well_p95_delta_ft": p95_delta,
            "worst_well_delta_ft": worst_delta,
        }
    return {
        "variants": output,
        "passed": bool(any(item["passed"] for item in output.values())),
    }


# %% [markdown]
# ## 7. Add-one candidate novelty readout

# %%
@dataclass
class OracleReference:
    row_best_sse: np.ndarray
    group_sse: dict[str, np.ndarray]
    candidate_total_sse: np.ndarray


def compute_oracle_reference(
    bank: CandidateBank,
    assignments: BlockAssignments,
    truth: np.ndarray,
) -> OracleReference:
    n_candidates = len(bank.candidate_ids)
    row_best = np.full(len(truth), np.inf, dtype=np.float64)
    candidate_total = np.zeros(n_candidates, dtype=np.float64)
    group_sse = {
        name: np.zeros((layout.n_groups, n_candidates), dtype=np.float64)
        for name, layout in assignments.layouts.items()
    }
    for position in range(n_candidates):
        prediction = np.asarray(bank.values[:, position], dtype=np.float64)
        error_squared = np.square(prediction - truth)
        row_best = np.minimum(row_best, error_squared)
        candidate_total[position] = float(error_squared.sum())
        for name, layout in assignments.layouts.items():
            group_sse[name][:, position] = np.bincount(
                layout.codes,
                weights=error_squared,
                minlength=layout.n_groups,
            )
    return OracleReference(
        row_best_sse=row_best,
        group_sse=group_sse,
        candidate_total_sse=candidate_total,
    )


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denominator <= 0.0:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def novelty_metric_record(
    *,
    variant: str,
    granularity: str,
    scope: str,
    fold: int | None,
    group_mask: np.ndarray,
    group_rows: np.ndarray,
    base_best_sse: np.ndarray,
    added_sse: np.ndarray,
    tie_atol: float,
) -> dict[str, Any]:
    selected = group_mask & (group_rows > 0)
    rows = int(group_rows[selected].sum())
    groups = int(selected.sum())
    base = base_best_sse[selected]
    added = added_sse[selected]
    improved = added + tie_atol < base
    oracle = np.where(improved, added, base)
    return {
        "variant": variant,
        "granularity": granularity,
        "scope": scope,
        "fold": fold,
        "rows": rows,
        "groups": groups,
        "base_oracle_rmse": float(np.sqrt(base.sum() / rows)),
        "add_one_oracle_rmse": float(np.sqrt(oracle.sum() / rows)),
        "oracle_improvement_ft": float(
            np.sqrt(base.sum() / rows) - np.sqrt(oracle.sum() / rows)
        ),
        "strict_unique_best_groups": int(improved.sum()),
        "strict_unique_best_fraction": float(improved.mean()),
        "strict_unique_best_rows": int(group_rows[selected][improved].sum()),
    }


def build_novelty_readouts(
    bank: CandidateBank,
    assignments: BlockAssignments,
    reference: OracleReference,
    variants: Mapping[str, VariantPrediction],
    truth: np.ndarray,
    tie_atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    correlation_records: list[dict[str, Any]] = []
    by_well_records: list[dict[str, Any]] = []
    row_mask = np.ones(len(truth), dtype=bool)
    row_rows = np.ones(len(truth), dtype=np.int64)
    row_folds = bank.keys["outer_fold"].to_numpy(dtype=np.int8)
    for variant, bundle in variants.items():
        added_values = bundle.values.astype(np.float32).astype(np.float64)
        added_row_sse = np.square(added_values - truth)
        metric_records.append(
            novelty_metric_record(
                variant=variant,
                granularity="row",
                scope="pooled",
                fold=None,
                group_mask=row_mask,
                group_rows=row_rows,
                base_best_sse=reference.row_best_sse,
                added_sse=added_row_sse,
                tie_atol=tie_atol,
            )
        )
        for position, candidate in enumerate(bank.candidate_ids):
            correlation_records.append(
                {
                    "variant": variant,
                    "bank_candidate": candidate,
                    "pearson_prediction_correlation": pearson_correlation(
                        added_values,
                        np.asarray(bank.values[:, position], dtype=np.float64),
                    ),
                }
            )
        for name, layout in assignments.layouts.items():
            added_group_sse = np.bincount(
                layout.codes,
                weights=added_row_sse,
                minlength=layout.n_groups,
            )
            base_best = np.min(reference.group_sse[name], axis=1)
            all_groups = np.ones(layout.n_groups, dtype=bool)
            metric_records.append(
                novelty_metric_record(
                    variant=variant,
                    granularity=name,
                    scope="pooled",
                    fold=None,
                    group_mask=all_groups,
                    group_rows=layout.group_rows,
                    base_best_sse=base_best,
                    added_sse=added_group_sse,
                    tie_atol=tie_atol,
                )
            )
            for fold in range(5):
                metric_records.append(
                    novelty_metric_record(
                        variant=variant,
                        granularity=name,
                        scope="fold",
                        fold=fold,
                        group_mask=layout.group_fold == fold,
                        group_rows=layout.group_rows,
                        base_best_sse=base_best,
                        added_sse=added_group_sse,
                        tie_atol=tie_atol,
                    )
                )
            if name == "h512":
                strict = added_group_sse + tie_atol < base_best
                for well_code, well in enumerate(assignments.well_names.astype(str)):
                    selected = layout.group_well == well_code
                    by_well_records.append(
                        {
                            "variant": variant,
                            "well": well,
                            "outer_fold": int(assignments.well_fold[well_code]),
                            "h512_blocks": int(selected.sum()),
                            "h512_strict_unique_best_blocks": int(
                                strict[selected].sum()
                            ),
                            "h512_strict_unique_best_fraction": float(
                                strict[selected].mean()
                            ),
                        }
                    )
        for fold in range(5):
            selected = row_folds == fold
            metric_records.append(
                novelty_metric_record(
                    variant=variant,
                    granularity="row",
                    scope="fold",
                    fold=fold,
                    group_mask=selected,
                    group_rows=row_rows,
                    base_best_sse=reference.row_best_sse,
                    added_sse=added_row_sse,
                    tie_atol=tie_atol,
                )
            )
    return (
        pd.DataFrame(metric_records),
        pd.DataFrame(correlation_records),
        pd.DataFrame(by_well_records),
    )


def one_novelty_metric(
    metrics: pd.DataFrame,
    variant: str,
    granularity: str,
    scope: str,
    fold: int | None = None,
) -> pd.Series:
    selected = metrics[
        metrics["variant"].eq(variant)
        & metrics["granularity"].eq(granularity)
        & metrics["scope"].eq(scope)
    ]
    if fold is None:
        selected = selected[selected["fold"].isna()]
    else:
        selected = selected[selected["fold"].eq(fold)]
    if len(selected) != 1:
        raise ValueError(
            f"novelty metric not unique: {variant}/{granularity}/{scope}/{fold}"
        )
    return selected.iloc[0]


def evaluate_novelty_guards(
    novelty_metrics: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    criteria = get_nested(
        config,
        "success_criteria.candidate_novelty_pass_for_at_least_one_variant",
    )
    output: dict[str, Any] = {}
    for variant in EXPECTED_VARIANTS:
        h512 = one_novelty_metric(
            novelty_metrics, variant, "h512", "pooled"
        )
        whole = one_novelty_metric(
            novelty_metrics, variant, "whole_well", "pooled"
        )
        fold_improvements = [
            float(
                one_novelty_metric(
                    novelty_metrics, variant, "h512", "fold", fold
                )["oracle_improvement_ft"]
            )
            for fold in range(5)
        ]
        checks = {
            "h512_oracle_improvement": float(h512["oracle_improvement_ft"])
            >= float(criteria["minimum_h512_oracle_rmse_improvement_ft"]),
            "whole_well_oracle_improvement": float(
                whole["oracle_improvement_ft"]
            )
            >= float(criteria["minimum_whole_well_oracle_rmse_improvement_ft"]),
            "h512_strict_unique_best_fraction": float(
                h512["strict_unique_best_fraction"]
            )
            >= float(criteria["minimum_h512_strict_unique_best_fraction"]),
            "improved_folds": int(np.sum(np.asarray(fold_improvements) > 0.0))
            >= int(criteria["minimum_improved_folds"]),
        }
        output[variant] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "h512_base_oracle_rmse": float(h512["base_oracle_rmse"]),
            "h512_add_one_oracle_rmse": float(h512["add_one_oracle_rmse"]),
            "h512_oracle_improvement_ft": float(h512["oracle_improvement_ft"]),
            "whole_well_oracle_improvement_ft": float(
                whole["oracle_improvement_ft"]
            ),
            "h512_strict_unique_best_fraction": float(
                h512["strict_unique_best_fraction"]
            ),
            "h512_fold_improvements_ft": fold_improvements,
            "improved_folds": int(np.sum(np.asarray(fold_improvements) > 0.0)),
        }
    return {
        "variants": output,
        "passed": bool(any(item["passed"] for item in output.values())),
    }


# %% [markdown]
# ## 8. PASS/FAIL decision and generated artifacts

# %%
def validate_execution_contract(
    config: Mapping[str, Any], *, require_kaggle_authorization: bool
) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp376 route must remain pf_beam")
    allowed_statuses = {
        "implementation_complete_not_executed",
        "kaggle_cpu_train_authorized_not_run",
    }
    if get_nested(config, "experiment.status") not in allowed_statuses:
        raise ValueError(
            "exp376 status must identify implemented or authorized-not-run state"
        )
    if not bool(get_nested(config, "execution.implementation")):
        raise ValueError("exp376 implementation flag is disabled")
    if not bool(get_nested(config, "execution.implementation_authorized")):
        raise ValueError("exp376 implementation is not authorized")
    if require_kaggle_authorization and not bool(
        get_nested(config, "execution.kaggle_execution_authorized")
    ):
        raise ValueError("exp376 Kaggle execution is not authorized")
    variants = {
        str(item["name"]): int(item["k_segments"])
        for item in get_nested(config, "model.params.variants")
    }
    if variants != EXPECTED_VARIANTS:
        raise ValueError(
            "exp376 must contain exactly one formation-conditioned K16 variant"
        )
    execution = get_nested(config, "execution")
    expected = {
        "active_scientific_variants": 1,
        "outer_evaluation_folds": 5,
        "total_variant_fold_runs": 5,
        "lightgbm_config_count": 0,
        "trained_fold_count": 0,
        "total_boosters": 0,
    }
    for key, value in expected.items():
        if int(execution[key]) != value:
            raise ValueError(f"execution contract changed: {key}")
    if bool(execution["parent_or_control_regeneration"]):
        raise ValueError("saved K16 control regeneration is forbidden")
    if bool(execution["gpu"]) or bool(execution["inference"]) or bool(
        execution["submission"]
    ):
        raise ValueError("GPU, inference, and submission must remain disabled")


def evaluate_stage0_guard(
    bank: CandidateBank,
    control: SavedControl,
    variants: Mapping[str, VariantPrediction],
    freeze: FreezeEvidence,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if set(variants) != set(EXPECTED_VARIANTS):
        raise ValueError("Stage 0 received an unexpected variant inventory")
    bundle = variants["formation_conditioned_k16"]
    support = bundle.formation_support
    reference = bundle.reference_manifest
    gates = get_nested(config, "validation.gates.stage0_target_free")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_segments = int(
        get_nested(config, "validation.expected_k16_segments")
    )
    factor_columns = [
        "formation_factor_min",
        "formation_factor_max",
        "formation_factor_mean",
        "formation_distance_squared_mean",
    ]
    ess_columns = [
        "parent_effective_sample_size",
        "new_effective_sample_size",
        "new_to_parent_ess_ratio",
    ]
    finite_columns = factor_columns + ess_columns + [
        "nearest_donor_distance",
        "formation_surface_distance",
    ]
    finite_coverage = float(
        np.isfinite(support[finite_columns].to_numpy(dtype=np.float64)).mean()
    )
    fallback_fraction = float(support["fallback"].astype(bool).mean())
    ess_ratio_p05 = float(
        support["new_to_parent_ess_ratio"].quantile(0.05)
    )
    factor_minimum = float(support["formation_factor_min"].min())
    factor_maximum = float(support["formation_factor_max"].max())
    validation_reference_count = int(
        reference["validation_reference_count"].sum()
    )
    validation_truth_read_count = int(
        reference["validation_truth_read_count"].sum()
    )
    checks = {
        "score_row_count_exact": len(bundle.frame) == expected_rows,
        "score_well_count_exact": int(bundle.frame["well"].nunique())
        == expected_wells,
        "k16_segment_count_exact": len(support) == expected_segments,
        "one_segment_inventory_per_well": bool(
            support.groupby("well", sort=False)["segment_id"]
            .nunique()
            .eq(16)
            .all()
        ),
        "factor_and_support_finite": finite_coverage
        >= float(gates["finite_coverage_minimum"]),
        "signature_finite": bool(support["signature_finite"].all()),
        "factor_minimum": factor_minimum
        >= float(gates["formation_factor_minimum"]),
        "factor_maximum": factor_maximum
        <= float(gates["formation_factor_maximum"]),
        "fallback_fraction": fallback_fraction
        <= float(gates["nonfinite_fallback_fraction_maximum"]),
        "ess_ratio_p05": ess_ratio_p05
        >= float(gates["segment_p05_new_to_parent_ess_ratio_minimum"]),
        "validation_reference_count_zero": validation_reference_count
        <= int(gates["validation_reference_count_maximum"]),
        "validation_truth_read_count_zero": validation_truth_read_count
        <= int(gates["validation_truth_read_count_maximum"]),
        "validation_formation_read_count_zero": int(
            reference["validation_formation_read_count"].sum()
        )
        == 0,
        "source_valid_overlap_zero": int(
            reference["source_valid_overlap"].sum()
        )
        == 0,
        "target_formation_columns_not_read": bool(
            (~reference["target_formation_columns_read"].astype(bool)).all()
        ),
        "donor_signature_self_exclusion": bool(
            reference["donor_signature_self_exclusion"].all()
        ),
        "parent_oof_decompressed_sha": control.decompressed_sha256
        == str(
            get_nested(
                config,
                "validation.parent_control.oof_decompressed_sha256",
            )
        ),
        "parent_prediction_parity": control.parity_max_abs_ft
        <= float(
            get_nested(
                config,
                "success_criteria.technical_all_required.required_control_parity_max_abs_ft",
            )
        ),
        "prediction_finite": bool(np.isfinite(bundle.values).all()),
        "truth_access_before_freeze_zero": (
            freeze.evaluation_truth_access_count_before_freeze == 0
        ),
    }
    return {
        "stage": "stage0_target_free",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rows": len(bundle.frame),
        "wells": int(bundle.frame["well"].nunique()),
        "segments": len(support),
        "formation_factor_minimum": factor_minimum,
        "formation_factor_maximum": factor_maximum,
        "finite_coverage": finite_coverage,
        "fallback_fraction": fallback_fraction,
        "ess_ratio_p05": ess_ratio_p05,
        "validation_reference_count": validation_reference_count,
        "validation_truth_read_count": validation_truth_read_count,
        "support_content_sha256": freeze.support_content_sha256[
            "formation_conditioned_k16"
        ],
        "reference_content_sha256": freeze.reference_content_sha256[
            "formation_conditioned_k16"
        ],
        "prediction_content_sha256": bundle.prediction_content_sha256,
        "fail_action": (
            None
            if all(checks.values())
            else "stop_before_truth_scoring_without_rescue_grid"
        ),
    }


def evaluate_technical_checks(
    bank: CandidateBank,
    assignments: BlockAssignments,
    control: SavedControl,
    variants: Mapping[str, VariantPrediction],
    freeze: FreezeEvidence,
    direct_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_bank_sha = str(
        get_nested(config, "data.exp293_bank.expected_candidate_content_sha256")
    )
    expected_block_sha = str(
        get_nested(
            config,
            "data.exp293_bank.expected_block_assignment_decompressed_sha256",
        )
    )
    control_rmse = one_direct_metric(direct_metrics, "exp226_k16", "pooled")
    checks = {
        "row_count_exact": len(bank.keys) == expected_rows,
        "well_count_exact": int(bank.keys["well"].nunique()) == expected_wells,
        "duplicate_rows_zero": not bank.keys["id"].duplicated().any(),
        "fold_inventory_exact": set(bank.keys["outer_fold"].astype(int))
        == set(range(5)),
        "saved_control_evaluation_fold_inventory_exact": set(
            np.unique(control.folds).astype(int)
        )
        == set(range(5)),
        "candidate_bank_sha_match": bank.candidate_content_sha256
        == expected_bank_sha,
        "block_assignment_sha_match": freeze.block_decompressed_sha256
        == expected_block_sha,
        "control_decompressed_sha_match": control.decompressed_sha256
        == str(
            get_nested(config, "data.exp226_oof.expected_decompressed_sha256")
        ),
        "control_bank_parity": control.parity_max_abs_ft
        <= float(
            get_nested(
                config,
                "success_criteria.technical_all_required.required_control_parity_max_abs_ft",
            )
        ),
        "control_rmse_parity": abs(
            control_rmse
            - float(get_nested(config, "validation.control_oof_rmse_ft"))
        )
        <= 0.001,
        "variant_inventory_exact": {
            name: bundle.k_segments for name, bundle in variants.items()
        }
        == EXPECTED_VARIANTS,
        "variant_finite_coverage_one": all(
            np.isfinite(bundle.values).all() for bundle in variants.values()
        ),
        "variant_row_count_exact": all(
            len(bundle.values) == expected_rows for bundle in variants.values()
        ),
        "variant_fold_runs_exact": all(
            bundle.fold_run_count == 5 for bundle in variants.values()
        ),
        "evaluation_truth_access_before_freeze_zero": (
            freeze.evaluation_truth_access_count_before_freeze == 0
        ),
        "parent_control_regeneration_zero": not bool(
            get_nested(config, "execution.parent_or_control_regeneration")
        ),
        "bank_and_block_shape_consistent": len(assignments.frame) == expected_rows,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def persist_outputs(
    bank: CandidateBank,
    control: SavedControl,
    variants: Mapping[str, VariantPrediction],
    freeze: FreezeEvidence,
    truth_evidence: Sequence[Mapping[str, Any]],
    truth_content_sha256: str,
    direct_metrics: pd.DataFrame,
    direct_by_well: pd.DataFrame,
    novelty_metrics: pd.DataFrame,
    correlations: pd.DataFrame,
    novelty_by_well: pd.DataFrame,
    stage0: Mapping[str, Any],
    technical: Mapping[str, Any],
    direct: Mapping[str, Any],
    novelty: Mapping[str, Any],
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    stage0_path = artifacts_dir / f"{OUTPUT_PREFIX}_stage0_technical_guard.json"
    write_json(stage0_path, stage0)
    direct_path = artifacts_dir / f"{OUTPUT_PREFIX}_direct_metrics.csv"
    direct_well_path = artifacts_dir / f"{OUTPUT_PREFIX}_direct_by_well.csv"
    novelty_path = artifacts_dir / f"{OUTPUT_PREFIX}_candidate_novelty_metrics.csv"
    correlation_path = artifacts_dir / f"{OUTPUT_PREFIX}_candidate_correlations.csv"
    novelty_well_path = artifacts_dir / f"{OUTPUT_PREFIX}_candidate_novelty_by_well.csv"
    truth_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_post_freeze_truth_manifest.csv"
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    sha_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    for path, frame in (
        (direct_path, direct_metrics),
        (direct_well_path, direct_by_well),
        (novelty_path, novelty_metrics),
        (correlation_path, correlations),
        (novelty_well_path, novelty_by_well),
        (truth_manifest_path, pd.DataFrame(truth_evidence)),
    ):
        _write_frame(path, frame)

    direct_pass = bool(direct["passed"])
    novelty_pass = bool(novelty["passed"])
    technical_pass = bool(technical["passed"])
    scientific_support = bool(technical_pass and (direct_pass or novelty_pass))
    decision = {
        "technical_passed": technical_pass,
        "direct_passed": bool(technical_pass and direct_pass),
        "candidate_novelty_passed": bool(technical_pass and novelty_pass),
        "scientific_support": scientific_support,
        "next_action": (
            "request_separate_candidate_bank_integration_approval"
            if technical_pass and novelty_pass
            else (
                "request_separate_direct_current_test_design_approval"
                if technical_pass and direct_pass
                else "close_formation_conditioned_donor_branch_without_rescue_grid"
            )
        ),
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_passed_stage1_stage2_completed",
        "route": "pf_beam",
        "completed_at": datetime.now(UTC).isoformat(),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "control_rmse": one_direct_metric(
            direct_metrics, "exp226_k16", "pooled"
        ),
        "stage0_target_free": stage0,
        "technical": technical,
        "direct": direct,
        "candidate_novelty": novelty,
        "decision": decision,
        "truth_content_sha256": truth_content_sha256,
        "candidate_bank_content_sha256": bank.candidate_content_sha256,
        "block_assignment_decompressed_sha256": (
            freeze.block_decompressed_sha256
        ),
        "variant_prediction_content_sha256": {
            name: bundle.prediction_content_sha256
            for name, bundle in variants.items()
        },
        "oracle_prediction_persisted": False,
        "execution": {
            "scientific_variants": 1,
            "outer_folds": 5,
            "variant_fold_runs": 5,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "parent_control_regeneration": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
    }
    write_json(summary_path, summary)

    artifact_paths = [
        freeze.contract_path,
        freeze.manifest_path,
        freeze.block_path,
        freeze.input_manifest_path,
        freeze.raw_input_manifest_path,
        freeze.formation_schema_path,
        stage0_path,
        *freeze.support_paths.values(),
        *freeze.reference_paths.values(),
        *freeze.prediction_paths.values(),
        *freeze.kappa_paths.values(),
        direct_path,
        direct_well_path,
        novelty_path,
        correlation_path,
        novelty_well_path,
        truth_manifest_path,
        summary_path,
    ]
    sha_records = []
    for path in artifact_paths:
        sha_records.append(
            {
                "artifact": path.name,
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "decompressed_content_sha256": (
                    sha256_decompressed_gzip(path)
                    if path.suffix == ".gz"
                    else None
                ),
            }
        )
    sha_manifest = pd.DataFrame(sha_records)
    _write_frame(sha_manifest_path, sha_manifest)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_passed_stage1_stage2_completed",
        "route": "pf_beam",
        "cv": min(
            item["pooled_rmse"] for item in direct["variants"].values()
        ),
        "public_lb": None,
        "private_lb": None,
        "technical_passed": technical_pass,
        "stage0_passed": bool(stage0["passed"]),
        "direct_passed": bool(technical_pass and direct_pass),
        "candidate_novelty_passed": bool(technical_pass and novelty_pass),
        "decision": decision["next_action"],
        "variants": direct["variants"],
        "candidate_novelty": novelty["variants"],
        "prediction_content_sha256": {
            name: bundle.prediction_content_sha256
            for name, bundle in variants.items()
        },
        "sha_manifest_file_sha256": sha256_file(sha_manifest_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config, require_kaggle_authorization=True)
    artifacts_dir = runtime_artifacts_dir()
    work_dir = runtime_work_dir()
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Variants:", EXPECTED_VARIANTS)
    print("Execution contract: 1 variant x 5 reporting folds = 5 CPU runs")
    print("LightGBM configs / trained folds / boosters: 0 / 0 / 0")
    print("Saved K16 control regeneration: 0")

    bank = build_candidate_bank(config, work_dir)
    expected_bank_sha = str(
        get_nested(config, "data.exp293_bank.expected_candidate_content_sha256")
    )
    if bank.candidate_content_sha256 != expected_bank_sha:
        raise ValueError("reconstructed candidate bank SHA differs from exp293")
    horizons = [
        int(value)
        for value in get_nested(
            config, "candidate_novelty_audit.block_horizons_rows"
        )
    ]
    assignments = build_block_assignments(bank.keys, horizons)
    hidden_sets, hidden_evidence = load_hidden_like_sets(
        config, set(bank.keys["well"].astype(str))
    )
    control = load_saved_control(config, bank)
    evaluation_assignments = assignments_with_evaluation_folds(
        assignments, control.folds
    )
    raw_dir, horizontal_files = resolve_raw_train_dir(
        config, set(bank.keys["well"].astype(str))
    )
    raw_manifest = build_raw_input_manifest(raw_dir, horizontal_files)

    variants = {
        variant: build_variant_oof(
            variant,
            config,
            bank,
            horizontal_files,
            control.fold_by_well,
        )
        for variant in EXPECTED_VARIANTS
    }
    freeze = freeze_target_free_bundle(
        bank,
        assignments,
        control,
        variants,
        raw_manifest,
        hidden_evidence,
        config,
        artifacts_dir,
    )
    print("Target-free predictions frozen:", freeze.prediction_content_sha256)
    print("Candidate bank frozen:", freeze.candidate_bank_content_sha256)
    print("Block assignment frozen:", freeze.block_decompressed_sha256)
    print(
        "Evaluation truth access before freeze:",
        freeze.evaluation_truth_access_count_before_freeze,
    )
    verify_freeze_before_truth(bank, variants, freeze, config)
    stage0 = evaluate_stage0_guard(
        bank, control, variants, freeze, config
    )
    stage0_path = (
        artifacts_dir / f"{OUTPUT_PREFIX}_stage0_technical_guard.json"
    )
    write_json(stage0_path, stage0)
    print("Stage 0 target-free PASS:", stage0["passed"])
    print("Stage 0 target-free metrics:")
    print(json.dumps(to_jsonable(stage0), indent=2, ensure_ascii=False))
    if not stage0["passed"]:
        summary = {
            "experiment": EXPERIMENT_NAME,
            "status": "stage0_target_free_failed",
            "route": "pf_beam",
            "completed_at": datetime.now(UTC).isoformat(),
            "stage0_target_free": stage0,
            "decision": {
                "technical_passed": False,
                "direct_passed": False,
                "candidate_novelty_passed": False,
                "next_action": "close_branch_without_truth_scoring_or_rescue_grid",
            },
            "truth_loaded": False,
            "parent_control_regenerated": False,
        }
        summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
        write_json(summary_path, summary)
        write_json(
            runtime_metrics_path(),
            {
                "experiment": EXPERIMENT_NAME,
                "status": "stage0_target_free_failed",
                "route": "pf_beam",
                "cv": None,
                "public_lb": None,
                "private_lb": None,
                "stage0_target_free": stage0,
                "decision": summary["decision"]["next_action"],
            },
        )
        print("Stage 0 failed; truth scoring was not opened.")
        return summary

    truth, truth_evidence, truth_sha = load_truth_after_freeze(
        bank, variants, freeze, config
    )
    direct_metrics, direct_by_well = build_direct_readouts(
        bank, control, variants, truth, hidden_sets
    )
    direct = evaluate_direct_guards(
        direct_metrics, direct_by_well, config
    )
    reference = compute_oracle_reference(bank, assignments, truth)
    novelty_metrics, correlations, novelty_by_well = build_novelty_readouts(
        bank,
        evaluation_assignments,
        reference,
        variants,
        truth,
        float(
            get_nested(
                config,
                "candidate_novelty_audit.tie_atol_squared_ft",
            )
        ),
    )
    novelty = evaluate_novelty_guards(novelty_metrics, config)
    technical = evaluate_technical_checks(
        bank,
        assignments,
        control,
        variants,
        freeze,
        direct_metrics,
        config,
    )
    technical["checks"]["stage0_target_free_pass"] = bool(stage0["passed"])
    technical["passed"] = bool(
        technical["passed"] and stage0["passed"]
    )
    print("Stage 1 direct metrics:")
    print(json.dumps(to_jsonable(direct), indent=2, ensure_ascii=False))
    print("Stage 2 fixed12 add-one metrics:")
    print(json.dumps(to_jsonable(novelty), indent=2, ensure_ascii=False))
    print("Final technical checks:")
    print(json.dumps(to_jsonable(technical), indent=2, ensure_ascii=False))
    summary = persist_outputs(
        bank,
        control,
        variants,
        freeze,
        truth_evidence,
        truth_sha,
        direct_metrics,
        direct_by_well,
        novelty_metrics,
        correlations,
        novelty_by_well,
        stage0,
        technical,
        direct,
        novelty,
        config,
        artifacts_dir,
    )
    print("Technical PASS:", technical["passed"])
    print("Direct PASS:", direct["passed"])
    print("Candidate novelty PASS:", novelty["passed"])
    print("Next action:", summary["decision"]["next_action"])
    print("Artifacts:", artifacts_dir)
    return summary


# %% [markdown]
# ## 9. Setup, contract preview, and execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Config:", CONFIG_PATH)
    print("Config SHA256:", sha256_file(CONFIG_PATH))
    validate_execution_contract(CONFIG, require_kaggle_authorization=True)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment"),
                "lineage": get_nested(CONFIG, "lineage"),
                "variants": get_nested(CONFIG, "model.params.variants"),
                "fixed_from_exp226": get_nested(
                    CONFIG, "model.params.fixed_from_exp226"
                ),
                "execution": get_nested(CONFIG, "execution"),
                "success_criteria": get_nested(CONFIG, "success_criteria"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    SUMMARY = run_audit(CONFIG)
    print(json.dumps(SUMMARY["decision"], indent=2, ensure_ascii=False))
