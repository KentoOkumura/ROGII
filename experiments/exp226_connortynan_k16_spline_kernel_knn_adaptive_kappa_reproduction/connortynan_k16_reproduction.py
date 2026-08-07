from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_PREFIX = "exp226"


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def dataframe_content_sha256(frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False).encode()
    return hashlib.sha256(data).hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False).encode()
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as fp:
        fp.write(data)
    return hashlib.sha256(data).hexdigest()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


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


@dataclass
class PredictionResult:
    pred: np.ndarray
    geop: np.ndarray
    delta: np.ndarray
    donor_distance: np.ndarray
    design: np.ndarray
    raw_drift: np.ndarray
    smooth_drift: np.ndarray
    gate_count: int
    summary: dict[str, Any]


def params_from_config(config: dict[str, Any]) -> K16Params:
    params = get_nested(config, "model.params", {}) or {}

    def tuple_float(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
        value = params.get(key, default)
        return tuple(float(v) for v in value)

    return K16Params(
        theta0=float(params.get("theta0", K16Params.theta0)),
        k_segments=int(params.get("k_segments", K16Params.k_segments)),
        local_linear_k=int(params.get("local_linear_k", K16Params.local_linear_k)),
        local_linear_bandwidth=float(
            params.get("local_linear_bandwidth", K16Params.local_linear_bandwidth)
        ),
        local_linear_ridge=float(params.get("local_linear_ridge", K16Params.local_linear_ridge)),
        smooth_rho=float(params.get("smooth_rho", K16Params.smooth_rho)),
        gate=float(params.get("gate", K16Params.gate)),
        field_min_proj=float(params.get("field_min_proj", K16Params.field_min_proj)),
        kbins=tuple_float("kbins", K16Params.kbins),
        kappa_regimes=tuple_float("kappa_regimes", K16Params.kappa_regimes),
        rot_max_deg=float(params.get("rot_max_deg", K16Params.rot_max_deg)),
        ancc_theta_bandwidth=float(
            params.get("ancc_theta_bandwidth", K16Params.ancc_theta_bandwidth)
        ),
        enable_gr_correction=bool(params.get("enable_gr_correction", True)),
        enable_u_projection=bool(params.get("enable_u_projection", True)),
    )


def list_wells(data_dir: Path) -> list[str]:
    return [
        path.name.split("__")[0] for path in sorted(Path(data_dir).glob("*__horizontal_well.csv"))
    ]


def last_known_index(tvt_input: np.ndarray) -> int:
    finite = np.where(np.isfinite(tvt_input))[0]
    if len(finite) == 0:
        raise ValueError("well has no finite TVT_input anchor")
    return int(finite.max())


def segment_geometry(
    x: np.ndarray,
    y: np.ndarray,
    s: int,
    n: int,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, n, params.k_segments + 1)
    step_idx = np.arange(1, n + 1.0)
    segid = np.clip(
        np.searchsorted(edges[1:], step_idx, side="left"),
        0,
        params.k_segments - 1,
    )
    mid = np.empty((params.k_segments, 2), dtype=float)
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
        proj[j] = np.cos(az[j] - theta)
    return segid.astype(int), mid, proj, az


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


def load_train_wells(
    train_dir: Path,
    params: K16Params,
    max_wells: int | None = None,
) -> list[WellData]:
    wells: list[WellData] = []
    for fp in sorted(Path(train_dir).glob("*__horizontal_well.csv")):
        if max_wells is not None and len(wells) >= max_wells:
            break
        wid = fp.name.split("__")[0]
        cols = ["X", "Y", "Z", "MD", "TVT", "TVT_input", "ANCC", "GR"]
        frame = pd.read_csv(fp, usecols=cols)
        x = frame["X"].to_numpy(float)
        y = frame["Y"].to_numpy(float)
        z = frame["Z"].to_numpy(float)
        tvt = frame["TVT"].to_numpy(float)
        ti = frame["TVT_input"].to_numpy(float)
        s = last_known_index(ti)
        ndz = -np.diff(z)[s:]
        n = len(ndz)
        if n <= 0:
            continue
        u = np.cumsum(ndz)
        r0 = tvt[s + 1 :] - tvt[s]
        segid, mid, proj, az = segment_geometry(x, y, s, n, params)
        c_raw = fit_coeffs(r0, u, n, params, rho=0.0)
        c_sm = fit_coeffs(r0, u, n, params, rho=params.smooth_rho)
        wells.append(
            WellData(
                wid=wid,
                wi=len(wells),
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
                gr=frame["GR"].to_numpy(float),
                typewell_path=fp.with_name(f"{wid}__typewell.csv"),
                tvt=tvt,
                r0=r0,
                anc=frame["ANCC"].to_numpy(float),
                c_raw=c_raw,
                c_sm=c_sm,
            )
        )
    return wells


def load_test_wells(
    test_dir: Path,
    params: K16Params,
    max_wells: int | None = None,
) -> list[WellData]:
    wells: list[WellData] = []
    for fp in sorted(Path(test_dir).glob("*__horizontal_well.csv")):
        if max_wells is not None and len(wells) >= max_wells:
            break
        wid = fp.name.split("__")[0]
        cols = ["X", "Y", "Z", "MD", "TVT_input", "GR"]
        frame = pd.read_csv(fp, usecols=cols)
        x = frame["X"].to_numpy(float)
        y = frame["Y"].to_numpy(float)
        z = frame["Z"].to_numpy(float)
        ti = frame["TVT_input"].to_numpy(float)
        s = last_known_index(ti)
        ndz = -np.diff(z)[s:]
        n = len(ndz)
        if n <= 0:
            continue
        segid, mid, proj, az = segment_geometry(x, y, s, n, params)
        wells.append(
            WellData(
                wid=wid,
                wi=-1,
                s=s,
                n=n,
                ndz=ndz,
                anchor=float(ti[s]),
                ti=ti,
                segid=segid,
                mid=mid,
                proj=proj,
                az=az,
                x=x,
                y=y,
                z=z,
                gr=frame["GR"].to_numpy(float),
                typewell_path=fp.with_name(f"{wid}__typewell.csv"),
            )
        )
    return wells


def build_fields(wells: list[WellData], params: K16Params) -> FieldPack:
    def pack(key: str) -> np.ndarray:
        rows = []
        for well in wells:
            coeffs = getattr(well, key)
            if coeffs is None:
                continue
            for j in range(params.k_segments):
                if abs(well.proj[j]) > params.field_min_proj:
                    rows.append(
                        (
                            well.mid[j, 0],
                            well.mid[j, 1],
                            coeffs[j] / well.proj[j],
                            well.wi,
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
    return FieldPack(
        f_raw=pack("c_raw"),
        f_sm=pack("c_sm"),
        surface_points=surface_points,
        global_theta=global_theta,
    )


def _safe_nearest_indices(d2: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
    if len(cand) == 0:
        return cand
    kk = min(max(int(k), 1), len(cand))
    return cand[np.argpartition(d2[cand], kk - 1)[:kk]]


def local_linear(
    field: np.ndarray,
    own_wi: int,
    mid: np.ndarray,
    params: K16Params,
    min_dist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    keep = field[:, 3] != own_wi
    fx, fy, fd = field[keep, 0], field[keep, 1], field[keep, 2]
    dh = np.empty(len(mid), dtype=float)
    dd = np.empty(len(mid), dtype=float)
    if len(fd) == 0:
        dh.fill(0.0)
        dd.fill(float("inf"))
        return dh, dd
    for j in range(len(mid)):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        cand = np.where(d2 >= min_dist * min_dist)[0] if min_dist else np.arange(len(d2))
        sel = _safe_nearest_indices(d2, cand, params.local_linear_k)
        if len(sel) == 0:
            dh[j] = float(np.median(fd))
            dd[j] = float("inf")
            continue
        w = np.exp(np.maximum(-d2[sel] / (2.0 * params.local_linear_bandwidth**2), -700))
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
    return dh, dd


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
        sel = _safe_nearest_indices(d2, cand, 15)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_field, donor_dist = local_linear(fields.f_raw, well.wi, well.mid, params, min_dist)
    smooth_field, _ = local_linear(fields.f_sm, well.wi, well.mid, params, min_dist)
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
    return design, raw_field, smooth_field, donor_dist


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
            design, _, _, _ = well_design(well, fields, params, min_dist=regime)
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
    design, raw_field, smooth_field, donor_dist = well_design(well, fields, params)
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


def _prediction_rows(well: WellData, result: PredictionResult) -> pd.DataFrame:
    if well.tvt is None:
        raise ValueError("train prediction rows require true TVT")
    row_idx = well.suffix_row_idx
    true = well.tvt[row_idx]
    frame = pd.DataFrame(
        {
            "well_id": well.wid,
            "row_idx": row_idx,
            "suffix_offset": np.arange(well.n, dtype=int),
            "tvt_true": true,
            "tvt_pred": result.pred,
            "tvt_geop": result.geop,
            "gr_delta": result.delta,
            "abs_error": np.abs(true - result.pred),
            "error": result.pred - true,
        }
    )
    return frame


def _metric_row(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "label": label,
        "rows": int(len(frame)),
        "rmse": rmse(frame["tvt_true"].to_numpy(), frame["tvt_pred"].to_numpy()),
        "mae": mae(frame["tvt_true"].to_numpy(), frame["tvt_pred"].to_numpy()),
        "bias": float(np.mean(frame["tvt_pred"] - frame["tvt_true"]))
        if len(frame)
        else float("nan"),
        "within10": float(np.mean(frame["abs_error"] <= 10.0)) if len(frame) else float("nan"),
        "within25": float(np.mean(frame["abs_error"] <= 25.0)) if len(frame) else float("nan"),
    }


def distance_bucket_metrics(frame: pd.DataFrame, buckets: list[float]) -> pd.DataFrame:
    bins = list(buckets)
    if not bins or bins[0] > 0:
        bins = [0.0, *bins]
    if bins[-1] < 1e12:
        bins.append(1e18)
    labels = [f"{int(bins[i]):04d}_{int(bins[i + 1]):04d}" for i in range(len(bins) - 1)]
    bucket = pd.cut(frame["suffix_offset"], bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        part = frame[bucket == label]
        if len(part):
            row = _metric_row(str(label), part)
            row["bucket"] = str(label)
            rows.append(row)
    return pd.DataFrame(rows)


def run_train_audit(paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    params = params_from_config(config)
    output_prefix = str(get_nested(config, "artifacts.output_prefix", EXP_PREFIX))
    max_wells = get_nested(config, "validation.max_wells")
    max_wells = int(max_wells) if max_wells is not None else None
    seed = int(get_nested(config, "validation.seed", 42))
    n_folds = int(get_nested(config, "validation.n_folds", 5))
    strategy = str(get_nested(config, "validation.strategy", "group_safe_kfold"))
    fail_fast = bool(get_nested(config, "validation.fail_fast", True))
    if get_nested(config, "validation.debug_max_wells") and max_wells is None:
        max_wells = int(get_nested(config, "validation.debug_max_wells"))

    wells = load_train_wells(paths.train_data_dir, params, max_wells=max_wells)
    if not wells:
        raise FileNotFoundError(f"No usable train wells found under {paths.train_data_dir}")
    if "leave_one" in strategy.lower():
        n_folds = len(wells)
    n_folds = max(1, min(n_folds, len(wells)))
    fold_by_well = assign_group_folds(wells, n_folds, seed)

    print(f"Loaded {len(wells)} train wells. CV strategy={strategy}, folds={n_folds}")
    all_predictions: list[pd.DataFrame] = []
    well_summary_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    kappa_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for fold in range(n_folds):
        valid_wells = [well for well in wells if fold_by_well[well.wid] == fold]
        source_wells = [well for well in wells if fold_by_well[well.wid] != fold]
        if not valid_wells or not source_wells:
            continue
        print(f"Fold {fold + 1}/{n_folds}: source={len(source_wells)} valid={len(valid_wells)}")
        fields = build_fields(source_wells, params)
        kappa = fit_kappa(source_wells, fields, params)
        kappa_rows.extend(kappa_terms(kappa, params, fold=f"fold{fold}"))
        fold_frames = []
        for well in valid_wells:
            try:
                result = predict_well(well, fields, kappa, params)
                pred_frame = _prediction_rows(well, result)
                pred_frame["fold"] = fold
                all_predictions.append(pred_frame)
                fold_frames.append(pred_frame)
                row = dict(result.summary)
                row.update(_metric_row(well.wid, pred_frame))
                row["fold"] = fold
                well_summary_rows.append(row)
            except Exception as exc:
                err = {"fold": fold, "well_id": well.wid, "error": repr(exc)}
                errors.append(err)
                print(f"  ERROR {well.wid}: {exc}")
                if fail_fast:
                    raise
        if fold_frames:
            fold_frame = pd.concat(fold_frames, ignore_index=True)
            row = _metric_row(f"fold{fold}", fold_frame)
            row["fold"] = fold
            row["valid_wells"] = len(valid_wells)
            fold_metric_rows.append(row)

    if not all_predictions:
        raise RuntimeError("CV produced no predictions")
    oof = pd.concat(all_predictions, ignore_index=True)
    by_well = pd.DataFrame(well_summary_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    kappa_frame = pd.DataFrame(kappa_rows)
    buckets = list(get_nested(config, "validation.distance_buckets", [0, 50, 100, 250, 500, 1000]))
    bucket_metrics = distance_bucket_metrics(oof, [float(v) for v in buckets])
    overall = _metric_row("overall", oof)

    artifacts_dir = Path(paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    oof_sha = write_csv_gzip(oof, artifacts_dir / f"{output_prefix}_train_oof_predictions.csv.gz")
    by_well.to_csv(artifacts_dir / f"{output_prefix}_by_well_metrics.csv", index=False)
    fold_metrics.to_csv(artifacts_dir / f"{output_prefix}_fold_metrics.csv", index=False)
    bucket_metrics.to_csv(
        artifacts_dir / f"{output_prefix}_distance_bucket_metrics.csv", index=False
    )
    kappa_frame.to_csv(artifacts_dir / f"{output_prefix}_kappa_by_fold.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(artifacts_dir / f"{output_prefix}_cv_errors.csv", index=False)

    summary = {
        "experiment": get_nested(config, "experiment.name"),
        "status": "train_audit_completed",
        "route": get_nested(config, "experiment.route"),
        "source_notebook": get_nested(config, "lineage.source_notebook"),
        "validation_strategy": strategy,
        "train_wells": len(wells),
        "cv_folds": n_folds,
        "oof_rows": int(len(oof)),
        "overall": overall,
        "oof_decompressed_sha256": oof_sha,
        "kappa_mean": {
            row["term"]: float(kappa_frame[kappa_frame["term"] == row["term"]]["value"].mean())
            for row in kappa_rows[: params.kappa_dim]
        },
        "leakage_guard": {
            "target_well_excluded_from_donor_fields": True,
            "target_well_excluded_from_kappa_fit": True,
            "target_unknown_true_tvt_used_only_for_metrics": True,
        },
        "artifacts": {
            "oof": f"{output_prefix}_train_oof_predictions.csv.gz",
            "by_well": f"{output_prefix}_by_well_metrics.csv",
            "fold_metrics": f"{output_prefix}_fold_metrics.csv",
            "distance_bucket_metrics": f"{output_prefix}_distance_bucket_metrics.csv",
            "kappa_by_fold": f"{output_prefix}_kappa_by_fold.csv",
        },
        "errors": errors,
    }
    write_json(artifacts_dir / f"{output_prefix}_summary.json", summary)
    write_json(
        Path(paths.metrics_path),
        {
            "experiment": get_nested(config, "experiment.name"),
            "status": "train_audit_completed",
            "cv": overall["rmse"],
            "public_lb": None,
            "private_lb": None,
            "metric": get_nested(config, "validation.metric", "rmse"),
            "summary": summary,
        },
    )
    return summary


def run_inference(paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    params = params_from_config(config)
    output_prefix = str(get_nested(config, "artifacts.output_prefix", EXP_PREFIX))
    max_train_wells = get_nested(config, "inference.max_train_wells")
    max_test_wells = get_nested(config, "inference.max_test_wells")
    max_train_wells = int(max_train_wells) if max_train_wells is not None else None
    max_test_wells = int(max_test_wells) if max_test_wells is not None else None
    if max_train_wells is not None or max_test_wells is not None:
        print("Debug inference subset is active; output is not a valid competition submission.")

    train_wells = load_train_wells(paths.train_data_dir, params, max_wells=max_train_wells)
    test_wells = load_test_wells(paths.test_data_dir, params, max_wells=max_test_wells)
    if not train_wells:
        raise FileNotFoundError(f"No usable train wells found under {paths.train_data_dir}")
    if not test_wells:
        raise FileNotFoundError(f"No usable test wells found under {paths.test_data_dir}")
    print(f"Loaded train={len(train_wells)} test={len(test_wells)}")
    fields = build_fields(train_wells, params)
    kappa = fit_kappa(train_wells, fields, params)
    print("kappa:", np.round(kappa, 3))

    predictions: dict[str, PredictionResult] = {}
    summary_rows: list[dict[str, Any]] = []
    for idx, well in enumerate(test_wells, start=1):
        result = predict_well(well, fields, kappa, params)
        predictions[well.wid] = result
        row = dict(result.summary)
        row["order"] = idx
        summary_rows.append(row)
        print(
            f"{idx}/{len(test_wells)} {well.wid}: rows {well.s + 1}..{well.s + well.n} "
            f"gate {row['gate_segments']}/{params.k_segments} "
            f"delta_med {row['delta_abs_median']:.2f}"
        )

    sample = pd.read_csv(paths.sample_submission_path)
    parts = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("sample_submission id format must be '<well>_<row_idx>'")
    wids = parts[0].to_numpy()
    row_idx = parts[1].astype(int).to_numpy()
    out = np.empty(len(sample), dtype=float)
    missing: list[str] = []
    for i, (wid, ridx) in enumerate(zip(wids, row_idx, strict=False)):
        result = predictions.get(wid)
        if result is None:
            missing.append(wid)
            out[i] = np.nan
            continue
        start = next(well.s + 1 for well in test_wells if well.wid == wid)
        offset = int(ridx) - start
        if offset < 0 or offset >= len(result.pred):
            raise IndexError(f"sample row {wid}_{ridx} outside predicted suffix")
        out[i] = result.pred[offset]
    if missing:
        unique_missing = sorted(set(missing))
        raise RuntimeError(f"sample_submission contains wells not predicted: {unique_missing[:10]}")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite values in submission")
    submission = sample[["id"]].copy()
    submission["tvt"] = out
    submission_path = Path(paths.submission_path)
    submission.to_csv(submission_path, index=False)

    artifacts_dir = Path(paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(
        artifacts_dir / f"{output_prefix}_test_prediction_summary.csv", index=False
    )
    pd.DataFrame(kappa_terms(kappa, params, fold="full_train")).to_csv(
        artifacts_dir / f"{output_prefix}_kappa_full_train.csv",
        index=False,
    )
    submission_sha = dataframe_content_sha256(submission)
    summary = {
        "experiment": get_nested(config, "experiment.name"),
        "status": "inference_completed",
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "submission_rows": int(len(submission)),
        "submission_file": str(submission_path),
        "submission_sha256": submission_sha,
        "tvt_min": float(submission["tvt"].min()),
        "tvt_max": float(submission["tvt"].max()),
        "tvt_mean": float(submission["tvt"].mean()),
        "tvt_std": float(submission["tvt"].std()),
        "artifacts": {
            "prediction_summary": f"{output_prefix}_test_prediction_summary.csv",
            "kappa_full_train": f"{output_prefix}_kappa_full_train.csv",
        },
    }
    write_json(artifacts_dir / f"{output_prefix}_inference_summary.json", summary)
    write_json(
        Path(paths.metrics_path),
        {
            "experiment": get_nested(config, "experiment.name"),
            "status": "inference_completed",
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": get_nested(config, "validation.metric", "rmse"),
            "summary": summary,
        },
    )
    return summary
